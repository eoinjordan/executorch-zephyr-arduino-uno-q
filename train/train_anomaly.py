#!/usr/bin/env python3
# Copyright (c) 2026 Eoin Jordan
# SPDX-License-Identifier: Apache-2.0
"""Train AnomalyNet on synthetic temperature windows and export it for Zephyr.

The data generator mimics a heater loop watched by an analog temperature
transducer: a slow sinusoidal cycle around a setpoint with sensor noise is
"normal"; spikes, sensor dropouts, thermal runaway and noise bursts are
"anomaly". Swap `make_dataset` for real recordings when you have them.

Outputs (relative to the repo root by default):
  firmware/model/anomaly_net.pte   ExecuTorch program, portable fp32 ops
  firmware/src/test_input.h        one normal + one anomaly window with reference logits
  train/out/anomaly_net.pt         PyTorch state_dict
"""

import argparse
import math
import pathlib
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import CLASSES, INPUT_SHAPE, SAMPLE_HZ, WINDOW, AnomalyNet, normalise  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]


def normal_window(g: torch.Generator) -> torch.Tensor:
    base = 50.0 + (torch.rand(1, generator=g).item() - 0.5) * 6.0
    phase = torch.rand(1, generator=g).item() * 2 * math.pi
    t = torch.arange(WINDOW, dtype=torch.float32) / SAMPLE_HZ
    cycle = 4.0 * torch.sin(2 * math.pi * t / 60.0 + phase)  # 60 s heater cycle
    noise = torch.randn(WINDOW, generator=g) * 0.3
    return base + cycle + noise


def anomaly_window(g: torch.Generator) -> torch.Tensor:
    x = normal_window(g)
    kind = int(torch.randint(0, 4, (1,), generator=g))
    sign = 1.0 if torch.rand(1, generator=g).item() < 0.5 else -1.0
    if kind == 0:  # spike: a few samples jump by 8..25 degrees
        i = int(torch.randint(0, WINDOW - 1, (1,), generator=g))
        w = int(torch.randint(1, 6, (1,), generator=g))
        x[i : i + w] += sign * (8.0 + 17.0 * torch.rand(1, generator=g).item())
    elif kind == 1:  # dropout: open/short transducer reads 0 V
        i = int(torch.randint(0, WINDOW - 4, (1,), generator=g))
        x[i:] = 0.0
    elif kind == 2:  # runaway: ramp of 0.4..1.5 degrees per sample
        i = int(torch.randint(0, WINDOW // 2, (1,), generator=g))
        slope = sign * (0.4 + 1.1 * torch.rand(1, generator=g).item())
        x[i:] += slope * torch.arange(WINDOW - i, dtype=torch.float32)
    else:  # noise burst: std 2..6 degrees
        x += torch.randn(WINDOW, generator=g) * (2.0 + 4.0 * torch.rand(1, generator=g).item())
    return x


def make_dataset(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for i in range(n):
        if i % 2 == 0:
            xs.append(normal_window(g))
            ys.append(0)
        else:
            xs.append(anomaly_window(g))
            ys.append(1)
    return normalise(torch.stack(xs)), torch.tensor(ys)


def train(model, x, y, epochs: int, lr: float, batch: int):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(y)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        loss_sum, correct = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            out = model(x[idx])
            loss = F.cross_entropy(out, y[idx])
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(idx)
            correct += (out.argmax(1) == y[idx]).sum().item()
        print(f"epoch {epoch + 1}/{epochs}: loss={loss_sum / n:.4f} acc={correct / n:.4f}")


@torch.no_grad()
def evaluate(model, x, y) -> float:
    model.eval()
    return (model(x).argmax(1) == y).float().mean().item()


def export_pte(model, out: pathlib.Path) -> None:
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    model.eval()
    exported = torch.export.export(model, (torch.zeros(*INPUT_SHAPE),))
    program = to_edge_transform_and_lower(
        exported, compile_config=EdgeCompileConfig(_check_ir_validity=False)
    ).to_executorch()
    ops = sorted(
        {str(n.target) for n in program.exported_program().graph.nodes if n.op == "call_function"}
    )
    print("operators in program:")
    for op in ops:
        print(f"  {op}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(program.buffer)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


def c_array(name: str, values, per_line: int = 8) -> list[str]:
    lines = [f"static const float {name}[{len(values)}] = {{"]
    for i in range(0, len(values), per_line):
        lines.append("    " + ", ".join(f"{v:.6f}f" for v in values[i : i + per_line]) + ",")
    lines.append("};")
    return lines


@torch.no_grad()
def write_test_header(model, out: pathlib.Path, seed: int) -> None:
    """Embed one normal and one anomaly window plus PyTorch logits for both."""
    model.eval()
    g = torch.Generator().manual_seed(seed + 1)
    normal = normal_window(g)
    anomaly = normal.clone()
    anomaly[12:15] += 18.0  # a clear spike
    xn, xa = normalise(normal), normalise(anomaly)
    ln, la = model(xn[None])[0], model(xa[None])[0]

    lines = [
        "/* Generated by train/train_anomaly.py. Do not edit. */",
        "#pragma once",
        "",
        f"#define TEST_WINDOW {WINDOW}",
        f"#define TEST_NUM_CLASSES {len(CLASSES)}",
        "",
        "/* Windows are already normalised with (t - MEAN) / SCALE. */",
    ]
    lines += c_array("kTestNormal", xn.tolist())
    lines += c_array("kTestNormalLogits", ln.tolist())
    lines += c_array("kTestAnomaly", xa.tolist())
    lines += c_array("kTestAnomalyLogits", la.tolist())
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {out}: normal argmax={int(ln.argmax())}, anomaly argmax={int(la.argmax())}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--samples", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pte", type=pathlib.Path, default=REPO / "firmware" / "model" / "anomaly_net.pte")
    p.add_argument("--header", type=pathlib.Path, default=REPO / "firmware" / "src" / "test_input.h")
    p.add_argument("--state-dict", type=pathlib.Path, default=REPO / "train" / "out" / "anomaly_net.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    x_train, y_train = make_dataset(args.samples, args.seed)
    x_test, y_test = make_dataset(args.samples // 5, args.seed + 100)

    model = AnomalyNet()
    n_params = sum(t.numel() for t in model.parameters())
    print(f"AnomalyNet: {n_params} parameters ({n_params * 4 / 1024:.1f} KiB fp32)")

    train(model, x_train, y_train, args.epochs, args.lr, args.batch_size)
    print(f"test accuracy: {evaluate(model, x_test, y_test):.4f}")

    args.state_dict.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.state_dict)
    export_pte(model, args.pte)
    write_test_header(model, args.header, args.seed)


if __name__ == "__main__":
    main()
