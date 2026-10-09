#!/usr/bin/env python3
# Copyright (c) 2026 Eoin Jordan
# SPDX-License-Identifier: Apache-2.0
"""Run the exported .pte on the host and check it against the embedded test windows."""

import argparse
import pathlib
import re

import torch
from executorch.runtime import Runtime

REPO = pathlib.Path(__file__).resolve().parents[1]


def read_array(text: str, name: str) -> torch.Tensor:
    body = text.split(f"{name}[")[1].split("};")[0]
    return torch.tensor([float(v) for v in re.findall(r"(-?\d+\.\d+)f", body)])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pte", type=pathlib.Path, default=REPO / "firmware" / "model" / "anomaly_net.pte")
    p.add_argument("--header", type=pathlib.Path, default=REPO / "firmware" / "src" / "test_input.h")
    args = p.parse_args()

    text = args.header.read_text()
    method = Runtime.get().load_program(args.pte).load_method("forward")
    ok = True
    for name, expect_cls in (("kTestNormal", 0), ("kTestAnomaly", 1)):
        x = read_array(text, name)
        ref = read_array(text, name + "Logits")
        logits = method.execute([x[None]])[0][0]
        diff = (logits - ref).abs().max().item()
        pred = int(logits.argmax())
        print(f"{name}: logits={[f'{v:.4f}' for v in logits.tolist()]} argmax={pred} max|pte-pytorch|={diff:.2e}")
        ok &= pred == expect_cls and diff < 1e-3
    print("OK" if ok else "MISMATCH")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
