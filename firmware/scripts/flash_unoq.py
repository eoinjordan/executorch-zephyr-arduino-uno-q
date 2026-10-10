#!/usr/bin/env python3
# Copyright (c) 2026 Eoin Jordan
# SPDX-License-Identifier: Apache-2.0
"""Flash a Zephyr ELF onto the UNO Q's STM32U585 through the Linux side.

The QRB2210 runs OpenOCD (`arduino-debug`) and bit-bangs SWD to the STM32.
`west flash` is not wired to it, so this script does what the board docs
describe by hand:

  adb forward tcp:3333 tcp:3333
  adb shell arduino-debug                      # gdb server on :3333
  arm-zephyr-eabi-gdb -batch ... load ...      # program flash, reset, run

Usage:
  python firmware/scripts/flash_unoq.py                # build/zephyr/zephyr.elf
  python firmware/scripts/flash_unoq.py --elf path/to/zephyr.elf
"""

import argparse
import glob
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time


def find_adb(explicit):
    cands = [explicit, shutil.which("adb")]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        cands.append(os.path.join(local, "Programs", "platform-tools", "adb.exe"))
        cands.append(os.path.join(local, "Android", "Sdk", "platform-tools", "adb.exe"))
    cands.append(os.path.expanduser("~/Android/Sdk/platform-tools/adb"))
    for c in cands:
        if c and os.path.isfile(c):
            return c
    sys.exit("adb not found: install Android platform-tools or pass --adb")


def find_gdb(explicit):
    cands = [explicit, shutil.which("arm-zephyr-eabi-gdb")]
    roots = [os.environ.get("ZEPHYR_SDK_INSTALL_DIR")] + sorted(
        glob.glob(os.path.expanduser("~/zephyr-sdk-*")), reverse=True
    ) + sorted(glob.glob("/opt/zephyr-sdk-*"), reverse=True)
    for r in roots:
        if not r:
            continue
        for sub in ("gnu/arm-zephyr-eabi/bin", "arm-zephyr-eabi/bin"):
            for name in ("arm-zephyr-eabi-gdb.exe", "arm-zephyr-eabi-gdb"):
                cands.append(os.path.join(r, sub, name))
    for c in cands:
        if c and os.path.isfile(c):
            return c
    sys.exit("arm-zephyr-eabi-gdb not found: set ZEPHYR_SDK_INSTALL_DIR or pass --gdb")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--elf", default="build/zephyr/zephyr.elf")
    p.add_argument("--adb")
    p.add_argument("--gdb")
    p.add_argument("--port", type=int, default=3333)
    args = p.parse_args()

    elf = pathlib.Path(args.elf)
    if not elf.is_file():
        sys.exit(f"{elf} not found; build first")
    adb, gdb = find_adb(args.adb), find_gdb(args.gdb)

    subprocess.run([adb, "forward", f"tcp:{args.port}", f"tcp:{args.port}"], check=True)

    # Start OpenOCD on the board and wait until it listens (or tell us one already does).
    srv = subprocess.Popen([adb, "shell", "arduino-debug"], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
    state = {"listening": False, "busy": False}

    def pump():
        for line in srv.stdout:
            if "Listening on port" in line and str(args.port) in line:
                state["listening"] = True
            if "Address already in use" in line or "already in use" in line:
                state["busy"] = True

    threading.Thread(target=pump, daemon=True).start()
    deadline = time.time() + 20
    while time.time() < deadline and not (state["listening"] or state["busy"]):
        if srv.poll() is not None:
            break
        time.sleep(0.2)
    if state["busy"] or (not state["listening"] and srv.poll() is not None):
        print("note: an OpenOCD server is already running on the board, using it")
    elif not state["listening"]:
        print("warning: OpenOCD did not report listening yet, trying anyway")

    cmds = [
        "set confirm off",
        f"target extended-remote localhost:{args.port}",
        "monitor reset halt",
        "load",
        "monitor reset run",
        "detach",
    ]
    argv = [gdb, "-batch", "-nx"]
    for c in cmds:
        argv += ["-ex", c]
    argv.append(str(elf))
    rc = subprocess.run(argv).returncode

    if state["listening"]:
        srv.terminate()
        subprocess.run([adb, "shell", "pkill", "-x", "openocd"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    sys.exit(rc)


if __name__ == "__main__":
    main()
