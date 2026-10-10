#!/usr/bin/env python3
# Copyright (c) 2026 Eoin Jordan
# SPDX-License-Identifier: Apache-2.0
"""Linux brain of the UNO Q sample: read the MCU's JSON lines and act on them.

Runs on the UNO Q's Debian side (or on a PC watching the serial port). The
MCU prints one JSON object per inference:

  {"seq":12,"t_ms":6500,"temp_c":52.31,"p_anomaly":0.012,"anomaly":0,"sensor_ok":1,"infer_us":41}

This script prints a one-line status, appends a CSV log, and calls `on_alarm`
on the rising edge of `anomaly`. Replace `on_alarm` with whatever the Linux
side should do: MQTT, a notification, a database write, a camera snapshot.

Usage, on the board (the STM32 UART is /dev/ttyHS1; stop arduino-router first):
  pip install pyserial
  python3 anomaly_bridge.py --port /dev/ttyHS1 --log anomalies.csv

Usage, from a PC over adb (no pyserial needed, --port - reads stdin):
  adb exec-out cat /dev/ttyHS1 | python3 anomaly_bridge.py --port - --log anomalies.csv
"""

import argparse
import csv
import json
import sys
import time


def on_alarm(event: dict) -> None:
    print(f"*** ALARM at seq {event['seq']}: temp {event['temp_c']} C, p={event['p_anomaly']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True, help="serial device connected to the STM32 console, or - for stdin")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--log", default="anomaly_log.csv")
    args = p.parse_args()

    if args.port == "-":
        readline = sys.stdin.buffer.readline
    else:
        import serial  # pip install pyserial

        ser = serial.Serial(args.port, args.baud, timeout=5)
        readline = ser.readline
    last = 0
    with open(args.log, "a", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["host_time", "seq", "t_ms", "temp_c", "p_anomaly", "anomaly", "sensor_ok", "infer_us"])
        while True:
            raw = readline()
            if not raw:
                if args.port == "-":
                    break
                continue
            line = raw.decode(errors="replace").strip()
            if not line.startswith("{"):
                if line:
                    print(f"[mcu] {line}")
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            w.writerow([time.time(), ev["seq"], ev["t_ms"], ev["temp_c"], ev["p_anomaly"],
                        ev["anomaly"], ev["sensor_ok"], ev["infer_us"]])
            f.flush()
            state = "ANOMALY" if ev["anomaly"] else "normal "
            print(f"{state}  temp={ev['temp_c']:6.2f} C  p={ev['p_anomaly']:.3f}  {ev['infer_us']} us")
            if ev["anomaly"] and not last:
                on_alarm(ev)
            last = ev["anomaly"]


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
