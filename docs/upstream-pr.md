# Upstreaming to pytorch/executorch `zephyr/samples`

Proposed sample: `zephyr/samples/anomaly-uno-q/` (this repo's `firmware/` plus `linux/` and the
training script under `models/`).

What it adds over `hello-executorch`:

- A model with an application story (sensor window in, probability out) on the UNO Q, which
  `hello-executorch` already lists as a tested board.
- An MCU-to-MPU hand-over pattern (JSON over the internal UART) with a reference consumer.

Checklist:

- [ ] Header convention: Meta copyright + BSD text in `src/`, Apache-2.0 SPDX for CMake/Kconfig.
- [ ] `lintrunner` clean.
- [ ] Console log from hardware with `SELF-TEST PASS` and `anomaly_bridge.py` output.
- [ ] Sign the Meta CLA.
