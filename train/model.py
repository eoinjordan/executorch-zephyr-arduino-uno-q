# Copyright (c) 2026 Eoin Jordan
# SPDX-License-Identifier: Apache-2.0
"""AnomalyNet: a tiny MLP that classifies a window of temperature samples.

Input: 32 consecutive readings (3.2 s at 10 Hz), normalised as (t - 50) / 20.
Output: 2 logits, [normal, anomaly]. About 850 parameters, so the fp32
portable ExecuTorch kernels run it in well under a millisecond on a
Cortex-M7 and the whole program is a few KB of flash.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

WINDOW = 32
SAMPLE_HZ = 10
MEAN = 50.0   # degrees C
SCALE = 20.0  # degrees C
CLASSES = ("normal", "anomaly")
INPUT_SHAPE = (1, WINDOW)


class AnomalyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(WINDOW, 24)
        self.fc2 = nn.Linear(24, len(CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


def normalise(celsius: torch.Tensor) -> torch.Tensor:
    return (celsius - MEAN) / SCALE
