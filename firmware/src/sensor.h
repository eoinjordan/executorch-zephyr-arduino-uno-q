/* Copyright (c) 2026 Eoin Jordan
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

int sensor_init(void);

/* One temperature reading in degrees C. Returns false if the input is out of
 * range (open or shorted). */
bool sensor_read_celsius(float *celsius);

#ifdef __cplusplus
}
#endif
