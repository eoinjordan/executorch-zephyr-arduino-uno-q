/* Copyright (c) 2026 Eoin Jordan
 * SPDX-License-Identifier: Apache-2.0
 *
 * Inference results on the UNO Q's 13x8 LED matrix. No-ops when the matrix
 * is compiled out (CONFIG_APP_LED_MATRIX=n).
 */
#pragma once

#include <errno.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef CONFIG_APP_LED_MATRIX

/* Returns 0 when the matrix is ready, -ENODEV/-ENOTSUP otherwise. */
int matrix_init(void);

/* Tick for pass, cross for fail. Stays until the first matrix_push(). */
void matrix_show_selftest(bool ok);

/* Scroll the chart left and add one column: height = p_anomaly, bright when
 * the MCU flagged an anomaly, dotted when the sensor read out of range. */
void matrix_push(float p_anomaly, bool anomaly, bool sensor_ok);

#else

static inline int matrix_init(void)
{
	return -ENODEV;
}
static inline void matrix_show_selftest(bool ok)
{
	(void)ok;
}
static inline void matrix_push(float p_anomaly, bool anomaly, bool sensor_ok)
{
	(void)p_anomaly;
	(void)anomaly;
	(void)sensor_ok;
}

#endif /* CONFIG_APP_LED_MATRIX */

#ifdef __cplusplus
}
#endif
