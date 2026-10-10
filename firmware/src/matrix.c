/* Copyright (c) 2026 Eoin Jordan
 * SPDX-License-Identifier: Apache-2.0
 *
 * Rolling anomaly-probability chart on the UNO Q's 13x8 charlieplexed LED
 * matrix, through Zephyr's display API. The driver exposes a framebuffer of
 * one byte per pixel, row-major, 0 .. (2^grayscale-bits - 1).
 *
 *   x = 0 is the oldest inference, x = 12 the newest.
 *   column height  = p(anomaly) scaled to the 8 rows (at least one pixel so
 *                    a healthy loop still shows a heartbeat)
 *   dim column     = normal
 *   bright column  = the MCU flagged an anomaly
 *   dotted column  = the sensor read out of range (open or shorted input)
 */

#include "matrix.h"

#include <math.h>
#include <string.h>

#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/display.h>
#include <zephyr/kernel.h>

#define MATRIX_NODE DT_CHOSEN(zephyr_display)
#define LEVELS      (1u << DT_PROP(MATRIX_NODE, grayscale_bits))
#define BRIGHT      (uint8_t)(LEVELS - 1)
#define DIM         (uint8_t)(LEVELS / 4 > 0 ? LEVELS / 4 : 1)

static const struct device *const matrix = DEVICE_DT_GET(MATRIX_NODE);
static uint8_t *fb;
static uint16_t width, height;
static bool showing_selftest;

static void set(uint16_t x, uint16_t y, uint8_t v)
{
	if (x < width && y < height) {
		fb[y * width + x] = v;
	}
}

int matrix_init(void)
{
	if (!device_is_ready(matrix)) {
		return -ENODEV;
	}

	struct display_capabilities caps;
	display_get_capabilities(matrix, &caps);
	width = caps.x_resolution;
	height = caps.y_resolution;

	fb = display_get_framebuffer(matrix);
	if (fb == NULL || width == 0 || height == 0) {
		return -ENOTSUP;
	}
	memset(fb, 0, (size_t)width * height);
	display_blanking_off(matrix);
	return 0;
}

void matrix_show_selftest(bool ok)
{
	if (fb == NULL) {
		return;
	}
	memset(fb, 0, (size_t)width * height);
	if (ok) {
		/* Tick: short stroke down from (2,4), long stroke up to (11,1). */
		for (int i = 0; i < 3; i++) {
			set(2 + i, 4 + i, BRIGHT);
		}
		for (int i = 0; i < 7; i++) {
			set(5 + i, 7 - i, BRIGHT);
		}
	} else {
		/* Cross over the middle 8 columns. */
		for (int i = 0; i < 8; i++) {
			set(2 + i, i, BRIGHT);
			set(9 - i, i, BRIGHT);
		}
	}
	showing_selftest = true;
}

void matrix_push(float p_anomaly, bool anomaly, bool sensor_ok)
{
	if (fb == NULL) {
		return;
	}
	if (showing_selftest) {
		memset(fb, 0, (size_t)width * height);
		showing_selftest = false;
	} else {
		for (uint16_t y = 0; y < height; y++) {
			memmove(&fb[y * width], &fb[y * width + 1], width - 1);
		}
	}

	if (p_anomaly < 0.f) {
		p_anomaly = 0.f;
	} else if (p_anomaly > 1.f) {
		p_anomaly = 1.f;
	}
	int bar = (int)lroundf(p_anomaly * height);
	if (bar < 1) {
		bar = 1;
	}

	const uint16_t x = width - 1;
	const uint8_t level = anomaly ? BRIGHT : DIM;
	for (uint16_t y = 0; y < height; y++) {
		bool lit = (int)y >= (int)height - bar;
		if (!sensor_ok) {
			lit = (y & 1u) == 0u;
		}
		set(x, y, lit ? level : 0);
	}
}
