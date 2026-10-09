/* Copyright (c) 2026 Eoin Jordan
 * SPDX-License-Identifier: Apache-2.0
 *
 * Analog temperature sensor on A0 (TMP36 by default), or a software
 * simulation of the same signal when CONFIG_APP_SIM_SENSOR=y.
 */

#include "sensor.h"

#include <math.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#ifdef CONFIG_APP_SIM_SENSOR

static float sim_t;          /* seconds */
static uint32_t lcg = 12345; /* tiny PRNG for sensor noise */

static float noise(void)
{
	lcg = lcg * 1664525u + 1013904223u;
	return ((float)(lcg >> 8) / 16777216.0f - 0.5f) * 0.6f; /* +-0.3 C */
}

int sensor_init(void)
{
	printk("sensor: simulated heater loop (CONFIG_APP_SIM_SENSOR=y); spike every 30 s\n");
	return 0;
}

bool sensor_read_celsius(float *celsius)
{
	float c = 50.0f + 4.0f * sinf(2.0f * 3.14159265f * sim_t / 60.0f) + noise();
	/* three-sample spike once every 30 s */
	float phase = fmodf(sim_t, 30.0f);

	if (phase >= 25.0f && phase < 25.3f) {
		c += 18.0f;
	}
	sim_t += CONFIG_APP_SAMPLE_PERIOD_MS / 1000.0f;
	*celsius = c;
	return true;
}

#else /* real ADC */

#include <zephyr/drivers/adc.h>

static const struct adc_dt_spec adc = ADC_DT_SPEC_GET_BY_IDX(DT_PATH(zephyr_user), 0);
static uint16_t sample;
static struct adc_sequence seq = {
	.buffer = &sample,
	.buffer_size = sizeof(sample),
};

int sensor_init(void)
{
	int ret;

	if (!adc_is_ready_dt(&adc)) {
		printk("sensor: ADC %s not ready\n", adc.dev->name);
		return -ENODEV;
	}
	ret = adc_channel_setup_dt(&adc);
	if (ret < 0) {
		return ret;
	}
	ret = adc_sequence_init_dt(&adc, &seq);
	if (ret < 0) {
		return ret;
	}
	printk("sensor: %s channel %u, %d.%d mV/C, %d mV at 0 C\n", adc.dev->name, adc.channel_id,
	       CONFIG_APP_MV_PER_DEGREE / 10, CONFIG_APP_MV_PER_DEGREE % 10, CONFIG_APP_MV_OFFSET);
	return 0;
}

bool sensor_read_celsius(float *celsius)
{
	int ret = adc_read_dt(&adc, &seq);

	if (ret < 0) {
		printk("sensor: adc_read %d\n", ret);
		*celsius = 0.0f;
		return false;
	}
	int32_t mv = sample;

	ret = adc_raw_to_millivolts_dt(&adc, &mv);
	if (ret < 0) {
		*celsius = 0.0f;
		return false;
	}
	*celsius = (float)(mv - CONFIG_APP_MV_OFFSET) * 10.0f / (float)CONFIG_APP_MV_PER_DEGREE;
	return mv > 20 && mv < 3280;
}

#endif
