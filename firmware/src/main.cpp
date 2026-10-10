/* Copyright (c) 2026 Eoin Jordan
 * SPDX-License-Identifier: Apache-2.0
 *
 * MCU brain of the Arduino UNO Q dual-brain sample.
 *
 *   A0 (or simulated) --10 Hz--> ring buffer of 32 samples
 *        --every 500 ms--> ExecuTorch AnomalyNet --> one JSON line on the UART
 *                                               --> RGB LED 3 green / red
 *
 * The Linux brain (QRB2210) reads the JSON lines; see linux/anomaly_bridge.py.
 */

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

#include <zephyr/drivers/gpio.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/version.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "model_pte.h"
#include "sensor.h"
#include "matrix.h"
#include "test_input.h"

using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::BufferDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::TensorInfo;

static constexpr float kMean = 50.0f;
static constexpr float kScale = 20.0f;
static constexpr size_t kWindow = TEST_WINDOW;
static constexpr size_t kNumClasses = TEST_NUM_CLASSES;
static constexpr float kThreshold = CONFIG_APP_ANOMALY_THRESHOLD_PERCENT / 100.0f;

#ifndef APP_METHOD_ALLOCATOR_POOL_SIZE
#define APP_METHOD_ALLOCATOR_POOL_SIZE (64 * 1024)
#endif
#ifndef APP_TEMP_ALLOCATOR_POOL_SIZE
#define APP_TEMP_ALLOCATOR_POOL_SIZE (8 * 1024)
#endif

alignas(16) static uint8_t method_pool[APP_METHOD_ALLOCATOR_POOL_SIZE];
alignas(16) static uint8_t temp_pool[APP_TEMP_ALLOCATOR_POOL_SIZE];

/* RGB LED 3 on the UNO Q (active low in devicetree, handled by the GPIO flags) */
static const struct gpio_dt_spec led_green = GPIO_DT_SPEC_GET(DT_NODELABEL(led3_green), gpios);
static const struct gpio_dt_spec led_red = GPIO_DT_SPEC_GET(DT_NODELABEL(led3_red), gpios);

namespace {

struct Runner {
	BufferDataLoader loader{model_pte, model_pte_len};
	MemoryAllocator method_allocator{APP_METHOD_ALLOCATOR_POOL_SIZE, method_pool};
	MemoryAllocator temp_allocator{APP_TEMP_ALLOCATOR_POOL_SIZE, temp_pool};
	std::vector<Span<uint8_t>> planned_spans;
	Program *program = nullptr;
	Method *method = nullptr;
	float *input = nullptr;
	size_t input_numel = 0;
	Tensor *input_tensor = nullptr;

	Error init();
	Error run(const float *window, float *logits, uint32_t *us);
};

Error Runner::init()
{
	Result<Program> program_res = Program::load(&loader);
	if (!program_res.ok()) {
		ET_LOG(Error, "Program::load failed: 0x%x", (unsigned)program_res.error());
		return program_res.error();
	}
	program = new Program(std::move(program_res.get()));

	const char *method_name = program->get_method_name(0).get();
	Result<MethodMeta> meta = program->method_meta(method_name);
	if (!meta.ok()) {
		return meta.error();
	}

	for (size_t id = 0; id < meta->num_memory_planned_buffers(); ++id) {
		size_t size = static_cast<size_t>(meta->memory_planned_buffer_size(id).get());
		uint8_t *buf = static_cast<uint8_t *>(method_allocator.allocate(size, 16));
		if (buf == nullptr) {
			ET_LOG(Error, "planned buffer %zu (%zu B) does not fit the method pool", id, size);
			return Error::MemoryAllocationFailed;
		}
		ET_LOG(Info, "planned buffer %zu: %zu bytes", id, size);
		planned_spans.push_back({buf, size});
	}

	static HierarchicalAllocator planned_memory({planned_spans.data(), planned_spans.size()});
	static MemoryManager memory_manager(&method_allocator, &planned_memory, &temp_allocator);

	Result<Method> method_res = program->load_method(method_name, &memory_manager);
	if (!method_res.ok()) {
		ET_LOG(Error, "load_method(%s) failed: 0x%x", method_name, (unsigned)method_res.error());
		return method_res.error();
	}
	method = new Method(std::move(method_res.get()));

	Result<TensorInfo> in_meta = meta->input_tensor_meta(0);
	if (!in_meta.ok()) {
		return in_meta.error();
	}
	input_numel = in_meta->nbytes() / sizeof(float);
	input = static_cast<float *>(method_allocator.allocate(in_meta->nbytes(), 16));
	if (input == nullptr) {
		return Error::MemoryAllocationFailed;
	}
	static TensorImpl impl(
		in_meta->scalar_type(), in_meta->sizes().size(),
		const_cast<TensorImpl::SizesType *>(in_meta->sizes().data()), input,
		const_cast<TensorImpl::DimOrderType *>(in_meta->dim_order().data()));
	static Tensor tensor(&impl);
	input_tensor = &tensor;

	ET_LOG(Info, "method '%s' loaded; input numel %zu; pool used %zu / %zu bytes", method_name,
	       input_numel, method_allocator.used_size(), method_allocator.size());
	return Error::Ok;
}

Error Runner::run(const float *window, float *logits, uint32_t *us)
{
	std::memcpy(input, window, input_numel * sizeof(float));
	Error err = method->set_input(EValue(*input_tensor), 0);
	if (err != Error::Ok) {
		return err;
	}
	uint32_t t0 = k_cycle_get_32();
	err = method->execute();
	uint32_t t1 = k_cycle_get_32();
	if (err != Error::Ok) {
		ET_LOG(Error, "execute failed: 0x%x", (unsigned)err);
		return err;
	}
	*us = k_cyc_to_us_floor32(t1 - t0);

	const EValue &out = method->get_output(0);
	if (!out.isTensor()) {
		return Error::InvalidState;
	}
	const float *data = out.toTensor().const_data_ptr<float>();
	for (size_t i = 0; i < kNumClasses; ++i) {
		logits[i] = data[i];
	}
	return Error::Ok;
}

float anomaly_probability(const float *logits)
{
	float m = fmaxf(logits[0], logits[1]);
	float e0 = expf(logits[0] - m), e1 = expf(logits[1] - m);
	return e1 / (e0 + e1);
}

bool self_test(Runner &runner, const char *name, const float *window, const float *expected,
	       size_t expect_cls)
{
	float logits[kNumClasses];
	uint32_t us = 0;
	if (runner.run(window, logits, &us) != Error::Ok) {
		printk("  %s: inference failed\n", name);
		return false;
	}
	float diff = 0.f;
	size_t pred = logits[1] > logits[0] ? 1 : 0;
	for (size_t i = 0; i < kNumClasses; ++i) {
		diff = fmaxf(diff, fabsf(logits[i] - expected[i]));
	}
	printk("  %s: logits [%.4f %.4f] p(anomaly)=%.3f pred=%zu expected=%zu max|m33-pytorch|=%.1e (%u us)\n",
	       name, (double)logits[0], (double)logits[1], (double)anomaly_probability(logits), pred,
	       expect_cls, (double)diff, us);
	return pred == expect_cls && diff < 1e-3f;
}

} // namespace

int main(void)
{
	executorch::runtime::runtime_init();

	printk("\nExecuTorch on %s (Zephyr %s)\n", CONFIG_BOARD_TARGET, KERNEL_VERSION_STRING);
	printk("model: %u bytes in flash, method pool %u B, temp pool %u B\n",
	       (unsigned)model_pte_len, (unsigned)APP_METHOD_ALLOCATOR_POOL_SIZE,
	       (unsigned)APP_TEMP_ALLOCATOR_POOL_SIZE);

	if (gpio_is_ready_dt(&led_green) && gpio_is_ready_dt(&led_red)) {
		gpio_pin_configure_dt(&led_green, GPIO_OUTPUT_INACTIVE);
		gpio_pin_configure_dt(&led_red, GPIO_OUTPUT_INACTIVE);
	}

	if (matrix_init() == 0) {
		printk("matrix: 13x8 LED matrix on PF0..PF10, rolling p(anomaly) chart\n");
	}

	static Runner runner;
	if (runner.init() != Error::Ok) {
		printk("FAIL: runtime init\n");
		return 1;
	}

	printk("\nself-test:\n");
	bool ok = self_test(runner, "normal ", kTestNormal, kTestNormalLogits, 0);
	ok &= self_test(runner, "anomaly", kTestAnomaly, kTestAnomalyLogits, 1);
	printk(ok ? "SELF-TEST PASS\n" : "SELF-TEST FAIL\n");
	matrix_show_selftest(ok);

	if (sensor_init() != 0) {
		printk("FAIL: sensor init\n");
		return 1;
	}

	static float ring[kWindow];
	size_t head = 0, filled = 0;
	float logits[kNumClasses];
	uint32_t us = 0, seq = 0;
	int64_t next_infer = k_uptime_get() + CONFIG_APP_INFER_PERIOD_MS;

	for (;;) {
		float c;
		bool in_range = sensor_read_celsius(&c);
		ring[head] = (c - kMean) / kScale;
		head = (head + 1) % kWindow;
		if (filled < kWindow) {
			filled++;
		}

		if (filled == kWindow && k_uptime_get() >= next_infer) {
			next_infer += CONFIG_APP_INFER_PERIOD_MS;
			float window[kWindow];
			for (size_t i = 0; i < kWindow; ++i) {
				window[i] = ring[(head + i) % kWindow];
			}
			if (runner.run(window, logits, &us) == Error::Ok) {
				float p = anomaly_probability(logits);
				bool anomaly = p >= kThreshold || !in_range;
				gpio_pin_set_dt(&led_green, !anomaly);
				gpio_pin_set_dt(&led_red, anomaly);
				matrix_push(p, anomaly, in_range);
				/* One JSON object per line: the contract with the Linux side. */
				printk("{\"seq\":%u,\"t_ms\":%lld,\"temp_c\":%.2f,\"p_anomaly\":%.3f,"
				       "\"anomaly\":%d,\"sensor_ok\":%d,\"infer_us\":%u}\n",
				       seq++, (long long)k_uptime_get(), (double)c, (double)p, anomaly,
				       in_range, us);
			}
		}
		k_msleep(CONFIG_APP_SAMPLE_PERIOD_MS);
	}
	return 0;
}
