# Extending the example: other sensors on the headers and the Qwiic connector

The firmware has one input hook and one output contract, so swapping the signal is a small job:

- `firmware/src/sensor.c` returns one `float` per sample through `sensor_read_celsius()`
  (plus `false` when the reading is out of range). Everything downstream, the ring buffer,
  the normalisation `(t - 50) / 20`, the model, the JSON line and the LED matrix, does not
  care where the number came from.
- `train/train_anomaly.py` builds the synthetic dataset in `make_dataset()` and the
  normalisation constants live in `train/model.py` (`MEAN`, `SCALE`) with their twins
  `kMean`/`kScale` in `firmware/src/main.cpp`. Change the signal and you change those two
  places, retrain, re-export, rebuild.

Everything on the UNO Q headers is **3.3 V**. Nothing here is 5 V tolerant; an old 5 V sensor
or shield needs a divider or a level shifter.

## Analog sensors on A0 to A5

The six analog pins go straight to the STM32U585's ADC1 (14-bit, 3.3 V full scale):

| Header | STM32 pin | ADC1 channel | Notes |
|---|---|---|---|
| A0 | PA4 | 9 | default input in this example, also DAC1 out |
| A1 | PA5 | 10 | also DAC2 out |
| A2 | PA6 | 11 | |
| A3 | PA7 | 12 | |
| A4 | PC1 | 2 | shared with I2C3 SDA (`Wire2` in Arduino terms) |
| A5 | PC0 | 1 | shared with I2C3 SCL |

The example reads channel 9 through `io-channels` in `firmware/app.overlay`:

```dts
/ {
	zephyr,user {
		io-channels = <&adc1 9>;     /* A0. Use <&adc1 10> for A1, and so on. */
	};
};
```

and converts millivolts to degrees with two Kconfig options that default to a TMP36:

```bash
west build -p always -b arduino_uno_q/stm32u585xx executorch-zephyr-arduino-uno-q/firmware -- \
  -DCONFIG_APP_SIM_SENSOR=n -DCONFIG_APP_MV_PER_DEGREE=100 -DCONFIG_APP_MV_OFFSET=500
```

`CONFIG_APP_MV_PER_DEGREE` is in tenths of a millivolt per degree (100 = 10 mV/C). For a
thermistor, a current-clamp output, a pressure transducer or anything else that is not linear
in temperature, replace the one line in `sensor_read_celsius()` that does the conversion and
rename the units to taste; the model only sees normalised floats.

A second channel (say vibration on A1 next to temperature on A0) means a two-feature window:
add the channel to `io-channels`, read both in `sensor.c`, widen the ring buffer and the
model's input in `train/model.py`, and keep `TEST_WINDOW` in `test_input.h` in step.

## Modulino nodes on the Qwiic connector

The Qwiic connector is on the STM32 side, on **I2C4 (PD12 SCL, PD13 SDA)**, 3.3 V, pulled up
on the board. In Arduino terms it is `Wire1`; the header SDA/SCL pins by AREF are I2C2
(`Wire`). Modulino nodes are plain I2C devices, so under Zephyr they are either an existing
sensor driver or a few `i2c_write_read_dt()` calls.

Enable I2C and the sensor subsystem (put these in `prj.conf` or an extra conf file):

```ini
CONFIG_I2C=y
CONFIG_SENSOR=y
```

### Modulino Thermo (HS3003 temperature + humidity, address 0x44)

Zephyr v4.4 has a driver. Devicetree overlay:

```dts
&i2c4 {
	status = "okay";
	modulino_thermo: hs300x@44 {
		compatible = "renesas,hs300x";
		reg = <0x44>;
	};
};
```

Reading it from `sensor.c` instead of the ADC:

```c
#include <zephyr/drivers/sensor.h>

static const struct device *const thermo = DEVICE_DT_GET(DT_NODELABEL(modulino_thermo));

int sensor_init(void)
{
	return device_is_ready(thermo) ? 0 : -ENODEV;
}

bool sensor_read_celsius(float *celsius)
{
	struct sensor_value t;
	if (sensor_sample_fetch(thermo) != 0 ||
	    sensor_channel_get(thermo, SENSOR_CHAN_AMBIENT_TEMP, &t) != 0) {
		return false;
	}
	*celsius = sensor_value_to_float(&t);
	return *celsius > -40.f && *celsius < 125.f;
}
```

Build with the extra files rather than editing the defaults:

```bash
west build ... -- -DEXTRA_DTC_OVERLAY_FILE=modulino_thermo.overlay -DEXTRA_CONF_FILE=modulino.conf
```

### Modulino Movement (LSM6DSOX accelerometer + gyro, address 0x6A)

Zephyr's `st,lsm6dso` driver covers the LSM6DSOX:

```dts
&i2c4 {
	status = "okay";
	modulino_movement: lsm6dso@6a {
		compatible = "st,lsm6dso";
		reg = <0x6a>;
		accel-odr = <4>;      /* 104 Hz */
		accel-range = <4>;    /* +-4 g */
	};
};
```

Then `sensor_channel_get(dev, SENSOR_CHAN_ACCEL_XYZ, v)` gives three values. A vibration
anomaly model would feed the window with the magnitude `sqrt(x*x + y*y + z*z)` sampled faster
than the 10 Hz used here, so raise `CONFIG_APP_SAMPLE_PERIOD_MS` accordingly and regenerate the
training data at the same rate.

### Modulino Distance (VL53L4CD, address 0x29)

Not in Zephyr v4.4's driver list. Start from the `st,vl53l1x` driver (same register
family) or talk to it with `i2c_write_read_dt()` using ST's ultra-lite driver sequence.

### Modulino Knob, Buttons, Buzzer, Pixels

These carry a small microcontroller with a simple register map rather than an off-the-shelf
sensor. There is no Zephyr driver; the register map is in the Arduino `Modulino` library
source and maps directly onto `i2c_write_dt()` / `i2c_write_read_dt()`. Their addresses are
configurable, so scan first.

### Finding addresses

A ten-line scan beats guessing:

```c
#include <zephyr/drivers/i2c.h>

static const struct device *const bus = DEVICE_DT_GET(DT_NODELABEL(i2c4));

void i2c_scan(void)
{
	uint8_t dummy;
	for (uint16_t addr = 0x08; addr < 0x78; addr++) {
		if (i2c_read(bus, &dummy, 1, addr) == 0) {
			printk("i2c4: device at 0x%02x\n", addr);
		}
	}
}
```

The Qwiic bus also appears on the JMISC connector pins PF14/PF15 (same I2C4, alternate
pinctrl); see `arduino_uno_q-common.dtsi` in Zephyr if you need that routing instead.

## Air quality: Grove Multichannel Gas v2, a PM2.5 sensor and a microphone

Pin map for the home air-quality build (the wake-word version lives in its own repo,
`unoq-echoglow-air-monitor`). Everything below is 3.3 V logic; the only 5 V in play is the
supply for the particulate sensor.

### Seeed Grove Multichannel Gas Sensor v2 (I2C, address 0x08)

| Grove pin | Wire colour | UNO Q | STM32 pin |
|---|---|---|---|
| 1 SCL | yellow | Qwiic SCL (or header SCL by AREF) | PD12 (I2C4) or PB10 (I2C2) |
| 2 SDA | white | Qwiic SDA (or header SDA) | PD13 (I2C4) or PB11 (I2C2) |
| 3 VCC | red | 3V3 | |
| 4 GND | black | GND | |

The easiest cable is a Grove-to-Qwiic (JST SH 1.0 mm) adapter straight into the Qwiic
connector, which is I2C4. The module runs its four sensor heaters from a 3.3 V supply fine; the
Seeed page quotes an output range of 3.3 to 5 V and the examples pair it with 3.3 V boards.
Budget around 100 mA for the heaters. The module needs a preheat before the readings mean
anything: minutes for a stable baseline, and Seeed quotes up to 48 h for a fresh GM-302B.

Protocol, from Seeed's Arduino library: write one command byte, then read four bytes
little-endian. The result is the module's 10-bit ADC count (0 to 1023 over 3.3 V):

| Command | Element | Gas |
|---|---|---|
| `0x01` | GM-102B | NO2 |
| `0x03` | GM-302B | ethanol (C2H5OH) |
| `0x05` | GM-502B | VOC |
| `0x07` | GM-702B | CO |
| `0xFE` | | heaters on (warm up), send once at boot |
| `0xFF` | | heaters off |
| `0x55` + new address | | change the I2C address |

```c
static const struct device *const bus = DEVICE_DT_GET(DT_NODELABEL(i2c4));

static int gas_read(uint8_t cmd, uint32_t *count)
{
	uint8_t buf[4];
	int ret = i2c_write(bus, &cmd, 1, 0x08);
	if (ret == 0) {
		ret = i2c_read(bus, buf, sizeof(buf), 0x08);
	}
	*count = sys_get_le32(buf);
	return ret;
}
```

The counts are qualitative (Seeed says so); for anomaly detection that is fine, the model
learns the house's normal baseline and flags departures. Feed the four channels as four
features, or just VOC and CO.

### PM2.5 sensor (laser scattering, UART)

Most hobby particulate sensors (Plantower PMS5003/PMSA003, Nova SDS011, Seeed's Grove HM3301
is the I2C exception) are the same deal: **5 V supply, 3.3 V logic UART at 9600 8N1**, one
32-byte frame per second. The UNO Q header UART is free in this example because the console
moved to the internal link.

| Sensor pin | UNO Q | STM32 pin |
|---|---|---|
| VCC (5 V) | 5V on the power header | |
| GND | GND | |
| TX (3.3 V logic) | D0 (RX) | PB7 (USART1 RX) |
| RX | D1 (TX), optional, only for sleep/wake commands | PB6 (USART1 TX) |

The sensor's fan needs the 5 V rail (up to about 100 mA on a PMS5003); its TX is 3.3 V so it
goes straight into D0. Never connect a 5 V-logic sensor's TX to the header. Devicetree:

```dts
&usart1 {
	status = "okay";
	current-speed = <9600>;
};
```

Plantower frames start `0x42 0x4D`, then a big-endian length (28), thirteen big-endian
16-bit values and a 16-bit checksum (sum of every byte before it). PM1.0/2.5/10 under
atmospheric conditions are values 4, 5 and 6. If your part is an SDS011 the frame is 10 bytes
starting `0xAA 0xC0` with PM2.5 and PM10 as little-endian tenths of a microgram.

### Microphone

The STM32U585 has an audio block (ADF/MDF for PDM mics) and SAI for I2S mics, but Zephyr
v4.4.0's STM32U5 devicetree does not expose either, so there is no driver to turn on. Three
routes, in order of effort:

1. **Analog electret (MAX9814 or MAX4466 breakout) on A1.** Out to A1 (PA5, ADC1 channel 10),
   VCC to 3V3, GND to GND. Sample at 16 kHz from a timer and run the wake-word model on the
   MCU with ExecuTorch. Works with what the example already uses (the ADC).
2. **I2S MEMS mic (INMP441, ICS-43434) on SAI1 block B.** SCK to D2 (PB3), WS to D1 (PB6), SD
   to D8 is not available, so use PB5 (ICSP MOSI / D11 alt) or move to block A: SCK on the
   header SCL pin (PB10), WS on D10 (PB9), SD on A4 (PC1). Needs an `st,stm32-sai` node added
   by hand in the overlay plus DMA; Zephyr has the driver, the SoC file just lacks the node.
3. **PDM mic, or the Linux side.** A PDM mic wants the ADF, which has no Zephyr driver yet.
   Put the mic on the QRB2210 instead (a USB mic, or an I2S mic on JMEDIA at 1.8 V) and run
   the wake word there with ExecuTorch on aarch64; the MCU keeps the sensors and the alarms.

## The LED matrix

`firmware/src/matrix.c` draws through Zephyr's display API, so any other feedback (a scrolling
temperature trace, a bigger "!" on alarm) is a framebuffer edit: 13 x 8 bytes, row-major,
values 0 to 7. The driver and devicetree node are vendored from upstream Zephyr main; once the
manifest moves to a release that includes `charlieplex-led-matrix`, delete `firmware/drivers/`
and `firmware/dts/` and the overlay keeps working.

## Checklist for a new signal

1. Wire it, 3.3 V only.
2. Point `sensor.c` at it (ADC channel, or a Zephyr sensor driver on I2C4).
3. Record a few minutes of normal operation and replace `make_dataset()` with real windows,
   or keep the simulator shape but change `MEAN`/`SCALE` to your signal's range.
4. `python train/train_anomaly.py && python train/run_pte_host.py`
5. Rebuild, flash, watch for `SELF-TEST PASS`, then watch the JSON on the Linux side.
