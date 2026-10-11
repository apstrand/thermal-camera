"""MLX90640 thermal camera, raw-SPI display path.

See code_displayio.py for the displayio version, which is slower (displayio
costs a flat 5.4 us per displayed pixel on this board, 150 ms for this image)
but shorter and known-good. Everything here was measured on the hardware;
the numbers in the comments are real.
"""

import gc
import time

import board
import busio
import displayio
import neopixel
from ulab import numpy as np

import adafruit_mlx90640

import font5x7
from mlx90640_fast import FastMLX90640
from st7789_fast import FastST7789

# --- tunables ------------------------------------------------------------
# The sensor emits one subpage per period; a displayed frame uses one. 2 Hz
# (the original setting) therefore capped everything at 1 fps. 32 Hz measured
# no faster than 16 and quadrupled the torn-frame count.
REFRESH_RATE = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
I2C_FREQ = 1_000_000       # MLX90640 supports FM+
# busio.SPI rounds DOWN to clk_peri/even, so 25 MHz silently becomes 20.8.
# 62.5 MHz is exact and measured 6.53 MB/s through busio.SPI.write.
SPI_FREQ = 62_500_000

SRC_W, SRC_H = 32, 24
SCALE = 6
IMG_W, IMG_H = SRC_W * SCALE, SRC_H * SCALE     # 192 x 144
IMG_X, IMG_Y = (240 - IMG_W) // 2, 24
ROW_BYTES = IMG_W * 2

TEXT_W, TEXT_H = 96, 32
TEXT_X, TEXT_Y = IMG_X, IMG_Y + IMG_H + 24
TEXT_EVERY = 8

TEMP_LO = 10.0             # degC at the cold end of the ramp
TEMP_SPAN = 0.5            # degC per colour step, 64 steps -> 10.0 .. 41.5
BG565 = 0x2104             # dark grey, big-endian RGB565

# CircuitPython keeps displays alive across soft reloads, so a display left
# over from a previous run would still be holding the SPI pins.
displayio.release_displays()

pixel = neopixel.NeoPixel(board.NEOPIXEL, 1)
pixel.brightness = 0.5
pixel.fill((255, 0, 0))

i2c = busio.I2C(scl=board.GP7, sda=board.GP6, frequency=I2C_FREQ)
# refine=False drops the driver's third solve pass: 9 ms per frame, for a
# measured max error of 0.023 degC between 0 and 80 degC (0.41 above 80).
# The colour ramp quantises at 0.5 degC and the sensor is +/-1-2 degC.
mlx = FastMLX90640(i2c, refine=False)
mlx.refresh_rate = REFRESH_RATE
pixel.fill((255, 255, 0))
print("MLX serial:", [hex(i) for i in mlx.serial_number])

spi = busio.SPI(clock=board.GP2, MOSI=board.GP3, MISO=board.GP4)
while not spi.try_lock():
    pass
spi.configure(baudrate=SPI_FREQ)
print("SPI", spi.frequency)
spi.unlock()

disp = FastST7789(spi, command=board.GP0, chip_select=board.GP1, reset=board.GP8)
disp.fill(0, 0, 240, 320, BG565)
pixel.fill((0, 255, 255))

# --- buffers -------------------------------------------------------------
# One image row per source row: 192 px of RGB565. The vertical scale-up is
# done by writing each row to the panel 6 times, which avoids materialising
# the 55 kB full-size image (two of those will not fit in the heap).
row_ba = bytearray(ROW_BYTES * SRC_H)                      # 9216 B
row16 = np.frombuffer(row_ba, dtype=np.uint16)             # view, 4608 px
_views = [memoryview(row_ba)[r * ROW_BYTES:(r + 1) * ROW_BYTES] for r in range(SRC_H)]
# Reversing src and writing the rows bottom-to-top mirrors the image
# horizontally only, which is what the original code's 31-w index did.
row_views = _views[::-1]

text_strip = bytearray(TEXT_W * TEXT_H * 2)
# Clearing this with a Python per-byte loop cost 92 ms per text update;
# assigning a prebuilt image is a single memcpy.
BG_STRIP = bytes((BG565 >> 8, BG565 & 0xFF)) * (TEXT_W * TEXT_H)

# --- colour ramp ---------------------------------------------------------
# The displayio version used a 64-entry palette. Reproducing it arithmetically
# keeps the per-pixel work inside ulab: across the ramp R falls linearly,
# G is a V (hence abs), and B rises then saturates.
# R5 << 11 as a single multiply-add: R5 = 31 - x*31/63, so
# R5 * 2048 == x * (-31/63 * 2048) + 31 * 2048.
R_MUL = -31.0 / 63.0 * 2048.0
R_ADD = 31.0 * 2048.0
INV_SPAN = 1.0 / TEMP_SPAN


def shade(temps):
    """768 temperatures -> 768 big-endian RGB565 values."""
    x = np.clip((temps - TEMP_LO) * INV_SPAN, 0.0, 63.0)
    c = x * R_MUL
    c += R_ADD                       # R5 << 11
    g = abs(x - 31.5)
    g *= 64.0                        # G6 << 5, with G6 = |x-31.5| * 2
    c += g
    c += np.minimum(x, 31.0)         # B5
    return np.array(c, dtype=np.uint16).byteswap()


CENTER = [h * SRC_W + w for h in range(10, 14) for w in range(14, 18)]

gc.collect()
print("free heap after setup:", gc.mem_free())
pixel.brightness = 0.1
pixel.fill((0, 200, 0))

tick = 0
fps = 0.0
errors = 0
last = time.monotonic_ns()

while True:
    try:
        mlx.wait_for_subpage()
        temps = mlx.temperatures()
    except (RuntimeError, OSError):
        errors += 1
        continue

    src = np.array(shade(temps)[::-1], dtype=np.uint16)
    for k in range(SCALE):
        row16[k::SCALE] = src

    disp.set_window(IMG_X, IMG_Y, IMG_X + IMG_W - 1, IMG_Y + IMG_H - 1)
    disp.blit_repeat_rows(row_views, SCALE)

    if tick % TEXT_EVERY == 0:
        lo = np.min(temps)
        hi = np.max(temps)
        cen = 0.0
        for p in CENTER:
            cen += temps[p]
        cen /= len(CENTER)

        text_strip[:] = BG_STRIP
        font5x7.draw(text_strip, TEXT_W, "MIN %.1f" % lo, 0, 0, 0xFF, 0xFF)
        font5x7.draw(text_strip, TEXT_W, "MAX %.1f" % hi, 0, 8, 0xFF, 0xFF)
        font5x7.draw(text_strip, TEXT_W, "CTR %.1f" % cen, 0, 16, 0xFF, 0xFF)
        font5x7.draw(text_strip, TEXT_W, "%.1f FPS" % fps, 0, 24, 0xFF, 0xFF)
        disp.set_window(TEXT_X, TEXT_Y, TEXT_X + TEXT_W - 1, TEXT_Y + TEXT_H - 1)
        disp.begin_pixels()
        spi.write(text_strip)
        disp.end_pixels()

        now = time.monotonic_ns()
        fps = TEXT_EVERY * 1e9 / (now - last)
        last = now
        pixel.fill((0, 200, 0) if tick % (2 * TEXT_EVERY) else (0, 100, 0))
        print("%.1f fps  torn %d  clamp %d  err %d  free %d"
              % (fps, mlx.torn_frames, mlx.clamped_frames, errors, gc.mem_free()))

    tick += 1
