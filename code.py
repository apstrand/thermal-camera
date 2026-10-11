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
import renderer
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

# SMOOTH interpolates the temperature field 2x before colouring, so the
# blocks are 3x3 instead of 6x6 and their edges are graded. Measured offline
# on synthetic scenes: on a 0.6 degC/px gradient the mean run of identical
# pixels drops 6.08 -> 3.10 and the distinct colour count rises 54 -> 115; on
# a face-like warm blob, 8.86 -> 5.73 and 39 -> 58. It does nothing on a very
# flat scene, where neighbouring sensor pixels already share a colour. It
# costs framerate -- see the figure printed at startup.
SMOOTH = True

SRC_W, SRC_H = 32, 24
IMG_W, IMG_H = SRC_W * 6, SRC_H * 6             # 192 x 144 either way
IMG_X, IMG_Y = (240 - IMG_W) // 2, 24

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

# --- renderer ------------------------------------------------------------
# Smooth wants ~48 kB of buffers against Nearest's ~9 kB, and its first draw
# is where any unsupported ulab slice assignment would surface. Prove both
# before committing to it, so a failure degrades to a working camera instead
# of a board that will not boot.
def _make_renderer():
    if SMOOTH:
        try:
            r = renderer.Smooth(disp, TEMP_LO, TEMP_SPAN, IMG_X, IMG_Y)
            mlx.wait_for_subpage()
            r.draw(mlx.temperatures())
            print("renderer: Smooth (2x bilinear, 3x3 blocks)")
            return r
        except (MemoryError, TypeError, ValueError, IndexError) as err:
            print("renderer: Smooth unavailable (%s: %s), falling back"
                  % (type(err).__name__, err))
            gc.collect()
    r = renderer.Nearest(disp, TEMP_LO, TEMP_SPAN, IMG_X, IMG_Y)
    print("renderer: Nearest (6x6 blocks)")
    return r


render = _make_renderer()

text_strip = bytearray(TEXT_W * TEXT_H * 2)
# Clearing this with a Python per-byte loop cost 92 ms per text update;
# assigning a prebuilt image is a single memcpy.
BG_STRIP = bytes((BG565 >> 8, BG565 & 0xFF)) * (TEXT_W * TEXT_H)

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

    render.draw(temps)

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
