r"""Slower displayio version of the thermal camera, kept as a fallback.

Measured 4.0 fps at 192x144 against code.py's 10.0, because displayio costs a
flat 5.4 us per displayed pixel on this board (150 ms for this image) and
FourWire.send only reaches 2.19 MB/s where busio.SPI.write reaches 6.53.
Shorter and less hand-rolled than code.py, so useful if the raw-SPI path ever
misbehaves. Drop IMAGE_SCALE for more speed here.
"""
import gc
import time

import board
import busio
import displayio
import terminalio
import neopixel
import bitmaptools

from fourwire import FourWire
from adafruit_display_text import label
from ulab import numpy as np

import adafruit_mlx90640
import adafruit_st7789

from mlx90640_fast import FastMLX90640

# --- tunables ------------------------------------------------------------
# 2 Hz (the old setting) capped the whole thing at 1 frame/s: the sensor
# emits one subpage every 500 ms and a full frame needs two. 16 Hz gives a
# subpage every 62.5 ms. REFRESH_32_HZ was measured to be no faster at any
# image size -- the display is the bottleneck, not the sensor -- while
# quadrupling the torn-frame count, so there is no reason to go above 16.
REFRESH_RATE = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
I2C_FREQ = 1_000_000      # MLX90640 supports FM+
# busio.SPI rounds the request DOWN to clk_peri/even: asking for 25 MHz gets
# you 20.8 MHz. 62.5 MHz is exact (clk_peri/2) and the panel is fine with it.
SPI_FREQ = 62_500_000
TEXT_EVERY = 8            # frames between text/NeoPixel updates

# IMAGE_SCALE is the framerate knob. displayio costs a flat 5.4 us per
# displayed pixel on this board -- measured, and independent of the scale
# factor, the row count and the width, so it is pure composite cost. Sensor
# read + conversion is about 65 ms, so the display dominates everything above
# scale 3. Measured end to end at REFRESH_16_HZ:
#
#   scale 6  192x144  27648 px   150 ms draw    4.3 fps
#   scale 4  128x96   12288 px    66 ms draw    7.1 fps
#   scale 3   96x72    6912 px    38 ms draw    9.0 fps
#   scale 2   64x48    3072 px    17 ms draw   11.3 fps
#
# (those four were measured before the RGB565 packing fix, which added
#  three floor passes; the raw-SPI code.py now runs 10.0 fps at scale 6)
#
# 6 keeps the original layout. Drop it for a smaller, faster image.
IMAGE_SCALE = 6

COLORS = 64
TEMP_LO = 10.0            # degC mapped to colour 0
TEMP_SPAN = 0.5           # degC per colour step

pixel = neopixel.NeoPixel(board.NEOPIXEL, 1)
pixel.brightness = 0.5
pixel.fill((255, 0, 0))

displayio.release_displays()

i2c = busio.I2C(scl=board.GP7, sda=board.GP6, frequency=I2C_FREQ)
spi = busio.SPI(clock=board.GP2, MOSI=board.GP3, MISO=board.GP4)

mlx = FastMLX90640(i2c)
mlx.refresh_rate = REFRESH_RATE
pixel.fill((255, 255, 0))
print("MLX addr detected on I2C")
print([hex(i) for i in mlx.serial_number])

while not spi.try_lock():
    pass
spi.configure(baudrate=SPI_FREQ)
spi.unlock()
print("SPI", spi.frequency)

tft_cs = board.GP1
tft_dc = board.GP0

display_bus = FourWire(spi, command=tft_dc, chip_select=tft_cs, reset=board.GP8)
display = adafruit_st7789.ST7789(display_bus, width=240, height=320, rowstart=0)
# Refresh once per sensor frame instead of whenever displayio feels like it.
display.auto_refresh = False

pixel.fill((0, 255, 255))
splash = displayio.Group()
bg = displayio.Bitmap(240, 320, 1)
bgp = displayio.Palette(1)
bgp[0] = 0xA0A0A0
bgt = displayio.TileGrid(bg, pixel_shader=bgp, x=0, y=0)
splash.append(bgt)
display.root_group = splash

start_color = (0.0, 0.0, 1.0)
mid_offset = COLORS / 2.0
mid_color = (0.5, 1.0, 0.0)
end_color = (1.0, 0.0, 0.0)


def mix(s, e, amount):
    r = s[0] * (1.0 - amount) + e[0] * amount
    g = s[1] * (1.0 - amount) + e[1] * amount
    b = s[2] * (1.0 - amount) + e[2] * amount
    return (
        int((1 - r) * 255) * 0x10000 + int((1 - g) * 255) * 0x100 + int((1 - b) * 255)
    )


color_bitmap = displayio.Bitmap(32, 24, COLORS)
color_palette = displayio.Palette(COLORS)
for ix in range(COLORS):
    if ix < mid_offset:
        color_palette[ix] = mix(start_color, mid_color, ix / mid_offset)
    else:
        color_palette[ix] = mix(
            mid_color, end_color, (ix - mid_offset) / (mid_offset - 1)
        )

bg_sprite = displayio.TileGrid(
    color_bitmap, pixel_shader=color_palette, x=4, y=4
)
# Mirror in the TileGrid rather than reversing the index arithmetic per pixel.
bg_sprite.flip_x = True

thermal = displayio.Group()
thermal.scale = IMAGE_SCALE
thermal.append(bg_sprite)
splash.append(thermal)

text_group = displayio.Group(x=32, y=200)
text_area = label.Label(terminalio.FONT, text="", color=0xFFFFFF)
text_group.append(text_area)
splash.append(text_group)


def map_color(t):
    i = int((t - TEMP_LO) / TEMP_SPAN + 0.5)
    return 0 if i < 0 else (COLORS - 1 if i > COLORS - 1 else i)


# Centre patch: rows 10..13, columns 14..17.
CENTER = [h * 32 + w for h in range(10, 14) for w in range(14, 18)]

pixel.brightness = 0.1
pixel.fill((0, 200, 0))

gc.collect()
print("free heap after setup:", gc.mem_free())

tick = 0
fps = 0.0
last_stamp = time.monotonic_ns()
errors = 0

while True:
    try:
        mlx.wait_for_subpage()
        temps = mlx.temperatures()
    except (RuntimeError, OSError):
        errors += 1
        continue

    bitmaptools.arrayblit(color_bitmap, mlx.palette_bytes(TEMP_LO, TEMP_SPAN, COLORS))

    if tick % TEXT_EVERY == 0:
        temp_min = np.min(temps)
        temp_max = np.max(temps)
        temp_center = 0.0
        for p in CENTER:
            temp_center += temps[p]
        temp_center /= len(CENTER)
        text_area.text = (
            "Min: %.1f (%d)\nMax: %.1f (%d)\nCenter: %.1f\n%.1f fps  t%d e%d"
            % (
                temp_min,
                map_color(temp_min),
                temp_max,
                map_color(temp_max),
                temp_center,
                fps,
                mlx.torn_frames,
                errors,
            )
        )
        now = time.monotonic_ns()
        fps = TEXT_EVERY * 1e9 / (now - last_stamp)
        last_stamp = now
        print("%.1f fps  torn %d  clamped %d  free %d"
              % (fps, mlx.torn_frames, mlx.clamped_frames, gc.mem_free()))
        pixel.fill((0, 200, 0) if tick % (2 * TEXT_EVERY) else (0, 100, 0))

    display.refresh(minimum_frames_per_second=0)
    tick += 1
