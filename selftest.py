"""Static test pattern, to check display geometry independently of the sensor.

Run this instead of code.py (copy it over code.py on the device) if the image
geometry looks wrong. It uses the same window, buffers and blit path as
code.py, but with data that does not depend on the sensor at all:

  - a 1 px white border around the image area
  - a horizontal sweep through the full colour ramp, cold (left) to hot
  - a vertical brightness step, so top and bottom are distinguishable
  - a single white pixel in the TOP-LEFT corner, 3 px in the TOP-RIGHT

Top-left single dot and top-right triple dot together pin down both the
orientation and the mirroring.
"""

import time

import board
import busio
import displayio
from ulab import numpy as np

import font5x7
from st7789_fast import FastST7789

SRC_W, SRC_H, SCALE = 32, 24, 6
IMG_W, IMG_H = SRC_W * SCALE, SRC_H * SCALE
IMG_X, IMG_Y = (240 - IMG_W) // 2, 24
ROW_BYTES = IMG_W * 2
BG565 = 0x2104

displayio.release_displays()
spi = busio.SPI(clock=board.GP2, MOSI=board.GP3, MISO=board.GP4)
while not spi.try_lock():
    pass
spi.configure(baudrate=62_500_000)
spi.unlock()
disp = FastST7789(spi, command=board.GP0, chip_select=board.GP1, reset=board.GP8)
disp.fill(0, 0, 240, 320, BG565)

row_ba = bytearray(ROW_BYTES * SRC_H)
row16 = np.frombuffer(row_ba, dtype=np.uint16)
_views = [memoryview(row_ba)[r * ROW_BYTES:(r + 1) * ROW_BYTES] for r in range(SRC_H)]
row_views = _views[::-1]

# Same ramp as code.py, built per source pixel with integer fields.
vals = np.zeros(SRC_W * SRC_H, dtype=np.uint16)
for r in range(SRC_H):
    for c in range(SRC_W):
        x = c * 63.0 / (SRC_W - 1)
        r5 = int(31 - x * 31 / 63)
        g6 = int(abs(x - 31.5) * 2)
        b5 = int(min(x, 31))
        if r < 4:                      # dim the top rows
            r5 >>= 1
            g6 >>= 1
            b5 >>= 1
        v = (r5 << 11) | (g6 << 5) | b5
        if r == 0 or r == SRC_H - 1 or c == 0 or c == SRC_W - 1:
            v = 0xFFFF                 # border
        vals[r * SRC_W + c] = ((v & 0xFF) << 8) | (v >> 8)   # panel byte order

# orientation markers, drawn in sensor coordinates
vals[1 * SRC_W + 1] = 0xFFFF                               # top-left: 1 dot
for c in (SRC_W - 4, SRC_W - 3, SRC_W - 2):                 # top-right: 3 dots
    vals[1 * SRC_W + c] = 0xFFFF

src = np.array(vals[::-1], dtype=np.uint16)
for k in range(SCALE):
    row16[k::SCALE] = src
disp.set_window(IMG_X, IMG_Y, IMG_X + IMG_W - 1, IMG_Y + IMG_H - 1)
disp.blit_repeat_rows(row_views, SCALE)

TEXT_W, TEXT_H = 96, 32
strip = bytearray(TEXT_W * TEXT_H * 2)
strip[:] = bytes((BG565 >> 8, BG565 & 0xFF)) * (TEXT_W * TEXT_H)
font5x7.draw(strip, TEXT_W, "1 DOT TL", 0, 0, 0xFF, 0xFF)
font5x7.draw(strip, TEXT_W, "3 DOT TR", 0, 8, 0xFF, 0xFF)
font5x7.draw(strip, TEXT_W, "COLD 0 -", 0, 16, 0xFF, 0xFF)
font5x7.draw(strip, TEXT_W, "-123.456", 0, 24, 0xFF, 0xFF)
disp.set_window(IMG_X, IMG_Y + IMG_H + 24, IMG_X + TEXT_W - 1, IMG_Y + IMG_H + 24 + TEXT_H - 1)
disp.begin_pixels()
spi.write(strip)
disp.end_pixels()

print("selftest pattern drawn:")
print("  border       1 px white, all four edges of the 192x144 image")
print("  top 4 rows   dimmed (so top/bottom differ)")
print("  markers      1 white dot near TOP-LEFT, 3 near TOP-RIGHT")
print("  ramp         left = cold end of palette, right = hot end")
print("  text         four lines below the image")
while True:
    time.sleep(1)
