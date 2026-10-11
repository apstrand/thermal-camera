"""Benchmark the stock adafruit_mlx90640 frame path against mlx90640_fast.

This measures the sensor/conversion side only, through displayio (see
code_displayio.py). The shipped code.py uses the raw-SPI path instead.

Copy over code.py (keep a backup) and watch the serial console; the results
are also drawn on the TFT in case no serial terminal is attached.

FastMLX90640 subclasses the stock driver, so both paths run against the same
object and the same sensor configuration -- the only difference measured is
the code doing the work.
"""

import gc
import time

import board
import busio
import displayio
import terminalio
import bitmaptools

from fourwire import FourWire
from adafruit_display_text import label
from ulab import numpy as np

import adafruit_mlx90640
import adafruit_st7789

from mlx90640_fast import FastMLX90640

I2C_FREQ = 1_000_000
SPI_FREQ = 62_500_000
ROUNDS = 10

RATES = (
    ("2 Hz", adafruit_mlx90640.RefreshRate.REFRESH_2_HZ),
    ("8 Hz", adafruit_mlx90640.RefreshRate.REFRESH_8_HZ),
    ("16 Hz", adafruit_mlx90640.RefreshRate.REFRESH_16_HZ),
    ("32 Hz", adafruit_mlx90640.RefreshRate.REFRESH_32_HZ),
)

displayio.release_displays()

i2c = busio.I2C(scl=board.GP7, sda=board.GP6, frequency=I2C_FREQ)
spi = busio.SPI(clock=board.GP2, MOSI=board.GP3, MISO=board.GP4)

# release_cal=False keeps the stock getFrame() path alive to compare against
mlx = FastMLX90640(i2c, release_cal=False)

while not spi.try_lock():
    pass
spi.configure(baudrate=SPI_FREQ)
spi.unlock()

display_bus = FourWire(spi, command=board.GP0, chip_select=board.GP1, reset=board.GP8)
display = adafruit_st7789.ST7789(display_bus, width=240, height=320, rowstart=0)
display.auto_refresh = False

root = displayio.Group()
color_bitmap = displayio.Bitmap(32, 24, 64)
color_palette = displayio.Palette(64)
for ix in range(64):
    color_palette[ix] = (ix * 4) * 0x10101
sprite = displayio.TileGrid(color_bitmap, pixel_shader=color_palette, x=4, y=4)
thermal = displayio.Group(scale=6)
thermal.append(sprite)
root.append(thermal)

out = label.Label(terminalio.FONT, text="benchmarking...", color=0xFFFFFF, x=4, y=180)
root.append(out)
display.root_group = root
display.refresh(minimum_frames_per_second=0)

print("I2C %d Hz, SPI %d Hz" % (I2C_FREQ, spi.frequency))
print("serial:", [hex(i) for i in mlx.serial_number])

lines = []


def report(text):
    print(text)
    lines.append(text)
    out.text = "\n".join(lines[-12:])
    display.refresh(minimum_frames_per_second=0)


def time_stock(rounds):
    """Stock getFrame: two subpage reads plus 768 pixels of Python math."""
    frame = [0.0] * 768
    gc.collect()
    ok = 0
    t0 = time.monotonic_ns()
    for _ in range(rounds):
        try:
            mlx.getFrame(frame)
            ok += 1
        except (ValueError, RuntimeError) as err:
            report("    stock raised %s" % type(err).__name__)
    dt = (time.monotonic_ns() - t0) / 1e9
    return dt / rounds, ok


def time_fast(rounds):
    """Fast path, split into read / convert / blit / display refresh."""
    gc.collect()
    t_read = t_conv = t_blit = t_disp = 0.0
    for _ in range(rounds):
        a = time.monotonic_ns()
        mlx.wait_for_subpage()
        b = time.monotonic_ns()
        temps = mlx.temperatures()
        c = time.monotonic_ns()
        bitmaptools.arrayblit(color_bitmap, mlx.palette_bytes(10.0, 0.5, 64))
        d = time.monotonic_ns()
        display.refresh(minimum_frames_per_second=0)
        e = time.monotonic_ns()
        t_read += b - a
        t_conv += c - b
        t_blit += d - c
        t_disp += e - d
    n = rounds * 1e9
    return t_read / n, t_conv / n, t_blit / n, t_disp / n


for name, rate in RATES:
    mlx.refresh_rate = rate
    report("--- sensor at %s ---" % name)

    # let the sensor settle at the new rate
    for _ in range(2):
        try:
            mlx.wait_for_subpage()
        except RuntimeError:
            pass

    read, conv, blit, disp = time_fast(ROUNDS)
    busy = read + conv + blit + disp
    report("  fast  %5.1f fps  (read %4.1f conv %4.1f blit %4.1f disp %4.1f ms)"
           % (1.0 / busy, read * 1000, conv * 1000, blit * 1000, disp * 1000))
    report("        torn %d  clamped %d  free %d"
           % (mlx.torn_frames, mlx.clamped_frames, gc.mem_free()))

    per, ok = time_stock(ROUNDS)
    if ok:
        report("  stock %5.1f fps  (%.0f ms/frame, %d/%d ok)"
               % (1.0 / per, per * 1000, ok, ROUNDS))
    else:
        report("  stock  failed at this rate")

report("done")
while True:
    time.sleep(1)
