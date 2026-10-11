
import time
import board
import busio
import adafruit_mlx90640
import adafruit_st7789
import displayio
import terminalio
import neopixel


from fourwire import FourWire
from adafruit_display_text import label


pixel = neopixel.NeoPixel(board.LED, 1)
pixel.brightness = 0.5
pixel.fill((255,0,0))

displayio.release_displays()

i2c = busio.I2C(scl=board.GP7, sda=board.GP6, frequency=600000)

spi = busio.SPI(clock=board.GP2, MOSI=board.GP3, MISO=board.GP4)

mlx = adafruit_mlx90640.MLX90640(i2c)
print("MLX addr detected on I2C")
print([hex(i) for i in mlx.serial_number])

while not spi.try_lock():
    pass
spi.configure(baudrate=25000000)
spi.unlock()
print(spi.frequency)
tft_cs = board.GP1
tft_dc = board.GP0

display_bus = FourWire(spi, command=tft_dc, chip_select=tft_cs, reset=None)
display = adafruit_st7789.ST7789(display_bus, width=240, height=320, rowstart=0)

splash = displayio.Group()
bg = displayio.Bitmap(240, 320, 1)
bgp = displayio.Palette(1)
bgp[0] = 0xa0a0a0
bgt = displayio.TileGrid(bg, pixel_shader=bgp, x=0, y=0)
splash.append(bgt)
display.root_group = splash

colors = 64

start_color = (0.0,0.0,1.0)

mid_offset = colors / 2.0
mid_color = (0.5,1.0,0.0)

end_color = (1.0,0.0,0.0)


def mix(s, e, amount):
    r = s[0]*(1.0-amount) + e[0]*amount
    g = s[1]*(1.0-amount) + e[1]*amount
    b = s[2]*(1.0-amount) + e[2]*amount
#    print(r,g,b)
    assert(r <= 1)
    assert(g <= 1)
    assert(b <= 1)
    return int((1-r)*255)*0x10000 + int((1-g)*255)*0x100 + int((1-b)*255)

def clamp(a, mi, ma):
    if a < mi:
        return mi
    elif a > ma:
        return ma
    else:
        return a

color_bitmap = displayio.Bitmap(32, 24, int(colors))
color_palette = displayio.Palette(int(colors))
for ix in range(colors):
    if ix < mid_offset:
        color_palette[ix] = mix(start_color, mid_color, ix / mid_offset)
    else:
        color_palette[ix] = mix(mid_color, end_color, (ix - mid_offset) / (mid_offset-1))
#    print(f"{ix:2} {color_palette[ix]:06x} - {(ix - mid_offset) / mid_offset}")


bg_sprite = displayio.TileGrid(color_bitmap,
                               pixel_shader=color_palette,
                               x=4, y=4)

thermal = displayio.Group()
thermal.scale=6
thermal.append(bg_sprite)
splash.append(thermal)


text_group = displayio.Group(x = 32, y = 200)
text_area = label.Label(terminalio.FONT, text="", color=0xffffff)
text_group.append(text_area)
splash.append(text_group)

def map_color(t):
    return int(clamp((t-10)*2, 0, int(colors-1)))

def show(mi, ma, cn):
    text_area.text = f"Min: {mi:.1f} ({map_color(mi)})\nMax: {ma:.1f} ({map_color(ma)})\nCenter: {cn:.1f}"


mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ

frame = [0] * 768
while True:
    stamp = time.monotonic()
    try:
        mlx.getFrame(frame)
    except ValueError:
        continue
#    print("Read 2 frames in %0.2f s" % (time.monotonic() - stamp))
    temp_min = min(frame)
    temp_max = max(frame)
    temp_range = temp_max - temp_min
    temp_center = 0
    for h in range(10, 14):
        for w in range(14, 18):
            temp_center += frame[h*32+w]
    temp_center /= 16
    show(temp_min, temp_max, temp_center)
    for h in range(24):
        for w in range(32):
            t = frame[h * 32 + (31-w)]
            color_bitmap[h*32+w] = map_color(t)
