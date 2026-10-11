"""Minimal ST7789 driver that talks to busio.SPI directly.

Why this exists: on this board FourWire.send sustains 2.19 MB/s while
busio.SPI.write sustains 6.53 MB/s (84% of the 62.5 MHz theoretical 7.81),
and displayio's own compositing costs a flat 5.4 us per displayed pixel --
150 ms for a 192x144 image. Driving the panel directly turns that same image
into ~22 ms. The cost is that displayio is gone, so text has to be drawn by
hand (see font5x7.py).

The SPI lock is taken once in __init__ and never released: nothing else on
this board uses the bus.
"""

import time

import digitalio

# cmd, data, delay_ms -- the sequence adafruit_st7789 uses, replayed directly
# so this module does not depend on displayio having run first.
_INIT = (
    (0x01, b"", 150),        # SWRESET
    (0x11, b"", 500),        # SLPOUT
    (0x3A, b"\x55", 10),     # COLMOD: 16 bit/px, RGB565
    (0x36, b"\x08", 0),      # MADCTL
    (0x21, b"", 10),         # INVON
    (0x13, b"", 10),         # NORON
    (0x36, b"\xC0", 0),      # MADCTL
    (0x29, b"", 500),        # DISPON
)

_CASET = 0x2A
_RASET = 0x2B
_RAMWR = 0x2C


class FastST7789:
    def __init__(self, spi, command, chip_select, reset, width=240, height=320):
        self.width = width
        self.height = height
        self._spi = spi
        self._dc = digitalio.DigitalInOut(command)
        self._dc.switch_to_output(value=True)
        self._cs = digitalio.DigitalInOut(chip_select)
        self._cs.switch_to_output(value=True)
        self._rst = digitalio.DigitalInOut(reset)
        self._rst.switch_to_output(value=True)
        self._win = bytearray(4)

        while not spi.try_lock():
            pass

        self._rst.value = False
        time.sleep(0.05)
        self._rst.value = True
        time.sleep(0.15)

        for cmd, data, delay in _INIT:
            self._command(cmd, data)
            if delay:
                time.sleep(delay / 1000.0)

    # ------------------------------------------------------------------ low

    def _command(self, cmd, data=b""):
        cs = self._cs
        dc = self._dc
        cs.value = False
        dc.value = False
        self._spi.write(bytes((cmd,)))
        if data:
            dc.value = True
            self._spi.write(data)
        cs.value = True

    def set_window(self, x0, y0, x1, y1):
        w = self._win
        w[0] = x0 >> 8
        w[1] = x0 & 0xFF
        w[2] = x1 >> 8
        w[3] = x1 & 0xFF
        self._command(_CASET, w)
        w[0] = y0 >> 8
        w[1] = y0 & 0xFF
        w[2] = y1 >> 8
        w[3] = y1 & 0xFF
        self._command(_RASET, w)

    def begin_pixels(self):
        """Issue RAMWR and leave the bus selected with DC high.

        Everything written until end_pixels() lands in the window set by the
        last set_window(), the controller advancing the address itself.
        """
        self._cs.value = False
        self._dc.value = False
        self._spi.write(b"\x2c")
        self._dc.value = True

    def end_pixels(self):
        self._cs.value = True

    # ----------------------------------------------------------------- high

    def blit_repeat_rows(self, rows, repeat):
        """Write each memoryview in `rows` to the panel `repeat` times.

        This is the vertical scale-up: 24 source rows written 6x each fill
        144 panel rows without ever materialising the 55 kB full-size image,
        which would not fit in the heap alongside everything else.
        """
        write = self._spi.write
        self.begin_pixels()
        for row in rows:
            for _ in range(repeat):
                write(row)
        self.end_pixels()

    def fill(self, x0, y0, w, h, color565):
        """Solid fill, used to clear the screen at startup."""
        chunk = min(h, max(1, 4096 // (w * 2)))
        row = bytearray(
            bytes((color565 >> 8, color565 & 0xFF)) * (w * chunk)
        )
        mv = memoryview(row)
        self.set_window(x0, y0, x0 + w - 1, y0 + h - 1)
        self.begin_pixels()
        done = 0
        while done < h:
            n = min(chunk, h - done)
            self._spi.write(mv[: w * n * 2])
            done += n
        self.end_pixels()
