"""ST7789 driven from a PIO state machine, with DMA-backed row writes.

Why not busio.SPI: two reasons, both measured on this board.

  * busio.SPI's clock comes from clk_peri, and CircuitPython re-points
    clk_peri to 48 MHz the moment cpu.frequency is assigned. Overclocking to
    250 MHz therefore caps SPI at 24 MHz. PIO clocks from clk_sys instead, so
    it keeps 62.5 Mbit/s at any CPU frequency -- the overclock and a fast
    panel stop being mutually exclusive.
  * background_write is DMA-backed and queues, so the per-call overhead of
    many small row writes disappears behind the transfer.

Blitting a 192x144 image as 24 rows written 6x each, at cpu 250 MHz:

    busio.SPI, 144 blocking writes      23.3 ms
    PIO,       144 blocking writes       8.7 ms
    PIO,       144 queued background     7.2 ms   <- what blit_repeat_rows does
    PIO,       one 55 kB write           7.2 ms

The last line is no better, and getting there would mean replicating 24 rows
into a contiguous 55 kB frame, which costs 20.8 ms. So the row structure
stays and the buffer stays at 9 kB.

displayio is not involved at all; see code_displayio.py for that version and
font5x7.py for the text, which has to be drawn by hand as a result.
"""

import array
import time

import digitalio
import rp2pio

# SPI mode 0 transmit, MSB first. SCK is a sideset pin, so each bit costs two
# instructions and the state machine runs at twice the bit rate.
#
#   .side_set 1
#   out pins, 1   side 0     ; present the bit while SCK is low
#   nop           side 1     ; raise SCK; the panel samples on the rising edge
_PROGRAM = array.array("H", (0x6001, 0xB042))

# cmd, data, delay_ms -- the sequence adafruit_st7789 uses, replayed directly.
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

MAX_BIT_RATE = 62_500_000


class FastST7789:
    def __init__(self, clock, mosi, command, chip_select, reset,
                 width=240, height=320, bit_rate=MAX_BIT_RATE):
        self.width = width
        self.height = height
        self._dc = digitalio.DigitalInOut(command)
        self._dc.switch_to_output(value=True)
        self._cs = digitalio.DigitalInOut(chip_select)
        self._cs.switch_to_output(value=True)
        self._rst = digitalio.DigitalInOut(reset)
        self._rst.switch_to_output(value=True)
        self._win = bytearray(4)
        self._cmd1 = bytearray(1)
        # StateMachine.writing reads True on a freshly constructed state
        # machine, before any background write has ever been issued, so it
        # cannot be polled unconditionally -- a drain loop would spin
        # forever. Track whether there is actually anything to wait for.
        self._queued = False

        self._sm = rp2pio.StateMachine(
            _PROGRAM,
            frequency=bit_rate * 2,
            first_out_pin=mosi, out_pin_count=1,
            first_sideset_pin=clock, sideset_pin_count=1,
            auto_pull=True, pull_threshold=8, out_shift_right=False,
        )
        self.bit_rate = self._sm.frequency // 2

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
        self._cmd1[0] = cmd
        self._cs.value = False
        self._dc.value = False
        self._sm.write(self._cmd1)
        if data:
            self._dc.value = True
            self._sm.write(data)
        self._cs.value = True

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
        self._cmd1[0] = _RAMWR
        self._cs.value = False
        self._dc.value = False
        self._sm.write(self._cmd1)
        self._dc.value = True

    def end_pixels(self):
        """Wait for any queued DMA to drain, then deselect."""
        if self._queued:
            while self._sm.writing:
                pass
            self._queued = False
        self._cs.value = True

    def write_pixels(self, buf):
        self._sm.write(buf)

    # ----------------------------------------------------------------- high

    def blit_repeat_rows(self, rows, repeat):
        """Write each memoryview in `rows` to the panel `repeat` times.

        This is the vertical scale-up: 24 source rows written 6x each fill
        144 panel rows without ever materialising the 55 kB full-size image.
        The writes are queued rather than blocking, so the cost of issuing
        144 of them hides behind the transfer itself.

        The caller must not touch the row buffers until this returns.
        """
        queue = self._sm.background_write
        self.begin_pixels()
        for row in rows:
            for _ in range(repeat):
                queue(once=row)
        self._queued = True
        self.end_pixels()

    def fill(self, x0, y0, w, h, color565):
        """Solid fill, used to clear the screen at startup."""
        chunk = min(h, max(1, 4096 // (w * 2)))
        row = bytearray(bytes((color565 >> 8, color565 & 0xFF)) * (w * chunk))
        mv = memoryview(row)
        self.set_window(x0, y0, x0 + w - 1, y0 + h - 1)
        self.begin_pixels()
        done = 0
        while done < h:
            n = min(chunk, h - done)
            self._sm.write(mv[: w * n * 2])
            done += n
        self.end_pixels()
