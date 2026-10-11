"""Two ways of turning 768 temperatures into a 192x144 panel image.

Nearest replicates each sensor pixel into a 6x6 block, which is fast but
visibly blocky. Smooth interpolates the temperature field 2x first, so the
blocks become 3x3 and the steps between them are graded rather than hard.

Why 2x and not a full 192x144 bilinear: the colour ramp has to be evaluated
*after* interpolation (interpolating one scalar is a third of the work of
interpolating R, G and B), and ulab costs roughly 1-3 us per element per pass.
Shading 768 pixels takes ~20 ms, so shading all 27648 would take well over
half a second. 2x puts 3072 pixels through the ramp, which is affordable.

Both classes expose the same two calls, so code.py can pick one and forget
about the difference:

    r = Smooth(disp, TEMP_LO, TEMP_SPAN, IMG_X, IMG_Y)
    r.draw(temps)
"""

from ulab import numpy as np

SRC_W, SRC_H = 32, 24
N_PIX = SRC_W * SRC_H

# The ramp reproduces the 64-entry palette of code_displayio.py: across the
# range R falls linearly, G is a V, and B rises then saturates.
R_SLOPE = -31.0 / 63.0


def _ramp(temps, lo, inv_span):
    """Temperatures -> ramp position, clipped to 0..63."""
    return np.clip((temps - lo) * inv_span, 0.0, 63.0)


def _shade(x, r, g):
    """Ramp position (already clipped) -> packed big-endian RGB565.

    r and g are caller-owned scratch arrays of the same length as x. Every
    field is floored before packing: a fractional R5 would carry up to 2047
    into the green and blue fields, which shows up as colour noise.
    """
    r[:] = x
    r *= R_SLOPE
    r += 31.0
    np.floor(r, out=r)
    r *= 2048.0                      # R5 << 11

    g[:] = abs(x - 31.5)
    g *= 2.0                         # G6 = |x - 31.5| * 2
    np.floor(g, out=g)
    g *= 32.0                        # G6 << 5
    r += g

    b = np.minimum(x, 31.0)          # B5
    np.floor(b, out=b)
    r += b
    return np.array(r, dtype=np.uint16).byteswap()


class _Base:
    def __init__(self, disp, lo, span, x0, y0):
        self.disp = disp
        self.lo = lo
        self.inv_span = 1.0 / span
        self.x0 = x0
        self.y0 = y0

    def _window(self):
        self.disp.set_window(
            self.x0, self.y0, self.x0 + self.W - 1, self.y0 + self.H - 1
        )


class Nearest(_Base):
    """Each sensor pixel becomes a 6x6 block. Measured ~11 fps."""

    SCALE = 6
    W, H = SRC_W * 6, SRC_H * 6

    def __init__(self, disp, lo, span, x0, y0):
        super().__init__(disp, lo, span, x0, y0)
        rb = self.W * 2
        self._ba = bytearray(rb * SRC_H)
        self._16 = np.frombuffer(self._ba, dtype=np.uint16)
        views = [memoryview(self._ba)[r * rb:(r + 1) * rb] for r in range(SRC_H)]
        # Reversing the pixels and the row order mirrors horizontally only,
        # which is what the original code's frame[h*32 + (31-w)] did.
        self._views = views[::-1]
        self._r = np.zeros(N_PIX, dtype=np.float)
        self._g = np.zeros(N_PIX, dtype=np.float)

    def draw(self, temps):
        x = _ramp(temps, self.lo, self.inv_span)
        shaded = _shade(x, self._r, self._g)
        src = np.array(shaded[::-1], dtype=np.uint16)
        for k in range(self.SCALE):
            self._16[k::self.SCALE] = src
        self._window()
        self.disp.blit_repeat_rows(self._views, self.SCALE)


class Smooth(_Base):
    """2x bilinear on the temperature field, then 3x blocks.

    The interpolation runs on the reversed (H and V flipped) field, and the
    row views are emitted in reverse, which cancels the vertical flip and
    leaves the horizontal mirror -- same trick as Nearest.
    """

    SCALE = 3
    MID_W = SRC_W * 2                      # 64
    MID_H = SRC_H * 2                      # 48
    W, H = MID_W * 3, MID_H * 3            # 192 x 144

    def __init__(self, disp, lo, span, x0, y0):
        super().__init__(disp, lo, span, x0, y0)
        f = np.float
        n_h = SRC_H * self.MID_W           # 1536, the horizontally doubled field
        self._row_len = self.MID_W
        self._next = np.zeros(N_PIX, dtype=f)
        self._avg = np.zeros(N_PIX, dtype=f)
        self._h = np.zeros(n_h, dtype=f)   # rows r
        self._m = np.zeros(n_h, dtype=f)   # midpoints between rows r and r+1
        self._r = np.zeros(n_h, dtype=f)
        self._g = np.zeros(n_h, dtype=f)

        rb = self.W * 2
        self._even_ba = bytearray(rb * SRC_H)
        self._odd_ba = bytearray(rb * SRC_H)
        self._even16 = np.frombuffer(self._even_ba, dtype=np.uint16)
        self._odd16 = np.frombuffer(self._odd_ba, dtype=np.uint16)
        ev = [memoryview(self._even_ba)[r * rb:(r + 1) * rb] for r in range(SRC_H)]
        od = [memoryview(self._odd_ba)[r * rb:(r + 1) * rb] for r in range(SRC_H)]
        # Reversed interleave: [M23, E23, M22, E22, ... M0, E0]. In sensor
        # order that reads row 0, row 0, mid(0,1), row 1, mid(1,2), ... row 23,
        # i.e. correct top to bottom with the top row doubled at the edge.
        views = []
        for r in range(SRC_H - 1, -1, -1):
            views.append(od[r])
            views.append(ev[r])
        self._views = views

    def draw(self, temps):
        rev = np.array(temps[::-1], dtype=np.float)
        x = _ramp(rev, self.lo, self.inv_span)

        # --- horizontal: 32 -> 64 ---------------------------------------
        # nxt[c] = x[c+1], except at the right edge of each row, where the
        # flat shift would otherwise pull in the next row's first column.
        nxt = self._next
        nxt[0:N_PIX - 1] = x[1:N_PIX]
        last = SRC_W - 1
        for r in range(SRC_H):
            i = r * SRC_W + last
            nxt[i] = x[i]

        avg = self._avg
        avg[:] = x
        avg += nxt
        avg *= 0.5

        h = self._h
        h[0::2] = x                      # co-located samples
        h[1::2] = avg                    # midpoints between columns

        # --- vertical: 24 -> 48 -----------------------------------------
        # Rows are contiguous, so one offset add gives every midpoint row at
        # once. The last row has no successor and keeps its own value.
        m = self._m
        m[:] = h
        span = SRC_H * self._row_len - self._row_len
        mv = m[0:span]
        mv += h[self._row_len:]
        mv *= 0.5

        # --- shade and expand 3x horizontally ---------------------------
        sh = _shade(h, self._r, self._g)
        for k in range(self.SCALE):
            self._even16[k::self.SCALE] = sh
        sm = _shade(m, self._r, self._g)
        for k in range(self.SCALE):
            self._odd16[k::self.SCALE] = sm

        self._window()
        self.disp.blit_repeat_rows(self._views, self.SCALE)
