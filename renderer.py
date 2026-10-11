"""Two ways of turning 768 temperatures into a 192x144 panel image.

Nearest replicates each sensor pixel into a 6x6 block, which is fast but
visibly blocky. Smooth interpolates the temperature field 2x first, so the
blocks become 3x3 and the steps between them are graded rather than hard.

Why 2x and not a full 192x144 bilinear: the colour ramp has to be evaluated
*after* interpolation (interpolating one scalar is a third of the work of
interpolating R, G and B), and ulab costs roughly 1-3 us per element per pass.
Shading 768 pixels takes ~20 ms, so shading all 27648 would take well over
half a second. 2x puts 3072 pixels through the ramp -- four times the
shading work, which is most of why Smooth costs what it does.

Measured on the hardware: Nearest 10.0 fps, Smooth 4.3 fps.

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

# ulab's slice assignment copies its right-hand side, so `dst[:] = src` on a
# 1536-element float array asks for a 6 kB contiguous block. At steady state
# this heap is fragmented enough that such a request fails with 33 kB still
# free, so Smooth works in 768-element chunks and never asks for more than
# 3 kB at once.
CHUNK = 768


def _ramp(temps, lo, inv_span):
    """Temperatures -> ramp position, clipped to 0..63."""
    return np.clip((temps - lo) * inv_span, 0.0, 63.0)


def _shade(x):
    """Ramp position (already clipped to 0..63) -> packed big-endian RGB565.

    Every field is floored before packing: a fractional R5 would carry up to
    2047 into the green and blue fields, which shows up as colour noise.

    Callers keep x at CHUNK elements or fewer so the temporaries stay around
    3 kB. A 1536-element version wanted 6 kB contiguous blocks, which this
    heap fails to supply at steady state with 33 kB still free.
    """
    r = x * R_SLOPE
    r += 31.0
    np.floor(r, out=r)
    r *= 2048.0                      # R5 << 11

    g = abs(x - 31.5)
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
    """Each sensor pixel becomes a 6x6 block. Measured 10.0 fps."""

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

    def draw(self, temps):
        x = _ramp(temps, self.lo, self.inv_span)
        shaded = _shade(x)
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
        self._n = n_h
        self._row_len = self.MID_W
        self._next = np.zeros(N_PIX, dtype=f)
        self._h = np.zeros(n_h, dtype=f)   # rows r
        self._m = np.zeros(n_h, dtype=f)   # midpoints between rows r and r+1
        self._out = np.zeros(n_h, dtype=np.uint16)

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
        rev -= self.lo
        rev *= self.inv_span
        x = np.clip(rev, 0.0, 63.0)

        # --- horizontal: 32 -> 64 ---------------------------------------
        # nxt[c] = x[c+1], except at the right edge of each row, where the
        # flat shift would otherwise pull in the next row's first column.
        nxt = self._next
        nxt[0:N_PIX - 1] = x[1:N_PIX]
        last = SRC_W - 1
        for r in range(SRC_H):
            i = r * SRC_W + last
            nxt[i] = x[i]

        # nxt becomes the column midpoints in place; x is still needed.
        nxt += x
        nxt *= 0.5

        h = self._h
        h[0::2] = x                      # co-located samples
        h[1::2] = nxt                    # midpoints between columns

        # --- shade the even rows ----------------------------------------
        # _m is not wanted until the midpoints are derived below, and _h is
        # consumed by that derivation, so no extra scratch is needed.
        out = self._out
        n = self._n
        for a in range(0, n, CHUNK):
            b = a + CHUNK
            out[a:b] = _shade(h[a:b])
        for k in range(self.SCALE):
            self._even16[k::self.SCALE] = out

        # --- vertical midpoints, one row at a time -----------------------
        # Doing this as two whole-array operations would again need 6 kB
        # blocks. A row is 64 floats, so these allocations are 256 bytes.
        rl = self._row_len
        m = self._m
        for r in range(SRC_H):
            a = r * rl
            dst = m[a:a + rl]
            dst[:] = h[a:a + rl]
            if r < SRC_H - 1:
                dst += h[a + rl:a + 2 * rl]
                dst *= 0.5

        # --- shade the odd rows -----------------------------------------
        for a in range(0, n, CHUNK):
            b = a + CHUNK
            out[a:b] = _shade(m[a:b])
        for k in range(self.SCALE):
            self._odd16[k::self.SCALE] = out

        self._window()
        self.disp.blit_repeat_rows(self._views, self.SCALE)
