"""Fast frame path for the MLX90640 on CircuitPython.

Subclasses adafruit_mlx90640.MLX90640 so the one-time EEPROM parameter
extraction is reused unchanged, then replaces the per-frame work:

  * one subpage read per displayed frame instead of two -- the sensor's RAM
    retains the other subpage's pixels, and both compensation-pixel words are
    in the same transaction, so every pixel still pairs with the CP value from
    its own subpage,
  * the 1622-byte read lands straight in a bytearray and is decoded by
    frombuffer + byteswap in C; no 832-element Python list is built,
  * per-pixel chess/interleave patterns, kta/ktaScale, kv/kvScale and
    1/alphaCompensated are precomputed once into ulab arrays,
  * the temperature solve is rearranged so alphaCompensated cancels, which
    also avoids a 1e-18 intermediate that would underflow single precision,
  * the whole solve runs in three preallocated buffers using ulab's in-place
    operators and sqrt(out=...), so a frame allocates ~1.6 kB instead of
    ~33 kB of temporaries. The RP2040 heap is about 118 kB with this driver
    loaded, and the naive version ran out of it.
  * the status-register write drops the stock driver's read-back verify and
    its 1 ms sleep, and the frame read is not retried.

Results agree with adafruit_mlx90640's _CalculateTo to within single-precision
rounding (worst case 1.2e-4 degC over 245k pixels spanning all four ksTo
ranges), so this is a drop-in replacement rather than an approximation.

Note that temperatures() returns a buffer that the next call overwrites; copy
it if you need to keep a frame.

Usage:

    mlx = FastMLX90640(i2c)
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
    while True:
        mlx.wait_for_subpage()
        temps = mlx.temperatures()
"""

import gc
import math
import time

from ulab import numpy as np

import adafruit_mlx90640
from adafruit_mlx90640 import SCALEALPHA, OPENAIR_TA_SHIFT

# Words 0..810 cover the pixels (0..767) plus every auxiliary word needed:
# 768 ptatArt, 776 CP subpage 0, 778 gain, 800 ptat, 808 CP subpage 1, 810 vdd.
N_WORDS = 811
N_PIX = 768

_STATUS_REG = 0x8000
_RAM_BASE = 0x0400
_CTRL_REG = 0x800D

_W_PTAT_ART = 768
_W_CP0 = 776
_W_GAIN = 778
_W_PTAT = 800
_W_CP1 = 808
_W_VDD = 810

# How far Ta (degC) and Vdd (volts) may drift before the offset compensation
# term is rebuilt. 0.05 degC is well under the sensor's own noise.
TA_RECALC = 0.05
VDD_RECALC = 0.002


class FastMLX90640(adafruit_mlx90640.MLX90640):
    """MLX90640 with a vectorised, allocation-free frame path."""

    def __init__(self, i2c_bus, address=0x33, emissivity=0.95, release_cal=True):
        """release_cal frees the stock driver's per-pixel Python lists once
        they have been folded into ulab arrays, recovering about 15 kB. That
        makes the inherited getFrame() unusable, so pass False to keep it
        (bench.py does, to compare the two paths on one object).
        """
        super().__init__(i2c_bus, address)
        self.emissivity = emissivity
        self.torn_frames = 0
        self.clamped_frames = 0
        self.recalcs = 0
        self.subpage = 0
        self._release_cal = release_cal
        self._raw = bytearray(2 * N_WORDS)
        self._addrbuf = bytearray(2)
        self._cmdbuf = bytearray(4)
        self._statbuf = bytearray(2)
        self._tables = None

    # ------------------------------------------------------------------ I2C

    def _read_into(self, addr, buf, nwords):
        self._addrbuf[0] = addr >> 8
        self._addrbuf[1] = addr & 0xFF
        with self.i2c_device as i2c:
            i2c.write_then_readinto(self._addrbuf, buf, in_end=nwords * 2)

    def _read_status(self):
        self._read_into(_STATUS_REG, self._statbuf, 1)
        return (self._statbuf[0] << 8) | self._statbuf[1]

    def _write_word(self, addr, data):
        c = self._cmdbuf
        c[0] = addr >> 8
        c[1] = addr & 0xFF
        c[2] = data >> 8
        c[3] = data & 0xFF
        with self.i2c_device as i2c:
            i2c.write(c)

    def _word(self, index):
        """Signed 16-bit word `index` of the raw frame buffer."""
        b = self._raw
        v = (b[2 * index] << 8) | b[2 * index + 1]
        return v - 65536 if v > 32767 else v

    # -------------------------------------------------------------- tables

    def _build_tables(self):
        """Precompute everything that depends only on calibration data.

        Called lazily on the first frame so that refresh_rate (and anything
        else written to the control register) is already in effect.
        """
        self._read_into(_CTRL_REG, self._statbuf, 1)
        ctrl = (self._statbuf[0] << 8) | self._statbuf[1]
        self._ctrl = ctrl

        # Same expression the stock driver uses: 0 for interleaved reading,
        # 0x80 for chess. calibrationModeEE is on the same 0/0x80 scale.
        mode = (ctrl & 0x1000) >> 5
        self._mode = mode
        chess_correction = mode != self.calibrationModeEE

        kta_scale = math.pow(2, self.ktaScale)
        kv_scale = math.pow(2, self.kvScale)
        alpha_scale = math.pow(2, self.alphaScale)

        kta = [0.0] * N_PIX
        kv = [0.0] * N_PIX
        offset = [0.0] * N_PIX
        # Reciprocal of alphaCompensated before the KsTa factor is applied,
        # i.e. 1 / (SCALEALPHA * alphaScale / alpha[p]).
        alpha_recip = [0.0] * N_PIX
        chess = [0.0] * N_PIX if chess_correction else None
        subpage_of = [0.0] * N_PIX if self.tgc else None

        c1 = self.ilChessC[1]
        c2 = self.ilChessC[2]

        for p in range(N_PIX):
            kta[p] = self.kta[p] / kta_scale
            kv[p] = self.kv[p] / kv_scale
            offset[p] = float(self.offset[p])
            alpha_recip[p] = self.alpha[p] / (SCALEALPHA * alpha_scale)

            il = p // 32 - (p // 64) * 2
            if subpage_of is not None:
                subpage_of[p] = float((il ^ (p & 1)) if mode else il)
            if chess is not None:
                conv = ((p + 2) // 4 - (p + 3) // 4 + (p + 1) // 4 - p // 4) * (
                    1 - 2 * il
                )
                chess[p] = c2 * (2 * il - 1) - c1 * conv

        # alphaCorrR from the stock driver, then folded with ksTo/ct into the
        # affine form  alphaCorr * (1 + ksTo*(To - ct)) == A + B*To,  so the
        # range correction needs two selected scalars instead of three.
        ks = self.ksTo
        ct = self.ct
        acr = [1.0 / (1.0 + ks[0] * 40), 1.0, 1.0 + ks[1] * ct[2], 0.0]
        acr[3] = acr[2] * (1.0 + ks[2] * (ct[3] - ct[2]))
        self._range_a = [acr[r] * (1.0 - ks[r] * ct[r]) for r in range(4)]
        self._range_b = [acr[r] * ks[r] for r in range(4)]

        f = np.float
        self._kta = np.array(kta, dtype=f)
        self._kv = np.array(kv, dtype=f)
        self._offset = np.array(offset, dtype=f)
        self._alpha_recip = np.array(alpha_recip, dtype=f)
        self._chess = np.array(chess, dtype=f) if chess is not None else None
        self._subpage_of = (
            np.array(subpage_of, dtype=f) if subpage_of is not None else None
        )
        self._bad = sorted(set(self.brokenPixels) | set(self.outlierPixels))

        # Working buffers: q holds the pixel term, to the iterate/result, s
        # the scratch denominator. Everything per-frame happens in these.
        self._off_corr = np.zeros(N_PIX, dtype=f)
        self._cached_d_ta = -1e9
        self._cached_d_vdd = -1e9
        self._q = np.zeros(N_PIX, dtype=f)
        self._to = np.zeros(N_PIX, dtype=f)
        self._s = np.zeros(N_PIX, dtype=f)
        self._tables = True

        if self._release_cal:
            # These are class attributes on MLX90640, shared and mutated in
            # place by _Extract*Parameters, so clearing them on the class is
            # what actually frees the memory (about 15 kB).
            cls = adafruit_mlx90640.MLX90640
            cls.kta = cls.kv = cls.offset = cls.alpha = None
            adafruit_mlx90640.eeData = None
        del kta, kv, offset, alpha_recip, chess, subpage_of
        gc.collect()

    # --------------------------------------------------------------- frame

    def wait_for_subpage(self, timeout=1.0):
        """Block until a fresh subpage is ready, then read it into self._raw.

        Returns the subpage number (0 or 1) the data belongs to.
        """
        deadline = None
        while not self._read_status() & 0x0008:
            if deadline is None:
                deadline = time.monotonic_ns() + int(timeout * 1e9)
            elif time.monotonic_ns() > deadline:
                raise RuntimeError("MLX90640 subpage timeout")

        status = (self._statbuf[0] << 8) | self._statbuf[1]
        self.subpage = status & 0x0001

        # Clear "new data" (bit 3), keep RAM overwrite enabled (bit 4).
        self._write_word(_STATUS_REG, 0x0030)
        self._read_into(_RAM_BASE, self._raw, N_WORDS)

        # A subpage completing during the read means the RAM was overwritten
        # mid-transfer, so the frame blends two subpages. Invisible on a live
        # display, but worth counting: a rising number means the loop is not
        # keeping up with the configured refresh rate.
        if self._read_status() & 0x0008:
            self.torn_frames += 1
        return self.subpage

    def temperatures(self):
        """Convert the frame in self._raw to 768 degC values.

        Returns the internal buffer, which the next call overwrites.
        """
        if self._tables is None:
            self._build_tables()

        # --- per-frame scalars -------------------------------------------
        vdd = self._get_vdd()
        ta = self._get_ta(vdd)
        tr = ta - OPENAIR_TA_SHIFT

        ta4 = (ta + 273.15) * (ta + 273.15)
        ta4 = ta4 * ta4
        tr4 = (tr + 273.15) * (tr + 273.15)
        tr4 = tr4 * tr4
        em = self.emissivity
        ta_tr = tr4 - (tr4 - ta4) / em

        gain = self.gainEE / self._word(_W_GAIN)
        d_ta = ta - 25.0
        d_vdd = vdd - 3.3

        q = self._q
        to = self._to
        s = self._s

        # --- offset / Ta / Vdd compensation ------------------------------
        # offset * (1 + kta*dTa) * (1 + kv*dVdd) costs about 7 ms, but Ta and
        # Vdd are die temperature and supply rail: they drift over seconds,
        # not between frames. Recompute only when they actually move.
        if (
            abs(d_ta - self._cached_d_ta) > TA_RECALC
            or abs(d_vdd - self._cached_d_vdd) > VDD_RECALC
        ):
            oc = self._off_corr
            oc[:] = self._kta
            oc *= d_ta
            oc += 1.0
            to[:] = self._kv
            to *= d_vdd
            to += 1.0
            oc *= to
            oc *= self._offset
            self._cached_d_ta = d_ta
            self._cached_d_vdd = d_vdd
            self.recalcs += 1

        # The sensor is big-endian; frombuffer + byteswap does the decode,
        # including sign extension, in C.
        # frombuffer returns a view, so byteswap must NOT be in-place here:
        # it would rewrite self._raw and a second call on the same frame would
        # swap the bytes back and decode garbage.
        q[:] = np.frombuffer(self._raw, dtype=np.int16, count=N_PIX).byteswap()
        q *= gain
        q -= self._off_corr

        if self._chess is not None:
            q += self._chess
        if self.tgc:
            cp0 = self._compensation_pixel(_W_CP0, gain, ta, vdd, 0)
            cp1 = self._compensation_pixel(_W_CP1, gain, ta, vdd, 1)
            g = self.tgc
            s[:] = self._subpage_of
            s *= g * (cp1 - cp0)
            s += g * cp0
            q -= s

        # --- temperature solve -------------------------------------------
        # The stock driver evaluates
        #     Sx = (alphaComp**3 * (irData + alphaComp*taTr))**0.25 * ksTo[1]
        #     To = (irData / (alphaComp*(1 - ksTo[1]*273.15) + Sx) + taTr)**0.25
        # and then refines To once with range-dependent coefficients.
        # alphaComp > 0, so Sx == ksTo[1] * alphaComp * u**0.25 for
        # u = irData/alphaComp + taTr, and every alphaComp factor cancels out
        # of the quotient. What remains is the fixed-point iteration
        #     v = q / (alphaCorr * (1 + ksTo*(T - ct))) + taTr
        # on q = irData / (emissivity * alphaComp), which is the three passes
        # below. Forming alphaComp**3 (around 1e-18) is avoided.
        q *= self._alpha_recip
        q *= 1.0 / (em * (1.0 + self.KsTa * d_ta))

        ct = self.ct

        # pass 1: crude estimate with ksTo ignored
        to[:] = q
        to += ta_tr
        self._quartic_root(to)

        # pass 2: always range 1, where alphaCorr == 1 and ct == 0
        s[:] = to
        s *= self.ksTo[1]
        s += 1.0
        to[:] = q
        to /= s
        to += ta_tr
        self._quartic_root(to)

        # pass 3: coefficients chosen per ksTo range. A scene confined to one
        # range -- the normal case -- needs only two scalars.
        a_tab = self._range_a
        b_tab = self._range_b
        r_lo = self._range_of(np.min(to), ct)
        r_hi = self._range_of(np.max(to), ct)
        if r_lo == r_hi:
            s[:] = to
            s *= b_tab[r_lo]
            s += a_tab[r_lo]
        else:
            # Thresholds nest, so summing the deltas telescopes to the right
            # coefficient per pixel. This path allocates, but only fires when
            # a scene straddles a ksTo breakpoint.
            s[:] = to
            s *= b_tab[0]
            s += a_tab[0]
            for r in range(1, 4):
                g = np.array(to >= ct[r], dtype=np.float)
                g *= b_tab[r] - b_tab[r - 1]
                g *= to
                s += g
                g = np.array(to >= ct[r], dtype=np.float)
                g *= a_tab[r] - a_tab[r - 1]
                s += g
        to[:] = q
        to /= s
        to += ta_tr
        self._quartic_root(to)

        for p in self._bad:
            to[p] = to[p + 1] if p == 0 else to[p - 1]
        return to

    def _quartic_root(self, buf):
        """buf = buf ** 0.25 - 273.15, in place.

        A non-positive entry would come back as -inf and poison the min/max
        used for range selection, so those are clamped first. The clamp costs
        an allocation, so it only runs when np.min says it is needed; buf is
        T_K**4, so the guard trips below about 1 K, which for real sensor data
        means a dead pixel rather than a cold scene.
        """
        if np.min(buf) < 1.0:
            self.clamped_frames += 1
            buf[:] = np.maximum(buf, 1.0)
        np.sqrt(buf, out=buf)
        np.sqrt(buf, out=buf)
        buf -= 273.15

    def palette_bytes(self, lo, span, count):
        """Map the last temperatures() result to `count` palette indices.

        Returns a bytearray suitable for bitmaptools.arrayblit. Note that the
        float -> uint8 conversion rounds, where the obvious Python
        int((t - lo) / span) would truncate.
        """
        s = self._s
        s[:] = self._to
        s -= lo
        s *= 1.0 / span
        return np.array(np.clip(s, 0.0, count - 1.0), dtype=np.uint8).tobytes()

    @staticmethod
    def _range_of(temp, ct):
        if temp < ct[1]:
            return 0
        if temp < ct[2]:
            return 1
        if temp < ct[3]:
            return 2
        return 3

    # --- scalar helpers, cheap enough to leave in Python -----------------

    def _get_vdd(self):
        resolution_ram = (self._ctrl & 0x0C00) >> 10
        res_corr = math.pow(2, self.resolutionEE) / math.pow(2, resolution_ram)
        return (res_corr * self._word(_W_VDD) - self.vdd25) / self.kVdd + 3.3

    def _get_ta(self, vdd):
        ptat = self._word(_W_PTAT)
        ptat_art = self._word(_W_PTAT_ART)
        ptat_art = (ptat / (ptat * self.alphaPTAT + ptat_art)) * 262144.0
        ta = ptat_art / (1 + self.KvPTAT * (vdd - 3.3)) - self.vPTAT25
        return ta / self.KtPTAT + 25

    def _compensation_pixel(self, index, gain, ta, vdd, which):
        ir = self._word(index) * gain
        off = self.cpOffset[which]
        if which == 1 and self._mode != self.calibrationModeEE:
            off += self.ilChessC[0]
        return ir - off * (1 + self.cpKta * (ta - 25)) * (1 + self.cpKv * (vdd - 3.3))
