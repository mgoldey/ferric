"""PROTOTYPE (value-changing options for the pair-FT kernel; nothing here is in production).

Measures, against an 80-bit long-double reference on the SAME f64 inputs, the error of the
per-(primitive pair, G) factor  e^{-G^2/4p} e^{-i G.P}  (the pair FT's only transcendental
part; the Hermite polynomial and contraction multiply it unchanged) for:

  base   : today's kernel, f64 libm exp/cos/sin on ph = G.P       (the error floor)
  V1-dir : separable phase over the integer G grid, G = h b1 + k b2 + l b3:
           prod_d e^{-i h_d (b_d.P)}, each 1-D factor by its own sincos (table per site)
  V1-rec : as V1 but the 1-D tables by recurrence u^h = u^{h-1} u (1 sincos per axis)
  V2     : image split P = P0 + (b/p) L (P0 the home-cell pair centre); G.L = 2 pi m (m int)
           e^{-i G.P} = e^{-i G.P0} * T_slot[m],  T_slot[m] = e^{-2 pi i m b/p} (plan table)
  V5     : vectorisable sincos (3-part Cody-Waite reduction + Taylor deg 15/16 on [-pi/4, pi/4])
           replacing libm; same ph
  V4     : f32 sincos of the f64 phase reduced mod 2 pi in f64, then rounded to f32
  V6     : term re-association  cc * ((e (cos, -sin)) Fx Fy Fz)  with the bracket built ONCE per
           site and shared by its entries (today: (((cc e)(cos, -sin)) Fx) Fy) Fz per entry);
           measured on random unit-scale complex Fx, Fy, Fz: |term - ref| / |term|

Cell: dry ice (cubic a = 10.6278 Bohr), |G| <= 10.94 (the RS-GDF LR gcut), centres anywhere
in the cell, images |n_d| <= 3, exponents 0.1517 .. 11720 (cc-pVDZ C/O).

Usage: python phase_options.py   (a few seconds, one core)
"""

import math

import numpy as np

LD = np.longdouble
A_CELL = 10.627819724553907
GCUT = 10.94232264355655
EXPS = np.array(
    [11720.0, 1759.0, 400.8, 113.7, 37.03, 13.27, 5.025, 1.013, 0.3023, 0.2753, 0.1517]
)
RNG = np.random.default_rng(20260928)


def g_grid():
    b = 2 * math.pi / A_CELL
    hmax = int(GCUT / b) + 1
    r = np.arange(-hmax, hmax + 1)
    h, k, m3 = np.meshgrid(r, r, r, indexing="ij")
    hkl = np.stack([h.ravel(), k.ravel(), m3.ravel()], 1)
    g = hkl * b
    keep = np.einsum("ij,ij->i", g, g) <= GCUT**2
    return hkl[keep], g[keep], b


def ref_factor(g, pc, p):
    """Long-double e^{-G^2/4p} e^{-iG.P} for f64 inputs g (n,3), pc (3,), p."""
    gl, pl = g.astype(LD), pc.astype(LD)
    ph = gl @ pl
    e = np.exp(-(gl * gl).sum(1) / (LD(4) * LD(p)))
    return e * np.cos(ph), -e * np.sin(ph)


def base_factor(g, pc, p):
    ph = g[:, 0] * pc[0] + g[:, 1] * pc[1] + g[:, 2] * pc[2]
    e = np.exp(-(g * g).sum(1) / (4.0 * p))
    return e * np.cos(ph), -e * np.sin(ph)


def cmul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


def v1_factor(hkl, g, pc, p, b, recur):
    e = np.exp(-(g * g).sum(1) / (4.0 * p))
    hmax = np.abs(hkl).max()
    hs = np.arange(-hmax, hmax + 1)
    tr, ti = np.ones(len(g)), np.zeros(len(g))
    for d in range(3):
        x = b * pc[d]  # b_d . P for the cubic cell
        if recur:
            ur, ui = math.cos(x), -math.sin(x)
            pr, pi_ = np.ones(len(hs)), np.zeros(len(hs))
            cr, ci = 1.0, 0.0
            for h in range(1, hmax + 1):
                cr, ci = cr * ur - ci * ui, cr * ui + ci * ur
                pr[hmax + h], pi_[hmax + h] = cr, ci
                pr[hmax - h], pi_[hmax - h] = cr, -ci
        else:
            pr, pi_ = np.cos(hs * x), -np.sin(hs * x)
        idx = hkl[:, d] + hmax
        tr, ti = cmul(tr, ti, pr[idx], pi_[idx])
    return e * tr, e * ti


def v2_factor(hkl, g, pc0, n, p, bexp):
    e = np.exp(-(g * g).sum(1) / (4.0 * p))
    ph0 = g[:, 0] * pc0[0] + g[:, 1] * pc0[1] + g[:, 2] * pc0[2]
    m = hkl @ n
    tr, ti = np.cos(2 * math.pi * m * (bexp / p)), -np.sin(2 * math.pi * m * (bexp / p))
    fr, fi = cmul(np.cos(ph0), -np.sin(ph0), tr, ti)
    return e * fr, e * fi


# fdlibm's 33-bit-leading 3-part pi/2 (k * PIO2_1 and k * PIO2_2 exact for |k| < 2^20).
PIO2_1 = 1.57079632673412561417e00
PIO2_2 = 6.07710050630396597660e-11
PIO2_3 = 2.02226624871116645580e-21
PIO2_3T = 8.47842766036889956997e-32


def v5_sincos(x):
    k = np.rint(x * (2.0 / math.pi))
    r = ((x - k * PIO2_1) - k * PIO2_2) - k * (PIO2_3 + PIO2_3T)
    r2 = r * r
    s = r
    c = np.ones_like(r)
    ts, tc = r.copy(), np.ones_like(r)
    for i in range(1, 9):
        ts = ts * (-r2) / ((2 * i) * (2 * i + 1))
        tc = tc * (-r2) / ((2 * i - 1) * (2 * i))
        s, c = s + ts, c + tc
    q = k.astype(np.int64) & 3
    sin = np.select([q == 0, q == 1, q == 2, q == 3], [s, c, -s, -c])
    cos = np.select([q == 0, q == 1, q == 2, q == 3], [c, -s, -c, s])
    return sin, cos


def v5_factor(g, pc, p):
    ph = g[:, 0] * pc[0] + g[:, 1] * pc[1] + g[:, 2] * pc[2]
    e = np.exp(-(g * g).sum(1) / (4.0 * p))
    s, c = v5_sincos(ph)
    return e * c, -e * s


def v4_factor(g, pc, p):
    ph = g[:, 0] * pc[0] + g[:, 1] * pc[1] + g[:, 2] * pc[2]
    ph = np.remainder(ph, 2 * math.pi).astype(np.float32)
    e = np.exp(-(g * g).sum(1) / (4.0 * p))
    return e * np.cos(ph).astype(np.float64), -e * np.sin(ph).astype(np.float64)


def err(ref, got):
    rr, ri = ref
    gr, gi = got
    return float(np.max(np.hypot(LD(gr) - rr, LD(gi) - ri)))


def main():
    hkl, g, b = g_grid()
    print(
        f"G grid: {len(g)} G (full sphere), |h|max = {np.abs(hkl).max()}, |G|max = {GCUT:.2f}"
    )
    worst = {k: 0.0 for k in ("base", "V1-dir", "V1-rec", "V2", "V5", "V4")}
    for _ in range(40):
        a, bexp = RNG.choice(EXPS, 2)
        p = a + bexp
        ca, cb = RNG.uniform(0, A_CELL, 3), RNG.uniform(0, A_CELL, 3)
        n = RNG.integers(-3, 4, 3)
        lvec = n * A_CELL
        bc = cb + lvec
        pc = (a * ca + bexp * bc) / p
        pc0 = (a * ca + bexp * cb) / p
        ref = ref_factor(g, pc, p)
        worst["base"] = max(worst["base"], err(ref, base_factor(g, pc, p)))
        worst["V1-dir"] = max(
            worst["V1-dir"], err(ref, v1_factor(hkl, g, pc, p, b, False))
        )
        worst["V1-rec"] = max(
            worst["V1-rec"], err(ref, v1_factor(hkl, g, pc, p, b, True))
        )
        worst["V2"] = max(worst["V2"], err(ref, v2_factor(hkl, g, pc0, n, p, bexp)))
        worst["V5"] = max(worst["V5"], err(ref, v5_factor(g, pc, p)))
        worst["V4"] = max(worst["V4"], err(ref, v4_factor(g, pc, p)))
    print(
        "max |factor - reference| over 40 random (prim pair, image) x all G (factor <= 1):"
    )
    for k, v in worst.items():
        print(f"  {k:7s} {v:.2e}")
    # V5 alone: absolute error of sin/cos (in units of 2^-53) over the kernel's phase range.
    x = RNG.uniform(-450, 450, 400_000)
    s, c = v5_sincos(x)
    xs = x.astype(LD)
    ulp1 = 2.0**-53
    e5 = max(np.abs(LD(s) - np.sin(xs)).max(), np.abs(LD(c) - np.cos(xs)).max()) / ulp1
    el = (
        max(
            np.abs(LD(np.sin(x)) - np.sin(xs)).max(),
            np.abs(LD(np.cos(x)) - np.cos(xs)).max(),
        )
        / ulp1
    )
    print(
        f"V5 sincos |x| <= 450: max abs err {float(e5):.2f} x 2^-53 (libm {float(el):.2f} x 2^-53)"
    )
    # V6: per-term relative error of both association orders vs long double.
    nt = 200_000
    cc = RNG.uniform(-3, 3, nt)
    e = np.exp(-RNG.uniform(0, 40, nt))
    ph = RNG.uniform(-450, 450, nt)
    f = [RNG.normal(size=nt) + 1j * RNG.normal(size=nt) for _ in range(3)]
    com = (cc * e) * np.cos(ph) - 1j * ((cc * e) * np.sin(ph))
    today = ((com * f[0]) * f[1]) * f[2]
    unit = e * np.cos(ph) - 1j * (e * np.sin(ph))
    v6 = cc * (((unit * f[0]) * f[1]) * f[2])
    exact_re, exact_im = _ld_term(cc, e, ph, f)
    for name, z in (("today", today), ("V6", v6)):
        d = np.hypot(LD(z.real) - exact_re, LD(z.imag) - exact_im) / np.hypot(
            exact_re, exact_im
        )
        print(
            f"{name:5s} term rel err: max {float(d.max()):.2e}  mean {float(d.mean()):.2e}"
        )


def _ld_term(cc, e, ph, f):
    """Long-double cc e (cos ph, -sin ph) fx fy fz (real/imag parts)."""
    m = LD(cc) * LD(e)
    ph = LD(ph)
    zr, zi = m * np.cos(ph), -m * np.sin(ph)
    for fk in f:
        ar, ai = LD(fk.real), LD(fk.imag)
        zr, zi = zr * ar - zi * ai, zr * ai + zi * ar
    return zr, zi


if __name__ == "__main__":
    main()
