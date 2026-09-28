"""Per-window radial-scheme study (the design choice of Iteration 25).

Every type-2 window is  R[K, lam, lam'] = c int_0^inf r^(n+K) exp(-p (r-r0)^2 - K0) ktil_lam(2 al A r) ktil_lam'(2 be B r) dr.
Schemes compared against an over-converged reference (GL 200 nodes on T = 10):
  GL(N, T): Gauss-Legendre on [max(0, r0 - T/sqrt p), rpk + T/sqrt p]
  HYB(NH, NL, tau): Gauss-Hermite NH nodes about r0 (scale 1/sqrt p) when t0 = r0 sqrt(p) >= tau (every node at r > 0
                    and the neglected r < 0 mass is <= exp(-tau^2)); otherwise GL NL nodes on [0, rpk + 6/sqrt p].
Windows come from the HI + AuH fixtures (radial_conv.triples) and, if present, windows.txt (triiodobenzene, cost.py).
  python radial_scheme.py"""

import math
import os
import sys

import numpy as np

import ecpq as E


def win_R(w, nodes):
    la, lb, l, al, A, be, B, n, zeta, c = w
    p = al + be + zeta
    r0 = (al * A + be * B) / p
    K0 = al * A * A + be * B * B - p * r0 * r0
    r, wt = nodes(p, r0, n + la + lb)
    ka = E.bessel_ktil(l + la, 2 * al * A * r)
    kb = E.bessel_ktil(l + lb, 2 * be * B * r)
    g = c * 16 * math.pi**2 * wt * r**n * np.exp(-p * (r - r0) ** 2 - K0)
    return np.einsum(
        "Kr,lr,qr,r->Klq", r[None, :] ** np.arange(la + lb + 1)[:, None], ka, kb, g
    )


def gl(N, T):
    def f(p, r0, deg):
        lo, hi = E.radial_window(p, r0, deg, T)
        return E.gl_nodes(lo, hi, N)

    return f


def hyb(NH, NL, tau, TL=6.0, per_sigma=None, nmin=16):
    """NL = max GL nodes; per_sigma: if set, GL nodes = clamp(ceil(per_sigma * (hi - lo) sqrt p), nmin, NL)."""
    xh, wh = np.polynomial.hermite.hermgauss(NH)

    def f(p, r0, deg):
        sp = math.sqrt(p)
        if r0 * sp >= tau:
            r = r0 + xh / sp
            return r, wh * np.exp(
                xh * xh
            ) / sp  # weights for the full integrand (envelope kept explicit)
        lo, hi = E.radial_window(p, r0, deg, TL)
        n = (
            NL
            if per_sigma is None
            else min(NL, max(nmin, math.ceil(per_sigma * (hi - lo) * sp)))
        )
        return E.gl_nodes(lo, hi, n)

    return f


def load_windows():
    ws = []
    import radial_conv as RC

    E.WINDOW_LOG = []
    for sa, sb, e in RC.triples():
        E.triple_cart(sa, sb, e, channels="semi")
    ws += [("fixtures", w) for w in E.WINDOW_LOG]
    E.WINDOW_LOG = None
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "windows.txt")
    if os.path.exists(fn):
        rows = [l.split() for l in open(fn)]
        tri = [
            (
                int(r[0]),
                int(r[1]),
                int(r[2]),
                float(r[3]),
                float(r[4]),
                float(r[5]),
                float(r[6]),
                int(r[7]),
                float(r[8]),
                float(r[9]),
            )
            for r in rows
        ]
        ws += [
            ("triiodobenzene", w)
            for w in tri[:: int(sys.argv[1]) if len(sys.argv) > 1 else 7]
        ]
    return ws


def main():
    ws = load_windows()
    ref = [win_R(w, gl(200, 10.0)) for _, w in ws]
    t0 = np.array(
        [
            ((w[3] * w[4] + w[5] * w[6]) / (w[3] + w[5] + w[8]))
            * math.sqrt(w[3] + w[5] + w[8])
            for _, w in ws
        ]
    )
    for src in sorted(set(s for s, _ in ws)):
        idx = [i for i, (s, _) in enumerate(ws) if s == src]
        print(
            f"{src}: {len(idx)} windows; t0 = r0 sqrt(p): fraction >= 6.5: {np.mean(t0[idx] >= 6.5):.2f}, "
            f">= 8: {np.mean(t0[idx] >= 8):.2f}"
        )
    if os.environ.get("SCHEMES") == "adaptive":
        schemes = [
            (f"HYB(20, 40, 6.5) GL {ps}/sigma", hyb(20, 40, 6.5, per_sigma=ps))
            for ps in (2.6, 3.0, 3.4, 3.8)
        ]
        schemes += [
            ("HYB(20, 44, 6.5) GL 3.4/sigma", hyb(20, 44, 6.5, per_sigma=3.4)),
            ("HYB(16, 40, 6.5) GL 3.4/sigma", hyb(16, 40, 6.5, per_sigma=3.4)),
        ]
    else:
        schemes = [
            ("GL(32, 6)", gl(32, 6.0)),
            ("GL(40, 6)", gl(40, 6.0)),
            ("GL(48, 6)", gl(48, 6.0)),
            ("HYB(16, 40, 6.5)", hyb(16, 40, 6.5)),
            ("HYB(20, 40, 6.5)", hyb(20, 40, 6.5)),
            ("HYB(24, 40, 7.0)", hyb(24, 40, 7.0)),
            ("HYB(20, 48, 6.5)", hyb(20, 48, 6.5)),
            ("HYB(24, 48, 7.0)", hyb(24, 48, 7.0)),
        ]
    for name, sch in schemes:
        ea = er = 0.0
        eh = el = 0.0
        nodes = 0
        for (s, w), r, t in zip(ws, ref, t0):
            v = win_R(w, sch)
            d = np.abs(v - r).max()
            ea = max(ea, d)
            er = max(
                er, d / max(np.abs(r).max(), 1e-300) if np.abs(r).max() > 1e-12 else 0.0
            )
            if name.startswith("HYB"):
                rel = d / np.abs(r).max() if np.abs(r).max() > 1e-12 else 0.0
            if name.startswith("HYB"):
                if t >= float(name.split(",")[2].split(")")[0]):
                    eh = max(eh, rel)
                else:
                    el = max(el, rel)
            nodes += len(
                sch(
                    w[3] + w[5] + w[8],
                    (w[3] * w[4] + w[5] * w[6]) / (w[3] + w[5] + w[8]),
                    w[7] + w[0] + w[1],
                )[0]
            )
        extra = (
            f"  (max rel: GH part {eh:.1e}, GL part {el:.1e})"
            if name.startswith("HYB")
            else ""
        )
        print(
            f"{name:18s}: max abs {ea:.2e}, max rel (|R| > 1e-12) {er:.2e}, mean nodes/window {nodes / len(ws):.1f}{extra}",
            flush=True,
        )


if __name__ == "__main__":
    main()
