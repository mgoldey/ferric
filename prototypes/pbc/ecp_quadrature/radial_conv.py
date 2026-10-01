"""Radial convergence of the per-(primitive pair, ECP term) Gauss-Legendre windows.

Reference = RAD_N 120, RAD_T 10 (vastly over-converged). Reports max |V(N, T) - V_ref| over every triple of the HI and
AuH fixtures (Cartesian values and raised-shell (derivative) blocks), absolute and relative to the block max."""

import sys

import numpy as np

import ecpq as E
import fixtures as F


def triples(with_raised=True):
    out = []
    bra = F.hi_shells(F.H0, F.I0)
    ket = F.hi_shells(F.H0 + F.LV, F.I0 + F.LV)
    cs = [
        F.I0,
        F.I0 + F.LV + np.array([0.3, 0.2, -0.25]),
        F.H0 + np.array([-0.5, 0.4, 0.6]),
        F.I0 + np.array([1.5, -1.2, 0.8]),
    ]
    out += [(sa, sb, F.ecp_i(c)) for sa in bra for sb in ket for c in cs]
    L = np.array([0.7, -1.1, 2.3])
    bra = F.auh_shells(F.AU0, F.H_AU)
    ket = F.auh_shells(F.AU0 + L, F.H_AU + L)
    cs = [
        F.AU0,
        F.AU0 + L + np.array([0.3, 0.2, -0.25]),
        F.H_AU + np.array([-0.5, 0.4, 0.6]),
    ]
    sel = [0, 5, 6, 9, 11, 12, 14]  # tight s, diffuse s, p, d, f, H s, H p
    out += [(bra[i], ket[j], F.ecp_au(c)) for i in sel for j in sel for c in cs]
    if with_raised:
        out += [(E._shifted_shell(sa, 1, True), sb, e) for sa, sb, e in out[::7]]
    return out


def run(configs):
    ts = triples()
    E.RAD_N, E.RAD_T = 120, 10.0
    ref = [E.triple_cart(sa, sb, e) for sa, sb, e in ts]
    print(f"{len(ts)} triples; reference N=120 T=10")
    for n, t in configs:
        E.RAD_N, E.RAD_T = n, t
        E.STATS.reset()
        ea = er = 0.0
        for (sa, sb, e), r in zip(ts, ref):
            v = E.triple_cart(sa, sb, e)
            d = np.abs(v - r).max()
            ea = max(ea, d)
            er = max(er, d / max(np.abs(r).max(), 1e-300))
        print(
            f"N {n:3d} T {t:4.1f}: max abs {ea:.2e}  max rel(block) {er:.2e}  windows {E.STATS.windows + E.STATS.t1_windows}",
            flush=True,
        )


if __name__ == "__main__":
    cfg = [(n, 7.0) for n in (12, 16, 20, 24, 28, 32, 40)] + [
        (32, t) for t in (5.0, 6.0, 8.0)
    ]
    if len(sys.argv) > 1:
        cfg = [
            tuple(float(x) if "." in x else int(x) for x in a.split(","))
            for a in sys.argv[1:]
        ]
    run(cfg)
