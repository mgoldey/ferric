"""ESTIMATE: pair-FT (prim pair, image, G) work, segmented columns (ferric today) vs one pass over
the UNIQUE primitives per (atom, l) with the columns contracted afterwards. Weighted by the
Cartesian block size ncart_a*ncart_b (the cost of one prim-pair G column). No ferric run."""

import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell_facts as cf  # noqa: E402


def evals(shells, lat, gcut, thresh=1e-15):
    g2 = cf.g_half_norms2(lat, gcut)
    gmax = math.sqrt(g2[-1])
    prims = [
        (s["c"], e, k, s["l"]) for s in shells for e, k in zip(s["e"], s["k"]) if k != 0
    ]
    C = np.array([p[0] for p in prims])
    E = np.array([p[1] for p in prims])
    Kc = np.array([p[2] for p in prims])
    Lq = np.array([p[3] for p in prims])
    nc = (Lq + 1) * (Lq + 2) // 2
    W = nc[:, None] * nc[None, :]
    amin = E.min()
    rpair = math.sqrt(2 * math.log(1e3 / thresh) / amin) + 2.0
    P = E[:, None] + E[None, :]
    CC = np.abs(Kc[:, None] * Kc[None, :]) * (math.pi / P) ** 1.5
    RED = E[:, None] * E[None, :] / P
    LL = (Lq[:, None] + Lq[None, :]) * math.log(max(gmax, 1.0))
    tot = 0.0
    for L in cf.lattice_points(lat, rpair):
        d = C[:, None, :] - (C[None, :, :] + L)
        r2 = np.einsum("ijk,ijk->ij", d, d)
        mag = CC * np.exp(-RED * r2)
        ok = mag >= thresh
        g2max = 4 * P[ok] * (np.log(mag[ok] / thresh) + 10.0 + LL[ok])
        tot += (np.searchsorted(g2, g2max, side="right") * W[ok]).sum()
    return tot


for cell in sys.argv[1:]:
    sym, xyz, lat = cf.read_cell(cell)
    seg = cf.ferric_shells("cc-pvdz", sym, xyz)
    uniq = []
    for s in (
        seg
    ):  # merge columns sharing (centre, l, exponents): keep max |coef| per primitive
        for u in uniq:
            if (
                u["l"] == s["l"]
                and np.allclose(u["c"], s["c"])
                and np.array_equal(u["e"], s["e"])
            ):
                u["k"] = np.where(np.abs(s["k"]) > np.abs(u["k"]), s["k"], u["k"])
                break
        else:
            uniq.append(dict(s, k=s["k"].copy()))
    gcut = 2 * math.sqrt(math.log(1e13))
    a, b = evals(seg, lat, gcut), evals(uniq, lat, gcut)
    print(f"{cell}: segmented {a:.3e}  unique-primitive {b:.3e}  ratio {a / b:.2f}")
