"""ESTIMATE: hcore SR Gaussian-nucleus triplets and V_LR G count vs the hcore Ewald split w_h
(cell_facts.hcore_predict's rule with w as a parameter). No ferric run."""

import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell_facts as cf  # noqa: E402


def run(cell, ws):
    sym, xyz, lat = cf.read_cell(cell)
    obs = cf.ferric_shells("cc-pvdz", sym, xyz)
    vol = abs(np.linalg.det(lat))
    prec = float(os.environ.get("HPREC", "1e-14"))
    amin = min(s["amin"] for s in obs)
    pmax_all = 2 * max(s["amax"] for s in obs)
    rpair = math.sqrt(2 * math.log(1e3 / (0.1 * prec)) / amin) + 2.0
    images = cf.lattice_points(lat, rpair)
    zmax = max(cf.Z[s] for s in sym)
    K, C, pmin, pmax = cf.pair_tables(obs)
    A = np.array([s["c"] for s in obs])
    for w in ws:
        wp0 = w * np.sqrt(pmin / (pmin + w * w))
        n3 = 0.0
        for L in images:
            B = A + L
            d = A[:, None, :] - B[None, :, :]
            r2 = np.einsum("ijk,ijk->ij", d, d)
            q = np.max(C * np.exp(-K * r2[:, :, None]), axis=2)
            pref = q * zmax * (1 + 2 * np.sqrt(pmax / math.pi))
            ok = pref > prec
            rad = np.sqrt(np.log(np.maximum(pref, prec) / prec)) / wp0 + 2.0
            seg = np.sqrt(r2)
            n3 += np.sum(
                ((math.pi * rad**2 * seg + 4.0 / 3.0 * math.pi * rad**3) / vol)[ok]
            ) * len(sym)
        gcut = min(2 * w, 2 * math.sqrt(pmax_all)) * math.sqrt(math.log(1 / prec))
        print(
            f"{cell} w_h={w:.3f} sr_nucleus_triplets={n3:.3e} n_g_half(hcore)={cf.count_g_half(lat, gcut)} gcut={gcut:.2f}",
            flush=True,
        )


for c in sys.argv[1:]:
    sym, xyz, lat = cf.read_cell(c)
    w0 = math.sqrt(math.pi) / abs(np.linalg.det(lat)) ** (1 / 3)
    run(c, [w0, 0.5, 0.96, 1.5])
