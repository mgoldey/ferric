"""MODEL: 1-thread CPU of ferric's hcore SR + LR nuclear attraction vs the hcore split w_h.

Counts re-implement hcore.rs's truncation (the perf_hcore_omega.py / cell_facts.py rules):
SR Gaussian-nucleus triplets (capsule estimator), SR segment tests (surviving pair images x nucleus
candidates), LR pair-FT evaluations and G chunks (cell_facts.lr_pairft_predict at the hcore gcut
min(2w, 2 sqrt(p_max)) sqrt(ln 1/prec), pair threshold prec/10). No ferric run.

Unit costs fitted on the measured quiet-box runs (reference/pbc/FINDINGS.md, "Benchmark series",
2026-09-28, cc-pVDZ, 1 thread):
  diamond_prim w=0.4174: hcore SR 56.073 s (28882956 triplets, 740937384 segment tests),
                         LR 0.765 s (68 half-G)
  diamond_prim w=0.96  : hcore SR 26.543 s (12762384 triplets, 506943624 segment tests),
                         LR 4.653 s (843 half-G)
  -> 1.689e-6 s/triplet + 9.84e-9 s/segment test (2x2 solve); 43.9 ns per pair-FT evaluation
     + 51 ns per (primitive pair x image) re-walk test per chunk (0.247 s intercept / 4.85e6 tests).
  Check (not fitted): dry ice at w=0.1668 models SR 157 s / LR 1.3 s vs measured 168.1 / 1.85 s
  (the estimator undercounts dry-ice segment tests by 36%).

Usage: python perf_hcore_omega_cost.py <cell> <c...>   (w = c * sqrt(pi)/V^(1/3); < 1 min per cell)
"""

import math
import sys

import numpy as np

sys.path.insert(0, "/home/matt/qc/ferric-pbc/reference/pbc/bench")
import cell_facts as cf  # noqa: E402

PREC = 1e-14
T_TRIPLET, T_SEGMENT, T_EVAL, T_REWALK = 1.689e-6, 9.84e-9, 43.9e-9, 51e-9


def counts(cell, w):
    sym, xyz, lat = cf.read_cell(cell)
    obs = cf.ferric_shells("cc-pvdz", sym, xyz)
    nao = sum(cf.nfun(s) for s in obs)
    vol = abs(np.linalg.det(lat))
    amin = min(s["amin"] for s in obs)
    pmax_all = 2 * max(s["amax"] for s in obs)
    rpair = math.sqrt(2 * math.log(1e3 / (0.1 * PREC)) / amin) + 2.0
    images = cf.lattice_points(lat, rpair)
    zmax = max(cf.Z[s] for s in sym)
    K, C, pmin, pmax = cf.pair_tables(obs)
    A = np.array([s["c"] for s in obs])
    pref0 = np.max(C, axis=2) * zmax * (1 + 2 * np.sqrt(pmax / math.pi))
    wp0 = w * np.sqrt(pmin / (pmin + w * w))
    rnm = np.nanmax(
        np.where(
            pref0 > PREC,
            np.sqrt(np.log(np.maximum(pref0, PREC) / PREC)) / wp0 + 2.0,
            np.nan,
        )
    )
    ncand = len(cf.lattice_points(lat, rnm + rpair)) * len(sym)
    n3, nlab = 0.0, 0
    for L in images:
        d = A[:, None, :] - (A + L)[None, :, :]
        r2 = np.einsum("ijk,ijk->ij", d, d)
        q = np.max(C * np.exp(-K * r2[:, :, None]), axis=2)
        pref = q * zmax * (1 + 2 * np.sqrt(pmax / math.pi))
        ok = pref > PREC
        rad = np.sqrt(np.log(np.maximum(pref, PREC) / PREC)) / wp0 + 2.0
        seg = np.sqrt(r2)
        n3 += np.sum(
            ((math.pi * rad**2 * seg + 4.0 / 3.0 * math.pi * rad**3) / vol)[ok]
        ) * len(sym)
        nlab += int(ok.sum())
    gcut = min(2 * w, 2 * math.sqrt(pmax_all)) * math.sqrt(math.log(1 / PREC))
    ng = cf.count_g_half(lat, gcut)
    lr = cf.lr_pairft_predict(obs, lat, nao, 0, gcut, thresh=0.1 * PREC)
    return dict(
        tri=n3,
        seg=nlab * ncand,
        ng=ng,
        ev=lr["prim_pair_G_evals"],
        ch=lr["n_chunks"],
        rewalk=lr["per_chunk_prim_pair_image_tests"],
    )


def main():
    cell = sys.argv[1]
    lat = cf.read_cell(cell)[2]
    w_ewald = math.sqrt(math.pi) / abs(np.linalg.det(lat)) ** (1 / 3)
    for c in [float(x) for x in sys.argv[2:]]:
        w = c * w_ewald
        k = counts(cell, w)
        sr = T_TRIPLET * k["tri"] + T_SEGMENT * k["seg"]
        lr = T_EVAL * k["ev"] + T_REWALK * k["rewalk"] * k["ch"]
        print(
            f"{cell} c={c:.3f} w={w:.4f} triplets={k['tri']:.4e} nG_half={k['ng']} "
            f"SR={sr:8.2f}s LR={lr:8.2f}s total={sr + lr:8.2f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
