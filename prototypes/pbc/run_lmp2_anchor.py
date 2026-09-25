"""Stage 8b exactness anchor + mutations (Gamma LMP2, pbc_lmp2.py).

Anchor: eps=0 (full mask) localised-basis masked CG == canonical Gamma MP2 (shifted denominators,
same periodic RS-GDF B).  Independent constructions sharing only B: canonical orbitals +
closed-form denominators vs Berghold LMOs + periodic VV-HV virtuals + CG.

Mutations (each must FAIL its anchor; the last is a pinned blind spot):
  M1 drop one hard virtual                                  -> eps=0 anchor fails
  M2 occupied pair cutoff at its TRIVIAL radius R* = max minimum-image centroid distance,
     but distances taken as raw Cartesian (non-minimum-image) -> pairs across the boundary
     are dropped -> anchor fails (min-image version must pass exactly)
  M3 per-pair periodic domain fit at its trivial radius (max min-image occ-aux distance),
     raw distances -> aux dropped -> max|J_dom - J_glob| > 0 (min-image version == global)
  M4 molecular (L=0, non-periodic) Boys position operator: eps=0 anchor still PASSES
     (unitary invariance: the anchor cannot see localisation) -- the translation-equivalence
     check is what catches it (run_lmp2_translation.py).
Usage: OPENBLAS_NUM_THREADS=1 scripts/ferric-limited -- python3 reference/pbc/run_lmp2_anchor.py
"""

import sys
import time

import numpy as np

import pbc_lmp2 as L
from pbc_gdf import ferric_basis
from pbc_supercell import Supercell, build_supercell

A0 = np.eye(3) * 7.0
_u = np.array([0.5, 0.6, 1.2])
_u = 1.4 * _u / np.linalg.norm(_u)
H2_ATOMS = [("H", (1.0, 1.2, 1.1)), ("H", tuple(np.array([1.0, 1.2, 1.1]) + _u))]
AUX = ferric_basis("cc-pvdz-ri", ["H"])


def system(n, basis="6-31g", w=0.5):
    sc = Supercell(A0, H2_ATOMS, basis, n)
    d = build_supercell(sc, AUX, w=w)
    scf = L.gamma_scf(d, 2 * sc.R)
    return sc, d, scf


def main(sizes=((1, 1, 4), (2, 2, 2))):
    for n in sizes:
        t0 = time.time()
        sc, d, scf = system(n)
        ec = L.canonical_mp2(scf, d["B"])
        p = L.prepare(sc, d, scf)
        r0 = L.solve(p, 0.0)
        print(
            f"== {n}: nao {d['S'].shape[0]} nocc {scf['nocc']} nvir {p['Cv'].shape[1]} (VV {p['n_l']}, HV {p['n_h']}) "
            f"naux {d['info']['naux']} kept {d['info']['naux_kept']}  E_HF {scf['e']:.12f}  E_MP2(canon) {ec:.12e}"
        )
        print(
            f"   construction: orth {p['dev_orth']:.1e} span {p['dev_span']:.1e}; Berghold f {p['loc']['f0']:.4f} -> "
            f"{p['loc']['f']:.6f} in {p['loc']['sweeps']} sweeps, |grad| {p['loc']['grad']:.1e}"
        )
        print(
            f"   ANCHOR eps=0: E {r0['e']:.12e}  dE {r0['e'] - ec:+.2e}  (cg {r0['niter']})"
        )
        # M1
        pm = L.prepare(sc, d, scf, drop_hv=1)
        rm = L.solve(pm, 0.0)
        print(f"   M1 drop one HV:            dE {rm['e'] - ec:+.2e}")
        # M2
        dist = L.pair_distances(p)
        Rstar = dist.max() + 1e-6
        rmi = L.solve(p, 0.0, pair_cut=Rstar)
        rraw = L.solve(p, 0.0, pair_cut=Rstar, raw=True)
        print(
            f"   M2 pair cutoff at trivial R*={Rstar:.4f}: min-image dE {rmi['e'] - ec:+.2e} (pairs {rmi['pairs_kept']}), "
            f"raw dE {rraw['e'] - ec:+.2e} (pairs {rraw['pairs_kept']}/{scf['nocc'] ** 2})"
        )
        # M3
        daux = L.min_image(p["cen"][:, None, :] - d["aux_xyz"][None, :, :], p["a_sc"])
        Raux = daux.max() + 1e-6
        Jmi, st = L.domain_fit_J(p, d, Raux)
        Jraw, st_raw = L.domain_fit_J(p, d, Raux, raw=True)
        e_mi = L.solve(p, 0.0, J=Jmi)["e"]
        print(
            f"   M3 domain fit at trivial R*={Raux:.4f}: min-image max|dJ| {abs(Jmi - p['J']).max():.1e} dE {e_mi - ec:+.2e} "
            f"(dom {st['dom_mean']:.0f}/{st['naux']}); raw max|dJ| {abs(Jraw - p['J']).max():.1e} "
            f"(dom mean {st_raw['dom_mean']:.1f}) dE {L.solve(p, 0.0, J=Jraw)['e'] - ec:+.2e}"
        )
        # M4
        pb = L.prepare(sc, d, scf, loc="boys-molecular")
        rb = L.solve(pb, 0.0)
        print(
            f"   M4 molecular Boys: eps=0 dE {rb['e'] - ec:+.2e} (blind spot: anchor passes)  [{time.time() - t0:.0f} s]",
            flush=True,
        )


if __name__ == "__main__":
    main(
        tuple(tuple(int(x) for x in s.split("x")) for s in sys.argv[1:])
        or ((1, 1, 4), (2, 2, 2))
    )
