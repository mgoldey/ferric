"""Ours-only half of run_kpts_eps_gap.py: the eigenvalue spread of pbc_kpts.krhf (the oracle's exact call,
|[F,DS]| < 1e-7, eps of the DIIS-extrapolated F) and of scf() at loose/tight residuals, vs a 1e-12 reference.
Hypothesis (b) predicts the spread matches the Iteration-9 PySCF gap (2x2x2: 5.1e-10 none / 4.1e-9 ewald).
Usage: python3 run_kpts_eps_gap_ours.py n1n2n3"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pbc_gamma import Cell  # noqa: E402
from pbc_kpts import build_k, krhf  # noqa: E402
from run_kpts_anchor import SP, TRI_A, TRI_ATOMS  # noqa: E402
from run_kpts_eps_gap import CACHE, NELEC, energy, maxd, scf  # noqa: E402

n = tuple(int(c) for c in sys.argv[1])
path = os.path.join(CACHE, f"kb_tri_{''.join(map(str, n))}.npz")
if os.path.exists(path):
    kb = dict(np.load(path))
    for key, f in (("Nk", int), ("enn", float), ("madelung", float)):
        kb[key] = f(kb[key])
else:
    kb = build_k(Cell(TRI_A, TRI_ATOMS, SP), n, exxdiv="ewald")
    np.savez(path, **kb)
for ex in (None, "ewald"):
    vm = kb["madelung"] if ex else 0.0
    dmR, FR, epsR, rR, _ = scf(kb, vm, 1e-12)
    eR = energy(kb, dmR, FR)
    e0, eps0, it0 = krhf(kb, NELEC, conv=1e-12, kshift=vm)
    print(
        f"tri {n} [{ex}] ref resid {rR:.1e} E {eR:.13f}; krhf(oracle call) it {it0} dE {e0 - eR:.1e} "
        f"max|d eps| {maxd(eps0, epsR):.1e}",
        flush=True,
    )
    for g in (1e-6, 1e-7, 1e-8, 1e-9, 1e-10):
        dm, F, eps, r, it = scf(kb, vm, g)
        print(
            f"tri {n} [{ex}] scf resid {r:.1e} (it {it}): dE {energy(kb, dm, F) - eR:.1e} "
            f"max|d eps| {maxd(eps, epsR):.1e} ratio {maxd(eps, epsR) / r:.2f}",
            flush=True,
        )
