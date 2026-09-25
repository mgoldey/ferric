"""Iteration 16 exactness anchors for the Gamma RHF forces (no PySCF pbc anywhere).

(a) analytic vs central FD (h=1e-4) of the prototype's OWN Gamma energy, both exxdiv, pure-AFT and Ewald-split routes;
(b) translation invariance: sum_A F_A = 0;
(c) mutations, each must move (a) by >> the FD noise (~1e-9):
      vne_no_basis   drop the basis-centre motion in V_ne (SR 3c and LR pair-FT parts)
      w_sign         +W instead of -W in the overlap (Pulay) term
      no_ewald_lr    forget the reciprocal part of dE_nn/dR
      no_madelung_s  drop -v_M/2 D S D from the S-coupled term (exxdiv=ewald only)
Predictions: correct code -> FD residual at the FD truncation level (h^2 f'''/6 ~ 1e-9) and sum F ~ 1e-15.
A mutant that leaves (a) at 1e-9 would mean the anchor is blind to that term -> redesign.
Note the SR route needs CONVERGED image cutoffs: pbc_gamma's defaults (rcut_bra 12) truncate the STO-3G overlap
range (pair e^{-0.084 L^2} = 6e-6 at L=12) and then analytic != FD by 5.6e-7 (w=1) / 2.9e-6 (w=2); rcut_bra 16 / rcut_2e 17
gives 2.2e-9.  The SR anchor below therefore uses a compact s+p basis for which the defaults are converged.
"""

import time

import numpy as np
from pyscf import gto

import pbc_grad as PGd
from pbc_gamma import Cell

H2 = (np.eye(3) * 4.0, [("H", (0.3, 0.2, 0.1)), ("H", (0.35, 0.12, 1.5))])
COMPACT = {"H": gto.parse("H S\n  0.5  1.0\nH P\n  0.8  1.0\n")}
COMPS = [(0, 0), (0, 2), (1, 1)]


def case(route, exxdiv):
    a, atoms = H2
    if route == "aft":
        return Cell(a, atoms, "sto-3g"), dict(w=None, exxdiv=exxdiv, gcut=12.0)
    return Cell(a, atoms, COMPACT), dict(w=1.0, exxdiv=exxdiv)


def residual(cell, kw, fd):
    r = PGd.gamma_rhf_grad(cell, 2, **kw)
    return max(abs(r["grad"][k] - v) for k, v in fd.items()), abs(r["grad"].sum(0)).max(), r


def main():
    for route, ex in (("aft", "none"), ("aft", "ewald"), ("sr", "ewald")):
        if True:
            cell, kw = case(route, ex)
            t = time.time()
            fd = PGd.fd_grad(cell, 2, COMPS if route == "aft" else COMPS[1:], **kw)
            res, sf, r = residual(cell, kw, fd)
            print(f"{route:3s} exxdiv={ex:5s}: max|analytic-FD| = {res:.1e}  |sum F| = {sf:.1e}  "
                  f"F(H0) = {np.array2string(r['grad'][0], precision=9)} ({time.time() - t:.0f}s)", flush=True)
            # SR route (s+p, nao 8) costs ~5 min per gradient here: only the mutant that has an SR-specific part
            muts = ["vne_no_basis"] if route == "sr" else ["vne_no_basis", "w_sign", "no_ewald_lr"] + (
                ["no_madelung_s"] if ex == "ewald" else [])
            for m in muts:
                PGd._MUTANT = m
                try:
                    res_m, sf_m, _ = residual(cell, kw, fd)
                finally:
                    PGd._MUTANT = None
                print(f"    MUTANT {m:14s}: max|analytic-FD| = {res_m:.2e}  |sum F| = {sf_m:.1e}", flush=True)


if __name__ == "__main__":
    main()
