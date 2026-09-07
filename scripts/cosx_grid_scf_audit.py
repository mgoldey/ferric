#!/usr/bin/env python3
"""ADVERSARIAL AUDIT of the COSX "~150x deficit" close.

WHAT THE EARLIER WORK MEASURED
------------------------------
scripts/cosx_proto.py measured max|dK| -- the max elementwise error of the
EXCHANGE MATRIX against analytic K, at a FIXED converged density.  Its GO bar
was max|dK| < 1e-6.  The cost model that produced "~150x" then assumed the
grid needed to hit that bar, and the Stage 2 sweep hardcoded (50,110).

WHY THAT IS THE WRONG CRITERION
-------------------------------
Nobody runs COSX to reproduce K elementwise.  What must be converged is the
SCF TOTAL ENERGY.  Two reasons the matrix bar is far too strict:

  1. E is a CONTRACTION of K with D.  Elementwise errors of random sign
     partially cancel in the trace, so |dE| << max|dK| generically.
  2. E is VARIATIONAL in the orbitals.  A perturbation dK to the Fock matrix
     moves the converged density by O(dK) but the ENERGY by O(dK^2) through
     that density change.  Only the direct (first-order) term survives.

Production COSX codes (ORCA) exploit exactly this: a COARSE grid during SCF
iterations, a finer one for the final energy.

THIS SCRIPT
-----------
Runs a FULL SCF to self-consistency with K replaced by COSX-K at each grid in
the original sweep, and reports the converged TOTAL ENERGY error vs an
analytic-K SCF on the identical molecule/basis.  That is the number the cost
model should have been denominated in.

EXACTNESS ANCHOR (repo rule: written before measuring)
------------------------------------------------------
`--anchor` checks that the COSX SCF driver, when handed a grid so fine the
quadrature error is negligible, reproduces the analytic SCF energy.  If the
driver itself is broken (wrong K sign, wrong symmetrization, D vs 2D
convention) this fails and NO grid row means anything.  A driver bug would
otherwise masquerade as "grid error that never converges".

ARTIFACT HYPOTHESIS
-------------------
  If the "coarse grid suffices" claim is REAL:
      |dE| at (25,50)/(35,86) is orders of magnitude BELOW max|dK|, and falls
      with refinement.  The gap between |dE| and max|dK| is the cancellation
      + variational effect described above.
  If my SCF driver is BROKEN:
      |dE| does NOT fall with grid refinement (a constant offset), or the
      anchor fails outright.  A driver that, say, double-counts K gives a
      large |dE| at EVERY grid including the finest -- flat, not falling.
  FLAT vs FALLING distinguishes them.  These predictions differ, so the
  experiment can discriminate.
"""

import argparse
import time

import numpy as np
from pyscf import gto, scf, dft


# ---------------------------------------------------------------- molecules
def get_mol(name, basis):
    geoms = {
        "water": """O  0.0000  0.0000  0.1173
                    H  0.0000  0.7572 -0.4692
                    H  0.0000 -0.7572 -0.4692""",
        "methane": """C  0.0000  0.0000  0.0000
                      H  0.6276  0.6276  0.6276
                      H  0.6276 -0.6276 -0.6276
                      H -0.6276  0.6276 -0.6276
                      H -0.6276 -0.6276  0.6276""",
        # n-butane, staggered; the "alkane_4" of the earlier sweep.
        "butane": """C -1.9092  0.5675  0.0000
                     C -0.6321 -0.2708  0.0000
                     C  0.6321  0.2708  0.0000
                     C  1.9092 -0.5675  0.0000
                     H -2.7953 -0.0741  0.0000
                     H -1.9508  1.2087  0.8853
                     H -1.9508  1.2087 -0.8853
                     H -0.6157 -0.9175  0.8858
                     H -0.6157 -0.9175 -0.8858
                     H  0.6157  0.9175  0.8858
                     H  0.6157  0.9175 -0.8858
                     H  2.7953  0.0741  0.0000
                     H  1.9508 -1.2087  0.8853
                     H  1.9508 -1.2087 -0.8853""",
    }
    return gto.M(atom=geoms[name], basis=basis, unit="Angstrom", verbose=0)


def build_grid(mol, nrad, nang):
    g = dft.gen_grid.Grids(mol)
    g.atom_grid = (nrad, nang)
    g.prune = None
    g.build()
    return g.coords, g.weights


# ---------------------------------------------------------------- COSX K
class CosxK:
    """COSX exchange builder. Caches the grid-dependent, density-INDEPENDENT
    pieces (X, and the fitting factor) once per geometry+grid, exactly as a
    real implementation would across SCF iterations."""

    def __init__(self, mol, coords, weights, fitted=False, chunk=1200):
        self.mol = mol
        self.coords = coords
        self.chunk = chunk
        self.fitted = fitted
        ao = dft.numint.eval_ao(mol, coords)
        sw = np.sqrt(np.abs(weights))
        self.X = (ao * sw[:, None]).T                    # (nbf, npts)
        self.npts = len(weights)
        if fitted:
            S = mol.intor("int1e_ovlp")
            S_num = self.X @ self.X.T
            # Q = S S_num^{-1}; keep as a solve, never an explicit inverse.
            self.Q = np.linalg.solve(S_num.T, S.T).T
        # A-matrix chunks are the expensive part and are density-independent
        # ONLY in the sense that the integrals are; they are recomputed each
        # build here (as in a direct implementation).

    def __call__(self, D):
        """K_COSX for density matrix D (PySCF convention: D = 2*C_occ C_occ^T
        for RHF, and get_k(D) contracts that same D)."""
        X = self.X
        F = D @ X
        G = np.empty_like(X)
        for lo in range(0, self.npts, self.chunk):
            hi = min(lo + self.chunk, self.npts)
            A = self.mol.intor("int1e_grids", grids=self.coords[lo:hi])
            G[:, lo:hi] = np.einsum("gnl,lg->ng", A, F[:, lo:hi], optimize=True)
        Kt = X @ G.T
        if self.fitted:
            Kt = self.Q @ Kt
        return 0.5 * (Kt + Kt.T)


def run_cosx_scf(mol, coords, weights, fitted=False, conv=1e-9):
    """Full RHF SCF with analytic J and COSX K."""
    builder = CosxK(mol, coords, weights, fitted=fitted)
    mf = scf.RHF(mol)
    mf.conv_tol = conv
    # Replace ONLY the exchange half. J stays analytic -- that is the actual
    # COSX production setup (J from density fitting/analytic, K seminumerical).
    # NB: J must come from the bare integral driver, NOT mf.get_j, which
    # dispatches back into the overridden get_jk (infinite recursion).
    bare_jk = scf.hf.get_jk

    def get_jk(mol_=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
        dm = np.asarray(dm)
        vj = bare_jk(mol, dm, hermi=hermi, with_j=True, with_k=False)[0] \
            if with_j else None
        vk = builder(dm) if with_k else None
        return vj, vk
    mf.get_jk = get_jk
    e = mf.kernel()
    return e, mf.converged, builder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mol", default="water")
    ap.add_argument("--basis", default="cc-pvdz")
    ap.add_argument("--anchor", action="store_true",
                    help="run the exactness anchor only")
    ap.add_argument("--fitted", action="store_true")
    args = ap.parse_args()

    mol = get_mol(args.mol, args.basis)
    mf_ref = scf.RHF(mol)
    mf_ref.conv_tol = 1e-11
    e_ref = mf_ref.kernel()
    D_ref = mf_ref.make_rdm1()
    K_ref = mf_ref.get_k(mol, D_ref)
    kmax = np.abs(K_ref).max()

    print(f"system: {args.mol}/{args.basis}  nbf={mol.nao}  natm={mol.natm}")
    print(f"E(RHF, analytic K) = {e_ref:.12f}   ||K||_max = {kmax:.4e}")
    print()

    if args.anchor:
        # EXACTNESS ANCHOR: a very fine grid must reproduce the analytic SCF.
        nrad, nang = 150, 590
        coords, weights = build_grid(mol, nrad, nang)
        e, conv, _ = run_cosx_scf(mol, coords, weights)
        dE = abs(e - e_ref)
        print(f"ANCHOR grid=({nrad},{nang}) npts={len(weights)}")
        print(f"  E_COSX = {e:.12f}  converged={conv}")
        print(f"  |dE|   = {dE:.3e} Ha")
        # A driver bug (sign/symmetrization/D-convention) cannot be fixed by
        # grid refinement, so this bar is load-bearing.
        print(f"  ANCHOR {'PASS' if dE < 1e-7 else 'FAIL'} (bar 1e-7 Ha)")
        return

    grids = [(25, 50), (35, 86), (50, 110), (75, 194), (99, 302)]
    print(f"{'grid':>12} {'npts':>9} {'max|dK|':>13} {'|dE_total|':>13} "
          f"{'iters':>6} {'conv':>6}")
    print("-" * 68)

    rows = []
    for (nrad, nang) in grids:
        coords, weights = build_grid(mol, nrad, nang)
        t0 = time.time()
        e, conv, builder = run_cosx_scf(mol, coords, weights, fitted=args.fitted)
        # max|dK| at the REFERENCE density, i.e. exactly the quantity
        # cosx_proto.py reported, so the two columns are comparable.
        dK = np.abs(builder(D_ref) - K_ref).max()
        dE = abs(e - e_ref)
        rows.append((nrad, nang, len(weights), dK, dE))
        print(f"{nrad:5d}x{nang:<6d} {len(weights):9d} {dK:13.4e} {dE:13.4e} "
              f"{'-':>6} {str(conv):>6}   ({time.time()-t0:.1f}s)")

    print()
    print("RATIO max|dK| / |dE_total|  -- how much stricter the matrix bar is:")
    for (nr, na, n, dK, dE) in rows:
        print(f"  {nr:3d}x{na:<4d}  {dK/dE:10.1f}x")

    print()
    # The decision-relevant question: which is the COARSEST grid meeting a
    # chemically meaningful ENERGY bar?
    for bar in (1e-5, 1e-6):
        ok = [(nr, na, n) for (nr, na, n, dK, dE) in rows if dE < bar]
        if ok:
            nr, na, n = ok[0]
            print(f"coarsest grid with |dE| < {bar:.0e} Ha: ({nr},{na}) "
                  f"npts={n}")
        else:
            print(f"coarsest grid with |dE| < {bar:.0e} Ha: none in series")

    # Cost implication vs the (50,110) the cost model assumed.
    n_assumed = [r[2] for r in rows if (r[0], r[1]) == (50, 110)][0]
    print()
    print(f"npts at the ASSUMED (50,110): {n_assumed}")
    for (nr, na, n, dK, dE) in rows:
        print(f"  ({nr},{na}): npts={n:8d}  cost factor vs assumed = "
              f"{n/n_assumed:6.3f}x   |dE|={dE:.3e}")


if __name__ == "__main__":
    main()
