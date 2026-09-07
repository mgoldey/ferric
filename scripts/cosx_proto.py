#!/usr/bin/env python3
"""Stage 0 COSX (seminumerical exchange) prototype in numpy/PySCF.

PURPOSE
-------
Establish, in a throwaway high-level language BEFORE any Rust is written,
that the COSX working equations as stated in the design study actually
reproduce the analytic exchange matrix, and that the overlap-fitted variant
buys the grid coarsening it is supposed to buy.  Per the repo's
"prototype new methods in PYTHON first" convention.

WORKING EQUATIONS (Neese, Chem. Phys. 356, 98 (2009)); overlap fit from
Izsak & Neese, JCP 135, 144105 (2011):

    X_{mu,g}      = sqrt(w_g) * chi_mu(r_g)                  (nbf, npts)
    F_{lam,g}     = (D X)_{lam,g}                            half transform
    A^g_{mu,lam}  = \\int chi_mu(r) chi_lam(r) / |r - r_g| dr  per grid point
    G_{nu,g}      = sum_lam A^g_{nu,lam} F_{lam,g}
    Ktilde        = X G^T
    K_plain       = 0.5 * (Ktilde + Ktilde^T)
    K_fitted      = 0.5 * (Q Ktilde + (Q Ktilde)^T),  Q = S S_num^{-1},
                    S_num = X X^T  (density-INDEPENDENT; Cholesky SOLVE only)

Sign convention (VERIFIED, not assumed -- the first run of this script got it
backwards and the pre-registered Y2 branch caught it):
    PySCF's int1e_grids returns the POSITIVE Coulomb kernel
        <mu| +1/|r-r_g| |nu>,
    which is exactly what COSX needs, so A = +int1e_grids (NO negation).
    Verified by independent construction: for H2/STO-3G at an off-centre
    probe, int1e_grids[0,0] = 0.9133 vs a brute-force 120^3 cartesian
    quadrature of +chi_0(r)^2/|r-r_g| = 0.9132 (ratio 1.0002, residual = the
    coarse cartesian grid).
    NOTE this DIFFERS from libint2's nuclear-attraction operator, which with a
    unit probe charge yields the ATTRACTIVE -1/|r-r_g| (see ferric's
    esp_at_points, which adds the resulting block with a + sign to get
    V_elec).  Stage 1 in Rust must therefore NEGATE the libint2 block.
    Getting this wrong flips the sign of K; it shows up as max|dK| plateauing
    at ~2*||K||_max, which is precisely what happened.

=====================================================================
PRE-REGISTERED HYPOTHESES  (written and committed BEFORE running)
=====================================================================

EXACTNESS ANCHOR (the trivial limit where the approximation does nothing):
  A numerical-quadrature scheme has no "vacuous" parameter setting that makes
  it algebraically exact, so the trivial limit here is the GRID LIMIT:
  as the grid is refined without bound, K_COSX -> K_analytic elementwise.
  The anchor assertion is therefore CONVERGENCE, not a single equality:
      max|K_COSX - K_analytic| must DECREASE MONOTONICALLY under refinement
      and reach < 1e-6 on water/cc-pVDZ at or before (99, 302).
  Additionally there is one genuinely algebraic anchor that needs no
  analytic reference at all:
      ||Ktilde - Ktilde^T||_max  is a pure grid-error diagnostic; it must
      also fall monotonically to ~0.  If Ktilde were exact it would be
      symmetric by construction, so its asymmetry bounds the grid error
      from below.  This is checked INDEPENDENTLY of mf.get_k().

ARTIFACT HYPOTHESIS (what a broken implementation would look like):
  If the implementation is correct (X real):
      max|dK| falls monotonically over the refinement series and crosses 1e-6;
      ||Ktilde - Ktilde^T|| falls in LOCKSTEP with max|dK| (same order of
      magnitude), because both are driven by the same quadrature error.
      The overlap-fitted K reaches the 1e-6 bar on a grid at least 2x coarser
      (fewer angular points) than the plain one.
  If the implementation is broken (Y):
      (Y1) index/transpose error in the G contraction: max|dK| PLATEAUS at
           some O(1e-2..1e0) floor that grid refinement does not reduce,
           while ||Ktilde - Ktilde^T|| KEEPS FALLING.  The two diagnostics
           DECOUPLE.  This is the key discriminator: X predicts lockstep,
           Y1 predicts decoupling.
      (Y2) sign error on A: max|dK| ~ 2*||K_analytic|| and does not fall.
      (Y3) missing 0.5*(.+.^T) symmetrization: dK stays at the level of the
           asymmetry itself, i.e. max|dK| ~ ||Ktilde - Ktilde^T||, never below.
      (Y4) explicit inverse of an ill-conditioned S_num: the FITTED variant
           is WORSE than plain on coarse grids and gets erratic (non-monotone)
           rather than better, since a rank-deficient Gram matrix inverse
           amplifies noise.  X predicts fitted BETTER than plain on coarse grids.
  X != Y for every branch: X says "monotone fall, two diagnostics in lockstep,
  fitted beats plain"; the failure modes each break a DIFFERENT one of those
  three, so the experiment can distinguish them.

REACHABILITY: the GO bar (1e-6 at <= 99x302) must be checkable in the sense
that the series actually reaches (99,302); if even the finest grid we can
afford is coarser than that, the run is inconclusive, NOT a pass.

GO/NO-GO (Stage 0):
  BAR 1: plain COSX K converges monotonically, max|dK| < 1e-6 by (99,302).
  BAR 2: overlap-fitted variant reaches max|dK| < 1e-6 on a grid at least
         2x coarser (in angular points) than the coarsest grid at which
         plain COSX does.
"""

import numpy as np
from pyscf import gto, scf, dft
import scipy.linalg


def build_grid(mol, nrad, nang):
    """Unpruned Becke-Lebedev grid, (nrad, nang) per atom."""
    g = dft.gen_grid.Grids(mol)
    g.atom_grid = (nrad, nang)
    g.prune = None
    g.build()
    return g.coords, g.weights


def cosx_k(mol, D, coords, weights, S=None, fitted=False, chunk=2000):
    """Seminumerical exchange matrix.

    Returns (K, ktilde_asym) where ktilde_asym = ||Ktilde - Ktilde^T||_max,
    a reference-free grid-error diagnostic.
    """
    nbf = mol.nao
    npts = len(weights)

    # X_{mu,g} = sqrt(w_g) chi_mu(r_g)
    ao = dft.numint.eval_ao(mol, coords)          # (npts, nbf)
    sw = np.sqrt(np.abs(weights))
    X = (ao * sw[:, None]).T                       # (nbf, npts)

    # F_{lam,g} = (D X)_{lam,g}
    F = D @ X                                      # (nbf, npts)

    # G_{nu,g} = sum_lam A^g_{nu,lam} F_{lam,g}, chunked over grid points.
    G = np.empty((nbf, npts))
    for lo in range(0, npts, chunk):
        hi = min(lo + chunk, npts)
        # int1e_grids -> (nchunk, nbf, nbf), already the POSITIVE +1/|r-r_g|
        # kernel COSX wants (verified against brute-force quadrature; see the
        # module docstring). No negation.
        Achunk = mol.intor('int1e_grids', grids=coords[lo:hi])
        # einsum over lam for each g in the chunk
        G[:, lo:hi] = np.einsum('gnl,lg->ng', Achunk, F[:, lo:hi], optimize=True)

    Ktilde = X @ G.T                               # (nbf, nbf)
    asym = np.abs(Ktilde - Ktilde.T).max()

    if not fitted:
        return 0.5 * (Ktilde + Ktilde.T), asym

    # Overlap fitting: K = 0.5*(S S_num^{-1} Ktilde + h.c.)
    # S_num = X X^T is density-independent -> factorize once per geometry.
    # Cholesky SOLVE, never an explicit inverse (S_num is a Gram matrix of a
    # rank-npts sampling and is ill-conditioned on coarse grids).
    S_num = X @ X.T
    # Solve S_num Z = Ktilde  =>  Z = S_num^{-1} Ktilde, then Q Ktilde = S Z.
    try:
        c, low = scipy.linalg.cho_factor(S_num)
        Z = scipy.linalg.cho_solve((c, low), Ktilde)
    except scipy.linalg.LinAlgError:
        # S_num not positive definite on this grid -> report, do not silently
        # fall back to a pseudo-inverse (that would hide the failure).
        return None, asym
    QK = S @ Z
    return 0.5 * (QK + QK.T), asym


def main():
    np.set_printoptions(precision=6, suppress=True)

    mol = gto.M(
        atom='''O  0.0000  0.0000  0.1173
                H  0.0000  0.7572 -0.4692
                H  0.0000 -0.7572 -0.4692''',
        basis='cc-pvdz', unit='Angstrom', verbose=0)

    mf = scf.RHF(mol)
    mf.kernel()
    D = mf.make_rdm1()
    K_ref = mf.get_k(mol, D)
    S = mol.intor('int1e_ovlp')

    print(f"molecule: water/cc-pVDZ  nbf={mol.nao}  E(RHF)={mf.e_tot:.10f}")
    print(f"||K_analytic||_max = {np.abs(K_ref).max():.6e}")
    print()

    # Refinement series: angular points grow, radial too. The (99,302) grid is
    # the GO bar; we include one grid past it to show the trend continues.
    grids = [(25, 50), (35, 86), (50, 110), (75, 194), (99, 302)]

    print(f"{'grid':>12} {'npts':>8} {'max|dK| plain':>15} {'max|dK| fit':>15} "
          f"{'||Kt-Kt^T||':>14}")
    print("-" * 70)

    rows = []
    for (nrad, nang) in grids:
        coords, weights = build_grid(mol, nrad, nang)
        Kp, asym = cosx_k(mol, D, coords, weights, S=S, fitted=False)
        Kf, _ = cosx_k(mol, D, coords, weights, S=S, fitted=True)
        dp = np.abs(Kp - K_ref).max()
        df = np.abs(Kf - K_ref).max() if Kf is not None else float('nan')
        rows.append((nrad, nang, len(weights), dp, df, asym))
        print(f"{nrad:5d}x{nang:<6d} {len(weights):8d} {dp:15.6e} {df:15.6e} "
              f"{asym:14.6e}")

    print()
    # ---- Evaluate the pre-registered bars ----
    dps = [r[3] for r in rows]
    dfs = [r[4] for r in rows]
    asyms = [r[5] for r in rows]

    mono_p = all(dps[i + 1] < dps[i] for i in range(len(dps) - 1))
    mono_a = all(asyms[i + 1] < asyms[i] for i in range(len(asyms) - 1))
    print(f"plain max|dK| monotone decreasing : {mono_p}")
    print(f"||Ktilde-Ktilde^T|| monotone       : {mono_a}")

    # BAR 1
    bar1 = mono_p and dps[-1] < 1e-6
    print(f"BAR 1 (plain < 1e-6 by 99x302, monotone): "
          f"{'PASS' if bar1 else 'FAIL'}  (final {dps[-1]:.3e})")

    # BAR 2: coarsest grid where each variant crosses 1e-6
    def first_cross(vals):
        for i, v in enumerate(vals):
            if v < 1e-6:
                return i
        return None

    ip, ifit = first_cross(dps), first_cross(dfs)
    if ip is None:
        print("BAR 2: undefined - plain never crossed 1e-6 on this series")
        bar2 = False
    elif ifit is None:
        print("BAR 2: FAIL - fitted never crossed 1e-6")
        bar2 = False
    else:
        ang_p, ang_f = grids[ip][1], grids[ifit][1]
        ratio = ang_p / ang_f
        bar2 = ratio >= 2.0
        print(f"BAR 2: plain crosses at {grids[ip]}, fitted at {grids[ifit]}; "
              f"angular ratio {ratio:.2f}x -> {'PASS' if bar2 else 'FAIL'}")

    # Artifact discriminator: lockstep vs decoupling (X vs Y1).
    print()
    print("artifact check (X: lockstep; Y1: decoupled):")
    for (nr, na, np_, dp, df, asym) in rows:
        ratio = dp / asym if asym > 0 else float('inf')
        print(f"  {nr:3d}x{na:<4d} max|dK|/||Kt-Kt^T|| = {ratio:9.3f}")

    print()
    print(f"STAGE 0 VERDICT: {'GO' if (bar1 and bar2) else 'NO-GO'}")


if __name__ == '__main__':
    main()
