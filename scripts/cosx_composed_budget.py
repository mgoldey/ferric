#!/usr/bin/env python3
"""COMPOSED COSX budget: do the four claimed factors actually multiply?

WHY THIS SCRIPT EXISTS
----------------------
The COSX "~150x deficit" close was retracted (commit e878a25c) because its
grid factor was judged on max|dK| (the K MATRIX) rather than on the converged
SCF ENERGY.  On the energy the overlap fit is worth ~17x at (35,86) on water.
The retraction then produced a range of 12x-219x by MULTIPLYING four factors
that were each measured in ISOLATION:

    (i)   overlap fit         measured on ONE system (water), ONE grid (35,86)
    (ii)  right-sized grid    chosen against an unstated energy criterion
    (iii) screening ~2x       measured at a DIFFERENT (fine) grid, (50,110)
    (iv)  MD 3c1e kernel ~7x  a FLOP-count assumption, never implemented

Multiplying four isolated factors is precisely the error that produced the
retracted 150x in the first place.  This script attacks the two questions
that decide the project:

    Q1 (generality) Is the fit's 17x a water accident or a general effect?
    Q2 (composition) Does the fit's benefit SURVIVE at the coarse grid you
       would actually right-size to, or is it grid-dependent -- i.e. do
       factors (i) and (ii) compose, or do they eat each other?

Q2 is the load-bearing one.  If the fit is worth 17x only at (35,86) and
worth ~1x at (25,50), then "fitted AND right-sized" is not fit_gain x
grid_gain; it is one or the other.

=====================================================================
EXACTNESS ANCHOR (repo rule: written and passing BEFORE any sweep)
=====================================================================
The trivial limit of a quadrature scheme is the GRID LIMIT.  `--anchor` runs
the COSX SCF at (150,590) and requires |dE| < 1e-7 Ha vs analytic-K RHF, for
BOTH the fitted and unfitted drivers.  A driver bug (sign, symmetrization,
D-convention, or a broken Q) cannot be refined away, so without this every
row below is meaningless.  The FITTED driver needs its own anchor: the
retraction only anchored the unfitted one, and the fit is the factor whose
value is now load-bearing.

ARTIFACT HYPOTHESIS (pre-registered)
------------------------------------
If the fit is a REAL error-cancelling correction:
    fit gain (unfitted |dE| / fitted |dE|) is > 1 at every grid, and is
    LARGEST on coarse grids -- because that is where quadrature error is
    largest and there is most for the fit to remove.  The gain then SHRINKS
    under refinement as both errors approach zero.  Under this hypothesis
    factors (i) and (ii) COMPOSE FAVOURABLY: right-sizing to a coarser grid
    makes the fit MORE valuable, not less.
If the 17x is a COINCIDENCE (two error curves crossing near zero):
    the gain is ERRATIC in grid -- large at one grid, ~1x or below 1x at the
    neighbouring ones -- and the sign of the fitted error flips around.  The
    17x is then a near-cancellation at one grid, transferable to nothing.
These predictions differ (monotone-and-largest-when-coarse vs erratic), so
the experiment can distinguish them.  NOTE: a THIRD outcome is possible and
would be the worst case -- gain > 1 but LARGEST on FINE grids and ~1x when
coarse.  That would mean (i) and (ii) ANTI-compose: you can have the fit's
benefit or the coarse grid's, not both.  All three are distinguishable from
the same table.

WHAT IS AND IS NOT MEASURED HERE
--------------------------------
MEASURED: converged-SCF-energy error for fitted and unfitted COSX across a
grid series, on 3-4 systems, plus an isodesmic reaction energy where error
cancellation between reactants and products is credited.
NOT MEASURED HERE: screening (factor iii) at the right-sized grid -- that
needs alkane_16-scale, which is out of reach for this dense-numpy prototype
(int1e_grids materializes (chunk, nbf, nbf)).  Factor (iv) is a FLOP
assumption and is not measurable in Python at all.  Both are labelled
ASSUMED in the final arithmetic and the product is reported as an UPPER
BOUND on the benefit, never as an estimate.
"""

import argparse
import numpy as np
import scipy.linalg
from pyscf import gto, scf, dft


# ------------------------------------------------------------------ systems
# Geometries kept small on purpose: the box is memory-contended and the
# quantity of interest (energy error) is load-independent, so there is no
# reason to run anything large.
GEOMS = {
    "water": """O  0.0000  0.0000  0.1173
                H  0.0000  0.7572 -0.4692
                H  0.0000 -0.7572 -0.4692""",
    "methane": """C  0.0000  0.0000  0.0000
                  H  0.6276  0.6276  0.6276
                  H  0.6276 -0.6276 -0.6276
                  H -0.6276  0.6276 -0.6276
                  H -0.6276 -0.6276  0.6276""",
    "ethane": """C  0.0000  0.0000  0.7680
                 C  0.0000  0.0000 -0.7680
                 H  1.0192  0.0000  1.1573
                 H -0.5096  0.8826  1.1573
                 H -0.5096 -0.8826  1.1573
                 H -1.0192  0.0000 -1.1573
                 H  0.5096 -0.8826 -1.1573
                 H  0.5096  0.8826 -1.1573""",
    # n-butane, staggered -- "alkane_4"; least symmetric case here, and the
    # system the retracted cost model's single timing datum came from.
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
    # Propane, for the isodesmic reaction below.
    "propane": """C  0.0000  0.5863  0.0000
                  C  1.2681 -0.2626  0.0000
                  C -1.2681 -0.2626  0.0000
                  H  0.0000  1.2449  0.8760
                  H  0.0000  1.2449 -0.8760
                  H  2.1614  0.3745  0.0000
                  H  1.3120 -0.9014  0.8968
                  H  1.3120 -0.9014 -0.8968
                  H -2.1614  0.3745  0.0000
                  H -1.3120 -0.9014  0.8968
                  H -1.3120 -0.9014 -0.8968""",
}


def get_mol(name, basis):
    return gto.M(atom=GEOMS[name], basis=basis, unit="Angstrom", verbose=0)


def build_grid(mol, nrad, nang):
    g = dft.gen_grid.Grids(mol)
    g.atom_grid = (nrad, nang)
    g.prune = None
    g.build()
    return g.coords, g.weights


# ------------------------------------------------------------------- COSX K
class CosxK:
    """Seminumerical exchange builder (Neese 2009; overlap fit Izsak/Neese 2011).

    Q = S S_num^{-1} with S_num = X X^T is density-INDEPENDENT, so it is
    factorized once per (geometry, grid) exactly as production code would.
    Cholesky SOLVE, never an explicit inverse: S_num is a Gram matrix of a
    rank-npts sampling and is ill-conditioned on coarse grids.
    """

    def __init__(self, mol, coords, weights, fitted=False, chunk=600):
        self.mol = mol
        self.coords = coords
        self.chunk = chunk
        self.fitted = fitted
        ao = dft.numint.eval_ao(mol, coords)
        sw = np.sqrt(np.abs(weights))
        self.X = (ao * sw[:, None]).T
        self.npts = len(weights)
        self.fit_ok = True
        if fitted:
            S = mol.intor("int1e_ovlp")
            S_num = self.X @ self.X.T
            try:
                c, low = scipy.linalg.cho_factor(S_num)
                self.chol = (c, low)
                self.S = S
            except scipy.linalg.LinAlgError:
                # Report rather than silently pseudo-inverting: a failed fit
                # is a result, not something to paper over.
                self.fit_ok = False

    def __call__(self, D):
        X = self.X
        F = D @ X
        G = np.empty_like(X)
        for lo in range(0, self.npts, self.chunk):
            hi = min(lo + self.chunk, self.npts)
            A = self.mol.intor("int1e_grids", grids=self.coords[lo:hi])
            G[:, lo:hi] = np.einsum("gnl,lg->ng", A, F[:, lo:hi], optimize=True)
            del A
        Kt = X @ G.T
        if self.fitted:
            if not self.fit_ok:
                return None
            Z = scipy.linalg.cho_solve(self.chol, Kt)
            Kt = self.S @ Z
        return 0.5 * (Kt + Kt.T)


def run_cosx_scf(mol, coords, weights, fitted=False, conv=1e-9):
    """Full RHF SCF: analytic J, seminumerical K. Returns (E, converged)."""
    builder = CosxK(mol, coords, weights, fitted=fitted)
    if fitted and not builder.fit_ok:
        return None, False
    mf = scf.RHF(mol)
    mf.conv_tol = conv
    mf.max_cycle = 200
    bare_jk = scf.hf.get_jk

    def get_jk(mol_=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        dm = np.asarray(dm)
        vj = bare_jk(mol, dm, hermi=hermi, with_j=True,
                     with_k=False)[0] if with_j else None
        vk = builder(dm) if with_k else None
        return vj, vk

    mf.get_jk = get_jk
    e = mf.kernel()
    return e, mf.converged


GRIDS = [(25, 50), (35, 86), (50, 110), (75, 194)]


# ------------------------------------------------------------------- anchor
def anchor(basis="cc-pvdz"):
    """Trivial-limit exactness for BOTH drivers. Must pass before any sweep."""
    print("=" * 72)
    print("EXACTNESS ANCHOR -- grid limit (150,590), water/%s" % basis)
    print("Bar: |dE| < 1e-7 Ha vs analytic-K RHF, for fitted AND unfitted.")
    print("=" * 72)
    mol = get_mol("water", basis)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    e_ref = mf.kernel()
    coords, weights = build_grid(mol, 150, 590)
    ok = True
    for fitted in (False, True):
        e, conv = run_cosx_scf(mol, coords, weights, fitted=fitted)
        if e is None:
            print(f"  fitted={fitted}: FIT FACTORIZATION FAILED")
            ok = False
            continue
        dE = abs(e - e_ref)
        good = dE < 1e-7 and conv
        ok &= good
        print(f"  fitted={str(fitted):5s} npts={len(weights)}  |dE|={dE:.3e} Ha"
              f"  converged={conv}   {'PASS' if good else 'FAIL'}")
    print(f"\nANCHOR {'PASS -- sweep may proceed' if ok else 'FAIL -- STOP'}")
    return ok


# -------------------------------------------------------------------- sweep
def sweep(systems, basis):
    """Fitted vs unfitted converged-SCF-energy error across the grid series.

    Returns {system: {(nrad,nang): (npts, dE_unfit, dE_fit, e_unfit, e_fit)}}
    """
    out = {}
    for name in systems:
        mol = get_mol(name, basis)
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-11
        e_ref = mf.kernel()
        print()
        print(f"### {name}/{basis}  nbf={mol.nao}  natm={mol.natm}  "
              f"E_ref={e_ref:.10f}")
        print(f"{'grid':>11} {'npts':>8} {'dE unfit':>12} {'dE fit':>12} "
              f"{'fit gain':>10}")
        print("-" * 58)
        rows = {}
        for (nr, na) in GRIDS:
            coords, weights = build_grid(mol, nr, na)
            eu, cu = run_cosx_scf(mol, coords, weights, fitted=False)
            ef, cf = run_cosx_scf(mol, coords, weights, fitted=True)
            du = abs(eu - e_ref) if eu is not None else float("nan")
            df = abs(ef - e_ref) if ef is not None else float("nan")
            gain = du / df if df > 0 else float("inf")
            rows[(nr, na)] = (len(weights), du, df, eu, ef, e_ref)
            flag = "" if (cu and cf) else "  [NOT CONVERGED]"
            print(f"{nr:4d}x{na:<5d} {len(weights):8d} {du:12.3e} {df:12.3e} "
                  f"{gain:9.1f}x{flag}")
            del coords, weights
        out[name] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", action="store_true")
    ap.add_argument("--basis", default="cc-pvdz")
    ap.add_argument("--systems", default="water,methane,ethane")
    ap.add_argument("--reaction", action="store_true")
    args = ap.parse_args()

    if args.anchor:
        anchor(args.basis)
        return

    if args.reaction:
        reaction(args.basis)
        return

    systems = args.systems.split(",")
    res = sweep(systems, args.basis)

    # ---- Q1: is the fit gain general, or water-specific? ----
    print()
    print("=" * 72)
    print("Q1  FIT GAIN (unfitted dE / fitted dE) BY SYSTEM AND GRID")
    print("=" * 72)
    hdr = f"{'grid':>11}" + "".join(f"{s:>12}" for s in systems)
    print(hdr)
    for (nr, na) in GRIDS:
        line = f"{nr:4d}x{na:<5d}"
        for s in systems:
            _, du, df, *_ = res[s][(nr, na)]
            line += f"{du/df:11.1f}x"
        print(line)

    # ---- Q2: does the gain survive coarsening (do (i) and (ii) compose)? ----
    print()
    print("=" * 72)
    print("Q2  COMPOSITION -- is the fit gain LARGEST where you want to be")
    print("    (coarse), or does it evaporate there?")
    print("=" * 72)
    for s in systems:
        gains = [res[s][g][1] / res[s][g][2] for g in GRIDS]
        coarse, fine = gains[0], gains[-1]
        if coarse >= max(gains) * 0.9:
            verdict = "COMPOSES (largest when coarse)"
        elif fine >= max(gains) * 0.9:
            verdict = "ANTI-COMPOSES (needs a FINE grid to pay)"
        else:
            verdict = "ERRATIC (peak mid-series -> near-cancellation, not a trend)"
        print(f"  {s:10s} gains {['%.1f' % g for g in gains]}  -> {verdict}")

    # ---- right-sized grid under stated criteria ----
    print()
    print("=" * 72)
    print("RIGHT-SIZED GRID under stated ABSOLUTE-ENERGY criteria")
    print("=" * 72)
    for bar, label in ((1e-5, "1e-5 Ha"), (1.6e-4, "0.1 kcal/mol abs")):
        print(f"\n  criterion: |dE_total| < {label}")
        for s in systems:
            for tag, idx in (("unfitted", 1), ("fitted", 2)):
                hit = [g for g in GRIDS if res[s][g][idx] < bar]
                got = f"{hit[0][0]}x{hit[0][1]} npts={res[s][hit[0]][0]}" \
                    if hit else "none in series"
                print(f"    {s:10s} {tag:9s} -> {got}")

    # Deliberately no file dump: every number that matters is in the tables
    # above, and the repo's scripts/ hygiene convention keeps probe artifacts
    # out of the tree.




# =====================================================================
# REACTION ENERGY -- the honest production criterion.
# =====================================================================
# Absolute total-energy error is EXTENSIVE: it grows with system size and no
# one converges it to 1e-5 Ha on a real molecule. What must be converged is a
# RELATIVE energy, where per-atom quadrature error partially cancels between
# reactants and products. This is the criterion that most favours COSX, so it
# is the one an honest budget must use.
#
# Isodesmic:  C2H6 + CH4  ->  C3H8 + H2 is NOT isodesmic (bond types differ).
# Use the genuinely isodesmic bond-conserving reaction
#       C3H8  +  CH4   ->   2 C2H6
# (propane + methane -> 2 ethane): C-C bonds 2+0 -> 2, C-H bonds 8+4 -> 12.
# Equal counts each side, so quadrature error per bond type cancels maximally.
def reaction(basis="cc-pvdz"):
    """dE_rxn error for C3H8 + CH4 -> 2 C2H6, fitted and unfitted, per grid."""
    stoich = {"propane": -1, "methane": -1, "ethane": +2}
    print("=" * 72)
    print("REACTION-ENERGY CRITERION:  C3H8 + CH4 -> 2 C2H6  (isodesmic)")
    print("Bar: |d(dE_rxn)| < 0.1 kcal/mol = 1.594e-4 Ha")
    print("=" * 72)

    mols = {n: get_mol(n, basis) for n in stoich}
    e_ref = {}
    for n, m in mols.items():
        mf = scf.RHF(m)
        mf.conv_tol = 1e-11
        e_ref[n] = mf.kernel()
    rxn_ref = sum(c * e_ref[n] for n, c in stoich.items())
    print(f"analytic-K dE_rxn = {rxn_ref:.10f} Ha "
          f"= {rxn_ref*627.5095:.4f} kcal/mol\n")

    print(f"{'grid':>11} {'unfit err':>13} {'fit err':>13} "
          f"{'unfit kcal':>11} {'fit kcal':>10}")
    print("-" * 62)
    rows = {}
    for (nr, na) in GRIDS:
        tot = {}
        for fitted in (False, True):
            acc = 0.0
            for n, c in stoich.items():
                coords, weights = build_grid(mols[n], nr, na)
                e, _ = run_cosx_scf(mols[n], coords, weights, fitted=fitted)
                acc += c * e
                del coords, weights
            tot[fitted] = acc
        du = abs(tot[False] - rxn_ref)
        df = abs(tot[True] - rxn_ref)
        rows[(nr, na)] = (du, df)
        print(f"{nr:4d}x{na:<5d} {du:13.3e} {df:13.3e} "
              f"{du*627.5095:11.4f} {df*627.5095:10.4f}")

    print()
    bar = 0.1 / 627.5095
    for tag, idx in (("unfitted", 0), ("fitted", 1)):
        hit = [g for g in GRIDS if rows[g][idx] < bar]
        print(f"  coarsest grid meeting 0.1 kcal/mol, {tag:8s}: "
              f"{hit[0] if hit else 'none in series'}")
    return rows


if __name__ == "__main__":
    main()
