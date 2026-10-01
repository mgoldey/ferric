"""PySCF references for the VALIDATION.md "COSX seminumerical exchange" row.

Consumer: crates/ferric-scf/tests/validation_cosx.rs.
Output:   testdata/reference/validation/cosx/<system>_<basis>.json

Systems:

    h2o     testdata/molecules/validation/h2o.xyz   aug-cc-pVDZ   RHF and B3LYP
    butane  testdata/molecules/alkane_4.xyz         def2-SVP      RHF

(butane is the all-anti C4H10 every COSX study in this repo used, including
the C1 "floor" investigation in wiki/validation-campaign-bugs-2026-09.md.)

# What ferric's COSX is, and what PySCF's SGX is

ferric (crates/ferric-scf/src/cosx_k.rs):

    X = sqrt(w) chi,  F = D X,  G^g = A^g F,  Ktilde = X G^T
    K_plain = (Ktilde + Ktilde^T)/2
    K_fit   = sym(Q Ktilde),  Q = S S_num^{-1},  S_num = X X^T

PySCF 2.13 `sgx` with `pjs=False` (get_jk_favorj): the same K_plain when
`fit_ovlp=False`; with `fit_ovlp=True` it applies `P = S_num^{-1} S` to the
DENSITY (`F = w chi^T (P D)`), i.e. K = sym(sum_g S~_g P D A^g) against
ferric's sym(sum_g A^g D S~_g P), S~_g = w chi chi^T. The two FIT VARIANTS
differ at finite grid (both -> exact K as the grid -> exact), so fitted
energies are NOT comparable grid by grid.

# The blocks written

* `exact` — conventional RHF/RKS, exact four-centre J and K. ferric's
  exact-K energy must equal this to the SCF floor: it checks the harness
  (geometry, basis, functional, XC grid) before any COSX number is read.
* `sgx_matched_nofit` — THE LIKE-FOR-LIKE BLOCK. PySCF SGX with exact
  (direct) J, `fit_ovlp=False`, no density screening, no grid-point
  removal (`gthrd = 0`), on a grid built exactly as ferric builds its COSX
  grid: flat (unpruned) Treutler-Ahlrichs radial x Lebedev, Becke
  partitioning with Becke's atomic-size adjustment (the recipe that matches
  ferric's KS energies to 1e-10, `gen_ks_energies.py`). Same quadrature,
  same operator: this is an independent construction of the SAME discrete
  COSX energy, so ferric (fit off, screens off) must match it at EVERY grid,
  not only in the dense limit.
* `sgx_pyscf_fit` — stock PySCF fit (`fit_ovlp=True`, favorj path), same
  grids. A different fit variant: only its approach to `exact` is
  comparable (information; the Rust test prints it).

Every block reports E and E - E_exact.

Run (heavy: the butane 590/974 grids are minutes each; use the slot):
    scripts/validation/run_slot.sh -- \\
        uv run --no-sync python scripts/validation/gen_cosx.py [--only h2o|butane]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "cosx"
ROW_NAME = "COSX seminumerical exchange"

RADIAL = 75
# Matched (fit off) ladder: ferric's tabulated Lebedev orders.
ANGULAR_MATCHED = (110, 302, 434, 590)
# PySCF's own fit, same ladder plus 974 (ferric has no 974 table) as the
# densest independent point.
ANGULAR_PYSCF_FIT = (110, 302, 434, 590, 974)
# XC grid for B3LYP: ferric's default DFT grid (75,110) flat, matched as in
# gen_ks_energies.py. It is the same on both sides of every E - E_exact.
XC_GRID = (75, 110)
CONV_TOL = 1e-11
CONV_TOL_GRAD = 1e-7
DIRECT_SCF_TOL = 1e-14

SYSTEMS = {
    # name -> (xyz path, basis, methods)
    "h2o": (common.MOL_DIR / "h2o.xyz", "aug-cc-pvdz", ("hf", "b3lyp")),
    "butane": (
        common.ROOT / "testdata" / "molecules" / "alkane_4.xyz",
        "def2-svp",
        ("hf",),
    ),
}
PYSCF_XC = {"b3lyp": "B3LYP"}


def matched_grids(mol, n_angular: int):
    """ferric's COSX grid in PySCF: flat (RADIAL, n_angular), Becke + Becke
    size adjustment, no point removal; then the bookkeeping get_gridss does."""
    from pyscf import dft

    g = dft.gen_grid.Grids(mol)
    g.atom_grid = (RADIAL, n_angular)
    g.prune = None
    g.radii_adjust = dft.radi.becke_atomic_radii_adjust
    g.build(with_non0tab=True)
    g.non0tab = g.make_mask(mol, g.coords)
    g.screen_index = g.non0tab
    return g


def make_scf(mol, method: str):
    from pyscf import dft, scf

    if method == "hf":
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol)
        mf.xc = PYSCF_XC[method]
        mf.grids.atom_grid = XC_GRID
        mf.grids.prune = None
        mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.direct_scf_tol = DIRECT_SCF_TOL
    mf.max_cycle = 200
    mf.verbose = 0
    return mf


def run_sgx(mol, method: str, n_angular: int, fit: bool, dm0):
    from pyscf.sgx import sgx as sgx_mod
    from pyscf.sgx import sgx_jk

    # auxbasis is never used (dfj=False); naming one skips predefined_auxbasis
    # lookups on a custom (ferric-JSON) basis dict.
    mf = sgx_mod.sgx_fit(make_scf(mol, method), auxbasis="weigend", pjs=False)
    w = mf.with_df
    w.dfj = False
    w.direct_j = True  # exact (direct) J: COSX is the only approximation
    w.optk = False  # favorj path, no density screening
    w.fit_ovlp = fit
    w.grids_level_i = w.grids_level_f = 0  # never switch grids mid-SCF
    w.grids_thrd = 0.0
    mf.rebuild_nsteps = 1  # full (non-incremental) K every iteration

    orig = sgx_jk.get_gridss
    sgx_jk.get_gridss = lambda m, level=1, gthrd=0.0, use_opt_grids=False: (
        matched_grids(m, n_angular)
    )
    try:
        t0 = time.time()
        e = mf.kernel(dm0=dm0)
        wall = time.time() - t0
        npts = int((w.grids.weights != 0).sum())
    finally:
        sgx_jk.get_gridss = orig
    if not mf.converged:
        raise RuntimeError(
            f"SGX SCF ({method}, {n_angular}, fit={fit}) did not converge"
        )
    return float(e), npts, wall, int(getattr(mf, "cycles", -1))


def main() -> int:
    import pyscf

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(SYSTEMS), default=None)
    args = ap.parse_args()

    written = []
    for system, (xyz, basis_name, methods) in SYSTEMS.items():
        if args.only and system != args.only:
            continue
        symbols, coords = common.read_xyz(xyz)
        mol = common.build_pyscf_mol(xyz, basis_name)
        basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
        blocks = {}
        for method in methods:
            mf = make_scf(mol, method)
            t0 = time.time()
            e_exact = float(mf.kernel())
            assert mf.converged, f"{system}/{method}: exact SCF did not converge"
            dm0 = mf.make_rdm1()
            print(
                f"{system:6s} {basis_name:11s} {method:5s} exact E={e_exact:.12f} "
                f"({time.time() - t0:.1f} s)",
                flush=True,
            )

            matched, fitted = {}, {}
            for ang, fit, out in [(a, False, matched) for a in ANGULAR_MATCHED] + [
                (a, True, fitted) for a in ANGULAR_PYSCF_FIT
            ]:
                e, npts, wall, cyc = run_sgx(mol, method, ang, fit, dm0)
                key = f"{RADIAL}_{ang}"
                out[key] = {
                    "radial": RADIAL,
                    "angular": ang,
                    "n_points": npts,
                    "energy": e,
                    "minus_exact": e - e_exact,
                    "scf_cycles": cyc,
                    "wall_s": round(wall, 1),
                }
                print(
                    f"  ({RADIAL},{ang:4d}) fit={'on ' if fit else 'off'} "
                    f"E={e:.12f} E-Eexact={e - e_exact:+.3e} npts={npts} "
                    f"{wall:.1f} s",
                    flush=True,
                )
            blocks[method] = {
                "xc": PYSCF_XC.get(method, "HF"),
                "exact": {"energy": e_exact},
                "sgx_matched_nofit": matched,
                "sgx_pyscf_fit": fitted,
            }

        payload = {
            "row": ROW_NAME,
            "system": system,
            "basis": basis_name,
            "charge": 0,
            "multiplicity": 1,
            "nao": mol.nao_nr(),
            "nuclear_repulsion": float(mol.energy_nuc()),
            "basis_check": basis_check,
            "methods": blocks,
            "provenance": common.provenance(
                code="PySCF",
                version=pyscf.__version__,
                keywords={
                    "scf": "RHF / RKS(B3LYP), exact direct J; conv_tol "
                    f"{CONV_TOL}, conv_tol_grad {CONV_TOL_GRAD}, direct_scf_tol "
                    f"{DIRECT_SCF_TOL}",
                    "sgx": "sgx_fit(pjs=False); dfj=False, direct_j=True, optk=False "
                    "(get_jk_favorj), grids_thrd=0, rebuild_nsteps=1, fit_ovlp "
                    "False (matched) / True (pyscf_fit)",
                    "sgx_grid": f"flat ({RADIAL}, N) Treutler radial x Lebedev, Becke "
                    "partition, becke_atomic_radii_adjust (get_gridss replaced)",
                    "xc_grid": f"{XC_GRID} flat, becke_atomic_radii_adjust (B3LYP only)",
                },
                basis_name=basis_name,
                xyz_path=xyz,
                coords_bohr=coords,
                symbols=symbols,
                grid={
                    "sgx_radial": RADIAL,
                    "sgx_angular_matched": list(ANGULAR_MATCHED),
                    "sgx_angular_pyscf_fit": list(ANGULAR_PYSCF_FIT),
                    "xc": list(XC_GRID),
                },
                aux=None,
                frozen_core=None,
                scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
                stability=None,
                generator="scripts/validation/gen_cosx.py",
            ),
        }
        written.append(common.write_reference(ROW, system, basis_name, payload))
    for p in written:
        print("wrote", p.relative_to(common.ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
