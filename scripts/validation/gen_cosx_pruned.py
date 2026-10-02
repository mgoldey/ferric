"""PySCF reference for ferric's PRUNED COSX grid (`prune = "sgx"`).

Consumer: crates/ferric-scf/tests/validation_cosx.rs
          (`pruned_sgx_grid_matches_pyscf_on_the_same_grid`).
Output:   testdata/reference/validation/cosx_pruned/<system>_<basis>.json

The like-for-like block of gen_cosx.py (`sgx_matched_nofit`: PySCF SGX with
exact direct J, fit off, no density screening, no grid-point removal), on a
grid built the way ferric builds its pruned COSX grid:

    radial   Treutler-Ahlrichs, the SAME count for every element (ferric's
             AtomicGridConfig has one n_radial)
    angular  PySCF's own `sgx_prune` (gen_grid.py: NWChem region boundaries on
             Bragg radii, region orders from SGX_ANG_MAPPING; at a 194 peak
             26/50/110/194/110), which ferric's PruneScheme::Sgx transcribes
    weights  Becke partitioning + becke_atomic_radii_adjust (ferric's), NOT
             PySCF's default LKO for SGX

An independent construction of the same discrete COSX energy: ferric (fit off,
screens off) must match it to the SCF floor.

Run (through the slot):
    scripts/validation/run_slot.sh -- \\
        ~/qc/ferric/.venv/bin/python scripts/validation/gen_cosx_pruned.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_cosx  # noqa: E402

ROW = "cosx_pruned"
GRIDS = ((35, 194), (50, 302))
SYSTEMS = {
    "h2o": (common.MOL_DIR / "h2o.xyz", "aug-cc-pvdz"),
    "butane": (common.ROOT / "testdata" / "molecules" / "alkane_4.xyz", "def2-svp"),
}


def pruned_grids(mol, n_radial: int, n_angular: int):
    from pyscf import dft

    g = dft.gen_grid.Grids(mol)
    g.atom_grid = (n_radial, n_angular)
    g.prune = dft.gen_grid.sgx_prune
    g.radii_adjust = dft.radi.becke_atomic_radii_adjust
    g.alignment = 1  # no zero-weight padding: n_points is then ferric's count
    g.build(with_non0tab=True)
    g.non0tab = g.make_mask(mol, g.coords)
    g.screen_index = g.non0tab
    return g


def run_pruned(mol, n_radial, n_angular, dm0):
    from pyscf.sgx import sgx as sgx_mod
    from pyscf.sgx import sgx_jk

    mf = sgx_mod.sgx_fit(gen_cosx.make_scf(mol, "hf"), auxbasis="weigend", pjs=False)
    w = mf.with_df
    w.dfj = False
    w.direct_j = True
    w.optk = False
    w.fit_ovlp = False
    w.grids_level_i = w.grids_level_f = 0
    w.grids_thrd = 0.0
    mf.rebuild_nsteps = 1
    orig = sgx_jk.get_gridss
    sgx_jk.get_gridss = lambda m, level=1, gthrd=0.0, use_opt_grids=False: pruned_grids(
        m, n_radial, n_angular
    )
    try:
        t0 = time.time()
        e = mf.kernel(dm0=dm0)
        wall = time.time() - t0
        npts = int((w.grids.weights != 0).sum())
        ntot = int(w.grids.weights.size)
    finally:
        sgx_jk.get_gridss = orig
    assert mf.converged
    return float(e), npts, ntot, wall


def main() -> int:
    import pyscf

    for system, (xyz, basis_name) in SYSTEMS.items():
        symbols, coords = common.read_xyz(xyz)
        mol = common.build_pyscf_mol(xyz, basis_name)
        basis_check = common.check_basis_like_for_like(mol, basis_name, symbols)
        mf = gen_cosx.make_scf(mol, "hf")
        e_exact = float(mf.kernel())
        assert mf.converged
        dm0 = mf.make_rdm1()
        blocks = {}
        for nr, na in GRIDS:
            e, npts, ntot, wall = run_pruned(mol, nr, na, dm0)
            blocks[f"{nr}_{na}"] = {
                "radial": nr,
                "angular_peak": na,
                "prune": "sgx",
                "n_points": ntot,
                "n_points_nonzero_weight": npts,
                "energy": e,
                "minus_exact": e - e_exact,
                "wall_s": round(wall, 1),
            }
            print(
                f"{system} {basis_name} sgx({nr},{na}) E={e:.12f} dE={e - e_exact:+.3e} npts={ntot}"
            )
        payload = {
            "row": "COSX pruned (sgx) grid",
            "system": system,
            "basis": basis_name,
            "charge": 0,
            "multiplicity": 1,
            "nao": mol.nao_nr(),
            "nuclear_repulsion": float(mol.energy_nuc()),
            "basis_check": basis_check,
            "exact": {"energy": e_exact},
            "sgx_pruned_matched_nofit": blocks,
            "provenance": common.provenance(
                code="PySCF",
                version=pyscf.__version__,
                keywords={
                    "scf": f"RHF, exact direct J; conv_tol {gen_cosx.CONV_TOL}",
                    "sgx": "sgx_fit(pjs=False); dfj=False, direct_j=True, optk=False, "
                    "grids_thrd=0, rebuild_nsteps=1, fit_ovlp=False",
                    "sgx_grid": "(n_radial, peak) Treutler radial (same count for every "
                    "element) x gen_grid.sgx_prune, Becke partition, "
                    "becke_atomic_radii_adjust (get_gridss replaced)",
                },
                basis_name=basis_name,
                xyz_path=xyz,
                coords_bohr=coords,
                symbols=symbols,
                grid={"sgx": [list(g) for g in GRIDS], "prune": "sgx_prune"},
                aux=None,
                frozen_core=None,
                scf_conv={"conv_tol": gen_cosx.CONV_TOL},
                stability=None,
                generator="scripts/validation/gen_cosx_pruned.py",
            ),
        }
        p = common.write_reference(ROW, system, basis_name, payload)
        print("wrote", p.relative_to(common.ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
