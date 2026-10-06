"""Reference for `tune_omega`'s cation-stability verdict (issue #288 item 3).

PySCF UKS doublet cation, NON-VV10 range-separated hybrid wB97X, def2-SVP,
`mf.omega` overridden, default (minao) guess and NOT stability-followed: the
point is the curvature of the symmetric branch, which is what `tune_omega`
converges. Records the lowest eigenvalue of PySCF's own UKS orbital Hessian
(dense `newton_ah.gen_g_hop_uhf`) per omega.

N2+ is the positive case (the symmetric cation becomes a saddle inside the
tuned range); H2O+ and NH3+ are the stable negative controls. wB97X carries no
VV10, so ferric's Hessian has no omitted term (unlike wB97X-V).

Like-for-like recipe = gen_rsh_omega.py's (exact J, DF-K in the attenuated
metric, jkfit aux, (75,110) unpruned Becke grid), minus VV10.

Run:  OPENBLAS_NUM_THREADS=1 scripts/validation/run_slot.sh --light -- \\
        ~/qc/ferric/.venv/bin/python scripts/validation/gen_rsh_cation_stability.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import gen_rsh_omega as g  # noqa: E402

ROW = "rsh_cation_stability"
XC_PYSCF = "wb97x"
XC_FERRIC = "HYB_GGA_XC_WB97X"
BASIS = g.BASIS
SYSTEMS = {
    "n2": (0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.80),
    "h2o": (0.30, 0.60, 1.00),
    "nh3": (0.30, 0.60, 1.00),
}


def _mf(cmol, aux, omega):
    from pyscf import df, dft

    mf = g._exact_j_dfk(dft.uks.UKS)(cmol, xc=XC_PYSCF)
    mf.ferric_dfobj = df.DF(cmol, auxbasis=aux)
    mf.ferric_rij_dfobj = None
    mf.omega = float(omega)
    mf.grids.atom_grid = g.MAIN_GRID
    mf.grids.prune = None
    mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    mf.conv_tol = g.CONV_TOL
    mf.conv_tol_grad = g.CONV_TOL_GRAD
    mf.max_cycle = 500
    mf.verbose = 0
    return mf


def main() -> int:
    t0 = time.time()
    only = set(sys.argv[1:])
    for system, omegas in SYSTEMS.items():
        if only and system not in only:
            continue
        xyz = common.MOL_DIR / f"{system}.xyz"
        symbols, coords = common.read_xyz(xyz)
        cmol = common.build_pyscf_mol(xyz, BASIS, charge=1, multiplicity=2)
        basis_check = common.check_basis_like_for_like(cmol, BASIS, symbols)
        aux = g._aux_basis(symbols)
        points = []
        for w in omegas:
            mf = _mf(cmol, aux, w)
            e = float(mf.kernel())
            if not mf.converged:
                raise RuntimeError(f"{system} w={w}: cation did not converge")
            lam = common.uhf_lambda_min(mf)
            ss, _ = mf.spin_square()
            points.append(
                {
                    "omega": w,
                    "e_cation": e,
                    "cation_s_squared": float(ss),
                    "cation_lambda_min": lam,
                }
            )
            print(f"{system} w={w:.3f} E={e:.10f} lam={lam:+.6e}", flush=True)
        payload = {
            "row": "RSH omega tuning (cation stability)",
            "system": system,
            "basis": BASIS,
            "functional": XC_FERRIC,
            "points": points,
        }
        import pyscf

        payload["provenance"] = common.provenance(
            code="PySCF",
            version=pyscf.__version__,
            keywords={
                "method": "dft.UKS doublet cation, mf.omega override, minao guess, "
                "not stability-followed; lambda_min = dense gen_g_hop_uhf",
                "xc": XC_PYSCF,
                "jk": g.JK_RECIPE,
            },
            basis_name=BASIS,
            xyz_path=xyz,
            coords_bohr=coords,
            symbols=symbols,
            grid={
                "main": {
                    "atom_grid": list(g.MAIN_GRID),
                    "prune": None,
                    "partition": "Becke (original_becke)",
                    "radii_adjust": "becke_atomic_radii_adjust",
                }
            },
            aux={
                "name": g.AUX,
                "json": str(common.basis_json_path(g.AUX).relative_to(common.ROOT)),
                "sha256": common.sha256_file(common.basis_json_path(g.AUX)),
            },
            frozen_core=None,
            scf_conv={"conv_tol": g.CONV_TOL, "conv_tol_grad": g.CONV_TOL_GRAD},
            stability={"cation": "NOT followed; lambda_min of the converged state"},
            generator="scripts/validation/gen_rsh_cation_stability.py",
            extra={"basis_self_check": basis_check, "runtime_s": round(time.time() - t0, 1)},
        )
        path = common.write_reference(ROW, system, BASIS, payload)
        print(f"wrote {path}", flush=True)
    print("GEN_RSH_CATION_STABILITY_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
