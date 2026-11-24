#!/usr/bin/env python3
"""PySCF references for the tau-only meta-GGA (r2SCAN) linear-response consumers.

Validation target (ferric side: crates/ferric-scf/tests/mgga_response_vs_pyscf.rs;
the TDDFT reference is gen_tddft_refs.py's r2SCAN case):

    ferric                                   PySCF construction                     factor
    uhf_internal_stability, UKS r2SCAN       newton_ah.gen_g_hop_uhf on dft.UKS     1
    rhf_internal_stability, RKS r2SCAN       newton_ah.gen_g_hop_rhf on dft.RKS     1/2

Both codes take f_xc (incl. the v2rhotau / v2sigmatau / v2tau2 blocks) from
libxc on the same (75,110) unpruned Becke grid with Becke-1988 radii and exact
(4-index) J, so the comparison is like-for-like up to SCF convergence.

Systems (6-31G, ferric's bundled JSON on both sides, the gen_scf_stability.py
geometries):
    nh2       NH2 doublet UKS/r2SCAN; dense UKS-internal Hessian spectrum.
    water_eq  water RKS/r2SCAN; dense singlet RKS-internal Hessian spectrum.

Run (light; seconds):
    flock -o /tmp/ferric-compile.lock env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
        uv run --no-sync python scripts/validation/gen_mgga_response.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

ROW = "mgga_response"
BASIS = "6-31g"
XC = "r2SCAN"
PYSCF_XC = "MGGA_X_R2SCAN,MGGA_C_R2SCAN"
MAIN_GRID = (75, 110)
CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-10
N_SPECTRUM = 8


def dense_from_hop(hop, n):
    import numpy as np

    hess = np.empty((n, n))
    e = np.zeros(n)
    for i in range(n):
        e[:] = 0.0
        e[i] = 1.0
        hess[:, i] = np.asarray(hop(e)).real.ravel()
    asym = float(np.max(np.abs(hess - hess.T)))
    return np.linalg.eigvalsh(0.5 * (hess + hess.T)), asym


def _ks(mol, restricted):
    from pyscf import dft

    mf = (dft.RKS if restricted else dft.UKS)(mol, xc=PYSCF_XC)
    mf.grids.atom_grid = MAIN_GRID
    mf.grids.prune = None
    mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = CONV_TOL_GRAD
    mf.max_cycle = 500
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.kernel(mf.mo_coeff, mf.mo_occ)
    if not mf.converged:
        raise RuntimeError(f"{'RKS' if restricted else 'UKS'}/{XC} did not converge")
    return mf


def _load(system, charge, mult):
    xyz = common.MOL_DIR / f"{system}.xyz"
    symbols, coords = common.read_xyz(xyz)
    mol = common.build_pyscf_mol(xyz, BASIS, charge=charge, multiplicity=mult)
    check = common.check_basis_like_for_like(mol, BASIS, symbols)
    return xyz, symbols, coords, mol, check


def _prov(xyz, symbols, coords, check, method):
    import pyscf

    return common.provenance(
        code="pyscf",
        version=pyscf.__version__,
        keywords={"method": method, "xc": PYSCF_XC, "jk": "EXACT J (4-index), no K"},
        basis_name=BASIS,
        xyz_path=xyz,
        coords_bohr=coords,
        symbols=symbols,
        grid={
            "atom_grid": list(MAIN_GRID),
            "prune": None,
            "partition": "Becke (original_becke)",
            "radii_adjust": "becke_atomic_radii_adjust",
        },
        scf_conv={"conv_tol": CONV_TOL, "conv_tol_grad": CONV_TOL_GRAD},
        generator="scripts/validation/gen_mgga_response.py",
        extra={"basis_self_check": check},
    )


def gen_nh2():
    from pyscf.soscf import newton_ah

    xyz, symbols, coords, mol, check = _load("nh2", 0, 2)
    mf = _ks(mol, restricted=False)
    _mo_i, _mo_e, stable_i, _ = mf.stability(
        internal=True, external=False, return_status=True
    )
    if not stable_i:
        raise RuntimeError("NH2 UKS/r2SCAN is internally unstable from minao")
    g, h_op, h_diag = newton_ah.gen_g_hop_uhf(mf, mf.mo_coeff, mf.mo_occ)
    ev, asym = dense_from_hop(h_op, len(h_diag))
    s2, _ = mf.spin_square()
    payload = {
        "row": "meta-GGA response",
        "system": "nh2",
        "basis": BASIS,
        "nao": int(mol.nao),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "uks_r2scan": {
            "energy": float(mf.e_tot),
            "s_squared": float(s2),
            "max_orbital_gradient": float(abs(g).max()),
            "hessian": {
                "construction": "dense newton_ah.gen_g_hop_uhf h_op (== ferric uhf_internal_stability, factor 1)",
                "dim": int(len(ev)),
                "lowest": [float(x) for x in ev[:N_SPECTRUM]],
                "max_asymmetry": asym,
            },
        },
        "provenance": _prov(xyz, symbols, coords, check, "dft.UKS r2SCAN"),
    }
    p = common.write_reference(ROW, "nh2", BASIS, payload)
    print(f"nh2 UKS/r2SCAN E={mf.e_tot:.10f} lowest={ev[:3]} -> {p}")


def gen_water():
    from pyscf.soscf import newton_ah

    xyz, symbols, coords, mol, check = _load("water_eq", 0, 1)
    mf = _ks(mol, restricted=True)
    g, h_op, h_diag = newton_ah.gen_g_hop_rhf(mf, mf.mo_coeff, mf.mo_occ)
    raw, asym = dense_from_hop(h_op, len(h_diag))
    sing = raw / 2.0
    if sing[0] <= 0:
        raise RuntimeError(f"water RKS/r2SCAN singlet lambda_min {sing[0]:+.3e} <= 0")

    payload = {
        "row": "meta-GGA response",
        "system": "water_eq",
        "basis": BASIS,
        "nao": int(mol.nao),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "rks_r2scan": {
            "energy": float(mf.e_tot),
            "max_orbital_gradient": float(abs(g).max()),
            "singlet_hessian": {
                "construction": "dense newton_ah.gen_g_hop_rhf h_op / 2 (ferric rhf_internal_stability convention)",
                "pyscf_raw_factor": 2.0,
                "dim": int(len(sing)),
                "lowest": [float(x) for x in sing[:N_SPECTRUM]],
                "max_asymmetry": asym,
            },
        },
        "provenance": _prov(xyz, symbols, coords, check, "dft.RKS r2SCAN"),
    }
    p = common.write_reference(ROW, "water_eq", BASIS, payload)
    print(f"water RKS/r2SCAN E={mf.e_tot:.10f} singlet lowest={sing[:3]} -> {p}")


if __name__ == "__main__":
    gen_nh2()
    gen_water()
