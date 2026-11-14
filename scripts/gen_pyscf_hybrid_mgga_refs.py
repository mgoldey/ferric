#!/usr/bin/env python3
"""Generate hybrid meta-GGA reference energies and gradients from PySCF.

Covers the τ-only hybrid meta-GGAs ferric resolves through libxc: TPSSh
(HYB_MGGA_XC_TPSSH, 10% exact exchange), M06-2X (HYB_MGGA_X_M06_2X +
MGGA_C_M06_2X, 54%), SCAN0 (HYB_MGGA_X_SCAN0 + MGGA_C_SCAN, 25%) and the
range-separated wB97M-V (HYB_MGGA_XC_WB97M_V with VV10).

Settings match ferric's (as in gen_pyscf_mgga_refs.py): the flat (75, 110)
Becke-Lebedev grid with no pruning and Becke-1988 radii adjustment, and
EXACT four-centre J and K (no density fitting; ferric runs with df_j_aux and
df_k_aux set to ""). Exact J/K removes the RI-J/RI-K convention differences
between the codes (~3e-7 Ha on H2O/cc-pVDZ for PBE0 and TPSS alike), so
what remains is the hybrid meta-GGA itself and the shared grid. The one
exception is wB97M-V: ferric builds range-separated exchange only by RI-K, so
its reference fits J and K with def2-universal-jkfit. wB97M-V's VV10 term uses a
(50, 50) NLC grid, ferric's default NLC grid, with libxc's own VV10
parameters (b = 6.0, C = 0.01). Gradients use grid_response = True, which
ferric's XC gradient includes.

Energies: cc-pVDZ, H2O (RKS) and NH2 doublet (UKS; TPSSh also ROKS).
Gradients: 6-31G (ferric's AO Hessians, which the meta-GGA gradient needs,
cover s/p shells only), H2O and NH2.

Usage:
    python scripts/gen_pyscf_hybrid_mgga_refs.py            # energies + gradients
    python scripts/gen_pyscf_hybrid_mgga_refs.py stability  # TPSSh orbital Hessians
"""

import json
import sys
from pathlib import Path

import numpy as np
from pyscf import dft, gto

REFDIR = Path(__file__).resolve().parents[1] / "testdata" / "reference"

MAIN_GRID = (75, 110)
NLC_GRID = (50, 50)
RSH_AUX = "def2-universal-jkfit"

MOLECULES = {
    "h2o": ("O 0 0 0; H 0 0.7572 0.5868; H 0 -0.7572 0.5868", 0, 1),
    "nh2": ("N 0 0 0.1414; H 0 0.8067 -0.4950; H 0 -0.8067 -0.4950", 0, 2),
}

# file tag -> PySCF xc string (the same libxc components ferric resolves).
PYSCF_XC = {
    "tpssh": "HYB_MGGA_XC_TPSSH",
    "m062x": "HYB_MGGA_X_M06_2X,MGGA_C_M06_2X",
    "scan0": "HYB_MGGA_X_SCAN0,MGGA_C_SCAN",
    "wb97m_v": "HYB_MGGA_XC_WB97M_V",
}

ENERGY_CASES = [
    (mol, "cc-pvdz", xc) for xc in ("tpssh", "m062x", "scan0") for mol in ("h2o", "nh2")
] + [("h2o", "cc-pvdz", "wb97m_v")]

GRAD_CASES = [
    ("h2o", "6-31g", "tpssh"),
    ("h2o", "6-31g", "m062x"),
    ("nh2", "6-31g", "tpssh"),
    ("nh2", "6-31g", "m062x"),
]


def build(label, basis, xc, restricted_open=False):
    spec, charge, mult = MOLECULES[label]
    mol = gto.M(atom=spec, basis=basis, charge=charge, spin=mult - 1, unit="Angstrom")
    if restricted_open:
        kind = dft.ROKS
    else:
        kind = dft.RKS if mult == 1 else dft.UKS
    mf = kind(mol, xc=PYSCF_XC[xc])
    mf.grids.atom_grid = MAIN_GRID
    mf.grids.prune = None
    mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    if xc == "wb97m_v":
        # ferric builds range-separated exchange only by RI-K, so this one
        # case fits J and K with def2-universal-jkfit in both codes.
        mf = mf.density_fit(auxbasis=RSH_AUX)
        mf.nlcgrids.atom_grid = NLC_GRID
        mf.nlcgrids.prune = None
        mf.nlcgrids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    mf.conv_tol = 1e-11
    mf.max_cycle = 300
    return mol, mf, charge, mult


def record(label, basis, xc, mol, mf, charge, mult, e_total):
    hyb = dft.libxc.hybrid_coeff(PYSCF_XC[xc], spin=mult - 1)
    omega, alpha, beta = dft.libxc.rsh_coeff(PYSCF_XC[xc])
    return {
        "label": label,
        "basis": basis,
        "xc": xc,
        "pyscf_xc": PYSCF_XC[xc],
        "charge": charge,
        "multiplicity": mult,
        "main_grid": list(MAIN_GRID),
        "nlc_grid": list(NLC_GRID) if xc == "wb97m_v" else None,
        "jk": f"df:{RSH_AUX}" if xc == "wb97m_v" else "exact",
        "hybrid_coeff": float(hyb),
        "rsh_coeff": [float(omega), float(alpha), float(beta)],
        "libxc_version": dft.libxc.__version__,
        "e_total": float(e_total),
        "e_nuc": float(mol.energy_nuc()),
        "converged": bool(mf.converged),
    }


def dense_lowest(h_op, n, k=6):
    """Lowest eigenvalues of the dense, symmetrized orbital Hessian."""
    hess = np.empty((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        hess[:, i] = np.asarray(h_op(e)).real.ravel()
    return np.linalg.eigvalsh(0.5 * (hess + hess.T))[:k]


def stability_refs():
    """TPSSh internal-stability spectra: the orbital Hessian with the τ f_xc
    kernel plus 0.1·K response. RKS: newton_ah.gen_g_hop_rhf / 2 (ferric's
    singlet convention); UKS: gen_g_hop_uhf (factor 1)."""
    from pyscf.soscf import newton_ah

    for label, gen, factor in (
        ("h2o", newton_ah.gen_g_hop_rhf, 0.5),
        ("nh2", newton_ah.gen_g_hop_uhf, 1.0),
    ):
        mol, mf, charge, mult = build(label, "cc-pvdz", "tpssh")
        e = mf.kernel()
        _g, h_op, h_diag = gen(mf, mf.mo_coeff, mf.mo_occ)
        low = factor * dense_lowest(h_op, len(h_diag))
        out = record(label, "cc-pvdz", "tpssh", mol, mf, charge, mult, e)
        out["hessian_lowest"] = [float(x) for x in low]
        out["hessian_convention"] = (
            "dense gen_g_hop_rhf / 2" if factor == 0.5 else "dense gen_g_hop_uhf"
        )
        path = REFDIR / f"{label}_cc-pvdz_tpssh_stability.json"
        path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {path}  lowest = {low[:3]}")


def main():
    REFDIR.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ["stability"]:
        stability_refs()
        return
    mol, mf, charge, mult = build("nh2", "cc-pvdz", "tpssh", restricted_open=True)
    e = mf.kernel()
    out = record("nh2", "cc-pvdz", "tpssh", mol, mf, charge, mult, e)
    out["reference"] = "roks"
    path = REFDIR / "nh2_cc-pvdz_tpssh_roks.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}  E = {e:.10f}  converged={mf.converged}")
    for label, basis, xc in ENERGY_CASES:
        mol, mf, charge, mult = build(label, basis, xc)
        e = mf.kernel()
        out = record(label, basis, xc, mol, mf, charge, mult, e)
        path = REFDIR / f"{label}_{basis}_{xc}.json"
        path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {path}  E = {e:.10f}  converged={mf.converged}")
    for label, basis, xc in GRAD_CASES:
        mol, mf, charge, mult = build(label, basis, xc)
        e = mf.kernel()
        g = mf.nuc_grad_method()
        g.grid_response = True
        grad = g.kernel()
        out = record(label, basis, xc, mol, mf, charge, mult, e)
        out["grid_response"] = True
        out["grad"] = [[float(v) for v in row] for row in grad]
        path = REFDIR / f"{label}_{basis}_{xc}_grad.json"
        path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {path}  E = {e:.10f}  converged={mf.converged}")


if __name__ == "__main__":
    main()
