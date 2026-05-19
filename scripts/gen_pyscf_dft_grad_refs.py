"""Generate KS-DFT nuclear-gradient references at ferric-matching settings.

For each {molecule, functional}, runs PySCF RKS with:
  * density_fit(auxbasis="def2-universal-jkfit")   — matches ferric RI-J/RI-K
  * grids.atom_grid = (75, 110), prune = None
  * grids.radii_adjust = becke_atomic_radii_adjust  — matches ferric Becke 1988
  * grad: default ("no grid response" — `Gradients.grid_response = False`)

Output: JSON {label, basis, xc, grad: [[gx, gy, gz], ...]} per atom.
"""
import json
import sys
from pathlib import Path

import numpy as np
from pyscf import dft, gto

REFDIR = Path(__file__).resolve().parent.parent / "testdata" / "reference"

MOLECULES = {
    "h2":  ("H 0 0 0; H 0 0 0.74", 0, 1),
    "h2o": ("O 0 0 0; H 0 0.7572 0.5868; H 0 -0.7572 0.5868", 0, 1),
}

PYSCF_XC = {
    "lda":     "LDA,VWN",
    "pbe":     "PBE,PBE",
    "b3lyp":   "B3LYP",
    "wb97x-v": "wB97X_V",
}

MAIN_GRID = (75, 110)
NLC_GRID  = (50, 50)   # matches ferric default


def run_one(label, atom_spec, charge, spin, basis, xc):
    mol = gto.M(atom=atom_spec, basis=basis, charge=charge,
                spin=spin - 1, unit="Angstrom")
    mf = dft.RKS(mol, xc=PYSCF_XC[xc])
    mf = mf.density_fit(auxbasis="def2-universal-jkfit")
    mf.grids.atom_grid = MAIN_GRID
    mf.grids.prune = None
    mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    if xc == "wb97x-v":
        mf.nlc = "VV10"
        mf.nlcgrids.atom_grid = NLC_GRID
        mf.nlcgrids.prune = None
        mf.nlcgrids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-7
    mf.kernel()

    g = mf.Gradients()
    g.grid_response = False
    g_arr = g.kernel()  # shape (natoms, 3)

    return {
        "label": label,
        "basis": basis,
        "xc": xc,
        "main_grid": list(MAIN_GRID),
        "e_total": float(mf.e_tot),
        "grad": g_arr.tolist(),
        "converged": bool(mf.converged),
    }


def main(only_xc=None):
    REFDIR.mkdir(parents=True, exist_ok=True)
    xcs = [only_xc] if only_xc else list(PYSCF_XC.keys())
    for xc in xcs:
        for label, (atom_spec, charge, spin) in MOLECULES.items():
            basis = "cc-pvdz"
            out = run_one(label, atom_spec, charge, spin, basis, xc)
            fname = f"{label}_{basis}_{xc.replace('-', '_')}_grad.json"
            path = REFDIR / fname
            path.write_text(json.dumps(out, indent=2))
            gmax = max(abs(v) for row in out["grad"] for v in row)
            print(f"wrote {path}  max|g|={gmax:.4e}  E={out['e_total']:.8f}  conv={out['converged']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
