#!/usr/bin/env python3
"""Generate ferric meta-GGA (SCAN / r2SCAN) reference GRADIENTS from PySCF.

Companion to `gen_pyscf_mgga_refs.py` (which does energies only). Uses the same
ferric-matching settings — RI-J Coulomb via def2-universal-jkfit, ferric's flat
(75, 110) Becke-Lebedev grid with no pruning, Becke-1988 atomic-size adjustment,
conv_tol 1e-10.

`grid_response = True`, unlike the older `gen_pyscf_dft_grad_refs.py`: ferric's
GGA and meta-GGA XC gradient paths BOTH include the P2.1 Becke-weight and
home-translation grid-response corrections, so the matching PySCF reference must
include them too. (The LDA/PBE/B3LYP refs in that older script predate the
grid-response work and were generated with it off.)

Basis sets are restricted to STO-3G / 6-31G because ferric's AO Hessians — which
the meta-GGA gradient needs for both the ∂²χ Pulay term and the τ term — are
implemented for s/p shells only.

Usage:
    python scripts/gen_pyscf_mgga_grad_refs.py
"""
import json
from pathlib import Path

from pyscf import dft, gto

REFDIR = Path(__file__).resolve().parents[1] / "testdata" / "reference"

# Ferric default main grid: (75, 110). Match exactly with no pruning.
MAIN_GRID = (75, 110)

# (label, basis, atom spec, charge, multiplicity, pyscf xc string, file tag)
CASES = [
    ("h2",  "sto-3g", "H 0 0 0; H 0 0 0.74",                            0, 1, "SCAN",   "scan"),
    ("h2o", "sto-3g", "O 0 0 0; H 0 0.7572 0.5868; H 0 -0.7572 0.5868", 0, 1, "SCAN",   "scan"),
    ("h2o", "sto-3g", "O 0 0 0; H 0 0.7572 0.5868; H 0 -0.7572 0.5868", 0, 1, "R2SCAN", "r2scan"),
    ("h2o", "6-31g",  "O 0 0 0; H 0 0.7572 0.5868; H 0 -0.7572 0.5868", 0, 1, "SCAN",   "scan"),
    # Open shell. NOTE (2026-07-27): ferric's spin-polarized SCAN SCF ENERGY
    # disagrees with PySCF by ~2e-4 Ha on this system (closed-shell SCAN agrees
    # to ~1e-8, and OH/PBE agrees to 3e-8), so the ferric-vs-PySCF gradient gap
    # here is dominated by the density, not the gradient formula. Kept as a
    # reference so the pre-existing energy defect stays measured.
    ("oh",  "sto-3g", "O 0 0 0; H 0 0 0.97",                            0, 2, "SCAN",   "scan"),
]


def run_one(label, basis, spec, charge, mult, xc, tag):
    mol = gto.M(atom=spec, basis=basis, charge=charge, spin=mult - 1, unit="Angstrom")
    mf = (dft.RKS if mult == 1 else dft.UKS)(mol, xc=xc)
    mf = mf.density_fit(auxbasis="def2-universal-jkfit")
    mf.grids.atom_grid = MAIN_GRID
    mf.grids.prune = None
    mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    mf.conv_tol = 1e-10
    mf.max_cycle = 300
    e_total = mf.kernel()
    g = mf.nuc_grad_method()
    g.grid_response = True
    grad = g.kernel()
    return {
        "label": label,
        "basis": basis,
        "xc": tag,
        "charge": charge,
        "multiplicity": mult,
        "main_grid": list(MAIN_GRID),
        "grid_response": True,
        "e_total": float(e_total),
        "converged": bool(mf.converged),
        "grad": [[float(v) for v in row] for row in grad],
    }


def main():
    REFDIR.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        label, basis, _, _, _, _, tag = case
        out = run_one(*case)
        path = REFDIR / f"{label}_{basis}_{tag}_grad.json"
        path.write_text(json.dumps(out, indent=2))
        print(f"wrote {path}  E_total = {out['e_total']:.10f}  "
              f"converged={out['converged']}")


if __name__ == "__main__":
    main()
