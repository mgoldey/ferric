#!/usr/bin/env python3
"""Generate ferric DFT reference values from PySCF.

For each (molecule, basis, functional) tuple, runs RKS with matched grids
(no pruning), converged to 1e-10. Saves a JSON file under testdata/reference/
ready to be diffed against ferric.

Usage:
    python scripts/gen_pyscf_dft_refs.py              # all functionals
    python scripts/gen_pyscf_dft_refs.py lda          # just LDA
    python scripts/gen_pyscf_dft_refs.py pbe          # just PBE
"""
import json
import os
import sys
from pathlib import Path

from pyscf import dft, gto

REFDIR = Path(__file__).resolve().parents[1] / "testdata" / "reference"

# Atom-spec lines use the same coords as ferric's tests.
MOLECULES = {
    "h2":      ("H 0 0 0; H 0 0 0.74",                                 0, 1),
    "h2o":     ("O 0 0 0; H 0 0.7572 0.5868; H 0 -0.7572 0.5868",      0, 1),
    "methane": ("C 0 0 0;"
                " H 0.6276 0.6276 0.6276;"
                " H -0.6276 -0.6276 0.6276;"
                " H -0.6276 0.6276 -0.6276;"
                " H 0.6276 -0.6276 -0.6276",                            0, 1),
    # Fourth molecule (widens past H2/H2O/CH4): NH3, C3v, same geometry as
    # testdata/molecules/nh3.xyz / the RPA-gradient row's NH3 tests.
    "nh3":     ("N 0.000000 0.000000 0.116489;"
                " H 0.000000 0.939731 -0.271808;"
                " H 0.813831 -0.469865 -0.271808;"
                " H -0.813831 -0.469865 -0.271808",                     0, 1),
}

# Bases to generate refs for. cc-pvdz is the routine/default one already
# exercised by dft_{lda,pbe,b3lyp,wb97xv}.rs; def2-svp widens past
# single-basis coverage (already bundled, used elsewhere e.g. RHF+ECP row).
BASES = ["cc-pvdz", "def2-svp"]

# Ferric default main grid: (75, 110). Match exactly with no pruning.
MAIN_GRID = (75, 110)
NLC_GRID  = (50, 50)

PYSCF_XC = {
    "lda":     "LDA,VWN",
    "pbe":     "PBE,PBE",
    "b3lyp":   "B3LYP",
    "wb97x-v": "wB97X_V",
}

def run_one(label, atom_spec, charge, spin, basis, xc):
    mol = gto.M(atom=atom_spec, basis=basis, charge=charge,
                spin=spin - 1, unit="Angstrom")
    mf = dft.RKS(mol, xc=PYSCF_XC[xc])
    # Match ferric's RI-J for the Coulomb piece. wB97X-V also needs RI-K.
    mf = mf.density_fit(auxbasis="def2-universal-jkfit")
    mf.grids.atom_grid = MAIN_GRID
    mf.grids.prune = None
    # Match ferric's Becke (1988) size correction (not PySCF's default Treutler).
    mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    if xc == "wb97x-v":
        mf.nlc = "VV10"
        mf.nlcgrids.atom_grid = NLC_GRID
        mf.nlcgrids.prune = None
    mf.conv_tol = 1e-10
    e_total = mf.kernel()
    e_nuc = mol.energy_nuc()
    return {
        "label": label,
        "basis": basis,
        "xc": xc,
        "main_grid": list(MAIN_GRID),
        "nlc_grid": list(NLC_GRID) if xc == "wb97x-v" else None,
        "e_total": float(e_total),
        "e_nuc": float(e_nuc),
        "n_e": int(mol.nelectron),
        "converged": bool(mf.converged),
    }

def main(only_xc=None):
    REFDIR.mkdir(parents=True, exist_ok=True)
    xcs = [only_xc] if only_xc else list(PYSCF_XC.keys())
    for xc in xcs:
        for label, (atom_spec, charge, spin) in MOLECULES.items():
            for basis in BASES:
                out = run_one(label, atom_spec, charge, spin, basis, xc)
                fname = f"{label}_{basis}_{xc.replace('-', '_')}.json"
                path = REFDIR / fname
                path.write_text(json.dumps(out, indent=2))
                print(f"wrote {path}  E_total = {out['e_total']:.10f}  converged={out['converged']}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
