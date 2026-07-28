#!/usr/bin/env python3
"""Generate ferric meta-GGA (SCAN / r2SCAN) reference energies from PySCF.

This is the FIRST meta-GGA reference in the ferric test suite (Phase A:
SCF single-point energy only — no gradients). It mirrors
`gen_pyscf_dft_refs.py`'s ferric-matching settings (RI-J Coulomb via
def2-universal-jkfit, ferric's flat (75, 110) Becke-Lebedev grid with no
pruning, Becke-1988 atomic-size adjustment) so the only remaining
differences are the semilocal kernel itself (τ-dependent) and grid-integration
noise.

The exact PySCF call is captured in `run_one` below; each JSON records
e_total, e_nuc, grids, and convergence.

Usage:
    python scripts/gen_pyscf_mgga_refs.py            # SCAN + r2SCAN
    python scripts/gen_pyscf_mgga_refs.py scan       # just SCAN
    python scripts/gen_pyscf_mgga_refs.py r2scan     # just r2SCAN
"""
import json
import sys
from pathlib import Path

from pyscf import dft, gto

REFDIR = Path(__file__).resolve().parents[1] / "testdata" / "reference"

# Same coords as the LDA/PBE reference set (Angstrom).
MOLECULES = {
    "h2":      ("H 0 0 0; H 0 0 0.74",                                 0, 1),
    "h2o":     ("O 0 0 0; H 0 0.7572 0.5868; H 0 -0.7572 0.5868",      0, 1),
    "methane": ("C 0 0 0;"
                " H 0.6276 0.6276 0.6276;"
                " H -0.6276 -0.6276 0.6276;"
                " H -0.6276 0.6276 -0.6276;"
                " H 0.6276 -0.6276 -0.6276",                            0, 1),
    # Open shell (UKS), added 2026-07-27. These pin the SPIN-POLARIZED
    # meta-GGA energy — the path with ρ_α ≠ ρ_β and τ_α ≠ τ_β, which the
    # closed-shell cases above cannot exercise at all.
    #
    # Deliberately NOT OH: OH's β-HOMO converges to ~-4e-4 Ha (near-zero gap),
    # so its SCF surface is near-flat and PySCF's own OH/SCAN energy shifts by
    # 1.9e-5 Ha for a conv_tol change alone. NH2 / CH3 / O2 are well-behaved
    # and reproduce ferric-vs-PySCF to ~1e-8.
    "nh2": ("N 0 0 0.1414; H 0 0.8067 -0.4950; H 0 -0.8067 -0.4950",   0, 2),
    "ch3": ("C 0 0 0; H 0 1.0790 0;"
            " H 0.9344 -0.5395 0; H -0.9344 -0.5395 0",                 0, 2),
    "o2":  ("O 0 0 0; O 0 0 1.2075",                                    0, 3),
}

# Ferric default main grid: (75, 110). Match exactly with no pruning.
MAIN_GRID = (75, 110)

# libxc functional strings. SCAN = MGGA_X_SCAN + MGGA_C_SCAN;
# r2SCAN = MGGA_X_R2SCAN + MGGA_C_R2SCAN — the same component pairs ferric's
# friendly-name resolver maps "SCAN" / "r2SCAN" to.
PYSCF_XC = {
    "scan":   "SCAN",
    "r2scan": "R2SCAN",
}


def run_one(label, atom_spec, charge, spin, basis, xc):
    mol = gto.M(atom=atom_spec, basis=basis, charge=charge,
                spin=spin - 1, unit="Angstrom")
    mf = (dft.RKS if spin == 1 else dft.UKS)(mol, xc=PYSCF_XC[xc])
    # Match ferric's RI-J for the Coulomb piece (ferric DFT uses DF-J).
    mf = mf.density_fit(auxbasis="def2-universal-jkfit")
    mf.grids.atom_grid = MAIN_GRID
    mf.grids.prune = None
    # Match ferric's Becke (1988) size correction (not PySCF's default Treutler).
    mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    mf.conv_tol = 1e-10
    e_total = mf.kernel()
    e_nuc = mol.energy_nuc()
    return {
        "label": label,
        "basis": basis,
        "xc": xc,
        "charge": charge,
        "multiplicity": spin,
        "main_grid": list(MAIN_GRID),
        "nlc_grid": None,
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
            basis = "cc-pvdz"
            out = run_one(label, atom_spec, charge, spin, basis, xc)
            fname = f"{label}_{basis}_{xc}.json"
            path = REFDIR / fname
            path.write_text(json.dumps(out, indent=2))
            print(f"wrote {path}  E_total = {out['e_total']:.10f}  "
                  f"converged={out['converged']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
