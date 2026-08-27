#!/usr/bin/env python3
"""Generate PySCF references for ferric's QM/MM electrostatic embedding under
KS-DFT (RKS and UKS) — the Lane E cross-check.

Same model as `scripts/gen_pyscf_qmmm_refs.py` (`pyscf.qmmm.mm_charge` folds
fixed classical point charges into hcore; no new physics), but the QM method
is now RKS/UKS instead of RHF/UHF, with settings copied VERBATIM from
`scripts/gen_pyscf_dft_refs.py` so the grid/RI-J/radii-adjust choices match
what ferric's own (non-QM/MM) DFT reference generator uses:

  * `mf.density_fit(auxbasis="def2-universal-jkfit")` — matches ferric's RI-J
    (and RI-K for the hybrid B3LYP case).
  * `mf.grids.atom_grid = (75, 110)`, `mf.grids.prune = None` — ferric's
    default Becke-Lebedev grid, unpruned.
  * `mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust` — matches
    ferric's Becke (1988) atomic-size correction (PySCF's default is
    Treutler).
  * `mf.conv_tol = 1e-10`.

Geometry (water/OH, the same off-axis MM charge set) matches
`gen_pyscf_qmmm_refs.py`'s `water_bohr()`/`oh_bohr()`/`CASES["offaxis"]`
bit-for-bit — same Å->Bohr recipe, PySCF fed the Bohr coordinates directly
(`unit="Bohr"`) so no second conversion happens on either side.

Gradient reference follows `gen_pyscf_dft_grad_refs.py`: `grid_response =
True` (PySCF's grid-coordinate + Becke-weight response), which ferric's
analytic KS gradient does NOT include — the gap between the two is exactly
the grid-response residual the existing dft_gradient_vs_pyscf.rs bar already
absorbs, so this script's job is only to feed the same settings through
`qmmm.mm_charge`.

Per case the reference records:
  e_total        total SCF energy INCLUDING the classical charge-nuclear term
  e_gas_phase    same QM atoms/basis/xc, no MM charges
  dipole         total dipole (a.u., origin at 0)
  mm_gradient    dE/dR of each MM charge (Hartree/Bohr); ferric's `mm_forces`
                 is the negative of this
  qm_gradient    dE/dR of each QM nucleus in the field (Hartree/Bohr),
                 PySCF grid_response=True

Usage:
    OPENBLAS_NUM_THREADS=1 ~/qc/ferric/.venv/bin/python scripts/gen_pyscf_qmmm_dft_refs.py
"""
import json
import math
from pathlib import Path

import numpy as np
import pyscf
from pyscf import dft, gto, qmmm

REFDIR = Path(__file__).resolve().parents[1] / "testdata" / "reference"
ANG2BOHR = 1.0 / 0.52917721092

MAIN_GRID = (75, 110)

PYSCF_XC = {
    "svwn": "LDA,VWN",
    "pbe": "PBE,PBE",
    "b3lyp": "B3LYP",
}


def water_bohr():
    """O at origin, H's at +z in the yz plane — same as gen_pyscf_qmmm_refs.py."""
    r = 0.9572 * ANG2BOHR
    half = math.radians(104.52) / 2.0
    return [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, r * math.sin(half), r * math.cos(half))),
        ("H", (0.0, -r * math.sin(half), r * math.cos(half))),
    ]


def oh_bohr():
    """OH radical along z, r = 0.97 Å (a doublet for the UKS case)."""
    return [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.97 * ANG2BOHR))]


# Off-axis fractional charges — same set as gen_pyscf_qmmm_refs.py's
# "offaxis" case: breaks C2v symmetry so every gradient component is a live
# comparison.
OFFAXIS_MM = [
    (-0.834, 3.1, -2.2, 4.0),
    (0.417, -2.5, 3.3, -3.7),
    (0.417, 1.7, 2.9, -5.1),
]
LONEPAIR_MM = [(1.0, 0.0, 0.0, -6.0)]

# (tag, atoms, charge, spin(2S), basis, mm charges)
CASES = [
    ("water_sto-3g_qmmm_dft_svwn", water_bohr(), 0, 0, "sto-3g", "svwn", OFFAXIS_MM),
    ("water_sto-3g_qmmm_dft_pbe", water_bohr(), 0, 0, "sto-3g", "pbe", OFFAXIS_MM),
    ("water_sto-3g_qmmm_dft_b3lyp", water_bohr(), 0, 0, "sto-3g", "b3lyp", OFFAXIS_MM),
    ("water_cc-pvdz_qmmm_dft_pbe", water_bohr(), 0, 0, "cc-pvdz", "pbe", OFFAXIS_MM),
    ("oh_sto-3g_uqmmm_dft_pbe", oh_bohr(), 0, 1, "sto-3g", "pbe", LONEPAIR_MM),
]


def _build_mf(mol, xc):
    mf = dft.UKS(mol) if mol.spin else dft.RKS(mol)
    mf.xc = PYSCF_XC[xc]
    mf = mf.density_fit(auxbasis="def2-universal-jkfit")
    mf.grids.atom_grid = MAIN_GRID
    mf.grids.prune = None
    mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
    mf.conv_tol = 1e-10
    return mf


def run_case(tag, atoms, charge, spin, basis, xc, mm):
    mol = gto.M(
        atom=[(s, xyz) for s, xyz in atoms],
        basis=basis,
        unit="Bohr",
        charge=charge,
        spin=spin,
        verbose=0,
    )
    coords = np.array([[x, y, z] for _, x, y, z in mm])
    charges = np.array([q for q, _, _, _ in mm])

    mf = _build_mf(mol, xc)
    mf = qmmm.mm_charge(mf, coords, charges, unit="Bohr")
    e = mf.kernel()
    assert mf.converged, tag

    dm = mf.make_rdm1()
    dm_tot = dm[0] + dm[1] if spin else dm
    ao_dip = mol.intor_symmetric("int1e_r", comp=3)
    el = -np.einsum("xij,ji->x", ao_dip, dm_tot)
    nuc = np.einsum("i,ix->x", mol.atom_charges(), mol.atom_coords())
    dip = el + nuc

    g = mf.nuc_grad_method()
    g.grid_response = True
    qm_grad = g.kernel()
    mm_grad = g.grad_hcore_mm(dm_tot) + g.grad_nuc_mm()

    # Gas-phase energy of the same QM atoms/basis/xc, for the shift.
    mol0 = gto.M(
        atom=[(s, xyz) for s, xyz in atoms],
        basis=basis, unit="Bohr", charge=charge, spin=spin, verbose=0,
    )
    mf0 = _build_mf(mol0, xc)
    e0 = mf0.kernel()
    assert mf0.converged, f"{tag}: gas phase"

    return {
        "molecule": tag.split("_")[0],
        "basis": basis,
        "xc": xc,
        "method": "uks" if spin else "rks",
        "main_grid": list(MAIN_GRID),
        "model": "electrostatic embedding, fixed point charges (pyscf.qmmm.mm_charge)",
        "units": "Bohr / Hartree / a.u.; gradients are dE/dR (ferric mm_forces = -mm_gradient)",
        "atoms": [{"symbol": s, "xyz_bohr": list(xyz)} for s, xyz in atoms],
        "charge": charge,
        "multiplicity": spin + 1,
        "mm_charges": [{"q": q, "xyz_bohr": [x, y, z]} for q, x, y, z in mm],
        "e_total": float(e),
        "e_gas_phase": float(e0),
        "dipole": [float(v) for v in dip],
        "qm_gradient": qm_grad.tolist(),
        "mm_gradient": mm_grad.tolist(),
        "grid_response": True,
        "converged": bool(mf.converged),
        "conv_tol": 1e-10,
        "pyscf_version": pyscf.__version__,
    }


def main():
    for tag, atoms, charge, spin, basis, xc, mm in CASES:
        ref = run_case(tag, atoms, charge, spin, basis, xc, mm)
        out = REFDIR / f"{tag}.json"
        out.write_text(json.dumps(ref, indent=2, sort_keys=True) + "\n")
        print(
            f"{out.name}: E = {ref['e_total']:.10f}  (gas {ref['e_gas_phase']:.10f})  "
            f"mu = {np.round(ref['dipole'], 6)}"
        )


if __name__ == "__main__":
    main()
