#!/usr/bin/env python3
"""Generate PySCF references for ferric's QM/MM electrostatic embedding.

Cross-code anchor for `crates/ferric-scf/tests/qmmm.rs`: until this file
existed every QM/MM test there was self-consistency only (sign of the
response, finite difference of ferric's own energy). This script uses
`pyscf.qmmm.mm_charge` — an independent implementation of exactly the same
model (fixed classical point charges folded into hcore, charge-nuclear
Coulomb term added to the energy) — to pin the absolute numbers.

Geometry matches `water_atoms()` in the ferric test BIT-FOR-BIT: the Bohr
coordinates are built from the same r(OH) = 0.9572 Å / 104.52° recipe with
the same Å->Bohr constant (1/0.52917721092, which is also PySCF's BOHR), and
PySCF is fed those Bohr values directly (unit="Bohr") so no second
conversion happens on either side.

Per case the reference records:
  energy         total SCF energy INCLUDING the classical charge-nuclear term
  dipole         total dipole (a.u., origin at 0) — the polarization check
  mm_gradient    dE/dR of each MM charge (Hartree/Bohr); ferric's `mm_forces`
                 returns the FORCE, i.e. the negative of this
  qm_gradient    dE/dR of each QM nucleus in the field (Hartree/Bohr)

Usage:
    OPENBLAS_NUM_THREADS=1 uv run --no-sync python scripts/gen_pyscf_qmmm_refs.py
"""
import json
import math
from pathlib import Path

import numpy as np
import pyscf
from pyscf import gto, mp, qmmm, scf

REFDIR = Path(__file__).resolve().parents[1] / "testdata" / "reference"
ANG2BOHR = 1.0 / 0.52917721092


def water_bohr():
    """O at origin, H's at +z in the yz plane — same as the ferric test."""
    r = 0.9572 * ANG2BOHR
    half = math.radians(104.52) / 2.0
    return [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, r * math.sin(half), r * math.cos(half))),
        ("H", (0.0, -r * math.sin(half), r * math.cos(half))),
    ]


def oh_bohr():
    """OH radical along z, r = 0.97 Å (a doublet for the UHF case)."""
    return [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.97 * ANG2BOHR))]


# (tag, atoms, charge, spin(2S), MM charges as [(q, x, y, z)] in Bohr)
CASES = [
    ("water_sto-3g_qmmm_plus_lonepair", water_bohr(), 0, 0, [(1.0, 0.0, 0.0, -6.0)]),
    ("water_sto-3g_qmmm_plus_hside", water_bohr(), 0, 0, [(1.0, 0.0, 0.0, 6.0)]),
    ("water_sto-3g_qmmm_two_charges", water_bohr(), 0, 0,
     [(1.0, 0.0, 0.0, -6.0), (-1.0, 0.0, 0.0, 9.0)]),
    # Off-axis fractional charges: breaks the C2v symmetry the on-axis cases
    # keep, so every gradient component is nonzero and a transposed or
    # sign-flipped component cannot hide behind a zero.
    ("water_sto-3g_qmmm_offaxis", water_bohr(), 0, 0,
     [(-0.834, 3.1, -2.2, 4.0), (0.417, -2.5, 3.3, -3.7), (0.417, 1.7, 2.9, -5.1)]),
    ("oh_sto-3g_uqmmm_plus_lonepair", oh_bohr(), 0, 1, [(1.0, 0.0, 0.0, -6.0)]),
]


def run_case(tag, atoms, charge, spin, mm, with_mp2=False):
    mol = gto.M(
        atom=[(s, xyz) for s, xyz in atoms],
        basis="sto-3g",
        unit="Bohr",
        charge=charge,
        spin=spin,
        verbose=0,
    )
    coords = np.array([[x, y, z] for _, x, y, z in mm])
    charges = np.array([q for q, _, _, _ in mm])
    mf = scf.UHF(mol) if spin else scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf = qmmm.mm_charge(mf, coords, charges, unit="Bohr")
    e = mf.kernel()
    assert mf.converged, tag
    dm = mf.make_rdm1()
    if spin:
        dm_tot = dm[0] + dm[1]
    else:
        dm_tot = dm
    # Total dipole about the origin, a.u.
    ao_dip = mol.intor_symmetric("int1e_r", comp=3)
    el = -np.einsum("xij,ji->x", ao_dip, dm_tot)
    nuc = np.einsum("i,ix->x", mol.atom_charges(), mol.atom_coords())
    dip = el + nuc

    # A QMMM-wrapped SCF object hands back the QMMM-aware gradient class for
    # both RHF and UHF (the explicit mm_charge_grad asserts on UHF).
    g = mf.nuc_grad_method()
    qm_grad = g.kernel()
    # grad_hcore_mm wants the spin-summed density (a one-electron property).
    mm_grad = g.grad_hcore_mm(dm_tot) + g.grad_nuc_mm()

    # Gas-phase energy of the same QM atoms, for the shift.
    mf0 = scf.UHF(mol) if spin else scf.RHF(mol)
    mf0.conv_tol = 1e-12
    e0 = mf0.kernel()
    assert mf0.converged

    ref = {
        "molecule": tag.split("_")[0],
        "basis": "sto-3g",
        "method": "uhf" if spin else "rhf",
        "model": "electrostatic embedding, fixed point charges (pyscf.qmmm.mm_charge)",
        "units": "Bohr / Hartree / a.u.; gradients are dE/dR (ferric mm_forces = -mm_gradient)",
        "atoms": [{"symbol": s, "xyz_bohr": list(xyz)} for s, xyz in atoms],
        "charge": charge,
        "multiplicity": spin + 1,
        "mm_charges": [{"q": q, "xyz_bohr": [x, y, z]} for q, x, y, z in mm],
        "energy": float(e),
        "energy_gas_phase": float(e0),
        "dipole": [float(v) for v in dip],
        "qm_gradient": qm_grad.tolist(),
        "mm_gradient": mm_grad.tolist(),
        "converged": bool(mf.converged),
        "conv_tol": 1e-12,
        "pyscf_version": pyscf.__version__,
    }

    if with_mp2:
        # Canonical (non-RI) MP2 on the SAME QMMM-wrapped SCF object — the
        # gradient class propagates the QM/MM state automatically (it wraps
        # the QMMM mean-field the same way the RHF/UHF gradient does), so no
        # separate qmmm.mm_charge_grad() call is needed for the QM gradient.
        # qmmm.mm_charge_grad() itself REJECTS an MP2 gradient object
        # (AssertionError on the class check), and pyscf.grad.mp2.Gradients
        # has no grad_hcore_mm/grad_nuc_mm — so no MM-side gradient is
        # recorded here (energy + QM gradient only, per the F1-2 plan).
        mpobj = mp.MP2(mf)
        mpobj.conv_tol = 1e-10
        mpobj.run()
        mp2_g = mpobj.nuc_grad_method()
        mp2_qm_grad = mp2_g.kernel()
        ref["mp2_energy"] = float(mpobj.e_tot)
        ref["mp2_corr"] = float(mpobj.e_corr)
        ref["mp2_qm_gradient"] = mp2_qm_grad.tolist()

    return ref


def main():
    mp2_tags = {"water_sto-3g_qmmm_plus_lonepair"}
    for tag, atoms, charge, spin, mm in CASES:
        ref = run_case(tag, atoms, charge, spin, mm, with_mp2=tag in mp2_tags)
        out = REFDIR / f"{tag}.json"
        out.write_text(json.dumps(ref, indent=2, sort_keys=True) + "\n")
        msg = (f"{out.name}: E = {ref['energy']:.10f}  (gas {ref['energy_gas_phase']:.10f})  "
               f"mu = {np.round(ref['dipole'], 6)}  F_mm = {np.round(-np.array(ref['mm_gradient']), 6).tolist()}")
        if "mp2_energy" in ref:
            msg += f"  MP2 total = {ref['mp2_energy']:.10f} (corr {ref['mp2_corr']:.10f})"
        print(msg)


if __name__ == "__main__":
    main()
