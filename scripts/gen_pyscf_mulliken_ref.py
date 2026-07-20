"""
Generate PySCF reference Mulliken atomic charges for cross-checking
ferric's `ferric_rpa::properties::mulliken_charges`.

Mulliken population analysis (the textbook, unambiguous definition -- unlike
Lowdin there is no meta/orthogonalization variant to disambiguate):
    M = D @ S
    atom_pop[A] = sum_{mu on A} M_{mu,mu}
    charge[A]   = Z_A - atom_pop[A]

Cross-checked two independent ways:
  1. Direct hand-rolled M = D @ S (matches ferric's own approach in
     crates/ferric-rpa/src/properties.rs::mulliken_charges exactly).
  2. PySCF's own `mf.mulliken_pop()` (its library path).

CRITICAL (same lesson as task F1 / gen_pyscf_lowdin_ref.py): the basis is
loaded from ferric's OWN bundled JSON and fed to PySCF as an explicit basis
dict -- NOT the string `basis='cc-pvdz'` (PySCF's internal table is a
*segmented* re-expression of Dunning's general contraction; different AO
identity means a string-loaded reference is not comparable to ferric
atom-by-atom, even though both span the same variational space). See
docs/basis-data-corrections.md.

Writes JSON to testdata/reference/{mol}_{basis}_mulliken.json.

Usage:
  python scripts/gen_pyscf_mulliken_ref.py water cc-pvdz
  python scripts/gen_pyscf_mulliken_ref.py methane cc-pvdz
  python scripts/gen_pyscf_mulliken_ref.py h2 cc-pvdz
"""
import json
import os
import sys

import numpy as np
from pyscf import gto, scf
from pyscf.data.elements import ELEMENTS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _bse_elem_to_pyscf(elem):
    """Convert one element's BSE `electron_shells` (as stored in ferric's
    bundled JSON) into PySCF's nested-list per-element basis format:
    `[l, [exp, c_col0, c_col1, ...], [exp, ...], ...]` per shell."""
    shells = []
    for sh in elem["electron_shells"]:
        exps = [float(x) for x in sh["exponents"]]
        cols = [[float(x) for x in col] for col in sh["coefficients"]]
        for l in sh["angular_momentum"]:
            block = [l]
            for i, e in enumerate(exps):
                block.append([e] + [cols[j][i] for j in range(len(cols))])
            shells.append(block)
    return shells


def load_ferric_basis(basis):
    """Load ferric's OWN bundled basis JSON as a PySCF basis dict, keyed by
    element symbol -- same crux as the Lowdin reference script: the Mulliken
    reference MUST be generated from the identical basis functions ferric
    uses, not PySCF's internal (segmented) table for the same name."""
    path = os.path.join(
        ROOT, "crates/ferric-core/src/basis/bundled", f"{basis}.json"
    )
    with open(path) as fh:
        d = json.load(fh)
    return {
        ELEMENTS[int(z)]: _bse_elem_to_pyscf(elem)
        for z, elem in d["elements"].items()
    }


def build_mol(label, basis):
    if label == "h2":
        # Same geometry as gen_pyscf_lowdin_ref.py / gen_pyscf_rpa_props.py.
        atom = "H 0 0 0; H 0 0 0.74083"
    else:
        xyz_path = os.path.join(ROOT, "testdata/molecules", f"{label}.xyz")
        with open(xyz_path) as fh:
            lines = fh.read().splitlines()
        body = "; ".join(lines[2:])
        atom = body
    ferric_basis = load_ferric_basis(basis)
    return gto.M(
        atom=atom, basis=ferric_basis, unit="Angstrom", charge=0, spin=0
    )


def mulliken_charges_direct(mol, dm):
    """Hand-rolled Mulliken, mirroring ferric's M = D @ S path exactly."""
    s = mol.intor_symmetric("int1e_ovlp")
    m = dm @ s

    ao_labels = mol.ao_labels(fmt=None)  # list of (atom_idx, elem, shell, ao)
    atom_pop = np.zeros(mol.natm)
    for mu, (atom_idx, *_rest) in enumerate(ao_labels):
        atom_pop[atom_idx] += m[mu, mu]
    charges_arr = mol.atom_charges() - atom_pop
    return charges_arr.tolist()


def mulliken_charges_via_pyscf(mf, mol, dm):
    """Cross-check via PySCF's own mf.mulliken_pop() library path."""
    _pop, charges_arr = mf.mulliken_pop(mol, dm, verbose=0)
    return charges_arr.tolist()


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "water"
    basis = sys.argv[2] if len(sys.argv) > 2 else "cc-pvdz"

    mol = build_mol(label, basis)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    dm = mf.make_rdm1()

    q_direct = mulliken_charges_direct(mol, dm)
    q_pyscf = mulliken_charges_via_pyscf(mf, mol, dm)

    max_diff = max(abs(a - b) for a, b in zip(q_direct, q_pyscf))
    print(f"max |direct - pyscf.mulliken_pop| = {max_diff:.3e}")
    assert max_diff < 1e-9, "hand-rolled and PySCF's own Mulliken disagree!"

    out = {
        "molecule": label,
        "basis": basis,
        "scf_energy": float(mf.e_tot),
        "mulliken_charges": [float(x) for x in q_direct],
        "mulliken_charges_cross_check_pyscf": [float(x) for x in q_pyscf],
        "charge_sum": float(sum(q_direct)),
        "note": (
            "Standard Mulliken population charges (D @ S diagonal), the "
            "unambiguous textbook definition (no meta/orthogonalization "
            "variant, unlike Lowdin). Basis loaded from ferric's OWN "
            "bundled JSON (authentic Dunning general contraction), NOT "
            "PySCF's internal segmented cc-pvdz.dat -- see "
            "docs/basis-data-corrections.md. Two independent code paths "
            "(hand D@S vs pyscf.scf.hf.mulliken_pop) agree to "
            f"{max_diff:.1e}."
        ),
    }

    ref_dir = os.path.join(ROOT, "testdata/reference")
    os.makedirs(ref_dir, exist_ok=True)
    out_path = os.path.join(ref_dir, f"{label}_{basis}_mulliken.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
