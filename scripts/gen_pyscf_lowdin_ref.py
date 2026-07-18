"""
Generate PySCF reference Löwdin (symmetric S^{1/2}DS^{1/2}) atomic charges
for cross-checking ferric's `ferric_rpa::properties::lowdin_charges`.

Löwdin population analysis (symmetric orthogonalization, NOT meta-lowdin):
    S^{1/2} = U diag(sqrt(lambda)) U^T   (S = U diag(lambda) U^T)
    M = S^{1/2} D S^{1/2}
    atom_pop[A] = sum_{mu on A} M_{mu,mu}
    charge[A]   = Z_A - atom_pop[A]

This is computed two independent ways as a cross-check:
  1. Direct eigh-based S^{1/2} construction (matches ferric's own approach
     in crates/ferric-rpa/src/properties.rs::lowdin_charges exactly).
  2. Via pyscf.lo.orth_ao(method='lowdin', pre_orth_ao=None), which returns
     C_orth = S^{-1/2}; population in the orthogonal-AO basis is
     diag(C_orth^T S D S C_orth) = diag(S^{1/2} D S^{1/2}) since S is
     symmetric -- same quantity via PySCF's own library path (guards
     against a hand-rolled eigh bug being self-consistent nonsense).

IMPORTANT: PySCF's `mf.mulliken_pop` is Mulliken, NOT Löwdin -- do not
substitute it. `pyscf.lo.orth_ao` defaults to method='meta_lowdin' (core/
valence/Rydberg-partitioned) which is a DIFFERENT quantity from plain
symmetric Löwdin; method='lowdin' must be passed explicitly.

CRITICAL (fixed 2026-07-17, task F1): the basis is loaded from ferric's
OWN bundled JSON (`crates/ferric-core/src/basis/bundled/{basis}.json`) and
fed to PySCF as an explicit basis dict -- NOT the string `basis='cc-pvdz'`
(which makes PySCF use its OWN internal cc-pvdz.dat, a *segmented*
re-expression that DROPS the most-diffuse s/p primitive from the tight/
medium general-contraction columns). Löwdin populations are AO-identity-
dependent: PySCF's segmented table and ferric's/BSE's authentic Dunning
general contraction span the same variational space (SCF energies agree to
~2e-13) but are DIFFERENT individual basis functions, so a Löwdin
reference built from PySCF's string-loaded basis is NOT comparable to
ferric atom-by-atom. See docs/basis-data-corrections.md and the F1
resolution in docs/open-work-triage-2026-07-14-open.md item #7.

Writes JSON to testdata/reference/{mol}_{basis}_lowdin.json.

Usage:
  python scripts/gen_pyscf_lowdin_ref.py water cc-pvdz
  python scripts/gen_pyscf_lowdin_ref.py methane cc-pvdz
  python scripts/gen_pyscf_lowdin_ref.py h2 cc-pvdz
"""
import json
import os
import sys

import numpy as np
from pyscf import gto, lo, scf
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
    element symbol. This is the crux of task F1: the Löwdin reference MUST be
    generated from the identical basis functions ferric uses, not PySCF's
    internal (segmented) table for the same name."""
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
        # Same geometry as gen_pyscf_rpa_props.py: 1.4 Bohr along z.
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


def lowdin_charges_direct(mol, dm):
    """Hand-rolled symmetric Löwdin, mirroring ferric's eigh-based path
    exactly (S = U diag(lambda) U^T -> S^{1/2} = U diag(sqrt(lambda)) U^T)."""
    s = mol.intor_symmetric("int1e_ovlp")
    lam, u = np.linalg.eigh(s)
    if np.any(lam <= 0):
        raise RuntimeError(f"non-positive overlap eigenvalue(s): {lam}")
    s_half = (u * np.sqrt(lam)) @ u.T
    m = s_half @ dm @ s_half

    charges = []
    ao_labels = mol.ao_labels(fmt=None)  # list of (atom_idx, elem, shell, ao)
    atom_pop = np.zeros(mol.natm)
    for mu, (atom_idx, *_rest) in enumerate(ao_labels):
        atom_pop[atom_idx] += m[mu, mu]
    charges_arr = mol.atom_charges() - atom_pop
    return charges_arr.tolist()


def lowdin_charges_via_orth_ao(mol, dm):
    """Cross-check via pyscf.lo.orth_ao(method='lowdin', pre_orth_ao=None).

    c_orth = S^{-1/2}.  Population in orthogonal-AO basis:
        P_orth = C_orth^T S D S C_orth
    diag(P_orth) equals diag(S^{1/2} D S^{1/2}) since S symmetric and
    C_orth = S^{-1/2}, so C_orth^T S = S^{-1/2} S = S^{1/2}.
    """
    s = mol.intor_symmetric("int1e_ovlp")
    c_orth = lo.orth_ao(mol, method="lowdin", pre_orth_ao=None, s=s)
    p_orth = c_orth.T @ s @ dm @ s @ c_orth

    ao_labels = mol.ao_labels(fmt=None)
    atom_pop = np.zeros(mol.natm)
    for mu, (atom_idx, *_rest) in enumerate(ao_labels):
        atom_pop[atom_idx] += p_orth[mu, mu]
    charges_arr = mol.atom_charges() - atom_pop
    return charges_arr.tolist()


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "water"
    basis = sys.argv[2] if len(sys.argv) > 2 else "cc-pvdz"

    mol = build_mol(label, basis)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    dm = mf.make_rdm1()

    q_direct = lowdin_charges_direct(mol, dm)
    q_orth_ao = lowdin_charges_via_orth_ao(mol, dm)

    max_diff = max(abs(a - b) for a, b in zip(q_direct, q_orth_ao))
    print(f"max |direct - orth_ao| = {max_diff:.3e}")
    assert max_diff < 1e-9, "two independent PySCF Löwdin paths disagree!"

    out = {
        "molecule": label,
        "basis": basis,
        "scf_energy": float(mf.e_tot),
        "lowdin_charges": [float(x) for x in q_direct],
        "lowdin_charges_cross_check_orth_ao": [float(x) for x in q_orth_ao],
        "charge_sum": float(sum(q_direct)),
        "note": (
            "Symmetric Lowdin population charges (S^1/2 D S^1/2), NOT "
            "meta-lowdin and NOT Mulliken. Basis loaded from ferric's OWN "
            "bundled JSON (authentic Dunning general contraction), NOT "
            "PySCF's internal segmented cc-pvdz.dat -- see task F1 / "
            "docs/basis-data-corrections.md. Two independent PySCF code "
            "paths (hand eigh vs pyscf.lo.orth_ao(method='lowdin')) agree "
            f"to {max_diff:.1e}."
        ),
    }

    ref_dir = os.path.join(ROOT, "testdata/reference")
    os.makedirs(ref_dir, exist_ok=True)
    out_path = os.path.join(ref_dir, f"{label}_{basis}_lowdin.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
