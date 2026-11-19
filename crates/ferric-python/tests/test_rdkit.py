"""`ferric.rdkit`: RDKit <-> ferric.Molecule. Skips without RDKit (optional)."""

from __future__ import annotations

import numpy as np
import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")
Chem = pytest.importorskip("rdkit.Chem", reason="RDKit is an optional dependency")

import ferric.rdkit as fr  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402


def _embedded(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 7
    assert AllChem.EmbedMolecule(mol, params) == 0
    return mol


def test_coordinates_preserved_exactly():
    rd = _embedded("CCO")
    mol = fr.from_rdkit(rd)
    # ferric applies Angstrom -> Bohr once; compare in Angstrom to 1e-12.
    ref = np.asarray(rd.GetConformer().GetPositions())
    assert np.abs(np.asarray(mol.coords()) - ref).max() < 1e-12
    assert mol.symbols() == [a.GetSymbol() for a in rd.GetAtoms()]


def test_formal_charge_is_read():
    rd = _embedded("[NH4+]")
    mol = fr.from_rdkit(rd)
    assert mol.nelec() == 10


def test_charge_override_changes_electrons():
    rd = _embedded("[NH4+]")
    assert fr.from_rdkit(rd, charge=1).nelec() == 10
    with pytest.raises(ValueError):
        fr.from_rdkit(rd, charge=0)  # 11 electrons cannot be a singlet


def test_missing_conformer_raises():
    rd = Chem.AddHs(Chem.MolFromSmiles("CO"))
    with pytest.raises(ValueError, match="no conformer"):
        fr.from_rdkit(rd)


def test_bad_conf_id_raises():
    with pytest.raises(ValueError, match="conformer 5"):
        fr.from_rdkit(_embedded("CO"), conf_id=5)


def test_from_smiles_is_deterministic_and_seeded():
    a = fr.from_smiles("CC(=O)O")
    b = fr.from_smiles("CC(=O)O")
    assert a.coords_bohr() == b.coords_bohr()
    assert a.symbols().count("H") == 4
    unopt = fr.from_smiles("CCCC", optimize=False)
    assert unopt.coords_bohr() != fr.from_smiles("CCCC").coords_bohr()


def test_from_smiles_charge_and_errors():
    assert fr.from_smiles("[NH4+]").nelec() == 10
    with pytest.raises(ValueError, match="parse"):
        fr.from_smiles("not a smiles(")


def test_round_trip_symbols_and_coords():
    mol = fr.from_smiles("CCO")
    rd = fr.to_rdkit(mol, charge=0)
    back = fr.from_rdkit(rd)
    assert back.symbols() == mol.symbols()
    assert np.abs(np.asarray(back.coords()) - np.asarray(mol.coords())).max() < 1e-12
    assert rd.GetNumBonds() == 8  # bonds perceived: C-C, C-O, 6 C/O-H


def test_to_rdkit_without_bonds_and_charge():
    mol = fr.from_smiles("[NH4+]")
    assert fr.to_rdkit(mol, perceive_bonds=False).GetNumBonds() == 0
    rd = fr.to_rdkit(mol)  # charge taken from the Molecule
    assert Chem.GetFormalCharge(rd) == 1


def test_to_rdkit_rejects_ghosts():
    mol = ferric.Molecule.from_arrays(
        ["H", "H", "He"],
        [[0, 0, 0], [0, 0, 0.74], [0, 0, 3.0]],
        ghost=[False, False, True],
    )
    with pytest.raises(ValueError, match="ghost"):
        fr.to_rdkit(mol)


def test_radical_smiles_defaults_to_doublet_and_explicit_overrides():
    mol = fr.from_smiles("[CH3]")
    assert mol.multiplicity == 2
    assert fr.from_smiles("[CH3]", multiplicity=4).multiplicity == 4
    assert fr.from_smiles("CCO").multiplicity == 1
