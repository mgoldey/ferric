"""PDB export (ferric.pdb)."""

from __future__ import annotations

import pytest

ferric = pytest.importorskip("ferric")


def test_to_pdb_has_atom_records():
    import ferric.pdb as pdb

    mol = ferric.Molecule.from_xyz_string("2\n\nH 0 0 0\nH 0 0 0.74\n")
    text = pdb.to_pdb(mol)
    assert text.startswith("ATOM")
    assert "END" in text
    assert text.count("ATOM") == 2


def test_write_pdb(tmp_path):
    import ferric.pdb as pdb

    mol = ferric.Molecule.from_xyz_string("1\n\nHe 0 0 0\n")
    path = tmp_path / "he.pdb"
    pdb.write_pdb(str(path), mol)
    assert path.read_text().count("ATOM") == 1
