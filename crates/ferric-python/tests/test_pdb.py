"""PDB export (ferric.pdb): the records sit in the fixed PDB columns.

The column test slices each field where the PDB format puts it, so a writer
that shifts a field by one character fails it; the OpenMM test reads the text
back with an independent parser.
"""

from __future__ import annotations

import numpy as np
import pytest

ferric = pytest.importorskip("ferric")
pdb = pytest.importorskip("ferric.pdb")

HBR = [("H", (0.0, 0.0, 0.0)), ("Br", (-1.25, 10.5, 123.4567))]


def _mol(atoms=HBR, ghost=None, multiplicity=1):
    return ferric.Molecule.from_arrays(
        [s for s, _ in atoms],
        [xyz for _, xyz in atoms],
        multiplicity=multiplicity,
        ghost=ghost,
    )


def test_records_sit_in_the_pdb_columns():
    lines = pdb.to_pdb(_mol()).splitlines()
    assert lines[-1] == "END"
    assert len(lines) == len(HBR) + 1
    for serial, (line, (symbol, xyz)) in enumerate(zip(lines, HBR), start=1):
        assert len(line) == 78
        assert line[0:6] == "HETATM"
        assert int(line[6:11]) == serial
        # The element is right-justified in columns 13-14 of the name field,
        # followed by its per-element counter.
        assert line[12:16] == f"{symbol.upper():>2s}1 "
        assert line[17:20] == "MOL"
        assert line[21] == "A"
        assert int(line[22:26]) == 1
        got = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        # Three decimals are written: half a unit in the last place.
        np.testing.assert_allclose(got, xyz, atol=5.0e-4, rtol=0.0)
        assert float(line[54:60]) == 1.0
        assert line[76:78] == f"{symbol.upper():>2s}"


def test_ghost_atoms_are_left_out():
    # A ghost H leaves Br alone: 35 electrons, a doublet.
    text = pdb.to_pdb(_mol(ghost=[True, False], multiplicity=2))
    records = [line for line in text.splitlines() if line.startswith("HETATM")]
    assert len(records) == 1
    assert records[0][76:78] == "BR"
    assert int(records[0][6:11]) == 1


def test_a_coordinate_too_wide_for_the_field_is_refused():
    with pytest.raises(ValueError, match="does not fit"):
        pdb.to_pdb(_mol([("He", (0.0, 0.0, 12345.0))]))


WATER = [
    ("O", (0.0, 0.0, 0.117)),
    ("H", (0.0, 0.757, -0.469)),
    ("H", (0.0, -0.757, -0.469)),
]


def _names(text):
    return [(line[12:16], int(line[22:26])) for line in text.splitlines()[:-1]]


def test_repeated_elements_get_distinct_names():
    assert _names(pdb.to_pdb(_mol(WATER))) == [(" O1 ", 1), (" H1 ", 1), (" H2 ", 1)]


def test_a_new_residue_starts_when_the_name_counter_is_full():
    # A two-letter symbol leaves two columns for the counter: HE99 is the
    # last name of a residue, and the 100th helium opens residue 2.
    atoms = [("He", (0.0, 0.0, 3.0 * k)) for k in range(100)]
    names = _names(pdb.to_pdb(_mol(atoms)))
    assert names[8:10] == [("HE9 ", 1), ("HE10", 1)]
    assert names[98:] == [("HE99", 1), ("HE1 ", 2)]
    assert len(set(names)) == 100


def test_write_pdb_writes_to_pdb_text(tmp_path):
    path = tmp_path / "hbr.pdb"
    pdb.write_pdb(path, _mol())
    assert path.read_text() == pdb.to_pdb(_mol())


def test_a_refused_molecule_leaves_the_existing_file_alone(tmp_path):
    path = tmp_path / "kept.pdb"
    path.write_text("kept")
    with pytest.raises(ValueError, match="does not fit"):
        pdb.write_pdb(path, _mol([("He", (0.0, 0.0, 12345.0))]))
    assert path.read_text() == "kept"


def test_openmm_reads_the_file_back(tmp_path):
    app = pytest.importorskip("openmm.app", reason="OpenMM not installed")
    from openmm import unit

    path = tmp_path / "hbr.pdb"
    pdb.write_pdb(path, _mol())
    parsed = app.PDBFile(str(path))
    atoms = list(parsed.topology.atoms())
    assert [a.element.symbol for a in atoms] == ["H", "Br"]
    got = np.array(parsed.positions.value_in_unit(unit.angstrom))
    np.testing.assert_allclose(got, [xyz for _, xyz in HBR], atol=5.0e-4, rtol=0.0)


def test_openmm_keeps_every_atom_of_a_repeated_element(tmp_path):
    """A reader that keys atoms by name within a residue must see both
    hydrogens of water."""
    app = pytest.importorskip("openmm.app", reason="OpenMM not installed")
    from openmm import unit

    path = tmp_path / "water.pdb"
    pdb.write_pdb(path, _mol(WATER))
    parsed = app.PDBFile(str(path))
    assert [a.element.symbol for a in parsed.topology.atoms()] == ["O", "H", "H"]
    got = np.array(parsed.positions.value_in_unit(unit.angstrom))
    np.testing.assert_allclose(got, [xyz for _, xyz in WATER], atol=5.0e-4, rtol=0.0)
