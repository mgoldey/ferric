from pathlib import Path

from tools.active_site.pqr_parser import ANGSTROM_TO_BOHR, parse_pqr, parse_pqr_atoms

FIXTURE = Path(__file__).parent / "fixture.pqr"


def test_parse_pqr_field_count_and_charges():
    charges = parse_pqr(FIXTURE)
    assert len(charges) == 5
    q, x, y, z = charges[0]
    assert q == 0.1812
    assert x == 17.047 * ANGSTROM_TO_BOHR
    assert y == 14.099 * ANGSTROM_TO_BOHR
    assert z == 3.625 * ANGSTROM_TO_BOHR


def test_parse_pqr_handles_wide_atom_name():
    # HG21 pushes the atom-name column wider than standard PDB; must still
    # parse as exactly 10 whitespace fields.
    charges = parse_pqr(FIXTURE)
    q, x, y, z = charges[-1]
    assert q == 0.0627


def test_parse_pqr_ignores_ter_and_end():
    charges = parse_pqr(FIXTURE)
    assert len(charges) == 5  # TER/END lines must not be parsed as atoms


# ── parse_pqr_atoms (residue-aware) ──


def test_parse_pqr_atoms_matches_parse_pqr_as_an_anchor():
    # Anchor: the (q, x, y, z) view must be bit-identical between the two
    # parsers — parse_pqr_atoms is a superset, not a different computation.
    plain = parse_pqr(FIXTURE)
    atoms = parse_pqr_atoms(FIXTURE)
    assert len(atoms) == len(plain)
    for a, (q, x, y, z) in zip(atoms, plain):
        assert a.q == q
        assert a.x == x
        assert a.y == y
        assert a.z == z


def test_parse_pqr_atoms_carries_residue_and_name_fields():
    atoms = parse_pqr_atoms(FIXTURE)
    a0 = atoms[0]
    assert a0.serial == 1
    assert a0.name == "N"
    assert a0.res_name == "THR"
    assert a0.res_seq == 1
    assert a0.radius == 1.8240 * ANGSTROM_TO_BOHR
    # every atom in the single-residue fixture shares the same res_seq
    assert {a.res_seq for a in atoms} == {1}
    # wide atom name (HG21) still parses whole
    assert atoms[-1].name == "HG21"
    assert atoms[-1].serial == 11
