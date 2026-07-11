from pathlib import Path

from tools.active_site.pqr_parser import ANGSTROM_TO_BOHR, parse_pqr

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
