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


def test_an_eleven_field_pqr_is_refused_not_misparsed(tmp_path):
    """A chain-ID PQR must ERROR, and the error must name the field count.

    `parse_pqr` requires exactly 10 whitespace fields and reads coordinates at
    `fields[5:9]`. A PQR carrying a chain ID has ELEVEN and shifts them to
    `fields[6:10]`, so parsing it positionally would read the RESIDUE NUMBER as
    x and silently place every charge somewhere else. That is the failure this
    pins against -- a clean refusal is correct here, a best-effort parse is not.

    WHY IT IS ONLY A REFUSAL AND NOT A FEATURE. MEASURED 2026-09-19: `pdb2pqr30`
    DROPS the chain ID. Fed a PDB whose ATOM records carry chain A, all 16
    output records came back 10-field. So the 11-field layout is not reachable
    through `derive_pocket_charges`, which is this repo's only producer, and
    adding support for it would be speculative. It IS reachable from a PQR
    written by another tool, and then the user gets an error that names the
    problem rather than coordinates off by one column.

    If someone later adds 11-field support, this test should be REPLACED by one
    asserting the coordinates land correctly -- not deleted.
    """
    import pytest

    from tools.active_site.pqr_parser import parse_pqr

    eleven = tmp_path / "chain.pqr"
    eleven.write_text(
        # ATOM serial name res CHAIN resnum x y z q radius  <- 11 fields
        "ATOM      1  N   THR A   1      17.047  14.099   3.625  0.1812 1.8240\n"
    )
    with pytest.raises(ValueError, match=r"field count \(11, expected 10\)"):
        parse_pqr(str(eleven))

    # THE ANCHOR: the same record WITHOUT the chain ID must parse, so the test
    # above is about the extra column and not about the file being malformed.
    ten = tmp_path / "nochain.pqr"
    ten.write_text(
        "ATOM      1  N   THR     1      17.047  14.099   3.625  0.1812 1.8240\n"
    )
    charges = parse_pqr(str(ten))
    assert len(charges) == 1
    q, x, _y, _z = charges[0]
    assert q == pytest.approx(0.1812)
    # 17.047 A in Bohr -- confirms the coordinate columns were read, not the
    # residue number (which would give 1.0).
    assert x == pytest.approx(17.047 * 1.8897261254578281, rel=1e-9)
