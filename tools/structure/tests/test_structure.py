"""Tests for the multi-format structure reader.

The load-bearing claim this module makes is in its docstring: every format
converges on ONE Rust parser, so two files describing the same geometry give a
bit-identical `Molecule`. `test_pdb_and_xyz_agree_bitwise` is that claim's
falsifier -- it fails if anyone adds a second conversion path or a unit
constant of their own.

Tests that need the compiled `ferric` extension skip cleanly when it is absent;
the format layer itself is tested through `read_structure`, which never imports
it, so a missing extension does not silently delete coverage of the parsers.
"""

from __future__ import annotations

import math
import textwrap

import pytest

from tools.structure import (
    MissingBackend,
    Structure,
    StructureError,
    read_structure,
)


def _have_ferric() -> bool:
    try:
        import ferric  # noqa: F401
    except ImportError:
        return False
    return True


needs_ferric = pytest.mark.skipif(
    not _have_ferric(),
    reason="the compiled `ferric` extension is not importable; build it with "
    "`cargo build --release -p ferric-python` (see CLAUDE.md)",
)

WATER_XYZ = textwrap.dedent(
    """\
    3
    water
    O  0.000000  0.000000  0.117300
    H  0.000000  0.757200 -0.469200
    H  0.000000 -0.757200 -0.469200
    """
)

# The same asymmetric geometry as `xyz3` in test_pdb_and_xyz_agree_bitwise.
# Every component is nonzero and the three axes differ, so a scale error on any
# ONE axis is detectable -- see that test's docstring for the mutation that
# survived a symmetric fixture.
ASYMMETRIC_PDB = textwrap.dedent(
    """\
    ATOM      1  O   HOH A   1       0.311   0.204   0.117  1.00  0.00           O
    ATOM      2  H1  HOH A   1       1.288   0.961  -0.469  1.00  0.00           H
    ATOM      3  H2  HOH A   1      -0.752  -0.643  -0.288  1.00  0.00           H
    END
    """
)

# Same three atoms, same Angstrom values, as a minimal PDB. Columns follow the
# PDB spec: element symbol right-justified in 77-78.
WATER_PDB = textwrap.dedent(
    """\
    ATOM      1  O   HOH A   1       0.000   0.000   0.117  1.00  0.00           O
    ATOM      2  H1  HOH A   1       0.000   0.757  -0.469  1.00  0.00           H
    ATOM      3  H2  HOH A   1       0.000  -0.757  -0.469  1.00  0.00           H
    END
    """
)


def _write(tmp_path, name: str, text: str):
    p = tmp_path / name
    p.write_text(text)
    return p


# ── the format layer, no compiled extension needed ──


def test_reads_xyz(tmp_path):
    s = read_structure(_write(tmp_path, "w.xyz", WATER_XYZ))
    assert s.symbols == ("O", "H", "H")
    assert s.coords[0] == (0.0, 0.0, 0.1173)
    assert s.charge == 0 and s.multiplicity == 1


def test_xyz_atom_count_mismatch_is_an_error(tmp_path):
    """MUTATION KILLED: trusting the header count without checking the body.

    A truncated XYZ is a common real failure (an interrupted write). Silently
    reading fewer atoms than the header promises would run a calculation on a
    fragment of the molecule.
    """
    bad = "5\nwater\nO 0 0 0\nH 0 0 1\n"
    with pytest.raises(StructureError, match="header says 5 atoms"):
        read_structure(_write(tmp_path, "bad.xyz", bad))


def test_unknown_suffix_names_the_known_ones(tmp_path):
    with pytest.raises(StructureError, match="Known:"):
        read_structure(_write(tmp_path, "thing.zzz", "x"))


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(StructureError, match="no such file"):
        read_structure(tmp_path / "absent.xyz")


def test_fmt_overrides_the_suffix(tmp_path):
    """An XYZ named `.txt` still reads when the format is stated."""
    s = read_structure(_write(tmp_path, "w.txt", WATER_XYZ), fmt="xyz")
    assert s.symbols == ("O", "H", "H")


def test_pdb_without_hydrogens_refuses(tmp_path):
    """MUTATION KILLED: silently accepting a crystallographic PDB.

    Most deposited structures carry no hydrogens. A QM calculation on such a
    structure is meaningless, and inventing them requires pH and residue
    context that this reader does not have -- so it must refuse, not guess.
    """
    pytest.importorskip("gemmi")
    heavy = "\n".join(ln for ln in WATER_PDB.splitlines() if ln.strip().endswith(" O"))
    with pytest.raises(StructureError, match="no hydrogens"):
        read_structure(_write(tmp_path, "dry.pdb", heavy + "\nEND\n"))


def test_the_bitwise_fixture_stays_asymmetric():
    """Guard the guard.

    `test_pdb_and_xyz_agree_bitwise` can only detect a per-axis scale error on
    an axis whose coordinates are nonzero -- and its first fixture was a
    symmetric water with x=0 everywhere, which let a real mutation through.
    This pins the property that made the fix work, so a future "cleanup" to a
    tidy symmetric geometry fails here instead of silently re-blinding that
    test.
    """
    rows = [
        tuple(float(v) for v in line.split()[5:8])
        for line in ASYMMETRIC_PDB.splitlines()
        if line.startswith("ATOM")
    ]
    assert len(rows) == 3
    for axis, name in enumerate("xyz"):
        column = [r[axis] for r in rows]
        assert all(v != 0.0 for v in column), (
            f"the {name} column contains a zero; a scale error on {name} would "
            f"be undetectable, which is exactly the blind spot this fixture "
            f"was rewritten to remove"
        )
    # Distinct axes: a fixture where two axes carry identical values could not
    # distinguish a swap from a correct read.
    assert len({tuple(r[a] for r in rows) for a in range(3)}) == 3, (
        "two axes carry identical columns; an axis swap would be undetectable"
    )


def test_structure_rejects_nonfinite_coordinates():
    with pytest.raises(StructureError, match="non-finite"):
        Structure(("H",), ((0.0, 0.0, math.nan),))


def test_structure_rejects_length_mismatch():
    with pytest.raises(StructureError, match="1 symbols but 2"):
        Structure(("H",), ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)))


def test_multiplicity_zero_is_rejected_and_explains_itself():
    """Multiplicity is 2S+1, so 0 is not a value it can take.

    The message must say so, because passing the UNPAIRED-ELECTRON COUNT is the
    natural mistake (0 unpaired -> someone writes 0, meaning a singlet).
    """
    with pytest.raises(StructureError, match=r"2S\+1"):
        Structure(("H", "H"), ((0.0, 0.0, 0.0), (0.0, 0.0, 0.74)), multiplicity=0)


def test_to_xyz_round_trips_floats_exactly():
    """MUTATION KILLED: formatting coordinates with `%.6f` or similar.

    The intermediate is text, so a lossy float format would silently perturb
    every coordinate. `repr` of a float64 is exact by construction.
    """
    x = 0.1234567890123456789
    s = Structure(("H",), ((x, -x, 1e-17),))
    body = s.to_xyz().splitlines()[2].split()
    assert float(body[1]) == x
    assert float(body[2]) == -x
    assert float(body[3]) == 1e-17


def test_missing_backend_names_the_extra(monkeypatch):
    """A missing optional dep must say what to install, not raise ImportError
    from three frames down inside rdkit."""
    import builtins

    real = builtins.__import__

    def fake(name, *a, **kw):
        if name == "rdkit":
            raise ImportError("no rdkit")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake)
    from tools.structure import _require

    with pytest.raises(MissingBackend, match=r"ferric\[docking\]"):
        _require("rdkit", "SMILES", "docking")


# ── the Rust seam ──


@needs_ferric
def test_xyz_reaches_a_molecule(tmp_path):
    from tools.structure import read

    mol = read(_write(tmp_path, "w.xyz", WATER_XYZ))
    assert mol.natoms() == 3
    assert mol.symbols() == ["O", "H", "H"]
    assert mol.nelec() == 10


@needs_ferric
def test_pdb_and_xyz_agree_bitwise(tmp_path):
    """THE central claim of this module.

    A PDB and an XYZ of the same geometry must produce bit-identical internal
    (Bohr) coordinates, because both are converted to the same Angstrom
    intermediate and handed to the same Rust parser.

    This fails the moment someone adds a second conversion path, applies their
    own Angstrom->Bohr constant, or rounds in `to_xyz`. `!=` on floats is
    deliberate: approximate equality here would pass with exactly the defect
    the test exists to catch.

    The PDB fixture carries 3 decimals (the format's precision), so the XYZ
    fixture is written to match; this compares the PIPELINES, not the formats'
    differing precision.

    EVERY COORDINATE IS NONZERO AND THE THREE AXES ARE MUTUALLY DISTINCT, on
    purpose. The original fixture was a symmetric water with x=0 on all three
    atoms; a mutation multiplying the PDB reader's x by 1.0000000001 SURVIVED
    it, because 0.0 times anything is still 0.0. A geometry with a zero
    component cannot detect a scale error on that component, so this test used
    a molecule that made its own central claim unfalsifiable on one axis in
    three. Do not "tidy" these back into a symmetric geometry.
    """
    pytest.importorskip("gemmi")
    from tools.structure import read

    xyz3 = textwrap.dedent(
        """\
        3
        water, deliberately asymmetric -- see the docstring
        O   0.311   0.204   0.117
        H   1.288   0.961  -0.469
        H  -0.752  -0.643  -0.288
        """
    )
    a = read(_write(tmp_path, "w.xyz", xyz3))
    b = read(_write(tmp_path, "w.pdb", ASYMMETRIC_PDB))
    assert a.symbols() == b.symbols()
    assert a.coords_bohr() == b.coords_bohr(), (
        "PDB and XYZ diverged. Both must funnel through Molecule.from_xyz_string; "
        "a second conversion path or unit constant has been introduced."
    )


@needs_ferric
def test_charge_and_multiplicity_reach_the_molecule(tmp_path):
    """MUTATION KILLED: dropping charge/multiplicity on the floor.

    A wrapper that accepted these and never passed them on would run every job
    as a neutral singlet, converging happily to the wrong answer. Hydroxide:
    10 electrons, charge -1.
    """
    from tools.structure import read

    oh = "2\nhydroxide\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n"
    mol = read(_write(tmp_path, "oh.xyz", oh), charge=-1, multiplicity=1)
    assert mol.nelec() == 10


@needs_ferric
def test_odd_electron_count_at_default_multiplicity_is_rejected(tmp_path):
    """ferric's parity check must be reachable THROUGH this wrapper.

    A methyl radical has 9 electrons and cannot be a singlet. If the wrapper
    ever bypassed `parse_xyz` (e.g. by switching to `from_coordinates`), this
    protection would vanish silently -- so the test pins that the check still
    fires on this path, not merely that it exists in Rust.
    """
    from tools.structure import read

    ch3 = textwrap.dedent(
        """\
        4
        methyl radical
        C  0.000  0.000  0.000
        H  1.079  0.000  0.000
        H -0.539  0.934  0.000
        H -0.539 -0.934  0.000
        """
    )
    with pytest.raises(Exception, match="(?i)multiplicit"):
        read(_write(tmp_path, "ch3.xyz", ch3), charge=0, multiplicity=1)

    # ...and the correct doublet goes through.
    mol = read(_write(tmp_path, "ch3.xyz", ch3), charge=0, multiplicity=2)
    assert mol.nelec() == 9


# --- element inference from PDB/PQR atom names -------------------------------
#
# These exist because the two-letter branch was UNREACHABLE for two months: the
# code built an `Xx`-cased symbol and tested it for membership in an
# all-uppercase frozenset, so nothing ever matched. `CL`/`ZN`/`NA` came back as
# carbon, `Z` and nitrogen. Every test below fails against that version.


def test_two_letter_elements_are_not_truncated_to_their_first_letter():
    """The bug, stated directly. Chloride is not carbon; zinc is not `Z`."""
    from tools.structure import element_from_pdb_atom_name as element

    assert element("CL") == "Cl"
    assert element("ZN") == "Zn"
    assert element("NA") == "Na"
    assert element("BR") == "Br"
    assert element("MG") == "Mg"
    assert element("FE") == "Fe"


def test_the_two_letter_branch_is_reachable_at_all():
    """A guard on the defect CLASS, not one instance.

    The original defect was not a wrong entry in the table -- it was that no
    input could reach the table. Assert every listed symbol round-trips, so a
    future case-handling change that re-breaks the comparison fails here even
    if `CL` happens to keep working.
    """
    from tools.structure import _TWO_LETTER_OK
    from tools.structure import element_from_pdb_atom_name as element

    for upper in sorted(_TWO_LETTER_OK):
        expected = upper[0] + upper[1].lower()
        assert element(upper) == expected, f"{upper} unreachable"
        assert element(upper.lower()) == expected, f"{upper} case-sensitive"


def test_backbone_carbons_stay_carbon():
    """The reason the table is short: PDB names look like element symbols.

    `CA` is an alpha carbon far more often than calcium, `CD` a delta carbon
    rather than cadmium. Getting the ions right must not cost the backbone.
    """
    from tools.structure import element_from_pdb_atom_name as element

    for name in ("CA", "CB", "CG", "CD", "CE", "CZ", "CD1", "CG2"):
        assert element(name) == "C", name
    for name in ("ND1", "NE2", "NZ"):
        assert element(name) == "N", name
    for name in ("OD1", "OG", "OXT"):
        assert element(name) == "O", name
    for name in ("HB2", "1HB", "HD21", "2HG1"):
        assert element(name) == "H", name


def test_an_atom_name_with_no_letters_is_refused_not_guessed():
    from tools.structure import StructureError
    from tools.structure import element_from_pdb_atom_name as element

    for bad in ("", "   ", "123", "4"):
        with pytest.raises(
            StructureError, match="no element letters|carries no element"
        ):
            element(bad)


def test_python_and_rust_element_heuristics_agree():
    """The two implementations must not drift.

    `crates/ferric-cli/src/config.rs::element_from_pqr_name` does the same job
    for the `[qmmm]` TOML path. If they disagree, the SAME PQR gives different
    nuclear charges from the CLI and from Python. This reads the Rust table out
    of the source rather than duplicating it, so adding a symbol on one side
    without the other fails here.
    """
    import re
    from pathlib import Path

    from tools.structure import _TWO_LETTER_OK

    src = Path(__file__).resolve().parents[3] / "crates/ferric-cli/src/config.rs"
    if not src.exists():  # pragma: no cover - source checkout only
        pytest.skip("ferric-cli source not present")
    text = src.read_text()
    if "fn element_from_pqr_name" not in text:
        pytest.skip("Rust element_from_pqr_name not on this branch yet")
    body = text.split("fn element_from_pqr_name", 1)[1]
    table = re.search(r"for two in \[([^\]]*)\]", body)
    assert table, "could not find the Rust two-letter table"
    rust = {s.strip().strip('"') for s in table.group(1).split(",") if s.strip()}
    # Rust excludes CA inside the loop body rather than from the list.
    if 'two != "CA"' in body:
        rust.discard("CA")
    assert "CA" not in rust and "CA" not in _TWO_LETTER_OK, (
        "CA must resolve to carbon on BOTH sides"
    )
    missing_in_rust = _TWO_LETTER_OK - rust
    missing_in_python = rust - _TWO_LETTER_OK
    assert not missing_in_rust, f"Python accepts {missing_in_rust}, Rust does not"
    assert not missing_in_python, f"Rust accepts {missing_in_python}, Python does not"


def test_a_pqr_of_ions_reads_the_right_elements_end_to_end():
    """Not just the helper -- the reader a caller actually uses.

    A zinc metalloenzyme active site is a core QM/MM case; reading the zinc as
    `Z` gives an SCF on a nonexistent element.
    """
    import tempfile
    from pathlib import Path

    from tools.structure import read_structure

    text = (
        "ATOM      1  CL  CL      1      0.000   0.000   0.000 -1.0000 1.7500\n"
        "ATOM      2  ZN  ZN      2      3.000   0.000   0.000  2.0000 1.3900\n"
        "ATOM      3  NA  NA      3      6.000   0.000   0.000  1.0000 1.3700\n"
        "ATOM      4  CA  ALA     4      9.000   0.000   0.000  0.0337 1.9080\n"
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ions.pqr"
        p.write_text(text)
        st = read_structure(p)
    assert st.symbols == ("Cl", "Zn", "Na", "C")


# --- GROMACS .gro ------------------------------------------------------------

GRO_TWO_WATERS = """MD of 2 waters, t= 0.0
    6
    1WATER   OW    1   0.126   1.624   1.679
    1WATER  HW1    2   0.190   1.661   1.747
    1WATER  HW2    3   0.177   1.568   1.613
    2WATER   OW    4   1.275   0.053   0.622
    2WATER  HW1    5   1.337   0.002   0.680
    2WATER  HW2    6   1.326   0.120   0.568
   1.82060   1.82060   1.82060
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_gro_coordinates_are_nanometres_and_get_scaled():
    """The one reader whose file is not already in Angstrom.

    A missing factor of 10 is the single most likely GRO bug and it does not
    look like an error -- it looks like a very small molecule.
    """
    import tempfile
    from pathlib import Path

    from tools.structure import read_structure

    with tempfile.TemporaryDirectory() as d:
        st = read_structure(_write(Path(d), "w.gro", GRO_TWO_WATERS))
    assert st.symbols == ("O", "H", "H", "O", "H", "H")
    # 0.126 nm -> 1.26 A, not 0.126 A.
    assert st.coords[0] == pytest.approx((1.26, 16.24, 16.79))
    # And the geometry has to be a water: O-H near 0.96 A, not 0.096 A.
    import math

    oh = math.dist(st.coords[0], st.coords[1])
    assert 0.9 < oh < 1.1, f"O-H = {oh} A -- wrong length unit?"


def test_gro_agrees_with_openmms_own_reader():
    """Cross-validation against an INDEPENDENT implementation.

    Reading the spec twice is one source, not two. OpenMM's `GromacsGroFile`
    is a separately written parser, so agreement with it distinguishes a
    correct reader from a self-consistent misreading.
    """
    import tempfile
    from pathlib import Path

    openmm_app = pytest.importorskip("openmm.app")
    from openmm.unit import angstrom

    from tools.structure import read_structure

    with tempfile.TemporaryDirectory() as d:
        p = _write(Path(d), "w.gro", GRO_TWO_WATERS)
        mine = read_structure(p).coords
        ref = openmm_app.GromacsGroFile(str(p)).getPositions(asNumpy=True)
    ref = ref.value_in_unit(angstrom)
    assert len(mine) == len(ref)
    for got, want in zip(mine, ref):
        assert got == pytest.approx(tuple(want), abs=1e-9)


def test_gro_fixed_columns_survive_fields_running_together():
    """Why this is not a `line.split()`.

    GRO is `%5d%-5s%5s%5d%8.3f%8.3f%8.3f` and the fields are allowed to touch.
    With a five-character residue name and a five-character atom name there is
    no space between them at all:

        `    1SOLVECLXYZ    1   1.000   2.000   3.000`

    A whitespace split sees `['1SOLVECLXYZ', '1', '1.000', ...]` and takes the
    ATOM INDEX as the atom name, so the element comes out of a bare `1`. The
    columns say `CLXYZ` -> chlorine.

    This test was rewritten once: the first version used a five-digit index
    with a short atom name (`OW112345`), which a split also mangles -- but only
    by appending digits, and the element heuristic stops at the first
    non-letter, so `split()` still produced the right element and the test
    PASSED against a deliberately broken reader. Truncation that does not
    change the answer is not a discriminating case.
    """
    import tempfile
    from pathlib import Path

    from tools.structure import read_structure

    rows = [
        (1, "SOLVE", "CLXYZ", 1, 1.234, 2.345, 3.456),
        (2, "WATER", "OWXYZ", 2, 1.334, 2.345, 3.456),
    ]
    body = "\n".join("%5d%-5s%5s%5d%8.3f%8.3f%8.3f" % r for r in rows)
    text = f"tight columns\n    2\n{body}\n   2.00000   2.00000   2.00000\n"

    # The premise: the residue and atom names really do touch, so a split
    # cannot recover the atom name at all.
    first = body.splitlines()[0]
    assert first.split()[0] == "1SOLVECLXYZ", first
    assert first[10:15] == "CLXYZ", first

    with tempfile.TemporaryDirectory() as d:
        st = read_structure(_write(Path(d), "tight.gro", text))
    assert st.symbols == ("Cl", "O")
    assert st.coords[0] == pytest.approx((12.34, 23.45, 34.56))


def test_gro_with_a_wrong_atom_count_is_refused():
    import tempfile
    from pathlib import Path

    from tools.structure import StructureError, read_structure

    text = GRO_TWO_WATERS.replace("    6", "    9", 1)
    with tempfile.TemporaryDirectory() as d:
        p = _write(Path(d), "bad.gro", text)
        with pytest.raises(StructureError, match="header says 9 atoms"):
            read_structure(p)


def test_gro_with_a_truncated_atom_line_is_refused_not_padded():
    import tempfile
    from pathlib import Path

    from tools.structure import StructureError, read_structure

    text = "t\n    1\n    1WATER   OW    1   0.126\n   1.0 1.0 1.0\n"
    with tempfile.TemporaryDirectory() as d:
        p = _write(Path(d), "short.gro", text)
        with pytest.raises(StructureError, match="need at least 44"):
            read_structure(p)


def test_gro_multi_frame_reads_frame_one_and_says_so():
    """A trajectory holds frames of the same molecule.

    Concatenating them invents atoms; averaging invents a geometry. Read the
    first and record that in `source`, matching `_read_pdb`'s model-1 rule.
    """
    import tempfile
    from pathlib import Path

    from tools.structure import read_structure

    two = GRO_TWO_WATERS + GRO_TWO_WATERS.replace("0.126", "0.226", 1)
    with tempfile.TemporaryDirectory() as d:
        st = read_structure(_write(Path(d), "traj.gro", two))
    assert len(st.symbols) == 6, "second frame must not be concatenated"
    assert st.coords[0][0] == pytest.approx(1.26), "must be frame 1, not frame 2"
    assert "frame 1 of 2" in st.source


def test_gro_is_listed_as_a_supported_suffix():
    from tools.structure import SUPPORTED_SUFFIXES

    assert SUPPORTED_SUFFIXES[".gro"] == "gro"
