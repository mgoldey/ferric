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
