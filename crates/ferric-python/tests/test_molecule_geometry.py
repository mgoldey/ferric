"""Anchor tests for the `Molecule` geometry accessors.

WHY THIS EXISTS. Until these getters landed, `Molecule` exposed only
`natoms`/`nelec`/`nuclear_repulsion` — you could hand ferric a geometry but
never get one back. That made any workflow with a geometry *in the loop*
impossible from Python: `tools/active_site/pose_relaxation.py` had to hardcode
`RelaxedPose.coords_angstrom = None` because `run_optimize(...).mol()` was an
opaque handle. These accessors close that gap, so the failure modes worth
pinning are the ones that would silently corrupt such a loop.

Three silent failure classes, all of which produce numbers that still run:

1.  **A unit error.** Coordinates are stored in Bohr; XYZ input and `coords()`
    are Ångström. A dropped conversion is a geometry wrong by 1.889726x that
    still converges to a plausible energy. Pinned against an independently
    computed nuclear repulsion, not against ferric's own arithmetic.

2.  **Atom-order scrambling.** `symbols()`, `coords()`, `atomic_numbers()` and
    `is_ghost()` are four separate loops over the same `Vec<Atom>`. If any one
    of them ever reorders, a downstream consumer silently pairs element X with
    element Y's position. Pinned on an asymmetric molecule where every atom is
    distinguishable.

3.  **A lossy geometry round-trip.** The whole point of these getters is
    feeding a geometry back in. `to_xyz_string` -> `from_xyz_string` is the
    round-trip that a pose-relaxation loop actually performs, and it must not
    drift. Note the binding's own doc: Å round-trips are ~1 ulp accurate, NOT
    bit-exact (float multiplication is not exactly invertible), so this asserts
    a tight tolerance rather than equality.
"""

import math

import pytest
from conftest import (
    BOHR_PER_ANGSTROM,
    WATER_ANGSTROM,
    WATER_SYMBOLS,
    nuclear_repulsion_from_bohr,
    water_xyz_string,
)

import ferric

# An asymmetric, all-distinguishable molecule: every atom has a different
# element AND a different coordinate, so any transposition between the four
# per-atom accessors is detectable. Symmetric water would hide a swap of its
# two hydrogens.
HCNO_SYMBOLS = ["H", "C", "N", "O"]
HCNO_ANGSTROM = [
    [0.100, 0.200, 0.300],
    [1.400, 0.500, -0.700],
    [2.600, -0.900, 1.100],
    [-1.300, 2.100, -2.500],
]
HCNO_Z = [1, 6, 7, 8]


def hcno_xyz_string():
    rows = "\n".join(
        f"{s} {r[0]!r} {r[1]!r} {r[2]!r}" for s, r in zip(HCNO_SYMBOLS, HCNO_ANGSTROM)
    )
    return f"{len(HCNO_SYMBOLS)}\n\n{rows}\n"


# ── 1. units ──

def test_coords_are_angstrom_matching_the_input():
    """`coords()` returns the Ångström numbers that went in."""
    mol = ferric.Molecule.from_xyz_string(water_xyz_string())
    got = mol.coords()
    assert len(got) == 3 and all(len(r) == 3 for r in got)
    for i in range(3):
        for j in range(3):
            assert got[i][j] == pytest.approx(WATER_ANGSTROM[i][j], abs=1e-12)


def test_coords_bohr_is_the_angstrom_values_scaled():
    """`coords_bohr()` is `coords()` times the Å->Bohr factor.

    The factor is the literal from conftest, not one read back out of ferric,
    so a change to ferric's constant fails here.
    """
    mol = ferric.Molecule.from_xyz_string(water_xyz_string())
    ang, bohr = mol.coords(), mol.coords_bohr()
    for i in range(3):
        for j in range(3):
            assert bohr[i][j] == pytest.approx(
                ang[i][j] * BOHR_PER_ANGSTROM, rel=1e-13, abs=1e-13
            )


def test_coords_bohr_reproduces_nuclear_repulsion_independently():
    """The strongest unit check available: an outside NRE from the Bohr
    coordinates must match ferric's own `nuclear_repulsion()`.

    This is the assertion that has teeth. `nuclear_repulsion_from_bohr` is
    reimplemented in Python (conftest), so agreement proves the returned array
    really is in Bohr — a factor-1.889726 unit slip would show up as a ~47%
    energy discrepancy, not a rounding difference.
    """
    mol = ferric.Molecule.from_xyz_string(hcno_xyz_string())
    expected = nuclear_repulsion_from_bohr(
        [list(row) for row in mol.coords_bohr()], HCNO_Z
    )
    assert mol.nuclear_repulsion() == pytest.approx(expected, rel=1e-12)


def test_nuclear_repulsion_would_fail_on_angstrom_coordinates():
    """Reachability check for the test above: confirm that feeding the SAME
    formula the Ångström array gives a decisively DIFFERENT answer.

    Without this, `test_coords_bohr_reproduces_nuclear_repulsion_independently`
    could be passing for a trivial reason (e.g. both sides accidentally using
    the same wrong units). Per CLAUDE.md's protocol: a test never seen to fail
    is an assumption.
    """
    mol = ferric.Molecule.from_xyz_string(hcno_xyz_string())
    wrong = nuclear_repulsion_from_bohr([list(r) for r in mol.coords()], HCNO_Z)
    # Å coordinates are numerically SMALLER, so 1/r is LARGER by exactly the
    # conversion factor. Assert the discrepancy is the expected large one.
    assert wrong == pytest.approx(mol.nuclear_repulsion() * BOHR_PER_ANGSTROM, rel=1e-12)
    assert not math.isclose(wrong, mol.nuclear_repulsion(), rel_tol=0.4)


# ── 2. atom order ──

def test_all_four_accessors_agree_on_atom_order():
    """`symbols`/`coords`/`atomic_numbers`/`is_ghost` are four independent
    loops over the same atom vector; they must stay index-aligned."""
    mol = ferric.Molecule.from_xyz_string(hcno_xyz_string())
    assert mol.natoms() == 4
    assert mol.symbols() == HCNO_SYMBOLS
    assert mol.atomic_numbers() == HCNO_Z
    assert mol.is_ghost() == [False, False, False, False]
    coords = mol.coords()
    for i, expected_row in enumerate(HCNO_ANGSTROM):
        for j in range(3):
            assert coords[i][j] == pytest.approx(expected_row[j], abs=1e-12)


def test_symbols_and_atomic_numbers_are_consistent():
    """Cross-check the two element accessors against each other via a table
    written here rather than sourced from ferric."""
    mol = ferric.Molecule.from_xyz_string(hcno_xyz_string())
    z_of = {"H": 1, "C": 6, "N": 7, "O": 8}
    assert [z_of[s] for s in mol.symbols()] == mol.atomic_numbers()


# ── 3. round-trip ──

def test_to_xyz_string_round_trips_the_geometry():
    """`to_xyz_string` -> `from_xyz_string` is the loop a pose-relaxation
    driver runs. Tolerance is 1e-11 Å, not equality: the binding's own doc
    records that Å round-trips are ~1 ulp accurate because float
    multiplication is not exactly invertible.
    """
    mol = ferric.Molecule.from_xyz_string(hcno_xyz_string())
    again = ferric.Molecule.from_xyz_string(mol.to_xyz_string())

    assert again.symbols() == mol.symbols()
    assert again.atomic_numbers() == mol.atomic_numbers()
    assert again.natoms() == mol.natoms()

    a, b = mol.coords(), again.coords()
    for i in range(mol.natoms()):
        for j in range(3):
            assert b[i][j] == pytest.approx(a[i][j], abs=1e-11)

    # NRE is the scalar summary of the whole geometry: if any coordinate had
    # drifted meaningfully this catches it in one number.
    assert again.nuclear_repulsion() == pytest.approx(
        mol.nuclear_repulsion(), rel=1e-12
    )


def test_round_trip_is_idempotent():
    """A second round-trip must not drift further — otherwise an iterative
    optimization loop that re-serializes each step would accumulate error."""
    mol = ferric.Molecule.from_xyz_string(hcno_xyz_string())
    once = ferric.Molecule.from_xyz_string(mol.to_xyz_string())
    twice = ferric.Molecule.from_xyz_string(once.to_xyz_string())
    a, b = once.coords(), twice.coords()
    for i in range(mol.natoms()):
        for j in range(3):
            assert b[i][j] == pytest.approx(a[i][j], abs=0.0, rel=0.0) or b[i][j] == a[i][j]


def test_to_xyz_string_records_charge_and_multiplicity():
    """XYZ cannot carry charge/multiplicity, so `to_xyz_string` puts them in
    the comment line. `from_xyz_string` does NOT parse them back (documented),
    so this pins the human-readable record only — and pins that the default
    round-trip really does come back neutral singlet, which is the trap.
    """
    mol = ferric.Molecule.from_xyz_string(hcno_xyz_string(), -1, 2)
    text = mol.to_xyz_string()
    assert "charge=-1" in text
    assert "multiplicity=2" in text
    # The documented caveat, asserted so nobody "fixes" it silently: the
    # round-trip drops them unless passed explicitly. An anion carries one
    # EXTRA electron, so re-reading it as neutral LOSES one.
    assert ferric.Molecule.from_xyz_string(text).nelec() == mol.nelec() - 1
    assert ferric.Molecule.from_xyz_string(text, -1, 2).nelec() == mol.nelec()


def test_coords_matches_conformer_ensemble_on_the_same_geometry():
    """`Molecule.coords()` and `ConformerEnsemble.coordinates()` are separate
    implementations of the same conversion. Agreement between two independent
    constructions is the check that actually distinguishes a systematic unit
    error (per CLAUDE.md: consistency across systems is not corroboration; an
    independent construction is).
    """
    mol = ferric.Molecule.from_xyz_string(water_xyz_string())
    ens = ferric.ConformerEnsemble.from_coordinates([WATER_ANGSTROM], WATER_SYMBOLS)
    ens_coords = ens.coordinates()[0]
    mol_coords = mol.coords()
    for i in range(3):
        for j in range(3):
            assert mol_coords[i][j] == pytest.approx(ens_coords[i][j], abs=1e-12)
    assert mol.symbols() == list(ens.elements())
