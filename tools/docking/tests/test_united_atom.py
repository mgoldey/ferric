"""A docked pose is not the molecule that was docked until its H's are back.

MEASURED 2026-09-21, danuglipron into 7LCJ: 71 atoms in, **42 out**. The
missing 29 are nonpolar hydrogens Vina merges into their carbons. Feeding that
42-atom object to a scorer returns a plausible number for a different molecule.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rdkit", reason="re-hydrogenation is RDKit-based")

from tools.docking.united_atom import restore_hydrogens  # noqa: E402

# Small, unambiguous, and has nonpolar hydrogens to lose: toluene.
TOLUENE = "Cc1ccccc1"
# Heavy atoms only, in SMILES heavy-atom order (a united-atom pose's shape).
TOLUENE_HEAVY = ["C", "C", "C", "C", "C", "C", "C"]
TOLUENE_COORDS = [
    (0.000, 0.000, 0.000),
    (1.500, 0.000, 0.000),
    (2.200, 1.212, 0.000),
    (3.600, 1.212, 0.000),
    (4.300, 0.000, 0.000),
    (3.600, -1.212, 0.000),
    (2.200, -1.212, 0.000),
]


def test_hydrogens_come_back_and_the_formula_is_the_molecule():
    syms, coords = restore_hydrogens(TOLUENE, TOLUENE_HEAVY, TOLUENE_COORDS)
    assert len(syms) == len(coords)
    # C7H8 -- the united-atom input had only the 7 carbons.
    assert syms.count("C") == 7
    assert syms.count("H") == 8
    assert len(syms) == 15 > len(TOLUENE_HEAVY)


def test_the_docked_heavy_atoms_DO_NOT_MOVE():
    """The heavy atoms are the docking result.

    If relaxation moved them, the scored pose would not be the pose that was
    docked -- a silent substitution of the answer.
    """
    syms, coords = restore_hydrogens(TOLUENE, TOLUENE_HEAVY, TOLUENE_COORDS)
    heavy_out = [c for s, c in zip(syms, coords) if s != "H"]
    assert len(heavy_out) == len(TOLUENE_COORDS)
    worst = max(
        max(abs(a - b) for a, b in zip(p, q)) for p, q in zip(TOLUENE_COORDS, heavy_out)
    )
    assert worst == pytest.approx(0.0, abs=1e-9), (
        f"heavy atoms moved by {worst} A -- they are FIXED during H placement"
    )


def test_a_heavy_atom_COUNT_MISMATCH_is_refused_not_truncated():
    """The failure that motivated this module.

    Zipping to the shorter list would score a fragment of the molecule and
    return a number rather than an error.
    """
    with pytest.raises(ValueError, match="different molecules"):
        restore_hydrogens(TOLUENE, TOLUENE_HEAVY[:-1], TOLUENE_COORDS[:-1])


def test_symbols_and_coords_must_be_per_atom():
    with pytest.raises(ValueError, match="per-atom"):
        restore_hydrogens(TOLUENE, TOLUENE_HEAVY, TOLUENE_COORDS[:-1])


def test_an_unparseable_smiles_is_refused():
    with pytest.raises(ValueError, match="could not parse"):
        restore_hydrogens("not-a-smiles((", TOLUENE_HEAVY, TOLUENE_COORDS)


def test_hydrogens_are_placed_at_chemically_sane_distances():
    """A vacuity guard: the H's must be BONDED, not dumped at the origin.

    Without this, a function that appended 8 hydrogens at (0,0,0) would pass
    every count-based assertion above.
    """
    import math

    syms, coords = restore_hydrogens(TOLUENE, TOLUENE_HEAVY, TOLUENE_COORDS)
    heavy = [c for s, c in zip(syms, coords) if s != "H"]
    for s, c in zip(syms, coords):
        if s != "H":
            continue
        nearest = min(math.dist(c, h) for h in heavy)
        assert 0.8 < nearest < 1.3, (
            f"hydrogen {nearest:.2f} A from its nearest heavy atom -- a C-H "
            "bond is ~1.09 A, so this H was not actually placed on the molecule"
        )
