"""`restore_hydrogens` must rebuild the real species without moving the pose.

The bug it exists to prevent is silent. MEASURED (RESULTS.md M14): a docked
danuglipron pose read from PDBQT has 42 atoms and 263 electrons where the real
molecule has 71 and 292, because PDBQT merges nonpolar hydrogens into their
carbons. `embed_ligand_from_coords` refuses that input; `pose_fit` scores it
without complaint and returns a normal-looking number for an incomplete
molecule.

So the assertions here are in two groups, and BOTH are load-bearing:

1. the hydrogens come back (the species is right), and
2. the heavy atoms DO NOT MOVE (the pose is still the one that was docked).

(2) is the one an implementation is likely to break: it is tempting to relax
the whole structure, which gives a clean-looking geometry that is no longer the
docking result.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rdkit", reason="needs the 'docking' extra")

from tools.docking.united_atom import restore_hydrogens  # noqa: E402

# Ethanol: 3 heavy atoms (C, C, O), 6 hydrogens. Small enough to assert on by
# hand, and it has a hydroxyl so polar and nonpolar hydrogens both appear.
ETHANOL = "CCO"
HEAVY_SYMS = ["C", "C", "O"]
HEAVY_COORDS = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (2.1, 1.2, 0.0)]


def test_hydrogens_are_restored():
    syms, coords = restore_hydrogens(ETHANOL, HEAVY_SYMS, HEAVY_COORDS)
    assert len(syms) == len(coords)
    assert syms.count("H") == 6, (
        f"ethanol has 6 hydrogens, got {syms.count('H')} -- a united-atom "
        "structure would come back with fewer and score as a different species"
    )
    assert len(syms) == 9


def test_the_docked_heavy_atoms_do_not_move():
    """THE assertion that matters: the pose must survive the fix.

    Relaxing the whole molecule would produce a tidier geometry that is no
    longer the docking result, and nothing downstream would notice.
    """
    syms, coords = restore_hydrogens(ETHANOL, HEAVY_SYMS, HEAVY_COORDS)
    heavy_out = [c for s, c in zip(syms, coords) if s != "H"]
    assert len(heavy_out) == 3
    for got, want in zip(heavy_out, HEAVY_COORDS):
        for g, w in zip(got, want):
            assert g == pytest.approx(w, abs=1e-6), (
                f"a docked heavy atom moved from {want} to {got}; the pose "
                "being scored is no longer the pose that was docked"
            )


def test_a_heavy_atom_count_mismatch_is_refused():
    """Different molecules must ERROR, not be silently truncated.

    Truncating would score whichever atoms happened to line up -- the exact
    class of silent wrongness this module was written to stop.
    """
    with pytest.raises(ValueError, match="different molecules"):
        restore_hydrogens(ETHANOL, ["C", "C"], HEAVY_COORDS[:2])
    with pytest.raises(ValueError, match="different molecules"):
        restore_hydrogens(ETHANOL, HEAVY_SYMS + ["C"], HEAVY_COORDS + [(3.0, 0.0, 0.0)])


def test_hydrogens_already_present_are_ignored_not_doubled():
    """A full-hydrogen input must not come back with 12 hydrogens.

    The function filters on symbol, so passing an already-complete structure is
    idempotent in the heavy-atom frame. Without the filter the H count doubles
    and the molecule is nonsense -- and it would still LOOK like a molecule.
    """
    syms, coords = restore_hydrogens(ETHANOL, HEAVY_SYMS, HEAVY_COORDS)
    syms2, _ = restore_hydrogens(ETHANOL, syms, coords)
    assert syms2.count("H") == 6, (
        f"re-running on a full-hydrogen structure gave {syms2.count('H')} "
        "hydrogens; the heavy-atom filter is not working"
    )


def test_an_unparseable_smiles_is_refused():
    with pytest.raises(ValueError, match="unparseable"):
        restore_hydrogens("not-a-smiles((", HEAVY_SYMS, HEAVY_COORDS)
