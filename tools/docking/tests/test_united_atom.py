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

from tools.docking.united_atom import (  # noqa: E402
    parse_smiles_idx_remark,
    restore_hydrogens,
)

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


# Real Meeko output for aspirin, captured 2026-09-19. Kept verbatim rather than
# regenerated at test time so the parser is pinned against the FORMAT and not
# against whatever the installed Meeko happens to emit today.
ASPIRIN_REMARK = (
    "REMARK SMILES CC(=O)Oc1ccccc1C(=O)O\n"
    "REMARK SMILES IDX 5 1 6 2 7 3 8 4 9 5 10 6 4 7 2 8 3 9 1 10 11 11 12 12 13 13\n"
    "REMARK H PARENT 13 14\n"
)


def test_the_meeko_mapping_is_parsed_and_is_NOT_the_identity():
    """Meeko reorders atoms, and the parser must see it.

    MEASURED on aspirin: **10 of 13** heavy atoms come back at a different
    position than RDKit assigned. A positional assignment misplaces an atom by
    up to **4.9 A** -- a scrambled molecule with the right atom count, the right
    elements, and no error raised anywhere.

    The `is not the identity` assertion is the load-bearing one. A parser that
    returned `{i: i}` would satisfy every downstream use and silently restore
    the bug.
    """
    m = parse_smiles_idx_remark(ASPIRIN_REMARK)
    assert len(m) == 13, f"expected 13 heavy atoms, parsed {len(m)}"
    # serial 5 -> rdkit index 0 (the remark is 1-based, the map is 0-based)
    assert m[5] == 0
    assert m[7] == 2
    assert m[1] == 9, "pdbqt serial 1 is RDKit atom 10 -- the reordering"
    reordered = sum(1 for ser, idx in m.items() if ser - 1 != idx)
    assert reordered == 10, (
        f"{reordered} of 13 atoms reordered; the captured remark says 10. If "
        "this changed, the fixture was regenerated with a different Meeko and "
        "the measured 4.9 A figure needs re-checking."
    )


def test_an_absent_remark_returns_empty_not_identity():
    """No remark means UNKNOWN mapping, which the caller must handle.

    Returning an identity map would be the most dangerous possible default: it
    is exactly the wrong assumption, and it looks like a successful parse.
    """
    assert parse_smiles_idx_remark("ATOM      1  C   UNL     1  0.0 0.0 0.0\n") == {}


def test_a_permutation_puts_coordinates_on_the_RIGHT_atoms():
    """With a mapping, the k-th docked coordinate lands on its RDKit atom.

    Built as a deliberate REVERSAL so a positional implementation cannot pass:
    without the mapping every coordinate goes to the wrong atom, and with it
    every one is exact.
    """
    coords = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (2.1, 1.2, 0.0)]
    # Docked order is the REVERSE of RDKit's heavy-atom order.
    reversed_coords = list(reversed(coords))
    mapping = [2, 1, 0]

    syms, out = restore_hydrogens(
        ETHANOL, HEAVY_SYMS, reversed_coords, rdkit_index_of_heavy=mapping
    )
    heavy_out = [c for s, c in zip(syms, out) if s != "H"]
    for got, want in zip(heavy_out, coords):
        for g, w in zip(got, want):
            assert g == pytest.approx(w, abs=1e-6), (
                f"with an explicit mapping the coordinates must land on the "
                f"named atoms; got {heavy_out} for {coords}"
            )


def test_a_mapping_that_is_not_a_permutation_is_refused():
    """A mapping and a SMILES that disagree are different molecules."""
    with pytest.raises(ValueError, match="permutation"):
        restore_hydrogens(
            ETHANOL, HEAVY_SYMS, HEAVY_COORDS, rdkit_index_of_heavy=[0, 0, 1]
        )
    with pytest.raises(ValueError, match="entries"):
        restore_hydrogens(
            ETHANOL, HEAVY_SYMS, HEAVY_COORDS, rdkit_index_of_heavy=[0, 1]
        )
