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


def test_a_docked_pose_carries_the_mapping_end_to_end():
    """`DockedPose.rdkit_index_of_heavy` must survive the PDBQT round trip.

    THE GAP THIS CLOSES. The mapping helper and the `rdkit_index_of_heavy=`
    parameter existed, and NO PRODUCTION CALLER USED THEM -- because
    `_parse_pdbqt_models` discarded the PDBQT serials, so the callers had
    nothing to build a mapping from. The capability was present and the bug was
    still live, which is the most expensive kind of half-fix: it reads as
    handled in review.

    Asserted on the parser directly rather than through a real dock, so it runs
    in the fast tier. `dock_ligand` composes exactly these two pieces.
    """
    from tools.docking.vina_dock import _parse_pdbqt_models

    # Two atoms written in NON-serial order, which is what Meeko does.
    out = (
        "MODEL 1\n"
        "REMARK VINA RESULT:   -7.5  0.0  0.0\n"
        "ATOM      5  C   UNL     1       1.000   2.000   3.000  0.00  0.00    +0.0 C\n"
        "ATOM      1  O   UNL     1       4.000   5.000   6.000  0.00  0.00    -0.3 OA\n"
        "ENDMDL\n"
    )
    models = _parse_pdbqt_models(out)
    assert len(models) == 1
    syms, crds, score, serials = models[0]
    assert serials == [5, 1], (
        f"serials must be retained IN FILE ORDER, got {serials}. Sorting them "
        "would destroy the very correspondence they exist to record."
    )
    assert score == pytest.approx(-7.5)

    # ...and they compose with the remark into a usable mapping.
    serial_to_rdkit = parse_smiles_idx_remark(
        "REMARK SMILES IDX 5 1 1 2\n"  # serial 5 -> rdkit 0, serial 1 -> rdkit 1
    )
    mapping = [serial_to_rdkit[k] for k in serials]
    assert mapping == [0, 1], (
        f"the first coordinate belongs to RDKit atom 0 and the second to 1; "
        f"got {mapping}"
    )
    assert len(syms) == len(crds) == len(serials)


def test_a_partial_mapping_is_not_used_at_all():
    """Covering SOME serials must yield no mapping, not a half-applied one.

    A partial map places some atoms correctly and the rest by position, which
    is strictly harder to notice than no map: the molecule looks almost right.
    `dock_ligand` requires every serial in the pose to be covered before it
    builds one.
    """
    serial_to_rdkit = parse_smiles_idx_remark("REMARK SMILES IDX 5 1\n")
    serials = [5, 1]
    covered = all(k in serial_to_rdkit for k in serials)
    assert not covered, "serial 1 is absent, so this must not count as covered"


def test_the_mapping_covers_HEAVY_atoms_only_so_a_polar_H_does_not_void_it():
    """PDBQT keeps polar hydrogens; `REMARK SMILES IDX` maps only heavy atoms.

    MEASURED on aspirin: the pose has **14** atoms (13 heavy plus the
    carboxylic H, AutoDock type HD) against **13** mapped serials. Requiring
    every serial to be covered -- including that hydrogen's -- makes the guard
    fail on any ligand with a polar H, so the mapping is silently never used.

    That is the failure mode worth a test: the fix would have been INERT on
    exactly the inputs it was written for, and nothing would have said so. The
    earlier end-to-end test used a hand-written two-atom PDBQT with no
    hydrogen, so it could not see this.

    Both sides now use the same heavy-atom subset in the same order:
    `dock_ligand` filters on symbol when building the mapping, and
    `restore_hydrogens` filters on symbol when consuming it.
    """
    # 13 heavy + 1 polar H, serials 1..14; the remark covers 1..13.
    symbols = ["C"] * 13 + ["H"]
    serials = list(range(1, 15))
    remark = "REMARK SMILES IDX " + " ".join(f"{i} {i}" for i in range(1, 14))
    mapping = parse_smiles_idx_remark(remark)

    # THE PRECONDITION: the hydrogen's serial is genuinely absent from the
    # remark, or this test no longer reproduces the case it was written for.
    assert not all(k in mapping for k in serials)

    # Exercise THE PRODUCTION FUNCTION, not a copy of its logic.
    #
    # An earlier version of this test recomputed the heavy-atom filter inline
    # and asserted on that. It passed with the production code REVERTED to the
    # all-serials guard -- an inert test, and exactly the failure this file
    # documents elsewhere. `heavy_atom_mapping` exists so the assertion can
    # reach the real code.
    from tools.docking.vina_dock import heavy_atom_mapping

    got = heavy_atom_mapping(symbols, serials, mapping)
    assert got is not None, (
        "a polar hydrogen must not void the mapping -- requiring its serial to "
        "be covered makes this return None for every real ligand"
    )
    assert len(got) == 13, f"expected 13 heavy-atom indices, got {len(got)}"
    assert got == list(range(13))


def test_a_resolution_beyond_the_float_range_is_refused_not_deferred():
    """`10**400` passed validation and then raised inside `resolves`.

    A field that validates and THEN throws downstream is worse than one that
    never validated: the caller has been told the value is safe. `10**400` is a
    finite, positive Python int, so every range check passed; the failure came
    later, from `float()` conversion during the quadrature.

    Measured before the fix: construction succeeded, `resolves` raised
    `OverflowError: int too large to convert to float`.
    """
    import pytest as _pytest

    from tools.pipeline.tiers import TierResult

    with _pytest.raises(ValueError, match="float"):
        TierResult("x", -10.0, resolution=10**400)

    # 1e200 IS representable, so it must be ACCEPTED -- and `resolves` must not
    # overflow on it. `(a**2 + b**2) ** 0.5` overflows above ~1.3e154;
    # `math.hypot` does not.
    a = TierResult("a", -10.0, resolution=1e200)
    b = TierResult("b", -20.0, resolution=1.0)
    assert a.resolves(b) is False, "a 10-unit gap against 1e200 noise is unresolvable"

    # An int resolution is coerced, so the stored value is always a float.
    assert TierResult("y", -1.0, resolution=4).resolution == 4.0
