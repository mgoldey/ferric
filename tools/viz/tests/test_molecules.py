"""Tests for `tools.viz.molecules`.

Asserts behaviour, not pixels: that a PNG comes back, that the highlight is
computed from the structures rather than asserted by the caller, and that an
unevaluated score is captioned differently from a bad one.

Every test skips cleanly without RDKit, which lives in the `docking` extra.
"""

from __future__ import annotations

import pytest

rdkit = pytest.importorskip("rdkit", reason="RDKit is in the 'docking' extra")

from tools.viz.molecules import (  # noqa: E402
    contact_map,
    contacting_atom_indices,
    depict,
    grid_with_scores,
    highlight_difference,
)  # noqa: E402

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

BENZENE = "c1ccccc1"
TOLUENE = "Cc1ccccc1"
PHENOL = "Oc1ccccc1"


def test_depict_returns_a_png():
    out = depict(BENZENE)
    assert out.startswith(PNG_MAGIC)
    assert len(out) > 200, "a real depiction is not a near-empty file"


def test_an_unparseable_smiles_raises_rather_than_drawing_a_blank():
    """RDKit returns None for bad SMILES, which becomes a blank image later."""
    with pytest.raises(ValueError, match="could not parse SMILES"):
        depict("this-is-not-smiles((")


def test_highlight_marks_the_substituent_and_not_the_shared_ring():
    """The highlight must be DERIVED from the two structures.

    Toluene vs benzene differ by exactly one heavy atom (the methyl carbon), so
    a correct MCS leaves exactly one highlighted atom. Asserting the count
    rather than the image is what makes this robust to RDKit's renderer.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    parent, analogue = Chem.MolFromSmiles(BENZENE), Chem.MolFromSmiles(TOLUENE)
    res = rdFMCS.FindMCS(
        [parent, analogue], ringMatchesRingOnly=True, completeRingsOnly=True, timeout=10
    )
    patt = Chem.MolFromSmarts(res.smartsString)
    shared = set(analogue.GetSubstructMatch(patt))
    changed = [a.GetIdx() for a in analogue.GetAtoms() if a.GetIdx() not in shared]
    assert len(changed) == 1, f"expected the methyl carbon alone, got {changed}"

    out = highlight_difference(BENZENE, TOLUENE)
    assert out.startswith(PNG_MAGIC)


def test_identical_molecules_highlight_nothing():
    """No difference to point at is the honest output, not an error."""
    same = highlight_difference(BENZENE, BENZENE)
    assert same.startswith(PNG_MAGIC)


def test_highlighting_changes_the_image():
    """A vacuity guard.

    Without this, every assertion above would also pass for an implementation
    that ignored the highlight entirely and always drew a plain depiction.
    """
    plain = highlight_difference(TOLUENE, TOLUENE)  # MCS covers all: no highlight
    marked = highlight_difference(BENZENE, TOLUENE)  # one atom highlighted
    assert plain != marked, (
        "highlighted and unhighlighted depictions of the same molecule are "
        "byte-identical, so the highlight is not being drawn"
    )


def test_grid_captions_a_missing_score_as_na_not_as_a_number():
    """An unevaluated candidate and a badly scoring one must not look alike.

    Asserted by RENDERING BOTH and comparing bytes. Checking only that a PNG
    came back does not test this at all: a first version of this test did
    exactly that, and a mutation replacing the "n/a" caption with `0.00`
    SURVIVED it. The caption is the whole point of the function, so the test
    has to be able to see it.
    """
    missing = grid_with_scores(
        [BENZENE, TOLUENE, PHENOL], [-5.2, None, -7.1], labels=["a", "b", "c"]
    )
    as_zero = grid_with_scores(
        [BENZENE, TOLUENE, PHENOL], [-5.2, 0.0, -7.1], labels=["a", "b", "c"]
    )
    assert missing.startswith(PNG_MAGIC)
    assert missing != as_zero, (
        "a None score renders identically to a 0.00 score, so an unevaluated "
        "candidate is indistinguishable from one that scored zero"
    )

    # And it must differ from a genuinely bad score too, not just from zero.
    as_bad = grid_with_scores(
        [BENZENE, TOLUENE, PHENOL], [-5.2, -99.0, -7.1], labels=["a", "b", "c"]
    )
    assert missing != as_bad


def test_grid_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="one-to-one"):
        grid_with_scores([BENZENE, TOLUENE], [-1.0])
    with pytest.raises(ValueError, match="labels for"):
        grid_with_scores([BENZENE], [-1.0], labels=["a", "b"])
    with pytest.raises(ValueError, match="nothing to draw"):
        grid_with_scores([], [])


# --- contact map -------------------------------------------------------------


def _ethanol_coords():
    """Ethanol with explicit H, in RDKit's atom order."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(m, randomSeed=0xF00D)
    return m, [tuple(float(v) for v in r) for r in m.GetConformer().GetPositions()]


def test_contact_map_highlights_only_atoms_within_the_cutoff():
    """Proximity decides the highlight, and the cutoff is the knob."""
    mol, coords = _ethanol_coords()
    # One pocket atom, parked right on the first ligand atom.
    pocket = [coords[0]]

    tight = contact_map("CCO", coords, pocket, cutoff_angstrom=0.5)
    loose = contact_map("CCO", coords, pocket, cutoff_angstrom=99.0)
    assert tight.startswith(PNG_MAGIC) and loose.startswith(PNG_MAGIC)
    assert tight != loose, (
        "a 0.5 A cutoff and a 99 A cutoff must not produce the same picture; "
        "if they do, the cutoff is not being applied"
    )

    # Comparing two RENDERS is not enough, and this is the lesson: an
    # implementation highlighting EVERY atom regardless of distance renders any
    # two cutoffs identically to each other, so a byte comparison cannot see
    # it. That mutation SURVIVED until the contact logic was split out and
    # asserted as a LIST OF INDICES.
    assert contacting_atom_indices(coords, pocket, 0.5) == [0], (
        "one pocket atom on ligand atom 0, cutoff 0.5 A -> exactly atom 0"
    )
    assert contacting_atom_indices(coords, pocket, 99.0) == list(range(len(coords))), (
        "a 99 A cutoff must reach every atom"
    )
    assert contacting_atom_indices(coords, [(500.0, 500.0, 500.0)], 4.0) == [], (
        "a pocket 500 A away must contact NOTHING -- an always-highlight "
        "implementation cannot produce an empty list"
    )


def test_an_atom_count_mismatch_raises_rather_than_marking_wrong_atoms():
    """The united-atom trap, caught rather than drawn.

    MEASURED (M14): a PDBQT-docked danuglipron has 42 atoms where the real
    molecule has 71, because nonpolar hydrogens are merged. Highlighting under
    that mismatch would mark the WRONG atoms and look entirely plausible.
    """
    _, coords = _ethanol_coords()
    with pytest.raises(ValueError, match="united-atom|coordinates were given"):
        contact_map("CCO", coords[:3], [(0.0, 0.0, 0.0)])


def test_an_empty_pocket_is_an_error_not_an_empty_highlight():
    """Zero contacts from zero pocket atoms is a claim, not an absence."""
    _, coords = _ethanol_coords()
    with pytest.raises(ValueError, match="no pocket coordinates"):
        contact_map("CCO", coords, [])


def test_the_legend_states_how_many_heavy_atoms_contact():
    """A picture of highlights without a count is hard to compare across poses."""
    mol, coords = _ethanol_coords()
    far = [(500.0, 500.0, 500.0)]
    none_touching = contact_map("CCO", coords, far)
    all_touching = contact_map("CCO", coords, [coords[0]], cutoff_angstrom=99.0)
    assert none_touching != all_touching, (
        "nothing-in-contact and everything-in-contact must render differently"
    )


def test_a_nonpositive_cutoff_is_rejected():
    _, coords = _ethanol_coords()
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="cutoff_angstrom"):
            contact_map("CCO", coords, [(0.0, 0.0, 0.0)], cutoff_angstrom=bad)


def test_contacting_atom_indices_refuses_a_nonpositive_cutoff():
    """A negative cutoff must ERROR, not silently act as its absolute value.

    The comparison is against `cutoff**2`, so `-4.0` squares to the same 16.0
    as `+4.0` and returns identical contacts. MEASURED before the fix:
    `contacting_atom_indices(coords, pocket, -4.0)` returned `[0]`, exactly
    what `+4.0` returns.

    `contact_map` validated its own cutoff, but this helper is PUBLIC -- it was
    made public precisely so the contact logic could be asserted without going
    through a rendered image -- so it has to validate its own input rather than
    assume a particular caller.
    """
    coords = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    pocket = [(0.5, 0.0, 0.0)]

    # The anchor: a positive cutoff still works, so the test is about the SIGN.
    assert contacting_atom_indices(coords, pocket, 4.0) == [0]

    for bad in (-4.0, 0.0, -1e-9):
        with pytest.raises(ValueError, match="positive"):
            contacting_atom_indices(coords, pocket, bad)


def test_contacting_atom_indices_refuses_an_empty_pocket():
    """Zero contacts against zero pocket atoms is a claim, not an absence."""
    with pytest.raises(ValueError, match="empty"):
        contacting_atom_indices([(0.0, 0.0, 0.0)], [], 4.0)
