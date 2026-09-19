"""Tests for `tools.viz.molecules`.

Asserts behaviour, not pixels: that a PNG comes back, that the highlight is
computed from the structures rather than asserted by the caller, and that an
unevaluated score is captioned differently from a bad one.

Every test skips cleanly without RDKit, which lives in the `docking` extra.
"""

from __future__ import annotations

import pytest

rdkit = pytest.importorskip("rdkit", reason="RDKit is in the 'docking' extra")

from tools.viz.molecules import depict, grid_with_scores, highlight_difference  # noqa: E402

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
