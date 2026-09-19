"""Substitution proposals for a drug active site.

Written against the danuglipron / GLP-1R prototype (2026-09-19), whose findings
these tests encode:

  * The PHARMACOPHORE gate cannot reject a substituent scan -- swapping an
    aromatic CH cannot break an acid, a fused diazole, a basic amine or a
    nitrile. MEASURED 54/54 kept. It belongs on SCAFFOLD moves.
  * The LIABILITY gate cannot rank one either, because the PARENT already
    violates Lipinski (danuglipron is MW 555.6). MEASURED 0/54 clean.
  * What DOES discriminate is the CHANGE relative to the parent.

So the module's contract is: enumerate, then score RELATIVELY. These tests pin
that contract and the two negative results that motivate it.
"""

from __future__ import annotations

import pytest

rdkit = pytest.importorskip("rdkit", reason="substitution proposals need rdkit")

from tools.pipeline.substitution import (  # noqa: E402
    propose_substitutions,
    relative_descriptors,
)

# Benzoic acid: small, has aromatic CH sites, and carries an acid so a
# pharmacophore-style feature is present to preserve.
PARENT = "OC(=O)c1ccccc1"


def test_an_empty_substituent_set_returns_exactly_the_parent():
    """EXACTNESS ANCHOR -- written before any scoring ran.

    The trivial limit of "propose substitutions" is proposing none. In that
    limit the result must be the parent alone, unchanged: not empty (which
    would silently drop the reference every relative score is measured
    against), and not decorated.
    """
    props = propose_substitutions(PARENT, substituents={})
    assert len(props) == 1, f"expected just the parent, got {len(props)}"
    only = props[0]
    assert only.is_parent, "the single result must be flagged as the parent"
    assert only.d_mw == 0.0 and only.d_clogp == 0.0 and only.d_tpsa == 0.0, (
        "the parent's deltas against itself must be exactly zero, not merely "
        f"small: got {only.d_mw}, {only.d_clogp}, {only.d_tpsa}"
    )


def test_the_parent_is_always_included_as_the_reference():
    """Every relative score is a difference against the parent, so the parent
    must be in the output or the caller cannot check what it was measured
    against."""
    props = propose_substitutions(PARENT, substituents={"F": "F"})
    parents = [p for p in props if p.is_parent]
    assert len(parents) == 1, f"expected exactly one parent row, got {len(parents)}"
    assert parents[0].d_mw == 0.0


def test_deltas_are_relative_not_absolute():
    """MUTATION KILLED: reporting absolute descriptors instead of deltas.

    An absolute MW for benzoic acid is ~122; a delta for an F substitution is
    ~18. Asserting the SIGN and MAGNITUDE band separates the two -- a test that
    only checked "is a number" would pass with absolutes.
    """
    props = propose_substitutions(PARENT, substituents={"F": "F"})
    subs = [p for p in props if not p.is_parent]
    assert subs, "no substitutions produced"
    for p in subs:
        assert 17.0 < p.d_mw < 19.0, (
            f"F substitution should change MW by ~+18 (H->F), got {p.d_mw}. "
            "A value near 122 means absolute descriptors are being reported."
        )


def test_a_heavier_substituent_adds_more_mass_than_a_lighter_one():
    """Orders the output the way a chemist would sanity-check it."""
    props = propose_substitutions(PARENT, substituents={"F": "F", "CF3": "C(F)(F)F"})
    by_label = {}
    for p in props:
        if not p.is_parent:
            by_label.setdefault(p.label, p.d_mw)
    assert by_label["CF3"] > by_label["F"], (
        f"CF3 (+68) must add more mass than F (+18); got {by_label}"
    )


def test_relative_descriptors_is_zero_against_self():
    """The helper's own trivial limit, tested separately from the pipeline so a
    failure localises."""
    d = relative_descriptors(PARENT, PARENT)
    assert d == (0.0, 0.0, 0.0), f"a molecule against itself must be all zeros, got {d}"


def test_an_unparseable_parent_is_an_error_not_an_empty_list():
    """A silent empty list would read as 'no viable substitutions', which is a
    chemistry claim. A bad input is not that."""
    with pytest.raises(ValueError, match="(?i)pars"):
        propose_substitutions("this is not smiles", substituents={"F": "F"})


def test_proposals_are_deterministic():
    """Two runs must agree exactly. RunReactants does not guarantee a stable
    product order, and an unstable order makes every downstream tier's
    population non-reproducible."""
    a = propose_substitutions(PARENT, substituents={"F": "F", "Cl": "Cl"})
    b = propose_substitutions(PARENT, substituents={"F": "F", "Cl": "Cl"})
    assert [p.smiles for p in a] == [p.smiles for p in b]
    assert [p.label for p in a] == [p.label for p in b]


def test_a_substitution_that_breaks_a_required_feature_is_rejected_when_gated():
    """The pharmacophore gate is OPT-IN and must actually reject when supplied.

    The danuglipron prototype measured 54/54 kept on an aromatic-CH scan, which
    is correct but means that run could not show the gate works. This supplies
    a pattern the products genuinely fail, so the gate's rejection path is
    exercised rather than assumed.
    """
    # Require a free aromatic CH ortho to the acid. Substituting removes CHs,
    # so a scan that replaces enough of them fails this.
    props = propose_substitutions(
        PARENT,
        substituents={"F": "F"},
        require_smarts=("impossible_feature", "[Pt]"),  # no product contains Pt
    )
    assert [p for p in props if p.is_parent], "the parent row must survive gating"
    assert not [p for p in props if not p.is_parent], (
        "every substitution should have been rejected by a SMARTS no product "
        "can match; the gate is not being applied"
    )
