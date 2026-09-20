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


def test_a_canonicalised_parent_is_zero_only_to_a_TOLERANCE():
    """The same molecule spelled two ways is NOT bit-identical in dMW.

    `relative_descriptors(s, s)` is exactly (0,0,0) -- the test above. But
    `props[0].smiles` is the CANONICAL form of the input, and comparing those
    two spellings gives dMW = -1.42e-14:

        relative_descriptors("O=C(O)c1ccccc1", "c1ccccc1C(=O)O")

    Same molecule, different atom ORDER, so the molecular-weight float sum
    lands one ulp apart. The golden-path quickstart printed this as
    "(-0.0, 0.0, 0.0) ... exactly zero, by construction" for a day, which is
    exactly the belief that produces an `== 0.0` gate.

    Pinned because the parent rides through every stage as the ddE anchor: a
    strict-equality check on "is this the parent?" would drop it from a
    campaign, silently, and only for inputs written in a non-canonical form.
    """
    from rdkit import Chem

    spelling_a = "c1ccccc1C(=O)O"
    spelling_b = Chem.CanonSmiles(spelling_a)
    assert spelling_a != spelling_b, (
        "this test needs two SPELLINGS of one molecule; pick another input"
    )

    d = relative_descriptors(spelling_b, spelling_a)
    assert d != (0.0, 0.0, 0.0), (
        f"expected a sub-ulp difference between two spellings, got exact {d} "
        "-- if the implementation now canonicalises both sides this test is "
        "obsolete rather than failing, but the doc comment must change too"
    )
    for v in d:
        assert abs(v) < 1e-9, (
            f"a spelling difference must be NUMERICAL NOISE, got {d}; "
            "anything larger means the two spellings are different molecules"
        )


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


# ── embedding: proposal -> 3-D coordinates ────────────────────────────────
#
# `SubstitutionProposal` carries SMILES; every pocket-side entry point
# (`embed_ligand_from_coords`, `batch_prescreen`, `compute_binding_energy`)
# needs 3-D coordinates. That hop is the ONLY missing piece between the
# enumeration half and the pocket half -- see
# wiki/substitution-pipeline-danuglipron-2026-09-19.md.


def test_embedding_returns_one_geometry_per_proposal():
    from tools.pipeline.substitution import embed_proposals, propose_substitutions

    props = propose_substitutions(PARENT, substituents={"F": "F"})
    embedded = embed_proposals(props)
    assert len(embedded) == len(props), (
        f"{len(embedded)} geometries for {len(props)} proposals -- every "
        "proposal must be embedded or explicitly reported as failed"
    )


def test_the_parent_is_embedded_too():
    """The parent is the reference every ddE is measured against, so it must
    reach the pocket alongside the analogues, not be filtered out as 'not a
    substitution'."""
    from tools.pipeline.substitution import embed_proposals, propose_substitutions

    embedded = embed_proposals(propose_substitutions(PARENT, substituents={"F": "F"}))
    parents = [e for e in embedded if e.proposal.is_parent]
    assert len(parents) == 1, f"expected one embedded parent, got {len(parents)}"
    assert parents[0].coords, "the parent was embedded with no coordinates"


def test_embedded_geometry_is_three_dimensional():
    """MUTATION KILLED: returning a flat or zeroed geometry.

    ETKDG must produce a real 3-D structure. A planar or collapsed geometry
    would still have the right SHAPE (n_atoms x 3) and would silently give
    nonsense in the pocket, so this asserts genuine extent in all three axes.
    """
    from tools.pipeline.substitution import embed_proposals, propose_substitutions

    e = embed_proposals(propose_substitutions(PARENT, substituents={}))[0]
    assert len(e.symbols) == len(e.coords), "symbols and coords disagree in length"
    assert len(e.coords) > 3, f"benzoic acid should have >3 atoms, got {len(e.coords)}"
    for axis, name in enumerate("xyz"):
        spread = max(c[axis] for c in e.coords) - min(c[axis] for c in e.coords)
        assert spread > 0.5, (
            f"the {name} extent is {spread:.3f} A -- the geometry is flat or "
            "collapsed, not a real 3-D embedding"
        )


def test_embedding_is_deterministic():
    """ETKDG is stochastic; an unseeded embedding makes every downstream
    energy irreproducible. Two calls must agree exactly."""
    from tools.pipeline.substitution import embed_proposals, propose_substitutions

    props = propose_substitutions(PARENT, substituents={"F": "F"})
    a = embed_proposals(props)
    b = embed_proposals(props)
    assert [x.coords for x in a] == [x.coords for x in b], (
        "two embeddings of the same proposals disagree -- the ETKDG seed is "
        "not being pinned"
    )


def test_an_unembeddable_proposal_is_reported_not_dropped():
    """A proposal ETKDG cannot embed must come back with `coords is None` and
    an error string, never be silently absent.

    Dropping it would make the output length disagree with the input and, worse,
    would read downstream as 'this analogue was not proposed' rather than 'this
    analogue could not be embedded' -- the same distinction
    tools/campaign/hierarchy.py rule 5 insists on.
    """
    from tools.pipeline.substitution import EmbeddedProposal, SubstitutionProposal

    # A proposal whose SMILES cannot be parsed at all.
    bad = SubstitutionProposal(
        smiles="not-smiles",
        label="bogus",
        is_parent=False,
        d_mw=0.0,
        d_clogp=0.0,
        d_tpsa=0.0,
    )
    from tools.pipeline.substitution import embed_proposals

    out = embed_proposals([bad])
    assert len(out) == 1, "the failed proposal was dropped instead of reported"
    assert isinstance(out[0], EmbeddedProposal)
    assert out[0].coords is None, "an unembeddable proposal must have no coordinates"
    assert out[0].error, "an unembeddable proposal must carry an error string"


def test_relative_descriptors_cannot_distinguish_SITES():
    """The cheap gate ranks SUBSTITUENTS, never PLACEMENTS -- pinned.

    The pipeline's stated unit is the (substituent, SITE) pair: MEASURED
    within/between ratio 0.94-0.95, so where a group goes matters as much as
    which group. This records that the cheap gate CANNOT express that unit,
    and that this is inherent rather than fixable.

    MW, cLogP (Crippen) and TPSA are whole-molecule sums over atoms and
    fragment types. Constitutional isomers share both, so all three are
    identical BY CONSTRUCTION -- no change to `relative_descriptors` alters
    that. Anyone reaching for this gate to choose a position needs a
    positional descriptor (3-D shape, per-atom charge, a QM property at the
    site), which is an addition, not a fix.

    Pinned as a test rather than left in a note because the failure mode is
    someone reading a ranked substituent table and assuming the ordering also
    tells them where to put the group.
    """
    ortho = relative_descriptors("Fc1ccccc1C(=O)O", "O=C(O)c1ccccc1")
    meta = relative_descriptors("O=C(O)c1cccc(F)c1", "O=C(O)c1ccccc1")
    para = relative_descriptors("O=C(O)c1ccc(F)cc1", "O=C(O)c1ccccc1")

    assert ortho == meta == para, (
        "these three differ only in WHERE the fluorine sits, and whole-molecule "
        f"descriptors cannot see that: {ortho} / {meta} / {para}"
    )

    # Vacuity guard: the gate MUST still distinguish different SUBSTITUENTS, or
    # the assertion above would pass for a function that returns a constant.
    chlorine = relative_descriptors("Clc1ccccc1C(=O)O", "O=C(O)c1ccccc1")
    assert chlorine != ortho, (
        "F and Cl must give different descriptor deltas; if they do not, the "
        "gate is inert rather than merely site-blind"
    )


def test_embed_proposals_warns_in_its_OWN_docstring_about_placement():
    """The warning has to live where a caller reads it.

    `embed_proposals` returns ETKDG conformers centred on the ORIGIN; a pocket
    from a PDB sits at its crystal coordinates. MEASURED on 7LCJ, 226 A apart.
    Feeding one to an embedded SCF is not an error -- it converges and reports
    ~0.00 kcal/mol, a gas-phase answer wearing a QM/MM label.

    The golden path documented this and its own text said the honest thing:
    "nothing in either signature says so". A caller reads the docstring, not
    the reference page, so the warning belongs in both. This pins the
    docstring half -- the one that is in reach at the call site.
    """
    from tools.pipeline.substitution import embed_proposals

    doc = embed_proposals.__doc__ or ""
    assert "ORIGIN" in doc, "the docstring must say the coordinates are origin-centred"
    assert "226" in doc, (
        "the docstring must carry the MEASURED separation -- a warning without "
        "a number reads as a theoretical caveat"
    )
    for remedy in ("dock", "coords_angstrom"):
        assert remedy in doc, f"the docstring must name the remedy ({remedy!r})"
