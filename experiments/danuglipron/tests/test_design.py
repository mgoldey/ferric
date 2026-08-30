"""Tests for the analogue design set and its pharmacophore constraint.

THE ANCHOR THIS FILE ENFORCES (CLAUDE.md, Experimental Protocol): every
approximation has a trivial limit where it does nothing. Here that limit is a
"zero-modification analogue" -- an `Analogue` whose SMILES *is* the parent. It
must be indistinguishable from the parent on every axis the campaign measures.
`test_zero_modification_analogue_is_identical_to_parent` is that assert, and it
catches the entire class of analogue-plumbing error (label/SMILES mismatch, a
transform applied to the wrong record, a descriptor computed on a stale mol)
before any ranking is produced.

THE SECOND THING WORTH TESTING is that the pharmacophore check can actually
FAIL. A constraint that passes for everything is not a constraint; it is
decoration. The negative controls exist for exactly this reason, and
`test_negative_controls_break_exactly_their_intended_feature` pins that they
break the feature they claim to and no other -- otherwise a control that broke
three features would not isolate anything.
"""
from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors

from experiments.danuglipron.design import (
    DANUGLIPRON_SMILES,
    GLP1R_PHARMACOPHORE,
    danuglipron_analogues,
)
from tools.morph.design import Analogue, PharmacophoreSpec


def test_every_analogue_smiles_parses():
    """A typo'd SMILES must fail here, not silently drop an arm of the study."""
    for a in danuglipron_analogues():
        assert Chem.MolFromSmiles(a.smiles) is not None, (
            f"analogue {a.label!r} has unparseable SMILES: {a.smiles}"
        )


def test_labels_are_unique():
    labels = [a.label for a in danuglipron_analogues()]
    assert len(labels) == len(set(labels))


def test_smiles_are_distinct_structures():
    """Two analogues resolving to the same molecule would double-count a design
    in the ranking. Compared on canonical SMILES, not the input strings."""
    canon = {}
    for a in danuglipron_analogues():
        c = Chem.MolToSmiles(Chem.MolFromSmiles(a.smiles))
        assert c not in canon, f"{a.label!r} duplicates {canon[c]!r} (same molecule)"
        canon[c] = a.label


def test_parent_entry_matches_the_canonical_danuglipron_smiles():
    parent = next(a for a in danuglipron_analogues() if a.label == "parent")
    assert Chem.MolToSmiles(Chem.MolFromSmiles(parent.smiles)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(DANUGLIPRON_SMILES)
    )


def test_parent_smiles_matches_the_committed_conformer_ensemble():
    """The 3D ensemble in testdata was generated from scripts/fetch_danuglipron.py's
    SMILES. If this module's parent SMILES ever drifts from it, every
    parent-vs-analogue fit comparison silently compares different molecules.
    """
    import re
    from pathlib import Path

    src = Path("scripts/fetch_danuglipron.py").read_text()
    m = re.search(r'^SMILES = "([^"]+)"', src, re.MULTILINE)
    assert m, "could not find SMILES in scripts/fetch_danuglipron.py"
    assert Chem.MolToSmiles(Chem.MolFromSmiles(m.group(1))) == Chem.MolToSmiles(
        Chem.MolFromSmiles(DANUGLIPRON_SMILES)
    )


# ── THE EXACTNESS ANCHOR ──

def test_zero_modification_analogue_is_identical_to_parent():
    """The trivial limit: an analogue that changes nothing must be
    indistinguishable from the parent on every measured axis.

    This is the assert that catches analogue-plumbing errors as a class, and
    per CLAUDE.md it was written before any ranking was produced.
    """
    parent = next(a for a in danuglipron_analogues() if a.label == "parent")
    null = Analogue(
        label="null-modification",
        smiles=DANUGLIPRON_SMILES,
        hypothesis="anchor",
        rationale="trivial limit: no modification at all",
        pharmacophore=GLP1R_PHARMACOPHORE,
    )

    p_mol, n_mol = Chem.MolFromSmiles(parent.smiles), Chem.MolFromSmiles(null.smiles)
    assert Chem.MolToSmiles(n_mol) == Chem.MolToSmiles(p_mol)
    assert Descriptors.MolWt(n_mol) == pytest.approx(Descriptors.MolWt(p_mol), abs=1e-9)
    assert Crippen.MolLogP(n_mol) == pytest.approx(Crippen.MolLogP(p_mol), abs=1e-9)
    assert null.check_pharmacophore() == parent.check_pharmacophore()
    assert null.retains_pharmacophore is parent.retains_pharmacophore

    # And the tox layer -- the other half of the campaign -- must agree too.
    from tools.tox.alerts import RdkitAlertsProvider
    from tools.tox.assess import assess_smiles

    prov = RdkitAlertsProvider()
    a_p = assess_smiles(parent.smiles, providers=[prov])
    a_n = assess_smiles(null.smiles, providers=[prov])
    assert a_n.liability_score == pytest.approx(a_p.liability_score, abs=1e-12)


# ── the constraint must be able to fail ──

def test_all_real_candidates_retain_the_pharmacophore():
    for a in danuglipron_analogues():
        if a.is_negative_control:
            continue
        broken = [k for k, v in a.check_pharmacophore().items() if not v]
        assert not broken, f"candidate {a.label!r} breaks {broken}"


def test_negative_controls_break_exactly_their_intended_feature():
    """A control must isolate ONE feature. Breaking several would confound it."""
    expected = {
        "NC1-methyl-ester": {"acid_or_bioisostere"},
        "NC2-decyano": {"electron_poor_aryl_terminus"},
    }
    controls = [a for a in danuglipron_analogues() if a.is_negative_control]
    assert {a.label for a in controls} == set(expected), (
        "the negative-control set changed; update the expected broken features"
    )
    for a in controls:
        broken = {k for k, v in a.check_pharmacophore().items() if not v}
        assert broken == expected[a.label], (
            f"{a.label!r} breaks {broken}, expected exactly {expected[a.label]}"
        )
        assert a.retains_pharmacophore is False


def test_pharmacophore_check_is_reachable_in_both_directions():
    """Reachability: the spec must be satisfiable AND violable. A check that
    can only ever return one answer returns arithmetic, not measurement."""
    results = [a.retains_pharmacophore for a in danuglipron_analogues()]
    assert True in results and False in results


def test_acid_bioisostere_feature_accepts_bioisosteres_not_just_acids():
    """If the acid feature were written as a literal -COOH, every H3 analogue
    would fail by construction and the whole H3 arm would be untestable. Pin
    that the union SMARTS really does accept the bioisosteres."""
    cases = {
        "carboxylic acid": "c1ccccc1C(=O)O",
        "tetrazole": "c1ccccc1c1nn[nH]n1",
        "acylsulfonamide": "c1ccccc1C(=O)NS(=O)(=O)C",
        "oxadiazolone": "c1ccccc1c1nc(=O)o[nH]1",
    }
    feature = next(f for f in GLP1R_PHARMACOPHORE.features if f[0] == "acid_or_bioisostere")
    patt = Chem.MolFromSmarts(feature[1])
    assert patt is not None
    for name, smi in cases.items():
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"test case {name!r} is itself unparseable"
        assert mol.GetSubstructMatches(patt), f"{name} not recognized as acid-like"
    # ...and it must REJECT the ester, which is the whole point of NC1.
    ester = Chem.MolFromSmiles("c1ccccc1C(=O)OC")
    assert not ester.GetSubstructMatches(patt), "methyl ester must NOT match"


def test_invalid_smarts_in_a_spec_raises():
    """A typo'd SMARTS must raise, not silently report the feature as absent
    (which would misclassify every molecule as pharmacophore-breaking)."""
    spec = PharmacophoreSpec(features=(("bogus", "c1cc(((", 1),))
    with pytest.raises(ValueError, match="invalid SMARTS"):
        spec.check(Chem.MolFromSmiles("c1ccccc1"))


def test_unparseable_analogue_smiles_raises_on_check():
    a = Analogue(label="bad", smiles="C1CC(((", hypothesis="h", rationale="r",
                 pharmacophore=GLP1R_PHARMACOPHORE)
    with pytest.raises(ValueError, match="unparseable SMILES"):
        a.check_pharmacophore()


# ── the design set must actually span its hypotheses ──

def test_every_hypothesis_arm_is_populated():
    arms = {a.hypothesis for a in danuglipron_analogues()}
    for required in ("H1-size-lipophilicity", "H2-metabolic-soft-spot",
                     "H3-acid-bioisostere", "negative-control"):
        assert required in arms, f"no analogue for {required}"


def test_h1_arm_actually_reduces_size():
    """An H1 analogue that does not reduce MW is mislabelled, and would make the
    size hypothesis untestable."""
    parent_mw = Descriptors.MolWt(Chem.MolFromSmiles(DANUGLIPRON_SMILES))
    h1 = [a for a in danuglipron_analogues() if a.hypothesis == "H1-size-lipophilicity"]
    assert h1
    for a in h1:
        mw = Descriptors.MolWt(Chem.MolFromSmiles(a.smiles))
        assert mw < parent_mw, f"{a.label!r} is H1 but MW {mw:.1f} >= parent {parent_mw:.1f}"
    # At least one must clear the rule-of-5 MW cut, or the arm cannot test its
    # own hypothesis.
    assert any(
        Descriptors.MolWt(Chem.MolFromSmiles(a.smiles)) < 500 for a in h1
    ), "no H1 analogue gets under MW 500"


def test_every_analogue_documents_a_hypothesis_and_rationale():
    """A structure with no stated reason cannot be interpreted later."""
    for a in danuglipron_analogues():
        assert a.hypothesis.strip(), f"{a.label!r} has no hypothesis"
        assert len(a.rationale.strip()) > 40, f"{a.label!r} has a stub rationale"


# ── ionization state ──
#
# This section exists because getting it wrong invalidated an entire campaign
# run (2026-08-29). Danuglipron's carboxylic acid has pKa ~4, so it is ANIONIC
# at pH 7.4, and that anion is the salt bridge which is the potency anchor.
# Scored in the 7LCJ pocket field at the cryo-EM geometry: neutral acid
# −22.86 kcal/mol, anion −165.81 kcal/mol. Modelling everything as neutral
# omitted ~143 kcal/mol of precisely the interaction under study, and made the
# methyl-ester negative control look equivalent to the parent.

def test_declared_net_charge_matches_the_scoring_species():
    """A declared charge that disagrees with the SMILES would be handed to the
    QM engine as the electron count -- silently solving for the wrong system."""
    for a in danuglipron_analogues():
        mol = Chem.MolFromSmiles(a.scoring_smiles)
        assert mol is not None, f"{a.label!r} has unparseable scoring SMILES"
        assert Chem.GetFormalCharge(mol) == a.net_charge, (
            f"{a.label!r} declares net_charge={a.net_charge} but its scoring "
            f"SMILES has formal charge {Chem.GetFormalCharge(mol)}"
        )


def test_every_acidic_candidate_is_scored_as_an_anion():
    """Any candidate retaining an ionizable acid/bioisostere must carry -1.

    The exception is the methyl-ester control, which cannot ionize -- and that
    is exactly what makes it the sharpest control in the set.
    """
    for a in danuglipron_analogues():
        if a.label == "NC1-methyl-ester":
            assert a.net_charge == 0, (
                "the methyl-ester control must stay NEUTRAL -- its inability to "
                "ionize is the hypothesis under test"
            )
            assert a.smiles_ionized is None
            continue
        assert a.net_charge == -1, (
            f"{a.label!r} retains an ionizable group but is scored at charge "
            f"{a.net_charge}; at pH 7.4 it should be -1"
        )
        assert a.smiles_ionized is not None


def test_ionized_and_neutral_forms_differ_by_exactly_one_proton():
    """The ionized form must be the SAME molecule minus one H, not a different
    structure introduced by a copy-paste error in the SMILES."""
    for a in danuglipron_analogues():
        if a.smiles_ionized is None:
            continue
        neutral = Chem.AddHs(Chem.MolFromSmiles(a.smiles))
        anion = Chem.AddHs(Chem.MolFromSmiles(a.smiles_ionized))
        n_h = sum(1 for at in neutral.GetAtoms() if at.GetSymbol() == "H")
        a_h = sum(1 for at in anion.GetAtoms() if at.GetSymbol() == "H")
        assert a_h == n_h - 1, (
            f"{a.label!r}: ionized form has {a_h} hydrogens vs {n_h} neutral; "
            "expected exactly one fewer"
        )
        # Heavy-atom composition must be identical.
        def heavy(m):
            from collections import Counter
            return Counter(at.GetSymbol() for at in m.GetAtoms() if at.GetSymbol() != "H")
        assert heavy(neutral) == heavy(anion), (
            f"{a.label!r}: ionized form changed the heavy-atom composition "
            f"({dict(heavy(neutral))} -> {dict(heavy(anion))})"
        )


def test_ionized_forms_still_satisfy_the_pharmacophore():
    """Deprotonation must not break the acid_or_bioisostere feature -- if the
    SMARTS only matched the protonated form, every scored species would read as
    pharmacophore-breaking."""
    from experiments.danuglipron.design import GLP1R_PHARMACOPHORE

    for a in danuglipron_analogues():
        if a.smiles_ionized is None or a.is_negative_control:
            continue
        chk = GLP1R_PHARMACOPHORE.check(Chem.MolFromSmiles(a.smiles_ionized))
        assert chk["acid_or_bioisostere"], (
            f"{a.label!r}'s ANIONIC form does not match the acid feature; the "
            "SMARTS union needs the deprotonated tautomers too"
        )
