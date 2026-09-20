"""What the liability score CAN and CANNOT separate.

The golden path's answer table now claims toxicology "discriminates between
MOTIFS, not between substituents that leave the motif alone". Both halves are
load-bearing -- the first is why the gate is worth running, the second bounds
what its output may be read for -- so both are pinned here.

MEASURED 2026-09-19. The numbers are a rank-only liability density, not a
probability of toxicity; every endpoint's own note says so.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rdkit", reason="the alert provider needs RDKit")

from tools.tox.alerts import RdkitAlertsProvider  # noqa: E402
from tools.tox.assess import assess_smiles  # noqa: E402


def _score(smiles: str) -> float:
    return assess_smiles(
        smiles, providers=[RdkitAlertsProvider()], label=smiles, include_web=False
    ).liability_score


def test_it_orders_known_liability_motifs_sensibly():
    """The half that makes the gate worth running."""
    clean = _score("c1ccccc1")
    ester = _score("CC(=O)Oc1ccccc1C(=O)O")  # aspirin, phenol ester (Brenk)
    nitro = _score("O=[N+]([O-])c1ccccc1")
    catechol = _score("Oc1ccccc1O")  # PAINS + NIH + BMS
    michael = _score("C=CC(=O)c1ccccc1")  # Michael acceptor

    assert clean == 0.0, f"benzene scored {clean}; it hits no alert set"
    assert ester > clean, "a phenol ester must outscore an unsubstituted ring"
    assert nitro > ester, "a nitroaromatic must outscore a phenol ester"
    assert michael > nitro, "a Michael acceptor is the worst of these"
    assert catechol > nitro, "catechol hits three alert sets"


def test_it_does_NOT_separate_substituents_that_leave_the_motif_alone():
    """The half that bounds what the output may be read for.

    A halogen/methyl scan around an unchanged scaffold gives every analogue the
    parent's score, because alerts are SUBSTRUCTURE matches and F/Cl/Me on a
    ring do not hit one. That is correct, and it means the gate REMOVES a
    liability-bearing motif from the set rather than ordering the survivors.

    If this test ever fails it is interesting either way: the alert sets
    changed, or someone made the score sensitive to something it was not.
    """
    from tools.pipeline.substitution import propose_substitutions

    parent = "CC(=O)Oc1ccccc1C(=O)O"
    parent_score = _score(parent)
    props = propose_substitutions(parent, {"F": "F", "Cl": "Cl", "Me": "C"})
    assert len(props) > 3, "need several proposals for this to mean anything"

    scores = {_score(p.smiles) for p in props}
    assert scores == {parent_score}, (
        f"expected every halogen/methyl analogue to score exactly the parent's "
        f"{parent_score}, got {sorted(scores)}. Either the alert sets now catch "
        "one of these substituents, or the score has become sensitive to "
        "something other than substructure hits -- both worth knowing."
    )
