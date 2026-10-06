"""#323: the funnel must not rank a TOTAL energy across differing formulas.

A total energy tracks electron count, so a chloro analogue "beats" its fluoro
sibling by ~8 electrons' worth of Hartree whatever it does in the pocket. The
funnel was built for isomers (same formula, where totals compare) and was then
fed substitution analogues (`pipeline.substitution`), where they do not.

Anchors:
* same-formula populations cut EXACTLY as before (`formula_bound` changes
  nothing for isomers);
* a stage that keeps everyone makes no ranking decision and is allowed;
* `paired_delta` of a candidate against itself is exactly 0.
"""

from __future__ import annotations

import pytest

from tools.campaign.hierarchy import Tier
from tools.isomers.model import Isomer
from tools.pipeline.funnel import (
    IncomparableError,
    Stage,
    formula,
    paired_delta,
    run_funnel,
)
from tools.pipeline.tiers import TierResult, tier2_forcefield

PARENT = "OC(=O)c1ccccc1"
FLUORO = Isomer("OC(=O)c1ccc(F)cc1", "substitutional", "F", PARENT)
CHLORO = Isomer("OC(=O)c1ccc(Cl)cc1", "substitutional", "Cl", PARENT)
# Three C3H8O isomers: one formula.
ISOMERS = [Isomer(s, "structural", "t", "CCCO") for s in ("CCCO", "CC(C)O", "COCC")]


def _stub(values, *, formula_bound, resolution=None):
    def fn(iso, ctx):
        return TierResult(
            iso.canonical,
            values[iso.canonical],
            formula_bound=formula_bound,
            resolution=resolution,
        )

    return fn


def test_the_REAL_force_field_tier_is_not_ranked_across_formulas():
    """Fails on main: there the MMFF totals are sorted and one analogue wins.

    Real tier 2, real MMFF totals for fluoro vs chloro benzoic acid, keep=1.
    """
    with pytest.raises(IncomparableError, match="differing formulas"):
        run_funnel(
            [FLUORO, CHLORO],
            [Stage(Tier.FORCE_FIELD, tier2_forcefield, keep=1, name="ff")],
            {"seed": 1},
        )


def test_a_heavier_halogen_cannot_win_on_its_electrons():
    """A stub with GFN2-like totals where Cl is lower only by electron count."""
    vals = {FLUORO.canonical: -40.0, CHLORO.canonical: -45.0}
    with pytest.raises(IncomparableError):
        run_funnel(
            [FLUORO, CHLORO],
            [Stage(Tier.SEMIEMPIRICAL, _stub(vals, formula_bound=True), 1, "xtb")],
            {},
        )


def test_a_comparable_score_still_ranks_across_formulas():
    """An interaction energy (formula_bound=False) is ranked as before."""
    vals = {FLUORO.canonical: -0.010, CHLORO.canonical: -0.004}
    rep = run_funnel(
        [CHLORO, FLUORO],
        [Stage(Tier.SEMIEMPIRICAL, _stub(vals, formula_bound=False), 1, "xtb")],
        {},
    )
    assert [i.canonical for i in rep.survivors] == [FLUORO.canonical]


def test_ISOMERS_are_cut_exactly_as_before():
    """The exactness anchor: same formula, so formula_bound is a no-op."""
    vals = {iso.canonical: v for iso, v in zip(ISOMERS, (-3.0, -5.0, -4.0))}
    assert len({formula(i) for i in ISOMERS}) == 1
    runs = [
        run_funnel(
            list(ISOMERS),
            [Stage(Tier.QUANTUM, _stub(vals, formula_bound=fb), 2, "dft")],
            {},
        )
        for fb in (False, True)
    ]
    got = [[i.canonical for i in r.survivors] for r in runs]
    assert got[0] == got[1] == [ISOMERS[1].canonical, ISOMERS[2].canonical]


def test_a_stage_that_does_not_cut_passes_differing_formulas_through_unranked():
    vals = {FLUORO.canonical: -40.0, CHLORO.canonical: -45.0}
    rep = run_funnel(
        [FLUORO, CHLORO],
        [Stage(Tier.FORCE_FIELD, _stub(vals, formula_bound=True), 2, "ff")],
        {},
    )
    # INPUT order, not the electron-count order (-45 first) a sort would give.
    assert [i.canonical for i in rep.survivors] == [FLUORO.canonical, CHLORO.canonical]
    assert "unranked" in rep.outcomes[0].note


def test_the_charge_is_part_of_the_formula():
    """Same atoms, different electron count: methyl cation, radical, anion.
    Their totals are no more comparable than two substituents'."""
    forms = {
        formula(Isomer(s, "parent", "t", s, net_charge=q))
        for s, q in (("[CH3+]", 1), ("[CH3]", 0), ("[CH3-]", -1))
    }
    assert len(forms) == 3, forms


# ── paired difference against the parent ────────────────────────────────────


def test_a_self_pair_is_exactly_zero():
    poses = [-0.0123456789, -1.5e-3, 7.25, -39.621219]
    assert paired_delta(poses, poses) == [0.0] * len(poses)


def test_paired_delta_is_analogue_minus_parent_pose_by_pose():
    assert paired_delta([1.0, 2.0], [1.5, 1.0]) == [0.5, -1.0]


def test_an_unpaired_remainder_is_refused():
    with pytest.raises(ValueError, match="cannot pair"):
        paired_delta([1.0, 2.0], [1.0])


# ── resolution: the cut does not separate what the tier cannot resolve ─────


def test_candidates_inside_the_resolution_are_not_separated_by_the_cut():
    """a and b are 0.5 apart with resolution 1.0 each (combined 1.41): a tie.
    keep=1 would drop one of them on sort order alone; both must survive."""
    vals = {
        ISOMERS[0].canonical: -10.0,
        ISOMERS[1].canonical: -9.5,
        ISOMERS[2].canonical: -2.0,
    }
    rep = run_funnel(
        list(ISOMERS),
        [Stage(Tier.QUANTUM, _stub(vals, formula_bound=False, resolution=1.0), 1, "q")],
        {},
    )
    assert {i.canonical for i in rep.survivors} == {
        ISOMERS[0].canonical,
        ISOMERS[1].canonical,
    }
    assert "unresolved group" in rep.outcomes[0].note


def test_a_resolved_gap_is_still_cut():
    """Reachability: the same setup with the pair 5 apart IS separated."""
    vals = {
        ISOMERS[0].canonical: -10.0,
        ISOMERS[1].canonical: -5.0,
        ISOMERS[2].canonical: -2.0,
    }
    rep = run_funnel(
        list(ISOMERS),
        [Stage(Tier.QUANTUM, _stub(vals, formula_bound=False, resolution=1.0), 1, "q")],
        {},
    )
    assert [i.canonical for i in rep.survivors] == [ISOMERS[0].canonical]
