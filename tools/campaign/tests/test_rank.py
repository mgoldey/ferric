"""Tests for the ranking layer and, above all, for the metric gate.

The most valuable test in this file is `test_gate_fails_when_a_control_scores_as
_well_as_the_parent`: it proves the gate can FAIL. A gate that always passes is
arithmetic dressed up as a check (CLAUDE.md: "check the pass condition is
REACHABLE"), and this particular gate is what licenses every fit claim in the
campaign, so its ability to reject matters more than its ability to accept.

The second theme is the missing-data discipline. Every axis is Optional, and a
candidate with an unmeasured axis must neither dominate nor be dominated on it.
The failure mode being prevented: an analogue that failed to embed gets
`fit=None`, a `None`-as-0 coercion makes it look like a perfect binder, and it
tops the recommended front.
"""
from __future__ import annotations

import pytest

from tools.campaign.rank import (
    Candidate,
    dominates,
    fit_discriminates_controls,
    format_table,
    noise_exceeds_signal,
    pareto_front,
)


def _set(**overrides) -> list[Candidate]:
    """A parent plus two controls that behave correctly (controls bind worse)."""
    cands = [
        Candidate("parent", liability=0.2, fit_kcal=-30.0, strain_kcal=2.0),
        Candidate("NC1-methyl-ester", liability=0.2, fit_kcal=-18.0,
                  strain_kcal=2.0, is_negative_control=True),
        Candidate("NC2-decyano", liability=0.1, fit_kcal=-20.0,
                  strain_kcal=1.5, is_negative_control=True),
    ]
    for c in cands:
        if c.label in overrides:
            for k, v in overrides[c.label].items():
                setattr(c, k, v)
    return cands


# ── the gate ──

def test_gate_passes_when_controls_bind_worse():
    g = fit_discriminates_controls(_set())
    assert g.passed, g.detail
    assert g.parent_fit == pytest.approx(-30.0)


def test_gate_fails_when_a_control_scores_as_well_as_the_parent():
    """THE reachability test. If this cannot fail, the gate is decoration."""
    g = fit_discriminates_controls(
        _set(**{"NC2-decyano": {"fit_kcal": -31.0}})  # BETTER than parent
    )
    assert not g.passed
    assert "DOES NOT DISCRIMINATE" in g.detail
    assert "NC2-decyano" in g.detail


def test_gate_failure_does_not_assert_an_unverified_cause():
    """The first version of this message claimed the metric was "tracking
    molecular size". That was measured and found FALSE (r(MW, fit) = +0.132 on
    2026-08-29). A failure message naming an unverified cause is worse than one
    naming none, because it sends the next reader down the wrong path.
    """
    g = fit_discriminates_controls(_set(**{"NC2-decyano": {"fit_kcal": -31.0}}))
    assert not g.passed
    assert "most likely tracking molecular size" not in g.detail
    # It should instead hand over a checklist, in priority order.
    assert "Check, in order" in g.detail


def test_gate_failure_reports_noise_domination_when_that_is_the_cause():
    """When the precision data IS available and says noise-dominated, the gate
    should say so rather than offer a generic checklist."""
    cands = [
        Candidate("parent", fit_kcal=-30.0, fit_sem_kcal=40.0),
        Candidate("NC1", fit_kcal=-35.0, fit_sem_kcal=40.0,
                  is_negative_control=True),
    ]
    g = fit_discriminates_controls(cands, parent_label="parent")
    assert not g.passed
    assert "NOISE-DOMINATED" in g.detail


def test_gate_fails_on_a_sub_margin_difference():
    """A control only 0.3 kcal/mol worse is not discrimination, it's noise."""
    g = fit_discriminates_controls(
        _set(**{"NC1-methyl-ester": {"fit_kcal": -29.7}}), margin_kcal=1.0
    )
    assert not g.passed


def test_gate_fails_with_no_controls():
    g = fit_discriminates_controls(
        [Candidate("parent", fit_kcal=-30.0), Candidate("H1a", fit_kcal=-28.0)]
    )
    assert not g.passed
    assert "no negative controls" in g.detail


def test_gate_fails_when_parent_unmeasured():
    g = fit_discriminates_controls(
        [Candidate("parent"), Candidate("NC1", fit_kcal=-10.0, is_negative_control=True)]
    )
    assert not g.passed
    assert "no fit measurement" in g.detail


def test_gate_fails_when_a_control_is_unmeasured():
    """An unmeasured control must not silently count as passing."""
    g = fit_discriminates_controls(
        _set(**{"NC2-decyano": {"fit_kcal": None}})
    )
    assert not g.passed
    assert "no fit measurement" in g.detail


# ── dominance / missing data ──

def test_dominates_requires_better_on_one_and_no_worse_on_any():
    a = {"liability": 0.1, "fit_loss": 0.0, "strain": 1.0}
    b = {"liability": 0.2, "fit_loss": 1.0, "strain": 2.0}
    assert dominates(a, b)
    assert not dominates(b, a)


def test_identical_candidates_do_not_dominate_each_other():
    a = {"liability": 0.1, "fit_loss": 0.0, "strain": 1.0}
    assert not dominates(a, dict(a))


def test_mixed_tradeoff_is_not_dominance():
    a = {"liability": 0.1, "fit_loss": 5.0, "strain": 1.0}
    b = {"liability": 0.5, "fit_loss": 0.0, "strain": 1.0}
    assert not dominates(a, b) and not dominates(b, a)


def test_missing_axis_is_skipped_not_defaulted():
    """A None axis must not act as a 0 (which would be best-possible for
    liability/strain and thus let an unmeasured candidate dominate)."""
    unmeasured = {"liability": None, "fit_loss": None, "strain": None}
    measured = {"liability": 0.5, "fit_loss": 5.0, "strain": 5.0}
    assert not dominates(unmeasured, measured), (
        "a fully unmeasured candidate dominated a measured one -- None is "
        "being coerced to a favorable value"
    )
    assert not dominates(measured, unmeasured)


def test_partial_measurement_compares_only_on_shared_axes():
    a = {"liability": 0.1, "fit_loss": None, "strain": 1.0}
    b = {"liability": 0.2, "fit_loss": 0.0, "strain": 2.0}
    assert dominates(a, b), "should compare on liability+strain only"


# ── the front ──

def test_pareto_front_excludes_negative_controls_by_default():
    """A control can be non-dominated (low liability because it deleted the
    acid); putting it on the recommended front would recommend an inactive."""
    cands = [
        Candidate("parent", liability=0.5, fit_kcal=-30.0, strain_kcal=2.0),
        Candidate("NC2", liability=0.0, fit_kcal=-20.0, strain_kcal=0.0,
                  is_negative_control=True),
    ]
    front = pareto_front(cands)
    assert [c.label for c in front] == ["parent"]
    assert "NC2" in [c.label for c in pareto_front(cands, exclude_controls=False)]


def test_pareto_front_keeps_the_tradeoff_set_and_drops_the_dominated():
    cands = [
        Candidate("parent", liability=0.4, fit_kcal=-30.0, strain_kcal=3.0),
        Candidate("good_all_round", liability=0.2, fit_kcal=-32.0, strain_kcal=1.0),
        Candidate("dominated", liability=0.5, fit_kcal=-25.0, strain_kcal=4.0),
        Candidate("low_liab_worse_fit", liability=0.05, fit_kcal=-22.0, strain_kcal=3.0),
    ]
    labels = {c.label for c in pareto_front(cands)}
    assert "good_all_round" in labels
    assert "low_liab_worse_fit" in labels, "a genuine trade-off must stay"
    assert "dominated" not in labels
    assert "parent" not in labels, "parent is dominated by good_all_round here"


def test_fit_loss_is_relative_to_the_parent_and_signed_correctly():
    """fit_loss must be POSITIVE for a worse binder, since all three axes are
    oriented lower-is-better. A sign slip here inverts the entire ranking."""
    parent = Candidate("parent", fit_kcal=-30.0)
    worse = Candidate("worse", fit_kcal=-25.0)
    better = Candidate("better", fit_kcal=-35.0)
    assert worse.axes(parent.fit_kcal)["fit_loss"] == pytest.approx(5.0)
    assert better.axes(parent.fit_kcal)["fit_loss"] == pytest.approx(-5.0)


def test_format_table_prints_missing_axes_as_dashes_not_zeros():
    """A 0.00 in a report reads as a measurement. Missing must look missing."""
    out = format_table([
        Candidate("parent", liability=0.2, fit_kcal=-30.0, strain_kcal=1.0),
        Candidate("unmeasured"),
    ])
    unmeasured_row = [l for l in out.splitlines() if l.startswith("unmeasured")][0]
    assert "--" in unmeasured_row
    assert "0.00" not in unmeasured_row, (
        "an unmeasured axis rendered as 0.00, which reads as a real measurement"
    )


# ── the precision check ──

def test_precision_check_fails_when_noise_exceeds_signal():
    """A 16 kcal/mol candidate range with 8 kcal/mol standard errors cannot be
    ordered: the resolution limit is 2*8*sqrt(2) = 22.6 kcal/mol."""
    cands = [
        Candidate("a", fit_kcal=-20.0, fit_sem_kcal=8.0),
        Candidate("b", fit_kcal=-30.0, fit_sem_kcal=8.0),
        Candidate("c", fit_kcal=-14.0, fit_sem_kcal=8.0),
    ]
    r = noise_exceeds_signal(cands)
    assert not r.passed
    assert "NOISE-DOMINATED" in r.detail
    assert "1/sqrt(n)" in r.detail


def test_precision_uses_the_standard_error_not_the_pose_range():
    """THE correction. A range grows with sample count while the precision of a
    mean falls as 1/sqrt(n), so judging precision by the range rejects metrics
    whose means are perfectly good. Measured 2026-08-29: pose ranges of 118-253
    kcal/mol against SEMs of 5-10 kcal/mol at n~40.

    A wide RANGE with a tight SEM must PASS.
    """
    cands = [
        Candidate("a", fit_kcal=-119.8, fit_sem_kcal=7.4, fit_range_kcal=179.1),
        Candidate("b", fit_kcal=-165.6, fit_sem_kcal=5.0, fit_range_kcal=118.5),
    ]
    r = noise_exceeds_signal(cands)
    assert r.passed, (
        "a 45.8 kcal/mol difference between means with ~6 kcal/mol standard "
        f"errors must be resolvable; got: {r.detail}"
    )


def test_precision_check_passes_when_the_estimator_is_precise():
    """Reachability in the other direction: a precise estimator must PASS, or
    the check is an unconditional rejection rather than a measurement."""
    cands = [
        Candidate("a", fit_kcal=-20.0, fit_sem_kcal=0.5),
        Candidate("b", fit_kcal=-30.0, fit_sem_kcal=0.5),
    ]
    r = noise_exceeds_signal(cands)
    assert r.passed, r.detail
    assert "precision adequate" in r.detail
    # The pass message must NOT be read as licensing every pairwise call.
    assert "pairwise" in r.detail


# ── pairwise significance ──

def test_pairwise_significance_distinguishes_clear_and_unclear_pairs():
    from tools.campaign.rank import significant_difference

    parent = Candidate("parent", fit_kcal=-119.8, fit_sem_kcal=7.4)
    far = Candidate("far", fit_kcal=-22.9, fit_sem_kcal=1.7)      # 96.9 apart
    near = Candidate("near", fit_kcal=-123.1, fit_sem_kcal=6.9)   # 3.3 apart

    assert significant_difference(parent, far) is True
    assert significant_difference(parent, near) is False, (
        "a 3.3 kcal/mol gap with ~7 kcal/mol standard errors is not a difference"
    )


def test_pairwise_significance_is_none_without_precision():
    from tools.campaign.rank import significant_difference

    a = Candidate("a", fit_kcal=-10.0)
    b = Candidate("b", fit_kcal=-20.0, fit_sem_kcal=1.0)
    assert significant_difference(a, b) is None, (
        "an unmeasured precision must give None, not a confident True"
    )


def test_pairwise_significance_is_symmetric():
    from tools.campaign.rank import significant_difference

    a = Candidate("a", fit_kcal=-100.0, fit_sem_kcal=5.0)
    b = Candidate("b", fit_kcal=-130.0, fit_sem_kcal=5.0)
    assert significant_difference(a, b) == significant_difference(b, a)


def test_precision_check_needs_two_candidates_with_precision():
    assert not noise_exceeds_signal([Candidate("a", fit_kcal=-1.0, fit_sem_kcal=1.0)]).passed
    assert not noise_exceeds_signal([Candidate("a", fit_kcal=-1.0)]).passed


def test_precision_check_ignores_candidates_missing_either_value():
    """A candidate with a fit but no precision estimate must not be silently
    treated as noise-free (which would make any metric look precise)."""
    cands = [
        Candidate("a", fit_kcal=-20.0, fit_sem_kcal=40.0),
        Candidate("b", fit_kcal=-30.0, fit_sem_kcal=40.0),
        Candidate("no_precision", fit_kcal=-25.0),
    ]
    r = noise_exceeds_signal(cands)
    assert not r.passed, "the precision-free candidate diluted the mean SEM"


# ── the charge confound ──
#
# This check exists because the campaign was misled by exactly this. The one
# neutral candidate (the methyl-ester control) separated from the ten anions by
# 109.5 kcal/mol against a 41.4 kcal/mol spread among the anions themselves.
# That clean separation read as "the metric discriminates the pharmacophore"
# when it was measuring ionization state.

def test_charge_confound_flags_a_mixed_charge_set():
    from tools.campaign.rank import charge_confound

    cands = [Candidate(f"anion{i}", fit_kcal=f, net_charge=-1)
             for i, f in enumerate([-120.0, -140.0, -160.0])]
    cands.append(Candidate("neutral", fit_kcal=-22.0, net_charge=0))
    r = charge_confound(cands)
    assert not r.passed
    assert "CHARGE-DOMINATED" in r.detail
    assert "ionization" in r.detail


def test_charge_confound_passes_for_a_single_charge_state():
    """The clean case: everything at the same charge, nothing to confound."""
    from tools.campaign.rank import charge_confound

    cands = [Candidate(f"a{i}", fit_kcal=f, net_charge=-1)
             for i, f in enumerate([-120.0, -140.0, -160.0])]
    r = charge_confound(cands)
    assert r.passed
    assert "every scored candidate has net charge -1" in r.detail


def test_charge_confound_passes_when_within_spread_exceeds_between():
    """Reachability the other way: a mixed-charge set where charge is NOT the
    dominant axis must pass, or the check is an unconditional rejection of any
    mixed set rather than a measurement."""
    from tools.campaign.rank import charge_confound

    cands = [Candidate("a", fit_kcal=-10.0, net_charge=-1),
             Candidate("b", fit_kcal=-90.0, net_charge=-1),
             Candidate("c", fit_kcal=-45.0, net_charge=0),
             Candidate("d", fit_kcal=-55.0, net_charge=0)]
    r = charge_confound(cands)
    assert r.passed, r.detail


def test_charge_confound_reproduces_the_campaign_numbers():
    """Pin the real case: 10 anions plus 1 neutral must be flagged."""
    from tools.campaign.rank import charge_confound

    anions = [-159.8, -144.6, -135.3, -131.5, -130.6, -126.3, -126.1, -124.6,
              -121.1, -118.4]
    cands = [Candidate(f"a{i}", fit_kcal=f, net_charge=-1)
             for i, f in enumerate(anions)]
    cands.append(Candidate("NC1-methyl-ester", fit_kcal=-22.3, net_charge=0,
                           is_negative_control=True))
    r = charge_confound(cands)
    assert not r.passed
    # between ~ -132 vs -22 = ~110; within (anions) ~ 41
    assert "109" in r.detail or "110" in r.detail or "111" in r.detail
