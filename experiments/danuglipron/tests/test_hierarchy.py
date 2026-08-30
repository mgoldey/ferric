"""The cost hierarchy must stay ordered, and unvalidated tiers must stay visible.

This file exists because the campaign's central failure was a HIERARCHY failure,
not a physics one: tier 3 was run alone, fed by free-solution conformers, with
no tier 1 to search and no tier 4 to make a final number. Four rounds of
statistics were spent on geometries 2.2-4.1 A from the binding mode.

What is worth pinning is therefore structural, not numeric:
  - the tiers stay ordered by cost (a hierarchy that is not cost-ordered is
    just a list of methods);
  - a tier with no ground-truth validation is REPORTED as unvalidated rather
    than silently trusted.
"""
from __future__ import annotations

from experiments.danuglipron.hierarchy import DANUGLIPRON_HIERARCHY
from tools.campaign.hierarchy import (
    Tier,
    TierOutcome,
    TierSpec,
    describe,
    unvalidated_tiers,
)


def test_tiers_are_ordered_by_increasing_cost():
    """A hierarchy whose costs are not monotone is not a funnel."""
    costs = [t.seconds_per_pose for t in DANUGLIPRON_HIERARCHY]
    assert costs == sorted(costs), f"tier costs are not monotone: {costs}"


def test_tiers_are_ordered_by_tier_number():
    nums = [int(t.tier) for t in DANUGLIPRON_HIERARCHY]
    assert nums == sorted(nums) == list(range(1, len(nums) + 1))


def test_cost_spans_many_orders_of_magnitude():
    """The whole point is that tier 1 is unimaginably cheaper than tier 4. If
    that spread collapses, the funnel has stopped buying anything."""
    lo = DANUGLIPRON_HIERARCHY[0].seconds_per_pose
    hi = DANUGLIPRON_HIERARCHY[-1].seconds_per_pose
    assert hi / lo > 1e6, f"only {hi / lo:.0e}x between cheapest and dearest tier"


def test_the_quantum_tier_is_reported_as_unvalidated():
    """ferric's DFT is tier 4 and has NEVER been used in this campaign. That is
    a real gap and must not be papered over: every fit number to date is GFN2.

    When tier 4 is finally validated against something, this test fails and the
    claim gets updated deliberately rather than by drift.
    """
    unval = [t.method for t in unvalidated_tiers(DANUGLIPRON_HIERARCHY)]
    assert "ferric DFT + dispersion" in unval, (
        "the quantum tier is now marked validated -- update this test and say "
        "in RESULTS.md what it was validated against"
    )


def test_every_validated_tier_states_its_evidence():
    """`validated_by` must be a real sentence, not a boolean in disguise."""
    for t in DANUGLIPRON_HIERARCHY:
        if t.validated:
            assert len(t.validated_by) > 30, (
                f"tier {int(t.tier)} claims validation with a stub: "
                f"{t.validated_by!r}"
            )


def test_search_tier_records_the_redocking_result():
    """The tier-1 claim rests on one measurement; keep the number attached."""
    search = DANUGLIPRON_HIERARCHY[0]
    assert search.tier is Tier.SEARCH
    assert "0.95" in search.validated_by, (
        "the redocking RMSD is the evidence for the search tier; it must stay "
        "in the record"
    )


def test_unvalidated_tiers_is_reachable_in_both_directions():
    """Reachability: the detector must be able to return empty AND non-empty,
    or it is not measuring anything."""
    assert unvalidated_tiers(DANUGLIPRON_HIERARCHY), "expected >=1 unvalidated"
    all_validated = tuple(
        TierSpec(t.tier, t.method, t.seconds_per_pose, t.typical_poses, t.job,
                 validated_by=t.validated_by or "validated in this fixture " * 2)
        for t in DANUGLIPRON_HIERARCHY
    )
    assert unvalidated_tiers(all_validated) == []


def test_describe_flags_unvalidated_tiers_in_its_output():
    out = describe(DANUGLIPRON_HIERARCHY)
    assert "NO" in out, "describe() must make an unvalidated tier visible"
    assert "ferric DFT" in out


def test_tier_outcome_reports_the_funnel_honestly():
    o = TierOutcome(Tier.SEARCH, n_in=20, n_out=4, n_failed=0,
                    note="4/20 poses under 2 A")
    assert o.retained_fraction == 0.2
    empty = TierOutcome(Tier.QUANTUM, n_in=0, n_out=0)
    assert empty.retained_fraction is None, (
        "an empty population must give None, not a division-by-zero or a 0.0 "
        "that reads as 'everything was discarded'"
    )
