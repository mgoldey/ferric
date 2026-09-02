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


def test_quantum_tier_is_validated_with_a_converged_measurement():
    """Tier 4 was unvalidated until 2026-09-02; this pins what changed.

    The previous version of this test asserted the OPPOSITE -- that the
    quantum tier appeared in `unvalidated_tiers` -- and its docstring said it
    would fail when the tier was finally validated, so the claim would be
    updated deliberately instead of drifting. It fired exactly as intended.

    Evidence: 612.4 s, 18 iterations, converged, on the 71-atom neutral acid
    at STO-3G/PBE, on a box verified free of memory pressure for the whole
    run. The prior ">57 min, did not finish" was memory contention (a 7.26 GB
    auto-budget against a ~9.5 GB need), not DFT cost. See RESULTS.md.
    """
    quantum = [t for t in DANUGLIPRON_HIERARCHY if t.tier is Tier.QUANTUM]
    assert len(quantum) == 1
    spec = quantum[0]
    assert spec.validated, "the quantum tier has a converged measurement now"
    assert "612" in spec.validated_by or "18 iterations" in spec.validated_by, (
        "the validating measurement must stay attached to the claim"
    )
    assert spec.method not in [t.method for t in
                               unvalidated_tiers(DANUGLIPRON_HIERARCHY)]


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
    """Reachability: the detector must return empty AND non-empty, or it is
    not measuring anything.

    Both fixtures are built explicitly. This used to assert that the REAL
    hierarchy had at least one unvalidated tier, which stopped being true when
    tier 4 was validated -- coupling a reachability check to a contingent fact
    about the campaign. Now it tests the detector.
    """
    none_validated = tuple(
        TierSpec(t.tier, t.method, t.seconds_per_pose, t.typical_poses, t.job,
                 validated_by=None)
        for t in DANUGLIPRON_HIERARCHY
    )
    assert len(unvalidated_tiers(none_validated)) == len(DANUGLIPRON_HIERARCHY)

    all_validated = tuple(
        TierSpec(t.tier, t.method, t.seconds_per_pose, t.typical_poses, t.job,
                 validated_by=t.validated_by or "validated in this fixture " * 2)
        for t in DANUGLIPRON_HIERARCHY
    )
    assert unvalidated_tiers(all_validated) == []

    # And the real hierarchy is currently fully validated -- if a tier is ever
    # added without evidence, that is a deliberate change, not a silent one.
    assert unvalidated_tiers(DANUGLIPRON_HIERARCHY) == []


def test_describe_flags_unvalidated_tiers_in_its_output():
    """`describe` must make a missing validation visible.

    Driven by an explicitly-unvalidated fixture: the real hierarchy is fully
    validated now, so asserting on it would test the campaign's state rather
    than the rendering.
    """
    unvalidated = tuple(
        TierSpec(t.tier, t.method, t.seconds_per_pose, t.typical_poses, t.job,
                 validated_by=None)
        for t in DANUGLIPRON_HIERARCHY
    )
    assert "NO" in describe(unvalidated), (
        "describe() must make an unvalidated tier visible"
    )

    out = describe(DANUGLIPRON_HIERARCHY)
    assert "ferric DFT" in out
    assert "NO" not in out, "every tier is validated as of 2026-09-02"


def test_tier_outcome_reports_the_funnel_honestly():
    o = TierOutcome(Tier.SEARCH, n_in=20, n_out=4, n_failed=0,
                    note="4/20 poses under 2 A")
    assert o.retained_fraction == 0.2
    empty = TierOutcome(Tier.QUANTUM, n_in=0, n_out=0)
    assert empty.retained_fraction is None, (
        "an empty population must give None, not a division-by-zero or a 0.0 "
        "that reads as 'everything was discarded'"
    )
