"""Tests for `tools.viz.report` — the worked example, and its refusals.

The report exists so the ten plotting functions have one obvious entry point.
Its job is as much to attach CAVEATS as to draw: a figure set handed to someone
without the resolution limit is how a noise-limited campaign turns into a
ranked table.
"""

from __future__ import annotations

import pytest

from tools.viz.report import campaign_report

# M12's real 15 docked-pose pose_fit scores (sd 28.75).
REAL_POSES = {
    "danuglipron": [
        -83.5,
        -129.5,
        -107.8,
        -57.5,
        -119.9,
        -77.4,
        -85.4,
        -159.4,
        -116.6,
        -160.8,
        -106.8,
        -106.2,
        -115.8,
        -99.6,
        -137.2,
    ]
}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def test_a_missing_noise_floor_is_refused():
    """The most common misuse is reading an order off an all-noise heatmap.

    `ddE_noise` has no default because a default would be a guess about someone
    else's protocol, and the guess would silently license that ordering.
    """
    for bad in (None, 0.0, -1.0, "4.68"):
        with pytest.raises(ValueError, match="ddE_noise"):
            campaign_report(ddE_noise=bad, ddE={("F", "C3"): -1.2})


def test_an_entirely_sub_noise_grid_says_so_in_words():
    """MEASURED on danuglipron: every realistic ddE is inside the floor.

    Greying the cells is necessary and not sufficient -- someone reading a
    figure needs the sentence too.
    """
    figs = campaign_report(
        ddE_noise=4.68,
        ddE={("F", "C3"): -1.2, ("CF3", "C3"): 0.8, ("CN", "C3"): -2.1},
    )
    assert figs.heatmap is not None
    joined = " ".join(figs.caveats)
    assert "EVERY ddE" in joined and "Do not order them" in joined


def test_a_partially_resolvable_grid_says_WHICH_are_usable():
    """Not every campaign is entirely sub-noise; the caveat must distinguish."""
    figs = campaign_report(
        ddE_noise=2.0,
        ddE={("F", "C3"): -1.0, ("CF3", "C3"): 12.0},  # one in, one out
    )
    joined = " ".join(figs.caveats)
    assert "1 of 2" in joined, f"expected a partial count, got {figs.caveats}"
    assert "EVERY ddE" not in joined


def test_the_pose_floor_check_derives_n_rather_than_using_a_ratio():
    """A consistent floor must NOT warn; an optimistic one must.

    MEASURED: sd 28.75 with a 4.68 floor is 6.1x -- which a naive "sd >> floor"
    check would flag. It should not: 4.68 is the n=100 averaged floor and is
    CORRECT for these poses. The discriminator is the IMPLIED pose count
    (n = 2*(sd/floor)^2), not the ratio.
    """
    consistent = campaign_report(ddE_noise=4.68, poses=REAL_POSES)
    assert not any("implies averaging" in c for c in consistent.caveats), (
        f"a correctly-stated floor must not warn: {consistent.caveats}"
    )

    optimistic = campaign_report(ddE_noise=0.5, poses=REAL_POSES)
    warn = [c for c in optimistic.caveats if "implies averaging" in c]
    assert warn, "an optimistic floor must be caught"
    assert "6611" in warn[0], f"the implied n must be NAMED: {warn[0]}"

    # THE DISCRIMINATING CASE, and the reason this test is not satisfied by a
    # ratio rule. At floor = 3.0 the two disagree:
    #
    #   implied n = 2*(28.75/3.0)^2 = 184, against 15 poses  -> MUST warn
    #   ratio     = 28.75/3.0       = 9.6x, under a 10x bar  -> would not
    #
    # A `sd > 10*floor` mutant passes every other assertion here; this one
    # kills it. MEASURED by searching for a floor where the rules differ.
    borderline = campaign_report(ddE_noise=3.0, poses=REAL_POSES)
    warn2 = [c for c in borderline.caveats if "implies averaging" in c]
    assert warn2, (
        "floor 3.0 implies averaging over ~184 poses against an ensemble of 15 "
        "and MUST warn -- a ratio rule (9.6x < 10x) would stay silent here"
    )
    assert "184" in warn2[0], f"the implied n must be named: {warn2[0]}"


def test_a_multi_site_grid_warns_about_site_blindness():
    """Cheap descriptors cannot see position; a site axis needs a pose tier."""
    figs = campaign_report(ddE_noise=4.68, ddE={("F", "C3"): -1.0, ("F", "C5"): -1.0})
    assert any("SITE-BLIND" in c for c in figs.caveats)

    single = campaign_report(ddE_noise=4.68, ddE={("F", "C3"): -1.0})
    assert not any("SITE-BLIND" in c for c in single.caveats), (
        "a single-site grid has no site axis to misread"
    )


def test_absent_inputs_give_None_not_an_empty_figure():
    """An empty plot reads as 'measured and found nothing'."""
    figs = campaign_report(ddE_noise=4.68)
    assert figs.heatmap is None and figs.poses is None
    assert figs.funnel is None and figs.liabilities is None
    assert figs.caveats == []


def test_close_releases_every_figure_it_made():
    import matplotlib.pyplot as plt

    before = len(plt.get_fignums())
    figs = campaign_report(
        ddE_noise=4.68,
        ddE={("F", "C3"): -1.0},
        poses=REAL_POSES,
        funnel_stages=["a", "b"],
        funnel_counts=[10, 5],
    )
    assert len(plt.get_fignums()) == before + 3
    figs.close()
    assert len(plt.get_fignums()) == before


def test_a_failure_midway_closes_the_figures_already_built():
    """A raise after the first figure must not leak it into pyplot.

    `campaign_report` owns every figure it has built, and on failure the caller
    never receives the `CampaignFigures` -- so it cannot call `close()`. A valid
    heatmap followed by an invalid liability `parent` is the reachable case.
    Without the cleanup this leaks one figure per attempt, which inside a
    per-candidate loop is how a batch run exhausts memory.

    Counts pyplot's OWN figure registry rather than a flag, so the assertion
    sees the leak itself and not a proxy for it.
    """
    import matplotlib.pyplot as plt

    plt.close("all")
    before = len(plt.get_fignums())
    # A specific type, not a blind `Exception`: a bare catch here would also
    # pass if the call failed for an unrelated reason (a typo in a kwarg, say)
    # and never built a figure at all -- which is the one outcome that would
    # make this test vacuous.
    with pytest.raises(ValueError):
        campaign_report(
            ddE_noise=1.0,
            ddE={("F", "R1"): 0.5},
            # `parent` names a compound absent from `liabilities` -- the
            # liability plot rejects it, AFTER the heatmap has been built.
            liabilities={"cmpd-a": {"hERG": (1.0, False)}},
            parent="not-a-compound-in-this-dict",
        )
    assert len(plt.get_fignums()) == before, (
        f"{len(plt.get_fignums()) - before} figure(s) leaked past the failure"
    )


def test_half_a_funnel_is_rejected_rather_than_silently_dropped():
    """Stages without counts (or vice versa) is a bug, not a smaller report.

    Silently omitting the funnel hides a caller's mistake in the one output
    nobody re-reads -- the reader sees a report with no funnel and concludes
    the campaign had no funnel.
    """
    for kwargs in (
        {"funnel_stages": ["dock", "ff"]},
        {"funnel_counts": [10, 5]},
    ):
        with pytest.raises(ValueError, match="together"):
            campaign_report(ddE_noise=1.0, **kwargs)

    # Both together still works, and neither is still fine.
    figs = campaign_report(
        ddE_noise=1.0, funnel_stages=["dock", "ff"], funnel_counts=[10, 5]
    )
    assert figs.funnel is not None
    figs.close()
    figs = campaign_report(ddE_noise=1.0)
    assert figs.funnel is None
    figs.close()
