"""Tests for `tools.viz.energy_plots`.

These assert the PROPERTIES the module docstring promises, not pixels. A test
that compares rendered images breaks on a matplotlib version bump and tells you
nothing about whether a gap was drawn as a zero.

The load-bearing ones are the "must refuse" cases: a plot that silently accepts
a wrong unit, or draws an unevaluated point at 0.0, is the failure this module
exists to prevent, so those get the exactness-anchor treatment -- the trivial
limit first, then the defect.
"""

from __future__ import annotations

import pytest

from tools.viz.energy_plots import (
    HARTREE_TO_KCAL,
    SeriesPoint,
    energy_profile,
    funnel_survival,
    liability_profile,
    pose_ensemble,
    site_substituent_heatmap,
    tier_comparison,
)


def _line_ys(fig):
    """Every finite y-value drawn as DATA on the figure.

    Excludes `axvline` artists. Their y-data is [0, 1] in AXES coordinates, not
    data coordinates, so including them injects a spurious 1.0 -- which is
    exactly what this helper got wrong on first writing, and the gap test
    caught it. An axvline is identifiable by its transform not being the axes'
    data transform.
    """
    ys: list[float] = []
    for ax in fig.axes:
        for ln in ax.get_lines():
            if ln.get_transform() != ax.transData:
                continue  # axvline/axhline: y is in axes coords
            ys.extend(float(v) for v in ln.get_ydata() if v == v)
    return ys


# --- unit handling: the 627x error ------------------------------------------


def test_hartree_is_converted_and_kcal_is_not():
    """The trivial limit first: a kcal/mol input must pass through untouched."""
    pts = [SeriesPoint("a", 0.0), SeriesPoint("b", 5.0)]
    fig = energy_profile(pts, unit="kcal/mol", relative_to=0)
    assert max(_line_ys(fig)) == pytest.approx(5.0)

    # Same numbers labelled Hartree must come out 627.5x larger.
    fig_h = energy_profile(pts, unit="hartree", relative_to=0)
    assert max(_line_ys(fig_h)) == pytest.approx(5.0 * HARTREE_TO_KCAL)


def test_an_unknown_unit_is_an_error_not_a_default():
    """Guessing the unit is a 627x error that still produces a valid-looking plot."""
    with pytest.raises(ValueError, match="unknown unit"):
        energy_profile([SeriesPoint("a", 1.0)], unit="eV")


# --- the central promise: a gap is not a zero -------------------------------


def test_an_unevaluated_point_is_a_gap_not_a_zero():
    """`value=None` must not be plotted at 0.0.

    A relative profile legitimately CONTAINS 0.0 (the reference), so the test
    asserts on the count: with one real non-reference point there must be
    exactly one non-zero y, and the None must contribute nothing.
    """
    pts = [
        SeriesPoint("start", -10.0),
        SeriesPoint("unconverged", None),
        SeriesPoint("end", -8.0),
    ]
    fig = energy_profile(pts, unit="kcal/mol", relative_to=0)
    ys = _line_ys(fig)
    # start -> 0.0, end -> +2.0, and NOTHING from the None point.
    assert sorted(v for v in ys if v != 0.0) == pytest.approx([2.0])
    assert len([v for v in ys if v == 0.0]) >= 1


def test_the_line_breaks_across_a_gap_rather_than_interpolating():
    """A path with a hole must not be drawn as a smooth curve through it."""
    pts = [
        SeriesPoint("a", 0.0),
        SeriesPoint("b", 1.0),
        SeriesPoint("gap", None),
        SeriesPoint("d", 3.0),
        SeriesPoint("e", 4.0),
    ]
    fig = energy_profile(pts, unit="kcal/mol", relative_to=None)
    # Two contiguous runs => at least two separate solid line artists.
    solid = [
        ln
        for ax in fig.axes
        for ln in ax.get_lines()
        if ln.get_linestyle() == "-" and len(ln.get_xdata()) > 1
    ]
    assert len(solid) >= 2, "the line must break at the gap, not span it"


def test_a_missing_reference_point_is_an_error():
    """Zeroing the plot on a point that was never evaluated is meaningless."""
    with pytest.raises(ValueError, match="cannot be the zero"):
        energy_profile(
            [SeriesPoint("a", None), SeriesPoint("b", 1.0)],
            unit="kcal/mol",
            relative_to=0,
        )


# --- SEM discipline ---------------------------------------------------------


def test_sem_bars_are_drawn_only_when_n_exceeds_one():
    """A single sample has no measured spread; a zero-length bar implies one."""

    def n_errorbars(fig):
        # Error bars are LineCollections on the axes.
        from matplotlib.collections import LineCollection

        return sum(
            1
            for ax in fig.axes
            for c in ax.collections
            if isinstance(c, LineCollection)
        )

    one = [SeriesPoint("a", 0.0, sem=0.3, n=1), SeriesPoint("b", 1.0, sem=0.3, n=1)]
    many = [SeriesPoint("a", 0.0, sem=0.3, n=5), SeriesPoint("b", 1.0, sem=0.3, n=5)]
    assert n_errorbars(energy_profile(one, unit="kcal/mol")) == 0
    assert n_errorbars(energy_profile(many, unit="kcal/mol")) > 0


# --- funnel invariants ------------------------------------------------------


def test_a_funnel_cannot_grow():
    """More candidates out than in is not a funnel stage."""
    with pytest.raises(ValueError, match="non-increasing"):
        funnel_survival(["enumerate", "dock", "xtb"], [10, 40, 5])


def test_funnel_rejects_mismatched_lengths_and_negative_counts():
    with pytest.raises(ValueError, match="one-to-one"):
        funnel_survival(["a", "b"], [5])
    with pytest.raises(ValueError, match="negative"):
        funnel_survival(["a", "b"], [5, -1])


def test_funnel_labels_carry_the_fraction_of_the_original_input():
    """Per-stage retention and cumulative survival look identical unstated."""
    fig = funnel_survival(["in", "dock", "dft"], [100, 40, 4])
    texts = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "40%" in texts and "4%" in texts


# --- tier comparison --------------------------------------------------------


def test_tier_comparison_omits_a_missing_score_rather_than_barring_zero():
    """A tier that did not score a candidate must not push it to the bottom."""
    fig = tier_comparison(
        ["c1", "c2", "c3"],
        {"xtb": [-1.0, -2.0, None], "dft": [-1.2, -2.1, -3.0]},
        unit="kcal/mol",
    )
    bars = [p for ax in fig.axes for p in ax.patches]
    # 2 real xtb + 3 dft = 5 bars, NOT 6.
    assert len(bars) == 5
    assert not any(b.get_height() == 0.0 for b in bars)
    assert "unevaluated" in " ".join(ax.get_title() for ax in fig.axes)


def test_tier_comparison_rejects_a_ragged_series():
    with pytest.raises(ValueError, match="values but there are"):
        tier_comparison(["a", "b"], {"xtb": [1.0]})


# --- site x substituent heatmap ---------------------------------------------


def test_an_unevaluated_pair_is_hatched_not_coloured_zero():
    """A missing (substituent, site) must not land on the colormap midpoint.

    A diverging colormap puts 0.0 in the middle, so an unevaluated pair drawn
    as 0.0 reads as "measured, and neutral" -- the one reading it must never
    have. It gets a hatched patch and an "n/a" label instead.
    """
    fig = site_substituent_heatmap(
        {("F", "C3"): -1.2, ("F", "C5"): None, ("Cl", "C3"): 0.8, ("Cl", "C5"): 2.0}
    )
    ax = fig.axes[0]
    hatched = [p for p in ax.patches if p.get_hatch()]
    assert len(hatched) == 1, (
        f"expected 1 hatched cell for the None, got {len(hatched)}"
    )
    labels = [t.get_text() for t in ax.texts]
    assert "n/a" in labels
    # And the n/a cell must NOT also carry a numeric label.
    assert sum(1 for t in labels if t.startswith(("+", "-"))) == 3


def test_a_missing_key_is_treated_the_same_as_an_explicit_none():
    """Absent and None are both "not evaluated" and must render identically."""
    explicit = site_substituent_heatmap({("F", "A"): 1.0, ("F", "B"): None})
    implied = site_substituent_heatmap({("F", "A"): 1.0, ("Cl", "B"): 1.0})
    assert len([p for p in explicit.axes[0].patches if p.get_hatch()]) == 1
    # The implied grid is 2x2 with two filled and two absent cells.
    assert len([p for p in implied.axes[0].patches if p.get_hatch()]) == 2


def test_the_noise_floor_is_named_in_the_colorbar_not_left_to_a_caption():
    """A resolution limit nobody can see is a limit nobody applies."""
    plain = site_substituent_heatmap({("F", "A"): 1.0, ("Cl", "A"): -1.0})
    flagged = site_substituent_heatmap(
        {("F", "A"): 1.0, ("Cl", "A"): -1.0}, noise_floor=4.07
    )
    plain_lbl = " ".join(a.get_ylabel() for a in plain.axes)
    flag_lbl = " ".join(a.get_ylabel() for a in flagged.axes)
    assert "NOISE FLOOR" not in plain_lbl
    assert "NOISE FLOOR" in flag_lbl and "4.07" in flag_lbl


def test_heatmap_refuses_an_entirely_unevaluated_grid():
    with pytest.raises(ValueError, match="unevaluated"):
        site_substituent_heatmap({("F", "A"): None, ("Cl", "A"): None})
    with pytest.raises(ValueError, match="nothing to plot"):
        site_substituent_heatmap({})


# --- pose ensemble -----------------------------------------------------------


def test_pose_ensemble_shows_the_spread_and_labels_the_sd():
    """The whole point: a reader who sees only means cannot see the problem."""
    fig = pose_ensemble(
        {"parent": [-100.0, -140.0, -80.0, -120.0], "analogue": [-105.0, -135.0, -85.0]}
    )
    ax = fig.axes[0]
    pts = [ln for ln in ax.get_lines() if ln.get_marker() == "o"]
    assert sum(len(ln.get_xdata()) for ln in pts) == 7, "every pose must be drawn"
    assert any("sd" in t.get_text() for t in ax.texts), "the sd must be stated"


def test_the_selected_pose_is_marked_but_is_not_the_value():
    """Selection is 10x worse than averaging (MEASURED), so it is an annotation.

    The mean marker must still be present and the selected pose must be drawn
    with a DIFFERENT marker, so a reader cannot mistake one for the other.
    """
    scores = {"parent": [-100.0, -140.0, -80.0]}
    without = pose_ensemble(scores)
    with_sel = pose_ensemble(scores, selected_index=0)
    xs = [ln for ln in with_sel.axes[0].get_lines() if ln.get_marker() == "x"]
    assert len(xs) == 1, "the selected pose must be marked"
    assert not [ln for ln in without.axes[0].get_lines() if ln.get_marker() == "x"]
    labels = [ln.get_label() for ln in with_sel.axes[0].get_lines()]
    assert any("NOT the value" in str(t) for t in labels), (
        "the legend must say the selected pose is not the candidate's value"
    )


def test_pose_ensemble_handles_a_candidate_with_no_usable_poses():
    """One failed candidate must not abort the figure."""
    fig = pose_ensemble({"ok": [-100.0, -120.0], "failed": []})
    ax = fig.axes[0]
    # The marker is an annotation in axes coordinates (so an empty candidate
    # cannot drag the y-limits toward 0), hence texts OR annotations.
    labels = [t.get_text() for t in ax.texts]
    assert "n/a" in labels, f"expected an n/a marker, got {labels}"


def test_pose_ensemble_refuses_an_empty_input():
    with pytest.raises(ValueError, match="nothing to plot"):
        pose_ensemble({})


# --- liability profile -------------------------------------------------------


def test_lower_is_worse_endpoints_are_negated_so_up_is_always_worse():
    """A mixed-polarity panel plotted raw inverts half its endpoints.

    `ToxEndpoint.higher_is_worse` has no default precisely because a wrong
    polarity silently flips a safety ranking. A plot flips it just as silently,
    so this asserts the orientation rather than trusting it.
    """
    fig = liability_profile({"a": {"hERG": (0.8, True), "solubility": (0.2, False)}})
    ys = [float(v) for ln in fig.axes[0].get_lines() for v in ln.get_ydata() if v == v]
    assert 0.8 in ys, "a higher-is-worse endpoint must plot as-is"
    assert -0.2 in ys, "a lower-is-worse endpoint must be negated"
    assert "UP = worse" in fig.axes[0].get_ylabel()


def test_inconsistent_polarity_across_candidates_is_an_error():
    """The same endpoint flipped for one compound and not another is a sign bug."""
    with pytest.raises(ValueError, match="higher_is_worse"):
        liability_profile(
            {
                "a": {"hERG": (0.8, True)},
                "b": {"hERG": (0.5, False)},
            }
        )


def test_an_unavailable_endpoint_is_a_gap_not_a_confident_zero():
    """For a probability endpoint 0.0 means "confidently negative", not "unknown"."""
    fig = liability_profile(
        {"a": {"hERG": (0.8, True), "ames": (None, True), "clint": (0.3, True)}}
    )
    ys = [float(v) for ln in fig.axes[0].get_lines() for v in ln.get_ydata() if v == v]
    assert 0.0 not in ys, "the None endpoint must not be plotted at zero"
    assert sorted(ys) == pytest.approx([0.3, 0.8])
    assert "unevaluated" in fig.axes[0].get_title()


def test_the_parent_is_drawn_as_a_reference_and_must_exist():
    """A liability readout only means something relative to what you improve on."""
    data = {"parent": {"hERG": (0.5, True)}, "analogue": {"hERG": (0.3, True)}}
    fig = liability_profile(data, parent="parent")
    dashed = [ln for ln in fig.axes[0].get_lines() if ln.get_linestyle() == "--"]
    assert len(dashed) == 1, "the parent must be visually distinct"
    assert any("parent" in str(ln.get_label()) for ln in fig.axes[0].get_lines())

    with pytest.raises(ValueError, match="not among the candidates"):
        liability_profile(data, parent="nonexistent")


def test_liability_profile_refuses_empty_inputs():
    with pytest.raises(ValueError, match="nothing to plot"):
        liability_profile({})
    with pytest.raises(ValueError, match="no endpoints"):
        liability_profile({"a": {}})
