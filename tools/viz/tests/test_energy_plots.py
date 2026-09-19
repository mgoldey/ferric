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
