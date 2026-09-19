"""Plots for the quantities this repo actually produces: funnels and paths.

Every function here takes already-computed numbers and returns a matplotlib
`Figure`. Nothing computes chemistry, and nothing re-derives a value it was
handed -- the same rule `tools/campaign/hierarchy.py` applies to tiers, for the
same reason: a plotting layer that quietly recomputes something is a second,
unvalidated implementation of it.

## What these plots refuse to do

A plot is read faster than a table and trusted more, so the failure mode worth
guarding is a figure that looks authoritative about something it does not know:

- **A missing value is drawn as a gap, never as zero.** `None` means a tier did
  not run or could not answer. Plotting that as 0.0 kcal/mol puts a candidate
  at the bottom of a ranking for not having been measured. Gaps are annotated
  in the legend with their count, so "nothing there" is visible rather than
  silently absent.
- **Error bars are SEM, and are drawn only when n > 1.** A single sample has no
  measured spread; drawing a zero-length bar on it implies a precision that was
  never established. See the repo's `precision-is-sem-not-range` convention.
- **Units are always in the axis label.** Hartree and kcal/mol differ by 627.5,
  and a mislabelled axis is a wrong answer that looks right.

## Why matplotlib and not a web viewer

`matplotlib` is already a declared dev dependency and renders headless, which
is what a CI job or a batch script needs. Interactive 3-D viewers (py3Dmol,
nglview) are not installed here, and `molecules.py` deliberately renders 2-D
depictions through RDKit rather than adding one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "SeriesPoint",
    "funnel_survival",
    "energy_profile",
    "tier_comparison",
]

#: Hartree -> kcal/mol. The one conversion this module performs, named so a
#: reader can check it rather than recognise 627.5.
HARTREE_TO_KCAL = 627.5094740631


# Import matplotlib lazily and non-interactively. Importing pyplot at module
# scope picks a GUI backend on an interactive machine and then FAILS in a
# headless CI job -- and it would do so at import time, breaking callers that
# only wanted a dataclass from this module.
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


@dataclass(frozen=True)
class SeriesPoint:
    """One measured point on an energy series.

    `value is None` means UNEVALUATED -- the tier did not run, or ran and could
    not answer. It is drawn as a gap. `sem` is the standard error of the mean
    and is `None` when only one sample exists, because a single sample has no
    measured spread.
    """

    label: str
    value: float | None
    sem: float | None = None
    n: int = 1


def _to_kcal(values: Sequence[float | None], unit: str) -> list[float | None]:
    """Convert to kcal/mol, or pass through if already there.

    Raises on an unknown unit rather than assuming one: silently treating
    Hartree as kcal/mol is a 627x error that produces a plausible-looking plot.
    """
    u = unit.lower()
    if u in ("kcal", "kcal/mol", "kcal_per_mol"):
        return list(values)
    if u in ("hartree", "ha", "au", "a.u."):
        return [None if v is None else v * HARTREE_TO_KCAL for v in values]
    raise ValueError(
        f"unknown unit {unit!r}; expected 'hartree' or 'kcal/mol'. There is no "
        "default: guessing wrong is a 627x error that still plots."
    )


def energy_profile(
    points: Sequence[SeriesPoint],
    *,
    unit: str = "hartree",
    relative_to: int | None = 0,
    title: str = "Energy profile",
    ylabel: str | None = None,
):
    """A reaction/conformer path: energy vs. an ordered sequence of states.

    This is the plot a transition-state search or a scan produces. `relative_to`
    is the index whose energy becomes zero (the usual convention: reactant = 0),
    or `None` to plot absolute energies. Subtracting a reference is done AFTER
    unit conversion so the offset is exact in the plotted unit.

    Points whose `value is None` break the line rather than interpolating
    across it: a path with an unconverged point is not a path, and drawing a
    smooth curve through the hole would claim otherwise.
    """
    plt = _plt()
    vals = _to_kcal([p.value for p in points], unit)
    sems = _to_kcal([p.sem for p in points], unit)

    if relative_to is not None:
        if not 0 <= relative_to < len(points):
            raise IndexError(
                f"relative_to={relative_to} is out of range for {len(points)} points"
            )
        ref = vals[relative_to]
        if ref is None:
            raise ValueError(
                f"relative_to={relative_to} ({points[relative_to].label!r}) has no "
                "value, so it cannot be the zero of the plot. Pick a point that "
                "was actually evaluated, or pass relative_to=None."
            )
        vals = [None if v is None else v - ref for v in vals]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    xs = list(range(len(points)))

    # Break the line at gaps: plot each contiguous run of real values.
    run_x: list[int] = []
    run_y: list[float] = []
    for i, v in enumerate(vals):
        if v is None:
            if len(run_x) > 1:
                ax.plot(run_x, run_y, "-", color="#1f77b4", lw=1.8, zorder=2)
            run_x, run_y = [], []
        else:
            run_x.append(i)
            run_y.append(v)
    if len(run_x) > 1:
        ax.plot(run_x, run_y, "-", color="#1f77b4", lw=1.8, zorder=2)

    real = [(i, v) for i, v in enumerate(vals) if v is not None]
    if real:
        ax.plot(
            [i for i, _ in real],
            [v for _, v in real],
            "o",
            color="#1f77b4",
            ms=7,
            zorder=3,
        )
    # SEM bars only where n > 1 -- see the module docstring.
    for i, (p, v, s) in enumerate(zip(points, vals, sems)):
        if v is not None and s is not None and p.n > 1:
            ax.errorbar(i, v, yerr=s, fmt="none", ecolor="#444444", capsize=3, zorder=4)

    n_gap = sum(1 for v in vals if v is None)
    if n_gap:
        # Mark the gaps on the axis so "not evaluated" is visible, not absent.
        for i, v in enumerate(vals):
            if v is None:
                ax.axvline(i, color="#cccccc", ls=":", lw=1.2, zorder=1)
        ax.plot([], [], ls=":", color="#cccccc", label=f"unevaluated ({n_gap})")
        ax.legend(frameon=False, fontsize=9)

    # Annotate the barrier when there is an interior maximum -- the number a
    # reader of a reaction path is actually after.
    if len(real) >= 3:
        imax, vmax = max(real, key=lambda t: t[1])
        if real[0][0] < imax < real[-1][0]:
            ax.annotate(
                f"barrier {vmax:.1f}",
                xy=(imax, vmax),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color="#a03030",
            )

    ax.set_xticks(xs)
    ax.set_xticklabels([p.label for p in points], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(
        ylabel
        or (
            "relative energy (kcal/mol)"
            if relative_to is not None
            else "energy (kcal/mol)"
        )
    )
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def funnel_survival(
    stage_names: Sequence[str],
    counts: Sequence[int],
    *,
    title: str = "Candidate funnel",
):
    """How many candidates survive each tier of the cost hierarchy.

    The bar labels carry both the absolute count and the fraction of the
    ORIGINAL input, because a funnel read as per-stage retention and a funnel
    read as cumulative survival tell different stories and look identical.
    """
    if len(stage_names) != len(counts):
        raise ValueError(
            f"{len(stage_names)} stage names but {len(counts)} counts -- these "
            "must correspond one-to-one"
        )
    if not counts:
        raise ValueError("a funnel needs at least one stage")
    if any(c < 0 for c in counts):
        raise ValueError(f"negative candidate count in {list(counts)}")
    for a, b in zip(counts, counts[1:]):
        if b > a:
            raise ValueError(
                f"funnel counts must be non-increasing, got {list(counts)}. A "
                "stage cannot emit more candidates than it received; if this is "
                "an enumeration step, it is not a funnel stage."
            )

    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    total = counts[0]
    ax.barh(range(len(counts)), counts, color="#4c72b0", height=0.6)
    for i, c in enumerate(counts):
        frac = (100.0 * c / total) if total else 0.0
        ax.text(c, i, f"  {c}  ({frac:.0f}% of input)", va="center", fontsize=9)
    ax.set_yticks(range(len(stage_names)))
    ax.set_yticklabels(stage_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("candidates surviving")
    ax.set_xlim(0, max(counts) * 1.35 if max(counts) else 1)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def tier_comparison(
    labels: Sequence[str],
    series: dict[str, Sequence[float | None]],
    *,
    unit: str = "kcal/mol",
    title: str = "Tier comparison",
    ylabel: str | None = None,
):
    """The same candidates scored by several tiers, grouped side by side.

    This is the plot that answers "does the cheap tier rank like the expensive
    one?". Missing values are gaps, not zeros -- a tier that did not score a
    candidate must not push it to the bottom of the chart.
    """
    if not series:
        raise ValueError("tier_comparison needs at least one series")
    for name, vals in series.items():
        if len(vals) != len(labels):
            raise ValueError(
                f"series {name!r} has {len(vals)} values but there are "
                f"{len(labels)} labels"
            )

    plt = _plt()
    fig, ax = plt.subplots(figsize=(max(7.0, 1.1 * len(labels)), 4.2))
    n_series = len(series)
    width = 0.8 / n_series
    n_missing = 0
    for k, (name, raw) in enumerate(series.items()):
        vals = _to_kcal(raw, unit)
        xs = [i + (k - (n_series - 1) / 2) * width for i in range(len(labels))]
        # Only bar the points that exist; a gap stays empty.
        bx = [x for x, v in zip(xs, vals) if v is not None]
        by = [v for v in vals if v is not None]
        n_missing += sum(1 for v in vals if v is None)
        ax.bar(bx, by, width=width * 0.92, label=name)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel or f"energy ({unit})")
    ax.set_title(
        title if not n_missing else f"{title}  ({n_missing} unevaluated, shown as gaps)"
    )
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
