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
    "reaction_path",
    "tier_comparison",
    "site_substituent_heatmap",
    "pose_ensemble",
    "liability_profile",
    "imaginary_mode",
    "close",
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


def close(fig) -> None:
    """Release a figure returned by this module.

    Every function here returns a `Figure` and does NOT close it -- the caller
    needs it to save or display. But pyplot RETAINS every figure it creates, so
    a batch that plots a thousand analogues holds a thousand figures and
    matplotlib warns at 20 ("More than 20 figures have been opened").

    Call this when done with one. It is a thin wrapper over `plt.close(fig)`,
    provided so a caller does not have to import pyplot (and risk picking a GUI
    backend) just to free a figure this module handed them.
    """
    _plt().close(fig)


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
    # `_to_kcal` already converted, so the axis is kcal/mol WHATEVER `unit`
    # said on the way in. Labelling it `unit` put hartree over kcal/mol
    # numbers -- a 627x mislabel that looks entirely plausible on a bar chart.
    ax.set_ylabel(ylabel or "energy (kcal/mol)")
    ax.set_title(
        title if not n_missing else f"{title}  ({n_missing} unevaluated, shown as gaps)"
    )
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def site_substituent_heatmap(
    ddE: dict[tuple[str, str], float | None],
    *,
    unit: str = "kcal/mol",
    title: str = "ddE vs parent, per (substituent, site)",
    noise_floor: float | None = None,
):
    """ddE for every (substituent, SITE) pair, as a grid.

    This is the substitution pipeline's actual output UNIT, and the reason it
    is a grid rather than a bar chart per substituent: MEASURED within/between
    ratio 0.94-0.95 on two independent constructions, i.e. WHERE a group goes
    matters as much as WHICH group. Collapsing the site axis averages over the
    larger of the two effects.

    `ddE` is keyed `(substituent, site)`. A missing key, or a `None` value, is
    drawn as an explicit hatched cell, NOT as zero and NOT as the colormap's
    midpoint -- an unevaluated pair and a pair that came out neutral demand
    opposite responses.

    `noise_floor` draws the resolution limit into the COLORBAR LABEL and greys
    every cell inside it. Pass the measured ddE noise for the protocol that
    produced these numbers. On this campaign that is ~4.07 kcal/mol (averaging
    n=100) against substituent effects of 1-2, so essentially every cell should
    grey out -- which is the honest picture, and exactly why the parameter
    exists rather than being left to a caption nobody reads.
    """
    if not ddE:
        raise ValueError("nothing to plot")
    plt = _plt()
    import numpy as np

    subs = sorted({k[0] for k in ddE})
    sites = sorted({k[1] for k in ddE})
    grid = np.full((len(subs), len(sites)), np.nan)
    for i, sub in enumerate(subs):
        for j, site in enumerate(sites):
            v = ddE.get((sub, site))
            if v is not None:
                grid[i, j] = v

    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        raise ValueError("every (substituent, site) pair is unevaluated")
    # Symmetric limits so the diverging colormap's midpoint is a real zero
    # rather than wherever the data happens to centre.
    lim = float(np.abs(finite).max()) or 1.0

    fig, ax = plt.subplots(
        figsize=(max(5.0, 1.0 * len(sites) + 3), max(3.0, 0.55 * len(subs) + 2))
    )
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")

    for i in range(len(subs)):
        for j in range(len(sites)):
            if not np.isfinite(grid[i, j]):
                # Unevaluated: hatched, captioned, unmistakably not a number.
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor="#f0f0f0",
                        edgecolor="#999999",
                        hatch="///",
                        linewidth=0.5,
                    )
                )
                ax.text(
                    j, i, "n/a", ha="center", va="center", fontsize=8, color="#666666"
                )
                continue
            v = grid[i, j]
            below = noise_floor is not None and abs(v) < noise_floor
            ax.text(
                j,
                i,
                f"{v:+.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#999999"
                if below
                else ("white" if abs(v) > 0.6 * lim else "black"),
                style="italic" if below else "normal",
            )

    ax.set_xticks(range(len(sites)))
    ax.set_xticklabels(sites, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(subs)))
    ax.set_yticklabels(subs, fontsize=9)
    ax.set_xlabel("site")
    ax.set_ylabel("substituent")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(
        f"ddE ({unit})"
        if noise_floor is None
        else f"ddE ({unit}) -- |ddE| < {noise_floor:g} is BELOW THE NOISE FLOOR (greyed)"
    )
    fig.tight_layout()
    return fig


def pose_ensemble(
    scores_by_candidate: dict[str, Sequence[float]],
    *,
    unit: str = "kcal/mol",
    title: str = "Per-pose scores by candidate",
    selected_index: int | None = None,
):
    """Every pose's score, per candidate, as a strip plot with the mean marked.

    The plot that makes the pose problem visible instead of a footnote. On this
    campaign the per-pose sd is ~29 kcal/mol against substituent effects of
    1-2, so the strips overlap almost completely -- and a reader who has only
    seen the MEANS has no way to know that. Showing the spread is the point.

    `selected_index` marks one pose per candidate (e.g. the top-docked one).
    MEASURED, selecting on an axis uncorrelated with the scorer is a single
    random draw and is sqrt(n) WORSE than averaging, so this is drawn as an
    annotation to be inspected, never as the candidate's value.
    """
    if not scores_by_candidate:
        raise ValueError("nothing to plot")
    plt = _plt()
    import statistics

    names = list(scores_by_candidate)
    fig, ax = plt.subplots(figsize=(max(6.0, 1.3 * len(names) + 2), 4.5))
    rng_state = 12345  # fixed jitter: a replot must not move the points

    # Label the FIRST artist of each kind that is actually drawn, NOT the one
    # at index 0. Candidate 0 can have no usable poses, in which case the loop
    # `continue`s before labelling anything and the legend comes out empty --
    # taking the "selected pose (NOT the value)" disclaimer with it, which is
    # the one piece of text on this figure that has to be there.
    labelled: set[str] = set()

    def _once(key: str, text: str) -> str | None:
        if key in labelled:
            return None
        labelled.add(key)
        return text

    for i, name in enumerate(names):
        vals = [v for v in scores_by_candidate[name] if v is not None]
        if not vals:
            # Place the marker in AXES coordinates on the x-position only, so a
            # candidate with no poses does not drag the y-limits toward 0 (and
            # does not trip tight_layout when every other point is at -100).
            ax.annotate(
                "n/a",
                xy=(i, 0.5),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="center",
                fontsize=9,
                color="#666666",
            )
            continue
        # Deterministic jitter so the figure is reproducible.
        jit = [
            ((rng_state * (k + 1) * (i + 7)) % 1000 / 1000.0 - 0.5) * 0.28
            for k in range(len(vals))
        ]
        ax.plot(
            [i + j for j in jit],
            vals,
            "o",
            ms=5,
            alpha=0.55,
            color="#4c72b0",
            label=_once("poses", "poses"),
        )
        m = statistics.fmean(vals)
        ax.plot(
            [i - 0.3, i + 0.3],
            [m, m],
            "-",
            lw=2.5,
            color="#c44e52",
            label=_once("mean", "mean"),
        )
        if len(vals) > 1:
            sd = statistics.stdev(vals)
            ax.annotate(
                f"sd {sd:.1f}",
                xy=(i, m),
                xytext=(0, -16),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#c44e52",
            )
        if selected_index is not None and 0 <= selected_index < len(vals):
            ax.plot(
                i,
                vals[selected_index],
                "x",
                ms=11,
                mew=2.0,
                color="#111111",
                label=_once("selected", "selected pose (NOT the value)"),
            )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(f"score ({unit})")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def liability_profile(
    endpoints_by_candidate: dict[str, dict[str, tuple[float | None, bool]]],
    *,
    title: str = "Liability profile vs parent",
    parent: str | None = None,
):
    """Toxicity/liability endpoints per candidate, oriented so WORSE is up.

    `endpoints_by_candidate[candidate][endpoint] = (value, higher_is_worse)`.
    The polarity flag is REQUIRED per endpoint, mirroring `tools.tox.model`'s
    `ToxEndpoint.higher_is_worse`, which deliberately has no default: a wrong
    polarity silently inverts a safety ranking, and a plot inverts it just as
    silently as a table. Endpoints with `higher_is_worse=False` are negated
    before plotting, and the axis says so.

    `value is None` means the source could not produce that endpoint. It is
    drawn as a GAP, never as 0.0 -- for a probability-valued endpoint 0.0 means
    "confidently predicted negative", the opposite of "unknown". That is
    `ToxEndpoint`'s own rule and this honours it.

    `parent=` plots that candidate as a dashed reference line, because a
    liability readout is only interpretable RELATIVE to the compound you are
    trying to improve on -- the same argument the pipeline makes for ddE.

    This does NOT aggregate endpoints into a score. An alert set is a
    literature flag, not a probability of harm (see `tools/tox/alerts.py`), and
    summing flags of different provenance into one number is exactly the
    laundering that docstring warns against.
    """
    if not endpoints_by_candidate:
        raise ValueError("nothing to plot")
    names = list(endpoints_by_candidate)
    if parent is not None and parent not in names:
        raise ValueError(
            f"parent {parent!r} is not among the candidates {names} -- a "
            "reference line must refer to something that was measured"
        )
    endpoints = sorted({e for d in endpoints_by_candidate.values() for e in d})
    if not endpoints:
        raise ValueError("no endpoints in any candidate")

    # Polarity must be CONSISTENT across candidates, or the same endpoint gets
    # flipped for one compound and not another -- a silent sign error.
    polarity: dict[str, bool] = {}
    for cand, d in endpoints_by_candidate.items():
        for ep, (_, worse) in d.items():
            if ep in polarity and polarity[ep] != worse:
                raise ValueError(
                    f"endpoint {ep!r} is declared higher_is_worse={polarity[ep]} for "
                    f"one candidate and {worse} for {cand!r}. One of them is wrong, "
                    "and plotting it would invert that endpoint for half the set."
                )
            polarity[ep] = worse

    plt = _plt()
    fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(endpoints) + 3), 4.4))
    xs = list(range(len(endpoints)))
    n_gap = 0
    for cand in names:
        d = endpoints_by_candidate[cand]
        ys: list[float | None] = []
        for ep in endpoints:
            v = d.get(ep, (None, polarity[ep]))[0]
            if v is None:
                ys.append(None)
                n_gap += 1
            else:
                # Orient so UP is always worse.
                ys.append(v if polarity[ep] else -v)
        real = [(i, y) for i, y in enumerate(ys) if y is not None]
        if not real:
            continue
        style = (
            {"ls": "--", "lw": 2.0, "color": "#333333"}
            if cand == parent
            else {"lw": 1.4}
        )
        ax.plot(
            [i for i, _ in real],
            [y for _, y in real],
            marker="o",
            ms=5,
            label=f"{cand} (parent)" if cand == parent else cand,
            **style,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(endpoints, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("liability (UP = worse; lower-is-worse endpoints negated)")
    ax.set_title(
        title if not n_gap else f"{title}  ({n_gap} unevaluated, shown as gaps)"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def reaction_path(
    *,
    saddle_energy: float,
    forward_energy: float,
    reverse_energy: float,
    forward_converged: bool,
    reverse_converged: bool,
    unit: str = "hartree",
    labels: tuple[str, str, str] = ("reverse endpoint", "TS", "forward endpoint"),
    title: str = "Reaction path (IRC)",
):
    """An IRC result: the two endpoints a saddle connects, and both barriers.

    `energy_profile` plots an ordered sequence and is the wrong shape for this.
    An IRC is not a path from A to B -- it is a saddle with TWO downhill
    branches, and the quantity a reader wants is the barrier in EACH direction.
    Feeding it as three ordered points hides that the middle one is the origin
    of both walks, not a waypoint between them.

    ## Why the convergence flags are required arguments

    `IrcBranch.converged` is `False` when the walk exhausted its step budget
    instead of reaching a basin. That branch identifies NO minimum: its
    endpoint is wherever the walk stopped, and the barrier computed against it
    is a lower bound at best.

    Drawing that identically to a converged branch is the failure this plot has
    to avoid, because the figure is what gets pasted into a slide. An
    unconverged endpoint is drawn with an OPEN marker, a dashed connector and
    an explicit annotation, and the title says so.

    ## What it does NOT claim

    That either endpoint is a stationary point. The IRC stops on a gradient
    threshold; confirming a minimum needs a Hessian there. A converged branch
    means "this walk reached a flat region", not "this is a minimum".
    """
    plt = _plt()
    e = _to_kcal([reverse_energy, saddle_energy, forward_energy], unit)
    rev_e, sad_e, fwd_e = e[0], e[1], e[2]
    # FINITE, not merely non-None. A NaN is the realistic case -- it is what an
    # unconverged SCF hands back -- and it would plot as a silently missing
    # point with the axis auto-scaled around the other two, which looks like a
    # deliberate omission rather than a failure.
    vals = {"reverse": rev_e, "saddle": sad_e, "forward": fwd_e}
    bad = [
        k
        for k, v in vals.items()
        if v is None or v != v or v in (float("inf"), float("-inf"))
    ]
    if bad:
        raise ValueError(
            f"reaction_path needs three real energies; {', '.join(bad)} "
            f"is not finite ({vals}). A non-finite energy is a failed "
            "calculation, not a point on a path."
        )
    # Relative to the LOWER endpoint, which is the conventional zero and makes
    # both barriers read directly off the y axis.
    zero = min(rev_e, fwd_e)
    ys = [rev_e - zero, sad_e - zero, fwd_e - zero]
    xs = [0.0, 1.0, 2.0]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    # Connectors: dashed to an endpoint whose walk did not converge.
    for (x0, x1), ok in (
        ((0.0, 1.0), reverse_converged),
        ((1.0, 2.0), forward_converged),
    ):
        i0, i1 = (0, 1) if x0 == 0.0 else (1, 2)
        ax.plot(
            [x0, x1],
            [ys[i0], ys[i1]],
            "-" if ok else "--",
            color="#4c72b0" if ok else "#999999",
            lw=2.0,
        )
    for k, (x, y) in enumerate(zip(xs, ys)):
        ok = True if k == 1 else (reverse_converged if k == 0 else forward_converged)
        ax.plot(
            x,
            y,
            "o" if ok else "o",
            ms=11,
            mfc="#4c72b0" if ok else "none",
            mec="#4c72b0" if ok else "#c44e52",
            mew=2.0,
        )
        ax.annotate(
            f"{y:.2f}",
            xy=(x, y),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
        if not ok:
            ax.annotate(
                "NOT converged\n(no basin reached)",
                xy=(x, y),
                xytext=(0, -30),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#c44e52",
            )

    # Both barriers, stated rather than left to be measured off the axis.
    ax.annotate(
        f"reverse barrier {ys[1] - ys[0]:.2f}",
        xy=(0.5, (ys[0] + ys[1]) / 2),
        ha="center",
        fontsize=9,
        color="#555555",
    )
    ax.annotate(
        f"forward barrier {ys[1] - ys[2]:.2f}",
        xy=(1.5, (ys[1] + ys[2]) / 2),
        ha="center",
        fontsize=9,
        color="#555555",
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(list(labels), fontsize=9)
    ax.set_ylabel("energy (kcal/mol)")
    unconverged = [
        n
        for n, ok in (("reverse", reverse_converged), ("forward", forward_converged))
        if not ok
    ]
    ax.set_title(
        title
        if not unconverged
        else f"{title}  ({' and '.join(unconverged)} did NOT converge)"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def imaginary_mode(
    symbols: Sequence[str],
    displacements: Sequence[float],
    *,
    top_n: int = 10,
    title: str = "Imaginary mode: which atoms move",
    expected_atoms: Sequence[int] | None = None,
):
    """Per-atom displacement magnitude of a transition state's imaginary mode.

    The plot golden-path step C4 needs and nothing else provides. Counting
    imaginary frequencies is NECESSARY AND NOT SUFFICIENT: exactly one means
    first-order saddle, not "the saddle for the reaction you meant" -- a methyl
    rotor gives one too. Completing C4 means checking the mode displaces the
    atoms whose bonds are breaking or forming, and that is a question about a
    3N vector that a frequency number cannot answer.

    `displacements` is the flat 3N Cartesian vector from
    `SaddleResult.imaginary_mode` (or a row of
    `FrequencyResult.normal_modes`). Bars are per-atom magnitudes
    `sqrt(dx^2+dy^2+dz^2)`, sorted largest first, because "which atoms move"
    is the question and atom index order buries the answer.

    `expected_atoms` are the indices you EXPECT to dominate -- the reacting
    centres. They are highlighted, and the caption states how many of them
    landed in the top `top_n`. That turns a visual impression into a check:
    a mode whose largest motions are nowhere near the reacting bonds is the
    methyl-rotor case, and it should be obvious rather than inferred.

    This does NOT decide whether the mode is right. It shows what the mode
    does; the chemistry is the caller's.
    """
    n3 = len(displacements)
    if n3 == 0 or n3 % 3 != 0:
        raise ValueError(
            f"displacements has {n3} entries, which is not a 3N Cartesian "
            "vector -- pass SaddleResult.imaginary_mode or one row of "
            "FrequencyResult.normal_modes"
        )
    n_atoms = n3 // 3
    if len(symbols) != n_atoms:
        raise ValueError(
            f"{len(symbols)} symbols but the mode covers {n_atoms} atoms -- "
            "these must describe the same molecule"
        )
    if expected_atoms is not None:
        bad = [i for i in expected_atoms if not 0 <= i < n_atoms]
        if bad:
            raise ValueError(
                f"expected_atoms {bad} are out of range for {n_atoms} atoms"
            )

    mags = [
        (
            displacements[3 * i] ** 2
            + displacements[3 * i + 1] ** 2
            + displacements[3 * i + 2] ** 2
        )
        ** 0.5
        for i in range(n_atoms)
    ]
    order = sorted(range(n_atoms), key=lambda i: mags[i], reverse=True)
    shown = order[: min(top_n, n_atoms)]

    plt = _plt()
    fig, ax = plt.subplots(figsize=(max(5.5, 0.6 * len(shown) + 2), 4.0))
    expect = set(expected_atoms or ())
    colors = ["#c44e52" if i in expect else "#4c72b0" for i in shown]
    ax.bar(range(len(shown)), [mags[i] for i in shown], color=colors)
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels(
        [f"{symbols[i]}{i}" for i in shown], rotation=45, ha="right", fontsize=9
    )
    ax.set_ylabel("|displacement| (arbitrary units)")

    if expected_atoms is not None:
        hit = len(expect & set(shown))
        ax.set_title(
            f"{title}\n{hit}/{len(expect)} expected reacting atoms in the top "
            f"{len(shown)}" + ("" if hit == len(expect) else "  -- CHECK THIS MODE")
        )
        ax.plot([], [], "s", color="#c44e52", label="expected reacting atom")
        ax.legend(frameon=False, fontsize=9)
    else:
        ax.set_title(
            title + "\n(no expected_atoms given -- this shows the mode, it does "
            "not check it)"
        )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
