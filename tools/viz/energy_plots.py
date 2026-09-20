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
    "optimization_trace",
    "qmmm_partition",
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
            # To the SIDE of the apex, not above: the maximum IS the top of
            # the data, so an upward offset lands on the title. VERIFIED by
            # rendering -- "barrier 13.2" printed through "Energy profile".
            ax.annotate(
                f"barrier {vmax:.1f}",
                xy=(imax, vmax),
                xytext=(12, -4),
                textcoords="offset points",
                ha="left",
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


def _floor_for(noise_floor, sub: str, site: str) -> float | None:
    """The resolution limit that applies to ONE cell.

    A scalar applies everywhere. A mapping applies per (substituent, site), and
    a cell it does not mention has NO floor rather than a zero one -- an
    unmeasured limit must not render as a limit that nothing can fall below.
    """
    if noise_floor is None:
        return None
    if isinstance(noise_floor, dict):
        v = noise_floor.get((sub, site))
        return float(v) if v is not None else None
    return float(noise_floor)


def _floor_label(noise_floor, unit: str) -> str:
    """The colorbar label, which must not imply one floor when there are many."""
    if noise_floor is None:
        return f"ddE ({unit})"
    if isinstance(noise_floor, dict):
        vals = [float(v) for v in noise_floor.values() if v is not None]
        if not vals:
            return f"ddE ({unit})"
        lo, hi = min(vals), max(vals)
        span = f"{lo:g}" if lo == hi else f"{lo:g}-{hi:g} (PER-CANDIDATE)"
        return f"ddE ({unit}) -- |ddE| < {span} is BELOW THE NOISE FLOOR (greyed)"
    return f"ddE ({unit}) -- |ddE| < {noise_floor:g} is BELOW THE NOISE FLOOR (greyed)"


def site_substituent_heatmap(
    ddE: dict[tuple[str, str], float | None],
    *,
    unit: str = "kcal/mol",
    title: str = "ddE vs parent, per (substituent, site)",
    noise_floor: float | dict[tuple[str, str], float] | None = None,
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

    **It also accepts a PER-CELL mapping** `{(substituent, site): floor}`,
    because a floor is not always one campaign-wide number. A PAIRED estimator
    (RESULTS.md M17) cancels pose-conformational noise only to the extent the
    two molecules share it, so its floor tracks rho and varies per candidate:
    MEASURED 0.221 kcal/mol at rho 0.860 (F) against 0.615 at rho 0.399 (Cl),
    a factor of 2.8 across two substituents on ONE scaffold. Collapsing that to
    a scalar either greys cells that are genuinely resolved or passes cells that
    are not, and both errors read as a finished measurement.

    A cell with no entry in the mapping is NOT greyed and is counted in the
    returned figure's caption as unbounded -- an unmeasured floor must not read
    as a floor of zero.

    **THE KEY MUST DISTINGUISH PLACEMENTS, and the obvious choice does not.**
    `SubstitutionProposal.label` is the substituent name, not a unique id:
    aspirin with {F, Cl} gives **9 proposals, 9 distinct SMILES, and 3 distinct
    labels**, because each halogen has four ring positions. Keying on `label`
    alone silently collapses nine candidates into three cells -- in a figure
    whose entire premise is that WHERE a group goes matters as much as which
    group. This function cannot detect that: it receives the collapsed dict and
    a legitimately small one looks identical. Key on `(label, site)` with a real
    site identifier, or on the SMILES.
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
            cell_floor = _floor_for(noise_floor, subs[i], sites[j])
            below = cell_floor is not None and abs(v) < cell_floor
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
    # The floor label is a sentence, and as a rotated colorbar label it was
    # CLIPPED at the canvas edge ("...BELOW THE NOISE FLO") -- losing the word
    # that makes it a warning. Short label on the bar, sentence under the axes.
    cb.set_label(f"ddE ({unit})")
    floor_note = _floor_label(noise_floor, unit)
    if floor_note != f"ddE ({unit})":
        fig.text(
            0.5,
            0.005,
            floor_note.split("-- ", 1)[-1],
            ha="center",
            va="bottom",
            fontsize=8,
            color="#555555",
        )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
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
    # TWO LINES. As one 58-character string this label is taller than the axes
    # and matplotlib does not shrink or wrap it -- MEASURED, it overran the
    # figure top by 16px and rendered as "...endpoints negate", losing the "d"
    # and the closing paren. `tight_layout` does not help: it reserves room for
    # the label's BOX, and the box is already taller than the canvas.
    ax.set_ylabel("liability (UP = worse;\nlower-is-worse endpoints negated)")
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
    # A SADDLE BELOW EITHER ENDPOINT IS NOT A REACTION PATH.
    #
    # The IRC walks DOWNHILL from the saddle, so both endpoints must end up at
    # or below it. A saddle underneath one means the path left the surface it
    # started on -- a diverged SCF, a geometry that fell apart, or the wrong
    # mode followed.
    #
    # Refused rather than drawn, because the plot renders it as a NEGATIVE
    # barrier under a perfectly ordinary title (MEASURED: "reverse barrier
    # -12.55" with no warning), and a negative activation energy read off a
    # figure is worse than no figure.
    for name, ev in (("forward", fwd_e), ("reverse", rev_e)):
        if ev > sad_e:
            raise ValueError(
                f"the {name} endpoint ({ev:.4f} kcal/mol) is ABOVE the saddle "
                f"({sad_e:.4f}). An IRC walks downhill, so this is not a "
                "reaction path -- the branch left the intended surface. Check "
                "the SCF converged along it and that the followed mode was the "
                "reaction coordinate."
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
        # The value sits to the SIDE of the marker. Directly above, it printed
        # on top of the TS point at the apex -- VERIFIED by rendering.
        ax.annotate(
            f"{y:.2f}",
            xy=(x, y),
            xytext=(10, 6),
            textcoords="offset points",
            ha="left",
            fontsize=9,
        )
        if not ok:
            # ABOVE the marker, not below: at -30 points it landed on the
            # x tick labels and the two overprinted. Offset further than the
            # value label so the two do not stack either.
            ax.annotate(
                "NOT converged\n(no basin reached)",
                xy=(x, y),
                xytext=(0, 20),
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
    # RELATIVE: the values are shifted so the lower endpoint is zero, which is
    # what makes both barriers readable off the axis. Labelling it plain
    # "energy" invites reading -55.4 as 0.
    ax.set_ylabel("energy relative to the lower endpoint (kcal/mol)")
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
    ax.margins(x=0.12)
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

    needs_bottom_margin = False
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
        # Wrapped, and the caveat carries a smaller font: at figure width the
        # single line ran off the canvas mid-word ("it does not che..."), which
        # VERIFIED BY RENDERING is invisible to any assertion on the string.
        # BELOW the axes. Two earlier attempts failed by rendering: as one
        # long title line it ran off the canvas mid-word, and moved just above
        # the axes it overprinted the title. There is room under the x labels
        # and nothing competes for it.
        ax.set_title(title, fontsize=11)
        ax.figure.text(
            0.5,
            0.005,
            "no expected_atoms given -- this SHOWS the mode, it does not CHECK it",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#555555",
        )
        # Reserve the room HERE, not in the caller: a figure that needs the
        # caller to call subplots_adjust is a figure that renders wrong by
        # default, and the default is what a test and a notebook both use.
        needs_bottom_margin = True
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=(0, 0.05, 1, 1) if needs_bottom_margin else None)
    return fig


def optimization_trace(
    energies: Sequence[float],
    *,
    gradient_norms: Sequence[float] | None = None,
    converged: bool = True,
    unit: str = "hartree",
    title: str = "Geometry optimization",
):
    """Energy and gradient norm per step, with divergence made unmissable.

    THE PLOT THAT WOULD HAVE SAVED A WRONG DIAGNOSIS. A geometry optimization
    reports `converged`, `steps` and a final energy, and those three numbers
    cannot distinguish "ran out of steps near a minimum" from "walked uphill and
    oscillated". MEASURED case: an embedded methyl whose link atom sat 0.443 A
    from an MM point charge climbed -39.82 -> -39.53 Ha over 200 steps. The
    numbers looked like a slow optimization; the trace shows an unphysical
    attractor at a glance.

    So this annotates the two things a scalar cannot say:

    * **the energy went UP** from its best value, and by how much -- drawn as a
      marked band from the minimum to the final point, because a rise is the
      signature of a step-control or force-field problem rather than a slow
      approach
    * **it did not converge**, stated in the title rather than left to the
      caller, since an unconverged trace and a converged one look identical
      when both flatten

    `gradient_norms` goes on a second log axis when supplied: energy can look
    flat while the gradient is still large, which is exactly the case where the
    geometry is not a stationary point and the frequencies computed on it mean
    nothing.
    """
    if not energies:
        raise ValueError("nothing to plot: no energies")
    # Keep the INDEX with each finite value. Without it the annotation below
    # lands on `steps[-1]`, which may be a step whose energy is None or NaN --
    # the figure then attaches the final finite energy to a step that has none.
    finite_pairs = [
        (i, e) for i, e in enumerate(energies) if e is not None and _isfinite(e)
    ]
    if not finite_pairs:
        raise ValueError("every energy is None or non-finite")
    finite = [e for _, e in finite_pairs]
    if gradient_norms is not None and len(gradient_norms) != len(energies):
        raise ValueError(
            f"gradient_norms has {len(gradient_norms)} entries for "
            f"{len(energies)} energies; they are per-step and must match, or a "
            "step's gradient is drawn against another step's energy"
        )
    if gradient_norms is not None:
        # The gradient axis is logarithmic, and matplotlib does not complain
        # about a value a log axis cannot show -- it keeps it in the data and
        # simply draws nothing there. A converged step reported as exactly 0.0,
        # or a NaN from a failed step, would then VANISH from the curve while
        # the energy trace beside it still shows that step. The reader sees a
        # shorter gradient history than the optimisation actually had and reads
        # the wrong step as the last one. Refuse instead.
        bad = [
            (i, v)
            for i, v in enumerate(gradient_norms)
            if not _isfinite(v) or float(v) <= 0.0
        ]
        if bad:
            shown = ", ".join(f"step {i}: {v!r}" for i, v in bad[:4])
            raise ValueError(
                f"gradient_norms must be finite and strictly positive to be "
                f"drawn on a log axis; got {shown}"
                + (f" (and {len(bad) - 4} more)" if len(bad) > 4 else "")
                + ". A converged step is a small gradient, not zero -- pass the "
                "true norm, or drop the trailing step."
            )

    plt = _plt()
    steps = list(range(len(energies)))
    best = min(finite)
    i_final, final = finite_pairs[-1]
    rose_by = final - best

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(steps, energies, marker="o", markersize=3, linewidth=1.2, color="#1f77b4")
    ax.set_xlabel("step")
    ax.set_ylabel(f"energy ({unit})")

    ax.axhline(best, linestyle=":", linewidth=0.9, color="#888888")
    # Anchored in AXES FRACTION, not at the best point's data coordinates. The
    # best energy is usually the LAST step, so a data-anchored label sits at
    # the right spine and is clipped -- RENDERED and seen, not deduced: the
    # label read "best  -39.72650" with the digits past the axis cut off.
    # Left-aligned just above the line it labels, where there is always room.
    ax.annotate(
        f"best  {best:.6f}",
        xy=(0.015, best),
        xycoords=("axes fraction", "data"),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=8,
        color="#555555",
    )

    # THE UPHILL BAND. Only drawn when the rise is real, so a converged run
    # stays uncluttered and a diverging one cannot be mistaken for it.
    if rose_by > 0:
        ax.axhspan(best, final, color="#d62728", alpha=0.10, zorder=0)
        # Anchored in AXES fraction, not offset from the final data point. With
        # a final point near the top of the plot the text ran through the
        # TITLE and both became unreadable -- VERIFIED by rendering, which is
        # the only way to see it. A test asserting the string is present
        # passes either way.
        ax.annotate(
            f"ENERGY ROSE {rose_by:+.4f} {unit} above its best\n"
            "a minimization that climbs is not converging slowly",
            xy=(0.98, 0.06),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=8,
            color="#d62728",
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.3",
                "fc": "white",
                "ec": "#d62728",
                "alpha": 0.85,
            },
        )

    if gradient_norms is not None:
        ax2 = ax.twinx()
        ax2.semilogy(
            steps, gradient_norms, linestyle="--", linewidth=1.0, color="#ff7f0e"
        )
        ax2.set_ylabel("|grad| (log)", color="#ff7f0e")
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    status = "converged" if converged else "DID NOT CONVERGE"
    # `pad` clears matplotlib's offset-text box (the "-3.9726e1" that appears
    # above the y axis whenever the energies share a large constant part, which
    # for a Hartree total energy is ALWAYS). Without it the exponent label and
    # the title overprint each other -- again seen by rendering, since every
    # string assertion about the title passes either way.
    ax.set_title(f"{title} -- {len(energies)} steps, {status}", pad=14)
    if not converged:
        # An unconverged run's last point is where the budget ran out, not a
        # stationary point. Saying so on the figure keeps it from being read
        # as a result.
        ax.annotate(
            "final point is where the step budget ran out,\nnot a stationary point",
            xy=(0.02, 0.94),
            xycoords="axes fraction",
            va="top",
            fontsize=8,
            color="#d62728",
        )
    fig.tight_layout()
    return fig


def _isfinite(x) -> bool:
    import math

    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def qmmm_partition(
    symbols: Sequence[str],
    coords_angstrom: Sequence[tuple[float, float, float]],
    qm_indices: Sequence[int],
    *,
    link_positions_angstrom: Sequence[tuple[float, float, float]] = (),
    charge_positions_angstrom: Sequence[tuple[float, float, float]] = (),
    charge_values: Sequence[float] = (),
    warn_below_angstrom: float = 1.0,
    expect_min_angstrom: float | None = None,
    title: str = "QM/MM partition",
):
    """Where the QM/MM cut falls, and whether a point charge sits on a link atom.

    THE PICTURE THE FRONTIER BUG NEEDED. `min_link_to_charge_distance()` returns
    a number; nothing showed WHERE the problem was. MEASURED case: keeping the
    host MM charge across a covalent cut puts a bare -0.27 charge **0.443 A**
    from the link hydrogen -- closer than a bond length -- and the optimization
    then diverges rather than failing loudly.

    Drawn as a projection onto the two axes of largest spread, which is a
    2-D view of a 3-D system and therefore understates some distances. That is
    acceptable here because the quantity being judged -- the SHORTEST
    link-to-charge distance -- is computed in 3-D and annotated, not measured
    off the picture.

    **EVERY argument is ANGSTROM**, and that matters more than usual here
    because ferric's own QM/MM accessors are NOT uniform:

        QmmmSystem.link_atom_positions()  ->  ANGSTROM   (e.g. 1.0971)
        QmmmSystem.point_charges()        ->  BOHR       (e.g. 2.91)

    Scaling both by 0.529 gives a nearest link-charge distance of 0.959 A where
    the true value is 0.443 -- a factor of 2.2, and in the SAFE direction, so a
    0.443 A frontier problem renders as an unremarkable 0.959. VERIFIED against
    `min_link_to_charge_distance()`, which is the authority: convert the
    charges and leave the link positions alone.

    Pass `expect_min_angstrom=` (typically that accessor's value) and the
    function will REFUSE a set of inputs whose geometry disagrees with it,
    rather than drawing a reassuring picture of mis-scaled coordinates.

    **This does not say the partition is chemically sensible.** It shows where
    the cut is, not whether the QM region contains the bonds that break. A
    figure cannot tell you that, and the C0 rule in the golden path is the
    thing that can.
    """
    if len(coords_angstrom) != len(symbols):
        raise ValueError(
            f"{len(coords_angstrom)} coordinates for {len(symbols)} symbols; "
            "they are per-atom and must match, or atoms are drawn at other "
            "atoms' positions"
        )
    if charge_values and len(charge_values) != len(charge_positions_angstrom):
        raise ValueError(
            f"{len(charge_values)} charge values for "
            f"{len(charge_positions_angstrom)} charge positions"
        )
    if not symbols:
        raise ValueError("nothing to plot: no atoms")

    plt = _plt()
    import numpy as np

    xyz = np.asarray([[float(c) for c in p] for p in coords_angstrom], dtype=float)
    qm = set(int(i) for i in qm_indices)
    bad = [i for i in qm if not 0 <= i < len(symbols)]
    if bad:
        raise ValueError(f"qm_indices out of range for {len(symbols)} atoms: {bad}")

    # Project onto the two axes of largest spread so the cut is visible rather
    # than edge-on. Chosen from the QM+MM atoms only -- charges can be far away
    # and would otherwise dominate the choice.
    spread = xyz.max(axis=0) - xyz.min(axis=0)
    a, b = sorted(range(3), key=lambda k: -spread[k])[:2]
    axis_name = "xyz"

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    mm = [i for i in range(len(symbols)) if i not in qm]
    if mm:
        ax.scatter(
            xyz[mm, a],
            xyz[mm, b],
            s=60,
            c="#bbbbbb",
            edgecolors="#888888",
            label=f"MM ({len(mm)})",
            zorder=2,
        )
    if qm:
        qi = sorted(qm)
        ax.scatter(
            xyz[qi, a],
            xyz[qi, b],
            s=110,
            c="#1f77b4",
            edgecolors="#10496f",
            label=f"QM ({len(qi)})",
            zorder=3,
        )
    for i, s in enumerate(symbols):
        ax.annotate(
            s,
            (xyz[i, a], xyz[i, b]),
            fontsize=7,
            ha="center",
            va="center",
            color="white" if i in qm else "#333333",
            zorder=4,
        )

    link = np.asarray(
        [[float(c) for c in p] for p in link_positions_angstrom], dtype=float
    ).reshape(-1, 3)
    if len(link):
        ax.scatter(
            link[:, a],
            link[:, b],
            s=150,
            marker="*",
            c="#2ca02c",
            edgecolors="#14521a",
            label=f"link atom ({len(link)})",
            zorder=5,
        )

    chg = np.asarray(
        [[float(c) for c in p] for p in charge_positions_angstrom], dtype=float
    ).reshape(-1, 3)
    closest = None
    if len(chg):
        vals = list(charge_values) or [0.0] * len(chg)
        neg = [k for k, v in enumerate(vals) if v < 0]
        pos = [k for k, v in enumerate(vals) if v >= 0]
        for idx, colour, lbl in (
            (neg, "#d62728", "MM charge -"),
            (pos, "#9467bd", "MM charge +"),
        ):
            if idx:
                ax.scatter(
                    chg[idx, a],
                    chg[idx, b],
                    s=45,
                    marker="x",
                    c=colour,
                    label=f"{lbl} ({len(idx)})",
                    zorder=3,
                )
        # THE MEASUREMENT, in 3-D, not off the projection.
        if len(link):
            d = np.linalg.norm(link[:, None, :] - chg[None, :, :], axis=2)
            li, ci = np.unravel_index(int(np.argmin(d)), d.shape)
            closest = float(d[li, ci])
            ax.plot(
                [link[li, a], chg[ci, a]],
                [link[li, b], chg[ci, b]],
                linestyle="--",
                linewidth=1.4,
                color="#d62728" if closest < warn_below_angstrom else "#777777",
                zorder=1,
            )
            mid = ((link[li, a] + chg[ci, a]) / 2, (link[li, b] + chg[ci, b]) / 2)
            # The short distance label stays on the line; the WARNING moves to
            # a corner. Rendering showed the multi-line warning printed over
            # the link atom and the charge it points at. An assertion that the
            # text exists cannot see that.
            ax.annotate(
                f"{closest:.3f} A",
                mid,
                fontsize=8,
                ha="center",
                va="bottom",
                color="#d62728" if closest < warn_below_angstrom else "#555555",
                weight="bold" if closest < warn_below_angstrom else "normal",
                zorder=6,
            )
            if closest < warn_below_angstrom:
                ax.annotate(
                    "INSIDE A BOND LENGTH -- the charge is an attractor\n"
                    "and the optimization will not settle",
                    xy=(0.5, 0.02),
                    xycoords="axes fraction",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#d62728",
                    weight="bold",
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "fc": "white",
                        "ec": "#d62728",
                        "alpha": 0.9,
                    },
                    zorder=7,
                )

    # CROSS-CHECK against the caller's authoritative number. The unit mismatch
    # described above is silent and lands in the SAFE direction, so a figure
    # drawn from mis-scaled coordinates looks fine. Refusing beats reassuring.
    if expect_min_angstrom is not None:
        if closest is None:
            raise ValueError(
                f"expect_min_angstrom={expect_min_angstrom} was given but no "
                "link/charge pair was supplied, so there is nothing to check "
                "it against"
            )
        if abs(closest - float(expect_min_angstrom)) > 0.01:
            raise ValueError(
                f"the supplied geometry gives a nearest link-charge distance "
                f"of {closest:.3f} A, but expect_min_angstrom is "
                f"{float(expect_min_angstrom):.3f} A. Check the UNITS: "
                "`link_atom_positions()` is Angstrom and `point_charges()` is "
                "Bohr, so converting both is a factor ~1.89 error that renders "
                "as a comfortably large distance."
            )

    ax.set_xlabel(f"{axis_name[a]} (Angstrom)")
    ax.set_ylabel(f"{axis_name[b]} (Angstrom)")
    suffix = "" if closest is None else f" -- nearest link-charge {closest:.3f} A"
    ax.set_title(title + suffix)
    ax.legend(fontsize=8, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return fig
