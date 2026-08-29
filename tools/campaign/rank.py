"""Arm D: rank candidates over (liability, fit loss, strain) — and gate the metric.

## The gate comes first

`fit_discriminates_controls` is not a diagnostic, it is a **precondition**. The
negative controls in `tools.morph.design` delete a pharmacophore contact while
barely changing molecular size (NC1: methyl ester, +14 Da; NC2: decyano, -43 Da).
The stated artifact hypothesis is:

- *if the fit metric works:* both controls score clearly worse than the parent;
- *if it is measuring size:* NC2 (smaller) scores BETTER than the parent, and the
  ranking of every real candidate is meaningless.

Those predictions differ, so the experiment is admissible — and if the gate
fails, the correct action is to report the metric as unusable, not to publish a
candidate ranking from it. Per CLAUDE.md: a negative result needs the same bar
as a positive one.

**OUTCOME (2026-08-29): the gate failed, and NEITHER branch above was the
reason.** The size hypothesis was measured and is false: r(MW, fit_mean) =
+0.132, no correlation, wrong sign. NC1 (charge anchor removed) is discriminated
by +96.9 kcal/mol; NC2 (keeps the carboxylate, deletes a distal nitrile) is
within noise. The limit is POSE DETERMINATION — a rigid scaffold overlay cannot
place a substituent 10+ A from the anchor well enough to resolve it. Recorded
here because a pre-registered hypothesis that turns out wrong is still sitting
in the docstring above, and the next reader must not act on it.

## Why a Pareto front and not a weighted score

Combining liability, fit and strain into one number requires exchange rates
between a structural-alert density, a kcal/mol interaction energy, and a
kcal/mol strain. No such exchange rate is defensible from anything measured
here, and inventing one would silently encode the conclusion. So the output is
the **non-dominated set**: candidates that no other candidate beats on every
axis simultaneously. That is a claim the data can support.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Candidate:
    """One molecule's measurements across all arms.

    Every axis is Optional. A candidate missing an axis is NOT given a default:
    it is excluded from Pareto comparison on that axis and flagged, because a
    filled-in default would let an unmeasured candidate dominate a measured one.
    """
    label: str
    smiles: str = ""
    hypothesis: str = ""
    is_negative_control: bool = False
    # Higher = more liability (from tools.tox). Rank-only, dimensionless.
    liability: float | None = None
    # Interaction energy with the pocket, kcal/mol. MORE NEGATIVE IS BETTER.
    fit_kcal: float | None = None
    # Conformational strain, kcal/mol. LOWER IS BETTER.
    strain_kcal: float | None = None
    # PRECISION of `fit_kcal`: the standard error of the mean, in kcal/mol.
    #
    # This must be the SEM, not the pose-to-pose range. The first version of
    # this field held the range, and that made `noise_exceeds_signal` compare
    # the wrong quantities: a range does not shrink as you sample more (it
    # GROWS, since extremes accumulate), whereas the precision of a mean falls
    # as 1/sqrt(n). Measured 2026-08-29 on the anion run: ranges of 118-253
    # kcal/mol but SEMs of only 5-10 kcal/mol at n~40. Judging precision by the
    # range declared a metric unusable when its means were in fact good to
    # ~7 kcal/mol.
    fit_sem_kcal: float | None = None
    # Pose-to-pose range, kept for reporting only -- it describes the pose
    # ENSEMBLE's width, which is a real property worth seeing, but it is not the
    # precision of the mean and must never be used as such.
    fit_range_kcal: float | None = None
    notes: list[str] = field(default_factory=list)

    def axes(self, reference_fit: float | None = None) -> dict[str, float | None]:
        """The three axes, all oriented LOWER-IS-BETTER for uniform comparison.

        `fit_loss` is the loss RELATIVE to a reference (normally the parent):
        positive means this candidate binds less favorably than the reference.
        Expressed as a loss rather than a raw energy so that "lower is better"
        holds on all three axes without a per-axis sign table -- a dropped sign
        in a Pareto comparison silently inverts the whole ranking.
        """
        fit_loss = None
        if self.fit_kcal is not None and reference_fit is not None:
            fit_loss = self.fit_kcal - reference_fit
        return {
            "liability": self.liability,
            "fit_loss": fit_loss,
            "strain": self.strain_kcal,
        }


@dataclass
class GateResult:
    passed: bool
    detail: str
    parent_fit: float | None = None
    control_fits: dict[str, float | None] = field(default_factory=dict)


def fit_discriminates_controls(
    candidates: list[Candidate],
    parent_label: str = "parent",
    margin_kcal: float = 1.0,
) -> GateResult:
    """Does the fit metric penalize the pharmacophore-breaking controls?

    Passes only if EVERY negative control scores at least `margin_kcal` worse
    (less negative interaction) than the parent. `margin_kcal` guards against
    calling a sub-noise difference a discrimination.

    Returns a `GateResult` rather than raising: a failed gate is a finding to be
    reported, not an exception to be swallowed.
    """
    by_label = {c.label: c for c in candidates}
    parent = by_label.get(parent_label)
    if parent is None or parent.fit_kcal is None:
        return GateResult(
            False,
            f"cannot gate: parent {parent_label!r} has no fit measurement, so "
            "there is nothing to compare the controls against",
        )

    controls = [c for c in candidates if c.is_negative_control]
    if not controls:
        return GateResult(
            False,
            "cannot gate: no negative controls in the candidate set. A fit "
            "ranking with no controls cannot distinguish a working metric from "
            "one that returns the same number for everything.",
        )

    measured = {c.label: c.fit_kcal for c in controls}
    unmeasured = [lbl for lbl, v in measured.items() if v is None]
    if unmeasured:
        return GateResult(
            False,
            f"cannot gate: controls {unmeasured} have no fit measurement",
            parent.fit_kcal, measured,
        )

    failures = []
    for c in controls:
        # fit is an interaction energy: more negative = better binding. A
        # control must be WORSE, i.e. LESS negative, i.e. numerically LARGER.
        # The bar is the larger of `margin_kcal` and the statistical resolution
        # limit for this pair -- demanding discrimination finer than the
        # estimator can measure would fail every metric regardless of merit.
        bar = margin_kcal
        if c.fit_sem_kcal is not None and parent.fit_sem_kcal is not None:
            se_diff = (c.fit_sem_kcal ** 2 + parent.fit_sem_kcal ** 2) ** 0.5
            bar = max(margin_kcal, SIGNIFICANCE_SIGMA * se_diff)
        if c.fit_kcal < parent.fit_kcal + bar:
            failures.append(
                f"{c.label} fit {c.fit_kcal:+.2f} vs parent {parent.fit_kcal:+.2f} "
                f"kcal/mol (needs to be at least {bar:.1f} worse)"
            )

    if failures:
        # Report the OBSERVATION, and only such causes as the data at hand
        # actually supports. The first version of this message asserted the
        # metric was "tracking molecular size"; that turned out to be FALSE
        # (r(MW, fit_mean) = +0.132, measured 2026-08-29). Naming an unverified
        # cause in a failure message is worse than naming none, because it
        # sends the next reader down the wrong path.
        diagnosis = (
            "Cause not established by this gate alone. Check, in order: (a) is "
            "the pose noise larger than the candidate-to-candidate range? "
            "(`noise_exceeds_signal` below); (b) are all candidates treated "
            "identically (same pose count, same estimator)?; (c) does fit "
            "correlate with molecular size?"
        )
        # Only substitute the specific diagnosis when precision was actually
        # MEASURED and found wanting. `noise_exceeds_signal` also returns
        # passed=False when it cannot assess precision at all, and reporting
        # "cannot assess" as the cause of the failure would be misleading --
        # "we don't know" is not a diagnosis.
        noise = noise_exceeds_signal(candidates)
        if not noise.passed and "NOISE-DOMINATED" in noise.detail:
            diagnosis = noise.detail
        return GateResult(
            False,
            "FIT METRIC DOES NOT DISCRIMINATE. A control that deletes a "
            "pharmacophore contact scored as well as or better than the parent: "
            + "; ".join(failures)
            + ". No candidate fit ranking derived from this metric should be "
            "reported. " + diagnosis,
            parent.fit_kcal, measured,
        )

    return GateResult(
        True,
        "fit metric penalizes every pharmacophore-breaking control by at least "
        f"{margin_kcal:.1f} kcal/mol relative to the parent",
        parent.fit_kcal, measured,
    )


# A candidate-to-candidate difference must clear this many standard errors to
# be called a difference. 2 is the conventional ~95% two-sided bar for a
# difference of two means; it is a reporting threshold, not a fitted parameter.
SIGNIFICANCE_SIGMA = 2.0


def noise_exceeds_signal(candidates: list[Candidate]) -> GateResult:
    """Is the fit estimator precise enough to resolve candidate differences?

    Compares the candidate-to-candidate range (the SIGNAL) against the typical
    standard error of the individual means (the PRECISION), and requires the
    signal to clear `SIGNIFICANCE_SIGMA` standard errors of a difference.

    IMPORTANT -- what the right noise measure is. This function originally used
    the pose-to-pose RANGE as "noise", which is wrong: a range grows with sample
    count (extremes accumulate) while the precision of a mean falls as
    1/sqrt(n). Measured 2026-08-29 on the anion run, ranges were 118-253 kcal/mol
    while the SEMs were 5-10 kcal/mol at n~40 -- so the range-based test called
    a metric unusable when its means were good to ~7 kcal/mol. Convergence was
    verified separately (out/convergence.log): SEM fell 5.66 -> 3.33 -> 1.99 ->
    1.35 kcal/mol at n = 5, 10, 20, 40, i.e. clean 1/sqrt(n) behaviour.

    `passed=True` means the estimator can resolve the spread of candidates.
    """
    usable = [
        c for c in candidates
        if c.fit_kcal is not None and c.fit_sem_kcal is not None
    ]
    if len(usable) < 2:
        return GateResult(
            False,
            "cannot assess precision: fewer than two candidates carry both a "
            "fit estimate and its standard error",
        )

    fits = [c.fit_kcal for c in usable]
    signal = max(fits) - min(fits)
    sem = sum(c.fit_sem_kcal for c in usable) / len(usable)
    # Standard error of a DIFFERENCE of two independent means.
    resolvable = SIGNIFICANCE_SIGMA * sem * (2 ** 0.5)

    if signal < resolvable:
        return GateResult(
            False,
            f"NOISE-DOMINATED: the candidate-to-candidate range is "
            f"{signal:.1f} kcal/mol, but with a typical standard error of "
            f"{sem:.1f} kcal/mol per mean, the smallest resolvable difference is "
            f"{resolvable:.1f} kcal/mol ({SIGNIFICANCE_SIGMA:g} sigma). No "
            "ordering of these candidates is supportable. More poses per "
            "candidate would shrink the standard error as 1/sqrt(n); a different "
            "scoring function would not.",
        )
    return GateResult(
        True,
        f"precision adequate: candidate range {signal:.1f} kcal/mol vs a "
        f"{resolvable:.1f} kcal/mol resolution limit "
        f"({SIGNIFICANCE_SIGMA:g} sigma on a {sem:.1f} kcal/mol standard error). "
        "NOTE this licenses only differences that individually clear that limit "
        "-- check pairwise before ranking two close candidates.",
    )


def significant_difference(
    a: Candidate, b: Candidate, sigma: float = SIGNIFICANCE_SIGMA
) -> bool | None:
    """Is a's fit distinguishable from b's? `None` if either lacks precision.

    Pairwise is the honest granularity: the aggregate precision check licenses
    the SPREAD of the set, not every pair within it. Two candidates 3 kcal/mol
    apart with 7 kcal/mol standard errors are not distinguishable even in a set
    whose overall range is comfortably resolvable.
    """
    if (a.fit_kcal is None or b.fit_kcal is None
            or a.fit_sem_kcal is None or b.fit_sem_kcal is None):
        return None
    se_diff = (a.fit_sem_kcal ** 2 + b.fit_sem_kcal ** 2) ** 0.5
    return abs(a.fit_kcal - b.fit_kcal) >= sigma * se_diff


def dominates(a: dict[str, float | None], b: dict[str, float | None]) -> bool:
    """True if `a` is at least as good as `b` on every COMPARABLE axis and
    strictly better on one. Axes where either side is missing are skipped, so an
    unmeasured candidate can neither dominate nor be dominated on that axis.
    """
    comparable = [k for k in a if a[k] is not None and b[k] is not None]
    if not comparable:
        return False
    if not all(a[k] <= b[k] for k in comparable):
        return False
    return any(a[k] < b[k] for k in comparable)


def pareto_front(
    candidates: list[Candidate],
    parent_label: str = "parent",
    exclude_controls: bool = True,
) -> list[Candidate]:
    """Non-dominated candidates over (liability, fit_loss, strain).

    Controls are excluded by default: they exist to validate the metric, not to
    be recommended, and a control can easily be non-dominated (e.g. low
    liability because it deleted the acid) which would put a deliberately
    inactive molecule on the recommended front.
    """
    by_label = {c.label: c for c in candidates}
    parent = by_label.get(parent_label)
    ref_fit = parent.fit_kcal if parent else None

    pool = [c for c in candidates if not (exclude_controls and c.is_negative_control)]
    front = []
    for c in pool:
        ac = c.axes(ref_fit)
        if not any(dominates(o.axes(ref_fit), ac) for o in pool if o is not c):
            front.append(c)
    return front


def format_table(
    candidates: list[Candidate], parent_label: str = "parent"
) -> str:
    """Human-readable summary. Missing axes print as `--`, never as 0."""
    by_label = {c.label: c for c in candidates}
    parent = by_label.get(parent_label)
    ref_fit = parent.fit_kcal if parent else None
    front = {c.label for c in pareto_front(candidates, parent_label)}

    def fmt(v, spec="8.2f"):
        return "      --" if v is None else format(v, spec)

    lines = [
        f"{'label':30s} {'liability':>9s} {'fit(kcal)':>10s} "
        f"{'fitloss':>8s} {'strain':>8s}  {'front':>5s}  hypothesis",
        "-" * 100,
    ]
    for c in candidates:
        ax = c.axes(ref_fit)
        tag = "NC" if c.is_negative_control else ("*" if c.label in front else "")
        lines.append(
            f"{c.label:30s} {fmt(c.liability, '9.3f')} {fmt(c.fit_kcal, '10.2f')} "
            f"{fmt(ax['fit_loss'])} {fmt(c.strain_kcal)}  {tag:>5s}  {c.hypothesis}"
        )
    return "\n".join(lines)
