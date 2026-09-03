"""The cost hierarchy: which method does which job, and how a tier is retired.

## The principle

Every tier exists to DISCARD, cheaply, what the next tier cannot afford to
examine. A tier is chosen for its COST and its DISCRIMINATION, not for being
"the best method" — the best method applied to the wrong candidates is waste,
and a cheap method asked to make a fine distinction is noise.

    tier                cost/pose      poses      job
    ------------------  -------------  ---------  --------------------------
    1  Vina (empirical) ~10 us         10^5-10^6  SEARCH pose space
    2  MMFF / GFN-FF    ~ms            10^2-10^3  relax, drop clashes
    3  GFN2-xTB (DFTB)  ~0.5 s         10-10^2    rank survivors
    4  DFT + dispersion minutes-hours  1-10       final energetics

## Why this campaign needed it stated

The danuglipron campaign ran tier 3 alone, fed by free-solution conformers. It
had no tier 1, so it never searched; and no tier 4, so no number was ever better
than semiempirical. Four rounds of increasingly careful statistics were spent
characterising a metric fed geometries 2.2-4.1 A from the binding mode. See
`experiments/danuglipron/RESULTS.md` M3-M8.

Measured, once tier 1 was added: Vina reproduced the crystal pose at **0.95 A**
in ~2 minutes. The best any tier-3 method achieved in 62 minutes was 2.41 A.

## The division of labour is empirical, not assumed

Also measured on that run: r(vina_score, pose RMSD) = **+0.461**, and only 4 of
20 poses were under 2.0 A. So the cheap score is weakly informative about pose
quality — it FINDS the right pose but cannot reliably RANK it. That is the
justification for tiers 2-4 existing at all: if the tier-1 score could pick its
own best pose, no rescoring would be needed.

## Rules for using this, and for retiring a tier

1.  **VALIDATE EACH TIER AGAINST GROUND TRUTH BEFORE TRUSTING IT.** For pose
    generation that means redocking a known complex and measuring RMSD; the
    check costs minutes and is the difference between a pipeline and a
    guess. `tools.campaign.align.pose_quality_gate` is the gate.
2.  **A TIER'S OUTPUT IS ONLY AS GOOD AS ITS INPUT.** No amount of tier-4 rigour
    rescues a tier-1 failure. Diagnose DOWNWARD: when results look wrong, check
    the cheapest tier first, because that is where the population is set.
3.  **DO NOT ASK A TIER FOR A DISTINCTION FINER THAN ITS NOISE.** Quantify it:
    compare the candidate-to-candidate range against the standard error of the
    estimate (`rank.noise_exceeds_signal`), and never against a sample range,
    which grows with n.
4.  **A TIER IS RETIRED WHEN A CHEAPER ONE MATCHES IT, OR WHEN ITS JOB IS
    SOMEONE ELSE'S.** MD-as-pose-search was retired here not because it failed
    at an adequate timescale — it was never run at one — but because reaching
    that timescale (141 s/ps, ~26 h for the analogue set at 20 ps) means using a
    quantum method for a job an empirical search does in seconds.
5.  **KEEP EACH TIER'S FAILURE VISIBLE.** A tier that cannot answer returns
    None/UNEVALUATED, never a neutral-looking number. A fabricated zero in a
    liability score reads as "maximally safe"; a fabricated pose reads as a
    binding mode.
6.  **DO NOT BUY PRECISION BELOW A METHOD'S OWN ERROR BAR.** A tier's tuning
    knobs have the same noise floor its outputs do, so past some setting the
    extra cost buys nothing measurable. Measured (RESULTS.md M11): raising
    Vina's `exhaustiveness` from 4 to 32 cost **6.8x** and improved the top
    score by **0.005 kcal/mol** — against a scoring function whose published
    RMSE is ~2.5 kcal/mol, i.e. a gain ~500x smaller than its own error bar.
    Redock RMSD across an 8x range of effort moved 0.097 A against a 0.131 A
    between-seed SEM, so it was not resolvable either.

    The corollary is where the budget SHOULD go: to whichever input the answer
    is actually sensitive to. Here that was the starting conformer (0.75-1.24 A
    across ETKDG seeds), so three seeds at the cheap setting beat one seed at
    the expensive one — cheaper AND better sampled. Find the sensitive variable
    by measuring, then spend there.
7.  **TIME EVERY TIER.** "Which tier costs the run" decides where optimization
    effort goes, and it is routinely not what the cost table predicts: this
    campaign twice attributed a run's cost to the wrong tier from estimates
    alone. `TierOutcome.seconds` is None when unmeasured, never 0.0, because a
    zero reads as "free".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Tier(IntEnum):
    """Cost tiers, ordered cheapest first."""
    SEARCH = 1          # empirical docking -- generates poses
    FORCE_FIELD = 2     # MMFF / GFN-FF -- relax, declash
    SEMIEMPIRICAL = 3   # GFN2-xTB -- rank
    QUANTUM = 4         # DFT + dispersion -- final energetics


@dataclass(frozen=True)
class TierSpec:
    """What one tier costs, what it is for, and how it was validated."""
    tier: Tier
    method: str
    seconds_per_pose: float
    typical_poses: str
    job: str
    validated_by: str | None = None

    @property
    def validated(self) -> bool:
        return self.validated_by is not None


@dataclass
class TierOutcome:
    """What a tier did to a candidate population, for an auditable funnel."""
    tier: Tier
    n_in: int
    n_out: int
    n_failed: int = 0
    note: str = ""
    errors: list[str] = field(default_factory=list)
    seconds: float | None = None
    """Wall time this tier spent, or None if it was not timed.

    None rather than 0.0 for an untimed tier: a funnel that reports 0.0 for
    "not measured" invites exactly the arithmetic that produced this campaign's
    worst claims -- a cost split inferred from numbers nobody recorded. The
    hierarchy's whole premise is that each tier's cost justifies the one above
    it, and that premise is untestable without per-tier times.
    """

    @property
    def retained_fraction(self) -> float | None:
        return None if self.n_in == 0 else self.n_out / self.n_in

    @property
    def seconds_per_candidate(self) -> float | None:
        """Wall seconds per candidate ENTERING this tier.

        Per-entrant, not per-survivor: the cost a tier imposes is set by what
        it must examine, not by what it lets through.
        """
        if self.seconds is None or self.n_in == 0:
            return None
        return self.seconds / self.n_in


def unvalidated_tiers(hierarchy: "tuple[TierSpec, ...]") -> list[TierSpec]:
    """Tiers with no ground-truth validation recorded.

    Using one of these is not forbidden -- it is UNVERIFIED, which is a
    different claim and should be reported as such.
    """
    return [t for t in hierarchy if not t.validated]


def describe(hierarchy: "tuple[TierSpec, ...]") -> str:
    lines = [f"{'tier':>4s}  {'method':26s} {'s/pose':>9s} {'poses':>10s}  validated",
             "-" * 78]
    for t in hierarchy:
        lines.append(
            f"{int(t.tier):>4d}  {t.method:26s} {t.seconds_per_pose:9.0e} "
            f"{t.typical_poses:>10s}  {'yes' if t.validated else 'NO'}"
        )
    return "\n".join(lines)
