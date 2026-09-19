"""One call that turns a campaign's output into the figures worth looking at.

`energy_plots` and `molecules` give ten functions and no guidance on which to
reach for. This is the worked answer: hand it what a substitution campaign
actually produces and get back the figure set, each one already carrying the
caveats the measurements demand.

## Why a report function rather than a notebook

A notebook drifts from the code it plots. This lives next to the plots, is
imported by tests, and fails loudly when an input shape changes -- so the
"example" cannot rot into something that no longer runs.

## What it refuses to do

**It will not plot a ranking it cannot support.** `campaign_report` takes
`ddE_noise` and passes it to every plot that can express a resolution limit. If
you do not supply one it raises, because the single most common misuse of these
figures is reading an ordering off a heatmap whose cells are all inside the
noise. MEASURED on danuglipron: the best available ddE noise is 4.07-4.68
kcal/mol against substituent effects of 1-2, so an honest heatmap of that
campaign is entirely greyed out.

That is not a limitation of the plotting -- it is the result, rendered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

__all__ = ["CampaignFigures", "campaign_report"]


@dataclass
class CampaignFigures:
    """The figures a substitution campaign should look at, and why.

    Every field is `None` when its input was not supplied -- never a blank
    figure, which would read as "measured and found empty".
    """

    #: Candidates surviving each tier. Answers "where did the budget go?".
    funnel: object | None = None
    #: ddE per (substituent, SITE), greyed below the noise floor. The unit the
    #: campaign reports on.
    heatmap: object | None = None
    #: Per-pose spread per candidate. The plot that makes the pose problem
    #: visible rather than a footnote.
    poses: object | None = None
    #: Liability endpoints, oriented so UP is worse.
    liabilities: object | None = None
    #: Notes attached to the figure set: what it may and may not be read for.
    caveats: list[str] = field(default_factory=list)

    def close(self) -> None:
        """Release every figure. pyplot retains them; a batch leaks otherwise."""
        from tools.viz.energy_plots import close as _close

        for f in (self.funnel, self.heatmap, self.poses, self.liabilities):
            if f is not None:
                _close(f)


def campaign_report(
    *,
    ddE_noise: float,
    funnel_stages: Sequence[str] | None = None,
    funnel_counts: Sequence[int] | None = None,
    ddE: dict[tuple[str, str], float | None] | None = None,
    poses: dict[str, Sequence[float]] | None = None,
    liabilities: dict[str, dict[str, tuple[float | None, bool]]] | None = None,
    parent: str | None = None,
    unit: str = "kcal/mol",
) -> CampaignFigures:
    """Build the figure set for one campaign.

    `ddE_noise` is REQUIRED and has no default. It is the measured resolution
    limit of whatever protocol produced `ddE` -- every plot that can express a
    limit gets it, and the caveats list says what that implies. A default would
    be a guess about someone else's protocol, and the guess would silently
    license a ranking.

    Each input is optional: pass what the campaign produced. A figure whose
    input is absent comes back `None` rather than empty.
    """
    from tools.viz.energy_plots import (
        funnel_survival,
        liability_profile,
        pose_ensemble,
        site_substituent_heatmap,
    )

    if not (isinstance(ddE_noise, (int, float)) and ddE_noise > 0):
        raise ValueError(
            f"ddE_noise must be a positive number, got {ddE_noise!r}. It is the "
            "MEASURED resolution limit of the protocol that produced these "
            "numbers; there is no sensible default, and omitting it would let a "
            "reader order candidates the data cannot separate."
        )

    figs = CampaignFigures()
    caveats: list[str] = []

    if funnel_stages is not None and funnel_counts is not None:
        figs.funnel = funnel_survival(funnel_stages, funnel_counts)

    if ddE:
        figs.heatmap = site_substituent_heatmap(ddE, unit=unit, noise_floor=ddE_noise)
        real = [v for v in ddE.values() if v is not None]
        below = [v for v in real if abs(v) < ddE_noise]
        if real and len(below) == len(real):
            caveats.append(
                f"EVERY ddE ({len(real)}/{len(real)}) is inside the {ddE_noise:g} "
                f"{unit} noise floor. The heatmap shows WHICH substitutions were "
                "tried, not which are better. Do not order them."
            )
        elif below:
            caveats.append(
                f"{len(below)} of {len(real)} ddE values are inside the "
                f"{ddE_noise:g} {unit} noise floor and are greyed; only the "
                "others carry an ordering."
            )
        # The site axis is a separate, harder limit than the noise floor.
        sites = {k[1] for k in ddE}
        if len(sites) > 1:
            caveats.append(
                "Cheap descriptor gates are SITE-BLIND by construction (whole-"
                "molecule MW/cLogP/TPSA are identical for constitutional "
                "isomers), so a per-site ddE must come from a pose-based tier -- "
                "check which produced these before reading the site axis."
            )

    if poses:
        figs.poses = pose_ensemble(poses, unit=unit)
        import statistics

        spreads = [
            statistics.stdev(v)
            for v in poses.values()
            if len([x for x in v if x is not None]) > 1
        ]
        if spreads:
            # Derive the implied pose count rather than testing against a magic
            # ratio. A ddE over two independent means of n poses each has noise
            # sd*sqrt(2/n), so a STATED floor implies n = 2*(sd/floor)^2. If
            # that n is absurd, the floor and the poses disagree.
            #
            # Worked: sd 28.75 with floor 4.68 implies n = 75, which is the
            # right order for the n=100 the floor was derived at -- so a naive
            # "sd >> floor" check would FALSE-ALARM here at 6.1x. The number
            # that matters is the implied n, not the ratio.
            sd_max = max(spreads)
            implied_n = 2.0 * (sd_max / ddE_noise) ** 2
            largest = max(len([x for x in v if x is not None]) for v in poses.values())
            if implied_n > 10 * max(largest, 1):
                caveats.append(
                    f"The stated {ddE_noise:g} {unit} floor implies averaging over "
                    f"~{implied_n:.0f} poses, but the largest ensemble here has "
                    f"{largest}. Either the floor came from a much bigger run, or "
                    "it is optimistic for THESE poses (per-pose sd "
                    f"{sd_max:.1f} {unit})."
                )

    if liabilities:
        figs.liabilities = liability_profile(liabilities, parent=parent)
        caveats.append(
            "Liability endpoints are published ALERT SETS, not probabilities of "
            "harm. A compound with no alerts is unflagged, not predicted safe."
        )

    figs.caveats = caveats
    return figs
