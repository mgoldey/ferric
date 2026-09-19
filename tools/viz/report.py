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
    #: The two minima an IRC connects, with BOTH barriers. `None` when no
    #: reaction path was supplied -- a catalyst campaign has one, a
    #: substitution campaign does not.
    reaction: object | None = None
    #: Notes attached to the figure set: what it may and may not be read for.
    caveats: list[str] = field(default_factory=list)

    def close(self) -> None:
        """Release every figure. pyplot retains them; a batch leaks otherwise."""
        from tools.viz.energy_plots import close as _close

        for f in (
            self.funnel,
            self.heatmap,
            self.poses,
            self.liabilities,
            self.reaction,
        ):
            if f is not None:
                _close(f)


def campaign_report(
    *,
    ddE_noise: float | dict[tuple[str, str], float],
    funnel_stages: Sequence[str] | None = None,
    funnel_counts: Sequence[int] | None = None,
    ddE: dict[tuple[str, str], float | None] | None = None,
    poses: dict[str, Sequence[float]] | None = None,
    liabilities: dict[str, dict[str, tuple[float | None, bool]]] | None = None,
    parent: str | None = None,
    unit: str = "kcal/mol",
    irc: dict | None = None,
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
        reaction_path,
        site_substituent_heatmap,
    )

    if isinstance(ddE_noise, dict):
        # A PER-CANDIDATE floor, which a paired estimator needs: its noise
        # tracks rho and varies per substituent (MEASURED 0.221 at rho 0.860 vs
        # 0.615 at rho 0.399, RESULTS.md M17). Every entry must still be a
        # positive number -- a None or a zero would read as "resolved at any
        # magnitude", which is the opposite of an unmeasured floor.
        if not ddE_noise:
            raise ValueError(
                "ddE_noise is an empty mapping. Pass the measured floor for at "
                "least one (substituent, site), or a scalar for the whole "
                "campaign; an empty mapping greys nothing and silently licenses "
                "the ordering this argument exists to prevent."
            )
        bad = {
            k: v
            for k, v in ddE_noise.items()
            if not (isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0)
        }
        if bad:
            raise ValueError(
                f"every ddE_noise entry must be a positive number; bad entries: "
                f"{bad!r}. A zero or None floor marks a cell as resolved at any "
                "magnitude -- if a candidate's floor was not measured, LEAVE IT "
                "OUT, which renders as unbounded rather than as perfect."
            )
    elif not (
        isinstance(ddE_noise, (int, float))
        and not isinstance(ddE_noise, bool)
        and ddE_noise > 0
    ):
        raise ValueError(
            f"ddE_noise must be a positive number, or a per-(substituent, site) "
            f"mapping of them, got {ddE_noise!r}. It is the "
            "MEASURED resolution limit of the protocol that produced these "
            "numbers; there is no sensible default, and omitting it would let a "
            "reader order candidates the data cannot separate."
        )

    # Half a funnel is a configuration error, not a smaller report. Silently
    # dropping the plot hides the mistake in the one output nobody re-reads.
    if (funnel_stages is None) != (funnel_counts is None):
        raise ValueError(
            "funnel_stages and funnel_counts must be given together or not at "
            f"all; got stages={'set' if funnel_stages is not None else 'None'}, "
            f"counts={'set' if funnel_counts is not None else 'None'}"
        )

    figs = CampaignFigures()
    caveats: list[str] = []

    try:
        _build(
            figs,
            caveats,
            ddE_noise=ddE_noise,
            funnel_stages=funnel_stages,
            funnel_counts=funnel_counts,
            ddE=ddE,
            poses=poses,
            liabilities=liabilities,
            parent=parent,
            unit=unit,
            irc=irc,
            funnel_survival=funnel_survival,
            liability_profile=liability_profile,
            pose_ensemble=pose_ensemble,
            reaction_path=reaction_path,
            site_substituent_heatmap=site_substituent_heatmap,
        )
    except Exception:
        # A later helper can raise after an earlier figure was built -- a valid
        # heatmap followed by an invalid liability parent, say. pyplot RETAINS
        # every figure it made, and the caller never receives `figs`, so it
        # cannot call `close()`. Without this the failure leaks a figure per
        # attempt, which in a loop is how a batch job runs out of memory.
        figs.close()
        raise

    figs.caveats = caveats
    return figs


def _build(
    figs: CampaignFigures,
    caveats: list[str],
    *,
    ddE_noise,
    funnel_stages,
    funnel_counts,
    ddE,
    poses,
    liabilities,
    parent,
    unit: str,
    irc,
    funnel_survival,
    liability_profile,
    pose_ensemble,
    reaction_path,
    site_substituent_heatmap,
) -> None:
    """Fill `figs` and `caveats`. Split out only so the caller can clean up."""
    if funnel_stages is not None and funnel_counts is not None:
        figs.funnel = funnel_survival(funnel_stages, funnel_counts)

    if ddE:
        figs.heatmap = site_substituent_heatmap(ddE, unit=unit, noise_floor=ddE_noise)

        # THE CAVEATS MUST BE COMPUTED PER CELL, not against one number.
        # With a mapping, "is this inside the floor?" has a different answer per
        # candidate, and a cell whose floor was never measured is neither inside
        # nor outside -- it is UNBOUNDED, and saying so is the point.
        def _floor(key):
            if isinstance(ddE_noise, dict):
                return ddE_noise.get(key)
            return ddE_noise

        real = [(k, v) for k, v in ddE.items() if v is not None]
        below = [
            (k, v) for k, v in real if (_floor(k) is not None and abs(v) < _floor(k))
        ]
        unbounded = [k for k, _ in real if _floor(k) is None]
        if unbounded:
            caveats.append(
                f"{len(unbounded)} of {len(real)} cells have NO measured noise "
                "floor and are drawn ungreyed. Ungreyed here means UNMEASURED, "
                "not resolved -- do not read an ordering across them."
            )
        floor_desc = (
            f"{min(v for v in ddE_noise.values()):g}-"
            f"{max(v for v in ddE_noise.values()):g} (per-candidate)"
            if isinstance(ddE_noise, dict)
            else f"{ddE_noise:g}"
        )
        if real and len(below) == len(real):
            caveats.append(
                f"EVERY ddE ({len(real)}/{len(real)}) is inside the {floor_desc} "
                f"{unit} noise floor. The heatmap shows WHICH substitutions were "
                "tried, not which are better. Do not order them."
            )
        elif below:
            # "the others" must mean the others WITH A MEASURED FLOOR. A cell
            # with no floor is not resolved, it is unjudged -- lumping it in
            # here contradicts the unbounded caveat added just above and hands
            # back exactly the ordering this argument exists to withhold.
            measured = len(real) - len(unbounded)
            caveats.append(
                f"{len(below)} of {measured} ddE values WITH A MEASURED FLOOR "
                f"are inside the {floor_desc} {unit} noise floor and are "
                f"greyed; only the remaining {measured - len(below)} measured "
                "cells carry an ordering"
                + (
                    f" (the {len(unbounded)} unmeasured cell(s) carry none)."
                    if unbounded
                    else "."
                )
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
            # With a per-candidate mapping there is no single floor to invert,
            # so use the SMALLEST (the most optimistic claim on offer) -- that
            # is the one whose implied pose count is hardest to justify.
            floor_for_n = (
                min(ddE_noise.values()) if isinstance(ddE_noise, dict) else ddE_noise
            )
            implied_n = 2.0 * (sd_max / floor_for_n) ** 2
            largest = max(len([x for x in v if x is not None]) for v in poses.values())
            if implied_n > 10 * max(largest, 1):
                caveats.append(
                    f"The stated {floor_for_n:g} {unit} floor implies averaging over "
                    f"~{implied_n:.0f} poses, but the largest ensemble here has "
                    f"{largest}. Either the floor came from a much bigger run, or "
                    "it is optimistic for THESE poses (per-pose sd "
                    f"{sd_max:.1f} {unit})."
                )

    if irc:
        # THE IRC HAS ITS OWN UNIT, and it is not the campaign's.
        #
        # `unit` describes the ddE values, which are kcal/mol by default.
        # `IrcResult` energies come straight from ferric and are HARTREE. Using
        # the shared `unit` would label -55.44 Ha as kcal/mol -- a 627x error,
        # and one that renders as a perfectly ordinary plot.
        #
        # Taken from the `irc` dict so a caller who really does have kcal/mol
        # can say so, defaulting to hartree because that is what the API
        # returns.
        irc_args = dict(irc)
        irc_unit = irc_args.pop("unit", "hartree")
        figs.reaction = reaction_path(unit=irc_unit, **irc_args)
        if not (irc.get("forward_converged") and irc.get("reverse_converged")):
            caveats.append(
                "An IRC branch did NOT converge: it stopped where its step "
                "budget ran out, not at a basin. That endpoint identifies no "
                "minimum, and the barrier against it is a lower bound."
            )
        else:
            caveats.append(
                "A converged IRC branch means the walk reached a FLAT REGION, "
                "not that the endpoint is a minimum -- confirming that needs a "
                "Hessian there (another 6N+1 gradients per side)."
            )

    if liabilities:
        figs.liabilities = liability_profile(liabilities, parent=parent)
        caveats.append(
            "Liability endpoints are published ALERT SETS, not probabilities of "
            "harm. A compound with no alerts is unflagged, not predicted safe."
        )
