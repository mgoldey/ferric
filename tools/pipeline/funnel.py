"""Run an ordered tier stack, narrowing the population at each step.

Every tier's job is to DISCARD, cheaply, what the next cannot afford to examine
(see `tools/campaign/hierarchy.py`). This module is that loop, plus the
bookkeeping that makes the funnel auditable: how many entered each tier, how
many survived, how many FAILED, and every raw result.

The bookkeeping is the point. A funnel that reports only its survivors cannot
distinguish "this candidate was rejected on its merits" from "this candidate
crashed and was quietly dropped" -- and those demand opposite responses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from tools.campaign.hierarchy import Tier, TierOutcome
from tools.isomers.model import Isomer
from tools.pipeline.tiers import TierResult

TierFn = Callable[[Isomer, dict], TierResult]


@dataclass
class Stage:
    """One tier in the stack. `keep` is how many survivors pass downward."""
    tier: Tier
    fn: TierFn
    keep: int
    name: str


@dataclass
class FunnelReport:
    outcomes: list[TierOutcome] = field(default_factory=list)
    survivors: list[Isomer] = field(default_factory=list)
    results: dict[str, list[TierResult]] = field(default_factory=dict)

    def value(self, stage_name: str, canonical: str) -> float | None:
        for r in self.results.get(stage_name, []):
            if r.candidate_id == canonical:
                return r.value
        return None

    def table(self) -> str:
        lines = [f"{'tier':>4s}  {'stage':12s} {'in':>5s} {'out':>5s} {'failed':>7s}  note",
                 "-" * 72]
        for o in self.outcomes:
            lines.append(f"{int(o.tier):>4d}  {o.note.split(':')[0]:12s} "
                         f"{o.n_in:5d} {o.n_out:5d} {o.n_failed:7d}  {o.note}")
        return "\n".join(lines)


def run_funnel(candidates: list[Isomer], stages: list[Stage],
               context: dict[str, Any]) -> FunnelReport:
    """Narrow `candidates` through `stages`, cheapest first.

    Ranking is ASCENDING by value at every tier, because every tier here reports
    an energy or an energy-like score where lower is better. A candidate the
    tier FAILED on is dropped and counted -- never ranked, and never treated as
    having scored well.

    Stops early on an empty population rather than running an expensive tier on
    nothing.
    """
    rep = FunnelReport()
    population = list(candidates)

    for stage in stages:
        if not population:
            break
        results = [stage.fn(iso, context) for iso in population]
        rep.results[stage.name] = results

        by_id = {r.candidate_id: r for r in results}
        ok = [iso for iso in population
              if iso.canonical in by_id and by_id[iso.canonical].ok]
        ok.sort(key=lambda iso: by_id[iso.canonical].value)
        survivors = ok[:stage.keep]

        rep.outcomes.append(TierOutcome(
            tier=stage.tier, n_in=len(population), n_out=len(survivors),
            n_failed=len(population) - len(ok),
            note=f"{stage.name}: kept {len(survivors)} of {len(ok)} scored",
            errors=[r.error for r in results if r.error][:10],
        ))
        population = survivors

    rep.survivors = population
    return rep
