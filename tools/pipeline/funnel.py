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

import time
from concurrent.futures import ProcessPoolExecutor
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
    workers: int = 1
    """Processes to fan this tier's candidates across. 1 = run in-process.

    PROCESSES, not threads: the expensive tiers call into libraries that are
    not thread-safe (libxtb) or that hold the GIL, and ferric/OpenBLAS wants
    one BLAS thread per process anyway. Candidates are independent, so this is
    a pure fan-out with no shared state.

    Results are re-ordered to match the input population before ranking, so
    the survivor set is IDENTICAL to a serial run. That is not incidental --
    the suite's reproducibility guarantee depends on it, and it is tested.

    Leave at 1 for tiers whose per-candidate cost is already negligible: the
    process-spawn and pickling overhead would dominate.
    """


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
        lines = [f"{'tier':>4s}  {'stage':12s} {'in':>5s} {'out':>5s} {'failed':>7s} "
                 f"{'secs':>8s} {'s/cand':>8s}  note",
                 "-" * 96]
        for o in self.outcomes:
            secs = "-" if o.seconds is None else f"{o.seconds:.1f}"
            per = ("-" if o.seconds_per_candidate is None
                   else f"{o.seconds_per_candidate:.2f}")
            lines.append(f"{int(o.tier):>4d}  {o.note.split(':')[0]:12s} "
                         f"{o.n_in:5d} {o.n_out:5d} {o.n_failed:7d} "
                         f"{secs:>8s} {per:>8s}  {o.note}")
        total = sum(o.seconds for o in self.outcomes if o.seconds is not None)
        if total:
            lines.append("-" * 96)
            lines.append(f"{'':4s}  {'TOTAL':12s} {'':5s} {'':5s} {'':7s} "
                         f"{total:8.1f}")
            # Which tier actually cost the run? That is the tuning question,
            # and the answer is routinely not the cost table's prediction.
            worst = max((o for o in self.outcomes if o.seconds is not None),
                        key=lambda o: o.seconds, default=None)
            if worst is not None:
                share = 100.0 * worst.seconds / total
                lines.append(f"{'':4s}  dominant tier {int(worst.tier)} "
                             f"({worst.note.split(':')[0]}) = {share:.0f}% of wall")
        return "\n".join(lines)


def _run_stage(stage: Stage, population: list[Isomer],
               context: dict[str, Any]) -> list[TierResult]:
    """Evaluate one tier over a population, serially or fanned out.

    The parallel path preserves INPUT ORDER (results are placed back by index),
    so the ranking it feeds is identical to the serial path's. A tier that
    returned results in completion order would silently reorder ties and break
    the suite's reproducibility guarantee.

    **A dead worker costs one candidate, not the run.** `pool.map` raises
    `BrokenProcessPool` if any worker dies -- an OS-level kill (OOM reaper,
    cgroup pressure, a segfault in a native library) is not catchable inside
    the worker -- and a bare `list(pool.map(...))` therefore discards every
    result already computed. On a 174-dock screen that is an hour of work lost
    to one casualty, and it presents as the whole pipeline vanishing with no
    traceback, which is what happened twice on 2026-09-03.

    Submitting per-future instead confines the damage: a dead worker yields a
    failed `TierResult` for its own candidate, which the funnel already knows
    how to count and report.
    """
    if stage.workers <= 1 or len(population) < 2:
        return [stage.fn(iso, context) for iso in population]

    results: list[TierResult | None] = [None] * len(population)
    with ProcessPoolExecutor(max_workers=stage.workers) as pool:
        futures = {pool.submit(_apply, (stage.fn, iso, context)): i
                   for i, iso in enumerate(population)}
        for fut, i in futures.items():
            try:
                results[i] = fut.result()
            except Exception as e:  # noqa: BLE001 -- incl. BrokenProcessPool
                results[i] = TierResult(
                    population[i].canonical, None,
                    f"worker died: {type(e).__name__}: {e}")
    return [r if r is not None else
            TierResult(population[i].canonical, None, "no result from worker")
            for i, r in enumerate(results)]


def _apply(args: tuple) -> TierResult:
    """Top-level so it is picklable by ProcessPoolExecutor."""
    fn, iso, context = args
    return fn(iso, context)


def run_funnel(candidates: list[Isomer], stages: list[Stage],
               context: dict[str, Any]) -> FunnelReport:
    """Narrow `candidates` through `stages`, cheapest first.

    Ranking is ASCENDING by value at every tier, because every tier here reports
    an energy or an energy-like score where lower is better. A candidate the
    tier FAILED on is dropped and counted -- never ranked, and never treated as
    having scored well.

    Stops early on an empty population rather than running an expensive tier on
    nothing.

    Each tier is TIMED. Without per-tier wall times a funnel cannot answer the
    only question that matters for tuning it -- which tier is actually costing
    the run -- and the answer is routinely not the one the cost table predicts.
    """
    rep = FunnelReport()
    population = list(candidates)

    for stage in stages:
        if not population:
            break
        t0 = time.time()
        results = _run_stage(stage, population, context)
        elapsed = time.time() - t0
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
            seconds=elapsed,
        ))
        population = survivors

    rep.survivors = population
    return rep
