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


class IncomparableError(ValueError):
    """A stage was asked to rank values that are not comparable.

    Raised, not reported per candidate: the defect is the RUN's configuration
    (a total-energy tier cutting a population of differing formulas), it
    applies to every candidate equally, and any survivor list it produced
    would be an ordering by electron count presented as a selection.
    """


def formula(iso: Isomer) -> str:
    """Molecular formula of `iso` INCLUDING hydrogens and net charge.

    The charge is part of it on purpose: a carboxylate and its acid differ by
    a proton, and their total energies are no more comparable than two
    different substituents'.
    """
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    return CalcMolFormula(Chem.MolFromSmiles(iso.canonical))


def require_same_formula(population: list[Isomer], stage_name: str) -> None:
    """Refuse to rank a per-molecule TOTAL across differing formulas.

    A total energy scales with electron count, so ordering a fluoro and a
    chloro analogue by it orders them by the chlorine's 8 extra electrons --
    the heavier substituent "wins" whatever it does in the pocket. Same-formula
    populations (the isomer campaigns the funnel was built for) pass.
    """
    formulas = sorted({formula(iso) for iso in population})
    if len(formulas) > 1:
        raise IncomparableError(
            f"stage {stage_name!r} ranks a TOTAL energy (formula_bound=True) "
            f"across differing formulas {formulas}; a total is comparable only "
            "between isomers. Score a difference against a common reference "
            "instead -- for tiers 3 and 4, context['score'] = 'interaction' "
            "with the pocket's point_charges -- or set keep >= the population "
            "so this stage does not cut."
        )


def paired_delta(parent: list[float], analogue: list[float]) -> list[float]:
    """Per-pose paired difference: analogue pose i minus parent pose i.

    Pairing cancels what the two share at pose i (the pose's own placement
    error), which is why the danuglipron campaign's paired ddE carried ~1-3
    kcal/mol of noise where an unpaired difference carried ~4. Refuses
    unequal lengths: an unpaired remainder is not a pair. A self-pair is
    exactly 0 at every pose (x - x == 0.0 for every finite float).
    """
    if len(parent) != len(analogue):
        raise ValueError(
            f"cannot pair {len(analogue)} analogue poses with "
            f"{len(parent)} parent poses"
        )
    return [a - p for p, a in zip(parent, analogue)]


def _cut(ok: list[Isomer], by_id: dict, keep: int) -> tuple[list[Isomer], str]:
    """Top-`keep` by ascending value, never splitting an UNRESOLVED group.

    Sorted candidates are grouped wherever a neighbour pair is NOT resolved
    (`TierResult.resolves(...) is False`: the gap is inside the two results'
    combined `resolution`). If `keep` falls inside a group, the WHOLE group
    survives and the note says so: the tier cannot order those candidates, so
    dropping some of them would be a decision made by sort order, not by the
    measurement. An uncharacterised resolution (`None`) ranks as before.
    """
    ranked = sorted(ok, key=lambda iso: by_id[iso.canonical].value)
    groups: list[list[Isomer]] = []
    for iso in ranked:
        if groups and (
            by_id[groups[-1][-1].canonical].resolves(by_id[iso.canonical]) is False
        ):
            groups[-1].append(iso)
        else:
            groups.append([iso])
    kept: list[Isomer] = []
    note = ""
    for g in groups:
        if len(kept) >= keep:
            break
        if len(kept) + len(g) > keep:
            note = (
                f"; keep={keep} falls inside an unresolved group of {len(g)}, "
                f"kept all {len(kept) + len(g)}"
            )
        kept += g
    return kept, note


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
        lines = [
            f"{'tier':>4s}  {'stage':12s} {'in':>5s} {'out':>5s} {'failed':>7s} "
            f"{'secs':>8s} {'s/cand':>8s}  note",
            "-" * 96,
        ]
        for o in self.outcomes:
            secs = "-" if o.seconds is None else f"{o.seconds:.1f}"
            per = (
                "-"
                if o.seconds_per_candidate is None
                else f"{o.seconds_per_candidate:.2f}"
            )
            lines.append(
                f"{int(o.tier):>4d}  {o.note.split(':')[0]:12s} "
                f"{o.n_in:5d} {o.n_out:5d} {o.n_failed:7d} "
                f"{secs:>8s} {per:>8s}  {o.note}"
            )
        total = sum(o.seconds for o in self.outcomes if o.seconds is not None)
        if total:
            lines.append("-" * 96)
            lines.append(f"{'':4s}  {'TOTAL':12s} {'':5s} {'':5s} {'':7s} {total:8.1f}")
            # Which tier actually cost the run? That is the tuning question,
            # and the answer is routinely not the cost table's prediction.
            worst = max(
                (o for o in self.outcomes if o.seconds is not None),
                key=lambda o: o.seconds,
                default=None,
            )
            if worst is not None:
                share = 100.0 * worst.seconds / total
                lines.append(
                    f"{'':4s}  dominant tier {int(worst.tier)} "
                    f"({worst.note.split(':')[0]}) = {share:.0f}% of wall"
                )
        return "\n".join(lines)


def _run_stage(
    stage: Stage, population: list[Isomer], context: dict[str, Any]
) -> list[TierResult]:
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
        futures = {
            pool.submit(_apply, (stage.fn, iso, context)): i
            for i, iso in enumerate(population)
        }
        for fut, i in futures.items():
            try:
                results[i] = fut.result()
            except Exception as e:  # noqa: BLE001 -- incl. BrokenProcessPool
                results[i] = TierResult(
                    population[i].canonical,
                    None,
                    f"worker died: {type(e).__name__}: {e}",
                )
    return [
        r
        if r is not None
        else TierResult(population[i].canonical, None, "no result from worker")
        for i, r in enumerate(results)
    ]


def _apply(args: tuple) -> TierResult:
    """Top-level so it is picklable by ProcessPoolExecutor."""
    fn, iso, context = args
    return fn(iso, context)


def _harvest_geometry(results: list[TierResult], context: dict[str, Any]) -> None:
    """Carry a geometry a tier produced into `context` for the tiers after it.

    `tiers._embedded` reads `context["geometry"][canonical]` so that tiers 3
    and 4 score the pose an earlier tier generated instead of re-embedding from
    SMILES. Nothing wrote that key until this function existed: `tier1_dock`
    returned the docked pose in its payload, `_embedded` looked for it, and the
    two were never connected -- so every docked pose (~2 min/ligand) was
    discarded and the quantum tiers scored a free-solution conformer that had
    never seen the pocket.

    **This must run in the driver, not inside a tier.** `_run_stage` dispatches
    through a `ProcessPoolExecutor`; a tier that mutates `context` mutates a
    per-worker COPY, so the write is lost under the parallel path while
    appearing to work under the serial one. Harvesting from the returned
    `results` is the only place that holds for both.

    Only SUCCESSFUL results contribute. Caching a failed candidate's geometry
    would hand a later tier a pose from a candidate the earlier tier rejected,
    which is worse than re-embedding because it looks like it worked.
    """
    for r in results:
        if not r.ok or not r.payload:
            continue
        coords = r.payload.get("coords")
        symbols = r.payload.get("symbols")
        if coords is None or symbols is None:
            continue
        context.setdefault("geometry", {})[r.candidate_id] = {
            "symbols": symbols,
            "coords": coords,
        }


def run_funnel(
    candidates: list[Isomer], stages: list[Stage], context: dict[str, Any]
) -> FunnelReport:
    """Narrow `candidates` through `stages`, cheapest first.

    **Comparability.** A stage whose results are `formula_bound` (a total
    energy) and whose population spans more than one formula raises
    `IncomparableError` if it would CUT (keep < scored); see
    `require_same_formula`. A stage that keeps everyone passes them on in
    input order. Same-formula populations rank exactly as before.

    **Resolution.** The cut never splits candidates the tier cannot resolve
    from each other (`TierResult.resolution`); see `_cut`.

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
        _harvest_geometry(results, context)

        by_id = {r.candidate_id: r for r in results}
        ok = [
            iso
            for iso in population
            if iso.canonical in by_id and by_id[iso.canonical].ok
        ]
        bound = any(by_id[iso.canonical].formula_bound for iso in ok)
        note = ""
        if bound and len(ok) > stage.keep:
            require_same_formula(ok, stage.name)
        if bound and len({formula(iso) for iso in ok}) > 1:
            # No cut, so no ranking decision: pass everyone on in INPUT order
            # rather than in an electron-count order that would look like one.
            survivors = list(ok)
            note = "; differing formulas on a total, passed through unranked"
        else:
            survivors, note = _cut(ok, by_id, stage.keep)

        rep.outcomes.append(
            TierOutcome(
                tier=stage.tier,
                n_in=len(population),
                n_out=len(survivors),
                n_failed=len(population) - len(ok),
                note=f"{stage.name}: kept {len(survivors)} of {len(ok)} scored{note}",
                errors=[r.error for r in results if r.error][:10],
                seconds=elapsed,
            )
        )
        population = survivors

    rep.survivors = population
    return rep
