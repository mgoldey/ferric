"""Funnel bookkeeping: narrowing, failure accounting, and early stop."""
from __future__ import annotations

from tools.campaign.hierarchy import Tier
from tools.isomers.model import Isomer
from tools.pipeline.funnel import FunnelReport, Stage, run_funnel
from tools.pipeline.tiers import TierResult

P = "OC(=O)c1ccccc1"
CANDS = [Isomer(s, "substitutional", "t", P) for s in
         ["OC(=O)c1ccccc1", "OC(=O)c1ccc(F)cc1", "OC(=O)c1ccc(Cl)cc1"]]


def _fake(score_by_canonical):
    """Deterministic stub tier keyed on canonical SMILES."""
    def fn(iso, ctx):
        v = score_by_canonical.get(iso.canonical)
        return TierResult(iso.canonical, v,
                          None if v is not None else "no value")
    return fn


ALL = {c.canonical: v for c, v in zip(CANDS, (-3.0, -1.0, -2.0))}


def test_funnel_narrows_to_the_keep_count():
    stages = [Stage(Tier.FORCE_FIELD, _fake(ALL), keep=2, name="ff")]
    assert len(run_funnel(CANDS, stages, {}).survivors) == 2


def test_funnel_keeps_the_lowest_values():
    stages = [Stage(Tier.FORCE_FIELD, _fake(ALL), keep=2, name="ff")]
    kept = {i.canonical for i in run_funnel(CANDS, stages, {}).survivors}
    assert kept == {CANDS[0].canonical, CANDS[2].canonical}


def test_failed_candidates_are_dropped_and_counted_not_ranked_as_best():
    """A failure must not be a good score. If a None became 0.0 it would sort
    FIRST here, ahead of every real -1 to -3."""
    partial = {CANDS[0].canonical: -3.0, CANDS[2].canonical: -2.0}
    stages = [Stage(Tier.FORCE_FIELD, _fake(partial), keep=3, name="ff")]
    rep = run_funnel(CANDS, stages, {})
    assert len(rep.survivors) == 2
    assert rep.outcomes[0].n_failed == 1
    assert CANDS[1].canonical not in {i.canonical for i in rep.survivors}


def test_outcomes_record_the_population_at_every_stage():
    stage2 = {CANDS[0].canonical: -9.0, CANDS[2].canonical: -8.0}
    stages = [
        Stage(Tier.FORCE_FIELD, _fake(ALL), keep=2, name="ff"),
        Stage(Tier.SEMIEMPIRICAL, _fake(stage2), keep=1, name="gfn2"),
    ]
    rep = run_funnel(CANDS, stages, {})
    assert [o.n_in for o in rep.outcomes] == [3, 2]
    assert [o.n_out for o in rep.outcomes] == [2, 1]
    assert len(rep.survivors) == 1


def test_an_empty_population_stops_the_funnel_cleanly():
    """A later, expensive tier must not run on nothing."""
    stages = [
        Stage(Tier.FORCE_FIELD, _fake({}), keep=2, name="ff"),
        Stage(Tier.SEMIEMPIRICAL, _fake(ALL), keep=1, name="gfn2"),
    ]
    rep = run_funnel(CANDS, stages, {})
    assert rep.survivors == []
    assert rep.outcomes[0].n_out == 0
    assert len(rep.outcomes) == 1, "ran a later tier on an empty population"


def test_errors_are_retained_for_diagnosis():
    stages = [Stage(Tier.FORCE_FIELD, _fake({}), keep=2, name="ff")]
    rep = run_funnel(CANDS, stages, {})
    assert rep.outcomes[0].errors, "failures were counted but not explained"


def test_raw_results_are_kept_for_every_stage():
    stages = [Stage(Tier.FORCE_FIELD, _fake(ALL), keep=1, name="ff")]
    rep = run_funnel(CANDS, stages, {})
    assert len(rep.results["ff"]) == 3, "results must cover all entrants, not just survivors"
    assert rep.value("ff", CANDS[1].canonical) == -1.0


def test_keep_larger_than_the_population_is_not_an_error():
    stages = [Stage(Tier.FORCE_FIELD, _fake(ALL), keep=99, name="ff")]
    assert len(run_funnel(CANDS, stages, {}).survivors) == 3


# ── timing and parallel fan-out ──────────────────────────────────────────────

def _slow_scorer(iso, context):
    """Deterministic score, with enough sleep that parallelism is observable."""
    import time as _t
    _t.sleep(0.05)
    return TierResult(iso.canonical, float(len(iso.canonical)))


def test_every_tier_records_its_wall_time():
    """A funnel that cannot say which tier cost the run cannot be tuned."""
    cands = [Isomer(s, "sub", "t", "CO") for s in ("CO", "CCO", "CCCO")]
    rep = run_funnel(cands, [Stage(Tier.SEARCH, _slow_scorer, 2, "slow")], {})
    o = rep.outcomes[0]
    assert o.seconds is not None and o.seconds > 0.0
    # 3 candidates x 50 ms, serial
    assert o.seconds >= 0.15
    assert o.seconds_per_candidate is not None
    assert abs(o.seconds_per_candidate - o.seconds / 3) < 1e-9


def test_untimed_outcome_reports_none_not_zero():
    """0.0 would read as 'free'; None reads as 'not measured'."""
    from tools.campaign.hierarchy import TierOutcome

    o = TierOutcome(tier=Tier.SEARCH, n_in=5, n_out=2)
    assert o.seconds is None
    assert o.seconds_per_candidate is None


def test_parallel_and_serial_produce_identical_survivors():
    """The reproducibility guarantee must survive fan-out.

    `executor.map` preserves input order, so ranking sees the same sequence it
    would serially. If a future change swapped in completion-order results,
    tied scores would reorder and the survivor set could differ -- this is the
    test that catches it.
    """
    cands = [Isomer(s, "sub", "t", "CO")
             for s in ("CO", "CCO", "CCCO", "CCCCO", "CCCCCO", "CCCCCCO")]

    serial = run_funnel(cands, [Stage(Tier.SEARCH, _slow_scorer, 3, "s")], {})
    par = run_funnel(cands, [Stage(Tier.SEARCH, _slow_scorer, 3, "s", workers=3)], {})

    assert [i.canonical for i in serial.survivors] == \
           [i.canonical for i in par.survivors]
    assert [r.value for r in serial.results["s"]] == \
           [r.value for r in par.results["s"]]


def test_parallel_preserves_input_order_in_results():
    """Results must be positionally aligned with the input population.

    Ranking zips results back to candidates by canonical SMILES, but the
    recorded `results` list is also read positionally by the driver, so a
    completion-ordered list would misattribute scores.
    """
    cands = [Isomer(s, "sub", "t", "CO")
             for s in ("CCCCCCO", "CO", "CCCO", "CCCCO")]
    rep = run_funnel(cands, [Stage(Tier.SEARCH, _slow_scorer, 4, "s", workers=4)], {})
    assert [r.candidate_id for r in rep.results["s"]] == \
           [i.canonical for i in cands]


def test_workers_of_one_stays_in_process():
    """Cheap tiers must not pay process-spawn overhead."""
    cands = [Isomer("CO", "sub", "t", "CO")]
    rep = run_funnel(cands, [Stage(Tier.SEARCH, _slow_scorer, 1, "s", workers=8)], {})
    assert rep.outcomes[0].n_out == 1


def test_table_names_the_dominant_tier():
    """The report must answer the tuning question, not leave it as arithmetic.

    "Which tier cost the run" is the only question that decides where to spend
    optimization effort, and this campaign has twice been wrong about the
    answer from estimates alone.
    """
    def slow(iso, ctx):
        import time as _t
        _t.sleep(0.03)
        return TierResult(iso.canonical, float(len(iso.canonical)))

    def fast(iso, ctx):
        return TierResult(iso.canonical, float(len(iso.canonical)))

    cands = [Isomer(s, "sub", "t", "CO") for s in ("CO", "CCO", "CCCO")]
    rep = run_funnel(cands, [Stage(Tier.SEARCH, slow, 2, "dock"),
                             Stage(Tier.FORCE_FIELD, fast, 1, "mmff")], {})
    out = rep.table()
    assert "dominant tier 1" in out
    assert "s/cand" in out
    # The cheap tier must not be blamed.
    assert "dominant tier 2" not in out


def test_table_survives_untimed_outcomes():
    """A hand-built outcome with seconds=None must not crash the report."""
    from tools.campaign.hierarchy import TierOutcome

    rep = FunnelReport()
    rep.outcomes.append(TierOutcome(tier=Tier.SEARCH, n_in=3, n_out=1,
                                    note="dock: kept 1 of 3 scored"))
    out = rep.table()
    assert "dock" in out
    assert "TOTAL" not in out      # nothing was timed, so no total is claimed
