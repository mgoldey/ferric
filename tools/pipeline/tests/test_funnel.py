"""Funnel bookkeeping: narrowing, failure accounting, and early stop."""
from __future__ import annotations

from tools.campaign.hierarchy import Tier
from tools.isomers.model import Isomer
from tools.pipeline.funnel import Stage, run_funnel
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
