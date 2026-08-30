"""Orchestration: global dedup, filtering, and an auditable loss report."""
from __future__ import annotations

from tools.isomers import enumerate_isomers
from tools.isomers.enumerate import enumerate_with_report

BENZOIC = "OC(=O)c1ccccc1"


def test_parent_is_always_first_and_present_exactly_once():
    out = enumerate_isomers(BENZOIC)
    assert out[0].is_parent
    assert sum(1 for i in out if i.is_parent) == 1


def test_results_are_globally_deduplicated_across_generators():
    """Two generators can reach the same product; only one may survive."""
    out = enumerate_isomers(BENZOIC)
    canon = [i.canonical for i in out]
    assert len(canon) == len(set(canon))


def test_both_isomer_kinds_are_represented():
    kinds = {i.kind for i in enumerate_isomers(BENZOIC)}
    assert "substitutional" in kinds
    assert "structural" in kinds


def test_molecular_weight_filter_rejects_and_records_why():
    out, rep = enumerate_with_report(BENZOIC, mw_range=(0.0, 130.0))
    assert rep.n_after_filter <= rep.n_after_dedup
    assert any("MW" in r for r in rep.rejected), rep.rejected


def test_max_candidates_caps_the_output_deterministically():
    a, _ = enumerate_with_report(BENZOIC, max_candidates=5)
    b, _ = enumerate_with_report(BENZOIC, max_candidates=5)
    assert len(a) <= 5
    assert [i.canonical for i in a] == [i.canonical for i in b]


def test_hitting_the_cap_is_recorded_not_silent():
    _, rep = enumerate_with_report(BENZOIC, max_candidates=3)
    assert any("max_candidates" in r for r in rep.rejected), rep.rejected


def test_the_whole_enumeration_is_reproducible():
    a = [i.canonical for i in enumerate_isomers(BENZOIC)]
    b = [i.canonical for i in enumerate_isomers(BENZOIC)]
    assert a == b


def test_report_counts_are_consistent():
    _, rep = enumerate_with_report(BENZOIC)
    assert rep.n_generated >= rep.n_after_dedup >= rep.n_after_filter


def test_generators_can_be_switched_off_individually():
    full = enumerate_isomers(BENZOIC)
    subs_only = enumerate_isomers(BENZOIC, include_stereo=False,
                                  include_rings=False, include_bioisosteres=False)
    assert len(subs_only) < len(full)
    assert {i.kind for i in subs_only} <= {"parent", "substitutional"}
