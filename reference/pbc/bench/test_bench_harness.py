"""Unit tests for the issue-#228 bench harness (no ferric, no PySCF runs; numpy only for the hook).

uv run --no-sync pytest reference/pbc/bench/test_bench_harness.py -q
"""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyscf_bench as pb  # noqa: E402
import scatter_driver as sd  # noqa: E402
import summarize as sm  # noqa: E402


# ----------------------------------------------------------------------- summarize: pairing ----
def _row(
    code, rep, wall, session="s1", attempt=0, run_id="r1", box="quiet", sha="a" * 64
):
    return dict(
        code=code,
        rep=rep,
        wall=wall,
        session=session,
        attempt=attempt,
        run_id=run_id,
        box_state=box,
        bin_sha256=sha,
    )


def test_pairing_keeps_only_reps_present_for_both_codes():
    f = [_row("ferric", 0, 1.0), _row("ferric", 1, 2.0), _row("ferric", 2, 3.0)]
    p = [_row("pyscf", 1, 20.0), _row("pyscf", 2, 30.0), _row("pyscf", 3, 40.0)]
    assert sm.pair_by_rep(f, p, "wall") == [(2.0, 20.0), (3.0, 30.0)]


def test_pairing_separates_sessions_and_takes_last_attempt():
    f = [
        _row("ferric", 0, 1.0, "a"),
        _row("ferric", 0, 9.0, "b", attempt=0),
        _row("ferric", 0, 5.0, "b", attempt=1),
    ]
    p = [_row("pyscf", 0, 10.0, "a"), _row("pyscf", 0, 50.0, "b")]
    # rep 0 of session a pairs with session a only; session b uses its last (re-run) attempt.
    assert sm.pair_by_rep(f, p, "wall") == [(1.0, 10.0), (5.0, 50.0)]


def test_pairing_is_by_rep_not_by_position():
    # Same values, shuffled row order: the pairing must follow rep, not list position.
    f = [_row("ferric", 2, 3.0), _row("ferric", 0, 1.0), _row("ferric", 1, 2.0)]
    p = [_row("pyscf", 0, 10.0), _row("pyscf", 1, 20.0), _row("pyscf", 2, 30.0)]
    assert sm.pair_by_rep(f, p, "wall") == [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)]


# --------------------------------------------------------------------- summarize: bootstrap ----
def test_bootstrap_constant_ratio_has_zero_width_ci():
    # Every pair has ratio exactly 0.5 and the medians of ANY resample are paired values, so the
    # ratio of medians is 0.5 in every resample only if the resample keeps pairs together.
    pairs = [(x, 2 * x) for x in (1.0, 2.0, 3.0, 5.0, 8.0)]
    pt, lo, hi = sm.paired_bootstrap_ci(pairs, n_boot=2000, seed=1)
    assert pt == pytest.approx(0.5)
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)


def test_bootstrap_unpaired_resampling_would_widen_the_ci():
    # Mutation check of the test above: resampling the two codes INDEPENDENTLY gives a wide CI on
    # the same data, so the zero-width result really depends on the pairing.
    import random

    pairs = [(x, 2 * x) for x in (1.0, 2.0, 3.0, 5.0, 8.0)]
    rng = random.Random(1)
    a = [x for x, _ in pairs]
    b = [y for _, y in pairs]
    rs = []
    for _ in range(2000):
        sa = [a[rng.randrange(5)] for _ in range(5)]
        sb = [b[rng.randrange(5)] for _ in range(5)]
        rs.append(sm.statistics.median(sa) / sm.statistics.median(sb))
    assert max(rs) - min(rs) > 0.5


def test_bootstrap_is_reproducible():
    pairs = [(10.0 + i % 3, 20.0 + (i * 7) % 5) for i in range(8)]
    a = sm.paired_bootstrap_ci(pairs, n_boot=500, seed=228)
    b = sm.paired_bootstrap_ci(pairs, n_boot=500, seed=228)
    c = sm.paired_bootstrap_ci(pairs, n_boot=500, seed=229)
    assert a == b
    assert a[0] == c[0]  # the point estimate does not depend on the seed
    assert a[1] <= a[0] <= a[2]


def test_bootstrap_ci_covers_the_true_ratio_on_noisy_synthetic_data():
    import random

    rng = random.Random(7)
    pairs = []
    for _ in range(12):
        drift = rng.uniform(0.9, 1.1)  # shared per-rep box state: what pairing exploits
        pairs.append(
            (0.6 * drift * rng.uniform(0.98, 1.02), drift * rng.uniform(0.98, 1.02))
        )
    pt, lo, hi = sm.paired_bootstrap_ci(pairs, n_boot=4000, seed=3)
    assert lo < 0.6 < hi
    assert hi - lo < 0.1


def test_spread_stats_and_noisy_label():
    st = sm.spread_stats([10.0, 10.0, 10.0, 10.0, 10.0])
    assert st["iqr_rel"] == 0 and st["rel_mad"] == 0 and not st["noisy"]
    st = sm.spread_stats([8.0, 9.0, 10.0, 11.0, 12.0])
    assert st["median"] == 10.0
    assert st["iqr_rel"] == pytest.approx(0.2)  # inclusive quartiles 9 and 11
    assert st["rel_mad"] == pytest.approx(0.1)
    assert st["noisy"]


def test_between_sessions_cv_and_ratio():
    bs = sm.between_sessions(
        {"s1": [10.0, 11.0, 12.0], "s2": [12.0, 12.0], "s3": [14.0]}
    )
    assert bs["medians"] == {"s1": 11.0, "s2": 12.0, "s3": 14.0}
    assert bs["max_over_min"] == pytest.approx(14.0 / 11.0)
    assert bs["cv"] == pytest.approx(
        sm.statistics.stdev([11.0, 12.0, 14.0]) / (37.0 / 3)
    )
    f = [_row("ferric", 0, 6.0, "s1"), _row("ferric", 0, 7.0, "s2")]
    p = [_row("pyscf", 0, 10.0, "s1"), _row("pyscf", 0, 10.0, "s2")]
    assert sm.session_ratios(f, p, "wall") == {"s1": 0.6, "s2": 0.7}


# ------------------------------------------------------------------------ scatter_driver ----
def test_private_binary_refuses_target_paths(tmp_path):
    t = tmp_path / "target" / "release"
    t.mkdir(parents=True)
    b = t / "ferric"
    b.write_text("#!/bin/sh\n")
    b.chmod(0o755)
    with pytest.raises(SystemExit, match="target/"):
        sd.check_private_binary(str(b))
    link = tmp_path / "ferric-link"
    link.symlink_to(b)
    with pytest.raises(SystemExit, match="target/"):
        sd.check_private_binary(str(link))
    priv = tmp_path / "ferric-private"
    priv.write_bytes(b.read_bytes())
    priv.chmod(0o755)
    assert sd.check_private_binary(str(priv)) == os.path.realpath(priv)
    # a path that merely CONTAINS "target" in a name is fine
    other = tmp_path / "targeted"
    other.mkdir()
    ok = other / "ferric"
    ok.write_bytes(b"x")
    ok.chmod(0o755)
    assert sd.check_private_binary(str(ok))
    assert len(sd.sha256_file(str(ok))) == 64


def test_default_cpus_and_count():
    assert sd.default_cpus(1) == "1"  # core 0 takes most IRQs
    assert sd.default_cpus(6) == "0-5"
    assert sd.default_cpus(3) == "0-2"
    with pytest.raises(SystemExit):
        sd.default_cpus(12)
    assert sd.cpu_list_count("0-5") == 6
    assert sd.cpu_list_count("0,2,4-5") == 4


def test_rotated_order_cycles_every_position():
    seq = ["F1", "F2", "P"]
    orders = [sd.rotated_order(seq, r) for r in range(3)]
    assert orders == [["F1", "F2", "P"], ["F2", "P", "F1"], ["P", "F1", "F2"]]
    # two entries reproduce the old F P / P F alternation
    assert sd.rotated_order(["F", "P"], 0) == ["F", "P"]
    assert sd.rotated_order(["F", "P"], 1) == ["P", "F"]
    assert sd.rotated_order(seq, 4, fixed=True) == seq


def test_box_state_label():
    assert sd.box_state(1.0, 0.01) == "quiet"
    assert sd.box_state(1.6, 0.01) == "ambient"
    assert sd.box_state(1.0, 0.03) == "ambient"


def test_snapshots_read_sysfs_layout(tmp_path):
    for i, khz in ((0, 3600000), (11, 1200000)):
        d = tmp_path / f"cpu{i}" / "cpufreq"
        d.mkdir(parents=True)
        (d / "scaling_cur_freq").write_text(f"{khz}\n")
    f = sd.snapshot_freq_khz(str(tmp_path / "cpu*" / "cpufreq" / "scaling_cur_freq"))
    assert f == {"0": 3600000, "11": 1200000}
    h0 = tmp_path / "hwmon" / "hwmon0"
    h0.mkdir(parents=True)
    (h0 / "name").write_text("coretemp\n")
    (h0 / "temp1_input").write_text("45000\n")
    (h0 / "temp1_label").write_text("Package id 0\n")
    h1 = tmp_path / "hwmon" / "hwmon1"
    h1.mkdir()
    (h1 / "name").write_text("nct6792\n")
    (h1 / "temp1_input").write_text("30000\n")
    assert sd.snapshot_temp_mc(str(tmp_path / "hwmon")) == {
        "coretemp/Package id 0": 45000
    }


# ----------------------------------------------------- pyscf_bench: ferric stopping rule ----
np = pytest.importorskip("numpy")


def _sym(rng, n, scale):
    a = rng.standard_normal((n, n)) * scale
    return a + a.T


def test_density_change_is_rms_over_all_elements_and_abs_max():
    d0 = np.zeros((3, 3))
    d1 = np.zeros((3, 3))
    d1[0, 1] = d1[1, 0] = 3e-10
    d1[2, 2] = -6e-10
    rms, mx = pb.ferric_density_change(d1, d0)
    assert rms == pytest.approx(math.sqrt((2 * 9e-20 + 36e-20) / 9))
    assert mx == pytest.approx(6e-10)


def test_gamma_rule_thresholds():
    rng = np.random.default_rng(0)
    n = 4
    s = np.eye(n)
    f = np.diag([-1.0, -0.5, 0.5, 1.0])
    d = np.diag([2.0, 2.0, 0.0, 0.0])  # commutes with f: zero gradient
    tiny = _sym(rng, n, 1e-12)
    assert pb.ferric_gamma_signals(-1.0, -1.0 + 1e-6, d + tiny, d, f, s)["converged"]
    # rms just above 1e-10 -> not converged
    big = np.full((n, n), 1.01e-10)
    assert not pb.ferric_gamma_signals(-1.0, -1.0, d + big, d, f, s)["converged"]
    # one element at 9e-10 (< 10*density_conv) with rms 9e-10/4 > 1e-10: refused by rms
    one = np.zeros((n, n))
    one[0, 0] = 9e-10
    assert not pb.ferric_gamma_signals(-1.0, -1.0, d + one, d, f, s)["converged"]
    # |dE| >= 1e-3 refuses even a frozen density
    assert not pb.ferric_gamma_signals(-1.0, -1.002, d, d, f, s)["converged"]
    # non-stationary frozen density (gradient 0.2 > 1e-3): the veto refuses
    f2 = f.copy()
    f2[0, 2] = f2[2, 0] = 0.1
    sig = pb.ferric_gamma_signals(-1.0, -1.0, d, d, f2, s)
    assert not sig["converged"] and sig["grad_max"] == pytest.approx(0.2)


def test_gamma_rule_dp_max_companion_is_ten_times_density_conv():
    # rms 2e-9/30 = 6.7e-11 passes, but the single element 2e-9 >= 10 * 1e-10: refused.
    n = 30
    s, f = np.eye(n), np.diag(np.arange(n, dtype=float))
    d = np.diag([2.0] * 5 + [0.0] * (n - 5))
    one = np.zeros((n, n))
    one[7, 7] = 2e-9
    sig = pb.ferric_gamma_signals(-1.0, -1.0, d + one, d, f, s)
    assert sig["dp_rms"] < 1e-10 and not sig["converged"]
    one[7, 7] = 0.9e-9
    assert pb.ferric_gamma_signals(-1.0, -1.0, d + one, d, f, s)["converged"]


def test_k_rule_uses_max_over_k_and_strict_bounds():
    s = [np.eye(2), np.eye(2)]
    d = [np.diag([2.0, 0.0])] * 2
    f_ok = [np.diag([-1.0, 1.0])] * 2
    f_bad = [np.diag([-1.0, 1.0]), np.array([[-1.0, 1e-9j], [-1e-9j, 1.0]])]
    assert pb.ferric_k_signals(-1.0, -1.0 + 5e-13, d, f_ok, s)["converged"]
    assert not pb.ferric_k_signals(-1.0, -1.0 + 2e-12, d, f_ok, s)["converged"]
    sig = pb.ferric_k_signals(-1.0, -1.0, d, f_bad, s)
    assert sig["grad_max"] == pytest.approx(2e-9) and not sig["converged"]


def test_orthogonalizer_drops_modes_below_absolute_cut():
    s = np.diag([1.0, 5e-7, 2.0])
    x = pb.ferric_canonical_orthogonalizer(s)
    assert x.shape == (3, 2)
    assert np.allclose(x.T @ s @ x, np.eye(2))


# ------------------------------------------------------ review fixes (PR #560): quoting ----
def _session(code, session, walls, box="quiet", sha="a" * 64):
    return [_row(code, i, w, session, box=box, sha=sha) for i, w in enumerate(walls)]


QUIET5 = [10.0, 10.1, 9.9, 10.05, 9.95]


def test_quotable_reachable_on_clean_quiet_data():
    f = _session("ferric", "s1", [0.6 * x for x in QUIET5]) + _session(
        "ferric", "s2", [0.61 * x for x in QUIET5]
    )
    p = _session("pyscf", "s1", QUIET5) + _session("pyscf", "s2", QUIET5)
    ok, why = sm.quotable(f, p, "wall", 6)
    assert ok, why


def test_ambient_rows_are_never_quotable():
    f = _session("ferric", "s1", [0.6 * x for x in QUIET5]) + _session(
        "ferric", "s2", [0.6 * x for x in QUIET5], box="ambient"
    )
    p = _session("pyscf", "s1", QUIET5) + _session("pyscf", "s2", QUIET5)
    ok, why = sm.quotable(f, p, "wall", 6)
    assert not ok and any("non-quiet" in w for w in why)
    # rows written before box_state existed ("unlabelled") are not quiet either
    for r in f:
        r["box_state"] = "unlabelled"
    assert not sm.quotable(f, p, "wall", 6)[0]


def test_noisy_session_group_is_never_quotable():
    noisy = [6.0, 4.0, 8.0, 5.0, 7.0]  # same median as 6.0, IQR/median 33%
    f = _session("ferric", "s1", noisy) + _session("ferric", "s2", [6.0] * 5)
    p = _session("pyscf", "s1", [10.0] * 5) + _session("pyscf", "s2", [10.0] * 5)
    ok, why = sm.quotable(f, p, "wall", 6)
    assert not ok and any("NOISY" in w for w in why)


def test_too_few_reps_is_labelled_and_not_quotable():
    st = sm.spread_stats([10.0, 30.0, 50.0, 70.0], min_n=5)
    assert st["label"] == "n too small" and st["noisy"] is None
    assert sm.spread_stats([10.0] * 3, min_n=3)["label"] == "ok"
    f = _session("ferric", "s1", [6.0] * 4) + _session("ferric", "s2", [6.0] * 4)
    p = _session("pyscf", "s1", [10.0] * 4) + _session("pyscf", "s2", [10.0] * 4)
    ok, why = sm.quotable(f, p, "wall", 6)  # N=6 needs 5 reps
    assert not ok and any("n too small" in w for w in why)
    assert sm.quotable(f, p, "wall", 1)[0]  # N=1 needs 3


def test_different_binaries_are_never_pooled_or_compared():
    rows = _session("ferric", "s1", [6.0] * 5, sha="a" * 64) + _session(
        "pyscf", "s1", [10.0] * 5, sha="b" * 64
    )
    by_input = {}
    for r in rows:
        r.update(cell="c", threads=6, method="rhf", toml="c_rhf.toml")
        k = (r["cell"], r["threads"], r["code"], sm.input_label(r), r["bin_sha256"])
        by_input.setdefault(k, []).append(r)
    assert sm.comparable_pairs(by_input) == []
    for r in by_input[next(k for k in by_input if k[2] == "pyscf")]:
        r["bin_sha256"] = "a" * 64
    by2 = {}
    for r in rows:
        k = (r["cell"], r["threads"], r["code"], sm.input_label(r), r["bin_sha256"])
        by2.setdefault(k, []).append(r)
    assert len(sm.comparable_pairs(by2)) == 1


def test_pairing_separates_runs_within_one_session():
    # a top-up run in the same --out restarts rep at 0: both rep-0s must survive, unmixed
    f = [_row("ferric", 0, 1.0, run_id="r1"), _row("ferric", 0, 2.0, run_id="r2")]
    p = [_row("pyscf", 0, 10.0, run_id="r1"), _row("pyscf", 0, 20.0, run_id="r2")]
    assert sm.pair_by_rep(f, p, "wall") == [(1.0, 10.0), (2.0, 20.0)]


# ------------------------------------------------------ review fixes: driver guards ----
def test_binary_change_mid_session_exits(tmp_path):
    b = tmp_path / "ferric"
    b.write_bytes(b"v1")
    sha = sd.sha256_file(str(b))
    assert sd.check_binary_unchanged(str(b), sha) == sha
    b.write_bytes(b"v2")
    with pytest.raises(SystemExit, match="changed during the session"):
        sd.check_binary_unchanged(str(b), sha)


def test_hardlinked_binary_is_refused(tmp_path):
    b = tmp_path / "ferric"
    b.write_bytes(b"x")
    b.chmod(0o755)
    assert sd.check_private_binary(str(b))
    os.link(b, tmp_path / "ferric-link")
    with pytest.raises(SystemExit, match="hard links"):
        sd.check_private_binary(str(b))


def test_existing_out_needs_append(tmp_path):
    out = tmp_path / "t.jsonl"
    sd.check_out_path(str(out), False)  # new file: fine
    out.write_text("")
    with pytest.raises(SystemExit, match="--append"):
        sd.check_out_path(str(out), False)
    sd.check_out_path(str(out), True)
    assert sd.new_run_id() != sd.new_run_id()


def test_lindep_override_is_refused():
    sd.check_no_lindep_override({})
    with pytest.raises(SystemExit, match="FERRIC_LINDEP_THRESH"):
        sd.check_no_lindep_override({"FERRIC_LINDEP_THRESH": "1e-5"})


def test_max_cycle_default_follows_the_rule():
    assert pb.resolve_max_cycle(None, True) == 200
    assert pb.resolve_max_cycle(None, False) == 100
    assert pb.resolve_max_cycle(50, True) == 50


# ------------------------------------------------------------ review fixes: ladder gate ----
def _ladder(e75, e200, e250):
    return {"75x302": e75, "200x590": e200, "250x590": e250}


def test_ladder_gate_pass_fail_undecidable():
    # pass: cross-code 75x302 gap 2e-6 <= 3e-6 + 3e-6 + 1e-8
    r = sm.ladder_gate(
        _ladder(-1.0, -1.000003, -1.000003), _ladder(-1.000002, -1.000005, -1.000005)
    )
    assert r["status"] == "pass"
    # fail: both codes move little but disagree by 1e-5 at 75x302
    r = sm.ladder_gate(
        _ladder(-1.0, -1.0000001, -1.0000001), _ladder(-1.00001, -1.0000101, -1.0000101)
    )
    assert r["status"] == "fail" and r["lhs"] > r["rhs"]
    # undecidable: the ref moves 2e-6 from 200x590 for one code
    r = sm.ladder_gate(_ladder(-1.0, -1.0, -1.000002), _ladder(-1.0, -1.0, -1.0))
    assert r["status"] == "undecidable"
    assert sm.ladder_gate({}, _ladder(-1, -1, -1))["status"] == "undecidable"


def test_grid_label_parses_both_codes():
    assert (
        sm.grid_label(dict(code="ferric", toml="x/dryice_pbe.toml"), "dryice")
        == "75x302"
    )
    assert (
        sm.grid_label(dict(code="ferric", toml="dryice_pbe_250x590.toml"), "dryice")
        == "250x590"
    )
    assert (
        sm.grid_label(dict(code="ferric", toml="dryice_pbe_w0.5.toml"), "dryice")
        is None
    )
    assert sm.grid_label(dict(code="pyscf", grid=[150, 434]), "dryice") == "150x434"


# ----------------------------------------------------------- review nit: cached X in hook ----
def test_hook_builds_the_orthogonaliser_once_and_matches_uncached(monkeypatch):
    calls = []
    real = pb.ferric_canonical_orthogonalizer

    def counting(s, lindep=pb.FERRIC_LINDEP):
        calls.append(1)
        return real(s, lindep)

    monkeypatch.setattr(pb, "ferric_canonical_orthogonalizer", counting)
    s = [np.eye(2), np.array([[1.0, 0.1], [0.1, 1.0]])]
    d = [np.diag([2.0, 0.0])] * 2
    f = [np.array([[-1.0, 1e-9], [1e-9, 1.0]])] * 2
    trace = []
    check = pb.make_ferric_conv_check(True, trace)
    for cyc in range(3):
        check(dict(e_tot=-1.0, last_hf_e=-1.0, dm=d, fock=f, s1e=s, cycle=cyc))
    assert len(calls) == 2  # once per k, not per cycle
    calls.clear()
    ref = pb.ferric_k_signals(-1.0, -1.0, d, f, s)
    assert trace[-1]["grad_max"] == pytest.approx(ref["grad_max"])


def test_between_session_range_of_5_percent_blocks_quoting():
    f = _session("ferric", "s1", [0.6 * x for x in QUIET5]) + _session(
        "ferric", "s2", [0.66 * x for x in QUIET5]
    )
    p = _session("pyscf", "s1", QUIET5) + _session("pyscf", "s2", QUIET5)
    ok, why = sm.quotable(f, p, "wall", 6)
    assert not ok and why == ["between-session range >= 5%"]


# ------------------------------------------------------------- overnight_228 / analyze_228 ----
import analyze_228 as an  # noqa: E402
import overnight_228 as ov  # noqa: E402


class _Th:
    max_load, max_busy, max_foreign, max_iowait, max_steal = 1.5, 0.02, 0.02, 0.02, 0.01
    max_python_cpu, max_proc_cpu, check_steal_iowait = 0.10, 0.25, True
    window, poll = 0.0, 0.0


def _reading(**kw):
    r = dict(load1=0.3, load1_adj=0.3, busy_frac=0.005, busy_phys=0.01, iowait_frac=0.0)
    r["steal_frac"] = 0.0
    r.update(dict(heavy=[]), **kw)
    return r


def test_own_load_is_the_kernels_exponential_average():
    assert ov.own_load([], 100.0) == 0.0
    # 6 busy threads for 10 min -> ~6 right at the end, decaying with a 60 s constant after.
    assert ov.own_load([(100.0, 6.0, 600.0)], 100.0) == pytest.approx(6.0, abs=1e-3)
    assert ov.own_load([(100.0, 6.0, 600.0)], 160.0) == pytest.approx(
        6.0 / math.e, rel=1e-3
    )


def test_judge_names_every_failing_reading():
    assert ov.judge(_reading(), _Th) == []
    why = ov.judge(
        _reading(load1=2.0, load1_adj=1.6, heavy=[dict(pid=1, comm="rustc", cpu=0.9)]),
        _Th,
    )
    assert len(why) == 2 and why[1].startswith("proc rustc")


def test_raw_load_mode_is_the_default_and_counts_our_own_tail():
    r = _reading(load1=1.6, load1_adj=0.4, own_load=1.2)
    assert (
        ov.judge(r, _Th)
        and ov.judge(r, type("T", (_Th,), dict(load_mode="adjusted"))) == []
    )
    assert ov.build_parser().parse_args(["--run-dir", "x"]).load_mode == "raw"
    assert ov.decay_wait(1, 600.0) == 0.0  # one thread never reaches 1.2
    assert ov.decay_wait(6, 600.0) == pytest.approx(
        60 * math.log(0.92 * 6 / 1.2), rel=1e-3
    )


def _fake_proc(monkeypatch, procs):
    """procs: {pid: (comm, cpu cores)} over a 1 s window; the box otherwise idle."""
    snaps = iter([0, 1])
    hz = os.sysconf("SC_CLK_TCK")

    def proc_times(own=()):
        k = next(snaps)
        return {p: (c, int(k * cpu * hz)) for p, (c, cpu) in procs.items()}

    t = iter([0.0, 1.0, 1.0])
    monkeypatch.setattr(ov, "proc_times", proc_times)
    monkeypatch.setattr(ov.time, "time", lambda: next(t))
    monkeypatch.setattr(ov.time, "sleep", lambda s: None)
    stats = iter(
        [dict(total=0, busy=0, iowait=0, steal=0)] * 1
        + [dict(total=1200, busy=1, iowait=0, steal=0)]
    )
    monkeypatch.setattr(ov, "read_stat", lambda: next(stats))


def test_gate_ignores_a_cargo_blocked_on_the_lock_but_not_a_compiler(monkeypatch):
    _fake_proc(monkeypatch, {10: ("cargo", 0.0), 11: ("bash", 0.01)})
    assert ov.Gate(_Th, 1e18).read()["heavy"] == []
    _fake_proc(monkeypatch, {10: ("cargo", 0.0), 12: ("cc1plus", 0.0)})
    assert [p["comm"] for p in ov.Gate(_Th, 1e18).read()["heavy"]] == ["cc1plus"]
    _fake_proc(monkeypatch, {13: ("python3", 0.2)})
    assert [p["comm"] for p in ov.Gate(_Th, 1e18).read()["heavy"]] == ["python3"]


def test_gate_after_discards_on_foreign_cpu_and_before_raises_at_the_deadline(
    monkeypatch,
):
    g = ov.Gate(_Th, deadline=0.0)
    monkeypatch.setattr(
        g, "read", lambda phase="before": dict(_reading(), reasons=[], ok=True)
    )
    monkeypatch.setattr(
        ov, "read_stat", lambda: dict(total=1, busy=0, iowait=0, steal=0)
    )
    assert g.gate("before")["ok"]
    after = g.gate("after", wall=10.0, cpu_s=10.0, foreign=0.03, threads=1)
    assert not after["ok"] and "foreign CPU" in after["reasons"][0]
    assert g.history and g.history[-1][1] == pytest.approx(1.0)
    monkeypatch.setattr(
        g, "read", lambda phase="before": dict(_reading(), reasons=["busy"], ok=False)
    )
    with pytest.raises(ov.Deadline):
        g.gate("before")
    g.enforce = False  # smoke / anchors: logged, never waited on
    assert g.gate("before")["enforced"] is False


def test_walk_is_rep_major_and_rotates_inside_a_rep():
    c = ov.Cell("x", "x", "rhf", {6: ("", "_w0.5")}, {6: 2}, "core", {6: (1.0, 1.0)})
    d = ov.Cell("y", "y", "rhf", {1: ("",)}, {1: 1}, "core", {1: (1.0, 1.0)})
    got = [(cc.id, code, tag, rep) for cc, code, tag, n, rep in ov.walk([c, d], None)]
    assert got == [
        ("x", "ferric", "", 0), ("x", "ferric", "_w0.5", 0), ("x", "pyscf", "", 0),
        ("y", "ferric", "", 0), ("y", "pyscf", "", 0),
        ("x", "ferric", "_w0.5", 1), ("x", "pyscf", "", 1), ("x", "ferric", "", 1),
    ]  # fmt: skip


def test_execute_resumes_and_retries_only_discarded_keys(monkeypatch, tmp_path):
    c = ov.Cell("x", "x", "rhf", {6: ("",)}, {6: 2}, "core", {6: (1.0, 1.0)})
    calls, verdicts = (
        [],
        iter([True, False, True, True, True]),
    )  # 2nd sample fails its gate once

    def fake(cell, code, tag, threads, rep, attempt, ctx):
        calls.append((code, rep, attempt))
        if len(calls) == 3 and ctx.get("die"):
            raise KeyboardInterrupt  # the box reboots mid-run
        ok = next(verdicts)
        return dict(plan_id=cell.id, code=code, variant=tag, threads=threads, rep=rep,
                    attempt=attempt, exit=0, energy=-1.0, converged=True, gated=ok,
                    gate_reasons=[], wall=1.0, kind="timed")  # fmt: skip

    monkeypatch.setattr(ov, "run_sample", fake)
    args = type("A", (), dict(max_attempts=3))
    out = str(tmp_path / "results.jsonl")
    gate = ov.Gate(_Th, 1e18)
    with pytest.raises(KeyboardInterrupt):
        ov.execute([c], dict(args=args, smoke=False, gate=gate, die=True), out)
    assert len(ov.load_rows(out)) == 2  # rep 0 ferric gated, rep 0 pyscf discarded
    calls.clear()
    assert ov.execute([c], dict(args=args, smoke=False, gate=gate), out) == []
    # resume: rep-0 ferric is done; rep-0 pyscf is retried (attempt 1); rep 1 rotates to P F.
    assert calls == [("pyscf", 0, 1), ("pyscf", 1, 0), ("ferric", 1, 0)]


def test_execute_reports_keys_that_never_got_a_gated_sample(monkeypatch, tmp_path):
    c = ov.Cell("x", "x", "rhf", {6: ("",)}, {6: 1}, "core", {6: (1.0, 1.0)})

    def fake(cell, code, tag, threads, rep, attempt, ctx):
        return dict(plan_id=cell.id, code=code, variant=tag, threads=threads, rep=rep,
                    attempt=attempt, exit=0, energy=-1.0, converged=True,
                    gated=code == "ferric", gate_reasons=["busy"], wall=1.0, kind="timed")  # fmt: skip

    monkeypatch.setattr(ov, "run_sample", fake)
    args = type("A", (), dict(max_attempts=2))
    out = str(tmp_path / "r.jsonl")
    ctx = dict(args=args, smoke=False, gate=ov.Gate(_Th, 1e18))
    assert ov.execute([c], ctx, out) == [("x", "pyscf", "", 6, 0)]
    assert len(ov.load_rows(out)) == 3  # ferric once, pyscf twice
    assert ov.execute([c], ctx, out) == [
        ("x", "pyscf", "", 6, 0)
    ]  # resume: still exhausted


# ------------------------------------------------ whole sample, before -> after, fake box ----
class _Box:
    """A fake box: clock, /proc/stat jiffies, loadavg (the kernel's 60 s exponential average of
    the runnable count, updated continuously) and our children's rusage."""

    HZ = os.sysconf("SC_CLK_TCK")

    def __init__(self):
        self.t, self.busy, self.total, self.load, self.own_cpu = 1e6, 0, 0, 0.1, 0.0
        self.events, self.held, self._locked_at, self._ran = [], [], None, 0.0

    def advance(self, dt, own=0.0, foreign=0.0, ambient=0.05):
        steps = max(1, int(dt))
        for _ in range(steps):
            h = dt / steps
            k = math.exp(-h / 60)
            running = (6.0 if own else 0.0) + foreign + ambient
            self.load = self.load * k + running * (1 - k)
            self.busy += (own + foreign + ambient) * self.HZ * h
            self.total += ov.LOGICAL * self.HZ * h
            self.t += h

    def install(self, monkeypatch, wall, foreign_mid_run=0.0):
        import types

        box = self
        fake_time = types.SimpleNamespace(
            time=lambda: box.t,
            sleep=lambda s: box.advance(s),
            strftime=__import__("time").strftime,
            localtime=__import__("time").localtime,
        )
        monkeypatch.setattr(ov, "time", fake_time)
        monkeypatch.setattr(sd, "time", fake_time)
        stat = lambda: dict(total=int(box.total), busy=int(box.busy), iowait=0, steal=0)  # noqa: E731
        monkeypatch.setattr(ov, "read_stat", stat)
        monkeypatch.setattr(sd, "cpu_jiffies", lambda: (int(box.busy), int(box.total)))
        monkeypatch.setattr(ov, "read_loadavg", lambda: (box.load, box.load))
        monkeypatch.setattr(sd, "load1", lambda: box.load)
        monkeypatch.setattr(ov, "proc_times", lambda own=(): {7: ("bash", 0)})
        ru = lambda who: types.SimpleNamespace(ru_utime=box.own_cpu, ru_stime=0.0)  # noqa: E731
        monkeypatch.setattr(sd.resource, "getrusage", ru)

        def child(cmd, env):
            box.events.append("run")
            box._ran += wall
            box.advance(wall, own=5.5, foreign=foreign_mid_run)
            box.own_cpu += 5.5 * wall
            return 0, "", "", 0

        monkeypatch.setattr(sd, "run_child", child)

        def lock():
            box.events.append("lock")
            box._locked_at, box._ran = box.t, 0.0
            return [], 0.0

        def unlock(fds):
            box.events.append("unlock")
            box.held.append(
                box.t - box._locked_at - box._ran
            )  # lock time NOT spent running

        monkeypatch.setattr(sd, "acquire_locks", lock)
        monkeypatch.setattr(sd, "release_locks", unlock)


class _ThReal(_Th):
    window, poll, load_mode = 5.0, 60.0, "raw"


def _one_sample(gate):
    ctl = ov.Ctl([], gate)
    _, g = sd.run_cmd(["ferric", "x.toml"], {}, 6, "0-5", False, ctl)
    return g


@pytest.mark.parametrize("wall", [15.0, 60.0, 300.0])
def test_quiet_n6_sample_passes_both_ends_whatever_its_own_load(monkeypatch, wall):
    box = _Box()
    box.install(monkeypatch, wall)
    gate = ov.Gate(_ThReal, box.t + 3600)
    for _ in range(2):  # back to back: the 2nd waits for the 1st one's own load tail
        g = _one_sample(gate)
        assert g["gate_before"]["ok"] and g["gate_after"]["ok"], g["gate_after"][
            "reasons"
        ]
        assert (
            g["gate_after"]["load1"] > 1.5 or wall < 20
        )  # the own load really is there
    # the wait for quiet (the 2nd sample waits ~1.5 min for the 1st's load tail at 60/300 s)
    # happened with NO lock held: under the locks only the recheck + after readings (2 x 5 s)
    assert box.events == ["lock", "run", "unlock"] * 2
    assert max(box.held) <= 2 * _ThReal.window + 1e-6, box.held


def test_n6_sample_with_a_foreign_job_mid_run_is_discarded(monkeypatch):
    box = _Box()
    box.install(monkeypatch, 60.0, foreign_mid_run=0.5)
    g = _one_sample(ov.Gate(_ThReal, box.t + 3600))
    assert g["gate_before"]["ok"] and not g["gate_after"]["ok"]
    assert g["gate_after"]["reasons"][0].startswith("foreign CPU over the run")
    assert g["gate_after"]["run_foreign_phys"] == pytest.approx(
        (0.5 + 0.05) / ov.PHYS, rel=0.02
    )  # job + ambient


def test_failed_recheck_releases_the_locks_and_waits_again(monkeypatch):
    box = _Box()
    box.install(monkeypatch, 15.0)
    gate = ov.Gate(_ThReal, box.t + 3600)
    real = gate.read
    n = {"recheck": 0}

    def read(phase="before"):
        r = real(phase)
        if (
            phase == "recheck" and n["recheck"] == 0
        ):  # a build starts between wait and lock
            n["recheck"] += 1
            r["reasons"], r["ok"] = ["proc rustc[9] 90%"], False
        return r

    monkeypatch.setattr(gate, "read", read)
    g = _one_sample(gate)
    assert g["gate_before"]["ok"] and box.events == [
        "lock",
        "unlock",
        "lock",
        "run",
        "unlock",
    ]


# -------------------------------------------------------------- jsonl, provenance, session ----
def test_truncated_last_line_is_skipped_and_the_next_append_starts_fresh(
    tmp_path, capsys
):
    out = str(tmp_path / "r.jsonl")
    ov.append_row(out, dict(code="pyscf", a=1))
    with open(out, "a") as f:
        f.write('{"a": 2, "b"')  # killed mid-write
    assert ov.load_rows(out) == [dict(code="pyscf", a=1)]
    assert "truncated last line" in capsys.readouterr().err
    ov.append_row(out, dict(code="pyscf", a=3))
    assert [r["a"] for r in ov.load_rows(out)] == [1, 3]
    assert [r["a"] for r in sm.load_rows([out])] == [1, 3]
    assert open(out + ".truncated").read() == '{"a": 2, "b"\n'  # kept, not parsed
    with open(out, "a") as f:
        f.write("not json\n")
    ov.append_row(out, dict(a=4))
    with pytest.raises(
        ValueError
    ):  # a bad line in the MIDDLE is an error, never skipped
        sm.read_jsonl(out)


def test_provenance_refuses_an_extension_without_build_info(monkeypatch, tmp_path):
    so = tmp_path / "libferric.so"
    so.write_bytes(b"x")
    c = "a" * 40
    v = dict(text="", commit=c, dirty=False, ok=True)
    monkeypatch.setattr(ov, "ferric_version", lambda b: v)
    monkeypatch.setattr(ov, "contains", lambda commit: True)
    for build, bad in (
        (None, "no __build__"),
        (dict(git_sha="unknown", dirty=False), "commit unknown"),
        (dict(git_sha=c, dirty=None), "dirty flag"),
        (dict(git_sha="b" * 40, dirty=False), "!= extension commit"),
    ):
        monkeypatch.setattr(
            ov, "so_build", lambda so, py, b=build: dict(build=b, version="0")
        )
        _, problems = ov.provenance("ferric", str(so), "python")
        assert any(bad in p for p in problems), (build, problems)
    monkeypatch.setattr(
        ov, "so_build", lambda so, py: dict(build=dict(git_sha=c, dirty=False))
    )
    assert ov.provenance("ferric", str(so), "python")[1] == []


def test_session_id_is_unique_per_run_dir_path():
    a, b = ov.session_id("/x/night1"), ov.session_id("/y/night1")
    assert a != b and a.startswith("night1-") and a == ov.session_id("/x/night1")


def test_plan_inputs_parse_and_pyscf_argv_is_well_formed():
    problems, cmds = ov.check_inputs(ov.PLAN + [ov.SMOKE], 1e-13)
    assert problems == []
    assert sum(1 for c in cmds if c[2] == "pyscf") >= len(ov.PLAN)


def _arow(code, rep, wall, var="", gated=True, session="s"):
    return dict(plan_id="p", threads=6, code=code, variant=var, rep=rep, wall=wall,
                gated=gated, valid=gated, exit=0, kind="timed", session=session,
                run_id=session, bin_sha256="a" * 64, energy=-1.0, method="rhf")  # fmt: skip


def test_analyze_flags_spread_over_the_bar_and_pairs_reps():
    f = [_arow("ferric", r, w) for r, w in enumerate([10.0, 10.1, 10.05])]
    p = [_arow("pyscf", r, w) for r, w in enumerate([20.0, 21.0, 20.0])]
    p.append(_arow("pyscf", 3, 99.0, gated=False))  # discarded: never in a statistic
    sp, ratios, energies = an.analyze(f + p, n_boot=200)
    flags = {d["key"][2]: (d["flag"], d["n"], d["discarded"]) for d in sp}
    assert flags == {"ferric": ("ok", 3, 0), "pyscf": ("SPREAD>2%", 3, 1)}
    assert ratios[0]["n_pairs"] == 3 and ratios[0]["ratio"] == pytest.approx(
        10.05 / 20.0
    )
    assert ratios[0]["flag"].startswith("ferric ok / pyscf SPREAD")
    assert energies[0]["flag"] == "ok"
    sp, _, _ = an.analyze(f[:2] + p, n_boot=10)
    assert {d["key"][2]: d["flag"] for d in sp}["ferric"] == "n<3"
