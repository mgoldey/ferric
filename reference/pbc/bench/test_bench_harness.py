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
def _row(code, rep, wall, session="s1", attempt=0):
    return dict(code=code, rep=rep, wall=wall, session=session, attempt=attempt)


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
    assert sd.default_cpus(1) == "0"
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
