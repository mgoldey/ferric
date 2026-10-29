"""Analysis of overnight_228.py results: gated samples only.

  python analyze_228.py RUN_DIR/results.jsonl [RUN_DIR2/results.jsonl ...]
                        [--metrics wall t_setup_s t_scf_s s_per_iter] [--bar 0.02]
                        [--boot 10000] [--seed 228]

Sections 1 and 2 are printed once PER METRIC, and every SPREAD flag and ratio flag is computed on
the metric of the block it sits in.  Read them per metric:
  wall        harness wall around the whole process: includes the `systemd-run --scope` start
              (scripts/ferric-limited, ~0.1-0.3 s) and, for PySCF, the python/pyscf import
              (~1.5 s).  Both are constant-ish offsets that dilute ratios and can by themselves
              approach the 2% bar on ~15 s cells.
  t_setup_s   ferric: every stage except SCF; PySCF: df build + hcore.
  t_scf_s     ferric: SCF stages; PySCF: kernel minus hcore.  Neither contains process start.
  s_per_iter  DIFFERENT definitions (ferric SCF / iterations; PySCF median callback gap), see
              README.md; compare within a code, not across.

1. Per (plan id, N, code, variant, binary): n gated / n discarded, median, range/median and
   IQR/median.  SPREAD flag iff range/median > --bar (default 2%, the pre-registered bar; the
   range is the conservative spread at n = 3-5, IQR is printed next to it); "n<3" when fewer than
   3 gated samples (then no spread is claimed).
2. ferric/PySCF ratio of medians over PAIRED reps (summarize.pair_by_rep: same session, run_id and
   rep), per ferric variant, same plan id, N and binary sha256, with summarize's paired bootstrap
   95% CI.  A ratio whose either side is flagged carries the flag.
3. Energy check: |median E_ferric - median E_PySCF| per (plan id, N, variant), flagged > 1e-8
   Ha/cell for RHF (the sessions' gate; PBE integrates on different grids, see README).
4. With several run dirs (sessions) of ONE binary: per-session medians and max/min of the
   per-session ratio (summarize.between_sessions).
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarize as sm  # noqa: E402

E_BAR = 1e-8
METRICS = ("wall", "t_setup_s", "t_scf_s", "s_per_iter")
MIN_N = 3


def gated(r):
    return r.get("kind") == "timed" and bool(r.get("gated")) and sm.is_valid(r)


def spread(v):
    med = statistics.median(v)
    return dict(
        n=len(v),
        median=med,
        range_rel=(max(v) - min(v)) / med if med else float("nan"),
        iqr_rel=sm.q(v) / med if med else float("nan"),
    )


def flag(s, bar):
    if s["n"] < MIN_N:
        return "n<3"
    return "SPREAD>%g%%" % (bar * 100) if s["range_rel"] > bar else "ok"


def groups(rows):
    g = defaultdict(list)
    for r in rows:
        g[
            (r["plan_id"], r["threads"], r["code"], r["variant"], r.get("bin_sha256"))
        ].append(r)
    return g


def analyze(rows, metric="wall", bar=0.02, n_boot=10000, seed=228):
    """Returns (spread rows, ratio rows, energy rows) as dicts (also used by the tests)."""
    rows = [sm.normalise(dict(r)) for r in rows if r.get("kind") == "timed"]
    all_g, ok_g = groups(rows), groups([r for r in rows if gated(r)])
    sp = []
    for k, rs in sorted(all_g.items(), key=lambda kv: str(kv[0])):
        v = [r[metric] for r in ok_g.get(k, []) if r.get(metric) is not None]
        s = spread(v) if v else dict(n=0, median=None, range_rel=None, iqr_rel=None)
        sp.append(
            dict(
                key=k, discarded=len(rs) - len(ok_g.get(k, [])), flag=flag(s, bar), **s
            )
        )
    flags = {d["key"]: d["flag"] for d in sp}
    ratios, energies = [], []
    for (pid, n, code, var, sha), rf in sorted(ok_g.items(), key=lambda kv: str(kv[0])):
        if code != "ferric":
            continue
        kp = (pid, n, "pyscf", "", sha)
        rp = ok_g.get(kp, [])
        pairs = sm.pair_by_rep(rf, rp, metric)
        d = dict(plan_id=pid, threads=n, variant=var, sha=sha, n_pairs=len(pairs))
        if pairs:
            d["ratio"], d["lo"], d["hi"] = sm.paired_bootstrap_ci(pairs, n_boot, seed)
        d["flag"] = (
            "ok"
            if flags.get((pid, n, code, var, sha)) == flags.get(kp) == "ok"
            else (
                f"ferric {flags.get((pid, n, code, var, sha))} / pyscf {flags.get(kp, 'none')}"
            )
        )
        ratios.append(d)
        ef = [r["energy"] for r in rf if r.get("energy") is not None]
        ep = [r["energy"] for r in rp if r.get("energy") is not None]
        if ef and ep:
            de = abs(statistics.median(ef) - statistics.median(ep))
            method = rf[0].get("method")
            energies.append(
                dict(
                    plan_id=pid,
                    threads=n,
                    variant=var,
                    method=method,
                    dE=de,
                    flag="ok" if method != "rhf" or de <= E_BAR else "dE>1e-8",
                )
            )
    return sp, ratios, energies


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--metrics", nargs="+", default=list(METRICS))
    ap.add_argument("--bar", type=float, default=0.02)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=228)
    a = ap.parse_args(argv)
    rows = sm.load_rows(a.paths)
    shas = {r.get("bin_sha256") for r in rows if r.get("kind") == "timed"}
    print(
        f"# {len(rows)} rows, binaries {sorted(sm.sha_short(s) for s in shas)}; "
        f"gated samples only; spread bar {a.bar:.0%} (range/median), applied per metric"
    )
    for metric in a.metrics:
        energies = report(rows, metric, a)
    print("\n## 3. |E_ferric - E_PySCF| (medians, Ha/cell; RHF bar 1e-8)")
    for d in energies:
        print(
            f"{d['plan_id']:24s} N={d['threads']} {d['variant'] or '-':8s} {d['method']} "
            f"{d['dE']:.2e}  {d['flag']}"
        )
    sessions = {r.get("session") for r in rows if gated(r)}
    if len(sessions) > 1 and len(shas) == 1:
        print("\n## 4. between sessions (one binary)")
        for metric in a.metrics:
            between(rows, metric)
    elif len(shas) > 1:
        print("\n## 4. several binaries: sessions are NOT compared across them")
    return 0


def report(rows, metric, a):
    sp, ratios, energies = analyze(rows, metric, a.bar, a.boot, a.seed)
    print(f"\n## 1 [{metric}]. spread per group (plan id, N, code, variant, binary)")
    for d in sp:
        pid, n, code, var, sha = d["key"]
        print(
            f"{pid:24s} N={n} {code:6s} {var or '-':8s} {sm.sha_short(sha)} n={d['n']} "
            f"(discarded {d['discarded']}) median {fmt(d['median'])} range/med "
            f"{fmt(d['range_rel'], 4)} IQR/med {fmt(d['iqr_rel'], 4)}  {d['flag']}"
        )
    print(
        f"\n## 2 [{metric}]. ferric / PySCF (ratio of medians over paired reps, paired "
        "bootstrap 95% CI; flag on this metric)"
    )
    for d in ratios:
        ci = (
            f"{d['ratio']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"
            if d["n_pairs"]
            else "no pairs"
        )
        print(
            f"{d['plan_id']:24s} N={d['threads']} {d['variant'] or '-':8s} "
            f"pairs {d['n_pairs']}: {ci}  {d['flag']}"
        )
    return energies


def between(rows, metric):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if gated(r) and r.get(metric) is not None:
            by[(r["plan_id"], r["threads"], r["code"], r["variant"])][
                r["session"]
            ].append(r[metric])
    for k, v in sorted(by.items(), key=lambda kv: str(kv[0])):
        b = sm.between_sessions(v)
        print(
            f"[{metric}]",
            k,
            {s: round(m, 3) for s, m in b["medians"].items()},
            f"max/min {b['max_over_min']:.4f}",
        )


if __name__ == "__main__":
    sys.exit(main())
