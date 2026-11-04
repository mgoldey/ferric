#!/usr/bin/env python3
"""Tables from scripts/queue/lmp2_sweep.sh rows.jsonl. NUMBERS ONLY: no
verdicts. Interpretation belongs in the wiki note, dated and provisional.

Prints:
  1. run inventory (status counts, flagged rows, build identity, env)
  2. §34-style table per config: per-N median method time (CLI stage sum
     maps+eri3+metric+pairs+solve) with every rep listed, and the tail-3
     log-log exponent; the same for t_post_scf (run-log time from the last SCF
     record to the result: adds aux prep, Boys, VV-HV, certificate)
  3. exact canonical reference times and the speedup at each N
  4. error vs exact per eps (absolute, per carbon, % of correlation)
  5. stage-share breakdown at the largest N
Flag markers: '*' = a rep carries a contention flag, '!' = not ok.

Usage: lmp2_sweep_analyze.py rows.jsonl
"""
import json
import math
import statistics
import sys
from collections import defaultdict

STAGES = ["t_maps_s", "t_eri3_s", "t_metric_s", "t_pairs_s", "t_solve_s"]


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tail_exp(ns, ys):
    """Least-squares slope of log y vs log N over the LAST THREE points."""
    pts = [(math.log(n), math.log(y)) for n, y in zip(ns, ys) if y and y > 0][-3:]
    if len(pts) < 2:
        return float("nan")
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    return sum((x - mx) * (y - my) for x, y in pts) / den if den else float("nan")


def fmt(x, spec="%.2f"):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else spec % x


def cfg_key(c):
    return (c["op"], c["eps"])


def main():
    rows = load(sys.argv[1])
    print("== inventory ==")
    st = defaultdict(int)
    for r in rows:
        st[r["status"]] += 1
    print("  rows %d  " % len(rows) + "  ".join("%s=%d" % kv for kv in sorted(st.items())))
    commits = {(r.get("version", {}).get("commit"), r.get("version", {}).get("dirty")) for r in rows}
    print("  build (commit, dirty): %s" % sorted(commits, key=str))
    envs = {json.dumps({k: r["env"].get(k) for k in ("OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS",
                                                       "FERRIC_MEM_BUDGET_GB", "hostname", "binary_sha256")},
                       sort_keys=True) for r in rows}
    for e in sorted(envs):
        print("  env %s" % e)
    flagged = [r for r in rows if r.get("flags")]
    print("  flagged rows: %d" % len(flagged))
    for r in flagged:
        c = r.get("contention", {})
        print("    run %3d %-18s r%d %s  load1 %s->%s excess %s foreign_cores %s psi %s->%s gate_wait %ss"
              % (r["run_idx"], r["cell_id"], r["rep"], ",".join(r["flags"]), fmt(c.get("load1_before")),
                 fmt(c.get("load1_after")), fmt(c.get("load1_after_excess")), fmt(c.get("foreign_cores_avg")),
                 fmt(c.get("psi_some_avg10_before")), fmt(c.get("psi_some_avg10_after")),
                 fmt(r.get("gate", {}).get("waited_s"), "%.0f")))
    for r in rows:
        if r["status"] != "ok":
            why = r.get("skip_reason") or " ".join((r.get("stderr_tail") or "").strip().splitlines()[-1:])
            print("    NOT OK run %3d %-18s r%d %s %s" % (r["run_idx"], r["cell_id"], r["rep"], r["status"], why))
    for r in rows:
        if r["status"] == "ok" and r["cell"]["op"] == "erfc":
            w = r.get("omega_bohr_printed")
            if w is not None and abs(w - 1.0) > 1e-4:
                print("    OMEGA MISMATCH run %d: printed %.5f Bohr^-1 (expected 1.0)" % (r["run_idx"], w))

    local = defaultdict(lambda: defaultdict(list))   # cfg -> N -> rows
    ref = defaultdict(lambda: defaultdict(list))     # op -> N -> rows
    for r in rows:
        c = r["cell"]
        (local[cfg_key(c)] if c["kind"] == "local" else ref[c["op"]])[c["n"]].append(r)
    sizes = sorted({r["cell"]["n"] for r in rows})

    def med(rs, key):
        v = [r.get(key) for r in rs if r["status"] == "ok" and r.get(key) is not None]
        return statistics.median(v) if v else None

    def reps(rs, key):
        out = []
        for r in sorted(rs, key=lambda r: r["rep"]):
            mark = ("*" if r.get("flags") else "") + ("" if r["status"] == "ok" else "!" + r["status"])
            v = r.get(key)
            out.append((fmt(v) if r["status"] == "ok" else "") + mark)
        return "/".join(out)

    for key, title in (("method_stage_sum_s", "method = CLI stage sum maps+eri3+metric+pairs+solve (s)"),
                       ("t_post_scf_s", "t_post_scf = run-log result.t - last SCF record t (s)")):
        print("\n== %s ==" % title)
        print("  %-11s " % "op/eps" + " ".join("%-22s" % ("C%d med [reps]" % n) for n in sizes) + " tail3_exp")
        for k in sorted(local, key=lambda k: (k[0], k[1])):
            ns, ys, cells = [], [], []
            for n in sizes:
                rs = local[k].get(n, [])
                m = med(rs, key)
                cells.append("%-22s" % ("%s [%s]" % (fmt(m), reps(rs, key)) if rs else "—"))
                if m is not None:
                    ns.append(n)
                    ys.append(m)
            print("  %-11s " % ("%s/%.0e" % k) + " ".join(cells) + " %s" % fmt(tail_exp(ns, ys)))

    print("\n== exact canonical reference (t_post_scf, s) and speedup = t_ref / t_local ==")
    for op in sorted(ref):
        for n in sorted(ref[op]):
            rs = ref[op][n]
            tr = med(rs, "t_post_scf_s")
            print("  %-4s C%-3d ref %s [%s]  e_corr %s" % (op, n, fmt(tr), reps(rs, "t_post_scf_s"),
                                                        fmt(med(rs, "e_corr"), "%.10f")))
            for k in sorted(local):
                if k[0] != op or n not in local[k]:
                    continue
                tm = med(local[k][n], "method_stage_sum_s")
                tp = med(local[k][n], "t_post_scf_s")
                sp1 = tr / tm if tr and tm else None
                sp2 = tr / tp if tr and tp else None
                print("      eps %.0e  speedup vs stage-sum %s  vs t_post_scf %s" % (k[1], fmt(sp1), fmt(sp2)))

    print("\n== error vs exact (e_corr local - e_corr exact) ==")
    print("  %-11s %-5s %-14s %-12s %-10s %-10s %s" % ("op/eps", "N", "dE (Ha)", "dE/C (Ha)", "dE/Ecorr %",
                                                      "keep", "rep spread (Ha)"))
    for k in sorted(local):
        for n in sizes:
            rs = [r for r in local[k].get(n, []) if r["status"] == "ok" and r.get("e_corr") is not None]
            if not rs:
                continue
            es = [r["e_corr"] for r in rs]
            e = statistics.median(es)
            spread = max(es) - min(es)
            exact = med(ref[k[0]].get(n, []), "e_corr")
            if exact is None:  # in-run opt-in reference, if a row carries one
                exact = med(rs, "e_corr_canonical_ri")
            de = e - exact if exact is not None else None
            print("  %-11s C%-4d %-14s %-12s %-10s %-10s %s"
                  % ("%s/%.0e" % k, n, fmt(de, "%+.4e"), fmt(de / n if de is not None else None, "%+.3e"),
                     fmt(100 * de / exact if de is not None and exact else None, "%+.4f"),
                     fmt(med(rs, "keep_fraction"), "%.4f"), "%.1e" % spread))

    nmax = max((r["cell"]["n"] for r in rows if r["cell"]["kind"] == "local" and r["status"] == "ok"), default=None)
    if nmax is not None:
        print("\n== stage share at C%d (median over ok reps; %% of stage sum; other = t_post_scf - stage sum) ==" % nmax)
        print("  %-11s " % "op/eps" + " ".join("%-8s" % s[2:-2] for s in STAGES) + " %-8s %-9s %-10s %s"
              % ("sum_s", "other_s", "peakRSS_GB", "cg"))
        for k in sorted(local):
            rs = [r for r in local[k].get(nmax, []) if r["status"] == "ok" and r.get("stages_stdout")]
            if not rs:
                print("  %-11s (no ok rows)" % ("%s/%.0e" % k))
                continue
            vals = {s: statistics.median(r["stages_stdout"][s] for r in rs) for s in STAGES}
            tot = sum(vals.values())
            post = med(rs, "t_post_scf_s")
            rss = med(rs, "peak_rss_bytes")
            cg = med([{"status": "ok", "cg": r.get("counters", {}).get("cg_iterations")} for r in rs], "cg")
            print("  %-11s " % ("%s/%.0e" % k) + " ".join("%-8s" % ("%.1f%%" % (100 * vals[s] / tot)) for s in STAGES)
                  + " %-8s %-9s %-10s %s" % (fmt(tot), fmt(post - tot if post is not None else None),
                                            fmt(rss / 1e9 if rss else None), fmt(cg, "%.0f")))

    print("\n== structural counters (median over ok reps) ==")
    for k in sorted(local):
        for n in sizes:
            rs = [r for r in local[k].get(n, []) if r["status"] == "ok" and r.get("counters")]
            if not rs:
                continue
            c = rs[0]["counters"]
            s = rs[0].get("strips", {})
            print("  %-11s C%-3d dom %.1f/%d cand %.1f/%d pairs %.3f gated %d strips r %.0f/%s c %.0f/%s Mtrip %s"
                  % ("%s/%.0e" % k, n, c["dom_mean"], c["dom_max"], c["cand_mean"], c["cand_max"], c["pairs"],
                     c["gated"], s.get("rows_mean", float("nan")), s.get("rows_max"), s.get("cols_mean", float("nan")),
                     s.get("cols_max"), fmt(s.get("eri3_mtriples_evald"), "%.1f")))


if __name__ == "__main__":
    main()
