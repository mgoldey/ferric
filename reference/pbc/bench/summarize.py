"""Summary tables from bench_results/timing_*.jsonl (valid runs only).

python summarize.py bench_results/timing_n1.jsonl [...] [--metric wall] [--boot 10000] [--seed 228]
                    [--quiet-only]

Sections (issue #228; the statistics are pre-registered in README.md "Session 4 pre-registration"):
1. Per (cell, N, code, input, box_state): median [min-max], IQR, IQR/median, relative MAD
   (median |x - median| / median) per metric; a row is labelled NOISY when IQR/median of the chosen
   metric exceeds 10%.  Rows written before `box_state` existed are labelled "unlabelled".
2. ferric/PySCF ratio of medians, same cell and same N, for every (ferric input, PySCF input) pair,
   with a PAIRED bootstrap 95% percentile CI: pairs are (session, rep) keys that have a valid row for
   both codes (last valid attempt of each); resampling draws pairs with replacement, so the two
   medians of one resample come from the same reps.  The RNG is random.Random(seed): reproducible.
3. Between sessions, per (cell, N, code, input): each session's median, the CV of those medians
   (sample stdev / mean) and max/min; and per (cell, N, ferric input, PySCF input) each session's
   ratio of medians with max/min.  QUOTABLE iff that ratio's max/min - 1 < 5% over >= 2 sessions.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scatter_driver import is_scf_stage  # noqa: E402

KEYS = [
    ("wall", "wall"),
    ("t_setup_s", "setup"),
    ("t_scf_s", "scf"),
    ("s_per_iter", "s/iter"),
    ("iterations", "iters"),
    ("t_df_build_s", "df"),
    ("t_hcore_s", "hcore"),
    ("foreign_cpu_frac", "foreign"),
]
NOISY = 0.10
QUOTE_RANGE = 0.05


def q(v):
    v = sorted(v)
    if len(v) < 2:
        return 0.0
    qs = statistics.quantiles(v, n=4, method="inclusive")
    return qs[2] - qs[0]


def spread_stats(v):
    """median, min, max, IQR, IQR/median, relative MAD, noisy flag (IQR/median > 10%)."""
    med = statistics.median(v)
    iqr = q(v)
    mad = statistics.median([abs(x - med) for x in v])
    iqr_rel = iqr / med if med else float("nan")
    return dict(
        n=len(v),
        median=med,
        min=min(v),
        max=max(v),
        iqr=iqr,
        iqr_rel=iqr_rel,
        rel_mad=mad / med if med else float("nan"),
        noisy=bool(iqr_rel > NOISY),
    )


def input_label(r):
    if r["code"] == "ferric":
        return r.get("toml", "").split("/")[-1]
    rule = r.get("conv_rule", "pyscf")
    extra = f" [{r['pyscf_args']}]" if r.get("pyscf_args") else ""
    return f"{r.get('method', 'rhf')} p{r.get('precision')} g{r.get('grid')} conv={rule}{extra}"


def normalise(r):
    """Recompute ferric's stage split from the raw stage table (one definition everywhere)."""
    if r["code"] == "ferric" and (r.get("stages") or {}).get("total") is not None:
        st = r["stages"]
        r["t_scf_s"] = sum(v for n, v in st.items() if is_scf_stage(n))
        r["t_setup_s"] = st["total"] - r["t_scf_s"]
        if r.get("iterations"):
            r["s_per_iter"] = r["t_scf_s"] / r["iterations"]
    r.setdefault("box_state", "unlabelled")
    return r


def is_valid(r):
    return bool(r.get("valid")) and r.get("exit") == 0 and not r.get("smoke")


def load_rows(paths):
    return [normalise(json.loads(ln)) for p in paths for ln in open(p) if ln.strip()]


def pair_by_rep(rows_a, rows_b, metric):
    """[(a, b)] metric values for (session, rep) keys present in both (last valid attempt each)."""

    def last(rows):
        d = {}
        for r in sorted(rows, key=lambda r: r.get("attempt", 0)):
            if r.get(metric) is not None:
                d[(r.get("session"), r.get("rep"))] = r[metric]
        return d

    da, db = last(rows_a), last(rows_b)
    keys = sorted(set(da) & set(db), key=lambda k: (str(k[0]), k[1]))
    return [(da[k], db[k]) for k in keys]


def median_ratio(pairs):
    return statistics.median([a for a, _ in pairs]) / statistics.median(
        [b for _, b in pairs]
    )


def paired_bootstrap_ci(pairs, n_boot=10000, seed=228, level=0.95):
    """(point, lo, hi): ratio of medians and its paired-bootstrap percentile CI."""
    if not pairs:
        raise ValueError("no pairs")
    rng = random.Random(seed)
    n = len(pairs)
    stats = []
    for _ in range(n_boot):
        s = [pairs[rng.randrange(n)] for _ in range(n)]
        stats.append(median_ratio(s))
    stats.sort()
    lo_i = int((1 - level) / 2 * (n_boot - 1) + 0.5)
    hi_i = int((1 + level) / 2 * (n_boot - 1) + 0.5)
    return median_ratio(pairs), stats[lo_i], stats[hi_i]


def between_sessions(values_by_session):
    """{session: [values]} -> per-session medians, CV of the medians, max/min."""
    meds = {s: statistics.median(v) for s, v in sorted(values_by_session.items()) if v}
    m = list(meds.values())
    cv = statistics.stdev(m) / statistics.mean(m) if len(m) >= 2 else float("nan")
    return dict(
        medians=meds, cv=cv, max_over_min=max(m) / min(m) if m else float("nan")
    )


def fmt(x, nd=3):
    return "nan" if x is None or x != x else f"{x:.{nd}f}"


def section_spread(g, bad, metric):
    print(
        "\n## 1. cell N code (input) box_state | n_valid n_discarded | metric: median [min-max] "
        "IQR IQR/med relMAD | energy spread | load range"
    )
    for k in sorted(set(g) | set(bad), key=str):
        rs = g.get(k, [])
        if not rs:
            print(k, f"n=0 bad={bad[k]} | no valid runs")
            continue
        out = []
        noisy = False
        for key, nm in KEYS:
            v = [r[key] for r in rs if r.get(key) is not None]
            if v:
                st = spread_stats(v)
                if key == metric:
                    noisy = st["noisy"]
                out.append(
                    f"{nm} {st['median']:.3f} [{st['min']:.3f}-{st['max']:.3f}] IQR {st['iqr']:.3f} "
                    f"({100 * st['iqr_rel']:.1f}%) relMAD {100 * st['rel_mad']:.1f}%"
                )
        e = [r["energy"] for r in rs if r.get("energy") is not None]
        e_txt = f"E {e[0]:.10f} spread {max(e) - min(e):.1e}" if e else "E missing"
        rss = [r["child_peak_rss_bytes"] for r in rs if r.get("child_peak_rss_bytes")]
        if rss:
            e_txt += f" | RSS {statistics.median(rss) / 2**30:.3f} GiB"
        conv = {r.get("converged") for r in rs}
        e_txt += f" | converged {sorted(conv, key=str)}"
        lo = min(min(r["load_start"], r["load_end"]) for r in rs)
        hi = max(max(r["load_start"], r["load_end"]) for r in rs)
        print(
            k,
            f"n={len(rs)} bad={bad.get(k, 0)}{' NOISY' if noisy else ''} |",
            " | ".join(out),
            f"| {e_txt} | load {lo:.2f}-{hi:.2f}",
        )


def section_ratios(by_input, metric, n_boot, seed):
    print(
        f"\n## 2. ferric/PySCF ratio of {metric} medians, paired on (session, rep); "
        f"bootstrap {n_boot}, seed {seed}"
    )
    cells = sorted({(c, n) for (c, n, _, _) in by_input}, key=str)
    for c, n in cells:
        fk = sorted(k for k in by_input if k[:3] == (c, n, "ferric"))
        pk = sorted(k for k in by_input if k[:3] == (c, n, "pyscf"))
        for f in fk:
            for p in pk:
                if not same_method(by_input[f], by_input[p]):
                    continue
                pairs = pair_by_rep(by_input[f], by_input[p], metric)
                if not pairs:
                    print(c, f"N={n}", f[3], "vs", p[3], "| no paired reps")
                    continue
                pt, lo, hi = paired_bootstrap_ci(pairs, n_boot, seed)
                print(
                    c,
                    f"N={n}",
                    f[3],
                    "vs",
                    p[3],
                    f"| pairs {len(pairs)} | ratio {pt:.3f} 95% CI [{lo:.3f}, {hi:.3f}]",
                )


def section_sessions(by_input, metric):
    print(f"\n## 3. between sessions ({metric}): per-session medians, CV, max/min")
    for k in sorted(by_input, key=str):
        vs = defaultdict(list)
        for r in by_input[k]:
            if r.get(metric) is not None:
                vs[r.get("session")].append(r[metric])
        bs = between_sessions(vs)
        meds = " ".join(f"{s}:{v:.3f}" for s, v in bs["medians"].items())
        print(k, f"| {meds} | CV {fmt(bs['cv'])} | max/min {fmt(bs['max_over_min'])}")
    print(
        "\n   per-session ferric/PySCF ratio of medians (QUOTABLE iff max/min - 1 < "
        f"{100 * QUOTE_RANGE:.0f}% over >= 2 sessions)"
    )
    cells = sorted({(c, n) for (c, n, _, _) in by_input}, key=str)
    for c, n in cells:
        fk = sorted(k for k in by_input if k[:3] == (c, n, "ferric"))
        pk = sorted(k for k in by_input if k[:3] == (c, n, "pyscf"))
        for f in fk:
            for p in pk:
                if not same_method(by_input[f], by_input[p]):
                    continue
                ratios = session_ratios(by_input[f], by_input[p], metric)
                if not ratios:
                    continue
                v = list(ratios.values())
                mm = max(v) / min(v)
                ok = len(v) >= 2 and mm - 1 < QUOTE_RANGE
                txt = " ".join(f"{s}:{x:.3f}" for s, x in ratios.items())
                print(
                    c,
                    f"N={n}",
                    f[3],
                    "vs",
                    p[3],
                    f"| {txt} | max/min {mm:.3f}",
                    "QUOTABLE" if ok else "not quotable",
                )


def same_method(rows_f, rows_p):
    return {r.get("method", "rhf") for r in rows_f} == {
        r.get("method", "rhf") for r in rows_p
    }


def session_ratios(rows_f, rows_p, metric):
    """{session: median_f / median_p} over sessions with valid rows for both codes."""
    vf, vp = defaultdict(list), defaultdict(list)
    for r in rows_f:
        if r.get(metric) is not None:
            vf[r.get("session")].append(r[metric])
    for r in rows_p:
        if r.get(metric) is not None:
            vp[r.get("session")].append(r[metric])
    return {
        s: statistics.median(vf[s]) / statistics.median(vp[s])
        for s in sorted(set(vf) & set(vp), key=str)
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--metric", default="wall")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=228)
    ap.add_argument(
        "--quiet-only", action="store_true", help="drop rows with box_state != quiet"
    )
    a = ap.parse_args(argv)
    rows = load_rows(a.paths)
    g = defaultdict(list)
    bad = defaultdict(int)
    by_input = defaultdict(
        list
    )  # (cell, N, code, input) -> valid rows (all box states)
    for r in rows:
        if a.quiet_only and r["box_state"] != "quiet":
            continue
        k = (r["cell"], r["threads"], r["code"], input_label(r), r["box_state"])
        if is_valid(r):
            g[k].append(r)
            by_input[k[:4]].append(r)
        else:
            bad[k] += 1
    section_spread(g, bad, a.metric)
    section_ratios(by_input, a.metric, a.boot, a.seed)
    section_sessions(by_input, a.metric)


if __name__ == "__main__":
    main()
