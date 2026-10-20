"""Median / min-max / IQR table from bench_results/timing_*.jsonl (valid runs only).

python summarize.py bench_results/timing_n1.jsonl [...]
"""

import json
import os
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


def q(v):
    v = sorted(v)
    if len(v) < 2:
        return 0.0
    qs = statistics.quantiles(v, n=4, method="inclusive")
    return qs[2] - qs[0]


def main(paths):
    rows = [json.loads(l) for p in paths for l in open(p) if l.strip()]
    g = defaultdict(list)
    bad = defaultdict(int)
    for r in rows:
        k = (
            r["cell"],
            r["threads"],
            r["code"],
            r.get("toml", "").split("/")[-1]
            if r["code"] == "ferric"
            else f"{r.get('method', 'rhf')} p{r.get('precision')} g{r.get('grid')}",
        )
        if r["code"] == "ferric" and r.get("stages", {}).get("total") is not None:
            st = r["stages"]
            r["t_scf_s"] = sum(v for n, v in st.items() if is_scf_stage(n))
            r["t_setup_s"] = st["total"] - r["t_scf_s"]
            if r.get("iterations"):
                r["s_per_iter"] = r["t_scf_s"] / r["iterations"]
        if r["valid"] and r["exit"] == 0:
            g[k].append(r)
        else:
            bad[k] += 1
    print(
        "cell N code (input) n_valid n_discarded | metric: median [min-max] IQR | energy spread | load range"
    )
    for k in sorted(set(g) | set(bad)):
        rs = g[k]
        if not rs:
            print(k, f"n=0 bad={bad[k]} | no valid runs")
            continue
        out = []
        for key, nm in KEYS:
            v = [r[key] for r in rs if r.get(key) is not None]
            if v:
                out.append(
                    f"{nm} {statistics.median(v):.3f} [{min(v):.3f}-{max(v):.3f}] IQR {q(v):.3f}"
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
            f"n={len(rs)} bad={bad[k]} |",
            " | ".join(out),
            f"| {e_txt} | load {lo:.2f}-{hi:.2f}",
        )


if __name__ == "__main__":
    main(sys.argv[1:])
