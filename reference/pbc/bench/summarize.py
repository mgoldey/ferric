"""Median / min-max / IQR table from bench_results/timing_*.jsonl (valid runs only).

python summarize.py bench_results/timing_n1.jsonl [...]
"""

import json
import statistics
import sys
from collections import defaultdict

KEYS = [
    ("wall", "wall"),
    ("t_setup_s", "setup"),
    ("t_scf_s", "scf"),
    ("s_per_iter", "s/iter"),
    ("iterations", "iters"),
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
            r.get("toml", "").split("/")[-1] if r["code"] == "ferric" else "",
        )
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
