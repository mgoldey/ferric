#!/usr/bin/env python3
"""Tabulate lmp2_eri3_bench.sh runs: one row per (run, config), then pooled rows.

Usage: lmp2_eri3_analyze.py OUTDIR [--ab]   (--ab compares binaries, not split on/off)
Reads OUTDIR/runs/*.out and rows.jsonl. No interpretation here.
"""
import glob
import json
import os
import re
import statistics as st
import sys

WORKERS = 6  # physical cores; RAYON_NUM_THREADS in the harness


def parse_run(path):
    rows, cur = [], None
    for line in open(path):
        if "alkane_" in line and " | " in line and "sys " not in line:
            body = line[line.index("alkane_"):]
            head, tail = body.rsplit(" | ", 1)
            parts = head.split()
            t = tail.split()
            cur = {"t_maps": float(parts[-5]), "t_metric": float(parts[-3]), "t_pairs": float(parts[-2]),
                   "t_asm": float(t[0]), "t_solve": float(t[1]), "e_corr": parts[3], "keep": parts[5]}
        elif line.startswith("ERI3 "):
            d = dict(kv.split("=") for kv in line.split()[1:])
            out = dict(cur or {})
            for k, v in d.items():
                if k in ("op", "eps", "e_bits"):
                    out[k] = v
                elif k in ("c", "n_batches", "n_slabs", "workers", "triples", "skipped", "integrals", "null", "cg"):
                    out[k] = int(v)
                else:
                    out[k] = float(v)
            out["total"] = out["t_asm"] + out["t_solve"]
            rows.append(out)
    return rows


def derive(r):
    thread = r["batch_mean"] * r["n_batches"]
    workers = min(r["workers"], WORKERS, r["n_batches"])
    r["thread"] = thread
    r["residual"] = thread - r["fill"] - r["transform"]
    parallel_wall = r["wall"] - r["setup"]
    r["eff"] = thread / (parallel_wall * workers)
    r["lb"] = max(r["batch_max"], thread / workers)  # lower bound on parallel wall
    return r


def main():
    o = sys.argv[1]
    ab = "--ab" in sys.argv
    meta = {}
    p = os.path.join(o, "rows.jsonl")
    if os.path.exists(p):
        for line in open(p):
            m = json.loads(line)
            meta[m["name"]] = m
    groups = {}
    print("run c op eps | wall setup total | thread fill eval scatter transform resid | eff lb_wall max/mean nb ns "
          "| Mtriples Mskipped Mints null | tGflop tGflop_own | e_bits cg | foreign_cpu")
    for path in sorted(glob.glob(os.path.join(o, "runs", "*.out"))):
        name = os.path.basename(path)[:-4]
        m = meta.get(name, {})
        for r in parse_run(path):
            derive(r)
            mode = m.get("bin") if ab else m.get("split")
            groups.setdefault((r["c"], r["op"], r["eps"], mode), []).append(r)
            print(
                f"{name} {r['c']} {r['op']} {r['eps']} | {r['wall']:.2f} {r['setup']:.2f} {r['total']:.2f} | "
                f"{r['thread']:.1f} {r['fill']:.1f} {r['eval']:.1f} {r['scatter']:.1f} {r['transform']:.1f} "
                f"{r['residual']:.1f} | {r['eff']:.3f} {r['lb']:.2f} {r['batch_max'] / r['batch_mean']:.2f} "
                f"{r['n_batches']} {r['n_slabs']} | {r['triples'] / 1e6:.1f} {r['skipped'] / 1e6:.1f} "
                f"{r['integrals'] / 1e6:.0f} {r['null']} | {r['tflops'] / 1e9:.0f} {r['tflops_own'] / 1e9:.0f} | "
                f"{r['e_bits']} {r['cg']} | {m.get('foreign_cpu_s', '?')}"
            )
    print("\n# pooled: mean (spread = max-min) by config and mode")
    for key in sorted(groups, key=str):
        g = groups[key]
        def f(k):
            v = [r[k] for r in g]
            return f"{st.mean(v):.2f}({max(v) - min(v):.2f})"
        bits = {r["e_bits"] for r in g}
        print(f"{key} n={len(g)} wall={f('wall')} total={f('total')} fill={f('fill')} eval={f('eval')} "
              f"scatter={f('scatter')} transform={f('transform')} resid={f('residual')} eff={f('eff')} "
              f"same_bits={len(bits) == 1}")


if __name__ == "__main__":
    main()
