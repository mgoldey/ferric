#!/usr/bin/env python3
"""Fit tail scaling exponents from bench_direct_alkane_series log rows.

Parses 'alkane_N op eps ...' rows (the LAST occurrence of each (N, op, eps)
wins, so re-runs supersede), then for each (op, eps) reports pairwise and
last-three least-squares exponents of t_asm, t_asm+t_solve, t_eri3, t_pairs
and eri3 Mtriples vs N. Protocol: fit the TAIL, never the whole series.

Usage: fit_tail_exponents.py <logfile> [minN]
"""
import math
import re
import sys
from collections import OrderedDict

log = sys.argv[1]
min_n = int(sys.argv[2]) if len(sys.argv) > 2 else 16

pat = re.compile(
    r"alkane_(\d+)\s+(\S+)\s+([0-9.e+-]+)\s+(-?\d+\.\d+)\s+([+-][\d.e-]+|NaN)\s+"
    r"([\d.]+)\s+(\d+)/(\d+)\s+(\d+)/(\d+)\s+(\d+)/(\d+)\s+(\d+)/(\d+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)

rows = OrderedDict()
for line in open(log):
    m = pat.search(line)
    if not m:
        continue
    n = int(m.group(1))
    op, eps = m.group(2), m.group(3)
    rows[(op, eps, n)] = {
        "E": float(m.group(4)),
        "dE": float("nan") if m.group(5) == "NaN" else float(m.group(5)),
        "Mtrip": float(m.group(15)),
        "t_eri3": float(m.group(17)),
        "t_pairs": float(m.group(19)),
        "t_asm": float(m.group(20)),
        "t_solve": float(m.group(21)),
        "t_ref": float(m.group(22)),
    }

def exp_fit(ns, ys):
    """least-squares slope of log y vs log n"""
    pts = [(math.log(n), math.log(y)) for n, y in zip(ns, ys) if y > 0]
    if len(pts) < 2:
        return float("nan")
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    num = sum((x - mx) * (y - my) for x, y in pts)
    den = sum((x - mx) ** 2 for x, y in pts)
    return num / den if den else float("nan")

series = {}
for (op, eps, n), d in rows.items():
    series.setdefault((op, eps), []).append((n, d))

for (op, eps), pts in sorted(series.items()):
    pts = sorted({n: d for n, d in pts}.items())
    pts = [(n, d) for n, d in pts if n >= min_n]
    if len(pts) < 2:
        continue
    ns = [n for n, _ in pts]
    print(f"\n== {op} eps={eps}  (N: {ns}) ==")
    for key in ["t_asm", "t_solve", "t_eri3", "t_pairs", "Mtrip", "t_ref"]:
        ys = [d[key] for _, d in pts]
        method_note = ""
        if key == "t_asm":
            ys_tot = [d["t_asm"] + d["t_solve"] for _, d in pts]
            e_tot = exp_fit(ns[-3:], ys_tot[-3:])
            method_note = f"   [method=asm+solve tail-3 exp {e_tot:.2f}]"
        pair_exps = [
            f"{ns[i]}->{ns[i+1]}:{exp_fit(ns[i:i+2], ys[i:i+2]):.2f}"
            for i in range(len(ns) - 1)
        ]
        tail3 = exp_fit(ns[-3:], ys[-3:])
        print(f"  {key:8} {['%.2f' % y for y in ys]}  pair[{' '.join(pair_exps)}] tail3={tail3:.2f}{method_note}")
    des = [d["dE"] for _, d in pts]
    per_c = [de / n for (n, _), de in zip(pts, des)]
    print(f"  dE       {['%+.3e' % d for d in des]}  per-C {['%+.2e' % p for p in per_c]}")
