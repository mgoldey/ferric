#!/usr/bin/env python3
"""A/B table from lmp2_eri3_bench.sh runs named cNN_rK_{main,aux}. Works on the main
row, which the frozen binary also prints. Usage: lmp2_eri3_ab.py AB_OUTDIR"""
import glob
import os
import re
import statistics as st
import sys

ROW = re.compile(r"alkane_(\d+)\s+(\S+)\s+(\S+)\s+(-\d+\.\d+)")


def rows(path):
    for line in open(path):
        if "alkane_" not in line or " | " not in line:
            continue
        body = line[line.index("alkane_"):]
        head, tail = body.rsplit(" | ", 1)
        m = ROW.match(head)
        if not m:
            continue
        p, t = head.split(), tail.split()
        yield (int(m[1]), m[2], m[3]), {"e": float(m[4]), "eri3": float(p[-3]), "total": float(t[0]) + float(t[1]),
                                        "triples": float(p[-5])}


def main():
    g = {}
    for path in glob.glob(os.path.join(sys.argv[1], "runs", "*.out")):
        name = os.path.basename(path)[:-4]
        arm = name.split("_")[-1]
        if "split" in name:
            continue
        for key, r in rows(path):
            g.setdefault((key, arm), []).append(r)
    print("| C | op | eps | t_eri3 main | t_eri3 aux | ratio | total main | total aux | ratio | Mtriples main | aux | max dE |")
    print("|" + "---|" * 12)
    for key in sorted({k for k, _ in g}):
        a, b = g.get((key, "main")), g.get((key, "aux"))
        if not a or not b:
            continue

        def v(x, f):
            vals = [r[f] for r in x]
            return f"{st.mean(vals):.2f} ({min(vals):.2f}-{max(vals):.2f})"

        m = lambda x, f: st.mean(r[f] for r in x)
        d = max(abs(u["e"] - w["e"]) for u in a for w in b)
        print(f"| {key[0]} | {key[1]} | {key[2]} | {v(a, 'eri3')} | {v(b, 'eri3')} | {m(a, 'eri3') / m(b, 'eri3'):.2f} | "
              f"{v(a, 'total')} | {v(b, 'total')} | {m(a, 'total') / m(b, 'total'):.2f} | "
              f"{a[0]['triples']:.2f} | {b[0]['triples']:.2f} | {d:.1e} |")


if __name__ == "__main__":
    main()
