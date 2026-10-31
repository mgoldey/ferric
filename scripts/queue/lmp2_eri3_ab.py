#!/usr/bin/env python3
"""A/B table from lmp2_eri3_bench.sh runs named cNN_rK_{main,aux}. Works on the main
row, which the frozen binary also prints. Usage: lmp2_eri3_ab.py AB_OUTDIR

The energy column compares that row's 8-decimal prints, so 0 means agreement to about
1e-8 Ha, not exact agreement; the frozen baseline prints nothing tighter. The tight
comparison is the bits probe. The flagged column counts runs whose gate timed out, and
a run with no gate record shows as ?; neither is dropped from the means. A run recorded
with a nonzero exit code is dropped, since it may have printed only some rows.
"""

import glob
import json
import os
import re
import statistics as st
import sys

ROW = re.compile(r"alkane_(\d+)\s+(\S+)\s+(\S+)\s+(-\d+\.\d+)")


def rows(path):
    for line in open(path):
        if "alkane_" not in line or " | " not in line:
            continue
        body = line[line.index("alkane_") :]
        head, tail = body.rsplit(" | ", 1)
        m = ROW.match(head)
        if not m:
            continue
        p, t = head.split(), tail.split()
        yield (
            (int(m[1]), m[2], m[3]),
            {
                "e": float(m[4]),
                "eri3": float(p[-3]),
                "total": float(t[0]) + float(t[1]),
                "triples": float(p[-5]),
            },
        )


def main():
    g = {}
    meta = {}
    with open(os.path.join(sys.argv[1], "rows.jsonl")) as fh:
        for line in fh:
            row = json.loads(line)
            meta[row["name"]] = row
    for path in glob.glob(os.path.join(sys.argv[1], "runs", "*.out")):
        name = os.path.basename(path)[:-4]
        arm = name.split("_")[-1]
        if "split" in name:
            continue
        if meta.get(name, {}).get("rc", 0) != 0:
            print(f"skipping failed run {name}", file=sys.stderr)
            continue
        for key, r in rows(path):
            r["flagged"] = meta.get(name, {}).get("gate_flagged")
            g.setdefault((key, arm), []).append(r)
    print(
        "| C | op | eps | t_eri3 main | t_eri3 aux | ratio | total main | total aux | ratio | Mtriples main | aux | max dE (8-decimal prints, about 1e-8 resolution) | flagged runs |"
    )
    print("|" + "---|" * 13)
    for key in sorted({k for k, _ in g}):
        a, b = g.get((key, "main")), g.get((key, "aux"))
        if not a or not b:
            continue

        def v(x, f):
            vals = [r[f] for r in x]
            return f"{st.mean(vals):.2f} ({min(vals):.2f}-{max(vals):.2f})"

        m = lambda x, f: st.mean(r[f] for r in x)
        marks = [r["flagged"] for r in a + b]
        flagged = "?" if None in marks else sum(marks)
        d = max(abs(u["e"] - w["e"]) for u in a for w in b)
        print(
            f"| {key[0]} | {key[1]} | {key[2]} | {v(a, 'eri3')} | {v(b, 'eri3')} | {m(a, 'eri3') / m(b, 'eri3'):.2f} | "
            f"{v(a, 'total')} | {v(b, 'total')} | {m(a, 'total') / m(b, 'total'):.2f} | "
            f"{a[0]['triples']:.2f} | {b[0]['triples']:.2f} | {d:.1e} | {flagged} |"
        )


if __name__ == "__main__":
    main()
