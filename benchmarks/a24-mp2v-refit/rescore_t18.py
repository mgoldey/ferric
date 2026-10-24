#!/usr/bin/env python3
"""Does the NLC grid explain the offset to paper SI table 18 at short range?

Re-scores MP2-V totals (published linked parameters r0 = 1.00 A, b = 11.0,
C = 0.0089, aTZ or aDZ, non-CP, frozen core) for the small A24 systems at
stretch factors 0.9 and 1.0 (G1 geometry) with the VV10 pair kernel damped by
Eq. 11 and, as a control, bare, on several NLC grids, and prints
ferric - paper(table 18) in kcal/mol. The paper used SG-1 (Q-Chem: 50 radial x
194 Lebedev angular, pruned per radial shell); ferric has no SG-1, so
(50, 194) UNPRUNED is the closest available arm (more points than SG-1).

Usage: rescore_t18.py --systems 2,4,5 --factors 0.9,1.0 --basis atz --grids 50x50,50x194 --out results/rescore_t18_atz.json
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import session as S  # noqa: E402
import stretch  # noqa: E402

SPECS = {
    "eq11": None,  # published lockstep construction
    "bare": {"form": "bare"},
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="2,4,5")
    ap.add_argument("--factors", default="0.9,1.0")
    ap.add_argument("--basis", default="atz")
    ap.add_argument("--grids", default="50x50,50x194")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ids = [int(x) for x in a.systems.split(",")]
    fs = [float(x) for x in a.factors.split(",")]
    grids = [tuple(int(x) for x in g.split("x")) for g in a.grids.split(",")]
    out = {"basis": a.basis, "rows": []}
    for g in grids:
        pool = S.Pool(a.basis, g, None, budget_gb=4.0)
        pool.add("a24", ids, fs)
        for i in ids:
            for f in fs:
                sid = f"{i}@{f:g}"
                paper = stretch.paper("18", i)[stretch.FACTORS.index(f)]
                row = {"system": i, "factor": f, "grid": list(g), "paper_t18": paper}
                for name, vv in SPECS.items():
                    p = S.P(1.0, None, 11.0, 0.0089, vv)
                    row[name] = pool.eint(p, "a24", sid, cp=False) - paper
                out["rows"].append(row)
                print(row, flush=True)
                Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
