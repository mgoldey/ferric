"""Stage B tail analysis (production maps, direct path).

usage: python tail.py LOGDIR   (reads LOGDIR/stageB_c<N>*.log)
Per (size, op, tol): eps*, keep%, kept amplitudes = keep% x (no*nv)^2, pairs, dom,
interpolated as in terfc_locality_map/analyze.py. Wall = mean over ALL eps rows of the
op at that size (wall is nearly eps-independent for terfc: integrals dominate).
Tail exponents: last-3-point and last-2-point log-log slopes vs N, plus all-points.
"""

from __future__ import annotations

import glob
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "terfc_locality_map"))
import analyze as A

OPS = ["coul", "erfc:0.42", "terfc:1.0:0.70710678", "terfc:0.991:1.354"]
NAME = {"coul": "Coulomb", "erfc:0.42": "erfc 0.42", "terfc:1.0:0.70710678": "terfc linked", "terfc:0.991:1.354": "terfc NCP"}


def main(logdir):
    sizes = {}
    for p in sorted(glob.glob(f"{logdir}/stageB_c*.log")):
        m = re.search(r"stageB_c(\d+)", p)
        if "_all" in p:
            continue
        sizes.setdefault(int(m.group(1)), []).append(p)
    res, wall = {}, {}
    for n, ps in sizes.items():
        ops, rows = A.parse(ps)
        for (mol, spec), rs in rows.items():
            wall[(n, spec)] = float(np.mean([r["wall"] for r in rs]))
            for tol in (1e-3, 1e-2):
                r = A.interp(rs, tol)
                res[(n, spec, tol)] = r
    ns = sorted(sizes)
    for tol in (1e-3, 1e-2):
        print(f"=== relerr {tol:g}")
        print("op N eps* keep% kept_amp pairs dom wall_mean")
        for spec in OPS:
            for n in ns:
                r = res.get((n, spec, tol))
                if r is None:
                    print(f"{NAME[spec]:<13} {n:>3} not bracketed")
                    continue
                no, nv = 3 * n + 1, 9 * n + 3
                ka = r["keep"] / 100 * (no * nv) ** 2
                print(f"{NAME[spec]:<13} {n:>3} {r['eps']:.2e} {r['keep']:6.3f} {ka:10.0f} {r['pairs']:6.0f} {r['dom']:6.1f} {wall.get((n, spec), float('nan')):7.1f}")
        print("-- tail exponents (log-log vs N)")
        for spec in OPS:
            pts = [(n, res[(n, spec, tol)]) for n in ns if res.get((n, spec, tol))]
            if len(pts) < 3:
                continue

            def slope(xs, ys):
                return np.polyfit(np.log(xs), np.log(ys), 1)[0]

            N = [p[0] for p in pts]
            KA = [p[1]["keep"] / 100 * ((3 * p[0] + 1) * (9 * p[0] + 3)) ** 2 for p in pts]
            PR = [p[1]["pairs"] for p in pts]
            W = [wall[(p[0], spec)] for p in pts]
            out = f"{NAME[spec]:<13}"
            for lab, y in (("kept-amp", KA), ("pairs", PR), ("wall", W)):
                out += f" | {lab}: all{len(N)}={slope(N, y):.2f} last3={slope(N[-3:], y[-3:]):.2f} last2={slope(N[-2:], y[-2:]):.2f}"
            print(out)
        print("-- NCP / linked ratios")
        for n in ns:
            a, b = res.get((n, "terfc:0.991:1.354", tol)), res.get((n, "terfc:1.0:0.70710678", tol))
            if a and b:
                print(f"N={n}: keep {a['keep']/b['keep']:.3f} pairs {a['pairs']/b['pairs']:.3f} dom {a['dom']/b['dom']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
