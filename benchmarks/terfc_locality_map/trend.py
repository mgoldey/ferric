"""Size trend (C8/C10/C12) at matched relative error. PROVISIONAL: 3 points,
below the ~C20 onset; kept-amplitude count = keep% x (no*nv)^2 with
no = 3N+1, nv = 9N+3 (alkane C_N, 6-31G, frozen core N).

usage: python trend.py LOG_C8 LOG_C10 LOG_C12
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A

SIZES = {"alkane_8.xyz": 8, "alkane_10.xyz": 10, "alkane_12.xyz": 12}


def main(paths):
    res = {}
    for p in paths:
        _, rows = A.parse([p])
        for (m, spec), rs in rows.items():
            for tol in (1e-3, 1e-2):
                r = A.interp(rs, tol)
                if r:
                    res.setdefault((spec, tol), {})[SIZES[m]] = r
    ns = [8, 10, 12]
    nn = [(3 * n + 1) * (9 * n + 3) for n in ns]
    for tol in (1e-3, 1e-2):
        print(f"tol {tol:g}")
        for (spec, t), d in res.items():
            if t != tol or len(d) < 3:
                continue
            kept = [d[n]["keep"] / 100 * m * m for n, m in zip(ns, nn)]
            pairs = [d[n]["pairs"] for n in ns]
            dom = [d[n]["dom"] for n in ns]

            def ex(y):
                return np.polyfit(np.log(ns), np.log(y), 1)[0]

            def tail(y):
                return math.log(y[2] / y[1]) / math.log(12 / 10)

            print(
                f"{spec:<22} keep% {d[8]['keep']:5.2f} {d[10]['keep']:5.2f} "
                f"{d[12]['keep']:5.2f} | kept-amp exp(3pt) {ex(kept):.2f} tail2 "
                f"{tail(kept):.2f} | pairs {pairs[0]:.0f} {pairs[1]:.0f} {pairs[2]:.0f} "
                f"exp {ex(pairs):.2f} tail2 {tail(pairs):.2f} | dom "
                f"{dom[0]:.1f} {dom[1]:.1f} {dom[2]:.1f}"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
