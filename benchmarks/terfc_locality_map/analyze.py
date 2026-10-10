"""Fair-criterion table from terfc_locality_map harness logs.

usage: python analyze.py LOG [LOG ...]  (wall = dense-path LMP2 wall at eps*, shared box)
For each (mol, operator): relerr(eps) = |E(eps)-E(0)|/|E(0)| (E(0) = same
operator's eps=0 ri_mp2). eps*(tol) by linear interpolation of log(relerr) vs
log(eps); keep%, pairs, mean domain interpolated linearly in log(eps) between
the bracketing runs. Prints one row per (mol, op, tol).
"""

from __future__ import annotations

import itertools
import math
import sys


def parse(paths):
    ops, rows = {}, {}
    for p in paths:
        with open(p) as fh:
            lines = fh.readlines()
        for ln in lines:
            t = ln.split()
            if not t:
                continue
            if t[0] == "ROWOP":
                ops[(t[1], t[2])] = {
                    "e0": float(t[4]),
                    "ratio": float(t[8]),
                    "t_eri3": float(t[12]),
                    "cost": float(t[16]),
                }
            elif t[0] == "ROW":
                k = (t[1], t[2])
                rows.setdefault(k, []).append(
                    {
                        "eps": float(t[4]),
                        "relerr": float(t[8]),
                        "keep": float(t[10]),
                        "pairs": float(t[12]),
                        "dom": float(t[14]),
                        "wall": float(t[18]),
                    }
                )
    return ops, rows


def interp(rs, tol):
    rs = sorted(rs, key=lambda r: r["eps"])
    for a, b in itertools.pairwise(rs):
        if a["relerr"] <= tol <= b["relerr"] and a["relerr"] > 0:
            f = (math.log(tol) - math.log(a["relerr"])) / (
                math.log(b["relerr"]) - math.log(a["relerr"])
            )
            out = {
                "eps": math.exp(
                    math.log(a["eps"]) + f * (math.log(b["eps"]) - math.log(a["eps"]))
                )
            }
            for k in ("keep", "pairs", "dom"):
                out[k] = a[k] + f * (b[k] - a[k])
            out["wall"] = a["wall"] + f * (b["wall"] - a["wall"])
            return out
    return None


def main(paths):
    ops, rows = parse(paths)
    print("mol op E/Ecoul cost | tol eps* keep% pairs dom_mean")
    for k, rs in rows.items():
        o = ops.get(k, {})
        for tol in (1e-3, 1e-2):
            r = interp(rs, tol)
            tail = (
                f"{r['eps']:.2e} {r['keep']:.2f} {r['pairs']:.0f} {r['dom']:.1f} "
                f"wall {r['wall']:.0f}s"
                if r
                else "not bracketed"
            )
            print(
                f"{k[0]} {k[1]:<26} {o.get('ratio', float('nan')):.4f} "
                f"{o.get('cost', float('nan')):.1f}x | {tol:.0e} {tail}"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
