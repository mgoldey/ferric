"""Markdown fair-criterion table (operator x {eps*, keep%, pairs, domain,
E_att/E_Coulomb, kernel range R(0.1), per-quartet cost}) from harness logs.

usage: python md_table.py COST_LOG LOG [LOG ...]
COST_LOG is a REPS>=3 SKIP_RI=1 timing run (min-of-reps eri3 wall); the
cost_ratio in the other logs is single-shot and contention-noisy.
"""

from __future__ import annotations

import sys
from pathlib import Path

from scipy.optimize import brentq
from scipy.special import erf, erfc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A


def r_at(spec: str, f: float) -> float:
    """Distance (Angstrom) where the attenuation function drops to f."""
    p = spec.split(":")
    if p[0] == "coul":
        return float("inf")
    if p[0] == "erfc":
        return brentq(lambda r: erfc(float(p[1]) * r) - f, 1e-6, 60)
    r0, w = float(p[1]), float(p[2]) / float(p[1])
    return brentq(
        lambda r: 1 - (erf(w * (r - r0)) + erf(w * (r + r0))) / 2 - f, 1e-6, 400
    )


def main(cost_log, logs):
    cost, _ = {}, None
    ops_c, _ = A.parse([cost_log])
    for k, v in ops_c.items():
        cost[k[1]] = v["cost"]
    ops, rows = A.parse(logs)
    print(
        "| mol | operator | E_att/E_Coul | R(0.1) A | cost vs Coul | tol | eps* | "
        "keep % | pairs | dom | wall s |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (m, spec), rs in rows.items():
        for tol in (1e-3, 1e-2):
            r = A.interp(rs, tol)
            c = cost.get(spec)
            cs = f"{c:.1f}x" if c is not None else "-"
            if not r:
                continue
            print(
                f"| {m[:-4]} | {spec} | {ops[(m, spec)]['ratio']:.3f} | "
                f"{r_at(spec, 0.1):.2f} | {cs} | {tol:g} | {r['eps']:.2e} | "
                f"{r['keep']:.2f} | {r['pairs']:.0f} | {r['dom']:.1f} | {r['wall']:.0f} |"
            )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
