"""v2 arms: per-system tables, ratios vs old-stable at matched eps and matched keep."""
import json, sys
from pathlib import Path
import numpy as np

ARMS = ["old", "old-stable", "screen", "lambda", "new", "new-r2"]
FLOOR = 1e-11  # roundoff (anchor level): ratios meaningless
MODELS = ["e2", "e3", "op-mp3", "ssd-opmp3"]


def rows(res):
    return [r for r in res["rows"] if r["mode"] == "mask"]


def at_keep(res, keep, m):
    """log|err| interpolated linearly in keep (sorted); None outside range."""
    rr = sorted(rows(res), key=lambda r: r["keep"])
    k = np.array([r["keep"] for r in rr])
    e = np.array([abs(r["err"][m]) for r in rr])
    ok = e > FLOOR
    k, e = k[ok], e[ok]
    if len(k) < 2 or keep < k[0] or keep > k[-1]:
        return None
    return float(np.exp(np.interp(keep, k, np.log(e))))


def main(paths):
    for p in paths:
        data = json.loads(Path(p).read_text())
        by = {r["arm"]: r for r in data if isinstance(r, dict) and "rows" in r}
        if not by:
            continue
        r0 = next(iter(by.values()))
        ec = r0["canonical"]["e2"]
        print(f"\n### {Path(p).stem}: no={r0['no']} nv={r0['nv']} E2={ec:.6f} "
              f"zeta={r0['zeta_can']:.3f}; anchors " +
              " ".join(f"{a}={by[a]['anchor_eps0']:.1e}" for a in ARMS if a in by))
        lb = by.get("new", {}).get("lambda_boys") or {}
        if lb:
            print(f"lambda-Boys: {lb['sweeps']} sweeps, offdiag F_oo^2 {lb['phi_boys']:.3e} -> "
                  f"{lb['phi_lambda']:.3e}, max||U|-1| {lb['max_abs_u_minus_1']:.3f}; "
                  f"spread occ old-stable {by['old-stable']['orbital_spread']['occ']:.3f} "
                  f"new {by['new']['orbital_spread']['occ']:.3f} Bohr^2")
        print("\n| eps | arm | keep | dE2 | dE3 | dOP | dSSD | a(OP) | pred/OP | E3gap/bound |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        eps_all = sorted({r["eps"] for r in rows(r0)}, reverse=True)
        for eps in eps_all:
            for a in ARMS:
                if a not in by:
                    continue
                r = next(x for x in rows(by[a]) if x["eps"] == eps)
                e = r["err"]
                sc = abs(e["e2"]) + abs(e["e3"])
                pr = r["pred"]["op-mp3"] / e["op-mp3"] if e["op-mp3"] else float("nan")
                gb = abs(r["e3_identity_gap"]) / r["e3_identity_bound"] if r["e3_identity_bound"] else 0
                print(f"| {eps:.0e} | {a} | {r['keep']:.3f} | {e['e2']:+.2e} | {e['e3']:+.2e} | "
                      f"{e['op-mp3']:+.2e} | {e['ssd-opmp3']:+.2e} | {abs(e['op-mp3'])/sc:.2f} | "
                      f"{pr:.3f} | {gb:.2f} |")
        base = by["old-stable"]
        print("\nratio |err(arm)|/|err(old-stable)| at matched eps (OP-MP3; E2 in brackets)")
        print("\n| eps | " + " | ".join(a for a in ARMS if a != "old-stable" and a in by) + " |")
        print("|---|" + "---|" * (len(by) - 1))
        for eps in eps_all:
            cells = []
            b = next(x for x in rows(base) if x["eps"] == eps)["err"]
            for a in ARMS:
                if a == "old-stable" or a not in by:
                    continue
                e = next(x for x in rows(by[a]) if x["eps"] == eps)["err"]
                if max(abs(b["op-mp3"]), abs(e["op-mp3"])) < FLOOR:
                    cells.append("floor"); continue
                cells.append(f"{abs(e['op-mp3'])/abs(b['op-mp3']):.3f} [{abs(e['e2'])/abs(b['e2']):.3f}]")
            print(f"| {eps:.0e} | " + " | ".join(cells) + " |")
        print("\nmatched keep (log-interp of old-stable at the arm's keep): ratio OP-MP3 [E2]")
        print("\n| eps | " + " | ".join(a for a in ARMS if a != "old-stable" and a in by) + " |")
        print("|---|" + "---|" * (len(by) - 1))
        for eps in eps_all:
            cells = []
            for a in ARMS:
                if a == "old-stable" or a not in by:
                    continue
                r = next(x for x in rows(by[a]) if x["eps"] == eps)
                if abs(r["err"]["op-mp3"]) < FLOOR:
                    cells.append("floor"); continue
                bo, b2 = at_keep(base, r["keep"], "op-mp3"), at_keep(base, r["keep"], "e2")
                cells.append("—" if bo is None else
                             f"{abs(r['err']['op-mp3'])/bo:.3f} [{abs(r['err']['e2'])/b2:.3f}]")
            print(f"| {eps:.0e} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
