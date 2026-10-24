#!/usr/bin/env python3
"""Stage 1 production driver (PREREGISTRATION.md Addendum F + G).

Detached, resumable. Every attMP2 and E_nl evaluation is cached to --cache
(JSON keyed by fragment|params), every phase's output is written to --outdir
and skipped on re-run, and a heartbeat file is rewritten after every node:
re-running the same command after any interruption resumes where it stopped.

Phases (per convention in --conventions, default "ncp,cp"):
  search   adaptive nested search (r0, s, b), lockstep Eq. 11, C fixed
  valley   linked line r0*omega = 1/sqrt2: optimal b at r0 = 0.85 ... 1.10 (anchor A1-lite)
  curv     Gauss-Newton J^T J at the optimum (h = 0.10 and 0.05)
  base     baselines: Coulomb MP2, published linked MP2-V, published at sharp seams
           (s = 4, 8) unrefit, b-only refit at the published seam
Then `summary.json`.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import session as S  # noqa: E402
from analyze import A24_CLASS  # noqa: E402

SYSTEMS = [2, 4, 5, 6, 8, 19, 20, 24]
VALLEY_R0 = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10]  # paper Table 1
VALLEY_B = [8.0, 9.0, 9.5, 11.0, 12.5, 14.5]
SIGMA = 0.055  # kcal/mol: sqrt(0.02^2 RI + 0.05^2 reference + grid^2), F.8


class Beat:
    """Heartbeat + plain log; both survive the process."""

    def __init__(self, outdir):
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.state = {}

    def log(self, msg):
        line = f"{time.strftime('%H:%M:%S')} load={os.getloadavg()[0]:.1f} {msg}"
        print(line, flush=True)
        with open(self.dir / "stage1.log", "a") as f:
            f.write(line + "\n")

    def beat(self, **kw):
        self.state.update(kw)
        self.state.update(
            wall_s=time.time() - self.t0,
            load1=os.getloadavg()[0],
            stamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        tmp = self.dir / "heartbeat.json.tmp"
        tmp.write_text(json.dumps(self.state, indent=1, default=float))
        os.replace(tmp, self.dir / "heartbeat.json")


def done(path):
    return Path(path).exists()


def dump(path, obj):
    tmp = str(path) + ".tmp"
    Path(tmp).write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def per_system(pool, ids, r0, s, b, cp):
    e = pool.errors_b(ids, r0, s, [b], cp)[:, 0]
    return {f"{sn}-{i}": float(x) for (sn, i), x in zip(ids, e)}, e


def class_table(ids, e):
    out = {}
    for cls in sorted(set(A24_CLASS[i] for _, i in ids)):
        sel = np.array([A24_CLASS[i] == cls for _, i in ids])
        out[cls] = {
            "n": int(sel.sum()),
            "rmsd": float(np.sqrt((e[sel] ** 2).mean())),
            "mse": float(e[sel].mean()),
        }
    return out


def stats(e):
    return {
        "rmsd": float(np.sqrt((e**2).mean())),
        "mse": float(e.mean()),
        "mae": float(np.abs(e).mean()),
        "max_abs": float(np.abs(e).max()),
    }


def run_search(pool, ids, conv, out, beat, cap):
    cp = conv == "cp"
    nodes_path = out / f"nodes_{conv}.json"
    seen = []
    t_start = time.time()

    def on_node(nd, n):
        _, e = per_system(pool, ids, nd["r0"], nd["s"], nd["b"], cp)
        rec = {
            "r0": nd["r0"],
            "s": S._sval(nd["s"]),
            "linked": nd["s"] is None,
            "b": nd["b"],
            "rmsd": nd["rmsd"],
            "b_edge": nd["edge"],
            "err_by_system": [float(x) for x in e],
        }
        seen.append(rec)
        dump(nodes_path, {"ids": [list(x) for x in ids], "nodes": seen})
        pool.cache.save()
        el = time.time() - t_start
        beat.beat(
            phase=f"search_{conv}",
            nodes_done=n,
            node_cap=cap,
            eta_upper_s=el / max(n, 1) * (cap - n),
            best_rmsd_so_far=min(r["rmsd"] for r in seen),
            n_mp2=pool.n_mp2,
        )
        beat.log(
            f"[{conv}] node {n}/{cap} r0={nd['r0']:.3f} s={S._sval(nd['s']):.3f} "
            f"b*={nd['b']:.2f} rmsd={nd['rmsd']:.4f}"
        )

    rules = dict(S.RULES)
    rules["max_nodes"] = cap
    res = S.adaptive_search(
        pool, ids, cp, rules=rules, log=lambda *_: None, on_node=on_node
    )
    pool.cache.save()
    best = res["best"]
    dump(
        out / f"search_{conv}.json",
        {
            "best": best,
            "edge": res["edge"],
            "n_nodes": res["n_nodes"],
            "rounds": res["rounds"],
            "budget_limited": res["n_nodes"] >= cap,
        },
    )
    return res


def run_valley(pool, ids, conv, out, beat):
    cp = conv == "cp"
    rows = []
    for r0, bp in zip(VALLEY_R0, VALLEY_B):
        S.prefetch_stage(pool, ids, [(r0, None)], cp)
        d = S.optimal_b(pool, ids, r0, None, cp)
        rows.append(
            {
                "r0": r0,
                "b_opt": d["b"],
                "rmsd": d["rmsd"],
                "edge": d["edge"],
                "b_paper": bp,
            }
        )
        beat.log(
            f"[{conv}] valley r0={r0} b*={d['b']:.2f} (paper {bp}) rmsd={d['rmsd']:.4f}"
        )
    pool.cache.save()
    x, y = np.log([r["r0"] for r in rows]), np.log([r["b_opt"] for r in rows])
    slope = float(np.polyfit(x, y, 1)[0])
    xp = np.log(VALLEY_B)
    slope_paper = float(np.polyfit(x, xp, 1)[0])
    mono = bool(np.all(np.diff([r["b_opt"] for r in rows]) > 0))
    dump(
        out / f"valley_{conv}.json",
        {
            "rows": rows,
            "dlnb_dlnr0": slope,
            "dlnb_dlnr0_paper_table1": slope_paper,
            "b_increasing": mono,
        },
    )


def run_curv(pool, ids, conv, out, beat):
    cp = conv == "cp"
    best = json.loads((out / f"search_{conv}.json").read_text())["best"]
    r = {}
    for h in (0.10, 0.05):
        hs = S.hessian(pool, ids, best, cp, h=h, sigma=SIGMA)
        r[f"h{h}"] = hs
        beat.log(
            f"[{conv}] GN h={h} eig={np.round(hs['gn_eigenvalues'], 4).tolist()} 1sig_param={np.round(hs['gn_one_sigma_param'], 4).tolist()}"
        )
    pool.cache.save()
    dump(out / f"curv_{conv}.json", r)


def run_base(pool, ids, conv, out, beat):
    cp = conv == "cp"
    best = json.loads((out / f"search_{conv}.json").read_text())["best"]
    res = {}
    e_coul = np.array(
        [pool.eint_coulomb(sn, i, cp) - pool.systems[(sn, i)] for sn, i in ids]
    )
    res["mp2_coulomb"] = {**stats(e_coul), "class": class_table(ids, e_coul)}
    for name, (r0, s, b) in {
        "published_linked_r1.00_b11": (1.0, None, 11.0),
        "published_sharp_s4_unrefit": (1.0, 4.0, 11.0),
        "published_sharp_s8_unrefit": (1.0, 8.0, 11.0),
    }.items():
        _, e = per_system(pool, ids, r0, s, b, cp)
        res[name] = {**stats(e), "class": class_table(ids, e)}
    S.prefetch_stage(pool, ids, [(1.0, None)], cp)
    d = S.optimal_b(pool, ids, 1.0, None, cp)
    _, e = per_system(pool, ids, 1.0, None, d["b"], cp)
    res["b_only_refit_r1.00_linked"] = {
        "b": d["b"],
        "edge": d["edge"],
        **stats(e),
        "class": class_table(ids, e),
    }
    _, e = per_system(pool, ids, best["r0"], best["s"], best["b"], cp)
    res["full_refit"] = {
        "r0": best["r0"],
        "s": S._sval(best["s"]),
        "b": best["b"],
        **stats(e),
        "class": class_table(ids, e),
        "err_by_system": {f"{sn}-{i}": float(x) for (sn, i), x in zip(ids, e)},
    }
    pool.cache.save()
    dump(out / f"base_{conv}.json", res)
    for k, v in res.items():
        beat.log(f"[{conv}] {k}: rmsd={v['rmsd']:.4f} mse={v['mse']:+.4f}")


def summarize(out, convs):
    s = {}
    for conv in convs:
        row = {}
        for k in ("search", "valley", "curv", "base"):
            f = out / f"{k}_{conv}.json"
            if f.exists():
                row[k] = json.loads(f.read_text())
        s[conv] = row
    if "ncp" in s and "cp" in s and "search" in s["ncp"] and "search" in s["cp"]:
        a, b = s["ncp"]["search"]["best"], s["cp"]["search"]["best"]
        s["cp_minus_ncp_shift_ln"] = {
            "r0": math.log(b["r0"] / a["r0"]),
            "s": math.log(S._sval(b["s"]) / S._sval(a["s"])),
            "b": math.log(b["b"] / a["b"]),
        }
    dump(out / "summary.json", s)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default=",".join(map(str, SYSTEMS)))
    ap.add_argument("--basis", default="adz")
    ap.add_argument("--cache", default=str(HERE / "results/stage1_cache_adz.json"))
    ap.add_argument("--outdir", default=str(HERE / "results/stage1"))
    ap.add_argument("--conventions", default="ncp,cp")
    ap.add_argument("--no-fast-damping", action="store_true")
    ap.add_argument("--cap", type=int, default=80)
    ap.add_argument("--budget-gb", type=float, default=6.0)
    a = ap.parse_args(argv)
    out = Path(a.outdir)
    beat = Beat(out)
    beat.log(
        f"start systems={a.systems} basis={a.basis} fast_damping={not a.no_fast_damping}"
    )
    pool = S.Pool(
        a.basis, (50, 50), a.cache, a.budget_gb, fast_damping=not a.no_fast_damping
    )
    pool.add("a24", [int(x) for x in a.systems.split(",")])
    ids = sorted(pool.systems)
    convs = a.conventions.split(",")
    for conv in convs:
        if not done(out / f"search_{conv}.json"):
            run_search(pool, ids, conv, out, beat, a.cap)
        for name, fn in (
            ("valley", run_valley),
            ("curv", run_curv),
            ("base", run_base),
        ):
            beat.beat(phase=f"{name}_{conv}")
            if not done(out / f"{name}_{conv}.json"):
                fn(pool, ids, conv, out, beat)
        summarize(out, convs)
    beat.beat(phase="complete", n_mp2=pool.n_mp2)
    beat.log("complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
