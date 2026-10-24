#!/usr/bin/env python3
"""Planted-parameter recovery for the nested adaptive search (session.py).

The attMP2 half is expensive, so the dry run uses a SURROGATE for it: each
fragment's attMP2 correlation energy is tabulated on a real (ln r0, ln s)
node set (real `Mp2vSession.att_mp2`) and interpolated with a bicubic spline;
the VV10 half is the REAL cheap evaluation on the real stored grid state.
"Experimental" energies are then generated from that same model at a PLANTED
parameter vector plus Gaussian noise at the measured noise floor, and the
unmodified `session.adaptive_search` / `hessian` must recover the planted
vector. The test validates the optimiser, the b-profile rule, the edge logic
and the Hessian-based identifiability numbers; it does NOT validate the
physics (the surrogate is not the real attMP2 surface between table nodes --
interpolation error is measured on held-out real nodes and reported).

Usage: planted.py --set a24 --systems 2,4,5 --basis adz --out results/planted.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import session as S  # noqa: E402

R0_TAB = [round(x, 4) for x in np.geomspace(0.8, 1.6, 4)]
S_TAB = [None, 2.0, 4.0, 6.0]  # None = linked; values are r0*omega


def tabulate(pool, tags, log=print):
    """Real attMP2 on the node set for every fragment -> spline per fragment."""
    splines = {}
    for key, f in pool.frags.items():
        if key[2] not in tags:
            continue
        tab = np.zeros((len(R0_TAB), len(S_TAB)))
        for a, r0 in enumerate(R0_TAB):
            for b, s in enumerate(S_TAB):
                tab[a, b] = f.att(r0, s)
        u = np.log(R0_TAB)
        v = np.log([S._sval(x) for x in S_TAB])
        splines[key] = (RectBivariateSpline(u, v, tab, kx=3, ky=3), tab)
        log(f"tabulated {f.name} ({time.time():.0f})")
    return splines


def install_surrogate(pool, splines):
    """Replace Fragment.att by the spline (HF and E_nl stay real)."""
    for key, f in pool.frags.items():
        if key not in splines:
            continue
        sp = splines[key][0]

        def att(r0, s, _sp=sp):
            return float(_sp(math.log(r0), math.log(S._sval(s)))[0, 0])

        f.att = att


def generate_refs(pool, ids, planted, cp, sigma, rng):
    """Replace pool.systems refs by model(planted) + N(0, sigma) (kcal/mol)."""
    truth = {}
    for sn, i in ids:
        e = pool.eint(planted, sn, i, cp)
        truth[(sn, i)] = e
        pool.systems[(sn, i)] = e + float(rng.normal(0.0, sigma))
    return truth


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="a24")
    ap.add_argument("--systems", required=True)
    ap.add_argument("--basis", default="adz")
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--cp", action="store_true")
    ap.add_argument(
        "--replay-gn",
        action="store_true",
        help="re-use the seeds in --out, recompute only the Hessians (adds Gauss-Newton fields)",
    )
    ap.add_argument("--nlc", default="20,26")
    ap.add_argument("--factors", default="")
    ap.add_argument("--tab", default="5,4", help="n_r0,n_s of the real attMP2 table")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--planted", default="1.15,3.0,13.0")
    a = ap.parse_args(argv)
    raw = [int(x) for x in a.systems.split(",")]
    factors = [float(x) for x in a.factors.split(",")] if a.factors else None
    r0p, sp_, bp = (float(x) for x in a.planted.split(","))
    planted = S.P(r0p, sp_, bp)

    pool = S.Pool(
        a.basis, tuple(int(x) for x in a.nlc.split(",")), a.cache, budget_gb=4.0
    )
    pool.add(a.set, raw, factors)
    ids = sorted(pool.systems, key=str)
    t0 = time.time()
    tags = S.FRAG_CP if a.cp else S.FRAG_NCP
    splines = tabulate(pool, tags)
    pool.cache.save()
    t_tab = time.time() - t0
    # interpolation error on held-out REAL off-grid nodes
    held = [(0.95, 1.3), (1.3, 3.0), (1.0, 5.0)]
    ierr = 0.0
    for key, f in pool.frags.items():
        if key not in splines:
            continue
        for r0, sv in held:
            real = f.att(r0, sv)
            spl = float(splines[key][0](math.log(r0), math.log(sv))[0, 0])
            ierr = max(ierr, abs(real - spl) * S.K)
    install_surrogate(pool, splines)
    rules = dict(S.RULES)
    rules["r0_coarse"] = R0_TAB
    rules["s_coarse"] = S_TAB
    rules["box"] = {"r0": (R0_TAB[0], R0_TAB[-1]), "s": (S._sval(S_TAB[0]), S_TAB[-1])}
    rules["max_extensions"] = 0

    if a.replay_gn:
        res = json.loads(Path(a.out).read_text())
        for sd in res["seeds"]:
            rng = np.random.default_rng(1000 + sd["seed"])
            generate_refs(pool, ids, planted, a.cp, a.sigma, rng)
            hs = S.hessian(pool, ids, sd["best"], a.cp, sigma=a.sigma)
            sd.update({k: v for k, v in hs.items() if k.startswith("gn_")})
            sd["hess_eigs"] = hs["eigenvalues"]
            sd["hess_vecs"] = hs["eigenvectors"]
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))
        return 0
    res = {
        "planted": planted,
        "sigma": a.sigma,
        "cp": a.cp,
        "tabulate_seconds": t_tab,
        "interp_err_kcal_max_per_fragment": ierr,
        "seeds": [],
    }
    for seed in range(a.seeds):
        rng = np.random.default_rng(1000 + seed)
        truth = generate_refs(pool, ids, planted, a.cp, a.sigma, rng)
        out = S.adaptive_search(pool, ids, a.cp, rules=rules, log=lambda *_: None)
        best = out["best"]
        hs = S.hessian(pool, ids, best, a.cp, sigma=a.sigma)
        res["seeds"].append(
            {
                "seed": seed,
                "best": {k: best[k] for k in ("r0", "s", "b", "rmsd", "edge")},
                "n_nodes": out["n_nodes"],
                "edge": out["edge"],
                "hess_eigs": hs["eigenvalues"],
                "hess_vecs": hs["eigenvectors"],
                "one_sigma": hs["one_sigma_ln_units"],
                **{k: v for k, v in hs.items() if k.startswith("gn_")},
                "rmsd_at_planted": float(
                    np.sqrt(np.mean([(truth[k] - pool.systems[k]) ** 2 for k in truth]))
                ),
            }
        )
        print(seed, res["seeds"][-1]["best"], flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    # summary
    arr = np.array(
        [
            [d["best"]["r0"], S._sval(d["best"]["s"]), d["best"]["b"]]
            for d in res["seeds"]
        ]
    )
    lp = np.log([r0p, S._sval(sp_), bp])
    dev = np.log(arr) - lp
    res["summary"] = {
        "mean_ln_dev": dev.mean(0).tolist(),
        "sd_ln_dev": dev.std(0, ddof=1).tolist() if len(arr) > 1 else None,
        "hess_one_sigma_median": np.median(
            [d["one_sigma"] for d in res["seeds"]], axis=0
        ).tolist(),
        "frac_edge": float(
            np.mean([any(v for v in d["edge"].values() if v) for d in res["seeds"]])
        ),
    }
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
