#!/usr/bin/env python3
"""Fit / validate MP2-V (r0, omega, b) from harness.py raw JSON. No recomputation.

Usage:
  analyze.py --train results/a24_atz.json [--holdout results/s22_atz.json ...]
             [--convention cp|ncp] [--out analysis_a24_atz.json]

Everything printed is derived from the raw fragment energies; the b grid in the
JSON is the only discretisation (minima are refined by a parabola through the
grid minimum and its neighbours and FLAGGED when the minimum sits on the grid
edge -- an edge minimum is not a fit).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
K = 627.509474

# Chemical class of each A24 system -- MY assignment from the complexes' names
# (psi4 TAGL), NOT taken from the source paper. Pre-registered in
# PREREGISTRATION.md; a reference-value classification is also reported.
A24_CLASS = {
    **{i: "hbond" for i in (1, 2, 3, 4, 5, 9)},
    **{i: "mixed" for i in (6, 7, 8, 10, 11, 12, 13, 14, 16)},
    **{i: "dispersion" for i in (15, 17, 18, 19, 20, 21)},
    **{i: "repulsive_stack" for i in (22, 23, 24)},
}


def load(path):
    return json.loads(Path(path).read_text())


def sys_ids(db):
    return sorted({int(k.split("|")[0].split("-")[1]) for k in db["fragments"]})


def totals(db, key):
    """-> dict with 'hf' (scalar), 'mp2c' (scalar|None), 'mp2v' ndarray (n_arm,n_b)."""
    f = db["fragments"][key]
    att = np.array([a["att_mp2_corr"] for a in f["arms"]])
    nl = np.array([a["vv10_e_nl"] for a in f["arms"]])
    hf = f["rhf_energy"]
    mp2c = f.get("mp2_coulomb_corr")
    return {
        "hf": hf,
        "att": hf + att,  # (n_arm,) HF + attMP2, no VV10
        "mp2c": None if mp2c is None else hf + mp2c,
        "mp2v": hf + att[:, None] + nl,
    }


def interaction(db, convention="cp"):
    """Interaction energies in kcal/mol, per system.

    Returns {sys: {"hf": x, "mp2c": x|None, "att": (n_arm,), "mp2v": (n_arm,n_b)}}.
    convention 'cp': dimer - mA_cp - mB_cp ; 'ncp': dimer - mA - mB.
    """
    pa, pb = ("mA_cp", "mB_cp") if convention == "cp" else ("mA", "mB")
    prefix = db["meta"]["set"]
    out = {}
    for i in sys_ids(db):
        keys = [f"{prefix}-{i:02d}|{t}" for t in ("dimer", pa, pb)]
        if not all(k in db["fragments"] for k in keys):
            continue
        d, a, b = (totals(db, k) for k in keys)
        rec = {"hf": (d["hf"] - a["hf"] - b["hf"]) * K,
               "att": (d["att"] - a["att"] - b["att"]) * K,
               "mp2v": (d["mp2v"] - a["mp2v"] - b["mp2v"]) * K}  # fmt: skip
        rec["mp2c"] = (
            None
            if d["mp2c"] is None
            else (d["mp2c"] - a["mp2c"] - b["mp2c"]) * K
        )
        out[i] = rec
    return out


def refs(setname):
    s = json.loads((HERE / "sets.json").read_text())[setname]["systems"]
    return {int(i): v["ref"] for i, v in s.items()}


def metrics(err):
    err = np.asarray(err, float)
    if err.size == 0:
        return {"n": 0}
    return {
        "n": int(err.size),
        "rmsd": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "mse": float(np.mean(err)),
        "max": float(np.max(np.abs(err))),
    }


def err_matrix(inter, ref, ids, arm):
    """(len(ids), n_b) error matrix of arm `arm` vs reference."""
    return np.array([inter[i]["mp2v"][arm] - ref[i] for i in ids])


def refine_min(bs, f):
    """argmin of f over grid bs, parabola-refined. -> (b*, f(b*), on_edge)."""
    bs = np.asarray(bs, float)
    k = int(np.argmin(f))
    edge = k in (0, len(bs) - 1)
    if edge:
        return float(bs[k]), float(f[k]), True
    x, y = bs[k - 1 : k + 2], f[k - 1 : k + 2]
    c = np.polyfit(x, y, 2)
    if c[0] <= 0:
        return float(bs[k]), float(f[k]), False
    xb = -c[1] / (2 * c[0])
    xb = min(max(xb, x[0]), x[2])
    return float(xb), float(np.polyval(c, xb)), False


def rmsd_curve(E):
    return np.sqrt(np.mean(E**2, axis=0))


def fit_arm(inter, ref, ids, arm, bs):
    E = err_matrix(inter, ref, ids, arm)
    b, r, edge = refine_min(bs, rmsd_curve(E))
    return {"b": b, "rmsd": r, "edge": edge}


def predict_at_b(inter, ref, ids, arm, bs, b):
    """Errors of arm at (possibly off-grid) b by linear interpolation in b."""
    out = []
    bs = np.asarray(bs, float)
    for i in ids:
        out.append(np.interp(b, bs, inter[i]["mp2v"][arm]) - ref[i])
    return np.array(out)


def loso(inter, ref, ids, arm, bs):
    """Leave-one-system-out: refit b without system j, predict j."""
    preds, bstar = [], []
    for j in ids:
        rest = [i for i in ids if i != j]
        f = fit_arm(inter, ref, rest, arm, bs)
        bstar.append(f["b"])
        preds.append(predict_at_b(inter, ref, [j], arm, bs, f["b"])[0])
    return np.array(preds), np.array(bstar)


def by_class(err, ids, cls):
    out = {}
    for c in sorted(set(cls.values())):
        sel = [k for k, i in enumerate(ids) if cls.get(i) == c]
        out[c] = metrics(np.asarray(err)[sel])
    return out


def analyze(train, holdouts, convention, class_map=None):
    meta = train["meta"]
    arms, bs = meta["arms"], np.array(meta["bs"], float)
    ref = refs(meta["set"])
    inter = interaction(train, convention)
    ids = sorted(inter)
    res = {"convention": convention, "set": meta["set"], "basis": meta["basis"],
           "n_systems": len(ids), "ids": ids, "bs": bs.tolist(), "arms": []}  # fmt: skip
    base = {}
    if all(inter[i]["mp2c"] is not None for i in ids):
        base["mp2_coulomb"] = metrics([inter[i]["mp2c"] - ref[i] for i in ids])
    base["hf"] = metrics([inter[i]["hf"] - ref[i] for i in ids])
    res["baselines"] = base
    for a, arm in enumerate(arms):
        e_att = [inter[i]["att"][a] - ref[i] for i in ids]
        fit = fit_arm(inter, ref, ids, a, bs)
        err_fit = predict_at_b(inter, ref, ids, a, bs, fit["b"])
        pl, bstar = loso(inter, ref, ids, a, bs)
        rec = {
            "arm": arm,
            "att_only": metrics(e_att),
            "fit": fit,
            "train_at_fit": metrics(err_fit),
            "loso": {**metrics(pl), "b_min": float(bstar.min()),
                     "b_max": float(bstar.max()), "b_sd": float(bstar.std()),
                     "worst_leverage_system": int(ids[int(np.argmax(np.abs(bstar - fit["b"])))]),
                     "b_by_left_out": dict(zip(map(int, ids), bstar.tolist()))},
            "rmsd_curve": rmsd_curve(err_matrix(inter, ref, ids, a)).tolist(),
        }  # fmt: skip
        if class_map:
            rec["by_class"] = by_class(err_fit, ids, class_map)
        rec["holdout"] = {}
        for h in holdouts:
            hi = interaction(h, convention)
            hr = refs(h["meta"]["set"])
            hids = sorted(hi)
            if h["meta"]["arms"] != arms or h["meta"]["bs"] != meta["bs"]:
                raise SystemExit("holdout arms/bs differ from train")
            eh = predict_at_b(hi, hr, hids, a, bs, fit["b"])
            rec["holdout"][h["meta"]["set"]] = {
                **metrics(eh),
                "own_optimum": fit_arm(hi, hr, hids, a, bs),
            }
        res["arms"].append(rec)
    return res


def fmt_table(res):
    L = [f"# {res['set']} {res['basis']} convention={res['convention']} n={res['n_systems']}"]
    L.append("baselines: " + "; ".join(
        f"{k}: RMSD {v['rmsd']:.3f} MAE {v['mae']:.3f} max {v['max']:.3f}"
        for k, v in res["baselines"].items()))  # fmt: skip
    L.append(
        "r0   r0*w   b*     edge  RMSD   MAE    max    MSE   | LOSO RMSD  b-range     | att-only RMSD"
    )
    for r in res["arms"]:
        a, f, t, lo = r["arm"], r["fit"], r["train_at_fit"], r["loso"]
        rw = "linked" if a["r0omega"] is None else f"{a['r0omega']:.2f}"
        L.append(
            f"{a['r0']:.2f} {rw:>6s} {f['b']:6.2f} {'EDGE' if f['edge'] else '    '} "
            f"{t['rmsd']:.3f} {t['mae']:.3f} {t['max']:.3f} {t['mse']:+.3f} | "
            f"{lo['rmsd']:.3f}   [{lo['b_min']:.1f},{lo['b_max']:.1f}] | {r['att_only']['rmsd']:.3f}"
        )
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--holdout", nargs="*", default=[])
    ap.add_argument("--convention", default="cp", choices=["cp", "ncp"])
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    train = load(a.train)
    cm = A24_CLASS if train["meta"]["set"] == "a24" else None
    res = analyze(train, [load(h) for h in a.holdout], a.convention, cm)
    print(fmt_table(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
