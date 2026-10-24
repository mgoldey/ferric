#!/usr/bin/env python3
"""Stage 0b: pre-registered gate + cost split for the tabulated VV10 damping.

GATE (PREREGISTRATION Addendum G): |E_nl(fast_damping) - E_nl(exact erf)| <= 1e-10 Ha
on the water dimer (A24 #2) and methane dimer (A24 #19), aDZ, NLC (50, 50),
for EVERY damping form (eq11, terf, bare) at the linked seam and at
r0*omega in {2, 4, 8}, two r0 values and two b values.

COST SPLIT: SCF+grid session, one attMP2, and the VV10 pair sum for
(1 seam x 1 b), (1 seam x 9 b), (5 seams x 9 b), EXACT and FAST interleaved,
with uptime before and after every block.

Usage: fastdamp_gate.py --systems 2,19 --out results/fastdamp_gate_adz.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import session as S  # noqa: E402


def load1():
    return os.getloadavg()[0]


def gate_tuples():
    out = []
    for r0 in (1.0, 1.35):
        for s in (None, 2.0, 4.0, 8.0):
            om = None if s is None else s / r0
            for b in (11.0, 6.0):
                out.append((b, 0.0089, "eq11", r0, om))
                out.append((b, 0.0089, "terf", r0, om))
    out.append((11.0, 0.0089, "bare", 1.0, None))
    return out


def timed(f):
    l0, t = load1(), time.time()
    v = f()
    return v, time.time() - t, l0, load1()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="2,19")
    ap.add_argument("--basis", default="adz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = {"basis": a.basis, "nlc": [50, 50], "systems": {}}
    tup = gate_tuples()
    for sid in [int(x) for x in a.systems.split(",")]:
        pool = S.Pool(a.basis, (50, 50), None, budget_gb=4.0)
        pool.add("a24", [sid])
        f = pool.frags[("a24", sid, "dimer")]
        _, t_scf, l0, l1 = timed(f.session)
        sess = f.sess
        _, t_att, _, _ = timed(lambda: sess.att_mp2(1.0, None))
        ex, t_ex, _, _ = timed(lambda: np.array(sess.vv10(tup)))
        fa, t_fa, _, _ = timed(lambda: np.array(sess.vv10(tup, fast_damping=True)))
        err = np.abs(ex - fa)
        by = {}
        for t, e in zip(tup, err):
            by.setdefault(
                f"{t[2]}|{'L' if t[4] is None else format(t[4] * t[3], '.3g')}", []
            ).append(float(e))
        r = {
            "n_pts": sess.n_nlc_points,
            "max_abs_err_Ha": float(err.max()),
            "max_rel_err": float((err / np.abs(ex)).max()),
            "worst_by_form_seam": {k: max(v) for k, v in by.items()},
            "n_tuples": len(tup),
            "gate_pass": bool(err.max() <= 1e-10),
            "E_nl_exact_range": [float(ex.min()), float(ex.max())],
            "t_session_s": t_scf,
            "t_att_mp2_s": t_att,
            "t_gate_exact_s": t_ex,
            "t_gate_fast_s": t_fa,
        }
        # cost split, interleaved exact/fast; each point uses UNSEEN seams so
        # nothing is cached anywhere (the session has no E_nl cache)
        bs9 = list(np.geomspace(2.0, 64.0, 9))
        split = {}
        for name, seams, bs in (
            ("1seam_1b", [(1.0, None)], [11.0]),
            ("1seam_9b", [(1.0, None)], bs9),
            (
                "5seam_9b",
                [(0.8, 2.0), (1.0, 2.0), (1.2, 4.0), (1.4, 2.0), (1.6, 4.0)],
                bs9,
            ),
        ):
            row = {}
            for fast in (False, True, False, True):
                l_a = load1()
                t0 = time.time()
                sess.vv10(
                    [
                        (b, 0.0089, "eq11", r0, None if s is None else s / r0)
                        for r0, s in seams
                        for b in bs
                    ],
                    fast_damping=fast,
                )
                dt = time.time() - t0
                row.setdefault("fast" if fast else "exact", []).append(
                    {"s": dt, "load_before": l_a, "load_after": load1()}
                )
            split[name] = row
        # bare kernel (no damping at all): the floor of the pair sum
        _, t_bare, _, _ = timed(lambda: sess.vv10([(11.0, 0.0089, "bare", 1.0, None)]))
        split["bare_1b_s"] = t_bare
        r["split"] = split
        res["systems"][str(sid)] = r
        print(
            json.dumps(
                {
                    sid: {
                        k: r[k]
                        for k in ("n_pts", "max_abs_err_Ha", "gate_pass", "t_att_mp2_s")
                    },
                    "split": split,
                }
            ),
            flush=True,
        )
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
