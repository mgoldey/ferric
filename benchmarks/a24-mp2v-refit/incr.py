#!/usr/bin/env python3
"""attMP2 / MP2-V / VV10-increment vs paper SI tables 17/18 (Addendum A).

  incr.py run --systems 2,4 --factors 1.0 [--nlc 50,50] [--aux atz] --out r.json [--cp]
  incr.py report r.json [r2.json ...]

Published linked parameters (terfc r0=1.00 A, b=11.0, C=0.0089, omega=None), aTZ,
frozen core (Ar=5), DF-JK def2-universal-jkfit, post-HF VV10. Raw fragment
energies are stored; E_int are formed in `report`.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness  # noqa: E402
import stretch  # noqa: E402

K = harness.K
ARMS = [{"r0": 1.00, "r0omega": None}]
BS = [11.0]


def run(a):
    import ferric

    sets = harness.load_sets()["a24"]["systems"]
    out = Path(a.out)
    db = json.loads(out.read_text()) if out.exists() else {}
    nlc = [int(x) for x in a.nlc.split(",")]
    arms = [{"r0": a.r0, "r0omega": None}]
    for sid in [int(x) for x in a.systems.split(",")]:
        rec = sets[str(sid)]
        for f in [float(x) for x in a.factors.split(",")]:
            frs = stretch.frags_for(rec, f, "G1")
            tags = ["dimer", "mA_cp", "mB_cp"] if a.cp else ["dimer"]
            todo = [(f"{sid}|{f}|{t}", frs[t]) for t in tags]
            todo += [(f"{sid}|mono|{t}", stretch.frags_for(rec, 1.0, "G1")[t]) for t in ("mA", "mB")]
            for key, atoms in todo:
                if key in db:
                    continue
                r = harness.run_fragment(
                    ferric, atoms, "atz", arms, BS, nlc, False, df_exact=a.exact_jk,
                )
                db[key] = {
                    "rhf": r["rhf_energy"],
                    "att": r["arms"][0]["att_mp2_corr"],
                    "nl": r["arms"][0]["vv10_e_nl"][0],
                    "fc": r["frozen_core"],
                    "npts": r["n_nlc_points"],
                    "sec": r["seconds"],
                }
                out.write_text(json.dumps(db, indent=1))
                print(key, f"{r['seconds']:.0f}s", flush=True)


def eint(db, sid, f, cp):
    pa, pb = (f"{sid}|{f}|mA_cp", f"{sid}|{f}|mB_cp") if cp else (f"{sid}|mono|mA", f"{sid}|mono|mB")
    d = db[f"{sid}|{f}|dimer"]
    A, B = db[pa], db[pb]
    g = lambda r: (r["rhf"] + r["att"], r["rhf"] + r["att"] + r["nl"])  # noqa: E731
    att = (g(d)[0] - g(A)[0] - g(B)[0]) * K
    v = (g(d)[1] - g(A)[1] - g(B)[1]) * K
    return att, v


def report(paths):
    for p in paths:
        db = json.loads(Path(p).read_text())
        print(f"== {p}")
        print("sys  f    ferric: attMP2   MP2-V    vv10(same r0) | paper: S17(own r0, see Addendum C)  S18 | d(att)  d(V)   [paper S18-S17 is NOT a VV10 increment]")
        keys = sorted({tuple(k.split("|")[:2]) for k in db if "mono" not in k})
        for s, f in keys:
            sid, f = int(s), float(f)
            i = stretch.FACTORS.index(f)
            p17, p18 = stretch.paper("17", sid)[i], stretch.paper("18", sid)[i]
            for cp in (False, True):
                try:
                    att, v = eint(db, sid, f, cp)
                except KeyError:
                    continue
                print(
                    f"{sid:2d} {f:.1f} {'CP ' if cp else 'nCP'} {att:+8.4f} {v:+8.4f} {v-att:+8.4f} | "
                    f"{p17:+8.4f} {p18:+8.4f} | {att-p17:+7.4f} {v-p18:+7.4f}"
                )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--systems", default="2,4,5,8,19")
    ap.add_argument("--factors", default="1.0")
    ap.add_argument("--nlc", default="50,50")
    ap.add_argument("--exact-jk", action="store_true")
    ap.add_argument("--r0", type=float, default=1.00)
    ap.add_argument("--cp", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    else:
        report(a.paths)
