#!/usr/bin/env python3
"""attMP2 (VV10 half unused) vs paper table 17: scan r0 (linked) and r0*omega at r0=1.00
(PREREGISTRATION.md Addendum C).
  att_scan.py --systems 2,4,5,8,19 --factors 1.0 --out r.json
Non-CP, G1 geometry, aTZ, frozen core. One SCF per fragment, all arms in one call."""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness  # noqa: E402
import stretch  # noqa: E402

K = harness.K
ARMS = [{"r0": r, "r0omega": None} for r in (1.00, 1.25, 1.30, 1.35, 1.40, 1.45)]
ARMS += [{"r0": 1.00, "r0omega": w} for w in (0.25, 0.35, 0.5, 1.0)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="2,4,5,8,19")
    ap.add_argument("--factors", default="1.0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import ferric

    sets = harness.load_sets()["a24"]["systems"]
    out = Path(a.out)
    db = json.loads(out.read_text()) if out.exists() else {}
    for sid in [int(x) for x in a.systems.split(",")]:
        rec = sets[str(sid)]
        for f in [float(x) for x in a.factors.split(",")]:
            todo = [(f"{sid}|{f}|dimer", stretch.frags_for(rec, f, "G1")["dimer"])]
            todo += [
                (f"{sid}|mono|{t}", stretch.frags_for(rec, 1.0, "G1")[t])
                for t in ("mA", "mB")
            ]
            for key, atoms in todo:
                if key in db:
                    continue
                r = harness.run_fragment(
                    ferric, atoms, "atz", ARMS, [11.0], (20, 26), False
                )
                db[key] = {
                    "rhf": r["rhf_energy"],
                    "att": [x["att_mp2_corr"] for x in r["arms"]],
                }
                out.write_text(json.dumps(db, indent=1))
                print(key, flush=True)
