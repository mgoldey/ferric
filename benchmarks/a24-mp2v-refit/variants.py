#!/usr/bin/env python3
"""Score alternative short-range VV10 weights (PREREGISTRATION.md Addendum A, Step 4).

Non-CP, rigid monomers (G1 geometry), aTZ, published linked r0=1.00 A, b=11.0, C=0.0089.
E_int(variant) = [E_HF + E_attMP2(r0=1.00)]_int + [E_nl(variant)]_int, with the HF+attMP2
parts read from incr*.json (same SCFs) and E_nl from `run_vv10_variants` (own SCF, same
settings). Compared with paper table 18 (MP2-V) at each factor.

  variants.py run --systems 2,4,5,8,19 --factors 1.0 --out v.json
  variants.py report v.json incr_f1.json [incr_stretch.json]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness  # noqa: E402
import stretch  # noqa: E402

K = harness.K
SPECS = [
    ("none", 1.0, None, 1),     # bare VV10
    ("eq11", 1.0, None, 1),     # V_C (current Eq. 11 reading)
    ("terf", 1.0, None, 1),     # V_A
    ("terf", 1.0, None, 2),     # V_B
    ("eq11", 5.29177e-4, None, 1),  # anchor: r0 -> 0 (1e-3 Bohr) == bare
]
NAMES = ["bare", "V_C", "V_A", "V_B", "anchor"]


def run(a):
    import ferric

    sets = harness.load_sets()["a24"]["systems"]
    out = Path(a.out)
    db = json.loads(out.read_text()) if out.exists() else {}
    obs = ferric.BasisSet.bundled("aug-cc-pvtz")
    nlc = tuple(int(x) for x in a.nlc.split(","))
    for sid in [int(x) for x in a.systems.split(",")]:
        rec = sets[str(sid)]
        for f in [float(x) for x in a.factors.split(",")]:
            todo = [(f"{sid}|{f}|dimer", stretch.frags_for(rec, f, "G1")["dimer"])]
            todo += [(f"{sid}|mono|{t}", stretch.frags_for(rec, 1.0, "G1")[t]) for t in ("mA", "mB")]
            for key, atoms in todo:
                if key in db and db[key].get("nlc") == list(nlc):
                    continue
                mol = ferric.Molecule.from_xyz_string(harness.xyz_text(atoms), 0, 1)
                e = ferric.run_vv10_variants(mol, obs, SPECS, b=11.0, c=0.0089, nlc_grid=nlc)
                db[key] = {"nlc": list(nlc), "e": dict(zip(NAMES, e))}
                out.write_text(json.dumps(db, indent=1))
                print(key, flush=True)


def terf(r_ang, r0=1.0):
    """terf(R; r0) with the linked width, R and r0 in Angstrom."""
    s = r0 * math.sqrt(2.0)
    return 0.5 * (math.erf((r_ang - r0) / s) + math.erf((r_ang + r0) / s))


def rcc_ang(sid, f):
    rec = harness.load_sets()["a24"]["systems"][str(sid)]
    fa, fb = stretch.stretched(rec, f, "G1")
    return float(np.linalg.norm(stretch.com(fb) - stretch.com(fa)))


def report(vpath, ipaths):
    v = json.loads(Path(vpath).read_text())
    inc = {}
    for p in ipaths:
        inc.update(json.loads(Path(p).read_text()))
    keys = sorted({tuple(k.split("|")[:2]) for k in v if "mono" not in k})
    print("sys f   | paper S18 | V_C err  V_A err  V_B err  bare err  V_D err | incr(V_C-bare) vs paper(S18-S17)")
    for s, f in keys:
        sid, f = int(s), float(f)
        i = stretch.FACTORS.index(f)
        d, A, B = inc[f"{sid}|{f}|dimer"], inc[f"{sid}|mono|mA"], inc[f"{sid}|mono|mB"]
        base = ((d["rhf"] + d["att"]) - (A["rhf"] + A["att"]) - (B["rhf"] + B["att"])) * K
        e = lambda n: (v[f"{sid}|{f}|dimer"]["e"][n] - v[f"{sid}|mono|mA"]["e"][n] - v[f"{sid}|mono|mB"]["e"][n]) * K  # noqa: E731
        tot = {n: base + e(n) for n in ("V_C", "V_A", "V_B", "bare")}
        w = terf(rcc_ang(sid, f))
        tot["V_D"] = base + e("bare") * w
        p18 = stretch.paper("18", sid)[i]
        anchor = (v[f"{sid}|{f}|dimer"]["e"]["anchor"] - v[f"{sid}|{f}|dimer"]["e"]["bare"])
        print(
            f"{sid:2d} {f:.1f} | {p18:+8.4f} | "
            + " ".join(f"{tot[n]-p18:+8.4f}" for n in ("V_C", "V_A", "V_B", "bare", "V_D"))
            + f" | inc_ferric(V_C)={e('V_C'):+.4f} bare={e('bare'):+.4f} V_A={e('V_A'):+.4f} V_B={e('V_B'):+.4f}"
            + f" | anchor dimer {anchor:+.2e} Ha, terf(Rcc)={w:.3f}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--systems", default="2,4,5,8,19")
    ap.add_argument("--factors", default="1.0")
    ap.add_argument("--nlc", default="50,50")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    else:
        report(a.paths[0], a.paths[1:])
