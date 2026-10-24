#!/usr/bin/env python3
"""NLC-grid sensitivity of the VV10 contribution to the INTERACTION energy.

For each grid (n_radial, n_angular) and system, E_nl of dimer / monomers (CP
and non-CP fragments) -> Delta E_nl in kcal/mol for several damping specs, and
the deviation from the largest grid. This decides the production NLC grid
(PREREGISTRATION.md Addendum F: required |change| < 0.005 kcal/mol in the
interaction energy). attMP2 does not depend on the NLC grid, so only E_nl moves.

Usage: gridsens.py --set a24 --systems 2,19 --basis adz --out results/gridsens_adz.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import session as S  # noqa: E402

GRIDS = [(30, 50), (50, 50), (50, 110), (75, 50), (30, 110), (30, 302)]
# (b, C, form, r0_vv, omega_vv) -- seams here are VV10-only (this study does no MP2)
SPECS = {
    "eq11_linked_r1.00_b11": (11.0, 0.0089, "eq11", 1.0, None),
    "eq11_s4_r1.00_b11": (11.0, 0.0089, "eq11", 1.0, 4.0 / 1.0),
    "terf_linked_r1.00_b11": (11.0, 0.0089, "terf", 1.0, None),
    "bare_b11": (11.0, 0.0089, "bare", 1.0, None),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="a24")
    ap.add_argument("--systems", required=True)
    ap.add_argument("--basis", default="adz")
    ap.add_argument("--grids", default="")
    ap.add_argument("--ncp-only", action="store_true", help="skip ghost fragments")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    grids = (
        [tuple(int(x) for x in g.split("x")) for g in a.grids.split(",")]
        if a.grids
        else GRIDS
    )
    ids = [int(x) for x in a.systems.split(",")]
    out = {
        "basis": a.basis,
        "specs": {k: list(v) for k, v in SPECS.items()},
        "rows": [],
    }
    tuples = list(SPECS.values())
    for g in grids:
        pool = S.Pool(a.basis, g, None, budget_gb=3.0)
        pool.add(a.set, ids)
        for i in ids:
            t0 = time.time()
            en = {
                tag: pool.frags[(a.set, i, tag)].nl(tuples)
                for tag in (S.FRAG_NCP if a.ncp_only else S.FRAG_CP + ("mA", "mB"))
            }
            npts = pool.frags[(a.set, i, "dimer")].session().n_nlc_points
            row = {
                "system": f"{a.set}-{i:02d}",
                "grid": list(g),
                "n_pts_dimer": npts,
                "seconds": time.time() - t0,
            }
            for k, name in enumerate(SPECS):
                row[name] = {
                    "ncp_kcal": (en["dimer"][k] - en["mA"][k] - en["mB"][k]) * S.K,
                    "dimer_Eh": float(en["dimer"][k]),
                }
                if not a.ncp_only:
                    row[name]["cp_kcal"] = (
                        en["dimer"][k] - en["mA_cp"][k] - en["mB_cp"][k]
                    ) * S.K
            out["rows"].append(row)
            print(row["system"], g, npts, f"{row['seconds']:.0f}s", flush=True)
            Path(a.out).write_text(json.dumps(out, indent=1))
            pool.drop(a.set, i)
    return 0


if __name__ == "__main__":
    sys.exit(main())
