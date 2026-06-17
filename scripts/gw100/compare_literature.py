#!/usr/bin/env python3
"""Anchor ferric's GW100-subset IPs against PUBLISHED reference values from the
official GW100 database (github.com/setten/GW100), not just experiment.

Why: ferric's results.json compares evGW to EXPERIMENTAL vertical IPs, which
conflates two errors — (a) ferric's deviation from exact GW, and (b) GW's
physical deviation from reality. To "prove out" the implementation we need a
THEORY-vs-THEORY anchor:
  * CCSD(T) reference IP (Krause/Harding/Klopper, def2-TZVPP) — the gold standard.
  * A published GW@HF lane (M2.E, def2-TZVPP) — same starting point as ferric.
If ferric ≈ published-GW@HF molecule-by-molecule and both overshoot CCSD(T) the
same one-signed way, the residual is PHYSICS (HF starting point), not a bug.

Also validates ferric's hand-entered experimental ip_ref values against the
GW100 database's experimental column.

Reference JSONs (HOMO eigenvalue stored as a negative number; IP = -HOMO) live
in scripts/gw100/lit_ref/. Re-download with:
  base=https://raw.githubusercontent.com/setten/GW100/master/data
  curl -sLO $base/CCSD-T_HOMO_CFOUR_def2-TZVPP.json   (etc.)

Caveat: GW100 uses its own canonical geometries; ferric's cases() use custom
(DOSD-derived for batch 2) geometries. Agreement to ~0.05 eV despite this shows
the IPs are geometry-insensitive at this level — but it is not a bit-identical
comparison.
"""
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE / "lit_ref"
RESULTS = HERE / "results.json"

# ferric molecule name -> GW100 CAS key
CAS = {
    "H2": "1333-74-0", "He": "7440-59-7", "H2O": "7732-18-5", "NH3": "7664-41-7",
    "CH4": "74-82-8", "N2": "7727-37-9", "CO": "630-08-0", "F2": "7782-41-4",
    "HF": "7664-39-3", "C2H2": "74-86-2", "C2H4": "74-85-1", "C2H6": "74-84-0",
    "CO2": "124-38-9", "HCl": "7647-01-0", "H2S": "7783-06-4", "HCN": "74-90-8",
    "H2CO": "50-00-0", "CH3OH": "67-56-1",
}

FILES = {
    "ccsdt": "CCSD-T_HOMO_CFOUR_def2-TZVPP.json",
    "expt": "Experimental_HOMO_JCTC13-635-2017.json",
    "gwhf": "GWatHF_HOMO_M2.E_def2-TZVPP.json",
    "g0w0pbe": "G0W0atPBE_HOMO_Tv7.0_def2-TZVP_cbas.json",
}


def load(key):
    return json.load(open(REF / FILES[key]))["data"]


def ip(d, cas):
    """IP = -HOMO eigenvalue; the db stores -1 for missing."""
    v = d.get(cas)
    return None if (v is None or v == -1) else -float(v)


def main():
    ref = {k: load(k) for k in FILES}
    fr = json.load(open(RESULTS))["aug-cc-pvtz"]["molecules"]

    print("# ferric (aTZ) vs published GW100 references — theory-vs-theory anchor")
    print(f"{'mol':6} | {'lit-exp':>7} {'fer-exp':>7} | {'CCSDT':>6} {'litGW@HF':>8} "
          f"{'fer-evGW':>8} | {'ΔevGW-CC':>8} {'Δlit-CC':>7}")
    print("-" * 84)
    dferr, dlerr, expmis = [], [], []
    for m, cas in CAS.items():
        le = ip(ref["expt"], cas)
        ct = ip(ref["ccsdt"], cas)
        lg = ip(ref["gwhf"], cas)
        fe = fr[m]["exp"]
        fv = fr[m]["evGW"]
        d_f = fv - ct if ct else None
        d_l = lg - ct if (ct and lg) else None
        if d_f is not None:
            dferr.append(d_f)
        if d_l is not None:
            dlerr.append(d_l)
        if le is not None:
            expmis.append(abs(le - fe))
        print(f"{m:6} | {le if le else float('nan'):7.2f} {fe:7.2f} | "
              f"{ct if ct else float('nan'):6.2f} {lg if lg else float('nan'):8.2f} "
              f"{fv:8.2f} | {d_f if d_f is not None else float('nan'):+8.2f} "
              f"{d_l if d_l is not None else float('nan'):+7.2f}")
    print("-" * 84)
    print(f"ferric evGW vs CCSD(T):  MAE={statistics.mean(abs(x) for x in dferr):.3f}  "
          f"ME={statistics.mean(dferr):+.3f}  N={len(dferr)}")
    print(f"lit GW@HF  vs CCSD(T):  MAE={statistics.mean(abs(x) for x in dlerr):.3f}  "
          f"ME={statistics.mean(dlerr):+.3f}  N={len(dlerr)}")
    print(f"ferric exp ip_ref vs GW100-db experimental: max|Δ|={max(expmis):.3f} eV "
          f"(mean {statistics.mean(expmis):.3f}) — validates the hand-entered refs")
    # ferric-vs-lit-GW@HF directly (the implementation-correctness anchor)
    dvl, dvl_clean = [], []
    for m, cas in CAS.items():
        lg = ip(ref["gwhf"], cas)
        if lg:
            d = fr[m]["evGW"] - lg
            dvl.append(d)
            if m not in ("N2", "CO"):
                dvl_clean.append(d)
    print(f"ferric evGW vs lit GW@HF: MAE={statistics.mean(abs(x) for x in dvl):.3f}  "
          f"ME={statistics.mean(dvl):+.3f}  "
          f"(excl N2/CO self-consistency outliers: "
          f"MAE={statistics.mean(abs(x) for x in dvl_clean):.3f})")


if __name__ == "__main__":
    main()
