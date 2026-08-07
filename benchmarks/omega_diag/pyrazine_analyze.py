#!/usr/bin/env python3
"""Analysis for the pyrazine-dimer (S22 #12) CP ω-sweep at aDZ.

Reads derisk/out/pyrazine_D_12_adz_*.out (written by pyrazine_cp_sweep.py),
computes CP binding per method × ω, errors vs CCSD(T)/CBS (S22B revised,
benchmarks/grid/refs.json: -4.255 kcal/mol), writes
derisk/PYRAZINE_CP.md + derisk/pyrazine_results.json. ADDITIVE: reads only,
runs nothing, deletes nothing. CP arm only (no plain monomers were computed —
per the de-risk verdict B/T must be CP-corrected; non-CP is the error).
"""
import os, re, json

ROOT = "/home/matt/qc/ferric"
os.chdir(ROOT)
OUT = "benchmarks/omega_diag/derisk"
KCAL = 627.509474
LABEL, SID, BT = "pyrazine_D", "12", "adz"
REF = -4.255  # S22B revised CCSD(T)/CBS, benchmarks/grid/refs.json s22[12]
OMEGAS = [0.20, 0.30, 0.42, 0.55, 0.673, 0.80]
FRAGS = ["dimer", "cpA", "cpB"]

TOTAL = r'Total energy\s*=\s*(-?[0-9.]+)'
RHF = r'RHF energy\s*=\s*(-?[0-9.]+)'
PATS = {
    "MP2": r'E\(MP2, Coulomb\)\s*=\s*(-?[0-9.]+)',
    "SRMP2": r'E\(SR-MP2, erfc\)\s*=\s*(-?[0-9.]+)',
    "naiveA": r'E_corr naive \(A\)\s*=\s*(-?[0-9.]+)',
}


def grab(key, pat):
    p = f"{OUT}/out/{key}.out"
    if not os.path.exists(p):
        return None
    m = re.search(pat, open(p).read())
    return float(m.group(1)) if m else None


rhf = {fr: grab(f"{LABEL}_{SID}_{BT}_RHF_{fr}", RHF) for fr in FRAGS}
results = {}
for omega in OMEGAS:
    kB = f"{LABEL}_{SID}_{BT}_w{omega}_B"
    kT = f"{LABEL}_{SID}_{BT}_w{omega}_T"
    # component methods ride on the B (delta-lr) outputs
    for method, pat in PATS.items():
        cs = {fr: grab(f"{kB}_{fr}", pat) for fr in FRAGS}
        if None in rhf.values() or None in cs.values():
            results[f"{omega}|{method}"] = None
            continue
        results[f"{omega}|{method}"] = (
            (rhf["dimer"] + cs["dimer"]) - (rhf["cpA"] + cs["cpA"])
            - (rhf["cpB"] + cs["cpB"])) * KCAL
    for method, k in [("B", kB), ("T", kT)]:
        ts = {fr: grab(f"{k}_{fr}", TOTAL) for fr in FRAGS}
        results[f"{omega}|{method}"] = (
            None if None in ts.values()
            else (ts["dimer"] - ts["cpA"] - ts["cpB"]) * KCAL)

json.dump({"binds": results, "ref": REF, "rhf": rhf},
          open(f"{OUT}/pyrazine_results.json", "w"), indent=1)

L = ["# Pyrazine dimer (S22 #12) CP ω-sweep — aDZ\n",
     f"CP binding kcal/mol; ref CCSD(T)/CBS (S22B) = {REF:+.3f}. "
     "Negative err = overbinding. CP arm only.\n",
     "| ω | method | CP bind | err |", "|---|---|---|---|"]
for omega in OMEGAS:
    for method in ["MP2", "SRMP2", "naiveA", "B", "T"]:
        v = results.get(f"{omega}|{method}")
        vs = f"{v:+.3f}" if v is not None else "—"
        es = f"{v - REF:+.3f}" if v is not None else "—"
        L.append(f"| {omega} | {method} | {vs} | {es} |")

L.append("\n## |err| minima per method (this system alone)\n")
L.append("| method | best ω | err at best |")
L.append("|---|---|---|")
for method in ["MP2", "SRMP2", "naiveA", "B", "T"]:
    pts = [(omega, results[f"{omega}|{method}"]) for omega in OMEGAS
           if results.get(f"{omega}|{method}") is not None]
    if not pts:
        L.append(f"| {method} | — | — |")
        continue
    w, v = min(pts, key=lambda p: abs(p[1] - REF))
    L.append(f"| {method} | {w} | {v - REF:+.3f} |")

open(f"{OUT}/PYRAZINE_CP.md", "w").write("\n".join(L) + "\n")
nv = sum(1 for v in results.values() if v is not None)
print(f"wrote PYRAZINE_CP.md + pyrazine_results.json ({nv}/{len(results)} binds)")
print("\n".join(L[2:]))
