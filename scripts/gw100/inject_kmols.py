#!/usr/bin/env python3
"""Inject the def2-TZVP K-molecule (BrK/HK/K2) GW rows into the GW100 results
JSONs and remove them from `failed`. Idempotent; safe to re-run.

These 3 molecules have NO aug-cc-pVNZ orbital basis (K=Z19 has none in the
correlation-consistent family — it stops at Ar). They route to def2-TZVP in
gw100_full.rs (commit on branch fix/gw100-k-basis). Rows captured from that
binary live in kmols_def2tzvp_rows.txt. Run this AFTER the live sweeps finish,
or any time you regenerate the rows, to fold them into the table.
"""
import json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
METHODS = ["Koop","dSCF","dRPA","G0W0","COHSEX","evGW0","evGW","G0W0pbe"]
rows = {}
for line in (HERE/"kmols_def2tzvp_rows.txt").read_text().splitlines():
    m = re.match(r'^(BrK|HK|K2)\s+([-+0-9.]+)\s+'+r'\s+'.join(r'([-+0-9.]+)' for _ in METHODS), line.strip())
    if m:
        g = m.groups(); mol = g[0]
        rows[mol] = {"exp": float(g[1])}
        rows[mol].update({k: float(v) for k,v in zip(METHODS, g[2:])})
assert set(rows) == {"BrK","HK","K2"}, f"expected 3 K-mols, got {sorted(rows)}"
for b in ("aug-cc-pvdz","aug-cc-pvtz"):
    p = HERE/f"results_{b}.json"
    d = json.loads(p.read_text())
    for mol,row in rows.items():
        d["molecules"][mol] = row
        d["failed"] = [f for f in d.get("failed",[]) if f != mol]
    d["failure_reasons"] = {k:v for k,v in d.get("failure_reasons",{}).items() if k in set(d.get("failed",[]))}
    tmp = p.with_suffix(".json.tmp"); tmp.write_text(json.dumps(d,indent=2,sort_keys=True)); tmp.replace(p)
    print(f"  {b}: injected {sorted(rows)} -> {len(d['molecules'])} conv, failed={sorted(d.get('failed',[]))}")
