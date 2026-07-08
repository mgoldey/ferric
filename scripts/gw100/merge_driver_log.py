#!/usr/bin/env python3
"""Merge GW100 per-molecule rows from a RAW driver stdout log into a results
JSON. Idempotent. Used to fold in rows produced by dedicated out-of-band workers
(e.g. the big-molecule long-budget tail worker) that write to their own log
instead of through run_sweep.py (to avoid the shared-file write race).

Only molecules with a parseable full data row are merged; a molecule already
present is overwritten (newer run wins) and removed from `failed`.

Usage: merge_driver_log.py <raw_driver.log> <basis>   e.g. ... gw_bigtail_atz_X.txt aug-cc-pvtz
"""
import json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
METHODS = ["Koop","dSCF","dRPA","G0W0","COHSEX","evGW0","evGW","G0W0pbe"]
ROW = re.compile(r'^(?P<mol>[A-Za-z0-9]+)\s+(?P<exp>[-+0-9.]+)\s+' +
                 r'\s+'.join(rf'(?P<{m}>[-+0-9.]+)' for m in METHODS) + r'\s*$')

def main():
    log, basis = sys.argv[1], sys.argv[2]
    rows = {}
    for line in Path(log).read_text().splitlines():
        m = ROW.match(line.strip())
        if m and m.group("mol") not in ("MAE","mol"):
            d = m.groupdict(); mol = d.pop("mol")
            # skip NaN-laden rows (failed columns print as the literal; guard finite exp)
            try:
                rows[mol] = {k: float(v) for k,v in d.items()}
            except ValueError:
                pass
    if not rows:
        print(f"no parseable rows in {log}"); return
    p = HERE/f"results_{basis}.json"
    d = json.loads(p.read_text())
    merged = []
    for mol, row in rows.items():
        # only accept if G0W0 is finite (the worker's headline number)
        if row.get("G0W0") == row.get("G0W0") and abs(row.get("G0W0", 1e9)) < 1e6:
            d["molecules"][mol] = row
            d["failed"] = [f for f in d.get("failed",[]) if f != mol]
            merged.append(mol)
    d["failure_reasons"] = {k:v for k,v in d.get("failure_reasons",{}).items() if k in set(d.get("failed",[]))}
    tmp = p.with_suffix(".json.tmp"); tmp.write_text(json.dumps(d,indent=2,sort_keys=True)); tmp.replace(p)
    print(f"  {basis}: merged {sorted(merged)} -> {len(d['molecules'])} conv, failed={sorted(d.get('failed',[]))}")

if __name__ == "__main__":
    main()
