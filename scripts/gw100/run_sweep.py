#!/usr/bin/env python3
"""Idempotent GW100 sweep runner: runs the gw100_full driver per basis, parses
the per-molecule IP table + MAE row, stores into results.json. Resumable — skips
a (basis) whose results are already present and valid unless --force.

Starts with the validated 10-molecule subset (the driver's `cases()`); expand by
adding molecules to gw100_full.rs's cases() and re-running (idempotent).

Each run is the user's job to launch memory-scoped/gated; this script only
orchestrates parsing + storage. Run ONE basis at a time to avoid box contention.

Usage:
  run_sweep.py aug-cc-pvdz      # run + store one basis
  run_sweep.py aug-cc-pvtz
  run_sweep.py --show           # print the aggregated cross-basis table
  run_sweep.py <basis> --force  # recompute even if present
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BIN = ROOT / "target" / "release" / "examples" / "gw100_full"
RESULTS = HERE / "results.json"

METHODS = ["Koop", "dSCF", "dRPA", "G0W0", "COHSEX", "evGW0", "evGW"]
# per-molecule data row: "H2O  12.62  13.889  10.989  12.549  12.890  14.672  12.835  12.801"
ROW = re.compile(
    r"^(?P<mol>[A-Za-z0-9]+)\s+(?P<exp>[-+0-9.]+)\s+" + r"\s+".join(
        rf"(?P<{m}>[-+0-9.]+)" for m in METHODS
    )
)
MAE = re.compile(r"^MAE\s+(?P<vals>.+)$")

ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
           MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1")


def load():
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else {}


def save(d):
    tmp = RESULTS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=2, sort_keys=True))
    tmp.replace(RESULTS)


def parse_output(txt):
    """Return {mol: {exp, Koop, ..., evGW}} + {'MAE': {method: val}}."""
    mols, mae = {}, {}
    for line in txt.splitlines():
        m = ROW.match(line.strip())
        if m:
            d = m.groupdict()
            mol = d.pop("mol")
            mols[mol] = {k: float(v) for k, v in d.items()}
        mm = MAE.match(line.strip())
        if mm:
            vals = mm.group("vals").split()
            for name, v in zip(METHODS, vals):
                try:
                    mae[name] = float(v)
                except ValueError:
                    pass
    return mols, mae


def run_basis(basis, force=False):
    res = load()
    if not force and basis in res and res[basis].get("molecules"):
        print(f"[skip] {basis} already present ({len(res[basis]['molecules'])} mols)")
        return
    if not BIN.exists():
        sys.exit(f"binary missing: {BIN} (build gw100_full first)")
    print(f"[run] gw100_full {basis} ...")
    out = subprocess.run([str(BIN), basis], env=ENV, capture_output=True, text=True)
    txt = out.stdout + "\n" + out.stderr
    mols, mae = parse_output(txt)
    if not mols:
        print(f"[FAIL] {basis}: no molecule rows parsed. Tail:\n{txt[-800:]}")
        return
    res[basis] = {"molecules": mols, "mae": mae,
                  "n_converged": len(mols), "n_attempted": txt.count("FAILED") + len(mols)}
    save(res)
    failed = re.findall(r"^(\w+)\s+FAILED", txt, re.M)
    print(f"[done] {basis}: {len(mols)} converged, {len(failed)} FAILED {failed}")
    print(f"       evGW MAE = {mae.get('evGW', '?')} eV")


def show():
    res = load()
    if not res:
        print("no results yet")
        return
    print(f"# GW100 sweep — MAE vs experiment (eV) by basis")
    print(f"{'basis':14} " + " ".join(f"{m:>7}" for m in METHODS) + f"  {'#mol':>5}")
    print("-" * 80)
    for basis in sorted(res):
        mae = res[basis].get("mae", {})
        n = res[basis].get("n_converged", 0)
        print(f"{basis:14} " + " ".join(f"{mae.get(m, float('nan')):>7.3f}" for m in METHODS)
              + f"  {n:>5}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if "--show" in flags or not args:
        show()
    else:
        run_basis(args[0], force="--force" in flags)
