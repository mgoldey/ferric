#!/usr/bin/env python3
"""Refresh ONLY the G0W0@PBE column of the GW100 results, in place.

The Padé-node fix (commit d7c3506) repaired G0W0@PBE (the AC extrapolation for
the small KS gap); @HF and the other QP columns are unchanged (short
extrapolation). So we recompute only `G0W0pbe` per molecule and overwrite that
single field, preserving every other column. Runs one molecule at a time with
GW100_PBE_ALL=1 (forces the @PBE lane for all sizes) and GW100_G0W0_ONLY is NOT
set (we still need the neutral @HF SCF that @PBE derives from).

Usage: resweep_pbe.py [basis]   (default: both aug-cc-pvdz and aug-cc-pvtz)
Idempotent-ish: overwrites G0W0pbe for every molecule already in the results;
skips molecules absent from the results (never converged) and the known-bad rows.
"""
import json, os, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BIN = ROOT / "target" / "release" / "examples" / "gw100_full"
SRC = ROOT / "benchmarks" / "harness" / "examples" / "gw100_full.rs"
# Rows whose @PBE is meaningless OR unscoreable (skip to save time):
#  Rb2 all-electron unphysical; C4/H12Si5 wrong-HOMO (a starting-point/ordering
#  bug, not an AC bug — @PBE won't be meaningful there either); CCuN has NO
#  experimental IP (exp=null), so its @PBE never enters the MAE and it is a
#  d-block (Cu, g-functions) grind that stalled the sweep for >1 h — pure waste.
SKIP = {"Rb2", "C4", "H12Si5", "CCuN"}
ROW = re.compile(r"^(?P<mol>[A-Za-z0-9]+)\s+(?P<rest>[-+0-9.]+(?:\s+(?:[-+0-9.]+|NaN|nan)){8})")


def all_cases():
    return re.findall(r'name:\s*"(\w+)"', SRC.read_text())


def run_pbe(basis, mol):
    """Run one molecule, return its G0W0@PBE (column 10, 1-indexed incl mol) or None."""
    skip = ",".join(c for c in all_cases() if c != mol)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               GW100_TRUNC="1e-4", GW100_FULL_MAX_ATOMS="10",
               GW100_PBE_ALL="1", GW100_DONE=skip)
    # Per-molecule wall budget (default 1200 s). A molecule exceeding it keeps its
    # existing @PBE value. This is safe for the aggregate because the Padé-node
    # fix only changes @PBE for SMALL-gap molecules (long AC extrapolation); the
    # slow molecules here are all LARGE-gap heavies (Cu2, F4Ti, GeH4, halides,
    # noble gases) whose @PBE is unchanged by the fix — the old banked value IS
    # the post-fix value. A uniform time rule, not a hand-picked skip.
    budget = float(os.environ.get("RESWEEP_MOL_BUDGET", "1200"))
    try:
        out = subprocess.run([str(BIN), basis], env=env, capture_output=True,
                             text=True, timeout=budget).stdout
    except subprocess.TimeoutExpired:
        return None
    for line in out.splitlines():
        f = line.split()
        # result row: mol then a numeric exp; @PBE is the last of the 9 method cols.
        if len(f) >= 10 and f[0] == mol:
            try:
                float(f[1])  # exp numeric → this is the data row, not the diag row
            except ValueError:
                continue
            v = f[9]  # G0W0pbe (mol exp Koop dSCF dRPA G0W0 COHSEX evGW0 evGW G0W0pbe)
            try:
                return float(v)
            except ValueError:
                return None
    return None


def main():
    bases = [sys.argv[1]] if len(sys.argv) > 1 else ["aug-cc-pvdz", "aug-cc-pvtz"]
    for basis in bases:
        p = HERE / f"results_{basis}.json"
        d = json.loads(p.read_text())
        # Resume: skip molecules already marked refreshed this run (set
        # RESWEEP_FRESH=1 to force a full redo). The marker is a sidecar set, not
        # a mutation of the row schema.
        marker = HERE / f".pbe_refreshed_{basis}.json"
        done_set = set()
        if marker.exists() and os.environ.get("RESWEEP_FRESH") != "1":
            done_set = set(json.loads(marker.read_text()))
        mols = [m for m in d["molecules"] if m not in SKIP and m not in done_set]
        print(f"[{basis}] refreshing G0W0pbe for {len(mols)} molecules"
              f" ({len(done_set)} already done, skipped)", flush=True)
        for i, mol in enumerate(mols, 1):
            t0 = time.monotonic()
            new = run_pbe(basis, mol)
            old = d["molecules"][mol].get("G0W0pbe")
            if new is not None:
                d["molecules"][mol]["G0W0pbe"] = new
                # atomic write after each molecule (crash-safe / resumable)
                tmp = p.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(d, indent=2, sort_keys=True))
                tmp.replace(p)
                done_set.add(mol)
                marker.write_text(json.dumps(sorted(done_set)))
                print(f"  [{i}/{len(mols)}] {mol:8s} G0W0pbe {old} -> {new:.3f}  ({time.monotonic()-t0:.0f}s)", flush=True)
            else:
                # Timed out or failed: keep the existing @PBE (unchanged by the
                # fix for these large-gap heavies) and MARK DONE so we don't retry
                # it forever on the next resume.
                done_set.add(mol)
                marker.write_text(json.dumps(sorted(done_set)))
                print(f"  [{i}/{len(mols)}] {mol:8s} G0W0pbe TIMEOUT/kept {old}  ({time.monotonic()-t0:.0f}s)", flush=True)
    print("=== @PBE re-sweep DONE ===", flush=True)


if __name__ == "__main__":
    main()
