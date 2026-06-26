#!/usr/bin/env python3
"""Serial GW100 completion — ONE molecule at a time, ALL cores.

The parallel runner (parallel_complete.py) splits the machine N ways, so each
slow-tail molecule gets only ~cores/N and times out before the GW QP solve
finishes. The slow tail (big aromatics, nucleobases) actually CONVERGES when
given the whole machine — proven: pyridine failed under 3-way split, converged
in ~5 min at RAYON=12.

This runner does the opposite of parallel_complete: it runs molecules strictly
serially, each with RAYON_NUM_THREADS = all cores, persisting each row the
instant it lands. Slowest-but-correct. Use this for the tail, not the bulk.

Usage: serial_complete.py [basis] [g0w0_only=1]
  basis defaults to aug-cc-pvdz. Honors GW100_ONLY=mol1,mol2 to target a subset.
Idempotent: skips already-converged molecules. A per-molecule watchdog
(GW100_MOL_BUDGET secs, default 3600) kills a hung solve and marks it failed.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BIN = ROOT / "target" / "release" / "examples" / "gw100_full"
SRC = ROOT / "benchmarks" / "harness" / "examples" / "gw100_full.rs"
METHODS = ["Koop", "dSCF", "dRPA", "G0W0", "COHSEX", "evGW", "evGW0", "G0W0pbe"]
# Note: column order in the table is Koop dSCF dRPA G0W0 COHSEX evGW0 evGW G0W0pbe
COLS = ["Koop", "dSCF", "dRPA", "G0W0", "COHSEX", "evGW0", "evGW", "G0W0pbe"]
ROW = re.compile(
    r"^(?P<mol>[A-Za-z0-9]+)\s+(?P<exp>[-+0-9.]+)\s+"
    + r"\s+".join(rf"(?P<{m}>[-+0-9.]+|NaN)" for m in COLS)
)
NCORES = os.cpu_count() or 12
MOL_BUDGET = float(os.environ.get("GW100_MOL_BUDGET", "3600"))
# Genuine, non-recoverable failures (K basis gap + d-block non-convergence).
GENUINE = {"BrK", "HK", "K2", "Cu2", "CCuN", "F4Ti"}


def all_cases():
    return re.findall(r'name:\s*"(\w+)"', SRC.read_text())


def natoms(txt, n):
    m = re.search(rf'name:\s*"{n}".*?xyz:\s*"(\d+)', txt, re.S)
    return int(m.group(1)) if m else 999


def remaining(basis):
    txt = SRC.read_text()
    d = json.loads((HERE / f"results_{basis}.json").read_text())
    converged = set(d["molecules"])
    only = os.environ.get("GW100_ONLY", "").strip()
    only_set = set(only.split(",")) if only else None
    todo = []
    for c in all_cases():
        if c in converged or c in GENUINE:
            continue
        if only_set is not None and c not in only_set:
            continue
        todo.append(c)
    todo.sort(key=lambda c: natoms(txt, c))  # small first
    return todo


def save_row(basis, mol, row):
    p = HERE / f"results_{basis}.json"
    d = json.loads(p.read_text())
    d["molecules"][mol] = row
    d["failed"] = [f for f in d.get("failed", []) if f != mol]
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=2, sort_keys=True))
    tmp.replace(p)


def mark_failed(basis, mol):
    p = HERE / f"results_{basis}.json"
    d = json.loads(p.read_text())
    if mol not in d["molecules"]:
        fl = set(d.get("failed", []))
        fl.add(mol)
        d["failed"] = sorted(fl)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, indent=2, sort_keys=True))
        tmp.replace(p)


def run_one(basis, mol):
    skip = ",".join(c for c in all_cases() if c != mol)
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        RAYON_NUM_THREADS=str(NCORES),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        GW100_TRUNC="1e-4",
        GW100_FULL_MAX_ATOMS=os.environ.get("GW100_FULL_MAX_ATOMS", "0"),
        GW100_G0W0_ONLY=os.environ.get("GW100_G0W0_ONLY", "1"),
        GW100_PBE_ALL=os.environ.get("GW100_PBE_ALL", "1"),
        GW100_DONE=skip,
    )
    t0 = time.monotonic()
    proc = subprocess.Popen([str(BIN), basis], env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    got = False

    def watchdog():
        while proc.poll() is None:
            time.sleep(15)
            if time.monotonic() - t0 > MOL_BUDGET:
                proc.kill()
                return
    threading.Thread(target=watchdog, daemon=True).start()

    for line in proc.stdout:
        m = ROW.match(line.strip())
        if m and m.group("mol") == mol:
            dd = m.groupdict()
            dd.pop("mol")
            row = {k: (float(v) if v != "NaN" else None) for k, v in dd.items()}
            if row.get("G0W0") is not None:
                save_row(basis, mol, row)
                got = True
    proc.wait()
    dt = time.monotonic() - t0
    if got:
        print(f"  [+] {mol} converged in {dt:.0f}s", flush=True)
    else:
        mark_failed(basis, mol)
        print(f"  [x] {mol} FAILED after {dt:.0f}s", flush=True)
    return got


def main():
    basis = sys.argv[1] if len(sys.argv) > 1 else "aug-cc-pvdz"
    if not BIN.exists():
        sys.exit(f"binary missing: {BIN}")
    todo = remaining(basis)
    print(f"[serial] {basis}: {len(todo)} molecules, RAYON={NCORES}, budget={MOL_BUDGET:.0f}s", flush=True)
    print(f"[serial] order: {todo}", flush=True)
    nconv = 0
    for mol in todo:
        print(f"[serial] -> {mol}", flush=True)
        if run_one(basis, mol):
            nconv += 1
    d = json.loads((HERE / f"results_{basis}.json").read_text())
    print(f"[serial] DONE: +{nconv} this run; {basis} now {len(d['molecules'])} conv, "
          f"{len(d.get('failed', []))} fail", flush=True)


if __name__ == "__main__":
    main()
