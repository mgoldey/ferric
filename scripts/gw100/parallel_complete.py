#!/usr/bin/env python3
"""Parallel GW100 table completion — disjoint-partition runner.

The bottleneck: each molecule takes ~2-10 min through the GW pipeline, and the
serial run_sweep.py does them one at a time per base. This runner pools ALL
remaining (basis, molecule) work across N concurrent workers, each running
gw100_full directly on a single molecule, writing results straight into the
per-basis results JSON under a lock (so no read-modify-write race).

Policy matches the driver: full-depth <=10 atoms, G0W0-only for big (the driver's
GW100_FULL_MAX_ATOMS=10 handles that internally). We just feed it one molecule at
a time via GW100_DONE=all-but-this.

Sizing: each molecule's SCF phase is ~1 core (BLAS-serial), RPA phase is rayon.
Run N workers at RAYON_NUM_THREADS=R; pick N*R ~= cores. Default 4 workers x 3.

Usage: parallel_complete.py [nworkers] [rayon_per_worker]
Idempotent: skips molecules already converged/failed. Run after stopping the
serial sweeps. Writes each row under a file lock; safe for concurrent workers.
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
BASES = ["aug-cc-pvdz", "aug-cc-pvtz"]
METHODS = ["Koop", "dSCF", "dRPA", "G0W0", "COHSEX", "evGW0", "evGW", "G0W0pbe"]
ROW = re.compile(
    r"^(?P<mol>[A-Za-z0-9]+)\s+(?P<exp>[-+0-9.]+|NaN|nan)\s+"
    + r"\s+".join(rf"(?P<{m}>[-+0-9.]+|NaN|nan)" for m in METHODS)
)
_lock = threading.Lock()
MOL_BUDGET = float(os.environ.get("GW100_MOL_BUDGET", "5400"))


def all_cases():
    return re.findall(r'name:\s*"(\w+)"', SRC.read_text())


def remaining():
    """List of (basis, mol) still to do, ordered small-first within each basis.

    GW100_ONLY=mol1,mol2,...  -> run ONLY these molecules (still skipping ones
    already converged), regardless of their `failed` status. Use for a dedicated
    high-budget retry of recoverable cost-timeouts, disjoint from the main runner.
    """
    txt = SRC.read_text()
    def natoms(n):
        m = re.search(rf'name:\s*"{n}".*?xyz:\s*"(\d+)', txt, re.S)
        return int(m.group(1)) if m else 999
    only = os.environ.get("GW100_ONLY", "").strip()
    only_set = set(only.split(",")) if only else None
    work = []
    for b in BASES:
        d = json.loads((HERE / f"results_{b}.json").read_text())
        converged = set(d["molecules"])
        if only_set is not None:
            # Targeted mode: only the named molecules, skip any already converged.
            todo = [c for c in all_cases() if c in only_set and c not in converged]
        else:
            done = converged | set(d.get("failed", []))
            todo = [c for c in all_cases() if c not in done]
        todo.sort(key=natoms)  # small first
        for c in todo:
            work.append((b, c))
    return work


def _locked_update(basis, mutate):
    """Read-mutate-write under BOTH the thread lock (in-process workers) and a
    cross-process flock (a concurrent run_sweep.py or second parallel_complete
    on the same basis — the threading.Lock alone cannot see them). Same .lock
    path as run_sweep.save_basis. Per-pid tmp so writers never share one tmp."""
    import fcntl
    with _lock:
        p = HERE / f"results_{basis}.json"
        with open(p.with_suffix(".lock"), "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            d = json.loads(p.read_text())
            if mutate(d):
                tmp = p.with_suffix(f".json.tmp.{os.getpid()}")
                tmp.write_text(json.dumps(d, indent=2, sort_keys=True))
                tmp.replace(p)


def save_row(basis, mol, row):
    def mutate(d):
        d["molecules"][mol] = row
        d["failed"] = [f for f in d.get("failed", []) if f != mol]
        return True
    _locked_update(basis, mutate)


def mark_failed(basis, mol):
    def mutate(d):
        if mol in d["molecules"]:
            return False
        fl = set(d.get("failed", [])); fl.add(mol)
        d["failed"] = sorted(fl)
        return True
    _locked_update(basis, mutate)


def run_one(basis, mol, rayon):
    """Run gw100_full for a single molecule; persist its row. Watchdog kills it
    past MOL_BUDGET and marks failed."""
    skip = ",".join(c for c in all_cases() if c != mol)
    # Depth knobs are overridable from the parent env so a "G0W0-only for the
    # slow tail" run can set GW100_FULL_MAX_ATOMS=0 (drop the full ladder for
    # every molecule) + GW100_PBE_ALL=1 (keep the @PBE column) — the two
    # PySCF-validated columns at ~5-10x the speed of the full ladder.
    max_atoms = os.environ.get("GW100_FULL_MAX_ATOMS", "10")
    pbe_all = os.environ.get("GW100_PBE_ALL", "0")
    env = dict(os.environ,
               OPENBLAS_NUM_THREADS="1", RAYON_NUM_THREADS=str(rayon),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               GW100_TRUNC="1e-4", GW100_FULL_MAX_ATOMS=max_atoms,
               GW100_PBE_ALL=pbe_all, GW100_DONE=skip)
    proc = subprocess.Popen([str(BIN), basis], env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    start = time.monotonic()
    got = False
    def watchdog():
        while proc.poll() is None:
            time.sleep(15)
            if time.monotonic() - start > MOL_BUDGET:
                proc.kill(); return
    wd = threading.Thread(target=watchdog, daemon=True); wd.start()
    for line in proc.stdout:
        m = ROW.match(line.strip())
        if m and m.group("mol") == mol:
            dd = m.groupdict(); dd.pop("mol")
            row = {k: float(v) for k, v in dd.items()}
            save_row(basis, mol, row)
            got = True
            print(f"  [+] {basis[:8]} {mol}  G0W0={row['G0W0']:.3f}  ({time.monotonic()-start:.0f}s)", flush=True)
    proc.wait()
    if not got:
        mark_failed(basis, mol)
        print(f"  [x] {basis[:8]} {mol} FAILED/timeout  ({time.monotonic()-start:.0f}s)", flush=True)


def main():
    nworkers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    rayon = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    if not BIN.exists():
        sys.exit(f"binary missing: {BIN}")
    work = remaining()
    print(f"[parallel] {len(work)} (basis,mol) to do; {nworkers} workers x RAYON={rayon}", flush=True)

    work_lock = threading.Lock()
    it = iter(work)
    def worker():
        while True:
            with work_lock:
                try:
                    b, m = next(it)
                except StopIteration:
                    return
            try:
                run_one(b, m, rayon)
            except Exception as e:
                print(f"  [!] {b} {m}: {e}", flush=True)

    threads = [threading.Thread(target=worker) for _ in range(nworkers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("[parallel] DONE", flush=True)
    for b in BASES:
        d = json.loads((HERE / f"results_{b}.json").read_text())
        print(f"  {b}: {len(d['molecules'])} conv, {len(d.get('failed', []))} fail")


if __name__ == "__main__":
    main()
