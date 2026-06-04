#!/usr/bin/env python3
"""Idempotent result creator for the DOSD C6 sweep.

For each (molecule, method, basis): if results.json already holds a valid,
finite molecular C6 for that key (and --force not given), skip. Otherwise run
ferric-cli on the generated TOML, parse the printed 'molecular C6 = X a.u.'
line, and store it. Runs SERIALLY with OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=8
(parallel BLAS/rayon oversubscribes a 12-core box).

The DOSD-comparable value is the CLI's printed molecular C6 (= c6_molecular_iso,
the global-origin molecular alpha(iw) Casimir-Polder integral), NOT npz
c6_iso.sum() (per-atom pair sum, omits inter-atomic coupling).

Usage:
  run_sweep.py                 # run everything missing/invalid
  run_sweep.py augccpvdz       # only the DZ basis
  run_sweep.py o2 augccpvdz    # only jobs matching ALL listed filters? no: ANY
  run_sweep.py --force         # recompute everything
"""
import json, os, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = HERE / "runs"
RESULTS = HERE / "results.json"
BIN = ROOT / "target" / "release" / "ferric-cli"

MOLS = ["h2", "n2", "co", "water", "nh3", "ch4", "co2", "c2h2", "c2h4", "c2h6",
        "hf", "hcl", "h2s", "benzene", "o2"]
METHODS = ["rpa_pbe", "rpa_hf", "ts"]
BASES = ["augccpvdz", "augccpvtz"]

C6_RE = re.compile(r"molecular C6\s*=\s*([-+0-9.eE]+)\s*a\.u\.")

ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
           RAYON_NUM_THREADS="8")


def load_results():
    if RESULTS.exists():
        return json.loads(RESULTS.read_text())
    return {}


def save_results(d):
    tmp = RESULTS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=2, sort_keys=True))
    tmp.replace(RESULTS)   # atomic; safe to Ctrl-C between jobs


def key(mol, method, basis):
    return f"{basis}/{mol}/{method}"


def valid(entry):
    return (entry is not None and isinstance(entry.get("c6"), (int, float))
            and entry["c6"] == entry["c6"] and entry["c6"] > 0)


def run_one(mol, method, basis):
    toml = RUNS / basis / f"{mol}_{method}.toml"
    log = RUNS / basis / f"{mol}_{method}.log"
    if not toml.exists():
        return {"c6": None, "error": "missing TOML", "ok": False}
    t0 = time.time()
    proc = subprocess.run([str(BIN), str(toml)], cwd=str(ROOT),
                          env=ENV, capture_output=True, text=True)
    dt = time.time() - t0
    out = proc.stdout + "\n=== STDERR ===\n" + proc.stderr
    log.write_text(out)
    m = C6_RE.search(proc.stdout)
    if m:
        return {"c6": float(m.group(1)), "ok": True, "seconds": round(dt, 1),
                "method": method, "basis": basis, "mol": mol}
    return {"c6": None, "ok": False, "seconds": round(dt, 1),
            "error": "no C6 parsed; see log", "rc": proc.returncode}


def main():
    force = "--force" in sys.argv
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    results = load_results()
    # Order: cheap molecules first, benzene/o2 last (slow / open-shell).
    order = sorted(MOLS, key=lambda m: (m == "benzene", m == "o2", m))
    jobs = [(m, me, b) for b in BASES for m in order for me in METHODS]
    if only:  # keep a job if it matches ANY listed filter token
        jobs = [j for j in jobs if any(o in j for o in only)]
    total = len(jobs)
    done = 0
    for mol, method, basis in jobs:
        k = key(mol, method, basis)
        done += 1
        if not force and valid(results.get(k)):
            print(f"[{done}/{total}] skip (cached) {k} = {results[k]['c6']}")
            continue
        print(f"[{done}/{total}] run {k} ...", flush=True)
        entry = run_one(mol, method, basis)
        results[k] = entry
        save_results(results)
        status = "OK" if entry.get("ok") else "FAIL"
        print(f"    -> {status} C6={entry.get('c6')} ({entry.get('seconds')}s)")
    save_results(results)
    ok = sum(1 for v in results.values() if valid(v))
    print(f"\ndone: {ok}/{len(results)} valid entries in {RESULTS}")


if __name__ == "__main__":
    main()
