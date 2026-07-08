#!/usr/bin/env python3
"""Same-geometry same-basis G0W0@HF cross-check: ferric vs PySCF, per molecule.

For each GW100 canonical geometry in geom/<mol>.xyz, runs BOTH:
  * ferric:  target/release/examples/gw_xcheck  (PDEP-as-W)
  * PySCF:   pyscf_g0w0.py                       (analytic continuation, gw_ac)
at def2-TZVP, identical xyz. Removes the geometry/basis approximation in the
GW100-database literature anchor (compare_literature.py) — this is the
bit-level implementation-correctness proof.

Idempotent: caches per-molecule results in xcheck_results.json; re-run resumes.
Each subprocess is single-thread; the USER launches this memory-scoped/gated.

Usage:
  xcheck_runner.py [basis]        # run all 18 (default def2-tzvp), store, print
  xcheck_runner.py --show         # print cached table only
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GEOM = HERE / "geom"
FERRIC = ROOT / "target" / "release" / "examples" / "gw_xcheck"
PYSCF = HERE / "pyscf_g0w0.py"
CACHE = HERE / "xcheck_results.json"

MOLS = ["H2", "He", "H2O", "NH3", "CH4", "N2", "CO", "F2", "HF",
        "C2H2", "C2H4", "C2H6", "CO2", "HCl", "H2S", "HCN", "H2CO", "CH3OH"]

ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
           MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1")
FER_RE = re.compile(r"XCHECK\s+([-+0-9.]+)\s+([-+0-9.]+)")
PY_RE = re.compile(r"PYSCF\s+([-+0-9.]+)\s+([-+0-9.]+)")


def load():
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def save(d):
    tmp = CACHE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=2, sort_keys=True))
    tmp.replace(CACHE)


def run_ferric(xyz, basis, aux):
    out = subprocess.run([str(FERRIC), str(xyz), basis, aux],
                         env=ENV, capture_output=True, text=True, timeout=3600)
    m = FER_RE.search(out.stdout + out.stderr)
    return (float(m.group(1)), float(m.group(2))) if m else None


def run_pyscf(xyz, basis):
    out = subprocess.run([sys.executable, str(PYSCF), str(xyz), basis],
                         env=ENV, capture_output=True, text=True, timeout=3600)
    m = PY_RE.search(out.stdout + out.stderr)
    return (float(m.group(1)), float(m.group(2))) if m else None


def run_all(basis):
    aux = f"{basis}-rifit"
    res = load()
    key = basis
    res.setdefault(key, {})
    for mol in MOLS:
        if res[key].get(mol) and "fer_g0w0" in res[key][mol]:
            print(f"[skip] {mol} (cached)")
            continue
        xyz = GEOM / f"{mol}.xyz"
        if not xyz.exists():
            print(f"[miss] {mol}: no geometry")
            continue
        print(f"[run]  {mol} ...", flush=True)
        fer = run_ferric(xyz, basis, aux)
        pys = run_pyscf(xyz, basis)
        if not fer or not pys:
            print(f"[FAIL] {mol}: ferric={fer} pyscf={pys}")
            continue
        res[key][mol] = {"fer_g0w0": fer[0], "fer_koop": fer[1],
                         "pys_g0w0": pys[0], "pys_koop": pys[1],
                         "d_g0w0_mev": (fer[0] - pys[0]) * 1000.0,
                         "d_koop_mev": (fer[1] - pys[1]) * 1000.0}
        save(res)
        print(f"       ferric {fer[0]:.3f}  pyscf {pys[0]:.3f}  Δ {res[key][mol]['d_g0w0_mev']:+.1f} meV")
    show(basis)


def show(basis=None):
    res = load()
    for key in ([basis] if basis else sorted(res)):
        rows = res.get(key, {})
        if not rows:
            continue
        print(f"\n# ferric vs PySCF G0W0@HF — IDENTICAL geometry+basis ({key})")
        print(f"{'mol':6} {'ferric':>8} {'pyscf':>8} {'Δ(meV)':>8} | {'Δkoop(meV)':>11}")
        print("-" * 52)
        dg = []
        for mol in MOLS:
            r = rows.get(mol)
            if not r:
                continue
            dg.append(abs(r["d_g0w0_mev"]))
            print(f"{mol:6} {r['fer_g0w0']:8.3f} {r['pys_g0w0']:8.3f} "
                  f"{r['d_g0w0_mev']:+8.1f} | {r['d_koop_mev']:+11.2f}")
        if dg:
            print("-" * 52)
            print(f"N={len(dg)}  MAD={sum(dg)/len(dg):.1f} meV  max={max(dg):.1f} meV")


if __name__ == "__main__":
    if "--show" in sys.argv:
        show()
    else:
        b = next((a for a in sys.argv[1:] if not a.startswith("--")), "def2-tzvp")
        run_all(b)
