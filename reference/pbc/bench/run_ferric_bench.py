"""Python-bindings alternative to run_ferric_bench.sh (FINDINGS "Real-size benchmark plan — 2026-09-25").

Use it when the CLI lacks a number: the CLI does not print the RS-GDF aux counts, while
run_rhf_gamma / run_rks_gamma return `naux`, `naux_kept`, `n_dropped` (the aux-metric drop count
that must agree with PySCF's `aux_metric` record for the energies to be comparable). It also
splits wall time into "call" (hcore + RS-GDF + grid + SCF: the bindings expose no finer split)
and reports iterations, so time per iteration needs the CLI run log.

Loads THIS worktree's bindings through a PYTHONPATH shim (out/pyshim/ferric -> target/release/
libferric.so; build it with `cargo build --release -p ferric-python`), never the main checkout's
shared site-packages symlink (CLAUDE.md). Runs in the main checkout's venv (numpy only needed).

Usage (from the worktree root, one at a time, under the memory cap):
  scripts/ferric-limited --max=12G -- env OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
    <main checkout>/.venv/bin/python reference/pbc/bench/run_ferric_bench.py diamond_prim_rhf
Runs are the toml/*.toml basenames; the TOML is read for cell/basis/aux/kind/functional/kmesh.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BOHR = 0.52917721092


def shim():
    so = os.path.join(ROOT, "target", "release", "libferric.so")
    if not os.path.exists(so):
        sys.exit(f"no {so}; cargo build --release -p ferric-python in {ROOT}")
    pkg = os.path.join(HERE, "out", "pyshim", "ferric")
    os.makedirs(pkg, exist_ok=True)
    link = os.path.join(pkg, "ferric.cpython-311-x86_64-linux-gnu.so")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(so, link)
    with open(os.path.join(pkg, "__init__.py"), "w") as f:
        f.write("from .ferric import *\n__doc__ = ferric.__doc__\n")
    sys.path.insert(0, os.path.dirname(pkg))
    import ferric

    assert os.path.realpath(ferric.ferric.__file__) == os.path.realpath(so), (
        ferric.ferric.__file__
    )
    return ferric


def peak_rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def main(run):
    cfg = tomllib.load(open(os.path.join(HERE, "toml", run + ".toml"), "rb"))
    ferric = shim()
    cell = cfg["cell"]
    lat_ang = [
        [v * BOHR for v in row] for row in cell["lattice"]
    ]  # bindings take Angstrom rows
    mol = ferric.Molecule.from_xyz(os.path.join(ROOT, cfg["molecule"]["xyz"]))
    bs = ferric.BasisSet.bundled(cfg["basis"]["name"])
    budget = cfg.get("memory", {}).get("budget_gb")
    common = dict(
        exxdiv=cell["exxdiv"],
        jk="rsgdf",
        auxbasis=cell["auxbasis"],
        memory_budget_gb=budget,
    )
    t0 = time.perf_counter()
    kind = cfg["method"]["kind"]
    if "kmesh" in cell:
        r = ferric.run_rhf_kpts(
            mol,
            lat_ang,
            bs,
            tuple(cell["kmesh"]),
            centring=cell.get("centring", "gamma"),
            **common,
        )
    elif kind == "ksdft":
        r = ferric.run_rks_gamma(
            mol,
            lat_ang,
            bs,
            cfg["dft"]["functional"],
            n_radial=cell.get("n_radial", 75),
            n_angular=cell.get("n_angular", 302),
            **common,
        )
    else:
        r = ferric.run_rhf_gamma(mol, lat_ang, bs, **common)
    wall = time.perf_counter() - t0
    rec = dict(
        code="ferric-python",
        run=run,
        wall_call_s=wall,
        peak_rss_bytes=peak_rss(),
        rayon_threads=os.environ.get("RAYON_NUM_THREADS"),
        loadavg_end=os.getloadavg(),
    )
    for k in (
        "energy",
        "converged",
        "iterations",
        "nao",
        "naux",
        "naux_kept",
        "n_dropped",
        "n_grid_points",
    ):
        if hasattr(r, k):
            v = getattr(r, k)
            rec[k] = v() if callable(v) else v
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    with open(os.path.join(HERE, "out", "ferric_python_summary.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
