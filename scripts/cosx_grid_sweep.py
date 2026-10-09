"""COSX grid sweep: error vs exact exchange, points per atom and wall time.

The confirmation experiment behind ferric's COSX grid defaults
(site/src/methods/scf.md, "Choosing how exchange is built"). For each system
it runs exact-J + exact-K RHF (the reference) and exact-J + COSX RHF on
several grids, so the energy difference is the COSX error alone (RI-J would
add its own, separable error; see crates/ferric-scf/tests/cosx_rijcosx.rs).

Pre-registered (from wiki/notes/cosx-default-grids-survey-2026-10-01.md):
if the pruned 194-peak grid is genuinely better, it beats the old flat
(50,110) default on EVERY system at fewer points and meets 0.1 kcal/mol on
the isodesmic C3H8 + CH4 -> 2 C2H6 reaction; if butane was a lucky sign
cancellation it loses on at least one system.

Run (heavy; through the CPU slot, with a private shim to this worktree's
build of the bindings):
    scripts/validation/run_slot.sh -- env PYTHONPATH=<shim> \\
        ~/qc/ferric/.venv/bin/python scripts/cosx_grid_sweep.py --out <file.jsonl>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import ferric

ROOT = Path(__file__).resolve().parent.parent
MOL = ROOT / "testdata" / "molecules"

SYSTEMS = [
    ("water", MOL / "validation" / "h2o.xyz", "aug-cc-pvdz"),
    ("butane", MOL / "alkane_4.xyz", "def2-svp"),
    ("butane", MOL / "alkane_4.xyz", "def2-tzvp"),
    ("benzene", MOL / "benzene.xyz", "def2-svp"),
    ("sulfamethoxazole", MOL / "sulfamethoxazole.xyz", "def2-svp"),
    ("methane", MOL / "alkane_1.xyz", "cc-pvdz"),
    ("ethane", MOL / "alkane_2.xyz", "cc-pvdz"),
    ("propane", MOL / "alkane_3.xyz", "cc-pvdz"),
]

# name -> run_rhf COSX kwargs (None = exact K)
ARMS = {
    "exact": None,
    "flat(50,110)+fit [old default]": {"cosx_grid": (50, 110)},
    "sgx(35,194)+fit": {"cosx_grid": (35, 194, "sgx")},
    "sgx(50,194)+fit": {"cosx_grid": (50, 194, "sgx")},
    "flat(50,194)+fit": {"cosx_grid": (50, 194)},
    "sgx(35,194) nofit": {"cosx_grid": (35, 194, "sgx"), "cosx_overlap_fit": False},
    "sgx(50,302)+fit": {"cosx_grid": (50, 302, "sgx")},
    "sgx(35,194)+fit, final sgx(50,302)": {
        "cosx_grid": (35, 194, "sgx"),
        "cosx_final_grid": (50, 302, "sgx"),
    },
}


def run(mol, bs, kw):
    args = dict(energy_conv=1e-9, density_conv=1e-8, max_iter=200)
    if kw is not None:
        args.update(k_builder="cosx", cosx_final_pass=False, **kw)
        if "cosx_final_grid" in kw:
            args.pop("cosx_final_pass")
    t0 = time.perf_counter()
    r = ferric.run_rhf(mol, bs, **args)
    wall = time.perf_counter() - t0
    assert r.converged, kw
    fp = r.cosx_final_pass
    return r.energy, r.iterations, wall, fp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    out = open(args.out, "a")
    for name, xyz, basis in SYSTEMS:
        if args.only and args.only != name:
            continue
        mol = ferric.Molecule.from_xyz(str(xyz))
        bs = ferric.BasisSet.bundled(basis)
        natoms = sum(1 for line in open(xyz).read().splitlines()[2:] if line.strip())
        for arm, kw in ARMS.items():
            e, it, wall, fp = run(mol, bs, kw)
            npts = None
            if kw is not None:
                npts = ferric.cosx_grid_point_count(mol, kw["cosx_grid"])
            rec = {
                "system": name,
                "basis": basis,
                "natoms": natoms,
                "arm": arm,
                "energy": e,
                "iterations": it,
                "wall_s": round(wall, 2),
                "npts": npts,
                "final_pass": fp,
            }
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
