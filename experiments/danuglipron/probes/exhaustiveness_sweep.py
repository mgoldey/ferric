#!/usr/bin/env python
"""What does docking thoroughness actually buy, and what does it cost?

`exhaustiveness` is tier 1's dominant cost knob and the least justified number
in the pipeline. Two different values are in use with no measurement behind
either:

  * `dock_ligand`'s default is **32**, justified in its docstring as "the
    failure being fixed is a SEARCH failure, and under-searching would
    reproduce it" -- a reasonable prior, never tested.
  * the screening pipeline runs at **16**, chosen for speed.

So the setting VALIDATED by redocking (32 -> 0.95 A, RESULTS.md M9) is not the
setting used to screen. This probe closes that gap by measuring both axes on
the one system where ground truth exists.

## The design

Redock danuglipron into 7LCJ at a range of exhaustiveness values and record,
for each: wall seconds, best-of-N RMSD to the crystal pose, and top-ranked
RMSD. The ligand is re-embedded from SMILES each time with a fixed seed, so the
search starts from a geometry carrying no information about the answer.

**Artifact hypothesis, written before running.** If thoroughness genuinely
matters, RMSD should fall with exhaustiveness and plateau somewhere. If the
pose is easy for this pocket, RMSD is flat from the cheapest setting up -- and
16 vs 32 is a pure waste of half the tier-1 budget. Those predict different
curves, so the experiment can distinguish them.

A THIRD outcome is possible and would be the most interesting: RMSD flat but
top-1 RANK unstable, meaning the search finds the pose reliably and the scoring
function orders it by luck. That is M9's finding (r(score,RMSD)=+0.461) showing
up as a function of search effort.

Seeds: Vina's search is stochastic. One seed per setting measures a single
draw, so `--repeats` runs several seeds and reports the spread. A single-seed
sweep would report noise as trend -- the exact error this campaign made once
already by reading a pose RANGE as a precision (RESULTS.md M6).

Usage:
    uv run --no-sync python experiments/danuglipron/probes/exhaustiveness_sweep.py \
        [--levels 4,8,16,32] [--repeats 3] [--n-poses 20]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve()
while not (_root / "pyproject.toml").is_file():
    if _root.parent == _root:
        raise RuntimeError("could not locate the repo root")
    _root = _root.parent
sys.path.insert(0, str(_root))

import numpy as np  # noqa: E402

from tools.campaign.align import align_to_reference  # noqa: E402
from tools.campaign.strain import load_xyz_ensemble  # noqa: E402
from tools.docking import dock_ligand, prepare_receptor  # noqa: E402
from experiments.danuglipron.design import DANUGLIPRON_SMILES  # noqa: E402

RECEPTOR_PDB = _root / "testdata/molecules/c9_systems/danuglipron/7LCJ_pocket.pdb"
RECEPTOR_PDBQT = _root / "experiments/danuglipron/out/7LCJ_receptor.pdbqt"
ENSEMBLE = _root / "testdata/molecules/c9_systems/danuglipron"
OUT = _root / "experiments/danuglipron/out/exhaustiveness_sweep.json"


def _embed(seed: int):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(DANUGLIPRON_SMILES))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    p.useSmallRingTorsions = True
    if AllChem.EmbedMolecule(mol, p) != 0:
        raise RuntimeError("ETKDG failed to embed the ligand")
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="4,8,16,32")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-poses", type=int, default=20)
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",")]

    ens = load_xyz_ensemble(ENSEMBLE)
    ref_i = ens.labels.index("conf_00_cryo_em")
    ref = np.asarray(ens.conformers[ref_i])
    ref_symbols = list(ens.symbols_per_conformer[ref_i])
    center = tuple(float(x) for x in ref.mean(axis=0))
    size = tuple(float(x) for x in (ref.max(axis=0) - ref.min(axis=0) + 8.0))

    receptor = prepare_receptor(RECEPTOR_PDB, RECEPTOR_PDBQT)
    print(f"receptor {Path(receptor).name}  box centre {np.round(center,1)}\n")

    rows = []
    for ex in levels:
        per_seed = []
        for rep in range(args.repeats):
            seed = 0xF00D + rep
            mol = _embed(seed)
            t0 = time.time()
            res = dock_ligand(mol, receptor, center, size,
                              exhaustiveness=ex, n_poses=args.n_poses, seed=seed)
            dt = time.time() - t0
            if not res.ok:
                print(f"  ex={ex:3d} seed={seed} FAILED: {res.error}", flush=True)
                continue
            rmsds = []
            for pose in res.poses:
                # Both sides are danuglipron -- this is a REDOCK -- but the
                # PDBQT pose and the crystal ensemble use different atom
                # orderings, so alignment must go through the SMILES and match
                # by element+connectivity, never by index.
                al = align_to_reference(
                    DANUGLIPRON_SMILES, pose.symbols, pose.coords_angstrom,
                    DANUGLIPRON_SMILES, ref_symbols,
                    [tuple(float(x) for x in row) for row in ref])
                if al.ok:
                    rmsds.append(al.rmsd_angstrom)
            if not rmsds:
                print(f"  ex={ex:3d} seed={seed}: no alignable pose", flush=True)
                continue
            per_seed.append({"seed": seed, "seconds": dt,
                             "best_rmsd": min(rmsds), "top1_rmsd": rmsds[0],
                             "n_aligned": len(rmsds),
                             "top1_score": res.poses[0].vina_score})
            print(f"  ex={ex:3d} seed={seed}  {dt:6.1f} s  "
                  f"best {min(rmsds):.2f} A  top1 {rmsds[0]:.2f} A", flush=True)
        if not per_seed:
            continue
        secs = [r["seconds"] for r in per_seed]
        best = [r["best_rmsd"] for r in per_seed]
        top1 = [r["top1_rmsd"] for r in per_seed]
        rows.append({
            "exhaustiveness": ex,
            "n_seeds": len(per_seed),
            "seconds_mean": statistics.mean(secs),
            "best_rmsd_mean": statistics.mean(best),
            "best_rmsd_max": max(best),
            "top1_rmsd_mean": statistics.mean(top1),
            "top1_rmsd_max": max(top1),
            # SEM, not range: precision of a mean falls as 1/sqrt(n); a range
            # GROWS with n (RESULTS.md M6).
            "top1_rmsd_sem": (statistics.stdev(top1) / len(top1) ** 0.5
                              if len(top1) > 1 else None),
            "runs": per_seed,
        })

    print(f"\n{'exhaust':>8s} {'s/dock':>8s} {'best RMSD':>10s} {'top1 RMSD':>10s} "
          f"{'top1 SEM':>9s}")
    for r in rows:
        sem = "-" if r["top1_rmsd_sem"] is None else f"{r['top1_rmsd_sem']:.2f}"
        print(f"{r['exhaustiveness']:8d} {r['seconds_mean']:8.1f} "
              f"{r['best_rmsd_mean']:10.2f} {r['top1_rmsd_mean']:10.2f} {sem:>9s}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"levels": levels, "repeats": args.repeats,
                               "n_poses": args.n_poses, "rows": rows}, indent=1))
    print(f"\nwrote {OUT.relative_to(_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
