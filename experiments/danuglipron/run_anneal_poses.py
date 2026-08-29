"""Does in-pocket simulated annealing reach the bound pose?

## The question

Conformer generation failed (RESULTS.md M7): the best of 20 ETKDG conformers is
2.23 A from the experimentally determined bound pose, against a 2.0 A
docking-success bar, and 18 of 20 are 2.7-3.7 A. Geometry OPTIMIZATION cannot
fix it -- measured improvement 0.04-0.16 A -- because relaxation settles bonds
and angles and stays in its starting basin. The error is TORSIONAL.

Per-atom breakdown of the best pose (2.23 A overall, 41/41 heavy atoms): median
deviation 1.76 A, max 5.08 A, with 24 of 41 atoms ALREADY under 2 A. So the
scaffold is largely right and the error is concentrated in a minority of atoms
-- a mispositioned tail, not a wrong binding mode.

MD at elevated temperature crosses torsional barriers, so it *can* change basin.
Run in the pocket's field, the receptor biases which torsions are populated.
This asks whether that is enough.

## Artifact hypotheses, stated before measuring

- *If annealing works:* the best-frame RMSD drops below the rigid-overlay
  starting value, ideally under 2.0 A, and the improvement is larger than the
  0.04-0.16 A that plain relaxation already achieves.
- *If it is thermal noise:* frames scatter around the starting pose with no
  systematic improvement -- the best frame beats the start only as much as
  random perturbation would, and the MEAN over frames does not improve.
- *If the field is doing nothing:* vacuum and in-field annealing give
  statistically indistinguishable best-RMSDs. This is why both are run.

Reporting only the best frame could not separate the first two, so the mean and
the full distribution are reported alongside.

## Cost

Each anneal is thousands of GFN2 gradient calls on a 70-atom anion. Scoped to
the PARENT only, where the answer is unambiguous: the bound pose is known
exactly, so "did we get closer" needs no modelling assumptions.

Run:
    LD_LIBRARY_PATH=$HOME/.local/lib/x86_64-linux-gnu:$HOME/.local/lib \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    uv run --no-sync python experiments/danuglipron/run_anneal_poses.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve()
while not (_root / "pyproject.toml").is_file():
    if _root.parent == _root:
        raise RuntimeError("could not locate the repo root")
    _root = _root.parent
sys.path.insert(0, str(_root))

from tools.active_site.pocket_charges import derive_pocket_charges  # noqa: E402
from tools.campaign.align import (  # noqa: E402
    DOCKING_SUCCESS_RMSD_ANGSTROM,
    align_to_reference,
)
from tools.campaign.fit import DEFAULT_FIELD_CUTOFF_BOHR, _trim_charges  # noqa: E402
from tools.campaign.strain import load_xyz_ensemble  # noqa: E402
from tools.campaign.xtb_engine import anneal, verify_xtb_build  # noqa: E402
from experiments.danuglipron.design import (  # noqa: E402
    DANUGLIPRON_SMILES,
    danuglipron_analogues,
)
from tools.morph.embed import embed_analogue  # noqa: E402

POCKET_PDB = _root / "testdata/molecules/c9_systems/danuglipron/7LCJ_pocket.pdb"
ENSEMBLE = _root / "testdata/molecules/c9_systems/danuglipron"
OUT = _root / "experiments/danuglipron/out/anneal_poses.json"

N_STARTS = 3            # independent starting conformers
PICOSECONDS = 4.0
TEMPERATURE_K = 500.0
DUMP_FS = 200.0         # -> 20 frames per anneal
ANGSTROM_TO_BOHR = 1.0 / 0.529_177_210_92


def rmsds_to_bound(smiles, symbols, frames, ref_symbols, ref_coords):
    """Scaffold RMSD of every frame against the bound pose."""
    out = []
    for f in frames:
        al = align_to_reference(smiles, symbols, f, DANUGLIPRON_SMILES,
                                ref_symbols, ref_coords)
        if al.ok:
            out.append(al.rmsd_angstrom)
    return out


def main() -> int:
    ok, err = verify_xtb_build()
    if not ok:
        print(f"ABORT: {err}")
        return 1
    print("xtb build check: PASS", flush=True)

    pocket = derive_pocket_charges(POCKET_PDB)
    ens = load_xyz_ensemble(ENSEMBLE)
    i = ens.labels.index("conf_00_cryo_em")
    ref_symbols, ref_coords = ens.symbols_per_conformer[i], ens.conformers[i]
    ana = next(a for a in danuglipron_analogues() if a.label == "parent")
    print(f"pocket: {pocket.n_charges} charges; target bar "
          f"{DOCKING_SUCCESS_RMSD_ANGSTROM:.1f} A", flush=True)

    emb = embed_analogue(ana, n_conformers=N_STARTS)
    if not emb.usable:
        print(f"ABORT: embedding failed -- {emb.error}")
        return 1

    results = {"n_starts": N_STARTS, "picoseconds": PICOSECONDS,
               "temperature_k": TEMPERATURE_K, "starts": []}

    for k, coords in enumerate(emb.conformers):
        al = align_to_reference(ana.scoring_smiles, emb.symbols, coords,
                                DANUGLIPRON_SMILES, ref_symbols, ref_coords)
        if not al.ok:
            print(f"start {k}: alignment failed -- {al.error}", flush=True)
            continue
        start_rmsd = al.rmsd_angstrom
        print(f"\n=== start {k}: rigid overlay is {start_rmsd:.2f} A from bound ===",
              flush=True)

        lig_bohr = [(x * ANGSTROM_TO_BOHR, y * ANGSTROM_TO_BOHR, z * ANGSTROM_TO_BOHR)
                    for x, y, z in al.coords_angstrom]
        near = _trim_charges(pocket.charges, lig_bohr, DEFAULT_FIELD_CUTOFF_BOHR)

        rec = {"start_index": k, "start_rmsd": start_rmsd,
               "n_pocket_charges": len(near)}
        for tag, pc in (("in_field", near), ("vacuum", None)):
            t0 = time.time()
            frames, run = anneal(
                emb.symbols, al.coords_angstrom, charge=ana.net_charge,
                point_charges=pc, temperature_k=TEMPERATURE_K,
                picoseconds=PICOSECONDS, dump_every_fs=DUMP_FS,
                skip_build_check=True,
            )
            dt = time.time() - t0
            if not run.ok:
                print(f"  {tag:9s} FAILED: {run.error}", flush=True)
                rec[tag] = {"error": run.error}
                continue
            rs = rmsds_to_bound(ana.scoring_smiles, emb.symbols, frames,
                                ref_symbols, ref_coords)
            if not rs:
                rec[tag] = {"error": "no frame could be aligned"}
                continue
            rec[tag] = {
                "n_frames": len(rs), "best": min(rs), "mean": st.mean(rs),
                "worst": max(rs), "improvement_vs_start": start_rmsd - min(rs),
                "wall_seconds": dt, "all_rmsds": rs,
            }
            print(f"  {tag:9s} {len(rs):2d} frames  best {min(rs):5.2f}  "
                  f"mean {st.mean(rs):5.2f}  worst {max(rs):5.2f}  "
                  f"(start {start_rmsd:.2f}, gain {start_rmsd - min(rs):+.2f})  "
                  f"{dt:.0f}s", flush=True)
        results["starts"].append(rec)

    # ── verdict ──
    print("\n" + "=" * 74)
    fielded = [r["in_field"] for r in results["starts"] if "best" in r.get("in_field", {})]
    vac = [r["vacuum"] for r in results["starts"] if "best" in r.get("vacuum", {})]
    if fielded:
        best_overall = min(f["best"] for f in fielded)
        mean_gain = st.mean(f["improvement_vs_start"] for f in fielded)
        results["best_in_field_rmsd"] = best_overall
        results["mean_improvement"] = mean_gain
        results["reached_docking_bar"] = bool(best_overall < DOCKING_SUCCESS_RMSD_ANGSTROM)
        print(f"best in-field frame:        {best_overall:.2f} A "
              f"({'UNDER' if best_overall < DOCKING_SUCCESS_RMSD_ANGSTROM else 'over'} "
              f"the {DOCKING_SUCCESS_RMSD_ANGSTROM:.1f} A bar)")
        print(f"mean improvement vs start:  {mean_gain:+.2f} A")
        print(f"  (plain relaxation achieves only 0.04-0.16 A, RESULTS.md M7)")
        if vac:
            vb = min(v["best"] for v in vac)
            results["best_vacuum_rmsd"] = vb
            print(f"best VACUUM frame:          {vb:.2f} A")
            print(f"  field advantage:          {vb - best_overall:+.2f} A "
                  f"({'field helps' if vb > best_overall else 'field does NOT help'})")
    else:
        print("no in-field anneal produced an alignable frame")
    print("=" * 74)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
