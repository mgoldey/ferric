"""Can Vina reproduce a pose we already hold? (self-validating tier-1 test)

## Why this is the first thing to run

RESULTS.md M7 established that pose GENERATION failed: free-solution conformers
sit 2.2-4.1 A from the experimentally determined bound pose, and neither
optimization (+0.04-0.16 A) nor 4 ps GFN2 MD (+0.41 A) closes the gap. The
diagnosis is that nothing in the pipeline ever SEARCHED pose space.

Vina supplies that search. But a docking pose is a hypothesis, and the honest
way to find out whether the search works on THIS target is to point it at a
question we already know the answer to: redock danuglipron into 7LCJ and measure
the RMSD against the crystal pose sitting in testdata.

This is cheap (seconds to minutes) and unambiguous. If Vina cannot reproduce a
pose we hold in hand, the analogue set is not worth attempting.

## Pass condition, stated before running

Vina's published redocking success rate is ~60-80% of cases under 2.0 A. So:

- **PASS**: best-of-N pose is < 2.0 A from the bound pose. Pose generation is
  solved for this target, and the tier-2/3/4 hierarchy has real geometries.
- **MARGINAL**: 2.0-2.5 A. Better than everything tried so far (2.41 A from MD),
  but still short of the conventional bar -- report as such, do not round down.
- **FAIL**: > 2.5 A. Docking does not solve this target either, and the problem
  is the receptor/box/protonation rather than the search algorithm.

## Artifact hypothesis

If the box or receptor is wrong, Vina will still return poses -- with plausible
scores -- that are nowhere near the site. So this reports the distance from each
pose's centroid to the known ligand centroid alongside the RMSD. A pose 10 A
away with a good score is a setup error, not a docking result.

Run:
    OPENBLAS_NUM_THREADS=1 uv run --no-sync python \
      experiments/danuglipron/run_redock_validation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_root = Path(__file__).resolve()
while not (_root / "pyproject.toml").is_file():
    if _root.parent == _root:
        raise RuntimeError("could not locate the repo root")
    _root = _root.parent
sys.path.insert(0, str(_root))

import numpy as np  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from tools.campaign.align import align_to_reference  # noqa: E402
from tools.campaign.strain import load_xyz_ensemble  # noqa: E402
from tools.docking import dock_ligand, prepare_receptor  # noqa: E402
from experiments.danuglipron.design import DANUGLIPRON_SMILES  # noqa: E402

RECEPTOR_PDB = _root / "testdata/molecules/c9_systems/danuglipron/7LCJ_pocket.pdb"
ENSEMBLE = _root / "testdata/molecules/c9_systems/danuglipron"
OUT = _root / "experiments/danuglipron/out/redock_validation.json"
RECEPTOR_PDBQT = _root / "experiments/danuglipron/out/7LCJ_receptor.pdbqt"

BAR_PASS, BAR_MARGINAL = 2.0, 2.5


def main() -> int:
    ens = load_xyz_ensemble(ENSEMBLE)
    i = ens.labels.index("conf_00_cryo_em")
    ref_symbols, ref_coords = ens.symbols_per_conformer[i], ens.conformers[i]
    ref = np.asarray(ref_coords)
    center = tuple(float(x) for x in ref.mean(axis=0))
    extent = ref.max(axis=0) - ref.min(axis=0)
    # Box: ligand extent plus 8 A of headroom per axis, so the search has room
    # to place the ligand differently rather than being forced onto the answer.
    size = tuple(float(x) for x in (extent + 8.0))
    print(f"bound-pose centroid : {np.round(center,2)}")
    print(f"search box size     : {np.round(size,1)} A", flush=True)

    print(f"\npreparing receptor from {RECEPTOR_PDB.name} ...", flush=True)
    try:
        rec = prepare_receptor(RECEPTOR_PDB, RECEPTOR_PDBQT)
    except Exception as e:  # noqa: BLE001
        print(f"ABORT: receptor preparation failed -- {type(e).__name__}: {e}")
        return 1
    print(f"  wrote {rec.name}", flush=True)

    # Ligand: the ANION (pH 7.4 species, per RESULTS.md M5), embedded fresh so
    # the search starts from a geometry that carries NO information about the
    # bound pose -- otherwise the test would be circular.
    smi = DANUGLIPRON_SMILES.replace("C(=O)O", "C(=O)[O-]")
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    if AllChem.EmbedMolecule(mol, params) != 0:
        print("ABORT: could not embed the ligand")
        return 1
    AllChem.MMFFOptimizeMolecule(mol)
    print(f"ligand: {mol.GetNumAtoms()} atoms, "
          f"charge {Chem.GetFormalCharge(mol):+d}", flush=True)

    print("\ndocking ...", flush=True)
    res = dock_ligand(mol, rec, center, size, exhaustiveness=32, n_poses=20)
    if not res.ok:
        print(f"ABORT: {res.error}")
        return 1
    print(f"  {len(res.poses)} poses returned", flush=True)

    rows = []
    print(f"\n{'rank':>4s} {'vina':>7s} {'RMSD':>7s} {'centroid dev':>13s}")
    for p in res.poses:
        al = align_to_reference(smi, p.symbols, p.coords_angstrom,
                                DANUGLIPRON_SMILES, ref_symbols, ref_coords)
        rmsd = al.rmsd_angstrom if al.ok else None
        dev = float(np.linalg.norm(np.asarray(p.coords_angstrom).mean(axis=0)
                                   - ref.mean(axis=0)))
        rows.append({"rank": p.rank, "vina_score": p.vina_score,
                     "rmsd_to_bound": rmsd, "centroid_dev": dev,
                     "align_error": None if al.ok else al.error})
        print(f"{p.rank:4d} {p.vina_score:7.2f} "
              f"{(f'{rmsd:7.2f}' if rmsd is not None else '    n/a')} "
              f"{dev:13.2f}")

    good = [r for r in rows if r["rmsd_to_bound"] is not None]
    best = min((r["rmsd_to_bound"] for r in good), default=None)
    top1 = next((r["rmsd_to_bound"] for r in rows if r["rank"] == 0), None)

    print("\n" + "=" * 66)
    if best is None:
        verdict = "NO ALIGNABLE POSE"
    elif best < BAR_PASS:
        verdict = "PASS"
    elif best < BAR_MARGINAL:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"
    print(f"best-of-{len(res.poses)} RMSD : {best if best is None else round(best,2)} A")
    print(f"top-ranked pose RMSD: {top1 if top1 is None else round(top1,2)} A")
    print(f"VERDICT: {verdict}   (pass <{BAR_PASS}, marginal <{BAR_MARGINAL} A)")
    if good:
        near = sum(1 for r in good if r["centroid_dev"] < 5.0)
        print(f"poses within 5 A of the known site: {near}/{len(good)}"
              + ("" if near else "   <- SETUP ERROR: box or receptor is wrong"))
    print("=" * 66)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "receptor": str(RECEPTOR_PDB.relative_to(_root)),
        "box_center": center, "box_size": size,
        "exhaustiveness": 32, "n_poses": len(res.poses),
        "best_rmsd": best, "top1_rmsd": top1, "verdict": verdict,
        "bar_pass": BAR_PASS, "bar_marginal": BAR_MARGINAL,
        "poses": rows,
    }, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
