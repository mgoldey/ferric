"""End-to-end: enumerate danuglipron isomers, funnel them through all four tiers.

This is the pipeline the campaign lacked. Previously: 11 hand-written analogues,
scored by tier 3 alone, on geometries that had never seen the receptor
(RESULTS.md M7). Now: enumerated candidates, docked, relaxed, GFN2-ranked, and
finally DFT'd -- with every tier's survivors, rejects and failures recorded.

## Cost, measured on this box

    tier 1  Vina           ~15-30 s/ligand at exhaustiveness 16
    tier 2  MMFF94         ~ms
    tier 3  GFN2-xTB       ~0.5 s single point
    tier 4  ferric DFT     96.1 s at 32 atoms -> 17-37 min at 70 (N^3-N^4)

Tier 4 dominates: with KEEPS[-1] = 3 it is roughly 1-2 hours, and everything
above it is minutes. That asymmetry is exactly why the funnel narrows hard
before reaching it.

## What to look for in the output

The interesting question is not which candidate wins -- no ranking is licensed
until the fit metric itself is validated (RESULTS.md M5). It is whether **tier 4
reorders tier 3's survivors**. If DFT changes the order, the cheap tiers cannot
substitute for it. If it does not, tier 4 is skippable for this system, which is
equally worth knowing and much cheaper to act on.

Run:
    LD_LIBRARY_PATH=$HOME/.local/lib/x86_64-linux-gnu:$HOME/.local/lib \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    uv run --no-sync python experiments/danuglipron/run_isomer_pipeline.py
"""
from __future__ import annotations

import json
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

from tools.campaign.hierarchy import Tier  # noqa: E402
from tools.campaign.strain import load_xyz_ensemble  # noqa: E402
from tools.docking import prepare_receptor  # noqa: E402
from tools.isomers import enumerate_with_report  # noqa: E402
from tools.pipeline import Stage, run_funnel  # noqa: E402
from tools.pipeline.tiers import (  # noqa: E402
    tier1_dock, tier2_forcefield, tier3_gfn2, tier4_dft,
)
from experiments.danuglipron.design import DANUGLIPRON_SMILES  # noqa: E402

RECEPTOR_PDB = _root / "testdata/molecules/c9_systems/danuglipron/7LCJ_pocket.pdb"
RECEPTOR_PDBQT = _root / "experiments/danuglipron/out/7LCJ_receptor.pdbqt"
ENSEMBLE = _root / "testdata/molecules/c9_systems/danuglipron"
OUT = _root / "experiments/danuglipron/out/isomer_pipeline.json"

# Survivors passed down from each tier. Tuned so tier 4 sees only a handful.
KEEPS = (24, 12, 5, 3)
SEED = 0xF00D
MAX_CANDIDATES = 60


def main() -> int:
    t_start = time.time()

    # ── enumerate ──
    cands, rep = enumerate_with_report(
        DANUGLIPRON_SMILES, max_candidates=MAX_CANDIDATES, mw_range=(300.0, 700.0)
    )
    print("=== ENUMERATION ===")
    print(f"  generated {rep.n_generated} -> dedup {rep.n_after_dedup} "
          f"-> filtered {rep.n_after_filter}")
    for r in rep.rejected[:5]:
        print(f"    rejected: {r[:95]}")
    from collections import Counter
    print(f"  by kind: {dict(Counter(i.kind for i in cands))}")

    # Score the pH-7.4 species (RESULTS.md M5: neutral-vs-anion was a
    # 143 kcal/mol error). Deprotonate the STRUCTURE rather than just declaring
    # a charge -- H+ leaves its electrons behind, so an anion has the same
    # electron count as its acid, and a neutral SMILES tagged charge=-1 asks
    # the solver for an electron that does not exist. That killed the first
    # run of this pipeline at tier 4 with "325 electrons ... n_alpha = 325/2".
    # A candidate with no acidic proton stays neutral rather than being forced.
    prepared, n_anion = [], 0
    for c in cands:
        anion = c.deprotonated()
        if anion is not None:
            prepared.append(anion)
            n_anion += 1
        else:
            prepared.append(c)
    cands = prepared
    print(f"  ionization: {n_anion}/{len(cands)} deprotonated to the pH-7.4 anion")

    # ── receptor + box ──
    ens = load_xyz_ensemble(ENSEMBLE)
    i = ens.labels.index("conf_00_cryo_em")
    ref = np.asarray(ens.conformers[i])
    center = tuple(float(x) for x in ref.mean(axis=0))
    size = tuple(float(x) for x in (ref.max(axis=0) - ref.min(axis=0) + 8.0))
    print(f"\n=== RECEPTOR ===\n  preparing {RECEPTOR_PDB.name} ...", flush=True)
    receptor = prepare_receptor(RECEPTOR_PDB, RECEPTOR_PDBQT)
    print(f"  {receptor.name}; box centre {np.round(center,1)} size {np.round(size,1)}")

    context = {
        "seed": SEED,
        "receptor_pdbqt": receptor,
        "box_center": center,
        "box_size": size,
        "exhaustiveness": 16,
        "n_poses": 10,
        "basis": "def2-svp",
        "functional": "PBE",
    }

    stages = [
        Stage(Tier.SEARCH, tier1_dock, keep=KEEPS[0], name="dock"),
        Stage(Tier.FORCE_FIELD, tier2_forcefield, keep=KEEPS[1], name="mmff"),
        Stage(Tier.SEMIEMPIRICAL, tier3_gfn2, keep=KEEPS[2], name="gfn2"),
        Stage(Tier.QUANTUM, tier4_dft, keep=KEEPS[3], name="dft"),
    ]

    print(f"\n=== FUNNEL ({len(cands)} candidates) ===", flush=True)
    report = run_funnel(cands, stages, context)
    print(report.table())

    # ── did the expensive tier change the cheap tier's mind? ──
    gfn2_order = [i.canonical for i in sorted(
        [c for c in cands if report.value("gfn2", c.canonical) is not None],
        key=lambda c: report.value("gfn2", c.canonical))][:KEEPS[3]]
    dft_order = [i.canonical for i in report.survivors]
    reordered = gfn2_order != dft_order

    print("\n=== SURVIVORS ===")
    for n, iso in enumerate(report.survivors, 1):
        print(f"  {n}. {iso.transform:24s} gfn2 {report.value('gfn2', iso.canonical)}"
              f"  dft {report.value('dft', iso.canonical)}")
    print(f"\ntier 4 reordered tier 3's ranking: {reordered}")
    print("  -> DFT is load-bearing here" if reordered else
          "  -> GFN2 ordering survived DFT; tier 4 is skippable for this system")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "parent": DANUGLIPRON_SMILES,
        "seed": SEED,
        "keeps": list(KEEPS),
        "enumeration": {"n_generated": rep.n_generated,
                        "n_after_dedup": rep.n_after_dedup,
                        "n_after_filter": rep.n_after_filter,
                        "rejected": rep.rejected},
        "outcomes": [{"tier": int(o.tier), "n_in": o.n_in, "n_out": o.n_out,
                      "n_failed": o.n_failed, "note": o.note,
                      "errors": o.errors} for o in report.outcomes],
        "results": {stage: [{"id": r.candidate_id, "value": r.value,
                             "error": r.error} for r in rs]
                    for stage, rs in report.results.items()},
        "survivors": [{"smiles": i.canonical, "kind": i.kind,
                       "transform": i.transform,
                       "gfn2": report.value("gfn2", i.canonical),
                       "dft": report.value("dft", i.canonical)}
                      for i in report.survivors],
        "tier4_reordered_tier3": reordered,
        "wall_seconds": time.time() - t_start,
    }, indent=2))
    print(f"\nwrote {OUT}  ({time.time() - t_start:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
