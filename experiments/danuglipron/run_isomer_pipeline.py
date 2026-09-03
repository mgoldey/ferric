"""End-to-end: enumerate danuglipron isomers, funnel them through all four tiers.

This is the pipeline the campaign lacked. Previously: 11 hand-written analogues,
scored by tier 3 alone, on geometries that had never seen the receptor
(RESULTS.md M7). Now: enumerated candidates, docked, relaxed, GFN2-ranked, and
finally DFT'd -- with every tier's survivors, rejects and failures recorded.

## Cost, measured on this box

    tier 1  Vina           26.4 s/dock at exhaustiveness 4, cpu=0 (MEASURED)
                           109.0 s at cpu=1 -- but 10 run concurrently, x3 seeds
    tier 2  MMFF94         ~ms
    tier 3  GFN2-xTB       ~0.5 s single point
    tier 4  ferric DFT     612.4 s / 18 iterations, converged, at 71 atoms
                           (MEASURED 2026-09-02; the earlier ">57 min, did not
                           finish" was memory contention, not cost -- M10)

**Tier 1 dominates, not tier 4.** It is the only tier that touches every
candidate: 60 ligands x 26.4 s is ~26 min, while tier 4 sees KEEPS[-1]=3 and
costs ~31 min. Those are the same order, which is what a well-balanced funnel
looks like -- and is only visible because every tier is now TIMED. The previous
run recorded a single 3326 s total and no way to attribute it.

Exhaustiveness is 4, not 16: measured on this target, 4 / 8 / 16 differ by
<=0.03 A in redock RMSD against a 0.13 A between-seed SEM, so the extra search
buys nothing resolvable at 2.6x the price (RESULTS.md M11).

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
import os
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
from tools.campaign.rank import tier_agreement  # noqa: E402
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
        # 4, not 16: RESULTS.md M11 measured 4/8/16 as indistinguishable on
        # this target (<=0.03 A vs a 0.13 A between-seed SEM) at 2.6x the cost.
        # Raise it for a NEW target until a redock says otherwise -- this is a
        # measurement about 7LCJ, not a universal setting.
        "exhaustiveness": 4,
        # 1 core per dock, because the stage below fans out across ligands.
        # Vina's own default (0) takes every core, and its internal
        # parallelism runs at only 34% efficiency (M11).
        "vina_cpu": 1,
        # 3 independent ETKDG embeddings per ligand. M11 measured the starting
        # conformer as the variable that MOVES redock RMSD (0.75-1.24 A across
        # seeds) while exhaustiveness does not (0.097 A across an 8x range,
        # below the 0.131 A between-seed SEM). Spending the budget freed by
        # ex=16 -> 4 on seeds instead is strictly better sampling, and the whole
        # screen is still ~2x faster than the single-seed ex=16 configuration
        # it replaces.
        "n_seeds": 3,
        "n_poses": 10,
        # 6-31G, not def2-SVP. Sized with tools/pipeline/cost.py BEFORE
        # spending the compute, which is the whole reason that model exists:
        #
        #   basis      nbf   AO cache   fits 9 GB   est/candidate
        #   sto-3g     234     4.32 GB        yes        10.2 min
        #   6-31g      434     8.02 GB        yes        35.1 min
        #   def2-svp   719    13.29 GB         NO        96.3 min
        #
        # def2-SVP would need 13.29 GB of resident AO cache -- more than the
        # pinned budget and more than this box has free -- so it would fall to
        # the batching path or OOM, and cost ~4.8 h for KEEPS[-1]=3. That is
        # M10's failure mode exactly. 6-31G is a genuine step up from minimal
        # basis and fits with headroom.
        "basis": "6-31g",
        "functional": "PBE",
        # PIN the DFT memory budget. ferric's default is 0.8 x *live*
        # MemAvailable, which makes the AO-cache Full-vs-Batched decision depend
        # on whatever else is running -- and an under-resolved budget is exactly
        # what turned tier 4's real ~10 min into M10's ">57 min, did not finish"
        # (it paged until the kernel OOM-killed it). 9 GB fits the 4.4 GB AO
        # cache plus SCF state with headroom on this 23 GB box.
        "mem_budget_gb": 9,
    }

    # DOCK_WORKERS env override: the 10 default assumes a quiet 12-core box
    # (M11). It is a real measured number for THAT condition, not a knob to
    # edit here for a transient one -- override it at invocation time instead
    # when something else is already on the machine.
    dock_workers = int(os.environ.get("DOCK_WORKERS", "10"))

    stages = [
        # workers=10 on a 12-core box: fan-out beats Vina's internal threading
        # above ~4 workers (M11), and leaving 2 cores free keeps the box usable
        # for whoever else is on it.
        Stage(Tier.SEARCH, tier1_dock, keep=KEEPS[0], name="dock",
              workers=dock_workers),
        Stage(Tier.FORCE_FIELD, tier2_forcefield, keep=KEEPS[1], name="mmff"),
        Stage(Tier.SEMIEMPIRICAL, tier3_gfn2, keep=KEEPS[2], name="gfn2"),
        Stage(Tier.QUANTUM, tier4_dft, keep=KEEPS[3], name="dft"),
    ]

    print(f"\n=== FUNNEL ({len(cands)} candidates) ===", flush=True)
    report = run_funnel(cands, stages, context)
    print(report.table())

    # ── did the expensive tier change the cheap tier's mind? ──
    #
    # Compared over the candidates BOTH tiers actually scored. The previous
    # version compared a GFN2 top-N list against the DFT survivor list, so when
    # DFT failed on everything the lists differed by LENGTH and it announced
    # "tier 4 reordered tier 3's ranking: True -> DFT is load-bearing" having
    # computed nothing (RESULTS.md M10). `tier_agreement` returns None rather
    # than a verdict when fewer than 2 candidates are common.
    gfn2_scores = {c.canonical: report.value("gfn2", c.canonical)
                   for c in cands if report.value("gfn2", c.canonical) is not None}
    dft_scores = {c: v for c in gfn2_scores
                  if (v := report.value("dft", c)) is not None}
    agreement = tier_agreement(gfn2_scores, dft_scores)

    print("\n=== SURVIVORS ===")
    for n, iso in enumerate(report.survivors, 1):
        print(f"  {n}. {iso.transform:24s} gfn2 {report.value('gfn2', iso.canonical)}"
              f"  dft {report.value('dft', iso.canonical)}")

    reordered = agreement["reordered"]
    print(f"\ntier 4 vs tier 3 on the {agreement['n_common']} candidates both scored:")
    if reordered is None:
        print(f"  -> UNTESTABLE: {agreement['note']}")
    elif reordered:
        print(f"  -> DFT REORDERED GFN2 (tau={agreement['kendall_tau']:.3f}); "
              f"tier 4 is load-bearing here")
    else:
        print(f"  -> GFN2 ordering survived DFT (tau={agreement['kendall_tau']:.3f}); "
              f"tier 4 is skippable for this system")

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
        "tier4_vs_tier3": agreement,
        "wall_seconds": time.time() - t_start,
    }, indent=2))
    print(f"\nwrote {OUT}  ({time.time() - t_start:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
