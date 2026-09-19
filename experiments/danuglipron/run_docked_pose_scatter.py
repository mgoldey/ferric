#!/usr/bin/env python3
"""M7: does a real pose SEARCH reduce the per-pose scatter that M5/M6 closed?

## Why this probe exists

RESULTS.md M4-M6 measured, on danuglipron against the GLP-1R pocket:

  | route                       | status |
  |-----------------------------|--------|
  | more poses (M4/M5)          | CLOSED -- sd is flat in n; the metric is refuted |
  | relax poses in field (M6)   | CLOSED -- real 15% sd reduction, ~3 orders short |
  | real pose search (docking)  | UNTESTED -- "no docking engine available in this repo" |

That last line is STALE. `tools/docking/vina_dock.py` now wraps AutoDock Vina
(1.2.7) with meeko (0.8.0), both importable in this environment. So the one
route M6 left open is now testable, and this probe tests it.

## The question, stated as a number

M6's parent numbers: rigid overlay sd = 34.23 kcal/mol, relaxed-in-field
sd = 29.07. The gap a candidate RANKING must resolve is 0.25 kcal/mol
(parent vs H1b at n=100), which needs

    n >= 4 * 2 * sd^2 / gap^2

At sd = 29.07 that is ~108,000 poses per candidate. For docking to rescue the
ranking at a plausible n = 100, the required sd is

    sd <= gap * sqrt(n / 8) = 0.25 * sqrt(12.5) = 0.88 kcal/mol

i.e. a **33x** reduction from 29.07. That is the bar, and it is stated here
BEFORE running so the result cannot be graded against a bar chosen afterwards.

## THE ARTIFACT HYPOTHESIS, written before measuring

Per the repo's experimental protocol, the two outcomes must be distinguishable
in advance:

* **If docking genuinely reduces the scatter** -- poses land in one binding
  mode, energies tighten, AND the poses stay geometrically distinct enough that
  the ensemble is still an ensemble. Expect sd down, mean pairwise RMSD down
  but NOT collapsed (M6 saw 3.98 -> 3.84 A, a 4% move, and called that real).

* **If it is an artifact** -- sd falls because every pose collapses onto ONE
  geometry. Then the "ensemble" has n=1 of information, the sd is measuring
  numerical noise rather than pose diversity, and averaging it is meaningless.
  Signature: mean pairwise RMSD collapsing toward ~0.

These predict DIFFERENT things (RMSD stays vs RMSD collapses), so the
experiment can distinguish them. That check is `geometric_spread_angstrom`
below, and it is reported whether or not the sd moves.

A third outcome is live and is NOT a failure of the probe: docking reduces sd
substantially but not by 33x. That closes the route with a number, the same way
M6 closed relaxation with "15%".

## THE COMPARISON MUST USE THE SAME INSTRUMENT

First version of this probe reported Vina's own `vina_score` sd (0.83
kcal/mol) against M6's 29.07 and computed a "35x reduction" that cleared the
bar. **That was a category error and the number is void.** `vina_score` is
documented in `tools/docking/vina_dock.py` as "empirical -- a ranking
heuristic only", on a ~-11 kcal/mol scale; M6's -159.98 is an xtb interaction
energy via `tools.campaign.fit.pose_fit`. Two different quantities on two
different scales have incomparable standard deviations, and the tidiness of
the result (0.83 landing just under an 0.88 bar) was the tell.

So this probe DOCKS with Vina and then RESCORES every pose with the SAME
`pose_fit` M6 used. The sd that is compared is the pose_fit sd. Vina's own
score is still reported, clearly labelled, because the spread of the search
heuristic is interesting -- it is just not the measurement.

## What this does NOT do

It scores ONE candidate (the parent) to characterise the scatter. It does not
rank candidates -- that is the thing the scatter currently forbids. It also
does not re-litigate M5: n is held at the M6 comparison point, because the
question is whether the SD moved, not whether the SEM can be driven down.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DANU = REPO / "testdata/molecules/c9_systems/danuglipron"
POCKET_PDB = DANU / "7LCJ_pocket.pdb"

#: M6's parent measurements, the baseline this probe is compared against.
M6_RIGID_SD = 34.23
M6_RELAXED_SD = 29.07
M6_RELAXED_RMSD = 3.84

#: The gap a candidate ranking must resolve (parent vs H1b, n=100).
RANKING_GAP = 0.25


def required_sd(gap: float, n: int) -> float:
    """sd at which a 2-sigma resolution of `gap` needs only `n` poses.

    Inverts n >= 4 * 2 * sd^2 / gap^2 (two independent means, equal sd).
    """
    return gap * (n / 8.0) ** 0.5


def mean_pairwise_rmsd(coord_sets: list) -> float:
    """Mean all-atom RMSD over every pose pair, in Angstrom.

    This is the ARTIFACT CHECK, not a nicety: a sd reduction that comes with
    this number collapsing toward zero is poses piling onto one geometry, which
    is not a tighter ensemble but a smaller one. M6 reported 3.84 A here.
    """
    import numpy as np

    if len(coord_sets) < 2:
        return float("nan")
    vals = []
    for a, b in combinations(coord_sets, 2):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            continue
        vals.append(float(np.sqrt(((a - b) ** 2).sum(axis=1).mean())))
    return statistics.fmean(vals) if vals else float("nan")


def pocket_box(pdb_path: Path) -> tuple[tuple[float, float, float], float]:
    """Docking box centre (pocket centroid) and a size covering it.

    Derived from the pocket PDB's own atoms rather than hardcoded, so the box
    cannot silently drift from the structure it is supposed to cover.
    """
    xs, ys, zs = [], [], []
    for line in pdb_path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            xs.append(float(line[30:38]))
            ys.append(float(line[38:46]))
            zs.append(float(line[46:54]))
    if not xs:
        raise ValueError(f"no ATOM/HETATM records in {pdb_path}")
    centre = (statistics.fmean(xs), statistics.fmean(ys), statistics.fmean(zs))
    # Cover the pocket with a little margin, clipped to something Vina can
    # search in reasonable time.
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    return centre, min(max(span * 0.6, 20.0), 30.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-poses", type=int, default=20)
    ap.add_argument("--exhaustiveness", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0xF00D)
    ap.add_argument(
        "--cpu",
        type=int,
        default=1,
        help="Vina threads. PINNED, not 0: the search result depends on thread "
        "count even at a fixed seed (vina_dock.dock_ligand's docstring), so a "
        "reproducible probe must fix it.",
    )
    ap.add_argument(
        "--net-charge",
        type=int,
        default=-1,
        help="Ligand net charge for pose_fit. Danuglipron's carboxylic acid is "
        "deprotonated at pH 7.4 -- the ionization state IS the measurement "
        "(see the campaign notes), so this is explicit rather than defaulted to 0.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/danuglipron/out/m7_docked_scatter.json"),
    )
    args = ap.parse_args()

    from experiments.danuglipron.design import DANUGLIPRON_SMILES
    from tools.docking.vina_dock import dock_ligand, prepare_receptor

    t0 = time.time()
    report: dict = {
        "probe": "M7 docked-pose scatter",
        "question": "does a real pose SEARCH reduce the per-pose sd that M5/M6 closed?",
        "baseline_m6": {
            "rigid_sd_kcal": M6_RIGID_SD,
            "relaxed_sd_kcal": M6_RELAXED_SD,
            "relaxed_mean_pairwise_rmsd_angstrom": M6_RELAXED_RMSD,
        },
        "bar_stated_before_running": {
            "ranking_gap_kcal": RANKING_GAP,
            "sd_needed_at_n100_kcal": round(required_sd(RANKING_GAP, 100), 4),
            "reduction_needed_vs_m6": round(
                M6_RELAXED_SD / required_sd(RANKING_GAP, 100), 1
            ),
        },
        "settings": vars(args) | {"out": str(args.out)},
    }

    # --- receptor -----------------------------------------------------------
    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    receptor = out_dir / "7LCJ_pocket.pdbqt"
    try:
        prepare_receptor(POCKET_PDB, receptor)
    except Exception as e:  # noqa: BLE001 -- report, do not half-run
        report["error"] = f"receptor preparation failed: {type(e).__name__}: {e}"
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 1

    centre, size = pocket_box(POCKET_PDB)
    report["box"] = {"centre": [round(c, 3) for c in centre], "size": round(size, 1)}

    # --- ligand -------------------------------------------------------------
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(DANUGLIPRON_SMILES))
    if AllChem.EmbedMolecule(mol, randomSeed=args.seed) != 0:
        report["error"] = "ETKDG failed to embed the parent"
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 1
    AllChem.MMFFOptimizeMolecule(mol)

    # --- dock ---------------------------------------------------------------
    res = dock_ligand(
        mol,
        receptor_pdbqt=receptor,
        box_center=centre,
        box_size=(size, size, size),
        exhaustiveness=args.exhaustiveness,
        n_poses=args.n_poses,
        seed=args.seed,
        cpu=args.cpu,
    )
    if getattr(res, "error", None):
        report["error"] = f"docking failed: {res.error}"
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 1

    poses = list(res.poses)
    # Field names are DockedPose's own: vina_score / coords_angstrom.
    vina_scores = [p.vina_score for p in poses]

    # Vina's heuristic, reported but NOT the measurement (see the module docstring).
    report["vina_heuristic_not_the_measurement"] = {
        "n_poses": len(poses),
        "mean_kcal": round(statistics.fmean(vina_scores), 3) if vina_scores else None,
        "sd_kcal": round(statistics.stdev(vina_scores), 3)
        if len(vina_scores) > 1
        else None,
        "note": "empirical docking score, ~-11 kcal/mol scale; NOT comparable to "
        "M6's pose_fit interaction energy and not used for the verdict",
    }

    # --- RESCORE with the SAME instrument M6 used ---------------------------
    from tools.active_site.pocket_charges import derive_pocket_charges
    from tools.campaign.fit import pose_fit

    from tools.docking.united_atom import restore_hydrogens

    pocket = derive_pocket_charges(str(POCKET_PDB))
    fit_scores, fit_geoms, fit_errors = [], [], []
    for p in poses:
        # PDBQT is UNITED-ATOM (M14): a danuglipron pose has 42 atoms and 263
        # electrons against the real 71 and 292. `pose_fit` accepts that
        # silently -- xtb runs fine on a molecule missing 29 hydrogens and
        # returns a normal-looking number -- while `embed_ligand_from_coords`
        # refuses it. Restore the hydrogens onto the docked heavy-atom frame so
        # the species scored is the real one.
        #
        # This does NOT invalidate the scatter conclusions computed before the
        # fix: the same omission was in every pose, so the SPREAD survives. It
        # is the ABSOLUTE energies that were of an incomplete species.
        try:
            syms, coords = restore_hydrogens(
                DANUGLIPRON_SMILES, list(p.symbols), list(p.coords_angstrom)
            )
        except Exception as exc:  # noqa: BLE001 -- one bad pose must not abort
            fit_errors.append(f"hydrogen restoration: {type(exc).__name__}: {exc}")
            continue
        fr = pose_fit(syms, coords, pocket.charges, charge=args.net_charge)
        if fr.ok:
            fit_scores.append(fr.interaction_kcal)
            fit_geoms.append(p.coords_angstrom)
        else:
            fit_errors.append(fr.error)

    report["docking"] = {
        "n_poses_docked": len(poses),
        "n_poses_rescored": len(fit_scores),
        "n_rescore_failures": len(fit_errors),
        "scores_kcal": [round(s, 3) for s in fit_scores],
        "mean_kcal": round(statistics.fmean(fit_scores), 3) if fit_scores else None,
        "sd_kcal": round(statistics.stdev(fit_scores), 3)
        if len(fit_scores) > 1
        else None,
        "range_kcal": round(max(fit_scores) - min(fit_scores), 3)
        if fit_scores
        else None,
        "geometric_spread_angstrom": round(mean_pairwise_rmsd(fit_geoms), 3)
        if len(fit_geoms) > 1
        else None,
        "instrument": "tools.campaign.fit.pose_fit -- the SAME one M6 used",
    }

    # --- the verdict, computed, not asserted --------------------------------
    sd = report["docking"]["sd_kcal"]
    rmsd = report["docking"]["geometric_spread_angstrom"]
    need = required_sd(RANKING_GAP, 100)
    if sd is not None:
        report["verdict"] = {
            "sd_vs_m6_relaxed": round(sd / M6_RELAXED_SD, 3),
            "sd_needed_at_n100": round(need, 4),
            "short_by_factor": round(sd / need, 1),
            "resolves_ranking_at_n100": bool(sd <= need),
            "artifact_check": (
                "INCONCLUSIVE -- no coordinates returned"
                if rmsd is None
                else (
                    f"poses COLLAPSED (mean pairwise RMSD {rmsd} A): a sd drop here is "
                    "a smaller ensemble, not a tighter one"
                    if rmsd < 0.5
                    else f"poses stay geometrically distinct (mean pairwise RMSD {rmsd} A "
                    f"vs M6's {M6_RELAXED_RMSD} A), so any sd change is energetic, not collapse"
                )
            ),
        }

    report["wall_s"] = round(time.time() - t0, 1)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
