#!/usr/bin/env python3
"""M14: is any AVAILABLE scorer less pose-sensitive than pose_fit?

## Why

M4-M13 closed every pose PROTOCOL: more poses (sd flat in n), relaxing them
(15%), docking them (1%), and selecting one (10x worse than averaging). The
conclusion those four leave is that the per-pose scatter is a property of the
SCORING METRIC, not of the poses -- so the remaining lever is a different
metric.

This tests that directly, on the cheapest possible experiment: take the SAME 15
docked geometries M12 already produced and score them with every scorer this
repo has. If one is materially less pose-sensitive, it is the candidate for
tier 3. If none is, the lever is not in this repo and that is worth knowing
before building anything.

## The comparison must be DIMENSIONLESS

M12's headline error was comparing a Vina-score sd (0.83) against an xtb sd
(28.75) and reporting a "35x reduction" that was a scale difference. Two
scorers on different scales have incomparable standard deviations, full stop.

So the figure of merit here is the COEFFICIENT OF VARIATION, sd/|mean|, which
is dimensionless and comparable across scorers. A scorer that is 10x smaller in
magnitude AND 10x smaller in sd has learned nothing.

Reported alongside it, because CV alone can mislead when a mean sits near zero:
the RANGE/|mean| ratio, and the raw numbers so a reader can check.

## THE ARTIFACT HYPOTHESIS, stated before running

* **If some scorer is genuinely less pose-sensitive** -- its CV is materially
  below pose_fit's, AND its pose-to-pose RANKING correlates with pose_fit's
  (Spearman well above 0). The second half matters: a scorer can be smooth
  because it is insensitive to everything, including the chemistry, and a flat
  constant has CV 0 while being useless.

* **If it is an artifact of scale** -- the CV looks better but the ranking
  correlation is ~0, i.e. the scorer is smooth because it is not measuring the
  same thing. That is Vina's failure mode (r = +0.461 vs RMSD, and its own
  sd 0.83 against xtb's 28.75 on identical geometries).

These predict opposite things about the correlation, so the experiment can
distinguish them. Both are reported whatever the outcome.

## Scope

One candidate (the parent), 15 poses, one pocket. This characterises SCATTER,
not accuracy: none of these scorers has been validated against measured
affinities, and this probe does not change that.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DANU = REPO / "testdata/molecules/c9_systems/danuglipron"
POCKET_PDB = DANU / "7LCJ_pocket.pdb"
M12_JSON = REPO / "experiments/danuglipron/out/m7_docked_scatter.json"
OUT = REPO / "experiments/danuglipron/out/m14_scorer_sensitivity.json"

#: Danuglipron's acid is deprotonated at pH 7.4; the anion/neutral split is
#: 143 kcal/mol, so this is explicit rather than defaulted (campaign rule).
NET_CHARGE = -1


def cv(values: list[float]) -> float | None:
    """Coefficient of variation, sd/|mean| -- dimensionless, so comparable.

    `None` when the mean is too near zero for the ratio to mean anything,
    rather than a huge number that would look like extreme sensitivity.
    """
    if len(values) < 2:
        return None
    m = statistics.fmean(values)
    if abs(m) < 1e-9:
        return None
    return statistics.stdev(values) / abs(m)


def _restore_hydrogens(
    smiles: str,
    heavy_symbols: list[str],
    heavy_coords: list[tuple[float, float, float]],
    rdkit_index_of_heavy: list[int] | None = None,
) -> tuple[list[str], list[tuple[float, float, float]]]:
    """Thin wrapper -- the implementation moved to `tools.docking.united_atom`.

    It lived here while it was one probe's fix. It is now needed by every
    consumer of a docked pose (M12's scatter used raw united-atom coordinates),
    so a copy per script is exactly how the inconsistency survives.
    """
    from tools.docking.united_atom import restore_hydrogens

    return restore_hydrogens(
        smiles,
        heavy_symbols,
        heavy_coords,
        rdkit_index_of_heavy=rdkit_index_of_heavy,
    )


def summarize(name: str, values: list[float]) -> dict:
    if not values:
        return {"scorer": name, "n": 0, "error": "no poses scored"}
    return {
        "scorer": name,
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "sd": round(statistics.stdev(values), 4) if len(values) > 1 else None,
        "range": round(max(values) - min(values), 4) if len(values) > 1 else None,
        "cv": (round(c, 4) if (c := cv(values)) is not None else None),
    }


def main() -> int:
    if not M12_JSON.is_file():
        print(
            f"missing {M12_JSON}\nRun run_docked_pose_scatter.py first (out/ is "
            "gitignored, so this is expected on a fresh checkout).",
            file=sys.stderr,
        )
        return 1

    from rdkit import Chem
    from rdkit.Chem import AllChem

    from experiments.danuglipron.design import DANUGLIPRON_SMILES
    from tools.active_site.ligand_embedding import embed_ligand_from_coords
    from tools.active_site.pocket_charges import derive_pocket_charges
    from tools.active_site.prescreen import prescreen_pose
    from tools.campaign.fit import pose_fit
    from tools.docking.vina_dock import dock_ligand, prepare_receptor

    report: dict = {
        "probe": "M14 scorer pose-sensitivity",
        "question": "is any AVAILABLE scorer less pose-sensitive than pose_fit?",
        "figure_of_merit": "coefficient of variation (sd/|mean|) -- dimensionless, "
        "because comparing raw sds across scorers on different scales is the "
        "category error M12 made",
        "net_charge": NET_CHARGE,
    }

    # --- re-dock to get geometries AND their Vina scores together -----------
    OUT.parent.mkdir(parents=True, exist_ok=True)
    receptor = OUT.parent / "7LCJ_pocket.pdbqt"
    if not receptor.is_file():
        prepare_receptor(POCKET_PDB, receptor)

    prev = json.loads(M12_JSON.read_text())
    centre = tuple(prev["box"]["centre"])
    size = float(prev["box"]["size"])

    mol = Chem.AddHs(Chem.MolFromSmiles(DANUGLIPRON_SMILES))
    if AllChem.EmbedMolecule(mol, randomSeed=0xF00D) != 0:
        report["error"] = "ETKDG failed to embed the parent"
        OUT.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 1
    AllChem.MMFFOptimizeMolecule(mol)

    res = dock_ligand(
        mol,
        receptor_pdbqt=receptor,
        box_center=centre,
        box_size=(size, size, size),
        exhaustiveness=32,
        n_poses=20,
        seed=0xF00D,
        cpu=1,
    )
    if getattr(res, "error", None):
        report["error"] = f"docking failed: {res.error}"
        OUT.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 1
    poses = list(res.poses)

    pocket = derive_pocket_charges(str(POCKET_PDB))

    # Gasteiger charges once: the ligand topology is identical across poses, so
    # recomputing per pose would only add noise of its own.
    gast = Chem.AddHs(Chem.MolFromSmiles(DANUGLIPRON_SMILES))
    AllChem.ComputeGasteigerCharges(gast)
    q = [a.GetDoubleProp("_GasteigerCharge") for a in gast.GetAtoms()]

    vina_s: list[float] = []
    fit_s: list[float] = []
    pre_s: list[float] = []
    errors: list[str] = []
    # Keep them ALIGNED: a pose contributes to the correlation only if every
    # scorer produced a number for it, or the ranks refer to different sets.
    for p in poses:
        fr = pose_fit(p.symbols, p.coords_angstrom, pocket.charges, charge=NET_CHARGE)
        if not fr.ok:
            errors.append(f"pose_fit: {fr.error}")
            continue
        try:
            # RESTORE HYDROGENS before any QM-shaped scorer.
            #
            # PDBQT is UNITED-ATOM: nonpolar hydrogens are merged into their
            # carbons, so a docked danuglipron has 42 atoms and 263 electrons
            # where the real molecule has 71 and 292. 263 is ODD, so a
            # singlet is arithmetically impossible and `embed_ligand_from_coords`
            # refuses -- correctly, since scoring a species that is missing 29
            # hydrogens is not scoring this molecule.
            #
            # `pose_fit` tolerates the united-atom structure (xtb will run on
            # it), which is exactly why this is easy to miss: two scorers on the
            # SAME geometry, one silently accepting an incomplete molecule. M9
            # hit the mirror image of this bug in its alignment check.
            full_syms, full_coords = _restore_hydrogens(
                DANUGLIPRON_SMILES,
                list(p.symbols),
                list(p.coords_angstrom),
                p.rdkit_index_of_heavy,
            )
            el = embed_ligand_from_coords(
                full_syms, full_coords, pocket=pocket, basis="sto-3g"
            )
            if len(q) != el.mol.natoms():
                errors.append(f"charge count {len(q)} != atom count {el.mol.natoms()}")
                continue
            pr = prescreen_pose(el, q)
        except Exception as exc:  # noqa: BLE001 -- one bad pose must not abort
            errors.append(f"prescreen: {type(exc).__name__}: {exc}")
            continue
        vina_s.append(p.vina_score)
        fit_s.append(fr.interaction_kcal)
        pre_s.append(pr.score)

    report["n_poses_docked"] = len(poses)
    report["n_poses_scored_by_all"] = len(fit_s)
    report["scoring_errors"] = errors[:5]
    report["scorers"] = [
        summarize("vina_score (empirical docking heuristic)", vina_s),
        summarize("pose_fit (xtb interaction energy)", fit_s),
        summarize("prescreen_pose (classical field)", pre_s),
    ]

    # --- the discriminator: does a smoother scorer still track pose_fit? ----
    try:
        from scipy.stats import spearmanr

        if len(fit_s) > 2:
            rs_pre, ps_pre = spearmanr(pre_s, fit_s)
            rs_vina, ps_vina = spearmanr(vina_s, fit_s)
            report["rank_correlation_vs_pose_fit"] = {
                "prescreen": {
                    "spearman": round(float(rs_pre), 4),
                    "p": round(float(ps_pre), 4),
                },
                "vina": {
                    "spearman": round(float(rs_vina), 4),
                    "p": round(float(ps_vina), 4),
                },
                "why": "a scorer can be smooth because it is insensitive to "
                "EVERYTHING. A low CV only counts if the ranking still tracks "
                "the reference.",
            }
    except ImportError:
        report["rank_correlation_vs_pose_fit"] = "scipy absent"

    # --- verdict, derived rather than asserted ------------------------------
    fit_cv = cv(fit_s)
    pre_cv = cv(pre_s)
    if fit_cv and pre_cv:
        corr = report.get("rank_correlation_vs_pose_fit", {})
        pre_r = (
            (corr.get("prescreen", {}) or {}).get("spearman")
            if isinstance(corr, dict)
            else None
        )
        better = pre_cv < 0.5 * fit_cv
        tracks = pre_r is not None and abs(pre_r) > 0.5
        report["verdict"] = {
            "prescreen_cv_vs_pose_fit_cv": round(pre_cv / fit_cv, 3),
            "prescreen_is_materially_smoother": bool(better),
            "prescreen_tracks_pose_fit": bool(tracks),
            "conclusion": (
                "CANDIDATE: smoother AND still tracking -- worth evaluating as a "
                "ranking tier"
                if better and tracks
                else "NOT A CANDIDATE: smoother but does not track pose_fit, i.e. "
                "smooth because it measures something else (Vina's failure mode)"
                if better
                else "NOT A CANDIDATE: not materially smoother than pose_fit"
            ),
        }
        # The number that decides whether ANY of this helps: what ddE noise
        # would the best scorer give, against the 0.25 kcal/mol ranking gap?
        # Deliberately NOT "best_sd over all scorers". A raw sd is only
        # comparable within one scorer's scale: prescreen's sd is 0.016 with a
        # mean of -0.0055, so quoting it as the best available noise would
        # report a scale artifact as a breakthrough -- the same category error
        # M12 made with Vina. Use the scorer with the LOWEST CV that still
        # TRACKS the reference; if none does, say so.
        tracking = [
            s
            for s in report["scorers"]
            if s.get("cv") is not None and s["scorer"].startswith("pose_fit")
        ]
        if tracks and pre_cv < fit_cv:
            tracking.append(
                next(s for s in report["scorers"] if "prescreen" in s["scorer"])
            )
        usable = min(tracking, key=lambda s: s["cv"], default=None)
        if usable is not None and usable.get("sd") is not None:
            report["verdict"]["usable_scorer"] = usable["scorer"]
            report["verdict"]["usable_sd"] = usable["sd"]
            report["verdict"]["ddE_noise_at_n100"] = round(
                (usable["sd"] / math.sqrt(100)) * math.sqrt(2), 4
            )
            report["verdict"]["note"] = (
                "sd is from the lowest-CV scorer that still TRACKS pose_fit. A "
                "smaller raw sd from a non-tracking scorer is a scale artifact, "
                "not less noise."
            )

    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
