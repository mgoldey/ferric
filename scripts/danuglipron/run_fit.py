"""Arms A/B: pocket electrostatic fit for the parent ensemble and the analogues.

Pipeline per candidate:
  1. embed SMILES -> conformers (RDKit ETKDGv3 + MMFF94, fixed seed)
  2. align each conformer's MCS scaffold onto the 7LCJ bound pose (Kabsch)
  3. GFN2-xTB vacuum and in-pocket-field single points at that geometry
  4. report the best (most negative) interaction energy

Then the metric GATE: the two pharmacophore-breaking negative controls must
score clearly worse than the parent. If they do not, the fit metric is tracking
molecular size and no candidate ranking is reported.

Run:
    LD_LIBRARY_PATH=$HOME/.local/lib/x86_64-linux-gnu:$HOME/.local/lib \
    OPENBLAS_NUM_THREADS=1 uv run --no-sync python scripts/danuglipron/run_fit.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.active_site.pocket_charges import derive_pocket_charges  # noqa: E402
from tools.campaign.align import align_to_reference  # noqa: E402
from tools.campaign.fit import pose_fit  # noqa: E402
from tools.campaign.rank import (  # noqa: E402
    Candidate,
    fit_discriminates_controls,
    format_table,
    noise_exceeds_signal,
)
from tools.campaign.strain import load_xyz_ensemble  # noqa: E402
from tools.campaign.xtb_engine import verify_xtb_build  # noqa: E402
from tools.morph.design import DANUGLIPRON_SMILES, danuglipron_analogues  # noqa: E402
from tools.morph.embed import embed_analogue  # noqa: E402
from tools.tox.alerts import RdkitAlertsProvider  # noqa: E402
from tools.tox.assess import assess_smiles  # noqa: E402

POCKET_PDB = "testdata/molecules/c9_systems/danuglipron/7LCJ_pocket.pdb"
ENSEMBLE = "testdata/molecules/c9_systems/danuglipron"
OUT = Path("scripts/danuglipron/out/fit_and_rank.json")
# Pose count per candidate. Measured convergence of the fit estimator
# (out/convergence.log, 2026-08-29): the standard error of the mean falls as
# 1/sqrt(n) -- 5.66 -> 3.33 -> 1.99 -> 1.35 kcal/mol at n = 5, 10, 20, 40 -- so
# the poses ARE samples from one distribution and averaging does converge. 40
# poses gives ~1.4 kcal/mol precision, which is the scale needed to resolve a
# pharmacophore contact. 6 (the earlier value) gave ~5 kcal/mol and could not.
N_CONF = 40


def main() -> int:
    ok, err = verify_xtb_build()
    if not ok:
        print(f"ABORT: {err}")
        return 1
    print("xtb build check: PASS", flush=True)

    print(f"deriving pocket charges from {POCKET_PDB} ...", flush=True)
    pocket = derive_pocket_charges(POCKET_PDB)
    print(f"  {pocket.n_charges} charges, net {sum(c[0] for c in pocket.charges):+.3f} e",
          flush=True)

    ens = load_xyz_ensemble(ENSEMBLE)
    i_ref = ens.labels.index("conf_00_cryo_em")
    ref_symbols = ens.symbols_per_conformer[i_ref]
    ref_coords = ens.conformers[i_ref]

    # Free-conformer strain, from Arm A (run_arm_a_free.py), if available.
    strain_by_label: dict[str, float] = {}
    free_json = Path("scripts/danuglipron/out/arm_a_free.json")
    free_min = None
    if free_json.exists():
        d = json.loads(free_json.read_text())
        free_min = d["free_min_energy_ha"]
        print(f"  free minimum from Arm A: {d['free_min_label']} "
              f"({free_min:.6f} Ha), spread {d['spread_kcal_mol']:.2f} kcal/mol",
              flush=True)

    tox_provider = RdkitAlertsProvider()
    candidates: list[Candidate] = []
    records = []

    print(f"\nscoring {len(danuglipron_analogues())} candidates "
          f"x up to {N_CONF} conformers ...\n", flush=True)

    for ana in danuglipron_analogues():
        t0 = time.time()
        cand = Candidate(
            label=ana.label, smiles=ana.smiles, hypothesis=ana.hypothesis,
            is_negative_control=ana.is_negative_control,
        )

        tox = assess_smiles(ana.smiles, providers=[tox_provider], label=ana.label)
        cand.liability = tox.liability_score

        # EVERY candidate -- the parent included -- goes through the SAME
        # embed -> align -> score path.
        #
        # WHY: the first run of this script scored the parent at its committed
        # cryo-EM pose (1 pose, RMSD 0.00 A) and every analogue at the best of 6
        # re-embedded, rigidly re-aligned poses (RMSD 1.9-3.5 A). Per-pose fits
        # within one analogue span up to 64 kcal/mol (H3b: -45.6 to +18.6), so
        # taking a minimum over 6 noisy samples against the parent's single
        # sample is a selection bias worth tens of kcal/mol. It made all nine
        # analogues AND both pharmacophore-breaking negative controls look
        # better than the parent -- which is exactly how the metric gate caught
        # it. Identical treatment is the fix; the parent's cryo-EM pose is still
        # scored, but as a separately reported reference rather than as the
        # parent's entry in the comparison.
        emb = embed_analogue(ana, n_conformers=N_CONF)
        if not emb.usable:
            cand.notes.append(f"embedding failed: {emb.error}")
            candidates.append(cand)
            records.append({"label": ana.label, "error": emb.error})
            print(f"{ana.label:30s} UNEVALUATED (embedding): {emb.error[:60]}",
                  flush=True)
            continue
        poses = []
        for k, coords in enumerate(emb.conformers):
            al = align_to_reference(
                ana.scoring_smiles, emb.symbols, coords,
                DANUGLIPRON_SMILES, ref_symbols, ref_coords,
            )
            if al.ok:
                poses.append((f"conf_{k:02d}", al.symbols, al.coords_angstrom,
                              al.rmsd_angstrom))
            else:
                cand.notes.append(f"conf_{k:02d} alignment failed: {al.error}")

        if not poses:
            cand.notes.append("no conformer could be aligned into the pocket")
            candidates.append(cand)
            records.append({"label": ana.label, "error": "no aligned pose"})
            print(f"{ana.label:30s} UNEVALUATED (alignment)", flush=True)
            continue

        fits = []
        for pose_label, syms, coords, rmsd in poses:
            fr = pose_fit(syms, coords, pocket.charges, label=pose_label,
                          charge=ana.net_charge)
            fits.append((fr, rmsd))

        good = [(f, r) for f, r in fits if f.ok]
        if not good:
            first_err = fits[0][0].error
            cand.notes.append(f"all fits failed: {first_err}")
            candidates.append(cand)
            records.append({"label": ana.label, "error": first_err})
            print(f"{ana.label:30s} UNEVALUATED (fit): {str(first_err)[:60]}", flush=True)
            continue

        best, best_rmsd = min(good, key=lambda fr: fr[0].interaction_kcal)
        values = [f.interaction_kcal for f, _ in good]
        mean_fit = sum(values) / len(values)
        spread = max(values) - min(values)

        # The RANKING axis is the MEAN over a fixed number of aligned poses, not
        # the minimum. A minimum is a biased estimator whose bias grows with the
        # number of samples drawn, so it is only comparable between candidates
        # that happened to yield the same pose count -- which they do not. The
        # mean over equal pose counts is comparable; the min is reported
        # alongside for transparency but is not ranked on.
        cand.fit_kcal = mean_fit
        # SEM, not the range: the precision of a MEAN falls as 1/sqrt(n) while a
        # range grows with n. Verified 1/sqrt(n) in out/convergence.log.
        import statistics as _st

        cand.fit_sem_kcal = (
            _st.stdev(values) / (len(values) ** 0.5) if len(values) > 1 else None
        )
        cand.fit_range_kcal = spread
        if spread > 20.0:
            cand.notes.append(
                f"pose-to-pose fit spread {spread:.1f} kcal/mol over "
                f"{len(good)} poses -- the rigid-overlay pose is poorly "
                "determined for this analogue, so its fit number is imprecise"
            )

        dt = time.time() - t0
        print(f"{ana.label:30s} fit_mean={mean_fit:+8.2f}  (min {min(values):+8.2f}, "
              f"spread {spread:5.1f}) n={len(good)}/{len(fits)} "
              f"rmsd {best_rmsd:.2f} A {dt:.0f}s"
              + ("  [NEG-CTRL]" if ana.is_negative_control else ""), flush=True)

        candidates.append(cand)
        records.append({
            "label": ana.label,
            "smiles": ana.smiles,
            "hypothesis": ana.hypothesis,
            "is_negative_control": ana.is_negative_control,
            "liability": cand.liability,
            "net_charge": ana.net_charge,
            "scoring_smiles": ana.scoring_smiles,
            "best_pose": best.label,
            "scaffold_rmsd_angstrom": best_rmsd,
            "fit_kcal_mol": mean_fit,
            "fit_mean_kcal_mol": mean_fit,
            "fit_min_kcal_mol": min(values),
            "fit_spread_kcal_mol": spread,
            "fit_sem_kcal_mol": cand.fit_sem_kcal,
            "e_vacuum_ha": best.e_vacuum,
            "e_in_field_ha": best.e_in_field,
            "n_pocket_charges": best.n_pocket_charges,
            "n_poses_scored": len(good),
            "all_pose_fits_kcal": [f.interaction_kcal for f, _ in good],
            "notes": cand.notes,
        })

    # ── the cryo-EM bound pose, reported separately ──
    #
    # Not part of the candidate comparison (it is a single pose from a different
    # provenance than the re-embedded ensembles, which is precisely the
    # apples-to-oranges the first run tripped on), but it IS the one geometry
    # here that is experimentally determined, so its score is worth recording as
    # a reference point.
    # Scored as the ANION, matching every real candidate: the carboxyl proton is
    # located geometrically (the only H within 1.15 A of an O) and removed.
    # Scoring the experimental pose as a neutral acid while the candidates are
    # anions is exactly the inconsistency that invalidated the earlier runs.
    import numpy as np

    rc = np.asarray(ref_coords)
    oh = [
        (float(np.linalg.norm(rc[h] - rc[o])), h)
        for h, s_h in enumerate(ref_symbols) if s_h == "H"
        for o, s_o in enumerate(ref_symbols) if s_o == "O"
        if float(np.linalg.norm(rc[h] - rc[o])) < 1.15
    ]
    cryo_neutral = pose_fit(ref_symbols, ref_coords, pocket.charges,
                            label="conf_00_cryo_em_neutral")
    cryo = cryo_neutral
    if oh:
        _, h_idx = min(oh)
        anion_symbols = [s_ for k, s_ in enumerate(ref_symbols) if k != h_idx]
        anion_coords = [c_ for k, c_ in enumerate(ref_coords) if k != h_idx]
        cryo = pose_fit(anion_symbols, anion_coords, pocket.charges,
                        label="conf_00_cryo_em_anion", charge=-1)
    if cryo.ok:
        print(f"\n[reference] experimental cryo-EM pose (7LCJ), ANION: "
              f"{cryo.interaction_kcal:+.2f} kcal/mol, "
              f"{cryo.n_pocket_charges} charges", flush=True)
    if cryo_neutral.ok:
        print(f"[reference] same pose as NEUTRAL acid (for contrast): "
              f"{cryo_neutral.interaction_kcal:+.2f} kcal/mol", flush=True)

    # ── the gate ──
    print("\n" + "=" * 78)
    precision = noise_exceeds_signal(candidates)
    print("PRECISION CHECK:", "PASS" if precision.passed else "FAIL")
    print(" ", precision.detail)
    print()
    gate = fit_discriminates_controls(candidates)
    print("METRIC GATE:", "PASS" if gate.passed else "FAIL")
    print(" ", gate.detail)
    print("=" * 78 + "\n")

    print(format_table(candidates))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "pocket_pdb": POCKET_PDB,
        "n_pocket_charges": pocket.n_charges,
        "method_fit": "GFN2-xTB in-field minus vacuum, same geometry",
        "method_align": "rigid Kabsch on MCS scaffold vs 7LCJ bound pose (NOT docking)",
        "method_tox": "RDKit FilterCatalog alert sets + Lipinski/Veber (offline baseline)",
        "n_conformers_per_analogue": N_CONF,
        "free_min_energy_ha": free_min,
        "gate_passed": gate.passed,
        "gate_detail": gate.detail,
        "gate_parent_fit_kcal": gate.parent_fit,
        "gate_control_fits_kcal": gate.control_fits,
        "precision_passed": precision.passed,
        "precision_detail": precision.detail,
        "ranking_axis": "mean interaction energy over a fixed number of aligned poses "
                        "(NOT the min, which is a biased estimator whose bias grows "
                        "with sample count)",
        "cryo_em_reference_fit_kcal_anion": cryo.interaction_kcal if cryo.ok else None,
        "cryo_em_reference_fit_kcal_neutral": (
            cryo_neutral.interaction_kcal if cryo_neutral.ok else None
        ),
        "ionization_note": (
            "every candidate retaining an ionizable acid/bioisostere is scored as "
            "its pH-7.4 ANION (net charge -1); NC1-methyl-ester cannot ionize and "
            "is scored neutral, which is the hypothesis that control tests"
        ),
        "candidates": records,
    }, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
