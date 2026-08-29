"""Does relaxing a pose IN the pocket field tighten the fit distribution?

## The question

`run_fit.py` scores rigid, MCS-overlaid poses. Their per-pose fit spread is
~46-50 kcal/mol (sd), which is what makes the estimator imprecise: a rigid
overlay drops an analogue's scaffold onto the parent's placement with no chance
to settle, so some poses land in clashing or badly-oriented local arrangements
that a real bound complex would never adopt.

If the spread is dominated by such unrelaxed poses, then relaxing each pose in
the pocket's point-charge field should pull them toward a common basin and
SHRINK the distribution. If instead the spread is real conformational diversity
of genuinely distinct binding arrangements, relaxation will move energies down
but leave the spread roughly intact.

Those two predictions differ, so the probe is admissible.

## Artifact hypothesis, stated before measuring

- *If relaxation genuinely helps:* sd falls substantially AND the relaxed poses
  converge toward fewer distinct geometries (lower pairwise RMSD spread).
- *If it is an artifact:* sd falls simply because every pose collapses onto the
  SAME geometry regardless of where it started -- which would mean we are no
  longer sampling poses at all, just re-finding one minimum. That would show up
  as near-zero RMSD spread among relaxed poses, and it would make the "improved
  precision" meaningless.

So the probe records BOTH the energy spread and the geometric spread. Reporting
only the energy spread could not distinguish them.

Run:
    LD_LIBRARY_PATH=$HOME/.local/lib/x86_64-linux-gnu:$HOME/.local/lib \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    uv run --no-sync python scripts/danuglipron/run_pose_relax_probe.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from tools.active_site.pocket_charges import derive_pocket_charges  # noqa: E402
from tools.campaign.align import align_to_reference  # noqa: E402
from tools.campaign.fit import DEFAULT_FIELD_CUTOFF_BOHR, _trim_charges, pose_fit  # noqa: E402
from tools.campaign.strain import load_xyz_ensemble  # noqa: E402
from tools.campaign.xtb_engine import relax, verify_xtb_build  # noqa: E402
from tools.morph.design import DANUGLIPRON_SMILES, danuglipron_analogues  # noqa: E402
from tools.morph.embed import embed_analogue  # noqa: E402

POCKET_PDB = "testdata/molecules/c9_systems/danuglipron/7LCJ_pocket.pdb"
ENSEMBLE = "testdata/molecules/c9_systems/danuglipron"
OUT = Path("scripts/danuglipron/out/pose_relax_probe.json")

# The two candidates whose separation the metric gate turns on. Keeping the set
# small matters: each pose here costs a full in-field GEOMETRY OPTIMIZATION of a
# 70+ atom anion, not a single point.
LABELS = ["parent", "NC2-decyano"]
N_POSES = 12
ANGSTROM_TO_BOHR = 1.0 / 0.529_177_210_92


def pairwise_rmsd_spread(poses: list[list[tuple[float, float, float]]]) -> float | None:
    """Mean pairwise all-atom RMSD over a set of poses, in Angstrom.

    No superposition: these are all in the SAME pocket frame, so the raw
    coordinate difference is the physically meaningful one (superposing would
    erase exactly the placement differences we are trying to measure).
    """
    if len(poses) < 2:
        return None
    arrs = [np.asarray(p) for p in poses]
    vals = []
    for i in range(len(arrs)):
        for j in range(i + 1, len(arrs)):
            vals.append(float(np.sqrt(((arrs[i] - arrs[j]) ** 2).sum(axis=1).mean())))
    return st.mean(vals)


def main() -> int:
    ok, err = verify_xtb_build()
    if not ok:
        print(f"ABORT: {err}")
        return 1
    print("xtb build check: PASS", flush=True)

    pocket = derive_pocket_charges(POCKET_PDB)
    ens = load_xyz_ensemble(ENSEMBLE)
    i_ref = ens.labels.index("conf_00_cryo_em")
    ref_symbols, ref_coords = ens.symbols_per_conformer[i_ref], ens.conformers[i_ref]
    print(f"pocket: {pocket.n_charges} charges", flush=True)

    out = {
        "n_poses": N_POSES,
        "method": "GFN2-xTB; rigid overlay vs in-field relaxed, same poses",
        "pocket_pdb": POCKET_PDB,
        "candidates": {},
    }

    for label in LABELS:
        ana = next(a for a in danuglipron_analogues() if a.label == label)
        print(f"\n=== {label} (q={ana.net_charge}) ===", flush=True)
        emb = embed_analogue(ana, n_conformers=N_POSES)
        if not emb.usable:
            print(f"  UNEVALUATED: {emb.error}")
            continue

        rigid_fits, relaxed_fits = [], []
        rigid_geoms, relaxed_geoms = [], []
        t0 = time.time()
        for k, coords in enumerate(emb.conformers):
            al = align_to_reference(
                ana.scoring_smiles, emb.symbols, coords,
                DANUGLIPRON_SMILES, ref_symbols, ref_coords,
            )
            if not al.ok:
                continue

            fr = pose_fit(al.symbols, al.coords_angstrom, pocket.charges,
                          charge=ana.net_charge)
            if not fr.ok:
                continue
            rigid_fits.append(fr.interaction_kcal)
            rigid_geoms.append(al.coords_angstrom)

            # Relax IN the field, then score at the relaxed geometry.
            lig_bohr = [(x * ANGSTROM_TO_BOHR, y * ANGSTROM_TO_BOHR, z * ANGSTROM_TO_BOHR)
                        for x, y, z in al.coords_angstrom]
            near = _trim_charges(pocket.charges, lig_bohr, DEFAULT_FIELD_CUTOFF_BOHR)
            rx = relax(al.symbols, al.coords_angstrom, charge=ana.net_charge,
                       point_charges=near, skip_build_check=True)
            if not rx.ok or rx.coords_angstrom is None:
                print(f"  pose {k:02d}: relax failed ({rx.error})", flush=True)
                continue
            fr2 = pose_fit(al.symbols, rx.coords_angstrom, pocket.charges,
                           charge=ana.net_charge)
            if fr2.ok:
                relaxed_fits.append(fr2.interaction_kcal)
                relaxed_geoms.append(rx.coords_angstrom)
            print(f"  pose {k:02d}: rigid {fr.interaction_kcal:+8.2f} -> "
                  f"relaxed {fr2.interaction_kcal:+8.2f} kcal/mol"
                  if fr2.ok else f"  pose {k:02d}: rescoring failed", flush=True)

        dt = time.time() - t0
        rec = {"n_rigid": len(rigid_fits), "n_relaxed": len(relaxed_fits),
               "wall_seconds": dt, "net_charge": ana.net_charge}
        for tag, fits, geoms in (("rigid", rigid_fits, rigid_geoms),
                                 ("relaxed", relaxed_fits, relaxed_geoms)):
            if len(fits) > 1:
                rec[f"{tag}_mean"] = st.mean(fits)
                rec[f"{tag}_sd"] = st.stdev(fits)
                rec[f"{tag}_sem"] = st.stdev(fits) / len(fits) ** 0.5
                rec[f"{tag}_rmsd_spread_angstrom"] = pairwise_rmsd_spread(geoms)
            rec[f"{tag}_fits"] = fits
        out["candidates"][label] = rec

        if "rigid_sd" in rec and "relaxed_sd" in rec:
            print(f"  -> rigid   mean {rec['rigid_mean']:+8.2f}  sd {rec['rigid_sd']:6.2f}  "
                  f"geom spread {rec['rigid_rmsd_spread_angstrom']:.2f} A")
            print(f"  -> relaxed mean {rec['relaxed_mean']:+8.2f}  sd {rec['relaxed_sd']:6.2f}  "
                  f"geom spread {rec['relaxed_rmsd_spread_angstrom']:.2f} A")
            print(f"  ({dt:.0f}s)")

    # The verdict, with the artifact check attached.
    both = [l for l in LABELS if l in out["candidates"]
            and "relaxed_sd" in out["candidates"][l]]
    if len(both) == 2:
        p, n = (out["candidates"][l] for l in both)
        for tag in ("rigid", "relaxed"):
            gap = abs(p[f"{tag}_mean"] - n[f"{tag}_mean"])
            se = (p[f"{tag}_sem"] ** 2 + n[f"{tag}_sem"] ** 2) ** 0.5
            out[f"{tag}_gap_kcal"] = gap
            out[f"{tag}_se_diff_kcal"] = se
            out[f"{tag}_resolved_2sigma"] = bool(gap >= 2 * se)
            print(f"\n{tag:8s}: gap {gap:6.2f}  se_diff {se:5.2f}  "
                  f"2-sigma resolved: {gap >= 2 * se}")
        collapse = min(p["relaxed_rmsd_spread_angstrom"], n["relaxed_rmsd_spread_angstrom"])
        out["geometric_collapse_warning"] = bool(collapse < 0.5)
        if collapse < 0.5:
            print("\nWARNING: relaxed poses collapsed to nearly one geometry "
                  f"(mean pairwise RMSD {collapse:.2f} A). Any precision gain is "
                  "then an artifact -- we are re-finding one minimum, not sampling.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
