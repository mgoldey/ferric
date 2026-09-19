#!/usr/bin/env python3
"""M16: does GFN2-xTB RANK conformers the way DFT does?

## Why this is the missing number

The golden path costs tier 3 (~0.5 s/pose) and says it "ranks survivors". The
only accuracy claim attached to it is a **143 kcal/mol** anion/neutral split --
a split so large that resolving it says almost nothing about whether xtb can
order two conformers a few kcal/mol apart, which is what ranking survivors
actually means.

This measures that directly: score the same conformer set with GFN2-xTB and
with ferric's DFT, and compare the ORDERINGS.

## Why rank correlation, not energy agreement

Absolute xtb and DFT energies are not comparable -- different Hamiltonians,
different zeros. Even their differences carry a systematic offset. What tier 3
is asked for is an ORDER: which survivors go up to tier 4. So the figure of
merit is Spearman rank correlation, plus the mean absolute error on relative
energies for scale.

This also avoids the category error M12 made (comparing standard deviations
across two scorers on different scales).

## THE ARTIFACT HYPOTHESIS, before running

* **If xtb is a usable ranker** -- Spearman well above 0 (say > 0.7), so the
  conformers it promotes are the ones DFT would have promoted.
* **If it is not** -- Spearman near 0, meaning tier 3 is filtering on noise and
  the funnel would do as well picking at random. That is a real possibility:
  M14 already showed one cheap scorer (Vina) that is smooth and uncorrelated.

A third outcome is live: STRONG correlation on big gaps and none on small ones,
which would say tier 3 is a coarse gate and not a ranker. The per-pair table
below is printed so that is visible rather than averaged away.

## Scope

One molecule, one conformer set, one basis. This characterises whether the
tiers AGREE, not whether either is right -- neither is validated against
experiment here, and this probe does not change that.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT = REPO / "experiments/danuglipron/out/m16_xtb_vs_dft.json"

HARTREE_TO_KCAL = 627.5094740631

#: Small enough that DFT on ~8 conformers is minutes, drug-like enough that the
#: conformers differ by torsions rather than trivia. Not danuglipron: 71 atoms
#: at ~600 s per DFT point would be hours for one probe.
DEFAULT_SMILES = "CC(=O)Nc1ccc(OCCN(C)C)cc1"  # paracetamol-like, one flexible tail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smiles", default=DEFAULT_SMILES)
    ap.add_argument("--n-conformers", type=int, default=8)
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--seed", type=int, default=0xF00D)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    import ferric
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from tools.campaign.xtb_engine import relax, verify_xtb_build

    ok, err = verify_xtb_build()
    if not ok:
        print(f"xtb unusable: {err}", file=sys.stderr)
        return 1

    report: dict = {
        "probe": "M16 xtb-vs-DFT conformer ranking",
        "question": "does GFN2-xTB rank conformers the way ferric DFT does?",
        "figure_of_merit": "Spearman rank correlation -- tier 3 is asked for an "
        "ORDER, and absolute xtb/DFT energies are not comparable",
        "same_geometry": "DFT is evaluated at the xtb-RELAXED geometry, not the "
        "MMFF one, so the two tiers score the same structures",
        "settings": {k: str(v) for k, v in vars(args).items()},
    }

    # --- conformers -------------------------------------------------------
    mol = Chem.AddHs(Chem.MolFromSmiles(args.smiles))
    cids = AllChem.EmbedMultipleConfs(
        mol, numConfs=args.n_conformers, randomSeed=args.seed
    )
    if not cids:
        report["error"] = "ETKDG produced no conformers"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 1
    AllChem.MMFFOptimizeMoleculeConfs(mol)
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    report["n_atoms"] = len(syms)
    report["n_conformers"] = len(cids)

    # --- score with both tiers --------------------------------------------
    xtb_e: list[float] = []
    dft_e: list[float] = []
    errors: list[str] = []
    t0 = time.time()
    for cid in cids:
        coords = [
            tuple(float(v) for v in r) for r in mol.GetConformer(cid).GetPositions()
        ]

        # xtb RELAXES, so it reports the energy of a geometry it moved to. The
        # DFT point must be taken at that SAME relaxed geometry, or the two
        # tiers are scoring different structures and the correlation measures
        # geometry drift rather than method agreement. This is the same
        # same-instrument discipline M12 got wrong in the other direction.
        rx = relax(syms, coords, charge=0, skip_build_check=True)
        if rx.energy is None or rx.error is not None or rx.coords_angstrom is None:
            errors.append(f"xtb conf {cid}: {rx.error}")
            continue
        relaxed = rx.coords_angstrom

        xyz = f"{len(syms)}\nconf\n" + "".join(
            f"{s} {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}\n" for s, c in zip(syms, relaxed)
        )
        try:
            m = ferric.Molecule.from_xyz_string(xyz)
            res = ferric.run_rhf(m, ferric.BasisSet.bundled(args.basis))
        except Exception as exc:  # noqa: BLE001 -- one bad conformer must not abort
            errors.append(f"dft conf {cid}: {type(exc).__name__}: {exc}")
            continue

        xtb_e.append(rx.energy)
        dft_e.append(res.energy)

    report["wall_s"] = round(time.time() - t0, 1)
    report["n_scored_by_both"] = len(xtb_e)
    report["errors"] = errors[:5]

    if len(xtb_e) < 3:
        report["error"] = f"only {len(xtb_e)} conformers scored by both; need >= 3"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 1

    # Relative to each method's OWN minimum -- the only comparable quantity.
    xr = [(e - min(xtb_e)) * HARTREE_TO_KCAL for e in xtb_e]
    dr = [(e - min(dft_e)) * HARTREE_TO_KCAL for e in dft_e]
    report["relative_kcal"] = {
        "xtb": [round(v, 2) for v in xr],
        "dft": [round(v, 2) for v in dr],
        "xtb_span": round(max(xr), 2),
        "dft_span": round(max(dr), 2),
    }
    report["mae_relative_kcal"] = round(
        statistics.fmean(abs(a - b) for a, b in zip(xr, dr)), 3
    )

    try:
        from scipy.stats import spearmanr

        rho, p = spearmanr(xr, dr)
        report["spearman"] = {"rho": round(float(rho), 4), "p": round(float(p), 4)}
        usable = rho > 0.7 and p < 0.05
        report["verdict"] = {
            "usable_as_a_ranker": bool(usable),
            "conclusion": (
                "xtb promotes what DFT would promote -- tier 3 ranks"
                if usable
                else "xtb does NOT reliably reproduce the DFT order on this set; "
                "treat tier 3 as a coarse gate, not a ranker"
            ),
        }
    except ImportError:
        report["spearman"] = "scipy absent"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
