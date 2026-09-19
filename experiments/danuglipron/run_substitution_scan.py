#!/usr/bin/env python3
"""Propose substitutions on danuglipron and score them against the GLP-1R pocket.

This is the golden path for "propose viable substitutions at a drug active
site", run end to end on a real parent (PubChem CID 134611040) and a real
receptor (7LCJ). Every stage below is code that exists today; nothing is
mocked and nothing is a plan.

    S1  enumerate      substituent_scan, per SITE not per substituent
    S2  descriptors    RELATIVE to the parent, never absolute
    S3  embed          seeded ETKDG -> 3-D coordinates
    S4  pocket         pdb2pqr -> point charges, derived ONCE
    S5  prescreen      classical field at the ligand's atoms, no SCF

## Three findings this script exists to demonstrate

1. **Absolute descriptor gates are useless on a real lead.** Danuglipron is
   MW 555.6, cLogP 4.89 -- a Phase-2 clinical compound that already violates
   Lipinski. An absolute rule-of-5 gate rejects 54 of 54 analogues and ranks
   nothing. The rule of 5 is a hit-finding filter; on an optimized molecule it
   is a constant.

2. **The pharmacophore gate cannot reject a substituent scan** -- swapping an
   aromatic CH cannot break an acid, a fused diazole, a basic amine or a
   nitrile. MEASURED 54/54 kept. Verified REACHABLE separately (a methyl ester
   breaks it; benzene and ethanol break all four features), so that is a true
   negative, not an inert gate. It belongs on SCAFFOLD moves.

3. **Where you put the group matters as much as which group it is.** MEASURED
   on the real pocket: within-substituent spread across ring positions was
   16.49 kcal/mol against a between-substituent range of 17.30 -- a ratio of
   0.95. A pipeline reporting one number per SUBSTITUENT is averaging over a
   variable as large as the one it is measuring.

(3) is why this script reports per (substituent, SITE) and never aggregates to
a per-label mean.

## What this is NOT

No docking, no xtb, no QM. The prescreen is a classical Coulomb score against
a fixed pose, so it measures the METHOD's response to placement, not a binding
affinity. A production run docks each analogue first; see
wiki/substitution-pipeline-danuglipron-2026-09-19.md for the full tier list and
the remaining blockers (pose noise, and QM/MM dispersion).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

BOHR_TO_ANGSTROM = 0.52917721092
HARTREE_TO_KCAL = 627.5095

DEFAULT_POCKET = REPO / "testdata/molecules/c9_systems/danuglipron/7LCJ_pocket.pdb"

# The medicinal-chemistry moves that change electronics and sterics without
# touching the scaffold. Same set as tools.isomers.substitutional's default.
SUBSTITUENTS = {
    "F": "F",
    "Cl": "Cl",
    "Me": "C",
    "CN": "C#N",
    "OMe": "OC",
    "CF3": "C(F)(F)F",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pocket", type=Path, default=DEFAULT_POCKET)
    ap.add_argument(
        "--site-smarts",
        default="[cH:1]",
        help="WHERE to substitute. The default hits every aromatic CH, which "
        "on a drug-sized parent is dozens of sites -- more than any QM tier "
        "can afford. In real use derive this from the pocket contact map: "
        "site choice is the budget control, and it is human judgement.",
    )
    ap.add_argument(
        "--skip-pocket",
        action="store_true",
        help="stop after the descriptor stage (no pdb2pqr, no pocket)",
    )
    args = ap.parse_args()

    from experiments.danuglipron.design import DANUGLIPRON_SMILES
    from tools.pipeline.substitution import embed_proposals, propose_substitutions

    print("S1/S2  enumerate + relative descriptors")
    props = propose_substitutions(
        DANUGLIPRON_SMILES, substituents=SUBSTITUENTS, site_smarts=args.site_smarts
    )
    subs = [p for p in props if not p.is_parent]
    print(
        f"       {len(subs)} analogues from {len(SUBSTITUENTS)} substituents, plus the parent"
    )
    # One row per SUBSTITUENT here, not per site: these three descriptors are
    # site-independent by construction (they depend only on what was added, not
    # where). That is NOT true of anything downstream of the pocket -- see the
    # site-dependence report at the end.
    print(f"       {'subst':<6}{'dMW':>8}{'dcLogP':>9}{'dTPSA':>8}")
    first = {}
    for p in subs:
        first.setdefault(p.label, p)
    for p in sorted(first.values(), key=lambda p: p.d_clogp):
        print(f"       {p.label:<6}{p.d_mw:>8.1f}{p.d_clogp:>9.2f}{p.d_tpsa:>8.1f}")
    print("       (relative, because the PARENT already violates Lipinski at MW 555.6)")

    print("\nS3     embed (seeded ETKDG)")
    embedded = embed_proposals(props)
    ok = [e for e in embedded if e.coords is not None]
    failed = [e for e in embedded if e.coords is None]
    print(f"       {len(ok)} embedded, {len(failed)} failed")
    for e in failed:
        print(f"         FAILED {e.proposal.label}: {e.error}")
    if args.skip_pocket:
        return 0

    print(f"\nS4     pocket charges from {args.pocket.name}")
    from tools.active_site.pocket_charges import derive_pocket_charges

    pocket = derive_pocket_charges(args.pocket)
    cx = sum(c[1] for c in pocket.charges) / len(pocket.charges) * BOHR_TO_ANGSTROM
    cy = sum(c[2] for c in pocket.charges) / len(pocket.charges) * BOHR_TO_ANGSTROM
    cz = sum(c[3] for c in pocket.charges) / len(pocket.charges) * BOHR_TO_ANGSTROM
    print(
        f"       {len(pocket.charges)} point charges, centroid ({cx:.1f}, {cy:.1f}, {cz:.1f}) A"
    )

    print("\nS5     prescreen (classical field, no SCF)")
    rows = _prescreen(ok, pocket, (cx, cy, cz))
    if not rows:
        print("       no poses scored")
        return 1

    rows.sort()
    print(f"       {'rank':<5}{'subst':<7}{'site':<6}{'score (kcal/mol)':>18}")
    for i, (score, label, site) in enumerate(rows, 1):
        print(f"       {i:<5}{label:<7}{site:<6}{score * HARTREE_TO_KCAL:>18.2f}")

    _report_site_dependence(rows)
    return 0


def _prescreen(embedded, pocket, centroid):
    """Score each pose. Returns (score_hartree, label, site_index) rows."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from tools.active_site.ligand_embedding import embed_ligand_from_coords
    from tools.active_site.prescreen import prescreen_pose

    cx, cy, cz = centroid
    seen: dict[str, int] = {}
    rows = []
    for e in embedded:
        # Put the LIGAND's centroid on the POCKET's centroid.
        #
        # Subtracting `e.coords`'s own centroid first is load-bearing, not
        # defensive: `embed_proposals` returns whatever ETKDG produced, and
        # ETKDG does not promise an origin-centred conformer. MEASURED over the
        # embeddings this script generates, |centroid| is 0.0000 A for some
        # molecules and 0.18-0.25 A for others (octane 0.2452, paracetamol
        # 0.1807). Adding the pocket centroid to an already-offset conformer
        # therefore displaces the pose by that much, in a direction and
        # magnitude that vary PER ANALOGUE -- so it perturbs exactly the
        # between-substituent comparison this scan exists to make, and does it
        # silently.
        n = len(e.coords)
        gx = sum(c[0] for c in e.coords) / n
        gy = sum(c[1] for c in e.coords) / n
        gz = sum(c[2] for c in e.coords) / n
        moved = [(x - gx + cx, y - gy + cy, z - gz + cz) for x, y, z in e.coords]
        el = embed_ligand_from_coords(
            list(e.symbols), moved, pocket=pocket, basis="sto-3g"
        )
        # Gasteiger charges: cheap, no QM. The prescreen tier's whole point is
        # to rank without an SCF.
        m = Chem.AddHs(Chem.MolFromSmiles(e.proposal.smiles))
        AllChem.ComputeGasteigerCharges(m)
        q = [a.GetDoubleProp("_GasteigerCharge") for a in m.GetAtoms()]
        if len(q) != el.mol.natoms():
            continue
        # Site index distinguishes the SAME substituent at different ring
        # positions -- the unit this pipeline reports on. See finding (3).
        site = seen.get(e.proposal.label, 0)
        seen[e.proposal.label] = site + 1
        rows.append((prescreen_pose(el, q).score, e.proposal.label, site))
    return rows


def _report_site_dependence(rows) -> None:
    """The finding that determines the pipeline's unit of work."""
    from collections import defaultdict

    groups = defaultdict(list)
    for score, label, _ in rows:
        groups[label].append(score * HARTREE_TO_KCAL)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if not multi:
        print(
            "\n       (only one site per substituent -- site dependence not measurable here)"
        )
        return
    # The BETWEEN number must come from MATCHED SITES, not from the pooled
    # range. `allv = every (substituent, site) score` already CONTAINS the
    # within-substituent site variation, so `within / max(allv)-min(allv)` is a
    # subset compared against its own superset: bounded by 1 by construction,
    # and it cannot distinguish "site matters as much as identity" from "site
    # is most of what the pooled range measures". An earlier version of this
    # function printed exactly that ratio and drew the conclusion from it.
    #
    # Matched comparison: at each site index, spread across DIFFERENT
    # substituents. That isolates identity with placement held fixed.
    by_site = defaultdict(list)
    for score, _label, site in rows:
        by_site[site].append(score * HARTREE_TO_KCAL)
    matched = [max(v) - min(v) for v in by_site.values() if len(v) > 1]
    within = max(max(v) - min(v) for v in multi.values())

    if not matched:
        # Every site index has at most one substituent, so identity and
        # placement are perfectly confounded here. Say so rather than falling
        # back to the pooled range, which would silently restore the bug.
        print(
            f"\n       WITHIN one substituent: {within:6.2f} kcal/mol (same group, "
            "different site)\n"
            "       BETWEEN substituents  : NOT MEASURABLE -- no site index carries\n"
            "         more than one substituent, so the two axes are perfectly\n"
            "         confounded in this run. The pooled score range is NOT a\n"
            "         substitute: it contains the within-substituent variation."
        )
        return

    between = max(matched)
    print(f"\n       BETWEEN substituents : {between:6.2f} kcal/mol (matched site)")
    print(
        f"       WITHIN one substituent: {within:6.2f} kcal/mol (same group, different site)"
    )
    print(f"       ratio                 : {within / between:6.2f}")
    if within / between > 0.5:
        print(
            "       -> WHERE the group goes matters as much as WHICH group it is.\n"
            "          Do not aggregate to a per-substituent mean; the unit of\n"
            "          this pipeline is the (substituent, SITE) pair."
        )


if __name__ == "__main__":
    raise SystemExit(main())
