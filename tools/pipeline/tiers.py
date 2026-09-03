"""Thin adapters presenting each existing tier with one uniform signature.

Deliberately thin: the chemistry lives in `tools/docking`, `tools/morph` and
`tools/campaign` and is NOT reimplemented here. This module exists so the funnel
can call four very different methods -- an empirical docking score, a force
field, a semiempirical Hamiltonian and a DFT SCF -- without knowing anything
about any of them.

Measured costs on this box (70-atom anion; def2-SVP/PBE for tier 4):

    tier 1  Vina           ~2 min/ligand at exhaustiveness 32
    tier 2  MMFF94         ~1 ms/pose
    tier 3  GFN2-xTB       ~0.5 s single point
    tier 4  ferric DFT     96.1 s at 32 atoms -> 17-37 min at 70 (N^3-N^4)

Tier 4's cost is the reason the funnel must narrow to a handful before reaching
it. See `tools/campaign/hierarchy.py` for the rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tools.isomers.model import Isomer


@dataclass
class TierResult:
    """One tier's verdict on one candidate.

    `value` is `None` for ANY failure -- never 0.0, which in an energy ranking
    reads as the best possible score and would promote a broken candidate to
    the top of the funnel.
    """
    candidate_id: str
    value: float | None
    error: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.value is not None


def tier2_forcefield(iso: Isomer, context: dict) -> TierResult:
    """MMFF94 embed + optimize. Cheap declash; NOT a ranking method.

    Measured: GFN2 moves an MMFF geometry by 12-14 kcal/mol and 0.13-0.39 A, so
    MMFF energies are adequate to reject a clashing structure and inadequate to
    order two reasonable ones.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(iso.canonical)
    if mol is None:
        return TierResult(iso.canonical, None, "unparseable SMILES")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = context.get("seed", 0xF00D)
    params.useSmallRingTorsions = True
    try:
        if AllChem.EmbedMolecule(mol, params) != 0:
            return TierResult(iso.canonical, None, "ETKDG could not embed")
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
        energy = float(res[0][1])
    except Exception as e:  # noqa: BLE001 - RDKit raises RuntimeError on cages
        return TierResult(iso.canonical, None,
                          f"MMFF failed: {type(e).__name__}: {e}")
    conf = mol.GetConformer()
    coords = [tuple(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    return TierResult(iso.canonical, energy,
                      payload={"coords": coords,
                               "symbols": [a.GetSymbol() for a in mol.GetAtoms()]})


def _embedded(iso: Isomer, context: dict):
    """Shared 3D geometry for tiers 3 and 4: reuse a cached one if present.

    Without the cache each tier would re-embed, and tiers 3 and 4 would then be
    scoring DIFFERENT geometries of the same candidate -- which makes their
    energies incomparable for no benefit.
    """
    cached = context.get("geometry", {}).get(iso.canonical)
    if cached:
        return cached["symbols"], cached["coords"]
    r = tier2_forcefield(iso, context)
    if not r.ok:
        return None, None
    return r.payload["symbols"], r.payload["coords"]


def tier1_dock(iso: Isomer, context: dict) -> TierResult:
    """AutoDock Vina pose search. `value` is the best Vina score (lower better).

    The score is an empirical ranking heuristic, NOT a binding free energy, and
    is used here only to order poses for the tiers above. Validated on this
    target by redocking 7LCJ to 0.95 A.

    **Spend the budget on SEEDS, not on exhaustiveness.** Measured on this
    target (RESULTS.md M11): across an 8x range of `exhaustiveness` the mean
    redock RMSD moved 0.097 A, which is SMALLER than the 0.131 A between-seed
    SEM -- and no seed improved monotonically with effort. What did move the
    number was the starting conformer. So `n_seeds` docks the ligand from
    several independent ETKDG embeddings and keeps the best-scoring pose, which
    buys real spread coverage where extra search effort bought noise.

    `n_seeds=1` reproduces the old single-embedding behaviour exactly.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from tools.docking import dock_ligand

    mol0 = Chem.MolFromSmiles(iso.canonical)
    if mol0 is None:
        return TierResult(iso.canonical, None, "unparseable SMILES")
    # Meeko requires a single connected molecule. A transform can split one --
    # e.g. a ring contraction that severs the ring rather than shrinking it --
    # and the resulting salt/fragment pair is not a dockable ligand. Rejected
    # here with a readable reason rather than 300 lines of Meeko traceback.
    if len(Chem.GetMolFrags(mol0)) > 1:
        return TierResult(iso.canonical, None,
                          f"not a single connected molecule "
                          f"({len(Chem.GetMolFrags(mol0))} fragments)")

    base_seed = context.get("seed", 0xF00D)
    n_seeds = max(1, int(context.get("n_seeds", 1)))
    best_overall = None
    prep_errors: list[str] = []

    for k in range(n_seeds):
        seed = base_seed + k
        mol = Chem.AddHs(Chem.Mol(mol0))
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.useSmallRingTorsions = True
        try:
            if AllChem.EmbedMolecule(mol, params) != 0:
                prep_errors.append(f"seed {seed}: ETKDG could not embed")
                continue
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception as e:  # noqa: BLE001
            prep_errors.append(f"seed {seed}: {type(e).__name__}: {e}")
            continue

        try:
            res = dock_ligand(mol, context["receptor_pdbqt"],
                              context["box_center"],
                              context.get("box_size", (24.0, 24.0, 24.0)),
                              exhaustiveness=context.get("exhaustiveness", 16),
                              n_poses=context.get("n_poses", 10),
                              seed=seed,
                              # Default 1, NOT Vina's 0: this tier runs inside a
                              # funnel that fans out across ligands, and two
                              # levels of parallelism oversubscribe the box.
                              # See dock_ligand's `cpu` docs.
                              cpu=context.get("vina_cpu", 1))
        except ImportError as e:
            # vina/meeko are an optional extra (`pip install ferric[docking]`),
            # because they are not installable on every Python the wheel
            # targets. A tier that cannot run must say WHY it cannot run --
            # reporting this as a docking failure would send the reader hunting
            # for a receptor or a bad ligand when the real answer is an
            # uninstalled package.
            return TierResult(iso.canonical, None,
                              f"docking unavailable ({e}); "
                              f"install the 'docking' extra to enable tier 1")
        if not res.ok:
            prep_errors.append(f"seed {seed}: {res.error}")
            continue
        cand = res.best
        if best_overall is None or cand.vina_score < best_overall[0].vina_score:
            best_overall = (cand, len(res.poses), seed)

    if best_overall is None:
        return TierResult(iso.canonical, None,
                          "; ".join(prep_errors) or "docking produced no pose")
    best, n_poses, winning_seed = best_overall
    return TierResult(iso.canonical, best.vina_score,
                      payload={"symbols": best.symbols,
                               "coords": best.coords_angstrom,
                               "n_poses": n_poses,
                               "n_seeds": n_seeds,
                               "winning_seed": winning_seed})


def tier3_gfn2(iso: Isomer, context: dict) -> TierResult:
    """GFN2-xTB single point, optionally in a pocket point-charge field."""
    from tools.campaign.xtb_engine import singlepoint

    symbols, coords = _embedded(iso, context)
    if symbols is None:
        return TierResult(iso.canonical, None, "no geometry for GFN2")
    run = singlepoint(symbols, coords, charge=iso.net_charge,
                      point_charges=context.get("point_charges"))
    if not run.ok:
        return TierResult(iso.canonical, None, run.error)
    return TierResult(iso.canonical, run.energy,
                      payload={"symbols": symbols, "coords": coords})


def tier4_dft(iso: Isomer, context: dict) -> TierResult:
    """ferric Kohn-Sham DFT -- the most expensive tier.

    Measured 96.1 s at 32 atoms (def2-SVP/PBE; re-measured 99.0 s on
    2026-09-02), so this must only ever see the handful the tiers above left.

    Cost here is driven by ATOM COUNT, not basis size: ferric's KS grid is
    75x110 per atom, so a smaller basis does NOT make a big molecule cheap.
    See `tools/pipeline/cost.py`.

    `mem_budget_gb` is forwarded to `FERRIC_MEM_BUDGET_GB` because ferric's
    default is `0.8 x` *live* MemAvailable. That makes the internal
    Full-vs-Batched AO-cache decision depend on whatever else happens to be
    running on the box -- the same candidate can take the fast path or the
    batching path between two runs of the SAME pipeline. Pinning it keeps the
    tier reproducible; leaving it None preserves ferric's auto-detect.
    """
    import os

    import ferric

    budget = context.get("mem_budget_gb")
    prior = os.environ.get("FERRIC_MEM_BUDGET_GB")
    if budget is not None:
        os.environ["FERRIC_MEM_BUDGET_GB"] = str(budget)
    try:
        return _tier4_dft_inner(iso, context, ferric)
    finally:
        if budget is not None:
            if prior is None:
                os.environ.pop("FERRIC_MEM_BUDGET_GB", None)
            else:
                os.environ["FERRIC_MEM_BUDGET_GB"] = prior


def _tier4_dft_inner(iso: Isomer, context: dict, ferric) -> TierResult:
    symbols, coords = _embedded(iso, context)
    if symbols is None:
        return TierResult(iso.canonical, None, "no geometry for DFT")
    xyz = [str(len(symbols)), "tier4"]
    for s, (x, y, z) in zip(symbols, coords):
        xyz.append(f"{s} {x:.8f} {y:.8f} {z:.8f}")
    try:
        mol = ferric.Molecule.from_xyz_string("\n".join(xyz) + "\n", iso.net_charge, 1)
        bs = ferric.BasisSet.bundled(context.get("basis", "def2-svp"))
        res = ferric.run_dft(mol, bs, functional=context.get("functional", "PBE"),
                             point_charges=context.get("point_charges"))
    except Exception as e:  # noqa: BLE001
        return TierResult(iso.canonical, None,
                          f"DFT failed: {type(e).__name__}: {e}")
    if not res.converged:
        return TierResult(iso.canonical, None, "DFT did not converge")
    return TierResult(iso.canonical, res.total_energy,
                      payload={"converged": True, "symbols": symbols, "coords": coords})
