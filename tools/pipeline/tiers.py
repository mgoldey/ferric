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
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from tools.docking import dock_ligand

    mol = Chem.MolFromSmiles(iso.canonical)
    if mol is None:
        return TierResult(iso.canonical, None, "unparseable SMILES")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = context.get("seed", 0xF00D)
    params.useSmallRingTorsions = True
    try:
        if AllChem.EmbedMolecule(mol, params) != 0:
            return TierResult(iso.canonical, None, "ETKDG could not embed for docking")
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception as e:  # noqa: BLE001
        return TierResult(iso.canonical, None, f"prep failed: {type(e).__name__}: {e}")

    res = dock_ligand(mol, context["receptor_pdbqt"], context["box_center"],
                      context.get("box_size", (24.0, 24.0, 24.0)),
                      exhaustiveness=context.get("exhaustiveness", 16),
                      n_poses=context.get("n_poses", 10),
                      seed=context.get("seed", 0xF00D))
    if not res.ok:
        return TierResult(iso.canonical, None, res.error)
    best = res.best
    return TierResult(iso.canonical, best.vina_score,
                      payload={"symbols": best.symbols,
                               "coords": best.coords_angstrom,
                               "n_poses": len(res.poses)})


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

    Measured 96.1 s at 32 atoms (def2-SVP/PBE), extrapolating to 17-37 min at
    70 atoms, so this must only ever see the handful the tiers above left.
    """
    import ferric

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
