"""Put a united-atom docked pose back on a full-hydrogen topology.

A PDBQT pose is UNITED-ATOM: Vina merges nonpolar hydrogens into the carbon
they sit on, so `DockedPose.symbols` is **not the molecule that was docked**.
MEASURED on danuglipron/7LCJ: 71 atoms in, **42 out**. On aspirin: 21 in, 14
out.

That matters because the downstream consumers are UNEVENLY protected. A QM
tier with a charge/multiplicity parity check may reject an odd electron count
by accident, but a neighbouring scorer has no such check and will happily
return a plausible number for a molecule nobody asked about -- measured
2591 kcal/mol wrong on aspirin at GFN2, with no error raised.

So: anything that takes a docked pose into an energy calculation must restore
the hydrogens first, and must FAIL rather than truncate when the heavy-atom
counts disagree.

Promoted from `experiments/danuglipron/run_scorer_pose_sensitivity.py`, where
this logic worked but was private to one script and therefore unavailable to
the pipeline tiers that need it.
"""

from __future__ import annotations

from collections.abc import Sequence

Coords = Sequence[tuple[float, float, float]]

__all__ = ["restore_hydrogens"]


def restore_hydrogens(
    smiles: str,
    heavy_symbols: Sequence[str],
    heavy_coords: Coords,
    *,
    random_seed: int = 0xF00D,
    max_iterations: int = 500,
) -> tuple[list[str], list[tuple[float, float, float]]]:
    """Return (symbols, coords) for the full-hydrogen molecule.

    Builds the topology from `smiles` (which knows every hydrogen), assigns the
    docked HEAVY-ATOM coordinates onto its heavy atoms in order, then places
    the hydrogens with MMFF while holding every heavy atom FIXED -- the heavy
    atoms are the docking result, and moving them would mean scoring a pose
    that was never docked.

    Raises `ValueError` if the heavy-atom counts disagree. Silently truncating
    (or zipping to the shorter list) would score a different molecule, which is
    the exact failure this function exists to prevent.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D

    if len(heavy_symbols) != len(heavy_coords):
        raise ValueError(
            f"restore_hydrogens: {len(heavy_symbols)} symbols but "
            f"{len(heavy_coords)} coordinate rows -- these are per-atom and "
            "must match"
        )

    parsed = Chem.MolFromSmiles(smiles)
    if parsed is None:
        raise ValueError(f"restore_hydrogens: RDKit could not parse {smiles!r}")
    mol = Chem.AddHs(parsed)

    heavy_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    docked_heavy = [
        (s, c) for s, c in zip(heavy_symbols, heavy_coords) if s.upper() != "H"
    ]
    if len(heavy_idx) != len(docked_heavy):
        raise ValueError(
            f"{len(docked_heavy)} docked heavy atoms but the SMILES has "
            f"{len(heavy_idx)} -- these are different molecules, not a "
            "hydrogen-count difference"
        )

    if AllChem.EmbedMolecule(mol, randomSeed=random_seed) != 0:
        raise ValueError(
            "restore_hydrogens: RDKit could not embed the full-hydrogen "
            "topology, so there is no conformer to write the pose onto"
        )
    conf = mol.GetConformer()
    for idx, (_, c) in zip(heavy_idx, docked_heavy):
        conf.SetAtomPosition(idx, Point3D(*[float(v) for v in c]))

    props = AllChem.MMFFGetMoleculeProperties(mol)
    ff = AllChem.MMFFGetMoleculeForceField(mol, props) if props is not None else None
    if ff is not None:
        for idx in heavy_idx:
            ff.AddFixedPoint(idx)
        ff.Minimize(maxIts=max_iterations)

    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    pos = mol.GetConformer().GetPositions()
    return syms, [tuple(float(v) for v in row) for row in pos]
