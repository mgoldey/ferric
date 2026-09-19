"""Put a united-atom docked pose back on a full-hydrogen topology.

## Why this is its own module

PDBQT — what AutoDock Vina reads and writes — is **united-atom**: nonpolar
hydrogens are merged into the carbons they sit on. MEASURED on danuglipron
(RESULTS.md M14): a docked pose has **42 atoms and 263 electrons** where the
real molecule has **71 and 292**.

That difference is not cosmetic, and the two consumers in this repo disagree
about it in the worst possible way:

* `embed_ligand_from_coords` **refuses** — 263 electrons at multiplicity 1
  implies `n_alpha = 263/2`, which is not an integer. A loud, immediate failure.
* `pose_fit` **accepts it silently**. xtb will happily run on a molecule missing
  29 hydrogens and return a number that looks entirely normal.

So the same pose is rejected by one scorer and quietly scored by the other. A
copy of this fix living inside one probe script is how that asymmetry persists;
it belongs where every pose consumer can reach it.

## What it does NOT fix

Restoring hydrogens does not make a docked pose a relaxed structure. The heavy
atoms are pinned exactly where docking put them — that is the point, since
moving them would score a different pose than the one that was docked — so any
strain in the docked heavy-atom frame is still there.
"""

from __future__ import annotations

__all__ = ["restore_hydrogens"]


def restore_hydrogens(
    smiles: str,
    heavy_symbols: list[str],
    heavy_coords: list[tuple[float, float, float]],
    *,
    seed: int = 0xF00D,
) -> tuple[list[str], list[tuple[float, float, float]]]:
    """Return `(symbols, coords)` for the full-hydrogen molecule at this pose.

    Builds the topology from SMILES (which knows every hydrogen), assigns the
    docked HEAVY-ATOM coordinates onto its heavy atoms in order, then places the
    hydrogens with MMFF while holding every heavy atom fixed.

    Raises when the heavy-atom counts disagree: that means the SMILES and the
    pose are different molecules, and silently truncating would score the wrong
    one — the exact failure mode this module exists to prevent.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D

    parsed = Chem.MolFromSmiles(smiles)
    if parsed is None:
        raise ValueError(f"unparseable SMILES: {smiles!r}")
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

    AllChem.EmbedMolecule(mol, randomSeed=seed)
    conf = mol.GetConformer()
    for idx, (_, c) in zip(heavy_idx, docked_heavy):
        conf.SetAtomPosition(idx, Point3D(*[float(v) for v in c]))
    # Optimise ONLY the hydrogens: the heavy atoms are the docking RESULT and
    # must not move, or the pose being scored is no longer the pose docked.
    props = AllChem.MMFFGetMoleculeProperties(mol)
    ff = AllChem.MMFFGetMoleculeForceField(mol, props) if props is not None else None
    if ff is not None:
        for idx in heavy_idx:
            ff.AddFixedPoint(idx)
        ff.Minimize(maxIts=500)

    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    pos = mol.GetConformer().GetPositions()
    return syms, [tuple(float(v) for v in r) for r in pos]
