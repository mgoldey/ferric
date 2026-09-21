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

__all__ = ["restore_hydrogens", "pose_to_rdkit_order", "StereochemistryError"]


class StereochemistryError(ValueError):
    """A restored pose whose stereocentres are not the molecule's."""


def restore_hydrogens(
    smiles: str,
    heavy_symbols: Sequence[str],
    heavy_coords: Coords,
    *,
    rdkit_index_of_heavy: Sequence[int] | None = None,
    check_stereo: bool = True,
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

    **`rdkit_index_of_heavy` is not optional in practice for a MEEKO pose.**
    Without it the docked heavy atoms are assigned to the SMILES heavy atoms
    IN ORDER, and meeko reorders atoms when it writes PDBQT: MEASURED on
    danuglipron, only **12 of 41** heavy positions matched in order, and the
    first source carbon landed on a nitrogen. The molecule that comes back is
    a scrambled isomer -- on danuglipron the declared (S) stereocentre came
    back **(R)**, i.e. the mirror image of the drug, with no error raised.

    Meeko writes the mapping itself, as `REMARK SMILES IDX` lines in the PDBQT
    (serial <-> index into meeko's own `REMARK SMILES`). Parse those and pass
    them here. `pose_to_rdkit_order` does exactly that.
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

    if rdkit_index_of_heavy is not None:
        if len(rdkit_index_of_heavy) != len(docked_heavy):
            raise ValueError(
                f"rdkit_index_of_heavy has {len(rdkit_index_of_heavy)} entries "
                f"for {len(docked_heavy)} docked heavy atoms"
            )
        if sorted(rdkit_index_of_heavy) != sorted(heavy_idx):
            raise ValueError(
                "rdkit_index_of_heavy is not a permutation of this molecule's "
                "heavy-atom indices -- it does not describe this topology"
            )
        heavy_idx = list(rdkit_index_of_heavy)

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

    if check_stereo:
        declared = Chem.FindMolChiralCenters(
            Chem.MolFromSmiles(smiles), useLegacyImplementation=False
        )
        if declared:
            probe = Chem.Mol(mol)
            Chem.AssignStereochemistryFrom3D(probe)
            got = Chem.FindMolChiralCenters(probe, useLegacyImplementation=False)
            if dict(got) != dict(declared):
                raise StereochemistryError(
                    f"the restored pose has stereocentres {got} but {smiles!r} "
                    f"declares {declared} -- this is a different isomer, not "
                    "the molecule that was docked. The usual cause is an atom "
                    "ORDER mismatch: pass rdkit_index_of_heavy (see "
                    "pose_to_rdkit_order)."
                )

    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    pos = mol.GetConformer().GetPositions()
    return syms, [tuple(float(v) for v in row) for row in pos]


def pose_to_rdkit_order(pdbqt_text: str) -> tuple[str, list[int]]:
    """Meeko's own atom mapping, read out of the PDBQT it wrote.

    Meeko emits `REMARK SMILES` (its canonical SMILES for the ligand) and
    `REMARK SMILES IDX <smiles_index> <pdbqt_serial> ...` pairs. Those are the
    ONLY authoritative statement of which docked atom is which -- meeko
    reorders atoms freely, and assuming its output order matches the input
    silently produces a scrambled isomer.

    Returns `(meeko_smiles, rdkit_index_of_heavy)` where the list is ordered by
    PDBQT serial (i.e. by the order coordinates appear in the pose) and holds
    0-based indices into the molecule built from `meeko_smiles`.

    Raises `ValueError` when the remarks are absent -- a caller that silently
    fell back to positional order is the bug this exists to prevent.
    """
    smiles = None
    tokens: list[str] = []
    for line in pdbqt_text.splitlines():
        if line.startswith("REMARK SMILES IDX"):
            tokens += line[len("REMARK SMILES IDX") :].split()
        elif line.startswith("REMARK SMILES") and smiles is None:
            parts = line.split(None, 2)
            if len(parts) == 3:
                smiles = parts[2].strip()

    if smiles is None or not tokens:
        raise ValueError(
            "this PDBQT carries no `REMARK SMILES`/`REMARK SMILES IDX` lines, "
            "so meeko's atom order cannot be recovered and any coordinate "
            "assignment would be positional -- which scrambles the molecule"
        )
    if len(tokens) % 2:
        raise ValueError(
            f"`REMARK SMILES IDX` has {len(tokens)} tokens, which is not an "
            "even number of (smiles_index, pdbqt_serial) pairs"
        )

    # (smiles_index, pdbqt_serial), both 1-based as meeko writes them.
    pairs = [(int(tokens[i]), int(tokens[i + 1])) for i in range(0, len(tokens), 2)]
    pairs.sort(key=lambda p: p[1])  # order by the serial the coords come in
    return smiles, [smi - 1 for smi, _ in pairs]
