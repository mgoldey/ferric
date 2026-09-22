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

__all__ = ["parse_smiles_idx_remark", "restore_hydrogens"]


class StereochemistryError(ValueError):
    """A restored pose whose stereocentres are not the molecule ones."""


def parse_smiles_idx_remark(pdbqt_text: str) -> dict[int, int]:
    """Meeko's `REMARK SMILES IDX` mapping: PDBQT serial -> RDKit index (0-based).

    Returns `{}` when the remark is absent, which a caller must treat as
    "mapping unknown" rather than "identity".

    **The indices are into MEEKO'S OWN `REMARK SMILES`, not into yours.** Meeko
    rewrites the SMILES (canonically identical, atom order different), so a map
    built here is only meaningful against the molecule built from that string.
    MEASURED on danuglipron: our SMILES and meeko's are the same molecule by
    canonical form but their element sequences differ, and applying the map to
    OUR topology still yields the (R) enantiomer of an (S) drug. Pair this with
    `smiles_from_pdbqt_remark` and build from that.

    **Meeko REORDERS atoms.** MEASURED on aspirin: 10 of 13 heavy atoms come
    back at a different position than RDKit gave them, and assigning
    coordinates by list order misplaces an atom by up to **4.9 A**. That is a
    scrambled molecule scored as if it were the pose -- same atom COUNT, same
    elements, no error anywhere.

    Meeko writes the remark as pairs across one or more lines:

        REMARK SMILES IDX 5 1 6 2 7 3 8 4 9 5 10 6 4 7 2 8 3 9 1 10 ...

    read as **(rdkit_index_1_based, pdbqt_serial)** -- the SMILES index comes
    FIRST, which is the opposite of what the name suggests.

    MEASURED on danuglipron (41 heavy atoms, 42 PDBQT atoms incl. one polar H):
    reading the pairs as (serial, index) makes the ELEMENT agree on only
    **21 of 41** atoms; reading them as (index, serial) agrees on **41 of 41**.
    The tell is in the data itself -- the second number reaches 42 (a serial)
    while the first stops at 41 (an index into a 41-atom molecule).

    Getting this backwards does not fail loudly. It produces a mapping that
    `heavy_atom_mapping` then rejects as incomplete, returning `None`, so
    `restore_hydrogens` silently falls back to POSITIONAL order -- the exact
    failure this mapping exists to prevent. On danuglipron that returned the
    (R) enantiomer of an (S) drug.
    """
    mapping: dict[int, int] = {}
    for line in pdbqt_text.splitlines():
        if not line.startswith("REMARK SMILES IDX"):
            continue
        nums = [int(x) for x in line.split()[3:]]
        if len(nums) % 2:
            raise ValueError(
                f"`REMARK SMILES IDX` line has an odd token count ({len(nums)}), "
                "so it is not (rdkit_index, pdbqt_serial) pairs: " + line.strip()
            )
        for i in range(0, len(nums), 2):
            rdkit_index_1based, serial = nums[i], nums[i + 1]
            mapping[serial] = rdkit_index_1based - 1
    return mapping


def restore_hydrogens(
    smiles: str,
    heavy_symbols: list[str],
    heavy_coords: list[tuple[float, float, float]],
    *,
    seed: int = 0xF00D,
    rdkit_index_of_heavy: list[int] | None = None,
    check_stereo: bool = True,
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

    # WHICH RDKit atom does each docked coordinate belong to?
    #
    # By default, list order -- which is correct only when the pose came back
    # in the order RDKit built the molecule. IT USUALLY HAS NOT: Meeko reorders
    # atoms for its torsion tree, MEASURED at 10 of 13 heavy atoms on aspirin,
    # and a positional assignment then misplaces an atom by up to 4.9 A while
    # every count and element still matches.
    #
    # `rdkit_index_of_heavy[k]` gives the RDKit index for the k-th docked
    # heavy atom; build it from `parse_smiles_idx_remark`. A caller that cannot
    # supply it is relying on the order being unchanged, which is a claim about
    # its own pipeline rather than about this function.
    if rdkit_index_of_heavy is not None:
        if len(rdkit_index_of_heavy) != len(docked_heavy):
            raise ValueError(
                f"rdkit_index_of_heavy has {len(rdkit_index_of_heavy)} entries "
                f"for {len(docked_heavy)} docked heavy atoms"
            )
        if sorted(rdkit_index_of_heavy) != sorted(heavy_idx):
            raise ValueError(
                "rdkit_index_of_heavy is not a permutation of this molecule's "
                "heavy-atom indices; the mapping and the SMILES disagree about "
                "which molecule this is"
            )
        targets = list(rdkit_index_of_heavy)
    else:
        targets = list(heavy_idx)

    AllChem.EmbedMolecule(mol, randomSeed=seed)
    conf = mol.GetConformer()
    for idx, (_, c) in zip(targets, docked_heavy):
        conf.SetAtomPosition(idx, Point3D(*[float(v) for v in c]))
    # Optimise ONLY the hydrogens: the heavy atoms are the docking RESULT and
    # must not move, or the pose being scored is no longer the pose docked.
    props = AllChem.MMFFGetMoleculeProperties(mol)
    ff = AllChem.MMFFGetMoleculeForceField(mol, props) if props is not None else None
    if ff is not None:
        for idx in heavy_idx:
            ff.AddFixedPoint(idx)
        ff.Minimize(maxIts=500)

    # THE CHECK THAT FIRES WHEN A CALLER OMITS THE MAPPING.
    #
    # Every validation above passes on a positionally-assigned pose: the counts
    # match, the elements match, the bond graph is the SMILES one. The molecule
    # is still wrong. MEASURED on danuglipron -- 41 heavy atoms, only 12 of them
    # in meeko output order -- the declared (S) stereocentre came back **(R)**,
    # the mirror image of the drug, with nothing raised.
    #
    # Stereochemistry is the cheapest observable that separates a correctly
    # placed pose from a scrambled one, so it is the guard.
    if check_stereo:
        # Compare ONLY the centres the SMILES declares.
        #
        # `FindMolChiralCenters` omits UNDEFINED centres by default, but
        # `AssignStereochemistryFrom3D` assigns them from the geometry -- so
        # the second call returns extra keys and a whole-dict comparison
        # rejects a pose whose declared centres are perfectly correct.
        # MEASURED on C[C@H](N)C(C)(O)CC(C)F: declared [(1,'S')], after 3D
        # [(1,'S'),(3,'S'),(7,'S')]. An undefined centre has no correct value
        # to check against, so it is not evidence of anything.
        declared = dict(
            Chem.FindMolChiralCenters(parsed, useLegacyImplementation=False)
        )
        if declared:
            probe = Chem.Mol(mol)
            Chem.AssignStereochemistryFrom3D(probe)
            got_all = dict(
                Chem.FindMolChiralCenters(probe, useLegacyImplementation=False)
            )
            got = {i: v for i, v in got_all.items() if i in declared}
            if got != declared:
                raise StereochemistryError(
                    f"the restored pose has stereocentres {got} but the SMILES "
                    f"declares {declared} -- this is a different isomer, not the "
                    "molecule that was docked. The usual cause is an atom ORDER "
                    "mismatch: pass rdkit_index_of_heavy, built from "
                    "parse_smiles_idx_remark()."
                )

    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    pos = mol.GetConformer().GetPositions()
    return syms, [tuple(float(v) for v in r) for r in pos]


def smiles_from_pdbqt_remark(pdbqt_text: str) -> str | None:
    """Meeko's own `REMARK SMILES` string, or None when absent.

    `parse_smiles_idx_remark`'s indices refer to THIS string's atom order, so
    the two are only correct together. Building the topology from a different
    (even canonically identical) SMILES and applying the map scrambles the
    molecule -- measured as an (R)/(S) inversion on danuglipron.
    """
    for line in pdbqt_text.splitlines():
        if line.startswith("REMARK SMILES") and not line.startswith(
            "REMARK SMILES IDX"
        ):
            parts = line.split(None, 2)
            if len(parts) == 3:
                return parts[2].strip()
    return None
