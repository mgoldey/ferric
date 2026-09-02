"""Substituent scans: replace a matched site with each of a set of groups.

A SUBSTITUTIONAL isomer decorates a fixed scaffold. `structural.py` handles the
case where the scaffold itself changes.

Verified 2026-08-30 that RDKit's `RunReactants` plus canonical-SMILES dedup
collapses symmetry-equivalent products correctly: a fluorine scan of benzoic
acid gives exactly 3 products (ortho/meta/para) from 5 aryl hydrogens. Without
the dedup it would give 5, silently triple-counting the two mirror pairs and
inflating every downstream population count.
"""
from __future__ import annotations

from .model import Isomer

# Label -> replacement fragment. Chosen to span the medicinal-chemistry moves
# that change electronics and sterics without touching the scaffold: a halogen
# scan, a methyl, a strong EWG, an ether, and a lipophilic EWG.
COMMON_SUBSTITUENTS: dict[str, str] = {
    "F": "F",
    "Cl": "Cl",
    "Me": "C",
    "CN": "C#N",
    "OMe": "OC",
    "CF3": "C(F)(F)F",
}


def substituent_scan(
    parent_smiles: str,
    substituents: dict[str, str],
    site_smarts: str = "[cH:1]",
) -> list[Isomer]:
    """Every distinct single substitution of `site_smarts` by each substituent.

    Deduplicated on canonical SMILES and returned in a deterministic order
    (transform label, then canonical SMILES), so two runs agree exactly --
    `RunReactants` does not guarantee a stable product order on its own.

    A product that fails sanitization is SKIPPED rather than raising: a
    substituent that makes chemical nonsense at one site is normal, and must not
    abort the whole scan.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    parent = Chem.MolFromSmiles(parent_smiles)
    if parent is None:
        raise ValueError(f"unparseable parent SMILES: {parent_smiles}")

    out: list[Isomer] = []
    for label in sorted(substituents):
        frag = substituents[label]
        rxn = AllChem.ReactionFromSmarts(f"{site_smarts}>>[c:1]{frag}")
        seen: set[str] = set()
        for products in rxn.RunReactants((parent,)):
            p = products[0]
            try:
                Chem.SanitizeMol(p)
            except Exception:  # noqa: BLE001 - a bad product is skipped, not fatal
                continue
            canon = Chem.MolToSmiles(p)
            if canon in seen:
                continue
            seen.add(canon)
            out.append(Isomer(
                smiles=canon, kind="substitutional",
                transform=f"{site_smarts} -> {label}",
                parent_smiles=parent_smiles,
            ))
    out.sort(key=lambda i: (i.transform, i.canonical))
    return out
