"""Structural isomers: stereochemistry, ring size, and bioisosteric swaps.

"Structural" here means the SCAFFOLD changes -- connectivity, ring size, or a
whole functional group -- as opposed to `substitutional.py`, which decorates a
fixed scaffold. Both produce `Isomer` records and are deduplicated the same way.
"""
from __future__ import annotations

from .model import Isomer

# label -> (SMARTS to find, SMILES to put in its place)
ACID_BIOISOSTERES: dict[str, tuple[str, str]] = {
    "tetrazole": ("[CX3](=O)[OX2H1]", "c1nn[nH]n1"),
    "acylsulfonamide": ("[CX3](=O)[OX2H1]", "C(=O)NS(=O)(=O)C"),
    "hydroxamic_acid": ("[CX3](=O)[OX2H1]", "C(=O)NO"),
}

RING_CONTRACTIONS: dict[str, tuple[str, str]] = {
    "piperidine_to_azetidine": ("C1CCNCC1", "C1CNC1"),
    "piperidine_to_pyrrolidine": ("C1CCNCC1", "C1CCNC1"),
    "cyclohexyl_to_cyclobutyl": ("C1CCCCC1", "C1CCC1"),
}


def _replace(parent_smiles: str, kind: str,
             table: dict[str, tuple[str, str]]) -> list[Isomer]:
    """Apply each (find, replace) pair once, deduplicating on canonical form."""
    from rdkit import Chem

    parent = Chem.MolFromSmiles(parent_smiles)
    if parent is None:
        raise ValueError(f"unparseable parent SMILES: {parent_smiles}")

    out: list[Isomer] = []
    seen: set[str] = set()
    for label in sorted(table):
        frm, to = table[label]
        patt, repl = Chem.MolFromSmarts(frm), Chem.MolFromSmiles(to)
        if patt is None or repl is None:
            # A malformed transform is a PROGRAMMING error, unlike a product
            # that fails to sanitize -- raise rather than skip silently.
            raise ValueError(f"bad transform {label!r}: {frm!r} -> {to!r}")
        if not parent.HasSubstructMatch(patt):
            continue
        for prod in Chem.ReplaceSubstructs(parent, patt, repl, replaceAll=False):
            try:
                Chem.SanitizeMol(prod)
            except Exception:  # noqa: BLE001
                continue
            canon = Chem.MolToSmiles(prod)
            if canon in seen or canon == Chem.MolToSmiles(parent):
                continue
            seen.add(canon)
            out.append(Isomer(smiles=canon, kind=kind, transform=label,
                              parent_smiles=parent_smiles))
    out.sort(key=lambda i: (i.transform, i.canonical))
    return out


def bioisostere_swaps(parent_smiles: str,
                      swaps: "dict[str, tuple[str, str]] | None" = None) -> list[Isomer]:
    """Replace a functional group with bioisosteres that keep its role.

    Returns [] when the group is absent -- a no-op, not an error: running a
    swap table against a molecule that lacks the group is normal in a batch.
    """
    return _replace(parent_smiles, "structural", swaps or ACID_BIOISOSTERES)


def ring_contractions(parent_smiles: str) -> list[Isomer]:
    """Shrink a saturated ring, changing scaffold geometry without deleting it."""
    return _replace(parent_smiles, "structural", RING_CONTRACTIONS)


def stereoisomers(parent_smiles: str, max_isomers: int = 32) -> list[Isomer]:
    """Enumerate UNASSIGNED stereocentres.

    `max_isomers` is a hard cap: n centres give 2^n isomers, a combinatorial
    trap on a flexible drug-like molecule. The cap is applied by RDKit itself so
    the truncation is deterministic, rather than by slicing an
    arbitrarily-ordered list afterwards.
    """
    from rdkit import Chem
    from rdkit.Chem.EnumerateStereoisomers import (
        EnumerateStereoisomers,
        StereoEnumerationOptions,
    )

    parent = Chem.MolFromSmiles(parent_smiles)
    if parent is None:
        raise ValueError(f"unparseable parent SMILES: {parent_smiles}")
    opts = StereoEnumerationOptions(onlyUnassigned=True, maxIsomers=max_isomers,
                                    unique=True)
    out = [
        Isomer(smiles=Chem.MolToSmiles(m), kind="structural",
               transform="stereoisomer", parent_smiles=parent_smiles)
        for m in EnumerateStereoisomers(parent, opts)
    ]
    out.sort(key=lambda i: i.canonical)
    return out
