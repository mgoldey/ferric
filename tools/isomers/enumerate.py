"""Run every generator, deduplicate globally, filter, and report what was lost.

The report is not decoration: an enumeration that silently drops candidates is
indistinguishable from one that never made them, and the point of this package
is that a candidate list can be regenerated and audited rather than trusted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .model import Isomer
from .structural import bioisostere_swaps, ring_contractions, stereoisomers
from .substitutional import COMMON_SUBSTITUENTS, substituent_scan


@dataclass
class EnumerationReport:
    """Population counts at each narrowing step, plus why things were dropped.

    `n_generated >= n_after_dedup >= n_after_filter` always holds; the tests
    assert it, because a report whose counts do not nest is describing
    something other than what happened.
    """
    n_generated: int = 0
    n_after_dedup: int = 0
    n_after_filter: int = 0
    rejected: list[str] = field(default_factory=list)


def enumerate_with_report(
    parent_smiles: str,
    *,
    substituents: "dict[str, str] | None" = None,
    site_smarts: str = "[cH:1]",
    include_stereo: bool = True,
    include_rings: bool = True,
    include_bioisosteres: bool = True,
    max_candidates: int = 200,
    mw_range: tuple[float, float] = (0.0, 700.0),
) -> "tuple[list[Isomer], EnumerationReport]":
    """Enumerate, dedupe across generators, filter, and report the losses.

    The parent is always emitted first and exactly once, so a downstream ranking
    always has its reference in the set.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    rep = EnumerationReport()
    generated: list[Isomer] = [Isomer(parent_smiles, "parent", "none", parent_smiles)]
    generated += substituent_scan(parent_smiles,
                                  substituents or COMMON_SUBSTITUENTS,
                                  site_smarts)
    if include_bioisosteres:
        generated += bioisostere_swaps(parent_smiles)
    if include_rings:
        generated += ring_contractions(parent_smiles)
    if include_stereo:
        generated += stereoisomers(parent_smiles)
    rep.n_generated = len(generated)

    # Global dedup across generators: a stereoisomer of the parent and the
    # parent itself can collide, as can two routes to the same product.
    # Parent first, then a stable (kind, transform, canonical) order.
    seen: set[str] = set()
    deduped: list[Isomer] = []
    for iso in sorted(generated,
                      key=lambda i: (not i.is_parent, i.kind, i.transform, i.canonical)):
        if iso.canonical in seen:
            continue
        seen.add(iso.canonical)
        deduped.append(iso)
    rep.n_after_dedup = len(deduped)

    kept: list[Isomer] = []
    lo, hi = mw_range
    for iso in deduped:
        mol = Chem.MolFromSmiles(iso.canonical)
        # A transform can sever a ring instead of contracting it, leaving two
        # disconnected fragments. That is not a candidate molecule, and it
        # fails much later (in ligand prep) with an opaque message if kept.
        n_frags = len(Chem.GetMolFrags(mol))
        if n_frags > 1:
            rep.rejected.append(
                f"{iso.transform} ({iso.canonical}): {n_frags} disconnected fragments"
            )
            continue
        mw = Descriptors.MolWt(mol)
        if not (lo <= mw <= hi):
            rep.rejected.append(
                f"{iso.transform} ({iso.canonical}): MW {mw:.1f} outside [{lo}, {hi}]"
            )
            continue
        kept.append(iso)
        if len(kept) >= max_candidates:
            remaining = len(deduped) - deduped.index(iso) - 1
            if remaining > 0:
                rep.rejected.append(
                    f"max_candidates={max_candidates} reached; "
                    f"{remaining} candidates not evaluated"
                )
            break
    rep.n_after_filter = len(kept)
    return kept, rep


def enumerate_isomers(parent_smiles: str, **kwargs) -> list[Isomer]:
    """Candidates only. Use `enumerate_with_report` when the losses matter."""
    return enumerate_with_report(parent_smiles, **kwargs)[0]
