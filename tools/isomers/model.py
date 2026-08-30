"""One enumerated isomer, with the provenance that makes it reproducible."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property


@dataclass
class Isomer:
    """A generated candidate.

    `transform` records HOW this was produced (the SMARTS or the operation
    name), so a candidate list can be REGENERATED and audited rather than
    trusted. That is the difference between an enumeration and a list of
    structures someone typed in.

    `kind` is "substitutional" (decorates a fixed scaffold), "structural"
    (changes the scaffold), or "parent".
    """
    smiles: str
    kind: str
    transform: str
    parent_smiles: str
    net_charge: int = 0
    notes: list[str] = field(default_factory=list)

    @cached_property
    def canonical(self) -> str:
        """Canonical SMILES — the identity used for deduplication.

        Raises rather than returning None for an unparseable structure: a
        candidate that cannot be interpreted must not silently become a
        distinct entry in a deduplicated set.
        """
        from rdkit import Chem

        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            raise ValueError(
                f"unparseable SMILES for transform {self.transform!r}: {self.smiles}"
            )
        return Chem.MolToSmiles(mol)

    @property
    def is_parent(self) -> bool:
        """Compared on CANONICAL form, not string equality — the same molecule
        written two ways is still the parent."""
        from rdkit import Chem

        p = Chem.MolFromSmiles(self.parent_smiles)
        return p is not None and self.canonical == Chem.MolToSmiles(p)
