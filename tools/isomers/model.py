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

    def deprotonated(self) -> "Isomer | None":
        """This isomer as its pH-7.4 anion, or None if it has no acidic proton.

        DEPROTONATION MUST CHANGE THE STRUCTURE, not just the declared charge.
        Removing H+ takes a proton and leaves its electrons behind, so the anion
        has the SAME electron count as the neutral acid. Setting net_charge=-1
        on the neutral SMILES therefore asks the solver for an electron that
        does not exist -- which ferric correctly rejects as "N electrons with
        multiplicity 1 implies n_alpha = N/2, not an integer" when N is odd.

        Returns a new Isomer whose SMILES carries the [O-]/[N-] explicitly.
        """
        from rdkit import Chem

        mol = Chem.MolFromSmiles(self.canonical)
        if mol is None:
            return None

        # SMARTS that locate the ACIDIC HEAVY ATOM itself (the O or N that
        # carries the proton), in the order we prefer to deprotonate. Editing
        # that atom directly is more robust than substructure replacement,
        # which needs a valid SMILES for the product fragment and silently
        # fails if the replacement string is SMARTS rather than SMILES.
        acidic_sites = (
            "[OX2H1][CX3]=O",                    # carboxylic acid O-H
            # Tetrazole N-H. Matched via the aromatic-N-in-a-tetrazole-ring
            # pattern with the acidic N FIRST, because a ring-closure SMARTS
            # written from the wrong atom does not match at all (verified:
            # "[nX3H1]1nnnc1" matches nothing on c1ccccc1c1nn[nH]n1).
            "[nX3H1]:n:n:n:c",
            "[nX3H1]:n:n:c:n",
            "[NX3H1]([SX4](=O)=O)[CX3]=O",       # acylsulfonamide N-H
            "[OX2H1][SX4](=O)=O",                # sulfonic acid O-H
        )
        for smarts in acidic_sites:
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                continue
            match = mol.GetSubstructMatch(patt)
            if not match:
                continue
            rw = Chem.RWMol(mol)
            atom = rw.GetAtomWithIdx(match[0])
            if atom.GetTotalNumHs() < 1:
                continue
            atom.SetNumExplicitHs(max(0, atom.GetTotalNumHs() - 1))
            atom.SetNoImplicit(True)
            atom.SetFormalCharge(-1)
            prod = rw.GetMol()
            try:
                Chem.SanitizeMol(prod)
            except Exception:  # noqa: BLE001
                continue
            if Chem.GetFormalCharge(prod) != -1:
                continue
            return Isomer(
                smiles=Chem.MolToSmiles(prod), kind=self.kind,
                transform=f"{self.transform} (deprotonated)",
                parent_smiles=self.parent_smiles, net_charge=-1,
                notes=[*self.notes, "pH-7.4 anion"],
            )
        return None

    @property
    def is_parent(self) -> bool:
        """Compared on CANONICAL form, not string equality — the same molecule
        written two ways is still the parent."""
        from rdkit import Chem

        p = Chem.MolFromSmiles(self.parent_smiles)
        return p is not None and self.canonical == Chem.MolToSmiles(p)
