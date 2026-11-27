"""PDB export for ``ferric.Molecule``.

A dependency-free writer for dropping a geometry into a viewer: one ``HETATM``
record per atom in the fixed PDB columns, coordinates in Angstrom, all atoms
in one residue ``MOL`` of chain ``A``. No connectivity, residues or cell is
written, and ghost atoms (basis functions without a nucleus) are left out.
"""

from __future__ import annotations

import os

import ferric

__all__ = ["to_pdb", "write_pdb"]

_MAX_SERIAL = 99999  # the serial field is columns 7-11


def _atom_name(element: str) -> str:
    """Columns 13-16: the element symbol right-justified in 13-14, as PDB
    aligns atom names (" C  ", "BR  ")."""
    return f"{element:>2s}  "


def to_pdb(mol: ferric.Molecule) -> str:
    """Return ``mol`` as PDB text (coordinates in Angstrom, three decimals).

    Raises ``ValueError`` for more than 99999 atoms or a coordinate that does
    not fit the 8-character field (outside -999.999 .. 9999.999 Angstrom).
    """
    atoms = [
        (symbol.upper(), xyz)
        for symbol, xyz, ghost in zip(mol.symbols(), mol.coords(), mol.is_ghost())
        if not ghost
    ]
    if len(atoms) > _MAX_SERIAL:
        raise ValueError(
            f"PDB serial numbers stop at {_MAX_SERIAL}; got {len(atoms)} atoms"
        )
    lines = []
    for serial, (element, xyz) in enumerate(atoms, start=1):
        fields = [f"{c:8.3f}" for c in xyz]
        if any(len(f) > 8 for f in fields):
            raise ValueError(
                f"atom {serial} coordinate {tuple(xyz)} does not fit the PDB 8.3f field"
            )
        lines.append(
            f"HETATM{serial:5d} {_atom_name(element)} MOL A   1    "
            f"{''.join(fields)}  1.00  0.00          {element:>2s}"
        )
    lines.append("END")
    return "\n".join(lines) + "\n"


def write_pdb(path: str | os.PathLike[str], mol: ferric.Molecule) -> None:
    """Write ``mol`` to ``path`` as PDB (see ``to_pdb``)."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(to_pdb(mol))
