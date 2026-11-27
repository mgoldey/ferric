"""PDB export for ``ferric.Molecule``.

A dependency-free writer for dropping a geometry into a viewer: one ``HETATM``
record per atom in the fixed PDB columns, coordinates in Angstrom, in residue
``MOL`` of chain ``A``. Atom names are the element symbol plus a per-element
counter (``H1``, ``H2``), unique within a residue as PDB readers expect; a
new residue starts when a counter would outgrow the four-column name. No
connectivity or cell is written, and ghost atoms (basis functions without a
nucleus) are left out.
"""

from __future__ import annotations

import os

import ferric

__all__ = ["to_pdb", "write_pdb"]

_MAX_SERIAL = 99999  # the serial field is columns 7-11
_MAX_RESIDUE = 9999  # the residue number is columns 23-26


def _atom_name(element: str, count: int) -> str | None:
    """Columns 13-16 for the ``count``-th atom of ``element`` in a residue:
    the symbol right-justified in 13-14, as PDB aligns atom names (" H1 ",
    "BR1 "), or flush left when a one-letter symbol needs all four ("H100").
    ``None`` when the counter does not fit."""
    name = f"{element}{count}"
    if len(name) > 4:
        return None
    return name if len(name) == 4 else f"{element:>2s}{count}".ljust(4)


def to_pdb(mol: ferric.Molecule) -> str:
    """Return ``mol`` as PDB text (coordinates in Angstrom, three decimals).

    Raises ``ValueError`` for more than 99999 atoms, more than 9999 residues
    or a coordinate that does not fit the 8-character field (outside
    -999.999 .. 9999.999 Angstrom).
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
    residue = 1
    counts: dict[str, int] = {}
    for serial, (element, xyz) in enumerate(atoms, start=1):
        fields = [f"{c:8.3f}" for c in xyz]
        if any(len(f) > 8 for f in fields):
            raise ValueError(
                f"atom {serial} coordinate {tuple(xyz)} does not fit the PDB 8.3f field"
            )
        counts[element] = counts.get(element, 0) + 1
        name = _atom_name(element, counts[element])
        if name is None:
            residue += 1
            counts = {element: 1}
            name = _atom_name(element, 1)
        if residue > _MAX_RESIDUE:
            raise ValueError(f"PDB residue numbers stop at {_MAX_RESIDUE}")
        lines.append(
            f"HETATM{serial:5d} {name} MOL A{residue:4d}    "
            f"{''.join(fields)}  1.00  0.00          {element:>2s}"
        )
    lines.append("END")
    return "\n".join(lines) + "\n"


def write_pdb(path: str | os.PathLike[str], mol: ferric.Molecule) -> None:
    """Write ``mol`` to ``path`` as PDB (see ``to_pdb``). The text is built
    first, so a refused molecule leaves an existing file untouched."""
    text = to_pdb(mol)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
