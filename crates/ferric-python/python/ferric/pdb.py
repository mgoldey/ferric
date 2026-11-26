"""PDB export for ``ferric.Molecule``.

A minimal, dependency-free PDB writer: one ``ATOM`` record per atom, coordinates
in Angstrom, element symbol in the element column. It is intentionally simple
(no residues, no connectivity) so a structure can be dropped into any viewer.
"""

from __future__ import annotations

import numpy as np

import ferric


def to_pdb(mol: ferric.Molecule) -> str:
    """Return a PDB-format string for ``mol`` (coordinates in Angstrom)."""
    coords = np.asarray(mol.coords(), dtype=float)
    symbols = list(mol.symbols())
    lines = []
    for i, (sym, (x, y, z)) in enumerate(zip(symbols, coords), start=1):
        element = sym.capitalize()
        lines.append(
            f"ATOM  {i:5d}  {element:<2s}  MOL A{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
        )
    lines.append("END")
    return "\n".join(lines) + "\n"


def write_pdb(path: str, mol: ferric.Molecule) -> None:
    """Write ``mol`` to ``path`` as PDB."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(to_pdb(mol))
