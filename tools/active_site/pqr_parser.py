"""Parser for PQR files (PDB2PQR output) into (q, x, y, z) point charges.

PQR ATOM/HETATM lines are whitespace-delimited (unlike fixed-width PDB),
because PDB2PQR widens the atom-name column for 4-character names (e.g.
`HG21`). A standard PQR ATOM/HETATM line has exactly 10 whitespace-separated
fields:

    ATOM  serial  name  resName  resSeq  x  y  z  charge  radius

verified against a live `pdb2pqr30 --ff=AMBER` run (crambin/1CRN): every
ATOM/HETATM line has NF == 10, with x/y/z/charge at fixed positions from the
end (radius last, charge second-to-last, then z, y, x).

PQR coordinates are in Angstrom; ferric's `point_charges` kwarg expects Bohr
(matching `PointCharge`/`Molecule` internal units — see
crates/ferric-core/src/mol.rs ANGSTROM_TO_BOHR), so this module converts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ANGSTROM_TO_BOHR = 1.0 / 0.529_177_210_92


def parse_pqr(pqr_path: str | Path) -> list[tuple[float, float, float, float]]:
    """Parse a PQR file into a list of (q, x, y, z) tuples, in Bohr."""
    charges: list[tuple[float, float, float, float]] = []
    with open(pqr_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            fields = line.split()
            if len(fields) != 10:
                raise ValueError(
                    f"Unexpected PQR field count ({len(fields)}, expected 10): {line!r}"
                )
            x, y, z, q = (float(v) for v in fields[5:9])
            charges.append((q, x * ANGSTROM_TO_BOHR, y * ANGSTROM_TO_BOHR, z * ANGSTROM_TO_BOHR))
    return charges


@dataclass(frozen=True)
class PqrAtom:
    """One ATOM/HETATM record of a PQR file, residue fields included.

    `x`, `y`, `z` and `radius` are in **Bohr** (converted from the PQR file's
    Ångström on load, same convention as `parse_pqr`).
    """

    serial: int
    name: str
    res_name: str
    res_seq: int
    q: float
    x: float
    y: float
    z: float
    radius: float


def parse_pqr_atoms(pqr_path: str | Path) -> list[PqrAtom]:
    """Parse a PQR file into `PqrAtom` records — the residue-aware superset
    of `parse_pqr`'s thin `(q, x, y, z)` view.

    Same 10-whitespace-field line format as `parse_pqr` (see module docs):

        ATOM  serial  name  resName  resSeq  x  y  z  charge  radius

    `parse_pqr_atoms(path)[i]` carries the same `(q, x, y, z)` as
    `parse_pqr(path)[i]`, bit for bit — this is a superset, not a different
    computation.
    """
    atoms: list[PqrAtom] = []
    with open(pqr_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            fields = line.split()
            if len(fields) != 10:
                raise ValueError(
                    f"Unexpected PQR field count ({len(fields)}, expected 10): {line!r}"
                )
            _, serial, name, res_name, res_seq, x, y, z, q, radius = fields
            atoms.append(
                PqrAtom(
                    serial=int(serial),
                    name=name,
                    res_name=res_name,
                    res_seq=int(res_seq),
                    q=float(q),
                    x=float(x) * ANGSTROM_TO_BOHR,
                    y=float(y) * ANGSTROM_TO_BOHR,
                    z=float(z) * ANGSTROM_TO_BOHR,
                    radius=float(radius) * ANGSTROM_TO_BOHR,
                )
            )
    return atoms
