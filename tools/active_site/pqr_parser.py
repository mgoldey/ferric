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
