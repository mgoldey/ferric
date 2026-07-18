"""Embed a ligand (as a ferric QM Molecule) against an optional pocket's
classical point-charge field.

`embed_ligand`/`embed_ligand_from_coords` are the composable unit between
"derive pocket charges once" (`pocket_charges.derive_pocket_charges`) and
"evaluate an energy" (`energy.compute_energy`) — call this once per ligand
geometry (once per conformer, in a screening/optimization loop) and reuse
the fixed `PocketCharges` across all of them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ferric

from .pocket_charges import ANGSTROM_TO_BOHR, PocketCharges, PointCharge


@dataclass
class _XyzAtom:
    symbol: str
    x: float
    y: float
    z: float


def _read_xyz_atoms(xyz_path: str | Path) -> list[_XyzAtom]:
    with open(xyz_path) as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    atoms = []
    for line in lines[2:2 + n]:
        parts = line.split()
        atoms.append(_XyzAtom(parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return atoms


@dataclass
class EmbeddedLigand:
    """A QM ligand (Molecule + basis) plus the (already ligand-overlap-filtered)
    pocket point charges it should be evaluated against, if any.
    """
    mol: object  # ferric.Molecule
    basis_set: object  # ferric.BasisSet
    basis_name: str
    point_charges: list[PointCharge] | None
    pocket: PocketCharges | None
    source_xyz: Path | None
    coords_angstrom: list[tuple[float, float, float]]
    symbols: list[str]


def _filter_pocket_for_ligand(
    pocket: PocketCharges | None,
    ligand_coords_angstrom: list[tuple[float, float, float]],
    overlap_cutoff_angstrom: float,
) -> list[PointCharge] | None:
    if pocket is None:
        return None
    from .pocket_charges import _too_close  # local import: private helper reuse

    ligand_bohr = [(x * ANGSTROM_TO_BOHR, y * ANGSTROM_TO_BOHR, z * ANGSTROM_TO_BOHR)
                   for x, y, z in ligand_coords_angstrom]
    cutoff_bohr = overlap_cutoff_angstrom * ANGSTROM_TO_BOHR
    return [c for c in pocket.charges if not _too_close(c[1], c[2], c[3], ligand_bohr, cutoff_bohr)]


def embed_ligand(
    ligand_xyz: str | Path,
    pocket: PocketCharges | None = None,
    basis: str = "def2-svp",
    overlap_cutoff_angstrom: float = 1.5,
) -> EmbeddedLigand:
    """Load a ligand from an xyz file and pair it with a pocket's point
    charges (re-filtered for overlap against THIS ligand's geometry — the
    pocket's own derivation is reused as-is, only the overlap check reruns).
    """
    mol = ferric.Molecule.from_xyz(str(ligand_xyz))
    basis_set = ferric.BasisSet.bundled(basis)
    atoms = _read_xyz_atoms(ligand_xyz)
    ligand_coords_angstrom = [(a.x, a.y, a.z) for a in atoms]
    point_charges = _filter_pocket_for_ligand(pocket, ligand_coords_angstrom, overlap_cutoff_angstrom)
    return EmbeddedLigand(
        mol=mol, basis_set=basis_set, basis_name=basis,
        point_charges=point_charges, pocket=pocket, source_xyz=Path(ligand_xyz),
        coords_angstrom=ligand_coords_angstrom, symbols=[a.symbol for a in atoms],
    )


def embed_ligand_from_coords(
    symbols: list[str],
    coords_angstrom: list[tuple[float, float, float]],
    pocket: PocketCharges | None = None,
    basis: str = "def2-svp",
    charge: int = 0,
    multiplicity: int = 1,
    overlap_cutoff_angstrom: float = 1.5,
) -> EmbeddedLigand:
    """Same as `embed_ligand`, but from in-memory symbols/coordinates (e.g. a
    generated conformer) instead of an xyz file — avoids a temp-file
    round-trip in a batch screening/optimization loop.
    """
    xyz_lines = [str(len(symbols)), "embed_ligand_from_coords"]
    for sym, (x, y, z) in zip(symbols, coords_angstrom):
        xyz_lines.append(f"{sym} {x:.10f} {y:.10f} {z:.10f}")
    xyz_string = "\n".join(xyz_lines) + "\n"

    mol = ferric.Molecule.from_xyz_string(xyz_string, charge, multiplicity)
    basis_set = ferric.BasisSet.bundled(basis)
    point_charges = _filter_pocket_for_ligand(pocket, coords_angstrom, overlap_cutoff_angstrom)
    return EmbeddedLigand(
        mol=mol, basis_set=basis_set, basis_name=basis,
        point_charges=point_charges, pocket=pocket, source_xyz=None,
        coords_angstrom=list(coords_angstrom), symbols=list(symbols),
    )
