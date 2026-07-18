"""Derive classical point charges for a protein pocket from a PDB file."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .pdb2pqr_runner import run_pdb2pqr
from .pqr_parser import ANGSTROM_TO_BOHR, parse_pqr

PointCharge = tuple[float, float, float, float]


@dataclass
class PocketCharges:
    """A pocket's classical point charges, derived once and reusable across
    many ligand evaluations — cheap to construct, plain-data (picklable),
    safe to pass into multiprocessing/BoTorch-style optimization workers.
    """
    charges: list[PointCharge]
    source_pdb: Path
    ff: str
    n_charges: int = field(init=False)

    def __post_init__(self):
        self.n_charges = len(self.charges)


def _too_close(px: float, py: float, pz: float, ligand_bohr: list[tuple[float, float, float]],
                cutoff_bohr: float) -> bool:
    for lx, ly, lz in ligand_bohr:
        d2 = (px - lx) ** 2 + (py - ly) ** 2 + (pz - lz) ** 2
        if d2 < cutoff_bohr * cutoff_bohr:
            return True
    return False


def pocket_point_charges(
    pocket_pdb: str | Path,
    ff: str = "AMBER",
    ligand_coords_angstrom: list[tuple[float, float, float]] | None = None,
    overlap_cutoff_angstrom: float = 1.5,
) -> list[PointCharge]:
    """Run PDB2PQR on `pocket_pdb` and return (q, x, y, z) charges in Bohr.

    If `ligand_coords_angstrom` is given, pocket atoms within
    `overlap_cutoff_angstrom` of any ligand atom are dropped — this guards
    against double-counting when the source PDB still contains the ligand's
    own HETATM records (e.g. a receptor file that wasn't pre-stripped of its
    bound ligand).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        pqr_path = Path(tmpdir) / "pocket.pqr"
        run_pdb2pqr(pocket_pdb, pqr_path, ff=ff)
        charges = parse_pqr(pqr_path)

    if not charges:
        raise RuntimeError(f"PDB2PQR produced no ATOM/HETATM charges for {pocket_pdb}")

    if ligand_coords_angstrom:
        ligand_bohr = [(x * ANGSTROM_TO_BOHR, y * ANGSTROM_TO_BOHR, z * ANGSTROM_TO_BOHR)
                        for x, y, z in ligand_coords_angstrom]
        cutoff_bohr = overlap_cutoff_angstrom * ANGSTROM_TO_BOHR
        charges = [c for c in charges if not _too_close(c[1], c[2], c[3], ligand_bohr, cutoff_bohr)]
        if not charges:
            raise RuntimeError(
                "All pocket charges were filtered out as ligand-overlapping — "
                "check ligand_coords_angstrom/overlap_cutoff_angstrom."
            )

    return charges


def derive_pocket_charges(
    pocket_pdb: str | Path,
    ff: str = "AMBER",
    ligand_coords_angstrom: list[tuple[float, float, float]] | None = None,
    overlap_cutoff_angstrom: float = 1.5,
) -> PocketCharges:
    """Derive a pocket's point charges once, as a reusable `PocketCharges`.

    Call this once per pocket, then reuse the result across many ligand
    evaluations (`embed_ligand`) instead of re-running PDB2PQR per candidate.
    """
    charges = pocket_point_charges(pocket_pdb, ff, ligand_coords_angstrom, overlap_cutoff_angstrom)
    return PocketCharges(charges=charges, source_pdb=Path(pocket_pdb), ff=ff)
