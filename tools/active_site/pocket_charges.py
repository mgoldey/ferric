"""Derive classical point charges for a protein pocket from a PDB file."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .pdb2pqr_runner import run_pdb2pqr
from .pqr_parser import ANGSTROM_TO_BOHR, parse_pqr, parse_pqr_atoms

PointCharge = tuple[float, float, float, float]


@dataclass
class PocketCharges:
    """A pocket's classical point charges, derived once and reusable across
    many ligand evaluations — cheap to construct, plain-data (picklable),
    safe to pass into multiprocessing/BoTorch-style optimization workers.

    `residue_ids`/`atom_names`/`res_names`, when present, are parallel to
    `charges` (same length, same order) and default to `None` — the old
    construction path (charges/source_pdb/ff only) is unchanged, and only
    `derive_pocket_charges` populates them (needed by lane C for OpenMM
    parametrisation and by `QmmmSystem`'s whole-residue selection).
    """
    charges: list[PointCharge]
    source_pdb: Path
    ff: str
    residue_ids: list[int] | None = None
    atom_names: list[str] | None = None
    res_names: list[str] | None = None
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

    Unlike `pocket_point_charges`, this also runs PDB2PQR through
    `parse_pqr_atoms` to populate `residue_ids`/`atom_names`/`res_names` —
    ready to feed straight into `QmSelection::WithinRadiusWholeResidues`.

    PQR (as PDB2PQR emits it) has no chain column, and `res_seq` restarts
    per chain — so residue ids are assigned **by contiguous run of
    `(res_name, res_seq)`**, never by `res_seq` (or `(res_name, res_seq)`)
    value alone: a new residue id starts the instant this atom's
    `(res_name, res_seq)` differs from the file's immediately preceding
    atom, for any reason (a genuinely new residue, a `res_seq` decrease —
    PDB2PQR's chain-break signal — or the SAME `(res_name, res_seq)` pair
    recurring later once other residues have intervened, e.g. chain B
    restarting at residue 1 with the same residue names as chain A).
    Because ids only ever increment forward as the file is walked, a pair
    seen earlier is NEVER merged back into its old run: chain A's residue
    12 and chain B's residue 12 always get distinct ids, even though both
    carry the same raw `res_seq`.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        pqr_path = Path(tmpdir) / "pocket.pqr"
        run_pdb2pqr(pocket_pdb, pqr_path, ff=ff)
        pqr_atoms = parse_pqr_atoms(pqr_path)

    if not pqr_atoms:
        raise RuntimeError(f"PDB2PQR produced no ATOM/HETATM charges for {pocket_pdb}")

    if ligand_coords_angstrom:
        ligand_bohr = [(x * ANGSTROM_TO_BOHR, y * ANGSTROM_TO_BOHR, z * ANGSTROM_TO_BOHR)
                        for x, y, z in ligand_coords_angstrom]
        cutoff_bohr = overlap_cutoff_angstrom * ANGSTROM_TO_BOHR
        pqr_atoms = [a for a in pqr_atoms if not _too_close(a.x, a.y, a.z, ligand_bohr, cutoff_bohr)]
        if not pqr_atoms:
            raise RuntimeError(
                "All pocket charges were filtered out as ligand-overlapping — "
                "check ligand_coords_angstrom/overlap_cutoff_angstrom."
            )

    residue_ids: list[int] = []
    next_id = -1
    prev_key: tuple[str, int] | None = None
    for a in pqr_atoms:
        # A residue's atoms are contiguous in the file and share the exact
        # same (res_name, res_seq) — so a new run starts the instant this
        # atom's (res_name, res_seq) differs from the atom immediately
        # before it, for ANY reason (a genuinely new residue, a res_seq
        # decrease/chain break, or a res_seq repeated under a different
        # res_name). Ids only ever increment, so a (res_name, res_seq) pair
        # that recurs later in the file (chain B restarting at residue 1)
        # gets a brand-new id — it can never merge back into an earlier run.
        key = (a.res_name, a.res_seq)
        if key != prev_key:
            next_id += 1
        residue_ids.append(next_id)
        prev_key = key

    charges = [(a.q, a.x, a.y, a.z) for a in pqr_atoms]
    return PocketCharges(
        charges=charges,
        source_pdb=Path(pocket_pdb),
        ff=ff,
        residue_ids=residue_ids,
        atom_names=[a.name for a in pqr_atoms],
        res_names=[a.res_name for a in pqr_atoms],
    )
