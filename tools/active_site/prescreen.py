"""Cheap classical electrostatic pre-screen for ligand poses, ahead of QM.

Wraps `pocket_field.pocket_field_at_atoms` around an already-`embed_ligand`'d
pose: no SCF, just Coulomb's law over the pose's own (overlap-filtered)
`point_charges` evaluated at the ligand's own atom positions. Useful for
ranking/filtering many candidate poses (docking output, conformer ensembles)
down to the handful worth spending real QM time on via
`binding_energy.compute_binding_energy` — the classical field a pocket
exerts at a badly-clashing or poorly-oriented pose's atoms is usually already
a strong, nearly-free signal before any electronic response is computed.

`batch_prescreen` is the entry point for a real screening run: derive the
pocket's charges ONCE (`pocket_charges.derive_pocket_charges`), then rank an
entire conformer/pose ensemble against it in one call. Typical shape of a
real workflow:

    pocket = derive_pocket_charges(pocket_pdb)
    ranked = batch_prescreen(pocket, ligand_xyz_paths, charge_source=my_charges)
    top_5 = [r for r in ranked if r.error is None][:5]
    for r in top_5:
        result = compute_binding_energy(r.ligand_xyz, pocket_pdb, ...)  # real QM
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .ligand_embedding import EmbeddedLigand, embed_ligand
from .pocket_charges import PocketCharges
from .pocket_field import pocket_field_at_atoms


@dataclass
class PrescreenResult:
    """Per-atom classical field data plus a single scalar `score` for ranking.

    `score` is the pocket's classical electrostatic potential energy on the
    ligand's (unpolarized, formal) charges, in Hartree: `Σ_atom q_atom *
    phi_pocket(atom)`. This is NOT the QM binding energy (no polarization,
    no exchange/dispersion, no self-consistent response) — it is a cheap,
    monotonic-ish proxy for "does this pose sit somewhere electrostatically
    reasonable," meant purely for ranking/filtering before QM, not for a
    final energy number.
    """
    field_at_atoms: np.ndarray  # (N, 4): [phi, Ex, Ey, Ez] per ligand atom, a.u.
    formal_charges: np.ndarray  # (N,) formal atomic charges used for `score`
    score: float  # Hartree; more negative = more electrostatically favorable
    n_pocket_charges: int  # how many (overlap-filtered) pocket charges contributed


# Formal atomic charges are a crude placeholder (all zero -> score is
# always 0) unless the caller supplies real partial charges (e.g. from a
# force-field-typed ligand, or a prior QM run's Hirshfeld/Löwdin charges via
# `properties.compute_charges`). Kept as an explicit, required argument
# rather than silently defaulting to formal charge 0 for every atom, which
# would make `score` meaningless without the caller realizing it.
def prescreen_pose(
    embedded: EmbeddedLigand,
    atom_charges: list[float] | np.ndarray,
) -> PrescreenResult:
    """Classical field-based pre-screen for one embedded ligand pose.

    `atom_charges` must have one entry per ligand atom, same order as
    `embedded.mol`'s atoms (e.g. from a force-field typing, or from a cheap
    prior QM run's `properties.compute_charges(...)["hirshfeld"]`). Raises
    `ValueError` if `embedded` has no pocket attached (nothing to score
    against) or the charge count doesn't match the atom count.
    """
    if embedded.pocket is None or not embedded.point_charges:
        raise ValueError(
            "prescreen_pose requires embedded.point_charges (embed_ligand must "
            "be called with a pocket, and at least one pocket charge must "
            "survive overlap filtering) -- nothing to score against."
        )

    charges = np.asarray(atom_charges, dtype=np.float64)
    if len(charges) != len(embedded.coords_angstrom):
        raise ValueError(
            f"atom_charges has {len(charges)} entries but the ligand has "
            f"{len(embedded.coords_angstrom)} atoms -- must match 1:1."
        )

    # Score against the SAME overlap-filtered charges the QM field
    # embedding would actually see (embedded.point_charges), not the raw
    # unfiltered pocket -- otherwise the classical proxy and the QM energy
    # it's meant to rank against would disagree about which pocket charges
    # are even in play.
    filtered_pocket = PocketCharges(
        charges=[(q, x, y, z) for (q, x, y, z) in embedded.point_charges],
        source_pdb=embedded.pocket.source_pdb,
        ff=embedded.pocket.ff,
    )
    field = pocket_field_at_atoms(filtered_pocket, embedded.coords_angstrom)
    score = float(np.dot(charges, field[:, 0]))

    return PrescreenResult(
        field_at_atoms=field,
        formal_charges=charges,
        score=score,
        n_pocket_charges=filtered_pocket.n_charges,
    )


ChargeSource = Callable[[EmbeddedLigand], "list[float] | np.ndarray"]


@dataclass
class BatchPrescreenEntry:
    """One conformer's outcome in a `batch_prescreen` run.

    `result` is `None` and `error` is set if this pose failed embedding or
    scoring (e.g. a malformed xyz, or every pocket charge filtered out as
    ligand-overlapping) — batch runs over real-world conformer ensembles
    routinely include a few bad poses, and a single failure must not abort
    the whole ranking. `rank` is filled in on the SORTED list `batch_prescreen`
    returns (1-based, ascending score = most favorable first); entries with
    `error is not None` are sorted to the end with `rank = None`.
    """
    ligand_xyz: Path
    result: PrescreenResult | None
    error: str | None
    rank: int | None = field(default=None)


def batch_prescreen(
    pocket: PocketCharges,
    ligand_xyz_paths: list[str | Path],
    charge_source: ChargeSource,
    basis: str = "def2-svp",
    overlap_cutoff_angstrom: float = 1.5,
) -> list[BatchPrescreenEntry]:
    """Rank a whole conformer/pose ensemble against one pocket by the cheap
    classical prescreen score, most electrostatically favorable first.

    `pocket` should be derived ONCE via `pocket_charges.derive_pocket_charges`
    and reused across the whole ensemble (this is the entire reason
    `PocketCharges` exists as its own reusable object — re-running PDB2PQR
    per conformer would be wasteful and pointless, since the pocket doesn't
    change between ligand poses).

    `charge_source(embedded) -> atom_charges` is called once per successfully
    embedded pose to get its `prescreen_pose` charges — e.g. a fixed
    per-atom-index force-field charge table for a single rigid ligand
    topology screened across many poses, or a cheap semi-empirical/QM charge
    computation per conformer if the atom ORDER can vary between them. There
    is deliberately no default charge source (same reasoning as
    `prescreen_pose`'s mandatory `atom_charges`: a silent all-zero fallback
    would make every score exactly 0 and the whole ranking meaningless
    without the caller necessarily noticing).

    Returns one `BatchPrescreenEntry` per input path, in RANKED order
    (ascending score; failed poses last, in input order, `rank=None`) — not
    in input order. Never raises for a single bad conformer; check `.error`
    on each entry. Basis mismatches, missing files, etc. are the same kind of
    per-pose failure and are caught here too.
    """
    entries: list[BatchPrescreenEntry] = []
    for xyz_path in ligand_xyz_paths:
        xyz_path = Path(xyz_path)
        try:
            embedded = embed_ligand(
                xyz_path, pocket=pocket, basis=basis,
                overlap_cutoff_angstrom=overlap_cutoff_angstrom,
            )
            charges = charge_source(embedded)
            result = prescreen_pose(embedded, charges)
            entries.append(BatchPrescreenEntry(ligand_xyz=xyz_path, result=result, error=None))
        except Exception as e:  # noqa: BLE001 -- deliberately broad: one bad
            # conformer (malformed xyz, all pocket charges filtered out,
            # charge_source raising, ...) must not abort the whole batch.
            entries.append(BatchPrescreenEntry(ligand_xyz=xyz_path, result=None, error=str(e)))

    ok = sorted((e for e in entries if e.error is None), key=lambda e: e.result.score)
    failed = [e for e in entries if e.error is not None]
    for i, e in enumerate(ok, start=1):
        e.rank = i
    return ok + failed
