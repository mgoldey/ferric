"""Cheap classical electrostatic pre-screen for ligand poses, ahead of QM.

Wraps `pocket_field.pocket_field_at_atoms` around an already-`embed_ligand`'d
pose: no SCF, just Coulomb's law over the pose's own (overlap-filtered)
`point_charges` evaluated at the ligand's own atom positions. Useful for
ranking/filtering many candidate poses (docking output, conformer ensembles)
down to the handful worth spending real QM time on via
`binding_energy.compute_binding_energy` — the classical field a pocket
exerts at a badly-clashing or poorly-oriented pose's atoms is usually already
a strong, nearly-free signal before any electronic response is computed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ligand_embedding import EmbeddedLigand
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
