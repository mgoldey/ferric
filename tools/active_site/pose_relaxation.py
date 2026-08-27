"""Relax a ligand pose's GEOMETRY inside a pocket's fixed point-charge field.

`binding_energy.py` only ever does single-point energies: the ligand's
Cartesian coordinates are frozen at whatever pose produced the input xyz
(cryo-EM, docking output, a generated conformer, ...). Real docking
refinement wants the pose to *settle* in the pocket's electrostatic
environment first -- letting bond lengths/angles/torsions relax under the
QM/MM Hamiltonian before scoring -- rather than scoring a geometry that was
never actually optimized against this pocket's field. ferric's
`ferric.run_optimize(..., point_charges=...)` already does exactly this
(geometry optimization with a fixed external point-charge field folded into
`hcore` once before each SCF call, per
docs/superpowers/specs/2026-07-10-external-potentials-design.md); this
module is the thin `tools/active_site` wrapper that was missing, following
the same `EmbeddedLigand.point_charges` threading convention as
`energy.compute_energy`.

Typical use, feeding into the rest of the pipeline:

    pocket   = derive_pocket_charges(pocket_pdb)
    embedded = embed_ligand(ligand_xyz, pocket=pocket, basis="sto-3g")
    relaxed  = relax_pose_in_pocket_field(embedded)
    if not relaxed.converged:
        ...  # a stalled optimization is a normal outcome for some poses,
             # not an exception -- the caller must check this explicitly
             # before trusting relaxed.energy as a settled-pose energy.

The optimized geometry comes back through `ferric.Molecule.coords()` (Å),
so `RelaxedPose.coords_angstrom` is the settled pose, ready for
`embed_ligand_from_coords`/further tooling. Atom order is preserved by the
optimizer, so `symbols` is taken from the input `EmbeddedLigand`.
"""
from __future__ import annotations

from dataclasses import dataclass

import ferric

from .ligand_embedding import EmbeddedLigand
from .pocket_charges import ANGSTROM_TO_BOHR


@dataclass
class RelaxedPose:
    """Outcome of relaxing one `EmbeddedLigand`'s geometry in its pocket field.

    `coords_angstrom` is the optimized geometry, in Angstrom, ready to feed
    back into `embed_ligand_from_coords`/further tooling. It is the geometry
    at which `energy` was evaluated whether or not the optimizer converged,
    so check `converged` before treating it as a settled pose.

    `converged` must be checked by the caller before trusting `energy` as a
    settled-pose energy -- a stalled geometry optimization is a normal,
    expected outcome for some poses (e.g. a badly clashing starting
    geometry), not a hard error, so this function never raises on
    non-convergence. It also never fabricates a "done" result: `converged`
    reports exactly what `ferric.run_optimize` reported, unmodified.
    """
    energy: float  # Hartree, in-field, at the (attempted) relaxed geometry
    converged: bool
    steps: int
    coords_angstrom: list[tuple[float, float, float]]
    symbols: list[str]  # unchanged from the input EmbeddedLigand (atom order preserved)
    n_pocket_charges: int


def relax_pose_in_pocket_field(
    embedded: EmbeddedLigand,
    max_steps: int = 100,
    e_conv: float = 1e-6,
) -> RelaxedPose:
    """Optimize `embedded`'s ligand geometry inside its pocket's fixed
    point-charge field via `ferric.run_optimize`.

    Requires `embedded` to have a pocket attached with at least one
    surviving (overlap-filtered) point charge -- same precondition as
    `prescreen.prescreen_pose`, raising `ValueError` for the same reason:
    relaxing a ligand with no field to relax against is not meaningfully
    different from a vacuum optimization, and silently falling back to
    vacuum would make the "in pocket field" contract in this function's name
    a lie. Use a plain (unwritten) vacuum `ferric.run_optimize` call directly
    if a vacuum relaxation is actually what's wanted.

    `max_steps`/`e_conv` pass straight through to `ferric.run_optimize` for
    convergence tuning (same defaults as the Rust side).
    """
    if embedded.pocket is None or not embedded.point_charges:
        raise ValueError(
            "relax_pose_in_pocket_field requires embedded.point_charges "
            "(embed_ligand must be called with a pocket, and at least one "
            "pocket charge must survive overlap filtering) -- nothing to "
            "relax the ligand geometry against."
        )

    result = ferric.run_optimize(
        embedded.mol,
        embedded.basis_name,
        max_steps=max_steps,
        e_conv=e_conv,
        point_charges=embedded.point_charges,
    )

    return RelaxedPose(
        energy=result.energy,
        converged=result.converged,
        steps=result.steps,
        coords_angstrom=[tuple(c) for c in result.mol().coords()],
        symbols=list(embedded.symbols),
        n_pocket_charges=embedded.pocket.n_charges,
    )


# Bohr -> Angstrom, the exact inverse of pqr_parser.ANGSTROM_TO_BOHR /
# ligand_embedding.py's Angstrom -> Bohr conversions (defined from the same
# constant rather than a second hardcoded literal). `ferric.Molecule.coords()`
# already returns Angstrom; this is for callers holding `coords_bohr()` values.
BOHR_TO_ANGSTROM = 1.0 / ANGSTROM_TO_BOHR
