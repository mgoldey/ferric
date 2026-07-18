"""Classical electrostatic potential/field from a pocket's point charges,
evaluated at arbitrary sites (e.g. ligand atom positions).

This is pure Coulomb's law over `PocketCharges.charges` — no SCF, no basis
set, no ferric QM call. It's the cheap classical-side counterpart to
`properties.compute_charges`/`compute_alpha_atomic` (which need a converged
QM density): those describe the ligand's own electronic response, this
describes the raw external field the pocket exerts before any ligand
response is computed. All positions and outputs are in Bohr atomic units.
"""
from __future__ import annotations

import numpy as np

from .pocket_charges import ANGSTROM_TO_BOHR, PocketCharges


def pocket_field_at_atoms(
    pocket: PocketCharges,
    site_coords_angstrom: list[tuple[float, float, float]],
) -> np.ndarray:
    """Classical Coulomb potential and field at each site from `pocket.charges`.

    Returns an (N, 4) array per site: `[phi, Ex, Ey, Ez]`, in Hartree atomic
    units (phi in e/Bohr, E in e/Bohr^2). `site_coords_angstrom` is typically
    a ligand's atom positions (e.g. from `_read_xyz_atoms`).
    """
    sites_bohr = np.array(
        [(x * ANGSTROM_TO_BOHR, y * ANGSTROM_TO_BOHR, z * ANGSTROM_TO_BOHR)
         for x, y, z in site_coords_angstrom],
        dtype=np.float64,
    )
    charges = np.array([c[0] for c in pocket.charges], dtype=np.float64)
    sources = np.array([(c[1], c[2], c[3]) for c in pocket.charges], dtype=np.float64)

    out = np.zeros((len(sites_bohr), 4), dtype=np.float64)
    for i, site in enumerate(sites_bohr):
        d = site[None, :] - sources  # (n_pocket, 3)
        r = np.linalg.norm(d, axis=1)
        if np.any(r == 0.0):
            raise ValueError(
                f"site {i} coincides exactly with a pocket point charge — "
                "check overlap filtering upstream (embed_ligand's overlap_cutoff_angstrom)."
            )
        out[i, 0] = np.sum(charges / r)
        out[i, 1:4] = np.sum((charges / r**3)[:, None] * d, axis=0)
    return out
