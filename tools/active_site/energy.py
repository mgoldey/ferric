"""Single-point energy evaluation for an EmbeddedLigand: vacuum or field."""
from __future__ import annotations

from dataclasses import dataclass

import ferric

from .ligand_embedding import EmbeddedLigand


@dataclass
class EnergyResult:
    energy: float
    converged: bool
    raw: object  # underlying ferric RhfResult/DftResult (keeps .density(), etc.)
    method: str
    field: bool  # whether point charges were actually applied


def compute_energy(
    embedded: EmbeddedLigand,
    method: str = "rhf",
    xc: str | None = None,
    use_field: bool = True,
    **scf_kwargs,
) -> EnergyResult:
    """Evaluate one SCF energy for `embedded`.

    `use_field=True` (default) applies `embedded.point_charges` if the ligand
    was embedded with a pocket; with no pocket attached this is silently a
    vacuum run. `use_field=False` forces vacuum even when a pocket is
    attached — lets one `EmbeddedLigand` serve both a vacuum and a field
    evaluation without re-embedding.

    `**scf_kwargs` passes through to `ferric.run_rhf`/`run_dft` (e.g.
    `max_iter`, `k_builder`, `level_shift`) for convergence tuning.
    """
    point_charges = embedded.point_charges if (use_field and embedded.point_charges) else None

    if method == "rhf":
        raw = ferric.run_rhf(embedded.mol, embedded.basis_set, point_charges=point_charges, **scf_kwargs)
    elif method == "dft":
        if xc is None:
            raise ValueError("method='dft' requires xc=<functional name>")
        raw = ferric.run_dft(embedded.mol, embedded.basis_set, functional=xc,
                              point_charges=point_charges, **scf_kwargs)
    else:
        raise ValueError(f"unknown method {method!r}; expected 'rhf' or 'dft'")

    return EnergyResult(
        energy=raw.energy if hasattr(raw, "energy") else raw.total_energy,
        converged=raw.converged, raw=raw, method=method,
        field=point_charges is not None,
    )
