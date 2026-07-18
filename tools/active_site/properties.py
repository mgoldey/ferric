"""Electronic properties (Hirshfeld/Löwdin charges, Hirshfeld polarizability)
for a converged EnergyResult."""
from __future__ import annotations

import numpy as np

import ferric

from .energy import EnergyResult
from .ligand_embedding import EmbeddedLigand

# Bundled RI-fit aux basis paired with each supported orbital basis, for
# compute_alpha_atomic's default. Extend as more orbital bases gain RI-fit
# coverage (see crates/ferric-core/src/basis/bundled/*-rifit.json).
_DEFAULT_RIFIT_AUXBASIS = {
    "def2-svp": "def2-svp-rifit",
    "def2-tzvp": "def2-tzvp-rifit",
    "def2-qzvp": "def2-qzvp-rifit",
    "aug-cc-pvdz": "aug-cc-pvdz-rifit",
    "aug-cc-pvtz": "aug-cc-pvtz-rifit",
}


def compute_charges(embedded: EmbeddedLigand, energy: EnergyResult) -> dict:
    """Hirshfeld and Löwdin partial charges for `energy`'s converged density.

    Works for both RHF and DFT `EnergyResult`s (ferric's property functions
    accept either result type).
    """
    return {
        "hirshfeld": ferric.hirshfeld_charges(embedded.mol, embedded.basis_set, energy.raw),
        "lowdin": ferric.lowdin_charges(embedded.mol, embedded.basis_set, energy.raw),
    }


def compute_alpha_atomic(
    embedded: EmbeddedLigand,
    energy: EnergyResult,
    auxbasis: str | None = None,
    memory_budget_gb: float | None = None,
) -> np.ndarray:
    """Per-atom Hirshfeld-partitioned static polarizability tensors (Bohr^3),
    shape (N, 3, 3).

    Closed-shell only. Materially more expensive than `compute_charges` — see
    `ferric.hirshfeld_polarizability`'s docstring; controlled by the
    `FERRIC_HIRSHFELD_SPACING`/`FERRIC_HIRSHFELD_MARGIN` env vars and the
    `memory_budget_gb` cap (auto-resolved to 80% of available RAM if omitted).

    `auxbasis` defaults to the RI-fit set paired with `embedded.basis_name`
    (see `_DEFAULT_RIFIT_AUXBASIS`); pass explicitly if using a basis with no
    default pairing.
    """
    if auxbasis is None:
        auxbasis = _DEFAULT_RIFIT_AUXBASIS.get(embedded.basis_name)
        if auxbasis is None:
            raise ValueError(
                f"no default RI-fit auxbasis for '{embedded.basis_name}' — pass auxbasis explicitly"
            )
    auxbasis_set = ferric.BasisSet.bundled(auxbasis)
    return ferric.hirshfeld_polarizability(
        embedded.mol, embedded.basis_set, auxbasis_set, energy.raw,
        memory_budget_gb=memory_budget_gb,
    )
