"""Electronic properties (Hirshfeld/Löwdin charges) for a converged EnergyResult."""
from __future__ import annotations

import ferric

from .energy import EnergyResult
from .ligand_embedding import EmbeddedLigand


def compute_charges(embedded: EmbeddedLigand, energy: EnergyResult) -> dict:
    """Hirshfeld and Löwdin partial charges for `energy`'s converged density.

    Works for both RHF and DFT `EnergyResult`s (ferric's property functions
    accept either result type).
    """
    return {
        "hirshfeld": ferric.hirshfeld_charges(embedded.mol, embedded.basis_set, energy.raw),
        "lowdin": ferric.lowdin_charges(embedded.mol, embedded.basis_set, energy.raw),
    }
