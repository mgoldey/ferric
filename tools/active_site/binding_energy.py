"""Ligand-in-pocket electrostatic binding energy: field-vs-vacuum QM.

E_int = E(ligand, QM, embedded in the pocket's classical point-charge field)
      - E(ligand, QM, vacuum)

The pocket is always classical point charges (via PDB2PQR); only the ligand
is ever a QM `Molecule`. This sidesteps BSSE/counterpoise entirely since
there is no QM-QM supermolecular interaction energy being computed.

This is a thin orchestration over four composable, independently reusable
stages — read it as a worked example for building your own variant (e.g.
cache `PocketCharges` across many ligands/conformers, skip `compute_charges`
for speed, or swap `use_field`/`method` per call):

    pocket    = derive_pocket_charges(pocket_pdb)      # once per pocket
    embedded  = embed_ligand(ligand_xyz, pocket)        # once per ligand geometry
    e_vacuum  = compute_energy(embedded, use_field=False)
    e_field   = compute_energy(embedded, use_field=True)
    charges   = compute_charges(embedded, e_field)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import psutil

from .energy import compute_energy
from .ligand_embedding import embed_ligand
from .pocket_charges import derive_pocket_charges
from .properties import compute_charges

HARTREE_TO_KCAL_MOL = 627.5094740631


def check_available_memory(min_available_gb: float) -> None:
    """Refuse to launch an SCF job when free system memory is below
    `min_available_gb`. ferric's Fock builders don't budget against total
    system RAM (OPENBLAS_NUM_THREADS=1 controls BLAS thread count, not peak
    working-set size), so a job launched on an already memory-pressured box
    can be OOM-killed alongside unrelated concurrent jobs. This is a
    pre-flight guard, not a hard per-process memory cap.
    """
    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    if available_gb < min_available_gb:
        raise MemoryError(
            f"Only {available_gb:.1f} GB available (need >= {min_available_gb:.1f} GB). "
            "Free up memory or lower min_available_gb before running this job — "
            "on a loaded box this job can be OOM-killed alongside other processes."
        )


@dataclass
class BindingEnergyResult:
    e_vacuum: float
    e_field: float
    delta_e_hartree: float
    delta_e_kcal_mol: float
    charges_vacuum: dict
    charges_field: dict
    n_pocket_charges: int


def compute_binding_energy(
    ligand_xyz: str | Path,
    pocket_pdb: str | Path,
    basis: str = "def2-svp",
    method: str = "rhf",
    xc: str | None = None,
    ff: str = "AMBER",
    min_available_gb: float = 2.0,
) -> BindingEnergyResult:
    """Compute the field-vs-vacuum electrostatic binding energy for a ligand
    in a protein pocket, plus Hirshfeld/Löwdin charges in both states.

    Raises MemoryError up front if fewer than `min_available_gb` GB are free
    (set to 0 to disable the check).
    """
    if min_available_gb > 0:
        check_available_memory(min_available_gb)

    pocket = derive_pocket_charges(pocket_pdb, ff=ff)
    embedded = embed_ligand(ligand_xyz, pocket=pocket, basis=basis)

    e_vac = compute_energy(embedded, method=method, xc=xc, use_field=False)
    e_field = compute_energy(embedded, method=method, xc=xc, use_field=True)

    charges_vacuum = compute_charges(embedded, e_vac)
    charges_field = compute_charges(embedded, e_field)

    delta_e = e_field.energy - e_vac.energy

    return BindingEnergyResult(
        e_vacuum=e_vac.energy,
        e_field=e_field.energy,
        delta_e_hartree=delta_e,
        delta_e_kcal_mol=delta_e * HARTREE_TO_KCAL_MOL,
        charges_vacuum=charges_vacuum,
        charges_field=charges_field,
        n_pocket_charges=pocket.n_charges,
    )
