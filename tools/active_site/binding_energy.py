"""Ligand-in-pocket electrostatic binding energy: field-vs-vacuum QM.

E_int = E(ligand, QM, embedded in the pocket's classical point-charge field)
      - E(ligand, QM, vacuum)

The pocket is always classical point charges (via PDB2PQR); only the ligand
is ever a QM `Molecule`. This sidesteps BSSE/counterpoise entirely since
there is no QM-QM supermolecular interaction energy being computed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import psutil

import ferric

from .pocket_charges import PointCharge, pocket_point_charges

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


def _run_energy(mol, basis, method: str, xc: str | None, point_charges: list[PointCharge] | None):
    if method == "rhf":
        return ferric.run_rhf(mol, basis, point_charges=point_charges)
    if method == "dft":
        if xc is None:
            raise ValueError("method='dft' requires xc=<functional name>")
        return ferric.run_dft(mol, basis, functional=xc, point_charges=point_charges)
    raise ValueError(f"unknown method {method!r}; expected 'rhf' or 'dft'")


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
    in a protein pocket, plus ESP/Hirshfeld/Löwdin charges in both states.

    Raises MemoryError up front if fewer than `min_available_gb` GB are free
    (set to 0 to disable the check).
    """
    if min_available_gb > 0:
        check_available_memory(min_available_gb)

    mol = ferric.Molecule.from_xyz(str(ligand_xyz))
    basis_set = ferric.BasisSet.bundled(basis)

    ligand_coords_angstrom = [(a.x, a.y, a.z) for a in _read_xyz_atoms(ligand_xyz)]
    point_charges = pocket_point_charges(
        pocket_pdb, ff=ff, ligand_coords_angstrom=ligand_coords_angstrom,
    )

    rhf_vac = _run_energy(mol, basis_set, method, xc, None)
    rhf_field = _run_energy(mol, basis_set, method, xc, point_charges)

    delta_e = rhf_field.energy - rhf_vac.energy

    charges_vacuum = {
        "hirshfeld": ferric.hirshfeld_charges(mol, basis_set, rhf_vac),
        "lowdin": ferric.lowdin_charges(mol, basis_set, rhf_vac),
    }
    charges_field = {
        "hirshfeld": ferric.hirshfeld_charges(mol, basis_set, rhf_field),
        "lowdin": ferric.lowdin_charges(mol, basis_set, rhf_field),
    }

    return BindingEnergyResult(
        e_vacuum=rhf_vac.energy,
        e_field=rhf_field.energy,
        delta_e_hartree=delta_e,
        delta_e_kcal_mol=delta_e * HARTREE_TO_KCAL_MOL,
        charges_vacuum=charges_vacuum,
        charges_field=charges_field,
        n_pocket_charges=len(point_charges),
    )


@dataclass
class _XyzAtom:
    symbol: str
    x: float
    y: float
    z: float


def _read_xyz_atoms(xyz_path: str | Path) -> list[_XyzAtom]:
    with open(xyz_path) as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    atoms = []
    for line in lines[2:2 + n]:
        parts = line.split()
        atoms.append(_XyzAtom(parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return atoms
