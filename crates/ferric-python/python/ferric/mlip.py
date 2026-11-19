"""Training-data export for machine-learned interatomic potentials.

Writes geometries, energies and forces as extended XYZ and as an ASE
database, from ferric's own result types rather than a recomputation:

    import ferric.qcschema as qcs
    from ferric.mlip import write_asedb, write_extxyz

    results = [qcs.run({..., "driver": "gradient", ...}) for geometry in geometries]
    write_extxyz("train.xyz", results)
    write_asedb("train.db", results)

Inputs (one or an iterable of them; mixing is fine):

* a QCSchema ``AtomicResult`` dict from ``ferric.qcschema.run`` with
  ``driver="gradient"`` (a failed result, an energy-only result, or a result
  with ghost atoms raises ``ValueError``: it has no usable training label);
* an ASE ``Atoms`` whose calculator is a ``ferric.ase.FerricCalculator`` with
  energy and forces computed;
* a ``Frame``, which is what the two above become (``frames()`` returns them).

Units are stated, not inherited. Every frame is in **positions Angstrom,
energy eV, forces eV/Angstrom** whatever the input unit was. QCSchema is
Bohr / Hartree / Hartree per Bohr; the conversion uses ``ferric.units``
(``HARTREE_TO_EV`` and ``BOHR_TO_ANGSTROM``): ``forces = -gradient *
HARTREE_TO_EV / BOHR_TO_ANGSTROM``. The extended XYZ comment line and the
database row both carry ``energy_unit``, ``forces_unit`` and
``positions_unit``.

Per-frame metadata (``Frame.info``; the extxyz comment line and the
database's key-value pairs; every key is flat and prefixed so it cannot
collide with ASE's reserved database keys):

* ``qc_method``, ``qc_basis``: ``model.method`` and ``model.basis``;
* ``qc_functional``: the XC functional (Kohn-Sham methods only);
* ``qc_dispersion``: the ``dispersion`` keyword, when one was set;
* ``qc_charge``, ``qc_multiplicity``;
* ``ferric_version``, ``ferric_git_sha``, ``ferric_dirty``: the version and
  build stamp of the ferric that produced the result (from the result's
  ``provenance`` when it has one, else the running ``ferric.__build__``;
  ``ferric_git_sha`` is ``"unknown"`` and ``ferric_dirty`` absent when the
  build carries no stamp);
* ``energy_unit``, ``forces_unit``, ``positions_unit``.

ASE is optional. Building frames needs only numpy; ``write_extxyz``,
``write_asedb`` and ``to_atoms`` import ASE when called and raise
``ImportError`` naming the install when it is absent.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import ferric

from . import qcschema
from .units import BOHR_TO_ANGSTROM, HARTREE_TO_EV

__all__ = [
    "ENERGY_UNIT",
    "FORCES_UNIT",
    "POSITIONS_UNIT",
    "Frame",
    "frames",
    "to_atoms",
    "write_asedb",
    "write_extxyz",
]

ENERGY_UNIT = "eV"
FORCES_UNIT = "eV/Angstrom"
POSITIONS_UNIT = "Angstrom"

_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM


@dataclass(frozen=True)
class Frame:
    """One labelled geometry: Angstrom, eV, eV/Angstrom.

    ``positions`` and ``forces`` are (N, 3) float arrays, ``energy`` a float.
    ``cell`` is a (3, 3) lattice in Angstrom for a periodic frame, else
    ``None``. ``info`` is the flat metadata dict described in the module
    docstring."""

    symbols: tuple[str, ...]
    positions: np.ndarray
    energy: float
    forces: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)
    cell: np.ndarray | None = None


def frames(results: Any) -> list[Frame]:
    """Normalise ``results`` (one input or an iterable of inputs; see the
    module docstring) to a list of ``Frame``. Needs no ASE unless an ASE
    ``Atoms`` is among the inputs."""
    if isinstance(results, (dict, Frame)) or hasattr(results, "get_potential_energy"):
        items: Iterable[Any] = [results]
    else:
        items = results
    out = []
    for i, item in enumerate(items):
        try:
            out.append(_frame(item))
        except ValueError as err:
            raise ValueError(f"result {i}: {err}") from err
    if not out:
        raise ValueError("no results to export")
    return out


def to_atoms(results: Any) -> list:
    """ASE ``Atoms`` (with a ``SinglePointCalculator`` holding energy and
    forces, and ``info`` set) for each frame. Needs ASE."""
    _require_ase()
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    atoms_list = []
    for frame in frames(results):
        atoms = Atoms(
            symbols=list(frame.symbols),
            positions=frame.positions,
            cell=frame.cell,
            pbc=frame.cell is not None,
        )
        atoms.calc = SinglePointCalculator(
            atoms, energy=frame.energy, forces=frame.forces
        )
        atoms.info.update(frame.info)
        atoms_list.append(atoms)
    return atoms_list


def write_extxyz(path: Any, results: Any) -> int:
    """Write the frames as extended XYZ (energy and forces as ASE's standard
    ``energy`` / ``forces`` properties, metadata on the comment line).
    Returns the number of frames. Needs ASE."""
    _require_ase()
    from ase.io import write

    atoms_list = to_atoms(results)
    write(str(path), atoms_list, format="extxyz")
    return len(atoms_list)


def write_asedb(path: Any, results: Any) -> int:
    """Append the frames to an ASE database (``.db`` SQLite or ``.json``);
    metadata goes in as key-value pairs, so it is queryable
    (``ase db train.db qc_method=hf``). Returns the number of frames. Needs
    ASE."""
    _require_ase()
    from ase.db import connect

    atoms_list = to_atoms(results)
    with connect(str(path)) as db:
        for atoms in atoms_list:
            db.write(atoms, key_value_pairs=dict(atoms.info))
    return len(atoms_list)


def _require_ase() -> None:
    try:
        import ase  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ferric.mlip writes through the Atomic Simulation Environment: pip install ase"
        ) from exc


def _frame(item: Any) -> Frame:
    if isinstance(item, Frame):
        return item
    if isinstance(item, dict):
        return _from_qcschema(item)
    if hasattr(item, "get_potential_energy"):
        return _from_atoms(item)
    raise ValueError(
        f"unsupported input {type(item).__name__}: expected a QCSchema AtomicResult "
        "dict, an ASE Atoms with a FerricCalculator, or a Frame"
    )


def _from_qcschema(out: dict[str, Any]) -> Frame:
    if not out.get("success", False):
        err = (out.get("error") or {}).get("error_message", "unknown failure")
        raise ValueError(f"the calculation failed: {err}")
    if out.get("driver") != "gradient":
        raise ValueError(
            f"driver {out.get('driver')!r} has no forces; run ferric.qcschema.run "
            "with driver='gradient'"
        )
    mol = out["molecule"]
    symbols = tuple(str(s) for s in mol["symbols"])
    if mol.get("real") is not None and not all(mol["real"]):
        raise ValueError("a molecule with ghost atoms is not a training label")
    n = len(symbols)
    positions = (
        np.asarray(mol["geometry"], dtype=float).reshape(n, 3) * BOHR_TO_ANGSTROM
    )
    gradient = np.asarray(out["return_result"], dtype=float).reshape(n, 3)
    forces = -gradient * _HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
    energy = float(out["properties"]["return_energy"]) * HARTREE_TO_EV
    model = out["model"]
    method = str(model["method"]).lower()
    keywords = out.get("keywords") or {}
    info = _level(
        method=method,
        basis=str(model["basis"]),
        functional=method if qcschema._family(method) == "dft" else None,
        dispersion=keywords.get("dispersion"),
        charge=int(mol.get("molecular_charge", 0)),
        multiplicity=_multiplicity(mol, symbols),
    )
    info.update(_stamp(out.get("provenance")))
    return Frame(symbols, positions, energy, forces, info)


def _multiplicity(mol: dict[str, Any], symbols: tuple[str, ...]) -> int:
    if mol.get("molecular_multiplicity") is not None:
        return int(mol["molecular_multiplicity"])
    from ._util import default_multiplicity

    return default_multiplicity(symbols, int(mol.get("molecular_charge", 0)))


def _from_atoms(atoms: Any) -> Frame:
    from .ase import FerricCalculator

    calc = atoms.calc
    if not isinstance(calc, FerricCalculator):
        raise ValueError(
            "the Atoms' calculator is not a FerricCalculator, so its level of theory "
            "cannot be recorded"
        )
    # ASE returns eV and eV/Angstrom already; both raise if not computed.
    energy = float(atoms.get_potential_energy())
    forces = np.array(atoms.get_forces(), dtype=float)
    p = calc.parameters
    charge, multiplicity = calc._charge_multiplicity(atoms)
    options = p.options or {}
    method = str(p.method).lower()
    basis = p.basis if isinstance(p.basis, str) else "custom"
    info = _level(
        method=method,
        basis=basis,
        functional=None if p.functional is None else str(p.functional),
        dispersion=options.get("dispersion"),
        charge=charge,
        multiplicity=multiplicity,
    )
    info.update(_stamp(None))
    periodic = bool(np.asarray(atoms.pbc).any())
    return Frame(
        tuple(atoms.get_chemical_symbols()),
        np.array(atoms.positions, dtype=float),
        energy,
        forces,
        info,
        np.array(atoms.cell, dtype=float) if periodic else None,
    )


def _level(
    *, method, basis, functional, dispersion, charge, multiplicity
) -> dict[str, Any]:
    info: dict[str, Any] = {"qc_method": method, "qc_basis": basis}
    if functional is not None:
        info["qc_functional"] = functional
    if dispersion is not None:
        info["qc_dispersion"] = str(dispersion)
    info["qc_charge"] = int(charge)
    info["qc_multiplicity"] = int(multiplicity)
    info["energy_unit"] = ENERGY_UNIT
    info["forces_unit"] = FORCES_UNIT
    info["positions_unit"] = POSITIONS_UNIT
    return info


def _stamp(provenance: dict[str, Any] | None) -> dict[str, Any]:
    """ferric version and build stamp: the result's own provenance when it was
    made by ferric, else the running build. An unknown stamp is never guessed."""
    version = ferric.__version__
    sha: Any = None
    dirty: Any = None
    if provenance and provenance.get("creator") == "ferric":
        version = provenance.get("version", version)
        build = provenance.get("build") or {}
        sha, dirty = build.get("commit"), build.get("dirty")
    else:
        build = getattr(ferric, "__build__", None)
        if build:
            sha, dirty = build.get("git_sha"), build.get("dirty")
    info: dict[str, Any] = {
        "ferric_version": str(version),
        "ferric_git_sha": "unknown" if not sha else str(sha),
    }
    if dirty is not None:
        info["ferric_dirty"] = bool(dirty)
    return info
