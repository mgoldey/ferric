"""ASE calculator backed by ferric.

``FerricCalculator`` gives ASE energies, forces and, for 3-D periodic cells,
stress from ferric's analytic derivatives:

    from ase.build import molecule
    from ferric.ase import FerricCalculator

    atoms = molecule("H2O")
    atoms.calc = FerricCalculator(method="rhf", basis="6-31g")
    atoms.get_forces()                    # eV/Angstrom
    atoms.calc.results["certificate"]     # ferric's validation certificate

Units are ASE's: positions and cell in Angstrom, energy in eV, forces in
eV/Angstrom, stress in eV/Angstrom^3 (Voigt order xx, yy, zz, yz, xz, xy,
sign convention ``sigma = (1/V) dE/d(strain)``). ferric works in Hartree and
Bohr; the conversion uses ``ase.units.Hartree`` and ``ase.units.Bohr``.

Molecular methods (``atoms.pbc`` all False): ``"hf"`` (RHF for a singlet,
UHF otherwise), ``"rhf"``, ``"uhf"``, ``"rohf"``, ``"dft"`` / ``"rks"``
(closed-shell Kohn-Sham, ``run_dft``), ``"rimp2"`` and ``"oo-rimp2"``
(``auxbasis`` required). Periodic methods (``atoms.pbc`` all True):
``"hf"``, ``"rhf"``, ``"uhf"``, ``"rohf"``, ``"dft"``, ``"rks"``, ``"uks"``,
``"roks"`` at the Gamma point (``kpts=None``) or on a ``kpts=(n1, n2, n3)``
mesh, through ``run_<method>_gamma`` / ``run_<method>_kpts``. Mixed
periodicity raises ``CalculatorSetupError`` (ferric's periodic code is 3-D
only).

``options`` is passed unchanged to the ferric driver as keyword arguments
(``density_conv``, ``jk``, ``omega``, ``solvent`` ...), so every driver
refusal applies. A property is computed only when ASE asks for it; asking for
``forces`` adds ``with_gradient=True`` and ``stress`` adds ``with_stress=True``
to the same call. The molecular SCF and Kohn-Sham drivers run with
``ri_j_precision="f64"`` unless ``options`` sets it, so every energy and force
comes from the same surface (the f32 RI-J surface is jagged between nearby
geometries). Results are cached by ASE's ``Calculator`` and cleared when
the atoms change or ``set()`` changes a parameter. ``calc.ferric_result``
holds the last ferric result object.
"""

from __future__ import annotations

import numpy as np

try:
    from ase.calculators.calculator import (
        Calculator,
        CalculatorSetupError,
        PropertyNotImplementedError,
        all_changes,
    )
    from ase.units import Bohr, Hartree
except ImportError as exc:  # pragma: no cover - exercised only without ASE
    raise ImportError(
        "ferric.ase needs the Atomic Simulation Environment: pip install ase"
    ) from exc

import ferric

__all__ = ["FerricCalculator"]

_MOLECULAR = ("hf", "rhf", "uhf", "rohf", "dft", "rks", "rimp2", "oo-rimp2")
_PERIODIC = ("hf", "rhf", "uhf", "rohf", "dft", "rks", "uks", "roks")
_KS = ("dft", "rks", "uks", "roks")


class FerricCalculator(Calculator):
    """ASE calculator running ferric. See the module docstring."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]
    # Every parameter changes the energy surface, so any change in set() resets.
    discard_results_on_any_change = True
    default_parameters = {
        "method": "hf",
        "basis": "sto-3g",
        "functional": None,
        "auxbasis": None,
        "charge": 0,
        "multiplicity": 1,
        "kpts": None,
        "options": None,
    }

    def __init__(self, **kwargs):
        unknown = sorted(
            set(kwargs)
            - set(self.default_parameters)
            - {"restart", "ignore_bad_restart_file", "label", "atoms", "directory"}
        )
        if unknown:
            raise CalculatorSetupError(
                f"FerricCalculator: unknown parameter(s) {unknown}; driver keywords go in "
                "options={...}"
            )
        Calculator.__init__(self, **kwargs)
        self.ferric_result = None

    def set(self, **kwargs):
        unknown = sorted(set(kwargs) - set(self.default_parameters))
        if unknown:
            raise CalculatorSetupError(
                f"FerricCalculator: unknown parameter(s) {unknown}"
            )
        return Calculator.set(self, **kwargs)

    def reset(self):
        Calculator.reset(self)
        self.ferric_result = None

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        atoms = self.atoms
        periodic = _periodicity(atoms)
        want_forces = "forces" in properties
        want_stress = "stress" in properties
        if want_stress and not periodic:
            raise PropertyNotImplementedError(
                "FerricCalculator: stress needs a 3-D periodic cell (atoms.pbc all True)"
            )
        p = self.parameters
        mol = ferric.Molecule.from_arrays(
            [int(z) for z in atoms.numbers],
            np.asarray(atoms.positions, dtype=float),
            units="angstrom",
            charge=int(p.charge),
            multiplicity=int(p.multiplicity),
        )
        basis = (
            p.basis
            if isinstance(p.basis, ferric.BasisSet)
            else ferric.BasisSet.bundled(p.basis)
        )
        options = dict(p.options or {})
        if periodic:
            result, energy = self._run_periodic(
                mol, atoms, basis, options, want_forces, want_stress
            )
        else:
            result, energy = self._run_molecular(mol, basis, options, want_forces)
        self.ferric_result = result
        self.results = {
            "energy": energy * Hartree,
            "free_energy": energy * Hartree,
            "certificate": result.certificate,
        }
        if want_forces:
            gradient = result.gradient()
            if gradient is None:
                raise RuntimeError(
                    "FerricCalculator: the ferric result carries no gradient"
                )
            self.results["forces"] = -np.asarray(gradient) * (Hartree / Bohr)
        if want_stress:
            sigma = np.asarray(result.stress())
            sigma = 0.5 * (sigma + sigma.T) * (Hartree / Bohr**3)
            self.results["stress"] = sigma[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]

    def _method(self, choices):
        p = self.parameters
        method = str(p.method).lower()
        if method not in choices:
            raise CalculatorSetupError(
                f"FerricCalculator: method {p.method!r} is not available here; choose one of "
                f"{list(choices)}"
            )
        closed = int(p.multiplicity) == 1
        if method == "hf":
            return "rhf" if closed else "uhf"
        if method == "dft":
            return "rks" if closed else "uks"
        return method

    def _run_molecular(self, mol, basis, options, want_forces):
        p = self.parameters
        method = self._method(_MOLECULAR)
        if p.kpts is not None:
            raise CalculatorSetupError("FerricCalculator: kpts needs a periodic cell")
        if want_forces:
            options["with_gradient"] = True
        if method in ("rhf", "uhf", "rohf", "rks"):
            # Energies and forces must come from one surface: an unset RI-J
            # precision is f64 here, as on every ferric geometry entry point.
            options.setdefault("ri_j_precision", "f64")
        if method in ("rhf", "uhf", "rohf"):
            result = getattr(ferric, f"run_{method}")(mol, basis, **options)
            return result, result.energy
        if method == "uks":
            raise CalculatorSetupError(
                "FerricCalculator: molecular Kohn-Sham is closed-shell only (run_dft); use "
                "method='uhf' or a singlet"
            )
        if method == "rks":
            if p.functional is None:
                raise CalculatorSetupError(
                    "FerricCalculator: method 'dft' needs functional="
                )
            result = ferric.run_dft(mol, basis, functional=p.functional, **options)
            return result, result.total_energy
        if p.auxbasis is None:
            raise CalculatorSetupError(
                f"FerricCalculator: method {p.method!r} needs auxbasis="
            )
        aux = (
            p.auxbasis
            if isinstance(p.auxbasis, ferric.BasisSet)
            else ferric.BasisSet.bundled(p.auxbasis)
        )
        driver = ferric.run_rimp2 if method == "rimp2" else ferric.run_oo_rimp2
        result = driver(mol, basis, aux, **options)
        return result, result.total_energy

    def _run_periodic(self, mol, atoms, basis, options, want_forces, want_stress):
        p = self.parameters
        method = self._method(_PERIODIC)
        lattice = [[float(v) for v in row] for row in np.asarray(atoms.cell)]
        if method in _KS:
            if p.functional is None:
                raise CalculatorSetupError(
                    f"FerricCalculator: method {p.method!r} needs functional="
                )
            options["functional"] = p.functional
        if p.auxbasis is not None:
            options["auxbasis"] = p.auxbasis
        if want_forces:
            options["with_gradient"] = True
        if want_stress:
            options["with_stress"] = True
        if p.kpts is None:
            driver = getattr(ferric, f"run_{method}_gamma")
        else:
            options["mesh"] = tuple(int(n) for n in p.kpts)
            driver = getattr(ferric, f"run_{method}_kpts")
        result = driver(mol, lattice, basis, **options)
        return result, result.energy


def _periodicity(atoms):
    pbc = np.asarray(atoms.pbc, dtype=bool)
    if pbc.all():
        return True
    if not pbc.any():
        return False
    raise CalculatorSetupError(
        f"FerricCalculator: ferric's periodic code is 3-D only; atoms.pbc is {pbc.tolist()}"
    )
