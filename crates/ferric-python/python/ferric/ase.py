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
(closed-shell Kohn-Sham, ``run_dft``), ``"uks"`` and ``"roks"`` (open-shell
Kohn-Sham, ``run_uks`` / ``run_roks``; ``"dft"`` is UKS for a multiplicity
above 1), ``"rimp2"`` and ``"oo-rimp2"``
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

``charge`` and ``multiplicity`` default to ``None``: the charge is then
``round(atoms.get_initial_charges().sum())`` and the multiplicity
``abs(round(atoms.get_initial_magnetic_moments().sum())) + 1``, so atoms
without initial charges or magnetic moments give the neutral singlet, as
before. A fractional initial-charge sum, non-collinear (N, 3) magnetic moments
or a magnetic-moment sum that is not an integer (within 1e-3) raise
``CalculatorSetupError``. An explicit value always wins.

``dipole`` (e*Angstrom, about the coordinate origin; origin-independent for a
neutral molecule) is available for the molecular SCF and Kohn-Sham methods
(rhf, uhf, rohf, rks, uks, roks): the SCF density contracted with
``AoIntegrals.dipole`` plus the nuclear term ``sum_A Z_A R_A`` (ECP-effective
``Z_A``, ghosts zero). ``charges`` (per-atom partial charges, e) is available
for the closed-shell rhf and rks, by the ``charge_scheme=`` parameter: ``"mulliken"``
(default), ``"lowdin"`` or ``"hirshfeld"`` (``ferric.mulliken_charges`` /
``lowdin_charges`` / ``hirshfeld_charges``). Both are computed from the SCF
already run for the energy when one is cached. Periodic cells, the RI-MP2
methods (no relaxed density is exposed) and open-shell ``charges`` raise
``PropertyNotImplementedError``. ``charge_scheme`` is validated when the
calculator is built and in ``set()``. Any ``set()`` discards the cached SCF.
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

_MOLECULAR = (
    "hf",
    "rhf",
    "uhf",
    "rohf",
    "dft",
    "rks",
    "uks",
    "roks",
    "rimp2",
    "oo-rimp2",
)
_PERIODIC = ("hf", "rhf", "uhf", "rohf", "dft", "rks", "uks", "roks")
_KS = ("dft", "rks", "uks", "roks")
# Resolved methods whose result carries the SCF density (dipole, charges).
_DENSITY = ("rhf", "uhf", "rohf", "rks", "uks", "roks")
_CHARGE_SCHEMES = ("mulliken", "lowdin", "hirshfeld")
_INT_TOL = 1e-3


def _check_scheme(kwargs):
    """Refuse an unknown ``charge_scheme`` as early as it is given."""
    if "charge_scheme" in kwargs:
        scheme = str(kwargs["charge_scheme"]).lower()
        if scheme not in _CHARGE_SCHEMES:
            raise CalculatorSetupError(
                f"FerricCalculator: charge_scheme={kwargs['charge_scheme']!r}; choose one of "
                f"{list(_CHARGE_SCHEMES)}"
            )


class FerricCalculator(Calculator):
    """ASE calculator running ferric. See the module docstring."""

    implemented_properties = [
        "energy",
        "free_energy",
        "forces",
        "stress",
        "dipole",
        "charges",
    ]
    # Every parameter changes the energy surface, so any change in set() resets.
    discard_results_on_any_change = True
    default_parameters = {
        "method": "hf",
        "basis": "sto-3g",
        "functional": None,
        "auxbasis": None,
        "charge": None,
        "multiplicity": None,
        "charge_scheme": "mulliken",
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
        _check_scheme(kwargs)
        Calculator.__init__(self, **kwargs)
        self.ferric_result = None

    def set(self, **kwargs):
        unknown = sorted(set(kwargs) - set(self.default_parameters))
        if unknown:
            raise CalculatorSetupError(
                f"FerricCalculator: unknown parameter(s) {unknown}"
            )
        _check_scheme(kwargs)
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
        want_dipole = "dipole" in properties
        want_charges = "charges" in properties
        if want_stress and not periodic:
            raise PropertyNotImplementedError(
                "FerricCalculator: stress needs a 3-D periodic cell (atoms.pbc all True)"
            )
        p = self.parameters
        charge, multiplicity = self._charge_multiplicity(atoms)
        if want_dipole or want_charges:
            method = self._method(_PERIODIC if periodic else _MOLECULAR)
            self._check_density_property(periodic, method, want_charges)
        mol = ferric.Molecule.from_arrays(
            [int(z) for z in atoms.numbers],
            np.asarray(atoms.positions, dtype=float),
            units="angstrom",
            charge=charge,
            multiplicity=multiplicity,
        )
        basis = (
            p.basis
            if isinstance(p.basis, ferric.BasisSet)
            else ferric.BasisSet.bundled(p.basis)
        )
        # A dipole or charges request after an energy/forces call on the same
        # atoms and parameters reuses that SCF (ASE resets the calculator, and
        # with it ferric_result, whenever either changes).
        rerun = (
            self.ferric_result is None
            or system_changes
            or "energy" not in self.results
            or (want_forces and "forces" not in self.results)
            or (want_stress and "stress" not in self.results)
        )
        if rerun:
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
        result = self.ferric_result
        if want_forces and "forces" not in self.results:
            gradient = result.gradient()
            if gradient is None:
                raise RuntimeError(
                    "FerricCalculator: the ferric result carries no gradient"
                )
            self.results["forces"] = -np.asarray(gradient) * (Hartree / Bohr)
        if want_stress and "stress" not in self.results:
            sigma = np.asarray(result.stress())
            sigma = 0.5 * (sigma + sigma.T) * (Hartree / Bohr**3)
            self.results["stress"] = sigma[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
        if want_dipole and "dipole" not in self.results:
            self.results["dipole"] = _dipole(mol, basis, result) * Bohr
        if want_charges and "charges" not in self.results:
            self.results["charges"] = _charges(mol, basis, result, p.charge_scheme)

    def _charge_multiplicity(self, atoms):
        """Explicit ``charge``/``multiplicity`` win; ``None`` falls back to
        the atoms' initial charges / magnetic moments (0 / 1 without them)."""
        p = self.parameters
        if p.charge is None:
            total = float(np.sum(atoms.get_initial_charges()))
            if abs(total - round(total)) > _INT_TOL:
                raise CalculatorSetupError(
                    f"FerricCalculator: the atoms' initial charges sum to {total:g}, not an "
                    "integer; pass charge= explicitly"
                )
            charge = round(total)
        else:
            charge = int(p.charge)
        if p.multiplicity is None:
            magmoms = np.asarray(atoms.get_initial_magnetic_moments())
            if magmoms.ndim != 1:
                raise CalculatorSetupError(
                    "FerricCalculator: non-collinear initial magnetic moments (shape "
                    f"{magmoms.shape}) have no multiplicity; pass multiplicity= explicitly"
                )
            total = float(np.sum(magmoms))
            if abs(total - round(total)) > _INT_TOL:
                raise CalculatorSetupError(
                    f"FerricCalculator: the atoms' initial magnetic moments sum to {total:g}, "
                    "not an integer; pass multiplicity= explicitly"
                )
            multiplicity = abs(round(total)) + 1
        else:
            multiplicity = int(p.multiplicity)
        return charge, multiplicity

    def _check_density_property(self, periodic, method, want_charges):
        if periodic:
            raise PropertyNotImplementedError(
                "FerricCalculator: dipole and charges are molecular-only (atoms.pbc all False)"
            )
        if method not in _DENSITY:
            raise PropertyNotImplementedError(
                f"FerricCalculator: dipole and charges need an SCF density; method "
                f"{method!r} exposes no relaxed density"
            )
        if want_charges:
            if method not in ("rhf", "rks"):
                raise PropertyNotImplementedError(
                    f"FerricCalculator: charges are closed-shell only (rhf, rks); method "
                    f"resolves to {method!r}"
                )

    def _method(self, choices):
        p = self.parameters
        method = str(p.method).lower()
        if method not in choices:
            raise CalculatorSetupError(
                f"FerricCalculator: method {p.method!r} is not available here; choose one of "
                f"{list(choices)}"
            )
        closed = self._charge_multiplicity(self.atoms)[1] == 1
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
        if method in ("rhf", "uhf", "rohf", "rks", "uks", "roks"):
            # Energies and forces must come from one surface: an unset RI-J
            # precision is f64 here, as on every ferric geometry entry point.
            options.setdefault("ri_j_precision", "f64")
        if method in ("rhf", "uhf", "rohf"):
            result = getattr(ferric, f"run_{method}")(mol, basis, **options)
            return result, result.energy
        if method in _KS:
            if p.functional is None:
                raise CalculatorSetupError(
                    f"FerricCalculator: method {p.method!r} needs functional="
                )
            if method == "rks":
                result = ferric.run_dft(mol, basis, functional=p.functional, **options)
                return result, result.total_energy
            driver = getattr(ferric, f"run_{method}")
            result = driver(mol, basis, p.functional, **options)
            return result, result.energy
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


def _density(result):
    """Total AO density of a molecular SCF / Kohn-Sham result."""
    try:
        if hasattr(result, "density_alpha"):
            return np.asarray(result.density_alpha()) + np.asarray(
                result.density_beta()
            )
        return np.asarray(result.density())
    except NotImplementedError as exc:  # e.g. a symmetry-frame result
        raise PropertyNotImplementedError(f"FerricCalculator: {exc}") from exc


def _dipole(mol, basis, result):
    """Dipole moment (3,) in e*Bohr about the coordinate origin."""
    density = _density(result)
    ints = ferric.AoIntegrals(mol, basis)
    electronic = -np.einsum("xij,ij->x", np.asarray(ints.dipole()), density)
    z_eff = np.array(
        [
            0.0 if ghost else float(z - basis.ecp_core_electrons(z))
            for z, ghost in zip(mol.atomic_numbers(), mol.is_ghost())
        ]
    )
    nuclear = z_eff @ np.asarray(mol.coords_bohr())
    return electronic + nuclear


def _charges(mol, basis, result, scheme):
    fn = getattr(ferric, f"{str(scheme).lower()}_charges")
    try:
        return np.asarray(fn(mol, basis, result), dtype=float)
    except NotImplementedError as exc:
        raise PropertyNotImplementedError(f"FerricCalculator: {exc}") from exc
