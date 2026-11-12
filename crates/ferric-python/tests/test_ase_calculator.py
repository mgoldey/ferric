"""`ferric.ase.FerricCalculator`: ASE energies, forces and periodic stress.

Anchors:

* Units: energy and forces equal the ferric driver's Hartree / Hartree-per-Bohr
  numbers times ``ase.units`` factors exactly (one multiplication each).
* Forces against ASE's own central finite differences of the calculator's
  energy (``ase.calculators.fd``, eps = 1e-3 Angstrom = 1.89e-3 Bohr).
  Bar: the derivation in test_python_gradients.py (h^2 |E'''| / 6 +
  delta_E / h) at h = 1.89e-3 Bohr gives <= 6.0e-7 + 5e-9 Ha/Bohr; FORCE_BAR
  = 2e-6 Ha/Bohr (1.03e-4 eV/Angstrom) is ~3x that.
* A BFGS optimisation by ase.optimize on water reaches the minimum ferric's
  own optimizer (run_optimize) finds: the two are independent optimizers on
  the same surface.
* Periodic stress against ASE's numerical stress (strain of cell AND atoms,
  ferric's convention), on the Gamma-point H2 cell of test_pbc_gradients.py.
* Caching / reset semantics and the setup refusals.

The whole module skips without ASE (an optional dependency).
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")
ase = pytest.importorskip("ase", reason="ASE is an optional dependency")

from ase import Atoms  # noqa: E402
from ase.calculators.calculator import (  # noqa: E402
    CalculatorSetupError,
    PropertyNotImplementedError,
)
from ase.calculators.fd import (  # noqa: E402
    calculate_numerical_forces,
    calculate_numerical_stress,
)
from ase.optimize import BFGS  # noqa: E402
from ase.units import Bohr, Hartree  # noqa: E402

try:
    from ferric.ase import FerricCalculator
except ModuleNotFoundError as err:  # the extension without the python package layer
    pytest.skip(f"ferric.ase is not importable: {err}", allow_module_level=True)

FORCE_BAR = 2e-6 * Hartree / Bohr
TIGHT = {"energy_conv": 1e-10, "density_conv": 1e-9}
WATER_POS = [(0.01, 0.02, 0.117), (0.03, 0.757, -0.467), (-0.02, -0.75, -0.45)]


def _water(**calc):
    atoms = Atoms("OH2", positions=WATER_POS)
    atoms.calc = FerricCalculator(**calc)
    return atoms


def test_energy_and_forces_are_the_driver_numbers_in_ase_units():
    atoms = _water(method="rhf", basis="6-31g", options=TIGHT)
    forces = atoms.get_forces()
    energy = atoms.get_potential_energy()
    mol = ferric.Molecule.from_arrays(["O", "H", "H"], WATER_POS)
    r = ferric.run_rhf(
        mol, ferric.BasisSet.bundled("6-31g"), with_gradient=True, **TIGHT
    )
    assert energy == r.energy * Hartree
    assert np.array_equal(forces, -r.gradient() * (Hartree / Bohr))
    cert = atoms.calc.results["certificate"]
    assert cert["capability"] == "rhf" and cert == r.certificate


@pytest.mark.parametrize(
    "calc",
    [
        {"method": "rhf", "basis": "6-31g"},
        {"method": "hf", "basis": "6-31g", "charge": 1, "multiplicity": 2},
        {"method": "dft", "basis": "6-31g", "functional": "PBE"},
        {"method": "rimp2", "basis": "6-31g", "auxbasis": "cc-pvdz-ri"},
    ],
    ids=["rhf", "uhf-cation", "pbe", "rimp2"],
)
def test_forces_match_ase_numerical_forces(calc):
    options = (
        {"scf_energy_conv": 1e-10, "scf_density_conv": 1e-10}
        if calc["method"] == "rimp2"
        else TIGHT
    )
    atoms = _water(options=options, **calc)
    analytic = atoms.get_forces()
    numeric = calculate_numerical_forces(atoms, eps=1e-3)
    assert np.abs(analytic).max() > 1e-2, "trivial forces, the FD check is vacuous"
    miss = np.abs(analytic - numeric).max()
    print(f"{calc['method']}: max |F - F_FD| = {miss:.2e} eV/A")
    assert miss < FORCE_BAR


def test_ase_bfgs_finds_the_run_optimize_minimum():
    atoms = _water(method="rhf", basis="sto-3g", options=TIGHT)
    assert BFGS(atoms, logfile=None).run(fmax=1e-3, steps=100)
    assert np.abs(atoms.get_forces()).max() < 1e-3
    ref = ferric.run_optimize(
        ferric.Molecule.from_arrays(["O", "H", "H"], WATER_POS), "sto-3g", e_conv=1e-10
    )
    assert ref.converged

    def internals(p):
        p = np.asarray(p)
        r1, r2 = p[1] - p[0], p[2] - p[0]
        cos = r1 @ r2 / np.linalg.norm(r1) / np.linalg.norm(r2)
        return sorted([np.linalg.norm(r1), np.linalg.norm(r2)]), np.degrees(
            np.arccos(cos)
        )

    (ra, ta), (rb, tb) = internals(atoms.positions), internals(ref.mol().coords())
    print(f"ASE BFGS r={ra} theta={ta:.4f}; run_optimize r={rb} theta={tb:.4f}")
    # fmax 1e-3 eV/A with bond force constants ~ 50 eV/A^2 leaves ~2e-5 A.
    assert np.abs(np.subtract(ra, rb)).max() < 2e-4
    assert abs(ta - tb) < 0.05


# Gamma-point H2 in a 4 Bohr cube (test_pbc_gradients.py H2_ATOMS, Bohr) with
# the nuclear-attraction split pinned (the default depends on the volume).
H2_BOHR = [(0.3, 0.2, 0.1), (0.35, 0.12, 1.5)]
CELL_A = 4.0 * Bohr
OMEGA = 0.8 / Bohr


def test_periodic_stress_matches_ase_numerical_stress():
    atoms = Atoms(
        "H2", positions=np.asarray(H2_BOHR) * Bohr, cell=np.eye(3) * CELL_A, pbc=True
    )
    atoms.calc = FerricCalculator(
        method="rhf", basis="sto-3g", options={"omega": OMEGA, "density_conv": 1e-10}
    )
    stress = atoms.get_stress()
    numeric = calculate_numerical_stress(atoms, eps=1e-4)
    assert np.abs(stress[:3]).max() > 1e-3, "trivial stress, the FD check is vacuous"
    miss = np.abs(stress - numeric).max()
    print(f"stress (eV/A^3) {stress}; max miss vs numerical {miss:.2e}")
    # Central FD at strain d = 1e-4 on a 64 Bohr^3 cell: d^2 |E'''| / 6 / V
    # with |E'''| ~ 1 Ha is ~3e-11 Ha/Bohr^3; delta_E / (d V) with delta_E
    # ~1e-11 Ha is ~2e-9 Ha/Bohr^3. 1e-7 Ha/Bohr^3 (6.7e-4 eV/A^3) is the
    # Python periodic suites' FULL_STRESS_BAR scale with the A round trip.
    assert miss < 1e-7 * Hartree / Bohr**3
    forces = atoms.get_forces()
    assert np.abs(forces.sum(axis=0)).max() < 1e-6


def test_results_are_cached_and_reset_on_change():
    atoms = _water(method="rhf", basis="sto-3g")
    e1 = atoms.get_potential_energy()
    first = atoms.calc.ferric_result
    assert atoms.get_potential_energy() == e1 and atoms.calc.ferric_result is first
    atoms.positions[0, 2] += 0.01
    assert atoms.get_potential_energy() != e1 and atoms.calc.ferric_result is not first
    atoms.calc.set(basis="6-31g")
    assert atoms.calc.results == {} and atoms.calc.ferric_result is None
    assert atoms.get_potential_energy() < e1
    atoms.calc.reset()
    assert atoms.calc.ferric_result is None


def test_setup_refusals():
    with pytest.raises(CalculatorSetupError, match="unknown parameter"):
        FerricCalculator(method="rhf", density_conv=1e-9)
    with pytest.raises(CalculatorSetupError, match="unknown parameter"):
        FerricCalculator().set(xc="PBE")
    with pytest.raises(CalculatorSetupError, match="not available"):
        _water(method="ccsd").get_potential_energy()
    with pytest.raises(CalculatorSetupError, match="auxbasis"):
        _water(method="rimp2").get_potential_energy()
    with pytest.raises(PropertyNotImplementedError, match="periodic"):
        _water().get_stress()
    slab = Atoms(
        "H2", positions=[(0, 0, 0), (0, 0, 0.74)], cell=np.eye(3) * 5, pbc=[1, 1, 0]
    )
    slab.calc = FerricCalculator()
    with pytest.raises(CalculatorSetupError, match="3-D only"):
        slab.get_potential_energy()


def _python(code):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_import_ferric_does_not_import_ase():
    out = _python("import sys, ferric; print('ase' in sys.modules)")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_ferric_ase_without_ase_names_the_install():
    out = _python("import sys; sys.modules['ase'] = None; import ferric.ase")
    assert out.returncode != 0
    assert "pip install ase" in out.stderr
