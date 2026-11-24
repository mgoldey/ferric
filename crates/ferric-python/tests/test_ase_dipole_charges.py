"""`ferric.ase.FerricCalculator`: dipole, partial charges and the charge /
multiplicity fallback to the atoms' initial charges and magnetic moments.

Anchors:

* Dipole against an INDEPENDENT route: the central finite difference of the
  SCF energy in a uniform external field, mu = -dE/dF (ferric's field enters
  as ``+F.r`` on the electrons plus the classical ``-F.sum_A Z_A R_A``, so the
  derivative is the full dipole about the coordinate origin). The calculator
  instead contracts the density with ``AoIntegrals.dipole`` and adds the
  nuclear term. Bar: F step h = 1e-4 a.u.; the FD truncation is h^2 |beta| / 6
  ~ 1e-8 * 10 a.u. and energy noise at energy_conv 1e-11 contributes
  1e-11 / h = 1e-7 a.u., so DIPOLE_BAR = 2e-6 a.u. (1e-6 e*Angstrom) is ~10x.
* Water/STO-3G RHF |mu| sits in the literature RHF/STO-3G range (Szabo &
  Ostlund, Table 3.13: 0.679 a.u. = 1.73 D at the experimental geometry);
  this geometry is slightly distorted, so the check is loose (0.6-0.75 a.u.).
* A neutral molecule's dipole is origin-independent: translating the atoms
  leaves it unchanged.
* Partial charges of every scheme sum to the total charge.
* ``charge=None``/``multiplicity=None`` (the defaults) take the atoms'
  initial charges/magnetic moments, and give exactly today's numbers for
  atoms without them; explicit values win.
* Unsupported cases raise ``PropertyNotImplementedError``.
"""

from __future__ import annotations

import numpy as np
import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")
ase = pytest.importorskip("ase", reason="ASE is an optional dependency")

from ase import Atoms  # noqa: E402
from ase.calculators.calculator import (  # noqa: E402
    CalculatorSetupError,
    PropertyNotImplementedError,
)
from ase.units import Bohr  # noqa: E402

try:
    from ferric.ase import FerricCalculator
except ModuleNotFoundError as err:  # the extension without the python package layer
    pytest.skip(f"ferric.ase is not importable: {err}", allow_module_level=True)

TIGHT = {"energy_conv": 1e-11, "density_conv": 1e-9}
WATER_POS = [(0.01, 0.02, 0.117), (0.03, 0.757, -0.467), (-0.02, -0.75, -0.45)]
H3O_POS = [
    (0.0, 0.0, 0.1),
    (0.0, 0.95, -0.2),
    (0.82, -0.47, -0.2),
    (-0.82, -0.47, -0.2),
]
FIELD_STEP = 1e-4
DIPOLE_BAR = 2e-6  # a.u.


def _atoms(symbols, positions, **calc):
    atoms = Atoms(symbols, positions=positions)
    atoms.calc = FerricCalculator(**calc)
    return atoms


def _fd_dipole(run, mol, bs, **kw):
    """-dE/dF by central differences, a.u. (e*Bohr)."""
    mu = np.zeros(3)
    for axis in range(3):
        e = []
        for sign in (1.0, -1.0):
            field = [0.0, 0.0, 0.0]
            field[axis] = sign * FIELD_STEP
            r = run(mol, bs, external_field=tuple(field), **kw)
            e.append(r.energy if hasattr(r, "energy") else r.total_energy)
        mu[axis] = -(e[0] - e[1]) / (2 * FIELD_STEP)
    return mu


@pytest.mark.parametrize(
    "case",
    [
        ("rhf", "OH2", WATER_POS, 0, 1),
        ("uhf", "OH2", WATER_POS, 1, 2),
        ("rks", "OH2", WATER_POS, 0, 1),
    ],
    ids=["rhf-water", "uhf-water-cation", "rks-pbe-water"],
)
def test_dipole_matches_finite_field_derivative(case):
    method, symbols, pos, charge, mult = case
    calc = {"method": method, "basis": "sto-3g", "options": TIGHT}
    calc.update(charge=charge, multiplicity=mult)
    kw = {"ri_j_precision": "f64", **TIGHT}
    if method == "rks":
        calc["functional"] = "PBE"
        kw["functional"] = "PBE"
    atoms = _atoms(symbols, pos, **calc)
    dipole = atoms.get_dipole_moment()  # e*Angstrom
    mol = ferric.Molecule.from_arrays(
        list(atoms.get_chemical_symbols()), pos, charge=charge, multiplicity=mult
    )
    run = {"rhf": ferric.run_rhf, "uhf": ferric.run_uhf, "rks": ferric.run_dft}[method]
    ref = _fd_dipole(run, mol, ferric.BasisSet.bundled("sto-3g"), **kw)
    miss = np.abs(dipole / Bohr - ref).max()
    print(f"{method}: dipole {dipole / Bohr} a.u., finite field {ref}, miss {miss:.2e}")
    assert np.linalg.norm(ref) > 0.1, "vanishing dipole, the check is vacuous"
    assert miss < DIPOLE_BAR


def test_water_sto3g_dipole_is_in_the_literature_range():
    atoms = _atoms("OH2", WATER_POS, method="rhf", basis="sto-3g")
    mu = np.linalg.norm(atoms.get_dipole_moment()) / Bohr
    assert 0.6 < mu < 0.75, mu


def test_neutral_dipole_is_origin_independent():
    atoms = _atoms("OH2", WATER_POS, method="rhf", basis="sto-3g", options=TIGHT)
    d0 = atoms.get_dipole_moment()
    atoms.translate((3.0, -2.0, 5.0))
    d1 = atoms.get_dipole_moment()
    assert np.abs(d1 - d0).max() < 1e-6


@pytest.mark.parametrize("scheme", ["mulliken", "lowdin", "hirshfeld"])
@pytest.mark.parametrize(
    "case",
    [
        ("OH2", WATER_POS, {"method": "rhf"}, 0),
        ("OH3", H3O_POS, {"method": "hf", "charge": 1}, 1),
        ("OH2", WATER_POS, {"method": "dft", "functional": "PBE"}, 0),
    ],
    ids=["rhf-water", "rhf-hydronium", "rks-water"],
)
def test_charges_sum_to_the_total_charge(case, scheme):
    symbols, pos, calc, total = case
    atoms = _atoms(symbols, pos, basis="sto-3g", charge_scheme=scheme, **calc)
    q = atoms.get_charges()
    assert q.shape == (len(atoms),)
    # Mulliken/Lowdin partition tr(DS) exactly; Hirshfeld integrates on a grid.
    tol = 1e-3 if scheme == "hirshfeld" else 1e-8
    assert abs(q.sum() - total) < tol, q
    if total == 0:
        assert q[0] < 0  # neutral water: oxygen carries the negative charge


def test_charges_scheme_routes_to_the_ferric_function():
    atoms = _atoms(
        "OH2", WATER_POS, method="rhf", basis="sto-3g", charge_scheme="lowdin"
    )
    q = atoms.get_charges()
    r = atoms.calc.ferric_result
    mol = ferric.Molecule.from_arrays(["O", "H", "H"], WATER_POS)
    ref = ferric.lowdin_charges(mol, ferric.BasisSet.bundled("sto-3g"), r)
    assert np.array_equal(q, np.asarray(ref))
    mull = ferric.mulliken_charges(mol, ferric.BasisSet.bundled("sto-3g"), r)
    assert not np.allclose(q, mull)


def test_dipole_and_charges_reuse_the_cached_scf():
    atoms = _atoms("OH2", WATER_POS, method="rhf", basis="sto-3g")
    atoms.get_potential_energy()
    first = atoms.calc.ferric_result
    atoms.get_dipole_moment()
    atoms.get_charges()
    assert atoms.calc.ferric_result is first
    atoms.get_forces()  # needs a gradient: a new SCF
    assert atoms.calc.ferric_result is not first
    assert "dipole" not in atoms.calc.results
    atoms.positions[0, 2] += 0.01
    d = atoms.get_dipole_moment()
    assert "energy" in atoms.calc.results and d.shape == (3,)


def _energy(atoms, **calc):
    atoms.calc = FerricCalculator(basis="sto-3g", **calc)
    return atoms.get_potential_energy()


def test_default_charge_and_multiplicity_are_unchanged_without_initial_values():
    atoms = Atoms("OH2", positions=WATER_POS)
    assert _energy(atoms) == _energy(atoms, charge=0, multiplicity=1)
    mol = ferric.Molecule.from_arrays(["O", "H", "H"], WATER_POS)
    atoms.calc = FerricCalculator()
    atoms.get_potential_energy()
    cert = atoms.calc.results["certificate"]
    assert cert["capability"] == "rhf"
    direct = ferric.run_rhf(
        mol, ferric.BasisSet.bundled("sto-3g"), ri_j_precision="f64"
    )
    assert atoms.get_potential_energy() / ase.units.Hartree == direct.energy


def test_charge_and_multiplicity_fall_back_to_the_atoms():
    atoms = Atoms("OH2", positions=WATER_POS)
    atoms.set_initial_charges([0.6, 0.2, 0.2])
    atoms.set_initial_magnetic_moments([1.0, 0.0, 0.0])
    fallback = _energy(atoms)
    assert atoms.calc.ferric_result.certificate["capability"] == "uhf"
    assert fallback == _energy(atoms, charge=1, multiplicity=2)
    # Explicit parameters win over the atoms' initial values.
    explicit = _energy(atoms, charge=1, multiplicity=4)
    assert explicit != fallback
    # Negative net moment: the multiplicity is |2S| + 1.
    atoms.set_initial_magnetic_moments([-1.0, 0.0, 0.0])
    assert _energy(atoms) == fallback
    # Changing initial charges invalidates the cached result.
    atoms.calc = FerricCalculator(basis="sto-3g")
    atoms.set_initial_magnetic_moments(None)
    atoms.set_initial_charges([1.0, 0.5, 0.5])
    e_dication = atoms.get_potential_energy()
    atoms.set_initial_charges(None)
    assert atoms.get_potential_energy() < e_dication


def test_fractional_initial_charge_sum_is_refused():
    atoms = Atoms("OH2", positions=WATER_POS)
    atoms.set_initial_charges([0.3, 0.1, 0.0])
    atoms.calc = FerricCalculator(basis="sto-3g")
    with pytest.raises(CalculatorSetupError, match="0.4"):
        atoms.get_potential_energy()
    # An explicit charge bypasses the fallback.
    atoms.calc = FerricCalculator(basis="sto-3g", charge=0)
    atoms.get_potential_energy()


def test_bad_initial_magnetic_moments_are_refused():
    atoms = Atoms("OH2", positions=WATER_POS)
    atoms.set_initial_magnetic_moments(np.array([[0, 0, 1.0], [0, 0, 0], [0, 0, 0]]))
    atoms.calc = FerricCalculator(basis="sto-3g")
    with pytest.raises(CalculatorSetupError, match="non-collinear"):
        atoms.get_potential_energy()
    atoms = Atoms("OH2", positions=WATER_POS)
    atoms.set_initial_magnetic_moments([0.5, 0.0, 0.0])
    atoms.calc = FerricCalculator(basis="sto-3g")
    with pytest.raises(CalculatorSetupError, match="magnetic moments sum to 0.5"):
        atoms.get_potential_energy()
    atoms.calc = FerricCalculator(basis="sto-3g", multiplicity=1)
    atoms.get_potential_energy()


def test_dipole_and_charges_refusals():
    with pytest.raises(PropertyNotImplementedError, match="closed-shell"):
        _atoms("OH2", WATER_POS, method="uhf", charge=1, multiplicity=2).get_charges()
    with pytest.raises(PropertyNotImplementedError, match="relaxed density"):
        _atoms(
            "OH2", WATER_POS, method="rimp2", auxbasis="cc-pvdz-ri"
        ).get_dipole_moment()
    with pytest.raises(CalculatorSetupError, match="charge_scheme="):
        FerricCalculator(charge_scheme="chelpg")
    calc = FerricCalculator()
    with pytest.raises(CalculatorSetupError, match="charge_scheme="):
        calc.set(charge_scheme="chelpg")
    # The old spelling (a typo for charge=) is no longer accepted.
    with pytest.raises(CalculatorSetupError, match="unknown parameter"):
        FerricCalculator(charges=-1)
    cell = Atoms(
        "H2", positions=[(0, 0, 0), (0, 0, 0.74)], cell=np.eye(3) * 4, pbc=True
    )
    cell.calc = FerricCalculator()
    with pytest.raises(PropertyNotImplementedError, match="molecular-only"):
        cell.get_dipole_moment()
    with pytest.raises(PropertyNotImplementedError, match="molecular-only"):
        cell.get_charges()


def test_charges_refuse_a_symmetry_frame_result():
    # run_rhf(symmetry=True) returns a SymmetricRhfResult, which the charge
    # functions do not accept: refused as unsupported, not a TypeError.
    from ferric.ase import _charges

    xyz = "3\n\n" + "\n".join(
        f"{s} {x} {y} {z}" for s, (x, y, z) in zip("OHH", WATER_POS)
    )
    mol = ferric.Molecule.from_xyz_string(xyz, 0, 1)
    basis = ferric.BasisSet.bundled("sto-3g")
    result = ferric.run_rhf(mol, basis, symmetry=True)
    with pytest.raises(PropertyNotImplementedError, match="symmetry-frame"):
        _charges(mol, basis, result, "mulliken")


def test_symmetry_option_end_to_end():
    # options={"symmetry": True} routes through run_rhf(symmetry=True), whose
    # SymmetricRhfResult has no certificate: the energy must still come out,
    # equal to the symmetry-free one, and charges are refused.
    plain = _atoms("OH2", WATER_POS)
    sym = _atoms("OH2", WATER_POS, options={"symmetry": True})
    assert abs(sym.get_potential_energy() - plain.get_potential_energy()) < 1e-8
    with pytest.raises(PropertyNotImplementedError, match="symmetry-frame"):
        sym.get_charges()
