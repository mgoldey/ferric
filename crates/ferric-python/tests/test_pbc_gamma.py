"""`run_rhf_gamma`: Gamma-point periodic RHF through the Python surface.

Stage-1 PBC design step 11 (reference/pbc/stage1-design.md): the step-5
H2/STO-3G numbers, reached from Python, plus the hard errors.

References (PySCF 2.13 AFTDF, mesh 61^3, pinned in
reference/pbc/test_prototype.py and the Rust suite
crates/ferric-pbc/tests/pbc_dense_aft_scf.rs):
    H2/STO-3G, cubic a = 4 Bohr: E = -1.658327061049 (ewald),
    -0.949002691179 (none).

Units. The binding takes Angstrom (coordinates via Molecule, lattice rows,
omega in 1/Angstrom). The references are defined in Bohr, so this file
converts Bohr -> Angstrom with the reciprocal of ferric's own factor
(1 / 0.52917721092); the binding multiplies back, which round-trips to ~1 ulp.

Basis. The pins were computed with PySCF's STO-3G digits, which are shorter
than ferric's bundled BSE copy; the difference is visible at 1e-8, so the
basis is written out here as a BSE JSON file with PySCF's digits (the same
fixture the Rust suite builds in tests/common/mod.rs::pyscf_sto3g_h).

Independent checks beyond the pins: E_ewald - E_none = -v_M * N_e / 2 (the
Madelung shift acts as -v_M on the occupied space without changing D), and
omega-independence of the energy (omega is a numerical Ewald split).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import ferric

BOHR_IN_ANGSTROM = 0.52917721092  # 1 / ferric's ANGSTROM_TO_BOHR

H2_ATOMS_BOHR = [(0.3, 0.2, 0.1), (0.3, 0.2, 1.5)]
A_BOHR = 4.0
E_EWALD = -1.658327061049
E_NONE = -0.949002691179
TOL = 1e-8

PYSCF_STO3G_H = {
    "name": "pyscf-sto-3g-H",
    "elements": {
        "1": {
            "electron_shells": [
                {
                    "function_type": "gto",
                    "angular_momentum": [0],
                    "exponents": ["3.42525091", "0.62391373", "0.1688554"],
                    "coefficients": [["0.15432897", "0.53532814", "0.44463454"]],
                }
            ]
        }
    },
}


def _xyz(atoms_bohr, symbol="H"):
    lines = [str(len(atoms_bohr)), "pbc gamma test (Angstrom)"]
    for x, y, z in atoms_bohr:
        lines.append(
            f"{symbol} {x * BOHR_IN_ANGSTROM!r} {y * BOHR_IN_ANGSTROM!r} "
            f"{z * BOHR_IN_ANGSTROM!r}"
        )
    return "\n".join(lines) + "\n"


def _cubic_angstrom(a_bohr):
    a = a_bohr * BOHR_IN_ANGSTROM
    return [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]


@pytest.fixture(scope="module")
def basis(tmp_path_factory):
    p = tmp_path_factory.mktemp("basis") / "pyscf-sto-3g-h.json"
    p.write_text(json.dumps(PYSCF_STO3G_H))
    return ferric.BasisSet.from_bse_json(str(p))


@pytest.fixture(scope="module")
def h2():
    return ferric.Molecule.from_xyz_string(_xyz(H2_ATOMS_BOHR))


@pytest.fixture(scope="module")
def runs(h2, basis):
    lat = _cubic_angstrom(A_BOHR)
    return {
        exx: ferric.run_rhf_gamma(h2, lat, basis, exxdiv=exx)
        for exx in ("ewald", "none")
    }


def test_molecule_coordinates_round_trip_to_bohr(h2):
    got = np.array(h2.coords_bohr())
    assert np.abs(got - np.array(H2_ATOMS_BOHR)).max() < 1e-13


@pytest.mark.parametrize("exx,e_ref", [("ewald", E_EWALD), ("none", E_NONE)])
def test_h2_sto3g_a4_matches_pinned_pyscf_aftdf(runs, exx, e_ref):
    r = runs[exx]
    assert r.converged
    assert r.exxdiv == exx
    assert abs(r.energy - e_ref) < TOL, f"{exx}: E {r.energy!r} vs {e_ref!r}"


def test_madelung_shift_identity(runs):
    # N_e = 2, so E_ewald - E_none = -v_M.
    shift = runs["ewald"].energy - runs["none"].energy
    assert abs(shift + runs["ewald"].madelung) < 1e-9
    assert runs["ewald"].madelung == runs["none"].madelung


def test_result_arrays_are_consistent(runs):
    r = runs["ewald"]
    assert r.nao == 2
    d, s, c = r.density(), r.overlap(), r.mo_coeff()
    assert d.shape == s.shape == c.shape == (2, 2)
    assert abs(np.trace(d @ s) - 2.0) < 1e-10
    assert np.abs(c.T @ s @ c - np.eye(2)).max() < 1e-10
    eps = r.mo_energy()
    assert eps.shape == (2,) and eps[0] < eps[1]
    assert np.isfinite(r.e_nuc) and r.n_g_half > 0


def test_energy_is_omega_independent(h2, basis, runs):
    # 0.8 Bohr^-1 (the Rust suite's split) expressed in 1/Angstrom.
    omega = 0.8 / BOHR_IN_ANGSTROM
    r = ferric.run_rhf_gamma(
        h2, _cubic_angstrom(A_BOHR), basis, exxdiv="ewald", omega=omega
    )
    assert abs(r.omega - omega) < 1e-12
    assert abs(r.omega - runs["ewald"].omega) > 0.1  # really a different split
    assert abs(r.energy - runs["ewald"].energy) < 1e-9


def test_charged_cell_is_a_hard_error(basis):
    ion = ferric.Molecule.from_xyz_string(_xyz(H2_ATOMS_BOHR), charge=-2)
    with pytest.raises(ValueError, match="charged cell"):
        ferric.run_rhf_gamma(ion, _cubic_angstrom(A_BOHR), basis)


def test_odd_electron_count_is_a_hard_error(basis):
    # Molecule already refuses a parity-inconsistent (H, multiplicity 1), so an
    # odd count reaches the binding as an open shell: RHF-only must refuse it.
    h = ferric.Molecule.from_xyz_string(_xyz(H2_ATOMS_BOHR[:1]), multiplicity=2)
    with pytest.raises(ValueError, match="multiplicity 2"):
        ferric.run_rhf_gamma(h, _cubic_angstrom(A_BOHR), basis)


@pytest.mark.parametrize("bad", ["vcut_sph", "", "ewald ", "madelung"])
def test_bad_exxdiv_is_a_hard_error(h2, basis, bad):
    with pytest.raises(ValueError, match="exxdiv"):
        ferric.run_rhf_gamma(h2, _cubic_angstrom(A_BOHR), basis, exxdiv=bad)


def test_oversize_cell_is_refused_before_any_work(h2, basis):
    # nao = 2 needs 8 * 2**4 = 128 bytes; 1e-8 GiB is ~10 bytes.
    with pytest.raises(ValueError, match="128 bytes"):
        ferric.run_rhf_gamma(h2, _cubic_angstrom(A_BOHR), basis, max_eri_gb=1e-8)


@pytest.mark.parametrize(
    "lattice",
    [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # not 3x3
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]],  # singular
    ],
)
def test_bad_lattice_is_a_hard_error(h2, basis, lattice):
    with pytest.raises(ValueError):
        ferric.run_rhf_gamma(h2, lattice, basis)


@pytest.mark.parametrize("kw", [{"omega": 0.0}, {"max_eri_gb": -1.0}])
def test_nonpositive_knobs_are_hard_errors(h2, basis, kw):
    with pytest.raises(ValueError):
        ferric.run_rhf_gamma(h2, _cubic_angstrom(A_BOHR), basis, **kw)
