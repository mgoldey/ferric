"""`sr_column_rotation`: the Gamma SR column rotation through the Python surface.

The Rust suite owns the numerics (crates/ferric-pbc/tests/pbc_sr_rotation.rs:
identity anchor bitwise, back-transformed J3 / V_SR covariance, H2CO/cc-pVDZ
RHF energy within 1e-10 Ha, thread-count bitwise, mutants). This file checks
that the bindings THREAD the kwarg into both Gamma builds and refuse it where
it cannot be honoured:

* H2CO / cc-pVDZ, cubic a = 9 Bohr, cc-pvdz-ri, hcore omega 0.8 Bohr^-1 (the
  Rust test's cell): sr_column_rotation=True gives the unrotated energy within
  1e-10 Ha, with both rotated-column counters set (8 = C 1s, 2s, p; O 1s, 2s,
  p; one s per H — pinned by the Rust test for the hcore; the RS-GDF walk
  rotates the same orbital basis). The counters are what show the kwarg
  reached the build, so the energy agreement is not "the kwarg was dropped".
* sr_column_rotation=False is bitwise the omitted kwarg (and sets no counter).
* An open-shell Gamma binding (pbc.rs path) threads it too (H2 / cc-pVDZ).
* Refusals: jk="dense", with_gradient / with_stress (ValueError, by name); the
  k-point bindings do not take the kwarg (TypeError).
"""

from __future__ import annotations

import pytest

import ferric

BOHR_IN_ANGSTROM = 0.52917721092  # 1 / ferric's ANGSTROM_TO_BOHR
AUX = "cc-pvdz-ri"
E_BAR = 1e-10  # Ha; the Rust test's RHF bar (pbc_sr_rotation.rs E_BAR)
HCORE_OMEGA_BOHR = 0.8  # pbc_sr_rotation.rs HCORE_OMEGA
HCORE_ROT = "hcore SR rotated columns"
RSGDF_ROT = "rsgdf SR3 rotated columns"

FORMALDEHYDE = """4
formaldehyde
C  0.0000  0.0000 -0.5290
O  0.0000  0.0000  0.6760
H  0.0000  0.9430 -1.1160
H  0.0000 -0.9430 -1.1160
"""

H2 = """2
h2
H  0.0000  0.0000  0.0000
H  0.0000  0.0000  0.7400
"""


def _cubic(a_bohr):
    a = a_bohr * BOHR_IN_ANGSTROM
    return [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]


def _per_angstrom(w_bohr):
    """A Bohr^-1 value as the 1/Angstrom kwarg."""
    return w_bohr / BOHR_IN_ANGSTROM


@pytest.fixture(scope="module")
def ccpvdz():
    return ferric.BasisSet.bundled("cc-pvdz")


@pytest.fixture(scope="module")
def h2co():
    return ferric.Molecule.from_xyz_string(FORMALDEHYDE)


@pytest.fixture(scope="module")
def h2():
    return ferric.Molecule.from_xyz_string(H2)


def _rhf(mol, basis, **kw):
    return ferric.run_rhf_gamma(
        mol,
        _cubic(9.0),
        basis,
        jk="rsgdf",
        auxbasis=AUX,
        omega=_per_angstrom(HCORE_OMEGA_BOHR),
        **kw,
    )


@pytest.fixture(scope="module")
def default(h2co, ccpvdz):
    return _rhf(h2co, ccpvdz)


@pytest.fixture(scope="module")
def rotated(h2co, ccpvdz):
    return _rhf(h2co, ccpvdz, sr_column_rotation=True)


def test_rotation_keeps_the_energy_and_reaches_both_builds(default, rotated):
    assert default.converged and rotated.converged
    c = rotated.timings["counters"]
    assert c[HCORE_ROT] == 8, c
    assert c[RSGDF_ROT] == c[HCORE_ROT], c
    assert abs(rotated.energy - default.energy) <= E_BAR, (
        rotated.energy - default.energy
    )


def test_false_is_bitwise_the_default(h2co, ccpvdz, default):
    r = _rhf(h2co, ccpvdz, sr_column_rotation=False)
    assert r.energy == default.energy
    for c in (r.timings["counters"], default.timings["counters"]):
        assert HCORE_ROT not in c and RSGDF_ROT not in c, c


def test_gamma_open_shell_binding_threads_it(h2, ccpvdz):
    kw = dict(jk="rsgdf", auxbasis=AUX)
    e0 = ferric.run_uhf_gamma(h2, _cubic(6.0), ccpvdz, **kw)
    e1 = ferric.run_uhf_gamma(h2, _cubic(6.0), ccpvdz, sr_column_rotation=True, **kw)
    assert e0.converged and e1.converged
    c = e1.timings["counters"]
    # One s column per H (4 primitives; the single-primitive 0.122 column).
    assert c[HCORE_ROT] == 2 and c[RSGDF_ROT] == 2, c
    assert HCORE_ROT not in e0.timings["counters"]
    assert abs(e1.energy - e0.energy) <= E_BAR, e1.energy - e0.energy


def test_dense_jk_refuses_it(h2, ccpvdz):
    lat = _cubic(6.0)
    with pytest.raises(ValueError, match="sr_column_rotation.*rsgdf"):
        ferric.run_rhf_gamma(h2, lat, ccpvdz, sr_column_rotation=True)
    with pytest.raises(ValueError, match="sr_column_rotation.*rsgdf"):
        ferric.run_uhf_gamma(h2, lat, ccpvdz, sr_column_rotation=True)
    with pytest.raises(ValueError, match="sr_column_rotation.*rsgdf"):
        ferric.run_mp2_gamma(
            h2, lat, ccpvdz, "ewald", "shifted", sr_column_rotation=True
        )


@pytest.mark.parametrize("deriv", ["with_gradient", "with_stress"])
def test_derivatives_refuse_it(h2, ccpvdz, deriv):
    kw = {"jk": "rsgdf", "auxbasis": AUX, "sr_column_rotation": True, deriv: True}
    with pytest.raises(ValueError, match="sr_column_rotation.*with_gradient"):
        ferric.run_rhf_gamma(h2, _cubic(6.0), ccpvdz, **kw)
    with pytest.raises(ValueError, match="sr_column_rotation.*with_gradient"):
        ferric.run_uhf_gamma(h2, _cubic(6.0), ccpvdz, **kw)


@pytest.mark.parametrize("flag", [True, False])
def test_kpoint_bindings_do_not_take_it(h2, ccpvdz, flag):
    lat = _cubic(6.0)
    kw = dict(jk="rsgdf", auxbasis=AUX, sr_column_rotation=flag)
    with pytest.raises(TypeError, match="sr_column_rotation"):
        ferric.run_rhf_kpts(h2, lat, ccpvdz, (1, 1, 2), **kw)
    with pytest.raises(TypeError, match="sr_column_rotation"):
        ferric.run_mp2_kpts(h2, lat, ccpvdz, (1, 1, 2), "ewald", "shifted", **kw)
