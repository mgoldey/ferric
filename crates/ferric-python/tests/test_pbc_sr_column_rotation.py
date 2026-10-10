"""`sr_column_rotation`: the Gamma SR column rotation through the Python surface.

The Rust suite owns the numerics (crates/ferric-pbc/tests/pbc_sr_rotation.rs:
identity anchor bitwise, back-transformed J3 / V_SR covariance, H2CO/cc-pVDZ
RHF energy within 1e-10 Ha, thread-count bitwise, mutants). This file checks
that the bindings RESOLVE the kwarg (None = auto, the default; True; False)
into both Gamma builds and refuse an explicit True where it cannot be
honoured:

* H2CO / cc-pVDZ, cubic a = 9 Bohr, cc-pvdz-ri, hcore omega 0.8 Bohr^-1 (the
  Rust test's cell): the default (None) is the rotation, bitwise
  sr_column_rotation=True, with both rotated-column counters set (8 = C 1s,
  2s, p; O 1s, 2s, p; one s per H — pinned by the Rust test for the hcore;
  the RS-GDF walk rotates the same orbital basis), and it matches the
  unrotated energy (False) within 1e-10 Ha. The counters are what show the
  rotation reached the build, so the energy agreement is not "the kwarg was
  dropped".
* sr_column_rotation=False sets no counter.
* An open-shell Gamma binding (pbc.rs path) resolves it too (H2 / cc-pVDZ).
* The default ROTATES with with_gradient / with_stress too (counters set,
  energy, forces and stress equal the explicit False run to the screening
  precision; the Rust suite owns the finite-difference and mutant anchors)
  and resolves OFF, silently, with jk="dense".
* Refusal of an explicit True: jk="dense" (ValueError, by name); the
  k-point bindings do not take the kwarg (TypeError) and run the library
  default, which ROTATES their RS-GDF energy builds (counter
  "k rsgdf SR3 rotated columns").
"""

from __future__ import annotations

import pytest

import ferric

BOHR_IN_ANGSTROM = 0.52917721092  # 1 / ferric's ANGSTROM_TO_BOHR
AUX = "cc-pvdz-ri"
E_BAR = 1e-10  # Ha; the Rust test's RHF bar (pbc_sr_rotation.rs E_BAR)
D_BAR = 1e-8  # forces (Ha/Bohr) and stress (Ha/Bohr^3), rotated vs unrotated
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


def _rotated_columns(r):
    """(hcore, rsgdf) rotated-column counters; absent = 0 (the counters are
    only written when the rotation ran)."""
    c = r.timings["counters"]
    return c.get(HCORE_ROT, 0), c.get(RSGDF_ROT, 0)


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
def unrotated(h2co, ccpvdz):
    return _rhf(h2co, ccpvdz, sr_column_rotation=False)


def test_default_is_the_rotation_in_both_builds(h2co, ccpvdz, default):
    assert default.converged
    c = default.timings["counters"]
    assert c[HCORE_ROT] == 8, c
    assert c[RSGDF_ROT] == c[HCORE_ROT], c
    # None resolves to exactly the explicit request.
    explicit = _rhf(h2co, ccpvdz, sr_column_rotation=True)
    assert explicit.energy == default.energy
    assert _rotated_columns(explicit) == (8, 8)


def test_rotation_keeps_the_energy(default, unrotated):
    assert default.converged and unrotated.converged
    assert abs(default.energy - unrotated.energy) <= E_BAR, (
        default.energy - unrotated.energy
    )


def test_false_sets_no_counter(unrotated):
    c = unrotated.timings["counters"]
    assert HCORE_ROT not in c and RSGDF_ROT not in c, c


def test_gamma_open_shell_binding_resolves_it(h2, ccpvdz):
    kw = dict(jk="rsgdf", auxbasis=AUX)
    e0 = ferric.run_uhf_gamma(h2, _cubic(6.0), ccpvdz, sr_column_rotation=False, **kw)
    e1 = ferric.run_uhf_gamma(h2, _cubic(6.0), ccpvdz, **kw)
    assert e0.converged and e1.converged
    # One s column per H (4 primitives; the single-primitive 0.122 column).
    assert _rotated_columns(e1) == (2, 2), e1.timings["counters"]
    assert _rotated_columns(e0) == (0, 0)
    assert HCORE_ROT not in e0.timings["counters"]
    assert abs(e1.energy - e0.energy) <= E_BAR, e1.energy - e0.energy


def _deriv_kw(deriv):
    return {"jk": "rsgdf", "auxbasis": AUX, deriv: True}


@pytest.mark.parametrize("deriv", ["with_gradient", "with_stress"])
def test_default_rotates_with_derivatives(h2, ccpvdz, deriv):
    # The force/stress builds differentiate the rotated walks: counters set
    # (one s column per H), and the derivative equals the unrotated run's.
    kw = _deriv_kw(deriv)
    auto = ferric.run_rhf_gamma(h2, _cubic(6.0), ccpvdz, **kw)
    off = ferric.run_rhf_gamma(h2, _cubic(6.0), ccpvdz, sr_column_rotation=False, **kw)
    assert auto.converged and off.converged
    assert _rotated_columns(auto) == (2, 2), auto.timings["counters"]
    assert _rotated_columns(off) == (0, 0), off.timings["counters"]
    assert abs(auto.energy - off.energy) <= E_BAR, auto.energy - off.energy
    got, ref = (
        (auto.gradient(), off.gradient())
        if deriv == "with_gradient"
        else (auto.stress(), off.stress())
    )
    assert got is not None and ref is not None
    assert abs(ref).max() > 0.0
    assert abs(got - ref).max() <= D_BAR, abs(got - ref).max()
    uauto = ferric.run_uhf_gamma(h2, _cubic(6.0), ccpvdz, **kw)
    assert _rotated_columns(uauto) == (2, 2), uauto.timings["counters"]


def test_default_resolves_off_with_dense_jk(h2, ccpvdz):
    lat = _cubic(6.0)
    auto = ferric.run_rhf_gamma(h2, lat, ccpvdz)
    off = ferric.run_rhf_gamma(h2, lat, ccpvdz, sr_column_rotation=False)
    assert auto.converged
    assert _rotated_columns(auto) == (0, 0), auto.timings["counters"]
    assert auto.energy == off.energy


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


@pytest.mark.parametrize("flag", [True, False, None])
def test_kpoint_bindings_do_not_take_it(h2, ccpvdz, flag):
    lat = _cubic(6.0)
    kw = dict(jk="rsgdf", auxbasis=AUX, sr_column_rotation=flag)
    with pytest.raises(TypeError, match="sr_column_rotation"):
        ferric.run_rhf_kpts(h2, lat, ccpvdz, (1, 1, 2), **kw)
    with pytest.raises(TypeError, match="sr_column_rotation"):
        ferric.run_mp2_kpts(h2, lat, ccpvdz, (1, 1, 2), "ewald", "shifted", **kw)


def test_kpoint_default_rotates(h2, ccpvdz):
    # The library default (Auto) rotates the k-point RS-GDF energy build on a
    # basis that rotates at Gamma (one s column per H, two H atoms). The
    # k-point counters carry their own name; the Gamma names stay absent.
    r = ferric.run_rhf_kpts(
        h2, _cubic(6.0), ccpvdz, (1, 1, 2), jk="rsgdf", auxbasis=AUX
    )
    assert r.converged
    c = r.timings["counters"]
    assert c.get("k rsgdf SR3 rotated columns") == 2, c
    assert _rotated_columns(r) == (0, 0), c
