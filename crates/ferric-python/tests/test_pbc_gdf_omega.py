"""`gdf_omega`: the RS-GDF Ewald split omega_gdf through the Python surface.

The Rust suite owns the physics (crates/ferric-pbc/tests/pbc_rsgdf.rs:
`rhf_energy_is_rsgdf_omega_independent_and_the_build_follows_omega`,
`fitted_eri_is_omega_independent`; pbc_hcore.rs:
`default_hcore_omega_cap_follows_the_rsgdf_omega`). This file checks that the
bindings THREAD the kwarg into the build and refuse it where it is not read:

* gdf_omega = 1 Bohr^-1 (given in 1/Angstrom) is bitwise the omitted kwarg
  (energy, LR G count, hcore split);
* gdf_omega = 0.5 Bohr^-1 on H2/STO-3G (PySCF digits), a = 4 Bohr,
  cc-pvdz-ri shrinks the LR half-sphere (the counter proves the knob reached
  the build, so the energy agreement is not "the kwarg was dropped"), agrees
  with the default energy to 1e-9 (the RS-GDF fitting bar of
  test_pbc_range_split.py), and lowers the default hcore split's cap (this
  cell is capped) while an explicit omega is kept;
* the range split's thresholds follow it: at lambda = 1 the orbital
  threshold omega^2/2 is 0.5 at omega_gdf = 1 (the 0.1689 STO-3G primitive
  is smooth) and 0.125 at 0.5 (nothing smooth), Gamma and k-point;
* refusals (jk="dense", omega_gdf <= 0 / non-finite) raise ValueError.
"""

from __future__ import annotations

import json
import math

import pytest

import ferric

BOHR_IN_ANGSTROM = 0.52917721092  # 1 / ferric's ANGSTROM_TO_BOHR
H2_ATOMS_BOHR = [(0.3, 0.2, 0.1), (0.3, 0.2, 1.5)]
A_BOHR = 4.0
AUX = "cc-pvdz-ri"
KMESH = (1, 1, 2)
# hcore_omega_cap(DEFAULT_HCORE_PRECISION) at the default omega_gdf, Bohr^-1
# (pbc_hcore.rs pins it); the cap is linear in omega_gdf.
HCORE_CAP_BOHR = 0.9636241116594315

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


def _per_angstrom(w_bohr):
    """A Bohr^-1 value as the 1/Angstrom kwarg (bitwise round trip)."""
    return w_bohr / BOHR_IN_ANGSTROM


def _lattice():
    a = A_BOHR * BOHR_IN_ANGSTROM
    return [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]


@pytest.fixture(scope="module")
def basis(tmp_path_factory):
    p = tmp_path_factory.mktemp("basis") / "pyscf-sto-3g-h.json"
    p.write_text(json.dumps(PYSCF_STO3G_H))
    return ferric.BasisSet.from_bse_json(str(p))


@pytest.fixture(scope="module")
def h2():
    lines = [str(len(H2_ATOMS_BOHR)), "gdf_omega test (Angstrom)"]
    for x, y, z in H2_ATOMS_BOHR:
        lines.append(
            f"H {x * BOHR_IN_ANGSTROM!r} {y * BOHR_IN_ANGSTROM!r} "
            f"{z * BOHR_IN_ANGSTROM!r}"
        )
    return ferric.Molecule.from_xyz_string("\n".join(lines) + "\n")


def _rhf(h2, basis, **kw):
    return ferric.run_rhf_gamma(h2, _lattice(), basis, jk="rsgdf", auxbasis=AUX, **kw)


def _krhf(h2, basis, **kw):
    return ferric.run_rhf_kpts(
        h2, _lattice(), basis, KMESH, jk="rsgdf", auxbasis=AUX, **kw
    )


def _counters(r):
    return r.timings["counters"]


@pytest.fixture(scope="module")
def default(h2, basis):
    return _rhf(h2, basis)


@pytest.fixture(scope="module")
def half(h2, basis):
    return _rhf(h2, basis, gdf_omega=_per_angstrom(0.5))


def test_one_bohr_is_bitwise_the_default(h2, basis, default):
    assert default.converged
    r = _rhf(h2, basis, gdf_omega=_per_angstrom(1.0))
    assert r.energy == default.energy
    assert r.omega == default.omega
    assert _counters(r)["rsgdf LR half-G"] == _counters(default)["rsgdf LR half-G"]


def test_half_bohr_reaches_the_build_and_keeps_the_energy(default, half):
    assert half.converged
    g_lo = _counters(half)["rsgdf LR half-G"]
    g_hi = _counters(default)["rsgdf LR half-G"]
    assert 0 < g_lo < g_hi, (g_lo, g_hi)
    assert abs(half.energy - default.energy) <= 1e-9, half.energy - default.energy


def test_default_hcore_split_follows_the_cap_and_explicit_omega_is_kept(
    h2, basis, default, half
):
    # a = 4 Bohr: 2.5 sqrt(pi) / 4 = 1.108 > the cap, so this cell is capped.
    assert math.isclose(
        default.omega, _per_angstrom(HCORE_CAP_BOHR), rel_tol=0, abs_tol=1e-12
    )
    assert math.isclose(
        half.omega, _per_angstrom(0.5 * HCORE_CAP_BOHR), rel_tol=0, abs_tol=1e-12
    )
    w = _per_angstrom(0.8)
    r = _rhf(h2, basis, omega=w, gdf_omega=_per_angstrom(0.5))
    assert r.omega == w


def test_range_split_thresholds_follow_gdf_omega(h2, basis):
    one = _counters(_rhf(h2, basis, range_split=1.0))
    lo = _counters(_rhf(h2, basis, range_split=1.0, gdf_omega=_per_angstrom(0.5)))
    assert one["rsgdf split orbital prims"] == lo["rsgdf split orbital prims"] > 0
    assert one["rsgdf split orbital prims smooth"] > 0, one
    assert lo["rsgdf split orbital prims smooth"] == 0, lo
    assert lo["rsgdf split aux prims smooth"] <= one["rsgdf split aux prims smooth"]


def test_kpoint_threads_gdf_omega(h2, basis):
    base = _krhf(h2, basis)
    r = _krhf(h2, basis, gdf_omega=_per_angstrom(0.5))
    assert base.converged and r.converged
    assert abs(r.energy - base.energy) <= 1e-9, r.energy - base.energy
    one = _counters(_krhf(h2, basis, range_split=1.0))
    lo = _counters(_krhf(h2, basis, range_split=1.0, gdf_omega=_per_angstrom(0.5)))
    assert one["rsgdf split orbital prims smooth"] > 0, one
    assert lo["rsgdf split orbital prims smooth"] == 0, lo


def test_gamma_uhf_threads_gdf_omega(h2, basis):
    kw = dict(jk="rsgdf", auxbasis=AUX)
    e0 = ferric.run_uhf_gamma(h2, _lattice(), basis, **kw)
    e1 = ferric.run_uhf_gamma(h2, _lattice(), basis, gdf_omega=_per_angstrom(1.0), **kw)
    assert e1.energy == e0.energy
    e2 = ferric.run_uhf_gamma(h2, _lattice(), basis, gdf_omega=_per_angstrom(0.5), **kw)
    g_lo = e2.timings["counters"]["rsgdf LR half-G"]
    assert 0 < g_lo < e0.timings["counters"]["rsgdf LR half-G"]
    assert abs(e2.energy - e0.energy) <= 1e-9, e2.energy - e0.energy


def test_dense_jk_refuses_gdf_omega(h2, basis):
    lat = _lattice()
    with pytest.raises(ValueError, match="gdf_omega.*dense"):
        ferric.run_rhf_gamma(h2, lat, basis, gdf_omega=1.0)
    with pytest.raises(ValueError, match="gdf_omega.*dense"):
        ferric.run_uhf_gamma(h2, lat, basis, gdf_omega=1.0)
    with pytest.raises(ValueError, match="gdf_omega.*dense"):
        ferric.run_mp2_gamma(h2, lat, basis, "ewald", "shifted", gdf_omega=1.0)
    with pytest.raises(ValueError, match="gdf_omega.*dense"):
        ferric.run_rhf_kpts(h2, lat, basis, KMESH, gdf_omega=1.0)
    with pytest.raises(ValueError, match="gdf_omega.*dense"):
        ferric.run_mp2_kpts(h2, lat, basis, KMESH, "ewald", "shifted", gdf_omega=1.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_nonpositive_or_nonfinite_gdf_omega_is_refused(h2, basis, bad):
    with pytest.raises(ValueError, match="gdf_omega"):
        _rhf(h2, basis, gdf_omega=bad)
    with pytest.raises(ValueError, match="gdf_omega"):
        _krhf(h2, basis, gdf_omega=bad)
