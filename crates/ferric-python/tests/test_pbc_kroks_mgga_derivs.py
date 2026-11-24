"""Issue #498 (meta-GGA ROKS derivatives, from #492): the k-point ROKS SCAN /
r2SCAN force and stress (library `kpoint_roks_gradient` /
`kpoint_roks_stress`, dense-AFT J/K) through `run_roks_kpts(with_gradient=,
with_stress=)`.

Anchors:

* Binding vs library: the pins below are the CLI `run_end` values of
  crates/ferric-cli/tests/periodic_kroks_mgga_derivs.rs (H3 doublet at the
  parsed-xyz Bohr positions X_H3, cubic a = 4 Bohr, bundled STO-3G, omega
  0.8 Bohr^-1, XC grid 20x50 with a 14 Bohr cutoff, dense AFT, exxdiv ewald,
  Gamma-centred 1x1x2), which that test proves equal to the library BIT FOR
  BIT. Bars E_BAR / G_BAR = 1e-12 (the binding builds its cell from
  repr-Angstrom, the CLI from 12-decimal xyz; the measured difference is
  printed), far below the UKS sibling gap that file asserts.
* Independent construction inside the binding: the SCAN force and the
  r2SCAN sigma_zz against a central FD of the binding's own energies (h =
  1e-4; uniaxial strain of the lattice c and every atom z for the stress).

Measured (2026-11-19): binding - library energy <= 4.4e-16, force 5.0e-16,
stress 4.7e-17; FD force 1.2e-9 (bar 1e-7), sigma_zz 8.8e-10 (bar 1e-8).

Fail-before: on origin/main every test here raised ValueError (the shared
derivative gate refused "the k-point ROKS meta-GGA force and stress are
library-only").

Mutation ledger (crates/ferric-python/src/pbc.rs mutated, rebuilt, reverted),
both KILLED: the ROKS force times (1 + 1e-6) fails the force pin (3.6e-7)
and the FD (3.5e-7); the ROKS stress transposed fails the stress pin (2.2e-4;
the sigma_zz FD is blind to it, the pin is not).
"""

from __future__ import annotations

import numpy as np
import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

from test_pbc_kpoint_gradients import _fd, _lat, _mol  # noqa: E402
from test_pbc_rsh_open_mgga_derivs import A, H3, X_H3, _kw, _rows  # noqa: E402

MESH = (1, 1, 2)
E_BAR = 1e-12
G_BAR = 1e-12
FD_H = 1e-4
FD_FORCE_BAR = 1e-7
FD_STRESS_BAR = 1e-8

# CLI run_end (= library) values, see the module doc.
KROKS_SCAN_E = -1.6220219536917795
KROKS_SCAN_G = [
    0.00013777347862394375,
    0.04203063345043023,
    0.21336555316846165,
    -0.005646226927778237,
    0.060924863804634216,
    -0.3558015251603164,
    0.0055084534491540585,
    -0.10295549725507099,
    0.1424359719918531,
]
KROKS_R2SCAN_E = -1.6199733010049107
KROKS_R2SCAN_SIGMA = [
    0.010523288540377096,
    -7.633999706638337e-5,
    -5.1863763707774756e-5,
    2.039210222353052e-5,
    -0.0032179863876405524,
    0.0006518831551539455,
    -0.00015494056056485842,
    0.0004283104928395648,
    -0.010425080304329596,
]


@pytest.fixture(scope="module")
def sto3g():
    return ferric.BasisSet.bundled("sto-3g")


def _roks(atoms, basis, functional, lat=None, **kw):
    return ferric.run_roks_kpts(
        _mol(atoms, 2), lat or _lat(A), basis, functional, MESH, **_kw(**kw)
    )


def test_roks_scan_force_equals_the_library(sto3g):
    r = _roks(_rows(X_H3), sto3g, "SCAN", with_gradient=True)
    assert r.converged and r.method == "roks" and r.stress() is None
    de = abs(r.energy - KROKS_SCAN_E)
    dg = float(np.abs(r.gradient().ravel() - np.array(KROKS_SCAN_G)).max())
    print(f"k ROKS SCAN binding - library: energy {de:.2e}, force {dg:.2e}")
    assert de < E_BAR and dg < G_BAR


def test_roks_r2scan_stress_equals_the_library(sto3g):
    r = _roks(_rows(X_H3), sto3g, "r2SCAN", with_stress=True)
    assert r.converged and r.gradient() is None
    de = abs(r.energy - KROKS_R2SCAN_E)
    ds = float(np.abs(r.stress().ravel() - np.array(KROKS_R2SCAN_SIGMA)).max())
    print(f"k ROKS r2SCAN binding - library: energy {de:.2e}, stress {ds:.2e}")
    assert np.abs(r.stress()).max() > 1e-6
    assert de < E_BAR and ds < G_BAR


def test_roks_scan_force_matches_fd(sto3g):
    r = _roks(H3, sto3g, "SCAN", with_gradient=True)
    g = r.gradient()
    fd = _fd(lambda at: _roks(at, sto3g, "SCAN").energy, H3, ((1, 2), (2, 1)), FD_H)
    miss = max(abs(g[i, x] - v) for (i, x), v in fd.items())
    print(f"k ROKS SCAN force vs FD: {miss:.2e}")
    assert np.abs(g).max() > 1e-2 and miss < FD_FORCE_BAR


def test_roks_r2scan_stress_matches_fd(sto3g):
    def energy(eps):
        at = [(x, y, z * (1.0 + eps)) for x, y, z in H3]
        return _roks(at, sto3g, "r2SCAN", lat=_lat(A, A * (1.0 + eps))).energy

    s = _roks(H3, sto3g, "r2SCAN", with_stress=True).stress()
    fd = (energy(FD_H) - energy(-FD_H)) / (2.0 * FD_H) / A**3
    miss = abs(fd - s[2, 2])
    print(
        f"k ROKS r2SCAN sigma_zz: binding {s[2, 2]:.12e} FD {fd:.12e} miss {miss:.2e}"
    )
    assert abs(s[2, 2]) > 1e-4 and miss < FD_STRESS_BAR
