"""Issue #498: the library routes wired to the Python bindings.

* ROKS with range-separated hybrids (energy): `run_roks_kpts` and
  `run_roks_gamma`.
* Closed-shell range-separated hybrid force and stress on the dense-AFT J/K:
  `run_rks_kpts` and `run_rks_gamma` `with_gradient` / `with_stress`.
* Open-shell UKS SCAN / r2SCAN force and stress (dense-AFT J/K):
  `run_uks_kpts` `with_gradient` / `with_stress`.

Anchors:

* Binding vs library: the numbers below are the CLI's `run_end` values for
  the same cell (bundled STO-3G, cubic a = 4 Bohr, omega 0.8 Bohr^-1, XC grid
  20x50 with a 14 Bohr cutoff, dense AFT, exxdiv ewald, 1x1x2 Gamma-centred
  mesh or the Gamma point), which crates/ferric-cli/tests/
  periodic_rsh_open_mgga_derivs.rs and periodic_kpoint_stress.rs prove equal
  to the library BIT FOR BIT. The geometries of the force pins are the
  optimizer-moved ones those runs logged (Bohr, exact repr). The binding
  builds its cell from Angstrom = Bohr * 0.529..., the CLI from the same
  value through its own path; the measured binding - library difference is
  recorded at each bar, which is ~10x it and far below the sibling-functional
  gap every pin also asserts (>= 5e-4).
* Independent constructions inside the binding: the k-point HSE06 and UKS
  SCAN forces against a central FD of the binding's own energies; the Gamma
  CAM-B3LYP force and stress against the (1, 1, 1) mesh.

Mutation ledger (binding source mutated, rebuilt, reverted; both KILLED):
`run_roks_gamma` with `exxdiv = none` forced misses the ROKS pin by 0.458 Ha;
the Gamma RSH stress returned transposed misses the 1x1x1 stress by 8.3e-5
(sigma is unsymmetrised on the atom grid).
"""

from __future__ import annotations

import numpy as np
import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

from test_pbc_kpoint_gradients import (  # noqa: E402
    BOHR_IN_ANGSTROM,
    _fd,
    _lat,
    _mol,
)

CAM = "HYB_GGA_XC_CAM_B3LYP"
A = 4.0
OMEGA = 0.8 / BOHR_IN_ANGSTROM
GRID = dict(n_radial=20, n_angular=50, neighbour_cutoff=14.0 * BOHR_IN_ANGSTROM)
H2 = [(0.3, 0.2, 0.1), (0.35, 0.12, 1.5)]
H3 = [(0.3, 0.2, 0.1), (0.35, 0.12, 1.5), (0.3, 1.9, 0.4)]

# Library (= CLI run_end) values, see the module doc.
LIB_KROKS_HSE06 = -1.6217160984287937
LIB_KROKS_PBE0 = -1.6219656406405636
LIB_GROKS_CAM = -1.574336912992121
LIB_GROKS_B3LYP = -1.5495921505050014
# k-point RKS HSE06 (H2, 1x1x2) at the optimizer's geometry.
KRKS_X = [
    0.28962072444573533,
    0.21693700326959886,
    0.11795542079555987,
    0.3603792755542646,
    0.10306299672964517,
    1.4820445792044414,
]
KRKS_E = -1.351917468165642
KRKS_G = [
    0.006376126758084773,
    -0.010626126131219776,
    0.005229074775570414,
    -0.006376126758084767,
    0.010626126131219764,
    -0.005229074775570192,
]
# Gamma RKS CAM-B3LYP (H2).
GRKS_X = [
    0.29275250226011057,
    0.2116424251517566,
    -0.040728002935465676,
    0.3572474977398895,
    0.10835757484748752,
    1.640728002935466,
]
GRKS_E = -1.6257074299364547
GRKS_G = [
    0.0050572074946240156,
    -0.008100792347349433,
    0.06732024855095817,
    -0.005057207494623972,
    0.008100792347349353,
    -0.06732024855095776,
]
# k-point UKS SCAN (H3 doublet, 1x1x2).
KUKS_X = [
    0.30119913349778354,
    0.18172371570717294,
    0.02618184918062595,
    0.35245159921766334,
    0.09399054958084878,
    1.65551290307369,
    0.29634926728455346,
    1.9442857347112168,
    0.31830524774568336,
]
KUKS_E = -1.694375238002249
KUKS_G = [
    -0.0006087426581896983,
    0.024764433137510374,
    0.10510005279581994,
    -0.004595809180346683,
    0.03759756745262778,
    -0.21703784945310944,
    0.005204551838536309,
    -0.06236200059013376,
    0.11193779665729968,
]
# Binding - library, measured: energy <= 7.1e-14 (ROKS; 0 for the force
# runs), force <= 5.6e-17. Bars ~10x the worst (and far below the >= 5e-4
# sibling gaps). FD: measured 2.8e-9 (HSE06, h^2 floor of pbc_krsh_force.rs),
# 7.6e-10 / 3.8e-10 (SCAN / r2SCAN); Gamma vs 1x1x1: 1.1e-16 / 1.7e-17.
E_BAR = 1e-12
G_BAR = 1e-12
FD_H = 1e-4
FD_BAR = 2e-8
GAMMA_BAR = 1e-10


@pytest.fixture(scope="module")
def sto3g():
    return ferric.BasisSet.bundled("sto-3g")


def _rows(flat):
    return [tuple(flat[3 * i : 3 * i + 3]) for i in range(len(flat) // 3)]


def _kw(**kw):
    return dict(omega=OMEGA, **GRID, **kw)


def test_roks_rsh_energies_equal_the_library(sto3g):
    k = ferric.run_roks_kpts(_mol(H3, 2), _lat(A), sto3g, "HSE06", (1, 1, 2), **_kw())
    g = ferric.run_roks_gamma(_mol(H3, 2), _lat(A), sto3g, CAM, **_kw())
    assert k.converged and g.converged
    assert k.method == "roks" and g.method == "roks"
    dk, dg = abs(k.energy - LIB_KROKS_HSE06), abs(g.energy - LIB_GROKS_CAM)
    print(f"ROKS binding - library: k HSE06 {dk:.2e}, Gamma CAM-B3LYP {dg:.2e}")
    assert dk < E_BAR and dg < E_BAR
    assert abs(LIB_KROKS_HSE06 - LIB_KROKS_PBE0) > 1e-4
    assert abs(LIB_GROKS_CAM - LIB_GROKS_B3LYP) > 1e-2
    # c_sr is reported as the fraction (HSE06: 0.25).
    assert k.exact_exchange_fraction == pytest.approx(0.25)


@pytest.mark.parametrize(
    "tag, run, functional, x, e, g, mult",
    [
        ("k RKS HSE06", "rks_k", "HSE06", KRKS_X, KRKS_E, KRKS_G, 1),
        ("Gamma RKS CAM-B3LYP", "rks_g", CAM, GRKS_X, GRKS_E, GRKS_G, 1),
        ("k UKS SCAN", "uks_k", "SCAN", KUKS_X, KUKS_E, KUKS_G, 2),
    ],
)
def test_forces_equal_the_library(sto3g, tag, run, functional, x, e, g, mult):
    mol, lat = _mol(_rows(x), mult), _lat(A)
    if run == "rks_k":
        r = ferric.run_rks_kpts(
            mol, lat, sto3g, functional, (1, 1, 2), with_gradient=True, **_kw()
        )
    elif run == "rks_g":
        r = ferric.run_rks_gamma(
            mol, lat, sto3g, functional, with_gradient=True, **_kw()
        )
    else:
        r = ferric.run_uks_kpts(
            mol, lat, sto3g, functional, (1, 1, 2), with_gradient=True, **_kw()
        )
    de = abs(r.energy - e)
    dg = float(np.abs(r.gradient().ravel() - np.array(g)).max())
    print(f"{tag}: binding - library energy {de:.2e}, force {dg:.2e}")
    assert de < E_BAR and dg < G_BAR


def test_kpoint_rsh_force_matches_fd_and_stress_runs(sto3g):
    def run(atoms, **kw):
        return ferric.run_rks_kpts(
            _mol(atoms), _lat(A), sto3g, "HSE06", (1, 1, 3), **_kw(**kw)
        )

    r = run(H2, with_gradient=True, with_stress=True)
    g = r.gradient()
    assert np.abs(g).max() > 1e-3
    fd = _fd(lambda at: run(at).energy, H2, ((0, 2), (1, 1)), FD_H)
    miss = max(abs(g[i, x] - v) for (i, x), v in fd.items())
    print(f"k RKS HSE06 force vs FD: {miss:.2e}")
    assert miss < FD_BAR
    s = r.stress()
    assert s.shape == (3, 3) and np.abs(s).max() > 1e-6
    assert run(H2).energy == r.energy


def test_gamma_rsh_derivatives_equal_the_one_point_mesh(sto3g):
    kw = _kw(with_gradient=True, with_stress=True)
    g = ferric.run_rks_gamma(_mol(H2), _lat(A), sto3g, CAM, **kw)
    k = ferric.run_rks_kpts(_mol(H2), _lat(A), sto3g, CAM, (1, 1, 1), **kw)
    dg = float(np.abs(g.gradient() - k.gradient()).max())
    ds = float(np.abs(g.stress() - k.stress()).max())
    print(f"Gamma vs 1x1x1 CAM-B3LYP: force {dg:.2e}, stress {ds:.2e}")
    assert np.abs(g.gradient()).max() > 1e-3 and np.abs(g.stress()).max() > 1e-6
    assert dg < GAMMA_BAR and ds < GAMMA_BAR


def test_unvalidated_rsh_derivatives_are_refused_by_name(sto3g):
    # wB97X runs as an energy; its force / stress are not validated.
    msg = (
        r"HYB_GGA_XC_WB97X: the range-separated hybrid force and stress are "
        r"validated for HSE06 / CAM-B3LYP \(HYB_GGA_XC_CAM_B3LYP\) only"
    )
    wb = "HYB_GGA_XC_WB97X"
    for kw in ("with_gradient", "with_stress"):
        with pytest.raises(ValueError, match=rf"^run_rks_kpts: {kw}: .*{msg}"):
            ferric.run_rks_kpts(
                _mol(H2), _lat(A), sto3g, wb, (1, 1, 2), **_kw(**{kw: True})
            )
        with pytest.raises(
            ValueError, match=rf'^run_rks_gamma: functional "{wb}": .*{msg}'
        ):
            ferric.run_rks_gamma(_mol(H2), _lat(A), sto3g, wb, **_kw(**{kw: True}))


@pytest.mark.parametrize("functional", ["SCAN", "r2SCAN"])
def test_uks_meta_gga_force_matches_fd_and_stress_runs(sto3g, functional):
    def run(atoms, **kw):
        return ferric.run_uks_kpts(
            _mol(atoms, 2), _lat(A), sto3g, functional, (1, 1, 2), **_kw(**kw)
        )

    r = run(H3, with_gradient=True, with_stress=True)
    g = r.gradient()
    fd = _fd(lambda at: run(at).energy, H3, ((0, 2), (2, 1)), FD_H)
    miss = max(abs(g[i, x] - v) for (i, x), v in fd.items())
    print(f"k UKS {functional} force vs FD: {miss:.2e}")
    assert np.abs(g).max() > 1e-3 and miss < FD_BAR
    assert r.stress().shape == (3, 3) and np.abs(r.stress()).max() > 1e-6
