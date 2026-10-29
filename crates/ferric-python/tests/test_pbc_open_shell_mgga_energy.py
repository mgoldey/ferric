"""Open-shell periodic SCAN / r2SCAN ENERGIES wired to the bindings by #498:
`run_roks_kpts` (dense-AFT J/K; library `solve_kroks`, #582), `run_uks_gamma`
and `run_roks_gamma` (dense-AFT J/K; library `gamma_uks` / `gamma_roks`,
#581 / #582) and `run_uks_kpts` on `jk="rsgdf"` (library `solve_kuks`, #579).

Anchor: the binding reaches the LIBRARY energy of the CLI anchor input of
crates/ferric-cli/tests/periodic_open_mgga_energy.rs (H3 doublet in a 4 Bohr
cube, STO-3G or STO-3G plus a p shell (zeta 1) on every H, exxdiv ewald,
omega 0.8 Bohr^-1, XC grid 20x50 with a 14 Bohr cutoff, Gamma-centred 1x1x2
or the Gamma point). That test prints the library values pinned below and
asserts the CLI equals them bit for bit. The binding builds its geometry from
Bohr * 0.529..., the CLI test from 12-decimal Angstrom, so the bar is
LIB_BAR = 5e-13 (the round trip moves these energies by ~5e-14), far below
the sibling distances (>= 6e-5 Ha: PBE for SCAN, SCAN for r2SCAN, UKS for
ROKS, dense for RS-GDF).

The force and stress of these routes stay refused by name (the k-point ROKS
meta-GGA derivatives are library work in progress, #586; the Gamma open-shell
meta-GGA ones do not exist; the UKS RS-GDF ones are not wired):
test_pbc_open_shell_mgga_rsh.py, test_pbc_kpoint_open_shell_and_stress.py.
"""

from __future__ import annotations

import json

import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

from test_pbc_kpoint_gradients import (  # noqa: E402
    BOHR_IN_ANGSTROM,
    OMEGA,
    _lat,
    _mol,
)

H3 = [(0.3, 0.2, 0.1), (0.35, 0.12, 1.5), (0.3, 1.9, 0.4)]
GRID = dict(n_radial=20, n_angular=50, neighbour_cutoff=14.0 * BOHR_IN_ANGSTROM)
KSCF = dict(energy_conv=1e-12, grad_conv=1e-9, max_iter=200)
GSCF = dict(density_conv=1e-10)
LIB_BAR = 5e-13

# Library energies of the CLI anchors (periodic_open_mgga_energy.rs, measured
# 2026-10-28; CLI == library bit for bit).
LIB = {
    "kroks-scan-sp": -1.638356659690284,
    "kroks-r2scan": -1.6199733010049095,
    "guks-scan": -1.5379830514206851,
    "groks-r2scan-sp": -1.5469183586040778,
    "kuks-scan-rsgdf": -1.6228455326685207,
    "kuks-scan-rsgdf-split": -1.622845532668506,
}

_H_S = {
    "function_type": "gto_spherical",
    "angular_momentum": [0],
    "exponents": ["0.3425250914E+01", "0.6239137298E+00", "0.1688554040E+00"],
    "coefficients": [["0.1543289673E+00", "0.5353281423E+00", "0.4446345422E+00"]],
}
_H_P = {
    "function_type": "gto_spherical",
    "angular_momentum": [1],
    "exponents": ["1.0000000000E+00"],
    "coefficients": [["1.0000000000E+00"]],
}


@pytest.fixture(scope="module")
def sto3g():
    return ferric.BasisSet.bundled("sto-3g")


@pytest.fixture(scope="module")
def h_sp(tmp_path_factory):
    p = tmp_path_factory.mktemp("basis") / "h_sp.json"
    p.write_text(json.dumps({"elements": {"1": {"electron_shells": [_H_S, _H_P]}}}))
    return ferric.BasisSet.from_bse_json(str(p))


def _check(tag, r, nao):
    d = r.energy - LIB[tag]
    print(f"{tag}: binding - library {d:.2e} ({r.energy!r})")
    assert r.converged and r.nao == nao
    assert abs(d) < LIB_BAR


def _kpts(fn, basis, functional, **kw):
    return fn(
        _mol(H3, 2),
        _lat(4.0),
        basis,
        functional,
        (1, 1, 2),
        omega=OMEGA,
        **GRID,
        **KSCF,
        **kw,
    )


def _gamma(fn, basis, functional):
    return fn(_mol(H3, 2), _lat(4.0), basis, functional, omega=OMEGA, **GRID, **GSCF)


def test_kpoint_roks_scan_p_shell_reaches_the_library(h_sp):
    _check("kroks-scan-sp", _kpts(ferric.run_roks_kpts, h_sp, "SCAN"), 12)


def test_kpoint_roks_r2scan_reaches_the_library(sto3g):
    _check("kroks-r2scan", _kpts(ferric.run_roks_kpts, sto3g, "r2SCAN"), 3)


def test_gamma_uks_scan_reaches_the_library(sto3g):
    _check("guks-scan", _gamma(ferric.run_uks_gamma, sto3g, "SCAN"), 3)


def test_gamma_roks_r2scan_p_shell_reaches_the_library(h_sp):
    _check("groks-r2scan-sp", _gamma(ferric.run_roks_gamma, h_sp, "r2SCAN"), 12)


def test_kpoint_uks_scan_on_rsgdf_reaches_the_library(sto3g):
    r = _kpts(ferric.run_uks_kpts, sto3g, "SCAN", jk="rsgdf", auxbasis="cc-pvdz-ri")
    assert r.jk == "rsgdf"
    _check("kuks-scan-rsgdf", r, 3)


def test_gamma_open_shell_meta_gga_on_rsgdf_is_refused(sto3g):
    for fn, route in ((ferric.run_uks_gamma, "UKS"), (ferric.run_roks_gamma, "ROKS")):
        with pytest.raises(
            ValueError,
            match=rf"Gamma-point {route} meta-GGA energy is validated on the "
            r"dense-AFT J/K only",
        ):
            fn(
                _mol(H3, 2),
                _lat(4.0),
                sto3g,
                "SCAN",
                omega=OMEGA,
                jk="rsgdf",
                auxbasis="cc-pvdz-ri",
            )


def test_kpoint_uks_scan_on_range_split_rsgdf_reaches_the_library(sto3g):
    # The split moves this energy by 1.5e-14 only (below LIB_BAR), so this pin
    # checks the route runs and stays on the library value; the CLI anchor
    # tells split from unsplit by the bits.
    r = _kpts(
        ferric.run_uks_kpts,
        sto3g,
        "SCAN",
        jk="rsgdf",
        auxbasis="cc-pvdz-ri",
        range_split=True,
    )
    _check("kuks-scan-rsgdf-split", r, 3)
