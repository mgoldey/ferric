"""Open-shell SCAN / r2SCAN and range-separated hybrid ENERGIES through the
bindings (issue #357; library PRs #479 / #480): `run_uks_kpts` (SCAN /
r2SCAN on jk="dense"; HSE06 / CAM-B3LYP on either J/K) and `run_uks_gamma`
(range-separated hybrids). Open-shell range-separated derivatives, ROKS
meta-GGA and the Gamma meta-GGA stay refused by name; the UKS SCAN / r2SCAN
forces and stress and the ROKS range-separated energies are anchored in
test_pbc_rsh_open_mgga_derivs.py.

Anchors, each an independent construction or an external number:

* Singlet limit: a closed-shell molecule through run_uks_kpts equals
  run_rks_kpts (library measured <= 3e-15 for SCAN, <= 4e-15 for CAM-B3LYP).
  Blind to a per-spin error by construction (D_alpha = D_beta), so it only
  catches gross routing errors; the polarized anchors below are the real test.
* Gamma vs 1x1x1: run_uks_gamma (Gamma J/XC/SCF + the k = 0 attenuated K) vs
  run_uks_kpts on a (1, 1, 1) mesh (the k-point J/XC/SCF), polarized H3.
* Cross-surface pin: the polarized H3 doublet of
  crates/ferric-cli/tests/periodic_kpoint_open_shell.rs (bundled STO-3G, cubic
  a = 4 Bohr, 1x1x2, grid 20x50, 14 Bohr cutoff, omega 0.8 Bohr^-1) gives the
  library `solve_kuks` energy that test prints; the CLI equals it to 1e-12
  and the binding must reach the same number.
* PySCF pins: the library pins (crates/ferric-pbc/tests/pbc_kuks_mgga.rs,
  pbc_oshell_rsh.rs) are on a UNIFORM XC grid the bindings do not expose, so
  they carry over only to the difference between that grid and the binding's
  atom-centred grid. The bars below are derived from the measured difference
  and are far below the mutant scale they must resolve (SCAN vs PBE, CAM-B3LYP
  vs PBE0: > 1e-2).
"""

from __future__ import annotations

import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

from test_pbc_kpoint_gradients import (  # noqa: E402,F401  (fixtures re-exported)
    BOHR_IN_ANGSTROM,
    H2_A,
    H2_ATOMS,
    OMEGA,
    _lat,
    _mol,
    basis,
)

SCF = dict(energy_conv=1e-12, grad_conv=1e-9, max_iter=200)
GRID_SMALL = dict(n_radial=40, n_angular=110)
CAM = "HYB_GGA_XC_CAM_B3LYP"

# Polarized H3 doublet of the CLI anchor (Bohr), cubic a = 4 Bohr.
H3_CLI = [(0.3, 0.2, 0.1), (0.35, 0.12, 1.5), (0.3, 1.9, 0.4)]
CLI_GRID = dict(n_radial=20, n_angular=50, neighbour_cutoff=14.0 * BOHR_IN_ANGSTROM)
# Library `solve_kuks` energies of that input (Gamma-centred 1x1x2, dense AFT),
# printed by cli_open_shell_scan_and_hse06_equal_library_solve_kuks.
# The binding builds its geometry from Bohr * 0.529... while the CLI test writes
# 12-decimal Angstrom; that round trip moves the energy by 4.5e-14 / 4.8e-14
# (measured), so the bar is 5e-13 (10x), against > 2e-2 for the mutants there.
LIB_H3_112 = {"SCAN": -1.6227824411493592, "HSE06": -1.6224626688847699}
LIB_BAR = 5e-13

# PySCF 2.13.1 KUKS / UKS pins (uniform XC grids, see the module doc).
# pbc_kuks_mgga.rs h3_scan_matches_pinned_pyscf: cubic a = 4 Bohr, 1x1x1.
H3_SCAN_ATOMS = [(0.3, 0.2, 0.1), (0.3, 0.2, 1.5), (0.3, 1.9, 0.4)]
PYSCF_H3_SCAN_111 = -1.524551950364
# testdata/reference/pbc_oshell_sto3g_rsh.json h3d, CAM-B3LYP, Gamma, ewald.
H3D_ATOMS = [(0.3, 0.2, 0.1), (0.3, 0.2, 1.5), (0.3, 0.2, 2.9)]
TRI_A_BOHR = [[4.6, 0.0, 0.0], [0.9, 4.3, 0.0], [0.5, 0.7, 4.8]]
PYSCF_H3D_CAM_GAMMA = -1.5724933181342036
# Measured binding - PySCF pin on the 75x302 grid: SCAN -1.63e-4 (radially
# converged: -1.65e-4 at 150x302 and 250x302; 75x302 is the periodic
# default), CAM-B3LYP 2.3e-7. SCAN is grid-sensitive, so its bar
# (1e-3) resolves only gross errors (PBE in its place: 3.8e-2); the tau
# mutants of the library (>= 2e-5) are below it and are caught by the library
# pins and by LIB_H3_112 above, not here. CAM-B3LYP's bar is 2e-6 (~10x).
SCAN_PIN_BAR = 1e-3
CAM_PIN_BAR = 2e-6


def _tri():
    return [[x * BOHR_IN_ANGSTROM for x in row] for row in TRI_A_BOHR]


def _uks(atoms, lat, basis, functional, mesh, mult=2, **kw):
    kw = {**GRID_SMALL, **SCF, **kw}
    return ferric.run_uks_kpts(
        _mol(atoms, mult), lat, basis, functional, mesh, omega=OMEGA, **kw
    )


@pytest.mark.parametrize("functional", ["SCAN", "r2SCAN", "HSE06", CAM])
def test_singlet_uks_equals_rks(basis, functional):
    u = _uks(H2_ATOMS, _lat(H2_A), basis, functional, (1, 1, 2), mult=1)
    r = ferric.run_rks_kpts(
        _mol(H2_ATOMS),
        _lat(H2_A),
        basis,
        functional,
        (1, 1, 2),
        omega=OMEGA,
        **GRID_SMALL,
        **SCF,
    )
    assert u.method == "uks" and (u.nalpha, u.nbeta) == (1, 1)
    d = abs(u.energy - r.energy)
    print(f"{functional} singlet UKS - RKS: {d:.2e}")
    assert d < 1e-9


@pytest.mark.parametrize("functional", ["HSE06", CAM])
def test_gamma_uks_rsh_equals_the_one_point_mesh(basis, functional):
    g = ferric.run_uks_gamma(
        _mol(H3D_ATOMS, 2),
        _tri(),
        basis,
        functional,
        omega=OMEGA,
        density_conv=1e-11,
        **GRID_SMALL,
    )
    k = _uks(H3D_ATOMS, _tri(), basis, functional, (1, 1, 1))
    assert g.converged and g.method == "uks" and g.s2 > 0.7
    d = abs(g.energy - k.energy)
    print(f"{functional} Gamma - 1x1x1: {d:.2e}")
    assert d < 1e-9
    # The mixed Madelung constant and the short-range fraction are reported.
    assert abs(g.madelung - k.madelung) < 1e-12
    assert g.exact_exchange_fraction == k.exact_exchange_fraction
    # MUTANT: the global hybrid on the same cell is a different number.
    p = ferric.run_uks_gamma(
        _mol(H3D_ATOMS, 2),
        _tri(),
        basis,
        "PBE0",
        omega=OMEGA,
        density_conv=1e-11,
        **GRID_SMALL,
    )
    assert abs(g.madelung - p.madelung) > 1e-3


@pytest.mark.parametrize("functional", ["SCAN", "HSE06"])
def test_binding_reaches_the_library_energy_of_the_cli_anchor(functional):
    bs = ferric.BasisSet.bundled("sto-3g")
    r = ferric.run_uks_kpts(
        _mol(H3_CLI, 2),
        _lat(4.0),
        bs,
        functional,
        (1, 1, 2),
        omega=OMEGA,
        **CLI_GRID,
        **SCF,
    )
    d = abs(r.energy - LIB_H3_112[functional])
    print(f"{functional} binding - library: {d:.2e} ({r.energy!r})")
    assert d < LIB_BAR


def test_scan_polarized_matches_the_pyscf_pin(basis):
    r = _uks(
        H3_SCAN_ATOMS, _lat(4.0), basis, "SCAN", (1, 1, 1), n_radial=75, n_angular=302
    )
    d = r.energy - PYSCF_H3_SCAN_111
    p = _uks(
        H3_SCAN_ATOMS, _lat(4.0), basis, "PBE", (1, 1, 1), n_radial=75, n_angular=302
    )
    print(
        f"SCAN H3 1x1x1 - PySCF pin: {d:.2e}; PBE - pin {p.energy - PYSCF_H3_SCAN_111:.2e}"
    )
    assert abs(d) < SCAN_PIN_BAR
    assert abs(p.energy - PYSCF_H3_SCAN_111) > 10 * SCAN_PIN_BAR


def test_cam_b3lyp_polarized_gamma_matches_the_pyscf_pin(basis):
    g = ferric.run_uks_gamma(
        _mol(H3D_ATOMS, 2),
        _tri(),
        basis,
        CAM,
        omega=OMEGA,
        density_conv=1e-11,
        n_radial=75,
        n_angular=302,
    )
    d = g.energy - PYSCF_H3D_CAM_GAMMA
    print(f"CAM-B3LYP H3d Gamma - PySCF pin: {d:.2e}")
    assert abs(d) < CAM_PIN_BAR


def test_rsgdf_energy_runs_and_meta_gga_needs_dense(basis):
    rs = dict(jk="rsgdf", auxbasis="cc-pvdz-ri")
    r = _uks(H3D_ATOMS, _tri(), basis, "HSE06", (1, 1, 2), **rs)
    d = _uks(H3D_ATOMS, _tri(), basis, "HSE06", (1, 1, 2))
    print(f"HSE06 1x1x2 RS-GDF - dense: {r.energy - d.energy:.2e}")
    assert r.converged and abs(r.energy - d.energy) < 1e-3
    with pytest.raises(
        ValueError,
        match=r"^run_uks_kpts: functional \"SCAN\": meta-GGA with RS-GDF.*use jk=\"dense\"$",
    ):
        _uks(H3D_ATOMS, _tri(), basis, "SCAN", (1, 1, 2), **rs)
    g = ferric.run_uks_gamma(
        _mol(H3D_ATOMS, 2),
        _tri(),
        basis,
        "HSE06",
        omega=OMEGA,
        density_conv=1e-11,
        **rs,
        **GRID_SMALL,
    )
    assert g.converged and g.jk == "rsgdf"


def test_derivatives_roks_and_gamma_meta_gga_stay_refused(basis):
    lat = _lat(4.0)
    wb = "HYB_GGA_XC_WB97X"
    with pytest.raises(
        ValueError,
        match=rf"^run_uks_kpts: with_stress: .*{wb}: the range-separated hybrid "
        r"force and stress are validated for HSE06 / CAM-B3LYP",
    ):
        _uks(H3_SCAN_ATOMS, lat, basis, wb, (1, 1, 2), with_stress=True)
    with pytest.raises(
        ValueError,
        match=r'^run_roks_kpts: functional "SCAN": the k-point ROKS SCAN / r2SCAN '
        r"meta-GGA energy is validated in the library \(#492\) but not yet wired "
        r"in the bindings \(#498\)",
    ):
        ferric.run_roks_kpts(
            _mol(H3_SCAN_ATOMS, 2), lat, basis, "SCAN", (1, 1, 2), omega=OMEGA
        )
    with pytest.raises(ValueError, match="validated for HSE06 / CAM-B3LYP"):
        ferric.run_uks_gamma(
            _mol(H3_SCAN_ATOMS, 2), lat, basis, wb, omega=OMEGA, with_stress=True
        )
    # Gamma UKS / ROKS meta-GGA: library energy only (#492), not wired (#498).
    with pytest.raises(ValueError, match=r"not yet wired \(#498\).*1x1x1"):
        ferric.run_uks_gamma(_mol(H3_SCAN_ATOMS, 2), lat, basis, "SCAN", omega=OMEGA)
    with pytest.raises(ValueError, match=r"Gamma-point ROKS SCAN.*not yet wired"):
        ferric.run_roks_gamma(_mol(H3_SCAN_ATOMS, 2), lat, basis, "SCAN", omega=OMEGA)
    with pytest.raises(ValueError, match="range_split"):
        ferric.run_uks_gamma(
            _mol(H3_SCAN_ATOMS, 2),
            lat,
            basis,
            "HSE06",
            omega=OMEGA,
            jk="rsgdf",
            auxbasis="cc-pvdz-ri",
            range_split=True,
        )
    with pytest.raises(
        ValueError,
        match="Gamma-point ROKS range-separated hybrid stress has no library entry",
    ):
        ferric.run_roks_gamma(
            _mol(H3_SCAN_ATOMS, 2), lat, basis, "HSE06", omega=OMEGA, with_stress=True
        )
