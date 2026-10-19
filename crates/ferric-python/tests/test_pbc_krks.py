"""k-point closed-shell KS-DFT binding (`run_rks_kpts`, src/pbc.rs).

Physics validation (PySCF 2.13 KRKS pins, k-mesh == supercell/N) lives in
crates/ferric-pbc/tests/pbc_krks.rs; here the binding must reach that driver
with the same inputs and refuse the unsupported combinations by name.
Reuses the H2 / cubic a = 4 Bohr fixtures of test_pbc_bindings.py.
"""

from __future__ import annotations

import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

from test_pbc_bindings import (  # noqa: E402,F401  (fixtures are re-exported)
    OMEGA,
    _cubic,
    basis,
    h2,
    h_atom,
)

# Small grid keeps these dense-AFT runs fast; both sides use the same grid.
GRID = {"n_radial": 40, "n_angular": 110}


@pytest.mark.parametrize("functional", ["LDA", "PBE", "PBE0"])
def test_gamma_mesh_equals_the_gamma_binding(h2, basis, functional):
    # k-mesh 1x1x1 (Gamma-centred) is the Gamma point: pbc_krks.rs pins the
    # library at 1e-9 Ha.
    g = ferric.run_rks_gamma(h2, _cubic(), basis, functional, omega=OMEGA, **GRID)
    k = ferric.run_rks_kpts(
        h2, _cubic(), basis, functional, (1, 1, 1), omega=OMEGA, **GRID
    )
    assert k.converged and k.nk == 1 and k.mesh == (1, 1, 1)
    assert k.functional == functional and k.n_grid_points == g.n_grid_points
    assert abs(k.energy - g.energy) < 1e-9, f"{k.energy!r} vs {g.energy!r}"
    assert abs(k.e_xc - g.e_xc) < 1e-9
    assert k.exact_exchange_fraction == g.exact_exchange_fraction


def test_two_k_mesh_runs_and_reports_the_mesh(h2, basis):
    r = ferric.run_rks_kpts(h2, _cubic(), basis, "PBE", (1, 1, 2), omega=OMEGA, **GRID)
    assert r.converged and r.nk == 2 and len(r.mo_energy) == 2 and len(r.kpts) == 2
    assert r.homo < r.lumo


def test_unsupported_combinations_are_refused_by_name(h2, h_atom, basis):
    lat = _cubic()
    with pytest.raises(ValueError, match="multiplicity 2"):
        ferric.run_rks_kpts(h_atom, lat, basis, "LDA", (1, 1, 2))
    # Open-shell SCAN / HSE06 run as energies (test_pbc_open_shell_mgga_rsh.py);
    # k-point ROKS keeps refusing them by name.
    for f in ("SCAN", "HSE06"):
        with pytest.raises(ValueError, match="k-point ROKS supports"):
            ferric.run_roks_kpts(h_atom, lat, basis, f, (1, 1, 2))
    # The k-point KS stress is wired (test_pbc_kpoint_open_shell_and_stress.py).
    r = ferric.run_rks_kpts(
        h2, lat, basis, "LDA", (1, 1, 2), omega=OMEGA, with_stress=True, **GRID
    )
    assert r.stress().shape == (3, 3)
    r = ferric.run_rks_kpts(
        h2,
        lat,
        basis,
        "PBE0",
        (1, 1, 2),
        omega=OMEGA,
        with_gradient=True,
        with_stress=True,
        **GRID,
    )
    assert r.stress().shape == (3, 3) and r.gradient().shape == (2, 3)


# Library cross-surface pin of the same input (H2/STO-3G, a = 4 Bohr, 1x1x2, omega 0.8,
# grid 40x110, cc-pVDZ-RI aux); the library anchors are in
# crates/ferric-pbc/tests/pbc_krsh_rsgdf.rs.
RSGDF = {"jk": "rsgdf", "auxbasis": "cc-pvdz-ri"}
HSE06_RSGDF_112 = -1.3484931727


def test_range_separated_hybrid_runs_on_rsgdf_energy_only(h2, h_atom, basis):
    lat = _cubic()
    r = ferric.run_rks_kpts(
        h2, lat, basis, "HSE06", (1, 1, 2), omega=OMEGA, **RSGDF, **GRID
    )
    assert r.converged
    assert abs(r.energy - HSE06_RSGDF_112) < 1e-7, f"{r.energy!r}"
    # The reported constant is the MIXED one that was applied, not the
    # full-Coulomb mesh constant of a global hybrid on the same mesh.
    p = ferric.run_rks_kpts(
        h2, lat, basis, "PBE0", (1, 1, 2), omega=OMEGA, **RSGDF, **GRID
    )
    assert abs(r.energy - p.energy) > 1e-4
    assert abs(r.madelung - p.madelung) > 1e-2
    # Derivatives stay refused by name (the open-shell energy runs:
    # test_pbc_open_shell_mgga_rsh.py).
    for kw in ({"with_gradient": True}, {"with_stress": True}):
        with pytest.raises(ValueError, match="range-separated"):
            ferric.run_rks_kpts(
                h2, lat, basis, "HSE06", (1, 1, 2), omega=OMEGA, **RSGDF, **GRID, **kw
            )
    with pytest.raises(ValueError, match="range-separated"):
        ferric.run_uks_kpts(
            h_atom,
            lat,
            basis,
            "HSE06",
            (1, 1, 2),
            omega=OMEGA,
            with_gradient=True,
            **RSGDF,
            **GRID,
        )


# Dense-AFT J/K (the default jk): the same functional, mesh, omega and grid as
# HSE06_RSGDF_112. The two J/K builders differ by the aux-fit error only
# (2.05e-4 Ha measured, bar 1e-3), so the dense energy is an independent
# construction of the RS-GDF one. The exact value is this binding's own output
# (a regression pin, not an external reference); the PySCF pins of the dense
# kernel are crates/ferric-pbc/tests/pbc_krsh.rs and the CLI == library bit
# anchor is crates/ferric-cli/tests/periodic_kpoint_ksdft.rs. Mutation-checked:
# dropping the dense RSH kernel moves the energy by ~1.9 Ha and fails the 1e-3 bar.
HSE06_DENSE_112 = -1.3482880002


def test_range_separated_hybrid_runs_on_dense_jk(h2, basis):
    lat = _cubic()
    d = ferric.run_rks_kpts(h2, lat, basis, "HSE06", (1, 1, 2), omega=OMEGA, **GRID)
    assert d.converged
    assert abs(d.energy - HSE06_RSGDF_112) < 1e-3, f"{d.energy!r}"
    assert abs(d.energy - HSE06_DENSE_112) < 1e-8, f"{d.energy!r}"
    p = ferric.run_rks_kpts(h2, lat, basis, "PBE0", (1, 1, 2), omega=OMEGA, **GRID)
    assert abs(d.energy - p.energy) > 1e-4
    assert abs(d.madelung - p.madelung) > 1e-2
    for kw in ({"with_gradient": True}, {"with_stress": True}):
        with pytest.raises(ValueError, match="range-separated"):
            ferric.run_rks_kpts(
                h2, lat, basis, "HSE06", (1, 1, 2), omega=OMEGA, **GRID, **kw
            )


@pytest.mark.parametrize("functional", ["SCAN", "r2SCAN", "TPSS"])
def test_gamma_meta_gga_energy_equals_the_one_point_mesh(h2, basis, functional):
    # Gamma meta-GGA (issue #355): closed-shell energy, anchored to the 1x1x1
    # k-mesh on the same grid (pbc_krks.rs pins both against PySCF).
    g = ferric.run_rks_gamma(h2, _cubic(), basis, functional, omega=OMEGA, **GRID)
    k = ferric.run_rks_kpts(
        h2, _cubic(), basis, functional, (1, 1, 1), omega=OMEGA, **GRID
    )
    assert g.converged and k.converged
    assert abs(k.energy - g.energy) < 1e-8, f"{k.energy!r} vs {g.energy!r}"


def test_gamma_meta_gga_refusals_are_early_and_exact(h2, basis):
    lat = _cubic()
    # Any meta-GGA other than SCAN / r2SCAN / TPSS is refused by name at Gamma.
    with pytest.raises(
        ValueError,
        match=r"^run_rks_gamma: functional \"MGGA_X_TPSS\": .*"
        r"Gamma-point RKS supports only the validated SCAN / r2SCAN / TPSS meta-GGAs",
    ):
        ferric.run_rks_gamma(h2, lat, basis, "MGGA_X_TPSS", omega=OMEGA, **GRID)
    # Forces / stress with a meta-GGA at Gamma are refused before any SCF.
    for kw in ("with_gradient", "with_stress"):
        with pytest.raises(
            ValueError,
            match=r"^run_rks_gamma: functional \"SCAN\": .*no tau term.*SCAN",
        ):
            ferric.run_rks_gamma(
                h2, lat, basis, "SCAN", omega=OMEGA, **{kw: True}, **GRID
            )


# ── Gamma-point range-separated hybrids (issue #354) ──
# Library anchors (PySCF Gamma pins to 6e-13, Gamma == 1x1x1 mesh to 3e-15,
# mutants) are in crates/ferric-pbc/tests/pbc_grsh.rs; here the binding must
# reach the same driver and keep the unsupported combinations refused.


@pytest.mark.parametrize("jk", [{}, RSGDF], ids=["dense", "rsgdf"])
@pytest.mark.parametrize("functional", ["HSE06", "HYB_GGA_XC_CAM_B3LYP"])
def test_gamma_range_separated_hybrid_equals_the_one_point_mesh(
    h2, basis, functional, jk
):
    g = ferric.run_rks_gamma(h2, _cubic(), basis, functional, omega=OMEGA, **jk, **GRID)
    k = ferric.run_rks_kpts(
        h2, _cubic(), basis, functional, (1, 1, 1), omega=OMEGA, **jk, **GRID
    )
    assert g.converged and k.converged
    assert abs(k.energy - g.energy) < 1e-8, f"{k.energy!r} vs {g.energy!r}"
    # c_sr is reported, and the mixed (applied) Madelung constant, as the mesh.
    assert g.exact_exchange_fraction == k.exact_exchange_fraction
    assert abs(g.madelung - k.madelung) < 1e-12
    p = ferric.run_rks_gamma(h2, _cubic(), basis, "PBE0", omega=OMEGA, **jk, **GRID)
    assert abs(g.energy - p.energy) > 1e-4


def test_gamma_range_separated_hybrid_refusals(h2, h_atom, basis):
    lat = _cubic()
    for kw in ({"with_gradient": True}, {"with_stress": True}):
        with pytest.raises(
            ValueError,
            match=r"^run_rks_gamma: functional \"HSE06\": .*range-separated hybrid "
            r"gradient / stress.*energy only",
        ):
            ferric.run_rks_gamma(h2, lat, basis, "HSE06", omega=OMEGA, **kw, **GRID)
    # The open-shell (UKS) energy runs (test_pbc_open_shell_mgga_rsh.py); its
    # derivatives and VV10 stay refused by name.
    with pytest.raises(ValueError, match="range-separated hybrid gradient / stress"):
        ferric.run_uks_gamma(
            h_atom, lat, basis, "HSE06", omega=OMEGA, with_gradient=True, **GRID
        )
    with pytest.raises(ValueError, match="VV10"):
        ferric.run_rks_gamma(h2, lat, basis, "wB97X-V", omega=OMEGA, **GRID)


def test_gamma_range_separated_hybrid_refuses_range_split(h2, basis):
    # range_split would shape only the Gamma J, never the attenuated K.
    with pytest.raises(ValueError, match=r"range_split.*attenuated K"):
        ferric.run_rks_gamma(
            h2, _cubic(), basis, "HSE06", omega=OMEGA, range_split=True, **RSGDF, **GRID
        )


def test_gamma_range_separated_hybrid_roks_is_refused_and_dense_budget_counts_rsh(
    h2, h_atom, basis
):
    lat = _cubic()
    with pytest.raises(ValueError, match="range-separated"):
        ferric.run_roks_gamma(h_atom, lat, basis, "HSE06", omega=OMEGA, **GRID)
    # H2/STO-3G: nao = 2, plain dense tensor 8 nao^4 = 128 B, RSH 40 nao^4 =
    # 640 B. A 322 B cap passes the plain hybrid and refuses the RSH one
    # BEFORE any tensor is built.
    cap = 3e-7
    r = ferric.run_rks_gamma(
        h2, lat, basis, "PBE0", omega=OMEGA, max_eri_gb=cap, **GRID
    )
    assert r.converged
    with pytest.raises(ValueError, match=r"dense AFT ERI tensors need 640 bytes"):
        ferric.run_rks_gamma(
            h2, lat, basis, "HSE06", omega=OMEGA, max_eri_gb=cap, **GRID
        )
