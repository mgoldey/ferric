"""k-point open-shell KS / ROHF bindings, KS stress and meta-GGA forces.

`run_uks_kpts`, `run_roks_kpts` and `run_rohf_kpts` expose
`ferric_pbc::solve_kuks_keep` / `solve_kroks_keep` / `solve_krohf_keep` with the
analytic force and stress of `kuxcgrad` / `kuxcstress`; `run_rks_kpts` gains
`with_stress` (`kpoint_rks_stress`) and the pure meta-GGAs SCAN / r2SCAN for the
energy, force and stress. The Rust suites (pbc_kuxcgrad*.rs, pbc_kuxcstress*.rs,
pbc_kxcgrad_mgga.rs, pbc_kxcstress_mgga.rs) hold the per-term anchors and
mutants; this file checks the BINDING (units, flags, routing, refusals).

The XC energy has a hard neighbour cutoff, so every energy in a comparison uses
ONE explicit `neighbour_cutoff` (see test_pbc_kpoint_ks_gradients.py).

* EXACTNESS ANCHORS, independent constructions:
  - analytic force vs a central FD of the binding's OWN k-point energies;
  - Omega tr(sigma) vs the FD of the energy under the isotropic strain of
    lattice AND atoms (the trace of the stress is not a symmetry accident:
    |Omega tr sigma| > 1e-3 is asserted);
  - a 1x1x1 mesh equals the Gamma binding's force and stress;
  - a closed-shell molecule through run_uks_kpts equals run_rks_kpts.
* Flags off: gradient() / stress() are None and the energy is not moved.
* Refusals with exact wording.
"""

from __future__ import annotations

import numpy as np
import pytest

ferric = pytest.importorskip("ferric", reason="the compiled extension is not built")

from test_pbc_kpoint_gradients import (  # noqa: E402,F401  (fixtures re-exported)
    BOHR_IN_ANGSTROM,
    H2_A,
    H2_ATOMS,
    H3_A,
    H3_ATOMS,
    OMEGA,
    _fd,
    _lat,
    _mol,
    basis,
)

from test_pbc_bindings import _hi_lattice, _hi_xyz, hi_basis  # noqa: E402,F401

FD_H = 1e-4
FORCE_BAR = 2e-8
# d(E)/d(strain) by a central difference at h = 1e-4 carries a ~h^2 residual;
# 2e-6 is ~10x the measured residuals below.
STRESS_BAR = 2e-6
CUTOFF = 14.0 * BOHR_IN_ANGSTROM
GRID = dict(n_radial=20, n_angular=50, neighbour_cutoff=CUTOFF)
SCF = dict(energy_conv=1e-13, grad_conv=1e-10, max_iter=600)


def _rks(atoms, basis, functional="PBE", mesh=(1, 1, 3), **kw):
    return ferric.run_rks_kpts(
        _mol(atoms),
        _lat(H2_A),
        basis,
        functional,
        mesh,
        omega=OMEGA,
        **GRID,
        **SCF,
        **kw,
    )


def _uks(atoms, basis, functional="PBE", mesh=(1, 1, 2), mult=2, **kw):
    return ferric.run_uks_kpts(
        _mol(atoms, mult),
        _lat(H3_A),
        basis,
        functional,
        mesh,
        omega=OMEGA,
        **GRID,
        **SCF,
        **kw,
    )


def _roks(atoms, basis, functional="PBE", mesh=(1, 1, 2), **kw):
    return ferric.run_roks_kpts(
        _mol(atoms, 2),
        _lat(H3_A),
        basis,
        functional,
        mesh,
        omega=OMEGA,
        **GRID,
        **SCF,
        **kw,
    )


def _rohf(atoms, basis, mesh=(1, 1, 2), **kw):
    return ferric.run_rohf_kpts(
        _mol(atoms, 2), _lat(H3_A), basis, mesh, omega=OMEGA, **SCF, **kw
    )


def _force_vs_fd(g, atoms, energy, tag, comps=((0, 0), (1, 2), (2, 1))):
    assert g.shape == (len(atoms), 3) and np.isfinite(g).all()
    assert np.abs(g).max() > 1e-3, f"{tag}: trivial gradient, FD check is vacuous"
    fd = _fd(energy, atoms, comps, FD_H)
    worst = max(abs(g[i, x] - v) for (i, x), v in fd.items())
    assert worst < FORCE_BAR, f"{tag}: analytic vs FD max miss {worst:.3e}"
    return worst


def _trace_vs_fd(sigma, vol, energy_scaled, tag):
    """Omega tr(sigma) against dE/dstrain of the isotropic strain."""
    analytic = vol * np.trace(sigma)
    de = (energy_scaled(FD_H) - energy_scaled(-FD_H)) / (2.0 * FD_H)
    assert abs(analytic) > 1e-3, f"{tag}: trivial stress trace, the check is vacuous"
    miss = abs(analytic - de)
    assert miss < STRESS_BAR, f"{tag}: Omega tr(sigma) {analytic!r} vs FD {de!r}"
    return miss


def _scaled(atoms, a, s):
    f = 1.0 + s
    return [tuple(f * c for c in p) for p in atoms], f * a


# ───────────────────────────────────────────────────── closed-shell RKS ──


@pytest.mark.parametrize("functional", ["LDA", "PBE", "PBE0", "SCAN", "r2SCAN"])
def test_rks_stress_trace_matches_isotropic_strain(basis, functional):
    r = _rks(H2_ATOMS, basis, functional, with_stress=True)
    assert r.converged
    sigma = r.stress()
    assert sigma.shape == (3, 3) and np.isfinite(sigma).all()

    def e_scaled(s):
        atoms, a = _scaled(H2_ATOMS, H2_A, s)
        return ferric.run_rks_kpts(
            _mol(atoms),
            _lat(a),
            basis,
            functional,
            (1, 1, 3),
            omega=OMEGA,
            **GRID,
            **SCF,
        ).energy

    miss = _trace_vs_fd(sigma, H2_A**3, e_scaled, f"RKS {functional}")
    print(f"RKS {functional} stress trace vs FD: {miss:.2e}")


# No Gamma meta-GGA binding exists, so SCAN has no 1x1x1 comparator here (its
# anchors are the FD tests and the Rust suites).
@pytest.mark.parametrize("functional", ["PBE", "PBE0"])
def test_rks_one_point_mesh_equals_the_gamma_binding_force_and_stress(
    basis, functional
):
    k = _rks(
        H2_ATOMS,
        basis,
        functional,
        mesh=(1, 1, 1),
        with_gradient=True,
        with_stress=True,
    )
    g = ferric.run_rks_gamma(
        _mol(H2_ATOMS),
        _lat(H2_A),
        basis,
        functional,
        omega=OMEGA,
        density_conv=1e-12,
        with_gradient=True,
        with_stress=True,
        **GRID,
    )
    assert np.abs(g.gradient()).max() > 1e-3
    dg = np.abs(k.gradient() - g.gradient()).max()
    ds = np.abs(k.stress() - g.stress()).max()
    print(f"{functional} 1x1x1 vs Gamma: force {dg:.2e} stress {ds:.2e}")
    assert dg < 1e-6 and ds < 1e-6


@pytest.mark.parametrize("functional", ["SCAN", "r2SCAN"])
def test_meta_gga_force_matches_fd_of_own_energy(basis, functional):
    r = _rks(H2_ATOMS, basis, functional, with_gradient=True)
    assert r.converged and r.functional == functional
    worst = _force_vs_fd(
        r.gradient(),
        H2_ATOMS,
        lambda at: _rks(at, basis, functional).energy,
        f"RKS {functional}",
        comps=((0, 0), (0, 2), (1, 1), (1, 2)),
    )
    print(f"RKS {functional} force vs FD: {worst:.2e}")


def test_meta_gga_rsgdf_force_is_available_and_stress_is_refused(basis):
    kw = dict(mesh=(1, 1, 2), jk="rsgdf", auxbasis="cc-pvdz-ri")
    r = _rks(H2_ATOMS, basis, "SCAN", with_gradient=True, **kw)
    assert r.jk == "rsgdf" and np.abs(r.gradient()).max() > 1e-3
    with pytest.raises(
        ValueError,
        match=r"^run_rks_kpts: the k-point meta-GGA stress is anchored on the dense-AFT J "
        r'only; use jk="dense" with with_stress \(with_gradient supports RS-GDF\)$',
    ):
        _rks(H2_ATOMS, basis, "SCAN", with_stress=True, **kw)
    # A GGA RS-GDF stress is not refused.
    assert _rks(H2_ATOMS, basis, "PBE", with_stress=True, **kw).stress() is not None


def test_rks_flags_off_leave_the_energy_and_the_none_results(basis):
    off = _rks(H2_ATOMS, basis, "PBE0")
    on = _rks(H2_ATOMS, basis, "PBE0", with_gradient=True, with_stress=True)
    assert off.gradient() is None and off.stress() is None
    assert on.gradient() is not None and on.stress() is not None
    assert abs(on.energy - off.energy) < 1e-12


# ─────────────────────────────────────────────────────────────── UKS ──


@pytest.mark.parametrize("functional", ["LDA", "PBE", "PBE0"])
def test_uks_force_and_stress_match_fd(basis, functional):
    r = _uks(H3_ATOMS, basis, functional, with_gradient=True, with_stress=True)
    assert r.converged and r.method == "uks" and r.functional == functional
    assert r.nalpha == 2 and r.nbeta == 1 and r.ewald_start == "staged"
    worst = _force_vs_fd(
        r.gradient(),
        H3_ATOMS,
        lambda at: _uks(at, basis, functional).energy,
        f"UKS {functional}",
    )

    def e_scaled(s):
        atoms, a = _scaled(H3_ATOMS, H3_A, s)
        return ferric.run_uks_kpts(
            _mol(atoms, 2),
            _lat(a),
            basis,
            functional,
            (1, 1, 2),
            omega=OMEGA,
            **GRID,
            **SCF,
        ).energy

    miss = _trace_vs_fd(r.stress(), H3_A**3, e_scaled, f"UKS {functional}")
    print(f"UKS {functional}: force vs FD {worst:.2e}, stress trace vs FD {miss:.2e}")


def test_uks_result_carries_the_ks_fields_and_hf_results_do_not(basis):
    r = _uks(H3_ATOMS, basis, "PBE0")
    assert r.e_xc < 0.0 and r.exact_exchange_fraction == pytest.approx(0.25)
    assert r.n_grid_points > 0 and abs(r.electrons_on_grid - 3.0) < 5e-2
    assert r.mo_energy_beta is not None and len(r.mo_energy) == r.nk
    u = ferric.run_uhf_kpts(
        _mol(H3_ATOMS, 2), _lat(H3_A), basis, (1, 1, 2), omega=OMEGA, **SCF
    )
    assert u.functional is None and u.e_xc is None and u.n_grid_points is None


def test_closed_shell_molecule_through_uks_equals_rks_energy_and_force(basis):
    u = ferric.run_uks_kpts(
        _mol(H2_ATOMS),
        _lat(H2_A),
        basis,
        "PBE",
        (1, 1, 3),
        omega=OMEGA,
        with_gradient=True,
        **GRID,
        **SCF,
    )
    r = _rks(H2_ATOMS, basis, "PBE", with_gradient=True)
    # The UKS driver really ran (the agreement below is not RKS compared to
    # itself); bitwise equality is expected, not suspicious: with
    # D_alpha = D_beta, J[D_a + D_b] = J[2 D_a] and the exchange scale are
    # exact power-of-two rescalings.
    assert (u.method, u.nalpha, u.nbeta) == ("uks", 1, 1) and abs(u.s2) < 1e-6
    assert r.energy == pytest.approx(u.energy, abs=1e-9)
    de = abs(u.energy - r.energy)
    dg = np.abs(u.gradient() - r.gradient()).max()
    print(f"closed-shell UKS vs RKS: energy {de:.2e} force {dg:.2e}")
    assert de < 1e-9 and dg < 1e-8
    assert np.abs(r.gradient()).max() > 1e-3


def test_uks_one_point_mesh_equals_the_gamma_binding(basis):
    k = _uks(
        H3_ATOMS, basis, "PBE", mesh=(1, 1, 1), with_gradient=True, with_stress=True
    )
    g = ferric.run_uks_gamma(
        _mol(H3_ATOMS, 2),
        _lat(H3_A),
        basis,
        "PBE",
        omega=OMEGA,
        density_conv=1e-12,
        with_gradient=True,
        with_stress=True,
        **GRID,
    )
    assert np.abs(g.gradient()).max() > 1e-3
    dg = np.abs(k.gradient() - g.gradient()).max()
    ds = np.abs(k.stress() - g.stress()).max()
    print(f"UKS 1x1x1 vs Gamma: force {dg:.2e} stress {ds:.2e}")
    assert dg < 1e-6 and ds < 1e-6


def test_uks_rsgdf_force_matches_fd(basis):
    kw = dict(jk="rsgdf", auxbasis="cc-pvdz-ri")
    r = _uks(H3_ATOMS, basis, "PBE", with_gradient=True, with_stress=True, **kw)
    assert r.jk == "rsgdf" and r.stress().shape == (3, 3)
    _force_vs_fd(
        r.gradient(),
        H3_ATOMS,
        lambda at: _uks(at, basis, "PBE", **kw).energy,
        "UKS rsgdf",
        comps=((0, 0), (2, 1)),
    )


# ──────────────────────────────────────────────────── ROKS and ROHF ──


@pytest.mark.parametrize("method", ["roks", "rohf"])
def test_restricted_open_shell_force_and_stress_match_fd(basis, method):
    def run(atoms, a=H3_A, **kw):
        if method == "roks":
            return ferric.run_roks_kpts(
                _mol(atoms, 2),
                _lat(a),
                basis,
                "PBE",
                (1, 1, 2),
                omega=OMEGA,
                **GRID,
                **SCF,
                **kw,
            )
        return ferric.run_rohf_kpts(
            _mol(atoms, 2), _lat(a), basis, (1, 1, 2), omega=OMEGA, **SCF, **kw
        )

    r = run(H3_ATOMS, with_gradient=True, with_stress=True)
    assert r.converged and r.method == method
    assert (r.functional == "PBE") == (method == "roks")
    worst = _force_vs_fd(
        r.gradient(), H3_ATOMS, lambda at: run(at).energy, method.upper()
    )

    def e_scaled(s):
        atoms, a = _scaled(H3_ATOMS, H3_A, s)
        return run(atoms, a).energy

    miss = _trace_vs_fd(r.stress(), H3_A**3, e_scaled, method.upper())
    print(f"{method}: force vs FD {worst:.2e}, stress trace vs FD {miss:.2e}")


def test_roks_and_rohf_one_point_mesh_equal_the_gamma_bindings(basis):
    kr = _roks(H3_ATOMS, basis, mesh=(1, 1, 1), with_gradient=True, with_stress=True)
    gr = ferric.run_roks_gamma(
        _mol(H3_ATOMS, 2),
        _lat(H3_A),
        basis,
        "PBE",
        omega=OMEGA,
        density_conv=1e-12,
        with_gradient=True,
        with_stress=True,
        **GRID,
    )
    kh = _rohf(H3_ATOMS, basis, mesh=(1, 1, 1), with_gradient=True, with_stress=True)
    gh = ferric.run_rohf_gamma(
        _mol(H3_ATOMS, 2),
        _lat(H3_A),
        basis,
        omega=OMEGA,
        density_conv=1e-12,
        with_gradient=True,
        with_stress=True,
    )
    for tag, k, g in (("ROKS", kr, gr), ("ROHF", kh, gh)):
        assert np.abs(g.gradient()).max() > 1e-3
        dg = np.abs(k.gradient() - g.gradient()).max()
        ds = np.abs(k.stress() - g.stress()).max()
        print(f"{tag} 1x1x1 vs Gamma: force {dg:.2e} stress {ds:.2e}")
        assert dg < 1e-6 and ds < 1e-6, tag


def test_roks_is_not_uks(basis):
    # The routing anchor: for a genuinely open-shell system ROKS and UKS are
    # different methods, so a wiring that sent ROKS through the UKS driver
    # would change the energy by far more than the bar below.
    u = _uks(H3_ATOMS, basis, "PBE", with_gradient=True)
    r = _roks(H3_ATOMS, basis, "PBE", with_gradient=True)
    de, dg = abs(u.energy - r.energy), np.abs(u.gradient() - r.gradient()).max()
    print(f"UKS vs ROKS on H3: energy {de:.2e}, force {dg:.2e}")
    # measured 8.6e-5 Ha; a mis-routed run differs by 0 (bar 1e-5 sits between)
    assert de > 1e-5
    assert dg > 1e-6
    assert u.method == "uks" and r.method == "roks"


# ─────────────────────────────────────────────────────────── refusals ──


def test_open_shell_refusals_have_exact_wording(basis):
    for fn, tag in ((_uks, "run_uks_kpts"), (_roks, "run_roks_kpts")):
        for f in ("SCAN", "r2SCAN"):
            with pytest.raises(
                ValueError,
                match=rf'^{tag}: functional "{f}": .*meta-GGA.*\. k-point open-shell KS '
                r"supports LDA, GGA and global-hybrid functionals only \(no meta-GGA\)$",
            ):
                fn(H3_ATOMS, basis, f)
        with pytest.raises(
            ValueError,
            match=rf'^{tag}: functional "HSE06": .*range-separated.*\. k-point '
            r"open-shell KS supports",
        ):
            fn(H3_ATOMS, basis, "HSE06")
    with pytest.raises(
        ValueError,
        match=r'^run_rks_kpts: functional "HSE06": .*range-separated.*\. k-point RKS '
        r"supports LDA, GGA, global-hybrid and SCAN / r2SCAN meta-GGA functionals only$",
    ):
        _rks(H2_ATOMS, basis, "HSE06", with_gradient=True)


def test_restricted_open_shell_derivatives_are_dense_only(basis):
    kw = dict(jk="rsgdf", auxbasis="cc-pvdz-ri")
    msg = (
        r"anchored on the dense-AFT J/K only; use jk=\"dense\" with with_gradient / "
        r"with_stress$"
    )
    with pytest.raises(
        ValueError, match=r"^run_roks_kpts: the restricted-open-shell .*" + msg
    ):
        _roks(H3_ATOMS, basis, with_gradient=True, **kw)
    with pytest.raises(
        ValueError, match=r"^run_rohf_kpts: the restricted-open-shell .*" + msg
    ):
        _rohf(H3_ATOMS, basis, with_stress=True, **kw)
    # The ROKS energy itself runs on RS-GDF.
    assert _roks(H3_ATOMS, basis, **kw).converged


def test_range_split_with_derivatives_is_refused(basis):
    with pytest.raises(ValueError, match="range_split"):
        _uks(
            H3_ATOMS,
            basis,
            jk="rsgdf",
            auxbasis="cc-pvdz-ri",
            range_split=True,
            with_stress=True,
        )


def test_ecp_cell_derivatives_are_refused(hi_basis):
    cell = ferric.Molecule.from_xyz_string(_hi_xyz(1))
    for kw in ({"with_gradient": True}, {"with_stress": True}):
        with pytest.raises(
            ValueError,
            match=r"^run_rks_kpts: k-point KS forces and stress with ECPs are not "
            r"validated \(with_gradient / with_stress on a cell whose basis carries an "
            r"ECP; the energy without them runs\)$",
        ):
            ferric.run_rks_kpts(cell, _hi_lattice(1), hi_basis, "PBE", (1, 1, 2), **kw)
