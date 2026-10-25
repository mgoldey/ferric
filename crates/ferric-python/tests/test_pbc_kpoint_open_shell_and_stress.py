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
* ECP cells (HI, H2I with the LANL2DZ ECP on I): force and two strain
  components against FD of the binding's own energies, all four routes.
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

from test_pbc_bindings import hi_basis  # noqa: E402,F401

FD_H = 1e-4
FORCE_BAR = 2e-8
# d(E)/d(strain) by a central difference at h = 1e-4 carries a ~h^2 residual;
# 2e-6 is ~10x the measured residuals below.
STRESS_BAR = 2e-6
CUTOFF = 14.0 * BOHR_IN_ANGSTROM
# 1x1x1 mesh vs the Gamma binding: measured 1e-16 .. 6e-13 (force and stress,
# RKS / UKS / ROKS / ROHF); 1e-10 leaves ~150x headroom over the worst, while a
# permuted or mis-scaled tensor misses by >1e-3.
GAMMA_BAR = 1e-10
# Monkhorst-Pack full-tensor stress vs FD: measured <= 8.8e-8 (diagonal) and
# 5.3e-9 (off-diagonal); 2e-7 is ~2x the worst.
FULL_STRESS_BAR = 2e-7
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


def _uks(atoms, basis, functional="PBE", mesh=(1, 1, 2), mult=2, lat=None, **kw):
    return ferric.run_uks_kpts(
        _mol(atoms, mult),
        lat if lat is not None else _lat(H3_A),
        basis,
        functional,
        mesh,
        omega=OMEGA,
        **GRID,
        **SCF,
        **kw,
    )


def _roks(atoms, basis, functional="PBE", mesh=(1, 1, 2), lat=None, **kw):
    return ferric.run_roks_kpts(
        _mol(atoms, 2),
        lat if lat is not None else _lat(H3_A),
        basis,
        functional,
        mesh,
        omega=OMEGA,
        **GRID,
        **SCF,
        **kw,
    )


def _rohf(atoms, basis, mesh=(1, 1, 2), lat=None, **kw):
    return ferric.run_rohf_kpts(
        _mol(atoms, 2),
        lat if lat is not None else _lat(H3_A),
        basis,
        mesh,
        omega=OMEGA,
        **SCF,
        **kw,
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


_STRESS_COMPS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def _strained(atoms, a, eps):
    """Atoms (Bohr) and lattice rows (Angstrom) under r -> (1 + eps) r."""
    f = np.eye(3) + eps
    at = [tuple(f @ np.array(p)) for p in atoms]
    lat = (a * np.eye(3)) @ f.T * BOHR_IN_ANGSTROM
    return at, lat.tolist()


def _full_stress_vs_fd(sigma, vol, energy, atoms, a, tag):
    """All six independent components of Omega sigma against central FD of the
    energy under the symmetric strain eps_ab = eps_ba = h (which gives
    Omega (sigma_ab + sigma_ba) off the diagonal). Also checks that a cyclic
    permutation of the axes of sigma would NOT pass (the trace is blind to it)."""
    fd = {}
    for i, j in _STRESS_COMPS:
        e = np.zeros((3, 3))
        e[i, j] = e[j, i] = FD_H
        ep, em = (energy(*_strained(atoms, a, s * e)) for s in (1.0, -1.0))
        fd[i, j] = (ep - em) / (2.0 * FD_H)

    def analytic(sg, i, j):
        return vol * (sg[i, j] if i == j else sg[i, j] + sg[j, i])

    def worst(sg):
        return max(abs(analytic(sg, i, j) - v) for (i, j), v in fd.items())

    assert max(abs(v) for (i, j), v in fd.items() if i != j) > 1e-4, tag
    miss = worst(sigma)
    assert miss < FULL_STRESS_BAR, f"{tag}: full-tensor stress vs FD {miss:.3e}"
    cyc = [1, 2, 0]
    assert worst(sigma[np.ix_(cyc, cyc)]) > 100 * FULL_STRESS_BAR, (
        f"{tag}: a cyclic axis permutation of sigma is not detected"
    )
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


# The Gamma meta-GGA force / stress (issue #492) are the Gamma SCF routed
# through the 1x1x1 k-point derivatives, so the 1x1x1 binding is their glue
# comparator here (independent anchors: FD of the Gamma energy in
# crates/ferric-pbc/tests/pbc_gamma_mgga_derivs.rs).
@pytest.mark.parametrize("functional", ["PBE", "PBE0", "SCAN", "TPSS"])
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
    assert dg < GAMMA_BAR and ds < GAMMA_BAR


@pytest.mark.parametrize("functional", ["SCAN", "r2SCAN", "TPSS"])
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


def test_meta_gga_rsgdf_force_and_stress_are_available(basis):
    kw = dict(mesh=(1, 1, 2), jk="rsgdf", auxbasis="cc-pvdz-ri")
    r = _rks(H2_ATOMS, basis, "SCAN", with_gradient=True, **kw)
    assert r.jk == "rsgdf"
    worst = _force_vs_fd(
        r.gradient(),
        H2_ATOMS,
        lambda at: _rks(at, basis, "SCAN", **kw).energy,
        "SCAN rsgdf",
        comps=((0, 0), (1, 2)),
    )
    print(f"SCAN RS-GDF force vs FD: {worst:.2e}")
    # The SCAN RS-GDF stress runs (library anchors: FD of the SCAN RS-GDF
    # energy and the PBE-to-SCAN fit-error transfer, pbc_kxcstress_mgga.rs) and
    # differs from the dense-J stress only by the fit error.
    s = np.asarray(_rks(H2_ATOMS, basis, "SCAN", with_stress=True, **kw).stress())
    d = np.asarray(
        _rks(H2_ATOMS, basis, "SCAN", with_stress=True, mesh=(1, 1, 2)).stress()
    )
    assert s.shape == (3, 3) and np.all(np.isfinite(s))
    print(
        f"SCAN stress rsgdf - dense: {np.abs(s - d).max():.2e} (|sigma| {np.abs(d).max():.2e})"
    )
    # Measured 1.5e-6 of |sigma| 6.6e-3 (2.3e-4 relative); bar ~9x that.
    assert np.abs(s - d).max() < 2e-3 * np.abs(d).max()
    # ...and is resolved, so the RS-GDF route really ran.
    assert np.abs(s - d).max() > 1e-8
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
    assert dg < GAMMA_BAR and ds < GAMMA_BAR


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

    def e_scaled(s):
        atoms, a = _scaled(H3_ATOMS, H3_A, s)
        return _uks(atoms, basis, "PBE", lat=_lat(a), **kw).energy

    miss = _trace_vs_fd(r.stress(), H3_A**3, e_scaled, "UKS rsgdf")
    print(f"UKS RS-GDF stress trace vs FD: {miss:.2e}")


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
        assert dg < GAMMA_BAR and ds < GAMMA_BAR, tag


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


def test_roks_level_shift_default_override_and_refusals(basis):
    # #356: run_roks_kpts(level_shift=None) is the solver's automatic shift
    # (0.05 Ha, ramped out, for a hybrid; none for LDA/GGA); a float overrides
    # it; the shift is ramped to zero at convergence, so the energy agrees.
    d = _roks(H3_ATOMS, basis, "PBE0")
    z = _roks(H3_ATOMS, basis, "PBE0", level_shift=0.0)
    assert d.converged and z.converged
    assert d.level_shift == 0.05 and z.level_shift == 0.0
    assert abs(d.energy - z.energy) < 1e-8, f"{d.energy - z.energy:.2e}"
    assert _roks(H3_ATOMS, basis, "PBE").level_shift == 0.0
    assert _rohf(H3_ATOMS, basis).level_shift == 0.0
    assert _uks(H3_ATOMS, basis, "PBE").level_shift is None
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(
            ValueError,
            match=r"^run_roks_kpts: level_shift must be a finite number >= 0",
        ):
            _roks(H3_ATOMS, basis, "PBE0", level_shift=bad)


# ──────────────────────────────────────── Monkhorst-Pack 1x1x2 meshes ──

MP = dict(mesh=(1, 1, 2), centring="mp")


@pytest.mark.parametrize("route", ["uks", "roks", "rohf", "scan"])
def test_monkhorst_pack_force_matches_fd(basis, route):
    run, atoms = {
        "uks": (lambda at, **kw: _uks(at, basis, **MP, **kw), H3_ATOMS),
        "roks": (lambda at, **kw: _roks(at, basis, **MP, **kw), H3_ATOMS),
        "rohf": (lambda at, **kw: _rohf(at, basis, **MP, **kw), H3_ATOMS),
        "scan": (lambda at, **kw: _rks(at, basis, "SCAN", **MP, **kw), H2_ATOMS),
    }[route]
    r = run(atoms, with_gradient=True)
    assert r.centring == "mp"
    comps = ((0, 0), (1, 2), (2, 1)) if route != "scan" else ((0, 0), (0, 2), (1, 2))
    worst = _force_vs_fd(
        r.gradient(), atoms, lambda at: run(at).energy, f"MP {route}", comps=comps
    )
    print(f"MP {route} force vs FD: {worst:.2e}")


@pytest.mark.parametrize("route", ["uks", "roks"])
def test_monkhorst_pack_full_stress_tensor_matches_fd(basis, route):
    fn = _uks if route == "uks" else _roks

    def energy(at, lat):
        return fn(at, basis, lat=lat, **MP).energy

    r = fn(H3_ATOMS, basis, with_stress=True, **MP)
    miss = _full_stress_vs_fd(
        r.stress(), H3_A**3, energy, H3_ATOMS, H3_A, f"MP {route}"
    )
    print(f"MP {route} full stress vs FD: {miss:.2e}")


# ─────────────────────────────────────────────────────────── refusals ──


def test_open_shell_refusals_have_exact_wording(basis):
    # k-point ROKS: no meta-GGA (UKS has SCAN / r2SCAN,
    # test_pbc_open_shell_mgga_rsh.py; the ROKS range-separated hybrid energy
    # runs, test_pbc_rsh_open_mgga_derivs.py).
    for f in ("SCAN", "r2SCAN"):
        with pytest.raises(
            ValueError,
            match=rf'^run_roks_kpts: functional "{f}": .*meta-GGA.*\. k-point ROKS '
            r"supports LDA, GGA, global-hybrid and range-separated hybrid functionals "
            r"only \(no meta-GGA for k-point ROKS; run_uks_kpts has SCAN / r2SCAN\)$",
        ):
            _roks(H3_ATOMS, basis, f)
    # Open-shell range-separated hybrids: the UKS force runs on the dense-AFT
    # J/K (test_pbc_rsh_open_mgga_derivs.py); the UKS stress, the UKS force on
    # RS-GDF and every ROKS derivative are refused by name.
    for run, kw, want in (
        (
            _uks,
            {"with_stress": True},
            r"^run_uks_kpts: with_stress: General error: HSE06: the k-point UKS "
            r"range-separated hybrid stress is not implemented \(the UKS stress has no "
            r"range-separated term\); the UKS force is available on the dense-AFT J/K$",
        ),
        (
            _uks,
            {"with_gradient": True, "jk": "rsgdf", "auxbasis": "cc-pvdz-ri"},
            r"^run_uks_kpts: with_gradient: General error: HSE06: the k-point UKS "
            r"range-separated hybrid force runs on the dense-AFT J/K only \(the library "
            r"UKS force has no derivative of the RS-GDF attenuated fit\); use jk = "
            r'"dense"$',
        ),
        (
            _roks,
            {"with_gradient": True},
            r"^run_roks_kpts: with_gradient: General error: HSE06: the k-point ROKS "
            r"range-separated hybrid force and stress are not implemented in the bindings "
            r"\(the library has the dense-AFT ROKS force, kpoint_roks_gradient, unwired; "
            r"the ROKS stress has no range-separated term\); the UKS force is available on "
            r"the dense-AFT J/K$",
        ),
    ):
        with pytest.raises(ValueError, match=want):
            run(H3_ATOMS, basis, "HSE06", **kw)
    # Both derivatives requested: each is gated (the stress refusal names
    # with_stress even though the force alone would run).
    with pytest.raises(ValueError, match=r"^run_uks_kpts: with_stress: "):
        _uks(H3_ATOMS, basis, "HSE06", with_gradient=True, with_stress=True)
    # UKS meta-GGA derivatives: dense-AFT J/K only.
    with pytest.raises(
        ValueError,
        match=r"^run_uks_kpts: with_stress: General error: SCAN: the open-shell \(UKS\) "
        r"meta-GGA force and stress run on the dense-AFT J/K only; use jk = "
        r'"dense"$',
    ):
        _uks(
            H3_ATOMS, basis, "SCAN", with_stress=True, jk="rsgdf", auxbasis="cc-pvdz-ri"
        )
    # Closed-shell range-separated hybrid derivatives: validated functionals
    # only (on either J/K).
    with pytest.raises(
        ValueError,
        match=r"^run_rks_kpts: with_stress: General error: HYB_GGA_XC_WB97X: the "
        r"range-separated hybrid force and stress are validated for HSE06 / CAM-B3LYP",
    ):
        _rks(
            H2_ATOMS,
            basis,
            "HYB_GGA_XC_WB97X",
            with_stress=True,
            jk="rsgdf",
            auxbasis="cc-pvdz-ri",
        )


def test_restricted_open_shell_derivatives_run_on_rsgdf(basis):
    # RS-GDF ROKS / ROHF forces and stress are anchored in the library by FD
    # (tests/pbc_kuxcstress_ro_rsgdf.rs); here only that the wiring accepts them.
    kw = dict(jk="rsgdf", auxbasis="cc-pvdz-ri")
    r = _roks(H3_ATOMS, basis, with_gradient=True, **kw)
    assert r.converged and np.isfinite(r.gradient()).all()
    r = _rohf(H3_ATOMS, basis, with_stress=True, **kw)
    assert r.converged and np.isfinite(r.stress()).all()


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


# ──────────────────────────────────────────────────── ECP cells ──

# HI (closed shell) and the H2I doublet (5 alpha / 4 beta valence) in
# diag(6, 6, 7) Bohr with the LANL2DZ ECP on I (`tests/kks_ecp_support` in
# ferric-pbc, where the per-route FD anchors and ECP mutants live). Here: the
# binding's force and two strain components against central FD of its own
# energies. h = 2.5e-5: at 1e-4 the strain FD carries an h^2 residual of
# ~6e-8 and an open-shell neighbour-mask jump of 1e-6 (library probe).
ECP_H = 2.5e-5
ECP_HI = [("H", (0.3, 0.2, 0.4)), ("I", (0.9, -0.4, 3.3))]
ECP_H2I = [("H", (0.3, 0.2, 0.4)), ("H", (2.0, 0.6, 1.0)), ("I", (0.9, -0.4, 3.3))]
ECP_A = np.diag([6.0, 6.0, 7.0])
# Library floor at h = 2.5e-5: force <= 3.1e-10, dE/deps <= 3.4e-9; the ECP
# parts are 6e-2 (force) and 1.5e-1 (dE/deps), so a dropped ECP term misses
# by ~1e7 x these bars.
ECP_FORCE_BAR = 3e-9
ECP_STRESS_BAR = 3e-8
# 16 Bohr, not 14: on this cell a 14 Bohr neighbour mask puts a one-step
# energy jump inside the xx / yy strain FD at h = 2.5e-5 (library probe).
ECP_GRID = dict(GRID, neighbour_cutoff=16.0 * BOHR_IN_ANGSTROM)


def _ecp_cell(atoms, lat_bohr):
    lines = [str(len(atoms)), "periodic HI / H2I (Angstrom)"]
    for sym, p in atoms:
        # float(): numpy scalars repr as "np.float64(...)" under numpy 2.
        x, y, z = (float(c * BOHR_IN_ANGSTROM) for c in p)
        lines.append(f"{sym} {x!r} {y!r} {z!r}")
    mol = ferric.Molecule.from_xyz_string(
        "\n".join(lines) + "\n", multiplicity=1 if len(atoms) == 2 else 2
    )
    return mol, (np.asarray(lat_bohr) * BOHR_IN_ANGSTROM).tolist()


def _ecp_run(method, atoms, lat_bohr, hi_basis, **kw):
    mol, lat = _ecp_cell(atoms, lat_bohr)
    args = dict(omega=OMEGA, **SCF, **kw)
    if method == "rohf":
        return ferric.run_rohf_kpts(mol, lat, hi_basis, (1, 1, 2), **args)
    run = {
        "rks": ferric.run_rks_kpts,
        "uks": ferric.run_uks_kpts,
        "roks": ferric.run_roks_kpts,
    }[method]
    f = "PBE0" if method == "uks" else "PBE"
    return run(mol, lat, hi_basis, f, (1, 1, 2), **ECP_GRID, **args)


@pytest.mark.parametrize("method", ["rks", "uks", "roks", "rohf"])
def test_ecp_cell_force_and_stress_match_fd(hi_basis, method):
    atoms = ECP_HI if method == "rks" else ECP_H2I
    r = _ecp_run(method, atoms, ECP_A, hi_basis, with_gradient=True, with_stress=True)
    assert r.converged
    g, sigma = r.gradient(), r.stress()
    assert g.shape == (len(atoms), 3) and np.isfinite(g).all()

    def energy_at(at, lat=ECP_A):
        return _ecp_run(method, at, lat, hi_basis).energy

    def moved(i, x, h):
        out = [(sym, list(p)) for sym, p in atoms]
        out[i][1][x] += h
        return out

    ia = len(atoms) - 1
    fmiss = 0.0
    for i, x in ((ia, 0), (0, 2)):
        fd = (energy_at(moved(i, x, ECP_H)) - energy_at(moved(i, x, -ECP_H))) / (
            2.0 * ECP_H
        )
        fmiss = max(fmiss, abs(g[i, x] - fd))
    vol = np.linalg.det(ECP_A)
    smiss = 0.0
    for i, j in ((2, 2), (0, 2)):
        fd = 0.0
        for sgn in (1.0, -1.0):
            e = np.zeros((3, 3))
            e[i, j] = e[j, i] = sgn * ECP_H
            f = np.eye(3) + e
            at = [(sym, tuple(f @ np.array(p))) for sym, p in atoms]
            fd += sgn * energy_at(at, ECP_A @ f.T)
        fd /= 2.0 * ECP_H
        an = vol * (sigma[i, j] if i == j else sigma[i, j] + sigma[j, i])
        smiss = max(smiss, abs(an - fd))
    print(f"ECP {method}: force vs FD {fmiss:.2e}, stress vs FD {smiss:.2e}")
    assert fmiss < ECP_FORCE_BAR, f"{method}: force {fmiss:.3e}"
    assert smiss < ECP_STRESS_BAR, f"{method}: stress {smiss:.3e}"
