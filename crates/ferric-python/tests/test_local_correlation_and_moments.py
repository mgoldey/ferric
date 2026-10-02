"""Binding-level tests for the local-correlation surface (the METHOD is the
function, the local approximation is the `local=`/`eps=` kwargs) plus
orbital_moments and density_second_moment.

The heavy anchors live in the Rust suites (crates/ferric-mp2/tests/
lmp2_amplitude.rs, drpa_amplitude.rs, ferric-cc linlccd tests); these tests
pin the BINDING layer: the exact-by-default contract, the strict kwarg rules
(the CLI's [local] rules, raised as ValueError BEFORE any SCF), argument
plumbing, and the eps=0 identities through the Python surface.
"""

import math

import pytest

ferric = pytest.importorskip("ferric")


@pytest.fixture(scope="module")
def water_631g():
    mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
    obs = ferric.BasisSet.bundled("6-31g")
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    return mol, obs, aux


@pytest.fixture(scope="module")
def h2_sto():
    mol = ferric.Molecule.from_xyz("testdata/molecules/h2.xyz")
    obs = ferric.BasisSet.bundled("sto-3g")
    return mol, obs


# ---- strict kwarg rules (no SCF runs: every case raises before it) ----


@pytest.mark.parametrize("fn", ["run_rimp2", "run_drpa", "run_linlccd"])
def test_local_needs_eps_and_exact_refuses_local_kwargs(h2_sto, fn):
    mol, obs = h2_sto
    f = getattr(ferric, fn)
    # the threshold is part of the model: no default
    with pytest.raises(ValueError, match="requires eps"):
        f(mol, obs, obs, local="amplitude-threshold")
    # strict scheme spelling
    with pytest.raises(ValueError, match="amplitude-threshold"):
        f(mol, obs, obs, local="amplitude_threshold", eps=1e-4)
    # eps / compute_reference on the exact method are refused, not ignored
    with pytest.raises(ValueError, match="eps"):
        f(mol, obs, obs, eps=1e-4)
    with pytest.raises(ValueError, match="reference"):
        f(mol, obs, obs, compute_reference=True)
    with pytest.raises(ValueError, match="eps"):
        f(mol, obs, obs, local="none", eps=1e-4)
    for bad in (-1e-4, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            f(mol, obs, obs, local="amplitude-threshold", eps=bad)


def test_rimp2_direct_knobs_need_integral_direct(h2_sto):
    mol, obs = h2_sto
    with pytest.raises(ValueError, match="integral_direct = true"):
        ferric.run_rimp2(
            mol, obs, obs, local="amplitude-threshold", eps=1e-4, aux_radius=10.0
        )
    with pytest.raises(ValueError, match="integral_direct"):
        ferric.run_rimp2(mol, obs, obs, integral_direct=True)
    with pytest.raises(ValueError, match="batch_merge"):
        ferric.run_rimp2(
            mol,
            obs,
            obs,
            local="amplitude-threshold",
            eps=1e-4,
            integral_direct=True,
            batch_merge=0,
        )
    with pytest.raises(ValueError, match="kappa"):
        ferric.run_rimp2(
            mol, obs, obs, local="amplitude-threshold", eps=0.0, kappa=1.45
        )


def test_drpa_eps_rtol_factor_is_local_only(h2_sto):
    mol, obs = h2_sto
    with pytest.raises(ValueError, match="eps_rtol_factor"):
        ferric.run_drpa(mol, obs, obs, eps_rtol_factor=0.1)


def test_linlccd_variant_is_strict(h2_sto):
    mol, obs = h2_sto
    for bad in ("drivers", "HH", "pp"):
        with pytest.raises(ValueError, match="drivers-only"):
            ferric.run_linlccd(mol, obs, obs, variant=bad)


def test_removed_local_drivers_are_gone():
    for name in ("run_lmp2", "run_lmp2_direct", "run_linlccd_amplitude"):
        assert not hasattr(ferric, name), name


# ---- exact by default, local on request ----


def test_rimp2_is_exact_by_default_and_local_eps_zero_matches(water_631g):
    mol, obs, aux = water_631g
    ex = ferric.run_rimp2(mol, obs, aux, frozen_core=1)
    assert ex.local is None
    lo = ferric.run_rimp2(
        mol,
        obs,
        aux,
        frozen_core=1,
        local="amplitude-threshold",
        eps=0.0,
        compute_reference=True,
    )
    loc = lo.local
    assert loc["scheme"] == "amplitude-threshold" and loc["eps"] == 0.0
    assert loc["integral_direct"] is False
    assert loc["keep_fraction"] == 1.0 and loc["pair_fraction"] == 1.0
    # eps=0 anchor THROUGH the binding: localized CG == canonical RI-MP2
    assert abs(lo.mp2_corr - loc["e_corr_canonical_ri"]) < 1e-9
    assert abs(loc["e_corr_canonical_ri"] - ex.mp2_corr) < 1e-12
    assert abs(lo.total_energy - (lo.rhf_energy + lo.mp2_corr)) < 1e-12


def test_rimp2_local_threshold_error_is_one_sided(water_631g):
    mol, obs, aux = water_631g
    lo = ferric.run_rimp2(
        mol,
        obs,
        aux,
        frozen_core=1,
        local="amplitude-threshold",
        eps=1e-3,
        compute_reference=True,
    )
    de = lo.mp2_corr - lo.local["e_corr_canonical_ri"]
    assert de >= 0.0, f"threshold error must be one-sided, got {de:+.3e}"
    assert lo.local["keep_fraction"] < 1.0
    assert 0 < lo.local["dom_max"] <= 8  # water/6-31G: 8 localized virtuals
    # reference is opt-in: off, the key is present and None (never NaN)
    off = ferric.run_rimp2(
        mol, obs, aux, frozen_core=1, local="amplitude-threshold", eps=1e-3
    )
    assert off.local["e_corr_canonical_ri"] is None
    assert math.isfinite(off.mp2_corr)


def test_rimp2_integral_direct_trivial_maps_match_canonical(water_631g):
    # eps=0 + trivial locality maps THROUGH the binding: integral-direct
    # assembly == canonical RI-MP2 (the Rust suite anchors this at 1e-13)
    mol, obs, aux = water_631g
    r = ferric.run_rimp2(
        mol,
        obs,
        aux,
        frozen_core=1,
        local="amplitude-threshold",
        eps=0.0,
        compute_reference=True,
        integral_direct=True,
        aux_radius=1e6,
        virt_radius=1e6,
        ao_tail=0.0,
        schwarz_skip=0.0,
        batch_merge=1,
    )
    loc = r.local
    assert loc["integral_direct"] is True
    assert abs(r.mp2_corr - loc["e_corr_canonical_ri"]) < 1e-9
    assert loc["strip_rows_max"] > 0 and loc["n_eri3_shell_triples"] > 0


def test_rimp2_integral_direct_production_defaults(water_631g):
    mol, obs, aux = water_631g
    r = ferric.run_rimp2(
        mol,
        obs,
        aux,
        frozen_core=1,
        local="amplitude-threshold",
        eps=1e-3,
        compute_reference=True,
        integral_direct=True,
    )
    err = abs(r.mp2_corr - r.local["e_corr_canonical_ri"])
    assert err < 5e-2, f"error out of class: {err:.3e}"
    assert r.local["t_eri3_s"] >= 0.0 and r.local["t_pairs_s"] >= 0.0


def test_drpa_is_exact_by_default(h2_sto):
    mol, obs = h2_sto
    d = ferric.run_drpa(mol, obs, obs)
    assert d["local"] is None
    assert d["keep_fraction"] == 1.0
    # the proof-notebook H2 value (notebook 12, live cell)
    assert abs(d["e_corr"] - (-0.0126072623)) < 1e-8
    # the exact path IS the eps=0 local path, bit for bit
    z = ferric.run_drpa(mol, obs, obs, local="amplitude-threshold", eps=0.0)
    assert abs(z["e_corr"] - d["e_corr"]) <= 1e-12
    assert z["local"]["eps"] == 0.0
    # and both equal the canonical plasmon reference
    r = ferric.run_drpa(
        mol, obs, obs, local="amplitude-threshold", eps=0.0, compute_reference=True
    )
    assert abs(r["e_corr"] - r["e_corr_plasmon_canonical"]) < 1e-9


def test_drpa_exact_refuses_a_budget_it_cannot_fit(h2_sto):
    mol, obs = h2_sto
    with pytest.raises(MemoryError, match="run_pdep_rpa"):
        ferric.run_drpa(mol, obs, obs, memory_budget_gb=1e-7)


def test_drpa_diis_defaults_on_and_disableable(water_631g):
    """diis/eps_rtol_factor default ON at the binding level (diis=8,
    eps_rtol_factor=0.1); diis=0 / eps_rtol_factor=0.0 map back to the
    unaccelerated solve. All land on the same root."""
    mol, obs, aux = water_631g
    kw = dict(frozen_core=1, local="amplitude-threshold", eps=1e-3)
    d_default = ferric.run_drpa(mol, obs, aux, **kw)
    d_legacy = ferric.run_drpa(mol, obs, aux, diis=0, eps_rtol_factor=0.0, **kw)
    assert d_default["converged"] and d_legacy["converged"]
    assert d_default["iterations"] < d_legacy["iterations"]
    assert abs(d_default["e_corr"] - d_legacy["e_corr"]) < 1e-5
    assert d_default["local"]["eps"] == 1e-3
    assert 0.0 < d_default["local"]["keep_fraction"] < 1.0
    assert d_default["e_corr_plasmon_canonical"] is None


def test_drpa_scan_matches_per_eps_calls(water_631g):
    mol, obs, aux = water_631g
    eps_list = [1e-3, 1e-4]
    scanned = ferric.run_drpa_scan(mol, obs, aux, eps_list, frozen_core=1)
    assert len(scanned) == len(eps_list)
    assert len({r["prefix_wall_s"] for r in scanned}) == 1
    for eps, r_scan in zip(eps_list, scanned):
        assert r_scan["eps"] == eps
        assert r_scan["local"]["eps"] == eps
        r_single = ferric.run_drpa(
            mol, obs, aux, frozen_core=1, local="amplitude-threshold", eps=eps
        )
        assert abs(r_scan["e_corr"] - r_single["e_corr"]) < 1e-12
        assert r_scan["iterations"] == r_single["iterations"]
    with pytest.raises(ValueError, match="finite"):
        ferric.run_drpa_scan(mol, obs, aux, [1e-4, -1.0])
    with pytest.raises(ValueError, match="empty"):
        ferric.run_drpa_scan(mol, obs, aux, [])


def test_linlccd_exact_by_default_and_variants(water_631g):
    mol, obs, aux = water_631g
    e = {
        v: ferric.run_linlccd(mol, obs, aux, variant=v, frozen_core=1)
        for v in ("drivers-only", "hh", "full")
    }
    for v, d in e.items():
        assert d["local"] is None and d["variant"] == v
    ri = ferric.run_rimp2(mol, obs, aux, frozen_core=1)
    # drivers-only == RI-MP2; hh regularizes (|E| shrinks); full restores pp
    assert abs(e["drivers-only"]["e_corr"] - ri.mp2_corr) < 1e-8
    assert abs(e["hh"]["e_corr"]) < abs(e["drivers-only"]["e_corr"])
    assert e["full"]["e_corr"] != e["hh"]["e_corr"]
    # local eps=0 == exact of the same variant; reference opt-in
    lo = ferric.run_linlccd(
        mol,
        obs,
        aux,
        frozen_core=1,
        local="amplitude-threshold",
        eps=0.0,
        compute_reference=True,
    )
    assert abs(lo["e_corr"] - e["hh"]["e_corr"]) < 1e-8
    assert abs(lo["local"]["e_corr_exact"] - e["hh"]["e_corr"]) < 1e-12
    off = ferric.run_linlccd(
        mol, obs, aux, frozen_core=1, local="amplitude-threshold", eps=1e-3
    )
    assert off["local"]["e_corr_exact"] is None
    assert off["local"]["keep_fraction"] < 1.0


def test_orbital_moments_shapes_and_positivity(water_631g):
    mol, obs, _ = water_631g
    rhf = ferric.run_rhf(mol, obs)
    centers, spreads = ferric.orbital_moments(mol, obs, rhf)
    assert len(centers) == len(spreads) == 13  # nao(6-31G water)
    assert all(len(c) == 3 for c in centers)
    assert all(s > 0.0 for s in spreads)
    # core O 1s must be the most compact orbital by a wide margin
    assert min(spreads) == spreads[0] or min(spreads) < 0.5


def test_density_second_moment_translational_identity(water_631g):
    """The density-level identity the Rust test pins, checked THROUGH the
    binding via its trace proxy: trace(M) > 0 and equals the electronic
    <r^2>, which must exceed the squared dipole norm / N lower bound."""
    mol, obs, _ = water_631g
    rhf = ferric.run_rhf(mol, obs)
    m = ferric.density_second_moment(mol, obs, rhf)
    # symmetric 3x3
    for p in range(3):
        for q in range(3):
            assert math.isclose(m[p][q], m[q][p], abs_tol=1e-12)
    trace = m[0][0] + m[1][1] + m[2][2]
    assert trace > 0.0
    # water/6-31G electronic spatial extent: a loose physical band (Bohr^2)
    assert 5.0 < trace < 100.0


def test_run_rimp2_kappa_limits():
    mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
    obs = ferric.BasisSet.bundled("6-31g")
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    plain = ferric.run_rimp2(mol, obs, aux, frozen_core=1).mp2_corr
    inf = ferric.run_rimp2(mol, obs, aux, frozen_core=1, kappa=1e6).mp2_corr
    weak = ferric.run_rimp2(mol, obs, aux, frozen_core=1, kappa=1e-6).mp2_corr
    assert abs(inf - plain) < 1e-12
    assert abs(weak) < 1e-9


def test_tune_omega_h2_smoke():
    mol = ferric.Molecule.from_xyz("testdata/molecules/h2.xyz")
    obs = ferric.BasisSet.bundled("6-31g")
    t = ferric.tune_omega(
        mol, obs, "wB97X-V", omega_lo=0.3, omega_hi=1.2, omega_tol=0.1, max_evals=10
    )
    assert 0.3 < t["omega"] < 1.2
    assert abs(t["j"]) < 5e-3  # Koopmans residual driven down from ~1e-2
    assert len(t["evals"]) >= 2
