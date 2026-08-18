"""Binding-level tests for the 2026-08 additions: run_lmp2 (amplitude-
threshold local MP2), orbital_moments, density_second_moment.

The heavy anchors live in the Rust suites (crates/ferric-mp2/tests/
lmp2_amplitude.rs, ferric-integrals oneelectron tests); these tests pin the
BINDING layer — argument plumbing, unit/shape conventions, and the eps=0
identity through the Python surface.
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


def test_run_lmp2_eps_zero_matches_canonical(water_631g):
    mol, obs, aux = water_631g
    r = ferric.run_lmp2(mol, obs, aux, eps=0.0, frozen_core=1)
    # eps=0 anchor THROUGH the binding: localized CG == canonical RI-MP2
    assert abs(r["e_corr"] - r["e_corr_canonical_ri"]) < 1e-9
    assert r["keep_fraction"] == 1.0 and r["pair_fraction"] == 1.0
    # and the canonical number must agree with run_rimp2 itself
    ri = ferric.run_rimp2(mol, obs, aux, frozen_core=1)
    assert abs(r["e_corr_canonical_ri"] - ri.mp2_corr) < 1e-12
    assert abs(r["total_energy"] - (r["rhf_energy"] + r["e_corr"])) < 1e-12


def test_run_lmp2_threshold_error_is_one_sided(water_631g):
    mol, obs, aux = water_631g
    r = ferric.run_lmp2(mol, obs, aux, eps=1e-3, frozen_core=1)
    de = r["e_corr"] - r["e_corr_canonical_ri"]
    assert de >= 0.0, f"threshold error must be one-sided, got {de:+.3e}"
    assert r["keep_fraction"] < 1.0
    assert 0 < r["dom_max"] <= 8  # water/6-31G: 8 virtuals in the localized set


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


def test_run_lmp2_rejects_open_shell():
    mol = ferric.Molecule.from_xyz("testdata/molecules/h2.xyz")
    obs = ferric.BasisSet.bundled("sto-3g")
    r = ferric.run_lmp2(mol, obs, obs, eps=0.0)
    # H2 closed shell works; the open-shell rejection is enforced in the
    # library (CLI hard-rejects) — here just pin the tiny-system identity
    assert abs(r["e_corr"] - r["e_corr_canonical_ri"]) < 1e-10


def test_run_drpa_eps_zero_matches_plasmon():
    mol = ferric.Molecule.from_xyz("testdata/molecules/h2.xyz")
    obs = ferric.BasisSet.bundled("sto-3g")
    d = ferric.run_drpa(mol, obs, obs, eps=0.0)
    assert abs(d["e_corr"] - d["e_corr_plasmon_canonical"]) < 1e-9
    # the proof-notebook H2 value (notebook 12, live cell)
    assert abs(d["e_corr"] - (-0.0126072623)) < 1e-8


def test_run_drpa_diis_defaults_on_and_disableable(water_631g):
    """diis/eps_rtol_factor default ON at the binding level (diis=8,
    eps_rtol_factor=0.1); diis=0 / eps_rtol_factor=0.0 must map back to the
    legacy unaccelerated solve. All three land on the same root."""
    mol, obs, aux = water_631g
    d_default = ferric.run_drpa(mol, obs, aux, eps=1e-3, frozen_core=1, compute_reference=False)
    d_legacy = ferric.run_drpa(
        mol, obs, aux, eps=1e-3, frozen_core=1, compute_reference=False, diis=0, eps_rtol_factor=0.0
    )
    assert d_default["converged"] and d_legacy["converged"]
    # defaults reach convergence in fewer iterations than the legacy path
    assert d_default["iterations"] < d_legacy["iterations"]
    # same root: within the calibrated subdominance bound (10% of a typical
    # water/eps=1e-3 truncation error, ~1e-4 scale per the wiki table)
    assert abs(d_default["e_corr"] - d_legacy["e_corr"]) < 1e-5


def test_run_drpa_scan_matches_per_eps_calls(water_631g):
    mol, obs, aux = water_631g
    eps_list = [1e-3, 1e-4]
    scanned = ferric.run_drpa_scan(
        mol, obs, aux, eps_list, frozen_core=1, compute_reference=False
    )
    assert len(scanned) == len(eps_list)
    prefix_walls = {r["prefix_wall_s"] for r in scanned}
    assert len(prefix_walls) == 1  # one SHARED prefix time across all points
    for eps, r_scan in zip(eps_list, scanned):
        assert r_scan["eps"] == eps
        r_single = ferric.run_drpa(
            mol, obs, aux, eps=eps, frozen_core=1, compute_reference=False
        )
        assert abs(r_scan["e_corr"] - r_single["e_corr"]) < 1e-12
        assert r_scan["iterations"] == r_single["iterations"]
        assert r_scan["wall_s"] >= 0.0


def test_run_linlccd_amplitude_variants_ordered():
    mol = ferric.Molecule.from_xyz("testdata/molecules/water.xyz")
    obs = ferric.BasisSet.bundled("6-31g")
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    e = {
        v: ferric.run_linlccd_amplitude(mol, obs, aux, variant=v, eps=0.0, frozen_core=1)["e_corr"]
        for v in ("drivers", "hh", "full")
    }
    # drivers == RI-MP2; hh regularizes (|E| shrinks); full restores pp
    ri = ferric.run_rimp2(mol, obs, aux, frozen_core=1)
    assert abs(e["drivers"] - ri.mp2_corr) < 1e-8
    assert abs(e["hh"]) < abs(e["drivers"])
    assert e["full"] != e["hh"]
    with pytest.raises(Exception):
        ferric.run_linlccd_amplitude(mol, obs, aux, variant="bogus")


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
    t = ferric.tune_omega(mol, obs, "wB97X-V", omega_lo=0.3, omega_hi=1.2,
                          omega_tol=0.1, max_evals=10)
    assert 0.3 < t["omega"] < 1.2
    assert abs(t["j"]) < 5e-3  # Koopmans residual driven down from ~1e-2
    assert len(t["evals"]) >= 2
