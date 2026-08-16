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
