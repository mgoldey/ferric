import ferric
import json
import os

import numpy as np
import pytest

TESTDATA = os.path.join(os.path.dirname(__file__), "..", "..", "..", "testdata")


def test_molecule_from_xyz():
    mol = ferric.Molecule.from_xyz(os.path.join(TESTDATA, "molecules", "water.xyz"))
    assert mol.natoms() == 3
    assert mol.nelec() == 10
    vnn = mol.nuclear_repulsion()
    assert abs(vnn - 9.189193229309746) < 1e-6


def test_basis_bundled():
    bs = ferric.BasisSet.bundled("sto-3g")
    assert bs is not None


def test_rhf_water_sto3g():
    mol = ferric.Molecule.from_xyz(os.path.join(TESTDATA, "molecules", "water.xyz"))
    bs = ferric.BasisSet.bundled("sto-3g")
    result = ferric.run_rhf(mol, bs)
    assert result.converged
    ref_path = os.path.join(TESTDATA, "reference", "h2o_sto-3g_rhf.json")
    with open(ref_path) as f:
        ref = json.load(f)
    assert abs(result.energy - ref["energy"]) < 5e-8, (
        f"got {result.energy:.10f}, ref {ref['energy']:.10f}"
    )


def test_rhf_result_arrays():
    mol = ferric.Molecule.from_xyz(os.path.join(TESTDATA, "molecules", "water.xyz"))
    bs = ferric.BasisSet.bundled("sto-3g")
    result = ferric.run_rhf(mol, bs)
    D = result.density()
    assert D.shape == (7, 7)
    eps = result.orbital_energies()
    assert len(eps) == 7


def test_rimp2_water_ccpvdz():
    mol = ferric.Molecule.from_xyz(os.path.join(TESTDATA, "molecules", "water.xyz"))
    bs = ferric.BasisSet.bundled("cc-pvdz")
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    result = ferric.run_rimp2(mol, bs, aux)
    ref_path = os.path.join(TESTDATA, "reference", "h2o_cc-pvdz_rimp2.json")
    with open(ref_path) as f:
        ref = json.load(f)
    assert abs(result.mp2_corr - ref["mp2_corr"]) < 1e-5, (
        f"MP2 corr: got {result.mp2_corr:.10f}, ref {ref['mp2_corr']:.10f}"
    )
    assert abs(result.total_energy - ref["total_energy"]) < 1e-4, (
        f"Total: got {result.total_energy:.10f}, ref {ref['total_energy']:.10f}"
    )
    assert abs(result.rhf_energy - ref["rhf_energy"]) < 1e-6, (
        f"RHF: got {result.rhf_energy:.10f}, ref {ref['rhf_energy']:.10f}"
    )


def _water_ccpvdz_sos():
    mol = ferric.Molecule.from_xyz(os.path.join(TESTDATA, "molecules", "water.xyz"))
    return (mol, ferric.BasisSet.bundled("cc-pvdz"),
            ferric.BasisSet.bundled("cc-pvdz-ri"))


def test_laplace_sos_mp2_reproduces_opposite_spin_energy():
    """c_os = 1.0 must recover the bare opposite-spin MP2 energy.

    This is the hard reference for the method: `run_scs_mp2` reaches e_os by a
    completely different route (canonical denominators, no Laplace transform),
    so agreement is a real cross-check rather than a self-consistency check.
    The residual is Laplace QUADRATURE error, which is why the tolerance is
    tied to n_quad below rather than being a single magic number.
    """
    mol, bs, aux = _water_ccpvdz_sos()
    bare = ferric.run_laplace_sos_mp2(mol, bs, aux, c_os=1.0)
    scs = ferric.run_scs_mp2(mol, bs, aux)
    assert abs(bare.e_os - scs.e_os) < 1e-6, (
        f"Laplace e_os {bare.e_os:.10f} vs canonical {scs.e_os:.10f}"
    )
    # c_os is applied to e_os, and e_os itself is reported UNSCALED.
    assert abs(bare.sos_corr - bare.e_os) < 1e-14
    assert abs(bare.total_energy - (bare.rhf_energy + bare.sos_corr)) < 1e-12


def test_laplace_sos_mp2_quadrature_converges_monotonically():
    """Tightening n_quad must not make the answer worse.

    Catches a silently-capped or mis-tabulated grid: if n_quad were being
    ignored, all three deviations would be identical.
    """
    mol, bs, aux = _water_ccpvdz_sos()
    ref = ferric.run_scs_mp2(mol, bs, aux).e_os
    devs = [
        abs(ferric.run_laplace_sos_mp2(mol, bs, aux, c_os=1.0, n_quad=n).e_os - ref)
        for n in (3, 5, 7)
    ]
    assert devs[0] > devs[1] > devs[2], f"not monotone in n_quad: {devs}"
    assert devs[2] < 1e-6, f"n_quad=7 should be tight, got {devs[2]:.3e}"


def test_laplace_sos_mp2_ao_matches_mo():
    """The AO pseudo-density path computes the SAME quantity as the MO path.

    They share only the quadrature grid, so this pins the pseudo-density
    algebra. Machine-epsilon agreement is what the Rust tests see too -- this
    is not a loosened binding-level tolerance.
    """
    mol, bs, aux = _water_ccpvdz_sos()
    mo = ferric.run_laplace_sos_mp2(mol, bs, aux, formulation="mo")
    ao = ferric.run_laplace_sos_mp2(mol, bs, aux, formulation="ao")
    assert mo.formulation == "mo" and ao.formulation == "ao"
    assert abs(ao.total_energy - mo.total_energy) < 1e-12, (
        f"AO {ao.total_energy:.12f} vs MO {mo.total_energy:.12f}"
    )


def test_laplace_sos_mp2_rejects_bad_config():
    """Unknown formulation / untabulated n_quad must RAISE, not fall back.

    A silent default here would hand back a different method than the one the
    caller asked for. "MO" is the realistic typo: right word, wrong case.
    """
    mol, bs, aux = _water_ccpvdz_sos()
    for bad in ("MO", "AO", "molecular-orbital", ""):
        with pytest.raises(Exception, match="unknown SOS-MP2 formulation"):
            ferric.run_laplace_sos_mp2(mol, bs, aux, formulation=bad)
    with pytest.raises(Exception, match="not tabulated"):
        ferric.run_laplace_sos_mp2(mol, bs, aux, n_quad=4)

    # "ao-sparse" needs a radius, and there is deliberately no default: the
    # usable radius is system-dependent (see the tutorial's measured table).
    with pytest.raises(Exception, match="requires a domain cutoff"):
        ferric.run_laplace_sos_mp2(mol, bs, aux, formulation="ao-sparse")
    for bad_r in (0.0, -1.0):
        with pytest.raises(Exception, match="finite and > 0"):
            ferric.run_laplace_sos_mp2(
                mol, bs, aux, formulation="ao-sparse", domain_cutoff_bohr=bad_r
            )
    # A cutoff on an EXACT formulation is a config error, not a no-op.
    for exact in (None, "mo", "ao"):
        with pytest.raises(Exception, match="only meaningful with formulation"):
            ferric.run_laplace_sos_mp2(
                mol, bs, aux, formulation=exact, domain_cutoff_bohr=8.0
            )


def test_laplace_sos_mp2_ao_sparse_is_exact_when_domain_spans_molecule():
    """The sparse variant is a controlled approximation, not a different method.

    Water is ~2.9 Bohr across, so a 20 Bohr domain contains every AO and the
    restriction is vacuous -- the energy must come back bit-identical to the
    exact AO path. (This is exactly why the *convergence* behaviour is tested
    in Rust on butane instead: on water every useful cutoff is already
    all-encompassing, so a convergence test here would pass vacuously.)
    """
    mol, bs, aux = _water_ccpvdz_sos()
    exact = ferric.run_laplace_sos_mp2(mol, bs, aux, formulation="ao")
    spanning = ferric.run_laplace_sos_mp2(
        mol, bs, aux, formulation="ao-sparse", domain_cutoff_bohr=20.0
    )
    assert spanning.formulation == "ao-sparse"
    assert abs(spanning.total_energy - exact.total_energy) < 1e-12, (
        f"spanning domain {spanning.total_energy:.12f} vs exact "
        f"{exact.total_energy:.12f}"
    )


def test_laplace_sos_mp2_memory_budget_is_enforced_and_not_over_eager():
    """The budget must bind on the AO path -- and an AMPLE budget must still run.

    Both halves matter. A budget that never binds is decorative; a guard that
    over-estimates trains users to inflate budgets until the wall stops
    meaning anything (see the memory-budget notes in CLAUDE.md).
    """
    mol, bs, aux = _water_ccpvdz_sos()
    with pytest.raises(Exception, match="budget"):
        ferric.run_laplace_sos_mp2(
            mol, bs, aux, formulation="ao", memory_budget_gb=0.0005
        )
    ok = ferric.run_laplace_sos_mp2(
        mol, bs, aux, formulation="ao", memory_budget_gb=8.0
    )
    unconstrained = ferric.run_laplace_sos_mp2(mol, bs, aux, formulation="ao")
    assert abs(ok.total_energy - unconstrained.total_energy) < 1e-12, (
        "an ample budget must not perturb the answer"
    )


def test_ksdft_h2o_lda_with_gradient():
    """run_ksdft returns analytic nuclear gradient when with_gradient=True.

    Sanity check: gradient shape (natoms, 3), translational invariance
    (rows sum ≈ 0 to within Becke-grid quadrature noise), and gradient
    is None when with_gradient is not requested.
    """
    mol = ferric.Molecule.from_xyz(os.path.join(TESTDATA, "molecules", "water.xyz"))
    bs = ferric.BasisSet.bundled("sto-3g")

    res_no_grad = ferric.run_ksdft(mol, bs, functional="LDA")
    assert res_no_grad.gradient() is None

    res = ferric.run_ksdft(mol, bs, functional="LDA", with_gradient=True)
    grad = res.gradient()
    assert grad is not None
    assert grad.shape == (mol.natoms(), 3)
    # Translational invariance: Σ_A ∂E/∂R_A ≈ 0 (limited by grid quadrature).
    tot = grad.sum(axis=0)
    assert abs(tot).max() < 5e-3, f"translational drift {tot}"


def test_run_rhf_with_external_point_charge():
    """A +1 point charge 20 Bohr from the molecule perturbs the RHF energy."""
    mol = ferric.Molecule.from_xyz(os.path.join(TESTDATA, "molecules", "water.xyz"))
    bs = ferric.BasisSet.bundled("sto-3g")
    base = ferric.run_rhf(mol, bs)
    perturbed = ferric.run_rhf(mol, bs, point_charges=[(1.0, 0.0, 0.0, 20.0)])
    assert abs(perturbed.energy - base.energy) > 1e-8


def test_run_rhf_with_external_field():
    """A uniform external electric field perturbs the RHF energy."""
    mol = ferric.Molecule.from_xyz(os.path.join(TESTDATA, "molecules", "water.xyz"))
    bs = ferric.BasisSet.bundled("sto-3g")
    base = ferric.run_rhf(mol, bs)
    perturbed = ferric.run_rhf(mol, bs, external_field=(0.0, 0.0, 0.01))
    assert abs(perturbed.energy - base.energy) > 1e-8


_OH_XYZ = "2\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n"


def test_run_uhf_with_external_point_charge():
    """Same perturbation check for the open-shell UHF driver (OH doublet;
    a bare H atom is too symmetric/small a system for this probe)."""
    mol = ferric.Molecule.from_xyz_string(_OH_XYZ, charge=0, multiplicity=2)
    bs = ferric.BasisSet.bundled("sto-3g")
    base = ferric.run_uhf(mol, bs)
    perturbed = ferric.run_uhf(mol, bs, point_charges=[(1.0, 0.0, 0.0, 20.0)])
    assert abs(perturbed.energy - base.energy) > 1e-8


def test_run_uhf_with_external_field():
    mol = ferric.Molecule.from_xyz_string(_OH_XYZ, charge=0, multiplicity=2)
    bs = ferric.BasisSet.bundled("sto-3g")
    base = ferric.run_uhf(mol, bs)
    perturbed = ferric.run_uhf(mol, bs, external_field=(0.0, 0.0, 0.01))
    assert abs(perturbed.energy - base.energy) > 1e-8


def test_run_ksdft_with_external_point_charge():
    """Same perturbation check for the KS-DFT driver (also exercises the
    gradient path picking up external_potential, not just the energy)."""
    mol = ferric.Molecule.from_xyz(os.path.join(TESTDATA, "molecules", "water.xyz"))
    bs = ferric.BasisSet.bundled("sto-3g")
    base = ferric.run_ksdft(mol, bs, functional="LDA")
    perturbed = ferric.run_ksdft(
        mol, bs, functional="LDA", point_charges=[(1.0, 0.0, 0.0, 20.0)]
    )
    assert abs(perturbed.total_energy - base.total_energy) > 1e-8


def test_run_ksdft_with_external_field():
    mol = ferric.Molecule.from_xyz(os.path.join(TESTDATA, "molecules", "water.xyz"))
    bs = ferric.BasisSet.bundled("sto-3g")
    base = ferric.run_ksdft(mol, bs, functional="LDA")
    perturbed = ferric.run_ksdft(mol, bs, functional="LDA", external_field=(0.0, 0.0, 0.01))
    assert abs(perturbed.total_energy - base.total_energy) > 1e-8


def test_run_ksdft_gradient_with_external_field_runs():
    """with_gradient=True must not choke once external_potential flows into
    ks_gradient_closed (replaces the Task-8 placeholder `None`)."""
    mol = ferric.Molecule.from_xyz(os.path.join(TESTDATA, "molecules", "water.xyz"))
    bs = ferric.BasisSet.bundled("sto-3g")
    res = ferric.run_ksdft(
        mol, bs, functional="LDA", with_gradient=True,
        external_field=(0.0, 0.0, 0.01),
    )
    grad = res.gradient()
    assert grad is not None
    assert grad.shape == (mol.natoms(), 3)


def test_oversized_ccsd_t_raises_not_oom():
    """M2 fail-fast guard: a deliberately-oversized run_ccsd_t must raise a
    Python exception (RuntimeError carrying the GB numbers), NOT walk into a
    TB-scale allocation that OOM-kills the interpreter. The tiny budget is
    passed explicitly via the memory_budget_gb kwarg (explicit beats env in
    ferric_core::memory::resolve_budget_bytes)."""
    mol = ferric.Molecule.from_xyz(os.path.join(TESTDATA, "molecules", "water.xyz"))
    obs = ferric.BasisSet.bundled("cc-pvdz")
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    try:
        ferric.run_ccsd_t(mol, obs, aux, memory_budget_gb=1e-6)
    except RuntimeError as e:
        msg = str(e)
        # The CCSD stage (runs before (T)) or the (T) stage fires first —
        # either way the process survives and the message names the budget.
        assert "budget is" in msg, msg
        assert "CCSD" in msg, msg
        return
    raise AssertionError("run_ccsd_t did not raise under a tiny memory budget")

BOHR_PER_ANG = 1.0 / 0.52917721092


def test_esp_at_points_far_field_is_dipolar():
    """`esp_at_points` must reproduce the classical far field.

    This is the physics anchor for the binding: at large r a NEUTRAL molecule's
    potential is dipole-dominated and decays as 1/r^2, so doubling r must
    quarter V. A wrong sign, a missing nuclear term, or a units error all break
    this immediately -- unlike a self-consistency check against ferric's own
    numbers, which would pass for all three.

    Note `esp_at_points` includes the nuclear Z/r contribution and therefore
    DIVERGES at a nucleus; it is not interchangeable with `esp_at_atoms`, which
    excludes the self-term.
    """
    mol = ferric.Molecule.from_xyz(os.path.join(TESTDATA, "molecules", "water.xyz"))
    bs = ferric.BasisSet.bundled("sto-3g")
    r = ferric.run_rhf(mol, bs)

    pts = np.array([[0.0, 0.0, z] for z in (10.0, 20.0, 40.0, 80.0)])
    v = np.array(ferric.esp_at_points(mol, bs, r, pts))
    assert np.all(np.isfinite(v)), f"non-finite ESP: {v}"

    ratios = [v[i] / v[i + 1] for i in range(len(v) - 1)]
    for k, ratio in enumerate(ratios):
        assert 3.0 < ratio < 5.0, (
            f"far-field decay is not 1/r^2 at step {k}: ratio {ratio:.3f}, "
            f"values {v}. ~2 would mean a net charge (wrong nuclear term); "
            f"~8 would mean quadrupole-dominated."
        )


def test_esp_at_points_rejects_bad_shape():
    """(N, 3) is required; anything else must raise, not silently reinterpret."""
    mol = ferric.Molecule.from_xyz(os.path.join(TESTDATA, "molecules", "water.xyz"))
    bs = ferric.BasisSet.bundled("sto-3g")
    r = ferric.run_rhf(mol, bs)
    with pytest.raises(Exception, match=r"shape \(N, 3\)"):
        ferric.esp_at_points(mol, bs, r, np.zeros((3, 2)))

