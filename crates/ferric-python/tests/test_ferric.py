import ferric
import json
import os

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
