"""`AoIntegrals`: ferric's AO integrals and J/K build as numpy arrays.

The anchors are independent constructions, not self-consistency:

* a ~20-line numpy SCF on `hcore()` + `jk()` reproduces `run_rhf`, which
  assembles its Fock matrix inside Rust;
* `jk()` (the production Schwarz-screened direct build) equals J/K contracted
  from `eri()` (a separate dense quartet loop with its own 8-fold scatter);
* `erf + erfc == coulomb` for every operator-taking method;
* density fitting from `eri3()`/`eri2()` approaches the exact Coulomb energy
  from BELOW (the Coulomb-metric fit is variational), so a wrong metric or a
  transposed tensor shows up as a sign, not only a magnitude.
"""

from __future__ import annotations

import numpy as np
import pytest

import ferric

WATER = "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n"


def _water(basis: str):
    mol = ferric.Molecule.from_xyz_string(WATER)
    bs = ferric.BasisSet.bundled(basis)
    return mol, bs, ferric.AoIntegrals(mol, bs)


def _scf_density(basis: str) -> tuple[ferric.AoIntegrals, np.ndarray]:
    """A converged density from run_rhf, for tests that only need some D."""
    mol, bs, ints = _water(basis)
    return ints, ferric.run_rhf(mol, bs, energy_conv=1e-10, density_conv=1e-9).density()


def _numpy_rhf(ints: ferric.AoIntegrals) -> tuple[float, np.ndarray]:
    """Plain Roothaan iterations with no DIIS, built only from the blocks."""
    s, h = ints.overlap(), ints.hcore()
    w, v = np.linalg.eigh(s)
    x = v / np.sqrt(w)  # S^{-1/2}-style orthogonalizer
    nocc = ints.nelec // 2
    d = np.zeros_like(s)
    for it in range(200):
        j, k = ints.jk(d)
        f = h + j - 0.5 * k
        _, c = np.linalg.eigh(x.T @ f @ x)
        c = x @ c
        d_new = 2.0 * c[:, :nocc] @ c[:, :nocc].T
        if np.abs(d_new - d).max() < 1e-10:
            d = d_new
            break
        d = 0.5 * (d + d_new) if it > 100 else d_new
    else:
        pytest.fail("numpy Roothaan loop did not converge")
    j, k = ints.jk(d)
    e = 0.5 * np.sum(d * (2.0 * h + j - 0.5 * k)) + ints.nuclear_repulsion()
    return e, d


@pytest.mark.parametrize("basis", ["sto-3g", "cc-pvdz"])
def test_numpy_scf_on_the_blocks_reproduces_run_rhf(basis):
    mol, bs, ints = _water(basis)
    e_np, d = _numpy_rhf(ints)
    r = ferric.run_rhf(mol, bs, energy_conv=1e-10, density_conv=1e-9)
    assert abs(e_np - r.energy) < 1e-9
    assert np.trace(d @ ints.overlap()) == pytest.approx(ints.nelec, abs=1e-9)


def test_numpy_scf_includes_the_ecp():
    # Iodine carries a def2 ECP: hcore() must add V_ECP and nelec must count
    # valence electrons only, exactly as run_rhf does.
    mol = ferric.Molecule.from_xyz_string("2\n\nH 0 0 0\nI 0 0 1.609\n")
    bs = ferric.BasisSet.bundled("def2-svp")
    ints = ferric.AoIntegrals(mol, bs)
    assert ints.ecp() is not None
    assert ints.nelec == 1 + 25
    np.testing.assert_allclose(
        ints.hcore(), ints.kinetic() + ints.nuclear() + ints.ecp(), atol=1e-12
    )
    e_np, _ = _numpy_rhf(ints)
    r = ferric.run_rhf(mol, bs, energy_conv=1e-10, density_conv=1e-9)
    assert abs(e_np - r.energy) < 1e-8


def test_no_ecp_basis_reports_none():
    _, _, ints = _water("sto-3g")
    assert ints.ecp() is None
    np.testing.assert_allclose(
        ints.hcore(), ints.kinetic() + ints.nuclear(), atol=1e-14
    )


@pytest.mark.parametrize("basis", ["sto-3g", "cc-pvdz"])
def test_jk_matches_the_dense_eri_contraction(basis):
    ints, d = _scf_density(basis)
    eri = ints.eri()
    j, k = ints.jk(d)
    np.testing.assert_allclose(j, np.einsum("ijkl,kl->ij", eri, d), atol=1e-10)
    np.testing.assert_allclose(k, np.einsum("ikjl,kl->ij", eri, d), atol=1e-10)


def test_eri_has_eightfold_symmetry_and_positive_diagonal():
    _, _, ints = _water("cc-pvdz")
    eri = ints.eri()
    for perm in [(1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)]:
        np.testing.assert_array_equal(eri, eri.transpose(perm))
    n = ints.nao
    pair = eri.reshape(n * n, n * n)
    # (μν|λσ) as a pair matrix is positive semidefinite (Coulomb kernel).
    assert np.linalg.eigvalsh(pair).min() > -1e-10


def test_erf_plus_erfc_is_coulomb():
    _, _, ints = _water("cc-pvdz")
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    w = 0.4
    np.testing.assert_allclose(
        ints.eri("erf", omega=w) + ints.eri("erfc", omega=w), ints.eri(), atol=1e-12
    )
    np.testing.assert_allclose(
        ints.eri3(aux, "erf", omega=w) + ints.eri3(aux, "erfc", omega=w),
        ints.eri3(aux),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        ints.eri2(aux, "erf", omega=w) + ints.eri2(aux, "erfc", omega=w),
        ints.eri2(aux),
        atol=1e-12,
    )
    _, d = _scf_density("cc-pvdz")
    j_lr, k_lr = ints.jk(d, "erf", omega=w)
    j_sr, k_sr = ints.jk(d, "erfc", omega=w)
    j, k = ints.jk(d)
    np.testing.assert_allclose(j_lr + j_sr, j, atol=1e-10)
    np.testing.assert_allclose(k_lr + k_sr, k, atol=1e-10)


def test_density_fitted_coulomb_energy_is_a_lower_bound():
    ints, d = _scf_density("cc-pvdz")
    j, _ = ints.jk(d)
    e_j = 0.5 * np.sum(d * j)
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    b = ints.eri3(aux)  # (P|μν)
    metric = ints.eri2(aux)  # (P|Q)
    assert b.shape == (metric.shape[0], ints.nao, ints.nao)
    np.testing.assert_allclose(b, b.transpose(0, 2, 1), atol=0)
    gamma = np.einsum("pij,ij->p", b, d)
    e_j_fit = 0.5 * gamma @ np.linalg.solve(metric, gamma)
    assert e_j_fit <= e_j + 1e-12
    assert (e_j - e_j_fit) / e_j < 1e-4


def test_dipole_shifts_by_origin_times_overlap():
    _, _, ints = _water("cc-pvdz")
    origin = (0.3, -1.1, 2.0)
    d0, d1 = ints.dipole(), ints.dipole(origin)
    s = ints.overlap()
    for axis in range(3):
        np.testing.assert_allclose(d1[axis], d0[axis] - origin[axis] * s, atol=1e-12)


def test_ao_bookkeeping():
    _, _, ints = _water("cc-pvdz")
    offsets = ints.shell_offsets()
    atoms = ints.ao_atoms()
    assert offsets[0] == 0 and offsets[-1] == len(atoms) == ints.nao == 24
    assert len(ints.shell_atoms()) == len(offsets) - 1
    for i, atom in enumerate(ints.shell_atoms()):
        assert set(atoms[offsets[i] : offsets[i + 1]]) == {atom}
    assert sorted(set(atoms)) == [0, 1, 2]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"operator": "yukawa"}, "unknown operator"),
        ({"operator": "erf"}, "positive finite omega"),
        ({"operator": "erfc", "omega": -1.0}, "positive finite omega"),
        ({"operator": "coulomb", "omega": 0.4}, "takes no omega"),
    ],
)
def test_operator_kwargs_are_strict(kwargs, match):
    _, _, ints = _water("sto-3g")
    with pytest.raises(ValueError, match=match):
        ints.eri(**kwargs)
    with pytest.raises(ValueError, match=match):
        ints.jk(np.zeros((ints.nao, ints.nao)), **kwargs)


def test_jk_rejects_bad_densities():
    _, _, ints = _water("sto-3g")
    n = ints.nao
    with pytest.raises(ValueError, match="shape"):
        ints.jk(np.zeros((n, n + 1)))
    asym = np.zeros((n, n))
    asym[0, 1] = 1.0
    with pytest.raises(ValueError, match="not symmetric"):
        ints.jk(asym)
    with pytest.raises(ValueError, match="NaN or inf"):
        ints.jk(np.full((n, n), np.nan))


def test_dense_eri_refuses_above_the_memory_cap():
    _, _, ints = _water("cc-pvdz")
    with pytest.raises(ValueError, match="max_gb"):
        ints.eri(max_gb=1e-6)


# ── einsum ──

_RNG = np.random.default_rng(7)

EINSUM_CASES = [
    ("ij,jk->ik", (3, 4), (4, 5)),
    ("ijcd,abcd->ijab", (2, 3, 4, 5), (3, 2, 4, 5)),
    ("ij,kl->ijkl", (2, 3), (4, 2)),
    ("kij,kij->ij", (5, 3, 4), (5, 3, 4)),
    ("pij,qij->pq", (3, 4, 5), (2, 4, 5)),
    ("ijk,kl->jil", (2, 3, 4), (4, 5)),
    ("kij,kji->ji", (4, 2, 3), (4, 3, 2)),
]


@pytest.mark.parametrize("spec, sa, sb", EINSUM_CASES)
def test_einsum_matches_numpy(spec, sa, sb):
    a, b = _RNG.standard_normal(sa), _RNG.standard_normal(sb)
    got = ferric.einsum(spec, a, b, scale=-0.5)
    np.testing.assert_allclose(
        got, -0.5 * np.einsum(spec, a, b), rtol=1e-13, atol=1e-13
    )


def test_einsum_scalar_spec_returns_a_float():
    a, b = _RNG.standard_normal((3, 4)), _RNG.standard_normal((3, 4))
    got = ferric.einsum("ij,ij->", a, b)
    assert isinstance(got, float)
    assert got == pytest.approx(np.sum(a * b), rel=1e-13)


def test_einsum_accepts_non_contiguous_views():
    a = _RNG.standard_normal((5, 4)).T  # Fortran-ordered view
    b = _RNG.standard_normal((6, 5))[::2]  # strided rows
    np.testing.assert_allclose(ferric.einsum("ij,kj->ik", a, b), a @ b.T, rtol=1e-13)


def test_einsum_builds_j_from_the_dense_eri():
    ints, d = _scf_density("sto-3g")
    j, _ = ints.jk(d)
    np.testing.assert_allclose(
        ferric.einsum("ijkl,kl->ij", ints.eri(), d), j, atol=1e-10
    )


@pytest.mark.parametrize(
    "spec, sa, sb, match",
    [
        ("ij,jk", (3, 4), (4, 5), "->"),
        ("ii,ij->j", (3, 3), (3, 4), "repeated"),
        ("ij,jk->ki", (3, 4), (4, 5), "listed as"),
        ("ij,jk->ik", (3, 4), (5, 6), "extent"),
        ("ijk,jk->i", (3, 4), (4, 5), "axes"),
    ],
)
def test_einsum_errors_are_value_errors(spec, sa, sb, match):
    with pytest.raises(ValueError, match=match):
        ferric.einsum(spec, np.ones(sa), np.ones(sb))


@pytest.mark.parametrize(
    "spec, sa, sb",
    [
        ("ij,jk->ik", (0, 3), (3, 4)),
        ("ij,jk->ik", (2, 0), (0, 4)),
        ("kij,kij->ij", (0, 2, 3), (0, 2, 3)),
        ("ij,ij->", (0, 3), (0, 3)),
    ],
)
def test_einsum_zero_extent_operands_give_zeros(spec, sa, sb):
    # Used to panic in the GEMM reshape (CodeRabbit, PR #594).
    got = ferric.einsum(spec, np.ones(sa), np.ones(sb), scale=2.0)
    ref = 2.0 * np.einsum(spec, np.ones(sa), np.ones(sb))
    assert np.shape(got) == np.shape(ref)
    np.testing.assert_array_equal(got, ref)
