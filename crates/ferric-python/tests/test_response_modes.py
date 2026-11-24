"""ferric.response_modes: RPA dielectric eigenmodes as aux-basis densities.

Anchors (independent of ferric's dielectric code; every reference is built in
numpy from ferric's raw integrals: AoIntegrals.eri3/eri2/dipole/overlap and the
SCF orbitals):

1. Mode-sum polarizability from the PHYSICAL coefficients. With R = (P|ia),
   V = (P|Q), D_w = Delta/(Delta^2 + w^2) and s[a, d] = 4 c_a^T R D_w mu_d,
       alpha(iw) = 4 mu^T D_w mu - sum_a s_a s_a^T / lambda_a      (all modes)
   must equal the pair-space RI-dRPA polarizability
       alpha(iw) = 4 mu^T Delta^1/2 (Delta^2 + w^2 + 4 Delta^1/2 K Delta^1/2)^-1 Delta^1/2 mu,
       K = R^T V^-1 R,
   which never forms a dielectric. The identity holds only if c are the
   generalized eigenvectors of (V + Pi) c = lambda V c with c^T V c = 1, so a
   wrong metric transform of c (c = F u instead of F^T u) fails it by O(1).
   Bar: a 95x95 SPD solve and an 84-mode sum, each O(1) numbers with
   condition numbers < 1e3, round to ~1e-13 relative; asserted 1e-9 relative.
2. The modes' grid functions are the integral functions: the aux overlap
   S_PQ from libint (AoIntegrals(atom, aux).overlap()) equals a numpy
   quadrature of ferric.eval_ao_on_points over the same functions on one atom
   (s through f). Quadrature: 16-point Gauss-Legendre in cos(theta) and
   32-point trapezoid in phi integrate the degree <= 6 angular products
   exactly; 300-point Gauss-Chebyshev (second kind) radial grid on a
   Becke r = (1 + x)/(1 - x) map converges the Gaussian radial products of
   cc-pVDZ-RI oxygen to ~1e-12. Bar 1e-9 absolute on unit-diagonal S.
3. write_response_mode_cube writes sum_P c_P chi_P(r) / l2_norm at the grid
   points, i.e. eval_ao_on_points @ c / l2 there. The cube format prints 5
   significant digits, so the bar is 1e-5 relative to the largest value.

Measured (water/cc-pVDZ, cc-pVDZ-RI): anchor 1 max|delta alpha| 3.6e-15
(omega 0) and 8.9e-16 (omega 0.5) a.u. on |alpha| ~ 4; anchor 2 1.2e-12.

MUTATION (run by hand): the response_modes back-transform
``inter.v_inv_sqrt.t().dot(&u)`` -> ``inter.v_inv_sqrt.dot(&u)`` fails anchor 1
at both frequencies and the cube test.
"""

from __future__ import annotations

import numpy as np
import pytest

import ferric

pytestmark = pytest.mark.skipif(
    not hasattr(ferric, "response_modes"),
    reason="needs ferric.response_modes",
)

WATER = "3\n\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.469\nH 0.0 -0.757 -0.469\n"


@pytest.fixture(scope="module")
def water():
    mol = ferric.Molecule.from_xyz_string(WATER)
    bs = ferric.BasisSet.bundled("cc-pvdz")
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    res = ferric.run_rhf(mol, bs)
    return mol, bs, aux, res


def _pair_space(mol, bs, aux, res):
    ints = ferric.AoIntegrals(mol, bs)
    nocc = ints.nelec // 2
    c = np.asarray(res.mo_coefficients())
    eps = np.asarray(res.orbital_energies())
    co, cv = c[:, :nocc], c[:, nocc:]
    r = np.einsum("pmn,mi,na->pia", ints.eri3(aux), co, cv).reshape(
        -1, co.shape[1] * cv.shape[1]
    )
    v = ints.eri2(aux)
    de = (eps[nocc:][None, :] - eps[:nocc][:, None]).reshape(-1)
    mu = np.stack([co.T @ d @ cv for d in ints.dipole()]).reshape(3, -1)
    return r, v, de, mu


@pytest.mark.parametrize("omega", [0.0, 0.5])
def test_mode_sum_from_physical_coefficients_is_the_ri_drpa_polarizability(
    water, omega
):
    mol, bs, aux, res = water
    naux = aux.n_functions(mol)
    modes = ferric.response_modes(mol, bs, aux, res, n_modes=naux, omega=omega)
    assert modes.n_modes == naux and modes.naux == naux
    r, v, de, mu = _pair_space(mol, bs, aux, res)

    d_w = de / (de**2 + omega**2)
    c = np.asarray(modes.coefficients)
    lam = np.asarray(modes.eigenvalues)
    s = 4.0 * c.T @ (r @ (d_w[:, None] * mu.T))  # (naux, 3)
    alpha_bare = 4.0 * (mu * d_w) @ mu.T
    alpha_modes = alpha_bare - (s / lam[:, None]).T @ s

    k = r.T @ np.linalg.solve(v, r)
    sq = np.sqrt(de)
    a = np.diag(de**2 + omega**2) + 4.0 * sq[:, None] * k * sq[None, :]
    rhs = sq[:, None] * mu.T
    alpha_ref = 4.0 * rhs.T @ np.linalg.solve(a, rhs)

    scale = np.abs(alpha_ref).max()
    assert np.abs(alpha_modes - alpha_ref).max() < 1e-9 * scale
    assert np.abs(np.asarray(modes.alpha_bare) - alpha_bare).max() < 1e-9 * scale
    assert (
        np.abs(np.asarray(modes.polarizability_from_modes()) - alpha_ref).max()
        < 1e-9 * scale
    )
    assert np.abs(np.asarray(modes.field_couplings) - s).max() < 1e-9 * np.abs(s).max()
    # Unit Coulomb self-energy and L2 norms from libint's aux overlap.
    assert np.abs(c.T @ v @ c - np.eye(naux)).max() < 1e-9
    s_aux = ferric.AoIntegrals(mol, aux).overlap()
    l2 = np.sqrt(np.einsum("pa,pq,qa->a", c, s_aux, c))
    assert np.allclose(np.asarray(modes.l2_norms), l2, rtol=1e-10, atol=0)


def test_leading_modes_are_a_prefix_of_the_full_set(water):
    mol, bs, aux, res = water
    few = ferric.response_modes(mol, bs, aux, res, n_modes=4)
    allm = ferric.response_modes(mol, bs, aux, res, n_modes=10_000)
    assert few.n_modes == 4 and allm.n_modes == aux.n_functions(mol)
    assert np.allclose(
        few.eigenvalues, np.asarray(allm.eigenvalues)[:4], rtol=0, atol=1e-12
    )
    assert np.all(np.diff(np.asarray(allm.eigenvalues)) <= 0)
    assert np.asarray(allm.eigenvalues)[-1] >= 1.0 - 1e-12


def _single_centre_quadrature(nr=300, nt=16, nphi=32):
    x, wx = np.polynomial.legendre.leggauss(nt)  # cos(theta)
    phi = 2.0 * np.pi * np.arange(nphi) / nphi
    k = np.arange(1, nr + 1)
    t = np.cos(k * np.pi / (nr + 1))
    wt = np.pi / (nr + 1) * np.sin(k * np.pi / (nr + 1)) ** 2 / np.sqrt(1 - t**2)
    rr = (1 + t) / (1 - t)
    wr = wt * 2.0 / (1 - t) ** 2 * rr**2
    st = np.sqrt(1 - x**2)
    dirs = np.stack(
        [
            np.outer(st, np.cos(phi)),
            np.outer(st, np.sin(phi)),
            np.outer(x, np.ones(nphi)),
        ],
        axis=-1,
    ).reshape(-1, 3)
    wang = np.outer(wx, np.full(nphi, 2.0 * np.pi / nphi)).reshape(-1)
    pts = (rr[:, None, None] * dirs[None, :, :]).reshape(-1, 3)
    w = (wr[:, None] * wang[None, :]).reshape(-1)
    return pts, w


def test_grid_aux_functions_are_the_integral_functions():
    atom = ferric.Molecule.from_xyz_string("1\n\nO 0 0 0\n", 0, 3)
    aux = ferric.BasisSet.bundled("cc-pvdz-ri")
    assert max(sh[0] for sh in aux.shells(8)) == 3  # covers s..f
    pts, w = _single_centre_quadrature()
    chi = np.asarray(ferric.eval_ao_on_points(atom, aux, np.ascontiguousarray(pts)))
    s_grid = (chi * w[:, None]).T @ chi
    s_libint = ferric.AoIntegrals(atom, aux).overlap()
    assert np.abs(s_grid - s_libint).max() < 1e-9


def _read_cube(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    natoms, *origin = lines[2].split()
    natoms = abs(int(natoms))
    axes = [lines[3 + i].split() for i in range(3)]
    shape = [int(a[0]) for a in axes]
    steps = np.array([[float(x) for x in a[1:]] for a in axes])
    vals = np.array(" ".join(lines[6 + natoms :]).split(), dtype=float)
    return np.array(origin, dtype=float), shape, steps, vals.reshape(shape)


def test_cube_holds_the_l2_normalised_mode_density(water, tmp_path):
    mol, bs, aux, res = water
    modes = ferric.response_modes(mol, bs, aux, res, n_modes=3)
    c = np.asarray(modes.coefficients)
    l2 = np.asarray(modes.l2_norms)
    for idx in range(3):
        path = tmp_path / f"mode{idx}.cube"
        ferric.write_response_mode_cube(
            str(path), mol, aux, modes, idx, spacing=0.5, padding=3.0
        )
        origin, shape, steps, data = _read_cube(path)
        ii, jj, kk = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
        pts = origin + np.stack([ii, jj, kk], axis=-1).reshape(-1, 3) @ steps
        ref = (
            np.asarray(ferric.eval_ao_on_points(mol, aux, np.ascontiguousarray(pts)))
            @ c[:, idx]
            / l2[idx]
        )
        assert np.abs(data.reshape(-1) - ref).max() < 1e-5 * np.abs(ref).max()
    with pytest.raises(IndexError):
        ferric.write_response_mode_cube(str(tmp_path / "x.cube"), mol, aux, modes, 3)
    with pytest.raises(ValueError):
        ferric.write_response_mode_cube(
            str(tmp_path / "x.cube"), mol, aux, modes, 0, normalization="max"
        )


def test_rejects_bad_arguments(water):
    mol, bs, aux, res = water
    with pytest.raises(ValueError):
        ferric.response_modes(mol, bs, aux, res, n_modes=0)
    with pytest.raises(ValueError):
        ferric.response_modes(mol, bs, aux, res, omega=-1.0)
    with pytest.raises(ValueError):
        ferric.response_modes(mol, ferric.BasisSet.bundled("sto-3g"), aux, res)
