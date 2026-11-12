"""`XcFunctional.hess` / `vxc_deriv` and their `*_polarized` twins: the XC
Hessian pieces of the analytic RKS and UKS Hessians.

Both are on the MOVING grid (points ride their home atom, Becke weights
differentiated), the convention of `run_frequencies(hessian=
"analytic-moving-grid")` and of `XcFunctional.grad` / `eval`, whose
derivatives they are. Anchors are independent constructions:

* `hess(D)` equals a Richardson-extrapolated central difference of
  `grad(D)` at FIXED D, with the basis AND the grid rebuilt at each displaced
  geometry (`grad` is `ks_gradient::xc_gradient_closed`, a separate code path
  from the Hessian's moving-grid skeleton);
* `vxc_deriv(D)` equals the same difference of `eval(D)`'s V_xc matrix, and
  Tr[D dV/dx] equals d/dt grad(t D) at t = 1 (V_xc = dE_xc/dD, so
  Tr[D d2E/dD dx] is the derivative of the gradient along the density
  scaling; for an exchange-only LDA it is (4/3) grad by homogeneity);
* a full RKS Hessian assembled in numpy from the `hess_*` pieces, these two
  and one CPKS solve per coordinate (`cphf(xc=)`, its kernel from ferric's
  ROKS/UKS f_xc store, not the Hessian's response kernel) reproduces
  `run_frequencies(hessian="analytic-moving-grid")`;
* `hess_polarized` / `vxc_deriv_polarized` equal the closed-shell pieces at
  D_a = D_b = D/2 and the same finite differences of `grad_polarized` /
  `eval_polarized` on the NH2 doublet.

Artifact hypothesis: the end-to-end test SHARES the XC skeleton and
first-order Fock code with the driver, so it cannot see a bug in either; only
the finite-difference tests can. It does check the assembly (signs, exchange
fraction, W1, the kernel in G) against the driver.
"""

from __future__ import annotations

import numpy as np
import pytest

import ferric

WATER = "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n"
TIGHT = {"energy_conv": 1e-11, "density_conv": 1e-10}
FREQ_TIGHT = {f"scf_{k}": v for k, v in TIGHT.items()}  # run_frequencies spelling
EXACT_JK = {"df_j_aux": "none", "df_k_aux": "none"}
COORDS = [(0, 2), (1, 1), (2, 2), (1, 0)]
FUNCTIONALS = ["LDA", "PBE", "B3LYP"]


def _ints(xyz: str, basis: str, multiplicity: int = 1):
    mol = ferric.Molecule.from_xyz_string(xyz, multiplicity=multiplicity)
    bs = ferric.BasisSet.bundled(basis)
    return mol, bs, ferric.AoIntegrals(mol, bs)


def _displaced(mol, xyz, atom, axis, h_bohr, multiplicity=1):
    bohr_per_ang = next(
        b / a
        for cb, ca in zip(mol.coords_bohr(), mol.coords())
        for b, a in zip(cb, ca)
        if abs(a) > 0.1
    )
    lines = xyz.strip("\n").split("\n")
    head, atoms = lines[:2], [ln.split() for ln in lines[2:]]
    atoms[atom][1 + axis] = repr(float(atoms[atom][1 + axis]) + h_bohr / bohr_per_ang)
    body = "\n".join(" ".join(a) for a in atoms)
    return ferric.Molecule.from_xyz_string(
        "\n".join(head) + "\n" + body + "\n", multiplicity=multiplicity
    )


def _richardson(f, mol, xyz, bs, atom, axis, h=2e-3, multiplicity=1):
    """d f(AoIntegrals at R + h e_x)/dh at 0, central, Richardson h -> h/2."""

    def at(step):
        moved = _displaced(mol, xyz, atom, axis, step, multiplicity)
        return np.asarray(f(ferric.AoIntegrals(moved, bs)))

    def central(step):
        return (at(step) - at(-step)) / (2 * step)

    return (4 * central(h / 2) - central(h)) / 3


def _density(mol, bs, functional):
    r = ferric.run_dft(mol, bs, functional=functional, **EXACT_JK, **TIGHT)
    return r.density()


# ── hess(D) vs finite differences of grad(D) ──


@pytest.mark.parametrize("functional", FUNCTIONALS)
def test_xc_hess_matches_finite_difference_of_xc_grad(functional):
    mol, bs, ints = _ints(WATER, "sto-3g")
    d = _density(mol, bs, functional)
    h = ints.xc_functional(functional).hess(d)
    assert h.shape == (9, 9)
    assert np.abs(h).max() > 1e-2
    for atom, axis in COORDS:
        fd = _richardson(
            lambda m: m.xc_functional(functional).grad(d), mol, WATER, bs, atom, axis
        ).reshape(-1)
        # Measured 2026-10-29: max 2.1e-11 (LDA), 5.8e-12 (PBE), 5.0e-10 (B3LYP).
        np.testing.assert_allclose(
            h[3 * atom + axis], fd, atol=5e-8, err_msg=f"{atom},{axis}"
        )


@pytest.mark.parametrize("functional", ["PBE", "B3LYP"])
def test_xc_hess_is_symmetric_and_translation_invariant(functional):
    mol, bs, ints = _ints(WATER, "6-31g")
    h = ints.xc_functional(functional).hess(_density(mol, bs, functional))
    np.testing.assert_allclose(h, h.T, atol=1e-14)
    # The grid follows the atoms, so a rigid translation changes nothing:
    # Σ_A H[(A, i), (B, j)] = 0 exactly (not only to the grid error).
    np.testing.assert_allclose(h.reshape(3, 3, 9).sum(axis=0), 0.0, atol=1e-14)


# ── vxc_deriv(D) vs finite differences of eval(D)'s V_xc ──


@pytest.mark.parametrize("functional", FUNCTIONALS)
def test_vxc_deriv_matches_finite_difference_of_vxc(functional):
    mol, bs, ints = _ints(WATER, "sto-3g")
    d = _density(mol, bs, functional)
    vx = ints.xc_functional(functional).vxc_deriv(d)
    n = ints.nao
    assert vx.shape == (9, n, n)
    np.testing.assert_allclose(vx, vx.transpose(0, 2, 1), atol=1e-12)
    for atom, axis in COORDS:
        fd = _richardson(
            lambda m: m.xc_functional(functional).eval(d)[1], mol, WATER, bs, atom, axis
        )
        # Measured 2026-10-29: max 2.5e-12 / 6.8e-12 / 4.3e-12.
        np.testing.assert_allclose(
            vx[3 * atom + axis], fd, atol=1e-9, err_msg=f"{atom},{axis}"
        )


@pytest.mark.parametrize("functional", ["LDA", "PBE"])
def test_vxc_deriv_contracts_to_the_density_scaling_derivative_of_grad(functional):
    # V = dE/dD, so Tr[D dV/dx] = d/dt [dE/dx](t D) at t = 1.
    mol, bs, ints = _ints(WATER, "6-31g")
    d = _density(mol, bs, functional)
    xc = ints.xc_functional(functional)
    got = np.einsum("xmn,mn->x", xc.vxc_deriv(d), d).reshape(3, 3)

    def central(t):
        return (xc.grad((1 + t) * d) - xc.grad((1 - t) * d)) / (2 * t)

    ref = (4 * central(5e-4) - central(1e-3)) / 3
    # Measured 2026-10-29: 5.0e-12 (LDA), 2.3e-11 (PBE).
    np.testing.assert_allclose(got, ref, atol=3e-9)
    # ... which is not just the gradient (E_xc is not homogeneous of degree 1).
    assert np.abs(got - xc.grad(d)).max() > 1e-3


def test_slater_exchange_vxc_deriv_is_four_thirds_of_grad():
    # E_x[t D] = t^(4/3) E_x[D] for Slater exchange on any fixed grid,
    # so Tr[D dV/dx] = (4/3) dE/dx.
    mol, bs, ints = _ints(WATER, "6-31g")
    d = _density(mol, bs, "LDA")
    xc = ints.xc_functional("LDA_X")
    got = np.einsum("xmn,mn->x", xc.vxc_deriv(d), d).reshape(3, 3)
    # Measured 2026-10-29: 1.7e-12.
    np.testing.assert_allclose(got, 4.0 / 3.0 * xc.grad(d), atol=2e-10)


# ── a full RKS Hessian assembled in numpy ──


def _numpy_rks_hessian(ints, xc, d):
    """Skeleton from `hess_*` + `xc.hess`, response from `cphf(xc=)`.

    F = h + J - (a/2) K + V_xc at the SCF density, F^x = h^x + J^x
    - (a/2) K^x + vxc_deriv, G(D1) = J(D1) - (a/2) K(D1) + f_xc[D1]; the
    response term is Tr[D1^x F^y] - Tr[W1^x S^y] with the gauge-invariant
    W1 = ½(D1 F D + D F D1 + D F1 D), F1 = F^x + G(D1^x).
    """
    a, c_lr, omega = xc.exchange_mix
    assert omega == 0.0 and a == c_lr
    j, k = ints.jk(d)
    _, vxc = xc.eval(d)
    f = ints.hcore() + j - 0.5 * a * k + vxc
    s = ints.overlap()
    w_, v_ = np.linalg.eigh(s)
    x = v_ / np.sqrt(w_)
    e, c = np.linalg.eigh(x.T @ f @ x)
    c = x @ c
    nocc = ints.nelec // 2
    co, cv = c[:, :nocc], c[:, nocc:]
    np.testing.assert_allclose(2 * co @ co.T, d, atol=1e-7)
    w = 0.5 * d @ f @ d

    hj, hk = ints.hess_jk(d)
    hess = (
        ints.hess_nuclear_repulsion()
        + ints.hess_hcore(d)
        - ints.hess_overlap(w)
        + hj
        - 0.5 * a * hk
        + xc.hess(d)
    )
    jx, kx = ints.jk_deriv(d)
    f1 = ints.hcore_deriv() + jx - 0.5 * a * kx + xc.vxc_deriv(d)
    s1 = ints.overlap_deriv()
    cphf = ints.cphf(c, e, nocc, xc=xc)

    def g(dm):
        jj, kk = ints.jk(dm)
        return jj - 0.5 * a * kk + xc.response(d, dm)

    d1, w1 = [], []
    for fx, sx in zip(f1, s1):
        d_s = -2.0 * co @ (co.T @ sx @ co) @ co.T
        rhs = -(cv.T @ fx @ co - (cv.T @ sx @ co) * e[:nocc] + cv.T @ g(d_s) @ co)
        u = cphf.solve(rhs, tol=1e-11, max_iter=200)
        dx = cphf.density(u) + d_s
        d1.append(dx)
        w1.append(0.5 * (dx @ f @ d + d @ f @ dx + d @ (fx + g(dx)) @ d))
    resp = np.einsum("xmn,ymn->xy", np.array(d1), f1) - np.einsum(
        "xmn,ymn->xy", np.array(w1), s1
    )
    return hess + resp


@pytest.mark.parametrize("functional", FUNCTIONALS)
def test_numpy_rks_hessian_from_pieces_matches_run_frequencies(functional):
    mol, bs, ints = _ints(WATER, "6-31g")
    xc = ints.xc_functional(functional)
    got = _numpy_rks_hessian(ints, xc, _density(mol, bs, functional))
    ref = ferric.run_frequencies(
        mol,
        "6-31g",
        xc=functional,
        hessian="analytic-moving-grid",
        **EXACT_JK,
        **FREQ_TIGHT,
    )
    assert ref.hessian_source == "analytic-moving-grid"
    ref = np.array(ref.cartesian_hessian)
    # Measured 2026-10-29: asymmetry <= 4.0e-12; vs the driver 5.8e-11 (LDA),
    # 3.9e-11 (PBE), 2.4e-10 (B3LYP).
    np.testing.assert_allclose(got, got.T, atol=1e-9)
    np.testing.assert_allclose(got, ref, atol=3e-8)


# ── spin-polarized pieces: hess_polarized / vxc_deriv_polarized ──
#
# E_xc^pol[D/2, D/2] = E_xc[D] and V_a = V_b = V there, so at equal spin
# densities hess_polarized equals hess and EACH spin's vxc_deriv_polarized
# equals vxc_deriv (not half of it). The two sides run libxc with nspin = 2
# and nspin = 1, so they agree to its evaluation round-off, not bitwise.
# That anchor cannot see an alpha/beta mix-up; the finite differences on the
# NH2 doublet (unequal spin densities) can.

NH2 = "3\n\nN 0 0 0\nH 0.12 0.80 0.60\nH -0.05 -0.74 0.52\n"


def _nh2_uhf():
    mol, bs, ints = _ints(NH2, "6-31g", multiplicity=2)
    r = ferric.run_uhf(mol, bs, **TIGHT)
    return mol, bs, ints, r.density_alpha(), r.density_beta()


@pytest.mark.parametrize("functional", FUNCTIONALS)
def test_polarized_pieces_equal_closed_shell_at_equal_spin_densities(functional):
    mol, bs, ints = _ints(WATER, "6-31g")
    d = _density(mol, bs, functional)
    xc = ints.xc_functional(functional)
    h = xc.hess(d)
    vx = xc.vxc_deriv(d)
    h_pol = xc.hess_polarized(0.5 * d, 0.5 * d)
    va, vb = xc.vxc_deriv_polarized(0.5 * d, 0.5 * d)
    # Measured 2026-11-11: hess 1.5e-15 (LDA), 1.2e-15 (PBE), 9.8e-13 (B3LYP);
    # V 3.1e-15 / 3.0e-15 / 2.6e-12. B3LYP's VWN term has separate libxc
    # nspin = 1 and nspin = 2 expressions, which round differently.
    np.testing.assert_allclose(h_pol, h, atol=1e-11)
    np.testing.assert_allclose(va, vx, atol=1e-11)
    np.testing.assert_allclose(vb, vx, atol=1e-11)


def test_hess_polarized_matches_finite_difference_of_grad_polarized():
    """Fixed (D_a, D_b): the density is not re-converged at displaced points.

    grad_polarized is ks_gradient::xc_gradient_open, a separate code path
    from the Hessian's polarized moving-grid skeleton. The bar is derived per
    row: Richardson's truncation error, estimated from the same extrapolation
    at h and 2h (error ~ h^4, so |R(2h) - R(h)| ~ 15 |error of R(h)|), plus
    the gradient's round-off amplified by the stencil, 3 eps / h for
    R = (4 C(h/2) - C(h)) / 3 with eps the residual of translation invariance
    sum_A g_A = 0 (exact for the moving grid).
    """
    mol, bs, ints, da, db = _nh2_uhf()
    xc = ints.xc_functional("PBE")
    h = xc.hess_polarized(da, db)
    assert h.shape == (9, 9)
    assert np.abs(h).max() > 1e-2
    # Measured 2026-11-11: asymmetry and translation residual 5.6e-17.
    np.testing.assert_allclose(h, h.T, atol=1e-13)
    np.testing.assert_allclose(h.reshape(3, 3, 9).sum(axis=0), 0.0, atol=1e-13)

    def grad(m):
        return m.xc_functional("PBE").grad_polarized(da, db)

    eps = max(np.abs(grad(ints).sum(axis=0)).max(), 1e-16 * np.abs(grad(ints)).max())
    step = 2e-3
    for atom, axis in [(0, 2), (1, 0), (2, 1), (1, 2)]:
        fd = [
            _richardson(grad, mol, NH2, bs, atom, axis, h=s, multiplicity=2).reshape(-1)
            for s in (step, 2 * step)
        ]
        trunc = np.abs(fd[1] - fd[0]).max() / 15
        bar = 3 * (trunc + 3 * eps / step)
        err = np.abs(h[3 * atom + axis] - fd[0]).max()
        # Measured 2026-11-11 over all 9 rows: err 1.9e-12..1.3e-11, truncation
        # estimate 2.2e-13..1.1e-12, eps 1.0e-14 (3 eps / h = 1.5e-11), so the
        # bar is about 5e-11 and the round-off term dominates it.
        assert bar < 1e-7, (atom, axis, bar)
        assert err < bar, (atom, axis, err, bar)


def test_vxc_deriv_polarized_matches_finite_difference_of_the_spin_potentials():
    mol, bs, ints, da, db = _nh2_uhf()
    xc = ints.xc_functional("PBE")
    va, vb = xc.vxc_deriv_polarized(da, db)
    n = ints.nao
    assert va.shape == vb.shape == (9, n, n)
    for v in (va, vb):
        np.testing.assert_allclose(v, v.transpose(0, 2, 1), atol=1e-12)
    for atom, axis in [(0, 2), (1, 0), (2, 1)]:
        for s, v in ((1, va), (2, vb)):
            fd = _richardson(
                lambda m, s=s: m.xc_functional("PBE").eval_polarized(da, db)[s],
                mol,
                NH2,
                bs,
                atom,
                axis,
                multiplicity=2,
            )
            # Measured 2026-11-11: max 9.0e-12 (alpha), 9.0e-12 (beta).
            np.testing.assert_allclose(
                v[3 * atom + axis], fd, atol=1e-9, err_msg=f"spin {s} {atom},{axis}"
            )
        # The spins differ here, so the test can see a swapped pair.
        assert np.abs(va[3 * atom + axis] - vb[3 * atom + axis]).max() > 1e-3


def test_vxc_deriv_polarized_contracts_to_the_density_scaling_derivative():
    # V_s = dE/dD_s, so sum_s Tr[D_s dV_s/dx] = d/dt grad_polarized(t D_a, t D_b).
    _, _, ints, da, db = _nh2_uhf()
    xc = ints.xc_functional("PBE")
    va, vb = xc.vxc_deriv_polarized(da, db)
    got = np.einsum("xmn,mn->x", va, da) + np.einsum("xmn,mn->x", vb, db)

    def central(t):
        hi = xc.grad_polarized((1 + t) * da, (1 + t) * db)
        lo = xc.grad_polarized((1 - t) * da, (1 - t) * db)
        return (hi - lo) / (2 * t)

    ref = (4 * central(5e-4) - central(1e-3)) / 3
    # Measured 2026-11-11: 1.7e-11; the bar is the closed-shell test's.
    np.testing.assert_allclose(got.reshape(3, 3), ref, atol=3e-9)
    # It differs from the gradient itself (measured difference 0.10).
    assert np.abs(got.reshape(3, 3) - xc.grad_polarized(da, db)).max() > 1e-3


# ── refusals ──


@pytest.mark.parametrize(
    "functional, what",
    [
        ("SCAN", "meta-GGA"),
        ("wB97X-V", "VV10"),
        ("HYB_GGA_XC_WB97X", "range-separated"),
    ],
)
def test_unsupported_functionals_raise(functional, what):
    _, _, ints = _ints(WATER, "sto-3g")
    xc = ints.xc_functional(functional)
    d = np.eye(ints.nao)
    for method in (xc.hess, xc.vxc_deriv):
        with pytest.raises(ValueError, match=what):
            method(d)
    for method in (xc.hess_polarized, xc.vxc_deriv_polarized):
        with pytest.raises(ValueError, match=what):
            method(0.5 * d, 0.5 * d)


def test_ecp_systems_raise():
    # The analytic KS Hessian refuses ECPs (ECP and embedding terms with the
    # XC kernel); its XC pieces do the same. def2-SVP carries an ECP on I.
    _, _, ints = _ints("2\n\nH 0 0 0\nI 0 0 1.609\n", "def2-svp")
    xc = ints.xc_functional("PBE")
    d = np.eye(ints.nao)
    for method, args in [
        (xc.hess, (d,)),
        (xc.vxc_deriv, (d,)),
        (xc.hess_polarized, (0.5 * d, 0.5 * d)),
        (xc.vxc_deriv_polarized, (0.5 * d, 0.5 * d)),
    ]:
        with pytest.raises(ValueError, match="an ECP"):
            method(*args)
    # Without the ECP the same functional is supported.
    _, _, light = _ints("2\n\nH 0 0 0\nCl 0 0 1.27\n", "def2-svp")
    assert np.all(np.isfinite(light.xc_functional("PBE").hess(np.eye(light.nao))))


def test_pruned_grid_is_supported_and_bad_densities_raise():
    _, _, ints = _ints(WATER, "sto-3g")
    n = ints.nao
    pruned = ints.xc_functional("PBE", grid_prune="nwchem")
    # A pruned grid is supported: every point still rides its home atom
    # (48225518c made nwchem-radial pruning the KS default).
    h = pruned.hess(np.eye(n))
    assert h.shape == (9, 9) and np.all(np.isfinite(h))
    v = pruned.vxc_deriv(np.eye(n))
    assert v.shape == (9, n, n) and np.all(np.isfinite(v))
    h = pruned.hess_polarized(0.6 * np.eye(n), 0.4 * np.eye(n))
    assert h.shape == (9, 9) and np.all(np.isfinite(h))
    va, vb = pruned.vxc_deriv_polarized(0.6 * np.eye(n), 0.4 * np.eye(n))
    assert va.shape == vb.shape == (9, n, n)
    assert np.all(np.isfinite(va)) and np.all(np.isfinite(vb))
    xc = ints.xc_functional("PBE")
    with pytest.raises(ValueError, match="shape"):
        xc.hess(np.eye(n + 1))
    with pytest.raises(ValueError, match="not symmetric"):
        xc.vxc_deriv(np.triu(np.ones((n, n))))
    with pytest.raises(ValueError, match="shape"):
        xc.hess_polarized(np.eye(n), np.eye(n + 1))
    with pytest.raises(ValueError, match="not symmetric"):
        xc.vxc_deriv_polarized(np.triu(np.ones((n, n))), np.eye(n))
