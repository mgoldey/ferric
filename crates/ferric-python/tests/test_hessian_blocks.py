"""`AoIntegrals` Hessian pieces: `hess_*` skeleton terms and `*_deriv` matrices.

Anchors are independent constructions:

* every `hess_*` block equals a Richardson-extrapolated central difference of
  its gradient twin (`grad_*`) at FIXED D (or W), with the basis moving with
  the atoms (an ECP case included);
* every `*_deriv` matrix equals a central difference of the matrix it
  differentiates (`hcore()`, `overlap()`, `jk(D)`), and contracting it with D
  reproduces the matching `grad_*` piece (a separate libint code path);
* a full analytic RHF Hessian assembled in numpy from the pieces and a
  `Cphf` solve per coordinate (W1 from the gauge-invariant ½(D1 F D + D F D1
  + D F1 D), not ferric's response code) reproduces `run_frequencies`.
"""

from __future__ import annotations

import numpy as np
import pytest

import ferric

WATER = "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n"
HI = "2\n\nH 0 0 0\nI 0 0 1.609\n"
TIGHT = {"energy_conv": 1e-11, "density_conv": 1e-10}


def _ints(xyz: str, basis: str):
    mol = ferric.Molecule.from_xyz_string(xyz)
    bs = ferric.BasisSet.bundled(basis)
    return mol, bs, ferric.AoIntegrals(mol, bs)


def _displaced(mol, xyz, atom, axis, h_bohr):
    """`xyz` with one coordinate moved by h_bohr, rebuilt via Angstrom."""
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
    return ferric.Molecule.from_xyz_string("\n".join(head) + "\n" + body + "\n")


def _richardson(f, mol, xyz, bs, atom, axis, h=2e-3):
    """d f(AoIntegrals at R + h e_x) / dh at 0, central, Richardson h -> h/2."""

    def central(step):
        lo = f(ferric.AoIntegrals(_displaced(mol, xyz, atom, axis, -step), bs))
        hi = f(ferric.AoIntegrals(_displaced(mol, xyz, atom, axis, step), bs))
        return (np.asarray(hi) - np.asarray(lo)) / (2 * step)

    return (4 * central(h / 2) - central(h)) / 3


def _scf(mol, bs):
    return ferric.run_rhf(mol, bs, **TIGHT)


def _w(ints, d):
    """Energy-weighted density ½ D F D (closed shell) at D's own Fock matrix."""
    j, k = ints.jk(d)
    f = ints.hcore() + j - 0.5 * k
    return 0.5 * d @ f @ d


COORDS = [(0, 2), (1, 1), (2, 2)]


# ── skeleton second derivatives vs FD of the gradient pieces ──


def test_hess_blocks_match_finite_difference_of_the_gradient_pieces():
    mol, bs, ints = _ints(WATER, "sto-3g")
    d = _scf(mol, bs).density()
    w = _w(ints, d)
    hj, hk = ints.hess_jk(d)
    cases = {
        "nuc": (ints.hess_nuclear_repulsion(), lambda m: m.grad_nuclear_repulsion()),
        "hcore": (ints.hess_hcore(d), lambda m: m.grad_hcore(d)),
        "overlap": (ints.hess_overlap(w), lambda m: m.grad_overlap(w)),
        "J": (hj, lambda m: m.grad_jk(d)[0]),
        "K": (hk, lambda m: m.grad_jk(d)[1]),
    }
    for name, (hess, grad) in cases.items():
        assert hess.shape == (9, 9), name
        np.testing.assert_allclose(hess, hess.T, atol=1e-10, err_msg=name)
        for atom, axis in COORDS:
            fd = _richardson(grad, mol, WATER, bs, atom, axis).reshape(-1)
            np.testing.assert_allclose(
                hess[3 * atom + axis], fd, atol=1e-8, err_msg=f"{name} {atom},{axis}"
            )


def test_hess_hcore_matches_finite_difference_with_an_ecp():
    mol, bs, ints = _ints(HI, "def2-svp")
    d = _scf(mol, bs).density()
    h = ints.hess_hcore(d)
    for atom in (0, 1):
        fd = _richardson(lambda m: m.grad_hcore(d), mol, HI, bs, atom, 2).reshape(-1)
        np.testing.assert_allclose(h[3 * atom + 2], fd, atol=1e-7)


def test_every_hessian_piece_is_translation_invariant():
    mol, bs, ints = _ints(WATER, "cc-pvdz")
    d = _scf(mol, bs).density()
    hj, hk = ints.hess_jk(d)
    for name, h in [
        ("nuc", ints.hess_nuclear_repulsion()),
        ("hcore", ints.hess_hcore(d)),
        ("overlap", ints.hess_overlap(_w(ints, d))),
        ("J", hj),
        ("K", hk),
    ]:
        # Σ_A H[(A, i), (B, j)] = 0: moving every atom together changes nothing.
        np.testing.assert_allclose(
            h.reshape(3, 3, 9).sum(axis=0), 0.0, atol=1e-8, err_msg=name
        )
        assert np.abs(h).max() > 1e-2, name


AUX = "cc-pvdz-ri"


def _densities(ints, mol, bs):
    """The SCF density (PSD) and an indefinite symmetric one."""
    d = _scf(mol, bs).density()
    rng = np.random.default_rng(5)
    m = rng.standard_normal(d.shape)
    return {"psd": d, "indefinite": d + 0.3 * (m + m.T)}


@pytest.mark.parametrize("kind", ["psd", "indefinite"])
@pytest.mark.parametrize(
    "operator, omega", [("coulomb", None), ("erfc", 0.4)], ids=["coulomb", "erfc"]
)
def test_df_hess_jk_matches_finite_difference_of_df_grad_jk(operator, omega, kind):
    mol, bs, ints = _ints(WATER, "sto-3g")
    aux = ferric.BasisSet.bundled(AUX)
    d = _densities(ints, mol, bs)[kind]
    kw = {"operator": operator, "omega": omega, "aux": aux}
    hj, hk = ints.hess_jk(d, **kw)
    exact_j, _ = ints.hess_jk(d, operator, omega)
    assert np.abs(hj - exact_j).max() > 1e-6  # the fit is really in use
    for name, hess, pick in [("J", hj, 0), ("K", hk, 1)]:
        np.testing.assert_allclose(hess, hess.T, atol=1e-9, err_msg=name)
        for atom, axis in COORDS:
            fd = _richardson(
                lambda m, p=pick: m.grad_jk(d, **kw)[p], mol, WATER, bs, atom, axis
            ).reshape(-1)
            np.testing.assert_allclose(
                hess[3 * atom + axis], fd, atol=1e-8, err_msg=f"{name} {atom},{axis}"
            )


@pytest.mark.parametrize("kind", ["psd", "indefinite"])
def test_df_jk_deriv_matches_finite_difference_of_the_fitted_matrices(kind):
    mol, bs, ints = _ints(WATER, "sto-3g")
    aux = ferric.BasisSet.bundled(AUX)
    d = _densities(ints, mol, bs)[kind]
    jx, kx = ints.jk_deriv(d, aux=aux)
    gj, gk = ints.grad_jk(d, aux=aux)
    np.testing.assert_allclose(
        0.5 * np.einsum("xmn,mn->x", jx, d).reshape(3, 3), gj, atol=1e-10
    )
    np.testing.assert_allclose(
        0.5 * np.einsum("xmn,mn->x", kx, d).reshape(3, 3), gk, atol=1e-10
    )
    for name, dm, pick in [("J", jx, "j"), ("K", kx, "k")]:
        for atom, axis in COORDS:
            fd = _richardson(
                lambda m, p=pick: getattr(m.df_jk(aux), p)(d),
                mol,
                WATER,
                bs,
                atom,
                axis,
            )
            np.testing.assert_allclose(
                dm[3 * atom + axis], fd, atol=1e-8, err_msg=f"{name} {atom},{axis}"
            )


# ── first-derivative matrices ──


def test_deriv_matrices_match_finite_difference_of_the_matrices():
    mol, bs, ints = _ints(WATER, "sto-3g")
    d = _scf(mol, bs).density()
    jx, kx = ints.jk_deriv(d)
    cases = {
        "hcore": (ints.hcore_deriv(), lambda m: m.hcore()),
        "overlap": (ints.overlap_deriv(), lambda m: m.overlap()),
        "J": (jx, lambda m: m.jk(d)[0]),
        "K": (kx, lambda m: m.jk(d)[1]),
    }
    n = ints.nao
    for name, (dm, mat) in cases.items():
        assert dm.shape == (9, n, n), name
        np.testing.assert_allclose(dm, dm.transpose(0, 2, 1), atol=1e-12, err_msg=name)
        for atom, axis in COORDS:
            fd = _richardson(mat, mol, WATER, bs, atom, axis)
            np.testing.assert_allclose(
                dm[3 * atom + axis], fd, atol=1e-8, err_msg=f"{name} {atom},{axis}"
            )


@pytest.mark.parametrize(
    "xyz, basis", [(WATER, "cc-pvdz"), (HI, "def2-svp")], ids=["water", "HI-ecp"]
)
def test_deriv_matrices_contract_to_the_gradient_pieces(xyz, basis):
    mol, bs, ints = _ints(xyz, basis)
    d = _scf(mol, bs).density()
    w = _w(ints, d)
    natoms = len(mol.coords())
    jx, kx = ints.jk_deriv(d)
    gj, gk = ints.grad_jk(d)
    pairs = [
        ("hcore", np.einsum("xmn,mn->x", ints.hcore_deriv(), d), ints.grad_hcore(d)),
        (
            "overlap",
            np.einsum("xmn,mn->x", ints.overlap_deriv(), w),
            ints.grad_overlap(w),
        ),
        ("J", 0.5 * np.einsum("xmn,mn->x", jx, d), gj),
        ("K", 0.5 * np.einsum("xmn,mn->x", kx, d), gk),
    ]
    for name, got, ref in pairs:
        np.testing.assert_allclose(
            got.reshape(natoms, 3), ref, atol=1e-10, err_msg=name
        )


# ── a full RHF Hessian assembled in numpy ──


def _numpy_rhf_hessian(ints, r, aux=None):
    """Skeleton from `hess_*`, response from one `Cphf.solve` per coordinate.

    D1 = density(U_vo) - 2 C_o S1_oo C_oᵀ (U_oo = -½ S1_oo); the response
    term is Tr[D1^x F^y] - Tr[W1^x S^y] with W1 = ½(D1 F D + D F D1 + D F1 D)
    and F1 = F^x + G(D1^x), the total first-order Fock matrix. `aux`: every
    J/K (SCF Fock, G, CPHF, skeleton, first-order) from the `aux` fits.
    """
    if aux is None:
        jk = ints.jk
    else:
        df = ints.df_jk(aux)

        def jk(dm):
            return df.j(dm), df.k(dm)

    d = r.density()
    j, k = jk(d)
    f = ints.hcore() + j - 0.5 * k
    s = ints.overlap()
    w_, v_ = np.linalg.eigh(s)
    x = v_ / np.sqrt(w_)
    e, c = np.linalg.eigh(x.T @ f @ x)
    c = x @ c
    nocc = ints.nelec // 2
    co, cv = c[:, :nocc], c[:, nocc:]
    w = 0.5 * d @ f @ d

    hj, hk = ints.hess_jk(d, aux=aux)
    hess = (
        ints.hess_nuclear_repulsion()
        + ints.hess_hcore(d)
        - ints.hess_overlap(w)
        + hj
        - 0.5 * hk
    )

    jx, kx = ints.jk_deriv(d, aux=aux)
    f1 = ints.hcore_deriv() + jx - 0.5 * kx
    s1 = ints.overlap_deriv()
    cphf = ints.cphf(c, e, nocc, aux=aux)

    def g(dm):
        jj, kk = jk(dm)
        return jj - 0.5 * kk

    d1 = []
    w1 = []
    for fx, sx in zip(f1, s1):
        s_oo = co.T @ sx @ co
        d_s = -2.0 * co @ s_oo @ co.T
        rhs = -(cv.T @ fx @ co - (cv.T @ sx @ co) * e[:nocc] + cv.T @ g(d_s) @ co)
        u = cphf.solve(rhs, tol=1e-11, max_iter=200)
        dx = cphf.density(u) + d_s
        f_tot = fx + g(dx)
        d1.append(dx)
        w1.append(0.5 * (dx @ f @ d + d @ f @ dx + d @ f_tot @ d))
    resp = np.einsum("xmn,ymn->xy", np.array(d1), f1) - np.einsum(
        "xmn,ymn->xy", np.array(w1), s1
    )
    return hess + resp


def test_numpy_rhf_hessian_from_pieces_matches_run_frequencies():
    mol, bs, ints = _ints(WATER, "6-31g")
    r = _scf(mol, bs)
    got = _numpy_rhf_hessian(ints, r)
    ref = ferric.run_frequencies(mol, "6-31g", hessian="analytic", **TIGHT)
    assert ref.hessian_source == "analytic"
    ref = np.array(ref.cartesian_hessian)
    np.testing.assert_allclose(got, got.T, atol=1e-8)
    np.testing.assert_allclose(got, ref, atol=1e-9)


def test_numpy_ri_jk_rhf_hessian_from_pieces_matches_run_frequencies():
    mol, bs, ints = _ints(WATER, "6-31g")
    fits = {"df_j_aux": AUX, "df_k_aux": AUX}
    r = ferric.run_rhf(mol, bs, **fits, **TIGHT)
    got = _numpy_rhf_hessian(ints, r, aux=ferric.BasisSet.bundled(AUX))
    ref = ferric.run_frequencies(mol, "6-31g", hessian="analytic", **fits, **TIGHT)
    assert ref.hessian_source == "analytic"
    ref = np.array(ref.cartesian_hessian)
    np.testing.assert_allclose(got, got.T, atol=1e-8)
    np.testing.assert_allclose(got, ref, atol=1e-9)
    # The fit is not the exact Hessian (else the test could not tell them apart).
    exact = _numpy_rhf_hessian(ints, r)
    assert np.abs(got - exact).max() > 1e-6


def test_hessian_pieces_reject_bad_matrices():
    _, _, ints = _ints(WATER, "sto-3g")
    n = ints.nao
    with pytest.raises(ValueError, match="shape"):
        ints.hess_hcore(np.eye(n + 1))
    with pytest.raises(ValueError, match="not symmetric"):
        ints.hess_jk(np.triu(np.ones((n, n))))
    with pytest.raises(ValueError, match="not symmetric"):
        ints.jk_deriv(np.triu(np.ones((n, n))))
    with pytest.raises(ValueError, match="NaN or inf"):
        ints.hess_overlap(np.full((n, n), np.nan))
