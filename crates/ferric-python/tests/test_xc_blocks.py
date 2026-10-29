"""`AoIntegrals.xc_functional()`: E_xc, V_xc and the XC gradient.

Anchors are independent constructions:

* a numpy KS SCF on `eval()` + `jk()` + `exchange_mix` reproduces `run_dft`
  (exact J/K) for an LDA-free GGA, a global hybrid and a meta-GGA;
* V_xc is the derivative of E_xc: a directional finite difference of
  `eval()` equals Tr[V X] (closed and polarized, incl. VV10);
* `grad()` equals finite differences of `eval()`'s energy at FIXED D with
  the atom (its basis functions and grid) displaced;
* a KS gradient assembled in numpy equals `run_dft(with_gradient=True)` and
  finite differences of `run_dft` energies;
* the polarized path at D_a = D_b = D/2 equals the closed-shell one.
"""

from __future__ import annotations

import numpy as np
import pytest

import ferric

WATER = "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n"
NH2_ASYM = "3\n\nN 0 0 0\nH 0.12 0.80 0.60\nH -0.05 -0.74 0.52\n"
# Range-separated: erfc AND erf exact exchange, mix (0.158, 1.0, 0.3).
WB97X = "HYB_GGA_XC_WB97X"
TIGHT = {"energy_conv": 1e-11, "density_conv": 1e-9}
EXACT_JK = {"df_j_aux": "none", "df_k_aux": "none"}
_RNG = np.random.default_rng(5)


def _ints(xyz, basis, multiplicity=1):
    mol = ferric.Molecule.from_xyz_string(xyz, multiplicity=multiplicity)
    bs = ferric.BasisSet.bundled(basis)
    return mol, bs, ferric.AoIntegrals(mol, bs)


def _sym(n):
    x = _RNG.standard_normal((n, n))
    return 0.5 * (x + x.T)


def _ks_fock(ints, xc, d):
    c_sr, c_lr, omega = xc.exchange_mix
    j, k = ints.jk(d)
    if omega > 0:
        k = (
            c_sr * ints.jk(d, "erfc", omega=omega)[1]
            + c_lr * ints.jk(d, "erf", omega=omega)[1]
        )
    else:
        k = c_sr * k
    e_xc, v = xc.eval(d)
    return ints.hcore() + j - 0.5 * k + v, j, k, e_xc


def _numpy_ks(ints, xc):
    s, h = ints.overlap(), ints.hcore()
    w, u = np.linalg.eigh(s)
    x = u / np.sqrt(w)
    nocc = ints.nelec // 2
    d = np.zeros_like(s)
    for it in range(400):
        f, *_ = _ks_fock(ints, xc, d)
        e, c = np.linalg.eigh(x.T @ f @ x)
        c = x @ c
        d_new = 2.0 * c[:, :nocc] @ c[:, :nocc].T
        if np.abs(d_new - d).max() < 1e-10:
            d = d_new
            break
        d = 0.5 * (d + d_new) if it > 30 else d_new
    else:
        pytest.fail("numpy KS loop did not converge")
    f, j, k, e_xc = _ks_fock(ints, xc, d)
    energy = 0.5 * np.sum(d * (2 * h + j - 0.5 * k)) + e_xc + ints.nuclear_repulsion()
    return energy, d, f


@pytest.mark.parametrize("functional", ["PBE", "B3LYP", "SCAN"])
def test_numpy_ks_on_eval_reproduces_run_dft(functional):
    mol, bs, ints = _ints(WATER, "6-31g")
    xc = ints.xc_functional(functional)
    e_np, _, _ = _numpy_ks(ints, xc)
    r = ferric.run_dft(mol, bs, functional=functional, **EXACT_JK, **TIGHT)
    assert abs(e_np - r.total_energy) < 1e-8


@pytest.mark.parametrize(
    "functional",
    [
        "PBE",
        "SCAN",
        WB97X,
        pytest.param(
            "wB97X-V",
            marks=pytest.mark.xfail(
                strict=True,
                reason="pre-existing: ferric's VV10 V_nl is not the exact derivative "
                "of E_nl. Measured 2026-10-29, NH2/6-31G, random symmetric X: "
                "FD(E) - Tr[V X] = -1.31e-7 (3.6e-6 relative), constant over "
                "h = 4e-4..2.5e-5; wB97X without VV10 agrees to 8.6e-14",
            ),
        ),
    ],
)
def test_vxc_is_the_derivative_of_exc(functional):
    # Own RNG: the direction must not depend on which tests ran first (the
    # VV10 defect's size depends on it, which once let the xfail pass).
    rng = np.random.default_rng(5)
    mol, bs, ints = _ints(NH2_ASYM, "6-31g", multiplicity=2)
    xc = ints.xc_functional(functional)
    r = ferric.run_uhf(mol, bs, **TIGHT)
    da, db = r.density_alpha(), r.density_beta()
    n, h = ints.nao, 1e-4
    x = 1e-2 * rng.standard_normal((n, n))
    x = x + x.T
    y = 1e-2 * rng.standard_normal((n, n))
    y = y + y.T
    _, v = xc.eval(da + db)
    fd = (xc.eval(da + db + h * x)[0] - xc.eval(da + db - h * x)[0]) / (2 * h)
    assert abs(fd - np.sum(v * x)) < 1e-9 * max(1.0, abs(fd))
    _, va, vb = xc.eval_polarized(da, db)
    ep = xc.eval_polarized(da + h * x, db + h * y)[0]
    em = xc.eval_polarized(da - h * x, db - h * y)[0]
    analytic = np.sum(va * x) + np.sum(vb * y)
    assert abs((ep - em) / (2 * h) - analytic) < 1e-9 * max(1.0, abs(analytic))


@pytest.mark.parametrize("functional", ["PBE", "SCAN"])
def test_polarized_equals_closed_at_equal_spin_densities(functional):
    mol, bs, ints = _ints(WATER, "6-31g")
    xc = ints.xc_functional(functional)
    d = ferric.run_rhf(mol, bs, **TIGHT).density()
    e, v = xc.eval(d)
    e2, va, vb = xc.eval_polarized(0.5 * d, 0.5 * d)
    assert abs(e - e2) < 1e-10
    np.testing.assert_allclose(va, v, atol=1e-9)
    np.testing.assert_allclose(vb, v, atol=1e-9)
    np.testing.assert_allclose(
        xc.grad_polarized(0.5 * d, 0.5 * d), xc.grad(d), atol=1e-9
    )


def _displaced(mol, xyz, atom, axis, h_bohr, multiplicity):
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


@pytest.mark.parametrize("functional", ["PBE", "SCAN"])
def test_xc_grad_matches_finite_difference_at_fixed_density(functional):
    xyz = NH2_ASYM
    mol, bs, ints = _ints(xyz, "sto-3g", multiplicity=2)
    r = ferric.run_uhf(mol, bs, **TIGHT)
    da, db = r.density_alpha(), r.density_beta()
    xc = ints.xc_functional(functional)
    g, g_pol = xc.grad(da + db), xc.grad_polarized(da, db)
    h = 1e-4
    for atom, axis in [(0, 2), (1, 0), (2, 1)]:
        e, e_pol = [], []
        for step in (h, -h):
            moved = ferric.AoIntegrals(_displaced(mol, xyz, atom, axis, step, 2), bs)
            mxc = moved.xc_functional(functional)
            e.append(mxc.eval(da + db)[0])
            e_pol.append(mxc.eval_polarized(da, db)[0])
        assert abs(g[atom, axis] - (e[0] - e[1]) / (2 * h)) < 2e-6, (atom, axis)
        assert abs(g_pol[atom, axis] - (e_pol[0] - e_pol[1]) / (2 * h)) < 2e-6


@pytest.mark.parametrize("functional", ["PBE", "B3LYP"])
def test_ks_gradient_from_pieces_matches_run_dft(functional):
    mol, bs, ints = _ints(WATER, "6-31g")
    xc = ints.xc_functional(functional)
    r = ferric.run_dft(
        mol, bs, functional=functional, with_gradient=True, **EXACT_JK, **TIGHT
    )
    _, d, f = _numpy_ks(ints, xc)
    s = ints.overlap()
    w_, u = np.linalg.eigh(s)
    x = u / np.sqrt(w_)
    e, c = np.linalg.eigh(x.T @ f @ x)
    c = x @ c
    nocc = ints.nelec // 2
    w = 2.0 * (c[:, :nocc] * e[:nocc]) @ c[:, :nocc].T
    c_sr, _, _ = xc.exchange_mix
    gj, gk = ints.grad_jk(d)
    g = (
        ints.grad_nuclear_repulsion()
        + ints.grad_hcore(d)
        - ints.grad_overlap(w)
        + gj
        - 0.5 * c_sr * gk
        + xc.grad(d)
    )
    np.testing.assert_allclose(g, r.gradient(), atol=1e-7)
    assert np.abs(g).max() > 1e-3


def test_xc_grad_is_translation_invariant_and_bad_names_raise():
    mol, bs, ints = _ints(WATER, "6-31g")
    d = ferric.run_rhf(mol, bs, **TIGHT).density()
    g = ints.xc_functional("PBE").grad(d)
    np.testing.assert_allclose(g.sum(axis=0), 0.0, atol=1e-6)
    with pytest.raises(ValueError, match="no-such-functional"):
        ints.xc_functional("no-such-functional")
    assert ints.xc_functional("B3LYP").exchange_mix[0] == pytest.approx(0.2)
    assert ints.xc_functional("PBE").exchange_mix == (0.0, 0.0, 0.0)


# ── AoIntegrals.veff (J + exchange-mix K + V_xc, exact or fitted) ──

AUX = "cc-pvdz-ri"
DF = {"df_j_aux": AUX, "df_k_aux": AUX}


def _veff_scf(ints, veff):
    """Roothaan on hcore() + V from `veff(D) -> (E, V)` alone."""
    s, h = ints.overlap(), ints.hcore()
    w, u = np.linalg.eigh(s)
    x = u / np.sqrt(w)
    nocc = ints.nelec // 2
    d = np.zeros_like(s)
    for it in range(400):
        _, v = veff(d)
        _, c = np.linalg.eigh(x.T @ (h + v) @ x)
        c = x @ c
        d_new = 2.0 * c[:, :nocc] @ c[:, :nocc].T
        if np.abs(d_new - d).max() < 1e-10:
            d = d_new
            break
        d = 0.5 * (d + d_new) if it > 30 else d_new
    else:
        pytest.fail("numpy veff SCF did not converge")
    e2, _ = veff(d)
    return np.sum(d * h) + e2 + ints.nuclear_repulsion()


@pytest.mark.parametrize(
    "functional, fitted",
    [
        (None, False),
        (None, True),
        ("PBE", False),
        ("B3LYP", False),
        ("B3LYP", True),
        (WB97X, True),
    ],
)
def test_numpy_scf_on_veff_reproduces_the_driver(functional, fitted):
    mol, bs, ints = _ints(WATER, "6-31g")
    xc = ints.xc_functional(functional) if functional else None
    aux = ferric.BasisSet.bundled(AUX) if fitted else None
    e_np = _veff_scf(ints, lambda d: ints.veff(d, xc=xc, aux=aux))
    jk = DF if fitted else EXACT_JK
    if functional:
        ref = ferric.run_dft(mol, bs, functional=functional, **jk, **TIGHT)
        assert abs(e_np - ref.total_energy) < 1e-8
    else:
        ref = ferric.run_rhf(mol, bs, **(DF if fitted else {}), **TIGHT)
        assert abs(e_np - ref.energy) < 1e-8


@pytest.mark.parametrize("fitted", [False, True], ids=["exact", "df"])
def test_numpy_uhf_on_veff_polarized_reproduces_run_uhf(fitted):
    mol, bs, ints = _ints(NH2_ASYM, "6-31g", multiplicity=2)
    aux = ferric.BasisSet.bundled(AUX) if fitted else None
    s, h = ints.overlap(), ints.hcore()
    w, u = np.linalg.eigh(s)
    x = u / np.sqrt(w)
    ref = ferric.run_uhf(mol, bs, **(DF if fitted else {}), **TIGHT)
    # Start from the converged densities: this checks that they are a fixed
    # point of hcore() + veff_polarized and that E matches, not convergence.
    da, db = ref.density_alpha(), ref.density_beta()
    for _ in range(3):
        _, va, vb = ints.veff_polarized(da, db, aux=aux)
        new = []
        for v, n in ((va, 5), (vb, 4)):
            _, c = np.linalg.eigh(x.T @ (h + v) @ x)
            c = x @ c
            new.append(c[:, :n] @ c[:, :n].T)
        da, db = new
    np.testing.assert_allclose(da, ref.density_alpha(), atol=1e-7)
    e2, _, _ = ints.veff_polarized(da, db, aux=aux)
    e = np.sum((da + db) * h) + e2 + ints.nuclear_repulsion()
    assert abs(e - ref.energy) < 1e-9


@pytest.mark.parametrize("functional", [None, "B3LYP"])
def test_veff_polarized_at_equal_spins_equals_closed(functional):
    mol, bs, ints = _ints(WATER, "6-31g")
    xc = ints.xc_functional(functional) if functional else None
    d = ferric.run_rhf(mol, bs, **TIGHT).density()
    e, v = ints.veff(d, xc=xc)
    e2, va, vb = ints.veff_polarized(0.5 * d, 0.5 * d, xc=xc)
    assert abs(e - e2) < 1e-10
    np.testing.assert_allclose(va, v, atol=1e-9)
    np.testing.assert_allclose(vb, v, atol=1e-9)


def test_veff_reuses_its_builders_and_refuses_a_foreign_functional():
    mol, bs, ints = _ints(WATER, "6-31g")
    aux = ferric.BasisSet.bundled(AUX)
    d = ferric.run_rhf(mol, bs, **TIGHT).density()
    first = ints.veff(d, aux=aux)
    np.testing.assert_array_equal(ints.veff(d, aux=aux)[1], first[1])
    assert np.abs(ints.veff(d)[1] - first[1]).max() > 1e-6
    _, _, other = _ints(NH2_ASYM, "6-31g", multiplicity=2)
    with pytest.raises(ValueError, match="another AoIntegrals"):
        ints.veff(d, xc=other.xc_functional("PBE"))
