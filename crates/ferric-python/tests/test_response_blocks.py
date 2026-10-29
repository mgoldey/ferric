"""Response pieces: `XcFunctional.response`, `AoIntegrals.cphf` / `ucphf`.

Anchors are independent constructions:

* the RHF orbital-Hessian action equals the dense (A + B) assembled from
  `ao2mo` integrals, 4(ai|bj) - (ab|ij) - (aj|bi) + (e_a - e_i);
* the f_xc kernel equals a finite difference of V_xc (closed and polarized);
* static polarizabilities from `solve` equal finite-field second derivatives
  of `run_rhf` / `run_dft` / `run_uhf` energies (field H' = +F.r);
* the open-shell solver at U_a = U_b on a closed shell equals the closed one.
"""

from __future__ import annotations

import numpy as np
import pytest

import ferric

WATER = "3\n\nO 0 0 0.117\nH 0 0.757 -0.467\nH 0 -0.757 -0.467\n"
NH2_ASYM = "3\n\nN 0 0 0\nH 0.12 0.80 0.60\nH -0.05 -0.74 0.52\n"
# Range-separated: erfc AND erf exact exchange, mix (0.158, 1.0, 0.3).
WB97X = "HYB_GGA_XC_WB97X"
TIGHT = {"energy_conv": 1e-11, "density_conv": 1e-10}
EXACT_JK = {"df_j_aux": "none", "df_k_aux": "none"}
_RNG = np.random.default_rng(3)


def _ints(xyz, basis, multiplicity=1):
    mol = ferric.Molecule.from_xyz_string(xyz, multiplicity=multiplicity)
    bs = ferric.BasisSet.bundled(basis)
    return mol, bs, ferric.AoIntegrals(mol, bs)


def _canonical(ints, f):
    w, v = np.linalg.eigh(ints.overlap())
    x = v / np.sqrt(w)
    e, c = np.linalg.eigh(x.T @ f @ x)
    return e, x @ c


def _rhf_mos(mol, bs, ints):
    d = ferric.run_rhf(mol, bs, **TIGHT).density()
    j, k = ints.jk(d)
    return _canonical(ints, ints.hcore() + j - 0.5 * k)


def _rks_mos(mol, bs, ints, xc):
    d = ferric.run_dft(mol, bs, functional=xc.name, **EXACT_JK, **TIGHT).density()
    c_sr, c_lr, omega = xc.exchange_mix
    j, k = ints.jk(d)
    f = ints.hcore() + j - 0.5 * c_sr * k + xc.eval(d)[1]
    return _canonical(ints, f)


def test_rhf_matvec_equals_the_dense_orbital_hessian():
    mol, bs, ints = _ints(WATER, "sto-3g")
    e, c = _rhf_mos(mol, bs, ints)
    no = ints.nelec // 2
    co, cv = c[:, :no], c[:, no:]
    aibj = ints.ao2mo(cv, co, cv, co)
    abij = ints.ao2mo(cv, cv, co, co)
    ab = 4 * aibj - abij.transpose(0, 2, 1, 3) - aibj.transpose(0, 3, 2, 1)
    nv = cv.shape[1]
    denom = e[no:, None] - e[None, :no]
    ab = ab.reshape(nv * no, nv * no) + np.diag(denom.ravel())
    cphf = ints.cphf(c, list(e), no)
    u = _RNG.standard_normal((nv, no))
    np.testing.assert_allclose(cphf.matvec(u).ravel(), ab @ u.ravel(), atol=1e-10)
    b = _RNG.standard_normal((nv, no))
    np.testing.assert_allclose(
        cphf.solve(b, tol=1e-11),
        np.linalg.solve(ab, b.ravel()).reshape(nv, no),
        atol=1e-9,
    )


@pytest.mark.parametrize("functional", ["PBE", "B3LYP", WB97X])
def test_xc_kernel_is_the_derivative_of_vxc(functional):
    mol, bs, ints = _ints(NH2_ASYM, "6-31g", multiplicity=2)
    xc = ints.xc_functional(functional)
    r = ferric.run_uhf(mol, bs, **TIGHT)
    da, db = r.density_alpha(), r.density_beta()
    n, h = ints.nao, 1e-4
    x = 1e-2 * _RNG.standard_normal((n, n))
    x = x + x.T
    y = 1e-2 * _RNG.standard_normal((n, n))
    y = y + y.T
    v1 = xc.response(da + db, x)
    fd = (xc.eval(da + db + h * x)[1] - xc.eval(da + db - h * x)[1]) / (2 * h)
    np.testing.assert_allclose(v1, fd, atol=1e-7)
    va, vb = xc.response_polarized(da, db, x, y)
    p = xc.eval_polarized(da + h * x, db + h * y)
    m = xc.eval_polarized(da - h * x, db - h * y)
    np.testing.assert_allclose(va, (p[1] - m[1]) / (2 * h), atol=1e-7)
    np.testing.assert_allclose(vb, (p[2] - m[2]) / (2 * h), atol=1e-7)


def _alpha_closed(ints, cphf, c, no):
    dip = ints.dipole()
    co, cv = c[:, :no], c[:, no:]
    alpha = np.zeros((3, 3))
    for y in range(3):
        d1 = cphf.density(cphf.solve(-(cv.T @ dip[y] @ co), tol=1e-11))
        for x in range(3):
            alpha[x, y] = -np.sum(d1 * dip[x])
    return alpha


def _alpha_zz_finite_field(run):
    """-d2E/dF_z2, Richardson-extrapolated over F and 2F."""

    def second(f):
        return (run((0, 0, f)) - 2 * run(None) + run((0, 0, -f))) / f**2

    f = 2e-3
    return -(4 * second(f) - second(2 * f)) / 3


def test_rhf_polarizability_matches_finite_field():
    mol, bs, ints = _ints(WATER, "6-31g")
    e, c = _rhf_mos(mol, bs, ints)
    no = ints.nelec // 2
    alpha = _alpha_closed(ints, ints.cphf(c, list(e), no), c, no)
    np.testing.assert_allclose(alpha, alpha.T, atol=1e-8)

    def run(field):
        kw = {"external_field": field} if field else {}
        return ferric.run_rhf(mol, bs, **kw, **TIGHT).energy

    assert abs(alpha[2, 2] - _alpha_zz_finite_field(run)) < 1e-5 * abs(alpha[2, 2])


@pytest.mark.parametrize("functional", ["PBE", "B3LYP"])
def test_rks_polarizability_matches_finite_field(functional):
    mol, bs, ints = _ints(WATER, "6-31g")
    xc = ints.xc_functional(functional)
    e, c = _rks_mos(mol, bs, ints, xc)
    no = ints.nelec // 2
    alpha = _alpha_closed(ints, ints.cphf(c, list(e), no, xc=xc), c, no)

    def run(field):
        kw = {"external_field": field} if field else {}
        return ferric.run_dft(
            mol, bs, functional=functional, **kw, **EXACT_JK, **TIGHT
        ).total_energy

    assert abs(alpha[2, 2] - _alpha_zz_finite_field(run)) < 1e-4 * abs(alpha[2, 2])


def test_uhf_polarizability_matches_finite_field():
    mol, bs, ints = _ints(NH2_ASYM, "6-31g", multiplicity=2)
    r = ferric.run_uhf(mol, bs, **TIGHT)
    da, db = r.density_alpha(), r.density_beta()
    h = ints.hcore()
    j, _ = ints.jk(da + db)
    frames = []
    for dsig, nsig in ((da, 5), (db, 4)):
        e, c = _canonical(ints, h + j - ints.jk(dsig)[1])
        frames.append((e, c, nsig))
    (ea, ca, na), (eb, cb, nb) = frames
    ucphf = ints.ucphf(ca, cb, list(ea), list(eb), na, nb)
    dz = ints.dipole()[2]
    rhs = [-(c[:, n:].T @ dz @ c[:, :n]) for _, c, n in frames]
    d1a, d1b = ucphf.density(*ucphf.solve(*rhs, tol=1e-11))
    alpha_zz = -np.sum((d1a + d1b) * dz)

    def run(field):
        kw = {"external_field": field} if field else {}
        return ferric.run_uhf(mol, bs, **kw, **TIGHT).energy

    assert abs(alpha_zz - _alpha_zz_finite_field(run)) < 1e-5 * abs(alpha_zz)


@pytest.mark.parametrize("functional", [None, "PBE"])
def test_open_shell_response_at_equal_spins_equals_closed(functional):
    mol, bs, ints = _ints(WATER, "6-31g")
    xc = ints.xc_functional(functional) if functional else None
    e, c = _rks_mos(mol, bs, ints, xc) if xc else _rhf_mos(mol, bs, ints)
    no = ints.nelec // 2
    cphf = ints.cphf(c, list(e), no, xc=xc)
    ucphf = ints.ucphf(c, c, list(e), list(e), no, no, xc=xc)
    u = _RNG.standard_normal((c.shape[1] - no, no))
    ra, rb = ucphf.matvec(u, u)
    np.testing.assert_allclose(ra, cphf.matvec(u), atol=1e-9)
    np.testing.assert_allclose(rb, ra, atol=1e-12)
    da, db = ucphf.density(u, u)
    np.testing.assert_allclose(da + db, cphf.density(u), atol=1e-12)


def test_response_refusals():
    mol, bs, ints = _ints(WATER, "sto-3g")
    e, c = _rhf_mos(mol, bs, ints)
    no = ints.nelec // 2
    for name in ("SCAN", "wB97X-V"):
        with pytest.raises(RuntimeError, match="no exchange-correlation kernel"):
            ints.cphf(c, list(e), no, xc=ints.xc_functional(name))
    with pytest.raises(ValueError, match="nocc"):
        ints.cphf(c, list(e), c.shape[1])
    cphf = ints.cphf(c, list(e), no)
    with pytest.raises(ValueError, match=r"\(nvir, nocc\)"):
        cphf.matvec(np.zeros((no, c.shape[1] - no)))
    with pytest.raises(ValueError, match="positive"):
        ints.cphf(c, list(e[::-1]), no)


# ── density-fitted response (cphf/ucphf aux=) ──

AUX = "cc-pvdz-ri"
DF = {"df_j_aux": AUX, "df_k_aux": AUX}


def _df_ks_fock(ints, aux, xc, d):
    """h + J_fit - 0.5 * mix K_fit (+ V_xc): the Fock of an aux-fitted SCF."""
    c_sr, c_lr, omega = xc.exchange_mix if xc else (1.0, 1.0, 0.0)
    f = ints.hcore() + ints.df_jk(aux).j(d)
    if omega > 0:
        k = c_sr * ints.df_jk(aux, "erfc", omega=omega).k(d) + c_lr * ints.df_jk(
            aux, "erf", omega=omega
        ).k(d)
    else:
        k = c_sr * ints.df_jk(aux).k(d)
    f = f - 0.5 * k
    if xc:
        f = f + xc.eval(d)[1]
    return f


def test_df_rhf_matvec_equals_the_df_jk_assembly():
    mol, bs, ints = _ints(WATER, "6-31g")
    aux = ferric.BasisSet.bundled(AUX)
    d = ferric.run_rhf(mol, bs, **DF, **TIGHT).density()
    e, c = _canonical(ints, _df_ks_fock(ints, aux, None, d))
    no = ints.nelec // 2
    cphf = ints.cphf(c, list(e), no, aux=aux)
    u = _RNG.standard_normal((c.shape[1] - no, no))
    d1 = cphf.density(u)
    j, k = ints.df_jk(aux).jk(d1)
    ref = (e[no:, None] - e[None, :no]) * u + c[:, no:].T @ (j - 0.5 * k) @ c[:, :no]
    np.testing.assert_allclose(cphf.matvec(u), ref, atol=1e-10)
    # The fit is in use: the exact response differs.
    assert np.abs(ints.cphf(c, list(e), no).matvec(u) - ref).max() > 1e-6


@pytest.mark.parametrize("functional", [None, "B3LYP", WB97X])
def test_df_closed_polarizability_matches_finite_field(functional):
    mol, bs, ints = _ints(WATER, "6-31g")
    aux = ferric.BasisSet.bundled(AUX)
    xc = ints.xc_functional(functional) if functional else None

    def scf(field=None):
        kw = {"external_field": field} if field else {}
        if functional:
            return ferric.run_dft(mol, bs, functional=functional, **kw, **DF, **TIGHT)
        return ferric.run_rhf(mol, bs, **kw, **DF, **TIGHT)

    e, c = _canonical(ints, _df_ks_fock(ints, aux, xc, scf().density()))
    no = ints.nelec // 2
    alpha = _alpha_closed(ints, ints.cphf(c, list(e), no, xc=xc, aux=aux), c, no)

    def energy(field):
        r = scf(field)
        return r.total_energy if functional else r.energy

    assert abs(alpha[2, 2] - _alpha_zz_finite_field(energy)) < 1e-4 * abs(alpha[2, 2])


def test_df_uhf_polarizability_matches_finite_field():
    mol, bs, ints = _ints(NH2_ASYM, "6-31g", multiplicity=2)
    aux = ferric.BasisSet.bundled(AUX)
    df = ints.df_jk(aux)
    r = ferric.run_uhf(mol, bs, **DF, **TIGHT)
    da, db = r.density_alpha(), r.density_beta()
    h, j = ints.hcore(), df.j(da + db)
    frames = []
    for dsig, nsig in ((da, 5), (db, 4)):
        e, c = _canonical(ints, h + j - df.k(dsig))
        frames.append((e, c, nsig))
    (ea, ca, na), (eb, cb, nb) = frames
    ucphf = ints.ucphf(ca, cb, list(ea), list(eb), na, nb, aux=aux)
    dz = ints.dipole()[2]
    rhs = [-(c[:, n:].T @ dz @ c[:, :n]) for _, c, n in frames]
    d1a, d1b = ucphf.density(*ucphf.solve(*rhs, tol=1e-11))
    alpha_zz = -np.sum((d1a + d1b) * dz)

    def run(field):
        kw = {"external_field": field} if field else {}
        return ferric.run_uhf(mol, bs, **kw, **DF, **TIGHT).energy

    assert abs(alpha_zz - _alpha_zz_finite_field(run)) < 1e-5 * abs(alpha_zz)
