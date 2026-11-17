#!/usr/bin/env python3
"""Sympy/numpy checks of the signed long-range fit of rs_mp2_rpa.rs (terf).

The terf kernel terf(r)/r has the Fourier transform
cos(k r0) exp(-k^2/(4 w^2)) / (2 pi^2 k^2), which is negative near k r0 = pi,
so its auxiliary metric V_t = (P|terf|Q) is indefinite. rs-mp2-rpa never
factors V_t. It fits Coulomb and terfc (both positive definite) and carries
the stacked factor B = [B_C; B_s] with the signature Sigma = diag(+1, -1):

  K_lr = B^T Sigma B = B_C^T B_C - B_s^T B_s,   B_x = L_x^-1 (P|x|ia),

and the dRPA energy per frequency is ln det(1 + Sigma Y S^2 Y^T) - tr(Sigma
Y S^2 Y^T), Y = B, S^2 = 4 e_ia/(w^2 + e_ia^2) closed shell (2 per spin open
shell), by LU (the matrix is not symmetric).

Book: chapter "Density fitting", section "The two factors" (the signed
factor), and chapter "RS-MP2 + LR-RPA", section "In the code". Code:
crates/ferric-mp2/src/rimp2.rs (signed_pair_block,
spin_components_from_signed_b_ov), crates/ferric-mp2/src/u_rimp2.rs
(u_ri_mp2_signed), crates/ferric-rpa/src/energy.rs
(signed_trace_log_summands), crates/ferric-rpa/src/rs_mp2_rpa.rs.

Checks (each prints "FAIL [<name>]" when it fails; exit status nonzero):
  1  [signed-factor]       sympy: B^T Sigma B = B_C^T B_C - B_s^T B_s for
                           symbolic blocks, and for densities in the aux span
                           (J_x = V_x c, exact rationals) Coulomb RI minus
                           terfc RI is c^T V_t c exactly.
  2  [one-mode-logdet]     sympy: d/dk of (1/2pi) int_0^inf [ln(1 + 4Dk/(D^2
                           + w^2)) - 4Dk/(D^2 + w^2)] dw equals d/dk of the
                           plasmon value 1/2 (sqrt(D(D + 4k)) - D - 2k); both
                           vanish at k = 0, so the identity holds for every
                           k > -D/4, negative k included. Numerical check at
                           k = -D/8 by Gauss-Legendre.
  3  [sylvester-signed]    det(1_m + Sigma Y G Y^T) = det(1_n + G Y^T Sigma Y)
                           for random Y and diagonal G > 0.
  4  [logdet-vs-plasmon]   closed shell: the frequency integral of the signed
                           LU summand equals the plasmon dRPA of the
                           indefinite K = B_C^T B_C - B_s^T B_s.
  5  [u-logdet-vs-plasmon] open shell: two spin channels on one signed aux
                           space (prefactor 2 each) equal the spin-orbital
                           direct-RPA plasmon energy.
  6  [kernel-fourier]      independent route for the kernel: the Fourier
                           integral of cos(k r0) exp(-k^2/4w^2) reproduces
                           terf(r)/r = [erf(w(r - r0)) + erf(w(r + r0))]/(2r)
                           and the Coulomb Gram of two s Gaussians matches
                           erf(sqrt(p) R)/R.
  7  [signed-is-exact]     on s-Gaussian Gram matrices with an indefinite
                           V_t (the precondition is asserted), densities in
                           the aux span: Coulomb RI minus terfc RI and the
                           signed pseudo-inverse reproduce c^T V_t c, while
                           the positive-subspace pseudo-inverse (negative
                           modes dropped) misses it [drop-misses].

Mutations (each turns the named check red):
  - the terfc block enters with signature +1                 [signed-factor]
  - the signed pseudo-inverse keeps only lambda > 1e-10      [signed-is-exact]
  - closed-shell chi0 prefactor 4 -> 2                       [logdet-vs-plasmon]

Run: python3 scripts/rpa_terf_signed_fit_sympy.py  (needs sympy, numpy)
"""

import math
import sys

import numpy as np
import sympy as sp

FAILED = []
SIGMA_SR = -1.0
SCALE_CS = 4.0
SCALE_U = 2.0


def check(name, ok, detail=""):
    if ok:
        print(f"ok   [{name}] {detail}")
    else:
        print(f"FAIL [{name}] {detail}")
        FAILED.append(name)


# ------------------------------------------------------- 1. signed factor
def signed_factor():
    bc = sp.Matrix(2, 3, sp.symbols("c0:6"))
    bs = sp.Matrix(2, 3, sp.symbols("s0:6"))
    stacked = bc.col_join(bs)
    sigma = sp.diag(1, 1, sp.Integer(int(SIGMA_SR)), sp.Integer(int(SIGMA_SR)))
    lhs = stacked.T * sigma * stacked
    ok_sym = sp.simplify(lhs - (bc.T * bc - bs.T * bs)) == sp.zeros(3, 3)
    # Exact rationals: V_C, V_s positive definite, V_t = V_C - V_s indefinite.
    vc = sp.Matrix([[4, 1, 0], [1, 3, 1], [0, 1, 2]])
    vs = sp.Matrix([[3, 0, 0], [0, 2, 0], [0, 0, 1]])
    vt = vc - vs
    c = sp.Matrix([[1, 2], [-1, 1], [3, 0]])
    jc, js = vc * c, vs * c
    lc = vc.cholesky()
    ls = vs.cholesky()
    b = (lc.inv() * jc).col_join(ls.inv() * js)
    sig = sp.diag(*([1] * 3 + [int(SIGMA_SR)] * 3))
    k_signed = b.T * sig * b
    ok_exact = sp.simplify(k_signed - c.T * vt * c) == sp.zeros(2, 2)
    neg = min(float(x) for x in vt.eigenvals())
    check(
        "signed-factor",
        ok_sym and ok_exact and neg < 0,
        f"B^T Sigma B = K_C - K_s; in-span fit = c^T V_t c exactly (min eig V_t {neg:.3f})",
    )


# ---------------------------------------------------- 2. one-mode identity
def one_mode_logdet():
    d, a, w = sp.symbols("D a omega", positive=True)
    k = sp.symbols("k", real=True)
    # d/dk of the integrand: 4D/(D^2 + w^2 + 4Dk) - 4D/(D^2 + w^2).
    # int_0^inf 1/(a^2 + w^2) dw = pi/(2a).
    base = sp.integrate(1 / (a**2 + w**2), (w, 0, sp.oo))
    lhs = (4 * d * base.subs(a, sp.sqrt(d**2 + 4 * d * k)) - 4 * d * base.subs(a, d)) / (
        2 * sp.pi
    )
    plasmon = sp.Rational(1, 2) * (sp.sqrt(d * (d + 4 * k)) - d - 2 * k)
    rhs = sp.diff(plasmon, k)
    ok_sym = sp.simplify(sp.simplify(lhs - rhs).subs(k, -d / 8)) == 0 and (
        sp.simplify((lhs - rhs).subs({d: 1, k: sp.Rational(-1, 5)})) == 0
    )
    ok_zero = plasmon.subs(k, 0) == 0
    # Numerical: D = 1.3, k = -D/8 (indefinite one-mode kernel).
    dd, kk = 1.3, -1.3 / 8
    x, wt = np.polynomial.legendre.leggauss(400)
    om = (1 + x) / (1 - x)
    wom = wt * 2 / (1 - x) ** 2
    p = SCALE_CS * dd * kk / (dd * dd + om * om)
    e_int = np.sum(wom * (np.log1p(p) - p)) / (2 * np.pi)
    e_pl = 0.5 * (math.sqrt(dd * (dd + 4 * kk)) - dd - 2 * kk)
    check(
        "one-mode-logdet",
        ok_sym and ok_zero and abs(e_int - e_pl) < 1e-12,
        f"k = -D/8: integral {e_int:.15f} plasmon {e_pl:.15f}",
    )


# ------------------------------------------------------------- models
rng = np.random.default_rng(11)


def model(no, nv):
    eo = np.sort(rng.uniform(-1.6, -0.6, no))
    ev = np.sort(rng.uniform(0.4, 2.0, nv))
    return (ev[None, :] - eo[:, None]).ravel()


def gl_frequencies(n=200, u0=0.5):
    x, wt = np.polynomial.legendre.leggauss(n)
    om = u0 * (1 + x) / (1 - x)
    return om, wt * 2 * u0 / (1 - x) ** 2


def signed_logdet_energy(channels, n_pos, n_neg):
    """(1/2pi) sum_k w_k [ln det(1 + Sigma P) - tr(Sigma P)], P = sum_c Y_c S_c^2 Y_c^T."""
    sig = np.r_[np.ones(n_pos), SIGMA_SR * np.ones(n_neg)]
    om, wom = gl_frequencies()
    e = 0.0
    for o, wq in zip(om, wom):
        p = np.zeros((n_pos + n_neg, n_pos + n_neg))
        for y, d, pref in channels:
            p += (y * (pref * d / (d * d + o * o))) @ y.T
        sp_ = sig[:, None] * p
        s, ld = np.linalg.slogdet(np.eye(len(sig)) + sp_)
        assert s > 0
        e += wq * (ld - np.trace(sp_))
    return e / (2 * np.pi)


def plasmon(a_minus_b, a_plus_b, a_diag):
    sd = np.sqrt(np.diag(a_minus_b))
    m = sd[:, None] * a_plus_b * sd[None, :]
    w = np.linalg.eigvalsh(m)
    assert w.min() > 0
    return 0.5 * (np.sqrt(w).sum() - np.sum(a_diag))


# ------------------------------------------------- 3. Sylvester (signed)
def sylvester_signed():
    m, n = 7, 11
    y = rng.standard_normal((m, n))
    sig = np.r_[np.ones(4), -np.ones(3)]
    g = np.diag(rng.uniform(0.01, 0.1, n))
    l1 = np.linalg.slogdet(np.eye(m) + sig[:, None] * (y @ g @ y.T))
    l2 = np.linalg.slogdet(np.eye(n) + g @ y.T @ (sig[:, None] * y))
    check(
        "sylvester-signed",
        l1[0] == l2[0] and abs(l1[1] - l2[1]) < 1e-12,
        f"{l1[1]:.15f} vs {l2[1]:.15f}",
    )


# ------------------------------------- 4. closed shell: LU vs plasmon
def logdet_vs_plasmon():
    d = model(3, 4)
    n = d.size
    bc = 0.30 * rng.standard_normal((6, n))
    bs = 0.12 * rng.standard_normal((5, n))
    k = bc.T @ bc - bs.T @ bs
    kmin = np.linalg.eigvalsh(k)[0]
    y = np.vstack([bc, bs])
    e_lu = signed_logdet_energy([(y, d, SCALE_CS)], 6, 5)
    # Closed-shell singlet direct RPA: A - B = D, A + B = D + 4K.
    e_pl = plasmon(np.diag(d), np.diag(d) + 4 * k, d + 2 * np.diag(k))
    check(
        "logdet-vs-plasmon",
        kmin < 0 and abs(e_lu - e_pl) < 1e-10,
        f"min eig K {kmin:.3f}; LU {e_lu:.12f} plasmon {e_pl:.12f}",
    )


# --------------------------------------- 5. open shell: LU vs plasmon
def u_logdet_vs_plasmon():
    da, db = model(3, 4), model(2, 5)
    na, nb = da.size, db.size
    bca, bcb = 0.3 * rng.standard_normal((6, na)), 0.3 * rng.standard_normal((6, nb))
    bsa, bsb = 0.12 * rng.standard_normal((5, na)), 0.12 * rng.standard_normal((5, nb))
    ya, yb = np.vstack([bca, bsa]), np.vstack([bcb, bsb])
    e_lu = signed_logdet_energy([(ya, da, SCALE_U), (yb, db, SCALE_U)], 6, 5)
    # Spin-orbital direct RPA over the alpha and beta excitations:
    # K = Y^T Sigma Y on the concatenated columns, A - B = D, A + B = D + 2K.
    sig = np.r_[np.ones(6), -np.ones(5)]
    y = np.hstack([ya, yb])
    k = y.T @ (sig[:, None] * y)
    d = np.r_[da, db]
    e_pl = plasmon(np.diag(d), np.diag(d) + 2 * k, d + np.diag(k))
    check(
        "u-logdet-vs-plasmon",
        np.linalg.eigvalsh(k)[0] < 0 and abs(e_lu - e_pl) < 1e-10,
        f"LU {e_lu:.12f} plasmon {e_pl:.12f}",
    )


# ------------------------------------------- 6./7. s-Gaussian Gram model
GX, GW = np.polynomial.legendre.leggauss(400)


def kgrid(kmax, nseg=8):
    edges = np.linspace(0.0, kmax, nseg + 1)
    ks = [0.5 * (b - a) * GX + 0.5 * (b + a) for a, b in zip(edges[:-1], edges[1:])]
    ws = [0.5 * (b - a) * GW for a, b in zip(edges[:-1], edges[1:])]
    return np.concatenate(ks), np.concatenate(ws)


def gram(f, e1, c1, e2=None, c2=None):
    """(rho_a|v|rho_b) for unit-charge s Gaussians on the z axis:
    (2/pi) int_0^inf F(k) exp(-k^2/4a - k^2/4b) sinc(kR) dk, with
    v(k) = F(k)/(2 pi^2 k^2)."""
    e2 = e1 if e2 is None else e2
    c2 = c1 if c2 is None else c2
    kmax = 1.5 * math.sqrt(160.0 * min(min(e1), min(e2)))
    k, wk = kgrid(kmax)
    fk = f(k)
    v = np.zeros((len(e1), len(e2)))
    for i, (a, ra) in enumerate(zip(e1, c1)):
        for j, (b, rb) in enumerate(zip(e2, c2)):
            s = np.sinc(k * abs(ra - rb) / np.pi)
            v[i, j] = (2 / np.pi) * np.sum(wk * fk * np.exp(-k * k * (0.25 / a + 0.25 / b)) * s)
    return v


R0 = 1.35 * 1.8897261246257702
OMEGA = 4.0 / R0


def f_terf(k):
    return np.cos(k * R0) * np.exp(-k * k / (4 * OMEGA * OMEGA))


def kernel_fourier():
    # terf(r)/r from its transform, against the real-space definition.
    k, wk = kgrid(12.0 * OMEGA, 16)
    errs = []
    for r in (0.5, 1.7, 2.551, 4.0, 7.3):
        vk = (2 / np.pi) * np.sum(wk * f_terf(k) * np.sinc(k * r / np.pi))
        vr = 0.5 * (math.erf(OMEGA * (r - R0)) + math.erf(OMEGA * (r + R0))) / r
        errs.append(abs(vk - vr))
    a, b, rr = 0.7, 1.9, 2.3
    p = a * b / (a + b)
    vq = gram(np.ones_like, [a], [0.0], [b], [rr])[0, 0]
    errs.append(abs(vq - math.erf(math.sqrt(p) * rr) / rr))
    check("kernel-fourier", max(errs) < 1e-10, f"max error {max(errs):.1e}")


def pinv(v, signed):
    w, u = np.linalg.eigh(v)
    keep = np.abs(w) > 1e-10 if signed else w > 1e-10
    return (u[:, keep] / w[keep]) @ u[:, keep].T


def signed_is_exact():
    rr = 3.6 * 1.8897261246257702
    ex = [0.08, 0.2, 0.5, 1.25, 3.0, 7.0] * 2
    ce = [0.0] * 6 + [rr] * 6
    vc = gram(np.ones_like, ex, ce)
    vt = gram(f_terf, ex, ce)
    vs = gram(lambda k: 1.0 - f_terf(k), ex, ce)
    n_neg = int((np.linalg.eigvalsh(vt) < -1e-10).sum())
    c = np.random.default_rng(1).normal(size=(len(ex), 3))
    exact = c.T @ vt @ c
    jc, jt, js = vc @ c, vt @ c, vs @ c
    lc, ls = np.linalg.cholesky(vc), np.linalg.cholesky(vs)
    y = np.vstack([np.linalg.solve(lc, jc), np.linalg.solve(ls, js)])
    sig = np.r_[np.ones(len(ex)), SIGMA_SR * np.ones(len(ex))]
    k_cms = y.T @ (sig[:, None] * y)
    k_sp = jt.T @ pinv(vt, True) @ jt
    k_drop = jt.T @ pinv(vt, False) @ jt
    scale = np.abs(exact).max()
    e_cms = np.abs(k_cms - exact).max() / scale
    e_sp = np.abs(k_sp - exact).max() / scale
    e_drop = np.abs(k_drop - exact).max() / scale
    check(
        "signed-is-exact",
        n_neg > 0 and e_cms < 1e-12 and e_sp < 1e-9,
        f"{n_neg} negative modes of V_t; rel error C-minus-sr {e_cms:.1e}, signed pinv {e_sp:.1e}",
    )
    check("drop-misses", e_drop > 1e-2, f"positive-subspace rel error {e_drop:.2f}")


signed_factor()
one_mode_logdet()
sylvester_signed()
logdet_vs_plasmon()
u_logdet_vs_plasmon()
kernel_fourier()
signed_is_exact()

if FAILED:
    print(f"\n{len(FAILED)} check(s) failed: {FAILED}")
    sys.exit(1)
print("\nall signed-fit identities hold")
