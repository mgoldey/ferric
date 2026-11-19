#!/usr/bin/env python3
"""Sympy/numpy checks of the closed- and open-shell dRPA energy identities.

Theory and notation: `book/src/18-rpa/01-drpa-three-ways.md#drpa-three-ways-plasmon-formula`, `book/src/18-rpa/03-pdep.md#pdep-dielectric-matrix-static-modes`, `book/src/18-rpa/04-open-shell-dielectric.md#open-shell-dielectric-spin-summed-dielectric`. Every check prints "FAIL [<name>]" when it fails, and the script
exits nonzero if any did.

The independent route is the definition: the spin-orbital direct RPA
(A = Delta + K, B = K over every spin-orbital ov pair, both spins, no
exchange), E_c = 1/2 sum_n (Omega_n - A_nn), solved by brute force. The code's
routes are transcribed and compared with it.

  1  One-mode closed form (sympy): with gap D and coupling b, the plasmon
     energy 1/2 (sqrt(D (D + 2 b)) - D - b), the drCCD Riccati root
     1/2 b t, b + 2 (D + b) t + b t^2 = 0, and the frequency integral
     1/(2 pi) int_0^inf [ln(1 + 2 b D/(w^2 + D^2)) - 2 b D/(w^2 + D^2)] dw
     are one function of (D, b). Its b^2 term is -b^2/(4 D) = 2 E_OS for
     b = 2 (ia|ia).
  2  Spin adaptation (numpy, closed shell): the spin-orbital definition equals
     canonical_plasmon_drpa (drpa_amplitude.rs: A = Delta + 2K, B = 2K,
     Omega^2 = eig (A - B)(A + B)).
  3  drCCD Riccati (drpa_amplitude.rs amplitude_drpa_dense / riccati_masked_solve,
     fock_super): the damped fixed point T <- T - R(T)/D,
     R = B + F(T) + B T + T B + T B T, run in a randomly rotated (localized)
     occupied and virtual basis with non-diagonal Fock blocks, gives
     E = 1/2 sum B o T equal to the definition.
  4  Frequency form (energy.rs trace_log_summands_of_projection,
     sternheimer.rs build_scale_factors): Pi(iw) = 4 Bt diag(e/(w^2 + e^2)) Bt^T,
     E = 1/(2 pi) int [ln det(1 + Pi) - tr Pi] dw equals the definition
     (adaptive quadrature, 1e-10), and the code's 48-point mapped
     Gauss-Legendre grid (quadrature.rs gauss_legendre_nodes, u0 = 0.5,
     energy_route.rs exact_drpa_config) is within 1e-6 relative of it.
  5  RI factor freedom (theory "The energy as a function of B and F"): the
     energy depends on Bt only through Bt^T Bt (Sylvester), so the Cholesky
     factor L^-1 (Q|ia) and the symmetric V^-1/2 (Q|ia) give the same energy.
  6  trace_log_summand_term (energy.rs): the series branch equals
     ln(lam) + 1 - lam, and the eigenvalue route equals the log-det route per
     frequency.
  7  PDEP projection (lib.rs run_pdep_rpa_eigensolve, energy.rs
     eval_eigenvalues_at_frequencies_budgeted): with the full static
     eigenbasis U of eps(0), the eigenvalues of U^T eps(iw) U (diagonalized at
     each frequency) give the aux log-det energy exactly; the diagonal
     <U_a|eps(iw)|U_a> alone does not (control). Truncation keeps the modes
     with |lam(0) - 1| > trunc_thresh and converges to the full energy.
  8  Open shell (sternheimer.rs build_scale_factors_with_prefactor at 2 per
     spin, energy.rs eval_eigenvalues_at_frequencies_unrestricted):
     eps_U = 1 + Pi_a + Pi_b equals the spin-orbital definition with different
     alpha and beta orbitals, gaps and occupations; with identical channels it
     equals the closed-shell Pi (prefactor 4).

  9  Intermediate steps the book writes out: T = Y X^-1 built from the
     plasmon eigenvectors is symmetric, solves the Riccati equation and gives
     the plasmon energy; det(1 + Pi(iu)) = prod_n (u^2 + Omega_n^2) /
     (u^2 + Delta_n^2) at three u; for any orthonormal U the eigenvalues of
     U^T eps U interlace those of eps, so a projected (truncated) summand is
     never below the full one (the PDEP truncation error is >= 0); and
     |ln(1 + x) - x| <= x^2/2 for x >= 0 (sympy).

Mutations (each turns the named check red):
  - SCALE_CS, the closed-shell Pi prefactor, 4.0 -> 2.0      [freq-vs-definition]
  - B2 in plasmon_closed, spin-adapted B = 2K, 2.0 -> 1.0     [plasmon-vs-definition]
  - SCALE_U, the per-spin Pi prefactor, 2.0 -> 4.0            [u-freq-vs-definition]
  - riccati_closed residual without its T B ring term         [riccati-vs-definition]
  - Cholesky factor on the wrong side, solve with L^T for L   [ri-factor-freedom]
  - T = X^-1 Y in place of Y X^-1                             [riccati-eigenvector-form]
  - the random projection scaled by 1.5 (not orthonormal)     [pdep-truncation-sign]
  - the static bound x^2/2 -> x^2/3                           [static-summand-bound]

Run: python3 scripts/rpa_drpa_pdep_sympy.py  (needs sympy, numpy, scipy)
"""

import sys

import mpmath
import numpy as np
import sympy as sp
from scipy.integrate import quad

FAILED = []


def check(name, ok, detail=""):
    if ok:
        print(f"ok   [{name}] {detail}")
    else:
        print(f"FAIL [{name}] {detail}")
        FAILED.append(name)


# Code constants transcribed from the Rust.
SCALE_CS = 4.0  # sternheimer.rs build_scale_factors: prefactor 4 (closed shell)
SCALE_U = 2.0  # build_scale_factors_with_prefactor(.., 2.0) per spin (open shell)


# ---------------------------------------------------------------- 1. one mode
def one_mode():
    D, b, w, t = sp.symbols("D b omega t", positive=True)
    omega_pl = sp.sqrt(D * (D + 2 * b))
    e_plasmon = sp.Rational(1, 2) * (omega_pl - D - b)
    # Riccati root continuous with t -> -b/(2D) at b -> 0.
    roots = sp.solve(b + 2 * (D + b) * t + b * t**2, t)
    root = [r for r in roots if sp.limit(r, b, 0) == 0][0]
    e_riccati = sp.Rational(1, 2) * b * root
    check(
        "one-mode-riccati",
        sp.simplify(e_riccati - e_plasmon) == 0,
        "1/2 b t(b) == 1/2 (Omega - D - b)",
    )
    # Frequency integral: differentiate in b under the integral sign; the
    # integrand vanishes at b = 0, so equal b-derivatives prove equality.
    pi_w = 2 * b * D / (w**2 + D**2)
    integrand = sp.log(1 + pi_w) - pi_w
    d_int = sp.integrate(sp.simplify(sp.diff(integrand, b)), (w, 0, sp.oo))
    d_freq = sp.simplify(d_int / (2 * sp.pi))
    check(
        "one-mode-frequency",
        sp.simplify(d_freq - sp.diff(e_plasmon, b)) == 0,
        f"d/db: {sp.simplify(d_freq)}",
    )
    second = sp.series(e_plasmon, b, 0, 3).removeO()
    k = sp.symbols("k", positive=True)  # k = (ia|ia), b = 2k, pair gap 2D
    e_os = k**2 / (-2 * D)
    check(
        "one-mode-second-order",
        sp.simplify(second.subs(b, 2 * k) - 2 * e_os) == 0,
        f"b^2 term {second} == 2 E_OS",
    )


# ------------------------------------------------------------- tiny models
rng = np.random.default_rng(20261114)


def model(no, nv, naux, scale=0.25):
    eo = np.sort(rng.uniform(-2.0, -0.6, no))
    ev = np.sort(rng.uniform(0.3, 2.5, nv))
    bt = scale * rng.standard_normal((naux, no * nv))
    return eo, ev, bt


def gaps(eo, ev):
    return (ev[None, :] - eo[:, None]).ravel()


def spin_orbital_definition(chans):
    """Spin-orbital dRPA: A = Delta + K, B = K over all spin channels."""
    blocks = [bt for (_, _, bt) in chans]
    bfull = np.hstack(blocks)  # (naux, sum nov_s)
    k = bfull.T @ bfull
    delta = np.concatenate([gaps(eo, ev) for (eo, ev, _) in chans])
    a = np.diag(delta) + k
    m = (a - k) @ (a + k)
    w2 = np.linalg.eigvals(m)
    assert np.all(np.abs(w2.imag) < 1e-10) and np.all(w2.real > 0)
    return 0.5 * (np.sum(np.sqrt(w2.real)) - np.trace(a))


def plasmon_closed(eo, ev, bt):
    """drpa_amplitude.rs canonical_plasmon_drpa."""
    K = bt.T @ bt
    B2 = 2.0 * K
    a = np.diag(gaps(eo, ev)) + B2
    w2 = np.linalg.eigvals((a - B2) @ (a + B2))
    return 0.5 * (np.sum(np.sqrt(w2.real)) - np.trace(a))


def riccati_closed(eo, ev, bt, max_iter=2000):
    """amplitude_drpa_dense in a rotated (localized) basis."""
    no, nv = len(eo), len(ev)
    o_rot = np.linalg.qr(rng.standard_normal((no, no)))[0]
    v_rot = np.linalg.qr(rng.standard_normal((nv, nv)))[0]
    f_oo = o_rot.T @ np.diag(eo) @ o_rot
    f_vv = v_rot.T @ np.diag(ev) @ v_rot
    b3 = bt.reshape(-1, no, nv)
    b3 = np.einsum("ij,Pjb,ba->Pia", o_rot.T, b3, v_rot)
    btl = b3.reshape(-1, no * nv)
    K = btl.T @ btl
    B2 = 2.0 * K
    dop = np.kron(np.eye(no), f_vv) - np.kron(f_oo, np.eye(nv))
    dd = np.diag(f_vv)[None, :] - np.diag(f_oo)[:, None]
    dd = dd.ravel()
    den = dd[:, None] + dd[None, :]
    t = np.zeros_like(B2)
    for _ in range(max_iter):
        f_t = dop @ t + t @ dop  # fock_super
        bt_ = B2 @ t
        rhs = B2 + f_t + bt_ + t @ B2 + t @ bt_
        if np.linalg.norm(rhs) < 1e-13 * np.linalg.norm(B2):
            break
        t = t - rhs / den
    return 0.5 * np.sum(B2 * t)


def trace_log_summand_term(lam):
    """energy.rs trace_log_summand_term."""
    x = lam - 1.0
    if abs(x) < 0.25:
        acc = 0.0
        for k in range(28, -1, -1):
            sign = -1.0 if k % 2 == 0 else 1.0
            acc = acc * x + sign / (k + 2)
        return acc * x * x
    return np.log(lam) - x


def pi_matrix(chans, w, scale):
    out = 0.0
    for eo, ev, bt in chans:
        e = gaps(eo, ev)
        g = scale * e / (w * w + e * e)
        out = out + (bt * g) @ bt.T
    return out


def summand(chans, w, scale):
    p = pi_matrix(chans, w, scale)
    sign, logdet = np.linalg.slogdet(np.eye(len(p)) + p)
    assert sign > 0
    return logdet - np.trace(p)


def freq_energy_exact(chans, scale):
    val, _ = quad(
        lambda w: summand(chans, w, scale),
        0,
        np.inf,
        epsabs=1e-14,
        epsrel=1e-13,
        limit=400,
    )
    return val / (2 * np.pi)


def gl_grid(n, u0):
    """quadrature.rs gauss_legendre_nodes."""
    x, wgl = np.polynomial.legendre.leggauss(n)
    return u0 * (1 + x) / (1 - x), wgl * 2 * u0 / (1 - x) ** 2


def freq_energy_grid(chans, scale, n=48, u0=0.5):
    f, w = gl_grid(n, u0)
    return sum(wk * summand(chans, fk, scale) for fk, wk in zip(f, w)) / (2 * np.pi)


def closed_shell_checks():
    eo, ev, bt = model(2, 3, 7)
    ref = spin_orbital_definition([(eo, ev, bt), (eo, ev, bt)])
    e_pl = plasmon_closed(eo, ev, bt)
    check(
        "plasmon-vs-definition", abs(e_pl - ref) < 1e-12, f"{e_pl:.15f} vs {ref:.15f}"
    )
    e_ric = riccati_closed(eo, ev, bt)
    check(
        "riccati-vs-definition", abs(e_ric - ref) < 1e-11, f"{e_ric:.15f} vs {ref:.15f}"
    )
    chans = [(eo, ev, bt)]
    e_fx = freq_energy_exact(chans, SCALE_CS)
    check("freq-vs-definition", abs(e_fx - ref) < 1e-10, f"{e_fx:.15f} vs {ref:.15f}")
    e_gl = freq_energy_grid(chans, SCALE_CS)
    check(
        "gl48-vs-definition",
        abs(e_gl - ref) < 1e-6 * abs(ref),
        f"rel {abs(e_gl - ref) / abs(ref):.2e}",
    )

    # 5. RI factor freedom: raw (Q|ia) = V^{1/2} Bt for a random metric V.
    naux = bt.shape[0]
    a = rng.standard_normal((naux, naux))
    v = a @ a.T + naux * np.eye(naux)
    wv, uv = np.linalg.eigh(v)
    qia = (uv * np.sqrt(wv)) @ uv.T @ bt
    lchol = np.linalg.cholesky(v)
    bt_chol = np.linalg.solve(lchol, qia)
    bt_sym = (uv / np.sqrt(wv)) @ uv.T @ qia
    e_chol = freq_energy_grid([(eo, ev, bt_chol)], SCALE_CS)
    e_sym = freq_energy_grid([(eo, ev, bt_sym)], SCALE_CS)
    check(
        "ri-factor-freedom",
        abs(e_chol - e_sym) < 1e-13,
        f"|dE| = {abs(e_chol - e_sym):.1e}",
    )
    f0, _ = gl_grid(48, 0.5)
    p = pi_matrix(chans, f0[5], SCALE_CS)
    e_ov = gaps(eo, ev)
    gsq = np.sqrt(SCALE_CS * e_ov / (f0[5] ** 2 + e_ov**2))
    kk = (bt * gsq).T @ (bt * gsq)
    s_aux = np.linalg.slogdet(np.eye(naux) + p)[1]
    s_ov = np.linalg.slogdet(np.eye(len(kk)) + kk)[1]
    check("sylvester", abs(s_aux - s_ov) < 1e-13, f"|d| = {abs(s_aux - s_ov):.1e}")

    # 6. trace-log summand and eigenvalue route.
    mpmath.mp.dps = 50

    def ref_summand(lam):
        x = mpmath.mpf(lam)
        return float(mpmath.log(x) + 1 - x)

    worst = max(
        abs(trace_log_summand_term(lam) - ref_summand(lam)) / abs(ref_summand(lam))
        for lam in [0.8, 0.95, 1 + 1e-9, 1.001, 1.1, 1.2499, 1.3, 3.0]
    )
    check(
        "series-summand",
        worst < 1e-14,
        f"rel {worst:.1e} vs 50-digit ln(lam) + 1 - lam",
    )
    lam = np.linalg.eigvalsh(np.eye(naux) + p)
    s_eig = sum(trace_log_summand_term(x) for x in lam)
    s_det = summand(chans, f0[5], SCALE_CS)
    check(
        "eig-vs-logdet", abs(s_eig - s_det) < 1e-13, f"|d| = {abs(s_eig - s_det):.1e}"
    )

    # 7. PDEP projection with the static eigenbasis.
    lam0, u = np.linalg.eigh(np.eye(naux) + pi_matrix(chans, 0.0, SCALE_CS))
    order = np.argsort(-np.abs(lam0 - 1))
    lam0, u = lam0[order], u[:, order]
    f, wq = gl_grid(48, 0.5)

    def pdep_energy(m, diagonal_only=False):
        y = u[:, :m].T @ bt
        tot = 0.0
        for fk, wk in zip(f, wq):
            eproj = np.eye(m) + pi_matrix([(eo, ev, y)], fk, SCALE_CS)
            lam_k = np.diag(eproj) if diagonal_only else np.linalg.eigvalsh(eproj)
            tot += wk * sum(trace_log_summand_term(x) for x in lam_k)
        return tot / (2 * np.pi)

    e_full = pdep_energy(naux)
    check(
        "pdep-full-rank", abs(e_full - e_gl) < 1e-13, f"|d| = {abs(e_full - e_gl):.1e}"
    )
    e_diag = pdep_energy(naux, diagonal_only=True)
    check(
        "pdep-diagonal-control",
        abs(e_diag - e_gl) > 1e-8,
        f"diag-only error {abs(e_diag - e_gl):.1e} (must be nonzero)",
    )
    errs = []
    for thresh in [1e-1, 1e-2, 1e-3, 0.0]:
        m = int(np.sum(np.abs(lam0 - 1) > thresh)) if thresh > 0 else naux
        errs.append(abs(pdep_energy(m) - e_gl))
    check(
        "pdep-truncation",
        errs[-1] < 1e-13 and errs[0] >= errs[2],
        f"errors {['%.1e' % e for e in errs]}",
    )


def open_shell_checks():
    eo_a, ev_a, bt_a = model(3, 3, 7)
    no_b, nv_b = 2, 4
    eo_b = np.sort(rng.uniform(-1.8, -0.7, no_b))
    ev_b = np.sort(rng.uniform(0.2, 2.2, nv_b))
    bt_b = 0.25 * rng.standard_normal((7, no_b * nv_b))
    chans = [(eo_a, ev_a, bt_a), (eo_b, ev_b, bt_b)]
    ref = spin_orbital_definition(chans)
    e_u = freq_energy_exact(chans, SCALE_U)
    check("u-freq-vs-definition", abs(e_u - ref) < 1e-10, f"{e_u:.15f} vs {ref:.15f}")
    eo, ev, bt = model(2, 3, 6)
    pa = pi_matrix([(eo, ev, bt), (eo, ev, bt)], 0.7, SCALE_U)
    pc = pi_matrix([(eo, ev, bt)], 0.7, SCALE_CS)
    check(
        "u-closed-shell-limit",
        np.max(np.abs(pa - pc)) < 1e-14,
        "Pi_a + Pi_b == Pi_closed",
    )


# ------------------------------------------------- 9. intermediate steps
def intermediate_step_checks():
    """Steps the book writes out between the conclusions checked above.

    Uses its own generator so the earlier checks keep their random models.
    """
    r2 = np.random.default_rng(20261119)
    no, nv, naux = 2, 3, 7
    eo = np.sort(r2.uniform(-2.0, -0.6, no))
    ev = np.sort(r2.uniform(0.3, 2.5, nv))
    bt = 0.25 * r2.standard_normal((naux, no * nv))
    d = gaps(eo, ev)
    kk = 2.0 * bt.T @ bt  # the book's K = 2(ia|jb)
    a, b = np.diag(d) + kk, kk
    n = len(d)

    # T = Y X^-1 from the plasmon eigenvectors (18.1, ring amplitudes).
    w, v = np.linalg.eig(np.block([[a, b], [-b, -a]]))
    assert np.max(np.abs(w.imag)) < 1e-12 and np.max(np.abs(v.imag)) < 1e-12
    pos = np.argsort(w.real)[n:]
    om = w.real[pos]
    x, y = v.real[:n, pos], v.real[n:, pos]
    t = y @ np.linalg.inv(x)
    res = np.linalg.norm(b + a @ t + t @ a + t @ b @ t) / np.linalg.norm(b)
    asym = np.max(np.abs(t - t.T))
    e_t = 0.5 * np.trace(b @ t)
    e_pl = 0.5 * (np.sum(om) - np.trace(a))
    check(
        "riccati-eigenvector-form",
        asym < 1e-12 and res < 1e-12 and abs(e_t - e_pl) < 1e-13,
        f"|T - T^T| {asym:.1e}, |R(T)|/|K| {res:.1e}, |1/2 tr K T - plasmon| "
        f"{abs(e_t - e_pl):.1e}",
    )

    # det(1 + Pi(iu)) = prod_n (u^2 + Omega_n^2) / (u^2 + Delta_n^2).
    worst = 0.0
    for u in (0.0, 0.7, 3.0):
        p = pi_matrix([(eo, ev, bt)], u, SCALE_CS)
        lhs = np.linalg.slogdet(np.eye(naux) + p)[1]
        rhs = np.sum(np.log((u * u + om**2) / (u * u + d**2)))
        worst = max(worst, abs(lhs - rhs))
    check("determinant-product", worst < 1e-12, f"max |ln det - ln prod| {worst:.1e}")

    # PDEP truncation has a sign: for any orthonormal U the eigenvalues mu of
    # U^T eps U interlace those of eps, 1 <= mu_i <= lam_i, and
    # s(lam) = ln lam + 1 - lam is <= 0 and decreasing on lam >= 1.
    ok, worst_gap = True, np.inf
    for u in (0.0, 0.4, 2.0):
        eps = np.eye(naux) + pi_matrix([(eo, ev, bt)], u, SCALE_CS)
        lam = np.sort(np.linalg.eigvalsh(eps))[::-1]
        s_full = sum(trace_log_summand_term(z) for z in lam)
        for m in range(1, naux):
            u_rand = np.linalg.qr(r2.standard_normal((naux, naux)))[0][:, :m]
            mu = np.sort(np.linalg.eigvalsh(u_rand.T @ eps @ u_rand))[::-1]
            s_proj = sum(trace_log_summand_term(z) for z in mu)
            ok &= bool(np.all(mu >= 1 - 1e-13) and np.all(mu <= lam[:m] + 1e-13))
            worst_gap = min(worst_gap, s_proj - s_full)
    check(
        "pdep-truncation-sign",
        ok and worst_gap >= -1e-14,
        f"interlacing {ok}; min (S_projected - S_full) {worst_gap:.1e} (must be >= 0)",
    )

    # The static-summand bound |s(1 + x)| <= x^2 / 2 for x >= 0 (18.3):
    # f(x) = ln(1 + x) - x + x^2/2 has f(0) = 0 and f'(x) = x^2/(1 + x) >= 0.
    xs = sp.symbols("x", nonnegative=True)
    f = sp.log(1 + xs) - xs + xs**2 / 2
    dfx = sp.simplify(sp.diff(f, xs) - xs**2 / (1 + xs))
    grid = np.concatenate([np.logspace(-8, 1, 200), [0.0]])
    num_ok = all(abs(trace_log_summand_term(1 + z)) <= z * z / 2 + 1e-300 for z in grid)
    check(
        "static-summand-bound",
        f.subs(xs, 0) == 0 and dfx == 0 and num_ok,
        "f(0) = 0, f' = x^2/(1 + x) >= 0; numeric |s| <= x^2/2 on 201 points",
    )


one_mode()
closed_shell_checks()
open_shell_checks()
intermediate_step_checks()

if FAILED:
    print(f"\n{len(FAILED)} check(s) failed: {FAILED}")
    sys.exit(1)
print("\nall dRPA/PDEP identities hold")
