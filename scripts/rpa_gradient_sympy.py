#!/usr/bin/env python3
"""Sympy/numpy checks of the analytic RI-dRPA gradient (gradient_analytic.rs).

Theory and notation: crates/ferric-rpa/theory.md, section "Analytic nuclear
gradient (closed-shell RI-dRPA, HF reference)" and its subsections. Every
check prints "FAIL [<name>]" when it fails, and the script exits nonzero if
any did.

  1  Divided difference (sympy, gradient_analytic.rs g_divided_difference):
     (g(x) - g(y))/(x - y) = 4(w^2 - x y)/((w^2 + x^2)(w^2 + y^2)) for
     g(e) = 4e/(w^2 + e^2), with limit g'(x) at y = x.
  2  Second-order limit (sympy): int_0^inf x y/((w^2+x^2)(w^2+y^2)) dw
     = pi/(2(x + y)), the integral that turns -1/2 tr Pi^2 into the
     direct-ring MP2 energy 2 sum (ia|jb)^2/D.
  3  First variation (numpy, rpa_ri_densities as coded): on a random model,
     Gamma = 4 x_ov and P = 2 p_oo, 2 p_vv (the code's normalization,
     Mp2Intermediates::from_ri_densities) are the derivatives of the n-point
     energy E_c(B, F_oo, F_vv) with respect to B and to the full symmetric
     Fock blocks (central differences, the Fock blocks taken off-diagonal so
     the Daleckii-Krein off-diagonal elements are tested).
  4  Full gradient (numpy): a toy closed-shell system with x-dependent
     overlap S(x), core Hamiltonian h(x) and exactly factorized integrals
     (mn|ls) = sum_P L_P,mn(x) L_P,ls(x) (unit RI metric). The theory's
     Lagrangian gradient (Y, P^AO, L, Z-vector A z = L, D_2, W, the Gamma
     integral term) equals a Richardson central difference of
     E_RHF + E_c with the RHF re-solved at x +/- h, with and without a
     frozen core orbital (the P_Ij block of "Frozen core").

Mutations (each turns the named check red):
  - Gamma accumulation scale w/(4 pi) -> w/(2 pi)             [gamma-vs-fd]
  - divided-difference numerator w^2 - xy -> w^2 + xy         [p-vs-fd]
  - W_ai = z_ai e_i -> 1/2 z_ai e_i                           [gradient-vs-fd]
  - frozen-core P_Ij without its factor 1/2                   [gradient-vs-fd-frozen-core]
  - Z-vector right-hand side 4 G[P^AO] -> 2 G[P^AO]           [gradient-vs-fd]

Run: python3 scripts/rpa_gradient_sympy.py  (needs sympy, numpy, scipy)
"""

import sys

import numpy as np
import sympy as sp
from scipy.linalg import eigh

FAILED = []


def check(name, ok, detail=""):
    if ok:
        print(f"ok   [{name}] {detail}")
    else:
        print(f"FAIL [{name}] {detail}")
        FAILED.append(name)


# ------------------------------------------------------------- 1, 2 sympy
def symbolic():
    x, y, w = sp.symbols("x y omega", positive=True)

    def g(e):
        return 4 * e / (w**2 + e**2)

    dd = sp.simplify((g(x) - g(y)) / (x - y))
    coded = 4 * (w**2 - x * y) / ((w**2 + x**2) * (w**2 + y**2))
    check("divided-difference", sp.simplify(dd - coded) == 0, f"{sp.factor(dd)}")
    check(
        "divided-difference-limit",
        sp.simplify(coded.subs(y, x) - sp.diff(g(x), x)) == 0,
        "g1(x, x) = g'(x)",
    )
    integ = sp.integrate(x * y / ((w**2 + x**2) * (w**2 + y**2)), (w, 0, sp.oo))
    check(
        "second-order-integral",
        sp.simplify(integ - sp.pi / (2 * (x + y))) == 0,
        f"{sp.simplify(integ)}",
    )


# ------------------------------------------------- code transcription (3)
def gl_grid(n, u0=0.5):
    """quadrature.rs gauss_legendre_nodes."""
    xg, wg = np.polynomial.legendre.leggauss(n)
    return u0 * (1 + xg) / (1 - xg), wg * 2 * u0 / (1 - xg) ** 2


FREQS, WEIGHTS = gl_grid(8)


def g_divided_difference(x, y, omega):
    w2 = omega * omega
    return 4.0 * (w2 - x * y) / ((w2 + x * x) * (w2 + y * y))


def rpa_ri_densities(b_ov, eps_occ, eps_vir):
    """gradient_analytic.rs rpa_ri_densities, RingOrder::Full."""
    no, nv = len(eps_occ), len(eps_vir)
    gaps = (eps_vir[None, :] - eps_occ[:, None]).ravel()
    naux = b_ov.shape[0]
    e_c = 0.0
    x_ov = np.zeros_like(b_ov)
    p_oo = np.zeros((no, no))
    p_vv = np.zeros((nv, nv))
    for omega, wk in zip(FREQS, WEIGHTS):
        g = 4.0 * gaps / (omega * omega + gaps * gaps)
        pi = (b_ov * g) @ b_ov.T
        lam, u = np.linalg.eigh(pi)
        s_k = np.sum(np.log1p(lam) - lam)
        q = (u * (-lam / (1.0 + lam))) @ u.T
        q_b = q @ b_ov
        e_c += wk * s_k / (2.0 * np.pi)
        xs = wk / (4.0 * np.pi)
        x_ov += xs * q_b * g
        ps = wk / (4.0 * np.pi)
        b3 = b_ov.reshape(naux, no, nv)
        qb3 = q_b.reshape(naux, no, nv)
        e2 = gaps.reshape(no, nv)
        for i in range(no):
            m = b3[:, i, :].T @ qb3[:, i, :]
            p_vv += ps * m * g_divided_difference(e2[i][:, None], e2[i][None, :], omega)
        for a in range(nv):
            m = b3[:, :, a].T @ qb3[:, :, a]
            p_oo += (
                -ps
                * m
                * g_divided_difference(e2[:, a][:, None], e2[:, a][None, :], omega)
            )
    return e_c, x_ov, p_oo, p_vv


def energy_noncanonical(b_ov, f_oo, f_vv):
    """E_c(B, F_oo, F_vv) with g(Delta) as a matrix function (theory eq.)."""
    no, nv = len(f_oo), len(f_vv)
    delta = np.kron(np.eye(no), f_vv) - np.kron(f_oo, np.eye(nv))
    ev, uv = np.linalg.eigh(delta)
    e_c = 0.0
    for omega, wk in zip(FREQS, WEIGHTS):
        gmat = (uv * (4.0 * ev / (omega**2 + ev**2))) @ uv.T
        pi = b_ov @ gmat @ b_ov.T
        lam = np.linalg.eigvalsh(pi)
        e_c += wk * np.sum(np.log1p(lam) - lam) / (2.0 * np.pi)
    return e_c


def first_variation():
    rng = np.random.default_rng(3)
    no, nv, naux = 2, 3, 6
    eo = np.array([-1.3, -0.8])
    ev = np.array([0.4, 0.9, 1.7])
    b = 0.3 * rng.standard_normal((naux, no * nv))
    e_c, x_ov, p_oo, p_vv = rpa_ri_densities(b, eo, ev)
    gamma, p_o, p_v = 4.0 * x_ov, 2.0 * p_oo, 2.0 * p_vv
    e_nc = energy_noncanonical(b, np.diag(eo), np.diag(ev))
    check(
        "energy-invariant-form", abs(e_nc - e_c) < 1e-14, f"|d| = {abs(e_nc - e_c):.1e}"
    )
    h = 1e-5
    fd = np.zeros_like(b)
    for idx in np.ndindex(b.shape):
        bp, bm = b.copy(), b.copy()
        bp[idx] += h
        bm[idx] -= h
        fd[idx] = (
            energy_noncanonical(bp, np.diag(eo), np.diag(ev))
            - energy_noncanonical(bm, np.diag(eo), np.diag(ev))
        ) / (2 * h)
    err = np.max(np.abs(fd - gamma))
    check("gamma-vs-fd", err < 1e-9, f"max |Gamma - dE/dB| = {err:.1e}")

    def dfock(which, p, q):
        fo, fv = np.diag(eo).copy(), np.diag(ev).copy()
        out = []
        for sgn in (1, -1):
            f = fo.copy() if which == "o" else fv.copy()
            f[p, q] += sgn * h
            if p != q:
                f[q, p] += sgn * h
            out.append(
                energy_noncanonical(b, f, fv)
                if which == "o"
                else energy_noncanonical(b, fo, f)
            )
        # symmetric perturbation of (p,q) and (q,p): derivative is 2 P_pq off-diagonal
        return (out[0] - out[1]) / (2 * h) / (1 if p == q else 2)

    err_p = max(
        max(abs(dfock("o", p, q) - p_o[p, q]) for p in range(no) for q in range(no)),
        max(abs(dfock("v", p, q) - p_v[p, q]) for p in range(nv) for q in range(nv)),
    )
    check(
        "p-vs-fd",
        err_p < 1e-9,
        f"max |P - dE/dF| = {err_p:.1e} (off-diagonal elements included)",
    )


# ---------------------------------------------------- 4. toy-model gradient
class Toy:
    def __init__(self, seed, n=6, naux=10, nocc=3):
        r = np.random.default_rng(seed)
        self.n, self.naux, self.nocc = n, naux, nocc

        def sym(m):
            return 0.5 * (m + m.T)

        self.h0 = np.diag([-4.0, -3.2, -2.6, 0.2, 0.7, 1.3]) + 0.1 * sym(
            r.standard_normal((n, n))
        )
        self.h1 = 0.2 * sym(r.standard_normal((n, n)))
        self.s1 = 0.05 * sym(r.standard_normal((n, n)))
        self.s2 = 0.02 * sym(r.standard_normal((n, n)))
        self.l0 = np.array([0.25 * sym(r.standard_normal((n, n))) for _ in range(naux)])
        self.l1 = np.array([0.1 * sym(r.standard_normal((n, n))) for _ in range(naux)])

    def at(self, x):
        s = np.eye(self.n) + x * self.s1 + x * x * self.s2
        return s, self.h0 + x * self.h1, self.l0 + x * self.l1

    def ds(self, x):
        return self.s1 + 2 * x * self.s2


def jk(lt, d):
    j = np.einsum("Pmn,P->mn", lt, np.einsum("Pls,ls->P", lt, d))
    k = np.einsum("Pml,ls,Psn->mn", lt, d, lt)
    return j, k


def g_op(lt, d):
    j, k = jk(lt, d)
    return j - 0.5 * k


def g_deriv(lt, dl, d):
    """G^[x][D]: the integral derivative contracted with a fixed density."""
    j1 = np.einsum("Pmn,P->mn", dl, np.einsum("Pls,ls->P", lt, d)) + np.einsum(
        "Pmn,P->mn", lt, np.einsum("Pls,ls->P", dl, d)
    )
    k1 = np.einsum("Pml,ls,Psn->mn", dl, d, lt) + np.einsum("Pml,ls,Psn->mn", lt, d, dl)
    return j1 - 0.5 * k1


def rhf(s, h, lt, nocc, c_guess=None):
    _, c = eigh(h, s) if c_guess is None else (None, c_guess)
    d = 2 * c[:, :nocc] @ c[:, :nocc].T
    hist_f, hist_e = [], []
    for _ in range(300):
        f = h + g_op(lt, d)
        err = f @ d @ s - s @ d @ f
        hist_f.append(f)
        hist_e.append(err)
        hist_f, hist_e = hist_f[-8:], hist_e[-8:]
        nb = len(hist_f)
        bmat = -np.ones((nb + 1, nb + 1))
        bmat[nb, nb] = 0
        for a in range(nb):
            for b in range(nb):
                bmat[a, b] = np.sum(hist_e[a] * hist_e[b])
        rhs = np.zeros(nb + 1)
        rhs[nb] = -1
        coef = np.linalg.solve(bmat, rhs)[:nb]
        f_ext = sum(cf * fi for cf, fi in zip(coef, hist_f))
        eps, c = eigh(f_ext, s)
        d_new = 2 * c[:, :nocc] @ c[:, :nocc].T
        if np.max(np.abs(d_new - d)) < 1e-14 and np.max(np.abs(err)) < 1e-13:
            d = d_new
            break
        d = d_new
    f = h + g_op(lt, d)
    eps, c = eigh(f, s)
    e_hf = 0.5 * np.sum(d * (h + f))
    return e_hf, eps, c


def total_energy(toy, x, nfc):
    s, h, lt = toy.at(x)
    e_hf, eps, c = rhf(s, h, lt, toy.nocc)
    no = toy.nocc
    b = np.einsum("mi,Pmn,na->Pia", c[:, nfc:no], lt, c[:, no:]).reshape(toy.naux, -1)
    e_c, *_ = rpa_ri_densities(b, eps[nfc:no], eps[no:])
    return e_hf + e_c


def analytic_gradient(toy, x, nfc):
    s, h, lt = toy.at(x)
    sx, hx, lx = toy.ds(x), toy.h1, toy.l1
    _, eps, c = rhf(s, h, lt, toy.nocc)
    n, nocc, naux = toy.n, toy.nocc, toy.naux
    nv = n - nocc
    co, cv = c[:, :nocc], c[:, nocc:]
    d_hf = 2 * co @ co.T
    # HF gradient (standard, independent of the correlation Lagrangian).
    w_hf = 2 * co @ np.diag(eps[:nocc]) @ co.T
    de_hf = (
        np.sum(d_hf * hx)
        + 0.5 * np.sum(d_hf * g_deriv(lt, lx, d_hf))
        - np.sum(w_hf * sx)
    )

    bmo = np.einsum("mp,Pmn,nq->Ppq", c, lt, c)  # B^P_pq, unit metric
    b_ov = bmo[:, nfc:nocc, nocc:].reshape(naux, -1)
    _, x_ov, p_oo, p_vv = rpa_ri_densities(b_ov, eps[nfc:nocc], eps[nocc:])
    gamma = (4.0 * x_ov).reshape(naux, nocc - nfc, nv)  # Gamma^P_ia, active i
    p_mat = np.zeros((n, n))
    p_mat[nfc:nocc, nfc:nocc] = 2.0 * p_oo
    p_mat[nocc:, nocc:] = 2.0 * p_vv

    # Y_pq (orbital derivative of E_c): columns i (active occupied) and a.
    y_mat = np.zeros((n, n))
    y_mat[:, nfc:nocc] = np.einsum("Pia,Ppa->pi", gamma, bmo[:, :, nocc:])
    y_mat[:, nocc:] = np.einsum("Pia,Pip->pa", gamma, bmo[:, nfc:nocc, :])
    if nfc:
        # Frozen core: P_Ij = P_jI = -Y_Ij / (2 (e_I - e_j)).
        p_ic = -y_mat[:nfc, nfc:nocc] / (2.0 * (eps[:nfc, None] - eps[None, nfc:nocc]))
        p_mat[:nfc, nfc:nocc] = p_ic
        p_mat[nfc:nocc, :nfc] = p_ic.T
    oo, vv = slice(0, nocc), slice(nocc, n)
    y_mat[oo, oo] += 2.0 * p_mat[oo, oo] * eps[:nocc, None]
    y_mat[vv, vv] += 2.0 * p_mat[vv, vv] * eps[nocc:, None]

    p_ao = c[:, oo] @ p_mat[oo, oo] @ c[:, oo].T + cv @ p_mat[vv, vv] @ cv.T
    lag = (
        y_mat[nocc:, :nocc] - y_mat[:nocc, nocc:].T + 4.0 * (cv.T @ g_op(lt, p_ao) @ co)
    )

    # Z-vector: (A U)_ai = e_ai U_ai + (C_v^T G[2 (C_v U C_o^T + h.c.)] C_o)_ai.
    amat = np.zeros((nv * nocc, nv * nocc))
    for k in range(nv * nocc):
        u = np.zeros(nv * nocc)
        u[k] = 1.0
        um = u.reshape(nv, nocc)
        dm = 2 * (cv @ um @ co.T)
        dm = dm + dm.T
        au = (eps[nocc:, None] - eps[None, :nocc]) * um + cv.T @ g_op(lt, dm) @ co
        amat[:, k] = au.ravel()
    check_sym = np.max(np.abs(amat - amat.T))
    z = np.linalg.solve(amat, lag.ravel()).reshape(nv, nocc)

    zao = cv @ z @ co.T
    d2 = p_ao - 0.5 * (zao + zao.T)
    w_mat = np.zeros((n, n))
    w_mat[oo, oo] = -0.5 * y_mat[oo, oo] - 2.0 * co.T @ g_op(lt, d2) @ co
    w_mat[vv, vv] = -0.5 * y_mat[vv, vv]
    w_mat[:nocc, nocc:] = -y_mat[:nocc, nocc:]
    w_mat[nocc:, :nocc] = z * eps[None, :nocc]
    s_mo = c.T @ sx @ c
    gamma_term = np.einsum("Pia,mi,Pmn,na->", gamma, c[:, nfc:nocc], lx, cv)
    de_c = np.sum(d2 * (hx + g_deriv(lt, lx, d_hf))) + np.sum(w_mat * s_mo) + gamma_term
    return de_hf + de_c, check_sym


def gradient_checks():
    for nfc, name in [(0, "gradient-vs-fd"), (1, "gradient-vs-fd-frozen-core")]:
        toy = Toy(11)
        x0 = 0.3
        ana, asym = analytic_gradient(toy, x0, nfc)
        h = 2e-3

        def cd(hh):
            return (
                total_energy(toy, x0 + hh, nfc) - total_energy(toy, x0 - hh, nfc)
            ) / (2 * hh)

        fd = (4 * cd(h) - cd(2 * h)) / 3  # Richardson, O(h^4)
        check(
            name,
            abs(ana - fd) < 1e-9,
            f"analytic {ana:.12f} vs FD {fd:.12f} (|d| = {abs(ana - fd):.1e}, A asym {asym:.1e})",
        )


symbolic()
first_variation()
gradient_checks()

if FAILED:
    print(f"\n{len(FAILED)} check(s) failed: {FAILED}")
    sys.exit(1)
print("\nall RI-dRPA gradient identities hold")
