#!/usr/bin/env python3
"""Proofs of the RI-MP2 static polarizability identities ferric-mp2 codes.

Theory: crates/ferric-mp2/theory.md section 10, "Static polarizability".
Code: crates/ferric-mp2/src/cpks_polar.rs (static_relaxed_density_ao,
first_order_fock_response, reference_at_field, relaxed_alpha_tensors,
mp2_polarizability_analytic_hf, build_apb_amb, dynamic_cphf_alpha_iw,
frozen_mp2_c6_molecular) and crates/ferric-mp2/src/zvector.rs
(solve_zvector, build_lagrangian, build_imat_ri, compute_az_product).

The model. A closed-shell toy system (6 basis functions, 2 occupied): a
non-orthogonal overlap S, a core Hamiltonian h, four-centre ERIs
(mn|ls) = sum_K L^K_mn L^K_ls for the reference SCF's J/K (the code's
`bounds.op` route), and an independent RI factorization for MP2,
(P|mn) and metric V, dressed with M = L_V^-1 (Cholesky, as
`metric_inverse_sqrt` does for Coulomb). Two dipole matrices R_x, R_y. A
uniform field F enters the core Hamiltonian as h + F R (electron charge -1),
so mu = -dE/dF and alpha = -d2E/dF2. The RHF is converged to a 1e-13
commutator at every field. Nuclear terms are omitted (they are constant in F).

Checks (each label carries "(claim Pn)"):
  P1  static_relaxed_density_ao's right-hand side L + (2J - K)[P + P^T]_vo,
      with L the four-term build_lagrangian loop, equals solve_zvector's
      (2J - K)[D_corr]_vo + I_ia - I_ai from build_imat_ri, and the two z
      agree (both solve (Delta + 1/2 compute_az_product) z = -X).
  P2  mu = -Tr[P_rel R] equals -dE/dF of E_RHF + E_MP2 by a Richardson
      difference with the RHF re-solved at each field.
  P3  The coded analytic alpha (CPHF U, dF = R + J[dD] - 1/2 K[dD], the
      canonical orbitals of F +- h dF as the field reference, central
      difference of P_rel at h = 1e-4, and its h -> 0 Richardson limit) equals
      -d2E/dF_x dF_y by a finite-field energy Hessian with the RHF re-solved,
      for xx, yy and xy; the h = 1e-4 value is off by its own O(h^2) only.
  P4  The HF-level pieces: 1/2 compute_az_product + Delta equals the
      build_apb_amb (A+B) matrix; the SCF-density difference along the field
      reference equals the CPHF alpha = 4 mu^T (A+B)^-1 mu
      (mp2_polarizability_analytic_hf's -4 mu.U); and dynamic_cphf_alpha_iw's
      4 mu^T [(A-B)(A+B) + u^2]^-1 (A-B) mu equals an independent complex
      TDHF linear-response solve at frequency iu (u = 0, 0.3, 1.7).
  P5  frozen_mp2_c6_molecular: the C6 of the rescaled HF shape,
      (3/pi) sum_k w_k [s alpha_HF(iu_k)]^2 with s = alpha_MP2(0)/alpha_HF(0),
      equals the closed-form pole sum
      (3/2) s^2 sum_mn f_m f_n / (W_m W_n (W_m + W_n)) of the TDHF excitations.
  P0  Trivial-limit anchor: with the MP2 RI factors switched off (u = 0, so
      t = 0), the relaxed density is the SCF density and the coded MP2 alpha
      equals the HF alpha exactly (to the CG-free dense-solve rounding).

Mutations (each turns the script red; exact one-occurrence substitutions):
  build_lagrangian term 3 with its sign flipped                   (P1 fails)
  static_relaxed_density_ao drops the (2J - K)[P + P^T] term      (P2 fails)
  first_order_fock_response with 2 J[dD]                          (P3 fails)
  dynamic system matrix (A-B)(A+B) + u instead of + u^2           (P4 fails)
  Casimir-Polder prefactor 3/(2 pi)                               (P5 fails)
Exits nonzero on any failed check; prints "FAIL Pn ...".

Run: python3 scripts/mp2_polarizability_sympy.py  (needs sympy, numpy, scipy)
"""

import sys

import numpy as np
import sympy as sp
from scipy.linalg import cholesky, eigh, solve_triangular

failures = []
RNG = np.random.default_rng(20261115)


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        failures.append(label)


# ================================================================ model
NBF, NOCC, NAUX, NL = 6, 2, 9, 7
NVIR = NBF - NOCC


def rsym(n, scale):
    m = RNG.standard_normal((n, n))
    return scale * (m + m.T)


S = np.eye(NBF) + rsym(NBF, 0.03)
H0 = np.diag(np.linspace(-4.0, 1.5, NBF)) + rsym(NBF, 0.15)
LF = np.array([rsym(NBF, 0.12) for _ in range(NL)])
ERI = np.einsum("Kmn,Kls->mnls", LF, LF)
U3 = np.array([rsym(NBF, 0.12) for _ in range(NAUX)])
_A = RNG.standard_normal((NAUX, NAUX))
VMET = _A @ _A.T / NAUX + np.eye(NAUX)
RDIP = [rsym(NBF, 0.4), rsym(NBF, 0.4)]


def jk(d):
    return np.einsum("mnls,ls->mn", ERI, d), np.einsum("mlns,ls->mn", ERI, d)


def rhf(h, tol=1e-13, max_iter=400):
    """DIIS RHF; returns (energy, C, eps, F) at the converged density."""
    c = eigh(h, S)[1]
    focks, errs = [], []
    for _ in range(max_iter):
        d = 2 * c[:, :NOCC] @ c[:, :NOCC].T
        jm, km = jk(d)
        f = h + jm - 0.5 * km
        err = (f @ d @ S - S @ d @ f).ravel()
        if np.max(np.abs(err)) < tol:
            eps, c = eigh(f, S)
            d = 2 * c[:, :NOCC] @ c[:, :NOCC].T
            jm, km = jk(d)
            f = h + jm - 0.5 * km
            return 0.5 * np.sum(d * (h + f)), c, eps, f
        focks.append(f)
        errs.append(err)
        focks, errs = focks[-8:], errs[-8:]
        nd = len(errs)
        bm = -np.ones((nd + 1, nd + 1))
        bm[nd, nd] = 0
        bm[:nd, :nd] = np.array([[e1 @ e2 for e2 in errs] for e1 in errs])
        rhs = np.zeros(nd + 1)
        rhs[nd] = -1
        coef = np.linalg.lstsq(bm, rhs, rcond=None)[0][:nd]
        c = eigh(sum(cf * fk for cf, fk in zip(coef, focks)), S)[1]
    raise AssertionError("SCF did not converge")


# ================================================================ transcriptions
def b_full_mo(c, u3):
    """compute_b_full_mo: B^P_pq = sum_Q (L_V^-1)_PQ (Q|pq), Cholesky dressing."""
    lv = cholesky(VMET, lower=True)
    return solve_triangular(
        lv, np.einsum("Qmn,mp,nq->Qpq", u3, c, c).reshape(NAUX, -1), lower=True
    ).reshape(NAUX, NBF, NBF)


def mp2_inter(c, eps, u3):
    """compute_mp2_intermediates + build_mp2_density: t = (ia|jb)/D, D < 0."""
    b = b_full_mo(c, u3)
    bov = b[:, :NOCC, NOCC:]
    g = np.einsum("Pia,Pjb->ijab", bov, bov)
    eo, ev = eps[:NOCC], eps[NOCC:]
    dneg = (
        eo[:, None, None, None]
        + eo[None, :, None, None]
        - ev[None, None, :, None]
        - ev[None, None, None, :]
    )
    t = g / dneg
    tt = 2 * t - t.transpose(0, 1, 3, 2)
    e2 = np.sum(g * tt)
    p_oo = -np.einsum("ikab,jkab->ij", t, tt)
    p_vv = np.einsum("ijac,ijbc->ab", t, tt)
    return e2, t, tt, p_oo, p_vv, b


def build_lagrangian(f_mo, t, p_oo, p_vv, b):
    """zvector.rs build_lagrangian: P*F terms plus the four-term integral loop."""
    eri = np.einsum("Ppq,Prs->pqrs", b, b)
    o, v = slice(0, NOCC), slice(NOCC, NBF)
    lag = np.einsum("ij,aj->ai", p_oo, f_mo[v, o]) + np.einsum(
        "ab,bi->ai", p_vv, f_mo[v, o]
    )
    grad = np.einsum("kjab,cajb->ck", t, 2 * eri[v, v, o, v]) - np.einsum(
        "kjab,cbja->ck", t, eri[v, v, o, v]
    )
    grad += np.einsum("ikab,iacb->ck", t, 2 * eri[o, v, v, v]) - np.einsum(
        "ikab,ibca->ck", t, eri[o, v, v, v]
    )
    grad -= np.einsum("ijcb,ikjb->ck", t, 2 * eri[o, o, o, v]) - np.einsum(
        "ijcb,ibjk->ck", t, eri[o, v, o, o]
    )
    grad -= np.einsum("ijac,iajk->ck", t, 2 * eri[o, v, o, o]) - np.einsum(
        "ijac,ikja->ck", t, eri[o, o, o, v]
    )
    return lag + grad


def imat_ri(tt, b):
    """zvector.rs build_imat_ri (no frozen core)."""
    x = np.einsum("ijab,Pjb->Pia", tt, b[:, :NOCC, NOCC:])
    imat = np.zeros((NBF, NBF))
    imat[:, :NOCC] = -2 * np.einsum("Pia,Pqa->qi", x, b[:, :, NOCC:])
    imat[:, NOCC:] = -2 * np.einsum("Pia,Pqi->qa", x, b[:, :, :NOCC])
    return imat


def veff_vo(c, dm_mo):
    jm, km = jk(c @ dm_mo @ c.T)
    return (c.T @ (2 * jm - km) @ c)[NOCC:, :NOCC]


def hessian_apply(c, eps, z):
    """(Delta + 1/2 compute_az_product) z: the solve_zvector / solve_cphf_cg operator."""
    co, cv = c[:, :NOCC], c[:, NOCC:]
    dz = cv @ z @ co.T + co @ z.T @ cv.T
    jz, kz = jk(dz)
    az = cv.T @ (4 * jz - kz - kz.T) @ co
    return 0.5 * az + (eps[NOCC:, None] - eps[None, :NOCC]) * z


def hessian_matrix(c, eps):
    n = NVIR * NOCC
    return np.array(
        [hessian_apply(c, eps, e.reshape(NVIR, NOCC)).ravel() for e in np.eye(n)]
    ).T


def solve_cphf(c, eps, rhs):
    return np.linalg.solve(hessian_matrix(c, eps), rhs.ravel()).reshape(NVIR, NOCC)


def static_relaxed_density_ao(c, eps, fock, u3):
    """cpks_polar.rs static_relaxed_density_ao: D = 2 delta_oo + (P + P^T) + z."""
    _, t, _, p_oo, p_vv, b = mp2_inter(c, eps, u3)
    dm_p = np.zeros((NBF, NBF))
    dm_p[:NOCC, :NOCC] = p_oo + p_oo.T
    dm_p[NOCC:, NOCC:] = p_vv + p_vv.T
    lag = build_lagrangian(c.T @ fock @ c, t, p_oo, p_vv, b)
    xvo = lag + veff_vo(c, dm_p)
    z = solve_cphf(c, eps, -xvo)
    d_mo = dm_p.copy()
    d_mo[:NOCC, :NOCC] += 2 * np.eye(NOCC)
    d_mo[NOCC:, :NOCC] += z
    d_mo[:NOCC, NOCC:] += z.T
    return c @ d_mo @ c.T, z


def zvector_z(c, eps, u3):
    """zvector.rs solve_zvector: X = (2J - K)[D_corr]_vo + I_ia - I_ai."""
    _, _, tt, p_oo, p_vv, b = mp2_inter(c, eps, u3)
    imat = imat_ri(tt, b)
    dm1 = np.zeros((NBF, NBF))
    dm1[:NOCC, :NOCC] = p_oo + p_oo.T
    dm1[NOCC:, NOCC:] = p_vv + p_vv.T
    xvo = veff_vo(c, dm1) + imat[:NOCC, NOCC:].T - imat[NOCC:, :NOCC]
    return solve_cphf(c, eps, -xvo), xvo


def first_order_fock_response(c, eps, r):
    """dF = r + J[dD] - 1/2 K[dD], dD = 2 (C_v U C_o^T + h.c.), (A+B) U = -r_vo."""
    co, cv = c[:, :NOCC], c[:, NOCC:]
    u = solve_cphf(c, eps, -(cv.T @ r @ co))
    half = cv @ u @ co.T
    jd, kd = jk(2 * (half + half.T))
    return r + jd - 0.5 * kd


def reference_at_field(c, fock, dfock, s):
    """Canonical orbitals of F + s dF in the unperturbed MO basis."""
    f_s = fock + s * dfock
    f_mo = c.T @ f_s @ c
    w, v = np.linalg.eigh(0.5 * (f_mo + f_mo.T))
    return c @ v, w, f_s


def relaxed_alpha_tensors(c, eps, fock, u3, h=1e-4):
    """(alpha_HF, alpha_MP2)[x][y] = -Tr[dD/dF_x r_y], cpks_polar.rs."""
    a_hf, a_mp2 = np.zeros((2, 2)), np.zeros((2, 2))
    for x in range(2):
        df = first_order_fock_response(c, eps, RDIP[x])
        d_hf, d_rel = [], []
        for s in (h, -h):
            cs, ws, fs = reference_at_field(c, fock, df, s)
            d_rel.append(static_relaxed_density_ao(cs, ws, fs, u3)[0])
            d_hf.append(2 * cs[:, :NOCC] @ cs[:, :NOCC].T)
        dd_hf = (d_hf[0] - d_hf[1]) / (2 * h)
        dd_rel = (d_rel[0] - d_rel[1]) / (2 * h)
        for y in range(2):
            a_hf[x, y] = -np.sum(dd_hf * RDIP[y])
            a_mp2[x, y] = -np.sum(dd_rel * RDIP[y])
    return a_hf, a_mp2


def apb_amb(c, eps):
    """build_apb_amb from the MO ERIs of the reference operator."""
    mo = np.einsum("mnls,mp,nq,lr,st->pqrt", ERI, c, c, c, c)
    o, v = slice(0, NOCC), slice(NOCC, NBF)
    coul = mo[v, o, v, o]  # (ai|bj)
    abij = mo[v, v, o, o].transpose(0, 2, 1, 3)  # (ab|ij) at [a,i,b,j]
    ajbi = mo[v, o, v, o].transpose(0, 3, 2, 1)  # (aj|bi) at [a,i,b,j]
    a_el = 2 * coul - abij
    b_el = 2 * coul - ajbi
    n = NVIR * NOCC
    de = np.diag((eps[NOCC:, None] - eps[None, :NOCC]).ravel())
    return (a_el + b_el).reshape(n, n) + de, (a_el - b_el).reshape(n, n) + de


def dynamic_cphf_alpha_iw(apb, amb, mus, omega):
    """4 mu_x^T [(A-B)(A+B) + omega^2]^-1 (A-B) mu_y (solve S t = (A-B) mu)."""
    n = apb.shape[0]
    sysm = amb @ apb + omega**2 * np.eye(n)
    return np.array(
        [[4 * mx @ np.linalg.solve(sysm, amb @ my) for my in mus] for mx in mus]
    )


def casimir_polder(weights, iso):
    return 3.0 / np.pi * np.sum(weights * iso * iso)


def gl_rpa_grid(n, u0=0.5):
    """ferric-rpa quadrature: Gauss-Legendre mapped by u = u0 (1+x)/(1-x)."""
    x, w = np.polynomial.legendre.leggauss(n)
    return u0 * (1 + x) / (1 - x), w * 2 * u0 / (1 - x) ** 2


# ================================================================ helpers
def energy(fx, fy, u3=U3):
    e_hf, c, eps, _ = rhf(H0 + fx * RDIP[0] + fy * RDIP[1])
    return e_hf + mp2_inter(c, eps, u3)[0]


def richardson1(f, h):
    d1 = (f(h) - f(-h)) / (2 * h)
    d2 = (f(2 * h) - f(-2 * h)) / (4 * h)
    return (4 * d1 - d2) / 3


def second_derivs(h):
    """-d2E/dFx dFy by finite field: 5-point diagonals, Richardson mixed term."""
    e0 = energy(0.0, 0.0)

    def diag(axis):
        def e(s):
            return energy(s, 0.0) if axis == 0 else energy(0.0, s)

        return (-e(2 * h) + 16 * e(h) - 30 * e0 + 16 * e(-h) - e(-2 * h)) / (12 * h * h)

    def mixed(k):
        return (energy(k, k) - energy(k, -k) - energy(-k, k) + energy(-k, -k)) / (
            4 * k * k
        )

    dxy = (4 * mixed(h) - mixed(2 * h)) / 3
    return -np.array([[diag(0), dxy], [dxy, diag(1)]])


# ================================================================ claims
def claim_p1(c, eps, fock):
    _, t, tt, p_oo, p_vv, b = mp2_inter(c, eps, U3)
    dm_p = np.zeros((NBF, NBF))
    dm_p[:NOCC, :NOCC] = p_oo + p_oo.T
    dm_p[NOCC:, NOCC:] = p_vv + p_vv.T
    x_static = build_lagrangian(c.T @ fock @ c, t, p_oo, p_vv, b) + veff_vo(c, dm_p)
    z_zv, x_zv = zvector_z(c, eps, U3)
    _, z_st = static_relaxed_density_ao(c, eps, fock, U3)
    ex = np.max(np.abs(x_static - x_zv)) / np.max(np.abs(x_zv))
    ez = np.max(np.abs(z_st - z_zv)) / np.max(np.abs(z_zv))
    check(
        "P1 static_relaxed_density_ao z = solve_zvector z (claim P1)",
        ex < 1e-12 and ez < 1e-12,
        f"rhs {ex:.1e} z {ez:.1e} (bar 1e-12)",
    )


def claim_p2(c, eps, fock):
    d_rel, _ = static_relaxed_density_ao(c, eps, fock, U3)
    worst = 0.0
    for x in range(2):
        mu = -np.sum(d_rel * RDIP[x])
        fd = -richardson1(lambda s: energy(s, 0.0) if x == 0 else energy(0.0, s), 1e-3)
        worst = max(worst, abs(mu - fd) / max(1.0, abs(fd)))
    # Richardson truncation O(h^4) ~ 1e-12; SCF/MP2 noise ~ 1e-13 / h = 1e-10.
    check(
        "P2 mu = -Tr[P_rel R] = -dE/dF (claim P2)",
        worst < 1e-8,
        f"{worst:.1e} (bar 1e-8)",
    )


def claim_p3(c, eps, fock):
    def coded(h):
        a = relaxed_alpha_tensors(c, eps, fock, U3, h)[1]
        return 0.5 * (a + a.T)  # analytic_alpha_relaxed symmetrizes

    a_h, a_2h = coded(1e-4), coded(2e-4)
    a_ext = (4 * a_h - a_2h) / 3  # removes the coded O(h^2) truncation
    ref = second_derivs(1e-3)
    scale = np.max(np.abs(ref))
    err_ext = np.max(np.abs(a_ext - ref)) / scale
    err_h = np.max(np.abs(a_h - ref)) / scale
    trunc = np.max(np.abs(a_h - a_2h)) / 3 / scale
    # Reference Hessian at h = 1e-3: energy noise ~1e-13 / h^2 ~ 1e-7 absolute
    # (~1e-8 relative) plus O(h^4) truncation; the coded h = 1e-4 value also
    # carries its own O(h^2) truncation, estimated by (a(h) - a(2h)) / 3.
    check(
        "P3 coded analytic alpha = -d2E/dF dF by finite field (claim P3)",
        err_ext < 1e-7 and err_h < 1.5 * trunc + 1e-7,
        f"extrapolated {err_ext:.1e} (bar 1e-7) coded h=1e-4 {err_h:.1e} "
        f"(its O(h^2) estimate {trunc:.1e}) alpha_MP2 = {np.round(a_h, 6).tolist()}",
    )


def claim_p4(c, eps, fock):
    apb, amb = apb_amb(c, eps)
    e_op = np.max(np.abs(hessian_matrix(c, eps) - apb))
    co, cv = c[:, :NOCC], c[:, NOCC:]
    mus = [(cv.T @ r @ co).ravel() for r in RDIP]
    cphf = np.array([[4 * mx @ np.linalg.solve(apb, my) for my in mus] for mx in mus])
    # mp2_polarizability_analytic_hf: U^x = -(A+B)^-1 mu^x, alpha_xy = -4 mu^y . U^x
    contract = np.array(
        [[-4 * my @ (-np.linalg.solve(apb, mx)) for my in mus] for mx in mus]
    )
    a_hf = relaxed_alpha_tensors(c, eps, fock, U3, 1e-4)[0]
    a_hf2 = relaxed_alpha_tensors(c, eps, fock, U3, 2e-4)[0]
    scale = np.max(np.abs(cphf))
    # The field-reference difference is O(h^2); its Richardson limit is exact.
    e_hf = np.max(np.abs((4 * a_hf - a_hf2) / 3 - cphf)) / scale
    e_hf_h = np.max(np.abs(a_hf - cphf)) / scale
    trunc = np.max(np.abs(a_hf - a_hf2)) / 3 / scale
    e_c = np.max(np.abs(contract - cphf))
    worst_dyn = 0.0
    for u in (0.0, 0.3, 1.7):
        code = dynamic_cphf_alpha_iw(apb, amb, mus, u)
        a_mat = 0.5 * (apb + amb)
        b_mat = 0.5 * (apb - amb)
        n = apb.shape[0]
        big = np.block([[a_mat, b_mat], [b_mat, a_mat]]) - 1j * u * np.block(
            [[np.eye(n), np.zeros((n, n))], [np.zeros((n, n)), -np.eye(n)]]
        )
        resp = np.zeros((2, 2), dtype=complex)
        for y, my in enumerate(mus):
            xy = np.linalg.solve(big, -np.concatenate([my, my]))
            for x, mx in enumerate(mus):
                resp[x, y] = -2 * mx @ (xy[:n] + xy[n:])
        worst_dyn = max(
            worst_dyn, np.max(np.abs(code - resp.real)), np.max(np.abs(resp.imag))
        )
    # Static limit alpha(i0) = static CPHF alpha.
    e0 = np.max(np.abs(dynamic_cphf_alpha_iw(apb, amb, mus, 0.0) - cphf))
    check(
        "P4 (A+B) operator, HF anchor and dynamic TDHF alpha(iu) (claim P4)",
        e_op < 1e-12
        and e_hf < 1e-9
        and e_hf_h < 1.5 * trunc + 1e-12
        and e_c < 1e-12
        and worst_dyn < 1e-11
        and e0 < 1e-11,
        f"operator {e_op:.1e} field-ref HF limit {e_hf:.1e} (bar 1e-9) at h=1e-4 "
        f"{e_hf_h:.1e} (O(h^2) estimate {trunc:.1e}) "
        f"contract {e_c:.1e} dynamic {worst_dyn:.1e} static {e0:.1e}",
    )
    return apb, amb, mus


def claim_p5(c, eps, fock, apb, amb, mus):
    _, a_mp2 = relaxed_alpha_tensors(c, eps, fock, U3)
    a_hf0 = dynamic_cphf_alpha_iw(apb, amb, mus, 0.0)
    # Isotropic average over the two model axes.
    scale = np.trace(a_mp2) / np.trace(a_hf0)
    freqs, weights = gl_rpa_grid(60)
    iso = np.array(
        [scale * np.trace(dynamic_cphf_alpha_iw(apb, amb, mus, u)) / 2 for u in freqs]
    )
    c6 = casimir_polder(weights, iso)
    # Pole form: Omega^2 eigen of (A-B)^1/2 (A+B) (A-B)^1/2.
    wm, vm = np.linalg.eigh(amb)
    amh = vm @ np.diag(np.sqrt(wm)) @ vm.T
    om2, zv = np.linalg.eigh(amh @ apb @ amh)
    om = np.sqrt(om2)
    f = np.zeros_like(om)
    for m in mus:
        f += 4 * (zv.T @ (amh @ m)) ** 2 / 2  # isotropic over 2 axes
    exact = (
        1.5
        * scale**2
        * np.sum(
            f[:, None]
            * f[None, :]
            / (om[:, None] * om[None, :] * (om[:, None] + om[None, :]))
        )
    )
    err = abs(c6 - exact) / abs(exact)
    sym_u, sym_a, sym_b = sp.symbols("u a b", positive=True)
    closed = sp.integrate(
        1 / ((sym_u**2 + sym_a**2) * (sym_u**2 + sym_b**2)), (sym_u, 0, sp.oo)
    )
    sym_ok = sp.simplify(closed - sp.pi / (2 * sym_a * sym_b * (sym_a + sym_b))) == 0
    check(
        "P5 frozen-amplitude C6 = pole-sum closed form (claim P5)",
        err < 1e-9 and sym_ok,
        f"{err:.1e} (bar 1e-9, 60-point mapped Gauss-Legendre) symbolic {sym_ok}",
    )


def claim_p0():
    u_off = np.zeros_like(U3)
    _, c, eps, fock = rhf(H0)
    a_hf, a_mp2 = relaxed_alpha_tensors(c, eps, fock, u_off)
    e2 = mp2_inter(c, eps, u_off)[0]
    err = np.max(np.abs(a_mp2 - a_hf))
    check(
        "P0 anchor: no correlation -> alpha_MP2 = alpha_HF exactly (claim P0)",
        err < 1e-12 and e2 == 0.0,
        f"{err:.1e} (bar 1e-12)",
    )


def main():
    _, c, eps, fock = rhf(H0)
    claim_p0()
    claim_p1(c, eps, fock)
    claim_p2(c, eps, fock)
    claim_p3(c, eps, fock)
    apb, amb, mus = claim_p4(c, eps, fock)
    claim_p5(c, eps, fock, apb, amb, mus)
    if failures:
        print(f"{len(failures)} check(s) failed")
        sys.exit(1)
    print("all MP2 polarizability identities verified")


if __name__ == "__main__":
    main()
