#!/usr/bin/env python3
"""U-OO-RI-MP2 orbital gradient: the integral-response loop, transcribed.

Theory: `book/src/12-oo-laplace/02-unrestricted-oo-rimp2.md#unrestricted-oo-rimp2-orbital-gradient` ("Unrestricted OO-RI-MP2
orbital gradient") and its subsection "The integral-response loop". Code:
crates/ferric-mp2/src/u_rimp2.rs (u_mp2_orbital_gradient_kernel,
build_u_mp2_density / u_mp2_density_from_amplitudes),
crates/ferric-mp2/src/orbital_gradient_gemm.rs (GradientB::extract,
pair_combination, pair_gamma_rows, cross_gamma_rows, contract_vv_gamma,
pair_ooov_term, ordered_task_sum) and crates/ferric-mp2/src/u_oo_rimp2.rs
(u_gradient_solver_frame, add_hf_gradient, embed_u_density,
compute_uhf_energy, UOoJk::spin_jk on the four-centre route). The functions
below keep the Rust names and loop structure, one call per Rust call, so a
change to a coefficient in the Rust has one place to land here.

Model: a random orthonormal-AO system of NBF = 7 functions, n_alpha = 3,
n_beta = 2, an independent RI factor B^P_mn (NAUX = 12) for MP2 and a
four-centre ERI from a second factor for the UHF J/K. Each check prints
"FAIL [<name>]" when it fails; the script exits 1 if any did.

  1  pair-combination (sympy): for amplitudes antisymmetric in (i,j) and in
     (a,b), pair_combination(A_kj, A_jk, 1, 1) = A_kj - A_kj^T + A_jk^T - A_jk
     equals 4 A_kj, so pair_gamma_rows' alpha = 1/2 gives 2 sum_j B_j A_kj^T.
  2  chain-rule-factors (sympy): at fixed denominators,
     d/dl [1/4 sum t K](g + l dg) = sum t dg for the same-spin block
     (K = g - g^swap, t = K/D) and d/dl [sum (g + l dg)^2/D] = 2 sum t dg for
     the opposite-spin block: the 1/2 * 4 = 2 of the same-spin rows and the
     explicit 2.0 on the ab blocks.
  3  kernel-equals-derivation (numpy): the transcribed kernel equals the
     derivation of the book section above, sum t d(ia|jb) with
     d(ia|jb) = d_ik (ca|jb) + d_jk (ia|cb) - d_ac (ik|jb) - d_bc (ia|jk),
     block by block (same_a, ab_a, same_b, ab_b), to 1e-13.
  4  kernel-vs-fd (numpy, independent route): each block equals a Richardson
     difference of that block's energy (1/4 sum K^2/D or sum (ia|JB)^2/D,
     denominators fixed) under C^s -> C^s expm(kappa^s), at a random point.
  5  kernel-vs-fd-frozen-core: as 4 with one frozen core orbital per spin
     (GradientB::extract's first_occ offset; columns are the active occupieds).
  6  assembled-vs-fd: the full u_gradient_solver_frame (UHF term, kernel,
     Fock response with J/K of P^s = C W^s C^T, rotation part F W - W F,
     back-rotation U_v g U_o^T) equals a Richardson difference of
     E_UHF + E_MP2(spin-orbital, semicanonical) at non-stationary,
     non-semicanonical orbitals.
  6b assembled-vs-fd-frozen-core: as 6 with one frozen core orbital per
     spin. The core orbitals are not rotated and are left out of the MP2 sum;
     the kernel's active columns land at MO columns first_occ.. (the
     embed_active_columns step) before the UHF and Fock-response terms.
  7  closed-shell-anchor (exactness anchor): with C_alpha = C_beta and
     n_alpha = n_beta the kernel gives same_a = same_b and ab_a = ab_b
     exactly, and with B = 0 it returns zero.

Tolerances. Richardson (4-point, h = 1e-3) truncation is h^4/30 |E^(5)|,
about 1e-13 here (|E^(5)| < 10); round-off is about eps |E| / h, 2e-16 * 10
/ 1e-3 = 2e-12. The FD bar 1e-9 (relative to max(1, max|g|)) is 500x above
both and far below the smallest mutation effect (>= 1e-3). Checks 3 and 7
compare two orderings of the same floating-point sums: bar 1e-13.

Mutations (each turns the named check red):
  - same-spin Gamma alpha 0.5 -> 1.0                            [kernel-vs-fd]
  - the ab-block factor 2.0 on g_a -> 1.0                       [kernel-vs-fd]
  - the alpha (ik|jb) term alpha -0.5 -> +0.5                   [kernel-vs-fd]
  - the beta cross (ia|JK) term -1.0 -> +1.0                    [kernel-vs-fd]
  - GradientB::extract's occupied offset first_occ + i -> i     [kernel-vs-fd-frozen-core]
  - back-rotation U_o^T -> U_o                                   [assembled-vs-fd]
  - add_hf_gradient 2 F_ck -> 1 F_ck                             [assembled-vs-fd]
  - embed_active_columns at columns 0.. instead of first_occ..   [assembled-vs-fd-frozen-core]
  - pair_combination's last term - q B -> + q B                  [pair-combination]

Run: python3 scripts/u_oo_rimp2_response_sympy.py  (needs sympy, numpy, scipy)
"""

import itertools
import sys

import numpy as np
import sympy as sp
from scipy.linalg import expm

FAILED = []
RNG = np.random.default_rng(20261115)
NBF, NAUX, NL = 7, 12, 8
VV_CONTRACT_AUX_SLAB = 5  # the Rust constant is 32; a small slab exercises the slab sum


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} [{name}] {detail}")
    if not ok:
        FAILED.append(name)


def richardson(f, h):
    d1 = (f(h) - f(-h)) / (2 * h)
    d2 = (f(2 * h) - f(-2 * h)) / (4 * h)
    return (4 * d1 - d2) / 3


# ========================================== orbital_gradient_gemm.rs, transcribed
class GradientB:
    """GradientB::extract: bov[P, i*nvir+a], boo[P, i*nocc+k], bvv[c, P*nvir+x]."""

    def __init__(self, b_full, nocc, nvir, first_occ, nocc_total):
        naux = b_full.shape[0]
        self.bov = np.zeros((naux, nocc * nvir))
        self.boo = np.zeros((naux, nocc * nocc))
        self.bvv = np.zeros((nvir, naux * nvir))
        for p in range(naux):
            bp = b_full[p]
            for i in range(nocc):
                row = bp[first_occ + i]
                self.bov[p, i * nvir : (i + 1) * nvir] = row[
                    nocc_total : nocc_total + nvir
                ]
                self.boo[p, i * nocc : (i + 1) * nocc] = row[
                    first_occ : first_occ + nocc
                ]
            for c in range(nvir):
                self.bvv[c, p * nvir : (p + 1) * nvir] = bp[
                    nocc_total + c, nocc_total : nocc_total + nvir
                ]
        self.naux, self.nocc, self.nvir = naux, nocc, nvir

    def bov_occ(self, j):
        return self.bov[:, j * self.nvir : (j + 1) * self.nvir]


def pair_combination(a, b, p, q):
    """out = p A - q A^T + p B^T - q B."""
    return p * a - q * a.T + p * b.T - q * b


def pair_gamma_rows(block, p, q, alpha, b, gamma):
    """Gamma_k = alpha sum_j B_j M_kj^T, M_kj = pair_combination(A_kj, A_jk)."""
    nocc, nvir, naux = b.nocc, b.nvir, b.naux
    assert gamma.shape == (nocc, naux * nvir)
    for k in range(nocc):
        g = np.zeros((naux, nvir))
        for j in range(nocc):
            m = pair_combination(block(k, j), block(j, k), p, q)
            g += alpha * b.bov_occ(j) @ m.T
        gamma[k] = g.reshape(-1)


def cross_gamma_rows(block, nvir_out, b_in, gamma):
    """Gamma_k = sum_j B_j block(k, j)^T, B_j from the OTHER spin."""
    naux = b_in.naux
    assert gamma.shape[1] == naux * nvir_out
    for k in range(gamma.shape[0]):
        g = np.zeros((naux, nvir_out))
        for j in range(b_in.nocc):
            g += b_in.bov_occ(j) @ block(k, j).T
        gamma[k] = g.reshape(-1)


def ordered_task_sum(n_tasks, dim, task):
    out = np.zeros(dim)
    for n in range(n_tasks):
        acc = np.zeros(dim)
        task(n, acc)
        out += acc
    return out


def contract_vv_gamma(bvv, gamma, naux, nvir, aux_slab):
    """G[c, r] = sum_{P,x} bvv[c, P*nvir+x] gamma[r, P*nvir+x], in P slabs."""
    rows = gamma.shape[0]
    n_slabs = -(-naux // aux_slab)

    def task(n, acc):
        p0 = n * aux_slab
        p1 = min(p0 + aux_slab, naux)
        cols = slice(p0 * nvir, p1 * nvir)
        acc[...] = bvv[:, cols] @ gamma[:, cols].T

    return ordered_task_sum(n_slabs, (nvir, rows), task)


def pair_ooov_term(block, p, q, alpha, ooov, i, nocc, nvir, acc):
    """acc[c, k] += alpha sum_j sum_y N_ij[c, y] ooov[i*nocc+k, j*nvir+y]."""
    rows = slice(i * nocc, (i + 1) * nocc)
    for j in range(nocc):
        n = pair_combination(block(i, j), block(j, i), p, q)
        o = ooov[rows, j * nvir : (j + 1) * nvir]
        acc += alpha * n @ o.T


# ========================================== u_rimp2.rs, transcribed
def u_mp2_orbital_gradient_kernel(t_aa, t_bb, t_ab, ba, bb, aux_slab):
    oa, va, ob, vb, naux = ba.nocc, ba.nvir, bb.nocc, bb.nvir, ba.naux

    def blk_aa(i, j):
        return t_aa[i, j]

    def blk_bb(i, j):
        return t_bb[i, j]

    def ab(i, jj):
        return t_ab[i, jj]

    ooov_a = ba.boo.T @ ba.bov
    ooov_b = bb.boo.T @ bb.bov
    ooov_ab = ba.boo.T @ bb.bov
    ovoo_ab = ba.bov.T @ bb.boo

    gamma_a = np.zeros((2 * oa, naux * va))
    pair_gamma_rows(blk_aa, 1.0, 1.0, 0.5, ba, gamma_a[:oa])
    cross_gamma_rows(ab, va, bb, gamma_a[oa:])
    gamma_b = np.zeros((2 * ob, naux * vb))
    pair_gamma_rows(blk_bb, 1.0, 1.0, 0.5, bb, gamma_b[:ob])

    def ab_t(k, i):
        return ab(i, k).T

    cross_gamma_rows(ab_t, vb, ba, gamma_b[ob:])

    g_a = contract_vv_gamma(ba.bvv, gamma_a, naux, va, aux_slab)
    g_b = contract_vv_gamma(bb.bvv, gamma_b, naux, vb, aux_slab)

    def task_a(i, acc):
        same, cross = acc[:, :oa], acc[:, oa:]
        pair_ooov_term(blk_aa, 1.0, 1.0, -0.5, ooov_a, i, oa, va, same)
        for jj in range(ob):
            o = ooov_ab[i * oa : (i + 1) * oa, jj * vb : (jj + 1) * vb]
            cross += -1.0 * ab(i, jj) @ o.T

    g_a = g_a + ordered_task_sum(oa, (va, 2 * oa), task_a)

    def task_b(n, acc):
        same, cross = acc[:, :ob], acc[:, ob:]
        if n < ob:
            pair_ooov_term(blk_bb, 1.0, 1.0, -0.5, ooov_b, n, ob, vb, same)
        else:
            i = n - ob
            for jj in range(ob):
                o = ovoo_ab[i * va : (i + 1) * va, jj * ob : (jj + 1) * ob]
                cross += -1.0 * ab(i, jj).T @ o

    g_b = g_b + ordered_task_sum(ob + oa, (vb, 2 * ob), task_b)
    return {
        "same_a": g_a[:, :oa],
        "ab_a": 2.0 * g_a[:, oa:],
        "same_b": g_b[:, :ob],
        "ab_b": 2.0 * g_b[:, ob:],
    }


def u_mp2_density_from_amplitudes(t_aa, t_bb, t_ab):
    """build_u_mp2_density: (P_oo^a, P_vv^a, P_oo^b, P_vv^b)."""
    oa, _, va, _ = t_aa.shape
    ob, _, vb, _ = t_bb.shape
    x_aa, x_ab, x_bb = t_aa.reshape(oa, -1), t_ab.reshape(oa, -1), t_bb.reshape(ob, -1)
    p_oo_a = -0.5 * x_aa @ x_aa.T - 1.0 * x_ab @ x_ab.T
    p_oo_b = -0.5 * x_bb @ x_bb.T
    for k in range(oa):
        y = t_ab[k].reshape(ob, va * vb)
        p_oo_b += -1.0 * y @ y.T
    p_vv_a = np.zeros((va, va))
    for i in range(oa):
        for j in range(oa):
            p_vv_a += 0.5 * t_aa[i, j] @ t_aa[i, j].T
        for jj in range(ob):
            p_vv_a += 1.0 * t_ab[i, jj] @ t_ab[i, jj].T
    p_vv_b = np.zeros((vb, vb))
    for n in range(ob):
        for j in range(ob):
            p_vv_b += 0.5 * t_bb[n, j] @ t_bb[n, j].T
    for i in range(oa):
        z = t_ab[i].reshape(ob * va, vb)
        p_vv_b += 1.0 * z.T @ z
    return p_oo_a, p_vv_a, p_oo_b, p_vv_b


# ========================================== amplitudes (compute_u_mp2_amplitudes)
def b_full_mo(bfac, c):
    return np.einsum("Pmn,mp,nq->Ppq", bfac, c, c)


def amplitudes(bfa, bfb, ea, eb, na, nb, fc):
    """t_aa = [(ia|jb) - (ib|ja)]/D, t_ab = (ia|JB)/D, D = e_i + e_j - e_a - e_b."""
    oa_, va_ = slice(fc, na), slice(na, None)
    ob_, vb_ = slice(fc, nb), slice(nb, None)
    g_aa = np.einsum("Pia,Pjb->ijab", bfa[:, oa_, va_], bfa[:, oa_, va_])
    g_bb = np.einsum("Pia,Pjb->ijab", bfb[:, ob_, vb_], bfb[:, ob_, vb_])
    g_ab = np.einsum("Pia,Pjb->ijab", bfa[:, oa_, va_], bfb[:, ob_, vb_])

    def den(e1o, e2o, e1v, e2v):
        return (
            e1o[:, None, None, None]
            + e2o[None, :, None, None]
            - e1v[None, None, :, None]
            - e2v[None, None, None, :]
        )

    d_aa = den(ea[oa_], ea[oa_], ea[va_], ea[va_])
    d_bb = den(eb[ob_], eb[ob_], eb[vb_], eb[vb_])
    d_ab = den(ea[oa_], eb[ob_], ea[va_], eb[vb_])
    t_aa = (g_aa - g_aa.transpose(0, 1, 3, 2)) / d_aa
    t_bb = (g_bb - g_bb.transpose(0, 1, 3, 2)) / d_bb
    t_ab = g_ab / d_ab
    return t_aa, t_bb, t_ab


def kernel_at(bfac, ca, cb, ea, eb, na, nb, fc, aux_slab=VV_CONTRACT_AUX_SLAB):
    bfa, bfb = b_full_mo(bfac, ca), b_full_mo(bfac, cb)
    t_aa, t_bb, t_ab = amplitudes(bfa, bfb, ea, eb, na, nb, fc)
    ba = GradientB(bfa, na - fc, NBF - na, fc, na)
    bb = GradientB(bfb, nb - fc, NBF - nb, fc, nb)
    return u_mp2_orbital_gradient_kernel(t_aa, t_bb, t_ab, ba, bb, aux_slab)


# ========================================== u_oo_rimp2.rs, transcribed
def spin_jk(eri, d_a, d_b):
    """UOoJk::spin_jk, four-centre route: (J[d_a + d_b], K[d_a], K[d_b])."""
    j_tot = np.einsum("mnls,ls->mn", eri, d_a + d_b)
    k_a = np.einsum("mlns,ls->mn", eri, d_a)
    k_b = np.einsum("mlns,ls->mn", eri, d_b)
    return j_tot, k_a, k_b


def compute_uhf_energy(eri, ca, cb, na, nb, h):
    d_a, d_b = ca[:, :na] @ ca[:, :na].T, cb[:, :nb] @ cb[:, :nb].T
    j_tot, k_a, k_b = spin_jk(eri, d_a, d_b)
    f_a, f_b = h + j_tot - k_a, h + j_tot - k_b
    e = 0.5 * np.sum((h + f_a) * d_a) + 0.5 * np.sum((h + f_b) * d_b)
    return e, f_a, f_b


def semicanonical_rotation(f_mo, first_occ, nocc_total):
    n = f_mo.shape[0]
    u = np.eye(n)
    eps = np.diag(f_mo).copy()
    for lo, hi in ((first_occ, nocc_total), (nocc_total, n)):
        w, v = np.linalg.eigh(f_mo[lo:hi, lo:hi])
        u[lo:hi, lo:hi], eps[lo:hi] = v, w
    return u, eps


def add_hf_gradient(g_mp2, f_mo, nocc):
    nvir, nocc_check = g_mp2.shape
    assert nocc == nocc_check
    g = g_mp2.copy()
    for a in range(nvir):
        for i in range(nocc):
            g[a, i] += 2.0 * f_mo[nocc + a, i]
    return g


def embed_active_columns(g_act, first_occ, nocc_total):
    nvir, nocc = g_act.shape
    assert first_occ + nocc == nocc_total
    g = np.zeros((nvir, nocc_total))
    g[:, first_occ:nocc_total] = g_act
    return g


def embed_u_density(p_oo, p_vv, first_occ, nocc_total, nmo):
    w = np.zeros((nmo, nmo))
    no, nv = p_oo.shape[0], p_vv.shape[0]
    w[first_occ : first_occ + no, first_occ : first_occ + no] = 0.5 * (p_oo + p_oo.T)
    w[nocc_total : nocc_total + nv, nocc_total : nocc_total + nv] = 0.5 * (
        p_vv + p_vv.T
    )
    return w


def u_gradient_solver_frame(h, eri, bfac, ca, cb, na, nb, fc=0):
    """Full gradient (g_a, g_b), shape (nvir, nocc - fc): active columns only."""
    _, f_a, f_b = compute_uhf_energy(eri, ca, cb, na, nb, h)
    u_a, eps_a = semicanonical_rotation(ca.T @ f_a @ ca, fc, na)
    u_b, eps_b = semicanonical_rotation(cb.T @ f_b @ cb, fc, nb)
    c_a_sc, c_b_sc = ca @ u_a, cb @ u_b
    bfa, bfb = b_full_mo(bfac, c_a_sc), b_full_mo(bfac, c_b_sc)
    t_aa, t_bb, t_ab = amplitudes(bfa, bfb, eps_a, eps_b, na, nb, fc)
    ba = GradientB(bfa, na - fc, NBF - na, fc, na)
    bb = GradientB(bfb, nb - fc, NBF - nb, fc, nb)
    bl = u_mp2_orbital_gradient_kernel(t_aa, t_bb, t_ab, ba, bb, VV_CONTRACT_AUX_SLAB)
    g_mp2_a, g_mp2_b = bl["same_a"] + bl["ab_a"], bl["same_b"] + bl["ab_b"]
    f_mo_a = c_a_sc.T @ f_a @ c_a_sc
    f_mo_b = c_b_sc.T @ f_b @ c_b_sc
    g_mp2_a = embed_active_columns(g_mp2_a, fc, na)
    g_a = add_hf_gradient(g_mp2_a, f_mo_a, na)
    g_mp2_b = embed_active_columns(g_mp2_b, fc, nb)
    g_b = add_hf_gradient(g_mp2_b, f_mo_b, nb)
    p_oo_a, p_vv_a, p_oo_b, p_vv_b = u_mp2_density_from_amplitudes(t_aa, t_bb, t_ab)
    w_a = embed_u_density(p_oo_a, p_vv_a, fc, na, NBF)
    w_b = embed_u_density(p_oo_b, p_vv_b, fc, nb, NBF)
    pa_ao = c_a_sc @ w_a @ c_a_sc.T
    pb_ao = c_b_sc @ w_b @ c_b_sc.T
    j_tot, k_a, k_b = spin_jk(eri, pa_ao, pb_ao)

    def add_response(g, c, k, w, f_mo, nocc_total):
        g_mo = c.T @ (j_tot - k) @ c
        fw, wf = f_mo @ w, w @ f_mo
        nvir, nocc_cols = g.shape
        for a in range(nvir):
            a_mo = nocc_total + a
            for i in range(nocc_cols):
                g[a, i] += 2.0 * g_mo[a_mo, i] + 2.0 * (fw[a_mo, i] - wf[a_mo, i])

    add_response(g_a, c_a_sc, k_a, w_a, f_mo_a, na)
    add_response(g_b, c_b_sc, k_b, w_b, f_mo_b, nb)

    def back(g, u, nocc, nocc_total):
        g_act = g[:, fc : fc + nocc]
        u_o = u[fc : fc + nocc, fc : fc + nocc]
        u_v = u[nocc_total:, nocc_total:]
        return u_v @ g_act @ u_o.T

    return back(g_a, u_a, na - fc, na), back(g_b, u_b, nb - fc, nb)


# ========================================== independent routes
def block_energies(bfac, ca, cb, ea, eb, na, nb, fc):
    """(E_aa, E_bb, E_ab) from their definitions, denominators fixed."""
    bfa, bfb = b_full_mo(bfac, ca), b_full_mo(bfac, cb)
    t_aa, t_bb, t_ab = amplitudes(bfa, bfb, ea, eb, na, nb, fc)
    oa_, va_, ob_, vb_ = slice(fc, na), slice(na, None), slice(fc, nb), slice(nb, None)
    g_aa = np.einsum("Pia,Pjb->ijab", bfa[:, oa_, va_], bfa[:, oa_, va_])
    g_bb = np.einsum("Pia,Pjb->ijab", bfb[:, ob_, vb_], bfb[:, ob_, vb_])
    g_ab = np.einsum("Pia,Pjb->ijab", bfa[:, oa_, va_], bfb[:, ob_, vb_])
    e_aa = 0.25 * np.sum(t_aa * (g_aa - g_aa.transpose(0, 1, 3, 2)))
    e_bb = 0.25 * np.sum(t_bb * (g_bb - g_bb.transpose(0, 1, 3, 2)))
    e_ab = np.sum(t_ab * g_ab)
    return e_aa, e_bb, e_ab


def rotated(c, nocc, c_vir, k_occ, step):
    k = np.zeros((NBF, NBF))
    k[nocc + c_vir, k_occ] = step
    return c @ expm(k - k.T)


def u_mp2_energy_so(bfac, ca, cb, ea, eb, na, nb, fc=0):
    """Brute-force spin-orbital MP2, 1/4 sum |<IJ||AB>|^2 / D (no block split),
    over the active occupied spin orbitals (the lowest fc of each spin frozen)."""
    n = NBF
    cso = np.hstack([ca, cb])
    spin = np.array([0] * n + [1] * n)
    eso = np.concatenate([ea, eb])
    occ = list(range(fc, na)) + [n + k for k in range(fc, nb)]
    vir = list(range(na, n)) + [n + k for k in range(nb, n)]
    bso = np.einsum("Pmn,mp,nq->Ppq", bfac, cso, cso)
    chem = np.einsum("Ppq,Prs->pqrs", bso, bso)
    same = spin[:, None] == spin[None, :]
    chem = chem * same[:, :, None, None] * same[None, None, :, :]
    phys = chem.transpose(0, 2, 1, 3) - chem.transpose(0, 2, 3, 1)
    asym = phys[np.ix_(occ, occ, vir, vir)]
    eo, ev = eso[occ], eso[vir]
    dd = eo[:, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None]
    dd = dd - ev[None, None, None, :]
    return 0.25 * np.sum(asym**2 / dd)


def u_functional(h, eri, bfac, ca, cb, na, nb, fc=0):
    """E_UHF + E_MP2 in each spin's semicanonical frame (numpy eigh, not the
    transcription); the core block (the lowest fc) is left as it is."""
    e_hf, fa, fb = compute_uhf_energy(eri, ca, cb, na, nb, h)
    out = []
    for c, f, no in ((ca, fa, na), (cb, fb, nb)):
        fm = c.T @ f @ c
        wo, vo = np.linalg.eigh(fm[fc:no, fc:no])
        wv, vv = np.linalg.eigh(fm[no:, no:])
        u = np.eye(NBF)
        u[fc:no, fc:no], u[no:, no:] = vo, vv
        eps = np.diag(fm).copy()
        eps[fc:no], eps[no:] = wo, wv
        out.append((c @ u, eps))
    (sa, ea), (sb, eb) = out
    return e_hf + u_mp2_energy_so(bfac, sa, sb, ea, eb, na, nb, fc)


def derivation_blocks(bfac, ca, cb, ea, eb, na, nb):
    """Section 14's formula: sum t d(ia|jb), loop over every (c, k), first_occ = 0."""
    bfa, bfb = b_full_mo(bfac, ca), b_full_mo(bfac, cb)
    t_aa, t_bb, t_ab = amplitudes(bfa, bfb, ea, eb, na, nb, 0)
    g = {
        "aa": np.einsum("Ppq,Prs->pqrs", bfa, bfa),
        "bb": np.einsum("Ppq,Prs->pqrs", bfb, bfb),
        "ab": np.einsum("Ppq,Prs->pqrs", bfa, bfb),
    }
    out = {}
    for spin, no, t_ss, gss in (("a", na, t_aa, g["aa"]), ("b", nb, t_bb, g["bb"])):
        o, v = slice(0, no), slice(no, NBF)
        nv = NBF - no
        same, cross = np.zeros((nv, no)), np.zeros((nv, no))
        for c, k in itertools.product(range(nv), range(no)):
            cc = no + c
            d = np.zeros((no, no, nv, nv))
            d[k] += gss[cc, v, o, v].transpose(1, 0, 2)
            d[:, k] += gss[o, v, cc, v]
            d[:, :, c, :] -= gss[o, k, o, v]
            d[:, :, :, c] -= gss[o, v, o, k].transpose(0, 2, 1)
            same[c, k] = np.sum(t_ss * d)
            gab = g["ab"]
            if spin == "a":
                obb, vbb = slice(0, nb), slice(nb, NBF)
                dd = np.zeros(t_ab.shape)
                dd[k] += gab[cc, v, obb, vbb].transpose(1, 0, 2)
                dd[:, :, c, :] -= gab[o, k, obb, vbb]
            else:
                oaa, vaa = slice(0, na), slice(na, NBF)
                dd = np.zeros(t_ab.shape)
                dd[:, k] += gab[oaa, vaa, cc, v]
                dd[:, :, :, c] -= gab[oaa, vaa, o, k].transpose(0, 2, 1)
            cross[c, k] = 2.0 * np.sum(t_ab * dd)
        out["same_" + spin], out["ab_" + spin] = same, cross
    return out


# ========================================== model
def rsym(n, scale):
    m = RNG.standard_normal((n, n))
    return scale * (m + m.T)


H = np.diag(np.linspace(-3.0, 1.8, NBF)) + rsym(NBF, 0.1)
LFAC = np.array([rsym(NBF, 0.1) for _ in range(NL)])
ERI = np.einsum("Kmn,Kls->mnls", LFAC, LFAC)
BFAC = np.array([rsym(NBF, 0.12) for _ in range(NAUX)])
NA, NB = 3, 2


def random_orbitals(scale):
    c0 = np.linalg.eigh(H)[1]
    k = RNG.standard_normal((NBF, NBF)) * scale
    return c0 @ expm(k - k.T)


def fixed_eps(no):
    return np.concatenate(
        [np.linspace(-1.6, -0.7, no), np.linspace(0.4, 2.1, NBF - no)]
    )


# ========================================== checks
def check_symbolic():
    no, nv = 3, 2
    t = sp.MutableDenseNDimArray.zeros(no, no, nv, nv)
    for i, j in itertools.combinations(range(no), 2):
        for a, b in itertools.combinations(range(nv), 2):
            s = sp.Symbol(f"t{i}{j}{a}{b}")
            for ii, jj, si in ((i, j, 1), (j, i, -1)):
                t[ii, jj, a, b] = si * s
                t[ii, jj, b, a] = -si * s

    def blk(i, j):
        return sp.Matrix(nv, nv, lambda a, b: t[i, j, a, b])

    worst = 0
    for k, j in itertools.product(range(no), repeat=2):
        m = pair_combination(blk(k, j), blk(j, k), 1, 1)
        worst += sum(sp.simplify(x) != 0 for x in (m - 4 * blk(k, j)))
    check("pair-combination", worst == 0, "M_kj = 4 A_kj for antisymmetric t")

    # Chain-rule factors on a 2-occ x 2-vir block with generic integrals.
    lam = sp.Symbol("l")
    idx = list(itertools.product(range(2), range(2), range(2), range(2)))
    g = {q: sp.Symbol(f"g{q[0]}{q[1]}{q[2]}{q[3]}") for q in idx}
    dg = {q: sp.Symbol(f"h{q[0]}{q[1]}{q[2]}{q[3]}") for q in idx}
    # D_ijab = D_ijba: only the a <-> b symmetry of the denominator is used.
    dn = {
        (i, j, a, b): sp.Symbol(f"D{i}{j}{min(a, b)}{max(a, b)}")
        for (i, j, a, b) in idx
    }

    def ival(m, i, a, j, b):
        key = sorted([(i, a), (j, b)])
        return m[(key[0][0], key[1][0], key[0][1], key[1][1])]

    def e_ss(s):
        tot = 0
        for i, j, a, b in idx:
            k_ = ival(g, i, a, j, b) + s * ival(dg, i, a, j, b)
            k_ -= ival(g, i, b, j, a) + s * ival(dg, i, b, j, a)
            tot += sp.Rational(1, 4) * k_**2 / dn[(i, j, a, b)]
        return tot

    t_ss = {
        (i, j, a, b): (ival(g, i, a, j, b) - ival(g, i, b, j, a)) / dn[(i, j, a, b)]
        for i, j, a, b in idx
    }
    lhs = sp.diff(e_ss(lam), lam).subs(lam, 0)
    rhs = sum(t_ss[q] * ival(dg, q[0], q[2], q[1], q[3]) for q in idx)
    ok_ss = sp.simplify(sp.expand(lhs - rhs)) == 0

    ga = {q: sp.Symbol(f"G{q[0]}{q[1]}{q[2]}{q[3]}") for q in idx}
    da = {q: sp.Symbol(f"H{q[0]}{q[1]}{q[2]}{q[3]}") for q in idx}
    dab = {q: sp.Symbol(f"E{q[0]}{q[1]}{q[2]}{q[3]}") for q in idx}
    e_os = sum((ga[q] + lam * da[q]) ** 2 / dab[q] for q in idx)
    lhs = sp.diff(e_os, lam).subs(lam, 0)
    rhs = 2 * sum(ga[q] / dab[q] * da[q] for q in idx)
    ok_os = sp.simplify(lhs - rhs) == 0
    check(
        "chain-rule-factors",
        ok_ss and ok_os,
        f"same-spin {ok_ss}, opposite-spin {ok_os}",
    )


def rel_err(a, b):
    return np.max(np.abs(a - b)) / max(1.0, np.max(np.abs(b)))


def check_kernel():
    ca, cb = random_orbitals(0.3), random_orbitals(0.3)
    ea, eb = fixed_eps(NA), fixed_eps(NB)
    got = kernel_at(BFAC, ca, cb, ea, eb, NA, NB, 0)
    want = derivation_blocks(BFAC, ca, cb, ea, eb, NA, NB)
    err = max(rel_err(got[k], want[k]) for k in got)
    check("kernel-equals-derivation", err < 1e-13, f"max rel {err:.1e} over 4 blocks")

    for fc, name in ((0, "kernel-vs-fd"), (1, "kernel-vs-fd-frozen-core")):
        got = kernel_at(BFAC, ca, cb, ea, eb, NA, NB, fc)
        worst, gmax = 0.0, 0.0
        for spin, no, keys in (
            ("a", NA, ("same_a", "ab_a")),
            ("b", NB, ("same_b", "ab_b")),
        ):
            fd = {k: np.zeros_like(got[k]) for k in keys}
            for c, k in itertools.product(range(NBF - no), range(no - fc)):

                def e_of(s, c=c, k=k, spin=spin, fc=fc):
                    if spin == "a":
                        e = block_energies(
                            BFAC, rotated(ca, NA, c, fc + k, s), cb, ea, eb, NA, NB, fc
                        )
                        return np.array([e[0], e[2]])
                    e = block_energies(
                        BFAC, ca, rotated(cb, NB, c, fc + k, s), ea, eb, NA, NB, fc
                    )
                    return np.array([e[1], e[2]])

                d = richardson(e_of, 1e-3)
                fd[keys[0]][c, k], fd[keys[1]][c, k] = d
            for key in keys:
                worst = max(worst, rel_err(got[key], fd[key]))
                gmax = max(gmax, np.max(np.abs(fd[key])))
        check(
            name,
            worst < 1e-9,
            f"max rel {worst:.1e} (max|g| {gmax:.2e}, frozen core {fc})",
        )

    # Exactness anchor: spin symmetry and the zero-coupling limit.
    c = random_orbitals(0.3)
    e2 = fixed_eps(NA)
    bl = kernel_at(BFAC, c, c, e2, e2, NA, NA, 0)
    sym_err = max(
        np.max(np.abs(bl["same_a"] - bl["same_b"])),
        np.max(np.abs(bl["ab_a"] - bl["ab_b"])),
    )
    zero = kernel_at(0.0 * BFAC, ca, cb, ea, eb, NA, NB, 0)
    zmax = max(np.max(np.abs(v)) for v in zero.values())
    check(
        "closed-shell-anchor",
        sym_err < 1e-13 and zmax == 0.0,
        f"|g_a - g_b| {sym_err:.1e}, B = 0 gives {zmax:.1e}",
    )


def check_assembled():
    ca, cb = random_orbitals(0.15), random_orbitals(0.15)
    for fc, name in ((0, "assembled-vs-fd"), (1, "assembled-vs-fd-frozen-core")):
        ga, gb = u_gradient_solver_frame(H, ERI, BFAC, ca, cb, NA, NB, fc)
        worst, gmax = 0.0, 0.0
        for spin, g, no in (("a", ga, NA), ("b", gb, NB)):
            fd = np.zeros_like(g)
            for c, k in itertools.product(range(NBF - no), range(no - fc)):

                def e_of(s, c=c, k=k, spin=spin, fc=fc):
                    if spin == "a":
                        return u_functional(
                            H, ERI, BFAC, rotated(ca, NA, c, fc + k, s), cb, NA, NB, fc
                        )
                    return u_functional(
                        H, ERI, BFAC, ca, rotated(cb, NB, c, fc + k, s), NA, NB, fc
                    )

                fd[c, k] = richardson(e_of, 1e-3)
            worst = max(worst, rel_err(g, fd))
            gmax = max(gmax, np.max(np.abs(fd)))
        check(
            name,
            worst < 1e-9,
            f"max rel {worst:.1e} (max|g| {gmax:.2e}, frozen core {fc})",
        )


check_symbolic()
check_kernel()
check_assembled()

if FAILED:
    print(f"\n{len(FAILED)} check(s) failed: {FAILED}")
    sys.exit(1)
print("\nall U-OO-RI-MP2 response identities hold")
