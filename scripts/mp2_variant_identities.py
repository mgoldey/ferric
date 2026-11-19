#!/usr/bin/env python3
"""Numpy checks of the algebraic identities ferric-mp2's MP2 variants rely on.

Theory and notation: the MP2 chapter, book/src/10-mp2/. Every claim builds random
inputs with the symmetry the code assumes (real orbitals, density-fitted
(pq|rs) = sum_P B^P_pq B^P_rs), checks the identity to 1e-10 relative, and
carries a negative control: a plausible wrong variant that must NOT satisfy
it. Exits nonzero on the first failure.

  1  Closed-shell spin components (rimp2.rs spin_components_from_g,
     pair_energy and its j >= i weighting): the spatial
     E_OS = sum (ia|jb)^2/D and E_SS = sum (ia|jb)[(ia|jb)-(ib|ja)]/D equal
     the opposite-spin and same-spin parts of the spin-orbital
     1/4 sum |<ij||ab>|^2/D, and the unique-pair sum (j >= i, off-diagonal
     weight 2) equals the full (i, j) sum. Controls: E_SS without the
     exchange term; the unique-pair sum with off-diagonal weight 1.
  2  Unrestricted components (u_rimp2.rs same_spin_block_energy :586,
     opposite_spin_block_energy :624): with independent alpha/beta orbitals
     1/4 sum_{j>=i} fac sum_ab K^2/D (K = (ia|jb) - (ib|ja)) is the
     spin-orbital alpha-alpha part and sum (ia|JB)^2/D the alpha-beta part;
     with alpha = beta, e_aa + e_bb = E_SS and e_ab = E_OS of claim 1.
     Control: the same-spin kernel without its 1/4.
  3  Laplace integrand at fixed t (laplace.rs weighted_b_mo :666,
     laplace_exchange_energy :479, laplace_ao_coulomb_energy :568,
     build_pseudo_density_occ/vir :1500/:1522): with
     B^P_ia(t) = B^P_ia exp(-t(e_a - e_i)/2),
     sum_PQ J_PQ(t)^2 = sum (ia|jb)^2 exp(-tD) and
     sum_PQij G_PQ[i,j] G_PQ[j,i] = sum (ia|jb)(ib|ja) exp(-tD), and the AO
     trace Tr[B^P P(t) B^Q Q(t)] equals J_PQ(t). Control: the AO exchange
     trace sum_PQ Tr(M^P N^Q M^Q N^P) (M = B P(t), N = B Q(t)) is not the
     exchange term.
  4  Attenuated-operator split (attenuated.rs rs_mp2_decomposition :345):
     with (ia|jb) = s + l (short + long range) and E[g] = sum g(2g - g^x)/D
     (g^x = (ib|ja)), E[s+l] = E[s] + E[l] + X with the cross term
     X = sum (4 s l - s l^x - l s^x)/D. X != 0, so E[erfc] + E[erf] is not
     E[Coulomb]. Control: dropping X fails. With the symmetric bilinear form
     B(g, h) = sum g(2h - h^x)/D (E[g] = B(g, g)), X = 2 B(s, l).
     Control: X is not B(s, l).
  5  Z-vector matvec (zvector.rs compute_az_product :460 / :502, used at
     :159 with factor 1/2): the vo block of (4J - K - K^T)[D^z] with
     D^z = sum_ai z_ai (C_a C_i^T + C_i C_a^T) equals
     2 sum_bj [4(ai|bj) - (ab|ij) - (aj|bi)] z_bj, i.e. twice the coupling
     of the (A+B) Hessian that cpks_polar.rs build_apb_amb builds. Control: the unhalved product is not (A+B)z.
  6  Cayley rotation (orbital_rotation.rs cayley_rotation :53):
     U = (I - k/2)^-1 (I + k/2) is orthogonal and U - expm(k) = O(|k|^3).
     Control: (I + k/2)^-1 (I - k/2) matches expm(-k), not expm(k).
  7  CABS (f12.rs build_cabs): coefficients [-S_OO^-1 S_Oa v; v]/sqrt(w)
     from the Schur complement S_aa - S_aO S_OO^-1 S_Oa = V w V^T are
     orthonormal in S_RR and orthogonal to OBS. Control: the eigenvectors
     of the full-RIBS complement S_RR - S_RO S_OO^-1 S_OR scaled by
     w^-1/2, without the OBS projection, are neither. The same holds on the
     interleaved atom-major union build_cabs uses (per atom, OBS functions
     then aux), partitioned by the provenance mask and scattered back.
     Control: the [O | a] rows read as union rows are not orthogonal to OBS.

  8  Density-fitting error (rimp2.rs cholesky_inverse_sqrt, the metric and
     the three-index integrals in one operator w): in the w inner product,
     with B = L^-1 (Q|ia), V = L L^T, the fitting error
     (ia|jb) - sum_P B^P_ia B^P_jb equals (delta_ia|delta_jb) with the
     residual delta_ia = rho_ia - sum_PQ chi_P V^-1_PQ (Q|ia), the residual
     is w-orthogonal to every chi_P, and the diagonal error is >= 0. Control:
     a metric in a second operator (mixed metric) breaks the identity.

  9  Chunked dressing (rimp2.rs stream_dressed_mo_band_budgeted): with the
     raw rows (Q|mu nu) transformed chunk by chunk to (Q|ia) and each chunk
     added into B with beta = 1, B = M (Q|ia) for chunk widths 1, 7, 32 and
     the full width, and an output band [p0, p1) gives rows p0..p1 of B.
     Control: dropping one chunk.
 10  Auxiliary bands (mpi_rimp2.rs run_mpi_ri_mp2_inner, parallel.rs
     aux_band): the bands of aux_band partition [0, naux); the band partials
     G^(i,r) = B_{P in band,i}^T B_{P in band} sum to the full-width G^(i);
     the round-robin owner energy with weight 1 over all j equals claim 1's
     E_OS and E_SS. Control: squaring each band's partial before the sum.
 11  Summation-order bound (rimp2_gpu.rs, f64 device path): two f64
     summation orders of N_aux = 912 products are each within
     gamma_N(u64) S of the exact sum (Fractions) and within 2 gamma_N(u64) S
     of each other, S = sum |x y|; 2 gamma_912(u64) = 2.0e-13. Control: an
     f32 sum breaks the f64 bound.
 12  Mixed-precision element bound (gpu/mixed.rs gemm_f32_f64acc_dev_into):
     f32 operands, f32 sums within panels of b = 64, f64 sum over panels,
     stay within eps_G S of the exact sum on random-sign data, positive data
     and a constructed panel whose small terms are each below half an f32
     ulp of the running sum (lost in f32, err/S = 1.9e-6, half of eps_G);
     eps_G = 3.93e-6 at N_aux = 84 and 912, the f32 panel term is 97% of
     it, gamma_912(u32) = 5.4e-5, and sqrt(f32_max/64) = 2.3e18. The f64
     panel sum of +2^20, 2^-9, -2^20 is exactly 2^-9 and an f32 sum is 0.
     Control: a dropped panel breaks the bound.

Mutations (each turns the script red with the named failing check):
  * claim 1, same-spin numerator (ia|jb) - (ib|ja) -> (ia|jb) + (ib|ja)
    ("FAIL E_SS spatial").
  * claim 2, same-spin kernel prefactor 0.25 -> 0.5 ("FAIL same-spin kernel").
  * claim 3, Laplace half-exponent of B(t), -0.5 t -> -1.0 t
    ("FAIL sum_PQ J_PQ(t)^2").
  * claim 4, cross-term coefficient 4 s l -> 2 s l ("FAIL E[s+l]").
  * claim 5, (4J - K - K^T) -> (4J - K) in the matvec ("FAIL (4J-K-K^T)").
  * claim 7, the OBS rows scattered without the minus sign
    ("FAIL interleaved S_O,R C == 0").
  * claim 4, the bilinear form with h^x added ("FAIL X == 2 B(s, l)").
  * claim 8, the residual built with the overlap metric chi^T chi in place
    of V ("FAIL (ia|jb) - sum_P B B == (delta_ia|delta_jb)").

  * claim 8, the dressing accumulates with beta = 0 instead of 1
    ("FAIL chunked dressing").
  * claim 9, aux_band gives every rank q rows, dropping the remainder
    ("FAIL band partials sum to G^(i)").
  * claim 10, u64 = 2^-53 -> 2^-70 in the bound ("FAIL two f64 orders").
  * claim 11, the panel term gamma_b(u32) -> gamma_b(u64) in eps_G
    ("FAIL mixed bound, sub-ulp panel").

Run: python3 scripts/mp2_variant_identities.py  (needs numpy, scipy)
"""

import sys
from fractions import Fraction

import numpy as np
from scipy.linalg import expm

RNG = np.random.default_rng(20261105)
TOL = 1e-10
FAILED = []


def check(name, got, want, tol=TOL):
    err = abs(got - want) / max(1.0, abs(want))
    ok = err <= tol
    print(
        f"  {'ok  ' if ok else 'FAIL'} {name}: got {got:.15e} want {want:.15e} rel {err:.1e}"
    )
    if not ok:
        FAILED.append(name)


def check_arrays(name, got, want, tol=TOL):
    """Max-element error relative to the largest element of `want` (or 1)."""
    scale = max(1.0, float(np.max(np.abs(want))))
    err = float(np.max(np.abs(got - want))) / scale
    ok = err <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max rel error {err:.1e}")
    if not ok:
        FAILED.append(name)


def check_differs(name, got, want, min_rel=1e-6):
    err = abs(got - want) / max(1e-300, abs(want))
    ok = err >= min_rel
    print(
        f"  {'ok  ' if ok else 'FAIL'} control {name}: rel diff {err:.1e} (must be >= {min_rel:.0e})"
    )
    if not ok:
        FAILED.append("control " + name)


def random_b(naux, n):
    """Symmetric RI factors B^P_pq = B^P_qp, so (pq|rs) has 8-fold symmetry."""
    b = RNG.standard_normal((naux, n, n))
    return 0.5 * (b + b.transpose(0, 2, 1))


def eri_from_b(b1, b2=None):
    b2 = b1 if b2 is None else b2
    return np.einsum("Ppq,Prs->pqrs", b1, b2)


def orbital_energies(no, nv):
    eo = np.sort(RNG.uniform(-2.0, -0.4, no))
    ev = np.sort(RNG.uniform(0.2, 3.0, nv))
    return eo, ev


def denom(eo, ev):
    """D_ijab = e_i + e_j - e_a - e_b (negative), the code's sign."""
    return (
        eo[:, None, None, None]
        + eo[None, :, None, None]
        - ev[None, None, :, None]
        - ev[None, None, None, :]
    )


# ---------------------------------------------------------------- claim 1
def claim1():
    print("claim 1: closed-shell OS/SS from the spin-orbital MP2 energy")
    no, nv, naux = 3, 4, 9
    n = no + nv
    b = random_b(naux, n)
    eri = eri_from_b(b)
    eo, ev = orbital_energies(no, nv)
    o, v = slice(0, no), slice(no, n)
    g = eri[o, v, o, v]  # g[i,a,j,b] = (ia|jb)
    giajb = g.transpose(0, 2, 1, 3)  # [i,j,a,b]
    gibja = g.transpose(0, 2, 3, 1)  # [i,j,a,b] -> (ib|ja)
    d = denom(eo, ev)
    e_os = np.sum(giajb**2 / d)
    e_ss = np.sum(giajb * (giajb - gibja) / d)

    # Spin-orbital reference: spin orbital 2p + s, s = 0 alpha, 1 beta.
    ns = 2 * n
    spin = np.array([s for p in range(n) for s in (0, 1)])
    spat = np.array([p for p in range(n) for s in (0, 1)])
    chem = eri[np.ix_(spat, spat, spat, spat)]
    same = spin[:, None] == spin[None, :]
    chem = chem * same[:, :, None, None] * same[None, None, :, :]
    # <pq||rs> = (pr|qs) - (ps|qr)
    phys = chem.transpose(0, 2, 1, 3) - chem.transpose(0, 2, 3, 1)
    occ = np.array([q for q in range(ns) if spat[q] < no])
    vir = np.array([q for q in range(ns) if spat[q] >= no])
    asym = phys[np.ix_(occ, occ, vir, vir)]
    eos_ = np.concatenate([eo, ev])
    e_so = eos_[spat]
    dso = (
        e_so[occ][:, None, None, None]
        + e_so[occ][None, :, None, None]
        - e_so[vir][None, None, :, None]
        - e_so[vir][None, None, None, :]
    )
    terms = 0.25 * asym**2 / dso
    si, sj = spin[occ][:, None], spin[occ][None, :]
    so_os = np.sum(terms[(si != sj)])
    so_ss = np.sum(terms[(si == sj)])
    check("E_OS spatial == spin-orbital opposite-spin", e_os, so_os)
    check("E_SS spatial == spin-orbital same-spin", e_ss, so_ss)
    check("E_OS + E_SS == 1/4 sum |<ij||ab>|^2/D", e_os + e_ss, np.sum(terms))

    # Unique-pair weighting of rimp2.rs:1410-1425.
    def pair_sum(off_weight):
        tot_os = tot_ss = 0.0
        for i in range(no):
            for j in range(i, no):
                fac = 1.0 if i == j else off_weight
                tot_os += fac * np.sum(giajb[i, j] ** 2 / d[i, j])
                tot_ss += fac * np.sum(
                    giajb[i, j] * (giajb[i, j] - gibja[i, j]) / d[i, j]
                )
        return tot_os, tot_ss

    p_os, p_ss = pair_sum(2.0)
    check("unique-pair (j>=i, fac 2) E_OS == full sum", p_os, e_os)
    check("unique-pair (j>=i, fac 2) E_SS == full sum", p_ss, e_ss)
    check_differs("E_SS without exchange", np.sum(giajb**2 / d), so_ss)
    check_differs("unique-pair with off-diagonal weight 1", pair_sum(1.0)[0], e_os)


# ---------------------------------------------------------------- claim 2
def claim2():
    print("claim 2: unrestricted alpha-alpha / alpha-beta kernels")
    nbf, naux = 7, 10
    no_a, no_b = 3, 2
    b_ao = random_b(naux, nbf)
    ca = np.linalg.qr(RNG.standard_normal((nbf, nbf)))[0]
    cb = np.linalg.qr(RNG.standard_normal((nbf, nbf)))[0]
    ea_o, ea_v = orbital_energies(no_a, nbf - no_a)
    eb_o, eb_v = orbital_energies(no_b, nbf - no_b)

    def mo_b(c, no):
        return np.einsum("Pmn,mi,na->Pia", b_ao, c[:, :no], c[:, no:])

    ba, bb = mo_b(ca, no_a), mo_b(cb, no_b)

    def same_spin(bs, eo, ev, scale=0.25):
        g = np.einsum("Pia,Pjb->ijab", bs, bs)
        k = g - g.transpose(0, 1, 3, 2)
        d = denom(eo, ev)
        tot = 0.0
        for i in range(len(eo)):
            for j in range(i, len(eo)):
                fac = 1.0 if i == j else 2.0
                tot += fac * np.sum(k[i, j] ** 2 / d[i, j])
        return scale * tot

    def opposite_spin(b1, e1o, e1v, b2, e2o, e2v):
        g = np.einsum("Pia,PJB->iJaB", b1, b2)
        d = (
            e1o[:, None, None, None]
            + e2o[None, :, None, None]
            - e1v[None, None, :, None]
            - e2v[None, None, None, :]
        )
        return np.sum(g**2 / d)

    e_aa = same_spin(ba, ea_o, ea_v)
    e_ab = opposite_spin(ba, ea_o, ea_v, bb, eb_o, eb_v)

    # Spin-orbital reference with separate alpha/beta MO sets.
    def so_parts():
        orbs = [
            (ca[:, p], 0, e) for p, e in zip(range(nbf), np.concatenate([ea_o, ea_v]))
        ]
        orbs += [
            (cb[:, p], 1, e) for p, e in zip(range(nbf), np.concatenate([eb_o, eb_v]))
        ]
        occ = [k for k in range(nbf)][:no_a] + [nbf + k for k in range(no_b)]
        vir = [k for k in range(no_a, nbf)] + [nbf + k for k in range(no_b, nbf)]
        cmat = np.stack([o[0] for o in orbs], axis=1)
        spin = np.array([o[1] for o in orbs])
        eps = np.array([o[2] for o in orbs])
        bso = np.einsum("Pmn,mp,nq->Ppq", b_ao, cmat, cmat)
        chem = np.einsum("Ppq,Prs->pqrs", bso, bso)
        same = spin[:, None] == spin[None, :]
        chem = chem * same[:, :, None, None] * same[None, None, :, :]
        phys = chem.transpose(0, 2, 1, 3) - chem.transpose(0, 2, 3, 1)
        asym = phys[np.ix_(occ, occ, vir, vir)]
        eo, ev = eps[occ], eps[vir]
        d = (
            eo[:, None, None, None]
            + eo[None, :, None, None]
            - ev[None, None, :, None]
            - ev[None, None, None, :]
        )
        t = 0.25 * asym**2 / d
        si, sj = spin[occ][:, None], spin[occ][None, :]
        return np.sum(t[(si == 0) & (sj == 0)]), np.sum(t[si != sj])

    so_aa, so_ab = so_parts()
    check("same-spin kernel == spin-orbital alpha-alpha", e_aa, so_aa)
    check("opposite-spin kernel == spin-orbital alpha-beta", e_ab, so_ab)
    check_differs("same-spin kernel without 1/4", same_spin(ba, ea_o, ea_v, 1.0), so_aa)

    # Restricted limit: alpha = beta reproduces claim 1's E_SS, E_OS.
    g = np.einsum("Pia,Pjb->ijab", ba, ba)
    d = denom(ea_o, ea_v)
    e_os = np.sum(g**2 / d)
    e_ss = np.sum(g * (g - g.transpose(0, 1, 3, 2)) / d)
    check(
        "restricted limit: e_aa + e_bb == E_SS", 2.0 * same_spin(ba, ea_o, ea_v), e_ss
    )
    check(
        "restricted limit: e_ab == E_OS",
        opposite_spin(ba, ea_o, ea_v, ba, ea_o, ea_v),
        e_os,
    )


# ---------------------------------------------------------------- claim 3
def claim3():
    print("claim 3: Laplace integrand factorization at fixed t")
    nbf, no, naux = 7, 3, 8
    nv = nbf - no
    b_ao = random_b(naux, nbf)
    c = RNG.standard_normal((nbf, nbf))  # any coefficients: the identity is algebraic
    eo, ev = orbital_energies(no, nv)
    t = 0.37
    b = np.einsum("Pmn,mi,na->Pia", b_ao, c[:, :no], c[:, no:])
    g = np.einsum("Pia,Pjb->ijab", b, b)
    w = np.exp(-t * (-denom(eo, ev)))  # exp(-t x), x = e_a + e_b - e_i - e_j > 0
    coul_ref = np.sum(g**2 * w)
    exch_ref = np.sum(g * g.transpose(0, 1, 3, 2) * w)

    bt = b * np.exp(-0.5 * t * (ev[None, None, :] - eo[None, :, None]))
    jpq = np.einsum("Pia,Qia->PQ", bt, bt)
    gpq = np.einsum("Pia,Qja->PQij", bt, bt)
    exch = np.einsum("PQij,PQji->", gpq, gpq)
    check("sum_PQ J_PQ(t)^2 == sum (ia|jb)^2 e^{-tx}", np.sum(jpq**2), coul_ref)
    check("sum G_PQ[i,j] G_PQ[j,i] == sum (ia|jb)(ib|ja) e^{-tx}", exch, exch_ref)

    pt = c[:, :no] @ np.diag(np.exp(t * eo)) @ c[:, :no].T
    qt = c[:, no:] @ np.diag(np.exp(-t * ev)) @ c[:, no:].T
    m = np.einsum("Pmn,nl->Pml", b_ao, pt)  # M^P = B^P P(t)
    nmat = np.einsum("Pmn,nl->Pml", b_ao, qt)  # N^Q = B^Q Q(t)
    j_ao = np.einsum("Pmn,Qnm->PQ", m, nmat)
    check_arrays("Tr[B^P P(t) B^Q Q(t)] == J_PQ(t)", j_ao, jpq)
    wrong = np.einsum("Pmn,Qnl,Qlk,Pkm->", m, nmat, m, nmat)
    check_differs("sum_PQ Tr(M^P N^Q M^Q N^P) as the exchange term", wrong, exch_ref)


# ---------------------------------------------------------------- claim 4
def claim4():
    print("claim 4: MP2 energy is not additive in a split operator")
    no, nv, naux = 3, 4, 7
    eo, ev = orbital_energies(no, nv)
    d = denom(eo, ev)
    bs = RNG.standard_normal((naux, no, nv))
    bl = 0.3 * RNG.standard_normal((naux, no, nv))
    gs = np.einsum("Pia,Pjb->ijab", bs, bs)
    gl = np.einsum("Pia,Pjb->ijab", bl, bl)

    def e(g):
        return np.sum(g * (2.0 * g - g.transpose(0, 1, 3, 2)) / d)

    x = np.sum(
        (4.0 * gs * gl - gs * gl.transpose(0, 1, 3, 2) - gl * gs.transpose(0, 1, 3, 2))
        / d
    )
    check("E[s+l] == E[s] + E[l] + X", e(gs + gl), e(gs) + e(gl) + x)
    check_differs("E[s] + E[l] as E[s+l] (X = 0)", e(gs) + e(gl), e(gs + gl))

    def bilinear(g, h):
        return np.sum(g * (2.0 * h - h.transpose(0, 1, 3, 2)) / d)

    check("B(s, l) == B(l, s)", bilinear(gs, gl), bilinear(gl, gs))
    check("X == 2 B(s, l)", x, 2.0 * bilinear(gs, gl))
    check_differs("X as B(s, l) (factor 2 dropped)", bilinear(gs, gl), x)


# ---------------------------------------------------------------- claim 5
def claim5():
    print("claim 5: Z-vector Hessian matvec factor")
    nbf, no, naux = 7, 3, 12
    nv = nbf - no
    b_ao = random_b(naux, nbf)
    eri_ao = eri_from_b(b_ao)
    c = RNG.standard_normal((nbf, nbf))
    z = RNG.standard_normal((nv, no))
    co, cv = c[:, :no], c[:, no:]
    dz = np.einsum("ai,ma,ni->mn", z, cv, co) + np.einsum("ai,mi,na->mn", z, co, cv)
    jz = np.einsum("mnls,ls->mn", eri_ao, dz)
    kz = np.einsum("mlns,ls->mn", eri_ao, dz)
    az = cv.T @ (4.0 * jz - kz - kz.T) @ co  # vo block, compute_az_product

    eri_mo = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri_ao, c, c, c, c)
    o, v = slice(0, no), slice(no, nbf)
    ai_bj = eri_mo[v, o, v, o]  # (ai|bj)
    ab_ij = eri_mo[v, v, o, o]  # (ab|ij)
    aj_bi = eri_mo[v, o, v, o].transpose(0, 3, 2, 1)  # (aj|bi) as [a,i,b,j]
    coupling = 4.0 * ai_bj - ab_ij.transpose(0, 2, 1, 3) - aj_bi
    apb_z = np.einsum("aibj,bj->ai", coupling, z)
    check_arrays("(4J-K-K^T)[D^z]_vo == 2 (A+B-De) z", az, 2.0 * apb_z)
    check_differs("unhalved product as (A+B-De) z", np.sum(az * z), np.sum(apb_z * z))


# ---------------------------------------------------------------- claim 6
def claim6():
    print("claim 6: Cayley rotation")
    n = 6
    k0 = RNG.standard_normal((n, n))
    k0 = k0 - k0.T
    eye = np.eye(n)

    def cayley(k):
        return np.linalg.solve(eye - 0.5 * k, eye + 0.5 * k)

    u = cayley(0.2 * k0)
    check_arrays("U^T U == I", u.T @ u, eye)
    errs = []
    for s in (1e-2, 5e-3):
        errs.append(np.linalg.norm(cayley(s * k0) - expm(s * k0)) / s**3)
    check(
        "|U(sk) - expm(sk)| / s^3 constant (O(k^3) error)", errs[1], errs[0], tol=2e-2
    )
    s = 1e-2
    rev = np.linalg.solve(eye + 0.5 * s * k0, eye - 0.5 * s * k0)
    check(
        "reversed Cayley == expm(-k) to O(k^3)",
        np.linalg.norm(rev - expm(-s * k0)) / s**3,
        errs[0],
        tol=2e-2,
    )
    check_differs(
        "reversed Cayley as expm(+k)",
        np.linalg.norm(rev - expm(s * k0)),
        np.linalg.norm(cayley(s * k0) - expm(s * k0)),
        min_rel=10.0,
    )


# ---------------------------------------------------------------- claim 7
def claim7():
    print("claim 7: CABS from the Schur complement")
    ngrid, nobs, naux = 40, 5, 6
    chi = RNG.standard_normal((ngrid, nobs + naux))  # functions sampled on a grid
    s = chi.T @ chi
    o, a = slice(0, nobs), slice(nobs, nobs + naux)
    s_oo, s_oa, s_aa = s[o, o], s[o, a], s[a, a]
    ls12 = np.linalg.solve(s_oo, s_oa)
    q = s_aa - s_oa.T @ ls12
    w, vecs = np.linalg.eigh(q)
    keep = w > 1e-8
    c2 = vecs[:, keep] / np.sqrt(w[keep])
    coeffs = np.vstack([-ls12 @ c2, c2])
    check_arrays("C^T S_RR C == I", coeffs.T @ s @ coeffs, np.eye(c2.shape[1]))
    check_arrays("S_O,RIBS C == 0", s[o, :] @ coeffs, np.zeros((nobs, c2.shape[1])))

    q_full = s - s[:, o] @ np.linalg.solve(s_oo, s[o, :])
    wf, vf = np.linalg.eigh(q_full)
    kf = wf > 1e-8
    naive = vf[:, kf] / np.sqrt(wf[kf])
    check_differs(
        "unprojected eigenvectors orthogonal to OBS",
        np.max(np.abs(s[o, :] @ naive)) + 1.0,
        1.0,
    )
    check_differs(
        "unprojected eigenvectors orthonormal in S_RR",
        np.max(np.abs(naive.T @ s @ naive - np.eye(naive.shape[1]))) + 1.0,
        1.0,
    )

    # The union as build_cabs builds it: atom-major, each atom's OBS functions
    # then its aux functions (union_obs_mask), so the two sets interleave.
    # Partition by the mask, build c in [O | a] order, scatter back.
    mask = np.array([True] * 3 + [False] * 2 + [True] * 2 + [False] * 4)
    chi_u = RNG.standard_normal((ngrid, mask.size))
    s_u = chi_u.T @ chi_u
    obs_idx, aux_idx = np.flatnonzero(mask), np.flatnonzero(~mask)
    ls12_u = np.linalg.solve(
        s_u[np.ix_(obs_idx, obs_idx)], s_u[np.ix_(obs_idx, aux_idx)]
    )
    q_u = s_u[np.ix_(aux_idx, aux_idx)] - s_u[np.ix_(obs_idx, aux_idx)].T @ ls12_u
    w_u, v_u = np.linalg.eigh(q_u)
    c2_u = v_u[:, w_u > 1e-8] / np.sqrt(w_u[w_u > 1e-8])
    coeffs_u = np.zeros((mask.size, c2_u.shape[1]))
    coeffs_u[obs_idx] = -ls12_u @ c2_u
    coeffs_u[aux_idx] = c2_u
    n_u = c2_u.shape[1]
    check_arrays(
        "interleaved C^T S_RR C == I", coeffs_u.T @ s_u @ coeffs_u, np.eye(n_u)
    )
    check_arrays(
        "interleaved S_O,R C == 0",
        s_u[obs_idx, :] @ coeffs_u,
        np.zeros((obs_idx.size, n_u)),
    )
    blocked = np.vstack([-ls12_u @ c2_u, c2_u])  # [O | a] rows read as union rows
    check_differs(
        "blocked rows used as interleaved rows orthogonal to OBS",
        np.max(np.abs(s_u[obs_idx, :] @ blocked)) + 1.0,
        1.0,
    )


# ---------------------------------------------------------------- claim 8
def claim8():
    print("claim 8: the density-fitting error is the residuals' self-interaction")
    # Functions live in R^n with the positive definite operator w as the inner
    # product, (f|g) = f^T w g. The "exact" products rho_ia are columns of rho,
    # the auxiliary functions chi_P columns of chi (naux < n, so the fit is
    # inexact). The three-index integrals and the metric use the same w.
    n, naux, npair = 30, 8, 6
    a = RNG.standard_normal((n, n))
    w = a @ a.T + n * np.eye(n)
    a2 = RNG.standard_normal((n, n))
    w_other = a2 @ a2.T + n * np.eye(n)  # a second operator, for the control
    chi = RNG.standard_normal((n, naux))
    rho = RNG.standard_normal((n, npair))

    def fitted(w_metric):
        three = chi.T @ w @ rho  # (Q|ia)
        v = chi.T @ w_metric @ chi  # (P|Q) in the metric operator
        l_inv = np.linalg.inv(np.linalg.cholesky(v))  # M = L^-1, M^T M = V^-1
        b = l_inv @ three  # B^P_ia
        return b.T @ b, three, v  # sum_P B^P_ia B^P_jb

    exact = rho.T @ w @ rho  # (ia|jb)
    approx, three, v = fitted(w)
    coef = np.linalg.solve(v, three)  # V^-1 (Q|ia)
    delta = rho - chi @ coef  # residual delta_ia
    check_arrays(
        "(ia|jb) - sum_P B B == (delta_ia|delta_jb)",
        exact - approx,
        delta.T @ w @ delta,
    )
    check_arrays("(chi_P|delta_ia) == 0", chi.T @ w @ delta, np.zeros((naux, npair)))
    check_arrays(
        "diagonal error >= 0", np.minimum(np.diag(exact - approx), 0.0), np.zeros(npair)
    )
    mixed, _, _ = fitted(w_other)  # metric operator != integral operator
    check_differs(
        "mixed-metric error as the residual self-interaction",
        np.max(np.abs(exact - mixed - delta.T @ w @ delta)) + 1.0,
        1.0,
    )


# ---------------------------------------------------------------- claim 8
def claim9():
    print("claim 9: chunked dressing equals the full dressing")
    nbf, no, naux = 7, 3, 40
    nv = nbf - no
    raw = random_b(naux, nbf)  # (Q|mu nu)
    c = RNG.standard_normal((nbf, nbf))
    m = np.tril(RNG.standard_normal((naux, naux)))  # M = L^-1 is lower triangular
    mo_full = np.einsum("Qmn,mi,na->Qia", raw, c[:, :no], c[:, no:]).reshape(naux, -1)
    b_full = m @ mo_full

    def dressed(width, p0=0, p1=naux, drop=None):
        b = np.zeros((p1 - p0, no * nv))
        for n, q0 in enumerate(range(0, naux, width)):
            q1 = min(q0 + width, naux)
            if n == drop:
                continue
            half = np.einsum("mi,Qmn->Qin", c[:, :no], raw[q0:q1])  # C_o^T (Q|mu nu)
            mo_blk = np.einsum("Qin,na->Qia", half, c[:, no:]).reshape(q1 - q0, -1)
            b += m[p0:p1, q0:q1] @ mo_blk  # beta = 1
        return b

    for width in (1, 7, 32, naux):
        check_arrays(
            f"chunked dressing == M (Q|ia), width {width}", dressed(width), b_full
        )
    check_arrays(
        "output band [9, 23) == rows 9..23 of B", dressed(7, 9, 23), b_full[9:23]
    )
    check_differs(
        "one chunk dropped", np.sum(dressed(7, drop=2) ** 2), np.sum(b_full**2)
    )


# ---------------------------------------------------------------- claim 9
def aux_band(naux, size, rank):
    """parallel.rs aux_band: ranks r < s get q + 1 rows, the others q."""
    q, s = divmod(naux, size)
    start = rank * q + min(rank, s)
    return start, start + (q + 1 if rank < s else q)


def claim10():
    print("claim 10: auxiliary-band decomposition of G^(i) and the energy")
    no, nv, naux, nr = 4, 5, 23, 3
    b = RNG.standard_normal((naux, no * nv))
    eo, ev = orbital_energies(no, nv)
    d = denom(eo, ev)
    bands = [aux_band(naux, nr, r) for r in range(nr)]
    covered = sorted(p for p0, p1 in bands for p in range(p0, p1))
    check("bands partition [0, naux)", float(covered == list(range(naux))), 1.0)

    def g_part(i, p0, p1):  # G^(i,r), full width over j
        return b[p0:p1, i * nv : (i + 1) * nv].T @ b[p0:p1]

    e_os = e_ss = e_sq = 0.0
    for i in range(no):
        g_i = sum(g_part(i, p0, p1) for p0, p1 in bands)
        check_arrays(
            f"band partials sum to G^(i), i = {i}",
            g_i,
            b[:, i * nv : (i + 1) * nv].T @ b,
        )
        for j in range(no):
            g_ab = g_i[:, j * nv : (j + 1) * nv]  # (ia|jb)
            e_os += np.sum(g_ab**2 / d[i, j])
            e_ss += np.sum(g_ab * (g_ab - g_ab.T) / d[i, j])
            for p0, p1 in bands:  # control: square each band's partial
                e_sq += np.sum(
                    g_part(i, p0, p1)[:, j * nv : (j + 1) * nv] ** 2 / d[i, j]
                )
    g = np.einsum("Pia,Pjb->ijab", b.reshape(naux, no, nv), b.reshape(naux, no, nv))
    check("owner energy E_OS == serial E_OS", e_os, np.sum(g**2 / d))
    check(
        "owner energy E_SS == serial E_SS",
        e_ss,
        np.sum(g * (g - g.transpose(0, 1, 3, 2)) / d),
    )
    check_differs("band partials squared before the sum", e_sq, e_os)


# ---------------------------------------------------------------- claims 10, 11
U64, U32 = 2.0**-53, 2.0**-24


def gamma(n, u):
    return n * u / (1.0 - n * u)


def exact_dot(x, y):
    return float(sum(Fraction(a) * Fraction(c) for a, c in zip(x, y)))


def bound_check(name, err, bound):
    ok = err <= bound
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max err/bound {err / bound:.2e}")
    if not ok:
        FAILED.append(name)


def claim11():
    print("claim 11: summation-order bound of the f64 device GEMM")
    naux = 912
    gam = gamma(naux, U64)
    check("2 gamma_912(u64) == 2.0e-13", round(2 * gam / 1e-13, 1), 2.0)
    worst_side = worst_pair = worst_f32 = 0.0
    for _ in range(8):
        x, y = RNG.standard_normal(naux), RNG.standard_normal(naux)
        s_ab = float(np.sum(np.abs(x * y)))
        ref = exact_dot(x, y)
        fwd = 0.0
        for a, c in zip(x, y):  # one order: sequential
            fwd += a * c
        rev = float(np.sum((x * y)[::-1]))  # another: numpy's pairwise sum, reversed
        f32 = float(
            np.cumsum(x.astype(np.float32) * y.astype(np.float32), dtype=np.float32)[-1]
        )
        worst_side = max(worst_side, abs(fwd - ref) / s_ab, abs(rev - ref) / s_ab)
        worst_pair = max(worst_pair, abs(fwd - rev) / s_ab)
        worst_f32 = max(worst_f32, abs(f32 - fwd) / s_ab)
    bound_check("each f64 order within gamma_N(u64) S", worst_side, gam)
    bound_check("two f64 orders within 2 gamma_N(u64) S", worst_pair, 2 * gam)
    check_differs(
        "an f32 sum inside the f64 bound", max(worst_f32, 2 * gam), 2 * gam, 1.0
    )


def eps_g(b, naux):
    npan = -(-naux // b)
    return (
        (1 + U32) ** 2 * (1 + gamma(b, U32)) * (1 + gamma(npan, U64))
        - 1
        + gamma(naux, U64)
    )


def mixed_dot(x, y, b, drop=None):
    """f32 operands, f32 sum inside each panel of b rows, f64 sum over panels."""
    xf, yf = x.astype(np.float32), y.astype(np.float32)
    total = 0.0
    for n, p0 in enumerate(range(0, len(x), b)):
        if n == drop:
            continue
        total += float(
            np.cumsum(xf[p0 : p0 + b] * yf[p0 : p0 + b], dtype=np.float32)[-1]
        )
    return total


def claim12():
    print("claim 12: mixed-precision element bound")
    b, naux = 64, 912
    eps = eps_g(b, naux)
    check("eps_G(64, 912) == 3.93e-6", round(eps / 1e-6, 2), 3.93)
    check("eps_G(64, 84) == 3.93e-6", round(eps_g(b, 84) / 1e-6, 2), 3.93)
    check("gamma_64(u32) / eps_G == 0.97", round(gamma(b, U32) / eps, 2), 0.97)
    check("gamma_912(u32) == 5.4e-5", round(gamma(naux, U32) / 1e-5, 1), 5.4)
    check(
        "sqrt(f32_max / 64) == 2.3e18",
        round(np.sqrt(np.finfo(np.float32).max / 64) / 1e18, 1),
        2.3,
    )
    sets = []
    for _ in range(6):
        x, y = RNG.standard_normal(naux), RNG.standard_normal(naux)
        sets += [(x, y), (np.abs(x), np.abs(y))]
    worst = worst_drop = 0.0
    for x, y in sets:
        s_ab = float(np.sum(np.abs(x * y)))
        ref = exact_dot(x, y)
        worst = max(worst, abs(mixed_dot(x, y, b) - ref) / s_ab)
        worst_drop = max(worst_drop, abs(mixed_dot(x, y, b, drop=3) - ref) / s_ab)
    bound_check("mixed bound, random data", worst, eps)
    # Each panel: 1, then 63 products of 2^-25, a quarter of an f32 ulp of 1.
    x = np.tile(np.r_[1.0, np.full(b - 1, 2.0**-12)], naux // b)
    y = np.tile(np.r_[1.0, np.full(b - 1, 2.0**-13)], naux // b)
    err = abs(mixed_dot(x, y, b) - exact_dot(x, y)) / float(np.sum(x * y))
    check("sub-ulp panel err/S == 1.9e-6", round(err / 1e-6, 1), 1.9)
    bound_check("mixed bound, sub-ulp panel", err, eps)
    check_differs("a dropped panel inside the bound", max(worst_drop, eps), eps, 1.0)
    p = [2.0**20, 2.0**-9, -(2.0**20)]
    check("f64 sum over panels +2^20, 2^-9, -2^20 == 2^-9", sum(p) / 2.0**-9, 1.0)
    f32 = np.float32(0.0)
    for v in p:
        f32 = np.float32(f32 + np.float32(v))
    check("f32 sum over the same panels == 0", float(f32), 0.0)


def main():
    for claim in (
        claim1,
        claim2,
        claim3,
        claim4,
        claim5,
        claim6,
        claim7,
        claim8,
        claim9,
        claim10,
        claim11,
        claim12,
    ):
        claim()
    if FAILED:
        print(f"FAILED: {FAILED}")
        sys.exit(1)
    print("all claims hold")


if __name__ == "__main__":
    main()
