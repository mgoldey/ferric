#!/usr/bin/env python3
"""Numpy checks of the algebraic identities ferric-mp2's MP2 variants rely on.

Theory and notation: crates/ferric-mp2/theory.md. Every claim builds random
inputs with the symmetry the code assumes (real orbitals, density-fitted
(pq|rs) = sum_P B^P_pq B^P_rs), checks the identity to 1e-10 relative, and
carries a negative control: a plausible wrong variant that must NOT satisfy
it. Exits nonzero on the first failure.

  1  Closed-shell spin components (rimp2.rs spin_components_from_g :960,
     pair_energy :1291, j >= i weighting :1417): the spatial
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
     E[Coulomb]. Control: dropping X fails.
  5  Z-vector matvec (zvector.rs compute_az_product :460 / :502, used at
     :159 with factor 1/2): the vo block of (4J - K - K^T)[D^z] with
     D^z = sum_ai z_ai (C_a C_i^T + C_i C_a^T) equals
     2 sum_bj [4(ai|bj) - (ab|ij) - (aj|bi)] z_bj, i.e. twice the coupling
     of the (A+B) Hessian that cpks_polar.rs build_apb_amb builds. Control: the unhalved product is not (A+B)z.
  6  Cayley rotation (orbital_rotation.rs cayley_rotation :53):
     U = (I - k/2)^-1 (I + k/2) is orthogonal and U - expm(k) = O(|k|^3).
     Control: (I + k/2)^-1 (I - k/2) matches expm(-k), not expm(k).
  7  CABS (f12.rs build_cabs :99): coefficients [-S_OO^-1 S_Oa v; v]/sqrt(w)
     from the Schur complement S_aa - S_aO S_OO^-1 S_Oa = V w V^T are
     orthonormal in S_RR and orthogonal to OBS. Control: the eigenvectors
     of the full-RIBS complement S_RR - S_RO S_OO^-1 S_OR scaled by
     w^-1/2, without the OBS projection, are neither.

Run: python3 scripts/mp2_variant_identities.py  (needs numpy, scipy)
"""

import sys

import numpy as np
from scipy.linalg import expm

RNG = np.random.default_rng(20261105)
TOL = 1e-10
FAILED = []


def check(name, got, want, tol=TOL):
    err = abs(got - want) / max(1.0, abs(want))
    ok = err <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: got {got:.15e} want {want:.15e} rel {err:.1e}")
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
    print(f"  {'ok  ' if ok else 'FAIL'} control {name}: rel diff {err:.1e} (must be >= {min_rel:.0e})")
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
    return eo[:, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :]


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
    dso = e_so[occ][:, None, None, None] + e_so[occ][None, :, None, None] - e_so[vir][None, None, :, None] - e_so[vir][None, None, None, :]
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
                tot_ss += fac * np.sum(giajb[i, j] * (giajb[i, j] - gibja[i, j]) / d[i, j])
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
        d = e1o[:, None, None, None] + e2o[None, :, None, None] - e1v[None, None, :, None] - e2v[None, None, None, :]
        return np.sum(g**2 / d)

    e_aa = same_spin(ba, ea_o, ea_v)
    e_ab = opposite_spin(ba, ea_o, ea_v, bb, eb_o, eb_v)

    # Spin-orbital reference with separate alpha/beta MO sets.
    def so_parts():
        orbs = [(ca[:, p], 0, e) for p, e in zip(range(nbf), np.concatenate([ea_o, ea_v]))]
        orbs += [(cb[:, p], 1, e) for p, e in zip(range(nbf), np.concatenate([eb_o, eb_v]))]
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
        d = eo[:, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :]
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
    check("restricted limit: e_aa + e_bb == E_SS", 2.0 * same_spin(ba, ea_o, ea_v), e_ss)
    check("restricted limit: e_ab == E_OS", opposite_spin(ba, ea_o, ea_v, ba, ea_o, ea_v), e_os)


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

    x = np.sum((4.0 * gs * gl - gs * gl.transpose(0, 1, 3, 2) - gl * gs.transpose(0, 1, 3, 2)) / d)
    check("E[s+l] == E[s] + E[l] + X", e(gs + gl), e(gs) + e(gl) + x)
    check_differs("E[s] + E[l] as E[s+l] (X = 0)", e(gs) + e(gl), e(gs + gl))


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
    az = (cv.T @ (4.0 * jz - kz - kz.T) @ co)  # vo block, compute_az_product

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
    check("|U(sk) - expm(sk)| / s^3 constant (O(k^3) error)", errs[1], errs[0], tol=2e-2)
    s = 1e-2
    rev = np.linalg.solve(eye + 0.5 * s * k0, eye - 0.5 * s * k0)
    check("reversed Cayley == expm(-k) to O(k^3)", np.linalg.norm(rev - expm(-s * k0)) / s**3, errs[0], tol=2e-2)
    check_differs("reversed Cayley as expm(+k)", np.linalg.norm(rev - expm(s * k0)), np.linalg.norm(cayley(s * k0) - expm(s * k0)), min_rel=10.0)


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
    check_differs("unprojected eigenvectors orthogonal to OBS", np.max(np.abs(s[o, :] @ naive)) + 1.0, 1.0)
    check_differs("unprojected eigenvectors orthonormal in S_RR", np.max(np.abs(naive.T @ s @ naive - np.eye(naive.shape[1]))) + 1.0, 1.0)


def main():
    for claim in (claim1, claim2, claim3, claim4, claim5, claim6, claim7):
        claim()
    if FAILED:
        print(f"FAILED: {FAILED}")
        sys.exit(1)
    print("all claims hold")


if __name__ == "__main__":
    main()
