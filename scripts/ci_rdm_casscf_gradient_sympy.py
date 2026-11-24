#!/usr/bin/env python3
"""RDM energy, direct-sigma split, the CASSCF orbital gradient and Hessians, by brute force.

Theory: `book/src/17-multireference/03-direct-sigma-davidson.md#direct-sigma-davidson-split`, `book/src/17-multireference/04-rdms.md#rdms-1-2-rdm`, `book/src/17-multireference/05-casscf.md#casscf`, `book/src/17-multireference/05-casscf.md#casscf-second-order-check`. Every check asserts; the script exits nonzero
on any failure and prints "FAIL [label]".

Model. Six orthonormal molecular orbitals (one core, three active, two
virtual), a random symmetric h and a random real 8-fold-symmetric
(pq|rs) = sum_P B^P_pq B^P_rs. The active space is handled with explicit
Jordan-Wigner matrices on its 2 n_a = 6 spin orbitals (alpha before beta, as
ferric-ci orders them), and every CI vector is a random normalized vector in a
fixed (N_alpha, N_beta) sector, so the RDMs are N-representable but no
stationarity is assumed.

Checks (labels in brackets are printed and cited by the book):
  [rdm-energy]      E_core + sum h_pq gamma_pq + 1/2 sum (pq|rs) Gamma_pqrs
                    equals <C|H|C> + E_core, with gamma and Gamma defined as
                    the spin-summed expectation values of the book's
                    `book/src/17-multireference/04-rdms.md#rdms-1-2-rdm` (PySCF
                    make_rdm12 order), for the (2,2) and (2,1) sectors.
  [rdm-identities]  Gamma_pqrs = <E_pq E_rs> - delta_qr gamma_ps,
                    tr gamma = N, sum Gamma_pprr = N(N-1),
                    sum_r Gamma_pqrr = (N-1) gamma_pq, Gamma_pqrs = Gamma_rspq
                    = Gamma_qpsr.
  [rdm-two-config]  for C = c1|0a 0b> + c2|1a 1b>: gamma = diag(2c1^2, 2c2^2),
                    Gamma_0000 = 2c1^2, Gamma_1111 = 2c2^2,
                    Gamma_0101 = Gamma_1010 = 2c1c2 and no other nonzero
                    element, and the energy 2c1^2 h00 + 2c2^2 h11
                    + c1^2 (00|00) + c2^2 (11|11) + 2c1c2 (01|01) equals
                    <C|H|C> (book `04-rdms.md`, "Two limits").
  [sigma-split]     H - E_core = sum g_pq E_pq
                    + 1/2 sum_sigma (pq|rs) E^s_pq E^s_rs
                    + sum (pq|rs) E^a_pq E^b_rs, g = h - 1/2 sum_r (pr|rq),
                    as 64 x 64 matrices.
  [casscf-gradient] with C(kappa) = exp(K), K_pq = kappa = -K_qp (p > q), the
                    central difference of the CASSCF energy at fixed RDMs,
                    E_core(C) + sum (C_a^T F^I C_a)_tu gamma_tu
                    + 1/2 sum (tu|vw)_C Gamma_tuvw, equals 2 (F_pq - F_qp) for
                    every pair p > q, with the generalized Fock matrix of
                    casscf.rs::generalized_fock: F_pi = 2 (F^I + F^A)_pi,
                    F_pt = sum_u F^I_pu gamma_ut + sum_uvw (pu|vw) Gamma_tuvw,
                    F_pa = 0. The non-redundant blocks reduce to
                    4 (F^I + F^A)_ai and 2 F_at.
  [redundant-residual] for active t, u: 2 (F_tu - F_ut) = -2 c^T (E_tu - E_ut) r,
                    r = H c - <c|H|c> c, so the active-active gradient vanishes
                    at a CI eigenvector.
  [ah-frame-derivative] for C(e) = C exp(e X), the book's frame derivatives
                    dF^I, dF^A, d(pu|vw) give H_f x = pack 2(dF - dF^T), equal
                    to the central difference of pack 2(F - F^T)(C(e)).
  [ah-hessian]      H x = H_f x - 1/2 pack [G, X], G = 2(F - F^T)
                    (casscf_newton.rs), is symmetric and equals the mixed
                    second difference of E(C exp K) at fixed RDMs, for both
                    y.Hx and x.Hy. The commutator term is antisymmetric, so a
                    directional second derivative cannot see it; the mixed
                    one can.
  [relaxed-hessian] at the lowest CI root c, H_rel x = H x + H_oc z(x),
                    (H - theta) z = -P dH(x) c, z orthogonal to c, and
                    H_oc z = pack g(gamma_T, Gamma_T) (casscf_stability.rs),
                    equals the second difference of E_rel(kappa), the
                    lowest CAS-CI energy re-solved in the rotated orbitals;
                    x.H_oc z(x) <= 0.
  [negative-controls] wrong variants must fail.

Mutations (each turns the script red with the marker given):
  - F_pi = 2 (F^I + F^A)_pi with the factor 2 replaced by 1
    (marker "FAIL [casscf-gradient]");
  - the exchange term of F^A dropped (marker "FAIL [casscf-gradient]");
  - Gamma built with a+_p a+_r a_q a_s instead of a+_p a+_r a_s a_q
    (marker "FAIL [rdm-energy]");
  - the pair term 2 c1 c2 (01|01) of the two-configuration energy with
    factor 1 (marker "FAIL [rdm-two-config]");
  - g = h - 1/2 sum_r (pr|rq) with the 1/2 replaced by 1
    (marker "FAIL [sigma-split]").
  - the commutator correction of the AH Hessian dropped
    (marker "FAIL [ah-hessian]");
  - the J/K response term of dF^A dropped (marker "FAIL [ah-frame-derivative]");
  - the CI response H_oc z(x) dropped from H_rel (marker "FAIL [relaxed-hessian]").
"""

import itertools
import sys

import numpy as np
import sympy  # noqa: F401  (the runner requires sympy; this proof is numerical)

FAILED = []


def check(ok, label, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} [{label}] {detail}")
    if not ok:
        FAILED.append(label)


rng = np.random.default_rng(1115)
NC, NA, NVIR = 1, 3, 2
NMO = NC + NA + NVIR
ACT = slice(NC, NC + NA)
B = rng.normal(scale=0.3, size=(9, NMO, NMO))
B = 0.5 * (B + B.transpose(0, 2, 1))
G = np.einsum("Ppq,Prs->pqrs", B, B)
X = rng.normal(size=(NMO, NMO))
H = 0.5 * (X + X.T) + np.diag(np.arange(NMO) * 0.7 - 2.0)
E_NUC = 1.234

# ------------------------------------------------ JW matrices, active space
NSO = 2 * NA  # mode p = p alpha, mode NA + p = p beta
DIM = 1 << NSO


def annihilator(p):
    m = np.zeros((DIM, DIM))
    for s in range(DIM):
        if s >> p & 1:
            m[s ^ (1 << p), s] = -1.0 if bin(s & ((1 << p) - 1)).count("1") % 2 else 1.0
    return m


AN = [annihilator(p) for p in range(NSO)]
CR = [a.T for a in AN]


def mode(p, s):
    return p + NA * s


def e_spin(p, q, s):
    return CR[mode(p, s)] @ AN[mode(q, s)]


E1 = np.array(
    [[e_spin(p, q, 0) + e_spin(p, q, 1) for q in range(NA)] for p in range(NA)]
)


def h_active(h, g):
    m = np.zeros((DIM, DIM))
    for p, q in itertools.product(range(NA), repeat=2):
        m += h[p, q] * E1[p, q]
    for p, q, r, s in itertools.product(range(NA), repeat=4):
        for a, b in itertools.product(range(2), repeat=2):
            m += (
                0.5
                * g[p, q, r, s]
                * (CR[mode(p, a)] @ CR[mode(r, b)] @ AN[mode(s, b)] @ AN[mode(q, a)])
            )
    return m


def sector(na, nb):
    idx = []
    for s in range(DIM):
        if bin(s & ((1 << NA) - 1)).count("1") == na and bin(s >> NA).count("1") == nb:
            idx.append(s)
    return idx


def random_ci(na, nb):
    c = np.zeros(DIM)
    idx = sector(na, nb)
    c[idx] = rng.normal(size=len(idx))
    return c / np.linalg.norm(c)


def rdms(c, swap=False):
    return rdms_pair(c, c, swap)


def rdms_pair(bra, ket, swap=False):
    """<bra|E_pq|ket> and <bra|a+_p a+_r a_s a_q|ket>, spin-summed."""
    gam = np.einsum("i,pqij,j->pq", bra, E1, ket)
    n = NA
    big = np.zeros((n, n, n, n))
    for p, q, r, s in itertools.product(range(n), repeat=4):
        for a, b in itertools.product(range(2), repeat=2):
            last = (
                (AN[mode(q, a)] @ AN[mode(s, b)])
                if swap
                else (AN[mode(s, b)] @ AN[mode(q, a)])
            )
            big[p, q, r, s] += bra @ CR[mode(p, a)] @ CR[mode(r, b)] @ last @ ket
    return gam, big


# ------------------------------------------------------- CASSCF energy model
def jk(d, g):
    return np.einsum("mnls,ls->mn", g, d), np.einsum("mlns,ls->mn", g, d)


def transform(c):
    return c.T @ H @ c, np.einsum(
        "pqrs,pi,qj,rk,sl->ijkl", G, c, c, c, c, optimize=True
    )


def inactive(c):
    """F^I (MO basis of c) and E_core, from the AO-basis h and (pq|rs)."""
    cc = c[:, :NC]
    dc = 2.0 * cc @ cc.T
    j, k = jk(dc, G)
    fi_ao = H + j - 0.5 * k
    e_core = E_NUC + 0.5 * np.sum(dc * (H + fi_ao))
    return c.T @ fi_ao @ c, e_core


def energy_at_fixed_rdms(c, gam, big):
    fi, e_core = inactive(c)
    _, g = transform(c)
    return (
        e_core
        + np.sum(fi[ACT, ACT] * gam)
        + 0.5 * np.einsum("tuvw,tuvw->", g[ACT, ACT, ACT, ACT], big)
    )


def generalized_fock(c, gam, big, core_factor=2.0, k_active=0.5):
    """casscf.rs::CasScfProblem::generalized_fock, with F^A from active_fock."""
    fi, _ = inactive(c)
    ca = c[:, ACT]
    da = ca @ gam @ ca.T
    j, k = jk(da, G)
    fa = c.T @ (j - k_active * k) @ c
    _, g = transform(c)
    f = np.zeros((NMO, NMO))
    f[:, :NC] = core_factor * (fi[:, :NC] + fa[:, :NC])
    paaa = g[:, ACT, ACT, ACT].reshape(NMO, NA**3)
    f[:, ACT] = fi[:, ACT] @ gam + paaa @ big.reshape(NA, NA**3).T
    return f, fi, fa


def frame_derivative(c, gam, big, x):
    """d/de of F^I, F^A, F and (pq|rs) at C exp(e X), e = 0 (the book's frame derivatives)."""
    f, fi, fa = generalized_fock(c, gam, big)
    _, g = transform(c)
    cx = c @ x
    ddc = 2.0 * cx[:, :NC] @ c[:, :NC].T
    ddc = ddc + ddc.T
    dda = cx[:, ACT] @ gam @ c[:, ACT].T
    dda = dda + dda.T
    jc, kc = jk(ddc, G)
    ja, ka = jk(dda, G)
    dfi = fi @ x - x @ fi + c.T @ (jc - 0.5 * kc) @ c
    dfa = fa @ x - x @ fa + c.T @ (ja - 0.5 * ka) @ c
    dg = (
        np.einsum("rp,ruvw->puvw", x, g)
        + np.einsum("ru,prvw->puvw", x, g)
        + np.einsum("rv,purw->puvw", x, g)
        + np.einsum("rw,puvr->puvw", x, g)
    )
    df = np.zeros((NMO, NMO))
    df[:, :NC] = 2.0 * (dfi[:, :NC] + dfa[:, :NC])
    df[:, ACT] = (
        dfi[:, ACT] @ gam
        + dg[:, ACT, ACT, ACT].reshape(NMO, NA**3) @ big.reshape(NA, NA**3).T
    )
    return f, fi, df, dfi, dg


PAIRS = [(p, q) for p in range(NMO) for q in range(p)]


def pack(m):
    return np.array([m[p, q] for p, q in PAIRS])


def unpack(v):
    k = np.zeros((NMO, NMO))
    for (p, q), val in zip(PAIRS, v):
        k[p, q], k[q, p] = val, -val
    return k


def ah_hessian_product(c, gam, big, x, commutator=0.5):
    """casscf_newton.rs: H x = pack 2(dF - dF^T) - 1/2 pack [G, X], G = 2(F - F^T)."""
    xm = unpack(x)
    f, _, df, _, _ = frame_derivative(c, gam, big, xm)
    gm = 2.0 * (f - f.T)
    return pack(2.0 * (df - df.T)) - commutator * pack(gm @ xm - xm @ gm)


def expm_antisym(k):
    w, v = np.linalg.eigh(1j * k)  # i K is Hermitian
    return (v @ np.diag(np.exp(-1j * w)) @ v.conj().T).real


# -------------------------------------------------------------------- claims
def claim_rdm():
    print("\n=== RDM energy and identities")
    fi, e_core = inactive(np.eye(NMO))
    _, g = transform(np.eye(NMO))
    h_eff, g_act = fi[ACT, ACT], g[ACT, ACT, ACT, ACT]
    hm = h_active(h_eff, g_act)
    for na, nb in ((2, 2), (2, 1)):
        c = random_ci(na, nb)
        gam, big = rdms(c)
        e_rdm = e_core + np.sum(h_eff * gam) + 0.5 * np.sum(g_act * big)
        d = abs(e_rdm - (c @ hm @ c + e_core))
        check(d < 1e-12, "rdm-energy", f"({na},{nb}): |E_rdm - <C|H|C>| = {d:.1e}")
        n = na + nb
        ee = np.einsum("i,pqij,rsjk,k->pqrs", c, E1, E1, c)
        d = np.abs(big - (ee - np.einsum("qr,ps->pqrs", np.eye(NA), gam))).max()
        check(
            d < 1e-12,
            "rdm-identities",
            f"({na},{nb}) Gamma = <EE> - delta gamma: {d:.1e}",
        )
        ids = [
            abs(np.trace(gam) - n),
            abs(np.einsum("pprr->", big) - n * (n - 1)),
            np.abs(np.einsum("pqrr->pq", big) - (n - 1) * gam).max(),
            np.abs(big - big.transpose(2, 3, 0, 1)).max(),
            np.abs(big - big.transpose(1, 0, 3, 2)).max(),
        ]
        check(
            max(ids) < 1e-12,
            "rdm-identities",
            f"({na},{nb}) traces and symmetries: {max(ids):.1e}",
        )
    return h_eff, g_act, hm


def claim_two_config(h_eff, g_act, hm):
    print("\n=== the (2,2) wavefunction c1|0a 0b> + c2|1a 1b>")
    vac = np.zeros(DIM)
    vac[0] = 1.0
    d0 = CR[mode(0, 0)] @ CR[mode(0, 1)] @ vac
    d1 = CR[mode(1, 0)] @ CR[mode(1, 1)] @ vac
    c1, c2 = 0.8, -0.6
    c = c1 * d0 + c2 * d1
    gam, big = rdms(c)
    want_g = np.zeros((NA, NA))
    want_g[0, 0], want_g[1, 1] = 2 * c1**2, 2 * c2**2
    want_b = np.zeros((NA,) * 4)
    want_b[0, 0, 0, 0], want_b[1, 1, 1, 1] = 2 * c1**2, 2 * c2**2
    want_b[0, 1, 0, 1] = want_b[1, 0, 1, 0] = 2 * c1 * c2
    d = max(np.abs(gam - want_g).max(), np.abs(big - want_b).max())
    check(d < 1e-14, "rdm-two-config", f"gamma and every Gamma element: {d:.1e}")
    e_formula = (
        2 * c1**2 * h_eff[0, 0]
        + 2 * c2**2 * h_eff[1, 1]
        + c1**2 * g_act[0, 0, 0, 0]
        + c2**2 * g_act[1, 1, 1, 1]
        + 2.0 * c1 * c2 * g_act[0, 1, 0, 1]
    )
    d = abs(e_formula - c @ hm @ c)
    check(d < 1e-12, "rdm-two-config", f"E formula vs <C|H|C>: {d:.1e}")


def claim_sigma(h_eff, g_act, hm, half=0.5):
    print("\n=== direct-sigma split of H")
    gg = h_eff - half * np.einsum("prrq->pq", g_act)
    m = np.zeros((DIM, DIM))
    for p, q in itertools.product(range(NA), repeat=2):
        m += gg[p, q] * E1[p, q]
    for p, q, r, s in itertools.product(range(NA), repeat=4):
        for sg in range(2):
            m += 0.5 * g_act[p, q, r, s] * (e_spin(p, q, sg) @ e_spin(r, s, sg))
        m += g_act[p, q, r, s] * (e_spin(p, q, 0) @ e_spin(r, s, 1))
    d = np.abs(m - hm).max()
    check(d < 1e-12, "sigma-split", f"max |H_split - H| = {d:.1e}")


def claim_gradient(quiet=False, **kw):
    if not quiet:
        print("\n=== CASSCF gradient 2 (F - F^T) vs a central difference")
    c0 = np.eye(NMO)
    worst, worst_nonred = 0.0, 0.0
    for na, nb in ((2, 2), (2, 1)):
        gam, big = rdms(random_ci(na, nb))
        f, fi, fa = generalized_fock(c0, gam, big, **kw)
        gmat = 2.0 * (f - f.T)
        h = 1e-4
        for p in range(NMO):
            for q in range(p):
                vals = []
                for s in (-2, -1, 1, 2):
                    k = np.zeros((NMO, NMO))
                    k[p, q], k[q, p] = s * h, -s * h
                    vals.append(energy_at_fixed_rdms(c0 @ expm_antisym(k), gam, big))
                fd = (vals[0] - 8 * vals[1] + 8 * vals[2] - vals[3]) / (12 * h)
                worst = max(worst, abs(fd - gmat[p, q]))
        ft = fi + fa
        nonred = [
            abs(gmat[a, i] - 4.0 * ft[a, i])
            for a in range(NC + NA, NMO)
            for i in range(NC)
        ]
        nonred += [
            abs(gmat[a, t] - 2.0 * f[a, t])
            for a in range(NC + NA, NMO)
            for t in range(NC, NC + NA)
        ]
        worst_nonred = max(worst_nonred, max(nonred))
    if quiet:
        return worst
    check(
        worst < 1e-8,
        "casscf-gradient",
        f"max |FD - 2(F - F^T)| over all pairs = {worst:.1e}",
    )
    check(
        worst_nonred < 1e-12,
        "casscf-gradient",
        f"(a,i) = 4(F^I+F^A)_ai, (a,t) = 2F_at: {worst_nonred:.1e}",
    )
    return worst


def claim_redundant(h_eff, g_act, hm):
    print("\n=== active-active gradient and the CI residual")
    c = random_ci(2, 2)
    gam, big = rdms(c)
    f, _, _ = generalized_fock(np.eye(NMO), gam, big)
    hc = hm @ c  # E_core cancels in the residual
    r = hc - (c @ hc) * c
    worst = 0.0
    for t, u in itertools.product(range(NA), repeat=2):
        lhs = 2.0 * (f[NC + t, NC + u] - f[NC + u, NC + t])
        rhs = -2.0 * c @ (E1[t, u] - E1[u, t]) @ r
        worst = max(worst, abs(lhs - rhs))
    check(
        worst < 1e-12,
        "redundant-residual",
        f"max |2(F_tu - F_ut) + 2 c(E_tu - E_ut) r| = {worst:.1e}",
    )


def claim_ah_hessian():
    print("\n=== AH orbital Hessian at fixed RDMs vs finite differences")
    c0 = np.eye(NMO)
    gam, big = rdms(random_ci(2, 2))
    npair = len(PAIRS)
    hmat = np.column_stack([ah_hessian_product(c0, gam, big, e) for e in np.eye(npair)])
    # frame derivative: H_f x = d/de pack 2(F - F^T) at C exp(e X)
    worst_f = 0.0
    for _ in range(3):
        x = rng.normal(size=npair)
        _, _, df, _, _ = frame_derivative(c0, gam, big, unpack(x))
        h = 1e-4
        vals = []
        for s in (-2, -1, 1, 2):
            f, _, _ = generalized_fock(c0 @ expm_antisym(s * h * unpack(x)), gam, big)
            vals.append(pack(2.0 * (f - f.T)))
        fd = (vals[0] - 8 * vals[1] + 8 * vals[2] - vals[3]) / (12 * h)
        worst_f = max(worst_f, np.abs(fd - pack(2.0 * (df - df.T))).max())
    check(worst_f < 1e-8, "ah-frame-derivative", f"max |FD - H_f x| = {worst_f:.1e}")

    # Hessian of E(C exp K): the mixed second difference, Richardson-extrapolated
    def mixed(x, y, h):
        e = 0.0
        for sx, sy, w in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
            e += w * energy_at_fixed_rdms(
                c0 @ expm_antisym(unpack(h * (sx * x + sy * y))), gam, big
            )
        return e / (4 * h * h)

    worst = 0.0
    for _ in range(4):
        x, y = rng.normal(size=npair), rng.normal(size=npair)
        fd = (4 * mixed(x, y, 5e-4) - mixed(x, y, 1e-3)) / 3
        worst = max(worst, abs(fd - y @ hmat @ x), abs(fd - x @ hmat @ y))
    asym = np.abs(hmat - hmat.T).max()
    check(
        worst < 1e-7 and asym < 1e-12,
        "ah-hessian",
        f"max |FD - y.Hx|, |FD - x.Hy| = {worst:.1e}; max |H - H^T| = {asym:.1e}",
    )


def claim_relaxed_hessian(ci_response=1.0):
    print("\n=== CI-relaxed orbital Hessian (Schur complement) vs E_rel(kappa)")
    c0 = np.eye(NMO)
    idx = sector(2, 2)

    def lowest_root(c):
        fi, e_core = inactive(c)
        _, g = transform(c)
        hm = h_active(fi[ACT, ACT], g[ACT, ACT, ACT, ACT])[np.ix_(idx, idx)]
        w, v = np.linalg.eigh(hm)
        return e_core + w[0], v[:, 0], hm

    _, v0, hsec = lowest_root(c0)
    theta = v0 @ hsec @ v0
    cvec = np.zeros(DIM)
    cvec[idx] = v0
    gam, big = rdms(cvec)
    npair = len(PAIRS)
    proj = np.eye(len(idx)) - np.outer(v0, v0)
    worst, most_positive = 0.0, -np.inf
    for _ in range(4):
        x = rng.normal(size=npair)
        xm = unpack(x)
        _, _, _, dfi, dg = frame_derivative(c0, gam, big, xm)
        dh = h_active(dfi[ACT, ACT], dg[ACT, ACT, ACT, ACT])[np.ix_(idx, idx)]
        rhs = -proj @ dh @ v0
        z = (
            proj
            @ np.linalg.lstsq(
                proj @ (hsec - theta * np.eye(len(idx))) @ proj, rhs, rcond=1e-12
            )[0]
        )
        zf = np.zeros(DIM)
        zf[idx] = z
        ga, ba = rdms_pair(zf, cvec)
        gt, bt = ga + ga.T, ba + ba.transpose(1, 0, 3, 2)
        f, fi, _ = generalized_fock(c0, gt, bt)
        f[:, :NC] -= 2.0 * fi[:, :NC]  # keep the part linear in the RDMs
        hoc_z = pack(2.0 * (f - f.T))
        hx = ah_hessian_product(c0, gam, big, x) + ci_response * hoc_z
        most_positive = max(most_positive, x @ hoc_z)
        h = 1e-3
        es = [lowest_root(c0 @ expm_antisym(s * h * xm))[0] for s in (-2, -1, 0, 1, 2)]
        fd = (-es[0] + 16 * es[1] - 30 * es[2] + 16 * es[3] - es[4]) / (12 * h * h)
        worst = max(worst, abs(fd - x @ hx))
    check(worst < 1e-6, "relaxed-hessian", f"max |d2 E_rel - x.H_rel x| = {worst:.1e}")
    check(
        most_positive <= 1e-12,
        "relaxed-hessian",
        f"x.H_oc z(x) <= 0 (largest {most_positive:.1e})",
    )


def claim_negative(h_eff, g_act, hm):
    print("\n=== negative controls (each must differ)")
    fi, e_core = inactive(np.eye(NMO))
    c = random_ci(2, 2)
    gam, big = rdms(c, swap=True)
    e_rdm = e_core + np.sum(h_eff * gam) + 0.5 * np.sum(g_act * big)
    check(
        abs(e_rdm - (c @ hm @ c + e_core)) > 1e-6,
        "negative-controls",
        "Gamma with a_q a_s order",
    )
    worst = claim_gradient(quiet=True, core_factor=1.0)
    check(
        worst > 1e-6,
        "negative-controls",
        f"F_pi with factor 1 misses the FD gradient by {worst:.1e}",
    )


if __name__ == "__main__":
    h_eff, g_act, hm = claim_rdm()
    claim_two_config(h_eff, g_act, hm)
    claim_sigma(h_eff, g_act, hm)
    claim_gradient()
    claim_redundant(h_eff, g_act, hm)
    claim_ah_hessian()
    claim_relaxed_hessian()
    claim_negative(h_eff, g_act, hm)
    if FAILED:
        print(f"\n{len(FAILED)} FAILED: {sorted(set(FAILED))}")
        sys.exit(1)
    print("\nall checks hold")
