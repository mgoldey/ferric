#!/usr/bin/env python3
"""CCD, CCSD and EOM-CCSD working equations against a brute-force e^{-T} H e^{T}.

Theory: crates/ferric-cc/theory.md, sections "CCD", "CCSD" and
"EE-EOM-CCSD singlets, closed shell". Every check asserts; the script exits
nonzero on any failure and prints "FAIL [label]".

Model. Five spatial orbitals (two occupied, three virtual), ten spin orbitals
interleaved as in ferric (2p = alpha, 2p + 1 = beta), a random real
8-fold-symmetric (pq|rs) = sum_P B^P_pq B^P_rs, and a one-electron h chosen so
that the Fock matrix of the reference is exactly diag(eps). The reference is
therefore a canonical Hartree-Fock stationary point (f_ia = 0, diagonal f_oo
and f_vv), which is the case the code assumes.

Independent route. The Hamiltonian H = sum h_PQ a+_P a_Q
+ 1/4 sum <PQ||RS> a+_P a+_Q a_S a_R is built as a Jordan-Wigner matrix on
the full 1024-state Fock space, T1 and T2 as matrices from the amplitudes,
and <mu| e^{-T} H e^{T} |0> by the terminating exponential series. No
diagram, intermediate or Wick contraction enters this route.

Checks (labels in brackets are printed and cited by the book):
  [ccd-residual]     at random antisymmetric t2 (t1 = 0), the eight-term CCD
                     residual of ccd.rs::ccd_spinorbital plus the Fock term
                     Delta * t equals <Phi_ij^ab| e^{-T2} H e^{T2} |0>, and
                     1/4 <ij||ab> t equals <0|e^{-T2}He^{T2}|0> - E_ref.
  [ccsd-residual]    at random t1 and t2, the Stanton-Gauss singles and
                     doubles residuals of ccsd.rs::ccsd (F_ae, F_mi, F_me,
                     W_mnij, W_abef, W_mbej, tau, tau-tilde) plus Delta * t
                     equal the brute-force projections.
  [ccsd-energy]      1/4 <ij||ab> t_ijab + 1/2 <ij||ab> t_ia t_jb equals
                     <0|e^{-T}He^{T}|0> - E_ref.
  [ccsd-ccd-limit]   the CCSD doubles residual at t1 = 0 is the CCD residual.
  [spin-adaptation]  on spin-orbital amplitudes expanded from closed-shell
                     spatial ones (expand_amplitudes_to_spin_orbital), the
                     brute-force residual is spin-adapted: alpha and beta
                     singles agree, the alpha-beta doubles block
                     R_ij^ab = R(i a, j b, a a, b b) obeys R_ij^ab = R_ji^ba,
                     the same-spin block is R_ij^ab - R_ij^ba, and the
                     closed-shell energy sum (2(ia|jb) - (ib|ja)) (t_ij^ab +
                     t_i^a t_j^b) equals the brute-force energy.
  [eom-jacobian]     at converged CCSD amplitudes, the Jacobian dR_mu/dt_nu
                     of the residual (5-point stencil, exact for a quartic)
                     equals <mu|e^{-T}He^{T}|nu> - E_CC delta_mu,nu on the
                     singles and doubles: EOM-CCSD is the CC Jacobian.
  [cis-limit]        at t = 0 the singlet singles block of H - E_ref is
                     delta_ij delta_ab (eps_a - eps_i) + 2(ia|jb) - (ij|ab).
  [spin-norm]        for a closed-shell (r1, r2) the norm of its spin-orbital
                     expansion over distinct determinants is
                     2|r1|^2 + |r2|^2 + 1/2 |r2 - r2^(ab)|^2 (spin_norm).
  [negative-controls] each deliberately wrong variant must fail its check.

Mutations (each turns the script red with the marker given):
  - the 0.5 of the CCD quadratic term 2 replaced by 1.0
    (marker "FAIL [ccd-residual]");
  - 0.5 t2 replaced by t2 in the W_mbej intermediate
    (marker "FAIL [ccsd-residual]");
  - the 0.5 of the t1 t1 energy term replaced by 0.25
    (marker "FAIL [ccsd-energy]");
  - the exchange integral dropped from the closed-shell energy sum
    (marker "FAIL [spin-adaptation]");
  - 2(ia|jb) replaced by (ia|jb) in the singlet CIS block
    (marker "FAIL [cis-limit]").
"""

import itertools
import sys

import numpy as np
import scipy.sparse as sps
import sympy  # noqa: F401  (the runner requires sympy; this proof is numerical)

FAILED = []


def check(ok, label, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} [{label}] {detail}")
    if not ok:
        FAILED.append(label)


# ------------------------------------------------------------------ the model
NO, NV = 2, 3
NSP = NO + NV
NO2, NV2 = 2 * NO, 2 * NV
NSO = 2 * NSP
OC = slice(0, NO2)
VI = slice(NO2, NSO)
rng = np.random.default_rng(20261115)

NAUX = 7
B = rng.normal(scale=0.18, size=(NAUX, NSP, NSP))
B = 0.5 * (B + B.transpose(0, 2, 1))
G = np.einsum("Ppq,Prs->pqrs", B, B)  # chemist (pq|rs), 8-fold symmetric
EPS = np.array([-1.1, -0.8, 0.35, 0.6, 0.9])
# h such that f_pq = h_pq + sum_k [2 (pq|kk) - (pk|kq)] = eps_p delta_pq.
HSP = np.diag(EPS) - (
    2.0 * np.einsum("pqkk->pq", G[:, :, :NO, :NO])
    - np.einsum("pkkq->pq", G[:, :NO, :NO, :])
)

spin = np.arange(NSO) % 2
spat = np.arange(NSO) // 2
same = spin[:, None] == spin[None, :]
H1 = HSP[np.ix_(spat, spat)] * same
# <PQ|RS> = (PR|QS) delta(sP,sR) delta(sQ,sS); <PQ||RS> = <PQ|RS> - <PQ|SR>.
PH = G[np.ix_(spat, spat, spat, spat)].transpose(0, 2, 1, 3)
PH = PH * same[:, None, :, None] * same[None, :, None, :]
VA = PH - PH.transpose(0, 1, 3, 2)
EO = EPS[spat[:NO2]]
EV = EPS[spat[NO2:]]
DIA = EO[:, None] - EV[None, :]
DIJAB = (
    EO[:, None, None, None]
    + EO[None, :, None, None]
    - EV[None, None, :, None]
    - EV[None, None, None, :]
)

# ----------------------------------------------------- Jordan-Wigner matrices
DIM = 1 << NSO


def annihilator(p):
    rows, cols, vals = [], [], []
    for s in range(DIM):
        if s >> p & 1:
            rows.append(s ^ (1 << p))
            cols.append(s)
            vals.append(-1.0 if bin(s & ((1 << p) - 1)).count("1") % 2 else 1.0)
    return sps.csr_matrix((vals, (rows, cols)), shape=(DIM, DIM))


A = [annihilator(p) for p in range(NSO)]
C = [a.T.tocsr() for a in A]
REF = np.zeros(DIM)
REF[(1 << NO2) - 1] = 1.0  # a+_0 a+_1 ... a+_{NO2-1} |vac>, coefficient +1

HAM = sps.csr_matrix((DIM, DIM))
for p, q in itertools.product(range(NSO), repeat=2):
    if abs(H1[p, q]) > 0:
        HAM = HAM + H1[p, q] * (C[p] @ A[q])
for p, q in itertools.combinations(range(NSO), 2):
    cc = C[p] @ C[q]
    for r, s in itertools.combinations(range(NSO), 2):
        if abs(VA[p, q, r, s]) > 0:
            HAM = HAM + VA[p, q, r, s] * (cc @ A[s] @ A[r])
E_REF = REF @ (HAM @ REF)


def t_operator(t1, t2):
    t = sps.csr_matrix((DIM, DIM))
    for i in range(NO2):
        for a in range(NV2):
            if t1[i, a] != 0:
                t = t + t1[i, a] * (C[NO2 + a] @ A[i])
    for i, j in itertools.combinations(range(NO2), 2):
        for a, b in itertools.combinations(range(NV2), 2):
            if t2[i, j, a, b] != 0:
                t = t + t2[i, j, a, b] * (C[NO2 + a] @ C[NO2 + b] @ A[j] @ A[i])
    return t


def exp_apply(t, v, sign=1.0):
    out, term = v.copy(), v.copy()
    for k in range(1, NO2 + 1):  # T raises excitation rank; NO2 terms suffice
        term = sign * (t @ term) / k
        out = out + term
    return out


SINGLES = [(i, a) for i in range(NO2) for a in range(NV2)]
DOUBLES = [
    (i, j, a, b)
    for i, j in itertools.combinations(range(NO2), 2)
    for a, b in itertools.combinations(range(NV2), 2)
]


def det_single(i, a):
    return C[NO2 + a] @ (A[i] @ REF)


def det_double(i, j, a, b):
    return C[NO2 + a] @ (C[NO2 + b] @ (A[j] @ (A[i] @ REF)))


PHI1 = np.array([det_single(*x) for x in SINGLES])
PHI2 = np.array([det_double(*x) for x in DOUBLES])


def brute(t1, t2):
    """<mu| e^{-T} H e^{T} |0> for mu = 0, singles, doubles (full arrays)."""
    t = t_operator(t1, t2)
    u = exp_apply(t, HAM @ exp_apply(t, REF), sign=-1.0)
    e = REF @ u - E_REF
    r1 = (PHI1 @ u).reshape(NO2, NV2)
    r2 = np.zeros((NO2, NO2, NV2, NV2))
    for (i, j, a, b), val in zip(DOUBLES, PHI2 @ u):
        r2[i, j, a, b], r2[j, i, a, b] = val, -val
        r2[i, j, b, a], r2[j, i, b, a] = -val, val
    return e, r1, r2


def antisym_t2(x):
    return (
        x - x.transpose(1, 0, 2, 3) - x.transpose(0, 1, 3, 2) + x.transpose(1, 0, 3, 2)
    )


def p_ij(x):
    return x - x.transpose(1, 0, 2, 3)


def p_ab(x):
    return x - x.transpose(0, 1, 3, 2)


# ------------------------------------------------- ferric's working equations
oovv, oooo, vvvv, ovvo = (
    VA[OC, OC, VI, VI],
    VA[OC, OC, OC, OC],
    VA[VI, VI, VI, VI],
    VA[OC, VI, VI, OC],
)
ooov, oovo, ovov, ovoo = (
    VA[OC, OC, OC, VI],
    VA[OC, OC, VI, OC],
    VA[OC, VI, OC, VI],
    VA[OC, VI, OC, OC],
)
ovvv, vovv, vvvo = VA[OC, VI, VI, VI], VA[VI, OC, VI, VI], VA[VI, VI, VI, OC]


def ccd_r(t):
    """ccd.rs::ccd_spinorbital residual R (without the Fock term)."""
    r = oovv.copy()
    r += 0.5 * np.einsum("ijcd,abcd->ijab", t, vvvv)
    r += 0.5 * np.einsum("klij,klab->ijab", oooo, t)
    x = np.einsum("kbcj,ikac->ijab", ovvo, t)
    r += x - x.transpose(1, 0, 2, 3) - x.transpose(0, 1, 3, 2) + x.transpose(1, 0, 3, 2)
    r += 0.25 * np.einsum("ijcd,klcd,klab->ijab", t, oovv, t)
    x = np.einsum("ikac,klcd,jlbd->ijab", t, oovv, t)
    r += 0.5 * (
        x - x.transpose(1, 0, 2, 3) - x.transpose(0, 1, 3, 2) + x.transpose(1, 0, 3, 2)
    )
    r -= 0.5 * p_ab(np.einsum("ijac,klcd,klbd->ijab", t, oovv, t))
    r -= 0.5 * p_ij(np.einsum("ikab,klcd,jlcd->ijab", t, oovv, t))
    return r


def ccsd_r(t1, t2, wmbej_t2=0.5):
    """ccsd.rs::ccsd singles and doubles residuals r1, r2 (without Fock terms)."""
    tt = np.einsum("ia,jb->ijab", t1, t1)
    taus = t2 + 0.5 * p_ij(tt)
    tau = t2 + p_ij(tt)
    fae = np.einsum("mf,mafe->ae", t1, ovvv) - 0.5 * np.einsum(
        "mnaf,mnef->ae", taus, oovv
    )
    fmi = np.einsum("ne,mnie->mi", t1, ooov) + 0.5 * np.einsum(
        "inef,mnef->mi", taus, oovv
    )
    fme = np.einsum("nf,mnef->me", t1, oovv)
    x = np.einsum("je,mnie->mnij", t1, ooov)
    wmnij = (
        oooo
        + x
        - x.transpose(0, 1, 3, 2)
        + 0.25 * np.einsum("ijef,mnef->mnij", tau, oovv)
    )
    wabef = (
        vvvv
        - np.einsum("mb,amef->abef", t1, vovv)
        + np.einsum("ma,bmef->abef", t1, vovv)
        + 0.25 * np.einsum("mnab,mnef->abef", tau, oovv)
    )
    xx = wmbej_t2 * t2 + np.einsum("jf,nb->jnfb", t1, t1)
    wmbej = (
        ovvo
        + np.einsum("jf,mbef->mbej", t1, ovvv)
        - np.einsum("nb,mnej->mbej", t1, oovo)
        - np.einsum("jnfb,mnef->mbej", xx, oovv)
    )
    r1 = (
        np.einsum("ie,ae->ia", t1, fae)
        - np.einsum("ma,mi->ia", t1, fmi)
        + np.einsum("imae,me->ia", t2, fme)
        - np.einsum("nf,naif->ia", t1, ovov)
        - 0.5 * np.einsum("imef,maef->ia", t2, ovvv)
        - 0.5 * np.einsum("mnae,nmei->ia", t2, oovo)
    )
    r2 = oovv.copy()
    r2 += p_ab(
        np.einsum("ijae,be->ijab", t2, fae - 0.5 * np.einsum("mb,me->be", t1, fme))
    )
    r2 -= p_ij(
        np.einsum("imab,mj->ijab", t2, fmi + 0.5 * np.einsum("je,me->mj", t1, fme))
    )
    r2 += 0.5 * np.einsum("mnab,mnij->ijab", tau, wmnij)
    r2 += 0.5 * np.einsum("ijef,abef->ijab", tau, wabef)
    x = np.einsum("imae,mbej->ijab", t2, wmbej) - np.einsum(
        "ie,ma,mbej->ijab", t1, t1, ovvo
    )
    r2 += (
        x - x.transpose(1, 0, 2, 3) - x.transpose(0, 1, 3, 2) + x.transpose(1, 0, 3, 2)
    )
    r2 += p_ij(np.einsum("ie,abej->ijab", t1, vvvo))
    r2 -= p_ab(np.einsum("ma,mbij->ijab", t1, ovoo))
    return r1, r2


def ccsd_energy(t1, t2, quarter_t1=0.5):
    return 0.25 * np.einsum("ijab,ijab->", oovv, t2) + quarter_t1 * np.einsum(
        "ijab,ia,jb->", oovv, t1, t1
    )


def full_residual(t1, t2, **kw):
    r1, r2 = ccsd_r(t1, t2, **kw)
    return r1 - DIA * t1, r2 - DIJAB * t2


# -------------------------------------------------------------------- claims
def random_t(scale=0.15):
    t1 = rng.normal(scale=scale, size=(NO2, NV2))
    t2 = antisym_t2(rng.normal(scale=scale, size=(NO2, NO2, NV2, NV2)))
    return t1, t2


def claim_ccd():
    print("\n=== CCD residual and energy vs brute force")
    _, t2 = random_t()
    z1 = np.zeros((NO2, NV2))
    e_b, r1_b, r2_b = brute(z1, t2)
    d = np.abs(ccd_r(t2) - DIJAB * t2 - r2_b).max()
    check(d < 1e-12, "ccd-residual", f"max |R_formula - R_brute| = {d:.1e}")
    d = abs(0.25 * np.einsum("ijab,ijab->", oovv, t2) - e_b)
    check(d < 1e-12, "ccd-residual", f"energy |dE| = {d:.1e}")
    d = np.abs(ccsd_r(z1, t2)[1] - ccd_r(t2)).max()
    check(d < 1e-12, "ccsd-ccd-limit", f"CCSD r2(t1=0) vs CCD r2: {d:.1e}")


def claim_ccsd():
    print("\n=== CCSD residuals and energy vs brute force")
    t1, t2 = random_t()
    e_b, r1_b, r2_b = brute(t1, t2)
    f1, f2 = full_residual(t1, t2)
    d1, d2 = np.abs(f1 - r1_b).max(), np.abs(f2 - r2_b).max()
    check(max(d1, d2) < 1e-12, "ccsd-residual", f"singles {d1:.1e}, doubles {d2:.1e}")
    d = abs(ccsd_energy(t1, t2) - e_b)
    check(d < 1e-12, "ccsd-energy", f"|dE| = {d:.1e}")
    return t1, t2


def expand(t1s, t2s):
    """ccsd_closed_shell.rs::expand_amplitudes_to_spin_orbital."""
    sv = spin[:NO2], spin[:NV2]
    so, sw = spat[:NO2], spat[:NV2]
    t1 = t1s[np.ix_(so, sw)] * (sv[0][:, None] == sv[1][None, :])
    di = sv[0][:, None, None, None]
    dj = sv[0][None, :, None, None]
    da = sv[1][None, None, :, None]
    db = sv[1][None, None, None, :]
    direct = t2s[np.ix_(so, so, sw, sw)] * ((di == da) & (dj == db))
    exch = t2s[np.ix_(so, so, sw, sw)].transpose(0, 1, 3, 2) * ((di == db) & (dj == da))
    return t1, direct - exch


def cs_energy(t1s, t2s, exchange=1.0):
    ovov_s = G[:NO, NO:, :NO, NO:]  # (ia|jb)
    tau = t2s + np.einsum("ia,jb->ijab", t1s, t1s)
    return 2.0 * np.einsum("iajb,ijab->", ovov_s, tau) - exchange * np.einsum(
        "ibja,ijab->", ovov_s, tau
    )


def claim_spin_adaptation():
    print("\n=== spin adaptation of the closed-shell residual and energy")
    t1s = rng.normal(scale=0.15, size=(NO, NV))
    x = rng.normal(scale=0.15, size=(NO, NO, NV, NV))
    t2s = x + x.transpose(1, 0, 3, 2)
    t1, t2 = expand(t1s, t2s)
    check(
        np.abs(t2 + t2.transpose(1, 0, 2, 3)).max() < 1e-15,
        "spin-adaptation",
        "expanded t2 is antisymmetric",
    )
    e_b, r1, r2 = brute(t1, t2)
    a, b = slice(0, None, 2), slice(1, None, 2)
    d = max(np.abs(r1[a, a] - r1[b, b]).max(), np.abs(r1[a, b]).max())
    check(d < 1e-12, "spin-adaptation", f"singles alpha == beta, no spin flip: {d:.1e}")
    rcs = r2[a, b, a, b]
    d = np.abs(rcs - rcs.transpose(1, 0, 3, 2)).max()
    check(d < 1e-12, "spin-adaptation", f"R_ij^ab == R_ji^ba: {d:.1e}")
    d = np.abs(r2[a, a, a, a] - (rcs - rcs.transpose(0, 1, 3, 2))).max()
    check(
        d < 1e-12, "spin-adaptation", f"same-spin block == R_ij^ab - R_ij^ba: {d:.1e}"
    )
    d = abs(cs_energy(t1s, t2s) - e_b)
    check(
        d < 1e-12, "spin-adaptation", f"closed-shell energy sum vs brute force: {d:.1e}"
    )
    return cs_energy, t1s, t2s, e_b


def solve_ccsd():
    t1 = np.zeros((NO2, NV2))
    t2 = oovv / DIJAB
    for _ in range(200):
        r1, r2 = ccsd_r(t1, t2)
        n1, n2 = r1 / DIA, r2 / DIJAB
        if max(np.abs(n1 - t1).max(), np.abs(n2 - t2).max()) < 1e-14:
            return n1, n2
        t1, t2 = n1, n2
    raise RuntimeError("model CCSD did not converge")


def pack(r1, r2):
    return np.concatenate([[r1[i, a] for i, a in SINGLES], [r2[x] for x in DOUBLES]])


def unpack(v):
    t1 = np.zeros((NO2, NV2))
    t2 = np.zeros((NO2, NO2, NV2, NV2))
    for k, (i, a) in enumerate(SINGLES):
        t1[i, a] = v[k]
    for k, (i, j, a, b) in enumerate(DOUBLES):
        val = v[len(SINGLES) + k]
        t2[i, j, a, b], t2[j, i, a, b], t2[i, j, b, a], t2[j, i, b, a] = (
            val,
            -val,
            -val,
            val,
        )
    return t1, t2


def claim_eom():
    print("\n=== EOM-CCSD matrix == CC Jacobian at convergence")
    t1, t2 = solve_ccsd()
    res = np.abs(pack(*full_residual(t1, t2))).max()
    check(res < 1e-12, "eom-jacobian", f"converged CCSD residual {res:.1e}")
    x0 = pack(t1, t2)
    n = x0.size
    h = 1e-3
    jac = np.zeros((n, n))
    for k in range(n):
        cols = []
        for s in (-2, -1, 1, 2):
            x = x0.copy()
            x[k] += s * h
            cols.append(pack(*full_residual(*unpack(x))))
        jac[:, k] = (cols[0] - 8 * cols[1] + 8 * cols[2] - cols[3]) / (12 * h)
    t = t_operator(t1, t2).toarray()
    m = HAM.toarray()
    hbar = m.copy()
    # e^{-T} H e^{T} as a terminating series of nested commutators.
    term = m.copy()
    for k in range(1, 2 * NO2 + 1):
        term = (term @ t - t @ term) / k
        hbar = hbar + term
    e_cc = REF @ hbar @ REF
    basis = np.vstack([PHI1, PHI2])
    hsd = basis @ hbar @ basis.T - e_cc * np.eye(n)
    d = np.abs(jac - hsd).max() / np.abs(hsd).max()
    check(d < 1e-10, "eom-jacobian", f"max |J - (Hbar - E_CC)| / max|Hbar| = {d:.1e}")
    ev = np.sort(np.linalg.eigvals(jac).real)
    print(f"       lowest EOM roots of the model: {ev[:3]}")


def claim_cis(coulomb=2.0):
    print("\n=== singlet CIS block at t = 0")
    m = HAM.toarray()
    nsing = NO * NV
    blk = np.zeros((nsing, nsing))
    vecs = []
    for i in range(NO):
        for a in range(NV):
            vecs.append(
                (det_single(2 * i, 2 * a) + det_single(2 * i + 1, 2 * a + 1))
                / np.sqrt(2.0)
            )
    vecs = np.array(vecs)
    blk = vecs @ m @ vecs.T - E_REF * np.eye(nsing)
    ref = np.zeros((nsing, nsing))
    for (i, a), (j, b) in itertools.product(
        itertools.product(range(NO), range(NV)), repeat=2
    ):
        ref[i * NV + a, j * NV + b] = (
            (EPS[NO + a] - EPS[i]) * (i == j) * (a == b)
            + coulomb * G[i, NO + a, j, NO + b]
            - G[i, j, NO + a, NO + b]
        )
    d = np.abs(blk - ref).max()
    check(d < 1e-12, "cis-limit", f"max |block - formula| = {d:.1e}")


def spin_norm(r1, r2):
    """eom_ccsd.rs::spin_norm, squared."""
    return (
        2.0 * (r1**2).sum()
        + (r2**2).sum()
        + 0.5 * ((r2 - r2.transpose(0, 1, 3, 2)) ** 2).sum()
    )


def claim_spin_norm():
    print("\n=== spin-summed EOM vector norm")
    r1s = rng.normal(size=(NO, NV))
    x = rng.normal(size=(NO, NO, NV, NV))
    r2s = x + x.transpose(1, 0, 3, 2)
    r1, r2 = expand(r1s, r2s)
    n_so = (r1**2).sum() + sum(r2[d] ** 2 for d in DOUBLES)
    d = abs(spin_norm(r1s, r2s) - n_so) / n_so
    check(d < 1e-13, "spin-norm", f"relative difference {d:.1e}")
    return r1s, r2s, n_so


def claim_negative_controls(t1, t2, sa, r1s, r2s, n_so):
    print("\n=== negative controls (each must differ)")
    e_b, r1_b, r2_b = brute(t1, t2)
    f2 = full_residual(t1, t2, wmbej_t2=1.0)[1]
    check(
        np.abs(f2 - r2_b).max() > 1e-6,
        "negative-controls",
        "W_mbej with t2 instead of t2/2",
    )
    check(
        abs(ccsd_energy(t1, t2, quarter_t1=0.25) - e_b) > 1e-6,
        "negative-controls",
        "t1 t1 energy weight 1/4",
    )
    cs_energy, t1s, t2s, e_cs = sa
    check(
        abs(cs_energy(t1s, t2s, exchange=0.0) - e_cs) > 1e-6,
        "negative-controls",
        "closed-shell energy without exchange",
    )
    bad = 2.0 * (r1s**2).sum() + (r2s**2).sum()
    check(
        abs(bad - n_so) > 1e-6,
        "negative-controls",
        "spin norm without the same-spin term",
    )


if __name__ == "__main__":
    claim_ccd()
    t1, t2 = claim_ccsd()
    sa = claim_spin_adaptation()
    claim_eom()
    claim_cis()
    r1s, r2s, n_so = claim_spin_norm()
    claim_negative_controls(t1, t2, sa, r1s, r2s, n_so)
    if FAILED:
        print(f"\n{len(FAILED)} FAILED: {sorted(set(FAILED))}")
        sys.exit(1)
    print("\nall checks hold")
