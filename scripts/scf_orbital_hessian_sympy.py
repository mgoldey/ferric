#!/usr/bin/env python3
"""Orbital gradients and Hessian-vector products of the SCF second-order
solvers and of stability analysis, checked against finite differences of the
energy.

Derivation: `book/src/08-convergence/03-second-order.md#second-order-orbital-gradient-hessian` and "SCF stability analysis". Code:
crates/ferric-scf/src/rhf_newton.rs (hessian_matvec_jk,
triplet_hessian_matvec_jk), uhf_newton.rs (hessian_matvec), rohf_trah.rs
(rohf_exact_gradient, rohf_exact_hessian_matvec), trah.rs
(solve_trust_region, offdiag_gap_correction), cayley.rs (cayley_unitary),
stability.rs.

Conventions (as coded): kappa antisymmetric, stored kappa_ai (virtual row,
occupied column), C <- C U(kappa) with the Cayley U = (1 - k/2)^-1 (1 + k/2).
J[D]_mn = sum (mn|ls) D_ls, K[D]_mn = sum (ml|ns) D_ls. Closed shell
D = 2 C_o C_o^T, F = h + J - K/2; UHF D_s = C_so C_so^T, F_s = h + J[D] - K[D_s].
The coded RHF and UHF quantities are fixed fractions of the true derivatives:
  RHF  g_code = F_ai            dE/dkappa = 4 g_code    d2E = 4 k.(H_code k)
  UHF  g_code = F^s_ai          dE/dkappa = 2 g_code    d2E = 2 k.(H_code k)
  ROHF rohf_exact_* are at true scale (rohf_newton's matvec is half of it).
The coded RHF/UHF H_code keeps only the diagonal gap (F_aa - F_ii); it is
exact at canonical orbitals only, and offdiag_gap_correction restores it.

Numerical checks (random h, positive (mn|ls) = sum_P B B, orthonormal AO
basis, n = 6; RHF 2 doubly occupied; UHF 2 alpha + 1 beta; ROHF 1 closed +
1 open). Every reference derivative is a fourth-order central difference of
the energy E(C U(t kappa)) itself, never of a coded gradient.
  [H1] RHF: dE/dt = 4 sum F_ai kappa_ai at a random orthonormal C.
  [H2] RHF: at the converged canonical C, d2E/dt2 = 4 k.(H_code k) with
       H_code k = (F_aa - F_ii) k_ai + C_v^T (J[dD] - K[dD]/2) C_o,
       dD = 2 C (k + k^T) C^T, and equals the MO-integral form
       (e_a - e_i) k_ai + sum_bj [4(ai|bj) - (ab|ij) - (aj|bi)] k_bj.
  [H3] RHF at a converged but non-canonical C (occupied block rotated): the
       diagonal-gap H_code misses the FD curvature; adding
       offdiag_gap_correction = F_vv k - k F_oo - (F_aa - F_ii) k recovers it.
  [H4] UHF: dE/dt = 2 sum_s F^s_ai k^s_ai; at the converged canonical UHF
       d2E/dt2 = 2 sum_s k^s.(H_code k)^s with dD_s = C_s (k_s + k_s^T) C_s^T,
       dF_s = J[dD_a + dD_b] - K[dD_s].
  [H5] ROHF (rohf_trah.rs, true scale): the exact gradient and the exact
       Hessian product G2 of the module doc equal the first and second FD of
       E(C U(t x)) at a random C, and their polarization y.Hx = x.Hy.
  [H6] DF-K response identity of occ_pair_response_k:
       K[(X+Y)(X+Y)^T] - K[(X-Y)(X-Y)^T] = K[2 (Y X^T + X Y^T)].
  [S1] Stability: at the RHF solution, the coded UHF operator in the singlet
       direction (k, k) equals the coded RHF singlet operator, and in the
       triplet direction (k, -k) equals the coded triplet operator
       (e_a - e_i) k - [(ab|ij) + (aj|bi)] k, with zero coupling between them.
  [S2] The second derivative of the UHF energy at the RHF solution along the
       triplet direction (v, -v) of a unit eigenvector v of the triplet
       operator is 4 lambda_T, so lambda_T < 0 means a lower UHF solution.
  [C1] (sympy) The Cayley U(t k) equals exp(t k) through t^2 and is
       orthogonal, so the FD derivatives below are those of E(C exp(t k)).
  [T1] Augmented Hessian (trah.rs solve_trust_region): the lowest eigenvector
       (v0, v) of [[0, a g^T], [a g, H]] gives k = v / (a v0) with
       (H - mu) k = -g, and the predicted change g.k + k.Hk/2 equals the
       cheap form (g.k + mu |k|^2) / 2 the code uses.

Mutations (each turns the script red; the runner applies them as exact
one-occurrence substitutions):
  rhf_hvp: dD = 2 C k~ C^T -> 1 C k~ C^T                (H2 fails)
  rhf_hvp: exchange -K/2 -> -K                           (H2 fails)
  uhf_hvp: Coulomb on the total dD -> on dD_s only       (H4 fails)
  triplet_hvp: drop the exchange term                    (S1 fails)
  rohf_hvp: Fock commutator 1/2 (X_s f - f X_s) -> 1     (H5 fails)
  ah_step: predicted change + mu |k|^2 -> - mu |k|^2      (T1 fails)
Exits nonzero if any check fails; prints "FAIL [<claim>] ...".
"""

import sys

import numpy as np
import sympy as sp

failures = []


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        failures.append(label)


N = 6
rng = np.random.default_rng(20261115)
_h = rng.normal(size=(N, N)) * 0.3
H = np.diag([-6.0, -5.0, -1.0, 0.0, 0.5, 1.5]) + 0.5 * (_h + _h.T)
_b = rng.normal(size=(12, N, N)) * 0.15
_b = _b + _b.transpose(0, 2, 1)
ERI = np.einsum("pmn,pls->mnls", _b, _b)


def jmat(d):
    return np.einsum("mnls,ls->mn", ERI, d)


def kmat(d):
    return np.einsum("mlns,ls->mn", ERI, d)


def cayley(k):
    one = np.eye(k.shape[0])
    return np.linalg.solve(one - 0.5 * k, one + 0.5 * k)


def random_orthogonal(n):
    return np.linalg.qr(rng.normal(size=(n, n)))[0]


def fd(fun, h=1e-3):
    """First and second derivatives at t = 0 by fourth-order central stencils."""
    e = {s: fun(s * h) for s in (-2, -1, 0, 1, 2)}
    d1 = (e[-2] - 8 * e[-1] + 8 * e[1] - e[2]) / (12 * h)
    d2 = (-e[-2] + 16 * e[-1] - 30 * e[0] + 16 * e[1] - e[2]) / (12 * h * h)
    return d1, d2


def kappa_full(kov, nocc):
    k = np.zeros((N, N))
    k[nocc:, :nocc] = kov
    k[:nocc, nocc:] = -kov.T
    return k


# ---------------------------------------------------------------- RHF
NO = 2


def rhf_energy(c):
    co = c[:, :NO]
    d = 2.0 * co @ co.T
    f = H + jmat(d) - 0.5 * kmat(d)
    return 0.5 * np.sum(d * (H + f))


def rhf_fock(c):
    co = c[:, :NO]
    d = 2.0 * co @ co.T
    return H + jmat(d) - 0.5 * kmat(d)


def rhf_scf():
    _, c = np.linalg.eigh(H)
    f_hist, e_hist = [], []
    for _ in range(300):
        f = rhf_fock(c)
        co = c[:, :NO]
        d = 2.0 * co @ co.T
        e = f @ d - d @ f
        if np.abs(e).max() < 1e-12:
            break
        f_hist.append(f)
        e_hist.append(e)
        del f_hist[:-8], e_hist[:-8]
        m = len(e_hist)
        a = np.zeros((m + 1, m + 1))
        a[:m, :m] = [[np.sum(x * y) for y in e_hist] for x in e_hist]
        a[:m, m] = a[m, :m] = 1.0
        rhs = np.zeros(m + 1)
        rhs[m] = 1.0
        cc = np.linalg.solve(a, rhs)[:m] if m > 1 else np.ones(1)
        _, c = np.linalg.eigh(sum(x * y for x, y in zip(cc, f_hist)))
    return c


def rhf_hvp(c, kov):
    """rhf_newton.rs::hessian_matvec_jk (HF): the coded quarter-scale product."""
    f_mo = c.T @ rhf_fock(c) @ c
    ksym = np.zeros((N, N))
    ksym[NO:, :NO] = kov
    ksym[:NO, NO:] = kov.T
    dd = 2.0 * c @ ksym @ c.T
    df = jmat(dd) - 0.5 * kmat(dd)
    out = (c.T @ df @ c)[NO:, :NO]
    gap = np.diag(f_mo)[NO:, None] - np.diag(f_mo)[None, :NO]
    return out + gap * kov


def offdiag_gap_correction(c, kov):
    f_mo = c.T @ rhf_fock(c) @ c
    fvv, foo = f_mo[NO:, NO:], f_mo[:NO, :NO]
    gap = np.diag(f_mo)[NO:, None] - np.diag(f_mo)[None, :NO]
    return fvv @ kov - kov @ foo - gap * kov


def triplet_hvp(c, kov):
    """rhf_newton.rs::triplet_hessian_matvec_jk (HF): no J, exchange only."""
    f_mo = c.T @ rhf_fock(c) @ c
    ksym = np.zeros((N, N))
    ksym[NO:, :NO] = kov
    ksym[:NO, NO:] = kov.T
    dd = 2.0 * c @ ksym @ c.T
    df = -0.5 * kmat(dd)
    gap = np.diag(f_mo)[NO:, None] - np.diag(f_mo)[None, :NO]
    return (c.T @ df @ c)[NO:, :NO] + gap * kov


def mo_eri(c):
    return np.einsum("mp,nq,mnls,lr,su->pqru", c, c, ERI, c, c, optimize=True)


# ---------------------------------------------------------------- UHF
NA, NB = 2, 1


def uhf_energy(ca, cb):
    da = ca[:, :NA] @ ca[:, :NA].T
    db = cb[:, :NB] @ cb[:, :NB].T
    j = jmat(da + db)
    fa, fb = H + j - kmat(da), H + j - kmat(db)
    return 0.5 * (np.sum(da * (H + fa)) + np.sum(db * (H + fb)))


def uhf_focks(ca, cb, na=NA, nb=NB):
    da = ca[:, :na] @ ca[:, :na].T
    db = cb[:, :nb] @ cb[:, :nb].T
    j = jmat(da + db)
    return H + j - kmat(da), H + j - kmat(db)


def uhf_scf():
    _, c = np.linalg.eigh(H)
    ca, cb = c.copy(), c.copy()
    for _ in range(2000):
        fa, fb = uhf_focks(ca, cb)
        ga = (ca.T @ fa @ ca)[NA:, :NA]
        gb = (cb.T @ fb @ cb)[NB:, :NB]
        if max(np.abs(ga).max(), np.abs(gb).max()) < 1e-12:
            break
        _, ca_new = np.linalg.eigh(fa)
        _, cb_new = np.linalg.eigh(fb)
        ca = ca_new
        cb = cb_new
    return ca, cb


def uhf_hvp(ca, cb, ka, kb, na=NA, nb=NB):
    """uhf_newton.rs::hessian_matvec (HF): the coded half-scale product."""
    fa, fb = uhf_focks(ca, cb, na, nb)

    def sym(kov, no):
        m = np.zeros((N, N))
        m[no:, :no] = kov
        m[:no, no:] = kov.T
        return m

    dda = ca @ sym(ka, na) @ ca.T
    ddb = cb @ sym(kb, nb) @ cb.T
    jt = jmat(dda + ddb)
    out = []
    for c, f, dd, k, no in ((ca, fa, dda, ka, na), (cb, fb, ddb, kb, nb)):
        df = jt - kmat(dd)
        f_mo = c.T @ f @ c
        gap = np.diag(f_mo)[no:, None] - np.diag(f_mo)[None, :no]
        out.append((c.T @ df @ c)[no:, :no] + gap * k)
    return out


# ---------------------------------------------------------------- ROHF
ND, NOPEN = 1, 1
NAL = ND + NOPEN


def rohf_pairs():
    """Packing vc | vo | oc, p the less occupied index (rohf_trah.rs pairs)."""
    vc = [(p, q) for p in range(NAL, N) for q in range(ND)]
    vo = [(p, q) for p in range(NAL, N) for q in range(ND, NAL)]
    oc = [(p, q) for p in range(ND, NAL) for q in range(ND)]
    return vc + vo + oc


PAIRS = rohf_pairs()
OCC_A = np.array([1.0 if i < NAL else 0.0 for i in range(N)])
OCC_B = np.array([1.0 if i < ND else 0.0 for i in range(N)])


def unpack(x):
    k = np.zeros((N, N))
    for v, (p, q) in zip(x, PAIRS):
        k[p, q] = v
        k[q, p] = -v
    return k


def rohf_energy(c):
    da = c @ np.diag(OCC_A) @ c.T
    db = c @ np.diag(OCC_B) @ c.T
    j = jmat(da + db)
    fa, fb = H + j - kmat(da), H + j - kmat(db)
    return 0.5 * (np.sum(da * (H + fa)) + np.sum(db * (H + fb)))


def rohf_mo_focks(c):
    da = c @ np.diag(OCC_A) @ c.T
    db = c @ np.diag(OCC_B) @ c.T
    j = jmat(da + db)
    return c.T @ (H + j - kmat(da)) @ c, c.T @ (H + j - kmat(db)) @ c


def rohf_exact_gradient(fa, fb):
    out = []
    for p, q in PAIRS:
        if p >= NAL and q < ND:
            out.append(2.0 * (fa[p, q] + fb[p, q]))
        elif p >= NAL:
            out.append(2.0 * fa[p, q])
        else:
            out.append(2.0 * fb[p, q])
    return np.array(out)


def rohf_hvp(c, x):
    """rohf_trah.rs::rohf_exact_hessian_matvec at true scale."""
    fa, fb = rohf_mo_focks(c)
    xk = unpack(x)
    xs = [xk * (occ[None, :] - occ[:, None]) for occ in (OCC_A, OCC_B)]
    dd = [c @ m @ c.T for m in xs]
    jt = jmat(dd[0] + dd[1])
    dfs = [c.T @ (jt - kmat(m)) @ c for m in dd]
    g = np.zeros((N, N))
    for occ, f, xsig, dfm in zip((OCC_A, OCC_B), (fa, fb), xs, dfs):
        a = f @ xk - xk @ f
        b = xsig @ f - f @ xsig
        dn = occ[:, None] - occ[None, :]
        g += 0.5 * dn * a + 0.5 * b + dn * dfm
    return np.array([g[q, p] - g[p, q] for p, q in PAIRS])


# ---------------------------------------------------------------- checks
def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def checks():
    # H1 RHF gradient at a random C
    c = random_orthogonal(N)
    kov = rng.normal(size=(N - NO, NO))
    g = (c.T @ rhf_fock(c) @ c)[NO:, :NO]
    d1, _ = fd(lambda t: rhf_energy(c @ cayley(t * kappa_full(kov, NO))))
    e = rel(d1, 4.0 * np.sum(g * kov))
    check("[H1] RHF dE/dt = 4 sum F_ai k_ai", e < 1e-8, f"rel {e:.1e}")

    # H2 RHF Hessian at the canonical solution
    c0 = rhf_scf()
    _, d2 = fd(lambda t: rhf_energy(c0 @ cayley(t * kappa_full(kov, NO))))
    hk = rhf_hvp(c0, kov)
    e_fd = rel(d2, 4.0 * np.sum(kov * hk))
    g_mo = mo_eri(c0)
    eps = np.diag(c0.T @ rhf_fock(c0) @ c0)
    o, v = slice(0, NO), slice(NO, N)
    mo_form = (eps[v, None] - eps[None, o]) * kov + np.einsum(
        "aibj,bj->ai",
        4 * g_mo[v, o, v, o]
        - g_mo[v, v, o, o].transpose(0, 2, 1, 3)
        - g_mo[v, o, v, o].transpose(0, 3, 2, 1),
        kov,
    )
    e_mo = np.abs(mo_form - hk).max()
    check(
        "[H2] RHF d2E = 4 k.(H_code k), H_code == MO-integral A+B form",
        e_fd < 1e-6 and e_mo < 1e-11,
        f"FD rel {e_fd:.1e}, MO form {e_mo:.1e}",
    )

    # H3 non-canonical converged orbitals
    u = np.eye(N)
    u[:NO, :NO] = random_orthogonal(NO)
    c1 = c0 @ u
    _, d2 = fd(lambda t: rhf_energy(c1 @ cayley(t * kappa_full(kov, NO))))
    plain = 4.0 * np.sum(kov * rhf_hvp(c1, kov))
    fixed = plain + 4.0 * np.sum(kov * offdiag_gap_correction(c1, kov))
    check(
        "[H3] non-canonical: diagonal gap misses, offdiag_gap_correction recovers",
        rel(fixed, d2) < 1e-6 and rel(plain, d2) > 1e-4,
        f"plain {rel(plain, d2):.1e}, corrected {rel(fixed, d2):.1e}",
    )

    # H4 UHF
    ca, cb = uhf_scf()
    ka = rng.normal(size=(N - NA, NA))
    kb = rng.normal(size=(N - NB, NB))

    def e_uhf(t, ca=ca, cb=cb):
        return uhf_energy(
            ca @ cayley(t * kappa_full(ka, NA)), cb @ cayley(t * kappa_full(kb, NB))
        )

    fa, fb = uhf_focks(ca, cb)
    ga = (ca.T @ fa @ ca)[NA:, :NA]
    gb = (cb.T @ fb @ cb)[NB:, :NB]
    cr = random_orthogonal(N)
    fa_r, fb_r = uhf_focks(cr, cr)
    d1, _ = fd(lambda t: e_uhf(t, cr, cr))
    e1 = rel(
        d1,
        2.0
        * (
            np.sum((cr.T @ fa_r @ cr)[NA:, :NA] * ka)
            + np.sum((cr.T @ fb_r @ cr)[NB:, :NB] * kb)
        ),
    )
    _, d2 = fd(e_uhf)
    ha, hb = uhf_hvp(ca, cb, ka, kb)
    e2 = rel(d2, 2.0 * (np.sum(ka * ha) + np.sum(kb * hb)))
    check(
        "[H4] UHF dE/dt = 2 sum F^s k^s, d2E = 2 sum k^s.(H_code k)^s",
        e1 < 1e-8 and e2 < 1e-6 and max(np.abs(ga).max(), np.abs(gb).max()) < 1e-10,
        f"grad rel {e1:.1e}, Hessian rel {e2:.1e}",
    )

    # H5 ROHF exact gradient and Hessian at a random C
    c = random_orthogonal(N)
    x = rng.normal(size=len(PAIRS))
    y = rng.normal(size=len(PAIRS))
    fa, fb = rohf_mo_focks(c)
    gx = rohf_exact_gradient(fa, fb)
    d1, d2 = fd(lambda t: rohf_energy(c @ cayley(t * unpack(x))))
    hx, hy = rohf_hvp(c, x), rohf_hvp(c, y)
    e1 = rel(d1, gx @ x)
    e2 = rel(d2, x @ hx)
    e3 = abs(y @ hx - x @ hy) / max(abs(y @ hx), 1e-12)
    check(
        "[H5] ROHF exact gradient and G2 Hessian product == FD of E, symmetric",
        e1 < 1e-8 and e2 < 1e-6 and e3 < 1e-10,
        f"grad {e1:.1e}, Hessian {e2:.1e}, symmetry {e3:.1e}",
    )

    # H6 DF-K response identity
    xo = rng.normal(size=(N, NO))
    yv = rng.normal(size=(N, NO))
    lhs = kmat((xo + yv) @ (xo + yv).T) - kmat((xo - yv) @ (xo - yv).T)
    rhs = kmat(2.0 * (yv @ xo.T + xo @ yv.T))
    e = np.abs(lhs - rhs).max() / np.abs(rhs).max()
    check(
        "[H6] K[(X+Y)(X+Y)^T] - K[(X-Y)(X-Y)^T] == K[2(YX^T + XY^T)]",
        e < 1e-13,
        f"rel {e:.1e}",
    )

    # S1 singlet / triplet channels of the UHF operator at the RHF solution
    k1 = rng.normal(size=(N - NO, NO))
    k2 = rng.normal(size=(N - NO, NO))
    hs_a, hs_b = uhf_hvp(c0, c0, k1, k1, NO, NO)
    ht_a, ht_b = uhf_hvp(c0, c0, k2, -k2, NO, NO)
    sing = rhf_hvp(c0, k1)
    trip = triplet_hvp(c0, k2)
    g_tr = (eps[v, None] - eps[None, o]) * k2 - np.einsum(
        "aibj,bj->ai",
        g_mo[v, v, o, o].transpose(0, 2, 1, 3) + g_mo[v, o, v, o].transpose(0, 3, 2, 1),
        k2,
    )
    coupling = abs(np.sum(k2 * hs_a) - np.sum(k2 * hs_b))
    ok = max(np.abs(hs_a - sing).max(), np.abs(hs_b - sing).max()) < 1e-11
    ok = ok and max(np.abs(ht_a - trip).max(), np.abs(ht_b + trip).max()) < 1e-11
    ok = ok and np.abs(trip - g_tr).max() < 1e-11 and coupling < 1e-10
    check(
        "[S1] UHF operator at the RHF point: singlet == RHF, triplet == coded triplet",
        ok,
    )

    # S2 the triplet eigenvalue is the UHF curvature / 4
    nov = (N - NO) * NO
    tmat = np.zeros((nov, nov))
    for col in range(nov):
        e_col = np.zeros(nov)
        e_col[col] = 1.0
        tmat[:, col] = triplet_hvp(c0, e_col.reshape(N - NO, NO)).ravel()
    lam, vec = np.linalg.eigh(0.5 * (tmat + tmat.T))
    vmin = vec[:, 0].reshape(N - NO, NO)
    _, d2 = fd(
        lambda t: uhf_energy_general(
            c0 @ cayley(t * kappa_full(vmin, NO)),
            c0 @ cayley(-t * kappa_full(vmin, NO)),
        )
    )
    check(
        "[S2] d2E_UHF along (v, -v) at the RHF point = 4 lambda_T",
        rel(d2, 4.0 * lam[0]) < 1e-6 and np.abs(tmat - tmat.T).max() < 1e-11,
        f"lambda_T {lam[0]:.4f}, rel {rel(d2, 4.0 * lam[0]):.1e}",
    )

    # T1 augmented Hessian step
    m = 7
    q = rng.normal(size=(m, m))
    hm = 0.5 * (q + q.T)
    gv = rng.normal(size=m)
    ok = True
    for alpha in (1.0, 3.0, 0.2):
        aug = np.zeros((m + 1, m + 1))
        aug[0, 1:] = aug[1:, 0] = alpha * gv
        aug[1:, 1:] = hm
        w, z = np.linalg.eigh(aug)
        mu, v0, vv = w[0], z[0, 0], z[1:, 0]
        kv = vv / (alpha * v0)
        ok = ok and np.abs((hm - mu * np.eye(m)) @ kv + gv).max() < 1e-10
        full = gv @ kv + 0.5 * kv @ hm @ kv
        cheap = 0.5 * (gv @ kv + mu * kv @ kv)
        ok = ok and abs(full - cheap) < 1e-10 * max(1.0, abs(full))
    check("[T1] AH step solves (H - mu) k = -g; predicted == (g.k + mu|k|^2)/2", ok)


def uhf_energy_general(ca, cb):
    """UHF energy with NO alpha and NO beta electrons (the RHF determinant)."""
    da = ca[:, :NO] @ ca[:, :NO].T
    db = cb[:, :NO] @ cb[:, :NO].T
    j = jmat(da + db)
    fa, fb = H + j - kmat(da), H + j - kmat(db)
    return 0.5 * (np.sum(da * (H + fa)) + np.sum(db * (H + fb)))


def symbolic_checks():
    """[C1] The Cayley U(t k) = (1 - t k/2)^-1 (1 + t k/2) agrees with exp(t k)
    through t^2 for a symbolic antisymmetric 3 x 3 k, so first and second
    derivatives of E(C U(t k)) at t = 0 are those of E(C exp(t k)); the t^3
    terms differ."""
    t = sp.Symbol("t")
    a, b, cc = sp.symbols("a b c", real=True)
    k = sp.Matrix([[0, -a, -b], [a, 0, -cc], [b, cc, 0]])
    one = sp.eye(3)
    u = (one - t * k / 2).inv() * (one + t * k / 2)
    ser = u.applyfunc(lambda x: sp.series(x, t, 0, 4).removeO())
    expo = one + t * k + t**2 * k * k / 2 + t**3 * k * k * k / 6
    diff = (ser - expo).applyfunc(sp.expand)
    low = diff.applyfunc(lambda x: sp.expand(x).coeff(t, 1) + sp.expand(x).coeff(t, 2))
    cubic = diff.applyfunc(lambda x: sp.expand(x).coeff(t, 3))
    ok = low == sp.zeros(3, 3) and cubic != sp.zeros(3, 3)
    ok = ok and sp.simplify(u.T * u - one) == sp.zeros(3, 3)
    check("[C1] Cayley U == exp through t^2, orthogonal, differs at t^3", ok)


symbolic_checks()
checks()
if failures:
    print(f"{len(failures)} check(s) failed")
    sys.exit(1)
print("all checks passed")
