#!/usr/bin/env python3
"""Analytic Hessian with density-fitted Coulomb and exchange (RI-J, RI-K).

Derivation: `book/src/14-derivatives/04-hf-hessian.md#hf-hessian-fitted-jk`.
Code: crates/ferric-scf/src/hessian_df.rs (rij_terms_with_aux_at_precision:
RijTerms::skeleton, first_order_j), crates/ferric-scf/src/hessian_df_k.rs
(rik_channel_terms_with_aux: skeleton, K^x), crates/ferric-scf/src/
hessian_df_stencil.rs (eri3_deriv2_block: fourth-order stencil of
first-derivative blocks, rigid per atom, one-atom mask), and the combination
in crates/ferric-scf/src/hessian.rs::rhf_hessian_parts_with and
crates/ferric-scf/src/hessian/uhf_df.rs.

Model: raw three-index integrals A_P,mn(lam) and metric V_PQ(lam), quadratic
in two parameters lam = (x, y), give the fitted integrals
(mn|ls)_fit = A_P,mn M_PQ A_Q,ls with M = V^-1 (full rank). At fixed D:
  RI-J  E_J = 1/2 d^T M d, d_P = sum D_mn A_P,mn, c = M d, Z_x = d_x - V_x c
        E_x = d_x^T c - 1/2 c^T V_x c
        E_xy = d_xy^T c - 1/2 c^T V_xy c + Z_x^T M Z_y
        J^x = A_x c + A (M Z_x)
  RI-K  E_K = alpha <L, M L>, L_P,ij = (X^T A_P X)_ij, Gamma = M L,
        Z_x = L_x - V_x Gamma
        E_x = alpha [2 <L_x, Gamma> - <Gamma, V_x Gamma>]
        E_xy = alpha [2 <L_xy, Gamma> - <Gamma, V_xy Gamma> + 2 <Z_x, M Z_y>]
        K^x = K1 + K1^T - sum_PQ (V_x)_PQ Chat_P Chat_Q^T (hessian_df_k.rs)

Independent routes: sympy derivatives of the closed-form energies and Fock
matrices (rational arithmetic); fourth-order central differences of the
analytic gradient with the SCF re-solved at every displaced point; the exact
four-centre Hessian assembly of scripts/hf_hessian_sympy.py on a fit that
reproduces the exact integrals.

Checks:
  S1  (sympy) RI-J: E_xy formula = d^2 E_J / dx dy (rationals).
  S2  (sympy) RI-K, two occupied columns: E_xy formula = d^2 E_K / dx dy.
  S3  (sympy) J^x = d J[D] / dx at fixed D, the aux-fit response included.
  S4  (sympy) K^x as hessian_df_k.rs builds it = d K[D] / dx at fixed D.
  N1  RI-J: E_xy formula = fourth-order FD of the analytic E_x (numeric).
  N2  RI-K: the same for E_K.
  N3  J^x and K^x = fourth-order FD of J[D], K[D] at fixed D (numeric).
  H1  fitted RHF Hessian (skeleton + CPHF with the fitted G) = FD of the
      fitted analytic gradient, SCF re-solved: RI-J only, RI-K only, both.
  H2  fitted UHF Hessian (RI-J on D_a + D_b, one RI-K channel per spin,
      alpha_s = -1/2) = FD of the fitted UHF gradient.
  A1  exactness anchor: A = R B, V = R R^T reproduces the exact integrals
      (B^T B); the fitted Hessian (both fitted) equals the exact four-centre
      Hessian, and the fitted 2e skeleton equals sum Gamma (mn|ls)^xy.
  C1  negative controls: dropping Z_x^T M Z_y (both fits) or the aux-fit
      response A (M Z_x) of J^x misses FD by far more than H1 agrees.
  T1  why a dropped metric mode is refused: with M = the truncated inverse V+
      (cut in a gap) the full-rank E_xy formula misses the mixed FD of the
      energy, while with nothing dropped it matches.
  ST1 the fourth-order stencil of exact first-derivative blocks, rigid per
      atom and folded onto atoms, gives the atom second derivative with
      O(h^4) error (step-halving ratio ~16), to 1e-10 at the production step.
  ST2 a triple with every centre on one atom has zero atom second derivative
      (translation invariance), so the mask that skips it is exact; moving
      only one centre of a same-atom pair misses (negative control).

Mutations (each turns the script red; exact one-occurrence substitutions):
  RI-J metric term -1/2 c^T V_xy c loses its 1/2         (N1, H1, H2 fail)
  RI-K cross term 2 alpha <Z_x, M Z_y> loses its 2      (N2, H1, H2 fail)
  J^x without the aux-fit response A (M Z_x)            (N3, H1, H2 fail)
  K^x without its metric term                           (N3, H1, H2 fail)
  stencil weights (1, -8, 8, -1)/12h -> second order (0, -6, 6, 0)/12h  (ST1 fails)
Exits nonzero on any failed check; prints "FAIL <label>".
"""

import random
import sys

import numpy as np
import sympy as sp

failures = []


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        failures.append(label)


def fail_hard(label):
    """A failed SCF is a failed check (exit 1 with a FAIL line), not a crash."""
    print(f"FAIL {label}")
    sys.exit(1)


def sym(m):
    return 0.5 * (m + m.T)


# ---------------------------------------------------------------------------
# S1-S4: sympy, rational coefficients, polynomial dependence on (x, y)
# ---------------------------------------------------------------------------
def symbolic_checks():
    rng = random.Random(11)
    x, y = sp.symbols("x y")
    nao, naux = 2, 2

    def rsym(n, scale=1):
        m = sp.Matrix(n, n, lambda i, j: sp.Rational(rng.randint(-3, 3), 4) * scale)
        return (m + m.T) / 2

    def poly_sym(n, diag=0):
        return (
            rsym(n)
            + diag * sp.eye(n)
            + x * rsym(n, sp.Rational(1, 2))
            + y * rsym(n, sp.Rational(1, 2))
            + x * y * rsym(n, sp.Rational(1, 3))
            + x * x * rsym(n, sp.Rational(1, 3))
        )

    a = [poly_sym(nao) for _ in range(naux)]
    v = poly_sym(naux, diag=3)
    dmat = sp.Matrix([[sp.Rational(3, 2), sp.Rational(1, 3)], [sp.Rational(1, 3), 1]])
    xf = sp.Matrix([[1, sp.Rational(1, 2)], [sp.Rational(-1, 3), 1]])  # m = 2
    alpha = sp.Rational(-1, 4)
    at0 = {x: 0, y: 0}

    def dvec(dm):
        return sp.Matrix(
            [
                sum(dm[m, n] * a[p][m, n] for m in range(nao) for n in range(nao))
                for p in range(naux)
            ]
        )

    def dd(expr, *vs):
        return sp.diff(expr, *vs).subs(at0) if vs else expr.subs(at0)

    vinv = v.inv()
    d = dvec(dmat)
    e_j = sp.Rational(1, 2) * (d.T * vinv * d)[0]
    # formula at lam = 0 from first and second derivatives of d and V only
    m0 = v.subs(at0).inv()
    c = m0 * d.subs(at0)
    dx, dy_, dxy = dd(d, x), dd(d, y), dd(d, x, y)
    vx, vy, vxy = dd(v, x), dd(v, y), dd(v, x, y)
    zx, zy = dx - vx * c, dy_ - vy * c
    formula = (
        (dxy.T * c)[0] - sp.Rational(1, 2) * (c.T * vxy * c)[0] + (zx.T * m0 * zy)[0]
    )
    check(
        "S1 RI-J: d_xy^T c - 1/2 c^T V_xy c + Z_x^T M Z_y = d2E_J/dxdy (claim S1)",
        sp.simplify(formula - dd(e_j, x, y)) == 0,
    )

    # RI-K: L_P,ij = (X^T A_P X)_ij, <a, b> = sum_P,ij a_P,ij b_P,ij
    lmat = [xf.T * a[p] * xf for p in range(naux)]

    def ip(u, w):
        return sum(
            u[p][i, j] * w[p][i, j]
            for p in range(naux)
            for i in range(2)
            for j in range(2)
        )

    def mapply(mm, t):
        return [
            sum((mm[p, q] * t[q] for q in range(naux)), sp.zeros(2, 2))
            for p in range(naux)
        ]

    e_k = alpha * ip(lmat, mapply(vinv, lmat))
    l0 = [lm.subs(at0) for lm in lmat]
    gam = mapply(m0, l0)
    lx = [dd(lm, x) for lm in lmat]
    ly = [dd(lm, y) for lm in lmat]
    lxy = [dd(lm, x, y) for lm in lmat]
    zkx = [lx[p] - mapply(vx, gam)[p] for p in range(naux)]
    zky = [ly[p] - mapply(vy, gam)[p] for p in range(naux)]
    formula_k = alpha * (
        2 * ip(lxy, gam) - ip(gam, mapply(vxy, gam)) + 2 * ip(zkx, mapply(m0, zky))
    )
    check(
        "S2 RI-K: alpha[2<L_xy,G> - <G,V_xy G> + 2<Z_x,M Z_y>] = d2E_K/dxdy (claim S2)",
        sp.simplify(formula_k - dd(e_k, x, y)) == 0,
    )

    # J[D]_mn = sum_P A_P,mn (M d)_P at fixed D; J^x = A_x c + A (M Z_x)
    cfull = vinv * d
    j_of = sum((a[p] * cfull[p] for p in range(naux)), sp.zeros(nao, nao))
    yx = m0 * zx
    jx = sum(
        (dd(a[p], x) * c[p] + a[p].subs(at0) * yx[p] for p in range(naux)),
        sp.zeros(nao, nao),
    )
    check(
        "S3 J^x = A_x c + A (M Z_x) equals dJ[D]/dx at fixed D (claim S3)",
        sp.simplify(jx - dd(j_of, x)) == sp.zeros(nao, nao),
    )

    # K[D]_ml = sum_ns D_ns (mn|ls)_fit, D = X X^T; K^x as hessian_df_k.rs
    dk = xf * xf.T
    k_of = sp.zeros(nao, nao)
    for mu in range(nao):
        for la in range(nao):
            k_of[mu, la] = sum(
                dk[nu, sg] * a[p][mu, nu] * vinv[p, q] * a[q][la, sg]
                for nu in range(nao)
                for sg in range(nao)
                for p in range(naux)
                for q in range(naux)
            )
    a0 = [ap.subs(at0) for ap in a]
    ax = [dd(ap, x) for ap in a]
    bmat = [a0[p] * xf for p in range(naux)]  # B_P,lj = sum_s A_P,ls X_sj
    chat = [
        sum((m0[p, q] * bmat[q] for q in range(naux)), sp.zeros(nao, 2))
        for p in range(naux)
    ]
    tmat = [xf * chat[p].T for p in range(naux)]  # T_P,nl = sum_j X_nj Chat_P,lj
    k1 = sum((ax[p] * tmat[p] for p in range(naux)), sp.zeros(nao, nao))
    kx = k1 + k1.T
    for p in range(naux):
        for q in range(naux):
            kx -= vx[p, q] * chat[p] * chat[q].T
    check(
        "S4 K^x = K1 + K1^T - sum V_x Chat Chat^T equals dK[D]/dx at fixed D (claim S4)",
        sp.simplify(kx - dd(k_of, x)) == sp.zeros(nao, nao),
    )


# ---------------------------------------------------------------------------
# Numeric toy model: one-electron part, exact ERI B^T B, two fits (A, V).
# ---------------------------------------------------------------------------
N, NAUX, NPAR = 7, 12, 2


def quad(m0, m1, m2, lam):
    out = m0.copy()
    for k in range(NPAR):
        out = out + lam[k] * m1[k]
        for ll in range(NPAR):
            out = out + 0.5 * lam[k] * lam[ll] * m2[k, ll]
    return out


def quad1(m1, m2, lam, k):
    out = m1[k].copy()
    for ll in range(NPAR):
        out = out + lam[ll] * m2[k, ll]
    return out


class Quad:
    """A quantity quadratic in lam with exact first and second derivatives."""

    def __init__(self, make, scale0, scale1, scale2):
        self.m0 = make(scale0)
        self.m1 = [make(scale1) for _ in range(NPAR)]
        self.m2 = {}
        for k in range(NPAR):
            for ll in range(k, NPAR):
                self.m2[k, ll] = self.m2[ll, k] = make(scale2)

    def val(self, lam):
        return quad(self.m0, self.m1, self.m2, lam)

    def d1(self, lam, k):
        return quad1(self.m1, self.m2, lam, k)

    def d2(self, lam, k, ll):
        return self.m2[k, ll]


class Model:
    def __init__(self, rng):
        def symr(scale):
            return scale * sym(rng.standard_normal((N, N)))

        def br(scale):
            b = scale * rng.standard_normal((NAUX, N, N))
            return 0.5 * (b + b.transpose(0, 2, 1))

        self.h = Quad(symr, 0.3, 0.3, 0.2)
        self.h.m0 = self.h.m0 + np.diag(np.linspace(-6.0, 3.0, N))
        self.s = Quad(symr, 0.08, 0.05, 0.03)
        self.s.m0 = self.s.m0 + np.eye(N)
        self.b = Quad(br, 0.2, 0.05, 0.03)
        self.e1 = rng.standard_normal(NPAR)
        self.e2 = sym(rng.standard_normal((NPAR, NPAR)))

    def eri(self, lam):
        b = self.b.val(lam)
        return np.einsum("Pmn,Pls->mnls", b, b)

    def deri(self, lam, k):
        b, db = self.b.val(lam), self.b.d1(lam, k)
        return np.einsum("Pmn,Pls->mnls", db, b) + np.einsum("Pmn,Pls->mnls", b, db)

    def d2eri(self, lam, k, ll):
        b = self.b.val(lam)
        bk, bl, bkl = self.b.d1(lam, k), self.b.d1(lam, ll), self.b.d2(lam, k, ll)

        def pair(u, w):
            return np.einsum("Pmn,Pls->mnls", u, w)

        return pair(bkl, b) + pair(bk, bl) + pair(bl, bk) + pair(b, bkl)

    def enn(self, lam):
        return 3.0 + self.e1 @ lam + 0.5 * lam @ self.e2 @ lam

    def denn(self, lam, k):
        return self.e1[k] + self.e2[k] @ lam


class QuadFit:
    """Raw three-index A_P,mn(lam) and SPD metric V(lam), both quadratic."""

    def __init__(self, rng, naux):
        def ar(scale):
            t = scale * rng.standard_normal((naux, N, N))
            return 0.5 * (t + t.transpose(0, 2, 1))

        def vr(scale):
            return scale * sym(rng.standard_normal((naux, naux)))

        self.A = Quad(ar, 0.2, 0.05, 0.03)
        self.V = Quad(vr, 0.15, 0.05, 0.03)
        self.V.m0 = self.V.m0 + np.eye(naux)

    def a(self, lam):
        return self.A.val(lam)

    def a1(self, lam, k):
        return self.A.d1(lam, k)

    def a2(self, lam, k, ll):
        return self.A.d2(lam, k, ll)

    def v(self, lam):
        return self.V.val(lam)

    def v1(self, lam, k):
        return self.V.d1(lam, k)

    def v2(self, lam, k, ll):
        return self.V.d2(lam, k, ll)


class SpanningFit:
    """A = R B, V = R R^T: A^T V^-1 A = B^T B, the exact integrals, at every lam."""

    def __init__(self, rng, model):
        def rr(scale):
            return scale * rng.standard_normal((NAUX, NAUX))

        self.B = model.b
        self.R = Quad(rr, 0.15, 0.05, 0.03)
        self.R.m0 = self.R.m0 + np.eye(NAUX)

    def a(self, lam):
        return np.einsum("PQ,Qmn->Pmn", self.R.val(lam), self.B.val(lam))

    def a1(self, lam, k):
        r, b = self.R, self.B
        return np.einsum("PQ,Qmn->Pmn", r.d1(lam, k), b.val(lam)) + np.einsum(
            "PQ,Qmn->Pmn", r.val(lam), b.d1(lam, k)
        )

    def a2(self, lam, k, ll):
        r, b = self.R, self.B
        out = np.einsum("PQ,Qmn->Pmn", r.d2(lam, k, ll), b.val(lam))
        out = out + np.einsum("PQ,Qmn->Pmn", r.d1(lam, k), b.d1(lam, ll))
        out = out + np.einsum("PQ,Qmn->Pmn", r.d1(lam, ll), b.d1(lam, k))
        return out + np.einsum("PQ,Qmn->Pmn", r.val(lam), b.d2(lam, k, ll))

    def v(self, lam):
        r = self.R.val(lam)
        return r @ r.T

    def v1(self, lam, k):
        r, rk = self.R.val(lam), self.R.d1(lam, k)
        return rk @ r.T + r @ rk.T

    def v2(self, lam, k, ll):
        r, rk, rl = self.R.val(lam), self.R.d1(lam, k), self.R.d1(lam, ll)
        rkl = self.R.d2(lam, k, ll)
        return rkl @ r.T + rk @ rl.T + rl @ rk.T + r @ rkl.T


def fit_eri(fit, lam):
    a = fit.a(lam)
    return np.einsum("Pmn,PQ,Qls->mnls", a, np.linalg.inv(fit.v(lam)), a)


# ---------------------------------------------------------------------------
# RI-J terms (hessian_df.rs::rij_terms_with_aux_at_precision)
# ---------------------------------------------------------------------------
def rij_terms(fit, lam, d, drop_cross=False, omit_aux_jx=False):
    a, mm = fit.a(lam), np.linalg.inv(fit.v(lam))
    c = mm @ np.einsum("Pmn,mn->P", a, d)
    dx = np.array([np.einsum("Pmn,mn->P", fit.a1(lam, k), d) for k in range(NPAR)])
    vc = np.array([fit.v1(lam, k) @ c for k in range(NPAR)])
    z = dx if drop_cross else dx - vc
    y = z @ mm.T
    grad = np.array([dx[k] @ c - 0.5 * c @ vc[k] for k in range(NPAR)])
    skel = np.zeros((NPAR, NPAR))
    for k in range(NPAR):
        for ll in range(NPAR):
            three_centre = c @ np.einsum("Pmn,mn->P", fit.a2(lam, k, ll), d)
            metric_second = -0.5 * c @ fit.v2(lam, k, ll) @ c
            skel[k, ll] = three_centre + metric_second + z[k] @ y[ll]
    jx = []
    for k in range(NPAR):
        j = np.einsum("P,Pmn->mn", c, fit.a1(lam, k))
        if not omit_aux_jx:
            j = j + np.einsum("P,Pmn->mn", y[k], a)
        jx.append(j)
    return grad, sym(skel), jx


# ---------------------------------------------------------------------------
# RI-K terms for channels (X_s, alpha_s) (hessian_df_k.rs)
# ---------------------------------------------------------------------------
def rik_terms(fit, lam, channels, drop_cross=False, drop_kx_metric=False):
    a, mm = fit.a(lam), np.linalg.inv(fit.v(lam))
    vx = [fit.v1(lam, k) for k in range(NPAR)]
    grad = np.zeros(NPAR)
    skel = np.zeros((NPAR, NPAR))
    fock = []
    for xm, alpha in channels:
        ell = np.einsum("mi,Pmn,nj->Pij", xm, a, xm)
        gam = np.einsum("PQ,Qij->Pij", mm, ell)
        lx = [np.einsum("mi,Pmn,nj->Pij", xm, fit.a1(lam, k), xm) for k in range(NPAR)]
        vg = [np.einsum("PQ,Qij->Pij", vx[k], gam) for k in range(NPAR)]
        z = lx if drop_cross else [lx[k] - vg[k] for k in range(NPAR)]
        gx = [np.einsum("PQ,Qij->Pij", mm, z[k]) for k in range(NPAR)]
        for k in range(NPAR):
            grad[k] += alpha * (2.0 * np.sum(lx[k] * gam) - np.sum(gam * vg[k]))
            for ll in range(NPAR):
                lxy = np.einsum("mi,Pmn,nj->Pij", xm, fit.a2(lam, k, ll), xm)
                vg2 = np.einsum("PQ,Qij->Pij", fit.v2(lam, k, ll), gam)
                skel[k, ll] += alpha * (2.0 * np.sum(lxy * gam) - np.sum(gam * vg2))
                skel[k, ll] += 2.0 * alpha * np.sum(z[k] * gx[ll])
        # K^x = K1 + K1^T - sum_PQ (V_x)_PQ Chat_P Chat_Q^T
        chat = np.einsum("PQ,Qls,sj->Plj", mm, a, xm)
        tt = np.einsum("nj,Plj->Pnl", xm, chat)
        fk = []
        for k in range(NPAR):
            k1 = np.einsum("Pmn,Pnl->ml", fit.a1(lam, k), tt)
            kx = k1 + k1.T
            if not drop_kx_metric:
                kx = kx - np.einsum("PQ,Pmj,Qlj->ml", vx[k], chat, chat)
            fk.append(2.0 * alpha * kx)
        fock.append(fk)
    return grad, sym(skel), fock


# ---------------------------------------------------------------------------
# SCF on the fitted functional; exact-path pieces where a fit is absent
# ---------------------------------------------------------------------------
def jmat(g, d):
    return np.einsum("mnls,ls->mn", g, d)


def kmat(g, d):
    return np.einsum("mlns,ls->mn", g, d)


class Diis:
    def __init__(self, size=8):
        self.f, self.e, self.size = [], [], size

    def extrapolate(self, f, e):
        self.f.append(f)
        self.e.append(e)
        if len(self.f) > self.size:
            self.f.pop(0)
            self.e.pop(0)
        m = len(self.f)
        b = -np.ones((m + 1, m + 1))
        b[m, m] = 0.0
        for i in range(m):
            for j in range(m):
                b[i, j] = np.sum(self.e[i] * self.e[j])
        rhs = np.zeros(m + 1)
        rhs[m] = -1.0
        try:
            cf = np.linalg.solve(b, rhs)[:m]
        except np.linalg.LinAlgError:
            return f
        return sum(ci * fi for ci, fi in zip(cf, self.f))


def gen_eigh(f, s):
    w, v = np.linalg.eigh(s)
    xo = v @ np.diag(w**-0.5) @ v.T
    e, cp = np.linalg.eigh(xo @ f @ xo)
    return e, xo @ cp


TOL = 1e-12


class Route:
    """Which of J and K are fitted (fit object) or exact (None)."""

    def __init__(self, model, fit_j, fit_k):
        self.model, self.fit_j, self.fit_k = model, fit_j, fit_k

    def gj(self, lam):
        return self.model.eri(lam) if self.fit_j is None else fit_eri(self.fit_j, lam)

    def gk(self, lam):
        return self.model.eri(lam) if self.fit_k is None else fit_eri(self.fit_k, lam)


def solve_scf(route, lam, nocc, guess=None):
    """RHF (nocc int) or UHF (nocc pair) on G = J_fit - K_fit (1/2 K for RHF)."""
    m = route.model
    h, s, gj, gk = m.h.val(lam), m.s.val(lam), route.gj(lam), route.gk(lam)
    spins = [nocc] if np.isscalar(nocc) else list(nocc)
    occ = 2.0 if len(spins) == 1 else 1.0
    if guess is None:
        _, c0 = gen_eigh(h, s)
        cs = [c0 for _ in spins]
        if len(spins) == 2:
            th, nb = 0.4, spins[1]
            rot = np.eye(N)
            rot[nb - 1, nb - 1] = rot[nb, nb] = np.cos(th)
            rot[nb - 1, nb], rot[nb, nb - 1] = np.sin(th), -np.sin(th)
            cs[1] = c0 @ rot
    else:
        cs = list(guess)
    diis = Diis()
    for _ in range(600):
        ds = [occ * c[:, :n] @ c[:, :n].T for c, n in zip(cs, spins)]
        jm = jmat(gj, sum(ds))
        fs = [h + jm - k_scale(occ) * kmat(gk, dd) for dd in ds]
        es = [f @ dd @ s - s @ dd @ f for f, dd in zip(fs, ds)]
        if max(np.max(np.abs(e)) for e in es) < TOL:
            out = [gen_eigh(f, s) for f in fs]
            return [o[1] for o in out], [o[0] for o in out]
        fx = diis.extrapolate(np.concatenate(fs), np.concatenate(es))
        cs = [gen_eigh(fx[i * N : (i + 1) * N], s)[1] for i in range(len(spins))]
    fail_hard("fitted SCF converges on the toy model")


def k_scale(occ):
    """RHF G = J(D) - 1/2 K(D); UHF G_s = J(D_a + D_b) - K(D_s)."""
    return 0.5 if occ == 2.0 else 1.0


def densities(cs, nocc):
    spins = [nocc] if np.isscalar(nocc) else list(nocc)
    occ = 2.0 if len(spins) == 1 else 1.0
    return [occ * c[:, :n] @ c[:, :n].T for c, n in zip(cs, spins)], spins, occ


def k_channels(cs, spins, occ):
    """RHF: X = sqrt2 C_occ, alpha = -1/4; UHF: X_s = C_s,occ, alpha_s = -1/2."""
    if occ == 2.0:
        return [(np.sqrt(2.0) * cs[0][:, : spins[0]], -0.25)]
    return [(c[:, :n], -0.5) for c, n in zip(cs, spins)]


def energy_weighted(cs, epss, spins, occ):
    return sum(
        occ * (c[:, :n] * e[:n]) @ c[:, :n].T for c, e, n in zip(cs, epss, spins)
    )


def exact_gamma(ds, occ, fit_j, fit_k):
    """The parts of the four-centre Gamma that stay exact (hessian.rs)."""
    dt = sum(ds)
    gam = np.zeros((N, N, N, N))
    if fit_j is None:
        gam += 0.5 * np.einsum("mn,ls->mnls", dt, dt)
    if fit_k is None:
        kf = 0.25 if occ == 2.0 else 0.5
        for dd in ds:
            gam -= kf * np.einsum("ml,ns->mnls", dd, dd)
    return gam


def gradient(route, lam, cs, epss, nocc):
    m = route.model
    ds, spins, occ = densities(cs, nocc)
    dt, w = sum(ds), energy_weighted(cs, epss, spins, occ)
    gam = exact_gamma(ds, occ, route.fit_j, route.fit_k)
    g = np.array(
        [
            m.denn(lam, k)
            + np.sum(dt * m.h.d1(lam, k))
            - np.sum(w * m.s.d1(lam, k))
            + np.sum(gam * m.deri(lam, k))
            for k in range(NPAR)
        ]
    )
    if route.fit_j is not None:
        g = g + rij_terms(route.fit_j, lam, dt)[0]
    if route.fit_k is not None:
        g = g + rik_terms(route.fit_k, lam, k_channels(cs, spins, occ))[0]
    return g


def solve_dense(apply, rhs):
    n = rhs.size
    a = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        a[:, i] = apply(e)
    return np.linalg.solve(a, rhs)


def cphf_response(model, lam, spins, kernel, first):
    """hessian.rs solve_cphf_block / cphf_rhs / finish_cphf / assemble_response
    for RHF (one spin, occ 2) and UHF (two spins, occ 1), any kernel G."""
    shapes = [(N - no, no) for (_, _, no, _) in spins]
    sizes = [p * q for p, q in shapes]

    def split(vec):
        out, off = [], 0
        for shp, sz in zip(shapes, sizes):
            out.append(vec[off : off + sz].reshape(shp))
            off += sz
        return out

    def d1(us):
        out = []
        for (c, _, no, occ), u in zip(spins, us):
            half = c[:, no:] @ u @ c[:, :no].T
            out.append(occ * (half + half.T))
        return out

    def apply(vec):
        us = split(vec)
        v = kernel(d1(us))
        out = []
        for (c, eps, no, _), u, vs in zip(spins, us, v):
            denom = eps[no:, None] - eps[None, :no]
            out.append(denom * u + c[:, no:].T @ vs @ c[:, :no])
        return np.concatenate([o.ravel() for o in out])

    f1_mo, s1_mo, u_all, e1_all = [], [], [], []
    for k in range(NPAR):
        f1 = first(k)
        sk = model.s.d1(lam, k)
        f1k = [c.T @ f @ c[:, :no] for (c, _, no, _), f in zip(spins, f1)]
        s1k = [c.T @ sk @ c[:, :no] for (c, _, no, _) in spins]
        d_oo = [
            -occ * c[:, :no] @ s[:no] @ c[:, :no].T
            for (c, _, no, occ), s in zip(spins, s1k)
        ]
        g_oo = kernel(d_oo)
        rhs = [
            s[no:] * eps[:no] - f[no:] - c[:, no:].T @ go @ c[:, :no]
            for (c, eps, no, _), f, s, go in zip(spins, f1k, s1k, g_oo)
        ]
        us = split(solve_dense(apply, np.concatenate([r.ravel() for r in rhs])))
        g_tot = kernel([p + q for p, q in zip(d_oo, d1(us))])
        uk, ek = [], []
        for (c, eps, no, _), f, s, u, gt in zip(spins, f1k, s1k, us, g_tot):
            e1 = f[:no] + c[:, :no].T @ gt @ c[:, :no]
            e1 = e1 - 0.5 * s[:no] * (eps[:no, None] + eps[None, :no])
            uk.append(np.vstack([-0.5 * s[:no], u]))
            ek.append(e1)
        f1_mo.append(f1k)
        s1_mo.append(s1k)
        u_all.append(uk)
        e1_all.append(ek)

    r = np.zeros((NPAR, NPAR))
    for x in range(NPAR):
        for y in range(NPAR):
            for i, (_, eps, no, occ) in enumerate(spins):
                hs = 2.0 * occ * (f1_mo[x][i] - s1_mo[x][i] * eps[:no])
                r[x, y] += np.sum(hs * u_all[y][i]) - occ * np.sum(
                    s1_mo[x][i][:no] * e1_all[y][i]
                )
    return sym(r)


def hessian(route, lam, cs, epss, nocc, drop_cross=False, omit_aux_jx=False):
    """hessian.rs::rhf_hessian_parts_with / hessian/uhf_df.rs, fitted or exact."""
    m = route.model
    ds, spins, occ = densities(cs, nocc)
    dt, w = sum(ds), energy_weighted(cs, epss, spins, occ)
    gam = exact_gamma(ds, occ, route.fit_j, route.fit_k)
    skel = np.zeros((NPAR, NPAR))
    for k in range(NPAR):
        for ll in range(NPAR):
            skel[k, ll] = (
                m.e2[k, ll]
                + np.sum(dt * m.h.d2(lam, k, ll))
                - np.sum(w * m.s.d2(lam, k, ll))
                + np.sum(gam * m.d2eri(lam, k, ll))
            )
    jx = kx = None
    if route.fit_j is not None:
        _, sj, jx = rij_terms(route.fit_j, lam, dt, drop_cross, omit_aux_jx)
        skel = skel + sj
    if route.fit_k is not None:
        _, sk, kx = rik_terms(route.fit_k, lam, k_channels(cs, spins, occ), drop_cross)
        skel = skel + sk
    gj, gk = route.gj(lam), route.gk(lam)
    kscale = k_scale(occ)

    def first(k):
        out = []
        for i, dd in enumerate(ds):
            f = m.h.d1(lam, k).copy()
            f += jmat(m.deri(lam, k), dt) if jx is None else jx[k]
            f -= kscale * kmat(m.deri(lam, k), dd) if kx is None else -kx[i][k]
            out.append(f)
        return out

    def kernel(dl):
        jm = jmat(gj, sum(dl))
        return [jm - kscale * kmat(gk, dd) for dd in dl]

    spin_list = [(c, e, n, occ) for c, e, n in zip(cs, epss, spins)]
    return skel + cphf_response(m, lam, spin_list, kernel, first)


def fd_of_gradient(route, lam, nocc, guess):
    """d grad_k / d lam_l, fourth-order stencil, SCF re-solved (h = 1e-3)."""
    hh = 1e-3
    out = np.zeros((NPAR, NPAR))
    for ll in range(NPAR):
        vals = {}
        for t in (-2, -1, 1, 2):
            lt = lam.copy()
            lt[ll] += t * hh
            cs, epss = solve_scf(route, lt, nocc, guess)
            vals[t] = gradient(route, lt, cs, epss, nocc)
        out[:, ll] = (8 * (vals[1] - vals[-1]) - (vals[2] - vals[-2])) / (12 * hh)
    return out


def fd4(fun, lam, k):
    hh = 1e-3
    vals = {}
    for t in (-2, -1, 1, 2):
        lt = lam.copy()
        lt[k] += t * hh
        vals[t] = fun(lt)
    return (8 * (vals[1] - vals[-1]) - (vals[2] - vals[-2])) / (12 * hh)


# ---------------------------------------------------------------------------
# N1-N3, H1, H2, A1, C1
# ---------------------------------------------------------------------------
def numeric_checks():
    rng = np.random.default_rng(7)
    model = Model(rng)
    fit_j = QuadFit(rng, 9)
    fit_k = QuadFit(rng, 10)
    lam0 = np.array([0.3, -0.2])

    # N1, N2, N3 at a fixed density / occupied factor
    xm = rng.standard_normal((N, 3)) * 0.5
    dfix = xm @ xm.T
    _, sj, jx = rij_terms(fit_j, lam0, dfix)
    fd = np.array(
        [fd4(lambda lt: rij_terms(fit_j, lt, dfix)[0], lam0, ll) for ll in range(NPAR)]
    ).T
    err = np.max(np.abs(sj - fd))
    check(
        "N1 RI-J E_xy = FD of analytic E_x (claim N1)", err < 1e-9, f"max err {err:.1e}"
    )
    ch = [(xm, -0.25)]
    _, sk, kx = rik_terms(fit_k, lam0, ch)
    fd = np.array(
        [fd4(lambda lt: rik_terms(fit_k, lt, ch)[0], lam0, ll) for ll in range(NPAR)]
    ).T
    err = np.max(np.abs(sk - fd))
    check(
        "N2 RI-K E_xy = FD of analytic E_x (claim N2)", err < 1e-9, f"max err {err:.1e}"
    )
    errs = []
    for k in range(NPAR):
        jfd = fd4(lambda lt: jmat(fit_eri(fit_j, lt), dfix), lam0, k)
        kfd = fd4(lambda lt: kmat(fit_eri(fit_k, lt), dfix), lam0, k)
        errs.append(np.max(np.abs(jx[k] - jfd)))
        errs.append(np.max(np.abs(kx[0][k] / (2.0 * -0.25) - kfd)))
    err = max(errs)
    check(
        "N3 J^x, K^x = FD of J[D], K[D] at fixed D (claim N3)",
        err < 1e-9,
        f"max err {err:.1e}",
    )

    # H1 RHF, three routes; H2 UHF both fitted
    bar = 1e-7
    agree = 0.0
    both = None
    for name, fj, fk in (
        ("RI-J", fit_j, None),
        ("RI-K", None, fit_k),
        ("RI-J+RI-K", fit_j, fit_k),
    ):
        route = Route(model, fj, fk)
        cs, epss = solve_scf(route, lam0, 3)
        h_an = hessian(route, lam0, cs, epss, 3)
        fd = fd_of_gradient(route, lam0, 3, cs)
        err = np.max(np.abs(h_an - fd))
        agree = max(agree, err)
        check(
            f"H1 fitted RHF Hessian ({name}) = FD of the fitted gradient (claim H1)",
            err < bar,
            f"max err {err:.2e}",
        )
        if fj is not None and fk is not None:
            both = (route, cs, epss, fd)
    route = Route(model, fit_j, fit_k)
    cs, epss = solve_scf(route, lam0, (4, 2))
    h_an = hessian(route, lam0, cs, epss, (4, 2))
    fd = fd_of_gradient(route, lam0, (4, 2), cs)
    err = np.max(np.abs(h_an - fd))
    check(
        "H2 fitted UHF Hessian (RI-J+RI-K) = FD of the fitted gradient (claim H2)",
        err < bar,
        f"max err {err:.2e}",
    )

    # C1 negative controls on the both-fitted RHF
    route, cs, epss, fd = both
    for label, kw in (
        ("drop Z_x^T M Z_y", {"drop_cross": True}),
        ("drop A (M Z_x) from J^x", {"omit_aux_jx": True}),
    ):
        miss = np.max(np.abs(hessian(route, lam0, cs, epss, 3, **kw) - fd))
        check(
            f"C1 negative control: {label} misses FD (claim C1)",
            miss > 1e3 * max(agree, 1e-12) and miss > 1e-5,
            f"miss {miss:.2e} vs agreement {agree:.1e}",
        )

    # A1 exactness anchor: a fit that reproduces the exact integrals
    span = SpanningFit(rng, model)
    gap = np.max(np.abs(fit_eri(span, lam0) - model.eri(lam0)))
    exact = Route(model, None, None)
    fitted = Route(model, span, span)
    cs, epss = solve_scf(exact, lam0, 3)
    h_exact = hessian(exact, lam0, cs, epss, 3)
    h_fit = hessian(fitted, lam0, cs, epss, 3)
    err = np.max(np.abs(h_fit - h_exact))
    ds, spins, occ = densities(cs, 3)
    gam = exact_gamma(ds, occ, None, None)
    sk_exact = np.array(
        [
            [np.sum(gam * model.d2eri(lam0, k, ll)) for ll in range(NPAR)]
            for k in range(NPAR)
        ]
    )
    sk_fit = (
        rij_terms(span, lam0, ds[0])[1]
        + rik_terms(span, lam0, k_channels(cs, spins, occ))[1]
    )
    err_s = np.max(np.abs(sk_fit - sk_exact))
    check(
        "A1 spanning fit: fitted Hessian = exact Hessian, fitted 2e skeleton = exact (claim A1)",
        gap < 1e-12 and err < 1e-10 and err_s < 1e-10,
        f"ERI gap {gap:.1e} Hessian {err:.1e} skeleton {err_s:.1e}",
    )


# ---------------------------------------------------------------------------
# T1: why a metric with a dropped mode is refused
# ---------------------------------------------------------------------------
def truncated_inverse(v, cut):
    lam, u = np.linalg.eigh(v)
    w = np.where(lam > cut, 1.0 / np.where(lam > cut, lam, 1.0), 0.0)
    return (u * w) @ u.T


def truncation_check():
    rng = np.random.default_rng(5)
    naux, nao = 5, 3
    q, _ = np.linalg.qr(rng.standard_normal((naux, naux)))
    v0 = (q * np.array([4.0, 2.0, 1.0, 0.5, 0.05])) @ q.T

    def r(scale, n=naux):
        return scale * sym(rng.standard_normal((n, n)))

    v1 = [r(0.05) for _ in range(NPAR)]
    a0 = rng.standard_normal((naux, nao, nao))
    a0 = 0.5 * (a0 + a0.transpose(0, 2, 1))
    a1 = [0.3 * sym(rng.standard_normal((nao, nao))) for _ in range(naux * NPAR)]
    a1 = [np.array(a1[k * naux : (k + 1) * naux]) for k in range(NPAR)]
    xm = rng.standard_normal((nao, 2))
    alpha = -0.25

    def av(lam):
        return a0 + lam[0] * a1[0] + lam[1] * a1[1]

    def vv(lam):
        return v0 + lam[0] * v1[0] + lam[1] * v1[1]

    def energy(lam, cut):
        ell = np.einsum("mi,Pmn,nj->Pij", xm, av(lam), xm).reshape(naux, -1)
        return alpha * np.sum(ell * (truncated_inverse(vv(lam), cut) @ ell))

    def formula(cut):
        mm = truncated_inverse(v0, cut)
        ell = np.einsum("mi,Pmn,nj->Pij", xm, a0, xm).reshape(naux, -1)
        gam = mm @ ell
        lx = [
            np.einsum("mi,Pmn,nj->Pij", xm, a1[k], xm).reshape(naux, -1)
            for k in range(NPAR)
        ]
        z = [lx[k] - v1[k] @ gam for k in range(NPAR)]
        # A and V are linear here: L_xy = 0, V_xy = 0
        return np.array(
            [
                [2 * alpha * np.sum(z[k] * (mm @ z[ll])) for ll in range(NPAR)]
                for k in range(NPAR)
            ]
        )

    def mixed_fd(cut):
        hh = 5e-3
        st = [(-2, 1.0), (-1, -8.0), (1, 8.0), (2, -1.0)]
        out = np.zeros((NPAR, NPAR))
        for k in range(NPAR):
            for ll in range(NPAR):
                tot = 0.0
                for tk, wk in st:
                    for tl, wl in st:
                        lt = np.zeros(NPAR)
                        lt[k] += tk * hh
                        lt[ll] += tl * hh
                        tot += wk * wl * energy(lt, cut)
                out[k, ll] = tot / (12 * hh) ** 2
        return out

    full = np.max(np.abs(formula(0.0) - mixed_fd(0.0)))
    miss = np.max(np.abs(formula(0.2) - mixed_fd(0.2)))
    check(
        "T1 full-rank formula = FD with nothing dropped; misses with a dropped mode (claim T1)",
        full < 1e-6 and miss > 1e3 * full,
        f"nothing dropped {full:.1e} one dropped {miss:.1e}",
    )


# ---------------------------------------------------------------------------
# ST1, ST2: the stencil of hessian_df_stencil.rs
# ---------------------------------------------------------------------------
STENCIL = [(-2.0, 1.0), (-1.0, -8.0), (1.0, 8.0), (2.0, -1.0)]
DEN = 12.0


def stencil_checks():
    cen = [sp.Matrix(sp.symbols(f"c{k}x c{k}y c{k}z")) for k in range(3)]
    flat = [s for cc in cen for s in cc]
    pa, pb, pc = cen
    r1, r2 = pa - pb, pb - pc
    f = sp.exp(-sp.Rational(7, 10) * r1.dot(r1) - sp.Rational(2, 5) * r2.dot(r2)) * (
        1 + (pa - pc)[0] * r2[1] + sp.Rational(1, 3) * r1[2] ** 2
    )
    grad = [sp.diff(f, s) for s in flat]
    zero_sum = all(
        sp.simplify(sum(grad[3 * k + ax] for k in range(3))) == 0 for ax in range(3)
    )
    g_num = sp.lambdify(flat, grad, "numpy")
    hess = sp.lambdify(flat, [[sp.diff(gi, s) for s in flat] for gi in grad], "numpy")

    def atom_hessian_exact(pos, atom_of):
        hh = np.array(hess(*pos.ravel()))
        atoms = sorted(set(atom_of))
        out = np.zeros((len(atoms), 3, len(atoms), 3))
        for k, ak in enumerate(atom_of):
            for ll, al in enumerate(atom_of):
                out[atoms.index(ak), :, atoms.index(al), :] += hh[
                    3 * k : 3 * k + 3, 3 * ll : 3 * ll + 3
                ]
        return out

    def atom_hessian_stencil(pos, atom_of, h, single_shell=False):
        """eri3_deriv2_block: displace every centre on atom b (rigid per atom),
        fold each centre's first-derivative block onto its atom."""
        atoms = sorted(set(atom_of))
        out = np.zeros((len(atoms), 3, len(atoms), 3))
        for ib, b in enumerate(atoms):
            for y in range(3):
                for off, wt in STENCIL:
                    sh = pos.copy()
                    for k, ak in enumerate(atom_of):
                        if ak == b:
                            sh[k, y] += off * h
                            if single_shell:
                                break
                    g = np.array(g_num(*sh.ravel())).reshape(3, 3)
                    for k, ak in enumerate(atom_of):
                        out[atoms.index(ak), :, ib, y] += wt / (DEN * h) * g[k]
        return out

    pos = np.array([[0.1, -0.2, 0.3], [0.4, 0.5, -0.1], [-0.6, 0.2, 0.7]])
    atom_of = [0, 0, 1]  # aux and bra on atom 0, ket on atom 1
    pos[1] = pos[0]
    ex = atom_hessian_exact(pos, atom_of)
    e1 = np.max(np.abs(atom_hessian_stencil(pos, atom_of, 0.1) - ex))
    e2 = np.max(np.abs(atom_hessian_stencil(pos, atom_of, 0.05) - ex))
    ep = np.max(np.abs(atom_hessian_stencil(pos, atom_of, 2e-3) - ex))
    ratio = e1 / e2
    check(
        "ST1 fourth-order stencil of first derivatives: O(h^4), 1e-10 at h = 2e-3 (claim ST1)",
        14.0 < ratio < 18.0 and ep < 1e-10,
        f"ratio {ratio:.2f} err(2e-3) {ep:.1e}",
    )
    one = atom_hessian_exact(pos, [0, 0, 0])
    single = np.max(
        np.abs(atom_hessian_stencil(pos, atom_of, 2e-3, single_shell=True) - ex)
    )
    check(
        "ST2 one-atom triple has zero second derivative; one-shell displacement misses (claim ST2)",
        zero_sum and np.max(np.abs(one)) < 1e-12 and single > 1e3 * ep,
        f"one-atom block {np.max(np.abs(one)):.1e} single-shell miss {single:.1e}",
    )


def main():
    symbolic_checks()
    numeric_checks()
    truncation_check()
    stencil_checks()
    if failures:
        print(f"{len(failures)} check(s) failed")
        sys.exit(1)
    print("all fitted J/K Hessian identities verified")


if __name__ == "__main__":
    main()
