#!/usr/bin/env python3
"""Kohn-Sham Z-vector: the orbital-relaxation term of a post-SCF quantity.

Derivation: crates/ferric-scf/theory.md, "Kohn-Sham Z-vector for post-SCF
corrections". Code: crates/ferric-scf/src/zvector_ks.rs
(relaxation_gradient_closed, relaxation_gradient_unrestricted,
relaxation_gradient_roks, rotated_projector, RoksBlocks::product,
roks_energy_weighted_density{,_derivative}).

Model: N = 5 AOs whose S, h, factorized ERIs (mn|ls) = sum_P L_Pmn L_Pls and
grid AO values phi_m(g) are linear in one parameter x; a hybrid functional
F = h + J - (c_x/2) K + V_xc with a spin-scaled LDA-type
E_xc = sum_g w_g (-c 2^(1/3)) (rho_a^(4/3) + rho_b^(4/3)), so the orbital
Hessian carries an f_xc kernel; a post-SCF quantity
Q(D; x) = Tr[A D] + q2 Tr[D B D B], V = dQ/dD = A + 2 q2 B D B. The SCF is
re-solved at displaced x for every finite difference.

As coded (D = 2 C_o C_o^T closed shell, D_s = C_s,o C_s,o^T per spin):
  RKS:  H Z = -4 V_vo,  (H x) = F_vv x - x F_oo + C_v^T dF[2(C_v x C_o^T + h.c.)] C_o,
        relaxation = 1/4 d/dt g(D(t), W + t Wdot) by a central difference,
        W = 1/2 D F D,  Wdot = 1/2 (dD F D + D F dD + D dF D),
        D(t) = 2 C_o(t) C_o(t)^T, C_o(t) = (C_o + t C_v Z)(1 + t^2 Z^T Z)^(-1/2).
  UKS:  H Z = -2 V_vo per spin, relaxation = 1/2 d/dt g(D_a(t), D_b(t), W + t Wdot),
        W = sum_s D_s F_s D_s.
  ROKS: blocks (vc, vo, oc), H Z = -(4 V_vc, 2 V_vo, 2 V_oc), relaxation
        1/2 d/dt g at the Cayley-rotated orbitals,
        W = D_a F_a D_a + P_c F_b P_c + 1/2 (P_c F_b P_o + P_o F_b P_c).
  Caller's fixed-orbital part: dQ/dx|_D + Tr[V dD_fix], dD_fix = -1/2 D S' D
  (RKS), -D_s S' D_s (UKS), and for ROKS dD_a = -D_a S' D_a,
  dD_b = -P_c S' P_c - 1/2 (P_o S' P_c + P_c S' P_o).

Checks (each label carries "(claim X)"):
  Z0  Anchor: Q = E_SCF gives V_vo = F_vo = 0, so Z = 0 and the relaxation
      vanishes; the fixed-orbital part alone is dE/dx (against the FD).
  Z1  RKS: fixed-orbital part + sum_ai Z_ai dF_ai/dx (dF_ai/dx by a central
      difference of F_ai along the explicit fixed-orbital path) equals a
      Richardson difference of Q with the SCF re-solved.
  Z2  RKS: the coded contraction 1/4 [g(+t) - g(-t)]/(2t) equals
      sum_ai Z_ai dF_ai/dx; its error is O(t^2) (halving t divides it by 4)
      with or without XC, so it is not exact for the one- and two-electron
      parts: the rotated projector D(t) has an odd t^3 term.
  Z3  UKS (2 alpha, 1 beta): fixed part + the 1/2 contraction equals the FD;
      at alpha = beta (closed-shell molecule run unrestricted) Z_s = Z_RKS / 2
      and the relaxation equals the RKS one.
  Z4  ROKS (1 closed, 1 open): fixed part (with the closed-open cross term)
      + the 1/2 contraction of the coded three-block solve equals the FD; the
      transcribed product is symmetric; dropping the cross term of the fixed
      path misses (negative control).

Tolerances: the SCFs are converged to max|F_ai| < 1e-13, so Q (not
variational) carries ~1e-13 and its Richardson difference at h = 1e-3 carries
~1e-13/1e-3 = 1e-10 plus O(h^4) ~ 1e-12. The contraction at
t max|dD| = 1e-4 carries O(t^2) ~ 1e-8 relative (measured 5e-8 with XC,
2e-9 without); its bar is 1e-6 relative. A check that sums both uses
1e-9 + 1e-6 |relaxation|.

Mutations (each turns the script red; exact one-occurrence substitutions):
  RKS right-hand side -2 V_vo instead of -4 V_vo           (Z1 fails)
  RKS contraction factor 1/2 instead of 1/4                (Z2 fails)
  Wdot without the D dF D term                             (Z2 fails)
  UKS right-hand side -4 V_vo instead of -2 V_vo           (Z3 fails)
  ROKS right-hand side 2 V_vc instead of 4 V_vc            (Z4 fails)
  ROKS Wdot without its closed-open cross term             (Z4 fails)
(The cross term of W itself vanishes at convergence, F_b,oc = 0, so a
mutation of W alone is an identity; only its derivative Wdot carries it.)
Exits nonzero on any failed check; prints "FAIL <label>".
"""

import sys

import numpy as np
import scipy.linalg as sla
import sympy as sp

failures = []


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        failures.append(label)


N, NAUX, NG = 5, 7, 14
CX = 0.25  # exact-exchange fraction
CXC = 0.3  # strength of the LDA-type term
Q2 = 0.4  # nonlinearity of Q
T_SCALE = 1e-4  # zvector_ks.rs T_SCALE
H_FD = 1e-3


def sym(m):
    return 0.5 * (m + m.T)


class Model:
    def __init__(self, seed, cxc=CXC):
        rng = np.random.default_rng(seed)
        r = lambda s=1.0: s * sym(rng.standard_normal((N, N)))  # noqa: E731
        self.s = [np.eye(N) + r(0.1), r(0.08)]
        self.h = [np.diag([-3.0, -2.4, 0.2, 0.8, 1.4]) + r(0.2), r(0.3)]
        self.l = [
            np.array([r(0.25) for _ in range(NAUX)]),
            np.array([r(0.05) for _ in range(NAUX)]),
        ]
        self.phi = [rng.uniform(0.3, 1.0, (NG, N)), 0.1 * rng.standard_normal((NG, N))]
        self.w = rng.uniform(0.2, 0.6, NG)
        self.a = [r(), r(0.5)]
        self.b = [r(0.5), r(0.3)]
        self.cxc = cxc

    def at(self, pair, x):
        return pair[0] + x * pair[1]

    # ---- integrals and their x-derivatives at fixed density ----
    def jk(self, d, x, deriv=False):
        L = self.at(self.l, x)
        if not deriv:
            c = np.einsum("pls,ls->p", L, d)
            j = np.einsum("p,pmn->mn", c, L)
            k = np.einsum("pml,ls,psn->mn", L, d, L)
            return j, k
        L1 = self.l[1]
        # d/dx of the factorized ERI is L1 L + L L1
        cj = np.einsum("pls,ls->p", L, d)
        cj1 = np.einsum("pls,ls->p", L1, d)
        j = np.einsum("p,pmn->mn", cj, L1) + np.einsum("p,pmn->mn", cj1, L)
        k = np.einsum("pml,ls,psn->mn", L1, d, L) + np.einsum(
            "pml,ls,psn->mn", L, d, L1
        )
        return j, k

    def rho(self, d, x):
        p = self.at(self.phi, x)
        return np.einsum("gm,mn,gn->g", p, d, p)

    def exc_spin(self, ra, rb):
        c = -self.cxc * 2.0 ** (1.0 / 3.0)
        e = c * (ra ** (4 / 3) + rb ** (4 / 3))
        va = c * (4 / 3) * ra ** (1 / 3)
        vb = c * (4 / 3) * rb ** (1 / 3)
        fa = c * (4 / 9) * ra ** (-2 / 3)
        fb = c * (4 / 9) * rb ** (-2 / 3)
        return e, (va, vb), (fa, fb)

    def vxc_spin(self, da, db, x):
        p = self.at(self.phi, x)
        e, (va, vb), _ = self.exc_spin(self.rho(da, x), self.rho(db, x))
        mat = lambda v: np.einsum("g,gm,gn->mn", self.w * v, p, p)  # noqa: E731
        return float(self.w @ e), mat(va), mat(vb)

    def fxc_spin(self, da, db, dda, ddb, x):
        """Response V_xc of spin-density perturbations (no alpha-beta cross term)."""
        p = self.at(self.phi, x)
        _, _, (fa, fb) = self.exc_spin(self.rho(da, x), self.rho(db, x))
        mat = lambda v: np.einsum("g,gm,gn->mn", self.w * v, p, p)  # noqa: E731
        return mat(fa * self.rho(dda, x)), mat(fb * self.rho(ddb, x))

    def exc_grad(self, da, db, x):
        """dE_xc/dx at fixed spin densities (AO-value derivative only)."""
        p, p1 = self.at(self.phi, x), self.phi[1]
        _, (va, vb), _ = self.exc_spin(self.rho(da, x), self.rho(db, x))
        dr = lambda d: 2.0 * np.einsum("gm,mn,gn->g", p1, d, p)  # noqa: E731
        return float(self.w @ (va * dr(da) + vb * dr(db)))

    # ---- spin-resolved energy, Focks and response ----
    def energy_spin(self, da, db, x):
        d = da + db
        j, _ = self.jk(d, x)
        _, ka = self.jk(da, x)
        _, kb = self.jk(db, x)
        exc, _, _ = self.vxc_spin(da, db, x)
        e = np.sum(self.at(self.h, x) * d) + 0.5 * np.sum(d * j)
        return e - 0.5 * CX * (np.sum(da * ka) + np.sum(db * kb)) + exc

    def fock_spin(self, da, db, x):
        j, _ = self.jk(da + db, x)
        _, ka = self.jk(da, x)
        _, kb = self.jk(db, x)
        _, va, vb = self.vxc_spin(da, db, x)
        h = self.at(self.h, x)
        return h + j - CX * ka + va, h + j - CX * kb + vb

    def dfock_spin(self, da, db, dda, ddb, x):
        """Response spin Focks (ResponseFock::apply_spin)."""
        j, _ = self.jk(dda + ddb, x)
        _, ka = self.jk(dda, x)
        _, kb = self.jk(ddb, x)
        fa, fb = self.fxc_spin(da, db, dda, ddb, x)
        return j - CX * ka + fa, j - CX * kb + fb

    def grad_expr_spin(self, da, db, w, x):
        """ks_gradient_uks_for_density: dE/dx at fixed (D_a, D_b) minus Tr[S' W]."""
        d = da + db
        j1, _ = self.jk(d, x, True)
        _, ka1 = self.jk(da, x, True)
        _, kb1 = self.jk(db, x, True)
        g = np.sum(self.h[1] * d) + 0.5 * np.sum(d * j1)
        g -= 0.5 * CX * (np.sum(da * ka1) + np.sum(db * kb1))
        return g + self.exc_grad(da, db, x) - np.sum(self.s[1] * w)

    # ---- the post-SCF quantity ----
    def q(self, d, x):
        b = self.at(self.b, x)
        return np.sum(self.at(self.a, x) * d) + Q2 * np.trace(d @ b @ d @ b)

    def v(self, d, x):
        b = self.at(self.b, x)
        return self.at(self.a, x) + 2.0 * Q2 * b @ d @ b

    def dq_dx(self, d, x):
        b, b1 = self.at(self.b, x), self.b[1]
        return np.sum(self.a[1] * d) + 2.0 * Q2 * np.trace(d @ b1 @ d @ b)


# ---- SCF solvers ---------------------------------------------------------------


def lowdin(c, s):
    m = c.T @ s @ c
    w, u = np.linalg.eigh(m)
    return c @ u @ np.diag(w**-0.5) @ u.T


def solve_uks(model, x, na, nb, c0=None):
    """Spin-unrestricted Roothaan iterations with DIIS; returns (C_a, C_b)."""
    s = model.at(model.s, x)
    if c0 is None:
        _, c = sla.eigh(model.at(model.h, x), s)
        ca = cb = c
    else:
        ca, cb = c0
    hist = []
    for _ in range(400):
        da, db = ca[:, :na] @ ca[:, :na].T, cb[:, :nb] @ cb[:, :nb].T
        fa, fb = model.fock_spin(da, db, x)
        err = np.concatenate(
            [(f @ dd @ s - s @ dd @ f).ravel() for f, dd in ((fa, da), (fb, db))]
        )
        hist.append((np.concatenate([fa.ravel(), fb.ravel()]), err))
        hist = hist[-8:]
        if (
            max(
                np.abs(ca[:, na:].T @ fa @ ca[:, :na]).max(),
                np.abs(cb[:, nb:].T @ fb @ cb[:, :nb]).max() if nb else 0.0,
            )
            < 1e-13
        ):
            return ca, cb
        m = len(hist)
        bm = -np.ones((m + 1, m + 1))
        bm[m, m] = 0.0
        for i in range(m):
            for k in range(m):
                bm[i, k] = hist[i][1] @ hist[k][1]
        rhs = np.zeros(m + 1)
        rhs[m] = -1.0
        coef = np.linalg.lstsq(bm, rhs, rcond=None)[0][:m]
        fvec = sum(ci * h[0] for ci, h in zip(coef, hist))
        fa, fb = fvec[: N * N].reshape(N, N), fvec[N * N :].reshape(N, N)
        _, ca = sla.eigh(sym(fa), s)
        _, cb = sla.eigh(sym(fb), s)
    raise RuntimeError("UKS did not converge")


def solve_rks(model, x, nocc, c0=None):
    ca, _ = solve_uks(model, x, nocc, nocc, None if c0 is None else (c0, c0))
    return ca


def roks_gradient_blocks(model, c, nc, no, x):
    """rohf_newton::gradient_blocks: (g_vc, g_vo, g_oc) = ((F_a+F_b)_vc, F_a,vo, F_b,oc)."""
    na = nc + no
    da, db = c[:, :na] @ c[:, :na].T, c[:, :nc] @ c[:, :nc].T
    fa, fb = model.fock_spin(da, db, x)
    fa, fb = c.T @ fa @ c, c.T @ fb @ c
    return (fa + fb)[na:, :nc], fa[na:, nc:na], fb[nc:na, :nc]


def kappa_from(blocks, nc, no):
    na = nc + no
    k = np.zeros((N, N))
    k[na:, :nc], k[na:, nc:na], k[nc:na, :nc] = blocks
    return k - k.T


def solve_roks(model, x, nc, no, c0):
    """ROKS by Newton steps on (vc, vo, oc) with an FD Jacobian of the gradient."""
    s = model.at(model.s, x)
    c = lowdin(c0, s)
    shapes = [(N - nc - no, nc), (N - nc - no, no), (no, nc)]
    sizes = [a * b for a, b in shapes]

    def unpack(v):
        out, o = [], 0
        for sh, sz in zip(shapes, sizes):
            out.append(v[o : o + sz].reshape(sh))
            o += sz
        return out

    def grad(cc, v):
        cr = cc @ sla.expm(kappa_from(unpack(v), nc, no))
        return np.concatenate(
            [b.ravel() for b in roks_gradient_blocks(model, cr, nc, no, x)]
        )

    for _ in range(60):
        g0 = grad(c, np.zeros(sum(sizes)))
        if np.abs(g0).max() < 1e-13:
            return c
        n = g0.size
        jac = np.zeros((n, n))
        eps = 1e-5
        for i in range(n):
            e = np.zeros(n)
            e[i] = eps
            jac[:, i] = (grad(c, e) - grad(c, -e)) / (2 * eps)
        step = -np.linalg.solve(jac, g0)
        if np.abs(step).max() > 0.2:
            step *= 0.2 / np.abs(step).max()
        c = c @ sla.expm(kappa_from(unpack(step), nc, no))
    raise RuntimeError("ROKS did not converge")


# ---- transcriptions of zvector_ks.rs ---------------------------------------------


def pcg_like_solve(hx, rhs):
    """The solve itself (PCG in the code): a dense solve of the same linear map."""
    shapes = [r.shape for r in rhs]
    sizes = [r.size for r in rhs]

    def unpack(v):
        out, o = [], 0
        for sh, sz in zip(shapes, sizes):
            out.append(v[o : o + sz].reshape(sh))
            o += sz
        return out

    n = sum(sizes)
    hmat = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        hmat[:, i] = np.concatenate([b.ravel() for b in hx(unpack(e))])
    sol = np.linalg.solve(hmat, np.concatenate([r.ravel() for r in rhs]))
    return unpack(sol), hmat


def rotated_projector(co, cv, z, tt, occ):
    m = np.eye(co.shape[1]) + tt * tt * z.T @ z
    w, u = np.linalg.eigh(m)
    ct = (co + tt * cv @ z) @ (u @ np.diag(w**-0.5) @ u.T)
    return occ * ct @ ct.T


def rks_relaxation(model, c, nocc, x, v_ao, t_scale=T_SCALE):
    """relaxation_gradient_closed: returns (contraction, Z)."""
    co, cv = c[:, :nocc], c[:, nocc:]
    d = 2.0 * co @ co.T
    f_ao, _ = model.fock_spin(d / 2, d / 2, x)
    f_mo = c.T @ f_ao @ c
    f_oo, f_vv = f_mo[:nocc, :nocc], f_mo[nocc:, nocc:]

    def dens(xx):
        a = cv @ xx @ co.T
        return 2.0 * (a + a.T)

    def resp(dd):
        dfa, _ = model.dfock_spin(d / 2, d / 2, 0.5 * dd, 0.5 * dd, x)
        return dfa

    def hx(blocks):
        (xx,) = blocks
        return [f_vv @ xx - xx @ f_oo + cv.T @ resp(dens(xx)) @ co]

    rhs = -4.0 * cv.T @ v_ao @ co
    (z,), hmat = pcg_like_solve(hx, [rhs])
    dd = dens(z)
    df = resp(dd)
    w_dot = 0.5 * (dd @ f_ao @ d + d @ f_ao @ dd + d @ df @ d)
    w0 = 0.5 * d @ f_ao @ d
    if np.abs(dd).max() == 0.0:
        return 0.0, z, hmat
    t = t_scale / np.abs(dd).max()

    def g_at(sgn):
        dt = rotated_projector(co, cv, z, sgn * t, 2.0)
        wt = w0 + sgn * t * w_dot
        return model.grad_expr_spin(dt / 2, dt / 2, wt, x)

    return 0.25 * (g_at(1.0) - g_at(-1.0)) / (2.0 * t), z, hmat


def uks_relaxation(model, ca, cb, na, nb, x, v_ao):
    """relaxation_gradient_unrestricted."""
    blocks = []
    for c, n in ((ca, na), (cb, nb)):
        co, cv = c[:, :n], c[:, n:]
        blocks.append((co, cv, co @ co.T))
    da, db = blocks[0][2], blocks[1][2]
    fa, fb = model.fock_spin(da, db, x)
    focks = (fa, fb)
    fmo = [(cv.T @ f @ cv, co.T @ f @ co) for (co, cv, _), f in zip(blocks, focks)]

    def dens(k, xx):
        co, cv, _ = blocks[k]
        a = cv @ xx @ co.T
        return a + a.T

    def hx(xs):
        dfa, dfb = model.dfock_spin(da, db, dens(0, xs[0]), dens(1, xs[1]), x)
        out = []
        for k, (xx, df) in enumerate(zip(xs, (dfa, dfb))):
            co, cv, _ = blocks[k]
            fvv, foo = fmo[k]
            out.append(fvv @ xx - xx @ foo + cv.T @ df @ co)
        return out

    rhs = [-2.0 * cv.T @ v_ao @ co for co, cv, _ in blocks]
    z, _ = pcg_like_solve(hx, rhs)
    dda, ddb = dens(0, z[0]), dens(1, z[1])
    dfa, dfb = model.dfock_spin(da, db, dda, ddb, x)
    w_dot = sum(
        dd @ f @ dsp + dsp @ f @ dd + dsp @ df @ dsp
        for dd, f, dsp, df in ((dda, fa, da, dfa), (ddb, fb, db, dfb))
    )
    w0 = da @ fa @ da + db @ fb @ db
    t = T_SCALE / max(np.abs(dda).max(), np.abs(ddb).max())

    def g_at(sgn):
        pa = rotated_projector(blocks[0][0], blocks[0][1], z[0], sgn * t, 1.0)
        pb = rotated_projector(blocks[1][0], blocks[1][1], z[1], sgn * t, 1.0)
        return model.grad_expr_spin(pa, pb, w0 + sgn * t * w_dot, x)

    return 0.5 * (g_at(1.0) - g_at(-1.0)) / (2.0 * t), z


def roks_w(p_c, p_o, f_a, f_b):
    """roks_energy_weighted_density."""
    d_a = p_c + p_o
    cross = p_c @ f_b @ p_o
    return d_a @ f_a @ d_a + p_c @ f_b @ p_c + 0.5 * (cross + cross.T)


def roks_w_dot(p_c, p_o, f_a, f_b, dd_a, dd_b, df_a, df_b):
    """roks_energy_weighted_density_derivative."""
    d_a = p_c + p_o
    dp_c, dp_o = dd_b, dd_a - dd_b
    alpha = dd_a @ f_a @ d_a + d_a @ f_a @ dd_a + d_a @ df_a @ d_a
    closed = dp_c @ f_b @ p_c + p_c @ f_b @ dp_c + p_c @ df_b @ p_c
    cross = dp_c @ f_b @ p_o + p_c @ df_b @ p_o + p_c @ f_b @ dp_o
    return alpha + closed + 0.5 * (cross + cross.T)


def cayley(kappa, tt):
    eye = np.eye(kappa.shape[0])
    return np.linalg.solve(eye - 0.5 * tt * kappa, eye + 0.5 * tt * kappa)


def roks_relaxation(model, c, nc, no, x, v_ao):
    """relaxation_gradient_roks: returns (contraction, Z blocks, Hessian matrix)."""
    na = nc + no
    cc, co, cv = c[:, :nc], c[:, nc:na], c[:, na:]
    p_c, p_o = cc @ cc.T, co @ co.T
    d_a, d_b = p_c + p_o, p_c
    f_a, f_b = model.fock_spin(d_a, d_b, x)
    fa, fb = c.T @ f_a @ c, c.T @ f_b @ c
    fs = fa + fb
    rc, ro, rv = slice(0, nc), slice(nc, na), slice(na, N)
    fs_vv, fs_cc = fs[rv, rv], fs[rc, rc]
    fa_vv, fa_oo, fa_oc = fa[rv, rv], fa[ro, ro], fa[ro, rc]
    fb_oo, fb_cc, fb_vo = fb[ro, ro], fb[rc, rc], fb[rv, ro]
    g_vc = 0.5 * (fb[rv, rc] - fa[rv, rc])

    def densities(xs):
        vc = cv @ xs[0] @ cc.T
        vo = cv @ xs[1] @ co.T
        oc = co @ xs[2] @ cc.T
        a, b = vc + vo, vc + oc
        return a + a.T, b + b.T

    def product(xs, df_a, df_b):
        x_vc, x_vo, x_oc = xs
        df_s = df_a + df_b
        h_vc = (
            fs_vv @ x_vc - x_vc @ fs_cc + fb_vo @ x_oc - x_vo @ fa_oc + cv.T @ df_s @ cc
        )
        h_vo = (
            fa_vv @ x_vo
            - x_vo @ fa_oo
            + g_vc @ x_oc.T
            - x_vc @ fa_oc.T
            + cv.T @ df_a @ co
        )
        h_oc = (
            fb_oo @ x_oc
            - x_oc @ fb_cc
            + fb_vo.T @ x_vc
            + x_vo.T @ g_vc
            + co.T @ df_b @ cc
        )
        return [h_vc, h_vo, h_oc]

    def hx(xs):
        dda, ddb = densities(xs)
        dfa, dfb = model.dfock_spin(d_a, d_b, dda, ddb, x)
        return product(xs, dfa, dfb)

    rhs = [-4.0 * cv.T @ v_ao @ cc, -2.0 * cv.T @ v_ao @ co, -2.0 * co.T @ v_ao @ cc]
    z, hmat = pcg_like_solve(hx, rhs)
    dda, ddb = densities(z)
    dfa, dfb = model.dfock_spin(d_a, d_b, dda, ddb, x)
    w0 = roks_w(p_c, p_o, f_a, f_b)
    w_dot = roks_w_dot(p_c, p_o, f_a, f_b, dda, ddb, dfa, dfb)
    t = T_SCALE / max(np.abs(dda).max(), np.abs(ddb).max())
    kappa = kappa_from(z, nc, no)

    def g_at(sgn):
        ct = c @ cayley(kappa, sgn * t)
        da_t, db_t = ct[:, :na] @ ct[:, :na].T, ct[:, :nc] @ ct[:, :nc].T
        return model.grad_expr_spin(da_t, db_t, w0 + sgn * t * w_dot, x)

    return 0.5 * (g_at(1.0) - g_at(-1.0)) / (2.0 * t), z, hmat


# ---- independent routes ------------------------------------------------------------


def richardson(f, h=H_FD):
    return (8.0 * (f(h) - f(-h)) - (f(2 * h) - f(-2 * h))) / (12.0 * h)


def fixed_path_orbitals(model, c0, nocc, x):
    """Occupied: Loewdin of C_o alone (dC_o = -1/2 C_o S'_oo); virtual: projected
    off the occupied space in the S(x) metric, then Loewdin."""
    s = model.at(model.s, x)
    co = lowdin(c0[:, :nocc], s)
    cv = c0[:, nocc:] - co @ (co.T @ s @ c0[:, nocc:])
    return co, lowdin(cv, s)


def rks_dfai_dx(model, c0, nocc, x0):
    def f_ai(h):
        co, cv = fixed_path_orbitals(model, c0, nocc, x0 + h)
        d = 2.0 * co @ co.T
        f, _ = model.fock_spin(d / 2, d / 2, x0 + h)
        return cv.T @ f @ co

    return richardson(f_ai)


def rks_fixed_part(model, c, nocc, x):
    d = 2.0 * c[:, :nocc] @ c[:, :nocc].T
    dd_fix = -0.5 * d @ model.s[1] @ d
    return model.dq_dx(d, x) + np.sum(model.v(d, x) * dd_fix)


NOCC = 2


def bar(relax):
    """FD floor (1e-10, docstring) plus the contraction's O(t^2) part (1e-6 relative)."""
    return 1e-9 + 1e-6 * abs(relax)


def claim_z0(model):
    c = solve_rks(model, 0.0, NOCC)
    d = 2.0 * c[:, :NOCC] @ c[:, :NOCC].T
    f, _ = model.fock_spin(d / 2, d / 2, 0.0)
    rel, z, _ = rks_relaxation(model, c, NOCC, 0.0, f)
    w = 0.5 * d @ f @ d
    g = model.grad_expr_spin(d / 2, d / 2, w, 0.0)

    def e_of(h):
        cc = solve_rks(model, h, NOCC, c)
        dd = 2.0 * cc[:, :NOCC] @ cc[:, :NOCC].T
        return model.energy_spin(dd / 2, dd / 2, h)

    fd = richardson(e_of)
    check(
        "Z0 anchor Q = E_SCF: Z = 0, no relaxation, fixed part = dE/dx (claim Z0)",
        np.abs(z).max() < 1e-11 and abs(rel) < 1e-11 and abs(g - fd) < 1e-8,
        f"max|Z| {np.abs(z).max():.1e} relax {abs(rel):.1e} |g - FD| {abs(g - fd):.1e}",
    )


def claim_z1_z2(model, tag):
    c = solve_rks(model, 0.0, NOCC)
    d = 2.0 * c[:, :NOCC] @ c[:, :NOCC].T
    v_ao = model.v(d, 0.0)
    rel, z, hmat = rks_relaxation(model, c, NOCC, 0.0, v_ao)
    explicit = float(np.sum(z * rks_dfai_dx(model, c, NOCC, 0.0)))
    fixed = rks_fixed_part(model, c, NOCC, 0.0)

    def q_of(h):
        cc = solve_rks(model, h, NOCC, c)
        return model.q(2.0 * cc[:, :NOCC] @ cc[:, :NOCC].T, h)

    fd = richardson(q_of)
    sym_err = np.abs(hmat - hmat.T).max()
    check(
        f"Z1 RKS{tag}: fixed part + Z.dF_ai/dx = FD of the re-solved Q (claim Z1)",
        abs(fixed + explicit - fd) < 1e-8 and abs(explicit) > 1e-3 and sym_err < 1e-12,
        f"|err| {abs(fixed + explicit - fd):.1e} relax {explicit:.3e} |H - H^T| {sym_err:.1e}",
    )
    rel_h, _, _ = rks_relaxation(model, c, NOCC, 0.0, v_ao, T_SCALE / 2)
    e1, e2 = abs(rel - explicit), abs(rel_h - explicit)
    order = np.log2(e1 / e2) if e2 > 0 else float("inf")
    check(
        f"Z2 RKS{tag}: 1/4 [g(+t) - g(-t)]/(2t) = Z.dF_ai/dx, error O(t^2) (claim Z2)",
        e1 < 1e-6 * abs(explicit) and 1.6 < order < 2.4,
        f"rel err {e1 / abs(explicit):.1e} at t, {e2 / abs(explicit):.1e} at t/2, order {order:.2f}",
    )


def claim_z3(model):
    na, nb = 2, 1
    ca, cb = solve_uks(model, 0.0, na, nb)
    d = ca[:, :na] @ ca[:, :na].T + cb[:, :nb] @ cb[:, :nb].T
    v_ao = model.v(d, 0.0)
    rel, z = uks_relaxation(model, ca, cb, na, nb, 0.0, v_ao)
    dd_fix = sum(
        -p @ model.s[1] @ p
        for p in (ca[:, :na] @ ca[:, :na].T, cb[:, :nb] @ cb[:, :nb].T)
    )
    fixed = model.dq_dx(d, 0.0) + np.sum(v_ao * dd_fix)

    def q_of(h):
        a, b = solve_uks(model, h, na, nb, (ca, cb))
        return model.q(a[:, :na] @ a[:, :na].T + b[:, :nb] @ b[:, :nb].T, h)

    fd = richardson(q_of)
    # closed-shell molecule run unrestricted
    c = solve_rks(model, 0.0, NOCC)
    dc = 2.0 * c[:, :NOCC] @ c[:, :NOCC].T
    rel_r, z_r, _ = rks_relaxation(model, c, NOCC, 0.0, model.v(dc, 0.0))
    rel_u, z_u = uks_relaxation(model, c, c, NOCC, NOCC, 0.0, model.v(dc, 0.0))
    lim = max(np.abs(z_u[0] - z_r / 2).max(), np.abs(z_u[1] - z_r / 2).max())
    check(
        "Z3 UKS: fixed part + 1/2 contraction = FD; alpha = beta gives Z_RKS/2 (claim Z3)",
        abs(fixed + rel - fd) < bar(rel)
        and abs(rel) > 1e-3
        and lim < 1e-10
        and abs(rel_u - rel_r) < 1e-6 * abs(rel_r),
        f"|err| {abs(fixed + rel - fd):.1e} relax {rel:.3e} |Z_u - Z_r/2| {lim:.1e} |dR| {abs(rel_u - rel_r):.1e}",
    )


def claim_z4(model):
    nc, no = 1, 1
    na = nc + no
    ca, _ = solve_uks(model, 0.0, na, nc)
    c = solve_roks(model, 0.0, nc, no, ca)
    p_c, p_o = c[:, :nc] @ c[:, :nc].T, c[:, nc:na] @ c[:, nc:na].T
    d = 2.0 * p_c + p_o
    v_ao = model.v(d, 0.0)
    rel, z, hmat = roks_relaxation(model, c, nc, no, 0.0, v_ao)
    s1 = model.s[1]
    d_a = p_c + p_o
    dd_a = -d_a @ s1 @ d_a
    dd_b = -p_c @ s1 @ p_c - 0.5 * (p_o @ s1 @ p_c + p_c @ s1 @ p_o)
    fixed = model.dq_dx(d, 0.0) + np.sum(v_ao * (dd_a + dd_b))
    fixed_ufm = model.dq_dx(d, 0.0) + np.sum(v_ao * (dd_a - p_c @ s1 @ p_c))

    def q_of(h):
        cc = solve_roks(model, h, nc, no, c)
        return model.q(cc[:, :na] @ cc[:, :na].T + cc[:, :nc] @ cc[:, :nc].T, h)

    fd = richardson(q_of)
    sym_err = np.abs(hmat - hmat.T).max()
    check(
        "Z4 ROKS: fixed part (closed-open cross term) + 1/2 contraction = FD (claim Z4)",
        abs(fixed + rel - fd) < bar(rel)
        and abs(rel) > 1e-3
        and sym_err < 1e-10
        and abs(fixed_ufm + rel - fd) > 1e-5,
        f"|err| {abs(fixed + rel - fd):.1e} relax {rel:.3e} |H - H^T| {sym_err:.1e} "
        f"without cross term {abs(fixed_ufm + rel - fd):.1e}",
    )


def claim_symbols():
    """The closed-shell factor 4: dD = 2 C (kappa + kappa^T) C^T gives dQ = 4 kappa.V."""
    k, v_ai, v_ia = sp.symbols("kappa V_ai V_ia")
    # one occupied i, one virtual a, orthonormal MOs: D_MO = 2 e_i e_i^T
    dd = 2 * sp.Matrix([[0, k], [k, 0]])  # rows (i, a)
    v = sp.Matrix([[0, v_ia], [v_ia, 0]]).subs(v_ia, v_ai)
    dq = sp.expand((dd.T * v).trace())
    check(
        "Z0 symbols: Tr[V dD] = 4 kappa V_ai closed shell, 2 per spin (claim Z0)",
        sp.simplify(dq - 4 * k * v_ai) == 0
        and sp.simplify((sp.Matrix([[0, k], [k, 0]]).T * v).trace() - 2 * k * v_ai)
        == 0,
        f"{dq}",
    )


def main():
    model = Model(11)
    claim_symbols()
    claim_z0(model)
    claim_z1_z2(model, "")
    claim_z1_z2(Model(11, cxc=0.0), " (no XC)")
    claim_z3(model)
    claim_z4(model)
    if failures:
        print(f"{len(failures)} check(s) failed")
        sys.exit(1)
    print("all KS Z-vector identities verified")


if __name__ == "__main__":
    main()
