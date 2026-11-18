#!/usr/bin/env python3
"""Hartree-Fock nuclear Hessian as ferric assembles it (RHF, UHF, ROHF).

Derivation: `book/src/14-derivatives/04-hf-hessian.md#hf-hessian-equations`.
Code: crates/ferric-scf/src/hessian.rs (rhf_hessian, uhf_hessian; skeleton
terms 1-4, cphf_rhs, finish_cphf, assemble_response, MoFrame) and
crates/ferric-scf/src/hessian/rohf.rs (rohf_hessian: Response::apply, rhs,
column, w_response).

Independent route: the toy model of scripts/hf_gradient_sympy.py (integrals
quadratic in two parameters lam = (x, y), non-orthogonal S), whose analytic
gradient that script checks against finite differences of the re-solved SCF
energy. Here the analytic Hessian (skeleton + CPHF response, as coded) is
compared with the fourth-order central difference (h = 1e-3) of that
analytic gradient, the SCF re-solved at every displaced point, bar 1e-7.

Checks:
  H1  RHF:  skeleton + 4 sum (F^x - S^x eps) U^y - 2 sum S^x_oo eps^y = FD.
  H2  Exactness anchor: a closed-shell UHF (C_a = C_b) gives the RHF Hessian.
  H3  UHF:  the coupled alpha/beta CPHF (occ 1 per spin) = FD.
  H4  ROHF: the commutator-form (vc, vo, oc) response with W^y from the AO
      densities and Focks (hessian/rohf.rs) = FD.
  S1  (sympy) U = -S/2 + kappa satisfies the differentiated orthonormality.
  S2  (sympy) an occupied-occupied antisymmetric kappa leaves D^x unchanged.

Mutations (each turns the script red; exact one-occurrence substitutions):
  RHF/UHF U_oo: -S_oo/2 -> -S_oo                         (H1, H3 fail)
  response prefactor 2 occ -> occ                        (H1, H3 fail)
  ROHF W^y: drop the D_a F_b^y D_b term                  (H4 fails)
  ROHF operator: [f, kappa] -> its diagonal-gap truncation
  (f_pp - f_qq) kappa_pq, the code's DiagonalGapOperator mutant  (H4 fails)
Exits nonzero on any failed check; prints "FAIL <label>".
"""

import sys

import numpy as np
import sympy as sp

failures = []


def fail_hard(label):
    """A failed SCF is a failed check (exit 1 with a FAIL line), not a crash."""
    print(f"FAIL {label}")
    sys.exit(1)


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        failures.append(label)


def sym(m):
    return 0.5 * (m + m.T)


# ---------------------------------------------------------------------------
# Toy model: integrals quadratic in lam, exact derivatives.
# ---------------------------------------------------------------------------
N, NAUX, NPAR = 7, 12, 2


class Model:
    def __init__(self, rng):
        def symr(scale):
            return scale * sym(rng.standard_normal((N, N)))

        def br(scale):
            b = scale * rng.standard_normal((NAUX, N, N))
            return 0.5 * (b + b.transpose(0, 2, 1))

        self.h0 = np.diag(np.linspace(-6.0, 3.0, N)) + symr(0.3)
        self.h1 = [symr(0.3) for _ in range(NPAR)]
        self.s0 = np.eye(N) + symr(0.08)
        self.s1 = [symr(0.05) for _ in range(NPAR)]
        self.b0 = br(0.2)
        self.b1 = [br(0.05) for _ in range(NPAR)]
        self.e1 = rng.standard_normal(NPAR)
        self.h2, self.s2, self.b2 = {}, {}, {}
        self.e2 = np.zeros((NPAR, NPAR))
        for k in range(NPAR):
            for ll in range(k, NPAR):
                self.h2[k, ll] = self.h2[ll, k] = symr(0.2)
                self.s2[k, ll] = self.s2[ll, k] = symr(0.03)
                self.b2[k, ll] = self.b2[ll, k] = br(0.03)
                self.e2[k, ll] = self.e2[ll, k] = rng.standard_normal()

    @staticmethod
    def _val(m0, m1, m2, lam):
        out = m0.copy()
        for k in range(NPAR):
            out = out + lam[k] * m1[k]
            for ll in range(NPAR):
                out = out + 0.5 * lam[k] * lam[ll] * m2[k, ll]
        return out

    @staticmethod
    def _d1(m1, m2, lam, k):
        out = m1[k].copy()
        for ll in range(NPAR):
            out = out + lam[ll] * m2[k, ll]
        return out

    def h(self, lam):
        return self._val(self.h0, self.h1, self.h2, lam)

    def s(self, lam):
        return self._val(self.s0, self.s1, self.s2, lam)

    def b(self, lam):
        return self._val(self.b0, self.b1, self.b2, lam)

    def eri(self, lam):
        b = self.b(lam)
        return np.einsum("Pmn,Pls->mnls", b, b)

    def enn(self, lam):
        return 3.0 + self.e1 @ lam + 0.5 * lam @ self.e2 @ lam

    def dh(self, lam, k):
        return self._d1(self.h1, self.h2, lam, k)

    def ds(self, lam, k):
        return self._d1(self.s1, self.s2, lam, k)

    def deri(self, lam, k):
        b, db = self.b(lam), self._d1(self.b1, self.b2, lam, k)
        return np.einsum("Pmn,Pls->mnls", db, b) + np.einsum("Pmn,Pls->mnls", b, db)

    def denn(self, lam, k):
        return self.e1[k] + self.e2[k] @ lam

    def d2eri(self, lam, k, ll):
        b = self.b(lam)
        bk = self._d1(self.b1, self.b2, lam, k)
        bl = self._d1(self.b1, self.b2, lam, ll)
        bkl = self.b2[k, ll]

        def pair(x, y):
            return np.einsum("Pmn,Pls->mnls", x, y)

        return pair(bkl, b) + pair(bk, bl) + pair(bl, bk) + pair(b, bkl)


def jmat(g, d):
    return np.einsum("mnls,ls->mn", g, d)


def kmat(g, d):
    return np.einsum("mlns,ls->mn", g, d)


def rhf_fock(h, g, d):
    return h + jmat(g, d) - 0.5 * kmat(g, d)


def uhf_focks(h, g, da, db):
    jm = jmat(g, da + db)
    return h + jm - kmat(g, da), h + jm - kmat(g, db)


def uhf_energy(h, g, da, db):
    fa, fb = uhf_focks(h, g, da, db)
    return 0.5 * (np.sum(da * (h + fa)) + np.sum(db * (h + fb)))


def roothaan_fock(fa, fb, da, db, s):
    """Transcription of rohf.rs roothaan_fock_inner."""
    fc = 0.5 * (fa + fb)
    pc = db @ s
    po = (da - db) @ s
    pv = np.eye(s.shape[0]) - da @ s
    f = 0.5 * pc.T @ fc @ pc + 0.5 * po.T @ fc @ po + 0.5 * pv.T @ fc @ pv
    f = f + po.T @ fb @ pc + po.T @ fa @ pv + pv.T @ fc @ pc
    return f + f.T


def rohf_gradient_mo(fa_mo, fb_mo, nc, no):
    """Transcription of rohf.rs rohf_gradient_mo."""
    na = nc + no
    g = np.zeros_like(fa_mo)
    g[na:, :nc] = fa_mo[na:, :nc] + fb_mo[na:, :nc]
    g[na:, nc:na] = fa_mo[na:, nc:na]
    g[nc:na, :nc] = fb_mo[nc:na, :nc]
    return g


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
            c = np.linalg.solve(b, rhs)[:m]
        except np.linalg.LinAlgError:
            return f
        return sum(ci * fi for ci, fi in zip(c, self.f))


def gen_eigh(f, s):
    w, v = np.linalg.eigh(s)
    x = v @ np.diag(w**-0.5) @ v.T
    e, cp = np.linalg.eigh(x @ f @ x)
    return e, x @ cp


TOL = 1e-12


def solve_rhf(h, s, g, nocc, c=None):
    if c is None:
        _, c = gen_eigh(h, s)
    diis = Diis()
    for _ in range(400):
        d = 2.0 * c[:, :nocc] @ c[:, :nocc].T
        f = rhf_fock(h, g, d)
        err = f @ d @ s - s @ d @ f
        if np.max(np.abs(err)) < TOL:
            eps, c = gen_eigh(f, s)
            return c, eps
        _, c = gen_eigh(diis.extrapolate(f, err), s)
    fail_hard("RHF SCF converges on the toy model")


def solve_uhf(h, s, g, na, nb, cs=None):
    if cs is None:
        _, c = gen_eigh(h, s)
        th = 0.4
        rot = np.eye(N)
        rot[nb - 1, nb - 1] = rot[nb, nb] = np.cos(th)
        rot[nb - 1, nb], rot[nb, nb - 1] = np.sin(th), -np.sin(th)
        cs = (c, c @ rot)
    ca, cb = cs
    diis = Diis()
    for _ in range(600):
        da, db = ca[:, :na] @ ca[:, :na].T, cb[:, :nb] @ cb[:, :nb].T
        fa, fb = uhf_focks(h, g, da, db)
        ea, eb = fa @ da @ s - s @ da @ fa, fb @ db @ s - s @ db @ fb
        if max(np.max(np.abs(ea)), np.max(np.abs(eb))) < TOL:
            (epsa, ca), (epsb, cb) = gen_eigh(fa, s), gen_eigh(fb, s)
            return (ca, cb), (epsa, epsb)
        fx = diis.extrapolate(np.concatenate([fa, fb]), np.concatenate([ea, eb]))
        _, ca = gen_eigh(fx[:N], s)
        _, cb = gen_eigh(fx[N:], s)
    fail_hard("UHF SCF converges on the toy model")


def rohf_densities(c, nc, no):
    db = c[:, :nc] @ c[:, :nc].T
    return db + c[:, nc : nc + no] @ c[:, nc : nc + no].T, db


def solve_rohf(h, s, g, nc, no, c=None):
    """DIIS on R_eff with the rohf_gradient_mo error; level shift (0.5 open,
    1.0 virtual) in the current MO frame for the update only; canonicalized
    inside the c, o, v blocks at exit."""
    if c is None:
        _, c = gen_eigh(h, s)
    shift = np.zeros(N)
    shift[nc : nc + no], shift[nc + no :] = 0.5, 1.0
    diis = Diis()
    for _ in range(600):
        da, db = rohf_densities(c, nc, no)
        fa, fb = uhf_focks(h, g, da, db)
        reff = roothaan_fock(fa, fb, da, db, s)
        gm = rohf_gradient_mo(c.T @ fa @ c, c.T @ fb @ c, nc, no)
        if np.max(np.abs(gm)) < TOL:
            reff_mo = c.T @ reff @ c
            eps = np.zeros(N)
            for blk in (slice(0, nc), slice(nc, nc + no), slice(nc + no, N)):
                eps[blk], u = np.linalg.eigh(reff_mo[blk, blk])
                c[:, blk] = c[:, blk] @ u
            return c, eps
        err = s @ c @ (gm - gm.T) @ c.T @ s
        f = diis.extrapolate(reff, err) + s @ c @ np.diag(shift) @ c.T @ s
        _, c = gen_eigh(f, s)
    fail_hard("ROHF SCF converges on the toy model")


# ---------------------------------------------------------------------------
# The code's gradient assembly.
# ---------------------------------------------------------------------------
def gamma_rhf(d):
    return 0.5 * np.einsum("mn,ls->mnls", d, d) - 0.25 * np.einsum("ml,ns->mnls", d, d)


def gamma_uhf(da, db):
    d = da + db
    return 0.5 * np.einsum("mn,ls->mnls", d, d) - 0.5 * (
        np.einsum("ml,ns->mnls", da, da) + np.einsum("ml,ns->mnls", db, db)
    )


def assemble_gradient(model, lam, d, gam, w):
    """V_nn^x + sum D h^x + sum Gamma g^x - sum W S^x (oneelectron_gradient +
    twoelectron_gradient)."""
    grad = np.zeros(NPAR)
    for k in range(NPAR):
        pulay = np.sum(w * model.ds(lam, k))
        grad[k] = (
            model.denn(lam, k)
            + np.sum(d * model.dh(lam, k))
            + np.sum(gam * model.deri(lam, k))
            - pulay
        )
    return grad


def rhf_w(c, eps, nocc):
    co = c[:, :nocc]
    return 2.0 * (co * eps[:nocc]) @ co.T


def uhf_w(cs, epss, nocc):
    return sum((c[:, :n] * e[:n]) @ c[:, :n].T for c, e, n in zip(cs, epss, nocc))


def rohf_w(da, db, fa, fb):
    w = da @ fa @ da + da @ fb @ db
    return w


# State per reference: energy and analytic gradient at lam (warm-started).
def rhf_state(model, lam, nocc, guess=None):
    h, s, g = model.h(lam), model.s(lam), model.eri(lam)
    c, eps = solve_rhf(h, s, g, nocc, guess)
    d = 2.0 * c[:, :nocc] @ c[:, :nocc].T
    e = 0.5 * np.sum(d * (h + rhf_fock(h, g, d))) + model.enn(lam)
    grad = assemble_gradient(model, lam, d, gamma_rhf(d), rhf_w(c, eps, nocc))
    return e, grad, c


def uhf_state(model, lam, na, nb, guess=None):
    h, s, g = model.h(lam), model.s(lam), model.eri(lam)
    cs, epss = solve_uhf(h, s, g, na, nb, guess)
    da, db = cs[0][:, :na] @ cs[0][:, :na].T, cs[1][:, :nb] @ cs[1][:, :nb].T
    e = uhf_energy(h, g, da, db) + model.enn(lam)
    grad = assemble_gradient(
        model, lam, da + db, gamma_uhf(da, db), uhf_w(cs, epss, (na, nb))
    )
    return e, grad, cs


def rohf_state(model, lam, nc, no, guess=None):
    h, s, g = model.h(lam), model.s(lam), model.eri(lam)
    c, eps = solve_rohf(h, s, g, nc, no, guess)
    da, db = rohf_densities(c, nc, no)
    fa, fb = uhf_focks(h, g, da, db)
    e = uhf_energy(h, g, da, db) + model.enn(lam)
    w_raw = rohf_w(da, db, fa, fb)
    grad = assemble_gradient(model, lam, da + db, gamma_uhf(da, db), sym(w_raw))
    return e, grad, c


# ---------------------------------------------------------------------------
# The code's Hessian assembly.
# ---------------------------------------------------------------------------
def skeleton(model, lam, d, gam, w):
    """Terms 1-4 of hessian.rs: V_nn'' + sum D h'' + sum Gamma g'' - sum W S''."""
    out = np.zeros((NPAR, NPAR))
    for k in range(NPAR):
        for ll in range(NPAR):
            out[k, ll] = (
                model.e2[k, ll]
                + np.sum(d * model.h2[k, ll])
                + np.sum(gam * model.d2eri(lam, k, ll))
                - np.sum(w * model.s2[k, ll])
            )
    return out


def solve_dense(apply, rhs):
    """Solve A x = rhs for the linear map `apply` by building A column by column."""
    n = rhs.size
    a = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        a[:, i] = apply(e)
    return np.linalg.solve(a, rhs)


def cphf_response(model, lam, spins, kernel, first):
    """Term 5 of hessian.rs for RHF (one spin, occ 2) and UHF (two spins, occ 1).

    spins: list of (C, eps, nocc, occ). kernel(list of AO densities per spin)
    -> list of AO two-electron responses per spin. first(k) -> list of AO
    first-derivative Focks h^k + G^k_s(D) per spin (D fixed).
    Follows solve_cphf_block / cphf_rhs / finish_cphf / assemble_response.
    """
    shapes = [(N - no, no) for (_, _, no, _) in spins]
    sizes = [a * b for a, b in shapes]

    def split(vec):
        out, off = [], 0
        for shp, sz in zip(shapes, sizes):
            out.append(vec[off : off + sz].reshape(shp))
            off += sz
        return out

    def d1(us):
        """D1(U) = occ (C_v U C_o^T + h.c.) per spin."""
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
        sk = model.ds(lam, k)
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
        g_tot = kernel([a + b for a, b in zip(d_oo, d1(us))])
        uk, ek = [], []
        for (c, eps, no, _), f, s, u, gt in zip(spins, f1k, s1k, us, g_tot):
            uf = np.vstack([-0.5 * s[:no], u])
            e1 = f[:no] + c[:, :no].T @ gt @ c[:, :no]
            e1 = e1 - 0.5 * s[:no] * (eps[:no, None] + eps[None, :no])
            uk.append(uf)
            ek.append(e1)
        f1_mo.append(f1k)
        s1_mo.append(s1k)
        u_all.append(uk)
        e1_all.append(ek)

    r = np.zeros((NPAR, NPAR))
    for x in range(NPAR):
        for y in range(NPAR):
            for sp_i, (_, eps, no, occ) in enumerate(spins):
                hs = 2.0 * occ * (f1_mo[x][sp_i] - s1_mo[x][sp_i] * eps[:no])
                s_oo = s1_mo[x][sp_i][:no]
                r[x, y] += np.sum(hs * u_all[y][sp_i]) - occ * np.sum(
                    s_oo * e1_all[y][sp_i]
                )
    return r


def rhf_hessian(model, lam, c, eps, nocc):
    g = model.eri(lam)
    d = 2.0 * c[:, :nocc] @ c[:, :nocc].T
    skel = skeleton(model, lam, d, gamma_rhf(d), rhf_w(c, eps, nocc))

    def kernel(ds):
        (x,) = ds
        return [jmat(g, x) - 0.5 * kmat(g, x)]

    def first(k):
        gk = model.deri(lam, k)
        return [model.dh(lam, k) + jmat(gk, d) - 0.5 * kmat(gk, d)]

    return skel + cphf_response(model, lam, [(c, eps, nocc, 2.0)], kernel, first)


def uhf_kernel(g):
    def kernel(ds):
        jm = jmat(g, ds[0] + ds[1])
        return [jm - kmat(g, ds[0]), jm - kmat(g, ds[1])]

    return kernel


def uhf_hessian(model, lam, cs, epss, nocc):
    g = model.eri(lam)
    da, db = (c[:, :n] @ c[:, :n].T for c, n in zip(cs, nocc))
    skel = skeleton(model, lam, da + db, gamma_uhf(da, db), uhf_w(cs, epss, nocc))

    def first(k):
        gk = model.deri(lam, k)
        jk = jmat(gk, da + db)
        return [
            model.dh(lam, k) + jk - kmat(gk, da),
            model.dh(lam, k) + jk - kmat(gk, db),
        ]

    spins = [(cs[0], epss[0], nocc[0], 1.0), (cs[1], epss[1], nocc[1], 1.0)]
    return skel + cphf_response(model, lam, spins, uhf_kernel(g), first)


def rohf_hessian(model, lam, c, nc, no):
    """hessian/rohf.rs: UHF skeleton with the ROHF W, plus the commutator-form
    response in the (vc, vo, oc) rotations with W^y from AO densities/Focks."""
    h, g = model.h(lam), model.eri(lam)
    na = nc + no
    da, db = rohf_densities(c, nc, no)
    fa, fb = uhf_focks(h, g, da, db)
    skel = skeleton(model, lam, da + db, gamma_uhf(da, db), sym(rohf_w(da, db, fa, fb)))
    kernel = uhf_kernel(g)
    nvec = [np.r_[np.ones(na), np.zeros(N - na)], np.r_[np.ones(nc), np.zeros(N - nc)]]
    f_mo = [c.T @ fa @ c, c.T @ fb @ c]
    blocks = [
        (slice(na, N), slice(0, nc)),
        (slice(na, N), slice(nc, na)),
        (slice(nc, na), slice(0, nc)),
    ]

    def to_ao(m):
        return c @ m @ c.T

    def project(ma, mb):
        return np.concatenate(
            [(ma + mb)[blocks[0]].ravel(), ma[blocks[1]].ravel(), mb[blocks[2]].ravel()]
        )

    def kappa_matrix(vec):
        k = np.zeros((N, N))
        off = 0
        for rows, cols in blocks:
            nr, nc_ = rows.stop - rows.start, cols.stop - cols.start
            blk = vec[off : off + nr * nc_].reshape(nr, nc_)
            k[rows, cols] = blk
            k[cols, rows] = -blk.T
            off += nr * nc_
        return k

    def kappa_densities(k):
        return [to_ao((n[None, :] - n[:, None]) * k) for n in nvec]

    def apply(vec):
        k = kappa_matrix(vec)
        v = kernel(kappa_densities(k))
        m = [c.T @ vs @ c + f @ k - k @ f for vs, f in zip(v, f_mo)]
        return project(m[0], m[1])

    def first(k):
        gk = model.deri(lam, k)
        jk = jmat(gk, da + db)
        return [
            model.dh(lam, k) + jk - kmat(gk, da),
            model.dh(lam, k) + jk - kmat(gk, db),
        ]

    f1 = [first(k) for k in range(NPAR)]
    s1 = [model.ds(lam, k) for k in range(NPAR)]
    r = np.zeros((NPAR, NPAR))
    for y in range(NPAR):
        s_mo = c.T @ s1[y] @ c
        ds_ = [to_ao(-0.5 * s_mo * (n[:, None] + n[None, :])) for n in nvec]
        gs = kernel(ds_)
        phi = [
            c.T @ f1[y][s] @ c
            + c.T @ gs[s] @ c
            - 0.5 * (s_mo @ f_mo[s] + f_mo[s] @ s_mo)
            for s in range(2)
        ]
        kap = kappa_matrix(solve_dense(apply, -project(phi[0], phi[1])))
        u = kap - 0.5 * s_mo
        dy = [to_ao(u * n[None, :] + n[:, None] * u.T) for n in nvec]
        gy = kernel(dy)
        fy = [f1[y][s] + gy[s] for s in range(2)]
        wy = (
            dy[0] @ fa @ da + da @ fy[0] @ da + da @ fa @ dy[0]
            + dy[0] @ fb @ db + da @ fy[1] @ db + da @ fb @ dy[1]
        )  # fmt: skip
        for x in range(NPAR):
            resp = sum(np.sum(dy[s] * f1[x][s]) for s in range(2))
            r[x, y] = resp - np.sum(wy * s1[x])
    return skel + r


def fd_of_gradient(state, lam, guess):
    """H_fd[k, ll] = d grad_k / d lam_l, fourth-order stencil, SCF re-solved."""
    hh = 1e-3
    out = np.zeros((NPAR, NPAR))
    for ll in range(NPAR):
        vals = {}
        for t in (-2, -1, 1, 2):
            lt = lam.copy()
            lt[ll] += t * hh
            vals[t] = state(lt, guess)[1]
        out[:, ll] = (8 * (vals[1] - vals[-1]) - (vals[2] - vals[-2])) / (12 * hh)
    return out


def report(label, h_an, h_fd, bar):
    err = np.max(np.abs(h_an - h_fd))
    asym = np.max(np.abs(h_an - h_an.T))
    check(label, err < bar, f"max err {err:.2e} asym {asym:.2e}")


def numeric_checks():
    rng = np.random.default_rng(7)
    model = Model(rng)
    lam0 = np.array([0.3, -0.2])
    bar = 1e-7

    h_, s_, g_ = model.h(lam0), model.s(lam0), model.eri(lam0)
    c, eps = solve_rhf(h_, s_, g_, 3)
    h_rhf = rhf_hessian(model, lam0, c, eps, 3)
    fd = fd_of_gradient(lambda lt, gs: rhf_state(model, lt, 3, gs), lam0, c)
    report("H1 RHF Hessian = FD of the analytic gradient (claim H1)", h_rhf, fd, bar)

    # Exactness anchor: closed-shell UHF reproduces the RHF Hessian.
    h_cs = uhf_hessian(model, lam0, (c, c), (eps, eps), (3, 3))
    err = np.max(np.abs(h_cs - h_rhf))
    check(
        "H2 closed-shell UHF Hessian = RHF Hessian (claim H2)",
        err < 1e-10,
        f"max err {err:.2e}",
    )

    cs, epss = solve_uhf(h_, s_, g_, 4, 2)
    h_uhf = uhf_hessian(model, lam0, cs, epss, (4, 2))
    fd = fd_of_gradient(lambda lt, gs: uhf_state(model, lt, 4, 2, gs), lam0, cs)
    report("H3 UHF Hessian = FD of the analytic gradient (claim H3)", h_uhf, fd, bar)

    c, _ = solve_rohf(h_, s_, g_, 2, 2)
    h_rohf = rohf_hessian(model, lam0, c, 2, 2)
    fd = fd_of_gradient(lambda lt, gs: rohf_state(model, lt, 2, 2, gs), lam0, c)
    report("H4 ROHF Hessian = FD of the analytic gradient (claim H4)", h_rohf, fd, bar)


def symbolic_checks():
    """U^x_oo = -S^x_oo / 2 is the symmetric solution of the differentiated
    orthonormality C^T S C = 1 (U + U^T + S^x_MO = 0), and the remaining
    antisymmetric part leaves an occupied-occupied density unchanged."""
    s12, u12, u21, k = sp.symbols("s12 u12 u21 k")
    s11, s22 = sp.symbols("s11 s22")
    smo = sp.Matrix([[s11, s12], [s12, s22]])
    u = sp.Matrix([[-s11 / 2, -s12 / 2 + k], [-s12 / 2 - k, -s22 / 2]])
    check(
        "S1 U = -S/2 + kappa satisfies U + U^T + S = 0 (claim S1)",
        sp.simplify(u + u.T + smo) == sp.zeros(2),
    )
    n = sp.diag(1, 1)
    d1 = u * n + n * u.T
    check(
        "S2 occupied-occupied kappa drops out of D^x (both occupied) (claim S2)",
        sp.simplify(d1 - (-smo)) == sp.zeros(2),
    )


def main():
    symbolic_checks()
    numeric_checks()
    if failures:
        print(f"{len(failures)} check(s) failed")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
