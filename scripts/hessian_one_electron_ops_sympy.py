#!/usr/bin/env python3
"""SCF Hessian terms of one-electron operators beyond the nuclear attraction.

Derivation: `book/src/14-derivatives/04-hf-hessian.md#hf-hessian-one-electron-operators`.
Code: crates/ferric-scf/src/hessian.rs (extra_skeleton, add_extra_first_order,
nuclear_term, contract_external_deriv2), crates/ferric-scf/src/hessian_ecp.rs
(ecp_skeleton_hessian, moving_triples, folded_first_derivative),
crates/ferric-scf/src/hessian_field.rs (field_skeleton_hessian, PairEnergy,
field_first_derivative_matrices), crates/ferric-scf/src/hessian_smeared.rs
(smeared_skeleton_hessian, pair_stencil), crates/ferric-core/src/
external_potential.rs (charge_nuclear_hessian, smeared_radial_derivs).

Model: the closed-shell toy of scripts/hf_hessian_sympy.py (integrals smooth in
two parameters lam, non-orthogonal S, RHF re-solved at every point) plus a
one-electron operator V(lam) in h (sin/cos/product dependence, so its second
derivative is not constant) and a classical external term: one nucleus at
R0 + (lam0, lam1, 0) in the field of a point charge, a Gaussian-smeared charge
and a uniform field.

Independent route: a fourth-order central difference (h = 1e-3) of the
analytic gradient, SCF re-solved at every displaced point.

Checks (each label carries "(claim X)"):
  O0  Exactness anchor: V = 0, no charges, no field reproduces the plain RHF
      Hessian bit for bit.
  O1  skeleton + Tr[D V_xy] + classical Hessian, with V_x in F^x of the CPHF
      right-hand side, equals the FD of the analytic gradient. Negative
      control: dropping V_x from F^x misses by far more than the agreement.
  O2  The skeleton Tr[D V_xy] from a fourth-order stencil (step 2e-3) of the
      analytic first derivative Tr[D V_y] matches the exact one, and the
      Hessian built with it still matches the FD.
  O3  (sympy) field-nucleus energy -Z E.R is linear in R (zero Hessian); the
      Hessian of f(|R - r|) is f'' d d^T/R^2 + (f'/R)(1 - d d^T/R^2); with
      f = 1/R it is the point-charge block Z q (3 d d^T/R^5 - 1/R^3); the
      code's f', f'' of f = erf(R/w)/R are the derivatives; small w tends to
      the point charge.
  O4  (sympy) the stencils D1 = (1,-8,8,-1)/(12h), D2 = (-1,16,-30,16,-1)/(12h^2)
      and D1 x D1/(144 h^2) have O(h^4) error, weights summing to zero; D2 and
      D1 x D1 give exactly zero on a function linear in the displaced
      coordinates.
  O5  Rigid-per-atom mask (hessian_ecp.rs): on a 1-D model with three-centre
      Gaussian integrals (bra, ket, potential centre on atoms), the stencil of
      the folded first derivatives over only the triples with one or two
      centres on the displaced atom equals the exact Hessian; the triples with
      none or all three centres on it contribute zero.
  O6  (sympy) field same-atom block M_AA = R_A S_AA + <mu|x - R_A|nu> is linear
      in R_A (second derivative zero, first derivative S_AA); the code's pair
      stencil (hessian_field.rs PairEnergy, same-atom blocks skipped) equals
      the exact field skeleton on a 1-D model.

Mutations (each turns the script red; exact one-occurrence substitutions):
  V_x dropped from F^x                                  (O1 fails)
  point-charge block 3 d d^T -> 2 d d^T                  (O1, O3 fail)
  mask keeps only triples with one centre on the atom   (O5 fails)
  displacement moves only the bra centre on the atom    (O5 fails)
Exits nonzero on any failed check; prints "FAIL <label>".
"""

import math
import sys

import numpy as np
import sympy as sp

failures = []


def fail_hard(label):
    print(f"FAIL {label}")
    sys.exit(1)


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        failures.append(label)


def sym(m):
    return 0.5 * (m + m.T)


# Constants as in the Rust (hessian_ecp.rs, hessian_field.rs, hessian_smeared.rs).
STEP = 2e-3
D1 = [(-2.0, 1.0), (-1.0, -8.0), (1.0, 8.0), (2.0, -1.0)]
D2 = [(-2.0, -1.0), (-1.0, 16.0), (0.0, -30.0), (1.0, 16.0), (2.0, -1.0)]

# ---------------------------------------------------------------------------
# Toy SCF model (as scripts/hf_hessian_sympy.py), RHF only.
# ---------------------------------------------------------------------------
N, NAUX, NPAR, NOCC = 7, 12, 2, 3


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

    def eri(self, lam):
        b = self._val(self.b0, self.b1, self.b2, lam)
        return np.einsum("Pmn,Pls->mnls", b, b)

    def enn(self, lam):
        return 3.0 + self.e1 @ lam + 0.5 * lam @ self.e2 @ lam

    def dh(self, lam, k):
        return self._d1(self.h1, self.h2, lam, k)

    def ds(self, lam, k):
        return self._d1(self.s1, self.s2, lam, k)

    def deri(self, lam, k):
        b = self._val(self.b0, self.b1, self.b2, lam)
        db = self._d1(self.b1, self.b2, lam, k)
        return np.einsum("Pmn,Pls->mnls", db, b) + np.einsum("Pmn,Pls->mnls", b, db)

    def denn(self, lam, k):
        return self.e1[k] + self.e2[k] @ lam

    def d2eri(self, lam, k, ll):
        b = self._val(self.b0, self.b1, self.b2, lam)
        bk = self._d1(self.b1, self.b2, lam, k)
        bl = self._d1(self.b1, self.b2, lam, ll)

        def pair(x, y):
            return np.einsum("Pmn,Pls->mnls", x, y)

        bkl = self.b2[k, ll]
        return pair(bkl, b) + pair(bk, bl) + pair(bl, bk) + pair(b, bkl)


class OneElectronOp:
    """V(lam) = c [V0 + sin(lam0) V1 + (1 - cos(lam1)) V2 + lam0 lam1 V3]."""

    def __init__(self, rng, scale):
        self.c = scale
        self.m = [sym(rng.standard_normal((N, N))) for _ in range(4)]

    def v(self, lam):
        m = self.m
        return self.c * (
            m[0]
            + math.sin(lam[0]) * m[1]
            + (1.0 - math.cos(lam[1])) * m[2]
            + lam[0] * lam[1] * m[3]
        )

    def dv(self, lam, k):
        m = self.m
        if k == 0:
            return self.c * (math.cos(lam[0]) * m[1] + lam[1] * m[3])
        return self.c * (math.sin(lam[1]) * m[2] + lam[0] * m[3])

    def d2v(self, lam, k, ll):
        m = self.m
        if (k, ll) == (0, 0):
            return self.c * (-math.sin(lam[0]) * m[1])
        if (k, ll) == (1, 1):
            return self.c * (math.cos(lam[1]) * m[2])
        return self.c * m[3]


class Classical:
    """Charge-nucleus and field-nucleus energy of one nucleus (charge Z) at
    R0 + (lam0, lam1, 0): Z q / |R - r_q| + Z q_s erf(|R - r_s|/w)/|R - r_s|
    - Z E.R (external_potential.rs)."""

    def __init__(self, z, q, rq, qs, rs, w, field):
        self.z, self.q, self.rq = z, q, np.array(rq)
        self.qs, self.rs, self.w = qs, np.array(rs), w
        self.field = np.array(field)
        self.r0 = np.array([0.1, -0.3, 0.2])

    def pos(self, lam):
        return self.r0 + np.array([lam[0], lam[1], 0.0])

    def smeared_radial(self, r):
        """smeared_radial_derivs: f', f'' of f(R) = erf(R/w)/R."""
        w = self.w
        g = math.erf(r / w)
        gp = 2.0 / (math.sqrt(math.pi) * w) * math.exp(-(r * r) / (w * w))
        f1 = gp / r - g / (r * r)
        f2 = -gp * (2.0 / (w * w) + 2.0 / (r * r)) + 2.0 * g / (r * r * r)
        return f1, f2

    def energy(self, lam):
        rr = self.pos(lam)
        rq = np.linalg.norm(rr - self.rq)
        rs = np.linalg.norm(rr - self.rs)
        e = self.z * self.q / rq + self.z * self.qs * math.erf(rs / self.w) / rs
        return e - self.z * self.field @ rr

    def gradient(self, lam):
        """charge_nuclear_gradient + field_nuclear_gradient, projected on lam."""
        rr = self.pos(lam)
        d = rr - self.rq
        r = np.linalg.norm(d)
        g = -self.z * self.q * d / r**3
        d = rr - self.rs
        r = np.linalg.norm(d)
        f1, _ = self.smeared_radial(r)
        g = g + self.z * self.qs * f1 * d / r
        g = g - self.z * self.field
        return g[:NPAR]

    def hessian(self, lam):
        """charge_nuclear_hessian (the field adds nothing), projected on lam."""
        rr = self.pos(lam)
        h = np.zeros((3, 3))
        d = rr - self.rq
        r2 = d @ d
        r3 = r2 * math.sqrt(r2)
        r5 = r3 * r2
        h += self.z * self.q * (3.0 * np.outer(d, d) / r5 - np.eye(3) / r3)
        d = rr - self.rs
        r2 = d @ d
        f1, f2 = self.smeared_radial(math.sqrt(r2))
        dd = np.outer(d, d)
        a, b = f2 / r2, f1 / math.sqrt(r2) / r2
        h += self.z * self.qs * (a * dd + np.eye(3) * f1 / math.sqrt(r2) - b * dd)
        return h[:NPAR, :NPAR]


def jmat(g, d):
    return np.einsum("mnls,ls->mn", g, d)


def kmat(g, d):
    return np.einsum("mlns,ls->mn", g, d)


def g_rhf(g, d):
    return jmat(g, d) - 0.5 * kmat(g, d)


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


def solve_rhf(h, s, g, c=None):
    if c is None:
        _, c = gen_eigh(h, s)
    diis = Diis()
    for _ in range(400):
        d = 2.0 * c[:, :NOCC] @ c[:, :NOCC].T
        f = h + g_rhf(g, d)
        err = f @ d @ s - s @ d @ f
        if np.max(np.abs(err)) < 1e-12:
            eps, c = gen_eigh(f, s)
            return c, eps
        _, c = gen_eigh(diis.extrapolate(f, err), s)
    fail_hard("RHF SCF converges on the toy model")


class System:
    """The toy SCF with optional operator V and classical term."""

    def __init__(self, model, op=None, cls=None):
        self.model, self.op, self.cls = model, op, cls

    def hcore(self, lam):
        h = self.model.h(lam)
        return h + self.op.v(lam) if self.op else h

    def state(self, lam, guess=None):
        """Energy, analytic gradient, orbitals, orbital energies."""
        m = self.model
        h, s, g = self.hcore(lam), m.s(lam), m.eri(lam)
        c, eps = solve_rhf(h, s, g, guess)
        co = c[:, :NOCC]
        d = 2.0 * co @ co.T
        w = 2.0 * (co * eps[:NOCC]) @ co.T
        gam = 0.5 * np.einsum("mn,ls->mnls", d, d) - 0.25 * np.einsum(
            "ml,ns->mnls", d, d
        )
        e = 0.5 * np.sum(d * (2.0 * h + g_rhf(g, d))) + m.enn(lam)
        grad = np.zeros(NPAR)
        for k in range(NPAR):
            grad[k] = (
                m.denn(lam, k)
                + np.sum(d * m.dh(lam, k))
                + np.sum(gam * m.deri(lam, k))
                - np.sum(w * m.ds(lam, k))
            )
            if self.op:
                grad[k] += np.sum(d * self.op.dv(lam, k))
        if self.cls:
            e += self.cls.energy(lam)
            grad = grad + self.cls.gradient(lam)
        return e, grad, c, eps

    def skeleton_v_stencil(self, lam, d):
        """Sum D V_xy by the fourth-order stencil of the analytic first
        derivative Sum D V_x (hessian_ecp.rs / hessian_smeared.rs route)."""
        out = np.zeros((NPAR, NPAR))
        for y in range(NPAR):
            for off, wt in D1:
                lt = lam.copy()
                lt[y] += off * STEP
                for x in range(NPAR):
                    out[x, y] += wt / (12.0 * STEP) * np.sum(d * self.op.dv(lt, x))
        return 0.5 * (out + out.T)

    def hessian(self, lam, c, eps, vx_in_fx=True, stencil_skeleton=False):
        m = self.model
        g = m.eri(lam)
        co, cv = c[:, :NOCC], c[:, NOCC:]
        d = 2.0 * co @ co.T
        w = 2.0 * (co * eps[:NOCC]) @ co.T
        gam = 0.5 * np.einsum("mn,ls->mnls", d, d) - 0.25 * np.einsum(
            "ml,ns->mnls", d, d
        )
        # Skeleton: terms 1-4, plus the operator and classical terms.
        skel = np.zeros((NPAR, NPAR))
        for k in range(NPAR):
            for ll in range(NPAR):
                skel[k, ll] = (
                    m.e2[k, ll]
                    + np.sum(d * m.h2[k, ll])
                    + np.sum(gam * m.d2eri(lam, k, ll))
                    - np.sum(w * m.s2[k, ll])
                )
        if self.op:
            if stencil_skeleton:
                skel += self.skeleton_v_stencil(lam, d)
            else:
                for k in range(NPAR):
                    for ll in range(NPAR):
                        skel[k, ll] += np.sum(d * self.op.d2v(lam, k, ll))
        if self.cls:
            skel += self.cls.hessian(lam)

        # First-order Fock matrices at fixed D (add_extra_first_order).
        def first(k):
            gk = m.deri(lam, k)
            f = m.dh(lam, k) + g_rhf(gk, d)
            if self.op and vx_in_fx:
                f = f + self.op.dv(lam, k)
            return f

        nv = N - NOCC
        denom = eps[NOCC:, None] - eps[None, :NOCC]

        def d1(u):
            half = cv @ u @ co.T
            return 2.0 * (half + half.T)

        amat = np.zeros((nv * NOCC, nv * NOCC))
        for i in range(nv * NOCC):
            e = np.zeros(nv * NOCC)
            e[i] = 1.0
            u = e.reshape(nv, NOCC)
            amat[:, i] = (denom * u + cv.T @ g_rhf(g, d1(u)) @ co).ravel()

        f1_mo, s1_mo, u_all, e1_all = [], [], [], []
        for k in range(NPAR):
            f1 = c.T @ first(k) @ co
            s1 = c.T @ m.ds(lam, k) @ co
            d_oo = -2.0 * co @ s1[:NOCC] @ co.T
            rhs = s1[NOCC:] * eps[:NOCC] - f1[NOCC:] - cv.T @ g_rhf(g, d_oo) @ co
            u = np.linalg.solve(amat, rhs.ravel()).reshape(nv, NOCC)
            gt = g_rhf(g, d_oo + d1(u))
            e1 = f1[:NOCC] + co.T @ gt @ co
            e1 = e1 - 0.5 * s1[:NOCC] * (eps[:NOCC, None] + eps[None, :NOCC])
            f1_mo.append(f1)
            s1_mo.append(s1)
            u_all.append(np.vstack([-0.5 * s1[:NOCC], u]))
            e1_all.append(e1)
        r = np.zeros((NPAR, NPAR))
        for x in range(NPAR):
            hs = 4.0 * (f1_mo[x] - s1_mo[x] * eps[:NOCC])
            for y in range(NPAR):
                r[x, y] = np.sum(hs * u_all[y]) - 2.0 * np.sum(
                    s1_mo[x][:NOCC] * e1_all[y]
                )
        return skel + 0.5 * (r + r.T)


def fd_of_gradient(system, lam, guess):
    hh = 1e-3
    out = np.zeros((NPAR, NPAR))
    for ll in range(NPAR):
        vals = {}
        for t in (-2, -1, 1, 2):
            lt = lam.copy()
            lt[ll] += t * hh
            vals[t] = system.state(lt, guess)[1]
        out[:, ll] = (8 * (vals[1] - vals[-1]) - (vals[2] - vals[-2])) / (12 * hh)
    return out


def scf_checks():
    rng = np.random.default_rng(7)
    model = Model(rng)
    op = OneElectronOp(rng, 0.3)
    cls = Classical(
        z=3.0,
        q=-0.8,
        rq=[2.1, 1.3, -1.7],
        qs=0.6,
        rs=[-1.4, 2.2, 0.9],
        w=1.1,
        field=[0.02, -0.03, 0.05],
    )
    lam0 = np.array([0.3, -0.2])

    # O0: the trivial limit of every new term reproduces the plain Hessian.
    plain = System(model)
    _, _, c, eps = plain.state(lam0)
    h_plain = plain.hessian(lam0, c, eps)
    zero_op = OneElectronOp(np.random.default_rng(1), 0.0)
    zero_cls = Classical(0.0, 0.0, [2.0, 0, 0], 0.0, [0, 2.0, 0], 1.0, [0.0] * 3)
    zero = System(model, zero_op, zero_cls)
    _, _, cz, epsz = zero.state(lam0)
    h_zero = zero.hessian(lam0, cz, epsz)
    dz = np.max(np.abs(h_zero - h_plain))
    check(
        "O0 V = 0, no charges, no field gives the plain RHF Hessian (claim O0)",
        dz == 0.0,
        f"max diff {dz:.1e}",
    )

    # O1: analytic Hessian with the operator and the classical term = FD.
    full = System(model, op, cls)
    _, _, c, eps = full.state(lam0)
    h_an = full.hessian(lam0, c, eps)
    fd = fd_of_gradient(full, lam0, c)
    err = np.max(np.abs(h_an - fd))
    check(
        "O1 skeleton D.V_xy + classical + V_x in F^x = FD of gradient (claim O1)",
        err < 1e-7,
        f"max err {err:.2e}",
    )
    h_neg = full.hessian(lam0, c, eps, vx_in_fx=False)
    miss = np.max(np.abs(h_neg - fd))
    check(
        "O1 negative control: V_x dropped from F^x misses (claim O1)",
        miss > 1e3 * max(err, 1e-12) and miss > 1e-4,
        f"miss {miss:.2e} vs agreement {err:.2e}",
    )
    h_cls = np.max(np.abs(cls.hessian(lam0)))
    check(
        "O1 the classical Hessian is not negligible in the comparison (claim O1)",
        h_cls > 1e-3,
        f"max |classical| {h_cls:.2e}",
    )

    # O2: stencil route for the skeleton of V.
    co = c[:, :NOCC]
    d = 2.0 * co @ co.T
    exact = np.array(
        [[np.sum(d * op.d2v(lam0, k, ll)) for ll in range(NPAR)] for k in range(NPAR)]
    )
    sten = full.skeleton_v_stencil(lam0, d)
    es = np.max(np.abs(sten - exact))
    check(
        "O2 stencil of Sum D V_x (h = 2e-3) = Sum D V_xy (claim O2)",
        es < 1e-9,
        f"max err {es:.2e}",
    )
    h_st = full.hessian(lam0, c, eps, stencil_skeleton=True)
    est = np.max(np.abs(h_st - fd))
    check(
        "O2 Hessian with the stencil skeleton = FD of gradient (claim O2)",
        est < 1e-7,
        f"max err {est:.2e}",
    )


# ---------------------------------------------------------------------------
# O3: classical terms in symbols.
# ---------------------------------------------------------------------------
def classical_symbolic():
    x, y, z = sp.symbols("x y z", real=True)
    rx, ry, rz = sp.symbols("r_x r_y r_z", real=True)
    zc, q = sp.symbols("Z q", positive=True)
    ex, ey, ez = sp.symbols("E_x E_y E_z", real=True)
    coords = [x, y, z]
    e_field = -zc * (ex * x + ey * y + ez * z)
    check(
        "O3 field-nucleus energy -Z E.R has zero Hessian (claim O3)",
        sp.hessian(e_field, coords) == sp.zeros(3),
    )

    d = sp.Matrix([x - rx, y - ry, z - rz])
    r = sp.sqrt(d.dot(d))
    rs = sp.Symbol("R", positive=True)
    f = sp.Function("f")
    hess = sp.hessian(f(r), coords)
    f1 = sp.diff(f(rs), rs).subs(rs, r)
    f2 = sp.diff(f(rs), rs, 2).subs(rs, r)
    radial = f2 * d * d.T / r**2 + (f1 / r) * (sp.eye(3) - d * d.T / r**2)
    check(
        "O3 Hessian of f(|R-r|) = f'' dd/R^2 + (f'/R)(1 - dd/R^2) (claim O3)",
        sp.simplify(hess - radial) == sp.zeros(3),
    )

    point = zc * q * (3 * d * d.T / r**5 - sp.eye(3) / r**3)
    check(
        "O3 point-charge block Z q (3 dd/R^5 - 1/R^3) = Hessian of Z q/R (claim O3)",
        sp.simplify(sp.hessian(zc * q / r, coords) - point) == sp.zeros(3),
    )

    w = sp.Symbol("w", positive=True)
    fs = sp.erf(rs / w) / rs
    g = sp.erf(rs / w)
    gp = 2 / (sp.sqrt(sp.pi) * w) * sp.exp(-(rs**2) / w**2)
    code_f1 = gp / rs - g / rs**2
    code_f2 = -gp * (2 / w**2 + 2 / rs**2) + 2 * g / rs**3
    ok = sp.simplify(sp.diff(fs, rs) - code_f1) == 0
    ok = ok and sp.simplify(sp.diff(fs, rs, 2) - code_f2) == 0
    check("O3 smeared_radial_derivs are f', f'' of erf(R/w)/R (claim O3)", ok)

    # Small width: the smeared block tends to the point-charge block.
    cls_w = Classical(1.0, 0.0, [0, 0, 5], 1.0, [1.2, -0.7, 0.4], 0.05, [0, 0, 0])
    cls_p = Classical(1.0, 1.0, [1.2, -0.7, 0.4], 0.0, [0, 0, 5], 1.0, [0, 0, 0])
    lam = np.array([0.05, 0.02])
    dw = np.max(np.abs(cls_w.hessian(lam) - cls_p.hessian(lam)))
    check(
        "O3 smeared block at w = 0.05 Bohr equals the point-charge block (claim O3)",
        dw < 1e-12,
        f"max diff {dw:.1e}",
    )


# ---------------------------------------------------------------------------
# O4: the stencils.
# ---------------------------------------------------------------------------
def stencil_symbolic():
    """Taylor expansions with the derivatives g_n = g^(n)(t) as symbols."""
    h = sp.Symbol("h")
    g = sp.symbols("g0:9")

    def shifted(k, o, order):
        """Taylor series of g^(k)(t + o h) through h^order."""
        return sum(g[k + n] * (o * h) ** n / sp.factorial(n) for n in range(order + 1))

    d1 = sp.expand(
        sum(sp.Integer(int(w)) * shifted(1, int(o), 6) for o, w in D1) / (12 * h)
    )
    check(
        "O4 D1 stencil of g' = g'' - h^4 g^(6)/30 + O(h^6) (claim O4)",
        sp.expand(d1 - (g[2] - h**4 * g[6] / 30)) == 0,
        f"D1[g'] = {d1}",
    )
    d2 = sp.expand(
        sum(sp.Integer(int(w)) * shifted(0, int(o), 5) for o, w in D2) / (12 * h**2)
    )
    check("O4 D2 stencil of g = g'' + O(h^4) (claim O4)", d2 == g[2], f"= {d2}")

    f = {(m, n): sp.Symbol(f"f{m}{n}") for m in range(6) for n in range(6)}

    def shifted2(oi, oj):
        """Taylor series of f(u + oi h, v + oj h) through total order 5."""
        return sum(
            f[m, n]
            * (oi * h) ** m
            * (oj * h) ** n
            / (sp.factorial(m) * sp.factorial(n))
            for m in range(6)
            for n in range(6 - m)
        )

    mixed = sp.expand(
        sum(
            sp.Integer(int(wi * wj)) * shifted2(int(oi), int(oj))
            for oi, wi in D1
            for oj, wj in D1
        )
        / (144 * h**2)
    )
    check(
        "O4 D1 x D1 stencil = d2/du dv + O(h^4) (claim O4)",
        mixed == f[1, 1],
        f"= {mixed}",
    )

    t, u, v = sp.symbols("t u v")
    check(
        "O4 stencil weights sum to zero (claim O4)",
        sum(w for _, w in D1) == 0.0 and sum(w for _, w in D2) == 0.0,
    )
    c0, c1, c2 = sp.symbols("c0 c1 c2")
    lin = c0 + c1 * t
    d2lin = sum(sp.Rational(int(w)) * lin.subs(t, t + int(o) * h) for o, w in D2)
    lin2 = c0 + c1 * u + c2 * v
    mixlin = sum(
        sp.Rational(int(wi * wj)) * lin2.subs({u: u + int(oi) * h, v: v + int(oj) * h})
        for oi, wi in D1
        for oj, wj in D1
    )
    check(
        "O4 D2 and D1 x D1 vanish on a linear function (claim O4)",
        sp.expand(d2lin) == 0 and sp.expand(mixlin) == 0,
    )


# ---------------------------------------------------------------------------
# O5: rigid-per-atom displacement and the mask (1-D three-centre model).
# ---------------------------------------------------------------------------
_A, _B, _C = sp.symbols("A B C", real=True)
_a, _b, _c = sp.symbols("a b c", positive=True)
_p = _a + _b + _c
_F = sp.sqrt(sp.pi / _p) * sp.exp(
    -(_a * _b * (_A - _B) ** 2 + _a * _c * (_A - _C) ** 2 + _b * _c * (_B - _C) ** 2)
    / _p
)
_ARGS = (_A, _B, _C, _a, _b, _c)
F_GRAD = [sp.lambdify(_ARGS, sp.diff(_F, v)) for v in (_A, _B, _C)]
F_HESS = [
    [sp.lambdify(_ARGS, sp.diff(_F, u, v)) for v in (_A, _B, _C)] for u in (_A, _B, _C)
]


class TripleModel:
    """E(X) = sum_{mu nu u} D_mu nu f(X[at mu], X[at nu], X[at u]), with
    f = int exp(-a(r-A)^2 - b(r-B)^2 - c(r-C)^2) dr (an s-type 'ECP')."""

    def __init__(self, rng):
        self.x = np.array([0.0, 1.3, 2.9])
        self.sh_atom = [0, 0, 1, 2, 2]
        self.sh_exp = [0.9, 0.35, 0.6, 1.1, 0.45]
        self.ecp_atom = [0, 2]
        self.ecp_exp = [0.8, 0.5]
        ns = len(self.sh_atom)
        self.d = sym(rng.standard_normal((ns, ns)))

    def triples(self):
        ns = len(self.sh_atom)
        for i in range(ns):
            for j in range(ns):
                for u in range(len(self.ecp_atom)):
                    yield i, j, u

    def centres(self, i, j, u):
        return [self.sh_atom[i], self.sh_atom[j], self.ecp_atom[u]]

    def exps(self, i, j, u):
        return (self.sh_exp[i], self.sh_exp[j], self.ecp_exp[u])

    def shifted(self, i, j, u, b, shift):
        """Positions of the triple with atom b displaced by `shift`: every
        centre sitting on b moves (rigid per atom)."""
        pos = [self.x[a] for a in self.centres(i, j, u)]
        for k, a in enumerate(self.centres(i, j, u)):
            if a == b:
                pos[k] += shift
        return pos

    def folded_first(self, b, shift, keep):
        """Sum D df/dX_A folded onto atoms, over the triples `keep` admits."""
        out = np.zeros(len(self.x))
        for i, j, u in self.triples():
            cen = self.centres(i, j, u)
            if not keep(cen):
                continue
            args = (*self.shifted(i, j, u, b, shift), *self.exps(i, j, u))
            for k in range(3):
                out[cen[k]] += self.d[i, j] * F_GRAD[k](*args)
        return out

    def stencil_hessian(self, keep):
        na = len(self.x)
        h = np.zeros((na, na))
        for b in range(na):
            for off, wt in D1:
                rows = self.folded_first(b, off * STEP, lambda cen, b=b: keep(cen, b))
                h[:, b] += wt / (12.0 * STEP) * rows
        return h

    def exact_hessian(self):
        na = len(self.x)
        h = np.zeros((na, na))
        for i, j, u in self.triples():
            cen = self.centres(i, j, u)
            args = (*[self.x[a] for a in cen], *self.exps(i, j, u))
            for k in range(3):
                for ll in range(3):
                    h[cen[k], cen[ll]] += self.d[i, j] * F_HESS[k][ll](*args)
        return h


def moving(cen, b):
    """moving_triples: some centre on b, but not all three."""
    return sum(1 for a in cen if a == b) in (1, 2)


def mask_checks():
    tm = TripleModel(np.random.default_rng(3))
    exact = tm.exact_hessian()
    masked = tm.stencil_hessian(moving)
    err = np.max(np.abs(masked - exact))
    check(
        "O5 masked stencil of folded first derivatives = exact Hessian (claim O5)",
        err < 1e-9,
        f"max err {err:.2e} (max |H| {np.max(np.abs(exact)):.2f})",
    )
    none_on = tm.stencil_hessian(lambda cen, b: b not in cen)
    all_on = tm.stencil_hessian(lambda cen, b: all(a == b for a in cen))
    has_all_on = any(len(set(tm.centres(*t))) == 1 for t in tm.triples())
    check(
        "O5 triples with no centre or all centres on the atom add zero (claim O5)",
        has_all_on
        and np.max(np.abs(none_on)) < 1e-10
        and np.max(np.abs(all_on)) < 1e-10,
        f"none {np.max(np.abs(none_on)):.1e}, all {np.max(np.abs(all_on)):.1e}",
    )


# ---------------------------------------------------------------------------
# O6: the field same-atom block and the pair stencil.
# ---------------------------------------------------------------------------
def field_checks():
    xs, ra = sp.symbols("x R_A", real=True)
    al, be = sp.symbols("alpha beta", positive=True)
    ok_lin, ok_first = True, True
    for la in (0, 1):
        for lb in (0, 1, 2):
            ga = (xs - ra) ** la * sp.exp(-al * (xs - ra) ** 2)
            gb = (xs - ra) ** lb * sp.exp(-be * (xs - ra) ** 2)
            m = sp.integrate(ga * xs * gb, (xs, -sp.oo, sp.oo))
            s_ab = sp.integrate(ga * gb, (xs, -sp.oo, sp.oo))
            ok_lin = ok_lin and sp.simplify(sp.diff(m, ra, 2)) == 0
            ok_first = ok_first and sp.simplify(sp.diff(m, ra) - s_ab) == 0
    check(
        "O6 same-atom dipole block is linear in R_A, d/dR_A = S_AA (claim O6)",
        ok_lin and ok_first,
    )

    # 1-D field skeleton: E sum D_mu nu <mu|x|nu>, s Gaussians on three atoms.
    xa, xb, a1, b1 = sp.symbols("xa xb a1 b1", real=True)
    p = a1 + b1
    m_ab = (
        sp.sqrt(sp.pi / p)
        * sp.exp(-a1 * b1 / p * (xa - xb) ** 2)
        * (a1 * xa + b1 * xb)
        / p
    )
    args = (xa, xb, a1, b1)
    mval = sp.lambdify(args, m_ab)
    mh = [[sp.lambdify(args, sp.diff(m_ab, u, v)) for v in (xa, xb)] for u in (xa, xb)]
    xpos = np.array([0.0, 1.2, 2.7])
    sh_atom = [0, 0, 1, 2, 2]
    sh_exp = [0.9, 0.35, 0.6, 1.1, 0.45]
    efield = 0.7
    dmat = sym(np.random.default_rng(5).standard_normal((5, 5)))
    na, ns = len(xpos), len(sh_atom)
    exact = np.zeros((na, na))
    for i in range(ns):
        for j in range(ns):
            cen = (sh_atom[i], sh_atom[j])
            for k in range(2):
                for ll in range(2):
                    v = mh[k][ll](xpos[cen[0]], xpos[cen[1]], sh_exp[i], sh_exp[j])
                    exact[cen[k], cen[ll]] += efield * dmat[i, j] * v

    def pair_energy(a, c, pa, pc):
        """PairEnergy::value: 2 sum_{mu in a, nu in c} D M at moved atoms."""
        tot = 0.0
        for i in range(ns):
            for j in range(ns):
                if sh_atom[i] == a and sh_atom[j] == c:
                    tot += dmat[i, j] * mval(pa, pc, sh_exp[i], sh_exp[j])
        return 2.0 * efield * tot

    code = np.zeros((na, na))
    for a in range(na):
        for c in range(a + 1, na):
            xa0, xc0 = xpos[a], xpos[c]
            cross = sum(
                wi * wj * pair_energy(a, c, xa0 + oi * STEP, xc0 + oj * STEP)
                for oi, wi in D1
                for oj, wj in D1
            ) / (144.0 * STEP * STEP)
            code[a, c] += cross
            code[c, a] += cross
            code[a, a] += sum(
                w * pair_energy(a, c, xa0 + o * STEP, xc0) for o, w in D2
            ) / (12.0 * STEP * STEP)
            code[c, c] += sum(
                w * pair_energy(a, c, xa0, xc0 + o * STEP) for o, w in D2
            ) / (12.0 * STEP * STEP)
    err = np.max(np.abs(code - exact))
    check(
        "O6 pair stencil with same-atom blocks skipped = exact field skeleton (claim O6)",
        err < 1e-8,
        f"max err {err:.2e} (max |H| {np.max(np.abs(exact)):.2f})",
    )


def main():
    scf_checks()
    classical_symbolic()
    stencil_symbolic()
    mask_checks()
    field_checks()
    if failures:
        print(f"{len(failures)} check(s) failed")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
