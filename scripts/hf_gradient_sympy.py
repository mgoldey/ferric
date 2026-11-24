#!/usr/bin/env python3
"""Hartree-Fock nuclear gradient as ferric assembles it (RHF, UHF, ROHF).

Derivation: `book/src/14-derivatives/01-hf-gradient.md#hf-gradient-gradient`.
Code: crates/ferric-scf/src/gradient.rs (rhf_gradient,
build_energy_weighted_density; uhf_gradient,
build_energy_weighted_density_uhf; rohf_gradient,
rohf_energy_weighted_density; oneelectron_gradient, twoelectron_gradient
with gamma / gamma_uhf).

The code evaluates, for a nuclear coordinate x,
  E^x = V_nn^x + sum D h^x + sum Gamma (mn|ls)^x - sum W S^x,
  RHF  D = 2 C_o C_o^T, W = 2 C_o diag(eps_o) C_o^T,
       Gamma = D_mn D_ls / 2 - D_ml D_ns / 4;
  UHF  D = D_a + D_b, W = sum_s C_so diag(eps_so) C_so^T,
       Gamma = D_mn D_ls / 2 - (Da_ml Da_ns + Db_ml Db_ns) / 2;
  ROHF the UHF Gamma at the ROHF densities, W = sym(D_a F_a D_a + D_a F_b D_b).

Independent route: a toy "molecule" whose integrals are smooth functions of
two parameters lam = (x, y): h(lam), S(lam) (non-orthogonal, positive
definite), (mn|ls)(lam) = sum_P B_P,mn(lam) B_P,ls(lam) (8-fold symmetric),
V_nn(lam), each quadratic in lam with exact first derivatives. The SCF is
re-solved at lam +/- h, 2h and the energy differenced with the fourth-order
central stencil (h = 1e-3, truncation ~1e-12, SCF converged to a 1e-12
orbital gradient). The analytic formula with the code's W must match to 1e-8.

Checks:
  G1  RHF:  E^x analytic = FD, both parameters.
  G2  UHF:  same (n_a = 4, n_b = 2).
  G3  ROHF: same (2 closed, 2 open, Guest-Saunders Roothaan Fock of rohf.rs).
  G4  The ROHF W is symmetric at convergence before sym() (closed-open block
      of F_b vanishes), to 1e-10.
  G5  Negative control: the eigenvalue form C diag(2 eps_c, eps_o) C^T of the
      Roothaan Fock misses the FD gradient by more than 1e-4 (the open shell
      couples to the closed shells in the toy).
  S1  (sympy) d/dlam of the orthonormality constraint C^T S C = 1 at fixed
      density gives the Pulay term: for one occupied orbital c in a 2-function
      basis, d/dlam [c^T h c] at c = c0/sqrt(c0^T S c0) has the -eps c^T S' c
      piece with eps = c^T F c.

Mutations (each turns the script red; exact one-occurrence substitutions):
  RHF W: drop the factor 2                           (G1 fails)
  ROHF W: D_a F_b D_b -> D_a F_a D_b                 (G3 fails)
  UHF Gamma: exchange weight 1/2 -> 1/4              (G2 fails)
  Pulay sign: - W S^x -> + W S^x                     (G1-G3 fail)
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


def rohf_state(model, lam, nc, no, guess=None, extra=None):
    h, s, g = model.h(lam), model.s(lam), model.eri(lam)
    c, eps = solve_rohf(h, s, g, nc, no, guess)
    da, db = rohf_densities(c, nc, no)
    fa, fb = uhf_focks(h, g, da, db)
    e = uhf_energy(h, g, da, db) + model.enn(lam)
    w_raw = rohf_w(da, db, fa, fb)
    grad = assemble_gradient(model, lam, da + db, gamma_uhf(da, db), sym(w_raw))
    if extra is not None:
        # Negative control: eigenvalue W from the Roothaan Fock.
        occ = np.zeros(N)
        occ[:nc], occ[nc : nc + no] = 2.0, 1.0
        w_bad = (c * (occ * eps)) @ c.T
        extra["bad"] = assemble_gradient(model, lam, da + db, gamma_uhf(da, db), w_bad)
        extra["asym"] = np.max(np.abs(w_raw - w_raw.T))
    return e, grad, c


def fd_gradient(state, lam, guess):
    """Fourth-order central difference of the SCF energy, re-solved per point."""
    hh = 1e-3
    out = np.zeros(NPAR)
    for k in range(NPAR):
        vals = {}
        for t in (-2, -1, 1, 2):
            lt = lam.copy()
            lt[k] += t * hh
            vals[t] = state(lt, guess)[0]
        out[k] = (8 * (vals[1] - vals[-1]) - (vals[2] - vals[-2])) / (12 * hh)
    return out


def numeric_checks():
    rng = np.random.default_rng(7)
    model = Model(rng)
    lam0 = np.array([0.3, -0.2])
    bar = 1e-8

    e, grad, c = rhf_state(model, lam0, 3)
    fd = fd_gradient(lambda lt, gs: rhf_state(model, lt, 3, gs), lam0, c)
    err = np.max(np.abs(grad - fd))
    check(
        "G1 RHF gradient = FD of re-solved SCF (claim G1)",
        err < bar,
        f"max err {err:.2e} grad {grad}",
    )

    e, grad, cs = uhf_state(model, lam0, 4, 2)
    fd = fd_gradient(lambda lt, gs: uhf_state(model, lt, 4, 2, gs), lam0, cs)
    err = np.max(np.abs(grad - fd))
    check(
        "G2 UHF gradient = FD of re-solved SCF (claim G2)",
        err < bar,
        f"max err {err:.2e} grad {grad}",
    )

    extra = {}
    e, grad, c = rohf_state(model, lam0, 2, 2, extra=extra)
    fd = fd_gradient(lambda lt, gs: rohf_state(model, lt, 2, 2, gs), lam0, c)
    err = np.max(np.abs(grad - fd))
    check(
        "G3 ROHF gradient = FD of re-solved SCF (claim G3)",
        err < bar,
        f"max err {err:.2e} grad {grad}",
    )
    check(
        "G4 ROHF W symmetric before sym() (claim G4)",
        extra["asym"] < 1e-10,
        f"max asym {extra['asym']:.2e}",
    )
    miss = np.max(np.abs(extra["bad"] - fd))
    check(
        "G5 eigenvalue-form ROHF W misses FD (negative control) (claim G5)",
        miss > 1e-4,
        f"miss {miss:.2e}",
    )


def symbolic_checks():
    """Pulay term for one doubly occupied orbital in a 2-function basis.

    E(lam) = c^T h c * 2 + (cc|cc) with c the normalized fixed direction
    c = c0 / sqrt(c0^T S c0). At fixed c0 (the density relaxation contributes
    nothing at a stationary point, so it is dropped here) dE/dlam picks up
    -2 eps c^T S' c with eps = c^T F c, F = h + J(D) - K(D)/2, D = 2 c c^T,
    which is -tr(W S') with W = 2 eps c c^T.
    """
    lam = sp.symbols("lam")
    a, b = sp.symbols("a b")
    c0 = sp.Matrix([a, b])
    h = sp.Matrix(2, 2, lambda i, j: sp.Function(f"h{min(i, j)}{max(i, j)}")(lam))
    s = sp.Matrix(2, 2, lambda i, j: sp.Function(f"s{min(i, j)}{max(i, j)}")(lam))
    gs = sp.Function("gcccc")  # (cc|cc) in the c0 basis scaled by norm^4
    nrm = (c0.T * s * c0)[0, 0]
    c = c0 / sp.sqrt(nrm)
    # (cc|cc) for the normalized orbital = g0(lam) / nrm^2 with g0 = (c0c0|c0c0).
    g0 = gs(lam)
    e = 2 * (c.T * h * c)[0, 0] + g0 / nrm**2
    de = sp.diff(e, lam)
    # eps = c^T F c = c^T h c + (cc|cc) (closed shell: J - K/2 with D = 2cc^T).
    eps = (c.T * h * c)[0, 0] + g0 / nrm**2
    skel = 2 * (c.T * sp.diff(h, lam) * c)[0, 0] + sp.diff(g0, lam) / nrm**2
    pulay = -2 * eps * (c.T * sp.diff(s, lam) * c)[0, 0]
    check(
        "S1 dE/dlam = skeleton - tr(W S') at fixed c0 (sympy) (claim S1)",
        sp.simplify(de - skel - pulay) == 0,
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
