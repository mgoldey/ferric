#!/usr/bin/env python3
"""Nuclear gradient of a Fock-hooked closed-shell SCF: explicit term plus Pulay.

Derivation: crates/ferric-scf/theory.md, "Fock hooks and their nuclear
gradient". Code: crates/ferric-scf/src/gradient.rs (with_fock_hook_gradient and
the energy-weighted density of the hooked orbitals).

Model: a three-function closed-shell system (one doubly occupied orbital) whose
h, S and (mu nu|la si) are smooth functions of one parameter x, plus a hook
E_h(D; x) = kappa Tr[D B(x)] + lam Tr[D S(x)] + (gam/2) sum D_mn^2, so
V = dE_h/dD = kappa B + lam S + gam D. The SCF is re-solved at displaced x.

Independent route: a fourth-order central difference of the re-solved hooked SCF
energy. Formula under test (theory.md):
  dE/dx = Tr[D h'] + sum Gamma (mn|ls)' + dE_h/dx|_D - Tr[W S'],
  W = 2 C eps C^T from the HOOKED Fock matrix, Gamma = D D/2 - D_ml D_ns/4,
  dE_h/dx|_D = kappa Tr[D B'] + lam Tr[D S'] (the gam term has none).

Checks (each label carries "(claim X)"):
  K1  The formula equals the finite difference of the re-solved hooked energy.
  K2  Dropping the explicit term, or building W from the un-hooked Fock matrix,
      misses by far more than the agreement of K1 (negative controls).
  K3  Exactness anchor: with E_h = lam Tr[D S] alone the explicit term
      lam Tr[D S'] and the hook's part of the Pulay term cancel exactly, because
      D S D = 2 D for a normalized doubly occupied orbital (symbols and
      numerics), so the hooked gradient equals the un-hooked one.

Mutations (each turns the script red; exact one-occurrence substitutions):
  explicit term loses its lam Tr[D S'] part                  (K1 fails)
  W built from the un-hooked Fock matrix                      (K1 fails)
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


N = 3


class Model:
    def __init__(self, rng):
        sym = lambda m: 0.5 * (m + m.T)  # noqa: E731
        self.h = [sym(rng.standard_normal((N, N))) for _ in range(3)]
        base = sym(rng.standard_normal((N, N))) * 0.2
        self.s = [np.eye(N) + base, 0.15 * sym(rng.standard_normal((N, N))), 0 * base]
        self.b = [sym(rng.standard_normal((N, N))) for _ in range(2)]
        self.m = [
            [sym(rng.standard_normal((N, N))) * 0.3 for _ in range(2)] for _ in range(3)
        ]

    def hx(self, x):
        return self.h[0] + x * self.h[1] + x * x * self.h[2] * 0.3

    def hxp(self, x):
        return self.h[1] + 2 * x * self.h[2] * 0.3

    def sx(self, x):
        return self.s[0] + x * self.s[1]

    def sxp(self, x):
        return self.s[1]

    def bx(self, x):
        return self.b[0] + x * self.b[1]

    def bxp(self, x):
        return self.b[1]

    def mk(self, x):
        return [mm[0] + x * mm[1] for mm in self.m]

    def mkp(self, x):
        return [mm[1] for mm in self.m]

    def eri(self, x):
        return sum(np.einsum("mn,ls->mnls", a, a) for a in self.mk(x))

    def eri_p(self, x):
        return sum(
            np.einsum("mn,ls->mnls", a, ap) + np.einsum("mn,ls->mnls", ap, a)
            for a, ap in zip(self.mk(x), self.mkp(x))
        )


HOOK = (0.4, 0.3, 0.2)  # (kappa, lam, gam)


def fock_hooked(model, x, d, hook, hooked=True):
    kappa, lam, gam = hook
    g = model.eri(x)
    j = np.einsum("mnls,ls->mn", g, d)
    k = np.einsum("mlns,ls->mn", g, d)
    f = model.hx(x) + j - 0.5 * k
    if hooked:
        f = f + kappa * model.bx(x) + lam * model.sx(x) + gam * d
    return f


def energy(model, x, d, hook):
    kappa, lam, gam = hook
    g = model.eri(x)
    j = np.einsum("mnls,ls->mn", g, d)
    k = np.einsum("mlns,ls->mn", g, d)
    e = np.sum(d * model.hx(x)) + 0.5 * np.sum(d * j) - 0.25 * np.sum(d * k)
    e += kappa * np.sum(d * model.bx(x)) + lam * np.sum(d * model.sx(x))
    e += 0.5 * gam * np.sum(d * d)
    return e


def solve(model, x, hook):
    s = model.sx(x)
    d = np.zeros((N, N))
    for _ in range(2000):
        f = fock_hooked(model, x, d, hook)
        eps, c = sla.eigh(f, s)
        c0 = c[:, :1]
        d_new = 2.0 * c0 @ c0.T
        if np.max(np.abs(d_new - d)) < 1e-14:
            d = d_new
            break
        d = 0.5 * d + 0.5 * d_new
    f = fock_hooked(model, x, d, hook)
    eps, c = sla.eigh(f, s)
    c0 = c[:, :1]
    d = 2.0 * c0 @ c0.T
    return d, c0, eps[:1], energy(model, x, d, hook)


def formula(model, x, d, c0, eps0, hook, use_hooked_w=True, with_explicit=True):
    kappa, lam, _ = hook
    gp = model.eri_p(x)
    gamma = 0.5 * np.einsum("mn,ls->mnls", d, d) - 0.25 * np.einsum("ml,ns->mnls", d, d)
    skel = np.sum(d * model.hxp(x)) + np.sum(gamma * gp)
    explicit = kappa * np.sum(d * model.bxp(x)) + lam * np.sum(d * model.sxp(x))
    if use_hooked_w:
        w = 2.0 * eps0[0] * (c0 @ c0.T)
    else:
        f_plain = fock_hooked(model, x, d, hook, hooked=False)
        w = 2.0 * (c0.T @ f_plain @ c0)[0, 0] * (c0 @ c0.T)
    pulay = -np.sum(w * model.sxp(x))
    return skel + (explicit if with_explicit else 0.0) + pulay


def fd_energy(model, x0, hook, h=1e-3):
    vals = {}
    for t in (-2, -1, 1, 2):
        vals[t] = solve(model, x0 + t * h, hook)[3]
    return (8 * (vals[1] - vals[-1]) - (vals[2] - vals[-2])) / (12 * h)


def main():
    rng = np.random.default_rng(4)
    model = Model(rng)
    x0 = 0.1
    d, c0, eps0, _ = solve(model, x0, HOOK)
    fd = fd_energy(model, x0, HOOK)
    an = formula(model, x0, d, c0, eps0, HOOK)
    err = abs(an - fd)
    check(
        "K1 hooked gradient = FD of the re-solved hooked SCF energy (claim K1)",
        err < 1e-8,
        f"analytic {an:.10f} fd {fd:.10f} err {err:.1e}",
    )
    miss_explicit = abs(formula(model, x0, d, c0, eps0, HOOK, with_explicit=False) - fd)
    miss_w = abs(formula(model, x0, d, c0, eps0, HOOK, use_hooked_w=False) - fd)
    check(
        "K2 dropping the explicit term or the hooked W misses (claim K2)",
        miss_explicit > 1e-3 and miss_w > 1e-3,
        f"explicit {miss_explicit:.1e} unhooked-W {miss_w:.1e}",
    )

    c1, c2, s12, s22 = sp.symbols("c1 c2 s12 s22", real=True)
    s11 = (1 - 2 * s12 * c1 * c2 - s22 * c2**2) / c1**2
    smat = sp.Matrix([[s11, s12], [s12, s22]])
    cv = sp.Matrix([c1, c2])
    dmat = 2 * cv * cv.T
    ok_sym = sp.simplify(dmat * smat * dmat - 2 * dmat) == sp.zeros(2, 2)
    only_lam = (0.0, HOOK[1], 0.0)
    no_hook = (0.0, 0.0, 0.0)
    d_h, c_h, e_h, _ = solve(model, x0, only_lam)
    d_p, c_p, e_p, _ = solve(model, x0, no_hook)
    diff = abs(
        formula(model, x0, d_h, c_h, e_h, only_lam)
        - formula(model, x0, d_p, c_p, e_p, no_hook)
    )
    check(
        "K3 lam Tr[D S] hook: explicit and Pulay parts cancel, D S D = 2 D (claim K3)",
        ok_sym and diff < 1e-9,
        f"|hooked - plain| {diff:.1e}",
    )
    if failures:
        print(f"{len(failures)} check(s) failed")
        sys.exit(1)
    print("all Fock-hook gradient identities verified")


if __name__ == "__main__":
    main()
