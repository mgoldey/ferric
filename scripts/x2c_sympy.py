#!/usr/bin/env python3
"""Spin-free X2C-1e: the modified Dirac equation, the decoupling and the spectrum.

Theory: book/src/03-integrals/06-pvp-x2c.md, section "Spin-free X2C-1e";
code: crates/ferric-scf/src/x2c.rs (x2c1e_transform); Rust tests:
crates/ferric-scf/tests/x2c.rs (PySCF sfx2c1e eigenvalues, the c^-2 fall-off).

Atomic units, energies measured from the rest mass (E -> E - c^2). The
one-electron Dirac equation of a potential V is

  V psi_L + c (sigma.p) psi_S        = E psi_L,
  c (sigma.p) psi_L + (V - 2c^2) psi_S = E psi_S.

Claims
  1  Dirac identity on a two-spinor of arbitrary functions, with V an
     arbitrary function of position:
       (sigma.p) V (sigma.p) psi = p.(V p psi) + i sigma.((p V) x (p psi)).
     The first term is the spin-free part p.Vp, whose real-basis matrix
     element is W = sum_a <d_a m|V|d_a n>.
  2  Modified Dirac equation. Substitute psi_S = (sigma.p / 2c) phi. The first
     equation becomes V psi_L + T phi = E psi_L with T = p^2/2. Apply
     (sigma.p / 2c) to the second equation: it becomes
       T psi_L + [(sigma.p) V (sigma.p) / (4c^2) - T] phi = E T phi / (2c^2),
     checked on a two-spinor of arbitrary functions, as operators.
  3  Decoupling (noncommuting matrix symbols). With C_S = X C_L, add X^T times
     the second block row of the 2n matrix problem to the first:
       L C_L - S~ C_L E,  L = V + T X + X^T T - X^T T X + X^T W X/(4c^2),
                          S~ = S + X^T T X/(2c^2).
  4  One basis function (n = 1, scalars s, t, v, w): the electronic eigenvalue
     of the 2 x 2 problem is E = E0 + c^-2 (w/4 - E0 t/2)/s + O(c^-4) with
     E0 = (t + v)/s, the nonrelativistic value. So h^X2C -> T + V and the
     correction falls as c^-2.
  5  Numerical cross-check by an independent route, on a random model with
     S, T positive definite and V, W negative definite (n = 7, c = 137.036):
     h^X2C built from the formulas of the book (the steps of x2c1e_transform)
     has generalized eigenvalues with S equal to the n highest eigenvalues of
     the 2n problem, found by diagonalizing the 2n problem directly
     ("spectrum"); R^T S~ R = S ("renorm"); and ||h^X2C(c) - (T + V)|| falls by
     a factor near 100 per decade of c ("c-scaling").

Mutations (each turns the script red with the named marker):
  - claim 1: drop the i of the spin term                   -> "FAIL [dirac-identity]"
  - claim 2: metric T/(2c^2) -> T/(4c^2) in the operator form -> "FAIL [modified-dirac]"
  - claim 3: X^T T X enters L with + instead of -          -> "FAIL [decoupling]"
  - claim 5: W/(4c^2) -> W/(2c^2) in the bracket of h^X2C   -> "FAIL [spectrum]"
  - claim 5: R without the inner (S^-1/2 S~ S^-1/2)^-1/2    -> "FAIL [renorm]"

Run: python3 scripts/x2c_sympy.py
"""

import sys

import numpy as np
import scipy.linalg
import sympy as sp

FAILED = []


def check(cond, tag, msg):
    print(f"{'ok  ' if cond else 'FAIL'} [{tag}] {msg}")
    if not cond:
        FAILED.append(tag)


# ------------------------------------------------------------- operators
x, y, z = sp.symbols("x y z", real=True)
c, E = sp.symbols("c E", positive=True)
XYZ = (x, y, z)
I = sp.I
SIG = (
    sp.Matrix([[0, 1], [1, 0]]),
    sp.Matrix([[0, -I], [I, 0]]),
    sp.Matrix([[1, 0], [0, -1]]),
)
V = sp.Function("V")(x, y, z)


def spinor(name):
    return sp.Matrix(
        [sp.Function(f"{name}1")(x, y, z), sp.Function(f"{name}2")(x, y, z)]
    )


def p(f, a):
    """p_a = -i d/da on a scalar or a spinor."""
    return -I * sp.diff(f, XYZ[a])


def sigma_p(psi):
    return sum((SIG[a] * p(psi, a) for a in range(3)), sp.zeros(2, 1))


def kin(psi):
    """T = p^2 / 2 on each component."""
    return sum((p(p(psi, a), a) for a in range(3)), sp.zeros(2, 1)) / 2


def is_zero(m):
    return all(sp.simplify(sp.expand(e)) == 0 for e in m)


def claim1():
    print("\n=== 1  Dirac identity (sigma.p) V (sigma.p) = p.Vp + i sigma.(pV x p)")
    psi = spinor("psi")
    lhs = sigma_p(V * sigma_p(psi))
    spin_free = sum((p(V * p(psi, a), a) for a in range(3)), sp.zeros(2, 1))
    pv = [p(V, a) for a in range(3)]
    pp = [p(psi, a) for a in range(3)]
    cross = [
        pv[1] * pp[2] - pv[2] * pp[1],
        pv[2] * pp[0] - pv[0] * pp[2],
        pv[0] * pp[1] - pv[1] * pp[0],
    ]
    spin = I * sum((SIG[a] * cross[a] for a in range(3)), sp.zeros(2, 1))
    check(is_zero(lhs - spin_free - spin), "dirac-identity", "both spinor components")


def claim2():
    print("\n=== 2  modified Dirac equation from psi_S = (sigma.p / 2c) phi")
    psi_l, phi = spinor("L"), spinor("phi")
    psi_s = sigma_p(phi) / (2 * c)
    row1 = V * psi_l + c * sigma_p(psi_s) - E * psi_l
    row2 = c * sigma_p(psi_l) + (V - 2 * c**2) * psi_s - E * psi_s
    claimed1 = V * psi_l + kin(phi) - E * psi_l
    check(
        is_zero(row1 - claimed1),
        "modified-dirac",
        "first row: V psi_L + T phi = E psi_L",
    )
    projected = sigma_p(row2) / (2 * c)
    metric = 2 * c**2
    claimed2 = (
        kin(psi_l)
        + sigma_p(V * sigma_p(phi)) / (4 * c**2)
        - kin(phi)
        - E * kin(phi) / metric
    )
    check(
        is_zero(projected - claimed2),
        "modified-dirac",
        "second row: T psi_L + [(s.p)V(s.p)/4c^2 - T] phi = E T phi / 2c^2",
    )


def claim3():
    print("\n=== 3  decoupling: X^T (row 2) + (row 1) = L C_L - S~ C_L E")
    n = sp.Symbol("n", integer=True, positive=True)
    S, T, Vm, W, X, CL, Em = (
        sp.MatrixSymbol(s, n, n) for s in ("S", "T", "V", "W", "X", "C_L", "E")
    )
    k = 1 / (4 * c**2)
    row1 = Vm * CL + T * X * CL - S * CL * Em
    row2 = T * CL + (k * W - T) * X * CL - T * X * CL * Em / (2 * c**2)
    lag = Vm + T * X + X.T * T - X.T * T * X + k * X.T * W * X
    s_tilde = S + X.T * T * X / (2 * c**2)
    diff = (X.T * row2 + row1) - (lag * CL - s_tilde * CL * Em)
    check(
        sp.expand(diff).doit() == sp.ZeroMatrix(n, n),
        "decoupling",
        "L and S~ as in the book",
    )


def claim4():
    print("\n=== 4  one basis function: E = E0 + c^-2 (w/4 - E0 t/2)/s + O(c^-4)")
    s_, t_, v_, w_, eps = sp.symbols("s t v w epsilon", positive=True)
    h2 = sp.Matrix([[v_, t_], [t_, w_ * eps / 4 - t_]])
    m2 = sp.Matrix([[s_, 0], [0, t_ * eps / 2]])
    # Electronic root: the one that stays finite as eps = c^-2 -> 0.
    e0 = (t_ + v_) / s_
    e1 = sp.Symbol("e1")
    e2 = sp.Symbol("e2")
    poly = sp.expand((h2 - (e0 + e1 * eps + e2 * eps**2) * m2).det())
    lead = [sp.expand(poly).coeff(eps, k) for k in range(3)]
    check(sp.simplify(lead[0]) == 0, "c-limit", "order c^0 vanishes at E0 = (t + v)/s")
    sol = sp.solve(lead[1], e1)
    check(
        len(sol) == 1 and sp.simplify(sol[0] - (w_ / 4 - e0 * t_ / 2) / s_) == 0,
        "c-limit",
        f"c^-2 coefficient {sp.simplify(sol[0]) if sol else None}",
    )
    # Independent: the exact root of the 2 x 2 problem. The series error is
    # O(c^-4), so it must fall by about 10^4 when c grows tenfold.
    vals = {s_: 1.3, t_: 0.8, v_: -2.1, w_: -9.0}
    resid = []
    for cv in (137.035999679940, 1370.35999679940):
        ev = scipy.linalg.eigh(
            np.array(h2.subs(vals).subs(eps, 1 / cv**2), dtype=float),
            np.array(m2.subs(vals).subs(eps, 1 / cv**2), dtype=float),
            eigvals_only=True,
        )[-1]
        approx = float((e0 + sol[0] * eps).subs(vals).subs(eps, 1 / cv**2))
        resid.append(abs(ev - approx))
    q = resid[0] / resid[1]
    check(
        5e3 < q < 2e4,
        "c-limit",
        f"exact root minus series: {resid[0]:.2e} at c, {resid[1]:.2e} at 10c (ratio {q:.0f})",
    )


# ------------------------------------------------------------- numerics
def spd(rng, n, scale, shift):
    a = rng.standard_normal((n, n))
    return scale * (a @ a.T / n) + shift * np.eye(n)


def sym_pow(m, k):
    e, u = np.linalg.eigh(m)
    return (u * e**k) @ u.T


def x2c(s, t, v, w, c):
    """The steps of x2c1e_transform, from the book's equations."""
    n = s.shape[0]
    k = 1 / (4 * c * c)
    h4 = np.block([[v, t], [t, w * k - t]])
    m4 = np.block([[s, np.zeros_like(s)], [np.zeros_like(s), t / (2 * c * c)]])
    _, vec = scipy.linalg.eigh(h4, m4)
    cl, cs = vec[:n, n:], vec[n:, n:]
    xx = np.linalg.solve(cl.T, cs.T).T  # X = C_S C_L^-1
    tx = t @ xx
    s_tilde = s + xx.T @ tx / (2 * c * c)
    lag = v + tx + tx.T - xx.T @ tx + xx.T @ w @ xx * k
    inner = sym_pow(sym_pow(s, -0.5) @ s_tilde @ sym_pow(s, -0.5), -0.5)
    r = sym_pow(s, -0.5) @ inner @ sym_pow(s, 0.5)
    h = r.T @ lag @ r
    return 0.5 * (h + h.T), r, s_tilde


def claim5():
    print("\n=== 5  numpy: spectrum of h^X2C, R^T S~ R = S, c^-2 fall-off")
    rng = np.random.default_rng(20261119)
    n = 7
    cv = 137.035999679940
    s = spd(rng, n, 0.3, 1.0)
    t = spd(rng, n, 2.0, 0.5)
    v = -spd(rng, n, 3.0, 1.0)
    w = -spd(rng, n, 400.0, 50.0)
    h, r, s_tilde = x2c(s, t, v, w, cv)
    # Independent route: the 2n problem diagonalized directly.
    h4 = np.block([[v, t], [t, w / (4 * cv * cv) - t]])
    m4 = np.block([[s, np.zeros_like(s)], [np.zeros_like(s), t / (2 * cv * cv)]])
    e4 = scipy.linalg.eigh(h4, m4, eigvals_only=True)
    e_el = e4[n:]
    e_x2c = scipy.linalg.eigh(h, s, eigvals_only=True)
    gap = e_el.min() - e4[:n].max()
    check(gap > cv**2, "spectrum", f"electronic/positronic gap {gap:.1f} > c^2")
    dev = np.max(np.abs(e_x2c - e_el)) / np.max(np.abs(e_el))
    check(dev < 1e-11, "spectrum", f"eig(h^X2C, S) vs electronic Dirac, rel {dev:.1e}")
    dev_r = np.max(np.abs(r.T @ s_tilde @ r - s))
    check(dev_r < 1e-12, "renorm", f"max |R^T S~ R - S| = {dev_r:.1e}")
    e_nr = scipy.linalg.eigh(t + v, s, eigvals_only=True)
    check(
        np.max(np.abs(e_x2c - e_nr)) > 1e-6,
        "spectrum",
        "control: the nonrelativistic spectrum differs",
    )
    ratios = []
    for f in (1.0, 10.0, 100.0):
        d1 = np.linalg.norm(x2c(s, t, v, w, cv * f)[0] - (t + v))
        d2 = np.linalg.norm(x2c(s, t, v, w, cv * f * 10)[0] - (t + v))
        ratios.append(d1 / d2)
    check(
        all(abs(q - 100) < 5 for q in ratios[1:]) and ratios[0] > 50,
        "c-scaling",
        "||h(c) - (T+V)|| / ||h(10c) - (T+V)|| = "
        + ", ".join(f"{q:.3f}" for q in ratios),
    )


if __name__ == "__main__":
    claim1()
    claim2()
    claim3()
    claim4()
    claim5()
    if FAILED:
        print(f"\n{len(FAILED)} check(s) failed: {sorted(set(FAILED))}")
        sys.exit(1)
    print("\nall checks passed")
