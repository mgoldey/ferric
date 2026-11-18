#!/usr/bin/env python3
"""Imaginary-frequency quadratures of the RPA: the maps, their Jacobians, exactness.

Derivation: `book/src/18-rpa/02-quadratures.md#quadratures-rule`.
Code: crates/ferric-rpa/src/quadrature.rs (build_quadrature,
gauss_legendre_nodes, gl_nodes_weights, legendre_and_deriv, optimized_u0,
chebyshev_tan_nodes). The functions below transcribe those loops.

As coded, int_0^oo f(u) du ~ sum_k w_k f(u_k) with
  Gauss-Legendre: u = u0 (1 + x)/(1 - x),  w_k = w_k^GL 2 u0/(1 - x_k)^2
  MiniMax:        the Gauss-Legendre map at u0 = 0.3 (n <= 8), 0.4 (n <= 16),
                  0.5 (n > 16); the configured u0 is not read
  Chebyshev-Tan:  theta_k = pi k/(n + 1), x_k = cos theta_k,
                  u = u0 tan(pi (1 + x)/4),
                  w_k = pi/(n + 1) sin theta_k (u0 pi/4) sec^2(pi (1 + x_k)/4)

Checks (each label carries "(claim Qn)"):
  Q1  du/dx of both maps symbolically: 2 u0/(1 - x)^2 and
      (u0 pi/4) sec^2(pi (1 + x)/4); both maps send (-1, 1) onto (0, oo)
      monotonically.
  Q2  Chebyshev-Tan is the trapezoid rule in theta: with x = cos theta,
      int_0^oo f du = int_0^pi f(u(cos theta)) u'(cos theta) sin theta dtheta,
      and the coded weight is the trapezoid step pi/(n + 1) times that
      Jacobian (symbolic identity, then numerically against the coded
      weights).
  Q3  Exactness anchor (Gauss-Legendre): an integrand whose pull-back
      f(u(x)) du/dx is x^m is integrated exactly for every m <= 2n - 1 (to a
      rounding bound derived from the sum), and not for m = 2n (negative
      control). This pins the nodes, the Newton/recurrence transcription and
      the Jacobian together.
  Q4  Exactness anchor (Chebyshev-Tan): for f = u0/(u0^2 + u^2) the Jacobian
      cancels f, f u' = pi/4, so the rule equals
      (pi/4)(pi/(n + 1)) cot(pi/(2(n + 1))) at every n (rounding bound), and
      the error against the exact pi/2 falls as n^-2 for a Lorentzian of
      another scale (ratio within 3.5 to 4.5 per doubling).
  Q5  Independent route: closed-form integrals of RPA-shaped integrands with
      every scheme, against the exact values: int a/(a^2 + u^2) = pi/2, and
      the one-mode dRPA integrand
      (1/2pi) int [ln(1 + 2bD/(u^2 + D^2)) - 2bD/(u^2 + D^2)] du
        = (sqrt(D (D + 2b)) - D - b)/2;
      Gauss-Legendre and MiniMax at 48 nodes (the exact-dRPA order) to the
      rounding bound of the sum, Chebyshev-Tan with error ratio 3.5 to 4.5 on
      each doubling n = 20 -> 40 -> 80.
  Q6  MiniMax is the Gauss-Legendre map with the tabulated u0, and the
      configured u0 does not change it; that table is 0.3/0.4/0.5 at the
      breakpoints 8 and 16 (Furche, JCP 122, 164106 (2005), as documented).

Mutations (each turns the script red; exact one-occurrence substitutions):
  Gauss-Legendre weight drops the factor 2 of the Jacobian      (Q3 fails)
  Chebyshev-Tan weight uses sin^2 theta (no 1/sqrt(1-x^2))      (Q2, Q4 fail)
  Chebyshev-Tan Jacobian u0 pi/4 -> u0 pi/2                     (Q2, Q4 fail)
  optimized_u0 middle entry 0.4 -> 0.5                          (Q6 fails)
Exits nonzero on any failed check; prints "FAIL <label>".
"""

import math
import sys

import numpy as np
import sympy as sp

failures = []
EPS = np.finfo(float).eps


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        failures.append(label)


# --- transcriptions of crates/ferric-rpa/src/quadrature.rs ----------------------


def legendre_and_deriv(n, x):
    if n == 0:
        return 1.0, 0.0
    if n == 1:
        return x, 1.0
    p_prev, p_curr = 1.0, x
    for k in range(2, n + 1):
        p_next = ((2 * k - 1) * x * p_curr - (k - 1) * p_prev) / k
        p_prev, p_curr = p_curr, p_next
    dp = n * (x * p_curr - p_prev) / (x * x - 1.0)
    return p_curr, dp


def gl_nodes_weights(n):
    x = [0.0] * n
    w = [0.0] * n
    for i in range((n + 1) // 2):
        xi = math.cos(math.pi * (4 * i + 3) / (4 * n + 2))
        for _ in range(100):
            p, dp = legendre_and_deriv(n, xi)
            dx = -p / dp
            xi += dx
            if abs(dx) < 1e-15:
                break
        _, dp = legendre_and_deriv(n, xi)
        wi = 2.0 / ((1.0 - xi * xi) * dp * dp)
        x[i], x[n - 1 - i] = -xi, xi
        w[i], w[n - 1 - i] = wi, wi
    pairs = sorted(zip(x, w))
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def gauss_legendre_nodes(n, u0):
    x, w = gl_nodes_weights(n)
    freqs = u0 * (1.0 + x) / (1.0 - x)
    weights = w * 2.0 * u0 / (1.0 - x) ** 2
    return freqs, weights


def optimized_u0(n):
    if n <= 8:
        return 0.3
    if n <= 16:
        return 0.4
    return 0.5


def chebyshev_tan_nodes(n, u0):
    np1 = n + 1
    freqs, weights = [], []
    for k in range(1, n + 1):
        theta_k = math.pi * k / np1
        x_k = math.cos(theta_k)
        arg = math.pi * (1.0 + x_k) / 4.0
        omega_k = u0 * math.tan(arg)
        sec2 = 1.0 / math.cos(arg) ** 2
        w_k = (math.pi / np1) * math.sin(theta_k) * (u0 * math.pi / 4.0) * sec2
        freqs.append(omega_k)
        weights.append(w_k)
    return np.array(freqs), np.array(weights)


def build_quadrature(scheme, n, u0):
    if scheme == "gauss-legendre":
        return gauss_legendre_nodes(n, u0)
    if scheme == "minimax":
        return gauss_legendre_nodes(n, optimized_u0(n))
    if scheme == "chebyshev-tan":
        return chebyshev_tan_nodes(n, u0)
    raise ValueError(scheme)


def rule(scheme, n, u0, f):
    u, w = build_quadrature(scheme, n, u0)
    vals = w * f(u)
    return vals.sum(), np.abs(vals).sum()


# --- checks ----------------------------------------------------------------------


def claim_q1():
    x, u0 = sp.symbols("x u0", positive=True)
    gl = u0 * (1 + x) / (1 - x)
    ct = u0 * sp.tan(sp.pi * (1 + x) / 4)
    d_gl = sp.simplify(sp.diff(gl, x) - 2 * u0 / (1 - x) ** 2) == 0
    d_ct = (
        sp.simplify(sp.diff(ct, x) - u0 * sp.pi / 4 / sp.cos(sp.pi * (1 + x) / 4) ** 2)
        == 0
    )
    ends = (
        sp.limit(gl, x, -1, "+") == 0
        and sp.limit(gl, x, 1, "-") == sp.oo
        and sp.limit(ct, x, -1, "+") == 0
        and sp.limit(ct, x, 1, "-") == sp.oo
    )
    check(
        "Q1 Jacobians du/dx of the GL and CT maps (claim Q1)",
        d_gl and d_ct and ends,
        f"GL {d_gl} CT {d_ct} endpoints {ends}",
    )


def claim_q2():
    th, u0 = sp.symbols("theta u0", positive=True)
    xx = sp.cos(th)
    jac_theta = sp.diff(u0 * sp.tan(sp.pi * (1 + xx) / 4), th)
    # dtheta-Jacobian is -(du/dx) sin(theta): the rule integrates over theta in (0, pi)
    target = -u0 * sp.pi / 4 / sp.cos(sp.pi * (1 + xx) / 4) ** 2 * sp.sin(th)
    sym = sp.simplify(jac_theta - target) == 0
    worst = 0.0
    for n in (1, 5, 20):
        for u0v in (0.2, 0.5):
            _, w = chebyshev_tan_nodes(n, u0v)
            for k in range(1, n + 1):
                t = math.pi * k / (n + 1)
                jac = -float(target.subs({th: t, u0: u0v}))
                worst = max(worst, abs(w[k - 1] - math.pi / (n + 1) * jac) / jac)
    check(
        "Q2 CT weight = trapezoid step x d u/d theta (claim Q2)",
        sym and worst < 1e-13,
        f"symbolic {sym}, worst relative weight error {worst:.1e}",
    )


def claim_q3():
    n, u0 = 6, 0.5
    worst, miss = 0.0, None
    for m in range(0, 2 * n + 1):

        def f(u, m=m):
            x = (u - u0) / (u + u0)
            return x**m * 2.0 * u0 / (u + u0) ** 2

        got, scale = rule("gauss-legendre", n, u0, f)
        exact = (1 - (-1) ** (m + 1)) / (m + 1)
        err = abs(got - exact)
        tol = 10 * n * EPS * max(scale, 1.0)
        if m <= 2 * n - 1:
            worst = max(worst, err / tol)
        else:
            miss = err
    check(
        "Q3 GL integrates pull-back x^m exactly for m <= 2n-1 (claim Q3)",
        worst < 1.0 and miss is not None and miss > 1e-6,
        f"worst err/bound {worst:.2f}; m = 2n error {miss:.1e} (must miss)",
    )


def claim_q4():
    worst = 0.0
    for u0 in (0.2, 0.5, 1.3):
        for n in (1, 4, 14, 20, 64):
            got, scale = rule(
                "chebyshev-tan", n, u0, lambda u, u0=u0: u0 / (u0 * u0 + u * u)
            )
            exact = (
                (math.pi / 4) * (math.pi / (n + 1)) / math.tan(math.pi / (2 * (n + 1)))
            )
            worst = max(worst, abs(got - exact) / (10 * n * EPS * scale))
    ratios = []
    for a in (0.3, 3.0):

        def err(n, a=a):
            got, _ = rule("chebyshev-tan", n, 0.5, lambda u: a / (a * a + u * u))
            return abs(got - math.pi / 2)

        e20, e40, e80 = err(20), err(40), err(80)
        ratios += [e20 / e40, e40 / e80]
    order2 = all(3.5 < r < 4.5 for r in ratios)
    check(
        "Q4 CT Lorentzian closed form and n^-2 error (claim Q4)",
        worst < 1.0 and order2,
        f"worst err/bound {worst:.2f}; doubling ratios {[round(r, 2) for r in ratios]}",
    )


def claim_q5():
    d, b = 0.7, 0.35
    exact_mode = 0.5 * (math.sqrt(d * (d + 2 * b)) - d - b)

    def mode(u):
        g = 2 * b * d / (u * u + d * d)
        return (np.log1p(g) - g) / (2 * math.pi)

    def lor(u):
        return 0.8 / (0.64 + u * u)

    tests = ((lor, math.pi / 2), (mode, exact_mode))
    # Gauss-Legendre and MiniMax (u0 = 0.5 above 16 nodes) at the exact-dRPA
    # order 48: the mapped rule has converged to rounding, so the bound is the
    # rounding bound of the sum.
    worst = 0.0
    for scheme in ("gauss-legendre", "minimax"):
        for f, exact in tests:
            got, scale = rule(scheme, 48, 0.5, f)
            worst = max(worst, abs(got - exact) / (10 * 48 * EPS * max(scale, 1.0)))
    # Chebyshev-Tan converges as n^-2 on both integrands.
    ratios = []
    for f, exact in tests:
        e = [abs(rule("chebyshev-tan", n, 0.5, f)[0] - exact) for n in (20, 40, 80)]
        ratios += [e[0] / e[1], e[1] / e[2]]
    order2 = all(3.5 < r < 4.5 for r in ratios)
    check(
        "Q5 closed-form RPA integrands with every scheme (claim Q5)",
        worst < 1.0 and order2,
        f"GL/MM48 worst err/bound {worst:.2f}; CT ratios {[round(r, 2) for r in ratios]}",
    )


def claim_q6():
    same = all(
        np.array_equal(
            build_quadrature("minimax", n, 9.9)[1], gauss_legendre_nodes(n, expect)[1]
        )
        for n, expect in ((4, 0.3), (8, 0.3), (9, 0.4), (16, 0.4), (17, 0.5), (40, 0.5))
    )
    ignores = np.array_equal(
        build_quadrature("minimax", 12, 0.1)[0], build_quadrature("minimax", 12, 3.0)[0]
    )
    check(
        "Q6 MiniMax = GL at the tabulated u0, configured u0 unread (claim Q6)",
        same and ignores,
        f"table {same} ignores u0 {ignores}",
    )


def main():
    claim_q1()
    claim_q2()
    claim_q3()
    claim_q4()
    claim_q5()
    claim_q6()
    if failures:
        print(f"{len(failures)} check(s) failed")
        sys.exit(1)
    print("all quadrature identities verified")


if __name__ == "__main__":
    main()
