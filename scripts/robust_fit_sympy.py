#!/usr/bin/env python3
"""Robust (Dunlap) density fitting with a fitting kernel different from the physical one.

Derivation: crates/ferric-integrals/theory.md, "Density fitting: the fit and
its error" and "Robust fitting with a mismatched metric". Code:
crates/ferric-mp2/src/rimp2.rs (ri_mp2_robust_attenuated_metric,
ri_mp2_robust_attenuated_metric_with, robust_fit_coulomb_parts,
solve_metric_system, spin_components_from_g).

Notation: physical kernel C (Coulomb here), fitting kernel w (erfc(w_m r)/r
here). For orbital-product densities rho_x and auxiliary functions chi_P,
  J_Px = (P|C|x),  A_Px = (P|w|x),  V_C = (P|C|Q),  V_w = (P|w|Q),
  c_x = V_w^-1 A_x                       (the fit, solved by Cholesky in code)
  G_xy = c_x.J_y + J_x.c_y - c_x.V_C.c_y (the robust interaction, code's g)
  naive_xy = J_x.V_w^-1.J_y              (the substitution that diverges)
  standard RI (w = C): J_x.V_C^-1.J_y = (B^T B)_xy, B = L^-1 J, V_C = L L^T.

Claims (each prints "ok [label]" or "FAIL [label]"):
  1 [residual]   (x|C|y) - G_xy = (delta_x|C|delta_y), delta = rho - sum_P c_P chi_P,
                 for ANY coefficients c (sympy, generic symmetric bilinear form
                 over two densities and two auxiliary functions).
  2 [collapse]   With w = C, G = J^T V_C^-1 J = B^T B with B = L^-1 J:
                 the robust form is standard RI (numeric model).
  3 [cholesky]   M = L^-1 satisfies M^T M = V^-1 but M M != V^-1: applying the
                 Cholesky factor twice as if it were symmetric is wrong (the
                 trap solve_metric_system avoids).
  4 [stationary] With w = C the diagonal error (delta_x|C|delta_x) is stationary
                 in c (first derivative zero along random directions), so it is
                 second order in the coefficient error; with w != C the
                 first derivative of the same error at the w-fit is nonzero.
  5 [model]      On a model of s Gaussians in 3-D (two orbital-product
                 densities per centre pair, an even-tempered s auxiliary set,
                 every integral in closed form), exact (x|C|y) - G_xy equals
                 (delta_x|C|delta_y) built from independent closed-form
                 integrals of the residual densities, to 1e-12 relative.
  6 [diverges]   As w_m grows, the naive form's error grows without bound
                 relative to the exact integrals while the robust error stays
                 below the exact integral and below the naive error; at
                 small w_m the robust error tends to that of standard RI.
  7 [psd-diag]   Robust diagonal errors (x|C|x) - G_xx are >= 0 (C positive
                 definite): the robust fit never overbinds a self-interaction.

Mutations (each turns the script red; the runner applies them as exact
one-occurrence substitutions):
  - drop the - c^T V_C c term of G                    ("FAIL [model]")
  - use V_w instead of V_C in the c^T V c term        ("FAIL [model]")
  - c^T J + J^T c -> 2 c^T J (G not symmetric)         ("FAIL [model]")
  - Cholesky factor applied twice, V^-1 -> M M         ("FAIL [collapse]")
The fit coefficients themselves cannot be mutated into a failure of claims 1
and 5: the residual identity holds for any c. What the choice of c changes is
the size of the error (claims 4, 6).
Exits nonzero when any claim fails.
"""

import sys

import mpmath as mp
import numpy as np
import sympy as sp

failures = []


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} [{label}] {detail}")
    if not ok:
        failures.append(label)


# 1. residual identity, symbolically ----------------------------------------
# basis of the bilinear form: rho_x, rho_y, chi_1, chi_2
g = sp.Matrix(4, 4, lambda i, j: sp.Symbol(f"g{min(i, j)}{max(i, j)}"))
cx = sp.Matrix(sp.symbols("cx1 cx2"))
cy = sp.Matrix(sp.symbols("cy1 cy2"))
ux = sp.Matrix([1, 0, -cx[0], -cx[1]])  # delta_x in the basis
uy = sp.Matrix([0, 1, -cy[0], -cy[1]])
jx, jy = g[2:4, 0], g[2:4, 1]
vc = g[2:4, 2:4]
robust = (cx.T * jy)[0] + (jx.T * cy)[0] - (cx.T * vc * cy)[0]
check("residual", sp.expand(g[0, 1] - robust - (ux.T * g * uy)[0]) == 0)

# ---------------------------------------------------------------- the model
mp.mp.dps = 30


def boys0(t):
    return mp.mpf(1) if t == 0 else mp.sqrt(mp.pi / (4 * t)) * mp.erf(mp.sqrt(t))


def pair(a, A, b, B):
    """Product of exp(-a|r-A|^2) exp(-b|r-B|^2): (exponent, centre, prefactor)."""
    p = a + b
    P = (a * A + b * B) / p
    return p, P, np.exp(-a * b / p * np.sum((A - B) ** 2))


def two_center(f1, f2, kernel, wm):
    """(f1|K|f2) for unnormalized Gaussians f = (exponent, centre, prefactor)."""
    p, P, kp = f1
    q, Q, kq = f2
    R2 = float(np.sum((P - Q) ** 2))
    pref = 2 * mp.pi**2.5 / (p * q * mp.sqrt(p + q)) * kp * kq
    coul = pref * boys0(p * q / (p + q) * R2)
    if kernel == "coulomb":
        return coul
    ph2 = 1 / (1 / p + 1 / q + 1 / wm**2)
    erf_part = pref * mp.sqrt(ph2 * (p + q) / (p * q)) * boys0(ph2 * R2)
    return coul - erf_part  # erfc(wm r)/r


centres = [
    np.array([0.0, 0.0, 0.0]),
    np.array([1.4, 0.0, 0.0]),
    np.array([0.3, 1.1, 0.0]),
]
orb = [(1.2, centres[0]), (0.35, centres[1]), (0.8, centres[2]), (2.5, centres[1])]
rho = [
    pair(orb[i][0], orb[i][1], orb[j][0], orb[j][1])
    for i in range(len(orb))
    for j in range(i, len(orb))
]
aux = [(0.15 * 2.6**k, c) for c in centres for k in range(4)]
aux_f = [(a, c, 1.0) for a, c in aux]


def mat(rows, cols, kernel, wm=1.0):
    return np.array(
        [[float(two_center(r, c, kernel, mp.mpf(wm))) for c in cols] for r in rows]
    )


exact = mat(rho, rho, "coulomb")
J = mat(aux_f, rho, "coulomb")
VC = mat(aux_f, aux_f, "coulomb")


def solve(v, rhs):
    """Cholesky solve, as solve_metric_system does."""
    lo = np.linalg.cholesky(v)
    return np.linalg.solve(lo.T, np.linalg.solve(lo, rhs))


def robust_g(coef, VCm, Jm):
    cj = coef.T @ Jm
    return cj + cj.T - coef.T @ VCm @ coef


def fit(wm):
    A = mat(aux_f, rho, "erfc", wm)
    Vw = mat(aux_f, aux_f, "erfc", wm)
    coef = solve(Vw, A)
    return coef, Vw, A


# 2. collapse to standard RI --------------------------------------------------
L = np.linalg.cholesky(VC)
M = np.linalg.inv(L)
B = M @ J
std = B.T @ B
Vinv_used = M.T @ M
g_c = robust_g(Vinv_used @ J, VC, J)
check(
    "collapse",
    np.max(np.abs(g_c - std)) < 1e-10 * np.max(np.abs(std))
    and np.max(np.abs(J.T @ np.linalg.solve(VC, J) - std))
    < 1e-10 * np.max(np.abs(std)),
    f"max |G - B^T B| = {np.max(np.abs(g_c - std)):.2e}",
)

# 3. Cholesky trap ------------------------------------------------------------
vinv = np.linalg.inv(VC)
check(
    "cholesky",
    np.max(np.abs(M.T @ M - vinv)) < 1e-8 * np.max(np.abs(vinv))
    and np.max(np.abs(M @ M - vinv)) > 1e-2 * np.max(np.abs(vinv)),
)


# 4. stationarity of the diagonal error in c ----------------------------------
def diag_err(coef_col, x):
    d = np.concatenate([[1.0], -coef_col])
    blk = np.block(
        [[exact[x : x + 1, x : x + 1], J[:, x : x + 1].T], [J[:, x : x + 1], VC]]
    )
    return d @ blk @ d


rng = np.random.default_rng(3)
x0 = 2
c_std = np.linalg.solve(VC, J[:, x0])
c_w = fit(1.5)[0][:, x0]
h = 1e-5
der_std, der_w = [], []
for _ in range(5):
    dvec = rng.standard_normal(len(aux_f))
    der_std.append(
        abs(diag_err(c_std + h * dvec, x0) - diag_err(c_std - h * dvec, x0)) / (2 * h)
    )
    der_w.append(
        abs(diag_err(c_w + h * dvec, x0) - diag_err(c_w - h * dvec, x0)) / (2 * h)
    )
check(
    "stationary",
    max(der_std) < 1e-6 * exact[x0, x0] and min(der_w) > 1e-4 * exact[x0, x0],
    f"|dE/dc| at the C fit {max(der_std):.1e}, at the w fit {min(der_w):.1e}",
)

# 5. the residual identity on the model, with independent integrals -----------
coef, Vw, A = fit(1.5)
G = robust_g(coef, VC, J)
# residual-density interaction from separately built integrals:
# (d_x|C|d_y) = (x|y) - sum_P c_Px (P|y) - sum_P c_Py (x|P) + sum_PQ c_Px (P|Q) c_Qy
dd = exact.copy()
for x in range(len(rho)):
    for y in range(len(rho)):
        s = 0.0
        for P in range(len(aux_f)):
            s -= coef[P, x] * float(two_center(aux_f[P], rho[y], "coulomb", 1.0))
            s -= coef[P, y] * float(two_center(rho[x], aux_f[P], "coulomb", 1.0))
            for Q in range(len(aux_f)):
                s += (
                    coef[P, x]
                    * coef[Q, y]
                    * float(two_center(aux_f[P], aux_f[Q], "coulomb", 1.0))
                )
        dd[x, y] += s
err = np.max(np.abs((exact - G) - dd)) / np.max(np.abs(exact))
check("model", err < 1e-12, f"max |(x|y) - G - (d|d)| / max|(x|y)| = {err:.1e}")

# 6. naive diverges, robust bounded -------------------------------------------
rows = []
for wm in (0.05, 0.5, 1.5, 3.0, 6.0):
    coef, Vw, A = fit(wm)
    G = robust_g(coef, VC, J)
    naive = J.T @ solve(Vw, J)
    rows.append(
        (
            wm,
            np.max(np.abs(exact - G)) / np.max(np.abs(exact)),
            np.max(np.abs(exact - naive)) / np.max(np.abs(exact)),
        )
    )
std_err = np.max(np.abs(exact - std)) / np.max(np.abs(exact))
for wm, er, en in rows:
    print(f"     w_m = {wm:4.2f}  robust rel err {er:.2e}  naive rel err {en:.2e}")
naive_grows = (
    all(rows[i + 1][2] > rows[i][2] for i in range(1, len(rows) - 1))
    and rows[-1][2] > 1.0
)
robust_ok = all(er < 1.0 and er <= en for _, er, en in rows[1:])
small_w = abs(rows[0][1] - std_err) < 0.1 * std_err + 1e-12
check(
    "diverges",
    naive_grows and robust_ok and small_w,
    f"standard RI rel err {std_err:.2e}",
)

# 7. robust diagonal errors are non-negative ----------------------------------
ok = True
for wm in (0.5, 1.5, 6.0):
    coef, _, _ = fit(wm)
    G = robust_g(coef, VC, J)
    ok &= bool(np.all(np.diag(exact - G) >= -1e-13 * np.max(np.abs(exact))))
check("psd-diag", ok)

if failures:
    print(f"{len(failures)} claim(s) failed: {', '.join(failures)}")
    sys.exit(1)
print("all robust-fit claims hold")
