#!/usr/bin/env python3
"""Sympy proof of the D3 Axilrod-Teller-Muto (ATM) three-body term's gradient.

Checks the closed forms ferric implements in crates/ferric-d3/src/lib.rs
(`atm_triple_factor`, `atm_energy`, `atm_gradient`). With squared distances
x = R_ij^2, y = R_jk^2, w = R_ik^2 and P = R_ij R_jk R_ik:

  1. ang = 3/8 (x+y-w)(x-y+w)(-x+y+w)/P^5 + 1/P^3
         = (3 cos(a) cos(b) cos(c) + 1)/P^3       (interior angles of the triangle)
  2. d ang/dq = 3/8 dN/dq / P^5 - 15/16 N/(q P^5) - 3/(2 q P^3),   q in {x, y, w},
     with dN/dq from the product rule
  3. fdmp = 1/(1 + 6 (r0/P)^(alpha/3)),  d fdmp/dq = alpha (r0/P)^(alpha/3) fdmp^2 / q
  4. E_t = s9 sqrt(C_ij C_ik C_jk) fdmp ang  =>  dE_t/dC_ij = E_t / (2 C_ij)
  5. the Cartesian gradient assembled from dE/dq * 2 (R_a - R_b) equals sympy's
     direct Cartesian derivative at a random triangle

Negative controls (each must FAIL): the 15/16 term dropped, alpha/6 instead of
alpha in d fdmp/dq, the w-component of dN with the x-component's signs, and
dE/dC = E/C (no 1/2).

Exit status is nonzero on any mismatch.

Run: python3 scripts/d3_atm_gradient_sympy.py
"""

import sys

import sympy as sp

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def control(name, wrong_is_zero):
    # A negative control passes when the WRONG formula does NOT match.
    check(f"negative control: {name} is detected", not wrong_is_zero)


x, y, w, r0, alpha, s9 = sp.symbols("x y w r0 alpha s9", positive=True)
cij, cik, cjk = sp.symbols("C_ij C_ik C_jk", positive=True)

P = sp.sqrt(x * y * w)
u, v, t = x + y - w, x - y + w, -x + y + w
N = u * v * t
ang = sp.Rational(3, 8) * N / P**5 + 1 / P**3
rr = (r0 / P) ** (alpha / 3)
fdmp = 1 / (1 + 6 * rr)

# --- 1. angular factor -----------------------------------------------------
a, b, c = sp.sqrt(x), sp.sqrt(y), sp.sqrt(w)  # R_ij, R_jk, R_ik
cos_i = (a**2 + c**2 - b**2) / (2 * a * c)  # at atom i, between R_ij and R_ik
cos_j = (a**2 + b**2 - c**2) / (2 * a * b)
cos_k = (b**2 + c**2 - a**2) / (2 * b * c)
ang_cos = (3 * cos_i * cos_j * cos_k + 1) / (a * b * c) ** 3
check("1. ang == (3 cos cos cos + 1)/P^3", sp.simplify(ang - ang_cos) == 0)

# --- 2/3. derivatives as implemented ---------------------------------------
dnum = {
    x: v * t + u * t - u * v,
    y: v * t - u * t + u * v,
    w: -v * t + u * t + u * v,
}


def impl_d(q, dn, k15=sp.Rational(15, 16), kalpha=alpha):
    dang = sp.Rational(3, 8) * dn / P**5 - k15 * N / (q * P**5) - sp.Rational(3, 2) / (q * P**3)
    dfd = kalpha * rr * fdmp**2 / q
    return dang * fdmp + ang * dfd


F = fdmp * ang
for q, name in [(x, "x=R_ij^2"), (y, "y=R_jk^2"), (w, "w=R_ik^2")]:
    check(f"2. dN/d{name} product rule", sp.expand(sp.diff(N, q) - dnum[q]) == 0)
    check(
        f"3. dF/d{name} (ang and fdmp parts)",
        sp.simplify(sp.diff(F, q) - impl_d(q, dnum[q])) == 0,
    )

# --- 4. C6 dependence --------------------------------------------------------
E_t = s9 * sp.sqrt(cij * cik * cjk) * F
check("4. dE_t/dC_ij == E_t/(2 C_ij)", sp.simplify(sp.diff(E_t, cij) - E_t / (2 * cij)) == 0)

# --- 5. assembled Cartesian gradient at a random triangle -------------------
R = [sp.Matrix(sp.symbols(f"X{i} Y{i} Z{i}")) for i in range(3)]
d2 = lambda A, B: (A - B).dot(A - B)  # noqa: E731
subs_q = {x: d2(R[0], R[1]), y: d2(R[1], R[2]), w: d2(R[0], R[2])}
E_cart = F.subs(subs_q)
pt = {
    **dict(zip(R[0], [sp.Rational(1, 10), sp.Rational(-2, 10), sp.Rational(3, 10)])),
    **dict(zip(R[1], [sp.Rational(71, 10), sp.Rational(4, 10), sp.Rational(-5, 10)])),
    **dict(zip(R[2], [sp.Rational(32, 10), sp.Rational(63, 10), sp.Rational(7, 10)])),
    r0: sp.Rational(1250, 1),
    alpha: 16,
}
dq = {q: impl_d(q, dnum[q]).subs(subs_q) for q in (x, y, w)}
assembled = [[0] * 3 for _ in range(3)]
for (aa, bb), q in [((0, 1), x), ((1, 2), y), ((0, 2), w)]:
    for k in range(3):
        comp = 2 * dq[q] * (R[aa][k] - R[bb][k])
        assembled[aa][k] += comp
        assembled[bb][k] -= comp
worst = max(
    abs(sp.N((sp.diff(E_cart, R[i][k]) - assembled[i][k]).subs(pt), 40))
    for i in range(3)
    for k in range(3)
)
scale = max(abs(sp.N(sp.diff(E_cart, R[i][k]).subs(pt), 40)) for i in range(3) for k in range(3))
check(
    "5. assembled Cartesian gradient == sympy dE/dR",
    worst < 1e-30 * scale,
    f"max|diff| = {float(worst):.2e}, max|g| = {float(scale):.2e}",
)

# --- negative controls -------------------------------------------------------
num_pt = {x: 7.3, y: 9.1, w: 11.7, r0: 30.0, alpha: 16}


def num_zero(expr):
    return abs(float(expr.subs(num_pt))) < 1e-14 * abs(float(sp.diff(F, x).subs(num_pt)))


control("dropped 15/16 term", num_zero(sp.diff(F, x) - impl_d(x, dnum[x], k15=0)))
control("alpha/6 in dfdmp", num_zero(sp.diff(F, x) - impl_d(x, dnum[x], kalpha=alpha / 6)))
control("w-component of dN with x signs", num_zero(sp.diff(F, w) - impl_d(w, dnum[x])))
control(
    "dE/dC = E/C (no 1/2)",
    abs(float((sp.diff(E_t, cij) - E_t / cij).subs({**num_pt, s9: 1, cij: 2, cik: 3, cjk: 5})))
    < 1e-30,
)

if FAILS:
    print(f"\n{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("\nall claims hold")
