#!/usr/bin/env python3
"""Sympy proofs for the Becke partition derivatives in ferric-dft.

Checks the identities `becke_weights_and_grad`
(crates/ferric-dft/src/becke.rs:365) and the grid-response weight builder
`build_atomic_grid_with_response` (crates/ferric-dft/src/grid.rs:673) rely on:

  1. d mu_AB / dR_A and d mu_AB / dR_B (the doc-comment formulas the loop codes)
  2. d f^3 / d nu = f2' f1' f0', f_k' = 1.5 (1 - f_k^2)   (Becke f(x) = (3x - x^3)/2)
  3. d nu / d mu = 1 - 2 a mu                              (size correction)
  4. translational invariance  sum_C dmu/dR_C + grad_r mu = 0, which grid.rs
     uses to rebuild grad_r w from the nuclear derivatives
  5. the assembled quotient-rule gradient dw^A/dR_C (code formula) equals the
     sympy derivative of w^A for a 3-atom, unequal-radius cluster at a
     generic point, and sum_A dw^A/dR_C = 0 (partition of unity).

Negative controls: each claim is re-run against a deliberately broken
formula and must FAIL. Exit status is nonzero on any mismatch.

Run: python3 scripts/dft_becke_weight_grad_sympy.py
"""

import sys

import sympy as sp

FAILS = []


def check(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        FAILS.append(name)


def is_zero(expr):
    return sp.simplify(expr) == 0


def vec(name):
    return sp.Matrix(sp.symbols(f"{name}x {name}y {name}z", real=True))


r, RA, RB = vec("r"), vec("A"), vec("B")
rA = sp.sqrt(((r - RA).T * (r - RA))[0])
rB = sp.sqrt(((r - RB).T * (r - RB))[0])
RAB = sp.sqrt(((RA - RB).T * (RA - RB))[0])
mu = (rA - rB) / RAB

# 1. d mu / dR_A, dR_B exactly as coded (becke.rs:431-435).
code_dmu_dRA = [
    -(r - RA)[k] / (rA * RAB) - mu / RAB**2 * (RA - RB)[k] for k in range(3)
]
code_dmu_dRB = [
    +(r - RB)[k] / (rB * RAB) + mu / RAB**2 * (RA - RB)[k] for k in range(3)
]
ok_a = all(is_zero(sp.diff(mu, RA[k]) - code_dmu_dRA[k]) for k in range(3))
ok_b = all(is_zero(sp.diff(mu, RB[k]) - code_dmu_dRB[k]) for k in range(3))
check("1. dmu_AB/dR_A matches becke.rs", ok_a)
check("1. dmu_AB/dR_B matches becke.rs", ok_b)
# negative control: drop the R_AB-length term
bad = -(r - RA)[0] / (rA * RAB)
check(
    "1. control: dropping the mu/R_AB^2 term is detected",
    not is_zero(sp.diff(mu, RA[0]) - bad),
)

# 4. translational invariance of mu (hence of every s_AB, P_A, w^A).
ok_t = all(
    is_zero(sp.diff(mu, RA[k]) + sp.diff(mu, RB[k]) + sp.diff(mu, r[k]))
    for k in range(3)
)
check("4. sum_C dmu/dR_C + grad_r mu = 0", ok_t)

# 2. derivative of the triply iterated Becke polynomial.
x = sp.Symbol("x", real=True)
f = lambda t: sp.Rational(1, 2) * t * (3 - t**2)  # noqa: E731
f0 = x
f1 = f(f0)
f2 = f(f1)
f3 = f(f2)
fp = lambda t: sp.Rational(3, 2) * (1 - t**2)  # noqa: E731
code_df3 = fp(f2) * fp(f1) * fp(f0)
check("2. d f^3/d nu = f2' f1' f0'", sp.expand(sp.diff(f3, x) - code_df3) == 0)
check(
    "2. control: f'(x) = 3(1-x^2) (no 1/2) is detected",
    sp.expand(sp.diff(f3, x) - 8 * code_df3) != 0,
)

# 3. size correction nu = mu + a (1 - mu^2).
m, a = sp.symbols("m a", real=True)
nu = m + a * (1 - m**2)
check("3. dnu/dmu = 1 - 2 a mu", sp.expand(sp.diff(nu, m) - (1 - 2 * a * m)) == 0)

# 5. full weight gradient on a 3-atom cluster with unequal radii.
# 60-digit Float geometry, generic point. The sympy side differentiates the
# full w^A expression along each nuclear coordinate (1-D restriction
# R_c^k -> R_c^k + t, d/dt at t = 0); the code side re-assembles the gradient
# exactly as becke.rs:384-493 does (pair derivatives -> dP -> dT -> quotient).
DPS = 60
F = lambda v: sp.Float(v, DPS)  # noqa: E731
coords = [
    sp.Matrix([F(0), F(0), F(0)]),
    sp.Matrix([F("2.1"), F("0.3"), F(0)]),
    sp.Matrix([F("-0.4"), F("1.9"), F("0.7")]),
]
radii = [F("0.70"), F("0.35"), F("0.65")]  # C-, H-, N-like Bragg radii (Bohr-ish)
N = 3
pt = sp.Matrix([F("0.5"), F("0.6"), F("0.2")])
tvar = sp.Symbol("t", real=True)


def a_corr(i, j):
    chi = radii[i] / radii[j]
    u = (chi - 1) / (chi + 1)
    return max(F("-0.5"), min(F("0.5"), u / (u**2 - 1)))


def dist(p, q):
    return sp.sqrt(((p - q).T * (p - q))[0])


def weights(R):
    def s_pair(i, j):
        mu_ij = (dist(pt, R[i]) - dist(pt, R[j])) / dist(R[i], R[j])
        nu_ij = mu_ij + a_corr(i, j) * (1 - mu_ij**2)
        return (1 - f(f(f(nu_ij)))) / 2

    P = [sp.Mul(*[s_pair(i, j) for j in range(N) if j != i]) for i in range(N)]
    T = sum(P)
    return [P[i] / T for i in range(N)]


def exact_dw(c, k):
    R = [cc.copy() for cc in coords]
    R[c][k] = R[c][k] + tvar
    return [sp.N(sp.diff(wi, tvar).subs(tvar, 0), DPS) for wi in weights(R)]


exact = {}
for c in range(N):
    for k in range(3):
        for i, v in enumerate(exact_dw(c, k)):
            exact[i, c, k] = v

# code assembly
s_val, ds_dRi, ds_dRj = {}, {}, {}
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        rI, rJ = dist(pt, coords[i]), dist(pt, coords[j])
        Rij = dist(coords[i], coords[j])
        mu_ij = (rI - rJ) / Rij
        ac = a_corr(i, j)
        g0 = mu_ij + ac * (1 - mu_ij**2)
        g1, g2 = f(g0), f(f(g0))
        g3 = f(g2)
        ds_dmu = -fp(g2) * fp(g1) * fp(g0) * (1 - 2 * ac * mu_ij) / 2
        s_val[i, j] = (1 - g3) / 2
        vij = coords[i] - coords[j]
        ds_dRi[i, j] = [
            ds_dmu * (-(pt - coords[i])[k] / (rI * Rij) - mu_ij / Rij**2 * vij[k])
            for k in range(3)
        ]
        ds_dRj[i, j] = [
            ds_dmu * ((pt - coords[j])[k] / (rJ * Rij) + mu_ij / Rij**2 * vij[k])
            for k in range(3)
        ]
Pv = [sp.Mul(*[s_val[i, j] for j in range(N) if j != i]) for i in range(N)]
Tv = sum(Pv)
dP = [[[0] * 3 for _ in range(N)] for _ in range(N)]
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        ratio = Pv[i] / s_val[i, j]
        for k in range(3):
            dP[i][i][k] += ratio * ds_dRi[i, j][k]
            dP[i][j][k] += ratio * ds_dRj[i, j][k]
dT = [[sum(dP[i][c][k] for i in range(N)) for k in range(3)] for c in range(N)]
code_dw = {
    (i, c, k): dP[i][c][k] / Tv - Pv[i] / Tv**2 * dT[c][k]
    for i in range(N)
    for c in range(N)
    for k in range(3)
}

scale = max(abs(v) for v in exact.values())
max_err = max(abs(exact[key] - code_dw[key]) for key in exact)
check(
    f"5. becke_weights_and_grad assembly == sympy dw/dR "
    f"(max err {float(max_err):.1e}, max |dw| {float(scale):.2e})",
    max_err < 1e-40 and scale > 1e-3,
)
pou = max(
    abs(sum(code_dw[i, c, k] for i in range(N))) for c in range(N) for k in range(3)
)
check(f"5. sum_A dw^A/dR_C = 0 (max {float(pou):.1e})", pou < 1e-40)
# grid.rs:646 rebuilds grad_r w^A as -sum_C dw^A/dR_C; check against the
# direct electron-coordinate derivative of w^A.
r_err = 0
for k in range(3):
    ptk = pt.copy()
    ptk[k] = ptk[k] + tvar
    saved, pt = pt, ptk
    direct = [sp.N(sp.diff(wi, tvar).subs(tvar, 0), DPS) for wi in weights(coords)]
    pt = saved
    for i in range(N):
        r_err = max(r_err, abs(direct[i] + sum(code_dw[i, c, k] for c in range(N))))
check(f"4. grad_r w^A = -sum_C dw^A/dR_C (max err {float(r_err):.1e})", r_err < 1e-40)
# negative control: drop the -P_A dT/T^2 normalization term
bad_err = abs(exact[0, 1, 0] - dP[0][1][0] / Tv)
check("5. control: dropping the dT/T^2 term is detected", bad_err > 1e-6)

if FAILS:
    print(f"\n{len(FAILS)} check(s) FAILED: {FAILS}")
    sys.exit(1)
print("\nall Becke identities verified")
