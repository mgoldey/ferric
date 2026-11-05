#!/usr/bin/env python3
"""Sympy proofs: the meta-GGA (and GGA) AO-derivative and grid-response terms
of ferric's XC nuclear gradient.

Model: two real Gaussian AOs, a p_x on nucleus A and an s on nucleus B,
evaluated at a free point r, with a symmetric density matrix D. Everything
is differentiated symbolically and compared at an exact rational point
(60 digits). Checks, against crates/ferric-dft/src/gradient.rs:

  1. tau = 1/2 sum_b sum_mn D_mn d_b chi_m d_b chi_n (eval_tau_closed,
     density_on_grid.rs:266);  d tau / d R_A,alpha at fixed (r, D) equals
       - sum_b sum_{m in A} d2_{alpha b} chi_m (D d_b chi)_m
     i.e. the 1/2 in tau cancels against the mu <-> nu doubling. The code
     returns partial = 1/2 sum ... (mgga_tau_partials, gradient.rs:232) and
     scatters -2 * partial (scatter_partials, gradient.rs:108).
  2. d_alpha tau(r) = sum_b sum_m d2_{alpha b} chi_m (D d_b chi)_m
     (tau_spatial_grad, gradient.rs:266) -- no residual 1/2.
  3. translational invariance  d tau/dR_A + d tau/dR_B + grad_r tau = 0
     (what makes the home-atom grid-response term the right correction).
  4. GGA AO-derivative partials (gga_ao_partials, gradient.rs:149) with
     t_rho = v_rho, c_b = 2 v_sigma d_b rho: for E = e(rho, sigma, tau),
       dE/dR_A,alpha = -2 sum_{m in A} [ v_rho d_alpha chi_m (D chi)_m
          + sum_b c_b (d_alpha chi_m (D d_b chi)_m + d2_{alpha b} chi_m (D chi)_m) ]
          + (tau term of 1.)
  5. grid-response home-translation column (mgga_closed_batch hess_col):
       d_alpha (d_b rho) = 2 [ sum_m d_alpha chi_m (D d_b chi)_m
                               + sum_m (D chi)_m d2_{alpha b} chi_m ]
     so  w * 2 v_sigma sum_b d_b rho * hess_col = w v_sigma d_alpha sigma.

Negative controls: the un-cancelled 1/2 (and a halved spatial gradient)
must fail. Exit nonzero on any mismatch.

Run: python3 scripts/dft_mgga_tau_gradient_sympy.py
"""

import sys

import sympy as sp

FAILS = []


def check(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        FAILS.append(name)


X = sp.symbols("x y z", real=True)
A = sp.symbols("Ax Ay Az", real=True)
B = sp.symbols("Bx By Bz", real=True)
alpha_A, alpha_B = sp.Rational(9, 10), sp.Rational(13, 10)
dA = [X[k] - A[k] for k in range(3)]
dB = [X[k] - B[k] for k in range(3)]
chiA = dA[0] * sp.exp(-alpha_A * sum(d * d for d in dA))  # p_x on A
chiB = sp.exp(-alpha_B * sum(d * d for d in dB))  # s on B
chis = [chiA, chiB]
on_A = [True, False]
D = [
    [sp.Rational(7, 10), sp.Rational(-3, 10)],
    [sp.Rational(-3, 10), sp.Rational(11, 10)],
]
NAO = 2
AX = range(3)

d1 = [[sp.diff(c, X[b]) for b in AX] for c in chis]
d2 = [[[sp.diff(c, X[a], X[b]) for b in AX] for a in AX] for c in chis]
Dchi = [sum(D[m][n] * chis[n] for n in range(NAO)) for m in range(NAO)]
Ddchi = [[sum(D[m][n] * d1[n][b] for n in range(NAO)) for b in AX] for m in range(NAO)]

rho = sum(D[m][n] * chis[m] * chis[n] for m in range(NAO) for n in range(NAO))
grad_rho = [sp.diff(rho, X[b]) for b in AX]
sigma = sum(g * g for g in grad_rho)
tau = sp.Rational(1, 2) * sum(
    D[m][n] * d1[m][b] * d1[n][b] for b in AX for m in range(NAO) for n in range(NAO)
)

point = {
    X[0]: sp.Rational(3, 10),
    X[1]: sp.Rational(-2, 10),
    X[2]: sp.Rational(5, 10),
    A[0]: 0,
    A[1]: sp.Rational(1, 10),
    A[2]: 0,
    B[0]: sp.Rational(1, 1),
    B[1]: 0,
    B[2]: sp.Rational(4, 10),
}


def num(e):
    return sp.N(e.subs(point), 60)


TOL = sp.Float("1e-50")

# 1. tau AO-derivative term
err1 = err1_bad = 0
for a in AX:
    exact = num(sp.diff(tau, A[a]))
    partial = sp.Rational(1, 2) * sum(
        d2[m][a][b] * Ddchi[m][b] for b in AX for m in range(NAO) if on_A[m]
    )
    code = -2 * partial
    err1 = max(err1, abs(exact - num(code)))
    err1_bad = max(err1_bad, abs(exact - num(code / 2)))
check(f"1. d tau/dR_A = -2 * mgga_tau_partials (err {float(err1):.1e})", err1 < TOL)
check(
    f"1. control: keeping the 1/2 of tau is detected (err {float(err1_bad):.1e})",
    err1_bad > 1e-6,
)

# 2. spatial gradient of tau
err2 = err2_bad = 0
for a in AX:
    exact = num(sp.diff(tau, X[a]))
    code = sum(d2[m][a][b] * Ddchi[m][b] for b in AX for m in range(NAO))
    err2 = max(err2, abs(exact - num(code)))
    err2_bad = max(err2_bad, abs(exact - num(code / 2)))
check(f"2. d_alpha tau == tau_spatial_grad (err {float(err2):.1e})", err2 < TOL)
check(
    f"2. control: halved spatial gradient is detected (err {float(err2_bad):.1e})",
    err2_bad > 1e-6,
)

# 3. translational invariance of rho, sigma, tau
err3 = 0
for q in (rho, sigma, tau):
    for a in AX:
        err3 = max(
            err3, abs(num(sp.diff(q, A[a]) + sp.diff(q, B[a]) + sp.diff(q, X[a])))
        )
check(
    f"3. d/dR_A + d/dR_B + grad_r = 0 for rho, sigma, tau (err {float(err3):.1e})",
    err3 < TOL,
)

# 4. GGA + tau AO-derivative partials for a generic e(rho, sigma, tau)
Rv, Sv, Tv = sp.symbols("R S T", real=True)
e = sp.Function("e")(Rv, Sv, Tv)
vr, vs, vt = (sp.Symbol(n, real=True) for n in ("vr", "vs", "vt"))
# chain rule with the partial derivatives as free symbols (any functional)
E_lin = lambda q_r, q_s, q_t: vr * q_r + vs * q_s + vt * q_t  # noqa: E731
err4 = 0
for a in AX:
    exact = E_lin(sp.diff(rho, A[a]), sp.diff(sigma, A[a]), sp.diff(tau, A[a]))
    c = [2 * vs * grad_rho[b] for b in AX]
    gga_partial = sum(
        vr * d1[m][a] * Dchi[m]
        + sum(c[b] * (d1[m][a] * Ddchi[m][b] + d2[m][a][b] * Dchi[m]) for b in AX)
        for m in range(NAO)
        if on_A[m]
    )
    tau_partial = (
        sp.Rational(1, 2)
        * vt
        * sum(d2[m][a][b] * Ddchi[m][b] for b in AX for m in range(NAO) if on_A[m])
    )
    code = -2 * gga_partial - 2 * tau_partial
    diff = sp.expand(exact - code)
    for coeff in (diff.coeff(vr), diff.coeff(vs), diff.coeff(vt)):
        err4 = max(err4, abs(num(coeff)))
check(
    f"4. -2 (gga_ao_partials + mgga_tau_partials) == dE/dR_A (err {float(err4):.1e})",
    err4 < TOL,
)

# 5. grid-response hess_col
err5 = 0
for a in AX:
    for b in AX:
        exact = num(sp.diff(grad_rho[b], X[a]))
        hess_col = 2 * (
            sum(d1[m][a] * Ddchi[m][b] for m in range(NAO))
            + sum(Dchi[m] * d2[m][a][b] for m in range(NAO))
        )
        err5 = max(err5, abs(exact - num(hess_col)))
    lhs = num(2 * sum(grad_rho[b] * sp.diff(grad_rho[b], X[a]) for b in AX))
    err5 = max(err5, abs(lhs - num(sp.diff(sigma, X[a]))))
check(
    f"5. hess_col == d_alpha grad rho; 2 grad rho . hess_col == d_alpha sigma (err {float(err5):.1e})",
    err5 < TOL,
)

if FAILS:
    print(f"\n{len(FAILS)} check(s) FAILED: {FAILS}")
    sys.exit(1)
print("\nall meta-GGA gradient identities verified")
