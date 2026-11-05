#!/usr/bin/env python3
"""Sympy proofs: ferric's per-point VV10 potentials are dE_nl/d(rho, sigma).

Checks `vv10_internal_screened` (crates/ferric-dft/src/vv10.rs:908) on a
two-point grid (self pairs included, as the dense loop includes p == i):

  E_nl = sum_i W_i rho_i [beta + 1/2 f_i],
  f_i  = -3/2 sum_p W_p rho_p / (g_i g_p (g_i + g_p)),  g = omega0 R^2 + kappa
  omega0 = sqrt(C (sigma/rho^2)^2 + 4 pi/3 rho),  kappa = k_vv rho^(1/6)

and the code's potentials
  vrho_i = beta + f_i + 3/2 (u_i * rho dkappa/drho + w_i * rho domega0/drho)
  vsig_i = 3/2 w_i * rho domega0/dsigma
with u_i = sum_p t_ip (1/g_i + 1/g_t), w_i = sum_p t_ip (1/g_i + 1/g_t) R^2,
t_ip = W_p rho_p / (g_i g_p g_t). Note the code's dw0_dr, dk_dr, dw0_dg are
rho TIMES the partial derivative (the rho_i prefactor of E is folded in).

Also checks the module-doc constants: k_vv = b (3 pi / 2) (9 pi)^(-1/6) and
beta = (3/b^2)^(3/4) / 32.

Scope: this proves the per-point (rho, sigma) partials. The AO V_nl matrix
then uses the closed-shell GGA form (vxc.rs identity, see
dft_vxc_matrix_elements_sympy.py). The measured residual FD(E) - Tr[V X] of
-1.31e-7 (3.6e-6 relative) on NH2/6-31G (test_xc_blocks.py, xfail; task
T015) is therefore NOT in these partials.

Negative controls: mutated formulas must fail. Exit nonzero on mismatch.

Run: python3 scripts/dft_vv10_potential_sympy.py
"""

import sys

import sympy as sp

FAILS = []


def check(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        FAILS.append(name)


pi = sp.pi
b, C = sp.symbols("b C", positive=True)
rho = sp.symbols("rho0:2", positive=True)
sig = sp.symbols("sigma0:2", positive=True)
W = sp.symbols("W0:2", positive=True)
R2 = sp.Symbol("R2", positive=True)  # squared distance between the two points
NP = 2


def r2(i, p):
    return 0 if i == p else R2


k_vv = b * sp.Rational(3, 2) * pi * (9 * pi) ** sp.Rational(-1, 6)
beta = (3 / b**2) ** sp.Rational(3, 4) / 32
check(
    "kappa: k_vv rho^(1/6) == b (3pi/2) (rho/(9pi))^(1/6)",
    sp.simplify(
        k_vv * rho[0] ** sp.Rational(1, 6)
        - b * sp.Rational(3, 2) * pi * (rho[0] / (9 * pi)) ** sp.Rational(1, 6)
    )
    == 0,
)


def w0(i):
    return sp.sqrt(C * (sig[i] / rho[i] ** 2) ** 2 + sp.Rational(4, 3) * pi * rho[i])


def kap(i):
    return k_vv * rho[i] ** sp.Rational(1, 6)


def g(i, p, at):
    return w0(at) * r2(i, p) + kap(at)


f = [
    -sp.Rational(3, 2)
    * sum(
        W[p] * rho[p] / (g(i, p, i) * g(i, p, p) * (g(i, p, i) + g(i, p, p)))
        for p in range(NP)
    )
    for i in range(NP)
]
E = sum(W[i] * rho[i] * (beta + sp.Rational(1, 2) * f[i]) for i in range(NP))

# module doc form: E = sum_i W rho [beta + 1/2 sum_p W rho Phi], Phi = -3/(2 g g'(g+g'))
Phi = lambda i, p: -3 / (2 * g(i, p, i) * g(i, p, p) * (g(i, p, i) + g(i, p, p)))  # noqa: E731
E_doc = sum(
    W[i]
    * rho[i]
    * (beta + sp.Rational(1, 2) * sum(W[p] * rho[p] * Phi(i, p) for p in range(NP)))
    for i in range(NP)
)


def code_potentials(i, rho_factor=True, fac=sp.Rational(3, 2)):
    r, s = rho[i], sig[i]
    w0tmp = C * (s / r**2) ** 2
    w0i = sp.sqrt(w0tmp + sp.Rational(4, 3) * pi * r)
    dw0_dr = (sp.Rational(1, 2) * sp.Rational(4, 3) * pi * r - 2 * w0tmp) / w0i
    dk_dr = kap(i) / 6
    dw0_dg = w0tmp * r / (s * w0i)
    if not rho_factor:  # mutation: plain partials, rho prefactor dropped
        dw0_dr, dk_dr, dw0_dg = dw0_dr / r, dk_dr / r, dw0_dg / r
    u = 0
    wv = 0
    for p in range(NP):
        gi, gp = g(i, p, i), g(i, p, p)
        gt = gi + gp
        tt = W[p] * rho[p] / (gi * gp * gt)
        u += tt * (1 / gi + 1 / gt)
        wv += tt * (1 / gi + 1 / gt) * r2(i, p)
    vrho = beta + f[i] + fac * (u * dk_dr + wv * dw0_dr)
    vsig = fac * wv * dw0_dg
    return vrho, vsig


rng_vals = {
    b: sp.Rational(59, 10),
    C: sp.Rational(1, 100),
    rho[0]: sp.Rational(3, 10),
    rho[1]: sp.Rational(1, 25),
    sig[0]: sp.Rational(2, 10),
    sig[1]: sp.Rational(3, 1000),
    W[0]: sp.Rational(7, 100),
    W[1]: sp.Rational(13, 10),
    R2: sp.Rational(17, 10),
}


def num(e):
    return sp.N(e.subs(rng_vals), 50)


check("E code form == module-doc Phi form", abs(num(E - E_doc)) < sp.Float("1e-45"))

for name, kwargs in [
    ("code", {}),
    ("control: rho prefactor dropped", {"rho_factor": False}),
    ("control: 3/2 -> 3/4 chain factor", {"fac": sp.Rational(3, 4)}),
]:
    err = 0
    scale = 0
    for i in range(NP):
        vr, vs = code_potentials(i, **kwargs)
        dEr = num(sp.diff(E, rho[i]))
        dEs = num(sp.diff(E, sig[i]))
        err = max(err, abs(dEr - num(W[i] * vr)), abs(dEs - num(W[i] * vs)))
        scale = max(scale, abs(dEr), abs(dEs))
    if name == "code":
        check(
            f"W_i vrho_i, W_i vsig_i == dE/drho_i, dE/dsigma_i (max err {float(err):.1e}, scale {float(scale):.2e})",
            err < sp.Float("1e-40") and scale > sp.Float("1e-6"),
        )
    else:
        check(f"{name} is detected (err {float(err):.1e})", err > sp.Float("1e-8"))

if FAILS:
    print(f"\n{len(FAILS)} check(s) FAILED: {FAILS}")
    sys.exit(1)
print("\nall VV10 potential identities verified")
