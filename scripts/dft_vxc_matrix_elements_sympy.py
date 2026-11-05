#!/usr/bin/env python3
"""Sympy proofs: ferric's semilocal V_xc matrix elements are dE_xc/dD.

Checks the AO-matrix assembly in crates/ferric-dft/src/vxc.rs:
  - closed shell `semilocal_vxc_closed_scratch` (vxc.rs:262): the LDA piece
    w v_rho chi_mu chi_nu, the GGA piece M + M^T with f_b = 2 w v_sigma d_b rho,
    and the tau piece (1/2) w v_tau sum_b d_b chi_mu d_b chi_nu;
  - spin-polarized `semilocal_vxc_polarized_scratch` (vxc.rs:542): libxc
    packing (sigma_aa, sigma_ab, sigma_bb), G^a_b = 2 v_saa d_b rho_a +
    v_sab d_b rho_b (cross coefficient 1, not 2), per-spin tau piece;
  - the matrix-element form equals the functional-derivative form
    v_xc = v_rho - 2 div(v_sigma grad rho) up to a pure divergence
    (integration by parts), so both define the same Fock matrix.

The integrand is a GENERIC sympy Function e(rho, sigma, tau) (resp. of the 7
polarized variables), evaluated at one grid point with two AOs whose values
and gradients are free symbols, so nothing depends on a particular
functional. D_mu,nu and D_nu,mu are independent symbols, matching the
non-symmetrized derivative the code assembles before its final 1/2(V+V^T).

Negative controls: each identity is re-checked against a mutated code
formula and must fail. Exit status is nonzero on any mismatch.

Run: python3 scripts/dft_vxc_matrix_elements_sympy.py
"""

import sys

import sympy as sp

FAILS = []


def check(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        FAILS.append(name)


NAO = 2
AX = range(3)
chi = sp.symbols("chi0:2", real=True)
dchi = [[sp.Symbol(f"g{m}{b}", real=True) for b in AX] for m in range(NAO)]


def dmat(name):
    return [
        [sp.Symbol(f"{name}{m}{n}", real=True) for n in range(NAO)] for m in range(NAO)
    ]


def density(D):
    rho = sum(D[m][n] * chi[m] * chi[n] for m in range(NAO) for n in range(NAO))
    grad = [
        sum(
            D[m][n] * (dchi[m][b] * chi[n] + chi[m] * dchi[n][b])
            for m in range(NAO)
            for n in range(NAO)
        )
        for b in AX
    ]
    tau = sp.Rational(1, 2) * sum(
        D[m][n] * dchi[m][b] * dchi[n][b]
        for b in AX
        for m in range(NAO)
        for n in range(NAO)
    )
    return rho, grad, tau


def dot(u, v):
    return sum(ui * vi for ui, vi in zip(u, v))


def zero(e):
    return sp.simplify(sp.expand(e)) == 0


# ── closed shell ───────────────────────────────────────────────────────────
D = dmat("D")
rho, grad, tau = density(D)
sigma = dot(grad, grad)
R, S, T = sp.symbols("R S T", real=True)
efun = sp.Function("e")(R, S, T)  # w * rho * eps_xc at the point (w absorbed)
at = {R: rho, S: sigma, T: tau}
v_r, v_s, v_t = (sp.diff(efun, X) for X in (R, S, T))


def closed_code(m, n, gga_fac=2, tau_fac=sp.Rational(1, 2)):
    f_b = [gga_fac * v_s * grad[b] for b in AX]
    lda = v_r * chi[m] * chi[n]
    gga = sum(
        f_b[b] * (chi[m] * dchi[n][b] + chi[n] * dchi[m][b]) for b in AX
    )  # M + M^T
    mgga = tau_fac * v_t * sum(dchi[m][b] * dchi[n][b] for b in AX)
    return lda + gga + mgga


E = efun.subs(at)
ok = all(
    zero(sp.diff(E, D[m][n]) - closed_code(m, n).subs(at))
    for m in range(NAO)
    for n in range(NAO)
)
check("closed: V_mn (LDA + 2 v_sigma M+M^T + 1/2 v_tau) == dE/dD_mn", ok)
check(
    "closed control: GGA factor 1 instead of 2 is detected",
    not zero(sp.diff(E, D[0][1]) - closed_code(0, 1, gga_fac=1).subs(at)),
)
check(
    "closed control: tau factor 1 instead of 1/2 is detected",
    not zero(sp.diff(E, D[0][1]) - closed_code(0, 1, tau_fac=1).subs(at)),
)

# ── spin polarized (libxc packing) ──────────────────────────────────────────
Da, Db = dmat("A"), dmat("B")
ra, ga, ta = density(Da)
rb, gb, tb = density(Db)
Ra, Rb, Saa, Sab, Sbb, Ta, Tb = sp.symbols("Ra Rb Saa Sab Sbb Ta Tb", real=True)
epol = sp.Function("e")(Ra, Rb, Saa, Sab, Sbb, Ta, Tb)
atp = {
    Ra: ra,
    Rb: rb,
    Saa: dot(ga, ga),
    Sab: dot(ga, gb),
    Sbb: dot(gb, gb),
    Ta: ta,
    Tb: tb,
}
Ep = epol.subs(atp)
dv = {X: sp.diff(epol, X) for X in (Ra, Rb, Saa, Sab, Sbb, Ta, Tb)}


def pol_code(m, n, spin, cross_fac=1):
    if spin == "a":
        vr, vself, vt, gs, gc = dv[Ra], dv[Saa], dv[Ta], ga, gb
    else:
        vr, vself, vt, gs, gc = dv[Rb], dv[Sbb], dv[Tb], gb, ga
    G = [2 * vself * gs[b] + cross_fac * dv[Sab] * gc[b] for b in AX]
    return (
        vr * chi[m] * chi[n]
        + sum(G[b] * (chi[m] * dchi[n][b] + chi[n] * dchi[m][b]) for b in AX)
        + sp.Rational(1, 2) * vt * sum(dchi[m][b] * dchi[n][b] for b in AX)
    )


ok_a = all(
    zero(sp.diff(Ep, Da[m][n]) - pol_code(m, n, "a").subs(atp))
    for m in range(NAO)
    for n in range(NAO)
)
ok_b = all(
    zero(sp.diff(Ep, Db[m][n]) - pol_code(m, n, "b").subs(atp))
    for m in range(NAO)
    for n in range(NAO)
)
check(
    "polarized: V^alpha_mn == dE/dD^alpha_mn (G^a = 2 v_saa grad rho_a + v_sab grad rho_b)",
    ok_a,
)
check("polarized: V^beta_mn == dE/dD^beta_mn", ok_b)
check(
    "polarized control: cross coefficient 2 v_sab is detected",
    not zero(sp.diff(Ep, Da[0][1]) - pol_code(0, 1, "a", cross_fac=2).subs(atp)),
)

# ── integration by parts: matrix-element form vs potential form ─────────────
x, y, z = sp.symbols("x y z", real=True)
X3 = (x, y, z)
fn = lambda name: sp.Function(name)(x, y, z)  # noqa: E731
vr3, vs3, rho3, cm, cn = fn("v_rho"), fn("v_sigma"), fn("rho"), fn("chi_m"), fn("chi_n")
grad3 = lambda u: [sp.diff(u, q) for q in X3]  # noqa: E731
matrix_form = vr3 * cm * cn + 2 * vs3 * dot(grad3(rho3), grad3(cm * cn))
v_xc = vr3 - 2 * sum(sp.diff(vs3 * sp.diff(rho3, q), q) for q in X3)
potential_form = v_xc * cm * cn
flux = [2 * vs3 * sp.diff(rho3, q) * cm * cn for q in X3]
divergence = sum(sp.diff(flux[i], X3[i]) for i in range(3))
check(
    "IBP: matrix form - (v_rho - 2 div(v_sigma grad rho)) chi chi == div(2 v_sigma grad rho chi chi)",
    zero(matrix_form - potential_form - divergence),
)
check(
    "IBP control: v_xc with -div(...) (factor 1) is detected",
    not zero(
        matrix_form
        - (vr3 - sum(sp.diff(vs3 * sp.diff(rho3, q), q) for q in X3)) * cm * cn
        - divergence
    ),
)

if FAILS:
    print(f"\n{len(FAILS)} check(s) FAILED: {FAILS}")
    sys.exit(1)
print("\nall V_xc matrix-element identities verified")
