#!/usr/bin/env python3
"""Sympy proofs: ferric's GGA f_xc response is the linearization of V_xc.

Checks `GgaFxcKernel::apply_with_ref` (crates/ferric-dft/src/fxc.rs:314) at
one grid point. The spin-polarized V_xc of vxc.rs enters the AO matrix only
through the per-point scalars v_rho_s and vectors
G^a = 2 v_saa grad rho_a + v_sab grad rho_b,  G^b = 2 v_sbb grad rho_b + v_sab grad rho_a
(the identity proved in dft_vxc_matrix_elements_sympy.py), and the AO
back-projection is linear in (v_rho, G). So delta V^s is exact iff the code's
per-point (u^s, w^s) equal d/dt of (v_rho_s, G^s) along
rho_s -> rho_s + t delta_rho_s, grad rho_s -> grad rho_s + t grad delta_rho_s.

Checked:
  1. delta sigma_aa = 2 grad rho_a . grad d_rho_a,
     delta sigma_ab = grad rho_a . grad d_rho_b + grad rho_b . grad d_rho_a
  2. u^a, u^b (fxc.rs dv_rho_*) from libxc's packed v2rho2 [aa, ab, bb],
     v2rhosigma [a.aa, a.ab, a.bb, b.aa, b.ab, b.bb], v2sigma2 upper triangle
     [aa.aa, aa.ab, aa.bb, ab.ab, ab.bb, bb.bb]
  3. w^a, w^b: the 2 dv_saa grad rho_a + dv_sab grad rho_b pass plus the
     2 v_saa grad d_rho_a + v_sab grad d_rho_b pass
  4. LDA: with e independent of sigma the response is f_rr . d_rho and w = 0
     (LdaFxcKernel, fxc.rs:121).

  5. numerical cross-check (independent route) of the meta-GGA kernel
     (MggaFxcKernel, linearize_point): a concrete 7-variable e with
     alpha-beta and tau couplings at one point; libxc-packed second
     derivatives scattered as scatter_rho_sigma / scatter_tau do, then
     linearize_point's (u, w, t), against the central difference in t of
     (v_rho_s, G^s, v_tau_s). Bar: (h^2/6)|q'''| plus rounding.

Mutations (each turns the script red):
  * v2sigmatau read tau-major (index 3 t + c instead of 2 c + t): check 5 fails;
  * reference pass with v_saa grad d_rho_a instead of 2 v_saa grad d_rho_a in
    the numeric w^a: check 5 fails;
  * symbolic w^a with dv_saa grad rho_a instead of 2 dv_saa grad rho_a:
    "3. w^a" fails.

e(rho_a, rho_b, s_aa, s_ab, s_bb) is a generic cubic (every monomial of degree
<= 3 with its own symbolic coefficient), so its first and second derivatives
at the point are independent symbols. Negative
controls: mutated code formulas must fail. Exit nonzero on any mismatch.

Run: python3 scripts/dft_fxc_kernel_sympy.py
"""

import sys

import mpmath as mp
import sympy as sp

FAILS = []


def check(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        FAILS.append(name)


def zero(e):
    return sp.expand(e) == 0


AX = range(3)
t = sp.Symbol("t", real=True)
ra, rb, dra, drb = sp.symbols("rho_a rho_b drho_a drho_b", real=True)
ga = [sp.Symbol(f"ga{b}", real=True) for b in AX]
gb = [sp.Symbol(f"gb{b}", real=True) for b in AX]
gda = [sp.Symbol(f"gda{b}", real=True) for b in AX]
gdb = [sp.Symbol(f"gdb{b}", real=True) for b in AX]


def dot(u, v):
    return sum(x * y for x, y in zip(u, v))


Ra, Rb, Saa, Sab, Sbb = VARS = sp.symbols("Ra Rb Saa Sab Sbb", real=True)


def generic_cubic(variables, prefix):
    """Sum of every monomial of degree <= 3 with its own symbolic coefficient.

    Its first and second derivatives at a point are independent generic
    values, so a chain-rule identity linear in them that holds for this e
    holds for every smooth functional.
    """
    mons = sorted(sp.itermonomials(variables, 3), key=sp.default_sort_key)
    coeffs = sp.symbols(f"{prefix}0:{len(mons)}", real=True)
    return sum(c * m for c, m in zip(coeffs, mons))


e = generic_cubic(VARS, "c")
vR = {X: sp.diff(e, X) for X in VARS}
H = {(X, Y): sp.diff(e, X, Y) for X in VARS for Y in VARS}

# perturbed point
pa = [ga[b] + t * gda[b] for b in AX]
pb = [gb[b] + t * gdb[b] for b in AX]
at_t = {
    Ra: ra + t * dra,
    Rb: rb + t * drb,
    Saa: dot(pa, pa),
    Sab: dot(pa, pb),
    Sbb: dot(pb, pb),
}
at_0 = {Ra: ra, Rb: rb, Saa: dot(ga, ga), Sab: dot(ga, gb), Sbb: dot(gb, gb)}


# 1. sigma-channel perturbations (fxc.rs dsig_*)
code_dsig = {
    Saa: 2 * dot(ga, gda),
    Sbb: 2 * dot(gb, gdb),
    Sab: dot(ga, gdb) + dot(gb, gda),
}
ok = all(
    sp.expand(sp.diff(at_t[X], t).subs(t, 0) - code_dsig[X]) == 0
    for X in (Saa, Sab, Sbb)
)
check("1. dsig_aa / dsig_ab / dsig_bb match fxc.rs (claim dsigma)", ok)
check(
    "1. control: dsig_ab = 2 grad rho_a . grad d_rho_b is detected",
    sp.expand(sp.diff(at_t[Sab], t).subs(t, 0) - 2 * dot(ga, gdb)) != 0,
)

# libxc packing as read by fxc.rs
v2rho2 = [H[Ra, Ra], H[Ra, Rb], H[Rb, Rb]]
v2rhosigma = [H[Ra, Saa], H[Ra, Sab], H[Ra, Sbb], H[Rb, Saa], H[Rb, Sab], H[Rb, Sbb]]
v2sigma2 = [
    H[Saa, Saa],
    H[Saa, Sab],
    H[Saa, Sbb],
    H[Sab, Sab],
    H[Sab, Sbb],
    H[Sbb, Sbb],
]
ds_aa, ds_ab, ds_bb = code_dsig[Saa], code_dsig[Sab], code_dsig[Sbb]


def code_u_w(cross2=False, drop_ref_pass=False, swap_packing=False):
    frr_aa, frr_ab, frr_bb = v2rho2
    frs_a_aa, frs_a_ab, frs_a_bb, frs_b_aa, frs_b_ab, frs_b_bb = v2rhosigma
    fss_aa_aa, fss_aa_ab, fss_aa_bb, fss_ab_ab, fss_ab_bb, fss_bb_bb = v2sigma2
    if swap_packing:  # mutation: read v2sigma2 slots 2 and 3 in the wrong order
        fss_aa_bb, fss_ab_ab = fss_ab_ab, fss_aa_bb
    dv_rho_a = (
        frr_aa * dra
        + frr_ab * drb
        + frs_a_aa * ds_aa
        + frs_a_ab * ds_ab
        + frs_a_bb * ds_bb
    )
    dv_rho_b = (
        frr_ab * dra
        + frr_bb * drb
        + frs_b_aa * ds_aa
        + frs_b_ab * ds_ab
        + frs_b_bb * ds_bb
    )
    dv_sig_aa = (
        frs_a_aa * dra
        + frs_b_aa * drb
        + fss_aa_aa * ds_aa
        + fss_aa_ab * ds_ab
        + fss_aa_bb * ds_bb
    )
    dv_sig_ab = (
        frs_a_ab * dra
        + frs_b_ab * drb
        + fss_aa_ab * ds_aa
        + fss_ab_ab * ds_ab
        + fss_ab_bb * ds_bb
    )
    dv_sig_bb = (
        frs_a_bb * dra
        + frs_b_bb * drb
        + fss_aa_bb * ds_aa
        + fss_ab_bb * ds_ab
        + fss_bb_bb * ds_bb
    )
    c = 2 if cross2 else 1
    w_a = [2 * dv_sig_aa * ga[b] + c * dv_sig_ab * gb[b] for b in AX]
    w_b = [2 * dv_sig_bb * gb[b] + c * dv_sig_ab * ga[b] for b in AX]
    if not drop_ref_pass:
        vaa, vab, vbb = vR[Saa], vR[Sab], vR[Sbb]
        w_a = [w_a[b] + 2 * vaa * gda[b] + vab * gdb[b] for b in AX]
        w_b = [w_b[b] + 2 * vbb * gdb[b] + vab * gda[b] for b in AX]
    return dv_rho_a, dv_rho_b, w_a, w_b


# exact linearization of (v_rho_s, G^s)
# G^s at the perturbed point (both v_sigma and the gradients move with t)
G_a_t = [2 * vR[Saa] * pa[b] + vR[Sab] * pb[b] for b in AX]
G_b_t = [2 * vR[Sbb] * pb[b] + vR[Sab] * pa[b] for b in AX]


def lin(expr):
    return sp.diff(expr.subs(at_t, simultaneous=True), t).subs(t, 0)


exact_u_a, exact_u_b = lin(vR[Ra]), lin(vR[Rb])
exact_w_a = [lin(G_a_t[b]) for b in AX]
exact_w_b = [lin(G_b_t[b]) for b in AX]

u_a, u_b, w_a, w_b = code_u_w()
sub0 = lambda v: v.subs(at_0, simultaneous=True)  # noqa: E731
check("2. u^a (dv_rho_a) == d v_rho_a / dt (claim u)", zero(exact_u_a - sub0(u_a)))
check("2. u^b (dv_rho_b) == d v_rho_b / dt", zero(exact_u_b - sub0(u_b)))
check(
    "3. w^a == d G^a / dt (claim w)", all(zero(exact_w_a[b] - sub0(w_a[b])) for b in AX)
)
check("3. w^b == d G^b / dt", all(zero(exact_w_b[b] - sub0(w_b[b])) for b in AX))
_, _, w_a_bad, _ = code_u_w(cross2=True)
check(
    "3. control: 2 dv_sab cross coefficient is detected",
    not zero(exact_w_a[0] - sub0(w_a_bad[0])),
)
_, _, w_a_bad, _ = code_u_w(drop_ref_pass=True)
check(
    "3. control: dropping the v_sigma grad d_rho pass is detected",
    not zero(exact_w_a[0] - sub0(w_a_bad[0])),
)

_, _, w_a_bad, _ = code_u_w(swap_packing=True)
check(
    "3. control: v2sigma2 slots [aa.bb, ab.ab] swapped is detected",
    not zero(exact_w_a[0] - sub0(w_a_bad[0])),
)

# 4. LDA limit: e independent of sigma
elda = generic_cubic((Ra, Rb), "l")
lda_u_a = sp.diff(elda, Ra).subs(at_t, simultaneous=True).diff(t).subs(t, 0)
check(
    "4. LDA: d v_rho_a/dt = f_aa d_rho_a + f_ab d_rho_b (claim lda)",
    zero(
        lda_u_a - (sp.diff(elda, Ra, Ra) * dra + sp.diff(elda, Ra, Rb) * drb).subs(at_0)
    ),
)

# 5. Numerical cross-check of the meta-GGA kernel (MggaFxcKernel's
# linearize_point; GGA and LDA are its tau-free and sigma-free cases).
# A concrete 7-variable e(rho_a, rho_b, s_aa, s_ab, s_bb, tau_a, tau_b) with
# alpha-beta couplings, one grid point, and a perturbation of both spins.
# Code route: libxc's packed second derivatives (v2rho2, v2rhosigma,
# v2sigma2, v2rhotau, v2sigmatau, v2tau2), scattered into the 7x7 Hessian as
# scatter_rho_sigma / scatter_tau do, then linearize_point. Independent route:
# the central difference in t of (v_rho_s, G^s, v_tau_s) evaluated at
# rho_s + t d_rho_s, grad rho_s + t grad d_rho_s, tau_s + t d_tau_s. Bar:
# (h^2/6) |q'''| estimated by a third difference at step H, plus rounding.
mp.mp.dps = 40
X7 = sp.symbols("xra xrb xsaa xsab xsbb xta xtb", positive=True)
xra, xrb, xsaa, xsab, xsbb, xta, xtb = X7
E7 = (
    -(xra ** sp.Rational(4, 3) + xrb ** sp.Rational(4, 3))
    - sp.Rational(1, 5)
    * (xsaa * xra ** sp.Rational(-4, 3) + xsbb * xrb ** sp.Rational(-4, 3))
    / (1 + sp.Rational(3, 10) * (xsaa + xsbb) * (xra + xrb) ** sp.Rational(-8, 3))
    - sp.Rational(1, 10) * xra * xrb / (xra + xrb) ** sp.Rational(2, 3)
    - sp.Rational(3, 100) * xsab * (xra + xrb) ** sp.Rational(-4, 3)
    + sp.Rational(1, 20)
    * (xta**2 * xra ** sp.Rational(-5, 3) + xtb**2 * xrb ** sp.Rational(-5, 3))
    + sp.Rational(1, 50) * xta * xtb * xsab * (xra + xrb) ** sp.Rational(-3, 1)
    + sp.Rational(1, 40) * xta * xsbb * (xra + xrb) ** sp.Rational(-4, 1)
)
d1_7 = [sp.lambdify(X7, sp.diff(E7, v), "mpmath") for v in X7]
d2_7 = {
    (i, j): sp.lambdify(X7, sp.diff(E7, X7[i], X7[j]), "mpmath")
    for i in range(7)
    for j in range(7)
}
I_RA, I_RB, I_SAA, I_SAB, I_SBB, I_TA, I_TB = range(7)


def mf(v):
    return mp.mpf(v)


p_rho = [mf("0.31"), mf("0.17")]
p_grad = [[mf("0.12"), mf("-0.08"), mf("0.21")], [mf("-0.05"), mf("0.14"), mf("0.09")]]
p_tau = [mf("0.23"), mf("0.11")]
p_drho = [mf("0.07"), mf("-0.04")]
p_dgrad = [[mf("0.03"), mf("0.05"), mf("-0.02")], [mf("-0.06"), mf("0.01"), mf("0.04")]]
p_dtau = [mf("-0.03"), mf("0.05")]


def point_vars(t):
    ga_t = [p_grad[0][c] + t * p_dgrad[0][c] for c in AX]
    gb_t = [p_grad[1][c] + t * p_dgrad[1][c] for c in AX]
    return (
        p_rho[0] + t * p_drho[0],
        p_rho[1] + t * p_drho[1],
        dot(ga_t, ga_t),
        dot(ga_t, gb_t),
        dot(gb_t, gb_t),
        p_tau[0] + t * p_dtau[0],
        p_tau[1] + t * p_dtau[1],
    ), (ga_t, gb_t)


def potential_at(t):
    """(v_rho_a, v_rho_b, G^a (3), G^b (3), v_tau_a, v_tau_b) at parameter t."""
    x, (ga_t, gb_t) = point_vars(t)
    v = [f(*x) for f in d1_7]
    G_a = [2 * v[I_SAA] * ga_t[c] + v[I_SAB] * gb_t[c] for c in AX]
    G_b = [2 * v[I_SBB] * gb_t[c] + v[I_SAB] * ga_t[c] for c in AX]
    return [v[I_RA], v[I_RB], *G_a, *G_b, v[I_TA], v[I_TB]]


x0, _ = point_vars(0)
hx = {k: f(*x0) for k, f in d2_7.items()}
# libxc packing (polarized), as the arrays fxc.rs reads
v2rho2_n = [hx[I_RA, I_RA], hx[I_RA, I_RB], hx[I_RB, I_RB]]
v2rhosigma_n = [hx[I_RA + s, I_SAA + c] for s in range(2) for c in range(3)]
v2sigma2_n = [
    hx[I_SAA + c1, I_SAA + c2]
    for c1, c2 in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
]
v2rhotau_n = [hx[I_RA + s, I_TA + t_] for s in range(2) for t_ in range(2)]
v2sigmatau_n = [hx[I_SAA + c, I_TA + t_] for c in range(3) for t_ in range(2)]
v2tau2_n = [hx[I_TA, I_TA], hx[I_TA, I_TB], hx[I_TB, I_TB]]
vs_n = [f(*x0) for f in d1_7[I_SAA : I_SBB + 1]]


def add_sym(h, i, j, v):
    h[i][j] += v
    if i != j:
        h[j][i] += v


def code_hessian():
    """scatter_rho_sigma + scatter_tau (fxc.rs)."""
    h = [[mp.mpf(0)] * 7 for _ in range(7)]
    rr, rs, ss = v2rho2_n, v2rhosigma_n, v2sigma2_n
    add_sym(h, I_RA, I_RA, rr[0])
    add_sym(h, I_RA, I_RB, rr[1])
    add_sym(h, I_RB, I_RB, rr[2])
    for s in range(2):
        for c in range(3):
            add_sym(h, I_RA + s, I_SAA + c, rs[3 * s + c])
    for k, (c1, c2) in enumerate([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]):
        add_sym(h, I_SAA + c1, I_SAA + c2, ss[k])
    rt, st, tt = v2rhotau_n, v2sigmatau_n, v2tau2_n
    for t_ in range(2):
        for s in range(2):
            add_sym(h, I_RA + s, I_TA + t_, rt[2 * s + t_])
        for c in range(3):
            add_sym(h, I_SAA + c, I_TA + t_, st[2 * c + t_])
    add_sym(h, I_TA, I_TA, tt[0])
    add_sym(h, I_TA, I_TB, tt[1])
    add_sym(h, I_TB, I_TB, tt[2])
    return h


def linearize_point(h, vs):
    """fxc.rs linearize_point: (u^a, u^b, w^a, w^b, t^a, t^b)."""
    ga, gb = p_grad
    da, db = p_dgrad
    dx = [0] * 7
    dx[I_RA] = p_drho[0]
    dx[I_RB] = p_drho[1]
    dx[I_SAA] = 2 * dot(ga, da)
    dx[I_SAB] = dot(ga, db) + dot(gb, da)
    dx[I_SBB] = 2 * dot(gb, db)
    dx[I_TA] = p_dtau[0]
    dx[I_TB] = p_dtau[1]
    dv = [sum(h[i][j] * dx[j] for j in range(7)) for i in range(7)]
    w_a = [
        2 * dv[I_SAA] * ga[c] + dv[I_SAB] * gb[c] + 2 * vs[0] * da[c] + vs[1] * db[c]
        for c in AX
    ]
    w_b = [
        2 * dv[I_SBB] * gb[c] + dv[I_SAB] * ga[c] + 2 * vs[2] * db[c] + vs[1] * da[c]
        for c in AX
    ]
    return [dv[I_RA], dv[I_RB], *w_a, *w_b, dv[I_TA], dv[I_TB]]


H_FD = mp.mpf("1e-8")
H3 = mp.mpf("1e-3")
q_p, q_m = potential_at(H_FD), potential_at(-H_FD)
q_2p, q_1p, q_1m, q_2m = (potential_at(s * H3) for s in (2, 1, -1, -2))
code_lin = linearize_point(code_hessian(), vs_n)
worst = absmax = 0
for k in range(len(code_lin)):
    fd = (q_p[k] - q_m[k]) / (2 * H_FD)
    d3 = (q_2p[k] - 2 * q_1p[k] + 2 * q_1m[k] - q_2m[k]) / (2 * H3**3)
    bar = 10 * H_FD**2 / 6 * abs(d3) + mp.mpf(10) ** (5 - mp.mp.dps) / H_FD
    worst = max(worst, abs(fd - code_lin[k]) / bar)
    absmax = max(absmax, abs(fd - code_lin[k]))
check(
    f"5. numeric: linearize_point (u, w, t) == central difference of (v_rho, G, v_tau) "
    f"(max |diff| {mp.nstr(absmax, 3)}, / bar {mp.nstr(worst, 3)}) (claim mgga)",
    worst < 1,
)
scale = max(abs(c) for c in code_lin)
check(
    f"5. numeric: response is not trivially small (max |u, w, t| {mp.nstr(scale, 3)})",
    scale > 1e-3,
)

if FAILS:
    print(f"\n{len(FAILS)} check(s) FAILED: {FAILS}")
    sys.exit(1)
print("\nall f_xc kernel identities verified")
