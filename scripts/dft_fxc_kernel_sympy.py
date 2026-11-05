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

e(rho_a, rho_b, s_aa, s_ab, s_bb) is a generic cubic (every monomial of degree
<= 3 with its own symbolic coefficient), so its first and second derivatives
at the point are independent symbols. Negative
controls: mutated code formulas must fail. Exit nonzero on any mismatch.

Run: python3 scripts/dft_fxc_kernel_sympy.py
"""

import sys

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
check("1. dsig_aa / dsig_ab / dsig_bb match fxc.rs", ok)
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
check("2. u^a (dv_rho_a) == d v_rho_a / dt", zero(exact_u_a - sub0(u_a)))
check("2. u^b (dv_rho_b) == d v_rho_b / dt", zero(exact_u_b - sub0(u_b)))
check("3. w^a == d G^a / dt", all(zero(exact_w_a[b] - sub0(w_a[b])) for b in AX))
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
    "4. LDA: d v_rho_a/dt = f_aa d_rho_a + f_ab d_rho_b",
    zero(
        lda_u_a - (sp.diff(elda, Ra, Ra) * dra + sp.diff(elda, Ra, Rb) * drb).subs(at_0)
    ),
)

if FAILS:
    print(f"\n{len(FAILS)} check(s) FAILED: {FAILS}")
    sys.exit(1)
print("\nall f_xc kernel identities verified")
