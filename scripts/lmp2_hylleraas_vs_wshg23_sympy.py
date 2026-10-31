#!/usr/bin/env python3
"""Sympy proofs: the WSHG23 local-MP2 energy vs ferric's Stage D (#605) energy.

Finding this script is built around: they are the SAME functional. WSHG23
(JCTC 19, 7577), Eq 10 and sec. 3.1, as summarized in
wiki/amplitude-threshold-lmp2.md sec. 2c, evaluates the Hylleraas functional
with the final CG residual folded in. ferric computed the plain linear form
until Stage D. So the comparison has three parts:

  ferric pre-D   E_lin = <t~, J>                   stop: relres < 1e-11
  WSHG23         E_H   = <t~, 2J + A t>            stop: ||r|| < 2.5 sqrt(eps)
  ferric Stage D E_H   = <t~, 2J + A t>            stop: relres < 1e-6, bound-gated

Conventions (crates/ferric-mp2/src/lmp2_amplitude.rs): t~ = (2 - S) t with S the
a<->b exchange swap, R = J + A t the residual, A the SPD pattern-projected Fock
superoperator, stationary point t* = -A^-1 J.

Claim 1 holds in any dimension, by a line-restriction argument in symbols for
the bilinear forms. Claim 2 is a one-line identity, stated for the record. Claim
3 is a derivation in comments, checked numerically in claim 6. Claims 4-6 use an
explicit one-pair, two-virtual model. Claims 1, 4, 5, 6 assert; negative
controls are included.
Claim 7 evaluates the two stopping rules and is arithmetic on inputs quoted from
the wiki and #605, not a proof.

Run: python scripts/lmp2_hylleraas_vs_wshg23_sympy.py
"""

import sympy as sp


def section(title):
    print(f"\n=== {title}")


# ---------------------------------------------------------------------------
# Any dimension: let B(x, y) = <(2-S)x, A y> and L(x) = <(2-S)x, J>. B is
# symmetric exactly when S and A commute and S preserves the pattern; claim 5
# shows what breaks otherwise. Stationarity of t*: B(x, t*) = -L(x) for all x
# in the pattern subspace, with x = t* and x = d as the two cases we need.
# Everything is then a polynomial in the step s along t = t* + s d.
# ---------------------------------------------------------------------------
s = sp.symbols("s", real=True)
L_star, L_d, B_ss, B_sd, B_dd = sp.symbols("L_star L_d B_ss B_sd B_dd", real=True)


def L_of_t():
    return L_star + s * L_d


def B_tt():
    return B_ss + 2 * s * B_sd + s**2 * B_dd


stationarity = {B_ss: -L_star, B_sd: -L_d}

E_lin = L_of_t()
E_H = 2 * L_of_t() + B_tt()
E_star = L_star

section("Claim 1: the forms agree at t*; E_H has no term linear in the error")
E_H_on_line = sp.expand(E_H.subs(stationarity))
assert sp.expand(E_H_on_line - (E_star + s**2 * B_dd)) == 0
assert sp.expand(E_lin.subs(s, 0) - E_star) == 0
assert E_H_on_line.coeff(s, 1) == 0
assert sp.expand(E_lin).coeff(s, 1) == L_d, "E_lin should keep a first-order term"
print("E_H(t*+s d)  =", E_H_on_line, " (no s^1 term)")
print("E_lin(t*+s d)=", E_lin, " (first order in s)")
# Negative control: without the stationarity condition a linear term survives.
assert sp.expand(E_H).coeff(s, 1) != 0
print("control: dropping stationarity leaves s^1 coeff =", sp.expand(E_H).coeff(s, 1))

section("Claim 2: 'residual folded in' IS the Hylleraas functional")
# With R = J + A t, <t~, J> + <t~, R> = <t~, 2J + A t> term by term, which is
# what WSHG23's folded energy and ferric's hylleraas_energy doc both state.
t_J, t_At = sp.symbols("tJ tAt", real=True)  # <t~, J>, <t~, A t>
assert sp.expand((t_J + (t_J + t_At)) - (2 * t_J + t_At)) == 0
print("E_lin + <t~,R> == E_H identically")

section("Claim 3: E_H - E* = <(2-S) A^-1 R, R> >= 0, bounded by 3 ||R||^2 / lmin")
# R = J + A(t* + d) = A d, so d = A^-1 R and B(d,d) = <(2-S)A^-1 R, R>.
# On commuting S, A, the eigenvalues of (2-S)A^-1 are c/l with c in {1, 3} and
# l in spec(A). The 3 is ||2-S||, from S^2 = 1, and it is checked in claim 4.
print("Hence ||R||^2/lmax <= E_H - E* <= 3 ||R||^2/lmin (lmin of the PROJECTED A).")
print(
    "Interlacing gives lmin(PAP on range P) >= lmin(A); the code's 2(e_LUMO - e_HOMO)"
)
print("estimate is not derived here (this model has diagonal f_oo). See the #605 test.")

# ---------------------------------------------------------------------------
# Explicit model: one pair (i != j), two virtuals. Amplitudes t_ab live in a
# 4-vector ordered (11, 12, 21, 22). S: t_ab -> t_ba. The occupied Fock is taken
# diagonal here, so f_oo off-diagonal couplings are outside this model.
# ---------------------------------------------------------------------------
f1, f2, g = sp.symbols("f1 f2 g", real=True)
e = sp.symbols("e", real=True)  # f_ii + f_jj
J11, J12, J21, J22 = sp.symbols("J11 J12 J21 J22", real=True)
F = sp.Matrix([[f1, g], [g, f2]])
A = (
    sp.kronecker_product(F, sp.eye(2))
    + sp.kronecker_product(sp.eye(2), F)
    - e * sp.eye(4)
)
S = sp.Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
C = 2 * sp.eye(4) - S
J = sp.Matrix([J11, J12, J21, J22])

section("Claim 4: model structure")
assert S * S == sp.eye(4)
assert sp.simplify(S * A - A * S) == sp.zeros(4)
assert sp.simplify((C * A).T - C * A) == sp.zeros(4)
assert sorted(C.eigenvals().keys()) == [1, 3]
# The exchange sign is pinned against the energy expression the code evaluates,
# E = sum (2 t_ab - t_ba) J_ab: 2+S has the same spectrum and would pass the rest.
_t = sp.Matrix(sp.symbols("t11 t12 t21 t22", real=True))
_code_energy = sum(
    2 * _t[a * 2 + b] * J[a * 2 + b] - _t[b * 2 + a] * J[a * 2 + b]
    for a in range(2)
    for b in range(2)
)
assert sp.expand((C * _t).dot(J) - _code_energy) == 0
print("S^2 = 1; [S, A] = 0; (2-S)A symmetric; eig(2-S) = {1, 3}, so ||2-S|| = 3")


def energies(mask):
    """Solve the masked system exactly; return helpers for E_lin and E_H."""
    idx = [k for k, keep in enumerate(mask) if keep]
    P = sp.zeros(4, len(idx))
    for col, k in enumerate(idx):
        P[k, col] = 1
    A_m = P.T * A * P
    # Numeric, SPD spot values: symbolic inverses add nothing to the argument.
    values = {
        f1: 1,
        f2: sp.Rational(3, 2),
        g: sp.Rational(1, 5),
        e: -1,
        J11: sp.Rational(1, 3),
        J12: sp.Rational(1, 7),
        J21: sp.Rational(-1, 11),
        J22: sp.Rational(1, 13),
    }
    A_num = A_m.subs(values)
    J_num = (P.T * J).subs(values)
    t_star = -(A_num.inv() * J_num)
    C_num = C

    def e_lin(t_m):
        return ((C_num * (P * t_m)).T * J.subs(values))[0]

    def e_h(t_m):
        return (
            (C_num * (P * t_m)).T * (2 * J.subs(values) + A.subs(values) * (P * t_m))
        )[0]

    return A_num, t_star, e_lin, e_h, len(idx)


section("Claim 5: swap-closed masks are stationary; a non-closed mask is not")
d1, d2, d3 = sp.symbols("d1 d2 d3", real=True)
# Closed: keep 11, 12, 21 (Eq-8's symmetric J-or-K test gives this shape).
A_c, t_c, lin_c, h_c, n_c = energies([True, True, True, False])
dvec = sp.Matrix([d1, d2, d3])
h_expanded = sp.expand(h_c(t_c + s * dvec))
assert h_expanded.coeff(s, 1) == 0, "closed mask must have no first-order term"
assert sp.simplify(h_c(t_c) - lin_c(t_c)) == 0
assert sp.expand(lin_c(t_c + s * dvec)).coeff(s, 1) != 0
print("closed mask (11,12,21): E_H has no s^1 term; E_H(t*) == E_lin(t*) =", h_c(t_c))
# Non-closed: keep 11 and 12 but drop their partner 21.
A_o, t_o, lin_o, h_o, n_o = energies([True, True, False, False])
dvec2 = sp.Matrix([d1, d2])
open_coeff = sp.expand(h_o(t_o + s * dvec2)).coeff(s, 1)
assert open_coeff != 0, "non-closed mask should NOT be stationary"
print(
    "non-closed mask (11,12): s^1 coeff =", open_coeff, " (first-order error survives)"
)
# This model contracts over the full space while the code's swap_contract sums over
# pattern positions only; on this mask both are non-stationary.
print("So check_swap_closed is required for the E_H error claims, not decoration.")

section(
    "Claim 6: the bound of claim 3 holds numerically on five directions of the model"
)
values = {f1: 1, f2: sp.Rational(3, 2), g: sp.Rational(1, 5), e: -1}
A_v = A.subs(values)
lams = [sp.N(ev) for ev in A_v.eigenvals(multiple=True)]
lmin, lmax = min(lams), max(lams)
assert lmin > 0, "model must be SPD for the bound to apply"
J_v = sp.Matrix(
    [sp.Rational(1, 3), sp.Rational(1, 7), sp.Rational(-1, 11), sp.Rational(1, 13)]
)
t_s = -(A_v.inv() * J_v)
for dv in ([1, 0, 0, 0], [1, 1, 1, 1], [1, -1, -1, 1], [0, 1, -1, 0], [3, -2, 5, 7]):
    d = sp.Matrix(dv) / sp.Integer(1000)
    t = t_s + d
    R = J_v + A_v * t
    gap = ((C * t).T * (2 * J_v + A_v * t))[0] - ((C * t_s).T * J_v)[0]
    rhs_upper = 3 * R.dot(R) / lmin
    rhs_lower = R.dot(R) / lmax
    assert rhs_lower <= sp.N(gap) * (1 + sp.Float(1e-12))
    assert sp.N(gap) <= rhs_upper * (1 + sp.Float(1e-12))
    print(
        f"d={dv}: E_H-E*={sp.N(gap, 6)}  lower={sp.N(rhs_lower, 6)}  upper={sp.N(rhs_upper, 6)}"
    )

section("Claim 7 (ARITHMETIC, not proof): what each stopping rule implies")
# Inputs, none proven here:
#  - WSHG23's ||r|| < 2.5 sqrt(eps): verified in sec. 3.1 of the paper (see
#    wiki/hylleraas-lmp2-lit-search-2026-10-30.md). It is absolute and uses the
#    unpreconditioned residual; the paper leaves the norm type and units unstated.
#  - 3||PJ||^2/lmin = 3.97 Ha for C8 Coulomb, and the measured gap/bound ratio
#    of about 0.18: from the #605 table (6-31G, eps 1e-4).
#  - lmin = 1.0 Ha is a PLACEHOLDER used only to get a magnitude.
lmin_placeholder = 1.0
c8_prefactor = 3.97
print("rule                       worst-case bound (Ha)")
print(
    f"ferric pre-D  rtol 1e-11   E_lin: FIRST order, no such bound; E_H-form {c8_prefactor * 1e-22:.1e}"
)
print(
    f"ferric D      rtol 1e-6    {c8_prefactor * 1e-12:.1e}  (gate: must stay <= 1e-10)"
)
for eps in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
    bound = 3 * (2.5**2) * eps / lmin_placeholder
    print(
        f"WSHG23 eps={eps:.0e}          {bound:.1e}  (= 18.75 eps / lmin; absolute ||r||)"
    )
print(
    "WLLH25 (preprint) later uses 5 sqrt(eps); the constant only scales these rows by 4."
)
print("Reading (interpretation): WSHG23 ties solver error to the truncation level eps,")
print(
    "ferric D fixes it absolutely, far tighter. Same functional, different stopping rule."
)

print("\nALL ASSERTIONS PASSED")
