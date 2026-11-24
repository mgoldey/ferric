#!/usr/bin/env python3
"""The terf / terfc kernel and the auxiliary function the table engine evaluates.

Derivation: `book/src/04-operators/01-erf-terf.md#erf-terf-pairs` and
"Two-center integrals over the terf kernel". Code:
crates/ferric-integrals/src/operator.rs (Operator::terf, Operator::terfc,
Operator::terfc_with_omega) and the table engine in
crates/ferric-integrals/shim/shim.cc (terf_G_series, terf_G_tail, terf_delta,
terf_far_field, terf_aux).

Definitions (atomic units):
  terf(r; r0, w)  = 1/2 [erf(w (r - r0)) + erf(w (r + r0))],  terfc = 1 - terf
  theta^2 = p q / (p + q),  phi^2 = 1 / (1/p + 1/q + 1/w^2)
  S = phi^2 R^2,  s = phi^2 r0^2
  G_{m}(S, s) = sum_i (2i)!!/(2i+1)!!  [nabla^m pmf_S](i)  cdf_s(i)
     (the defining Poisson series of terf_G_series; nabla f(i) = f(i) - f(i-1),
      pmf_S(i) = e^-S S^i / i!, cdf_s(i) = sum_{j<=i} e^-s s^j / j!)

Claims (each prints "ok [label]" or "FAIL [label]"):
  1 [complement]   terf + terfc = 1; terf is odd in r; terf -> erf(w r) as
                   r0 -> 0 and terf -> 0 as r0 -> inf at fixed w (sympy).
  2 [curvature]    terf(r)/r = c0 + c2 r^2 + O(r^4) with
                   c2 proportional to (2 w^2 r0^2 - 1): the long-range kernel is
                   flat at r = 0 exactly when r0 w = 1/sqrt(2), the default of
                   Operator::terf / Operator::terfc (sympy series).
  3 [s-bound]      phi <= w, so s <= (w r0)^2, and s <= 1/2 under the
                   curvature constraint (sympy).
  4 [two-center]   For s Gaussians exp(-p|r-A|^2), exp(-q|r-B|^2) at distance R,
                   (a|terf|b) = 2 pi^(5/2) / (p q sqrt(p+q)) (phi/theta) G0c(S,s)
                   with the closed form
                   G0c(S,s) = sqrt(pi)/(4 sqrt S) [erf(sqrt S + sqrt s) + erf(sqrt S - sqrt s)],
                   checked against a radial quadrature of the defining
                   six-dimensional integral (mpmath, 30 digits).
  5 [series]       The Poisson series G_m of the code equals (-d/dS)^m G0c at
                   fixed s, m = 0..6 (closed form differentiated by sympy).
  6 [boys-shape]   dI/dX of the two-center integral equals
                   -2 phi^2 X C (phi/theta) G_1(S,s): G_m enters the
                   McMurchie-Davidson recursion with reduced exponent phi^2
                   exactly where F_m(T) enters for Coulomb (central
                   differences of the quadrature of claim 4).
  7 [s-zero]       G_m(S, 0) = F_m(S), the Boys function.
  8 [tail-form]    G_m = F_m - Delta_m, Delta_m = sum_i df(2i) [nabla^m pmf](i) tail_s(i)
                   (terf_delta).
  9 [far-field]    G_m = F_m to 1e-12 relative when S > 50 and
                   sqrt S - sqrt s > 6 (terf_far_field); a point with S > 50 but
                   sqrt S - sqrt s < 6 differs at O(1e-2) or more, so the second
                   test is needed.
 10 [terfc-split]  (a|terfc|b) = (a|1/r|b) - (a|terf|b) from independent
                   quadratures of the three kernels.
 11 [taylor-steps] With g(r) = erf(w (r + r0)): g'(0) = (2w/sqrt pi) e^{-w^2 r0^2},
                   g'''(0) = (2w/sqrt pi)(4 w^4 r0^2 - 2 w^2) e^{-w^2 r0^2},
                   tau3 = g'''(0)/6 is the r^2 coefficient of terf/r, and
                   tau3 -> -2 w^3/(3 sqrt pi), the erf coefficient, at r0 = 0
                   (sympy).
 12 [two-center-steps] The steps of the two-center derivation: the prefactor
                   identity p q sqrt(p+q) theta = (p q)^{3/2}; the k integral
                   (2/pi) int_0^inf e^{-k^2/4 phi^2} sin(k x)/k dk = erf(phi x)
                   (mpmath quadrature); G0(S -> 0, s) = e^{-s} and
                   G0(S, 0) = F_0(S) (sympy).

Mutations (each turns the script red; the runner applies them as exact
one-occurrence substitutions):
  - closed form: erf(sqrt S - sqrt s) -> erf(sqrt s - sqrt S)  ("FAIL [two-center]")
  - series double-factorial ratio (2i)/(2i+1) -> (2i+1)/(2i+2) ("FAIL [series]")
  - drop the phi/theta prefactor of the two-center formula     ("FAIL [two-center]")
  - curvature constraint r0 w = 1/sqrt 2 -> 1/2                 ("FAIL [curvature]")
  - g'''(0) bracket 4 w^4 r0^2 - 2 w^2 -> 4 w^4 r0^2 - w^2      ("FAIL [taylor-steps]")
Exits nonzero when any claim fails.
"""

import sys

import mpmath as mp
import sympy as sp

mp.mp.dps = 30
failures = []


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} [{label}] {detail}")
    if not ok:
        failures.append(label)


r, r0, w, S, s, k = sp.symbols("r r0 w S s k", positive=True)
terf = (sp.erf(w * (r - r0)) + sp.erf(w * (r + r0))) / 2
terfc = 1 - terf

# 1. complement, oddness, limits ------------------------------------------------
x = sp.symbols("x", real=True)
terf_x = (sp.erf(w * (x - r0)) + sp.erf(w * (x + r0))) / 2
odd = sp.simplify(terf_x.subs(x, -x) + terf_x) == 0
lim0 = sp.simplify(sp.limit(terf, r0, 0) - sp.erf(w * r)) == 0
# sympy 1.14 returns 1 for this limit (the two erf terms tend to -1 and +1);
# take it term by term instead.
liminf = (
    sp.limit(sp.erf(w * (r - r0)), r0, sp.oo)
    + sp.limit(sp.erf(w * (r + r0)), r0, sp.oo)
    == 0
)
check("complement", sp.simplify(terf + terfc - 1) == 0 and odd and lim0 and liminf)

# 2. curvature at r = 0 ----------------------------------------------------------
CURVATURE_R0W = 1 / sp.sqrt(2)
ser = sp.series(terf / r, r, 0, 4).removeO()
c2 = sp.simplify(ser.coeff(r, 2))
c2_at = sp.simplify(c2.subs(w, CURVATURE_R0W / r0))
# c2 factors as (positive) * (2 w^2 r0^2 - 1): its only positive root in w is 1/(sqrt2 r0).
w_roots = sp.solve(sp.Eq(c2, 0), w)
check(
    "curvature",
    c2_at == 0 and w_roots == [1 / (sp.sqrt(2) * r0)],
    f"c2 = {sp.factor(c2)}",
)

# 3. s <= (w r0)^2 ---------------------------------------------------------------
p, q = sp.symbols("p q", positive=True)
phi2 = 1 / (1 / p + 1 / q + 1 / w**2)
s_con = sp.simplify(phi2.subs(w, CURVATURE_R0W / r0) * r0**2)
check(
    "s-bound",
    sp.simplify(w**2 - phi2 - w**2 * (1 / p + 1 / q) * phi2) == 0
    and sp.simplify(
        sp.Rational(1, 2) - s_con - (p + q) / (2 * (2 * p * q * r0**2 + p + q))
    )
    == 0,
    f"constrained s = {s_con}",
)

# 4. two-center closed form against quadrature ----------------------------------
G0c = (
    sp.sqrt(sp.pi)
    / (4 * sp.sqrt(S))
    * (sp.erf(sp.sqrt(S) + sp.sqrt(s)) + sp.erf(sp.sqrt(S) - sp.sqrt(s)))
)
G0c_num = sp.lambdify((S, s), G0c, "mpmath")


def kernel(name, rr, r0v, wv):
    if name == "coulomb":
        return 1 / rr
    t = (mp.erf(wv * (rr - r0v)) + mp.erf(wv * (rr + r0v))) / 2
    return t / rr if name == "terf" else (1 - t) / rr


def two_center_quad(name, pv, qv, R, r0v, wv):
    """(a|K|b) for unnormalized s Gaussians, by the radial integral over the
    relative coordinate: the pair convolution is a Gaussian of exponent theta^2
    centred at distance R, whose spherical average is
    (theta^2/pi)^(3/2) * pi/(theta^2 x R) [e^{-theta^2 (x-R)^2} - e^{-theta^2 (x+R)^2}] / 2 * 2."""
    th2 = pv * qv / (pv + qv)
    pref = (mp.pi**2 / (pv * qv)) ** mp.mpf(1.5) * (th2 / mp.pi) ** mp.mpf(1.5)

    def f(xx):
        ang = (
            mp.pi
            / (th2 * xx * R)
            * (mp.exp(-th2 * (xx - R) ** 2) - mp.exp(-th2 * (xx + R) ** 2))
        )
        return xx**2 * kernel(name, xx, r0v, wv) * ang

    pts = [0, max(R - 8 / mp.sqrt(th2), 0), R, R + 8 / mp.sqrt(th2), mp.inf]
    pts = sorted(set(pts))
    return pref * mp.quad(f, pts)


def two_center_closed(pv, qv, R, r0v, wv):
    th2 = pv * qv / (pv + qv)
    ph2 = 1 / (1 / pv + 1 / qv + 1 / wv**2)
    pref = 2 * mp.pi**2.5 / (pv * qv * mp.sqrt(pv + qv))
    return pref * mp.sqrt(ph2 / th2) * G0c_num(ph2 * R**2, ph2 * r0v**2)


cases = [
    (
        mp.mpf("0.8"),
        mp.mpf("1.3"),
        mp.mpf("1.7"),
        mp.mpf("1.4"),
        1 / (mp.mpf("1.4") * mp.sqrt(2)),
    ),
    (
        mp.mpf("5.0"),
        mp.mpf("0.4"),
        mp.mpf("0.6"),
        mp.mpf("2.0"),
        1 / (mp.mpf("2.0") * mp.sqrt(2)),
    ),
    (
        mp.mpf("2.0"),
        mp.mpf("3.0"),
        mp.mpf("3.5"),
        mp.mpf("1.0"),
        mp.mpf("2.5"),
    ),  # free omega
]
worst = 0
for pv, qv, R, r0v, wv in cases:
    a = two_center_quad("terf", pv, qv, R, r0v, wv)
    b = two_center_closed(pv, qv, R, r0v, wv)
    worst = max(worst, abs(a - b) / abs(a))
check("two-center", worst < 1e-20, f"worst rel {mp.nstr(worst, 3)}")


# 5. Poisson series (the code's definition) = (-d/dS)^m G0c ---------------------
def g_series(Sv, sv, mmax):
    """terf_G_series, in mpmath: rows of nabla^m pmf_S times cdf_s."""
    n = int(Sv + 12 * mp.sqrt(Sv) + 60)
    pmf = [mp.e ** (-Sv)]
    for i in range(1, n):
        pmf.append(pmf[-1] * Sv / i)
    cdf, t, acc = [], mp.e ** (-sv), mp.mpf(0)
    for i in range(n):
        if i > 0:
            t *= sv / i
        acc += t
        cdf.append(acc)
    df = [mp.mpf(1)]
    for i in range(1, n):
        df.append(df[-1] * (2 * i) / (2 * i + 1))
    row, out = pmf[:], []
    for m in range(mmax + 1):
        if m > 0:
            row = [row[0]] + [row[i] - row[i - 1] for i in range(1, n)]
        out.append(sum(df[i] * row[i] * cdf[i] for i in range(n)))
    return out, pmf, cdf, df


def g_tail(Sv, sv, mmax):
    """F_m - Delta_m, Delta_m = sum df * nabla^m pmf * (1 - cdf)."""
    _, pmf, cdf, df = g_series(Sv, sv, mmax)
    n = len(pmf)
    row, out = pmf[:], []
    for m in range(mmax + 1):
        if m > 0:
            row = [row[0]] + [row[i] - row[i - 1] for i in range(1, n)]
        delta = sum(df[i] * row[i] * (1 - cdf[i]) for i in range(n))
        out.append(boys(m, Sv) - delta)
    return out


def boys(m, T):
    if T == 0:
        return mp.mpf(1) / (2 * m + 1)
    return mp.gammainc(m + mp.mpf(0.5), 0, T) / (2 * T ** (m + mp.mpf(0.5)))


MMAX = 6
derivs = [G0c]
for m in range(1, MMAX + 1):
    derivs.append(-sp.diff(derivs[-1], S))
deriv_num = [sp.lambdify((S, s), d, "mpmath") for d in derivs]
worst = 0
for Sv, sv in [(0.05, 0.3), (1.3, 0.5), (4.0, 2.0), (12.0, 0.25), (19.0, 15.0)]:
    Sv, sv = mp.mpf(Sv), mp.mpf(sv)
    ser_vals, *_ = g_series(Sv, sv, MMAX)
    for m in range(MMAX + 1):
        ref = deriv_num[m](Sv, sv)
        worst = max(worst, abs(ser_vals[m] - ref) / abs(ref))
check("series", worst < 1e-15, f"worst rel {mp.nstr(worst, 3)} over m <= {MMAX}")

# 6. the Boys-replacement shape: dI/dX = -2 phi^2 X C (phi/theta) G_1 -----------
pv, qv, r0v, wv = (
    mp.mpf("1.1"),
    mp.mpf("0.7"),
    mp.mpf("1.2"),
    1 / (mp.mpf("1.2") * mp.sqrt(2)),
)
X0, Y0 = mp.mpf("1.3"), mp.mpf("0.9")
h = mp.mpf("1e-6")


def i_at(X):
    return two_center_quad("terf", pv, qv, mp.sqrt(X**2 + Y0**2), r0v, wv)


fd = (i_at(X0 + h) - i_at(X0 - h)) / (2 * h)
th2 = pv * qv / (pv + qv)
ph2 = 1 / (1 / pv + 1 / qv + 1 / wv**2)
cpref = 2 * mp.pi**2.5 / (pv * qv * mp.sqrt(pv + qv))
g1 = g_series(ph2 * (X0**2 + Y0**2), ph2 * r0v**2, 1)[0][1]
analytic = -2 * ph2 * X0 * cpref * mp.sqrt(ph2 / th2) * g1
check(
    "boys-shape",
    abs(fd - analytic) / abs(analytic) < 1e-9,
    f"rel {mp.nstr(abs(fd - analytic) / abs(analytic), 3)}",
)

# 7. s = 0 gives the Boys function ----------------------------------------------
worst = 0
for Sv in [0.01, 0.7, 3.0, 15.0]:
    vals, *_ = g_series(mp.mpf(Sv), mp.mpf(0), MMAX)
    for m in range(MMAX + 1):
        worst = max(worst, abs(vals[m] - boys(m, mp.mpf(Sv))) / boys(m, mp.mpf(Sv)))
check("s-zero", worst < 1e-20, f"worst rel {mp.nstr(worst, 3)}")

# 8. tail form ----------------------------------------------------------------
worst = 0
for Sv, sv in [(0.5, 0.2), (8.0, 0.5), (30.0, 3.0)]:
    a, *_ = g_series(mp.mpf(Sv), mp.mpf(sv), MMAX)
    b = g_tail(mp.mpf(Sv), mp.mpf(sv), MMAX)
    for m in range(MMAX + 1):
        worst = max(worst, abs(a[m] - b[m]) / abs(a[m]))
check("tail-form", worst < 1e-20, f"worst rel {mp.nstr(worst, 3)}")


# 9. far-field shortcut G_m -> F_m(S) ------------------------------------------
def far_field(Sv, sv):
    return Sv > 50 and mp.sqrt(Sv) - mp.sqrt(sv) > 6


inside, outside = [], []
for Sv, sv in [
    (60.0, 0.5),
    (75.0, 4.0),
    (120.0, 20.0),
    (60.0, 40.0),
    (55.0, 45.0),
    (80.0, 60.0),
]:
    Sv, sv = mp.mpf(Sv), mp.mpf(sv)
    vals = [deriv_num[m](Sv, sv) for m in range(5)]
    rel = max(abs(vals[m] - boys(m, Sv)) / boys(m, Sv) for m in range(5))
    (inside if far_field(Sv, sv) else outside).append(rel)
check(
    "far-field",
    len(inside) >= 3
    and max(inside) < 1e-12
    and len(outside) >= 2
    and min(outside) > 1e-2,
    f"inside max {mp.nstr(max(inside), 3)}, outside min {mp.nstr(min(outside), 3)}",
)

# 10. terfc = Coulomb - terf at the integral level ------------------------------
pv, qv, R, r0v, wv = (
    mp.mpf("1.5"),
    mp.mpf("0.9"),
    mp.mpf("2.2"),
    mp.mpf("1.0"),
    1 / mp.sqrt(2),
)
ic = two_center_quad("coulomb", pv, qv, R, r0v, wv)
it = two_center_quad("terf", pv, qv, R, r0v, wv)
itc = two_center_quad("terfc", pv, qv, R, r0v, wv)
th2 = pv * qv / (pv + qv)
coul_closed = 2 * mp.pi**2.5 / (pv * qv * mp.sqrt(pv + qv)) * boys(0, th2 * R**2)
check(
    "terfc-split",
    abs(itc - (ic - it)) / abs(itc) < 1e-20 and abs(ic - coul_closed) / ic < 1e-20,
)

# 11. Taylor steps of the curvature derivation --------------------------------
g = sp.erf(w * (r + r0))
g1 = sp.diff(g, r).subs(r, 0)
g3 = sp.diff(g, r, 3).subs(r, 0)
g1_book = 2 * w / sp.sqrt(sp.pi) * sp.exp(-(w**2) * r0**2)
g3_book = (
    2 * w / sp.sqrt(sp.pi) * (4 * w**4 * r0**2 - 2 * w**2) * sp.exp(-(w**2) * r0**2)
)
tau3 = w**3 / (3 * sp.sqrt(sp.pi)) * (4 * w**2 * r0**2 - 2) * sp.exp(-(w**2) * r0**2)
check(
    "taylor-steps",
    sp.simplify(g1 - g1_book) == 0
    and sp.simplify(g3 - g3_book) == 0
    and sp.simplify(g3_book / 6 - tau3) == 0
    and sp.simplify(ser.coeff(r, 2) - tau3) == 0
    and sp.simplify(tau3.subs(r0, 0) + 2 * w**3 / (3 * sp.sqrt(sp.pi))) == 0,
)

# 12. steps of the two-center derivation ---------------------------------------
theta_ = sp.sqrt(p * q / (p + q))
pref_ok = (
    sp.simplify(p * q * sp.sqrt(p + q) * theta_ - (p * q) ** sp.Rational(3, 2)) == 0
)
worst_k = mp.mpf(0)
for phv, xv in ((mp.mpf("0.7"), mp.mpf("1.3")), (mp.mpf("2.0"), mp.mpf("0.4"))):
    kint = (
        2
        / mp.pi
        * mp.quad(
            lambda kk: mp.exp(-(kk**2) / (4 * phv**2)) * mp.sin(kk * xv) / kk,
            [0, mp.inf],
        )
    )
    worst_k = max(worst_k, abs(kint - mp.erf(phv * xv)))
tq = sp.symbols("tq", positive=True)  # tq = sqrt(S)
g0 = sp.sqrt(sp.pi) / (4 * tq) * (sp.erf(tq + sp.sqrt(s)) + sp.erf(tq - sp.sqrt(s)))
g0_s0 = sp.simplify(sp.series(g0, tq, 0, 1).removeO() - sp.exp(-s)) == 0
g0_boys = sp.simplify(g0.subs(s, 0) - sp.sqrt(sp.pi) * sp.erf(tq) / (2 * tq)) == 0
check(
    "two-center-steps",
    pref_ok and worst_k < 1e-20 and g0_s0 and g0_boys,
    f"k integral worst abs {mp.nstr(worst_k, 3)}",
)

if failures:
    print(f"{len(failures)} claim(s) failed: {', '.join(failures)}")
    sys.exit(1)
print("all terf identities hold")
