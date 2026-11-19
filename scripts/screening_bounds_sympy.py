#!/usr/bin/env python3
"""Schwarz, QQR and CSB screening bounds, and why Schwarz cannot see attenuation.

Derivation: `book/src/04-operators/02-terf-table-engine.md#terf-table-engine-kernels-fourier-space`, `book/src/05-screening/01-schwarz.md#schwarz-bound-admitted-kernels`, `book/src/05-screening/01-schwarz.md#schwarz-combined-schwarz-bound`, `book/src/05-screening/02-qqr.md#qqr`. Code: crates/ferric-integrals/src/schwarz.rs (schwarz,
check_operator_supported, bound_inflation), crates/ferric-integrals/src/csb.rs,
crates/ferric-scf/src/qqr.rs (QqrBounds estimate),
crates/ferric-integrals/src/qqr3.rs (QqrBounds3::estimate3).

All integrals here are over s-type Gaussian charge distributions, for which
every two-electron integral has a closed form: two normalized Gaussian
densities of exponents P1, P2 and charges q1, q2 at distance R interact through
kernel K as
  Coulomb  q1 q2 erf(theta R)/R,            theta^2 = P1 P2/(P1 + P2)
  erf(w)   q1 q2 erf(phi R)/R,              phi^2   = 1/(1/P1 + 1/P2 + 1/w^2)
  terf     q1 q2 [erf(phi(R+r0)) + erf(phi(R-r0))]/(2R)
(the last is claim 4 of scripts/terf_identities_sympy.py), with the R -> 0
limits 2 theta/sqrt(pi) and so on.

Claims (each prints "ok [label]" or "FAIL [label]"):
  1 [fourier]     The 3-D Fourier transforms of the short-range kernels,
                  K^(k) = (4 pi/k) int_0^inf r K(r) sin(k r) dr, are
                  erfc:  (4 pi/k^2)(1 - exp(-k^2/4w^2))
                  terfc: (4 pi/k^2)(1 - cos(k r0) exp(-k^2/4w^2)),
                  so terf^ = cos(k r0) erf^ (quadrature at several k).
  2 [sign]        terf^ is negative for k r0 in (pi/2, 3 pi/2); its first zero
                  is k r0 = pi/2. terfc^ >= (4 pi/k^2)(1 - exp(-k^2/4w^2)) > 0.
  3 [indefinite]  On s Gaussians on one centre the terf Gram matrix has a
                  negative eigenvalue while the erf, erfc and terfc Gram
                  matrices are positive definite (40-digit eigenvalues). From
                  the negative direction two densities a, b with (a|a), (b|b) > 0
                  and |(a|b)| > sqrt((a|a)(b|b)) are built: Schwarz fails for
                  terf. The same direction has (rho|terfc|rho) > (rho|1/r|rho):
                  the Coulomb diagonal does not bound the terfc diagonal.
  4 [schwarz]     For erf, erfc and terfc, |(ab|cd)| <= Q_ab Q_cd on random s
                  shell pairs on random centres (no violation).
  5 [csb]         terf(r) >= 0 for r >= 0, and the CSB bound
                  |(mn|ls)| <= sqrt((mm|ll)(nn|ss)) holds for terf as well as
                  for terfc on the same random quartets.
  6 [blind]       The Schwarz factor of a pair is a self-interaction. For an
                  s pair density of exponent P, Q_erfc^2/Q_coul^2 = 1 - phi/theta
                  with theta^2 = P/2, phi^2 = 1/(2/P + 1/w^2); it tends to 1 as
                  P/w^2 -> inf (compact pairs: no attenuation in the bound), and
                  it does not depend on the distance to any other pair, while
                  the true erfc integral between two pairs falls below 1e-12 of
                  the bound at R = 8/w.
  7 [qqr-s]       For primitive s pairs, |(ab|cd)|/(Q_ab Q_cd)
                  <= sqrt(pi/8)(ext_ab + ext_cd)/R, ext = 1/sqrt(P), so the
                  QQR envelope min(1, 1.10 ext_sum/R_eff), R_eff = R - ext_sum,
                  is an upper bound for every positive charge (sympy for the
                  inequality chain, then a sweep over exponents and R).
  8 [qqr-erfc]    The same envelope over erfc Schwarz factors holds on the
                  same sweep for w in {0.2, 1, 5} (measured worst ratio
                  reported; a bound, not a proof).
  9 [fourier-self] The Fourier form of the self-interaction,
                  (rho|w|rho) = (2 pi)^-3 int |rho~(k)|^2 K^(k) dk, gives for a
                  unit s Gaussian of exponent a under Coulomb
                  (2/pi) int_0^inf exp(-k^2/2a) dk = sqrt(2a/pi) = 2 theta/sqrt(pi),
                  theta^2 = a/2 (sympy).
 10 [diffuse]     With x = P/w^2, phi/theta = sqrt(2/(2 + x)) and the Schwarz
                  ratio 1 - phi/theta = x/4 + O(x^2) for a diffuse pair (sympy).

Mutations (each turns the script red; the runner applies them as exact
one-occurrence substitutions):
  - terfc transform cos(k r0) sign flipped                    ("FAIL [fourier]")
  - terf same-centre value exp(-s) -> 1 (the erf kernel)      ("FAIL [indefinite]")
  - QQR safety factor 1.10 -> 0.50                            ("FAIL [qqr-s]")
  - Schwarz factor of a pair left squared (no square root)     ("FAIL [schwarz]")
  - diffuse-limit coefficient x/4 -> x/2                       ("FAIL [diffuse]")
Exits nonzero when any claim fails.
"""

import random
import sys

import mpmath as mp
import sympy as sp

mp.mp.dps = 40
failures = []


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} [{label}] {detail}")
    if not ok:
        failures.append(label)


# ---------------------------------------------------------------- kernels
def terf_r(rr, r0, w):
    return (mp.erf(w * (rr - r0)) + mp.erf(w * (rr + r0))) / 2


def ft_quad(r_times_kernel, kk):
    """(4 pi / k) int_0^inf r K(r) sin(k r) dr for a decaying r K(r)."""
    return (
        4
        * mp.pi
        / kk
        * mp.quadosc(
            lambda rr: r_times_kernel(rr) * mp.sin(kk * rr), [0, mp.inf], omega=kk
        )
    )


def ft_erfc(kk, w):
    return 4 * mp.pi / kk**2 * (1 - mp.exp(-(kk**2) / (4 * w**2)))


def ft_terfc(kk, r0v, w):
    return 4 * mp.pi / kk**2 * (1 - mp.cos(kk * r0v) * mp.exp(-(kk**2) / (4 * w**2)))


def ft_terf(kk, r0v, w):
    return 4 * mp.pi / kk**2 - ft_terfc(kk, r0v, w)


# 1. Fourier transforms by quadrature ---------------------------------------
mp.mp.dps = 20
w0, r00 = mp.mpf("0.9"), mp.mpf("1.3")
worst = 0
for kk in [mp.mpf(x) for x in ("0.4", "1.1", "2.0", "3.7")]:
    a = ft_quad(lambda rr: mp.erfc(w0 * rr), kk)
    b = ft_quad(lambda rr: 1 - terf_r(rr, r00, w0), kk)
    worst = max(
        worst,
        abs(a - ft_erfc(kk, w0)) / ft_erfc(kk, w0),
        abs(b - ft_terfc(kk, r00, w0)) / ft_terfc(kk, r00, w0),
    )
check("fourier", worst < 1e-12, f"worst rel {mp.nstr(worst, 3)}")
mp.mp.dps = 40

# 2. sign structure -----------------------------------------------------------
kk, r0s, ws = sp.symbols("k r0 w", positive=True)
ft_terf_sym = 4 * sp.pi / kk**2 * sp.cos(kk * r0s) * sp.exp(-(kk**2) / (4 * ws**2))
first_zero = sp.solve(sp.Eq(sp.cos(kk * r0s), 0), kk)
lower = sp.simplify(
    (4 * sp.pi / kk**2) * (1 - sp.cos(kk * r0s) * sp.exp(-(kk**2) / (4 * ws**2)))
    - (4 * sp.pi / kk**2) * (1 - sp.exp(-(kk**2) / (4 * ws**2)))
    - (4 * sp.pi / kk**2) * (1 - sp.cos(kk * r0s)) * sp.exp(-(kk**2) / (4 * ws**2))
)
neg_mid = ft_terf_sym.subs({kk: sp.pi / r0s, ws: 1}).simplify()
check(
    "sign",
    min(first_zero, key=lambda e: e.subs(r0s, 1)) == sp.pi / (2 * r0s)
    and lower == 0
    and neg_mid.subs(r0s, 1) < 0
    and ft_terf(mp.pi / (2 * r00) * mp.mpf("0.99"), r00, w0) > 0
    and ft_terf(mp.pi / (2 * r00) * mp.mpf("1.01"), r00, w0) < 0,
    f"first zero k = {min(first_zero, key=lambda e: e.subs(r0s, 1))}",
)


# ------------------------------------------- closed forms over s densities
def coul(q1, q2, p1, p2, R):
    th = mp.sqrt(p1 * p2 / (p1 + p2))
    return q1 * q2 * (2 * th / mp.sqrt(mp.pi) if R == 0 else mp.erf(th * R) / R)


def erf_k(q1, q2, p1, p2, R, w):
    ph = mp.sqrt(1 / (1 / p1 + 1 / p2 + 1 / w**2))
    return q1 * q2 * (2 * ph / mp.sqrt(mp.pi) if R == 0 else mp.erf(ph * R) / R)


def terf_k(q1, q2, p1, p2, R, r0v, w):
    ph = mp.sqrt(1 / (1 / p1 + 1 / p2 + 1 / w**2))
    if R == 0:
        return q1 * q2 * 2 * ph / mp.sqrt(mp.pi) * mp.exp(-(ph**2) * r0v**2)
    return q1 * q2 * (mp.erf(ph * (R + r0v)) + mp.erf(ph * (R - r0v))) / (2 * R)


def kern(name, q1, q2, p1, p2, R, w, r0v):
    if name == "coulomb":
        return coul(q1, q2, p1, p2, R)
    if name == "erf":
        return erf_k(q1, q2, p1, p2, R, w)
    if name == "erfc":
        return coul(q1, q2, p1, p2, R) - erf_k(q1, q2, p1, p2, R, w)
    if name == "terf":
        return terf_k(q1, q2, p1, p2, R, r0v, w)
    if name == "terfc":
        return coul(q1, q2, p1, p2, R) - terf_k(q1, q2, p1, p2, R, r0v, w)
    raise ValueError(name)


# 3. indefiniteness of terf on one centre ---------------------------------
r0v = mp.mpf("2.0")
wv = 1 / (r0v * mp.sqrt(2))
expo = [mp.mpf("0.02") * mp.mpf("2.2") ** i for i in range(12)]


def gram(name):
    n = len(expo)
    m = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            if name == "terf":
                ph = mp.sqrt(1 / (1 / expo[i] + 1 / expo[j] + 1 / wv**2))
                m[i, j] = 2 * ph / mp.sqrt(mp.pi) * mp.exp(-(ph**2) * r0v**2)
            else:
                m[i, j] = kern(name, 1, 1, expo[i], expo[j], 0, wv, r0v)
    return m


ev = {
    name: mp.eigsy(gram(name))[0]
    for name in ("erf", "erfc", "terf", "terfc", "coulomb")
}
mins = {name: min(ev[name]) for name in ev}
tev, tvec = mp.eigsy(gram("terf"))
i_neg = min(range(len(expo)), key=lambda i: tev[i])
i_pos = max(range(len(expo)), key=lambda i: tev[i])
lam_n, lam_p = tev[i_neg], tev[i_pos]
vt = gram("terf")
vneg = tvec[:, i_neg]
vpos = tvec[:, i_pos]
t = mp.sqrt(lam_p / (-lam_n)) / 2 if lam_n < 0 else mp.mpf(1)
a_vec, b_vec = vpos + t * vneg, vpos - t * vneg
aa = (a_vec.T * vt * a_vec)[0]
bb = (b_vec.T * vt * b_vec)[0]
ab = (a_vec.T * vt * b_vec)[0]
dc = (vneg.T * gram("coulomb") * vneg)[0]
dtc = (vneg.T * gram("terfc") * vneg)[0]
check(
    "indefinite",
    mins["terf"] < 0
    and all(mins[n] > 0 for n in ("erf", "erfc", "terfc"))
    and aa > 0
    and bb > 0
    and abs(ab) > mp.sqrt(aa * bb)
    and dtc > dc,
    f"lambda_min: terf {mp.nstr(mins['terf'], 3)}, terfc {mp.nstr(mins['terfc'], 3)}, "
    f"erfc {mp.nstr(mins['erfc'], 3)}; |(a|b)|/sqrt((a|a)(b|b)) = {mp.nstr(abs(ab) / mp.sqrt(aa * bb), 4)}",
)


# 4/5. Schwarz and CSB on random s quartets -------------------------------
def gauss_product(a, A, b, B):
    p = a + b
    P = [(a * A[i] + b * B[i]) / p for i in range(3)]
    K = mp.exp(-a * b / p * sum((A[i] - B[i]) ** 2 for i in range(3)))
    return p, P, K * (mp.pi / p) ** mp.mpf(1.5)  # exponent, centre, charge


def dist(X, Y):
    return mp.sqrt(sum((X[i] - Y[i]) ** 2 for i in range(3)))


def eri_s(name, f1, f2, f3, f4, w, r0v):
    p1, P1, q1 = gauss_product(*f1, *f2)
    p2, P2, q2 = gauss_product(*f3, *f4)
    return kern(name, q1, q2, p1, p2, dist(P1, P2), w, r0v)


def schwarz_q(name, fa, fb, w, r0v):
    return mp.sqrt(eri_s(name, fa, fb, fa, fb, w, r0v))


rng = random.Random(7)


def rand_fn():
    a = mp.mpf(10 ** rng.uniform(-1.3, 1.2))
    return (a, [mp.mpf(rng.uniform(-3, 3)) for _ in range(3)])


worst_s = {n: 0 for n in ("erf", "erfc", "terfc")}
worst_csb = {n: 0 for n in ("terf", "terfc")}
for _ in range(300):
    f = [rand_fn() for _ in range(4)]
    w = mp.mpf(10 ** rng.uniform(-0.7, 0.7))
    r0q = mp.mpf(rng.uniform(0.6, 4.0))
    for name in worst_s:
        true = abs(eri_s(name, *f, w, r0q))
        bound = schwarz_q(name, f[0], f[1], w, r0q) * schwarz_q(
            name, f[2], f[3], w, r0q
        )
        worst_s[name] = max(worst_s[name], true / bound)
    for name in worst_csb:
        true = abs(eri_s(name, *f, w, r0q))
        m1 = eri_s(name, f[0], f[0], f[2], f[2], w, r0q)
        m2 = eri_s(name, f[1], f[1], f[3], f[3], w, r0q)
        worst_csb[name] = max(worst_csb[name], true / mp.sqrt(m1 * m2))
check(
    "schwarz",
    all(v <= 1 for v in worst_s.values()),
    " ".join(f"{k} {mp.nstr(v, 4)}" for k, v in worst_s.items()),
)
terf_nonneg = all(
    terf_r(mp.mpf(x) / 10, mp.mpf(r0) / 2, mp.mpf(wq) / 4) >= 0
    for x in range(0, 120, 3)
    for r0 in (1, 3, 8)
    for wq in (1, 3, 9)
)
check(
    "csb",
    terf_nonneg and all(v <= 1 for v in worst_csb.values()),
    " ".join(f"{k} {mp.nstr(v, 4)}" for k, v in worst_csb.items()),
)

# 6. Schwarz is blind to attenuation ----------------------------------------
P, w = sp.symbols("P w", positive=True)
theta = sp.sqrt(P / 2)
phi = 1 / sp.sqrt(2 / P + 1 / w**2)
ratio = 1 - phi / theta
lim_compact = sp.limit(ratio, P, sp.oo)
# numeric: the ratio of the self-interactions by the closed forms
Pn, wn = mp.mpf("3.0"), mp.mpf("0.7")
q_c = coul(1, 1, Pn, Pn, 0)
q_e = coul(1, 1, Pn, Pn, 0) - erf_k(1, 1, Pn, Pn, 0, wn)
rat_num = q_e / q_c
rat_sym = ratio.subs({P: 3, w: sp.Rational(7, 10)}).evalf(30)
# two compact pairs (P = 20) at R = 8/w: true erfc integral vs Schwarz product
R8 = 8 / wn
true_far = coul(1, 1, 20, 20, R8) - erf_k(1, 1, 20, 20, R8, wn)
bound = mp.sqrt(coul(1, 1, 20, 20, 0) - erf_k(1, 1, 20, 20, 0, wn)) ** 2
check(
    "blind",
    lim_compact == 1 and abs(rat_num - rat_sym) < 1e-25 and true_far / bound < 1e-12,
    f"Q_erfc^2/Q_C^2 at P/w^2 -> inf: {lim_compact}; erfc true/bound at R = 8/w: {mp.nstr(true_far / bound, 3)}",
)

# 7. QQR for primitive s pairs ---------------------------------------------
QQR_SAFETY = mp.mpf("1.10")
P1, P2, R, q1, q2 = sp.symbols("P1 P2 R q1 q2", positive=True)
qq = q1 * q2 * sp.sqrt(2 / sp.pi) * (P1 * P2) ** sp.Rational(1, 4)  # Q_1 Q_2
# erf(theta R)/R <= 1/R, so true/QQ <= sqrt(pi/2) (P1 P2)^(-1/4) / R; AM-GM on
# (P1 P2)^(-1/4) = sqrt(e1 e2) <= (e1 + e2)/2 with e = 1/sqrt(P).
step = sp.simplify(
    q1 * q2 / R / qq - sp.sqrt(sp.pi / 2) * (P1 * P2) ** sp.Rational(-1, 4) / R
)
e1, e2 = sp.symbols("e1 e2", positive=True)
amgm = sp.simplify((e1 + e2) ** 2 / 4 - e1 * e2 - (e1 - e2) ** 2 / 4)
check_sym = step == 0 and amgm == 0 and float(sp.sqrt(sp.pi / 8)) < float(QQR_SAFETY)


def qqr_envelope(ext_sum, Rv):
    r_eff = max(Rv - ext_sum, 0)
    return 1 if r_eff == 0 else min(1, QQR_SAFETY * ext_sum / r_eff)


def sweep(name, w):
    worst = 0
    for pa in (0.05, 0.3, 1.0, 4.0, 30.0):
        for pb in (0.05, 0.5, 2.0, 15.0):
            pa_, pb_ = mp.mpf(pa), mp.mpf(pb)
            ext = 1 / mp.sqrt(pa_) + 1 / mp.sqrt(pb_)
            qa = mp.sqrt(kern(name, 1, 1, pa_, pa_, 0, w, 0))
            qb = mp.sqrt(kern(name, 1, 1, pb_, pb_, 0, w, 0))
            for Rv in [mp.mpf(x) / 4 for x in range(1, 200)]:
                true = abs(kern(name, 1, 1, pa_, pb_, Rv, w, 0))
                worst = max(worst, true / (qa * qb * qqr_envelope(ext, Rv)))
    return worst


worst_c = sweep("coulomb", 1)
check("qqr-s", check_sym and worst_c <= 1, f"worst |true|/bound {mp.nstr(worst_c, 4)}")
worst_e = {w: sweep("erfc", mp.mpf(w)) for w in ("0.2", "1.0", "5.0")}
check(
    "qqr-erfc",
    all(v <= 1 for v in worst_e.values()),
    " ".join(f"w={k}: {mp.nstr(v, 4)}" for k, v in worst_e.items()),
)

# 9. Fourier form of the self-interaction, unit s Gaussian under Coulomb -----
a_, k_ = sp.symbols("a k", positive=True)
rho_k = sp.exp(-(k_**2) / (4 * a_))  # transform of (a/pi)^(3/2) exp(-a r^2)
self_int = (
    sp.integrate(4 * sp.pi * k_**2 * rho_k**2 * 4 * sp.pi / k_**2, (k_, 0, sp.oo))
    / (2 * sp.pi) ** 3
)
check(
    "fourier-self",
    sp.simplify(self_int - sp.sqrt(2 * a_ / sp.pi)) == 0
    and sp.simplify(self_int - 2 * sp.sqrt(a_ / 2) / sp.sqrt(sp.pi)) == 0,
    f"(rho|1/r|rho) = {sp.simplify(self_int)}",
)

# 10. diffuse-pair limit of the Schwarz ratio ---------------------------------
x_ = sp.symbols("x", positive=True)
ratio_x = 1 - (1 / sp.sqrt(2 / P + 1 / w**2)) / sp.sqrt(P / 2)
ratio_x = sp.simplify(ratio_x.subs(P, x_ * w**2))
diffuse_coeff = sp.Rational(1, 4)
check(
    "diffuse",
    sp.simplify(ratio_x - (1 - sp.sqrt(2 / (2 + x_)))) == 0
    and sp.series(ratio_x, x_, 0, 2).removeO() - diffuse_coeff * x_ == 0,
    f"1 - phi/theta = {sp.series(ratio_x, x_, 0, 2)}",
)

if failures:
    print(f"{len(failures)} claim(s) failed: {', '.join(failures)}")
    sys.exit(1)
print("all screening-bound claims hold")
