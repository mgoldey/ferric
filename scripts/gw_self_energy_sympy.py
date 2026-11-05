#!/usr/bin/env python3
"""Sympy proofs of the identities ferric-gw's self-energy code relies on.

Each claim names the Rust function it checks. Theory and notation:
crates/ferric-gw/theory.md. Every claim asserts; the script exits nonzero on
the first mismatch. Negative controls (a deliberately wrong variant that must
NOT satisfy the identity) are included where a pass could otherwise be vacuous.

  1  One-pair RPA dielectric -> single-pole W. With the closed-shell Pi
     prefactor 4 (ferric_rpa::sternheimer::build_scale_factors), the dynamic
     W~_d = eps~^-1 - 1 of one (ia) pair is -2*Omega*s/(w^2 + Omega^2), with
     Omega^2 = e(e + 4b^2) = (A-B)(A+B) of singlet direct RPA. The
     unrestricted prefactor 2 per spin (build_scale_factors_with_prefactor(.., 2))
     reproduces it when alpha = beta.
  2  Frequency integral of the RI Sigma_c (ferric_gw::sigma::sigma_c_at_z) for
     a single-pole W equals the closed-form residue result
     s/(z - eps_n + Omega) (Re(z - eps_n) > 0, occupied side) and
     s/(z - eps_n - Omega) (Re(z - eps_n) < 0, virtual side). Proven
     symbolically for real arguments, checked at complex z to 40 digits.
  3  Static-W limit = COHSEX (ferric_gw::cohsex::cohsex_pieces). A frequency-
     independent W in the same integral gives -(W/2) sgn(Re(z - eps_n)); at z in
     the gap that is  -sum_i W_mi + 1/2 sum_p W_mp = dSigma_SEX + Sigma_COH.
     The plasmon model reaches the same limit as Omega -> infinity at fixed
     W(0).
  4  QP equation (ferric_gw::sigma::solve_qp_for_mo). The linearized start
     eps + Z (Sigma(eps) + Delta) is exactly one Newton step from eps, with
     Z = 1/(1 - Sigma'(eps)); Z at a root is the residue of the QP pole of
     G = 1/(w - eps - Delta - Sigma(w)); Newton on a single-pole Sigma
     converges quadratically to the exact root.
  5  The 4-point derivative stencil of ferric_gw::pade::PadeCF::deriv_real is
     exact through degree 4, error -h^4 f^(5)/30.
  6  Thiele continued fraction (ferric_gw::pade::PadeCF::fit / eval),
     re-implemented line for line in exact arithmetic: interpolates its nodes,
     and reproduces a rational function exactly once the node count reaches
     its type. Exact over-sampling drives the next Thiele denominator to 0,
     the case PadeCF::fit refuses.
  7  RPAx static and dynamic polarizability
     (ferric_gw::bse::rpax_static_polarizability_with_screening,
     run_bse_c6[_ks]): the code's  4 mu^T [(A-B)(A+B) + w^2]^-1 (A-B) mu  is
     the symmetric response 4 mu^T [(A+B) + w^2 (A-B)^-1]^-1 mu, equals the
     sum-over-states 2 sum_n |<0|r|n>|^2 Omega_n/(Omega_n^2 + w^2), and at
     w = 0 equals 4 mu^T (A+B)^-1 mu. The reversed order is a different number.
  8  Complete orthonormal PDEP basis: sum_alpha M_alpha,pq M_alpha,rs equals
     the bare RI (pq|rs), so Sigma_x + dSigma_SEX = full screened exchange
     (ferric_gw::cohsex::run_cohsex) and v + sum_alpha w_alpha M M = sum_alpha
     (1/lambda_alpha) M M (ferric_gw::bse::run_bse_tda's `screened`).

Run: python3 scripts/gw_self_energy_sympy.py
"""

import sys

import sympy as sp
from mpmath import mp, mpf, mpc, quad, inf

FAILED = []


def section(title):
    print(f"\n=== {title}")


def check(cond, what):
    ok = bool(cond)
    print(f"  [{'ok' if ok else 'FAIL'}] {what}")
    if not ok:
        FAILED.append(what)


def is_zero(expr):
    return sp.simplify(sp.together(expr)) == 0


w, z = sp.symbols("omega z")
e, b, Om, s, a = sp.symbols("e b Omega s a", positive=True)

# ---------------------------------------------------------------------------
section("1  one-pair RPA dielectric is a single-pole W")
# ferric_rpa::sternheimer::build_scale_factors: Pi = prefactor * B^2 e/(w^2+e^2),
# prefactor 4 closed shell; energy.rs eval_inv_dielectric_matrices: W~_d = eps~^-1 - I.
Pi = 4 * b**2 * e / (w**2 + e**2)
Wd = 1 / (1 + Pi) - 1
Omega2 = e * (e + 4 * b**2)
s_pole = 2 * b**2 * e / sp.sqrt(Omega2)
model = -2 * sp.sqrt(Omega2) * s_pole / (w**2 + Omega2)
check(is_zero(Wd - model), "eps~^-1 - 1 = -2 Omega s/(w^2+Omega^2), Omega^2=e(e+4b^2), s=2b^2 e/Omega")
A_rpa, B_rpa = e + 2 * b**2, 2 * b**2  # singlet direct RPA, K_ia,ia = b^2
check(sp.expand((A_rpa - B_rpa) * (A_rpa + B_rpa) - Omega2) == 0,
      "Omega^2 = (A-B)(A+B) of singlet direct RPA (A = e + 2K, B = 2K)")
Pi_u = 2 * b**2 * e / (w**2 + e**2) + 2 * b**2 * e / (w**2 + e**2)  # Pi_alpha + Pi_beta
check(is_zero(Pi_u - Pi), "unrestricted I + Pi_a + Pi_b (prefactor 2 each) = closed-shell I + Pi at alpha=beta")
Pi_bad = 2 * b**2 * e / (w**2 + e**2)  # negative control: one spin only
check(not is_zero(1 / (1 + Pi_bad) - 1 - model), "negative control: prefactor 2 (one spin) is NOT the closed-shell pole")
check(is_zero(Wd.subs(w, 0) + 4 * b**2 / (e + 4 * b**2)), "W~_d(0) = -4b^2/(e+4b^2) < 0 (eps~ >= I screens)")

# ---------------------------------------------------------------------------
section("2  frequency integral of sigma_c_at_z vs closed-form residues")
# sigma_c_at_z: Sigma = -(1/pi) sum_n integral_0^inf dw' W_mn(iw') (z-eps_n)/((z-eps_n)^2 + w'^2)
W_pole = -2 * Om * s / (w**2 + Om**2)


def sigma_integral(W_of_w, arg):
    return -sp.integrate(W_of_w * arg / (arg**2 + w**2), (w, 0, sp.oo)) / sp.pi


occ = sigma_integral(W_pole, a)    # a = z - eps_n > 0
vir = sigma_integral(W_pole, -a)   # z - eps_n = -a < 0
check(is_zero(occ - s / (a + Om)), "Re(z-eps_n)>0: Sigma = s/(z - eps_n + Omega)  [occupied pole]")
check(is_zero(vir - s / (-a - Om)), "Re(z-eps_n)<0: Sigma = s/(z - eps_n - Omega)  [virtual pole]")
check(not is_zero(occ - s / (a - Om)), "negative control: occupied side is NOT s/(z-eps_n-Omega)")
check(not is_zero(sigma_integral(W_pole, a) / 2 - s / (a + Om)),
      "negative control: a 1/(2pi) prefactor (no even-integrand doubling) misses")

# Complex z on the AC line z = ef + i w0 (what solve_qp_for_mo samples).
mp.dps = 40
for (Omv, sv, epsn, zc) in [(mpf("0.9"), mpf("0.3"), mpf("-0.6"), mpc("-0.1", "0.7")),
                            (mpf("0.9"), mpf("0.3"), mpf("0.4"), mpc("-0.1", "0.7")),
                            (mpf("1.7"), mpf("0.05"), mpf("-1.2"), mpc("-0.25", "2.5"))]:
    d = zc - epsn
    num = -quad(lambda x: (-2 * Omv * sv / (x**2 + Omv**2)) * d / (d**2 + x**2), [0, 1, inf]) / mp.pi
    closed = sv / (d + Omv) if d.real > 0 else sv / (d - Omv)
    check(abs(num - closed) < mpf(10) ** -30, f"complex z={zc}, eps_n={epsn}: |quad - closed| = {mp.nstr(abs(num - closed), 3)}")

# ---------------------------------------------------------------------------
section("3  static W limit = COHSEX (dSigma_SEX + Sigma_COH)")
W0 = sp.symbols("W0", negative=True)
check(is_zero(sigma_integral(W0, a) - (-W0 / 2)), "static W, Re(z-eps_n)>0: -(1/pi) int = -W/2")
check(is_zero(sigma_integral(W0, -a) - (W0 / 2)), "static W, Re(z-eps_n)<0: -(1/pi) int = +W/2")
# Sum over n with z in the gap: occupied n give -W/2, virtual +W/2.
nocc, nvir = 3, 4
Wm = sp.symbols(f"W0:{nocc + nvir}")  # W_mn(0) = sum_alpha w_alpha M_alpha,mn^2
sig = sum(-Wm[i] / 2 for i in range(nocc)) + sum(Wm[nocc + k] / 2 for k in range(nvir))
dsex = -sum(Wm[i] for i in range(nocc))
coh = sp.Rational(1, 2) * sum(Wm)
check(sp.expand(sig - (dsex + coh)) == 0, "sum_n: -1/2 sum_i W + 1/2 sum_a W = -sum_i W + 1/2 sum_p W (cohsex_pieces)")
check(sp.expand(sig - (dsex + 2 * coh)) != 0, "negative control: a COH weight of 1 (not 1/2) misses")
# Plasmon model at fixed W(0) = -2 s/Omega, Omega -> infinity.
s_of = -W0 * Om / 2
check(sp.limit((s_of / (a + Om)), Om, sp.oo) == -W0 / 2, "plasmon occupied s/(a+Omega) -> -W(0)/2 as Omega->inf")
check(sp.limit((s_of / (-a - Om)), Om, sp.oo) == W0 / 2, "plasmon virtual s/(-a-Omega) -> +W(0)/2 as Omega->inf")

# ---------------------------------------------------------------------------
section("4  QP equation: Newton, linearized start, Z")
eps, Dl = sp.symbols("epsilon Delta", real=True)
Sig = sp.Function("Sigma")
f = lambda x: x - eps - Dl - Sig(x)  # solve_qp_for_mo residual (sigma.rs)
x = sp.symbols("x")
fprime = sp.diff(f(x), x)
newton1 = (x - f(x) / fprime).subs(x, eps).doit()
Z = 1 / (1 - sp.Derivative(Sig(x), x).subs(x, eps).doit())
lin = eps + Z * (Sig(eps) + Dl)
check(is_zero(newton1 - lin), "one Newton step from eps = eps + Z (Sigma(eps) + Delta), Z = 1/(1 - Sigma'(eps))")
# Single-pole Sigma: exact roots, Z = residue of G at the QP pole.
p, r = sp.symbols("p r", real=True)
Sig1 = r / (x - p)
G = 1 / (x - eps - Sig1)
roots = sp.solve(sp.numer(sp.together(x - eps - Sig1)), x)
check(len(roots) == 2, "single-pole Sigma: QP equation has two roots (quasiparticle + satellite)")
for rt in roots:
    res = sp.limit((x - rt) * G, x, rt)
    Zr = 1 / (1 - sp.diff(Sig1, x)).subs(x, rt)
    check(is_zero(res - Zr), "residue of G at the root = 1/(1 - Sigma'(root))")
# Quadratic convergence on numbers: eps=-0.5, Delta=0, r=0.02, p=0.9 (virtual-like pole).
vals = {eps: sp.Rational(-1, 2), Dl: 0, r: sp.Rational(1, 50), p: sp.Rational(9, 10)}
fn = sp.lambdify(x, (x - eps - Dl - Sig1).subs(vals), "mpmath")
dfn = sp.lambdify(x, sp.diff(x - eps - Dl - Sig1, x).subs(vals), "mpmath")
exact = min((sp.N(rt.subs(vals), 50) for rt in roots), key=lambda v: abs(v - vals[eps]))
xk = mpf(-0.5)
errs = []
for _ in range(6):
    xk = xk - fn(xk) / dfn(xk)
    errs.append(abs(xk - mpf(str(exact))))
ratios = [errs[k + 1] / errs[k] ** 2 for k in range(3) if errs[k] > mpf(10) ** -35]
check(errs[-1] < mpf(10) ** -35, f"Newton reaches the exact root (final error {mp.nstr(errs[-1], 3)})")
check(all(rt_ < 10 for rt_ in ratios), f"error ratio e_(k+1)/e_k^2 bounded ({[mp.nstr(q, 3) for q in ratios]}): quadratic")

# ---------------------------------------------------------------------------
section("5  PadeCF::deriv_real stencil")
h, x0 = sp.symbols("h x0")
for deg in range(6):
    poly = x**deg
    st = (-poly.subs(x, x0 + 2 * h) + 8 * poly.subs(x, x0 + h) - 8 * poly.subs(x, x0 - h)
          + poly.subs(x, x0 - 2 * h)) / (12 * h)
    errp = sp.expand(st - sp.diff(poly, x).subs(x, x0))
    if deg <= 4:
        check(errp == 0, f"stencil exact for x^{deg}")
    else:
        check(sp.expand(errp - (-h**4 / 30) * sp.diff(poly, x, 5).subs(x, x0)) == 0,
              "x^5: error = -h^4 f^(5)/30")

# ---------------------------------------------------------------------------
section("6  Thiele continued fraction (pade.rs fit/eval, exact arithmetic)")


def thiele_fit(zs, fs):
    """Line-for-line port of PadeCF::fit (pade.rs). Returns coeffs or None on den == 0."""
    n = len(zs)
    g_prev = list(fs)
    coeffs = [sp.Integer(0)] * n
    coeffs[0] = g_prev[0]
    for pp in range(1, n):
        g_curr = [sp.Integer(0)] * n
        for k in range(pp, n):
            num = g_prev[pp - 1] - g_prev[k]
            den = sp.expand((zs[k] - zs[pp - 1]) * g_prev[k])
            if sp.simplify(den) == 0:
                return None, (pp, k)
            g_curr[k] = sp.nsimplify(sp.simplify(num / den))
        coeffs[pp] = g_curr[pp]
        g_prev = g_curr
    return coeffs, None


def thiele_eval(zs, coeffs, zz):
    """Line-for-line port of PadeCF::eval (bottom-up)."""
    acc = sp.Integer(1)
    for pp in range(len(coeffs) - 1, 0, -1):
        acc = 1 + coeffs[pp] * (zz - zs[pp - 1]) / acc
    return coeffs[0] / acc


# Two-pole model (the pade.rs shifted_nodes test function): type [1/2].
f_rat = 1 / (z - sp.Rational(1, 2)) + sp.Rational(3, 10) / (z + sp.Rational(6, 5))
ef = sp.Rational(-1, 10)
nodes = [ef + sp.I * sp.Rational(k, 3) for k in range(1, 8)]
fv = [sp.nsimplify(sp.simplify(f_rat.subs(z, zz))) for zz in nodes]
c4, err4 = thiele_fit(nodes[:4], fv[:4])
check(err4 is None, "4-node fit succeeds")
cf4 = thiele_eval(nodes[:4], c4, z)
check(is_zero(cf4 - f_rat), "4 nodes reproduce the [1/2] rational function identically in z")
c3, _ = thiele_fit(nodes[:3], fv[:3])
cf3 = thiele_eval(nodes[:3], c3, z)
check(all(is_zero(cf3.subs(z, nd) - fv[k]) for k, nd in enumerate(nodes[:3])), "3 nodes: interpolates every node")
check(not is_zero(cf3 - f_rat), "negative control: 3 nodes (one short of the type) do NOT reproduce it")
c5, err5 = thiele_fit(nodes[:5], fv[:5])
check(err5 is None and c5[4] == 0, "5 nodes: the surplus Thiele coefficient a_5 is exactly 0")
_, err6 = thiele_fit(nodes[:6], fv[:6])
check(err6 is not None, f"6 nodes: next denominator is exactly 0 at (p, k) = {err6} -> PadeCF::fit returns Err")
# Single-pole Sigma_c of claim 2 on the AC line is itself [1/1] per pole pair.
sp_sig = sp.Rational(3, 10) / (z + sp.Rational(1, 2) + sp.Rational(9, 10)) \
    + sp.Rational(1, 5) / (z - sp.Rational(2, 5) - sp.Rational(9, 10))
fvs = [sp.nsimplify(sp.simplify(sp_sig.subs(z, zz))) for zz in nodes[:4]]
cs, _ = thiele_fit(nodes[:4], fvs)
check(is_zero(thiele_eval(nodes[:4], cs, z) - sp_sig),
      "two-state single-pole Sigma_c continued exactly from 4 imaginary-axis nodes to all z")

# ---------------------------------------------------------------------------
section("7  RPAx polarizability")
# bse.rs solves t = sysm^-1 (A-B) mu with sysm = (A-B)(A+B) + w^2, alpha = 4 mu^T t.
A11, A12, A22, B11, B12, B22, wq = sp.symbols("A11 A12 A22 B11 B12 B22 w")
Am = sp.Matrix([[A11, A12], [A12, A22]])
Bm = sp.Matrix([[B11, B12], [B12, B22]])
code0 = ((Am - Bm) * (Am + Bm)).inv() * (Am - Bm)
check(sp.simplify(code0 - (Am + Bm).inv()) == sp.zeros(2, 2),
      "[(A-B)(A+B)]^-1 (A-B) = (A+B)^-1: static alpha depends on A+B only")
# Dynamic, on exact rationals: sysm^-1 (A-B) = [(A+B) + w^2 (A-B)^-1]^-1, a symmetric response.
An = sp.Matrix([[3, sp.Rational(1, 2)], [sp.Rational(1, 2), 2]])
Bn = sp.Matrix([[sp.Rational(1, 3), sp.Rational(1, 5)], [sp.Rational(1, 5), sp.Rational(1, 4)]])
mun, wn = sp.Matrix([1, 2]), sp.Rational(7, 10)
sysm = (An - Bn) * (An + Bn) + wn**2 * sp.eye(2)
code_w = sysm.inv() * (An - Bn)
check(sp.simplify(code_w - ((An + Bn) + wn**2 * (An - Bn).inv()).inv()) == sp.zeros(2, 2),
      "sysm^-1 (A-B) = [(A+B) + w^2 (A-B)^-1]^-1")
check(sp.simplify(code_w - code_w.T) == sp.zeros(2, 2), "sysm^-1 (A-B) is symmetric")
wrong = (mun.T * (An - Bn) * sysm.inv() * mun)[0]
check(wrong != (mun.T * code_w * mun)[0],
      "negative control: the other order 4 mu^T (A-B) sysm^-1 mu is a different number")
# 1x1 sum over states: Omega^2 = (A-B)(A+B), <0|r|n> = sqrt(2) mu (X+Y), (X+Y)^2 = (A-B)/Omega.
A1, B1, mu = sp.symbols("A1 B1 mu", positive=True)
Omg = sp.sqrt((A1 - B1) * (A1 + B1))
T2 = 2 * mu**2 * (A1 - B1) / Omg
alpha_code = 4 * mu * (A1 - B1) / ((A1 - B1) * (A1 + B1) + wq**2) * mu
alpha_sos = 2 * T2 * Omg / (Omg**2 + wq**2)
check(is_zero(alpha_code - alpha_sos), "1x1: 4 mu [(A-B)(A+B)+w^2]^-1 (A-B) mu = 2|<0|r|n>|^2 Omega/(Omega^2+w^2)")
check(is_zero(alpha_code.subs(wq, 0) - 4 * mu**2 / (A1 + B1)), "w = 0: alpha = 4 mu^2/(A+B)")
check(not is_zero(2 * mu * (A1 - B1) / ((A1 - B1) * (A1 + B1)) * mu - alpha_sos.subs(wq, 0)),
      "negative control: prefactor 2 (no singlet spin factor) misses")

# ---------------------------------------------------------------------------
section("8  complete orthonormal PDEP basis")
# Exact rational orthogonal V via the Cayley transform of a skew matrix.
S = sp.Matrix([[0, 1, -2], [-1, 0, sp.Rational(1, 3)], [2, -sp.Rational(1, 3), 0]])
V = (sp.eye(3) - S) * (sp.eye(3) + S).inv()
check(sp.simplify(V.T * V - sp.eye(3)) == sp.zeros(3, 3), "Cayley V is exactly orthogonal")
Bpq = sp.Matrix([sp.Rational(1, 2), -sp.Rational(1, 7), sp.Rational(2, 5)])   # B~^P_pq
Brs = sp.Matrix([sp.Rational(3, 4), sp.Rational(1, 9), -sp.Rational(1, 3)])   # B~^P_rs
Mpq, Mrs = V.T * Bpq, V.T * Brs  # project_b_into_pdep: M = V^T B~
check(sp.simplify((Mpq.T * Mrs)[0] - (Bpq.T * Brs)[0]) == 0, "sum_alpha M_pq M_rs = sum_P B~_pq B~_rs = (pq|rs)")
lam = [sp.Rational(5, 2), sp.Rational(3, 2), sp.Rational(11, 10)]
scr_code = (Bpq.T * Brs)[0] + sum((1 / l - 1) * Mpq[k] * Mrs[k] for k, l in enumerate(lam))
scr_full = sum((1 / l) * Mpq[k] * Mrs[k] for k, l in enumerate(lam))
check(sp.simplify(scr_code - scr_full) == 0, "v + sum w_alpha M M = sum (1/lambda_alpha) M M (bse.rs `screened`)")
Vt = V[:, :2]  # truncated basis: the identity needs completeness
check(sp.simplify(((Vt.T * Bpq).T * (Vt.T * Brs))[0] - (Bpq.T * Brs)[0]) != 0,
      "negative control: a truncated PDEP basis does NOT reproduce (pq|rs)")

print()
if FAILED:
    print(f"FAILED {len(FAILED)} claim(s):")
    for fl in FAILED:
        print(f"  - {fl}")
    sys.exit(1)
print("all claims verified")
