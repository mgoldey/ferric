#!/usr/bin/env python3
"""Is CSB (and plain Schwarz) valid for ferric's `terfc` / `terf` operators?

Answer: YES for `terfc`, NO for `terf`, with a closed form rather than a sweep.

WHY THIS SCRIPT EXISTS
----------------------
Thompson & Ochsenfeld (JCP 147, 144101 (2017)) prove that the Schwarz
inequality — and hence CSB, which is a `min` of three Schwarz-type bounds —
applies to a multiplicative kernel `G(r12)` exactly when `G` is POSITIVE
DEFINITE, which by Bochner's theorem means its 3-D Fourier transform satisfies
`F_G(k) > 0` for all `k > 0`. Their Appendix B works this out for six kernels
(Table VI): `1/r`, `e^{-g r}`, `e^{-g r}/r`, `erf(w r)/r`, `erfc(w r)/r`,
`e^{-a r^2}`.

ferric has two kernels NOT on that list, implemented via the Dutoi/Goldey 2-D
interpolation-table engine rather than libint2. Appendix B is a recipe, not
just a table, so we apply it.

THE KERNELS, read from the shim (crates/ferric-integrals/shim/shim.cc:717-718)
rather than from the name:

    terfc(r,r0)/r = 1/r  -  terf(r,r0)/r
    terf(r,r0)/r  = ( erf(w(r-r0)) + erf(w(r+r0)) ) / (2 r),   w = 1/(r0 sqrt2)

so  G_terfc(r) = t(r)/r  with
    t(r) = 1 - [ erf(w(r-r0)) + erf(w(r+r0)) ] / 2

a smoothed step from t(0)=1 to t(inf)=0, edge at r=r0, edge width ~1/w.

NOTE a discrepancy this script also pins: terf-tables/base_terfc_closed.py's
docstring writes the same expression WITHOUT the /2. That form goes negative
past r~r0 (check_convention below) and is not the shipped kernel.

THE RESULT
----------
For a radial G, the 3-D transform reduces to a 1-D sine transform:

    F_G(k) = (1 / (2 pi^2 k)) * int_0^inf r G(r) sin(k r) dr

For G = t(r)/r the `r` cancels and the integral is just `P(k) = int_0^inf
t(r) sin(kr) dr`. Because `t(r) - 1` is ODD (verified below), that integral has
a closed form:

    P(k)       = ( 1 - cos(k r0) e^{-k^2/(4 w^2)} ) / k
    F_terfc(k) = ( 1 - cos(k r0) e^{-k^2/(4 w^2)} ) / (2 pi^2 k^2)

POSITIVITY IS THEN IMMEDIATE, with no numerics: |cos(k r0)| <= 1 and
e^{-k^2/(4w^2)} < 1 strictly for every k>0 and finite w, so the bracket is
>= 1 - e^{-k^2/(4w^2)} > 0. terfc is positive definite for ALL (r0, w).

And by `terf = Coulomb - terfc`:

    F_terf(k) = cos(k r0) e^{-k^2/(4 w^2)} / (2 pi^2 k^2)

which is NEGATIVE wherever cos(k r0) < 0. terf is NOT positive definite, so
neither CSB nor plain Schwarz is valid for it — not a tightness issue, the
underlying inner-product axiom fails.

WHAT THIS SCRIPT CHECKS (all four must pass)
--------------------------------------------
  1. convention  — which of the two written forms of `t` is the shipped one
  2. parity      — t(r) + t(-r) == 2, the fact that makes the closed form work
  3. closed form — P(k) closed vs direct oscillatory quadrature, 72 points
  4. limits      — r0->0 reproduces the paper's own Table VI erfc row;
                   w->inf reproduces the sharp-cutoff (1-cos)/k boundary case

Relationship to scripts/terfc_pd_check.py (2026-08-13): that script SWEEPS
P(u) numerically over c = r0*w in [0.2, 50] and finds min P = 5.0e-4 > 0. This
one proves the same statement in closed form for all (r0, w). A sweep can only
fail to find a violation; this shows there is none. They agree.

Run: python3 scripts/terfc_csb_positivity.py     (needs numpy + scipy)
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import erf

# Reference points on the c = r0*w axis. 1/sqrt2 is the curvature-LINKED value
# the shim's engine-create asserts; 2.06987 is the Dutoi bound; the others
# bracket the decoupled family (commit 03214031 freed r0 and omega).
C_REFERENCE = [0.3, 1.0 / np.sqrt(2.0), 1.0, 2.06987, 8.0]


def t_shipped(r, r0, w):
    """t(r) with the /2 — the shim's form (shim.cc:717-718)."""
    return 1.0 - 0.5 * (erf(w * (r - r0)) + erf(w * (r + r0)))


def t_unhalved(r, r0, w):
    """t(r) WITHOUT the /2 — base_terfc_closed.py's docstring form."""
    return 1.0 - (erf(w * (r - r0)) + erf(w * (r + r0)))


def p_closed(k, r0, w):
    """P(k) = int_0^inf t(r) sin(kr) dr, in closed form."""
    return (1.0 - np.cos(k * r0) * np.exp(-k * k / (4.0 * w * w))) / k


def p_quadrature(k, r0, w):
    """P(k) by direct oscillatory quadrature — the INDEPENDENT construction.

    Integrated half-period by half-period of sin(kr) out to where t(r) has
    decayed below ~1e-30, so the oscillation never defeats the quadrature.
    """
    xmax = r0 + 9.0 / w
    period = np.pi / k
    edges, x = [0.0], 0.0
    while x < xmax:
        x += period
        edges.append(min(x, xmax))
    return sum(
        quad(lambda r: t_shipped(r, r0, w) * np.sin(k * r), a, b, limit=300)[0]
        for a, b in zip(edges[:-1], edges[1:])
    )


def check_convention():
    """Which written form is the shipped kernel? Only one can be a screener."""
    print("1. CONVENTION: terfc must decay 1 -> 0 and stay non-negative.")
    print("   (tests/terfc_base_validation.rs asserts terfc/coulomb in (0,1).)\n")
    r0, w = 1.0, 1.0 / np.sqrt(2.0)
    print(f"   {'r':>8}{'shipped (/2)':>16}{'un-halved':>16}")
    bad = False
    for r in [1e-6, 0.5, 1.0, 2.0, 5.0, 20.0]:
        a, b = t_shipped(r, r0, w), t_unhalved(r, r0, w)
        print(f"   {r:>8.3g}{a:>16.6e}{b:>16.6e}")
        if b < -1e-12:
            bad = True
    assert not np.any([t_shipped(r, r0, w) < -1e-12 for r in np.linspace(0, 30, 3000)]), \
        "shipped form went negative — it is not a valid short-range attenuator"
    assert bad, "un-halved form did NOT go negative; the two forms are not distinguishable here"
    print("\n   -> the /2 form is the shipped kernel; the un-halved docstring form")
    print("      goes negative past r~r0 and is a DOC BUG in base_terfc_closed.py.\n")


def check_parity():
    """t(r) - 1 must be ODD. This is what makes the closed form exist."""
    print("2. PARITY: t(r) + t(-r) == 2  <=>  t-1 is odd.\n")
    worst = 0.0
    for r0 in [0.7, 1.3, 2.0]:
        for c in C_REFERENCE:
            w = c / r0
            for r in [0.3, 1.0, 2.0, 4.5]:
                worst = max(worst, abs(t_shipped(r, r0, w) + t_shipped(-r, r0, w) - 2.0))
    print(f"   max |t(r) + t(-r) - 2| over the reference grid = {worst:.3e}")
    assert worst < 1e-12, "t-1 is not odd; the closed-form derivation does not apply"
    print("   -> odd. int_0^inf t sin = 1/k - (1/2) int_{-inf}^{inf} (t-1) sin.\n")


def check_closed_form():
    """Closed form vs INDEPENDENT quadrature. The load-bearing check."""
    print("3. CLOSED FORM vs direct oscillatory quadrature.\n")
    print("   P(k) = (1 - cos(k r0) exp(-k^2/(4 w^2))) / k\n")
    worst, n = 0.0, 0
    for r0 in [0.7, 1.0, 1.98]:
        for c in [0.3, 1.0 / np.sqrt(2.0), 2.06987, 8.0]:
            w = c / r0
            # Include the k = 2 pi m / r0 points: the sharp-cutoff limit's
            # zeros, i.e. exactly where positivity is most at risk.
            for k in [0.05, 1.0, 2 * np.pi / r0, 4 * np.pi / r0, 13.0, 40.0]:
                d = abs(p_closed(k, r0, w) - p_quadrature(k, r0, w))
                worst = max(worst, d)
                n += 1
    print(f"   {n} (r0, c, k) points;  max |closed - quadrature| = {worst:.3e}")
    assert worst < 1e-12, f"closed form disagrees with quadrature by {worst:.3e}"
    print("   -> the closed form is correct.\n")


def check_limits():
    """Both limits must land on things the PAPER already derived."""
    print("4. LIMITS — each must reproduce a known result.\n")

    # r0 -> 0 must give the paper's Table VI erfc row.
    print("   (a) r0 -> 0  =>  (1 - exp(-k^2/(4w^2))) / k   [paper Table VI, erfc]")
    worst = 0.0
    for w in [0.4, 1.0, 3.0]:
        for k in [0.2, 1.0, 5.0, 25.0]:
            want = (1.0 - np.exp(-k * k / (4 * w * w))) / k
            worst = max(worst, abs(p_closed(k, 0.0, w) - want))
    print(f"       max deviation = {worst:.3e}")
    assert worst < 1e-14, "r0->0 does not reduce to the paper's erfc row"

    # w -> inf must give the sharp-cutoff boundary case.
    print("   (b) w -> inf =>  (1 - cos(k r0)) / k           [sharp cutoff]")
    worst = 0.0
    for r0 in [0.8, 1.5]:
        for k in [0.5, 2 * np.pi / r0, 7.0]:
            want = (1.0 - np.cos(k * r0)) / k
            worst = max(worst, abs(p_closed(k, r0, 1e7) - want))
    print(f"       max deviation = {worst:.3e}")
    assert worst < 1e-12, "w->inf does not reduce to the sharp-cutoff form"
    print("       (this limit TOUCHES zero at k = 2 pi m / r0 — it is the")
    print("        boundary case, and the finite edge is what lifts it.)\n")


def report_verdicts():
    """The two answers, stated plainly, with the margin that supports each."""
    print("=" * 72)
    print("VERDICTS")
    print("=" * 72 + "\n")

    print("TERFC: POSITIVE DEFINITE for all (r0, w).  CSB and Schwarz are VALID.\n")
    print("   F_terfc(k) = (1 - cos(k r0) exp(-k^2/(4 w^2))) / (2 pi^2 k^2)")
    print("   |cos| <= 1 and exp(-k^2/(4w^2)) < 1 strictly for k>0, finite w,")
    print("   so the bracket >= 1 - exp(-k^2/(4w^2)) > 0. No sweep needed.\n")
    # Scan the bracket away from the k->0 endpoint. Near k=0 the bracket is
    # O(k^2) -- analytically positive, but `1 - cos(k) exp(-k^2/4c^2)` in
    # float64 CANCELS to exactly 0.0 below k ~ 1e-8, because both factors are
    # within 1 ulp of 1. That is a floating-point artifact of evaluating the
    # bracket in this form, NOT a positivity failure: the small-k expansion is
    #     bracket(k) = (1/2 + 1/(4c^2)) k^2 + O(k^4)  >  0,
    # checked separately below. Scanning from k=1e-3 keeps the printed minimum
    # a real measurement rather than a report of catastrophic cancellation.
    print(f"   {'c = r0 w':>10}{'min bracket, k in [1e-3, 200]':>32}")
    k = np.linspace(1e-3, 200.0, 2_000_000)
    for c in C_REFERENCE + [50.0]:
        br = 1.0 - np.cos(k * 1.0) * np.exp(-k * k / (4 * c * c))  # r0 = 1
        assert br.min() > 0.0, f"bracket hit {br.min()} at c={c}"
        print(f"   {c:>10.4f}{br.min():>32.6e}")
    print()
    print("   small-k behaviour, checked against the analytic expansion")
    print("   bracket(k) -> (1/2 + 1/(4 c^2)) k^2 :\n")
    print(f"   {'c':>8}{'k':>10}{'bracket/k^2':>16}{'predicted':>14}")
    for c in [0.3, 1.0, 8.0]:
        for kk in [1e-2, 1e-3]:
            br = 1.0 - np.cos(kk) * np.exp(-kk * kk / (4 * c * c))
            pred = 0.5 + 1.0 / (4 * c * c)
            print(f"   {c:>8.2f}{kk:>10.0e}{br / kk**2:>16.6f}{pred:>14.6f}")
            assert abs(br / kk**2 - pred) < 1e-3 * pred, "small-k expansion disagrees"
    print("\n   -> positive as k->0 too; the coefficient is strictly positive for")
    print("      every finite c, so the bracket has no zero on k > 0.\n")

    print("TERF:  NOT positive definite.  CSB and Schwarz are BOTH INVALID.\n")
    print("   F_terf(k) = cos(k r0) exp(-k^2/(4 w^2)) / (2 pi^2 k^2)")
    print("   cos(k r0) changes sign, so F_terf < 0 on k r0 in (pi/2, 3pi/2):\n")
    print(f"   {'k r0':>10}{'cos(k r0)':>14}{'sign of F_terf':>18}")
    for kr0 in [0.5 * np.pi, 0.75 * np.pi, np.pi, 1.25 * np.pi]:
        c = np.cos(kr0)
        print(f"   {kr0:>10.4f}{c:>14.6f}{('NEGATIVE' if c < 0 else 'ok'):>18}")
    assert np.cos(np.pi) < 0
    print("\n   Appendix A requirement four (<e,e> >= 0) FAILS, so the two-electron")
    print("   integral over terf is not an inner product and the Cauchy-Schwarz")
    print("   step behind BOTH Q and M is unavailable. A 'Schwarz bound' for terf")
    print("   could be violated outright — this is not a tightness question.\n")

    print("-" * 72)
    print("PRACTICAL BLOCKER for terfc, despite validity:")
    print("  CSB needs (PP|QQ) and (PQ|PQ) shell QUARTETS. The terfc engine")
    print("  exposes only scf_compute_terfc_eri3 / _eri2 (shim.h:201,206) —")
    print("  there is no 4-centre terfc kernel. So schwarz() and csb_m_table()")
    print("  reject Terfc today for a MISSING-ENGINE reason, not a positivity")
    print("  one. Write the quartet kernel and CSB follows with no new theory.")
    print()
    print("  Remaining caveat: the terfc engine is INTERPOLATED (2-D tables in")
    print("  (S,s)). The proof above is for the EXACT kernel. A Q or M entry")
    print("  computed slightly LOW by table error breaks the bound by that much,")
    print("  so a terfc CSB needs the table error bounded and folded in as a")
    print("  relative inflation of Q/M — not merely floored. Not solved here.")
    print("-" * 72)


def main():
    print(__doc__.split("Run:")[0])
    print("=" * 72 + "\n")
    check_convention()
    check_parity()
    check_closed_form()
    check_limits()
    report_verdicts()
    print("\nALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
