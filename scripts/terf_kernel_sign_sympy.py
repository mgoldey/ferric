#!/usr/bin/env python3
"""Sympy/mpmath checks of the two sign statements of the terf long range.

Book: chapter "RS-MP2 + LR-RPA", section "The terf long range"
(book/src/26-rs-mp2-rpa/04-two-formulations.md, subsections "Why the terf
kernel is not positive definite" and "The sign of B's correction").
Code consequence: ferric-rpa/src/rs_mp2_rpa.rs never fits terf in its own
metric and `energy.rs::eval_signed_trace_log_summands` refuses a nonpositive
determinant.

Notation: terf(r) = [erf(w (r - r0)) + erf(w (r + r0))] / 2 (eq. erf-terf-2),
kernel W(r) = terf(r)/r, finite at r = 0.

Checks (each prints "FAIL [<name>]" when it fails; exit status nonzero):
  1  [two-charge]         sympy: W(0) = (2 w / sqrt(pi)) exp(-w^2 r0^2) and
                          W(r0) = erf(2 w r0) / (2 r0). The self-interaction
                          of +q at the origin and -q at distance r,
                          q^2 [2 W(0) - 2 W(r)], is -0.917 q^2 / r0 at
                          r0 w = 2, negative at r0 w = 1 and 4, positive at
                          r0 w = 1/sqrt(2) (where the maximum of W sits at the
                          origin and the argument does not apply). The maximum
                          of W lies at 1.26 to 1.39 r0 for r0 w = 1, 2, 4.
  2  [two-charge-fourier] independent route: the same self-interaction as the
                          k-space integral (2/pi) int_0^inf cos(k r0)
                          exp(-k^2/(4 w^2)) [2 - 2 sin(k r)/(k r)] dk, which
                          needs no real-space kernel, for several r0 w and r.
                          The transform is negative at k r0 = pi for every w,
                          so the kernel is indefinite for every finite r0, w.
  3  [one-mode-sign]     sympy: with s = sqrt(1 + 2 k/D) the one-mode bracket
                          dE(k) = E(k) + k^2/(4 D), E(k) = (sqrt(D (D + 2k))
                          - D - k)/2, equals D (s - 1)^3 (s + 3) / 16, whose
                          sign is that of k for every k > -D/2. Numerical
                          scan over -D/2 < k <= 5 D.
  4  [one-mode-integral] independent route: dE(k) from the frequency integral
                          (1/2 pi) int_0^inf [ln(1 + p) - p] dw, p = 2 D k /
                          (D^2 + w^2), by adaptive quadrature at positive and
                          negative k; equals the closed form to 1e-12. At
                          k <= -D/2 the integrand's determinant 1 + p(0) is
                          nonpositive: the boundary where the signed log-det
                          refuses.

Mutations (each turns the named check red):
  - the first erf argument is (r + r0) instead of (r - r0)       [two-charge]
  - the transform drops its cos(k r0) factor                     [two-charge-fourier]
  - the bracket adds k^2/(2 D) instead of k^2/(4 D)              [one-mode-sign]

Run: python3 scripts/terf_kernel_sign_sympy.py  (needs sympy, mpmath, numpy)
"""

import sys

import mpmath as mp
import numpy as np
import sympy as sp

FAILED = []
FIRST_ARG_SIGN = -1  # terf = [erf(w (r + FIRST_ARG_SIGN r0)) + erf(w (r + r0))] / 2
BRACKET_DIV = 4  # the bracket is E(k) + k^2 / (BRACKET_DIV D)

mp.mp.dps = 30


def check(name, ok, detail=""):
    if ok:
        print(f"ok   [{name}] {detail}")
    else:
        print(f"FAIL [{name}] {detail}")
        FAILED.append(name)


# ---------------------------------------------------------- real space
def terf_over_r(r, w, r0):
    """W(r) = terf(r)/r, mpmath."""
    return (mp.erf(w * (r + FIRST_ARG_SIGN * r0)) + mp.erf(w * (r + r0))) / (2 * r)


def pair_energy_real(r, w, r0):
    """q^2 [2 W(0) - 2 W(r)] per q^2, with W(0) as the limit r -> 0."""
    w0 = mp.limit(lambda x: terf_over_r(x, w, r0), 0)
    return 2 * (w0 - terf_over_r(r, w, r0))


def two_charge():
    r, w, r0 = sp.symbols("r omega r0", positive=True)
    num = sp.erf(w * (r + FIRST_ARG_SIGN * r0)) + sp.erf(w * (r + r0))
    num0 = sp.simplify(num.subs(r, 0))
    w_zero = sp.simplify(sp.diff(num, r).subs(r, 0) / 2)
    w_zero_ref = 2 * w / sp.sqrt(sp.pi) * sp.exp(-(w**2) * r0**2)
    ok_sym = num0 == 0 and sp.simplify(w_zero - w_zero_ref) == 0
    w_r0 = sp.simplify(num.subs(r, r0) / (2 * r0))
    ok_sym = ok_sym and sp.simplify(w_r0 - sp.erf(2 * w * r0) / (2 * r0)) == 0
    # r0 = 1, so energies are in units of q^2 / r0.
    pair = sp.lambdify(w, 2 * (w_zero - w_r0).subs(r0, 1), "mpmath")
    at = {c: pair(mp.mpf(c)) for c in (1 / mp.sqrt(2), 1, 2, 4)}
    ok_value = abs(at[2] + mp.mpf("0.917")) < 5e-4
    ok_signs = at[1] < 0 and at[2] < 0 and at[4] < 0 and at[1 / mp.sqrt(2)] > 0
    # Maximum of W: W'(r) = 0 for r > 0, started near 1.3 r0.
    peaks = []
    for c in (1, 2, 4):
        try:
            peaks.append(
                mp.findroot(
                    lambda x, c=c: mp.diff(lambda y: terf_over_r(y, c, 1), x), 1.3
                )
            )
        except ValueError:  # no stationary point near 1.3 r0: the kernel is not terf
            peaks.append(mp.nan)
    ok_peak = all(mp.mpf("1.26") - 5e-3 <= p <= mp.mpf("1.39") + 5e-3 for p in peaks)
    # The sympy limit must agree with the numerical limit of the transcription.
    num_lim = pair_energy_real(mp.mpf(1), mp.mpf(2), mp.mpf(1))
    ok_limit = abs(num_lim - at[2]) < 1e-12
    check(
        "two-charge",
        ok_sym and ok_value and ok_signs and ok_peak and ok_limit,
        f"pair at r0: r0w=1/sqrt2 {float(at[1 / mp.sqrt(2)]):+.4f}, 1 {float(at[1]):+.4f}, "
        f"2 {float(at[2]):+.4f}, 4 {float(at[4]):+.4f} q^2/r0; "
        f"maximum at {[round(float(p), 3) for p in peaks]} r0",
    )


# ------------------------------------------------------------- k space
def transform_phase(k, r0):
    return mp.cos(k * r0)  # PHASE


def pair_energy_fourier(r, w, r0):
    """(2/pi) int_0^inf phase(k) exp(-k^2/4w^2) [2 - 2 sin(k r)/(k r)] dk."""

    def f(k):
        sinc = mp.sin(k * r) / (k * r) if k != 0 else mp.mpf(1)
        return transform_phase(k, r0) * mp.exp(-(k * k) / (4 * w * w)) * (2 - 2 * sinc)

    top = 14 * w
    pts = [0] + [top * i / 28 for i in range(1, 29)]
    return 2 / mp.pi * mp.quad(f, pts)


def two_charge_fourier():
    errs = []
    for c in (1, 2, 4):
        for r in (0.5, 1.0, 1.7):
            r0 = mp.mpf(1)
            w = mp.mpf(c)
            kspace = pair_energy_fourier(mp.mpf(r), w, r0)
            real = pair_energy_real(mp.mpf(r), w, r0)
            errs.append(abs(kspace - real))
    neg_at_pi = all(
        transform_phase(mp.pi / r0, r0) * mp.exp(-((mp.pi / r0) ** 2) / (4 * w * w)) < 0
        for r0 in (mp.mpf("0.7"), mp.mpf(1), mp.mpf(3))
        for w in (mp.mpf("0.2"), mp.mpf(1), mp.mpf(9))
    )
    check(
        "two-charge-fourier",
        max(errs) < 1e-12 and neg_at_pi,
        f"max |k-space - real space| {float(max(errs)):.1e}; transform < 0 at k r0 = pi",
    )


# ------------------------------------------------------------ one mode
def bracket(k, d):
    s = sp.sqrt(d * (d + 2 * k))
    return (s - d - k) / 2 + k**2 / (BRACKET_DIV * d)


def one_mode_sign():
    d, k = sp.symbols("D k", real=True)
    s = sp.symbols("s", positive=True)
    # k = D (s^2 - 1) / 2, s = sqrt(1 + 2 k / D) > 0 exactly when k > -D/2.
    sub = sp.simplify(
        bracket(d * (s**2 - 1) / 2, d).subs(sp.sqrt(d * (d + d * (s**2 - 1))), d * s)
    )
    ok_form = sp.simplify(sub - d * (s - 1) ** 3 * (s + 3) / 16) == 0
    # sign(s - 1) = sign(k) for s > 0 since k = D (s - 1)(s + 1) / 2; s + 3 > 0.
    ok_sign = sp.simplify(d * (s**2 - 1) / 2 - d * (s - 1) * (s + 1) / 2) == 0
    # Scan -D/2 < k <= 5 D at D = 1.3.
    dd = 1.3
    ks = np.concatenate(
        [np.linspace(-0.4999 * dd, -1e-3, 2000), np.linspace(1e-3, 5 * dd, 2000)]
    )
    vals = np.array([float(bracket(sp.Float(x), sp.Float(dd))) for x in ks[::40]])
    ok_scan = bool(np.all(np.sign(vals) == np.sign(ks[::40])))
    check(
        "one-mode-sign",
        ok_form and ok_sign and ok_scan,
        "bracket = D (s-1)^3 (s+3)/16, sign of k for every k > -D/2",
    )


def one_mode_integral():
    dd = mp.mpf("1.3")
    errs = []
    for kk in (-0.45, -0.3, -0.1, 0.05, 0.4, 2.0, 5.0):
        kk = mp.mpf(kk) * dd

        def f(w, kk=kk):
            p = 2 * dd * kk / (dd * dd + w * w)
            return mp.log(1 + p) - p

        e_int = mp.quad(f, [0, 1, 10, mp.inf]) / (2 * mp.pi)
        br_int = e_int + kk**2 / (BRACKET_DIV * dd)
        br_closed = (mp.sqrt(dd * (dd + 2 * kk)) - dd - kk) / 2 + kk**2 / (
            BRACKET_DIV * dd
        )
        errs.append(abs(br_int - br_closed))
        if mp.sign(br_int) != mp.sign(kk):
            errs.append(mp.mpf(1))
    # The boundary: 1 + p(w = 0) = 1 + 2 k / D is nonpositive for k <= -D/2.
    at_boundary = [1 + 2 * (mp.mpf(x) * dd) / dd for x in ("-0.5", "-0.6")]
    ok_boundary = all(v <= 0 for v in at_boundary) and 1 + 2 * (mp.mpf("-0.49")) > 0
    check(
        "one-mode-integral",
        max(errs) < 1e-12 and ok_boundary,
        f"max |integral - closed form| {float(max(errs)):.1e}",
    )


two_charge()
two_charge_fourier()
one_mode_sign()
one_mode_integral()

if FAILED:
    print(f"\n{len(FAILED)} check(s) failed: {FAILED}")
    sys.exit(1)
print("\nall terf sign identities hold")
