#!/usr/bin/env python3
"""Radial Gaussian overlap and unit self-overlap of a contracted shell.

Theory: book/src/02-molecules-and-numerics/02-basis-sets.md, "Normalization"
(equation 2.5); code: crates/ferric-core/src/basis.rs (parse_bse_json
renormalizes each contraction to unit self-overlap; the Rust test
every_bundled_contraction_has_unit_self_overlap checks every bundled shell).

Primitives are the radial parts r^l exp(-a r^2); the angular part is a
normalized spherical harmonic, so the radial integral carries r^2 dr:

  S_l(a, b) = int_0^oo r^{2l+2} exp(-(a+b) r^2) dr = Gamma(l+3/2) / (2 (a+b)^{l+3/2}).

The overlap of two NORMALIZED primitives is S_l(a,b)/sqrt(S_l(a,a) S_l(b,b)),

  (2 sqrt(ab) / (a + b))^{l+3/2}.                                        (2.5)

Claims
  1  The closed form (2.5) equals the symbolic integral, l = 0..6.
  2  The same closed form equals numerical quadrature of the two normalized
     radial functions (mpmath, independent of the symbolic integral), for random
     exponents and l = 0..6, to 1e-12.
  3  A contraction with random coefficients, renormalized with the
     closed-form overlap matrix (2.5), has unit self-overlap by quadrature of
     the contracted radial function. Negative control: the unrenormalized
     contraction does not.

Mutations (each turns the script red, marker "FAIL"):
  - exponent of the closed form in closed_form() l + 3/2 -> l + 1 (claims 1
    and 3 fail; claim 2 carries its own copy of (2.5) and is the independent
    route);
  - the contraction renormalized with the sum of squared coefficients instead
    of the overlap matrix (claim 3 fails).

Run: python3 scripts/basis_normalization_sympy.py
"""

import sys

import mpmath as mp
import numpy as np
import sympy as sp

FAILED = []


def check(cond, msg):
    print(f"{'ok  ' if cond else 'FAIL'} {msg}")
    if not cond:
        FAILED.append(msg)


def closed_form(a, b, l):
    return (2 * sp.sqrt(a * b) / (a + b)) ** (l + sp.Rational(3, 2))


def claim1():
    print("\n=== 1  closed form == symbolic integral of normalized primitives")
    a, b, r = sp.symbols("a b r", positive=True)
    ok = True
    for l in range(7):
        s_ab = sp.integrate(r ** (2 * l + 2) * sp.exp(-(a + b) * r**2), (r, 0, sp.oo))
        s_aa = sp.integrate(r ** (2 * l + 2) * sp.exp(-2 * a * r**2), (r, 0, sp.oo))
        s_bb = sp.integrate(r ** (2 * l + 2) * sp.exp(-2 * b * r**2), (r, 0, sp.oo))
        ratio = s_ab / sp.sqrt(s_aa * s_bb)
        # numeric spot check of the symbolic identity at a fixed pair, exact in sympy
        pt = {a: sp.Rational(3, 7), b: sp.Rational(11, 5)}
        diff = sp.simplify(ratio.subs(pt) - closed_form(a, b, l).subs(pt))
        ok &= sp.N(diff, 30) == 0 or abs(sp.N(diff, 30)) < 1e-25
        # and as a function: the squared ratio is a rational function of a, b
        sq = sp.simplify(ratio**2 - closed_form(a, b, l) ** 2)
        ok &= sq == 0
    check(ok, "S_l(a,b)/sqrt(S_l(a,a)S_l(b,b)) == (2 sqrt(ab)/(a+b))^(l+3/2), l=0..6")


def norm_radial(a, l):
    """Normalization constant of r^l exp(-a r^2) with the r^2 dr measure."""
    return 1 / mp.sqrt(
        mp.gamma(l + mp.mpf(3) / 2) / (2 * (2 * a) ** (l + mp.mpf(3) / 2))
    )


def claim2():
    print("\n=== 2  closed form == quadrature of the normalized radial functions")
    mp.mp.dps = 30
    rng = np.random.default_rng(11)
    worst = 0.0
    for l in range(7):
        for _ in range(4):
            a, b = (float(x) for x in np.exp(rng.uniform(-3, 4, 2)))
            na, nb = norm_radial(a, l), norm_radial(b, l)
            q = mp.quad(
                lambda r: na * nb * r ** (2 * l + 2) * mp.exp(-(a + b) * r**2),
                [0, 1, 5, mp.inf],
            )
            cf = (2 * mp.sqrt(a * b) / (a + b)) ** (l + 1.5)
            worst = max(worst, float(abs(q - cf)))
    check(worst < 1e-12, f"max |quadrature - closed form| = {worst:.1e} (l=0..6)")


def contraction_overlap(exps, coef, l, normalize_prims=True):
    """Self-overlap of sum_i c_i N_i r^l exp(-a_i r^2) from the closed form."""
    n = len(exps)
    s = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            s[i, j] = float(closed_form(exps[i], exps[j], l))
    return coef @ s @ coef


def claim3():
    print("\n=== 3  renormalized contraction has unit self-overlap (quadrature)")
    mp.mp.dps = 30
    rng = np.random.default_rng(5)
    worst, raw_dev = 0.0, []
    for l in range(5):
        exps = np.sort(np.exp(rng.uniform(-2, 4, 4)))[::-1]
        coef = rng.uniform(0.1, 1.0, 4) * np.array([1, -1, 1, -1])
        s = contraction_overlap(exps, coef, l)
        c_unit = coef / np.sqrt(s)  # overlap-matrix renormalization
        for cc, tag in ((c_unit, "unit"), (coef, "raw")):

            def f(r, cc=cc):
                v = sum(
                    float(c)
                    * norm_radial(float(a), l)
                    * r**l
                    * mp.exp(-float(a) * r**2)
                    for c, a in zip(cc, exps)
                )
                return v * v * r**2

            q = mp.quad(f, [0, 0.5, 2, 8, mp.inf])
            if tag == "unit":
                worst = max(worst, float(abs(q - 1)))
            else:
                raw_dev.append(float(abs(q - 1)))
    check(worst < 1e-12, f"max |<phi|phi> - 1| after renormalization = {worst:.1e}")
    check(min(raw_dev) > 1e-6, "negative control: the raw contraction is not unit-norm")


def main():
    claim1()
    claim2()
    claim3()
    if FAILED:
        print(f"\n{len(FAILED)} FAILED")
        sys.exit(1)
    print("\nall claims hold")


if __name__ == "__main__":
    main()
