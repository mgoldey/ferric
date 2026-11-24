#!/usr/bin/env python3
"""Closed forms of the pVp integral for same-centre primitives.

pVp_mn = sum_a <d_a m | V | d_a n>, V(r) = -Z / |r - C|, is the one-electron
block the spin-free X2C-1e and DKH Hamiltonians need (W = pVp / (4 c^2)).
ferric computes it in crates/ferric-integrals/src/pvp.rs from libint2's
nuclear-attraction second derivatives; crates/ferric-integrals/tests/pvp.rs
checks that code against the closed forms derived here.

  1  s/s, same centre A, charge Z at distance D from A. With
     m = N_a exp(-a r^2), n = N_b exp(-b r^2), p = a + b:
     grad m . grad n = 4 a b r^2 exp(-p r^2) N_a N_b, and r^2 exp(-p r^2) is
     -d/dp exp(-p r^2), so with the Coulomb potential of a Gaussian,
     int exp(-p r^2)/|r - C| = (pi/p)^(3/2) erf(sqrt(p) D)/D,
       pVp = -4 a b Z N_a N_b I2,
       I2  = 3 pi^(3/2) erf(sqrt(p) D) / (2 D p^(5/2)) - pi exp(-p D^2)/p^2,
     with the D -> 0 limit I2 = 2 pi / p^2.
     Checked against an independent 1-D quadrature that uses the angular
     average of 1/|r - C| over a sphere of radius r about A, 1/max(r, D).
  2  p/p, same centre, charge on the centre: the trace over the three
     Cartesian components m_i = N x_i exp(-a r^2), N = (2a/pi)^(3/4) 2 sqrt(a),
     tr pVp = -Z N_a N_b 4 pi int r [3 - 2 p r^2 + 4 a b r^4] exp(-p r^2) dr
            = -Z N_a N_b 4 pi [3/(2p) - 1/p + 4 a b/p^3].
     The trace is the same in a spherical (pure) p shell, so the check does not
     depend on component order.

Negative controls (each must NOT reproduce the quadrature): dropping one axis
(r^2 -> x^2 + y^2, angular average 2/3), the opposite sign, and the bra-bra
block <d_a d_a m | V | n> (Laplacian on the bra). One integration by parts
gives -<lap m | V | n> = pVp + <grad m . grad V | n>: the bra-bra block is
close to -pVp, the gradient of the remainder sits on m, and the remainder,
-4 pi Z a N_a N_b exp(-p D^2) / p for s/s, is nonzero for every D. The script
checks that identity against quadrature of the bra-bra block (marker
"FAIL bra-bra").

Independent route: the s/s closed form `pvp_ss` (the same expression as
`i2` in crates/ferric-integrals/tests/pvp.rs) is compared with a 30-digit 1-D
radial quadrature of the definition; the p/p trace is rebuilt from the
Cartesian components by direct symbolic integration.

Mutations (each turns the script red, marker "FAIL s/s"):
  - drop the 2 in the 2 D p^(5/2) denominator of the numeric closed form;
  - flip the sign of the exp(-p D^2) term of the numeric closed form;
  - the bra-bra identity with +<lap m|V|n> (marker "FAIL bra-bra");
  - the remainder with the gradient on n (exponent b) instead of m
    (marker "FAIL bra-bra");
  - a remainder that vanishes at D = 0 (marker "FAIL bra-bra").

Exits nonzero on any mismatch. Prints reference values used in the
Rust test docs.
"""

import sys

import mpmath
import sympy as sp

a, b, p, D, r, Z = sp.symbols("a b p D r Z", positive=True)

failures = 0


def check(label, ok):
    global failures
    print(f"{'ok  ' if ok else 'FAIL'} {label}")
    if not ok:
        failures += 1


def norm_s(x):
    return (2 * x / sp.pi) ** sp.Rational(3, 4)


def norm_p(x):
    return (2 * x / sp.pi) ** sp.Rational(3, 4) * 2 * sp.sqrt(x)


# 1. s/s closed form.
coulomb_gauss = (sp.pi / p) ** sp.Rational(3, 2) * sp.erf(sp.sqrt(p) * D) / D
i2 = sp.simplify(-sp.diff(coulomb_gauss, p))
i2_claimed = (
    3
    * sp.pi ** sp.Rational(3, 2)
    * sp.erf(sp.sqrt(p) * D)
    / (2 * D * p ** sp.Rational(5, 2))
    - sp.pi * sp.exp(-p * D**2) / p**2
)
check("I2 = -d/dp of the Gaussian Coulomb potential", sp.simplify(i2 - i2_claimed) == 0)
i2_d0 = sp.limit(i2_claimed, D, 0)
check("I2(D -> 0) = 2 pi / p^2", sp.simplify(i2_d0 - 2 * sp.pi / p**2) == 0)
# Direct radial integral at D = 0: 4 pi int r^2 * r^2 exp(-p r^2) / r dr.
i2_direct0 = sp.integrate(4 * sp.pi * r**3 * sp.exp(-p * r**2), (r, 0, sp.oo))
check("I2(D = 0) by direct radial integration", sp.simplify(i2_direct0 - i2_d0) == 0)


def pvp_ss(av, bv, zv, dv):
    pv = av + bv
    na = (2 * av / mpmath.pi) ** 0.75
    nb = (2 * bv / mpmath.pi) ** 0.75
    if dv == 0.0:
        i2v = 2 * mpmath.pi / pv**2
    else:
        i2v = (
            3 * mpmath.pi**1.5 * mpmath.erf(mpmath.sqrt(pv) * dv) / (2 * dv * pv**2.5)
            - mpmath.pi * mpmath.exp(-pv * dv**2) / pv**2
        )
    return -4 * av * bv * zv * na * nb * i2v


def pvp_ss_quadrature(av, bv, zv, dv, weight=1):
    """Independent: 4 pi int r^2 [weight * 4ab r^2 e^{-p r^2}] / max(r, D) dr."""
    pv = av + bv
    na = (2 * av / mpmath.pi) ** 0.75
    nb = (2 * bv / mpmath.pi) ** 0.75

    def f(x):
        return (
            4
            * mpmath.pi
            * x**2
            * weight
            * 4
            * av
            * bv
            * x**2
            * mpmath.exp(-pv * x**2)
            / max(x, dv)
        )

    pts = [0, dv, mpmath.inf] if dv > 0 else [0, mpmath.inf]
    return -zv * na * nb * mpmath.quad(f, pts)


mpmath.mp.dps = 30
cases = [
    (0.7, 1.3, 1.0, 0.0),
    (0.7, 1.3, 1.0, 0.9),
    (2.5, 0.4, 2.0, 1.7),
    (11.0, 11.0, 1.0, 0.25),
]
for av, bv, zv, dv in [tuple(mpmath.mpf(v) for v in c) for c in cases]:
    cf = pvp_ss(av, bv, zv, dv)
    qd = pvp_ss_quadrature(av, bv, zv, dv)
    rel = abs(cf - qd) / abs(qd)
    check(
        f"s/s a={float(av)} b={float(bv)} Z={float(zv)} D={float(dv)}: closed {mpmath.nstr(cf, 17)} quad rel {mpmath.nstr(rel, 3)}",
        rel < 1e-20,
    )
    # Control: drop one axis (angular weight 2/3 when the charge sits on A;
    # for D > 0 the dropped axis is the one along A-C, still not 2/3 exactly,
    # but the full-sphere quadrature with weight 2/3 is the D = 0 statement).
    if dv == 0.0:
        wrong = pvp_ss_quadrature(av, bv, zv, dv, weight=sp.Rational(2, 3))
        check("  control: dropped axis differs", abs(wrong - cf) > 1e-3 * abs(cf))
    check("  control: opposite sign differs", abs(-cf - qd) > 1e-3 * abs(qd))


# Bra-bra block. One integration by parts (the gradient moves off the
# Laplacian onto V n) gives
#   -<lap m | V | n> = pVp + <grad m . grad V | n>,
# and for same-centre s primitives the remainder has the closed form
#   <grad m . grad V | n> = -4 pi Z a N_a N_b exp(-p D^2) / p,
# because grad m = -2 a r m and the angular average of r . grad V over a sphere
# of radius r is r d/dr(-Z / max(r, D)) = Z / r for r > D and 0 inside. It is
# nonzero for every D, including a charge on the centre, so <lap m|V|n> is
# neither pVp nor -pVp. Evaluate <lap m | V | n> by quadrature:
# lap e^{-a r^2} = (4 a^2 r^2 - 6 a) e^{-a r^2}.
def lap_bra(av, bv, zv, dv):
    na, nb = (2 * av / mpmath.pi) ** 0.75, (2 * bv / mpmath.pi) ** 0.75
    pv = av + bv
    pts = [0, dv, mpmath.inf] if dv > 0 else [0, mpmath.inf]
    return (
        -zv
        * na
        * nb
        * mpmath.quad(
            lambda x: (
                4
                * mpmath.pi
                * x**2
                * (4 * av**2 * x**2 - 6 * av)
                * mpmath.exp(-pv * x**2)
                / max(x, dv)
            ),
            pts,
        )
    )


def bra_grad_remainder(av, bv, zv, dv):
    """<grad m . grad V | n>: the gradient sits on m, the Laplacian's function."""
    na, nb = (2 * av / mpmath.pi) ** 0.75, (2 * bv / mpmath.pi) ** 0.75
    pv = av + bv
    return -4 * mpmath.pi * zv * av * na * nb * mpmath.exp(-pv * dv**2) / pv


for av, bv, zv, dv in [
    (0.7, 1.3, 1.0, 0.0),
    (0.7, 1.3, 1.0, 0.9),
    (2.5, 0.4, 2.0, 1.7),
]:
    av, bv, zv, dv = (mpmath.mpf(v) for v in (av, bv, zv, dv))
    lb = lap_bra(av, bv, zv, dv)
    cf = pvp_ss(av, bv, zv, dv)
    rem = bra_grad_remainder(av, bv, zv, dv)
    resid = abs(-lb - cf - rem) / abs(cf)
    check(
        f"bra-bra D={float(dv)}: -<lap m|V|n> = pVp + <grad m.grad V|n>, rel {mpmath.nstr(resid, 3)}",
        resid < 1e-20,
    )
    # The remainder falls as exp(-p D^2); at D = 1.7 it is 0.2% of pVp, so the
    # control is stated where it is large.
    if dv < 1:
        check(
            f"  control: bra-bra <lap m|V|n> = {mpmath.nstr(lb, 10)} is neither pVp nor -pVp ({mpmath.nstr(cf, 10)})",
            abs(lb - cf) > 1e-3 * abs(cf) and abs(lb + cf) > 1e-3 * abs(cf),
        )
# The remainder at D = 0, unnormalized primitives: -4 pi Z a / p, not zero.
rem0 = bra_grad_remainder(mpmath.mpf(0.7), mpmath.mpf(1.3), 1, 0) / (
    (2 * mpmath.mpf(0.7) / mpmath.pi) ** 0.75
    * (2 * mpmath.mpf(1.3) / mpmath.pi) ** 0.75
)
check(
    f"bra-bra remainder at D = 0 is -4 pi Z a / p = {mpmath.nstr(rem0, 10)}",
    abs(rem0 + 4 * mpmath.pi * 0.7 / 2.0) < 1e-20,
)

# 2. p/p trace, charge on the centre.
integrand = r * (3 - 2 * p * r**2 + 4 * a * b * r**4) * sp.exp(-p * r**2)
tr_rad = sp.integrate(integrand, (r, 0, sp.oo))
tr_claimed = sp.Rational(3, 2) / p - 1 / p + 4 * a * b / p**3
check(
    "p/p radial integral = 3/(2p) - 1/p + 4ab/p^3",
    sp.simplify(tr_rad - tr_claimed) == 0,
)

# Independent: build the three components symbolically in Cartesians and integrate
# grad m_i . grad n_i summed over i, times 1/r, in spherical coordinates.
x, y, z = sp.symbols("x y z", real=True)
th, ph = sp.symbols("theta phi", real=True)
rr = sp.sqrt(x**2 + y**2 + z**2)
tot = 0
for comp in (x, y, z):
    m = comp * sp.exp(-a * rr**2)
    n = comp * sp.exp(-b * rr**2)
    tot += sum(sp.diff(m, q) * sp.diff(n, q) for q in (x, y, z))
sph = {
    x: r * sp.sin(th) * sp.cos(ph),
    y: r * sp.sin(th) * sp.sin(ph),
    z: r * sp.cos(th),
}
tot_s = sp.simplify(tot.subs(sph))
ang = sp.integrate(sp.integrate(tot_s * sp.sin(th), (th, 0, sp.pi)), (ph, 0, 2 * sp.pi))
full = sp.integrate(sp.simplify(ang.subs(a + b, p)) * r**2 / r, (r, 0, sp.oo))
check(
    "p/p trace by full Cartesian construction",
    sp.simplify(full.subs(p, a + b) - 4 * sp.pi * tr_claimed.subs(p, a + b)) == 0,
)

for av, bv in [(0.7, 1.3), (3.0, 3.0)]:
    val = (
        -1.0
        * float(norm_p(av) * norm_p(bv))
        * 4
        * float(sp.pi)
        * float(tr_claimed.subs({a: av, b: bv, p: av + bv}))
    )
    print(f"     p/p trace Z=1 a={av} b={bv}: {val:.17e}")

if failures:
    print(f"{failures} check(s) failed")
    sys.exit(1)
print("all checks passed")
