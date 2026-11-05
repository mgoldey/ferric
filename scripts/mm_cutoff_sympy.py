#!/usr/bin/env python3
"""Symbolic checks for ferric-mm's non-periodic reaction-field cutoff
(`crates/ferric-mm/src/cutoff.rs`, theory.md section 8). Exits nonzero on any
mismatch; each claim has a negative control that must fail.

Claims:
  1. c_rf = 3 eps / ((2 eps + 1) r_c) equals 1/r_c + k_rf r_c^2 with
     k_rf = (eps - 1)/((2 eps + 1) r_c^3)  (`CutoffParams::reaction_field_constants`,
     crates/ferric-mm/src/cutoff.rs:68).
  2. The reaction-field pair energy q q (1/r + k_rf r^2 - c_rf) vanishes at
     r = r_c; its slope there is -3 q q / ((2 eps + 1) r_c^2) (nonzero for
     finite eps), and the code's dE/dr = q q (-1/r^2 + 2 k_rf r) is its
     derivative (`nonbonded_cutoff`, Coulomb branch, cutoff.rs:253-262).
  3. The switching polynomial the code evaluates in Horner form,
     S = 1 + x^3 (-10 + x (15 - 6 x)), equals 1 - 10x^3 + 15x^4 - 6x^5 with
     S(0) = 1, S(1) = 0 and S', S'' zero at both ends; the code's
     dS/dr = x^2 (-30 + x (60 - 30 x)) / w is dS/dr for x = (r - r_sw)/w
     (cutoff.rs:237-247).
  4. The switched LJ derivative dedr*S + E*dS/dr (code order) is d(E S)/dr.
"""

import sys

import sympy as sp

r, rc, eps, qq, rsw, w, x, sig, ep = sp.symbols(
    "r r_c epsilon qq r_sw w x sigma e", positive=True
)
fails = []


def check(name, expr, expect_zero=True):
    z = sp.simplify(expr) == 0
    if z != expect_zero:
        fails.append(name)
    print(f"{'ok ' if z == expect_zero else 'BAD'} {name}")


k_rf = (eps - 1) / ((2 * eps + 1) * rc**3)
c_rf = 3 * eps / ((2 * eps + 1) * rc)
check("1 c_rf = 1/rc + k_rf rc^2", c_rf - (1 / rc + k_rf * rc**2))
check(
    "1 control: c_rf with 2 eps",
    2 * eps / ((2 * eps + 1) * rc) - (1 / rc + k_rf * rc**2),
    False,
)

e_rf = qq * (1 / r + k_rf * r**2 - c_rf)
check("2 E_rf(r_c) = 0", e_rf.subs(r, rc))
check("2 slope at r_c", sp.diff(e_rf, r).subs(r, rc) + 3 * qq / ((2 * eps + 1) * rc**2))
check("2 code dE/dr", sp.diff(e_rf, r) - qq * (-1 / r**2 + 2 * k_rf * r))
check(
    "2 control: code dE/dr without the 2",
    sp.diff(e_rf, r) - qq * (-1 / r**2 + k_rf * r),
    False,
)

s_code = 1 + x**3 * (-10 + x * (15 - 6 * x))
s_ref = 1 - 10 * x**3 + 15 * x**4 - 6 * x**5
check("3 Horner S", s_code - s_ref)
for k in (1, 2):
    for end in (0, 1):
        check(f"3 S^({k})({end}) = 0", sp.diff(s_ref, x, k).subs(x, end))
check("3 S(0) = 1", s_ref.subs(x, 0) - 1)
check("3 S(1) = 0", s_ref.subs(x, 1))
xr = (r - rsw) / w
ds_code = (x**2 * (-30 + x * (60 - 30 * x)) / w).subs(x, xr)
check("3 code dS/dr", sp.diff(s_ref.subs(x, xr), r) - ds_code)
check(
    "3 control: dS/dr missing 1/w", sp.diff(s_ref.subs(x, xr), r) - ds_code * w, False
)

e_lj = 4 * ep * ((sig / r) ** 12 - (sig / r) ** 6)
dedr = 4 * ep * (-12 * (sig / r) ** 12 + 6 * (sig / r) ** 6) / r
S = s_ref.subs(x, xr)
check("4 dedr = dE_LJ/dr", sp.diff(e_lj, r) - dedr)
check("4 d(E S)/dr", sp.diff(e_lj * S, r) - (dedr * S + e_lj * ds_code))
check(
    "4 control: minus sign", sp.diff(e_lj * S, r) - (dedr * S - e_lj * ds_code), False
)

if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("all claims hold")
