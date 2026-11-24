#!/usr/bin/env python3
"""Symbolic checks for ferric-mm's non-periodic reaction-field cutoff
(`crates/ferric-mm/src/cutoff.rs`, `book/src/23-embedding/04-molecular-mechanics.md#molecular-mechanics-opt-nonbonded-cutoff`). Exits nonzero on any
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
  5. Independent route: the code's pair energy and dE/dr, transcribed in
     floating point from `nonbonded_cutoff`, agree with a central finite
     difference of that energy (50-digit mpmath, h = 1e-12, so the O(h^2)
     truncation is below 1e-20) at radii below r_sw and inside the switching
     window; E_rf(r_c) = 0 numerically.

  6. The plain (uncut) bond, LJ and Coulomb gradients of energy.rs
     (`bonded_terms`, `plain_pair`): row i gets dE/dr (r_i - r_j)/r and row j
     its negative, with dE/dr = 2 k (r - r0), 4 e s (-12 (sigma/r)^12 +
     6 (sigma/r)^6)/r and -s qq/r^2; each equals the symbolic gradient of
     k (r - r0)^2, 4 e s ((sigma/r)^12 - (sigma/r)^6) and s qq/r.

Mutations (each turns the script red):
  - drop the 1/w from the transcribed switch derivative dS/dr (marker "BAD 5");
  - drop the factor 2 from the transcribed reaction-field force 2 qq k_rf r
    (marker "BAD 5");
  - drop the factor 2 from the transcribed bond dE/dr 2 k (r - r0)
    (marker "BAD 6 bond").
"""

import sys

import mpmath as mp
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
check("1 c_rf = 1/rc + k_rf rc^2 (claim 1)", c_rf - (1 / rc + k_rf * rc**2))
check(
    "1 control: c_rf with 2 eps",
    2 * eps / ((2 * eps + 1) * rc) - (1 / rc + k_rf * rc**2),
    False,
)

e_rf = qq * (1 / r + k_rf * r**2 - c_rf)
check("2 E_rf(r_c) = 0 (claim 2)", e_rf.subs(r, rc))
check(
    "2 slope at r_c (claim 2)",
    sp.diff(e_rf, r).subs(r, rc) + 3 * qq / ((2 * eps + 1) * rc**2),
)
check("2 code dE/dr (claim 2)", sp.diff(e_rf, r) - qq * (-1 / r**2 + 2 * k_rf * r))
check(
    "2 control: code dE/dr without the 2",
    sp.diff(e_rf, r) - qq * (-1 / r**2 + k_rf * r),
    False,
)

s_code = 1 + x**3 * (-10 + x * (15 - 6 * x))
s_ref = 1 - 10 * x**3 + 15 * x**4 - 6 * x**5
check("3 Horner S (claim 3)", s_code - s_ref)
for k in (1, 2):
    for end in (0, 1):
        check(f"3 S^({k})({end}) = 0 (claim 3)", sp.diff(s_ref, x, k).subs(x, end))
check("3 S(0) = 1 (claim 3)", s_ref.subs(x, 0) - 1)
check("3 S(1) = 0 (claim 3)", s_ref.subs(x, 1))
xr = (r - rsw) / w
ds_code = (x**2 * (-30 + x * (60 - 30 * x)) / w).subs(x, xr)
check("3 code dS/dr (claim 3)", sp.diff(s_ref.subs(x, xr), r) - ds_code)
check(
    "3 control: dS/dr missing 1/w", sp.diff(s_ref.subs(x, xr), r) - ds_code * w, False
)

e_lj = 4 * ep * ((sig / r) ** 12 - (sig / r) ** 6)
dedr = 4 * ep * (-12 * (sig / r) ** 12 + 6 * (sig / r) ** 6) / r
S = s_ref.subs(x, xr)
check("4 dedr = dE_LJ/dr (claim 4)", sp.diff(e_lj, r) - dedr)
check("4 d(E S)/dr (claim 4)", sp.diff(e_lj * S, r) - (dedr * S + e_lj * ds_code))
check(
    "4 control: minus sign", sp.diff(e_lj * S, r) - (dedr * S - e_lj * ds_code), False
)

# 5. Numerical route: central differences of the transcribed code energy.
mp.mp.dps = 50


def code_pair(rr, qq_, eps_, rc_, rsw_, sig_, ep_):
    """Pair energy and dE/dr as `nonbonded_cutoff` evaluates them."""
    k = (eps_ - 1) / ((2 * eps_ + 1) * rc_ * rc_ * rc_)
    c = 3 * eps_ / ((2 * eps_ + 1) * rc_)
    sr6 = (sig_ / rr) ** 6
    sr12 = sr6 * sr6
    e_lj = 4.0 * ep_ * (sr12 - sr6)
    d_lj = 4.0 * ep_ * (-12.0 * sr12 + 6.0 * sr6) / rr
    if rr > rsw_:
        w_ = rc_ - rsw_
        xx = (rr - rsw_) / w_
        x2 = xx * xx
        s = 1.0 + x2 * xx * (-10.0 + xx * (15.0 - 6.0 * xx))
        ds = x2 * (-30.0 + xx * (60.0 - 30.0 * xx)) / w_  # dS/dr
        d_lj = d_lj * s + e_lj * ds
        e_lj *= s
    e_c = qq_ / rr + qq_ * k * rr * rr - c * qq_
    d_c = -qq_ / (rr * rr) + 2.0 * qq_ * k * rr
    return e_lj + e_c, d_lj + d_c, e_c


par = dict(
    qq_=mp.mpf("-0.42"),
    eps_=mp.mpf("78.3"),
    rc_=mp.mpf("9.5"),
    rsw_=mp.mpf("7.6"),
    sig_=mp.mpf("6.1"),
    ep_=mp.mpf("0.00024"),
)
h = mp.mpf("1e-12")
worst = mp.mpf(0)
for rr in ("5.3", "6.9", "7.9", "8.6", "9.3"):
    rr = mp.mpf(rr)
    _, d, _ = code_pair(rr, **par)
    fd = (code_pair(rr + h, **par)[0] - code_pair(rr - h, **par)[0]) / (2 * h)
    worst = max(worst, abs(d - fd))
ok5 = worst < 1e-18
print(
    f"{'ok ' if ok5 else 'BAD'} 5 FD of code energy vs code dE/dr (claim 5): {mp.nstr(worst, 3)}"
)
if not ok5:
    fails.append("5 FD")
e_at_rc = code_pair(par["rc_"], **par)[2]
ok5b = abs(e_at_rc) < 1e-40
print(
    f"{'ok ' if ok5b else 'BAD'} 5 E_rf(r_c) = 0 numerically (claim 5): {mp.nstr(e_at_rc, 3)}"
)
if not ok5b:
    fails.append("5 E_rf(r_c)")

# Claim 6: the plain (uncut) bond and pair gradients in energy.rs. The code adds
# dE/dr * (r_i - r_j)/r to row i and its negative to row j.
xi_, yi_, zi_, xj_, yj_, zj_ = sp.symbols("x_i y_i z_i x_j y_j z_j", real=True)
kb, r0, s_lj, s_c = sp.symbols("k r_0 s_lj s_coul", positive=True)
ri, rj = (xi_, yi_, zi_), (xj_, yj_, zj_)
rij = [a - b for a, b in zip(ri, rj)]
rn = sp.sqrt(sum(d * d for d in rij))


def bond_dedr(rr):
    dr = rr - r0
    return 2 * kb * dr  # energy.rs bonded_terms: dedr = 2.0 * b.k * dr


def lj_dedr(rr):
    sr6 = (sig / rr) ** 6
    return s_lj * 4 * ep * (-12 * sr6 * sr6 + 6 * sr6) / rr  # plain_pair


def coul_dedr(rr):
    return -s_c * qq / (rr * rr)  # plain_pair


terms = (
    ("bond k (r - r0)^2", kb * (rn - r0) ** 2, bond_dedr),
    (
        "LJ 4 e s ((sigma/r)^12 - (sigma/r)^6)",
        s_lj * 4 * ep * ((sig / rn) ** 12 - (sig / rn) ** 6),
        lj_dedr,
    ),
    ("Coulomb s qq/r", s_c * qq / rn, coul_dedr),
)
for label, energy, dedr in terms:
    row_i = [dedr(rn) * d / rn for d in rij]
    gi = sum((sp.diff(energy, c) - g) ** 2 for c, g in zip(ri, row_i))
    gj = sum((sp.diff(energy, c) + g) ** 2 for c, g in zip(rj, row_i))
    check(f"6 {label}: rows i, j = +/- dE/dr r_hat (claim 6)", gi + gj)
check(
    "6 control: Coulomb dE/dr with +qq/r^2",
    sp.diff(s_c * qq / rn, xi_) + coul_dedr(rn) * rij[0] / rn,
    False,
)

if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("all claims hold")
