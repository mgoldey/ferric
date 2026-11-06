#!/usr/bin/env python3
"""Sympy checks of the dihedral and bond-angle gradients in ferric-mm.

Theory and notation: crates/ferric-mm/theory.md (sections 3 and 4). Checks
`dihedral_and_gradient` (crates/ferric-mm/src/energy.rs) and the angle block of
`gradient` (energy.rs). Each claim asserts and carries a negative control (a
deliberately wrong variant that must NOT satisfy the identity). Exits nonzero on
the first mismatch.

  1  m x n = r_kj (r_ij . n) with m = r_ij x r_kj, n = r_kj x r_kl (exact,
     symbolic). Hence the code's phi (acos of cos, sign of r_ij . n) is
       phi = atan2(|r_kj| (r_ij . n), m . n).
  2  The Bekker gradient formulas in `dihedral_and_gradient` equal the exact
     d phi / d R of that atan2 form, all four atoms, 12 components, at three
     rational non-planar geometries (60-digit evaluation, residual < 1e-50).
     Controls: dphi/dRi with the opposite sign; the (r_ij.r_kj) and
     (r_kl.r_kj) projections swapped.
  3  Translation invariance (sum over atoms = 0) and rotation invariance
     (sum_a R_a x grad_a = 0) of the gradient: symbolic in the projections p, q.
  4  Bond-angle gradient (energy.rs angle block) equals d theta / d R from
     theta = acos(r_ji . r_jk / (|r_ji||r_jk|)). Control: dropping the
     cos(theta) term.
"""

import sys

import sympy as sp

PREC = 60
TOL = sp.Float(10) ** (-50)


def fail(msg):
    print("FAIL:", msg)
    sys.exit(1)


def vec(name):
    return sp.Matrix(sp.symbols(f"{name}x {name}y {name}z", real=True))


Ri, Rj, Rk, Rl = vec("Ri"), vec("Rj"), vec("Rk"), vec("Rl")
r_ij, r_kj, r_kl = Ri - Rj, Rk - Rj, Rk - Rl
m = r_ij.cross(r_kj)
n = r_kj.cross(r_kl)

# Claim 1
lhs = m.cross(n)
rhs = r_kj * (r_ij.dot(n))
if sp.simplify(sp.expand(lhs - rhs)) != sp.zeros(3, 1):
    fail("claim 1: m x n != r_kj (r_ij . n)")
print("1 ok: m x n = r_kj (r_ij . n)")

kj = sp.sqrt(r_kj.dot(r_kj))
phi = sp.atan2(kj * r_ij.dot(n), m.dot(n))
allvars = list(Ri) + list(Rj) + list(Rk) + list(Rl)
dphi_exact = [sp.diff(phi, v) for v in allvars]


def code_gradient(vals, flip_i=False, swap_pq=False):
    """Port of the gradient block of `dihedral_and_gradient` (energy.rs)."""
    sub = dict(zip(allvars, vals))
    mm, nn = m.subs(sub), n.subs(sub)
    a, b, c = r_ij.subs(sub), r_kj.subs(sub), r_kl.subs(sub)
    knorm = sp.sqrt(b.dot(b))
    gi = (knorm / mm.dot(mm)) * mm
    gl = -(knorm / nn.dot(nn)) * nn
    p = a.dot(b) / b.dot(b)
    q = c.dot(b) / b.dot(b)
    if swap_pq:
        p, q = q, p
    gj = -gi + p * gi - q * gl
    gk = -gl - p * gi + q * gl
    if flip_i:
        gi = -gi
    return list(gi) + list(gj) + list(gk) + list(gl)


geoms = [
    [1, 0, -1, 0, 0, 0, 0, 0, 1, sp.Rational(1, 2), sp.Rational(87, 100), 2],
    [
        sp.Rational(3, 2),
        sp.Rational(-1, 3),
        -1,
        0,
        sp.Rational(1, 7),
        0,
        sp.Rational(1, 5),
        0,
        sp.Rational(6, 5),
        -1,
        sp.Rational(-2, 3),
        sp.Rational(5, 2),
    ],
    [
        1,
        sp.Rational(1, 100),
        -1,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        sp.Rational(-1, 200),
        2,
    ],  # near phi = 0
]
worst_ok = 0
ctrl = {"flip_i": 0, "swap_pq": 0}
for g in geoms:
    sub = dict(zip(allvars, g))
    ex = [sp.N(d.subs(sub), PREC) for d in dphi_exact]
    cd = [sp.N(x, PREC) for x in code_gradient(g)]
    err = max(abs(a - b) for a, b in zip(ex, cd))
    if err > TOL:
        fail(f"claim 2: code gradient != exact d phi/dR, max err {err}")
    worst_ok = max(worst_ok, err)
    for key in ctrl:
        bad = [sp.N(x, PREC) for x in code_gradient(g, **{key: True})]
        ctrl[key] = max(ctrl[key], max(abs(a - b) for a, b in zip(ex, bad)))
for key, val in ctrl.items():
    if val < sp.Float("1e-3"):
        fail(f"claim 2 control {key} not detected (max diff {val})")
print(
    f"2 ok: max |code - exact| = {float(worst_ok):.1e}; controls {[(k, sp.N(v, 3)) for k, v in ctrl.items()]}"
)

# Claim 3: invariances, symbolic in gi, gl, p, q
gi_s, gl_s = vec("gi"), vec("gl")
p_s, q_s = sp.symbols("p q", real=True)
gj_s = -gi_s + p_s * gi_s - q_s * gl_s
gk_s = -gl_s - p_s * gi_s + q_s * gl_s
if sp.simplify(gi_s + gj_s + gk_s + gl_s) != sp.zeros(3, 1):
    fail("claim 3: translation invariance")
print("3 ok: sum of the four gradients = 0 identically in p, q")

# Rotation invariance at a geometry (exact): sum_a R_a x grad_a = 0
g = geoms[1]
cd = code_gradient(g)
tot = sp.zeros(3, 1)
for a in range(4):
    R = sp.Matrix(g[3 * a : 3 * a + 3])
    tot += R.cross(sp.Matrix(cd[3 * a : 3 * a + 3]))
if max(abs(sp.N(x, PREC)) for x in tot) > TOL:
    fail("claim 3: rotation invariance")
print("3 ok: sum_a R_a x grad_a = 0 at a generic geometry")

# Claim 4: bond-angle gradient
ri, rj, rk = vec("Ai"), vec("Aj"), vec("Ak")
a_, b_ = ri - rj, rk - rj
cos_t = a_.dot(b_) / (sp.sqrt(a_.dot(a_)) * sp.sqrt(b_.dot(b_)))
theta = sp.acos(cos_t)
av = list(ri) + list(rj) + list(rk)
pt = dict(
    zip(
        av,
        [
            sp.Rational(9, 10),
            sp.Rational(2, 5),
            sp.Rational(1, 5),
            sp.Rational(1, 10),
            sp.Rational(-1, 5),
            sp.Rational(1, 20),
            sp.Rational(-3, 10),
            sp.Rational(11, 10),
            sp.Rational(3, 5),
        ],
    )
)
A, B = a_.subs(pt), b_.subs(pt)
an, bn = sp.sqrt(A.dot(A)), sp.sqrt(B.dot(B))
ct = A.dot(B) / (an * bn)
sin_t = sp.sqrt(1 - ct**2)
dti = -(1 / sin_t) * (B / (an * bn) - ct * A / an**2)
dtk = -(1 / sin_t) * (A / (an * bn) - ct * B / bn**2)
dtj = -(dti + dtk)
code = list(dti) + list(dtj) + list(dtk)
ex = [sp.N(sp.diff(theta, v).subs(pt), PREC) for v in av]
err = max(abs(sp.N(c, PREC) - e) for c, e in zip(code, ex))
if err > TOL:
    fail(f"claim 4: angle gradient err {err}")
bad = (
    list(-(1 / sin_t) * (B / (an * bn)))
    + [0] * 3
    + list(-(1 / sin_t) * (A / (an * bn)))
)
cerr = max(abs(sp.N(c, PREC) - e) for c, e in zip(bad, ex))
if cerr < sp.Float("1e-3"):
    fail("claim 4 control not detected")
print(
    f"4 ok: angle gradient err {float(err):.1e}; control (no cos term) err {sp.N(cerr, 3)}"
)
print("all claims passed")
