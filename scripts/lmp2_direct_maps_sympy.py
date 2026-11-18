#!/usr/bin/env python3
"""Sympy proofs: how the integral-direct LMP2 maps enter the eps = 0 energy.

Conventions (crates/ferric-mp2/theory.md sec. 12.1, 12.4 and 12.6;
crates/ferric-mp2/src/lmp2_direct.rs `build_locality_maps`): the Hylleraas
functional in localized orbitals is E(t) = 2<t,J> + <t,At>, with A the SPD
Fock superoperator (built from the UNTRUNCATED localized orbitals) and J the
fitted pair integrals; the exact amplitudes are t* = -A^-1 J and
E* = -<J, A^-1 J>. At eps = 0 the amplitude mask keeps everything, so two
kinds of map remain:

  - a PATTERN map (`virt_radius`: t_ij^ab = 0 unless a, b in V_i u V_j)
    restricts t to a kept block K and leaves A and J alone;
  - an INTEGRAL map (`ao_tail`, `schwarz_skip`, `aux_radius`) changes J to
    J + dJ and leaves the pattern and A alone (`ao_tail` zeroes MO
    coefficients only inside the strip transform; the Fock blocks come from
    the full orbitals).

Claim 1 (pattern maps are variational). E_K - E* = t*_D^T S t*_D with S the
Schur complement of A_KK in A, so E_K >= E* whenever A is SPD: the virtual-
domain error is >= 0 and second order in the dropped EXACT amplitudes.
Proved in symbols on a 2+1 block (the identity) and checked for sign on a
random SPD model.

Claim 2 (integral maps are first order and sign-indefinite).
E*(J + dJ) - E*(J) = 2<t*, dJ> - <dJ, A^-1 dJ>: the leading term is linear
in dJ with the sign of <t*, dJ>, which the map does not control. Proved in
symbols (exact, both orders) on a 3-dim model; a random model exhibits both
signs.

Claim 3 (the two errors do not add). With both maps on,
E_K(J + dJ) - E* = t*_D^T S t*_D + 2<t_K, dJ_K> - <dJ_K, A_KK^-1 dJ_K>,
t_K = -A_KK^-1 J_K the restricted amplitudes: the integral error is only
incurred on the kept block, so ablating the maps one at a time does not
predict the production error, and a small production total can be a
cancellation of a positive pattern error against an integral error of the
other sign. Proved in symbols on a 2+1 block.

Independent route: brute-force linear algebra on a random rational SPD
model (sympy rationals, no numpy) checks claims 1-3 numerically, including
the non-additivity: (both on) != (pattern only) + (integral only) - 0.

Every check is an assert, so a failure exits nonzero with an AssertionError.
Mutations (each turns the script red; recorded in
crates/ferric-python/tests/test_derivation_scripts.py):
  - flip the sign of the Schur complement's coupling term
    (marker "claim 1: Schur");
  - drop the quadratic term of the integral-map expansion
    (marker "claim 2: expansion");
  - charge the integral map on the full space instead of the kept block
    (marker "claim 3: kept-block").

Run: python scripts/lmp2_direct_maps_sympy.py
"""

import random

import sympy as sp


def section(title):
    print(f"\n=== {title}")


def sym_matrix(n, name):
    syms = iter(sp.symbols(f"{name}0:{n * (n + 1) // 2}", real=True))
    m = sp.zeros(n, n)
    for r in range(n):
        for c in range(r, n):
            v = next(syms)
            m[r, c] = v
            m[c, r] = v
    return m


def hylleraas(A, J, t):
    return (2 * t.T * J + t.T * A * t)[0, 0]


def restricted_solve(A, J, nk):
    t = sp.zeros(A.rows, 1)
    t[:nk, 0] = -A[:nk, :nk].inv() * J[:nk, 0]
    return t


# ---------------------------------------------------------------------------
# Claim 1: pattern map -> Schur form, >= 0
# ---------------------------------------------------------------------------
section("claim 1: a pattern map's error is the Schur norm of the dropped t*")
nk, nd = 2, 1
n = nk + nd
A = sym_matrix(n, "a")
J = sp.Matrix(sp.symbols(f"j0:{n}", real=True))
t_star = -A.inv() * J
t_k = restricted_solve(A, J, nk)
schur = A[nk:, nk:] - A[nk:, :nk] * A[:nk, :nk].inv() * A[:nk, nk:]  # claim 1: Schur
t_d = t_star[nk:, 0]
lhs = hylleraas(A, J, t_k) - hylleraas(A, J, t_star)
rhs = (t_d.T * schur * t_d)[0, 0]
assert sp.simplify(sp.together(lhs - rhs)) == 0, "claim 1: Schur identity FAILED"
print("E_K - E* == t*_D^T (A_DD - A_DK A_KK^-1 A_KD) t*_D   (symbolic, 2+1)")

# ---------------------------------------------------------------------------
# Claim 2: integral map -> 2<t*, dJ> - <dJ, A^-1 dJ>
# ---------------------------------------------------------------------------
section("claim 2: an integral map's error is first order in dJ")
m = 3
A3 = sym_matrix(m, "b")
J3 = sp.Matrix(sp.symbols(f"k0:{m}", real=True))
dJ = sp.Matrix(sp.symbols(f"d0:{m}", real=True))


def e_star(A_, J_):
    return -(J_.T * A_.inv() * J_)[0, 0]


t3 = -A3.inv() * J3
lhs = e_star(A3, J3 + dJ) - e_star(A3, J3)
rhs = 2 * (t3.T * dJ)[0, 0] - (dJ.T * A3.inv() * dJ)[0, 0]  # claim 2: expansion
assert sp.simplify(sp.together(lhs - rhs)) == 0, "claim 2: expansion FAILED"
print("E*(J+dJ) - E*(J) == 2<t*,dJ> - <dJ,A^-1 dJ>   (symbolic, exact)")

# ---------------------------------------------------------------------------
# Claim 3: both maps on -> integral error only on the kept block
# ---------------------------------------------------------------------------
section("claim 3: the integral error is charged on the kept block only")
dJn = sp.Matrix(sp.symbols(f"e0:{n}", real=True))
both = hylleraas(A, J + dJn, restricted_solve(A, J + dJn, nk)) - hylleraas(A, J, t_star)
AKK = A[:nk, :nk]
dJk = dJn[:nk, 0]
tk = t_k[:nk, 0]
pred = (
    (t_d.T * schur * t_d)[0, 0]
    + 2 * (tk.T * dJk)[0, 0]  # claim 3: kept-block
    - (dJk.T * AKK.inv() * dJk)[0, 0]
)
assert sp.simplify(sp.together(both - pred)) == 0, "claim 3: kept-block form FAILED"
print(
    "E_K(J+dJ) - E* == Schur(t*_D) + 2<t_K,dJ_K> - <dJ_K,A_KK^-1 dJ_K>  (symbolic, 2+1)"
)

# ---------------------------------------------------------------------------
# Independent route: random rational SPD model, brute-force solves
# ---------------------------------------------------------------------------
section("numerical cross-check: brute-force linear algebra on a 6-dim model")
rng = random.Random(20261117)
N, NK = 6, 4


def rat(scale):
    return sp.Rational(rng.randint(-1000, 1000), 1000) * scale


An = sp.diag(*[sp.Rational(2) + sp.Rational(k, 3) for k in range(N)])
for r in range(N):
    for c in range(r + 1, N):
        v = rat(sp.Rational(1, 5))
        An[r, c] += v
        An[c, r] += v
assert all(ev > 0 for ev in An.eigenvals()), "model must be SPD"
Jn = sp.Matrix([rat(1) for _ in range(N)])


def solve_energy(J_, nk):
    t = sp.zeros(N, 1)
    t[:nk, 0] = -An[:nk, :nk].solve(J_[:nk, 0])
    return hylleraas(An, J_, t)


e_exact = solve_energy(Jn, N)
assert e_exact == -(Jn.T * An.solve(Jn))[0, 0], "full solve must give E*"
e_pattern = solve_energy(Jn, NK)
assert e_pattern - e_exact > 0, "claim 1: a pattern map must raise E"
print(f"pattern only: E_K - E* = {float(e_pattern - e_exact):+.6e} (> 0)")

signs = set()
for trial in range(40):
    dj = sp.Matrix([rat(sp.Rational(1, 100)) for _ in range(N)])
    de = solve_energy(Jn + dj, N) - e_exact
    ts = -An.solve(Jn)
    exp2 = 2 * (ts.T * dj)[0, 0] - (dj.T * An.solve(dj))[0, 0]
    assert de == exp2, "claim 2: expansion cross-check FAILED"
    signs.add(de > 0)
assert signs == {True, False}, "claim 2: integral-map error must take both signs"
print("integral only: 40 random dJ, expansion exact, errors of BOTH signs")

dj = sp.Matrix([rat(sp.Rational(1, 100)) for _ in range(N)])
e_int = solve_energy(Jn + dj, N) - e_exact
e_both = solve_energy(Jn + dj, NK) - e_exact
e_sum = (e_pattern - e_exact) + e_int
tk = -An[:NK, :NK].solve(Jn[:NK, 0])
djk = dj[:NK, 0]
pred = (
    (e_pattern - e_exact)
    + 2 * (tk.T * djk)[0, 0]
    - (djk.T * An[:NK, :NK].solve(djk))[0, 0]
)
assert e_both == pred, "claim 3: kept-block cross-check FAILED"
assert e_both != e_sum, "claim 3: the two errors must NOT add"
print(
    f"both on: {float(e_both):+.6e} == kept-block prediction; "
    f"one-at-a-time sum {float(e_sum):+.6e} differs"
)
print("\nALL CLAIMS PASS")
