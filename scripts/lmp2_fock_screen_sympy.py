#!/usr/bin/env python3
"""Sympy proofs: the Fock-propagated screen of the amplitude-threshold LMP2 mask.

Conventions (crates/ferric-mp2/theory.md sec. 12.1 and 12.6,
crates/ferric-mp2/src/ragged.rs `fock_dressed_gate`): A is the SPD Fock
superoperator in localized orbitals, A = Delta + A' with Delta its diagonal
(orbital-energy denominators) and A' the off-diagonal Fock coupling; the exact
amplitudes are t* = -A^-1 J. A mask splits the amplitude space into kept (K)
and dropped (D) blocks; the masked solve is t_K = -A_KK^-1 J_K, t_D = 0.

Claim 1 (why an integral-size mask leaves a floor). For a stationary energy
of the form E(t) = 2<t,J> + <t,At> (the Hylleraas form with the exchange
weighting absorbed; theory.md 12.1), the masked error is the Schur-complement
norm of the dropped EXACT amplitudes,
    E_mask - E* = t*_D^T (A_DD - A_DK A_KK^-1 A_KD) t*_D  >= 0,
so it is controlled by |t*_D|, not by the dropped integrals J_D and not by
the dropped amplitudes' own energy <t*_D, J_D>. Proved in symbols on a
block 2+1 model (any SPD A).

Claim 2 (what the screen estimates). Expanding t* in the coupling strength
lambda (A' -> lambda A'),
    t*_D = -J_D/Delta_D + lambda (A' J/Delta)_D / Delta_D + O(lambda^2),
and the screen's one-step estimate t_est = -(J + A' t0)/Delta with
t0 = -J/Delta matches it through O(lambda). The dressed integral the gate
thresholds is g~ = -Delta t_est = J + A' t0. A dropped element with J_D = 0
has t*_D = O(lambda): Eq 8 can never see it, the screen does.

Claim 3 (the gate is pair-transpose covariant). With one pair block g and
virtual Fock F (diagonal removed), gate(g^T; i<->j) = gate(g)^T, so the
screened pattern of (j,i) is the transpose of that of (i,j) and the swap
closure of Eq 8 is preserved. Checked in symbols on a 3x3 block.

Independent route (claims 1 and 2): a random 6-dimensional SPD model solved
by brute-force linear algebra (numpy-free, sympy rationals): the masked
energy error equals the Schur formula exactly, and the one-step estimate
matches the exact dropped amplitude to second order in lambda (the ratio of
residuals at lambda and lambda/2 is ~4).

Every check is an assert, so a failure exits nonzero with an AssertionError.
Mutations (each turns the script red; recorded in
crates/ferric-python/tests/test_derivation_scripts.py):
  - flip the sign of the Schur complement's coupling term
    (marker "claim 1: Schur");
  - drop the Fock term from the one-step estimate, t_est = -J/Delta
    (marker "claim 2: first order");
  - drop the right Fock product t0 F' from the gate transcription
    (marker "claim 3: transpose");
  - drop the Fock weights w from the claim-4 bound (marker "claim 4: bound").

Claim 4 (the candidate pre-filter under the screen; ragged.rs
`fock_bound_weights` / `fock_bound_keeps`). With |g_ab| <= q_ia q_jb and
D_c = f_cc + min_b f_bb - 2 max_k f_kk > 0 a lower bound on every Delta,
    |g~_ab| <= q_ia q_jb + q_jb w_ia + q_ia w_jb,
    w_ia = sum_{c != a} |F_ac| q_ic / D_c.
Term-wise slack proved in symbols; the bound checked in exact arithmetic on
random factorized pairs (where the plain q_ia q_jb bound fails), and on the
PR #671 review toy.

Run: python scripts/lmp2_fock_screen_sympy.py
"""

import random

import sympy as sp


def section(title):
    print(f"\n=== {title}")


# ---------------------------------------------------------------------------
# Claim 1: masked error = Schur-complement norm of t*_D (symbolic, 2+1)
# ---------------------------------------------------------------------------
section("claim 1: masked Hylleraas error is the Schur norm of the dropped exact t")
nk, nd = 2, 1
n = nk + nd
# a generic symmetric A; positivity is not needed for the identity
asym = sp.symbols(f"a0:{n * (n + 1) // 2}", real=True)
A = sp.zeros(n, n)
it = iter(asym)
for r in range(n):
    for c in range(r, n):
        v = next(it)
        A[r, c] = v
        A[c, r] = v
J = sp.Matrix(sp.symbols(f"j0:{n}", real=True))


def energy(t):
    return (2 * t.T * J + t.T * A * t)[0, 0]


A_KK, A_KD = A[:nk, :nk], A[:nk, nk:]
A_DK, A_DD = A[nk:, :nk], A[nk:, nk:]
t_star = -A.inv() * J
t_mask = sp.zeros(n, 1)
t_mask[:nk, 0] = -A_KK.inv() * J[:nk, 0]
schur = A_DD - A_DK * A_KK.inv() * A_KD  # claim 1: Schur
t_d = t_star[nk:, 0]
lhs = energy(t_mask) - energy(t_star)
rhs = (t_d.T * schur * t_d)[0, 0]
assert sp.simplify(sp.together(lhs - rhs)) == 0, "claim 1: Schur identity FAILED"
print("E_mask - E* == t*_D^T (A_DD - A_DK A_KK^-1 A_KD) t*_D   (symbolic, 2+1)")

# ---------------------------------------------------------------------------
# Claim 2: the one-step estimate is exact through first order in the coupling
# ---------------------------------------------------------------------------
section("claim 2: one-step Fock-propagated estimate, first order in lambda")
lam = sp.symbols("lambda", real=True)
m = 3
dg = sp.symbols(f"d0:{m}", positive=True)
off = sp.symbols(f"f0:{m * (m - 1) // 2}", real=True)
Delta = sp.diag(*dg)
Ap = sp.zeros(m, m)
it = iter(off)
for r in range(m):
    for c in range(r + 1, m):
        v = next(it)
        Ap[r, c] = v
        Ap[c, r] = v
Jm = sp.Matrix(sp.symbols(f"k0:{m}", real=True))
Afull = Delta + lam * Ap
t_ex = -Afull.inv() * Jm
t0 = -Delta.inv() * Jm
t_est = -Delta.inv() * (Jm + lam * Ap * t0)  # claim 2: first order
for r in range(m):
    diff = sp.series(t_ex[r] - t_est[r], lam, 0, 2).removeO()
    assert sp.simplify(diff) == 0, (
        f"claim 2: first order FAILED at component {r}: {diff}"
    )
print("t*_D - t_est = O(lambda^2) for every component (symbolic, 3x3)")
# A dropped element with zero integral still carries a first-order amplitude
first = sp.diff(t_ex[0], lam).subs(lam, 0).subs(Jm[0], 0)
assert sp.simplify(first) != 0, "claim 2: zero-integral amplitude must be Fock-driven"
print(
    "J_0 = 0 still gives d t*_0 / d lambda != 0: invisible to Eq 8, seen by the screen"
)

# ---------------------------------------------------------------------------
# Claim 3: gate transpose covariance (one pair block, 3 virtuals)
# ---------------------------------------------------------------------------
section("claim 3: dressed gate is covariant under the pair transpose")
nv = 3
fv = sp.symbols(f"e0:{nv}", real=True)
fi, fj = sp.symbols("o_i o_j", real=True)
Fo = sp.zeros(nv, nv)
offv = iter(sp.symbols(f"F0:{nv * (nv - 1) // 2}", real=True))
for r in range(nv):
    for c in range(r + 1, nv):
        v = next(offv)
        Fo[r, c] = v
        Fo[c, r] = v
G = sp.Matrix(nv, nv, sp.symbols(f"g0:{nv * nv}", real=True))


def dressed(g, oi, oj):
    t = sp.Matrix(nv, nv, lambda a, b: -g[a, b] / (fv[a] + fv[b] - oi - oj))
    return g + Fo * t + t * Fo  # claim 3: transpose


lhs = dressed(G.T, fj, fi)
rhs = dressed(G, fi, fj).T
assert sp.simplify(lhs - rhs) == sp.zeros(nv, nv), (
    "claim 3: transpose covariance FAILED"
)
print("g~(g^T, j, i) == g~(g, i, j)^T   (symbolic, 3x3)")

# ---------------------------------------------------------------------------
# Independent route: brute force on a random rational SPD model
# ---------------------------------------------------------------------------
section("numerical cross-check: brute-force linear algebra on a 6-dim model")
rng = random.Random(20261116)
N, NK = 6, 3


def rat(scale):
    return sp.Rational(rng.randint(-1000, 1000), 1000) * scale


D_num = sp.diag(*[sp.Rational(2) + sp.Rational(k, 3) for k in range(N)])
P = sp.zeros(N, N)
for r in range(N):
    for c in range(r + 1, N):
        v = rat(sp.Rational(1, 5))
        P[r, c] = v
        P[c, r] = v
J_num = sp.Matrix([rat(1) if k < NK else 0 for k in range(N)])  # dropped J == 0
A_num = D_num + P
assert all(ev > 0 for ev in A_num.eigenvals()), "model must be SPD"
t_s = -A_num.solve(J_num)
tm = sp.zeros(N, 1)
tm[:NK, 0] = -A_num[:NK, :NK].solve(J_num[:NK, 0])
e = lambda t: (2 * t.T * J_num + t.T * A_num * t)[0, 0]  # noqa: E731
S = A_num[NK:, NK:] - A_num[NK:, :NK] * A_num[:NK, :NK].inv() * A_num[:NK, NK:]
err = e(tm) - e(t_s)
sch = (t_s[NK:, 0].T * S * t_s[NK:, 0])[0, 0]
assert err == sch and err > 0, f"claim 1: Schur cross-check FAILED {err} vs {sch}"
print(
    f"E_mask - E* = {float(err):.6e} == Schur form (exact rationals); J_D = 0, t*_D != 0"
)
own = (t_s[NK:, 0].T * J_num[NK:, 0])[0, 0]
assert own == 0, "the dropped amplitudes' own energy is zero here, the error is not"
print("dropped amplitudes' own energy <t*_D, J_D> = 0 while the error is > 0")
resid = []
for lam_v in (sp.Rational(1, 10), sp.Rational(1, 20)):
    Al = D_num + lam_v * P
    tx = -Al.solve(J_num)
    t0n = -D_num.inv() * J_num
    te = -D_num.inv() * (J_num + lam_v * P * t0n)
    resid.append(max(abs(float(x)) for x in (tx - te)))
ratio = resid[0] / resid[1]
assert 3.0 < ratio < 5.0, (
    f"claim 2: first order FAILED (residual ratio {ratio:.2f}, want ~4)"
)
print(f"|t* - t_est| at lambda=0.1 / 0.05 = {ratio:.2f} (second order: ~4)")

# ---------------------------------------------------------------------------
# Claim 4: the Fock-aware Schwarz bound on the dressed gate
# ---------------------------------------------------------------------------
section("claim 4: |g~_ab| <= q_ia q_jb + q_jb w_ia + q_ia w_jb")
# Term by term: with |g_cb| <= q_ic q_jb (Cauchy-Schwarz on g = B_i^T B_j) and
# Delta_cb >= D_c = f_cc + min f_vv - 2 max f_oo > 0, each Fock term of g~ obeys
# |F_ac g_cb / Delta_cb| <= |F_ac| q_ic q_jb / D_c. Symbolically, the slack
# of one term is a product of non-negative factors:
Fac, qic, qjb, Dc, x = sp.symbols("F_ac q_ic q_jb D_c x", positive=True)
# g_cb = s q_ic q_jb with |s| <= 1, Delta_cb = D_c + x with x >= 0
sg = sp.symbols("s", real=True)
term = Fac * sg * qic * qjb / (Dc + x)
slack = Fac * qic * qjb / Dc - term  # for 0 <= s <= 1 (s < 0 is the mirror)
assert (
    sp.simplify(slack - Fac * qic * qjb * (Dc * (1 - sg) + x) / (Dc * (Dc + x))) == 0
), "claim 4: term slack algebra FAILED"
print("one Fock term: slack = |F| q q (D(1-s) + x) / (D (D + x)) >= 0")


def bound_check(rng4, nv4, naux4, with_w):
    """Exact rationals: g = B_i^T B_j, q^2 = column norms^2; compare squared."""
    R = lambda: sp.Rational(rng4.randint(-999, 999), 1000)  # noqa: E731
    Bi = sp.Matrix(naux4, nv4, lambda p_, a: R() / (1 + a))
    Bj = sp.Matrix(naux4, nv4, lambda p_, a: R() / (1 + 2 * a))
    g = Bi.T * Bj
    qi = [sp.sqrt((Bi[:, a].T * Bi[:, a])[0, 0]) for a in range(nv4)]
    qj = [sp.sqrt((Bj[:, a].T * Bj[:, a])[0, 0]) for a in range(nv4)]
    f = [sp.Rational(3, 5) + sp.Rational(a, 10) for a in range(nv4)]
    oi, oj = sp.Rational(-1, 2), sp.Rational(-2, 5)
    F = sp.zeros(nv4, nv4)
    for r in range(nv4):
        for c in range(r):
            F[r, c] = F[c, r] = R() * sp.Rational(3, 10)
    t0 = sp.Matrix(nv4, nv4, lambda a, b: -g[a, b] / (f[a] + f[b] - oi - oj))
    gt = g + F * t0 + t0 * F
    D = [f[c] + min(f) - 2 * max(oi, oj) for c in range(nv4)]
    wi = [sum(abs(F[a, c]) * qi[c] / D[c] for c in range(nv4)) for a in range(nv4)]
    wj = [sum(abs(F[a, c]) * qj[c] / D[c] for c in range(nv4)) for a in range(nv4)]
    worst = None
    for a in range(nv4):
        for b in range(nv4):
            beta = qi[a] * qj[b]
            if with_w:
                beta += qj[b] * wi[a] + qi[a] * wj[b]
            margin = float(beta - abs(gt[a, b]))
            worst = margin if worst is None else min(worst, margin)
    return worst


rng4 = random.Random(20261119)
worst = min(bound_check(rng4, 4, 3, with_w=True) for _ in range(6))
assert worst >= 0, f"claim 4: bound FAILED (margin {worst:.3e})"  # claim 4: bound
print(f"bound holds on 6 random factorized 4-virtual pairs, min margin {worst:.3e}")
rng4 = random.Random(20261119)
plain = min(bound_check(rng4, 4, 3, with_w=False) for _ in range(6))
assert plain < 0, "the plain Schwarz bound never fails here: the check is vacuous"
print(f"plain Schwarz bound q_ia q_jb fails on the same data (margin {plain:.3e})")

# The review toy (PR #671): one occupied, B = [1e-6, 1], f_oo = -1,
# F_vv = [[2, 1/2], [1/2, 2]], eps = 1e-4. Plain filter drops virtual 0; the
# dressed (0, 1) element clears eps, and the bound keeps virtual 0.
b_t = [sp.Rational(1, 10**6), sp.Integer(1)]
g_t = sp.Matrix(2, 2, lambda a, b: b_t[a] * b_t[b])
F_t = sp.Matrix([[0, sp.Rational(1, 2)], [sp.Rational(1, 2), 0]])
t_t = sp.Matrix(2, 2, lambda a, b: -g_t[a, b] / (4 + 2))
gt_t = g_t + F_t * t_t + t_t * F_t
eps_t = sp.Rational(1, 10**4)
assert b_t[0] * max(b_t) < eps_t and abs(gt_t[0, 1]) > eps_t, "toy: premise FAILED"
w_t = [sum(abs(F_t[a, c]) * b_t[c] / (2 + 2 + 2) for c in range(2)) for a in range(2)]
keep0 = b_t[0] * max(b_t) + max(b_t) * w_t[0] + b_t[0] * max(w_t)
assert keep0 >= eps_t, "claim 4: toy not kept"
print(
    f"toy: g~_01 = {float(gt_t[0, 1]):.4f} > eps; bound for virtual 0 = {float(keep0):.4f}"
)
print("\nALL CLAIMS PASS")
