"""Sympy proof: the lambda-Boys occupied localization of amplitude-threshold LMP2.

Book: book/src/13-local-mp2/01-amplitude-lmp2.md#amplitude-lmp2-lambda-boys.
Code: crates/ferric-mp2/src/boys.rs (lambda_boys_rotation,
lambda_boys_pair_gradient, lambda_boys_pair_angle, rotate_sym_pair).

Objective, over orthogonal rotations of the occupied orbitals,

    G = sum_{x=1..3} sum_i (r^x_ii)^2 + lambda * sum_i F_ii^2 ,

with r^x the occupied dipole matrices and F the occupied Fock block. The
code's plane rotation of orbitals p < q is p' = c p + s q, q' = -s p + c q
(c = cos theta, s = sin theta), i.e. X -> R^T X R with R[p,p] = R[q,q] = c,
R[q,p] = s, R[p,q] = -s.

Claims:

S1 (gradient). dG/dtheta at theta = 0 is 4 sum_X w_X (X_pp - X_qq) X_pq,
   (X, w_X) in {(r^x, 1), (F, lambda)}: lambda_boys_pair_gradient. Diagonal
   entries other than p, q do not move.
S2 (Jacobi angle). With d_X = (X_pp - X_qq)/2, c_X = X_pq,
   a = sum w (d^2 - c^2), b = 2 sum w d c, the in-plane objective is
   G(theta) = const + a cos 4theta + b sin 4theta. theta* = atan2(b, a)/4
   has G'(theta*) = 0 and G''(theta*) = -16 sqrt(a^2 + b^2) <= 0: the
   in-plane MAXIMUM (lambda_boys_pair_angle).
S3 (stationarity). The pair gradient equals 4 b, so a point is stationary
   iff b_pq = 0 for every pair:
   sum_x r^x_pq (r^x_pp - r^x_qq) + lambda F_pq (F_pp - F_qq) = 0. At such
   a point theta*_pq = 0 exactly when a_pq > 0, so a converged Jacobi sweep
   leaves a stationary point that is a maximum in every plane.
S4 (lambda = 0 is Boys). With lambda = 0 the step is the step of the plain
   Boys code (boys.rs::boys_localize, rotation x' = c x - s y, angle
   atan2(-B, A)/4 with A = a, B = b): the two rotate orbitals identically.

Numerical cross-check, by an independent route: the analytic antisymmetric
gradient matrix is compared against a finite difference of G along random
directions through the matrix exponential (no plane rotation, no pair
formula); the Jacobi optimum's gradient vanishes, and finite-difference
second derivatives along random antisymmetric directions are <= 0 there
(a local maximum, not a saddle). Monotonicity: G never falls during the
sweeps.

Mutations (crates/ferric-python/tests/test_derivation_scripts.py):
  - drop the factor 4 from the pair gradient (marker "S1: gradient");
  - take the in-plane minimum branch, atan2(-b, -a) (marker "S2: maximum");
  - drop the lambda term from the Jacobi angle (marker "S3: stationarity");
  - flip the sign of the Boys-code angle in S4 (marker "S4: Boys").
"""

import numpy as np
import sympy as sp


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
section("S1 (symbolic): the pair gradient in the code's rotation convention")
th = sp.symbols("theta", real=True)
lam = sp.symbols("lambda", nonnegative=True)


def sym3(prefix):
    e = sp.symbols(f"{prefix}00 {prefix}01 {prefix}02 {prefix}11 {prefix}12 {prefix}22", real=True)
    return sp.Matrix([[e[0], e[1], e[2]], [e[1], e[3], e[4]], [e[2], e[4], e[5]]])


Xs = [sym3("x"), sym3("y"), sym3("z"), sym3("f")]
ws = [1, 1, 1, lam]
p, q = 0, 2  # a non-adjacent pair, orbital 1 is a bystander
R = sp.eye(3)
R[p, p], R[q, q], R[q, p], R[p, q] = sp.cos(th), sp.cos(th), sp.sin(th), -sp.sin(th)
Xr = [R.T * X * R for X in Xs]
G = sum(w * sum(Xr_[i, i] ** 2 for i in range(3)) for w, Xr_ in zip(ws, Xr))
dG0 = sp.diff(G, th).subs(th, 0)
pair_grad = sum(4 * w * (X[p, p] - X[q, q]) * X[p, q] for w, X in zip(ws, Xs))
assert sp.expand(dG0 - pair_grad) == 0, "S1: gradient FAILED"
for X, X2 in zip(Xs, Xr):
    assert sp.simplify(X2[1, 1] - X[1, 1]) == 0, "S1: bystander FAILED"
print("dG/dtheta|_0 = 4 sum_X w_X (X_pp - X_qq) X_pq ; bystander diagonal unchanged")

# ---------------------------------------------------------------------------
section("S2 (symbolic): G(theta) = const + a cos4theta + b sin4theta, maximum at atan2(b, a)/4")
dd = [(X[p, p] - X[q, q]) / 2 for X in Xs]
cc = [X[p, q] for X in Xs]
a_s = sum(w * (d * d - c * c) for w, d, c in zip(ws, dd, cc))
b_s = sum(w * 2 * d * c for w, d, c in zip(ws, dd, cc))
G0 = G.subs(th, 0)
diff_form = sp.expand_trig(G - G0 - (a_s * (sp.cos(4 * th) - 1) + b_s * sp.sin(4 * th)))
assert sp.simplify(diff_form) == 0, "S2: Jacobi form FAILED"
A, B = sp.symbols("a b", real=True)
Gp = A * sp.cos(4 * th) + B * sp.sin(4 * th)


def pair_angle(a, b):
    return sp.atan2(b, a) / 4


ths = pair_angle(A, B)
# Write the angle through r = sqrt(a^2+b^2): cos 4theta* = a/r, sin 4theta* = b/r
rr = sp.sqrt(A**2 + B**2)
c4, s4 = sp.cos(4 * ths), sp.sin(4 * ths)
for (av, bv) in [(2, 1), (-3, 2), (-1, -4), (5, -1), (sp.Rational(1, 3), 7)]:
    sub = {A: av, B: bv}
    assert sp.simplify((c4 - A / rr).subs(sub)) == 0 and sp.simplify((s4 - B / rr).subs(sub)) == 0, (
        "S2: maximum FAILED (angle)"
    )
    d1 = sp.diff(Gp, th).subs(th, ths).subs(sub)
    d2 = sp.diff(Gp, th, 2).subs(th, ths).subs(sub)
    assert sp.simplify(d1) == 0, "S2: maximum FAILED (not stationary)"
    assert sp.simplify(d2 + 16 * rr.subs(sub)) == 0, "S2: maximum FAILED (G'' != -16 r)"
print("theta* = atan2(b, a)/4: G'(theta*) = 0, G''(theta*) = -16 sqrt(a^2+b^2) <= 0")

# ---------------------------------------------------------------------------
section("S3 (symbolic): pair gradient = 4 b; stationarity and the converged-sweep condition")
assert sp.expand(pair_grad - 4 * b_s) == 0, "S3: stationarity FAILED (gradient != 4b)"
# at b = 0: theta* = 0 iff a > 0
for av in (1, sp.Rational(5, 2)):
    assert pair_angle(sp.Integer(av), sp.Integer(0)) == 0, "S3: stationarity FAILED"
assert pair_angle(sp.Integer(-1), sp.Integer(0)) == sp.pi / 4, "S3: saddle branch FAILED"
print(
    "pair gradient = 4b; stationary <=> b_pq = 0 for all p<q; theta* = 0 there iff a_pq > 0"
)

# ---------------------------------------------------------------------------
section("S4 (symbolic): at lambda = 0 the step is the Boys code's step")
# boys.rs: A_b = sum[(q_ii-q_jj)^2/4 - q_ij^2], B_b = sum q_ij (q_ii - q_jj),
# theta_b = atan2(-B_b, A_b)/4, rotation x' = c x - s y on (i, j).
a0 = a_s.subs(lam, 0)
b0 = b_s.subs(lam, 0)
Ab = sum((X[p, p] - X[q, q]) ** 2 / 4 - X[p, q] ** 2 for X in Xs[:3])
Bb = sum(X[p, q] * (X[p, p] - X[q, q]) for X in Xs[:3])
assert sp.expand(a0 - Ab) == 0 and sp.expand(b0 - Bb) == 0, "S4: Boys FAILED (a, b)"


def boys_theta(Ab_, Bb_):
    return sp.atan2(-Bb_, Ab_) / 4


# the Boys rotation by theta_b is the code's rotation by -theta_b
for (av, bv) in [(2, 1), (-3, 2), (-1, -4), (5, -1)]:
    tb = boys_theta(sp.Integer(av), sp.Integer(bv))
    tl = pair_angle(sp.Integer(av), sp.Integer(bv))
    assert sp.simplify(tl + tb) == 0, "S4: Boys FAILED (angle)"
print("lambda = 0: a = A_Boys, b = B_Boys, and the code's angle = -theta_Boys (same rotation)")

# ---------------------------------------------------------------------------
section("Numerical cross-check: FD via the matrix exponential, Jacobi optimum, Hessian sign")


def G_num(Xm, Fm, lam_v):
    return sum((np.diag(X) ** 2).sum() for X in Xm) + lam_v * (np.diag(Fm) ** 2).sum()


def grad_mat(Xm, Fm, lam_v):
    """dG/dtheta_pq for p<q (code convention), as an antisymmetric-indexed array."""
    n = Fm.shape[0]
    g = np.zeros((n, n))
    for X, w in [(X, 1.0) for X in Xm] + [(Fm, lam_v)]:
        d = np.diag(X)
        g += 4 * w * (d[:, None] - d[None, :]) * X
    return g


def angle(Xm, Fm, lam_v, p_, q_):
    a = b = 0.0
    for X, w in list(zip(Xm, (1.0, 1.0, 1.0))) + [(Fm, lam_v)]:
        d, c = (X[p_, p_] - X[q_, q_]) / 2, X[p_, q_]
        a += w * (d * d - c * c)
        b += w * 2 * d * c
    return np.arctan2(b, a) / 4


def rot(n, p_, q_, t):
    Rm = np.eye(n)
    Rm[p_, p_] = Rm[q_, q_] = np.cos(t)
    Rm[q_, p_], Rm[p_, q_] = np.sin(t), -np.sin(t)
    return Rm


def jacobi(Xm, Fm, lam_v, tol=1e-11, max_sweep=2000):
    """Returns the rotated matrices, U, and whether G ever fell in a step."""
    Xm, Fm = [X.copy() for X in Xm], Fm.copy()
    n = Fm.shape[0]
    U = np.eye(n)
    fell = False
    g_prev = G_num(Xm, Fm, lam_v)
    for _ in range(max_sweep):
        if np.abs(grad_mat(Xm, Fm, lam_v)).max() < tol:
            break
        for p_ in range(n):
            for q_ in range(p_ + 1, n):
                t = angle(Xm, Fm, lam_v, p_, q_)
                Rm = rot(n, p_, q_, t)
                Xm = [Rm.T @ X @ Rm for X in Xm]
                Fm = Rm.T @ Fm @ Rm
                U = U @ Rm
                g_new = G_num(Xm, Fm, lam_v)
                fell |= g_new < g_prev - 1e-10
                g_prev = g_new
    return Xm, Fm, U, fell


rng = np.random.default_rng(11)
n = 6


def symm(scale):
    m = rng.normal(size=(n, n)) * scale
    return (m + m.T) / 2


Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
cent = rng.normal(size=(n, 3)) * 2.0
X0 = [Q.T @ (np.diag(cent[:, k]) + symm(0.3)) @ Q for k in range(3)]
F0 = Q.T @ (np.diag(np.sort(rng.uniform(-1.2, -0.4, n))) + symm(0.1)) @ Q
lam_v = 10.0


def expm_antisym(K):
    w, V = np.linalg.eigh(1j * K)
    return (V @ np.diag(np.exp(-1j * w)) @ V.conj().T).real


def G_along(K, s, Xm, Fm):
    U = expm_antisym(s * K)
    return G_num([U.T @ X @ U for X in Xm], U.T @ Fm @ U, lam_v)


# FD of the gradient at a non-stationary point (independent of the pair formula)
gm = grad_mat(X0, F0, lam_v)
for _ in range(3):
    K = rng.normal(size=(n, n))
    K = np.triu(K, 1) - np.triu(K, 1).T
    # exp(sK) = 1 + sK + ...; the code's R(theta) has R[q,p] = +theta, R[p,q] = -theta
    # to first order, so the direction K moves theta_pq by s*K[q,p].
    h = 1e-5
    fd = (G_along(K, h, X0, F0) - G_along(K, -h, X0, F0)) / (2 * h)
    an = sum(gm[pp, qq] * K[qq, pp] for pp in range(n) for qq in range(pp + 1, n))
    assert abs(fd - an) < 1e-6 * max(1.0, abs(an)), f"S1: gradient FAILED (FD {fd} vs {an})"
print("analytic pair gradient == FD of G along exp(sK), 3 random directions")

Xc, Fc, Uc, fell = jacobi(X0, F0, lam_v)
gmax = np.abs(grad_mat(Xc, Fc, lam_v)).max()
assert gmax < 1e-10, f"S3: stationarity FAILED (max |grad| {gmax:.2e} at the Jacobi optimum)"
assert not fell, "S2: maximum FAILED (G fell in a Jacobi step)"
# the lambda-dependent stationarity condition holds; the lambda = 0 one does not
d0 = np.abs(grad_mat(Xc, Fc, 0.0)).max()
assert d0 > 1e-6, "S3: stationarity FAILED (the F term is inert on this model)"
# FD second derivatives along random antisymmetric directions: <= 0 (a maximum)
for _ in range(10):
    K = rng.normal(size=(n, n))
    K = np.triu(K, 1) - np.triu(K, 1).T
    h = 1e-3
    d2 = (G_along(K, h, Xc, Fc) - 2 * G_num(Xc, Fc, lam_v) + G_along(K, -h, Xc, Fc)) / h**2
    assert d2 < 1e-6, f"S2: maximum FAILED (FD curvature {d2:.3e} > 0 at the optimum)"
print(
    f"Jacobi optimum: max |grad| {gmax:.1e}, lambda=0 gradient there {d0:.1e} (F term live); "
    "curvature <= 0 along 10 random directions"
)
off = lambda M: np.sqrt((M**2).sum() - (np.diag(M) ** 2).sum())  # noqa: E731
_, F_boys, _, _ = jacobi(X0, F0, 0.0)
print(f"||F_offdiag||: Boys {off(F_boys):.4f}, lambda-Boys(10) {off(Fc):.4f}")
print("ALL CHECKS PASSED")
