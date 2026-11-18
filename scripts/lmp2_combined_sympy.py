#!/usr/bin/env python3
"""Sympy proofs: the combined local-MP2 truncation (Fock screen + WLLH25 embedding + lambda-Boys).

Conventions (crates/ferric-mp2/theory.md sec. 12.1, 12.6 and 12.7): closed-shell
LMP2 in ORTHONORMAL localized orbitals (Boys / lambda-Boys occupieds, VV-HV
virtuals), so the amplitude-space metric is the Euclidean one. Amplitudes
t[i,j,a,b] over ordered pairs; the Fock superoperator
    (A t)_ij = F t_ij + t_ij F - sum_k (f_ik t_kj + f_kj t_ik)
is SPD when eps_LUMO > eps_HOMO; J[i,j,a,b] = (ia|jb); S swaps a<->b,
W = 2 - S (eigenvalues 1, 3) commutes with A; the Hylleraas functional is
E_H(t) = <W t, 2J + A t>, minimized at t* = -A^-1 J. A pattern K (dropped D)
is closed under S and under the pair transpose P (ij,ab)->(ji,ba).
Delta = diag(A), A' = A - Delta.

P1 (truncation error). Galerkin solve A_KK t_K = -J_K, t_D = 0, residual
R_D = J_D + A_DK t_K, Schur complement S_A = A_DD - A_DK A_KK^-1 A_KD:
    E_K - E* = R_D^T W_D S_A^-1 R_D = t*_D^T W_D S_A t*_D,
    |R_D|^2 / lmax(A) <= E_K - E* <= 3 |R_D|^2 / lmin(A),
and K1 subset K2 => E* <= E_K2 <= E_K1 (any screen that ADDS elements to
the Eq-8 pattern is a monotone improvement and keeps the upper bound).

P2 (the screen is one Jacobi step). The occupied-extended dressed integral
    g~ = g + F'_vv t0 + t0 F'_vv - sum_{k!=i} f_ik t0^{kj} - sum_{k!=j} t0^{ik} f_kj
equals J + A' t0 with t0 = -J/Delta, and exactly
    t* - (-g~/Delta) = (Delta^-1 A')^2 t*,
so |t*_e + g~_e/Delta_e| <= |Delta^-1 A'|^2 |t*|: the ESTIMATE is second
order. The virtual-only estimate misses the first-order occupied term. (This
is about the estimator; a mask at a fixed threshold still drops exact
amplitudes of first order, so its miss norm is first order for both forms.) Leading-order
error estimator: dE ~ g~_D^T W_D Delta_D^-1 g~_D, relative error O(|A'|/Delta).

P3 (WLLH25 embedding). B = A with f_oo off-diagonals removed (block diagonal
over pairs), C = A - B (inter-pair occupied coupling, |C|_2 = 2|f'_oo|_2).
tau = -B^-1 J; A_KK t_K = -J_K - B_KD tau_D; t_emb = (t_K, tau_D). Then
    rho_K = (J + A t_emb)_K = C_KD tau_D            (exactly),
    C = 0  =>  t_emb = t*                            (exactly),
    E_H(t_emb) - E* = rho^T W A^-1 rho >= 0, O(|C|^2)   (upper bound kept),
    E_L(t_emb) = <W t_emb, J> (WLLH25 Eq 41 at r = 0) - E* = O(|C|), SIGN
    INDEFINITE: the linear-energy embedding can undershoot E* (DISPROVED as
    a bound; counterexample found below). E_H(t_emb) can also exceed the
    plain E_K (counterexample below). The embedded Galerkin energy
    min over (t_K, alpha) of E_H(t_K, alpha tau_D) satisfies
    E* <= E_G <= min(E_K, E_H(t_emb)) by construction; with one alpha per
    pair it is size-consistent, with one global alpha it is not (shown).

P4 (lambda-Boys). Maximize G = sum_x sum_i (r^x_ii)^2 + lambda sum_i F_ii^2
(WLLH25 Eq 9). Gradient dG/dkappa_pq = 4 sum_X w_X X_pq (X_qq - X_pp) for
U = 1 + kappa; the 2x2 Jacobi step is G(theta) = const + a cos4theta +
b sin4theta, a = sum w (d^2 - c^2), b = sum w 2dc, d = (X_pp - X_qq)/2,
c = X_pq: lambda-Boys is Boys with a fourth "dipole" sqrt(lambda) F. The
penalty Phi = sum_{i!=j} F_ij^2 >= |f'_oo|_2^2 bounds |C|_2 <= 2 sqrt(Phi),
and Phi at the global optimum is non-increasing in lambda.

P6 (limits). eps = 0 is exact for every arm; kappa -> inf gives the Eq-8
pattern; untruncated E is invariant to occupied/virtual rotations; every arm
is size-consistent for non-interacting fragments with fragment-pure orbitals,
and a Jacobi lambda-Boys sweep never mixes non-interacting fragments.

Independent route: dense brute-force linear algebra (numpy) on random
instances with the real LMP2 block structure (3 occupied, 4 virtual, ordered
pairs, shared virtuals, nonzero f_oo and F_vv off-diagonals, J = B^T B from
an RI factor with log-uniform magnitudes so masks split it).

Every check is an assert, so a failure exits nonzero with an AssertionError.
Mutations (each turns the script red; recorded in
crates/ferric-python/tests/test_derivation_scripts.py):
  - wrong Schur block: A_DD in place of the Schur complement
    (marker "P1: residual identity");
  - drop the occupied coupling from the explicit dressed integral
    (marker "P2: Jacobi");
  - flip the sign of the RHS embedding m = B_KD tau_D
    (marker "P3: exact when f_oo diagonal");
  - drop the lambda term from the Jacobi rotation angle
    (marker "P4: stationarity");
  - rotate one occupied index of J by the identity instead of U_o
    (marker "P6: rotational invariance");
  - drop the factor 4 from the symbolic gradient (marker "P4: gradient").

Run: python scripts/lmp2_combined_sympy.py
"""

import numpy as np
import sympy as sp


def section(title):
    print(f"\n=== {title}")


# ===========================================================================
# Symbolic proofs
# ===========================================================================
section("P1 (symbolic): E_K - E* = rho_D^T S^-1 rho_D = t*_D^T S t*_D, 2+1 blocks")
nk, n = 2, 3
it = iter(sp.symbols(f"m0:{n * (n + 1) // 2}", real=True))
M = sp.zeros(n, n)
for r in range(n):
    for c in range(r, n):
        M[r, c] = M[c, r] = next(it)
b = sp.Matrix(sp.symbols(f"b0:{n}", real=True))
E = lambda t: (2 * t.T * b + t.T * M * t)[0, 0]  # noqa: E731
M_KK, M_KD, M_DK, M_DD = M[:nk, :nk], M[:nk, nk:], M[nk:, :nk], M[nk:, nk:]
t_star = -M.inv() * b
t_K = -M_KK.inv() * b[:nk, 0]
t_gal = sp.Matrix.vstack(t_K, sp.zeros(n - nk, 1))
rho_D = b[nk:, 0] + M_DK * t_K
S_M = M_DD - M_DK * M_KK.inv() * M_KD
lhs = E(t_gal) - E(t_star)
assert sp.simplify(sp.together(lhs - (rho_D.T * S_M.inv() * rho_D)[0, 0])) == 0, (
    "P1: residual identity FAILED (symbolic)"
)
assert sp.simplify(t_star[nk:, 0] + S_M.inv() * rho_D) == sp.zeros(1, 1), (
    "P1: t*_D = -S^-1 rho_D FAILED"
)
print("E_K - E* == rho_D^T S^-1 rho_D and t*_D == -S^-1 rho_D   (generic symmetric M)")
# quadratic-form identity used by P3: E(t) - E* = (t - t*)^T M (t - t*)
tv = sp.Matrix(sp.symbols(f"x0:{n}", real=True))
assert (
    sp.simplify(
        sp.together(E(tv) - E(t_star) - ((tv - t_star).T * M * (tv - t_star))[0, 0])
    )
    == 0
)
print("E(t) - E* == (t - t*)^T M (t - t*)                       (generic symmetric M)")

section("P2 (symbolic): t* + g~/Delta = (Delta^-1 A')^2 t*, exactly")
m3 = 3
Dl = sp.diag(*sp.symbols(f"d0:{m3}", positive=True))
it = iter(sp.symbols(f"p0:{m3 * (m3 - 1) // 2}", real=True))
Ap = sp.zeros(m3, m3)
for r in range(m3):
    for c in range(r + 1, m3):
        Ap[r, c] = Ap[c, r] = next(it)
Jm = sp.Matrix(sp.symbols(f"k0:{m3}", real=True))
ts = -(Dl + Ap).inv() * Jm
t0 = -Dl.inv() * Jm
gt = Jm + Ap * t0
rem = (Dl.inv() * Ap) ** 2 * ts
assert sp.simplify(ts + Dl.inv() * gt - rem) == sp.zeros(m3, 1), (
    "P2: remainder identity FAILED"
)
print("t* - t_est == (Delta^-1 A')^2 t*   (generic 3x3)")
# what the screen misses: a two-step path (chain 0-1-2, J only on 0)
lam = sp.symbols("lambda", real=True)
Ach = sp.Matrix([[0, lam, 0], [lam, 0, lam], [0, lam, 0]])
Jc = sp.Matrix([1, 0, 0])
tsc = -(Dl + Ach).inv() * Jc
gtc = Jc + Ach * (-Dl.inv() * Jc)
assert gtc[2] == 0 and sp.series(tsc[2], lam, 0, 3).removeO() != 0, "P2: two-step path"
print("chain 0-1-2: g~_2 = 0 but t*_2 = O(lambda^2) != 0 (second-order path is missed)")

section("P2 (symbolic): explicit occupied-extended g~ == J + A' t0 (no=2, nv=2)")
no_s, nv_s = 2, 2


def sym_sym(prefix, k):
    it = iter(sp.symbols(f"{prefix}0:{k * (k + 1) // 2}", real=True))
    X = sp.zeros(k, k)
    for r in range(k):
        for c in range(r, k):
            X[r, c] = X[c, r] = next(it)
    return X


fo, Fv = sym_sym("f", no_s), sym_sym("F", nv_s)
Js = sp.symbols(f"J0:{no_s**2 * nv_s**2}", real=True)
Io, Iv = sp.eye(no_s), sp.eye(nv_s)
kp = sp.kronecker_product
A_s = kp(Io, Io, Fv, Iv) + kp(Io, Io, Iv, Fv) - kp(fo, Io, Iv, Iv) - kp(Io, fo, Iv, Iv)
Ns = A_s.shape[0]
Dg = [A_s[e, e] for e in range(Ns)]
t0s = [-Js[e] / Dg[e] for e in range(Ns)]
Aoff = A_s - sp.diag(*Dg)
gt_op = sp.Matrix(Js) + Aoff * sp.Matrix(t0s)


def idx(i, j, a, b_):
    return ((i * no_s + j) * nv_s + a) * nv_s + b_


def T(i, j, a, b_):
    return t0s[idx(i, j, a, b_)]


ok = True
for i in range(no_s):
    for j in range(no_s):
        for a in range(nv_s):
            for b_ in range(nv_s):
                g = Js[idx(i, j, a, b_)]
                g += sum(Fv[a, c] * T(i, j, c, b_) for c in range(nv_s) if c != a)
                g += sum(T(i, j, a, c) * Fv[c, b_] for c in range(nv_s) if c != b_)
                g -= sum(fo[i, k] * T(k, j, a, b_) for k in range(no_s) if k != i)
                g -= sum(T(i, k, a, b_) * fo[k, j] for k in range(no_s) if k != j)
                ok &= sp.simplify(g - gt_op[idx(i, j, a, b_)]) == 0
assert ok, "P2: Jacobi identity FAILED (explicit g~ != J + A' t0)"
print("g + F' t0 + t0 F' - sum_{k!=i} f_ik t0^kj - sum_{k!=j} t0^ik f_kj == J + A' t0")

section("P3 (symbolic): embedding algebra for a generic split A = B + C (2+1)")
Bm = sym_sym("B", 3)
Cm = sym_sym("C", 3)
Jv3 = sp.Matrix(sp.symbols("j0:3", real=True))
tau = sp.Matrix(
    sp.symbols("u0:3", real=True)
)  # any tau: rho_K identity is pure algebra
Am = Bm + Cm
tKe = -Am[:2, :2].inv() * (Jv3[:2, 0] + Bm[:2, 2:] * tau[2:, 0])
temb = sp.Matrix.vstack(tKe, tau[2:, 0])
rho = Jv3 + Am * temb
assert sp.simplify(rho[:2, 0] - Cm[:2, 2:] * tau[2:, 0]) == sp.zeros(2, 1), "P3: rho_K"
print("rho_K == C_KD tau_D   (generic B, C, tau)")
tau_b = -Bm.inv() * Jv3
tKb = -Bm[:2, :2].inv() * (Jv3[:2, 0] + Bm[:2, 2:] * tau_b[2:, 0])
temb_b = sp.Matrix.vstack(tKb, tau_b[2:, 0])
assert sp.simplify(temb_b + Bm.inv() * Jv3) == sp.zeros(3, 1), "P3: exact at C=0"
print("C = 0 => t_emb == t* = -B^-1 J   (generic B)")

section("P4 (symbolic): lambda-Boys gradient and 2x2 Jacobi angle")
X3 = sym_sym("X", 3)
k01, k02, k12 = sp.symbols("k01 k02 k12", real=True)
K3 = sp.Matrix([[0, k01, k02], [-k01, 0, k12], [-k02, -k12, 0]])
Xr = (sp.eye(3) + K3).T * X3 * (sp.eye(3) + K3)
Gx = sum(Xr[i, i] ** 2 for i in range(3))
for (p, q), kv in (((0, 1), k01), ((0, 2), k02), ((1, 2), k12)):
    d = sp.diff(Gx, kv).subs({k01: 0, k02: 0, k12: 0})
    assert sp.expand(d - 4 * X3[p, q] * (X3[q, q] - X3[p, p])) == 0, (
        "P4: gradient FAILED"
    )
print("d/dkappa_pq sum_i X_ii^2 = 4 X_pq (X_qq - X_pp)")
th, xa, xb, xc = sp.symbols("theta a b c", real=True)
cth, sth = sp.cos(th), sp.sin(th)
Xpp = cth**2 * xa + sth**2 * xb + 2 * cth * sth * xc
Xqq = sth**2 * xa + cth**2 * xb - 2 * cth * sth * xc
dd, cc = (xa - xb) / 2, xc
target = (
    (xa + xb) ** 2 / 2
    + dd**2
    + cc**2
    + (dd**2 - cc**2) * sp.cos(4 * th)
    + 2 * dd * cc * sp.sin(4 * th)
)
assert sp.simplify(sp.expand_trig(Xpp**2 + Xqq**2 - target)) == 0, (
    "P4: Jacobi form FAILED"
)
print("X'_pp^2 + X'_qq^2 = const + (d^2-c^2) cos4theta + 2dc sin4theta")
B1, B2, P1s, P2s, l1, l2 = sp.symbols("B1 B2 Phi1 Phi2 l1 l2", real=True)
assert (
    sp.expand(
        (B2 + l1 * P2s - B1 - l1 * P1s)
        + (B1 + l2 * P1s - B2 - l2 * P2s)
        - (l2 - l1) * (P1s - P2s)
    )
    == 0
)
print(
    "optimality at l1 and l2 sums to (l2-l1)(Phi1-Phi2) >= 0: Phi non-increasing in lambda"
)

# ===========================================================================
# Numerical cross-checks on the LMP2 block structure
# ===========================================================================


class Model:
    """Dense LMP2 model; amplitudes t[i,j,a,b] flattened in C order."""

    def __init__(self, f, F, J):
        self.f, self.F, self.J4 = f, F, J
        no, nv = f.shape[0], F.shape[0]
        self.no, self.nv = no, nv
        Io, Iv = np.eye(no), np.eye(nv)
        k = lambda *m: m[0] if len(m) == 1 else np.kron(m[0], k(*m[1:]))  # noqa: E731
        self.Av = k(Io, Io, F, Iv) + k(Io, Io, Iv, F)
        self.Ao = -k(f, Io, Iv, Iv) - k(Io, f, Iv, Iv)
        self.A = self.Av + self.Ao
        fd = np.diag(np.diag(f))
        self.B = self.Av - k(fd, Io, Iv, Iv) - k(Io, fd, Iv, Iv)  # f_oo diagonal
        self.C = self.A - self.B
        self.J = J.reshape(-1)
        self.N = self.J.size
        ii = np.arange(self.N).reshape(no, no, nv, nv)
        self.Sperm = ii.transpose(0, 1, 3, 2).reshape(-1)
        self.Pperm = ii.transpose(1, 0, 3, 2).reshape(-1)
        self.W = 2 * np.eye(self.N) - np.eye(self.N)[self.Sperm]
        self.Dg = np.diag(self.A).copy()
        self.tstar = -np.linalg.solve(self.A, self.J)
        self.Estar = self.EH(self.tstar)

    def EH(self, t):
        return (self.W @ t) @ (2 * self.J + self.A @ t)

    def EL(self, t):
        return (self.W @ t) @ self.J

    def eq8(self, eps):
        return (np.abs(self.J) > eps) | (np.abs(self.J[self.Sperm]) > eps)

    def galerkin(self, K):
        t = np.zeros(self.N)
        t[K] = -np.linalg.solve(self.A[np.ix_(K, K)], self.J[K])
        return t

    def gtilde(self, occ=True):
        t0 = -self.J / self.Dg
        Ap = (self.A if occ else self.Av + np.diag(np.diag(self.Ao))) - np.diag(self.Dg)
        return self.J + Ap @ t0

    def screen(self, eps, kappa, occ=True):
        g = self.gtilde(occ)
        return (
            self.eq8(eps)
            | (np.abs(g) > kappa * eps)
            | (np.abs(g[self.Sperm]) > kappa * eps)
        )

    def embed(self, K):
        D = ~K
        tau = -np.linalg.solve(self.B, self.J)
        rhs = self.J[K] + self.B[np.ix_(K, D)] @ tau[D]
        t = np.zeros(self.N)
        t[K] = -np.linalg.solve(self.A[np.ix_(K, K)], rhs)
        t[D] = tau[D]
        return t, tau

    def embed_galerkin(self, K):
        """Galerkin on K plus one contracted vector tau_D^(ij) per ordered pair.

        min E_H over t = (t_K, alpha_ij tau_D^(ij)) contains alpha = 0 (plain
        E_K) and, for alpha = 1, the minimum over t_K at fixed tau_D, which is
        <= E_H(t_emb): so E* <= E_G <= min(E_K, E_H(t_emb)). One alpha per
        pair keeps size-consistency; ONE global alpha does not (P6).
        """
        D = ~K
        tau = -np.linalg.solve(self.B, self.J)
        cols = [np.eye(self.N)[:, K]]
        npair = self.no * self.no
        blk = self.N // npair
        for pr in range(npair):
            v = np.zeros(self.N)
            sl = slice(pr * blk, (pr + 1) * blk)
            v[sl] = np.where(D[sl], tau[sl], 0.0)
            if np.linalg.norm(v) > 0:
                cols.append(v[:, None])
        V = np.hstack(cols)
        WA = self.W @ self.A
        y = -np.linalg.solve(V.T @ WA @ V, V.T @ (self.W @ self.J))
        return V @ y


def sym_off(rng, k, s):
    x = rng.normal(size=(k, k)) * s
    x = (x + x.T) / 2
    np.fill_diagonal(x, 0.0)
    return x


def make(seed, no=3, nv=4, s_o=0.08, s_v=0.15, naux=8):
    rng = np.random.default_rng(seed)
    f = np.diag(np.sort(rng.uniform(-1.2, -0.5, no))) + sym_off(rng, no, s_o)
    F = np.diag(np.sort(rng.uniform(0.3, 2.0, nv))) + sym_off(rng, nv, s_v)
    Bf = rng.normal(size=(naux, no, nv)) * 10 ** rng.uniform(-3, 0, size=(naux, no, nv))
    J = np.einsum("Pia,Pjb->ijab", Bf, Bf)
    assert np.linalg.eigvalsh(F).min() > np.linalg.eigvalsh(f).max(), (
        "model needs a gap"
    )
    return Model(f, F, J)


def relerr(x, y):
    return abs(x - y) / max(abs(y), 1e-300)


SEEDS = range(6)
EPS = (1e-2, 1e-3, 1e-4)

section("P1 (numeric): error identity, two-sided bound, nesting")
for seed in SEEDS:
    md = make(seed)
    A, W = md.A, md.W
    assert np.allclose(md.Sperm[md.Sperm], np.arange(md.N))
    assert np.allclose(A @ W, W @ A), "W must commute with A"
    lmin, lmax = np.linalg.eigvalsh(A)[[0, -1]]
    for eps in EPS:
        K = md.eq8(eps)
        D = ~K
        assert np.array_equal(K[md.Sperm], K) and np.array_equal(K[md.Pperm], K)
        if D.sum() == 0:
            continue
        tK = md.galerkin(K)
        err = md.EH(tK) - md.Estar
        R_D = (md.J + A @ tK)[D]
        A_DD = A[np.ix_(D, D)]
        schur_a = A_DD - A[np.ix_(D, K)] @ np.linalg.solve(
            A[np.ix_(K, K)], A[np.ix_(K, D)]
        )
        WD = W[np.ix_(D, D)]
        f_res = R_D @ WD @ np.linalg.solve(schur_a, R_D)
        f_amp = md.tstar[D] @ WD @ schur_a @ md.tstar[D]
        tol = 1e-6 * err + 1e-13 * abs(md.Estar)  # err is a difference of energies
        assert abs(f_res - err) < tol, f"P1: residual identity FAILED {f_res} vs {err}"
        assert abs(f_amp - err) < tol, f"P1: amplitude identity FAILED {f_amp} vs {err}"
        nr2 = R_D @ R_D
        assert nr2 / lmax * (1 - 1e-12) <= err <= 3 * nr2 / lmin * (1 + 1e-12), (
            "P1: bound"
        )
        assert err >= 0, "P1: upper bound"
        Ks = md.screen(eps, 1.0)
        assert np.all(Ks >= K)
        es = md.EH(md.galerkin(Ks)) - md.Estar
        assert -1e-14 <= es <= err + 1e-14, "P1: nesting (screen must not lose)"
print(
    f"{len(SEEDS)} seeds x {len(EPS)} eps: identity to cancellation, bounds hold, screen nested & no worse"
)

section("P2 (numeric): one-step estimator error, second order, estimator -> exact")
for seed in SEEDS:
    md = make(seed)
    Ap = md.A - np.diag(md.Dg)
    test = -md.gtilde() / md.Dg
    rem = md.tstar - test
    G = Ap / md.Dg[:, None]
    assert np.allclose(rem, G @ G @ md.tstar, atol=1e-13), (
        "P2: remainder identity (numeric)"
    )
    bound = np.linalg.norm(G, 2) ** 2 * np.linalg.norm(md.tstar)
    assert np.abs(rem).max() <= bound * (1 + 1e-12), "P2: elementwise bound"
    # explicit tensor formula with occupied coupling, n-dim
    t0 = (-md.J / md.Dg).reshape(md.no, md.no, md.nv, md.nv)
    Fp = md.F - np.diag(np.diag(md.F))
    fp = md.f - np.diag(np.diag(md.f))
    ge = (
        md.J4
        + np.einsum("ac,ijcb->ijab", Fp, t0)
        + np.einsum("ijac,cb->ijab", t0, Fp)
        - np.einsum("ik,kjab->ijab", fp, t0)
        - np.einsum("ikab,kj->ijab", t0, fp)
    )
    assert np.allclose(ge.reshape(-1), md.gtilde(), atol=1e-14), "P2: Jacobi (numeric)"
    g = md.gtilde()
    assert np.allclose(g[md.Pperm], g, atol=1e-14), "P2: P covariance"
print(
    "t* + g~/Delta == (Delta^-1 A')^2 t*; |.|_max <= |Delta^-1 A'|^2 |t*|; tensor form == operator"
)
# order in the coupling of the ESTIMATOR error: occupied-extended 2nd order,
# virtual-only 1st order in f'_oo
base = make(11)
r_full, r_virt, ratio_est = [], [], []
for s in (0.2, 0.1, 0.05):
    f = np.diag(np.diag(base.f)) + s * (base.f - np.diag(np.diag(base.f)))
    F = np.diag(np.diag(base.F)) + s * (base.F - np.diag(np.diag(base.F)))
    md = Model(f, F, base.J4)
    r_full.append(np.abs(md.tstar + md.gtilde(True) / md.Dg).max())
    r_virt.append(np.abs(md.tstar + md.gtilde(False) / md.Dg).max())
    K = md.eq8(1e-3)
    D = ~K
    err = md.EH(md.galerkin(K)) - md.Estar
    gD = md.gtilde()[D]
    est = gD @ md.W[np.ix_(D, D)] @ (gD / md.Dg[D])
    ratio_est.append(abs(est / err - 1))
q_full, q_virt = r_full[0] / r_full[1], r_virt[0] / r_virt[1]
assert 3.5 < q_full < 4.5 and 1.7 < q_virt < 2.3, f"P2: order FAILED {q_full} {q_virt}"
assert (
    ratio_est[2] < ratio_est[1] < ratio_est[0] and ratio_est[0] / ratio_est[1] > 1.6
), f"P2: estimator must converge linearly {ratio_est}"
print(
    f"halving coupling: occupied-extended estimator error /{q_full:.2f}, virtual-only /{q_virt:.2f}"
)
print(f"|dE_est/dE - 1| at s=0.2,0.1,0.05: {', '.join(f'{x:.2e}' for x in ratio_est)}")

section("P3 (numeric): WLLH25 embedding")
b0 = make(3)
md0 = Model(np.diag(np.diag(b0.f)), b0.F, b0.J4)
te0, _ = md0.embed(md0.eq8(1e-3))
assert abs(md0.EH(te0) - md0.Estar) < 1e-13 and abs(md0.EL(te0) - md0.Estar) < 1e-13, (
    "P3: exact when f_oo diagonal FAILED"
)
print("f_oo diagonal (C = 0): E_H(t_emb) = E_L(t_emb) = E* to 1e-13")
under, over, h_worse = 0, 0, 0
for seed in range(20):
    for sign in (1, -1):
        b0 = make(seed)
        f = np.diag(np.diag(b0.f)) + sign * (b0.f - np.diag(np.diag(b0.f)))
        md = Model(f, b0.F, b0.J4)
        assert (
            abs(
                np.linalg.norm(md.C, 2) - 2 * np.linalg.norm(f - np.diag(np.diag(f)), 2)
            )
            < 1e-12
        )
        K = md.eq8(1e-3)
        D = ~K
        if D.sum() == 0:
            continue
        te, tau = md.embed(K)
        rho = md.J + md.A @ te
        assert np.allclose(rho[K], md.C[np.ix_(K, D)] @ tau[D], atol=1e-13), (
            "P3: rho_K (numeric)"
        )
        eH = md.EH(te) - md.Estar
        eL = md.EL(te) - md.Estar
        assert eH >= -1e-13 * abs(md.Estar), (
            "P3: Hylleraas embedding must stay an upper bound"
        )
        qf = rho @ md.W @ np.linalg.solve(md.A, rho)
        assert abs(eH - qf) < 1e-6 * qf + 1e-13 * abs(md.Estar), (
            f"P3: E_H identity {eH} {qf}"
        )
        lmin = np.linalg.eigvalsh(md.A)[0]
        assert eH <= 3 * rho @ rho / lmin * (1 + 1e-12), "P3: certificate"
        under += eL < 0
        over += eL > 0
        eK = md.EH(md.galerkin(K)) - md.Estar
        h_worse += eH > eK
        eG = md.EH(md.embed_galerkin(K)) - md.Estar
        tl = 1e-13 * abs(md.Estar)
        assert -tl <= eG <= min(eK, eH) + tl, (
            f"P3: embedded Galerkin bound {eG} {eK} {eH}"
        )
assert under > 0 and over > 0, (
    f"P3: E_L sign must be indefinite (under={under} over={over})"
)
assert h_worse > 0, "P3: expected a counterexample to 'embedding never hurts'"
# a larger instance where each pair drops many elements (the contraction is not trivially exact)
for seed in range(2):
    md = make(seed, no=3, nv=6, s_o=0.15, s_v=0.2)
    K = md.eq8(1e-2)
    assert (~K).reshape(md.no**2, -1).sum(1).max() >= 10
    eK = md.EH(md.galerkin(K)) - md.Estar
    eH = md.EH(md.embed(K)[0]) - md.Estar
    eG = md.EH(md.embed_galerkin(K)) - md.Estar
    assert 1e-8 < eG <= min(eK, eH), (
        f"P3: embedded Galerkin bound (nv=6) {eG} {eK} {eH}"
    )
    print(
        f"nv=6 seed {seed}: plain {eK:.2e}  E_H(t_emb) {eH:.2e}  embedded Galerkin {eG:.2e}"
    )
print(
    f"E_H(t_emb) >= E* in all; E_L(t_emb) - E*: {under} below, {over} above (indefinite)"
)
print(
    f"E_H(t_emb) worse than the plain Galerkin E_K in {h_worse} of {under + over} instances"
)
print(
    "embedded Galerkin (alpha_ij tau_D^ij + re-solved t_K): E* <= E_G <= min(E_K, E_H(t_emb)) in all"
)
# orders in |C|
b0 = make(3)
eh, el = [], []
for mu in (0.4, 0.2, 0.1):
    f = np.diag(np.diag(b0.f)) + mu * (b0.f - np.diag(np.diag(b0.f)))
    md = Model(f, b0.F, b0.J4)
    te, _ = md.embed(md.eq8(1e-3))
    eh.append(md.EH(te) - md.Estar)
    el.append(abs(md.EL(te) - md.Estar))
qh, ql = eh[1] / eh[2], el[1] / el[2]
assert 3.4 < qh < 4.6 and 1.7 < ql < 2.3, f"P3: orders in C FAILED {qh} {ql}"
print(
    f"halving |C|: E_H error /{qh:.2f} (second order), |E_L error| /{ql:.2f} (first order)"
)

section("P4 (numeric): Jacobi lambda-Boys, stationarity, monotonicity, fragment purity")


def lb_G(R, F, lam):
    return sum((np.diag(X) ** 2).sum() for X in R) + lam * (np.diag(F) ** 2).sum()


def lb_grad(R, F, lam):
    g = np.zeros_like(F)
    for X, w in [(X, 1.0) for X in R] + [(F, lam)]:
        d = np.diag(X)
        g += 4 * w * X * (d[None, :] - d[:, None])
    return g


def lb_jacobi(R, F, lam, tol=1e-12, max_sweep=500):
    R, F = [X.copy() for X in R], F.copy()
    n = F.shape[0]
    U = np.eye(n)
    G_prev = lb_G(R, F, lam)
    for _ in range(max_sweep):
        for p in range(n):
            for q in range(p + 1, n):
                aa = bb = 0.0
                for X, w in [(X, 1.0) for X in R] + [(F, lam)]:
                    d, c = (X[p, p] - X[q, q]) / 2, X[p, q]
                    aa += w * (d * d - c * c)
                    bb += w * 2 * d * c
                theta = np.arctan2(bb, aa) / 4
                cs, sn = np.cos(theta), np.sin(theta)
                rot = np.eye(n)
                rot[p, p], rot[q, p], rot[p, q], rot[q, q] = cs, sn, -sn, cs
                R = [rot.T @ X @ rot for X in R]
                F = rot.T @ F @ rot
                U = U @ rot
                G_new = lb_G(R, F, lam)
                # Each step re-forms 1 + 3 rotated n x n matrices and re-sums
                # their squared diagonals, so G carries a rounding error of a
                # few n * eps * G per evaluation; a true decrease from a wrong
                # angle is O(|grad|^2), far above that until convergence.
                # 1e3 * eps * G (~2e-11 here) clears the rounding on every
                # BLAS (the fixed 1e-12 failed on macOS arm64) and stays far
                # below a real decrease.
                slack = 1e3 * np.finfo(float).eps * max(1.0, abs(G_prev))
                assert G_new >= G_prev - slack, (
                    "P4: stationarity FAILED (a Jacobi step decreased G by "
                    f"{G_prev - G_new:.2e}, slack {slack:.2e})"
                )
                G_prev = G_new
        if np.abs(lb_grad(R, F, lam)).max() < tol:
            break
    return R, F, U


rng = np.random.default_rng(7)
n_orb = 5
Q, _ = np.linalg.qr(rng.normal(size=(n_orb, n_orb)))
cent = rng.normal(size=(n_orb, 3)) * 2.0
R0 = [Q.T @ (np.diag(cent[:, x]) + sym_off(rng, n_orb, 0.3)) @ Q for x in range(3)]
F0 = (
    Q.T
    @ (np.diag(np.sort(rng.uniform(-1.2, -0.4, n_orb))) + sym_off(rng, n_orb, 0.1))
    @ Q
)
lam = 10.0
Rc, Fc, Uc = lb_jacobi(R0, F0, lam)
gmax = np.abs(lb_grad(Rc, Fc, lam)).max()
assert gmax < 1e-9, f"P4: stationarity FAILED (max |grad| {gmax:.2e})"
# finite-difference check of the gradient at a non-stationary point
K = sym_off(rng, n_orb, 1.0)
K = np.triu(K, 1) - np.triu(K, 1).T
h = 1e-6


def rotG(s):
    w, V = np.linalg.eigh(1j * K)  # expm of antisymmetric s*K via Hermitian 1j*K
    U = (V @ np.diag(np.exp(-1j * s * w)) @ V.conj().T).real
    return lb_G([U.T @ X @ U for X in R0], U.T @ F0 @ U, lam)


fd = (rotG(h) - rotG(-h)) / (2 * h)
an = sum(
    lb_grad(R0, F0, lam)[p, q] * K[p, q]
    for p in range(n_orb)
    for q in range(p + 1, n_orb)
)
assert relerr(an, fd) < 1e-6, f"P4: gradient FD FAILED {an} vs {fd}"
print(
    f"Jacobi converged, max |grad| = {gmax:.1e}; analytic gradient == FD ({relerr(an, fd):.1e})"
)
# penalty vs lambda (continuation from the previous optimum; global optimum not guaranteed)
phis = []
Rw, Fw = R0, F0
for lam_v in (0.0, 1.0, 10.0, 100.0):
    Rw, Fw, _ = lb_jacobi(Rw, Fw, lam_v)
    phis.append((Fw**2).sum() - (np.diag(Fw) ** 2).sum())
print(
    "Phi = sum_{i!=j} F_ij^2 at lambda = 0, 1, 10, 100: "
    + ", ".join(f"{p:.3e}" for p in phis)
)
assert all(phis[k + 1] <= phis[k] + 1e-12 for k in range(3)), (
    "P4: Phi monotone (this instance)"
)
# fragment purity: two well separated fragments are never mixed
nA, nB = 3, 2
RA = [np.diag(rng.normal(size=nA)) + sym_off(rng, nA, 0.3) for _ in range(3)]
RB = [np.diag(rng.normal(size=nB) + 50.0) + sym_off(rng, nB, 0.3) for _ in range(3)]
FA = np.diag(rng.uniform(-1, -0.5, nA)) + sym_off(rng, nA, 0.1)
FB = np.diag(rng.uniform(-1, -0.5, nB)) + sym_off(rng, nB, 0.1)


def bd(X, Y):
    z = np.zeros((X.shape[0], Y.shape[0]))
    return np.block([[X, z], [z.T, Y]])


_, _, Uab = lb_jacobi([bd(x, y) for x, y in zip(RA, RB)], bd(FA, FB), lam)
assert np.abs(Uab[:nA, nA:]).max() == 0.0 and np.abs(Uab[nA:, :nA]).max() == 0.0, (
    "P4: purity"
)
_, _, Ua = lb_jacobi(RA, FA, lam)
assert np.allclose(np.abs(Uab[:nA, :nA]), np.abs(Ua), atol=1e-8), (
    "P4: fragment == isolated"
)
print(
    "separated fragments: cross-fragment rotation angle is exactly 0; block == isolated run"
)

section("P6 (numeric): limits and invariances")
md = make(2)
for occ in (True, False):
    K = md.screen(0.0, 1.0, occ)
    assert K.all()
assert abs(md.EH(md.galerkin(md.eq8(0.0))) - md.Estar) < 1e-14
te, _ = md.embed(md.eq8(0.0))
assert abs(md.EH(te) - md.Estar) < 1e-14 and abs(md.EL(te) - md.Estar) < 1e-14
print("eps = 0: plain, screen (both forms) and embedding all give E* exactly")
for eps in EPS:
    assert np.array_equal(md.screen(eps, 1e300), md.eq8(eps))
print("kappa -> inf: screened pattern == Eq-8 pattern")
rng = np.random.default_rng(5)
Uo, _ = np.linalg.qr(rng.normal(size=(md.no, md.no)))
Uv, _ = np.linalg.qr(rng.normal(size=(md.nv, md.nv)))
J4r = np.einsum("ijab,ik,jl,ac,bd->klcd", md.J4, Uo, Uo, Uv, Uv)
mr = Model(Uo.T @ md.f @ Uo, Uv.T @ md.F @ Uv, J4r)
assert relerr(mr.Estar, md.Estar) < 1e-12, "P6: rotational invariance"
dK = abs(mr.EH(mr.galerkin(mr.eq8(1e-3))) - md.EH(md.galerkin(md.eq8(1e-3))))
print(
    f"rotated orbitals: E* unchanged ({relerr(mr.Estar, md.Estar):.1e}); truncated E moves {dK:.1e}"
)


def combine(mA, mB):
    noA, nvA, noB, nvB = mA.no, mA.nv, mB.no, mB.nv
    no, nv = noA + noB, nvA + nvB
    J = np.zeros((no, no, nv, nv))
    J[:noA, :noA, :nvA, :nvA] = mA.J4
    J[noA:, noA:, nvA:, nvA:] = mB.J4
    return Model(bd(mA.f, mB.f), bd(mA.F, mB.F), J)


mA = make(31, no=2, nv=3)
mB = make(81, no=2, nv=2)
EPS_G = 1e-2
mAB = combine(mA, mB)
assert relerr(mAB.Estar, mA.Estar + mB.Estar) < 1e-12, "P6: size-consistency (exact)"
for eps in (1e-2, 1e-3):
    assert (~mA.eq8(eps)).any() and (~mB.eq8(eps)).any(), "both fragments must truncate"
    for name, fn in (
        ("plain", lambda m: m.EH(m.galerkin(m.eq8(eps)))),
        ("screen", lambda m: m.EH(m.galerkin(m.screen(eps, 1.0)))),
        ("emb-H", lambda m: m.EH(m.embed(m.eq8(eps))[0])),
        ("emb-L", lambda m: m.EL(m.embed(m.eq8(eps))[0])),
        ("screen+emb", lambda m: m.EH(m.embed(m.screen(eps, 1.0))[0])),
        ("emb-Galerkin", lambda m: m.EH(m.embed_galerkin(m.screen(eps, 1.0)))),
    ):
        eab, ea, eb = fn(mAB), fn(mA), fn(mB)
        assert relerr(eab, ea + eb) < 1e-11, (
            f"P6: size-consistency FAILED ({name}, eps={eps})"
        )
print("non-interacting fragments: every arm gives E(A+B) = E(A) + E(B) to 1e-11")


def global_alpha(m, K):
    """The same embedded Galerkin with ONE alpha for the whole system."""
    D = ~K
    tau = -np.linalg.solve(m.B, m.J)
    x0, x1 = np.zeros(m.N), np.zeros(m.N)
    x0[K] = -np.linalg.solve(m.A[np.ix_(K, K)], m.J[K])
    x1[K] = -np.linalg.solve(m.A[np.ix_(K, K)], m.A[np.ix_(K, D)] @ tau[D])
    x1[D] = tau[D]
    WA = m.W @ m.A
    alpha = -((m.W @ x1) @ m.J + x1 @ WA @ x0) / (x1 @ WA @ x1)
    return m.EH(x0 + alpha * x1)


assert (~mA.eq8(EPS_G)).any() and (~mB.eq8(EPS_G)).any()
gab = global_alpha(mAB, mAB.eq8(EPS_G))
gsum = global_alpha(mA, mA.eq8(EPS_G)) + global_alpha(mB, mB.eq8(EPS_G))
assert relerr(gab, gsum) > 1e-9, (
    "P6: a single global alpha should NOT be size-consistent"
)
print(
    f"one global alpha instead: E(A+B) - E(A) - E(B) = {gab - gsum:+.2e} (not size-consistent)"
)
print("\nALL CLAIMS PASS")
