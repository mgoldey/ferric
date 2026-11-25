#!/usr/bin/env python3
"""Sympy proofs: the provable-empty pair skip of the integral-direct LMP2.

Conventions (book/src/13-local-mp2/02-integral-direct.md, section
"Proving a pair empty"; crates/ferric-mp2/src/lmp2_pair_skip.rs): the stage-5
fit of pair (i,j) on the domain D = D_i u D_j is J_ab = a^T V^-1 b with
a = B_i[D, a], b = B_j[D, b] the COMPUTED strip columns and V = V_DD the
same-kernel metric block. Split D = A + B with A = D_i, B = D_j \\ D_i, and
write Dg = blockdiag(V_AA, V_BB), alpha = Dg^-1/2 a, beta = Dg^-1/2 b,
V = Dg^1/2 (I + X) Dg^1/2, X = [[0, Y], [Y^T, 0]],
Y = V_AA^-1/2 V_AB V_BB^-1/2, gamma = ||Y||_2.

Claim 1 (block-Neumann split, exact). J = alpha^T (I + X)^-1 beta and
(I + X)^-1 = I - X + X^2 (I + X)^-1, so
J = alpha_A.beta_A + alpha_B.beta_B - (alpha_A^T Y beta_B + alpha_B^T Y^T beta_A)
    + alpha^T X^2 (I + X)^-1 beta.
Proved in symbols on a 1+1 block (scalar blocks).

Claim 2 (Schur monotonicity, exact). For SPD M and x = (x1, x2),
x^T M^-1 x - x1^T M11^-1 x1 = r^T S^-1 r >= 0, r = x2 - M21 M11^-1 x1,
S = M22 - M21 M11^-1 M12. So ||V_BB^-1/2 b_B|| <= ||V_{D_j}^-1/2 b_{D_j}||
(= q_jb, the own-domain fitted norm) for B a subset of D_j. Symbolic, 1+1.

Claim 3 (the attenuated kernels are positive definite). The Fourier
transform of erfc(w r)/r is 4 pi (1 - exp(-k^2/(4 w^2)))/k^2 > 0, so every
metric block is SPD and the fit is a projection. Symbolic.

Claim 4 (the bound). With gamma < 1, ||X|| = gamma, ||(I+X)^-1|| <= 1/(1-gamma):
|J_ab| <= uA vA + uB vB + gamma (uA vB + uB vA)
          + gamma^2/(1-gamma) sqrt(uA^2+uB^2) sqrt(vA^2+vB^2),
uA >= ||alpha_A||, uB >= ||alpha_B||, vA >= ||beta_A||, vB >= ||beta_B||.
The code's inputs: uA = q_i^max (own domain), vB = q_j^max (claim 2),
uB = ||a_B|| / sqrt(lam_j), vA = ||b_A|| / sqrt(lam_i),
gamma <= ||V_AB||_F / sqrt(lam_i lam_j), lam_k <= lambda_min(V_{D_k D_k})
<= lambda_min of any principal sub-block (Cauchy interlacing).
Independent route: J by a direct dense solve on random SPD instances,
including the adversarial one (a_B = b_A = 0, alpha_A aligned with the top
singular pair of Y), where the gamma^2 remainder is needed.

Claim 5 (the eigenvalue floor). trace(V^-1) = ||L^-1||_F^2 >= 1/lambda_min,
so lam = 1/||L^-1||_F^2 <= lambda_min. Checked on random SPD matrices.

Claim 6 (the Fock-dressed gate). With |g_cb| <= G for every element of the
pair block, D_c = f_cc + min f_vv - 2 max f_oo > 0 and
w_a = sum_{c != a} max(|F_ac|, |F_ca|)/D_c, every element of the dressed
gate g~ = g + F' t0 + t0 F' (t0 = -g/Delta) obeys |g~_ab| <= G (1 + w_a + w_b)
<= G (1 + 2 max_a w_a). Checked on random Fock matrices and pair blocks.

Every check is an assert, so a failure exits nonzero with an AssertionError.
Mutations (each turns the script red; recorded in
crates/ferric-python/tests/test_derivation_scripts.py):
  - drop the X^2 remainder from the split (marker "claim 1: split");
  - flip the sign of the Schur coupling (marker "claim 2: Schur");
  - drop the gamma^2 remainder from the bound (marker "claim 4: bound");
  - use the largest eigenvalue as the floor (marker "claim 5: floor");
  - drop the Fock weights from the dressed bound (marker "claim 6: dressed");
  - verify V + (sigma + c) I instead of V - (sigma + c) I, so a shift past
    lambda_min is accepted (marker "claim 7");
  - square the shell norms in the shell-wise bound (marker "claim 8");
  - inflate the prefilter's lower bound a hundredfold, past u_A v_A
    (marker "claim 9");
  - drop the Cholesky term delta from the rounding allowance (marker
    "claim 10").

Claim 10 (worst-case rounding). Higham's model: Cholesky L^ L^T = V + dV
with |dV| <= g_{d+1}|L^||L^T|, so ||dV|| <= eta = g_{d+1}/(1-g_{d+1}) tr V;
each solve column (L^ + dL) x^ = a, |dL| <= g_{2d}|L^|, moves x by
rho = theta/(1-theta), theta = g_{2d} sqrt(tr V/(1-g_{d+1})) ||L^-1||;
each product adds g_{2d} |x|.|y|. Then
|J^ - a^T V^-1 b| <= N_a N_b [delta + (1+rho)^2 (1+g_{2d}) - 1]/(1-delta),
delta = eta/lambda; the trace floor is (1-rho)^2/||Z||_F^2 - eta; the
own-norm inflation is sqrt(1+delta)/(1-rho). Checked with every
perturbation applied at its largest allowed size (an artificial unit
roundoff 1e-7), the Cholesky one along the lowest eigenvector.

Claim 9 (the prefilter). The code only attempts a pair whose cheap lower
bound sqrt(max_P t_i[P] / F_i) * min(sqrt(sum_P t_j[P] / m_i),
sum_P sqrt(dim_P t_j[P] / sqrt(f_P))) on u_A v_A stays below the cutoff, with
f_P = ||V_PP||_F^2, F_i = ||V||_F >= lambda_max and m_i = min_P sqrt(f_P) >=
min diag >= lambda_min; (V^-1)_kk >= 1/V_kk >= 1/sqrt(f_P) for k in P. It
decides cost only (a pair left out is fitted), so a wrong cut can lose a
skip but never cause one; the check confirms it loses none.

Claim 7 (the shift-verified floor). Rump's theorem: for symmetric M with
trace t, if the floating-point Cholesky of fl(M - cI) completes with
c = gamma_{d+1}/(1 - gamma_{d+1}) t (plus underflow), M is positive
definite. With M = V - sigma I, a completed Cholesky of V - (sigma + c) I
proves lambda_min(V) > sigma. The code estimates lambda_min from above by
power iteration on V^-1 and tries sigma = lam_est/2, then lam_est/10.
Checked against eigvalsh on random SPD matrices (kappa 1e2..1e11).

Claim 8 (the shell-wise alternatives). For B a subset of D_j and x zero
outside B, ||V_BB^-1/2 x_B|| <= ||L_j^-1 x|| (claim 2 with a zero x2) <=
sum_P ||L_j^-1[:, P]||_F ||x_P|| over the shells P of B; and
gamma <= ||L_A^-1 V_AB||_F / sqrt(lam_min(V_BB)) <= sum_P ||L_A^-1[:, P]||_F
||V_PB||_F / sqrt(lam_B) (and the mirror). Checked numerically.

Run: python scripts/lmp2_attenuated_pair_skip_sympy.py
"""

import numpy as np
import sympy as sp


def section(title):
    print(f"\n=== {title}")


# ---------------------------------------------------------------------------
# Claim 1: block-Neumann split (symbolic, 1+1 scalar blocks)
# ---------------------------------------------------------------------------
section("claim 1: J = alpha^T (I+X)^-1 beta and the X^2 remainder")
p, q, c = sp.symbols("p q c", positive=True)
a1, a2, b1, b2 = sp.symbols("a1 a2 b1 b2", real=True)
V = sp.Matrix([[p, c], [c, q]])
av = sp.Matrix([a1, a2])
bv = sp.Matrix([b1, b2])
J_direct = (av.T * V.inv() * bv)[0, 0]
Dg_mh = sp.diag(1 / sp.sqrt(p), 1 / sp.sqrt(q))
y = c / sp.sqrt(p * q)
X = sp.Matrix([[0, y], [y, 0]])
I2 = sp.eye(2)
alpha = Dg_mh * av
beta = Dg_mh * bv
J_split = (alpha.T * (I2 + X).inv() * beta)[0, 0]
assert sp.simplify(J_direct - J_split) == 0, "claim 1: whitened form FAILED"
neumann = I2 - X + X * X * (I2 + X).inv()  # claim 1: split
assert sp.simplify(neumann - (I2 + X).inv()) == sp.zeros(2, 2), "claim 1: split FAILED"
print("a^T V^-1 b == alpha^T (I+X)^-1 beta, (I+X)^-1 == I - X + X^2 (I+X)^-1")

# ---------------------------------------------------------------------------
# Claim 2: Schur monotonicity of the fitted norm (symbolic, 1+1)
# ---------------------------------------------------------------------------
section("claim 2: x^T M^-1 x - x1^T M11^-1 x1 = r^T S^-1 r >= 0")
m11, m22, m12 = sp.symbols("m11 m22 m12", real=True)
x1, x2 = sp.symbols("x1 x2", real=True)
M = sp.Matrix([[m11, m12], [m12, m22]])
xv = sp.Matrix([x1, x2])
S = m22 - m12 * m12 / m11  # claim 2: Schur
r = x2 - m12 / m11 * x1
lhs = (xv.T * M.inv() * xv)[0, 0] - x1 * x1 / m11
assert sp.simplify(lhs - r * r / S) == 0, "claim 2: Schur FAILED"
print("difference is r^2 / S with S > 0 for SPD M: the sub-block norm is smaller")

# ---------------------------------------------------------------------------
# Claim 3: erfc kernel is positive definite
# ---------------------------------------------------------------------------
section("claim 3: FT[erfc(w r)/r] = 4 pi (1 - exp(-k^2/4w^2))/k^2 > 0")
rr, k, w = sp.symbols("r k w", positive=True)
ft = sp.integrate(4 * sp.pi * rr**2 * sp.erfc(w * rr) / rr * sp.sin(k * rr) / (k * rr), (rr, 0, sp.oo))
ft = sp.simplify(ft)
target = 4 * sp.pi * (1 - sp.exp(-(k**2) / (4 * w**2))) / k**2
assert sp.simplify(ft - target) == 0, f"claim 3: FT FAILED: {ft}"
for kv in [0.01, 0.5, 3.0, 40.0]:
    for wv in [0.2223, 1.0]:
        assert float(target.subs({k: kv, w: wv})) > 0
print(f"FT = {target}; positive at every sampled (k, w)")

# ---------------------------------------------------------------------------
# Claims 4-5: the bound on random SPD instances (independent route: solve)
# ---------------------------------------------------------------------------
section("claims 4-5: |a^T V^-1 b| <= bound (dense solve as the independent route)")
rng = np.random.default_rng(20261124)


def lam_floor(V):
    L = np.linalg.cholesky(V)
    Linv = np.linalg.solve(L, np.eye(V.shape[0]))
    lam_lb = 1.0 / np.sum(Linv * Linv)  # claim 5: floor
    return lam_lb


def bound(uA, uB, vA, vB, g):
    if not g < 1.0:
        return np.inf
    rem = g * g / (1.0 - g) * np.hypot(uA, uB) * np.hypot(vA, vB)  # claim 4: bound
    return uA * vA + uB * vB + g * (uA * vB + uB * vA) + rem


def spd(n, cond):
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    ev = np.geomspace(1.0 / cond, 1.0, n)
    return (Q * ev) @ Q.T


def own_q(Vk, x):
    """fitted norm sqrt(x^T Vk^-1 x) via Cholesky (as whiten_own_domains)."""
    L = np.linalg.cholesky(Vk)
    return np.linalg.norm(np.linalg.solve(L, x))


worst = 0.0
for trial in range(400):
    na, nb, nov = rng.integers(2, 6), rng.integers(2, 6), rng.integers(0, 3)
    # D_i = A (na) ; D_j = B (nb) plus nov functions shared with A (D_j overlaps D_i)
    n = na + nb
    V = spd(n, 10 ** rng.uniform(0, 4))
    # weaken the A-B coupling so gamma < 1 in most trials
    s = 10 ** rng.uniform(-4, -0.3)
    V[:na, na:] *= s
    V[na:, :na] *= s
    if np.linalg.eigvalsh(V).min() <= 0:
        continue
    adv = trial % 4 == 0
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    a[na:] *= 0.0 if adv else 10 ** rng.uniform(-6, 0)
    b[:na] *= 0.0 if adv else 10 ** rng.uniform(-6, 0)
    A, B = slice(0, na), slice(na, n)
    VAA, VBB, VAB = V[A, A], V[B, B], V[A, B]
    # D_j's own domain: B plus the last `nov` functions of A (a superset of B)
    dj = list(range(na - nov, na)) + list(range(na, n))
    Vjj = V[np.ix_(dj, dj)]
    lam_i, lam_j = lam_floor(VAA), lam_floor(Vjj)
    assert lam_i <= np.linalg.eigvalsh(VAA).min() * (1 + 1e-12), "claim 5: floor FAILED"
    # interlacing: B is a sub-block of D_j, so lambda_min(V_BB) >= lam_j
    assert np.linalg.eigvalsh(VBB).min() >= lam_j * (1 - 1e-12), "claim 5: interlacing FAILED"
    if adv:
        # align alpha_A, beta_B with Y's top singular pair (worst case)
        Wa = np.linalg.inv(np.linalg.cholesky(VAA))
        Wb = np.linalg.inv(np.linalg.cholesky(VBB))
        Y = Wa @ VAB @ Wb.T
        U, sv, Vt = np.linalg.svd(Y)
        a[A] = np.linalg.solve(Wa, U[:, 0])
        b[B] = np.linalg.solve(Wb, -Vt[0])
    J = a @ np.linalg.solve(V, b)
    uA = own_q(VAA, a[A])
    vB_own = own_q(Vjj, b[dj])
    vB_sub = own_q(VBB, b[B])
    assert vB_sub <= vB_own * (1 + 1e-10), "claim 2: Schur FAILED (numeric)"
    uB = np.linalg.norm(a[B]) / np.sqrt(lam_j)
    vA = np.linalg.norm(b[A]) / np.sqrt(lam_i)
    g = np.linalg.norm(VAB) / np.sqrt(lam_i * lam_j)
    if adv:
        # the bound holds for any gamma >= ||Y||_2; the exact one makes it tight
        assert sv[0] <= g * (1 + 1e-10), "claim 4: gamma FAILED"
        g = sv[0]
    G = bound(uA, uB, vA, vB_own, g)
    if np.isfinite(G):
        assert abs(J) <= G * (1 + 1e-9), f"claim 4: bound FAILED trial {trial}: {J} > {G}"
        worst = max(worst, abs(J) / G)
print(f"bound holds on every finite-gamma trial; max |J|/bound = {worst:.3f}")
assert worst > 0.9, "claim 4: bound never approached (test is vacuous)"

# ---------------------------------------------------------------------------
# Claim 6: Fock-dressed gate bound
# ---------------------------------------------------------------------------
section("claim 6: |g~| <= G (1 + 2 max w)")
worst = 0.0
for trial in range(300):
    nv = rng.integers(2, 7)
    fv = np.sort(rng.uniform(0.3, 2.0, nv))
    fo = rng.uniform(-1.5, -0.4, 2)
    F = rng.standard_normal((nv, nv)) * 0.1
    F = 0.5 * (F + F.T)
    np.fill_diagonal(F, fv)
    g = rng.standard_normal((nv, nv))
    if trial % 2 == 0:
        g = np.sign(g)  # every element at the bound
    G = np.abs(g).max()
    Fp = F - np.diag(np.diag(F))
    t0 = -g / (fv[:, None] + fv[None, :] - fo.sum())
    gt = g + Fp @ t0 + t0 @ Fp
    D = fv + fv.min() - 2 * fo.max()
    assert (D > 0).all()
    wv = (np.abs(Fp) / D[None, :]).sum(axis=1)  # claim 6: dressed
    dressed = G * (1 + 2 * wv.max())
    assert np.abs(gt).max() <= dressed * (1 + 1e-12), "claim 6: dressed FAILED"
    worst = max(worst, np.abs(gt).max() / dressed)
print(f"dressed bound holds; max ratio = {worst:.3f}")
assert worst > 0.5, "claim 6: dressed bound never approached (test is vacuous)"

# ---------------------------------------------------------------------------
# Claim 7: the shift-verified eigenvalue floor (Rump, BIT 46 (2006) 433)
# ---------------------------------------------------------------------------
section("claim 7: a Cholesky of V - (sigma + c) I proves lambda_min > sigma")


def rump_shift(d, trace, diag_max):
    u = np.finfo(float).eps / 2
    g = (d + 1) * u / (1 - (d + 1) * u)
    return 2.0 * (g / (1 - g) * trace + 4 * u * diag_max)


def verified_floor(V, shifts=(0.5, 0.1)):
    """lmp2_pair_skip.rs::verified_floor: power iteration on V^-1, then the
    first of sigma = lam_est/2, lam_est/10 whose shifted Cholesky succeeds."""
    d = V.shape[0]
    Linv = np.linalg.inv(np.linalg.cholesky(V))
    x = 1.0 + (np.arange(d) % 7) * 0.1
    for _ in range(40):
        y = Linv.T @ (Linv @ x)
        mu = x @ y / (x @ x)
        x = y / np.linalg.norm(y)
    lam_est = 1.0 / mu
    c = rump_shift(d, np.trace(V), np.diag(V).max())
    for f in shifts:
        sigma = f * lam_est
        try:
            np.linalg.cholesky(V - (sigma + c) * np.eye(d))  # claim 7: verify
            return sigma, lam_est
        except np.linalg.LinAlgError:
            pass
    return None, lam_est


n_ok, gain = 0, []
for trial in range(60):
    n = int(rng.integers(10, 80))
    V = spd(n, 10.0 ** rng.uniform(2, 11))
    lam = np.linalg.eigvalsh(V)[0]
    sigma, lam_est = verified_floor(V)
    # lam_est is only an estimate (the computed L^-1 rounds at kappa ~ 1e11);
    # the shifted Cholesky is what proves the floor
    if sigma is None:
        continue
    n_ok += 1
    assert sigma <= lam, f"claim 7: verified floor {sigma} above lambda_min {lam}"
    # a shift past lambda_min must be refused by the verification
    bad, _ = verified_floor(V, shifts=(2.0,))
    assert bad is None or bad <= lam, "claim 7: an unverified shift was accepted"
    gain.append(sigma / lam_floor(V))
print(f"verified floors: {n_ok}/60 found, all <= lambda_min; "
      f"gain over the trace floor: min {min(gain):.1f}, median {np.median(gain):.1f}")
assert n_ok >= 50, "claim 7: the shifted Cholesky almost never verifies (vacuous)"

# ---------------------------------------------------------------------------
# Claim 8: the shell-wise whitening bounds (alternatives the code takes the
# minimum with)
# ---------------------------------------------------------------------------
section("claim 8: shell-wise bounds on ||V_BB^-1/2 x_B|| and on gamma")


def shell_split(n, nsh):
    cuts = np.sort(rng.choice(np.arange(1, n), size=nsh - 1, replace=False))
    return np.split(np.arange(n), cuts)


worst8 = 0.0
for trial in range(200):
    nj = int(rng.integers(6, 30))
    Vj = spd(nj, 10.0 ** rng.uniform(1, 8))  # V_{D_j D_j}
    Lj_inv = np.linalg.inv(np.linalg.cholesky(Vj))
    shells = shell_split(nj, int(rng.integers(2, min(nj, 8))))
    c = [np.linalg.norm(Lj_inv[:, sh]) for sh in shells]  # Frobenius per block
    nb = int(rng.integers(1, len(shells)))
    Bsh = sorted(rng.choice(len(shells), size=nb, replace=False))
    B = np.concatenate([shells[k] for k in Bsh])
    x = np.zeros(nj)
    x[B] = rng.standard_normal(B.size) * 10.0 ** rng.uniform(-6, 0, B.size)
    exact = np.linalg.norm(np.linalg.solve(np.linalg.cholesky(Vj[np.ix_(B, B)]), x[B]))
    padded = np.linalg.norm(Lj_inv @ x)  # Schur monotonicity, zero padding
    shellwise = sum(c[k] * np.linalg.norm(x[shells[k]]) for k in Bsh)  # claim 8: shellwise
    assert exact <= padded * (1 + 1e-10), "claim 8: zero-padded monotonicity FAILED"
    assert padded <= shellwise * (1 + 1e-10), "claim 8: shell-wise triangle FAILED"
    worst8 = max(worst8, exact / shellwise)
    # gamma one-sided: ||V_AA^-1/2 V_AB V_BB^-1/2|| <= ||L_A^-1 V_AB||_F / sqrt(lam_B)
    na = int(rng.integers(3, 12))
    V = spd(na + B.size, 10.0 ** rng.uniform(1, 6))
    VAA, VBB, VAB = V[:na, :na], V[na:, na:], V[:na, na:]
    w, U = np.linalg.eigh(VAA)
    VAA_mh = (U / np.sqrt(w)) @ U.T
    w, U = np.linalg.eigh(VBB)
    VBB_mh = (U / np.sqrt(w)) @ U.T
    g_exact = np.linalg.norm(VAA_mh @ VAB @ VBB_mh, 2)
    LA_inv = np.linalg.inv(np.linalg.cholesky(VAA))
    ash = shell_split(na, int(rng.integers(1, min(na, 4) + 1))) if na > 1 else [np.arange(na)]
    side = sum(np.linalg.norm(LA_inv[:, sh]) * np.linalg.norm(VAB[sh, :]) for sh in ash)
    g_side = side / np.sqrt(np.linalg.eigvalsh(VBB)[0])
    assert g_exact <= g_side * (1 + 1e-10), "claim 8: one-sided gamma FAILED"
print(f"shell-wise bounds hold on 200 trials; max exact/shell-wise = {worst8:.3f}")
assert worst8 > 0.3, "claim 8: shell-wise bound never approached (test is vacuous)"

# ---------------------------------------------------------------------------
# Claim 9: the cost-only prefilter's lower bound on the gamma-free term
# ---------------------------------------------------------------------------
section("claim 9: prefilter lower bound on u_A and v_A")
worst9 = 0.0
for trial in range(200):
    n = int(rng.integers(6, 30))
    V = spd(n, 10.0 ** rng.uniform(1, 8)) * 10.0 ** rng.uniform(-2, 2)  # V_{D_i D_i}
    shells = shell_split(n, int(rng.integers(2, min(n, 8))))
    f = [np.sum(V[np.ix_(sh, sh)] ** 2) for sh in shells]  # ||V_PP||_F^2
    F = np.linalg.norm(V)  # ||V||_F >= lambda_max
    m = min(np.sqrt(fp) for fp in f)  # >= min diag >= lambda_min
    ncol = int(rng.integers(1, 6))
    Bi = rng.standard_normal((n, ncol)) * 10.0 ** rng.uniform(-6, 0, (n, 1))
    bj = rng.standard_normal(n) * 10.0 ** rng.uniform(-6, 0, n)
    ti = [max(np.sum(Bi[sh, a] ** 2) for a in range(ncol)) for sh in shells]
    tj = [np.sum(bj[sh] ** 2) for sh in shells]
    L = np.linalg.cholesky(V)
    Linv = np.linalg.inv(L)
    uA = max(np.linalg.norm(np.linalg.solve(L, Bi[:, a])) for a in range(ncol))
    lam = np.linalg.eigvalsh(V)[0]
    c = [np.linalg.norm(Linv[:, sh]) for sh in shells]
    vA = min(np.sqrt(sum(tj) / lam), sum(ck * np.sqrt(t) for ck, t in zip(c, tj)))
    v_lb = min(
        np.sqrt(sum(tj) / m),
        sum(np.sqrt(len(sh) * t / np.sqrt(fp)) for sh, t, fp in zip(shells, tj, f)),
    )
    lb = np.sqrt(max(ti) / F) * v_lb  # claim 9: prefilter
    assert lb <= uA * vA * (1 + 1e-10), "claim 9: prefilter lower bound FAILED"
    worst9 = max(worst9, lb / (uA * vA))
print(f"prefilter lower bound holds on 200 trials; max lb/(u_A v_A) = {worst9:.3f}")
assert worst9 > 1e-3, "claim 9: prefilter bound never approached (test is vacuous)"

# ---------------------------------------------------------------------------
# Claim 10: the worst-case rounding allowances (Higham's error model)
# ---------------------------------------------------------------------------
section("claim 10: worst-case Cholesky / solve / product allowances")
# Every modelled perturbation is applied explicitly at its largest allowed
# size, with an artificial unit roundoff U so that the perturbations dwarf
# float64's own rounding. The formulas are those of lmp2_pair_skip.rs.
U = 1e-7


def gam(n):
    return n * U / (1 - n * U)


def chol_eta(d, tr):
    return gam(d + 1) / (1 - gam(d + 1)) * tr


def solve_theta(d, tr, inv2):
    th = gam(2 * d) * np.sqrt(tr / (1 - gam(d + 1)) * inv2)
    return th if th <= 0.25 else None


def allowance(d, tr, lam_d, na_nb):
    eta = chol_eta(d, tr)
    delta = eta / lam_d  # claim 10: delta
    th = solve_theta(d, tr, 1.0 / (lam_d - eta))
    r = th / (1 - th)
    return na_nb * (delta + (1 + r) ** 2 * (1 + gam(2 * d)) - 1) / (1 - delta)


def perturbed_solve(Lh, rhs, g):
    """(Lh + dL) x = rhs with |dL| = g |Lh| and random signs (lower)."""
    sgn = np.tril(rng.choice([-1.0, 1.0], size=Lh.shape))
    return np.linalg.solve(Lh + g * sgn * np.abs(Lh), rhs)


worst10 = 0.0
for trial in range(120):
    d = int(rng.integers(5, 40))
    V = spd(d, 10.0 ** rng.uniform(1, 3.5))
    w, Q = np.linalg.eigh(V)
    lam = w[0]
    tr = np.trace(V)
    # (a) eta bounds ||dV|| for any |dV| <= g |Lh||Lh^T|
    Lh = np.linalg.cholesky(V)
    E = gam(d + 1) * rng.choice([-1.0, 1.0], size=(d, d)) * (np.abs(Lh) @ np.abs(Lh).T)
    E = np.triu(E) + np.triu(E, 1).T
    Vx = Lh @ Lh.T - E  # the exact matrix whose computed factor is Lh
    assert np.linalg.norm(E, 2) <= chol_eta(d, np.trace(Vx)) * (1 + 1e-9), "claim 10: eta FAILED"
    # (b) the J allowance, adversarial: dV = -eta v v^T along lambda_min
    eta = chol_eta(d, tr)
    if eta / lam >= 0.5 or solve_theta(d, tr, 1.0 / (lam - eta)) is None:
        continue
    v0 = Q[:, 0]
    Lh = np.linalg.cholesky(V - eta * np.outer(v0, v0))
    a = v0 + 0.1 * rng.standard_normal(d)
    b = v0 + 0.1 * rng.standard_normal(d)
    exact = a @ np.linalg.solve(V, b)
    xa = perturbed_solve(Lh, a, gam(2 * d))
    xb = perturbed_solve(Lh, b, gam(2 * d))
    J = xa @ xb + gam(2 * d) * (np.abs(xa) @ np.abs(xb)) * np.sign(xa @ xb - exact or 1.0)
    na = np.sqrt(a @ np.linalg.solve(V, a))
    nb = np.sqrt(b @ np.linalg.solve(V, b))
    R = allowance(d, tr, lam, na * nb)
    assert abs(J - exact) <= R * (1 + 1e-9), "claim 10: J allowance FAILED"
    worst10 = max(worst10, abs(J - exact) / R)
    # (c) the trace floor from a perturbed inverse factor
    Z = np.column_stack([perturbed_solve(Lh, e, gam(2 * d)) for e in np.eye(d)])
    f2 = np.sum(Z * Z)
    th = solve_theta(d, tr, 2.25 * f2)
    if th is not None:
        r = th / (1 - th)
        floor = (1 - r) ** 2 / f2 - eta
        assert floor <= lam * (1 + 1e-12), "claim 10: floor FAILED"
        # (d) the norm inflation (floor lam: the exact one is allowed)
        infl = np.sqrt(1 + eta / lam) / (1 - r)
        assert na <= infl * np.linalg.norm(xa) * (1 + 1e-12), "claim 10: inflation FAILED"
print(f"rounding allowances hold; max |J error|/allowance = {worst10:.3f}")
assert worst10 > 0.3, "claim 10: allowance never approached (test is vacuous)"

print("\nall claims hold")
