#!/usr/bin/env python3
"""Fock-builder algebra as the SCF codes it: RI-J, RI-K, the incremental
build and the COSX quadrature, checked against exact J/K.

Derivation: crates/ferric-scf/theory.md, "Direct and incremental J/K builds",
"Density-fitted Coulomb and exchange (RI-J, RI-K, Split-RI-J)" and
"LinK and COSX exchange". Code: crates/ferric-scf/src/df_j.rs (DfJ),
df_k.rs (DfK), split_rij.rs (SplitRiJ), direct_jk.rs (DirectJK),
cosx_k.rs (CosxK).

Conventions (book/notation.md, section 7): raw three-index (P|mn), metric
V_PQ = (P|Q), dressed B^P_mn = sum_Q M_PQ (Q|mn) with M^T M = V^-1,
J[D]_mn = sum_ls (mn|ls) D_ls, K[D]_mn = sum_ls (ml|ns) D_ls,
closed shell D = 2 C_o C_o^T.
  RI-J   gamma_P = sum_ls (P|ls) D_ls,  V c = gamma,  J_mn = sum_P (mn|P) c_P.
  RI-K   K_mn = sum_P sum_ls B^P_ml D_ls B^P_sn = sum_P (B^P D B^P)_mn;
         occupied route K[2 C_o C_o^T] = 2 sum_P (B^P C_o)(B^P C_o)^T.

The toy model. Orbital products and auxiliary functions live in one
R-dimensional "complete" function space with a positive definite Coulomb
metric W (W_RS = (phi_R|phi_S)). Each AO product chi_m chi_n is the vector
t_mn (symmetric in m, n), so the exact integrals are (mn|ls) = t_mn^T W t_ls
(8-fold symmetric and positive semidefinite by construction). Auxiliary
function P is the vector a_P, so (P|mn) = a_P^T W t_mn and V = a W a^T. With
a square and invertible (the auxiliary set spans the space) density fitting
must be exact; with fewer auxiliary functions it is a projection.

Symbolic checks (sympy):
  S1  The Coulomb-metric fit is a projection: Pi = a^T V^-1 a W satisfies
      Pi^2 = Pi and W Pi = Pi^T W (W-self-adjoint), symbolically for R = 2 and
      one auxiliary function.
  S2  Fitting error of the Coulomb energy is the residual's self-interaction:
      (rho|rho) - gamma^T V^-1 gamma = (delta|delta), delta = rho - a^T c,
      symbolically for R = 2, one auxiliary function.
  S3  The robust (Dunlap) functional 2 gamma^T c' - c'^T V c' equals
      gamma^T V^-1 gamma at c' = c and its first derivative in c' vanishes
      there; the plain gamma^T c' does not have zero derivative.
Numerical checks (random model, n = 6 AOs, R = 21 = n(n+1)/2):
  N1  Complete auxiliary set (a square, invertible): RI-J and RI-K equal the
      exact J and K to 1e-10 relative (the exactness anchor).
  N2  The choice of M does not matter: M = L^-1 (Cholesky V = L L^T) and
      M = V^-1/2 give the same J and K to 1e-12.
  N3  RI-J from the fit coefficients, J = (mn|P) c_P with V c = gamma, equals
      sum_P B^P_mn (sum_ls B^P_ls D_ls) (both reduced auxiliary sets).
  N4  Occupied route: K_occ(X) = sum_P (B^P X)(B^P X)^T carries no
      occupation factor and the RHF caller doubles it, so for D = 2 C_o C_o^T,
      2 K_occ(C_o) = sum_P B^P D B^P (reduced set).
  N5  Signed factor route for a symmetric non-idempotent D (a difference
      density), as DfK::build_from_density_factor codes it: D = V diag(lam) V^T,
      X+ = V sqrt(lam) over lam > cut, X- = V sqrt(|lam|) over lam < -cut,
      K_occ(X+) - K_occ(X-) equals sum_P B^P D B^P; dropping X- does not.
  N6  Reduced set: the Coulomb energy error (1/2) tr[D (J - J_RI)] is >= 0 and
      equals (1/2)(delta|delta); it shrinks as auxiliary functions are added
      and is zero for the complete set.
  N7  Packed-triangle contraction: gamma_P from the lower triangle with
      off-diagonal weight 2 (D symmetric) equals the full double sum.
  N8  The K metric is the truncated symmetric V^-1/2 (df_k.rs
      v_inv_sqrt_lindep): with nothing below the cut it gives the Cholesky K;
      on a near-singular V (three duplicated auxiliary functions) the cut
      removes exactly the duplicated modes and K equals the K of the
      non-duplicated set.

  N9  Incremental build: accumulating J[dD] and K[dD] over iterations, with a
      full rebuild every 8 iterations as solve_rhf does, equals J[D] and K[D]
      of the current density to rounding.
COSX checks (cosx_k.rs: X_mg = sqrt(w_g) chi_m(r_g), F = D X,
A_nl(g) = int chi_n chi_l / |r - r_g|, G_ng = sum_l A_nl(g) F_lg,
K~ = X G^T, overlap fit K = sym(S S_num^-1 K~), S_num = X X^T):
  C1  One normalized s Gaussian (exponent alpha): with A(r) = erf(sqrt(2 alpha) r)/r
      done analytically, the radial quadrature of K~ / D converges to
      (ss|ss) = 2 sqrt(alpha/pi) (the grid does one electron, the analytic
      integral the other).
  C2  The overlap fit is exact for a constant potential: with A_nl(g) = c S_nl
      on any grid, fitted COSX gives c S D S exactly while the plain K~ gives
      c S_num D S.

Mutations (each turns the script red; the runner applies them as exact
one-occurrence substitutions):
  rij_coefficients: solve V c = gamma -> c = gamma       (N1 RI-J fails)
  rhf_k: caller factor 2 -> 1                            (N4 fails)
  rik_signed: K(X+) - K(X-) -> K(X+) + K(X-)            (N5 fails)
  packed_gamma: off-diagonal weight 2 -> 1               (N7 fails)
  cosx_fit: S S_num^-1 K~ -> K~                          (C2 fails)
Exits nonzero if any check fails; prints "FAIL [<claim>] ...".
"""

import sys

import numpy as np
import sympy as sp

failures = []


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        failures.append(label)


# ---------------------------------------------------------------------------
# Symbolic part
# ---------------------------------------------------------------------------
def symbolic_checks():
    w11, w12, w22 = sp.symbols("w11 w12 w22", real=True)
    a1, a2 = sp.symbols("a1 a2", real=True)
    r1, r2 = sp.symbols("r1 r2", real=True)
    w = sp.Matrix([[w11, w12], [w12, w22]])
    a = sp.Matrix([[a1, a2]])  # one auxiliary function in R = 2
    v = a * w * a.T
    vinv = v.inv()
    proj = a.T * vinv * a * w
    ok = sp.simplify(proj * proj - proj) == sp.zeros(2, 2) and sp.simplify(
        w * proj - proj.T * w
    ) == sp.zeros(2, 2)
    check("[S1] Coulomb-metric fit is a W-self-adjoint projection", ok)

    rho = sp.Matrix([r1, r2])
    gamma = a * w * rho
    c = vinv * gamma
    delta = rho - a.T * c
    lhs = (rho.T * w * rho)[0] - (gamma.T * vinv * gamma)[0]
    rhs = (delta.T * w * delta)[0]
    check(
        "[S2] (rho|rho) - gamma V^-1 gamma = (delta|delta)", sp.simplify(lhs - rhs) == 0
    )

    cp = sp.symbols("cp", real=True)
    robust = 2 * gamma[0] * cp - v[0] * cp**2
    plain = gamma[0] * cp
    at_c = sp.simplify(robust.subs(cp, c[0]) - (gamma.T * vinv * gamma)[0])
    d_robust = sp.simplify(sp.diff(robust, cp).subs(cp, c[0]))
    d_plain = sp.simplify(sp.diff(plain, cp).subs(cp, c[0]))
    check(
        "[S3] robust functional is stationary at the fit, plain is not",
        at_c == 0 and d_robust == 0 and d_plain != 0,
    )


# ---------------------------------------------------------------------------
# Numerical toy model
# ---------------------------------------------------------------------------
N = 6
R = N * (N + 1) // 2
rng = np.random.default_rng(20261115)

pairs = [(m, n) for m in range(N) for n in range(m + 1)]
t = np.zeros((R, N, N))  # t[:, m, n] is the product chi_m chi_n in R-space
for k, (m, n) in enumerate(pairs):
    vec = rng.normal(size=R)
    t[:, m, n] = vec
    t[:, n, m] = vec
q = rng.normal(size=(R, R))
w_metric = q @ q.T + R * np.eye(R)  # positive definite Coulomb metric

eri = np.einsum("rmn,rq,qls->mnls", t, w_metric, t)


def j_exact(d):
    return np.einsum("mnls,ls->mn", eri, d)


def k_exact(d):
    return np.einsum("mlns,ls->mn", eri, d)


def three_index(a):
    """(P|mn) = a_P^T W t_mn, shape (naux, N, N)."""
    return np.einsum("pr,rs,smn->pmn", a, w_metric, t)


def metric(a):
    return a @ w_metric @ a.T


def rij_coefficients(p3, v, d):
    gamma = np.einsum("pls,ls->p", p3, d)
    return np.linalg.solve(v, gamma)


def rij(p3, v, d):
    c = rij_coefficients(p3, v, d)
    return np.einsum("pmn,p->mn", p3, c)


def dress_cholesky(p3, v):
    """M = L^-1 from V = L L^T (the dressing the MP2 code uses)."""
    ell = np.linalg.cholesky(v)
    m = np.linalg.inv(ell)
    return np.einsum("pq,qmn->pmn", m, p3)


def dress_symmetric(p3, v, cut=0.0):
    """M = U diag(lam^-1/2) U^T over the modes with lam >= cut (df_k.rs
    v_inv_sqrt_lindep); with cut = 0 and a positive definite V it is V^-1/2."""
    lam, u = np.linalg.eigh(v)
    keep = lam >= cut
    m = (u[:, keep] * lam[keep] ** -0.5) @ u[:, keep].T
    return np.einsum("pq,qmn->pmn", m, p3)


def rik_density(b, d):
    return np.einsum("pml,ls,psn->mn", b, d, b)


def rik_occupied(b, x):
    """K_occ(X) = sum_P (B^P X)(B^P X)^T, no occupation factor (occ_chunk_k)."""
    half = np.einsum("pml,li->pmi", b, x)
    return np.einsum("pmi,pni->mn", half, half)


def rhf_k(b, c_occ):
    """The RHF caller doubles the occupied-route K: D = 2 C_o C_o^T."""
    return 2.0 * rik_occupied(b, c_occ)


def rik_signed(b, d, rel_cut=1e-11):
    """build_from_density_factor: D = V diag(lam) V^T, X+ = V sqrt(lam) over
    lam > cut, X- = V sqrt(|lam|) over lam < -cut, K = K_occ(X+) - K_occ(X-)."""
    lam, vec = np.linalg.eigh(d)
    cut = rel_cut * np.abs(lam).max()
    xp = vec[:, lam > cut] * np.sqrt(lam[lam > cut])
    xm = vec[:, lam < -cut] * np.sqrt(-lam[lam < -cut])
    return rik_occupied(b, xp) - rik_occupied(b, xm)


def packed_gamma(p3, d):
    gamma = np.zeros(p3.shape[0])
    for m in range(N):
        for n in range(m + 1):
            wt = 1.0 if m == n else 2.0
            gamma += wt * p3[:, m, n] * d[m, n]
    return gamma


def rel(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)


def numerical_checks():
    c_occ = np.linalg.qr(rng.normal(size=(N, 2)))[0]
    d = 2.0 * c_occ @ c_occ.T
    x = rng.normal(size=(N, N))
    d_diff = 0.3 * (x + x.T)  # symmetric, indefinite: a difference density

    # N1 complete set
    a_full = rng.normal(size=(R, R))
    p3 = three_index(a_full)
    v = metric(a_full)
    b = dress_cholesky(p3, v)
    e1 = rel(rij(p3, v, d), j_exact(d))
    e2 = rel(rik_density(b, d), k_exact(d))
    check("[N1] complete auxiliary set: RI-J == exact J", e1 < 1e-10, f"rel {e1:.1e}")
    check("[N1] complete auxiliary set: RI-K == exact K", e2 < 1e-10, f"rel {e2:.1e}")

    # reduced sets
    a_red = rng.normal(size=(R - 6, R))
    p3r = three_index(a_red)
    vr = metric(a_red)
    bl = dress_cholesky(p3r, vr)
    bs = dress_symmetric(p3r, vr)

    e = max(
        rel(rik_density(bl, d), rik_density(bs, d)),
        rel(
            np.einsum("pmn,p->mn", bl, np.einsum("pls,ls->p", bl, d)),
            np.einsum("pmn,p->mn", bs, np.einsum("pls,ls->p", bs, d)),
        ),
    )
    check(
        "[N2] M = L^-1 and M = V^-1/2 give the same J and K", e < 1e-12, f"rel {e:.1e}"
    )

    jb = np.einsum("pmn,p->mn", bl, np.einsum("pls,ls->p", bl, d))
    e = rel(rij(p3r, vr, d), jb)
    check("[N3] (mn|P) c_P == sum_P B^P (B^P . D)", e < 1e-12, f"rel {e:.1e}")

    e = rel(rhf_k(bl, c_occ), rik_density(bl, d))
    check(
        "[N4] occupied route 2 sum (B C_o)(B C_o)^T == sum B D B",
        e < 1e-12,
        f"rel {e:.1e}",
    )

    e = rel(rik_signed(bl, d_diff), rik_density(bl, d_diff))
    wts, u = np.linalg.eigh(d_diff)
    pos = (u * np.clip(wts, 0, None)) @ u.T
    e_drop = rel(rik_density(bl, pos), rik_density(bl, d_diff))
    check(
        "[N5] signed factor route K(X+) - K(X-) == sum B D B for an indefinite D",
        e < 1e-12 and e_drop > 1e-3,
        f"rel {e:.1e}, positive part only {e_drop:.1e}",
    )

    # N8 the K metric: truncated symmetric V^-1/2. Nothing below the cut ->
    # the same K as the Cholesky dressing (trivial limit); a near-singular V
    # with its small modes cut -> a pseudo-inverse fit that differs from the
    # full-inverse (Cholesky) fit that RI-J uses.
    bt = dress_symmetric(p3r, vr, cut=1e-10)
    e0 = rel(rik_density(bt, d), rik_density(bl, d))
    a_sing = np.vstack([a_red, a_red[:3] + 1e-7 * rng.normal(size=(3, R))])
    p3s, vs = three_index(a_sing), metric(a_sing)
    lam = np.linalg.eigvalsh(vs)
    cut = 1e-6 * lam.max()
    nkept = int((lam >= cut).sum())
    ks = rik_density(dress_symmetric(p3s, vs, cut=cut), d)
    k_full = rik_density(dress_symmetric(p3s, vs), d)
    e_cut = rel(ks, rik_density(bl, d))
    check(
        "[N8] truncated V^-1/2: == Cholesky K when nothing is cut; a cut V fits the kept span",
        e0 < 1e-12 and nkept == R - 6 and e_cut < 1e-6 and rel(k_full, ks) > 0,
        f"no cut {e0:.1e}; kept {nkept} of {len(lam)}, vs reduced-set K {e_cut:.1e}",
    )

    # N6 error is the residual self-energy, positive, monotone in the aux set
    rho = np.einsum("rmn,mn->r", t, d)
    errs = []
    for naux in (R - 12, R - 6, R - 2, R):
        a_k = a_full[:naux]
        p3k, vk = three_index(a_k), metric(a_k)
        jr = rij(p3k, vk, d)
        err = 0.5 * np.sum(d * (j_exact(d) - jr))
        c = rij_coefficients(p3k, vk, d)
        delta = rho - a_k.T @ c
        errs.append((err, 0.5 * delta @ w_metric @ delta))
    ok = all(abs(x - y) < 1e-9 * max(1.0, abs(y)) for x, y in errs)
    ok = ok and all(x >= -1e-10 for x, _ in errs)
    ok = ok and all(errs[i][0] >= errs[i + 1][0] - 1e-10 for i in range(len(errs) - 1))
    ok = ok and abs(errs[-1][0]) < 1e-8 * abs(0.5 * np.sum(d * j_exact(d)))
    check(
        "[N6] Coulomb energy error = (delta|delta)/2 >= 0, monotone, zero when complete",
        ok,
        " ".join(f"{x:.3e}" for x, _ in errs),
    )

    e = rel(packed_gamma(p3r, d), np.einsum("pls,ls->p", p3r, d))
    check(
        "[N7] packed lower triangle, off-diagonal weight 2 == full sum",
        e < 1e-13,
        f"rel {e:.1e}",
    )


def incremental_and_cosx_checks():
    # N9 incremental accumulation with a full rebuild every 8 iterations
    ds = []
    c = np.linalg.qr(rng.normal(size=(N, 2)))[0]
    for _ in range(20):
        c = np.linalg.qr(c + 0.05 * rng.normal(size=(N, 2)))[0]
        ds.append(2.0 * c @ c.T)
    jb = kb = d_last = None
    worst = 0.0
    for it, d in enumerate(ds, start=1):
        if d_last is None or it % 8 == 1:
            jb, kb = j_exact(d), k_exact(d)
        else:
            dd = d - d_last
            jb, kb = jb + j_exact(dd), kb + k_exact(dd)
        d_last = d
        worst = max(worst, rel(jb, j_exact(d)), rel(kb, k_exact(d)))
    check(
        "[N9] incremental J/K with a rebuild every 8 iterations == J/K of D",
        worst < 1e-13,
        f"rel {worst:.1e}",
    )

    # C1 one s Gaussian: radial Gauss-Laguerre-like quadrature on r in (0, R)
    from math import erf, pi, sqrt

    alpha = 0.8
    exact = 2.0 * sqrt(alpha / pi)
    errs = []
    for npts in (10, 20, 40, 80):
        x, wq = np.polynomial.legendre.leggauss(npts)
        rmax = 8.0 / sqrt(alpha)
        r = 0.5 * rmax * (x + 1.0)
        w = 0.5 * rmax * wq * 4.0 * pi * r**2  # spherical weights
        chi = (2 * alpha / pi) ** 0.75 * np.exp(-alpha * r**2)
        a_esp = np.array([erf(sqrt(2 * alpha) * ri) / ri for ri in r])
        xg = np.sqrt(w) * chi
        k_tilde = np.sum(xg * a_esp * xg)  # D = 1: K~ = sum_g X G^T
        s_num = np.sum(xg * xg)
        errs.append((abs(k_tilde - exact), abs(k_tilde / s_num - exact), s_num))
    ok = errs[-1][0] < 1e-10 and errs[-1][1] < 1e-10 and errs[0][0] > errs[-1][0]
    check(
        "[C1] COSX K of one s Gaussian -> (ss|ss) = 2 sqrt(alpha/pi) as the grid grows",
        ok,
        " ".join(f"{e[0]:.1e}" for e in errs),
    )

    # C2 overlap fit exact for a constant potential, on a deliberately poor grid
    ngrid = 40
    xmat = rng.normal(size=(N, ngrid))  # X_mg on some grid
    q = rng.normal(size=(N, N))
    s_exact = q @ q.T / N + np.eye(N)  # the analytic overlap
    s_num = xmat @ xmat.T
    cconst = 0.37
    d = rng.normal(size=(N, N))
    d = d + d.T
    f = d @ xmat
    g = cconst * s_exact @ f  # G_ng = sum_l A_nl F_lg with A = c S
    k_tilde = xmat @ g.T
    k_fit = cosx_fit(s_exact, s_num, k_tilde)
    want = cconst * s_exact @ d @ s_exact
    plain = 0.5 * (k_tilde + k_tilde.T)
    e_fit = rel(k_fit, want)
    e_plain = rel(plain, want)
    check(
        "[C2] overlap-fitted COSX is exact for a constant potential, plain is not",
        e_fit < 1e-12 and e_plain > 1e-2,
        f"fitted {e_fit:.1e}, plain {e_plain:.1e}",
    )


def cosx_fit(s_exact, s_num, k_tilde):
    """finalize_fitted: K = sym(S S_num^-1 K~); only the left factor is fitted."""
    qk = s_exact @ np.linalg.solve(s_num, k_tilde)
    return 0.5 * (qk + qk.T)


symbolic_checks()
numerical_checks()
incremental_and_cosx_checks()
if failures:
    print(f"{len(failures)} check(s) failed")
    sys.exit(1)
print("all checks passed")
