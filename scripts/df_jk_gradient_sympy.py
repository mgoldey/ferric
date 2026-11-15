#!/usr/bin/env python3
"""Nuclear gradient of density-fitted J and K at fixed density.

Derivation: crates/ferric-scf/theory.md, "Density-fitted J/K nuclear
gradient"; book/src/14-derivatives/02-df-jk-gradient.md. Code:
crates/ferric-scf/src/df_gradient.rs (df_jk_gradient_core,
truncated_inverse_divided_difference) and df_gradient/deriv3c.rs.

Model: raw three-index integrals A_P,mn(x) = A0 + x A1 and metric V(x) =
V0 + x V1 depend on one parameter x, at a fixed density D = X X^T.
  E_J = 1/2 d^T V^-1 d,  d_P = sum D_mn A_P,mn
  E_K = alpha sum_ij L_ij^T V+ L_ij,  L_P,ij = sum X_mi A_P,mn X_nj
with V+ the metric inverse (the truncated spectral inverse for K).
Formula under test (the two weights of the book's equation 14.5):
  dE = sum G_P,mn A1_P,mn + sum H_PQ V1_PQ,
  G_P,mn = c_P D_mn + 2 alpha (X Gamma_P X^T)_mn,   c = V^-1 d,  Gamma = V+ L,
  H = -1/2 c c^T + U (F o U^T Y U) U^T,   Y = alpha sum_ij L_ij L_ij^T.

Checks (each label carries "(claim X)"):
  D1  RI-J: the formula equals the exact sympy derivative of E_J (rationals).
  D2  RI-K with every mode kept: the formula, with F = -1/(l_k l_l), equals the
      exact sympy derivative of E_K (rationals).
  D3  RI-K with dropped modes: the Daleckii-Krein kernel of the Rust code
      (kept/kept, kept/dropped, dropped/dropped) equals a fourth-order finite
      difference of E_K(x) with the spectral cut in a gap; the plain
      -V+ dV V+ misses (negative control).

Mutations (each turns the script red; exact one-occurrence substitutions):
  RI-J metric term -1/2 c c^T loses its 1/2                   (D1 fails)
  kept/dropped kernel entry 1/(lk (lk - ll)) -> 0             (D3 fails)
Exits nonzero on any failed check; prints "FAIL <label>".
"""

import random
import sys

import numpy as np
import sympy as sp

failures = []


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        failures.append(label)


def divided_difference(lk, kk, ll, kl):
    """df_gradient.rs::truncated_inverse_divided_difference."""
    if kk and kl:
        return -1.0 / (lk * ll)
    if kk and not kl:
        return 1.0 / (lk * (lk - ll))
    if kl and not kk:
        return 1.0 / (ll * (ll - lk))
    return 0.0


# ---------------------------------------------------------------------------
# D1, D2: exact rationals
# ---------------------------------------------------------------------------
def rnd_sym(rng, n, scale=1):
    m = sp.Matrix(n, n, lambda i, j: sp.Rational(rng.randint(-3, 3), 4) * scale)
    return (m + m.T) / 2


def exact_checks():
    rng = random.Random(7)
    nao, naux = 2, 3
    x = sp.Symbol("x")
    a0 = [rnd_sym(rng, nao) for _ in range(naux)]
    a1 = [rnd_sym(rng, nao) for _ in range(naux)]
    v0 = sp.Matrix(naux, naux, lambda i, j: sp.Rational(rng.randint(-2, 2), 5))
    v0 = (v0 + v0.T) / 2 + 3 * sp.eye(naux)
    v1 = rnd_sym(rng, naux)
    xf = sp.Matrix([sp.Rational(1, 2), sp.Rational(-1, 3)])
    d = xf * xf.T
    alpha = sp.Rational(-1, 4)

    def a_x(p):
        return a0[p] + x * a1[p]

    vx = v0 + x * v1
    d_p = sp.Matrix(
        [
            sum(d[m, n] * a_x(p)[m, n] for m in range(nao) for n in range(nao))
            for p in range(naux)
        ]
    )
    vinv = vx.inv()
    e_j = sp.Rational(1, 2) * (d_p.T * vinv * d_p)[0]
    de_j = sp.diff(e_j, x).subs(x, 0)

    # formula at x = 0
    d_p0 = d_p.subs(x, 0)
    v0i = v0.inv()
    c = v0i * d_p0
    term3 = sum(
        c[p] * sum(d[m, n] * a1[p][m, n] for m in range(nao) for n in range(nao))
        for p in range(naux)
    )
    term2 = -sp.Rational(1, 2) * (c.T * v1 * c)[0]
    check(
        "D1 RI-J gradient formula equals the exact derivative of 1/2 d^T V^-1 d (claim D1)",
        sp.simplify(term3 + term2 - de_j) == 0,
    )

    # RI-K, one channel, X is 2x1 (m = 1), all modes kept: V+ = V^-1
    def l_vec(xv):
        return sp.Matrix([(xf.T * a_x(p).subs(x, xv) * xf)[0] for p in range(naux)])

    l_x = sp.Matrix([(xf.T * a_x(p) * xf)[0] for p in range(naux)])
    e_k = alpha * (l_x.T * vinv * l_x)[0]
    de_k = sp.diff(e_k, x).subs(x, 0)
    l0 = l_vec(0)
    gam = v0i * l0
    three_c = sum(2 * alpha * gam[p] * (xf.T * a1[p] * xf)[0] for p in range(naux))
    y = alpha * l0 * l0.T
    # U (F o U^T Y U) U^T with F = -1/(l_k l_l) equals -V^-1 Y V^-1
    h_k = -v0i * y * v0i
    two_c = sum(h_k[p, q] * v1[p, q] for p in range(naux) for q in range(naux))
    check(
        "D2 RI-K gradient formula (all modes kept) equals the exact derivative (claim D2)",
        sp.simplify(three_c + two_c - de_k) == 0,
    )


# ---------------------------------------------------------------------------
# D3: truncated metric
# ---------------------------------------------------------------------------
def truncated_inverse(v, cut):
    lam, u = np.linalg.eigh(v)
    w = np.where(lam > cut, 1.0 / np.where(lam > cut, lam, 1.0), 0.0)
    return (u * w) @ u.T


def truncated_checks():
    rng = np.random.default_rng(12)
    naux = 5
    cut = 1e-3
    q, _ = np.linalg.qr(rng.standard_normal((naux, naux)))
    lam0 = np.array([6.0, 3.0, 1.5, 0.4, 2e-6])
    v0 = (q * lam0) @ q.T
    v1 = rng.standard_normal((naux, naux))
    v1 = 0.5 * (v1 + v1.T) * 0.3
    nao = 3
    a0 = rng.standard_normal((naux, nao, nao))
    a1 = rng.standard_normal((naux, nao, nao))
    a0 = 0.5 * (a0 + a0.transpose(0, 2, 1))
    a1 = 0.5 * (a1 + a1.transpose(0, 2, 1))
    xf = rng.standard_normal((nao, 2))
    alpha = -0.25

    def energy(xv):
        a = a0 + xv * a1
        v = v0 + xv * v1
        ell = np.einsum("mi,pmn,nj->pij", xf, a, xf).reshape(naux, -1)
        vp = truncated_inverse(v, cut)
        return alpha * np.sum(ell * (vp @ ell))

    hh = 1e-4
    fd = (8 * (energy(hh) - energy(-hh)) - (energy(2 * hh) - energy(-2 * hh))) / (
        12 * hh
    )

    lam, u = np.linalg.eigh(v0)
    keep = lam > cut
    ell = np.einsum("mi,pmn,nj->pij", xf, a0, xf).reshape(naux, -1)
    lt = u.T @ ell
    w = np.where(keep, 1.0 / np.where(keep, lam, 1.0), 0.0)
    gam = (u * w) @ lt
    three_c = (
        2
        * alpha
        * np.sum(gam.reshape(naux, 2, 2) * np.einsum("mi,pmn,nj->pij", xf, a1, xf))
    )
    y_t = alpha * lt @ lt.T
    f = np.array(
        [
            [
                divided_difference(lam[k], keep[k], lam[ll], keep[ll])
                for ll in range(naux)
            ]
            for k in range(naux)
        ]
    )
    h_k = u @ (f * y_t) @ u.T
    two_c = np.sum(h_k * v1)
    an = three_c + two_c
    # plain -V+ dV V+ (the formula that misses the kept/dropped rotation)
    vp0 = truncated_inverse(v0, cut)
    h_plain = -vp0 @ (alpha * ell @ ell.T) @ vp0
    plain = three_c + np.sum(h_plain * v1)
    err = abs(an - fd)
    miss = abs(plain - fd)
    check(
        "D3 truncated-metric derivative equals FD; the plain formula misses (claim D3)",
        err < 1e-8 and miss > 1e-5,
        f"kernel {err:.1e} plain miss {miss:.1e}",
    )


def main():
    exact_checks()
    truncated_checks()
    if failures:
        print(f"{len(failures)} check(s) failed")
        sys.exit(1)
    print("all DF J/K gradient identities verified")


if __name__ == "__main__":
    main()
