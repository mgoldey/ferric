#!/usr/bin/env python3
"""Factored RI-K of the open-shell CPHF response densities (UHF, UKS, ROHF).

Derivation: `book/src/14-derivatives/04-hf-hessian.md#hf-hessian-factored-ri-k`. Code: crates/ferric-scf/src/hessian.rs (TwoElectronBuilder::
uhf_fitted_exchanges, spin_factors, solve_ucphf_blocks_of),
crates/ferric-scf/src/hessian/rohf.rs (Frame::spin_factors, Response::g_of)
and crates/ferric-scf/src/df_k/response.rs (DfK::build_factored_many).

The fitted exchange is K[D] = sum_P B_P D B_P with every B_P symmetric. The
claim is that each spin's CPHF response density is D = X C_o^T + C_o X^T for
the spin's occupied block C_o, so K[D] = G + G^T, G = sum_P (B_P X)(B_P C_o)^T,
and which X and C_o each route passes.

Independent route for the numerical checks: the four-index exchange
K_mn = sum_ls (ml|ns) D_ls with (ml|ns) = sum_P B_P,ml B_P,ns, contracted by
einsum over the explicit four-index tensor (never through B_P D B_P).

Checks:
  F1  (sympy) K[X C_o^T + C_o X^T] = G + G^T for symbolic symmetric B_P and
      symbolic X, C_o (n = 3, o = 2, two auxiliary functions).
  F2  UHF / UKS: per spin, the rotation density C_v U C_o^T + h.c. (X = C_v U)
      and the occupied-block density -C_o S_oo C_o^T (X = -1/2 C_o S_oo, S_oo
      symmetric) give K_fac = the four-index K, alpha and beta with their own
      C_o (n_alpha != n_beta, non-orthonormal MOs).
  F3  (sympy) ROHF: C (U n + n U^T) C^T = X C_o^T + C_o X^T with
      C_o = C[:, :n_s], X = (C U)[:, :n_s] for a symbolic generator U; the
      kappa density (n_q - n_p) kappa_pq is the case U = kappa (antisymmetric)
      and the overlap density -1/2 S_pq (n_p + n_q) the case U = -S/2
      (symmetric).
  F4  ROHF numerics: for both spins, the kappa, overlap and total (kappa - S/2)
      densities through X = (C U)[:, :n_s] give K_fac = the four-index K.
  F5  Closed-shell limit: C_a = C_b, U_a = U_b: v_s = J(D1_a + D1_b) - K(D1_s)
      through the factored per-spin K equals the RHF G = J(D) - K(D)/2 of the
      RHF factored route (occ = 2, X = 2 C_v U).

Mutations (each turns the script red; exact one-occurrence substitutions):
  drop the h.c. term: return g + g.T -> return g          (F1, F2, F4, F5 fail)
  beta exchange from the alpha occupied block              (F2 fails)
  ROHF factor from U^T instead of U                        (F3, F4 fail)
Exits nonzero on any failed check; prints "FAIL <label>".
"""

import sys

import numpy as np
import sympy as sp

failures = []


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        failures.append(label)


def k_factored(bs, x, c_o):
    """DfK::build_factored_many: G = sum_P (B_P X)(B_P C_o)^T, K = G + G^T."""
    g = sum((b @ x @ (b @ c_o).T for b in bs[1:]), bs[0] @ x @ (bs[0] @ c_o).T)
    return g + g.T


def k_four_index(bs, d):
    """K_mn = sum_ls (ml|ns) D_ls over the explicit four-index tensor."""
    b = np.array(bs)
    eri = np.einsum("pml,pns->mlns", b, b)
    return np.einsum("mlns,ls->mn", eri, d)


def rel(a, b):
    return np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300)


# Bar of the numerical checks: both routes form sums of at most
# L = n^2 + naux * n terms over O(1) entries; 1e-12 relative is ~100 L eps.
TOL = 1e-12


def f1_symbolic():
    n, o = 3, 2
    bs = []
    for p in range(2):
        b = sp.Matrix(n, n, lambda i, j: sp.Symbol(f"b{p}_{min(i, j)}{max(i, j)}"))
        bs.append(b)
    x = sp.Matrix(n, o, lambda i, j: sp.Symbol(f"x{i}{j}"))
    c = sp.Matrix(n, o, lambda i, j: sp.Symbol(f"c{i}{j}"))
    d = x * c.T + c * x.T
    k_density = sum((b * d * b for b in bs[1:]), bs[0] * d * bs[0])
    diff = (k_factored(bs, x, c) - k_density).applyfunc(sp.expand)
    check("[F1] K[X C_o^T + C_o X^T] = G + G^T (symbolic)", diff == sp.zeros(n, n))


rng = np.random.default_rng(20261115)
N, NAUX = 9, 14
BS = []
for _ in range(NAUX):
    m = rng.standard_normal((N, N))
    BS.append(0.5 * (m + m.T))


def mo_set(nocc):
    """A non-orthonormal (n x n) MO set split into occupied / virtual."""
    c = rng.standard_normal((N, N))
    return c[:, :nocc], c[:, nocc:]


def f2_uhf():
    na, nb = 4, 2
    ca_o, ca_v = mo_set(na)
    cb_o, cb_v = mo_set(nb)
    # Rotation densities, one generator per spin.
    ua = rng.standard_normal((N - na, na))
    ub = rng.standard_normal((N - nb, nb))
    xa, xb = ca_v @ ua, cb_v @ ub
    da = xa @ ca_o.T + ca_o @ xa.T
    db = xb @ cb_o.T + cb_o @ xb.T
    ka = k_factored(BS, xa, ca_o)
    kb = k_factored(BS, xb, cb_o)
    worst = max(rel(ka, k_four_index(BS, da)), rel(kb, k_four_index(BS, db)))
    # Occupied-block densities -C_o S_oo C_o^T, X = -1/2 C_o S_oo.
    for c_o, k in [(ca_o, na), (cb_o, nb)]:
        s = rng.standard_normal((k, k))
        s = 0.5 * (s + s.T)
        d_oo = -c_o @ s @ c_o.T
        k_oo = k_factored(BS, -0.5 * c_o @ s, c_o)
        worst = max(worst, rel(k_oo, k_four_index(BS, d_oo)))
    check(
        "[F2] UHF per-spin factored K = four-index K (rotation + occupied block)",
        worst < TOL,
        f"rel {worst:.1e}",
    )


def rohf_factor(c, u, k):
    """Frame::spin_factors: X_s = (C U)[:, :n_s]."""
    return (c @ u)[:, :k]


def f3_symbolic():
    m, k = 4, 2
    c = sp.Matrix(m, m, lambda i, j: sp.Symbol(f"C{i}{j}"))
    u = sp.Matrix(m, m, lambda i, j: sp.Symbol(f"u{i}{j}"))
    nv = sp.diag(*[1 if p < k else 0 for p in range(m)])
    lhs = c * (u * nv + nv * u.T) * c.T
    x = rohf_factor(c, u, k)
    c_o = c[:, :k]
    diff = (lhs - (x * c_o.T + c_o * x.T)).applyfunc(sp.expand)
    ok = diff == sp.zeros(m, m)
    # The two specializations the response uses.
    kap = sp.Matrix(
        m, m, lambda i, j: sp.Symbol(f"k{min(i, j)}{max(i, j)}") * int(np.sign(j - i))
    )
    s = sp.Matrix(m, m, lambda i, j: sp.Symbol(f"s{min(i, j)}{max(i, j)}"))
    occ = [1 if p < k else 0 for p in range(m)]
    dk = sp.Matrix(m, m, lambda p, q: (occ[q] - occ[p]) * kap[p, q])
    ds = sp.Matrix(m, m, lambda p, q: -sp.Rational(1, 2) * s[p, q] * (occ[p] + occ[q]))
    ok &= (kap * nv + nv * kap.T - dk).applyfunc(sp.expand) == sp.zeros(m, m)
    hs = -s / 2
    ok &= (hs * nv + nv * hs.T - ds).applyfunc(sp.expand) == sp.zeros(m, m)
    check(
        "[F3] ROHF C (U n + n U^T) C^T = X C_o^T + C_o X^T, X = (C U)[:, :n] (symbolic)",
        ok,
    )


def f4_rohf():
    na, nb = 5, 3
    c = rng.standard_normal((N, N))
    kap = np.zeros((N, N))
    # vc, vo, oc blocks (closed 0..nb, open nb..na, virtual na..).
    for p, q in [(p, q) for p in range(na, N) for q in range(na)] + [
        (p, q) for p in range(nb, na) for q in range(nb)
    ]:
        kap[p, q] = rng.standard_normal()
        kap[q, p] = -kap[p, q]
    s = rng.standard_normal((N, N))
    s = 0.5 * (s + s.T)
    worst = 0.0
    for nocc_s in (na, nb):
        occ = np.array([1.0 if p < nocc_s else 0.0 for p in range(N)])
        dk = (occ[None, :] - occ[:, None]) * kap
        ds = -0.5 * s * (occ[:, None] + occ[None, :])
        for u, d_mo in [(kap, dk), (-0.5 * s, ds), (kap - 0.5 * s, dk + ds)]:
            d_ao = c @ d_mo @ c.T
            x = rohf_factor(c, u, nocc_s)
            k_fac = k_factored(BS, x, c[:, :nocc_s])
            worst = max(worst, rel(k_fac, k_four_index(BS, d_ao)))
    check(
        "[F4] ROHF kappa / overlap / total densities, both spins: factored K = four-index K",
        worst < TOL,
        f"rel {worst:.1e}",
    )


def f5_closed_shell_limit():
    no = 3
    c_o, c_v = mo_set(no)
    u = rng.standard_normal((N - no, no))
    x_s = c_v @ u
    d1 = x_s @ c_o.T + c_o @ x_s.T
    j = lambda d: np.einsum("pmn,pls,ls->mn", np.array(BS), np.array(BS), d)  # noqa: E731
    k_s = k_factored(BS, x_s, c_o)
    v_uhf = j(2.0 * d1) - k_s
    x_r = 2.0 * c_v @ u
    d_r = x_r @ c_o.T + c_o @ x_r.T
    g_rhf = j(d_r) - 0.5 * k_factored(BS, x_r, c_o)
    e = rel(v_uhf, g_rhf)
    e2 = rel(g_rhf, j(d_r) - 0.5 * k_four_index(BS, d_r))
    check(
        "[F5] closed-shell limit: UHF factored v_s = RHF factored G = four-index G",
        max(e, e2) < TOL,
        f"rel {max(e, e2):.1e}",
    )


if __name__ == "__main__":
    f1_symbolic()
    f2_uhf()
    f3_symbolic()
    f4_rohf()
    f5_closed_shell_limit()
    if failures:
        sys.exit(1)
