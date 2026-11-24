#!/usr/bin/env python3
"""Identities behind ferric's local MP2 paths: the domain-local fit and the
PNO pair energy.

Derivation: `book/src/13-local-mp2/02-integral-direct.md#integral-direct`, `book/src/13-local-mp2/03-pno-dlpno.md#pno-dlpno` (book
13.2 and 13.3). Code: crates/ferric-mp2/src/lmp2_direct.rs
(fit_domain_group, whiten, gather_vdd) and crates/ferric-mp2/src/dlpno_mp2.rs
(pair_energy_in_pno_basis), crates/ferric-mp2/src/local_pno.rs (pno_for_pair).

Checks (each label carries "(claim X)"):
  L1  Whitening by the Cholesky factor of the domain metric, W = L^-1 A with
      V = L L^T, gives W_i^T W_j = A_i^T V^-1 A_j (symbols, 3x3 metric).
  L2  The Dunlap robust form A_i^T c_j + c_i^T A_j - c_i^T V c_j with
      c = V^-1 A equals the Gram form when one domain and one kernel serve
      the whole pair, so the robust correction cancels (symbols).
  L3  A PNO transform Q with all columns kept, followed by the
      semicanonicalization of the code, reproduces the canonical pair energy;
      keeping only the diagonal f_aa = sum_c Q_ca^2 eps_c instead does not
      (numerics against the canonical sum).
  L4  The pair density T T^T + T^T T is symmetric and positive semidefinite,
      its trace is 2 ||T||_F^2, and its eigenvalues sorted descending give the
      occupation numbers the PNO truncation thresholds (numerics).
  L5  The discarded PNO weight: with P = Q Q^T the projector onto the kept
      eigenvectors of the pair density, the dropped occupations sum to
      ||(1-P) T||_F^2 + ||T (1-P)||_F^2 (numerics, every truncation rank).
  L6  Restricting the fit to an auxiliary domain D can only lower each
      fitted diagonal element: A^T V^-1 A - A_D^T V_DD^-1 A_D = y^T S^-1 y
      with S = V_bb - V_bD V_DD^-1 V_Db and y = A_b - V_bD V_DD^-1 A_D
      (b = outside D), which is positive semidefinite (symbols on a 2+1
      block, numerics on a 4+3 block).

Mutations (each turns the script red; exact one-occurrence substitutions):
  semicanonicalization replaced by the diagonal of f_pno     (L3 fails)
  whitened Gram product W_i^T W_j with W_i^T W_i              (L1 fails)
  second index dropped from the discarded weight               (L5 fails)
  V_bD V_DD^-1 A_D dropped from y                              (L6 fails)
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


# ---------------------------------------------------------------------------
# L1, L2: the domain-local fit
# ---------------------------------------------------------------------------
def whitened_gram(l_chol, a_i, a_j):
    """fit_domain_group: W_k = L^-1 A_k once per occupied, J_ij = W_i^T W_j."""
    w_i = np.linalg.solve(l_chol, a_i)
    w_j = np.linalg.solve(l_chol, a_j)
    return w_i.T @ w_j


def fit_checks():
    n = 3
    lsym = sp.Matrix(n, n, lambda i, j: sp.Symbol(f"l{i}{j}") if i >= j else 0)
    v = lsym * lsym.T
    a_i = sp.Matrix(n, 2, lambda i, j: sp.Symbol(f"a{i}{j}"))
    a_j = sp.Matrix(n, 2, lambda i, j: sp.Symbol(f"b{i}{j}"))
    w_i = lsym.inv() * a_i
    w_j = lsym.inv() * a_j
    gram = a_i.T * v.inv() * a_j
    ok1 = sp.simplify(w_i.T * w_j - gram) == sp.zeros(2, 2)
    rng = np.random.default_rng(2)
    lnum = np.tril(rng.standard_normal((5, 5))) + 3 * np.eye(5)
    vnum = lnum @ lnum.T
    ai = rng.standard_normal((5, 4))
    aj = rng.standard_normal((5, 4))
    ref = ai.T @ np.linalg.inv(vnum) @ aj
    err = np.max(np.abs(whitened_gram(lnum, ai, aj) - ref))
    check(
        "L1 whitened Gram product equals A_i^T V^-1 A_j (claim L1)",
        ok1 and err < 1e-10,
        f"num {err:.1e}",
    )

    c_i = v.inv() * a_i
    c_j = v.inv() * a_j
    robust = a_i.T * c_j + c_i.T * a_j - c_i.T * v * c_j
    check(
        "L2 robust fit equals the Gram form in one shared domain (claim L2)",
        sp.simplify(robust - gram) == sp.zeros(2, 2),
    )


# ---------------------------------------------------------------------------
# L3, L4: the PNO pair energy
# ---------------------------------------------------------------------------
def pair_energy_pno(g_ij, g_ji, q, e_i, e_j, eps_v, fac):
    """dlpno_mp2.rs::pair_energy_in_pno_basis, transcribed."""
    t_ij = q.T @ g_ij @ q
    t_ji = q.T @ g_ji @ q
    npno = q.shape[1]
    f_pno = np.zeros((npno, npno))
    for a in range(npno):
        for b in range(npno):
            f_pno[a, b] = np.sum(q[:, a] * q[:, b] * eps_v)
    e_pno, u = np.linalg.eigh(f_pno)
    t_ij = u.T @ t_ij @ u
    t_ji = u.T @ t_ji @ u
    e_os = e_ss = 0.0
    for a in range(npno):
        for b in range(npno):
            d = e_i + e_j - e_pno[a] - e_pno[b]
            iajb = t_ij[a, b]
            ibja = t_ji[a, b]
            e_os += fac * iajb * iajb / d
            e_ss += fac * iajb * (iajb - ibja) / d
    return e_os, e_ss


def pair_energy_diag_shortcut(g_ij, g_ji, q, e_i, e_j, eps_v, fac):
    """The wrong construction: only the diagonal of f in the PNO basis."""
    t_ij = q.T @ g_ij @ q
    t_ji = q.T @ g_ji @ q
    npno = q.shape[1]
    e_pno = np.array([np.sum(q[:, a] ** 2 * eps_v) for a in range(npno)])
    e_os = e_ss = 0.0
    for a in range(npno):
        for b in range(npno):
            d = e_i + e_j - e_pno[a] - e_pno[b]
            e_os += fac * t_ij[a, b] ** 2 / d
            e_ss += fac * t_ij[a, b] * (t_ij[a, b] - t_ji[a, b]) / d
    return e_os, e_ss


def canonical_pair(g_ij, e_i, e_j, eps_v, fac):
    e_os = e_ss = 0.0
    nv = len(eps_v)
    for a in range(nv):
        for b in range(nv):
            d = e_i + e_j - eps_v[a] - eps_v[b]
            e_os += fac * g_ij[a, b] ** 2 / d
            e_ss += fac * g_ij[a, b] * (g_ij[a, b] - g_ij[b, a]) / d
    return e_os, e_ss


def pno_checks():
    rng = np.random.default_rng(8)
    nv = 6
    eps_v = np.sort(rng.uniform(0.3, 2.0, nv))
    e_i, e_j = -0.6, -0.45
    g = 0.1 * rng.standard_normal((nv, nv))
    g_ji = g.T
    q, _ = np.linalg.qr(rng.standard_normal((nv, nv)))
    ref = canonical_pair(g, e_i, e_j, eps_v, 2.0)
    got = pair_energy_pno(g, g_ji, q, e_i, e_j, eps_v, 2.0)
    bad = pair_energy_diag_shortcut(g, g_ji, q, e_i, e_j, eps_v, 2.0)
    err = max(abs(got[0] - ref[0]), abs(got[1] - ref[1]))
    miss = max(abs(bad[0] - ref[0]), abs(bad[1] - ref[1]))
    check(
        "L3 full-rank PNO with semicanonicalization is the canonical pair energy (claim L3)",
        err < 1e-12 and miss > 1e-4,
        f"err {err:.1e} diagonal-shortcut miss {miss:.1e}",
    )

    t = rng.standard_normal((nv, nv))
    d = t @ t.T + t.T @ t
    evals = np.linalg.eigvalsh(d)
    ok = (
        np.max(np.abs(d - d.T)) < 1e-14
        and evals.min() > -1e-12
        and abs(np.trace(d) - 2 * np.sum(t * t)) < 1e-10
    )
    check(
        "L4 pair density is symmetric, positive semidefinite, trace 2 ||T||_F^2 (claim L4)",
        ok,
    )


def discarded_weight_checks():
    rng = np.random.default_rng(11)
    nv = 7
    t = rng.standard_normal((nv, nv))
    d = t @ t.T + t.T @ t
    evals, evecs = np.linalg.eigh(d)
    order = np.argsort(evals)[::-1]
    worst = 0.0
    for nkeep in range(nv + 1):
        q = evecs[:, order[:nkeep]]
        comp = np.eye(nv) - q @ q.T
        dropped = float(np.sum(evals[order[nkeep:]]))
        norm = discarded_norm(t, comp)
        worst = max(worst, abs(dropped - norm) / max(1.0, abs(dropped)))
    check(
        "L5 dropped PNO occupations equal the amplitude norm outside the kept space (claim L5)",
        worst < 1e-12,
        f"max rel err {worst:.1e}",
    )


def discarded_norm(t, comp):
    """||(1-P) T||_F^2 + ||T (1-P)||_F^2."""
    return float(np.sum((comp @ t) ** 2) + np.sum((t @ comp) ** 2))


def aux_restriction_residual(v, a, nd):
    """y and S of the block elimination of V over the domain D = [:nd]."""
    v_dd, v_db, v_bd, v_bb = v[:nd, :nd], v[:nd, nd:], v[nd:, :nd], v[nd:, nd:]
    a_d, a_b = a[:nd, :], a[nd:, :]
    y = a_b - v_bd * v_dd.inv() * a_d
    s_schur = v_bb - v_bd * v_dd.inv() * v_db
    return y, s_schur


def aux_restriction_checks():
    # symbols: generic symmetric positive 3x3 metric, D = first two functions
    v = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"v{min(i, j)}{max(i, j)}"))
    a = sp.Matrix(3, 1, lambda i, j: sp.Symbol(f"a{i}"))
    y, s_schur = aux_restriction_residual(v, a, 2)
    lhs = (a.T * v.inv() * a)[0] - (a[:2, :].T * v[:2, :2].inv() * a[:2, :])[0]
    rhs = (y.T * s_schur.inv() * y)[0]
    ok_sym = sp.simplify(lhs - rhs) == 0
    # numerics: 4+3 block, two densities
    rng = np.random.default_rng(12)
    m = rng.standard_normal((7, 7))
    vnum = sp.Matrix(m @ m.T + 7 * np.eye(7))
    anum = sp.Matrix(rng.standard_normal((7, 2)))
    yn, sn = aux_restriction_residual(vnum, anum, 4)
    full = anum.T * vnum.inv() * anum
    dom = anum[:4, :].T * vnum[:4, :4].inv() * anum[:4, :]
    diff = np.array((full - dom - yn.T * sn.inv() * yn).evalf(), dtype=float)
    gap = np.array((full - dom).evalf(), dtype=float)
    ok_num = np.max(np.abs(diff)) < 1e-10 and np.linalg.eigvalsh(gap).min() > -1e-12
    check(
        "L6 domain restriction lowers each fitted diagonal: gap = y^T S^-1 y (claim L6)",
        ok_sym and ok_num,
        f"num {np.max(np.abs(diff)):.1e}",
    )


def main():
    fit_checks()
    pno_checks()
    discarded_weight_checks()
    aux_restriction_checks()
    if failures:
        print(f"{len(failures)} check(s) failed")
        sys.exit(1)
    print("all local-MP2 fit and PNO identities verified")


if __name__ == "__main__":
    main()
