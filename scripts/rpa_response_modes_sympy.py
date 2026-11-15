#!/usr/bin/env python3
"""Proofs: the molecular RPA polarizability from the dielectric modes (ferric-rpa).

Code transcribed here (closed shell, auxiliary index first, `b_ov[P, ia]`,
`Delta_ia = e_a - e_i`, `g_ia(u) = Delta_ia/(Delta_ia^2 + u^2)`, K = b_ov^T b_ov):

  * properties.rs molecular_dynamic_polarizability and pdep_polarizability_static:
        eps~(iu) = 1 + 4 b_ov diag(g) b_ov^T,   w_d = b_ov (g o mu_d)
        alpha_de(iu) = 4 mu_d^T diag(g) mu_e - 16 w_d^T eps~^-1 w_e
  * response_modes.rs ResponseModes::polarizability_from_modes:
        alpha(iu) = alpha_bare - sum_xi s_xi s_xi^T / lambda_xi,
        alpha_bare = 4 mu^T diag(g) mu,  s_xi,d = 4 U_xi . w_d,
        (lambda_xi, U_xi) the eigenpairs of eps~(iu)

Checked (claim names are the check labels):
  M1  the dielectric form equals an independent direct solve of the dRPA
      response, alpha(iu) = 4 mu^T [Delta (Delta + 4 K) + u^2]^-1 Delta mu
      (A - B = Delta, A + B = Delta + 4 K, the form of theory.md section 12 with
      W = 0), at u = 0 and u = 0.9, for the full tensor.
  M2  polarizability_from_modes with every mode kept equals the dielectric
      form; keeping fewer modes does not (negative control: the truncated
      tensor differs by sum over the dropped modes of s s^T / lambda).
  M3  the static limit: g = 1/Delta at u = 0 and the u -> 0 limit of the
      dynamic form agree.
  M4  open shell (properties.rs molecular_dynamic_polarizability, (U) branch):
      eps~_U = 1 + 2 sum_s b_s diag(g_s) b_s^T, w = sum_s b_s (g_s o mu_s),
      alpha = sum_s 2 mu_s^T diag(g_s) mu_s - 4 w^T eps~_U^-1 w. For unequal
      alpha and beta spaces (2 occ/4 vir and 1 occ/5 vir) it equals an
      independent direct solve of the spin-orbital dRPA response,
      alpha = 2 mu_so^T [Delta (Delta + 2 K_so) + u^2]^-1 Delta mu_so with K_so
      the Coulomb coupling between every spin pair, at u = 0 and 0.9
      (negative control: leaving the beta channel out of w misses).

Mutations (the runner substitutes the code line once; each exits 1):
  the screening prefactor 16 -> 4                 -> M1 fails
  the mode-sum sign + instead of -                -> M2 fails
  the open-shell screening prefactor 4 -> 16      -> M4 fails

Run: python3 scripts/rpa_response_modes_sympy.py   (needs numpy)
"""

import sys

import numpy as np

FAILS = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} [{name}] {detail}")
    if not ok:
        FAILS.append(name)


def maxabs(x):
    return float(np.max(np.abs(x)))


rng = np.random.default_rng(20261118)
no, nv, naux = 2, 4, 7
eo = np.array([-1.1, -0.7])
ev = np.array([0.3, 0.6, 1.1, 1.8])
gaps = (ev[None, :] - eo[:, None]).ravel()
nov = no * nv
b_ov = 0.3 * rng.standard_normal((naux, nov))
mu = 0.5 * rng.standard_normal((3, nov))


def g_of(u):
    return gaps / (u * u + gaps * gaps)


def dielectric_form(u):
    g = g_of(u)
    eps = np.eye(naux) + 4.0 * (b_ov * g) @ b_ov.T
    w = np.array([b_ov @ (mu[d] * g) for d in range(3)])
    bare = 4.0 * (mu * g) @ mu.T
    return bare - 16.0 * w @ np.linalg.solve(eps, w.T), eps, w, bare


def direct_form(u):
    k = b_ov.T @ b_ov
    apb = np.diag(gaps) + 4.0 * k
    amb = np.diag(gaps)
    sysm = amb @ apb + u * u * np.eye(nov)
    return 4.0 * mu @ np.linalg.solve(sysm, amb @ mu.T)


def modes_form(u, keep):
    _, eps, w, bare = dielectric_form(u)
    lam, vec = np.linalg.eigh(eps)
    order = np.argsort(lam)[::-1][:keep]  # leading (largest) modes first
    out = bare.copy()
    for k in order:
        s = 4.0 * (w @ vec[:, k])
        out -= np.outer(s, s) / lam[k]
    return out


for u in (0.0, 0.9):
    alpha, _, _, _ = dielectric_form(u)
    d = maxabs(alpha - direct_form(u))
    check(
        "M1",
        d < 1e-12,
        f"u = {u}: dielectric form = direct dRPA solve, max |d| = {d:.1e}",
    )
    full = modes_form(u, naux)
    d2 = maxabs(full - alpha)
    check(
        "M2",
        d2 < 1e-12,
        f"u = {u}: all modes kept = dielectric form, max |d| = {d2:.1e}",
    )
trunc = modes_form(0.0, 3)
check(
    "M2",
    maxabs(trunc - dielectric_form(0.0)[0]) > 1e-4,
    "negative control: 3 of 7 modes differs",
)
st = dielectric_form(0.0)[0]
lim = dielectric_form(1e-7)[0]
check(
    "M3",
    maxabs(st - lim) < 1e-10,
    f"static g = 1/Delta = the u -> 0 limit: {maxabs(st - lim):.1e}",
)

# ---- M4 open shell, unequal alpha and beta spaces
nob, nvb = 1, 5
eo_b = np.array([-0.9])
ev_b = np.array([0.2, 0.5, 0.9, 1.4, 2.0])
gaps_b = (ev_b[None, :] - eo_b[:, None]).ravel()
b_b = 0.3 * rng.standard_normal((naux, nob * nvb))
mu_b = 0.5 * rng.standard_normal((3, nob * nvb))


def open_shell(u):
    ga, gb = gaps / (u * u + gaps * gaps), gaps_b / (u * u + gaps_b * gaps_b)
    eps = np.eye(naux) + 2.0 * (b_ov * ga) @ b_ov.T + 2.0 * (b_b * gb) @ b_b.T
    w_t = np.array([b_ov @ (mu[d] * ga) + b_b @ (mu_b[d] * gb) for d in range(3)])
    bare = 2.0 * (mu * ga) @ mu.T + 2.0 * (mu_b * gb) @ mu_b.T
    return bare - 4.0 * w_t @ np.linalg.solve(eps, w_t.T), w_t, eps


def open_shell_direct(u):
    b_so = np.concatenate([b_ov, b_b], axis=1)
    mu_so = np.concatenate([mu, mu_b], axis=1)
    gap_so = np.concatenate([gaps, gaps_b])
    k_so = b_so.T @ b_so
    amb = np.diag(gap_so)
    apb = amb + 2.0 * k_so
    sysm = amb @ apb + u * u * np.eye(len(gap_so))
    return 2.0 * mu_so @ np.linalg.solve(sysm, amb @ mu_so.T)


for u in (0.0, 0.9):
    d = maxabs(open_shell(u)[0] - open_shell_direct(u))
    check(
        "M4",
        d < 1e-12,
        f"u = {u}: open-shell dielectric form = spin-orbital direct solve, max |d| = {d:.1e}",
    )
_, w_t, eps_u = open_shell(0.0)
alpha_bad = (
    2.0 * (mu * g_of(0.0)) @ mu.T
    + 2.0 * (mu_b * (gaps_b / gaps_b**2)) @ mu_b.T
    - 4.0
    * (w_t - np.array([b_b @ (mu_b[d] / gaps_b) for d in range(3)]))
    @ np.linalg.solve(
        eps_u, (w_t - np.array([b_b @ (mu_b[d] / gaps_b) for d in range(3)])).T
    )
)
check(
    "M4",
    maxabs(alpha_bad - open_shell_direct(0.0)) > 1e-4,
    "negative control: leaving the beta channel out of w misses",
)

print(f"{len(FAILS)} failed: {FAILS}" if FAILS else "all identities hold")
sys.exit(1 if FAILS else 0)
