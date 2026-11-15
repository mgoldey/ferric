#!/usr/bin/env python3
"""Step formulas of ferric's geometry searches: BFGS, P-RFO, Bofill, IRC.

Derivation: book/src/14-derivatives/08-geometry-search.md (no theory.md section
yet). Code: crates/ferric-scf/src/optimize.rs (the inverse-Hessian update of
optimize_coordinates and run_bfgs_internal), crates/ferric-scf/src/saddle.rs
(prfo_step, bofill_update) and crates/ferric-scf/src/irc.rs (walk). Each
transcription below mirrors the Rust loop line by line.

Independent routes: the BFGS update is compared with the direct update of the
Hessian inverted numerically; the P-RFO step with the eigenvector of the
augmented Hessian [[B, g], [g^T, 0]] from numpy.linalg.eigh; the Bofill update
with the two secant conditions in sympy; the IRC step with the chain rule in
sympy.

Checks (each label carries "(claim X)"):
  B1  BFGS inverse update satisfies the secant condition H' y = s, in symbols.
  B2  It equals the inverse of the direct BFGS update of B = H^-1, is
      symmetric, and stays positive definite when y.s > 0 (numerics).
  P1  The followed-mode shift lambda_p = (b + sqrt(b^2 + 4 g^2))/2 is the
      LARGEST root of lambda^2 - b lambda - g^2 = 0, the secular equation of
      the 2x2 augmented Hessian, and dq = -g/(b - lambda) (symbols).
  P2  prfo_step equals the augmented-Hessian solution of each partition:
      largest eigenpair for the followed mode, smallest for the rest (numerics,
      several random spectra with one negative eigenvalue).
  F1  Murtagh-Sargent and Powell-symmetric-Broyden corrections each satisfy
      the secant condition, so any mixture phi does (symbols).
  F2  bofill_update satisfies H' dx = dg and is symmetric, and keeps a
      negative eigenvalue that BFGS removes (numerics).
  I1  The mass-weighted gradient is dE/dq = (dE/dx)/sqrt(m), so the downhill
      step of length l in q is dx_i = -l g_i /(m_i |g_q|), which differs from
      unweighted steepest descent unless all masses are equal.

Mutations (each turns the script red; exact one-occurrence substitutions):
  BFGS term1 = (ys + yhy) rho^2 -> yhy rho^2                 (B2 fails)
  P-RFO followed shift: + sqrt -> - sqrt                      (P2 fails)
  Bofill PSB projector term dropped                          (F2 fails)
  IRC Cartesian step: divide by m, not sqrt(m)               (I1 fails)
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
# Transcriptions of the Rust loops.
# ---------------------------------------------------------------------------
def bfgs_inverse_update(h, s, y):
    """optimize.rs, the `ys.abs() > 1e-12` branch: H <- H + term1 s s^T
    - rho (H y s^T + s y^T H)."""
    ys = y @ s
    hy = h @ y
    yhy = y @ hy
    rho = 1.0 / ys
    term1 = (ys + yhy) * rho * rho
    return h + term1 * np.outer(s, s) - rho * (np.outer(hy, s) + np.outer(s, hy))


def prfo_step(g_q, evals, follow):
    """saddle.rs::prfo_step."""
    n = len(g_q)
    dq = np.zeros(n)
    b_p = evals[follow]
    g_p = g_q[follow]
    lam_p = 0.5 * (b_p + np.sqrt(b_p * b_p + 4.0 * g_p * g_p))
    den_p = b_p - lam_p
    dq[follow] = -g_p / den_p if abs(den_p) > 1e-14 else 0.0
    others = [i for i in range(n) if i != follow]
    b_min = min(evals[i] for i in others)
    lam_n = min(b_min - 1.0, 0.0)
    for _ in range(100):
        f = df = 0.0
        for i in others:
            d = lam_n - evals[i]
            if abs(d) < 1e-14:
                continue
            f += g_q[i] ** 2 / d
            df -= g_q[i] ** 2 / (d * d)
        h = lam_n - f
        dh = 1.0 - df
        if abs(dh) < 1e-14:
            break
        step = h / dh
        lam_n -= step
        if abs(step) < 1e-12:
            break
    if lam_n >= b_min:
        lam_n = b_min - 1.0e-3
    for i in others:
        den = evals[i] - lam_n
        if abs(den) > 1e-14:
            dq[i] = -g_q[i] / den
    return dq


def bofill_update(h, dx, dg):
    """saddle.rs::bofill_update."""
    n = len(dx)
    e = dg - h @ dx
    dx_dx = dx @ dx
    if dx_dx < 1e-20:
        return h.copy()
    e_dx = e @ dx
    e_e = e @ e
    phi = (e_dx * e_dx) / (e_e * dx_dx) if e_e * dx_dx > 1e-20 else 0.0
    out = h.copy()
    if phi > 0.0 and abs(e_dx) > 1e-14:
        out += phi * np.outer(e, e) / e_dx
    if phi < 1.0:
        w = 1.0 - phi
        for i in range(n):
            for j in range(n):
                out[i, j] += w * (
                    (e[i] * dx[j] + dx[i] * e[j]) / dx_dx
                    - e_dx * dx[i] * dx[j] / (dx_dx * dx_dx)
                )
    return 0.5 * (out + out.T)


def irc_cartesian_step(g, masses, length):
    """irc.rs::walk: q = sqrt(m) x, gq = g/sqrt(m), q -= len gq/|gq|, x = q/sqrt(m).
    Returns the Cartesian displacement."""
    sm = np.sqrt(masses)
    gq = g / sm
    gn = np.linalg.norm(gq)
    dq = -length * gq / gn
    return dq / sm


def irc_cartesian_step_closed_form(g, masses, length):
    sm = np.sqrt(masses)
    gn = np.linalg.norm(g / sm)
    return -length * g / (masses * gn)


# ---------------------------------------------------------------------------
# B: BFGS
# ---------------------------------------------------------------------------
def bfgs_checks():
    n = 3
    hs = sp.Matrix(n, n, lambda i, j: sp.Symbol(f"h{min(i, j)}{max(i, j)}"))
    s = sp.Matrix(sp.symbols("s0:3"))
    y = sp.Matrix(sp.symbols("y0:3"))
    ys = (y.T * s)[0]
    hy = hs * y
    yhy = (y.T * hy)[0]
    rho = 1 / ys
    term1 = (ys + yhy) / ys**2
    h_new = hs + term1 * s * s.T - rho * (hy * s.T + s * hy.T)
    resid = sp.simplify(h_new * y - s)
    check(
        "B1 BFGS inverse update: H' y = s (symbols) (claim B1)", resid == sp.zeros(n, 1)
    )

    rng = np.random.default_rng(11)
    worst_inv = worst_sym = 0.0
    pd_ok = True
    for _ in range(20):
        m = rng.standard_normal((n, n))
        h = m @ m.T + 0.5 * np.eye(n)
        s_v = rng.standard_normal(n)
        y_v = rng.standard_normal(n)
        if y_v @ s_v < 0:
            y_v = -y_v
        h_new = bfgs_inverse_update(h, s_v, y_v)
        b = np.linalg.inv(h)
        bs = b @ s_v
        b_new = b - np.outer(bs, bs) / (s_v @ bs) + np.outer(y_v, y_v) / (y_v @ s_v)
        worst_inv = max(worst_inv, np.max(np.abs(h_new - np.linalg.inv(b_new))))
        worst_sym = max(worst_sym, np.max(np.abs(h_new - h_new.T)))
        pd_ok = pd_ok and np.min(np.linalg.eigvalsh(h_new)) > 0
    check(
        "B2 inverse of the direct BFGS update, symmetric, positive definite (claim B2)",
        worst_inv < 1e-9 and worst_sym < 1e-12 and pd_ok,
        f"inv {worst_inv:.1e} sym {worst_sym:.1e}",
    )


# ---------------------------------------------------------------------------
# P: P-RFO
# ---------------------------------------------------------------------------
def prfo_checks():
    b, g, lam = sp.symbols("b g lam", real=True)
    lam_p = (b + sp.sqrt(b**2 + 4 * g**2)) / 2
    lam_m = (b - sp.sqrt(b**2 + 4 * g**2)) / 2

    def secular(v):
        return sp.expand(v**2 - b * v - g**2)

    roots_ok = sp.simplify(secular(lam_p)) == 0 and sp.simplify(secular(lam_m)) == 0
    # the augmented Hessian [[b, g], [g, 0]] has char poly lam^2 - b lam - g^2
    aug = sp.Matrix([[b, g], [g, 0]])
    cp = sp.expand((aug - lam * sp.eye(2)).det() - (lam**2 - b * lam - g**2))
    larger = sp.simplify(lam_p - lam_m) == sp.sqrt(b**2 + 4 * g**2)
    # dq = -g/(b - lam_p) solves (b - lam) dq + g = 0 and lam = g dq
    dq = -g / (b - lam_p)
    rfo_ok = sp.simplify((b - lam_p) * dq + g) == 0 and sp.simplify(lam_p - g * dq) == 0
    check(
        "P1 followed-mode shift is the largest root of the 2x2 secular equation (claim P1)",
        roots_ok and cp == 0 and larger and rfo_ok,
    )

    rng = np.random.default_rng(5)
    worst = 0.0
    for trial in range(8):
        n = 5
        evals = np.sort(rng.uniform(0.1, 1.5, n))
        evals[0] = -rng.uniform(0.05, 0.6)
        g_q = rng.uniform(-0.05, 0.05, n)
        follow = 0
        dq = prfo_step(g_q, evals, follow)
        # followed mode: largest eigenpair of [[b, g], [g, 0]]
        a_p = np.array([[evals[0], g_q[0]], [g_q[0], 0.0]])
        w, v = np.linalg.eigh(a_p)
        ref_p = v[0, -1] / v[1, -1]
        # the rest: smallest eigenpair of [[diag(b_i), g_i], [g_i^T, 0]]
        idx = [i for i in range(n) if i != follow]
        m = len(idx)
        a_n = np.zeros((m + 1, m + 1))
        a_n[:m, :m] = np.diag(evals[idx])
        a_n[:m, m] = g_q[idx]
        a_n[m, :m] = g_q[idx]
        w2, v2 = np.linalg.eigh(a_n)
        ref_n = v2[:m, 0] / v2[m, 0]
        worst = max(worst, abs(dq[0] - ref_p), float(np.max(np.abs(dq[idx] - ref_n))))
    check(
        "P2 prfo_step equals the augmented-Hessian eigenvectors of each partition (claim P2)",
        worst < 1e-9,
        f"max err {worst:.1e}",
    )


# ---------------------------------------------------------------------------
# F: Bofill
# ---------------------------------------------------------------------------
def bofill_checks():
    n = 3
    e = sp.Matrix(sp.symbols("e0:3"))
    dx = sp.Matrix(sp.symbols("d0:3"))
    ed = (e.T * dx)[0]
    dd = (dx.T * dx)[0]
    ms = e * e.T / ed
    psb = (e * dx.T + dx * e.T) / dd - ed * dx * dx.T / (dd * dd)
    ok = sp.simplify(ms * dx - e) == sp.zeros(n, 1) and sp.simplify(
        psb * dx - e
    ) == sp.zeros(n, 1)
    check("F1 MS and PSB corrections each satisfy the secant condition (claim F1)", ok)

    rng = np.random.default_rng(3)
    worst_sec = worst_sym = 0.0
    kept = True
    for _ in range(20):
        m = rng.standard_normal((n, n))
        h = 0.5 * (m + m.T)
        dx_v = 0.1 * rng.standard_normal(n)
        dg_v = rng.standard_normal(n)
        h_new = bofill_update(h, dx_v, dg_v)
        worst_sec = max(worst_sec, np.max(np.abs(h_new @ dx_v - dg_v)))
        worst_sym = max(worst_sym, np.max(np.abs(h_new - h_new.T)))
    # an indefinite Hessian whose update is consistent with it keeps its
    # negative eigenvalue; BFGS (inverse, positive definite) cannot
    h0 = np.diag([-0.4, 0.3, 0.8])
    dx_v = np.array([0.05, -0.04, 0.03])
    dg_v = h0 @ dx_v + 0.01 * np.array([1.0, -1.0, 0.5])
    kept = np.min(np.linalg.eigvalsh(bofill_update(h0, dx_v, dg_v))) < 0
    check(
        "F2 bofill_update: secant, symmetric, keeps a negative eigenvalue (claim F2)",
        worst_sec < 1e-9 and worst_sym < 1e-12 and kept,
        f"secant {worst_sec:.1e} sym {worst_sym:.1e}",
    )


# ---------------------------------------------------------------------------
# I: IRC
# ---------------------------------------------------------------------------
def irc_checks():
    x = sp.symbols("x0:3", real=True)
    m = sp.symbols("m0:3", positive=True)
    q = sp.symbols("q0:3", real=True)
    f = sp.Function("E")
    e_x = f(*x)
    e_q = e_x.subs({x[i]: q[i] / sp.sqrt(m[i]) for i in range(3)}, simultaneous=True)
    ok = True
    for i in range(3):
        lhs = sp.diff(e_q, q[i])
        rhs = (sp.diff(e_x, x[i]) / sp.sqrt(m[i])).subs(
            {x[j]: q[j] / sp.sqrt(m[j]) for j in range(3)}, simultaneous=True
        )
        ok = ok and sp.simplify(lhs.doit() - rhs.doit()) == 0
    rng = np.random.default_rng(9)
    g = rng.standard_normal(6)
    masses = np.repeat(np.array([12.011, 1.008]), 3)
    a = irc_cartesian_step(g, masses, 0.1)
    b = irc_cartesian_step_closed_form(g, masses, 0.1)
    unweighted = -0.1 * g / np.linalg.norm(g)
    differs = np.max(np.abs(a - unweighted)) > 1e-3
    same = (
        np.max(
            np.abs(
                irc_cartesian_step(g, np.ones(6), 0.1) - (-0.1 * g / np.linalg.norm(g))
            )
        )
        < 1e-14
    )
    check(
        "I1 mass-weighted gradient, Cartesian step -l g/(m |g_q|), differs from unweighted (claim I1)",
        ok and np.max(np.abs(a - b)) < 1e-14 and differs and same,
        f"closed form {np.max(np.abs(a - b)):.1e}",
    )


def main():
    bfgs_checks()
    prfo_checks()
    bofill_checks()
    irc_checks()
    if failures:
        print(f"{len(failures)} check(s) failed")
        sys.exit(1)
    print("all geometry-search step identities verified")


if __name__ == "__main__":
    main()
