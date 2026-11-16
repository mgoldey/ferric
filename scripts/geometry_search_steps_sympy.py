#!/usr/bin/env python3
"""Step formulas of ferric's geometry searches: BFGS, P-RFO, Bofill, IRC.

Derivation: crates/ferric-scf/theory.md, "Geometry searches";
book/src/14-derivatives/08-geometry-search.md. Code:
crates/ferric-scf/src/optimize.rs (the inverse-Hessian update of
optimize_coordinates and run_bfgs_internal, the internal
clip and ratio rule), crates/ferric-scf/src/saddle.rs
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
  B3  The curvature guard: the secant condition forces y^T H' y = y.s, so no
      update satisfying it can be positive definite when y.s <= 0 (symbols);
      the guarded update (skip when y.s <= eps |y||s|) keeps H positive
      definite for every pair, the unguarded one loses it on every y.s < 0
      pair, and above the threshold the two are bitwise identical (numerics).
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
  T1  The ratio rule's predicted change: for s = alpha p, p = -H g and
      B = H^-1, g.s + s.B s / 2 = alpha g.p (1 - alpha/2), and it is negative
      for H positive definite and 0 < alpha <= 1 (symbols).
  T2  ratio_trust_update, transcribed: shrinks to s/4 when r < 1/4, doubles
      only when r > 3/4 AND the step reached the boundary, keeps the radius in
      between and inside the noise band, clamps to [r_min, r_max], and rejects
      exactly the steps that raised the energy by 1e-8 Ha or more; driven on
      the Rosenbrock function from a far start, the ratio search never keeps
      an uphill step and reaches the minimum, while the fixed-radius search at
      the same radius keeps uphill steps (numerics).
  T3  The radius never grows on a rejected step, nor on a step whose
      predicted change is >= 0 (a grid sweep over every branch), and the Rust
      source has the same shrink condition.
  C1  Internal clip (trust_update = "ratio"): one factor alpha keeps
      dq = alpha p, so g.dq = -alpha g^T H g < 0 (symbols). Separate factors
      for the distance and angular blocks fix a direction that does not
      change with the trust scale (symbols) and is uphill for some positive
      definite H; it stays uphill at every radius (numerics). A model direction
      that does not descend falls back to -D g, D the positive diagonal seed.
  C2  The back-transformed step is s = P dq to first order, P = G G^-, and
      g_q.s = g_q.dq = g_x.dx, the last the Cartesian first-order change
      (numerics, random rank-deficient B).
  I1  The mass-weighted gradient is dE/dq = (dE/dx)/sqrt(m), so the downhill
      step of length l in q is dx_i = -l g_i /(m_i |g_q|), which differs from
      unweighted steepest descent unless all masses are equal.

Mutations (each turns the script red; exact one-occurrence substitutions):
  BFGS term1 = (ys + yhy) rho^2 -> yhy rho^2                 (B2 fails)
  BFGS curvature guard removed                               (B3 fails)
  P-RFO followed shift: + sqrt -> - sqrt                      (P2 fails)
  Bofill PSB projector term dropped                          (F2 fails)
  IRC Cartesian step: divide by m, not sqrt(m)               (I1 fails)
  ratio rule thresholds inverted (shrink above 3/4)          (T2 fails)
  ratio rule: shrink only on r < 1/4 (no rejection guard)    (T3 fails)
  ratio internal clip: per block, not one factor             (C1 fails)
Exits nonzero on any failed check; prints "FAIL <label>".
"""

import os
import re
import sys
from pathlib import Path

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


CURVATURE_EPS = 1e-8


def guarded_bfgs_inverse_update(h, s, y):
    """optimize.rs::guarded_inverse_bfgs_update: skip the update (keep H) when
    y.s <= CURVATURE_EPS |y| |s|, otherwise apply bfgs_inverse_update.
    Returns (H', applied)."""
    ys = y @ s
    if ys <= CURVATURE_EPS * np.linalg.norm(y) * np.linalg.norm(s):
        return h.copy(), False
    return bfgs_inverse_update(h, s, y), True


TRUST_SHRINK_RATIO = 0.25
TRUST_GROW_RATIO = 0.75
TRUST_ENERGY_NOISE = 1e-8


def ratio_trust_update(actual, predicted, step_norm, radius, r_min, r_max):
    """optimize.rs::ratio_trust_update. Returns (accept, next radius)."""
    accept = actual < TRUST_ENERGY_NOISE
    if abs(predicted) < TRUST_ENERGY_NOISE and abs(actual) < TRUST_ENERGY_NOISE:
        return accept, radius
    ratio = actual / predicted
    if not accept or predicted >= 0.0 or ratio < TRUST_SHRINK_RATIO:
        nxt = TRUST_SHRINK_RATIO * step_norm
    elif ratio > TRUST_GROW_RATIO and step_norm >= 0.99 * radius:
        nxt = 2.0 * radius
    else:
        nxt = radius
    return accept, min(max(nxt, r_min), r_max)


def clip_per_block(p, angular, dist_trust, ang_trust):
    """optimize.rs::clip_internal_step (trust_update = "fixed"): the distance
    and the angular block each scaled onto its own radius."""
    dn = np.linalg.norm(p[~angular])
    an = np.linalg.norm(p[angular])
    out = p.copy()
    out[~angular] *= dist_trust / dn if dn > dist_trust else 1.0
    out[angular] *= ang_trust / an if an > ang_trust else 1.0
    return out


def clip_uniform(p, angular, dist_trust, ang_trust):
    """optimize.rs::clip_internal_step_uniform (trust_update = "ratio"): one
    factor alpha = min(1, dist_trust/|p_dist|, ang_trust/|p_ang|) for the whole
    step."""
    dn = np.linalg.norm(p[~angular])
    an = np.linalg.norm(p[angular])
    alpha = 1.0
    if dn > dist_trust:
        alpha = min(alpha, dist_trust / dn)
    if an > ang_trust:
        alpha = min(alpha, ang_trust / an)
    return alpha * p


def internal_ratio_direction(g, h, seed):
    """optimize.rs::internal_model_direction: p = -H g, or -D g with D the
    positive diagonal seed when -H g is not a descent direction."""
    p = -h @ g
    if g @ p >= 0.0:
        return -seed @ g, True
    return p, False


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


def bfgs_guard_checks():
    n = 3
    hs = sp.Matrix(n, n, lambda i, j: sp.Symbol(f"h{min(i, j)}{max(i, j)}"))
    s = sp.Matrix(sp.symbols("s0:3"))
    y = sp.Matrix(sp.symbols("y0:3"))
    ys = (y.T * s)[0]
    hy = hs * y
    yhy = (y.T * hy)[0]
    h_new = hs + (ys + yhy) / ys**2 * s * s.T - (hy * s.T + s * hy.T) / ys
    # y^T H' y = y.s: the curvature H' assigns along y IS y.s, so y.s <= 0
    # rules out a positive-definite H' for any secant update.
    quad = sp.simplify((y.T * h_new * y)[0] - ys)

    rng = np.random.default_rng(17)
    guarded_pd = True
    unguarded_lost = True
    n_neg = n_pos = 0
    identical = True
    for _ in range(400):
        m = rng.standard_normal((n, n))
        h = m @ m.T + 0.1 * np.eye(n)
        s_v = rng.standard_normal(n)
        y_v = rng.standard_normal(n)
        ys_v = y_v @ s_v
        h_g, applied = guarded_bfgs_inverse_update(h, s_v, y_v)
        guarded_pd = guarded_pd and np.min(np.linalg.eigvalsh(h_g)) > 0
        h_u = bfgs_inverse_update(h, s_v, y_v)
        if ys_v < 0:
            n_neg += 1
            unguarded_lost = unguarded_lost and np.min(np.linalg.eigvalsh(h_u)) < 0
        elif applied:
            n_pos += 1
            identical = identical and np.array_equal(h_g, h_u)
    check(
        "B3 curvature guard: y^T H' y = y.s, guarded update positive definite, "
        "unguarded loses it, no-op above the threshold (claim B3)",
        quad == 0
        and guarded_pd
        and unguarded_lost
        and identical
        and n_neg > 50
        and n_pos > 50,
        f"neg {n_neg} pos {n_pos}",
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


def _rust_source(rel):
    cands = [Path(__file__).resolve().parents[1], Path.cwd(), *Path.cwd().parents]
    if os.environ.get("FERRIC_ROOT"):
        cands.insert(0, Path(os.environ["FERRIC_ROOT"]))
    for c in cands:
        if (c / rel).is_file():
            return (c / rel).read_text(encoding="utf-8")
    return None


def trust_checks():
    # T1: the predicted change of a scaled quasi-Newton step, in symbols.
    n = 3
    a = sp.symbols("alpha", positive=True)
    hs = sp.Matrix(n, n, lambda i, j: sp.Symbol(f"h{min(i, j)}{max(i, j)}"))
    g = sp.Matrix(sp.symbols(f"g0:{n}"))
    p = -hs * g
    s_vec = a * p
    b = hs.inv()
    model = (g.T * s_vec)[0] + sp.Rational(1, 2) * (s_vec.T * b * s_vec)[0]
    closed = a * (g.T * p)[0] * (1 - a / 2)
    check(
        "T1 predicted change = alpha g.p (1 - alpha/2) (claim T1)",
        sp.simplify(model - closed) == 0,
    )
    rng = np.random.default_rng(5)
    neg = True
    for _ in range(50):
        m = rng.normal(size=(4, 4))
        h = m @ m.T + 0.1 * np.eye(4)
        gv = rng.normal(size=4)
        pv = -h @ gv
        for al in (0.1, 0.5, 1.0):
            neg &= al * (gv @ pv) * (1 - al / 2) < 0
    check("T1 predicted change negative for H > 0, 0 < alpha <= 1 (claim T1)", neg)

    # T2: the decision table.
    cases = [
        # actual, pred, step, radius -> accept, radius
        ((-0.1e-3, -1e-3, 0.1, 0.1), (True, 0.025)),  # r = 0.1: shrink to s/4
        ((+1e-3, -1e-3, 0.1, 0.1), (False, 0.025)),  # uphill: reject + shrink
        ((-0.9e-3, -1e-3, 0.1, 0.1), (True, 0.2)),  # r = 0.9 on boundary: grow
        ((-0.9e-3, -1e-3, 0.05, 0.1), (True, 0.1)),  # r = 0.9 inside: keep
        ((-0.5e-3, -1e-3, 0.1, 0.1), (True, 0.1)),  # r = 0.5: keep
        ((-1.5e-3, -1e-3, 0.1, 0.1), (True, 0.2)),  # r = 1.5 on boundary: grow
        ((5e-9, -5e-9, 0.1, 0.1), (True, 0.1)),  # both below noise: keep
        ((-0.9e-3, -1e-3, 1.0, 1.0), (True, 1.0)),  # growth clamped at r_max
        ((+1e-3, -1e-3, 1e-3, 1e-3), (False, 1e-3)),  # shrink clamped at r_min
    ]
    ok = all(ratio_trust_update(*args, 1e-3, 1.0) == want for args, want in cases)
    check("T2 ratio_trust_update decision table (claim T2)", ok)
    rust = _rust_source("crates/ferric-scf/src/optimize.rs")
    if rust is None:
        check("T2 optimize.rs found (set FERRIC_ROOT) (claim T2)", False)
    else:
        consts = {}
        for name in ("TRUST_SHRINK_RATIO", "TRUST_GROW_RATIO", "TRUST_ENERGY_NOISE"):
            m = re.search(rf"pub const {name}: f64 = ([0-9.e-]+);", rust)
            consts[name] = float(m.group(1)) if m else None
        mine = {
            "TRUST_SHRINK_RATIO": TRUST_SHRINK_RATIO,
            "TRUST_GROW_RATIO": TRUST_GROW_RATIO,
            "TRUST_ENERGY_NOISE": TRUST_ENERGY_NOISE,
        }
        check(
            "T2 transcribed constants = optimize.rs (claim T2)",
            consts == mine,
            f"{consts}",
        )

    def rosen(x):
        f = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        gr = np.array(
            [
                -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
                200 * (x[1] - x[0] ** 2),
            ]
        )
        return f, gr

    def search(ratio, radius0=0.5, steps=400):
        x = np.array([-1.2, 1.0])
        e, g = rosen(x)
        h = np.eye(2)
        radius, kept_uphill = radius0, 0
        for _ in range(steps):
            if np.linalg.norm(g) < 1e-8:
                break
            pv = -h @ g
            pn = np.linalg.norm(pv)
            al = min(1.0, radius / pn)
            sv = al * pv
            e_t, g_t = rosen(x + sv)
            accept = True
            if ratio:
                accept, radius = ratio_trust_update(
                    e_t - e,
                    al * (g @ pv) * (1 - al / 2),
                    al * pn,
                    radius,
                    0.01 * radius0,
                    10 * radius0,
                )
            h, _ = guarded_bfgs_inverse_update(h, sv, g_t - g)
            if accept:
                kept_uphill += e_t > e + TRUST_ENERGY_NOISE
                x, e, g = x + sv, e_t, g_t
        return x, kept_uphill

    xr, up_r = search(True)
    xf, up_f = search(False)
    check(
        "T2 ratio search on Rosenbrock: no uphill step kept (claim T2)",
        up_r == 0,
        f"{up_r}",
    )
    check(
        "T2 ratio search reaches the minimum (claim T2)",
        np.linalg.norm(xr - 1.0) < 1e-6,
        f"x = {xr}",
    )
    check("T2 fixed radius keeps uphill steps (claim T2)", up_f > 0, f"{up_f}")


def ratio_rule_never_grows_checks():
    # T3: a rejected step, or one the model does not predict to descend, never
    # grows the radius. Swept over a grid that includes every branch.
    worst = 0.0
    n_rej = n_pos = 0
    for actual in (-2e-3, -1e-3, -1e-5, -5e-9, 0.0, 5e-9, 2e-8, 1e-5, 1e-3, 2e-3):
        for predicted in (-2e-3, -1e-3, -1e-5, -5e-9, 5e-9, 1e-5, 1e-3, 2e-3):
            for radius in (0.01, 0.1, 1.0):
                for frac in (0.25, 0.99, 1.0):
                    acc, nxt = ratio_trust_update(
                        actual, predicted, frac * radius, radius, 0.01, 4.0
                    )
                    if not acc or predicted >= 0.0:
                        n_rej += not acc
                        n_pos += predicted >= 0.0
                        worst = max(worst, nxt - radius)
    check(
        "T3 the radius never grows on a rejected step or a predicted change >= 0 (claim T3)",
        worst <= 0.0 and n_rej > 50 and n_pos > 50,
        f"max growth {worst:.1e}, rejected {n_rej}, predicted>=0 {n_pos}",
    )
    # the two cases that used to grow: a rise predicted as a rise (r > 3/4 on
    # the boundary) and a fall predicted as a rise
    cases = [
        ((+2e-3, +1e-3, 0.1, 0.1), (False, 0.025)),
        ((-1e-3, +1e-3, 0.1, 0.1), (True, 0.025)),
    ]
    ok = all(ratio_trust_update(*args, 1e-3, 1.0) == want for args, want in cases)
    check("T3 predicted >= 0 shrinks, accepted or not (claim T3)", ok)
    rust = _rust_source("crates/ferric-scf/src/optimize.rs")
    check(
        "T3 optimize.rs has the same shrink condition (claim T3)",
        rust is not None
        and "if !accept || predicted >= 0.0 || ratio < TRUST_SHRINK_RATIO {" in rust,
    )


def clip_checks():
    # C1: one factor keeps the model direction, so the first-order change is
    # -alpha g^T H g < 0 (symbols); separate factors do not.
    a_d, a_a = sp.symbols("a_d a_a", positive=True)
    al = sp.symbols("alpha", positive=True)
    hs = sp.Matrix(4, 4, lambda i, j: sp.Symbol(f"h{min(i, j)}{max(i, j)}"))
    g = sp.Matrix(sp.symbols("g0:4"))
    p = -hs * g
    uniform = (g.T * (al * p))[0]
    check(
        "C1 uniform clip: g.dq = -alpha g^T H g (symbols) (claim C1)",
        sp.simplify(uniform + al * (g.T * hs * g)[0]) == 0,
    )
    # separate factors: g.dq = a_d g_d.p_d + a_a g_a.p_a, and with both blocks
    # on their radii a_d / a_a = (rho_d |p_a|)/(rho_a |p_d|) does not depend on
    # the trust scale f (rho = f rho0), so the clipped DIRECTION is fixed as f
    # shrinks.
    f, rd, ra, pd_, pa_ = sp.symbols("f rho_d rho_a pd pa", positive=True)
    ratio = (f * rd / pd_) / (f * ra / pa_)
    check(
        "C1 per-block clip direction is independent of the trust scale (symbols) (claim C1)",
        sp.diff(sp.simplify(ratio), f) == 0,
    )
    rng = np.random.default_rng(23)
    n, nd = 8, 4
    angular = np.array([False] * nd + [True] * (n - nd))
    n_bad = 0
    uniform_ok = True
    stuck = True
    for _ in range(2000):
        # inverse Hessian spanning three decades, as stretches against torsions
        qm, _ = np.linalg.qr(rng.standard_normal((n, n)))
        h = qm @ np.diag(np.logspace(-1, 2, n)) @ qm.T
        gv = rng.standard_normal(n)
        pv = -h @ gv
        for scale in (1.0, 1e-2):
            dq_u = clip_uniform(pv, angular, 0.1 * scale, 0.3 * scale)
            uniform_ok &= gv @ dq_u < 0.0
        dq_b = clip_per_block(pv, angular, 0.1 * 1e-3, 0.3 * 1e-3)
        if gv @ dq_b > 0.0:
            n_bad += 1
            # shrinking the radius a hundredfold keeps it uphill
            stuck &= gv @ clip_per_block(pv, angular, 0.1 * 1e-5, 0.3 * 1e-5) > 0.0
    check(
        "C1 uniform clip always descends; per-block clip has uphill cases that stay "
        "uphill at every radius (claim C1)",
        uniform_ok and n_bad > 20 and stuck,
        f"per-block uphill in {n_bad} of 2000",
    )
    # fallback: -D g with D the positive diagonal seed descends when -H g does not
    seed = np.diag(rng.uniform(0.5, 20.0, n))
    h_bad = -np.eye(n)
    gv = rng.standard_normal(n)
    pv, fell_back = internal_ratio_direction(gv, h_bad, seed)
    pv2, fell_back2 = internal_ratio_direction(gv, np.eye(n), seed)
    check(
        "C1 non-descent model direction falls back to -D g (claim C1)",
        fell_back and gv @ pv < 0 and not fell_back2 and np.array_equal(pv2, -gv),
    )

    # C2: the back-transformed step. To first order the achieved internal step
    # is s = P dq with P = G G^- (G = B B^T), and g_q = G^- B g_x lies in the
    # range of G, so g_q.s = g_q.dq. Independent route: the Cartesian
    # first-order change g_x.dx with dx = B^T G^- dq.
    worst = 0.0
    for _ in range(50):
        nq, nx, rank = 12, 9, 6
        b = rng.standard_normal((nq, rank)) @ rng.standard_normal((rank, nx))
        gx = b.T @ rng.standard_normal(nq)  # a gradient of q(x): in range(B^T)
        gmat = b @ b.T
        ginv = np.linalg.pinv(gmat, rcond=1e-10)
        gq = ginv @ b @ gx
        dq = rng.standard_normal(nq)
        dx = b.T @ ginv @ dq
        s_vec = b @ dx
        worst = max(worst, abs(gq @ s_vec - gq @ dq), abs(gx @ dx - gq @ dq))
    check(
        "C2 achieved step: g_q.(P dq) = g_q.dq = g_x.dx to first order (claim C2)",
        worst < 1e-9,
        f"max diff {worst:.1e}",
    )


def main():
    trust_checks()
    ratio_rule_never_grows_checks()
    clip_checks()
    bfgs_checks()
    bfgs_guard_checks()
    prfo_checks()
    bofill_checks()
    irc_checks()
    if failures:
        print(f"{len(failures)} check(s) failed")
        sys.exit(1)
    print("all geometry-search step identities verified")


if __name__ == "__main__":
    main()
