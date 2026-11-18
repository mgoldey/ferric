#!/usr/bin/env python3
"""Thole polarizable embedding: induction, Fock term and gradients.

Derivation: `book/src/23-embedding/03-polarizable-embedding.md#polarizable-embedding-induction-model`. Code:
crates/ferric-scf/src/polarizable.rs (thole_screening, thole_tensor,
solve_induction, induce, build_v_induced, qm_gradient_contribution,
site_gradient, charge_gradient_contribution, norm_int_p_shell,
p_axis_for_field_axis).

As coded: sites i with polarizability alpha_i solve
  (alpha^-1 - T) mu = E,   E_i = E_i^QM(D) + E_i^perm,
  T_ij = (3 lam5 rr^T - lam3 I)/r^3,  u = r/(alpha_i alpha_j)^(1/6), v = a u,
  lam3 = 1 - (1 + v + v^2/2) e^-v,  lam5 = lam3 - v^3/6 e^-v,
  E_pol = -1/2 mu . E,   V_ind = -sum_i mu_i . (dE_i^QM/dD)   (no 1/2),
  dE/dR_i = -mu_i . dE_i/dR_i - sum_j mu_i . dT_ij/dR_i . mu_j  (coefficient 1),
with dT/dR by a central difference of thole_tensor, h = 1e-6.

Checks (each label carries "(claim X)"):
  T1  lam5 = lam3 - (u/3) dlam3/du, and T_ab = d_a d_b phi(r) with
      phi' = -lam3/r^2 (symbols), so T is symmetric and even in r_i - r_j;
      lam3, lam5 -> 1 as a -> oo (the undamped tensor).
  T2  W(mu) = -mu.E + 1/2 mu.B.mu, B = alpha^-1 - T symmetric, is stationary
      at the induction solution and W(mu*) = -1/2 mu*.E = E_pol (symbols).
  T3  On a classical model (three sites, fixed point charges standing in for
      E^QM and E^perm) the site gradient with coefficient 1 on the T term, and
      the charge gradient -sum_i mu_i . dE_i/dr_c, equal central differences
      of E_pol with mu re-solved; the coefficient-1/2 T term and the naive
      fixed-mu derivative of -1/2 mu.E both miss (negative controls). The
      h = 1e-6 central difference of thole_tensor matches the analytic dT to
      the rounding bound eps |T| / h.
  T4  The Fock term: with E^QM_i linear in D, dE/dD of
      Tr[hD] + E_pol(D) is h - sum_i mu_i . F_i, no factor 1/2 (central
      difference; E_pol is quadratic in D so the difference is exact up to
      rounding).
  T5  Trivial limit: alpha -> 0 gives mu -> 0 and E_pol -> 0.
  T6  norm_int_p_shell(zeta) = 2^(1/4) zeta^(5/4) / (2 pi^(3/4)) is the
      reciprocal of the dipole moment of a unit-normalized p Gaussian
      N x exp(-zeta r^2), so (mu nu|p) * norm_int_p is the point-dipole field
      integral as zeta -> oo; and p_axis_for_field_axis(c) = (c + 2) mod 3 is
      the real-solid-harmonic order m = -1, 0, +1 = (y, z, x) of a pure p
      shell.

Mutations (each turns the script red; exact one-occurrence substitutions):
  T-derivative coefficient 1/2 instead of 1                  (T3 fails)
  Fock term with a factor 1/2                                 (T4 fails)
  lam5 with v^3/2 instead of v^3/6                            (T1 fails)
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


THOLE_A = 2.1304

# --- transcriptions of the code ------------------------------------------------


def thole_screening(v, exp=np.exp):
    e = exp(-v)
    lam3 = 1 - (1 + v + v * v / 2) * e
    return lam3, lam3 - v * v * v / 6 * e


def thole_tensor(ri, rj, ai, aj, a=THOLE_A):
    d = ri - rj
    r = np.sqrt(d @ d)
    rhat = d / r
    lam3, lam5 = (
        (1.0, 1.0) if a is None else thole_screening(a * r / (ai * aj) ** (1 / 6))
    )
    return (3 * lam5 * np.outer(rhat, rhat) - lam3 * np.eye(3)) / r**3


def solve_induction(pos, alpha, e_tot):
    n = len(alpha)
    b = np.zeros((3 * n, 3 * n))
    for i in range(n):
        b[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = np.eye(3) / alpha[i]
        for j in range(n):
            if i != j:
                t = thole_tensor(pos[i], pos[j], alpha[i], alpha[j])
                b[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = -t
    return np.linalg.solve(b, e_tot.ravel()).reshape(n, 3)


def t_derivative_term(pos, alpha, mu, i, h=1e-6):
    """site_gradient step 3b: -coef * mu_i . dT_ij/dR_i . mu_j, FD of T."""
    g = np.zeros(3)
    for j in range(len(alpha)):
        if j == i:
            continue
        for k in range(3):
            rp, rm = pos[i].copy(), pos[i].copy()
            rp[k] += h
            rm[k] -= h
            tp = thole_tensor(rp, pos[j], alpha[i], alpha[j])
            tm = thole_tensor(rm, pos[j], alpha[i], alpha[j])
            d_dot_dr = (mu[i] @ tp @ mu[j] - mu[i] @ tm @ mu[j]) / (2 * h)
            g[k] += -1.0 * d_dot_dr
    return g


def fock_term(mu, f_ints):
    """build_v_induced: V = -sum_i mu_i . F_i (F_i the field integrals)."""
    return -np.einsum("ic,icmn->mn", mu, f_ints)


# --- point-charge fields of the model -----------------------------------------


def field_at(r, charges):
    e = np.zeros(3)
    for q, rc in charges:
        d = r - rc
        e += q * d / np.linalg.norm(d) ** 3
    return e


def field_grad_wrt_site(r, q, rc):
    """dE_c/dr (3x3, [field comp, coordinate]) of one point charge."""
    d = r - rc
    rn = np.linalg.norm(d)
    return q * (np.eye(3) / rn**3 - 3 * np.outer(d, d) / rn**5)


# --- T1 ------------------------------------------------------------------------


def claim_t1():
    u, a, r = sp.symbols("u a r", positive=True)
    lam3, lam5 = thole_screening(a * u, exp=sp.exp)
    ident = sp.simplify(lam5 - (lam3 - u / 3 * sp.diff(lam3, u))) == 0
    lam3_r = sp.Function("L3")(r)
    phi_p = -lam3_r / r**2
    phi_pp = sp.diff(phi_p, r)
    lam5_r = lam3_r - r / 3 * sp.diff(lam3_r, r)
    radial = sp.simplify((phi_pp - phi_p / r) - 3 * lam5_r / r**3) == 0
    tangential = sp.simplify(phi_p / r + lam3_r / r**3) == 0
    lim = (
        sp.limit(lam3.subs(u, 1), a, sp.oo) == 1
        and sp.limit(lam5.subs(u, 1), a, sp.oo) == 1
    )
    rng = np.random.default_rng(3)
    ri, rj = rng.standard_normal(3) * 2, rng.standard_normal(3) * 2
    t = thole_tensor(ri, rj, 1.3, 0.7)
    sym = np.allclose(t, t.T, atol=1e-15) and np.allclose(
        t, thole_tensor(rj, ri, 0.7, 1.3)
    )
    check(
        "T1 lam5 identity, T = Hessian of phi, symmetric and even, a -> oo undamped (claim T1)",
        ident and radial and tangential and lim and sym,
        f"identity {ident} hessian {radial and tangential} limit {lim} symmetric {sym}",
    )


# --- T2 ------------------------------------------------------------------------


def claim_t2():
    b11, b12, b22, e1, e2 = sp.symbols("b11 b12 b22 e1 e2", real=True)
    bm = sp.Matrix([[b11, b12], [b12, b22]])
    ev = sp.Matrix([e1, e2])
    mu = sp.Matrix(sp.symbols("m1 m2", real=True))
    w = -(mu.T * ev)[0] + sp.Rational(1, 2) * (mu.T * bm * mu)[0]
    grad = sp.Matrix([sp.diff(w, m) for m in mu])
    star = bm.inv() * ev
    stat = sp.simplify(grad.subs(dict(zip(mu, star)))) == sp.zeros(2, 1)
    val = (
        sp.simplify(w.subs(dict(zip(mu, star))) + sp.Rational(1, 2) * (star.T * ev)[0])
        == 0
    )
    check(
        "T2 W stationary at the induction solution, W* = E_pol (claim T2)",
        stat and val,
    )


# --- T3 ------------------------------------------------------------------------


def model():
    pos = np.array([[0.0, 0.0, 0.0], [3.1, 0.4, -0.3], [0.8, 2.9, 1.2]])
    alpha = np.array([9.7, 6.4, 11.2])
    charges = [(1.3, np.array([-2.5, 0.3, 0.9])), (-0.8, np.array([1.9, -2.6, 0.4]))]
    return pos, alpha, charges


def e_pol(pos, alpha, charges):
    e = np.array([field_at(r, charges) for r in pos])
    mu = solve_induction(pos, alpha, e)
    return -0.5 * np.sum(mu * e), mu, e


def analytic_dT(ri, rj, ai, aj):
    x = sp.symbols("x0:3", real=True)
    y = sp.symbols("y0:3", real=True)
    d = sp.Matrix([x[k] - y[k] for k in range(3)])
    r = sp.sqrt(d.dot(d))
    lam3, lam5 = thole_screening(
        THOLE_A * r / sp.Float(ai * aj) ** sp.Rational(1, 6), exp=sp.exp
    )
    t = (3 * lam5 * d * d.T / r**2 - lam3 * sp.eye(3)) / r**3
    sub = {**dict(zip(x, ri)), **dict(zip(y, rj))}
    return np.array(
        [
            [
                [float(sp.diff(t[a, b], x[k]).subs(sub)) for k in range(3)]
                for b in range(3)
            ]
            for a in range(3)
        ]
    )


def claim_t3():
    pos, alpha, charges = model()
    _, mu, _ = e_pol(pos, alpha, charges)
    h = 1e-5
    worst, miss_half, miss_naive = 0.0, 0.0, 0.0
    for i in range(3):
        egrad = sum(field_grad_wrt_site(pos[i], q, rc) for q, rc in charges)
        perm = -mu[i] @ egrad
        tterm = t_derivative_term(pos, alpha, mu, i)
        an = perm + tterm
        fd = np.zeros(3)
        for k in range(3):
            pp, pm = pos.copy(), pos.copy()
            pp[i, k] += h
            pm[i, k] -= h
            fd[k] = (e_pol(pp, alpha, charges)[0] - e_pol(pm, alpha, charges)[0]) / (
                2 * h
            )
        worst = max(worst, np.abs(an - fd).max())
        miss_half = max(miss_half, np.abs(perm + 0.5 * tterm - fd).max())
        miss_naive = max(miss_naive, np.abs(-0.5 * mu[i] @ egrad - fd).max())
    cworst = 0.0
    for c, (q, rc) in enumerate(charges):
        an = sum(-mu[i] @ (-field_grad_wrt_site(pos[i], q, rc)) for i in range(3))
        fd = np.zeros(3)
        for k in range(3):
            cp = [(qq, p.copy()) for qq, p in charges]
            cm = [(qq, p.copy()) for qq, p in charges]
            cp[c][1][k] += h
            cm[c][1][k] -= h
            fd[k] = (e_pol(pos, alpha, cp)[0] - e_pol(pos, alpha, cm)[0]) / (2 * h)
        cworst = max(cworst, np.abs(an - fd).max())
    check(
        "T3 site and charge gradients = FD of E_pol with mu re-solved (claim T3)",
        worst < 1e-9 and cworst < 1e-9 and miss_half > 1e-5 and miss_naive > 1e-5,
        f"site {worst:.1e} charge {cworst:.1e} controls coef-1/2 {miss_half:.1e} naive {miss_naive:.1e}",
    )
    dt_exact = analytic_dT(pos[0], pos[1], alpha[0], alpha[1])
    hh = 1e-6
    dt_fd = np.zeros((3, 3, 3))
    for k in range(3):
        rp, rm = pos[0].copy(), pos[0].copy()
        rp[k] += hh
        rm[k] -= hh
        dt_fd[:, :, k] = (
            thole_tensor(rp, pos[1], alpha[0], alpha[1])
            - thole_tensor(rm, pos[1], alpha[0], alpha[1])
        ) / (2 * hh)
    tmax = np.abs(thole_tensor(pos[0], pos[1], alpha[0], alpha[1])).max()
    bound = 10 * np.finfo(float).eps * tmax / hh
    err = np.abs(dt_fd - dt_exact).max()
    check(
        "T3b thole_tensor central difference at h = 1e-6 within the rounding bound (claim T3)",
        err < bound,
        f"err {err:.1e} bound {bound:.1e}",
    )


# --- T4 ------------------------------------------------------------------------


def claim_t4():
    rng = np.random.default_rng(5)
    pos, alpha, charges = model()
    nbf = 4
    f_ints = rng.standard_normal((3, 3, nbf, nbf)) * 0.05
    f_ints = 0.5 * (f_ints + f_ints.transpose(0, 1, 3, 2))
    e_nuc = np.array([field_at(r, charges) for r in pos])
    hcore = rng.standard_normal((nbf, nbf))
    hcore = 0.5 * (hcore + hcore.T)

    def energy(d):
        e = e_nuc + np.einsum("icmn,mn->ic", f_ints, d)
        mu = solve_induction(pos, alpha, e)
        return np.sum(hcore * d) - 0.5 * np.sum(mu * e), mu

    d0 = rng.standard_normal((nbf, nbf))
    d0 = 0.5 * (d0 + d0.T)
    dd = rng.standard_normal((nbf, nbf))
    dd = 0.5 * (dd + dd.T)
    _, mu = energy(d0)
    an = np.sum((hcore + fock_term(mu, f_ints)) * dd)
    h = 1e-3
    fd = (energy(d0 + h * dd)[0] - energy(d0 - h * dd)[0]) / (2 * h)
    check(
        "T4 dE/dD = h - sum_i mu_i . F_i, no 1/2 (claim T4)",
        abs(an - fd) < 1e-10 * max(1.0, abs(fd)),
        f"analytic {an:.12f} fd {fd:.12f}",
    )


# --- T5, T6 --------------------------------------------------------------------


def claim_t5():
    pos, _, charges = model()
    ep, mu, _ = e_pol(pos, np.array([1e-12, 1e-12, 1e-12]), charges)
    check(
        "T5 alpha -> 0 gives no induction (claim T5)",
        np.abs(mu).max() < 1e-11 and abs(ep) < 1e-11,
        f"|mu| {np.abs(mu).max():.1e} E_pol {ep:.1e}",
    )


def claim_t6():
    z = sp.Symbol("zeta", positive=True)
    x, y, w = sp.symbols("x y w", real=True)
    g = sp.exp(-z * (x * x + y * y + w * w))
    n2 = 1 / sp.integrate(
        x * x * g * g, (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo), (w, -sp.oo, sp.oo)
    )
    nrm = sp.sqrt(sp.simplify(n2))
    dip = nrm * sp.integrate(
        x * x * g, (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo), (w, -sp.oo, sp.oo)
    )
    code = (
        2 ** sp.Rational(1, 4)
        * z ** sp.Rational(5, 4)
        / (2 * sp.pi ** sp.Rational(3, 4))
    )
    ok = sp.simplify(1 / dip - code) == 0
    order = ["y", "z", "x"]  # libint real solid harmonics, l = 1, m = -1, 0, +1
    perm = all(order[(c + 2) % 3] == "xyz"[c] for c in range(3))
    check(
        "T6 norm_int_p = 1/dipole of a unit p Gaussian; (c+2)%3 is the m order (claim T6)",
        ok and perm,
        f"1/d = {sp.simplify(1 / dip)}",
    )


def main():
    claim_t1()
    claim_t2()
    claim_t3()
    claim_t4()
    claim_t5()
    claim_t6()
    if failures:
        print(f"{len(failures)} check(s) failed")
        sys.exit(1)
    print("all Thole polarizable-embedding identities verified")


if __name__ == "__main__":
    main()
