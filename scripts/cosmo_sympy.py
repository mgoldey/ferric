#!/usr/bin/env python3
"""COSMO: conductor-limit charges, energy, reaction-field Fock term, S matrix.

Derivation: crates/ferric-scf/theory.md, "COSMO reaction field". Code:
crates/ferric-scf/src/cosmo.rs (CosmoConfig::f_epsilon,
CosmoCavity::build_s_matrix, cosmo_reaction_field) and
crates/ferric-scf/src/driver.rs (solvent_terms).

As coded, per SCF iteration with total AO density P:
  v_k = sum_A Z^eff_A / |s_k - R_A| + sum_mn P_mn M_k,mn,
        M_k,mn = <m| -1/|r - s_k| |n>            (engine, +1 probe at s_k)
  S q = -f v,  f = (eps - 1)/(eps + 1/2)
  E_cosmo = 1/2 q . v,   V_mn = sum_k q_k M_k,mn  (added to every spin Fock)

Checks (each label carries "(claim X)"):
  C1  dE_cosmo/dP_mn = sum_k q_k M_k,mn exactly (symbols, two segments, two
      AOs, general symmetric S): the reaction-field Fock term is the energy
      derivative, with no factor 1/2, because S is symmetric.
  C2  The same identity numerically against an independent route: a central
      difference of E_cosmo(P) along a random symmetric direction (E is
      quadratic in P, so the difference is exact up to rounding), on a
      Gaussian-smeared S built from segments on a sphere; and E_cosmo <= 0
      for a positive definite S. Negative controls: V/2 misses, and with a
      non-symmetric S the operator sum_k q_k M_k is no longer the derivative
      (the identity needs the symmetric S the code builds).
  C3  f(eps) = 2(eps - 1)/(2 eps + 1), Kirkwood's k_1; f(1) = 0 and
      f -> 1 as eps -> oo (the conductor; trivial-limit anchor). C-PCM's
      (eps - 1)/eps is a different function (negative control).
  C4  The GaussianSmeared S: for unit Gaussians (xi^2/pi)^(3/2) exp(-xi^2 r^2)
      the Coulomb interaction is erf(xi_kl r)/r with
      xi_kl = xi_k xi_l / sqrt(xi_k^2 + xi_l^2), and the self term is
      xi sqrt(2/pi); both against an independent Fourier-space quadrature.
      Anchor: erf(xi r)/r -> 1/r as xi -> oo (the PointCharge off-diagonal).

Mutations (each turns the script red; exact one-occurrence substitutions):
  Fock term carries a factor 1/2                              (C2 fails)
  f uses x = 0 (the C-PCM form)                               (C3 fails)
  xi_kl combined as xi_k xi_l / (xi_k + xi_l)                 (C4 fails)
Exits nonzero on any failed check; prints "FAIL <label>".
"""

import sys

import numpy as np
import sympy as sp
from scipy import integrate
from scipy.special import erf

failures = []


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        failures.append(label)


# --- transcriptions of the code ------------------------------------------------


def f_epsilon(eps):
    return (eps - 1) / (eps + sp.Rational(1, 2))


def xi_pair(xk, xl):
    return xk * xl / np.sqrt(xk * xk + xl * xl)


def s_gaussian(pos, xi, swf):
    """CosmoCavity::build_s_matrix, SMatrixKind::GaussianSmeared."""
    n = len(xi)
    s = np.zeros((n, n))
    for k in range(n):
        s[k, k] = xi[k] * np.sqrt(2 / np.pi) / swf[k]
        for j in range(k):
            r = np.linalg.norm(pos[k] - pos[j])
            v = erf(xi_pair(xi[k], xi[j]) * r) / r
            s[k, j] = s[j, k] = v
    return s


def cosmo_terms(s, f, v_nuc, m, p):
    """cosmo_reaction_field: v, q, E_cosmo and the Fock term V."""
    v = v_nuc + np.einsum("kmn,mn->k", m, p)
    q = np.linalg.solve(s, -f * v)
    e = 0.5 * q @ v
    return q, e, fock(q, m)


def fock(q, m):
    return np.einsum("k,kmn->mn", q, m)


# --- C1 ------------------------------------------------------------------------


def claim_c1():
    fs = sp.Symbol("f", positive=True)
    s11, s12, s22 = sp.symbols("s11 s12 s22")
    smat = sp.Matrix([[s11, s12], [s12, s22]])
    p = sp.Matrix(2, 2, sp.symbols("p11 p12 p21 p22"))
    mk = [
        sp.Matrix(2, 2, lambda i, j, k=k: sp.Symbol(f"m{k}_{min(i, j)}{max(i, j)}"))
        for k in range(2)
    ]
    vn = sp.symbols("vn0 vn1")
    v = sp.Matrix(
        [
            vn[k] + sum(p[i, j] * mk[k][i, j] for i in range(2) for j in range(2))
            for k in range(2)
        ]
    )
    q = -fs * smat.inv() * v
    e = sp.Rational(1, 2) * (q.T * v)[0]
    ok = True
    for i in range(2):
        for j in range(2):
            lhs = sp.diff(e, p[i, j])
            rhs = sum(q[k] * mk[k][i, j] for k in range(2))
            ok = ok and sp.simplify(lhs - rhs) == 0
    check("C1 dE_cosmo/dP = sum_k q_k M_k symbolically (claim C1)", ok)


# --- C2 ------------------------------------------------------------------------


def sphere_points(n, radius):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5**0.5) * i
    return radius * np.stack(
        [np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1
    )


def claim_c2():
    rng = np.random.default_rng(7)
    nseg, nbf = 40, 4
    pos = sphere_points(nseg, 3.0)
    xi = 3.0 + rng.random(nseg)
    swf = 0.6 + 0.4 * rng.random(nseg)
    s = s_gaussian(pos, xi, swf)
    m = rng.standard_normal((nseg, nbf, nbf)) * 0.1
    m = 0.5 * (m + m.transpose(0, 2, 1))
    v_nuc = rng.standard_normal(nseg)
    p = rng.standard_normal((nbf, nbf))
    p = 0.5 * (p + p.T)
    dp = rng.standard_normal((nbf, nbf))
    dp = 0.5 * (dp + dp.T)
    f = float(f_epsilon(sp.Float(78.39)))
    _, e0, vmat = cosmo_terms(s, f, v_nuc, m, p)
    h = 1e-3
    ep = cosmo_terms(s, f, v_nuc, m, p + h * dp)[1]
    em = cosmo_terms(s, f, v_nuc, m, p - h * dp)[1]
    fd = (ep - em) / (2 * h)
    an = np.sum(vmat * dp)
    err = abs(fd - an)
    spd = np.linalg.eigvalsh(s).min() > 0
    check(
        "C2 Tr[V dP] = FD of E_cosmo(P), E_cosmo <= 0 (claim C2)",
        err < 1e-9 * max(1.0, abs(an)) and spd and e0 <= 0,
        f"analytic {an:.12f} fd {fd:.12f} err {err:.1e} E {e0:.3e} S spd {spd}",
    )
    miss_half = abs(0.5 * an - fd)
    a = s + np.triu(rng.standard_normal((nseg, nseg)) * 0.05, 1)
    qa = np.linalg.solve(a, -f * (v_nuc + np.einsum("kmn,mn->k", m, p)))
    ep = cosmo_terms(a, f, v_nuc, m, p + h * dp)[1]
    em = cosmo_terms(a, f, v_nuc, m, p - h * dp)[1]
    miss_nonsym = abs(np.sum(fock(qa, m) * dp) - (ep - em) / (2 * h))
    check(
        "C2b V/2 and a non-symmetric S both miss (claim C2)",
        miss_half > 1e-4 and miss_nonsym > 1e-6,
        f"V/2 {miss_half:.1e} non-symmetric {miss_nonsym:.1e}",
    )


# --- C3 ------------------------------------------------------------------------


def claim_c3():
    eps = sp.Symbol("epsilon", positive=True)
    f = f_epsilon(eps)
    k1 = 2 * (eps - 1) / (2 * eps + 1)
    ok = (
        sp.simplify(f - k1) == 0
        and f.subs(eps, 1) == 0
        and sp.limit(f, eps, sp.oo) == 1
    )
    cpcm = (eps - 1) / eps
    differs = sp.simplify(f - cpcm) != 0
    check(
        "C3 f = Kirkwood k_1, f(1) = 0, f -> 1 (claim C3)",
        ok and differs,
        f"f = {sp.simplify(f)}",
    )


# --- C4 ------------------------------------------------------------------------


def gaussian_coulomb_quad(xk, xl, r):
    """Coulomb energy of two unit Gaussians a distance r apart, k-space route."""
    c = 0.25 * (1 / xk**2 + 1 / xl**2)
    if r == 0.0:
        return (2 / np.pi) * integrate.quad(lambda k: np.exp(-c * k * k), 0, np.inf)[0]
    val = integrate.quad(
        lambda k: np.sinc(k * r / np.pi) * np.exp(-c * k * k),
        0,
        np.inf,
        limit=400,
        epsabs=1e-14,
        epsrel=1e-13,
    )[0]
    return (2 / np.pi) * val


def claim_c4():
    worst = 0.0
    for xk, xl, r in [
        (1.3, 2.2, 0.7),
        (3.0, 3.0, 1.9),
        (0.8, 4.1, 2.5),
        (2.0, 0.5, 4.0),
    ]:
        code = erf(xi_pair(xk, xl) * r) / r
        worst = max(worst, abs(code - gaussian_coulomb_quad(xk, xl, r)))
    self_err = max(
        abs(x * np.sqrt(2 / np.pi) - gaussian_coulomb_quad(x, x, 0.0))
        for x in (0.7, 2.4)
    )
    xs, rs = sp.symbols("xi r", positive=True)
    point = sp.limit(sp.erf(xs * rs) / rs, xs, sp.oo) == 1 / rs
    selfsym = sp.limit(sp.erf(xs * rs / sp.sqrt(2)) / rs, rs, 0) - xs * sp.sqrt(
        2 / sp.pi
    )
    check(
        "C4 Gaussian S off-diagonal and self term = k-space Coulomb (claim C4)",
        worst < 1e-9 and self_err < 1e-9 and point and sp.simplify(selfsym) == 0,
        f"off-diagonal {worst:.1e} self {self_err:.1e} point limit {point}",
    )


def main():
    claim_c1()
    claim_c2()
    claim_c3()
    claim_c4()
    if failures:
        print(f"{len(failures)} check(s) failed")
        sys.exit(1)
    print("all COSMO identities verified")


if __name__ == "__main__":
    main()
