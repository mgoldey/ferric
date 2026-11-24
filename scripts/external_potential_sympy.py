#!/usr/bin/env python3
"""External potential: point and smeared charges, a uniform field, gradients.

Derivation: `book/src/23-embedding/01-external-potentials.md#external-potentials-operator-classical-energy`.
Code: crates/ferric-core/src/external_potential.rs (charge_nuclear_energy,
field_nuclear_energy, charge_nuclear_gradient, charge_nuclear_hessian,
smeared_radial_derivs, smeared_charge_nuclear_site_gradient,
field_nuclear_gradient), crates/ferric-integrals/src/oneelectron.rs
(hcore_with_external, smeared_attraction, field_hcore_term) and
crates/ferric-scf/src/gradient.rs (oneelectron_gradient,
field_density_gradient_fd).

Checks (each label carries "(claim X)"):
  E1  erf(r/w)/r is the potential of the unit Gaussian
      (zeta/pi)^(3/2) exp(-zeta r^2), zeta = 1/w^2 (Poisson's equation in
      symbols); it is 2/(sqrt(pi) w) at r = 0 and tends to 1/r as w -> 0
      (the point-charge limit, the trivial-limit anchor).
  E2  smeared_radial_derivs: f' = g'/r - g/r^2 and
      f'' = -g'(2/w^2 + 2/r^2) + 2g/r^3 with g = erf(r/w) are the first and
      second derivatives of f = erf(r/w)/r (symbols).
  E3  The charge-nuclear gradient and Hessian blocks as coded equal the
      gradient and Hessian of Z q f(|R_A - r_c|) (sympy, numerically
      evaluated); the site gradient is minus the atom gradient; the point
      forms are the w -> 0 limit.
  E4  Field origin: with H' = +E.(r - O) and E_nuc = -E.sum_A Z_A (R_A - O),
      moving O changes the total energy by (E.O)(sum_A Z_A - N) = (E.O) Q,
      a constant: the code's O = 0 is exact for a neutral molecule and an
      orbital- and gradient-free shift otherwise. The un-negated electronic
      term would not cancel (negative control).
  E5  The field-density term d/dR_A Tr[P E.<r>] that gradient.rs takes by a
      central difference (h = 1e-4) equals the analytic derivative of the
      dipole integrals of an s-Gaussian model to the Richardson-estimated
      O(h^2) error, and obeys the sum rule sum_A dE/dR_A = E Tr[P S]; with the
      field-nuclear term -Z_A E the net force on the molecule is Q E.

Mutations (each turns the script red; exact one-occurrence substitutions):
  f'' with 1/w^2 in place of 2/w^2                           (E2 fails)
  electronic field term with the wrong sign                  (E4 fails)
  central difference divided by h, not 2h                    (E5 fails)
Exits nonzero on any failed check; prints "FAIL <label>".
"""

import sys

import numpy as np
import sympy as sp
from scipy.special import erf

failures = []


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        failures.append(label)


# --- transcriptions of the code ------------------------------------------------


def smeared_radial_derivs(r, width):
    g = erf(r / width)
    gp = 2.0 / (np.sqrt(np.pi) * width) * np.exp(-(r * r) / (width * width))
    f1 = gp / r - g / (r * r)
    f2 = -gp * (2.0 / (width * width) + 2.0 / (r * r)) + 2.0 * g / (r * r * r)
    return f1, f2


def charge_nuclear_gradient_smeared(za, q, ra, rc, width):
    d = ra - rc
    r = np.linalg.norm(d)
    sz_r = r / width
    dpot_dr = (2.0 / (np.sqrt(np.pi) * width)) * np.exp(-sz_r * sz_r) / r - erf(
        sz_r
    ) / (r * r)
    return za * q * dpot_dr / r * d


def charge_nuclear_hessian_smeared(za, q, ra, rc, width):
    d = ra - rc
    r2 = d @ d
    f1, f2 = smeared_radial_derivs(np.sqrt(r2), width)
    a, b = f2 / r2, f1 / np.sqrt(r2) / r2
    return (
        za
        * q
        * (a * np.outer(d, d) + f1 / np.sqrt(r2) * np.eye(3) - b * np.outer(d, d))
    )


def charge_nuclear_hessian_point(za, q, ra, rc):
    d = ra - rc
    r2 = d @ d
    r3 = r2 * np.sqrt(r2)
    return za * q * (3.0 * np.outer(d, d) / (r3 * r2) - np.eye(3) / r3)


def field_hcore_sign():
    """field_hcore_term: H' = +E . <mu|r - O|nu> (electron charge folded in)."""
    return +1


def field_density_gradient_fd(energy_of, coords, h=1e-4):
    grad = np.zeros_like(coords)
    for a in range(coords.shape[0]):
        for c in range(3):
            cp = coords.copy()
            cm = coords.copy()
            cp[a, c] += h
            cm[a, c] -= h
            grad[a, c] = (energy_of(cp) - energy_of(cm)) / (2.0 * h)
    return grad


# --- E1, E2 --------------------------------------------------------------------


def claim_e1_e2():
    r, w = sp.symbols("r w", positive=True)
    zeta = 1 / w**2
    phi = sp.erf(r / w) / r
    rho = (zeta / sp.pi) ** sp.Rational(3, 2) * sp.exp(-zeta * r**2)
    lap = sp.diff(r * phi, r, 2) / r
    poisson = sp.simplify(lap + 4 * sp.pi * rho) == 0
    at0 = sp.simplify(sp.limit(phi, r, 0) - 2 / (sp.sqrt(sp.pi) * w)) == 0
    point = sp.limit(phi, w, 0, "+") == 1 / r
    check(
        "E1 erf(r/w)/r solves Poisson for the unit Gaussian; w -> 0 is 1/r (claim E1)",
        poisson and at0 and point,
        f"poisson {poisson} r=0 {at0} point {point}",
    )
    f1s = sp.diff(phi, r)
    f2s = sp.diff(phi, r, 2)
    worst = 0.0
    for rv, wv in [(0.3, 0.5), (1.7, 0.9), (4.0, 2.5), (0.05, 0.2)]:
        f1, f2 = smeared_radial_derivs(rv, wv)
        e1 = float(f1s.subs({r: rv, w: wv}))
        e2 = float(f2s.subs({r: rv, w: wv}))
        worst = max(worst, abs(f1 - e1) / (1 + abs(e1)), abs(f2 - e2) / (1 + abs(e2)))
    check(
        "E2 smeared_radial_derivs are f' and f'' of erf(r/w)/r (claim E2)",
        worst < 1e-12,
        f"max rel {worst:.1e}",
    )


# --- E3 ------------------------------------------------------------------------


def claim_e3():
    x = sp.symbols("x0:3", real=True)
    c = sp.symbols("c0:3", real=True)
    w, zq = sp.symbols("w zq", positive=True)
    rr = sp.sqrt(sum((x[i] - c[i]) ** 2 for i in range(3)))
    e_sm = zq * sp.erf(rr / w) / rr
    e_pt = zq / rr
    g_sm = [sp.diff(e_sm, x[i]) for i in range(3)]
    h_sm = [[sp.diff(e_sm, x[i], x[j]) for j in range(3)] for i in range(3)]
    h_pt = [[sp.diff(e_pt, x[i], x[j]) for j in range(3)] for i in range(3)]
    g_site = [sp.diff(e_sm, c[i]) for i in range(3)]
    ra = np.array([0.4, -0.3, 1.1])
    rc = np.array([-0.9, 0.8, -0.2])
    za, q, wv = 7.0, -0.6, 0.8
    sub = {**dict(zip(x, ra)), **dict(zip(c, rc)), w: wv, zq: za * q}
    ref_g = np.array([float(e.subs(sub)) for e in g_sm])
    ref_h = np.array([[float(e.subs(sub)) for e in row] for row in h_sm])
    ref_hp = np.array([[float(e.subs(sub)) for e in row] for row in h_pt])
    ref_site = np.array([float(e.subs(sub)) for e in g_site])
    code_g = charge_nuclear_gradient_smeared(za, q, ra, rc, wv)
    code_h = charge_nuclear_hessian_smeared(za, q, ra, rc, wv)
    code_hp = charge_nuclear_hessian_point(za, q, ra, rc)
    errs = [
        np.abs(code_g - ref_g).max(),
        np.abs(code_h - ref_h).max(),
        np.abs(code_hp - ref_hp).max(),
        np.abs(-code_g - ref_site).max(),
    ]
    tight = np.abs(charge_nuclear_hessian_smeared(za, q, ra, rc, 1e-3) - code_hp).max()
    check(
        "E3 smeared/point gradient and Hessian blocks, site = -atom, w -> 0 (claim E3)",
        max(errs) < 1e-12 and tight < 1e-12,
        f"errs {[f'{e:.1e}' for e in errs]} w->0 {tight:.1e}",
    )


# --- E4 ------------------------------------------------------------------------


def claim_e4():
    e = sp.Matrix(sp.symbols("E0:3", real=True))
    o = sp.Matrix(sp.symbols("O0:3", real=True))
    mu = sp.Matrix(sp.symbols("m0:3", real=True))  # Tr[P <r>] about 0
    n_el, zsum = sp.symbols("N Zsum", real=True)
    zr = sp.Matrix(sp.symbols("zr0:3", real=True))  # sum_A Z_A R_A
    sgn = field_hcore_sign()

    def total(origin):
        elec = sgn * (e.T * (mu - n_el * origin))[0]
        nuc = -(e.T * (zr - zsum * origin))[0]
        return elec + nuc

    shift = sp.expand(total(o) - total(sp.zeros(3, 1)))
    want = sp.expand((e.T * o)[0] * (zsum - n_el))
    ok = sp.simplify(shift - want) == 0
    neutral = sp.simplify(shift.subs(n_el, zsum)) == 0
    wrong = sp.expand(-(e.T * (mu - n_el * o))[0] - (e.T * (zr - zsum * o))[0])
    wrong0 = sp.expand(-(e.T * mu)[0] - (e.T * zr)[0])
    control = sp.simplify(sp.expand(wrong - wrong0).subs(n_el, zsum)) != 0
    check(
        "E4 field origin shift = (E.O)(sum Z - N), zero when neutral (claim E4)",
        ok and neutral and control,
        f"shift = {sp.factor(shift)}",
    )


# --- E5 ------------------------------------------------------------------------


class SModel:
    """Normalized s Gaussians, two per atom, on three atoms."""

    def __init__(self):
        self.alpha = np.array([0.8, 2.3, 0.6, 1.7, 1.1, 3.0])
        self.atom = np.array([0, 0, 1, 1, 2, 2])
        self.norm = (2 * self.alpha / np.pi) ** 0.75

    def s_and_dip(self, coords):
        a = self.alpha
        n = len(a)
        s = np.zeros((n, n))
        dip = np.zeros((3, n, n))
        for i in range(n):
            for j in range(n):
                ai, aj = a[i], a[j]
                ri, rj = coords[self.atom[i]], coords[self.atom[j]]
                p = ai + aj
                d2 = np.sum((ri - rj) ** 2)
                sij = self.norm[i] * self.norm[j] * (np.pi / p) ** 1.5
                sij *= np.exp(-ai * aj / p * d2)
                s[i, j] = sij
                dip[:, i, j] = (ai * ri + aj * rj) / p * sij
        return s, dip

    def d_dip_analytic(self, coords, pmat, field):
        """d/dR_A Tr[P E.<r>] from the closed-form derivatives of P_x S."""
        a = self.alpha
        n = len(a)
        g = np.zeros_like(coords)
        s, _ = self.s_and_dip(coords)
        for i in range(n):
            for j in range(n):
                ai, aj = a[i], a[j]
                ri, rj = coords[self.atom[i]], coords[self.atom[j]]
                p = ai + aj
                cen = (ai * ri + aj * rj) / p
                ds_dri = -2 * ai * aj / p * (ri - rj) * s[i, j]
                for c in range(3):
                    for k in range(3):
                        dcen = ai / p if c == k else 0.0
                        term_i = dcen * s[i, j] + cen[k] * ds_dri[c]
                        term_j = (aj / p if c == k else 0.0) * s[i, j] - cen[
                            k
                        ] * ds_dri[c]
                        g[self.atom[i], c] += pmat[i, j] * field[k] * term_i
                        g[self.atom[j], c] += pmat[i, j] * field[k] * term_j
        return g


def claim_e5():
    model = SModel()
    coords = np.array([[0.0, 0.0, 0.0], [1.4, 0.3, -0.2], [-0.5, 1.2, 0.7]])
    rng = np.random.default_rng(11)
    pmat = rng.standard_normal((6, 6))
    pmat = 0.5 * (pmat + pmat.T)
    field = np.array([0.02, -0.03, 0.05])
    zs = np.array([2.0, 1.0, 3.0])

    def e_field_density(cc):
        _, dip = model.s_and_dip(cc)
        return np.einsum("mn,kmn,k->", pmat, dip, field)

    an = model.d_dip_analytic(coords, pmat, field)
    fd = field_density_gradient_fd(e_field_density, coords)
    fd2 = field_density_gradient_fd(e_field_density, coords, h=2e-4)
    richardson = np.abs(fd - fd2).max() / 3.0
    err = np.abs(fd - an).max()
    s, _ = model.s_and_dip(coords)
    n_el = np.sum(pmat * s)
    sum_rule = np.abs(an.sum(axis=0) - field * n_el).max()
    nuc = -zs[:, None] * field[None, :]
    q_mol = zs.sum() - n_el
    net = np.abs((an + nuc).sum(axis=0) + q_mol * field).max()
    check(
        "E5 FD field-density term = analytic to O(h^2); sum rule; net force Q E (claim E5)",
        err <= 2 * richardson + 1e-11 and sum_rule < 1e-12 and net < 1e-12,
        f"fd err {err:.1e} richardson {richardson:.1e} sum rule {sum_rule:.1e} net {net:.1e}",
    )


def main():
    claim_e1_e2()
    claim_e3()
    claim_e4()
    claim_e5()
    if failures:
        print(f"{len(failures)} check(s) failed")
        sys.exit(1)
    print("all external-potential identities verified")


if __name__ == "__main__":
    main()
