#!/usr/bin/env python3
"""Second nuclear derivatives of the Becke home-cell weight.

Derivation: crates/ferric-dft/src/becke_hessian.rs module documentation; used by
the moving-grid KS Hessian (book/src/14-derivatives/05-ks-hessian.md). The first
derivatives are proved in scripts/dft_becke_weight_grad_sympy.py.

Checks (each label carries "(claim X)"):
  W1  The pair formulas of the module doc for mu_AB = (r_A - r_B)/R_AB with
      r_X = |r - R_X| and R = R_AB, N = r_A - r_B:
        mu_x  = N_x/R - mu R_x/R
        mu_xy = N_xy/R - (N_x R_y + N_y R_x)/R^2 - mu R_xy/R + 2 mu R_x R_y/R^2
      equal the exact sympy derivatives with respect to the six nuclear
      coordinates of one pair (a point and two nuclei in 3-D), and
      s_xy = s_mm mu_x mu_y + s_m mu_xy for s = s(mu) (the last on a spread of
      six coordinate pairs, for run time).
  W2  The log-form assembly of the cell weight w = P_h/T with P_A = prod_B s_AB:
        P' = P L',  P'' = P (L'' + L' L'^T),  L = sum_B ln s_AB,
        w'' = P_h''/T - (P_h' T'^T + T' P_h'^T)/T^2 - P_h T''/T^2
              + 2 P_h T' T'^T/T^3
      equals the exact sympy second derivative of the quotient, for the Becke
      three-iteration step function on three nuclei (collinear with the point;
      the assembly is dimension-free), with the pair size adjustment held
      constant.

Mutations (each turns the script red; exact one-occurrence substitutions):
  mu_xy drops its 2 mu R_x R_y/R^2 term                      (W1 fails)
  w'' drops the 2 P_h T' T'^T/T^3 term                       (W2 fails)
Exits nonzero on any failed check; prints "FAIL <label>".
"""

import sys

import sympy as sp

failures = []


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}")
    if not ok:
        failures.append(label)


def step(nu):
    f = lambda t: (3 * t - t**3) / 2  # noqa: E731
    return (1 - f(f(f(nu)))) / 2


# ---------------------------------------------------------------------------
# W1: one pair in 3-D
# ---------------------------------------------------------------------------
def pair_checks():
    ra = sp.symbols("ax ay az")
    rb = sp.symbols("bx by bz")
    pt = sp.Matrix([sp.Rational(7, 5), sp.Rational(-1, 3), sp.Rational(11, 7)])
    ca, cb = sp.Matrix(ra), sp.Matrix(rb)
    dist = lambda u, v: sp.sqrt(sum((u[i] - v[i]) ** 2 for i in range(3)))  # noqa: E731
    r_a, r_b, rab = dist(pt, ca), dist(pt, cb), dist(ca, cb)
    n = r_a - r_b
    mu = n / rab
    coords = list(ra) + list(rb)
    sub = {
        ra[0]: sp.Rational(1, 3),
        ra[1]: sp.Rational(1, 5),
        ra[2]: sp.Rational(-1, 2),
        rb[0]: sp.Rational(-1, 4),
        rb[1]: sp.Rational(3, 5),
        rb[2]: sp.Rational(1, 7),
    }
    at = lambda e: sp.N(e.subs(sub), 40)  # noqa: E731
    mu0, rv = at(mu), at(rab)
    worst_x = worst_xy = worst_s = 0
    scale_xy = scale_s = 0
    m = sp.Symbol("m")
    s_expr = step(m + sp.Rational(1, 5) * (1 - m**2))
    s_m = sp.diff(s_expr, m)
    s_mm = sp.diff(s_expr, m, 2)
    s_pairs = {(0, 0), (0, 3), (1, 4), (2, 5), (3, 3), (5, 5)}
    for i, xi in enumerate(coords):
        n_x = at(sp.diff(n, xi))
        r_x = at(sp.diff(rab, xi))
        mu_x_form = n_x / rv - mu0 * r_x / rv
        worst_x = max(worst_x, abs(mu_x_form - at(sp.diff(mu, xi))))
        for j, xj in enumerate(coords):
            n_y = at(sp.diff(n, xj))
            r_y = at(sp.diff(rab, xj))
            n_xy = at(sp.diff(n, xi, xj))
            r_xy = at(sp.diff(rab, xi, xj))
            mu_xy_form = (
                n_xy / rv
                - (n_x * r_y + n_y * r_x) / rv**2
                - mu0 * r_xy / rv
                + 2 * mu0 * r_x * r_y / rv**2
            )
            exact = at(sp.diff(mu, xi, xj))
            worst_xy = max(worst_xy, abs(mu_xy_form - exact))
            scale_xy = max(scale_xy, abs(exact))
            # chain rule through s(mu), s = step(nu(mu)), a = 1/5 constant; the
            # composite is differentiated for a spread of coordinate pairs only
            if (i, j) in s_pairs:
                mu_x_i = at(sp.diff(mu, xi))
                mu_x_j = at(sp.diff(mu, xj))
                sm0 = sp.N(s_m.subs(m, mu0), 40)
                smm0 = sp.N(s_mm.subs(m, mu0), 40)
                s_form = smm0 * mu_x_i * mu_x_j + sm0 * exact
                s_exact = at(
                    sp.diff(step(mu + sp.Rational(1, 5) * (1 - mu**2)), xi, xj)
                )
                worst_s = max(worst_s, abs(s_form - s_exact))
                scale_s = max(scale_s, abs(s_exact))
    check(
        "W1 mu_x, mu_xy and s_xy pair formulas equal the exact derivatives (claim W1)",
        worst_x < 1e-30
        and worst_xy < 1e-30
        and worst_s < 1e-28
        and scale_xy > 1e-3
        and scale_s > 1e-3,
        f"mu_x {float(worst_x):.1e} mu_xy {float(worst_xy):.1e} s_xy {float(worst_s):.1e} "
        f"scales {float(scale_xy):.1e} {float(scale_s):.1e}",
    )


# ---------------------------------------------------------------------------
# W2: log-form assembly of w''
# ---------------------------------------------------------------------------
def cell_checks():
    x = sp.symbols("xa xb xc")
    pt = sp.Integer(1)
    a_adj = {
        (0, 1): sp.Rational(1, 7),
        (0, 2): sp.Rational(-1, 5),
        (1, 2): sp.Rational(1, 9),
    }

    def a_of(i, j):
        return a_adj[(i, j)] if i < j else -a_adj[(j, i)]

    def s_pair(i, j):
        # the point sits between nucleus 0 and nuclei 1, 2, so the signs are fixed
        r_i = pt - x[i] if i == 0 else x[i] - pt
        r_j = pt - x[j] if j == 0 else x[j] - pt
        r_ij = (x[j] - x[i]) if i < j else (x[i] - x[j])
        mu = (r_i - r_j) / r_ij
        return step(mu + a_of(i, j) * (1 - mu**2))

    cells = []
    for a in range(3):
        p = 1
        for b in range(3):
            if b != a:
                p *= s_pair(a, b)
        cells.append(p)
    t_sum = sum(cells)
    home = 1
    w = cells[home] / t_sum
    sub = {
        x[0]: sp.Rational(1, 10),
        x[1]: sp.Rational(13, 10),
        x[2]: sp.Rational(29, 10),
    }
    at = lambda e: sp.N(e.subs(sub), 40)  # noqa: E731

    ell = []
    for a in range(3):
        ell.append(sum(sp.log(s_pair(a, b)) for b in range(3) if b != a))

    def grad_hess(f):
        g = sp.Matrix([sp.diff(f, xi) for xi in x])
        h = sp.Matrix(3, 3, lambda i, j: sp.diff(f, x[i], x[j]))
        return g, h

    p_val, p_g, p_h = [], [], []
    for a in range(3):
        lg, lh = grad_hess(ell[a])
        pv = at(cells[a])
        lg0 = lg.applyfunc(at)
        lh0 = lh.applyfunc(at)
        p_val.append(pv)
        p_g.append(pv * lg0)
        p_h.append(pv * (lh0 + lg0 * lg0.T))
    t0 = sum(p_val)
    tg = sum(p_g, sp.zeros(3, 1))
    th = sum(p_h, sp.zeros(3, 3))
    ph_g, ph_h = p_g[home], p_h[home]
    w2 = (
        ph_h / t0
        - (ph_g * tg.T + tg * ph_g.T) / t0**2
        - p_val[home] * th / t0**2
        + 2 * p_val[home] * tg * tg.T / t0**3
    )
    exact = sp.Matrix(3, 3, lambda i, j: at(sp.diff(w, x[i], x[j])))
    err = max(abs(w2[i, j] - exact[i, j]) for i in range(3) for j in range(3))
    # non-vacuity: a point deep inside one cell would give derivatives that are
    # zero to 40 digits and every term would pass trivially
    scale = max(abs(exact[i, j]) for i in range(3) for j in range(3))
    last = 2 * p_val[home] * tg * tg.T / t0**3
    t_scale = max(abs(last[i, j]) for i in range(3) for j in range(3))
    check(
        "W2 log-form w'' assembly equals the exact second derivative (claim W2)",
        err < 1e-30 and scale > 1e-3 and t_scale > 1e-9,
        f"max err {float(err):.1e} |w''| {float(scale):.1e} last term {float(t_scale):.1e}",
    )


def main():
    pair_checks()
    cell_checks()
    if failures:
        print(f"{len(failures)} check(s) failed")
        sys.exit(1)
    print("all Becke weight Hessian identities verified")


if __name__ == "__main__":
    main()
