#!/usr/bin/env python3
"""Sympy proof of the molecular-dynamics integrators in ferric-md.

Checks the algebra behind the integrators in crates/ferric-md/src/driver.rs
(theory: `book/src/28-molecular-dynamics/01-integrators-and-thermostats.md`).
Every check has a symbolic part and a numerical cross-check in 40-digit
arithmetic (mpmath) that does not reuse the symbolic expression.

Checks (tags are the `claim=` names in the book's verified boxes):

  verlet-symplectic         the velocity-Verlet map (x, p) -> (x', p') for an
                            ABSTRACT potential V has Jacobian determinant 1
                            (area preservation); numerical: a finite-difference
                            Jacobian of a Morse step has determinant 1.
  verlet-time-reversal      flip p, step, flip p, step returns (x, p) for an
                            abstract force; numerical: 100 Morse steps forward,
                            flip, 100 steps back.
  verlet-shadow-energy      on the harmonic oscillator
                            H~ = p^2/2m + (k x^2/2)(1 - w^2 dt^2/4), w^2 = k/m,
                            is exactly conserved by one step, the step matrix has
                            cos(theta) = 1 - w^2 dt^2/2; the true energy
                            differs from H~ by k x^2 w^2 dt^2/8, which is the
                            O(dt^2) energy error; numerical: 2000 steps.
  baoab-ou-variance         the O step v' = c v + sqrt((1-c^2) kT/m) xi,
                            c = exp(-gamma dt), preserves the variance kT/m
                            and is the exact solution of the OU variance ODE;
                            numerical: Gauss-Hermite quadrature of the update.
  baoab-harmonic-config     for BAOAB on the harmonic oscillator the stationary
                            covariance of (x, v) solves S = M S M^T + N; the
                            result is <x^2> = kT/k exactly (no dt error),
                            <x v> = 0, <v^2> = (kT/m)(1 - w^2 dt^2/4);
                            numerical: the same linear system solved in mpmath
                            and the Lyapunov series sum_n M^n N M^nT.
  berendsen-rate            lambda^2 = 1 + (dt/tau)(T0/T - 1) gives
                            (T' - T)/dt = (T0 - T)/tau exactly, the explicit
                            Euler step of dT/dt = (T0 - T)/tau; numerical: the
                            iterated map converges to T0 + (T(0)-T0) e^{-t/tau}
                            at first order in dt.

Mutations (each turns the script red; recorded in
crates/ferric-python/tests/test_derivation_scripts.py):
  - the second half-kick uses the old position (`pn = ph - dt / 2 * g(xn)` ->
    `g(xx)` in `vv_step`), marker "FAIL [verlet-symplectic]";
  - the first half kick doubled in the numeric reversal check
    (`ph = pp - h / 2 * g(xx)` -> `h * g(xx)` in `vv_step_numeric`), marker
    "FAIL [verlet-time-reversal]";
  - the shadow factor 1/4 -> 1/2, marker "FAIL [verlet-shadow-energy]";
  - the OU noise amplitude (1 - c^2) -> (1 - c), marker
    "FAIL [baoab-ou-variance]";
  - the O friction factor c -> c^2 in the BAOAB matrix, marker
    "FAIL [baoab-harmonic-config]";
  - the Berendsen factor T0/T - 1 -> T/T0 - 1, marker
    "FAIL [berendsen-rate]".

Exit status is nonzero on any mismatch.

Run: python3 scripts/md_integrators_sympy.py
"""

import sys

import mpmath as mp
import sympy as sp

FAILS = []
mp.mp.dps = 40


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        print(f"FAIL [{name.split()[0]}] {detail}")
        FAILS.append(name)


# ---------------------------------------------------------------------------
# velocity Verlet: symbolic with an abstract force
# ---------------------------------------------------------------------------
x, p, m, dt = sp.symbols("x p m dt", positive=True)
gf = sp.Function("g")  # g = dV/dx, the gradient


def vv_step(xx, pp, g=gf):
    """kick dt/2, drift dt, new force, kick dt/2 (driver.rs velocity Verlet)."""
    ph = pp - dt / 2 * g(xx)
    xn = xx + dt * ph / m
    pn = ph - dt / 2 * g(xn)  # second kick at the NEW position
    return xn, pn


def claim_symplectic():
    xn, pn = vv_step(x, p)
    jac = sp.Matrix(
        [[sp.diff(xn, x), sp.diff(xn, p)], [sp.diff(pn, x), sp.diff(pn, p)]]
    )
    det = sp.simplify(jac.det())
    ok_sym = det == 1
    check(
        "verlet-symplectic symbolic: det J = 1 for abstract V", ok_sym, f"det = {det}"
    )

    # numerical: Morse V = D (1 - exp(-a (x-1)))^2, finite-difference Jacobian
    D, a, mm, h = mp.mpf(3), mp.mpf("1.3"), mp.mpf(2), mp.mpf("0.07")

    def g(xx):
        return 2 * D * a * (1 - mp.exp(-a * (xx - 1))) * mp.exp(-a * (xx - 1))

    def step(z):
        xx, pp = z
        ph = pp - h / 2 * g(xx)
        xn_ = xx + h * ph / mm
        pn_ = ph - h / 2 * g(xn_)
        return [xn_, pn_]

    z0 = [mp.mpf("1.4"), mp.mpf("0.6")]
    eps = mp.mpf("1e-15")
    cols = []
    for i in range(2):
        zp, zm = list(z0), list(z0)
        zp[i] += eps
        zm[i] -= eps
        fp, fm = step(zp), step(zm)
        cols.append([(fp[j] - fm[j]) / (2 * eps) for j in range(2)])
    detn = cols[0][0] * cols[1][1] - cols[1][0] * cols[0][1]
    check(
        "verlet-symplectic numeric: det of finite-difference Jacobian of a Morse step",
        abs(detn - 1) < mp.mpf("1e-25"),
        f"|det-1| = {mp.nstr(abs(detn - 1), 3)}",
    )


def vv_step_numeric(xx, pp, h, mm, g):
    ph = pp - h / 2 * g(xx)
    xn = xx + h * ph / mm
    pn = ph - h / 2 * g(xn)
    return xn, pn


def claim_time_reversal():
    xn, pn = vv_step(x, p)
    # reversed step: start at (xn, -pn), step, flip the momentum
    xr, pr = vv_step(xn, -pn)
    pr = -pr
    dx = sp.simplify(sp.expand(xr - x))
    dp = sp.simplify(sp.expand(pr - p))
    check(
        "verlet-time-reversal symbolic: R S R S = id",
        dx == 0 and dp == 0,
        f"dx={dx}, dp={dp}",
    )

    D, a, mm, h = mp.mpf(3), mp.mpf("1.3"), mp.mpf(2), mp.mpf("0.05")

    def g(xx):
        return 2 * D * a * (1 - mp.exp(-a * (xx - 1))) * mp.exp(-a * (xx - 1))

    x0, p0 = mp.mpf("1.4"), mp.mpf("0.6")
    xx, pp = x0, p0
    for _ in range(100):
        xx, pp = vv_step_numeric(xx, pp, h, mm, g)
    pp = -pp
    for _ in range(100):
        xx, pp = vv_step_numeric(xx, pp, h, mm, g)
    pp = -pp
    err = max(abs(xx - x0), abs(pp - p0))
    check(
        "verlet-time-reversal numeric: 100 steps, flip, 100 steps",
        err < mp.mpf("1e-25"),
        f"err = {mp.nstr(err, 3)}",
    )


def claim_shadow():
    k = sp.symbols("k", positive=True)
    w2 = k / m
    xn, pn = vv_step(x, p, g=lambda q: k * q)
    shadow = lambda xx, pp: pp**2 / (2 * m) + k * xx**2 / 2 * (1 - w2 * dt**2 / 4)
    d = sp.simplify(sp.expand(shadow(xn, pn) - shadow(x, p)))
    check(
        "verlet-shadow-energy symbolic: H~ conserved by one step", d == 0, f"dH~ = {d}"
    )
    true_e = lambda xx, pp: pp**2 / (2 * m) + k * xx**2 / 2
    gap = sp.simplify(true_e(x, p) - shadow(x, p) - k * x**2 * w2 * dt**2 / 8)
    check(
        "verlet-shadow-energy symbolic: H - H~ = k x^2 w^2 dt^2 / 8",
        gap == 0,
        f"gap = {gap}",
    )

    # the harmonic step is the matrix R with entries d(x', p')/d(x, p); its trace is
    # 2 - w^2 dt^2, so the discrete angle satisfies cos(theta) = 1 - w^2 dt^2 / 2
    r = sp.Matrix([[sp.diff(xn, x), sp.diff(xn, p)], [sp.diff(pn, x), sp.diff(pn, p)]])
    tr = sp.simplify(r.trace() / 2 - (1 - w2 * dt**2 / 2))
    check(
        "verlet-shadow-energy symbolic: cos(theta) = 1 - w^2 dt^2 / 2",
        tr == 0,
        f"residual = {tr}",
    )

    kk, mm, h = mp.mpf(2), mp.mpf(3), mp.mpf("0.2")
    ww = kk / mm
    sh = lambda xx, pp: pp**2 / (2 * mm) + kk * xx**2 / 2 * (1 - ww * h**2 / 4)
    te = lambda xx, pp: pp**2 / (2 * mm) + kk * xx**2 / 2
    xx, pp = mp.mpf("0.8"), mp.mpf("0.3")
    s0, e0 = sh(xx, pp), te(xx, pp)
    drift, span = mp.mpf(0), mp.mpf(0)
    for _ in range(2000):
        xx, pp = vv_step_numeric(xx, pp, h, mm, lambda q: kk * q)
        drift = max(drift, abs(sh(xx, pp) - s0))
        span = max(span, abs(te(xx, pp) - e0))
    check(
        "verlet-shadow-energy numeric: 2000 steps, H~ constant while H oscillates at O(dt^2)",
        drift < mp.mpf("1e-30") and span > mp.mpf("1e-4"),
        f"H~ drift {mp.nstr(drift, 3)}, H span {mp.nstr(span, 3)}",
    )


# ---------------------------------------------------------------------------
# BAOAB
# ---------------------------------------------------------------------------
def claim_ou_variance():
    c, kT, gam, t, s2 = sp.symbols("c kT gamma t s2", positive=True)
    cc = sp.exp(-gam * dt)
    amp2 = (1 - cc**2) * kT / m  # variance of the noise term  (OU_AMP)
    var_new = cc**2 * (kT / m) + amp2
    check(
        "baoab-ou-variance symbolic: Var v' = kT/m when Var v = kT/m",
        sp.simplify(var_new - kT / m) == 0,
        f"diff = {sp.simplify(var_new - kT / m)}",
    )
    # exact OU variance solution: s(t) = kT/m + (s0 - kT/m) exp(-2 gamma t), ODE ds/dt = -2 gamma s + 2 gamma kT/m
    s0 = sp.symbols("s0", positive=True)
    s_t = kT / m + (s0 - kT / m) * sp.exp(-2 * gam * t)
    ode = sp.simplify(sp.diff(s_t, t) - (-2 * gam * s_t + 2 * gam * kT / m))
    one_step = sp.simplify(cc**2 * s0 + amp2 - s_t.subs(t, dt))
    check(
        "baoab-ou-variance symbolic: the O step equals the exact OU flow for any initial variance",
        ode == 0 and one_step == 0,
        f"ode residual {ode}, step residual {one_step}",
    )
    # numeric: Gauss-Hermite quadrature of E[(c v + sqrt((1-c^2) kT/m) xi)^2], v ~ N(0, kT/m)
    g_, h_, kT_, m_ = mp.mpf("0.9"), mp.mpf("0.4"), mp.mpf("1.7"), mp.mpf("2.3")
    c_ = mp.exp(-g_ * h_)
    s_ = mp.sqrt((1 - c_**2) * kT_ / m_)
    sv = mp.sqrt(kT_ / m_)
    # 2-D tensor Gauss-Hermite (probabilists' weight) with 8 nodes per axis, exact for degree 15
    nodes, weights = zip(
        *[(mp.sqrt(2) * xn, w / mp.sqrt(mp.pi)) for xn, w in _gauss_hermite(8)]
    )
    second = mp.mpf(0)
    for xv, wv in zip(nodes, weights):
        for xi, wi in zip(nodes, weights):
            second += wv * wi * (c_ * sv * xv + s_ * xi) ** 2
    check(
        "baoab-ou-variance numeric: quadrature of the O update preserves kT/m",
        abs(second - kT_ / m_) < mp.mpf("1e-30"),
        f"|var - kT/m| = {mp.nstr(abs(second - kT_ / m_), 3)}",
    )


def _gauss_hermite(n):
    """Nodes and weights of physicists' Gauss-Hermite from mpmath's Hermite roots."""
    out = []
    xs = mp.polyroots(
        mp.taylor(lambda z: mp.hermite(n, z), 0, n)[::-1], maxsteps=200, extraprec=200
    )
    for xr in xs:
        xr = mp.re(xr)
        w = (
            2 ** (n - 1)
            * mp.factorial(n)
            * mp.sqrt(mp.pi)
            / (n**2 * mp.hermite(n - 1, xr) ** 2)
        )
        out.append((xr, w))
    return out


def baoab_matrices(k, mm, h, c):
    """Linear map M of one BAOAB step on (x, v) and the noise-propagation matrix G
    (the noise enters at O and is then carried by A, B)."""
    half = h / 2
    B = sp.Matrix([[1, 0], [-half * k / mm, 1]])
    A = sp.Matrix([[1, half], [0, 1]])
    O = sp.Matrix([[1, 0], [0, c]])
    return B * A * O * A * B, B * A


def claim_baoab_config():
    k, kT, gam = sp.symbols("k kT gamma", positive=True)
    c = sp.exp(-gam * dt)  # exp(-gamma dt)
    M, G = baoab_matrices(k, m, dt, c)
    noise = G * sp.Matrix([[0, 0], [0, (1 - c**2) * kT / m]]) * G.T
    sxx, sxv, svv = sp.symbols("sxx sxv svv")
    S = sp.Matrix([[sxx, sxv], [sxv, svv]])
    sol = sp.solve(list(S - (M * S * M.T + noise)), [sxx, sxv, svv], dict=True)[0]
    w2 = k / m
    ok = (
        sp.simplify(sol[sxx] - kT / k) == 0
        and sp.simplify(sol[sxv]) == 0
        and sp.simplify(sol[svv] - kT / m * (1 - w2 * dt**2 / 4)) == 0
    )
    check(
        "baoab-harmonic-config symbolic: <x^2> = kT/k, <xv> = 0, <v^2> = (kT/m)(1 - w^2 dt^2/4)",
        ok,
        f"<x^2> = {sp.simplify(sol[sxx])}, <v^2> = {sp.simplify(sol[svv])}",
    )

    # numeric: Lyapunov series sum_n M^n N M^nT in mpmath, no symbolic solve
    kk, mm, h, kTn, g_ = (
        mp.mpf("1.9"),
        mp.mpf("1.3"),
        mp.mpf("0.3"),
        mp.mpf("0.7"),
        mp.mpf("1.1"),
    )
    cn = mp.exp(-g_ * h)
    Mn, Gn = baoab_matrices(
        sp.Float(str(kk), 40),
        sp.Float(str(mm), 40),
        sp.Float(str(h), 40),
        sp.Float(str(cn), 40),
    )
    Mn = mp.matrix([[mp.mpf(str(Mn[i, j])) for j in range(2)] for i in range(2)])
    Gn = mp.matrix([[mp.mpf(str(Gn[i, j])) for j in range(2)] for i in range(2)])
    Nn = Gn * mp.matrix([[0, 0], [0, (1 - cn**2) * kTn / mm]]) * Gn.T
    S = mp.zeros(2, 2)
    term = Nn
    Mp = mp.eye(2)
    for _ in range(4000):
        S += term
        Mp = Mn * Mp
        term = Mp * Nn * Mp.T
        if abs(term[0, 0]) + abs(term[1, 1]) < mp.mpf("1e-45"):
            break
    err = max(
        abs(S[0, 0] - kTn / kk),
        abs(S[0, 1]),
        abs(S[1, 1] - kTn / mm * (1 - kk / mm * h**2 / 4)),
    )
    check(
        "baoab-harmonic-config numeric: Lyapunov series gives the same covariance",
        err < mp.mpf("1e-25"),
        f"err = {mp.nstr(err, 3)}",
    )


# ---------------------------------------------------------------------------
# Berendsen
# ---------------------------------------------------------------------------
def claim_berendsen():
    T, T0, tau = sp.symbols("T T0 tau", positive=True)
    lam2 = 1 + dt / tau * (T0 / T - 1)  # BERENDSEN_LAMBDA2
    rate = sp.simplify((lam2 * T - T) / dt - (T0 - T) / tau)
    check(
        "berendsen-rate symbolic: (T' - T)/dt = (T0 - T)/tau",
        rate == 0,
        f"residual = {rate}",
    )
    # equivalently the velocity factor is lambda, and kinetic energy scales with lambda^2
    # numeric: iterate to t = 3 tau and compare with the exact exponential relaxation
    tau_n, T0n, Tn = mp.mpf(2), mp.mpf(300), mp.mpf(100)
    errs = []
    for steps in (100, 200):
        h = 3 * tau_n / steps
        t = Tn
        for _ in range(steps):
            t = t * (1 + h / tau_n * (T0n / t - 1))
        exact = T0n + (Tn - T0n) * mp.exp(-3)
        errs.append(abs(t - exact))
    ratio = errs[0] / errs[1]
    check(
        "berendsen-rate numeric: error halves when dt halves (first order), relaxes toward T0",
        mp.mpf("1.9") < ratio < mp.mpf("2.1"),
        f"errors {mp.nstr(errs[0], 4)}, {mp.nstr(errs[1], 4)}, ratio {mp.nstr(ratio, 4)}",
    )


def main():
    claim_symplectic()
    claim_time_reversal()
    claim_shadow()
    claim_ou_variance()
    claim_baoab_config()
    claim_berendsen()
    if FAILS:
        print(f"{len(FAILS)} check(s) FAILED")
        sys.exit(1)
    print("all md integrator checks passed")


if __name__ == "__main__":
    main()
