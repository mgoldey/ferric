#!/usr/bin/env python3
"""Sympy/mpmath checks of the identities behind ferric-pbc's Ewald sums, pair
Fourier transform, Ewald-split nuclear attraction, exxdiv correction and
strain derivatives. Theory: crates/ferric-pbc/theory.md (section numbers below
match its sections).

Each claim asserts; the script exits nonzero on the first mismatch.

  S1  Ewald omega-independence: d/domega of the SR, LR, self and background
      terms cancel (symbolic, via the Gaussian Poisson sum), and a pure-python
      Ewald sum on a CHARGED triclinic toy cell is omega-independent to 1e-12;
      textbook anchors: simple-cubic v_M*a = 2.83729747948062, rock-salt
      Madelung 1.74756459463318; negative control: dropping the background
      term makes the charged-cell sum omega-dependent.
      Rust: crates/ferric-pbc/src/ewald.rs ewald_point_charges, madelung_constant.
  S2  Pair FT: Gaussian product theorem (exact exponent identity), the 1-D
      Gaussian FT, (-iG)^t per Hermite order, and the McMurchie-Davidson
      E-coefficient expansion for (l_a, l_b) = (1,0), (1,1), (2,0) vs mpmath
      quadrature of the complex 1-D integral.
      Rust: crates/ferric-pbc/src/pair_ft/plan.rs walk/site_terms/f_rows.
  S3  Ewald-split nuclear attraction: FT[erfc(wr)/r] = 4pi/G^2 (1 - e^{-G^2/4w^2}),
      its G -> 0 limit = int erfc(wr)/r d^3r = pi/w^2 (the V_G0 / background
      constant c0 = pi/(w^2 Omega)), and the omega-cancellation of V_SR + V_LR + V_G0.
      Rust: crates/ferric-pbc/src/hcore.rs periodic_hcore_impl (v_g0), reciprocal_nuclear.
  S4  exxdiv: the Madelung shift K += v_M S D S of a closed-shell idempotent
      D gives Delta E = -N v_M / 2 and Fock term -1/2 v_M S D S.
      Rust: crates/ferric-pbc/src/dense_aft.rs DenseAftK::build.
  S5  Strain: dOmega/deps_ab = Omega delta_ab, d(G^2)/deps_ab = -2 G_a G_b at fixed
      Miller index, d|d|/deps_ab = d_a d_b/|d|, the LR kernel derivative
      k' = -k (1/4w^2 + 1/G^2); the pair-FT strain formula for s and p_x bra
      shells; pure-python FD of the Ewald energy vs the analytic strain parts
      (frozen index sets) and Euler homogeneity tr dv_M/deps = -v_M.
      Rust: crates/ferric-pbc/src/ewald.rs ewald_point_charges_strain,
      madelung_strain; crates/ferric-pbc/src/pair_ft.rs pair_ft_strain_chunked.

Run: python3 scripts/pbc_ewald_pairft_stress_sympy.py   (about a minute)
"""

import math
import sys

import mpmath as mp
import sympy as sp

FAILURES = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def is_zero(expr):
    return sp.simplify(expr) == 0


# ---------------------------------------------------------------------------
# Pure-python Ewald (mirror of ewald.rs, fixed integer index sets)
# ---------------------------------------------------------------------------


def det3(m):
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def recip(lat):
    """Rows b_i with a_i . b_j = 2 pi delta_ij."""
    a1, a2, a3 = lat
    vol = det3(lat)

    def cross(u, v):
        return [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]

    f = 2.0 * math.pi / vol
    return [
        [f * x for x in cross(a2, a3)],
        [f * x for x in cross(a3, a1)],
        [f * x for x in cross(a1, a2)],
    ]


def comb(n, rows):
    return [n[0] * rows[0][c] + n[1] * rows[1][c] + n[2] * rows[2][c] for c in range(3)]


def index_sets(lat, omega_min, omega_max, s=7.1):
    """Integer translation / Miller sets covering rcut = s/omega_min and
    gcut = 2 s omega_max on the REFERENCE lattice (frozen afterwards)."""
    b = recip(lat)
    rcut, gcut = s / omega_min + 6.0, 2.0 * s * omega_max
    nl = [
        int(rcut * math.sqrt(sum(x * x for x in b[i])) / (2 * math.pi)) + 1
        for i in range(3)
    ]
    ng = [
        int(gcut * math.sqrt(sum(x * x for x in lat[i])) / (2 * math.pi)) + 1
        for i in range(3)
    ]
    ls, gs = [], []
    for i in range(-nl[0], nl[0] + 1):
        for j in range(-nl[1], nl[1] + 1):
            for k in range(-nl[2], nl[2] + 1):
                v = comb((i, j, k), lat)
                if math.sqrt(sum(x * x for x in v)) <= rcut:
                    ls.append((i, j, k))
    for i in range(-ng[0], ng[0] + 1):
        for j in range(-ng[1], ng[1] + 1):
            for k in range(-ng[2], ng[2] + 1):
                if (i, j, k) == (0, 0, 0):
                    continue
                v = comb((i, j, k), b)
                if math.sqrt(sum(x * x for x in v)) <= gcut:
                    gs.append((i, j, k))
    return ls, gs


def ewald(lat, q, pos, omega, sets, background=True, strain=False):
    """Energy (and, with strain=True, the analytic dE/deps parts as one 3x3)."""
    ls, gs = sets
    b = recip(lat)
    vol = det3(lat)
    sr, dsr = [], [[[] for _ in range(3)] for _ in range(3)]
    tws = 2.0 * omega / math.sqrt(math.pi)
    for n in ls:
        L = comb(n, lat)
        for i, (qi, ri) in enumerate(zip(q, pos)):
            for j, (qj, rj) in enumerate(zip(q, pos)):
                if n == (0, 0, 0) and i == j:
                    continue
                d = [ri[c] - rj[c] - L[c] for c in range(3)]
                r = math.sqrt(sum(x * x for x in d))
                sr.append(0.5 * qi * qj * math.erfc(omega * r) / r)
                if strain:
                    fp = -(
                        math.erfc(omega * r) / r**2
                        + tws * math.exp(-((omega * r) ** 2)) / r
                    )
                    for a_ in range(3):
                        for b_ in range(3):
                            dsr[a_][b_].append(0.5 * qi * qj * fp * d[a_] * d[b_] / r)
    lr, dlr = [], [[[] for _ in range(3)] for _ in range(3)]
    for n in gs:
        g = comb(n, b)
        g2 = sum(x * x for x in g)
        re = sum(
            qq * math.cos(sum(g[c] * r[c] for c in range(3))) for qq, r in zip(q, pos)
        )
        im = -sum(
            qq * math.sin(sum(g[c] * r[c] for c in range(3))) for qq, r in zip(q, pos)
        )
        k = math.exp(-g2 / (4 * omega**2)) / g2
        s2 = re * re + im * im
        lr.append(k * s2)
        if strain:
            kp = -k * (1 / (4 * omega**2) + 1 / g2)
            for a_ in range(3):
                for b_ in range(3):
                    dlr[a_][b_].append(s2 * kp * (-2.0) * g[a_] * g[b_])
    e_lr = 2 * math.pi / vol * math.fsum(lr)
    e_self = -omega / math.sqrt(math.pi) * sum(x * x for x in q)
    e_g0 = -math.pi * sum(q) ** 2 / (2 * vol * omega**2) if background else 0.0
    e = math.fsum([math.fsum(sr), e_lr, e_self, e_g0])
    if not strain:
        return e
    out = [[0.0] * 3 for _ in range(3)]
    for a_ in range(3):
        for b_ in range(3):
            v = math.fsum(dsr[a_][b_]) + 2 * math.pi / vol * math.fsum(dlr[a_][b_])
            if a_ == b_:
                v += -e_lr - e_g0
            out[a_][b_] = v
    return e, out


def strained(lat, pos, eps):
    m = [[(1.0 if i == j else 0.0) + eps[i][j] for j in range(3)] for i in range(3)]

    def ap(v):
        return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]

    return [ap(a) for a in lat], [ap(r) for r in pos]


# ---------------------------------------------------------------------------
# S1  Ewald omega-independence
# ---------------------------------------------------------------------------


def s1():
    r, w, G, x = sp.symbols("r omega G x", positive=True)
    # SR kernel derivative.
    check(
        "S1 d/dw erfc(wr)/r = -(2/sqrt(pi)) exp(-w^2 r^2)",
        is_zero(
            sp.diff(sp.erfc(w * r) / r, w) + 2 / sp.sqrt(sp.pi) * sp.exp(-(w**2) * r**2)
        ),
    )
    # 1-D Gaussian FT -> 3-D by separability.
    g1 = sp.symbols("g", real=True)
    ft1 = sp.integrate(
        sp.exp(-(w**2) * x**2 - sp.I * g1 * x), (x, -sp.oo, sp.oo), conds="none"
    )
    ok = is_zero(ft1 - sp.sqrt(sp.pi) / w * sp.exp(-(g1**2) / (4 * w**2)))
    check("S1 1-D FT of exp(-w^2 x^2) = sqrt(pi)/w exp(-g^2/4w^2)", ok)
    gauss_ft3 = (sp.pi / w**2) ** sp.Rational(3, 2) * sp.exp(-(G**2) / (4 * w**2))
    # LR kernel derivative equals minus the FT of the SR kernel derivative.
    k_lr = 4 * sp.pi / G**2 * sp.exp(-(G**2) / (4 * w**2))
    check(
        "S1 d/dw [4pi/G^2 e^{-G^2/4w^2}] = (2/sqrt(pi)) FT[e^{-w^2 r^2}]",
        is_zero(sp.diff(k_lr, w) - 2 / sp.sqrt(sp.pi) * gauss_ft3),
    )
    # Bookkeeping. Poisson: sum_L e^{-w^2|d-L|^2} = (1/Omega) sum_G FT(G) e^{iG.d}.
    # With T(G) = sum_ij Z_i Z_j e^{iG.(R_i-R_j)} = |S(G)|^2:
    #   dE_SR/dw = -(1/sqrt(pi)) [ (1/Omega) sum_G FT(G) |S(G)|^2 - sum_i Z_i^2 ]
    #   dE_LR/dw = (1/(2 Omega)) sum_{G!=0} dk_LR/dw |S(G)|^2
    #   dE_self/dw = -(1/sqrt(pi)) sum Z^2;  dE_G0/dw = pi Q^2/(Omega w^3)
    Om, Q, Z2, SG2 = sp.symbols("Omega Q Z2 S_G2", positive=True)
    ft0 = gauss_ft3.subs(G, 0)
    per_g_sr = -1 / sp.sqrt(sp.pi) / Om * gauss_ft3 * SG2
    per_g_lr = sp.Rational(1, 2) / Om * sp.diff(k_lr, w) * SG2
    check("S1 per G != 0: dE_SR/dw + dE_LR/dw = 0", is_zero(per_g_sr + per_g_lr))
    e_g0 = -sp.pi * Q**2 / (2 * Om * w**2)
    rest = (
        -1
        / sp.sqrt(sp.pi)
        * (ft0 * Q**2 / Om - Z2)  # SR: G = 0 Poisson term + excluded i=j,L=0
        + sp.diff(-w / sp.sqrt(sp.pi) * Z2, w)  # self
        + sp.diff(e_g0, w)  # background
    )
    check("S1 G = 0 + self + background derivatives cancel", is_zero(rest))

    # Numeric toy: CHARGED triclinic cell, three charges (net +2).
    lat = [[4.1, 0.0, 0.0], [0.7, 4.4, 0.0], [-0.4, 0.5, 4.8]]
    q = [1.0, 2.0, -1.0]
    pos = [[0.1, 0.2, 0.3], [1.9, 2.4, 1.1], [3.0, 0.8, 3.3]]
    sets = index_sets(lat, 0.6, 1.8)
    es = [ewald(lat, q, pos, om, sets) for om in (0.6, 1.1, 1.8)]
    spread = max(es) - min(es)
    check(
        "S1 charged triclinic toy: omega spread < 1e-12",
        spread < 1e-12,
        f"E={es[0]:.14f} spread={spread:.2e}",
    )
    esb = [ewald(lat, q, pos, om, sets, background=False) for om in (0.6, 1.8)]
    check(
        "S1 negative control: no background term -> omega-dependent",
        abs(esb[0] - esb[1]) > 1e-3,
        f"spread={abs(esb[0] - esb[1]):.2e}",
    )
    # Simple cubic Madelung (one unit charge, background): v_M a = 2.837297479480620.
    a = 3.0
    sc = [[a, 0, 0], [0, a, 0], [0, 0, a]]
    sets_sc = index_sets(sc, 1.0, 1.0)
    vm = -2.0 * ewald(sc, [1.0], [[0.0, 0.0, 0.0]], 1.0, sets_sc)
    check(
        "S1 simple-cubic v_M a = 2.8372974794806196",
        abs(vm * a - 2.8372974794806196) < 1e-11,
        f"{vm * a:.15f}",
    )
    # Rock salt (neutral, mixed signs): E = -8 M / a per conventional cell.
    a = 5.0
    fcc = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    pos_ns = [[a * f[0], a * f[1], a * f[2]] for f in fcc] + [
        [a * (f[0] + 0.5), a * f[1], a * f[2]] for f in fcc
    ]
    q_ns = [1.0] * 4 + [-1.0] * 4
    cub = [[a, 0, 0], [0, a, 0], [0, 0, a]]
    sets_ns = index_sets(cub, 0.7, 1.0)
    m_ns = [-ewald(cub, q_ns, pos_ns, om, sets_ns) * a / 8.0 for om in (0.7, 1.0)]
    check(
        "S1 rock-salt Madelung 1.747564594633182, omega-independent",
        abs(m_ns[0] - 1.747564594633182) < 1e-11 and abs(m_ns[0] - m_ns[1]) < 1e-12,
        f"{m_ns[0]:.15f}, {m_ns[1]:.15f}",
    )


# ---------------------------------------------------------------------------
# S2  Gaussian pair Fourier transform
# ---------------------------------------------------------------------------


def s2():
    x, A, B, G = sp.symbols("x A B G", real=True)
    a, b = sp.symbols("alpha beta", positive=True)
    p = a + b
    P = (a * A + b * B) / p
    # Gaussian product theorem (exact exponent identity).
    lhs = -a * (x - A) ** 2 - b * (x - B) ** 2
    rhs = -p * (x - P) ** 2 - a * b / p * (A - B) ** 2
    check("S2 Gaussian product theorem", is_zero(sp.expand(lhs - rhs)))
    # Hermite: d/dP of the FT phase e^{-iGP} brings (-iG).
    Pc = sp.symbols("P", real=True)
    s_ft = sp.sqrt(sp.pi / p) * sp.exp(-(G**2) / (4 * p) - sp.I * G * Pc)
    check(
        "S2 d/dP FT[Lambda_0] = (-iG) FT[Lambda_0]",
        is_zero(sp.diff(s_ft, Pc) - (-sp.I * G) * s_ft),
    )

    # E-coefficient expansions vs quadrature of the complex 1-D integral.
    mp.mp.dps = 30
    av, bv, Av, Bv, Gv = (
        mp.mpf("0.9"),
        mp.mpf("1.7"),
        mp.mpf("0.3"),
        mp.mpf("-0.55"),
        mp.mpf("1.3"),
    )
    pv = av + bv
    Pv = (av * Av + bv * Bv) / pv
    xpa, xpb = Pv - Av, Pv - Bv
    cases = {
        (0, 0): [mp.mpf(1)],
        (1, 0): [xpa, 1 / (2 * pv)],
        (1, 1): [xpa * xpb + 1 / (2 * pv), (xpa + xpb) / (2 * pv), 1 / (4 * pv**2)],
        (2, 0): [xpa**2 + 1 / (2 * pv), xpa / pv, 1 / (4 * pv**2)],
    }
    pref = (
        mp.sqrt(mp.pi / pv)
        * mp.exp(-av * bv / pv * (Av - Bv) ** 2)
        * mp.exp(-(Gv**2) / (4 * pv))
        * mp.exp(-1j * Gv * Pv)
    )
    worst = 0.0
    for (la, lb), e in cases.items():

        def f(t, la=la, lb=lb):
            return (
                (t - Av) ** la
                * (t - Bv) ** lb
                * mp.exp(-av * (t - Av) ** 2 - bv * (t - Bv) ** 2 - 1j * Gv * t)
            )

        num = mp.quad(f, [-mp.inf, Pv, mp.inf])
        closed = pref * sum(et * (-1j * Gv) ** t for t, et in enumerate(e))
        worst = max(worst, float(abs(num - closed)))
    check(
        "S2 MD expansion (l_a,l_b) in {00,10,11,20} vs quadrature",
        worst < 1e-20,
        f"max err {worst:.1e}",
    )


# ---------------------------------------------------------------------------
# S3  Ewald-split nuclear attraction
# ---------------------------------------------------------------------------


def s3():
    r, w, G = sp.symbols("r omega G", positive=True)
    vol_int = sp.integrate(4 * sp.pi * r * sp.erfc(w * r), (r, 0, sp.oo))
    check("S3 int erfc(wr)/r d^3r = pi/w^2", is_zero(vol_int - sp.pi / w**2))
    ft_sr = 4 * sp.pi / G**2 * (1 - sp.exp(-(G**2) / (4 * w**2)))
    check(
        "S3 lim_{G->0} FT[erfc(wr)/r] = pi/w^2",
        is_zero(sp.limit(ft_sr, G, 0) - sp.pi / w**2),
    )
    # FT[erfc(wr)/r](G) = (4 pi/G) int_0^inf sin(Gr) erfc(wr) dr, by quadrature.
    mp.mp.dps = 25
    worst = 0.0
    for wv, gv in ((0.8, 0.7), (1.3, 2.1), (2.0, 5.0)):
        num = (
            4
            * mp.pi
            / gv
            * mp.quad(
                lambda t, gv=gv, wv=wv: mp.sin(gv * t) * mp.erfc(wv * t), [0, mp.inf]
            )
        )
        worst = max(
            worst,
            float(abs(num - 4 * mp.pi / gv**2 * (1 - mp.exp(-(gv**2) / (4 * wv**2))))),
        )
    check(
        "S3 FT[erfc(wr)/r] = 4pi/G^2 (1 - e^{-G^2/4w^2}) (quadrature)",
        worst < 1e-15,
        f"{worst:.1e}",
    )
    # omega-cancellation per G and at G = 0 (Z_tot = Z, rho(G) = P(G)):
    Om, Z = sp.symbols("Omega Z", positive=True)
    v_sr_g = -Z / Om * ft_sr  # V_SR as a reciprocal sum, per G != 0
    v_lr_g = -Z / Om * 4 * sp.pi / G**2 * sp.exp(-(G**2) / (4 * w**2))
    check(
        "S3 V_SR + V_LR per G != 0 is the bare -4pi Z/(Omega G^2)",
        is_zero(v_sr_g + v_lr_g + 4 * sp.pi * Z / (Om * G**2)),
    )
    v_sr_0 = -Z / Om * sp.pi / w**2
    v_g0 = sp.pi * Z / (w**2 * Om)
    check(
        "S3 V_G0 = +pi Z_tot/(w^2 Omega) cancels the SR G = 0 average",
        is_zero(v_sr_0 + v_g0),
    )


# ---------------------------------------------------------------------------
# S4  exxdiv (Madelung) correction
# ---------------------------------------------------------------------------


def s4():
    vm = sp.symbols("v_M", positive=True)
    S = sp.Matrix(
        [
            [1, sp.Rational(1, 3), sp.Rational(1, 5)],
            [sp.Rational(1, 3), 1, sp.Rational(1, 4)],
            [sp.Rational(1, 5), sp.Rational(1, 4), 1],
        ]
    )
    # Two S-orthonormal occupied orbitals by Gram-Schmidt in the S metric.
    c = [sp.Matrix([1, 0, 0]), sp.Matrix([0, 1, 1])]
    occ = []
    for v in c:
        for u in occ:
            v = v - (u.T * S * v)[0] * u
        occ.append(v / sp.sqrt((v.T * S * v)[0]))
    C = sp.Matrix.hstack(*occ)
    D = 2 * C * C.T
    check(
        "S4 D S D = 2 D (closed-shell idempotency)",
        (D * S * D - 2 * D).applyfunc(sp.simplify) == sp.zeros(3),
    )
    N = sp.simplify((D * S).trace())
    dE = -sp.Rational(1, 4) * vm * (D * S * D * S).trace()
    check(
        "S4 Delta E = -(v_M/4) tr(DSDS) = -N v_M/2",
        is_zero(dE + N * vm / 2),
        f"N = {N}",
    )
    # Fock contribution: dE/dD (symmetric) = -1/2 v_M S D S = -1/2 K_M.
    d = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"d{min(i, j)}{max(i, j)}"))
    e = -sp.Rational(1, 4) * vm * (d * S * d * S).trace()
    grad = sp.Matrix(3, 3, lambda i, j: sp.diff(e, d[i, j]) / (1 if i == j else 2))
    check(
        "S4 dE/dD = -1/2 v_M S D S",
        (grad + vm / 2 * S * d * S).applyfunc(sp.expand) == sp.zeros(3),
    )


# ---------------------------------------------------------------------------
# S5  Strain derivatives
# ---------------------------------------------------------------------------


def s5():
    eps = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"e{i}{j}"))
    zero = {e: 0 for e in eps}
    M = sp.eye(3) + eps
    Gs = sp.Matrix(sp.symbols("G0:3", real=True))
    ds = sp.Matrix(sp.symbols("d0:3", real=True))
    lat0 = sp.Matrix(3, 3, sp.symbols("a0:9", real=True))  # rows = lattice vectors
    vol = (lat0 * M.T).det()
    ok_vol = all(
        is_zero(sp.diff(vol, eps[i, j]).subs(zero) - (lat0.det() if i == j else 0))
        for i in range(3)
        for j in range(3)
    )
    check("S5 dOmega/deps_ab = Omega delta_ab", ok_vol)
    Ge = M.T.inv() * Gs
    g2 = (Ge.T * Ge)[0]
    ok_g = all(
        is_zero(sp.diff(g2, eps[i, j]).subs(zero) + 2 * Gs[i] * Gs[j])
        for i in range(3)
        for j in range(3)
    )
    check("S5 d(G^2)/deps_ab = -2 G_a G_b (fixed Miller index)", ok_g)
    de = M * ds
    dn = sp.sqrt((de.T * de)[0])
    dn0 = sp.sqrt((ds.T * ds)[0])
    ok_d = all(
        is_zero(sp.diff(dn, eps[i, j]).subs(zero) - ds[i] * ds[j] / dn0)
        for i in range(3)
        for j in range(3)
    )
    check("S5 d|d|/deps_ab = d_a d_b / |d|", ok_d)
    w, X = sp.symbols("omega X", positive=True)  # X = G^2
    k = sp.exp(-X / (4 * w**2)) / X
    check(
        "S5 k'(G^2) = -k (1/4w^2 + 1/G^2)",
        is_zero(sp.diff(k, X) + k * (1 / (4 * w**2) + 1 / X)),
    )

    # Pair-FT strain formula (pair_ft.rs pair_ft_strain_chunked doc):
    #   dP/deps_ab = [Q_a + i G_a (alpha/p) P](A - B')_b + G_a G_b/(2p) P - G_a P^g_b
    al, be = sp.Rational(9, 10), sp.Rational(17, 10)
    p = al + be
    A = sp.Matrix(sp.symbols("A0:3", real=True))
    B = sp.Matrix(sp.symbols("B0:3", real=True))

    def pair(Av, Bv, Gv, poly):
        Pc = (al * Av + be * Bv) / p
        ab = Av - Bv
        g2v = (Gv.T * Gv)[0]
        s = (sp.pi / p) ** sp.Rational(3, 2) * sp.exp(
            -al * be / p * (ab.T * ab)[0] - g2v / (4 * p) - sp.I * (Gv.T * Pc)[0]
        )
        return s * poly(Av, Bv, Gv, Pc), s

    polys = {
        "s|s": lambda Av, Bv, Gv, Pc: sp.Integer(1),
        "px|s": lambda Av, Bv, Gv, Pc: (Pc[0] - Av[0]) + (-sp.I * Gv[0]) / (2 * p),
    }
    vals = {
        A[0]: sp.Rational(3, 10),
        A[1]: sp.Rational(-1, 5),
        A[2]: sp.Rational(7, 10),
        B[0]: sp.Rational(-11, 20),
        B[1]: sp.Rational(2, 5),
        B[2]: sp.Rational(1, 10),
        Gs[0]: sp.Rational(13, 10),
        Gs[1]: sp.Rational(-7, 10),
        Gs[2]: sp.Rational(1, 2),
    }
    for name, poly in polys.items():
        Pfull, s_only = pair(A, B, Gs, poly)
        Pstr, _ = pair(M * A, M * B, M.T.inv() * Gs, poly)
        worst = 0.0
        for a_ in range(3):
            Qa = sp.diff(Pfull, A[a_])
            for b_ in range(3):
                lhs = sp.diff(Pstr, eps[a_, b_]).subs(zero)
                Pg_b = s_only * sp.diff(poly(A, B, Gs, (al * A + be * B) / p), Gs[b_])
                rhs = (
                    (Qa + sp.I * Gs[a_] * al / p * Pfull) * (A - B)[b_]
                    + Gs[a_] * Gs[b_] / (2 * p) * Pfull
                    - Gs[a_] * Pg_b
                )
                diff = complex(sp.N((lhs - rhs).subs(vals), 40))
                worst = max(worst, abs(diff))
        check(
            f"S5 pair-FT strain formula, {name}",
            worst < 1e-30,
            f"max |lhs - rhs| {worst:.1e}",
        )

    # Numeric: analytic Ewald strain vs central FD at frozen index sets.
    lat = [[4.1, 0.0, 0.0], [0.7, 4.4, 0.0], [-0.4, 0.5, 4.8]]
    q = [1.0, 2.0, -1.0]
    pos = [[0.1, 0.2, 0.3], [1.9, 2.4, 1.1], [3.0, 0.8, 3.3]]
    om = 1.1
    sets = index_sets(lat, om, om)
    _, ana = ewald(lat, q, pos, om, sets, strain=True)
    h = 1e-5
    worst = 0.0
    for a_ in range(3):
        for b_ in range(3):
            e = [[0.0] * 3 for _ in range(3)]
            e[a_][b_] = h
            lp, pp = strained(lat, pos, e)
            e[a_][b_] = -h
            lm, pm = strained(lat, pos, e)
            fd = (ewald(lp, q, pp, om, sets) - ewald(lm, q, pm, om, sets)) / (2 * h)
            worst = max(worst, abs(fd - ana[a_][b_]))
    check(
        "S5 Ewald strain (SR virial + LR + background) vs FD",
        worst < 1e-7,
        f"max err {worst:.1e}",
    )
    # Euler homogeneity of v_M (degree -1 in lengths): tr dv_M/deps = -v_M.
    sets1 = index_sets(lat, 1.0, 1.0)
    vm, dv = ewald(lat, [1.0], [[0.0] * 3], 1.0, sets1, strain=True)
    vm, tr = -2 * vm, -2 * (dv[0][0] + dv[1][1] + dv[2][2])
    check(
        "S5 tr dv_M/deps = -v_M (Euler homogeneity)",
        abs(tr + vm) < 1e-12,
        f"v_M={vm:.12f} tr={tr:.12f}",
    )


if __name__ == "__main__":
    for f in (s1, s2, s3, s4, s5):
        f()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("all checks passed")
