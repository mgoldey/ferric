#!/usr/bin/env python3
"""Sympy proofs: the hand-derived second-order XC terms of the analytic KS
Hessian, `crates/ferric-scf/src/ks_fixed_grid.rs`, fixed and moving grid.

The energy is E = sum_g W_g f(rho(r_g), sigma(r_g)), sigma = |grad rho|^2, on
a model of three atoms carrying s, p and d_xy Gaussians (every AO derivative
up to third order is non-zero), a symmetric 5x5 density matrix and a concrete
GGA-like f(rho, sigma) whose f_rs and f_ss are non-zero. The exact side takes
sympy derivatives, in the nine nuclear coordinates, of the definitions
rho(r_g; R), grad rho(r_g; R), chi(r_g; R) and W(r_g; R), and combines them by
the multivariate chain rule for f(rho, sigma) (checked symbolically, item 0);
each code-form quantity is assembled term by term, as the Rust
function does, from the electron-coordinate AO derivatives at the reference
geometry. Both are evaluated at 50 digits.

Checked (file:line is crates/ferric-scf/src/ks_fixed_grid.rs):
  1. g = 2 sum_mu d_gam chi_mu P_mu                    gradient_densities  :654
  2. rho_x = -2 sum_{mu in B} d_a chi_mu P_mu           rho_derivative      :731
  3. g_{gam,x} = -2 sum_{mu in B}[d_g d_a chi P + d_a chi Q_gam]
                                                        grad_rho_derivative :745
  4. rho_xy and g_{gam,xy} of the module doc (same-centre delta_BC part and
     the cross-centre part with its factor 2)            module doc :16-20
  5. F rho_xy + G.g_xy as add_same_centre + add_cross_centre      :859, :888
  6. dF, dG (weights_response) = d/dt of (w f_r, 2 w f_s g)       :456
  7. skeleton = kernel_part + (5) = d^2E/dR_x dR_y at frozen D    :807, :825
  8. dE/dR_x = F rho_x + G.g_x, F = w f_r, G = 2 w f_s g          :766, :700
  9. first-order Fock V[dF[x], dG[x]] - (T_mu rows and columns)
     = dV_mn/dR_x at frozen D                           first_order_fock :935, :969
 10. V1[D1] = V[dF[rho1, g1], dG[rho1, g1]], rho1 = sum D1 chi chi,
     g1 = 2 sum D1 d chi chi, = d/dt V(D + t D1)        XcResponse::apply :1317
 11. moving grid (points r_g = R_h + u_g, weights W(r; R) translation
     invariant): d^2E/dR dR = T_h G_h T_h^T with G_h the point-fixed
     W d2f + W' df^T + df W'^T + W'' f    skeleton_hessian_moving :1092,
     f_derivative :1077, translation_fill :1295
 12. moving first-order Fock: V^x (x not on h) = point-fixed dV/dR_x incl.
     V[W'_x f_r, 2 W'_x f_s g], V^h = -sum_{B != h} V^B
     first_order_fock_moving :1125, weight_fock :1164, add_ao_derivative_fock :969

Not re-derived here (T008, wiki/sympy_derivations.md section 16):
V_xc = dE/dD (vmat :441; scripts/dft_vxc_matrix_elements_sympy.py), the
per-point GGA f_xc linearization (scripts/dft_fxc_kernel_sympy.py) and the
Becke weight derivatives W' (scripts/dft_becke_weight_grad_sympy.py). The
weight here is a generic translation-invariant smooth function, so (11)-(12)
prove the riding / translation-fill algebra for any such W.

Negative controls: each mutated formula (a factor or sign the Rust code could
plausibly get wrong) must miss. Exit nonzero on any mismatch.

Run: python3 scripts/ks_fixed_grid_hessian_sympy.py   (a few minutes)
"""

import sys

import mpmath as mp
import sympy as sp

mp.mp.dps = 50
TOL = mp.mpf("1e-35")
CONTROL = mp.mpf("1e-6")
FAILS = []


def check(name, err, ok=None):
    ok = err < TOL if ok is None else ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name} (max |diff| {mp.nstr(err, 3)})")
    if not ok:
        FAILS.append(name)


def control(name, err):
    check(f"control: {name} is detected", err, err > CONTROL)


Q = sp.Rational
NAT = 3
AX = range(3)
r = sp.symbols("rx ry rz", real=True)
R = [sp.symbols(f"R{a}x R{a}y R{a}z", real=True) for a in range(NAT)]
RF = [c for a in R for c in a]
N3 = 3 * NAT
R_REF = [
    [Q(0), Q(1, 10), Q(-1, 5)],
    [Q(13, 10), Q(-1, 5), Q(3, 10)],
    [Q(-2, 5), Q(6, 5), Q(1, 2)],
]
REF = {R[a][k]: R_REF[a][k] for a in range(NAT) for k in AX}

# (atom, exponent, polynomial in d = r - R_atom)
AOS = [
    (0, Q(9, 10), lambda d: 1),
    (0, Q(7, 10), lambda d: d[1]),
    (1, Q(13, 10), lambda d: d[0]),
    (1, Q(6, 5), lambda d: d[0] * d[1]),
    (2, Q(11, 10), lambda d: d[2]),
]
NAO = len(AOS)
AO_ATOM = [a for a, _, _ in AOS]


def chi_expr(mu):
    a, alpha, poly = AOS[mu]
    d = [r[k] - R[a][k] for k in AX]
    return poly(d) * sp.exp(-alpha * sum(x * x for x in d))


CHI = [chi_expr(m) for m in range(NAO)]
_D = [
    [Q(7, 10), Q(-1, 5), Q(1, 10), Q(3, 10), Q(-1, 10)],
    [Q(-1, 5), Q(9, 10), Q(1, 5), Q(-1, 10), Q(1, 5)],
    [Q(1, 10), Q(1, 5), Q(6, 5), Q(-3, 10), Q(1, 10)],
    [Q(3, 10), Q(-1, 10), Q(-3, 10), Q(4, 5), Q(1, 5)],
    [Q(-1, 10), Q(1, 5), Q(1, 10), Q(1, 5), Q(1, 1)],
]
D1 = [
    [Q((3 * i + 5 * j) % 7 - 3, 10) if i <= j else None for j in range(NAO)]
    for i in range(NAO)
]
for i in range(NAO):
    for j in range(i):
        D1[i][j] = D1[j][i]

# f(rho, sigma): rho^{4/3} (LDA-like) plus a B88-like enhancement and a weak
# rho^{5/3} sigma^2 term, so f_r, f_s, f_rr, f_rs, f_ss are all non-zero.
RHO, SIG = sp.symbols("rho sigma", positive=True)
F_EXPR = (
    -(RHO ** Q(4, 3))
    - Q(1, 5) * SIG * RHO ** Q(-4, 3) / (1 + Q(3, 10) * SIG * RHO ** Q(-8, 3))
    + Q(1, 50) * SIG**2 * RHO ** Q(-5, 3)
)
FDER = {
    k: sp.lambdify((RHO, SIG), e, "mpmath")
    for k, e in {
        "f": F_EXPR,
        "r": sp.diff(F_EXPR, RHO),
        "s": sp.diff(F_EXPR, SIG),
        "rr": sp.diff(F_EXPR, RHO, 2),
        "rs": sp.diff(F_EXPR, RHO, SIG),
        "ss": sp.diff(F_EXPR, SIG, 2),
    }.items()
}


def densities(dm):
    rho = sum(dm[m][n] * CHI[m] * CHI[n] for m in range(NAO) for n in range(NAO))
    return rho, [sp.diff(rho, r[k]) for k in AX]


def mpval(e, subs):
    return mp.mpf(sp.N(e.xreplace(subs), 60))


def at_ref(e):
    """Value of an expression of the nuclear coordinates at the reference."""
    return mp.mpf(sp.N(e.xreplace(REF), 60))


class Exact:
    """Sympy R-derivatives of rho(r_g; R) and grad rho(r_g; R) (r_g may ride
    an atom), and E_g = W f(rho, sigma) differentiated by the multivariate
    chain rule (itself checked symbolically below)."""

    def __init__(self, rho_g, g_g):
        self.rho = at_ref(rho_g)
        self.rx = [at_ref(sp.diff(rho_g, RF[i])) for i in range(N3)]
        self.rxy = [
            [at_ref(sp.diff(rho_g, RF[i], RF[j])) for j in range(N3)] for i in range(N3)
        ]
        self.g = [at_ref(x) for x in g_g]
        gx = [[sp.diff(g_g[c], RF[i]) for c in AX] for i in range(N3)]
        self.gx = [[at_ref(gx[i][c]) for c in AX] for i in range(N3)]
        self.gxy = [
            [[at_ref(sp.diff(gx[i][c], RF[j])) for j in range(N3)] for i in range(N3)]
            for c in AX
        ]
        self.sigma = sum(x * x for x in self.g)
        self.sx = [2 * sum(self.g[c] * self.gx[i][c] for c in AX) for i in range(N3)]
        self.sxy = [
            [
                2
                * sum(
                    self.gx[i][c] * self.gx[j][c] + self.g[c] * self.gxy[c][i][j]
                    for c in AX
                )
                for j in range(N3)
            ]
            for i in range(N3)
        ]
        self.k = {key: fn(self.rho, self.sigma) for key, fn in FDER.items()}

    def fx(self):
        k = self.k
        return [k["r"] * self.rx[i] + k["s"] * self.sx[i] for i in range(N3)]

    def fxy(self):
        k, rx, sx = self.k, self.rx, self.sx
        return [
            [
                k["rr"] * rx[i] * rx[j]
                + k["rs"] * (rx[i] * sx[j] + sx[i] * rx[j])
                + k["ss"] * sx[i] * sx[j]
                + k["r"] * self.rxy[i][j]
                + k["s"] * self.sxy[i][j]
                for j in range(N3)
            ]
            for i in range(N3)
        ]

    def fr_x(self):
        k = self.k
        return [k["rr"] * self.rx[i] + k["rs"] * self.sx[i] for i in range(N3)]

    def fs_x(self):
        k = self.k
        return [k["rs"] * self.rx[i] + k["ss"] * self.sx[i] for i in range(N3)]


def exact_vx(ex, chi_g, d1_g, W, Wx):
    """d V_mn / dR_x for V_mn = W [f_r chi chi + 2 f_s g.(d chi chi + chi d chi)],
    every R-dependence (W, f_r, f_s, g, chi, d chi) differentiated by sympy."""
    chi = [at_ref(c) for c in chi_g]
    d1 = [[at_ref(c) for c in row] for row in d1_g]
    chix = [[at_ref(sp.diff(c, RF[i])) for i in range(N3)] for c in chi_g]
    d1x = [
        [[at_ref(sp.diff(c, RF[i])) for i in range(N3)] for c in row] for row in d1_g
    ]
    frx, fsx = ex.fr_x(), ex.fs_x()
    fr, fs = ex.k["r"], ex.k["s"]
    out = []
    for x in range(N3):
        v = [[mp.mpf(0)] * NAO for _ in range(NAO)]
        for m in range(NAO):
            for n in range(NAO):
                cc = chi[m] * chi[n]
                ccx = chix[m][x] * chi[n] + chi[m] * chix[n][x]
                gt = sum(ex.g[c] * (d1[m][c] * chi[n] + chi[m] * d1[n][c]) for c in AX)
                gtx = sum(
                    ex.gx[x][c] * (d1[m][c] * chi[n] + chi[m] * d1[n][c])
                    + ex.g[c]
                    * (
                        d1x[m][c][x] * chi[n]
                        + d1[m][c] * chix[n][x]
                        + chix[m][x] * d1[n][c]
                        + chi[m] * d1x[n][c][x]
                    )
                    for c in AX
                )
                inner = fr * cc + 2 * fs * gt
                inner_x = frx[x] * cc + fr * ccx + 2 * (fsx[x] * gt + fs * gtx)
                v[m][n] = Wx[x] * inner + W * inner_x
        out.append(v)
    return out


# The chain rule used by Exact, checked once for a generic f and generic
# rho(a, b), sigma(a, b).
_a, _b = sp.symbols("a b")
_rf, _sf = sp.Function("rr")(_a, _b), sp.Function("ss")(_a, _b)
_f = sp.Function("f")
_lhs = sp.diff(_f(_rf, _sf), _a, _b)
_x, _y = sp.symbols("X Y")
_fx = sp.Function("f")(_x, _y)
_d = {
    "r": sp.diff(_fx, _x),
    "s": sp.diff(_fx, _y),
    "rr": sp.diff(_fx, _x, 2),
    "rs": sp.diff(_fx, _x, _y),
    "ss": sp.diff(_fx, _y, 2),
}
_sub = {_x: _rf, _y: _sf}
_rhs = (
    _d["rr"].subs(_sub) * sp.diff(_rf, _a) * sp.diff(_rf, _b)
    + _d["rs"].subs(_sub)
    * (sp.diff(_rf, _a) * sp.diff(_sf, _b) + sp.diff(_sf, _a) * sp.diff(_rf, _b))
    + _d["ss"].subs(_sub) * sp.diff(_sf, _a) * sp.diff(_sf, _b)
    + _d["r"].subs(_sub) * sp.diff(_rf, _a, _b)
    + _d["s"].subs(_sub) * sp.diff(_sf, _a, _b)
)
_chain_ok = sp.simplify((_lhs - _rhs).doit()) == 0
print(
    f"[{'PASS' if _chain_ok else 'FAIL'}] 0. multivariate chain rule for f(rho, sigma) used by the exact side"
)
if not _chain_ok:
    FAILS.append("chain rule")


# ---------------------------------------------------------------------------
# Code-form building blocks at one point (numbers at the reference geometry)
# ---------------------------------------------------------------------------


class Point:
    """Electron-coordinate AO derivatives and densities at r_g (numbers)."""

    def __init__(self, rg, dm):
        subs = {**{r[k]: rg[k] for k in AX}, **REF}
        self.chi = [mpval(c, subs) for c in CHI]
        self.d1 = [[mpval(sp.diff(c, r[a]), subs) for a in AX] for c in CHI]
        self.d2 = [
            [[mpval(sp.diff(c, r[a], r[b]), subs) for b in AX] for a in AX] for c in CHI
        ]
        self.d3 = [
            [
                [[mpval(sp.diff(c, r[a], r[b], r[g]), subs) for g in AX] for b in AX]
                for a in AX
            ]
            for c in CHI
        ]
        self.dm = dm
        self.P = [sum(dm[m][n] * self.chi[n] for n in range(NAO)) for m in range(NAO)]
        self.Qg = [
            [sum(dm[m][n] * self.d1[n][g] for n in range(NAO)) for g in AX]
            for m in range(NAO)
        ]
        # (1) gradient_densities: g = 2 sum d chi P
        self.g = [2 * sum(self.d1[m][g] * self.P[m] for m in range(NAO)) for g in AX]
        self.rho = sum(self.chi[m] * self.P[m] for m in range(NAO))
        self.sigma = sum(x * x for x in self.g)
        self.k = {key: fn(self.rho, self.sigma) for key, fn in FDER.items()}

    # (2) rho_derivative
    def rho_x(self, two=2):
        out = [mp.mpf(0)] * N3
        for m in range(NAO):
            for a in AX:
                out[3 * AO_ATOM[m] + a] += -two * self.d1[m][a] * self.P[m]
        return out

    # (3) grad_rho_derivative
    def g_x(self, drop_q=False):
        out = [[mp.mpf(0)] * 3 for _ in range(N3)]
        for m in range(NAO):
            for a in AX:
                for g in AX:
                    t = self.d2[m][g][a] * self.P[m]
                    if not drop_q:
                        t += self.d1[m][a] * self.Qg[m][g]
                    out[3 * AO_ATOM[m] + a][g] += -2 * t
        return out

    # (4) rho_xy, g_xy (module doc)
    def rho_xy(self, cross=2):
        h = [[mp.mpf(0)] * N3 for _ in range(N3)]
        for m in range(NAO):
            for a in AX:
                for b in AX:
                    x = 3 * AO_ATOM[m]
                    h[x + a][x + b] += 2 * self.d2[m][a][b] * self.P[m]
                for n in range(NAO):
                    for b in AX:
                        h[3 * AO_ATOM[m] + a][3 * AO_ATOM[n] + b] += (
                            cross * self.dm[m][n] * self.d1[m][a] * self.d1[n][b]
                        )
        return h

    def g_xy(self, gam):
        h = [[mp.mpf(0)] * N3 for _ in range(N3)]
        for m in range(NAO):
            x = 3 * AO_ATOM[m]
            for a in AX:
                for b in AX:
                    h[x + a][x + b] += 2 * (
                        self.d3[m][gam][a][b] * self.P[m]
                        + self.d2[m][a][b] * self.Qg[m][gam]
                    )
                for n in range(NAO):
                    y = 3 * AO_ATOM[n]
                    for b in AX:
                        h[x + a][y + b] += (
                            2
                            * self.dm[m][n]
                            * (
                                self.d2[m][gam][a] * self.d1[n][b]
                                + self.d1[m][a] * self.d2[n][gam][b]
                            )
                        )
        return h

    def FG(self, w, g_factor=2):
        return w * self.k["r"], [g_factor * w * self.k["s"] * self.g[g] for g in AX]

    # (6) weights_response
    def dFdG(self, w, rho1, g1, two_rs=2, two_out=2):
        k = self.k
        gg1 = sum(self.g[c] * g1[c] for c in AX)
        df = w * (k["rr"] * rho1 + two_rs * k["rs"] * gg1)
        coef = two_out * w * (k["rs"] * rho1 + 2 * k["ss"] * gg1)
        dg = [coef * self.g[c] + 2 * w * k["s"] * g1[c] for c in AX]
        return df, dg

    # (5) add_same_centre + add_cross_centre: F rho_xy + G.g_xy
    def second_part(self, F, G, cross_two=2):
        h = [[mp.mpf(0)] * N3 for _ in range(N3)]
        for m in range(NAO):
            x = 3 * AO_ATOM[m]
            X = F * self.P[m] + sum(G[c] * self.Qg[m][c] for c in AX)
            for a in AX:
                for b in AX:
                    t = (
                        self.d2[m][a][b] * X
                        + sum(G[c] * self.d3[m][c][a][b] for c in AX) * self.P[m]
                    )
                    h[x + a][x + b] += 2 * t
        for c_atom in range(NAT):
            on_c = [n for n in range(NAO) if AO_ATOM[n] == c_atom]
            for b in AX:
                pd = [
                    sum(self.dm[m][n] * self.d1[n][b] for n in on_c) for m in range(NAO)
                ]
                pdg = [
                    [sum(self.dm[m][n] * self.d2[n][c][b] for n in on_c) for c in AX]
                    for m in range(NAO)
                ]
                for m in range(NAO):
                    wv = F * pd[m] + sum(G[c] * pdg[m][c] for c in AX)
                    for a in AX:
                        ha = sum(G[c] * self.d2[m][c][a] for c in AX)
                        arr = self.d1[m][a] * wv + ha * pd[m]
                        h[3 * AO_ATOM[m] + a][3 * c_atom + b] += cross_two * arr
        return h

    # (7) batch_skeleton = kernel_part + second_part
    def skeleton(self, w, mut=None):
        mut = mut or {}
        F, G = self.FG(w, mut.get("g_factor", 2))
        rx, gx = self.rho_x(), self.g_x()
        h2 = self.second_part(F, G, mut.get("cross_two", 2))
        h = [[mut.get("vxc", 1) * h2[i][j] for j in range(N3)] for i in range(N3)]
        dfg = [
            self.dFdG(w, rx[y], gx[y], mut.get("two_rs", 2), mut.get("two_out", 2))
            for y in range(N3)
        ]
        for i in range(N3):
            for j in range(N3):
                df, dg = dfg[j]
                h[i][j] += rx[i] * df + sum(gx[i][c] * dg[c] for c in AX)
        return h

    # (8) gradient
    def grad(self, w, g_factor=2):
        F, G = self.FG(w, g_factor)
        rx, gx = self.rho_x(), self.g_x()
        return [F * rx[i] + sum(G[c] * gx[i][c] for c in AX) for i in range(N3)]

    # vmat
    def vmat(self, F, G):
        return [
            [
                F * self.chi[m] * self.chi[n]
                + sum(
                    G[c] * (self.d1[m][c] * self.chi[n] + self.chi[m] * self.d1[n][c])
                    for c in AX
                )
                for n in range(NAO)
            ]
            for m in range(NAO)
        ]

    # (9) first_order_fock: V[dF[x], dG[x]] + the AO-derivative rows/columns
    def fock(self, w, home=None, fill=-1, mut=None):
        mut = mut or {}
        F, G = self.FG(w)
        rx, gx = self.rho_x(), self.g_x()
        out = []
        for x in range(N3):
            df, dg = self.dFdG(w, rx[x], gx[x])
            out.append(self.vmat(df, dg))
        for a in AX:
            T = [
                [
                    F * self.d1[m][a] * self.chi[n]
                    + sum(
                        G[c]
                        * (
                            self.d2[m][c][a] * self.chi[n]
                            + self.d1[m][a] * self.d1[n][c]
                        )
                        for c in AX
                    )
                    for n in range(NAO)
                ]
                for m in range(NAO)
            ]
            for m in range(NAO):
                coef = -mut.get("ao_sign", 1)
                targets = [(AO_ATOM[m], coef)]
                if home is not None:
                    if AO_ATOM[m] == home:
                        continue
                    targets.append((home, coef * fill))
                for atom, cf in targets:
                    v = out[3 * atom + a]
                    for n in range(NAO):
                        v[m][n] += cf * T[m][n]
                        v[n][m] += cf * T[m][n]
        return out


def maxdiff(a, b):
    if isinstance(a, list):
        return max(maxdiff(x, y) for x, y in zip(a, b))
    return abs(a - b)


def zeros2(n, m):
    return [[mp.mpf(0)] * m for _ in range(n)]


def add2(acc, x, scale=1):
    for i in range(len(acc)):
        for j in range(len(acc[i])):
            acc[i][j] += scale * x[i][j]


def add3(acc, x):
    for a, b in zip(acc, x):
        add2(a, b)


def zeros_fock():
    return [zeros2(NAO, NAO) for _ in range(N3)]


rho_sym, g_sym = densities(_D)


def at_point(rv):
    """rho, grad rho, chi, d chi at the (possibly riding) point rv."""
    subs_r = {r[k]: rv[k] for k in AX}
    return (
        rho_sym.xreplace(subs_r),
        [x.xreplace(subs_r) for x in g_sym],
        [c.xreplace(subs_r) for c in CHI],
        [[sp.diff(c, r[a]).xreplace(subs_r) for a in AX] for c in CHI],
    )


# ---------------------------------------------------------------------------
# Fixed grid: two points, fixed in space
# ---------------------------------------------------------------------------
print("fixed grid ...", flush=True)
PTS = [
    ([Q(3, 10), Q(2, 5), Q(1, 10)], Q(7, 10)),
    ([Q(4, 5), Q(-1, 10), Q(1, 5)], Q(11, 10)),
]
MUTANTS = {
    "G without its factor 2 (G = w f_s g)": {"g_factor": 1},
    "cross-centre factor 2 dropped": {"cross_two": 1},
    "2 f_rs g.g1 -> f_rs g.g1 in dF": {"two_rs": 1},
    "outer factor 2 of dG dropped": {"two_out": 1},
    "F rho_xy + G.g_xy dropped": {"vxc": 0},
}
errs = {k: mp.mpf(0) for k in ["g", "rx", "gx", "rxy", "gxy", "second", "dFdG"]}
ctrl = {k: mp.mpf(0) for k in ["rx1", "gxq", "rxy1"]}
H_ex, H_code = zeros2(N3, N3), zeros2(N3, N3)
H_mut = {k: zeros2(N3, N3) for k in MUTANTS}
gr_ex, gr_code, gr_mut = [mp.mpf(0)] * N3, [mp.mpf(0)] * N3, [mp.mpf(0)] * N3
V_ex, V_code, V_mut = zeros_fock(), zeros_fock(), zeros_fock()
pts_obj = []
for rg, w in PTS:
    rho_g, gg, chi_g, d1_g = at_point(rg)
    ex = Exact(rho_g, gg)
    p = Point(rg, _D)
    pts_obj.append((p, w))
    errs["g"] = max(errs["g"], maxdiff(p.g, ex.g))
    errs["rx"] = max(errs["rx"], maxdiff(p.rho_x(), ex.rx))
    ctrl["rx1"] = max(ctrl["rx1"], maxdiff(p.rho_x(two=1), ex.rx))
    errs["gx"] = max(errs["gx"], maxdiff(p.g_x(), ex.gx))
    ctrl["gxq"] = max(ctrl["gxq"], maxdiff(p.g_x(drop_q=True), ex.gx))
    errs["rxy"] = max(errs["rxy"], maxdiff(p.rho_xy(), ex.rxy))
    ctrl["rxy1"] = max(ctrl["rxy1"], maxdiff(p.rho_xy(cross=1), ex.rxy))
    errs["gxy"] = max(errs["gxy"], max(maxdiff(p.g_xy(c), ex.gxy[c]) for c in AX))
    F, G = p.FG(w)
    sec = [
        [F * ex.rxy[i][j] + sum(G[c] * ex.gxy[c][i][j] for c in AX) for j in range(N3)]
        for i in range(N3)
    ]
    errs["second"] = max(errs["second"], maxdiff(p.second_part(F, G), sec))
    # (6) dF, dG vs d/dt (w f_r, 2 w f_s g) along (rho_x, g_x)
    t = sp.Symbol("t")
    for x in (0, 4, 8):
        r1, g1 = ex.rx[x], ex.gx[x]
        rt = p.rho + t * r1
        gt = [p.g[c] + t * g1[c] for c in AX]
        st = sum(v * v for v in gt)
        fr_t = sp.diff(F_EXPR, RHO).xreplace({RHO: rt, SIG: st})
        fs_t = sp.diff(F_EXPR, SIG).xreplace({RHO: rt, SIG: st})
        dF_ex = mp.mpf(sp.N(sp.diff(w * fr_t, t).subs(t, 0), 60))
        dG_ex = [
            mp.mpf(sp.N(sp.diff(2 * w * fs_t * gt[c], t).subs(t, 0), 60)) for c in AX
        ]
        df, dg = p.dFdG(w, r1, g1)
        errs["dFdG"] = max(errs["dFdG"], abs(df - dF_ex), maxdiff(dg, dG_ex))
    # (7), (8)
    add2(H_ex, ex.fxy(), w)
    add2(H_code, p.skeleton(w))
    for name, mut in MUTANTS.items():
        add2(H_mut[name], p.skeleton(w, mut))
    for i, (a, b, c) in enumerate(zip(ex.fx(), p.grad(w), p.grad(w, 1))):
        gr_ex[i] += w * a
        gr_code[i] += b
        gr_mut[i] += c
    # (9)
    add3(V_ex, exact_vx(ex, chi_g, d1_g, w, [0] * N3))
    add3(V_code, p.fock(w))
    add3(V_mut, p.fock(w, mut={"ao_sign": -1}))

check("1. g = 2 sum d_gam chi P (gradient_densities)", errs["g"])
check("2. rho_x = -2 sum_{mu in B} d_a chi P (rho_derivative)", errs["rx"])
control("rho_x without its factor 2", ctrl["rx1"])
check(
    "3. g_x = -2 sum_{mu in B} [d_g d_a chi P + d_a chi Q] (grad_rho_derivative)",
    errs["gx"],
)
control("g_x without the d_a chi Q_gam term", ctrl["gxq"])
check(
    "4. rho_xy (same-centre 2 d_a d_b chi P + cross 2 D d_a chi d_b chi)", errs["rxy"]
)
control("rho_xy cross term without its factor 2", ctrl["rxy1"])
check("4. g_{gam,xy} of the module doc", errs["gxy"])
check("5. add_same_centre + add_cross_centre = F rho_xy + G.g_xy", errs["second"])
check("6. weights_response dF, dG = d/dt (w f_r, 2 w f_s g)", errs["dFdG"])
check("7. skeleton (kernel_part + F rho_xy + G.g_xy) = d2E/dRdR", maxdiff(H_code, H_ex))
for name in MUTANTS:
    control(f"skeleton with {name}", maxdiff(H_mut[name], H_ex))
check("8. gradient F rho_x + G.g_x = dE/dR", maxdiff(gr_code, gr_ex))
control("gradient with G = w f_s g", maxdiff(gr_mut, gr_ex))
check("9. first_order_fock = dV/dR at frozen D", maxdiff(V_code, V_ex))
control(
    "first-order Fock with the AO-derivative rows of the wrong sign",
    maxdiff(V_mut, V_ex),
)

# ---------------------------------------------------------------------------
# (10) XcResponse::apply
# ---------------------------------------------------------------------------
print("response ...", flush=True)
t = sp.Symbol("t")
DT = [[_D[i][j] + t * D1[i][j] for j in range(NAO)] for i in range(NAO)]
rho_t, g_t = densities(DT)
V1_ex, V1_code, V1_mut = zeros2(NAO, NAO), zeros2(NAO, NAO), zeros2(NAO, NAO)
for (rg, w), (p, _) in zip(PTS, pts_obj):
    subs = {**{r[k]: rg[k] for k in AX}, **REF}
    rho1_ex = mp.mpf(sp.N(sp.diff(rho_t.xreplace(subs), t).subs(t, 0), 60))
    g1_ex = [mp.mpf(sp.N(sp.diff(x.xreplace(subs), t).subs(t, 0), 60)) for x in g_t]
    # exact: V is linear in (F, G) through vmat, so dV/dt = vmat(dF/dt, dG/dt)
    # with (dF/dt, dG/dt) from the definition (w f_r, 2 w f_s g) at D + t D1.
    rt = p.rho + t * rho1_ex
    gt = [p.g[c] + t * g1_ex[c] for c in AX]
    st = sum(v * v for v in gt)
    fr_t = sp.diff(F_EXPR, RHO).xreplace({RHO: rt, SIG: st})
    fs_t = sp.diff(F_EXPR, SIG).xreplace({RHO: rt, SIG: st})
    dF = mp.mpf(sp.N(sp.diff(w * fr_t, t).subs(t, 0), 60))
    dG = [mp.mpf(sp.N(sp.diff(2 * w * fs_t * gt[c], t).subs(t, 0), 60)) for c in AX]
    add2(V1_ex, p.vmat(dF, dG))
    pd = [sum(D1[m][n] * p.chi[n] for n in range(NAO)) for m in range(NAO)]
    rho1 = sum(p.chi[m] * pd[m] for m in range(NAO))
    for two, tgt in ((2, V1_code), (1, V1_mut)):
        g1 = [two * sum(p.d1[m][c] * pd[m] for m in range(NAO)) for c in AX]
        df, dg = p.dFdG(w, rho1, g1)
        add2(tgt, p.vmat(df, dg))
check("10. XcResponse::apply V1[D1] = d/dt V(D + t D1)", maxdiff(V1_code, V1_ex))
control(
    "response with g1 = sum D1 d chi chi (factor 2 dropped)", maxdiff(V1_mut, V1_ex)
)

# ---------------------------------------------------------------------------
# Moving grid: points ride home atom 0, weights W(r; R) translation invariant
# ---------------------------------------------------------------------------
print("moving grid ...", flush=True)
HOME = 0
U = [
    ([Q(2, 5), Q(-3, 10), Q(1, 5)], Q(9, 10)),
    ([Q(-1, 5), Q(1, 2), Q(3, 10)], Q(6, 5)),
]


def weight_expr(rv):
    """A smooth, translation-invariant cell weight of home 0."""
    d2 = [sum((rv[k] - R[a][k]) ** 2 for k in AX) for a in range(NAT)]
    cells = [sp.exp(-Q(1, 2) * d2[a]) * (1 + Q(a, 5)) for a in range(NAT)]
    return cells[HOME] / sum(cells)


def tgt(G, fill):
    """translation_fill: T G T^T."""
    T = zeros2(N3, N3)
    for x in range(N3):
        if x // 3 != HOME:
            T[x][x] = mp.mpf(1)
            T[3 * HOME + x % 3][x] = mp.mpf(fill)
    TG = [
        [sum(T[i][k] * G[k][j] for k in range(N3)) for j in range(N3)]
        for i in range(N3)
    ]
    return [
        [sum(TG[i][k] * T[j][k] for k in range(N3)) for j in range(N3)]
        for i in range(N3)
    ]


HKEYS = ("prod", "nofill", "flip", "noW2", "nocross")
FKEYS = ("prod", "nofill", "noWfock")
Hm_ex, Hm = zeros2(N3, N3), {k: zeros2(N3, N3) for k in HKEYS}
Vm_ex, Vm = zeros_fock(), {k: zeros_fock() for k in FKEYS}
for u, s in U:
    rv = [R[HOME][k] + u[k] for k in AX]
    rho_g, gg, chi_g, d1_g = at_point(rv)
    ex = Exact(rho_g, gg)
    Wr = s * weight_expr(rv)
    W, Wx = at_ref(Wr), [at_ref(sp.diff(Wr, RF[i])) for i in range(N3)]
    Wxy = [[at_ref(sp.diff(Wr, RF[i], RF[j])) for j in range(N3)] for i in range(N3)]
    fv, fx, fxy = ex.k["f"], ex.fx(), ex.fxy()
    add2(
        Hm_ex,
        [
            [
                Wxy[i][j] * fv + Wx[i] * fx[j] + fx[i] * Wx[j] + W * fxy[i][j]
                for j in range(N3)
            ]
            for i in range(N3)
        ],
    )
    add3(Vm_ex, exact_vx(ex, chi_g, d1_g, W, Wx))
    # code form: point-fixed quantities at r_g = R_home_ref + u
    rg = [R_REF[HOME][k] + u[k] for k in AX]
    p = Point(rg, _D)
    q = sp.symbols("qx qy qz", real=True)
    Wq = s * weight_expr(q)
    sq = {q[k]: rg[k] for k in AX}
    W0 = at_ref(Wq.xreplace(sq))
    W1 = [at_ref(sp.diff(Wq, RF[i]).xreplace(sq)) for i in range(N3)]
    W2 = [
        [at_ref(sp.diff(Wq, RF[i], RF[j]).xreplace(sq)) for j in range(N3)]
        for i in range(N3)
    ]
    rx, gx = p.rho_x(), p.g_x()
    # f_derivative (unweighted): f_r rho_x + 2 f_s g.g_x
    dfx = [
        p.k["r"] * rx[i] + 2 * p.k["s"] * sum(p.g[c] * gx[i][c] for c in AX)
        for i in range(N3)
    ]
    sk = p.skeleton(W0)
    for key in HKEYS:
        G = [
            [
                sk[i][j]
                + (0 if key == "noW2" else W2[i][j] * p.k["f"])
                + (0 if key == "nocross" else W1[i] * dfx[j] + dfx[i] * W1[j])
                for j in range(N3)
            ]
            for i in range(N3)
        ]
        add2(Hm[key], tgt(G, {"nofill": 0, "flip": 1}.get(key, -1)))
    # first_order_fock_moving
    ao_only = p.fock(W0, home=HOME, fill=-1)
    ao_nofill = p.fock(W0, home=HOME, fill=0)
    for key in FKEYS:
        fill = 0 if key == "nofill" else -1
        ao = ao_nofill if key == "nofill" else ao_only
        out = zeros_fock()
        for x in range(N3):
            df, dg = p.dFdG(W0, rx[x], gx[x])
            vk = p.vmat(df, dg)
            add2(out[x], ao[x])
            add2(
                out[x], vk, -1
            )  # p.fock also adds V[dF, dG] for every x: keep only its AO part
            if x // 3 == HOME:
                continue
            c = vk
            if key != "noWfock":
                wf = p.vmat(
                    W1[x] * p.k["r"], [2 * W1[x] * p.k["s"] * p.g[cc] for cc in AX]
                )
                c = [[c[m][n] + wf[m][n] for n in range(NAO)] for m in range(NAO)]
            add2(out[x], c)
            add2(out[3 * HOME + x % 3], c, fill)
        add3(Vm[key], out)

check(
    "11. moving skeleton T_h G_h T_h^T = d2E/dRdR (points riding home)",
    maxdiff(Hm["prod"], Hm_ex),
)
control("moving skeleton without the translation fill", maxdiff(Hm["nofill"], Hm_ex))
control("moving skeleton with the fill sign flipped", maxdiff(Hm["flip"], Hm_ex))
control("moving skeleton without W'' f", maxdiff(Hm["noW2"], Hm_ex))
control("moving skeleton without W' df^T + df W'^T", maxdiff(Hm["nocross"], Hm_ex))
check(
    "12. first_order_fock_moving = dV/dR (points riding home)",
    maxdiff(Vm["prod"], Vm_ex),
)
control("moving Fock without the translation fill", maxdiff(Vm["nofill"], Vm_ex))
control("moving Fock without V[W' f_r, 2 W' f_s g]", maxdiff(Vm["noWfock"], Vm_ex))

if FAILS:
    print(f"\n{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("\nall identities hold")
