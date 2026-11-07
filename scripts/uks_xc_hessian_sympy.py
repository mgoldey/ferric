#!/usr/bin/env python3
"""Sympy proofs: the SPIN-POLARIZED (UKS) second-order XC terms of the analytic
open-shell KS Hessian, `crates/ferric-scf/src/ks_fixed_grid/polarized.rs`,
fixed and moving grid.

The energy is E = sum_g W_g f(rho_a, rho_b, s_aa, s_ab, s_bb) with per-spin
densities rho_s = sum D^s chi chi, g^s = grad rho_s, s_st = g^s . g^t. The
model is the one of `scripts/ks_fixed_grid_hessian_sympy.py` (three atoms with
s, p and d_xy Gaussians) with two DIFFERENT symmetric density matrices D^a !=
D^b, two different response densities D1^a != D1^b, and a concrete f that is
NOT spin-blind (it has explicit rho_a rho_b and s_ab couplings), so all 3 / 6
/ 6 second derivatives (rho-rho, rho-sigma, sigma-sigma) are non-zero.

The per-spin AO forms rho^s_x, g^s_x, rho^s_xy, g^s_xy are the closed-shell
ones with D -> D^s (proved for any symmetric D by
`ks_fixed_grid_hessian_sympy.py` items 1-5); here they are taken exactly from
sympy, so what this script proves is the SPIN ALGEBRA built on them:

  1. F^s = W f_{rho_s},  G^a = W (2 f_saa g^a + f_sab g^b)  (b: a <-> b),
     dE/dR_x = sum_s F^s rho^s_x + G^s . g^s_x                 (gradient)
  2. the polarized kernel (weights_response_polarized)
       dF^s = W [sum_t f_{rs rt} rho1^t + sum_u f_{rs u} ds_u]
       dG^a = W [2 g^a dfs_aa + g^b dfs_ab + 2 f_saa g1^a + f_sab g1^b]
       ds_aa = 2 g^a.g1^a, ds_ab = g^a.g1^b + g^b.g1^a, ds_bb = 2 g^b.g1^b
     = d/dt (F^s, G^s) along a perturbation (rho1^t, g1^t) of BOTH spins
  3. fixed-grid skeleton sum_s [rho^s_x dF^s[y] + g^s_x.dG^s[y]
                                + F^s rho^s_xy + G^s.g^s_xy] = d2E/dRdR
  4. first-order Fock V^s[dF^s[x], dG^s[x]] - (T^s rows and columns, from
     (F^s, G^s)) = dV^s/dR_x at frozen (D^a, D^b), V^s = dE/dD^s
  5. response V1^s[D1^a, D1^b] = V^s[dF^s, dG^s] (rho1^t = sum D1^t chi chi,
     g1^t = 2 sum D1^t d chi chi) = d/dt V^s(D^a + t D1^a, D^b + t D1^b)
  6. moving grid (points riding home 0, a translation-invariant weight):
     skeleton T_h G_h T_h^T with G_h = W d2f + W' df^T + df W'^T + W'' f,
     df = sum_s f_{rho_s} rho^s_x + Gu^s . g^s_x (Gu^s = G^s / W);
     first-order Fock with the weight term V^s[W'_x f_{rho_s}, W'_x Gu^s]
     and the home rows filled by translation
  7. closed-shell limit: at D^a = D^b = D/2, D1^a = D1^b = D1/2 and
     f_closed(rho, sigma) = f(rho/2, rho/2, sigma/4, sigma/4, sigma/4) the
     polarized skeleton, first-order Fock (spin a) and V1^a equal the
     closed-shell ones (theory.md "Kohn-Sham XC Hessian terms") exactly

Negative controls (each must miss item 3 or 5 with unequal spins):
  * the alpha-beta cross kernel dropped (f_{ra rb}, f_{ra sbb}, f_{rb saa},
    f_{saa sbb} set to 0)
  * the kernel rows of the two spins swapped
  * the "total-density kernel": each spin's perturbation replaced by the
    spin average (rho1^a + rho1^b)/2 before the kernel. ARTIFACT HYPOTHESIS:
    this mutant must PASS item 7 (every closed-shell perturbation is
    spin-symmetric) and FAIL item 5 -- the reason the Rust anchors need
    unequal alpha/beta densities and response densities, not only the
    singlet == RKS check.
  * G^s without the f_sab cross-gradient term (gradient, item 1)
  * the moving-grid weight terms dropped (item 6)

Exit nonzero on any mismatch.  Run: python3 scripts/uks_xc_hessian_sympy.py
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


def control_passes(name, err):
    check(f"control: {name} is NOT detected (artifact hypothesis)", err, err < TOL)


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


def sym(rows):
    m = [[None] * NAO for _ in range(NAO)]
    for i in range(NAO):
        for j in range(i, NAO):
            m[i][j] = m[j][i] = rows(i, j)
    return m


DA = sym(lambda i, j: Q(7, 10) if i == j else Q(((2 * i + 3 * j) % 5) - 2, 10))
DB = sym(lambda i, j: Q(2, 5) if i == j else Q(((3 * i + j) % 4) - 1, 20))
D1A = sym(lambda i, j: Q((3 * i + 5 * j) % 7 - 3, 10))
D1B = sym(lambda i, j: Q((i + 2 * j) % 5 - 2, 7))
DS = [DA, DB]
D1S = [D1A, D1B]

# f(ra, rb, saa, sab, sbb): not spin-blind; every first and second derivative
# is non-zero at the model's densities.
V = sp.symbols("ra rb saa sab sbb", positive=True)
ra, rb, saa, sab, sbb = V
F_EXPR = (
    -(ra ** Q(4, 3) + rb ** Q(4, 3))
    - Q(1, 5) * saa * ra ** Q(-4, 3) / (1 + Q(3, 10) * saa * ra ** Q(-8, 3))
    - Q(1, 5) * sbb * rb ** Q(-4, 3) / (1 + Q(3, 10) * sbb * rb ** Q(-8, 3))
    + Q(1, 10) * ra * rb / (ra + rb + 1) * (1 + Q(1, 7) * sab * (ra + rb) ** Q(-8, 3))
    + Q(1, 40) * saa * sbb * (ra + rb) ** Q(-10, 3)
    + Q(1, 30) * sab**2 * (ra * rb) ** Q(-5, 6)
)
NV = 5
FD1 = [sp.lambdify(V, sp.diff(F_EXPR, V[i]), "mpmath") for i in range(NV)]
FD2 = [
    [sp.lambdify(V, sp.diff(F_EXPR, V[i], V[j]), "mpmath") for j in range(NV)]
    for i in range(NV)
]
F0 = sp.lambdify(V, F_EXPR, "mpmath")
# Index of the variables: 0 ra, 1 rb, 2 saa, 3 sab, 4 sbb.
CROSS_SPIN = {(0, 1), (0, 4), (1, 2), (2, 4)}


def at_ref(e):
    return mp.mpf(sp.N(e.xreplace(REF), 60))


def densities(dm):
    rho = sum(dm[m][n] * CHI[m] * CHI[n] for m in range(NAO) for n in range(NAO))
    return rho, [sp.diff(rho, r[k]) for k in AX]


class Kern:
    """f derivatives at one point; `mut` breaks the second derivatives."""

    def __init__(self, vals, mut=None):
        mut = mut or {}
        self.vals = vals
        self.f = F0(*vals)
        self.d1 = [FD1[i](*vals) for i in range(NV)]
        self.d2 = [[FD2[i][j](*vals) for j in range(NV)] for i in range(NV)]
        if mut.get("no_cross"):
            for i, j in CROSS_SPIN:
                self.d2[i][j] = self.d2[j][i] = mp.mpf(0)


class Spin:
    """Exact R-derivatives of rho_s and g^s at a (possibly riding) point."""

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


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


class Exact:
    """E_g = f(vars(R)) differentiated by the multivariate chain rule."""

    def __init__(self, spins):
        a, b = spins
        self.s = spins
        self.vals = [a.rho, b.rho, dot(a.g, a.g), dot(a.g, b.g), dot(b.g, b.g)]
        self.k = Kern(self.vals)
        pairs = [(0, 0), (0, 1), (1, 1)]
        vx = [a.rx, b.rx]
        vxy = [a.rxy, b.rxy]
        for p, q in pairs:
            sp_, sq = spins[p], spins[q]
            vx.append([dot(sp_.gx[i], sq.g) + dot(sp_.g, sq.gx[i]) for i in range(N3)])
            vxy.append(
                [
                    [
                        sum(
                            sp_.gxy[c][i][j] * sq.g[c]
                            + sp_.gx[i][c] * sq.gx[j][c]
                            + sp_.gx[j][c] * sq.gx[i][c]
                            + sp_.g[c] * sq.gxy[c][i][j]
                            for c in AX
                        )
                        for j in range(N3)
                    ]
                    for i in range(N3)
                ]
            )
        self.vx, self.vxy = vx, vxy

    def fx(self):
        return [sum(self.k.d1[v] * self.vx[v][i] for v in range(NV)) for i in range(N3)]

    def fxy(self):
        k = self.k
        return [
            [
                sum(
                    k.d2[u][v] * self.vx[u][i] * self.vx[v][j]
                    for u in range(NV)
                    for v in range(NV)
                )
                + sum(k.d1[v] * self.vxy[v][i][j] for v in range(NV))
                for j in range(N3)
            ]
            for i in range(N3)
        ]

    def dvar(self, v, i):
        """d f_v / dR_i (f_v the first derivative in variable v)."""
        return sum(self.k.d2[v][u] * self.vx[u][i] for u in range(NV))


# ---------------------------------------------------------------------------
# Code form (what polarized.rs computes)
# ---------------------------------------------------------------------------
SIG_SELF = [2, 4]  # saa, sbb


def gu(k, g, s, cross_g=1):
    """Gu^s = 2 f_sss g^s + f_sab g^t (unweighted G^s)."""
    t = 1 - s
    return [2 * k.d1[SIG_SELF[s]] * g[s][c] + cross_g * k.d1[3] * g[t][c] for c in AX]


def FG(k, g, w, cross_g=1):
    return [(w * k.d1[s], [w * x for x in gu(k, g, s, cross_g)]) for s in (0, 1)]


def dFdG(k, g, w, rho1, g1, mut=None):
    """weights_response_polarized: (dF^s, dG^s) for s = a, b."""
    mut = mut or {}
    if mut.get("total_kernel"):
        rho1 = [(rho1[0] + rho1[1]) / 2] * 2
        g1 = [[(g1[0][c] + g1[1][c]) / 2 for c in AX]] * 2
    ds = [
        2 * dot(g[0], g1[0]),
        dot(g[0], g1[1]) + dot(g[1], g1[0]),
        2 * dot(g[1], g1[1]),
    ]
    pert = [rho1[0], rho1[1], ds[0], ds[1], ds[2]]

    def dfv(v):
        return sum(k.d2[v][u] * pert[u] for u in range(NV))

    out = []
    for s in (0, 1):
        row = 1 - s if mut.get("swap_rows") else s
        t = 1 - s
        df = w * dfv(row)
        dss, dab = dfv(SIG_SELF[s]), dfv(3)
        dg = [
            w
            * (
                2 * g[s][c] * dss
                + g[t][c] * dab
                + 2 * k.d1[SIG_SELF[s]] * g1[s][c]
                + k.d1[3] * g1[t][c]
            )
            for c in AX
        ]
        out.append((df, dg))
    return out


def skeleton_fixed(spins, k, w, mut=None):
    g = [sp_.g for sp_ in spins]
    fg = FG(k, g, w)
    h = [[mp.mpf(0)] * N3 for _ in range(N3)]
    resp = [
        dFdG(k, g, w, [s_.rx[y] for s_ in spins], [s_.gx[y] for s_ in spins], mut)
        for y in range(N3)
    ]
    for s in (0, 1):
        F, G = fg[s]
        sx = spins[s]
        for i in range(N3):
            for j in range(N3):
                df, dg = resp[j][s]
                h[i][j] += (
                    sx.rx[i] * df
                    + dot(sx.gx[i], dg)
                    + F * sx.rxy[i][j]
                    + sum(G[c] * sx.gxy[c][i][j] for c in AX)
                )
    return h


class AoPoint:
    """Electron-coordinate AO derivatives at r_g (for vmat and T rows)."""

    def __init__(self, rg):
        subs = {**{r[k]: rg[k] for k in AX}, **REF}
        self.chi = [mp.mpf(sp.N(c.xreplace(subs), 60)) for c in CHI]
        self.d1 = [
            [mp.mpf(sp.N(sp.diff(c, r[a]).xreplace(subs), 60)) for a in AX] for c in CHI
        ]
        self.d2 = [
            [
                [mp.mpf(sp.N(sp.diff(c, r[a], r[b]).xreplace(subs), 60)) for b in AX]
                for a in AX
            ]
            for c in CHI
        ]

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

    def ao_rows(self, F, G, out, home=None, fill=-1):
        """add_ao_derivative_fock: -(T rows and columns), T from (F, G)."""
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
                targets = [(AO_ATOM[m], -1)]
                if home is not None:
                    if AO_ATOM[m] == home:
                        continue
                    targets.append((home, -fill))
                for atom, cf in targets:
                    v = out[3 * atom + a]
                    for n in range(NAO):
                        v[m][n] += cf * T[m][n]
                        v[n][m] += cf * T[m][n]


def zeros2(n, m):
    return [[mp.mpf(0)] * m for _ in range(n)]


def zeros_fock():
    return [zeros2(NAO, NAO) for _ in range(N3)]


def add2(acc, x, scale=1):
    for i in range(len(acc)):
        for j in range(len(acc[i])):
            acc[i][j] += scale * x[i][j]


def add3(acc, x):
    for a, b in zip(acc, x):
        add2(a, b)


def maxdiff(a, b):
    if isinstance(a, list):
        return max(maxdiff(x, y) for x, y in zip(a, b))
    return abs(a - b)


def exact_vx(ex, s, chi_g, d1_g, W, Wx):
    """dV^s_mn/dR_x for V^s = W [f_rs chi chi + Gu^s.(d chi chi + chi d chi)]."""
    chi = [at_ref(c) for c in chi_g]
    d1 = [[at_ref(c) for c in row] for row in d1_g]
    chix = [[at_ref(sp.diff(c, RF[i])) for i in range(N3)] for c in chi_g]
    d1x = [
        [[at_ref(sp.diff(c, RF[i])) for i in range(N3)] for c in row] for row in d1_g
    ]
    k = ex.k
    t = 1 - s
    gs, gt = ex.s[s], ex.s[t]
    vs = SIG_SELF[s]
    out = []
    for x in range(N3):
        fr, frx = k.d1[s], ex.dvar(s, x)
        gu_ = [2 * k.d1[vs] * gs.g[c] + k.d1[3] * gt.g[c] for c in AX]
        gux = [
            2 * ex.dvar(vs, x) * gs.g[c]
            + 2 * k.d1[vs] * gs.gx[x][c]
            + ex.dvar(3, x) * gt.g[c]
            + k.d1[3] * gt.gx[x][c]
            for c in AX
        ]
        v = zeros2(NAO, NAO)
        for m in range(NAO):
            for n in range(NAO):
                cc = chi[m] * chi[n]
                ccx = chix[m][x] * chi[n] + chi[m] * chix[n][x]
                pr = [d1[m][c] * chi[n] + chi[m] * d1[n][c] for c in AX]
                prx = [
                    d1x[m][c][x] * chi[n]
                    + d1[m][c] * chix[n][x]
                    + chix[m][x] * d1[n][c]
                    + chi[m] * d1x[n][c][x]
                    for c in AX
                ]
                inner = fr * cc + dot(gu_, pr)
                inner_x = frx * cc + fr * ccx + dot(gux, pr) + dot(gu_, prx)
                v[m][n] = Wx[x] * inner + W * inner_x
        out.append(v)
    return out


def code_fock(p, spins, k, w, s):
    """first_order_fock (spin s) at a fixed point."""
    g = [sp_.g for sp_ in spins]
    F, G = FG(k, g, w)[s]
    out = zeros_fock()
    for x in range(N3):
        df, dg = dFdG(k, g, w, [s_.rx[x] for s_ in spins], [s_.gx[x] for s_ in spins])[
            s
        ]
        add2(out[x], p.vmat(df, dg))
    p.ao_rows(F, G, out)
    return out


SYM = [densities(dm) for dm in DS]


def at_point(rv):
    subs_r = {r[k]: rv[k] for k in AX}
    dens = [(rho.xreplace(subs_r), [x.xreplace(subs_r) for x in g]) for rho, g in SYM]
    chi = [c.xreplace(subs_r) for c in CHI]
    d1 = [[sp.diff(c, r[a]).xreplace(subs_r) for a in AX] for c in CHI]
    return dens, chi, d1


# ---------------------------------------------------------------------------
# Fixed grid
# ---------------------------------------------------------------------------
print("fixed grid (unequal spins) ...", flush=True)
PTS = [
    ([Q(3, 10), Q(2, 5), Q(1, 10)], Q(7, 10)),
    ([Q(4, 5), Q(-1, 10), Q(1, 5)], Q(11, 10)),
]
SK_MUT = {
    "the alpha-beta cross kernel dropped": {"no_cross": True},
    "the kernel rows of the two spins swapped": {"swap_rows": True},
    "the total-density (spin-averaged perturbation) kernel": {"total_kernel": True},
}
H_ex, H_code = zeros2(N3, N3), zeros2(N3, N3)
H_mut = {n: zeros2(N3, N3) for n in SK_MUT}
gr_ex, gr_code, gr_mut = [mp.mpf(0)] * N3, [mp.mpf(0)] * N3, [mp.mpf(0)] * N3
V_ex = [zeros_fock(), zeros_fock()]
V_code = [zeros_fock(), zeros_fock()]
err_dfdg = mp.mpf(0)
point_data = []
t = sp.Symbol("t")
for rg, w in PTS:
    dens, chi_g, d1_g = at_point(rg)
    spins = [Spin(rho, g) for rho, g in dens]
    ex = Exact(spins)
    k = ex.k
    p = AoPoint(rg)
    point_data.append((rg, w, spins, ex, p))
    g = [s_.g for s_ in spins]
    # (1) gradient
    fg = FG(k, g, w)
    fg_mut = FG(k, g, w, cross_g=0)
    for i, fx in enumerate(ex.fx()):
        gr_ex[i] += w * fx
        gr_code[i] += sum(
            fg[s][0] * spins[s].rx[i] + dot(fg[s][1], spins[s].gx[i]) for s in (0, 1)
        )
        gr_mut[i] += sum(
            fg_mut[s][0] * spins[s].rx[i] + dot(fg_mut[s][1], spins[s].gx[i])
            for s in (0, 1)
        )
    # (2) kernel vs d/dt (F^s, G^s) along an unequal two-spin perturbation
    for x, y in ((0, 4), (8, 2)):
        rho1 = [spins[0].rx[x], spins[1].rx[y]]
        g1 = [spins[0].gx[x], spins[1].gx[y]]
        vt = [g[0], g[1]]
        gt = [[vt[s][c] + t * g1[s][c] for c in AX] for s in (0, 1)]
        vals_t = [
            spins[0].rho + t * rho1[0],
            spins[1].rho + t * rho1[1],
            dot(gt[0], gt[0]),
            dot(gt[0], gt[1]),
            dot(gt[1], gt[1]),
        ]
        sub = dict(zip(V, vals_t))
        fv = [sp.diff(F_EXPR, V[i]).xreplace(sub) for i in range(NV)]
        code = dFdG(k, g, w, rho1, g1)
        for s in (0, 1):
            ts = 1 - s
            F_t = w * fv[s]
            G_t = [w * (2 * fv[SIG_SELF[s]] * gt[s][c] + fv[3] * gt[ts][c]) for c in AX]
            dF = mp.mpf(sp.N(sp.diff(F_t, t).subs(t, 0), 60))
            dG = [mp.mpf(sp.N(sp.diff(G_t[c], t).subs(t, 0), 60)) for c in AX]
            df, dg = code[s]
            err_dfdg = max(err_dfdg, abs(df - dF), maxdiff(dg, dG))
    # (3) skeleton
    add2(H_ex, ex.fxy(), w)
    add2(H_code, skeleton_fixed(spins, k, w))
    for name, mut in SK_MUT.items():
        km = Kern(ex.vals, mut)
        add2(H_mut[name], skeleton_fixed(spins, km, w, mut))
    # (4) first-order Fock per spin
    for s in (0, 1):
        add3(V_ex[s], exact_vx(ex, s, chi_g, d1_g, w, [0] * N3))
        add3(V_code[s], code_fock(p, spins, k, w, s))

check("1. dE/dR_x = sum_s F^s rho^s_x + G^s.g^s_x", maxdiff(gr_code, gr_ex))
control("G^s without the f_sab cross-gradient term", maxdiff(gr_mut, gr_ex))
check(
    "2. polarized dF^s, dG^s = d/dt (F^s, G^s), unequal two-spin perturbation", err_dfdg
)
check("3. fixed-grid polarized skeleton = d2E/dRdR", maxdiff(H_code, H_ex))
for name in SK_MUT:
    control(f"skeleton with {name}", maxdiff(H_mut[name], H_ex))
check("4. first-order Fock spin a = dV^a/dR at frozen D", maxdiff(V_code[0], V_ex[0]))
check("4. first-order Fock spin b = dV^b/dR at frozen D", maxdiff(V_code[1], V_ex[1]))

# ---------------------------------------------------------------------------
# (5) response V1^s[D1^a, D1^b]
# ---------------------------------------------------------------------------
print("response ...", flush=True)
DT = [
    [[DS[s][i][j] + t * D1S[s][i][j] for j in range(NAO)] for i in range(NAO)]
    for s in (0, 1)
]
SYM_T = [densities(dm) for dm in DT]


def response_code(p, k, g, w, d1s, mut=None):
    rho1, g1 = [], []
    for s in (0, 1):
        pd = [sum(d1s[s][m][n] * p.chi[n] for n in range(NAO)) for m in range(NAO)]
        rho1.append(sum(p.chi[m] * pd[m] for m in range(NAO)))
        g1.append([2 * sum(p.d1[m][c] * pd[m] for m in range(NAO)) for c in AX])
    return [p.vmat(df, dg) for df, dg in dFdG(k, g, w, rho1, g1, mut)]


V1_ex = [zeros2(NAO, NAO), zeros2(NAO, NAO)]
V1_code = [zeros2(NAO, NAO), zeros2(NAO, NAO)]
V1_tot = [zeros2(NAO, NAO), zeros2(NAO, NAO)]
for rg, w, spins, ex, p in point_data:
    subs = {**{r[kk]: rg[kk] for kk in AX}, **REF}
    rt = [rho.xreplace(subs) for rho, _ in SYM_T]
    gt = [[x.xreplace(subs) for x in gg] for _, gg in SYM_T]
    vals_t = [rt[0], rt[1], dot(gt[0], gt[0]), dot(gt[0], gt[1]), dot(gt[1], gt[1])]
    sub = dict(zip(V, vals_t))
    fv = [sp.diff(F_EXPR, V[i]).xreplace(sub) for i in range(NV)]
    for s in (0, 1):
        ts = 1 - s
        F_t = w * fv[s]
        G_t = [w * (2 * fv[SIG_SELF[s]] * gt[s][c] + fv[3] * gt[ts][c]) for c in AX]
        dF = mp.mpf(sp.N(sp.diff(F_t, t).subs(t, 0), 60))
        dG = [mp.mpf(sp.N(sp.diff(G_t[c], t).subs(t, 0), 60)) for c in AX]
        add2(V1_ex[s], p.vmat(dF, dG))
    g = [s_.g for s_ in spins]
    for s, v in enumerate(response_code(p, ex.k, g, w, D1S)):
        add2(V1_code[s], v)
    for s, v in enumerate(response_code(p, ex.k, g, w, D1S, {"total_kernel": True})):
        add2(V1_tot[s], v)
check("5. V1^s[D1^a, D1^b] = d/dt V^s(D + t D1), spin a", maxdiff(V1_code[0], V1_ex[0]))
check("5. V1^s[D1^a, D1^b] = d/dt V^s(D + t D1), spin b", maxdiff(V1_code[1], V1_ex[1]))
control("response with the total-density kernel", maxdiff(V1_tot, V1_ex))

# ---------------------------------------------------------------------------
# (6) moving grid
# ---------------------------------------------------------------------------
print("moving grid ...", flush=True)
HOME = 0
U = [
    ([Q(2, 5), Q(-3, 10), Q(1, 5)], Q(9, 10)),
    ([Q(-1, 5), Q(1, 2), Q(3, 10)], Q(6, 5)),
]


def weight_expr(rv):
    d2 = [sum((rv[kk] - R[a][kk]) ** 2 for kk in AX) for a in range(NAT)]
    cells = [sp.exp(-Q(1, 2) * d2[a]) * (1 + Q(a, 5)) for a in range(NAT)]
    return cells[HOME] / sum(cells)


def tgt(G, fill):
    T = zeros2(N3, N3)
    for x in range(N3):
        if x // 3 != HOME:
            T[x][x] = mp.mpf(1)
            T[3 * HOME + x % 3][x] = mp.mpf(fill)
    TG = [
        [sum(T[i][kk] * G[kk][j] for kk in range(N3)) for j in range(N3)]
        for i in range(N3)
    ]
    return [
        [sum(TG[i][kk] * T[j][kk] for kk in range(N3)) for j in range(N3)]
        for i in range(N3)
    ]


Hm_ex = zeros2(N3, N3)
Hm = {kk: zeros2(N3, N3) for kk in ("prod", "noW")}
Vm_ex = [zeros_fock(), zeros_fock()]
Vm = {kk: [zeros_fock(), zeros_fock()] for kk in ("prod", "noWfock")}
qv = sp.symbols("qx qy qz", real=True)
for u, sc in U:
    rv = [R[HOME][kk] + u[kk] for kk in AX]
    dens, chi_g, d1_g = at_point(rv)
    ex = Exact([Spin(rho, g) for rho, g in dens])
    Wr = sc * weight_expr(rv)
    W, Wx = at_ref(Wr), [at_ref(sp.diff(Wr, RF[i])) for i in range(N3)]
    Wxy = [[at_ref(sp.diff(Wr, RF[i], RF[j])) for j in range(N3)] for i in range(N3)]
    fv, fx, fxy = ex.k.f, ex.fx(), ex.fxy()
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
    for s in (0, 1):
        add3(Vm_ex[s], exact_vx(ex, s, chi_g, d1_g, W, Wx))
    # code form at the point fixed at r_g = R_home_ref + u
    rg = [R_REF[HOME][kk] + u[kk] for kk in AX]
    dens0, _, _ = at_point(rg)
    spins = [Spin(rho, g) for rho, g in dens0]  # point-fixed derivatives
    k = Exact(spins).k
    p = AoPoint(rg)
    Wq = sc * weight_expr(qv)
    sq = {qv[kk]: rg[kk] for kk in AX}
    W0 = at_ref(Wq.xreplace(sq))
    W1 = [at_ref(sp.diff(Wq, RF[i]).xreplace(sq)) for i in range(N3)]
    W2 = [
        [at_ref(sp.diff(Wq, RF[i], RF[j]).xreplace(sq)) for j in range(N3)]
        for i in range(N3)
    ]
    g = [s_.g for s_ in spins]
    gus = [gu(k, g, s) for s in (0, 1)]
    dfx = [
        sum(k.d1[s] * spins[s].rx[i] + dot(gus[s], spins[s].gx[i]) for s in (0, 1))
        for i in range(N3)
    ]
    sk = skeleton_fixed(spins, k, W0)
    for key in ("prod", "noW"):
        on = 0 if key == "noW" else 1
        G = [
            [
                sk[i][j] + on * (W2[i][j] * k.f + W1[i] * dfx[j] + dfx[i] * W1[j])
                for j in range(N3)
            ]
            for i in range(N3)
        ]
        add2(Hm[key], tgt(G, -1))
    fg = FG(k, g, W0)
    for key in ("prod", "noWfock"):
        for s in (0, 1):
            out = zeros_fock()
            p.ao_rows(fg[s][0], fg[s][1], out, home=HOME, fill=-1)
            for x in range(N3):
                if x // 3 == HOME:
                    continue
                df, dg = dFdG(
                    k, g, W0, [s_.rx[x] for s_ in spins], [s_.gx[x] for s_ in spins]
                )[s]
                c = p.vmat(df, dg)
                if key == "prod":
                    wf = p.vmat(W1[x] * k.d1[s], [W1[x] * v for v in gus[s]])
                    add2(c, wf)
                add2(out[x], c)
                add2(out[3 * HOME + x % 3], c, -1)
            add3(Vm[key][s], out)
check(
    "6. moving polarized skeleton T_h G_h T_h^T = d2E/dRdR", maxdiff(Hm["prod"], Hm_ex)
)
control("moving skeleton without the weight terms", maxdiff(Hm["noW"], Hm_ex))
for s, lab in ((0, "a"), (1, "b")):
    check(
        f"6. moving polarized first-order Fock spin {lab} = dV/dR",
        maxdiff(Vm["prod"][s], Vm_ex[s]),
    )
control("moving Fock without V^s[W' f_rs, W' Gu^s]", maxdiff(Vm["noWfock"], Vm_ex))

# ---------------------------------------------------------------------------
# (7) closed-shell limit
# ---------------------------------------------------------------------------
print("closed-shell limit ...", flush=True)
RHO, SIG = sp.symbols("rho sigma", positive=True)
FC = F_EXPR.xreplace(
    {ra: RHO / 2, rb: RHO / 2, saa: SIG / 4, sab: SIG / 4, sbb: SIG / 4}
)
FCD = {
    key: sp.lambdify((RHO, SIG), e, "mpmath")
    for key, e in {
        "r": sp.diff(FC, RHO),
        "s": sp.diff(FC, SIG),
        "rr": sp.diff(FC, RHO, 2),
        "rs": sp.diff(FC, RHO, SIG),
        "ss": sp.diff(FC, SIG, 2),
    }.items()
}
DH = [[(DA[i][j] + DB[i][j]) / 2 for j in range(NAO)] for i in range(NAO)]  # D/2
D1H = [[(D1A[i][j] + D1B[i][j]) / 2 for j in range(NAO)] for i in range(NAO)]  # D1/2
half_sym = densities(DH)
err7 = {"skel": mp.mpf(0), "fock": mp.mpf(0), "v1": mp.mpf(0), "v1tot": mp.mpf(0)}
for rg, w in PTS:
    subs_r = {r[kk]: rg[kk] for kk in AX}
    sh = Spin(half_sym[0].xreplace(subs_r), [x.xreplace(subs_r) for x in half_sym[1]])
    spins = [sh, sh]
    k = Exact(spins).k
    p = AoPoint(rg)
    # closed shell at the total density D = 2 (D/2): rho, g and derivatives double
    rho, gg = 2 * sh.rho, [2 * x for x in sh.g]
    rx = [2 * x for x in sh.rx]
    gx = [[2 * x for x in row] for row in sh.gx]
    rxy = [[2 * x for x in row] for row in sh.rxy]
    gxy = [[[2 * x for x in row] for row in plane] for plane in sh.gxy]
    sig = dot(gg, gg)
    kc = {key: fn(rho, sig) for key, fn in FCD.items()}
    Fc, Gc = w * kc["r"], [2 * w * kc["s"] * x for x in gg]

    def dfdg_c(rho1, g1):
        gg1 = dot(gg, g1)
        df = w * (kc["rr"] * rho1 + 2 * kc["rs"] * gg1)
        coef = 2 * w * (kc["rs"] * rho1 + 2 * kc["ss"] * gg1)
        return df, [coef * gg[c] + 2 * w * kc["s"] * g1[c] for c in AX]

    hc = [
        [
            rx[i] * dfdg_c(rx[j], gx[j])[0]
            + dot(gx[i], dfdg_c(rx[j], gx[j])[1])
            + Fc * rxy[i][j]
            + sum(Gc[c] * gxy[c][i][j] for c in AX)
            for j in range(N3)
        ]
        for i in range(N3)
    ]
    err7["skel"] = max(err7["skel"], maxdiff(skeleton_fixed(spins, k, w), hc))
    fock_c = zeros_fock()
    for x in range(N3):
        add2(fock_c[x], p.vmat(*dfdg_c(rx[x], gx[x])))
    p.ao_rows(Fc, Gc, fock_c)
    err7["fock"] = max(err7["fock"], maxdiff(code_fock(p, spins, k, w, 0), fock_c))
    # V1 for a spin-symmetric response D1^a = D1^b = D1/2 (closed V1[D1])
    pd = [sum(D1H[m][n] * p.chi[n] for n in range(NAO)) for m in range(NAO)]
    rho1 = 2 * sum(p.chi[m] * pd[m] for m in range(NAO))
    g1 = [2 * 2 * sum(p.d1[m][c] * pd[m] for m in range(NAO)) for c in AX]
    v1c = p.vmat(*dfdg_c(rho1, g1))
    g = [sh.g, sh.g]
    err7["v1"] = max(err7["v1"], maxdiff(response_code(p, k, g, w, [D1H, D1H])[0], v1c))
    err7["v1tot"] = max(
        err7["v1tot"],
        maxdiff(
            response_code(p, Kern(k.vals), g, w, [D1H, D1H], {"total_kernel": True})[0],
            v1c,
        ),
    )
check("7. closed-shell limit: polarized skeleton == closed skeleton", err7["skel"])
check("7. closed-shell limit: first-order Fock (spin a) == closed", err7["fock"])
check("7. closed-shell limit: V1^a[D1/2, D1/2] == closed V1[D1]", err7["v1"])
control_passes("the total-density kernel in the closed-shell limit", err7["v1tot"])

if FAILS:
    print(f"\n{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("\nall identities hold")
