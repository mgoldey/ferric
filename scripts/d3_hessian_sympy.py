#!/usr/bin/env python3
"""Sympy proof of the two-body D3(BJ) nuclear Hessian, CN chain rule to second order.

The energy is E = sum_P -C6_P(CN_i, CN_j) G_P(r_P) with
G = s6/(r^6+f^6) + 3 q s8/(r^8+f^8) and CN_a = sum_b c_ab(r_ab). This script
checks the Hessian ferric implements in crates/ferric-d3/src/hessian.rs
(theory: crates/ferric-d3/theory.md, "Hessian"):

  1. counting function   c'' = -(2/r) c' + a (1-2c) c',  a = -k rc/r^2   (counting)
  2. Gaussian weights    d2W/dCN2 = (alpha^2 - 2 wf) W, alpha = -2 wf (CN - CNref)
                                                          (c6_second_derivatives)
  3. quotient rule       C6 = N/D for ANY N, D (abstract functions):
                         C6_aa = (N_aa - 2 C6_a D_a - C6 D_aa)/D
                         C6_ab = (N_ab - C6_a D_b - C6_b D_a - C6 D_ab)/D
                                                          (c6_second_derivatives)
  4. radial factor       G', G''                          (radial_g)
  5. pair block          d2 phi(|R_i - R_j|)/dR dR = phi'' n n^T + (phi'/r)(1 - n n^T)
                         on (i,i), (j,j), minus that on (i,j), (j,i)   (add_pair_block)
  6. assembled Hessian (direct + mixed + CN-CN + e_a d2CN_a, d3bj_hessian) for a
     4-atom, two-element model with 3 and 2 reference CNs, the code's KCN = 16
     and WF = 4, equals
       (a) a 60-digit central finite difference of the code-transcribed GRADIENT
           (the transcription scripts/d3_cn_chain_rule_gradient_sympy.py proves
           equal to sympy's exact dE/dR), and
       (b) a 60-digit second difference of the ENERGY (no gradient involved).

Anchors and controls:
  - trivial limit: one reference per element => every CN term vanishes and the
    Hessian is the sum of direct pair blocks (to 60-digit rounding);
  - symmetry and translational invariance;
  - each CN part is a non-negligible fraction of the Hessian in the model;
  - negative controls (each must FAIL claim 6): the mixed term dropped, the
    CN-CN term dropped, the e_a d2CN_a term dropped, the mixed term without its
    transpose, the (1-2c) factor of c'' dropped.

Mutations (each turns the script red; recorded in
crates/ferric-python/tests/test_derivation_scripts.py):
  - the mixed term without its transpose (`h = madd(h, mixed_tt)` -> `h = h`),
    marker "[FAIL] 6a.";
  - 72 -> 36 in G'' of the transcribed Hessian, marker "[FAIL] 6a.";
  - the off-diagonal CN-CN element dropped (`g * cij` -> `0 * cij`), marker
    "[FAIL] 6a.";
  - the (1-2c) factor dropped from c'', marker "[FAIL] 1. c''".

Exit status is nonzero on any mismatch.

Run: python3 scripts/d3_hessian_sympy.py
"""

import sys

import mpmath as mp
import sympy as sp

FAILS = []
mp.mp.dps = 60


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def is_zero(expr):
    return sp.simplify(sp.together(expr)) == 0


def num_zero(expr, n=3):
    """`expr` vanishes at `n` random rational points to 50 digits (a nonzero
    analytic expression vanishes only on a set of measure zero)."""
    import random

    rng = random.Random(7)
    free = sorted(expr.free_symbols, key=str)
    for _ in range(n):
        pt = {x: sp.Rational(rng.randint(-30, 30), 10) for x in free}
        if abs(sp.N(expr.subs(pt), 60)) > sp.Float("1e-45"):
            return False
    return True


# ---------------------------------------------------------------------------
# 1-5. Symbolic building blocks
# ---------------------------------------------------------------------------
k, rc, R, wf = sp.symbols("k rc R wf", positive=True)
cfun = 1 / (1 + sp.exp(-k * (rc / R - 1)))
a_ = -k * rc / R**2
c1 = a_ * cfun * (1 - cfun)
code_c2 = -2 / R * c1 + a_ * (1 - 2 * cfun) * c1  # counting, c2
check("1. c' = -k rc/r^2 c(1-c)", is_zero(sp.diff(cfun, R) - c1))
check("1. c'' = -(2/r) c' + a (1-2c) c'", is_zero(sp.diff(cfun, R, 2) - code_c2))
check(
    "1. control: c'' without (1-2c) fails",
    not is_zero(sp.diff(cfun, R, 2) - (-2 / R * c1 + a_ * c1)),
)

CN, cnr = sp.symbols("CN cnr", real=True)
W = sp.exp(-wf * (CN - cnr) ** 2)
alpha = -2 * wf * (CN - cnr)
check("2. dW/dCN = alpha W", is_zero(sp.diff(W, CN) - alpha * W))
check(
    "2. d2W/dCN2 = (alpha^2 - 2 wf) W",
    is_zero(sp.diff(W, CN, 2) - (alpha**2 - 2 * wf) * W),
)

# 3. Quotient rule for abstract N(a, b), D(a, b): a proof for any interpolation.
ca, cb = sp.symbols("ca cb", real=True)
Nf = sp.Function("N")(ca, cb)
Df = sp.Function("D")(ca, cb)
C6f = Nf / Df
C6a = (sp.diff(Nf, ca) - C6f * sp.diff(Df, ca)) / Df
C6b = (sp.diff(Nf, cb) - C6f * sp.diff(Df, cb)) / Df
C6aa = (sp.diff(Nf, ca, 2) - 2 * C6a * sp.diff(Df, ca) - C6f * sp.diff(Df, ca, 2)) / Df
C6ab = (
    sp.diff(Nf, ca, cb)
    - C6a * sp.diff(Df, cb)
    - C6b * sp.diff(Df, ca)
    - C6f * sp.diff(Df, ca, cb)
) / Df
check("3. C6_a quotient form", is_zero(sp.diff(C6f, ca) - C6a))
check("3. C6_aa quotient form", is_zero(sp.diff(C6f, ca, 2) - C6aa))
check("3. C6_ab quotient form", is_zero(sp.diff(C6f, ca, cb) - C6ab))
check(
    "3. control: C6_ab without -C6 D_ab fails",
    not is_zero(sp.diff(C6f, ca, cb) - (C6ab + C6f * sp.diff(Df, ca, cb) / Df)),
)

s6, s8, ff, qq = sp.symbols("s6 s8 f qq", positive=True)
d6, d8 = R**6 + ff**6, R**8 + ff**8
Gs = s6 / d6 + 3 * qq * s8 / d8
S8 = 3 * qq * s8
code_G1 = -6 * s6 * R**5 / d6**2 - 8 * S8 * R**7 / d8**2  # radial_g, g1
code_G2 = s6 * (-30 * R**4 / d6**2 + 72 * R**10 / d6**3) + S8 * (
    -56 * R**6 / d8**2 + 128 * R**14 / d8**3
)  # radial_g, g2
check("4. G'", is_zero(sp.diff(Gs, R) - code_G1))
check("4. G''", is_zero(sp.diff(Gs, R, 2) - code_G2))

# 5. Pair block of a radial function, with a concrete smooth phi.
xi = sp.symbols("xi0:3", real=True)
xj = sp.symbols("xj0:3", real=True)
rr = sp.sqrt(sum((xi[t] - xj[t]) ** 2 for t in range(3)))
lam = sp.symbols("lam", positive=True)
phi = sp.exp(-lam * R) / R**3
p1, p2 = sp.diff(phi, R), sp.diff(phi, R, 2)
nvec = [(xi[t] - xj[t]) / rr for t in range(3)]
ok5 = True
for u in range(3):
    for v in range(3):
        kblk = (
            p2 * nvec[u] * nvec[v]
            + p1 / rr * ((1 if u == v else 0) - nvec[u] * nvec[v])
        ).subs(R, rr)
        e = phi.subs(R, rr)
        ok5 &= num_zero(sp.diff(e, xi[u], xi[v]) - kblk)
        ok5 &= num_zero(sp.diff(e, xi[u], xj[v]) + kblk)
check("5. pair block of phi(|R_i - R_j|)", ok5)
bad = (p2 * nvec[0] * nvec[1]).subs(R, rr)  # the (1 - n n^T)/r part dropped
check(
    "5. control: pair block without (phi'/r)(1 - n n^T) fails",
    not num_zero(sp.diff(phi.subs(R, rr), xi[0], xi[1]) - bad),
)

# ---------------------------------------------------------------------------
# 6. Assembled Hessian on the 4-atom model of the gradient proof
# ---------------------------------------------------------------------------
KCN = 16  # const KCN
WF = 4  # const WF
Fr = sp.Rational
ELEM = {
    "X": {"rcov": Fr(189, 100), "sq": Fr(17, 10), "cnref": [Fr(0), Fr(1), Fr(5, 2)]},
    "Y": {"rcov": Fr(81, 100), "sq": Fr(9, 10), "cnref": [Fr(0), Fr(9, 10)]},
}
C6REF = {
    ("X", "X"): [
        [Fr(49), Fr(43), Fr(30)],
        [Fr(43), Fr(38), Fr(27)],
        [Fr(30), Fr(27), Fr(19)],
    ],
    ("X", "Y"): [[Fr(15), Fr(10)], [Fr(13), Fr(9)], [Fr(10), Fr(7)]],
    ("Y", "Y"): [[Fr(7), Fr(5)], [Fr(5), Fr(3)]],
}
PARAMS = {
    "s6": Fr(1),
    "s8": Fr(19889, 10000),
    "a1": Fr(3981, 10000),
    "a2": Fr(44211, 10000),
}
ATOMS = ["X", "Y", "X", "Y"]
GEOM = [
    [Fr(0), Fr(0), Fr(0)],
    [Fr(21, 10), Fr(3, 10), Fr(-1, 10)],
    [Fr(-13, 10), Fr(26, 10), Fr(2, 10)],
    [Fr(-4, 10), Fr(-11, 10), Fr(22, 10)],
]
NAT = len(ATOMS)
N3 = 3 * NAT


def c6ref(ei, ej, a, b, table):
    if (ei, ej) in table:
        return table[(ei, ej)][a][b]
    return table[(ej, ei)][b][a]


def mpf_geom(xyz):
    return [[mp.mpf(v) for v in row] for row in xyz]


def dist(x, i, j):
    return mp.sqrt(sum((x[i][t] - x[j][t]) ** 2 for t in range(3)))


def cns(x, elem):
    cn = [mp.mpf(0)] * NAT
    for i in range(NAT):
        for j in range(i):
            rcij = elem[ATOMS[i]]["rcov"] + elem[ATOMS[j]]["rcov"]
            g = 1 / (1 + mp.exp(-KCN * (rcij / dist(x, i, j) - 1)))
            cn[i] += g
            cn[j] += g
    return cn


def energy(xyz, elem, table):
    """Transcription of d3bj_energy (mpmath)."""
    x = mpf_geom(xyz)
    cn = cns(x, elem)
    e = mp.mpf(0)
    for i in range(NAT):
        for j in range(i):
            ei, ej = ATOMS[i], ATOMS[j]
            wi = [mp.exp(-WF * (cn[i] - r0) ** 2) for r0 in elem[ei]["cnref"]]
            wj = [mp.exp(-WF * (cn[j] - r0) ** 2) for r0 in elem[ej]["cnref"]]
            num = sum(
                c6ref(ei, ej, a, b, table) * wa * wb
                for a, wa in enumerate(wi)
                for b, wb in enumerate(wj)
            )
            den = sum(wa * wb for wa in wi for wb in wj)
            c6 = num / den
            q = mp.mpf(elem[ei]["sq"] * elem[ej]["sq"])
            fd = PARAMS["a1"] * mp.sqrt(3 * q) + PARAMS["a2"]
            r = dist(x, i, j)
            e -= PARAMS["s6"] * c6 / (r**6 + fd**6) + PARAMS["s8"] * 3 * c6 * q / (
                r**8 + fd**8
            )
    return e


def gradient(xyz, elem, table):
    """Transcription of d3bj_gradient (the two-pass form proved in
    d3_cn_chain_rule_gradient_sympy.py)."""
    x = mpf_geom(xyz)
    cn = cns(x, elem)
    grad = [mp.mpf(0)] * N3
    de_dcn = [mp.mpf(0)] * NAT
    for i in range(NAT):
        for j in range(i):
            c6, ci, cj = c6_derivs(i, j, cn, elem, table)[:3]
            q = mp.mpf(elem[ATOMS[i]]["sq"] * elem[ATOMS[j]]["sq"])
            g, g1, _ = radial(q, dist(x, i, j))
            r = dist(x, i, j)
            for t in range(3):
                comp = (x[i][t] - x[j][t]) / r * (-c6 * g1)
                grad[3 * i + t] += comp
                grad[3 * j + t] -= comp
            de_dcn[i] -= g * ci
            de_dcn[j] -= g * cj
    for i in range(NAT):
        for j in range(i):
            r = dist(x, i, j)
            cc = counting(i, j, r, elem)
            w = (de_dcn[i] + de_dcn[j]) * cc[1]
            for t in range(3):
                comp = (x[i][t] - x[j][t]) / r * w
                grad[3 * i + t] += comp
                grad[3 * j + t] -= comp
    return grad


def c6_derivs(i, j, cn, elem, table):
    """c6_second_derivatives: [C6, d_i, d_j, d_ii, d_ij, d_jj]."""
    ei, ej = ATOMS[i], ATOMS[j]
    n = [mp.mpf(0)] * 6
    d = [mp.mpf(0)] * 6
    for a, ra in enumerate(elem[ei]["cnref"]):
        wa = mp.exp(-WF * (cn[i] - ra) ** 2)
        al = -2 * WF * (cn[i] - ra)
        da, dda = al * wa, (al * al - 2 * WF) * wa
        for b, rb in enumerate(elem[ej]["cnref"]):
            wb = mp.exp(-WF * (cn[j] - rb) ** 2)
            bl = -2 * WF * (cn[j] - rb)
            db, ddb = bl * wb, (bl * bl - 2 * WF) * wb
            cc = mp.mpf(c6ref(ei, ej, a, b, table))
            terms = [wa * wb, da * wb, wa * db, dda * wb, da * db, wa * ddb]
            for t in range(6):
                n[t] += cc * terms[t]
                d[t] += terms[t]
    c6 = n[0] / d[0]
    ci = (n[1] - c6 * d[1]) / d[0]
    cj = (n[2] - c6 * d[2]) / d[0]
    cii = (n[3] - 2 * ci * d[1] - c6 * d[3]) / d[0]
    cij = (n[4] - ci * d[2] - cj * d[1] - c6 * d[4]) / d[0]
    cjj = (n[5] - 2 * cj * d[2] - c6 * d[5]) / d[0]
    return [c6, ci, cj, cii, cij, cjj]


def radial(q, r, g2_coeff=72):
    """radial_g: G, G', G''."""
    fd = PARAMS["a1"] * mp.sqrt(3 * q) + PARAMS["a2"]
    d6, d8 = r**6 + fd**6, r**8 + fd**8
    s6, s8 = PARAMS["s6"], 3 * q * PARAMS["s8"]
    g = s6 / d6 + s8 / d8
    g1 = -6 * s6 * r**5 / d6**2 - 8 * s8 * r**7 / d8**2
    g2 = s6 * (-30 * r**4 / d6**2 + g2_coeff * r**10 / d6**3) + s8 * (
        -56 * r**6 / d8**2 + 128 * r**14 / d8**3
    )
    return g, g1, g2


def counting(i, j, r, elem, one_minus_2c=True):
    """counting: c, c', c''."""
    rcij = elem[ATOMS[i]]["rcov"] + elem[ATOMS[j]]["rcov"]
    c = 1 / (1 + mp.exp(-KCN * (rcij / r - 1)))
    a = -KCN * rcij / r**2
    cp = a * c * (1 - c)
    cpp = -2 / r * cp + a * ((1 - 2 * c) if one_minus_2c else 1) * cp
    return c, cp, cpp


def zeros(n, m):
    return [[mp.mpf(0)] * m for _ in range(n)]


def madd(h, other, s=1):
    return [[h[a][b] + s * other[a][b] for b in range(N3)] for a in range(N3)]


def pair_block(h, x, i, j, d1, d2):
    r = dist(x, i, j)
    n = [(x[i][t] - x[j][t]) / r for t in range(3)]
    for u in range(3):
        for v in range(3):
            kk = d2 * n[u] * n[v] + d1 / r * ((1 if u == v else 0) - n[u] * n[v])
            h[3 * i + u][3 * i + v] += kk
            h[3 * j + u][3 * j + v] += kk
            h[3 * i + u][3 * j + v] -= kk
            h[3 * j + u][3 * i + v] -= kk


def radial_row(m, row, x, i, j, s):
    r = dist(x, i, j)
    for t in range(3):
        m[row][3 * i + t] += s * (x[i][t] - x[j][t]) / r
        m[row][3 * j + t] -= s * (x[i][t] - x[j][t]) / r


def code_hessian(xyz, elem, table, mixed=True, cncn=True, cn2=True, both=True, **kw):
    """Term-by-term port of d3bj_hessian. Keyword arguments switch on the
    negative-control mutations."""
    x = mpf_geom(xyz)
    cn = cns(x, elem)
    h = zeros(N3, N3)
    jac = zeros(NAT, N3)
    u = zeros(NAT, N3)
    m = zeros(NAT, NAT)
    e_cn = [mp.mpf(0)] * NAT
    pairs = [(i, j) for i in range(NAT) for j in range(i)]
    for i, j in pairs:
        cc = counting(i, j, dist(x, i, j), elem, kw.get("one_minus_2c", True))
        radial_row(jac, i, x, i, j, cc[1])
        radial_row(jac, j, x, i, j, cc[1])
    for i, j in pairs:
        c6, ci, cj, cii, cij, cjj = c6_derivs(i, j, cn, elem, table)
        q = mp.mpf(elem[ATOMS[i]]["sq"] * elem[ATOMS[j]]["sq"])
        g, g1, g2 = radial(q, dist(x, i, j), kw.get("g2_coeff", 72))
        pair_block(h, x, i, j, -c6 * g1, -c6 * g2)
        e_cn[i] -= g * ci
        e_cn[j] -= g * cj
        radial_row(u, i, x, i, j, -g1 * ci)
        radial_row(u, j, x, i, j, -g1 * cj)
        m[i][i] -= g * cii
        m[j][j] -= g * cjj
        m[i][j] -= g * cij
        m[j][i] -= g * cij
    # mixed = u^T jac, added with its transpose.
    mixed_t = [
        [sum(u[a][p] * jac[a][q] for a in range(NAT)) for q in range(N3)]
        for p in range(N3)
    ]
    mixed_tt = [[mixed_t[q][p] for q in range(N3)] for p in range(N3)]
    if mixed:
        h = madd(h, mixed_t)
        if both:
            h = madd(h, mixed_tt)
    if cncn:
        cc_term = [
            [
                sum(
                    jac[a][p] * m[a][b] * jac[b][q]
                    for a in range(NAT)
                    for b in range(NAT)
                )
                for q in range(N3)
            ]
            for p in range(N3)
        ]
        h = madd(h, cc_term)
    if cn2:
        for i, j in pairs:
            cc = counting(i, j, dist(x, i, j), elem, kw.get("one_minus_2c", True))
            w = e_cn[i] + e_cn[j]
            pair_block(h, x, i, j, w * cc[1], w * cc[2])
    return h


def maxdiff(a, b):
    return max(abs(a[p][q] - b[p][q]) for p in range(N3) for q in range(N3))


def maxabs(a):
    return max(abs(v) for row in a for v in row)


def displaced(xyz, p, s):
    out = [[mp.mpf(v) for v in row] for row in xyz]
    out[p // 3][p % 3] += s
    return out


H = code_hessian(GEOM, ELEM, C6REF)
scale = maxabs(H)
TOL = mp.mpf("1e-25") * scale

# (a) central FD of the transcribed gradient, h = 1e-20 (truncation ~1e-40).
hg = mp.mpf("1e-20")
H_fdg = zeros(N3, N3)
for q in range(N3):
    gp = gradient(displaced(GEOM, q, hg), ELEM, C6REF)
    gm = gradient(displaced(GEOM, q, -hg), ELEM, C6REF)
    for p in range(N3):
        H_fdg[p][q] = (gp[p] - gm[p]) / (2 * hg)

# (b) second difference of the energy, h = 1e-15 (truncation ~1e-30).
he = mp.mpf("1e-15")
H_fde = zeros(N3, N3)
for p in range(N3):
    for q in range(p, N3):
        acc = mp.mpf(0)
        for sp_, sq_, w in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]:
            xyz = displaced(GEOM, p, sp_ * he)
            xyz[q // 3][q % 3] += sq_ * he
            acc += w * energy(xyz, ELEM, C6REF)
        H_fde[p][q] = H_fde[q][p] = acc / (4 * he * he)

d_g = maxdiff(H, H_fdg)
d_e = maxdiff(H, H_fde)
check(
    "6a. code Hessian == 60-digit FD of the gradient",
    d_g < TOL,
    f"max|diff| = {mp.nstr(d_g, 3)}",
)
check(
    "6b. code Hessian == 60-digit second difference of E",
    d_e < TOL,
    f"max|diff| = {mp.nstr(d_e, 3)}",
)

asym = max(abs(H[p][q] - H[q][p]) for p in range(N3) for q in range(N3))
check("6.  symmetric", asym < TOL, mp.nstr(asym, 3))
trans = max(
    abs(sum(H[p][3 * b + t] for b in range(NAT))) for p in range(N3) for t in range(3)
)
check("6.  translational invariance sum_B H[a, B] = 0", trans < TOL, mp.nstr(trans, 3))

for name, kw in [
    ("mixed term", {"mixed": False}),
    ("CN-CN term", {"cncn": False}),
    ("e_a d2CN_a term", {"cn2": False}),
]:
    frac = maxdiff(H, code_hessian(GEOM, ELEM, C6REF, **kw)) / scale
    check(
        f"6.  {name} is a non-negligible part (> 1e-3 of max|H|)",
        frac > mp.mpf("1e-3"),
        f"{mp.nstr(frac, 3)}",
    )

for name, kw in [
    ("mixed term dropped", {"mixed": False}),
    ("CN-CN term dropped", {"cncn": False}),
    ("e_a d2CN_a term dropped", {"cn2": False}),
    ("mixed term without its transpose", {"both": False}),
    ("(1-2c) factor of c'' dropped", {"one_minus_2c": False}),
    ("72 -> 36 in G''", {"g2_coeff": 36}),
]:
    d = maxdiff(code_hessian(GEOM, ELEM, C6REF, **kw), H_fde)
    check(
        f"6.  control: {name} fails",
        d > mp.mpf("1e-6") * scale,
        f"max|diff| = {mp.nstr(d, 3)}",
    )

# Trivial-limit anchor: one reference per element => C6 constant, every CN
# term vanishes and the Hessian is the sum of direct pair blocks.
ELEM1 = {kk: dict(v, cnref=v["cnref"][:1]) for kk, v in ELEM.items()}
TABLE1 = {kk: [row[:1] for row in v[:1]] for kk, v in C6REF.items()}
H1 = code_hessian(GEOM, ELEM1, TABLE1)
H1_direct = code_hessian(GEOM, ELEM1, TABLE1, mixed=False, cncn=False, cn2=False)
d1 = maxdiff(H1, H1_direct)
check(
    "7.  anchor: one reference per element => CN terms == 0 (60-digit ulps)",
    d1 < mp.mpf("1e-50") * maxabs(H1),
    f"max|CN part| = {mp.nstr(d1, 3)}",
)

if FAILS:
    print(f"\n{len(FAILS)} check(s) FAILED: {FAILS}")
    sys.exit(1)
print("\nAll D3 Hessian checks passed.")
