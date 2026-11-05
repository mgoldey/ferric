#!/usr/bin/env python3
"""Sympy proof of the D3(BJ) nuclear gradient, including the CN chain rule.

The two-body D3(BJ) energy depends on the geometry directly (the damped
R^-6 / R^-8 pair terms) and indirectly through C6_AB(CN_A, CN_B), where every
coordination number CN_A depends on every distance R_AB. This script checks
the gradient ferric implements in crates/ferric-d3/src/lib.rs:

  1. counting function   g(R) = 1/(1+exp(-k(rc/R - 1)))
                         dg/dR = -k rc/R^2 g(1-g)               (lib.rs:484-487)
  2. Gaussian weights    W_i(CN) = exp(-wf (CN - CNref_i)^2)    (lib.rs:202-211)
                         dW_i/dCN = -2 wf (CN - CNref_i) W_i    (lib.rs:527,530)
  3. C6 interpolation    C6 = N/D, dC6/dCN = (dN/dCN - C6 dD/dCN)/D
                                                                (lib.rs:213-246, 513-548)
  4. BJ pair terms       dE_p/dR  = s6 C6 6R^5/d6^2 + s8 C8 8R^7/d8^2
                                                                (lib.rs:449-450)
                         dE_p/dC6 = -(s6/d6 + 3 qq s8/d8)       (lib.rs:463)
                         R0 = sqrt(C8/C6) = sqrt(3 qq), no geometry dependence
                                                                (lib.rs:428-430)
  5. assembled gradient (two passes: explicit pair term + banked dE/dCN_A,
     then sum_{pairs} (dE/dCN_i + dE/dCN_j) dg/dR_ij dR_ij/dx)
                                                                (lib.rs:368-500)
     for a 4-atom, two-element model with 3 and 2 reference CNs, the code's
     KCN = 16 and WF = 4 (lib.rs:97,99), equals
       (a) sympy's exact derivative of the energy (lib.rs:262-339 transcribed),
       (b) a 60-digit central finite difference of the same energy.

Anchors and controls:
  - trivial limit: one reference per element => dC6/dCN = 0 identically and the
    chain term vanishes (to 60-digit rounding);
  - the chain term is a non-negligible fraction of the gradient in the model;
  - negative controls (each must FAIL): chain term dropped, only the i-side of
    dE/dCN banked, dW/dCN with -wf instead of -2 wf, dg/dR sign flipped.

Exit status is nonzero on any mismatch.

Run: python3 scripts/d3_cn_chain_rule_gradient_sympy.py
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
    # cancel() is exact rational-function arithmetic over the exp/symbol
    # generators, so a zero result is a proof. simplify() on the 6-term C6
    # quotient takes minutes. The negative controls that rely on a nonzero
    # result are backed by the numeric controls of claim 5.
    return sp.cancel(sp.together(expr)) == 0


# ---------------------------------------------------------------------------
# 1-4. Symbolic identities on the building blocks
# ---------------------------------------------------------------------------
k, rc, R, wf = sp.symbols("k rc R wf", positive=True)
g = 1 / (1 + sp.exp(-k * (rc / R - 1)))
code_dg = -k * rc / R**2 * g * (1 - g)  # lib.rs:487
check("1. dg/dR = -k rc/R^2 g(1-g)", is_zero(sp.diff(g, R) - code_dg))
check(
    "1. control: +k rc/R^2 g(1-g) fails",
    not is_zero(sp.diff(g, R) + code_dg),
)

CN, cnr = sp.symbols("CN cnr", real=True)
W = sp.exp(-wf * (CN - cnr) ** 2)
check(
    "2. dW/dCN = -2 wf (CN - CNref) W",
    is_zero(sp.diff(W, CN) - (-2 * wf * (CN - cnr) * W)),
)

# C6 interpolation for nref_A = 3, nref_B = 2 with symbolic reference data.
CNA, CNB = sp.symbols("CN_A CN_B", real=True)
ra = sp.symbols("a0:3", real=True)
rb = sp.symbols("b0:2", real=True)
c = sp.Matrix(3, 2, lambda i, j: sp.Symbol(f"c{i}{j}", real=True))
Wa = [sp.exp(-wf * (CNA - x) ** 2) for x in ra]
Wb = [sp.exp(-wf * (CNB - x) ** 2) for x in rb]
dWa = [-2 * wf * (CNA - x) * w for x, w in zip(ra, Wa)]
dWb = [-2 * wf * (CNB - x) * w for x, w in zip(rb, Wb)]
Num = sum(c[i, j] * Wa[i] * Wb[j] for i in range(3) for j in range(2))
Den = sum(Wa[i] * Wb[j] for i in range(3) for j in range(2))
C6 = Num / Den
dN_A = sum(c[i, j] * dWa[i] * Wb[j] for i in range(3) for j in range(2))
dD_A = sum(dWa[i] * Wb[j] for i in range(3) for j in range(2))
dN_B = sum(c[i, j] * Wa[i] * dWb[j] for i in range(3) for j in range(2))
dD_B = sum(Wa[i] * dWb[j] for i in range(3) for j in range(2))
code_dC6_A = (dN_A - C6 * dD_A) / Den  # lib.rs:544
code_dC6_B = (dN_B - C6 * dD_B) / Den
check("3. dC6/dCN_A quotient form", is_zero(sp.diff(C6, CNA) - code_dC6_A))
check("3. dC6/dCN_B quotient form", is_zero(sp.diff(C6, CNB) - code_dC6_B))
check(
    "3. control: dropping -C6 dD/dCN fails",
    not is_zero(sp.diff(C6, CNA) - dN_A / Den),
)

s6, s8, a1, a2, qq, c6s = sp.symbols("s6 s8 a1 a2 qq C6", positive=True)
c8s = 3 * c6s * qq
r0 = sp.sqrt(c8s / c6s)
check("4. R0 = sqrt(C8/C6) = sqrt(3 qq) (C6 cancels)", is_zero(r0 - sp.sqrt(3 * qq)))
f = a1 * sp.sqrt(3 * qq) + a2
d6, d8 = R**6 + f**6, R**8 + f**8
Ep = -(s6 * c6s / d6 + s8 * c8s / d8)  # lib.rs:335
code_dEdR = s6 * c6s * 6 * R**5 / d6**2 + s8 * c8s * 8 * R**7 / d8**2  # :449-450
code_dEdC6 = -(s6 / d6 + 3 * qq * s8 / d8)  # lib.rs:463
check("4. dE_pair/dR", is_zero(sp.diff(Ep, R) - code_dEdR))
check("4. dE_pair/dC6", is_zero(sp.diff(Ep, c6s) - code_dEdC6))

# ---------------------------------------------------------------------------
# 5. Assembled gradient on a 4-atom model
# ---------------------------------------------------------------------------
KCN = 16  # lib.rs:97
WF = 4  # lib.rs:99
# Two model elements. rcov in Bohr (lib.rs:107 scale), sqrt_q (lib.rs:117),
# reference CNs. The C6 table is stored for (X,X), (X,Y), (Y,Y); (Y,X) is the
# transpose, which is what reference_c6's swap (lib.rs:135-139) enforces.
Fr = sp.Rational
ELEM = {
    "X": dict(rcov=Fr(189, 100), sq=Fr(17, 10), cnref=[Fr(0), Fr(1), Fr(5, 2)]),
    "Y": dict(rcov=Fr(81, 100), sq=Fr(9, 10), cnref=[Fr(0), Fr(9, 10)]),
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
PARAMS = dict(
    s6=Fr(1), s8=Fr(12177, 10000), a1=Fr(4145, 10000), a2=Fr(48593, 10000)
)  # PBE0

ATOMS = ["X", "Y", "X", "Y"]
# Generic, non-symmetric geometry in the transitional CN region (Bohr).
GEOM = [
    [Fr(0), Fr(0), Fr(0)],
    [Fr(21, 10), Fr(3, 10), Fr(-1, 10)],
    [Fr(-13, 10), Fr(26, 10), Fr(2, 10)],
    [Fr(-4, 10), Fr(-11, 10), Fr(22, 10)],
]


def c6ref(ei, ej, a, b, table):
    if (ei, ej) in table:
        return table[(ei, ej)][a][b]
    return table[(ej, ei)][b][a]


def energy(xyz, elem, table, exp, sqrt):
    """Transcription of d3bj_energy (lib.rs:262-339) over a generic backend."""
    n = len(ATOMS)

    def dist(i, j):
        return sqrt(sum((xyz[i][a] - xyz[j][a]) ** 2 for a in range(3)))

    cn = [0] * n
    for i in range(n):  # coordination_numbers, lib.rs:170-190
        for j in range(i):
            rc = elem[ATOMS[i]]["rcov"] + elem[ATOMS[j]]["rcov"]
            gij = 1 / (1 + exp(-KCN * (rc / dist(i, j) - 1)))
            cn[i] += gij
            cn[j] += gij
    e = 0
    for i in range(n):
        for j in range(i):
            ei, ej = ATOMS[i], ATOMS[j]
            wi = [exp(-WF * (cn[i] - x) ** 2) for x in elem[ei]["cnref"]]
            wj = [exp(-WF * (cn[j] - x) ** 2) for x in elem[ej]["cnref"]]
            num = sum(
                c6ref(ei, ej, a, b, table) * wa * wb
                for a, wa in enumerate(wi)
                for b, wb in enumerate(wj)
            )
            den = sum(wa * wb for wa in wi for wb in wj)
            c6 = num / den
            q = elem[ei]["sq"] * elem[ej]["sq"]
            c8 = 3 * c6 * q
            fd = PARAMS["a1"] * sqrt(3 * q) + PARAMS["a2"]
            r = dist(i, j)
            e -= PARAMS["s6"] * c6 / (r**6 + fd**6) + PARAMS["s8"] * c8 / (r**8 + fd**8)
    return e


def code_gradient(
    xyz,
    elem,
    table,
    chain=True,
    i_side_only=False,
    wf_factor=2,
    dg_sign=-1,
    return_parts=False,
):
    """Term-by-term mpmath port of d3bj_gradient (lib.rs:368-500) and
    c6_interpolated_derivatives (lib.rs:513-548). Keyword arguments switch on
    the negative-control mutations."""
    n = len(ATOMS)
    x = [[mp.mpf(v) for v in row] for row in xyz]

    def dist(i, j):
        return mp.sqrt(sum((x[i][a] - x[j][a]) ** 2 for a in range(3)))

    cn = [mp.mpf(0)] * n
    for i in range(n):
        for j in range(i):
            rc = elem[ATOMS[i]]["rcov"] + elem[ATOMS[j]]["rcov"]
            gij = 1 / (1 + mp.exp(-KCN * (rc / dist(i, j) - 1)))
            cn[i] += gij
            cn[j] += gij

    grad = [[mp.mpf(0)] * 3 for _ in range(n)]
    explicit = [[mp.mpf(0)] * 3 for _ in range(n)]
    de_dcn = [mp.mpf(0)] * n
    for i in range(n):
        for j in range(i):
            ei, ej = ATOMS[i], ATOMS[j]
            r = dist(i, j)
            # c6_interpolated_derivatives, lib.rs:513-548
            num = den = dni = ddi = dnj = ddj = mp.mpf(0)
            for a, cri in enumerate(elem[ei]["cnref"]):
                wa = mp.exp(-WF * (cn[i] - cri) ** 2)
                da = -wf_factor * WF * (cn[i] - cri) * wa  # lib.rs:527
                for b, crj in enumerate(elem[ej]["cnref"]):
                    wb = mp.exp(-WF * (cn[j] - crj) ** 2)
                    db = -2 * WF * (cn[j] - crj) * wb  # lib.rs:530
                    cc = mp.mpf(c6ref(ei, ej, a, b, table))
                    num += cc * wa * wb
                    den += wa * wb
                    dni += cc * da * wb
                    ddi += da * wb
                    dnj += cc * wa * db
                    ddj += wa * db
            c6 = num / den
            dc6_di = (dni - c6 * ddi) / den  # lib.rs:544
            dc6_dj = (dnj - c6 * ddj) / den
            q = mp.mpf(elem[ei]["sq"] * elem[ej]["sq"])  # lib.rs:426
            c8 = 3 * c6 * q  # lib.rs:427
            r0 = mp.sqrt(3 * q)  # lib.rs:430
            fdmp = PARAMS["a1"] * r0 + PARAMS["a2"]  # lib.rs:432
            d6 = r**6 + fdmp**6
            d8 = r**8 + fdmp**8
            de_dr = (
                PARAMS["s6"] * c6 * 6 * r**5 / d6**2
                + PARAMS["s8"] * c8 * 8 * r**7 / d8**2
            )  # lib.rs:449-450
            for a in range(3):  # lib.rs:454-458
                comp = (x[i][a] - x[j][a]) / r * de_dr
                grad[i][a] += comp
                grad[j][a] -= comp
                explicit[i][a] += comp
                explicit[j][a] -= comp
            de_dc6 = -(PARAMS["s6"] / d6 + 3 * q * PARAMS["s8"] / d8)  # lib.rs:463
            if chain:  # lib.rs:465-466
                de_dcn[i] += de_dc6 * dc6_di
                if not i_side_only:
                    de_dcn[j] += de_dc6 * dc6_dj
    for i in range(n):  # contraction pass, lib.rs:474-497
        for j in range(i):
            r = dist(i, j)
            rc = elem[ATOMS[i]]["rcov"] + elem[ATOMS[j]]["rcov"]
            gij = 1 / (1 + mp.exp(-KCN * (rc / r - 1)))
            dg_dr = dg_sign * KCN * rc / r**2 * gij * (1 - gij)  # lib.rs:487
            w = (de_dcn[i] + de_dcn[j]) * dg_dr  # lib.rs:489
            for a in range(3):
                comp = (x[i][a] - x[j][a]) / r * w
                grad[i][a] += comp
                grad[j][a] -= comp
    if return_parts:
        return grad, explicit
    return grad


def maxdiff(a, b):
    return max(
        abs(mp.mpf(a[i][k]) - mp.mpf(b[i][k])) for i in range(len(a)) for k in range(3)
    )


def maxabs(a):
    return max(abs(mp.mpf(v)) for row in a for v in row)


# (a) sympy's exact derivative of the transcribed energy.
syms = [[sp.Symbol(f"x{i}{a}", real=True) for a in range(3)] for i in range(len(ATOMS))]
E_sym = energy(syms, ELEM, C6REF, sp.exp, sp.sqrt)
flat = [s for row in syms for s in row]
point = [mp.mpf(GEOM[i][a]) for i in range(len(ATOMS)) for a in range(3)]


def sym_gradient(expr):
    """Exact sympy derivative, evaluated in 60-digit mpmath."""
    out = []
    for i in range(len(ATOMS)):
        row = []
        for a in range(3):
            fn = sp.lambdify(flat, sp.diff(expr, syms[i][a]), modules="mpmath")
            row.append(fn(*point))
        out.append(row)
    return out


g_sym = sym_gradient(E_sym)

# (b) 60-digit central finite difference of the same energy.
h = mp.mpf("1e-15")


def e_num(xyz):
    return energy(
        [[mp.mpf(v) for v in row] for row in xyz], ELEM, C6REF, mp.exp, mp.sqrt
    )


g_fd = []
for i in range(len(ATOMS)):
    row = []
    for a in range(3):
        p = [[mp.mpf(v) for v in rr] for rr in GEOM]
        m = [[mp.mpf(v) for v in rr] for rr in GEOM]
        p[i][a] += h
        m[i][a] -= h
        row.append((e_num(p) - e_num(m)) / (2 * h))
    g_fd.append(row)

g_code, g_expl = code_gradient(GEOM, ELEM, C6REF, return_parts=True)
scale = maxabs(g_code)
TOL = mp.mpf("1e-25") * scale

d_sym = maxdiff(g_code, g_sym)
d_fd = maxdiff(g_code, g_fd)
d_symfd = maxdiff(g_sym, g_fd)
check(
    "5a. code gradient == sympy dE/dR", d_sym < TOL, f"max|diff| = {mp.nstr(d_sym, 3)}"
)
check(
    "5b. code gradient == 60-digit central FD",
    d_fd < TOL,
    f"max|diff| = {mp.nstr(d_fd, 3)}",
)
check("5.  sympy == FD", d_symfd < TOL, f"max|diff| = {mp.nstr(d_symfd, 3)}")

net = max(abs(sum(g_code[i][a] for i in range(len(ATOMS)))) for a in range(3))
check("5.  translational invariance sum_A dE/dR_A = 0", net < TOL, f"{mp.nstr(net, 3)}")

chain_frac = maxdiff(g_code, g_expl) / scale
check(
    "5.  chain term is non-negligible (|chain|/|grad| > 1e-2)",
    chain_frac > mp.mpf("1e-2"),
    f"max|chain|/max|grad| = {mp.nstr(chain_frac, 3)}",
)

# Negative controls: each mutation must disagree with sympy far above TOL.
for name, kw in [
    ("chain term dropped", dict(chain=False)),
    ("only i-side dE/dCN banked", dict(i_side_only=True)),
    ("dW/dCN with -wf (not -2 wf)", dict(wf_factor=1)),
    ("dg/dR sign flipped", dict(dg_sign=+1)),
]:
    d = maxdiff(code_gradient(GEOM, ELEM, C6REF, **kw), g_sym)
    check(
        f"5.  control: {name} fails",
        d > mp.mpf("1e-6") * scale,
        f"max|diff| = {mp.nstr(d, 3)}",
    )

# Trivial-limit anchor: one reference per element => C6 is a constant, the
# chain term is exactly zero, and the explicit term alone is the gradient.
ELEM1 = {kk: dict(v, cnref=v["cnref"][:1]) for kk, v in ELEM.items()}
TABLE1 = {kk: [row[:1] for row in v[:1]] for kk, v in C6REF.items()}
g1, g1_expl = code_gradient(GEOM, ELEM1, TABLE1, return_parts=True)
# (dN - C6 dD)/D is c*dW*W' - (c*W*W'/(W*W'))*dW*W' with one reference: zero
# in exact arithmetic, a few ulps at 60 digits.
d1 = maxdiff(g1, g1_expl)
check(
    "6.  anchor: one reference per element => chain term == 0 (60-digit ulps)",
    d1 < mp.mpf("1e-50") * maxabs(g1),
    f"max|chain| = {mp.nstr(d1, 3)}",
)
E1 = energy(syms, ELEM1, TABLE1, sp.exp, sp.sqrt)
g1_sym = sym_gradient(E1)
check(
    "6.  anchor: explicit term == sympy gradient there",
    maxdiff(g1, g1_sym) < mp.mpf("1e-25") * maxabs(g1),
)

if FAILS:
    print(f"\n{len(FAILS)} check(s) FAILED: {FAILS}")
    sys.exit(1)
print("\nAll D3 gradient checks passed.")
