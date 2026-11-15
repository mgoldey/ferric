#!/usr/bin/env python3
"""Sympy proofs: ferric's semilocal V_xc matrix elements are dE_xc/dD.

Checks the AO-matrix assembly in crates/ferric-dft/src/vxc.rs:
  - closed shell `semilocal_vxc_closed_scratch` (vxc.rs:262): the LDA piece
    w v_rho chi_mu chi_nu, the GGA piece M + M^T with f_b = 2 w v_sigma d_b rho,
    and the tau piece (1/2) w v_tau sum_b d_b chi_mu d_b chi_nu;
  - spin-polarized `semilocal_vxc_polarized_scratch` (vxc.rs:542): libxc
    packing (sigma_aa, sigma_ab, sigma_bb), G^a_b = 2 v_saa d_b rho_a +
    v_sab d_b rho_b (cross coefficient 1, not 2), per-spin tau piece;
  - the matrix-element form equals the functional-derivative form
    v_xc = v_rho - 2 div(v_sigma grad rho) up to a pure divergence
    (integration by parts), so both define the same Fock matrix.

The integrand is a GENERIC sympy Function e(rho, sigma, tau) (resp. of the 7
polarized variables), evaluated at one grid point with two AOs whose values
and gradients are free symbols, so nothing depends on a particular
functional. D_mu,nu and D_nu,mu are independent symbols, matching the
non-symmetrized derivative the code assembles before its final 1/2(V+V^T).

Numerical cross-check (independent route): a concrete meta-GGA-like
e(rho, sigma, tau) and a 7-variable polarized e with alpha-beta couplings on
4 grid points and 3 AOs with fixed values and gradients. The central
difference of E_xc(D) along every symmetric unit X must equal Tr[V X], with
V assembled as vxc.rs does; the bar is the Taylor remainder (h^2/6)|E'''|
(E''' from a third difference) plus rounding.

Negative controls: each identity is re-checked against a mutated code
formula and must fail. Exit status is nonzero on any mismatch.

Mutations (each turns the script red):
  * tau piece with w v_tau instead of (1/2) w v_tau in the numeric
    back-projection: "numeric closed" (and both polarized checks) fail;
  * polarized cross coefficient 2 v_sab instead of v_sab in the numeric G^s:
    "numeric polarized alpha" fails;
  * default GGA factor 1 instead of 2 in the symbolic closed_code:
    "closed: V_mn" fails.

Runtime: about 4 minutes (the symbolic part; the numeric block takes ~1 s).

Run: python3 scripts/dft_vxc_matrix_elements_sympy.py
"""

import sys

import mpmath as mp
import sympy as sp

FAILS = []


def check(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        FAILS.append(name)


NAO = 2
AX = range(3)
chi = sp.symbols("chi0:2", real=True)
dchi = [[sp.Symbol(f"g{m}{b}", real=True) for b in AX] for m in range(NAO)]


def dmat(name):
    return [
        [sp.Symbol(f"{name}{m}{n}", real=True) for n in range(NAO)] for m in range(NAO)
    ]


def density(D):
    rho = sum(D[m][n] * chi[m] * chi[n] for m in range(NAO) for n in range(NAO))
    grad = [
        sum(
            D[m][n] * (dchi[m][b] * chi[n] + chi[m] * dchi[n][b])
            for m in range(NAO)
            for n in range(NAO)
        )
        for b in AX
    ]
    tau = sp.Rational(1, 2) * sum(
        D[m][n] * dchi[m][b] * dchi[n][b]
        for b in AX
        for m in range(NAO)
        for n in range(NAO)
    )
    return rho, grad, tau


def dot(u, v):
    return sum(ui * vi for ui, vi in zip(u, v))


def zero(e):
    return sp.simplify(sp.expand(e)) == 0


# ── closed shell ───────────────────────────────────────────────────────────
D = dmat("D")
rho, grad, tau = density(D)
sigma = dot(grad, grad)
R, S, T = sp.symbols("R S T", real=True)
efun = sp.Function("e")(R, S, T)  # w * rho * eps_xc at the point (w absorbed)
at = {R: rho, S: sigma, T: tau}
v_r, v_s, v_t = (sp.diff(efun, X) for X in (R, S, T))


def closed_code(m, n, gga_fac=2, tau_fac=sp.Rational(1, 2)):
    f_b = [gga_fac * v_s * grad[b] for b in AX]
    lda = v_r * chi[m] * chi[n]
    gga = sum(
        f_b[b] * (chi[m] * dchi[n][b] + chi[n] * dchi[m][b]) for b in AX
    )  # M + M^T
    mgga = tau_fac * v_t * sum(dchi[m][b] * dchi[n][b] for b in AX)
    return lda + gga + mgga


E = efun.subs(at)
ok = all(
    zero(sp.diff(E, D[m][n]) - closed_code(m, n).subs(at))
    for m in range(NAO)
    for n in range(NAO)
)
check("closed: V_mn (LDA + 2 v_sigma M+M^T + 1/2 v_tau) == dE/dD_mn", ok)
check(
    "closed control: GGA factor 1 instead of 2 is detected",
    not zero(sp.diff(E, D[0][1]) - closed_code(0, 1, gga_fac=1).subs(at)),
)
check(
    "closed control: tau factor 1 instead of 1/2 is detected",
    not zero(sp.diff(E, D[0][1]) - closed_code(0, 1, tau_fac=1).subs(at)),
)

# ── spin polarized (libxc packing) ──────────────────────────────────────────
Da, Db = dmat("A"), dmat("B")
ra, ga, ta = density(Da)
rb, gb, tb = density(Db)
Ra, Rb, Saa, Sab, Sbb, Ta, Tb = sp.symbols("Ra Rb Saa Sab Sbb Ta Tb", real=True)
epol = sp.Function("e")(Ra, Rb, Saa, Sab, Sbb, Ta, Tb)
atp = {
    Ra: ra,
    Rb: rb,
    Saa: dot(ga, ga),
    Sab: dot(ga, gb),
    Sbb: dot(gb, gb),
    Ta: ta,
    Tb: tb,
}
Ep = epol.subs(atp)
dv = {X: sp.diff(epol, X) for X in (Ra, Rb, Saa, Sab, Sbb, Ta, Tb)}


def pol_code(m, n, spin, cross_fac=1):
    if spin == "a":
        vr, vself, vt, gs, gc = dv[Ra], dv[Saa], dv[Ta], ga, gb
    else:
        vr, vself, vt, gs, gc = dv[Rb], dv[Sbb], dv[Tb], gb, ga
    G = [2 * vself * gs[b] + cross_fac * dv[Sab] * gc[b] for b in AX]
    return (
        vr * chi[m] * chi[n]
        + sum(G[b] * (chi[m] * dchi[n][b] + chi[n] * dchi[m][b]) for b in AX)
        + sp.Rational(1, 2) * vt * sum(dchi[m][b] * dchi[n][b] for b in AX)
    )


ok_a = all(
    zero(sp.diff(Ep, Da[m][n]) - pol_code(m, n, "a").subs(atp))
    for m in range(NAO)
    for n in range(NAO)
)
ok_b = all(
    zero(sp.diff(Ep, Db[m][n]) - pol_code(m, n, "b").subs(atp))
    for m in range(NAO)
    for n in range(NAO)
)
check(
    "polarized: V^alpha_mn == dE/dD^alpha_mn (G^a = 2 v_saa grad rho_a + v_sab grad rho_b)",
    ok_a,
)
check("polarized: V^beta_mn == dE/dD^beta_mn", ok_b)
check(
    "polarized control: cross coefficient 2 v_sab is detected",
    not zero(sp.diff(Ep, Da[0][1]) - pol_code(0, 1, "a", cross_fac=2).subs(atp)),
)

# ── integration by parts: matrix-element form vs potential form ─────────────
x, y, z = sp.symbols("x y z", real=True)
X3 = (x, y, z)
fn = lambda name: sp.Function(name)(x, y, z)  # noqa: E731
vr3, vs3, rho3, cm, cn = fn("v_rho"), fn("v_sigma"), fn("rho"), fn("chi_m"), fn("chi_n")
grad3 = lambda u: [sp.diff(u, q) for q in X3]  # noqa: E731
matrix_form = vr3 * cm * cn + 2 * vs3 * dot(grad3(rho3), grad3(cm * cn))
v_xc = vr3 - 2 * sum(sp.diff(vs3 * sp.diff(rho3, q), q) for q in X3)
potential_form = v_xc * cm * cn
flux = [2 * vs3 * sp.diff(rho3, q) * cm * cn for q in X3]
divergence = sum(sp.diff(flux[i], X3[i]) for i in range(3))
check(
    "IBP: matrix form - (v_rho - 2 div(v_sigma grad rho)) chi chi == div(2 v_sigma grad rho chi chi)",
    zero(matrix_form - potential_form - divergence),
)
check(
    "IBP control: v_xc with -div(...) (factor 1) is detected",
    not zero(
        matrix_form
        - (vr3 - sum(sp.diff(vs3 * sp.diff(rho3, q), q) for q in X3)) * cm * cn
        - divergence
    ),
)

# ── numerical cross-check: central difference of E_xc(D) ────────────────────
# A concrete meta-GGA-like e(rho, sigma, tau) (and a 7-variable polarized one
# with alpha-beta couplings) on 4 grid points with 3 AOs whose values and
# gradients are fixed numbers. The independent route is the central
# difference (E(D + hX) - E(D - hX)) / 2h along each symmetric unit X; the
# code route is Tr[V X] with V assembled as vxc.rs does it (LDA GEMM,
# M + M^T with f_b = 2 w v_sigma d_b rho, (1/2) w v_tau, then 1/2(V + V^T);
# polarized G^s = 2 v_sss grad rho_s + v_sab grad rho_t). Bar: the central
# difference error is (h^2/6) |E'''| (Taylor remainder) plus rounding
# ~10^-dps |E| / h; |E'''| is estimated by a third difference at step H.
mp.mp.dps = 40
NG = 4
H_FD = mp.mpf("1e-8")
H3 = mp.mpf("1e-3")
CHI_N = [  # chi_mu(r_g): NAO_N x NG
    [mp.mpf(v) for v in row]
    for row in (
        ["0.62", "0.35", "0.18", "0.51"],
        ["0.27", "0.48", "0.66", "0.12"],
        ["0.41", "0.09", "0.33", "0.58"],
    )
]
NAO_N = len(CHI_N)
DCHI_N = [  # d_b chi_mu(r_g): [mu][b][g]
    [[mp.mpf((7 * m + 3 * b + 5 * g) % 11 - 5) / 10 for g in range(NG)] for b in AX]
    for m in range(NAO_N)
]
W_N = [mp.mpf(v) for v in ("0.3", "0.7", "0.5", "1.1")]
D_N = [
    [mp.mpf(v) for v in row]
    for row in (["0.9", "0.2", "-0.1"], ["0.2", "0.8", "0.15"], ["-0.1", "0.15", "0.7"])
]
DB_N = [
    [mp.mpf(v) for v in row]
    for row in (["0.5", "-0.1", "0.05"], ["-0.1", "0.6", "0.1"], ["0.05", "0.1", "0.4"])
]


def lam(expr, args):
    return sp.lambdify(args, expr, "mpmath")


RHO_, SIG_, TAU_ = sp.symbols("rho_ sigma_ tau_", positive=True)
E_CL = (
    -(RHO_ ** sp.Rational(4, 3))
    - sp.Rational(1, 5)
    * SIG_
    * RHO_ ** sp.Rational(-4, 3)
    / (1 + sp.Rational(3, 10) * SIG_ * RHO_ ** sp.Rational(-8, 3))
    + sp.Rational(1, 20) * TAU_**2 * RHO_ ** sp.Rational(-5, 3)
)
CL_ARGS = (RHO_, SIG_, TAU_)
e_cl = lam(E_CL, CL_ARGS)
de_cl = [lam(sp.diff(E_CL, v), CL_ARGS) for v in CL_ARGS]

PV = sp.symbols("ra_ rb_ saa_ sab_ sbb_ ta_ tb_", positive=True)
ra_, rb_, saa_, sab_, sbb_, ta_, tb_ = PV
E_POL = (
    E_CL.subs({RHO_: 2 * ra_, SIG_: 4 * saa_, TAU_: 2 * ta_}) / 2
    + E_CL.subs({RHO_: 2 * rb_, SIG_: 4 * sbb_, TAU_: 2 * tb_}) / 2
    - sp.Rational(1, 10) * ra_ * rb_ / (ra_ + rb_) ** sp.Rational(2, 3)
    - sp.Rational(3, 100) * sab_ * (ra_ + rb_) ** sp.Rational(-4, 3)
    + sp.Rational(1, 50) * ta_ * tb_ * (ra_ + rb_) ** sp.Rational(-5, 3)
)
e_pol = lam(E_POL, PV)
de_pol = [lam(sp.diff(E_POL, v), PV) for v in PV]


def grid_density(dm, g):
    rho = sum(
        dm[m][n] * CHI_N[m][g] * CHI_N[n][g] for m in range(NAO_N) for n in range(NAO_N)
    )
    grad = [
        sum(
            dm[m][n] * (DCHI_N[m][b][g] * CHI_N[n][g] + CHI_N[m][g] * DCHI_N[n][b][g])
            for m in range(NAO_N)
            for n in range(NAO_N)
        )
        for b in AX
    ]
    tau = (
        sum(
            dm[m][n] * DCHI_N[m][b][g] * DCHI_N[n][b][g]
            for b in AX
            for m in range(NAO_N)
            for n in range(NAO_N)
        )
        / 2
    )
    return rho, grad, tau


def energy_closed(dm):
    tot = 0
    for g in range(NG):
        rho, grad, tau = grid_density(dm, g)
        tot += W_N[g] * e_cl(rho, dot(grad, grad), tau)
    return tot


def energy_pol(dms):
    tot = 0
    for g in range(NG):
        ra, ga, ta = grid_density(dms[0], g)
        rb, gb, tb = grid_density(dms[1], g)
        tot += W_N[g] * e_pol(ra, rb, dot(ga, ga), dot(ga, gb), dot(gb, gb), ta, tb)
    return tot


def assemble(coef):
    """vxc.rs back-projection: coef[g] = (w v_rho, f_b (3), w v_tau) per point."""
    v = [[mp.mpf(0)] * NAO_N for _ in range(NAO_N)]
    for g in range(NG):
        lda, fb, vt = coef[g]
        for m in range(NAO_N):
            for n in range(NAO_N):
                v[m][n] += lda * CHI_N[m][g] * CHI_N[n][g]
                for b in AX:
                    mab = fb[b] * CHI_N[m][g] * DCHI_N[n][b][g]  # M_axis
                    v[m][n] += mab
                    v[n][m] += mab  # + M_axis^T
                    v[m][n] += (
                        sp.Rational(1, 2) * vt * DCHI_N[m][b][g] * DCHI_N[n][b][g]
                    )
    return [[(v[m][n] + v[n][m]) / 2 for n in range(NAO_N)] for m in range(NAO_N)]


def vxc_closed_code(dm, gga_fac=2):
    coef = []
    for g in range(NG):
        rho, grad, tau = grid_density(dm, g)
        vr, vs, vt = (f(rho, dot(grad, grad), tau) for f in de_cl)
        coef.append(
            (W_N[g] * vr, [gga_fac * W_N[g] * vs * grad[b] for b in AX], W_N[g] * vt)
        )
    return assemble(coef)


def vxc_pol_code(dms, spin):
    coef = []
    for g in range(NG):
        ra, ga, ta = grid_density(dms[0], g)
        rb, gb, tb = grid_density(dms[1], g)
        d = [f(ra, rb, dot(ga, ga), dot(ga, gb), dot(gb, gb), ta, tb) for f in de_pol]
        vr, vss, vt = (d[0], d[2], d[5]) if spin == 0 else (d[1], d[4], d[6])
        gs, gc = (ga, gb) if spin == 0 else (gb, ga)
        fb = [W_N[g] * (2 * vss * gs[b] + d[3] * gc[b]) for b in AX]
        coef.append((W_N[g] * vr, fb, W_N[g] * vt))
    return assemble(coef)


def unit_sym(m, n):
    x = [[mp.mpf(0)] * NAO_N for _ in range(NAO_N)]
    x[m][n] = x[n][m] = mp.mpf(1)
    return x


def shifted(dm, x, t):
    return [[dm[i][j] + t * x[i][j] for j in range(NAO_N)] for i in range(NAO_N)]


def fd_check(label, energy_of, v_code, base, slot):
    """max |FD - Tr[V X]| / bar over every symmetric unit X."""
    worst = absmax = 0
    for m in range(NAO_N):
        for n in range(m, NAO_N):
            x = unit_sym(m, n)

            def line(t):
                dms = [d for d in base]
                dms[slot] = shifted(base[slot], x, t)
                return energy_of(dms)

            fd = (line(H_FD) - line(-H_FD)) / (2 * H_FD)
            d3 = (line(2 * H3) - 2 * line(H3) + 2 * line(-H3) - line(-2 * H3)) / (
                2 * H3**3
            )
            bar = 10 * H_FD**2 / 6 * abs(d3) + mp.mpf(10) ** (5 - mp.mp.dps) / H_FD
            tr = sum(v_code[i][j] * x[i][j] for i in range(NAO_N) for j in range(NAO_N))
            worst = max(worst, abs(fd - tr) / bar)
            absmax = max(absmax, abs(fd - tr))
    check(
        f"{label} (max |FD - Tr[VX]| {mp.nstr(absmax, 3)}, / bar {mp.nstr(worst, 3)})",
        worst < 1,
    )
    return worst


fd_check(
    "numeric closed: central difference of E_xc(D) == Tr[V_code X]",
    lambda dms: energy_closed(dms[0]),
    vxc_closed_code(D_N),
    [D_N],
    0,
)
for spin, name in ((0, "alpha"), (1, "beta")):
    fd_check(
        f"numeric polarized {name}: central difference of E_xc(D^a, D^b) == Tr[V^s_code X]",
        energy_pol,
        vxc_pol_code([D_N, DB_N], spin),
        [D_N, DB_N],
        spin,
    )
# reachability: the bar is far below an O(1) factor error
bad = vxc_closed_code(D_N, gga_fac=1)
x01 = unit_sym(0, 1)
fd01 = (
    energy_closed(shifted(D_N, x01, H_FD)) - energy_closed(shifted(D_N, x01, -H_FD))
) / (2 * H_FD)
check(
    "numeric control: GGA factor 1 instead of 2 misses the central difference",
    abs(fd01 - 2 * bad[0][1]) > mp.mpf("1e-6"),
)

if FAILS:
    print(f"\n{len(FAILS)} check(s) FAILED: {FAILS}")
    sys.exit(1)
print("\nall V_xc matrix-element identities verified")
