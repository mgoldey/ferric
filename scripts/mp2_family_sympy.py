#!/usr/bin/env python3
"""Sympy and numpy proofs of the MP2-family energy identities ferric-mp2 codes.

Theory: `book/src/10-mp2/01-closed-shell-ri-mp2.md#closed-shell-ri-mp2-energy-splits`, `book/src/10-mp2/04-attenuated-mp2.md#attenuated-mp2-energy`, 8 and 13. Each part checks
an identity symbolically (sympy) or on a small random model against an
independent route (brute-force spin-orbital MP2, adaptive quadrature, mpmath),
prints its residual, and records a failure. The script runs every part and
exits 1 if any check failed.

  A  Closed-shell spin adaptation (rimp2.rs pair_energy). Symbolic, 2 occupied
     x 2 virtual spatial orbitals, every (ia|jb) a free symbol: the spin-orbital
     1/4 sum |<IJ||AB>|^2/D splits into E_OS = sum g^2/D (opposite-spin pairs)
     and E_SS = sum g(g - g^x)/D (same-spin pairs), g^x = (ib|ja).
  B  Unrestricted same-spin factor (u_rimp2.rs same_spin_block_energy).
     Symbolic, 2 alpha occupied x 2 alpha virtual: 1/4 sum_{ijab} K^2/D with
     K = (ia|jb) - (ib|ja) equals the alpha-alpha spin-orbital energy
     sum_{i<j, a<b} |<ij||ab>|^2/D.
  C  SCS and SCS-MP2(2terfc) (scs.rs scs_mp2, scs_mp2_2terfc). With
     E_x[r0] the spin components under terfc(r; r0)/r, the code's assembly
     c_OS E_OS[r1] + c_SS (E_SS[r2] - E_SS[r1]) has the operator limits
     (r1 -> 0, r2 -> inf) = c_SS E_SS[1/r] and (r1, r2 -> inf) = c_OS E_OS[1/r]
     (part D gives terfc -> 0 and -> 1 in those limits).
  D  terfc / terf kernel (ferric-integrals shim.cc terfc, Operator::terfc,
     ferric-dft vv10.rs terfc_scalar / terfc_scalar_decoupled):
     terf(r; r0) = 1/2 [erf(w(r - r0)) + erf(w(r + r0))], terf + terfc = 1,
     terfc(0) = 1, terf odd in r. The curvature constraint r0 w = 1/sqrt 2 is
     exactly the vanishing of the r^2 term of terf(r)/r at r = 0. The linked
     form (r -+ r0)/(r0 sqrt 2) equals the decoupled form at w = 1/(r0 sqrt 2).
     Limits: r0 -> inf gives terfc -> 1 (Coulomb), r0 -> 0 gives terfc -> 0
     for r > 0. The erfc control of att_vv10.rs uses erfc(w r) with the same w.
  E  MP2-V damping (vv10.rs Vv10Damping::factor_from_r2, damping_r_one):
     1 - terfc(R)^2 is 0 at R = 0 and -> 1 at large R, and beyond
     r_one = r0 + 5/w the factor is exactly 1.0 in float64 (mpmath, 30 digits).
  F  Laplace MP2 and SOS-MP2 (laplace.rs). Symbolic: int_0^inf e^{-tx} dt = 1/x
     for x > 0, so num/D = -num int e^{-tx} dt with D = -x, and the weighted
     factor B(t) = B exp(-t(e_a - e_i)/2) squares to the full exponent.
     Numerical: the RI-Laplace energy -int dt [2 sum J(t)^2 - sum G.G] and its
     Coulomb part -int dt sum J(t)^2, with J from the MO weighted factors and
     from the AO pseudo-densities P(t), Q(t), integrated by adaptive
     quadrature, equal brute-force spin-orbital MP2 (E_OS + E_SS) and E_OS;
     SOS-MP2 is c_OS E_OS.

Mutations (each turns the script red; the runner applies them one at a time
as exact substitutions of the lines tagged with these comments):
  * closed-shell kernel: g(2g - g^x)/D -> g(2g + g^x)/D breaks A ("FAIL A").
  * u_rimp2 same-spin prefactor 1/4 -> 1/2 breaks B ("FAIL B").
  * scs.rs same-spin window E_SS[r2] - E_SS[r1] reversed breaks C ("FAIL C").
  * Laplace half-exponent of B(t), 1/2 -> 1, breaks F ("FAIL F").
  * Operator::terfc linked width 1/(r0 sqrt 2) -> 1/(2 r0) breaks D ("FAIL D").

Run: python3 scripts/mp2_family_sympy.py  (needs sympy, numpy, scipy, mpmath)
"""

import itertools
import sys

import mpmath
import numpy as np
import sympy as sp
from scipy.integrate import quad

FAILED = []


def report(part, name, ok, detail=""):
    tag = "ok  " if ok else f"FAIL {part}"
    print(f"  {tag} {name} {detail}")
    if not ok:
        FAILED.append(f"{part}: {name}")


def check_zero(part, name, expr):
    ok = sp.simplify(sp.together(expr)) == 0
    report(part, name, ok)


def check_close(part, name, got, want, tol=1e-10):
    err = abs(got - want) / max(1.0, abs(want))
    report(part, name, err <= tol, f"got {got:.15e} want {want:.15e} rel {err:.1e}")


# ---------------------------------------------------------------- helpers
def closed_shell_kernel(g, gx, d):
    return g * (2 * g - gx) / d  # closed-shell kernel


def symbolic_integrals(no, nv):
    """(ia|jb) symbols with the (ia)<->(jb) symmetry of real orbitals."""
    g = {}
    for i, a, j, b in itertools.product(range(no), range(nv), range(no), range(nv)):
        key = min((i, a, j, b), (j, b, i, a))
        g[(i, a, j, b)] = sp.Symbol(f"g_{key[0]}{key[1]}{key[2]}{key[3]}")
    return g


def spin_orbital_parts(no, nv, g, eo, ev):
    """Brute-force 1/4 sum |<IJ||AB>|^2/D over spin orbitals 2p + s.

    Returns (opposite-spin, same-spin) sums. <IJ||AB> = (IA|JB) - (IB|JA)
    with spin delta on each chemist pair.
    """
    occ = [(i, s) for i in range(no) for s in (0, 1)]
    vir = [(a, s) for a in range(nv) for s in (0, 1)]
    e_os = e_ss = 0
    for (i, si), (j, sj) in itertools.product(occ, occ):
        for (a, sa), (b, sb) in itertools.product(vir, vir):
            direct = g[(i, a, j, b)] if (si == sa and sj == sb) else 0
            exch = g[(i, b, j, a)] if (si == sb and sj == sa) else 0
            asym = direct - exch
            if asym == 0:
                continue
            term = sp.Rational(1, 4) * asym**2 / (eo[i] + eo[j] - ev[a] - ev[b])
            if si == sj:
                e_ss += term
            else:
                e_os += term
    return e_os, e_ss


# ---------------------------------------------------------------- A
def part_a():
    print("A: closed-shell E_OS / E_SS from the spin-orbital MP2 energy (symbolic)")
    no, nv = 2, 2
    g = symbolic_integrals(no, nv)
    eo = sp.symbols("e_o0:2")
    ev = sp.symbols("e_v0:2")
    so_os, so_ss = spin_orbital_parts(no, nv, g, eo, ev)
    e_os = e_tot = 0
    for i, j, a, b in itertools.product(range(no), range(no), range(nv), range(nv)):
        d = eo[i] + eo[j] - ev[a] - ev[b]
        e_os += g[(i, a, j, b)] ** 2 / d
        e_tot += closed_shell_kernel(g[(i, a, j, b)], g[(i, b, j, a)], d)
    check_zero("A", "E_OS == opposite-spin spin-orbital sum", e_os - so_os)
    check_zero("A", "E_OS + E_SS == 1/4 sum |<ij||ab>|^2/D", e_tot - so_os - so_ss)


# ---------------------------------------------------------------- B
def part_b():
    print("B: unrestricted same-spin 1/4 factor (symbolic, one spin)")
    no, nv = 2, 2
    quarter = sp.Rational(1, 4)  # u_rimp2 same-spin
    g = symbolic_integrals(no, nv)
    eo = sp.symbols("e_o0:2")
    ev = sp.symbols("e_v0:2")
    code = 0
    for i, j, a, b in itertools.product(range(no), range(no), range(nv), range(nv)):
        k = g[(i, a, j, b)] - g[(i, b, j, a)]
        code += quarter * k**2 / (eo[i] + eo[j] - ev[a] - ev[b])
    # Brute force over alpha spin orbitals only: <ij||ab> = (ia|jb) - (ib|ja).
    ref = 0
    for i, j in itertools.combinations(range(no), 2):
        for a, b in itertools.combinations(range(nv), 2):
            k = g[(i, a, j, b)] - g[(i, b, j, a)]
            ref += k**2 / (eo[i] + eo[j] - ev[a] - ev[b])
    check_zero("B", "1/4 sum_ijab K^2/D == sum_{i<j,a<b} |<ij||ab>|^2/D", code - ref)


# ---------------------------------------------------------------- D (kernel)
def terf_expr(r, r0, w):
    return sp.Rational(1, 2) * (sp.erf(w * (r - r0)) + sp.erf(w * (r + r0)))


def part_d():
    print("D: terfc kernel, curvature constraint, limits")
    r = sp.Symbol("r", positive=True)
    r0, w = sp.symbols("r0 omega", positive=True)
    omega_link = 1 / (r0 * sp.sqrt(2))  # Operator::terfc
    terf = terf_expr(r, r0, w)
    terfc = 1 - terf
    check_zero("D", "terf + terfc == 1", terf + terfc - 1)
    check_zero("D", "terfc(0) == 1", terfc.subs(r, 0) - 1)
    x = sp.Symbol("x", real=True)
    t_x = terf_expr(x, r0, w)
    check_zero("D", "terf(-r) == -terf(r)", t_x.subs(x, -x) + t_x)
    # terf(r)/r = c1 + c3 r^2 + ...; c3 is the curvature at the origin.
    ser = sp.series(t_x / x, x, 0, 4).removeO()
    c3 = sp.simplify(ser.coeff(x, 2))
    sol = sp.solve(sp.Eq(c3, 0), w)
    check_zero(
        "D", "curvature r^2 term vanishes at the linked omega", c3.subs(w, omega_link)
    )
    report(
        "D", "the linked omega is the only positive root", len(sol) == 1, f"roots {sol}"
    )
    if sol:
        check_zero("D", "root == 1/(r0 sqrt 2)", sol[0] - 1 / (r0 * sp.sqrt(2)))
    linked = sp.Rational(1, 2) * (
        sp.erf((r - r0) / (r0 * sp.sqrt(2))) + sp.erf((r + r0) / (r0 * sp.sqrt(2)))
    )
    check_zero(
        "D",
        "linked vv10 form == decoupled form at the linked omega",
        linked - terf.subs(w, omega_link),
    )
    tc_link = 1 - terf.subs(w, omega_link)
    lim_inf = sp.limit(tc_link, r0, sp.oo)
    lim_zero = sp.limit(tc_link, r0, 0, "+")
    report(
        "D",
        "r0 -> inf: terfc -> 1 (Coulomb)",
        sp.simplify(lim_inf - 1) == 0,
        f"{lim_inf}",
    )
    report(
        "D", "r0 -> 0: terfc -> 0 at r > 0", sp.simplify(lim_zero) == 0, f"{lim_zero}"
    )
    report("D", "r -> inf: terfc -> 0", sp.limit(tc_link, r, sp.oo) == 0)
    return tc_link, r, r0


# ---------------------------------------------------------------- C
def part_c():
    print("C: SCS and SCS-MP2(2terfc) assembly and its operator limits")
    c_os, c_ss = sp.symbols("c_OS c_SS")
    e_os_c, e_ss_c = sp.symbols("E_OS_coul E_SS_coul")

    # MP2 is quadratic in the integrals; under terfc(r; r0)/r every integral is
    # the Coulomb one times the kernel limit, so a component scales by lim^2.
    def components(lim):
        return lim**2 * e_os_c, lim**2 * e_ss_c

    def code_2terfc(lim1, lim2):
        e_os1, e_ss1 = components(lim1)
        _, e_ss2 = components(lim2)
        e_ss_win = e_ss2 - e_ss1  # scs.rs
        return c_os * e_os1 + c_ss * e_ss_win

    # Part D: terfc -> 0 as r0 -> 0, -> 1 as r0 -> inf.
    check_zero(
        "C", "(r1 -> 0, r2 -> inf) == c_SS E_SS[1/r]", code_2terfc(0, 1) - c_ss * e_ss_c
    )
    check_zero(
        "C", "(r1, r2 -> inf) == c_OS E_OS[1/r]", code_2terfc(1, 1) - c_os * e_os_c
    )
    check_zero(
        "C",
        "SCS at (1, 1) == MP2",
        (c_os * e_os_c + c_ss * e_ss_c).subs({c_os: 1, c_ss: 1}) - e_os_c - e_ss_c,
    )


# ---------------------------------------------------------------- E
def part_e():
    print("E: MP2-V damping 1 - terfc^2 and the r_one skip")
    r = sp.Symbol("r", positive=True)
    r0 = sp.Symbol("r0", positive=True)
    tc = 1 - terf_expr(r, r0, 1 / (r0 * sp.sqrt(2)))
    damp = 1 - tc**2
    check_zero("E", "damping(0) == 0", damp.subs(r, 0))
    report("E", "damping -> 1 at large R", sp.limit(damp, r, sp.oo) == 1)
    mpmath.mp.dps = 30
    damp_one_sigmas = 5.0  # vv10.rs DAMP_ONE_SIGMAS
    for r0v in (0.5, 1.0 * 1.8897261246, 2.5):
        for omega in (None, 0.3, 2.0):
            w = 1.0 / (r0v * np.sqrt(2.0)) if omega is None else omega
            r_one = r0v + damp_one_sigmas / w
            tcv = 0.5 * mpmath.erfc(w * (r_one - r0v)) + 0.5 * mpmath.erfc(
                w * (r_one + r0v)
            )
            tcf = float(tcv)
            ok = tcf < 1e-12 and (1.0 - tcf * tcf) == 1.0
            report(
                "E",
                f"r0={r0v:.3f} omega={omega}: factor == 1.0 beyond r_one",
                ok,
                f"terfc {tcf:.2e}",
            )


# ---------------------------------------------------------------- F
def part_f():
    print("F: Laplace MP2 / SOS-MP2")
    t, x = sp.symbols("t x", positive=True)
    check_zero(
        "F",
        "int_0^inf exp(-t x) dt == 1/x",
        sp.integrate(sp.exp(-t * x), (t, 0, sp.oo)) - 1 / x,
    )
    ei, ej, ea, eb = sp.symbols("e_i e_j e_a e_b", real=True)
    half_w = sp.Rational(1, 2)  # Laplace factor
    w_ia = sp.exp(-half_w * t * (ea - ei))
    w_jb = sp.exp(-half_w * t * (eb - ej))
    # J_PQ(t)^2 carries w_ia^2 w_jb^2 for each (ia, jb) pair.
    full = sp.exp(-t * (ea + eb - ei - ej))
    check_zero(
        "F",
        "B(t) weights square to exp(-t x)",
        sp.simplify(sp.expand_power_exp(w_ia**2 * w_jb**2) - full),
    )
    # The AO pseudo-densities carry exp(+t e_i) and exp(-t e_a) unhalved.
    check_zero(
        "F",
        "P(t) Q(t) weights == B(t) weight squared",
        sp.simplify(sp.exp(t * ei) * sp.exp(-t * ea) - w_ia**2),
    )

    rng = np.random.default_rng(20261114)
    no, nv, naux = 3, 4, 8
    nbf = no + nv
    b_ao = rng.standard_normal((naux, nbf, nbf))
    b_ao = 0.5 * (b_ao + b_ao.transpose(0, 2, 1))
    c = np.linalg.qr(rng.standard_normal((nbf, nbf)))[0]
    eo = np.sort(rng.uniform(-2.0, -0.5, no))
    ev = np.sort(rng.uniform(0.3, 2.5, nv))
    b = np.einsum("Pmn,mi,na->Pia", b_ao, c[:, :no], c[:, no:])
    g = np.einsum("Pia,Pjb->iajb", b, b)

    # Brute-force spin-orbital MP2 (numeric g in the helper).
    gd = {
        k: g[k] for k in itertools.product(range(no), range(nv), range(no), range(nv))
    }
    so_os, so_ss = spin_orbital_parts(no, nv, gd, eo, ev)
    so_os, so_ss = float(so_os), float(so_ss)
    hw = float(half_w)

    def integrand(tv, coulomb_only, ao):
        bt = b * np.exp(-hw * tv * (ev[None, None, :] - eo[None, :, None]))
        if ao:
            pt = c[:, :no] @ np.diag(np.exp(tv * eo)) @ c[:, :no].T
            qt = c[:, no:] @ np.diag(np.exp(-tv * ev)) @ c[:, no:].T
            jpq = np.einsum("Pmn,nl,Qlk,km->PQ", b_ao, pt, b_ao, qt)
        else:
            jpq = np.einsum("Pia,Qia->PQ", bt, bt)
        coul = np.sum(jpq**2)
        if coulomb_only:
            return -coul
        gpq = np.einsum("Pia,Qja->PQij", bt, bt)
        return -(2.0 * coul - np.einsum("PQij,PQji->", gpq, gpq))

    opts = dict(epsabs=1e-14, epsrel=1e-13, limit=200)
    e_mo = quad(integrand, 0, np.inf, args=(False, False), **opts)[0]
    e_os_mo = quad(integrand, 0, np.inf, args=(True, False), **opts)[0]
    e_os_ao = quad(integrand, 0, np.inf, args=(True, True), **opts)[0]
    check_close(
        "F",
        "RI-Laplace MP2 (MO) == spin-orbital E_OS + E_SS",
        e_mo,
        so_os + so_ss,
        1e-9,
    )
    check_close(
        "F", "Laplace Coulomb term (MO) == spin-orbital E_OS", e_os_mo, so_os, 1e-9
    )
    check_close(
        "F", "Laplace Coulomb term (AO P(t),Q(t)) == E_OS", e_os_ao, so_os, 1e-9
    )
    c_os = 1.3
    check_close("F", "SOS-MP2 = c_OS E_OS", c_os * e_os_ao, c_os * so_os, 1e-9)


def main():
    part_a()
    part_b()
    part_d()
    part_c()
    part_e()
    part_f()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): {FAILED}")
        sys.exit(1)
    print("all MP2-family energy identities hold")


if __name__ == "__main__":
    main()
