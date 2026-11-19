#!/usr/bin/env python3
"""Proofs: the eigenvalue-self-consistent GW updates (evGW0 and evGW) of ferric-gw.

Derivation: `book/src/19-gw-bse/06-eigenvalue-self-consistency.md`. The code
transcribed here (ferric-gw/src/sigma.rs):

  * run_evgw0: eps_qp starts at eps_mf; sweep k copies the previous sweep's
    eps_qp into the propagator energies eps_prop of the QP window, then solves
    every state's QP equation  E = eps_m + Delta_m + Sigma^c_m(E; eps_prop)
    with W, the mean-field eps_m, Delta_m and eps_F all fixed (Jacobi update);
    stops when max |eps_qp,new - eps_qp| < ev_conv_thresh.
  * run_evgw: the same loop, and on sweeps it > 0 W is rebuilt by PDEP-RPA
    from a reference whose orbital energies are the current eps_qp, so the
    chi0 denominators Delta_i^a become QP gaps.

Model: one occupied state i and one virtual a, one auxiliary function, pair
densities B_ii, B_ia = b, B_aa. W~_d is then the one-pair single pole of
gw_self_energy_sympy.py claim 1 (Omega^2 = e(e + 4b^2), strength
s = 2 b^2 e / Omega), and Sigma^c_m is claim 2's residue form
  Sigma^c_m(E) = sum_n B_mn^2 s / (E - E_n + Omega)   (n occupied)
               + sum_n B_mn^2 s / (E - E_n - Omega)   (n virtual),
with E_n the propagator energies. The QP equation is solved by Newton on the
closed form (the continuation is claim 6 of gw_self_energy_sympy.py; this
script checks the outer update rule only).

Checked:
  EV1  Sweep 0 of the evGW0 and evGW loops reproduces the G0W0 roots exactly
       (eps_qp^(-1) = eps_mf, and run_evgw rebuilds W only for it > 0).
  EV2  evGW0 fixed point. At self-consistency the self-term of each state is
       B_mm^2 s / (+-Omega), and the QP gap g = E_a - E_i solves
       (g - e0)(g + Omega) = 2 B_ia^2 s,  e0 = eps_a + D_a - eps_i - D_i
       - (B_ii^2 + B_aa^2) s / Omega   (symbolic), whose positive root is
       g = [e0 - Omega + sqrt((e0 + Omega)^2 + 8 B_ia^2 s)] / 2. The
       transcribed loop converges to it (both energies, 1e-12); the G0W0 roots
       are not the fixed point (negative control).
  EV3  evGW. With QP energies in chi0 the one-pair pole moves to
       Omega'^2 = e'(e' + 4b^2), e' = E_a - E_i (symbolic, claim 1 with e -> e').
       The transcribed loop converges to the root of the two fixed-point
       equations with Omega(g) and s(g), found independently by mpmath
       findroot (1e-12); the evGW0 fixed point is not it (negative control).

Mutations (the runner substitutes the code line once; each exits 1):
  evGW0 propagator energies stay at eps_mf (G0W0 every sweep) -> EV2 fails
  eps_qp starts away from eps_mf                              -> EV1 fails
  evGW rebuilds W from the mean-field gap                     -> EV3 fails

Run: python3 scripts/gw_evgw_sympy.py
"""

import sys

import sympy as sp
from mpmath import findroot, mp, mpf, sqrt

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


mp.dps = 30
OCC, VIR = 0, 1
EPS_MF = [mpf("-0.5"), mpf("0.3")]
SHIFT = [mpf(0), mpf(0)]  # HF reference: Delta = 0
B = [[mpf("0.6"), mpf("0.3")], [mpf("0.3"), mpf("0.5")]]
EF = (EPS_MF[OCC] + EPS_MF[VIR]) / 2  # fermi_level: mean-field mid-gap, fixed
EV_CONV_THRESH = mpf(10) ** -20


def one_pair_pole(gap):
    """Claim 1 of gw_self_energy_sympy.py: W~_d = -2 Omega s / (u^2 + Omega^2)."""
    b2 = B[OCC][VIR] ** 2
    om = sqrt(gap * (gap + 4 * b2))
    return om, 2 * b2 * gap / om


def sigma_c(m, e, eps_prop, om, s):
    """Claim 2 residue form; occupation by the side of the fixed eps_F."""
    tot = mpf(0)
    for n in (OCC, VIR):
        w = B[m][n] ** 2 * s
        tot += (
            w / (e - eps_prop[n] + om)
            if eps_prop[n] < EF
            else w / (e - eps_prop[n] - om)
        )
    return tot


def solve_qp(m, eps_prop, om, s):
    """solve_qp_for_mo: Newton on E - eps_m - Delta_m - Sigma^c_m(E) from eps_m."""
    e = EPS_MF[m]
    for _ in range(100):
        f = e - EPS_MF[m] - SHIFT[m] - sigma_c(m, e, eps_prop, om, s)
        h = mpf(10) ** -12
        fp = 1 - (
            sigma_c(m, e + h, eps_prop, om, s) - sigma_c(m, e - h, eps_prop, om, s)
        ) / (2 * h)
        step = f / fp
        e -= step
        if abs(step) < mpf(10) ** -25:
            break
    return e


def g0w0():
    om, s = one_pair_pole(EPS_MF[VIR] - EPS_MF[OCC])
    return [solve_qp(m, EPS_MF, om, s) for m in (OCC, VIR)]


def run_evgw0(max_ev_iter):
    om, s = one_pair_pole(EPS_MF[VIR] - EPS_MF[OCC])  # W frozen
    eps_qp = [EPS_MF[m] for m in (OCC, VIR)]
    sweeps = []
    for it in range(max_ev_iter):
        eps_prop = [eps_qp[m] for m in (OCC, VIR)]
        new = [solve_qp(m, eps_prop, om, s) for m in (OCC, VIR)]
        dev = max(abs(new[m] - eps_qp[m]) for m in (OCC, VIR))
        eps_qp = new
        sweeps.append(list(eps_qp))
        if dev < EV_CONV_THRESH:
            break
    return sweeps


def run_evgw(max_ev_iter):
    om, s = one_pair_pole(EPS_MF[VIR] - EPS_MF[OCC])
    eps_qp = list(EPS_MF)
    sweeps = []
    for it in range(max_ev_iter):
        if it > 0:
            om, s = one_pair_pole(eps_qp[VIR] - eps_qp[OCC])
        eps_prop = list(eps_qp)
        new = [solve_qp(m, eps_prop, om, s) for m in (OCC, VIR)]
        dev = max(abs(new[m] - eps_qp[m]) for m in (OCC, VIR))
        eps_qp = new
        sweeps.append(list(eps_qp))
        if dev < EV_CONV_THRESH:
            break
    return sweeps


def part_symbolic():
    ei, ea, di, da, om, s, bii, bia, baa = sp.symbols(
        "epsilon_i epsilon_a Delta_i Delta_a Omega s B_ii B_ia B_aa", real=True
    )
    Ei, g = sp.symbols("E_i g", real=True)
    Ea = Ei + g
    # Fixed point: propagator energy of each state equals its own root.
    eq_i = Ei - (ei + di + bii**2 * s / (Ei - Ei + om) + bia**2 * s / (Ei - Ea - om))
    eq_a = Ea - (ea + da + bia**2 * s / (Ea - Ei + om) + baa**2 * s / (Ea - Ea - om))
    e0 = ea + da - ei - di - (bii**2 + baa**2) * s / om
    quad = (g - e0) * (g + om) - 2 * bia**2 * s
    ok = sp.simplify((eq_a - eq_i) * (g + om) - quad) == 0
    check(
        "EV2 gap at the evGW0 fixed point: (g - e0)(g + Omega) = 2 B_ia^2 s (claim EV2)",
        ok,
    )

    e, b, u = sp.symbols("e b u", positive=True)
    pi = 4 * b**2 * e / (u**2 + e**2)
    om2 = e * (e + 4 * b**2)
    sp_ = 2 * b**2 * e / sp.sqrt(om2)
    ok = sp.simplify(1 / (1 + pi) - 1 + 2 * sp.sqrt(om2) * sp_ / (u**2 + om2)) == 0
    check(
        "EV3 chi0 with QP gap e': W~_d pole Omega'^2 = e'(e' + 4b^2) (claim EV3)",
        ok,
    )


def part_numeric():
    ref = g0w0()
    s0 = run_evgw0(1)[0]
    s1 = run_evgw(1)[0]
    d = max(abs(s0[m] - ref[m]) for m in (OCC, VIR))
    d1 = max(abs(s1[m] - ref[m]) for m in (OCC, VIR))
    check(
        "EV1 sweep 0 of evGW0 and evGW = G0W0 (claim EV1)",
        d < mpf(10) ** -25 and d1 < mpf(10) ** -25,
        f"{float(d):.1e}, {float(d1):.1e}",
    )

    # EV2: closed form vs the transcribed loop.
    om, s = one_pair_pole(EPS_MF[VIR] - EPS_MF[OCC])
    bii2, bia2, baa2 = B[OCC][OCC] ** 2, B[OCC][VIR] ** 2, B[VIR][VIR] ** 2
    e0 = EPS_MF[VIR] + SHIFT[VIR] - EPS_MF[OCC] - SHIFT[OCC] - (bii2 + baa2) * s / om
    g = (e0 - om + sqrt((e0 + om) ** 2 + 8 * bia2 * s)) / 2
    e_i = EPS_MF[OCC] + SHIFT[OCC] + bii2 * s / om + bia2 * s / (-g - om)
    e_a = e_i + g
    sweeps = run_evgw0(200)
    fin = sweeps[-1]
    err = max(abs(fin[OCC] - e_i), abs(fin[VIR] - e_a))
    check(
        "EV2 evGW0 loop converges to the closed-form fixed point (claim EV2)",
        err < mpf(10) ** -12,
        f"{len(sweeps)} sweeps, |d| = {float(err):.1e}, gap {float(g):.6f} Ha",
    )
    ctrl = max(abs(ref[OCC] - e_i), abs(ref[VIR] - e_a))
    check(
        "EV2 negative control: the G0W0 roots are not the evGW0 fixed point",
        ctrl > mpf(10) ** -3,
        f"|d| = {float(ctrl):.1e}",
    )

    # EV3: independent root of the evGW fixed-point equations.
    def residual(ei, ea):
        omg, sg = one_pair_pole(ea - ei)
        props = [ei, ea]
        return [
            ei - EPS_MF[OCC] - SHIFT[OCC] - sigma_c(OCC, ei, props, omg, sg),
            ea - EPS_MF[VIR] - SHIFT[VIR] - sigma_c(VIR, ea, props, omg, sg),
        ]

    root = findroot(residual, (e_i, e_a))
    sweeps = run_evgw(200)
    fin = sweeps[-1]
    err = max(abs(fin[OCC] - root[0]), abs(fin[VIR] - root[1]))
    check(
        "EV3 evGW loop converges to the root of the fixed-point equations (claim EV3)",
        err < mpf(10) ** -12,
        f"{len(sweeps)} sweeps, |d| = {float(err):.1e}",
    )
    ctrl = max(abs(root[0] - e_i), abs(root[1] - e_a))
    check(
        "EV3 negative control: the evGW0 fixed point is not the evGW one",
        ctrl > mpf(10) ** -4,
        f"|d| = {float(ctrl):.1e}",
    )


def main():
    part_symbolic()
    part_numeric()
    print(f"{len(FAILS)} failed: {FAILS}" if FAILS else "all identities hold")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
