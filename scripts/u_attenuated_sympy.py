#!/usr/bin/env python3
"""Sympy/numpy checks of the open-shell attenuated, SCS and SR-MP2 + LR-RPA energies.

Theory: crates/ferric-mp2/theory.md, section "Open-shell attenuated and
spin-scaled MP2", and crates/ferric-rpa/theory.md, section "Open-shell SR-MP2
+ LR-RPA". Every check prints "FAIL [<name>]" when it fails, and the script
exits nonzero if any did.

Code:
  crates/ferric-mp2/src/u_rimp2.rs  ri_mp2_spin_components_any, spin_components:
      E_aa = 1/2 sum_{ijab in a} (ia|jb)[(ia|jb) - (ib|ja)] / D,  E_bb likewise,
      E_ab = sum_{i a in alpha, J B in beta} (ia|JB)^2 / D,
      E_OS = E_ab,  E_SS = E_aa + E_bb    (what att-rimp2 / scs-mp2 read)
  crates/ferric-rpa/src/rs_mp2_rpa.rs  scan_impl_unrestricted, Rings::assemble:
      dMP2 = E_ab + 1/2 (D_aa + D_bb),  D_ss = sum_{ijab in s} (ia|jb)^2 / D
      B: E = E_MP2[C] + E_dRPA[lr] - dMP2[lr]
      T: E = E_MP2[C] + (E_dRPA[C] - dMP2[C]) - (E_dRPA[sr] - dMP2[sr])
  crates/ferric-rpa/src/lib.rs  u_pdep_rpa_from_intermediates: spin-summed
      eps(iw) = I + Pi_a + Pi_b,  Pi_s = B_s diag(2 d / (w^2 + d^2)) B_s^T
with D = e_i + e_j - e_a - e_b < 0 and d = e_a - e_i > 0.

Checks:
  1  freq-integral (sympy): int_0^inf d1 d2 / ((w^2+d1^2)(w^2+d2^2)) dw
     = pi / (2 (d1 + d2)). With it, the second-order term of the spin-summed
     dRPA is -1/(4 pi) int tr(Pi^2) dw = 1/2 sum_{s,t} sum (ia|jb)^2 / D
     = E_ab + 1/2 (D_aa + D_bb) (sympy, symbolic spin blocks).
  2  u-mp2-vs-spin-orbital (numpy): E_aa + E_bb + E_ab equals the
     spin-orbital MP2 1/4 sum |<ij||ab>|^2 / D on an open-shell model with
     different alpha and beta orbitals and occupation counts (independent
     route: the spin-orbital sum, no spin blocks).
     os-is-ab / ss-is-aa-plus-bb: the spin-orbital sum over pairs with
     spin(i) != spin(j) is E_ab, and over spin(i) == spin(j) is E_aa + E_bb.
  3  u-freq-vs-plasmon: the frequency form with prefactor 2 per spin equals
     the spin-orbital plasmon formula 1/2 (sum Omega - tr A).
     u-ring2-equals-dmp2: the lambda^2 Taylor coefficient of the plasmon
     energy of lambda K (Cauchy integral, independent route) equals dMP2.
     2eab-misses-open-shell: 2 E_ab, the closed-shell dMP2, differs from it
     on the open-shell model by far more than the check tolerance.
  4  singlet: with alpha == beta, E_ab = E_OS and E_aa + E_bb = E_SS of the
     closed-shell kernel, dMP2 = 2 E_OS, and the U-dRPA equals the
     closed-shell dRPA (plasmon with B = 2K).
  5  u-b-form-second-order / u-t-form-second-order: with every integral
     scaled by lambda and (ia|jb)_C = (ia|jb)_sr + (ia|jb)_lr in each spin,
     the lambda^2 coefficient of B and of T equals E_MP2[C] (open shell).
     u-limits: K_lr = 0 (omega -> 0) gives E_MP2[C] for both forms; K_sr = 0
     (omega -> inf) gives E_MP2[C] + E_dRPA[C] - dMP2[C] for both.

Mutations (each turns the named check red):
  - dMP2 taken as 2 E_ab (the closed-shell rule)     [u-ring2-equals-dmp2]
  - the beta channel dropped from Pi                 [u-freq-vs-plasmon]
  - same-spin block weighted 1 instead of 1/2        [u-mp2-vs-spin-orbital]
  - E_OS taken as E_ab + E_aa                        [os-is-ab]
  - T drops the dMP2[sr] it adds back                [u-t-form-second-order]

Run: python3 scripts/u_attenuated_sympy.py  (needs sympy, numpy)
"""

import sys

import numpy as np
import sympy as sp

FAILED = []


def check(name, ok, detail=""):
    if ok:
        print(f"ok   [{name}] {detail}")
    else:
        print(f"FAIL [{name}] {detail}")
        FAILED.append(name)


# ---------------------------------------------------------- 1. symbolic
def symbolic():
    w = sp.symbols("omega", real=True)
    d1, d2 = sp.symbols("d1 d2", positive=True)
    integral = sp.integrate(d1 * d2 / ((w**2 + d1**2) * (w**2 + d2**2)), (w, 0, sp.oo))
    check(
        "freq-integral",
        sp.simplify(integral - sp.pi / (2 * (d1 + d2))) == 0,
        f"{sp.simplify(integral)}",
    )
    # Second-order term of (1/2pi) int [ln det(1+Pi) - tr Pi]: -1/(4pi) int tr(Pi^2).
    # Pi = sum_s sum_{ia in s} 2 d_ia/(w^2 + d_ia^2) |ia><ia| in the aux space, so
    # tr(Pi^2) = sum_{s,t} sum_{ia in s, jb in t} 4 d d' (ia|jb)^2 / (..)(..).
    # Symbolic spin blocks: one (ia) per spin, couplings g_st = (ia_s|jb_t).
    da, db = sp.symbols("d_a d_b", positive=True)
    gaa, gbb, gab = sp.symbols("g_aa g_bb g_ab", real=True)
    pairs = [(da, da, gaa), (db, db, gbb), (da, db, gab), (db, da, gab)]
    e2 = 0
    for x, y, g in pairs:
        e2 += -sp.Rational(1, 4) / sp.pi * 4 * g**2 * (sp.pi / (2 * (x + y)))
    # D = -(d + d'): D_aa = gaa^2/(-2 da), D_bb likewise, E_ab = gab^2/(-(da+db)).
    d_aa = gaa**2 / (-(da + da))
    d_bb = gbb**2 / (-(db + db))
    e_ab = gab**2 / (-(da + db))
    check(
        "ring2-spin-blocks",
        sp.simplify(e2 - (e_ab + sp.Rational(1, 2) * (d_aa + d_bb))) == 0,
        "E2 = E_ab + (D_aa + D_bb)/2",
    )


# ------------------------------------------------------------ the model
rng = np.random.default_rng(20261115)
NO_A, NV_A, NO_B, NV_B, NAUX_S, NAUX_L = 3, 3, 2, 4, 6, 5
EO_A = np.sort(rng.uniform(-1.4, -0.4, NO_A))
EV_A = np.sort(rng.uniform(0.2, 1.3, NV_A))
EO_B = np.sort(rng.uniform(-1.3, -0.5, NO_B))
EV_B = np.sort(rng.uniform(0.1, 1.2, NV_B))


def spin(no, nv, eo, ev):
    return {
        "no": no,
        "nv": nv,
        "eo": eo,
        "ev": ev,
        "d": (ev[None, :] - eo[:, None]).ravel(),
    }


SA = spin(NO_A, NV_A, EO_A, EV_A)
SB = spin(NO_B, NV_B, EO_B, EV_B)


def b_pair(scale_s, scale_l):
    """(B_sr, B_lr) per spin on ONE shared aux space; C = [sr; lr] stacked."""
    out = []
    for sp_ in (SA, SB):
        n = sp_["no"] * sp_["nv"]
        out.append(
            (
                scale_s * rng.standard_normal((NAUX_S, n)),
                scale_l * rng.standard_normal((NAUX_L, n)),
            )
        )
    return out


def same_spin(b, s):
    """u_rimp2 same-spin block: 1/2 sum (ia|jb)[(ia|jb) - (ib|ja)]/D, and D_ss."""
    no, nv, eo, ev = s["no"], s["nv"], s["eo"], s["ev"]
    g = (b.T @ b).reshape(no, nv, no, nv)
    e = 0.0
    direct = 0.0
    for i in range(no):
        for j in range(no):
            for a in range(nv):
                for c in range(nv):
                    den = eo[i] + eo[j] - ev[a] - ev[c]
                    e += g[i, a, j, c] * (g[i, a, j, c] - g[i, c, j, a]) / den
                    direct += g[i, a, j, c] ** 2 / den
    return 0.5 * e, direct  # same-spin block


def opposite_spin(ba, bb):
    g = (ba.T @ bb).reshape(SA["no"], SA["nv"], SB["no"], SB["nv"])
    e = 0.0
    for i in range(SA["no"]):
        for j in range(SB["no"]):
            for a in range(SA["nv"]):
                for c in range(SB["nv"]):
                    e += g[i, a, j, c] ** 2 / (
                        SA["eo"][i] + SB["eo"][j] - SA["ev"][a] - SB["ev"][c]
                    )
    return e


def u_blocks(ba, bb):
    """(E_aa, E_bb, E_ab, D_aa, D_bb) of u_rimp2 + the same-spin direct sums."""
    e_aa, d_aa = same_spin(ba, SA)
    e_bb, d_bb = same_spin(bb, SB)
    return e_aa, e_bb, opposite_spin(ba, bb), d_aa, d_bb


def scs_split(e_aa, e_bb, e_ab):
    """URiMp2Components::spin_components."""
    e_os = e_ab
    e_ss = e_aa + e_bb
    return e_os, e_ss


def dmp2_u(e_ab, d_aa, d_bb):
    """rs_mp2_rpa.rs scan_impl_unrestricted parts_of."""
    return e_ab + 0.5 * (d_aa + d_bb)


def spin_orbital_mp2(ba, bb):
    """Independent route: 1/4 sum |<ij||ab>|^2/D over spin orbitals, split by
    whether the two occupied spins agree."""
    orbs = []  # (spin, local index, energy, is_occ)
    for s, sp_ in ((0, SA), (1, SB)):
        orbs += [(s, i, sp_["eo"][i], True) for i in range(sp_["no"])]
        orbs += [(s, a, sp_["ev"][a], False) for a in range(sp_["nv"])]
    occ = [o for o in orbs if o[3]]
    vir = [o for o in orbs if not o[3]]
    bs = (ba, bb)
    nvs = (SA["nv"], SB["nv"])

    def eri(o, v, o2, v2):  # (ov|o2v2), zero unless spins conserve
        if o[0] != v[0] or o2[0] != v2[0]:
            return 0.0
        x = bs[o[0]][:, o[1] * nvs[o[0]] + v[1]]
        y = bs[o2[0]][:, o2[1] * nvs[o2[0]] + v2[1]]
        return float(x @ y)

    same = 0.0
    opp = 0.0
    for i in occ:
        for j in occ:
            for a in vir:
                for b in vir:
                    t = eri(i, a, j, b) - eri(i, b, j, a)
                    val = 0.25 * t * t / (i[2] + j[2] - a[2] - b[2])
                    if i[0] == j[0]:
                        same += val
                    else:
                        opp += val
    return same, opp


def pi_matrix(bs, w, scale_u=2.0):
    pi_a = bs[0] @ np.diag(scale_u * SA["d"] / (w * w + SA["d"] ** 2)) @ bs[0].T
    pi_b = bs[1] @ np.diag(scale_u * SB["d"] / (w * w + SB["d"] ** 2)) @ bs[1].T
    pi = pi_a + pi_b
    return pi


def u_drpa_freq(bs, nw=400, x0=0.5):
    """(1/2pi) int_0^inf [ln det(1 + Pi) - tr Pi] dw on a mapped Gauss-Legendre grid."""
    x, wt = np.polynomial.legendre.leggauss(nw)
    w = x0 * (1 + x) / (1 - x)
    wt = wt * 2 * x0 / (1 - x) ** 2
    e = 0.0
    for wk, ck in zip(w, wt):
        pi = pi_matrix(bs, wk)
        e += (
            ck
            / (2 * np.pi)
            * (np.linalg.slogdet(np.eye(pi.shape[0]) + pi)[1] - np.trace(pi))
        )
    return e


def k_spin_orbital(bs):
    """Direct-ring coupling over all spin-conserving excitations (alpha then beta)."""
    bt = np.hstack(bs)
    return bt.T @ bt


def plasmon(k, d):
    a = np.diag(d) + k
    w2 = np.linalg.eigvals((a - k) @ (a + k))
    if np.isrealobj(k):
        w2 = w2.real
    return 0.5 * (np.sum(np.sqrt(w2)) - np.trace(a))


D_SO = np.concatenate([SA["d"], SB["d"]])


def lam_coeffs(fun, n=3, radius=0.2, npts=64):
    th = 2 * np.pi * np.arange(npts) / npts
    z = radius * np.exp(1j * th)
    vals = np.array([fun(x) for x in z])
    return [
        float(np.real(np.mean(vals * np.exp(-1j * m * th)) / radius**m))
        for m in range(2, n + 1)
    ]


def b_form(e_mp2_c, e_rpa_l, dmp2_l):
    return e_mp2_c + e_rpa_l - dmp2_l


def t_form(e_mp2_c, e_rpa_c, dmp2_c, e_rpa_s, dmp2_s):
    return e_mp2_c + (e_rpa_c - dmp2_c) - (e_rpa_s - dmp2_s)


# --------------------------------------------------------- 2-3 open shell
def open_shell():
    (bsa, bla), (bsb, blb) = b_pair(0.15, 0.1)
    ba = np.vstack([bsa, bla])
    bb = np.vstack([bsb, blb])
    e_aa, e_bb, e_ab, d_aa, d_bb = u_blocks(ba, bb)
    same, opp = spin_orbital_mp2(ba, bb)
    e_so = same + opp
    e_u = e_aa + e_bb + e_ab
    check(
        "u-mp2-vs-spin-orbital",
        abs(e_u - e_so) < 1e-13 * abs(e_so),
        f"E_aa+E_bb+E_ab {e_u:.15f} vs {e_so:.15f}",
    )
    e_os, e_ss = scs_split(e_aa, e_bb, e_ab)
    check("os-is-ab", abs(e_os - opp) < 1e-13 * abs(opp), f"{e_os:.15f} vs {opp:.15f}")
    check(
        "ss-is-aa-plus-bb",
        abs(e_ss - same) < 1e-13 * abs(same),
        f"{e_ss:.15f} vs {same:.15f}",
    )

    k = k_spin_orbital((ba, bb))
    e_pl = plasmon(k, D_SO)
    e_fr = u_drpa_freq((ba, bb))
    check(
        "u-freq-vs-plasmon",
        abs(e_fr - e_pl) < 1e-10 * abs(e_pl),
        f"freq {e_fr:.14f} vs plasmon {e_pl:.14f}",
    )
    c2, _ = lam_coeffs(lambda x: plasmon(x * k, D_SO))
    dm = dmp2_u(e_ab, d_aa, d_bb)
    check(
        "u-ring2-equals-dmp2",
        abs(c2 - dm) < 1e-12 * abs(dm),
        f"lambda^2 coeff {c2:.15f} vs dMP2 {dm:.15f}",
    )
    miss = abs(2 * e_ab - c2)
    check(
        "2eab-misses-open-shell",
        miss > 1e3 * 1e-12 * abs(c2),
        f"|2 E_ab - E2| = {miss:.3e} (closed-shell dMP2 rule is wrong here)",
    )

    # 5. B and T forms, second order and limits, all open shell.
    def parts(sa, sb):
        x = u_blocks(sa, sb)
        return x[0] + x[1] + x[2], dmp2_u(x[2], x[3], x[4]), k_spin_orbital((sa, sb))

    mp2_c, dm_c, k_c = parts(ba, bb)
    _, dm_s, k_s = parts(bsa, bsb)
    _, dm_l, k_l = parts(bla, blb)

    def e_b(x):
        return b_form(x * x * mp2_c, plasmon(x * k_l, D_SO), x * x * dm_l)

    def e_t(x):
        return t_form(
            x * x * mp2_c,
            plasmon(x * k_c, D_SO),
            x * x * dm_c,
            plasmon(x * k_s, D_SO),
            x * x * dm_s,
        )

    cb2, _ = lam_coeffs(e_b)
    ct2, _ = lam_coeffs(e_t)
    check(
        "u-b-form-second-order",
        abs(cb2 - mp2_c) < 1e-12 * abs(mp2_c),
        f"{cb2:.15f} vs E_MP2[C] {mp2_c:.15f}",
    )
    check(
        "u-t-form-second-order",
        abs(ct2 - mp2_c) < 1e-12 * abs(mp2_c),
        f"{ct2:.15f} vs E_MP2[C] {mp2_c:.15f}",
    )
    zero = 0 * k_c
    full = mp2_c + plasmon(k_c, D_SO) - dm_c
    lim0_b = b_form(mp2_c, plasmon(zero, D_SO), 0.0)
    lim0_t = t_form(mp2_c, plasmon(k_c, D_SO), dm_c, plasmon(k_c, D_SO), dm_c)
    liminf_b = b_form(mp2_c, plasmon(k_c, D_SO), dm_c)
    liminf_t = t_form(mp2_c, plasmon(k_c, D_SO), dm_c, plasmon(zero, D_SO), 0.0)
    check(
        "u-limits",
        max(
            abs(lim0_b - mp2_c),
            abs(lim0_t - mp2_c),
            abs(liminf_b - full),
            abs(liminf_t - full),
        )
        < 1e-14,
        "omega->0: E_MP2[C]; omega->inf: E_MP2[C] + E_dRPA[C] - dMP2[C]",
    )


# --------------------------------------------------------------- 4 singlet
def singlet():
    """alpha == beta: same orbitals, energies and B for both spins."""
    global SA, SB, D_SO
    keep = (SA, SB, D_SO)
    SB = SA
    D_SO = np.concatenate([SA["d"], SA["d"]])
    b = 0.15 * rng.standard_normal((NAUX_S + NAUX_L, SA["no"] * SA["nv"]))
    e_aa, e_bb, e_ab, d_aa, d_bb = u_blocks(b, b)
    # closed-shell kernel (rimp2.rs pair_energy): E_OS = D, E_SS = D - X
    _, direct = same_spin(b, SA)
    e_os_cs = direct
    e_ss_cs = 2 * e_aa
    check(
        "singlet-scs-split",
        abs(e_ab - e_os_cs) < 1e-13 * abs(e_os_cs)
        and abs((e_aa + e_bb) - e_ss_cs) < 1e-13 * abs(e_ss_cs),
        f"E_ab {e_ab:.15f} = E_OS {e_os_cs:.15f}",
    )
    check(
        "singlet-dmp2",
        abs(dmp2_u(e_ab, d_aa, d_bb) - 2 * e_os_cs) < 1e-13 * abs(e_os_cs),
        "dMP2 = 2 E_OS",
    )
    k1 = b.T @ b
    e_cs = plasmon(2.0 * k1, SA["d"])
    e_u = plasmon(k_spin_orbital((b, b)), D_SO)
    check(
        "singlet-drpa",
        abs(e_u - e_cs) < 1e-12 * abs(e_cs),
        f"U {e_u:.15f} vs closed {e_cs:.15f}",
    )
    SA, SB, D_SO = keep


symbolic()
open_shell()
singlet()

if FAILED:
    print(f"\n{len(FAILED)} check(s) failed: {FAILED}")
    sys.exit(1)
print("\nall open-shell attenuated / SCS / SR-MP2 + LR-RPA identities hold")
