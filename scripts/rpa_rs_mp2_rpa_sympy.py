#!/usr/bin/env python3
"""Sympy/numpy checks of the SR-MP2 + LR-RPA combinations of rs_mp2_rpa.rs.

Theory and notation: `book/src/26-rs-mp2-rpa/01-cross-term.md#cross-term-split-energy`. Every check prints "FAIL [<name>]" when it fails, and the script
exits nonzero if any did.

Code (crates/ferric-rpa/src/rs_mp2_rpa.rs rs_mp2_lr_rpa,
crates/ferric-mp2/src/rimp2.rs pair_energy):
  E_OS  = sum_ijab (ia|jb)^2 / D,  E_SS = sum_ijab (ia|jb)[(ia|jb) - (ib|ja)] / D,
  D = e_i + e_j - e_a - e_b,  E_MP2 = E_OS + E_SS,
  B (DeltaLr, default):  E = E_MP2[C] + E_dRPA[lr] - 2 E_OS[lr]
  T (CoupledRings):      E = E_MP2[C] + (E_dRPA[C] - 2 E_OS[C]) - (E_dRPA[sr] - 2 E_OS[sr])
with (ia|jb)_C = (ia|jb)_sr + (ia|jb)_lr (erf + erfc = 1).

  1  One-mode ring model (sympy): E_dRPA(b) = 1/2 (sqrt(D (D + 2b)) - D - b)
     = -b^2/(4D) + b^3/(4D^2) + O(b^4). With b = b_s + b_l the B form is
     exact through second order, its third-order term is b_l^3/(4D^2), the
     T form's is (b_C^3 - b_s^3)/(4D^2), and T - B = 3 b_s b_l (b_s + b_l)/(4D^2)
     at third order: the mixed SR x LR rings B drops (the residual quoted in
     the RsMp2RpaFormulation doc).
  2  Ring expansion of drCCD (numpy, many modes): with B = 2K, the Riccati
     amplitudes T = sum_n lambda^n T_n give E^(2) = 1/2 sum B o T_1 = 2 E_OS
     exactly (T_1 = -B/Dpair), i.e. the second-order term of dRPA is twice
     the opposite-spin MP2 energy as rimp2.rs pair_energy computes it.
     Independent route: the plasmon energy of lambda K, Taylor coefficients
     from the Cauchy integral on a circle in complex lambda.
  3  No double counting: on a model with K_C = K_s + K_l, the lambda^2
     coefficient of both combinations equals E_MP2[C] (exchange included),
     and the lambda^3 coefficients are E^(3)[lr] (B) and
     E^(3)[C] - E^(3)[sr] (T), from the exact ring recursion.
  4  E_OS + E_SS equals the spin-orbital MP2 1/4 sum |<ij||ab>|^2/D
     (independent route for the pair_energy transcription).
  5  Limits: K_l = 0 (omega -> 0) gives E_MP2[C] for both forms; K_s = 0
     (omega -> inf) gives E_MP2[C] + E_dRPA[C] - 2 E_OS[C] for both.

Mutations (each turns the named check red):
  - b_form subtracts E_OS[lr] once instead of twice           [b-form-second-order]
  - pair_energy opposite-spin term weighted 2                 [ring2-equals-2eos]
  - pair_energy exchange sign flipped, (ia|jb) + (ib|ja)      [mp2-vs-spin-orbital]
  - t_form drops the 2 E_OS[sr] it adds back                  [t-form-second-order]

Run: python3 scripts/rpa_rs_mp2_rpa_sympy.py  (needs sympy, numpy)
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


# ------------------------------------------------------------ 1. one mode
def one_mode():
    D, b, bs, bl, lam = sp.symbols("D b b_s b_l lambda", positive=True)

    def e_rpa(x):
        return sp.Rational(1, 2) * (sp.sqrt(D * (D + 2 * x)) - D - x)

    def ring(x, n):
        return sp.series(e_rpa(lam * x), lam, 0, n + 1).removeO().coeff(lam, n)

    def e2(x):
        return ring(x, 2)

    check(
        "one-mode-series",
        sp.simplify(ring(b, 2) + b**2 / (4 * D)) == 0
        and sp.simplify(ring(b, 3) - b**3 / (4 * D**2)) == 0,
        "-b^2/(4D) + b^3/(4D^2)",
    )
    bc = bs + bl
    # The ring model carries the direct channel only, so its MP2 is the
    # second-order ring term lambda^2 e2(b_C).
    e_b = lam**2 * e2(bc) + (e_rpa(lam * bl) - e2(bl) * lam**2)
    e_t = (
        lam**2 * e2(bc)
        + (e_rpa(lam * bc) - e2(bc) * lam**2)
        - (e_rpa(lam * bs) - e2(bs) * lam**2)
    )
    sb = sp.series(e_b, lam, 0, 4).removeO()
    st = sp.series(e_t, lam, 0, 4).removeO()
    check(
        "one-mode-b-second",
        sp.simplify(sb.coeff(lam, 2) - e2(bc)) == 0
        and sp.simplify(st.coeff(lam, 2) - e2(bc)) == 0,
        "B and T: lambda^2 term is E_MP2[C]",
    )
    third_b = sp.simplify(sb.coeff(lam, 3))
    third_t = sp.simplify(st.coeff(lam, 3))
    check(
        "one-mode-b-third", sp.simplify(third_b - bl**3 / (4 * D**2)) == 0, f"{third_b}"
    )
    check(
        "one-mode-t-third",
        sp.simplify(third_t - (bc**3 - bs**3) / (4 * D**2)) == 0,
        f"{sp.factor(third_t)}",
    )
    check(
        "one-mode-t-minus-b",
        sp.simplify(third_t - third_b - 3 * bs * bl * (bs + bl) / (4 * D**2)) == 0,
        "3 b_s b_l (b_s + b_l)/(4 D^2)",
    )


# ----------------------------------------------------------------- models
rng = np.random.default_rng(7)
no, nv = 3, 4
eo = np.sort(rng.uniform(-1.6, -0.6, no))
ev = np.sort(rng.uniform(0.4, 2.0, nv))
e_ov = (ev[None, :] - eo[:, None]).ravel()
dpair = e_ov[:, None] + e_ov[None, :]  # positive, = -D


def ri_factor(naux, scale):
    return scale * rng.standard_normal((naux, no * nv))


def spin_components(bt):
    """rimp2.rs pair_energy over all (i, j), (ia|jb) = sum_P B_P,ia B_P,jb."""
    g = (bt.T @ bt).reshape(no, nv, no, nv)
    eos = 0.0
    ess = 0.0
    for i in range(no):
        for j in range(no):
            for a in range(nv):
                for b in range(nv):
                    g_ab = g[i, a, j, b]
                    g_ba = g[i, b, j, a]
                    denom = eo[i] + eo[j] - ev[a] - ev[b]
                    eos += g_ab * g_ab / denom
                    ess += g_ab * (g_ab - g_ba) / denom
    return eos, eos + ess


def spin_orbital_mp2(bt):
    """Independent MP2: 1/4 sum |<ij||ab>|^2 / D over spin orbitals."""
    g = (bt.T @ bt).reshape(no, nv, no, nv)  # (ia|jb)
    so_o = [(i, s) for i in range(no) for s in (0, 1)]
    so_v = [(a, s) for a in range(nv) for s in (0, 1)]
    e = 0.0
    for i, si in so_o:
        for j, sj in so_o:
            for a, sa in so_v:
                for b, sb in so_v:
                    direct = g[i, a, j, b] if (si == sa and sj == sb) else 0.0
                    exch = g[i, b, j, a] if (si == sb and sj == sa) else 0.0
                    e += 0.25 * (direct - exch) ** 2 / (eo[i] + eo[j] - ev[a] - ev[b])
    return e


def plasmon(k):
    b2 = 2.0 * k
    a = np.diag(e_ov) + b2
    w2 = np.linalg.eigvals((a - b2) @ (a + b2))
    if np.isrealobj(k):
        w2 = w2.real
    return 0.5 * (np.sum(np.sqrt(w2)) - np.trace(a))


def ring_terms(k, nmax=4):
    """Exact ring orders from the drCCD Riccati recursion, B = 2K canonical:
    Dpair o T_n = -(B T_{n-1} + T_{n-1} B + sum_{p+q=n-1} T_p B T_q), T_0 = 0."""
    b2 = 2.0 * k
    ts = {0: np.zeros_like(b2)}
    out = {}
    for n in range(1, nmax):
        rhs = b2 if n == 1 else b2 @ ts[n - 1] + ts[n - 1] @ b2
        for p in range(1, n - 1):
            q = n - 1 - p
            if q >= 1:
                rhs = rhs + ts[p] @ b2 @ ts[q]
        ts[n] = -rhs / dpair
        out[n + 1] = 0.5 * np.sum(b2 * ts[n])
    return out  # {2: E^(2), 3: E^(3)}


def b_form(e_mp2_c, e_rpa_l, eos_l):
    return e_mp2_c + e_rpa_l - 2.0 * eos_l


def t_form(e_mp2_c, e_rpa_c, eos_c, e_rpa_s, eos_s):
    return e_mp2_c + (e_rpa_c - 2.0 * eos_c) - (e_rpa_s - 2.0 * eos_s)


def lam_coeffs(fun, n=3, radius=0.2, npts=64):
    """Taylor coefficients 2..n of the analytic fun(lambda) by the Cauchy
    integral on |lambda| = radius (trapezoid rule, geometric convergence)."""
    th = 2 * np.pi * np.arange(npts) / npts
    z = radius * np.exp(1j * th)
    vals = np.array([fun(x) for x in z])
    return [
        float(np.real(np.mean(vals * np.exp(-1j * m * th)) / radius**m))
        for m in range(2, n + 1)
    ]


def many_mode():
    bs = ri_factor(6, 0.15)
    bl = ri_factor(5, 0.1)
    bc = np.vstack([bs, bl])
    ks, kl, kc = bs.T @ bs, bl.T @ bl, bc.T @ bc
    eos_c, mp2_c = spin_components(bc)
    eos_s, _ = spin_components(bs)
    eos_l, _ = spin_components(bl)
    e_so = spin_orbital_mp2(bc)
    check(
        "mp2-vs-spin-orbital",
        abs(mp2_c - e_so) < 1e-13 * abs(e_so),
        f"E_OS + E_SS {mp2_c:.15f} vs {e_so:.15f}",
    )
    rc = ring_terms(kc)
    check(
        "ring2-equals-2eos",
        abs(rc[2] - 2 * eos_c) < 1e-13 * abs(eos_c),
        f"E2 {rc[2]:.15f} vs 2 E_OS {2 * eos_c:.15f}",
    )
    c2, c3 = lam_coeffs(lambda x: plasmon(x * kc))
    check(
        "ring-orders-vs-plasmon",
        abs(c2 - rc[2]) < 1e-12 * abs(rc[2]) and abs(c3 - rc[3]) < 1e-11 * abs(rc[3]),
        f"Cauchy {c2:.14f},{c3:.14f} vs {rc[2]:.14f},{rc[3]:.14f}",
    )

    # 3. Scale every integral by lambda: MP2 pieces scale as lambda^2.
    def e_b(x):
        return b_form(x * x * mp2_c, plasmon(x * kl), x * x * eos_l)

    def e_t(x):
        return t_form(
            x * x * mp2_c,
            plasmon(x * kc),
            x * x * eos_c,
            plasmon(x * ks),
            x * x * eos_s,
        )

    b2c, b3c = lam_coeffs(e_b)
    t2c, t3c = lam_coeffs(e_t)
    rl, rs = ring_terms(kl), ring_terms(ks)
    check(
        "b-form-second-order",
        abs(b2c - mp2_c) < 1e-12 * abs(mp2_c),
        f"lambda^2 {b2c:.10f} vs E_MP2[C] {mp2_c:.10f}",
    )
    check(
        "t-form-second-order",
        abs(t2c - mp2_c) < 1e-12 * abs(mp2_c),
        f"lambda^2 {t2c:.10f} vs E_MP2[C] {mp2_c:.10f}",
    )
    check(
        "b-form-third-order",
        abs(b3c - rl[3]) < 1e-11 * abs(rl[3]),
        f"{b3c:.8f} vs E3[lr] {rl[3]:.8f}",
    )
    check(
        "t-form-third-order",
        abs(t3c - (rc[3] - rs[3])) < 1e-11 * abs(rc[3]),
        f"{t3c:.8f} vs E3[C]-E3[sr] {rc[3] - rs[3]:.8f}",
    )
    mixed = rc[3] - rs[3] - rl[3]
    check(
        "mixed-rings-nonzero",
        abs(mixed) > 1e-6,
        f"T - B at third order = {mixed:.3e} (the SR x LR rings)",
    )

    # 5. Limits.
    z = np.zeros((0, no * nv))
    eos0, _ = spin_components(z)
    lim0_b = b_form(mp2_c, plasmon(0 * kc), eos0)
    lim0_t = t_form(mp2_c, plasmon(kc), eos_c, plasmon(kc), eos_c)
    check(
        "omega-to-zero",
        abs(lim0_b - mp2_c) < 1e-14 and abs(lim0_t - mp2_c) < 1e-14,
        "both -> E_MP2[C]",
    )
    full = mp2_c + plasmon(kc) - 2 * eos_c
    liminf_b = b_form(mp2_c, plasmon(kc), eos_c)
    liminf_t = t_form(mp2_c, plasmon(kc), eos_c, plasmon(0 * kc), eos0)
    check(
        "omega-to-infinity",
        abs(liminf_b - full) < 1e-14 and abs(liminf_t - full) < 1e-14,
        "both -> MP2 + dRPA - 2 E_OS",
    )


one_mode()
many_mode()

if FAILED:
    print(f"\n{len(FAILED)} check(s) failed: {FAILED}")
    sys.exit(1)
print("\nall SR-MP2 + LR-RPA identities hold")
