#!/usr/bin/env python3
"""Proofs of the attenuated (erfc) RI-MP2 nuclear-gradient identities ferric-mp2 codes.

Theory: `book/src/15-correlated-gradients/02-rimp2-gradient.md#rimp2-gradient-attenuated-erfc-ri-mp2`.
Code: crates/ferric-mp2/src/att_rimp2_gradient.rs (att_rimp2_gradient_analytical,
total_att_rimp2_gradient_on) over gradient.rs (relaxed_gradient_from_intermediates:
the four-centre term on bounds.op, the RI response on the correlation operator).

The energy is E = E_HF[1/r] + s E_c[erfc(w r)/r]: Coulomb RHF orbitals, and the
correlation's 3-centre integrals (P|mn) AND metric (P|Q) both under erfc.

  E  erfc derivative chain rule (sympy). For s-type Gaussian charge
     distributions (exponent p at P, prefactor K1; q at Q, K2),
       I_c   = A F0(rho R^2),                       A = 2 pi^(5/2) K1 K2 / (p q sqrt(p+q))
       I_erf = A sqrt(w^2/(w^2+rho)) F0(rho_w R^2),  rho = pq/(p+q), rho_w = rho w^2/(rho+w^2)
       I_erfc = I_c - I_erf,
     E1 the closed form equals an independent route, a quadrature of the
        Gaussian-kernel integral over t in erfc(w r)/r = (2/sqrt pi) int_w^inf
        exp(-t^2 r^2) dt (no Boys function, no rho_w);
     E2 sympy's derivative of the 3-centre (P|erfc|mn) and 2-centre (P|erfc|Q)
        closed forms, through F0' = -F1, equals a Richardson difference of the
        closed form, at separated AND coincident centres;
     E3 the centre derivatives of each sum to zero (translational invariance),
        symbolically.
  W  The w -> 0 limit (sympy). lim_{w->0} I_erfc = I_c, and the first order is a
     constant kernel: dI_erfc/dw at w = 0 equals -(2/sqrt pi) n1 n2 with
     n = K (pi/p)^(3/2) the charges, so (P|mn) moves by -(2w/sqrt pi) n_P S_mn
     (zero in (P|ia), S_ia = 0) and V by a rank-one term.
  L  The assembled gradient on a 1-D model with real integrals: three atoms on
     the z axis, two s Gaussians per atom (overlap S from the closed form,
     core Hamiltonian h(x) a smooth model), Coulomb four-centre ERIs for the
     RHF, erfc (P|mn) and (P|Q) on three aux s Gaussians per atom for the
     correlation; the parameter x is the z of the middle atom. The RHF is
     solved to 1e-13 at every x; the analytic Lagrangian (Coulomb in the
     Z-vector right-hand side's veff, the CPHF coupling, vhf_s1occ and the
     four-centre derivative term; erfc in the amplitudes, b_full, Imat and the
     3c/2c response with the E2 derivatives) equals the Richardson derivative
     of the re-solved E_HF + s E_c(erfc), for s = 1 and 1.2, with and without
     one frozen core.
  Z  At w = 1e-6 (L's model) the attenuated gradient equals the Coulomb RI-MP2
     gradient of the same model (u, V under 1/r) to 1e-9 relative, and at the
     production w = 0.2222 Bohr^-1 differs from it by far more.

Mutations (each turns the script red; the runner applies them one at a time
as exact substitutions of the tagged lines):
  * the erfc metric-derivative (2-centre) term dropped breaks L ("FAIL L").
  * the 3-centre derivative on the Coulomb operator instead of erfc breaks L
    ("FAIL L").
  * the Z-vector right-hand side without the veff of the correlation density
    breaks L ("FAIL L").
  * the four-centre derivative term on the erfc ERIs (the correlation operator
    where the reference's belongs) breaks L ("FAIL L").
  * the 2-centre closed form with rho_w -> rho in the erf part breaks E
    ("FAIL E").

Run: python3 scripts/att_rimp2_gradient_sympy.py  (needs sympy, numpy, scipy)
"""

import itertools
import math
import sys

import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.linalg import eigh
from scipy.special import erf

FAILED = []
RNG = np.random.default_rng(20261115)
OMEGA = 0.420 * 0.529177210903  # 0.420 A^-1 in Bohr^-1


def report(part, name, err, tol):
    ok = err <= tol
    print(f"  {'ok  ' if ok else 'FAIL ' + part} {name}: {err:.2e} (bar {tol:.0e})")
    if not ok:
        FAILED.append(f"{part}: {name}")


def richardson(f, x0, h):
    """4-point central difference, truncation O(h^4)."""
    d1 = (f(x0 + h) - f(x0 - h)) / (2 * h)
    d2 = (f(x0 + 2 * h) - f(x0 - 2 * h)) / (4 * h)
    return (4 * d1 - d2) / 3


# ================================================================ Boys functions
class F1(sp.Function):
    """Boys F1 (only its value is needed: the gradient is first order)."""


class F0(sp.Function):
    """Boys F0 with F0'(T) = -F1(T)."""

    def fdiff(self, argindex=1):
        return -F1(self.args[0])


def boys0(t):
    t = float(t)
    if t < 1e-6:
        return 1.0 - t / 3.0 + t * t / 10.0
    return 0.5 * math.sqrt(math.pi / t) * math.erf(math.sqrt(t))


def boys1(t):
    t = float(t)
    if t < 1e-3:
        return 1.0 / 3.0 - t / 5.0 + t * t / 14.0 - t**3 / 54.0
    return (boys0(t) - math.exp(-t)) / (2.0 * t)


MODULES = [{"F0": boys0, "F1": boys1}, "math"]

# ================================================================ closed forms
w = sp.Symbol("w", positive=True)
a, b, c, d, al, be = sp.symbols("a b c d alpha beta", positive=True)
za, zb, zc, zd, zp, zq = sp.symbols("z_a z_b z_c z_d z_p z_q", real=True)


def norm(e):
    return (2 * e / sp.pi) ** sp.Rational(3, 4)


def pair(e1, z1, e2, z2):
    """(exponent, centre, prefactor) of the product of two normalized s functions."""
    p = e1 + e2
    return (
        p,
        (e1 * z1 + e2 * z2) / p,
        norm(e1) * norm(e2) * sp.exp(-e1 * e2 / p * (z1 - z2) ** 2),
    )


def single(e, z):
    return e, z, norm(e)


RHO_W_SCALE = 1  # erf-part exponent: rho_w = rho w^2/(rho + w^2)


def dist_int(d1, d2, kernel):
    (p, zp_, k1), (q, zq_, k2) = d1, d2
    rho = p * q / (p + q)
    r2 = (zp_ - zq_) ** 2
    pref = 2 * sp.pi ** sp.Rational(5, 2) * k1 * k2 / (p * q * sp.sqrt(p + q))
    coul = pref * F0(rho * r2)
    if kernel == "coul":
        return coul
    rho_w = rho * w**2 / (rho + w**2) if RHO_W_SCALE else rho
    erf_part = pref * sp.sqrt(w**2 / (w**2 + rho)) * F0(rho_w * r2)
    if kernel == "erf":
        return erf_part
    return coul - erf_part


def eri3_expr(kernel):
    return dist_int(single(al, zp), pair(a, za, b, zb), kernel)


def eri2_expr(kernel):
    return dist_int(single(al, zp), single(be, zq), kernel)


def eri4_expr(kernel):
    return dist_int(pair(a, za, b, zb), pair(c, zc, d, zd), kernel)


def compile_int(expr, centres):
    """Value and per-centre derivatives, lambdified."""
    args = (a, b, c, d, al, be, za, zb, zc, zd, zp, zq, w)
    val = sp.lambdify(args, expr, modules=MODULES)
    ders = [sp.lambdify(args, sp.diff(expr, z), modules=MODULES) for z in centres]
    return val, ders


# ================================================================ part E
def part_e():
    print("E: erfc closed forms and their derivative chain rule")
    # E1: the 2-centre (P|erfc|Q) closed form vs a quadrature over t.
    i2 = sp.lambdify((al, be, zp, zq, w), eri2_expr("erfc"), modules=MODULES)
    worst = 0.0
    for alv, bev, r, wv in [
        (0.4, 1.2, 1.3, OMEGA),
        (2.0, 0.7, 0.0, 0.5),
        (1.1, 1.1, 3.0, 1.7),
    ]:
        n1, n2 = (2 * alv / np.pi) ** 0.75, (2 * bev / np.pi) ** 0.75
        rho = alv * bev / (alv + bev)

        def kern(t):
            g = (np.pi**2 / (alv * bev)) ** 1.5 * (rho / (rho + t * t)) ** 1.5
            return g * np.exp(-rho * t * t / (rho + t * t) * r * r)

        ref = (
            n1
            * n2
            * 2
            / np.sqrt(np.pi)
            * quad(kern, wv, np.inf, epsabs=1e-14, epsrel=1e-13)[0]
        )
        worst = max(worst, abs(i2(alv, bev, 0.0, r, wv) - ref) / abs(ref))
    report("E", "E1 (P|erfc|Q) closed form vs t-quadrature", worst, 1e-10)

    # E2: symbolic derivatives vs Richardson of the closed form.
    worst = 0.0
    base = dict(a=0.8, b=1.9, c=0.0, d=0.0, al=0.6, be=1.4, w=OMEGA)
    for kind, expr, cents in [
        ("3c", eri3_expr("erfc"), (zp, za, zb)),
        ("2c", eri2_expr("erfc"), (zp, zq)),
    ]:
        val, ders = compile_int(expr, cents)
        for geom in ([0.3, -0.4, 1.1], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]):
            z = dict(zip((zp, za, zb) if kind == "3c" else (zp, zq), geom))

            def call(f, shift=None, h=0.0):
                zz = {k: v for k, v in z.items()}
                if shift is not None:
                    zz[shift] += h
                return f(
                    base["a"],
                    base["b"],
                    0.0,
                    0.0,
                    base["al"],
                    base["be"],
                    zz.get(za, 0.0),
                    zz.get(zb, 0.0),
                    0.0,
                    0.0,
                    zz.get(zp, 0.0),
                    zz.get(zq, 0.0),
                    base["w"],
                )

            for cen, der in zip(cents, ders):
                fd = richardson(lambda h: call(val, cen, h), 0.0, 1e-3)
                worst = max(worst, abs(call(der) - fd))
    report("E", "E2 d(P|erfc|mn), d(P|erfc|Q) vs Richardson", worst, 1e-10)

    # E3: translational invariance, symbolically.
    s3 = sp.simplify(sum(sp.diff(eri3_expr("erfc"), z) for z in (zp, za, zb)))
    s2 = sp.simplify(sum(sp.diff(eri2_expr("erfc"), z) for z in (zp, zq)))
    report(
        "E",
        "E3 sum of centre derivatives (3c, 2c) == 0",
        float(s3 != 0) + float(s2 != 0),
        0,
    )


# ================================================================ part W
def part_w():
    print("W: the w -> 0 limit")
    for name, expr, charges in [
        ("2c", eri2_expr, (single(al, zp), single(be, zq))),
        ("3c", eri3_expr, (single(al, zp), pair(a, za, b, zb))),
    ]:
        e_erfc, e_coul = expr("erfc"), expr("coul")
        lim = sp.limit(expr("erf").subs(F0, sp.Lambda(sp.Symbol("t"), 1)), w, 0)
        # The erf part is the whole difference; at w -> 0 its F0 argument -> 0.
        report("W", f"lim_(w->0) I_erf ({name}) == 0", float(sp.simplify(lim) != 0), 0)
        n = [k * (sp.pi / p) ** sp.Rational(3, 2) for p, _, k in charges]
        # F0(T) = 1 - T/3 + O(T^2) and T = O(w^2), so to first order in w:
        series_expr = e_erfc.replace(
            F0, sp.Lambda(sp.Symbol("t"), 1 - sp.Symbol("t") / 3)
        )
        slope = sp.diff(series_expr, w).subs(w, 0)
        target = sp.diff(
            e_coul.replace(F0, sp.Lambda(sp.Symbol("t"), 1 - sp.Symbol("t") / 3)), w
        )
        resid = sp.simplify(slope - target + 2 / sp.sqrt(sp.pi) * n[0] * n[1])
        report(
            "W", f"dI_erfc/dw|_0 == -(2/sqrt pi) n1 n2 ({name})", float(resid != 0), 0
        )


# ================================================================ part L model
ATOM_Z0 = np.array([0.0, 1.4, 2.9])
MOVING = 1
AO_EXP = [0.6, 2.4]
AUX_EXP = [0.35, 1.1, 3.8]
NOCC = 3


def compiled():
    out = {}
    for kern in ("coul", "erfc"):
        out[("4", kern)] = compile_int(eri4_expr(kern), (za, zb, zc, zd))
        out[("3", kern)] = compile_int(eri3_expr(kern), (zp, za, zb))
        out[("2", kern)] = compile_int(eri2_expr(kern), (zp, zq))
    return out


INTS = None


def basis(x, exps):
    z = ATOM_Z0.copy()
    z[MOVING] += x
    return [(e, z[at], at) for at in range(len(z)) for e in exps]


def build(x, omega):
    """S, dS, Coulomb and erfc ERIs (+ d/dx), erfc u, V (+ d/dx), Coulomb u, V."""
    ao, aux = basis(x, AO_EXP), basis(x, AUX_EXP)
    n, m = len(ao), len(aux)
    s = np.zeros((n, n))
    ds = np.zeros((n, n))
    for i, j in itertools.product(range(n), range(n)):
        (e1, z1, a1), (e2, z2, a2) = ao[i], ao[j]
        p = e1 + e2
        val = (4 * e1 * e2 / p**2) ** 0.75 * math.exp(-e1 * e2 / p * (z1 - z2) ** 2)
        s[i, j] = val
        dr = (a1 == MOVING) - (a2 == MOVING)
        ds[i, j] = val * (-2 * e1 * e2 / p * (z1 - z2)) * dr
    out = {"s": s, "ds": ds}
    for kern in ("coul", "erfc"):
        v4, d4 = INTS[("4", kern)]
        eri = np.zeros((n, n, n, n))
        deri = np.zeros_like(eri)
        for i, j, k, l in itertools.product(range(n), repeat=4):
            f = (
                ao[i][0],
                ao[j][0],
                ao[k][0],
                ao[l][0],
                0.0,
                0.0,
                ao[i][1],
                ao[j][1],
                ao[k][1],
                ao[l][1],
                0.0,
                0.0,
                omega,
            )
            eri[i, j, k, l] = v4(*f)
            deri[i, j, k, l] = sum(
                dd(*f) for dd, idx in zip(d4, (i, j, k, l)) if ao[idx][2] == MOVING
            )
        out[f"eri_{kern}"], out[f"deri_{kern}"] = eri, deri
        v3, d3 = INTS[("3", kern)]
        u = np.zeros((m, n, n))
        du = np.zeros_like(u)
        for pp, i, j in itertools.product(range(m), range(n), range(n)):
            f = (
                ao[i][0],
                ao[j][0],
                0.0,
                0.0,
                aux[pp][0],
                0.0,
                ao[i][1],
                ao[j][1],
                0.0,
                0.0,
                aux[pp][1],
                0.0,
                omega,
            )
            u[pp, i, j] = v3(*f)
            du[pp, i, j] = sum(
                dd(*f)
                for dd, cen in zip(d3, (aux[pp], ao[i], ao[j]))
                if cen[2] == MOVING
            )
        v2, d2 = INTS[("2", kern)]
        v = np.zeros((m, m))
        dv = np.zeros_like(v)
        for pp, qq in itertools.product(range(m), range(m)):
            f = (
                0.0,
                0.0,
                0.0,
                0.0,
                aux[pp][0],
                aux[qq][0],
                0.0,
                0.0,
                0.0,
                0.0,
                aux[pp][1],
                aux[qq][1],
                omega,
            )
            v[pp, qq] = v2(*f)
            dv[pp, qq] = sum(
                dd(*f) for dd, cen in zip(d2, (aux[pp], aux[qq])) if cen[2] == MOVING
            )
        out[f"u_{kern}"], out[f"du_{kern}"] = u, du
        out[f"v_{kern}"], out[f"dv_{kern}"] = v, dv
    return out


H0 = None
H1 = None


def core(x):
    return H0 + x * H1


def jk(eri, dm):
    return np.einsum("mnls,ls->mn", eri, dm), np.einsum("mlns,ls->mn", eri, dm)


def rhf(s, h, eri, nocc, tol=1e-13, max_iter=300):
    c = eigh(h, s)[1]
    focks, errs = [], []
    for _ in range(max_iter):
        dm = 2 * c[:, :nocc] @ c[:, :nocc].T
        jm, km = jk(eri, dm)
        f = h + jm - 0.5 * km
        err = (f @ dm @ s - s @ dm @ f).ravel()
        if np.max(np.abs(err)) < tol:
            eps, c = eigh(f, s)
            return 0.5 * np.sum(dm * (h + f)), c, eps
        focks.append(f)
        errs.append(err)
        focks, errs = focks[-8:], errs[-8:]
        nd = len(errs)
        bm = -np.ones((nd + 1, nd + 1))
        bm[nd, nd] = 0
        bm[:nd, :nd] = np.array([[e1 @ e2 for e2 in errs] for e1 in errs])
        rhs = np.zeros(nd + 1)
        rhs[nd] = -1
        coef = np.linalg.lstsq(bm, rhs, rcond=None)[0][:nd]
        c = eigh(sum(cf * fk for cf, fk in zip(coef, focks)), s)[1]
    raise AssertionError("SCF did not converge")


def inv_sqrt(v):
    ev, q = np.linalg.eigh(v)
    return q @ np.diag(ev**-0.5) @ q.T


def mp2_pieces(c, eps, u, v, nocc, fc, scale):
    vm12 = inv_sqrt(v)
    b_full = np.einsum("PQ,Qmn,mp,nq->Ppq", vm12, u, c, c)
    ia = b_full[:, fc:nocc, nocc:]
    g = np.einsum("Pia,Pjb->ijab", ia, ia)
    eo, ev = eps[fc:nocc], eps[nocc:]
    dd = (
        eo[:, None, None, None]
        + eo[None, :, None, None]
        - ev[None, None, :, None]
        - ev[None, None, None, :]
    )
    t = g / dd
    tt = scale * (2 * t - t.transpose(0, 1, 3, 2))
    return np.sum(g * tt), t, tt, b_full, vm12


def total_energy(x, omega, fc, scale, corr="erfc"):
    m = build(x, omega)
    e_hf, c, eps = rhf(m["s"], core(x), m["eri_coul"], NOCC)
    return e_hf + mp2_pieces(c, eps, m[f"u_{corr}"], m[f"v_{corr}"], NOCC, fc, scale)[0]


def analytic_gradient(x, omega, fc, scale, corr="erfc"):
    m = build(x, omega)
    s, h, ds, dh = m["s"], core(x), m["ds"], H1
    eri_ref, deri_ref = m["eri_coul"], m["deri_coul"]
    deri_ref = m["deri_coul"]  # four-centre derivative operator
    u, v, du, dv = m[f"u_{corr}"], m[f"v_{corr}"], m[f"du_{corr}"], m[f"dv_{corr}"]
    du_3c = m[f"du_{corr}"]  # 3-centre derivative operator
    _, c, eps = rhf(s, h, eri_ref, NOCC)
    nocc, nmo = NOCC, c.shape[1]
    nv = nmo - nocc
    _, t, tt, b_full, vm12 = mp2_pieces(c, eps, u, v, nocc, fc, scale)
    p_oo = -np.einsum("ikab,jkab->ij", t, tt)
    p_vv = np.einsum("ijac,ijbc->ab", t, tt)
    xov = np.einsum("ijab,Pjb->Pia", tt, b_full[:, fc:nocc, nocc:])
    imat = np.zeros((nmo, nmo))
    imat[:, fc:nocc] = -2 * np.einsum("Pia,Pqa->qi", xov, b_full[:, :, nocc:])
    imat[:, nocc:] = -2 * np.einsum("Pia,Pqi->qa", xov, b_full[:, :, fc:nocc])
    dm1 = np.zeros((nmo, nmo))
    dm1[fc:nocc, fc:nocc] = p_oo + p_oo.T
    dm1[nocc:, nocc:] = p_vv + p_vv.T
    for k in range(fc):
        for i in range(fc, nocc):
            dm1[k, i] = dm1[i, k] = imat[k, i] / (eps[k] - eps[i])
    jc, kc = jk(eri_ref, c @ dm1 @ c.T)
    veff_mo = c.T @ (2 * jc - kc) @ c
    i_ia, i_ai = imat[:nocc, nocc:].T, imat[nocc:, :nocc]
    xvo = veff_mo[nocc:, :nocc] + i_ia - i_ai  # Z-vector RHS
    co, cv = c[:, :nocc], c[:, nocc:]

    def a_z(zvec):
        zm = zvec.reshape(nv, nocc)
        dz = cv @ zm @ co.T + co @ zm.T @ cv.T
        jz, kz = jk(eri_ref, dz)
        return (
            (eps[nocc:, None] - eps[None, :nocc]) * zm
            + 0.5 * (cv.T @ (4 * jz - kz - kz.T) @ co)
        ).ravel()

    amat = np.array([a_z(e) for e in np.eye(nv * nocc)]).T
    z = np.linalg.solve(amat, -xvo.ravel()).reshape(nv, nocc)
    dm1[nocc:, :nocc] = z
    dm1[:nocc, nocc:] = z.T
    d_hf = 2 * co @ co.T
    d_rel = c @ dm1 @ c.T
    zeta = 0.5 * (eps[:, None] + eps[None, :]) * dm1
    zeta[nocc:, :nocc] = eps[None, :nocc] * dm1[nocc:, :nocc]
    zeta[:nocc, nocc:] = eps[:nocc, None] * dm1[:nocc, nocc:]
    zeta_ao = c @ zeta @ c.T + 2 * co @ np.diag(eps[:nocc]) @ co.T
    ip = imat.copy()
    ip[nocc:, :nocc] = imat[:nocc, nocc:].T
    im1 = c @ ip @ c.T
    jr, kr = jk(eri_ref, d_rel)
    p_occ = co @ co.T
    vhf = p_occ @ (2 * jr - kr) @ p_occ
    grad = np.sum((d_hf + d_rel) * dh) + np.sum((im1 - zeta_ao - vhf) * ds)
    jd, kd = jk(deri_ref, d_hf + 2 * d_rel)
    grad += 0.5 * np.sum(d_hf * (jd - 0.5 * kd))
    # RI response at fixed C on the correlation operator.
    u_ia = np.einsum("Qmn,mi,na->Qia", u, c[:, fc:nocc], c[:, nocc:])
    du_ia = np.einsum("Qmn,mi,na->Qia", du_3c, c[:, fc:nocc], c[:, nocc:])
    b_ia = np.einsum("PQ,Qia->Pia", vm12, u_ia)
    y = np.einsum("PQ,Qia->Pia", vm12, np.einsum("ijab,Pjb->Pia", tt, b_ia))
    m_u = np.einsum("PQ,Qia->Pia", np.linalg.inv(v), u_ia)
    grad += 4 * np.sum(y * du_ia)
    grad += -2 * np.einsum("PQ,Pia,Qia->", dv, m_u, y)  # metric derivative
    return grad


def part_l():
    print("L: attenuated RI-MP2 gradient vs re-solved finite differences")
    x0 = 0.05
    for scale, fc in itertools.product((1.0, 1.2), (0, 1)):
        g = analytic_gradient(x0, OMEGA, fc, scale)
        fd = richardson(lambda xx: total_energy(xx, OMEGA, fc, scale), x0, 2e-3)
        report(
            "L", f"s = {scale}, frozen core {fc}", abs(g - fd) / max(1.0, abs(fd)), 1e-8
        )


def part_z():
    print("Z: w -> 0 gives the Coulomb RI-MP2 gradient")
    x0 = 0.05
    g_att = analytic_gradient(x0, 1e-6, 0, 1.0)
    g_mp2 = analytic_gradient(x0, 1e-6, 0, 1.0, corr="coul")
    report(
        "Z",
        "|g_att(w=1e-6) - g_RI-MP2|",
        abs(g_att - g_mp2) / max(1.0, abs(g_mp2)),
        1e-9,
    )
    gap = abs(analytic_gradient(x0, OMEGA, 0, 1.0) - g_mp2)
    report("Z", "non-vacuity: 1e-5 / |g_att(w=0.2222) - g_RI-MP2|", 1e-5 / gap, 1.0)


def main():
    global INTS, H0, H1
    part_e()
    part_w()
    INTS = compiled()
    n = len(AO_EXP) * len(ATOM_Z0)
    m0 = RNG.standard_normal((n, n))
    m1 = RNG.standard_normal((n, n))
    H0 = np.diag(np.linspace(-5.0, 1.0, n)) + 0.1 * (m0 + m0.T)
    H1 = 0.1 * (m1 + m1.T)
    part_l()
    part_z()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): {FAILED}")
        sys.exit(1)
    print("all attenuated RI-MP2 gradient identities hold")


if __name__ == "__main__":
    main()
