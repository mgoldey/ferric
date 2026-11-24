#!/usr/bin/env python3
"""Proofs of the terfc-attenuated RI-MP2 nuclear-gradient identities ferric codes.

Theory: `book/src/15-correlated-gradients/02-rimp2-gradient.md#rimp2-gradient`;
`book/src/04-operators/02-terf-table-engine.md#terf-table-engine-derivative-integrals`.
Code: crates/ferric-integrals/shim/shim.cc (compute_cart_eri3_deriv,
compute_cart_eri2_deriv, hermite_deriv_first, md_kernel_R);
crates/ferric-integrals/src/engine.rs (Engine::new_3center_deriv /
new_2center_deriv for Operator::terfc); crates/ferric-mp2/src/att_rimp2_gradient.rs
(terfc_rimp2_gradient_analytical, total_terfc_rimp2_gradient_on).

The energy is E = E_HF[1/r] + E_c[terfc(r; r0, w)/r]: Coulomb RHF orbitals, the
correlation's 3-centre (P|mn) AND metric (P|Q) both under terfc, with
    terf(r)  = [erf(w(r - r0)) + erf(w(r + r0))]/2,   terfc = 1 - terf.

  K  The s-type terf integral and its derivative.
       I_terf = A sqrt(phi^2/rho) G0(phi^2 R^2, phi^2 r0^2),
       G0(S, s) = sqrt(pi)/(4 sqrt S) [erf(sqrt S + sqrt s) + erf(sqrt S - sqrt s)],
       1/phi^2 = 1/rho + 1/w^2,   A = 2 pi^(5/2) K1 K2/(p q sqrt(p+q)),
     and I_terfc = I_coul - I_terf.
     K1 the closed form equals an independent route: the k-space integral
        (pi^2/pq)^(3/2) K1 K2 (2/pi) int_0^inf exp(-k^2/4 phi^2) cos(k r0)
        sin(kR)/(kR) dk (the Fourier transform of the kernel, no G0, no erf);
     K2 the R-derivative through G0' = -G1 equals the k-space integral with
        d/dR [sin(kR)/(kR)] (no finite difference anywhere);
     K3 G1 = -dG0/dS from the closed form G1 = (G0 - [e^{-(x+a)^2} +
        e^{-(x-a)^2}]/2)/(2S) (x = sqrt S, a = sqrt s) holds symbolically, and
        reduces to the Boys F1 at s = 0;
     K4 sympy's derivatives of the 3-centre and 2-centre terfc closed forms
        equal Richardson differences, at separated and coincident centres, and
        the centre derivatives of each sum to zero (translational invariance,
        the shim's dB = -(dP + dA)).
  H  The derivative kernel. The shim differentiates a Cartesian Gaussian
     through d/dA [x_A^i e^{-a x_A^2}] = 2a x_A^{i+1} e - i x_A^{i-1} e, so the
     Hermite expansion of the differentiated product x_A^i x_B^j e^{-a x_A^2 -
     b x_B^2} has coefficients 2a E^{i+1,j}_t - i E^{i-1,j}_t over the SAME
     Hermite Gaussians Lambda_t(x; P), t <= i+j+1. Checked symbolically for
     i, j <= 3 with E from the shim's recurrence (build_hermite_E).
  L  The assembled gradient on a 1-D model with real integrals (three atoms on
     z, two s Gaussians per atom, Coulomb four-centre ERIs for the RHF, terfc
     (P|mn) and (P|Q) on three aux s Gaussians per atom for the correlation;
     x is the z of the middle atom): the analytic Lagrangian (Coulomb in the
     Z-vector's veff, CPHF coupling, vhf_s1occ and four-centre derivative;
     terfc in amplitudes, b_full, Imat and the 3c/2c response with the K4
     derivatives) equals the Richardson derivative of the re-solved
     E_HF + E_c(terfc), r0 = 1.05 A on the curvature constraint and a
     decoupled w r0 = 4, with and without one frozen core.
  R  The r0 -> 0 limit at fixed w: G0(S, 0) = F0(S) symbolically, so terfc
     becomes erfc(w r)/r; on L's model the terfc gradient at r0 = 1e-6 Bohr
     equals the erfc gradient to 1e-9 relative, and at r0 = 1.05 A differs from
     it by far more.

Mutations (each turns the script red; the runner applies them one at a time
as exact substitutions of the tagged lines):
  * the terfc metric-derivative (2-centre) term dropped breaks L ("FAIL L").
  * the 2-centre metric derivative on the Coulomb operator instead of terfc
    breaks L ("FAIL L").
  * the four-centre derivative term on the terfc ERIs breaks L ("FAIL L").
  * the Z-vector coupling (orbital Hessian) on the terfc ERIs breaks L
    ("FAIL L").
  * terfc = Coulomb + terf (wrong sign of the split) breaks K ("FAIL K").
  * the -i E^{i-1,j} term of the Hermite derivative dropped breaks H
    ("FAIL H").

Run: python3 scripts/terfc_rimp2_gradient_sympy.py  (needs sympy, numpy, scipy, mpmath)
"""

import itertools
import math
import sys

import mpmath
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.linalg import eigh

FAILED = []
RNG = np.random.default_rng(20261116)
BOHR_PER_ANG = 1.0 / 0.529177210903
R0 = 1.05 * BOHR_PER_ANG  # the aug-cc-pVDZ default r0, Bohr
W_LINKED = 1.0 / (R0 * math.sqrt(2.0))  # the curvature constraint
W_FREE = 4.0 / R0  # a decoupled sharpness, w r0 = 4
mpmath.mp.dps = 60


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


# ================================================================ special functions
def _g0_mp(sv, tv):
    """G0(S, s) at 60 digits; S = 0 is the limit exp(-s)."""
    sv, tv = mpmath.mpf(sv), mpmath.mpf(tv)
    if sv == 0:
        return mpmath.e ** (-tv)
    x, a = mpmath.sqrt(sv), mpmath.sqrt(tv)
    return mpmath.sqrt(mpmath.pi) / (4 * x) * (mpmath.erf(x + a) + mpmath.erf(x - a))


def _g1_mp(sv, tv):
    """G1 = -dG0/dS = (G0 - [e^{-(x+a)^2} + e^{-(x-a)^2}]/2)/(2S)."""
    sv = max(mpmath.mpf(sv), mpmath.mpf("1e-40"))
    tv = mpmath.mpf(tv)
    x, a = mpmath.sqrt(sv), mpmath.sqrt(tv)
    tail = (mpmath.e ** (-((x + a) ** 2)) + mpmath.e ** (-((x - a) ** 2))) / 2
    return (_g0_mp(sv, tv) - tail) / (2 * sv)


def g0num(sv, tv):
    return float(_g0_mp(sv, tv))


def g1num(sv, tv):
    return float(_g1_mp(sv, tv))


def boys0(t):
    return g0num(t, 0.0)


def boys1(t):
    return g1num(t, 0.0)


class F1(sp.Function):
    """Boys F1."""


class F0(sp.Function):
    """Boys F0 with F0'(T) = -F1(T)."""

    def fdiff(self, argindex=1):
        return -F1(self.args[0])


class G1(sp.Function):
    """G1(S, s) = -dG0/dS."""


class G0(sp.Function):
    """The terf auxiliary G0(S, s); only S carries geometry."""

    def fdiff(self, argindex=1):
        if argindex != 1:
            raise ValueError("s carries no geometry")
        return -G1(*self.args)


MODULES = [{"F0": boys0, "F1": boys1, "G0": g0num, "G1": g1num}, "math"]

# ================================================================ closed forms
w, r0 = sp.symbols("w r0", positive=True)
a, b, c, d, al, be = sp.symbols("a b c d alpha beta", positive=True)
za, zb, zc, zd, zp, zq = sp.symbols("z_a z_b z_c z_d z_p z_q", real=True)

TERFC_SIGN = -1  # terfc = Coulomb + TERFC_SIGN * terf


def norm(e):
    return (2 * e / sp.pi) ** sp.Rational(3, 4)


def pair(e1, z1, e2, z2):
    p = e1 + e2
    return (
        p,
        (e1 * z1 + e2 * z2) / p,
        norm(e1) * norm(e2) * sp.exp(-e1 * e2 / p * (z1 - z2) ** 2),
    )


def single(e, z):
    return e, z, norm(e)


def dist_int(d1, d2, kernel):
    """(d1|K|d2) for s Gaussian distributions; K in coul, erfc, terf, terfc."""
    (p, zp_, k1), (q, zq_, k2) = d1, d2
    rho = p * q / (p + q)
    r2 = (zp_ - zq_) ** 2
    pref = 2 * sp.pi ** sp.Rational(5, 2) * k1 * k2 / (p * q * sp.sqrt(p + q))
    coul = pref * F0(rho * r2)
    if kernel == "coul":
        return coul
    phi2 = rho * w**2 / (rho + w**2)
    if kernel == "erfc":
        return coul - pref * sp.sqrt(phi2 / rho) * F0(phi2 * r2)
    terf = pref * sp.sqrt(phi2 / rho) * G0(phi2 * r2, phi2 * r0**2)
    if kernel == "terf":
        return terf
    return coul + TERFC_SIGN * terf


def eri3_expr(kernel):
    return dist_int(single(al, zp), pair(a, za, b, zb), kernel)


def eri2_expr(kernel):
    return dist_int(single(al, zp), single(be, zq), kernel)


def eri4_expr(kernel):
    return dist_int(pair(a, za, b, zb), pair(c, zc, d, zd), kernel)


ARGS = (a, b, c, d, al, be, za, zb, zc, zd, zp, zq, w, r0)


def compile_int(expr, centres):
    val = sp.lambdify(ARGS, expr, modules=MODULES)
    ders = [sp.lambdify(ARGS, sp.diff(expr, z), modules=MODULES) for z in centres]
    return val, ders


# ================================================================ part K
def kspace_2c(alv, bev, rv, wv, r0v, deriv):
    """(P|terf|Q) or its R-derivative from the kernel's Fourier transform."""
    n1, n2 = (2 * alv / np.pi) ** 0.75, (2 * bev / np.pi) ** 0.75
    rho = alv * bev / (alv + bev)
    phi2 = rho * wv * wv / (rho + wv * wv)

    def f(k):
        if deriv:
            kr = k * rv
            radial = (kr * math.cos(kr) - math.sin(kr)) / (k * rv * rv)
        else:
            radial = math.sin(k * rv) / (k * rv)
        return math.exp(-k * k / (4 * phi2)) * math.cos(k * r0v) * radial

    kmax = 2 * math.sqrt(phi2) * 9.0
    val = quad(f, 0.0, kmax, limit=400, epsabs=1e-15, epsrel=1e-13)[0]
    return n1 * n2 * (np.pi**2 / (alv * bev)) ** 1.5 * 2 / np.pi * val


def part_k():
    print("K: the terf / terfc s integrals and their derivatives")
    i2 = sp.lambdify((al, be, zp, zq, w, r0), eri2_expr("terf"), modules=MODULES)
    d2 = sp.lambdify(
        (al, be, zp, zq, w, r0), sp.diff(eri2_expr("terf"), zq), modules=MODULES
    )
    i2c = sp.lambdify((al, be, zp, zq, w, r0), eri2_expr("terfc"), modules=MODULES)
    i2coul = sp.lambdify((al, be, zp, zq, w, r0), eri2_expr("coul"), modules=MODULES)
    worst_v, worst_d, worst_split = 0.0, 0.0, 0.0
    for alv, bev, rv, wv, r0v in [
        (0.4, 1.2, 1.3, W_LINKED, R0),
        (2.0, 0.7, 0.35, W_LINKED, R0),
        (1.1, 1.1, 3.0, W_FREE, R0),
        (0.9, 3.0, 2.2, 1.7, 0.6),
    ]:
        ref = kspace_2c(alv, bev, rv, wv, r0v, False)
        worst_v = max(worst_v, abs(i2(alv, bev, 0.0, rv, wv, r0v) - ref) / abs(ref))
        dref = kspace_2c(alv, bev, rv, wv, r0v, True)
        worst_d = max(worst_d, abs(d2(alv, bev, 0.0, rv, wv, r0v) - dref) / abs(dref))
        # terfc = Coulomb - terf, the terf from k-space (independent of G0).
        tc = i2c(alv, bev, 0.0, rv, wv, r0v)
        worst_split = max(
            worst_split, abs(tc - (i2coul(alv, bev, 0.0, rv, wv, r0v) - ref)) / abs(tc)
        )
    report("K", "K1 (P|terf|Q) closed form vs k-space quadrature", worst_v, 1e-10)
    report("K", "K2 d(P|terf|Q)/dR (G0' = -G1) vs k-space quadrature", worst_d, 1e-9)
    report("K", "K1 (P|terfc|Q) = Coulomb - k-space terf", worst_split, 1e-9)

    # K3: G1 closed form, symbolically.
    s_, t_ = sp.symbols("S s", positive=True)
    x_, a_ = sp.sqrt(s_), sp.sqrt(t_)
    g0 = sp.sqrt(sp.pi) / (4 * x_) * (sp.erf(x_ + a_) + sp.erf(x_ - a_))
    g1 = (g0 - (sp.exp(-((x_ + a_) ** 2)) + sp.exp(-((x_ - a_) ** 2))) / 2) / (2 * s_)
    resid = sp.simplify(sp.diff(g0, s_) + g1)
    report("K", "K3 G1 == -dG0/dS (symbolic)", float(resid != 0), 0)
    f1 = (sp.sqrt(sp.pi) / (2 * x_) * sp.erf(x_) - sp.exp(-s_)) / (2 * s_)
    resid = sp.simplify(sp.limit(g1, t_, 0) - f1)
    report("K", "K3 G1(S, 0) == F1(S) (symbolic)", float(resid != 0), 0)

    # K4: chain-rule derivatives vs Richardson; translational invariance.
    worst = 0.0
    for kind, expr, cents in [
        ("3c", eri3_expr("terfc"), (zp, za, zb)),
        ("2c", eri2_expr("terfc"), (zp, zq)),
    ]:
        val, ders = compile_int(expr, cents)
        for wv, r0v in [(W_LINKED, R0), (W_FREE, R0)]:
            for geom in ([0.3, -0.4, 1.1], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]):
                z = dict(zip(cents, geom))

                def call(f, shift=None, h=0.0, z=z, wv=wv, r0v=r0v):
                    zz = dict(z)
                    if shift is not None:
                        zz[shift] += h
                    return f(
                        0.8, 1.9, 0.0, 0.0, 0.6, 1.4,
                        zz.get(za, 0.0), zz.get(zb, 0.0), 0.0, 0.0,
                        zz.get(zp, 0.0), zz.get(zq, 0.0), wv, r0v,
                    )  # fmt: skip

                for cen, der in zip(cents, ders):
                    fd = richardson(lambda h: call(val, cen, h), 0.0, 1e-3)
                    worst = max(worst, abs(call(der) - fd))
    report("K", "K4 d(P|terfc|mn), d(P|terfc|Q) vs Richardson", worst, 1e-10)
    s3 = sp.simplify(sum(sp.diff(eri3_expr("terfc"), z) for z in (zp, za, zb)))
    s2 = sp.simplify(sum(sp.diff(eri2_expr("terfc"), z) for z in (zp, zq)))
    report(
        "K",
        "K4 sum of centre derivatives (3c, 2c) == 0",
        float(s3 != 0) + float(s2 != 0),
        0,
    )


# ================================================================ part H
HERMITE_LOWER = 1  # weight of the -i E^{i-1,j} term of the derivative


def hermite_e(av, bv, ax, bx, li, lj):
    """E^{i,j}_t by the shim's recurrence (build_hermite_E), sympy exact."""
    p = av + bv
    mu = av * bv / p
    px = (av * ax + bv * bx) / p
    pa, pb = px - ax, px - bx
    e = {(0, 0, 0): sp.exp(-mu * (ax - bx) ** 2)}

    def g(i, j, t):
        return e.get((i, j, t), 0) if 0 <= t <= i + j else 0

    for i in range(li + 1):
        for j in range(lj + 1):
            if i == 0 and j == 0:
                continue
            for t in range(i + j + 1):
                if i > 0:
                    v = g(i - 1, j, t - 1) / (2 * p) + pa * g(i - 1, j, t)
                    v += (t + 1) * g(i - 1, j, t + 1)
                else:
                    v = g(i, j - 1, t - 1) / (2 * p) + pb * g(i, j - 1, t)
                    v += (t + 1) * g(i, j - 1, t + 1)
                e[(i, j, t)] = v
    return e, px


def part_h():
    print("H: the Hermite expansion of the differentiated Gaussian product")
    x, ax, bx = sp.symbols("x A_x B_x", real=True)
    av, bv = sp.Rational(7, 10), sp.Rational(13, 10)
    worst = 0.0
    for i, j in itertools.product(range(4), range(4)):
        e, px = hermite_e(av, bv, ax, bx, i + 1, j)
        p = av + bv
        psym = sp.Symbol("P_x", real=True)
        lam = [
            sp.diff(sp.exp(-p * (x - psym) ** 2), psym, t).subs(psym, px)
            for t in range(i + j + 2)
        ]
        # Lambda_t = (d/dP)^t exp(-p x_P^2); the shim's tables use this basis.
        expansion = sum(
            (
                2 * av * e.get((i + 1, j, t), 0)
                - HERMITE_LOWER * i * e.get((i - 1, j, t), 0)
            )
            * lam[t]
            for t in range(i + j + 2)
        )
        target = sp.diff(
            (x - ax) ** i
            * (x - bx) ** j
            * sp.exp(-av * (x - ax) ** 2 - bv * (x - bx) ** 2),
            ax,
        )
        for xv, axv, bxv in [(0.3, -0.2, 0.9), (1.4, 0.5, -0.7), (-0.6, 0.0, 0.0)]:
            sub = {x: xv, ax: axv, bx: bxv}
            ev = float(expansion.subs(sub).evalf(30))
            tv = float(target.subs(sub).evalf(30))
            worst = max(worst, abs(ev - tv))
    report(
        "H",
        "H sum_t (2a E^{i+1,j}_t - i E^{i-1,j}_t) Lambda_t == d/dA (i,j <= 3)",
        worst,
        1e-13,
    )


# ================================================================ part L model
ATOM_Z0 = np.array([0.0, 1.4, 2.9])
MOVING = 1
AO_EXP = [0.6, 2.4]
AUX_EXP = [0.35, 1.1, 3.8]
NOCC = 3
INTS = None
H0 = None
H1 = None


def compiled():
    out = {}
    for kern in ("coul", "erfc", "terfc"):
        out[("4", kern)] = compile_int(eri4_expr(kern), (za, zb, zc, zd))
        out[("3", kern)] = compile_int(eri3_expr(kern), (zp, za, zb))
        out[("2", kern)] = compile_int(eri2_expr(kern), (zp, zq))
    return out


def basis(x, exps):
    z = ATOM_Z0.copy()
    z[MOVING] += x
    return [(e, z[at], at) for at in range(len(z)) for e in exps]


def build(x, omega, r0v, kernels=("coul", "terfc")):
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
    for kern in kernels:
        v4, d4 = INTS[("4", kern)]
        eri = np.zeros((n, n, n, n))
        deri = np.zeros_like(eri)
        for i, j, k, s4 in itertools.product(range(n), repeat=4):
            f = (
                ao[i][0], ao[j][0], ao[k][0], ao[s4][0], 0.0, 0.0,
                ao[i][1], ao[j][1], ao[k][1], ao[s4][1], 0.0, 0.0, omega, r0v,
            )  # fmt: skip
            eri[i, j, k, s4] = v4(*f)
            deri[i, j, k, s4] = sum(
                dd(*f) for dd, idx in zip(d4, (i, j, k, s4)) if ao[idx][2] == MOVING
            )
        out[f"eri_{kern}"], out[f"deri_{kern}"] = eri, deri
        v3, d3 = INTS[("3", kern)]
        u = np.zeros((m, n, n))
        du = np.zeros_like(u)
        for pp, i, j in itertools.product(range(m), range(n), range(n)):
            if j < i:
                u[pp, i, j], du[pp, i, j] = u[pp, j, i], du[pp, j, i]
                continue
            f = (
                ao[i][0], ao[j][0], 0.0, 0.0, aux[pp][0], 0.0,
                ao[i][1], ao[j][1], 0.0, 0.0, aux[pp][1], 0.0, omega, r0v,
            )  # fmt: skip
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
                0.0, 0.0, 0.0, 0.0, aux[pp][0], aux[qq][0],
                0.0, 0.0, 0.0, 0.0, aux[pp][1], aux[qq][1], omega, r0v,
            )  # fmt: skip
            v[pp, qq] = v2(*f)
            dv[pp, qq] = sum(
                dd(*f) for dd, cen in zip(d2, (aux[pp], aux[qq])) if cen[2] == MOVING
            )
        out[f"u_{kern}"], out[f"du_{kern}"] = u, du
        out[f"v_{kern}"], out[f"dv_{kern}"] = v, dv
    return out


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


def mp2_pieces(c, eps, u, v, nocc, fc):
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
    tt = 2 * t - t.transpose(0, 1, 3, 2)
    return np.sum(g * tt), t, tt, b_full, vm12


def total_energy(x, omega, r0v, fc, corr="terfc"):
    m = build(x, omega, r0v, ("coul", corr))
    e_hf, c, eps = rhf(m["s"], core(x), m["eri_coul"], NOCC)
    return e_hf + mp2_pieces(c, eps, m[f"u_{corr}"], m[f"v_{corr}"], NOCC, fc)[0]


def analytic_gradient(x, omega, r0v, fc, corr="terfc", kernels=None):
    m = build(x, omega, r0v, kernels or ("coul", corr))
    s, h, ds, dh = m["s"], core(x), m["ds"], H1
    eri_ref = m["eri_coul"]  # Z-vector and veff operator
    deri_ref = m["deri_coul"]  # four-centre derivative operator
    u, v = m[f"u_{corr}"], m[f"v_{corr}"]
    du_3c = m[f"du_{corr}"]  # 3-centre derivative operator
    dv = m[f"dv_{corr}"]  # 2-centre derivative operator
    _, c, eps = rhf(s, h, eri_ref, NOCC)
    nocc, nmo = NOCC, c.shape[1]
    nv = nmo - nocc
    _, t, tt, b_full, vm12 = mp2_pieces(c, eps, u, v, nocc, fc)
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
    xvo = veff_mo[nocc:, :nocc] + i_ia - i_ai
    co, cv = c[:, :nocc], c[:, nocc:]

    def a_z(zvec):
        zm = zvec.reshape(nv, nocc)
        dz = cv @ zm @ co.T + co @ zm.T @ cv.T
        jz, kz = jk(eri_ref, dz)  # Z-vector coupling operator
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
    print("L: terfc RI-MP2 gradient vs re-solved finite differences")
    x0 = 0.05
    for (wv, label), fc in itertools.product(
        ((W_LINKED, "w r0 = 1/sqrt 2"), (W_FREE, "w r0 = 4")), (0, 1)
    ):
        g = analytic_gradient(x0, wv, R0, fc)
        fd = richardson(lambda xx: total_energy(xx, wv, R0, fc), x0, 2e-3)
        report("L", f"{label}, frozen core {fc}", abs(g - fd) / max(1.0, abs(fd)), 1e-8)


def part_r():
    print("R: r0 -> 0 at fixed w gives the erfc gradient")
    s_ = sp.Symbol("S", positive=True)
    x_ = sp.sqrt(s_)
    g0_s0 = sp.sqrt(sp.pi) / (4 * x_) * (sp.erf(x_) + sp.erf(x_))
    f0 = sp.sqrt(sp.pi) / (2 * x_) * sp.erf(x_)
    report("R", "G0(S, 0) == F0(S) (symbolic)", float(sp.simplify(g0_s0 - f0) != 0), 0)
    x0 = 0.05
    wv = 0.420 * 0.529177210903
    kern = ("coul", "erfc", "terfc")
    g_t = analytic_gradient(x0, wv, 1e-6, 0, "terfc", kern)
    g_e = analytic_gradient(x0, wv, 1e-6, 0, "erfc", kern)
    report(
        "R", "|g_terfc(r0=1e-6) - g_erfc|", abs(g_t - g_e) / max(1.0, abs(g_e)), 1e-9
    )
    gap = abs(analytic_gradient(x0, wv, R0, 0, "terfc", kern) - g_e)
    report("R", "non-vacuity: 1e-5 / |g_terfc(r0=1.05 A) - g_erfc|", 1e-5 / gap, 1.0)


def main():
    global INTS, H0, H1
    part_k()
    part_h()
    INTS = compiled()
    n = len(AO_EXP) * len(ATOM_Z0)
    m0 = RNG.standard_normal((n, n))
    m1 = RNG.standard_normal((n, n))
    H0 = np.diag(np.linspace(-5.0, 1.0, n)) + 0.1 * (m0 + m0.T)
    H1 = 0.1 * (m1 + m1.T)
    part_l()
    part_r()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): {FAILED}")
        sys.exit(1)
    print("all terfc RI-MP2 gradient identities hold")


if __name__ == "__main__":
    main()
