#!/usr/bin/env python3
"""Proofs of the SCS-MP2(2terfc) and open-shell terfc RI-MP2 gradient identities ferric codes.

Theory: crates/ferric-mp2/theory.md, "9.6 SCS-MP2(2terfc) gradient" and "15.2
Open-shell terfc and SCS-MP2(2terfc) gradients".
Code: crates/ferric-mp2/src/scs_2terfc_gradient.rs (scs_2terfc_terms,
scs_mp2_2terfc_gradient_analytical, u_scs_mp2_2terfc_gradient_analytical);
crates/ferric-mp2/src/gradient.rs (rimp2_terms_gradient_timed);
crates/ferric-mp2/src/u_rimp2_gradient.rs (u_terms_gradient_timed,
u_energy_weights_scaled); crates/ferric-mp2/src/u_rimp2.rs
(build_u_mp2_density_scaled); crates/ferric-mp2/src/att_rimp2_gradient.rs
(u_terfc_rimp2_gradient_analytical).

The energy (scs.rs scs_mp2_2terfc, JCTC 11, 4159 (2015) Eq. 12 without VV10):
    E = E_HF[1/r] + c_os E_OS[t1] + c_ss (E_SS[t2] - E_SS[t1]),
    tk = terfc(r; r0(k), w_k)/r,  w_k = 1/(sqrt 2 r0(k)),
on a Coulomb RHF or UHF; each E_c[tk] has its (P|mn) and (P|Q) under tk.

  S  Symbolic identities.
     S1 the two terms (t1, (c_os, -c_ss)) and (t2, (0, c_ss)), each a spin
        scaling E[t; (a, b)] = a E_OS[t] + b E_SS[t], sum to Eq. 12;
     S2 closed shell: sum K ((a + b) t - b tbar) = a E_OS + b E_SS with
        E_OS = sum K t, E_SS = sum K (t - tbar) (the T~ of theory.md 9.2);
     S3 open shell: for E_w = w_ss (E_aa + E_bb) + w_os E_ab on symbolic
        integrals B (one aux function), the denominator derivative
        dE_w/deps^s_p is the diagonal of the weighted Gram densities
        (w_ss on the same-spin products with weight 1/2, w_os on the
        opposite-spin ones with weight 1), and dE_w/dB^s = 2 x^s with
        x^a = w_ss sum t_aa B^a + w_os sum t_ab B^b (u_energy_weights_scaled,
        build_u_mp2_density_scaled).
  C  Closed shell, one Z-vector: on a 1-D model with real integrals (three
     atoms on z, two s Gaussians per atom, Coulomb four-centre ERIs for the
     RHF, terfc (P|mn) and (P|Q) at both cutoffs on three aux s Gaussians per
     atom), the analytic gradient (each term its own amplitudes, densities,
     Imat and 3c/2c response on its cutoff; ONE Coulomb Z-vector on the
     summed Imat and P) equals the Richardson derivative of the re-solved
     Eq. 12 energy, frozen core 0 and 1.
  U  Open shell (UHF, n_a = 3, n_b = 2): the same with the unrestricted
     Lagrangian (theory.md 15.1) and the spin-weighted x^s and P^s of S3;
     plus the open-shell terfc RI-MP2 gradient (one term, (1, 1)); frozen
     core 0 and 1.
  L  Limits: the open-shell assembly at n_a = n_b equals the closed-shell
     one; with r0(1) = r0(2) the same-spin terms cancel and the closed
     gradient is the single-term SOS (c_os, 0) gradient.

The Eq. 12 energy is evaluated from E_OS and E_SS directly, never from the
term list, so a wrong term (cutoff or weight) shows as a disagreement in C/U.

Mutations (each turns the script red; the runner applies them one at a time
as exact substitutions of the tagged lines):
  * the second term on the bonded cutoff r0(1) breaks C ("FAIL C").
  * the same-spin weight on the opposite-spin products breaks U ("FAIL U").
  * the beta Z-vector dropped breaks U ("FAIL U").
  * the Z-vector coupling on the terfc (RI) kernel instead of Coulomb breaks
    C ("FAIL C").
  * only the last term's Imat in the Z-vector right-hand side breaks C
    ("FAIL C").
  * the weights of x^a swapped (c_ss on the opposite-spin sum) breaks S
    ("FAIL S").

Run: python3 scripts/scs_2terfc_gradient_sympy.py  (needs sympy, numpy, scipy, mpmath)
"""

import itertools
import math
import sys

import mpmath
import numpy as np
import sympy as sp
from scipy.linalg import eigh

FAILED = []
RNG = np.random.default_rng(20261117)
BOHR_PER_ANG = 1.0 / 0.529177210903
R0_1 = 0.75 * BOHR_PER_ANG  # bonded cutoff, Bohr
R0_2 = 1.05 * BOHR_PER_ANG  # non-bonded cutoff, Bohr
C_OS, C_SS = 1.27, 4.05
mpmath.mp.dps = 40


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


def terms(c_os, c_ss, r1, r2):
    """The correlation terms of Eq. 12: (cutoff, c_os of the term, c_ss of the term)."""
    return [
        (r1, c_os, -c_ss),  # term 1: the bonded cutoff
        (r2, 0 * c_os, c_ss),  # term 2: the non-bonded cutoff
    ]


# ================================================================ part S
def part_s():
    print("S: symbolic identities")
    c_os, c_ss = sp.symbols("c_os c_ss")
    r1, r2 = sp.symbols("r1 r2", positive=True)
    e_os = {r1: sp.Symbol("EOS1"), r2: sp.Symbol("EOS2")}
    e_ss = {r1: sp.Symbol("ESS1"), r2: sp.Symbol("ESS2")}
    total = sum(a * e_os[r] + b * e_ss[r] for r, a, b in terms(c_os, c_ss, r1, r2))
    eq12 = c_os * e_os[r1] + c_ss * (e_ss[r2] - e_ss[r1])
    report("S", "S1 terms sum to Eq. 12", float(sp.simplify(total - eq12) != 0), 0)

    # S2: closed shell, 2 occupied x 2 virtual, symbolic K and denominators.
    a_, b_ = sp.symbols("a b")
    no, nv = 2, 2
    kk = {
        idx: sp.Symbol("K_%d%d%d%d" % idx)
        for idx in itertools.product(range(no), range(no), range(nv), range(nv))
    }
    eo = sp.symbols("eo0:2")
    ev = sp.symbols("ev0:2")

    def dd(i, j, a, b):
        return eo[i] + eo[j] - ev[a] - ev[b]

    # K_ij^ab = (ia|jb) is symmetric under (i a) <-> (j b).
    def k(i, j, a, b):
        return kk[(min(i, j), max(i, j), a, b) if i <= j else (j, i, b, a)]

    def t(i, j, a, b):
        return k(i, j, a, b) / dd(i, j, a, b)

    idx = list(itertools.product(range(no), range(no), range(nv), range(nv)))
    e_os2 = sum(k(*q) * t(*q) for q in idx)
    e_ss2 = sum(k(i, j, a, b) * (t(i, j, a, b) - t(i, j, b, a)) for i, j, a, b in idx)
    e_tt = sum(
        k(i, j, a, b) * ((a_ + b_) * t(i, j, a, b) - b_ * t(i, j, b, a))
        for i, j, a, b in idx
    )
    report(
        "S",
        "S2 sum K T~ = a E_OS + b E_SS",
        float(sp.simplify(e_tt - a_ * e_os2 - b_ * e_ss2) != 0),
        0,
    )
    part_s3()


def part_s3():
    """S3: the weighted open-shell densities and energy weights, symbolically."""
    w_ss, w_os = sp.symbols("w_ss w_os")
    oa, ob, va, vb = 2, 1, 2, 2
    ea = sp.symbols("ea0:%d" % (oa + va))
    eb = sp.symbols("eb0:%d" % (ob + vb))
    ba = sp.Matrix(oa, va, lambda i, a: sp.Symbol(f"Ba_{i}{a}"))
    bb = sp.Matrix(ob, vb, lambda i, a: sp.Symbol(f"Bb_{i}{a}"))

    def dn(e1, e2, o1, o2, i, j, a, b):
        return e1[i] + e2[j] - e1[o1 + a] - e2[o2 + b]

    def same(bm, e, o, v):
        g = lambda i, j, a, b: bm[i, a] * bm[j, b]  # noqa: E731
        t = {
            (i, j, a, b): (g(i, j, a, b) - g(i, j, b, a)) / dn(e, e, o, o, i, j, a, b)
            for i, j, a, b in itertools.product(range(o), range(o), range(v), range(v))
        }
        en = sp.Rational(1, 2) * sum(g(*q) * t[q] for q in t)
        return en, t

    e_aa, t_aa = same(ba, ea, oa, va)
    e_bb, t_bb = same(bb, eb, ob, vb)
    t_ab = {
        (i, j, a, b): ba[i, a] * bb[j, b] / dn(ea, eb, oa, ob, i, j, a, b)
        for i, j, a, b in itertools.product(range(oa), range(ob), range(va), range(vb))
    }
    e_ab = sum(ba[i, a] * bb[j, b] * t_ab[(i, j, a, b)] for i, j, a, b in t_ab)
    e_w = w_ss * (e_aa + e_bb) + w_os * e_ab

    # The code's weighted Gram densities (u_rimp2.rs u_mp2_density_weighted).
    def p_oo_a(k):
        return -sp.Rational(1, 2) * w_ss * sum(
            t_aa[(k, j, a, b)] ** 2
            for j, a, b in itertools.product(range(oa), range(va), range(va))
        ) - w_os * sum(
            t_ab[(k, j, a, b)] ** 2
            for j, a, b in itertools.product(range(ob), range(va), range(vb))
        )

    def p_vv_b(c):
        return sp.Rational(1, 2) * w_ss * sum(
            t_bb[(i, j, c, b)] ** 2
            for i, j, b in itertools.product(range(ob), range(ob), range(vb))
        ) + w_os * sum(
            t_ab[(i, j, a, c)] ** 2
            for i, j, a in itertools.product(range(oa), range(ob), range(va))
        )

    num = {s: v for s, v in zip(ea + eb, [-1.3, -0.9, 0.4, 0.8, -1.1, 0.5, 0.9])}
    num.update({s: float(RNG.uniform(0.2, 1.0)) for s in list(ba) + list(bb)})
    num.update({w_ss: 4.05, w_os: 1.27})

    def ev(expr):
        return float(expr.subs(num))

    err = max(abs(ev(sp.diff(e_w, ea[k]) - p_oo_a(k))) for k in range(oa)) + max(
        abs(ev(sp.diff(e_w, eb[ob + c]) - p_vv_b(c))) for c in range(vb)
    )
    report("S", "S3 dE_w/deps = weighted Gram densities (diagonal)", err, 1e-12)

    # dE_w/dB^a_{ia} = 2 x^a_{ia}, x^a = w_ss sum t_aa B^a + w_os sum t_ab B^b.
    w_same, w_opp = w_ss, w_os  # (same-spin, opposite-spin) weights of x

    def x_a(i, a):
        return w_same * sum(
            t_aa[(i, j, a, b)] * ba[j, b]
            for j, b in itertools.product(range(oa), range(va))
        ) + w_opp * sum(
            t_ab[(i, j, a, b)] * bb[j, b]
            for j, b in itertools.product(range(ob), range(vb))
        )

    # At fixed denominators: differentiate the numerator only.
    err = 0.0
    for i, a in itertools.product(range(oa), range(va)):
        d_e = sp.diff(e_w, ba[i, a])
        # remove the denominators' (absent) B dependence: E is quadratic over
        # B-free denominators, so the derivative is exact as is.
        err = max(err, abs(ev(d_e - 2 * x_a(i, a))))
    report("S", "S3 dE_w/dB^a = 2 x^a (weighted energy weights)", err, 1e-12)


# ================================================================ integrals
def _g0(sv, tv):
    sv, tv = mpmath.mpf(sv), mpmath.mpf(tv)
    if sv == 0:
        return mpmath.e ** (-tv)
    x, a = mpmath.sqrt(sv), mpmath.sqrt(tv)
    return mpmath.sqrt(mpmath.pi) / (4 * x) * (mpmath.erf(x + a) + mpmath.erf(x - a))


def _g1(sv, tv):
    sv = max(mpmath.mpf(sv), mpmath.mpf("1e-30"))
    tv = mpmath.mpf(tv)
    x, a = mpmath.sqrt(sv), mpmath.sqrt(tv)
    tail = (mpmath.e ** (-((x + a) ** 2)) + mpmath.e ** (-((x - a) ** 2))) / 2
    return (_g0(sv, tv) - tail) / (2 * sv)


def g0num(sv, tv):
    return float(_g0(sv, tv))


def g1num(sv, tv):
    return float(_g1(sv, tv))


class F1(sp.Function):
    """Boys F1."""


class F0(sp.Function):
    """Boys F0 with F0' = -F1."""

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


MODULES = [
    {
        "F0": lambda t: g0num(t, 0.0),
        "F1": lambda t: g1num(t, 0.0),
        "G0": g0num,
        "G1": g1num,
    },
    "math",
]

w, r0 = sp.symbols("w r0", positive=True)
a, b, c, d, al, be = sp.symbols("a b c d alpha beta", positive=True)
za, zb, zc, zd, zp, zq = sp.symbols("z_a z_b z_c z_d z_p z_q", real=True)
ARGS = (a, b, c, d, al, be, za, zb, zc, zd, zp, zq, w, r0)


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
    """(d1|K|d2) for s Gaussian distributions; K in coul, terfc."""
    (p, zp_, k1), (q, zq_, k2) = d1, d2
    rho = p * q / (p + q)
    r2 = (zp_ - zq_) ** 2
    pref = 2 * sp.pi ** sp.Rational(5, 2) * k1 * k2 / (p * q * sp.sqrt(p + q))
    coul = pref * F0(rho * r2)
    if kernel == "coul":
        return coul
    phi2 = rho * w**2 / (rho + w**2)
    return coul - pref * sp.sqrt(phi2 / rho) * G0(phi2 * r2, phi2 * r0**2)


def compile_int(expr, centres):
    val = sp.lambdify(ARGS, expr, modules=MODULES)
    ders = [sp.lambdify(ARGS, sp.diff(expr, z), modules=MODULES) for z in centres]
    return val, ders


ATOM_Z0 = np.array([0.0, 1.4, 2.9])
MOVING = 1
AO_EXP = [0.6, 2.4]
AUX_EXP = [0.35, 1.1, 3.8]
INTS = None
H0 = None
H1 = None


def compiled():
    return {
        "4": compile_int(
            dist_int(pair(a, za, b, zb), pair(c, zc, d, zd), "coul"), (za, zb, zc, zd)
        ),
        "3": compile_int(
            dist_int(single(al, zp), pair(a, za, b, zb), "terfc"), (zp, za, zb)
        ),
        "2": compile_int(dist_int(single(al, zp), single(be, zq), "terfc"), (zp, zq)),
    }


def basis(x, exps):
    z = ATOM_Z0.copy()
    z[MOVING] += x
    return [(e, z[at], at) for at in range(len(z)) for e in exps]


BUILD_CACHE = {}


def build(x, cutoffs):
    """S, dS, Coulomb ERIs and their x-derivative; terfc (P|mn), (P|Q) per cutoff."""
    key = (x, tuple(cutoffs))
    if key in BUILD_CACHE:
        return BUILD_CACHE[key]
    ao, aux = basis(x, AO_EXP), basis(x, AUX_EXP)
    n, m = len(ao), len(aux)
    s = np.zeros((n, n))
    ds = np.zeros((n, n))
    for i, j in itertools.product(range(n), range(n)):
        (e1, z1, a1), (e2, z2, a2) = ao[i], ao[j]
        p = e1 + e2
        val = (4 * e1 * e2 / p**2) ** 0.75 * math.exp(-e1 * e2 / p * (z1 - z2) ** 2)
        s[i, j] = val
        ds[i, j] = (
            val * (-2 * e1 * e2 / p * (z1 - z2)) * ((a1 == MOVING) - (a2 == MOVING))
        )
    out = {"s": s, "ds": ds}
    v4, d4 = INTS["4"]
    eri = np.zeros((n, n, n, n))
    deri = np.zeros_like(eri)
    for i, j, k, q in itertools.product(range(n), repeat=4):
        if (j, i, k, q) < (i, j, k, q) or (k, q, i, j) < (i, j, k, q):
            continue
        f = (ao[i][0], ao[j][0], ao[k][0], ao[q][0], 0.0, 0.0)
        f += (ao[i][1], ao[j][1], ao[k][1], ao[q][1], 0.0, 0.0, 1.0, 1.0)
        val = v4(*f)
        der = sum(dd(*f) for dd, idx in zip(d4, (i, j, k, q)) if ao[idx][2] == MOVING)
        for p1, p2 in itertools.product(((i, j), (j, i)), ((k, q), (q, k))):
            for perm in ((p1, p2), (p2, p1)):
                eri[perm[0] + perm[1]] = val
                deri[perm[0] + perm[1]] = der
    out["eri"], out["deri"] = eri, deri
    for rc in cutoffs:
        wv = 1.0 / (math.sqrt(2.0) * rc)
        v3, d3 = INTS["3"]
        u = np.zeros((m, n, n))
        du = np.zeros_like(u)
        for pp, i, j in itertools.product(range(m), range(n), range(n)):
            if j < i:
                u[pp, i, j], du[pp, i, j] = u[pp, j, i], du[pp, j, i]
                continue
            f = (ao[i][0], ao[j][0], 0.0, 0.0, aux[pp][0], 0.0)
            f += (ao[i][1], ao[j][1], 0.0, 0.0, aux[pp][1], 0.0, wv, rc)
            u[pp, i, j] = v3(*f)
            du[pp, i, j] = sum(
                dd(*f)
                for dd, cen in zip(d3, (aux[pp], ao[i], ao[j]))
                if cen[2] == MOVING
            )
        v2, d2 = INTS["2"]
        v = np.zeros((m, m))
        dv = np.zeros_like(v)
        for pp, qq in itertools.product(range(m), range(m)):
            f = (0.0, 0.0, 0.0, 0.0, aux[pp][0], aux[qq][0])
            f += (0.0, 0.0, 0.0, 0.0, aux[pp][1], aux[qq][1], wv, rc)
            v[pp, qq] = v2(*f)
            dv[pp, qq] = sum(
                dd(*f) for dd, cen in zip(d2, (aux[pp], aux[qq])) if cen[2] == MOVING
            )
        out[rc] = (u, du, v, dv)
    BUILD_CACHE[key] = out
    return out


def core(x):
    return H0 + x * H1


def jk(eri, dm):
    return np.einsum("mnls,ls->mn", eri, dm), np.einsum("mlns,ls->mn", eri, dm)


def diis_scf(s, h, fock_of, dens_of, nspin, tol=1e-13, max_iter=400):
    c0 = eigh(h, s)[1]
    cs = [c0] * nspin
    focks, errs = [], []
    for _ in range(max_iter):
        dd = dens_of(cs)
        f = fock_of(dd)
        err = np.concatenate(
            [(fk @ dk @ s - s @ dk @ fk).ravel() for fk, dk in zip(f, dd)]
        )
        if np.max(np.abs(err)) < tol:
            out = [eigh(fs, s) for fs in f]
            return [o[1] for o in out], [o[0] for o in out]
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
        fx = [sum(cf * fk[k] for cf, fk in zip(coef, focks)) for k in range(nspin)]
        cs = [eigh(fs, s)[1] for fs in fx]
    raise AssertionError("SCF did not converge")


def rhf(s, h, eri, nocc):
    def dens_of(cs):
        return [2 * cs[0][:, :nocc] @ cs[0][:, :nocc].T]

    def fock_of(dd):
        jm, km = jk(eri, dd[0])
        return [h + jm - 0.5 * km]

    cs, eps = diis_scf(s, h, fock_of, dens_of, 1)
    dm = dens_of(cs)[0]
    f = fock_of([dm])[0]
    return 0.5 * np.sum(dm * (h + f)), cs[0], eps[0]


def uhf(s, h, eri, na, nb):
    def dens_of(cs):
        return [cs[0][:, :na] @ cs[0][:, :na].T, cs[1][:, :nb] @ cs[1][:, :nb].T]

    def fock_of(dd):
        jt = jk(eri, dd[0] + dd[1])[0]
        return [h + jt - jk(eri, dd[0])[1], h + jt - jk(eri, dd[1])[1]]

    cs, eps = diis_scf(s, h, fock_of, dens_of, 2)
    dd = dens_of(cs)
    f = fock_of(dd)
    return 0.5 * sum(np.sum(dk * (h + fk)) for dk, fk in zip(dd, f)), cs, eps


def inv_sqrt(v):
    ev, q = np.linalg.eigh(v)
    return q @ np.diag(ev**-0.5) @ q.T


def denom(e1, e2, o1, o2, fc):
    eo1, ev1 = e1[fc:o1], e1[o1:]
    eo2, ev2 = e2[fc:o2], e2[o2:]
    return (
        eo1[:, None, None, None]
        + eo2[None, :, None, None]
        - ev1[None, None, :, None]
        - ev2[None, None, None, :]
    )


# ================================================================ closed shell
NOCC = 3


def closed_amplitudes(c, eps, u, v, fc):
    vm12 = inv_sqrt(v)
    b_full = np.einsum("PQ,Qmn,mp,nq->Ppq", vm12, u, c, c)
    g = np.einsum("Pia,Pjb->ijab", b_full[:, fc:NOCC, NOCC:], b_full[:, fc:NOCC, NOCC:])
    t = g / denom(eps, eps, NOCC, NOCC, fc)
    return g, t, b_full, vm12


def closed_energy(x, fc, r1=R0_1, r2=R0_2):
    """Eq. 12 from E_OS and E_SS directly (scs.rs), never from the term list."""
    m = build(x, (r1, r2))
    e_hf, c, eps = rhf(m["s"], core(x), m["eri"], NOCC)
    comp = {}
    for rc in (r1, r2):
        u, _, v, _ = m[rc]
        g, t, _, _ = closed_amplitudes(c, eps, u, v, fc)
        comp[rc] = (np.sum(g * t), np.sum(g * (t - t.transpose(0, 1, 3, 2))))
    return e_hf + C_OS * comp[r1][0] + C_SS * (comp[r2][1] - comp[r1][1])


def closed_gradient(x, fc, term_list):
    """rimp2_terms_gradient_timed on the model at x."""
    cut = sorted({tm[0] for tm in term_list})
    m = build(x, cut)
    s, h, ds, dh = m["s"], core(x), m["ds"], H1
    eri_ref, deri_ref = m["eri"], m["deri"]
    _, c, eps = rhf(s, h, eri_ref, NOCC)
    nocc, nmo = NOCC, c.shape[1]
    nv = nmo - nocc
    imat_sum = np.zeros((nmo, nmo))
    p_oo_sum = np.zeros((nocc - fc, nocc - fc))
    p_vv_sum = np.zeros((nv, nv))
    grad = 0.0
    for rc, a_, b_ in term_list:
        u, du, v, dv = m[rc]
        _, t, b_full, vm12 = closed_amplitudes(c, eps, u, v, fc)
        tt = (a_ + b_) * t - b_ * t.transpose(0, 1, 3, 2)
        p_oo_sum += -np.einsum("ikab,jkab->ij", t, tt)
        p_vv_sum += np.einsum("ijac,ijbc->ab", t, tt)
        xov = np.einsum("ijab,Pjb->Pia", tt, b_full[:, fc:nocc, nocc:])
        imat = np.zeros((nmo, nmo))
        imat[:, fc:nocc] = -2 * np.einsum("Pia,Pqa->qi", xov, b_full[:, :, nocc:])
        imat[:, nocc:] = -2 * np.einsum("Pia,Pqi->qa", xov, b_full[:, :, fc:nocc])
        imat_sum = imat_sum + imat  # every term's Imat
        # 3c/2c RI response at fixed C, on the term's cutoff.
        u_ia = np.einsum("Qmn,mi,na->Qia", u, c[:, fc:nocc], c[:, nocc:])
        du_ia = np.einsum("Qmn,mi,na->Qia", du, c[:, fc:nocc], c[:, nocc:])
        y = np.einsum("PQ,Qia->Pia", vm12, xov)
        m_u = np.einsum("PQ,Qia->Pia", np.linalg.inv(v), u_ia)
        grad += 4 * np.sum(y * du_ia) - 2 * np.einsum("PQ,Pia,Qia->", dv, m_u, y)
    dm1 = np.zeros((nmo, nmo))
    dm1[fc:nocc, fc:nocc] = p_oo_sum + p_oo_sum.T
    dm1[nocc:, nocc:] = p_vv_sum + p_vv_sum.T
    for k in range(fc):
        for i in range(fc, nocc):
            dm1[k, i] = dm1[i, k] = imat_sum[k, i] / (eps[k] - eps[i])
    jc, kc = jk(eri_ref, c @ dm1 @ c.T)
    veff_mo = c.T @ (2 * jc - kc) @ c
    xvo = veff_mo[nocc:, :nocc] + imat_sum[:nocc, nocc:].T - imat_sum[nocc:, :nocc]
    co, cv = c[:, :nocc], c[:, nocc:]
    eri_z = eri_ref  # Z-vector coupling kernel

    def a_z(zvec):
        zm = zvec.reshape(nv, nocc)
        dz = cv @ zm @ co.T + co @ zm.T @ cv.T
        jz, kz = jk(eri_z, dz)
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
    ip = imat_sum.copy()
    ip[nocc:, :nocc] = imat_sum[:nocc, nocc:].T
    im1 = c @ ip @ c.T
    jr, kr = jk(eri_ref, d_rel)
    p_occ = co @ co.T
    vhf = p_occ @ (2 * jr - kr) @ p_occ
    grad += np.sum((d_hf + d_rel) * dh) + np.sum((im1 - zeta_ao - vhf) * ds)
    jd, kd = jk(deri_ref, d_hf + 2 * d_rel)
    grad += 0.5 * np.sum(d_hf * (jd - 0.5 * kd))
    return grad


def part_c():
    print("C: closed-shell SCS-MP2(2terfc) gradient, one Z-vector, vs re-solved FD")
    x0 = 0.05
    for fc in (0, 1):
        g = closed_gradient(x0, fc, terms(C_OS, C_SS, R0_1, R0_2))
        fd = richardson(lambda xx: closed_energy(xx, fc), x0, 2e-3)
        report("C", f"frozen core {fc}", abs(g - fd) / max(1.0, abs(fd)), 1e-8)


# ================================================================ open shell
NOCC_U = (3, 2)


def u_amplitudes(cs, eps, u, v, nocc, fc):
    vm12 = inv_sqrt(v)
    bfull = [np.einsum("PQ,Qmn,mp,nq->Ppq", vm12, u, ck, ck) for ck in cs]
    bov = [bf[:, fc:n, n:] for bf, n in zip(bfull, nocc)]
    g = lambda b1, b2: np.einsum("Pia,Pjb->ijab", b1, b2)  # noqa: E731
    gaa, gbb, gab = g(bov[0], bov[0]), g(bov[1], bov[1]), g(bov[0], bov[1])
    taa = (gaa - gaa.transpose(0, 1, 3, 2)) / denom(
        eps[0], eps[0], nocc[0], nocc[0], fc
    )
    tbb = (gbb - gbb.transpose(0, 1, 3, 2)) / denom(
        eps[1], eps[1], nocc[1], nocc[1], fc
    )
    tab = gab / denom(eps[0], eps[1], nocc[0], nocc[1], fc)
    e_ss = 0.5 * np.sum(gaa * taa) + 0.5 * np.sum(gbb * tbb)
    e_os = np.sum(gab * tab)
    return (e_os, e_ss), (taa, tbb, tab), bfull, bov, vm12


def open_energy(x, fc, nocc, scs=True):
    """Eq. 12 on a UHF from E_OS = E_ab and E_SS = E_aa + E_bb; or terfc RI-MP2."""
    m = build(x, (R0_1, R0_2))
    e_hf, cs, eps = uhf(m["s"], core(x), m["eri"], *nocc)
    comp = {}
    for rc in (R0_1, R0_2):
        u, _, v, _ = m[rc]
        comp[rc] = u_amplitudes(cs, eps, u, v, nocc, fc)[0]
    if not scs:
        return e_hf + sum(comp[R0_2])
    return e_hf + C_OS * comp[R0_1][0] + C_SS * (comp[R0_2][1] - comp[R0_1][1])


def open_gradient(x, fc, nocc, term_list):
    """u_terms_gradient_timed on the model at x."""
    m = build(x, (R0_1, R0_2))
    s, h, ds, dh = m["s"], core(x), m["ds"], H1
    eri_ref, deri_ref = m["eri"], m["deri"]
    _, cs, eps = uhf(s, h, eri_ref, *nocc)
    nmo = cs[0].shape[1]
    imat = [np.zeros((nmo, nmo)) for _ in range(2)]
    p_oo = [np.zeros((n - fc, n - fc)) for n in nocc]
    p_vv = [np.zeros((nmo - n, nmo - n)) for n in nocc]
    grad = 0.0
    for rc, a_, b_ in term_list:
        u, du, v, dv = m[rc]
        _, (taa, tbb, tab), bfull, bov, vm12 = u_amplitudes(cs, eps, u, v, nocc, fc)
        w_ss, w_os = b_, a_  # (same-spin, opposite-spin) weights
        xs = [
            w_ss * np.einsum("ijab,Pjb->Pia", taa, bov[0])
            + w_os * np.einsum("iJaB,PJB->Pia", tab, bov[1]),
            w_ss * np.einsum("IJAB,PJB->PIA", tbb, bov[1])
            + w_os * np.einsum("iJaB,Pia->PJB", tab, bov[0]),
        ]
        p_oo[0] += -0.5 * w_ss * np.einsum(
            "ikab,jkab->ij", taa, taa
        ) - w_os * np.einsum("iKaB,jKaB->ij", tab, tab)
        p_oo[1] += -0.5 * w_ss * np.einsum(
            "ikab,jkab->ij", tbb, tbb
        ) - w_os * np.einsum("kIaB,kJaB->IJ", tab, tab)
        p_vv[0] += 0.5 * w_ss * np.einsum("ijac,ijbc->ab", taa, taa) + w_os * np.einsum(
            "iJaC,iJbC->ab", tab, tab
        )
        p_vv[1] += 0.5 * w_ss * np.einsum("ijac,ijbc->ab", tbb, tbb) + w_os * np.einsum(
            "iJcA,iJcB->AB", tab, tab
        )
        vinv = np.linalg.inv(v)
        for k in range(2):
            n = nocc[k]
            imat[k][:, fc:n] -= np.einsum("Pia,Pqa->qi", xs[k], bfull[k][:, :, n:])
            imat[k][:, n:] -= np.einsum("Pia,Pqi->qa", xs[k], bfull[k][:, :, fc:n])
            co, cv = cs[k][:, fc:n], cs[k][:, n:]
            u_ia = np.einsum("Qmn,mi,na->Qia", u, co, cv)
            du_ia = np.einsum("Qmn,mi,na->Qia", du, co, cv)
            yk = np.einsum("PQ,Qia->Pia", vm12, xs[k])
            m_u = np.einsum("PQ,Qia->Pia", vinv, u_ia)
            grad += 2 * np.sum(yk * du_ia) - np.einsum("PQ,Pia,Qia->", dv, m_u, yk)
    dm = []
    for k in range(2):
        n = nocc[k]
        dk = np.zeros((nmo, nmo))
        dk[fc:n, fc:n] = 0.5 * (p_oo[k] + p_oo[k].T)
        dk[n:, n:] = 0.5 * (p_vv[k] + p_vv[k].T)
        for kk in range(fc):
            for i in range(fc, n):
                dk[kk, i] = dk[i, kk] = imat[k][kk, i] / (eps[k][kk] - eps[k][i])
        dm.append(dk)

    def veff(dao):
        jt = jk(eri_ref, dao[0] + dao[1])[0]
        return [jt - jk(eri_ref, dao[0])[1], jt - jk(eri_ref, dao[1])[1]]

    vp = veff([cs[k] @ dm[k] @ cs[k].T for k in range(2)])
    xvo = []
    for k in range(2):
        n = nocc[k]
        vmo = cs[k].T @ vp[k] @ cs[k]
        xvo.append(vmo[n:, :n] + imat[k][:n, n:].T - imat[k][n:, :n])
    dims = [(nmo - n) * n for n in nocc]

    def split(zvec):
        return [
            zvec[: dims[0]].reshape(nmo - nocc[0], nocc[0]),
            zvec[dims[0] :].reshape(nmo - nocc[1], nocc[1]),
        ]

    def a_z(zvec):
        z = split(zvec)
        dz = []
        for k in range(2):
            n = nocc[k]
            t = cs[k][:, n:] @ z[k] @ cs[k][:, :n].T
            dz.append(t + t.T)
        coupled = veff(dz)
        out = []
        for k in range(2):
            n = nocc[k]
            gap = eps[k][n:, None] - eps[k][None, :n]
            out.append((gap * z[k] + (cs[k].T @ coupled[k] @ cs[k])[n:, :n]).ravel())
        return np.concatenate(out)

    amat = np.array([a_z(e) for e in np.eye(sum(dims))]).T
    z = split(np.linalg.solve(amat, -np.concatenate([xv.ravel() for xv in xvo])))
    z[1] = 1.0 * z[1]  # beta Z-vector
    d_hf, d_rel, wmat = [], [], np.zeros_like(s)
    for k in range(2):
        n = nocc[k]
        dk = dm[k].copy()
        dk[n:, :n] = z[k]
        dk[:n, n:] = z[k].T
        dm[k] = dk
        d_rel.append(cs[k] @ dk @ cs[k].T)
        co = cs[k][:, :n]
        d_hf.append(co @ co.T)
    vr = veff(d_rel)
    for k in range(2):
        n, e, ckk = nocc[k], eps[k], cs[k]
        zeta = 0.5 * (e[:, None] + e[None, :]) * dm[k]
        zeta[n:, :n] = e[None, :n] * dm[k][n:, :n]
        zeta[:n, n:] = e[:n, None] * dm[k][:n, n:]
        ip = imat[k].copy()
        ip[n:, :n] = imat[k][:n, n:].T
        co = ckk[:, :n]
        p_occ = co @ co.T
        wmat += ckk @ (ip - zeta) @ ckk.T - co @ np.diag(e[:n]) @ co.T
        wmat -= p_occ @ vr[k] @ p_occ
    dtot = d_hf[0] + d_hf[1] + d_rel[0] + d_rel[1]
    grad += np.sum(dtot * dh) + np.sum(wmat * ds)
    y = [d_hf[k] + 2 * d_rel[k] for k in range(2)]
    grad += 0.5 * np.sum((d_hf[0] + d_hf[1]) * jk(deri_ref, y[0] + y[1])[0])
    grad -= 0.5 * sum(np.sum(d_hf[k] * jk(deri_ref, y[k])[1]) for k in range(2))
    return grad


def part_u():
    print("U: open-shell gradients (UHF, n_a = 3, n_b = 2) vs re-solved FD")
    x0 = 0.05
    for fc in (0, 1):
        g = open_gradient(x0, fc, NOCC_U, terms(C_OS, C_SS, R0_1, R0_2))
        fd = richardson(lambda xx: open_energy(xx, fc, NOCC_U), x0, 2e-3)
        report(
            "U",
            f"SCS-MP2(2terfc), frozen core {fc}",
            abs(g - fd) / max(1.0, abs(fd)),
            1e-8,
        )
    g = open_gradient(x0, 0, NOCC_U, [(R0_2, 1.0, 1.0)])
    fd = richardson(lambda xx: open_energy(xx, 0, NOCC_U, scs=False), x0, 2e-3)
    report("U", "terfc RI-MP2", abs(g - fd) / max(1.0, abs(fd)), 1e-8)


def part_l():
    print("L: limits")
    x0 = 0.05
    tl = terms(C_OS, C_SS, R0_1, R0_2)
    g_c = closed_gradient(x0, 1, tl)
    g_u = open_gradient(x0, 1, (NOCC, NOCC), tl)
    report(
        "L",
        "open-shell assembly at n_a = n_b == closed",
        abs(g_u - g_c) / max(1.0, abs(g_c)),
        1e-10,
    )
    same = terms(C_OS, C_SS, R0_1, R0_1)
    sos = [(R0_1, C_OS, 0.0)]
    d = abs(closed_gradient(x0, 0, same) - closed_gradient(x0, 0, sos))
    report("L", "r0(1) = r0(2): the SOS gradient", d, 1e-12)
    gap = abs(closed_gradient(x0, 0, tl) - closed_gradient(x0, 0, sos))
    report("L", "non-vacuity: 1e-8 / |g_2terfc - g_SOS|", 1e-8 / gap, 1.0)


def main():
    global INTS, H0, H1
    part_s()
    INTS = compiled()
    n = len(AO_EXP) * len(ATOM_Z0)
    m0 = RNG.standard_normal((n, n))
    m1 = RNG.standard_normal((n, n))
    H0 = np.diag(np.linspace(-5.0, 1.0, n)) + 0.1 * (m0 + m0.T)
    H1 = 0.1 * (m1 + m1.T)
    part_c()
    part_u()
    part_l()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): {FAILED}")
        sys.exit(1)
    print("all SCS-MP2(2terfc) and open-shell terfc gradient identities hold")


if __name__ == "__main__":
    main()
