#!/usr/bin/env python3
"""Proofs of the open-shell attenuated (erfc) RI-MP2 nuclear-gradient identities.

Theory: `book/src/15-correlated-gradients/07-u-rimp2-gradient.md#u-rimp2-gradient-attenuated-erfc-variant`. Code: crates/ferric-mp2/src/att_rimp2_gradient.rs
(u_att_rimp2_gradient_analytical, total_u_att_rimp2_gradient_on) over
crates/ferric-mp2/src/u_rimp2_gradient.rs (u_correlated_gradient_timed: the
correlation operator in the amplitudes, b_full, Imat and the RI response; the
reference's Coulomb operator in the Z-vector, veff and the two-electron
derivative; the scaling on x^s and the unrelaxed densities).

The energy is E = E_UHF[1/r] + s (E_aa + E_bb + E_ab)[erfc(w r)/r]: Coulomb UHF
orbitals and orbital energies, and the correlation's 3-centre integrals (P|mn)
AND metric (P|Q) both under erfc.

  K  The sympy closed forms of the s-Gaussian Coulomb and erfc integrals
     (`book/src/15-correlated-gradients/02-rimp2-gradient.md#rimp2-gradient-attenuated-erfc-ri-mp2`) give the model's integrals: the 2-centre (P|erfc|Q)
     closed form equals an independent quadrature over t of
     erfc(w r)/r = (2/sqrt pi) int_w^inf exp(-t^2 r^2) dt, and sympy's
     centre derivatives of the 3- and 2-centre closed forms equal Richardson
     differences of the closed forms.
  U  The assembled gradient on a 1-D model with real integrals: three atoms
     on the z axis, two s Gaussians per atom, a smooth model core
     Hamiltonian, Coulomb four-centre ERIs for the UHF (n_alpha = 3,
     n_beta = 2), erfc (P|mn) and (P|Q) on three aux s Gaussians per atom
     for the correlation; the parameter x is the z of the middle atom. The
     UHF is solved to 1e-13 at every x. The analytic Lagrangian of
     u_correlated_gradient_timed (per-spin x^s, Imat^s, unrelaxed P^s, the
     coupled UHF Z-vector on the Coulomb J/K, the two-electron derivative
     U((D^a, D^b), (D^a + 2P^a, D^b + 2P^b)) on the Coulomb ERIs, one erfc RI
     response channel per spin) equals the Richardson derivative of the
     re-solved E_UHF + s E_c[erfc], for s = 1 and 1.2, with and without one
     frozen core. Independent route: the energy is re-solved from scratch.
  C  Closed-shell limit: for n_alpha = n_beta the open-shell assembly equals
     the closed-shell attenuated assembly of
     scripts/att_rimp2_gradient_sympy.py part L (re-derived here), for
     s = 1.2 and with and without a frozen core.
  Z  At w = 1e-6 the open-shell attenuated gradient equals the U-RI-MP2
     gradient of the same model (u, V under 1/r) to 1e-9 relative; at the
     production w = 0.2222 Bohr^-1 it differs from it by far more.

Mutations (each turns the script red; the runner applies them one at a time
as exact substitutions of the tagged lines):
  * Coulomb in the correlation pieces (amplitudes, b_full, Imat, RI response)
    breaks U ("FAIL U").
  * erfc in the reference's two-electron derivative term breaks U ("FAIL U").
  * erfc in the Z-vector coupling and veff (the reference's J/K) breaks U
    ("FAIL U").
  * the beta spin block of the relaxed density dropped from the
    one-electron term breaks U ("FAIL U").
  * the beta Z-vector dropped (z^beta = 0) breaks U ("FAIL U").
  * the scaling s left off the unrelaxed densities breaks U ("FAIL U").

Run: python3 scripts/u_att_rimp2_gradient_sympy.py  (needs sympy, numpy, scipy)
"""

import itertools
import math
import sys

import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.linalg import eigh

FAILED = []
RNG = np.random.default_rng(20261116)
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


def dist_int(d1, d2, kernel):
    (p, zp_, k1), (q, zq_, k2) = d1, d2
    rho = p * q / (p + q)
    r2 = (zp_ - zq_) ** 2
    pref = 2 * sp.pi ** sp.Rational(5, 2) * k1 * k2 / (p * q * sp.sqrt(p + q))
    coul = pref * F0(rho * r2)
    if kernel == "coul":
        return coul
    rho_w = rho * w**2 / (rho + w**2)
    return coul - pref * sp.sqrt(w**2 / (w**2 + rho)) * F0(rho_w * r2)


def eri3_expr(kernel):
    return dist_int(single(al, zp), pair(a, za, b, zb), kernel)


def eri2_expr(kernel):
    return dist_int(single(al, zp), single(be, zq), kernel)


def eri4_expr(kernel):
    return dist_int(pair(a, za, b, zb), pair(c, zc, d, zd), kernel)


ARGS = (a, b, c, d, al, be, za, zb, zc, zd, zp, zq, w)


def compile_int(expr, centres):
    """Value and per-centre derivatives, lambdified."""
    val = sp.lambdify(ARGS, expr, modules=MODULES)
    ders = [sp.lambdify(ARGS, sp.diff(expr, z), modules=MODULES) for z in centres]
    return val, ders


# ================================================================ part K
def part_k(ints):
    print("K: the model's integrals (closed forms vs independent routes)")
    i2 = ints[("2", "erfc")][0]
    worst = 0.0
    for alv, bev, r, wv in [(0.4, 1.2, 1.3, OMEGA), (2.0, 0.7, 0.0, 0.5)]:
        n1, n2 = (2 * alv / np.pi) ** 0.75, (2 * bev / np.pi) ** 0.75
        rho = alv * bev / (alv + bev)

        def kern(t):
            g = (np.pi**2 / (alv * bev)) ** 1.5 * (rho / (rho + t * t)) ** 1.5
            return g * np.exp(-rho * t * t / (rho + t * t) * r * r)

        ref = n1 * n2 * 2 / np.sqrt(np.pi) * quad(kern, wv, np.inf, epsabs=1e-14)[0]
        got = i2(0, 0, 0, 0, alv, bev, 0, 0, 0, 0, 0.0, r, wv)
        worst = max(worst, abs(got - ref) / abs(ref))
    report("K", "(P|erfc|Q) closed form vs t-quadrature", worst, 1e-10)
    worst = 0.0
    for key, slots in [(("3", "erfc"), (10, 6, 7)), (("2", "erfc"), (10, 11))]:
        val, ders = ints[key]
        args = [0.8, 1.9, 0.0, 0.0, 0.6, 1.4, 0.3, -0.4, 0.0, 0.0, 1.1, 0.2, OMEGA]
        for slot, der in zip(slots, ders):

            def shifted(h, slot=slot):
                aa = list(args)
                aa[slot] += h
                return val(*aa)

            worst = max(worst, abs(der(*args) - richardson(shifted, 0.0, 1e-3)))
    report("K", "d(P|erfc|mn), d(P|erfc|Q) vs Richardson", worst, 1e-10)


# ================================================================ model
ATOM_Z0 = np.array([0.0, 1.4, 2.9])
MOVING = 1
AO_EXP = [0.6, 2.4]
AUX_EXP = [0.35, 1.1, 3.8]
NOCC = (3, 2)
INTS = None
H0 = None
H1 = None


def compiled():
    out = {}
    for kern in ("coul", "erfc"):
        out[("4", kern)] = compile_int(eri4_expr(kern), (za, zb, zc, zd))
        out[("3", kern)] = compile_int(eri3_expr(kern), (zp, za, zb))
        out[("2", kern)] = compile_int(eri2_expr(kern), (zp, zq))
    return out


def basis(x, exps):
    z = ATOM_Z0.copy()
    z[MOVING] += x
    return [(e, z[at], at) for at in range(len(z)) for e in exps]


def build(x, omega):
    """S, dS; Coulomb and erfc ERIs, (P|mn), (P|Q) and their x-derivatives."""
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
    for kern in ("coul", "erfc"):
        v4, d4 = INTS[("4", kern)]
        eri = np.zeros((n, n, n, n))
        deri = np.zeros_like(eri)
        for i, j, k, q in itertools.product(range(n), repeat=4):
            f = (ao[i][0], ao[j][0], ao[k][0], ao[q][0], 0.0, 0.0)
            f += (ao[i][1], ao[j][1], ao[k][1], ao[q][1], 0.0, 0.0, omega)
            eri[i, j, k, q] = v4(*f)
            deri[i, j, k, q] = sum(
                dd(*f) for dd, idx in zip(d4, (i, j, k, q)) if ao[idx][2] == MOVING
            )
        out[f"eri_{kern}"], out[f"deri_{kern}"] = eri, deri
        v3, d3 = INTS[("3", kern)]
        u = np.zeros((m, n, n))
        du = np.zeros_like(u)
        for pp, i, j in itertools.product(range(m), range(n), range(n)):
            f = (ao[i][0], ao[j][0], 0.0, 0.0, aux[pp][0], 0.0)
            f += (ao[i][1], ao[j][1], 0.0, 0.0, aux[pp][1], 0.0, omega)
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
            f = (0.0, 0.0, 0.0, 0.0, aux[pp][0], aux[qq][0])
            f += (0.0, 0.0, 0.0, 0.0, aux[pp][1], aux[qq][1], omega)
            v[pp, qq] = v2(*f)
            dv[pp, qq] = sum(
                dd(*f) for dd, cen in zip(d2, (aux[pp], aux[qq])) if cen[2] == MOVING
            )
        out[f"u_{kern}"], out[f"du_{kern}"] = u, du
        out[f"v_{kern}"], out[f"dv_{kern}"] = v, dv
    return out


def jk(eri, dm):
    return np.einsum("mnls,ls->mn", eri, dm), np.einsum("mlns,ls->mn", eri, dm)


def uhf(s, h, eri, na, nb, tol=1e-13, max_iter=400):
    """Converged UHF by DIIS: (E, [C_a, C_b], [eps_a, eps_b])."""

    def dens_of(cs):
        return [cs[0][:, :na] @ cs[0][:, :na].T, cs[1][:, :nb] @ cs[1][:, :nb].T]

    def fock_of(dd):
        jt = jk(eri, dd[0] + dd[1])[0]
        return [h + jt - jk(eri, dd[0])[1], h + jt - jk(eri, dd[1])[1]]

    c0 = eigh(h, s)[1]
    cs = [c0, c0]
    focks, errs = [], []
    for _ in range(max_iter):
        dd = dens_of(cs)
        f = fock_of(dd)
        err = np.concatenate(
            [(fk @ dk @ s - s @ dk @ fk).ravel() for fk, dk in zip(f, dd)]
        )
        if np.max(np.abs(err)) < tol:
            out = [eigh(fs, s) for fs in f]
            cs = [o[1] for o in out]
            dd = dens_of(cs)
            f = fock_of(dd)
            e = 0.5 * sum(np.sum(dk * (h + fk)) for dk, fk in zip(dd, f))
            return e, cs, [o[0] for o in out]
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
        fx = [sum(cf * fk[k] for cf, fk in zip(coef, focks)) for k in range(2)]
        cs = [eigh(fs, s)[1] for fs in fx]
    raise AssertionError("UHF did not converge")


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


def u_mp2(cs, eps, u, v, nocc, fc):
    """U-RI-MP2 correlation energy, amplitudes and full-MO factors per spin."""
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
    e = 0.5 * np.sum(gaa * taa) + 0.5 * np.sum(gbb * tbb) + np.sum(gab * tab)
    return e, (taa, tbb, tab), bfull, bov, vm12


def core(x):
    return H0 + x * H1


def total_energy(x, omega, nocc, fc, scale, corr="erfc"):
    m = build(x, omega)
    e_hf, cs, eps = uhf(m["s"], core(x), m["eri_coul"], *nocc)
    return e_hf + scale * u_mp2(cs, eps, m[f"u_{corr}"], m[f"v_{corr}"], nocc, fc)[0]


def u_gradient(x, omega, nocc, fc, scale, corr="erfc"):
    """u_correlated_gradient_timed on the model at x."""
    m = build(x, omega)
    s, h, ds, dh = m["s"], core(x), m["ds"], H1
    eri_ref = m["eri_coul"]  # Z-vector and veff operator
    deri_ref = m["deri_coul"]  # two-electron derivative operator
    ck = corr  # correlation operator of amplitudes, b_full, Imat, RI response
    u, v, du, dv = m[f"u_{ck}"], m[f"v_{ck}"], m[f"du_{ck}"], m[f"dv_{ck}"]
    _, cs, eps = uhf(s, h, m["eri_coul"], *nocc)
    _, (taa, tbb, tab), bfull, bov, vm12 = u_mp2(cs, eps, u, v, nocc, fc)
    xs = [
        np.einsum("ijab,Pjb->Pia", taa, bov[0])
        + np.einsum("iJaB,PJB->Pia", tab, bov[1]),
        np.einsum("IJAB,PJB->PIA", tbb, bov[1])
        + np.einsum("iJaB,Pia->PJB", tab, bov[0]),
    ]
    xs = [scale * xk for xk in xs]
    p_scale = scale  # scaling of the unrelaxed densities
    nmo = cs[0].shape[1]
    p_oo = [
        -0.5 * np.einsum("ikab,jkab->ij", taa, taa)
        - np.einsum("iKaB,jKaB->ij", tab, tab),
        -0.5 * np.einsum("ikab,jkab->ij", tbb, tbb)
        - np.einsum("kIaB,kJaB->IJ", tab, tab),
    ]
    p_vv = [
        0.5 * np.einsum("ijac,ijbc->ab", taa, taa)
        + np.einsum("iJaC,iJbC->ab", tab, tab),
        0.5 * np.einsum("ijac,ijbc->ab", tbb, tbb)
        + np.einsum("iJcA,iJcB->AB", tab, tab),
    ]
    imat, dm = [], []
    for k in range(2):
        n = nocc[k]
        im = np.zeros((nmo, nmo))
        im[:, fc:n] = -np.einsum("Pia,Pqa->qi", xs[k], bfull[k][:, :, n:])
        im[:, n:] = -np.einsum("Pia,Pqi->qa", xs[k], bfull[k][:, :, fc:n])
        dk = np.zeros((nmo, nmo))
        dk[fc:n, fc:n] = 0.5 * p_scale * (p_oo[k] + p_oo[k].T)
        dk[n:, n:] = 0.5 * p_scale * (p_vv[k] + p_vv[k].T)
        e = eps[k]
        for kk in range(fc):
            for i in range(fc, n):
                dk[kk, i] = dk[i, kk] = im[kk, i] / (e[kk] - e[i])
        imat.append(im)
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
    dtot = d_hf[0] + d_hf[1] + d_rel[0] + d_rel[1]  # relaxed density
    grad = np.sum(dtot * dh) + np.sum(wmat * ds)
    y = [d_hf[k] + 2 * d_rel[k] for k in range(2)]
    grad += 0.5 * np.sum((d_hf[0] + d_hf[1]) * jk(deri_ref, y[0] + y[1])[0])
    grad -= 0.5 * sum(np.sum(d_hf[k] * jk(deri_ref, y[k])[1]) for k in range(2))
    vinv = np.linalg.inv(v)
    for k in range(2):
        n = nocc[k]
        co, cv = cs[k][:, fc:n], cs[k][:, n:]
        u_ia = np.einsum("Qmn,mi,na->Qia", u, co, cv)
        du_ia = np.einsum("Qmn,mi,na->Qia", du, co, cv)
        yk = np.einsum("PQ,Qia->Pia", vm12, xs[k])
        m_u = np.einsum("PQ,Qia->Pia", vinv, u_ia)
        grad += 2 * np.sum(yk * du_ia) - np.einsum("PQ,Pia,Qia->", dv, m_u, yk)
    return grad


def closed_gradient(x, omega, nocc, fc, scale):
    """The closed-shell attenuated assembly (att_rimp2_gradient_sympy.py part L)."""
    m = build(x, omega)
    s, h, ds, dh = m["s"], core(x), m["ds"], H1
    eri_ref, deri_ref = m["eri_coul"], m["deri_coul"]
    u, v, du, dv = m["u_erfc"], m["v_erfc"], m["du_erfc"], m["dv_erfc"]
    _, cs, eps = uhf(s, h, eri_ref, nocc, nocc)
    c, eps = cs[0], eps[0]
    nmo = c.shape[1]
    nv = nmo - nocc
    vm12 = inv_sqrt(v)
    b_full = np.einsum("PQ,Qmn,mp,nq->Ppq", vm12, u, c, c)
    ia = b_full[:, fc:nocc, nocc:]
    t = np.einsum("Pia,Pjb->ijab", ia, ia) / denom(eps, eps, nocc, nocc, fc)
    tt = scale * (2 * t - t.transpose(0, 1, 3, 2))
    p_oo = -np.einsum("ikab,jkab->ij", t, tt)
    p_vv = np.einsum("ijac,ijbc->ab", t, tt)
    xov = np.einsum("ijab,Pjb->Pia", tt, ia)
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
    xvo = veff_mo[nocc:, :nocc] + imat[:nocc, nocc:].T - imat[nocc:, :nocc]
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
    jr, kr = jk(eri_ref, d_rel)
    p_occ = co @ co.T
    vhf = p_occ @ (2 * jr - kr) @ p_occ
    grad = np.sum((d_hf + d_rel) * dh) + np.sum((c @ ip @ c.T - zeta_ao - vhf) * ds)
    jd, kd = jk(deri_ref, d_hf + 2 * d_rel)
    grad += 0.5 * np.sum(d_hf * (jd - 0.5 * kd))
    u_ia = np.einsum("Qmn,mi,na->Qia", u, c[:, fc:nocc], cv)
    du_ia = np.einsum("Qmn,mi,na->Qia", du, c[:, fc:nocc], cv)
    y = np.einsum("PQ,Qia->Pia", vm12, xov)
    m_u = np.einsum("PQ,Qia->Pia", np.linalg.inv(v), u_ia)
    grad += 4 * np.sum(y * du_ia) - 2 * np.einsum("PQ,Pia,Qia->", dv, m_u, y)
    return grad


# ================================================================ parts
def part_u():
    print("U: open-shell attenuated RI-MP2 gradient vs re-solved finite differences")
    x0 = 0.05
    for scale, fc in itertools.product((1.0, 1.2), (0, 1)):
        g = u_gradient(x0, OMEGA, NOCC, fc, scale)
        fd = richardson(lambda xx: total_energy(xx, OMEGA, NOCC, fc, scale), x0, 2e-3)
        report(
            "U", f"s = {scale}, frozen core {fc}", abs(g - fd) / max(1.0, abs(fd)), 1e-8
        )


def part_c():
    print("C: closed-shell limit (n_alpha = n_beta = 3) of the open-shell assembly")
    x0 = 0.05
    for fc in (0, 1):
        gu = u_gradient(x0, OMEGA, (3, 3), fc, 1.2)
        gc = closed_gradient(x0, OMEGA, 3, fc, 1.2)
        report(
            "C",
            f"g_U = g_closed, s = 1.2, frozen core {fc}",
            abs(gu - gc) / max(1.0, abs(gc)),
            1e-10,
        )


def part_z():
    print("Z: w -> 0 gives the U-RI-MP2 gradient")
    x0 = 0.05
    g_mp2 = u_gradient(x0, 1e-6, NOCC, 0, 1.0, corr="coul")
    g_att = u_gradient(x0, 1e-6, NOCC, 0, 1.0)
    report(
        "Z",
        "|g_att(w=1e-6) - g_U-RI-MP2|",
        abs(g_att - g_mp2) / max(1.0, abs(g_mp2)),
        1e-9,
    )
    gap = abs(
        u_gradient(x0, OMEGA, NOCC, 0, 1.0)
        - u_gradient(x0, OMEGA, NOCC, 0, 1.0, corr="coul")
    )
    report("Z", "non-vacuity: 1e-5 / |g_att(w=0.2222) - g_U-RI-MP2|", 1e-5 / gap, 1.0)


def main():
    global INTS, H0, H1
    INTS = compiled()
    part_k(INTS)
    n = len(AO_EXP) * len(ATOM_Z0)
    m0 = RNG.standard_normal((n, n))
    m1 = RNG.standard_normal((n, n))
    H0 = np.diag(np.linspace(-5.0, 1.0, n)) + 0.1 * (m0 + m0.T)
    H1 = 0.1 * (m1 + m1.T)
    part_u()
    part_c()
    part_z()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): {FAILED}")
        sys.exit(1)
    print("all open-shell attenuated RI-MP2 gradient identities hold")


if __name__ == "__main__":
    main()
