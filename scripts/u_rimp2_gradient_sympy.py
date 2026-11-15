#!/usr/bin/env python3
"""Proofs of the unrestricted RI-MP2 nuclear-gradient identities ferric-mp2 codes.

Theory: crates/ferric-mp2/theory.md, "15. Unrestricted RI-MP2 nuclear gradient".
Code: crates/ferric-mp2/src/u_rimp2_gradient.rs (u_energy_weights,
unrelaxed_density_mo, solve_u_zvector, u_rimp2_gradient_timed).

  S  Symbolic (sympy). With B^s_{P,ia} free symbols (one auxiliary function,
     n_alpha = n_beta = 2 occupied, 2 virtual per spin, fixed orbital
     energies), the U-MP2 energy
       E = 1/4 sum |<ij||ab>|^2/D (aa) + same (bb) + sum (ia|JB)^2/D (ab)
     has dE/dB^s = 2 x^s with
       x^a_{P,ia} = sum_jb t^aa_ijab B^a_{P,jb} + sum_JB t^ab_iJaB B^b_{P,JB},
       t^aa = [(ia|jb) - (ib|ja)]/D, t^ab = (ia|JB)/D (u_rimp2.rs layout),
     and likewise for beta (the alpha-beta block enters BOTH spins' weights).
  G  The assembled gradient on a model whose overlap S(x), core Hamiltonian
     h(x), four-centre ERIs (the reference SCF's J/K, a low-rank
     factorization) and RI factors u^P_mn(x), V(x) all move with one
     parameter x. The UHF (n_alpha = 3, n_beta = 2) is solved to 1e-13 at
     every x, and the code's Lagrangian
       Tr[D_tot h'] + sum (im1 - zeta - W_HF - vhf_s1occ) S'
       + d/dx U((D^a, D^b), (D^a + 2P^a, D^b + 2P^b)) + RI response
     with per-spin Imat^s = -sum x^s b^s, the coupled CPHF
       (e_a - e_i) z^s + [C_s^T (J[Dz_a + Dz_b] - K[Dz_s]) C_s]_ai = -Xvo^s
     and one RI-response channel of weight 1 per spin, equals the Richardson
     derivative of the re-solved E_UHF + E_UMP2 (G1, with and without a frozen
     core), and Tr[(D_HF + P_rel) R] equals dE/dlam for h -> h + lam R (G2).
     Independent route: the energy is re-solved from scratch at displaced x.
  C  Closed-shell limit: for n_alpha = n_beta the per-spin quantities are
     half of the closed-shell ones (Imat^s = Imat/2, P^s = P/2 total,
     z^s = z/2) and the gradient equals the closed-shell assembly of
     scripts/mp2_gradient_sympy.py part G (re-derived here).

Mutations (each turns the script red; the runner applies them one at a time
as exact substitutions of the tagged lines):
  * the alpha-beta block counted once (only in x^alpha) breaks S and G
    ("FAIL S", "FAIL G").
  * the beta Z-vector dropped (z^beta = 0) breaks G ("FAIL G").
  * the opposite-spin Coulomb coupling of the CPHF dropped
    (J[Dz_a + Dz_b] -> J[Dz_s]) breaks G ("FAIL G").
  * the RI response weight per spin 1 -> 2 breaks G ("FAIL G").

Run: python3 scripts/u_rimp2_gradient_sympy.py  (needs sympy, numpy, scipy)
"""

import itertools
import sys

import numpy as np
import sympy as sp
from scipy.linalg import eigh

FAILED = []
RNG = np.random.default_rng(20261115)

# The opposite-spin block enters x^beta with this weight (1: per spin).
AB_IN_X_BETA = 1.0  # alpha-beta weight in x^beta
# The RI 3c/2c response weight of each spin channel.
RI_SPIN_WEIGHT = 1.0  # RI response weight per spin


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


def energy_weights(ba, bb, taa, tbb, tab):
    """x^a, x^b of u_energy_weights (P, i, a layout)."""
    xa = np.einsum("ijab,Pjb->Pia", taa, ba) + np.einsum("iJaB,PJB->Pia", tab, bb)
    xb = np.einsum("IJAB,PJB->PIA", tbb, bb) + AB_IN_X_BETA * np.einsum(
        "iJaB,Pia->PJB", tab, ba
    )
    return xa, xb


# ================================================================ part S
def part_s():
    print("S: dE/dB^s = 2 x^s (symbolic)")
    no, nv = 2, 2
    ba = np.array(
        [[[sp.Symbol(f"a{i}{a}") for a in range(nv)] for i in range(no)]], dtype=object
    )
    bb = np.array(
        [[[sp.Symbol(f"b{i}{a}") for a in range(nv)] for i in range(no)]], dtype=object
    )
    ea = sp.symbols("ea0:4")
    eb = sp.symbols("eb0:4")

    def d(e1, e2, i, j, a, b):
        return e1[i] + e2[j] - e1[no + a] - e2[no + b]

    g = lambda b1, b2: np.einsum("Pia,Pjb->ijab", b1, b2)  # noqa: E731
    gaa, gbb, gab = g(ba, ba), g(bb, bb), g(ba, bb)
    idx = list(itertools.product(range(no), range(no), range(nv), range(nv)))
    taa = np.empty((no, no, nv, nv), dtype=object)
    tbb = np.empty_like(taa)
    tab = np.empty_like(taa)
    for i, j, a, b in idx:
        taa[i, j, a, b] = (gaa[i, j, a, b] - gaa[i, j, b, a]) / d(ea, ea, i, j, a, b)
        tbb[i, j, a, b] = (gbb[i, j, a, b] - gbb[i, j, b, a]) / d(eb, eb, i, j, a, b)
        tab[i, j, a, b] = gab[i, j, a, b] / sp.Add(
            ea[i], eb[j], -ea[no + a], -eb[no + b]
        )
    e = sum(
        sp.Rational(1, 4) * (gaa[q] - gaa[q[0], q[1], q[3], q[2]]) * taa[q]
        + sp.Rational(1, 4) * (gbb[q] - gbb[q[0], q[1], q[3], q[2]]) * tbb[q]
        + gab[q] * tab[q]
        for q in idx
    )
    xa, xb = energy_weights(ba, bb, taa, tbb, tab)
    bad = 0
    for i, a in itertools.product(range(no), range(nv)):
        bad += sp.simplify(sp.diff(e, ba[0, i, a]) - 2 * xa[0, i, a]) != 0
        bad += sp.simplify(sp.diff(e, bb[0, i, a]) - 2 * xb[0, i, a]) != 0
    print(f"  {'ok  ' if bad == 0 else 'FAIL S'} dE/dB^a = 2 x^a and dE/dB^b = 2 x^b")
    if bad:
        FAILED.append("S: dE/dB")


# ================================================================ model
NBF, NAUX, NL = 7, 10, 8


def make_model():
    def rsym(n, scale):
        m = RNG.standard_normal((n, n))
        return scale * (m + m.T)

    m = {}
    m["s1"] = rsym(NBF, 0.03)
    m["s2"] = rsym(NBF, 0.01)
    m["h0"] = np.diag(np.linspace(-4.0, 1.5, NBF)) + rsym(NBF, 0.15)
    m["h1"] = rsym(NBF, 0.2)
    m["l0"] = np.array([rsym(NBF, 0.12) for _ in range(NL)])
    m["l1"] = np.array([rsym(NBF, 0.05) for _ in range(NL)])
    m["u0"] = np.array([rsym(NBF, 0.12) for _ in range(NAUX)])
    m["u1"] = np.array([rsym(NBF, 0.05) for _ in range(NAUX)])
    a = RNG.standard_normal((NAUX, NAUX))
    m["v0"] = a @ a.T / NAUX + np.eye(NAUX)
    m["v1"] = rsym(NAUX, 0.05)
    return m


MODEL = make_model()


def inputs(x):
    m = MODEL
    s = np.eye(NBF) + x * m["s1"] + x * x * m["s2"]
    h = m["h0"] + x * m["h1"]
    lf = m["l0"] + x * m["l1"]
    eri = np.einsum("Kmn,Kls->mnls", lf, lf)
    return s, h, eri, m["u0"] + x * m["u1"], m["v0"] + x * m["v1"]


def d_inputs(x):
    m = MODEL
    lf = m["l0"] + x * m["l1"]
    deri = np.einsum("Kmn,Kls->mnls", m["l1"], lf) + np.einsum(
        "Kmn,Kls->mnls", lf, m["l1"]
    )
    return m["s1"] + 2 * x * m["s2"], m["h1"], deri, m["u1"], m["v1"]


def jk(eri, d):
    return np.einsum("mnls,ls->mn", eri, d), np.einsum("mlns,ls->mn", eri, d)


def diis_scf(s, fock_of, dens_of, c0, tol=1e-13, max_iter=400):
    c = c0
    focks, errs = [], []
    for _ in range(max_iter):
        d = dens_of(c)
        f = fock_of(d)
        err = np.concatenate(
            [(fk @ dk @ s - s @ dk @ fk).ravel() for fk, dk in zip(f, d)]
        )
        if np.max(np.abs(err)) < tol:
            return f
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
        fx = [sum(cf * fk[k] for cf, fk in zip(coef, focks)) for k in range(len(f))]
        c = [eigh(fs, s)[1] for fs in fx]
    raise AssertionError("SCF did not converge")


def uhf(s, h, eri, na, nb):
    """Converged UHF: (E, [C_a, C_b], [eps_a, eps_b])."""

    def dens_of(c):
        return [c[0][:, :na] @ c[0][:, :na].T, c[1][:, :nb] @ c[1][:, :nb].T]

    def fock_of(d):
        jt = jk(eri, d[0] + d[1])[0]
        return [h + jt - jk(eri, d[0])[1], h + jt - jk(eri, d[1])[1]]

    c0 = eigh(h, s)[1]
    f = diis_scf(s, fock_of, dens_of, [c0, c0])
    out = [eigh(fs, s) for fs in f]
    eps = [o[0] for o in out]
    c = [o[1] for o in out]
    d = dens_of(c)
    f = fock_of(d)
    e = 0.5 * sum(np.sum(dk * (h + fk)) for dk, fk in zip(d, f))
    return e, c, eps


def inv_sqrt(v):
    w, q = np.linalg.eigh(v)
    return q @ np.diag(w**-0.5) @ q.T


def denom(e1, e2, o1, o2, fc):
    eo1, ev1 = e1[fc:o1], e1[o1:]
    eo2, ev2 = e2[fc:o2], e2[o2:]
    return (
        eo1[:, None, None, None]
        + eo2[None, :, None, None]
        - ev1[None, None, :, None]
        - ev2[None, None, None, :]
    )


def u_mp2(c, eps, u, v, nocc, fc):
    """U-RI-MP2 energy, amplitudes and full-MO factors of each spin."""
    vm12 = inv_sqrt(v)
    bfull = [np.einsum("PQ,Qmn,mp,nq->Ppq", vm12, u, cs, cs) for cs in c]
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


def total_energy(x, nocc, fc, dh=None, lam=0.0):
    s, h, eri, u, v = inputs(x)
    if dh is not None:
        h = h + lam * dh
    e_hf, c, eps = uhf(s, h, eri, *nocc)
    return e_hf + u_mp2(c, eps, u, v, nocc, fc)[0]


def u_gradient(x, nocc, fc, dh_probe):
    """The code's assembly (u_rimp2_gradient_timed) at x."""
    s, h, eri, u, v = inputs(x)
    ds, dh, deri, du, dv = d_inputs(x)
    _, c, eps = uhf(s, h, eri, *nocc)
    _, (taa, tbb, tab), bfull, bov, vm12 = u_mp2(c, eps, u, v, nocc, fc)
    xs = energy_weights(bov[0], bov[1], taa, tbb, tab)
    nmo = c[0].shape[1]
    # Unrelaxed spin densities (build_u_mp2_density) and Imat^s.
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
    for sp_ in range(2):
        n = nocc[sp_]
        im = np.zeros((nmo, nmo))
        im[:, fc:n] = -np.einsum("Pia,Pqa->qi", xs[sp_], bfull[sp_][:, :, n:])
        im[:, n:] = -np.einsum("Pia,Pqi->qa", xs[sp_], bfull[sp_][:, :, fc:n])
        d = np.zeros((nmo, nmo))
        d[fc:n, fc:n] = 0.5 * (p_oo[sp_] + p_oo[sp_].T)
        d[n:, n:] = 0.5 * (p_vv[sp_] + p_vv[sp_].T)
        e = eps[sp_]
        for k in range(fc):
            for i in range(fc, n):
                d[k, i] = d[i, k] = im[k, i] / (e[k] - e[i])
        imat.append(im)
        dm.append(d)

    def veff(dao):
        jt = jk(eri, dao[0] + dao[1])[0]
        return [jt - jk(eri, dao[0])[1], jt - jk(eri, dao[1])[1]]

    p_ao = [c[k] @ dm[k] @ c[k].T for k in range(2)]
    vp = veff(p_ao)
    xvo = []
    for k in range(2):
        n = nocc[k]
        vmo = c[k].T @ vp[k] @ c[k]
        xvo.append(vmo[n:, :n] + imat[k][:n, n:].T - imat[k][n:, :n])
    # Coupled CPHF as one dense linear system over (z^a, z^b).
    dims = [(nmo - n) * n for n in nocc]

    def a_z(zvec):
        z = [
            zvec[: dims[0]].reshape(nmo - nocc[0], nocc[0]),
            zvec[dims[0] :].reshape(nmo - nocc[1], nocc[1]),
        ]
        dz = []
        for k in range(2):
            n = nocc[k]
            t = c[k][:, n:] @ z[k] @ c[k][:, :n].T
            dz.append(t + t.T)
        ja, ka = jk(eri, dz[0])
        jb, kb = jk(eri, dz[1])
        coupled = [ja + jb - ka, ja + jb - kb]  # CPHF opposite-spin Coulomb
        out = []
        for k in range(2):
            n = nocc[k]
            gap = eps[k][n:, None] - eps[k][None, :n]
            out.append((gap * z[k] + (c[k].T @ coupled[k] @ c[k])[n:, :n]).ravel())
        return np.concatenate(out)

    amat = np.array([a_z(e) for e in np.eye(sum(dims))]).T
    zz = np.linalg.solve(amat, -np.concatenate([xv.ravel() for xv in xvo]))
    z = [
        zz[: dims[0]].reshape(nmo - nocc[0], nocc[0]),
        zz[dims[0] :].reshape(nmo - nocc[1], nocc[1]),
    ]
    z[1] = 1.0 * z[1]  # beta Z-vector
    d_hf, d_rel, w = [], [], np.zeros((NBF, NBF))
    for k in range(2):
        n = nocc[k]
        d = dm[k].copy()
        d[n:, :n] = z[k]
        d[:n, n:] = z[k].T
        dm[k] = d
        d_rel.append(c[k] @ d @ c[k].T)
        co = c[k][:, :n]
        d_hf.append(co @ co.T)
    vr = veff(d_rel)
    for k in range(2):
        n, e, ck = nocc[k], eps[k], c[k]
        zeta = 0.5 * (e[:, None] + e[None, :]) * dm[k]
        zeta[n:, :n] = e[None, :n] * dm[k][n:, :n]
        zeta[:n, n:] = e[:n, None] * dm[k][:n, n:]
        ip = imat[k].copy()
        ip[n:, :n] = imat[k][:n, n:].T
        co = ck[:, :n]
        p_occ = co @ co.T
        w += ck @ (ip - zeta) @ ck.T - co @ np.diag(e[:n]) @ co.T
        w -= p_occ @ vr[k] @ p_occ
    dtot = d_hf[0] + d_hf[1] + d_rel[0] + d_rel[1]
    g2 = np.sum(dtot * dh_probe)
    grad = np.sum(dtot * dh) + np.sum(w * ds)
    # d/dx U((D^a, D^b), (D^a + 2P^a, D^b + 2P^b)) through the ERIs.
    y = [d_hf[k] + 2 * d_rel[k] for k in range(2)]
    jy = jk(deri, y[0] + y[1])[0]
    grad += 0.5 * np.sum((d_hf[0] + d_hf[1]) * jy)
    grad -= 0.5 * sum(np.sum(d_hf[k] * jk(deri, y[k])[1]) for k in range(2))
    # RI response: one channel per spin.
    vinv = np.linalg.inv(v)
    for k in range(2):
        n = nocc[k]
        co, cv = c[k][:, fc:n], c[k][:, n:]
        u_ia = np.einsum("Qmn,mi,na->Qia", u, co, cv)
        du_ia = np.einsum("Qmn,mi,na->Qia", du, co, cv)
        yk = np.einsum("PQ,Qia->Pia", vm12, xs[k])
        m_u = np.einsum("PQ,Qia->Pia", vinv, u_ia)
        grad += RI_SPIN_WEIGHT * (
            2 * np.sum(yk * du_ia) - np.einsum("PQ,Pia,Qia->", dv, m_u, yk)
        )
    return grad, g2, z


# ================================================================ part G
def part_g():
    print("G: U-RI-MP2 Lagrangian gradient vs re-solved finite differences")
    x0, nocc = 0.1, (3, 2)
    dh_probe = RNG.standard_normal((NBF, NBF))
    dh_probe = 0.5 * (dh_probe + dh_probe.T)
    for fc in (0, 1):
        grad, g2, _ = u_gradient(x0, nocc, fc, dh_probe)
        fd1 = richardson(lambda xx: total_energy(xx, nocc, fc), x0, 1e-3)
        fd2 = richardson(
            lambda lam: total_energy(x0, nocc, fc, dh_probe, lam), 0.0, 1e-3
        )
        report(
            "G",
            f"G1 Lagrangian gradient, frozen core {fc}",
            abs(grad - fd1) / max(1, abs(fd1)),
            1e-8,
        )
        report(
            "G",
            f"G2 relaxed density, frozen core {fc}",
            abs(g2 - fd2) / max(1, abs(fd2)),
            1e-8,
        )


# ================================================================ part C
def closed_gradient(x, nocc, fc):
    """The closed-shell assembly of mp2_gradient_sympy.py part G (MP2)."""
    s, h, eri, u, v = inputs(x)
    ds, dh, deri, du, dv = d_inputs(x)
    _, c, eps = uhf(s, h, eri, nocc, nocc)
    c, eps = c[0], eps[0]
    nmo = c.shape[1]
    nv = nmo - nocc
    vm12 = inv_sqrt(v)
    bf = np.einsum("PQ,Qmn,mp,nq->Ppq", vm12, u, c, c)
    b = bf[:, fc:nocc, nocc:]
    t = np.einsum("Pia,Pjb->ijab", b, b) / denom(eps, eps, nocc, nocc, fc)
    tt = 2 * t - t.transpose(0, 1, 3, 2)
    p_oo = -np.einsum("ikab,jkab->ij", t, tt)
    p_vv = np.einsum("ijac,ijbc->ab", t, tt)
    xov = np.einsum("ijab,Pjb->Pia", tt, b)
    imat = np.zeros((nmo, nmo))
    imat[:, fc:nocc] = -2 * np.einsum("Pia,Pqa->qi", xov, bf[:, :, nocc:])
    imat[:, nocc:] = -2 * np.einsum("Pia,Pqi->qa", xov, bf[:, :, fc:nocc])
    dm1 = np.zeros((nmo, nmo))
    dm1[fc:nocc, fc:nocc] = p_oo + p_oo.T
    dm1[nocc:, nocc:] = p_vv + p_vv.T
    for k in range(fc):
        for i in range(fc, nocc):
            dm1[k, i] = dm1[i, k] = imat[k, i] / (eps[k] - eps[i])
    jc, kc = jk(eri, c @ dm1 @ c.T)
    vmo = c.T @ (2 * jc - kc) @ c
    xvo = vmo[nocc:, :nocc] + imat[:nocc, nocc:].T - imat[nocc:, :nocc]
    co, cv = c[:, :nocc], c[:, nocc:]

    def a_z(zv):
        zm = zv.reshape(nv, nocc)
        dz = cv @ zm @ co.T + co @ zm.T @ cv.T
        jz, kz = jk(eri, dz)
        return (
            (eps[nocc:, None] - eps[None, :nocc]) * zm
            + 0.5 * cv.T @ (4 * jz - kz - kz.T) @ co
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
    jr, kr = jk(eri, d_rel)
    p_occ = co @ co.T
    grad = np.sum((d_hf + d_rel) * dh)
    grad += np.sum((c @ ip @ c.T - zeta_ao - p_occ @ (2 * jr - kr) @ p_occ) * ds)
    jd, kd = jk(deri, d_hf + 2 * d_rel)
    grad += 0.5 * np.sum(d_hf * (jd - 0.5 * kd))
    u_ia = np.einsum("Qmn,mi,na->Qia", u, c[:, fc:nocc], cv)
    du_ia = np.einsum("Qmn,mi,na->Qia", du, c[:, fc:nocc], cv)
    y = np.einsum("PQ,Qia->Pia", vm12, xov)
    m_u = np.einsum("PQ,Qia->Pia", np.linalg.inv(v), u_ia)
    grad += 4 * np.sum(y * du_ia) - 2 * np.einsum("PQ,Pia,Qia->", dv, m_u, y)
    return grad, z


def part_c():
    print("C: closed-shell limit (n_alpha = n_beta)")
    x0, n = 0.1, 3
    probe = np.zeros((NBF, NBF))
    for fc in (0, 1):
        gu, _, zu = u_gradient(x0, (n, n), fc, probe)
        gc, zc = closed_gradient(x0, n, fc)
        report(
            "C",
            f"g_U = g_closed, frozen core {fc}",
            abs(gu - gc) / max(1, abs(gc)),
            1e-10,
        )
        report(
            "C",
            f"z^a = z^b = z/2, frozen core {fc}",
            max(np.max(np.abs(zu[0] - zc / 2)), np.max(np.abs(zu[1] - zc / 2))),
            1e-10,
        )


if __name__ == "__main__":
    part_s()
    part_g()
    part_c()
    if FAILED:
        print("FAILED:", *FAILED, sep="\n  ")
        sys.exit(1)
    print("all checks passed")
