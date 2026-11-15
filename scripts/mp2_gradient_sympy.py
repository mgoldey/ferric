#!/usr/bin/env python3
"""Proofs of the MP2-family gradient identities ferric-mp2 codes.

Theory: crates/ferric-mp2/theory.md sections 9, 9.2, 9.3 and 14. Each part is
checked on a tiny model against an independent exact route: a full re-solve of
HF (and of OO-MP2) at displaced parameters, differentiated by Richardson
central differences. The script runs every part and exits 1 if any failed.

The model. A "geometry" parameter x moves every input smoothly: the AO overlap
S(x), core Hamiltonian h(x), four-centre ERIs (mn|ls)(x) = sum_K L^K_mn L^K_ls
(the reference SCF's J/K) and an independent RI factorization for MP2,
3-centre u^P_mn(x) = (P|mn) and metric V(x), (ia|jb) = u_ia V^-1 u_jb. The
RHF is solved to 1e-14 at every x; nuclear repulsion is omitted.

  S  Symbolic (sympy, one (i, j) pair block, 2 occupied x 2 virtual): with
     E = sum K Ttilde, t = K/D, Ttilde = (c_OS + c_SS) t - c_SS t^swap
     (rimp2.rs SpinScaling direct/exchange): dE/dK = 2 Ttilde, and
     dE/de_k = (P + P^T)_kk, dE/de_c = (P + P^T)_cc with
     P_ij = -sum t_ik^ab Ttilde_jk^ab, P_ab = sum t_ij^ac Ttilde_ij^bc
     (build_mp2_density / Mp2Intermediates::apply_spin_scaling).
  G  RI-MP2 / SCS-MP2 relaxed gradient (zvector.rs solve_zvector,
     correlation_density_mo, frozen_core_block, build_imat_ri; gradient.rs
     mp2_relaxed_lagrangian_gradient and integral_response_gradient_3c2c) for
     (c_OS, c_SS) = (1, 1) and (6/5, 1/3), with and without one frozen core:
       G1 dE/dlam of h -> h + lam R equals Tr[(D_HF + D_rel) R]
          (the relaxed density is the energy derivative);
       G2 dE/dx equals the code's Lagrangian assembly
          Tr[D_tot h'] + sum (im1 - zeta - vhf_s1occ) S'
          + d/dx B(D_HF, D_HF + 2 D_rel) + the RI 3c/2c response.
  O  OO-RI-MP2 zero-response nuclear gradient (oo_rimp2_gradient.rs). OO-MP2 is
     converged at every x (semicanonical full-Fock functional, max|g| < 1e-11);
     dE/dx equals the z = 0 Lagrangian with zeta's occ-vir block (F W)_ia and
     the occupied energy-weighted term 2 C_o F_oo C_o^T.
  U  U-OO-RI-MP2 orbital gradient (u_oo_rimp2.rs u_gradient_solver_frame):
     at non-stationary, non-semicanonical orbitals of an open shell
     (n_alpha = 3, n_beta = 2),
       g^t_ck = 2 F^t_ck + [U-MP2 integral response]_ck
              + 2 [C^T (J[P^a + P^b] - K[P^t]) C]_ck + 2 (F^t W^t - W^t F^t)_ck,
     evaluated in each spin's semicanonical frame and carried back by
     U_v g U_o^T, equals dE/dkappa^t_ck of the spin-orbital energy
     E_UHF + E_aa + E_bb + E_ab (C^t -> C^t expm(kappa^t)).

Mutations (each turns the script red; the runner applies them one at a time
as exact substitutions of the lines tagged with these comments):
  * Z-vector coupling 1/2 of (4J - K - K^T) -> 1 breaks G ("FAIL G").
  * RHS term I_ia - I_ai with the sign of I_ai flipped breaks G ("FAIL G").
  * the SCS density weight (c_OS - c_SS) -> c_OS breaks G ("FAIL G").
  * zeta occ-vir weight eps_i -> eps_a breaks G ("FAIL G").
  * vhf_s1occ dropped breaks O ("FAIL O").
  * U-OO Fock-response density part J[P^a + P^b] -> J[P^a] breaks U ("FAIL U").
  * U-OO rotation part 2 (F W - W F) -> 2 (W F - F W) breaks U ("FAIL U").

Run: python3 scripts/mp2_gradient_sympy.py  (needs sympy, numpy, scipy)
"""

import itertools
import sys

import numpy as np
import sympy as sp
from scipy.linalg import eigh, expm

FAILED = []
RNG = np.random.default_rng(20261114)


def report(part, name, err, tol):
    ok = err <= tol
    print(f"  {'ok  ' if ok else 'FAIL ' + part} {name}: {err:.2e} (bar {tol:.0e})")
    if not ok:
        FAILED.append(f"{part}: {name}")


def sym(m):
    return 0.5 * (m + m.swapaxes(-1, -2))


def richardson(f, x0, h):
    """4-point central difference, truncation O(h^4)."""
    d1 = (f(x0 + h) - f(x0 - h)) / (2 * h)
    d2 = (f(x0 + 2 * h) - f(x0 - 2 * h)) / (4 * h)
    return (4 * d1 - d2) / 3


# ================================================================ part S
def part_s():
    print("S: SCS pair weights and densities (symbolic)")
    c_os, c_ss = sp.symbols("c_OS c_SS")
    no, nv = 2, 2
    eo = sp.symbols("e_o0:2")
    ev = sp.symbols("e_v0:2")
    k = {}
    for i, a, j, b in itertools.product(range(no), range(nv), range(no), range(nv)):
        key = min((i, a, j, b), (j, b, i, a))
        k[(i, j, a, b)] = sp.Symbol("K_%d%d%d%d" % key)

    def d(i, j, a, b):
        return eo[i] + eo[j] - ev[a] - ev[b]

    def t(i, j, a, b):
        return k[(i, j, a, b)] / d(i, j, a, b)

    def tt(i, j, a, b):
        return (c_os + c_ss) * t(i, j, a, b) - c_ss * t(i, j, b, a)

    idx = list(itertools.product(range(no), range(no), range(nv), range(nv)))
    e = sum(k[q] * tt(*q) for q in idx)
    # dE/dK for one representative symbol: K appears at (i,j,a,b) and (j,i,b,a).
    q = (0, 1, 0, 1)
    lhs = sp.diff(e, k[q])
    rhs = 2 * tt(*q) + 2 * tt(1, 0, 1, 0)
    err = sp.simplify(lhs - rhs)
    print(f"  {'ok  ' if err == 0 else 'FAIL S'} dE/dK_ijab == 2 Ttilde (both slots)")
    if err != 0:
        FAILED.append("S: dE/dK")
    for kk in range(no):
        p_kk = -sum(
            t(kk, j, a, b) * tt(kk, j, a, b)
            for j in range(no)
            for a in range(nv)
            for b in range(nv)
        )
        err = sp.simplify(sp.diff(e, eo[kk]) - 2 * p_kk)
        print(f"  {'ok  ' if err == 0 else 'FAIL S'} dE/de_{kk} == (P + P^T)_kk")
        if err != 0:
            FAILED.append("S: dE/de_occ")
    for cc in range(nv):
        p_cc = sum(
            t(i, j, cc, c) * tt(i, j, cc, c)
            for i in range(no)
            for j in range(no)
            for c in range(nv)
        )
        err = sp.simplify(sp.diff(e, ev[cc]) - 2 * p_cc)
        print(f"  {'ok  ' if err == 0 else 'FAIL S'} dE/de_v{cc} == (P + P^T)_cc")
        if err != 0:
            FAILED.append("S: dE/de_vir")


# ================================================================ model
NBF, NAUX, NL = 6, 9, 7


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
    u = m["u0"] + x * m["u1"]
    v = m["v0"] + x * m["v1"]
    return s, h, eri, u, v


def d_inputs(x):
    m = MODEL
    ds = m["s1"] + 2 * x * m["s2"]
    lf = m["l0"] + x * m["l1"]
    deri = np.einsum("Kmn,Kls->mnls", m["l1"], lf) + np.einsum(
        "Kmn,Kls->mnls", lf, m["l1"]
    )
    return ds, m["h1"], deri, m["u1"], m["v1"]


def jk(eri, d):
    return np.einsum("mnls,ls->mn", eri, d), np.einsum("mlns,ls->mn", eri, d)


def diis_scf(s, h, fock_of, dens_of, c0, tol=1e-13, max_iter=300):
    """DIIS SCF on the commutator F D S - S D F; returns converged (c, f)."""
    c = c0
    focks, errs = [], []
    for _ in range(max_iter):
        d = dens_of(c)
        f = fock_of(d)
        err = np.concatenate(
            [(fk @ dk @ s - s @ dk @ fk).ravel() for fk, dk in zip(f, d)]
        )
        if np.max(np.abs(err)) < tol:
            return c, f
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
        fx = [sum(cf * fk[s_] for cf, fk in zip(coef, focks)) for s_ in range(len(f))]
        c = [eigh(fs, s)[1] for fs in fx]
    raise AssertionError("SCF did not converge")


def rhf(s, h, eri, nocc):
    def dens_of(c):
        return [2 * c[0][:, :nocc] @ c[0][:, :nocc].T]

    def fock_of(d):
        jm, km = jk(eri, d[0])
        return [h + jm - 0.5 * km]

    c, f = diis_scf(s, h, fock_of, dens_of, [eigh(h, s)[1]])
    eps, c = eigh(f[0], s)
    d = 2 * c[:, :nocc] @ c[:, :nocc].T
    e = 0.5 * np.sum(d * (h + f[0]))
    return e, c, eps, f[0]


def inv_sqrt(v):
    w, q = np.linalg.eigh(v)
    return q @ np.diag(w**-0.5) @ q.T


def mp2_pieces(c, eps, u, v, nocc, fc, c_os, c_ss):
    """Amplitudes and SCS energy (i over active occupied)."""
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
    tt = (c_os + c_ss) * t - c_ss * t.transpose(0, 1, 3, 2)
    return np.sum(g * tt), t, tt, b_full, vm12


def total_energy(x, nocc, fc, c_os, c_ss, dh=None, lam=0.0):
    s, h, eri, u, v = inputs(x)
    if dh is not None:
        h = h + lam * dh
    e_hf, c, eps, _ = rhf(s, h, eri, nocc)
    e2 = mp2_pieces(c, eps, u, v, nocc, fc, c_os, c_ss)[0]
    return e_hf + e2


# ================================================================ part G
def zvector_gradient(x, nocc, fc, c_os, c_ss, dh_probe):
    s, h, eri, u, v = inputs(x)
    ds, dh, deri, du, dv = d_inputs(x)
    _, c, eps, _ = rhf(s, h, eri, nocc)
    nmo = c.shape[1]
    nv = nmo - nocc
    _, t, tt, b_full, vm12 = mp2_pieces(c, eps, u, v, nocc, fc, c_os, c_ss)
    # Unrelaxed densities with Ttilde in the second slot.
    p_mp2_oo = -np.einsum("ikab,jkab->ij", t, 2 * t - t.transpose(0, 1, 3, 2))
    p_mp2_vv = np.einsum("ijac,ijbc->ab", t, 2 * t - t.transpose(0, 1, 3, 2))
    p_os_oo = -np.einsum("ikab,jkab->ij", t, t)
    p_os_vv = np.einsum("ijac,ijbc->ab", t, t)
    p_oo = c_ss * p_mp2_oo + (c_os - c_ss) * p_os_oo  # SCS density weight
    p_vv = c_ss * p_mp2_vv + (c_os - c_ss) * p_os_vv
    # x^P_ia and the Lagrangian matrix I (active-occ and virtual columns).
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
    # Z-vector right-hand side and CPHF over every occupied orbital.
    jc, kc = jk(eri, c @ dm1 @ c.T)
    veff_mo = c.T @ (2 * jc - kc) @ c
    i_ia, i_ai = imat[:nocc, nocc:].T, imat[nocc:, :nocc]
    xvo = veff_mo[nocc:, :nocc] + i_ia - i_ai  # RHS
    co, cv = c[:, :nocc], c[:, nocc:]

    def a_z(zvec):
        zm = zvec.reshape(nv, nocc)
        dz = cv @ zm @ co.T + co @ zm.T @ cv.T
        jz, kz = jk(eri, dz)
        az = cv.T @ (4 * jz - kz - kz.T) @ co
        coupling = 0.5 * az  # Z-vector coupling
        return ((eps[nocc:, None] - eps[None, :nocc]) * zm + coupling).ravel()

    amat = np.array([a_z(e) for e in np.eye(nv * nocc)]).T
    z = np.linalg.solve(amat, -xvo.ravel()).reshape(nv, nocc)
    dm1[nocc:, :nocc] = z
    dm1[:nocc, nocc:] = z.T
    d_hf = 2 * co @ co.T
    d_rel = c @ dm1 @ c.T
    g1 = np.sum((d_hf + d_rel) * dh_probe)
    # Energy-weighted pieces.
    zeta = 0.5 * (eps[:, None] + eps[None, :]) * dm1
    zeta[nocc:, :nocc] = eps[None, :nocc] * dm1[nocc:, :nocc]  # zeta occ-vir weight
    zeta[:nocc, nocc:] = eps[:nocc, None] * dm1[:nocc, nocc:]
    zeta_ao = c @ zeta @ c.T + 2 * co @ np.diag(eps[:nocc]) @ co.T
    ip = imat.copy()
    ip[nocc:, :nocc] = imat[:nocc, nocc:].T
    im1 = c @ ip @ c.T
    jr, kr = jk(eri, d_rel)
    p_occ = co @ co.T
    vhf = p_occ @ (2 * jr - kr) @ p_occ
    grad = np.sum((d_hf + d_rel) * dh) + np.sum((im1 - zeta_ao - vhf) * ds)
    d2 = d_hf + 2 * d_rel
    jd, kd = jk(deri, d2)
    grad += 0.5 * np.sum(d_hf * (jd - 0.5 * kd))
    grad += ri_response(c, nocc, fc, tt, u, v, du, dv, vm12)
    return grad, g1


def ri_response(c, nocc, fc, tt, u, v, du, dv, vm12):
    """Explicit derivative of sum (ia|jb) Ttilde through u and V at fixed C."""
    u_ia = np.einsum("Qmn,mi,na->Qia", u, c[:, fc:nocc], c[:, nocc:])
    du_ia = np.einsum("Qmn,mi,na->Qia", du, c[:, fc:nocc], c[:, nocc:])
    b_ia = np.einsum("PQ,Qia->Pia", vm12, u_ia)
    xov = np.einsum("ijab,Pjb->Pia", tt, b_ia)
    y = np.einsum("PQ,Qia->Pia", vm12, xov)  # V^-1/2 x
    m_u = np.einsum("PQ,Qia->Pia", np.linalg.inv(v), u_ia)
    e3c = 4 * np.sum(y * du_ia)
    e2c = -2 * np.einsum("PQ,Pia,Qia->", dv, m_u, y)
    return e3c + e2c


def part_g():
    print("G: RI-MP2 / SCS-MP2 relaxed gradient vs re-solved finite differences")
    x0, nocc = 0.1, 3
    dh_probe = sym(RNG.standard_normal((NBF, NBF)))
    for (c_os, c_ss), fc in itertools.product(((1.0, 1.0), (1.2, 1.0 / 3.0)), (0, 1)):
        label = f"(c_OS, c_SS) = ({c_os:.3g}, {c_ss:.3g}), frozen core {fc}"
        grad, g1 = zvector_gradient(x0, nocc, fc, c_os, c_ss, dh_probe)
        fd1 = richardson(
            lambda lam: total_energy(x0, nocc, fc, c_os, c_ss, dh_probe, lam), 0.0, 1e-3
        )
        fd2 = richardson(lambda xx: total_energy(xx, nocc, fc, c_os, c_ss), x0, 1e-3)
        report(
            "G", f"G1 relaxed density, {label}", abs(g1 - fd1) / max(1, abs(fd1)), 1e-8
        )
        report(
            "G",
            f"G2 Lagrangian gradient, {label}",
            abs(grad - fd2) / max(1, abs(fd2)),
            1e-8,
        )


# ================================================================ U-MP2 machinery
def semicanonical(c, f, nocc):
    fm = c.T @ f @ c
    u = np.zeros_like(fm)
    wo, uo = np.linalg.eigh(fm[:nocc, :nocc])
    wv, uv = np.linalg.eigh(fm[nocc:, nocc:])
    u[:nocc, :nocc], u[nocc:, nocc:] = uo, uv
    return u, np.concatenate([wo, wv])


def uhf(h, eri, ca, cb, na, nb):
    da, db = ca[:, :na] @ ca[:, :na].T, cb[:, :nb] @ cb[:, :nb].T
    jt = jk(eri, da + db)[0]
    ka, kb = jk(eri, da)[1], jk(eri, db)[1]
    fa, fb = h + jt - ka, h + jt - kb
    e = 0.5 * (np.sum(da * (h + fa)) + np.sum(db * (h + fb)))
    return e, fa, fb


def mo_ints(bfac, c1, c2):
    b1 = np.einsum("Pmn,mp,nq->Ppq", bfac, c1, c1)
    b2 = np.einsum("Pmn,mp,nq->Ppq", bfac, c2, c2)
    return np.einsum("Ppq,Prs->pqrs", b1, b2)


def denom(e1o, e2o, e1v, e2v):
    return (
        e1o[:, None, None, None]
        + e2o[None, :, None, None]
        - e1v[None, None, :, None]
        - e2v[None, None, None, :]
    )


def u_mp2_energy_so(bfac, ca, cb, ea, eb, na, nb):
    """Brute-force spin-orbital MP2: 1/4 sum |<IJ||AB>|^2 / D."""
    n = ca.shape[1]
    cso = np.hstack([ca, cb])
    spin = np.array([0] * n + [1] * n)
    eso = np.concatenate([ea, eb])
    occ = list(range(na)) + [n + k for k in range(nb)]
    vir = list(range(na, n)) + [n + k for k in range(nb, n)]
    bso = np.einsum("Pmn,mp,nq->Ppq", bfac, cso, cso)
    chem = np.einsum("Ppq,Prs->pqrs", bso, bso)
    same = spin[:, None] == spin[None, :]
    chem = chem * same[:, :, None, None] * same[None, None, :, :]
    phys = chem.transpose(0, 2, 1, 3) - chem.transpose(0, 2, 3, 1)
    asym = phys[np.ix_(occ, occ, vir, vir)]
    dd = denom(eso[occ], eso[occ], eso[vir], eso[vir])
    return 0.25 * np.sum(asym**2 / dd)


def u_functional(h, eri, bfac, ca, cb, na, nb):
    e_hf, fa, fb = uhf(h, eri, ca, cb, na, nb)
    ua, ea = semicanonical(ca, fa, na)
    ub, eb = semicanonical(cb, fb, nb)
    return e_hf + u_mp2_energy_so(bfac, ca @ ua, cb @ ub, ea, eb, na, nb)


def u_gradient(h, eri, bfac, ca, cb, na, nb, closed=False):
    """u_gradient_solver_frame transcribed: returns (g_a, g_b) [vir, occ]."""
    _, fa, fb = uhf(h, eri, ca, cb, na, nb)
    ua, ea = semicanonical(ca, fa, na)
    ub, eb = semicanonical(cb, fb, nb)
    sa, sb = ca @ ua, cb @ ub
    n = sa.shape[1]
    gaa, gbb, gab = mo_ints(bfac, sa, sa), mo_ints(bfac, sb, sb), mo_ints(bfac, sa, sb)
    oa, va, ob, vb = slice(0, na), slice(na, n), slice(0, nb), slice(nb, n)
    # Amplitude-like weights dE/d(pq|rs) of each block.
    kaa = gaa[oa, va, oa, va].transpose(0, 2, 1, 3)
    kaa = kaa - kaa.transpose(0, 1, 3, 2)
    taa = kaa / denom(ea[oa], ea[oa], ea[va], ea[va])
    kbb = gbb[ob, vb, ob, vb].transpose(0, 2, 1, 3)
    kbb = kbb - kbb.transpose(0, 1, 3, 2)
    tbb = kbb / denom(eb[ob], eb[ob], eb[vb], eb[vb])
    kab = gab[oa, va, ob, vb].transpose(0, 2, 1, 3)
    tab = kab / denom(ea[oa], eb[ob], ea[va], eb[vb])

    def integral_response(g_ss, g_os, t_ss, t_os, no, eps, os_first):
        """dE/dkappa_ck at fixed denominators for a rotation of one spin."""
        nv = n - no
        o, v = slice(0, no), slice(no, n)
        gr = np.zeros((nv, no))
        for c, k in itertools.product(range(nv), range(no)):
            cc = no + c
            # same-spin: dE = sum_ijab t_ijab d(ia|jb)
            d_ss = np.zeros((no, no, nv, nv))
            d_ss[k] += g_ss[cc, v, o, v].transpose(1, 0, 2)
            d_ss[:, k] += g_ss[o, v, cc, v]
            d_ss[:, :, c, :] -= g_ss[o, k, o, v]
            d_ss[:, :, :, c] -= g_ss[o, v, o, k].transpose(0, 2, 1)
            val = np.sum(t_ss * d_ss)
            # opposite-spin: dE = 2 sum t_iJaB d(ia|JB), rotation on the first index pair
            gos = g_os
            if os_first:
                ob_ = slice(0, t_os.shape[1])
                vb_ = slice(t_os.shape[1], n)
                d_os = np.zeros(t_os.shape)
                d_os[k] += gos[cc, v, ob_, vb_].transpose(1, 0, 2)
                d_os[:, :, c, :] -= gos[o, k, ob_, vb_]
                val += 2 * np.sum(t_os * d_os)
            else:
                oa_ = slice(0, t_os.shape[0])
                va_ = slice(t_os.shape[0], n)
                d_os = np.zeros(t_os.shape)
                d_os[:, k] += gos[oa_, va_, cc, v]
                d_os[:, :, :, c] -= gos[oa_, va_, o, k].transpose(0, 2, 1)
                val += 2 * np.sum(t_os * d_os)
            gr[c, k] = val
        return gr

    g_int_a = integral_response(gaa, gab, taa, tab, na, ea, True)
    g_int_b = integral_response(gbb, gab, tbb, tab, nb, eb, False)

    # Unrelaxed U-MP2 densities W^s = dE/dF^s (occ-occ, vir-vir).
    def w_blocks(t_ss, t_os, first):
        if first:
            woo = -0.5 * np.einsum("ikab,jkab->ij", t_ss, t_ss) - np.einsum(
                "iKaB,jKaB->ij", t_os, t_os
            )
            wvv = 0.5 * np.einsum("ijac,ijbc->ab", t_ss, t_ss) + np.einsum(
                "iJaC,iJbC->ab", t_os, t_os
            )
        else:
            woo = -0.5 * np.einsum("ikab,jkab->ij", t_ss, t_ss) - np.einsum(
                "kiab,kjab->ij", t_os, t_os
            )
            wvv = 0.5 * np.einsum("ijac,ijbc->ab", t_ss, t_ss) + np.einsum(
                "ijac,ijad->cd", t_os, t_os
            )
        return woo, wvv

    def embed(woo, wvv, no):
        w = np.zeros((n, n))
        w[:no, :no] = sym(woo)
        w[no:, no:] = sym(wvv)
        return w

    wa = embed(*w_blocks(taa, tab, True), na)
    wb = embed(*w_blocks(tbb, tab, False), nb)
    pa, pb = sa @ wa @ sa.T, sb @ wb @ sb.T
    j_tot = jk(eri, pa + pb)[0]  # J[P^a + P^b]
    k_a, k_b = jk(eri, pa)[1], jk(eri, pb)[1]
    out = []
    for s_c, f, w, k_s, g_int, no, u in (
        (sa, fa, wa, k_a, g_int_a, na, ua),
        (sb, fb, wb, k_b, g_int_b, nb, ub),
    ):
        f_mo = s_c.T @ f @ s_c
        g_mo = s_c.T @ (j_tot - k_s) @ s_c
        rot = 2 * (f_mo @ w - w @ f_mo)  # U-OO rotation part
        g = 2 * f_mo[no:, :no] + g_int + 2 * g_mo[no:, :no] + rot[no:, :no]
        out.append(u[no:, no:] @ g @ u[:no, :no].T)
    return out


def part_u():
    print("U: U-OO-RI-MP2 orbital gradient vs FD of the spin-orbital energy")
    _, h, eri, u, v = inputs(0.0)
    bfac = np.einsum("PQ,Qmn->Pmn", inv_sqrt(v), u)
    na, nb = 3, 2
    # Near-UHF orbitals (orthonormal AO basis), then a random rotation that
    # mixes occ-occ, vir-vir AND occ-vir so the point is neither stationary
    # nor semicanonical.
    eye = np.eye(NBF)

    def dens_of(c):
        return [c[0][:, :na] @ c[0][:, :na].T, c[1][:, :nb] @ c[1][:, :nb].T]

    def fock_of(d):
        jt = jk(eri, d[0] + d[1])[0]
        return [h + jt - jk(eri, d[0])[1], h + jt - jk(eri, d[1])[1]]

    c0 = eigh(h, eye)[1]
    cu, fu = diis_scf(eye, h, fock_of, dens_of, [c0, c0])
    ca, cb = (eigh(fs, eye)[1] for fs in fu)
    kap = RNG.standard_normal((NBF, NBF)) * 0.05
    ca = ca @ expm(kap - kap.T)
    kap = RNG.standard_normal((NBF, NBF)) * 0.05
    cb = cb @ expm(kap - kap.T)
    ga, gb = u_gradient(h, eri, bfac, ca, cb, na, nb)

    def e_rot(spin, a, i, step):
        k = np.zeros((NBF, NBF))
        k[na + a if spin == 0 else nb + a, i] = step
        k = k - k.T
        if spin == 0:
            return u_functional(h, eri, bfac, ca @ expm(k), cb, na, nb)
        return u_functional(h, eri, bfac, ca, cb @ expm(k), na, nb)

    for spin, g, no in ((0, ga, na), (1, gb, nb)):
        fd = np.zeros_like(g)
        for a, i in itertools.product(range(NBF - no), range(no)):
            fd[a, i] = richardson(lambda s, a=a, i=i: e_rot(spin, a, i, s), 0.0, 1e-3)
        err = np.max(np.abs(g - fd)) / max(1.0, np.max(np.abs(fd)))
        report(
            "U",
            f"spin {'ab'[spin]}: g == dE/dkappa (max|g| {np.max(np.abs(fd)):.2e})",
            err,
            1e-8,
        )


# ================================================================ part O
def oo_converge(s, h, eri, bfac, nocc, c):
    """Closed-shell OO-MP2 = U-OO with C_a = C_b; g = g_a + g_b."""
    x = np.linalg.cholesky(s)
    xinv = np.linalg.inv(x)
    corth = x.T @ c  # orthonormal-basis coefficients
    corth = corth @ inv_sqrt(
        corth.T @ corth
    )  # a guess from another x is not S(x)-orthonormal
    h_o = xinv @ h @ xinv.T
    eri_o = np.einsum("mnls,am,bn,cl,ds->abcd", eri, xinv, xinv, xinv, xinv)
    bf_o = np.einsum("Pmn,am,bn->Pab", bfac, xinv, xinv)

    def energy(cc):
        return u_functional(h_o, eri_o, bf_o, cc, cc, nocc, nocc)

    e_old = energy(corth)
    for _ in range(40):
        ga, gb = u_gradient(h_o, eri_o, bf_o, corth, corth, nocc, nocc)
        g = ga + gb
        if np.max(np.abs(g)) < 1e-11:
            break
        # Newton step with a finite-difference Hessian of the analytic gradient
        # (only the converged point matters, not how it is reached).
        nv = NBF - nocc
        hess = np.zeros((nv * nocc, nv * nocc))
        for col in range(nv * nocc):
            kk = np.zeros((NBF, NBF))
            kk[nocc:, :nocc].flat[col] = 1e-5
            gp = sum(
                u_gradient(
                    h_o, eri_o, bf_o, *(2 * [corth @ expm(kk - kk.T)]), nocc, nocc
                )
            )
            gm = sum(
                u_gradient(
                    h_o, eri_o, bf_o, *(2 * [corth @ expm(kk.T - kk)]), nocc, nocc
                )
            )
            hess[:, col] = (gp - gm).ravel() / 2e-5
        k = np.zeros((NBF, NBF))
        k[nocc:, :nocc] = np.linalg.solve(sym(hess), -g.ravel()).reshape(nv, nocc)
        for _ in range(30):  # backtrack until the energy does not rise
            trial = corth @ expm(k - k.T)
            e_new = energy(trial)
            if e_new <= e_old + 1e-13:
                break
            k *= 0.5
        corth, e_old = trial, e_new
    return xinv.T @ corth, np.max(np.abs(g))


def oo_energy(x, nocc, c_start):
    s, h, eri, u, v = inputs(x)
    bfac = np.einsum("PQ,Qmn->Pmn", inv_sqrt(v), u)
    c, _ = oo_converge(s, h, eri, bfac, nocc, c_start)
    da = c[:, :nocc] @ c[:, :nocc].T
    jt, ka = jk(eri, 2 * da)[0], jk(eri, da)[1]
    fa = h + jt - ka
    ua, ea = semicanonical(c, fa, nocc)
    e_hf = np.sum(da * (h + fa))
    return e_hf + u_mp2_energy_so(bfac, c @ ua, c @ ua, ea, ea, nocc, nocc), c


def part_o():
    print("O: OO-RI-MP2 zero-response gradient vs re-converged finite differences")
    x0, nocc = 0.1, 3
    s, h, eri, u, v = inputs(x0)
    ds, dh, deri, du, dv = d_inputs(x0)
    _, c_hf, _, _ = rhf(s, h, eri, nocc)
    bfac = np.einsum("PQ,Qmn->Pmn", inv_sqrt(v), u)
    c, gmax = oo_converge(s, h, eri, bfac, nocc, c_hf)
    if not gmax < 1e-11:
        report("O", "OO-MP2 converged (max|g|)", gmax, 1e-11)
        return
    jt = jk(eri, 2 * c[:, :nocc] @ c[:, :nocc].T)
    f = h + jt[0] - 0.5 * jt[1]
    ua, eps = semicanonical(c, f, nocc)
    c = c @ ua  # semicanonical frame: amplitudes and densities live here
    _, t, tt, b_full, vm12 = mp2_pieces(c, eps, u, v, nocc, 0, 1.0, 1.0)
    p_oo = -np.einsum("ikab,jkab->ij", t, tt)
    p_vv = np.einsum("ijac,ijbc->ab", t, tt)
    nmo = NBF
    w = np.zeros((nmo, nmo))
    w[:nocc, :nocc] = p_oo + p_oo.T
    w[nocc:, nocc:] = p_vv + p_vv.T
    xov = np.einsum("ijab,Pjb->Pia", tt, b_full[:, :nocc, nocc:])
    imat = np.zeros((nmo, nmo))
    imat[:, :nocc] = -2 * np.einsum("Pia,Pqa->qi", xov, b_full[:, :, nocc:])
    imat[:, nocc:] = -2 * np.einsum("Pia,Pqi->qa", xov, b_full[:, :, :nocc])
    ip = imat.copy()
    ip[nocc:, :nocc] = imat[:nocc, nocc:].T
    im1 = c @ ip @ c.T
    f_mo = c.T @ f @ c
    fw = f_mo @ w
    zeta = 0.5 * (fw + fw.T)
    zeta[:nocc, nocc:] = fw[:nocc, nocc:]
    zeta[nocc:, :nocc] = fw[:nocc, nocc:].T
    co = c[:, :nocc]
    zeta_ao = c @ zeta @ c.T + 2 * co @ f_mo[:nocc, :nocc] @ co.T
    d_hf = 2 * co @ co.T
    d_c = c @ w @ c.T
    jr, kr = jk(eri, 2 * d_c)
    vhf = co @ co.T @ (jr - 0.5 * kr) @ co @ co.T
    w_overlap = im1 - zeta_ao - vhf  # oo_rimp2_gradient w_overlap
    grad = np.sum((d_hf + d_c) * dh) + np.sum(w_overlap * ds)
    jd, kd = jk(deri, d_hf + 2 * d_c)
    grad += 0.5 * np.sum(d_hf * (jd - 0.5 * kd))
    grad += ri_response(c, nocc, 0, tt, u, v, du, dv, vm12)
    fd = richardson(lambda xx: oo_energy(xx, nocc, c)[0], x0, 1e-3)
    print(f"  OO converged, max|g| {gmax:.1e}")
    report(
        "O",
        "z = 0 Lagrangian gradient == dE_OO/dx",
        abs(grad - fd) / max(1, abs(fd)),
        1e-8,
    )


def main():
    part_s()
    part_g()
    part_u()
    part_o()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): {FAILED}")
        sys.exit(1)
    print("all MP2 gradient identities hold")


if __name__ == "__main__":
    main()
