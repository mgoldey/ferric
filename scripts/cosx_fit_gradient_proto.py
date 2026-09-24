#!/usr/bin/env python3
"""Exact gradient of the OVERLAP-FITTED COSX energy (Z-vector), prototype.

Sibling of scripts/cosx_gradient_proto.py (plain COSX).  ferric's default COSX
exchange is overlap-fitted:

    K_f(D) = sym(Q Kt(D)),   Q = S S_num^{-1},  S_num = sum_g w_g phi_g phi_g^T,
    Kt(M)  = sum_g w_g phi_g phi_g^T M A^g    (M symmetric; ferric's builder
             with a general M computes sum_g P_g M^T A^g)

and the SCF uses F = h + J(D) - (c/2) K_f(D) (+ V_xc), E_x = -(c/4) tr[D K_f(D)].
K_f is NOT self-adjoint: tr[Y K_f(B)] = tr[B L(Y)] with the adjoint

    L(Y) = sym( sum_g P_g Q^T Y A^g ) = sym( Kt_builder(Y Q) ).

So dE/dD = F + Delta,  Delta = (c/4) (K_f(D) - L(D))  (RHF, total D), and the
SCF energy is not stationary in the orbitals: the gradient needs the orbital
response.

DERIVATION (RHF/RKS; Handy-Schaefer Lagrangian, all AO matrices symmetric)
------------------------------------------------------------------------
  Lag = E(D) + sum_ai z_ai (C^T F C)_ai - sum_pq w_pq (C^T S C - 1)_pq
  Stationarity in C gives
    Z-vector:  (e_a - e_i) z_ai + 4 [C^T R(Zs) C]_ai = -4 Delta_ai
               Zs = sym(C_v z C_o^T),  R(Y) = J(Y) - (c/2) L(Y) + fxc_D(Y)
    W_tot   =  2 C_o (e_o + Delta_oo + R(Zs)_oo) C_o^T
               + 1/2 [C_v (z e_o) C_o^T + h.c.]
  and dE/dx = dLag/dx at fixed C:
    g = V_nn^x + tr[(D + Zs) h^x] - tr[S^x W_tot]
        + (1/2) sum (mn|ls)^x D D + sum (mn|ls)^x Zs D
        - (c/4) dT(D, D) - (c/2) dT(Zs, D)
        + E_xc^x(D) + tr[Zs V_xc^x(D)]                 (KS only)
  where T(Y, B) = tr[Y Q Kt(B)] = sum_g w_g (B phi)^T A^g (Y Q phi) and dT is its
  EXPLICIT (fixed-matrix) derivative: AO values, 3c1e integrals, Becke
  weights AND dQ = dS S_num^-1 - Q dS_num S_num^-1.

UHF (per spin s, D_s = C_os C_os^T, E_x = -(c/2) sum_s tr[D_s K_f(D_s)]):
    Delta_s = (c/2)(K_f - L)(D_s)
    (e_a - e_i) z^s + 2 C_v^T [J(Zs_a + Zs_b) - c L(Zs_s)] C_o = -2 Delta_s,vo
    W_s = C_o (e_o + Delta_s,oo + R_s,oo) C_o^T + 1/2 [C_v (z^s e_o) C_o^T + h.c.]
    g = ... + tr[(D + Zs_a + Zs_b) h^x] + sum (..)^x (Zs_a+Zs_b) D
            - (c/2) sum_s dT(D_s, D_s) - c sum_s dT(Zs_s, D_s)

TRIVIAL LIMIT (exactness anchor): Q = I makes K_f = L, Delta = 0, z = 0, and the
whole fitted pipeline must reproduce the plain (fit-off) gradient.

PROTOTYPE-ONLY SHORTCUT, flagged: tr[Zs V_xc^x(D)] (the XC Fock matrix's
nuclear derivative at fixed D, grid response included) is evaluated here by a
4-point central difference in the GEOMETRY at FIXED AO matrices (no SCF).
ferric has no analytic version of that term (hessian.rs is a stub), so the Rust
port refuses fitted COSX for KS; the B3LYP rows below validate the Lagrangian,
not a shippable XC term.

RESULTS (2026-09-24, PySCF 2.13.1, (30,110) Becke grid, SCF conv 1e-12, FD h=1e-4)
  ANCHOR Q=I:   z == 0 exactly; |g_fit-pipeline - g_plain| 2.2e-16 / 1.7e-16 (STO-3G / 6-31G)
  case                  | with response | without response | ratio
  water RHF   STO-3G    |    1.71e-09   |     1.06e-06     |  623x
  water RHF   6-31G     |    1.86e-09   |     1.37e-06     |  735x
  water B3LYP STO-3G    |    1.68e-09   |     2.19e-07     |  131x   (XC Fock term by FD, see above)
  water B3LYP 6-31G     |    1.84e-09   |     2.54e-07     |  139x   (XC Fock term by FD, see above)
  HO2 UHF     STO-3G    |    4.75e-09   |     7.15e-06     | 1503x
  VERDICT: PASS (every residual at the FD floor; every no-response miss >= 100x it)

Run: OPENBLAS_NUM_THREADS=4 scripts/ferric-limited -- .venv/bin/python scripts/cosx_fit_gradient_proto.py
"""

import os
import sys

import numpy as np
from pyscf import dft, gto, scf
from pyscf.grad.rks import grids_response_cc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cosx_gradient_proto as P  # noqa: E402

GRID = (30, 110)
HO2 = [("O", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.33)), ("H", (0.05, 0.92, 1.6))]
BOHR = 0.52917721092


def sym(m):
    return 0.5 * (m + m.T)


class Ops:
    """Geometry-dependent COSX pieces for one molecule/grid, q_mode 'fit' or 'I'."""

    def __init__(self, mol, grids, q_mode):
        self.mol, self.grids, self.q_mode = mol, grids, q_mode
        phi = mol.eval_gto("GTOval", grids.coords)
        self.x = (phi * np.sqrt(np.abs(grids.weights))[:, None]).T
        self.a = mol.intor("int1e_grids", grids=grids.coords)
        self.s = mol.intor("int1e_ovlp")
        self.snum = self.x @ self.x.T
        self.sninv = np.linalg.inv(self.snum)
        self.q = self.s @ self.sninv if q_mode == "fit" else np.eye(mol.nao)

    def kt(self, m):
        """ferric's builder on a general M: sum_g P_g M^T A^g (unsymmetrized)."""
        g = np.einsum("gmn,ng->mg", self.a, m @ self.x)
        return self.x @ g.T

    def kf(self, d):
        return sym(self.q @ self.kt(d))

    def lop(self, y):
        return sym(self.kt(y @ self.q))


def texp(mol, y, b, grids, q_mode):
    """EXPLICIT nuclear derivative of T(Y,B) = tr[Y Q Kt(B)] at fixed Y, B."""
    natm = mol.natm
    aoslice = mol.aoslice_by_atom()
    out = np.zeros((natm, 3))
    ops = Ops(mol, grids, q_mode)
    q = ops.q
    yq = y @ q
    kt = np.zeros((mol.nao, mol.nao))
    per_point = []
    for home, (coords, w, w1) in enumerate(grids_response_cc(grids.pyscf_grids)):
        ao_d = mol.eval_gto("GTOval_sph_deriv1", coords)
        phi, dphi = ao_d[0], ao_d[1:4]
        a = mol.intor("int1e_grids", grids=coords)
        aip = mol.intor("int1e_grids_ip", grids=coords)
        f = phi @ b  # F = B phi
        h = phi @ yq.T  # H = Y Q phi
        ga = np.einsum("gmn,gn->gm", a, f)
        ha = np.einsum("gmn,gn->gm", a, h)
        e = np.einsum("gm,gm->g", f, ha)
        kt += (phi * w[:, None]).T @ ga
        dbas = np.zeros((len(w), natm, 3))
        v = ha @ b + ga @ yq  # d e / d phi (row form)
        for c in range(3):
            con = -dphi[c] * v
            iph = np.einsum("gmn,gn->gm", aip[c], h)
            ipf = np.einsum("gmn,gn->gm", aip[c], f)
            con = con - (f * iph + h * ipf)
            for at in range(natm):
                sl = slice(aoslice[at, 2], aoslice[at, 3])
                dbas[:, at, c] += con[:, sl].sum(axis=1)
        dtot = dbas.copy()
        dtot[:, home, :] -= dbas.sum(axis=1)
        out += np.einsum("g,gbx->bx", w, dtot)
        out += np.einsum("bxg,g->bx", w1, e)
        per_point.append((home, w, w1, phi, dphi))
    if q_mode == "fit":
        m = kt @ y
        p1 = ops.sninv @ m
        p2 = ops.sninv @ m @ ops.s @ ops.sninv
        ipov = mol.intor("int1e_ipovlp")
        for at in range(natm):
            sl = slice(aoslice[at, 2], aoslice[at, 3])
            for c in range(3):
                ds = np.zeros((mol.nao, mol.nao))
                ds[sl, :] -= ipov[c][sl, :]
                ds[:, sl] -= ipov[c][sl, :].T
                out[at, c] += np.sum(ds * p1.T)
        p2s = p2 + p2.T
        for home, w, w1, phi, dphi in per_point:
            s_g = np.einsum("gm,mn,gn->g", phi, p2, phi)
            out -= np.einsum("bxg,g->bx", w1, s_g)
            dbas = np.zeros((len(w), natm, 3))
            v = phi @ p2s.T
            for c in range(3):
                con = -dphi[c] * v
                for at in range(natm):
                    sl = slice(aoslice[at, 2], aoslice[at, 3])
                    dbas[:, at, c] += con[:, sl].sum(axis=1)
            dtot = dbas.copy()
            dtot[:, home, :] -= dbas.sum(axis=1)
            out -= np.einsum("g,gbx->bx", w, dtot)
    return out


# ------------------------------------------------------ PySCF-side pieces
def no_k_grad(mf):
    """PySCF gradient with its exchange derivative zeroed (nuc, hcore(D),
    -S W0, J(D), and for KS the XC term with grid response)."""
    gobj = mf.nuc_grad_method()
    if hasattr(gobj, "grid_response"):
        gobj.grid_response = True
    base = gobj.get_jk

    def get_jk_nok(mol_=None, dm_=None, hermi=0, omega=None):
        vj, vk = base(mol_, dm_, hermi, omega)
        return vj, np.zeros_like(vk)

    gobj.get_jk = get_jk_nok
    return gobj.kernel(), gobj


def hcore_term(gobj, mol, p):
    h1 = gobj.hcore_generator(mol)
    return np.array([np.einsum("xij,ij->x", h1(k), p) for k in range(mol.natm)])


def ovlp_term(gobj, mol, w):
    s1 = gobj.get_ovlp(mol)
    out = np.zeros((mol.natm, 3))
    for k, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        out[k] -= 2 * np.einsum("xij,ij->x", s1[:, p0:p1], w[p0:p1])
    return out


def j_term(gobj, mol, p):
    """d/dx of 1/2 sum (mn|ls) P P  (quadratic in P)."""
    vj = gobj.get_j(mol, p)
    out = np.zeros((mol.natm, 3))
    for k, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        out[k] += 2 * np.einsum("xij,ij->x", vj[:, p0:p1], p[p0:p1])
    return out


def jmat(mol, p):
    return scf.hf.get_jk(mol, p, with_k=False)[0]


def xc_fock_deriv_fd(mf, zs, d, h=1e-3):
    """PROTOTYPE-ONLY: tr[Zs V_xc^x(D)] by a 4-point geometry stencil at
    fixed AO matrices (grid rebuilt at each displaced geometry)."""
    mol = mf.mol
    ni = mf._numint

    def f(m):
        g = dft.gen_grid.Grids(m)
        g.atom_grid = mf.grids.atom_grid
        g.prune = None
        g.build()
        vmat = ni.nr_rks(m, g, mf.xc, d)[2]
        return np.sum(zs * vmat)

    out = np.zeros((mol.natm, 3))
    xyz = mol.atom_coords()  # Bohr
    for a in range(mol.natm):
        for c in range(3):
            vals = []
            for s in (1, -1, 2, -2):
                x = xyz.copy()
                x[a, c] += s * h
                vals.append(f(mol.set_geom_(x, unit="Bohr", inplace=False)))
            out[a, c] = (8 * (vals[0] - vals[1]) - (vals[2] - vals[3])) / (12 * h)
    return out


# ------------------------------------------------------------ RHF / RKS
def rhf_fit_grad(mf, grids, cx, q_mode="fit", response=True):
    mol = mf.mol
    ops = Ops(mol, grids, q_mode)
    c, eps, occ = mf.mo_coeff, mf.mo_energy, mf.mo_occ
    co, cv = c[:, occ > 0], c[:, occ == 0]
    eo, ev = eps[occ > 0], eps[occ == 0]
    no, nv = co.shape[1], cv.shape[1]
    d = mf.make_rdm1()
    is_ks = hasattr(mf, "xc")

    def rop(y):
        r = jmat(mol, y) - 0.5 * cx * ops.lop(y)
        if is_ks:
            r = r + mf._numint.nr_rks_fxc(mol, mf.grids, mf.xc, d, y, 0, 1)
        return r

    delta = 0.25 * cx * (ops.kf(d) - ops.lop(d))
    rhs = -4 * cv.T @ delta @ co
    if response:
        amat = np.zeros((nv * no, nv * no))
        for k in range(nv * no):
            z = np.zeros(nv * no)
            z[k] = 1.0
            z = z.reshape(nv, no)
            zs = sym(cv @ z @ co.T)
            amat[:, k] = (
                (ev[:, None] - eo[None, :]) * z + 4 * cv.T @ rop(zs) @ co
            ).ravel()
        z = np.linalg.solve(amat, rhs.ravel()).reshape(nv, no)
    else:
        z = np.zeros((nv, no))
    zs = sym(cv @ z @ co.T)
    g0, gobj = no_k_grad(mf)
    w0 = 2 * co @ np.diag(eo) @ co.T
    g = g0 - 0.25 * cx * texp(mol, d, d, grids, q_mode)
    if not response:
        return g, 0.0
    rz = rop(zs) if np.any(z) else np.zeros_like(d)
    wt = 2 * co @ (np.diag(eo) + co.T @ delta @ co + co.T @ rz @ co) @ co.T
    wt += sym(cv @ (z * eo[None, :]) @ co.T)  # 1/2 (M + M^T) = sym
    g = g + hcore_term(gobj, mol, zs) + ovlp_term(gobj, mol, wt - w0)
    g = g + j_term(gobj, mol, d + zs) - j_term(gobj, mol, zs) - j_term(gobj, mol, d)
    g = g - 0.5 * cx * texp(mol, zs, d, grids, q_mode)
    if is_ks and np.any(z):
        g = g + xc_fock_deriv_fd(mf, zs, d)
    return g, np.abs(z).max()


def rhf_case(basis, xc):
    mol = P.make_mol(P.WATER, basis)
    mf, grids, cx = P.run_scf(mol, GRID, True, xc)
    g, zmax = rhf_fit_grad(mf, grids, cx)
    g_nr, _ = rhf_fit_grad(mf, grids, cx, response=False)
    fd = P.fd_grad(P.WATER, basis, GRID, True, xc)
    e1, e0 = np.abs(g - fd).max(), np.abs(g_nr - fd).max()
    print(
        f"{basis:7s} {xc or 'HF':6s} fit | max|z| {zmax:.2e} | with response {e1:.2e} | "
        f"without response {e0:.2e} | ratio {e0 / e1:.0f}x",
        flush=True,
    )
    return e1, e0


def anchor_case(basis):
    """Q = I: Delta = 0 => z = 0, and the fitted pipeline = the plain gradient."""
    mol = P.make_mol(P.WATER, basis)
    mf, grids, cx = P.run_scf(
        mol, GRID, False, None
    )  # plain SCF (K_f = K_plain at Q=I)
    g_i, zmax = rhf_fit_grad(mf, grids, cx, q_mode="I")
    g_plain, _ = P.analytic_grad(mf, grids, cx, False)
    diff = np.abs(g_i - g_plain).max()
    print(
        f"ANCHOR Q=I {basis}: max|z| = {zmax:.1e}, max|g_fitpipeline - g_plain| = {diff:.2e}",
        flush=True,
    )
    return zmax, diff


# ------------------------------------------------------------------ UHF
def run_uhf(mol, grid, conv=1e-12):
    grids = P.make_grids(mol, grid)
    s_ao = mol.intor("int1e_ovlp")
    mf = scf.UHF(mol)

    def get_jk(mol_, dm, hermi=1, with_j=True, with_k=True, omega=None):
        dm = np.asarray(dm)
        vj = scf.hf.get_jk(mol_, dm, with_k=False)[0]
        vk = np.array(
            [
                P.cosx_k(mol_, dd, grids, True, s_ao)
                for dd in dm.reshape(-1, *dm.shape[-2:])
            ]
        )
        return vj, vk.reshape(dm.shape)

    mf.get_jk = get_jk
    mf.conv_tol = conv
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 300
    mf.kernel()
    assert mf.converged
    return mf, grids


def uhf_fit_grad(mf, grids, response=True):
    mol = mf.mol
    ops = Ops(mol, grids, "fit")
    cx = 1.0
    dm = mf.make_rdm1()
    dtot = dm[0] + dm[1]
    blocks = []
    for s in range(2):
        c, eps, occ = mf.mo_coeff[s], mf.mo_energy[s], mf.mo_occ[s]
        blocks.append((c[:, occ > 0], c[:, occ == 0], eps[occ > 0], eps[occ == 0]))
    deltas = [0.5 * cx * (ops.kf(dm[s]) - ops.lop(dm[s])) for s in range(2)]
    sizes = [b[1].shape[1] * b[0].shape[1] for b in blocks]

    def unpack(v):
        out, o = [], 0
        for s in range(2):
            co, cv = blocks[s][0], blocks[s][1]
            z = v[o : o + sizes[s]].reshape(cv.shape[1], co.shape[1])
            out.append(z)
            o += sizes[s]
        return out

    def zsyms(zz):
        return [sym(blocks[s][1] @ zz[s] @ blocks[s][0].T) for s in range(2)]

    def rops(zs):
        jt = jmat(mol, zs[0] + zs[1])
        return [jt - cx * ops.lop(zs[s]) for s in range(2)]

    rhs = np.concatenate(
        [(-2 * blocks[s][1].T @ deltas[s] @ blocks[s][0]).ravel() for s in range(2)]
    )
    n = rhs.size
    if response:
        amat = np.zeros((n, n))
        for k in range(n):
            v = np.zeros(n)
            v[k] = 1.0
            zz = unpack(v)
            rr = rops(zsyms(zz))
            cols = []
            for s in range(2):
                co, cv, eo, ev = blocks[s]
                cols.append(
                    (
                        (ev[:, None] - eo[None, :]) * zz[s] + 2 * cv.T @ rr[s] @ co
                    ).ravel()
                )
            amat[:, k] = np.concatenate(cols)
        zz = unpack(np.linalg.solve(amat, rhs))
    else:
        zz = unpack(np.zeros(n))
    zs = zsyms(zz)
    g0, gobj = no_k_grad(mf)
    g = g0.copy()
    for s in range(2):
        g -= 0.5 * cx * texp(mol, dm[s], dm[s], grids, "fit")
    if not response:
        return g
    rr = rops(zs)
    w0 = sum(blocks[s][0] @ np.diag(blocks[s][2]) @ blocks[s][0].T for s in range(2))
    wt = np.zeros_like(dtot)
    for s in range(2):
        co, cv, eo, _ = blocks[s]
        wt += co @ (np.diag(eo) + co.T @ deltas[s] @ co + co.T @ rr[s] @ co) @ co.T
        wt += sym(cv @ (zz[s] * eo[None, :]) @ co.T)
    zt = zs[0] + zs[1]
    g += hcore_term(gobj, mol, zt) + ovlp_term(gobj, mol, wt - w0)
    g += j_term(gobj, mol, dtot + zt) - j_term(gobj, mol, zt) - j_term(gobj, mol, dtot)
    for s in range(2):
        g -= cx * texp(mol, zs[s], dm[s], grids, "fit")
    return g


def uhf_case():
    mol = gto.M(atom=HO2, basis="sto-3g", unit="Angstrom", spin=1, verbose=0)
    mf, grids = run_uhf(mol, GRID)
    g = uhf_fit_grad(mf, grids)
    g_nr = uhf_fit_grad(mf, grids, response=False)
    h = 1e-4
    fd = np.zeros((mol.natm, 3))
    for a in range(mol.natm):
        for c in range(3):
            e = []
            for s in (1, -1):
                at = [(sy, list(xyz)) for sy, xyz in HO2]
                at[a][1][c] += s * h * BOHR
                m = gto.M(atom=at, basis="sto-3g", unit="Angstrom", spin=1, verbose=0)
                e.append(run_uhf(m, GRID)[0].e_tot)
            fd[a, c] = (e[0] - e[1]) / (2 * h)
    e1, e0 = np.abs(g - fd).max(), np.abs(g_nr - fd).max()
    print(
        f"HO2 UHF sto-3g fit | with response {e1:.2e} | without response {e0:.2e} | "
        f"ratio {e0 / e1:.0f}x",
        flush=True,
    )
    return e1, e0


def main():
    print(f"=== fitted COSX gradient, grid {GRID}, FD h=1e-4 Bohr, SCF conv 1e-12 ===")
    rows = [anchor_case("sto-3g"), anchor_case("6-31g")]
    res = [
        rhf_case("sto-3g", None),
        rhf_case("6-31g", None),
        rhf_case("sto-3g", "b3lyp"),
        rhf_case("6-31g", "b3lyp"),
        uhf_case(),
    ]
    ok = all(e1 < 1e-7 and e0 > 10 * e1 for e1, e0 in res) and all(
        z == 0.0 and d < 1e-10 for z, d in rows
    )
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
