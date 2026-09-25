"""Iteration 16: analytic nuclear gradients (forces) for the Gamma-point periodic RHF of pbc_gamma.py.

Energy (pbc_gamma conventions, G=0 dropped everywhere, optional Madelung shift v_M):
    E = sum D h + sum Gam_mnls I_mnls - v_M/4 tr(DSDS) + E_nn,   Gam = 1/2 D_mn D_ls - 1/4 D_ml D_ns
    h = T + V_SR + V_LR + c0 Ztot S,   I = I_SR + I_LR - c0 S(x)S,   c0 = pi/(w^2 Omega) (0 for pure AFT)
Moving atom A moves its nucleus AND every basis function centred on it, in EVERY lattice image.

dE/dR_A =  sum D dT/dR_A                                   (lattice-summed <grad m|n_L>)
         + sum D dV_SR/dR_A   basis centres + nucleus       (3c erfc, nucleus = -(bra+ket) by translation invariance)
         + sum D dV_LR/dR_A   basis: d/dA of the pair FT;  nucleus: -iG Z_A e^{-iG.R_A} in the structure factor
         + sum Gam dI_SR/dR_A (4 x the bra-slot derivative, the Gamma I is 8-fold symmetric)
         + sum Gam dI_LR/dR_A = (4/Omega) Re sum_G v(G) sum_{m in A,n} conj(Q_mn(G)) Z_mn(G)
         + sum M dS/dR_A,     M = -W + c0 (Ztot - N) D + (c0/2 - v_M/2) D S D,   W = 1/2 D F D
         + dE_nn/dR_A         (Ewald: erfc real space + reciprocal; self and background terms are R-independent)
Q_mn(G) = d/dA_m P_mn(G) (bra centre only); the ket derivative is Q_nm because P_mn = P_nm at Gamma, and
Q_mn + Q_nm = -iG P_mn (moving both centres = translating the pair) is checked as a unit identity.

PySCF molecular intor stands in for libint2 (int1e_ipovlp/ipkin, int3c2e_ip1, int2e_ip1 under erfc).
_MUTANT (module global) switches on deliberate defects for the mutation tests.
"""

from __future__ import annotations

import numpy as np
from pyscf import gto
from scipy.special import erfc

from pbc_gamma import Cell, _lattice_points, build_integrals, cart_comps, hermite_E, pair_ft, rhf, shell_table

_MUTANT = None  # None | 'vne_no_basis' | 'w_sign' | 'no_ewald_lr' | 'no_madelung_s'


def ao_atom(mol):
    out = np.zeros(mol.nao, int)
    for A, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        out[p0:p1] = A
    return out


# ------------------------------------------------------------ pair FT and its bra-centre derivative
def pair_ft_deriv(cell, G, thresh=1e-14):
    """(P, Q): P[m,n,g] exactly as pbc_gamma.pair_ft (raw Cartesian), Q[x,m,n,g] = d P / d A_m,x
    (bra centre only, all lattice images of m move together; here m is the home-cell function).
    d/dA_x of x_A^i e^{-a x_A^2} = 2a x_A^{i+1} e - i x_A^{i-1} e : raise/lower the bra Hermite table."""
    mol = cell.mol
    sh = shell_table(mol)
    nao, ng = mol.nao, len(G)
    G2 = np.einsum("gi,gi->g", G, G)
    P = np.zeros((nao, nao, ng), complex)
    Q = np.zeros((3, nao, nao, ng), complex)
    amin = min(s["exps"].min() for s in sh)
    rpair = np.sqrt(2 * np.log(1 / thresh) / amin) + 2.0  # same image set as pair_ft
    Ls = cell.translations(rpair)
    lmax = max(s["l"] for s in sh)
    powg = [np.stack([(-1j * G[:, d]) ** t for t in range(2 * lmax + 4)]) for d in range(3)]
    for sa in sh:
        ca, la = cart_comps(sa["l"]), sa["l"]
        for sb in sh:
            cb, lb = cart_comps(sb["l"]), sb["l"]
            bP = np.zeros((len(ca), len(cb), ng), complex)
            bQ = np.zeros((3, len(ca), len(cb), ng), complex)
            for L in Ls:
                B = sb["A"] + L
                AB = sa["A"] - B
                for a, cA in zip(sa["exps"], sa["coef"]):
                    for b, cB in zip(sb["exps"], sb["coef"]):
                        p = a + b
                        if abs(cA * cB * np.exp(-a * b / p * AB @ AB)) < thresh:
                            continue
                        Pc = (a * sa["A"] + b * B) / p
                        gm = np.nonzero(G2 < 4 * p * (np.log(1 / thresh) + 10))[0]
                        if len(gm) == 0:
                            continue
                        common = cA * cB * (np.pi / p) ** 1.5 * np.exp(-G2[gm] / (4 * p) - 1j * (G[gm] @ Pc))
                        F, Fd = [], []
                        for d in range(3):
                            E = hermite_E(la + 1, lb, a, b, AB[d])  # bra raised by one
                            f = np.einsum("ijt,tg->ijg", E, powg[d][: E.shape[2], gm])
                            fd = 2 * a * f[1:]  # (la+1, lb+1): 2a F[i+1]
                            fd[1:] -= np.arange(1, la + 1)[:, None, None] * f[: la]  # - i F[i-1]
                            F.append(f[: la + 1])
                            Fd.append(fd)
                        for u, (ax, ay, az) in enumerate(ca):
                            for v, (bx, by, bz) in enumerate(cb):
                                fx, fy, fz = F[0][ax, bx], F[1][ay, by], F[2][az, bz]
                                bP[u, v, gm] += common * fx * fy * fz
                                bQ[0, u, v, gm] += common * Fd[0][ax, bx] * fy * fz
                                bQ[1, u, v, gm] += common * fx * Fd[1][ay, by] * fz
                                bQ[2, u, v, gm] += common * fx * fy * Fd[2][az, bz]
            P[sa["off"] : sa["off"] + len(ca), sb["off"] : sb["off"] + len(cb)] = bP
            Q[:, sa["off"] : sa["off"] + len(ca), sb["off"] : sb["off"] + len(cb)] = bQ
    return P, Q


# ----------------------------------------------------------------------------------- Ewald E_nn
def ewald_grad(cell, w=1.0, rcut=None, gcut=None):
    """dE_nn/dR_i (natm, 3) for the Ewald energy of pbc_gamma._ewald (same cutoffs)."""
    Z, R = cell.Z, cell.R
    rcut = rcut or 7.0 / w
    gcut = gcut or 2 * w * np.sqrt(np.log(1e16))
    g = np.zeros_like(R)
    pad = np.linalg.norm(cell.a.sum(0)) + np.ptp(R, axis=0).max() if len(R) > 1 else 0.0
    for L in _lattice_points(cell.a, rcut + 2 + pad):
        dv = R[:, None, :] - R[None, :, :] - L
        d = np.linalg.norm(dv, axis=2)
        m = d > 1e-12
        ds = np.where(m, d, 1.0)
        fp = -(erfc(w * ds) / ds**2 + 2 * w / np.sqrt(np.pi) * np.exp(-(w * ds) ** 2) / ds)
        g += np.einsum("ij,ijx->ix", np.where(m, Z[:, None] * Z[None, :] * fp / ds, 0.0), dv)
    if _MUTANT != "no_ewald_lr":
        G = cell.gvectors(gcut)
        G = G[np.einsum("gi,gi->g", G, G) > 1e-12]
        G2 = np.einsum("gi,gi->g", G, G)
        eg = np.exp(-1j * G @ R.T)  # (g, atom)
        SG = eg @ Z
        k = 2 * np.pi / cell.vol * np.exp(-G2 / (4 * w * w)) / G2
        # d|S|^2/dR_i = 2 Re[conj(S) (-iG) Z_i e^{-iG.R_i}]
        g += 2 * np.einsum("g,gi,gx->ix", k, np.conj(SG)[:, None] * (-1j) * eg * Z, G).real
    return g


# ---------------------------------------------------------------------------- SR (erfc) pieces
def _latsum_ip(cell, intor, rcut_1e):
    """X[x,m,n] = sum_L <grad m_0 | O | n_L> (electron-coordinate gradient on the bra)."""
    mol = cell.mol
    nao, nb0 = mol.nao, mol.nbas
    L1 = cell.translations(rcut_1e)
    sm = cell.supermol(L1)
    return sm.intor(intor, comp=3, shls_slice=(0, nb0, 0, sm.nbas)).reshape(3, nao, len(L1), nao).sum(2)


def sr_vne_grad(cell, D, w, rcut_1e):
    """gb[A, C, x]: basis-centre part of d(sum D V_SR)/dR_A coming from nucleus C (all images).
    basis gradient = gb.sum(1); nucleus gradient = -gb.sum(0) (translation invariance per 3c integral)."""
    mol = cell.mol
    nao, nb0, natm = mol.nao, mol.nbas, mol.natm
    L1 = cell.translations(rcut_1e)
    sm = cell.supermol(L1)
    fm = gto.fakemol_for_charges(sm.atom_coords())
    fm.cart = True
    mm = sm + fm
    mm.cart = True
    Zs = np.tile(cell.Z, len(L1))
    catm = np.tile(np.arange(natm), len(L1))
    aoat = ao_atom(mol)
    gb = np.zeros((natm, natm, 3))
    with mm.with_range_coulomb(-w):
        for k in range(len(L1)):
            y = mm.intor("int3c2e_ip1_cart", comp=3, shls_slice=(0, nb0, k * nb0, (k + 1) * nb0, sm.nbas, mm.nbas))
            # dV_mn/dA_m = Z_c (grad m n | c);  x2 for the ket slot (V^C symmetric per nucleus)
            t = 2 * np.einsum("xmnc,mn->mcx", y.reshape(3, nao, nao, -1), D) * Zs[None, :, None]
            np.add.at(gb, (aoat[:, None], catm[None, :]), t)
    return gb


def sr_eri_grad(cell, Gam, w, rcut_2e, rcut_bra):
    """g[A,x] = sum Gam dI_SR/dR_A = 4 sum_{m in A} Gam_mnls * (-(grad m n|l s)) over the same images as sr_terms."""
    mol = cell.mol
    nao, nb0 = mol.nao, mol.nbas
    L2 = cell.translations(rcut_2e)
    nbra = int(np.sum(np.linalg.norm(L2, axis=1) <= rcut_bra + 1e-9))
    sm2 = cell.supermol(L2)
    n2 = len(L2)
    Y = np.zeros((3,) + (nao,) * 4)
    with sm2.with_range_coulomb(-w):
        for k in range(n2):
            blk = sm2.intor(
                "int2e_ip1_cart", comp=3,
                shls_slice=(0, nb0, 0, nbra * nb0, k * nb0, (k + 1) * nb0, 0, sm2.nbas),
            ).reshape(3, nao, nbra, nao, nao, n2, nao)
            Y += blk.sum(axis=(2, 5))
    per_m = -4 * np.einsum("xmnls,mnls->mx", Y, Gam)
    return _fold(per_m, ao_atom(mol), mol.natm)


def _fold(per_ao, aoat, natm):
    g = np.zeros((natm, 3))
    np.add.at(g, aoat, per_ao)
    return g


# --------------------------------------------------------------------------------- driver
def gamma_rhf_grad(cell, nelec, w=None, exxdiv=None, gcut=None, conv=1e-12, rcut_1e=22.0, rcut_2e=None,
                   rcut_bra=None, ints=None):
    """Converge the pbc_gamma Gamma RHF and return dict(e, grad (natm,3), parts{...}, comm)."""
    if ints is None:
        ints = build_integrals(cell, w, rcut_1e=rcut_1e, rcut_2e=rcut_2e, rcut_bra=rcut_bra, verbose=False,
                               exxdiv=exxdiv, gcut=gcut)
    S, h, I, enn, P, G, vM = (ints[k] for k in ("S", "h", "I", "enn", "P", "G", "madelung"))
    e, eps, it, C = rhf(S, h, I, enn, nelec, conv=conv, kshift=vM, return_mo=True, maxiter=200)
    mol = cell.mol
    natm, nao = mol.natm, mol.nao
    nocc = nelec // 2
    Co = C[:, :nocc]
    D = 2 * Co @ Co.T
    J = np.einsum("mnls,ls->mn", I, D)
    K = np.einsum("mlsn,ls->mn", I, D) + vM * S @ D @ S
    F = h + J - 0.5 * K
    comm = abs(F @ D @ S - S @ D @ F).max()
    W = 0.5 * D @ F @ D
    aoat = ao_atom(mol)
    parts = {}

    # overlap-coupled terms (Pulay/W, G=0 bookkeeping, Madelung S D S)
    c0 = 0.0 if w is None else np.pi / (w * w * cell.vol)
    N = np.sum(D * S)
    M = (W if _MUTANT == "w_sign" else -W) + c0 * (cell.Z.sum() - N) * D + 0.5 * c0 * D @ S @ D
    if _MUTANT != "no_madelung_s":
        M = M - 0.5 * vM * D @ S @ D
    XS = _latsum_ip(cell, "int1e_ipovlp_cart", rcut_1e)
    XT = _latsum_ip(cell, "int1e_ipkin_cart", rcut_1e)
    # d/dA_m <m|n> = -<grad m|n>; x2 for the ket slot (symmetric M, D)
    parts["S"] = _fold(-2 * np.einsum("xmn,mn->mx", XS, M), aoat, natm)
    parts["T"] = _fold(-2 * np.einsum("xmn,mn->mx", XT, D), aoat, natm)

    # long range (reciprocal) V_ne and ERI: need Q = d P / dA on the same (calibrated) G set
    Praw, Q = pair_ft_deriv(cell, G)
    P0 = pair_ft(cell, np.zeros((1, 3)))[..., 0].real
    nrm = np.sqrt(np.diag(S) / np.diag(P0))
    nn = nrm[:, None] * nrm[None, :]
    Q = Q * nn[None, :, :, None]
    Pc = Praw * nn[:, :, None]
    p_err = abs(Pc - P).max()  # same P as the energy used (sanity)
    G2 = np.einsum("gi,gi->g", G, G)
    v = 4 * np.pi / G2 * (np.exp(-G2 / (4 * w * w)) if w is not None else 1.0)
    SG_at = np.exp(-1j * G @ cell.R.T) * cell.Z  # (g, atom)
    SG = SG_at.sum(1)
    rho = np.einsum("mn,mng->g", D, P)
    # basis: rho'_A = 2 sum_{m in A,n} D_mn Q_mn ;  E = -(1/Omega) sum v Re[conj(rho) SG]
    rq = _fold_g(2 * np.einsum("mn,xmng->mxg", D, Q), aoat, natm)  # (A, x, g)
    g_vlr_basis = -(np.einsum("g,axg->ax", v, np.conj(rq) * SG).real) / cell.vol
    g_vlr_nuc = -(np.einsum("g,ga,gx->ax", v * np.conj(rho), SG_at * (-1j), G).real) / cell.vol
    if _MUTANT == "vne_no_basis":
        g_vlr_basis = 0 * g_vlr_basis
    parts["Vlr_basis"], parts["Vlr_nuc"] = g_vlr_basis, g_vlr_nuc
    # 2e LR: Z_mn(G) = 1/2 D_mn rho - 1/4 (D P D)_mn
    Zt = 0.5 * D[:, :, None] * rho[None, None, :] - 0.25 * np.einsum("ml,lsg,sn->mng", D, P, D)
    per_m = 4 / cell.vol * np.einsum("g,xmng,mng->mx", v, np.conj(Q), Zt).real
    parts["Ilr"] = _fold(per_m, aoat, natm)

    if w is not None:
        gb = sr_vne_grad(cell, D, w, rcut_1e)
        parts["Vsr_basis"] = 0 * gb.sum(1) if _MUTANT == "vne_no_basis" else gb.sum(1)
        parts["Vsr_nuc"] = -gb.sum(0)
        Gam = 0.5 * np.einsum("mn,ls->mnls", D, D) - 0.25 * np.einsum("ml,ns->mnls", D, D)
        parts["Isr"] = sr_eri_grad(cell, Gam, w, rcut_2e or (4.5 / w + 8.0), rcut_bra or 12.0)
    parts["nn"] = ewald_grad(cell, w if w is not None else 1.0)
    grad = sum(parts.values())
    return dict(e=e, grad=grad, parts=parts, comm=comm, p_err=p_err, D=D, it=it,
                qsym=abs(Q + Q.transpose(0, 2, 1, 3) + 1j * G.T[:, None, None, :] * P[None]).max())


def _fold_g(per_ao, aoat, natm):
    out = np.zeros((natm,) + per_ao.shape[1:], per_ao.dtype)
    np.add.at(out, aoat, per_ao)
    return out


def gamma_energy(cell, nelec, w=None, exxdiv=None, gcut=None, conv=1e-12, rcut_1e=22.0, **kw):
    ints = build_integrals(cell, w, rcut_1e=rcut_1e, verbose=False, exxdiv=exxdiv, gcut=gcut, **kw)
    return rhf(ints["S"], ints["h"], ints["I"], ints["enn"], nelec, conv=conv, kshift=ints["madelung"],
               maxiter=200)[0]


def displaced(cell, A, x, h):
    atoms = [(s, np.array(r, float)) for s, r in cell.atoms]
    atoms[A] = (atoms[A][0], atoms[A][1] + h * np.eye(3)[x])
    return Cell(cell.a, atoms, cell.basis)


def fd_grad(cell, nelec, comps, h=1e-4, **kw):
    """Central finite difference of the prototype's own Gamma energy for [(A, x), ...]."""
    return {(A, x): (gamma_energy(displaced(cell, A, x, h), nelec, **kw)
                     - gamma_energy(displaced(cell, A, x, -h), nelec, **kw)) / (2 * h) for A, x in comps}
