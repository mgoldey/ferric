"""Iteration 19: Gamma-point analytic STRESS tensor for the periodic prototype (RHF / UHF / RKS / UKS, dense AFT J/K and
the Ewald-split SR route, exxdiv none / ewald).

DEFINITION.  Homogeneous strain r -> (1 + eps) r applied to the lattice rows AND the atoms (fixed fractional coordinates):
a_i -> (1 + eps) a_i, R_A -> (1 + eps) R_A.  sigma_ij = (1/Omega) dE/d eps_ij (PySCF rks_stress convention, same sign).
Basis functions keep their exponents and move with their atoms in every image; the G-vectors become
G = 2 pi n b(eps) = (1 + eps)^{-T} G_0 with the Miller indices n FIXED; Omega -> Omega det(1 + eps).

FIXED INDEX SETS (what makes E(eps) smooth).  Every truncated set is chosen ONCE at the reference cell and then kept as
integer indices: the G sphere (Miller indices n), the Ewald G sphere, and every real-space image list (lattice indices).
`FixedCell` does this (gvectors / translations are looked up by cutoff and scaled with the strained a / b).  Re-selecting
the sphere |G| <= gcut at every strain makes E(eps) discontinuous (G-vectors crossing the sphere) -- measured in the
anchor runner.  The analytic stress below is the exact derivative of the fixed-index energy; at a truncated G set it
differs from the derivative of the converged energy by a "Pulay stress" of the plane-wave kind that vanishes with gcut.

ENERGY (Iterations 16/17 conventions; alpha = exact-exchange fraction, v_M Madelung (exxdiv='ewald'), c0 = pi/(w^2 Omega),
0 for pure AFT):
    E = sum D (T + V_SR + V_LR + c0 Z S) + E_SR2 + E_LR2 - c0 [1/2 tr(DS)^2 - alpha/2 sum_s tr(D_s S D_s S)]
        - alpha v_M/2 sum_s tr(D_s S D_s S) + E_xc + E_nn
    V_LR = -(1/Omega) sum_G v(G) Re[conj(P(G)) S(G)],   E_LR2 = (1/Omega) sum_G v(G) Re sum_mn conj(P_mn) Z_mn,
    Z_mn(G) = 1/2 D_mn rho(G) - alpha/2 sum_s (D_s P D_s)_mn,  rho = sum D P,  v = 4 pi/G^2 [e^{-G^2/4w^2}].

STRAIN DERIVATIVE (dE/d eps_ij; every piece is "sum over centres (dE/dX_c,i) X_c,j" plus explicit G / Omega terms):
 1. Overlap-coupled (Pulay, c0 and Madelung S-terms, as for forces): sum M dS/d eps,
    M = -sum_s D_s F_s D_s + c0 (Z - N) D + alpha (c0 - v_M) sum_s D_s S D_s,
    dS_mn/d eps_ij = -sum_L <d_i m | n_L> (A_m - B_n - L)_j      (per-IMAGE derivative weighted by the pair vector).
 2. Kinetic: sum D dT, same per-image virial with int1e_ipkin.
 3. Pair FT: P_prim(G) = e^{-iG.Pc} R(G, AB) with G.Pc strain-invariant, so
    dP/d eps_ij = [Q_i + i G_i (a/p) P] (A - B')_j + G_i G_j/(2p) P - G_i Pg_j,
    Q = bra-centre derivative (Iteration 16), Pg_j = common * dF_j/dG_j * prod_{d != j} F_d (G-derivative of the
    Hermite polynomial factor), B' = B + L the ket image.  The structure factor S(G) = sum Z e^{-iG.R} is invariant.
 4. Kernel and volume: dv/d eps_ij = dv/d(G^2) (-2 G_i G_j), d(1/Omega)/d eps_ij = -delta_ij / Omega, so every
    reciprocal energy E_LR = (1/Omega) sum v Phi gives -delta_ij E_LR + (1/Omega) sum [dv Phi + v dPhi].
 5. G = 0 bookkeeping (Ewald-split route only): -delta_ij E_c0 (c0 ~ 1/Omega) + its S part inside M.
 6. Madelung: -alpha/2 sum_s tr(D_s S D_s S) dv_M/d eps, dv_M = -2 x Ewald stress of one unit charge + background.
 7. Ewald E_nn: SR 1/2 sum Z Z f'(d) d_i d_j / d, LR -delta E_lr + (2 pi/Omega) sum |S|^2 k'(G^2)(-2 G_i G_j),
    background -delta_ij E_g0; the self term is strain-free.
 8. SR V_ne (erfc 3c, Gaussian nuclei): 2 sum D_mn Z_C sum_{L,K} (d_i m n_L | C_K)(A_m - X_C - K)_j   (ket slot by
    relabelling, as for forces).   SR ERI (erfc 4c): -sum Gam sum_{LMN} (d_i m n|l s)(3 X_m - X_n - X_l - X_s)_j
    (origin at the bra; the n / l / s slots map onto the bra slot by the 8-fold symmetry of the lattice sum).
 9. XC (atom-centred grid, points rigidly attached to their home atom, offsets unstrained):
    AO part: d chi_m(r_g)/d eps_ij = sum_L d_i chi_m(r_g - X_L) (O_g - X_L)_j with O_g = home atom position (the point
    moves with it); GGA: the same with the AO Hessian for d(grad chi).  Weight part: sum_g e(r_g) dW_g/d eps_ij,
    dW/d eps_ij = w0 sum_B dP/dX_B,i (X_B - R_home)_j over IMAGE atoms B.  Uniform grid (construction B): O_g = r_g
    (points at fixed fractional coordinates) and dW/d eps_ij = delta_ij W (weights Omega/n^3).

_MUTANT (module global) switches on deliberate defects for the mutation tests:
    'no_pulay'        drop sum M dS (overlap / Pulay term)
    'ft_no_centres'   pair FT: drop the basis-centre (AB) part of dP  (basis functions do not move)
    'g_unstrained'    G treated as unstrained: drop the G parts of dP and dv (the "fixed Cartesian G" mistake)
    'no_volume'       drop the -delta_ij E_LR volume terms of V_LR / ERI_LR
    'no_ewald_lr'     drop the reciprocal part of the Ewald stress
    'no_ewald_bg'     drop the Ewald background (G = 0) volume term
    'no_madelung'     drop dv_M/d eps (exxdiv='ewald' only)
    'madelung_s'      drop -alpha v_M sum D_s S D_s from M (the force-level Madelung term)
    'no_c0_volume'    drop -delta_ij E_c0 (Ewald-split route only)
    'no_sr_images'    SR 3c/4c virial with the image translations dropped from the pair vectors
    'xc_no_weight'    drop the grid-weight strain derivative     'xc_no_ao'  drop the XC AO strain term
    'xc_point_fixed'  atom grid: anchor O_g = r_g (points treated as fractional, the uniform-grid formula)
  RS-GDF only: 'no_metric' (drop sum Wm dJ2), 'no_aux_ft' (aux FT strain-free in J3 AND J2), 'no_aux_ft3' (J3 only),
    'no_j2_g0_vol' / 'no_j3_g0_vol' (drop the c0 ~ 1/Omega volume term of -c0 q q^T / -c0 S q^T), 'no_g0' (drop -c0 dS q)
"""

from __future__ import annotations

import numpy as np
from pyscf import gto
from pyscf.dft import libxc
from scipy.special import erfc

import pbc_dft as pd
import pbc_grad as PGd
import pbc_grad_open as PGo
from pbc_gamma import Cell, _lattice_points, build_integrals, cart_comps, hermite_E, pair_ft, shell_table

_MUTANT = None
EYE = np.eye(3)
HIDX = PGo.HIDX


# ================================================================================================= fixed index sets
class FixedCell(Cell):
    """Cell whose G spheres and image lists are frozen as integer indices at the reference geometry."""

    def __init__(self, a, atoms, basis, sets=None):
        super().__init__(a, atoms, basis)
        self.frozen = sets is not None
        self.sets = sets if sets is not None else {"g": {}, "t": {}}

    def gvectors(self, gcut):
        key, tab = round(float(gcut), 9), self.sets["g"]
        if key not in tab:
            if self.frozen:
                raise KeyError(f"G sphere gcut={gcut} was not fixed at the reference cell")
            tab[key] = np.rint(Cell.gvectors(self, gcut) @ self.a.T / (2 * np.pi)).astype(int)
        return tab[key] @ self.b

    def translations(self, rcut):
        key, tab = round(float(rcut), 9), self.sets["t"]
        if key not in tab:
            if self.frozen:
                raise KeyError(f"image list rcut={rcut} was not fixed at the reference cell")
            tab[key] = np.rint(Cell.translations(self, rcut) @ np.linalg.inv(self.a)).astype(int)
        return tab[key] @ self.a

    def strained(self, eps):
        F = EYE + np.asarray(eps, float)
        return FixedCell(self.a @ F.T, [(s, F @ np.asarray(r, float)) for s, r in self.atoms], self.basis, self.sets)


def rescaled_strain_cell(cell, eps):
    """The MISTAKE: a plain Cell (sphere and image lists re-selected at the strained geometry)."""
    F = EYE + np.asarray(eps, float)
    return Cell(cell.a @ F.T, [(s, F @ np.asarray(r, float)) for s, r in cell.atoms], cell.basis)


# ========================================================================================================= Ewald
def ewald_stress(cell, Z, R, w, rcut=None, gcut=None, mutable=True):
    """(dE/d eps (3,3), parts) of pbc_gamma._ewald(cell, Z, R, w) with the same cutoffs."""
    rcut = rcut or 7.0 / w
    gcut = gcut or 2 * w * np.sqrt(np.log(1e16))
    Z, R = np.asarray(Z, float), np.asarray(R, float).reshape(-1, 3)
    sr = np.zeros((3, 3))
    pad = np.linalg.norm(cell.a.sum(0)) + np.ptp(R, axis=0).max() if len(R) > 1 else 0.0
    for L in _lattice_points(cell.a, rcut + 2 + pad):
        dv = R[:, None, :] - R[None, :, :] - L
        d = np.linalg.norm(dv, axis=2)
        m = d > 1e-12
        ds = np.where(m, d, 1.0)
        fp = -(erfc(w * ds) / ds**2 + 2 * w / np.sqrt(np.pi) * np.exp(-((w * ds) ** 2)) / ds)
        sr += 0.5 * np.einsum("ab,abi,abj->ij", np.where(m, Z[:, None] * Z[None, :] * fp / ds, 0.0), dv, dv)
    G = cell.gvectors(gcut)
    G = G[np.einsum("gi,gi->g", G, G) > 1e-12]
    G2 = np.einsum("gi,gi->g", G, G)
    S2 = abs(np.exp(-1j * G @ R.T) @ Z) ** 2
    k = np.exp(-G2 / (4 * w * w)) / G2
    e_lr = 2 * np.pi / cell.vol * np.sum(k * S2)
    kp = -k * (1 / (4 * w * w) + 1 / G2)
    lr = -e_lr * EYE + 2 * np.pi / cell.vol * np.einsum("g,gi,gj->ij", S2 * kp, -2 * G, G)
    e_g0 = -np.pi * Z.sum() ** 2 / (2 * cell.vol * w * w)
    bg = -e_g0 * EYE
    mut = _MUTANT if mutable else None
    parts = dict(sr=sr, lr=0 * lr if mut == "no_ewald_lr" else lr, bg=0 * bg if mut == "no_ewald_bg" else bg)
    return sum(parts.values()), parts


def madelung_stress(cell):
    """d v_M / d eps  (v_M = -2 E_ewald(one unit charge), w = 1 as pbc_gamma.madelung)."""
    return -2.0 * ewald_stress(cell, [1.0], np.zeros((1, 3)), 1.0, mutable=False)[0]


# ============================================================================================ one-electron virials
def latsum_virial(cell, intor, rcut_1e):
    """V[i,j,m,n] = dX_mn/d eps_ij for X_mn = sum_L <m_0|O|n_L>:  -sum_L <d_i m|O|n_L> (A_m - B_n - L)_j."""
    mol = cell.mol
    nao, nb0 = mol.nao, mol.nbas
    L1 = cell.translations(rcut_1e)
    sm = cell.supermol(L1)
    X = sm.intor(intor, comp=3, shls_slice=(0, nb0, 0, sm.nbas)).reshape(3, nao, len(L1), nao)
    Am = cell.R[PGd.ao_atom(mol)]
    rel = Am[:, None, None, :] - Am[None, None, :, :] - L1[None, :, None, :]  # (m, L, n, j)
    if _MUTANT == "no_sr_images":
        rel = rel + L1[None, :, None, :]
    return -np.einsum("imLn,mLnj->ijmn", X, rel)


# ================================================================================================ pair FT strain
def pair_ft_strain(cell, G, thresh=1e-14):
    """(P, dP): P[m,n,g] raw Cartesian exactly as pbc_gamma.pair_ft; dP[i,j,m,n,g] = dP/d eps_ij at fixed Miller indices."""
    mol = cell.mol
    sh = shell_table(mol)
    nao, ng = mol.nao, len(G)
    G2 = np.einsum("gi,gi->g", G, G)
    P = np.zeros((nao, nao, ng), complex)
    dP = np.zeros((3, 3, nao, nao, ng), complex)
    amin = min(s["exps"].min() for s in sh)
    rpair = np.sqrt(2 * np.log(1 / thresh) / amin) + 2.0
    Ls = cell.translations(rpair)
    lmax = max(s["l"] for s in sh)
    nt = 2 * lmax + 4
    powg = [np.stack([(-1j * G[:, d]) ** t for t in range(nt)]) for d in range(3)]
    dpowg = [np.stack([t * (-1j) * (-1j * G[:, d]) ** max(t - 1, 0) for t in range(nt)]) for d in range(3)]
    no_centres, g_fixed = _MUTANT == "ft_no_centres", _MUTANT == "g_unstrained"
    for sa in sh:
        ca, la = cart_comps(sa["l"]), sa["l"]
        for sb in sh:
            cb, lb = cart_comps(sb["l"]), sb["l"]
            bP = np.zeros((len(ca), len(cb), ng), complex)
            bD = np.zeros((3, 3, len(ca), len(cb), ng), complex)
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
                        Gm = G[gm].T  # (3, g)
                        common = cA * cB * (np.pi / p) ** 1.5 * np.exp(-G2[gm] / (4 * p) - 1j * (G[gm] @ Pc))
                        F, Fd, Fg = [], [], []
                        for d in range(3):
                            E = hermite_E(la + 1, lb, a, b, AB[d])
                            f = np.einsum("ijt,tg->ijg", E, powg[d][: E.shape[2], gm])
                            fg = np.einsum("ijt,tg->ijg", E[: la + 1], dpowg[d][: E.shape[2], gm])
                            fd = 2 * a * f[1:]
                            fd[1:] -= np.arange(1, la + 1)[:, None, None] * f[:la]
                            F.append(f[: la + 1])
                            Fd.append(fd)
                            Fg.append(fg)
                        GG = Gm[:, None, :] * Gm[None, :, :] / (2 * p)
                        for u, (ax, ay, az) in enumerate(ca):
                            for v, (bx, by, bz) in enumerate(cb):
                                fx, fy, fz = F[0][ax, bx], F[1][ay, by], F[2][az, bz]
                                p0 = common * fx * fy * fz
                                q = common * np.stack([Fd[0][ax, bx] * fy * fz, fx * Fd[1][ay, by] * fz,
                                                       fx * fy * Fd[2][az, bz]])
                                pg = common * np.stack([Fg[0][ax, bx] * fy * fz, fx * Fg[1][ay, by] * fz,
                                                        fx * fy * Fg[2][az, bz]])
                                bP[u, v, gm] += p0
                                t = np.zeros((3, 3, len(gm)), complex)
                                if not no_centres:
                                    t += np.einsum("ig,j->ijg", q + 1j * (a / p) * Gm * p0, AB)
                                if not g_fixed:
                                    t += GG * p0 - Gm[:, None, :] * pg[None, :, :]
                                bD[:, :, u, v, gm] += t
            so, to = sa["off"], sb["off"]
            P[so : so + len(ca), to : to + len(cb)] = bP
            dP[:, :, so : so + len(ca), to : to + len(cb)] = bD
    return P, dP


# ================================================================================================= SR (erfc) virials
def sr_vne_virial(cell, D, w, rcut_1e):
    """d(sum D V_SR)/d eps = 2 sum D_mn Z_C sum_{L,K} (d_i m n_L|C_K)(A_m - X_{C,K})_j."""
    mol = cell.mol
    nao, nb0 = mol.nao, mol.nbas
    L1 = cell.translations(rcut_1e)
    sm = cell.supermol(L1)
    fm = gto.fakemol_for_charges(sm.atom_coords())
    fm.cart = True
    mm = sm + fm
    mm.cart = True
    Xc = sm.atom_coords()
    Zs = np.tile(cell.Z, len(L1))
    Am = cell.R[PGd.ao_atom(mol)]
    out = np.zeros((3, 3))
    with mm.with_range_coulomb(-w):
        for k in range(len(L1)):
            y = mm.intor("int3c2e_ip1_cart", comp=3, shls_slice=(0, nb0, k * nb0, (k + 1) * nb0, sm.nbas, mm.nbas))
            t = 2 * np.einsum("xmnc,mn,c->xmc", y.reshape(3, nao, nao, -1), D, Zs)
            rel = Am[:, None, :] - Xc[None, :, :]
            if _MUTANT == "no_sr_images":  # image translation of the nucleus dropped
                rel = Am[:, None, :] - np.tile(cell.R, (len(L1), 1))[None]
            out += np.einsum("xmc,mcj->xj", t, rel)
    return out


def sr_eri_virial(cell, Gam, w, rcut_2e, rcut_bra):
    """sum Gam dI_SR/d eps = -sum Gam sum_{LMN} (d_i m n_L|l_M s_N)_erfc (3 X_m - X_n - X_l - X_s)_j."""
    mol = cell.mol
    nao, nb0 = mol.nao, mol.nbas
    L2 = cell.translations(rcut_2e)
    nbra = int(np.sum(np.linalg.norm(L2, axis=1) <= rcut_bra + 1e-9))
    sm2 = cell.supermol(L2)
    n2 = len(L2)
    Am = cell.R[PGd.ao_atom(mol)]
    base = 3 * Am[:, None, None, None, :] - Am[None, :, None, None, :] - Am[None, None, :, None, :] \
        - Am[None, None, None, :, :]  # (m, n, l, s, j)
    imgs = _MUTANT != "no_sr_images"
    out = np.zeros((3, 3))
    with sm2.with_range_coulomb(-w):
        for k in range(n2):
            blk = sm2.intor(
                "int2e_ip1_cart", comp=3,
                shls_slice=(0, nb0, 0, nbra * nb0, k * nb0, (k + 1) * nb0, 0, sm2.nbas),
            ).reshape(3, nao, nbra, nao, nao, n2, nao)
            Y0 = blk.sum(axis=(2, 5))
            out -= np.einsum("xmnls,mnls,mnlsj->xj", Y0, Gam, base - (L2[k] if imgs else 0.0))
            if imgs:
                YL = np.einsum("xmbnlNs,bj->xjmnls", blk, L2[:nbra]) + np.einsum("xmbnlNs,Nj->xjmnls", blk, L2)
                out += np.einsum("xjmnls,mnls->xj", YL, Gam)
    return out


# ============================================================================================================ XC
def partition_weight_strain(pts, home, X, Zs, D, scheme, adjust):
    """(w, dws[p,i,j]) with w == pbc_dft.partition_weight and dws = dw/d eps_ij at a point rigidly attached to its home
    atom: sum_B dw/dX_B,i (X_B - X_home)_j over IMAGE atoms B (the point motion is the -sum_B d/dX_B term).
    Same algebra as pbc_grad_open.partition_weight_deriv, stopped before the fold to cell atoms."""
    dv = pts[:, None, :] - X[None]
    d = np.linalg.norm(dv, axis=2)
    mask = d <= D
    Rv = X[:, None, :] - X[None, :, :]
    Rbc = np.linalg.norm(Rv, axis=2)
    np.fill_diagonal(Rbc, 1.0)
    e = Rv / Rbc[..., None]
    acorr = pd.size_adjust(Zs, adjust)
    mu = (d[:, :, None] - d[:, None, :]) / Rbc[None]
    nu = mu + acorr[None] * (1 - mu * mu)
    g, gp = PGo._smooth_d(nu, scheme)
    s = 0.5 * (1 - g)
    q = -0.5 * gp * (1 - 2 * acorr[None] * mu)
    di = np.arange(len(X))
    s[:, di, di] = 1.0
    q[:, di, di] = 0.0
    s = np.where(mask[:, None, :], s, 1.0)
    q = np.where(mask[:, None, :], q, 0.0)
    P = s.prod(axis=2) * mask
    tot = P.sum(axis=1)
    ok = tot > 0
    tsafe = np.where(ok, tot, 1.0)
    w = np.where(ok, P[:, home] / tsafe, 0.0)
    pre = np.ones_like(s)
    pre[:, :, 1:] = np.cumprod(s[:, :, :-1], axis=2)
    suf = np.ones_like(s)
    suf[:, :, :-1] = np.cumprod(s[:, :, :0:-1], axis=2)[:, :, ::-1]
    coef = q * pre * suf * mask[:, :, None] / Rbc[None]
    u = dv / np.where(d > 0, d, 1.0)[..., None]
    selfd = -coef.sum(2)[..., None] * u - np.einsum("pbc,pbc,bcx->pbx", coef, mu, e)
    dPsum = selfd + np.einsum("pbd,pdx->pdx", coef, u) + np.einsum("pbd,pbd,bdx->pdx", coef, mu, e)
    dPh = coef[:, home, :, None] * (u + mu[:, home, :, None] * e[home][None])
    dPh[:, home] += selfd[:, home]
    dwX = (dPh - w[:, None, None] * dPsum) / tsafe[:, None, None] * ok[:, None, None]
    return w, np.einsum("pbi,bj->pij", dwX, X - X[home])


class StrainGrid:
    """Grid + AO values + AO strain derivatives.  kind='atom': pbc_grad_open.GradGrid's A2 grid point-for-point with
    dW[p,i,j]; kind='uniform': construction B (fractional points, weights Omega/n^3)."""

    def __init__(self, cell, kind="atom", n_rad=40, n_ang=50, D=10.0, n=None, scheme="ssf", adjust=True, deriv=2,
                 box=1.5, chunk=64, ao=True):
        self.kind = kind
        if kind == "uniform":
            f = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3) / n
            self.coords = f @ cell.a
            self.weights = np.full(len(f), cell.vol / len(f))
            self.anchor = self.coords.copy()
            self.dW = self.weights[:, None, None] * EYE[None]
        else:
            R, Z = cell.R, cell.mol.atom_charges()
            coords, weights, anch, dWs, pids = [], [], [], [], []
            for A in range(len(R)):  # identical point set / order to pbc_grad_open.GradGrid
                off, w0, _ = pd.atomic_grid(int(Z[A]), n_rad, n_ang)
                pts = R[A] + off
                keep = np.linalg.norm(off, axis=1) <= D
                pts, w0 = pts[keep], w0[keep]
                pid0 = np.nonzero(keep)[0]
                nb_xyz, nb_z, nb_idx = pd.image_atoms(cell.a, R, Z, R[A], 2 * D + 2 * box)
                home = int(np.nonzero((nb_idx == A) & (np.linalg.norm(nb_xyz - R[A], axis=1) < 1e-9))[0][0])
                key = np.floor(pts / box).astype(np.int64)
                _, inv = np.unique(key, axis=0, return_inverse=True)
                inv = inv.ravel()
                for b in range(inv.max() + 1):
                    ib = np.nonzero(inv == b)[0]
                    p = pts[ib]
                    cen = (np.floor(p[0] / box) + 0.5) * box
                    near = np.linalg.norm(nb_xyz - cen, axis=1) <= D + 0.87 * box + 1e-9
                    near[home] = True
                    sel = np.nonzero(near)[0]
                    hs = int(np.nonzero(sel == home)[0][0])
                    ck = max(1, int(chunk * 4e4 / max(len(sel), 1) ** 2))
                    for q0 in range(0, len(ib), ck):
                        pp = p[q0 : q0 + ck]
                        w, dws = partition_weight_strain(pp, hs, nb_xyz[sel], nb_z[sel], D, scheme, adjust)
                        ww0 = w0[ib[q0 : q0 + ck]]
                        coords.append(pp)
                        weights.append(ww0 * w)
                        dWs.append(ww0[:, None, None] * dws)
                        anch.append(np.repeat(R[A][None], len(pp), 0))
                        pids.append(A * 10**7 + pid0[ib[q0 : q0 + ck]])
            self.pid = np.concatenate(pids)
            self.coords, self.weights = np.vstack(coords), np.concatenate(weights)
            self.dW, self.anchor = np.concatenate(dWs), np.vstack(anch)
            if _MUTANT == "xc_point_fixed":
                self.anchor = self.coords.copy()
        self.ao, self.sao = ao_strain(cell, self.coords, self.anchor, deriv) if ao else (None, None)

    @property
    def size(self):
        return len(self.weights)


def ao_strain(cell, pts, anchor, deriv=2, thresh=1e-15, chunk=128):
    """(ao (comp, p, nao), sao (comp, 3, p, nao)):  ao = lattice-summed AO derivatives (as pbc_grad_open.ao_gamma_d2),
    sao[c, j] = sum_L chi^c(r - X_L) (O - X_L)_j with X_L the AO centre of image L and O the point's anchor."""
    mol = cell.mol
    nao = mol.nao
    rc = pd.ao_rcut(mol, thresh)
    comp = {1: 4, 2: 10}[deriv]
    name = {1: "GTOval_cart_deriv1", 2: "GTOval_cart_deriv2"}[deriv]
    cen = cell.R.mean(0)
    reach = np.linalg.norm(pts - cen, axis=1).max() + rc
    diam = np.linalg.norm(cell.R - cen, axis=1).max()
    Ls = pd.lattice_points(cell.a, reach + diam)
    Ls = Ls[np.linalg.norm(cell.R[None] + Ls[:, None] - cen, axis=2).min(1) <= reach]
    sm = cell.supermol(Ls)
    XL = cell.R[PGd.ao_atom(mol)][None] + Ls[:, None, :]  # (L, m, 3)
    ao = np.zeros((comp, len(pts), nao))
    sao = np.zeros((comp, 3, len(pts), nao))
    for p0 in range(0, len(pts), chunk):
        idx = np.arange(p0, min(p0 + chunk, len(pts)))
        v = np.asarray(sm.eval_gto(name, pts[idx], cutoff=thresh)).reshape(comp, len(idx), len(Ls), nao)
        ao[:, idx] = v.sum(2)
        rel = anchor[idx][:, None, None, :] - XL[None]  # (p, L, m, j)
        sao[:, :, idx] = np.einsum("cpLm,pLmj->cjpm", v, rel)
    return ao, sao


def xc_stress(sg, Da, Db, xc, restricted):
    """dict(ao, weight, exc): the XC strain derivative at a converged density on StrainGrid sg."""
    gga = pd.xc_family(xc) == "GGA"
    if pd.xc_family(xc) == "MGGA":
        raise NotImplementedError
    ao, sao, w = sg.ao, sg.sao, sg.weights
    chi, dchi = ao[0], ao[1:4]

    def dens(Dm):
        c = chi @ Dm
        r = np.einsum("pm,pm->p", c, chi)
        dr = 2 * np.einsum("ijpm,pm->ijp", sao[1:4], c)
        if not gga:
            return r, None, dr, None
        gr = 2 * np.einsum("pm,kpm->kp", c, dchi)
        dg = np.zeros((3, 3, 3, len(w)))
        for k in range(3):
            ck = dchi[k] @ Dm
            for i in range(3):
                dg[k, i] = 2 * (np.einsum("jpm,pm->jp", sao[HIDX[k][i]], c) + np.einsum("jpm,pm->jp", sao[1 + i], ck))
        return r, gr, dr, dg

    if restricted:
        r, gr, dr, dg = dens(Da + Db)
        exc, vxc = libxc.eval_xc(xc, np.vstack([r[None], gr]) if gga else r, spin=0, deriv=1)[:2]
        s_ao = np.einsum("p,ijp->ij", w * vxc[0], dr)
        if gga:
            s_ao += np.einsum("p,kp,kijp->ij", w * vxc[1], 2 * gr, dg)
        e = r * exc
    else:
        ra, ga, dra, dga = dens(Da)
        rb, gb, drb, dgb = dens(Db)
        ins = (np.vstack([ra[None], ga]), np.vstack([rb[None], gb])) if gga else (ra, rb)
        exc, vxc = libxc.eval_xc(xc, ins, spin=1, deriv=1)[:2]
        vr = vxc[0]
        s_ao = np.einsum("p,ijp->ij", w * vr[:, 0], dra) + np.einsum("p,ijp->ij", w * vr[:, 1], drb)
        if gga:
            vs = vxc[1]
            s_ao += np.einsum("p,kp,kijp->ij", w * vs[:, 0], 2 * ga, dga)
            s_ao += np.einsum("p,kp,kijp->ij", w * vs[:, 1], ga, dgb) + np.einsum("p,kp,kijp->ij", w * vs[:, 1], gb, dga)
            s_ao += np.einsum("p,kp,kijp->ij", w * vs[:, 2], 2 * gb, dgb)
        e = (ra + rb) * exc
    out = dict(ao=s_ao, weight=np.einsum("p,pij->ij", e, sg.dW), exc=np.sum(w * e))
    if _MUTANT == "xc_no_weight":
        out["weight"] = 0 * out["weight"]
    if _MUTANT == "xc_no_ao":
        out["ao"] = 0 * out["ao"]
    return out


# ================================================================================================== full stress
def gamma_stress(cell, ints, Da, Db, Fa, Fb, hyb, w=None, sgrid=None, xc=None, restricted=False, rcut_1e=22.0,
                 rcut_2e=None, rcut_bra=None, chunk_mb=200.0, eri=True):
    """(dE/d eps (3,3), parts) at a converged (D_s, F_s).  sigma = dE/d eps / cell.vol.
    eri=False: skip the dense two-electron pieces (the RS-GDF route adds its own, gdf_stress_2e)."""
    S, G, vM = ints["S"], ints["G"], ints["madelung"]
    mol = cell.mol
    nao = mol.nao
    D = Da + Db
    spins = (Da, Db)
    Fs = (Fa, Fb)
    parts = {}
    c0 = 0.0 if w is None else np.pi / (w * w * cell.vol)
    N = np.sum(D * S)
    Ztot = cell.Z.sum()
    DSDs = sum(Ds @ S @ Ds for Ds in spins)
    # 1-2: overlap-coupled and kinetic
    M = -sum(Ds @ F @ Ds for Ds, F in zip(spins, Fs)) + c0 * (Ztot - N) * D + hyb * c0 * DSDs
    if _MUTANT != "madelung_s":
        M = M - hyb * vM * DSDs
    dS = latsum_virial(cell, "int1e_ipovlp_cart", rcut_1e)
    dT = latsum_virial(cell, "int1e_ipkin_cart", rcut_1e)
    parts["S"] = 0 * dS[:, :, 0, 0] if _MUTANT == "no_pulay" else np.einsum("ijmn,mn->ij", dS, M)
    parts["T"] = np.einsum("ijmn,mn->ij", dT, D)
    # 3-4: reciprocal V_ne and ERI (chunked over G)
    P0 = pair_ft(cell, np.zeros((1, 3)))[..., 0].real
    nrm = np.sqrt(np.diag(S) / np.diag(P0))
    nn = nrm[:, None] * nrm[None, :]
    G2 = np.einsum("gi,gi->g", G, G)
    ex = np.exp(-G2 / (4 * w * w)) if w is not None else np.ones_like(G2)
    v = 4 * np.pi / G2 * ex
    dvdG2 = -v * (1 / G2 + (1 / (4 * w * w) if w is not None else 0.0))
    SG = np.exp(-1j * G @ cell.R.T) @ cell.Z
    EV = EI = 0.0
    sV, sI = np.zeros((3, 3)), np.zeros((3, 3))
    ch = max(1, int(chunk_mb * 1e6 / (144 * nao * nao)))
    p_err = 0.0
    for g0 in range(0, len(G), ch):
        sl = slice(g0, g0 + ch)
        Pr, dPr = pair_ft_strain(cell, G[sl])
        Pc, dPc = Pr * nn[:, :, None], dPr * nn[None, None, :, :, None]
        p_err = max(p_err, abs(Pc - ints["P"][:, :, sl]).max())
        Gc, vc = G[sl], v[sl]
        dv = dvdG2[sl] * (-2 * Gc.T[:, None, :] * Gc.T[None, :, :])  # (3,3,g)
        if _MUTANT == "g_unstrained":
            dv = 0 * dv
        rho = np.einsum("mn,mng->g", D, Pc)
        drho = np.einsum("mn,ijmng->ijg", D, dPc)
        phiV = -(np.conj(rho) * SG[sl]).real
        EV += np.sum(vc * phiV) / cell.vol
        sV += (np.einsum("ijg,g->ij", dv, phiV) - np.einsum("g,ijg->ij", vc, (np.conj(drho) * SG[sl]).real)) / cell.vol
        if not eri:
            continue
        Zt = 0.5 * D[:, :, None] * rho[None, None, :] - 0.5 * hyb * sum(
            np.einsum("ml,lsg,sn->mng", Ds, Pc, Ds) for Ds in spins)
        phiI = np.einsum("mng,mng->g", np.conj(Pc), Zt).real
        EI += np.sum(vc * phiI) / cell.vol
        sI += (np.einsum("ijg,g->ij", dv, phiI) + 2 * np.einsum("g,ijmng,mng->ij", vc, np.conj(dPc), Zt).real) / cell.vol
    vol = 0.0 if _MUTANT == "no_volume" else 1.0
    parts["Vlr"] = sV - vol * EV * EYE
    if eri:
        parts["Ilr"] = sI - vol * EI * EYE
    # 5: G = 0 volume term (Ewald-split route)
    if w is not None:
        Ec0 = c0 * Ztot * N - 0.5 * c0 * N * N + 0.5 * hyb * c0 * sum(np.trace(Ds @ S @ Ds @ S) for Ds in spins)
        parts["c0_vol"] = 0 * EYE if _MUTANT == "no_c0_volume" else -Ec0 * EYE
        parts["Vsr"] = sr_vne_virial(cell, D, w, rcut_1e)
        Gam = 0.5 * np.einsum("mn,ls->mnls", D, D) - 0.5 * hyb * sum(np.einsum("ml,ns->mnls", Ds, Ds) for Ds in spins)
        parts["Isr"] = sr_eri_virial(cell, Gam, w, rcut_2e or (4.5 / w + 8.0), rcut_bra or 12.0)
    # 6: Madelung
    if vM and _MUTANT != "no_madelung":
        tr = sum(np.trace(Ds @ S @ Ds @ S) for Ds in spins)
        parts["madelung"] = -0.5 * hyb * tr * madelung_stress(cell)
    # 7: Ewald
    parts["nn"] = ewald_stress(cell, cell.Z, cell.R, w if w is not None else 1.0)[0]
    # 9: XC
    if sgrid is not None and str(xc).upper() != "HF":
        xs = xc_stress(sgrid, Da, Db, xc, restricted)
        parts["xc_ao"], parts["xc_weight"] = xs["ao"], xs["weight"]
    return sum(parts.values()), dict(parts, p_err=p_err, EV=EV, EI=EI)


# ================================================================================================= energy + FD
def make_ints(cell, spec):
    return build_integrals(cell, spec.get("w"), verbose=False, exxdiv=spec.get("exxdiv", "none"), gcut=spec["gcut"],
                           **spec.get("rc", {}))


def energy_grid(cell, spec):
    g = spec.get("grid")
    if g is None or str(spec["xc"]).upper() == "HF":
        return None
    if g[0] == "uniform":
        return PGo.uniform_grad_grid(cell, g[1], deriv=1)
    return PGo.GradGrid(cell, n_rad=g[1], n_ang=g[2], D=g[3], want_dw=False, deriv=1)


def solve(cell, spec, guess=None, tol=1e-11):
    ints = make_ints(cell, spec)
    grid = energy_grid(cell, spec)

    def run(ms, g):
        return PGo.scf(ints["S"], ints["h"], pd.dense_jk(ints["I"]), ints["enn"], spec["na"], spec["nb"], grid, spec["xc"],
                       ms, spec.get("restricted", False), g, tol)

    try:
        r = run(ints["madelung"], guess)
    except RuntimeError:
        if guess is not None or not ints["madelung"]:
            raise
        # tri UHF triplet + ewald stalls from the core guess (DIIS); seed it with the exxdiv = none density
        r0 = run(0.0, None)
        r = run(ints["madelung"], (r0["Da"], r0["Db"]))
    return ints, grid, r


def stress_grid(cell, spec):
    g = spec.get("grid")
    if g is None or str(spec["xc"]).upper() == "HF":
        return None
    deriv = 2 if pd.xc_family(spec["xc"]) == "GGA" else 1
    if g[0] == "uniform":
        return StrainGrid(cell, "uniform", n=g[1], deriv=deriv)
    return StrainGrid(cell, "atom", n_rad=g[1], n_ang=g[2], D=g[3], deriv=deriv)


def analytic(cell, spec, ints, r):
    sg = stress_grid(cell, spec)
    if sg is not None:  # the stress grid must be the energy grid point-for-point
        eg = energy_grid(cell, spec)
        assert np.allclose(sg.coords, eg.coords, atol=1e-12) and np.allclose(sg.weights, eg.weights, atol=1e-14)
    return gamma_stress(cell, ints, r["Da"], r["Db"], r["Fa"], r["Fb"], r["hyb"], w=spec.get("w"), sgrid=sg,
                        xc=spec["xc"], restricted=spec.get("restricted", False), **_rc_kw(spec))


def _rc_kw(spec):
    rc = spec.get("rc", {})
    return {k: rc[k] for k in ("rcut_1e", "rcut_2e", "rcut_bra") if k in rc}


def _fd_energy(c, spec, guess):
    """FD energies: tol 1e-11 on max|X^T[F,D]X|; if DIIS stalls (seen once, tri UHF triplet), accept 1e-9 -- the energy
    error is second order in the residual (~1e-18 Ha), so the FD value is unaffected."""
    try:
        return solve(c, spec, guess)[2]["e"]
    except RuntimeError as err:
        print(f"    [fd] {err}; retrying with tol 1e-9", flush=True)
        return solve(c, spec, guess, tol=1e-9)[2]["e"]


def fd_stress(cell, spec, comps, guess, h=1e-4, rescaled=False):
    """Central FD of the prototype's own energy in eps_ij (fixed index sets unless rescaled=True)."""
    out = {}
    for i, j in comps:
        e = []
        for s in (1, -1):
            eps = np.zeros((3, 3))
            eps[i, j] = s * h
            c = rescaled_strain_cell(cell, eps) if rescaled else cell.strained(eps)
            e.append(_fd_energy(c, spec, guess))
        out[(i, j)] = (e[0] - e[1]) / (2 * h)
    return out


# ================================================================================================ RS-GDF (Iteration 18 energy)
def aux_ft_strain(auxmol, G):
    """(X, dX): X[P,g] = pbc_gdf.aux_ft (normalised Cartesian one-centre FT), dX[i,j,P,g] = dX/d eps_ij.
    X = e^{-iG.C} x(G) with G.C strain-invariant, so dX_ij = G_i G_j/(2a) X_prim - G_i (common dF_j/dG_j prod F)."""
    from pbc_gdf import _raw_ft, aux_ft

    G = np.asarray(G, float).reshape(-1, 3)
    G2 = np.einsum("gi,gi->g", G, G)
    naux = auxmol.nao
    X = np.zeros((naux, len(G)), complex)
    dX = np.zeros((3, 3, naux, len(G)), complex)
    off = 0
    for ib in range(auxmol.nbas):
        l, C = auxmol.bas_angular(ib), auxmol.atom_coord(auxmol.bas_atom(ib))
        exps, coef = auxmol.bas_exp(ib), auxmol._libcint_ctr_coeff(ib)
        comps = cart_comps(l)
        powg = [np.stack([(-1j * G[:, d]) ** t for t in range(l + 1)]) for d in range(3)]
        dpowg = [np.stack([t * (-1j) * (-1j * G[:, d]) ** max(t - 1, 0) for t in range(l + 1)]) for d in range(3)]
        phase = np.exp(-1j * (G @ C))
        for ic in range(auxmol.bas_nctr(ib)):
            for u, lxyz in enumerate(comps):
                for a, ca in zip(exps, coef[:, ic]):
                    common = ca * (np.pi / a) ** 1.5 * np.exp(-G2 / (4 * a)) * phase
                    F, Fg = [], []
                    for d in range(3):
                        E = hermite_E(lxyz[d], 0, a, 0.0, 0.0)[lxyz[d], 0][: l + 1]
                        F.append(E @ powg[d][: len(E)])
                        Fg.append(E @ dpowg[d][: len(E)])
                    x = common * F[0] * F[1] * F[2]
                    xg = common * np.stack([Fg[0] * F[1] * F[2], F[0] * Fg[1] * F[2], F[0] * F[1] * Fg[2]])
                    X[off + u] += x
                    if _MUTANT != "g_unstrained":
                        dX[:, :, off + u] += G.T[:, None, :] * G.T[None, :, :] / (2 * a) * x - G.T[:, None, :] * xg[None]
            off += len(comps)
    so = _raw_ft(auxmol, np.zeros((1, 3)))[1]
    nrm = np.sqrt(np.diag(auxmol.intor("int1e_ovlp_cart")) / so)  # pbc_gdf.aux_ft's calibration (strain-free)
    return X * nrm[:, None], dX * nrm[None, None, :, None], abs(X * nrm[:, None] - aux_ft(auxmol, G)).max()


def gdf_stress_2e(cell, gd, auxmol, Y, Wm, w, spherical):
    """(sum Y dJ3/d eps + sum Wm dJ2/d eps (3,3), parts) for pbc_gdf.build_gdf's primed J3 / J2 (same ranges gd['rng'])."""
    import pbc_grad_gdf as GG
    from pbc_gdf import _images

    mol = cell.mol
    nao, nb0 = mol.nao, mol.nbas
    rng = gd["rng"]
    c2s = GG.c2s_of(auxmol, spherical)
    Yc = np.einsum("mnk,pk->mnp", Y, c2s)
    Wc = c2s @ Wm @ c2s.T
    naux = auxmol.nao
    c0 = np.pi / (w * w * cell.vol)
    owner = np.zeros(naux, int)
    for k, (_, _, p0, p1) in enumerate(auxmol.aoslice_by_atom()):
        owner[p0:p1] = k
    Cp = auxmol.atom_coords()[owner]
    Am = cell.R[PGd.ao_atom(mol)]
    parts = {}
    imgs = 0.0 if _MUTANT == "no_sr_images" else 1.0
    # ---- SR 3c: 2 sum Y_mnP (d_m (m_0 n_L|P_T)) (A_m - C_P - T), d_m = -int3c2e_ip1 (ket slot by relabelling)
    L1 = cell.translations(rng["rcut_pair"])
    sm = cell.supermol(L1)
    T3 = cell.translations(rng["rcut_aux3"])
    out = np.zeros((3, 3))
    chunk = max(1, int(4e6 // (nao * nao * len(L1) * naux)))
    for t0 in range(0, len(T3), chunk):
        Tc = T3[t0 : t0 + chunk]
        big = gto.conc_mol(sm, _images(auxmol, Tc))
        with big.with_range_coulomb(-w):
            blk = big.intor("int3c2e_ip1_cart", comp=3, shls_slice=(0, nb0, 0, sm.nbas, sm.nbas, big.nbas)).reshape(
                3, nao, len(L1), nao, len(Tc), naux)
        b0 = np.einsum("xmLntp,mnp->xmtp", blk, Yc)
        rel = Am[:, None, None, :] - Cp[None, None, :, :] - imgs * Tc[None, :, None, :]
        out -= 2 * np.einsum("xmtp,mtpj->xj", b0, rel)
    parts["J3_sr"] = out
    # ---- SR 2c: sum Wc_PQ sum_T d_P (P_0|Q_T) (C_P - C_Q - T), d_P = -int2c2e_ip1
    T2 = cell.translations(rng["rcut_aux2"])
    img = _images(auxmol, T2)
    with img.with_range_coulomb(-w):
        d2 = img.intor("int2c2e_ip1_cart", comp=3, shls_slice=(0, auxmol.nbas, 0, img.nbas)).reshape(3, naux, len(T2), naux)
    rel2 = Cp[:, None, None, :] - Cp[None, None, :, :] - imgs * T2[None, :, None, :]
    parts["J2_sr"] = -np.einsum("xpTq,pq,pTqj->xj", d2, Wc, rel2)
    # ---- LR: J3 = (1/Om) sum v Re[conj(P_mn) X_P],  J2 = (1/Om) sum v Re[conj(X_P) X_Q]  (v = 4pi/G^2 e^{-G^2/4w^2})
    G = cell.gvectors(rng["gcut"])
    G = G[np.einsum("gi,gi->g", G, G) > 1e-12]
    G2 = np.einsum("gi,gi->g", G, G)
    v = 4 * np.pi / G2 * np.exp(-G2 / (4 * w * w))
    dv = -v * (1 / G2 + 1 / (4 * w * w)) * (-2 * G.T[:, None, :] * G.T[None, :, :])
    if _MUTANT == "g_unstrained":
        dv = 0 * dv
    X, dX, x_err = aux_ft_strain(auxmol, G)
    if _MUTANT == "no_aux_ft":
        dX = 0 * dX
    dX3 = 0 * dX if _MUTANT == "no_aux_ft3" else dX
    P0 = pair_ft(cell, np.zeros((1, 3)))[..., 0].real
    pn = np.sqrt(np.diag(gd["S"]) / np.diag(P0))
    Pr, dPr = pair_ft_strain(cell, G)
    P, dP = Pr * (pn[:, None] * pn[None, :])[:, :, None], dPr * (pn[:, None] * pn[None, :])[None, None, :, :, None]
    YX = np.einsum("mnp,pg->mng", Yc, X)
    phi3 = np.einsum("mng,mng->g", np.conj(P), YX).real
    vol = 0.0 if _MUTANT == "no_volume" else 1.0
    parts["J3_lr"] = (-vol * np.sum(v * phi3) * EYE + np.einsum("ijg,g->ij", dv, phi3)
                      + np.einsum("g,ijmng,mng->ij", v, np.conj(dP), YX).real
                      + np.einsum("g,mng,mnp,ijpg->ij", v, np.conj(P), Yc, dX3).real) / cell.vol
    WX = Wc @ X
    phi2 = np.einsum("pg,pg->g", np.conj(X), WX).real
    parts["J2_lr"] = (-vol * np.sum(v * phi2) * EYE + np.einsum("ijg,g->ij", dv, phi2)
                      + 2 * np.einsum("g,ijpg,pg->ij", v, np.conj(dX), WX).real) / cell.vol
    # ---- G = 0: J3 -= c0 S q^T, J2 -= c0 q q^T;  dc0/d eps = -c0 delta, q strain-free
    from pbc_gdf import aux_ft

    qc = aux_ft(auxmol, np.zeros((1, 3)))[:, 0].real
    Yq = np.einsum("mnp,p->mn", Yc, qc)
    dS = latsum_virial(cell, "int1e_ipovlp_cart", rng["rcut_pair"])
    parts["J3_g0_S"] = 0 * EYE if _MUTANT == "no_g0" else -c0 * np.einsum("ijmn,mn->ij", dS, Yq)
    parts["J3_g0_vol"] = 0 * EYE if _MUTANT == "no_j3_g0_vol" else c0 * np.sum(Yq * gd["S"]) * EYE
    parts["J2_g0_vol"] = 0 * EYE if _MUTANT == "no_j2_g0_vol" else c0 * (qc @ Wc @ qc) * EYE
    if _MUTANT == "no_metric":
        parts["J2_sr"], parts["J2_lr"], parts["J2_g0_vol"] = 0 * EYE, 0 * EYE, 0 * EYE
    return sum(parts.values()), dict(parts, x_err=x_err)


def gdf_solve(cell, spec, guess=None, tol=1e-11):
    import pbc_grad_gdf as GG

    ints = GG.integrals(cell, spec["gcut"], spec.get("exxdiv", "none"))
    aux = GG.make_auxmol(cell, spec["aux"])
    gd = GG.gdf(cell, aux, spec["w"], spherical=spec.get("spherical", True), lindep=spec.get("lindep", 0.0))
    grid = energy_grid(cell, spec)

    def run(ii, g, t):
        return GG.energy(cell, ii, gd, spec["na"], spec["nb"], spec["xc"], grid, spec.get("restricted", False), g, t)

    try:
        r = run(ints, guess, tol)
    except RuntimeError as err:
        if guess is not None:  # FD energy: accept a 1e-9 residual (second-order energy error)
            print(f"    [fd] {err}; retrying with tol 1e-9", flush=True)
            r = run(ints, guess, 1e-9)
        elif ints["madelung"]:  # same core-guess stall as solve(): seed with the exxdiv = none density
            r0 = run(dict(ints, madelung=0.0), None, tol)
            r = run(ints, (r0["Da"], r0["Db"]), tol)
        else:
            raise
    return ints, gd, aux, r


def gdf_analytic(cell, spec, ints, gd, aux, r, metric="dk"):
    """Total RS-GDF stress: pure-AFT h / S / Ewald / Madelung / XC pieces of gamma_stress (eri=False) + gdf_stress_2e."""
    import pbc_grad_gdf as GG

    sg = stress_grid(cell, spec)
    s1, p1 = gamma_stress(cell, ints, r["Da"], r["Db"], r["Fa"], r["Fb"], r["hyb"], w=None, sgrid=sg, xc=spec["xc"],
                          restricted=spec.get("restricted", False), eri=False)
    Y, Wm, diag = GG.fit_densities(gd["J2"], gd["J3"], [r["Da"], r["Db"]], r["hyb"], spec.get("lindep", 0.0), metric)
    s2, p2 = gdf_stress_2e(cell, gd, aux, Y, Wm, spec["w"], spec.get("spherical", True))
    return s1 + s2, dict(p1, **p2, diag=diag)


def gdf_fd_stress(cell, spec, comps, guess, h=1e-4):
    out = {}
    for i, j in comps:
        e = []
        for s in (1, -1):
            eps = np.zeros((3, 3))
            eps[i, j] = s * h
            e.append(gdf_solve(cell.strained(eps), spec, guess)[3]["e"])
        out[(i, j)] = (e[0] - e[1]) / (2 * h)
    return out
