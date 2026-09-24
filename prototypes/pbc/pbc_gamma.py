"""Gamma-point periodic RHF prototype, built from MOLECULAR integral primitives.

The point: show that ferric's existing machinery (libint2 erfc operator, ghost-atom
image placement, point/Gaussian charges, McMurchie-Davidson E tables) plus ONE new
primitive (analytic Fourier transform of Gaussian pair densities) is enough for an
all-electron Gamma-point PBC Hartree-Fock. PySCF's *molecular* `intor` stands in for
libint2 here; PySCF's *pbc* module is used ONLY as the independent oracle.

Ewald split of every Coulomb interaction:  1/r = erfc(w r)/r  +  erf(w r)/r
  SR  (erfc): real-space lattice sums of ordinary molecular integrals over images
  LR  (erf) : reciprocal space, sum over G != 0 of 4pi/G^2 exp(-G^2/4w^2) with
              analytic pair-density FTs  P_mn(G) = sum_L FT[phi_m(r) phi_n(r-L)](G)
  G=0       : the real-space erfc sum implicitly contains the G=0 term pi/(w^2 Omega);
              it is subtracted so every term follows the standard "drop G=0" convention
              (J, V_ne, E_nn: cancels exactly for a neutral cell; K: == exxdiv=None).

At Gamma, everything collapses to a nao^4 real 8-fold-symmetric tensor
  I[m,n,l,s] = sum_{L,M,N} (m0 nL | lM sN)
so the SCF below is literally a molecular RHF on (S, h, I). Units: Bohr / Hartree.
Cartesian basis throughout (spherical is a cart2sph afterwards, not new physics).
"""

from __future__ import annotations

import itertools
import time

import numpy as np
from pyscf import gto


# ----------------------------------------------------------------------------- cell
class Cell:
    def __init__(self, a, atoms, basis):
        self.a = np.asarray(a, float)  # rows are lattice vectors (Bohr)
        self.atoms = atoms  # [(sym, (x,y,z))] Bohr
        self.basis = basis
        self.mol = gto.M(atom=atoms, basis=basis, unit="B", cart=True, verbose=0)
        self.vol = abs(np.linalg.det(self.a))
        self.b = 2 * np.pi * np.linalg.inv(self.a).T  # rows: reciprocal vectors
        self.Z = self.mol.atom_charges().astype(float)
        self.R = self.mol.atom_coords()

    def translations(self, rcut):
        """Lattice vectors L such that some atom of image L is within rcut of cell 0."""
        # conservative n-range from the shortest interplanar spacing
        spacing = 1.0 / np.linalg.norm(np.linalg.inv(self.a), axis=0)
        nmax = np.ceil(rcut / spacing).astype(int) + 1
        out = []
        for n in itertools.product(*(range(-k, k + 1) for k in nmax)):
            L = np.array(n) @ self.a
            d = np.linalg.norm(
                self.R[:, None, :] - (self.R[None, :, :] + L), axis=2
            ).min()
            if d <= rcut:
                out.append(L)
        out.sort(key=lambda v: np.linalg.norm(v))
        return np.array(out)  # out[0] == 0

    def gvectors(self, gcut):
        spacing = np.linalg.norm(self.b, axis=1)
        # |n_i| <= gcut * |a_i| / 2pi bounds any G inside the sphere
        nmax = np.ceil(gcut * np.linalg.norm(self.a, axis=1) / (2 * np.pi)).astype(int)
        del spacing
        n = np.array(list(itertools.product(*(range(-k, k + 1) for k in nmax))))
        G = n @ self.b
        keep = np.einsum("gi,gi->g", G, G) <= gcut**2
        return G[keep]

    def supermol(self, Ls):
        atoms = [(s, np.asarray(r) + L) for L in Ls for (s, r) in self.atoms]
        return gto.M(atom=atoms, basis=self.basis, unit="B", cart=True, verbose=0)


# ------------------------------------------------ analytic FT of Gaussian pair densities
def hermite_E(la, lb, a, b, Xab):
    """McMurchie-Davidson E^{ij}_t for one Cartesian direction (Helgaker 9.5)."""
    p = a + b
    mu = a * b / p
    XPA, XPB = -b / p * Xab, a / p * Xab
    E = np.zeros((la + 1, lb + 1, la + lb + 2))
    E[0, 0, 0] = np.exp(-mu * Xab * Xab)
    for i in range(la + 1):
        for j in range(lb + 1):
            if i == 0 and j == 0:
                continue
            if i > 0:
                src, X = E[i - 1, j], XPA
            else:
                src, X = E[i, j - 1], XPB
            for t in range(i + j + 1):
                E[i, j, t] = (
                    (src[t - 1] / (2 * p) if t > 0 else 0.0)
                    + X * src[t]
                    + (t + 1) * src[t + 1]
                )
    return E


def cart_comps(l):
    return [
        (lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)
    ]


def shell_table(mol):
    sh = []
    off = 0
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        nc = mol.bas_nctr(ib)
        exps = mol.bas_exp(ib)
        coef = mol._libcint_ctr_coeff(ib)
        center = mol.atom_coord(mol.bas_atom(ib))
        ncomp = (l + 1) * (l + 2) // 2
        for ic in range(nc):
            sh.append(
                dict(l=l, exps=exps, coef=coef[:, ic], A=center, off=off + ic * ncomp)
            )
        off += nc * ncomp
    return sh


def pair_ft(cell, G, thresh=1e-14):
    """P[m,n,g] = sum_L  FT[ phi_m(r) phi_n(r-L) ](G_g),  FT f(G) = int f(r) e^{-iG.r} dr.

    Unnormalised Cartesian components; normalisation is recovered by calibrating the
    G=0 value against the lattice-summed overlap (see ao_norms)."""
    mol = cell.mol
    sh = shell_table(mol)
    nao = mol.nao
    ng = len(G)
    G2 = np.einsum("gi,gi->g", G, G)
    P = np.zeros((nao, nao, ng), complex)
    amin = min(s["exps"].min() for s in sh)
    rpair = np.sqrt(2 * np.log(1 / thresh) / amin) + 2.0  # generous pair range
    Ls = cell.translations(rpair)
    lmax = max(s["l"] for s in sh)
    powg = [
        np.stack([(-1j * G[:, d]) ** t for t in range(2 * lmax + 2)]) for d in range(3)
    ]
    for sa in sh:
        ca = cart_comps(sa["l"])
        for sb in sh:
            cb = cart_comps(sb["l"])
            blk = np.zeros((len(ca), len(cb), ng), complex)
            for L in Ls:
                B = sb["A"] + L
                AB = sa["A"] - B
                for a, cA in zip(sa["exps"], sa["coef"]):
                    for b, cB in zip(sb["exps"], sb["coef"]):
                        p = a + b
                        pref = cA * cB * np.exp(-a * b / p * AB @ AB)
                        if abs(pref) < thresh:
                            continue
                        Pc = (a * sa["A"] + b * B) / p
                        # per-primitive-pair G window: e^{-G^2/4p} < thresh*e^{-10} is dropped
                        # (diffuse pairs need few G; changes nothing above thresh)
                        gm = np.nonzero(G2 < 4 * p * (np.log(1 / thresh) + 10))[0]
                        if len(gm) == 0:
                            continue
                        Gm = G[gm]
                        common = (
                            cA
                            * cB
                            * (np.pi / p) ** 1.5
                            * np.exp(-G2[gm] / (4 * p) - 1j * (Gm @ Pc))
                        )
                        F = []
                        for d in range(3):
                            E = hermite_E(sa["l"], sb["l"], a, b, AB[d])
                            F.append(
                                np.einsum("ijt,tg->ijg", E, powg[d][: E.shape[2], gm])
                            )
                        for u, (ax, ay, az) in enumerate(ca):
                            for v, (bx, by, bz) in enumerate(cb):
                                blk[u, v, gm] += (
                                    common * F[0][ax, bx] * F[1][ay, by] * F[2][az, bz]
                                )
            P[sa["off"] : sa["off"] + len(ca), sb["off"] : sb["off"] + len(cb)] = blk
    return P


# --------------------------------------------------------------------------- Ewald E_nn
def _ewald(cell, Z, R, w, rcut=None, gcut=None):
    """Ewald energy of point charges Z at R in the lattice of `cell`, with the uniform
    neutralising background (G=0 term) and without self-interaction."""
    from scipy.special import erfc

    rcut = rcut or 7.0 / w
    gcut = gcut or 2 * w * np.sqrt(np.log(1e16))
    Z, R = np.asarray(Z, float), np.asarray(R, float).reshape(-1, 3)
    e_sr = 0.0
    # translations are generated from the cell's own atoms; they only need to cover
    # |L| <= rcut + max|R_i - R_j|, so pad with the cell diameter
    pad = np.linalg.norm(cell.a.sum(0)) + np.ptp(R, axis=0).max() if len(R) > 1 else 0.0
    for L in _lattice_points(cell.a, rcut + 2 + pad):
        d = np.linalg.norm(R[:, None, :] - R[None, :, :] - L, axis=2)
        m = d > 1e-12
        e_sr += 0.5 * np.sum(
            (Z[:, None] * Z[None, :] * erfc(w * d) / np.where(m, d, 1))[m]
        )
    G = cell.gvectors(gcut)
    G = G[np.einsum("gi,gi->g", G, G) > 1e-12]
    G2 = np.einsum("gi,gi->g", G, G)
    SG = np.exp(-1j * G @ R.T) @ Z
    e_lr = 2 * np.pi / cell.vol * np.sum(np.exp(-G2 / (4 * w * w)) / G2 * abs(SG) ** 2)
    e_self = -w / np.sqrt(np.pi) * np.sum(Z * Z)
    e_g0 = -np.pi * Z.sum() ** 2 / (2 * cell.vol * w * w)
    return e_sr + e_lr + e_self + e_g0


def _lattice_points(a, rcut):
    """All n @ a with |n @ a| <= rcut."""
    spacing = 1.0 / np.linalg.norm(np.linalg.inv(a), axis=0)
    nmax = np.ceil(rcut / spacing).astype(int) + 1
    n = np.array(list(itertools.product(*(range(-k, k + 1) for k in nmax))))
    L = n @ a
    return L[np.linalg.norm(L, axis=1) <= rcut]


def ewald_nn(cell, w, rcut=None, gcut=None):
    return _ewald(cell, cell.Z, cell.R, w, rcut, gcut)


def madelung(cell, w=1.0):
    """Gamma-point Madelung constant v_M = -2 E_ewald(one unit charge + background).

    PySCF convention (pbc/tools/pbc.py madelung, omega=0 branch): v_M = -2*ecell.ewald()
    for a cell holding a single unit point charge; exxdiv='ewald' then adds
    v_M * S D S to K (df_jk._ewald_exxdiv_for_G0).  v_M > 0 and is w-independent."""
    return -2.0 * _ewald(cell, [1.0], np.zeros((1, 3)), w)


# ------------------------------------------------------------ real-space SR (molecular)
def sr_terms(cell, w, rcut_1e, rcut_2e, rcut_bra):
    """S, T, V_SR and the SR ERI tensor via ordinary molecular integrals over images."""
    mol = cell.mol
    nao, nb0 = mol.nao, mol.nbas

    # 1e: S, T need the full overlap range; V_SR needs nuclei within erfc range.
    L1 = cell.translations(rcut_1e)
    sm = cell.supermol(L1)
    S = (
        sm.intor("int1e_ovlp_cart", shls_slice=(0, nb0, 0, sm.nbas))
        .reshape(nao, len(L1), nao)
        .sum(1)
    )
    T = (
        sm.intor("int1e_kin_cart", shls_slice=(0, nb0, 0, sm.nbas))
        .reshape(nao, len(L1), nao)
        .sum(1)
    )
    # nuclei as unit Gaussian charges of exponent 1e16 (what ferric's SiteBasis/eri3 route would do)
    fm = gto.fakemol_for_charges(sm.atom_coords())
    fm.cart = True
    mm = sm + fm
    mm.cart = True
    with mm.with_range_coulomb(-w):
        v3 = mm.intor("int3c2e_cart", shls_slice=(0, nb0, 0, sm.nbas, sm.nbas, mm.nbas))
    Zs = np.tile(cell.Z, len(L1))
    V = -np.einsum("mnA,A->mn", v3, Zs).reshape(nao, len(L1), nao).sum(1)

    # 2e SR: I[m,n,l,s] = sum_{L,M,N} (m0 nL | lM sN)_erfc, bra images within rcut_bra,
    # ket images within rcut_2e. Chunked one ket image (third index) at a time.
    L2 = cell.translations(rcut_2e)
    nbra = int(np.sum(np.linalg.norm(L2, axis=1) <= rcut_bra + 1e-9))
    sm2 = cell.supermol(L2)
    n2 = len(L2)
    I = np.zeros((nao,) * 4)
    with sm2.with_range_coulomb(-w):
        for k in range(n2):
            blk = sm2.intor(
                "int2e_cart",
                shls_slice=(0, nb0, 0, nbra * nb0, k * nb0, (k + 1) * nb0, 0, sm2.nbas),
            ).reshape(nao, nbra, nao, nao, n2, nao)
            I += blk.sum(axis=(1, 4))
    return S, T, V, I, dict(n_img_1e=len(L1), n_img_2e=n2, n_bra=nbra)


# ------------------------------------------------------------------- assemble + SCF
def build_integrals(
    cell, w, rcut_1e=22.0, rcut_2e=None, rcut_bra=None, verbose=True, exxdiv=None
):
    """Return S, h, I (full periodic Gamma ERI), E_nn. w=None => pure reciprocal (no SR).

    exxdiv: None/'none' => G=0 of exchange dropped (PySCF exxdiv=None);
            'ewald'     => Madelung probe-charge correction, returned as 'madelung' (v_M);
                           pass it to rhf(kshift=...) so K -> K + v_M S D S."""
    if exxdiv is not None and str(exxdiv).lower() not in ("none", "ewald"):
        raise ValueError(f"exxdiv must be None, 'none' or 'ewald', got {exxdiv!r}")
    t0 = time.time()
    nao = cell.mol.nao
    if w is None:  # w -> infinity: all Coulomb in reciprocal space, SR vanishes
        L1 = cell.translations(rcut_1e)
        sm = cell.supermol(L1)
        nb0 = cell.mol.nbas
        S = (
            sm.intor("int1e_ovlp_cart", shls_slice=(0, nb0, 0, sm.nbas))
            .reshape(nao, len(L1), nao)
            .sum(1)
        )
        T = (
            sm.intor("int1e_kin_cart", shls_slice=(0, nb0, 0, sm.nbas))
            .reshape(nao, len(L1), nao)
            .sum(1)
        )
        Vsr = np.zeros((nao, nao))
        Isr = np.zeros((nao,) * 4)
        info = {}
        pmax = max(cell.mol.bas_exp(i).max() for i in range(cell.mol.nbas)) * 2
        gcut = 2 * np.sqrt(pmax * np.log(1e14))
        c0 = 0.0
        enn = ewald_nn(cell, 1.0)  # E_nn is w-independent; any split works
    else:
        rcut_2e = rcut_2e or (4.5 / w + 8.0)
        rcut_bra = rcut_bra or 12.0
        S, T, Vsr, Isr, info = sr_terms(cell, w, rcut_1e, rcut_2e, rcut_bra)
        gcut = 2 * w * np.sqrt(np.log(1e14))
        c0 = np.pi / (w * w * cell.vol)
        enn = ewald_nn(cell, w)
    t1 = time.time()
    G = cell.gvectors(gcut)
    G = G[np.einsum("gi,gi->g", G, G) > 1e-12]
    G2 = np.einsum("gi,gi->g", G, G)
    v = 4 * np.pi / G2 * (np.exp(-G2 / (4 * w * w)) if w is not None else 1.0)
    # normalisation: calibrate our raw Cartesian FT at G=0 against the lattice-summed
    # overlap (P(0) IS S), then check every element, not just the diagonal we fitted.
    P0 = pair_ft(cell, np.zeros((1, 3)))[..., 0].real
    nrm = np.sqrt(np.diag(S) / np.diag(P0))
    s_err = abs(nrm[:, None] * P0 * nrm[None, :] - S).max()
    P = pair_ft(cell, G) * nrm[:, None, None] * nrm[None, :, None]
    Pn = P.reshape(nao * nao, -1)
    # LR ERI: (1/Omega) sum_G v(G) conj(P_mn(G)) P_ls(G)
    Ilr = ((Pn.conj() * v) @ Pn.T).real.reshape((nao,) * 4) / cell.vol
    SG = np.exp(-1j * G @ cell.R.T) @ cell.Z
    Vlr = -((P.conj() * (v * SG)).sum(-1)).real / cell.vol
    t2 = time.time()

    # G=0 bookkeeping: remove the implicit pi/(w^2 Omega) term of the real-space erfc sums.
    Ztot = cell.Z.sum()
    V = Vsr + Vlr + c0 * Ztot * S
    I = Isr + Ilr - c0 * np.einsum("mn,ls->mnls", S, S)
    asym = abs(I - I.transpose(2, 3, 0, 1)).max()
    I = 0.5 * (I + I.transpose(2, 3, 0, 1))
    if verbose:
        print(
            f"  w={w}: SR {t1 - t0:.1f}s ({info}), LR {t2 - t1:.1f}s ({len(G)} G), "
            f"FT(G=0) vs S {s_err:.1e}, ERI (mn|ls)-(ls|mn) asym before symmetrising {asym:.1e}"
        )
    vm = madelung(cell) if str(exxdiv).lower() == "ewald" else 0.0
    return dict(S=S, T=T, V=V, h=T + V, I=I, enn=enn, P=P, G=G, madelung=vm)


def rhf(S, h, I, enn, nelec, conv=1e-10, maxiter=100, kshift=0.0, jk=None):
    """Molecular RHF on (S, h, I).  kshift = Madelung v_M: K -> K + v_M S D S
    (PySCF exxdiv='ewald'); the energy picks up -v_M/4 tr(DSDS) through K.
    jk: optional D -> (J, K) callable replacing the dense I contraction (e.g. GDF B tensors)."""
    s, U = np.linalg.eigh(S)
    X = U[:, s > 1e-8] / np.sqrt(s[s > 1e-8])
    nocc = nelec // 2
    D = np.zeros_like(S)
    from collections import deque

    focks, errs = deque(maxlen=8), deque(maxlen=8)
    e_old = 0.0
    for it in range(maxiter):
        if jk is None:
            J = np.einsum("mnls,ls->mn", I, D)
            K = np.einsum("mlsn,ls->mn", I, D)
        else:
            J, K = jk(D)
        if kshift:
            K = K + kshift * S @ D @ S
        F = h + J - 0.5 * K
        e = 0.5 * np.sum(D * (h + F)) + enn
        err = X.T @ (F @ D @ S - S @ D @ F) @ X
        if it > 0:  # it=0 is the D=0 core guess: zero commutator, useless to DIIS
            focks.append(F)
            errs.append(err)
        if len(focks) > 1:
            n = len(focks)
            B = -np.ones((n + 1, n + 1))
            B[-1, -1] = 0
            for i in range(n):
                for j in range(n):
                    B[i, j] = np.sum(errs[i] * errs[j])
            rhs = np.zeros(n + 1)
            rhs[-1] = -1
            c = np.linalg.lstsq(B, rhs, rcond=None)[0][:n]
            F = sum(ci * fi for ci, fi in zip(c, focks))
        eps, C = np.linalg.eigh(X.T @ F @ X)
        C = X @ C
        D = 2 * C[:, :nocc] @ C[:, :nocc].T
        if abs(e - e_old) < conv and abs(err).max() < 1e-7:
            return e, eps, it
        e_old = e
    raise RuntimeError("SCF not converged")
