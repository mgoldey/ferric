"""Gamma-point SUPERCELL integrals by folding primitive-cell lattice sums (translation symmetry).

Why: `pbc_gdf.build_gdf` on an explicit n1 x n2 x n3 supercell evaluates the SR 3c lattice sum as
an unscreened (nb_sc) x (nb_sc |L|) x (naux_sh |T|) block, i.e. O(N^2)..O(N^3) in the number of
cells N.  But every supercell quantity is a lattice sum over PRIMITIVE translations, grouped by
the residue of the translation modulo the supercell lattice:

    X_sc[(c,m),(c',n)]            = sum_{D == c'-c (mod n)}             X_prim[m, n_D]
    J3_sc[(c,m),(c',n),(c'',P)]   = sum_{D1 == c'-c, D2 == c''-c}        (m_0 n_D1 | P_D2)'
    J2_sc[(c,P),(c',Q)]           = sum_{T == c'-c}                      (P_0 | Q_T)'

so the SR real-space integrals are computed ONCE on the primitive cell (cost independent of N)
and folded; only the LR reciprocal sum uses the (finer) supercell G lattice, through per-residue
pair FTs  Q^r_mn(G) = sum_{D == r} FT[chi_m(x) chi_n(x - D)](G)  and the identity
P^sc_{(c,m),(c',n)}(G) = exp(-iG.t_c) Q^{res(c'-c)}_mn(G)  for G in the supercell reciprocal lattice.

Conventions are exactly those of pbc_gamma / pbc_gdf (G=0-dropped kernel, erfc/erf split, the
implicit pi/(w^2 Omega_sc) G=0 term of the erfc sums subtracted).  Validated against
`build_gdf` on explicit supercells and against PySCF pbc (tests).  Orbital basis Cartesian,
aux spherical by default.  Units Bohr/Hartree.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
from pyscf import gto

from pbc_gamma import Cell, cart_comps, ewald_nn, hermite_E, madelung, shell_table
from pbc_gdf import _images, _rcut_erfc, aux_ft


# ----------------------------------------------------------------------------- geometry
class Supercell:
    """Diagonal supercell n = (n1, n2, n3) of a primitive cell.  AO/aux/atom order is
    cell-major: index = c * n_prim + k, c = (c1*n2 + c2)*n3 + c3."""

    def __init__(self, a_prim, atoms_prim, basis, ncell, wrap=None):
        self.prim = Cell(a_prim, atoms_prim, basis)
        self.n = tuple(int(x) for x in ncell)
        self.R = int(np.prod(self.n))
        self.cvec = np.array(
            list(itertools.product(*(range(k) for k in self.n)))
        )  # (R, 3)
        self.t = self.cvec @ self.prim.a  # cell translations
        a_sc = np.diag(self.n) @ self.prim.a
        atoms = [(s, np.asarray(r, float) + t) for t in self.t for (s, r) in atoms_prim]
        if (
            wrap is not None
        ):  # optional: move atoms by supercell lattice vectors (physics-neutral)
            atoms = [
                (s, r + np.asarray(wrap(i, r), float) @ a_sc)
                for i, (s, r) in enumerate(atoms)
            ]
        self.atoms = atoms
        self.sc = Cell(a_sc, atoms, basis)
        # D[c, c'] = residue index of (c' - c) mod n
        self.D = np.array(
            [
                [self.res(self.cvec[cp] - self.cvec[c]) for cp in range(self.R)]
                for c in range(self.R)
            ]
        )

    def res(self, nvec):
        n1, n2, n3 = self.n
        m = np.mod(np.asarray(nvec, int), self.n)
        return int((m[0] * n2 + m[1]) * n3 + m[2])

    def lattice_ints(self, Ls):
        return np.rint(Ls @ np.linalg.inv(self.prim.a)).astype(int)

    def residues(self, Ls):
        return np.array([self.res(k) for k in self.lattice_ints(Ls)])

    def one_hot(self, Ls):
        r = self.residues(Ls)
        M = np.zeros((len(Ls), self.R))
        M[np.arange(len(Ls)), r] = 1.0
        return M

    def unfold2(self, F):
        """F[r, a, b] -> X[(c,a),(c',b)] = F[D[c,c'], a, b]."""
        R, na, nb = F.shape
        return F[self.D].transpose(0, 2, 1, 3).reshape(R * na, R * nb)

    def ao_shift_perm(self, axis, nao_p):
        """Index map for a translation by one primitive vector along `axis`:
        AO (c, m) -> (c + e_axis mod n, m).  (T C)[perm] = C."""
        e = np.zeros(3, int)
        e[axis] = 1
        tgt = np.array([self.res(self.cvec[c] + e) for c in range(self.R)])
        return (tgt[:, None] * nao_p + np.arange(nao_p)[None, :]).reshape(-1)


# ---------------------------------------------------------- per-residue pair FT (primitive AOs)
def pair_ft_residues(scell, G, thresh=1e-14):
    """Q[r, m, n, g] = sum_{D == r} FT[chi_m(x) chi_n(x - D)](G_g) (raw, unnormalised Cartesian;
    same kernel as pbc_gamma.pair_ft, but accumulated by residue of the primitive translation)."""
    prim = scell.prim
    sh = shell_table(prim.mol)
    nao, ng = prim.mol.nao, len(G)
    G2 = np.einsum("gi,gi->g", G, G)
    Q = np.zeros((scell.R, nao, nao, ng), complex)
    amin = min(s["exps"].min() for s in sh)
    Ls = prim.translations(np.sqrt(2 * np.log(1 / thresh) / amin) + 2.0)
    rs = scell.residues(Ls)
    lmax = max(s["l"] for s in sh)
    powg = [
        np.stack([(-1j * G[:, d]) ** t for t in range(2 * lmax + 2)]) for d in range(3)
    ]
    for sa in sh:
        ca = cart_comps(sa["l"])
        for sb in sh:
            cb = cart_comps(sb["l"])
            for L, r in zip(Ls, rs):
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
                        common = (
                            cA
                            * cB
                            * (np.pi / p) ** 1.5
                            * np.exp(-G2[gm] / (4 * p) - 1j * (G[gm] @ Pc))
                        )
                        F = []
                        for d in range(3):
                            E = hermite_E(sa["l"], sb["l"], a, b, AB[d])
                            F.append(
                                np.einsum("ijt,tg->ijg", E, powg[d][: E.shape[2], gm])
                            )
                        for u, (ax, ay, az) in enumerate(ca):
                            for v, (bx, by, bz) in enumerate(cb):
                                Q[r, sa["off"] + u, sb["off"] + v, gm] += (
                                    common * F[0][ax, bx] * F[1][ay, by] * F[2][az, bz]
                                )
    return Q


def _fold_1e(scell, intor, other=None, rcut=None):
    """F[r] = sum_{D == r} <m_0 | op | n_D> (n from `other` basis if given)."""
    prim = scell.prim
    Ls = prim.translations(rcut)
    if other is None:
        sm = prim.supermol(Ls)
        v = sm.intor(intor, shls_slice=(0, prim.mol.nbas, 0, sm.nbas))
        nb = prim.mol.nao
    else:
        sm = gto.M(
            atom=[(s, np.asarray(r) + L) for L in Ls for (s, r) in prim.atoms],
            basis=other,
            unit="B",
            cart=True,
            verbose=0,
        )
        v = gto.intor_cross(intor, prim.mol, sm)
        nb = sm.nao // len(Ls)
    v = v.reshape(prim.mol.nao, len(Ls), nb)
    return np.einsum("aLb,Lr->rab", v, scell.one_hot(Ls))


# ------------------------------------------------------------------------------ builder
def build_supercell(
    scell,
    auxbasis,
    w=0.5,
    prec=1e-13,
    spherical=True,
    lindep=1e-10,
    minimal_basis="sto-3g",
    gcut=None,
    verbose=False,
    keep_fold=False,
):
    """All Gamma supercell quantities needed for SCF + (L)MP2.  Returns a dict with
    S, T, V, h, enn (Ewald, = N * E_nn(prim)), madelung (supercell v_M), B (naux_kept, nao, nao),
    J2 (naux, naux), J3 (nao, nao, naux) [G=0-dropped kernel, spherical aux], q (aux charges),
    Zk (3 complex nao x nao matrices <mu| e^{+i b_k.r} |nu>, lattice-summed, b_k = supercell
    reciprocal vectors: the Resta/Berghold position operator), Sx (cross overlap to the
    minimal basis, lattice-summed), aux_xyz (centre of each spherical aux function), info."""
    t0 = time.time()
    prim, sc = scell.prim, scell.sc
    mol = prim.mol
    nao_p, nb0, R = mol.nao, mol.nbas, scell.R
    auxmol = gto.M(atom=prim.atoms, basis=auxbasis, unit="B", cart=True, verbose=0)
    c2s = None
    if spherical:
        sph = auxmol.copy()
        sph.cart = False
        c2s = sph.cart2sph_coeff()
    naux_c = auxmol.nao
    q_c = aux_ft(auxmol, np.zeros((1, 3)))[:, 0].real
    amin_aux = min(auxmol.bas_exp(i).min() for i in range(auxmol.nbas))
    amin_orb = min(mol.bas_exp(i).min() for i in range(nb0))
    vol_sc = sc.vol
    c0 = np.pi / (w * w * vol_sc)

    rcut_pair = np.sqrt(2 * np.log(1 / prec) / amin_orb) + 2.0
    qmax = max(1.0, abs(q_c).max())
    th3 = 1.0 / (1.0 / amin_aux + 1.0 / (2 * amin_orb) + 1.0 / w**2)
    rcut_aux3 = _rcut_erfc(th3, prec / qmax) + 2.0
    th2 = 1.0 / (2.0 / amin_aux + 1.0 / w**2)
    rcut_aux2 = _rcut_erfc(th2, prec / qmax**2) + 2.0
    thn = 1.0 / (1.0 / (2 * amin_orb) + 1.0 / w**2)
    rcut_nuc = 0.5 * rcut_pair + _rcut_erfc(thn, prec / max(prim.Z)) + 2.0
    gcut = gcut or 2 * w * np.sqrt(np.log(1 / prec))

    # ---- 1e: S, T, SR nuclear attraction (Gaussian nuclei, erfc), minimal-basis cross overlap
    L1 = prim.translations(rcut_pair)
    M1 = scell.one_hot(L1)
    sm = prim.supermol(L1)
    S1 = np.einsum(
        "aLb,Lr->rab",
        sm.intor("int1e_ovlp_cart", shls_slice=(0, nb0, 0, sm.nbas)).reshape(
            nao_p, len(L1), nao_p
        ),
        M1,
    )
    T1 = np.einsum(
        "aLb,Lr->rab",
        sm.intor("int1e_kin_cart", shls_slice=(0, nb0, 0, sm.nbas)).reshape(
            nao_p, len(L1), nao_p
        ),
        M1,
    )
    LN = prim.translations(rcut_nuc)
    nuc_xyz = np.array([r + L for L in LN for r in prim.R])
    nuc_Z = np.tile(prim.Z, len(LN))
    fm = gto.fakemol_for_charges(nuc_xyz)
    fm.cart = True
    mm = sm + fm
    mm.cart = True
    with mm.with_range_coulomb(-w):
        v3 = mm.intor("int3c2e_cart", shls_slice=(0, nb0, 0, sm.nbas, sm.nbas, mm.nbas))
    Vsr1 = -np.einsum(
        "aLbA,A,Lr->rab", v3.reshape(nao_p, len(L1), nao_p, -1), nuc_Z, M1
    )
    Sx1 = (
        _fold_1e(scell, "int1e_ovlp_cart", other=minimal_basis, rcut=rcut_pair)
        if minimal_basis
        else None
    )
    t1 = time.time()

    # ---- SR aux metric and SR 3c, folded
    T2 = prim.translations(rcut_aux2)
    ai2 = _images(auxmol, T2)
    with ai2.with_range_coulomb(-w):
        j2 = ai2.intor("int2c2e_cart", shls_slice=(0, auxmol.nbas, 0, ai2.nbas))
    J2f = np.einsum(
        "PTQ,Tr->rPQ", j2.reshape(naux_c, len(T2), naux_c), scell.one_hot(T2)
    )
    T3 = prim.translations(rcut_aux3)
    M3 = scell.one_hot(T3)
    J3f = np.zeros((R, R, nao_p, nao_p, naux_c))
    chunk = max(1, int(2e7 // (nao_p * nao_p * len(L1) * naux_c)))
    for s0 in range(0, len(T3), chunk):
        ai = _images(auxmol, T3[s0 : s0 + chunk])
        big = gto.conc_mol(sm, ai)
        with big.with_range_coulomb(-w):
            v = big.intor(
                "int3c2e_cart", shls_slice=(0, nb0, 0, sm.nbas, sm.nbas, big.nbas)
            )
        nt = len(T3[s0 : s0 + chunk])
        v = v.reshape(nao_p, len(L1), nao_p, nt, naux_c)
        J3f += np.einsum(
            "aLbtP,Lr,ts->rsabP", v, M1, M3[s0 : s0 + chunk], optimize=True
        )
    t2 = time.time()

    # ---- LR on the supercell G lattice (G != 0), folded; plus the Resta matrices at b_k
    G = sc.gvectors(gcut)
    G = G[np.einsum("gi,gi->g", G, G) > 1e-12]
    G2 = np.einsum("gi,gi->g", G, G)
    vlr = 4 * np.pi / G2 * np.exp(-G2 / (4 * w * w)) / vol_sc
    Gk = sc.b.copy()  # supercell reciprocal vectors (rows)
    Q0 = pair_ft_residues(scell, np.zeros((1, 3)))[..., 0].real
    nrm = np.sqrt(np.diag(S1.sum(0)) / np.diag(Q0.sum(0)))
    s_err = abs(Q0 * np.outer(nrm, nrm)[None] - S1).max()
    Qk = (
        pair_ft_residues(scell, Gk)
        * nrm[None, :, None, None]
        * nrm[None, None, :, None]
    )
    X = aux_ft(auxmol, G)
    ph = np.exp(-1j * G @ scell.t.T).T  # (R, ng): exp(-iG.t_r)
    SGsc = np.exp(-1j * G @ np.array([r for _, r in scell.atoms]).T) @ sc.Z
    J2lr = np.stack([((X.conj() * vlr) @ (X * ph[r]).T).real for r in range(R)])
    J3lr = np.zeros((R * nao_p * nao_p, R * naux_c))
    Vlr1 = np.zeros((R, nao_p, nao_p))
    gchunk = max(1, int(4e6 // (R * nao_p * nao_p)))
    for g0 in range(0, len(G), gchunk):
        sl = slice(g0, g0 + gchunk)
        Qg = (
            pair_ft_residues(scell, G[sl])
            * nrm[None, :, None, None]
            * nrm[None, None, :, None]
        )
        Qv = (Qg.conj() * vlr[sl]).reshape(R * nao_p * nao_p, -1)
        Xr = (X[None, :, sl] * ph[:, None, sl]).reshape(R * naux_c, -1)
        J3lr += (Qv @ Xr.T).real
        Vlr1 -= (Qv @ SGsc[sl]).real.reshape(R, nao_p, nao_p)
    J3f += J3lr.reshape(R, nao_p, nao_p, R, naux_c).transpose(0, 3, 1, 2, 4)
    J2f = J2f + J2lr
    t3 = time.time()

    # ---- G=0 bookkeeping (supercell c0), spherical aux, unfold
    V1 = Vsr1 + Vlr1 + c0 * sc.Z.sum() * S1
    J2f = J2f - c0 * np.outer(q_c, q_c)[None]
    J3f = J3f - c0 * S1[:, None, :, :, None] * q_c[None, None, None, None, :]
    q = q_c
    if c2s is not None:
        J2f = np.einsum("pP,rpq,qQ->rPQ", c2s, J2f, c2s)
        J3f = J3f @ c2s
        q = c2s.T @ q_c
    naux_p = J2f.shape[1]
    S, T, V = scell.unfold2(S1), scell.unfold2(T1), scell.unfold2(V1)
    J2 = scell.unfold2(J2f)
    D = scell.D
    J3 = J3f[D[:, :, None], D[:, None, :]]  # (c, c', c'', m, n, P)
    J3 = J3.transpose(0, 3, 1, 4, 2, 5).reshape(R * nao_p, R * nao_p, R * naux_p)
    asym2, asym3 = abs(J2 - J2.T).max(), abs(J3 - J3.transpose(1, 0, 2)).max()
    J2 = 0.5 * (J2 + J2.T)
    J3 = 0.5 * (J3 + J3.transpose(1, 0, 2))
    s, U = np.linalg.eigh(J2)
    keep = s > lindep
    B = np.einsum("Pk,mnP->kmn", U[:, keep] / np.sqrt(s[keep]), J3, optimize=True)
    # Resta matrices on supercell AOs: Z^k_{(c,m),(c',n)} = conj(exp(-i b_k.t_c) Q^{D[c,c']}(b_k))
    ph_k = np.exp(-1j * scell.t @ Gk.T)  # (R, 3)
    Zk = [
        np.conj(ph_k[:, k][:, None, None, None] * Qk[D, :, :, k])
        .transpose(0, 2, 1, 3)
        .reshape(R * nao_p, R * nao_p)
        for k in range(3)
    ]
    Sx = scell.unfold2(Sx1) if Sx1 is not None else None
    aux_atom = np.array(
        [lbl[0] for lbl in (sph if spherical else auxmol).ao_labels(fmt=None)]
    )
    aux_xyz = np.concatenate([prim.R[aux_atom] + t for t in scell.t])
    enn = R * ewald_nn(prim, 1.0)
    info = dict(
        naux=J2.shape[0],
        naux_kept=int(keep.sum()),
        metric_min=s.min(),
        nG=len(G),
        n_pair_images=len(L1),
        n_aux_images_3c=len(T3),
        n_aux_images_2c=len(T2),
        n_nuc_images=len(LN),
        s_err=s_err,
        asym_J2=asym2,
        asym_J3=asym3,
        t_1e=t1 - t0,
        t_sr=t2 - t1,
        t_lr=t3 - t2,
        t_total=time.time() - t0,
    )
    if verbose:
        print(
            "  supercell:",
            {k: (f"{v:.3g}" if isinstance(v, float) else v) for k, v in info.items()},
            flush=True,
        )
    out = dict(
        S=S,
        T=T,
        V=V,
        h=T + V,
        enn=enn,
        madelung=madelung(sc),
        B=B,
        J2=J2,
        J3=J3,
        q=np.tile(q, R),
        Zk=Zk,
        Gk=Gk,
        Sx=Sx,
        aux_xyz=aux_xyz,
        info=info,
    )
    if keep_fold:
        out.update(S1=S1, V1=V1, J2f=J2f, J3f=J3f)
    return out


__all__ = ["Supercell", "build_supercell", "pair_ft_residues"]
