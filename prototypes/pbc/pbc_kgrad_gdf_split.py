"""Iteration 26c: k-point RS-GDF ENERGY and FORCES with the RANGE SPLIT (Iteration 23 partition, grouping A).

Built on pbc_kgrad_gdf.build / grad (Iteration 21b: every q explicit, residue-binned SR integrals and derivatives,
complex Daleckii-Krein metric weight).  Per q (k = k' - q, K = G + q, K != 0):
    J3^{kk'}_P,ml = SR_kept sum_{L,T} e^{ik'.L} e^{-iq.T} [(m_0 l^c_L | P^c_T) + (m^c_0 l^s_L | P^c_T)]_erfc
                  + sum_K { v_LR(K) conj(X_P) a_ml + v_SR(K) conj(X^s_P) a_ml + v_SR(K) conj(X^c_P) a^ss_ml }
                  - [q = 0] c0 q^c_P (S(k) - S_ss(k))_ml                     (q = 0: Hermitised in (m,l))
    J2(q)_PQ      = SR_kept sum_T e^{iq.T} (P^c_0 | Q^c_T)_erfc
                  + sum_K { v_LR conj(X_P) X_Q + v_SR [conj(X^s_P) X_Q + conj(X^c_P) X^s_Q] } - [q = 0] c0 q^c_P q^c_Q
                  (= v conj(X) X - v_SR conj(X^c) X^c without the cancellation; bitwise Iteration 21b at lam = 0)
    v = 4pi/K^2 (/Omega), v_LR = v e^{-K^2/4w^2}, v_SR = v - v_LR; a^{k'}_ml(K) = sum_r e^{ik'.t_r} p_r,ml(K) the
    residue-binned Bloch pair FT (pbc_supercell.pair_ft_residues), a^ss the same on the smooth orbital pieces.
Why only q = 0 carries the G = 0 term: K = 0 exists only on the q = 0 lattice.  At q != 0 v_SR(|K|) is finite at every
K, including the K = q point nearest the origin (v_SR -> pi/w^2 as K -> 0); nothing is subtracted.
The SR real-space sum of a moved block at momentum q equals sum_{K in G+q} v_SR(K) FT..FT (Poisson summation with
the e^{iq.T} phase), which is why the moved blocks must use v_SR(|G+q|), not v_SR(|G|).

FORCES = pbc_kgrad_gdf.grad with: SR bins from the kept calls (bra -ip1, aux -ip2, ket ip1 + ip2 per call); G space
weight conj(v_LR X + v_SR X^s) on the full-pair residue derivatives and conj(v_SR X^c) on the smooth-pair residue
derivatives (pbc_kgrad.pair_ft_deriv_residues on the smooth-piece supercell -- NEW); aux -iK derivatives on X, X^s, X^c;
metric v_LR on (X, X) + v_SR on (X^s, X) + (X^c, X^s); q = 0 G0 term Mg0 = -c0 sum_P q^c_P Z_P on the image-resolved
d(S - S_ss) (smooth-piece overlap derivative per image -- NEW).

_MUTANT: 'gamma_kernel' (moved blocks' kernel v_SR(|K - q|), i.e. the Gamma kernel, with the FTs still at K),
         'no_dSss', 'g0_full' (as pbc_grad_gdf_split), 'no_ss_pair' (drop the smooth-pair derivative term).
build(..., mutant='gamma_kernel') applies the same mistake to the ENERGY (for the energy-level supercell check).
"""

from __future__ import annotations

import copy
import time

import numpy as np
from pyscf import gto

import pbc_grad as PGd
import pbc_kgrad as KG
from pbc_gamma import pair_ft
from pbc_gdf import _images, aux_ft
from pbc_gdf_split import Criterion, piece_mol
from pbc_grad_gdf_split import Pieces
from pbc_kgrad_gdf import _loewner, _ranges, make_auxmol  # noqa: F401  (re-exported for drivers)
from pbc_kpts import mesh_index, mp_mesh, shifted_gvectors
from pbc_supercell import Supercell, pair_ft_residues

_MUTANT = None


def _vsr(K, q, w, gamma_kernel=False):
    """v_SR per K (no 1/Omega).  gamma_kernel: evaluated at |K - q| (the mistake), with the K - q = 0 limit pi/w^2."""
    Kk = K - q if gamma_kernel else K
    K2 = np.einsum("gi,gi->g", Kk, Kk)
    out = np.full(len(K2), np.pi / (w * w))
    nz = K2 > 1e-12
    out[nz] = 4 * np.pi / K2[nz] * (1 - np.exp(-K2[nz] / (4 * w * w)))
    return out


def _smooth_scell(scell, pc):
    s = copy.copy(scell)
    s.prim = pc.cell_s
    return s


def build(
    cell,
    n,
    auxmol,
    w=1.0,
    lam=1.0,
    crit=None,
    prec=1e-13,
    spherical=True,
    lindep=1e-10,
    pair_thresh=1e-14,
    deriv=True,
    mutant=None,
):
    """Split analogue of pbc_kgrad_gdf.build (every q explicit).  lam = 0 moves nothing (== pbc_kgrad_gdf.build)."""
    t0 = time.time()
    crit = crit or Criterion(w, lam)
    mol = cell.mol
    nao, nb0 = mol.nao, mol.nbas
    n = tuple(int(x) for x in n)
    ints, kpts = mp_mesh(cell, n)
    Nk = len(kpts)
    c2s = np.eye(auxmol.nao)
    if spherical:
        sph = auxmol.copy()
        sph.cart = False
        c2s = sph.cart2sph_coeff()
    naux_c = auxmol.nao
    rg = _ranges(cell, auxmol, w, prec)
    pc = Pieces(cell, auxmol, crit)
    q_c = pc.q_c
    c0 = np.pi / (w * w * cell.vol)
    scell = Supercell(cell.a, cell.atoms, cell.basis, n)
    scell_s = _smooth_scell(scell, pc)
    ph = np.exp(1j * kpts @ scell.t.T)
    L1 = cell.translations(rg["rcut_pair"])
    sm = cell.supermol(L1)
    OL = scell.one_hot(L1)
    S_L = sm.intor("int1e_ovlp_cart", shls_slice=(0, nb0, 0, sm.nbas)).reshape(
        nao, len(L1), nao
    )
    sm_s = piece_mol(sm, pc.orb_s)
    S_Ls = sm_s.intor("int1e_ovlp_cart", shls_slice=(0, nb0, 0, sm_s.nbas)).reshape(
        nao, len(L1), nao
    )
    Sk = np.einsum("kr,rmn->kmn", ph, np.einsum("mLn,Lr->rmn", S_L, OL))
    Sk_ss = np.einsum("kr,rmn->kmn", ph, np.einsum("mLn,Lr->rmn", S_Ls, OL))
    # ---- SR 2c kept: (P^c | Q^c)
    T2 = cell.translations(rg["rcut_aux2"])
    ai2 = piece_mol(_images(auxmol, T2), pc.aux_c)
    O2 = scell.one_hot(T2)
    with ai2.with_range_coulomb(-w):
        J2T = ai2.intor(
            "int2c2e_cart", shls_slice=(0, auxmol.nbas, 0, ai2.nbas)
        ).reshape(naux_c, len(T2), naux_c)
        d2T = (
            ai2.intor(
                "int2c2e_ip1_cart", comp=3, shls_slice=(0, auxmol.nbas, 0, ai2.nbas)
            ).reshape(3, naux_c, len(T2), naux_c)
            if deriv
            else None
        )
    J2res = np.einsum("PTQ,Tr->rPQ", J2T, O2)
    d2res = np.einsum("xPTQ,Tr->xrPQ", d2T, O2) if deriv else None
    # ---- SR 3c kept calls, residue bins
    T3 = cell.translations(rg["rcut_aux3"])
    OT = scell.one_hot(T3)
    J3res = np.zeros((Nk, Nk, nao, nao, naux_c))
    d3b = np.zeros((3, Nk, Nk, nao, nao, naux_c)) if deriv else None
    d3a = np.zeros((3, Nk, Nk, nao, nao, naux_c)) if deriv else None
    chunk = max(1, int(2e7 // (nao * nao * len(L1) * naux_c)))
    for b0, ket in pc.calls(cell, sm):
        for c in range(0, len(T3), chunk):
            ai = piece_mol(_images(auxmol, T3[c : c + chunk]), pc.aux_c)
            big = gto.conc_mol(gto.conc_mol(b0, ket), ai)
            nt = len(T3[c : c + chunk])
            sl = (0, nb0, nb0, nb0 + ket.nbas, nb0 + ket.nbas, big.nbas)
            with big.with_range_coulomb(-w):
                v = big.intor("int3c2e_cart", shls_slice=sl).reshape(
                    nao, len(L1), nao, nt, naux_c
                )
                J3res += np.einsum(
                    "mLltP,La,tb->abmlP", v, OL, OT[c : c + chunk], optimize=True
                )
                if deriv:
                    for nm, acc in (
                        ("int3c2e_ip1_cart", d3b),
                        ("int3c2e_ip2_cart", d3a),
                    ):
                        v = big.intor(nm, comp=3, shls_slice=sl).reshape(
                            3, nao, len(L1), nao, nt, naux_c
                        )
                        acc += np.einsum(
                            "xmLltP,La,tb->xabmlP",
                            v,
                            OL,
                            OT[c : c + chunk],
                            optimize=True,
                        )
    P0 = pair_ft(cell, np.zeros((1, 3)))[..., 0].real
    nrm = np.sqrt(np.diag(S_L.sum(1)) / np.diag(P0))
    nn = nrm[:, None] * nrm[None, :]
    per_q = []
    B = {}
    for iq in range(Nk):
        q = kpts[iq]
        kof = [mesh_index(n, ints[j] - ints[iq]) for j in range(Nk)]
        J2 = np.einsum("r,rPQ->PQ", np.exp(1j * q @ scell.t.T), J2res)
        J3 = np.einsum("ja,b,abmlP->jPml", ph, np.exp(-1j * q @ scell.t.T), J3res)
        K = shifted_gvectors(cell, q, rg["gcut"])
        K2 = np.einsum("gi,gi->g", K, K)
        vlr = 4 * np.pi / K2 * np.exp(-K2 / (4 * w * w)) / cell.vol
        vsr = _vsr(K, q, w, mutant == "gamma_kernel") / cell.vol
        X = aux_ft(auxmol, K)
        Xs, Xc = pc.piece_ft(pc.aux_sm, K), pc.piece_ft(pc.aux_cm, K)
        # v_LR (X,X) + v_SR [(X^s,X) + (X^c,X^s)]: exactly 0 extra at lam = 0 (X^s == 0), so bitwise == Iteration 21b;
        # 'v on (X,X) minus v_SR on (X^c,X^c)' is the same sum but not bitwise at lam = 0 (2.8e-14 in J2, measured)
        J2 = (
            J2
            + (X.conj() * vlr) @ X.T
            + ((Xs.conj() * vsr) @ X.T + (Xc.conj() * vsr) @ Xs.T)
        )
        A = np.einsum(
            "jr,rmlg->jmlg",
            ph,
            pair_ft_residues(scell, K, pair_thresh) * nn[None, :, :, None],
        )
        Ass = np.einsum(
            "jr,rmlg->jmlg",
            ph,
            pair_ft_residues(scell_s, K, pair_thresh) * nn[None, :, :, None],
        )
        J3 = (
            J3
            + np.einsum("Pg,jmlg->jPml", X.conj() * vlr + Xs.conj() * vsr, A)
            + np.einsum("Pg,jmlg->jPml", Xc.conj() * vsr, Ass)
        )
        del A, Ass
        if iq == 0:
            J2 = J2 - c0 * np.outer(q_c, q_c)
            J3 = J3 - c0 * np.einsum("P,jml->jPml", q_c, Sk - Sk_ss)
            J3 = 0.5 * (J3 + J3.conj().transpose(0, 1, 3, 2))
        J2 = c2s.T @ J2 @ c2s
        J3 = np.einsum("jPml,Pa->jaml", J3, c2s)
        J2 = 0.5 * (J2 + J2.conj().T)
        s, U = np.linalg.eigh(J2)
        keep = s > lindep
        Wf = U[:, keep].conj() / np.sqrt(s[keep].astype(complex))
        for j in range(Nk):
            B[(kof[j], j)] = np.einsum("Pa,Pml->aml", Wf, J3[j])
        per_q.append(dict(q=q, kof=kof, J2=J2, J3=J3, s=s, U=U, keep=keep, nK=len(K)))
    return dict(
        B=B,
        S=Sk,
        S_ss=Sk_ss,
        kpts=kpts,
        ints=ints,
        n=n,
        Nk=Nk,
        per_q=per_q,
        c2s=c2s,
        rg=rg,
        w=w,
        c0=c0,
        nn=nn,
        scell=scell,
        scell_s=scell_s,
        ph=ph,
        auxmol=auxmol,
        pc=pc,
        crit=crit,
        J2res=J2res,
        J3res=J3res,
        d2res=d2res,
        d3b=d3b,
        d3a=d3a,
        pair_thresh=pair_thresh,
        wall=time.time() - t0,
    )


def _image_ip_piece(cell, rcut, keep):
    mol = cell.mol
    nao, nb0 = mol.nao, mol.nbas
    L1 = cell.translations(rcut)
    sm = piece_mol(cell.supermol(L1), keep)
    return L1, sm.intor(
        "int1e_ipovlp_cart", comp=3, shls_slice=(0, nb0, 0, sm.nbas)
    ).reshape(3, nao, len(L1), nao)


def grad(cell, kb, g, Da, Db, Fa, Fb, restricted=False):
    """dE/dR per cell for the split k RS-GDF energy (kb = pbc_kpts.build_k for the non-2e terms, g = build(...))."""
    mol = cell.mol
    natm, nao = mol.natm, mol.nao
    aoat = PGd.ao_atom(mol)
    Nk, kpts, ph, scell, scell_s = g["Nk"], g["kpts"], g["ph"], g["scell"], g["scell_s"]
    auxmol, c2s, rg, w, c0, pc = (
        g["auxmol"],
        g["c2s"],
        g["rg"],
        g["w"],
        g["c0"],
        g["pc"],
    )
    aux_owner = np.zeros(auxmol.nao, int)
    for A, (_, _, p0, p1) in enumerate(auxmol.aoslice_by_atom()):
        aux_owner[p0:p1] = A
    g1, parts = KG.kgrad(cell, kb, Da, Db, Fa, Fb, restricted=restricted, two_e=False)
    Ds = [0.5 * (Da + Db)] * 2 if restricted else [Da, Db]
    D = Da + Db
    R = scell.R
    Zbin = np.zeros((R, R, nao, nao, auxmol.nao), complex)
    g_aux = np.zeros((auxmol.nao, 3))
    g_orb = np.zeros((natm, 3))
    g_g0 = np.zeros((natm, 3))
    ss = 0.0 if _MUTANT == "no_ss_pair" else 1.0
    diag = []
    for iq, pq in enumerate(g["per_q"]):
        J3, s, U, keep, kof, q = (
            pq["J3"],
            pq["s"],
            pq["U"],
            pq["keep"],
            pq["kof"],
            pq["q"],
        )
        f, Lo = _loewner(s, keep)
        Jinv = (U * f) @ U.conj().T
        M = Jinv.conj()
        T = np.zeros((Nk,) + J3.shape[1:], complex)
        for j in range(Nk):
            T[j] = sum(
                np.einsum("la,Qba,bm->Qlm", Dsp[j], J3[j].conj(), Dsp[kof[j]])
                for Dsp in Ds
            )
        H = -np.einsum("jPml,jQlm->PQ", J3, T) / (2 * Nk * Nk)
        Z = -np.einsum("PQ,jQlm->jPlm", M, T) / (Nk * Nk)
        if iq == 0:
            rho = np.einsum("jPmn,jnm->P", J3, D) / Nk
            c = Jinv @ rho
            H = H + 0.5 * np.outer(rho, rho.conj())
            Z = Z + np.einsum("P,jlm->jPlm", c.conj(), D) / Nk
        diag.append(dict(iq=iq, naux=len(s), kept=int(keep.sum()), smin=s.min()))
        Zc = np.einsum("Pa,jalm->jPlm", c2s, Z)
        if iq == 0:
            Zc = 0.5 * (Zc + Zc.conj().transpose(0, 1, 3, 2))
        Wm = U @ (Lo * (U.conj().T @ H @ U)) @ U.conj().T
        Wc = c2s @ Wm @ c2s.T
        # ---- SR kept bins
        Zbin += np.einsum("ja,b,jPlm->abmlP", ph, np.exp(-1j * q @ scell.t.T), Zc)
        d2q = np.einsum("r,xrPQ->xPQ", np.exp(1j * q @ scell.t.T), g["d2res"])
        g_aux += (
            -np.einsum("xPQ,QP->Px", d2q, Wc).real
            + np.einsum("xPQ,QP->Qx", d2q, Wc).real
        )
        # ---- G space on K = G + q
        K = shifted_gvectors(cell, q, rg["gcut"])
        Zr = np.einsum("jr,jPlm->rPlm", ph, Zc)
        chunk = max(200, int(2e8 / (16 * 4 * R * nao * nao)))
        for c0k in range(0, len(K), chunk):
            Kc = K[c0k : c0k + chunk]
            K2 = np.einsum("gi,gi->g", Kc, Kc)
            vlr = 4 * np.pi / K2 * np.exp(-K2 / (4 * w * w)) / cell.vol
            vsr = _vsr(Kc, q, w, _MUTANT == "gamma_kernel") / cell.vol
            X = aux_ft(auxmol, Kc)
            Xs, Xc = pc.piece_ft(pc.aux_sm, Kc), pc.piece_ft(pc.aux_cm, Kc)
            wX = vlr * X + vsr * Xs  # full-pair weight (per aux column)
            wXc = vsr * Xc  # smooth-pair weight
            for scl, wgt, fac in ((scell, wX, 1.0), (scell_s, wXc, ss)):
                if fac == 0.0:
                    continue
                Pr, Qb = KG.pair_ft_deriv_residues(scl, Kc, g["pair_thresh"])
                Pr *= g["nn"][None, :, :, None]
                Qb *= g["nn"][None, None, :, :, None]
                Qk = -1j * Kc.T[:, None, None, None, :] * Pr[None] - Qb
                Yt = np.einsum("Pg,rPlm->rlmg", wgt.conj(), Zr)
                gb = np.einsum("xrmlg,rlmg->mx", Qb, Yt).real
                gk = np.einsum("xrmlg,rlmg->lx", Qk, Yt).real
                np.add.at(g_orb, aoat, gb + gk)
                aZ = np.einsum("rmlg,rPlm->Pg", Pr, Zr)
                g_aux += np.einsum("Pg,gx,Pg->Px", wgt.conj(), 1j * Kc, aZ).real
            # metric: v_LR (X,X) + v_SR [(X^s,X) + (X^c,X^s)]  (non-cancelling form of v_SR [(X,X) - (X^c,X^c)])
            for vv, A1, A2 in ((vlr, X, X), (vsr, Xs, X), (vsr, Xc, Xs)):
                g_aux += np.einsum(
                    "g,Pg,gx,Pg->Px", vv, A1.conj(), 1j * Kc, Wc.T @ A2
                ).real
                g_aux += np.einsum(
                    "g,Qg,gx,Qg->Qx", vv, Wc @ A1.conj(), -1j * Kc, A2
                ).real
        # ---- G0 at q = 0 with the kept inputs: -c0 q^c Z on d(S - S_ss), image-resolved
        if iq == 0:
            qg = pc.q if _MUTANT == "g0_full" else pc.q_c
            Mg0 = -c0 * np.einsum("P,jPlm->jlm", qg, Zc)
            L1, XS = KG.image_ip(cell, "int1e_ipovlp_cart", rg["rcut_pair"])
            if _MUTANT not in ("g0_full", "no_dSss"):
                _, XSs = _image_ip_piece(cell, rg["rcut_pair"], pc.orb_s)
                XS = XS - XSs
            Mt = np.einsum("kL,knm->Lnm", np.exp(1j * kpts @ L1.T), Mg0)
            g_g0 += KG._fold_bra_ket(np.einsum("xmLn,Lnm->xmn", XS, Mt), aoat, natm)
    gb = -np.einsum("xabmlP,abmlP->mx", g["d3b"], Zbin).real
    gk = np.einsum("xabmlP,abmlP->lx", g["d3b"] + g["d3a"], Zbin).real
    np.add.at(g_orb, aoat, gb + gk)
    g_aux += -np.einsum("xabmlP,abmlP->Px", g["d3a"], Zbin).real
    per_c = np.zeros((auxmol.natm, 3))
    np.add.at(per_c, aux_owner, g_aux)
    parts["gdf_orb"] = g_orb
    parts["gdf_aux"] = per_c
    parts["gdf_g0"] = g_g0
    return sum(parts.values()), parts, diag


def jk(g):
    from pbc_kgdf import jk_from_kB

    return jk_from_kB(g)


__all__ = ["build", "grad", "jk"]
