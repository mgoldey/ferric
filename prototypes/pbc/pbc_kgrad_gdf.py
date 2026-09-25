"""Iteration 21b: analytic nuclear forces for k-point RHF / UHF with k-point RS-GDF J/K (pbc_kgdf conventions).

Energy = pbc_kgrad's (pure-AFT h(k), S(k), E_nn, mesh v_M from pbc_kpts.build_k) with the two-electron part from
the per-q fit of pbc_kgdf.  Per q (every q built explicitly here, i.e. pbc_kgdf's 'no_time_reversal' mode, which
Iteration 11 anchored to the time-reversal fill at 5.8e-15) and per k' (k = k' - q):
    J3^{kk'}_P,ml = SR sum_{L,T} e^{ik'.L} e^{-iq.T} (m_0 l_L|P_T)_erfc + LR sum_K vlr(K) conj(X_P(K)) a^{k'}_ml(K)
                    - [q = 0] c0 q_P S_ml(k)                          (q = 0: Hermitised in (m,l), as pbc_kgdf)
    J2(q)_PQ      = SR sum_T e^{iq.T} (P_0|Q_T)_erfc + LR sum_K vlr conj(X_P) X_Q - [q = 0] c0 q_P q_Q  (Hermitised)
    (spherical aux: J3s = c2s^T J3, J2s = c2s^T J2 c2s);  f(J2) = U diag(f(s)) U^H, f = 1/s above lindep, 0 below.
    E_J = 1/2 rho^H f(J2(0)) rho,  rho_P = (1/Nk) sum_k tr[J3^{kk}_P D(k)]
    E_K = -(1/(2 Nk^2)) sum_s sum_q sum_k' sum_PQ conj(f)_PQ tr[J3_P D_s(k') J3_Q^+ D_s(k)]
    (== pbc_kgdf.jk_from_kB with B = s^{-1/2} U^H J3; checked numerically in run_kgrad_gdf.py).
So E_2e = sum_q tr[f(J2(q)) H(q)], H = 1/2 rho rho^H [q=0] + A(q), A_PQ = -(1/2Nk^2) sum tr[J3_P D' J3_Q^+ D] (Hermitian).

DERIVATIVE (orbital centres, aux centres and nuclei of atom A move in every image):
    dE_2e = Re sum_q sum_k' sum_P tr[dJ3^{k'-q,k'}_P Z^{k'}_P] + Re sum_q tr[Wm(q) dJ2(q)],
    Z^{k'}_P = -(1/Nk^2) sum_Q conj(f)_PQ sum_s D_s(k') J3_Q^+ D_s(k)  +  [q = 0] conj(c_P) D(k')/Nk,  c = f rho,
    (q = 0: Z -> (Z + Z^+)/2, the derivative of the (m,l) Hermitisation), Z back to cart aux by c2s,
    Wm = U (Lo o U^H H U) U^H (Daleckii-Krein; Lo = -f_i f_j kept-kept, (f_i - f_j)/(s_i - s_j) otherwise;
    Iteration 18's complex-Hermitian generalisation), back to cart by c2s Wm c2s^T.
  dJ3 SR: residue bins of int3c2e_ip1 / ip2 with the energy's phases e^{ik'.t_rL} e^{-iq.t_rT};
          bra d/dA_m = -ip1, aux d/dC_P = -ip2, ket d/dA_l = ip1 + ip2 (3-centre translation invariance).
  dJ3 LR: pbc_kgrad.pair_ft_deriv_residues (bra), ket = -iK p - bra, aux: d conj(X_P)/dC = +iK conj(X_P).
  dJ3 G0 (q = 0): -c0 q_P dS(k) -> Mg0(k) = -c0 sum_P q_P Z^k_P contracted with the phase-weighted image-resolved
          overlap derivative (range rcut_pair, as pbc_kgdf's S(k)).
  dJ2 SR: residue bins of int2c2e_ip1 with e^{iq.t_r}: d/dC_P = -ip1, d/dC_Q = +ip1.
  dJ2 LR: d/dC_P conj(X_P) = +iK conj(X_P);  d/dC_Q X_Q = -iK X_Q.
  Everything else (T, V_ne LR basis + nucleus, overlap/W + Madelung M-term, Ewald) = pbc_kgrad.kgrad(two_e=False).

_MUTANT: 'aux_phase' (e^{+iq.T} on the aux images of dJ3 SR), 'no_metric' (drop dJ2), 'no_g0' (drop -c0 q dS),
         'no_herm' (skip the q = 0 Hermitisation of Z).
"""

from __future__ import annotations

import time

import numpy as np
from pyscf import gto

import pbc_grad as PGd
import pbc_kgrad as KG
from pbc_gamma import pair_ft
from pbc_gdf import _images, _rcut_erfc, aux_ft
from pbc_kpts import mesh_index, mp_mesh, shifted_gvectors
from pbc_supercell import Supercell, pair_ft_residues

_MUTANT = None


def make_auxmol(cell, auxbasis):
    nel = sum(gto.charge(s) for s, _ in cell.atoms)
    return gto.M(atom=cell.atoms, basis=auxbasis, unit="B", cart=True, verbose=0, spin=nel % 2)


def _ranges(cell, auxmol, w, prec):
    mol = cell.mol
    q_c = aux_ft(auxmol, np.zeros((1, 3)))[:, 0].real
    amin_aux = min(auxmol.bas_exp(i).min() for i in range(auxmol.nbas))
    amin_orb = min(mol.bas_exp(i).min() for i in range(mol.nbas))
    qmax = max(1.0, abs(q_c).max())
    th3 = 1.0 / (1.0 / amin_aux + 1.0 / (2 * amin_orb) + 1.0 / w**2)
    th2 = 1.0 / (2.0 / amin_aux + 1.0 / w**2)
    return dict(q_c=q_c, rcut_pair=np.sqrt(2 * np.log(1 / prec) / amin_orb) + 2.0,
                rcut_aux3=_rcut_erfc(th3, prec / qmax) + 2.0, rcut_aux2=_rcut_erfc(th2, prec / qmax**2) + 2.0,
                gcut=2 * w * np.sqrt(np.log(1 / prec)))


def build(cell, n, auxmol, w=1.0, prec=1e-13, spherical=True, lindep=1e-10, pair_thresh=1e-14, deriv=True):
    """pbc_kgdf.build_kgdf (every q explicit) keeping J2, J3, U, s per q, plus (deriv) the residue-binned SR
    derivative integrals.  Returns dict usable by jk() and grad()."""
    t0 = time.time()
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
    q_c = rg["q_c"]
    c0 = np.pi / (w * w * cell.vol)
    scell = Supercell(cell.a, cell.atoms, cell.basis, n)
    ph = np.exp(1j * kpts @ scell.t.T)
    L1 = cell.translations(rg["rcut_pair"])
    sm = cell.supermol(L1)
    S_L = sm.intor("int1e_ovlp_cart", shls_slice=(0, nb0, 0, sm.nbas)).reshape(nao, len(L1), nao)
    OL = scell.one_hot(L1)
    Sk = np.einsum("kr,rmn->kmn", ph, np.einsum("mLn,Lr->rmn", S_L, OL))
    T2 = cell.translations(rg["rcut_aux2"])
    ai2 = _images(auxmol, T2)
    O2 = scell.one_hot(T2)
    with ai2.with_range_coulomb(-w):
        J2T = ai2.intor("int2c2e_cart", shls_slice=(0, auxmol.nbas, 0, ai2.nbas)).reshape(naux_c, len(T2), naux_c)
        d2T = (ai2.intor("int2c2e_ip1_cart", comp=3, shls_slice=(0, auxmol.nbas, 0, ai2.nbas))
               .reshape(3, naux_c, len(T2), naux_c) if deriv else None)
    J2res = np.einsum("PTQ,Tr->rPQ", J2T, O2)
    d2res = np.einsum("xPTQ,Tr->xrPQ", d2T, O2) if deriv else None
    T3 = cell.translations(rg["rcut_aux3"])
    OT = scell.one_hot(T3)
    J3res = np.zeros((Nk, Nk, nao, nao, naux_c))
    d3b = np.zeros((3, Nk, Nk, nao, nao, naux_c)) if deriv else None
    d3a = np.zeros((3, Nk, Nk, nao, nao, naux_c)) if deriv else None
    chunk = max(1, int(2e7 // (nao * nao * len(L1) * naux_c)))
    for c in range(0, len(T3), chunk):
        ai = _images(auxmol, T3[c : c + chunk])
        big = gto.conc_mol(sm, ai)
        nt = len(T3[c : c + chunk])
        sl = (0, nb0, 0, sm.nbas, sm.nbas, big.nbas)
        with big.with_range_coulomb(-w):
            v = big.intor("int3c2e_cart", shls_slice=sl).reshape(nao, len(L1), nao, nt, naux_c)
            J3res += np.einsum("mLltP,La,tb->abmlP", v, OL, OT[c : c + chunk], optimize=True)
            if deriv:
                for nm, acc in (("int3c2e_ip1_cart", d3b), ("int3c2e_ip2_cart", d3a)):
                    v = big.intor(nm, comp=3, shls_slice=sl).reshape(3, nao, len(L1), nao, nt, naux_c)
                    acc += np.einsum("xmLltP,La,tb->xabmlP", v, OL, OT[c : c + chunk], optimize=True)
    del v
    P0 = pair_ft(cell, np.zeros((1, 3)))[..., 0].real
    nrm = np.sqrt(np.diag(S_L.sum(1)) / np.diag(P0))
    nn = nrm[:, None] * nrm[None, :]
    per_q = []
    B = {}
    for iq in range(Nk):
        q = kpts[iq]
        kof = [mesh_index(n, ints[j] - ints[iq]) for j in range(Nk)]
        phq2 = np.exp(1j * q @ scell.t.T)
        phq3 = np.exp(-1j * q @ scell.t.T)
        J2 = np.einsum("r,rPQ->PQ", phq2, J2res)
        J3 = np.einsum("ja,b,abmlP->jPml", ph, phq3, J3res)
        K = shifted_gvectors(cell, q, rg["gcut"])
        K2 = np.einsum("gi,gi->g", K, K)
        vlr = 4 * np.pi / K2 * np.exp(-K2 / (4 * w * w)) / cell.vol
        X = aux_ft(auxmol, K)
        J2 = J2 + (X.conj() * vlr) @ X.T
        Q = pair_ft_residues(scell, K, pair_thresh) * nn[None, :, :, None]
        A = np.einsum("jr,rmlg->jmlg", ph, Q)
        del Q
        J3 = J3 + np.einsum("Pg,jmlg->jPml", X.conj() * vlr, A)
        del A
        if iq == 0:
            J2 = J2 - c0 * np.outer(q_c, q_c)
            J3 = J3 - c0 * np.einsum("P,jml->jPml", q_c, Sk)
            J3 = 0.5 * (J3 + J3.conj().transpose(0, 1, 3, 2))
        J2 = c2s.T @ J2 @ c2s
        J3 = np.einsum("jPml,Pa->jaml", J3, c2s)
        J2 = 0.5 * (J2 + J2.conj().T)
        s, U = np.linalg.eigh(J2)
        keep = s > lindep
        W = U[:, keep].conj() / np.sqrt(s[keep].astype(complex))
        for j in range(Nk):
            B[(kof[j], j)] = np.einsum("Pa,Pml->aml", W, J3[j])
        per_q.append(dict(q=q, kof=kof, J2=J2, J3=J3, s=s, U=U, keep=keep, nK=len(K)))
    out = dict(B=B, S=Sk, kpts=kpts, ints=ints, n=n, Nk=Nk, per_q=per_q, c2s=c2s, rg=rg, w=w, c0=c0, nn=nn,
               scell=scell, ph=ph, auxmol=auxmol, J2res=J2res, J3res=J3res, d2res=d2res, d3b=d3b, d3a=d3a,
               pair_thresh=pair_thresh, wall=time.time() - t0)
    return out


def _loewner(s, keep):
    f = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
    ds = s[:, None] - s[None, :]
    same = np.abs(ds) < 1e-300
    with np.errstate(divide="ignore", invalid="ignore"):
        Lo = np.where(same, 0.0, (f[:, None] - f[None, :]) / np.where(same, 1.0, ds))
    kk = keep[:, None] & keep[None, :]
    return f, np.where(kk, -f[:, None] * f[None, :], Lo)


def grad(cell, kb, g, Da, Db, Fa, Fb, restricted=False):
    """dE/dR per cell for the RS-GDF k energy at converged (D_s(k), F_s(k)); kb = pbc_kpts.build_k at the same
    geometry (h, S, v_M, gcut for the non-2e terms); g = build(...)."""
    mol = cell.mol
    natm, nao = mol.natm, mol.nao
    aoat = PGd.ao_atom(mol)
    Nk, n, ints, kpts, ph, scell = g["Nk"], g["n"], g["ints"], g["kpts"], g["ph"], g["scell"]
    auxmol, c2s, rg, w, c0 = g["auxmol"], g["c2s"], g["rg"], g["w"], g["c0"]
    aux_owner = np.zeros(auxmol.nao, int)
    for A, (_, _, p0, p1) in enumerate(auxmol.aoslice_by_atom()):
        aux_owner[p0:p1] = A
    g1, parts = KG.kgrad(cell, kb, Da, Db, Fa, Fb, restricted=restricted, two_e=False)
    Ds = [0.5 * (Da + Db)] * 2 if restricted else [Da, Db]
    D = Da + Db
    R = scell.R
    Zbin = np.zeros((R, R, nao, nao, auxmol.nao), complex)  # [rL, rT, m, l, P] holds Z_P,lm
    g_aux = np.zeros((auxmol.nao, 3))
    g_orb = np.zeros((natm, 3))
    g_g0 = np.zeros((natm, 3))
    diag = []
    for iq, pq in enumerate(g["per_q"]):
        J2, J3, s, U, keep, kof, q = pq["J2"], pq["J3"], pq["s"], pq["U"], pq["keep"], pq["kof"], pq["q"]
        f, Lo = _loewner(s, keep)
        Jinv = (U * f) @ U.conj().T
        M = Jinv.conj()
        # H(q) and Z (spherical aux)
        T = np.zeros((Nk,) + J3.shape[1:], complex)  # T[j][Q] = sum_s D_s(j) J3_Q^+ D_s(kof j)  ([l, m])
        for j in range(Nk):
            T[j] = sum(np.einsum("la,Qba,bm->Qlm", Dsp[j], J3[j].conj(), Dsp[kof[j]]) for Dsp in Ds)
        H = -np.einsum("jPml,jQlm->PQ", J3, T) / (2 * Nk * Nk)
        Z = -np.einsum("PQ,jQlm->jPlm", M, T) / (Nk * Nk)
        if iq == 0:
            rho = np.einsum("jPmn,jnm->P", J3, D) / Nk
            c = Jinv @ rho
            H = H + 0.5 * np.outer(rho, rho.conj())
            Z = Z + np.einsum("P,jlm->jPlm", c.conj(), D) / Nk
        diag.append(dict(iq=iq, naux=len(s), kept=int(keep.sum()), smin=s.min(), Himag=abs(H - H.conj().T).max()))
        Zc = np.einsum("Pa,jalm->jPlm", c2s, Z)  # cart aux
        if iq == 0 and _MUTANT != "no_herm":
            Zc = 0.5 * (Zc + Zc.conj().transpose(0, 1, 3, 2))
        Wm = U @ (Lo * (U.conj().T @ H @ U)) @ U.conj().T
        Wc = c2s @ Wm @ c2s.T
        if _MUTANT == "no_metric":
            Wc = 0 * Wc
        # ---- SR dJ3 bins: Zbin[rL, rT, m, l, P] += e^{ik'.t_rL} e^{-iq.t_rT} Z^{k'}_P,lm
        sgnT = +1.0 if _MUTANT == "aux_phase" else -1.0
        phq3 = np.exp(sgnT * 1j * q @ scell.t.T)
        Zbin += np.einsum("ja,b,jPlm->abmlP", ph, phq3, Zc)
        # ---- SR dJ2: sum_r e^{iq.t_r} d2res[x,r,P,Q] Wc_QP
        phq2 = np.exp(1j * q @ scell.t.T)
        d2q = np.einsum("r,xrPQ->xPQ", phq2, g["d2res"])
        g_aux += -np.einsum("xPQ,QP->Px", d2q, Wc).real + np.einsum("xPQ,QP->Qx", d2q, Wc).real
        # ---- LR on K = G + q
        K = shifted_gvectors(cell, q, rg["gcut"])
        chunk = max(200, int(2e8 / (16 * 4 * R * nao * nao)))
        for c0k in range(0, len(K), chunk):
            Kc = K[c0k : c0k + chunk]
            K2 = np.einsum("gi,gi->g", Kc, Kc)
            vlr = 4 * np.pi / K2 * np.exp(-K2 / (4 * w * w)) / cell.vol
            X = aux_ft(auxmol, Kc)
            Pr, Qb = KG.pair_ft_deriv_residues(scell, Kc, g["pair_thresh"])
            Pr *= g["nn"][None, :, :, None]
            Qb *= g["nn"][None, None, :, :, None]
            Qk = -1j * Kc.T[:, None, None, None, :] * Pr[None] - Qb
            # Yt[r, l, m, g] = vlr conj(X_P) sum_k' e^{ik'.t_r} Z^{k'}_P,lm   (orbital derivative weight)
            Zr = np.einsum("jr,jPlm->rPlm", ph, Zc)
            Yt = np.einsum("Pg,rPlm->rlmg", X.conj() * vlr, Zr)
            gb = np.einsum("xrmlg,rlmg->mx", Qb, Yt).real
            gk = np.einsum("xrmlg,rlmg->lx", Qk, Yt).real
            np.add.at(g_orb, aoat, gb + gk)
            # aux: sum_k' tr[a^{k'} Z^{k'}_P] with d conj(X_P)/dC = +iK conj(X_P)
            aZ = np.einsum("rmlg,rPlm->Pg", Pr, Zr)
            g_aux += np.einsum("g,Pg,gx,Pg->Px", vlr, X.conj(), 1j * Kc, aZ).real
            if _MUTANT != "no_metric":
                g_aux += np.einsum("g,Pg,gx,Pg->Px", vlr, X.conj(), 1j * Kc, Wc.T @ X).real
                g_aux += np.einsum("g,Qg,gx,Qg->Qx", vlr, Wc @ X.conj(), -1j * Kc, X).real
        # ---- G0 at q = 0: Mg0(k) = -c0 sum_P q_P Z^k_P ([l, m]) with the image-resolved overlap derivative
        if iq == 0 and _MUTANT != "no_g0":
            Mg0 = -c0 * np.einsum("P,jPlm->jlm", rg["q_c"], Zc)
            L1, XS = KG.image_ip(cell, "int1e_ipovlp_cart", rg["rcut_pair"])
            Mt = np.einsum("kL,knm->Lnm", np.exp(1j * kpts @ L1.T), Mg0)
            g_g0 += KG._fold_bra_ket(np.einsum("xmLn,Lnm->xmn", XS, Mt), aoat, natm)
    # ---- SR dJ3 contraction: bra -ip1 (m), ket ip1 + ip2 (l), aux -ip2 (P)
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
