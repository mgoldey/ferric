"""Issue #492 item 1: forces of the range-separated-hybrid (RSH) exact exchange on k-point RS-GDF.

The RSH exchange kernel is c_sr/r + (c_lr - c_sr) erf(w r)/r (crates/ferric-pbc/src/rsh.rs), so at a FIXED density
    E_x^RSH = c_sr E_K[full fit] + (c_lr - c_sr) E_K[attenuated fit] + E_M(v_M^mix),
    v_M^mix = c_sr v_M^full + (c_lr - c_sr) v_M^LR(w)          (both of the diag(n) supercell),
    E_M(v)  = -(v/2)(1/Nk) sum_s sum_k tr[D_s S D_s S].
E_K[full fit] is pbc_kgrad_gdf's RS-GDF exchange (Ewald split w0, SR real-space walks, -c0 G=0 terms).
E_K[attenuated fit] is klr.rs's KRsGdf::build_attenuated: the LR pass ALONE with w0 -> w, every K = G + q != 0 with
|K| <= 2 w sqrt(ln 1/prec), NO SR walks, NO c0:
    J2(q)_PQ      = sum_K v_w(K) conj(X_P(K)) X_Q(K),          v_w = 4 pi/K^2 e^{-K^2/4w^2} / Omega
    J3^{kk'}_P,ml = sum_K v_w(K) conj(X_P(K)) a^{k'}_ml(K)      (q = 0: Hermitised in (m,l))
    E_K = sum_q tr[f(J2(q)) H(q)],  H_PQ = -(1/2Nk^2) sum_k' sum_s tr[J3_P D_s(k') J3_Q^+ D_s(k)].
Its derivative is pbc_kgrad_gdf.grad's LR + metric passes with vlr -> v_w and nothing else (no SR bins, no G0 term).
`fit_xgrad` below is ONE code path for both fits: SR bins / G0 run iff the fit carries them.

_MUTANT-style switches (build_att(mutant=...), rsh_force(mutant=...)):
    'keep_c0'           attenuated fit keeps the plain fit's -c0(w) q q / -c0 q S(k) at q = 0 (+ its G0 force term)
    'omega0'            attenuated fit built with the Ewald split w0 instead of the RSH w
    'no_scale'          the (c_lr - c_sr) factor on the attenuated piece dropped (coefficient 1)
    'madelung_unmixed'  v_M^mix -> c_sr v_M^full (the LR Madelung piece left out)
    'madelung_full_lr'  v_M^mix -> c_lr v_M^full (the LR piece with the FULL constant, not v_M^LR(w))
    'madelung_scaled'   v_M^mix scaled AGAIN by the exact-exchange fraction c_sr (kgrad.rs m_term_madelung's
                        exch_scale applied on top of an already-mixed constant)
"""

from __future__ import annotations

import numpy as np
from pyscf.pbc import gto as pgto
from pyscf.pbc import tools as ptools

import pbc_grad as PGd
import pbc_kgrad as KG
from pbc_gamma import pair_ft
from pbc_gdf import aux_ft
from pbc_kgrad_gdf import _loewner
from pbc_kpts import kmesh_madelung, mesh_index, mp_mesh, shifted_gvectors
from pbc_supercell import Supercell, pair_ft_residues


# ------------------------------------------------------------------------------------------ attenuated fit (energy)
def build_att(
    cell,
    n,
    auxmol,
    omega,
    prec=1e-13,
    spherical=True,
    lindep=1e-10,
    pair_thresh=1e-14,
    gcut=None,
    mutant=None,
    w0=1.0,
):
    """klr.rs build_attenuated in the pbc_kgrad_gdf.build layout (every q explicit).  gcut overrides the Gaussian
    cut 2 w sqrt(ln 1/prec) (only the w -> infinity anchor uses it)."""
    w = w0 if mutant == "omega0" else omega
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
    gcut = 2 * w * np.sqrt(np.log(1 / prec)) if gcut is None else gcut
    scell = Supercell(cell.a, cell.atoms, cell.basis, n)
    ph = np.exp(1j * kpts @ scell.t.T)
    rcut_pair = (
        np.sqrt(
            2 * np.log(1 / prec) / min(mol.bas_exp(i).min() for i in range(mol.nbas))
        )
        + 2.0
    )
    L1 = cell.translations(rcut_pair)
    sm = cell.supermol(L1)
    S_L = sm.intor("int1e_ovlp_cart", shls_slice=(0, nb0, 0, sm.nbas)).reshape(
        nao, len(L1), nao
    )
    Sk = np.einsum("kr,rmn->kmn", ph, np.einsum("mLn,Lr->rmn", S_L, scell.one_hot(L1)))
    P0 = pair_ft(cell, np.zeros((1, 3)))[..., 0].real
    nrm = np.sqrt(np.diag(S_L.sum(1)) / np.diag(P0))
    nn = nrm[:, None] * nrm[None, :]
    q_c = aux_ft(auxmol, np.zeros((1, 3)))[:, 0].real
    c0 = np.pi / (w * w * cell.vol) if mutant == "keep_c0" else 0.0
    per_q = []
    B = {}
    for iq in range(Nk):
        q = kpts[iq]
        kof = [mesh_index(n, ints[j] - ints[iq]) for j in range(Nk)]
        K = shifted_gvectors(cell, q, gcut)
        K2 = np.einsum("gi,gi->g", K, K)
        vw = 4 * np.pi / K2 * np.exp(-K2 / (4 * w * w)) / cell.vol
        X = aux_ft(auxmol, K)
        J2 = (X.conj() * vw) @ X.T
        A = np.einsum(
            "jr,rmlg->jmlg",
            ph,
            pair_ft_residues(scell, K, pair_thresh) * nn[None, :, :, None],
        )
        J3 = np.einsum("Pg,jmlg->jPml", X.conj() * vw, A)
        del A
        if iq == 0:
            if c0:
                J2 = J2 - c0 * np.outer(q_c, q_c)
                J3 = J3 - c0 * np.einsum("P,jml->jPml", q_c, Sk)
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
    rg = dict(q_c=q_c, rcut_pair=rcut_pair, gcut=gcut)
    return dict(
        B=B,
        S=Sk,
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
        ph=ph,
        auxmol=auxmol,
        d2res=None,
        d3b=None,
        d3a=None,
        pair_thresh=pair_thresh,
        attenuated=True,
    )


# ------------------------------------------------------------------------------------------ energy / gradient
def _HZ(g, pq, Ds, with_j):
    """(H, Z) of one q class: exchange always, Coulomb (rho rho^H, c^* D) iff with_j."""
    Nk = g["Nk"]
    J3, s, U, keep, kof = pq["J3"], pq["s"], pq["U"], pq["keep"], pq["kof"]
    f, Lo = _loewner(s, keep)
    Jinv = (U * f) @ U.conj().T
    T = np.zeros((Nk,) + J3.shape[1:], complex)
    for j in range(Nk):
        T[j] = sum(
            np.einsum("la,Qba,bm->Qlm", Dsp[j], J3[j].conj(), Dsp[kof[j]]) for Dsp in Ds
        )
    H = -np.einsum("jPml,jQlm->PQ", J3, T) / (2 * Nk * Nk)
    Z = -np.einsum("PQ,jQlm->jPlm", Jinv.conj(), T) / (Nk * Nk)
    if with_j and np.allclose(pq["q"], 0):
        D = sum(Ds)
        rho = np.einsum("jPmn,jnm->P", J3, D) / Nk
        H = H + 0.5 * np.outer(rho, rho.conj())
        Z = Z + np.einsum("P,jlm->jPlm", (Jinv @ rho).conj(), D) / Nk
    return Jinv, Lo, H, Z


def fit_energy(g, Ds, with_j=False):
    """E_K (+ E_J) = sum_q tr[f(J2(q)) H(q)] of a fit (plain or attenuated)."""
    e = 0.0
    for pq in g["per_q"]:
        Jinv, _, H, _ = _HZ(g, pq, Ds, with_j)
        e += np.einsum("PQ,QP->", Jinv, H).real
    return e


def fit_xgrad(cell, g, Ds, with_j=False):
    """dE/dR (natm, 3) of fit_energy at FIXED D_s(k) (pbc_kgrad_gdf.grad's 2e part, generalised): the SR bins and
    the q = 0 G0 term run iff the fit carries them (plain fit), the K = G + q weight is v_w (attenuated) or vlr."""
    mol = cell.mol
    natm, nao = mol.natm, mol.nao
    aoat = PGd.ao_atom(mol)
    kpts, ph, scell = g["kpts"], g["ph"], g["scell"]
    auxmol, c2s, rg, w, c0 = g["auxmol"], g["c2s"], g["rg"], g["w"], g["c0"]
    sr = g["d3b"] is not None
    aux_owner = np.zeros(auxmol.nao, int)
    for A, (_, _, p0, p1) in enumerate(auxmol.aoslice_by_atom()):
        aux_owner[p0:p1] = A
    R = scell.R
    Zbin = np.zeros((R, R, nao, nao, auxmol.nao), complex) if sr else None
    g_aux = np.zeros((auxmol.nao, 3))
    g_orb = np.zeros((natm, 3))
    g_g0 = np.zeros((natm, 3))
    for iq, pq in enumerate(g["per_q"]):
        q, U = pq["q"], pq["U"]
        _, Lo, H, Z = _HZ(g, pq, Ds, with_j)
        Zc = np.einsum("Pa,jalm->jPlm", c2s, Z)
        if iq == 0:
            Zc = 0.5 * (Zc + Zc.conj().transpose(0, 1, 3, 2))
        Wc = c2s @ (U @ (Lo * (U.conj().T @ H @ U)) @ U.conj().T) @ c2s.T
        if sr:
            Zbin += np.einsum("ja,b,jPlm->abmlP", ph, np.exp(-1j * q @ scell.t.T), Zc)
            d2q = np.einsum("r,xrPQ->xPQ", np.exp(1j * q @ scell.t.T), g["d2res"])
            g_aux += (
                -np.einsum("xPQ,QP->Px", d2q, Wc).real
                + np.einsum("xPQ,QP->Qx", d2q, Wc).real
            )
        K = shifted_gvectors(cell, q, rg["gcut"])
        Zr = np.einsum("jr,jPlm->rPlm", ph, Zc)
        chunk = max(200, int(2e8 / (16 * 4 * R * nao * nao)))
        for k0 in range(0, len(K), chunk):
            Kc = K[k0 : k0 + chunk]
            K2 = np.einsum("gi,gi->g", Kc, Kc)
            vw = 4 * np.pi / K2 * np.exp(-K2 / (4 * w * w)) / cell.vol
            X = aux_ft(auxmol, Kc)
            Pr, Qb = KG.pair_ft_deriv_residues(scell, Kc, g["pair_thresh"])
            Pr *= g["nn"][None, :, :, None]
            Qb *= g["nn"][None, None, :, :, None]
            Qk = -1j * Kc.T[:, None, None, None, :] * Pr[None] - Qb
            Yt = np.einsum("Pg,rPlm->rlmg", X.conj() * vw, Zr)
            gb = np.einsum("xrmlg,rlmg->mx", Qb, Yt).real
            gk = np.einsum("xrmlg,rlmg->lx", Qk, Yt).real
            np.add.at(g_orb, aoat, gb + gk)
            aZ = np.einsum("rmlg,rPlm->Pg", Pr, Zr)
            g_aux += np.einsum("g,Pg,gx,Pg->Px", vw, X.conj(), 1j * Kc, aZ).real
            g_aux += np.einsum("g,Pg,gx,Pg->Px", vw, X.conj(), 1j * Kc, Wc.T @ X).real
            g_aux += np.einsum("g,Qg,gx,Qg->Qx", vw, Wc @ X.conj(), -1j * Kc, X).real
        if iq == 0 and c0:
            Mg0 = -c0 * np.einsum("P,jPlm->jlm", rg["q_c"], Zc)
            L1, XS = KG.image_ip(cell, "int1e_ipovlp_cart", rg["rcut_pair"])
            Mt = np.einsum("kL,knm->Lnm", np.exp(1j * kpts @ L1.T), Mg0)
            g_g0 += KG._fold_bra_ket(np.einsum("xmLn,Lnm->xmn", XS, Mt), aoat, natm)
        # (H Hermitian per class; keep the Im leak visible to the caller via the g dict)
        pq["Himag"] = abs(H - H.conj().T).max()
    if sr:
        gb = -np.einsum("xabmlP,abmlP->mx", g["d3b"], Zbin).real
        gk = np.einsum("xabmlP,abmlP->lx", g["d3b"] + g["d3a"], Zbin).real
        np.add.at(g_orb, aoat, gb + gk)
        g_aux += -np.einsum("xabmlP,abmlP->Px", g["d3a"], Zbin).real
    per_c = np.zeros((auxmol.natm, 3))
    np.add.at(per_c, aux_owner, g_aux)
    return dict(orb=g_orb, aux=per_c, g0=g_g0, total=g_orb + per_c + g_g0)


# ------------------------------------------------------------------------------------------ Madelung (explicit S)
def madelung_energy(cell, Sk, Ds, vM):
    Nk = len(Sk)
    return (
        -0.5
        * vM
        / Nk
        * sum(np.einsum("kab,kbc,kcd,kda->", D, Sk, D, Sk).real for D in Ds)
    )


def madelung_grad(cell, kpts, Ds, Sk, vM, rcut):
    """d/dR of madelung_energy at fixed D: -(v_M/Nk) sum_s tr[dS D_s S D_s] (pbc_kgrad item 1, M-term only)."""
    natm = cell.mol.natm
    aoat = PGd.ao_atom(cell.mol)
    Nk = len(kpts)
    Mk = -vM * sum(np.einsum("kab,kbc,kcd->kad", D, Sk, D) for D in Ds)
    L1, XS = KG.image_ip(cell, "int1e_ipovlp_cart", rcut)
    Mt = np.einsum("kL,knm->Lnm", np.exp(1j * kpts @ L1.T), Mk) / Nk
    return KG._fold_bra_ket(np.einsum("xmLn,Lnm->xmn", XS, Mt), aoat, natm)


def madelung_pair(cell, n, omega):
    """(v_M^full, v_M^LR(w)) of the diag(n) supercell; LR from PySCF tools.pbc.madelung(omega=w) (as rsh.rs)."""
    full = kmesh_madelung(cell, n)
    pc = pgto.Cell(
        a=cell.a, atom=[("H", (0, 0, 0))], basis="sto-3g", unit="B", spin=1, verbose=0
    )
    pc.build()
    lr = ptools.madelung(pc, pc.make_kpts(n), omega=omega)
    return full, lr


def v_mix(c_sr, c_lr, vfull, vlr, mutant=None):
    if mutant == "madelung_unmixed":
        return c_sr * vfull
    if mutant == "madelung_full_lr":
        return c_sr * vfull + (c_lr - c_sr) * vfull
    if mutant == "madelung_scaled":
        return c_sr * (c_sr * vfull + (c_lr - c_sr) * vlr)
    return c_sr * vfull + (c_lr - c_sr) * vlr


__all__ = [
    "build_att",
    "fit_energy",
    "fit_xgrad",
    "madelung_energy",
    "madelung_grad",
    "madelung_pair",
    "v_mix",
]
