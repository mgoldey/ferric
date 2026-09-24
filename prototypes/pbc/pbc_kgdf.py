"""k-point range-separated Gaussian density fitting (RS-GDF per momentum transfer q), built from
MOLECULAR integral primitives (PySCF molecular intor as libint2 stand-in) + the analytic pair / aux FTs
of pbc_gamma / pbc_supercell / pbc_gdf.  PySCF pbc is NOT used here (oracle only).

Target (pbc_kpts conventions, chi_mk = sum_L e^{ik.L} phi_m(r - L)): for every pair (k, k'), q = k' - k,

    Kker[k,k'][m,l,n,s] = (1/Omega) sum_{K in G+q, K != 0} v(K) a_ml(K) conj(a_ns(K)),
    a_ml(K) = P^{kk'}_ml(K) = sum_L e^{ik'.L} FT[phi_m phi_l(.-L)](K)

and Jker[k,k'][m,n,l,s] from the q = 0 densities.  We fit a_ml in the q-Bloch aux functions
X^q_P(r) = sum_T e^{iq.T} chi_P(r - T), in the Coulomb metric of the SAME (K = 0-dropped) kernel:

    J2(q)[P,Q]      = <X_P|X_Q>  = (1/Omega) sum_K v conj(X_P(K)) X_Q(K)           (complex Hermitian)
                    = SR  sum_T e^{+iq.T} (P_0|Q_T)_erfc          + LR (reciprocal, K = G+q)
    J3(k,k')[P,ml]  = <X_P|a_ml> = (1/Omega) sum_K v conj(X_P(K)) a_ml(K)
                    = SR  sum_{L,T} e^{ik'.L} e^{-iq.T} (m_0 l_L | P_T)_erfc  + LR
    J2(q) = U s U^H, keep s > lindep (per q), B^k''_ml(k,k') = s^{-1/2} (U^H J3)       =>
    Kker[k,k'][m,l,n,s] ~= sum_P B^P_ml(k,k') conj(B^P_ns(k,k')),   Jker ~= sum_P B^P_mn(k,k) conj(B^P_ls(k',k')).

G = 0: the erfc real-space sum with phase e^{iq.T} equals (1/Omega) sum_{K in G+q} v_erfc(K)...; v_erfc is finite
at K = 0 (pi/w^2), and K = 0 exists only on the q = 0 lattice.  So ONLY q = 0 carries Iteration 2's bookkeeping:
J2 -= c0 q q^T, J3(k,k) -= c0 q S(k), c0 = pi/(w^2 Omega); nothing is subtracted at q != 0.

The SR integrals do not depend on q or k: they are computed ONCE (the Gamma triplet set) and folded into
residue bins (L mod mesh, T mod mesh); each (k,k') is then a (Nk x Nk) phase contraction of the bins.
Time reversal: J2(-q) = conj J2(q), J3(-k,-k') = conj J3(k,k'); choosing U(-q) = conj U(q) gives
B(-k,-k') = conj B(k,k'), so only half of the q classes are built.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
from pyscf import gto

from pbc_gamma import pair_ft
from pbc_gdf import _images, _rcut_erfc, aux_ft
from pbc_kpts import mesh_index, mp_mesh, shifted_gvectors
from pbc_supercell import Supercell, pair_ft_residues

# Test-only mutation switch (monkeypatched by tests).  None in production.
#   'q_phase_sign'      : e^{+iq.T} instead of e^{-iq.T} on the aux images in the SR 3c sum
#   'g0_all_q'          : Gamma G=0 bookkeeping applied at every q (with S(k') at q != 0)
#   'no_lindep_q'       : lindep cut applied only at q = 0 (plain s^{-1/2} on every eigenvalue at q != 0)
#   'no_time_reversal'  : build every q explicitly (reference for the time-reversal fill)
#   'no_herm_q0'        : skip the (m,n) Hermitisation of J3(k,k) at q = 0
_MUTANT = None


def build_kgdf(
    cell,
    n,
    auxbasis=None,
    w=1.0,
    prec=1e-13,
    spherical=True,
    lindep=1e-10,
    auxmol=None,
    pair_thresh=1e-14,
    verbose=False,
):
    """Complex RS-GDF tensors B[(k,k')] (naux_kept(q), nao, nao) for the Gamma-centred mesh n."""
    t0 = time.time()
    mol = cell.mol
    nao, nb0 = mol.nao, mol.nbas
    n = tuple(int(x) for x in n)
    ints, kpts = mp_mesh(cell, n)
    Nk = len(kpts)
    if auxmol is None:
        nel = sum(gto.charge(s) for s, _ in cell.atoms)
        auxmol = gto.M(
            atom=cell.atoms,
            basis=auxbasis,
            unit="B",
            cart=True,
            verbose=0,
            spin=nel % 2,
        )
    c2s = None
    if spherical:
        sph = auxmol.copy()
        sph.cart = False
        c2s = sph.cart2sph_coeff()
    naux_c = auxmol.nao
    q_c = aux_ft(auxmol, np.zeros((1, 3)))[:, 0].real
    amin_aux = min(auxmol.bas_exp(i).min() for i in range(auxmol.nbas))
    amin_orb = min(mol.bas_exp(i).min() for i in range(nb0))
    c0 = np.pi / (w * w * cell.vol)

    # ---- ranges: identical to pbc_gdf (phases have modulus 1, so the Gamma bounds hold for every q)
    rcut_pair = np.sqrt(2 * np.log(1 / prec) / amin_orb) + 2.0
    qmax = max(1.0, abs(q_c).max())
    th3 = 1.0 / (1.0 / amin_aux + 1.0 / (2 * amin_orb) + 1.0 / w**2)
    rcut_aux3 = _rcut_erfc(th3, prec / qmax) + 2.0
    th2 = 1.0 / (2.0 / amin_aux + 1.0 / w**2)
    rcut_aux2 = _rcut_erfc(th2, prec / qmax**2) + 2.0
    gcut = 2 * w * np.sqrt(np.log(1 / prec))

    scell = Supercell(cell.a, cell.atoms, cell.basis, n)
    assert np.array_equal(scell.cvec, ints)
    ph = np.exp(1j * kpts @ scell.t.T)  # (Nk, R): e^{ik.t_r}

    # ---- lattice overlap S(k) (for the q = 0 G = 0 term), residue-folded
    L1 = cell.translations(rcut_pair)
    sm = cell.supermol(L1)
    S_L = sm.intor("int1e_ovlp_cart", shls_slice=(0, nb0, 0, sm.nbas)).reshape(
        nao, len(L1), nao
    )
    OL = scell.one_hot(L1)  # (nL, R)
    S_res = np.einsum("mLn,Lr->rmn", S_L, OL)
    Sk = np.einsum("kr,rmn->kmn", ph, S_res)

    # ---- SR 2c: J2res[r, P, Q] = sum_{T == r} (P_0|Q_T)_erfc
    T2 = cell.translations(rcut_aux2)
    ai2 = _images(auxmol, T2)
    with ai2.with_range_coulomb(-w):
        J2T = ai2.intor(
            "int2c2e_cart", shls_slice=(0, auxmol.nbas, 0, ai2.nbas)
        ).reshape(naux_c, len(T2), naux_c)
    J2res = np.einsum("PTQ,Tr->rPQ", J2T, scell.one_hot(T2))
    del J2T

    # ---- SR 3c: J3res[rL, rT, m, l, P] = sum_{L == rL, T == rT} (m_0 l_L | P_T)_erfc  (the Gamma triplet set, once)
    T3 = cell.translations(rcut_aux3)
    OT = scell.one_hot(T3)
    J3res = np.zeros((Nk, Nk, nao, nao, naux_c))
    chunk = max(1, int(2e7 // (nao * nao * len(L1) * naux_c)))
    for c in range(0, len(T3), chunk):
        ai = _images(auxmol, T3[c : c + chunk])
        big = gto.conc_mol(sm, ai)
        with big.with_range_coulomb(-w):
            v = big.intor(
                "int3c2e_cart", shls_slice=(0, nb0, 0, sm.nbas, sm.nbas, big.nbas)
            )
        nt = len(T3[c : c + chunk])
        v = v.reshape(nao, len(L1), nao, nt, naux_c)
        J3res += np.einsum(
            "mLltP,La,tb->abmlP", v, OL, OT[c : c + chunk], optimize=True
        )
    del v

    # ---- pair-FT normalisation (Gamma calibration, as pbc_kpts.build_k)
    P0 = pair_ft(cell, np.zeros((1, 3)))[..., 0].real
    nrm = np.sqrt(np.diag(S_L.sum(1)) / np.diag(P0))
    nn = nrm[:, None] * nrm[None, :]

    neg = [mesh_index(n, -ints[k]) for k in range(Nk)]
    B, info_q = {}, {}
    sgnT = +1.0 if _MUTANT == "q_phase_sign" else -1.0
    for iq in range(Nk):
        if neg[iq] < iq and _MUTANT != "no_time_reversal":
            continue  # filled from +q below
        q = kpts[iq]
        kof = [
            mesh_index(n, ints[j] - ints[iq]) for j in range(Nk)
        ]  # k = k' - q for k' = j
        phq2 = np.exp(1j * q @ scell.t.T)  # e^{+iq.t_r} for J2
        phq3 = np.exp(sgnT * 1j * q @ scell.t.T)  # e^{-iq.t_r} for J3's aux images
        J2 = np.einsum("r,rPQ->PQ", phq2, J2res)
        # SR J3[j][P, m, l] for every k' = j
        J3 = np.einsum("ja,b,abmlP->jPml", ph, phq3, J3res)
        # LR on K = G + q, K != 0
        K = shifted_gvectors(cell, q, gcut)
        K2 = np.einsum("gi,gi->g", K, K)
        vlr = 4 * np.pi / K2 * np.exp(-K2 / (4 * w * w)) / cell.vol
        X = aux_ft(auxmol, K)  # (naux_c, nK)
        J2 = J2 + (X.conj() * vlr) @ X.T
        Q = pair_ft_residues(scell, K, pair_thresh) * nn[None, :, :, None]
        A = np.einsum("jr,rmlg->jmlg", ph, Q)  # a_ml^{k'=j}(K)
        del Q
        J3 = J3 + np.einsum("Pg,jmlg->jPml", X.conj() * vlr, A)
        del A
        if iq == 0 or _MUTANT == "g0_all_q":
            J2 = J2 - c0 * np.outer(q_c, q_c)
            J3 = J3 - c0 * np.einsum(
                "P,jml->jPml", q_c, Sk
            )  # q = 0: k = k' = j, a_ml(0) = S_ml(k')
        asym3 = 0.0
        if (
            iq == 0
        ):  # q = 0: B^P(k,k) must be Hermitian in (m,n) for a Hermitian J (real aux: exact identity);
            # truncated SR image sets break it at the prec level, so symmetrise as pbc_gdf does at Gamma
            asym3 = abs(J3 - J3.conj().transpose(0, 1, 3, 2)).max()
            if _MUTANT != "no_herm_q0":
                J3 = 0.5 * (J3 + J3.conj().transpose(0, 1, 3, 2))
        if c2s is not None:
            J2 = c2s.T @ J2 @ c2s
            J3 = np.einsum("jPml,Pa->jaml", J3, c2s)
        asym = abs(J2 - J2.conj().T).max()
        J2 = 0.5 * (J2 + J2.conj().T)
        s, U = np.linalg.eigh(J2)
        if _MUTANT == "no_lindep_q" and iq != 0:
            keep = np.ones(len(s), bool)
        else:
            keep = s > lindep
        W = U[:, keep].conj() / np.sqrt(s[keep].astype(complex))  # (naux, kept)
        for j in range(Nk):
            Bj = np.einsum("Pa,Pml->aml", W, J3[j])
            B[(kof[j], j)] = Bj
            if neg[iq] != iq and _MUTANT != "no_time_reversal":
                B[(neg[kof[j]], neg[j])] = Bj.conj()
        info_q[iq] = dict(
            naux=len(s),
            kept=int(keep.sum()),
            smin=s.min(),
            smax=s.max(),
            nK=len(K),
            asym_J2=asym,
            asym_J3_q0=asym3,
        )
        if neg[iq] != iq and _MUTANT != "no_time_reversal":
            info_q[neg[iq]] = dict(info_q[iq], mirrored=True)
    info = dict(
        Nk=Nk,
        naux=(J2res.shape[1] if c2s is None else c2s.shape[1]),
        n_pair_images=len(L1),
        n_aux_images_3c=len(T3),
        n_aux_images_2c=len(T2),
        gcut=gcut,
        n_3c_shell_triplets=nb0 * sm.nbas * auxmol.nbas * len(T3),
        per_q=info_q,
        n_q_built=sum(1 for v in info_q.values() if not v.get("mirrored")),
        B_elems=sum(b.size for b in B.values()),
        wall=time.time() - t0,
    )
    if verbose:
        print(
            f"  kgdf n={n}: naux={info['naux']} kept per q={[info_q[i]['kept'] for i in range(Nk)]} "
            f"triplets={info['n_3c_shell_triplets']} {info['wall']:.1f}s",
            flush=True,
        )
    return dict(B=B, S=Sk, kpts=kpts, ints=ints, n=n, Nk=Nk, info=info)


# ------------------------------------------------------------------------------ JK
def jk_from_kB(kg):
    """dm stack -> (J, K) stacks, one q class at a time (K) and from q = 0 (J)."""
    B, Nk, n, ints = kg["B"], kg["Nk"], kg["n"], kg["ints"]

    def jk(dm):
        J = np.zeros_like(dm)
        K = np.zeros_like(dm)
        # J: rho_P = (1/Nk) sum_k' sum_ls conj(B^P_ls(k',k')) dm_k'[l,s]; J(k) = sum_P B^P(k,k) rho_P
        rho = (
            sum(np.einsum("Pls,ls->P", B[(j, j)].conj(), dm[j]) for j in range(Nk)) / Nk
        )
        for k in range(Nk):
            J[k] = np.einsum("Pmn,P->mn", B[(k, k)], rho)
        # K(k) = (1/Nk) sum_k' sum_P B^P(k,k') dm_k' B^P(k,k')^H, q class by q class
        for iq in range(Nk):
            for j in range(Nk):
                k = mesh_index(n, ints[j] - ints[iq])
                Bq = B[(k, j)]
                K[k] += np.einsum("Pml,ls,Pns->mn", Bq, dm[j], Bq.conj()) / Nk
        return J, K

    return jk


def kernels_from_kB(kg):
    """Dense Jker/Kker (pbc_kpts layout) from B: for the exactness anchor only."""
    B, Nk = kg["B"], kg["Nk"]
    nao = B[(0, 0)].shape[1]
    Jk = np.zeros((Nk, Nk, nao, nao, nao, nao), complex)
    Kk = np.zeros_like(Jk)
    for k, j in itertools.product(range(Nk), range(Nk)):
        Kk[k, j] = np.einsum("Pml,Pns->mlns", B[(k, j)], B[(k, j)].conj())
        Jk[k, j] = np.einsum("Pmn,Pls->mnls", B[(k, k)], B[(j, j)].conj())
    return Jk, Kk


__all__ = ["build_kgdf", "jk_from_kB", "kernels_from_kB"]
