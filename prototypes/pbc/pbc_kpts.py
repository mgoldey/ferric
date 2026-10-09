"""k-point sampled periodic RHF (Monkhorst-Pack, Gamma-centred mesh), pure analytic-FT (AFT) route.

Stage 3 prototype.  Extends pbc_gamma (same Cell, same pair-FT kernel, same G=0 conventions) to
complex Bloch AOs

    chi_{m k}(r) = sum_L e^{i k.L} phi_m(r - L)                         (PySCF convention)

Per-cell quantities (Bohr / Hartree):
    S_mn(k)  = sum_L e^{ik.L} <phi_m | phi_n(. - L)>          (Hermitian, S(-k) = S(k)^*)
    pair FT  P^{k k'}_mn(K) = sum_L e^{ik'.L} FT[phi_m(r) phi_n(r - L)](K),  K = G + q,  q = k' - k
             (= FT of chi_mk^* chi_nk' over one cell; only K on the q-shifted G lattice survive)
    ERI      (m k1 n k2 | l k3 s k4) = (1/Omega) sum_{K != 0} v(K) P^{k1k2}_mn(K) conj(P^{k4k3}_sl(K)),
             v(K) = 4pi/|K|^2, K = G + (k2 - k1), momentum conservation k2-k1 = k3-k4 (mod G)
    dm_k = 2 C_occ(k) C_occ(k)^dagger,  E1 = (1/Nk) sum_k tr(h_k dm_k)
    J_mn(k)  = (1/Nk) sum_k' sum_ls Jker[k,k'][m,n,l,s] dm_k'[l,s],  Jker = (1/Omega) sum_{G!=0} v P^{kk}_mn conj(P^{k'k'}_ls)
    K_mn(k)  = (1/Nk) sum_k' sum_ls Kker[k,k'][m,l,n,s] dm_k'[l,s],  Kker = (1/Omega) sum_{K!=0} v P^{kk'}_ml conj(P^{kk'}_ns)
    exxdiv='ewald': K(k) += v_M S_k dm_k S_k with v_M = Madelung constant of the N1 x N2 x N3 SUPERCELL
             (PySCF tools.pbc.madelung(cell, kpts) scales the lattice by the mesh; df_jk._ewald_exxdiv_for_G0).

The only G=0 term dropped is K = 0, i.e. the q = 0 (k = k') head of exchange, exactly the term the
Gamma supercell drops; J and V_ne drop G=0 (neutral cell).  Consequence (tested): an N1xN2xN3 k-mesh
is term-by-term the Gamma point of the N1xN2xN3 supercell, because {G + q} over the mesh IS the
supercell reciprocal lattice.

Implementation: the lattice sum is grouped by the residue r of L modulo the mesh (pbc_supercell.
pair_ft_residues), because e^{ik'.L} depends only on r for mesh k'.  One residue-resolved FT per q
gives P^{k'-q, k'} for EVERY k' by a (Nk x R) phase product, so the build costs about Nk Gamma pair FTs.
Dense Nk^2 nao^4 kernels (Jker, Kker): toy/oracle scale only.
"""

from __future__ import annotations

import itertools
import time

import numpy as np

from pbc_gamma import Cell, _ewald, ewald_nn, pair_ft
from pbc_supercell import Supercell, pair_ft_residues

# Test-only mutation switch (monkeypatched by test_prototype.py).  None in production.
#   'phase_P'        : e^{-ik'.L} in the pair FT only (S, T keep +ik.L)
#   'kernel_no_q'    : v(G) instead of v(G+q) in exchange (the q-shift of the kernel dropped)
#   'madelung_prim'  : primitive-cell v_M instead of the supercell (k-mesh) v_M
_MUTANT = None


# ------------------------------------------------------------------------------ mesh
def mp_mesh(cell, n):
    """Gamma-centred mesh k = sum_i (m_i / n_i) b_i, m_i = 0..n_i-1 (PySCF cell.make_kpts(n) order)."""
    n = tuple(int(x) for x in n)
    ints = np.array(list(itertools.product(*(range(k) for k in n))))
    return ints, (ints / np.array(n)) @ cell.b


def mesh_index(n, m):
    m = np.mod(np.asarray(m, int), n)
    return int((m[0] * n[1] + m[1]) * n[2] + m[2])


def aft_gcut(cell, prec=1e-14):
    """pbc_gamma's pure-AFT cutoff: 2 sqrt(p_max ln(1/prec)), p_max = 2 max exponent."""
    pmax = max(cell.mol.bas_exp(i).max() for i in range(cell.mol.nbas)) * 2
    return 2 * np.sqrt(pmax * np.log(1 / prec))


def shifted_gvectors(cell, q, gcut):
    """All K = G + q with 0 < |K| <= gcut."""
    nmax = (
        np.ceil(
            (gcut + np.linalg.norm(q)) * np.linalg.norm(cell.a, axis=1) / (2 * np.pi)
        ).astype(int)
        + 1
    )
    n = np.array(list(itertools.product(*(range(-k, k + 1) for k in nmax))))
    K = n @ cell.b + q
    K2 = np.einsum("gi,gi->g", K, K)
    return K[(K2 <= gcut**2) & (K2 > 1e-12)]


def kmesh_madelung(cell, n):
    """v_M of the k-mesh == Gamma v_M of the diagonal supercell (PySCF tools.pbc.madelung(cell, kpts))."""
    sc = np.diag(np.asarray(n, float)) @ cell.a

    class _L:  # _ewald only reads .a, .vol, .gvectors
        pass

    lat = _L()
    lat.a, lat.vol, lat.b = sc, abs(np.linalg.det(sc)), 2 * np.pi * np.linalg.inv(sc).T
    lat.gvectors = lambda gcut: Cell.gvectors(lat, gcut)
    return -2.0 * _ewald(lat, [1.0], np.zeros((1, 3)), 1.0)


# ------------------------------------------------------------------------------ build
def build_k(
    cell,
    n,
    exxdiv="ewald",
    gcut=None,
    thresh=1e-14,
    rcut_1e=22.0,
    mem_bytes=8e8,
    verbose=True,
):
    """Dense k-point integrals for an n = (n1, n2, n3) Gamma-centred mesh (pure AFT, no SR split)."""
    if exxdiv is not None and str(exxdiv).lower() not in ("none", "ewald"):
        raise ValueError(f"exxdiv must be None, 'none' or 'ewald', got {exxdiv!r}")
    t0 = time.time()
    n = tuple(int(x) for x in n)
    ints, kpts = mp_mesh(cell, n)
    Nk, nao = len(kpts), cell.mol.nao
    gcut = gcut or aft_gcut(cell)

    # 1e: phase-weighted lattice sums of the molecular S, T (k.L, +i convention)
    L1 = cell.translations(rcut_1e)
    sm = cell.supermol(L1)
    nb0 = cell.mol.nbas
    S_L = sm.intor("int1e_ovlp_cart", shls_slice=(0, nb0, 0, sm.nbas)).reshape(
        nao, len(L1), nao
    )
    T_L = sm.intor("int1e_kin_cart", shls_slice=(0, nb0, 0, sm.nbas)).reshape(
        nao, len(L1), nao
    )
    ph1 = np.exp(1j * kpts @ L1.T)  # (Nk, nL)
    S = np.einsum("kL,mLn->kmn", ph1, S_L)
    T = np.einsum("kL,mLn->kmn", ph1, T_L)

    # raw Cartesian FT normalisation, calibrated at Gamma exactly as pbc_gamma does
    P0 = pair_ft(cell, np.zeros((1, 3)))[..., 0].real
    S0 = S_L.sum(1)
    nrm = np.sqrt(np.diag(S0) / np.diag(P0))
    nn = nrm[:, None] * nrm[None, :]

    scell = Supercell(cell.a, cell.atoms, cell.basis, n)
    assert np.array_equal(
        scell.cvec, ints
    )  # residue r <-> mesh index share one ordering
    sgn = -1.0 if _MUTANT == "phase_P" else 1.0
    ph_r = np.exp(
        sgn * 1j * kpts @ scell.t.T
    )  # (Nk, R): e^{ik'.L} = e^{ik'.t_r} for L == t_r mod mesh

    Jker = np.zeros((Nk, Nk, nao, nao, nao, nao), complex)
    Kker = np.zeros((Nk, Nk, nao, nao, nao, nao), complex)
    Vk = np.zeros((Nk, nao, nao), complex)
    chunk = max(500, int(mem_bytes / (16 * Nk * nao * nao * 2)))
    nK_tot = 0
    # time reversal: P^{-k,-k'}(-K) = conj P^{k,k'}(K) (real phi), so Kker[-k,-k'] = conj Kker[k,k'] and
    # only one of each (q, -q) pair of passes is computed
    neg = [mesh_index(n, -ints[k]) for k in range(Nk)]
    for iq in range(Nk):
        if neg[iq] < iq and _MUTANT != "no_time_reversal":
            continue
        q = kpts[iq]
        K = shifted_gvectors(cell, q, gcut)
        nK_tot += len(K)
        kof = [
            mesh_index(n, ints[j] - ints[iq]) for j in range(Nk)
        ]  # k = k' - q for k' = j
        for c0 in range(0, len(K), chunk):
            Kc = K[c0 : c0 + chunk]
            K2 = np.einsum("gi,gi->g", Kc, Kc)
            Q = pair_ft_residues(scell, Kc, thresh) * nn[None, :, :, None]
            P = np.einsum("jr,rmng->jmng", ph_r, Q)  # P[j] = P^{kof[j], j}(Kc)
            del Q
            if _MUTANT == "kernel_no_q":
                G = Kc - q
                G2 = np.einsum("gi,gi->g", G, G)
                v = np.where(G2 > 1e-12, 4 * np.pi / np.where(G2 > 1e-12, G2, 1.0), 0.0)
            else:
                v = 4 * np.pi / K2
            for j in range(Nk):
                blk = np.einsum("mlg,nsg->mlns", P[j] * v, P[j].conj()) / cell.vol
                Kker[kof[j], j] += blk
                if neg[iq] != iq and _MUTANT != "no_time_reversal":
                    Kker[neg[kof[j]], neg[j]] += blk.conj()
            if iq == 0:  # q = 0: kof[j] == j, P[j] = P^{jj}(G), G != 0
                Jker += np.einsum("kmng,jlsg->kjmnls", P * v, P.conj()) / cell.vol
                SG = np.exp(-1j * Kc @ cell.R.T) @ cell.Z
                Vk -= np.einsum("kmng,g->kmn", P, v * SG.conj()) / cell.vol
            del P
    enn = ewald_nn(cell, 1.0)
    if str(exxdiv).lower() == "ewald":
        vm = kmesh_madelung(cell, (1, 1, 1) if _MUTANT == "madelung_prim" else n)
    else:
        vm = 0.0
    if verbose:
        print(
            f"  build_k n={n}: Nk={Nk}, sum_q nK={nK_tot}, gcut={gcut:.2f}, v_M={vm:.10f}, "
            f"{time.time() - t0:.1f}s",
            flush=True,
        )
    return dict(
        S=S,
        T=T,
        V=Vk,
        h=T + Vk,
        Jker=Jker,
        Kker=Kker,
        enn=enn,
        madelung=vm,
        kpts=kpts,
        ints=ints,
        n=n,
        Nk=Nk,
        gcut=gcut,
    )


# ------------------------------------------------------------------------------ SCF
def jk_k(kb, dm):
    Nk = kb["Nk"]
    J = np.einsum("kjmnls,jls->kmn", kb["Jker"], dm) / Nk
    K = np.einsum("kjmlns,jls->kmn", kb["Kker"], dm) / Nk
    return J, K


def krhf(
    kb,
    nelec,
    conv=1e-10,
    maxiter=200,
    lindep=1e-8,
    return_mo=False,
    kshift=None,
    jk=None,
):
    """Complex RHF per k with global aufbau (PySCF KRHF get_occ) and DIIS over all k blocks.
    kshift defaults to kb['madelung'] (0 for exxdiv none).  Returns (E per cell, eps list, iterations).
    jk: optional dm -> (J, K) callable (per-k stacks) replacing the dense kernels (e.g. pbc_kgdf B tensors)."""
    S, h, Nk = kb["S"], kb["h"], kb["Nk"]
    vm = kb["madelung"] if kshift is None else kshift
    nocc = nelec // 2
    X = []
    for k in range(Nk):
        s, U = np.linalg.eigh(S[k])
        X.append(U[:, s > lindep] / np.sqrt(s[s > lindep]))
    dm = np.zeros_like(S)
    focks, errs = [], []
    e_old = 0.0
    for it in range(maxiter):
        J, K = jk_k(kb, dm) if jk is None else jk(dm)
        if vm:
            K = K + vm * np.einsum("kab,kbc,kcd->kad", S, dm, S)
        F = h + J - 0.5 * K
        e = 0.5 * np.einsum("kmn,knm->", h + F, dm).real / Nk + kb["enn"]
        err = [
            X[k].conj().T @ (F[k] @ dm[k] @ S[k] - S[k] @ dm[k] @ F[k]) @ X[k]
            for k in range(Nk)
        ]
        emax = max(abs(x).max() for x in err)
        if it > 0:
            focks.append(F)
            errs.append(err)
            focks, errs = focks[-8:], errs[-8:]
        if len(focks) > 1:
            m = len(focks)
            B = -np.ones((m + 1, m + 1))
            B[-1, -1] = 0
            for i in range(m):
                for j in range(m):
                    B[i, j] = sum(np.vdot(a, b).real for a, b in zip(errs[i], errs[j]))
            rhs = np.zeros(m + 1)
            rhs[-1] = -1
            c = np.linalg.lstsq(B, rhs, rcond=None)[0][:m]
            F = sum(ci * fi for ci, fi in zip(c, focks))
        eps, C = [], []
        for k in range(Nk):
            ek, ck = np.linalg.eigh(X[k].conj().T @ F[k] @ X[k])
            eps.append(ek)
            C.append(X[k] @ ck)
        # global aufbau over all k (insulators: nocc per k; checked by the caller via eps)
        allE = np.concatenate(
            [[(ek[i], k, i) for i in range(len(ek))] for k, ek in enumerate(eps)]
        )
        order = np.argsort(allE[:, 0], kind="stable")[: nocc * Nk]
        occ = [np.zeros(len(ek), bool) for ek in eps]
        for o in order:
            occ[int(allE[o, 1])][int(allE[o, 2])] = True
        dm = np.array(
            [2 * C[k][:, occ[k]] @ C[k][:, occ[k]].conj().T for k in range(Nk)]
        )
        if abs(e - e_old) < conv and emax < 1e-7:
            out = (e, eps, it)
            return out + (C, occ) if return_mo else out
        e_old = e
    raise RuntimeError("k-RHF not converged")


# ----------------------------------------------------- independent reference: explicit supercell
def supercell_cell(cell, n):
    t = np.array(list(itertools.product(*(range(k) for k in n)))) @ cell.a
    atoms = [(s, np.asarray(r, float) + tc) for tc in t for (s, r) in cell.atoms]
    return Cell(np.diag(np.asarray(n, float)) @ cell.a, atoms, cell.basis)


def gamma_aft(cell, gcut=None, thresh=1e-14, rcut_1e=22.0, mem_bytes=2.5e8):
    """pbc_gamma's pure-AFT Gamma build (build_integrals(cell, None)) with an explicit gcut and G
    chunking, so an explicit supercell fits in memory.  Uses pbc_gamma.pair_ft directly (no residue
    fold): an independent construction for the k-mesh == supercell anchor."""
    nao = cell.mol.nao
    gcut = gcut or aft_gcut(cell)
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
    P0 = pair_ft(cell, np.zeros((1, 3)))[
        ..., 0
    ].real  # calibration at the default thresh, as build_k
    nrm = np.sqrt(np.diag(S) / np.diag(P0))
    nn = nrm[:, None, None] * nrm[None, :, None]
    G = cell.gvectors(gcut)
    G = G[np.einsum("gi,gi->g", G, G) > 1e-12]
    I = np.zeros((nao,) * 4)
    V = np.zeros((nao, nao))
    chunk = max(500, int(mem_bytes / (16 * nao * nao * 2)))
    for c0 in range(0, len(G), chunk):
        Gc = G[c0 : c0 + chunk]
        v = 4 * np.pi / np.einsum("gi,gi->g", Gc, Gc)
        P = pair_ft(cell, Gc, thresh) * nn
        Pn = P.reshape(nao * nao, -1)
        I += ((Pn.conj() * v) @ Pn.T).real.reshape((nao,) * 4) / cell.vol
        SG = np.exp(-1j * Gc @ cell.R.T) @ cell.Z
        V -= ((P.conj() * (v * SG)).sum(-1)).real / cell.vol
    I = 0.5 * (I + I.transpose(2, 3, 0, 1))
    return dict(S=S, T=T, V=V, h=T + V, I=I, enn=ewald_nn(cell, 1.0), nG=len(G))


# ----------------------------------------------------- Marzari-Vanderbilt gauge-invariant spread
def omega_I(cell, kb, C, occ, thresh=1e-14):
    """Omega_I = (1/Nk) sum_{k,b} w_b (nocc - sum_mn |M_mn(k,b)|^2), b = +-b_i/n_i (orthorhombic
    finite-difference weights w_b = 1/(2|b|^2) per axis pair).  M_mn(k,b) = <u_mk|u_n,k+b> =
    C_k^dagger P^{k,k+b}(q = b, G = 0) C_{k+b}.  Needs every n_i >= 3 (n = 2 has b == -b)."""
    n, ints, kpts, Nk = kb["n"], kb["ints"], kb["kpts"], kb["Nk"]
    if min(n) < 3:
        raise ValueError("omega_I needs n_i >= 3 on every axis")
    if abs(cell.a - np.diag(np.diag(cell.a))).max() > 1e-12:
        raise NotImplementedError("orthorhombic cells only")
    S_L = None
    P0 = pair_ft(cell, np.zeros((1, 3)))[..., 0].real
    L1 = cell.translations(22.0)
    sm = cell.supermol(L1)
    nao, nb0 = cell.mol.nao, cell.mol.nbas
    S_L = sm.intor("int1e_ovlp_cart", shls_slice=(0, nb0, 0, sm.nbas)).reshape(
        nao, len(L1), nao
    )
    nrm = np.sqrt(np.diag(S_L.sum(1)) / np.diag(P0))
    nn = nrm[:, None] * nrm[None, :]
    scell = Supercell(cell.a, cell.atoms, cell.basis, n)
    ph_r = np.exp(1j * kpts @ scell.t.T)
    tot = 0.0
    for ax in range(3):
        for sgn in (1, -1):
            dm_int = np.zeros(3, int)
            dm_int[ax] = sgn
            b = sgn * cell.b[ax] / n[ax]
            Q = (
                pair_ft_residues(scell, b[None, :], thresh)[..., 0] * nn[None]
            )  # (R, nao, nao) at K = b
            wb = 1.0 / (2 * b @ b)
            for k in range(Nk):
                kp = mesh_index(n, ints[k] + dm_int)
                # P^{k,k'} with k' = k + b: phase e^{ik'.t_r}; K = G + (k' - k) = b needs k' - k == b exactly
                # (true when k'=k+b stays inside the mesh cell; otherwise k' - k = b - b_ax and K = b still
                # lies on that shifted lattice with G = b_ax: evaluate P at K = b regardless)
                Pkk = np.einsum("r,rmn->mn", ph_r[kp], Q)
                Co, Cp = C[k][:, occ[k]], C[kp][:, occ[kp]]
                M = Co.conj().T @ Pkk @ Cp / 2**0  # C normalised with S(k): <u|u> = 1
                tot += wb * (Co.shape[1] - np.sum(abs(M) ** 2))
    return tot / Nk
