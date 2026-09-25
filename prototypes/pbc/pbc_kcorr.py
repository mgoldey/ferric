"""k-point closed-shell MP2 and direct RPA (stage 9 prototype), from either the dense k-point AFT pair FT
(pbc_kpts conventions; the exact reference) or complex RS-GDF B tensors (pbc_kgdf).  numpy + PySCF molecular
intor only; PySCF pbc is the oracle (drivers/tests), never imported here.

Conventions (pbc_kpts: chi_mk = sum_L e^{ik.L} phi_m(r - L), per-cell normalised Bloch orbitals C(k)):
  ov pair density of class q = k' - k:   rho_ia^{k,k'}(K) = C_i(k)^H P^{kk'}(K) C_a(k'),  K = G + q, K != 0
  B route: B^P(k,k') plays sqrt(v(K)/Omega) P^{kk'}(K) (aux P of class q; pbc_kgdf), so
      (i ki a ka | j kj b kb) = sum_P Bov^P_ia(ki,ka) conj(Bvo^P_bj(kb,kj)),   ka - ki = kj - kb = q
      (momentum conservation ki + kj = ka + kb; both legs in the SAME q class, hence the same aux set).
  V[ki, kj, ka][i,a,j,b] = (i ki a ka | j kj b kb) with kb = ki + kj - ka  (PySCF kmp2 oovv layout, transposed).
  KMP2 per cell (PySCF kmp2.py: ERIs x 1/Nk, sum over ki,kj,ka, then / Nk):
      E = (1/Nk^3) sum_{ki,kj,ka} sum_ijab conj(V_iajb) [2 V_iajb - V_ibja] / (e_i + e_j - e_a - e_b)
  k-dRPA per cell: one polarizability per momentum transfer q, the SAME GL frequency quadrature as pbc_rpa:
      Pi_PQ(q, iw) = (4/Nk) sum_k sum_ia Bov^P_ia(k,k+q) conj(Bov^Q_ia(k,k+q)) e/(w^2 + e^2),  e = e_a(k+q) - e_i(k)
      E = (1/Nk) sum_q (1/2pi) int_0^inf dw [ln det(1 + Pi(q)) - tr Pi(q)]
  Independent per-q route (no quadrature): with the ov Gram Kq[(k,ia),(k',jb)] = sum_P Bov_ia conj(Bov_jb) / Nk,
      E_q = 1/2 [sum sqrt eig(D^2 + 4 D^1/2 Kq D^1/2) - tr D - 2 tr Kq]   (the integral done analytically).
  Denominators: at fixed C, exxdiv='ewald' moves every occupied level of every k by -v_M(mesh) (the supercell v_M,
  Iteration 9), so 'shifted' = eps_none_occ - v_M, 'unshifted' = eps_none (as pbc_mp2.denominators, per k).
  1/N_k^(1/1/1) normalisations make an N1xN2xN3 mesh term-by-term the Gamma point of the explicit supercell.

Optional q = 0 head (head=True, AFT route only): the K = 0 term of the q = 0 class, dropped by the mesh, restored as
its cubic spherical average (4pi/3 Omega) g_ia . conj(g_bj), g = d rho^{kk}(K)/dK at K = 0 (central difference).
Approximation: rho^{kk}(K) at fixed k (no k-derivative / Bloch-phase term of the true lim_{q->0} rho^{k,k+q}(q)),
exact only in the flat-band limit; cubic W = I/3 only.  Iteration 5b's "missing head", for k-points.

Test-only mutation switch _MUTANT: 'kb_wrong' (kb = ki - kj + ka in KMP2), 'no_conj' (no conj on the second leg),
'no_madelung' ('shifted' silently unshifted), 'rpa_k_wrong' (occupied energy of k' + q instead of k' - q in Pi).
"""

from __future__ import annotations

import numpy as np

from pbc_gamma import pair_ft
from pbc_kpts import aft_gcut, mesh_index, mp_mesh, shifted_gvectors
from pbc_rpa import _log1p_minus, gl_quadrature
from pbc_supercell import Supercell, pair_ft_residues

_MUTANT = None


# ------------------------------------------------------------------------------ denominators
def k_denominators(eps_none, nocc, vm, convention):
    """Per-k (eo, ev) lists from the exxdiv=None eigenvalues; 'shifted' subtracts the mesh v_M from occupied."""
    if convention not in ("shifted", "unshifted"):
        raise ValueError(
            f"convention must be 'shifted' or 'unshifted', got {convention!r}"
        )
    sh = vm if convention == "shifted" and _MUTANT != "no_madelung" else 0.0
    return [np.asarray(e[:nocc]) - sh for e in eps_none], [
        np.asarray(e[nocc:]) for e in eps_none
    ]


# ------------------------------------------------------------------------------ ov integrals
def _new_store(Nk, no, nv):
    return dict(
        V=np.zeros((Nk, Nk, Nk, no, nv, no, nv), complex),
        Kq=np.zeros((Nk, Nk * no * nv, Nk * no * nv), complex),
    )


def _accumulate(st, iq, kof, P, Co, Cv, Nk, Bq=None):
    """P[j] (nao, nao, naux) = the class-q pair tensor of (kof[j], j) (aux axis last, v^(1/2) included)."""
    Bov = np.einsum("jmi,jmng,jna->jiag", Co[kof].conj(), P, Cv, optimize=True)
    Bvo = np.einsum("jma,jmng,jni->jaig", Cv[kof].conj(), P, Co, optimize=True)
    Bvo2 = Bvo if _MUTANT == "no_conj" else Bvo.conj()
    T = np.einsum("xiag,ybjg->xyiajb", Bov, Bvo2, optimize=True)
    for x in range(Nk):
        for y in range(Nk):
            st["V"][kof[x], y, x] += T[x, y]  # ki = kof[x], ka = x, kj = y, kb = kof[y]
    nov = Bov.shape[1] * Bov.shape[2]
    st["Kq"][iq] += (
        np.einsum("xiag,yjbg->xiayjb", Bov, Bov.conj(), optimize=True).reshape(
            Nk * nov, Nk * nov
        )
        / Nk
    )
    if Bq is not None:
        Bq.append(
            Bov.transpose(3, 0, 1, 2).reshape(Bov.shape[3], Nk * nov) / np.sqrt(Nk)
        )


def _mo_stacks(C, nocc):
    return np.array([c[:, :nocc] for c in C]), np.array([c[:, nocc:] for c in C])


def _pair_norm(cell):
    L1 = cell.translations(22.0)
    sm = cell.supermol(L1)
    nao, nb0 = cell.mol.nao, cell.mol.nbas
    S0 = (
        sm.intor("int1e_ovlp_cart", shls_slice=(0, nb0, 0, sm.nbas))
        .reshape(nao, len(L1), nao)
        .sum(1)
    )
    P0 = pair_ft(cell, np.zeros((1, 3)))[..., 0].real
    nrm = np.sqrt(np.diag(S0) / np.diag(P0))
    return nrm[:, None] * nrm[None, :]


def aft_ov(
    cell, n, C, nocc, gcut=None, thresh=1e-14, mem_bytes=3e8, head=False, delta=1e-4
):
    """Exact (dense pure-AFT) V and per-q ov Grams Kq, streamed over K chunks of every q class (pbc_kpts.build_k's
    K sets, phases and normalisation).  C: per-k MO coefficients (columns sorted by energy, nocc occupied per k)."""
    n = tuple(int(x) for x in n)
    ints, kpts = mp_mesh(cell, n)
    Nk = len(kpts)
    gcut = gcut or aft_gcut(cell)
    nn = _pair_norm(cell)
    scell = Supercell(cell.a, cell.atoms, cell.basis, n)
    ph_r = np.exp(1j * kpts @ scell.t.T)
    Co, Cv = _mo_stacks(C, nocc)
    nao = cell.mol.nao
    st = _new_store(Nk, nocc, Cv.shape[2])
    chunk = max(200, int(mem_bytes / (16 * Nk * nao * nao * 3)))
    for iq in range(Nk):
        q = kpts[iq]
        kof = [mesh_index(n, ints[j] - ints[iq]) for j in range(Nk)]
        K = shifted_gvectors(cell, q, gcut)
        for c0 in range(0, len(K), chunk):
            Kc = K[c0 : c0 + chunk]
            v = 4 * np.pi / np.einsum("gi,gi->g", Kc, Kc) / cell.vol
            Q = pair_ft_residues(scell, Kc, thresh) * nn[None, :, :, None]
            P = np.einsum("jr,rmng->jmng", ph_r, Q) * np.sqrt(v)
            del Q
            _accumulate(st, iq, kof, P, Co, Cv, Nk)
    if head:
        kd = np.vstack([s * delta * np.eye(3) for s in (1, -1)])  # +x,+y,+z,-x,-y,-z
        Q = pair_ft_residues(scell, kd, thresh) * nn[None, :, :, None]
        P = np.einsum("jr,rmng->jmng", ph_r, Q)
        g = (
            (P[..., :3] - P[..., 3:])
            / (2 * delta)
            * np.sqrt(4 * np.pi / (3 * cell.vol))
        )
        _accumulate(st, 0, list(range(Nk)), g, Co, Cv, Nk)
    st.update(n=n, ints=ints, Nk=Nk)
    return st


def kB_ov(kg, C, nocc):
    """V, Kq and the aux-side per-q B stacks (naux_q, Nk*nocc*nvir) from pbc_kgdf B[(k,k')] (naux_q, nao, nao)."""
    B, n, ints, Nk = kg["B"], kg["n"], kg["ints"], kg["Nk"]
    Co, Cv = _mo_stacks(C, nocc)
    st = _new_store(Nk, nocc, Cv.shape[2])
    Bq = []
    for iq in range(Nk):
        kof = [mesh_index(n, ints[j] - ints[iq]) for j in range(Nk)]
        P = np.array([B[(kof[j], j)].transpose(1, 2, 0) for j in range(Nk)])
        _accumulate(st, iq, kof, P, Co, Cv, Nk, Bq=Bq)
    st.update(n=n, ints=ints, Nk=Nk, Bq=Bq)
    return st


# ------------------------------------------------------------------------------ KMP2
def kmp2(st, eo, ev):
    """(E, E_os, E_ss) per cell from V (see module doc)."""
    V, n, ints, Nk = st["V"], st["n"], st["ints"], st["Nk"]
    e_os = e_ss = 0.0
    for ki in range(Nk):
        for kj in range(Nk):
            for ka in range(Nk):
                if _MUTANT == "kb_wrong":
                    kb = mesh_index(n, ints[ki] - ints[kj] + ints[ka])
                else:
                    kb = mesh_index(n, ints[ki] + ints[kj] - ints[ka])
                d = (
                    eo[ki][:, None, None, None]
                    - ev[ka][None, :, None, None]
                    + eo[kj][None, None, :, None]
                    - ev[kb][None, None, None, :]
                )
                v = V[ki, kj, ka]
                x = V[ki, kj, kb].transpose(0, 3, 2, 1)
                t = v.conj() / d
                e_os += np.sum(t * v).real
                e_ss += np.sum(t * (v - x)).real
    s = 1.0 / Nk**3
    return (e_os + e_ss) * s, e_os * s, e_ss * s


def direct_kmp2(st, eo, ev):
    """2 sum |V|^2 / D per cell (the O(V^2) term of dRPA)."""
    V, n, ints, Nk = st["V"], st["n"], st["ints"], st["Nk"]
    tot = 0.0
    for ki in range(Nk):
        for kj in range(Nk):
            for ka in range(Nk):
                kb = mesh_index(n, ints[ki] + ints[kj] - ints[ka])
                d = (
                    eo[ki][:, None, None, None]
                    - ev[ka][None, :, None, None]
                    + eo[kj][None, None, :, None]
                    - ev[kb][None, None, None, :]
                )
                tot += 2 * np.sum(abs(V[ki, kj, ka]) ** 2 / d)
    return tot / Nk**3


# ------------------------------------------------------------------------------ k-dRPA
def q_excitations(st, eo, ev, iq):
    """e_ia for class q in the (k' = j, i, a) order of Kq / Bq: e_a(j) - e_i(j - q)."""
    n, ints, Nk = st["n"], st["ints"], st["Nk"]
    sgn = 1 if _MUTANT == "rpa_k_wrong" else -1
    kof = [mesh_index(n, ints[j] + sgn * ints[iq]) for j in range(Nk)]
    return np.concatenate(
        [(ev[j][None, :] - eo[kof[j]][:, None]).ravel() for j in range(Nk)]
    )


def _factor(Kq):
    s, U = np.linalg.eigh(Kq)
    keep = s > 1e-14 * max(1.0, s.max())
    return (U[:, keep] * np.sqrt(s[keep])).T  # Kq = Bf^T conj(Bf)


def kdrpa_plasmon(st, eo, ev):
    """Per-cell k-dRPA with the frequency integral done analytically per q (no quadrature, no aux)."""
    Nk = st["Nk"]
    tot = 0.0
    for iq in range(Nk):
        e = q_excitations(st, eo, ev, iq)
        if np.any(e <= 0):
            raise ValueError("non-positive e_ia")
        K = st["Kq"][iq]
        sd = np.sqrt(e)
        lam = np.linalg.eigvalsh(np.diag(e * e) + 4 * sd[:, None] * K * sd[None, :])
        if lam.min() <= 0:
            raise FloatingPointError("k-dRPA instability")
        tot += 0.5 * (np.sqrt(lam).sum() - e.sum() - 2 * np.trace(K).real)
    return tot / Nk


def _summand_c(Bf, e, w):
    Bs = Bf * np.sqrt(4 * e / (w * w + e * e))
    G = Bs @ Bs.conj().T if Bs.shape[0] <= Bs.shape[1] else Bs.T @ Bs.conj()
    lam = np.linalg.eigvalsh(0.5 * (G + G.conj().T))
    if lam.min() < -1e-12 * max(1.0, lam.max()):
        raise FloatingPointError(f"Pi(q) has a negative eigenvalue {lam.min():.2e}")
    return _log1p_minus(np.clip(lam, 0.0, None)).sum()


def kdrpa_quad(st, eo, ev, n=40, x0=0.5, aux=True):
    """Per-cell k-dRPA by the pbc_rpa GL quadrature, one Pi(q) per class.  aux=True uses the aux-side B stacks
    (B route; st['Bq']), else the factorised ov Gram Kq (AFT route).  Also returns the per-q energies."""
    Nk = st["Nk"]
    w, wt = gl_quadrature(n, x0)
    per_q = []
    for iq in range(Nk):
        e = q_excitations(st, eo, ev, iq)
        Bf = st["Bq"][iq] if aux else _factor(st["Kq"][iq])
        per_q.append(
            sum(wk * _summand_c(Bf, e, x) for x, wk in zip(w, wt)) / (2 * np.pi)
        )
    return sum(per_q) / Nk, per_q


def kdrpa_second_order(st, eo, ev, n=256, x0=0.5):
    """-(1/Nk) sum_q (1/2pi) int tr Pi(q)^2 / 2: the O(V^2) term of k-dRPA (should equal direct_kmp2)."""
    Nk = st["Nk"]
    w, wt = gl_quadrature(n, x0)
    tot = 0.0
    for iq in range(Nk):
        e = q_excitations(st, eo, ev, iq)
        Bf = _factor(st["Kq"][iq])
        for x, wk in zip(w, wt):
            Bs = Bf * np.sqrt(4 * e / (x * x + e * e))
            Pi = Bs @ Bs.conj().T
            tot += wk * (-0.5 * np.sum(abs(Pi) ** 2))
    return tot / (2 * np.pi) / Nk


__all__ = [
    "kdrpa_second_order",
    "k_denominators",
    "aft_ov",
    "kB_ov",
    "kmp2",
    "direct_kmp2",
    "kdrpa_plasmon",
    "kdrpa_quad",
    "q_excitations",
]
