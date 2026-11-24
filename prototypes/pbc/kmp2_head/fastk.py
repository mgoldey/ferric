"""Fast k-MP2 head accounting for s-only bases in an orthorhombic cell (H2/STO-3G), issue #230.

An independent re-implementation of the k-MP2 pipeline of pbc_kpts.py / pbc_kcorr.py that is cheap enough to
reach n = 7 Gamma-centred meshes. Conventions are the prototype's:
  P^{k,k'}_{mn}(K) = sum_L e^{ik'.L} FT[phi_m(r) phi_n(r-L)](K)   (depends on k' only)
For s Gaussians the pair Fourier transform factorises per Cartesian axis, which removes the Python loop over
lattice images. `run` builds the SCF and the (ov|ov) integrals; `account` splits the head-corrected energy into
  S    shifted-denominator k-MP2 with the q = 0, K = 0 term omitted,
  E1   S + linear (L) and quadratic (Qall) cubic-average head on the direct term,
  E3d  E1 + (Phi-tensor quadratic head - cubic quadratic head) + first-order Fock head (F).
E3d is the head-corrected energy whose N_k^-1 coefficient is complete (FINDINGS.md, issue #230 resolution);
test_head_residual.py pins that.  nproc <= 1 runs serially (no process pool).
"""

import itertools
import multiprocessing as _mp
import os
import sys
import time

_CTX = _mp.get_context("spawn")
Pool = _CTX.Pool

import numpy as np

PROTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROTO)
import pbc_kcorr as KC
import pbc_kpts as PK
from pbc_gamma import Cell, ewald_nn, shell_table
from pbc_kpts import mesh_index, mp_mesh, shifted_gvectors

H2_Z = [("H", (0.3, 0.2, 0.1)), ("H", (0.3, 0.2, 1.5))]
PHI4 = -0.17159902
PHI22 = (1 - 3 * PHI4) / 6
THRESH = 1e-14


def h2_atoms(orient):
    if orient == "111":
        c, u = np.array([0.3, 0.2, 0.8]), np.ones(3) / np.sqrt(3)
        return [("H", tuple(c - 0.7 * u)), ("H", tuple(c + 0.7 * u))]
    return H2_Z


def phi_tensor():
    T = np.zeros((3, 3, 3, 3))
    for a, b, c, d in itertools.product(range(3), repeat=4):
        idx = sorted((a, b, c, d))
        if idx[0] == idx[3]:
            T[a, b, c, d] = PHI4
        elif idx[0] == idx[1] and idx[2] == idx[3]:
            T[a, b, c, d] = PHI22
    return T


class Fast:
    """Per-(cell, mesh) constants for the separable pair FT."""

    def __init__(self, cell, n):
        a = cell.a
        assert np.allclose(a, np.diag(np.diag(a))), "orthorhombic only"
        self.cell, self.n = cell, tuple(int(x) for x in n)
        self.alat = np.diag(a).copy()
        self.sh = shell_table(cell.mol)
        assert all(s["l"] == 0 for s in self.sh), "s shells only"
        self.nao = cell.mol.nao
        self.nn = KC._pair_norm(cell)
        self.ints, self.kpts = mp_mesh(cell, self.n)
        self.Nk = len(self.kpts)
        # per-axis k' components for the mesh index along that axis
        self.kax = [
            2 * np.pi * np.arange(self.n[d]) / (self.n[d] * self.alat[d])
            for d in range(3)
        ]

    def P(self, K):
        """P[j, m, n, g] = P^{., k_j}(K_g), normalised (nn) like the prototype's Q * nn.
        Per shell pair: P = sum_p [base_p Wx_p Wy_p](g; jx jy) x Wz_p(g; jz), one batched matmul over g."""
        K = np.asarray(K, float)
        ng = len(K)
        K2 = np.einsum("gi,gi->g", K, K)
        out = np.empty((self.Nk, self.nao, self.nao, ng), complex)
        nx, ny, nz = self.n
        for sa in self.sh:
            for sb in self.sh:
                AB0 = sa["A"] - sb["A"]
                prs = [
                    (ea, cA, eb, cB)
                    for ea, cA in zip(sa["exps"], sa["coef"])
                    for eb, cB in zip(sb["exps"], sb["coef"])
                ]
                XY = np.empty((ng, nx * ny, len(prs)), complex)
                Zm = np.empty((ng, len(prs), nz), complex)
                for ip, (ea, cA, eb, cB) in enumerate(prs):
                    p = ea + eb
                    mu = ea * eb / p
                    Pc0 = (ea * sa["A"] + eb * sb["A"]) / p
                    base = (
                        cA
                        * cB
                        * (np.pi / p) ** 1.5
                        * np.exp(-K2 / (4 * p) - 1j * (K @ Pc0))
                    )
                    W = []
                    for d in range(3):
                        al = self.alat[d]
                        rng = np.sqrt(np.log(1e22) / mu)
                        lmin = int(np.floor((AB0[d] - rng) / al))
                        lmax = int(np.ceil((AB0[d] + rng) / al))
                        ls = np.arange(lmin, lmax + 1) * al
                        e = np.exp(-mu * (AB0[d] - ls) ** 2)  # (nl,)
                        ek = e[None, :] * np.exp(
                            1j * self.kax[d][:, None] * ls[None, :]
                        )  # (n_d, nl)
                        eK = np.exp(
                            -1j * (eb / p) * ls[:, None] * K[None, :, d]
                        )  # (nl, ng)
                        W.append(ek @ eK)  # (n_d, ng)
                    xy = (W[0] * base[None, :])[:, None, :] * W[1][
                        None, :, :
                    ]  # (nx, ny, ng)
                    XY[:, :, ip] = xy.reshape(nx * ny, ng).T
                    Zm[:, ip, :] = W[2].T
                acc = np.matmul(
                    XY, Zm
                )  # (ng, nx*ny, nz); mesh index j = (jx*ny + jy)*nz + jz
                out[:, sa["off"], sb["off"], :] = (
                    acc.reshape(ng, self.Nk).T * self.nn[sa["off"], sb["off"]]
                )
        return out


# ------------------------------------------------------------------------------ SCF kernels
_G = {}


def _init(atoms, a, n, gcut):
    cell = Cell(np.diag(a), atoms, "sto-3g")
    _G["fp"] = Fast(cell, n)
    _G["cell"] = cell
    _G["gcut"] = gcut


def _kker_class(iq):
    fp, cell, gcut = _G["fp"], _G["cell"], _G["gcut"]
    Nk = fp.Nk
    q = fp.kpts[iq]
    K = shifted_gvectors(cell, q, gcut)
    nao = fp.nao
    blk = np.zeros((Nk, nao, nao, nao, nao), complex)
    extra = None
    chunk = max(500, int(2.0e8 / (16 * Nk * nao * nao)))
    if iq == 0:
        Pall = np.empty((Nk, nao, nao, len(K)), complex)
        vall = np.empty(len(K))
        Vk = np.zeros((Nk, nao, nao), complex)
    for c0 in range(0, len(K), chunk):
        Kc = K[c0 : c0 + chunk]
        v = 4 * np.pi / np.einsum("gi,gi->g", Kc, Kc)
        P = fp.P(Kc)
        blk += np.einsum("jmlg,jnsg->jmlns", P * v, P.conj(), optimize=True) / cell.vol
        if iq == 0:
            SG = np.exp(-1j * Kc @ cell.R.T) @ cell.Z
            Vk -= np.einsum("kmng,g->kmn", P, v * SG.conj()) / cell.vol
            Pall[..., c0 : c0 + len(Kc)] = P
            vall[c0 : c0 + len(Kc)] = v
    if iq == 0:
        extra = (Vk, Pall, vall)
    return iq, blk, extra


def build_k(cell, atoms, n, gcut, nproc=6, verbose=True):
    t0 = time.time()
    n = tuple(int(x) for x in n)
    ints, kpts = mp_mesh(cell, n)
    Nk, nao = len(kpts), cell.mol.nao
    L1 = cell.translations(22.0)
    sm = cell.supermol(L1)
    nb0 = cell.mol.nbas
    S_L = sm.intor("int1e_ovlp_cart", shls_slice=(0, nb0, 0, sm.nbas)).reshape(
        nao, len(L1), nao
    )
    T_L = sm.intor("int1e_kin_cart", shls_slice=(0, nb0, 0, sm.nbas)).reshape(
        nao, len(L1), nao
    )
    ph1 = np.exp(1j * kpts @ L1.T)
    S = np.einsum("kL,mLn->kmn", ph1, S_L)
    T = np.einsum("kL,mLn->kmn", ph1, T_L)
    neg = [mesh_index(n, -ints[k]) for k in range(Nk)]
    todo = [iq for iq in range(Nk) if not neg[iq] < iq]
    Kker = np.zeros((Nk, Nk, nao, nao, nao, nao), complex)

    def put(iq, blk):
        kof = np.array([mesh_index(n, ints[j] - ints[iq]) for j in range(Nk)])
        jj = np.arange(Nk)
        Kker[kof, jj] += blk
        if neg[iq] != iq:
            negk = np.array(neg)
            Kker[negk[kof], negk[jj]] += blk.conj()

    _init(atoms, np.diag(cell.a), n, gcut)
    _, blk0, (Vk, P00, v00) = _kker_class(0)
    put(0, blk0)
    rest = [iq for iq in todo if iq != 0]
    if rest and nproc <= 1:
        for iq in rest:
            put(iq, _kker_class(iq)[1])
    elif rest:
        with Pool(
            nproc, initializer=_init, initargs=(atoms, np.diag(cell.a), n, gcut)
        ) as pool:
            for iq, blk, _ in pool.imap_unordered(_kker_class, rest, chunksize=1):
                put(iq, blk)
    vm = PK.kmesh_madelung(cell, n)

    def jk(dm):
        rho = np.einsum("jlsg,jls->g", P00.conj(), dm) / Nk
        J = np.einsum("kmng,g->kmn", P00, v00 * rho) / cell.vol
        K = np.einsum("kjmlns,jls->kmn", Kker, dm) / Nk
        return J, K

    if verbose:
        print(
            f"  build_k n={n}: Nk={Nk} v_M={vm:.10f} {time.time() - t0:.1f}s",
            flush=True,
        )
    return {
        "S": S,
        "T": T,
        "V": Vk,
        "h": T + Vk,
        "Kker": Kker,
        "enn": ewald_nn(cell, 1.0),
        "madelung": vm,
        "kpts": kpts,
        "ints": ints,
        "n": n,
        "Nk": Nk,
        "gcut": gcut,
        "jk": jk,
        "P00": P00,
        "v00": v00,
    }


# ------------------------------------------------------------------------------ MP2 integrals (1 occ, 1 vir)
def _ov_class(iq):
    fp, cell, gcut = _G["fp"], _G["cell"], _G["gcut"]
    Co, Cv = _G["Co"], _G["Cv"]
    n, ints, Nk = fp.n, fp.ints, fp.Nk
    q = fp.kpts[iq]
    kof = np.array([mesh_index(n, ints[j] - ints[iq]) for j in range(Nk)])
    K = shifted_gvectors(cell, q, gcut)
    T = np.zeros((Nk, Nk), complex)
    chunk = max(500, int(2.0e8 / (16 * Nk * 4)))
    for c0 in range(0, len(K), chunk):
        Kc = K[c0 : c0 + chunk]
        v = 4 * np.pi / np.einsum("gi,gi->g", Kc, Kc) / cell.vol
        P = fp.P(Kc)
        Bov = np.einsum("jm,jmng,jn->jg", Co[kof].conj(), P, Cv, optimize=True)
        Bvo = np.einsum("jm,jmng,jn->jg", Cv[kof].conj(), P, Co, optimize=True)
        T += (Bov * v) @ Bvo.conj().T
    return iq, kof, T


def _init_ov(atoms, a, n, gcut, Co, Cv):
    _init(atoms, a, n, gcut)
    _G["Co"], _G["Cv"] = Co, Cv


def aft_ov(cell, atoms, n, C, gcut, nproc=6):
    """V[ki, kj, ka] = (i ki a ka | j kj b kb), 1 occupied + 1 virtual band."""
    n = tuple(int(x) for x in n)
    _, kpts = mp_mesh(cell, n)
    Nk = len(kpts)
    C = np.asarray(C)
    Co, Cv = C[:, :, 0], C[:, :, 1]
    V = np.zeros((Nk, Nk, Nk), complex)

    def put(iq, kof, T):
        # ki = kof[x], ka = x, kj = y
        V[kof, :, np.arange(Nk)] = T

    if nproc <= 1:
        _init_ov(atoms, np.diag(cell.a), n, gcut, Co, Cv)
        for iq in range(Nk):
            put(*_ov_class(iq))
        return V
    with Pool(
        nproc, initializer=_init_ov, initargs=(atoms, np.diag(cell.a), n, gcut, Co, Cv)
    ) as pool:
        for res in pool.imap_unordered(_ov_class, range(Nk), chunksize=1):
            put(*res)
    return V


def kb_table(n, ints):
    n = np.array(n)
    Nk = len(ints)
    mult = np.array([n[1] * n[2], n[2], 1])
    tab = np.empty((Nk, Nk, Nk), np.int32)
    for ki in range(Nk):
        m = np.mod(ints[ki][None, None, :] + ints[:, None, :] - ints[None, :, :], n)
        tab[ki] = m @ mult
    return tab


def kmp2(V, n, ints, eo, ev, KB=None, head=None, lam=1.0):
    """(direct = 2 OS, exchange) per cell.  head: optional (Nk, Nk) H[x, y] added to V[x, y, x] (scaled by lam)."""
    Nk = len(ints)
    eo, ev = np.asarray(eo).ravel(), np.asarray(ev).ravel()
    if KB is None:
        KB = kb_table(n, ints)
    ed = ex = 0.0
    jj = np.arange(Nk)
    for ki in range(Nk):
        v = V[ki]
        if head is not None:
            v = v.copy()
            v[:, ki] += lam * head[ki]
            # exchange partner X[ki,kj,ka] = V[ki,kj,kb]: head class is kb == ki, already inside v
        kb = KB[ki]
        x = v[jj[:, None], kb]
        d = eo[ki] + eo[:, None] - ev[None, :] - ev[kb]
        t = v.conj() / d
        ed += 2 * np.sum(t * v).real
        ex += -np.sum(t * x).real
    return ed / Nk**3, ex / Nk**3


# ------------------------------------------------------------------------------ heads
def pair_derivs(fp, delta=1e-4, delta2=2e-3):
    Ks = np.vstack(
        [np.zeros((1, 3))]
        + [s * dl * np.eye(3) for dl in (delta, delta2) for s in (1, -1)]
    )
    P = fp.P(Ks)
    P0 = P[..., 0]
    A = (P[..., 1:4] - P[..., 4:7]) / (2 * delta)
    B = (P[..., 7:10] + P[..., 10:13] - 2 * P0[..., None]) / delta2**2
    return P0, A, B


def fock_dK(cell, Nk, P0, A, B, C, weights=(1 / 3, 1 / 3, 1 / 3)):
    """Fixed-k dK(k) (AO) from the missing q = 0, K = 0 exchange term beyond v_M S dm S; direction weights
    w_a = <u_a^2> (cubic mesh: 1/3 each; needle along z: (0, 0, 1))."""
    pref = 4 * np.pi / (cell.vol * Nk)
    out = []
    for k in range(Nk):
        dm = 2 * np.outer(C[k][:, 0], C[k][:, 0].conj())
        S = P0[k]
        M = np.zeros((S.shape[0],) * 4, complex)
        for ax in range(3):
            if weights[ax] == 0:
                continue
            M += weights[ax] * (
                np.einsum("ml,ns->mlns", A[k, ..., ax], A[k, ..., ax].conj())
                + 0.5
                * (
                    np.einsum("ml,ns->mlns", B[k, ..., ax], S.conj())
                    + np.einsum("ml,ns->mlns", S, B[k, ..., ax].conj())
                )
            )
        out.append(pref * np.einsum("mlns,ls->mn", M, dm))
    return np.array(out)


def run(orient, n, a=6.0, prec=1e-10, nproc=6, save=None, verbose=True, skip_ov=False):
    atoms = h2_atoms(orient)
    cell = Cell(np.eye(3) * a, atoms, "sto-3g")
    gcut = PK.aft_gcut(cell, prec)
    n = tuple(int(x) for x in n)
    t0 = time.time()
    kb = build_k(cell, atoms, n, gcut, nproc=nproc, verbose=verbose)
    e_hf, eps, it, C, occ = PK.krhf(
        kb, 2, conv=1e-12, kshift=0.0, return_mo=True, jk=kb["jk"]
    )
    assert all(int(o.sum()) == 1 and o[0] for o in occ)
    vm = kb["madelung"]
    t1 = time.time()
    C = np.array(C)
    eps = np.array(eps)
    S_ao = kb["S"]
    ints = kb["ints"]
    del kb
    V = None if skip_ov else aft_ov(cell, atoms, n, C, gcut, nproc=nproc)
    t2 = time.time()
    fp = Fast(cell, n)
    P0, A, B = pair_derivs(fp)
    res = {
        "orient": orient,
        "n": n,
        "a": a,
        "e_hf": e_hf,
        "vm": vm,
        "eps": eps,
        "C": C,
        "P0": P0,
        "A": A,
        "B": B,
        "ints": ints,
        "t_scf": t1 - t0,
        "t_ov": t2 - t1,
        "S_ao": S_ao,
    }
    if save:
        np.savez(
            save,
            V=V,
            **{k: v for k, v in res.items() if k not in ("orient",)},
            orient=orient,
        )
    res["V"] = V
    if verbose:
        print(
            f"  [{orient}] n={n}: HF {e_hf:.12f} it {it} scf {t1 - t0:.0f}s ov {t2 - t1:.0f}s",
            flush=True,
        )
    return res, cell


def account(res, cell, weights=None, phi=None, verbose=True):
    """All pieces from V, C, eps, A, B.  weights: direction weights <u_a u_b> diag (None = cubic 1/3);
    phi: 4th-rank lattice tensor for the direct quadratic head (None = cubic PHI tensor)."""
    V, C, eps, A, B, P0, ints, n = (
        res[k] for k in ("V", "C", "eps", "A", "B", "P0", "ints", "n")
    )
    Nk = len(ints)
    vm = res["vm"]
    w = (
        np.array([1 / 3, 1 / 3, 1 / 3])
        if weights is None
        else np.asarray(weights, float)
    )
    PHI = phi_tensor() if phi is None else phi
    eo = eps[:, 0] - vm
    ev = eps[:, 1]
    KB = kb_table(n, ints)
    Sd, Sx = kmp2(V, n, ints, eo, ev, KB)
    ci, ca = C[:, :, 0], C[:, :, 1]
    gov = np.einsum("km,kmnx,kn->kx", ci.conj(), A, ca)
    gvo = np.einsum("km,kmnx,kn->kx", ca.conj(), A, ci)
    c = 4 * np.pi / cell.vol
    G = c * np.einsum("ia,jb->ijab", gov, gvo.conj())
    Hc = np.einsum("ijaa,a->ij", G, w)
    D = eo[:, None] + eo[None, :] - ev[:, None] - ev[None, :]
    Qc = 2 * np.sum(abs(Hc) ** 2 / D) / Nk**3
    Ql = 2 * np.sum(np.einsum("ijab,ijcd,abcd->ij", G, G.conj(), PHI).real / D) / Nk**3
    e1 = kmp2(V, n, ints, eo, ev, KB, head=Hc, lam=1.0)
    e2 = kmp2(V, n, ints, eo, ev, KB, head=Hc, lam=2.0)
    d1 = np.array(e1) - np.array([Sd, Sx])
    d2 = np.array(e2) - np.array([Sd, Sx])
    Qall = (d2 - 2 * d1) / 2  # (direct, exchange)
    L = d1 - Qall
    dK = fock_dK(cell, Nk, P0, A, B, C, weights=w)
    dKmo = np.einsum("kmp,kmn,knq->kpq", C.conj(), dK, C)
    deo = 0.5 * dKmo[:, 0, 0].real
    dev = 0.5 * dKmo[:, 1, 1].real
    Fd, Fx = kmp2(V, n, ints, eo - deo, ev - dev, KB)
    Fd, Fx = Fd - Sd, Fx - Sx
    out = {
        "S": Sd + Sx,
        "Sd": Sd,
        "Sx": Sx,
        "L": L.sum(),
        "Ld": L[0],
        "Lx": L[1],
        "Qc": Qc,
        "Ql": Ql,
        "Qall": Qall.sum(),
        "Qalld": Qall[0],
        "Qallx": Qall[1],
        "F": Fd + Fx,
        "Fd": Fd,
        "Fx": Fx,
        "deo": deo,
        "dev": dev,
        "dK_ov": dKmo[:, 0, 1],
    }
    out["E1"] = out["S"] + out["L"] + out["Qall"]
    out["E3d"] = out["E1"] + (Ql - Qc) + out["F"]
    if verbose:
        N = Nk
        print(
            f"   S {out['S']:.13e} E1 {out['E1']:.13e} E3d {out['E3d']:.13e} | xNk: L {out['L'] * N:+.5e} "
            f"Qc {Qc * N:+.5e} Qall {out['Qall'] * N:+.5e} Ql {Ql * N:+.5e} F {out['F'] * N:+.5e}",
            flush=True,
        )
    return out
