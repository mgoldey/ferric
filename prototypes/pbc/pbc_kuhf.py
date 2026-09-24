"""k-point sampled periodic UHF (open-shell HF), Gamma-centred Monkhorst-Pack mesh.

Stage 9 (k-point UHF) prototype.  Extends pbc_kpts.krhf the way pbc_uhf extends pbc_gamma.rhf. Per spin
sigma in {a, b} and mesh point k (Bloch convention and kernels of pbc_kpts; unit occupations):

    D_s(k)  = C_s,occ(k) C_s,occ(k)^dagger
    F_s(k)  = h(k) + J[D_a + D_b](k) - K[D_s](k) - v_M S(k) D_s(k) S(k)
    E/cell  = (1/Nk) sum_k sum_s 1/2 Re tr[(h(k) + F_s(k)) D_s(k)] + E_nn

J(k) needs only the q = 0 (k-summed) density; K_s(k) = (1/Nk) sum_k' Kker[k,k'] D_s(k'), i.e. the exchange of
spin s at momentum transfer q = k' - k from THAT spin's density.  Madelung (exxdiv='ewald'): the SAME v_M as
k-RHF (the diag(n) supercell Madelung constant), on the single-spin density, coefficient 1, per k with no 1/Nk
(PySCF df_jk._ewald_exxdiv_for_G0 is called with the (2, Nk) dm stack; pbc/scf/kuhf.py get_veff forms
vj[0]+vj[1]-vk).  k-RHF's "K + v_M S D S, F = h + J - K/2" is the same with D = 2 D_s.

Occupations: GLOBAL aufbau per spin over all k (PySCF kuhf.get_occ: the na*Nk lowest alpha levels over the whole
mesh, the nb*Nk lowest beta levels, independently).  DIIS over the stacked (spin, k) commutators, real
coefficients from Re sum_{s,k} <e_i, e_j>.

J/K source: dense pbc_kpts kernels (default) or any dm-stack -> (J, K) callable `jk` (e.g. pbc_kgdf.jk_from_kB).
Seams `_jk_spin`, `_madelung_term`, `_aufbau` are module functions so tests can mutate them.  Units Bohr/Hartree.
"""

from __future__ import annotations

import numpy as np

from pbc_kpts import jk_k


# --------------------------------------------------------------- mutable seams (tests)
def _madelung_term(S, Ds, vm):
    """v_M S(k) D_s(k) S(k) for every k (one spin)."""
    return vm * np.einsum("kab,kbc,kcd->kad", S, Ds, S)


def _jk_spin(jk, Da, Db):
    """J from the TOTAL density, K per spin from that spin's density (both linear in dm)."""
    Ja, Ka = jk(Da)
    Jb, Kb = jk(Db)
    return Ja + Jb, Ka, Kb


def _aufbau(eps, n_total):
    """Global aufbau: the n_total lowest levels over all k (ties broken by (k, i) order, stable)."""
    allE = [(e, k, i) for k, ek in enumerate(eps) for i, e in enumerate(ek)]
    allE.sort(key=lambda t: (t[0], t[1], t[2]))
    occ = [np.zeros(len(ek), bool) for ek in eps]
    for _, k, i in allE[:n_total]:
        occ[k][i] = True
    homo = allE[n_total - 1][0] if n_total > 0 else -np.inf
    lumo = allE[n_total][0] if n_total < len(allE) else np.inf
    return occ, homo, lumo


def _aufbau_per_k(eps, n_total):
    """MUTANT reference: n_total/Nk lowest levels at EVERY k (a molecule-style per-k aufbau)."""
    nk = len(eps)
    per = n_total // nk
    occ = [np.arange(len(ek)) < per for ek in eps]
    return (
        occ,
        max(ek[per - 1] for ek in eps) if per else -np.inf,
        min(ek[per] for ek in eps),
    )


# --------------------------------------------------------------------------- helpers
def orthogonalizers(S, lindep=1e-8):
    X = []
    for Sk in S:
        s, U = np.linalg.eigh(Sk)
        X.append(U[:, s > lindep] / np.sqrt(s[s > lindep]))
    return X


def _diag(F, X):
    eps, C = [], []
    for Fk, Xk in zip(F, X):
        e, c = np.linalg.eigh(Xk.conj().T @ Fk @ Xk)
        eps.append(e)
        C.append(Xk @ c)
    return eps, C


def _dm(C, occ):
    return np.array([C[k][:, occ[k]] @ C[k][:, occ[k]].conj().T for k in range(len(C))])


def spin_square(Ca, occa, Cb, occb, S):
    """<S^2> of the GIANT (supercell) determinant, PySCF KUHF.spin_square convention:
    Sz(Sz+1) + N_b - sum_k ||C_a,occ(k)^H S(k) C_b,occ(k)||^2 (alpha/beta Bloch orbitals at different k are
    orthogonal).  Divide by nothing: <S^2> is NOT per cell (Sz^2 grows as Nk^2)."""
    na = sum(int(o.sum()) for o in occa)
    nb = sum(int(o.sum()) for o in occb)
    sz = 0.5 * (na - nb)
    ov = sum(
        np.sum(abs(Ca[k][:, occa[k]].conj().T @ S[k] @ Cb[k][:, occb[k]]) ** 2)
        for k in range(len(S))
    )
    return sz * (sz + 1) + nb - ov


def core_guess(kb, na, nb, X=None, mix=0.0):
    """Hcore guess per k with global aufbau per spin; mix>0 rotates the beta global HOMO/LUMO (same k only)."""
    X = X or orthogonalizers(kb["S"])
    Nk = kb["Nk"]
    eps, C = _diag(kb["h"], X)
    occa, *_ = _aufbau(eps, na * Nk)
    occb, *_ = _aufbau(eps, nb * Nk)
    Cb = [c.copy() for c in C]
    if mix and nb > 0:
        # beta HOMO (highest occupied) and the lowest virtual AT THE SAME k
        kh, ih = max(
            ((k, i) for k in range(Nk) for i in np.where(occb[k])[0]),
            key=lambda t: eps[t[0]][t[1]],
        )
        il = int(np.where(~occb[kh])[0][0])
        c, s = np.cos(mix), np.sin(mix)
        ho, lu = C[kh][:, ih].copy(), C[kh][:, il].copy()
        Cb[kh][:, ih], Cb[kh][:, il] = c * ho + s * lu, -s * ho + c * lu
    return _dm(C, occa), _dm(Cb, occb)


# --------------------------------------------------------------------------- SCF
def kuhf(
    kb,
    na,
    nb,
    conv=1e-11,
    maxiter=300,
    kshift=None,
    jk=None,
    guess=None,
    mix=0.0,
    lindep=1e-8,
    level_shift=0.0,
    diis_start=1,
    aufbau=None,
):
    """k-point UHF.  na / nb = electrons per spin PER CELL (the mesh holds na*Nk / nb*Nk).
    kshift defaults to kb['madelung'].  guess: None -> core guess, or (Da(k) stack, Db(k) stack).
    Returns dict(e, eps_a, eps_b, Ca, Cb, occ_a, occ_b, Da, Db, it, s2, gap_a, gap_b, nocc_a_k, nocc_b_k)."""
    S, h, Nk = kb["S"], kb["h"], kb["Nk"]
    vm = kb["madelung"] if kshift is None else kshift
    jk = jk or (lambda dm: jk_k(kb, dm))
    auf = aufbau or _aufbau
    X = orthogonalizers(S, lindep)
    if guess is None:
        Da, Db = core_guess(kb, na, nb, X, mix)
    else:
        Da, Db = (np.array(g, complex) for g in guess)
    focks, errs = [], []
    e_old = 0.0
    for it in range(maxiter):
        J, Ka, Kb = _jk_spin(jk, Da, Db)
        if vm:
            Ka = Ka + _madelung_term(S, Da, vm)
            Kb = Kb + _madelung_term(S, Db, vm)
        Fa, Fb = h + J - Ka, h + J - Kb
        e = (
            0.5
            * (
                np.einsum("kmn,knm->", h + Fa, Da) + np.einsum("kmn,knm->", h + Fb, Db)
            ).real
            / Nk
            + kb["enn"]
        )
        err = [
            X[k].conj().T @ (F[k] @ D[k] @ S[k] - S[k] @ D[k] @ F[k]) @ X[k]
            for F, D in ((Fa, Da), (Fb, Db))
            for k in range(Nk)
        ]
        emax = max(abs(x).max() for x in err)
        if it > 0 and abs(e - e_old) < conv and emax < 1e-7:
            break
        e_old = e
        if it >= diis_start:
            focks.append((Fa, Fb))
            errs.append(err)
            focks, errs = focks[-8:], errs[-8:]
        Fa_u, Fb_u = Fa, Fb
        if len(focks) > 1:
            m = len(focks)
            B = -np.ones((m + 1, m + 1))
            B[-1, -1] = 0
            for i in range(m):
                for j in range(m):
                    B[i, j] = sum(np.vdot(x, y).real for x, y in zip(errs[i], errs[j]))
            rhs = np.zeros(m + 1)
            rhs[-1] = -1
            c = np.linalg.lstsq(B, rhs, rcond=None)[0][:m]
            Fa_u = sum(ci * f[0] for ci, f in zip(c, focks))
            Fb_u = sum(ci * f[1] for ci, f in zip(c, focks))
        if level_shift:
            Fa_u = Fa_u + level_shift * (S - np.einsum("kab,kbc,kcd->kad", S, Da, S))
            Fb_u = Fb_u + level_shift * (S - np.einsum("kab,kbc,kcd->kad", S, Db, S))
        eps_a, Ca = _diag(Fa_u, X)
        eps_b, Cb = _diag(Fb_u, X)
        occa, *_ = auf(eps_a, na * Nk)
        occb, *_ = auf(eps_b, nb * Nk)
        Da, Db = _dm(Ca, occa), _dm(Cb, occb)
    else:
        raise RuntimeError(
            f"k-UHF not converged (last dE {e - e_old:.2e}, err {emax:.2e})"
        )
    # final orbitals of the converged, unextrapolated Fock; occupations from THOSE levels
    eps_a, Ca = _diag(Fa, X)
    eps_b, Cb = _diag(Fb, X)
    occa, ha, la = auf(eps_a, na * Nk)
    occb, hb, lb = auf(eps_b, nb * Nk)
    return dict(
        e=e,
        eps_a=eps_a,
        eps_b=eps_b,
        Ca=Ca,
        Cb=Cb,
        occ_a=occa,
        occ_b=occb,
        Da=Da,
        Db=Db,
        it=it,
        s2=spin_square(Ca, occa, Cb, occb, S),
        gap_a=la - ha,
        gap_b=lb - hb,
        err=emax,
        nocc_a_k=[int(o.sum()) for o in occa],
        nocc_b_k=[int(o.sum()) for o in occb],
    )


def staged_kuhf(kb, na, nb, **kw):
    """The Iteration-6 ewald-trap remedy at k: converge with the Madelung term OFF, then switch it on from that
    density (v_M S D_s S = v_M x occupied projector at every k, so the None stationary density is also
    stationary under ewald)."""
    kw = dict(kw)
    vm = kw.pop("kshift", None)
    vm = kb["madelung"] if vm is None else vm
    r0 = kuhf(kb, na, nb, kshift=0.0, **kw)
    kw.pop("guess", None)
    kw.pop("mix", None)
    return kuhf(kb, na, nb, kshift=vm, guess=(r0["Da"], r0["Db"]), **kw), r0


# ------------------------------------------------------- supercell unfolding (for the Gamma-supercell anchor)
def unfold_dm(cell, kb, D):
    """Real-space supercell density D_sc[(T,m),(T',n)] = (1/Nk) sum_k e^{ik.(T - T')} D(k)_mn from Bloch D(k)
    (chi_mk = sum_L e^{ik.L} phi_m(r - L); supercell AOs ordered cell-image-major as pbc_kpts.supercell_cell)."""
    T = kb["ints"] @ cell.a
    ph = np.exp(1j * kb["kpts"] @ T.T)  # (Nk, Nk_images)
    Nk, nao = kb["Nk"], D.shape[1]
    Dsc = np.einsum("kt,ku,kmn->tmun", ph, ph.conj(), D) / Nk
    return Dsc.reshape(Nk * nao, Nk * nao)


__all__ = [
    "kuhf",
    "staged_kuhf",
    "spin_square",
    "core_guess",
    "orthogonalizers",
    "unfold_dm",
]
