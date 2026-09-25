"""Gamma-point periodic UHF (open-shell HF) on the dense AFT ERI or RS-GDF B tensors.

At Gamma the periodic UHF is the molecular UHF on (S, h, I, E_nn) exactly as pbc_gamma.rhf is
the molecular RHF: every lattice quantity is already folded into those arrays. Per spin
sigma in {a, b}:

    F_sigma = h + J[D_a + D_b] - K_sigma,      K_sigma = K[D_sigma] + v_M S D_sigma S
    E       = 1/2 sum_sigma tr D_sigma (h + F_sigma) + E_nn

Madelung (exxdiv='ewald') per spin: PySCF pbc/df/df_jk.py `_ewald_exxdiv_for_G0` adds
v_M S dm S to vk[i] for EACH density in the list it receives; pbc/scf/uhf.py get_veff passes
(D_a, D_b) and forms vj[0]+vj[1]-vk. So the per-spin shift uses the SAME v_M as RHF, applied to
the single-spin density with coefficient 1. RHF's "K + v_M S D S then F = h + J - K/2" is the
same thing because D = 2 D_sigma. Energy: -v_M/2 (N_a + N_b) at convergence (idempotent D_sigma).

Everything here is geometry-agnostic, like the Rust seam should be: S, h, E_nn and the J/K
source (dense I or a D -> (J, K) callable) are injected.  Units Bohr / Hartree.
"""

from __future__ import annotations

from collections import deque

import numpy as np


# --------------------------------------------------------------- mutable seams (tests)
def _madelung_term(S, Ds, vm):
    """v_M S D_sigma S for one spin density (PySCF _ewald_exxdiv_for_G0, per dm)."""
    return vm * S @ Ds @ S


def _jk_spin(jk, Da, Db):
    """J from the TOTAL density, K per spin from that spin's density (same ERI / B)."""
    Ja, Ka = jk(Da)
    Jb, Kb = jk(Db)
    return Ja + Jb, Ka, Kb


def dense_jk(I):
    def jk(D):
        return np.einsum("mnls,ls->mn", I, D), np.einsum("mlsn,ls->mn", I, D)

    return jk


# --------------------------------------------------------------------------- guess
def core_guess(S, h, na, nb, mix=0.0):
    """Hcore guess; occupy na / nb lowest. mix>0 rotates the beta HOMO/LUMO by `mix` rad
    (ferric solve_uhf does a small HOMO/LUMO mixing when na == nb to allow symmetry breaking)."""
    X = _orth(S)
    e, C = np.linalg.eigh(X.T @ h @ X)
    C = X @ C
    Cb = C.copy()
    if mix and nb > 0 and nb < C.shape[1]:
        c, s = np.cos(mix), np.sin(mix)
        ho, lu = C[:, nb - 1].copy(), C[:, nb].copy()
        Cb[:, nb - 1], Cb[:, nb] = c * ho + s * lu, -s * ho + c * lu
    return C[:, :na] @ C[:, :na].T, Cb[:, :nb] @ Cb[:, :nb].T


def _orth(S, thresh=1e-8):
    s, U = np.linalg.eigh(S)
    return U[:, s > thresh] / np.sqrt(s[s > thresh])


# --------------------------------------------------------------------------- SCF
def uhf(
    S,
    h,
    I,
    enn,
    na,
    nb,
    conv=1e-10,
    maxiter=200,
    kshift=0.0,
    jk=None,
    guess=None,
    mix=0.0,
    return_mo=False,
    diis_start=1,
    level_shift=0.0,
):
    """Gamma UHF on (S, h, I or jk, E_nn).  kshift = v_M (exxdiv='ewald'), applied per spin.

    guess: None -> core guess (with optional beta HOMO/LUMO `mix`), or (Da, Db).
    Returns dict(e, eps_a, eps_b, Ca, Cb, Da, Db, it, s2)."""
    if jk is None:
        jk = dense_jk(I)
    X = _orth(S)
    Da, Db = (
        core_guess(S, h, na, nb, mix)
        if guess is None
        else (np.array(guess[0]), np.array(guess[1]))
    )
    focks, errs = deque(maxlen=8), deque(maxlen=8)
    e_old = 0.0
    for it in range(maxiter):
        J, Ka, Kb = _jk_spin(jk, Da, Db)
        if kshift:
            Ka = Ka + _madelung_term(S, Da, kshift)
            Kb = Kb + _madelung_term(S, Db, kshift)
        Fa, Fb = h + J - Ka, h + J - Kb
        e = 0.5 * (np.sum(Da * (h + Fa)) + np.sum(Db * (h + Fb))) + enn
        ea = X.T @ (Fa @ Da @ S - S @ Da @ Fa) @ X
        eb = X.T @ (Fb @ Db @ S - S @ Db @ Fb) @ X
        err = max(abs(ea).max(), abs(eb).max())
        if abs(e - e_old) < conv and err < 1e-7 and it > 0:
            break
        e_old = e
        if it >= diis_start:
            focks.append((Fa, Fb))
            errs.append(np.concatenate([ea.ravel(), eb.ravel()]))
        if len(focks) > 1:
            n = len(focks)
            B = -np.ones((n + 1, n + 1))
            B[-1, -1] = 0
            for i in range(n):
                for j in range(n):
                    B[i, j] = errs[i] @ errs[j]
            rhs = np.zeros(n + 1)
            rhs[-1] = -1
            c = np.linalg.lstsq(B, rhs, rcond=None)[0][:n]
            Fa = sum(ci * f[0] for ci, f in zip(c, focks))
            Fb = sum(ci * f[1] for ci, f in zip(c, focks))
        if level_shift:
            Fa = Fa + level_shift * (S - S @ Da @ S)
            Fb = Fb + level_shift * (S - S @ Db @ S)
        eps_a, Ca = np.linalg.eigh(X.T @ Fa @ X)
        eps_b, Cb = np.linalg.eigh(X.T @ Fb @ X)
        Ca, Cb = X @ Ca, X @ Cb
        Da, Db = Ca[:, :na] @ Ca[:, :na].T, Cb[:, :nb] @ Cb[:, :nb].T
    else:
        raise RuntimeError(
            f"UHF not converged (last dE {e - e_old:.2e}, err {err:.2e})"
        )
    # final orbitals of the converged (undamped, unshifted) Fock
    eps_a, Ca = np.linalg.eigh(X.T @ (h + J - Ka) @ X)
    eps_b, Cb = np.linalg.eigh(X.T @ (h + J - Kb) @ X)
    Ca, Cb = X @ Ca, X @ Cb
    out = dict(
        e=e,
        eps_a=eps_a,
        eps_b=eps_b,
        Ca=Ca,
        Cb=Cb,
        Da=Da,
        Db=Db,
        it=it,
        s2=spin_square(Ca[:, :na], Cb[:, :nb], S),
    )
    return out


def spin_square(Coa, Cob, S):
    """<S^2> of a UHF determinant: Sz(Sz+1) + N_b - sum_ij |<i_a|j_b>|^2 (S = lattice overlap at Gamma)."""
    na, nb = Coa.shape[1], Cob.shape[1]
    sz = 0.5 * (na - nb)
    ov = Coa.T @ S @ Cob
    return sz * (sz + 1) + nb - np.sum(ov * ov)


# ------------------------------------------------------------------ box-limit predictor
def uhf_c3_closed_form(mol, Ca, Cb, na, nb):
    """First-order (Hellmann-Feynman, orbitals need NOT relax) a^-3 coefficient of the
    exxdiv='ewald' box residual for a UHF determinant in a CUBIC box:

        after the Madelung term the G=0-dropped kernel = 1/r + (k/2)|r-r'|^2 + O(a^-5), k = 4pi/3a^3
        Hartree + e-n + n-n (neutral):     (k/2) * (-|d|^2)
        exchange, spin sigma:              -(k/2) * Omega_sigma,
        Omega_sigma = sum_i <i|r^2|i> - sum_ij |<i|r|j>|^2    (occupied sigma; Foster-Boys invariant spread)

    => c3 = -(2 pi / 3) (|d|^2 + Omega_a + Omega_b).   RHF check: Omega_a = Omega_b = sigma^2 (one orbital)
    gives -(4pi/3) sigma^2, Iteration 1's formula.  exxdiv=None adds +v_M (N_a+N_b)/2 exactly (1/a)."""
    r = mol.intor("int1e_r")
    r2 = np.einsum("xxmn->mn", mol.intor("int1e_rr").reshape(3, 3, mol.nao, mol.nao))
    Z, R = mol.atom_charges().astype(float), mol.atom_coords()
    omega = []
    dip = Z @ R
    for C, n in ((Ca, na), (Cb, nb)):
        Co = C[:, :n]
        rij = np.einsum("xmn,mi,nj->xij", r, Co, Co)
        omega.append(np.einsum("mn,mi,ni->", r2, Co, Co) - np.sum(rij * rij))
        dip = dip - np.einsum("xii->x", rij)
    c3 = -(2 * np.pi / 3) * (dip @ dip + omega[0] + omega[1])
    return c3, dict(omega_a=omega[0], omega_b=omega[1], dipole=dip)


def uhf_r2_kernel_c3(mol, na, nb, h=1e-4, conv=1e-13, guess=None):
    """Independent construction of the same c3: add the harmonic kernel (k/2)|r-r'|^2 to EVERY Coulomb
    interaction of the molecular UHF (ee J and K, en, nn), re-converge (orbitals relax), d/dk at k=0
    by central difference.  (pbc_rpa.r2_kernel_c3 generalised to UHF.)"""
    S = mol.intor("int1e_ovlp")
    r = mol.intor("int1e_r")
    r2 = np.einsum("xxmn->mn", mol.intor("int1e_rr").reshape(3, 3, mol.nao, mol.nao))
    I0 = mol.intor("int2e")
    h0 = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    Z, R = mol.atom_charges().astype(float), mol.atom_coords()
    dI = 0.5 * (
        np.einsum("mn,ls->mnls", S, r2)
        + np.einsum("mn,ls->mnls", r2, S)
        - 2 * np.einsum("xmn,xls->mnls", r, r)
    )
    dV = -0.5 * sum(
        z * (r2 - 2 * np.einsum("x,xmn->mn", Ra, r) + (Ra @ Ra) * S)
        for z, Ra in zip(Z, R)
    )
    dE = 0.5 * sum(
        Z[i] * Z[j] * np.sum((R[i] - R[j]) ** 2)
        for i in range(len(Z))
        for j in range(i + 1, len(Z))
    )
    enn0 = mol.energy_nuc()
    es = [
        uhf(S, h0 + k * dV, I0 + k * dI, enn0 + k * dE, na, nb, conv=conv, guess=guess)[
            "e"
        ]
        for k in (h, -h)
    ]
    return (es[0] - es[1]) / (2 * h) * (4 * np.pi / 3)


__all__ = [
    "uhf",
    "spin_square",
    "core_guess",
    "dense_jk",
    "uhf_c3_closed_form",
    "uhf_r2_kernel_c3",
]
