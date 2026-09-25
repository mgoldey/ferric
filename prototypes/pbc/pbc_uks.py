"""Gamma-point periodic UKS / ROKS (Iteration 10): spin-polarized XC on the periodic grid of Iteration 8
(pbc_dft.PeriodicGrid, lattice-summed AOs) + the per-spin exchange/Madelung machinery of Iteration 6
(pbc_uhf).  Only molecular primitives (libxc via pyscf.dft.libxc, the dense pure-AFT I or any
jk(D) -> (J, K) callable); PySCF pbc is the oracle only (run_uks_*.py, tests).

Per spin s in {a, b}, global hybrid fraction alpha = libxc.hybrid_coeff(xc) (1 for xc='HF'):

    F_s = h + J[D_a + D_b] - alpha (K[D_s] + v_M S D_s S) + V_xc^s[rho_a, rho_b]
    E   = sum_s tr D_s h + 1/2 tr D J - alpha/2 sum_s tr D_s (K[D_s] + v_M S D_s S) + E_xc + E_nn

i.e. the Madelung term (exxdiv='ewald') rides on the EXACT-EXCHANGE part only (as Iteration 8's RKS) and is
applied per spin with coefficient 1 on D_s (as Iteration 6's UHF).  PySCF 2.13 pbc/dft/uks.py get_veff:
vk = hyb * get_k(dm_a, dm_b) with the exxdiv G=0 term inside get_k -> identical.  Closed shell
(D_a = D_b = D/2): F = h + J - (alpha/2)(K[D] + v_M S D S) + V_xc = pbc_dft.rks exactly.
Stationary densities are the same with and without v_M (it is alpha v_M x occupied projector per spin);
E_ewald - E_none = -alpha v_M (N_a + N_b)/2.  Per-spin gap criterion for the ewald trap: gap_s >= alpha v_M.

Mutation seam: module-level _MUTANT in {None, 'unpolarized', 'madelung_full_k', 'hyb_half_per_spin'}.
Units Bohr / Hartree, Cartesian AOs."""

from __future__ import annotations

from collections import deque

import numpy as np
from pyscf.dft import libxc

import pbc_dft as pd

_MUTANT = None


# ----------------------------------------------------------------------------- numint
def eval_vxc_uks(grid, Da, Db, xc):
    """Spin-polarized semilocal (E_xc, V_a, V_b) on (grid.weights, grid.ao).  For hybrids libxc returns
    only the DFT part.  GGA: V_s = int vrho_s chi chi + (2 vsigma_ss grad rho_s + vsigma_ab grad rho_s')
    . grad(chi chi)  (libxc sigma order aa, ab, bb)."""
    fam = pd.xc_family(xc)
    if fam == "MGGA":
        raise NotImplementedError("meta-GGA not prototyped")
    gga = fam == "GGA"
    if (
        _MUTANT == "unpolarized"
    ):  # MUTANT: closed-shell kernel on the total density, same V for both spins
        exc, V, _ = pd.eval_vxc(grid, Da + Db, xc)
        return exc, V, V
    ra, rb = pd.rho_on_grid(grid, Da, gga), pd.rho_on_grid(grid, Db, gga)
    exc, vxc = libxc.eval_xc(xc, (ra, rb), spin=1, deriv=1)[:2]
    w = grid.weights
    rt = (ra[0] + rb[0]) if gga else (ra + rb)
    e = np.dot(w, rt * exc)
    vrho = vxc[0]
    if not gga:
        a0 = pd._ao0(grid)
        return (
            e,
            a0.T @ (a0 * (w * vrho[:, 0])[:, None]),
            a0.T @ (a0 * (w * vrho[:, 1])[:, None]),
        )
    vs = vxc[1]
    ao = grid.ao
    out = []
    for s, (r_s, r_o, v_ss) in enumerate(((ra, rb, vs[:, 0]), (rb, ra, vs[:, 2]))):
        wv = np.empty((4, len(w)))
        wv[0] = 0.5 * w * vrho[:, s]
        wv[1:] = w * (2 * v_ss * r_s[1:] + vs[:, 1] * r_o[1:])
        V = ao[0].T @ np.einsum("xpi,xp->pi", ao, wv)
        out.append(V + V.T)
    return e, out[0], out[1]


def hybrid_fraction(xc):
    if str(xc).upper() == "HF":
        return 1.0
    if libxc.rsh_coeff(xc)[0] != 0:
        raise NotImplementedError(
            "range-separated hybrids need the attenuated periodic K"
        )
    return libxc.hybrid_coeff(xc)


# ------------------------------------------------------------------------- Fock pieces
def _fock_energy(S, h, jk, enn, Da, Db, grid, xc, hyb, kshift):
    """(E, F_a, F_b, E_xc) at (D_a, D_b)."""
    J, Ka, Kb = (
        (lambda a, b: (a[0] + b[0], a[1], b[1]))(jk(Da), jk(Db))
        if hyb
        else (jk(Da)[0] + jk(Db)[0], 0.0, 0.0)
    )
    h_eff = (
        0.5 * hyb if _MUTANT == "hyb_half_per_spin" else hyb
    )  # MUTANT: RKS's 1/2 carried into the per-spin K
    if (
        _MUTANT == "madelung_full_k"
    ):  # MUTANT: Madelung on the full K instead of the exact-exchange fraction
        Kea = h_eff * Ka + kshift * S @ Da @ S
        Keb = h_eff * Kb + kshift * S @ Db @ S
    else:
        Kea = h_eff * (Ka + kshift * S @ Da @ S)
        Keb = h_eff * (Kb + kshift * S @ Db @ S)
    exc, Va, Vb = (0.0, 0.0, 0.0) if grid is None else eval_vxc_uks(grid, Da, Db, xc)
    Fa, Fb = h + J - Kea + Va, h + J - Keb + Vb
    e = (
        np.sum((Da + Db) * h)
        + 0.5 * np.sum((Da + Db) * J)
        - 0.5 * (np.sum(Da * Kea) + np.sum(Db * Keb))
        + exc
        + enn
    )
    return e, Fa, Fb, exc


def _orth(S, thresh=1e-8):
    s, U = np.linalg.eigh(S)
    return U[:, s > thresh] / np.sqrt(s[s > thresh])


def _diis(focks, errs):
    n = len(focks)
    B = -np.ones((n + 1, n + 1))
    B[-1, -1] = 0
    for i in range(n):
        for j in range(n):
            B[i, j] = errs[i] @ errs[j]
    rhs = np.zeros(n + 1)
    rhs[-1] = -1
    return np.linalg.lstsq(B, rhs, rcond=None)[0][:n]


def spin_square(Coa, Cob, S):
    na, nb = Coa.shape[1], Cob.shape[1]
    sz = 0.5 * (na - nb)
    ov = Coa.T @ S @ Cob
    return sz * (sz + 1) + nb - np.sum(ov * ov)


def occ_gap(F, D, S, X=None):
    """Occupation-aware gap of Fock F for density D: min eps(unoccupied) - max eps(occupied), occupations
    n_i = c_i^T S D S c_i of F's eigenvectors (NOT the sorted aufbau gap: a hole state gives a NEGATIVE value).
    inf if either set is empty."""
    X = _orth(S) if X is None else X
    eps, C = np.linalg.eigh(X.T @ F @ X)
    C = X @ C
    n = np.einsum("mi,mn,ni->i", C, S @ D @ S, C)
    occ, vir = eps[n > 0.5], eps[n <= 0.5]
    return vir.min() - occ.max() if len(occ) and len(vir) else np.inf


# --------------------------------------------------------------------------------- UKS
def uks(
    S,
    h,
    jk,
    enn,
    na,
    nb,
    grid,
    xc,
    kshift=0.0,
    conv=1e-11,
    maxiter=300,
    guess=None,
    mix=0.0,
    staged=False,
    level_shift=0.0,
    diis_start=0,
):
    """Gamma UKS.  xc='HF' -> UHF (grid unused).  kshift = v_M (exxdiv='ewald').
    guess: None -> core guess (pbc_uhf.core_guess, beta HOMO/LUMO `mix`), or (D_a, D_b).
    diis_start > maxiter with level_shift > 0: plain level-shifted Roothaan (for DIIS stagnation near degeneracy).
    staged=True: converge with kshift=0 first, then switch v_M on from that density (Iteration 6 remedy).
    Returns dict(e, eps_a, eps_b, Ca, Cb, Da, Db, it, s2, gaps, trap) -- trap = min per-spin gap < hyb v_M."""
    hyb = hybrid_fraction(xc)
    if str(xc).upper() == "HF":
        grid = None
    if staged and kshift:
        first = uks(
            S,
            h,
            jk,
            enn,
            na,
            nb,
            grid,
            xc,
            0.0,
            conv,
            maxiter,
            guess,
            mix,
            level_shift=level_shift,
            diis_start=diis_start,
        )
        guess, it0 = (first["Da"], first["Db"]), first["it"]
    else:
        it0 = 0
    from pbc_uhf import core_guess

    X = _orth(S)
    Da, Db = (
        core_guess(S, h, na, nb, mix)
        if guess is None
        else (np.array(guess[0]), np.array(guess[1]))
    )
    focks, errs = deque(maxlen=8), deque(maxlen=8)
    e_old = 0.0
    for it in range(maxiter):
        e, Fa, Fb, exc = _fock_energy(S, h, jk, enn, Da, Db, grid, xc, hyb, kshift)
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
            c = _diis(focks, errs)
            Fa = sum(ci * f[0] for ci, f in zip(c, focks))
            Fb = sum(ci * f[1] for ci, f in zip(c, focks))
        if level_shift:
            Fa = Fa + level_shift * (S - S @ Da @ S)
            Fb = Fb + level_shift * (S - S @ Db @ S)
        Ca, Cb = (
            X @ np.linalg.eigh(X.T @ Fa @ X)[1],
            X @ np.linalg.eigh(X.T @ Fb @ X)[1],
        )
        Da, Db = Ca[:, :na] @ Ca[:, :na].T, Cb[:, :nb] @ Cb[:, :nb].T
    else:
        raise RuntimeError(
            f"UKS not converged ({xc}, last dE {e - e_old:.2e}, err {err:.2e})"
        )
    e, F0a, F0b, exc = _fock_energy(
        S, h, jk, enn, Da, Db, grid, xc, hyb, kshift
    )  # undamped, at final D
    eps_a, Ca = np.linalg.eigh(X.T @ F0a @ X)
    eps_b, Cb = np.linalg.eigh(X.T @ F0b @ X)
    Ca, Cb = X @ Ca, X @ Cb
    gaps = (occ_gap(F0a, Da, S, X), occ_gap(F0b, Db, S, X))
    return dict(
        e=e,
        exc=exc,
        eps_a=eps_a,
        eps_b=eps_b,
        Ca=Ca,
        Cb=Cb,
        Da=Da,
        Db=Db,
        it=it + it0,
        hyb=hyb,
        s2=spin_square(Ca[:, :na], Cb[:, :nb], S),
        gaps=gaps,
        trap=bool(min(gaps) < hyb * kshift),
    )


# -------------------------------------------------------------------------------- ROKS
def roks(
    S, h, jk, enn, na, nb, grid, xc, kshift=0.0, conv=1e-11, maxiter=300, guess=None
):
    """Gamma ROKS: common spatial orbitals, Roothaan effective Fock (PySCF scf/rohf.get_roothaan_fock
    projector form, Fc = (F_a+F_b)/2 on the diagonal blocks); DIIS on [F_eff, D_a + D_b].
    Energy functional identical to uks(); only the variational space is restricted."""
    hyb = hybrid_fraction(xc)
    if str(xc).upper() == "HF":
        grid = None
    X = _orth(S)
    n = S.shape[0]
    if guess is None:
        C = X @ np.linalg.eigh(X.T @ h @ X)[1]
    else:
        C = X @ np.linalg.eigh(X.T @ (S @ (guess[0] + guess[1]) @ S) @ X)[1][:, ::-1]
    Da, Db = C[:, :na] @ C[:, :na].T, C[:, :nb] @ C[:, :nb].T
    focks, errs = deque(maxlen=8), deque(maxlen=8)
    e_old = 0.0
    for it in range(maxiter):
        e, Fa, Fb, exc = _fock_energy(S, h, jk, enn, Da, Db, grid, xc, hyb, kshift)
        fc = 0.5 * (Fa + Fb)
        pc, po, pv = Db @ S, (Da - Db) @ S, np.eye(n) - Da @ S
        F = (
            0.5 * (pc.T @ fc @ pc + po.T @ fc @ po + pv.T @ fc @ pv)
            + po.T @ Fb @ pc
            + po.T @ Fa @ pv
            + pv.T @ fc @ pc
        )
        F = F + F.T
        Dt = Da + Db
        er = X.T @ (F @ Dt @ S - S @ Dt @ F) @ X
        err = abs(er).max()
        if abs(e - e_old) < conv and err < 1e-7 and it > 0:
            break
        e_old = e
        focks.append(F)
        errs.append(er.ravel())
        Fd = (
            F
            if len(focks) < 2
            else sum(ci * f for ci, f in zip(_diis(focks, errs), focks))
        )
        C = X @ np.linalg.eigh(X.T @ Fd @ X)[1]
        Da, Db = C[:, :na] @ C[:, :na].T, C[:, :nb] @ C[:, :nb].T
    else:
        raise RuntimeError(
            f"ROKS not converged ({xc}, last dE {e - e_old:.2e}, err {err:.2e})"
        )
    eps, C = np.linalg.eigh(X.T @ F @ X)
    C = X @ C
    return dict(
        e=e,
        exc=exc,
        eps=eps,
        C=C,
        Da=Da,
        Db=Db,
        it=it,
        s2=spin_square(C[:, :na], C[:, :nb], S),
    )


# -------------------------------------------------------------------------- box limit
def uks_c3_closed_form(mol, Ca, Cb, na, nb, hyb):
    """a^-3 coefficient of the exxdiv='ewald' UKS box residual: the UHF formula with the exchange spread
    scaled by the exact-exchange fraction (semilocal XC is local -> no a^-3; Hartree+en+nn of a neutral
    cell unchanged):  c3 = -(2 pi/3)(|d|^2 + hyb (Omega_a + Omega_b)), Omega from the KS orbitals."""
    from pbc_uhf import uhf_c3_closed_form

    _, p = uhf_c3_closed_form(mol, Ca, Cb, na, nb)
    return -(2 * np.pi / 3) * (
        p["dipole"] @ p["dipole"] + hyb * (p["omega_a"] + p["omega_b"])
    ), p


class MolGrid:
    """Molecular grid (coords, weights) with molecular AOs -- lets uks() run a MOLECULAR UKS (box-limit
    independent construction)."""

    def __init__(self, mol, coords, weights):
        self.coords, self.weights = coords, weights
        self.ao = np.asarray(mol.eval_gto("GTOval_cart_deriv1", coords))


def uks_r2_kernel_c3(
    mol, na, nb, grid, xc, h=1e-4, conv=1e-13, guess=None, second=False
):
    """Independent construction of c3: harmonic kernel (k/2)|r-r'|^2 added to every Coulomb interaction
    (ee in J AND K -- K then carries the hybrid fraction automatically --, en, nn) of a MOLECULAR UKS on a
    fixed molecular grid; orbitals relax; d/dk at 0 by central difference x (4 pi/3)."""
    S = mol.intor("int1e_ovlp_cart")
    r = mol.intor("int1e_r_cart")
    r2 = np.einsum(
        "xxmn->mn", mol.intor("int1e_rr_cart").reshape(3, 3, mol.nao, mol.nao)
    )
    I0 = mol.intor("int2e_cart")
    h0 = mol.intor("int1e_kin_cart") + mol.intor("int1e_nuc_cart")
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
    es = []
    for k in (h, -h, 0.0):
        I = I0 + k * dI
        es.append(
            uks(
                S,
                h0 + k * dV,
                pd.dense_jk(I),
                enn0 + k * dE,
                na,
                nb,
                grid,
                xc,
                conv=conv,
                guess=guess,
            )["e"]
        )
    c3 = (es[0] - es[1]) / (2 * h) * (4 * np.pi / 3)
    if not second:
        return c3
    # relaxation (second order in k = 4pi/3a^3): dE += (1/2) E''(0) k^2 = c6 / a^6
    return c3, 0.5 * (es[0] + es[1] - 2 * es[2]) / h**2 * (4 * np.pi / 3) ** 2


__all__ = [
    "uks",
    "roks",
    "eval_vxc_uks",
    "hybrid_fraction",
    "spin_square",
    "occ_gap",
    "uks_c3_closed_form",
    "uks_r2_kernel_c3",
    "MolGrid",
]
