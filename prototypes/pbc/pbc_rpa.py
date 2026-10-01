"""Gamma-point closed-shell direct RPA (dRPA) correlation energy from the periodic Gamma ERI
(pbc_gamma) or from the RS-GDF B tensor (pbc_gdf).  Real arithmetic (Gamma orbitals are real).

Frequency-integral form (ferric ferric-rpa/energy.rs and PySCF gw/rpa.py, same numbers):

    Pi_PQ(iw) = 4 sum_ia B^P_ia B^Q_ia e_ia / (w^2 + e_ia^2),   e_ia = eps_a - eps_i > 0
    E_c       = (1/2pi) int_0^inf dw  { ln det[1 + Pi(iw)] - tr Pi(iw) }

(ferric: Pi = -chi0 v >= 0, "factor 4 = 2 spin x 2 from +-w"; PySCF writes diel = -Pi and
ln det(1 - diel) + tr diel -- identical.)  The half-line integral with 1/(2pi) is correct: its
O(Pi^2) term  -(1/2pi) int tr Pi^2/2  is exactly the direct (Coulomb-only) MP2 energy
2 sum (ia|jb)^2 / (eps_i + eps_j - eps_a - eps_b) (tested).

Independent construction (no frequency integral, no aux): the plasmon formula for the singlet
direct-RPA problem  A = D + 2K, B = 2K  (K_{ia,jb} = (ia|jb), D = diag e_ia),
    Omega^2 = eig[ D^{1/2} (D + 4K) D^{1/2} ],    E_c = 1/2 ( sum Omega - tr A ),
and the ring-CCD Riccati equation  B + A T + T A + T B T = 0,  E_c = 1/2 tr(B T)  (third route).

Denominators: identical story to pbc_mp2.py -- at Gamma exxdiv='ewald' moves eps_occ by -v_M and
changes nothing else; use pbc_mp2.denominators(eps_none, nocc, v_M, 'shifted'|'unshifted').
Units Bohr/Hartree.  numpy + PySCF molecular intor only (PySCF pbc is never imported here).
"""

from __future__ import annotations

import numpy as np

from pbc_mp2 import bia_from_B, ovov_from_B, ovov_from_eri


# ------------------------------------------------------------------------- quadrature
def gl_quadrature(n, x0=0.5):
    """Gauss-Legendre mapped to [0, inf) by w = x0 (1+x)/(1-x): ferric quadrature.rs
    gauss_legendre_nodes and PySCF _get_scaled_legendre_roots (same map, same weights)."""
    x, wt = np.polynomial.legendre.leggauss(n)
    return x0 * (1 + x) / (1 - x), wt * 2 * x0 / (1 - x) ** 2


def _log1p_minus(x):
    """log(1+x) - x without cancellation (x >= 0 up to roundoff)."""
    small = np.abs(x) < 1e-3
    xs = x[small]
    out = np.empty_like(x)
    out[small] = (
        xs * xs * (-1 / 2 + xs * (1 / 3 + xs * (-1 / 4 + xs * (1 / 5 - xs / 6))))
    )
    out[~small] = np.log1p(x[~small]) - x[~small]
    return out


def _summand(Bf, e, w):
    """ln det(1 + Pi) - tr Pi = sum_k [log(1+l_k) - l_k] over eigenvalues of Pi, at one frequency.
    Bf (naux, nov), e (nov,).  NOT slogdet(1+Pi) - tr(Pi): at the large-w GL nodes Pi ~ w^-2, the
    weight ~ w^2, and that difference cancels catastrophically (measured: 2e-10 drift at n=1024 on
    H2O/6-31G DF, naux 100+).  The smaller Gram matrix is used (same nonzero spectrum)."""
    Bs = Bf * np.sqrt(4 * e / (w * w + e * e))
    G = Bs @ Bs.T if Bs.shape[0] <= Bs.shape[1] else Bs.T @ Bs
    lam = np.linalg.eigvalsh(G)
    if lam.min() < -1e-12 * max(1.0, lam.max()):
        raise FloatingPointError(f"Pi has a negative eigenvalue {lam.min():.2e}")
    return _log1p_minus(np.clip(lam, 0.0, None)).sum()


def drpa_quad(Bia, eo, ev, n=None, x0=0.5, tol=1e-12, nmax=1024, return_n=False):
    """dRPA from B^P_ia (naux, nocc, nvir).  n given: fixed n-point GL; n None: double n from
    16 until two successive estimates differ by < tol (returns the finer one)."""
    naux = Bia.shape[0]
    Bf = Bia.reshape(naux, -1)
    e = (ev[None, :] - eo[:, None]).ravel()
    if np.any(e <= 0):
        raise ValueError("non-positive e_ia: dRPA frequency integral undefined")

    def at(n):
        w, wt = gl_quadrature(n, x0)
        return sum(wk * _summand(Bf, e, x) for x, wk in zip(w, wt)) / (2 * np.pi)

    if n is not None:
        return (at(n), n) if return_n else at(n)
    k, prev = 16, at(16)
    while True:
        k *= 2
        cur = at(k)
        if abs(cur - prev) < tol:
            return (cur, k) if return_n else cur
        if k >= nmax:
            raise RuntimeError(
                f"quadrature not converged to {tol} at n={k}: last change {abs(cur - prev):.2e}"
            )
        prev = cur


def drpa_second_order_quad(Bia, eo, ev, n=256, x0=0.5):
    """-(1/2pi) int_0^inf tr Pi^2 / 2: the O(V^2) term of the log expansion (should be direct MP2)."""
    naux = Bia.shape[0]
    Bf = Bia.reshape(naux, -1)
    e = (ev[None, :] - eo[:, None]).ravel()
    w, wt = gl_quadrature(n, x0)
    tot = 0.0
    for x, wk in zip(w, wt):
        Bs = Bf * np.sqrt(4 * e / (x * x + e * e))
        Pi = Bs @ Bs.T
        tot += wk * (-0.5 * np.sum(Pi * Pi))
    return tot / (2 * np.pi)


# ------------------------------------------------------- dense (ia|jb) constructions
def _ov_matrix(ovov):
    no, nv = ovov.shape[:2]
    return ovov.reshape(no * nv, no * nv)


def drpa_plasmon(ovov, eo, ev):
    """Plasmon formula, singlet direct RPA.  ovov = (ia|jb) as [i,a,j,b]."""
    K = _ov_matrix(ovov)
    e = (ev[None, :] - eo[:, None]).ravel()
    sd = np.sqrt(e)
    M = np.diag(e * e) + 4 * sd[:, None] * K * sd[None, :]
    lam = np.linalg.eigvalsh(M)
    if lam.min() <= 0:
        raise FloatingPointError(
            "dRPA instability: (A-B)^1/2 (A+B) (A-B)^1/2 not positive"
        )
    return 0.5 * (np.sqrt(lam).sum() - np.sum(e + 2 * np.diag(K)))


def drpa_riccati(ovov, eo, ev, tol=1e-13, maxiter=500):
    """Ring-CCD (singlet, A = D + 2K, B = 2K): solve B + A T + T A + T B T = 0 by
    T_new(ij) = -[B + K'T + T K' + T B T]_ij / (e_i + e_j) with K' = 2K (off-diag of A)."""
    K = _ov_matrix(ovov)
    e = (ev[None, :] - eo[:, None]).ravel()
    Bm = 2 * K
    den = e[:, None] + e[None, :]
    T = -Bm / den
    for _ in range(maxiter):
        R = Bm + Bm @ T + T @ Bm + T @ Bm @ T
        Tn = -R / den
        if abs(Tn - T).max() < tol:
            T = Tn
            break
        T = Tn
    else:
        raise RuntimeError("ring-CCD Riccati did not converge")
    return 0.5 * np.sum(Bm * T)


def direct_mp2(ovov, eo, ev):
    """2 sum (ia|jb)^2 / (eps_i + eps_j - eps_a - eps_b)."""
    d = (
        eo[:, None, None, None]
        - ev[None, :, None, None]
        + eo[None, None, :, None]
        - ev[None, None, None, :]
    )
    return 2 * np.sum(ovov * ovov / d)


def gamma_drpa(C, eps, nocc, eri=None, B=None, frozen=0, method="quad", **kw):
    """dRPA at Gamma from converged RHF (C, eps as used in the denominator).  Exactly one of
    eri (dense nao^4) or B (naux, nao, nao).  method: 'quad' (B route; with eri the ov block is
    eigen-factorised first), 'plasmon', 'riccati'."""
    if (eri is None) == (B is None):
        raise ValueError("give exactly one of eri= or B=")
    if frozen < 0 or frozen > nocc:
        raise ValueError(f"frozen={frozen} outside [0, nocc={nocc}]")
    Co, Cv = C[:, frozen:nocc], C[:, nocc:]
    eo, ev = eps[frozen:nocc], eps[nocc:]
    if method == "quad":
        if B is not None:
            Bia = bia_from_B(B, Co, Cv)
        else:
            Bia = ov_factor(ovov_from_eri(eri, Co, Cv))
        return drpa_quad(Bia, eo, ev, **kw)
    ovov = ovov_from_eri(eri, Co, Cv) if eri is not None else ovov_from_B(B, Co, Cv)
    if method == "plasmon":
        return drpa_plasmon(ovov, eo, ev)
    if method == "riccati":
        return drpa_riccati(ovov, eo, ev)
    raise ValueError(f"method must be quad/plasmon/riccati, got {method!r}")


def ov_factor(ovov, thresh=1e-14):
    """B^P_ia with sum_P B B = (ia|jb), from the eigen-decomposition of the PSD ov block."""
    no, nv = ovov.shape[:2]
    s, U = np.linalg.eigh(_ov_matrix(ovov))
    keep = s > thresh * max(1.0, s.max())
    return (U[:, keep] * np.sqrt(s[keep])).T.reshape(-1, no, nv)


# --------------------------------------------------------------- Makov-Payne-type prediction
def drpa_moment_prediction_h2_minimal(mol, C, eps, edge):
    """Predicted a^-3 residual of shifted-denominator dRPA vs the molecule, for ONE occupied and
    ONE virtual orbital (H2/minimal), from molecular quantities only.  Same frozen-orbital
    lattice model as pbc_mp2.dipole_prediction_h2_minimal (k = 4pi/3a^3):
        d(ia|ia) = -k |d_ia|^2,  d eps_i = -k sigma_i^2,  d eps_a = +k |d_ia|^2
    With nov = 1:  E = 1/2 [ sqrt(D (D + 4K)) - D - 2K ],  D = eps_a - eps_i,  K = (ia|ia):
        dE/dD = 1/2 [ (D + 2K)/Omega - 1 ],   dE/dK = D/Omega - 1,   Omega = sqrt(D(D+4K))."""
    r = mol.intor("int1e_r")
    rr = mol.intor("int1e_rr").reshape(3, 3, mol.nao, mol.nao)
    ci, ca = C[:, 0], C[:, 1]
    d_ia = np.einsum("xm,m->x", r @ ca, ci)
    cen = np.einsum("xmn,m,n->x", r, ci, ci)
    s2 = np.einsum("xxmn,m,n->", rr, ci, ci) - cen @ cen
    k = 4 * np.pi / (3 * edge**3)
    K = mol.ao2mo(C, compact=False).reshape(2, 2, 2, 2)[0, 1, 0, 1]
    D = eps[1] - eps[0]
    Om = np.sqrt(D * (D + 4 * K))
    dEdD = 0.5 * ((D + 2 * K) / Om - 1)
    dEdK = D / Om - 1
    dD = k * (s2 + d_ia @ d_ia)
    dK = -k * d_ia @ d_ia
    return dEdD * dD + dEdK * dK, dict(
        d_ia=d_ia, sigma2=s2, K=K, D=D, dEdD=dEdD, dEdK=dEdK
    )


__all__ = [
    "gl_quadrature",
    "drpa_quad",
    "drpa_second_order_quad",
    "drpa_plasmon",
    "drpa_riccati",
    "direct_mp2",
    "gamma_drpa",
    "ov_factor",
    "drpa_moment_prediction_h2_minimal",
]


# ------------------------------------------- general a^-3 prediction: the |r-r'|^2 kernel
def r2_kernel_c3(mol, h=1e-4, conv=1e-13):
    """Leading box-limit residual (shifted/exxdiv='ewald' convention) of HF, MP2 and dRPA for ANY
    molecule/basis, from a molecular calculation only.  After the Madelung term, the cubic-lattice
    G=0-dropped Coulomb kernel differs from 1/r at O(a^-3) by the harmonic kernel
        dv(r, r') = (k/2) |r - r'|^2,      k = 4 pi / (3 a^3)
    (it reproduces delta(1|2) = -k [d1.d2 - (q1 R2 + q2 R1)/2] of pbc_mp2).  Apply it to every
    Coulomb interaction -- ee (J and K), en and nn -- run the molecular RHF (orbitals RELAX, unlike the
    frozen-orbital nov=1 models), MP2 and dRPA, and differentiate in k at k=0 (central difference).
    Returns dict c3[method] with residual = c3 / a^3."""
    from pbc_gamma import rhf
    from pbc_mp2 import gamma_mp2

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
    nel = mol.nelectron
    nocc = nel // 2

    def energies(k):
        e, eps, _, C = rhf(
            S, h0 + k * dV, I0 + k * dI, enn0 + k * dE, nel, conv=conv, return_mo=True
        )
        I = I0 + k * dI
        return np.array(
            [
                e,
                gamma_mp2(C, eps, nocc, eri=I)[0],
                gamma_drpa(C, eps, nocc, eri=I, method="plasmon"),
            ]
        )

    d = (energies(h) - energies(-h)) / (2 * h) * (4 * np.pi / 3)
    return dict(hf=d[0], mp2=d[1], drpa=d[2])
