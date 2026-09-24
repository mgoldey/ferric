"""Gamma-point closed-shell MP2 from the periodic Gamma ERI (pbc_gamma) or from the RS-GDF
B tensor (pbc_gdf).  Real arithmetic: at Gamma the orbitals are real and

    E_MP2 = sum_{ijab} (ia|jb) [2 (ia|jb) - (ib|ja)] / (e_i + e_j - e_a - e_b).

(ia|jb) is taken in the SAME G=0-dropped kernel as the HF ERI.  No G=0 correction is needed in
the integrals: ov pair densities are neutral (C_o^T S C_v = 0), so their G->0 limit is finite.
The exchange divergence enters ONLY through the orbital energies: at Gamma, exxdiv='ewald'
(K += v_M S D S) leaves C unchanged and moves every occupied level by -v_M.  Two denominator
conventions therefore exist, and nothing else differs between them:

    'shifted'   : occupied eps from the exxdiv='ewald' Fock (= none-Fock eps_occ - v_M).
                  What PySCF pbc.mp.RMP2 / KMP2 use when the HF ran with exxdiv='ewald'
                  (they take mf.mo_energy), and what PySCF pbc CCSD ALWAYS uses (it rebuilds
                  the Fock with exxdiv=None and then applies _adjust_occ(eps, nocc, -madelung)).
    'unshifted' : occupied eps from the exxdiv=None Fock.  What PySCF RMP2/KMP2 use when the
                  HF ran with exxdiv=None.

Units Bohr/Hartree.  Molecular PySCF intor + numpy only (PySCF pbc is the oracle, in tests).
"""

from __future__ import annotations

import numpy as np


def ovov_from_eri(I, Co, Cv):
    """(ia|jb) as [i,a,j,b] from a dense (mn|ls) tensor."""
    return np.einsum("mnls,mi,na,lj,sb->iajb", I, Co, Cv, Co, Cv, optimize=True)


def bia_from_B(B, Co, Cv):
    """B^P_mn -> B^P_ia (the MO-transformed three-index tensor ri_mp2 consumes)."""
    return np.einsum("Pmn,mi,na->Pia", B, Co, Cv, optimize=True)


def ovov_from_B(B, Co, Cv):
    Bia = bia_from_B(B, Co, Cv)
    return np.einsum("Pia,Pjb->iajb", Bia, Bia, optimize=True)


def mp2_energy(ovov, eo, ev):
    """Closed-shell MP2 correlation energy, and its (os, ss) components."""
    d = (
        eo[:, None, None, None]
        - ev[None, :, None, None]
        + eo[None, None, :, None]
        - ev[None, None, None, :]
    )
    t = ovov / d
    e_os = np.einsum("iajb,iajb->", t, ovov)
    e_ss = np.einsum("iajb,iajb->", t, ovov - ovov.transpose(0, 3, 2, 1))
    return e_os + e_ss, e_os, e_ss


def denominators(eps_none, nocc, vm, convention):
    """Orbital energies for the MP2 denominator from the exxdiv=None eigenvalues."""
    if convention == "unshifted":
        e = eps_none
    elif convention == "shifted":
        e = eps_none.copy()
        e[:nocc] -= vm
    else:
        raise ValueError(
            f"convention must be 'shifted' or 'unshifted', got {convention!r}"
        )
    return e[:nocc], e[nocc:]


def gamma_mp2(C, eps, nocc, eri=None, B=None, frozen=0):
    """MP2 at Gamma from converged RHF (C, eps as used in the denominator).  Exactly one of
    eri (dense nao^4) or B (naux, nao, nao) must be given.  frozen: number of core orbitals."""
    if (eri is None) == (B is None):
        raise ValueError("give exactly one of eri= or B=")
    if frozen < 0 or frozen > nocc:
        raise ValueError(f"frozen={frozen} outside [0, nocc={nocc}]")
    Co, Cv = C[:, frozen:nocc], C[:, nocc:]
    ovov = ovov_from_eri(eri, Co, Cv) if eri is not None else ovov_from_B(B, Co, Cv)
    return mp2_energy(ovov, eps[frozen:nocc], eps[nocc:])


# --------------------------------------------------------------- Makov-Payne-type prediction
def dipole_prediction_h2_minimal(mol, C, eps, edge):
    """Predicted a^-3 residual of shifted-denominator MP2 vs the molecular limit, for a system
    with ONE occupied and ONE virtual orbital (H2/minimal basis), from molecular quantities only.

    Leading cubic-lattice G->0 correction of a Coulomb integral between densities (q1,d1,R1)
    and (q2,d2,R2) (charge, dipole, int rho r^2 about a common origin), G=0-dropped kernel,
    relative to the isolated value and after the exxdiv='ewald' Madelung term:
        delta(1|2) = -(4pi/3 Omega) [d1.d2 - (q1 R2 + q2 R1)/2]  +  O(a^-5).
    Applied at frozen molecular orbitals:
      (ia|ia):      q=0, delta = -(4pi/3 Omega) |d_ia|^2
      eps_i:        exchange with the unit-charge |phi_i|^2 -> -(4pi/3 Omega) sigma_i^2
                    (J + V_ne of the NEUTRAL cell gives a p-independent constant: cancels in D)
      eps_a:        -(ia|ai) exchange with the neutral ov density -> +(4pi/3 Omega) |d_ia|^2
    E = (ia|ia)^2 / D,  D = 2 eps_i - 2 eps_a  ->  dE = 2 (ia|ia) d(ia|ia)/D - (ia|ia)^2 dD/D^2."""
    r = mol.intor("int1e_r")
    rr = mol.intor("int1e_rr").reshape(3, 3, mol.nao, mol.nao)
    ci, ca = C[:, 0], C[:, 1]
    d_ia = np.einsum("xm,m->x", r @ ca, ci)
    cen = np.einsum("xmn,m,n->x", r, ci, ci)
    s2 = np.einsum("xxmn,m,n->", rr, ci, ci) - cen @ cen
    k = 4 * np.pi / (3 * edge**3)
    iaia = mol.ao2mo(C, compact=False).reshape(2, 2, 2, 2)[0, 1, 0, 1]
    D = 2 * eps[0] - 2 * eps[1]
    d_iaia = -k * d_ia @ d_ia
    dD = 2 * (-k * s2) - 2 * (k * d_ia @ d_ia)
    return 2 * iaia * d_iaia / D - iaia**2 * dD / D**2, dict(
        d_ia=d_ia, sigma2=s2, iaia=iaia, D=D
    )


__all__ = [
    "gamma_mp2",
    "mp2_energy",
    "ovov_from_eri",
    "ovov_from_B",
    "bia_from_B",
    "denominators",
    "dipole_prediction_h2_minimal",
]
