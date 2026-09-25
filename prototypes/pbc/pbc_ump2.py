"""Gamma-point open-shell MP2 (UMP2) and direct RPA (URPA) from the periodic Gamma ERI (pbc_gamma)
or from the RS-GDF B tensor (pbc_gdf), on a Gamma UHF reference (pbc_uhf).  Real arithmetic.

UMP2 (ferric u_rimp2.rs, PySCF mp.ump2 — identical):
    E_aa = 1/2 sum_{ijab in a} (ia|jb) [(ia|jb) - (ib|ja)] / D     (= 1/4 sum |<ia||jb>|^2 / D)
    E_bb = same with beta
    E_ab = sum_{i a in alpha, J B in beta} (ia|JB)^2 / D            (no exchange)
with ONE shared aux metric: (ia|jb)_s = sum_P B^P_ia,s B^P_jb,s,  (ia|JB) = sum_P B^P_ia,a B^P_JB,b,
and B^P_ia,s = C_s-transform of the SAME periodic B^P_mn used for SCF J/K.

URPA (ferric run_u_pdep_rpa + sternheimer::dielectric_apply_unrestricted, PySCF gw.urpa — identical):
    Pi(iw) = sum_s 2 B_s diag(e_ia,s / (w^2 + e_ia,s^2)) B_s^T           (per-spin factor 2)
    E_c    = (1/2pi) int_0^inf dw { ln det[1 + Pi(iw)] - tr Pi(iw) }
Closed shell (B_a = B_b, e_a = e_b) gives the restricted 4 B diag(..) B^T: factor 4 = 2 spins x 2.
Independent construction (no quadrature, no aux): the spin-orbital direct-RPA plasmon formula on
the joint ov space (ia,a) + (ia,b),  A = D + K, B = K with K the full spin-blocked (ia s|jb t)
matrix:  Omega^2 = eig[D^1/2 (D + 2K) D^1/2],  E_c = 1/2 (sum Omega - tr A).  Its O(K^2) term is
direct UMP2 = 1/2 sum_aa (ia|jb)^2/D + 1/2 sum_bb + sum_ab (ia|JB)^2/D.

Denominators (Gamma, exxdiv='ewald'): K_s += v_M S D_s S shifts EVERY spin's occupied levels by -v_M and
changes no orbital.  'shifted' = eps_occ,s(none) - v_M for s = a, b (what PySCF pbc UCCSD always does,
pbc/cc/ccsd.py:90 `_adjust_occ(mo_energy[s], nocc_s, -madelung)`, and what pbc.mp.UMP2 does when the
UHF ran with exxdiv='ewald'); 'unshifted' = exxdiv=None eigenvalues.  Units Bohr/Hartree.
numpy + PySCF molecular intor only (PySCF pbc is the oracle, in tests / run scripts).
"""

from __future__ import annotations

import numpy as np

from pbc_rpa import _log1p_minus, gl_quadrature

# per-spin chi0 prefactor (2 from +-w; the closed-shell 4 also carries the spin sum).  Module seam for
# mutation tests.
SPIN_FACTOR = 2.0


# ----------------------------------------------------------------------------- denominators
def u_denominators(eps_a, eps_b, na, nb, vm, convention):
    """Per-spin (eo_a, ev_a, eo_b, ev_b) from the exxdiv=None UHF eigenvalues; 'shifted' lowers each
    spin's occupied levels by the SAME v_M (per-spin Madelung, coefficient 1)."""
    if convention not in ("shifted", "unshifted"):
        raise ValueError(
            f"convention must be 'shifted' or 'unshifted', got {convention!r}"
        )
    s = vm if convention == "shifted" else 0.0
    return eps_a[:na] - s, eps_a[na:], eps_b[:nb] - s, eps_b[nb:]


# ------------------------------------------------------------------------------ transforms
def _bia(B, C, nocc, frozen):
    return np.einsum("Pmn,mi,na->Pia", B, C[:, frozen:nocc], C[:, nocc:], optimize=True)


def u_bia(B, Ca, Cb, na, nb, frozen=0):
    """(B_ia,a, B_ia,b) from ONE B^P_mn (shared metric).  Module seam: tests mutate it."""
    return _bia(B, Ca, na, frozen), _bia(B, Cb, nb, frozen)


def _ovov_eri(I, C1, n1, C2, n2, frozen):
    return np.einsum(
        "mnls,mi,na,lj,sb->iajb",
        I,
        C1[:, frozen:n1],
        C1[:, n1:],
        C2[:, frozen:n2],
        C2[:, n2:],
        optimize=True,
    )


def u_ovov(Ca, Cb, na, nb, eri=None, B=None, frozen=0):
    """dict aa, bb, ab of (ia|jb) blocks as [i,a,j,b] from a dense ERI or from B."""
    if (eri is None) == (B is None):
        raise ValueError("give exactly one of eri= or B=")
    if eri is not None:
        return dict(
            aa=_ovov_eri(eri, Ca, na, Ca, na, frozen),
            bb=_ovov_eri(eri, Cb, nb, Cb, nb, frozen),
            ab=_ovov_eri(eri, Ca, na, Cb, nb, frozen),
        )
    Ba, Bb = u_bia(B, Ca, Cb, na, nb, frozen)
    f = lambda X, Y: np.einsum("Pia,Pjb->iajb", X, Y, optimize=True)  # noqa: E731
    return dict(aa=f(Ba, Ba), bb=f(Bb, Bb), ab=f(Ba, Bb))


def _den(eo1, ev1, eo2, ev2):
    return (
        eo1[:, None, None, None]
        - ev1[None, :, None, None]
        + eo2[None, None, :, None]
        - ev2[None, None, None, :]
    )


# ------------------------------------------------------------------------------------ UMP2
def ump2_energy(ov, eoa, eva, eob, evb):
    """Returns (E_total, E_aa, E_bb, E_ab)."""

    def same(x, eo, ev):
        if x.size == 0:
            return 0.0
        return 0.5 * np.sum(x * (x - x.transpose(0, 3, 2, 1)) / _den(eo, ev, eo, ev))

    e_aa, e_bb = same(ov["aa"], eoa, eva), same(ov["bb"], eob, evb)
    e_ab = np.sum(ov["ab"] ** 2 / _den(eoa, eva, eob, evb)) if ov["ab"].size else 0.0
    return e_aa + e_bb + e_ab, e_aa, e_bb, e_ab


def direct_ump2(ov, eoa, eva, eob, evb):
    """Coulomb-only (ring) second order: the O(Pi^2) term of URPA."""
    t = lambda x, d: np.sum(x * x / d) if x.size else 0.0  # noqa: E731
    return (
        0.5 * t(ov["aa"], _den(eoa, eva, eoa, eva))
        + 0.5 * t(ov["bb"], _den(eob, evb, eob, evb))
        + t(ov["ab"], _den(eoa, eva, eob, evb))
    )


def gamma_ump2(Ca, Cb, den, na, nb, eri=None, B=None, frozen=0):
    """UMP2 at Gamma.  den = (eo_a, ev_a, eo_b, ev_b) from u_denominators (frozen applied here)."""
    _check_frozen(frozen, na, nb)
    eoa, eva, eob, evb = den
    return ump2_energy(
        u_ovov(Ca, Cb, na, nb, eri=eri, B=B, frozen=frozen),
        eoa[frozen:],
        eva,
        eob[frozen:],
        evb,
    )


def _check_frozen(frozen, na, nb):
    if frozen < 0 or frozen > min(na, nb):
        raise ValueError(f"frozen={frozen} outside [0, min(na, nb)={min(na, nb)}]")


# ------------------------------------------------------------------------------------ URPA
def _eia(eo, ev):
    return (ev[None, :] - eo[:, None]).ravel()


def _urpa_summand(Bfs, es, w):
    cols = [
        Bf * np.sqrt(SPIN_FACTOR * e / (w * w + e * e))
        for Bf, e in zip(Bfs, es)
        if e.size
    ]
    Bs = np.hstack(cols)
    G = Bs @ Bs.T if Bs.shape[0] <= Bs.shape[1] else Bs.T @ Bs
    lam = np.linalg.eigvalsh(G)
    if lam.min() < -1e-12 * max(1.0, lam.max()):
        raise FloatingPointError(f"Pi has a negative eigenvalue {lam.min():.2e}")
    return _log1p_minus(np.clip(lam, 0.0, None)).sum()


def urpa_quad(Ba, Bb, eoa, eva, eob, evb, n=None, x0=0.5, tol=1e-12, nmax=1024):
    """URPA from per-spin B^P_ia (shared aux).  n None: double from 16 until |dE| < tol."""
    Bfs = [Ba.reshape(Ba.shape[0], -1), Bb.reshape(Bb.shape[0], -1)]
    es = [_eia(eoa, eva), _eia(eob, evb)]
    if any(np.any(e <= 0) for e in es):
        raise ValueError("non-positive e_ia: URPA frequency integral undefined")

    def at(k):
        w, wt = gl_quadrature(k, x0)
        return sum(wk * _urpa_summand(Bfs, es, x) for x, wk in zip(w, wt)) / (2 * np.pi)

    if n is not None:
        return at(n)
    k, prev = 16, at(16)
    while True:
        k *= 2
        cur = at(k)
        if abs(cur - prev) < tol:
            return cur
        if k >= nmax:
            raise RuntimeError(
                f"URPA quadrature not converged at n={k}: {abs(cur - prev):.2e}"
            )
        prev = cur


def urpa_second_order_quad(Ba, Bb, eoa, eva, eob, evb, n=256, x0=0.5):
    """-(1/2pi) int tr Pi^2 / 2 (should equal direct_ump2)."""
    Bfs = [Ba.reshape(Ba.shape[0], -1), Bb.reshape(Bb.shape[0], -1)]
    es = [_eia(eoa, eva), _eia(eob, evb)]
    w, wt = gl_quadrature(n, x0)
    tot = 0.0
    for x, wk in zip(w, wt):
        Bs = np.hstack(
            [Bf * np.sqrt(SPIN_FACTOR * e / (x * x + e * e)) for Bf, e in zip(Bfs, es)]
        )
        Pi = Bs @ Bs.T
        tot += wk * (-0.5 * np.sum(Pi * Pi))
    return tot / (2 * np.pi)


def _joint_K(ov):
    na_ = ov["aa"].shape[0] * ov["aa"].shape[1]
    nb_ = ov["bb"].shape[0] * ov["bb"].shape[1]
    ka, kb, kab = (
        ov["aa"].reshape(na_, na_),
        ov["bb"].reshape(nb_, nb_),
        ov["ab"].reshape(na_, nb_),
    )
    return np.block([[ka, kab], [kab.T, kb]]), na_


def urpa_plasmon(ov, eoa, eva, eob, evb):
    """Spin-orbital direct RPA plasmon formula on the joint (alpha + beta) ov space."""
    K, _ = _joint_K(ov)
    e = np.concatenate([_eia(eoa, eva), _eia(eob, evb)])
    sd = np.sqrt(e)
    lam = np.linalg.eigvalsh(np.diag(e * e) + 2 * sd[:, None] * K * sd[None, :])
    if lam.min() <= 0:
        raise FloatingPointError(
            "URPA instability: (A-B)^1/2 (A+B) (A-B)^1/2 not positive"
        )
    return 0.5 * (np.sqrt(lam).sum() - np.sum(e + np.diag(K)))


def u_ov_factor(ov, thresh=1e-14):
    """(B_a, B_b) with a SHARED aux index from the eigen-decomposition of the joint PSD ov matrix."""
    K, nova = _joint_K(ov)
    s, U = np.linalg.eigh(K)
    keep = s > thresh * max(1.0, s.max())
    L = (U[:, keep] * np.sqrt(s[keep])).T
    na_, va = ov["aa"].shape[:2]
    nb_, vb = ov["bb"].shape[:2]
    return L[:, :nova].reshape(len(L), na_, va), L[:, nova:].reshape(len(L), nb_, vb)


def gamma_urpa(Ca, Cb, den, na, nb, eri=None, B=None, frozen=0, method="quad", **kw):
    """URPA at Gamma.  method 'quad' (B route; with eri the joint ov block is eigen-factorised) or 'plasmon'."""
    _check_frozen(frozen, na, nb)
    eoa, eva, eob, evb = den
    eoa, eob = eoa[frozen:], eob[frozen:]
    if method == "quad":
        if B is not None:
            Ba, Bb = u_bia(B, Ca, Cb, na, nb, frozen)
        else:
            Ba, Bb = u_ov_factor(u_ovov(Ca, Cb, na, nb, eri=eri, frozen=frozen))
        return urpa_quad(Ba, Bb, eoa, eva, eob, evb, **kw)
    if method == "plasmon":
        return urpa_plasmon(
            u_ovov(Ca, Cb, na, nb, eri=eri, B=B, frozen=frozen), eoa, eva, eob, evb
        )
    raise ValueError(f"method must be quad/plasmon, got {method!r}")


# -------------------------------------------- a^-3 predictor: harmonic kernel on the molecule
def u_r2_kernel_c3(mol, na, nb, h=1e-4, conv=1e-13, guess=None, second=False):
    """Shifted-convention box-limit coefficients (residual = c3/a^3) of UHF, UMP2 and URPA from a
    MOLECULAR calculation only: add (k/2)|r-r'|^2 (k = 4pi/3a^3) to every ee/en/nn interaction,
    re-converge the UHF (orbitals relax), evaluate UMP2 / URPA (plasmon) with the modified ERI, and
    differentiate in k at k=0 (central difference).  pbc_rpa.r2_kernel_c3 generalised to UHF."""
    from pbc_uhf import uhf

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

    def energies(k):
        I = I0 + k * dI
        u = uhf(S, h0 + k * dV, I, enn0 + k * dE, na, nb, conv=conv, guess=guess)
        den = u_denominators(u["eps_a"], u["eps_b"], na, nb, 0.0, "unshifted")
        return np.array(
            [
                u["e"],
                gamma_ump2(u["Ca"], u["Cb"], den, na, nb, eri=I)[0],
                gamma_urpa(u["Ca"], u["Cb"], den, na, nb, eri=I, method="plasmon"),
            ]
        )

    ep, em = energies(h), energies(-h)
    d = (ep - em) / (2 * h) * (4 * np.pi / 3)
    if not second:
        return dict(hf=d[0], mp2=d[1], rpa=d[2])
    # c6: SECOND order in the r^2 kernel (orbital relaxation in the harmonic en/ee field), 1/2 E'' k^2, k^2 ~ a^-6.
    # It is the leading correction for a SPHERICAL system (no l=4 cubic term -> no c5); a non-spherical one also has c5.
    d2 = (ep - 2 * energies(0.0) + em) / h**2 * 0.5 * (4 * np.pi / 3) ** 2
    return dict(hf=d[0], mp2=d[1], rpa=d[2], hf6=d2[0], mp26=d2[1], rpa6=d2[2])


def uniform_occ_shift_slope(Ca, Cb, eps_a, eps_b, na, nb, eri, h=1e-5):
    """dE/ds for eps_occ(both spins) -> eps_occ - s at fixed integrals/orbitals (central difference),
    for (UMP2, URPA).  Predicts the unshifted-convention 1/a term: E_unshifted - E_shifted ~
    -(v_M a) * slope / a  (+ O(a^-2))."""

    def e(s):
        den = u_denominators(eps_a, eps_b, na, nb, s, "shifted")
        return np.array(
            [
                gamma_ump2(Ca, Cb, den, na, nb, eri=eri)[0],
                gamma_urpa(Ca, Cb, den, na, nb, eri=eri, method="plasmon"),
            ]
        )

    return (e(h) - e(-h)) / (2 * h)


__all__ = [
    "u_denominators",
    "u_bia",
    "u_ovov",
    "ump2_energy",
    "direct_ump2",
    "gamma_ump2",
    "urpa_quad",
    "urpa_second_order_quad",
    "urpa_plasmon",
    "u_ov_factor",
    "gamma_urpa",
    "u_r2_kernel_c3",
    "uniform_occ_shift_slope",
]
