"""Gamma-point amplitude-threshold local MP2 (LMP2) on a periodic supercell.

Periodic port of scripts/amplitude_lmp2_proto.py (Wang/Shen/Head-Gordon single-threshold LMP2,
the Python rig ferric's Rust `amplitude_lmp2` was ported from).  The masked-CG solver, Hylleraas
energy, pair energies, pivoted Cholesky and orthonormalisations are IMPORTED from that script
unchanged; what is new here is only what periodicity changes:

1. Localisation functional.  At Gamma the orbitals are supercell-periodic, so the molecular
   Boys functional (built from <mu|r|nu>) is undefined: r is not a periodic operator, and the
   L=0-only molecular integrals see the cell boundary.  We maximise the Resta/Berghold
   functional  sum_k w_k sum_i |z_k,ii|^2,  z_k = <phi_i| exp(i b_k.r) |phi_j>,  b_k the supercell
   reciprocal vectors, w_k = (|a_k|/2pi)^2 (orthorhombic cells) [Resta PRL 80 1800 (1998);
   Berghold et al. PRB 61 10040 (2000); Silvestrelli PRB 59 9703 (1999)].  exp(i b_k.r) IS
   periodic in the supercell, and its AO matrix is the lattice-summed pair FT at G = -b_k — the
   same primitive (pbc_gamma.pair_ft) the periodic integrals already use.  sum_k w_k (1-|z_ii|^2)
   is the Wannier spread up to O((b sigma)^4); for L >> sigma the maximiser is the Gamma-point
   maximally-localised Wannier function.  Re z_k and Im z_k are real symmetric, so the Jacobi
   sweep is literally Boys' with 6 weighted "coordinates".
   Why not Pipek-Mezey: PM (lattice-summed S + atomic partition) is also periodic-safe and needs
   no position operator; Berghold was chosen because it is the Boys analogue (what ferric's
   amplitude_lmp2 uses molecularly), gives periodic centroids arg(z_ii)/2pi (needed for
   minimum-image pair and fit domains), and gives the spreads the VV-HV hard-virtual weights need.
2. Centroids are phases: f_k = arg(z_k,ii)/2pi (fractional), defined modulo the lattice.  Every
   distance (pair cutoff, aux fit domain) is therefore a MINIMUM-IMAGE distance.
3. Integrals: (ia|jb) at Gamma = sum_P B^P_ia B^P_jb from the periodic RS-GDF B (G=0-dropped
   kernel): it already contains every lattice image of j, so the Eq-8 integral mask needs no
   distance at all.  Per-pair domain-local fits use the PERIODIC metric J2 and J3.
4. Denominators: shifted (exxdiv='ewald' Fock, Iteration 3); Foo from the ewald Fock.

Units Bohr/Hartree.  Real arithmetic (Gamma).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts")
)
import amplitude_lmp2_proto as alp  # noqa: E402  (molecular rig: solver, energy, orthonormalisations)

from pbc_gamma import rhf  # noqa: E402
from pbc_gdf import jk_from_B  # noqa: E402
from pbc_mp2 import gamma_mp2  # noqa: E402


# --------------------------------------------------------------------------------- SCF
def gamma_scf(data, nelec, conv=1e-12):
    """Gamma RHF on the supercell B (exxdiv='ewald').  Returns canonical C/eps RE-DIAGONALISED
    from the ewald Fock of the final density, so canonical and localised paths share one F."""
    S, h, B, vm = data["S"], data["h"], data["B"], data["madelung"]
    e, _, it, C = rhf(
        S,
        h,
        None,
        data["enn"],
        nelec,
        conv=conv,
        kshift=vm,
        jk=jk_from_B(B),
        return_mo=True,
    )
    nocc = nelec // 2
    D = 2 * C[:, :nocc] @ C[:, :nocc].T
    J, K = jk_from_B(B)(D)
    F = h + J - 0.5 * (K + vm * S @ D @ S)
    s, U = np.linalg.eigh(S)
    X = U[:, s > 1e-8] / np.sqrt(s[s > 1e-8])
    eps, Cp = np.linalg.eigh(X.T @ F @ X)
    return dict(e=e, eps=eps, C=X @ Cp, F=F, nocc=nocc, iters=it)


def canonical_mp2(scf, B):
    return gamma_mp2(scf["C"], scf["eps"], scf["nocc"], B=B)[0]


# ------------------------------------------------------------------------ localisation
def resta_weights(a_sc, tol=1e-10):
    """w_k = (|a_k|/2pi)^2; only valid for orthorhombic cells (general cells need Silvestrelli's
    metric weights over more G vectors) -> refuse instead of silently mis-weighting."""
    g = a_sc @ a_sc.T
    if abs(g - np.diag(np.diag(g))).max() > tol:
        raise NotImplementedError(
            "Resta/Berghold weights implemented for orthorhombic supercells only"
        )
    return (np.linalg.norm(a_sc, axis=1) / (2 * np.pi)) ** 2


def resta_mats(Zk, weights):
    """(real symmetric matrix, weight) list for the Jacobi sweep: Re z_k and Im z_k per k."""
    out = []
    for Z, wk in zip(Zk, weights):
        out += [(Z.real, wk), (Z.imag, wk)]
    return out


def jacobi_localize(C, mats, maxsweep=1000, tol=1e-11, seed=None):
    """Maximise f = sum_c w_c sum_i (C_i^T M_c C_i)^2 by 2x2 Jacobi rotations (Boys/PM form):
    df(theta) = A (1 - cos 4t) + B sin 4t,  A = sum_c w_c [X_ij^2 - (X_ii - X_jj)^2/4],
    B = sum_c w_c X_ij (X_ii - X_jj); optimum 4t = atan2(B, -A).  `seed` applies a random
    orthogonal start (to test that the maximum is start-independent)."""
    C = C.copy()
    n = C.shape[1]
    if seed is not None and n > 1:
        q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))
        C = C @ q
    X = np.array([C.T @ M @ C for M, _ in mats])
    w = np.array([wc for _, wc in mats])

    def fval():
        return float(np.einsum("c,ci->", w, np.einsum("cii->ci", X) ** 2))

    f0 = fval()
    for sweep in range(maxsweep):
        tmax = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                xij, dd = X[:, i, j], X[:, i, i] - X[:, j, j]
                A = w @ (xij**2 - 0.25 * dd**2)
                Bv = w @ (xij * dd)
                if np.hypot(A, Bv) < 1e-15:
                    continue
                t = 0.25 * np.arctan2(Bv, -A)
                if abs(t) < 1e-15:
                    continue
                tmax = max(tmax, abs(t))
                c, s = np.cos(t), np.sin(t)
                ci, cj = C[:, i].copy(), C[:, j].copy()
                C[:, i], C[:, j] = c * ci + s * cj, -s * ci + c * cj
                Xi, Xj = X[:, i, :].copy(), X[:, j, :].copy()
                X[:, i, :], X[:, j, :] = c * Xi + s * Xj, -s * Xi + c * Xj
                Xi, Xj = X[:, :, i].copy(), X[:, :, j].copy()
                X[:, :, i], X[:, :, j] = c * Xi + s * Xj, -s * Xi + c * Xj
        if tmax < tol:
            break
    # gradient of f wrt antisymmetric generator (i,j): 4 B_ij -> should vanish at a stationary point
    d = np.einsum("cii->ci", X)
    grad = 4 * np.einsum("c,cij,cij->ij", w, X, d[:, :, None] - d[:, None, :])
    return C, dict(
        f0=f0,
        f=fval(),
        sweeps=sweep + 1,
        grad=float(abs(np.triu(grad, 1)).max()) if n > 1 else 0.0,
    )


def resta_centroids(C, Zk, a_sc):
    """Periodic centroids: fractional f_k = arg(z_k,ii)/2pi (mod 1) -> Cartesian f @ a_sc."""
    ph = np.stack([np.angle(np.einsum("mi,mn,ni->i", C, Z, C)) for Z in Zk], axis=1)
    return (np.mod(ph / (2 * np.pi), 1.0)) @ a_sc


def resta_spreads(C, Zk, weights):
    return sum(
        wk * (1 - np.abs(np.einsum("mi,mn,ni->i", C, Z, C)) ** 2)
        for Z, wk in zip(Zk, weights)
    )


def boys_molecular_mats(sc_mol):
    """MUTANT localisation operator: molecular <mu|r|nu> of the supercell's cell-0 AOs (L=0 only).
    Not translation-covariant: it sees the supercell boundary."""
    r = sc_mol.intor("int1e_r_cart")
    return [(r[k], 1.0) for k in range(3)]


def min_image(dvec, a_sc, raw=False):
    """|d| under the minimum-image convention (general cell: reduce, then scan the 27 neighbours).
    raw=True is the MUTANT (plain Cartesian distance)."""
    d = np.asarray(dvec, float)
    if raw:
        return np.linalg.norm(d, axis=-1)
    f = d @ np.linalg.inv(a_sc)
    f -= np.rint(f)
    best = None
    for n in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).reshape(3, -1).T:
        r = np.linalg.norm((f + n) @ a_sc, axis=-1)
        best = r if best is None else np.minimum(best, r)
    return best


# ------------------------------------------------------------------ periodic VV-HV virtuals
def vvhv_periodic(data, scf, loc_mats, Zk, weights, atom_of_ao, drop_hv=0):
    """VV-HV orthogonal localised virtuals ([WSHG23] Sec 2.1, rig deviations as in the molecular
    proto), with every S / position quantity replaced by its periodic counterpart:
    lattice-summed S and minimal-basis cross overlap, Berghold localisation of the valence
    virtuals, Resta spreads for the hard-virtual weights.  drop_hv: MUTANT, drop that many HVs."""
    S, F, Sx = data["S"], scf["F"], data["Sx"]
    nao = S.shape[0]
    Co = scf["C"][:, : scf["nocc"]]
    no = Co.shape[1]
    T = np.linalg.solve(S, Sx)
    Q = np.eye(nao) - Co @ (Co.T @ S)
    n_l = Sx.shape[1] - no
    C_L = alp.canonical_orth(Q @ T, S, n_l) if n_l > 0 else np.zeros((nao, 0))
    if n_l > 1:
        C_L, _ = jacobi_localize(C_L, loc_mats)
    n_h = nao - no - n_l
    C_H = np.zeros((nao, 0))
    hv_atom = np.zeros(0, int)
    if n_h > 0:
        C_E = np.hstack([Co, C_L])
        X = np.eye(nao) - C_E @ (C_E.T @ S)
        nrm2 = np.einsum("mi,mn,ni->i", X, S, X)
        keepable = nrm2 > 1e-8
        Xn = X[:, keepable] / np.sqrt(nrm2[keepable])
        parents = np.nonzero(keepable)[0]
        spread = np.maximum(resta_spreads(Xn, Zk, weights), 1e-6)
        wv = 1.0 / spread
        ov = Xn.T @ S @ Xn
        sel = np.array(
            alp.pivoted_cholesky_order(
                ov * np.outer(wv, wv) / np.outer(wv, wv).max(), n_h
            )
        )
        C_H = alp.lowdin(Xn[:, sel], S)
        hv_atom = atom_of_ao[parents[sel]]
        Fh = C_H.T @ F @ C_H
        for A in np.unique(hv_atom):
            idx = np.nonzero(hv_atom == A)[0]
            _, u = np.linalg.eigh(Fh[np.ix_(idx, idx)])
            C_H[:, idx] = C_H[:, idx] @ u
        if drop_hv:
            C_H, hv_atom = C_H[:, :-drop_hv], hv_atom[:-drop_hv]
    return np.hstack([C_L, C_H]), n_l, n_h


def check_construction(S, C_vloc, C_vcan):
    o = C_vloc.T @ S @ C_vloc
    U = C_vcan.T @ S @ C_vloc
    return float(abs(o - np.eye(len(o))).max()), float(
        max(
            abs(U @ U.T - np.eye(U.shape[0])).max(),
            abs(U.T @ U - np.eye(U.shape[1])).max(),
        )
    )


# -------------------------------------------------------------------------------- driver
def prepare(scell, data, scf, loc="berghold", drop_hv=0, seed=None):
    """Localised occupieds + VV-HV virtuals + (ia|jb) in the localised basis (global periodic RI)."""
    t0 = time.time()
    a_sc = scell.sc.a
    wts = resta_weights(a_sc)
    mats = (
        resta_mats(data["Zk"], wts)
        if loc == "berghold"
        else boys_molecular_mats(scell.sc.mol)
    )
    nocc = scf["nocc"]
    Co, loc_info = jacobi_localize(scf["C"][:, :nocc], mats, seed=seed)
    nao_p = scell.prim.mol.nao
    ao_atom_p = np.array([lbl[0] for lbl in scell.prim.mol.ao_labels(fmt=None)])
    natm_p = scell.prim.mol.natm
    atom_of_ao = np.concatenate([ao_atom_p + c * natm_p for c in range(scell.R)])
    Cv, n_l, n_h = vvhv_periodic(
        data, scf, mats, data["Zk"], wts, atom_of_ao, drop_hv=drop_hv
    )
    dev_orth, dev_span = check_construction(data["S"], Cv, scf["C"][:, nocc:])
    F = scf["F"]
    Foo, Fvv = Co.T @ F @ Co, Cv.T @ F @ Cv
    Bia = np.einsum("Pmn,mi,na->Pia", data["B"], Co, Cv, optimize=True)
    no, nv = Co.shape[1], Cv.shape[1]
    J = (Bia.reshape(len(Bia), -1).T @ Bia.reshape(len(Bia), -1)).reshape(
        no, nv, no, nv
    )
    cen = resta_centroids(Co, data["Zk"], a_sc)
    return dict(
        Co=Co,
        Cv=Cv,
        Foo=Foo,
        Fvv=Fvv,
        J=J,
        cen=cen,
        spreads=resta_spreads(Co, data["Zk"], wts),
        vcen=resta_centroids(Cv, data["Zk"], a_sc),
        n_l=n_l,
        n_h=n_h,
        dev_orth=dev_orth,
        dev_span=dev_span,
        loc=loc_info,
        a_sc=a_sc,
        t_prep=time.time() - t0,
    )


def pair_distances(prep, raw=False):
    cen = prep["cen"]
    return min_image(cen[:, None, :] - cen[None, :, :], prep["a_sc"], raw=raw)


def domain_fit_J(prep, data, fit_radius, raw=False, lindep=1e-10):
    """Per-pair domain-local fit in the PERIODIC metric: aux domain D_ij = aux functions within
    fit_radius (minimum image) of either occupied centroid; J_ij = A_i,D J2_DD^+ A_j,D with the
    same eig/lindep pseudo-inverse as the global B.  raw=True: MUTANT non-minimum-image distances."""
    Co, Cv = prep["Co"], prep["Cv"]
    A = np.einsum("mnP,mi,na->iaP", data["J3"], Co, Cv, optimize=True)
    J2 = data["J2"]
    no, nv, _ = A.shape
    dist = min_image(
        prep["cen"][:, None, :] - data["aux_xyz"][None, :, :], prep["a_sc"], raw=raw
    )
    inr = dist <= fit_radius
    J = np.zeros((no, nv, no, nv))
    sizes = []
    for i in range(no):
        for j in range(i, no):
            d = np.nonzero(inr[i] | inr[j])[0]
            sizes.append(len(d))
            s, U = np.linalg.eigh(J2[np.ix_(d, d)])
            k = s > lindep
            Wt = U[:, k] / np.sqrt(s[k])
            blk = (A[i][:, d] @ Wt) @ (A[j][:, d] @ Wt).T
            J[i, :, j, :] = blk
            J[j, :, i, :] = blk.T
    return J, dict(
        dom_mean=float(np.mean(sizes)), dom_max=int(np.max(sizes)), naux=J2.shape[0]
    )


def solve(
    prep,
    eps,
    pair_cut=None,
    raw=False,
    J=None,
    rtol=1e-11,
    solver="dense",
    gate="J",
    J_unif=None,
    pair_ecut=None,
    addback=None,
):
    """Amplitude-threshold LMP2 at threshold eps (Eq-8 mask, swap-closed), optionally AND-ed with a
    minimum-image occupied pair cutoff.  Returns energy, pair energies, mask statistics.

    Iteration 5b options (uniform q=0-head coupling, see `uniform_coupling`):
      gate="J"       : threshold the integrals that are solved with (Iteration 5 behaviour).
      gate="J-unif"  : threshold J - J_unif (J_unif required), but SOLVE with J.  Far pairs whose
                       only coupling is the uniform field are then dropped at any N.
      pair_ecut      : additionally drop occupied pairs whose semicanonical pair-energy estimate from
                       the GATING tensor is below pair_ecut (energy-based pair screen, eps-independent).
      addback        : (nocc, nocc) analytic uniform-field pair energies (`uniform_pair_energies`);
                       their sum over ENTIRELY dropped off-diagonal pairs is added to e (reported
                       separately as e_addback).  None = no add-back.
    The eps=0 / no-cut limit keeps every element, so every option reduces to the anchor exactly."""
    J = prep["J"] if J is None else J
    no = J.shape[0]
    if gate == "J":
        Jg = J
    elif gate == "J-unif":
        if J_unif is None:
            raise ValueError("gate='J-unif' needs J_unif (uniform_coupling)")
        Jg = J - J_unif
    else:
        raise ValueError(f"gate must be 'J' or 'J-unif', got {gate!r}")
    if eps == 0.0:
        mask = np.ones(J.shape, bool)
    else:
        mask = (np.abs(Jg) > eps) | (np.abs(Jg.transpose(0, 3, 2, 1)) > eps)
    if pair_cut is not None:
        keep = pair_distances(prep, raw=raw) <= pair_cut
        mask &= keep[:, None, :, None]
    if pair_ecut is not None:
        keep = np.abs(pair_energy_estimate(prep, Jg)) >= pair_ecut
        np.fill_diagonal(keep, True)
        mask &= keep[:, None, :, None]
    if (
        solver == "ragged"
    ):  # per-pair domain blocks (cost tracks retained work); same fixed-mask problem
        t, niter, relres, _ = alp.solve_masked_mp2_ragged(
            J, prep["Foo"], prep["Fvv"], mask, rtol=rtol
        )
    else:
        t, niter, relres = alp.solve_masked_mp2(
            J, prep["Foo"], prep["Fvv"], mask, rtol=rtol
        )
    e = alp.mp2_energy(t, J)
    pairs = mask.any(axis=(1, 3))
    e_add = 0.0
    if addback is not None:
        dropped = ~pairs
        np.fill_diagonal(dropped, False)
        e_add = float(addback[dropped].sum())
    return dict(
        e=e + e_add,
        e_solve=e,
        e_addback=e_add,
        epair=alp.pair_energies(t, J),
        keep=float(mask.mean()),
        pairs_kept=int(pairs.sum()),
        pair_frac=float(pairs.mean()),
        partners=pairs.sum(1),
        niter=niter,
        relres=relres,
        mask=mask,
    )


# -------------------------------------------------------------- translation equivalence
def translation_map(scell, C, S, axis):
    """Translate every column of C by one primitive vector along `axis` (an exact AO permutation
    at Gamma) and match it to the best column of C.  Returns (perm, dev) with dev_i =
    1 - max_j |<C_j|S|T C_i>| (0 for exactly equivalent orbitals)."""
    nao_p = scell.prim.mol.nao
    p = scell.ao_shift_perm(axis, nao_p)
    TC = np.zeros_like(C)
    TC[p] = C
    O = np.abs(C.T @ S @ TC)
    perm = O.argmax(0)
    dev = 1 - O[perm, np.arange(C.shape[1])]
    return perm, dev


def translation_equivalence(scell, data, prep, epair):
    """Max deviations over all periodic axes: LMO overlap, VV-HV overlap, pair energies
    |e_ij - e_T(i)T(j)|, and whether T maps LMOs bijectively."""
    out = dict(occ_dev=0.0, vir_dev=0.0, pair_dev=0.0, bijective=True, spread_dev=0.0)
    for ax in range(3):
        if scell.n[ax] == 1:
            continue
        po, do = translation_map(scell, prep["Co"], data["S"], ax)
        _, dv = translation_map(scell, prep["Cv"], data["S"], ax)
        out["occ_dev"] = max(out["occ_dev"], float(do.max()))
        out["vir_dev"] = max(out["vir_dev"], float(dv.max()))
        out["bijective"] &= len(set(po.tolist())) == len(po)
        out["pair_dev"] = max(
            out["pair_dev"], float(abs(epair[np.ix_(po, po)] - epair).max())
        )
        out["spread_dev"] = max(
            out["spread_dev"], float(abs(prep["spreads"][po] - prep["spreads"]).max())
        )
    return out


# ----------------------------------------------------------- uniform-field (G->0) predictor
def transition_dipoles_z(prep, data, axis=2):
    """mu_ia along supercell axis from the Resta matrix at the smallest b: z_ia exp(-i arg z_ii)
    ~ i b <i|(r - r_i)|a>  (O((b sigma)^2) accurate; b = 2pi/L_axis)."""
    Z = data["Zk"][axis]
    b = np.linalg.norm(data["Gk"][axis])
    zi = np.einsum("mi,mn,ni->i", prep["Co"], Z, prep["Co"])
    zia = prep["Co"].T @ Z @ prep["Cv"]
    return (zia * np.exp(-1j * np.angle(zi))[:, None]).imag / b


# ------------------------------------------ Iteration 5b: the uniform (q=0 head) coupling
# A Gamma supercell is a k-mesh on the primitive cell; for a 1 x 1 x N needle the momentum transfers q
# lie on the z axis.  For q != 0 the G=0 ("head") term of (ia|jb) is (4pi/Omega_sc) (q.mu_ia)(q.mu_jb)/q^2
# = (4pi/Omega_sc) mu_ia,z mu_jb,z, constant in q; at q = 0 it is dropped (G=0-dropped kernel).  Back in
# real space: sum_{q != 0} e^{iqd} = N delta_{d0} - 1, so every pair carries -(4pi/Omega_sc) mu mu: the
# uniform coupling is exactly MINUS the omitted q=0 head.  Adding it back (J' = J - J_unif) makes far-pair
# couplings vanish (exponentially) and gives the self pair its N-independent (4pi/Omega_prim) mu mu.
def resta_z_at(scell, data, axis, mult):
    """Resta matrix <mu| exp(+i m b_axis.r) |nu> (lattice-summed) at the supercell reciprocal vector
    m b_axis, by the same residue fold as build_supercell's Zk (m = 1 reproduces data['Zk'][axis])."""
    from pbc_supercell import pair_ft_residues

    nao_p, R = scell.prim.mol.nao, scell.R
    G = (mult * data["Gk"][axis])[None, :]
    S1diag = data["S"][:nao_p].reshape(nao_p, R, nao_p).sum(1).diagonal()
    Q0 = pair_ft_residues(scell, np.zeros((1, 3)))[..., 0].real.sum(0).diagonal()
    nrm = np.sqrt(S1diag / Q0)
    Q = pair_ft_residues(scell, G)[..., 0] * np.outer(nrm, nrm)[None]
    ph = np.exp(-1j * scell.t @ G[0])
    return (
        np.conj(ph[:, None, None, None] * Q[scell.D])
        .transpose(0, 2, 1, 3)
        .reshape(R * nao_p, R * nao_p)
    )


def _resta_dipole(Co, Cv, Z, b):
    zi = np.einsum("mi,mn,ni->i", Co, Z, Co)
    return ((Co.T @ Z @ Cv) * np.exp(-1j * np.angle(zi))[:, None]).imag / b


def transition_dipoles_richardson(scell, data, prep, axis=2, Z2=None):
    """mu_ia along `axis`: Resta estimate f(b) = Im(z_ia e^{-i arg z_ii})/b = mu + c b^2 + O(b^4) at b and 2b,
    Richardson (4 f(b) - f(2b))/3 -> O(b^4).  (transition_dipoles_z is f(b) alone, O(b^2).)"""
    b = np.linalg.norm(data["Gk"][axis])
    Z2 = resta_z_at(scell, data, axis, 2) if Z2 is None else Z2
    f1 = _resta_dipole(prep["Co"], prep["Cv"], data["Zk"][axis], b)
    f2 = _resta_dipole(prep["Co"], prep["Cv"], Z2, 2 * b)
    return (4 * f1 - f2) / 3, dict(f1=f1, f2=f2)


def uniform_coupling(prep, mu, omega=None):
    """J_unif[i,a,j,b] = -(4pi/Omega_sc) mu_ia mu_jb: the needle (1 x 1 x N) form, depolarisation tensor
    z z.  A general supercell has -(4pi/Omega) mu . W . mu with W the constant term of its periodic dipole
    tensor (cubic: I/3, as in Iteration 3's c3); only the needle is implemented/measured here."""
    omega = abs(np.linalg.det(prep["a_sc"])) if omega is None else omega
    return -(4 * np.pi / omega) * np.einsum("ia,jb->iajb", mu, mu)


def uniform_pair_energies(prep, J_unif):
    """Analytic pair energies of the uniform coupling alone (direct term, amplitudes from the FULL
    non-canonical Fvv by Sylvester in its eigenbasis, diagonal Foo): the add-back for dropped pairs.
    e_ij = -2 sum_ab P_ij,ab^2 / (l_a + l_b - f_i - f_j), P = Vv^T J_unif[i,:,j,:] Vv.  Valid for FAR
    pairs (exchange-type (ib|ja) and the near-field J neglected: that is what makes them droppable)."""
    fo = np.diag(prep["Foo"])
    lv, Vv = np.linalg.eigh(prep["Fvv"])
    P = np.einsum("iajb,ac,bd->icjd", J_unif, Vv, Vv, optimize=True)
    den = (
        lv[None, :, None, None]
        + lv[None, None, None, :]
        - fo[:, None, None, None]
        - fo[None, None, :, None]
    )
    return -2 * np.einsum("iajb,iajb->ij", P**2, 1.0 / den)


def pair_energy_estimate(prep, Jg):
    """Semicanonical MP2 pair-energy estimate from a gating tensor Jg (Fvv eigenbasis, diagonal Foo,
    direct + exchange): the energy-based pair screen (ORCA TCutPairs / MRCC eps_w style, but from the
    actual RI integrals rather than a multipole model)."""
    fo = np.diag(prep["Foo"])
    lv, Vv = np.linalg.eigh(prep["Fvv"])
    P = np.einsum("iajb,ac,bd->icjd", Jg, Vv, Vv, optimize=True)
    den = (
        lv[None, :, None, None]
        + lv[None, None, None, :]
        - fo[:, None, None, None]
        - fo[None, None, :, None]
    )
    return -np.einsum("iajb,iajb->ij", P * (2 * P - P.transpose(0, 3, 2, 1)), 1.0 / den)


def canonical_mp2_from_local(prep, scf, S, J):
    """Closed-form canonical MP2 of a LOCAL-basis integral tensor J (rotated to the canonical orbitals of
    scf): the independent reference for any modified J (e.g. J - J_unif).  J = prep['J'] reproduces
    canonical_mp2 (unitary invariance of (ia|jb))."""
    no = scf["nocc"]
    Uo = prep["Co"].T @ S @ scf["C"][:, :no]
    Uv = prep["Cv"].T @ S @ scf["C"][:, no:]
    Jc = np.einsum("kcld,ki,ca,lj,db->iajb", J, Uo, Uv, Uo, Uv, optimize=True)
    eo, ev = scf["eps"][:no], scf["eps"][no:]
    d = (
        eo[:, None, None, None]
        - ev[None, :, None, None]
        + eo[None, None, :, None]
        - ev[None, None, None, :]
    )
    return float(np.einsum("iajb,iajb->", Jc / d, 2 * Jc - Jc.transpose(0, 3, 2, 1)))


def fock_head_correction(scell, data, prep, mu, axis=2, Z2=None, omega=None):
    """Restore the q=0 head in the EXCHANGE part of the Fock blocks too (the Iteration-3 d eps terms, needle
    form), so the whole O(1) Gamma finite-size term is removed, not only its ERI part:
      Fvv'_ab = Fvv_ab - (4pi/Omega) sum_k mu_ka mu_kb             (neutral ov exchange densities)
      Foo'_ii = Foo_ii + (4pi/Omega) [sigma_i^2 - sum_{k!=i} mu_ik^2]  (charged rho_ii: its q^2 term;
                                                                    neutral rho_ik, k != i)
    Foo off-diagonals are NOT corrected (odd-in-q / origin-dependent terms; small for LMOs).  sigma^2 and
    the occ-occ dipoles by the same (b, 2b) Richardson as mu.  Returns (Foo', Fvv')."""
    omega = abs(np.linalg.det(prep["a_sc"])) if omega is None else omega
    c = 4 * np.pi / omega
    b = np.linalg.norm(data["Gk"][axis])
    Z1 = data["Zk"][axis]
    Z2 = resta_z_at(scell, data, axis, 2) if Z2 is None else Z2
    Co = prep["Co"]
    moo = (4 * _resta_dipole(Co, Co, Z1, b) - _resta_dipole(Co, Co, Z2, 2 * b)) / 3
    np.fill_diagonal(moo, 0.0)

    def s2(Z, bb):
        return 2 * (1 - np.abs(np.einsum("mi,mn,ni->i", Co, Z, Co))) / bb**2

    sig2 = (4 * s2(Z1, b) - s2(Z2, 2 * b)) / 3
    Foo = prep["Foo"] + np.diag(c * (sig2 - (moo**2).sum(1)))
    Fvv = prep["Fvv"] - c * mu.T @ mu
    return Foo, Fvv, dict(sigma2=sig2, mu_oo=moo)


def mp2_local_closed_form(J, Foo, Fvv):
    """Closed-form MP2 of a local-basis J with (possibly modified) Foo/Fvv: rotate J to their eigenbases."""
    fo, Uo = np.linalg.eigh(Foo)
    fv, Uv = np.linalg.eigh(Fvv)
    Jc = np.einsum("kcld,ki,ca,lj,db->iajb", J, Uo, Uv, Uo, Uv, optimize=True)
    d = (
        fo[:, None, None, None]
        - fv[None, :, None, None]
        + fo[None, None, :, None]
        - fv[None, None, None, :]
    )
    return float(np.einsum("iajb,iajb->", Jc / d, 2 * Jc - Jc.transpose(0, 3, 2, 1)))


__all__ = [
    "gamma_scf",
    "canonical_mp2",
    "prepare",
    "solve",
    "domain_fit_J",
    "jacobi_localize",
    "resta_weights",
    "resta_mats",
    "resta_centroids",
    "resta_spreads",
    "min_image",
    "pair_distances",
    "translation_map",
    "translation_equivalence",
    "transition_dipoles_z",
    "boys_molecular_mats",
    "resta_z_at",
    "transition_dipoles_richardson",
    "uniform_coupling",
    "uniform_pair_energies",
    "pair_energy_estimate",
    "canonical_mp2_from_local",
    "fock_head_correction",
    "mp2_local_closed_form",
]
