"""Iteration 26: Gamma RS-GDF FORCES and STRESS for the RANGE-SPLIT fit of Iteration 23 (pbc_gdf_split.py, grouping A).

Nothing in pbc_gdf_split / pbc_grad_gdf / pbc_stress is modified: Y and Wm come from pbc_grad_gdf.fit_densities
(unchanged), the 1e / V_LR / Ewald / Madelung / XC pieces from pbc_grad_gdf / pbc_stress (unchanged); only the
fitted-integral derivatives dJ3, dJ2 are new, because the split changes WHICH integrals J3 and J2 are made of.

SPLIT ENERGY (grouping A; chi = chi^c + chi^s, X = X^c + X^s by primitive exponent, pbc_gdf_split.Criterion)
    J3[mn,P] = K[mn,P]                                                       (SR erfc, kept calls, real space)
             + (1/Om) sum_{G!=0} { v_LR Re[conj(P_mn) X_P] + v_SR Re[conj(P_mn) X^s_P] + v_SR Re[conj(P^ss_mn) X^c_P] }
             - c0 (S - S_ss)_mn q^c_P
      K = sum_{L,T} (m_0 n^c_L | P^c_T)_erfc + sum_{L,T} (m^c_0 n^s_L | P^c_T)_erfc      (the two "twocall" calls)
    J2[P,Q]  = sum_T (P^c_0 | Q^c_T)_erfc + (1/Om) sum_{G!=0} { v Re[conj(X_P) X_Q] - v_SR Re[conj(X^c_P) X^c_Q] }
             - c0 q^c_P q^c_Q
    v = 4pi/G^2 = v_LR + v_SR,  v_LR = v e^{-G^2/4w^2},  q^c = int X^c (compact aux piece charge),
    S, S_ss = lattice overlaps (range rcut_pair) of the full / smooth-piece orbitals, P^ss = pair FT of the smooth
    pieces (same calibration pn as P), X^s / X^c = piece FTs with the FULL aux calibration.
The per-aux-column form of the Rust port (weight v on X^s, v_LR on X^c) is the same sum: v_LR X + v_SR X^s.

FORCE (dE_2e = sum Y dJ3 + sum Wm dJ2, Y / Wm of Iteration 18; everything of atom A moves in every image)
  SR K    bra -int3c2e_ip1 on the two kept calls, ket either x2 bra ('sym': K is symmetric in (m,n) as a function --
          cc + cs + sc -- so the relabelling of Iteration 18 holds for the SUM of the two calls) or explicit
          ip1 + ip2 (3-centre translation invariance per call); aux -int3c2e_ip2 on the same calls
  G       bra: pbc_grad.pair_ft_deriv Q (full pairs) with weight v_LR X + v_SR X^s, x2;
               Q^ss (pair_ft_deriv on the smooth pieces) with weight v_SR X^c, x2            <- NEW
          aux: -iG on X (v_LR, full P), X^s (v_SR, full P), X^c (v_SR, P^ss)                 <- NEW columns
  G = 0   M_g0 = -c0 sum_P Y[mn,P] q^c_P contracted with d(S - S_ss): the smooth-piece overlap derivative is NEW
  SR J2   (P^c|Q^c) only: -int2c2e_ip1 on the compact-aux images
  G J2    2 v_LR Re[(iG) conj(X_P) (Wm X)_P] + v_SR d/dC Re[X^s+ Wm X + X^c+ Wm X^s] (the non-cancelling form of
          v_SR [(X,X) - (X^c,X^c)], as rsgdf/split.rs builds J2); Wm SYMMETRISED first (FINDINGS It. 26 (e)) <- NEW
STRESS (sigma = dE/d eps / Om, pbc_stress conventions, fixed index sets)
  SR K    2 sum Y (-ip1)(A_m - C_P - T) over the two kept calls (bra-relabelling, as Iteration 19)
  SR J2   compact pieces only, Iteration 19's formula
  G J3    -delta E_G + (1/Om) sum [dv_LR phi_LR + dv_SR phi_SR] + (1/Om) sum v [dP, dX, dX^s, dP^ss, dX^c terms]
          dv_SR/dG^2 = dv/dG^2 - dv_LR/dG^2 = -4pi/G^4 + v_LR (1/G^2 + 1/4w^2)                   <- NEW
          pair_ft_strain on the smooth pieces (P^ss), aux_ft_strain on the pieces (FULL calibration)  <- NEW
  G J2    -delta E_G2 + (1/Om) sum [dv_LR phi_A + dv_SR phi_M + 2 v_LR Re(dX^+ Wm X) + v_SR dphi_M],
          phi_M = Re[X^s+ Wm X + X^c+ Wm X^s]
  G = 0   J3: -c0 d(S - S_ss):(Y q^c)  +  c0 sum (Y q^c)(S - S_ss) delta;   J2: c0 q^c+ Wm q^c delta
          (the smooth-piece overlap virial is NEW)

_MUTANT (module global):
  'no_dSss'        G = 0 force/stress: contract M_g0 with dS only (omit the smooth-piece overlap derivative / virial)
  'vol_full_S'     stress only: the J3 G = 0 VOLUME term with S instead of S - S_ss (q^c kept)
  'no_Sss'         stress only: S_ss forgotten in BOTH G = 0 stress terms (virial and volume; q^c kept)
  'g0_full'        G = 0 with the FULL inputs (dS, q) -- the unsplit Iteration 18/19 term
  'no_sr_kstrain'  stress: drop dv_SR/d eps (the SR kernel-weight strain of the moved G-space blocks)
  'no_ss_pair'     drop the NEW (P^ss | X^c) derivative term (bra and aux; stress: its dP^ss and dX^c)
  'moved_lr'       the moved blocks differentiated with v_LR instead of v_SR (kernel mix-up)
  'asym_wm'        NOT a defect: Wm used without symmetrisation (pbc_grad_gdf does not symmetrise; roundoff probe)
  'cancel_metric'  NOT a defect: forces with the cancelling metric form v_SR [(X,X) - (X^c,X^c)] (roundoff probe)
"""

from __future__ import annotations

import copy

import numpy as np
from pyscf import gto

import pbc_grad as PGd
import pbc_grad_gdf as GG
from pbc_gamma import cart_comps, hermite_E, pair_ft
from pbc_gdf import _images, _raw_ft, aux_ft
from pbc_gdf_split import Criterion, build_gdf_split, piece_mol

_MUTANT = None
EYE = np.eye(3)


# ------------------------------------------------------------------------------------------------ pieces
class Pieces:
    """Every piece object of one (cell, auxmol, criterion); geometry enters only through cell / auxmol."""

    def __init__(self, cell, auxmol, crit):
        self.crit = crit
        self.orb_s = crit.orb_smooth
        self.orb_c = lambda e: not crit.orb_smooth(e)
        self.aux_s = crit.aux_smooth
        self.aux_c = lambda e: not crit.aux_smooth(e)
        mol = cell.mol
        self.mol0_c = piece_mol(mol, self.orb_c)
        self.cell_s = copy.copy(cell)
        self.cell_s.mol = piece_mol(mol, self.orb_s)
        self.aux_cm = piece_mol(auxmol, self.aux_c)
        self.aux_sm = piece_mol(auxmol, self.aux_s)
        _, so = _raw_ft(auxmol, np.zeros((1, 3)))
        self.anrm = np.sqrt(
            np.diag(auxmol.intor("int1e_ovlp_cart")) / so
        )  # FULL calibration (as the split build)
        self.q = aux_ft(auxmol, np.zeros((1, 3)))[:, 0].real
        self.q_c = self.piece_ft(self.aux_cm, np.zeros((1, 3)))[:, 0].real

    def piece_ft(self, pm, G):
        return _raw_ft(pm, G)[0] * self.anrm[:, None]

    def calls(self, cell, sm):
        """The two kept SR calls: (bra mol at cell 0, ket supermol piece)."""
        return [
            (cell.mol, piece_mol(sm, self.orb_c)),
            (self.mol0_c, piece_mol(sm, self.orb_s)),
        ]


def split_gdf(
    cell, auxmol, w=1.0, lam=1.0, crit=None, prec=1e-13, spherical=True, lindep=1e-10
):
    """pbc_gdf_split.build_gdf_split, grouping A, split metric; + the ranges and the criterion."""
    crit = crit or Criterion(w, lam)
    gd = build_gdf_split(
        cell,
        None,
        w=w,
        prec=prec,
        crit=crit,
        split_metric=True,
        spherical=spherical,
        lindep=lindep,
        auxmol=auxmol,
        g0="A",
    )
    gd["rng"] = GG.gdf_ranges(cell, auxmol, w, prec)
    gd["crit"], gd["w"] = crit, w
    return gd


def _kernels(G, w):
    G2 = np.einsum("gi,gi->g", G, G)
    v = 4 * np.pi / G2
    vlr = v * np.exp(-G2 / (4 * w * w))
    vsr = v - vlr
    dv = -v / G2  # d/dG^2
    dvlr = -vlr * (1 / G2 + 1 / (4 * w * w))
    return G2, v, vlr, vsr, dv, dvlr, dv - dvlr


# ------------------------------------------------------------------------------------ derivative integrals
def deriv_ints_split(cell, ints, gd, auxmol, w, rcut_1e=22.0, chunk_elems=4e6):
    mol = cell.mol
    nao, nb0 = mol.nao, mol.nbas
    naux = auxmol.nao
    rng, crit = gd["rng"], gd["crit"]
    pc = Pieces(cell, auxmol, crit)
    L1 = cell.translations(rng["rcut_pair"])
    nL = len(L1)
    sm = cell.supermol(L1)
    T3 = cell.translations(rng["rcut_aux3"])
    d3b = np.zeros((3, nao, nao, naux))
    d3a = np.zeros((3, nao, nao, naux))
    for b0, ket in pc.calls(cell, sm):
        chunk = max(1, int(chunk_elems // (nao * nao * nL * naux)))
        for t0 in range(0, len(T3), chunk):
            Tc = T3[t0 : t0 + chunk]
            ai = piece_mol(_images(auxmol, Tc), pc.aux_c)
            big = gto.conc_mol(gto.conc_mol(b0, ket), ai)
            sl = (0, nb0, nb0, nb0 + ket.nbas, nb0 + ket.nbas, big.nbas)
            with big.with_range_coulomb(-w):
                d3b += (
                    big.intor("int3c2e_ip1_cart", comp=3, shls_slice=sl)
                    .reshape(3, nao, nL, nao, len(Tc), naux)
                    .sum(axis=(2, 4))
                )
                d3a += (
                    big.intor("int3c2e_ip2_cart", comp=3, shls_slice=sl)
                    .reshape(3, nao, nL, nao, len(Tc), naux)
                    .sum(axis=(2, 4))
                )
    T2 = cell.translations(rng["rcut_aux2"])
    img = piece_mol(_images(auxmol, T2), pc.aux_c)
    with img.with_range_coulomb(-w):
        d2 = img.intor(
            "int2c2e_ip1_cart", comp=3, shls_slice=(0, auxmol.nbas, 0, img.nbas)
        )
    d2 = d2.reshape(3, naux, len(T2), naux).sum(2)
    G = cell.gvectors(rng["gcut"])
    G = G[np.einsum("gi,gi->g", G, G) > 1e-12]
    _, v, vlr, vsr, _, _, _ = _kernels(G, w)
    X = aux_ft(auxmol, G)
    Xs, Xc = pc.piece_ft(pc.aux_sm, G), pc.piece_ft(pc.aux_cm, G)
    P0 = pair_ft(cell, np.zeros((1, 3)))[..., 0].real
    nn = np.sqrt(np.diag(gd["S"]) / np.diag(P0))
    nn = nn[:, None] * nn[None, :]
    P, Q = PGd.pair_ft_deriv(cell, G)
    Pss, Qss = PGd.pair_ft_deriv(pc.cell_s, G)
    P, Q, Pss, Qss = (
        P * nn[:, :, None],
        Q * nn[None, :, :, None],
        Pss * nn[:, :, None],
        Qss * nn[None, :, :, None],
    )
    S = ints["S"]
    _, Qh = PGd.pair_ft_deriv(cell, ints["G"])
    nrm = np.sqrt(np.diag(S) / np.diag(P0))
    Qh = Qh * (nrm[:, None] * nrm[None, :])[None, :, :, None]
    XS = PGd._latsum_ip(cell, "int1e_ipovlp_cart", rcut_1e)
    XT = PGd._latsum_ip(cell, "int1e_ipkin_cart", rcut_1e)
    XSg = PGd._latsum_ip(cell, "int1e_ipovlp_cart", rng["rcut_pair"])
    sm_s = piece_mol(sm, pc.orb_s)
    XSg_ss = (
        sm_s.intor("int1e_ipovlp_cart", comp=3, shls_slice=(0, nb0, 0, sm_s.nbas))
        .reshape(3, nao, nL, nao)
        .sum(2)
    )
    aux_owner = np.zeros(naux, int)
    for k, (_, _, p0, p1) in enumerate(auxmol.aoslice_by_atom()):
        aux_owner[p0:p1] = k
    vol = cell.vol
    return dict(
        d3b=d3b,
        d3a=d3a,
        d2=d2,
        G=G,
        v=v / vol,
        vlr=vlr / vol,
        vsr=vsr / vol,
        X=X,
        Xs=Xs,
        Xc=Xc,
        P=P,
        Q=Q,
        Pss=Pss,
        Qss=Qss,
        Qh=Qh,
        XS=XS,
        XT=XT,
        XSg=XSg,
        XSg_ss=XSg_ss,
        q=pc.q,
        q_c=pc.q_c,
        aux_owner=aux_owner,
        natm_aux=auxmol.natm,
    )


# -------------------------------------------------------------------------------------------- gradient
def gamma_gdf_split_grad(
    cell,
    ints,
    gd,
    auxmol,
    Da,
    Db,
    Fa,
    Fb,
    alpha,
    w=1.0,
    lindep=1e-10,
    spherical=True,
    metric="dk",
    aux_jac=None,
    ket="sym",
    cache=None,
):
    """dE/dR (natm, 3), parts, diag for the split RS-GDF energy at a converged (D_s, F_s) (HF / hybrid 2e only;
    XC pieces as Iteration 18 if needed)."""
    if cache is None:
        cache = {}
    if not cache:
        cache.update(deriv_ints_split(cell, ints, gd, auxmol, w))
    k = cache
    S, Gh, vM = ints["S"], ints["G"], ints["madelung"]
    mol = cell.mol
    natm = mol.natm
    aoat = PGd.ao_atom(mol)
    D = Da + Db
    spins = ((Da, Fa), (Db, Fb))
    parts = {}
    c2s = GG.c2s_of(auxmol, spherical)
    c0 = np.pi / (w * w * cell.vol)
    Y, Wm, diag = GG.fit_densities(gd["J2"], gd["J3"], [Da, Db], alpha, lindep, metric)
    Yc = np.einsum("mnk,pk->mnp", Y, c2s)
    Wc = c2s @ Wm @ c2s.T
    if (
        _MUTANT != "asym_wm"
    ):  # only Wm's symmetric part enters (dJ2 is symmetric); rsgdf/deriv.rs symmetrises too
        Wc = 0.5 * (Wc + Wc.T)

    # ---- 1e / Madelung / V_LR / Ewald: Iteration 18 unchanged
    W = sum(Ds @ Fs @ Ds for Ds, Fs in spins)
    M = -W - alpha * vM * sum(Ds @ S @ Ds for Ds, _ in spins)
    parts["S"] = PGd._fold(-2 * np.einsum("xmn,mn->mx", k["XS"], M), aoat, natm)
    parts["T"] = PGd._fold(-2 * np.einsum("xmn,mn->mx", k["XT"], D), aoat, natm)
    Ph = ints["P"]
    v = 4 * np.pi / np.einsum("gi,gi->g", Gh, Gh)
    SG_at = np.exp(-1j * Gh @ cell.R.T) * cell.Z
    SG = SG_at.sum(1)
    rho = np.einsum("mn,mng->g", D, Ph)
    rq = PGd._fold_g(2 * np.einsum("mn,xmng->mxg", D, k["Qh"]), aoat, natm)
    parts["Vlr_basis"] = -(np.einsum("g,axg->ax", v, np.conj(rq) * SG).real) / cell.vol
    parts["Vlr_nuc"] = (
        -(np.einsum("g,ga,gx->ax", v * np.conj(rho), SG_at * (-1j), Gh).real) / cell.vol
    )
    parts["nn"] = PGd.ewald_grad(cell, 1.0)

    # ---- G = 0 of J3: -c0 (S - S_ss) q^c  ->  M_g0 = -c0 Y q^c on d(S - S_ss)
    if _MUTANT == "g0_full":
        Mg0, dSe = -c0 * np.einsum("mnp,p->mn", Yc, k["q"]), k["XSg"]
    else:
        Mg0 = -c0 * np.einsum("mnp,p->mn", Yc, k["q_c"])
        dSe = k["XSg"] if _MUTANT == "no_dSss" else k["XSg"] - k["XSg_ss"]
    parts["J3_g0"] = PGd._fold(-2 * np.einsum("xmn,mn->mx", dSe, Mg0), aoat, natm)

    # ---- SR kept calls
    if ket == "sym":
        parts["J3_bra_sr"] = PGd._fold(
            -2 * np.einsum("xmnp,mnp->mx", k["d3b"], Yc), aoat, natm
        )
    else:
        gbra = -np.einsum("xmnp,mnp->mx", k["d3b"], Yc)
        gket = np.einsum("xmnp,mnp->nx", k["d3b"] + k["d3a"], Yc)
        parts["J3_bra_sr"] = PGd._fold(gbra + gket, aoat, natm)
    ga3_sr = -np.einsum("xmnp,mnp->px", k["d3a"], Yc)

    # ---- G space: full pairs (v_LR X + v_SR X^s) and smooth pairs (v_SR X^c)
    G, vlr, vsr, vfull = k["G"], k["vlr"], k["vsr"], k["v"]
    vmv = (
        vlr if _MUTANT == "moved_lr" else vsr
    )  # the kernel the moved blocks are differentiated with
    X, Xs, Xc, P, Q, Pss, Qss = (
        k["X"],
        k["Xs"],
        k["Xc"],
        k["P"],
        k["Q"],
        k["Pss"],
        k["Qss"],
    )
    wX = vlr * X + vmv * Xs
    wXc = vmv * Xc
    ss = 0.0 if _MUTANT == "no_ss_pair" else 1.0
    gb = (
        2 * np.einsum("xmng,mng->mx", np.conj(Q), np.einsum("mnp,pg->mng", Yc, wX)).real
    )
    gb += (
        ss
        * 2
        * np.einsum(
            "xmng,mng->mx", np.conj(Qss), np.einsum("mnp,pg->mng", Yc, wXc)
        ).real
    )
    parts["J3_bra_g"] = PGd._fold(gb, aoat, natm)
    PY = np.einsum("mnp,mng->pg", Yc, np.conj(P))
    PssY = np.einsum("mnp,mng->pg", Yc, np.conj(Pss))
    ga3_g = (
        np.einsum("pg,gx,pg->px", PY, -1j * G, wX).real
        + ss * np.einsum("pg,gx,pg->px", PssY, -1j * G, wXc).real
    )

    # ---- J2: kept SR (P^c|Q^c); G: v on (X, X) minus v_SR on (X^c, X^c)
    ga2_sr = -np.einsum("xpq,pq->px", k["d2"], Wc) + np.einsum(
        "xpq,pq->qx", k["d2"], Wc
    )

    # moved metric block v_SR [(X,X) - (X^c,X^c)] in the non-cancelling form v_SR [(X^s,X) + (X^c,X^s)] (the form
    # rsgdf/split.rs builds J2 with; the two agree to 2e-16 in sum F -- 'cancel_metric' probe).  What DOES matter is
    # Wm's symmetry: sum_R of this term is Re[iG X^H (Wm - Wm^T) X], and the v_SR weight at large G amplifies an
    # unsymmetrised Wm's roundoff to sum F 2.2e-11 on the H2 x3 supercell ('asym_wm'); symmetrised: 2.2e-16.
    def bil(
        v_, A1, A2
    ):  # d/dC_R of sum_G v Re[A1^H W A2]  (both factors centred at C_R)
        return (
            np.einsum("g,pg,gx,pg->px", v_, np.conj(A1), 1j * G, Wc @ A2)
            - np.einsum("g,pg,gx,pg->px", v_, Wc @ np.conj(A1), 1j * G, A2)
        ).real

    ga2_g = 2 * np.einsum("g,pg,gx,pg->px", vlr, np.conj(X), 1j * G, Wc @ X).real + (
        bil(vmv, Xs, X) + bil(vmv, Xc, Xs)
        if _MUTANT != "cancel_metric"
        else 2
        * (
            np.einsum("g,pg,gx,pg->px", vmv, np.conj(X), 1j * G, Wc @ X).real
            - np.einsum("g,pg,gx,pg->px", vmv, np.conj(Xc), 1j * G, Wc @ Xc).real
        )
    )
    jac = np.eye(natm) if aux_jac is None else np.asarray(aux_jac)

    def fold_aux(per_aux):
        per_center = np.zeros((k["natm_aux"], 3))
        np.add.at(per_center, k["aux_owner"], per_aux)
        return jac.T @ per_center

    parts["J3_aux_sr"] = fold_aux(ga3_sr)
    parts["J3_aux_g"] = fold_aux(ga3_g)
    parts["J2_sr"] = fold_aux(ga2_sr)
    parts["J2_g"] = fold_aux(ga2_g)
    return sum(parts.values()), parts, diag


# ================================================================================================ STRESS
def aux_ft_strain_n(pm, G, nrm):
    """pbc_stress.aux_ft_strain for a PIECE mol with an externally supplied (full) calibration nrm."""
    G = np.asarray(G, float).reshape(-1, 3)
    G2 = np.einsum("gi,gi->g", G, G)
    X = np.zeros((pm.nao, len(G)), complex)
    dX = np.zeros((3, 3, pm.nao, len(G)), complex)
    off = 0
    for ib in range(pm.nbas):
        l, C = pm.bas_angular(ib), pm.atom_coord(pm.bas_atom(ib))
        exps, coef = pm.bas_exp(ib), pm._libcint_ctr_coeff(ib)
        comps = cart_comps(l)
        powg = [
            np.stack([(-1j * G[:, d]) ** t for t in range(l + 1)]) for d in range(3)
        ]
        dpowg = [
            np.stack(
                [t * (-1j) * (-1j * G[:, d]) ** max(t - 1, 0) for t in range(l + 1)]
            )
            for d in range(3)
        ]
        phase = np.exp(-1j * (G @ C))
        for ic in range(pm.bas_nctr(ib)):
            for u, lxyz in enumerate(comps):
                for a, ca in zip(exps, coef[:, ic]):
                    common = ca * (np.pi / a) ** 1.5 * np.exp(-G2 / (4 * a)) * phase
                    F, Fg = [], []
                    for d in range(3):
                        E = hermite_E(lxyz[d], 0, a, 0.0, 0.0)[lxyz[d], 0][: l + 1]
                        F.append(E @ powg[d][: len(E)])
                        Fg.append(E @ dpowg[d][: len(E)])
                    x = common * F[0] * F[1] * F[2]
                    xg = common * np.stack(
                        [Fg[0] * F[1] * F[2], F[0] * Fg[1] * F[2], F[0] * F[1] * Fg[2]]
                    )
                    X[off + u] += x
                    dX[:, :, off + u] += (
                        G.T[:, None, :] * G.T[None, :, :] / (2 * a) * x
                        - G.T[:, None, :] * xg[None]
                    )
            off += len(comps)
    return X * nrm[:, None], dX * nrm[None, None, :, None]


def _piece_latsum_virial(cell, rcut, keep):
    """pbc_stress.latsum_virial('int1e_ipovlp_cart') on the orbital piece `keep` (both sides)."""
    mol = cell.mol
    nao, nb0 = mol.nao, mol.nbas
    L1 = cell.translations(rcut)
    sm = piece_mol(cell.supermol(L1), keep)
    X = sm.intor("int1e_ipovlp_cart", comp=3, shls_slice=(0, nb0, 0, sm.nbas)).reshape(
        3, nao, len(L1), nao
    )
    Am = cell.R[PGd.ao_atom(mol)]
    rel = Am[:, None, None, :] - Am[None, None, :, :] - L1[None, :, None, :]
    return -np.einsum("imLn,mLnj->ijmn", X, rel)


def split_stress_2e(cell, gd, auxmol, Y, Wm, w, spherical):
    """(sum Y dJ3/d eps + sum Wm dJ2/d eps (3,3), parts) for the split J3 / J2 (same ranges gd['rng'])."""
    import pbc_stress as ST

    mol = cell.mol
    nao, nb0 = mol.nao, mol.nbas
    rng, crit = gd["rng"], gd["crit"]
    pc = Pieces(cell, auxmol, crit)
    c2s = GG.c2s_of(auxmol, spherical)
    Yc = np.einsum("mnk,pk->mnp", Y, c2s)
    Wc = c2s @ Wm @ c2s.T
    Wc = 0.5 * (
        Wc + Wc.T
    )  # only the symmetric part enters (as the forces; rsgdf/deriv.rs symmetrises)
    naux = auxmol.nao
    c0 = np.pi / (w * w * cell.vol)
    owner = np.zeros(naux, int)
    for kk, (_, _, p0, p1) in enumerate(auxmol.aoslice_by_atom()):
        owner[p0:p1] = kk
    Cp = auxmol.atom_coords()[owner]
    Am = cell.R[PGd.ao_atom(mol)]
    parts = {}
    # ---- SR kept 3c: 2 sum Y (-ip1)(A_m - C_P - T) over both kept calls
    L1 = cell.translations(rng["rcut_pair"])
    sm = cell.supermol(L1)
    T3 = cell.translations(rng["rcut_aux3"])
    out = np.zeros((3, 3))
    for b0, ket in pc.calls(cell, sm):
        chunk = max(1, int(4e6 // (nao * nao * len(L1) * naux)))
        for t0 in range(0, len(T3), chunk):
            Tc = T3[t0 : t0 + chunk]
            big = gto.conc_mol(
                gto.conc_mol(b0, ket), piece_mol(_images(auxmol, Tc), pc.aux_c)
            )
            sl = (0, nb0, nb0, nb0 + ket.nbas, nb0 + ket.nbas, big.nbas)
            with big.with_range_coulomb(-w):
                blk = big.intor("int3c2e_ip1_cart", comp=3, shls_slice=sl).reshape(
                    3, nao, len(L1), nao, len(Tc), naux
                )
            b0y = np.einsum("xmLntp,mnp->xmtp", blk, Yc)
            rel = Am[:, None, None, :] - Cp[None, None, :, :] - Tc[None, :, None, :]
            out -= 2 * np.einsum("xmtp,mtpj->xj", b0y, rel)
    parts["J3_sr"] = out
    # ---- SR kept 2c (P^c | Q^c)
    T2 = cell.translations(rng["rcut_aux2"])
    img = piece_mol(_images(auxmol, T2), pc.aux_c)
    with img.with_range_coulomb(-w):
        d2 = img.intor(
            "int2c2e_ip1_cart", comp=3, shls_slice=(0, auxmol.nbas, 0, img.nbas)
        ).reshape(3, naux, len(T2), naux)
    rel2 = Cp[:, None, None, :] - Cp[None, None, :, :] - T2[None, :, None, :]
    parts["J2_sr"] = -np.einsum("xpTq,pq,pTqj->xj", d2, Wc, rel2)
    # ---- G space
    G = cell.gvectors(rng["gcut"])
    G = G[np.einsum("gi,gi->g", G, G) > 1e-12]
    _, v, vlr, vsr, dv, dvlr, dvsr = _kernels(G, w)
    GGt = -2 * G.T[:, None, :] * G.T[None, :, :]
    dv, dvlr, dvsr = dv * GGt, dvlr * GGt, dvsr * GGt
    if _MUTANT == "no_sr_kstrain":
        dvsr = 0 * dvsr
    vmv, dvmv = (vlr, dvlr) if _MUTANT == "moved_lr" else (vsr, dvsr)
    ss = 0.0 if _MUTANT == "no_ss_pair" else 1.0
    X, dX = aux_ft_strain_n(auxmol, G, pc.anrm)
    Xs, dXs = aux_ft_strain_n(pc.aux_sm, G, pc.anrm)
    Xc, dXc = aux_ft_strain_n(pc.aux_cm, G, pc.anrm)
    P0 = pair_ft(cell, np.zeros((1, 3)))[..., 0].real
    pn = np.sqrt(np.diag(gd["S"]) / np.diag(P0))
    nn = pn[:, None] * pn[None, :]
    Pr, dPr = ST.pair_ft_strain(cell, G)
    P, dP = Pr * nn[:, :, None], dPr * nn[None, None, :, :, None]
    Psr, dPsr = ST.pair_ft_strain(pc.cell_s, G)
    Pss, dPss = Psr * nn[:, :, None], dPsr * nn[None, None, :, :, None]
    YX, YXs, YXc = (np.einsum("mnp,pg->mng", Yc, A) for A in (X, Xs, Xc))
    phiL = np.einsum("mng,mng->g", np.conj(P), YX).real
    phiS = (
        np.einsum("mng,mng->g", np.conj(P), YXs).real
        + np.einsum("mng,mng->g", np.conj(Pss), YXc).real
    )
    E3 = np.sum(vlr * phiL + vmv * phiS)
    t = np.einsum("g,ijmng,mng->ij", vlr, np.conj(dP), YX) + np.einsum(
        "g,mng,mnp,ijpg->ij", vlr, np.conj(P), Yc, dX
    )
    t += np.einsum("g,ijmng,mng->ij", vmv, np.conj(dP), YXs) + np.einsum(
        "g,mng,mnp,ijpg->ij", vmv, np.conj(P), Yc, dXs
    )
    t += ss * (
        np.einsum("g,ijmng,mng->ij", vmv, np.conj(dPss), YXc)
        + np.einsum("g,mng,mnp,ijpg->ij", vmv, np.conj(Pss), Yc, dXc)
    )
    parts["J3_g"] = (
        -E3 * EYE
        + np.einsum("ijg,g->ij", dvlr, phiL)
        + np.einsum("ijg,g->ij", dvmv, phiS)
        + t.real
    ) / cell.vol
    # J2: v_LR (X,X) + v_SR [(X^s,X) + (X^c,X^s)]   (the non-cancelling form of v_SR [(X,X) - (X^c,X^c)])
    WX, WXs = Wc @ X, Wc @ Xs
    phiA = np.einsum("pg,pg->g", np.conj(X), WX).real
    phiM = (
        np.einsum("pg,pg->g", np.conj(Xs), WX).real
        + np.einsum("pg,pg->g", np.conj(Xc), WXs).real
    )
    E2 = np.sum(vlr * phiA + vmv * phiM)
    dphiM = (
        np.einsum("ijpg,pg->ijg", np.conj(dXs), WX)
        + np.einsum("pg,pq,ijqg->ijg", np.conj(Xs), Wc, dX)
        + np.einsum("ijpg,pg->ijg", np.conj(dXc), WXs)
        + np.einsum("pg,pq,ijqg->ijg", np.conj(Xc), Wc, dXs)
    ).real
    parts["J2_g"] = (
        -E2 * EYE
        + np.einsum("ijg,g->ij", dvlr, phiA)
        + np.einsum("ijg,g->ij", dvmv, phiM)
        + 2 * np.einsum("g,ijpg,pg->ij", vlr, np.conj(dX), WX).real
        + np.einsum("g,ijg->ij", vmv, dphiM)
    ) / cell.vol
    # ---- G = 0 (kept inputs)
    dS = ST.latsum_virial(cell, "int1e_ipovlp_cart", rng["rcut_pair"])
    if _MUTANT == "g0_full":
        qg, dSe, Se = pc.q, dS, gd["S"]
    else:
        qg = pc.q_c
        dSs = _piece_latsum_virial(cell, rng["rcut_pair"], pc.orb_s)
        dSe = dS if _MUTANT in ("no_dSss", "no_Sss") else dS - dSs
        Se = gd["S"] if _MUTANT in ("vol_full_S", "no_Sss") else gd["S"] - gd["S_ss"]
    Yq = np.einsum("mnp,p->mn", Yc, qg)
    parts["J3_g0_S"] = -c0 * np.einsum("ijmn,mn->ij", dSe, Yq)
    parts["J3_g0_vol"] = c0 * np.sum(Yq * Se) * EYE
    parts["J2_g0_vol"] = c0 * (qg @ Wc @ qg) * EYE
    return sum(parts.values()), parts


# ------------------------------------------------------------------------------------ solve / FD drivers
def solve(cell, spec, guess=None, tol=1e-11):
    """Split RS-GDF SCF on pure-AFT h (pbc_grad_gdf.integrals).  spec: na, nb, xc, restricted, exxdiv, gcut, aux,
    w, lam, lindep, spherical."""
    ints = GG.integrals(cell, spec["gcut"], spec.get("exxdiv", "none"))
    aux = (
        spec["aux_of"](cell) if "aux_of" in spec else GG.make_auxmol(cell, spec["aux"])
    )
    gd = split_gdf(
        cell,
        aux,
        spec["w"],
        spec.get("lam", 1.0),
        spherical=spec.get("spherical", True),
        lindep=spec.get("lindep", 1e-10),
    )

    def run(ii, g, t):
        return GG.energy(
            cell,
            ii,
            gd,
            spec["na"],
            spec["nb"],
            spec.get("xc", "HF"),
            None,
            spec.get("restricted", False),
            g,
            t,
        )

    try:
        r = run(ints, guess, tol)
    except RuntimeError as err:
        if guess is not None:
            print(f"    [fd] {err}; retrying with tol 1e-9", flush=True)
            r = run(ints, guess, 1e-9)
        elif ints["madelung"]:
            r0 = run(dict(ints, madelung=0.0), None, tol)
            r = run(ints, (r0["Da"], r0["Db"]), tol)
        else:
            raise
    return ints, gd, aux, r


def stress_analytic(cell, spec, ints, gd, aux, r, metric="dk"):
    """Total split RS-GDF stress (dE/d eps): pure-AFT 1e / Ewald / Madelung pieces of pbc_stress.gamma_stress
    (eri=False) + split_stress_2e."""
    import pbc_stress as ST

    s1, p1 = ST.gamma_stress(
        cell,
        ints,
        r["Da"],
        r["Db"],
        r["Fa"],
        r["Fb"],
        r["hyb"],
        w=None,
        sgrid=None,
        xc=spec.get("xc", "HF"),
        restricted=spec.get("restricted", False),
        eri=False,
    )
    Y, Wm, diag = GG.fit_densities(
        gd["J2"],
        gd["J3"],
        [r["Da"], r["Db"]],
        r["hyb"],
        spec.get("lindep", 1e-10),
        metric,
    )
    s2, p2 = split_stress_2e(
        cell, gd, aux, Y, Wm, spec["w"], spec.get("spherical", True)
    )
    return s1 + s2, dict(p1, **p2, diag=diag)


def fd_stress(cell, spec, comps, guess, h=1e-4):
    out = {}
    for i, j in comps:
        e = []
        for s in (1, -1):
            eps = np.zeros((3, 3))
            eps[i, j] = s * h
            e.append(solve(cell.strained(eps), spec, guess)[3]["e"])
        out[(i, j)] = (e[0] - e[1]) / (2 * h)
    return out


__all__ = [
    "Pieces",
    "split_gdf",
    "deriv_ints_split",
    "gamma_gdf_split_grad",
    "split_stress_2e",
    "stress_analytic",
    "solve",
    "fd_stress",
]
