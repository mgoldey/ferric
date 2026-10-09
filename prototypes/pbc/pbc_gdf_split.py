"""Gamma RS-GDF with the RANGE SPLIT (Iteration 23): every (orbital-pair piece, aux piece) whose FULL
Coulomb interaction already converges inside the existing LR G sphere is evaluated in reciprocal space
instead of by the SR erfc real-space lattice sum.  Built on pbc_gdf.py (same J2/J3/B definitions, same
G = 0-dropped kernel, same eigendecomposed metric with the lindep cut); PySCF pbc is not used here.

Partition (exact, linear in the contraction coefficients):
    orbital shell  chi = chi^c + chi^s,  chi^s = its primitives with a <= a_s   (default a_s = lam w^2/2)
    aux shell      X   = X^c   + X^s,    X^s   = its primitives with alpha <= al_s (default al_s = lam w^2)
    pair           chi_m chi_n = [cc + cs + sc] + ss
Moved (G space) combinations: (any pair | X^s) and (ss pair | X^c).  Every moved primitive combination has
1/p + 1/alpha >= 1/w^2 when lam <= 1, i.e. its product FT decays at least as fast as the LR kernel factor
e^{-G^2/4w^2} that sets ferric's gcut = 2w sqrt(ln 1/prec): NO new G vectors.  lam > 1 moves combinations
that do NOT converge in the sphere (the negative control).
Kept (real space, SR erfc): (cc + cs + sc | X^c) = (chi, chi^c | X^c) + (chi^c, chi^s | X^c).

    J3 = SR_real[kept] + (1/Omega) sum_{G in sphere, INCL. G=0} v_SR(G) Re[conj(P) X_s + conj(P_ss) X_c]
         + (1/Omega) sum_{G != 0} v_LR(G) Re[conj(P) X]  -  c0 S q
    v_SR(G) = 4pi/G^2 (1 - e^{-G^2/4w^2}),  v_SR(0) = pi/w^2,  v_LR = 4pi/G^2 e^{-G^2/4w^2},  c0 = pi/(w^2 Omega)

G = 0: the kept SR real-space sum implicitly carries c0 (S - S_ss) q_c.  The moved block's SR part is
evaluated in G space WITH its G = 0 term c0 (S q_s + S_ss q_c) (S, S_ss = lattice overlaps from the 1e
integrals, not the pair FT), so the ONE global subtract (c0 S q, pbc_gdf._subtract_g0) is unchanged.
Equivalently (PySCF rsdf_builder gen_j3c_loader vbar): full 4pi/G^2 kernel on moved blocks at G != 0 and
subtract only c0 (S - S_ss) q_c.  Same algebra, different grouping.
J2 (split_metric=True): (P^c|Q^c) real space; the rest [X X^+ - X_c X_c^+] with v_SR in G space incl. G=0
(PySCF get_2c2e does the same partition: compact-compact analytic SR, everything else by FT).

Units Bohr/Hartree; orbital basis Cartesian (as pbc_gamma), aux spherical by default (as ferric).
"""

from __future__ import annotations

import copy
import math

import numpy as np
from pyscf import gto
from pyscf.gto.mole import NPRIM_OF, NCTR_OF, PTR_COEFF, PTR_EXP

import pbc_gdf
from pbc_gamma import pair_ft
from pbc_gdf import _images, _raw_ft, _rcut_erfc, aux_ft


# ------------------------------------------------------------------------------ pieces
def piece_mol(mol, keep):
    """Copy of a BUILT mol whose every shell keeps only the primitives with keep(exponent) True, with the
    SAME (already normalised) libcint coefficients: chi = piece(keep) + piece(not keep) exactly.
    A shell with no kept primitive becomes one primitive with coefficient 0 (the AO layout never changes).
    Primitives are selected by exponent only (zero-coefficient primitives follow their exponent), so a
    piece that keeps every primitive is bit-for-bit the original shell."""
    pm = copy.copy(mol)
    env = [mol._env]
    n_env = len(mol._env)
    bas = mol._bas.copy()
    cache = {}
    for ib in range(mol.nbas):
        npr, nc, pe, pc = (
            int(bas[ib, k]) for k in (NPRIM_OF, NCTR_OF, PTR_EXP, PTR_COEFF)
        )
        key = (npr, nc, pe, pc)
        if key not in cache:
            e = mol._env[pe : pe + npr]
            c = mol._env[pc : pc + npr * nc].reshape(nc, npr)
            m = np.array([bool(keep(x)) for x in e])
            if m.all():
                cache[key] = (pe, pc, npr)
            else:
                e2, c2 = (e[m], c[:, m]) if m.any() else (e[:1], np.zeros((nc, 1)))
                new_e = n_env
                new_c = n_env + len(e2)
                env += [e2, c2.ravel()]
                n_env += len(e2) + c2.size
                cache[key] = (new_e, new_c, len(e2))
        bas[ib, PTR_EXP], bas[ib, PTR_COEFF], bas[ib, NPRIM_OF] = cache[key]
    pm._bas = bas
    pm._env = np.concatenate(env)
    return pm


class Criterion:
    """Which primitives are 'smooth' (-> G space).  lam = 1 is the rigorous V4s rule; lam = 0 moves nothing;
    lam > 1 moves combinations whose FT does NOT converge in the sphere (negative control)."""

    def __init__(self, w, lam=1.0, a_orb=None, a_aux=None):
        self.w, self.lam = w, lam
        self.a_orb = lam * w * w / 2 if a_orb is None else a_orb
        self.a_aux = lam * w * w if a_aux is None else a_aux

    def orb_smooth(self, e):
        return e <= self.a_orb

    def aux_smooth(self, e):
        return e <= self.a_aux


# -------------------------------------------------------------------- SR real-space 3c
def _sr3(bra0, ket_sm, auxmol, T3, nb0, nL, w, aux_keep=None, same=None):
    """sum_{L,T} (bra_0 ket_L | aux_T)_erfc over the images T3, (nao*nao, naux_c).
    bra0: mol holding the cell-0 bra shells (nb0 of them); ket_sm: supermol over the pair images.
    same=True reproduces pbc_gdf.build_gdf's call exactly (bra = first nb0 shells of ket_sm)."""
    nao = bra0.nao if not same else ket_sm.nao // nL
    naux_c = auxmol.nao
    J3 = np.zeros((nao * nao, naux_c))
    chunk = max(1, int(2e7 // (nao * nao * nL * naux_c)))
    for t0 in range(0, len(T3), chunk):
        ai = _images(auxmol, T3[t0 : t0 + chunk])
        if aux_keep is not None:
            ai = piece_mol(ai, aux_keep)
        nt = len(T3[t0 : t0 + chunk])
        if same:
            big = gto.conc_mol(ket_sm, ai)
            sl = (0, nb0, 0, ket_sm.nbas, ket_sm.nbas, big.nbas)
        else:
            big = gto.conc_mol(gto.conc_mol(bra0, ket_sm), ai)
            sl = (0, nb0, nb0, nb0 + ket_sm.nbas, nb0 + ket_sm.nbas, big.nbas)
        with big.with_range_coulomb(-w):
            v = big.intor("int3c2e_cart", shls_slice=sl)
        J3 += (
            v.reshape(nao, nL, nao, nt, naux_c)
            .sum(axis=(1, 3))
            .reshape(nao * nao, naux_c)
        )
    return J3


def _sr2(auxmol, T2, w, keep=None):
    """sum_T (P_0 | Q_T)_erfc, optionally restricted to the aux piece `keep` on both sides."""
    naux_c = auxmol.nao
    img = _images(auxmol, T2)
    if keep is not None:
        img = piece_mol(img, keep)
    with img.with_range_coulomb(-w):
        J2 = img.intor("int2c2e_cart", shls_slice=(0, auxmol.nbas, 0, img.nbas))
    return J2.reshape(naux_c, len(T2), naux_c).sum(1)


# -------------------------------------------------------------------------- builder
def build_gdf_split(
    cell,
    auxbasis,
    w=1.0,
    prec=1e-13,
    lam=1.0,
    crit=None,
    split_metric=True,
    spherical=True,
    lindep=1e-10,
    auxmol=None,
    reference=False,
    mutant=None,
    rcut_aux3_kept=None,
    verbose=False,
    ref_cache=None,
    g0="B",
):
    """Split RS-GDF.  Returns dict like pbc_gdf.build_gdf plus 'parts' (per-block pieces) and, with
    reference=True, the UNSPLIT J2/J3/B computed in the same run (the moved blocks' real-space sums are then
    available block by block: R_full - R_kept vs the G-space moved term).

    mutant: None | 'no_g0' (moved blocks' SR G=0 term dropped) | 'both' (moved block ALSO left in the
    real-space sum: counted twice) | 'double_ss_s' (ss x X^s counted in both G-space terms).
    ref_cache: optional dict; the unsplit real-space sums (criterion-independent) are reused per (w, prec).
    g0: 'B' (default): moved blocks carry their SR G=0 term, ONE global subtract c0 S q (unchanged).
        'A' (PySCF grouping, recommended for the Rust port): moved blocks get NO G=0 term and the one subtract
        gets the kept inputs, c0 (S - S_ss) q_c (J3) and c0 q_c q_c^T (J2) -- nothing moved => S_ss = 0, q_c = q."""
    crit = crit or Criterion(w, lam)
    mol = cell.mol
    nao, nb0 = mol.nao, mol.nbas
    if auxmol is None:
        nel = sum(gto.charge(s) for s, _ in cell.atoms)
        auxmol = gto.M(
            atom=cell.atoms,
            basis=auxbasis,
            unit="B",
            cart=True,
            verbose=0,
            spin=nel % 2,
        )
    c2s = None
    if spherical:
        sph = auxmol.copy()
        sph.cart = False
        c2s = sph.cart2sph_coeff()
    orb_c, orb_s = (lambda e: not crit.orb_smooth(e)), crit.orb_smooth
    aux_c, aux_s = (lambda e: not crit.aux_smooth(e)), crit.aux_smooth
    q_c_full = aux_ft(auxmol, np.zeros((1, 3)))[:, 0].real
    amin_aux = min(auxmol.bas_exp(i).min() for i in range(auxmol.nbas))
    amin_orb = min(mol.bas_exp(i).min() for i in range(nb0))
    c0 = np.pi / (w * w * cell.vol)

    # ---- ranges: IDENTICAL to pbc_gdf.build_gdf (so lam = 0 is the same computation)
    rcut_pair = np.sqrt(2 * np.log(1 / prec) / amin_orb) + 2.0
    qmax = max(1.0, abs(q_c_full).max())
    th3 = 1.0 / (1.0 / amin_aux + 1.0 / (2 * amin_orb) + 1.0 / w**2)
    rcut_aux3 = _rcut_erfc(th3, prec / qmax) + 2.0
    th2 = 1.0 / (2.0 / amin_aux + 1.0 / w**2)
    rcut_aux2 = _rcut_erfc(th2, prec / qmax**2) + 2.0
    gcut = 2 * w * np.sqrt(np.log(1 / prec))

    L1 = cell.translations(rcut_pair)
    sm = cell.supermol(L1)
    nL = len(L1)
    S = (
        sm.intor("int1e_ovlp_cart", shls_slice=(0, nb0, 0, sm.nbas))
        .reshape(nao, nL, nao)
        .sum(1)
    )
    sm_c, sm_s = piece_mol(sm, orb_c), piece_mol(sm, orb_s)
    mol0_c = piece_mol(mol, orb_c)
    S_ss = (
        sm_s.intor("int1e_ovlp_cart", shls_slice=(0, nb0, 0, sm_s.nbas))
        .reshape(nao, nL, nao)
        .sum(1)
    )

    # aux piece charges and FTs use the FULL aux normalisation (piece FT = raw piece FT x full calibration)
    _, so_full = _raw_ft(auxmol, np.zeros((1, 3)))
    anrm = np.sqrt(np.diag(auxmol.intor("int1e_ovlp_cart")) / so_full)
    aux_cm, aux_sm = piece_mol(auxmol, aux_c), piece_mol(auxmol, aux_s)

    def aux_piece_ft(pm, G):
        return _raw_ft(pm, G)[0] * anrm[:, None]

    q_s = aux_piece_ft(aux_sm, np.zeros((1, 3)))[:, 0].real
    q_cc = aux_piece_ft(aux_cm, np.zeros((1, 3)))[:, 0].real

    # ---- metric
    T2 = cell.translations(rcut_aux2)
    ckey = (w, prec)
    cached = ref_cache.get(ckey) if ref_cache is not None else None
    if cached is not None:
        J2_full_sr = cached["J2_full_sr"]
    else:
        J2_full_sr = (
            _sr2(auxmol, T2, w)
            if (reference or not split_metric or mutant == "both")
            else None
        )
    J2_kept_sr = _sr2(auxmol, T2, w, keep=aux_c) if split_metric else J2_full_sr

    # ---- SR 3c: kept blocks (and the full reference)
    T3 = cell.translations(rcut_aux3)
    T3k = T3 if rcut_aux3_kept is None else cell.translations(rcut_aux3_kept)
    if (reference or mutant == "both") and cached is None:
        cached = dict(
            J2_full_sr=J2_full_sr,
            R_full=_sr3(None, sm, auxmol, T3, nb0, nL, w, same=True),
        )
        if ref_cache is not None:
            ref_cache[ckey] = cached
    if mutant == "both":
        R_kept = cached["R_full"].copy()
    else:
        R_kept = _sr3(mol, sm_c, auxmol, T3k, nb0, nL, w, aux_keep=aux_c)
        R_kept += _sr3(mol0_c, sm_s, auxmol, T3k, nb0, nL, w, aux_keep=aux_c)
    R_full = cached["R_full"] if reference else None

    # ---- G space
    Gall = cell.gvectors(gcut)
    nz = np.einsum("gi,gi->g", Gall, Gall) > 1e-12
    G = Gall[nz]
    G2 = np.einsum("gi,gi->g", G, G)
    vlr = 4 * np.pi / G2 * np.exp(-G2 / (4 * w * w)) / cell.vol
    vsr = 4 * np.pi / G2 * (1 - np.exp(-G2 / (4 * w * w))) / cell.vol
    X = aux_ft(auxmol, G)
    Xs = aux_piece_ft(aux_sm, G)
    Xcc = aux_piece_ft(aux_cm, G)
    P0 = pair_ft(cell, np.zeros((1, 3)))[..., 0].real
    pn = np.sqrt(np.diag(S) / np.diag(P0))
    P = (pair_ft(cell, G) * pn[:, None, None] * pn[None, :, None]).reshape(
        nao * nao, -1
    )
    cell_s = copy.copy(cell)
    cell_s.mol = piece_mol(mol, orb_s)
    Pss = (pair_ft(cell_s, G) * pn[:, None, None] * pn[None, :, None]).reshape(
        nao * nao, -1
    )
    Pss0 = pair_ft(cell_s, np.zeros((1, 3)))[..., 0].real * pn[:, None] * pn[None, :]

    LR3 = ((P.conj() * vlr) @ X.T).real
    LR2 = ((X.conj() * vlr) @ X.T).real
    if mutant == "double_ss_s":
        MG3 = ((P.conj() * vsr) @ Xs.T).real + ((Pss.conj() * vsr) @ X.T).real
        g03 = c0 * (
            S.reshape(-1)[:, None] * q_s[None, :]
            + S_ss.reshape(-1)[:, None] * q_c_full[None, :]
        )
    else:
        MG3 = ((P.conj() * vsr) @ Xs.T).real + ((Pss.conj() * vsr) @ Xcc.T).real
        g03 = c0 * (
            S.reshape(-1)[:, None] * q_s[None, :]
            + S_ss.reshape(-1)[:, None] * q_cc[None, :]
        )
    MG2 = ((X.conj() * vsr) @ X.T).real - ((Xcc.conj() * vsr) @ Xcc.T).real
    g02 = c0 * (np.outer(q_c_full, q_c_full) - np.outer(q_cc, q_cc))
    if mutant == "no_g0" or g0 == "A":
        g03 = 0.0 * g03
        g02 = 0.0 * g02
    moved3 = MG3 + g03
    moved2 = MG2 + g02 if split_metric else 0.0 * LR2

    def finish(J2, J3, S_g0=S, q_g0=q_c_full):
        J2, J3 = pbc_gdf._subtract_g0(J2, J3, S_g0, q_g0, c0)
        if c2s is not None:
            J2, J3 = c2s.T @ J2 @ c2s, J3 @ c2s
        J2 = 0.5 * (J2 + J2.T)
        J3 = J3.reshape(nao, nao, -1)
        J3 = 0.5 * (J3 + J3.transpose(1, 0, 2))
        s, U = np.linalg.eigh(J2)
        keep = s > lindep
        B = np.einsum("Pk,mnP->kmn", U[:, keep] / np.sqrt(s[keep]), J3)
        return dict(J2=J2, J3=J3, B=B, naux_kept=int(keep.sum()), metric_min=s.min())

    J3 = R_kept + moved3 + LR3 if mutant != "both" else R_kept + moved3 + LR3
    J2 = J2_kept_sr + moved2 + LR2
    if g0 == "A":
        if not split_metric:
            raise ValueError(
                "g0='A' needs split_metric=True (J2 and J3 share the one subtract's q_c)"
            )
        out = finish(J2, J3, S - S_ss, q_cc)
    else:
        out = finish(J2, J3)
    out.update(
        S=S,
        S_ss=S_ss,
        info=dict(
            nG=len(G),
            gcut=gcut,
            n_pair_images=nL,
            n_aux3=len(T3),
            n_aux3_kept=len(T3k),
            n_aux2=len(T2),
            a_orb=crit.a_orb,
            a_aux=crit.a_aux,
            Pss0_vs_Sss=float(abs(Pss0 - S_ss).max()),
            n_aux_smooth=int(
                sum(
                    np.all([crit.aux_smooth(e) for e in auxmol.bas_exp(i)])
                    for i in range(auxmol.nbas)
                )
            ),
            n_aux_mixed=int(
                sum(
                    len({crit.aux_smooth(e) for e in auxmol.bas_exp(i)}) == 2
                    for i in range(auxmol.nbas)
                )
            ),
            n_orb_prims_smooth=int(
                sum(
                    np.sum([crit.orb_smooth(e) for e in mol.bas_exp(i)])
                    for i in range(nb0)
                )
            ),
        ),
    )
    out["parts"] = dict(
        R_kept=R_kept, moved3=moved3, moved2=moved2, LR3=LR3, LR2=LR2, g03=g03, g02=g02
    )
    if reference:
        ref = finish(J2_full_sr + LR2, R_full + LR3)
        out["ref"] = ref
        out["parts"].update(R_full=R_full, J2_full_sr=J2_full_sr, J2_kept_sr=J2_kept_sr)
    if verbose:
        print(
            "  split:",
            {
                k: (f"{v:.3g}" if isinstance(v, float) else v)
                for k, v in out["info"].items()
            },
        )
    return out


# ------------------------------------------------------------------- ferric-rule counts
def count_sr3_ferric(
    obs, aux, lat, w=1.0, prec=1e-13, crit=None, mode="trim", return_sr2=True
):
    """EXACT enumeration of ferric's SR 3-centre triplets (rsgdf.rs sr3_pair_image: pair_bound + sr_radius +
    segment test; zero-coefficient primitives kept, as ferric does) for the unsplit build (crit=None) or the
    split build.  obs/aux: cell_facts.ferric_shells lists (spherical, ferric normalisation).

    mode='trim': one call per ORIGINAL (mu-shell, nu-shell, P-shell) whose kept part is nonzero; the pair
      bound runs over the kept primitive pairs only (not both smooth) and the aux bound over P's compact
      primitives -- what a libint ket ShellPair with the ss primitive pairs removed would compute.
    mode='twocall': (chi, chi^c | X^c) + (chi^c, chi^s | X^c): two ordinary contracted calls, no ShellPair surgery
      (what pbc_gdf_split._sr3 evaluates).
    mode='pieces': shells decontracted into c/s pieces, one call per (c,c), (c,s), (s,c) piece pair."""
    import itertools

    sq = math.sqrt(math.pi)

    def qbound(e, k, lq):
        return float(np.max(np.abs(k) * (math.pi / e) ** 1.5 * (1 + e**-0.5) ** lq))

    # aux (compact pieces when split); None = nothing left in real space
    aq, amn, amx, alive = [], [], [], []
    for p in aux:
        m = (
            np.ones(len(p["e"]), bool)
            if crit is None
            else ~np.array([crit.aux_smooth(x) for x in p["e"]])
        )
        alive.append(bool(m.any()))
        aq.append(qbound(p["e"][m], p["k"][m], p["l"]) if m.any() else 0.0)
        amn.append(p["e"][m].min() if m.any() else 1.0)
        amx.append(p["e"][m].max() if m.any() else 1.0)
    aq, amn, amx, alive = map(np.array, (aq, amn, amx, alive))
    # pair-image radius: ferric's pair_image_radius on the FULL shells (unchanged by the split)
    qfull = np.array([qbound(p["e"], p["k"], p["l"]) for p in aux])
    amin_orb = min(s["e"].min() for s in obs)
    pmax_orb = 2 * max(s["e"].max() for s in obs)
    vfac = max(
        1.0,
        max(
            1 + 2 * math.sqrt(pmax_orb * p["e"].max() / (pmax_orb + p["e"].max())) / sq
            for p in aux
        ),
    )
    rpair = (
        math.sqrt(
            2 * math.log(1e3 / (0.1 * prec / max(qfull.max() * vfac, 1.0))) / amin_orb
        )
        + 2.0
    )

    b = 2 * np.pi * np.linalg.inv(lat).T

    def lattice_points(r, x0=np.zeros(3)):
        rng = [
            int(
                math.ceil((r + np.linalg.norm(x0)) * np.linalg.norm(b[j]) / (2 * np.pi))
            )
            + 1
            for j in range(3)
        ]
        n = np.array(list(itertools.product(*[range(-k, k + 1) for k in rng])), float)
        t = n @ lat
        return t[np.linalg.norm(t - x0, axis=1) <= r]

    images = lattice_points(rpair)
    # orbital "pair units" per ordered original shell pair
    units = []  # (i, j, kk, cc, pmin, pmax)
    for i, a in enumerate(obs):
        for j, bb in enumerate(obs):
            sa = (
                np.zeros(len(a["e"]), bool)
                if crit is None
                else np.array([crit.orb_smooth(x) for x in a["e"]])
            )
            sb = (
                np.zeros(len(bb["e"]), bool)
                if crit is None
                else np.array([crit.orb_smooth(x) for x in bb["e"]])
            )
            if mode == "trim" or crit is None:
                masks = [~(sa[:, None] & sb[None, :])]
            elif mode == "twocall":
                masks = [
                    np.ones_like(sa)[:, None] & ~sb[None, :],
                    ~sa[:, None] & sb[None, :],
                ]
            else:
                masks = [
                    ma[:, None] & mb[None, :]
                    for ma, mb in ((~sa, ~sb), (~sa, sb), (sa, ~sb))
                ]
            p = a["e"][:, None] + bb["e"][None, :]
            for pm in masks:
                if not pm.any():
                    continue
                kk = (a["e"][:, None] * bb["e"][None, :] / p)[pm]
                cc = (
                    np.abs(a["k"][:, None] * bb["k"][None, :]) * (math.pi / p) ** 1.5
                )[pm]
                units.append((i, j, kk, cc, p[pm].min(), p[pm].max()))
    ncmax = max(len(u[2]) for u in units)
    KK = np.zeros((len(units), ncmax))
    CC = np.zeros((len(units), ncmax))
    for u, (_, _, kk, cc, _, _) in enumerate(units):
        KK[u, : len(kk)] = kk
        CC[u, : len(cc)] = cc
    PMIN = np.array([u[4] for u in units])
    PMAX = np.array([u[5] for u in units])
    UI = np.array([u[0] for u in units])
    UJ = np.array([u[1] for u in units])
    mu = np.sqrt(PMAX[:, None] * amx[None, :] / (PMAX[:, None] + amx[None, :]))
    nu = 1 / np.sqrt(1 / PMIN[:, None] + 1 / amn[None, :] + 1 / w**2)
    fac = aq[None, :] * (1 + 2 * mu / sq)  # (units, aux)
    ocen = np.array([s["c"] for s in obs])
    acen = np.array([p["c"] for p in aux])
    # group by centres
    oat = {}
    for u in range(len(units)):
        oat.setdefault((tuple(ocen[UI[u]]), tuple(ocen[UJ[u]])), []).append(u)
    aat = {}
    for k in range(len(aux)):
        aat.setdefault(tuple(acen[k]), []).append(k)
    n3 = 0
    rmax_glob = 0.0
    for L in images:
        for (ca, cb), us in oat.items():
            A, Bc = np.array(ca), np.array(cb) + L
            r2 = float((A - Bc) @ (A - Bc))
            us = np.array(us)
            q = np.max(CC[us] * np.exp(-KK[us] * r2), axis=1)
            pref = q[:, None] * fac[us]
            with np.errstate(divide="ignore", invalid="ignore"):
                rad = np.where(
                    (pref > prec) & alive[None, :],
                    np.sqrt(np.log(np.maximum(pref, prec) / prec)) / nu[us] + 2.0,
                    np.nan,
                )
            if np.all(np.isnan(rad)):
                continue
            mid, half = 0.5 * (A + Bc), 0.5 * math.sqrt(r2)
            ab = Bc - A
            l2 = ab @ ab
            for cen, ks in aat.items():
                rk = rad[:, ks]
                ok = ~np.isnan(rk)
                if not ok.any():
                    continue
                rmax = rk[ok].max()
                rmax_glob = max(rmax_glob, rmax)
                cen = np.array(cen)
                pts = lattice_points(rmax + half, mid - cen) + cen
                t = (
                    np.clip(((pts - A) @ ab) / l2, 0, 1)
                    if l2 > 0
                    else np.zeros(len(pts))
                )
                dist = np.sort(np.linalg.norm(pts - A - t[:, None] * ab, axis=1))
                n3 += int(np.searchsorted(dist, rk[ok], side="right").sum())
    out = dict(
        n_sr3=n3,
        n_pair_images=len(images),
        rpair=rpair,
        n_units=len(units),
        rmax=rmax_glob,
    )
    if return_sr2:
        n2 = 0
        for ip in range(len(aux)):
            for iq in range(len(aux)):
                if not (alive[ip] and alive[iq]):
                    continue
                m_ = math.sqrt(amx[ip] * amx[iq] / (amx[ip] + amx[iq]))
                n_ = 1 / math.sqrt(1 / amn[ip] + 1 / amn[iq] + 1 / w**2)
                pref = aq[ip] * aq[iq] * (1 + 2 * m_ / sq)
                if pref > prec:
                    r = math.sqrt(math.log(pref / prec)) / n_ + 2.0
                    n2 += len(lattice_points(r, acen[ip] - acen[iq]))
        out["n_sr2"] = n2
    return out


__all__ = ["Criterion", "piece_mol", "build_gdf_split", "count_sr3_ferric"]
