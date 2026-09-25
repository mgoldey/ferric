"""Iteration 18: Gamma-point analytic nuclear forces when J/K come from RS-GDF (pbc_gdf.py) instead of dense AFT.

Covers RHF, UHF and (by composition with Iteration 17's XC terms, pbc_grad_open.xc_grad) RKS/UKS global hybrids.
Nothing in pbc_gdf.py / pbc_grad.py / pbc_grad_open.py is modified: the 1e, Ewald, Madelung and XC pieces are
imported, the fitted two-electron piece is new.

ENERGY (the one pbc_gdf + pbc_grad_open.scf actually converge; alpha = exact-exchange fraction, D = D_a + D_b)
    E = sum D h + sum_mnls Gam_mnls I^fit_mnls - alpha v_M/2 sum_s tr(D_s S D_s S) + E_xc + E_nn
    Gam = 1/2 D_mn D_ls - alpha/2 sum_s D^s_ml D^s_ns,     I^fit = J3 f(J2) J3^T
    J2[P,Q]  = SR sum_T (P_0|Q_T)_erfc + LR (1/Om) sum_G v_w conj(X_P) X_Q  - c0 q_P q_Q
    J3[mn,P] = SR sum_{L,T} (m_0 n_L|P_T)_erfc + LR (1/Om) sum_G v_w conj(P_mn) X_P - c0 S_mn q_P
    f(J2) = U diag(f(s)) U^T,  f(s) = 1/s for s > lindep, 0 otherwise   (the eigenvalue-cut pseudo-inverse)
h, S, E_nn, v_M are pbc_gamma.build_integrals' (pure AFT, point nuclei, gcut explicit) -- the SAME as the dense-AFT
energy of Iterations 16/17, so the only difference between the two energies is I^fit vs I.

DERIVATIVE (everything of atom A moves in every image: nucleus, orbital centres, aux centres)
    dE_2e = sum_{mn,P} Y[mn,P] dJ3[mn,P] + sum_{PQ} Wm[P,Q] dJ2[P,Q]
    Y[mn,P] = 2 sum_ls Gam_mnls C[P,ls] = D_mn c_P - alpha sum_s (D_s C_P D_s)_mn,   C = f(J2) J3^T, c = C.D
    Wm      = U (Lo o (U^T H U)) U^T,   H[P,Q] = sum Gam_mnls J3[mn,P] J3[ls,Q]
              Lo_ij = (f(s_i) - f(s_j)) / (s_i - s_j),  Lo_ii = f'(s_i)     (Daleckii-Krein / Loewner matrix)
       kept-kept block: Lo = -1/(s_i s_j)  =>  Wm = -f H f = -(1/2 c c^T - alpha/2 sum_s tr(C_P D_s C_Q D_s))
                        (the textbook "-1/2 c^T (P|Q)' c" DF metric term; mode 'std' uses ONLY this block)
       kept-dropped:    Lo = (1/s_i) / (s_i - s_j)   (the eigenvectors of the cut subspace rotate with R)
       dropped-dropped: 0.
    With no eigenvalue dropped, 'std' == 'dk' identically (the exactness anchor for the cut).
dJ3 pieces:
    SR bra    sum_{L,T} (d/dA m_0 n_L|P_T) = -int3c2e_ip1, x2 for the ket (Y symmetric in mn; J3[mn]=J3[nm] at Gamma)
    SR aux    sum_{L,T} (m_0 n_L|d/dC P_T) = -int3c2e_ip2
    LR bra    (1/Om) sum_G v_w Re[conj(Q_mn) X_P], Q = dP_mn/dA_m (pbc_grad.pair_ft_deriv), x2 for the ket
    LR aux    (1/Om) sum_G v_w Re[conj(P_mn) (-iG) X_P]        (an aux function translates rigidly: dX/dC = -iG X)
    G=0       -c0 dS_mn q_P   (q_P = int chi_P is position-free)  -> folded into the S contraction as
              M_g0 = -c0 sum_P Y[mn,P] q_P
dJ2 pieces:
    SR        d/dC_P sum_T (P_0|Q_T) = -int2c2e_ip1[P,Q];  d/dC_Q = +int2c2e_ip1[P,Q]  (2-centre translation invariance)
    LR        (1/Om) sum_G v_w Re[conj(-iG X_P) X_Q] + (P<->Q)
    G=0       c0 q q^T: constant.
Everything else is Iteration 16/17 (T, V_LR basis+nucleus, Ewald, M = -W - alpha v_M sum_s D_s S D_s; c0_h = 0 here
because h is pure AFT) plus, for KS, pbc_grad_open.xc_grad.
Aux centres may be functions of the atom positions (ghost aux of the exact-span cross-check): aux_jac[k, A] = dC_k/dR_A
(scalar, same for x/y/z) folds the per-aux-centre gradient onto atoms.  Default: aux on atoms, jac = identity.

_MUTANT (module global): 'no_metric' (drop every dJ2 term), 'no_aux' (drop the aux-centre derivative of J3; J2 kept),
'no_g0' (drop -c0 dS q), 'g0_dense' (use the dense-AFT ERI's G=0 M-term  c0(-N D + alpha sum D_s S D_s) instead of
-c0 Y q), 'no_lr3' (drop the LR part of dJ3, bra and aux), 'std' is not a mutant but a mode (metric='std').
"""

from __future__ import annotations

import numpy as np
from pyscf import gto

import pbc_grad as PGd
import pbc_grad_open as PGO
from pbc_gamma import build_integrals, pair_ft
from pbc_gdf import _images, _rcut_erfc, aux_ft, build_gdf, jk_from_B

_MUTANT = None


# ------------------------------------------------------------------------------------------------ setup
def make_auxmol(cell, auxbasis):
    nel = sum(gto.charge(s) for s, _ in cell.atoms)
    return gto.M(atom=cell.atoms, basis=auxbasis, unit="B", cart=True, verbose=0, spin=nel % 2)


def gdf_ranges(cell, auxmol, w, prec=1e-13):
    """Exactly pbc_gdf.build_gdf's default ranges (geometry independent: exponents and aux charges only)."""
    mol = cell.mol
    q_c = aux_ft(auxmol, np.zeros((1, 3)))[:, 0].real
    amin_aux = min(auxmol.bas_exp(i).min() for i in range(auxmol.nbas))
    amin_orb = min(mol.bas_exp(i).min() for i in range(mol.nbas))
    qmax = max(1.0, abs(q_c).max())
    th3 = 1.0 / (1.0 / amin_aux + 1.0 / (2 * amin_orb) + 1.0 / w**2)
    th2 = 1.0 / (2.0 / amin_aux + 1.0 / w**2)
    return dict(rcut_pair=np.sqrt(2 * np.log(1 / prec) / amin_orb) + 2.0,
                rcut_aux3=_rcut_erfc(th3, prec / qmax) + 2.0,
                rcut_aux2=_rcut_erfc(th2, prec / qmax**2) + 2.0,
                gcut=2 * w * np.sqrt(np.log(1 / prec)))


def gdf(cell, auxmol, w=1.0, prec=1e-13, spherical=True, lindep=1e-10, rng=None):
    rng = rng or gdf_ranges(cell, auxmol, w, prec)
    out = build_gdf(cell, None, w=w, prec=prec, spherical=spherical, lindep=lindep, auxmol=auxmol, **rng)
    out["rng"] = rng
    return out


def c2s_of(auxmol, spherical):
    if not spherical:
        return np.eye(auxmol.nao)
    sph = auxmol.copy()
    sph.cart = False
    return sph.cart2sph_coeff()


def B_from(J2, J3, lindep):
    """B exactly as build_gdf forms it, for any cut (one GDF build serves every lindep)."""
    s, U = np.linalg.eigh(J2)
    keep = s > lindep
    return np.einsum("Pk,mnP->kmn", U[:, keep] / np.sqrt(s[keep]), J3)


# ------------------------------------------------------------------------------------ fitted densities
def fit_densities(J2, J3, Ds, alpha, lindep, metric="dk"):
    """Y (nao,nao,naux) and Wm (naux,naux) of the module docstring, plus diagnostics.
    Ds = [D_a, D_b] (RHF: [D/2, D/2]).  metric: 'dk' exact derivative of the cut energy, 'std' kept block only."""
    nao = J3.shape[0]
    s, U = np.linalg.eigh(J2)
    keep = s > lindep
    f = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
    Jinv = (U * f) @ U.T
    J3f = J3.reshape(nao * nao, -1)
    C = (Jinv @ J3f.T).reshape(-1, nao, nao)  # C[P] = fitted coefficients of every pair
    D = sum(Ds)
    c = np.einsum("pmn,mn->p", C, D)
    Y = D[:, :, None] * c[None, None, :]
    for Dsp in Ds:
        Y = Y - alpha * np.einsum("ml,pls,sn->mnp", Dsp, C, Dsp)
    # H = J3^T Gam J3 in the aux space
    b = J3f.T @ D.reshape(-1)
    H = 0.5 * np.outer(b, b)
    J3p = J3.transpose(2, 0, 1)
    for Dsp in Ds:
        t = np.einsum("ml,pls,sn->pmn", Dsp, J3p, Dsp)  # (D_s J3_P D_s)
        H = H - 0.5 * alpha * np.einsum("pmn,qmn->pq", t, J3p)
    Hu = U.T @ H @ U
    ds = s[:, None] - s[None, :]
    same = np.abs(ds) < 1e-300
    with np.errstate(divide="ignore", invalid="ignore"):
        Lo = np.where(same, 0.0, (f[:, None] - f[None, :]) / np.where(same, 1.0, ds))
    kk = keep[:, None] & keep[None, :]
    Lo = np.where(kk, -f[:, None] * f[None, :], Lo)  # exact form in the kept block (no cancellation)
    if metric == "std":
        Lo = np.where(kk, Lo, 0.0)
    Wm = U @ (Lo * Hu) @ U.T
    Wm_std = -Jinv @ H @ Jinv
    diag = dict(naux=len(s), n_drop=int((~keep).sum()), s_min=s.min(), s_kept_min=s[keep].min(),
                s_drop_max=s[~keep].max() if (~keep).any() else 0.0,
                cross_norm=abs(Wm - Wm_std).max())
    return Y, Wm, diag


# ------------------------------------------------------------------------------------ derivative integrals
def deriv_ints(cell, ints, gd, auxmol, w, rcut_1e=22.0, chunk_elems=4e6, verbose=False):
    """Every density-INDEPENDENT derivative quantity, computed once per geometry (mutants/metrics/spin cases reuse it).
      d3b[x,m,n,P] = sum_{L,T} (grad_e m_0 n_L | P_T)_erfc      d3a[x,m,n,P] = sum_{L,T} (m_0 n_L | grad_e P_T)_erfc
      d2[x,P,Q]    = sum_T (grad_e P_0 | Q_T)_erfc
      LR: G (GDF sphere), vlr, X (cart aux FT), P, Q (pair FT and bra derivative, build_gdf's normalisation)
      1e: XS, XT (rcut_1e), XSg (build_gdf's rcut_pair), Qh on ints' G set."""
    import time

    mol = cell.mol
    nao, nb0 = mol.nao, mol.nbas
    naux_c = auxmol.nao
    rng = gd["rng"]
    tm = {}
    t = time.time()
    L1 = cell.translations(rng["rcut_pair"])
    sm = cell.supermol(L1)
    T3 = cell.translations(rng["rcut_aux3"])
    d3b = np.zeros((3, nao, nao, naux_c))
    d3a = np.zeros((3, nao, nao, naux_c))
    chunk = max(1, int(chunk_elems // (nao * nao * len(L1) * naux_c)))
    for t0 in range(0, len(T3), chunk):
        ai = _images(auxmol, T3[t0 : t0 + chunk])
        big = gto.conc_mol(sm, ai)
        nt = len(T3[t0 : t0 + chunk])
        sl = (0, nb0, 0, sm.nbas, sm.nbas, big.nbas)
        with big.with_range_coulomb(-w):
            d3b += big.intor("int3c2e_ip1_cart", comp=3, shls_slice=sl).reshape(
                3, nao, len(L1), nao, nt, naux_c).sum(axis=(2, 4))
            d3a += big.intor("int3c2e_ip2_cart", comp=3, shls_slice=sl).reshape(
                3, nao, len(L1), nao, nt, naux_c).sum(axis=(2, 4))
    tm["sr3c"] = time.time() - t
    t = time.time()
    T2 = cell.translations(rng["rcut_aux2"])
    img = _images(auxmol, T2)
    with img.with_range_coulomb(-w):
        d2 = img.intor("int2c2e_ip1_cart", comp=3, shls_slice=(0, auxmol.nbas, 0, img.nbas))
    d2 = d2.reshape(3, naux_c, len(T2), naux_c).sum(2)
    tm["sr2c"] = time.time() - t
    t = time.time()
    G = cell.gvectors(rng["gcut"])
    G = G[np.einsum("gi,gi->g", G, G) > 1e-12]
    G2 = np.einsum("gi,gi->g", G, G)
    vlr = 4 * np.pi / G2 * np.exp(-G2 / (4 * w * w)) / cell.vol
    X = aux_ft(auxmol, G)
    Praw, Q = PGd.pair_ft_deriv(cell, G)
    P0 = pair_ft(cell, np.zeros((1, 3)))[..., 0].real
    pn = np.sqrt(np.diag(gd["S"]) / np.diag(P0))  # the SAME normalisation build_gdf applies
    nn = pn[:, None] * pn[None, :]
    P, Q = Praw * nn[:, :, None], Q * nn[None, :, :, None]
    tm["lr_gdf"] = time.time() - t
    t = time.time()
    S = ints["S"]
    Ph, Qh = PGd.pair_ft_deriv(cell, ints["G"])
    nrm = np.sqrt(np.diag(S) / np.diag(P0))
    Qh = Qh * (nrm[:, None] * nrm[None, :])[None, :, :, None]
    tm["lr_h"] = time.time() - t
    t = time.time()
    XS = PGd._latsum_ip(cell, "int1e_ipovlp_cart", rcut_1e)
    XT = PGd._latsum_ip(cell, "int1e_ipkin_cart", rcut_1e)
    XSg = PGd._latsum_ip(cell, "int1e_ipovlp_cart", rng["rcut_pair"])
    tm["1e"] = time.time() - t
    aux_owner = np.zeros(naux_c, int)
    for k, (_, _, p0, p1) in enumerate(auxmol.aoslice_by_atom()):
        aux_owner[p0:p1] = k
    if verbose:
        print("  deriv_ints timings (s): " + ", ".join(f"{k} {v:.0f}" for k, v in tm.items()), flush=True)
    return dict(d3b=d3b, d3a=d3a, d2=d2, G=G, vlr=vlr, X=X, P=P, Q=Q, Qh=Qh, XS=XS, XT=XT, XSg=XSg,
                q_c=aux_ft(auxmol, np.zeros((1, 3)))[:, 0].real, aux_owner=aux_owner, natm_aux=auxmol.natm,
                timings=tm)


# -------------------------------------------------------------------------------------------- gradient
def gamma_gdf_grad(cell, ints, gd, auxmol, Da, Db, Fa, Fb, alpha, w=1.0, lindep=1e-10, spherical=True, metric="dk",
                   aux_jac=None, grid=None, xc=None, restricted=False, response=True, rcut_1e=22.0, cache=None):
    """dE/dR (natm, 3), parts, diag for the RS-GDF energy at a converged (D_s, F_s).
    ints: pbc_gamma.build_integrals(cell, None, gcut=...) (pure-AFT h, S, E_nn, P, G, madelung)
    gd:   gdf(cell, auxmol, w, ...) output (J2, J3 in the spherical/cart aux space, S, rng)
    cache: dict filled by deriv_ints on first use (pass the same dict for every call at this geometry)."""
    if cache is None:
        cache = {}
    if not cache:
        cache.update(deriv_ints(cell, ints, gd, auxmol, w, rcut_1e))
    k = cache
    S, Gh, vM = ints["S"], ints["G"], ints["madelung"]
    mol = cell.mol
    natm = mol.natm
    aoat = PGd.ao_atom(mol)
    D = Da + Db
    spins = ((Da, Fa), (Db, Fb))
    parts = {}
    c2s = c2s_of(auxmol, spherical)
    c0 = np.pi / (w * w * cell.vol)

    # ---- fitted two-electron densities (Y: 3-index, Wm: metric)
    Y, Wm, diag = fit_densities(gd["J2"], gd["J3"], [Da, Db], alpha, lindep, metric)
    Yc = np.einsum("mnk,pk->mnp", Y, c2s)  # back to cart aux (c2s is position-free)
    Wc = c2s @ Wm @ c2s.T
    if _MUTANT == "no_metric":
        Wc = 0 * Wc

    # ---- overlap-coupled (1e Pulay + Madelung; h is pure AFT so no c0_h term)
    W = sum(Ds @ Fs @ Ds for Ds, Fs in spins)
    M = -W - alpha * vM * sum(Ds @ S @ Ds for Ds, _ in spins)
    parts["S"] = PGd._fold(-2 * np.einsum("xmn,mn->mx", k["XS"], M), aoat, natm)
    parts["T"] = PGd._fold(-2 * np.einsum("xmn,mn->mx", k["XT"], D), aoat, natm)
    # ---- G=0 of J3: -c0 S_mn q_P  (S = build_gdf's lattice overlap, range rcut_pair) -> S contraction
    if _MUTANT == "g0_dense":
        N = np.sum(D * S)
        Mg0 = c0 * (-N * D + alpha * sum(Ds @ S @ Ds for Ds, _ in spins))
    else:
        Mg0 = -c0 * np.einsum("mnp,p->mn", Yc, k["q_c"])
    if _MUTANT == "no_g0":
        Mg0 = 0 * Mg0
    parts["J3_g0"] = PGd._fold(-2 * np.einsum("xmn,mn->mx", k["XSg"], Mg0), aoat, natm)

    # ---- V_LR (pure AFT, point nuclei) and Ewald: Iteration 16 formulas on ints' G set
    Ph = ints["P"]
    v = 4 * np.pi / np.einsum("gi,gi->g", Gh, Gh)
    SG_at = np.exp(-1j * Gh @ cell.R.T) * cell.Z
    SG = SG_at.sum(1)
    rho = np.einsum("mn,mng->g", D, Ph)
    rq = PGd._fold_g(2 * np.einsum("mn,xmng->mxg", D, k["Qh"]), aoat, natm)
    parts["Vlr_basis"] = -(np.einsum("g,axg->ax", v, np.conj(rq) * SG).real) / cell.vol
    parts["Vlr_nuc"] = -(np.einsum("g,ga,gx->ax", v * np.conj(rho), SG_at * (-1j), Gh).real) / cell.vol
    parts["nn"] = PGd.ewald_grad(cell, 1.0)

    # ---- fitted 2e: dJ3 (SR/LR x bra/aux) and dJ2 (SR/LR); d/dcentre = -(electron gradient)
    gb_sr = -2 * np.einsum("xmnp,mnp->mx", k["d3b"], Yc)  # x2: ket (Y symmetric)
    ga3_sr = -np.einsum("xmnp,mnp->px", k["d3a"], Yc)
    G, vlr, X, P, Q = k["G"], k["vlr"], k["X"], k["P"], k["Q"]
    XY = np.einsum("mnp,pg->mng", Yc, X)
    gb_lr = 2 * np.einsum("g,xmng,mng->mx", vlr, np.conj(Q), XY).real
    PY = np.einsum("mnp,mng->pg", Yc, np.conj(P))
    ga3_lr = np.einsum("g,pg,gx,pg->px", vlr, PY, -1j * G, X).real
    # J2: d/dC_P (P_0|Q_T) = -d2[P,Q], d/dC_Q = +d2[P,Q];  LR: conj(-iG X_P) = +iG conj(X_P), x2 (Wc symmetric)
    ga2_sr = -np.einsum("xpq,pq->px", k["d2"], Wc) + np.einsum("xpq,pq->qx", k["d2"], Wc)
    ga2_lr = 2 * np.einsum("g,pg,gx,pg->px", vlr, np.conj(X), 1j * G, Wc @ X).real
    if _MUTANT == "no_lr3":
        gb_lr, ga3_lr = 0 * gb_lr, 0 * ga3_lr
    if _MUTANT == "no_aux":  # drop the aux-centre derivative of J3 only (the metric term is kept)
        ga3_sr, ga3_lr = 0 * ga3_sr, 0 * ga3_lr
    jac = np.eye(natm) if aux_jac is None else np.asarray(aux_jac)

    def fold_aux(per_aux):
        per_center = np.zeros((k["natm_aux"], 3))
        np.add.at(per_center, k["aux_owner"], per_aux)
        return jac.T @ per_center

    parts["J3_bra_sr"] = PGd._fold(gb_sr, aoat, natm)
    parts["J3_bra_lr"] = PGd._fold(gb_lr, aoat, natm)
    parts["J3_aux_sr"] = fold_aux(ga3_sr)
    parts["J3_aux_lr"] = fold_aux(ga3_lr)
    parts["J2_sr"] = fold_aux(ga2_sr)
    parts["J2_lr"] = fold_aux(ga2_lr)

    if grid is not None and str(xc).upper() != "HF":
        xg = PGO.xc_grad(grid, Da, Db, xc, restricted, natm, aoat, response)
        parts["xc_ao"], parts["xc_point"], parts["xc_weight"] = xg["ao"], xg["point"], xg["weight"]
    return sum(parts.values()), parts, diag


# ------------------------------------------------------------------------------------------ convenience
def integrals(cell, gcut, exxdiv="none"):
    return build_integrals(cell, None, verbose=False, exxdiv=exxdiv, gcut=gcut)


def energy(cell, ints, gd, na, nb, xc="HF", grid=None, restricted=False, guess=None, tol=1e-11):
    r = PGO.scf(ints["S"], ints["h"], jk_from_B(gd["B"]), ints["enn"], na, nb, grid, xc, ints["madelung"],
                restricted, guess, tol)
    return r


def run(cell, auxmol, na, nb, xc="HF", gcut=10.0, w=1.0, lindep=1e-10, spherical=True, exxdiv="none", metric="dk",
        aux_jac=None, grid=None, restricted=False, guess=None, tol=1e-11, ints=None, gd=None, response=True, cache=None):
    ints = ints or integrals(cell, gcut, exxdiv)
    gd = gd or gdf(cell, auxmol, w, spherical=spherical, lindep=lindep)
    r = energy(cell, ints, gd, na, nb, xc, grid, restricted, guess, tol)
    g, parts, diag = gamma_gdf_grad(cell, ints, gd, auxmol, r["Da"], r["Db"], r["Fa"], r["Fb"], r["hyb"], w=w,
                                    lindep=lindep, spherical=spherical, metric=metric, aux_jac=aux_jac, grid=grid,
                                    xc=xc, restricted=restricted, response=response, cache=cache)
    return dict(e=r["e"], grad=g, parts=parts, diag=diag, scf=r, ints=ints, gd=gd)
