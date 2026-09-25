"""Iteration 22: Gamma-point analytic forces with the periodic ECP term (pure-AFT RHF/UHF route).

Energy (pbc_ecp + pbc_gamma conventions; Z = Z_eff wherever a nucleus is a Coulomb source):
    E = sum D (T + V_ne + V_ECP) + E_2e + E_nn,    V_ECP_mn = sum_{L in So} sum_{(M, C) in Se} <m_0 | U_C(r - R_C - M) | n_L>
with So / Se the orbital- and ECP-image sets.  The ECP term is one-electron and adds to hcore's derivative:

    dE_ECP/dR_A = sum_{(M,C), L} sum_mn D_mn [ delta_{A,atom(m)} d/dA_m + delta_{A,atom(n)} d/dB_n + delta_{A,C} d/dC ]
                                           <m_0 | U_C(. - R_C - M) | n_L>
    d/dA_m <m|U|n> = <d_A m|U|n>,  d_A x^i e^{-a x^2} = 2a x^{i+1} e - i x^{i-1} e   (VALUE integrals over
                     raised/lowered primitive shells, deriv_basis; PySCF's ECPscalar_ipnuc is NOT used by default:
                     it is wrong by up to 1.4e-7 on off-centre elements, FINDINGS Iteration 22)
    d/dB_n <m|U|n_L> = the same with the raised shells at the ket image R_B + L
    d/dC           = -(d/dA_m + d/dB_n)                      (translation invariance of EACH (bra, centre, ket-image) triple)

All three centres move with their atoms in every image: the ket at R_B + L and the ECP centre at R_C + M move with
R_B and R_C for every L, M.  The image sets So, Se are FROZEN (fixed L, M vectors chosen at the reference geometry),
so the analytic force is the exact derivative of a fixed partial sum; the FD anchor displaces atoms with the same sets.
V_ECP is symmetrised (1/2 (V + V^T)) before it enters h: a truncated set is not exactly symmetric, and E = sum D V is
unchanged by it (D symmetric), while the eigensolver would otherwise read one triangle only.

Everything else is Iteration 16/17's force (pbc_grad pieces REUSED: pair_ft_deriv, ewald_grad, _latsum_ip, _fold),
assembled here in G CHUNKS (pure AFT, w = None) so a large box fits in memory, in the per-spin form of Iteration 17:
    M = -sum_s D_s F_s D_s - v_M sum_s D_s S D_s ;  Z_mn(G) = 1/2 D_mn rho(G) - 1/2 sum_s (D_s P(G) D_s)_mn
(RHF: D_s = D/2, F_s = F).  F_s includes V_ECP, so W (and the Pulay term) sees it automatically.

_MUTANT / assemble_ecp_grad(mutant=...) switches (ECP term only):
    'no_centre'   drop the ECP-centre derivative
    'centre_sign' centre = +(bra + ket)
    'L0_only'     orbital image L = 0 only (ket images treated as unshifted) in the derivative
    'M0_only'     home ECP image only in the derivative
    'ket_transpose' (diagnostic, not a defect): ket term := bra term with m <-> n (the S/T shortcut), centre -2 bra
PySCF molecular ECP intors stand in for ferric's libecpint shim.
"""

from __future__ import annotations

import numpy as np
from pyscf import gto

import pbc_ecp as PE
import pbc_grad as PGd
from pbc_gamma import pair_ft

_MUTANT = None


# ----------------------------------------------------------------------------------------- image sets
def frozen_images(cell, prec=1e-10, rcut_ecp=None, rcut_orb=None):
    """Orbital / ECP image translation lists at the CURRENT (reference) geometry, to be reused unchanged."""
    r_e, r_o, _, _ = PE.ecp_ranges(cell, prec)
    rcut_ecp = rcut_ecp or r_e
    rcut_orb = rcut_orb or r_o
    return dict(Lorb=cell.translations(rcut_orb), Lecp=cell.translations(rcut_ecp), rcut_ecp=rcut_ecp,
                rcut_orb=rcut_orb)


# PySCF 2.13's molecular ECP code screens a shell pair by (roughly) its MOST DIFFUSE exponents and the two
# shell-to-ECP-centre distances, as if the integrand were concentrated at the ECP centre; with diffuse ECP radial
# terms (LANL2DZ I: exponent 0.86) that drops non-negligible integrals EXACTLY to zero (measured: a single
# primitive H p(3.425) at 3.0 Bohr x I s at 7.0 Bohr from the centre, true 1.774e-7 by independent radial x Lebedev
# quadrature, PySCF 0.0).  Guard: every shell of the ECP-integral molecules carries one extra primitive of exponent
# AUG_EXP with coefficient 0 -- the function is unchanged (normalisation included), the screen can no longer fire.
SCREEN_GUARD = True
AUG_EXP = 1e-3


def _aug_shells(shells):
    out = []
    for sh in shells:
        l, prims = sh[0], sh[1:]
        out.append([l] + [list(p) for p in prims] + [[AUG_EXP] + [0.0] * (len(prims[0]) - 1)])
    return out


def aug_basis(cell):
    """cell.basis with the zero-weight AUG_EXP primitive appended to every shell (per element)."""
    out = {}
    for sym, _ in cell.atoms:
        spec = cell.basis[sym] if isinstance(cell.basis, dict) else cell.basis
        out[sym] = _aug_shells(gto.basis.load(spec, sym) if isinstance(spec, str) else spec)
    return out


def _parity(atoms):
    return int(sum(gto.charge(s) for s, _ in atoms)) % 2


def _supermol(cell, imgs):
    """Image supermolecule at the cell's CURRENT positions: orbital images first, then the extra ECP images
    (pbc_ecp.ecp_images' layout).  Returns (sm, slot of every ECP image, ecpbas copy, image id / cell atom per row)."""
    Lorb, Lecp = imgs["Lorb"], imgs["Lecp"]
    key = lambda v: tuple(np.round(v, 8))  # noqa: E731
    pos = {key(L): i for i, L in enumerate(Lorb)}
    extra = [L for L in Lecp if key(L) not in pos]
    Lall = np.vstack([Lorb] + ([np.array(extra)] if extra else []))
    pos = {key(L): i for i, L in enumerate(Lall)}
    if SCREEN_GUARD:
        atoms = [(sym, np.asarray(r) + L) for L in Lall for (sym, r) in cell.atoms]
        sm = gto.M(atom=atoms, basis=aug_basis(cell), ecp=cell.ecp, unit="B", cart=True, verbose=0,
                   spin=_parity(atoms))
    else:
        sm = cell.ecp_supermol(Lall)
    nat = len(cell.atoms)
    eb = sm._ecpbas.copy()
    return sm, [pos[key(M)] for M in Lecp], eb, eb[:, gto.ATOM_OF] // nat, eb[:, gto.ATOM_OF] % nat


def _centres(cell, imgs):
    """(image index iM, cell atom C) for every ECP centre (M, C)."""
    ecp_atoms = sorted(set(int(a) for a in cell.mol._ecpbas[:, gto.ATOM_OF]))
    return [(iM, C) for iM in range(len(imgs["Lecp"])) for C in ecp_atoms]


def ecp_V(cell, imgs):
    """pbc_ecp-format image blocks V[M, m, L, n] over the frozen sets (feeds pbc_ecp.gamma_ecp(img=...))."""
    sm, slot, eb, img_of, at_of = _supermol(cell, imgs)
    nb0, nao, nL = cell.mol.nbas, cell.mol.nao, len(imgs["Lorb"])
    V = np.zeros((len(imgs["Lecp"]), nao, nL, nao))
    try:
        for iM in range(len(imgs["Lecp"])):
            rows = eb[img_of == slot[iM]]
            if len(rows) == 0:
                continue
            sm._ecpbas = rows
            V[iM] = sm.intor("ECPscalar_cart", shls_slice=(0, nb0, 0, nL * nb0)).reshape(nao, nL, nao)
    finally:
        sm._ecpbas = eb
    return dict(Lorb=imgs["Lorb"], Lecp=imgs["Lecp"], V=V)


def ecp_gamma(cell, imgs):
    """Symmetrised Gamma V_ECP and its raw asymmetry."""
    V = PE.ecp_k(ecp_V(cell, imgs), np.zeros(3))[0].real
    return 0.5 * (V + V.T), abs(V - V.T).max()


# ------------------------------------------------------------- derivative integrals from VALUE integrals
_FAC = {0: 0.282094791773878143, 1: 0.488602511902919921}  # libcint cart common factor (1 for l >= 2)


def _fac(l):
    return _FAC.get(l, 1.0)


def deriv_basis(cell, shifts):
    """Raised/lowered PRIMITIVE shells of the cell basis, one copy per shift (no ECP).  For every original shell
    (l, a_p) the Mole carries a single-primitive shell l+1 (exp a_p) and, if l >= 1, l-1 (exp a_p), unit input
    coefficient (PySCF stores gto_norm(l', a_p)).  Returns (dmol, R) with R[x] (nao, n_dao) such that for any ket
        <d/dA_x phi_m | U | psi> = sum_d R[x, m, d] <dmol_d(home copy) | U | psi>
    from d/dA_x (x^i e^{-a x^2}) = 2a x^{i+1} e - i x^{i-1} e (x relative to the centre A)."""
    from pbc_gamma import cart_comps

    mol = cell.mol
    labels = [f"{s}{i + 1}" for i, (s, _) in enumerate(cell.atoms)]
    basis = {lab: [] for lab in labels}
    for sh in range(mol.nbas):
        A, l = mol.bas_atom(sh), mol.bas_angular(sh)
        for a in mol.bas_exp(sh):
            extra = [[AUG_EXP, 0.0]] if SCREEN_GUARD else []
            basis[labels[A]].append([l + 1, [float(a), 1.0]] + extra)
            if l >= 1:
                basis[labels[A]].append([l - 1, [float(a), 1.0]] + extra)
    atoms = [(lab, np.asarray(r, float) + L) for L in shifts for lab, (_, r) in zip(labels, cell.atoms)]
    nelec = sum(gto.charge(s) for s, _ in cell.atoms) * len(shifts)
    dmol = gto.M(atom=atoms, basis=basis, unit="B", cart=True, verbose=0, spin=int(nelec) % 2)
    # locate home-copy shells by (atom, l, exponent): PySCF sorts shells by l within an atom
    nat = len(cell.atoms)
    home = {}
    aoloc = dmol.ao_loc_nr()
    for sh in range(dmol.nbas):
        if dmol.bas_atom(sh) >= nat:
            break
        home[(dmol.bas_atom(sh), dmol.bas_angular(sh), round(float(dmol.bas_exp(sh)[0]), 12))] = sh
    n_dao = aoloc[max(home.values()) + 1]
    R = np.zeros((3, mol.nao, n_dao))
    moloc = mol.ao_loc_nr()
    for sh in range(mol.nbas):
        A, l = mol.bas_atom(sh), mol.bas_angular(sh)
        exps = mol.bas_exp(sh)
        cenv = mol._env[mol._bas[sh, gto.PTR_COEFF] : mol._bas[sh, gto.PTR_COEFF] + len(exps)]  # nctr == 1 here
        assert mol.bas_nctr(sh) == 1
        comps = cart_comps(l)
        for p, a in enumerate(exps):
            for lp, fac_mul in ((l + 1, 2 * a), (l - 1, None)):
                if lp < 0:
                    continue
                dsh = home[(A, lp, round(float(a), 12))]
                assert dmol.bas_exp(dsh)[0] == a
                cd = dmol._env[dmol._bas[dsh, gto.PTR_COEFF]]  # the real primitive (the AUG one has weight 0)
                scale = _fac(l) * cenv[p] / (_fac(lp) * cd)
                dcomps = {c: i for i, c in enumerate(cart_comps(lp))}
                for u, (i, j, k) in enumerate(comps):
                    for x in range(3):
                        e = np.eye(3, dtype=int)[x]
                        ijk = np.array([i, j, k])
                        if lp == l + 1:
                            R[x, moloc[sh] + u, aoloc[dsh] + dcomps[tuple(ijk + e)]] += scale * fac_mul
                        elif ijk[x] > 0:
                            R[x, moloc[sh] + u, aoloc[dsh] + dcomps[tuple(ijk - e)]] -= scale * ijk[x]
    return dmol, R


# ----------------------------------------------------------------------------------- ECP gradient
KERNEL = "raised"  # 'raised' (value integrals of raised/lowered shells) | 'ipnuc' (PySCF ECPscalar_ipnuc: see FINDINGS 22)


def ecp_grad_blocks(cell, imgs, D, kernel=None):
    """Per ECP centre (M, C) and orbital image L, the D-contracted bra and ket derivatives (not yet folded):
        bra[k][L, m, x] = sum_n D_mn d/dA_m <m_0 | U_k | n_L>      (atom of m)
        ket[k][L, n, x] = sum_m D_mn d/dB_n <m_0 | U_k | n_L>      (atom of n; the ket image moves with B)
    kept separate so every mutant is post-processing of one evaluation.
    kernel 'raised': from ECPscalar VALUE integrals over raised/lowered primitive shells (deriv_basis) -- PySCF's
    value integrals agree with an independent radial x Lebedev quadrature to 1e-16, its ECPscalar_ipnuc does NOT
    (1.1e-7 on an H1s-H1s(L) element); kernel 'ipnuc' keeps the PySCF derivative intor for that comparison."""
    kernel = kernel or KERNEL
    sm, slot, eb, img_of, at_of = _supermol(cell, imgs)
    nb0, nao, nL = cell.mol.nbas, cell.mol.nao, len(imgs["Lorb"])
    cen = _centres(cell, imgs)
    bra = np.zeros((len(cen), nL, nao, 3))
    ket = np.zeros((len(cen), nL, nao, 3))
    Vg = np.zeros((nao, nao))
    if kernel == "raised":
        dmol, R = deriv_basis(cell, imgs["Lorb"])
        nbd = dmol.nbas // nL
        n_dao = R.shape[2]
        big = gto.conc_mol(sm, dmol)  # sm (with the ECP rows) first, so ECP atom ids are unchanged
        nbA = sm.nbas
        eb_big = big._ecpbas.copy()
        assert np.array_equal(eb_big[:, gto.ATOM_OF], eb[:, gto.ATOM_OF])
    else:
        big, eb_big = sm, eb
    try:
        for k, (iM, C) in enumerate(cen):
            sel = (img_of == slot[iM]) & (at_of == C)
            if not sel.any():
                continue
            big._ecpbas = eb_big[sel]
            if kernel == "raised":
                Vb = big.intor("ECPscalar_cart", shls_slice=(nbA, nbA + nbd, 0, nL * nb0))  # (n_dao, nL*nao)
                Vk = big.intor("ECPscalar_cart", shls_slice=(0, nb0, nbA, nbA + nL * nbd))  # (nao, nL*n_dao)
                Xd = np.einsum("xmd,dLn->xmLn", R, Vb.reshape(n_dao, nL, nao))
                Yd = np.einsum("mLd,xnd->xmLn", Vk.reshape(nao, nL, n_dao), R)
                bra[k] = np.einsum("xmLn,mn->Lmx", Xd, D)
                ket[k] = np.einsum("xmLn,mn->Lnx", Yd, D)
            else:
                X = big.intor("ECPscalar_ipnuc_cart", comp=3, shls_slice=(0, nb0, 0, nL * nb0)).reshape(3, nao, nL, nao)
                Y = big.intor("ECPscalar_ipnuc_cart", comp=3, shls_slice=(0, nL * nb0, 0, nb0)).reshape(3, nL, nao, nao)
                bra[k] = -np.einsum("xmLn,mn->Lmx", X, D)
                ket[k] = -np.einsum("xLnm,mn->Lnx", Y, D)
            Vg += big.intor("ECPscalar_cart", shls_slice=(0, nb0, 0, nL * nb0)).reshape(nao, nL, nao).sum(1)
    finally:
        big._ecpbas = eb_big
    L0 = np.linalg.norm(imgs["Lorb"], axis=1) < 1e-9
    M0 = np.array([np.linalg.norm(imgs["Lecp"][iM]) < 1e-9 for iM, _ in cen])
    return dict(bra=bra, ket=ket, cen=cen, L0=L0, M0=M0, asym=abs(Vg - Vg.T).max(), E=np.sum(D * Vg))


def assemble_ecp_grad(blocks, cell, mutant=None):
    """dE_ECP/dR (natm, 3) from ecp_grad_blocks; mutant defaults to the module _MUTANT."""
    mutant = _MUTANT if mutant is None else mutant
    mol = cell.mol
    aoat = PGd.ao_atom(mol)
    g = np.zeros((mol.natm, 3))
    Lmask = blocks["L0"] if mutant == "L0_only" else np.ones(len(blocks["L0"]), bool)
    for k, (_, C) in enumerate(blocks["cen"]):
        if mutant == "M0_only" and not blocks["M0"][k]:
            continue
        b = blocks["bra"][k][Lmask].sum(0)
        kt = b if mutant == "ket_transpose" else blocks["ket"][k][Lmask].sum(0)
        g += PGd._fold(b, aoat, mol.natm) + PGd._fold(kt, aoat, mol.natm)
        if mutant == "no_centre":
            continue
        sgn = 1.0 if mutant == "centre_sign" else -1.0
        g[C] += sgn * (b.sum(0) + kt.sum(0))
    return g


# ------------------------------------------------------------------------------------- SCF (energy)
def solve(cell, imgs, gcut, rcut_1e, exxdivs=("none", "ewald"), guess=None, tol=1e-11):
    """Pure-AFT Gamma integrals (pbc_ecp.gamma_ecp, Z_eff) with the frozen-image, symmetrised V_ECP, then a tight
    RHF (pbc_grad_open.scf, commutator < tol) per exxdiv.  guess: a total density D (it is never accepted as-is)."""
    import pbc_dft as pd
    import pbc_grad_open as GO
    from pbc_gamma import rhf

    g = PE.gamma_ecp(cell, "ewald", gcut=gcut, img=ecp_V(cell, imgs), rcut_1e=rcut_1e)
    Vs = 0.5 * (g["Vecp"] + g["Vecp"].T)
    h = g["h"] - g["Vecp"] + Vs
    jk = pd.dense_jk(g["I"])
    n = NELEC_OF(cell)
    out = {}
    for ex in exxdivs:
        vM = g["madelung"] if ex == "ewald" else 0.0
        if guess is None:
            C = rhf(g["S"], h, g["I"], g["enn"], n, conv=1e-10, kshift=vM, return_mo=True, maxiter=300)[3]
            D0 = 2 * C[:, : n // 2] @ C[:, : n // 2].T
        else:
            D0 = guess
        r = GO.scf(g["S"], h, jk, g["enn"], n // 2, n // 2, None, "HF", vM=vM, restricted=True,
                   guess=(0.5 * D0, 0.5 * D0), tol=tol)
        r.update(D=r["Da"] + r["Db"], vM=vM, S=g["S"], h=h, Vecp=Vs, nG=g["nG"])
        out[ex] = r
    return out


def NELEC_OF(cell):  # noqa: N802
    return int(round(cell.Z.sum()))  # neutral cell: electrons = sum Z_eff


# ----------------------------------------------------------------------------- total force (chunked)
def aft_grad_parts(cell, Da, Db, Fa, Fb, S, vM, gcut, rcut_1e, chunk=40000):
    """Iteration 16/17 pure-AFT (w = None) HF force pieces, G-chunked.  Returns dict of (natm, 3)."""
    mol = cell.mol
    natm, nao = mol.natm, mol.nao
    aoat = PGd.ao_atom(mol)
    D = Da + Db
    spins = ((Da, Fa), (Db, Fb))
    parts = {}
    M = -sum(Ds @ Fs @ Ds for Ds, Fs in spins) - vM * sum(Ds @ S @ Ds for Ds, _ in spins)
    XS = PGd._latsum_ip(cell, "int1e_ipovlp_cart", rcut_1e)
    XT = PGd._latsum_ip(cell, "int1e_ipkin_cart", rcut_1e)
    parts["S"] = PGd._fold(-2 * np.einsum("xmn,mn->mx", XS, M), aoat, natm)
    parts["T"] = PGd._fold(-2 * np.einsum("xmn,mn->mx", XT, D), aoat, natm)
    P0 = pair_ft(cell, np.zeros((1, 3)))[..., 0].real  # calibration exactly as pbc_kpts.gamma_aft
    nrm = np.sqrt(np.diag(S) / np.diag(P0))
    nn = nrm[:, None] * nrm[None, :]
    G = cell.gvectors(gcut)
    G = G[np.einsum("gi,gi->g", G, G) > 1e-12]
    vb = np.zeros((natm, 3))
    vn = np.zeros((natm, 3))
    il = np.zeros((nao, 3))
    for c in range(0, len(G), chunk):
        Gc = G[c : c + chunk]
        Praw, Q = PGd.pair_ft_deriv(cell, Gc)
        P = Praw * nn[:, :, None]
        Q = Q * nn[None, :, :, None]
        v = 4 * np.pi / np.einsum("gi,gi->g", Gc, Gc)
        SG_at = np.exp(-1j * Gc @ cell.R.T) * cell.Z
        SG = SG_at.sum(1)
        rho = np.einsum("mn,mng->g", D, P)
        rq = PGd._fold_g(2 * np.einsum("mn,xmng->mxg", D, Q), aoat, natm)
        vb -= np.einsum("g,axg->ax", v, np.conj(rq) * SG).real / cell.vol
        vn -= np.einsum("g,ga,gx->ax", v * np.conj(rho), SG_at * (-1j), Gc).real / cell.vol
        Zt = 0.5 * D[:, :, None] * rho[None, None, :] - 0.5 * sum(
            np.einsum("ml,lsg,sn->mng", Ds, P, Ds) for Ds, _ in spins)
        il += 4 / cell.vol * np.einsum("g,xmng,mng->mx", v, np.conj(Q), Zt).real
        del P, Q, Praw
    parts["Vlr_basis"], parts["Vlr_nuc"], parts["Ilr"] = vb, vn, PGd._fold(il, aoat, natm)
    parts["nn"] = PGd.ewald_grad(cell, 1.0)
    return parts


def gamma_ecp_grad(cell, imgs, r, gcut, rcut_1e):
    """Total Gamma force for a converged solve() entry r.  Returns (grad, parts, ecp blocks)."""
    parts = aft_grad_parts(cell, r["Da"], r["Db"], r["Fa"], r["Fb"], r["S"], r["vM"], gcut, rcut_1e)
    blocks = ecp_grad_blocks(cell, imgs, r["D"])
    parts["ecp"] = assemble_ecp_grad(blocks, cell)
    return sum(parts.values()), parts, blocks


def crosscheck_dense(cell, imgs, r, gcut, rcut_1e):
    """max | chunked non-ECP assembly - pbc_grad_open.gamma_grad (dense P, pbc_gamma.build_integrals) | at r's D."""
    import pbc_grad_open as GO
    from pbc_gamma import build_integrals

    ints = build_integrals(cell, None, rcut_1e=rcut_1e, verbose=False, exxdiv=None, gcut=gcut)
    ints["madelung"] = r["vM"]
    gd, _ = GO.gamma_grad(cell, ints, r["Da"], r["Db"], r["Fa"], r["Fb"], 1.0, rcut_1e=rcut_1e)
    pc = aft_grad_parts(cell, r["Da"], r["Db"], r["Fa"], r["Fb"], r["S"], r["vM"], gcut, rcut_1e)
    return abs(sum(pc.values()) - gd).max()


# --------------------------------------------------------------------------------------- FD anchors
def displaced(cell, A, x, h):
    atoms = [(s, np.array(p, float)) for s, p in cell.atoms]
    atoms[A] = (atoms[A][0], atoms[A][1] + h * np.eye(3)[x])
    return PE.EcpCell(cell.a, atoms, cell.basis, cell.ecp)


def fd_ecp_term(cell, imgs, D, comps, h=1e-4):
    """Central FD of E_ECP(R) = sum D V_ECP(R) at FIXED D, frozen image sets."""
    out = {}
    for A, x in comps:
        ep = np.sum(D * ecp_gamma(displaced(cell, A, x, h), imgs)[0])
        em = np.sum(D * ecp_gamma(displaced(cell, A, x, -h), imgs)[0])
        out[(A, x)] = (ep - em) / (2 * h)
    return out


def fd_total(cell, imgs, gcut, rcut_1e, comps, h=1e-4, guess=None):
    """Central FD of the total SCF energy (both exxdiv from one integral build per displacement)."""
    out = {}
    for A, x in comps:
        ep = solve(displaced(cell, A, x, h), imgs, gcut, rcut_1e, guess=guess)
        em = solve(displaced(cell, A, x, -h), imgs, gcut, rcut_1e, guess=guess)
        out[(A, x)] = {ex: (ep[ex]["e"] - em[ex]["e"]) / (2 * h) for ex in ep}
    return out


# ------------------------------------------------------------------------------- molecular references
def molecular_ecp_term(mol, D):
    """PySCF's own molecular ECP gradient term: bra ECPscalar_ipnuc + centre ECPscalar_iprinv at the rinv origin,
    symmetrised exactly as pyscf.grad.rhf.hcore_generator does it."""
    h1 = -mol.intor("ECPscalar_ipnuc", comp=3)
    ecp_atoms = set(mol._ecpbas[:, gto.ATOM_OF])
    g = np.zeros((mol.natm, 3))
    for A, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        vr = np.zeros_like(h1)
        if A in ecp_atoms:
            with mol.with_rinv_at_nucleus(A):
                vr = mol.intor("ECPscalar_iprinv", comp=3)
        vr[:, p0:p1] += h1[:, p0:p1]
        g[A] = np.einsum("xmn,mn->x", vr + vr.transpose(0, 2, 1), D)
    return g


def molecular_ecp_term_fd(atoms, basis, ecp, D, h=1e-4):
    def e(at):
        m = gto.M(atom=at, basis=basis, ecp=ecp, unit="B", cart=True, verbose=0)
        return np.sum(D * m.intor("ECPscalar"))

    g = np.zeros((len(atoms), 3))
    for A in range(len(atoms)):
        for x in range(3):
            ap = [(s, np.array(p, float) + (h * np.eye(3)[x] if i == A else 0)) for i, (s, p) in enumerate(atoms)]
            am = [(s, np.array(p, float) - (h * np.eye(3)[x] if i == A else 0)) for i, (s, p) in enumerate(atoms)]
            g[A, x] = (e(ap) - e(am)) / (2 * h)
    return g


def molecular_grad(mf):
    return mf.nuc_grad_method().kernel()


def c3_prime(atoms, basis, ecp, h=1e-3):
    """d/dR of the Iteration 14 box-limit coefficient c3 = -(4pi/3) Omega_I - (2pi/3)|p|^2 (central FD, z only:
    the reference molecule lies on z, transverse derivatives vanish by symmetry)."""
    out = np.zeros((len(atoms), 3))
    for A in range(len(atoms)):
        c = []
        for s in (1, -1):
            at = [(sy, np.array(p, float) + (s * h * np.eye(3)[2] if i == A else 0)) for i, (sy, p) in enumerate(atoms)]
            c.append(PE.molecular_reference(at, basis, ecp, conv=1e-12)["c3"])
        out[A, 2] = (c[0] - c[1]) / (2 * h)
    return out
