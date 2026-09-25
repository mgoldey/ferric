"""Iteration 21: analytic nuclear forces for k-point RHF / UHF (Gamma-centred mesh), dense AFT J/K (pbc_kpts).

Energy (per cell; pbc_kpts / pbc_kuhf conventions: chi_mk = sum_L e^{ik.L} phi_m(r - L), N_k mesh points, pure AFT,
only the K = G + q = 0 term dropped; v_M = the Madelung constant of the diag(n) SUPERCELL for exxdiv='ewald', 0 for none):

    E = (1/Nk) sum_k tr[h(k) D(k)] + (1/2 Omega) sum_{G!=0} v(G) |rho(G)|^2
        - (1/(2 Omega Nk^2)) sum_s sum_{k,k'} sum_{K in G+q, K!=0} v(K) tr[P^{kk'}(K) D_s(k') P^{kk'}(K)^+ D_s(k)]
        - (v_M/2) (1/Nk) sum_s sum_k tr[D_s(k) S(k) D_s(k) S(k)] + E_nn,

    D = D_a + D_b, D_s(k) = C_s,occ(k) C_s,occ(k)^+ (complex Hermitian; RHF: D_s = D/2), q = k' - k,
    rho(G) = (1/Nk) sum_k tr[P^{kk}(G) D(k)],  v(K) = 4 pi / |K|^2,  tr[X Y] = sum_mn X_mn Y_nm,
    h(k) = T(k) + V(k),  V(k)_mn = -(1/Omega) sum_{G!=0} v P^{kk}_mn(G) conj(S(G)),  S(G) = sum_A Z_A e^{-iG.R_A}.
    (Checked: this is exactly 0.5 (1/Nk) sum_k Re tr[(h + F_s) D_s] of pbc_kuhf, J/K from pbc_kpts.Jker/Kker.)

DERIVATION.  Moving atom A moves its nucleus and every basis function centred on it in EVERY lattice image.
All k dependence of the integrals sits in Bloch phases of IMAGE-RESOLVED molecular quantities:

    X(k)_mn = sum_L e^{ik.L} X_mn(L),  X_mn(L) = <phi_m | O | phi_n(. - L)>                      (S, T)
    P^{kk'}_mn(K) = sum_L e^{ik'.L} p^L_mn(K),  p^L_mn(K) = FT[phi_m phi_n(. - L)](K),  K = G + k' - k.

The phases do not depend on the atoms, so d/dR_A acts on X_mn(L) and p^L_mn(K) only.  Since e^{ik'.L} depends on L
only modulo the mesh, the image sum is folded into residues r (t_r = residue representative): p_r = sum_{L==r} p^L.

 1. One-electron / overlap terms.  (1/Nk) sum_k tr[dX(k) M(k)] = sum_L sum_mn dX_mn(L) Mt_nm(L),
        Mt(L) = (1/Nk) sum_k e^{ik.L} M(k)          (the real-space, supercell-periodic density; real for a
                                                     time-reversal-symmetric state, Re taken at the end)
    with d/dR_A X_mn(L) = <grad m_0|O|n_L> (-[m in A] + [n in A])   (bra: -grad; ket: +, integration by parts).
    M = D for T; for the overlap derivative
        M(k) = -W(k) - v_M sum_s D_s S D_s,   W(k) = sum_s D_s(k) F_s(k) D_s(k)
    (orthonormality C_s^+ S C_s = 1 per k and spin => -(1/Nk) sum_k tr[W dS]; the explicit S-dependence of the
     Madelung term -(v_M/2)(1/Nk) sum_s tr[D_s S D_s S] gives -(v_M/Nk) sum_s tr[dS D_s S D_s]).
    RHF: W = 1/2 D F D, Madelung M = -(v_M/2) D S D.  Prediction: F(ewald) == F(none) per k and spin, since
    D_s S D_s = D_s and the -v_M S D_s S piece of F_s puts -v_M D_s into W (as at Gamma, Iterations 16-17).
 2. q = 0 (J and V_ne).  drho(G) = sum_r sum_mn dp_r,mn(G) Delta_r,nm,  Delta_r = (1/Nk) sum_k e^{ik.t_r} D(k),
        dE_J + dE_V(basis) = (1/Omega) sum_G v Re[(conj rho - conj S(G)) drho],
        dE_V(nucleus)     = -(1/Omega) sum_G v Re[rho conj(dS(G)/dR_A)],  dS/dR_A = -iG Z_A e^{-iG.R_A}.
 3. Exchange (every q, every pair k = k' - q, k').  Only P depends on R:
        dE_K = -(1/(Omega Nk^2)) sum_s sum_{k,k'} sum_K v(K) Re tr[dP^{kk'}(K) D_s(k') P^{kk'}(K)^+ D_s(k)]
    (the two P's of the trace give complex-conjugate contributions: 2 Re, x 1/2).  Per q pass:
        Y^{k'}_nm(K) = sum_s [D_s(k') P^{k'-q,k'}(K)^+ D_s(k'-q)]_nm,  Yt_r = sum_k' e^{ik'.t_r} Y^{k'}
    so that sum_k' tr[dP^{k'-q,k'} Y^{k'}] = sum_r tr[dp_r Yt_r]: ONE residue-resolved derivative FT per q.
 4. Pair-FT derivative.  bra (m): Qb = d p_r / dA_m from the raised bra Hermite table (pbc_grad.pair_ft_deriv, but
    binned by residue).  ket (n): translation identity d/dA_m + d/dA_n = -iK, so Qk_r = -iK p_r - Qb_r.
 5. dE_nn: pbc_grad.ewald_grad (per cell, unchanged).
 Nk = 1 reduces term by term to pbc_grad.gamma_rhf_grad (Iteration 16, w = None); the N-mesh force equals the force on
 any single copy of A in the diag(n) supercell (E_cell = E_sc / N and all copies move together:
 dE_cell/dR_A = (1/N) sum_c dE_sc/dR_{A,c} = dE_sc/dR_{A,c} for each c by translation symmetry).

_MUTANT (module global) switches on deliberate defects:
    'no_phase'   Bloch phase dropped on every derivative block (Mt, Delta, Yt use phase 1)
    'conj_D'     D_s(k), M(k) conjugated inside the gradient (== using D(-k) at k; row/column-major slip)
    'wrong_q'    exchange derivative pairs k' with k = k' + q instead of k' - q (wrong k' - k)
    'gamma_vm'   explicit Madelung M-term with the PRIMITIVE-cell Gamma v_M instead of the mesh (supercell) v_M
    'no_invNk'   missing 1/Nk in Mt (the overlap / kinetic real-space density)
"""

from __future__ import annotations

import numpy as np

import pbc_grad as PGd
import pbc_kpts as PK
from pbc_gamma import cart_comps, hermite_E, madelung, pair_ft, shell_table
from pbc_kuhf import _aufbau, _diag, _dm, orthogonalizers
from pbc_supercell import Supercell

_MUTANT = None


# ------------------------------------------------------------------ residue-resolved pair FT + bra derivative
def pair_ft_deriv_residues(scell, K, thresh=1e-14):
    """(P, Qb): P[r,m,n,g] exactly pbc_supercell.pair_ft_residues (raw Cartesian), Qb[x,r,m,n,g] = d P / d A_m,x
    (bra centre only).  Same image set and pair screening as pair_ft_residues / pbc_grad.pair_ft_deriv."""
    prim = scell.prim
    sh = shell_table(prim.mol)
    nao, ng = prim.mol.nao, len(K)
    G2 = np.einsum("gi,gi->g", K, K)
    P = np.zeros((scell.R, nao, nao, ng), complex)
    Q = np.zeros((3, scell.R, nao, nao, ng), complex)
    amin = min(s["exps"].min() for s in sh)
    Ls = prim.translations(np.sqrt(2 * np.log(1 / thresh) / amin) + 2.0)
    rs = scell.residues(Ls)
    lmax = max(s["l"] for s in sh)
    powg = [np.stack([(-1j * K[:, d]) ** t for t in range(2 * lmax + 4)]) for d in range(3)]
    for sa in sh:
        ca, la = cart_comps(sa["l"]), sa["l"]
        for sb in sh:
            cb, lb = cart_comps(sb["l"]), sb["l"]
            for L, r in zip(Ls, rs):
                B = sb["A"] + L
                AB = sa["A"] - B
                for a, cA in zip(sa["exps"], sa["coef"]):
                    for b, cB in zip(sb["exps"], sb["coef"]):
                        p = a + b
                        if abs(cA * cB * np.exp(-a * b / p * AB @ AB)) < thresh:
                            continue
                        Pc = (a * sa["A"] + b * B) / p
                        gm = np.nonzero(G2 < 4 * p * (np.log(1 / thresh) + 10))[0]
                        if len(gm) == 0:
                            continue
                        common = cA * cB * (np.pi / p) ** 1.5 * np.exp(-G2[gm] / (4 * p) - 1j * (K[gm] @ Pc))
                        F, Fd = [], []
                        for d in range(3):
                            E = hermite_E(la + 1, lb, a, b, AB[d])
                            f = np.einsum("ijt,tg->ijg", E, powg[d][: E.shape[2], gm])
                            fd = 2 * a * f[1:]
                            fd[1:] -= np.arange(1, la + 1)[:, None, None] * f[:la]
                            F.append(f[: la + 1])
                            Fd.append(fd)
                        for u, (ax, ay, az) in enumerate(ca):
                            for v, (bx, by, bz) in enumerate(cb):
                                fx, fy, fz = F[0][ax, bx], F[1][ay, by], F[2][az, bz]
                                i, j = sa["off"] + u, sb["off"] + v
                                P[r, i, j, gm] += common * fx * fy * fz
                                Q[0, r, i, j, gm] += common * Fd[0][ax, bx] * fy * fz
                                Q[1, r, i, j, gm] += common * fx * Fd[1][ay, by] * fz
                                Q[2, r, i, j, gm] += common * fx * fy * Fd[2][az, bz]
    return P, Q


def image_ip(cell, intor, rcut_1e=22.0):
    """(L1, X[x,m,L,n]) = <grad m_0 | O | n_L> per lattice image (pbc_grad._latsum_ip without the L sum)."""
    mol = cell.mol
    nao, nb0 = mol.nao, mol.nbas
    L1 = cell.translations(rcut_1e)
    sm = cell.supermol(L1)
    X = sm.intor(intor, comp=3, shls_slice=(0, nb0, 0, sm.nbas)).reshape(3, nao, len(L1), nao)
    return L1, X


def _fold_bra_ket(Ymn, aoat, natm):
    """Ymn[x,m,n] = sum_images dX_mn . Mt_nm  ->  g[A,x] = Re( -sum_{m in A} + sum_{n in A} )."""
    g = np.zeros((natm, 3))
    np.add.at(g, aoat, -Ymn.sum(2).T.real)
    np.add.at(g, aoat, Ymn.sum(1).T.real)
    return g


# ------------------------------------------------------------------------------ tight k SCF
def kscf(kb, na, nb, restricted=False, guess=None, tol=1e-11, maxiter=400, kshift=None, lindep=1e-8, ndiis=8, jk=None):
    """Tight k-point RHF (restricted: D_a = D_b, one diagonalisation) / UHF.  Global aufbau per spin over the mesh
    (pbc_kuhf._aufbau).  Stops on max|X^+[F_s,D_s S]X| < tol, never before one diagonalisation (a seed from another
    geometry is not S-normalised: Iteration 17 lesson).  guess: (Da, Db) stacks or None (core).
    jk: optional dm-stack -> (J, K) callable (e.g. pbc_kgdf.jk_from_kB) replacing the dense kernels.
    Returns dict(e, Da, Db, Fa, Fb, err, it, occ_a, occ_b, gap_a, gap_b, nocc_a_k, nocc_b_k)."""
    S, h, Nk = kb["S"], kb["h"], kb["Nk"]
    vm = kb["madelung"] if kshift is None else kshift
    jk = jk or (lambda dm: PK.jk_k(kb, dm))
    X = orthogonalizers(S, lindep)
    if guess is None:
        eps, C = _diag(h, X)
        oa, *_ = _aufbau(eps, na * Nk)
        ob, *_ = _aufbau(eps, nb * Nk)
        Da, Db = _dm(C, oa), _dm(C, ob)
    else:
        Da, Db = (np.array(g, complex) for g in guess)
    if restricted:
        assert na == nb
        Db = Da
    focks, errs = [], []
    for it in range(maxiter):
        Ja, Ka = jk(Da)
        if restricted:
            Jb, Kb = Ja, Ka
        else:
            Jb, Kb = jk(Db)
        J = Ja + Jb
        if vm:
            Ka = Ka + vm * np.einsum("kab,kbc,kcd->kad", S, Da, S)
            Kb = Kb + vm * np.einsum("kab,kbc,kcd->kad", S, Db, S)
        Fa, Fb = h + J - Ka, h + J - Kb
        e = 0.5 * (np.einsum("kmn,knm->", h + Fa, Da) + np.einsum("kmn,knm->", h + Fb, Db)).real / Nk + kb["enn"]
        pairs = ((Fa, Da),) if restricted else ((Fa, Da), (Fb, Db))
        err = [X[k].conj().T @ (F[k] @ D[k] @ S[k] - S[k] @ D[k] @ F[k]) @ X[k] for F, D in pairs for k in range(Nk)]
        emax = max(abs(x).max() for x in err)
        if it > 0 and emax < tol:
            break
        focks.append((Fa, Fb))
        errs.append(err)
        focks, errs = focks[-ndiis:], errs[-ndiis:]
        Fa_u, Fb_u = Fa, Fb
        if len(focks) > 1:
            m = len(focks)
            B = -np.ones((m + 1, m + 1))
            B[-1, -1] = 0
            for i in range(m):
                for j in range(m):
                    B[i, j] = sum(np.vdot(x, y).real for x, y in zip(errs[i], errs[j]))
            rhs = np.zeros(m + 1)
            rhs[-1] = -1
            c = np.linalg.lstsq(B, rhs, rcond=None)[0][:m]
            Fa_u = sum(ci * f[0] for ci, f in zip(c, focks))
            Fb_u = sum(ci * f[1] for ci, f in zip(c, focks))
        eps_a, Ca = _diag(Fa_u, X)
        oa, *_ = _aufbau(eps_a, na * Nk)
        Da = _dm(Ca, oa)
        if restricted:
            Db = Da
        else:
            eps_b, Cb = _diag(Fb_u, X)
            ob, *_ = _aufbau(eps_b, nb * Nk)
            Db = _dm(Cb, ob)
    else:
        raise RuntimeError(f"k-SCF not converged (err {emax:.2e})")
    eps_a, _ = _diag(Fa, X)
    eps_b, _ = _diag(Fb, X)
    oa, ha, la = _aufbau(eps_a, na * Nk)
    ob, hb, lb = _aufbau(eps_b, nb * Nk)
    return dict(e=e, Da=Da, Db=Db, Fa=Fa, Fb=Fb, err=emax, it=it, gap_a=la - ha, gap_b=lb - hb,
                nocc_a_k=[int(o.sum()) for o in oa], nocc_b_k=[int(o.sum()) for o in ob])


# ------------------------------------------------------------------------------ the force
def kgrad(cell, kb, Da, Db, Fa, Fb, restricted=False, thresh=1e-14, rcut_1e=22.0, mem_bytes=4e8, two_e=True):
    """dE/dR (natm, 3) per cell for k-point RHF (restricted=True: Da = Db = D/2, Fa = Fb = F, RHF formula branch) or
    UHF at converged (D_s(k), F_s(k)).  kb = pbc_kpts.build_k at the SAME geometry (supplies S, mesh, gcut, v_M).
    two_e=False: skip the J and K parts (the RS-GDF route supplies its own fitted 2e derivative)."""
    mol = cell.mol
    natm, nao = mol.natm, mol.nao
    aoat = PGd.ao_atom(mol)
    n, ints, kpts, Nk, gcut, vM = kb["n"], kb["ints"], kb["kpts"], kb["Nk"], kb["gcut"], kb["madelung"]
    S = kb["S"]
    if _MUTANT == "conj_D":
        Da, Db, Fa, Fb = Da.conj(), Db.conj(), Fa.conj(), Fb.conj()
    D = Da + Db
    spins = ((Da, Fa),) if restricted else ((Da, Fa), (Db, Fb))
    parts = {}
    vM_m = madelung(cell) if (_MUTANT == "gamma_vm" and vM) else vM

    # ---- 1. overlap-coupled + kinetic, image-resolved with Bloch phases
    if restricted:
        W = 0.5 * np.einsum("kab,kbc,kcd->kad", D, Fa, D)
        Mk = -W - 0.5 * vM_m * np.einsum("kab,kbc,kcd->kad", D, S, D)
    else:
        W = sum(np.einsum("kab,kbc,kcd->kad", Ds, Fs, Ds) for Ds, Fs in spins)
        Mk = -W - vM_m * sum(np.einsum("kab,kbc,kcd->kad", Ds, S, Ds) for Ds, _ in spins)
    L1, XS = image_ip(cell, "int1e_ipovlp_cart", rcut_1e)
    _, XT = image_ip(cell, "int1e_ipkin_cart", rcut_1e)
    ph1 = np.ones((Nk, len(L1))) if _MUTANT == "no_phase" else np.exp(1j * kpts @ L1.T)
    nrm1 = 1.0 if _MUTANT == "no_invNk" else 1.0 / Nk
    MtS = np.einsum("kL,knm->Lnm", ph1, Mk) * nrm1
    MtT = np.einsum("kL,knm->Lnm", ph1, D) * nrm1
    parts["S"] = _fold_bra_ket(np.einsum("xmLn,Lnm->xmn", XS, MtS), aoat, natm)
    parts["T"] = _fold_bra_ket(np.einsum("xmLn,Lnm->xmn", XT, MtT), aoat, natm)

    # ---- pair-FT normalisation (as build_k), residue geometry
    P0 = pair_ft(cell, np.zeros((1, 3)))[..., 0].real
    L0 = cell.translations(rcut_1e)
    sm = cell.supermol(L0)
    S0 = sm.intor("int1e_ovlp_cart", shls_slice=(0, mol.nbas, 0, sm.nbas)).reshape(nao, len(L0), nao).sum(1)
    nrm = np.sqrt(np.diag(S0) / np.diag(P0))
    nn = nrm[:, None] * nrm[None, :]
    scell = Supercell(cell.a, cell.atoms, cell.basis, n)
    ph_r = np.ones((Nk, scell.R)) if _MUTANT == "no_phase" else np.exp(1j * kpts @ scell.t.T)  # (k', r)
    ph_true = np.exp(1j * kpts @ scell.t.T)  # energy-side phase (never mutated)

    # ---- 2. q = 0: J + V_ne
    Delta = np.einsum("kr,knm->rnm", ph_r, D) / Nk
    Delta_true = np.einsum("kr,knm->rnm", ph_true, D) / Nk
    SGfun = lambda Kc: np.exp(-1j * Kc @ cell.R.T) * cell.Z  # noqa: E731  (g, atom)
    g_j = np.zeros((natm, 3))
    g_vb = np.zeros((natm, 3))
    g_vnuc = np.zeros((natm, 3))
    g_k = np.zeros((natm, 3))
    chunk = max(200, int(mem_bytes / (16 * 4 * scell.R * nao * nao)))
    for iq in range(Nk if two_e else 1):
        q = kpts[iq]
        K = PK.shifted_gvectors(cell, q, gcut)
        sgn = +1 if _MUTANT == "wrong_q" else -1
        kof = [PK.mesh_index(n, ints[j] + sgn * ints[iq]) for j in range(Nk)]  # k = k' - q (pair k, k' = j)
        for c0 in range(0, len(K), chunk):
            Kc = K[c0 : c0 + chunk]
            v = 4 * np.pi / np.einsum("gi,gi->g", Kc, Kc)
            Pr, Qb = pair_ft_deriv_residues(scell, Kc, thresh)
            Pr *= nn[None, :, :, None]
            Qb *= nn[None, None, :, :, None]
            Qk = -1j * Kc.T[:, None, None, None, :] * Pr[None] - Qb
            # Qk[x,r,m,n,g] is d p_r,mn / d A_n,x
            if iq == 0:
                rho = np.einsum("rmng,rnm->g", Pr, Delta_true)
                SGa = SGfun(Kc)
                wts = ((v * rho.conj() / cell.vol, g_j),) if two_e else ()
                for cw, gacc in wts + ((-v * SGa.sum(1).conj() / cell.vol, g_vb),):
                    gb = np.einsum("g,xrmng,rnm->mx", cw, Qb, Delta).real
                    gk = np.einsum("g,xrmng,rnm->nx", cw, Qk, Delta).real
                    np.add.at(gacc, aoat, gb + gk)
                g_vnuc += -(np.einsum("g,ga,gx->ax", v * rho, (-1j * SGa).conj(), Kc).real) / cell.vol
            if not two_e:
                continue
            # exchange: Y^{k'} = sum_s D_s(k') P^{k,k'}^+ D_s(k),  P^{k,k'} = sum_r e^{ik'.t_r} p_r
            Pj = np.einsum("jr,rmng->jmng", ph_true, Pr)
            Yt = np.zeros((scell.R, nao, nao, len(Kc)), complex)
            for j in range(Nk):
                kk = kof[j]
                if restricted:
                    Y = 0.5 * np.einsum("nl,slg,sm->nmg", D[j], Pj[j].conj(), D[kk])  # RHF: sum_s (D/2)P^+(D/2)
                else:
                    Y = sum(np.einsum("nl,slg,sm->nmg", Ds[j], Pj[j].conj(), Ds[kk]) for Ds, _ in spins)
                Yt += ph_r[j][:, None, None, None] * Y[None]
            cx = -v / (cell.vol * Nk * Nk)
            gb = np.einsum("g,xrmng,rnmg->mx", cx, Qb, Yt).real
            gk = np.einsum("g,xrmng,rnmg->nx", cx, Qk, Yt).real
            np.add.at(g_k, aoat, gb + gk)
    parts["J"] = g_j
    parts["V_basis"] = g_vb
    parts["V_nuc"] = g_vnuc
    parts["K"] = g_k
    parts["nn"] = PGd.ewald_grad(cell, 1.0)
    return sum(parts.values()), parts


# ------------------------------------------------------------------------------ drivers
def run(cell, n, na, nb, exxdiv, gcut, restricted=False, guess=None, thresh=1e-14, tol=1e-11, kb=None):
    kb = kb or PK.build_k(cell, n, exxdiv=exxdiv, gcut=gcut, thresh=thresh, verbose=False)
    r = kscf(kb, na, nb, restricted=restricted, guess=guess, tol=tol)
    g, parts = kgrad(cell, kb, r["Da"], r["Db"], r["Fa"], r["Fb"], restricted=restricted, thresh=thresh)
    return dict(e=r["e"], grad=g, parts=parts, scf=r, kb=kb)


def energy(cell, n, na, nb, exxdiv, gcut, restricted=False, guess=None, thresh=1e-14, tol=1e-11):
    kb = PK.build_k(cell, n, exxdiv=exxdiv, gcut=gcut, thresh=thresh, verbose=False)
    return kscf(kb, na, nb, restricted=restricted, guess=guess, tol=tol)


def fd(cell, n, na, nb, exxdiv, gcut, comps, guess, restricted=False, h=1e-4, thresh=1e-14):
    """Central FD of the prototype's own k-point energy; returns {(A,x): (fd, info)} with state-continuity info."""
    out = {}
    for A, x in comps:
        rp = energy(PGd.displaced(cell, A, x, h), n, na, nb, exxdiv, gcut, restricted, guess, thresh)
        rm = energy(PGd.displaced(cell, A, x, -h), n, na, nb, exxdiv, gcut, restricted, guess, thresh)
        out[(A, x)] = ((rp["e"] - rm["e"]) / (2 * h),
                       dict(nocc=(rp["nocc_a_k"], rm["nocc_a_k"], rp["nocc_b_k"], rm["nocc_b_k"]),
                            gap=min(rp["gap_a"], rm["gap_a"], rp["gap_b"] if nb else 9, rm["gap_b"] if nb else 9)))
    return out
