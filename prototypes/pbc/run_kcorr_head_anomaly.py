"""k-point MP2 q = 0 head anomaly (Iteration 12): why does restoring the q = 0 ERI head remove ~100% of the MP2
1/N_k term when the molecular ERI/Fock split predicted 78% (dRPA: 83-85% measured vs 80.5% predicted)?

Modes
  python3 run_kcorr_head_anomaly.py mol                 molecular split by an INDEPENDENT construction (H2)
  python3 run_kcorr_head_anomaly.py mesh a n [n ...]    mesh decomposition (H1/H4/H5), one npz per n
  python3 run_kcorr_head_anomaly.py eps a n [n ...]     HF eigenvalues at Gamma vs n (empirical Fock head)
  python3 run_kcorr_head_anomaly.py phi                 lattice constants Phi_p (H6)
  python3 run_kcorr_head_anomaly.py account a [E_inf_MP2 E_inf_dRPA] n [n ...]   E1/E2/E3 accounting from the npz rows
  HEAD_ORIENT=111 (env) rotates the H2 bond onto [111] for mesh/account.  Data: head_anomaly_data/*.npz

Notation (per cell, shifted = exxdiv 'ewald' occupied eps - v_M, as run_kcorr_convergence.py):
  S(n)  = E(shifted, no head)                      R(n) = S(n) - E_inf       (measured R n^3 ~ 2.2e-3 MP2)
  Hc    = E(cubic head, the Iteration-12 construction) - S         (V_head = (4pi/3 Omega) g_ia . conj(g_bj))
  Hc(l) = same with the head scaled by l: MP2 is exactly quadratic in l -> Hc = L + Q (linear / quadratic parts)
  Hleb  = <E(head(u))>_u - S, u on a Lebedev sphere, head(u) = (4pi/Omega)(u.g_ia) conj(u.g_bj): the exact
          direction average of the q -> 0 integrand (the cubic head replaces <h^2> by <h>^2)
  F     = E(shifted, occupied/virtual eps corrected by the missing q = 0 EXCHANGE head, no ERI head) - S
          dK_mn(k) = (4pi/(Omega Nk)) sum_ls <lim_{K->0}(P_ml(K) conj P_ns(K) - S_ml conj S_ns)/K^2>_cubic dm_ls,
          eps_true = eps_mesh - 0.5 <p|dK|p>   (first order, frozen C; fixed-k P^{kk}(K) like the ERI head)

Molecular (a = 6, Iteration 3/4 frozen-orbital model, k = 4pi/3a^3):  c3/a^3 = ERI 2.440e-3 + Fock 6.66e-4 (MP2),
ERI 3.440e-3 + Fock 8.32e-4 (dRPA).  Measured mesh: R n^3 = 2.20e-3 (MP2) / 3.85e-3 (dRPA); after the cubic head
~0 (|c| < 4e-5, MP2) / 5.6e-4 (dRPA).

HYPOTHESES AND WHAT EACH PREDICTS (written before the mesh runs; only the mol mode had been run)
 H1  "the Fock-head share is already inside the Madelung-shifted mesh denominators" (i.e. the mesh eigenvalues carry
     NO O(1/Nk) error beyond v_M):  eps-mode Gamma eigenvalue differences eps(3)-eps(4) are << the fixed-k head
     prediction dE(3)-dE(4) (occupied -k s2 (1/27-1/64), virtual +k|d|^2 (1/27-1/64) scale), and F n^3 ~ 0.
     NOT H1 (Fock head present, the physics of Iteration 9's HF -(4pi/3)Omega_I and the pbc_kpts K build that drops
     only the K = 0 term and corrects ONLY the occupied monopole): eigenvalue differences match the fixed-k
     prediction to its own approximation (flat band, few %), and F n^3 ~ -6.7e-4 (MP2) / -8.3e-4 (dRPA).
     Distinguishable: 0 vs 2.5e-5 in F at n = 3 (MP2); eigenvalue errors ~1e-3 at n = 3 vs ~0.
 H2  "the molecular c3 split is wrong for MP2": an independent molecular construction (harmonic kernel switched on
     ONLY in the SCF -> Fock part, ONLY in the correlation ERIs -> ERI part, orbitals free to relax; no closed form)
     gives an ERI share != 0.527/0.671 (78.5%).  NOT H2: it reproduces 0.527138 / 0.143956 to finite-difference
     precision (1e-6) and the dRPA 0.742969 / 0.179771 likewise.
 H4  "the Iteration-12 head is not the ERI part of the molecular split": flat-band algebra (derived in the FINDINGS
     section) says the cubic head adds L + Q = 2 V_reg(0) h/D + 2 h^2/D per Nk with V_reg(0) = V_iso - h (Lorentz),
     so L + Q = 2 V_iso h / D = exactly the molecular ERI part; and the TRUE q -> 0 average has 2<h^2> instead of
     2<h>^2 (<h^2> = 9/5 <h>^2 for one dipole direction), so the cubic head UNDER-removes by the anisotropy
     A = 2(<h^2>-<h>^2)/D per Nk (sign: leaves a POSITIVE residual).  Prediction (flat band): Hc n^3 = -2.44e-3,
     (Hleb - Hc) n^3 = -1.6 k^2 d^4 a^0 / |D|-scale (numbers printed by mol mode), residual after Hleb + F ~ 0.
     If instead Hc n^3 ~ -R n^3 = -2.2e-3 while F n^3 ~ -6.7e-4, then S + Hc + F overshoots E_inf by ~F: the anomaly is
     then NOT that the head removes the Fock share, but that the shifted residual lacks ~(F + A) that flat-band
     theory adds -- a missing NEGATIVE contribution X = R - (-Hleb) - (-F) that the mesh accounting isolates.
 H3  "accidental cancellation specific to a = 6": X/F differs at another cell (run at a = 10, not 8: the build cost
     barely grows with a; h/V_iso smaller by 0.22, band
     dispersion exponentially smaller); if X is a dispersion (band-width) effect it vanishes at a = 8 and the head
     residual there becomes ~F; if X scales like the Fock head it stays ~ -F at every a.
 H5  (added after the n = 1..3 rows) "the fixed-k head/Fock vectors are not the true (Bloch-paired) q -> 0 limit
     at a = 6": berry_pairs compares C(k)^H P^{k,k+b}(b) C(k+b) with the fixed-k pairing at the SAME finite b.
     Predicts true/fixed head ratio << 1 if H5 explains the missing ~0.86e-3.
 H6  (added after the n = 1..3 rows and the lattice-sum computation, BEFORE the n = 4, a = 10 n = 4 and [111]
     rows): the q -> 0 integrand's degree-0 part f0(q^) = F((q^.n)^2) - F(0) is corrected by the mesh NOT with its
     angular average but with Phi[f0] = -Z[f0], Z the Gaussian-regularised cubic-lattice sum (lattice_phi):
     Phi(u_z^2p) = 1, 1/3, -0.1716, -0.3484, -0.5482 (p = 0..4) vs <.> = 1, 1/3, 1/5, 1/7, 1/9 (cross-checked by the
     Epstein-zeta functional equation to 3e-5).  Linear-in-h terms are l <= 2 and exact under the cubic average;
     the O(h^2) term of MP2 (|V_reg + h|^2) is l = 4 and is NOT.  E3 = S + Phi-head + F should be flat in n.
     PREDICTIONS (fixed before those runs):
       a = 6 (z) n = 4: E3 - E_inf ~ -7e-5 / n^3 (the n = 3 residual) and E1 flat as measured before.
       a = 10 (z) n = 4: E1(4) - E1(3) = -2.5e-6 (MP2) / -3.6e-6 (dRPA); E3(4) - E3(3) ~ 0 (|.| < 1e-6).
       a = 6, H2 along [111] (same centre, same bond; Phi_111 = 1, 1/3, +0.4477, +0.5081, +0.4839): the l = 4 term
       now ADDS to the Fock head instead of cancelling it: MP2 head-restored residual c ~ +1.4e-3 (the cubic head
       removes ~61%, not ~100%); dRPA ~73%.  E3 (with Phi_111) flat; E1 drifts by ~+3e-5 between n = 3 and 4 (MP2).
       Alternative (the ~100% is structural for MP2, not accidental): E1 flat again for [111].
Artifact hypotheses / anchors (run before any interpretation):
  A1 l = 1 cubic head reproduces the Iteration-12 head-restored numbers (-1.3356869e-2 MP2 at n = 3) to 1e-12.
  A2 vectorised MP2 == pbc_kcorr.kmp2 to 1e-14 (no head, head).
  A3 E(l) is exactly quadratic for MP2 (4-point check); small-l slope of the Lebedev head == cubic head slope
     (the linear parts are identical by construction: <u u^T> = I/3).
  A4 fixed-k P^{kk}(K = 0) == S(k) (the Madelung term is v_M S dm S, so the dK subtraction is consistent).
  A5 mutation: dK with the second-derivative (S B) term removed must change the OCCUPIED shift only; with the
     A A* term removed must zero the VIRTUAL shift (nocc = nvir = 1).
"""
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import pbc_kcorr as KC  # noqa: E402
import pbc_kpts as PK  # noqa: E402
from pbc_gamma import Cell, rhf  # noqa: E402
from pbc_kpts import mesh_index, mp_mesh  # noqa: E402
from pbc_supercell import Supercell, pair_ft_residues  # noqa: E402
from run_kpts_anchor import H2_ATOMS  # noqa: E402

import os  # noqa: E402

# H3/H6 orientation test: HEAD_ORIENT=111 rotates the H2 bond (same centre, same 1.4 Bohr) from z onto [111]
_ORIENT = os.environ.get("HEAD_ORIENT", "z")
if _ORIENT == "111":
    _c, _u = np.array([0.3, 0.2, 0.8]), np.ones(3) / np.sqrt(3)
    H2_ATOMS = [("H", tuple(_c - 0.7 * _u)), ("H", tuple(_c + 0.7 * _u))]
_SUF = "" if _ORIENT == "z" else f"_{_ORIENT}"


# ------------------------------------------------------------------------------ molecular, independent construction
def molecular_split_independent(h=1e-4):
    """c3 = d/dk E at k = 0 with the harmonic kernel (k/2)|r-r'|^2 (r2_kernel_c3's dI, dV) switched on separately
    in the SCF (Fock part: eps AND orbitals relax) and in the correlation ERIs (ERI part).  x 4pi/3 -> c3."""
    from pyscf import gto

    from pbc_mp2 import gamma_mp2
    from pbc_rpa import gamma_drpa

    mol = gto.M(atom=H2_ATOMS, basis="sto-3g", unit="B", cart=True, verbose=0)
    S = mol.intor("int1e_ovlp")
    r = mol.intor("int1e_r")
    r2 = np.einsum("xxmn->mn", mol.intor("int1e_rr").reshape(3, 3, mol.nao, mol.nao))
    I0 = mol.intor("int2e")
    h0 = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    Z, R = mol.atom_charges().astype(float), mol.atom_coords()
    dI = 0.5 * (np.einsum("mn,ls->mnls", S, r2) + np.einsum("mn,ls->mnls", r2, S) - 2 * np.einsum("xmn,xls->mnls", r, r))
    dV = -0.5 * sum(z * (r2 - 2 * np.einsum("x,xmn->mn", Ra, r) + (Ra @ Ra) * S) for z, Ra in zip(Z, R))

    def energies(ks, kc):
        e, eps, _, C = rhf(S, h0 + ks * dV, I0 + ks * dI, mol.energy_nuc(), 2, conv=1e-13, return_mo=True)
        I = I0 + kc * dI
        return np.array([gamma_mp2(C, eps, 1, eri=I)[0], gamma_drpa(C, eps, 1, eri=I, method="plasmon"),
                         eps[0], eps[1]])

    f = 4 * np.pi / 3
    tot = (energies(h, h) - energies(-h, -h)) / (2 * h) * f
    fock = (energies(h, 0) - energies(-h, 0)) / (2 * h) * f
    eri = (energies(0, h) - energies(0, -h)) / (2 * h) * f
    # closed-form ingredients (frozen orbitals)
    e, eps, _, C = rhf(S, h0, I0, mol.energy_nuc(), 2, conv=1e-13, return_mo=True)
    ci, ca = C[:, 0], C[:, 1]
    d = np.einsum("xm,m->x", r @ ca, ci)
    cen = np.einsum("xmn,m,n->x", r, ci, ci)
    s2 = np.einsum("mn,m,n->", r2, ci, ci) - cen @ cen
    V = mol.ao2mo(C, compact=False).reshape(2, 2, 2, 2)[0, 1, 0, 1]
    return dict(tot=tot, fock=fock, eri=eri, d2=d @ d, s2=s2, V=V, D=eps[1] - eps[0], eps=eps)


def run_mol():
    x = molecular_split_independent()
    for name, j in (("MP2", 0), ("dRPA", 1)):
        print(f"{name}: c3 total {x['tot'][j]:.6f} = ERI {x['eri'][j]:.6f} + Fock {x['fock'][j]:.6f} "
              f"(sum {x['eri'][j] + x['fock'][j]:.6f}); ERI share {x['eri'][j] / x['tot'][j]:.4f}")
    f = 4 * np.pi / 3
    print(f"d eps_i = {x['fock'][2]:.6f}, d eps_a = {x['fock'][3]:.6f} (x a^-3; each carries the common Bethe constant); "
          f"d(eps_a - eps_i) = {x['fock'][3] - x['fock'][2]:.6f} vs closed form (4pi/3)(s2 + |d|^2) = "
          f"{f * (x['s2'] + x['d2']):.6f}")
    a = 6.0
    k = 4 * np.pi / (3 * a**3)
    V, d2, Dm = x["V"], x["d2"], 2 * x["D"]  # MP2 denominator magnitude |2 eps_i - 2 eps_a|
    lin = 2 * V * k * d2 / Dm
    aniso = 2 * (9 / 5 - 1) * (k * d2) ** 2 / Dm
    print(f"a = 6: V_iso {V:.6f} |d|^2 {d2:.6f} s2 {x['s2']:.6f} D_mp2 {-Dm:.6f} k {k:.6e} h = k|d|^2 {k * d2:.6e} "
          f"(h / V_iso {k * d2 / V:.4f})")
    print(f"flat-band MP2 per Nk: ERI linear 2 V h/|D| = {lin:.4e} (= mol ERI/a^3 {x['eri'][0] / a**3:.4e}); "
          f"anisotropy 2(<h^2>-<h>^2)/|D| = {aniso:.4e} ({aniso / lin:.3f} of ERI); Fock {x['fock'][0] / a**3:.4e}")
    print(f"a = 8: h/V_iso {4 * np.pi / (3 * 512) * d2 / V:.4f}")


# ------------------------------------------------------------------------------ mesh helpers
def pair_derivs(cell, n, delta=1e-4, delta2=2e-3, thresh=1e-14):
    """Fixed-k P^{kk}(K) near K = 0: P0 (Nk,nao,nao), A[..., a] = dP/dK_a, B[..., a] = d2P/dK_a^2 (per axis)."""
    ints, kpts = mp_mesh(cell, n)
    scell = Supercell(cell.a, cell.atoms, cell.basis, n)
    ph_r = np.exp(1j * kpts @ scell.t.T)
    nn = KC._pair_norm(cell)
    Ks = np.vstack([np.zeros((1, 3))] + [s * dl * np.eye(3) for dl in (delta, delta2) for s in (1, -1)])
    Q = pair_ft_residues(scell, Ks, thresh) * nn[None, :, :, None]
    P = np.einsum("jr,rmng->jmng", ph_r, Q)
    P0 = P[..., 0]
    A = (P[..., 1:4] - P[..., 4:7]) / (2 * delta)
    B = (P[..., 7:10] + P[..., 10:13] - 2 * P0[..., None]) / delta2**2
    return P0, A, B


def with_head(st, P, lam=1.0):
    """Copy of st (V, Kq) with a q = 0 head added: P (Nk, nao, nao, naux) plays sqrt(v) rho (already scaled)."""
    Nk = st["Nk"]
    new = dict(st)
    new["V"], new["Kq"] = st["V"].copy(), st["Kq"].copy()
    no = st["V"].shape[3]
    KC._accumulate(new, 0, list(range(Nk)), np.sqrt(lam) * P, st["Co"], st["Cv"], Nk)
    assert new["V"].shape[3] == no
    return new


def kmp2_vec(st, eo, ev):
    V, n, ints, Nk = st["V"], st["n"], st["ints"], st["Nk"]
    ii = np.arange(Nk)
    KB = np.array([[[mesh_index(n, ints[a] + ints[b] - ints[c]) for c in ii] for b in ii] for a in ii])
    eo, ev = np.array(eo), np.array(ev)
    X = V[ii[:, None, None], ii[None, :, None], KB].transpose(0, 1, 2, 3, 6, 5, 4)
    d = (eo[:, None, None, :, None, None, None] - ev[None, None, :, None, :, None, None]
         + eo[None, :, None, None, None, :, None] - ev[KB][:, :, :, None, None, None, :])
    t = V.conj() / d
    return np.sum(t * (2 * V - X)).real / Nk**3


def fock_head_shifts(cell, st, P0, A, B, C, nocc, mutant=None):
    """First-order eps corrections (mesh - true) from the missing q = 0 K = 0 exchange term beyond v_M S dm S."""
    Nk = st["Nk"]
    pref = 4 * np.pi / (cell.vol * Nk)
    deo, dev = [], []
    for k in range(Nk):
        Ck = C[k]
        dm = 2 * Ck[:, :nocc] @ Ck[:, :nocc].conj().T
        S = P0[k]
        M = np.zeros((S.shape[0],) * 4, complex)
        for ax in range(3):
            if mutant != "no_AA":
                M += np.einsum("ml,ns->mlns", A[k, ..., ax], A[k, ..., ax].conj())
            if mutant != "no_SB":
                M += 0.5 * (np.einsum("ml,ns->mlns", B[k, ..., ax], S.conj())
                            + np.einsum("ml,ns->mlns", S, B[k, ..., ax].conj()))
        M /= 3
        dK = pref * np.einsum("mlns,ls->mn", M, dm)
        de = 0.5 * np.einsum("mp,mn,np->p", Ck.conj(), dK, Ck).real
        deo.append(de[:nocc])
        dev.append(de[nocc:])
    return deo, dev



# ------------------------------------------------------------------------------ H5: Bloch-paired (true) q -> 0 limit
def berry_pairs(cell, n, C, thresh=1e-14):
    """Finite-b pair densities at the smallest mesh q = +-b_x,y,z (|b| = 2pi/(n a), cubic), with the TRUE Bloch
    pairing M(k, b) = C(k)^H P^{k,k+b}(b) C(k+b) (the q -> 0 integrand the BZ quadrature actually approaches,
    Berry/k.p term included) and the FIXED-k pairing C(k)^H P^{kk}(b) C(k) (what the head construction uses, at the
    same finite b; its b -> 0 limit is pair_derivs).  C(k+-b) is parallel-transported onto C(k) column by column
    (phase of the diagonal overlap), so the central difference is gauge-consistent.
    Returns (Mt, Mf, bnorm): Mt/Mf (6, Nk, nmo, nmo), direction order +x,+y,+z,-x,-y,-z."""
    ints, kpts = mp_mesh(cell, n)
    Nk = len(kpts)
    scell = Supercell(cell.a, cell.atoms, cell.basis, n)
    ph_r = np.exp(1j * kpts @ scell.t.T)
    nn = KC._pair_norm(cell)
    dirs = [(ax, sg) for sg in (1, -1) for ax in range(3)]
    Ks = np.array([sg * cell.b[ax] / n[ax] for ax, sg in dirs])
    Q = pair_ft_residues(scell, Ks, thresh) * nn[None, :, :, None]
    nmo = C[0].shape[1]
    Mt = np.zeros((6, Nk, nmo, nmo), complex)
    Mf = np.zeros_like(Mt)
    for d, (ax, sg) in enumerate(dirs):
        dm = np.zeros(3, int)
        dm[ax] = sg
        for k in range(Nk):
            kp = mesh_index(n, ints[k] + dm)
            Pt = np.einsum("r,rmn->mn", ph_r[kp], Q[..., d])
            Pf = np.einsum("r,rmn->mn", ph_r[k], Q[..., d])
            M = C[k].conj().T @ Pt @ C[kp]
            ph = np.diag(M).conj() / abs(np.diag(M))
            Mt[d, k] = M * ph[None, :]
            Mf[d, k] = C[k].conj().T @ Pf @ C[k]
    return Mt, Mf, np.linalg.norm(Ks[0])


def heads_from_pairs(M, bnorm, nocc):
    """Central-difference ov / vo head vectors (Nk, no, nv, 3), (Nk, nv, no, 3) from +-b pair densities."""
    g = (M[:3] - M[3:]) / (2 * bnorm)  # (3, Nk, nmo, nmo)
    g = g.transpose(1, 2, 3, 0)
    return g[:, :nocc, nocc:, :], g[:, nocc:, :nocc, :]


def fock_from_pairs(cell, M, bnorm, nocc, Nk):
    """First-order eps corrections (mesh - true) from finite-b pair densities, cubic average over +-x,y,z."""
    pref = 4 * np.pi / (cell.vol * Nk)
    w = abs(M) ** 2  # (6, Nk, nmo, nmo)
    occ = w[:, :, :nocc, :nocc].sum(3)  # sum over occupied partner
    vir = w[:, :, nocc:, :nocc].sum(3)
    eye = np.ones(nocc)
    deo = pref * ((occ - eye).mean(0)) * 3 / bnorm**2 / 3  # (1/3) sum_axes of the +- average = mean over 6 dirs
    dev = pref * vir.mean(0) / bnorm**2
    return list(deo), list(dev)


def with_head_mo(st, Bov, Bvo, lam=1.0):
    """Copy of st with an MO-level q = 0 head: Bov (Nk,no,nv,naux), Bvo (Nk,nv,no,naux), sqrt(v) included."""
    Nk = st["Nk"]
    new = dict(st)
    new["V"], new["Kq"] = st["V"].copy(), st["Kq"].copy()
    Bov, Bvo = np.sqrt(lam) * Bov, np.sqrt(lam) * Bvo
    T = np.einsum("xiag,ybjg->xyiajb", Bov, Bvo.conj(), optimize=True)
    for x in range(Nk):
        new["V"][x, :, x] += T[x]
    nov = Bov.shape[1] * Bov.shape[2]
    new["Kq"][0] += np.einsum("xiag,yjbg->xiayjb", Bov, Bov.conj(), optimize=True).reshape(Nk * nov, Nk * nov) / Nk
    return new


def mo_head_rows(cell, st, gov, gvo, eo, ev, grid, tag, rows, eoF=None, evF=None):
    """cubic + Lebedev heads from MO-level head vectors; optionally with Fock-corrected denominators."""
    c = np.sqrt(4 * np.pi / (3 * cell.vol))
    rows[f"{tag}_cub"] = energies(with_head_mo(st, c * gov, c * gvo), eo, ev)
    pref = np.sqrt(4 * np.pi / cell.vol)
    lebE = lambda o, v: tuple(sum(u[3] * np.array(energies(with_head_mo(
        st, pref * np.einsum("kiax,x->kia", gov, u[:3])[..., None], pref * np.einsum("kaix,x->kai", gvo, u[:3])[..., None]),
        o, v)) for u in grid))
    rows[f"{tag}_leb"] = lebE(eo, ev)
    if eoF is not None:
        rows[f"{tag}_fock"] = energies(st, eoF, evF)
        rows[f"{tag}_fock+leb"] = lebE(eoF, evF)


def energies(st, eo, ev):
    return kmp2_vec(st, eo, ev), KC.kdrpa_plasmon(st, eo, ev)


def run_mesh(a, ns, nleb=26, tag=""):
    cell = Cell(np.eye(3) * a, H2_ATOMS, "sto-3g")
    gcut = PK.aft_gcut(cell, 1e-10)
    for m in ns:
        t0 = time.time()
        n = (m, m, m)
        kb = PK.build_k(cell, n, exxdiv="ewald", gcut=gcut, verbose=False)
        e_hf, eps, it, C, occ = PK.krhf(kb, 2, conv=1e-12, kshift=0.0, return_mo=True)
        assert all(int(o.sum()) == 1 and o[0] for o in occ)
        vm = kb["madelung"]
        t1 = time.time()
        st = KC.aft_ov(cell, n, C, 1, gcut=gcut)
        st["Co"], st["Cv"] = KC._mo_stacks(C, 1)
        t2 = time.time()
        eo, ev = KC.k_denominators(eps, 1, vm, "shifted")
        P0, A, B = pair_derivs(cell, n)
        Nk = st["Nk"]
        # A4 anchor
        a4 = max(abs(P0[k] - kb["S"][k]).max() for k in range(Nk))
        # A2 anchor
        e_sh = energies(st, eo, ev)
        a2 = abs(e_sh[0] - KC.kmp2(st, eo, ev)[0])
        cub = A * np.sqrt(4 * np.pi / (3 * cell.vol))
        rows = {"sh": e_sh}
        for lam in (1.0, 0.5, 2.0, 1e-3):
            rows[f"cub{lam:g}"] = energies(with_head(st, cub, lam), eo, ev)
        a2h = abs(rows["cub1"][0] - KC.kmp2(with_head(st, cub), eo, ev)[0])
        # Lebedev-averaged head (exact direction average of the q->0 integrand)
        grid = __import__("pyscf.dft.gen_grid", fromlist=["x"]).MakeAngularGrid(nleb)
        pref = np.sqrt(4 * np.pi / cell.vol)
        grid110 = __import__("pyscf.dft.gen_grid", fromlist=["x"]).MakeAngularGrid(110)
        for lam, gr in ((1.0, grid), (1e-3, grid), (1.0, grid110)):
            acc = np.zeros(2)
            for u in gr:
                Pu = pref * np.einsum("kmna,a->kmn", A, u[:3])[..., None]
                acc += u[3] * np.array(energies(with_head(st, Pu, lam), eo, ev))
            rows[f"leb{lam:g}" + ("_110" if len(gr) == 110 else "")] = tuple(acc)
        # Fock head (first-order eigenvalue corrections)
        deo, dev = fock_head_shifts(cell, st, P0, A, B, C, 1)
        eoF = [x - y for x, y in zip(eo, deo)]
        evF = [x - y for x, y in zip(ev, dev)]
        rows["fock"] = energies(st, eoF, evF)
        rows["fock+cub"] = energies(with_head(st, cub), eoF, evF)
        rows["fock+leb"] = tuple(sum(u[3] * np.array(energies(with_head(st, pref * np.einsum("kmna,a->kmn", A, u[:3])[..., None]), eoF, evF))
                                     for u in grid))
        # virtual-only and occupied-only Fock corrections
        rows["fock_occ"] = energies(st, eoF, ev)
        rows["fock_vir"] = energies(st, eo, evF)
        # A5 mutations
        mo = fock_head_shifts(cell, st, P0, A, B, C, 1, mutant="no_SB")
        mv = fock_head_shifts(cell, st, P0, A, B, C, 1, mutant="no_AA")
        # anchor: MO-level head path with fixed-k derivative vectors == AO-level path
        Cs = np.array(C)
        gov_d = np.einsum("kmi,kmnx,kna->kiax", Cs[:, :, :1].conj(), A, Cs[:, :, 1:])
        gvo_d = np.einsum("kma,kmnx,kni->kaix", Cs[:, :, 1:].conj(), A, Cs[:, :, :1])
        cc = np.sqrt(4 * np.pi / (3 * cell.vol))
        a_mo = abs(energies(with_head_mo(st, cc * gov_d, cc * gvo_d), eo, ev)[0] - rows["cub1"][0])
        # rows with a flat (k-averaged) Fock correction: tests whether the per-k distribution matters
        mo_, mv_ = np.mean([x[0] for x in deo]), np.mean([x[0] for x in dev])
        rows["fock_kflat"] = energies(st, [x - mo_ for x in eo], [x - mv_ for x in ev])
        if m >= 3:
            Mt, Mf, bn = berry_pairs(cell, n, C)
            for tg, M in (("true_b", Mt), ("fixk_b", Mf)):
                gov, gvo = heads_from_pairs(M, bn, 1)
                deoB, devB = fock_from_pairs(cell, M, bn, 1, Nk)
                eoB = [x - y for x, y in zip(eo, deoB)]
                evB = [x - y for x, y in zip(ev, devB)]
                mo_head_rows(cell, st, gov, gvo, eo, ev, grid, tg, rows, eoB, evB)
                print(f"   {tg}: b {bn:.4f} Fock shifts Gamma occ {deoB[0][0] * m**3:+.6e} vir {devB[0][0] * m**3:+.6e} "
                      f"(x n^3); k-avg occ {np.mean(deoB) * m**3:+.6e} vir {np.mean(devB) * m**3:+.6e}; "
                      f"|g_ov|^2 k-avg {np.mean(np.sum(abs(gov) ** 2, axis=(1, 2, 3))):.6f}", flush=True)
            print(f"   fixed-k derivative |g_ov|^2 k-avg {np.mean(np.sum(abs(gov_d) ** 2, axis=(1, 2, 3))):.6f}")
        g0 = PK.mesh_index(n, np.zeros(3, int))
        print(f"   anchor MO-level head == AO-level cubic head: {a_mo:.1e}")
        print(f"== [{_ORIENT}] a={a} n={m}: Nk {Nk} v_M {vm:.8f} HF {e_hf:.12f} it {it} (scf {t1 - t0:.0f}s, aft_ov {t2 - t1:.0f}s)")
        print(f"   anchors: A4 max|P0-S| {a4:.1e}; A2 |vec-kmp2| {a2:.1e} (no head) {a2h:.1e} (head)")
        print(f"   Fock shifts at Gamma: d_eps_occ {deo[g0][0]:+.6e} d_eps_vir {dev[g0][0]:+.6e}  (x n^3: "
              f"{deo[g0][0] * m**3:+.6e} {dev[g0][0] * m**3:+.6e}); k-avg occ {np.mean([x[0] for x in deo]):+.6e} "
              f"vir {np.mean([x[0] for x in dev]):+.6e}")
        print(f"   A5: no_SB occ {mo[0][g0][0]:+.3e} vir {mo[1][g0][0]:+.3e}; no_AA occ {mv[0][g0][0]:+.3e} "
              f"vir {mv[1][g0][0]:+.3e}")
        for key, val in rows.items():
            print(f"   {key:10s} MP2 {val[0]:.12e}  dRPA {val[1]:.12e}  | minus sh x n^3: MP2 {(val[0] - e_sh[0]) * m**3:+.6e}"
                  f" dRPA {(val[1] - e_sh[1]) * m**3:+.6e}", flush=True)
        np.savez(f"head_anomaly_data/head_anomaly_a{a:g}_n{m}{_SUF}{tag}.npz", rows=np.array([list(v) for v in rows.values()]),
                 keys=np.array(list(rows.keys())), eps=np.array(eps), vm=vm, deo=np.array(deo), dev=np.array(dev),
                 e_hf=e_hf, V=st["V"], Kq=st["Kq"], C=np.array(C), A=A, B=B, P0=P0)
        print(f"   ({time.time() - t0:.0f}s)", flush=True)


def run_eps(a, ns):
    """Gamma-point HF eigenvalues (exxdiv ewald, occupied also with v_M removed) vs n: the empirical Fock head."""
    cell = Cell(np.eye(3) * a, H2_ATOMS, "sto-3g")
    gcut = PK.aft_gcut(cell, 1e-10)
    for m in ns:
        t0 = time.time()
        n = (m, m, m)
        kb = PK.build_k(cell, n, exxdiv="ewald", gcut=gcut, verbose=False)
        e_hf, eps, it, C, occ = PK.krhf(kb, 2, conv=1e-12, kshift=0.0, return_mo=True)
        g0 = mesh_index(n, np.zeros(3, int))
        vm = kb["madelung"]
        e = eps[g0]
        ea = np.array(eps)
        print(f"a={a} n={m}: HF {e_hf:.12f} v_M {vm:.10f} Gamma eps_none {e[0]:.12f} {e[1]:.12f} shifted occ "
              f"{e[0] - vm:.12f} | k-avg shifted occ {ea[:, 0].mean() - vm:.12f} vir {ea[:, 1].mean():.12f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        np.savez(f"head_anomaly_data/head_anomaly_eps_a{a:g}_n{m}.npz", eps=ea, vm=vm, e_hf=e_hf, kpts=kb["kpts"])


# ------------------------------------------------------------------------------ H6: lattice constant of a degree-0 head
def lattice_phi(pmax=4, R=110, eps=(0.01, 0.005, 0.0025)):
    """Phi_p = -Z(u_z^(2p)): the correction the Gamma-centred cubic mesh actually needs for a q -> 0 integrand term
    f0(q^) = u_z^(2p) (degree 0).  Z(g) = lim [sum_{m != 0} g(m^) e^(-eps m^2) - (pi/eps)^(3/2) <g>] (Richardson in
    eps).  For p = 0, 1 (l <= 2) Phi = <g> (the missing point = the angular average; no l = 2 cubic invariant); for
    p >= 2 the l >= 4 cubic harmonics have a NONZERO lattice sum, so the angular (Lebedev) average is not the
    trapezoid error.  Cross-check (independent construction): Epstein zeta functional equation for P4 = sum x^4 -
    3/5 r^4: Z_P(2) = pi^-1.5 Gamma(3.5) Z_P(3.5), spherical summation -> 1.11477 vs 3 Z(z^4) + 3/5 = 1.11480."""
    m = np.arange(-R, R + 1, dtype=float)
    z2 = (m**2)[None, None, :]
    r2 = (m**2)[:, None, None] + (m**2)[None, :, None] + z2
    msk = r2 > 0
    c2 = np.where(msk, z2 / np.where(msk, r2, 1.0), 0.0)
    out = []
    for p in range(pmax + 1):
        g = np.where(msk, c2**p, 0.0)
        z = [(g * np.exp(-e * r2)).sum() - (np.pi / e) ** 1.5 / (2 * p + 1) for e in eps]
        out.append(-(8 * z[2] - 6 * z[1] + z[0]) / 3)
    return np.array(out)


PHI_Z = np.array([1.0, 1 / 3, -0.17159902, -0.34844273, -0.54823523])  # lattice_phi() output, simple cubic, u_z
PHI_111 = np.array([1.0, 1 / 3, 0.447733, 0.508099, 0.483903])  # same for (u . [111]/sqrt3)^2p
PHI = PHI_111 if _ORIENT == "111" else PHI_Z


def head_poly(row, key):
    """F(t) - F(0) = sum_{p=1..4} c_p t^p from the cubic-head rows (t = lam/3, head along one direction: the cubic
    head is F(1/3), l x cubic is F(l/3)); returns (c, Lebedev prediction sum c_p/(2p+1), lattice sum c_p Phi_p)."""
    f0 = row["sh"][key]
    ts = np.array([1e-3, 0.5, 1.0, 2.0]) / 3
    ys = np.array([row[f"cub{l:g}"][key] for l in (1e-3, 0.5, 1.0, 2.0)]) - f0
    Amat = np.array([[t**p for p in range(1, 5)] for t in ts])
    c = np.linalg.solve(Amat, ys)
    return c, sum(c[p - 1] / (2 * p + 1) for p in range(1, 5)), sum(c[p - 1] * PHI[p] for p in range(1, 5))


def run_account(a, ns, einf=None):
    """Accounting table from the saved rows: S, E1 = cubic head, E2 = Lebedev head + Fock, E3 = lattice head + Fock."""
    print(f"a = {a}: lattice Phi_p = {PHI}")
    for key, name in ((0, "MP2"), (1, "dRPA")):
        for m in ns:
            z = np.load(f"head_anomaly_data/head_anomaly_a{a:g}_n{m}{_SUF}.npz")
            row = {k: v for k, v in zip(z["keys"], z["rows"])}
            c, leb_pred, lat = head_poly(row, key)
            S = row["sh"][key]
            F = row["fock"][key] - S
            cross = (row["fock+leb"][key] - S) - (row["leb1"][key] - S) - F
            E1 = row["cub1"][key]
            E2 = row["fock+leb"][key]
            E3 = S + lat + F + cross
            n3 = m**3
            txt = (f"{name} n={m}: S {S:.10e} | E1 cub {E1:.10e} | E2 leb+F {E2:.10e} | E3 lattice+F {E3:.10e} || x n^3: "
                   f"L {c[0] / 3 * n3:+.4e} cub-quad {(row['cub1'][key] - S - c[0] / 3) * n3:+.4e} lattice-quad "
                   f"{(lat - c[0] / 3) * n3:+.4e} F {F * n3:+.4e} cross {cross * n3:+.1e}; Leb check "
                   f"{(leb_pred - (row['leb1'][key] - S)) * n3:+.1e}; c = {np.array2string(c, precision=4)}")
            if einf is not None:
                e = einf[key]
                txt += (f"\n      vs E_inf {e:.8e}: R n^3 {(S - e) * n3:+.4e}; predicted -(lattice head + F) "
                        f"{-(lat + F + cross) * n3:+.4e}; E1-E_inf {(E1 - e) * n3:+.3e} E2-E_inf {(E2 - e) * n3:+.3e} "
                        f"E3-E_inf {(E3 - e) * n3:+.3e}; cubic head removes {(S - E1) / (S - e) * 100:.1f}% of R")
            print(txt)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "mol":
        run_mol()
    elif mode == "mesh":
        run_mesh(float(sys.argv[2]), [int(x) for x in sys.argv[3:]])
    elif mode == "phi":
        print(lattice_phi())
    elif mode == "account":
        a = float(sys.argv[2])
        einf = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 and "." in sys.argv[3] else None
        run_account(a, [int(x) for x in sys.argv[5 if einf else 3:]], einf)
    elif mode == "eps":
        run_eps(float(sys.argv[2]), [int(x) for x in sys.argv[3:]])
    else:
        raise SystemExit(__doc__)
