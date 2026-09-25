"""Iteration 20: Gamma-point analytic nuclear forces for periodic ROHF and ROKS (LDA / GGA / global hybrid), dense
pure-AFT J/K, and RS-GDF J/K by composition.

Nothing lattice-specific is new.  The force is Iteration 17's `pbc_grad_open.gamma_grad` (UHF / UKS functional:
per-spin exchange Gam and Z(G), per-spin Madelung S-term, polarized XC AO term + grid response), evaluated on the ROHF
spin densities D_a, D_b and the SPIN Focks F_a, F_b.  The one ROHF-specific question is the energy-weighted matrix W.
This module derives W from the ROHF constraint, builds it independently in the MO basis, and adds the difference
(W_ro - W_uhf) to the overlap-derivative contraction.  That keeps every other term exactly as Iteration 17 measured it.
The derivation shows the difference is zero at the ROHF stationary point, and the anchors test that claim.

ENERGY.  One set of orthonormal spatial orbitals C = [C_c | C_o | C_v] (nd closed, no open, virtual), occupations
    n^a = diag(1_c, 1_o, 0_v),  n^b = diag(1_c, 0_o, 0_v),   D_a = C n^a C^T,  D_b = C n^b C^T,
    E_RO(C) = E_U[D_a(C), D_b(C)]      (the UHF/UKS functional of pbc_grad_open, restricted variational space)
    F_s = dE_U/dD_s = h + J[D] - alpha (K[D_s] + v_M S D_s S) + V_xc^s   (the SPIN Focks, NOT the Roothaan F_eff).

DERIVATION OF W (orthonormality Lagrangian of ONE orbital set).
    L(C, eps) = E_RO(C) - tr[eps (C^T S C - 1)],  eps symmetric.
    dE_RO/dC = 2 (F_a C n^a + F_b C n^b),  so  dL/dC = 0  <=>  F_a C n^a + F_b C n^b = S C eps.
    Left-multiply by C^T (f_s = C^T F_s C, the MO-basis spin Focks):   eps = f_a n^a + f_b n^b,  i.e.
        eps_ij = (f_a)_ij n^a_j + (f_b)_ij n^b_j        (column j carries the occupation of orbital j).
    eps must be symmetric.  Reading off each block:
        (v, c): eps_vc = (f_a + f_b)_vc,   eps_cv = 0      => (f_a + f_b)_vc = 0      closed-virtual rotation
        (v, o): eps_vo = (f_a)_vo,         eps_ov = 0      => (f_a)_vo = 0            open-virtual rotation
        (o, c): eps_oc = (f_a + f_b)_oc,   eps_co = (f_a)_co => (f_b)_co = 0          closed-open rotation
    These three are exactly the ROHF stationarity conditions (the orbital gradient that
    roks_replica.mo_gradient / ferric's rohf.rs drive to zero).  So at a converged ROHF the Lagrangian exists and
        eps (occupied block) = [[ (f_a + f_b)_cc , (f_a)_co ],
                                [ (f_a)_oc       , (f_a)_oo ]],       W_ro = C_occ eps C_occ^T.
    dE/dR_A = dE_U/dR_A |_(D_a, D_b fixed)  -  tr(W_ro dS/dR_A),  and everything in the first term is Iteration 17.
    In the AO basis:  W_ro = sym(D_a F_a D_a + D_a F_b D_b)   (ferric's molecular rohf_energy_weighted_density;
    the unsymmetrized C_occ (f_a n^a + f_b n^b)_occ C_occ^T IS D_a F_a D_a + D_a F_b D_b).

    UHF form evaluated with the ROHF densities:  W_U = D_a F_a D_a + D_b F_b D_b = C[cc: f_a+f_b, co: f_a, oo: f_a]C^T.
        W_ro - W_U = sym((D_a - D_b) F_b D_b) = 1/2 [C_o (f_b)_oc C_c^T + C_c (f_b)_co C_o^T].
    That is 1/2 x the closed-open beta orbital gradient.  It is ZERO at the ROHF stationary point, where (f_b)_co = 0.
    CONSEQUENCE: at convergence the ROHF W IS the UHF-form W with the ROHF densities (PySCF's pyscf/grad/rohf.py
    make_rdm1e uses exactly D_a F_a D_a + D_b F_b D_b).  A "UHF-form W" mutant is an IDENTITY at convergence, like
    the D_a F_b D_b -> D_b F_b D_b mutant in memory "A mutation can be an identity at convergence".  Its miss is
    |(f_b)_co| ~ the SCF stop threshold.  It is reported as such (w_uhf, below), not as a mutant that must fail.

    What IS wrong, and is not an identity:
      * W from the Roothaan EFFECTIVE Fock's eigenpairs, W = C diag(2 e_c, e_o) C^T, which is the RHF-style
        "sum_i n_i e_i c_i c_i^T" port.  The Guest-Saunders F_eff has cc block 1/2(f_a+f_b)_cc (x2 = correct),
        oo block 1/2(f_a+f_b)_oo (correct: (f_a)_oo), and co block (f_b)_co = 0 (correct: (f_a)_co).
            W_feff - W_ro = C_o [1/2 (f_b - f_a)_oo] C_o^T - [C_c (f_a)_co C_o^T + C_o (f_a)_oc C_c^T].
        Nonzero whenever the open shell has exchange polarization ((f_a - f_b)_oo ~ K_o) or couples to the closed shell.
      * dropping the closed-open coupling: W_no_co = W_ro - [C_c (f_a)_co C_o^T + h.c.]  (nonzero iff (f_a)_co != 0).
      * spin-summed RHF form 1/2 D Fbar D (Iteration 17's w_spin_sum).
    Each mutant's (W_mut - W_ro) is printed, and so is its contraction with dS.  A mutant whose algebra vanishes would
    survive for the wrong reason.

    Other terms (unchanged from Iteration 17, and why):
      * Madelung.  Per spin, alpha (c0 - v_M) sum_s D_s S D_s in M.  D_s S D_s = D_s still holds (each D_s is an
        S-orthonormal projector built from the one C), so F(ewald) == F(none) is predicted as for UHF.
      * XC.  The POLARIZED kernel on (rho_a, rho_b): restricted orbitals do not make the density unpolarized.
        Routing ROKS through the RKS gradient path (unpolarized kernel on D_a + D_b) is the mutant xc_rks_form.
      * 2e Gam / Z(G): per spin on D_a, D_b (Iteration 17 exch_total mutant = RHF form on the total density).

RS-GDF: `gamma_ro_grad_gdf` = pbc_grad_gdf.gamma_gdf_grad(D_a, D_b, F_a, F_b) + the same W correction on the same XS
contraction.  With GDF the Madelung / G=0 bookkeeping lives where Iteration 18 put it, untouched.

SCF (`rohf_scf`): Guest-Saunders Roothaan F_eff (roks_replica.roothaan == pbc_uks.roks == ferric rohf.rs), DIIS on
ferric's error S C (g - g^T) C^T S (err_kind='mo'; 'fds' = pbc_uks.roks's X^T[F_eff, D_a + D_b]X), the virtual level shift ramped as ls err/(err + 1e-3) (ferric's ramp; default 0.05 for
hybrid ROKS, 0 otherwise), and the stop on the ROHF ORBITAL GRADIENT max|g| < tol, where g holds the three blocks
above.  The analytic force is first order in g.  The stop never fires at iteration 0, so a seed from another geometry
always gets at least one diagonalisation.  The seed is Loewdin-orthonormalised in the new metric.

_MUTANT (module global):
    'w_uhf'         W = D_a F_a D_a + D_b F_b D_b             (PREDICTED IDENTITY at convergence; miss ~ |(f_b)_co|)
    'w_feff_eps'    W = C_can diag(2 e_c, e_o) C_can^T from the Roothaan F_eff eigenpairs
    'w_no_co'       W_ro without the (f_a)_co closed-open coupling block
    'w_spin_sum'    W = 1/2 D Fbar D, Fbar = (F_a + F_b)/2
    'xc_rks_form'   XC gradient (AO term + grid response) through the RKS (unpolarized) path
    'madelung_total', 'exch_total'   forwarded to pbc_grad_open._MUTANT (RHF-form Madelung S-term / exchange Gam)
"""

from __future__ import annotations

from collections import deque

import numpy as np

import pbc_grad as PGd
import pbc_grad_open as O
from pbc_uhf import _orth
from roks_replica import mo_gradient, roothaan

_MUTANT = None


# ================================================================================================ ROHF / ROKS SCF
def dens(C, nd, no):
    Db = C[:, :nd] @ C[:, :nd].T
    return Db + C[:, nd:nd + no] @ C[:, nd:nd + no].T, Db


def mo_blocks(C, Fa, Fb, nd, no):
    """MO-basis spin Focks and the three ROHF orbital-gradient blocks (co: f_b, ov: f_a, cv: f_a + f_b)."""
    fa, fb = C.T @ Fa @ C, C.T @ Fb @ C
    na = nd + no
    c, o, v = slice(0, nd), slice(nd, na), slice(na, None)
    gco, gov, gcv = fb[c, o], fa[o, v], (fa + fb)[c, v]
    err = max([abs(x).max() for x in (gco, gov, gcv) if x.size] + [0.0])
    return dict(fa=fa, fb=fb, gco=gco, gov=gov, gcv=gcv, err=err)


def orthonormalize(C0, S):
    s, U = np.linalg.eigh(C0.T @ S @ C0)
    return C0 @ (U / np.sqrt(s)) @ U.T


def occ_overlap(C, Cref, S, nd, no):
    """(closed, open) subspace overlaps ||C_blk^T S C_ref,blk||_F^2; = (nd, no) when the displaced state continues the
    reference occupation pattern."""
    na = nd + no
    oc = np.sum((C[:, :nd].T @ S @ Cref[:, :nd]) ** 2)
    oo = np.sum((C[:, nd:na].T @ S @ Cref[:, nd:na]) ** 2)
    return oc, oo


def spin_gaps(C, Fa, Fb, nd, no):
    """ROHF occupation-aware Koopmans-type margins at convergence: alpha = min eig (f_a)_vv - max eig (f_a)_occ,
    beta = min eig (f_b)_(o+v) - max eig (f_b)_cc (negative => a hole state in that spin)."""
    b = mo_blocks(C, Fa, Fb, nd, no)
    na = nd + no

    def ev(m):
        return np.linalg.eigvalsh(m) if m.size else np.array([np.nan])

    ga = ev(b["fa"][na:, na:]).min() - ev(b["fa"][:na, :na]).max() if C.shape[1] > na else np.inf
    gb = ev(b["fb"][nd:, nd:]).min() - ev(b["fb"][:nd, :nd]).max() if nd else np.inf
    return ga, gb


def rohf_scf(S, h, jk, enn, nd, no, grid, xc, vM=0.0, C0=None, tol=1e-11, maxiter=1500, level_shift=None,
             diis_size=8, verbose=False, err_kind="mo"):
    """Gamma ROHF (xc='HF') / ROKS on (S, h, jk, E_nn[, grid]); the energy is pbc_grad_open.fock_energy (UKS
    functional).  Returns dict(e, C [c|o|v], Da, Db, Fa, Fb, Feff, it, err, hyb, gaps)."""
    hyb = O.hybrid_fraction(xc)
    if str(xc).upper() == "HF":
        grid = None
    if level_shift is None:
        level_shift = 0.05 if 0.0 < hyb < 1.0 else 0.0
    X = _orth(S)
    na = nd + no
    if C0 is None:
        C = X @ np.linalg.eigh(X.T @ h @ X)[1]
    else:
        C = orthonormalize(np.array(C0), S)
    focks, errs = deque(maxlen=diis_size), deque(maxlen=diis_size)
    err = np.inf
    for it in range(maxiter):
        Da, Db = dens(C, nd, no)
        e, Fa, Fb = O.fock_energy(S, h, jk, enn, Da, Db, grid, xc, hyb, vM, False)
        err = mo_blocks(C, Fa, Fb, nd, no)["err"]
        if verbose and (it < 5 or it % 50 == 0):
            print(f"    it {it:4d} E {e:.12f} err {err:.2e}", flush=True)
        if err < tol and it > 0:
            break
        F = roothaan(Fa, Fb, Da, Db, S)
        if err_kind == "mo":  # ferric rohf.rs: S C (g - g^T) C^T S with g the ROHF MO-gradient blocks
            g = mo_gradient(C, Fa, Fb, nd, no)
            er = S @ C @ (g - g.T) @ C.T @ S
        else:  # pbc_uks.roks: [F_eff, D_a + D_b] in the X basis
            Dt = Da + Db
            er = X.T @ (F @ Dt @ S - S @ Dt @ F) @ X
        focks.append(F)
        errs.append(er.ravel())
        Fd = F
        if len(focks) > 1:
            n = len(focks)
            B = -np.ones((n + 1, n + 1))
            B[-1, -1] = 0
            for i in range(n):
                for j in range(n):
                    B[i, j] = errs[i] @ errs[j]
            rhs = np.zeros(n + 1)
            rhs[-1] = -1
            cc = np.linalg.lstsq(B, rhs, rcond=None)[0][:n]
            Fd = sum(ci * f for ci, f in zip(cc, focks))
        if level_shift:
            Cv = C[:, na:]
            Fd = Fd + level_shift * err / (err + 1e-3) * (S @ Cv @ Cv.T @ S)
        C = X @ np.linalg.eigh(X.T @ Fd @ X)[1]
        if it and it % 300 == 0:
            focks.clear()
            errs.clear()
    else:
        raise RuntimeError(f"ROHF/ROKS not converged ({xc}, err {err:.2e}, E {e:.10f})")
    Feff = roothaan(Fa, Fb, Da, Db, S)
    return dict(e=e, C=C, Da=Da, Db=Db, Fa=Fa, Fb=Fb, Feff=Feff, it=it, err=err, hyb=hyb, nd=nd, no=no,
                gaps=spin_gaps(C, Fa, Fb, nd, no))


# ======================================================================================================= W forms
def w_ro(C, Fa, Fb, nd, no):
    """The Lagrangian W = C_occ eps C_occ^T, eps = sym(f_a n^a + f_b n^b) on the occupied block (MO construction)."""
    na = nd + no
    b = mo_blocks(C, Fa, Fb, nd, no)
    na_occ = np.r_[np.ones(na)]
    nb_occ = np.r_[np.ones(nd), np.zeros(no)]
    L = b["fa"][:na, :na] * na_occ[None, :] + b["fb"][:na, :na] * nb_occ[None, :]
    eps = 0.5 * (L + L.T)
    Co = C[:, :na]
    return Co @ eps @ Co.T


def w_uhf(Da, Db, Fa, Fb):
    return Da @ Fa @ Da + Db @ Fb @ Db


def w_feff_eps(Feff, S, nd, no):
    X = _orth(S)
    e, U = np.linalg.eigh(X.T @ Feff @ X)
    C = X @ U
    n = np.r_[2 * np.ones(nd), np.ones(no)]
    Co = C[:, :nd + no]
    return (Co * (n * e[:nd + no])) @ Co.T


def w_no_co(C, Fa, Fb, nd, no):
    na = nd + no
    fa = C.T @ Fa @ C
    Cc, Cop = C[:, :nd], C[:, nd:na]
    cpl = Cc @ fa[:nd, nd:na] @ Cop.T
    return w_ro(C, Fa, Fb, nd, no) - cpl - cpl.T


def w_of(mode, r, S):
    C, Fa, Fb, nd, no = r["C"], r["Fa"], r["Fb"], r["nd"], r["no"]
    if mode in (None, "ro"):
        return w_ro(C, Fa, Fb, nd, no)
    if mode == "w_uhf":
        return w_uhf(r["Da"], r["Db"], Fa, Fb)
    if mode == "w_feff_eps":
        return w_feff_eps(r["Feff"], S, nd, no)
    if mode == "w_no_co":
        return w_no_co(C, Fa, Fb, nd, no)
    if mode == "w_spin_sum":
        D = r["Da"] + r["Db"]
        return 0.5 * D @ (0.5 * (Fa + Fb)) @ D
    raise ValueError(mode)


def s_contract(cell, dW, XS=None, rcut_1e=22.0):
    """Force change from W -> W + dW (M = -W + ... is contracted as -2 sum XS M, Iteration 16)."""
    if XS is None:
        XS = PGd._latsum_ip(cell, "int1e_ipovlp_cart", rcut_1e)
    return PGd._fold(2 * np.einsum("xmn,mn->mx", XS, dW), PGd.ao_atom(cell.mol), cell.mol.natm)


# ================================================================================================ gradient
def _w_correction(cell, ints, r, XS):
    """(force correction W_U -> W_used, diagnostics)."""
    S = ints["S"]
    Wu = w_uhf(r["Da"], r["Db"], r["Fa"], r["Fb"])
    Wr = w_ro(r["C"], r["Fa"], r["Fb"], r["nd"], r["no"])
    mode = _MUTANT if (_MUTANT or "").startswith("w_") else None
    Wm = w_of(mode, r, S)
    b = mo_blocks(r["C"], r["Fa"], r["Fb"], r["nd"], r["no"])
    nd, na = r["nd"], r["nd"] + r["no"]
    diag = dict(fb_co=abs(b["gco"]).max() if b["gco"].size else 0.0,
                fa_co=abs(b["fa"][:nd, nd:na]).max() if nd and r["no"] else 0.0,
                fab_oo=abs((b["fa"] - b["fb"])[nd:na, nd:na]).max() if r["no"] else 0.0,
                dW_ro_uhf=abs(Wr - Wu).max(), dW_used_ro=abs(Wm - Wr).max(),
                dF_used_ro=abs(s_contract(cell, Wm - Wr, XS)).max())
    return s_contract(cell, Wm - Wu, XS), diag


def gamma_ro_grad(cell, ints, r, w=None, grid=None, xc="HF", response=True, rcut_1e=22.0):
    """dE/dR (natm, 3), parts, diag for a converged rohf_scf result r (dense pure-AFT or Ewald-split ints)."""
    fwd = _MUTANT if _MUTANT in ("madelung_total", "exch_total") else None
    O._MUTANT = fwd
    try:
        g, parts = O.gamma_grad(cell, ints, r["Da"], r["Db"], r["Fa"], r["Fb"], r["hyb"], w=w,
                                grid=None if str(xc).upper() == "HF" else grid, xc=xc, restricted=False,
                                response=response, rcut_1e=rcut_1e)
    finally:
        O._MUTANT = None
    XS = PGd._latsum_ip(cell, "int1e_ipovlp_cart", rcut_1e)
    parts["S_ro"], diag = _w_correction(cell, ints, r, XS)
    if _MUTANT == "xc_rks_form" and grid is not None and str(xc).upper() != "HF":
        xg = O.xc_grad(grid, r["Da"], r["Db"], xc, True, cell.mol.natm, PGd.ao_atom(cell.mol), response)
        parts["xc_ao"], parts["xc_point"], parts["xc_weight"] = xg["ao"], xg["point"], xg["weight"]
    return sum(parts.values()), parts, diag


def gamma_ro_grad_gdf(cell, ints, gd, auxmol, r, w=1.0, lindep=1e-10, spherical=True, grid=None, xc="HF",
                      cache=None, **kw):
    """RS-GDF composition: Iteration 18's gamma_gdf_grad on (D_a, D_b, F_a, F_b) + the ROHF W correction."""
    import pbc_grad_gdf as GG

    if cache is None:
        cache = {}
    g, parts, gdiag = GG.gamma_gdf_grad(cell, ints, gd, auxmol, r["Da"], r["Db"], r["Fa"], r["Fb"], r["hyb"], w=w,
                                        lindep=lindep, spherical=spherical,
                                        grid=None if str(xc).upper() == "HF" else grid, xc=xc, restricted=False,
                                        cache=cache, **kw)
    parts["S_ro"], diag = _w_correction(cell, ints, r, cache["XS"])
    return sum(parts.values()), parts, dict(diag, **gdiag)


# ============================================================================================= convenience
def run_case(cell, nd, no, xc, ints, grid=None, C0=None, w=None, tol=1e-11, level_shift=None, response=True,
             verbose=False):
    import pbc_dft as pd

    jk = pd.dense_jk(ints["I"])
    r = rohf_scf(ints["S"], ints["h"], jk, ints["enn"], nd, no, grid, xc, ints["madelung"], C0=C0, tol=tol,
                 level_shift=level_shift, verbose=verbose)
    g, parts, diag = gamma_ro_grad(cell, ints, r, w=w, grid=grid, xc=xc, response=response)
    return dict(e=r["e"], grad=g, parts=parts, diag=diag, scf=r)


def energy_only(cell, nd, no, xc, ints, grid, C0, tol=1e-11, level_shift=None):
    import pbc_dft as pd

    return rohf_scf(ints["S"], ints["h"], pd.dense_jk(ints["I"]), ints["enn"], nd, no, grid, xc, ints["madelung"],
                    C0=C0, tol=tol, level_shift=level_shift)
