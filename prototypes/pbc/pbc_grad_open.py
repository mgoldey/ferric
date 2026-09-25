"""Iteration 17: Gamma-point analytic nuclear forces for periodic UHF and KS-DFT (RKS / UKS, LDA / GGA / global hybrid).

Extends pbc_grad.py (Gamma RHF, Iteration 16) -- every lattice/Coulomb derivative piece (pair-FT derivative Q, Ewald
gradient, SR erfc 3c/4c derivative integrals, lattice-summed <grad m|n_L>) is REUSED unchanged; what is new is the
spin bookkeeping of the density-matrix contractions, the energy-weighted matrix, and the XC gradient on the periodic
SSF/Becke grid of pbc_dft.py (construction A2).

ENERGY (one functional covers UHF, RKS, UKS; alpha = exact-exchange fraction, 1 for HF, 0.25 for PBE0, 0 for LDA/PBE;
D = D_a + D_b; the Madelung shift v_M (exxdiv='ewald') rides on the exact-exchange part only, per spin, coefficient 1
on D_s -- Iterations 6 and 10):
    E = sum D h + 1/2 sum D_mn D_ls I_mnls - alpha/2 sum_s sum D^s_ml D^s_ns I_mnls
        - alpha v_M/2 sum_s tr(D_s S D_s S) + E_xc[rho_a, rho_b] + E_nn,
    h = T + V_SR + V_LR + c0 Z_tot S,   I = I_SR + I_LR - c0 S(x)S,   c0 = pi/(w^2 Omega)  (0 for pure AFT)
    F_s = dE/dD_s = h + J[D] - alpha (K[D_s] + v_M S D_s S) + V_xc^s.
RKS/RHF = the same with D_a = D_b = D/2 (the XC kernel is then evaluated unpolarized, spin=0).

DERIVATION (dE/dR_A; everything moves with atom A in EVERY image: nucleus, basis centres, grid points homed on A)
 1. One-electron and nuclear pieces see only D (spin-free): sum D dT, sum D dV_SR, sum D dV_LR, dE_nn -- identical to
    Iteration 16 with D = D_a + D_b.
 2. Two-electron density (the "Gamma" contracted with dI):
        Gam_mnls = 1/2 D_mn D_ls - alpha/2 sum_s D^s_ml D^s_ns
    (RHF: D^s = D/2 gives 1/2 DD - 1/4 DD, Iteration 16's Gam.)  For the LR (reciprocal) part the bra-derivative
    contraction becomes Z_mn(G) = 1/2 D_mn rho(G) - alpha/2 sum_s (D_s P(G) D_s)_mn   (RHF: -1/4 D P D).
    SR part: sr_eri_grad(Gam) unchanged.
 3. Overlap-coupled terms. Orbital constraint per spin C_s^T S C_s = 1 gives the Lagrangian -sum_s tr(W_s S) with
    W_s = D_s F_s D_s (occupations 1; RHF: sum_s (D/2) F (D/2) = 1/2 D F D).  Explicit S-dependence of E at fixed D_s:
        c0 Z_tot tr(D S)                       -> + c0 Z_tot D
        -c0/2 tr(DS)^2 (Hartree G=0)           -> - c0 N D
        +alpha c0/2 sum_s tr(D_s S D_s S)      -> + alpha c0 sum_s D_s S D_s        (exchange G=0)
        -alpha v_M/2 sum_s tr(D_s S D_s S)     -> - alpha v_M sum_s D_s S D_s        (Madelung, PER SPIN)
    => M = -sum_s D_s F_s D_s + c0 (Z_tot - N) D + alpha (c0 - v_M) sum_s D_s S D_s, contracted as sum M dS.
    RHF check: sum_s D_s S D_s = 1/2 D S D  -> Iteration 16's (c0/2 - v_M/2) D S D.
    Prediction F(ewald) == F(none) per spin: D_s S D_s = D_s (idempotent), and the -alpha v_M S D_s S piece of F_s
    enters W_s as -alpha v_M D_s; -W_s cancels the explicit -alpha v_M D_s exactly (holds for alpha < 1 too).
 4. XC, AO-derivative term (grid points FIXED): chi^Gamma_m(r) = sum_L chi_m(r - R_A - L), so d chi_m/dR_A = -grad chi_m
    for m on A.  With e(r) = f(rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb) the XC energy density,
        dE_xc/dR_A|_grid = -2 sum_g w_g sum_s sum_{m in A} [ v_rho^s d_j chi_m (D_s chi)_m
                                + sum_i c^s_i ( d_j d_i chi_m (D_s chi)_m + d_j chi_m (D_s d_i chi)_m ) ],
        c^a = 2 v_saa grad rho_a + v_sab grad rho_b,  c^b = 2 v_sbb grad rho_b + v_sab grad rho_a
    (RKS spin=0: c = 2 v_sigma grad rho with rho the total density).  The c-term is the "GGA term" (needs AO Hessians).
 5. XC grid response (the periodic grid moves with the atoms).  E_xc = sum_A sum_{g in grid(A)} w0_g P_A(r_g) e(r_g),
    P_A = Becke/SSF weight of the home atom computed over IMAGE atoms B = cell atom + L.  Two pieces:
      (a) point motion: r_g = R_A + offset moves with its home atom A:  + sum_{g home A} w_g grad e(r_g)
          = - sum_{g home A} sum_C a_{g,C}  (translation invariance of e at fixed r: grad_r e = - sum_C de/dR_C, with
          a_{g,C} the per-point AO term of item 4) -- no new AO derivatives needed;
      (b) weight derivative: + sum_g w0_g e(r_g) dP_home(r_g)/dR_A, total derivative with all images of A moving and,
          if A is the home atom, the point moving too (by translation invariance of P: d/dr = -sum_D d/dX_D).
          dP_B/dX_D is analytic: s_BC = 1/2 (1 - g(nu_BC)), nu = mu + a_BC (1 - mu^2), mu = (d_B - d_C)/R_BC,
          dmu/dX_B = -(u_B + mu e_BC)/R_BC,  dmu/dX_C = (u_C + mu e_BC)/R_BC  (u_B = (r - X_B)/d_B, e_BC = (X_B-X_C)/R_BC),
          products via prefix/suffix exclusion (SSF has exact zeros); w = P_h / sum_B P_B.
    With the full response the force is translation invariant (sum F = 0 exactly); without it it is not.

PySCF molecular intor/eval_gto and libxc stand in for libint2 / ferric's ao_grid and libxc wrapper.
_MUTANT (module global) switches on deliberate defects for the mutation tests:
    'w_spin_sum'      W = 1/2 D Fbar D with Fbar = (F_a+F_b)/2 (RHF formula on the total density)
    'madelung_total'  Madelung S-term -alpha v_M/2 D S D (RHF formula) instead of per spin
    'exch_total'      exchange Gam / Z with -alpha/4 D D (RHF formula) instead of per spin
    'xc_no_ao'        drop the XC AO-derivative term (item 4)
    'xc_no_gga'       drop the c (sigma) part of item 4
    'no_point_motion' drop item 5a      'no_weight_deriv' drop item 5b
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np
from pyscf.dft import libxc

import pbc_dft as pd
import pbc_grad as PGd
from pbc_gamma import build_integrals, pair_ft

_MUTANT = None
HIDX = ((4, 5, 6), (5, 7, 8), (6, 8, 9))  # PySCF deriv2 order: xx xy xz yy yz zz at 4..9


def hybrid_fraction(xc):
    if str(xc).upper() == "HF":
        return 1.0
    if libxc.rsh_coeff(xc)[0] != 0:
        raise NotImplementedError("range-separated hybrids need the attenuated periodic K")
    return libxc.hybrid_coeff(xc)


# ============================================================================ grid with analytic weight derivatives
def _smooth_d(nu, scheme):
    """(g(nu), dg/dnu) for the cell function s = 1/2 (1 - g)."""
    if scheme == "ssf":
        a = 0.64
        m = nu / a
        m2 = m * m
        inside = np.abs(nu) < a
        g = np.where(nu <= -a, -1.0, np.where(nu >= a, 1.0, m * (35 + m2 * (-35 + m2 * (21 - 5 * m2))) / 16))
        gp = np.where(inside, 35 * (1 - m2) ** 3 / (16 * a), 0.0)
        return g, gp
    k = {"becke": 3, "becke3": 3, "becke2": 2, "becke1": 1, "becke4": 4}[scheme]
    x, gp = nu, np.ones_like(nu)
    for _ in range(k):
        gp = gp * 1.5 * (1 - x * x)
        x = 0.5 * x * (3 - x * x)
    return x, gp


def partition_weight_deriv(pts, home, X, Zs, cidx, D, scheme, adjust, natm, home_atom):
    """(w, dw) with w == pbc_dft.partition_weight (same mask / truncation) and dw[p, A, x] the TOTAL derivative of w
    with respect to cell atom A: every image of A moves, and the point moves too when A == home_atom."""
    dv = pts[:, None, :] - X[None]  # r - X_B
    d = np.linalg.norm(dv, axis=2)
    mask = d <= D
    Rv = X[:, None, :] - X[None, :, :]
    Rbc = np.linalg.norm(Rv, axis=2)
    np.fill_diagonal(Rbc, 1.0)
    e = Rv / Rbc[..., None]
    acorr = pd.size_adjust(Zs, adjust)
    mu = (d[:, :, None] - d[:, None, :]) / Rbc[None]
    nu = mu + acorr[None] * (1 - mu * mu)
    g, gp = _smooth_d(nu, scheme)
    s = 0.5 * (1 - g)
    q = -0.5 * gp * (1 - 2 * acorr[None] * mu)
    nb = len(X)
    di = np.arange(nb)
    s[:, di, di] = 1.0
    q[:, di, di] = 0.0
    s = np.where(mask[:, None, :], s, 1.0)
    q = np.where(mask[:, None, :], q, 0.0)
    P = s.prod(axis=2) * mask
    tot = P.sum(axis=1)
    ok = tot > 0
    tsafe = np.where(ok, tot, 1.0)
    w = np.where(ok, P[:, home] / tsafe, 0.0)
    # product over C' != C by prefix/suffix (exact zeros are common with SSF)
    pre = np.ones_like(s)
    pre[:, :, 1:] = np.cumprod(s[:, :, :-1], axis=2)
    suf = np.ones_like(s)
    suf[:, :, :-1] = np.cumprod(s[:, :, :0:-1], axis=2)[:, :, ::-1]
    coef = q * pre * suf * mask[:, :, None] / Rbc[None]  # G_BC / R_BC
    u = dv / np.where(d > 0, d, 1.0)[..., None]
    # dP_B/dX_B = -sum_C coef_BC (u_B + mu_BC e_BC) ;  dP_B/dX_C = coef_BC (u_C + mu_BC e_BC)
    selfd = -coef.sum(2)[..., None] * u - np.einsum("pbc,pbc,bcx->pbx", coef, mu, e)
    dPsum = selfd + np.einsum("pbd,pdx->pdx", coef, u) + np.einsum("pbd,pbd,bdx->pdx", coef, mu, e)
    dPh = coef[:, home, :, None] * (u + mu[:, home, :, None] * e[home][None])
    dPh[:, home] += selfd[:, home]
    dwX = (dPh - w[:, None, None] * dPsum) / tsafe[:, None, None] * ok[:, None, None]
    dw = np.zeros((len(pts), natm, 3))
    for A in range(natm):
        dw[:, A] = dwX[:, cidx == A].sum(1)
    dw[:, home_atom] -= dwX.sum(1)
    return w, dw


class GradGrid:
    """pbc_dft.PeriodicGrid (construction A2) rebuilt with the same points/weights PLUS dW[p, A, x] (analytic total
    derivative of the final weight w0 * P_home w.r.t. cell atom A) and AO values up to second derivatives.
    points_from / ao-only mode: GradGrid.frozen(ref_grid, cell) evaluates AOs of `cell` on ref_grid's points/weights."""

    def __init__(self, cell, n_rad=40, n_ang=50, D=10.0, scheme="ssf", adjust=True, deriv=2, ao_thresh=1e-15,
                 chunk=64, box=1.5, want_dw=True):
        self.cell, self.D = cell, D
        a, R = cell.a, cell.R
        Z = cell.mol.atom_charges()
        natm = len(R)
        coords, weights, homes, dws, w0s, pids = [], [], [], [], [], []
        for A in range(natm):
            off, w0, _ = pd.atomic_grid(int(Z[A]), n_rad, n_ang)
            pts = R[A] + off
            keep = np.linalg.norm(off, axis=1) <= D
            pts, w0 = pts[keep], w0[keep]
            pid0 = np.nonzero(keep)[0]
            nb_xyz, nb_z, nb_idx = pd.image_atoms(a, R, Z, R[A], 2 * D + 2 * box)
            home = int(np.nonzero((nb_idx == A) & (np.linalg.norm(nb_xyz - R[A], axis=1) < 1e-9))[0][0])
            key = np.floor(pts / box).astype(np.int64)
            _, inv = np.unique(key, axis=0, return_inverse=True)
            inv = inv.ravel()
            for b in range(inv.max() + 1):
                ib = np.nonzero(inv == b)[0]
                p = pts[ib]
                cen = (np.floor(p[0] / box) + 0.5) * box
                near = np.linalg.norm(nb_xyz - cen, axis=1) <= D + 0.87 * box + 1e-9
                near[home] = True
                sel = np.nonzero(near)[0]
                hs = int(np.nonzero(sel == home)[0][0])
                ck = max(1, int(chunk * 4e4 / max(len(sel), 1) ** 2))
                for q0 in range(0, len(ib), ck):
                    pp = p[q0 : q0 + ck]
                    if want_dw:
                        w, dw = partition_weight_deriv(pp, hs, nb_xyz[sel], nb_z[sel], nb_idx[sel], D, scheme,
                                                       adjust, natm, A)
                        dws.append(w0[ib[q0 : q0 + ck], None, None] * dw)
                    else:
                        w, _ = pd.partition_weight(pp, hs, nb_xyz[sel], nb_z[sel], D, scheme, adjust)
                    coords.append(pp)
                    w0s.append(w0[ib[q0 : q0 + ck]])
                    weights.append(w0[ib[q0 : q0 + ck]] * w)
                    homes.append(np.full(len(pp), A))
                    pids.append(A * 10**7 + pid0[ib[q0 : q0 + ck]])
        self.pid = np.concatenate(pids)  # stable point identity (home atom, atomic-grid index) across geometries
        self.coords = np.vstack(coords)
        self.weights = np.concatenate(weights)
        self.w0 = np.concatenate(w0s)
        self.home = np.concatenate(homes)
        self.dW = np.concatenate(dws) if want_dw else None
        self.ao = ao_gamma_d2(cell, self.coords, deriv, ao_thresh) if deriv is not None else None

    @classmethod
    def frozen(cls, ref, cell, deriv=2, ao_thresh=1e-15):
        g = cls.__new__(cls)
        g.cell, g.D = cell, ref.D
        g.coords, g.weights, g.w0, g.home, g.dW = ref.coords, ref.weights, ref.w0, ref.home, None
        g.ao = ao_gamma_d2(cell, g.coords, deriv, ao_thresh)
        return g

    @property
    def size(self):
        return len(self.weights)


def ao_gamma_d2(cell, pts, deriv=2, thresh=1e-15, chunk=448, box=2.0):
    """pbc_dft.ao_gamma extended to deriv=2: (10, npts, nao) value, grad, Hessian (xx xy xz yy yz zz)."""
    mol = cell.mol
    nao = mol.nao
    rc = pd.ao_rcut(mol, thresh)
    comp = {0: 1, 1: 4, 2: 10}[deriv]
    name = {0: "GTOval_cart", 1: "GTOval_cart_deriv1", 2: "GTOval_cart_deriv2"}[deriv]
    out = np.zeros((comp, len(pts), nao))
    cen = cell.R.mean(0)
    reach = np.linalg.norm(pts - cen, axis=1).max() + rc
    diam = np.linalg.norm(cell.R - cen, axis=1).max()
    Ls = pd.lattice_points(cell.a, reach + diam)
    Ls = Ls[np.linalg.norm(cell.R[None] + Ls[:, None] - cen, axis=2).min(1) <= reach]
    sm = cell.supermol(Ls)
    order = np.lexsort(np.floor(pts / box).T[::-1])
    for p0 in range(0, len(pts), chunk):
        idx = order[p0 : p0 + chunk]
        v = np.asarray(sm.eval_gto(name, pts[idx], cutoff=thresh)).reshape(comp, len(idx), len(Ls), nao)
        out[:, idx] = v.sum(2)
    return out


def uniform_grad_grid(cell, n, deriv=2):
    """Construction B (fixed uniform points, weight Omega/n^3): no grid response exists (points do not move)."""
    f = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3) / n
    g = GradGrid.__new__(GradGrid)
    g.cell, g.D = cell, None
    g.coords = f @ cell.a
    g.weights = np.full(len(g.coords), cell.vol / len(g.coords))
    g.w0, g.home, g.dW = g.weights, np.full(len(g.coords), -1), None
    g.ao = ao_gamma_d2(cell, g.coords, deriv)
    return g


# ============================================================================================ XC energy / potential
def _view4(grid):
    return SimpleNamespace(ao=grid.ao[:4], weights=grid.weights)


def eval_xc(grid, Da, Db, xc, restricted):
    """(E_xc, V_a, V_b): restricted -> unpolarized kernel on D = D_a + D_b (pbc_dft.eval_vxc), else pbc_uks."""
    import pbc_uks as U

    g4 = _view4(grid)
    if restricted:
        exc, V, _ = pd.eval_vxc(g4, Da + Db, xc)
        return exc, V, V
    return U.eval_vxc_uks(g4, Da, Db, xc)


# ========================================================================================================== SCF
def fock_energy(S, h, jk, enn, Da, Db, grid, xc, hyb, vM, restricted):
    Ja, Ka = jk(Da)
    Jb, Kb = (Ja, Ka) if restricted else jk(Db)
    J = Ja + Jb
    Kea = hyb * (Ka + vM * S @ Da @ S)
    Keb = hyb * (Kb + vM * S @ Db @ S)
    exc, Va, Vb = (0.0, 0.0, 0.0) if grid is None else eval_xc(grid, Da, Db, xc, restricted)
    Fa, Fb = h + J - Kea + Va, h + J - Keb + Vb
    e = np.sum((Da + Db) * h) + 0.5 * np.sum((Da + Db) * J) - 0.5 * (np.sum(Da * Kea) + np.sum(Db * Keb)) + exc + enn
    return e, Fa, Fb


def scf(S, h, jk, enn, na, nb, grid, xc, vM=0.0, restricted=False, guess=None, tol=1e-11, maxiter=400, mix=0.0):
    """Tight DIIS UKS/RKS/UHF/RHF (xc='HF' -> grid unused).  Stops when max|X^T[F_s,D_s]X| < tol (the analytic force
    is first-order in this residual)."""
    from pbc_uhf import _orth, core_guess

    hyb = hybrid_fraction(xc)
    if str(xc).upper() == "HF":
        grid = None
    if restricted:
        assert na == nb
    X = _orth(S)
    Da, Db = core_guess(S, h, na, nb, mix) if guess is None else (np.array(guess[0]), np.array(guess[1]))
    focks, errs = deque(maxlen=10), deque(maxlen=10)
    for it in range(maxiter):
        e, Fa, Fb = fock_energy(S, h, jk, enn, Da, Db, grid, xc, hyb, vM, restricted)
        ea = X.T @ (Fa @ Da @ S - S @ Da @ Fa) @ X
        eb = X.T @ (Fb @ Db @ S - S @ Db @ Fb) @ X
        err = max(abs(ea).max(), abs(eb).max())
        if err < tol and it > 0:  # never accept an input guess: it need not be S-normalised at THIS geometry
            break
        if it and it % 60 == 0:  # DIIS stagnation near degeneracy: restart the subspace
            focks.clear()
            errs.clear()
        focks.append((Fa, Fb))
        errs.append(np.concatenate([ea.ravel(), eb.ravel()]))
        if len(focks) > 1:
            n = len(focks)
            B = -np.ones((n + 1, n + 1))
            B[-1, -1] = 0
            for i in range(n):
                for j in range(n):
                    B[i, j] = errs[i] @ errs[j]
            rhs = np.zeros(n + 1)
            rhs[-1] = -1
            c = np.linalg.lstsq(B, rhs, rcond=None)[0][:n]
            Fa = sum(ci * f[0] for ci, f in zip(c, focks))
            Fb = sum(ci * f[1] for ci, f in zip(c, focks))
        Ca = X @ np.linalg.eigh(X.T @ Fa @ X)[1]
        Cb = Ca if restricted else X @ np.linalg.eigh(X.T @ Fb @ X)[1]
        Da, Db = Ca[:, :na] @ Ca[:, :na].T, Cb[:, :nb] @ Cb[:, :nb].T
    else:
        raise RuntimeError(f"SCF not converged ({xc}, err {err:.2e})")
    return dict(e=e, Da=Da, Db=Db, Fa=Fa, Fb=Fb, it=it, err=err, hyb=hyb)


# ================================================================================================== XC gradient
def xc_grad(grid, Da, Db, xc, restricted, natm, aoat, response=True):
    """dict(ao=(natm,3), point=(natm,3), weight=(natm,3), sumF_ao=...) -- items 4, 5a, 5b of the module docstring."""
    fam = pd.xc_family(xc)
    if fam == "MGGA":
        raise NotImplementedError
    gga = fam == "GGA"
    ao = grid.ao
    w = grid.weights
    chi, dchi = ao[0], ao[1:4]

    def rho_of(Dm):
        c = chi @ Dm
        r = np.einsum("pi,pi->p", c, chi)
        gr = 2 * np.einsum("pi,xpi->xp", c, dchi) if gga else None
        return c, r, gr

    if restricted:
        c, r, gr = rho_of(Da + Db)
        rho_in = np.vstack([r[None], gr]) if gga else r
        exc, vxc = libxc.eval_xc(xc, rho_in, spin=0, deriv=1)[:2]
        spins = [(Da + Db, c, vxc[0], (2 * vxc[1] * gr) if gga else None)]
        eps = r * exc
    else:
        ca, ra, ga = rho_of(Da)
        cb, rb, gb = rho_of(Db)
        ins = ((np.vstack([ra[None], ga]), np.vstack([rb[None], gb])) if gga else (ra, rb))
        exc, vxc = libxc.eval_xc(xc, ins, spin=1, deriv=1)[:2]
        vr = vxc[0]
        if gga:
            vs = vxc[1]
            c_a = 2 * vs[:, 0] * ga + vs[:, 1] * gb
            c_b = 2 * vs[:, 2] * gb + vs[:, 1] * ga
        else:
            c_a = c_b = None
        spins = [(Da, ca, vr[:, 0], c_a), (Db, cb, vr[:, 1], c_b)]
        eps = (ra + rb) * exc
    t = np.zeros((len(w), chi.shape[1], 3))  # per point, per AO, per j
    for Dm, c, vrho, cvec in spins:
        t += (vrho[:, None, None] * c[:, :, None]) * dchi.transpose(1, 2, 0)
        if gga and _MUTANT != "xc_no_gga":
            for i in range(3):
                dic = dchi[i] @ Dm  # (D d_i chi)_m
                for j in range(3):
                    t[:, :, j] += cvec[i][:, None] * (ao[HIDX[j][i]] * c + dchi[j] * dic)
    per_pa = np.zeros((len(w), natm, 3))
    for A in range(natm):
        per_pa[:, A] = -2 * w[:, None] * t[:, aoat == A].sum(1)
    out = dict(ao=per_pa.sum(0), point=np.zeros((natm, 3)), weight=np.zeros((natm, 3)))
    if _MUTANT == "xc_no_ao":
        out["ao"] = 0 * out["ao"]
    if response and grid.dW is not None:
        tot = per_pa.sum(1)  # sum_C a_{g,C} = - w_g grad e(r_g)
        for A in range(natm):
            out["point"][A] = -tot[grid.home == A].sum(0)
        out["weight"] = np.einsum("p,pax->ax", eps, grid.dW)
        if _MUTANT == "no_point_motion":
            out["point"] = 0 * out["point"]
        if _MUTANT == "no_weight_deriv":
            out["weight"] = 0 * out["weight"]
    return out


# ================================================================================================ full gradient
def gamma_grad(cell, ints, Da, Db, Fa, Fb, hyb, w=None, grid=None, xc=None, restricted=False, response=True,
               rcut_1e=22.0, rcut_2e=None, rcut_bra=None):
    """dE/dR (natm, 3) and per-term parts for the functional of the module docstring at a converged (D_s, F_s)."""
    S, P, G, vM = ints["S"], ints["P"], ints["G"], ints["madelung"]
    mol = cell.mol
    natm = mol.natm
    aoat = PGd.ao_atom(mol)
    D = Da + Db
    spins = ((Da, Fa), (Db, Fb))
    parts = {}
    c0 = 0.0 if w is None else np.pi / (w * w * cell.vol)
    N = np.sum(D * S)
    # ---- overlap-coupled
    if _MUTANT == "w_spin_sum":
        W = 0.5 * D @ (0.5 * (Fa + Fb)) @ D
    else:
        W = sum(Ds @ Fs @ Ds for Ds, Fs in spins)
    DSDs = sum(Ds @ S @ Ds for Ds, _ in spins)
    M = -W + c0 * (cell.Z.sum() - N) * D + hyb * c0 * DSDs
    if _MUTANT == "madelung_total":
        M = M - 0.5 * hyb * vM * D @ S @ D
    else:
        M = M - hyb * vM * DSDs
    XS = PGd._latsum_ip(cell, "int1e_ipovlp_cart", rcut_1e)
    XT = PGd._latsum_ip(cell, "int1e_ipkin_cart", rcut_1e)
    parts["S"] = PGd._fold(-2 * np.einsum("xmn,mn->mx", XS, M), aoat, natm)
    parts["T"] = PGd._fold(-2 * np.einsum("xmn,mn->mx", XT, D), aoat, natm)
    # ---- long range V_ne and ERI
    Praw, Q = PGd.pair_ft_deriv(cell, G)
    P0 = pair_ft(cell, np.zeros((1, 3)))[..., 0].real
    nrm = np.sqrt(np.diag(S) / np.diag(P0))
    nn = nrm[:, None] * nrm[None, :]
    Q = Q * nn[None, :, :, None]
    G2 = np.einsum("gi,gi->g", G, G)
    v = 4 * np.pi / G2 * (np.exp(-G2 / (4 * w * w)) if w is not None else 1.0)
    SG_at = np.exp(-1j * G @ cell.R.T) * cell.Z
    SG = SG_at.sum(1)
    rho = np.einsum("mn,mng->g", D, P)
    rq = PGd._fold_g(2 * np.einsum("mn,xmng->mxg", D, Q), aoat, natm)
    parts["Vlr_basis"] = -(np.einsum("g,axg->ax", v, np.conj(rq) * SG).real) / cell.vol
    parts["Vlr_nuc"] = -(np.einsum("g,ga,gx->ax", v * np.conj(rho), SG_at * (-1j), G).real) / cell.vol
    if _MUTANT == "exch_total":
        xpart = 0.25 * hyb * np.einsum("ml,lsg,sn->mng", D, P, D)
    else:
        xpart = 0.5 * hyb * sum(np.einsum("ml,lsg,sn->mng", Ds, P, Ds) for Ds, _ in spins)
    Zt = 0.5 * D[:, :, None] * rho[None, None, :] - xpart
    parts["Ilr"] = PGd._fold(4 / cell.vol * np.einsum("g,xmng,mng->mx", v, np.conj(Q), Zt).real, aoat, natm)
    # ---- short range (Ewald split)
    if w is not None:
        gb = PGd.sr_vne_grad(cell, D, w, rcut_1e)
        parts["Vsr_basis"], parts["Vsr_nuc"] = gb.sum(1), -gb.sum(0)
        if _MUTANT == "exch_total":
            X2 = 0.25 * hyb * np.einsum("ml,ns->mnls", D, D)
        else:
            X2 = 0.5 * hyb * sum(np.einsum("ml,ns->mnls", Ds, Ds) for Ds, _ in spins)
        Gam = 0.5 * np.einsum("mn,ls->mnls", D, D) - X2
        parts["Isr"] = PGd.sr_eri_grad(cell, Gam, w, rcut_2e or (4.5 / w + 8.0), rcut_bra or 12.0)
    parts["nn"] = PGd.ewald_grad(cell, w if w is not None else 1.0)
    if grid is not None and str(xc).upper() != "HF":
        xg = xc_grad(grid, Da, Db, xc, restricted, natm, aoat, response)
        parts["xc_ao"], parts["xc_point"], parts["xc_weight"] = xg["ao"], xg["point"], xg["weight"]
    return sum(parts.values()), parts


# ============================================================================================= convenience
def integrals(cell, w=None, exxdiv="none", gcut=None, **kw):
    return build_integrals(cell, w, verbose=False, exxdiv=exxdiv, gcut=gcut, **kw)


def run_case(cell, na, nb, xc, ints, grid=None, restricted=False, guess=None, tol=1e-11, w=None, response=True,
             mix=0.0, **kw):
    """Converge and differentiate.  Returns dict(e, grad, parts, scf)."""
    jk = pd.dense_jk(ints["I"])
    r = scf(ints["S"], ints["h"], jk, ints["enn"], na, nb, grid, xc, ints["madelung"], restricted, guess, tol, mix=mix)
    g, parts = gamma_grad(cell, ints, r["Da"], r["Db"], r["Fa"], r["Fb"], r["hyb"], w=w,
                          grid=None if str(xc).upper() == "HF" else grid, xc=xc, restricted=restricted,
                          response=response, **kw)
    return dict(e=r["e"], grad=g, parts=parts, scf=r)


def energy_only(cell, na, nb, xc, ints, grid, restricted, guess, tol=1e-11):
    jk = pd.dense_jk(ints["I"])
    return scf(ints["S"], ints["h"], jk, ints["enn"], na, nb, grid, xc, ints["madelung"], restricted, guess, tol)["e"]
