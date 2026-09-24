#!/usr/bin/env python3
"""COSX analytic nuclear-gradient prototype (numpy/PySCF), validated by FD.

PURPOSE
-------
ferric's `k_builder = "cosx"` computes K seminumerically, but every gradient
path differentiates EXACT exchange.  Before any Rust is written (repo rule:
prototype new methods in Python first), this script:

  1. re-implements ferric's COSX energy expression (crates/ferric-scf/src/
     cosx_k.rs, read 2026-09-24) inside a PySCF RHF / hybrid-RKS SCF,
  2. derives and implements the analytic gradient of THAT energy,
  3. checks it against central finite differences of the SAME prototype
     energy (tight SCF), and
  4. measures what the overlap fit does to the gradient.

THE ENERGY FERRIC COMPUTES (cosx_k.rs, trusting the code over the docs)
-----------------------------------------------------------------------
  grid     : its OWN atom-centred Becke-Lebedev grid (ferric: Treutler-
             Ahlrichs M4 radial x Lebedev, Becke 1988 partition with
             Bragg-Slater size adjustment, unpruned by default; points
             move rigidly with their home atom).  Here: PySCF's Becke grid
             of the same structure -- the derivation does not depend on the
             radial scheme, only on "atom-centred + Becke partition".
  X_{mu g} = sqrt(w_g) chi_mu(r_g)
  F        = D X,   G_g = A^g F_g,   A^g_{mu nu} = int chi_mu chi_nu / |r - r_g|
  Ktilde   = X G^T
  K_plain  = sym(Ktilde)                             (overlap_fit = false)
  K_fit    = sym(Q Ktilde), Q = S S_num^{-1}, S_num = X X^T   (default true)
  SCF      : F_fock = h + J - (c/2) K(D),  E = 1/2 tr D (h + F_fock)
             => E_x = -(c/4) tr[D K(D)]           (D = total RHF density)
  Screens (density-driven pair screen 1e-7, sparse half transforms 1e-10)
  only drop contributions below those thresholds; they are ignored here and
  their effect on the Rust gradient is measured, not assumed, in Rust.

THE GRADIENT (overlap_fit = false: plain COSX)
----------------------------------------------
  E_x = -(c/4) sum_g w_g e_g,   e_g = F_g^T A^g F_g,  F_g = D phi_g (unweighted AOs)

  tr[D K_plain(D)] = sum_g w_g e_g is a SYMMETRIC quadratic form in D and
  dE_x/dD = -(c/2) K_plain(D) -- exactly the exchange part of the Fock matrix
  the SCF used.  So the plain-COSX SCF energy is variational, and the
  gradient is the ordinary Hellmann-Feynman + Pulay (-W dS) form with the
  exact-K 4-centre derivative REPLACED by the explicit derivative of
  sum_g w_g e_g at fixed D:

    dE_x/dR_A = -(c/4) sum_g [ dw_g/dR_A e_g + w_g de_g/dR_A ]

  e_g depends on the basis centres (via phi and A^g) and on r_g (via phi and
  the charge point of A^g).  r_g = R_home(g) + const, so

    de_g/dR_A = d_A e_g |_basis + delta_{A,home(g)} d e_g/d r_g,
    d e_g/d r_g = - sum_B d_B e_g |_basis       (translation invariance of e_g)

  and the basis-centre partials are (nabla = electron-coordinate gradient)

    d_B e_g|_basis = -2 sum_{mu in B} nabla phi_mu(r_g) (D G_g)_mu        (a) AO
                     -2 sum_{lam in B} F_lam (A^g_ip F)_lam                (b) ESP ints
    A^g_ip[lam,sig] = < nabla lam | 1/|r-r_g| | sig >  (PySCF int1e_grids_ip)

  and dw_g/dR_A is the Becke grid-weight response (c) (PySCF weight1 from
  grids_response_cc; ferric: build_atomic_grid_with_response's weight1,
  same convention -- the total derivative with the point riding its atom).
  There is no (d): no fit.  This is the structure of Plessow & Weigend,
  J. Comput. Chem. 33, 810 (2012) (seminumerical-exchange gradients with a
  moving, weight-responsive grid) and of PySCF's sgx/grad/rhf.py
  grid-response branch (Bystrom), which this follows.

THE OVERLAP FIT (ferric's default)
----------------------------------
  tr[D K_fit(D)] = sum_g w_g phi_g^T D A^g D Q phi_g is NOT symmetric in its
  two D's (Q != I, Q != Q^T), so dE/dD != the fitted Fock matrix the SCF
  used: the fitted SCF energy is NOT stationary and an exact gradient needs
  the orbital response (a Z-vector / CPHF solve through the fitted K).
  PySCF refuses this too ("SGX-JK neglects the overlap fitting contribution
  to the gradient ... set fit_ovlp=False" / "_symm_ovlp_fit=True for exact
  gradients" -- a DIFFERENT, symmetric fit, i.e. a different energy).  This
  script MEASURES the size of the problem: the explicit (fixed-D, all of
  dQ = dS S_num^-1 - Q dS_num S_num^-1 included) derivative of the fitted
  energy vs FD of the fitted SCF energy.  Their difference is the orbital
  response the fit makes necessary.

PRE-REGISTERED (written before the first run)
---------------------------------------------
  ANCHOR: the plain-COSX analytic gradient equals central FD (h = 1e-4 Bohr,
  SCF conv_tol 1e-12) to <= 1e-7 Ha/Bohr on water/STO-3G and water/6-31G,
  RHF and B3LYP.  The FD error of a central difference at h = 1e-4 is
  O(h^2 E''') ~ 1e-9 and SCF noise ~1e-12/1e-4 = 1e-8, so 1e-7 is reachable
  and a missing term (a), (b) or (c) (each ~1e-3..1e-5, measured below by
  dropping it) cannot hide under it.
  NEGATIVE CONTROL: exact-K gradient vs FD(COSX energy) ~1e-5 (the reported
  defect) -- the harness must see it.
  ARTIFACT HYPOTHESIS: a sign error in A_ip or a missing home-atom term would
  show as an error that does NOT fall as the grid is refined? No -- it would
  show as a grid-INDEPENDENT O(1e-3) mismatch vs FD; a real residual (FD
  noise) is ~1e-8 and flat.  Per-term ablation prints each term's size so a
  pass cannot come from terms that are all negligible.

RESULTS (2026-09-24, PySCF 2.13.1, one run, 12 min wall, OPENBLAS=4)
------------------------------------------------------------------
Distorted water, (30,110) Becke grid (9900 pts), SCF conv 1e-12, FD h=1e-4:

  basis   method  fit | max|analytic - FD| | max|exactK - FD| (old pairing)
  sto-3g  HF      0   |      2.11e-09      |      3.28e-04
  sto-3g  B3LYP   0   |      1.87e-09      |      6.54e-05
  6-31g   HF      0   |      1.46e-09      |      1.02e-04
  6-31g   B3LYP   0   |      1.85e-09      |      2.15e-05
  sto-3g  HF      1   |      1.06e-06  (explicit-only; NOT exact, see above)
  6-31g   HF      1   |      1.37e-06  (explicit-only)

  Ablation (every term load-bearing; dropping one, max|grad - FD|):
    AO term (a) 1.7e-01..8.6e-01, ESP-integral term (b) 7.0e-02..4.0e-01,
    grid-weight term (c) 1.5e-01..7.8e-01.
  => PLAIN ANCHOR PASS (<= 1e-7 on all four, at ~2e-9 = the FD floor).
  => NEGATIVE CONTROL SEEN: the old pairing is 2e-5..3e-4 off, 1e4..1e5x
     above the new residual.
  => FIT: the explicit derivative leaves 1.1-1.4e-6 Ha/Bohr, the orbital
     response a non-variational energy needs.  An exact fitted gradient needs
     a Z-vector (CPHF through K_fit and its adjoint) -- not attempted here.

  Trivial-limit (grid refinement), water/6-31G RHF, max|g_COSX - g_exact|
  with each gradient taken at its OWN converged SCF:
    (20,50) 1.72e-03 | (30,110) 1.03e-04 | (50,110) 1.15e-04 | (75,302) 5.30e-07
  NOT monotone in point count: (50,110) is marginally worse than (30,110)
  (same angular order, more radial points).  The angular order drives it;
  bounded, and falls 3 decades by (75,302).  |E_COSX - E_exact| over the
  same grids: 6.9e-4 / 4.7e-5 / 2.4e-5 / 6.7e-11.

Run:  /home/matt/qc/ferric/.venv/bin/python scripts/cosx_gradient_proto.py
"""

import sys

import numpy as np
import scipy.linalg
from pyscf import dft, gto, scf
from pyscf.grad.rks import grids_response_cc

np.set_printoptions(precision=3, suppress=False, linewidth=150)

WATER = [
    ("O", (0.00, 0.02, 0.11)),
    ("H", (0.03, 0.76, -0.47)),
    ("H", (-0.02, -0.75, -0.45)),
]  # Angstrom; deliberately non-symmetric so every Cartesian component is live


def make_mol(atoms, basis):
    return gto.M(atom=atoms, basis=basis, unit="Angstrom", verbose=0)


class RespGrid:
    """Energy grid built FROM grids_response_cc, so the energy's points and
    weights are by construction the ones the gradient differentiates (PySCF's
    Grids.build may sort/pad/drop points; ferric's CosxK uses the raw
    atom-major Becke grid, which is what this reproduces)."""

    def __init__(self, mol, grid):
        g = dft.gen_grid.Grids(mol)
        g.atom_grid = grid
        g.prune = None
        self.pyscf_grids = g
        cs, ws = [], []
        for c, w, _ in grids_response_cc(g):
            cs.append(c)
            ws.append(w)
        self.coords = np.concatenate(cs)
        self.weights = np.concatenate(ws)


def make_grids(mol, grid):
    return RespGrid(mol, grid)


# ---------------------------------------------------------------- energy side
def cosx_k(mol, dm, grids, fit, s_ao=None):
    """ferric's COSX K (cosx_k.rs working equations), returns K (symmetrized)."""
    coords, w = grids.coords, grids.weights
    ao = mol.eval_gto("GTOval", coords)  # (ng, nao)
    x = (ao * np.sqrt(np.abs(w))[:, None]).T  # (nao, ng)
    f = dm @ x
    a = mol.intor("int1e_grids", grids=coords)  # (ng, nao, nao), +1/|r-rg|
    g = np.einsum("gmn,ng->mg", a, f)
    kt = x @ g.T
    if not fit:
        return 0.5 * (kt + kt.T)
    snum = x @ x.T
    z = scipy.linalg.cho_solve(scipy.linalg.cho_factor(snum), kt)
    qk = s_ao @ z
    return 0.5 * (qk + qk.T)


def run_scf(mol, grid, fit, xc=None, conv=1e-12):
    grids = make_grids(mol, grid)
    s_ao = mol.intor("int1e_ovlp")
    mf = scf.RHF(mol) if xc is None else dft.RKS(mol, xc=xc)
    if xc is not None:
        mf.grids.atom_grid = (50, 194)
        mf.grids.prune = None
        # XC grid response is on in the XC gradient below, so the XC part is
        # exact; only the exchange part is under test.
    cx = 1.0 if xc is None else mf._numint.hybrid_coeff(xc)

    def get_jk(mol_, dm, hermi=1, with_j=True, with_k=True, omega=None):
        vj = scf.hf.get_jk(mol_, dm, with_k=False)[0] if with_j else None
        vk = cosx_k(mol_, dm, grids, fit, s_ao) if with_k else None
        return vj, vk

    mf.get_jk = get_jk
    mf.conv_tol = conv
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged, "SCF did not converge"
    return mf, grids, cx


def energy(atoms, basis, grid, fit, xc=None):
    return run_scf(make_mol(atoms, basis), grid, fit, xc)[0].e_tot


# -------------------------------------------------------------- gradient side
def exchange_grad(mol, dm, grids, fit, terms=("ao", "int", "w")):
    """d/dR of  T = tr[D K(D)] = sum_g w_g e_g  at FIXED D (the explicit part).

    Plain: exact gradient of T.  Fit: explicit part only (see module doc).
    `terms` lets a term be dropped to show it is load-bearing.
    """
    natm = mol.natm
    aoslice = mol.aoslice_by_atom()
    ao_atom = np.empty(mol.nao, dtype=int)
    for a in range(natm):
        ao_atom[aoslice[a, 2] : aoslice[a, 3]] = a
    out = np.zeros((natm, 3))

    if fit:
        # geometry-only fit pieces
        ao_all = mol.eval_gto("GTOval", grids.coords)
        snum = (ao_all * grids.weights[:, None]).T @ ao_all
        s_ao = mol.intor("int1e_ovlp")
        sninv = np.linalg.inv(snum)  # prototype only; ferric uses Cholesky solves
        q = s_ao @ sninv
        # Kt = X G^T needed for dQ terms
        kt = np.zeros((mol.nao, mol.nao))
    else:
        q = np.eye(mol.nao)

    dq_mat = dm @ q  # H_g = D Q phi_g   (= F_g when plain)
    per_point = []  # (home, w, w1, e_g, de_basis (natm,3))  gathered per batch
    for home, (coords, w, w1) in enumerate(grids_response_cc(grids.pyscf_grids)):
        # NOTE: PySCF grids_response_cc yields points grouped by home atom
        # in the SAME order as grids.coords/weights (becke original scheme).
        ao_d = mol.eval_gto("GTOval_sph_deriv1", coords)  # (4, ng, nao)
        phi, dphi = ao_d[0], ao_d[1:4]
        a = mol.intor("int1e_grids", grids=coords)  # (ng, nao, nao)
        aip = mol.intor(
            "int1e_grids_ip", grids=coords
        )  # (3, ng, nao, nao) <nabla mu|..|nu>
        f = phi @ dm  # (ng, nao)  F_g (D symmetric)
        h = phi @ dq_mat.T  # (ng, nao)  H_g = D Q phi_g
        ga = np.einsum("gmn,gn->gm", a, f)  # A F
        ha = np.einsum("gmn,gn->gm", a, h)  # A H
        e = np.einsum("gm,gm->g", f, ha)  # e_g = F^T A H
        if fit:
            kt += (phi * w[:, None]).T @ ga
        # basis-centre partials, per point per atom
        dbas = np.zeros((len(w), natm, 3))
        if "ao" in terms:
            # phi enters F (through D) and H (through D Q): M1 = D A D Q
            # d e / d phi = D A H + Q^T D A F  -> (D ha + Q^T D ga)
            v = ha @ dm + ga @ (dm @ q)  # rows: (D A H)^T + (Q^T D A F)^T
            for comp in range(3):
                contrib = -dphi[comp] * v  # (ng, nao)
                for b in range(natm):
                    sl = slice(aoslice[b, 2], aoslice[b, 3])
                    dbas[:, b, comp] += contrib[:, sl].sum(axis=1)
        if "int" in terms:
            # sum F_lam H_sig dA_lam,sig/dR_B = -sum_{lam in B} F_lam (ip H)_lam
            #                                   -sum_{sig in B} H_sig (ip F)_sig
            for comp in range(3):
                iph = np.einsum("gmn,gn->gm", aip[comp], h)
                ipf = np.einsum("gmn,gn->gm", aip[comp], f)
                contrib = -(f * iph + h * ipf)
                for b in range(natm):
                    sl = slice(aoslice[b, 2], aoslice[b, 3])
                    dbas[:, b, comp] += contrib[:, sl].sum(axis=1)
        # total per-point derivative: basis part + home gets -sum (r_g rides home)
        dtot = dbas.copy()
        dtot[:, home, :] -= dbas.sum(axis=1)
        out += np.einsum("g,gbx->bx", w, dtot)
        if "w" in terms:
            out += np.einsum("bxg,g->bx", w1, e)
        per_point.append((home, coords, w, w1, phi, dphi))

    if fit:
        # dQ terms: T = tr(D Q Kt) = tr(Q M), M = Kt D  (Kt = sum w phi (A F)^T)
        m = kt @ dm
        p1 = sninv @ m  # tr(dS P1)
        p2 = sninv @ m @ s_ao @ sninv  # -tr(dS_num P2)
        ipov = mol.intor("int1e_ipovlp")  # (3,nao,nao) <nabla mu|nu>
        for b in range(natm):
            sl = slice(aoslice[b, 2], aoslice[b, 3])
            for comp in range(3):
                # dS_mu nu/dR_B = -(ip[mu,nu] mu in B + ip[nu,mu] nu in B)
                ds = np.zeros((mol.nao, mol.nao))
                ds[sl, :] -= ipov[comp][sl, :]
                ds[:, sl] -= ipov[comp][sl, :].T
                out[b, comp] += np.sum(ds * p1.T)
        p2s = p2 + p2.T
        for home, coords, w, w1, phi, dphi in per_point:
            s_g = np.einsum("gm,mn,gn->g", phi, p2, phi)
            if "w" in terms:
                out -= np.einsum("bxg,g->bx", w1, s_g)
            if "ao" in terms:
                dbas = np.zeros((len(w), natm, 3))
                v = phi @ p2s.T  # (P2 + P2^T) phi
                for comp in range(3):
                    contrib = -dphi[comp] * v
                    for b in range(natm):
                        sl = slice(aoslice[b, 2], aoslice[b, 3])
                        dbas[:, b, comp] += contrib[:, sl].sum(axis=1)
                dtot = dbas.copy()
                dtot[:, home, :] -= dbas.sum(axis=1)
                out -= np.einsum("g,gbx->bx", w, dtot)
    return out


def analytic_grad(mf, grids, cx, fit, terms=("ao", "int", "w")):
    """Full SCF gradient with the exchange part from COSX.

    The non-exchange part is PySCF's own gradient with its K derivative
    switched off (vk = 0 in the gradient's get_jk), keeping hcore, J,
    -W dS (W from the COSX SCF's own orbitals/energies), V_nn and -- for
    RKS -- XC including grid response.
    """
    mol = mf.mol
    dm = mf.make_rdm1()
    gobj = mf.nuc_grad_method()
    if hasattr(gobj, "grid_response"):
        gobj.grid_response = True
    base_get_jk = gobj.get_jk

    def get_jk_noK(mol_=None, dm_=None, hermi=0, omega=None):
        vj, vk = base_get_jk(mol_, dm_, hermi, omega)
        return vj, np.zeros_like(vk)

    gobj.get_jk = get_jk_noK
    g0 = gobj.kernel()
    gx = exchange_grad(mol, dm, grids, fit, terms)
    return g0 - 0.25 * cx * gx, g0


def exactk_grad(mf):
    """Old pairing: COSX SCF solution, EXACT 4-centre K derivative."""
    gobj = mf.nuc_grad_method()
    if hasattr(gobj, "grid_response"):
        gobj.grid_response = True
    return gobj.kernel()


def fd_grad(atoms, basis, grid, fit, xc=None, h=1e-4):
    bohr = 0.52917721092
    out = np.zeros((len(atoms), 3))
    for i in range(len(atoms)):
        for c in range(3):
            ep = []
            for s in (+1, -1):
                at = [(sym, list(xyz)) for sym, xyz in atoms]
                at[i][1][c] += s * h * bohr
                ep.append(energy(at, basis, grid, fit, xc))
            out[i, c] = (ep[0] - ep[1]) / (2 * h)
    return out


def case(basis, grid, fit, xc=None, ablate=True):
    mol = make_mol(WATER, basis)
    mf, grids, cx = run_scf(mol, grid, fit, xc)
    ga, g0 = analytic_grad(mf, grids, cx, fit)
    gfd = fd_grad(WATER, basis, grid, fit, xc)
    gex = exactk_grad(mf)
    tag = f"{basis:8s} {str(grid):10s} {'HF' if xc is None else xc:6s} fit={int(fit)}"
    err = np.abs(ga - gfd).max()
    err_old = np.abs(gex - gfd).max()
    print(
        f"{tag}  npts={len(grids.weights):6d}  max|analytic-FD| = {err:.2e}   "
        f"max|exactK-FD| (old pairing) = {err_old:.2e}",
        flush=True,
    )
    if ablate:
        for drop in ("ao", "int", "w"):
            keep = tuple(t for t in ("ao", "int", "w") if t != drop)
            gd, _ = analytic_grad(mf, grids, cx, fit, keep)
            print(
                f"    drop term {drop:3s}: max|grad - FD| = {np.abs(gd - gfd).max():.2e}"
            )
    return err, err_old, ga, gex


def main():
    print("=== PLAIN COSX (overlap_fit = false): exact gradient vs FD ===")
    res = []
    for basis in ("sto-3g", "6-31g"):
        for xc in (None, "b3lyp"):
            res.append(case(basis, (30, 110), False, xc))
    print()
    print("=== grid refinement (trivial limit): |g_COSX - g_exactK-on-exact-SCF| ===")
    mol = make_mol(WATER, "6-31g")
    ref = scf.RHF(mol)
    ref.conv_tol = 1e-12
    ref.kernel()
    g_exact = ref.nuc_grad_method().kernel()
    for grid in ((20, 50), (30, 110), (50, 110), (75, 302)):
        mf, grids, cx = run_scf(mol, grid, False)
        ga, _ = analytic_grad(mf, grids, cx, False)
        print(
            f"  {str(grid):10s} npts={len(grids.weights):6d}  "
            f"max|g_cosx - g_exact| = {np.abs(ga - g_exact).max():.2e}  "
            f"|E_cosx - E_exact| = {abs(mf.e_tot - ref.e_tot):.2e}"
        )
    print()
    print("=== OVERLAP FIT (ferric default): explicit-only gradient vs FD ===")
    for basis in ("sto-3g", "6-31g"):
        case(basis, (30, 110), True, None, ablate=False)
    fails = [r for r in res if r[0] > 1e-7]
    print()
    print("PLAIN ANCHOR (<= 1e-7 on all four):", "PASS" if not fails else "FAIL")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
