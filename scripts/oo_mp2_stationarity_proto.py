#!/usr/bin/env python3
"""OO-RI-MP2 stationarity prototype (validation campaign defect F2).

Rebuilds ferric's OO-RI-MP2 energy functional on PySCF integrals and checks its
orbital and nuclear-gradient machinery against finite differences. Everything
here is an INDEPENDENT dense construction (numpy on PySCF integrals); it reuses
no ferric code, only mirrors the formulas ferric implements so they can be
tested in isolation.

What ferric computes (crates/ferric-mp2/src/oo_rimp2.rs):
  * HF part: EXACT 4-centre J/K (build_jk_with_pool), F = h + J[D] - K[D]/2,
    D = 2 C_occ C_occ^T, E_HF = tr(D (h+F))/2 + V_nn.
  * MP2 part: RI, B^P_pq = sum_Q V^{-1/2}_PQ (Q|pq), and orbital energies read
    off the DIAGONAL of C^T F C ("diag" functional below).
  * Orbital rotation: C <- C U(kappa), Cayley U = (I - k/2)^{-1}(I + k/2),
    kappa[a,i] = k_ai, kappa[i,a] = -k_ai; gradient g_ai = +dE/dk_ai at k=0
    around the CURRENT C (the solver re-references every macro-iteration).

Two functionals are compared:
  * "diag":     E_HF(C) + MP2 with eps_p = (C^T F C)_pp        (ferric today)
  * "textbook": E_HF(C) + MP2 in the SEMICANONICAL basis of C (diagonalise the
                occ-occ and vir-vir blocks of C^T F C first). This equals the
                full-Fock non-canonical Hylleraas OMP2 functional and is
                invariant to occ-occ and vir-vir rotations.

Subcommands (all tiny systems, OPENBLAS_NUM_THREADS=1, a few seconds each):
  grad   (a)+(b)  analytic orbital gradient vs 4-point central FD at a reference
                  C0 U(kappa) with random antisymmetric kappa, |kappa| in
                  {0, 0.05, 0.2}, for three formulas:
                    old   = ferric's shipped closed form (RI d eps_p/dk, no
                            F_ov rotation term)
                    diag  = corrected gradient of the diag functional
                    tb    = gradient of the textbook functional
  solve  (c)      converge both functionals with a ferric-like solver and FD
                  the converged point; also the non-orthonormality ferric's
                  C-DIIS leaves behind.
  gap    (d)      diag vs textbook energies, the diag functional's residual
                  occ-occ / vir-vir gradient, and its path dependence.
  psi4            prototype anchor: textbook OMP2 with exact integrals vs Psi4.
  open   (f)      the same checks for U-OO-RI-MP2 (u_oo_rimp2.rs).
  nuc    (e)      nuclear-gradient ingredients: re-converged dE under an
                  overlap perturbation S -> S + e R and an hcore perturbation
                  h -> h + e R vs the energy-weighted density / 1-PDM that
                  oo_rimp2_gradient.rs contracts with dS/dR and dh/dR.

Run:  cd ~/qc/ferric && OPENBLAS_NUM_THREADS=1 nice uv run --no-sync \
        python scripts/oo_mp2_stationarity_proto.py all
"""

from __future__ import annotations

import sys

import numpy as np
import scipy.linalg as sla
from pyscf import df, gto, scf

np.set_printoptions(linewidth=160)

GEOMS = {
    "h2": "H 0 0 0; H 0 0 0.74",
    "h2o": "O 0.000000 0.000000 0.117790; H 0.000000 0.755453 -0.471161; "
    "H 0.000000 -0.755453 -0.471161",
    "nh3": "N 0.0000 0.0000 0.1162; H 0.0000 0.9397 -0.2711; "
    "H 0.8138 -0.4699 -0.2711; H -0.8138 -0.4699 -0.2711",
    # Psi4 tests/omp2-1 geometry (O; H 1 0.958; H 1 0.958 2 104.4776), as in
    # oo_rimp2.rs::test_oo_rimp2_h2o_ccpvdz_matches_psi4_omp2_reference
    "h2o_psi4": "O 0 0 0; H 0 0 0.958; H 0 0.927579144347 -0.239501421649",
    "oh": "O 0 0 0; H 0 0 0.97",
    "nh2": "N 0 0 0.1400; H 0 0.8003 -0.4900; H 0 -0.8003 -0.4900",
}


# ----------------------------------------------------------------------------
# Integrals (mutable: the "nuc" checks perturb S and h in place)
# ----------------------------------------------------------------------------
class System:
    def __init__(
        self, name: str, basis: str, auxbasis: str = "cc-pvdz-ri", spin: int = 0
    ):
        """auxbasis="exact": B from an eigen-decomposition of the full ERI
        supermatrix, i.e. CONVENTIONAL MP2 in the same code path."""
        self.name = f"{name}/{basis}" + (
            "" if auxbasis == "cc-pvdz-ri" else f"[{auxbasis}]"
        )
        mol = gto.M(atom=GEOMS[name], basis=basis, spin=spin, verbose=0)
        self.mol = mol
        self.S = mol.intor("int1e_ovlp")
        self.h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
        self.eri = mol.intor("int2e", aosym="s1")
        self.enuc = mol.energy_nuc()
        nao = mol.nao
        if auxbasis == "exact":
            w, v = np.linalg.eigh(self.eri.reshape(nao * nao, nao * nao))
            keep = w > 1e-13
            self.Bao = (v[:, keep] * np.sqrt(w[keep])).T.reshape(-1, nao, nao)
        else:
            auxmol = df.addons.make_auxmol(mol, auxbasis)
            j3 = df.incore.aux_e2(mol, auxmol, "int3c2e", aosym="s1")  # (nao,nao,naux)
            j2 = auxmol.intor("int2c2e")
            # V^{-1/2} dressing; any square root of V^{-1} gives the same B^T B.
            w, v = np.linalg.eigh(j2)
            vmh = (v / np.sqrt(w)) @ v.T
            self.Bao = np.einsum("PQ,mnQ->Pmn", vmh, j3)
        self.nao = mol.nao
        self.nalpha, self.nbeta = mol.nelec
        self.nocc = mol.nelectron // 2
        self.nvir = self.nao - self.nocc

    # ---- mean field -------------------------------------------------------
    def jk(self, d):
        j = np.einsum("mnls,ls->mn", self.eri, d)
        k = np.einsum("mlns,ls->mn", self.eri, d)
        return j, k

    def G(self, d):
        """F = h + G[D] with D the full (doubly occupied) density."""
        j, k = self.jk(d)
        return j - 0.5 * k

    def fock(self, c):
        o = self.nocc
        d = 2.0 * c[:, :o] @ c[:, :o].T
        f = self.h + self.G(d)
        e = 0.5 * np.sum(d * (self.h + f)) + self.enuc
        return e, f, d

    def rhf(self):
        mf = scf.RHF(self.mol)
        mf.conv_tol = 1e-13
        mf.conv_tol_grad = 1e-10
        mf.get_ovlp = lambda *a: self.S
        mf.get_hcore = lambda *a: self.h
        mf._eri = self.eri.reshape(self.nao**2, self.nao**2)
        mf.kernel()
        assert mf.converged
        return mf.mo_coeff.copy()

    # ---- MP2 pieces -------------------------------------------------------
    def bmo(self, c):
        return np.einsum("Pmn,mp,nq->Ppq", self.Bao, c, c, optimize=True)

    def amplitudes(self, c, eps):
        o = self.nocc
        b = self.bmo(c)
        bov = b[:, :o, o:]
        k = np.einsum("Pia,Pjb->iajb", bov, bov)
        eo, ev = eps[:o], eps[o:]
        dd = (
            eo[:, None, None, None]
            + eo[None, None, :, None]
            - ev[None, :, None, None]
            - ev[None, None, None, :]
        )
        t = k / dd
        tt = 2.0 * t - t.transpose(0, 3, 2, 1)
        emp2 = np.sum(t * (2.0 * k - k.transpose(0, 3, 2, 1)))
        return b, k, t, tt, emp2


def cayley(kappa):
    n = kappa.shape[0]
    i = np.eye(n)
    return np.linalg.solve(i - 0.5 * kappa, i + 0.5 * kappa)


def kappa_from_ov(kov, nocc):
    nv, no = kov.shape
    k = np.zeros((no + nv, no + nv))
    k[nocc:, :nocc] = kov
    k[:nocc, nocc:] = -kov.T
    return k


def semicanonicalize(sysm, c):
    o = sysm.nocc
    _, f, _ = sysm.fock(c)
    fmo = c.T @ f @ c
    wo, uo = np.linalg.eigh(fmo[:o, :o])
    wv, uv = np.linalg.eigh(fmo[o:, o:])
    u = sla.block_diag(uo, uv)
    return c @ u, np.concatenate([wo, wv]), u


# ----------------------------------------------------------------------------
# Energies
# ----------------------------------------------------------------------------
def energy(sysm, c, kind):
    if kind == "tb":
        c, eps, _ = semicanonicalize(sysm, c)
        ehf, _, _ = sysm.fock(c)
    else:
        ehf, f, _ = sysm.fock(c)
        eps = np.diag(c.T @ f @ c)
    *_, emp2 = sysm.amplitudes(c, eps)
    return ehf + emp2


# ----------------------------------------------------------------------------
# Analytic orbital gradients (g_ai = +dE/dk_ai at k = 0 around c)
# ----------------------------------------------------------------------------
def mp2_densities(t):
    """ferric build_mp2_density: P_oo = -sum t(2t - t^swap), P_vv = +sum ..."""
    p_oo = -np.einsum("iakb,jakb->ij", t, 2 * t - t.transpose(0, 3, 2, 1))
    p_vv = np.einsum("iajc,ibjc->ab", t, 2 * t - t.transpose(0, 3, 2, 1))
    return p_oo, p_vv


def grad_pieces(sysm, c, eps):
    """HF term + MP2 integral response at fixed denominators (Term0..Term4)."""
    o = sysm.nocc
    _, f, _ = sysm.fock(c)
    fmo = c.T @ f @ c
    b, k, t, tt, _ = sysm.amplitudes(c, eps)
    # dE/dK_iajb = 2*Tt ; dK = dB_ia B_jb + B_ia dB_jb  ->  dE = 4 sum Y dB
    y = np.einsum("iajb,Pjb->Pia", tt, b[:, :o, o:])
    g_int = 4.0 * (
        np.einsum("Pia,Pca->ci", y, b[:, o:, o:])
        - np.einsum("Pka,Pki->ai", y, b[:, :o, :o])
    )
    g_hf = 4.0 * fmo[o:, :o]
    return fmo, b, t, g_hf, g_int


def grad_old(sysm, c):
    """ferric's shipped formula (oo_rimp2.rs compute_orbital_gradient_panelled):
    d eps_p/dk_ck = 4(pp|ck)_RI - 2(pc|pk)_RI, no F_ov rotation term."""
    o = sysm.nocc
    _, f, _ = sysm.fock(c)
    eps = np.diag(c.T @ f @ c)
    fmo, b, t, g_hf, g_int = grad_pieces(sysm, c, eps)
    p_oo, p_vv = mp2_densities(t)
    w = np.concatenate([2 * np.diag(p_oo), 2 * np.diag(p_vv)])  # dE/d eps_p
    bd = np.einsum("Ppp->Pp", b)
    pp_ck = np.einsum("Pp,Pck->pck", bd, b[:, o:, :o])
    pc_pk = np.einsum("Ppc,Ppk->pck", b[:, :, o:], b[:, :, :o])
    deps = 4.0 * pp_ck - 2.0 * pc_pk
    return g_hf + g_int + np.einsum("p,pck->ck", w, deps)


def pw_response(sysm, c, fmo, pw):
    """sum_pq Pw_pq dF_pq/dk_ck for a symmetric MO weight Pw (oo/vv blocks).

    dF_pq = dC_p^T F C_q + C_p^T F dC_q + C_p^T G[dD] C_q with dC_k = C_c,
    dC_c = -C_k, dD = 2 (C_c C_k^T + C_k C_c^T):
      = 2 (sum_q Pw_kq F_cq - sum_q Pw_cq F_kq) + 4 G[C Pw C^T]_ck
    """
    o = sysm.nocc
    rot = 2.0 * (fmo[o:, :] @ pw[:, :o] - pw[o:, :] @ fmo[:, :o])
    gp = c.T @ sysm.G(c @ pw @ c.T) @ c
    return rot + 4.0 * gp[o:, :o]


def grad_diag(sysm, c):
    """Corrected gradient of the diag functional: exact 4c G, plus rotation term."""
    _, f, _ = sysm.fock(c)
    eps = np.diag(c.T @ f @ c)
    fmo, b, t, g_hf, g_int = grad_pieces(sysm, c, eps)
    p_oo, p_vv = mp2_densities(t)
    pw = np.diag(np.concatenate([2 * np.diag(p_oo), 2 * np.diag(p_vv)]))
    return g_hf + g_int + pw_response(sysm, c, fmo, pw)


def grad_diag_parts(sysm, c):
    """Split old-vs-corrected diag gradient into (RI-vs-exact G, rotation term)."""
    _, f, _ = sysm.fock(c)
    eps = np.diag(c.T @ f @ c)
    fmo, _, t, _, _ = grad_pieces(sysm, c, eps)
    p_oo, p_vv = mp2_densities(t)
    pw = np.diag(np.concatenate([2 * np.diag(p_oo), 2 * np.diag(p_vv)]))
    o = sysm.nocc
    rot = 2.0 * (fmo[o:, :] @ pw[:, :o] - pw[o:, :] @ fmo[:, :o])
    return rot


def grad_tb(sysm, c):
    """Gradient of the textbook functional, evaluated at the semicanonical
    rotation of c and rotated back to c's own oo/vv gauge."""
    o = sysm.nocc
    csc, eps, u = semicanonicalize(sysm, c)
    fmo, b, t, g_hf, g_int = grad_pieces(sysm, csc, eps)
    p_oo, p_vv = mp2_densities(t)
    pw = sla.block_diag(p_oo + p_oo.T, p_vv + p_vv.T)
    g_sc = g_hf + g_int + pw_response(sysm, csc, fmo, pw)
    uo, uv = u[:o, :o], u[o:, o:]
    return uv @ g_sc @ uo.T


def fd_grad(sysm, c, kind, h=1e-3, blocks="ov"):
    """4-point central FD of E(c U(k)) in the ov (or oo/vv) rotation params."""
    o, n = sysm.nocc, sysm.nao
    if blocks == "ov":
        idx = [(a, i) for a in range(o, n) for i in range(o)]
    elif blocks == "oo":
        idx = [(i, j) for i in range(o) for j in range(i)]
    else:
        idx = [(a, b) for a in range(o, n) for b in range(o, a)]
    out = {}
    for p, q in idx:
        vals = []
        for s in (2, 1, -1, -2):
            k = np.zeros((n, n))
            k[p, q] = s * h
            k[q, p] = -s * h
            vals.append(energy(sysm, c @ cayley(k), kind))
        out[(p, q)] = (-vals[0] + 8 * vals[1] - 8 * vals[2] + vals[3]) / (12 * h)
    if blocks == "ov":
        g = np.zeros((n - o, o))
        for (a, i), v in out.items():
            g[a - o, i] = v
        return g
    return np.array(list(out.values()))


def random_kappa(n, scale, seed):
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((n, n))
    k = k - k.T
    return k * (scale / np.linalg.norm(k))  # Frobenius norm == scale


# ----------------------------------------------------------------------------
# (a)+(b)
# ----------------------------------------------------------------------------
def cmd_grad(systems):
    print("\n=== (a)+(b) analytic orbital gradient vs 4-point FD (h=1e-3) ===")
    print(
        "reference C = C_RHF U(kappa), kappa random antisymmetric (ALL blocks), ||kappa||_F given"
    )
    print(
        "ARTIFACT HYPOTHESIS: an FD-setup artifact is step-dependent and kappa-independent;"
    )
    print(
        "a real defect in the missing F_ov rotation term scales ~linearly with |kappa|"
    )
    print(
        "(F_ov(C0 U(kappa)) = O(kappa)), and an RI-vs-exact G error is present already at kappa=0."
    )
    print(
        f"{'system':16s} {'|k|':>5s} | {'old-FD':>9s} {'diag-FD':>9s} {'tb-FD':>9s} | {'|rotterm|':>9s} {'FD h/2':>9s}"
    )
    for sysm in systems:
        c0 = sysm.rhf()
        for scale in (0.0, 0.05, 0.2):
            c = c0 @ cayley(random_kappa(sysm.nao, scale, 7)) if scale else c0
            fdd = fd_grad(sysm, c, "diag")
            fdt = fd_grad(sysm, c, "tb")
            e_old = np.abs(grad_old(sysm, c) - fdd).max()
            e_diag = np.abs(grad_diag(sysm, c) - fdd).max()
            e_tb = np.abs(grad_tb(sysm, c) - fdt).max()
            rot = np.abs(grad_diag_parts(sysm, c)).max()
            fdd2 = fd_grad(sysm, c, "diag", h=5e-4)
            print(
                f"{sysm.name:16s} {scale:5.2f} | {e_old:9.2e} {e_diag:9.2e} {e_tb:9.2e} | {rot:9.2e} {np.abs(fdd2 - fdd).max():9.2e}"
            )


# ----------------------------------------------------------------------------
# (c) solver
# ----------------------------------------------------------------------------
def solve(
    sysm, c0, kind, gfun, shift=0.1, tol=1e-10, maxit=3000, cdiis=False, diis_n=6
):
    # NOTE: a Newton-consistent diagonal Hessian for this energy is ~4*(e_a - e_i)
    # (the RHF orbital Hessian), not ferric's (e_a - e_i); used here so the plain
    # (no-DIIS, no-backtracking) iteration contracts.
    """ferric-like solver: diagonal level-shifted Newton, re-reference each step.
    cdiis=True emulates ferric's DIIS on C itself (linear combination of C's)."""
    o = sysm.nocc
    c = c0.copy()
    hist_c, hist_e = [], []
    for it in range(1, maxit + 1):
        g = gfun(sysm, c)
        gn = np.linalg.norm(g)
        if gn < tol:
            return c, it, gn
        _, f, _ = sysm.fock(c)
        eps = np.diag(c.T @ f @ c)
        kov = -g / (4.0 * (eps[o:, None] - eps[None, :o]) + shift)
        m = np.abs(kov).max()
        if m > 0.3:
            kov *= 0.3 / m
        cn = c @ cayley(kappa_from_ov(kov, o))
        if cdiis:
            ga = kappa_from_ov(g, o)
            hist_c.append(cn)
            hist_e.append(cn @ ga @ cn.T)
            hist_c, hist_e = hist_c[-diis_n:], hist_e[-diis_n:]
            nh = len(hist_c)
            bm = -np.ones((nh + 1, nh + 1))
            bm[nh, nh] = 0
            for i in range(nh):
                for j in range(nh):
                    bm[i, j] = np.sum(hist_e[i] * hist_e[j])
            rhs = np.zeros(nh + 1)
            rhs[nh] = -1
            coef = np.linalg.lstsq(bm, rhs, rcond=None)[0][:nh]
            cn = sum(ci * cc for ci, cc in zip(coef, hist_c))
        c = cn
    return c, maxit, gn


def orth_err(sysm, c):
    return np.abs(c.T @ sysm.S @ c - np.eye(sysm.nao)).max()


def lowdin(sysm, c):
    m = c.T @ sysm.S @ c
    w, v = np.linalg.eigh(m)
    return c @ (v / np.sqrt(w)) @ v.T


def solve_ferric(
    sysm,
    c0,
    kind,
    gfun,
    grad_conv=1e-6,
    energy_conv=1e-8,
    shift=0.1,
    reorth=False,
    max_iter=200,
    diis_n=6,
):
    """Faithful emulation of oo_rimp2.rs::oo_ri_mp2's loop: step -g/(gap+mu),
    0.3 cap, DIIS on C with err = C_new g_antisym C_new^T, 1e-4 uphill
    backtracking with DIIS reset, and the energy_conv early exit that accepts
    |g| < 10*grad_conv. reorth=True Loewdin-orthonormalises the DIIS output."""
    o = sysm.nocc
    c = c0.copy()
    e = energy(sysm, c, kind)
    hist_c, hist_e = [], []
    for it in range(1, max_iter + 1):
        g = gfun(sysm, c)
        gn = np.linalg.norm(g)
        if gn < grad_conv:
            return c, it, gn
        _, f, _ = sysm.fock(c)
        eps = np.diag(c.T @ f @ c)
        kov = -g / (eps[o:, None] - eps[None, :o] + shift)
        m = np.abs(kov).max()
        if m > 0.3:
            kov *= 0.3 / m
        u = cayley(kappa_from_ov(kov, o))
        cn = c @ u
        hist_c.append(cn)
        hist_e.append(cn @ kappa_from_ov(g, o) @ cn.T)
        hist_c, hist_e = hist_c[-diis_n:], hist_e[-diis_n:]
        nh = len(hist_c)
        if nh > 1:
            bm = -np.ones((nh + 1, nh + 1))
            bm[nh, nh] = 0
            for i in range(nh):
                for j in range(nh):
                    bm[i, j] = np.sum(hist_e[i] * hist_e[j])
            rhs = np.zeros(nh + 1)
            rhs[nh] = -1
            coef = np.linalg.lstsq(bm, rhs, rcond=None)[0][:nh]
            cn = sum(ci * cc for ci, cc in zip(coef, hist_c))
        if reorth:
            cn = lowdin(sysm, cn)
        en = energy(sysm, cn, kind)
        de = abs(en - e)
        if en > e + 1e-4:
            bk = kov.copy()
            for _ in range(10):
                bk *= 0.5
                cn = c @ cayley(kappa_from_ov(bk, o))
                en = energy(sysm, cn, kind)
                if en <= e + 1e-12:
                    break
            hist_c, hist_e = [], []
        c, e = cn, en
        if de < energy_conv and it > 1:
            g2 = gfun(sysm, c)
            gn = np.linalg.norm(g2)
            if gn < 10 * grad_conv:
                return c, it, gn
    return c, max_iter, gn


def cmd_solve(systems):
    print(
        "\n=== (c) converged point: analytic |g| vs FD |g| (ferric-faithful solver) ==="
    )
    print(
        f"{'system':16s} {'variant':34s} {'it':>3s} {'|g|an':>8s} {'max|g_FD|':>9s} {'|CtSC-1|':>8s} {'E':>17s}"
    )
    for sysm in systems:
        c0 = sysm.rhf()
        for label, kind, gfun, conv, reorth in (
            ("shipped: diag/old, conv 1e-6", "diag", grad_old, 1e-6, False),
            ("diag/corrected grad, conv 1e-8", "diag", grad_diag, 1e-8, False),
            ("diag/corrected + reorth, 1e-8", "diag", grad_diag, 1e-8, True),
            ("FIX: tb + reorth, conv 1e-6", "tb", grad_tb, 1e-6, True),
            ("FIX: tb + reorth, conv 1e-8", "tb", grad_tb, 1e-8, True),
        ):
            c, it, gn = solve_ferric(
                sysm, c0, kind, gfun, grad_conv=conv, energy_conv=1e-11, reorth=reorth
            )
            gfd = np.abs(fd_grad(sysm, c, kind)).max()
            print(
                f"{sysm.name:16s} {label:34s} {it:3d} {gn:8.1e} {gfd:9.2e} {orth_err(sysm, c):8.1e} {energy(sysm, c, kind):17.12f}"
            )


# ----------------------------------------------------------------------------
# (d) functional gap
# ----------------------------------------------------------------------------
def _offdiag_max(m):
    return np.abs(m - np.diag(np.diag(m))).max()


def cmd_gap(systems):
    print("\n=== (d) diag-Fock vs textbook (full-Fock) OMP2 functional ===")
    for sysm in systems:
        c0 = sysm.rhf()
        e_rhf = energy(sysm, c0, "diag")  # canonical: diag == tb
        cd1, _, _ = solve(sysm, c0, "diag", grad_diag, shift=0.1)
        cd2, _, _ = solve(sysm, c0, "diag", grad_diag, shift=0.5)
        ct1, _, _ = solve(sysm, c0, "tb", grad_tb, shift=0.1)
        ct2, _, _ = solve(sysm, c0, "tb", grad_tb, shift=0.5)
        ed1, ed2 = energy(sysm, cd1, "diag"), energy(sysm, cd2, "diag")
        et1, et2 = energy(sysm, ct1, "tb"), energy(sysm, ct2, "tb")
        # textbook energy AT the diag-converged orbitals, and vice versa
        e_tb_at_d = energy(sysm, cd1, "tb")
        # residual oo/vv gradient of each functional at its own converged point
        goo_d = (
            np.abs(fd_grad(sysm, cd1, "diag", blocks="oo")).max()
            if sysm.nocc > 1
            else 0.0
        )
        gvv_d = np.abs(fd_grad(sysm, cd1, "diag", blocks="vv")).max()
        gvv_t = np.abs(fd_grad(sysm, ct1, "tb", blocks="vv")).max()
        _, f, _ = sysm.fock(cd1)
        fmo = cd1.T @ f @ cd1
        o = sysm.nocc
        print(f"{sysm.name}:")
        print(f"  E(RI-MP2 @ RHF)                         = {e_rhf:.12f}")
        print(
            f"  E_diag converged (shift 0.1 / 0.5)      = {ed1:.12f} / {ed2:.12f}  path spread {abs(ed1 - ed2):.2e}"
        )
        print(
            f"  E_tb   converged (shift 0.1 / 0.5)      = {et1:.12f} / {et2:.12f}  path spread {abs(et1 - et2):.2e}"
        )
        print(f"  E_diag - E_tb (each at its own minimum) = {ed1 - et1:+.3e} Ha")
        print(
            f"  E_tb at the diag-converged orbitals     = {e_tb_at_d:.12f} ({e_tb_at_d - et1:+.2e} above E_tb min)"
        )
        print(
            f"  diag-functional residual |dE/dk| oo, vv = {goo_d:.2e}, {gvv_d:.2e}   (tb vv: {gvv_t:.2e})"
        )
        print(
            f"  diag-converged max|F_oo offdiag|, |F_vv offdiag|, |F_ov| = {_offdiag_max(fmo[:o, :o]):.2e}, {_offdiag_max(fmo[o:, o:]):.2e}, {np.abs(fmo[o:, :o]).max():.2e}"
        )


# ----------------------------------------------------------------------------
# (e) nuclear-gradient ingredients: overlap / hcore perturbation
# ----------------------------------------------------------------------------
def rust_like_w(sysm, c, fixed=False):
    """The energy-weighted density oo_rimp2_gradient.rs contracts with dS/dR,
    w = im1 - zeta_ao (shipped), and the corrected version (fixed=True):
      + vhf_s1occ-like term 2 C_o G[dm]_oo C_o^T (subtracted), and
      zeta ov block = (F dm)_ia instead of (F dm + dm F)_ia / 2, and
      HF energy-weighted density 2 C_o F_oo C_o^T (full, not diag)."""
    o = sysm.nocc
    _, f, dhf = sysm.fock(c)
    fmo = c.T @ f @ c
    eps = np.diag(fmo)
    b, k, t, tt, _ = sysm.amplitudes(c, eps)
    x = np.einsum("iajb,Pjb->Pia", tt, b[:, :o, o:])
    n = sysm.nao
    imat = np.zeros((n, n))
    imat[:, :o] = -2.0 * np.einsum("Pia,Pqa->qi", x, b[:, :, o:])
    imat[:, o:] = -2.0 * np.einsum("Pia,Pqi->qa", x, b[:, :, :o])
    ip = imat.copy()
    ip[o:, :o] = imat[:o, o:].T
    im1 = c @ ip @ c.T
    p_oo, p_vv = mp2_densities(t)
    dm = sla.block_diag(p_oo + p_oo.T, p_vv + p_vv.T)
    fd = fmo @ dm
    zeta = 0.5 * (fd + fd.T)
    if fixed:
        zeta[:o, o:] = fd[:o, o:]
        zeta[o:, :o] = fd[:o, o:].T
    zeta_ao = c @ zeta @ c.T
    co = c[:, :o]
    if fixed:
        zeta_ao += 2.0 * co @ fmo[:o, :o] @ co.T
        dm_ao = c @ dm @ c.T
        vhf = co @ co.T @ sysm.G(2.0 * dm_ao) @ co @ co.T
        zeta_ao += vhf
    else:
        zeta_ao += 2.0 * (co * eps[:o]) @ co.T
    w = im1 - zeta_ao
    gamma = dhf + c @ dm @ c.T
    return w, gamma


def reconverged_energy(sysm, kind):
    gfun = grad_tb if kind == "tb" else grad_diag
    c0 = sysm.rhf()
    c, _, gn = solve(sysm, c0, kind, gfun, tol=1e-11)
    return energy(sysm, c, kind), c


def cmd_nuc(systems):
    print(
        "\n=== (e) nuclear-gradient ingredients (re-converged central FD, e=1e-4) ==="
    )
    print(
        "dE/de for S -> S + e R (tests the energy-weighted density w: predicted sum R*w)"
    )
    print(
        "and h -> h + e R (tests the 1-PDM gamma: predicted sum R*gamma), R random symmetric."
    )
    for sysm in systems:
        rng = np.random.default_rng(11)
        r = rng.standard_normal((sysm.nao, sysm.nao))
        r = 0.05 * (r + r.T)
        for kind in ("tb", "diag"):
            e0, c = reconverged_energy(sysm, kind)
            if kind == "tb":
                c = semicanonicalize(sysm, c)[0]
            w_old, gam = rust_like_w(sysm, c, fixed=False)
            w_new, _ = rust_like_w(sysm, c, fixed=True)
            eps_ = 1e-4
            s0, h0 = sysm.S.copy(), sysm.h.copy()
            res = {}
            for tag in ("S", "h"):
                vals = []
                for sgn in (1, -1):
                    if tag == "S":
                        sysm.S = s0 + sgn * eps_ * r
                    else:
                        sysm.h = h0 + sgn * eps_ * r
                    vals.append(reconverged_energy(sysm, kind)[0])
                    sysm.S, sysm.h = s0.copy(), h0.copy()
                res[tag] = (vals[0] - vals[1]) / (2 * eps_)
            # connection-dependence probe: FD with the converged orbitals carried
            # to the perturbed metric by Loewdin orthonormalisation (no re-solve)
            vals = []
            for sgn in (1, -1):
                sysm.S = s0 + sgn * eps_ * r
                m = c.T @ sysm.S @ c
                wv, vv = np.linalg.eigh(m)
                cl = c @ (vv / np.sqrt(wv)) @ vv.T
                vals.append(energy(sysm, cl, kind))
                sysm.S = s0.copy()
            lowdin = (vals[0] - vals[1]) / (2 * eps_)
            print(f"{sysm.name} [{kind}]:")
            print(
                f"  S: re-converged FD {res['S']:+.10f}  Loewdin-carried FD {lowdin:+.10f}  (diff {res['S'] - lowdin:+.2e})"
            )
            print(
                f"     shipped w  {np.sum(r * w_old):+.10f} (err {np.sum(r * w_old) - res['S']:+.2e})"
            )
            print(
                f"     fixed   w  {np.sum(r * w_new):+.10f} (err {np.sum(r * w_new) - res['S']:+.2e})"
            )
            print(
                f"  h: re-converged FD {res['h']:+.10f}  gamma {np.sum(r * gam):+.10f} (err {np.sum(r * gam) - res['h']:+.2e})"
            )


# ----------------------------------------------------------------------------
# (f) open shell: U-OO-RI-MP2 (u_oo_rimp2.rs)
# ----------------------------------------------------------------------------
# ferric's U gradient (u_oo_rimp2.rs) = +2 F^s_ai (add_hf_gradient) + the U-MP2
# integral response at FIXED orbital energies (u_rimp2::compute_u_mp2_orbital_gradient,
# whose own doc says "at fixed orbital energies (integral-response only)").
# There is NO orbital-energy (denominator) response at all -- the same term the
# closed-shell code lacked before 2026-07-20.
def ufock(sysm, ca, cb):
    na, nb = sysm.nalpha, sysm.nbeta
    da = ca[:, :na] @ ca[:, :na].T
    db = cb[:, :nb] @ cb[:, :nb].T
    jt, _ = sysm.jk(da + db)
    ka = sysm.jk(da)[1]
    kb = sysm.jk(db)[1]
    fa, fb = sysm.h + jt - ka, sysm.h + jt - kb
    e = 0.5 * (np.sum((sysm.h + fa) * da) + np.sum((sysm.h + fb) * db)) + sysm.enuc
    return e, fa, fb


def u_semicanon(c, f, n):
    fmo = c.T @ f @ c
    wo, uo = np.linalg.eigh(fmo[:n, :n])
    wv, uv = np.linalg.eigh(fmo[n:, n:])
    u = sla.block_diag(uo, uv)
    return c @ u, np.concatenate([wo, wv]), u


def u_mp2(sysm, ca, cb, ea, eb):
    na, nb = sysm.nalpha, sysm.nbeta
    ba = sysm.bmo(ca)[:, :na, na:]
    bb = sysm.bmo(cb)[:, :nb, nb:]

    def den(eo, ev, eo2, ev2):
        return (
            eo[:, None, None, None]
            + eo2[None, None, :, None]
            - ev[None, :, None, None]
            - ev2[None, None, None, :]
        )

    kaa = np.einsum("Pia,Pjb->iajb", ba, ba)
    kbb = np.einsum("Pia,Pjb->iajb", bb, bb)
    kab = np.einsum("Pia,Pjb->iajb", ba, bb)
    aaa = kaa - kaa.transpose(0, 3, 2, 1)
    abb = kbb - kbb.transpose(0, 3, 2, 1)
    taa = aaa / den(ea[:na], ea[na:], ea[:na], ea[na:])
    tbb = abb / den(eb[:nb], eb[nb:], eb[:nb], eb[nb:])
    tab = kab / den(ea[:na], ea[na:], eb[:nb], eb[nb:])
    e = 0.25 * np.sum(taa * aaa) + 0.25 * np.sum(tbb * abb) + np.sum(tab * kab)
    return e, taa, tbb, tab


def u_energy(sysm, ca, cb, kind):
    na, nb = sysm.nalpha, sysm.nbeta
    ehf, fa, fb = ufock(sysm, ca, cb)
    if kind == "tb":
        ca, ea, _ = u_semicanon(ca, fa, na)
        cb, eb, _ = u_semicanon(cb, fb, nb)
    else:
        ea, eb = np.diag(ca.T @ fa @ ca), np.diag(cb.T @ fb @ cb)
    return ehf + u_mp2(sysm, ca, cb, ea, eb)[0]


def u_grad(sysm, ca, cb, variant):
    """variant: "shipped" (2F + fixed-eps integral response, ferric today),
    "tb" (textbook functional: + Fock response of both spins)."""
    na, nb = sysm.nalpha, sysm.nbeta
    _, fa, fb = ufock(sysm, ca, cb)
    if variant == "tb":
        ca, ea, ua = u_semicanon(ca, fa, na)
        cb, eb, ub = u_semicanon(cb, fb, nb)
    else:
        ea, eb = np.diag(ca.T @ fa @ ca), np.diag(cb.T @ fb @ cb)
    fma, fmb = ca.T @ fa @ ca, cb.T @ fb @ cb
    _, taa, tbb, tab = u_mp2(sysm, ca, cb, ea, eb)
    Ba, Bb = sysm.bmo(ca), sysm.bmo(cb)
    ya = 2 * np.einsum("iajb,Pjb->Pia", taa, Ba[:, :na, na:]) + 2 * np.einsum(
        "iajb,Pjb->Pia", tab, Bb[:, :nb, nb:]
    )
    yb = 2 * np.einsum("iajb,Pjb->Pia", tbb, Bb[:, :nb, nb:]) + 2 * np.einsum(
        "jbia,Pjb->Pia", tab, Ba[:, :na, na:]
    )

    def gint(y, B, n):
        return np.einsum("Pia,Pca->ci", y, B[:, n:, n:]) - np.einsum(
            "Pka,Pki->ai", y, B[:, :n, :n]
        )

    ga = 2 * fma[na:, :na] + gint(ya, Ba, na)
    gb = 2 * fmb[nb:, :nb] + gint(yb, Bb, nb)
    if variant == "shipped":
        return ga, gb
    # U-MP2 unrelaxed densities (u_rimp2::build_u_mp2_density) = dE/dF^s_pq
    poa = -0.5 * np.einsum(
        "ikab,jkab->ij", taa.transpose(0, 2, 1, 3), taa.transpose(0, 2, 1, 3)
    ) - np.einsum("iakb,jakb->ij", tab, tab)
    pva = 0.5 * np.einsum(
        "ijac,ijbc->ab", taa.transpose(0, 2, 1, 3), taa.transpose(0, 2, 1, 3)
    ) + np.einsum("iajc,ibjc->ab", tab, tab)
    pob = -0.5 * np.einsum(
        "ikab,jkab->ij", tbb.transpose(0, 2, 1, 3), tbb.transpose(0, 2, 1, 3)
    ) - np.einsum("kaib,kajb->ij", tab, tab)
    pvb = 0.5 * np.einsum(
        "ijac,ijbc->ab", tbb.transpose(0, 2, 1, 3), tbb.transpose(0, 2, 1, 3)
    ) + np.einsum("icja,icjb->ab", tab, tab)
    pwa, pwb = sla.block_diag(poa, pva), sla.block_diag(pob, pvb)
    pa_ao, pb_ao = ca @ pwa @ ca.T, cb @ pwb @ cb.T
    jt = sysm.jk(pa_ao + pb_ao)[0]
    kpa, kpb = sysm.jk(pa_ao)[1], sysm.jk(pb_ao)[1]
    ga = (
        ga
        + 2 * (ca.T @ (jt - kpa) @ ca)[na:, :na]
        + 2 * (fma[na:, :] @ pwa[:, :na] - pwa[na:, :] @ fma[:, :na])
    )
    gb = (
        gb
        + 2 * (cb.T @ (jt - kpb) @ cb)[nb:, :nb]
        + 2 * (fmb[nb:, :] @ pwb[:, :nb] - pwb[nb:, :] @ fmb[:, :nb])
    )
    return ua[na:, na:] @ ga @ ua[:na, :na].T, ub[nb:, nb:] @ gb @ ub[:nb, :nb].T


def u_fd(sysm, ca, cb, kind, h=1e-3):
    n = sysm.nao
    out = []
    for spin, nocc in ((0, sysm.nalpha), (1, sysm.nbeta)):
        g = np.zeros((n - nocc, nocc))
        for a in range(nocc, n):
            for i in range(nocc):
                vals = []
                for s_ in (2, 1, -1, -2):
                    k = np.zeros((n, n))
                    k[a, i], k[i, a] = s_ * h, -s_ * h
                    u = cayley(k)
                    vals.append(
                        u_energy(
                            sysm,
                            ca @ u if spin == 0 else ca,
                            cb @ u if spin == 1 else cb,
                            kind,
                        )
                    )
                g[a - nocc, i] = (-vals[0] + 8 * vals[1] - 8 * vals[2] + vals[3]) / (
                    12 * h
                )
        out.append(g)
    return out


def u_solve(sysm, ca, cb, kind, variant, tol=1e-10, maxit=2000, shift=0.1):
    na, nb = sysm.nalpha, sysm.nbeta
    for it in range(maxit):
        ga, gb = u_grad(sysm, ca, cb, variant)
        gn = np.sqrt(np.sum(ga**2) + np.sum(gb**2))
        if gn < tol:
            break
        _, fa, fb = ufock(sysm, ca, cb)
        for c, g, f, n, sp in ((ca, ga, fa, na, 0), (cb, gb, fb, nb, 1)):
            e = np.diag(c.T @ f @ c)
            kov = -g / (2.0 * (e[n:, None] - e[None, :n]) + shift)
            m = np.abs(kov).max()
            if m > 0.3:
                kov *= 0.3 / m
            cn = c @ cayley(kappa_from_ov(kov, n))
            if sp == 0:
                ca_new = cn
            else:
                cb_new = cn
        ca, cb = ca_new, cb_new
    return ca, cb, it, gn


def cmd_open(names):
    print(
        "\n=== (f) open shell U-OO-RI-MP2: analytic vs 4-point FD, and the converged point ==="
    )
    for name, basis in names:
        sysm = System(name, basis, spin=1)
        mf = scf.UHF(sysm.mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        ca0, cb0 = mf.mo_coeff
        for scale in (0.0, 0.05, 0.2):
            if scale:
                ca = ca0 @ cayley(random_kappa(sysm.nao, scale, 3))
                cb = cb0 @ cayley(random_kappa(sysm.nao, scale, 5))
            else:
                ca, cb = ca0, cb0
            fdd = u_fd(sysm, ca, cb, "diag")
            fdt = u_fd(sysm, ca, cb, "tb")
            gs = u_grad(sysm, ca, cb, "shipped")
            gt = u_grad(sysm, ca, cb, "tb")
            es = max(np.abs(gs[0] - fdd[0]).max(), np.abs(gs[1] - fdd[1]).max())
            et = max(np.abs(gt[0] - fdt[0]).max(), np.abs(gt[1] - fdt[1]).max())
            print(
                f"  {sysm.name:12s} |k|={scale:4.2f}: shipped-vs-FD(diag) {es:9.2e}   tb-vs-FD(tb) {et:9.2e}"
            )
        ca, cb, it, gn = u_solve(sysm, ca0, cb0, "tb", "tb")
        fdt = u_fd(sysm, ca, cb, "tb")
        e_tb = u_energy(sysm, ca, cb, "tb")
        cas, cbs, it2, gn2 = u_solve(sysm, ca0, cb0, "diag", "shipped", tol=1e-8)
        fds = u_fd(sysm, cas, cbs, "diag")
        print(
            f"  {sysm.name:12s} converged tb:      |g|an {gn:.1e}  max|g_FD| {max(np.abs(fdt[0]).max(), np.abs(fdt[1]).max()):.2e}  E {e_tb:.12f}"
        )
        print(
            f"  {sysm.name:12s} converged shipped: |g|an {gn2:.1e}  max|g_FD| {max(np.abs(fds[0]).max(), np.abs(fds[1]).max()):.2e}  E {u_energy(sysm, cas, cbs, 'diag'):.12f}"
        )


def cmd_psi4():
    """Anchor the prototype itself: textbook OMP2 with EXACT integrals must
    reproduce Psi4 tests/omp2-1 (conventional OMP2, cc-pVDZ):
    refomp2 = -76.23167598916250 (refscf -76.02676109559437)."""
    print("\n=== prototype anchor: conventional textbook OMP2 vs Psi4 omp2-1 ===")
    for aux in ("exact", "cc-pvdz-ri"):
        sysm = System("h2o_psi4", "cc-pvdz", auxbasis=aux)
        c0 = sysm.rhf()
        c, it, gn = solve(sysm, c0, "tb", grad_tb, tol=1e-9)
        e = energy(sysm, c, "tb")
        cd, _, _ = solve(sysm, c0, "diag", grad_diag, tol=1e-9)
        ed = energy(sysm, cd, "diag")
        print(
            f"  {sysm.name:28s} E_HF(RHF) {sysm.fock(c0)[0]:.10f}  E_tb {e:.10f} (vs Psi4 {e + 76.23167598916250:+.2e})"
            f"   E_diag {ed:.10f} (vs Psi4 {ed + 76.23167598916250:+.2e})"
        )


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    small = [System("h2", "cc-pvdz"), System("h2o", "sto-3g"), System("nh3", "6-31g")]
    if which in ("grad", "all"):
        cmd_grad(small)
    if which in ("solve", "all"):
        cmd_solve(small)
    if which in ("gap", "all"):
        cmd_gap([System("h2o", "cc-pvdz"), System("nh3", "cc-pvdz")])
    if which in ("nuc", "all"):
        cmd_nuc([System("h2", "cc-pvdz"), System("h2o", "sto-3g")])
    if which in ("psi4", "all"):
        cmd_psi4()
    if which in ("open", "all"):
        cmd_open([("oh", "sto-3g"), ("oh", "6-31g"), ("nh2", "6-31g")])


if __name__ == "__main__":
    main()
