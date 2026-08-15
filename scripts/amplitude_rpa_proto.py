#!/usr/bin/env python
"""Amplitude-space single-threshold direct-RPA (drCCD Riccati) prototype.

Dense correctness-reference prototype of an amplitude-threshold dRPA, built
"the same way" as scripts/amplitude_lmp2_proto.py (which it imports for the
shared VV-HV localized-virtual machinery, Boys occupieds, frozen core,
domain stats): a single threshold eps gating INTEGRAL magnitudes
(keep (i,a,j,b) iff |B_iajb|>eps; B is (ia)<->(jb)-symmetric so the mask is
automatically closed under that swap — there is NO exchange term, so the
MP2 rig's a<->b "J or K" closure has no analogue here), fixed sparsity
pattern, masked damped fixed-point Riccati solve. NO sparse maps / local DF
here: B is exact dense. Nothing about speed or scaling may be quoted.

Formulation (closed-shell spin-adapted ring-CCD form of dRPA, derived from
the spin-orbital drCCD 0 = B~ + A~T~ + T~A~ + T~B~T~, E = 1/2 Tr[B~T~] with
A~ = Fock + K~, B~_iajb = (ia|jb) direct; mixed-spin blocks vanish, the
same-spin and opposite-spin blocks obey identical equations so T_ss = T_os,
and in T = T_ss + T_os the spatial-orbital equations close):

    R(T) = B + F(T) + B.T + T.B + T.B.T = 0
    B_iajb = 2 (ia|jb)          (chemist notation, localized or canonical)
    F(T)   = Fvv T + T Fvv - Foo T - T Foo      (the MP2 rig's Aop, pos.def.)
    E_c    = 1/2 Tr[B T] = 1/2 sum_iajb B_iajb T_iajb   (T is symmetric)

MP2 limit: T ~ -B/D gives E ~ -2 sum (ia|jb)^2/D = direct-MP2 (no exchange).
This matches PySCF's closed-shell dRPA convention A = e_ov + 2K, B = 2K,
E_c = 1/2(sum omega - Tr A) (pyscf/gw/rpa.py self-test) — the plasmon
formula on the SAME integrals is independent-algorithm cross-check #1, and
PySCF's adiabatic-connection frequency integration (pyscf.gw.rpa.RPA, DF
integrals) is the truly independent construction #2.

Design note: wiki/amplitude-threshold-drpa.md (anchor + artifact hypothesis
registered there BEFORE any sweep).

Usage (always under a memory cap):
  scripts/ferric-limited --max=2G --high=1800M -- nice -n 10 \
    env OPENBLAS_NUM_THREADS=2 uv run --no-sync python \
    scripts/amplitude_rpa_proto.py --xyz testdata/molecules/water.xyz \
    --basis 6-31g [--anchor-only] [--mutate] [--mutate-riccati] \
    [--eps 1e-3,1e-4,1e-5] [--omega 1.0]
"""
import argparse
import os
import sys
import time

import numpy as np
from pyscf import gto, scf, lo, ao2mo, df
from pyscf.data import elements
from pyscf.gw import rpa as pyscf_rpa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import amplitude_lmp2_proto as lmp2  # shared machinery; NOT modified

log = lmp2.log

# Anchor bars (registered in wiki/amplitude-threshold-drpa.md Sec 4):
BAR_RICCATI = 1e-9   # localized eps=0 Riccati vs canonical Riccati
BAR_PLASMON = 1e-9   # canonical Riccati vs plasmon formula (same integrals)
BAR_AC = 5e-5        # canonical Riccati vs PySCF AC dRPA (DF + quadrature)


def fock_superop(T, Foo, Fvv):
    """F(T) = Fvv T + T Fvv - Foo T - T Foo (the MP2 rig's Aop, unmasked)."""
    r = np.einsum("ac,icjb->iajb", Fvv, T, optimize=True)
    r += np.einsum("iajc,cb->iajb", T, Fvv, optimize=True)
    r -= np.einsum("ik,kajb->iajb", Foo, T, optimize=True)
    r -= np.einsum("iakb,kj->iajb", T, Foo, optimize=True)
    return r


def solve_riccati_masked(B, Foo, Fvv, mask, damp=1.0, rtol=1e-12,
                         maxiter=3000):
    """Masked drCCD Riccati by damped fixed-point iteration.

    T <- T - damp * R(T)/D, with the mask applied to B once and to T/R
    every iteration; D is the MP2 rig's positive denominator tensor.
    Returns (T, niter, relres, diverged).
    """
    fo, fv = np.diag(Foo).copy(), np.diag(Fvv).copy()
    D = (fv[None, :, None, None] + fv[None, None, None, :]
         - fo[:, None, None, None] - fo[None, None, :, None])
    assert D.min() > 0, "non-positive denominator: not a gapped system?"
    no, nv = B.shape[0], B.shape[1]
    nov = no * nv
    Bm = np.where(mask, B, 0.0)
    Bmat = Bm.reshape(nov, nov)
    bnorm = np.linalg.norm(Bm)
    if bnorm == 0.0:
        return np.zeros_like(B), 0, 0.0, False
    T = -Bm / D                                    # masked MP2-like start
    best = np.inf
    relres = np.inf
    for it in range(1, maxiter + 1):
        Tmat = T.reshape(nov, nov)
        BT = Bmat @ Tmat
        TB = Tmat @ Bmat
        TBT = Tmat @ BT
        R = Bm + fock_superop(T, Foo, Fvv) + (BT + TB + TBT).reshape(B.shape)
        np.multiply(R, mask, out=R)
        relres = np.linalg.norm(R) / bnorm
        if relres < rtol:
            return T, it, relres, False
        if relres > 1e3 * max(best, 1e-30) or not np.isfinite(relres):
            log(f"  WARNING: Riccati DIVERGED at it={it}, relres={relres:.2e}")
            return T, it, relres, True
        best = min(best, relres)
        T -= damp * (R / D)
    log(f"  WARNING: Riccati hit maxiter={maxiter}, relres={relres:.2e}")
    return T, maxiter, relres, False


def drpa_energy(T, B, mask):
    """E_c = 1/2 Tr[B T] restricted to the mask (T lives on the mask)."""
    return 0.5 * np.vdot(np.where(mask, B, 0.0), T)


def drpa_plasmon(K, e_ov):
    """Independent-algorithm dRPA on the SAME integrals: plasmon formula.

    A = diag(e_ov) + 2K, B = 2K (closed-shell singlet channel; triplet
    contributes zero in dRPA). A-B = diag(e_ov) > 0, so
    C = (A-B)^1/2 (A+B) (A-B)^1/2 is symmetric; omega = sqrt(eig(C));
    E_c = 1/2 (sum omega - Tr A). Eigensolve vs fixed-point: no shared
    solver code with the Riccati path.
    """
    sq = np.sqrt(e_ov)
    ApB = np.diag(e_ov) + 4.0 * K
    C = sq[:, None] * ApB * sq[None, :]
    w2 = np.linalg.eigvalsh(C)
    assert w2.min() > 0, "RPA instability (negative omega^2)"
    return 0.5 * (np.sqrt(w2).sum() - (e_ov.sum() + 2.0 * np.trace(K)))


def canonical_drpa(mol, mf, ncore, omega, damp, mutate_riccati=False):
    """Canonical-orbital references: exact-integral Riccati (same code path
    as the localized solve, canonical inputs) + plasmon cross-check.

    IMPORTANT (measured on H2/6-31G): dRPA is NON-variational, so it is
    FIRST-order sensitive to Fock inconsistencies — at scf conv_tol=1e-10,
    mf.mo_energy and diag(C^T get_fock() C) differ by ~4e-7 and that alone
    shifted E_c by 1.667e-9, failing the 1e-9 localized-vs-canonical bar
    (the MP2 rig never saw this: Hylleraas is quadratic in such errors).
    Both references therefore use the SAME recomputed Fock the localized
    path sees: the Riccati takes the full C^T F C blocks; the plasmon
    formula takes their exact semicanonicalization (eigenbasis of the
    occ/vir blocks, K rotated along).

    Returns (e_riccati, e_plasmon, niter). mutate_riccati corrupts B inside
    the Riccati path ONLY (plasmon untouched) — the plasmon cross-check and
    the AC anchor must then FAIL (non-variational => first-order
    sensitivity: a 1e-3 element perturbation is ample).
    """
    nocc = np.count_nonzero(mf.mo_occ > 0)
    Co, Cv = mf.mo_coeff[:, ncore:nocc], mf.mo_coeff[:, nocc:]
    no, nv = Co.shape[1], Cv.shape[1]
    F = mf.get_fock()
    Foo, Fvv = Co.T @ F @ Co, Cv.T @ F @ Cv
    if omega is not None:
        with mol.with_range_coulomb(-omega):
            K = ao2mo.general(mol, (Co, Cv, Co, Cv), compact=False)
    else:
        K = ao2mo.general(mol, (Co, Cv, Co, Cv), compact=False)
    K = K.reshape(no, nv, no, nv)
    # exact semicanonicalization for the plasmon path
    wo, Uo = np.linalg.eigh(Foo)
    wv, Uv = np.linalg.eigh(Fvv)
    Ksc = np.einsum("iajb,ik->kajb", K, Uo, optimize=True)
    Ksc = np.einsum("kajb,ac->kcjb", Ksc, Uv, optimize=True)
    Ksc = np.einsum("kcjb,jl->kclb", Ksc, Uo, optimize=True)
    Ksc = np.einsum("kclb,bd->kcld", Ksc, Uv, optimize=True)
    e_ov = (wv[None, :] - wo[:, None]).reshape(no * nv)
    e_pl = drpa_plasmon(Ksc.reshape(no * nv, no * nv), e_ov)
    del Ksc
    B = 2.0 * K
    if mutate_riccati:
        log("  MUTATION(riccati): B[0,0,0,0] += 1e-3 in the Riccati path "
            "only; plasmon/AC cross-checks must FAIL")
        B = B.copy()
        B[0, 0, 0, 0] += 1e-3
    mask = np.ones(B.shape, dtype=bool)
    T, niter, relres, div = solve_riccati_masked(B, Foo, Fvv, mask, damp=damp)
    if div:
        raise SystemExit("canonical Riccati diverged")
    return drpa_energy(T, B, mask), e_pl, niter


def pyscf_ac_drpa(mol, mf, ncore, nw=100):
    """Independent construction #2: PySCF adiabatic-connection dRPA
    (pyscf.gw.rpa.RPA — DF integrals + imaginary-frequency quadrature).

    Aux quality is the anchor's binding confound: measured on H2/6-31G the
    AC energy matches the plasmon formula ON THE SAME DF INTEGRALS to
    ~1e-12 at nw=100 (quadrature exact at this scale), so the whole
    Riccati-vs-AC gap is DF error. aug-cc-pV5Z-RI pushes it to ~3e-8 on H2
    (aug_etb beta=1.7 left 1.9e-4, which FAILED the 5e-5 bar)."""
    mf2 = mf.density_fit(auxbasis="aug-cc-pv5z-ri")
    r = pyscf_rpa.RPA(mf2, frozen=ncore if ncore > 0 else None)
    r.verbose = 0
    return r.kernel(nw=nw)


def run(xyz, basis, eps_list, anchor_only=False, mutate=False,
        mutate_riccati=False, out=None, omega=None, damp=1.0, nw=100):
    t0 = time.time()
    atom = lmp2.load_xyz(xyz)
    mol = gto.M(atom=atom, basis=basis, verbose=0, max_memory=300)
    ncore = elements.chemcore(mol)
    log(f"== {xyz} basis={basis} nao={mol.nao} ncore={ncore}")
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged, "SCF not converged"
    log(f"  E(RHF) = {mf.e_tot:.10f}  ({time.time()-t0:.1f}s)")

    # ---- canonical references (independent constructions) ----
    e_can, e_pl, it_can = canonical_drpa(mol, mf, ncore, omega, damp,
                                         mutate_riccati=mutate_riccati)
    wtag = "coulomb" if omega is None else f"w={omega:g}"
    log(f"  E_corr(canonical Riccati, {wtag}) = {e_can:.10f}  "
        f"(fp iters={it_can})")
    log(f"  E_corr(plasmon formula,  {wtag}) = {e_pl:.10f}  "
        f"|d|={abs(e_can-e_pl):.3e}")
    pl_ok = abs(e_can - e_pl) <= BAR_PLASMON
    if mutate_riccati:
        verdict = ("MUTATION-OK (plasmon xcheck FAILED as required)"
                   if not pl_ok else
                   "MUTATION-BROKEN: plasmon xcheck still passes!")
        log(f"  {verdict}  |d|={abs(e_can-e_pl):.3e}")
        return
    if not pl_ok:
        raise SystemExit("plasmon/Riccati cross-check FAILED; no sweep run")
    log(f"  XCHECK Riccati-vs-plasmon PASSED (bar {BAR_PLASMON:g})")

    if omega is None:
        e_ac = pyscf_ac_drpa(mol, mf, ncore, nw=nw)
        gap = e_can - e_ac
        log(f"  E_corr(PySCF AC dRPA, nw={nw}, aug-cc-pV5Z-RI DF) = "
            f"{e_ac:.10f}  "
            f"gap={gap:+.3e} (DF+quadrature)")
        if abs(gap) > BAR_AC:
            raise SystemExit(f"AC anchor FAILED: |gap|={abs(gap):.3e} > "
                             f"{BAR_AC:g}; no sweep run")
        log(f"  ANCHOR Riccati-vs-AC PASSED (bar {BAR_AC:g})")
    else:
        # PySCF's AC-RPA has no SR kernel: the SR reference is the canonical
        # SR Riccati itself (+ plasmon xcheck above, both at SR K). Guard
        # against a silently no-op'd with_range_coulomb: SR correlation
        # must be strictly smaller than full Coulomb.
        e_full, e_pl_full, _ = canonical_drpa(mol, mf, ncore, None, damp)
        log(f"  E_corr(canonical Riccati, coulomb) = {e_full:.10f} (SR guard)")
        if not abs(e_can) < abs(e_full) - 1e-10:
            raise SystemExit("SR guard FAILED: |E_sr| >= |E_coulomb| — "
                             "with_range_coulomb no-op or sign error")

    # ---- localized basis (same machinery as the MP2 rig) ----
    nocc_tot = np.count_nonzero(mf.mo_occ > 0)
    C_occ_all = mf.mo_coeff[:, :nocc_tot]
    C_act = mf.mo_coeff[:, ncore:nocc_tot]
    C_act = lo.Boys(mol, C_act).kernel()
    C_vloc, n_l, n_h = lmp2.build_vvhv(mol, mf, C_occ_all)
    log(f"  VV-HV: n_valence_virt={n_l} n_hard_virt={n_h} "
        f"nocc_act={C_act.shape[1]}")
    if mutate:
        log("  MUTATION: dropping one hard virtual (span check bypassed)")
        C_vloc = C_vloc[:, :-1]
    else:
        lmp2.check_construction(mol, mf, C_act, C_vloc)

    no, nv = C_act.shape[1], C_vloc.shape[1]
    F = mf.get_fock()
    Foo = C_act.T @ F @ C_act
    Fvv = C_vloc.T @ F @ C_vloc
    log(f"  transforming (ia|jb): no={no} nv={nv} "
        f"tensor {8*(no*nv)**2/1e6:.0f} MB")
    if omega is not None:
        with mol.with_range_coulomb(-omega):
            K = ao2mo.general(mol, (C_act, C_vloc, C_act, C_vloc),
                              compact=False)
    else:
        K = ao2mo.general(mol, (C_act, C_vloc, C_act, C_vloc), compact=False)
    B = 2.0 * K.reshape(no, nv, no, nv)
    del K
    log(f"  integrals done ({time.time()-t0:.1f}s)")

    e_loc0 = None
    eps_run = [0.0] + ([] if anchor_only or mutate else eps_list)
    for eps in eps_run:
        if eps == 0.0:
            mask = np.ones(B.shape, dtype=bool)
        else:
            mask = np.abs(B) > eps        # (ia)<->(jb)-symmetric already
        st = lmp2.domain_stats(mask)
        t1 = time.time()
        T, niter, relres, div = solve_riccati_masked(B, Foo, Fvv, mask,
                                                     damp=damp)
        e = drpa_energy(T, B, mask)
        de_can = e - e_can
        de_loc = e - e_loc0 if e_loc0 is not None else 0.0
        tag = "ANCHOR" if eps == 0.0 else f"{eps:g}"
        row = (f"{wtag:>8s} {tag:>8s}  E_corr={e:.10f}  dE_can={de_can:+.3e} "
               f"dE_loc0={de_loc:+.3e}  keep={st['frac']:.4f} "
               f"pairs={st['pair_frac']:.3f} "
               f"dom(mean/max)={st['dom_mean']:.1f}/{st['dom_max']} of {nv}  "
               f"fp={niter} relres={relres:.1e}"
               f"{' DIVERGED' if div else ''} ({time.time()-t1:.1f}s)")
        log("  " + row)
        if out:
            with open(out, "a") as f:
                f.write(f"{xyz} {basis} {row}\n")
        if eps == 0.0:
            e_loc0 = e
            ok = abs(de_can) < BAR_RICCATI and not div
            if mutate:
                verdict = ("MUTATION-OK (anchor FAILED as required)"
                           if abs(de_can) > 1e-6 else
                           "MUTATION-BROKEN: anchor still passes!")
                log(f"  {verdict}  |dE|={abs(de_can):.3e}")
                return
            log(f"  ANCHOR localized-vs-canonical "
                f"{'PASSED' if ok else 'FAILED'} |dE|={abs(de_can):.3e} "
                f"(bar {BAR_RICCATI:g})")
            if not ok:
                raise SystemExit("exactness anchor failed; no sweep run")
            if anchor_only:
                return
    log(f"  total {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz", required=True)
    ap.add_argument("--basis", default="6-31g")
    ap.add_argument("--eps", default="1e-3,1e-4,1e-5")
    ap.add_argument("--anchor-only", action="store_true")
    ap.add_argument("--mutate", action="store_true",
                    help="drop one hard virtual; anchor must FAIL")
    ap.add_argument("--mutate-riccati", action="store_true",
                    help="corrupt B inside the canonical Riccati path only; "
                         "the plasmon cross-check must then FAIL")
    ap.add_argument("--out", default=None)
    ap.add_argument("--omega", type=float, default=None,
                    help="SR erfc attenuation of B, Bohr^-1 (sparsity-"
                         "composition probe only — the physical role of RPA "
                         "in this repo's compositions is the LR erf channel)")
    ap.add_argument("--damp", type=float, default=1.0,
                    help="fixed-point damping factor")
    ap.add_argument("--nw", type=int, default=100,
                    help="frequency grid for the PySCF AC reference")
    a = ap.parse_args()
    eps_list = [float(x) for x in a.eps.split(",") if x]
    run(a.xyz, a.basis, eps_list, a.anchor_only, a.mutate, a.mutate_riccati,
        a.out, omega=a.omega, damp=a.damp, nw=a.nw)


if __name__ == "__main__":
    main()
