#!/usr/bin/env python
"""Amplitude-space single-threshold local MP2 prototype (Wang/Shen/Head-Gordon).

Dense correctness-reference prototype of the [WSHG23] method (JCTC 19, 7577
(2023), DOI 10.1021/acs.jctc.3c00744): Boys occupieds + VV-HV orthogonal
localized virtuals, a single threshold eps gating INTEGRAL magnitudes
(keep (i,a,j,b) iff |(ia|jb)|>eps or |(ib|ja)|>eps, Eq 8), fixed sparsity
pattern, masked preconditioned-CG amplitude solve, Hylleraas-variational
energy. NO sparse maps / local DF here: J is exact dense. Nothing about
speed or scaling may be quoted from this prototype.

Design note: wiki/amplitude-threshold-lmp2.md (anchor + artifact hypothesis
defined there BEFORE any sweep).

Usage (always under a memory cap, 1 CPU):
  scripts/ferric-limited --max=1500M --high=1200M -- \
    taskset -c 1 nice -n 10 env OPENBLAS_NUM_THREADS=1 \
    uv run --no-sync python scripts/amplitude_lmp2_proto.py \
      --xyz testdata/molecules/water.xyz --basis 6-31g \
      [--anchor-only] [--mutate] [--eps 1e-4,1e-5,...]
"""
import argparse
import sys
import time

import numpy as np
from pyscf import gto, scf, mp, lo, ao2mo
from pyscf.data import elements

LINDEP = 1e-8


def log(*a):
    print(*a, flush=True)


def load_xyz(path):
    with open(path) as f:
        lines = f.read().strip().splitlines()
    n = int(lines[0].split()[0])
    return "\n".join(lines[2:2 + n])


def pivoted_cholesky_order(M, rank):
    """Greedy pivoted Cholesky on PSD matrix M; return `rank` pivot indices."""
    d = np.diag(M).copy().astype(float)
    n = len(d)
    L = np.zeros((rank, n))
    piv = []
    for k in range(rank):
        j = int(np.argmax(d))
        if d[j] <= 0:
            raise RuntimeError(f"pivoted Cholesky broke down at k={k}")
        piv.append(j)
        L[k] = (M[j] - L[:k].T @ L[:k, j]) / np.sqrt(d[j])
        d -= L[k] ** 2
        d[piv] = -np.inf  # never repick
    return piv


def lowdin(C, S):
    """Symmetric (Loewdin) orthonormalization of columns of C w.r.t. metric S."""
    o = C.T @ S @ C
    w, v = np.linalg.eigh(o)
    if w.min() < 1e-10:
        raise RuntimeError(f"Loewdin set near-singular: min eig {w.min():.2e}")
    return C @ (v * (1.0 / np.sqrt(w))) @ v.T


def canonical_orth(C, S, rank):
    """Canonical orthonormalization keeping the `rank` largest-eig directions."""
    o = C.T @ S @ C
    w, v = np.linalg.eigh(o)
    idx = np.argsort(w)[::-1][:rank]
    if w[idx].min() < LINDEP:
        raise RuntimeError(f"canonical orth: rank {rank} unreachable "
                           f"(min kept eig {w[idx].min():.2e})")
    return C @ (v[:, idx] * (1.0 / np.sqrt(w[idx])))


def build_vvhv(mol, mf, C_occ_all):
    """VV-HV orthogonal localized virtuals ([WSHG23] Sec 2.1).

    Deviation from the paper (documented in the wiki note Sec 4): hard
    virtuals via spread-weighted pivoted-Cholesky SELECTION + Loewdin,
    instead of weighted symmetric orthogonalization of the full redundant
    set. Atom-wise pseudo-canonicalization as in the paper.
    Returns (C_vloc, n_valence_virt, n_hard_virt).
    """
    S = mol.intor("int1e_ovlp")
    nao = S.shape[0]
    no = C_occ_all.shape[1]

    # --- valence virtuals L: projected STO-3G minus occupied span ---
    mol_min = gto.M(atom=mol.atom, basis="sto-3g", unit=mol.unit,
                    charge=mol.charge, spin=mol.spin)
    S_x = gto.intor_cross("int1e_ovlp", mol, mol_min)          # (nao, nmin)
    T = np.linalg.solve(S, S_x)                                # projected minimal
    Q = np.eye(nao) - C_occ_all @ (C_occ_all.T @ S)            # 1 - |occ><occ|S
    Tv = Q @ T
    n_l = mol_min.nao - no
    if n_l > 0:
        C_L = canonical_orth(Tv, S, n_l)
        C_L = lo.Boys(mol, C_L).kernel()                       # localize VVs
    else:
        C_L = np.zeros((nao, 0))
    # --- hard virtuals H: project E = occ + L out of each AO ---
    n_h = nao - no - n_l
    if n_h > 0:
        C_E = np.hstack([C_occ_all, C_L])
        QE = np.eye(nao) - C_E @ (C_E.T @ S)
        X = QE.copy()                                          # candidate per AO (columns)
        nrm2 = np.einsum("mi,mn,ni->i", X, S, X)
        keepable = nrm2 > 1e-8
        # spatial spreads of normalized candidates
        r2 = mol.intor("int1e_r2")
        rints = mol.intor("int1e_r")                           # (3, nao, nao)
        Xn = X[:, keepable] / np.sqrt(nrm2[keepable])
        parents = np.nonzero(keepable)[0]
        r2v = np.einsum("mi,mn,ni->i", Xn, r2, Xn)
        rv = np.einsum("mi,xmn,ni->xi", Xn, rints, Xn)
        spread = r2v - np.einsum("xi,xi->i", rv, rv)
        spread = np.maximum(spread, 1e-6)
        w = 1.0 / spread                                       # inverse-spread weights
        # weighted pivoted-Cholesky selection of n_h well-conditioned, compact fns
        ov = Xn.T @ S @ Xn
        piv = pivoted_cholesky_order(ov * np.outer(w, w) / np.outer(w, w).max(), n_h)
        sel = np.array(piv)
        C_H = lowdin(Xn[:, sel], S)
        # atom-wise pseudo-canonicalization (block-diagonal in parent atom)
        F = mf.get_fock()
        ao2atom = np.array([lbl[0] for lbl in mol.ao_labels(fmt=None)])
        hv_atom = ao2atom[parents[sel]]
        Fh = C_H.T @ F @ C_H
        C_H = C_H.copy()
        for A in np.unique(hv_atom):
            idx = np.nonzero(hv_atom == A)[0]
            _, u = np.linalg.eigh(Fh[np.ix_(idx, idx)])
            C_H[:, idx] = C_H[:, idx] @ u
    else:
        C_H = np.zeros((nao, 0))
    C_vloc = np.hstack([C_L, C_H])
    return C_vloc, n_l, n_h


def check_construction(mol, mf, C_occ_act, C_vloc):
    """Span + orthonormality asserts (part of the exactness anchor)."""
    S = mol.intor("int1e_ovlp")
    o = C_vloc.T @ S @ C_vloc
    dev_orth = np.abs(o - np.eye(o.shape[0])).max()
    nocc_tot = np.count_nonzero(mf.mo_occ > 0)
    C_vcan = mf.mo_coeff[:, nocc_tot:]
    U = C_vcan.T @ S @ C_vloc
    dev_span = max(np.abs(U @ U.T - np.eye(U.shape[0])).max(),
                   np.abs(U.T @ U - np.eye(U.shape[1])).max())
    log(f"  construction check: orthonormality dev {dev_orth:.2e}, "
        f"span dev {dev_span:.2e}")
    if dev_orth > 1e-8 or dev_span > 1e-8:
        raise RuntimeError("VV-HV construction check FAILED")


def solve_masked_mp2(J, Foo, Fvv, mask, rtol=1e-11, maxiter=400):
    """Fixed-sparsity non-canonical MP2: solve P A P t = -P J by precond. CG.

    A(t)_iajb = [Fvv t]_iajb + [t Fvv]_iajb - [Foo t]_iajb - [t Foo]_iajb
    (positive definite for a gapped system). Returns (t, niter, relres).
    """
    fo = np.diag(Foo).copy()
    fv = np.diag(Fvv).copy()
    D = (fv[None, :, None, None] + fv[None, None, None, :]
         - fo[:, None, None, None] - fo[None, None, :, None])
    assert D.min() > 0, "non-positive denominator: not a gapped system?"

    def Aop(t):
        r = np.einsum("ac,icjb->iajb", Fvv, t, optimize=True)
        r += np.einsum("iajc,cb->iajb", t, Fvv, optimize=True)
        r -= np.einsum("ik,kajb->iajb", Foo, t, optimize=True)
        r -= np.einsum("iakb,kj->iajb", t, Foo, optimize=True)
        np.multiply(r, mask, out=r)
        return r

    r = np.where(mask, -J, 0.0)
    bnorm = np.linalg.norm(r)
    if bnorm == 0.0:
        return np.zeros_like(J), 0, 0.0
    t = np.zeros_like(J)
    z = r / D
    p = z.copy()
    rz = np.vdot(r, z)
    for it in range(1, maxiter + 1):
        Ap = Aop(p)
        alpha = rz / np.vdot(p, Ap)
        t += alpha * p
        r -= alpha * Ap
        relres = np.linalg.norm(r) / bnorm
        if relres < rtol:
            return t, it, relres
        np.divide(r, D, out=Ap)  # reuse buffer as z
        z = Ap
        rz_new = np.vdot(r, z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    log(f"  WARNING: CG hit maxiter={maxiter}, relres={relres:.2e}")
    return t, maxiter, relres


def mp2_energy(t, J):
    """E = sum (2 t_iajb - t_ibja) J_iajb (spin-adapted closed shell)."""
    return 2.0 * np.vdot(t, J) - np.vdot(t.transpose(0, 3, 2, 1), J)


def domain_stats(mask):
    """Retention + per-LMO domain sizes from the boolean mask."""
    frac = mask.mean()
    pair_any = mask.any(axis=(1, 3))                 # (no, no)
    pair_frac = pair_any.mean()
    dom = mask.any(axis=(2, 3)).sum(axis=1)          # |{a: any (i,a,j,b) kept}|
    return dict(frac=frac, pair_frac=pair_frac,
                dom_mean=float(dom.mean()), dom_max=int(dom.max()),
                dom_min=int(dom.min()))


def run(xyz, basis, eps_list, anchor_only=False, mutate=False, out=None):
    t0 = time.time()
    atom = load_xyz(xyz)
    # NOTE: max_memory is PySCF's WORKING budget on top of already-resident
    # arrays; inside a 1500M cgroup, 900 throttle-stalled both C12 and C10
    # (worker RSS pinned at the cap, ~0 CPU ticks). Keep it small.
    mol = gto.M(atom=atom, basis=basis, verbose=0, max_memory=300)
    ncore = elements.chemcore(mol)
    log(f"== {xyz} basis={basis} nao={mol.nao} ncore={ncore}")
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged, "SCF not converged"
    log(f"  E(RHF) = {mf.e_tot:.10f}  ({time.time()-t0:.1f}s)")

    pt = mp.MP2(mf)
    pt.frozen = ncore if ncore > 0 else None
    e_ref = pt.kernel()[0]
    log(f"  E_corr(canonical MP2, frozen={ncore}) = {e_ref:.10f}")

    nocc_tot = np.count_nonzero(mf.mo_occ > 0)
    C_occ_all = mf.mo_coeff[:, :nocc_tot]
    C_core = mf.mo_coeff[:, :ncore]
    C_act = mf.mo_coeff[:, ncore:nocc_tot]
    # Boys-localize active occupieds (unitary within the active block)
    C_act = lo.Boys(mol, C_act).kernel()
    C_vloc, n_l, n_h = build_vvhv(mol, mf, C_occ_all)
    log(f"  VV-HV: n_valence_virt={n_l} n_hard_virt={n_h} "
        f"nocc_act={C_act.shape[1]}")

    if mutate:
        log("  MUTATION: dropping one hard virtual (span check bypassed)")
        C_vloc = C_vloc[:, :-1]
    else:
        check_construction(mol, mf, C_act, C_vloc)

    no, nv = C_act.shape[1], C_vloc.shape[1]
    F = mf.get_fock()
    Foo = C_act.T @ F @ C_act
    Fvv = C_vloc.T @ F @ C_vloc
    log(f"  transforming (ia|jb): no={no} nv={nv} "
        f"tensor {8*(no*nv)**2/1e6:.0f} MB")
    J = ao2mo.general(mol, (C_act, C_vloc, C_act, C_vloc), compact=False)
    J = J.reshape(no, nv, no, nv)
    log(f"  integrals done ({time.time()-t0:.1f}s)")

    rows = []
    eps_run = [0.0] + ([] if anchor_only or mutate else eps_list)
    for eps in eps_run:
        if eps == 0.0:
            mask = np.ones(J.shape, dtype=bool)
        else:
            K = J.transpose(0, 3, 2, 1)              # K_iajb = (ib|ja), view
            mask = (np.abs(J) > eps) | (np.abs(K) > eps)   # Eq 8, swap-closed
        st = domain_stats(mask)
        t1 = time.time()
        t, niter, relres = solve_masked_mp2(J, Foo, Fvv, mask)
        e = mp2_energy(t, J)
        de = e - e_ref
        tag = "ANCHOR" if eps == 0.0 else f"{eps:g}"
        row = (f"{tag:>8s}  E_corr={e:.10f}  dE={de:+.3e}  "
               f"keep={st['frac']:.4f} pairs={st['pair_frac']:.3f} "
               f"dom(mean/max)={st['dom_mean']:.1f}/{st['dom_max']} of {nv}  "
               f"cg={niter} ({time.time()-t1:.1f}s)")
        log("  " + row)
        rows.append((eps, e, de, st, niter))
        if out:
            with open(out, "a") as f:
                f.write(f"{xyz} {basis} {row}\n")
        if eps == 0.0:
            ok = abs(de) < 1e-9
            if mutate:
                verdict = ("MUTATION-OK (anchor FAILED as required)"
                           if abs(de) > 1e-6 else
                           "MUTATION-BROKEN: anchor still passes!")
                log(f"  {verdict}  |dE|={abs(de):.3e}")
                return
            log(f"  ANCHOR {'PASSED' if ok else 'FAILED'} |dE|={abs(de):.3e}")
            if not ok:
                raise SystemExit("exactness anchor failed; no sweep run")
            if anchor_only:
                return
    log(f"  total {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz", required=True)
    ap.add_argument("--basis", default="6-31g")
    ap.add_argument("--eps", default="1e-3,1e-4,1e-5,1e-6,1e-7,1e-8")
    ap.add_argument("--anchor-only", action="store_true")
    ap.add_argument("--mutate", action="store_true",
                    help="deliberately break the construction; anchor must FAIL")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    eps_list = [float(x) for x in a.eps.split(",") if x]
    run(a.xyz, a.basis, eps_list, a.anchor_only, a.mutate, a.out)


if __name__ == "__main__":
    main()
