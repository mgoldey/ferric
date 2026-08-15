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


def build_ragged(mask):
    """Per-pair domain blocks from the boolean mask.

    For each occupied pair (i,j) with any retained (a,b): the union virtual
    domains Da = {a: any b kept}, Db = {b: any a kept}, and the retained
    pattern restricted to the (Da x Db) block. Amplitudes live ONLY on these
    blocks — this is the ragged-list layout whose matvec cost tracks retained
    work instead of the dense tensor size.
    """
    no, nv = mask.shape[0], mask.shape[1]
    pairs = []
    for i in range(no):
        for j in range(no):
            m = mask[i, :, j, :]
            if not m.any():
                continue
            da = np.nonzero(m.any(axis=1))[0]
            db = np.nonzero(m.any(axis=0))[0]
            pairs.append((i, j, da, db, m[np.ix_(da, db)]))
    return pairs


def solve_masked_mp2_ragged(J, Foo, Fvv, mask, rtol=1e-11, maxiter=400,
                            mutate=False):
    """Same fixed-sparsity problem as solve_masked_mp2, but on ragged per-pair
    domain blocks: Fvv terms are (d_a x d_a)@(d_a x d_b) block GEMMs, Foo
    terms are gathers between pairs sharing an occupied index. Independent
    algebra path from the dense masked solver (cross-check target).

    Returns (t_dense, niter, relres, work) where work has flops/matvec for
    the ragged path and the dense-equivalent count.
    """
    no, nv = J.shape[0], J.shape[1]
    fo, fv = np.diag(Foo).copy(), np.diag(Fvv).copy()
    Fvv = Fvv.copy()
    if mutate:
        # ragged-path-only corruption; xcheck MUST fail. Sized large because
        # the Hylleraas-type energy is QUADRATICALLY insensitive to operator
        # perturbations (measured: 1e-3 here moved E by only 1.8e-7).
        Fvv[0, 1] += 5e-2
    pairs = build_ragged(mask)
    npair = len(pairs)
    # pair lookup and per-(i,*)/(*,j) partner lists for the Foo gathers
    idx = {(i, j): p for p, (i, j, *_ ) in enumerate(pairs)}
    by_j = {}
    by_i = {}
    for p, (i, j, *_ ) in enumerate(pairs):
        by_j.setdefault(j, []).append(p)
        by_i.setdefault(i, []).append(p)

    def blocks_like():
        return [np.zeros((len(da), len(db))) for (_, _, da, db, _) in pairs]

    # rhs, denominators, pattern masks per block
    rhs, denom, pat = [], [], []
    for (i, j, da, db, m) in pairs:
        rhs.append(np.where(m, -J[i, :, j, :][np.ix_(da, db)], 0.0))
        denom.append(fv[da][:, None] + fv[db][None, :] - fo[i] - fo[j])
        pat.append(m)
    assert all(d.min() > 0 for d in denom), "non-positive denominator"

    flops = 0

    def matvec(t):
        nonlocal flops
        r = blocks_like()
        for p, (i, j, da, db, m) in enumerate(pairs):
            # + Fvv t + t Fvv (virtual couplings, block-local)
            r[p] += Fvv[np.ix_(da, da)] @ t[p]
            r[p] += t[p] @ Fvv[np.ix_(db, db)]
            flops += len(da) * len(da) * len(db) + len(da) * len(db) * len(db)
            # - Foo t (couples (k,j) into (i,j) on shared j)
            for q in by_j[j]:
                k, _, dak, dbk, _ = pairs[q]
                f = Foo[i, k]
                if f == 0.0:
                    continue
                ca, ia_, ka_ = np.intersect1d(da, dak, return_indices=True)
                cb, ib_, kb_ = np.intersect1d(db, dbk, return_indices=True)
                if len(ca) and len(cb):
                    r[p][np.ix_(ia_, ib_)] -= f * t[q][np.ix_(ka_, kb_)]
                    flops += len(ca) * len(cb)
            # - t Foo (couples (i,k) into (i,j) on shared i)
            for q in by_i[i]:
                _, k, dak, dbk, _ = pairs[q]
                f = Foo[k, j]
                if f == 0.0:
                    continue
                ca, ia_, ka_ = np.intersect1d(da, dak, return_indices=True)
                cb, ib_, kb_ = np.intersect1d(db, dbk, return_indices=True)
                if len(ca) and len(cb):
                    r[p][np.ix_(ia_, ib_)] -= f * t[q][np.ix_(ka_, kb_)]
                    flops += len(ca) * len(cb)
            np.multiply(r[p], pat[p], out=r[p])
        return r

    def dot(x, y):
        return sum(np.vdot(a, b) for a, b in zip(x, y))

    bnorm = np.sqrt(dot(rhs, rhs))
    t = blocks_like()
    if bnorm == 0.0:
        return np.zeros_like(J), 0, 0.0, dict(flops_per_matvec=0)
    r = [b.copy() for b in rhs]
    z = [rb / d for rb, d in zip(r, denom)]
    p_ = [zb.copy() for zb in z]
    rz = dot(r, z)
    it = 0
    for it in range(1, maxiter + 1):
        Ap = matvec(p_)
        alpha = rz / dot(p_, Ap)
        for k in range(npair):
            t[k] += alpha * p_[k]
            r[k] -= alpha * Ap[k]
        relres = np.sqrt(dot(r, r)) / bnorm
        if relres < rtol:
            break
        z = [rb / d for rb, d in zip(r, denom)]
        rz_new = dot(r, z)
        beta = rz_new / rz
        p_ = [zb + beta * pb for zb, pb in zip(z, p_)]
        rz = rz_new
    else:
        log(f"  WARNING: ragged CG hit maxiter={maxiter}, relres={relres:.2e}")

    t_dense = np.zeros_like(J)
    for pn, (i, j, da, db, _) in enumerate(pairs):
        t_dense[i, :, j, :][np.ix_(da, db)] = t[pn]
    dense_flops = 2 * (no * no * nv**3 + no**3 * nv**2)
    work = dict(flops_per_matvec=flops // max(it, 1),
                dense_flops_per_matvec=dense_flops, npair=npair)
    return t_dense, it, relres, work


def mp2_energy(t, J):
    """E = sum (2 t_iajb - t_ibja) J_iajb (spin-adapted closed shell)."""
    return 2.0 * np.vdot(t, J) - np.vdot(t.transpose(0, 3, 2, 1), J)


def pair_energies(t, J):
    """Exact per-pair energies e_ij (sum over a,b); sums to mp2_energy."""
    return (2.0 * np.einsum("iajb,iajb->ij", t, J, optimize=True)
            - np.einsum("ibja,iajb->ij", t, J, optimize=True))


def pair_gate_stats(mol, C_act, t, J, eps_list):
    """How well does an integral-free R^-6 estimator rank occupied pairs?

    Estimator: London-type e_est_ij = s_i^3 s_j^3 / R_ij^6 from Boys
    centroids R and spreads s (dipole/r2 one-electron integrals only — the
    quantities an integral-direct code has BEFORE any 4-index work).
    Prints Spearman rank correlation on the off-diagonal pairs and, for each
    theta = 1e-2*eps, the energy lost by dropping pairs below theta under
    (a) oracle ranking by exact |e_ij| and (b) estimator ranking calibrated
    so its scale matches exact |e_ij| at the median retained pair.
    """
    no = t.shape[0]
    e_ex = pair_energies(t, J)
    e_abs = np.abs(e_ex)
    rints = mol.intor("int1e_r")
    r2 = mol.intor("int1e_r2")
    cen = np.einsum("xmn,mi,ni->ix", rints, C_act, C_act)
    spread2 = np.einsum("mi,mn,ni->i", C_act, r2, C_act) - (cen**2).sum(axis=1)
    s = np.sqrt(np.maximum(spread2, 1e-10))
    R = np.linalg.norm(cen[:, None, :] - cen[None, :, :], axis=2)
    off = ~np.eye(no, dtype=bool)
    est = np.where(off, (s[:, None] * s[None, :])**3
                   / np.maximum(R, 1e-6)**6, np.inf)  # diagonal never gated
    # Spearman on off-diagonal upper triangle
    iu = np.triu_indices(no, 1)
    a, b = e_abs[iu], est[iu]
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    rho = np.corrcoef(ra, rb)[0, 1]
    # CONSERVATIVE calibration: p95 of the exact/estimate ratio, so the
    # scaled estimator over-predicts nearly every pair energy and
    # est_cal < theta (almost) implies exact < theta. Median calibration
    # was measured 6x worse than oracle at the same theta (over-drops).
    finite = np.isfinite(est) & off & (e_abs > 0)
    cal = np.percentile(e_abs[finite] / est[finite], 95)
    est_cal = est * cal
    log(f"  pair-gate: {no} occ, spearman(est,exact)={rho:.3f} "
        f"cal(p95)={cal:.2e}")
    for eps in eps_list:
        theta = 1e-2 * eps
        for name, score in (("oracle", e_abs), ("est   ", est_cal)):
            drop = off & (score < theta)
            elost = e_ex[drop].sum()
            log(f"    theta=1e-2*{eps:g} {name}: dropped "
                f"{drop.sum()}/{no*no} pairs, E_lost={elost:+.3e} Ha")


def domain_stats(mask):
    """Retention + per-LMO domain sizes from the boolean mask."""
    frac = mask.mean()
    pair_any = mask.any(axis=(1, 3))                 # (no, no)
    pair_frac = pair_any.mean()
    dom = mask.any(axis=(2, 3)).sum(axis=1)          # |{a: any (i,a,j,b) kept}|
    return dict(frac=frac, pair_frac=pair_frac,
                dom_mean=float(dom.mean()), dom_max=int(dom.max()),
                dom_min=int(dom.min()))


def canonical_sr_mp2(mol, mf, ncore, omega):
    """Canonical-basis SR (erfc) MP2 — the independent reference for --omega.

    Same physics convention as attenuated MP2 (Goldey/Head-Gordon): full-SCF
    Fock/denominators, only the perturbation ERIs are attenuated.
    """
    nocc = np.count_nonzero(mf.mo_occ > 0)
    Co, Cv = mf.mo_coeff[:, ncore:nocc], mf.mo_coeff[:, nocc:]
    eo, ev = mf.mo_energy[ncore:nocc], mf.mo_energy[nocc:]
    with mol.with_range_coulomb(-omega):
        ovov = ao2mo.general(mol, (Co, Cv, Co, Cv), compact=False)
    no, nv = Co.shape[1], Cv.shape[1]
    ovov = ovov.reshape(no, nv, no, nv)
    D = (eo[:, None, None, None] - ev[None, :, None, None]
         + eo[None, None, :, None] - ev[None, None, None, :])
    t = ovov / D
    return 2.0 * np.vdot(t, ovov) - np.vdot(t.transpose(0, 3, 2, 1), ovov)


def run(xyz, basis, eps_list, anchor_only=False, mutate=False, out=None,
        omega=None, solver="dense", xcheck=False, mutate_ragged=False,
        pair_stats=False):
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
    e_full = pt.kernel()[0]
    log(f"  E_corr(canonical MP2, frozen={ncore}) = {e_full:.10f}")
    if omega is not None:
        e_ref = canonical_sr_mp2(mol, mf, ncore, omega)
        log(f"  E_corr(canonical SR-MP2, omega={omega} Bohr^-1) = {e_ref:.10f}")
        # shared-bug guard: a silently no-op'd with_range_coulomb makes both
        # paths full-Coulomb and the anchor passes vacuously
        if not abs(e_ref) < abs(e_full) - 1e-10:
            raise SystemExit("SR guard FAILED: |E_sr| >= |E_coulomb| — "
                             "with_range_coulomb no-op or sign error")
    else:
        e_ref = e_full

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
    if omega is not None:
        with mol.with_range_coulomb(-omega):
            J = ao2mo.general(mol, (C_act, C_vloc, C_act, C_vloc),
                              compact=False)
    else:
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
        extra = ""
        if solver == "ragged" and eps > 0.0:
            t, niter, relres, work = solve_masked_mp2_ragged(
                J, Foo, Fvv, mask, mutate=mutate_ragged)
            ratio = work["dense_flops_per_matvec"] / max(work["flops_per_matvec"], 1)
            extra = (f" ragged[npair={work['npair']} "
                     f"mflop/mv={work['flops_per_matvec']/1e6:.1f} "
                     f"densex={ratio:.0f}]")
        else:
            t, niter, relres = solve_masked_mp2(J, Foo, Fvv, mask)
        e = mp2_energy(t, J)
        if solver == "ragged" and eps > 0.0 and xcheck:
            td, _, _ = solve_masked_mp2(J, Foo, Fvv, mask)
            ed = mp2_energy(td, J)
            dx = abs(e - ed)
            if mutate_ragged:
                # judged at the xcheck's own 1e-10 bar, not a separate one
                verdict = ("MUTATION-OK (xcheck FAILED as required)"
                           if dx > 1e-10 else
                           "MUTATION-BROKEN: xcheck still passes!")
                log(f"  {verdict}  |dE(ragged-dense)|={dx:.3e}")
                return
            log(f"  XCHECK ragged-vs-dense |dE|={dx:.3e} "
                f"{'PASSED' if dx <= 1e-10 else 'FAILED'}")
            if dx > 1e-10:
                raise SystemExit("ragged/dense cross-check failed")
        de = e - e_ref
        tag = "ANCHOR" if eps == 0.0 else f"{eps:g}"
        wtag = "coulomb" if omega is None else f"w={omega:g}"
        row = (f"{wtag:>8s} {tag:>8s}  E_corr={e:.10f}  dE={de:+.3e}  "
               f"keep={st['frac']:.4f} pairs={st['pair_frac']:.3f} "
               f"dom(mean/max)={st['dom_mean']:.1f}/{st['dom_max']} of {nv}  "
               f"cg={niter} ({time.time()-t1:.1f}s){extra}")
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
            if pair_stats:
                pair_gate_stats(mol, C_act, t, J, eps_list)
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
    ap.add_argument("--omega", type=float, default=None,
                    help="SR erfc attenuation, Bohr^-1 (proxy for sharp terfc); "
                         "reference becomes canonical SR-MP2 at the same omega")
    ap.add_argument("--solver", choices=["dense", "ragged"], default="dense",
                    help="ragged = per-pair domain-block CG (cost tracks "
                         "retained work); dense = masked dense einsum CG")
    ap.add_argument("--xcheck", action="store_true",
                    help="with --solver ragged: also run the dense solver on "
                         "the same mask and require |dE| <= 1e-10")
    ap.add_argument("--mutate-ragged", action="store_true",
                    help="corrupt Fvv inside the ragged path only; "
                         "--xcheck must then FAIL")
    ap.add_argument("--pair-stats", action="store_true",
                    help="after the anchor: exact pair energies vs the "
                         "integral-free R^-6 estimator (spearman + theta scan)")
    a = ap.parse_args()
    eps_list = [float(x) for x in a.eps.split(",") if x]
    run(a.xyz, a.basis, eps_list, a.anchor_only, a.mutate, a.out,
        omega=a.omega, solver=a.solver, xcheck=a.xcheck or a.mutate_ragged,
        mutate_ragged=a.mutate_ragged, pair_stats=a.pair_stats)


if __name__ == "__main__":
    main()
