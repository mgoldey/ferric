#!/usr/bin/env python
"""Local (amplitude-threshold) OP-MP3 prototype in numpy on PySCF.

Question: does the WSHG23 single-threshold local approximation used for
ferric's LMP2 carry over to OP-MP3 (Bystrom/Berkelbach/Hait) without the
matrix Pade amplifying the truncation error? Working equations, the
consistent truncation choice and the hypotheses are in
local_opmp3_derivation.md (next to this file). Correctness reference only:
nothing about speed may be quoted from it except the cost proxies it
counts.

Pipeline (closed-shell RHF, frozen core):

1. Canonical reference: first- and second-order doubles amplitudes
   c1 = -K/D, c2 = -R(c1)/D in canonical orbitals; W^(2), W^(3) split into
   same-spin (SS) and opposite-spin (OS) channels; OP-MP3, SOP-MP3,
   SSD-OPMP3 from the matrix Pade.
2. Local: Boys occupied + VV-HV virtuals (build_vvhv from
   scripts/amplitude_lmp2_proto.py, the prototype ferric's lmp2_amplitude
   was ported from). c1 and c2 solve the non-canonical equations
   A c = rhs (A = the Fock superoperator) by masked PCG on the WSHG23
   integral mask (keep (i,a,j,b) iff |(ia|jb)| > eps or |(ib|ja)| > eps).
3. Errors vs the canonical reference at each eps, the Pade spectrum, a
   first-order (linearized Pade) prediction of the OP-MP3 error from the
   measured W errors, and cost proxies (domain ladder / ring work).

Usage (one heavy op at a time; under the memory cap):
  scripts/ferric-limited --max=6G --high=5G -- box-lease exec -- env \\
    OPENBLAS_NUM_THREADS=1 /home/matt/qc/ferric/.venv/bin/python \\
    scripts/prototypes/local_opmp3_proto.py --xyz testdata/molecules/alkane_4.xyz \\
    --basis sto-3g --eps 1e-3,1e-4,1e-5,1e-6,1e-7 --json out.json
  ... --anchor-only            eps = 0 anchor and stop
  ... --ferric                 also cross-check canonical vs ferric run_opmp3
  ... --stretch 1.5            stretch the central C-C bond of an alkane
  local_opmp3_proto.py --summarize local_opmp3_data/*.json   (tables)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# SSD-OPMP3 same-spin damping: ferric's PAPER_X_SS for an HF reference is 0;
# 0.5 (PBE0 / wB97X-V value) is also reported so the SS Pade is exercised.
X_SS_HF = 0.0
X_SS_ALT = 0.5


def _lmp2():
    """Import scripts/amplitude_lmp2_proto.py (build_vvhv, load_xyz, ...)."""
    spec = importlib.util.spec_from_file_location(
        "amplitude_lmp2_proto", ROOT / "scripts" / "amplitude_lmp2_proto.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# Matrix Pade (paper Eq. 13) and the models (Eqs. 14, 19-22)
# ---------------------------------------------------------------------------


def pade(a, b):
    """P(A, B) = A (A - B)^{-1} A; rows where A vanishes are dropped."""
    keep = np.flatnonzero(np.abs(a).sum(axis=0) > 0)
    out = np.zeros_like(a)
    if keep.size == 0:
        return out
    ak, bk = a[np.ix_(keep, keep)], b[np.ix_(keep, keep)]
    out[np.ix_(keep, keep)] = ak @ np.linalg.solve(ak - bk, ak)
    return out


def tr(m):
    return float(np.trace(m))


def models(w):
    """Correlation energies from the closed-shell W set.

    w[(n, ch)] is the alpha block of order n in {2, 3}, channel in
    {'ss', 'os'}; the beta block equals it for RHF, so every model is
    twice its alpha trace.
    """
    a2 = w[(2, "ss")] + w[(2, "os")]
    a3 = w[(3, "ss")] + w[(3, "os")]
    e2, e3 = 2 * tr(a2), 2 * tr(a3)
    pos = 2 * tr(pade(w[(2, "os")], w[(3, "os")]))

    def ssd(x):
        return 2 * tr(pade(w[(2, "ss")], x * w[(3, "ss")])) + pos

    return {
        "e2": e2,
        "e3": e3,
        "e2_ss": 2 * tr(w[(2, "ss")]),
        "e2_os": 2 * tr(w[(2, "os")]),
        "e3_ss": 2 * tr(w[(3, "ss")]),
        "e3_os": 2 * tr(w[(3, "os")]),
        "mp3": e2 + e3,
        "p-mp3": e2 * e2 / (e2 - e3),
        "op-mp3": 2 * tr(pade(a2, a3)),
        "sop-mp3": ssd(1.0),
        "ssd-opmp3": ssd(X_SS_HF),
        "ssd-opmp3(0.5)": ssd(X_SS_ALT),
    }


def pade_grads(a, b):
    """dTr P = Tr[G_A dA] + Tr[G_B dB] at (A, B) (all symmetric).

    M = A - B, P = A M^-1 A:
    dP = dA M^-1 A + A M^-1 dA - A M^-1 (dA - dB) M^-1 A, so
    G_A = M^-1 A + A M^-1 - M^-1 A^2 M^-1 and G_B = M^-1 A^2 M^-1.
    """
    mi = np.linalg.inv(a - b)
    ga = mi @ a + a @ mi - mi @ a @ a @ mi
    gb = mi @ a @ a @ mi
    return ga, gb


def spectrum(a, b):
    """Pade diagnostics: eigenvalue of A - B nearest zero, and the largest
    zeta = eig(A^-1 B) (each eigen-channel's scalar Pade is a/(1-zeta))."""
    lam = np.linalg.eigvalsh(a - b)
    zeta = np.linalg.eigvals(np.linalg.solve(a, b)).real
    return float(lam[np.argmin(np.abs(lam))]), float(zeta.max())


# ---------------------------------------------------------------------------
# Closed-shell amplitudes, residual and W
# ---------------------------------------------------------------------------


def mo_ints(mol, co, cv, g=None):
    """Chemist (ia|jb), (ij|ab), (ij|kl), (ab|cd) over the given orbitals.

    g(c1, c2, c3, c4) -> (pq|rs) is the integral engine; default PySCF
    ao2mo on `mol` (the ferric cross-check passes ferric's AoIntegrals.ao2mo).
    """
    if g is None:
        from pyscf import ao2mo

        def g(c1, c2, c3, c4):
            shape = tuple(c.shape[1] for c in (c1, c2, c3, c4))
            return ao2mo.general(mol, (c1, c2, c3, c4), compact=False).reshape(shape)

    return {
        "ovov": g(co, cv, co, cv),
        "oovv": g(co, co, cv, cv),
        "oooo": g(co, co, co, co),
        "vvvv": g(cv, cv, cv, cv),
    }


class DFFactors:
    """Same-metric RI factors B^P_pq = sum_Q L^-1_PQ (Q|pq), V = L L^T,
    from ONE AO 3-index tensor; canonical and local B are both transforms of
    it, so the canonical-vs-local anchor is exact at any auxiliary basis."""

    def __init__(self, mol, auxbasis):
        from pyscf import df

        if auxbasis == "auto":
            auxbasis = df.addons.make_auxbasis(mol, mp2fit=True)
        self.auxbasis = auxbasis
        auxmol = df.addons.make_auxmol(mol, auxbasis)
        self.ao3 = df.incore.aux_e2(mol, auxmol, intor="int3c2e")  # (nao,nao,P)
        self.lchol = np.linalg.cholesky(auxmol.intor("int2c2e"))
        self.naux = self.ao3.shape[2]

    def b(self, c1, c2):
        """B[P, p, q] over the columns of c1, c2."""
        from scipy.linalg import solve_triangular

        x = np.einsum("mnP,mp,nq->Ppq", self.ao3, c1, c2, optimize=True)
        n1, n2 = x.shape[1], x.shape[2]
        x = solve_triangular(self.lchol, x.reshape(self.naux, -1), lower=True)
        return x.reshape(self.naux, n1, n2)

    def ints(self, co, cv, dense_vvvv=False):
        boo, bov, bvv = self.b(co, co), self.b(co, cv), self.b(cv, cv)
        no, nv = co.shape[1], cv.shape[1]
        nax = self.naux
        out = {
            "ovov": (bov.reshape(nax, -1).T @ bov.reshape(nax, -1)).reshape(
                no, nv, no, nv
            ),
            "oovv": (boo.reshape(nax, -1).T @ bvv.reshape(nax, -1)).reshape(
                no, no, nv, nv
            ),
            "oooo": (boo.reshape(nax, -1).T @ boo.reshape(nax, -1)).reshape(
                no, no, no, no
            ),
            "bvv": bvv,
        }
        if dense_vvvv:
            # independent algebra for the DF ladder (small systems only)
            out["vvvv"] = (bvv.reshape(nax, -1).T @ bvv.reshape(nax, -1)).reshape(
                nv, nv, nv, nv
            )
        return out


def swap_ab(t):
    """t[i,a,j,b] -> t[i,b,j,a]."""
    return t.transpose(0, 3, 2, 1)


def df_ladder(t, bvv, mask=None, mutate=None):
    """Particle ladder sum_cd (ac|bd) t_ijcd from DF factors, per pair inside
    the pair's domain, the way an integral-direct port would do it.

    bvv[P, a, c] = B^P_ac with (ac|bd) = sum_P B^P_ac B^P_bd. For pair (i, j)
    with retained pattern S_ij, Da = {a: (a, b) in S_ij for some b} and Db
    likewise; t_ij vanishes outside Da x Db, so on Da x Db
        R_ij = sum_P B^P[Da, Da] t_ij[Da, Db] B^P[Db, Db]^T
    is exact. Cost naux (|Da|^2 |Db| + |Da| |Db|^2) per pair. Only the
    i <= j blocks are formed; R_ji = R_ij^T. Entries outside the domain blocks
    are left zero (the masked solve never reads them), so mask=None (full
    domains) is the only setting whose R is valid everywhere.

    mutate='pair_domain_ii' reads pair (i, j)'s domain from pair (i, i): an
    index-set mix that a vacuous mask cannot see (the finite-eps domain
    check must fail).
    """
    no, nv = t.shape[0], t.shape[1]
    naux = bvv.shape[0]
    r = np.zeros_like(t)
    full = np.arange(nv)
    for i in range(no):
        for j in range(i, no):
            if mask is None:
                da = db = full
            else:
                m = mask[i, :, i, :] if mutate == "pair_domain_ii" else mask[i, :, j, :]
                if not m.any():
                    continue
                da = np.flatnonzero(m.any(axis=1))
                db = np.flatnonzero(m.any(axis=0))
            x = t[i, :, j, :][np.ix_(da, db)]
            ba = bvv[:, da][:, :, da]  # (P, a, c)
            bb = bvv[:, db][:, :, db]  # (P, b, d)
            y = np.matmul(ba, x)  # (P, a, d)
            yt = y.transpose(1, 0, 2).reshape(len(da), naux * len(db))
            blk = yt @ bb.transpose(0, 2, 1).reshape(naux * len(db), len(db))
            r[i, :, j, :][np.ix_(da, db)] = blk
            if j != i:
                r[j, :, i, :][np.ix_(db, da)] = blk.T
    return r


def residual(t, ints, drop_term=None, mask=None):
    """Closed-shell MP3 residual R(t) for the alpha-beta amplitude
    t[i,a,j,b] = c^{ab}_{iJaB} (theory.md 6.1, R^{alpha beta} with
    alpha = beta). Second-order amplitudes solve A c2 = -R(c1).

    The particle ladder is the dense (ac|bd) contraction when ints has
    'vvvv', else the per-pair DF contraction df_ladder over the domains of
    `mask` (R is then valid on the mask's domain blocks only).
    drop_term (1..8) removes one term: a mutation for the ferric cross-check.
    """
    ovov, oovv = ints["ovov"], ints["oovv"]
    no, nv = t.shape[0], t.shape[1]
    n = no * nv
    taa = t - swap_ab(t)
    m = ovov.reshape(n, n)
    tf, taf = t.reshape(n, n), taa.reshape(n, n)
    terms = {
        # particle ladder sum_cd (ac|bd) t_ijcd
        1: lambda: (
            np.einsum("acbd,icjd->iajb", ints["vvvv"], t, optimize=True)
            if "vvvv" in ints
            else df_ladder(t, ints["bvv"], mask, ints.get("ladder_mutate"))
        ),
        # hole ladder sum_kl (ki|lj) t_klab
        2: lambda: np.einsum("kilj,kalb->iajb", ints["oooo"], t, optimize=True),
        # sum_kc (kc|jb) taa_ikac and its (ia)<->(jb) mirror
        3: lambda: (taf @ m).reshape(t.shape) + (m @ taf.T).reshape(t.shape),
        # sum_kc (kc|jb) t_ikac and mirror
        4: lambda: (tf @ m).reshape(t.shape) + (m @ tf).reshape(t.shape),
        # - sum_kc (kj|bc) t_ikac and mirror - sum_kc (ik|ac) t_kjcb
        5: lambda: (
            -np.einsum("kjbc,iakc->iajb", oovv, t, optimize=True)
            - np.einsum("ikac,kcjb->iajb", oovv, t, optimize=True)
        ),
        # - sum_kc (ki|bc) t_kjac
        7: lambda: -np.einsum("kibc,kajc->iajb", oovv, t, optimize=True),
        # - sum_kc (kj|ac) t_ikcb
        8: lambda: -np.einsum("kjac,ickb->iajb", oovv, t, optimize=True),
    }
    r = np.zeros_like(t)
    for k, f in terms.items():
        if k != drop_term:
            r += f()
    return r


def energy(t, j):
    """E = sum (2 t - t_swap) (ia|jb)."""
    return 2.0 * np.vdot(t, j) - np.vdot(swap_ab(t), j)


def w_mats(t, j):
    """Alpha-block W^SS and W^OS of one order from its amplitudes
    (theory.md 6.2 with alpha = beta): W^OS_ij = 1/4 sum t_iakb (ja|kb) +
    (i<->j); W^SS the same with t - t_swap."""
    xos = 0.25 * np.einsum("iakb,jakb->ij", t, j, optimize=True)
    xss = 0.25 * np.einsum("iakb,jakb->ij", t - swap_ab(t), j, optimize=True)
    return xss + xss.T, xos + xos.T


def apply_a(t, foo, fvv):
    """Fock superoperator: (Fvv t + t Fvv - Foo t - t Foo)_iajb."""
    r = np.einsum("ac,icjb->iajb", fvv, t, optimize=True)
    r += np.einsum("iajc,cb->iajb", t, fvv, optimize=True)
    r -= np.einsum("ik,kajb->iajb", foo, t, optimize=True)
    r -= np.einsum("iakb,kj->iajb", t, foo, optimize=True)
    return r


def denom(foo, fvv):
    fo, fv = np.diag(foo), np.diag(fvv)
    return (
        fv[None, :, None, None]
        + fv[None, None, None, :]
        - fo[:, None, None, None]
        - fo[None, None, :, None]
    )


def pcg(rhs, foo, fvv, mask, rtol=1e-12, maxiter=800):
    """Solve P A P x = P rhs on the mask by Jacobi-preconditioned CG."""
    d = denom(foo, fvv)
    if d.min() <= 0:
        raise RuntimeError("non-positive denominator")
    r = np.where(mask, rhs, 0.0)
    bnorm = np.linalg.norm(r)
    x = np.zeros_like(rhs)
    if bnorm == 0.0:
        return x, 0, 0.0
    z = r / d
    p = z.copy()
    rz = np.vdot(r, z)
    for it in range(1, maxiter + 1):
        ap = apply_a(p, foo, fvv)
        ap *= mask
        alpha = rz / np.vdot(p, ap)
        x += alpha * p
        r -= alpha * ap
        rn = np.linalg.norm(r)
        if rn / bnorm < rtol:
            return x, it, float(rn)
        z = r / d
        rz_new = np.vdot(r, z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    raise RuntimeError(f"PCG did not converge in {maxiter} iterations")


# ---------------------------------------------------------------------------
# The system: reference, canonical and local orbitals
# ---------------------------------------------------------------------------


def stretched_alkane(atom_str, factor):
    """Stretch the central C-C bond of an alkane by `factor` (rigid shift
    of one half along the bond axis)."""
    lines = [ln.split() for ln in atom_str.splitlines() if ln.strip()]
    sym = [ln[0] for ln in lines]
    x = np.array([[float(v) for v in ln[1:4]] for ln in lines])
    nat = len(sym)
    d = np.linalg.norm(x[:, None] - x[None], axis=2)
    cut = np.where(
        (np.array(sym)[:, None] == "C") & (np.array(sym)[None] == "C"), 1.7, 1.25
    )
    adj = (d < cut) & ~np.eye(nat, dtype=bool)
    cc = [
        (p, q)
        for p in range(nat)
        for q in range(p + 1, nat)
        if adj[p, q] and sym[p] == sym[q] == "C"
    ]
    best = None
    for p, q in cc:
        side, stack = {q}, [q]
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(adj[u]):
                if (u, v) in ((q, p),) or v == p or v in side:
                    continue
                side.add(int(v))
                stack.append(int(v))
        bal = abs(len(side) - (nat - len(side)))
        if best is None or bal < best[0]:
            best = (bal, p, q, side)
    _, p, q, side = best
    u = x[q] - x[p]
    shift = (factor - 1.0) * u
    for k in side:
        x[k] += shift
    return "\n".join(f"{s} {a:.10f} {b:.10f} {c:.10f}" for s, (a, b, c) in zip(sym, x))


def replicate(atom_str, copies, sep_bohr):
    """`copies` copies of the molecule, translated by sep_bohr along x each
    (the size-consistency supersystem)."""
    shift = sep_bohr * 0.529177210903
    out = []
    for n in range(copies):
        for ln in atom_str.splitlines():
            if not ln.strip():
                continue
            sym, *xyz = ln.split()
            x, y, z = (float(v) for v in xyz[:3])
            out.append(f"{sym} {x + n * shift:.10f} {y:.10f} {z:.10f}")
    return "\n".join(out)


BOYS_STABLE = {"on": False}  # set from --boys-stable


def boys_stabilize(mol, c, tol=1e-8, max_sweeps=50):
    """Leave a Boys saddle point: Jacobi sweeps over orbital pairs. For a
    pair rotation by theta the Boys spread is exactly A + B cos 4theta +
    C sin 4theta, so three evaluations give its minimum; a pair is rotated
    when that lowers the spread by more than tol. Then Boys again from the
    result. Returns (C, number of pair rotations applied, spread before,
    spread after)."""
    from pyscf import lo

    r1 = mol.intor("int1e_r")
    r2 = mol.intor("int1e_r2")

    def sp(m):
        m1 = np.einsum("pk,xpq,qk->xk", m, r1, m)
        return float((np.einsum("pk,pq,qk->k", m, r2, m) - (m1**2).sum(0)).sum())

    def rot(m, th):
        cs, sn = np.cos(th), np.sin(th)
        return np.column_stack(
            [cs * m[:, 0] + sn * m[:, 1], -sn * m[:, 0] + cs * m[:, 1]]
        )

    c = c.copy()
    before, n_rot = sp(c), 0
    for _ in range(max_sweeps):
        changed = False
        for p in range(c.shape[1]):
            for q in range(p + 1, c.shape[1]):
                m = c[:, [p, q]]
                f0, f1, f2 = sp(m), sp(rot(m, np.pi / 8)), sp(rot(m, np.pi / 4))
                # f(t) = A + B cos4t + C sin4t at t = 0, pi/8, pi/4
                a_ = 0.5 * (f0 + f2)
                b_ = f0 - a_
                c_ = f1 - a_
                th = (np.arctan2(-c_, -b_)) / 4.0  # minimizer of B cos + C sin
                if sp(rot(m, th)) < f0 - tol:
                    c[:, [p, q]] = rot(m, th)
                    n_rot += 1
                    changed = True
        if not changed:
            break
    mlo = lo.Boys(mol, c)
    mlo.init_guess = None
    c = mlo.kernel()
    return c, n_rot, before, sp(c)


# Occupied localization: 'pyscf' (pyscf.lo.Boys as is; can stop at a saddle,
# the pre-2026-11-17 tables), 'stable' (boys_stabilize on occ and VV, the
# analogue of ferric's Jacobi Boys), 'lambda' (stable, then lambda-Boys on the
# occupied block, ferric's default since e0b5ff1f3). Set from --orbitals /
# --maps; BOYS_STABLE['on'] is the legacy spelling of 'stable'.
ORB = {"occ": "pyscf", "lambda": 10.0}
# ferric-mp2/src/boys.rs LAMBDA_BOYS_GRAD_TOL / LAMBDA_BOYS_MAX_SWEEPS
LAMBDA_BOYS_GRAD_TOL = 1e-8
LAMBDA_BOYS_MAX_SWEEPS = 2000


def lambda_boys_objective(x, f, lam):
    """G = sum_x sum_i (r^x_ii)^2 + lam sum_i F_ii^2 (book eq
    lmp2-combined-lambda-boys)."""
    return float(
        sum((np.diag(m) ** 2).sum() for m in x) + lam * (np.diag(f) ** 2).sum()
    )


def lambda_boys_max_grad(x, f, lam):
    """max_{p<q} |4 sum_X w_X (X_pp - X_qq) X_pq|."""
    g = np.zeros_like(f)
    for m, w in [(m, 1.0) for m in x] + [(f, lam)]:
        d = np.diag(m)
        g += w * (d[:, None] - d[None, :]) * m
    iu = np.triu_indices(f.shape[0], 1)
    return float(4.0 * np.abs(g[iu]).max()) if iu[0].size else 0.0


def lambda_boys_rotation(
    x,
    f,
    lam,
    grad_tol=LAMBDA_BOYS_GRAD_TOL,
    max_sweeps=LAMBDA_BOYS_MAX_SWEEPS,
    mutate=None,
):
    """Port of ferric-mp2/src/boys.rs::lambda_boys_rotation: Jacobi
    maximization of G, pairs p < q in fixed order, theta = atan2(b, a)/4
    with d = (X_pp - X_qq)/2, c = X_pq, a = sum w (d^2 - c^2),
    b = 2 sum w d c; rotation p' = c p + s q, q' = -s p + c q. Stops when the
    largest pair gradient after a sweep is below grad_tol. mutate='sign'
    uses -theta (a broken optimiser: G must not rise, the gradient check
    must fail). Returns (U, sweeps, max_grad, G_start, G_end)."""
    x = [m.copy() for m in x]
    f = f.copy()
    n = f.shape[0]
    u = np.eye(n)
    mats = x + [f]
    wts = [1.0, 1.0, 1.0, lam]
    g0 = lambda_boys_objective(x, f, lam)
    mg = lambda_boys_max_grad(x, f, lam)
    sweeps = 0
    while mg >= grad_tol and sweeps < max_sweeps:
        sweeps += 1
        for p in range(n):
            for q in range(p + 1, n):
                a = b = 0.0
                for m, w in zip(mats, wts):
                    d = 0.5 * (m[p, p] - m[q, q])
                    cc = m[p, q]
                    a += w * (d * d - cc * cc)
                    b += w * 2.0 * d * cc
                th = 0.25 * np.arctan2(b, a)
                if mutate == "sign":
                    th = -th
                if th == 0.0:
                    continue
                c, s = np.cos(th), np.sin(th)
                for m in mats:
                    mp, mq = m[:, p].copy(), m[:, q].copy()
                    m[:, p], m[:, q] = c * mp + s * mq, -s * mp + c * mq
                    mp, mq = m[p, :].copy(), m[q, :].copy()
                    m[p, :], m[q, :] = c * mp + s * mq, -s * mp + c * mq
                up, uq = u[:, p].copy(), u[:, q].copy()
                u[:, p], u[:, q] = c * up + s * uq, -s * up + c * uq
        mg = lambda_boys_max_grad(x, f, lam)
        if mutate == "sign" and sweeps >= 20:
            break
    return u, sweeps, mg, g0, lambda_boys_objective(x, f, lam)


def lambda_boys_localize(mol, c_boys, fock_ao, lam, mutate=None):
    """lambda-Boys from converged Boys orbitals c_boys (ferric's
    lambda_boys_localize_checked). r^x = <i|x|j> about the AO origin, as
    ferric's dipole matrices; F = the AO Fock. lam = 0 returns c_boys."""
    if lam == 0.0:
        return c_boys, {"lambda": 0.0, "sweeps": 0, "max_grad": 0.0}
    r1 = mol.intor("int1e_r")
    x = [c_boys.T @ r1[k] @ c_boys for k in range(3)]
    f = c_boys.T @ fock_ao @ c_boys
    t0 = time.time()
    u, sweeps, mg, g0, g1 = lambda_boys_rotation(x, f, lam, mutate=mutate)
    if mg >= LAMBDA_BOYS_GRAD_TOL and mutate is None:
        raise RuntimeError(f"lambda-Boys not converged: {sweeps} sweeps, grad {mg:.2e}")
    off = lambda m: float((m**2).sum() - (np.diag(m) ** 2).sum())
    rep = {
        "lambda": lam,
        "sweeps": sweeps,
        "max_grad": mg,
        "G_start": g0,
        "G_end": g1,
        "phi_boys": off(f),
        "phi_lambda": off(u.T @ f @ u),
        "max_abs_u_minus_1": float(np.abs(np.abs(u) - np.eye(len(u))).max()),
        "t_s": time.time() - t0,
    }
    return c_boys @ u, rep


def orbital_spreads(sys_):
    """Mean Boys spread <r^2> - |<r>|^2 (Bohr^2) per local orbital, by space."""
    r1, r2 = sys_.mol.intor("int1e_r"), sys_.mol.intor("int1e_r2")

    def per(c):
        if c.shape[1] == 0:
            return None
        m1 = np.einsum("pk,xpq,qk->xk", c, r1, c)
        return float((np.einsum("pk,pq,qk->k", c, r2, c) - (m1**2).sum(0)).mean())

    return {
        "occ": per(sys_.co_loc),
        "vv": per(sys_.cv_loc[:, : sys_.n_vv]),
        "hv": per(sys_.cv_loc[:, sys_.n_vv :]),
    }


class System:
    """RHF reference, canonical and local active orbitals, MO integrals."""

    def __init__(
        self,
        xyz,
        basis,
        stretch=None,
        conv_tol=1e-11,
        copies=1,
        sep=100.0,
        reference="hf",
        auxbasis=None,
    ):
        from pyscf import dft, gto, scf
        from pyscf.data import elements

        lm = _lmp2()
        t0 = time.time()
        atom = lm.load_xyz(xyz)
        if stretch:
            atom = stretched_alkane(atom, stretch)
        if copies > 1:
            atom = replicate(atom, copies, sep)
        self.atom = atom
        mol = gto.M(atom=atom, basis=basis, verbose=0, max_memory=2000)
        self.mol, self.basis, self.xyz, self.stretch = mol, basis, xyz, stretch
        self.reference = reference
        self.ncore = elements.chemcore(mol)
        if reference == "hf":
            mf = scf.RHF(mol)
        else:
            mf = dft.RKS(mol, xc=reference)
            mf.grids.level = 5
        mf.conv_tol = conv_tol
        mf.conv_tol_grad = np.sqrt(conv_tol) * 0.1
        mf.max_cycle = 300
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("SCF not converged")
        self.mf = mf
        nocc = int(np.count_nonzero(mf.mo_occ > 0))
        self.nocc = nocc
        s = mol.intor("int1e_ovlp")
        if reference == "hf":
            c_all, e_all = mf.mo_coeff, mf.mo_energy
            self.e_hf, self.e_nbs = mf.e_tot, 0.0
        else:
            # theory.md 6.2 "Reference": HF Fock of the KS density,
            # occupied and virtual blocks diagonalized separately
            # (semicanonical); F_ov enters only the singles energy.
            dm = mf.make_rdm1()
            hf = scf.RHF(mol)
            f_hf = hf.get_fock(dm=dm)
            self.e_hf = float(hf.energy_tot(dm=dm))
            c_all = np.empty_like(mf.mo_coeff)
            e_all = np.empty(mf.mo_coeff.shape[1])
            for sl in (slice(0, nocc), slice(nocc, None)):
                c = mf.mo_coeff[:, sl]
                e, u = np.linalg.eigh(c.T @ f_hf @ c)
                c_all[:, sl], e_all[sl] = c @ u, e
            fov = c_all[:, self.ncore : nocc].T @ f_hf @ c_all[:, nocc:]
            d_ia = e_all[self.ncore : nocc, None] - e_all[None, nocc:]
            self.e_nbs = float(2.0 * np.sum(fov**2 / d_ia))
        # Zeroth-order Fock = the one the canonical reference uses:
        # F = S C diag(eps) C^T S (the oo + vv blocks of the HF Fock).
        # For HF, mf.get_fock() rebuilt from the final density differs from
        # it by the SCF residual (~1e-10 at conv_tol 1e-11), which would put
        # SCF noise into the eps = 0 anchor.
        fock = s @ (c_all * e_all) @ c_all.T @ s
        self.co = c_all[:, self.ncore : nocc]
        self.cv = c_all[:, nocc:]
        self.eo = e_all[self.ncore : nocc]
        self.ev = e_all[nocc:]
        # local orbitals
        from pyscf import lo

        self.co_loc = lo.Boys(mol, self.co).kernel()
        self.cv_loc, n_l, n_h = lm.build_vvhv(mol, mf, mf.mo_coeff[:, :nocc])
        self.n_vv, self.n_hv = n_l, n_h
        self.boys_stab = None
        self.lambda_boys = None
        self.orbitals = (
            "stable" if BOYS_STABLE["on"] and ORB["occ"] == "pyscf" else ORB["occ"]
        )
        if self.orbitals in ("stable", "lambda"):
            # Rotations inside the occupied span and inside the VV span leave
            # both spans (and so the hard virtuals) unchanged.
            self.co_loc, ro, bo, ao = boys_stabilize(mol, self.co_loc)
            cvv, rv, bv, av = boys_stabilize(mol, self.cv_loc[:, :n_l])
            self.cv_loc = np.hstack([cvv, self.cv_loc[:, n_l:]])
            self.boys_stab = {"occ": [ro, bo, ao], "vv": [rv, bv, av]}
            log(
                f"  boys-stable: occ {ro} rotations spread {bo:.6f}->{ao:.6f}; "
                f"VV {rv} rotations spread {bv:.6f}->{av:.6f}"
            )
        self.co_boys = self.co_loc
        if self.orbitals == "lambda":
            self.co_loc, self.lambda_boys = lambda_boys_localize(
                mol, self.co_loc, fock, ORB["lambda"]
            )
            lb = self.lambda_boys
            log(
                f"  lambda-Boys lam={lb['lambda']:g}: {lb['sweeps']} sweeps, grad "
                f"{lb['max_grad']:.1e}, G {lb.get('G_start', 0):.6f}->{lb.get('G_end', 0):.6f}, "
                f"off-diag F_oo^2 {lb.get('phi_boys', 0):.3e}->{lb.get('phi_lambda', 0):.3e}, "
                f"max||U|-1| {lb.get('max_abs_u_minus_1', 0):.3f}"
            )
        lm.check_construction(mol, mf, self.co_loc, self.cv_loc)
        self.u_occ = self.co.T @ s @ self.co_loc  # C_loc = C_can U
        self.u_vir = self.cv.T @ s @ self.cv_loc
        self.foo = self.co_loc.T @ fock @ self.co_loc
        self.fvv = self.cv_loc.T @ fock @ self.cv_loc
        self.no, self.nv = self.co.shape[1], self.cv.shape[1]
        self.df = DFFactors(mol, auxbasis) if auxbasis else None
        self.canonical_ladder = "dense exact"
        log(
            f"== {Path(xyz).name} {basis} ref={reference} stretch={stretch} "
            f"df={self.df.auxbasis if self.df else None} "
            f"nao={mol.nao} "
            f"no={self.no} nv={self.nv} (VV {n_l} + HV {n_h}) "
            f"E_HF[rho]={self.e_hf:.10f} E_NBS={self.e_nbs:.3e} "
            f"({time.time() - t0:.1f}s)"
        )


def canonical_w(co, cv, eo, ev, ints_fn, drop_term=None):
    """Canonical-orbital W set (the reference): c1 = -K/D, c2 = -R(c1)/D."""
    ints = ints_fn(co, cv)
    j = ints["ovov"]
    d = denom(np.diag(eo), np.diag(ev))
    c1 = -j / d
    c2 = -residual(c1, ints, drop_term) / d
    w = {}
    w[(2, "ss")], w[(2, "os")] = w_mats(c1, j)
    w[(3, "ss")], w[(3, "os")] = w_mats(c2, j)
    return w


def canonical(sys_, drop_term=None, dense_vvvv=None):
    """Canonical W set; exact integrals, or the DF factors of sys_.df (with
    the dense (ac|bd) = B.B ladder when it fits, an algebra independent of
    df_ladder; otherwise df_ladder with full domains)."""
    if sys_.df is None:
        fn = lambda co, cv: mo_ints(sys_.mol, co, cv)  # noqa: E731
    else:
        if dense_vvvv is None:
            dense_vvvv = 8 * sys_.nv**4 <= DENSE_VVVV_BYTES
        sys_.canonical_ladder = "dense B.B" if dense_vvvv else "df_ladder full"
        fn = lambda co, cv: sys_.df.ints(co, cv, dense_vvvv=dense_vvvv)  # noqa: E731
    return canonical_w(sys_.co, sys_.cv, sys_.eo, sys_.ev, fn, drop_term)


def local_ints(sys_):
    if sys_.df is None:
        return mo_ints(sys_.mol, sys_.co_loc, sys_.cv_loc)
    return sys_.df.ints(sys_.co_loc, sys_.cv_loc)


def wsg_mask(j, eps):
    """WSHG23 Eq. 8 integral mask, closed under a<->b and (ia)<->(jb)."""
    if eps == 0.0:
        return np.ones(j.shape, dtype=bool)
    return (np.abs(j) > eps) | (np.abs(swap_ab(j)) > eps)


MASK_ARM = {"arm": "eq8", "kappa": 1.0}  # set from --mask-arm / --kappa


def fock_mask(j, foo, fvv, eps, kappa=1.0):
    """Eq. 8 mask OR a one-step Fock-propagated screen (the LMP2 lane's
    second arm): keep (ia,jb) if |g| > eps or |g + F'.t0 + t0.F'| > kappa eps,
    with t0 = -g / D (diagonal denominators) and F' the off-diagonal virtual
    Fock matrix acting on a and on b. F' is the full off-diagonal block, not
    restricted to a pair's candidate set (a superset of the restricted screen).
    eps = 0 keeps everything, so the trivial-limit anchor is unchanged."""
    if eps == 0.0:
        return np.ones(j.shape, dtype=bool)
    t0 = -j / denom(foo, fvv)
    fp = fvv - np.diag(np.diag(fvv))
    g1 = (
        j
        + np.einsum("ac,icjb->iajb", fp, t0, optimize=True)
        + np.einsum("iajc,cb->iajb", t0, fp, optimize=True)
    )
    keep = (np.abs(j) > eps) | (np.abs(g1) > kappa * eps)
    return keep | swap_ab(keep)


def make_mask(j, foo, fvv, eps):
    if MASK_ARM["arm"] == "fock":
        return fock_mask(j, foo, fvv, eps, MASK_ARM["kappa"])
    return wsg_mask(j, eps)


def cost_proxies(mask):
    """Work of the masked ladder and rings, relative to canonical.

    Ladder: per ordered pair (ij) with retained set S_ij inside the domain
    block Da x Db, the dense-domain ladder costs (|Da||Db|)^2 (canonical
    o^2 v^4). Ring: X_ij,ab = sum_kc M_kc,jb t_ik,ac costs
    |Da_ij||Db_ij||Dc_ik| for every retained (ij), (ik) (canonical o^3 v^3).
    """
    no, nv = mask.shape[0], mask.shape[1]
    pair = mask.any(axis=(1, 3))
    na = mask.any(axis=3).sum(axis=1).astype(float)  # [i, j] = |Da_ij|
    nb = mask.any(axis=1).sum(axis=2).astype(float)  # [i, j] = |Db_ij|
    ladder = float(np.sum((na * nb) ** 2))
    ring = float(np.einsum("ij,ik->", na * nb * pair, nb * pair))
    return {
        "ladder_frac": ladder / (no**2 * nv**4),
        "ring_frac": ring / (no**3 * nv**3),
        "mean_domain": float(na[pair].mean()) if pair.any() else 0.0,
    }


def local_run(
    sys_, ints, eps, modes=("mask",), mutate=None, rtol=1e-12, check_lmp2=False
):
    """Local W sets at threshold eps, one per c2 mode; c1 is shared.

    mode 'mask': c2 on the same mask as c1 (the consistent choice);
    'full': c2 unmasked (diagnostic: isolates the first-order truncation).
    mutate: 'semicanonical' drops the Foo/Fvv off-diagonals (the eps = 0
    anchor must fail); 'c2_mask_shift' solves the 'mask' c2 on the mask of
    10*eps (the E3 identity must fail).
    """
    j = ints["ovov"]
    foo, fvv = sys_.foo, sys_.fvv
    if mutate == "semicanonical":
        foo, fvv = np.diag(np.diag(foo)), np.diag(np.diag(fvv))
    mask = make_mask(j, sys_.foo, sys_.fvv, eps)
    c1, it1, res1 = pcg(-j, foo, fvv, mask, rtol=rtol)
    n_r2_added = 0
    if MASK_ARM.get("r2") and eps > 0.0:
        # Second-order screen (v2 'new-r2' arm): the second-order amplitudes
        # solve A c2 = -R(c1), so screen R(c1) exactly as the first-order
        # screen treats J: keep (ia,jb) if |R| > eps or its one-step
        # Fock-propagated value exceeds kappa eps. R is evaluated unmasked
        # from the first-pass c1 (needs the dense ladder). c1 and c2 are then
        # BOTH solved on the union, so one mask holds for both orders and the
        # E3 identity still applies. eps = 0 keeps everything (anchor intact).
        if "vvvv" not in ints:
            raise ValueError("the second-order screen needs the dense ladder")
        r_est = residual(c1, ints, mask=None)
        mask2 = mask | fock_mask(r_est, sys_.foo, sys_.fvv, eps, MASK_ARM["kappa"])
        n_r2_added = int(mask2.sum() - mask.sum())
        mask = mask2
        c1, it1, res1 = pcg(-j, foo, fvv, mask, rtol=rtol)
    if "vvvv" not in ints and any(m == "full" for m in modes):
        raise ValueError("mode 'full' needs the dense ladder (DF R is masked)")
    r1 = residual(c1, ints, mask=None if eps == 0.0 else mask)
    w2 = w_mats(c1, j)
    e2_lmp2 = None
    if check_lmp2 and eps > 0 and sys_.no <= RAGGED_MAX_NO:
        # Independent finite-eps check of the first-order truncation: the
        # LMP2 prototype's ragged per-pair-domain solver (its independent
        # algebra path: block GEMMs and pair gathers, not dense einsums).
        # Its Python pair loops limit it to small occupied spaces.
        lm = _lmp2()
        t_l, _, _, _ = lm.solve_masked_mp2_ragged(j, foo, fvv, mask, rtol=1e-12)
        e2_lmp2 = float(lm.mp2_energy(t_l, j))
    # Wigner 2n+1: with c1 and c2 on one mask, E3 = c2 . K = c1 . R(c1)
    e3_r = energy(c1, r1)
    base = {
        "eps": eps,
        "keep": float(mask.mean()),
        "pair_keep": float(mask.any(axis=(1, 3)).mean()),
        "cg1": it1,
        "r2_added_frac": n_r2_added / mask.size,
        "e2": float(energy(c1, j)),
        "e2_lmp2_proto": e2_lmp2,
        **cost_proxies(mask),
    }
    out = {}
    for mode in modes:
        if mode == "full":
            mask2 = np.ones_like(mask)
        elif mutate == "c2_mask_shift":
            mask2 = wsg_mask(j, 10 * eps if eps > 0 else 1e-4)
        else:
            mask2 = mask
        c2, it2, res2 = pcg(-r1, foo, fvv, mask2, rtol=rtol)
        w = {(2, "ss"): w2[0], (2, "os"): w2[1]}
        w[(3, "ss")], w[(3, "os")] = w_mats(c2, j)
        # On one mask, gap = (2c1 - c1s).res2 - (2c2 - c2s).res1 with res the
        # CG residuals, so |gap| <= 3 (|c1| |res2| + |c2| |res1|).
        bound = 3.0 * (np.linalg.norm(c1) * res2 + np.linalg.norm(c2) * res1)
        info = dict(
            base,
            cg2=it2,
            e3_identity_gap=float(energy(c2, j) - e3_r),
            e3_identity_bound=float(bound),
        )
        out[mode] = (w, info)
    return out


def rotate_to_canonical(w, u):
    """Local-basis W -> canonical frame: W_can = U W_loc U^T."""
    return {k: u @ v @ u.T for k, v in w.items()}


def compare(sys_, w_can, w_loc):
    """Errors of every model, the Pade spectrum, and the linearized
    prediction of the OP-type errors from the measured W errors."""
    mc, ml = models(w_can), models(w_loc)
    err = {k: ml[k] - mc[k] for k in mc}
    wl = rotate_to_canonical(w_loc, sys_.u_occ)
    out = {"err": err, "can": mc}
    a2c = w_can[(2, "ss")] + w_can[(2, "os")]
    a3c = w_can[(3, "ss")] + w_can[(3, "os")]
    a2l = wl[(2, "ss")] + wl[(2, "os")]
    a3l = wl[(3, "ss")] + wl[(3, "os")]
    out["lam_can"], out["zeta_can"] = spectrum(a2c, a3c)
    out["lam_loc"], out["zeta_loc"] = spectrum(a2l, a3l)
    pred = {}
    for name, (ac, bc, al, bl) in {
        "op-mp3": (a2c, a3c, a2l, a3l),
        "os": (w_can[(2, "os")], w_can[(3, "os")], wl[(2, "os")], wl[(3, "os")]),
        "ss": (w_can[(2, "ss")], w_can[(3, "ss")], wl[(2, "ss")], wl[(3, "ss")]),
    }.items():
        ga, gb = pade_grads(ac, bc)
        pred[name] = 2 * (np.vdot(ga, al - ac) + np.vdot(gb, bl - bc))
    pred["sop-mp3"] = pred["os"] + pred["ss"]
    out["pred"] = {k: float(v) for k, v in pred.items()}
    return out


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------

ANCHOR_BAR = 1e-10
# Dense (ac|bd) is formed only below this size (canonical DF reference).
DENSE_VVVV_BYTES = 2.0e9
# DF domain ladder vs dense B.B ladder on the mask: the same sums in another
# order, so rounding only, ~ sqrt(n) eps_mach |R| ~ 3e-12 |R| for the
# n ~ naux nv^2 ~ 1e9 terms here. Bar 1e-11 |R|; an index-set mix-up moves
# R by O(|R|).
DF_DOMAIN_BAR = 1e-11
# Largest active occupied space for the ragged LMP2 cross-check.
RAGGED_MAX_NO = 30
# Both solves stop at relative residual 1e-12; the energy t.J is linear in
# the amplitude error, |dE| <= 3 |J| |dt| <= 3 |J| kappa 1e-12 |t|, a few
# 1e-12 Ha here. 1e-10 is the ragged prototype's own cross-check bar.
LMP2_BAR = 1e-10


def anchor_eps0(sys_, ints, w_can, mutate=None):
    """eps = 0: local W rotated to canonical == canonical W elementwise;
    eigenvalues of W^(2), W^(3) and every model energy within ANCHOR_BAR.
    Returns (passed, max deviation)."""
    w_loc, _ = local_run(sys_, ints, 0.0, mutate=mutate, rtol=1e-14)["mask"]
    wl = rotate_to_canonical(w_loc, sys_.u_occ)
    dw = max(float(np.abs(wl[k] - w_can[k]).max()) for k in w_can)
    de = max(abs(v) for v in compare(sys_, w_can, w_loc)["err"].values())
    dev = {}
    for n in (2, 3):
        a = w_can[(n, "ss")] + w_can[(n, "os")]
        b = w_loc[(n, "ss")] + w_loc[(n, "os")]
        dev[n] = float(np.abs(np.linalg.eigvalsh(a) - np.linalg.eigvalsh(b)).max())
    worst = max(dw, de, dev[2], dev[3])
    log(
        f"  ANCHOR eps=0: max|W_loc->can - W_can|={dw:.2e} "
        f"max|dE(models)|={de:.2e} eig dev W2={dev[2]:.2e} W3={dev[3]:.2e} "
        f"-> {'PASS' if worst <= ANCHOR_BAR else 'FAIL'} (bar {ANCHOR_BAR:g})"
    )
    return worst <= ANCHOR_BAR, worst


def anchor_invariance(w_can, seed=0):
    """Tr P(U A U^T, U B U^T) == Tr P(A, B) for a random orthogonal U,
    for every channel the models use. Returns the max deviation."""
    rng = np.random.default_rng(seed)
    n = w_can[(2, "ss")].shape[0]
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    rot = {k: q @ v @ q.T for k, v in w_can.items()}
    m0, m1 = models(w_can), models(rot)
    return max(abs(m0[k] - m1[k]) for k in m0)


def ferric_cross_check(sys_, drop_term=None):
    """This file's canonical formulas vs ferric run_opmp3, on ONE set of
    orbitals and integrals: ferric run_rhf orbitals + ferric
    AoIntegrals.ao2mo feed canonical_w; run_opmp3 runs the same SCF
    settings. Separates the formula check from the PySCF-vs-ferric SCF
    difference (reported as dE_HF). W is compared by eigenvalues (W is
    covariant under the degenerate-orbital freedom of the two runs)."""
    import ferric

    mol = ferric.Molecule.from_xyz_string(sys_.atom_xyz_block())
    bs = ferric.BasisSet.bundled(sys_.basis)
    kw = dict(scf_energy_conv=1e-12, scf_density_conv=1e-10, scf_max_iter=300)
    res = ferric.run_opmp3(mol, bs, frozen_core=sys_.ncore, **kw)
    rhf = ferric.run_rhf(mol, bs, energy_conv=1e-12, density_conv=1e-10, max_iter=300)
    aoi = ferric.AoIntegrals(mol, bs)
    c, e = rhf.mo_coefficients(), rhf.orbital_energies()
    nocc = sys_.nocc
    nc = sys_.ncore
    w_f = canonical_w(
        c[:, nc:nocc],
        c[:, nocc:],
        e[nc:nocc],
        e[nocc:],
        lambda co, cv: mo_ints(None, co, cv, g=lambda *cs: aoi.ao2mo(*cs, max_gb=4.0)),
        drop_term,
    )
    mc = models(w_f)
    mc["mp2"] = mc["e2"]
    dev_e = {
        k: res.correlation[k] - mc[k]
        for k in ("mp2", "mp3", "p-mp3", "op-mp3", "sop-mp3", "ssd-opmp3")
    }
    dev_w = {}
    for n in (2, 3):
        wf = res.w(n)
        for ch in ("ss", "os"):
            a = np.linalg.eigvalsh(wf[f"alpha_{ch}"])
            b = np.linalg.eigvalsh(w_f[(n, ch)])
            dev_w[f"W{n}{ch}"] = float(np.abs(a - b).max())
    de_hf = res.e_hf - sys_.e_hf
    de_pyscf = models(canonical(sys_))["op-mp3"] - mc["op-mp3"]
    return dev_e, dev_w, de_hf, de_pyscf, ferric.__build__


def ferric_dft_reference_check(sys_, w_can):
    """KS reference: PySCF canonical vs ferric run_opmp3(reference=<xc>).
    The two codes use different DFT grids, so the gap is a grid-level
    agreement (reported, not gated); the formula identity is the HF check."""
    import ferric

    mol = ferric.Molecule.from_xyz_string(sys_.atom_xyz_block())
    res = ferric.run_opmp3(
        mol,
        ferric.BasisSet.bundled(sys_.basis),
        reference=sys_.reference,
        x_ss=0.5,
        frozen_core=sys_.ncore,
        scf_energy_conv=1e-12,
        scf_density_conv=1e-10,
        scf_max_iter=300,
    )
    mc = models(w_can)
    out = {
        "dE_hf": res.e_hf - sys_.e_hf,
        "dE_nbs": res.e_nbs - sys_.e_nbs,
        "d_mp2": res.correlation["mp2"] - mc["e2"],
        "d_mp3": res.correlation["mp3"] - mc["mp3"],
        "d_op-mp3": res.correlation["op-mp3"] - mc["op-mp3"],
        "d_ssd-opmp3(0.5)": res.correlation["ssd-opmp3"] - mc["ssd-opmp3(0.5)"],
        "build": ferric.__build__,
    }
    return out


def _xyz_block(self):
    lines = [ln for ln in self.atom.splitlines() if ln.strip()]
    return f"{len(lines)}\n\n" + "\n".join(lines) + "\n"


System.atom_xyz_block = _xyz_block


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


# Named map configurations (--maps / --arms). 'old' is what the 2026-11-16/17
# tables ran (Eq-8 mask, pyscf.lo.Boys as returned); 'old-stable' is ferric's
# pre-fix production analogue (Eq-8, converged Boys); 'new' is ferric's
# integral-direct default since c29c023f1 + e0b5ff1f3 (screen kappa = 1,
# lambda-Boys lambda = 10). 'screen' and 'lambda' attribute the gain.
MAPS = {
    "old": {"arm": "eq8", "kappa": 1.0, "occ": "pyscf", "lambda": 0.0},
    "old-stable": {"arm": "eq8", "kappa": 1.0, "occ": "stable", "lambda": 0.0},
    "screen": {"arm": "fock", "kappa": 1.0, "occ": "stable", "lambda": 0.0},
    "lambda": {"arm": "eq8", "kappa": 1.0, "occ": "lambda", "lambda": 10.0},
    "new": {"arm": "fock", "kappa": 1.0, "occ": "lambda", "lambda": 10.0},
    # new + the second-order screen on R(c1) (not in ferric; v2 candidate)
    "new-r2": {
        "arm": "fock",
        "kappa": 1.0,
        "occ": "lambda",
        "lambda": 10.0,
        "r2": True,
    },
}


def set_maps(name):
    m = MAPS[name]
    MASK_ARM.update(arm=m["arm"], kappa=m["kappa"], r2=m.get("r2", False))
    ORB.update(occ=m["occ"], **{"lambda": m["lambda"]})
    BOYS_STABLE["on"] = False


CANONICAL_CACHE = {}  # (xyz, basis, stretch, reference, df) -> (e_hf, w_can)


def run(args):
    if args.copies > 1:
        return size_consistency(args)
    if args.rust_lmp2:
        from pyscf import gto
        from pyscf.data import elements

        lm = _lmp2()
        mol = gto.M(atom=lm.load_xyz(args.xyz), basis=args.basis, verbose=0)
        return rust_lmp2_series(
            args.xyz,
            args.basis,
            args.rust_lmp2,
            args.eps,
            elements.chemcore(mol),
            maps=args.rust_maps,
        )
    sys_ = System(
        args.xyz,
        args.basis,
        stretch=args.stretch,
        reference=args.reference,
        auxbasis=args.df,
    )
    if args.plateau is not None:
        ints = local_ints(sys_)
        out = plateau_diagnostic(sys_, ints, args.plateau)
        log("  PLATEAU " + json.dumps(out, indent=1))
        return {"xyz": Path(args.xyz).name, "basis": args.basis, "plateau": out}
    t0 = time.time()
    key = (args.xyz, args.basis, args.stretch, args.reference, args.df)
    if key in CANONICAL_CACHE:
        e_hf0, w_can = CANONICAL_CACHE[key]
        if abs(e_hf0 - sys_.e_hf) > 1e-10:
            raise SystemExit(
                f"cached canonical W from another SCF ({e_hf0 - sys_.e_hf:.1e})"
            )
        if sys_.df is not None:
            sys_.canonical_ladder = (
                "dense B.B" if 8 * sys_.nv**4 <= DENSE_VVVV_BYTES else "df_ladder full"
            )
    else:
        w_can = canonical(sys_)
        CANONICAL_CACHE[key] = (sys_.e_hf, w_can)
    mc = models(w_can)
    a2 = w_can[(2, "ss")] + w_can[(2, "os")]
    a3 = w_can[(3, "ss")] + w_can[(3, "os")]
    lam, zeta = spectrum(a2, a3)
    log(
        "  canonical: "
        + " ".join(f"{k}={v:.10f}" for k, v in mc.items())
        + f"  lam(A-B)~0={lam:.4e} zeta_max={zeta:.4f} ({time.time() - t0:.1f}s)"
    )
    inv = anchor_invariance(w_can)
    log(f"  Tr P invariance under a random occupied rotation: max dev {inv:.2e}")
    result = {
        "xyz": Path(args.xyz).name,
        "basis": args.basis,
        "stretch": args.stretch,
        "no": sys_.no,
        "nv": sys_.nv,
        "ncore": sys_.ncore,
        "reference": sys_.reference,
        "e_hf": sys_.e_hf,
        "e_nbs": sys_.e_nbs,
        "canonical": mc,
        "lam_can": lam,
        "zeta_can": zeta,
        "invariance_dev": inv,
        "boys": "pyscf-default" if sys_.orbitals == "pyscf" else "stable",
        "orbitals": sys_.orbitals,
        "lambda_boys": sys_.lambda_boys,
        "boys_stab": sys_.boys_stab,
        "orbital_spread": orbital_spreads(sys_),
        "rows": [],
    }
    if args.ferric and sys_.reference != "hf":
        chk = ferric_dft_reference_check(sys_, w_can)
        log(
            "  ferric (DFT grids differ): "
            + " ".join(f"{k}={v:.2e}" for k, v in chk.items() if k != "build")
        )
        result["ferric_dft"] = chk
    elif args.ferric:
        dev_e, dev_w, de_hf, de_py, build = ferric_cross_check(
            sys_, drop_term=args.drop_term
        )
        log(
            f"  ferric {build}: E_HF(ferric)-E_HF(pyscf)={de_hf:.2e}, "
            f"OP-MP3 on pyscf vs ferric orbitals {de_py:.2e} (SCF difference)"
        )
        log("    model dev: " + " ".join(f"{k}={v:.2e}" for k, v in dev_e.items()))
        log("    W eig dev: " + " ".join(f"{k}={v:.2e}" for k, v in dev_w.items()))
        result["ferric"] = {
            "dE": dev_e,
            "dW": dev_w,
            "dE_hf": de_hf,
            "dOP_pyscf_vs_ferric_orbitals": de_py,
            "build": build,
        }
    ints = local_ints(sys_)
    result["df"] = sys_.df.auxbasis if sys_.df else None
    result["canonical_ladder"] = sys_.canonical_ladder
    if sys_.df is not None and args.df_check is not None:
        dr, rmax, keep = df_domain_check(sys_, ints, args.df_check)
        ok_dom = dr <= DF_DOMAIN_BAR * rmax
        log(
            f"  DF domain check eps={args.df_check:g} keep={keep:.3f}: "
            f"max|R_dom - R_dense| on mask = {dr:.2e} (|R|max {rmax:.2e}) -> "
            f"{'PASS' if ok_dom else 'FAIL'}"
        )
        dm, _, _ = df_domain_check(sys_, ints, args.df_check, mutate="pair_domain_ii")
        log(
            f"  MUTATION pair_domain_ii: max|dR| = {dm:.2e} -> "
            + ("FAILED as required" if dm > DF_DOMAIN_BAR * rmax else "STILL PASSES")
        )
        result["df_domain_check"] = {"dR": dr, "Rmax": rmax, "mutation_dR": dm}
        if not ok_dom:
            raise SystemExit("DF domain check failed")
    exact = None
    if sys_.df is not None and args.compare_exact:
        sys_x = System(
            args.xyz, args.basis, stretch=args.stretch, reference=args.reference
        )
        w_x = canonical(sys_x)
        mx = models(w_x)
        ri = {k: mc[k] - mx[k] for k in mc}
        log(
            "  RI error (DF canonical - exact canonical): "
            + " ".join(f"{k}={v:+.2e}" for k, v in ri.items())
        )
        result["ri_error"] = ri
        exact = (sys_x, local_ints(sys_x), w_x)
    if args.mutate == "c2_mask_shift":
        _, info = local_run(sys_, ints, 1e-4, mutate=args.mutate)["mask"]
        gap, bound = info["e3_identity_gap"], info["e3_identity_bound"]
        log(
            f"  MUTATION c2_mask_shift: E3 identity gap {gap:+.2e} vs bound "
            f"{bound:.1e}: "
            + ("FAILED as required" if abs(gap) > bound else "STILL PASSES")
        )
        result["mutation_identity_gap"] = gap
        return result
    ok, worst = anchor_eps0(sys_, ints, w_can, mutate=args.mutate)
    result["anchor_eps0"] = worst
    if args.mutate:
        log(
            f"  MUTATION {args.mutate}: anchor "
            + ("FAILED as required" if not ok else "STILL PASSES (mutation broken)")
        )
        return result
    if not ok:
        raise SystemExit("eps=0 anchor failed; no sweep run")
    if args.anchor_only:
        return result
    ec = mc["e2"]
    hdr = (
        f"  {'eps':>7} {'mode':>4} {'keep':>6} {'pairs':>5} "
        f"{'dE2':>10} {'dE3':>10} {'dMP3':>10} {'dOP':>10} {'dSOP':>10} "
        f"{'dSSD':>10} {'OP/MP3':>7} {'pred/OP':>7} {'lam_loc':>10} {'E3gap':>8} "
        f"{'ladder':>7} {'ring':>7}"
    )
    log(hdr)
    for eps in args.eps:
        t1 = time.time()
        runs = local_run(
            sys_, ints, eps, modes=args.modes, check_lmp2=not args.no_lmp2_check
        )
        if exact is not None:
            sx, ix, wx = exact
            wlx, _ = local_run(sx, ix, eps)["mask"]
            ex_err = compare(sx, wx, wlx)["err"]
            df_err = compare(sys_, w_can, runs["mask"][0])["err"]
            diff = {k: df_err[k] - ex_err[k] for k in ("e2", "e3", "mp3", "op-mp3")}
            log(
                f"  eps={eps:g}: truncation error DF route - exact route: "
                + " ".join(f"{k}={v:+.1e}" for k, v in diff.items())
                + f" (exact-route dOP {ex_err['op-mp3']:+.2e})"
            )
            result.setdefault("trunc_err_df_minus_exact", []).append(
                {"eps": eps, **diff, "exact_dOP": ex_err["op-mp3"]}
            )
        base = runs[args.modes[0]][1]
        if base["e2_lmp2_proto"] is not None:
            d_l = base["e2"] - base["e2_lmp2_proto"]
            log(f"  eps={eps:g}: E2 vs LMP2 prototype ragged solver {d_l:+.1e}")
            if abs(d_l) > LMP2_BAR:
                raise SystemExit("finite-eps E2 disagrees with the LMP2 prototype")
        for mode in args.modes:
            w_loc, info = runs[mode]
            if (
                mode == "mask"
                and abs(info["e3_identity_gap"]) > info["e3_identity_bound"]
            ):
                raise SystemExit(
                    f"E3 identity violated: gap {info['e3_identity_gap']:.2e} > "
                    f"bound {info['e3_identity_bound']:.2e}"
                )
            cmp_ = compare(sys_, w_can, w_loc)
            e = cmp_["err"]
            amp = e["op-mp3"] / e["mp3"] if e["mp3"] != 0 else float("nan")
            pr = cmp_["pred"]["op-mp3"] / e["op-mp3"] if e["op-mp3"] else float("nan")
            log(
                f"  {eps:7.0e} {mode:>4} {info['keep']:6.4f} {info['pair_keep']:5.3f} "
                f"{e['e2']:+10.3e} {e['e3']:+10.3e} {e['mp3']:+10.3e} "
                f"{e['op-mp3']:+10.3e} {e['sop-mp3']:+10.3e} {e['ssd-opmp3']:+10.3e} "
                f"{amp:7.3f} {pr:7.3f} {cmp_['lam_loc']:+10.3e} "
                f"{info['e3_identity_gap']:+8.1e} {info['ladder_frac']:7.4f} "
                f"{info['ring_frac']:7.4f} ({time.time() - t1:.0f}s)"
            )
            result["rows"].append(
                {
                    "mode": mode,
                    **info,
                    "err": e,
                    "rel_err_vs_ecorr2": {k: v / ec for k, v in e.items()},
                    "amp_op_over_mp3": amp,
                    "pred": cmp_["pred"],
                    "lam_loc": cmp_["lam_loc"],
                    "zeta_loc": cmp_["zeta_loc"],
                }
            )
    return result


def local_models(sys_, ints, eps):
    w_loc, info = local_run(sys_, ints, eps)["mask"]
    return models(w_loc), info


def size_consistency(args):
    """E_loc(n copies far apart) - n E_loc(one copy), every model, each eps
    (eps = 0 included: it tests the localization, not the mask)."""
    mono = System(
        args.xyz,
        args.basis,
        stretch=args.stretch,
        reference=args.reference,
        auxbasis=args.df,
    )
    sup = System(
        args.xyz,
        args.basis,
        stretch=args.stretch,
        copies=args.copies,
        sep=args.sep,
        reference=args.reference,
        auxbasis=args.df,
    )
    im, isup = local_ints(mono), local_ints(sup)
    rows = []
    keys = ("e2", "e3", "mp3", "op-mp3", "sop-mp3", "ssd-opmp3", "ssd-opmp3(0.5)")
    for eps in [0.0] + args.eps:
        mm, _ = local_models(mono, im, eps)
        ms, info = local_models(sup, isup, eps)
        gap = {k: ms[k] - args.copies * mm[k] for k in keys}
        log(
            f"  eps={eps:7.0e} keep(sup)={info['keep']:.4f} pairs(sup)="
            f"{info['pair_keep']:.4f}  gap: "
            + " ".join(f"{k}={v:+.1e}" for k, v in gap.items())
            + f"  (E_op(mono)={mm['op-mp3']:.10f})"
        )
        rows.append(
            {
                "eps": eps,
                "gap": gap,
                "keep_sup": info["keep"],
                "pair_keep_sup": info["pair_keep"],
                "mono": mm,
            }
        )
    diag = separability_diag(mono, sup, im, isup, args.eps)
    log(f"  separability: {json.dumps(diag)}")
    return {
        "xyz": Path(args.xyz).name,
        "basis": args.basis,
        "copies": args.copies,
        "sep_bohr": args.sep,
        "size_consistency": rows,
        "separability": diag,
    }


def separability_diag(mono, sup, im, isup, eps_list):
    """Is the supersystem's local basis the direct sum of copies of the
    monomer's? (Independent of the energies: it compares orbitals.) For copy 0
    (the first nao AOs of the supersystem) match each supersystem local orbital
    to a monomer local orbital by |overlap|; then compare the copy-0 block of
    the supersystem ovov integrals with the monomer's, mapped through the
    match, and count mask entries that differ at each eps.
    Artifact hypothesis: a non-separable construction (orbitals spread over
    both copies, or a copy's virtual space unlike the monomer's) gives
    overlaps well below 1 and many mask differences at every eps.
    Threshold noise instead gives overlaps ~1, |dJ| at the SCF-noise level and
    mask differences only for entries with |J| within |dJ| of eps."""
    nao = mono.mol.nao
    s = mono.mol.intor("int1e_ovlp")
    out = {}

    def match(cm, cd):
        w0 = np.einsum("pk,pq,qk->k", cd[:nao], s, cd[:nao])
        on0 = np.flatnonzero(w0 > 0.5)
        ov = cm.T @ s @ cd[:nao, on0]
        best = np.abs(ov).argmax(axis=0)
        sign = np.sign(ov[best, np.arange(on0.size)])
        return (
            {
                "copy_weight_max_dev": float(np.minimum(w0, 1 - w0).max()),
                "n_on_copy0": int(on0.size),
                "n_mono": int(cm.shape[1]),
                "overlap_min_of_max": float(np.abs(ov).max(axis=0).min()),
                "injective": bool(np.unique(best).size == best.size),
            },
            on0,
            best,
            sign,
        )

    mo, io, bo, so = match(mono.co_loc, sup.co_loc)
    mv, iv, bv, sv = match(mono.cv_loc, sup.cv_loc)
    out["occ"], out["vir"] = mo, mv
    # Boys spread of the same space localized two ways: the monomer's local
    # virtuals and the supersystem's copy-0 virtuals (same AOs, same origin).
    # Equal subspaces, so a lower total spread marks the better Boys optimum;
    # a higher one marks a stationary point that is not the minimum.
    r1 = mono.mol.intor("int1e_r")
    r2 = mono.mol.intor("int1e_r2")

    def spread(c):
        m1 = np.einsum("pk,xpq,qk->xk", c, r1, c)
        return float((np.einsum("pk,pq,qk->k", c, r2, c) - (m1**2).sum(0)).sum())

    for tag, cm, cd, on0 in (
        ("occ", mono.co_loc, sup.co_loc, io),
        ("vir", mono.cv_loc, sup.cv_loc, iv),
    ):
        c0 = cd[:nao, on0]
        out[tag]["boys_spread_mono"] = spread(cm)
        out[tag]["boys_spread_sup_copy0"] = spread(c0)
        out[tag]["subspace_dev"] = float(np.abs(cm @ cm.T @ s @ c0 - c0).max())

    def pair_spread(c2, th):
        cs, sn = np.cos(th), np.sin(th)
        return spread(
            np.column_stack(
                [cs * c2[:, 0] + sn * c2[:, 1], -sn * c2[:, 0] + cs * c2[:, 1]]
            )
        )

    # A vs B: every copy-0 virtual whose best monomer overlap is < 0.99 is
    # decomposed onto its two largest monomer partners; scan the Boys spread
    # of that monomer pair along their mutual rotation angle.
    from pyscf import lo

    cmv = mono.cv_loc
    ov = cmv.T @ s @ sup.cv_loc[:nao, iv]
    pairs, h = [], 1e-3
    for k in np.flatnonzero(np.abs(ov).max(axis=0) < 0.99):
        top = np.argsort(-np.abs(ov[:, k]))[:2]
        key = tuple(sorted(int(t) for t in top))
        if any(p["mono_pair"] == list(key) for p in pairs):
            continue
        c2 = cmv[:, list(key)]
        f0, fp, fm = pair_spread(c2, 0.0), pair_spread(c2, h), pair_spread(c2, -h)
        m1 = np.einsum("pk,xpq,qk->kx", c2, r1, c2)
        pairs.append(
            {
                "mono_pair": list(key),
                "coverage": float((ov[list(key), k] ** 2).sum()),
                "hessian_along_rotation": float((fp - 2 * f0 + fm) / h**2),
                "spread_change_at_45deg": float(pair_spread(c2, np.pi / 4) - f0),
                "individual_spreads": [spread(c2[:, :1]), spread(c2[:, 1:])],
                "centroids_bohr": m1.tolist(),
            }
        )
    out["mismatched_pairs"] = pairs
    # monomer reproducibility: Boys from random rotations of its own local
    # valence virtuals (the Boys-localized block of the VV-HV construction)
    cvv = cmv[:, : mono.n_vv]
    rng = np.random.default_rng(7)
    starts = []
    for _ in range(2):
        q, _r = np.linalg.qr(rng.standard_normal((cvv.shape[1], cvv.shape[1])))
        mlo = lo.Boys(mono.mol, cvv @ q)
        mlo.init_guess = None
        c = mlo.kernel()
        o = np.abs(cvv.T @ s @ c)
        starts.append(
            {
                "spread_vv": spread(c),
                "spread_vv_mono": spread(cvv),
                "overlap_min_of_max_vs_mono": float(o.max(axis=0).min()),
            }
        )
    out["vir"]["random_start_boys"] = starts
    if not (
        mo["injective"]
        and mv["injective"]
        and mo["n_on_copy0"] == mo["n_mono"]
        and mv["n_on_copy0"] == mv["n_mono"]
    ):
        out["mapped"] = False
        return out
    jm = im["ovov"]
    js = isup["ovov"][np.ix_(io, iv, io, iv)]
    # put the supersystem block in monomer order with monomer signs
    po, pv = np.argsort(bo), np.argsort(bv)
    js = js[np.ix_(po, pv, po, pv)]
    sgn = so[po][:, None, None, None] * sv[pv][None, :, None, None]
    js = js * sgn * so[po][None, None, :, None] * sv[pv][None, None, None, :]
    dj = np.abs(js - jm)
    out["mapped"] = True
    out["max_abs_dJ"] = float(dj.max())
    out["masks"] = []
    for eps in eps_list:
        a, b = wsg_mask(jm, eps), wsg_mask(js, eps)
        diff = a != b
        near = np.abs(np.abs(jm) - eps) <= dj.max()
        out["masks"].append(
            {
                "eps": eps,
                "n_diff": int(diff.sum()),
                "n_kept_mono": int(a.sum()),
                "n_within_dJ_of_eps": int(near.sum()),
            }
        )
    return out


def rust_boys_probe(xyz, sep=100.0):
    """Read-only probe of the PRODUCTION VV Boys (crates/ferric-mp2/src/boys.rs
    via ferric.boys_localize: Jacobi pair sweeps, max |theta| < 1e-8 per sweep,
    200 sweeps) on monomer vs dimer, STO-3G only. In STO-3G the projected
    minimal basis is the whole basis (T = S^-1 S_x = I), so production's VV
    start is canonical_orth((1 - C_occ C_occ^T S), S, nao - nocc), built here
    from ferric's own SCF, overlap and dipoles (ferric AO order throughout).
    Compares the Boys functional sum_k |<k|r|k>|^2 of the monomer's VVs with
    the dimer's copy-0 VVs (same subspace, origin-independent difference) and
    their orbital overlaps."""
    import tempfile

    import ferric

    lm = _lmp2()
    atom = lm.load_xyz(xyz)
    bs = ferric.BasisSet.bundled("sto-3g")
    out = {}
    with tempfile.TemporaryDirectory() as td:
        sets = {}
        for tag, a in (("mono", atom), ("dimer", replicate(atom, 2, sep))):
            n = len(a.splitlines())
            path = Path(td) / f"{tag}.xyz"
            path.write_text(f"{n}\n{tag}\n{a}\n", encoding="utf-8")
            mol = ferric.Molecule.from_xyz(str(path))
            r = ferric.run_rhf(mol, bs, energy_conv=1e-11, density_conv=1e-9)
            ao = ferric.AoIntegrals(mol, bs)
            s = ao.overlap()
            c = r.mo_coefficients()
            nocc = ao.nelec // 2
            co = c[:, :nocc]
            q = np.eye(s.shape[0]) - co @ co.T @ s
            cl = lm.canonical_orth(q, s, s.shape[0] - nocc)
            dip = np.asarray(ao.dipole((0.0, 0.0, 0.0)))
            b = ferric.boys_localize(mol, bs, cl, 200)
            bo = ferric.boys_localize(mol, bs, co, 200)
            sets[tag] = (b.c_loc(), s, dip)
            sets[tag + "_occ"] = (bo.c_loc(), s, dip)
            out[f"{tag}_converged"] = bool(b.converged)
            out[f"{tag}_sweeps"] = int(b.iterations)
            out[f"{tag}_occ_converged"] = bool(bo.converged)
            out[f"{tag}_occ_sweeps"] = int(bo.iterations)
            if tag == "mono":
                # reproducibility: production Boys from random rotations
                rng = np.random.default_rng(11)
                for space, c0_ in (("vv", cl), ("occ", co)):
                    fs = []
                    for _ in range(3):
                        qr, _r = np.linalg.qr(rng.standard_normal((c0_.shape[1],) * 2))
                        cr = ferric.boys_localize(mol, bs, c0_ @ qr, 200).c_loc()
                        fs.append(
                            float((np.einsum("pk,xpq,qk->xk", cr, dip, cr) ** 2).sum())
                        )
                    out[f"mono_{space}_random_start_F"] = fs

    def boys_f(c, dip):
        return float((np.einsum("pk,xpq,qk->xk", c, dip, c) ** 2).sum())

    for space, (km, kd) in (
        ("vv", ("mono", "dimer")),
        ("occ", ("mono_occ", "dimer_occ")),
    ):
        cm, sm, dm = sets[km]
        cd, _sd, _dd = sets[kd]
        nao = cm.shape[0]
        w0 = np.einsum("pk,pq,qk->k", cd[:nao], sm, cd[:nao])
        c0 = cd[:nao, w0 > 0.5]
        ov = np.abs(cm.T @ sm @ c0)
        out[space] = dict(
            n_mono=int(cm.shape[1]),
            n_copy0=int(c0.shape[1]),
            copy_weight_max_dev=float(np.minimum(w0, 1 - w0).max()),
            subspace_dev=float(np.abs(cm @ cm.T @ sm @ c0 - c0).max()),
            boys_F_mono=boys_f(cm, dm),
            boys_F_copy0=boys_f(c0, dm),
            overlap_min_of_max=float(ov.max(axis=0).min()),
            injective=bool(np.unique(ov.argmax(axis=0)).size == c0.shape[1]),
        )
    log(f"  RUST BOYS PROBE {json.dumps(out)}")
    return {"xyz": Path(xyz).name, "basis": "sto-3g", "rust_boys_probe": out}


def df_domain_check(sys_, ints, eps, mutate=None):
    """Finite-eps check of df_ladder's domain contraction: on the mask,
    R from the per-pair DF ladder equals R from the dense (ac|bd) = B.B
    ladder applied to the same masked c1 (an independent algebra). The
    vacuous-mask anchor cannot see a domain mix-up; this can.
    Returns (max |dR| on the mask, max |R| on the mask)."""
    nax, nv = ints["bvv"].shape[0], sys_.nv
    if 8 * nv**4 > DENSE_VVVV_BYTES:
        raise ValueError("dense B.B ladder does not fit; pick a smaller system")
    j = ints["ovov"]
    mask = wsg_mask(j, eps)
    c1, _, _ = pcg(-j, sys_.foo, sys_.fvv, mask)
    bv = ints["bvv"].reshape(nax, -1)
    dense = dict(ints, vvvv=(bv.T @ bv).reshape(nv, nv, nv, nv))
    r_dense = residual(c1, dense)
    r_dom = residual(c1, dict(ints, ladder_mutate=mutate), mask=mask)
    dr = float(np.abs(np.where(mask, r_dom - r_dense, 0.0)).max())
    return dr, float(np.abs(np.where(mask, r_dense, 0.0)).max()), float(mask.mean())


def plateau_diagnostic(sys_, ints, eps):
    """Why does the first-order error stop falling at tight eps?

    Hypothesis (H-Fock): the mask selects by INTEGRAL size, but in the
    non-canonical local basis an amplitude is also driven through the
    off-diagonal Fock couplings, t = -A^-1 J; entries with |J| < eps can
    carry amplitudes fed by retained neighbours. Tests:
    1. split of the exact (eps = 0) local amplitudes outside the mask into
       the part a diagonal (semicanonical) estimate -J/D predicts and the
       rest;
    2. add the dropped entries with the largest one-step Fock-propagated
       amplitude estimate (top 1/5/20% of the dropped set) and re-solve;
    3. control: add the same number of dropped entries at random.
    (A first version widened by one whole Fock hop; at tau = 1e-2 that kept
    99.9% of entries, so its equal-size random control was unreachable.)
    """
    j = ints["ovov"]
    foo, fvv = sys_.foo, sys_.fvv
    full = np.ones(j.shape, dtype=bool)
    t_ex, _, _ = pcg(-j, foo, fvv, full)
    e_ex = energy(t_ex, j)
    mask = wsg_mask(j, eps)
    t_m, _, _ = pcg(-j, foo, fvv, mask)
    err0 = energy(t_m, j) - e_ex
    out = {"eps": eps, "keep": float(mask.mean()), "err_masked": float(err0)}
    d = denom(foo, fvv)
    off = ~mask
    out["norm_t_ex_dropped"] = float(np.linalg.norm(t_ex[off]))
    out["norm_diag_estimate_dropped"] = float(np.linalg.norm((j / d)[off]))
    out["energy_on_dropped_exact_t"] = float(
        2.0 * np.vdot(np.where(off, t_ex, 0.0), j)
        - np.vdot(swap_ab(np.where(off, t_ex, 0.0)), j)
    )
    nvl = sys_.cv_loc.shape[1]
    hv = slice(sys_.n_vv, nvl)
    fv_off = fvv - np.diag(np.diag(fvv))
    out["max_fvv_offdiag_vv"] = float(np.abs(fv_off[: sys_.n_vv, : sys_.n_vv]).max())
    out["max_fvv_offdiag_hv"] = (
        float(np.abs(fv_off[hv, hv]).max()) if sys_.n_hv else 0.0
    )
    out["max_fvv_vv_hv"] = (
        float(np.abs(fv_off[: sys_.n_vv, hv]).max()) if sys_.n_hv else 0.0
    )
    # Discriminating test: rank the dropped entries by a one-step
    # Fock-propagated estimate t_est = -(J + A_off t_M) / D (A_off = the
    # off-diagonal Fock part of A), add the top fraction f to the mask, and
    # compare with adding the same number of dropped entries at random. If
    # H-Fock holds, the ranked entries remove far more error than random.
    a_off = apply_a(t_m, foo, fvv) - d * t_m
    t_est = np.where(off, -(j + a_off) / d, 0.0)
    drop_idx = np.flatnonzero(off)
    order = drop_idx[np.argsort(-np.abs(t_est.reshape(-1)[drop_idx]))]
    rng = np.random.default_rng(0)

    def close(m):
        m = m | swap_ab(m)
        return m | m.transpose(2, 3, 0, 1)

    out["fock_ranked"] = []
    for frac in (0.01, 0.05, 0.2):
        n_add = int(frac * drop_idx.size)
        top = mask.copy().reshape(-1)
        top[order[:n_add]] = True
        top = close(top.reshape(mask.shape))
        rnd = mask.copy().reshape(-1)
        rnd[rng.choice(drop_idx, size=n_add, replace=False)] = True
        rnd = close(rnd.reshape(mask.shape))
        t_t, _, _ = pcg(-j, foo, fvv, top)
        t_r, _, _ = pcg(-j, foo, fvv, rnd)
        out["fock_ranked"].append(
            {
                "frac_of_dropped": frac,
                "keep_ranked": float(top.mean()),
                "err_ranked": float(energy(t_t, j) - e_ex),
                "keep_random": float(rnd.mean()),
                "err_random": float(energy(t_r, j) - e_ex),
            }
        )
    # where the exact dropped amplitudes live, by virtual type
    isv = np.zeros(nvl, dtype=bool)
    isv[: sys_.n_vv] = True
    blocks = {}
    t2 = np.where(off, t_ex, 0.0) ** 2
    for na, ma in (("vv", isv), ("hv", ~isv)):
        for nb, mb in (("vv", isv), ("hv", ~isv)):
            blocks[f"{na}-{nb}"] = float(t2[:, ma][:, :, :, mb].sum())
    tot = sum(blocks.values())
    out["dropped_t_norm2_share"] = {k: v / tot for k, v in blocks.items()}
    return out


RUST_MAPS = {
    # in-core lmp2_amplitude, plain Boys (the pre-fix in-core path)
    "old": {"lambda_boys": 0.0},
    # integral-direct with the pre-2026-11-17 maps, no screen, plain Boys
    "old-direct": {
        "integral_direct": True,
        "fock_screen": False,
        "lambda_boys": 0.0,
        "virt_radius": 12.0,
        "ao_tail": 1e-3,
    },
    # integral-direct production defaults: virt_radius 20, ao_tail 1e-4,
    # screen kappa = 1, lambda-Boys 10
    "new": {"integral_direct": True},
    # in-core with the lambda-Boys default (the in-core path has no screen)
    "new-incore": {},
}


def rust_lmp2_series(xyz, basis, auxbasis, eps_list, frozen_core, maps=("old",)):
    """First-order truncation error and domain statistics from ferric's Rust
    amplitude-threshold LMP2 (ported from amplitude_lmp2_proto.py; same
    Eq. 8 mask, same VV-HV construction), for systems the dense-ovov
    prototype cannot hold. dE2 = local - canonical RI-MP2, same aux basis."""
    import ferric

    mol = ferric.Molecule.from_xyz(str(xyz))
    bs, aux = ferric.BasisSet.bundled(basis), ferric.BasisSet.bundled(auxbasis)
    rows, e_ref = [], None
    for mp in maps:
        for eps in eps_list:
            t0 = time.time()
            r = ferric.run_rimp2(
                mol,
                bs,
                aux,
                frozen_core=frozen_core,
                local="amplitude-threshold",
                eps=eps,
                compute_reference=e_ref is None,
                **RUST_MAPS[mp],
            )
            loc = r.local
            if e_ref is None:
                e_ref = loc["e_corr_canonical_ri"]
            row = {
                "maps": mp,
                "lambda_boys": loc.get("lambda_boys"),
                "eps": eps,
                "dE2": r.mp2_corr - e_ref,
                "e2_ref": e_ref,
                "keep": loc["keep_fraction"],
                "pair_fraction": loc["pair_fraction"],
                "n_pairs_unique": loc["n_pairs_unique"],
                "dom_mean": loc["dom_mean"],
                "dom_max": loc["dom_max"],
                "cg_energy_error_bound": loc["cg_energy_error_bound"],
                "wall_s": time.time() - t0,
                "build": loc["build_commit"],
            }
            log(
                f"  rust lmp2 [{mp}] eps={eps:.0e} dE2={row['dE2']:+.3e} keep={row['keep']:.4f} "
                f"pairs={row['n_pairs_unique']} dom={row['dom_mean']:.1f}/{row['dom_max']} "
                f"({row['wall_s']:.0f}s)"
            )
            rows.append(row)
    return {"xyz": Path(xyz).name, "basis": basis, "aux": auxbasis, "rust_lmp2": rows}


def amplification(err):
    """a = |dOP| / (|dE2| + |dE3|): the OP error on the no-cancellation
    scale of the MP3 error (derivation note 6)."""
    scale = abs(err["e2"]) + abs(err["e3"])
    return {
        m: abs(err[m]) / scale if scale else float("nan")
        for m in ("op-mp3", "sop-mp3", "ssd-opmp3", "ssd-opmp3(0.5)")
    }


def summarize(paths):
    """Markdown tables of the 'mask' rows of the JSON files."""
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        for res in data if isinstance(data, list) else [data]:
            if "size_consistency" in res:
                print(
                    f"\n### [{Path(p).stem}] size consistency {res['xyz']} {res['basis']} "
                    f"x{res['copies']} at {res['sep_bohr']} Bohr\n"
                )
                print(
                    "| eps | keep(sup) | "
                    + " | ".join(res["size_consistency"][0]["gap"])
                    + " |"
                )
                print("|" + "---|" * (2 + len(res["size_consistency"][0]["gap"])))
                for r in res["size_consistency"]:
                    print(
                        f"| {r['eps']:.0e} | {r['keep_sup']:.4f} | "
                        + " | ".join(f"{v:+.1e}" for v in r["gap"].values())
                        + " |"
                    )
                continue
            if "canonical" not in res:  # plateau / Rust LMP2 records: read directly
                continue
            ec = res["canonical"]["e2"]
            print(
                f"\n### [{Path(p).stem}] {res['xyz']} {res['basis']} "
                f"ref={res.get('reference', 'hf')} df={res.get('df')} stretch={res['stretch']} "
                f"no={res['no']} nv={res['nv']}; E2={ec:.6f} E3/E2="
                f"{res['canonical']['e3'] / ec:+.3f} zeta_max={res['zeta_can']:.3f} "
                f"lam(A-B)={res['lam_can']:+.3e}; anchor {res['anchor_eps0']:.1e}\n"
            )
            print(
                "| eps | keep | pairs | dE2 | dE3 | dMP3 | dOP | dSOP | dSSD | "
                "dOP/Ec | a(OP) | a(SOP) | a(SSD) | pred/OP | lam_loc | ladder | ring |"
            )
            print("|" + "---|" * 17)
            for r in res["rows"]:
                if r["mode"] != "mask":
                    continue
                e, a = r["err"], amplification(r["err"])
                pr = r["pred"]["op-mp3"] / e["op-mp3"] if e["op-mp3"] else float("nan")
                print(
                    f"| {r['eps']:.0e} | {r['keep']:.3f} | {r['pair_keep']:.3f} | "
                    f"{e['e2']:+.2e} | {e['e3']:+.2e} | {e['mp3']:+.2e} | "
                    f"{e['op-mp3']:+.2e} | {e['sop-mp3']:+.2e} | {e['ssd-opmp3']:+.2e} | "
                    f"{e['op-mp3'] / ec:.1e} | {a['op-mp3']:.2f} | {a['sop-mp3']:.2f} | "
                    f"{a['ssd-opmp3']:.2f} | {pr:.3f} | {r['lam_loc']:+.3e} | "
                    f"{r['ladder_frac']:.3f} | {r['ring_frac']:.3f} |"
                )


ACONF_REACTIONS = [  # GMTKN55 ACONF: reactant -> product
    ("B_T", "B_G"),
    ("P_TT", "P_TG"),
    ("P_TT", "P_GG"),
    ("P_TT", "P_GX"),
    ("H_ttt", "H_gtt"),
    ("H_ttt", "H_tgt"),
    ("H_ttt", "H_tgg"),
    ("H_ttt", "H_gtg"),
    ("H_ttt", "H_g+t+g-"),
    ("H_ttt", "H_ggg"),
    ("H_ttt", "H_g+x-t+"),
    ("H_ttt", "H_t+g+x-"),
    ("H_ttt", "H_g+x-g-"),
    ("H_ttt", "H_x+g-g-"),
    ("H_ttt", "H_x+g-x+"),
]
KCAL = 627.5094740631


def _aconf20_reactions():
    """The 21 ACONF20 reactions, read from the ACONFL reference file."""
    p = ROOT / "testdata" / "molecules" / "aconfl" / "ref_aconf20.csv"
    rows = p.read_text(encoding="utf-8").splitlines()[1:]
    return [tuple(x.strip() for x in r.split(",")[1].split("->")) for r in rows]


RDKIT_C20_REACTIONS = [("c20_conf0", f"c20_conf{k}") for k in (1, 2, 3)]


def reaction_set(prefix):
    if prefix.startswith("aconfl"):
        return _aconf20_reactions()  # ACONFL ACONF20: every conformer vs A2000
    if prefix.startswith("rdkit"):
        return RDKIT_C20_REACTIONS
    return ACONF_REACTIONS


def aconf_summary(data_dir, basis_tag):
    """Truncation error of each ACONF conformer energy difference, per model
    and eps: d(dE) = err(product) - err(reactant), with err = local -
    canonical for the same model on the same structure (no experimental
    reference enters). Reports MAX and RMS over the reactions present."""
    keys = ("e2", "mp3", "op-mp3", "sop-mp3", "ssd-opmp3")
    errs = {}
    for f in Path(data_dir).glob(f"{basis_tag}_*.json"):
        res = json.loads(f.read_text(encoding="utf-8"))[0]
        name = f.stem.removeprefix(f"{basis_tag}_")
        errs[name] = {r["eps"]: r for r in res["rows"] if r["mode"] == "mask"}
    allrx = reaction_set(basis_tag)
    rx = [(a, b) for a, b in allrx if a in errs and b in errs]
    out = {"set": basis_tag, "n_reactions": len(rx), "rows": []}
    print(
        f"\n### {basis_tag}: {len(rx)}/{len(allrx)} reactions; "
        "d(dE) = local - canonical, kcal/mol\n"
    )
    print(
        "| eps | keep(mean) | " + " | ".join(f"MAX {k} | RMS {k}" for k in keys) + " |"
    )
    print("|" + "---|" * (2 + 2 * len(keys)))
    for eps in sorted({e for v in errs.values() for e in v}, reverse=True):
        if not all(eps in errs[a] and eps in errs[b] for a, b in rx):
            continue
        row = {
            "eps": eps,
            "keep_mean": float(np.mean([errs[n][eps]["keep"] for n in errs])),
        }
        cells = []
        for k in keys:
            d = (
                np.array(
                    [errs[b][eps]["err"][k] - errs[a][eps]["err"][k] for a, b in rx]
                )
                * KCAL
            )
            row[k] = {
                "max": float(np.abs(d).max()),
                "rms": float(np.sqrt((d**2).mean())),
                "per_reaction": dict(zip([f"{a}->{b}" for a, b in rx], d.tolist())),
            }
            cells += [f"{row[k]['max']:.3f}", f"{row[k]['rms']:.3f}"]
        out["rows"].append(row)
        print(f"| {eps:.0e} | {row['keep_mean']:.3f} | " + " | ".join(cells) + " |")
    return out


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "--summarize":
        summarize(argv[1:])
        return {}
    if argv and argv[0] == "--rust-boys-probe":  # --rust-boys-probe <xyz> [json]
        res = rust_boys_probe(argv[1])
        if "--json" in argv:
            out = Path(argv[argv.index("--json") + 1])
            out.write_text(json.dumps([res], indent=1), encoding="utf-8")
        return res
    if argv and argv[0] == "--aconf":  # --aconf <data_dir> <prefix, e.g. aconf_631g>
        res = aconf_summary(argv[1], argv[2])
        if len(argv) > 3:
            Path(argv[3]).write_text(json.dumps(res, indent=1), encoding="utf-8")
        return res
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--xyz", required=True)
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--eps", default="1e-3,1e-4,1e-5,1e-6,1e-7")
    ap.add_argument("--modes", default="mask,full")
    ap.add_argument("--stretch", type=float, default=None)
    ap.add_argument("--anchor-only", action="store_true")
    ap.add_argument("--ferric", action="store_true")
    ap.add_argument(
        "--mutate", choices=["semicanonical", "c2_mask_shift"], default=None
    )
    ap.add_argument(
        "--drop-term",
        type=int,
        default=None,
        help="mutation: drop residual term N in the ferric cross-check",
    )
    ap.add_argument(
        "--copies", type=int, default=1, help="size-consistency supersystem"
    )
    ap.add_argument("--sep", type=float, default=100.0, help="copy spacing, Bohr")
    ap.add_argument(
        "--reference",
        default="hf",
        help="hf, or a PySCF xc name (pbe0): HF Fock of the KS density, "
        "semicanonical, with non-Brillouin singles",
    )
    ap.add_argument(
        "--df",
        default=None,
        help="auxiliary basis (e.g. cc-pvdz-ri, or 'auto'): DF integrals and "
        "the per-pair DF ladder",
    )
    ap.add_argument("--df-check", type=float, default=None, metavar="EPS")
    ap.add_argument("--compare-exact", action="store_true")
    ap.add_argument("--no-lmp2-check", action="store_true")
    ap.add_argument("--plateau", type=float, default=None, metavar="EPS")
    ap.add_argument(
        "--rust-lmp2",
        default=None,
        metavar="AUXBASIS",
        help="first-order error + domain stats from ferric run_rimp2 local",
    )
    ap.add_argument("--mask-arm", choices=["eq8", "fock"], default="eq8")
    ap.add_argument("--boys-stable", action="store_true")
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument(
        "--orbitals", choices=["pyscf", "stable", "lambda"], default="pyscf"
    )
    ap.add_argument("--lambda-boys", type=float, default=10.0)
    ap.add_argument(
        "--arms",
        default=None,
        help="comma list of named map configs (" + ",".join(MAPS) + "); overrides "
        "--mask-arm/--kappa/--orbitals/--boys-stable; canonical W computed once",
    )
    ap.add_argument(
        "--rust-maps",
        default="old,old-direct,new",
        help="with --rust-lmp2: old (in-core, lambda 0), old-direct (direct, "
        "screen off, lambda 0, virt_radius 12, ao_tail 1e-3), new (direct defaults)",
    )
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    a.eps = [float(x) for x in a.eps.split(",") if x]
    a.modes = [m for m in a.modes.split(",") if m]
    a.rust_maps = [m for m in a.rust_maps.split(",") if m]
    if a.arms:
        arms = [m for m in a.arms.split(",") if m]
    else:
        arms = [None]
    out = []
    for arm in arms:
        if arm is None:
            MASK_ARM.update(arm=a.mask_arm, kappa=a.kappa, r2=False)
            ORB.update(occ=a.orbitals, **{"lambda": a.lambda_boys})
            BOYS_STABLE["on"] = a.boys_stable
        else:
            set_maps(arm)
            log(f"#### arm {arm}: {MAPS[arm]}")
        res = run(a)
        if isinstance(res, dict):
            res["mask_arm"] = dict(MASK_ARM)
            res["arm"] = arm
        out.append(res)
        if a.json:
            p = Path(a.json)
            allres = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
            allres.append(res)
            p.write_text(json.dumps(allres, indent=1), encoding="utf-8")
    return out[0] if len(out) == 1 else out


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
