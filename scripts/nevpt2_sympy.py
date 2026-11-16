#!/usr/bin/env python3
"""Proofs for strongly contracted NEVPT2 as ferric-ci codes it.

Theory: crates/ferric-ci/theory.md, "SC-NEVPT2 (`src/nevpt2.rs`)" and "NEVPT
RDMs (`src/rdm3.rs`)".
Code: crates/ferric-ci/src/nevpt2.rs (norm_to_energy, the make_a* and make_k27
intermediates, the eight class functions sr ... sir, the Blocks built in
sc_nevpt2_on), crates/ferric-ci/src/rdm3.rs (nevpt_rdms, e_product_rdms,
hole_rdms).

The model: 2 core, 4 active (CAS(4,4), n_alpha = n_beta = 2) and 2 virtual
orbitals with an orthonormal basis, a random one-electron h with core below,
active around and virtuals above the Fermi level, and positive semidefinite
low-rank 8-fold-symmetric (pq|rs). Psi_0 is the lowest root of the active
Hamiltonian built on F^I; the core and virtual orbitals are rotated to
diagonalize the core and virtual blocks of F = F^I + F^A, as sc_nevpt2_on does.

Claims (the book cites claim D, claim R, claim T and claim C):

  D  (sympy) One strongly contracted perturber Psi_l = P_l H Psi_0 with
     Dyall's H_D gives the second-order term -|<Psi_l|H|Psi_0>|^2 / N_l /
     (E_l - E_0), E_l = <Psi_l|H_D|Psi_l>/N_l; with <Psi_l|H|Psi_0> = N_l and
     E_l - E_0 = Delta_l + K_l/N_l this is -N_l/(Delta_l + K_l/N_l), the
     norm_to_energy formula.
  R  RDM inputs, on the model's CI vector, every one by two routes.
       R1 the normal-ordered dm1..dm3 (annihilated vectors a a a Psi, Gram
          matrices) against the E-product <E E>, <E E E> (dense E_pq
          matrices): the reorder formula of rdm3.rs maps G3 to dm3, and
          e_product_rdms maps dm1..dm3 back to G2, G3.
       R2 hole_rdms (transcribed) against <a a a+ a+> and <a a a a+ a+ a+>
          from created vectors a+ a+ a+ Psi in the N+1..N+3 spaces.
       R3 f3ca and f3ac by the rdm3.rs construction
          (<E_dc X_ba | R_ef>, R = sum_j E_ej W_fj, R' = sum_j E_jf W_ej,
          W_fj = sum_ki (ki|fj) X_ki) against the contraction of a stored
          E-product 4-RDM <E_ab E_cd E_ef E_gh> with (aa|aa).
  T  Transcription: the einsum specifications and permutation axes of every
     class and intermediate function below, in order, equal those in
     nevpt2.rs (parsed from the file), and so do the integral-block
     permutations of sc_nevpt2_on.
  C  Classes, on the lowest singlet and on the lowest M_s = 0 triplet root of
     the (2, 2) sector: each of the eight class (norm, energy) pairs of the
     transcription equals the definition evaluated in the full determinant
     space (4900 determinants: Psi_l = P_l H Psi_0 grouped by external
     occupation, N_l = |Psi_l|^2, K_l = <Psi_l|H_act - E_0|Psi_l>, Dyall's
     H_act on active indices only, E -= N/(Delta + K/N)).
       C0 anchors: <Psi_0|H|Psi_0> = E_core + E_0; the sum of all label norms
          is |Q H Psi_0|^2 with Q the complement of the CAS space;
       C1..C8 Sr, Si, Sijrs, Sijr, Srsi, Srs, Sij, Sir.

The transcription evaluates every einsum into an owned array, as the Rust
einsum does: numpy returns a view for a pure transpose, and the in-place
updates of make_a7 would otherwise write through into the 3-RDM.

Mutations (crates/ferric-python/tests/test_derivation_scripts.py applies each
as an exact substitution and requires the named FAIL):
  * the f3ca term of make_a16 with its sign flipped ("FAIL C1");
  * a sign flip in make_a12 ("FAIL C8");
  * the pair diagonal of Sijr not halved ("FAIL C4");
  * theta = 2v - v_swap of Sijrs replaced by v ("FAIL C3");
  * the delta_qt term of e_product_rdms dropped ("FAIL R1");
  * a coefficient of hole_rdms 2 -> 1 ("FAIL R2");
  * R' built with the R index order ("FAIL R3");
  * the sign of the Sij denominator flipped ("FAIL C7");
  * one Blocks permutation changed, i <-> j of v_sijr ("FAIL T"; Sijr is
    symmetrized over the pair, so only part T sees it).

Run: python3 scripts/nevpt2_sympy.py  (needs numpy, sympy)
"""

from __future__ import annotations

import inspect
import itertools
import os
import re
import sys
from pathlib import Path

import numpy as np
import sympy as sp

FAILED = []
EPS = np.finfo(float).eps
NORM_ZERO = 1e-14
NC, NA, NV = 2, 4, 2
NMO = NC + NA + NV
NEL_ACT = (2, 2)


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILED.append(label)


def close(label, got, want, bar):
    err = float(np.max(np.abs(np.asarray(got) - np.asarray(want))))
    check(label, err <= bar, f"max|d| {err:.1e}, bar {bar:.1e}")


def ein(spec, *ops):
    # always an owned array, as the Rust einsum returns: a transposing spec
    # gives a numpy view, and an in-place update of it would write through to
    # its operand
    return np.array(np.einsum(spec, *ops, optimize=True), copy=True)


def perm(a, axes):
    return np.ascontiguousarray(np.transpose(a, axes))


# ------------------------------------------------------------------ part D
def part_d():
    print("D  one strongly contracted perturber")
    n, k, d, e0 = sp.symbols("N K Delta E_0", positive=True)
    # <Psi_l|H|Psi_0> = <P_l H Psi_0|H Psi_0> = <P_l H Psi_0|P_l H Psi_0> = N.
    coupling = n
    e_l = e0 + d + k / n  # <Psi_l|H_D|Psi_l>/N_l with H_D = eps-sum + H_act
    e2 = -(coupling**2) / n / (e_l - e0)
    check(
        "D  -|<l|H|0>|^2/N/(E_l - E_0) = -N/(Delta + K/N)",
        sp.simplify(e2 + n / (d + k / n)) == 0,
    )


# ---------------------------------------------------- second quantization
def popcount(x):
    return bin(x).count("1")


def op(state, p, spin, create):
    """a+_{p spin} (create) or a_{p spin} on a dict {(alpha, beta): c},
    spin orbitals ordered alpha before beta."""
    out = {}
    for (a, b), c in state.items():
        m = a if spin == 0 else b
        if bool(m >> p & 1) == create:
            continue
        sgn = popcount(m & ((1 << p) - 1)) + (popcount(a) if spin == 1 else 0)
        m2 = m ^ (1 << p)
        key = (m2, b) if spin == 0 else (a, m2)
        out[key] = out.get(key, 0.0) + (-1.0) ** sgn * c
    return out


def strings(norb, nel):
    return [
        sum(1 << i for i in occ) for occ in itertools.combinations(range(norb), nel)
    ]


def gram(vectors, shape):
    """Gram matrix of a dict-of-dicts {label: state} reshaped to shape+shape."""
    labels = list(itertools.product(*[range(s) for s in shape]))
    keys = sorted({k for v in vectors.values() for k in v})
    kidx = {k: i for i, k in enumerate(keys)}
    m = np.zeros((len(labels), len(keys)))
    for li, lab in enumerate(labels):
        for k, c in vectors.get(lab, {}).items():
            m[li, kidx[k]] += c
    return (m @ m.T).reshape(tuple(shape) + tuple(shape))


# ------------------------------------------------------------------- model
def build_model():
    rng = np.random.default_rng(20261115)
    h = np.diag([-4.0, -3.4, -0.6, -0.2, 0.3, 0.7, 2.2, 2.9])
    x = 0.08 * rng.standard_normal((NMO, NMO))
    h = h + x + x.T
    b = rng.standard_normal((12, NMO, NMO))
    b = 0.25 * (b + b.transpose(0, 2, 1))
    g = ein("Ppq,Prs->pqrs", b, b)
    return h, g


def fock_i(h, g):
    c = range(NC)
    return h + sum(2.0 * g[:, :, i, i] - g[:, i, i, :] for i in c)


def active_ci(h_act, g_act, spin2):
    """Lowest root of H_act in the (2, 2) sector, as a dict state and the
    dense E_pq matrices of that sector."""
    sa = strings(NA, NEL_ACT[0])
    sb = strings(NA, NEL_ACT[1])
    dets = [(a, b) for a in sa for b in sb]
    idx = {d: i for i, d in enumerate(dets)}
    e = np.zeros((NA, NA, len(dets), len(dets)))
    for j, d in enumerate(dets):
        for p, q in itertools.product(range(NA), repeat=2):
            for s in (0, 1):
                v = op(op({d: 1.0}, q, s, False), p, s, True)
                for k, c in v.items():
                    e[p, q, idx[k], j] += c
    ham = ein("pq,pqij->ij", h_act, e)
    ee = ein("pqik,rskj->pqrsij", e, e)
    ham += 0.5 * ein("pqrs,pqrsij->ij", g_act, ee)
    ham -= 0.5 * ein("pqqs,psij->ij", g_act, e)
    # S^2 = S_- S_+ + S_z^2 + S_z, with S_z = 0 in this sector
    plus = {}
    for j, d in enumerate(dets):
        for p in range(NA):
            for k, cf in op(op({d: 1.0}, p, 1, False), p, 0, True).items():
                plus[(k, j)] = plus.get((k, j), 0.0) + cf
    up = sorted({k for k, _ in plus})
    sp_m = np.zeros((len(up), len(dets)))
    for (k, j), cf in plus.items():
        sp_m[up.index(k), j] += cf
    s2 = sp_m.T @ sp_m
    w, v = np.linalg.eigh(ham)
    s2_roots = ein("ik,ij,jk->k", v, s2, v)
    k = int(np.flatnonzero(np.abs(s2_roots - spin2) < 1e-8)[0])
    c = v[:, k]
    return {d: c[i] for i, d in enumerate(dets)}, c, e, w[k]


# ------------------------------------------------------------------ part R
def direct_rdms(state, c, e):
    """Normal-ordered dm1..dm3, E-product G2..G4, hole RDMs, all by
    definition."""
    n = NA
    t1 = {(p, s): op(state, p, s, False) for p in range(n) for s in (0, 1)}
    raw1 = gram(t1, (n, 2))
    dm1 = ein("psqs->pq", raw1)
    t2 = {
        (p, r, s, t): op(t1[(p, s)], r, t, False)
        for p in range(n)
        for r in range(n)
        for s in (0, 1)
        for t in (0, 1)
    }
    raw2 = gram(t2, (n, n, 2, 2))
    dm2 = perm(ein("prstqust->prqu", raw2), (0, 2, 1, 3))  # [p,q,r,s]
    t3 = {
        (p, r, u, s, t, w): op(t2[(p, r, s, t)], u, w, False)
        for p, r, u in itertools.product(range(n), repeat=3)
        for s, t, w in itertools.product((0, 1), repeat=3)
    }
    raw3 = gram(t3, (n, n, n, 2, 2, 2))
    dm3 = perm(ein("pruabcqsvabc->pruqsv", raw3), (0, 3, 1, 4, 2, 5))
    # hole RDMs: U_{a..} = a+_c a+_b a+_a Psi, bra the same with p, q, r
    u1 = {(p, s): op(state, p, s, True) for p in range(n) for s in (0, 1)}
    hdm1 = ein("psqs->pq", gram(u1, (n, 2)))
    u2 = {
        (p, q, s, t): op(u1[(p, s)], q, t, True)
        for p in range(n)
        for q in range(n)
        for s in (0, 1)
        for t in (0, 1)
    }
    hdm2 = ein("pqstabst->pqab", gram(u2, (n, n, 2, 2)))
    u3 = {
        (p, q, r, s, t, w): op(u2[(p, q, s, t)], r, w, True)
        for p, q, r in itertools.product(range(n), repeat=3)
        for s, t, w in itertools.product((0, 1), repeat=3)
    }
    hdm3 = ein("pqrstwabcstw->pqrabc", gram(u3, (n, n, n, 2, 2, 2)))
    # E-products with dense E matrices: <E_pq E_rs ...> = c^T E_pq E_rs ... c
    x = ein("rsij,j->rsi", e, c)  # E_rs c
    y = ein("pqij,rsj->pqrsi", e, x)
    z = ein("pqij,rstuj->pqrstui", e, y)
    bra = ein("qpij,j->pqi", e, c)  # (c^T E_pq)^T = E_qp c
    g2 = ein("pqi,rsi->pqrs", bra, x)
    g3 = ein("pqi,rstui->pqrstu", bra, y)
    g4 = ein("pqi,rstuvwi->pqrstuvw", bra, z)
    return dict(
        dm1=dm1, dm2=dm2, dm3=dm3, hdm1=hdm1, hdm2=hdm2, hdm3=hdm3, g2=g2, g3=g3, g4=g4
    )


def dl(i, j):
    return 1.0 if i == j else 0.0


def reorder_dm3(dm1, g2, g3):
    """The rdm3.rs module formula: dm3 = G3 - d_qr dm2[p,s,t,u] - d_st dm2[p,q,r,u]
    - d_qt dm2[p,u,r,s] - d_qr d_st dm1[u,p], with dm2 = G2 - d_qr dm1[p,s]."""
    n = dm1.shape[0]
    i = np.eye(n)
    dm2 = g2 - ein("qr,ps->pqrs", i, dm1)
    dm3 = g3 - ein("qr,pstu->pqrstu", i, dm2)
    dm3 -= ein("st,pqru->pqrstu", i, dm2)
    dm3 -= ein("qt,purs->pqrstu", i, dm2)
    dm3 -= ein("qr,st,up->pqrstu", i, i, dm1)
    return dm2, dm3


def e_product_rdms(dm1, dm2, dm3):
    """rdm3.rs e_product_rdms."""
    n = dm1.shape[0]
    g2 = np.zeros((n,) * 4)
    for p, q, r, s_ in itertools.product(range(n), repeat=4):
        g2[p, q, r, s_] = dm2[p, q, r, s_] + dl(q, r) * dm1[p, s_]
    g3 = dm3.copy()
    for p, q, r, s_, t, u in itertools.product(range(n), repeat=6):
        val = 0.0
        if q == r:
            val += dm2[p, s_, t, u]
        if s_ == t:
            val += dm2[p, q, r, u]
        if q == t:
            val += dm2[p, u, r, s_]
        if q == r and s_ == t:
            val += dm1[u, p]
        g3[p, q, r, s_, t, u] += val
    return g2, g3


def hole_rdms(dm1, dm2, dm3):
    """rdm3.rs hole_rdms."""
    n = dm1.shape[0]
    g2, g3 = e_product_rdms(dm1, dm2, dm3)
    hdm1 = np.zeros((n, n))
    for p, a in itertools.product(range(n), repeat=2):
        hdm1[p, a] = 2.0 * dl(p, a) - dm1[a, p]

    def t2(i, j, k, l_):
        return g2[i, k, j, l_] - dl(j, k) * dm1[i, l_]

    hdm2 = np.zeros((n,) * 4)
    for i, j, k, l_ in itertools.product(range(n), repeat=4):
        hdm2[i, j, k, l_] = (
            t2(k, l_, i, j)
            + dl(i, l_) * dm1[k, j]
            + dl(j, k) * dm1[l_, i]
            - 2.0 * dl(i, k) * dm1[l_, j]
            - 2.0 * dl(j, l_) * dm1[k, i]
            - 2.0 * dl(i, l_) * dl(j, k)
            + 4.0 * dl(i, k) * dl(j, l_)
        )
    hdm3 = np.zeros((n,) * 6)
    for p, q, r, a, b, c in itertools.product(range(n), repeat=6):
        hdm3[p, q, r, a, b, c] = (
            -dl(p, b) * hdm2[q, r, a, c]
            - dl(b, r) * hdm2[p, q, a, c]
            + 2.0 * dl(b, q) * hdm2[p, r, a, c]
            + 2.0 * dl(a, p) * g2[b, q, c, r]
            - 4.0 * dl(a, p) * dl(c, r) * dm1[b, q]
            + 2.0 * dl(c, r) * g2[b, q, a, p]
            - g3[b, q, a, p, c, r]
            + 2.0 * dl(a, r) * dl(p, c) * dm1[b, q]
            - dl(a, r) * g2[b, q, c, p]
        )
    return hdm1, hdm2, hdm3


def f3_rdm3_route(c, e, g_act):
    """rdm3.rs: f3ca = <E_dc X_ba | R_ef>, f3ac = <E_dc X_ba | R'_ef>."""
    x = ein("rsij,j->rsi", e, c)  # X_rs = E_rs C
    w = ein("kifj,kiD->fjD", g_act, x)  # W_fj = sum_ki (ki|fj) X_ki
    r = ein("ejDE,fjE->efD", e, w)  # R_ef = sum_j E_ej W_fj
    rp = ein("jfDE,ejE->efD", e, w)  # R'_ef = sum_j E_jf W_ej
    lm = ein("dcDE,baE->dcbaD", e, x)  # L = E_dc X_ba
    f3ca = ein("dcbaD,efD->abcdef", lm, r)
    f3ac = ein("dcbaD,efD->abcdef", lm, rp)
    return f3ca, f3ac


def part_r(state, c, e, g_act):
    print("R  RDM inputs, two routes each")
    d = direct_rdms(state, c, e)
    n = NA
    scale = max(1.0, float(np.max(np.abs(d["g4"]))))
    # rounding of sums of at most 36 dets x 8^3 spin/label terms of numbers <= scale
    bar = 1e3 * EPS * scale * n**2
    dm2_r, dm3_r = reorder_dm3(d["dm1"], d["g2"], d["g3"])
    close("R1 reorder(G2) = normal-ordered dm2", dm2_r, d["dm2"], bar)
    close("R1 reorder(G3) = normal-ordered dm3", dm3_r, d["dm3"], bar)
    g2, g3 = e_product_rdms(d["dm1"], d["dm2"], d["dm3"])
    close("R1 e_product_rdms G2 = <E E>", g2, d["g2"], bar)
    close("R1 e_product_rdms G3 = <E E E>", g3, d["g3"], bar)
    close(
        "R1 partial trace sum_x dm3[p,q,r,s,x,x] = (N - 2) dm2",
        ein("pqrsxx->pqrs", d["dm3"]),
        (sum(NEL_ACT) - 2) * d["dm2"],
        bar,
    )
    h1, h2, h3 = hole_rdms(d["dm1"], d["dm2"], d["dm3"])
    close("R2 hole_rdms hdm1 = <a a+>", h1, d["hdm1"], bar)
    close("R2 hole_rdms hdm2 = <a a a+ a+>", h2, d["hdm2"], bar)
    close("R2 hole_rdms hdm3 = <a a a a+ a+ a+>", h3, d["hdm3"], bar)
    f3ca, f3ac = f3_rdm3_route(c, e, g_act)
    want_ca = ein("kifj,abcdejki->abcdef", g_act, d["g4"])
    want_ac = ein("ikje,abcdjfik->abcdef", g_act, d["g4"])
    gbar = bar * max(1.0, float(np.sum(np.abs(g_act))))
    close(
        "R3 f3ca (rdm3.rs route) = sum (ki|fj) <E_ab E_cd E_ej E_ki>",
        f3ca,
        want_ca,
        gbar,
    )
    close(
        "R3 f3ac (rdm3.rs route) = sum (ik|je) <E_ab E_cd E_jf E_ik>",
        f3ac,
        want_ac,
        gbar,
    )
    del n
    return dict(
        dm1=d["dm1"], dm2=g2, dm3=g3, f3ca=f3ca, f3ac=f3ac, hdm1=h1, hdm2=h2, hdm3=h3
    )


# ------------------------------------------- transcription of nevpt2.rs
def eye(n):
    return np.eye(n)


def norm_to_energy(labels):
    nt, e = 0.0, 0.0
    for n, h, d in labels:
        nt += n
        if abs(n) > NORM_ZERO:
            e -= n / (d + h / n)
    return nt, e


def make_a16(x):
    h1e, h2e, dm3 = x["h1e"], x["h2e"], x["dm3"]
    n = h1e.shape[0]
    a = -ein("ib,rpqiac->pqrabc", h1e, dm3)
    a += ein("ia,rpqbic->pqrabc", h1e, dm3)
    a -= ein("ci,rpqbai->pqrabc", h1e, dm3)
    a -= perm(x["f3ca"], (1, 4, 0, 2, 5, 3))
    a -= ein("kbia,rpqcki->pqrabc", h2e, dm3)
    a -= ein("kbaj,rpqjkc->pqrabc", h2e, dm3)
    a += ein("cbij,rpqjai->pqrabc", h2e, dm3)
    fdm2 = ein("kbij,rpajki->prab", h2e, dm3)
    a += ein("prab,qc->pqrabc", fdm2, eye(n))
    a += perm(x["f3ac"], (1, 2, 0, 4, 3, 5))
    a -= perm(x["f3ca"], (1, 2, 0, 4, 3, 5))
    a += ein("jbij,rpqiac->pqrabc", h2e, dm3)
    a -= ein("cjka,rpqbjk->pqrabc", h2e, dm3)
    a += ein("jcij,rpqbai->pqrabc", h2e, dm3)
    return a


def make_a22(x):
    h1e, h2e, dm2, dm3 = x["h1e"], x["h2e"], x["dm2"], x["dm3"]
    n = h1e.shape[0]
    d = eye(n)
    a = -ein("pb,kipjac->ijkabc", h1e, dm3)
    a -= ein("pa,kibjpc->ijkabc", h1e, dm3)
    a += ein("cp,kibjap->ijkabc", h1e, dm3)
    a += ein("cqra,kibjqr->ijkabc", h2e, dm3)
    a -= ein("qcpq,kibjap->ijkabc", h2e, dm3)
    a -= perm(x["f3ac"], (1, 5, 0, 2, 4, 3))
    fdm2 = ein("pqrb,kiqcpr->ikbc", h2e, dm3)
    a -= ein("ikbc,ja->ijkabc", fdm2, d)
    a -= ein("pqab,kiqjpc->ijkabc", h2e, dm3)
    a += ein("pcrb,kiajpr->ijkabc", h2e, dm3)
    a += ein("cqrb,kiqjar->ijkabc", h2e, dm3)
    a -= perm(x["f3ac"], (1, 3, 0, 4, 2, 5))
    a += perm(x["f3ca"], (1, 3, 0, 4, 2, 5))
    a += 2.0 * ein("jb,kiac->ijkabc", h1e, dm2)
    a += 2.0 * ein("pjrb,kiprac->ijkabc", h2e, dm3)
    f = ein("pa,kipc->ikac", h1e, dm2)
    f -= ein("cp,kiap->ikac", h1e, dm2)
    f -= ein("cqra,kiqr->ikac", h2e, dm2)
    f += ein("qcpq,kiap->ikac", h2e, dm2)
    f += ein("pqra,kiqcpr->ikac", h2e, dm3)
    f -= ein("rcpq,kiaqrp->ikac", h2e, dm3)
    a += 2.0 * ein("ikac,jb->ijkabc", f, d)
    return a


def h1e_shifted(h1e, h2e):
    return h1e - ein("mjjn->mn", h2e)


def make_a17(x):
    h = h1e_shifted(x["h1e"], x["h2e"])
    return -ein("pi,cabi->abcp", h, x["dm2"]) - ein(
        "kpij,cabjki->abcp", x["h2e"], x["dm3"]
    )


def make_a19(x):
    h = h1e_shifted(x["h1e"], x["h2e"])
    return -ein("pi,ai->ap", h, x["dm1"]) - ein("kpij,ajki->ap", x["h2e"], x["dm2"])


def make_a23(x):
    h1e, h2e = x["h1e"], x["h2e"]
    a = -ein("ip,caib->abcp", h1e, x["dm2"])
    a -= ein("pijk,cajbik->abcp", h2e, x["dm3"])
    a += 2.0 * ein("bp,ca->abcp", h1e, x["dm1"])
    a += 2.0 * ein("pibk,caik->abcp", h2e, x["dm2"])
    return a


def make_a25(x):
    h1e, h2e = x["h1e"], x["h2e"]
    a = -ein("pi,ai->ap", h1e, x["dm1"])
    a -= ein("pijk,jaik->ap", h2e, x["dm2"])
    a += 2.0 * ein("ap->pa", h1e)
    a += 2.0 * ein("piaj,ij->ap", h2e, x["dm1"])
    return a


def make_a3(x):
    h1e, h2e = x["h1e"], x["h2e"]
    d = eye(h1e.shape[0])
    a = ein("ia,ip->pa", h1e, x["hdm1"])
    a += 2.0 * ein("ijka,pj,ik->pa", h2e, d, x["dm1"])
    a -= ein("ijka,jpik->pa", h2e, x["dm2"])
    return a


def make_k27(x):
    h1e, h2e = x["h1e"], x["h2e"]
    k = -ein("ai,pi->pa", h1e, x["dm1"])
    k -= ein("iajk,pkij->pa", h2e, x["dm2"])
    k += ein("iaji,pj->pa", h2e, x["dm1"])
    return k


def make_a7(x):
    h1e, h2e = x["h1e"], x["h2e"]
    d = eye(h1e.shape[0])
    rm2 = ein("iljk->ijkl", x["dm2"]) - ein("ik,jl->ijkl", x["dm1"], d)
    rm3 = ein("injmkl->ijklmn", x["dm3"])
    rm3 -= ein("jn,imkl->ijklmn", d, x["dm2"])
    rm3 -= ein("km,ijln->ijklmn", d, rm2)
    rm3 -= ein("kn,ijml->ijklmn", d, rm2)
    a7 = -ein("bi,pqia->pqab", h1e, rm2)
    a7 -= ein("ai,pqbi->pqab", h1e, rm2)
    a7 -= ein("kbij,pqkija->pqab", h2e, rm3)
    a7 -= ein("kaij,pqkibj->pqab", h2e, rm3)
    a7 -= ein("baij,pqij->pqab", h2e, rm2)
    return rm2, a7


def make_a9(x):
    h1e, h2e, hdm2, hdm3 = x["h1e"], x["h2e"], x["hdm2"], x["hdm3"]
    a = ein("ib,pqai->pqab", h1e, hdm2)
    a += 2.0 * ein("ijib,pqaj->pqab", h2e, hdm2)
    a -= ein("ijjb,pqai->pqab", h2e, hdm2)
    a -= ein("ijkb,pkqaij->pqab", h2e, hdm3)
    a += ein("ia,pqib->pqab", h1e, hdm2)
    a -= ein("ijja,pqib->pqab", h2e, hdm2)
    a -= ein("ijba,pqji->pqab", h2e, hdm2)
    a += 2.0 * ein("ijia,pqjb->pqab", h2e, hdm2)
    a -= ein("ijka,pqkjbi->pqab", h2e, hdm3)
    return a


def make_a12(x):
    h1e, h2e, dm2, dm3 = x["h1e"], x["h2e"], x["dm2"], x["dm3"]
    a = ein("ia,qpib->pqab", h1e, dm2)
    a -= ein("bi,qpai->pqab", h1e, dm2)
    a += ein("ijka,qpjbik->pqab", h2e, dm3)
    a -= ein("kbij,qpajki->pqab", h2e, dm3)
    a -= ein("bjka,qpjk->pqab", h2e, dm2)
    a += ein("jbij,qpai->pqab", h2e, dm2)
    return a


def make_a13(x):
    h1e, h2e, dm1, dm2, dm3 = x["h1e"], x["h2e"], x["dm1"], x["dm2"], x["dm3"]
    d = eye(h1e.shape[0])
    a = -ein("ia,qbip->pqab", h1e, dm2)
    a += 2.0 * ein("pa,qb->pqab", h1e, dm1)
    a += ein("bi,qiap->pqab", h1e, dm2)
    a -= 2.0 * ein("pa,bi,qi->pqab", d, h1e, dm1)
    a -= ein("ijka,qbjpik->pqab", h2e, dm3)
    a += ein("kbij,qjapki->pqab", h2e, dm3)
    a += ein("blma,qmlp->pqab", h2e, dm2)
    a += 2.0 * ein("kpma,qbkm->pqab", h2e, dm2)
    a -= 2.0 * ein("bpma,qm->pqab", h2e, dm1)
    a -= ein("lbkl,qkap->pqab", h2e, dm2)
    a -= 2.0 * ein("ap,mbkl,qlmk->pqab", d, h2e, dm2)
    a += 2.0 * ein("ap,lbkl,qk->pqab", d, h2e, dm1)
    return a


def sr(x, b):
    v, h = b["v_sr"], b["h_sr"]
    a16 = make_a16(x)
    a17 = make_a17(x)
    a19 = make_a19(x)
    ener = (
        ein("ipqr,pqrabc,iabc->i", v, a16, v)
        + 2.0 * ein("ipqr,pqra,ia->i", v, a17, h)
        + ein("ip,pa,ia->i", h, a19, h)
    )
    norm = (
        ein("ipqr,rpqbac,iabc->i", v, x["dm3"], v)
        + 2.0 * ein("ipqr,rpqa,ia->i", v, x["dm2"], h)
        + ein("ip,pa,ia->i", h, x["dm1"], h)
    )
    return norm_to_energy((norm[r], ener[r], x["eps_v"][r]) for r in range(x["nv"]))


def si(x, b):
    v, h = b["v_si"], b["h_si"]
    n = x["h1e"].shape[0]
    d = eye(n)
    a22 = make_a22(x)
    a23 = make_a23(x)
    a25 = make_a25(x)
    dm3_h = 2.0 * ein("abef,cd->abcdef", x["dm2"], d) - perm(
        x["dm3"], (0, 1, 3, 2, 4, 5)
    )
    dm2_h = 2.0 * ein("ab,cd->abcd", x["dm1"], d) - perm(x["dm2"], (0, 1, 3, 2))
    dm1_h = 2.0 * d - perm(x["dm1"], (1, 0))
    ener = (
        ein("qpir,pqrabc,baic->i", v, a22, v)
        + 2.0 * ein("qpir,pqra,ai->i", v, a23, h)
        + ein("pi,pa,ai->i", h, a25, h)
    )
    norm = (
        ein("qpir,rpqbac,baic->i", v, dm3_h, v)
        + 2.0 * ein("qpir,rpqa,ai->i", v, dm2_h, h)
        + ein("pi,pa,ai->i", h, dm1_h, h)
    )
    return norm_to_energy((norm[i], ener[i], -x["eps_c"][i]) for i in range(x["nc"]))


def sijrs(x, b):
    g = b["cvcv"]
    norm, e = 0.0, 0.0
    ec, ev = x["eps_c"], x["eps_v"]
    for i in range(x["nc"]):
        for a in range(x["nv"]):
            for j in range(x["nc"]):
                for bb in range(x["nv"]):
                    v = g[i, a, j, bb]
                    theta = 2.0 * v - g[i, bb, j, a]
                    norm += v * theta
                    e += v * theta / (ec[i] + ec[j] - ev[a] - ev[bb])
    return norm, e


def pair_symmetrized(s1, s2, v, m, swap):
    t = 2.0 * ein(s1, v, v, m) - ein(s2, v, v, m)
    t += perm(t, swap)
    return t


def sijr(x, b):
    v = b["v_sijr"]
    a3 = make_a3(x)
    s1, s2 = ("rpji,raji,pa->rji", "rpji,raij,pa->rji")
    norm = pair_symmetrized(s1, s2, v, x["hdm1"], (0, 2, 1))
    h = pair_symmetrized(s1, s2, v, a3, (0, 2, 1))
    labels = []
    for r in range(x["nv"]):
        for j in range(x["nc"]):
            for i in range(j, x["nc"]):
                f = 0.5 if i == j else 1.0
                diff = x["eps_v"][r] - x["eps_c"][j] - x["eps_c"][i]
                labels.append((f * norm[r, j, i], f * h[r, j, i], diff))
    return norm_to_energy(labels)


def srsi(x, b):
    v = b["v_srsi"]
    k27 = make_k27(x)
    s1, s2 = ("rsip,rsia,pa->rsi", "rsip,sria,pa->rsi")
    norm = pair_symmetrized(s1, s2, v, x["dm1"], (1, 0, 2))
    h = pair_symmetrized(s1, s2, v, k27, (1, 0, 2))
    labels = []
    for r in range(x["nv"]):
        for s_ in range(r, x["nv"]):
            f = 0.5 if r == s_ else 1.0
            for i in range(x["nc"]):
                diff = x["eps_v"][r] + x["eps_v"][s_] - x["eps_c"][i]
                labels.append((f * norm[r, s_, i], f * h[r, s_, i], diff))
    return norm_to_energy(labels)


def srs(x, b):
    v = b["v_srs"]
    rm2, a7 = make_a7(x)
    norm = 0.5 * ein("rsqp,rsba,pqba->rs", v, v, rm2)
    h = 0.5 * ein("rsqp,rsba,pqab->rs", v, v, a7)
    labels = []
    for r in range(x["nv"]):
        for s_ in range(x["nv"]):
            labels.append((norm[r, s_], h[r, s_], x["eps_v"][r] + x["eps_v"][s_]))
    return norm_to_energy(labels)


def sij(x, b):
    v = b["v_sij"]
    a9 = make_a9(x)
    norm = 0.5 * ein("qpij,baij,pqab->ij", v, v, x["hdm2"])
    h = 0.5 * ein("qpij,baij,pqab->ij", v, v, a9)
    labels = []
    for i in range(x["nc"]):
        for j in range(x["nc"]):
            labels.append((norm[i, j], h[i, j], -(x["eps_c"][i] + x["eps_c"][j])))
    return norm_to_energy(labels)


def sir(x, b):
    v1, v2, h1 = b["v1_sir"], b["v2_sir"], b["h_sir"]
    dm1, dm2 = x["dm1"], x["dm2"]
    norm = 2.0 * ein("rpiq,raib,qpab->ir", v1, v1, dm2)
    norm -= ein("rpiq,rabi,qpab->ir", v1, v2, dm2)
    norm -= ein("rpqi,raib,qpab->ir", v2, v1, dm2)
    norm += 2.0 * ein("raqi,rabi,qb->ir", v2, v2, dm1)
    norm -= ein("rpqi,rabi,qbap->ir", v2, v2, dm2)
    norm += ein("rpqi,raai,qp->ir", v2, v2, dm1)
    norm += 4.0 * ein("rpiq,ri,qp->ir", v1, h1, dm1)
    norm -= 2.0 * ein("rpqi,ri,qp->ir", v2, h1, dm1)
    norm += 2.0 * ein("ri,ri->ir", h1, h1)
    a12 = make_a12(x)
    a13 = make_a13(x)
    h = 2.0 * ein("rpiq,raib,pqab->ir", v1, v1, a12)
    h -= ein("rpiq,rabi,pqab->ir", v1, v2, a12)
    h -= ein("rpqi,raib,pqab->ir", v2, v1, a12)
    h += ein("rpqi,rabi,pqab->ir", v2, v2, a13)
    labels = []
    for i in range(x["nc"]):
        for r in range(x["nv"]):
            labels.append((norm[i, r], h[i, r], x["eps_v"][r] - x["eps_c"][i]))
    return norm_to_energy(labels)


CLASS_FNS = [
    ("Sr", sr),
    ("Si", si),
    ("Sijrs", sijrs),
    ("Sijr", sijr),
    ("Srsi", srsi),
    ("Srs", srs),
    ("Sij", sij),
    ("Sir", sir),
]
# external (virtual particles, core holes) of each class
CLASS_EXT = {
    "Sr": (1, 0),
    "Si": (0, 1),
    "Sijrs": (2, 2),
    "Sijr": (1, 2),
    "Srsi": (2, 1),
    "Srs": (2, 0),
    "Sij": (0, 2),
    "Sir": (1, 1),
}


def blocks(g, fi):
    """sc_nevpt2_on's Blocks, on MO integrals g[p,q,r,s] = (pq|rs)."""
    rc, ra, rv = range(NC), range(NC, NC + NA), range(NC + NA, NMO)
    cc, ca, cv = list(rc), list(ra), list(rv)

    def blk(o):
        return g[np.ix_(*o)]

    def mat(r, q):
        return fi[np.ix_(list(r), list(q))]

    v_sr = perm(blk([cv, ca, ca, ca]), (0, 2, 1, 3))
    h_sr = mat(rv, ra) - ein("mbbn->mn", v_sr)
    return dict(
        h_sr=h_sr,
        v_sr=v_sr,
        v_si=perm(blk([ca, cc, ca, ca]), (0, 2, 1, 3)),
        h_si=mat(ra, rc),
        cvcv=blk([cc, cv, cc, cv]),
        v_sijr=perm(blk([cc, ca, cc, cv]), (3, 1, 2, 0)),
        v_srsi=perm(blk([cv, ca, cc, cv]), (3, 0, 2, 1)),
        v_srs=perm(blk([cv, ca, cv, ca]), (0, 2, 1, 3)),
        v_sij=perm(blk([cc, ca, cc, ca]), (1, 3, 0, 2)),
        v1_sir=perm(blk([cv, cc, ca, ca]), (0, 2, 1, 3)),
        v2_sir=perm(blk([cv, ca, cc, ca]), (0, 3, 1, 2)),
        h_sir=mat(rv, rc),
    )


# ------------------------------------------------------------------ part T
def repo_root():
    cands = [Path(__file__).resolve().parents[1], Path.cwd(), *Path.cwd().parents]
    if os.environ.get("FERRIC_ROOT"):
        cands.insert(0, Path(os.environ["FERRIC_ROOT"]))
    for c in cands:
        if (c / "crates" / "ferric-ci" / "src" / "nevpt2.rs").is_file():
            return c
    return None


RUST_TOK = re.compile(r'"([a-z,]*->[a-z]*)"|&\[(\d+(?:,\s*\d+)+)\]')
PY_TOK = re.compile(r'"([a-z,]*->[a-z]*)"|\((\d+(?:,\s*\d+)+)\)')


def tokens(text, pat):
    return [s if s else "perm" + re.sub(r"\s", "", a) for s, a in pat.findall(text)]


def rust_fn(src, name):
    m = re.search(rf"\nfn {name}\(", src)
    if not m:
        return None
    depth, i = 0, src.index("{", m.start())
    for j in range(i, len(src)):
        depth += {"{": 1, "}": -1}.get(src[j], 0)
        if depth == 0:
            return src[m.start() : j + 1]
    return None


def part_t():
    print("T  transcription against nevpt2.rs")
    root = repo_root()
    if root is None:
        check("T  nevpt2.rs found (set FERRIC_ROOT)", False)
        return
    src = (root / "crates" / "ferric-ci" / "src" / "nevpt2.rs").read_text(
        encoding="utf-8"
    )
    names = [
        "make_a16",
        "make_a22",
        "h1e_shifted",
        "make_a17",
        "make_a19",
        "make_a23",
        "make_a25",
    ]
    names += ["make_a3", "make_k27", "make_a7", "make_a9", "make_a12", "make_a13"]
    names += ["sr", "si", "sijr", "srsi", "srs", "sij", "sir", "pair_symmetrized"]
    for name in names:
        body = rust_fn(src, name)
        if body is None:
            check(f"T  fn {name} in nevpt2.rs", False)
            continue
        rt = tokens(body, RUST_TOK)
        pt = tokens(inspect.getsource(globals()[name]), PY_TOK)
        check(
            f"T  {name}: {len(rt)} einsum specs and permutations",
            rt == pt,
            "" if rt == pt else f"rust {rt} / script {pt}",
        )
    m = re.search(r"let v_sr = .*?drop\(src\);", src, re.S)
    rt = tokens(m.group(0), RUST_TOK) if m else []
    pt = tokens(inspect.getsource(blocks), PY_TOK)
    check(
        "T  sc_nevpt2_on Blocks: block permutations",
        bool(rt) and rt == pt,
        "" if rt == pt else f"rust {rt} / script {pt}",
    )
    # the class-to-energy map: Sijrs is the direct loop with theta = 2v - v_swap
    body = rust_fn(src, "sijrs") or ""
    check(
        "T  sijrs: theta = 2v - g[i,b,j,a], denominator e_i + e_j - e_a - e_b",
        "let theta = 2.0 * v - g[[i, bb, j, a]];" in body
        and "(x.eps_c[i] + x.eps_c[j] - x.eps_v[a] - x.eps_v[bb])" in body,
    )
    body = rust_fn(src, "norm_to_energy") or ""
    check(
        "T  norm_to_energy: e -= n / (d + h / n) for |n| > NORM_ZERO",
        "e -= n / (d + h / n);" in body and "n.abs() > NORM_ZERO" in body,
    )


# ------------------------------------------------------------------ part C
def emats(strs):
    idx = {s: i for i, s in enumerate(strs)}
    e = np.zeros((NMO, NMO, len(strs), len(strs)))
    for col, s in enumerate(strs):
        for q in range(NMO):
            if not s >> q & 1:
                continue
            s1 = s ^ (1 << q)
            sg1 = (-1) ** popcount(s1 & ((1 << q) - 1))
            for p in range(NMO):
                if s1 >> p & 1:
                    continue
                s2 = s1 | (1 << p)
                sg2 = (-1) ** popcount(s1 & ((1 << p) - 1))
                e[p, q, idx[s2], col] = sg1 * sg2
    return e


def brute_classes(h, g, fi, eps, cstate):
    """The SC-NEVPT2 definition in the full determinant space."""
    nal, nbe = NC + NEL_ACT[0], NC + NEL_ACT[1]
    sa, sb = strings(NMO, nal), strings(NMO, nbe)
    ia = {s: i for i, s in enumerate(sa)}
    ib = {s: i for i, s in enumerate(sb)}
    ea, eb = emats(sa), emats(sb)

    def apply_e(p, q, v):
        return ea[p, q] @ v + v @ eb[p, q].T

    core = (1 << NC) - 1
    psi = np.zeros((len(sa), len(sb)))
    for (a, b), c in cstate.items():
        psi[ia[core | (a << NC)], ib[core | (b << NC)]] = c
    phi = np.array([[apply_e(r, s, psi) for s in range(NMO)] for r in range(NMO)])
    xx = ein("pqrs,rsij->pqij", g, phi)
    kk = ein("pqqs->ps", g)
    hpsi = np.zeros_like(psi)
    for p in range(NMO):
        for q in range(NMO):
            hpsi += apply_e(p, q, h[p, q] * psi + 0.5 * xx[p, q])
            hpsi -= 0.5 * kk[p, q] * phi[p, q]
    e_full = float(np.sum(psi * hpsi))
    act = range(NC, NC + NA)

    def hact(v):
        ph = {(r, s): apply_e(r, s, v) for r in act for s in act}
        out = np.zeros_like(v)
        for p in act:
            for q in act:
                acc = fi[p, q] * v
                for r in act:
                    for s in act:
                        acc = acc + 0.5 * g[p, q, r, s] * ph[(r, s)]
                out += apply_e(p, q, acc)
                kpq = sum(g[p, r, r, q] for r in act)
                out -= 0.5 * kpq * ph[(p, q)]
        return out

    e0 = float(np.sum(psi * hact(psi)))
    y = hpsi.copy()
    labels = {}
    for i, s in enumerate(sa):
        for j, t in enumerate(sb):
            holes = [c for c in range(NC) for m in (s, t) if not m >> c & 1]
            parts = [r for r in range(NV) for m in (s, t) if m >> (NC + NA + r) & 1]
            lab = (tuple(sorted(parts)), tuple(sorted(holes)))
            if not lab[0] and not lab[1]:
                y[i, j] = 0.0
                continue
            labels.setdefault(lab, []).append((i, j))
    q_norm = float(np.sum(y * y))
    hy = hact(y)
    ec, ev = eps[:NC], eps[NC + NA :]
    out = {name: [0.0, 0.0] for name in CLASS_EXT}
    cls_of = {v: k for k, v in CLASS_EXT.items()}
    beyond = 0.0
    for (vl, cl), members in labels.items():
        ii, jj = np.array(members).T
        nn = float(np.sum(y[ii, jj] ** 2))
        name = cls_of.get((len(vl), len(cl)))
        if name is None:  # three or more externals: H reaches none of them
            beyond = max(beyond, nn)
            continue
        out[name][0] += nn
        if nn <= NORM_ZERO:
            continue
        kl = float(np.sum(y[ii, jj] * hy[ii, jj])) - e0 * nn
        delta = sum(ev[r] for r in vl) - sum(ec[i] for i in cl)
        out[name][1] -= nn / (delta + kl / nn)
    out["beyond"] = beyond
    return out, e_full, e0, q_norm


def part_c(h, g, rd, cstate, e0_ci):
    print("C  classes against the determinant-space definition")
    fi = fock_i(h, g)
    a = slice(NC, NC + NA)
    e_core = float(sum(h[i, i] + fi[i, i] for i in range(NC)))
    x = dict(
        nc=NC,
        nv=NV,
        h1e=fi[a, a],
        h2e=perm(g[a, a, a, a], (0, 2, 1, 3)),
        **rd,
    )
    fa = ein("pqtu,tu->pq", g, np.pad(rd["dm1"], ((NC, NV), (NC, NV)))) - 0.5 * ein(
        "ptuq,tu->pq", g, np.pad(rd["dm1"], ((NC, NV), (NC, NV)))
    )
    f = fi + fa
    eps = np.diag(f).copy()
    off = max(
        np.max(np.abs(f[:NC, :NC] - np.diag(eps[:NC]))),
        np.max(np.abs(f[NC + NA :, NC + NA :] - np.diag(eps[NC + NA :]))),
    )
    check(
        "C0 core and virtual blocks of F are diagonal (canonical orbitals)",
        off < 1e-12,
        f"{off:.1e}",
    )
    x["eps_c"], x["eps_v"] = eps[:NC], eps[NC + NA :]
    b = blocks(g, fi)
    brute, e_full, e0, q_norm = brute_classes(h, g, fi, eps, cstate)
    scale = max(1.0, float(np.max(np.abs(g))), float(np.max(np.abs(h))))
    # rounding: the determinant route sums ~NMO^4 applications on 4900-element
    # arrays, the transcription ~n^8 products; both are well inside this bar
    bar = 1e5 * EPS * scale**2
    close("C0 <Psi_0|H|Psi_0> = E_core + E_0", e_full, e_core + e0, bar)
    close("C0 E_0 = the CAS eigenvalue", e0, e0_ci, bar)
    total_norm, total_e = 0.0, 0.0
    for k, (name, fn) in enumerate(CLASS_FNS, start=1):
        n_l, e_l = fn(x, b)
        total_norm += n_l
        total_e += e_l
        print(
            f"     {name:6s} N {n_l: .10f}  E {e_l: .12f}  (definition {brute[name][1]: .12f})"
        )
        close(f"C{k} {name} norm", n_l, brute[name][0], bar)
        close(f"C{k} {name} energy", e_l, brute[name][1], bar)
        check(f"C{k} {name} is not zero (|E| > 1e-6)", abs(e_l) > 1e-6, f"{e_l:.2e}")
    close("C0 sum of class norms = |Q H Psi_0|^2", total_norm, q_norm, bar)
    check(
        "C0 no weight beyond two externals",
        brute["beyond"] == 0.0,
        f"{brute['beyond']:.1e}",
    )
    print(f"     E2 total {total_e: .12f}")


def canonical_model(spin2):
    h, g = build_model()
    fi = fock_i(h, g)
    a = slice(NC, NC + NA)
    cstate, c, e, e0 = active_ci(fi[a, a], g[a, a, a, a], spin2)
    d1 = ein("pqij,i,j->pq", e, c, c)
    full = np.zeros((NMO, NMO))
    full[a, a] = d1
    f = fi + ein("pqtu,tu->pq", g, full) - 0.5 * ein("ptuq,tu->pq", g, full)
    u = np.eye(NMO)
    for lo, hi in ((0, NC), (NC + NA, NMO)):
        _, v = np.linalg.eigh(f[lo:hi, lo:hi])
        u[lo:hi, lo:hi] = v
    h = u.T @ h @ u
    g = ein("pP,qQ,rR,sS,pqrs->PQRS", u, u, u, u, g)
    return h, g, cstate, c, e, e0


if __name__ == "__main__":
    part_d()
    part_t()
    act = slice(NC, NC + NA)
    for spin2, root in ((0.0, "singlet"), (2.0, "M_s = 0 triplet")):
        print(f"--- reference: lowest {root} root of the (2, 2) sector")
        h, g, cstate, c, e, e0 = canonical_model(spin2)
        rd = part_r(cstate, c, e, g[act, act, act, act])
        part_c(h, g, rd, cstate, e0)
    if FAILED:
        print("FAILED:", *FAILED, sep="\n  ")
        sys.exit(1)
    print("all SC-NEVPT2 identities hold")
