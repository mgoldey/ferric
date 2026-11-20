#!/usr/bin/env python3
"""Sympy proof of the identities local (amplitude-threshold) OP-MP3 relies on.

Method: OP-MP3 (book/src/10-mp2/11-opmp3.md) evaluated on first- and
second-order doubles amplitudes solved in localized orbitals on ONE
amplitude mask (the amplitude-threshold LMP2 mask, book section 13.1).
Prototype: scripts/prototypes/local_opmp3_proto.py, derivation
scripts/prototypes/local_opmp3_derivation.md.

Closed-shell conventions: spatial orbitals, real; t[i,a,j,b] is the
alpha-beta amplitude; the swap (S t)[i,a,j,b] = t[i,b,j,a]; the closed-shell
energy weight is G = 2 - S, so E = <(2 - S) c, J>; A is the Fock
superoperator (A t)_iajb = F_ac t_icjb + t_iajc F_cb - F_ik t_kajb - t_iakb F_kj;
W_ij = 1/4 sum_kab c_iakb J_jakb + (i <-> j); P(A, B) = A (A - B)^-1 A.

Claims (exact sympy unless marked numerical):

  a1  Tr P(U^T A U, U^T B U) = Tr P(A, B) for symbolic symmetric 2x2 A, B
      and a symbolic rotation U (rational parametrization).
  a2  W covariance: rotating the occupied index of c and J by U and the
      virtual indices by V (both orthogonal, symbolic angles) maps W to
      U^T W U (no = nv = 2, every amplitude and integral an independent
      symbol). With a1, every OP-type model is invariant under the
      occupied and virtual localization at a vacuous mask.
  b   Wigner 2n+1 on ONE mask: for a symbolic invertible A with G A
      symmetric and G block-diagonal on the kept set K, the masked
      solutions c1_K = -A_KK^-1 J_K and c2_K = -A_KK^-1 R_K satisfy
      <G c2, J> = <G c1, R> (3-dimensional model, |K| = 2).
  b2  the same with c2 solved on another mask K2 != K is NOT an identity
      (rational counterexample).
  N   NUMERICAL, independent route, closed-shell model (no = 3, nv = 4,
      integrals from random 3-index factors so (pq|rs) has the physical
      permutational symmetry): canonical route (diagonal Fock, division)
      vs local route (random orthogonal U, V, full non-diagonal Fock,
      dense linear solve), with the ladder + hole-ladder MP3 residual:
        N1  vacuous mask: OP-MP3 and SSD-OPMP3(x_ss = 1/2) equal canonical;
        N2  a finite swap-closed mask: <G c2, J> = <G c1, R(c1)>;
        N3  a finite mask: the non-Brillouin singles energy
            E_NBS = -2 sum F_ia^2 / (e_a - e_i) (semicanonical) equals the
            local-frame singles solve, so E_NBS is a reference-wide term
            independent of the doubles mask.

Mutations (LOCAL_OPMP3_MUTATION), each turns its check red:
  pade_half_rotated   rotate A but not B in a1: a1 fails.
  w_virtual_once      rotate only the virtual index a of J (not b) in a2:
                      a2 fails.
  two_masks           solve c2 on K2 in b / on a shifted mask in N2: b, N2
                      fail.
  semicanonical       drop the off-diagonal local Fock in the local route:
                      N1 fails.
  nbs_local_diag      local singles solve with the diagonal of the local
                      Fock only (a mask-like truncation of E_NBS): N3 fails.

Exit code 0 iff every check passes; "all identities hold" is printed then.
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import sympy as sp

MUTATION = os.environ.get("LOCAL_OPMP3_MUTATION", "")
MUTATIONS = (
    "",
    "pade_half_rotated",
    "w_virtual_once",
    "two_masks",
    "semicanonical",
    "nbs_local_diag",
)
if MUTATION not in MUTATIONS:
    sys.exit(f"unknown LOCAL_OPMP3_MUTATION {MUTATION!r}")

FAILED = []


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}".rstrip())
    if not ok:
        FAILED.append(label)


def rot(t):
    """Rational 2x2 rotation: cos = (1 - t^2)/(1 + t^2), sin = 2t/(1 + t^2)."""
    c, s = (1 - t**2) / (1 + t**2), 2 * t / (1 + t**2)
    return sp.Matrix([[c, -s], [s, c]])


def sym2(name):
    a, b, c = sp.symbols(f"{name}0:3")
    return sp.Matrix([[a, b], [b, c]])


def pade(a, b):
    return a * (a - b).inv() * a


# ---------------------------------------------------------------------------
# a1, a2: invariance
# ---------------------------------------------------------------------------


def prove_a1():
    a, b = sym2("A"), sym2("B")
    u = rot(sp.Symbol("t"))
    ar = u.T * a * u
    br = b if MUTATION == "pade_half_rotated" else u.T * b * u
    d = sp.simplify(pade(ar, br).trace() - pade(a, b).trace())
    # claim a1: the trace Pade is invariant under an occupied rotation
    check("a1 Tr P(U^T A U, U^T B U) == Tr P(A, B)", d == 0)


def prove_a2():
    n = 2
    c = sp.MutableDenseNDimArray(
        [
            sp.Symbol(f"c{i}{a}{k}{b}")
            for i, a, k, b in itertools.product(range(n), repeat=4)
        ],
        (n,) * 4,
    )
    j = sp.MutableDenseNDimArray(
        [
            sp.Symbol(f"J{i}{a}{k}{b}")
            for i, a, k, b in itertools.product(range(n), repeat=4)
        ],
        (n,) * 4,
    )
    u, v = rot(sp.Symbol("t")), rot(sp.Symbol("s"))

    def w_of(cc, jj):
        w = sp.zeros(n, n)
        for i, jx in itertools.product(range(n), repeat=2):
            w[i, jx] = sp.Rational(1, 4) * sum(
                cc[i, a, k, b] * jj[jx, a, k, b] + cc[jx, a, k, b] * jj[i, a, k, b]
                for a, k, b in itertools.product(range(n), repeat=3)
            )
        return w

    def rotate(x, virt_b=True):
        out = sp.MutableDenseNDimArray.zeros(n, n, n, n)
        for i, a, k, b in itertools.product(range(n), repeat=4):
            out[i, a, k, b] = sum(
                u[p, i]
                * v[q, a]
                * u[r, k]
                * (v[s, b] if virt_b else (1 if s == b else 0))
                * x[p, q, r, s]
                for p, q, r, s in itertools.product(range(n), repeat=4)
            )
        return out

    cr = rotate(c)
    jr = rotate(j, virt_b=MUTATION != "w_virtual_once")
    d = (w_of(cr, jr) - u.T * w_of(c, j) * u).applyfunc(sp.simplify)
    # claim a2: W is covariant under the occupied and virtual rotations
    check("a2 W[c U V, J U V] == U^T W[c, J] U", d == sp.zeros(n, n))


# ---------------------------------------------------------------------------
# b: Wigner 2n+1 on one mask
# ---------------------------------------------------------------------------


def prove_b():
    # G A = H symmetric; G block-diagonal on K = {0, 1}, {2}
    h = sp.Matrix(3, 3, lambda r, c: sp.Symbol(f"h{min(r, c)}{max(r, c)}"))
    g = sp.Matrix(
        [
            [sp.Symbol("g00"), sp.Symbol("g01"), 0],
            [sp.Symbol("g01"), sp.Symbol("g11"), 0],
            [0, 0, sp.Symbol("g22")],
        ]
    )
    a = g.inv() * h
    jv = sp.Matrix(sp.symbols("J0:3"))
    rv = sp.Matrix(sp.symbols("R0:3"))
    k = [0, 1]
    akk = a.extract(k, k)
    c1 = -akk.inv() * jv.extract(k, [0])
    k2 = [0, 2] if MUTATION == "two_masks" else k
    c2k = -a.extract(k2, k2).inv() * rv.extract(k2, [0])
    c2 = sp.zeros(3, 1)
    for idx, kk in enumerate(k2):
        c2[kk] = c2k[idx]
    c1f = sp.zeros(3, 1)
    for idx, kk in enumerate(k):
        c1f[kk] = c1[idx]
    lhs = (c2.T * g * jv)[0]
    rhs = (c1f.T * g * rv)[0]
    # claim b: the 2n+1 form of E3 holds on one mask
    check("b <G c2, J> == <G c1, R> on one mask", sp.simplify(lhs - rhs) == 0)
    # b2: two masks is not an identity (rational counterexample)
    a2 = a.subs(
        {
            "h00": 3,
            "h01": 1,
            "h11": 4,
            "h02": 1,
            "h12": 1,
            "h22": 5,
            "g00": 1,
            "g01": 0,
            "g11": 1,
            "g22": 1,
        }
    )
    vals = {"J0": 1, "J1": 2, "J2": 3, "R0": 1, "R1": -1, "R2": 2}
    c1n = -a2.extract(k, k).inv() * sp.Matrix([1, 2])
    c2n = -a2.extract([0, 2], [0, 2]).inv() * sp.Matrix([1, 2])
    lhs2 = c2n[0] * vals["J0"] + c2n[1] * vals["J2"]
    rhs2 = c1n[0] * vals["R0"] + c1n[1] * vals["R1"]
    check(
        "b2 two masks: the identity fails (counterexample)",
        sp.simplify(lhs2 - rhs2) != 0,
        f"({lhs2} vs {rhs2})",
    )


# ---------------------------------------------------------------------------
# N: closed-shell numerical model, canonical vs local route
# ---------------------------------------------------------------------------

NO, NV, NAUX = 3, 4, 9


def swap(t):
    return t.transpose(0, 3, 2, 1)


def energy(c, j):
    return float(np.sum((2 * c - swap(c)) * j))


def w_mats(c, j):
    """alpha-block W^SS, W^OS (book 10.11 closed-shell form)."""
    y_os = 0.25 * np.einsum("iakb,jakb->ij", c, j)
    y_ss = 0.25 * np.einsum("iakb,jakb->ij", c - swap(c), j)
    return y_ss + y_ss.T, y_os + y_os.T


def models(w2, w3, x_ss=0.5):
    """2 x the alpha trace: OP-MP3 and SSD-OPMP3(x_ss)."""
    p = lambda a, b: a @ np.linalg.solve(a - b, a)
    a2, a3 = w2[0] + w2[1], w3[0] + w3[1]
    op = 2 * np.trace(p(a2, a3))
    ssd = 2 * (np.trace(p(w2[0], x_ss * w3[0])) + np.trace(p(w2[1], w3[1])))
    return {"op-mp3": op, "ssd": ssd}


def apply_a(t, foo, fvv):
    return (
        np.einsum("ac,icjb->iajb", fvv, t)
        + np.einsum("iajc,cb->iajb", t, fvv)
        - np.einsum("ik,kajb->iajb", foo, t)
        - np.einsum("iakb,kj->iajb", t, foo)
    )


def solve_masked(rhs, foo, fvv, mask):
    """Dense solve of P A P c = P rhs on the kept entries."""
    idx = np.flatnonzero(mask.ravel())
    n = rhs.size
    a = np.zeros((n, n))
    for col in range(n):
        e = np.zeros(n)
        e[col] = 1.0
        a[:, col] = apply_a(e.reshape(rhs.shape), foo, fvv).ravel()
    c = np.zeros(n)
    c[idx] = np.linalg.solve(a[np.ix_(idx, idx)], rhs.ravel()[idx])
    return c.reshape(rhs.shape)


def residual(t, b):
    """Ladder + hole ladder of the alpha-beta MP3 residual from 3-index
    factors b[P, p, q] (occupied first): sum_cd (ac|bd) t_icjd +
    sum_kl (ki|lj) t_kalb. Covariant under orbital rotations."""
    bvv = b[:, NO:, NO:]
    boo = b[:, :NO, :NO]
    vvvv = np.einsum("Pac,Pbd->acbd", bvv, bvv)
    oooo = np.einsum("Pki,Plj->kilj", boo, boo)
    return np.einsum("acbd,icjd->iajb", vvvv, t) + np.einsum("kilj,kalb->iajb", oooo, t)


def numerical():
    rng = np.random.default_rng(20261118)
    n = NO + NV
    b = rng.standard_normal((NAUX, n, n)) * 0.3
    b = 0.5 * (b + b.transpose(0, 2, 1))
    eo = np.sort(rng.uniform(-1.5, -0.5, NO))
    ev = np.sort(rng.uniform(0.3, 1.5, NV))
    # a reference-wide occ-vir Fock block (KS reference, non-Brillouin)
    fov = rng.standard_normal((NO, NV)) * 0.05

    def route(uo, uv, mask, semican=False, mask2=None):
        bl = b.copy()
        rot_ = np.zeros((n, n))
        rot_[:NO, :NO], rot_[NO:, NO:] = uo, uv
        bl = np.einsum("Ppq,pr,qs->Prs", bl, rot_, rot_)
        foo, fvv = uo.T @ np.diag(eo) @ uo, uv.T @ np.diag(ev) @ uv
        if semican:
            foo, fvv = np.diag(np.diag(foo)), np.diag(np.diag(fvv))
        j = np.einsum("Pia,Pjb->iajb", bl[:, :NO, NO:], bl[:, :NO, NO:])
        c1 = solve_masked(-j, foo, fvv, mask)
        r1 = residual(c1, bl)
        c2 = solve_masked(-r1, foo, fvv, mask if mask2 is None else mask2)
        return j, c1, c2, r1

    full = np.ones((NO, NV, NO, NV), dtype=bool)
    j0, c10, c20, _ = route(np.eye(NO), np.eye(NV), full)
    m0 = models(w_mats(c10, j0), w_mats(c20, j0))
    uo, _ = np.linalg.qr(rng.standard_normal((NO, NO)))
    uv, _ = np.linalg.qr(rng.standard_normal((NV, NV)))
    j1, c11, c21, _ = route(uo, uv, full, semican=MUTATION == "semicanonical")
    m1 = models(w_mats(c11, j1), w_mats(c21, j1))
    dev = max(abs(m0[k] - m1[k]) for k in m0)
    check(
        "N1 vacuous mask: local route == canonical (OP-MP3, SSD 1/2)",
        dev < 1e-12,
        f"(max dev {dev:.1e}, OP-MP3 {m0['op-mp3']:.12f})",
    )

    # N2: finite swap-closed mask in the local frame
    j1, _, _, _ = route(uo, uv, full)
    thr = np.quantile(np.abs(j1), 0.45)
    mask = (np.abs(j1) > thr) | (np.abs(swap(j1)) > thr)
    mask |= mask.transpose(2, 3, 0, 1)
    mask2 = None
    if MUTATION == "two_masks":
        mask2 = mask.copy()
        flip = np.argwhere(mask)[0]
        mask2[tuple(flip)] = False
        mask2[flip[0], flip[3], flip[2], flip[1]] = False
    j, c1, c2, r1 = route(uo, uv, mask, mask2=mask2)
    gap = energy(c2, j) - energy(c1, r1)
    check(
        "N2 one mask: <G c2, J> == <G c1, R(c1)>",
        abs(gap) < 1e-13,
        f"(keep {mask.mean():.2f}, gap {gap:.1e})",
    )

    # claim N3: E_NBS is reference-wide: semicanonical formula == local singles solve
    d_ia = ev[None, :] - eo[:, None]
    e_can = -2.0 * float(np.sum(fov**2 / d_ia))
    fov_l = uo.T @ fov @ uv
    foo_l, fvv_l = uo.T @ np.diag(eo) @ uo, uv.T @ np.diag(ev) @ uv
    if MUTATION == "nbs_local_diag":
        foo_l, fvv_l = np.diag(np.diag(foo_l)), np.diag(np.diag(fvv_l))
    a1 = np.kron(np.eye(NO), fvv_l) - np.kron(foo_l, np.eye(NV))
    t1 = np.linalg.solve(a1, -fov_l.ravel())
    e_loc = 2.0 * float(t1 @ fov_l.ravel())
    check(
        "N3 E_NBS: semicanonical == local singles solve",
        abs(e_can - e_loc) < 1e-14,
        f"({e_can:.12f})",
    )


def main():
    if MUTATION:
        print(f"mutation: {MUTATION}")
    prove_a1()
    prove_a2()
    prove_b()
    numerical()
    if FAILED:
        print(f"{len(FAILED)} check(s) failed")
        return 1
    print("all identities hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
