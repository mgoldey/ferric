#!/usr/bin/env python3
"""Sympy proof that the spin-blocked unrestricted MP3 of ferric-mp2 equals the
spin-orbital MP3, block by block and with its prefactors.

The code under test is crates/ferric-mp2/src/u_mp3.rs (c^(1), energies) and
crates/ferric-mp2/src/u_mp3_residual.rs (the c^(2) residual). The equations
are in crates/ferric-mp2/theory.md section 6.1. Lower case = alpha, upper
case = beta, (pq|rs) chemist, t = c^(1), D = e_i + e_j - e_a - e_b:

  c1_aa = [(ia|jb) - (ib|ja)] / D,  c1_ab[iJaB] = (ia|JB) / D
  R_aa  = sum_cd (ac|bd) t_ijcd + sum_kl (ki|lj) t_klab + P(ij)P(ab) X_iajb
          X_iajb = sum_kc [(kc|jb) - (kj|bc)] t_ikac + sum_KC (KC|jb) t_iKaC
  R_ab  = sum_cD (ac|BD) t_iJcD + sum_kL (ki|LJ) t_kLaB
          + sum_kc (kc|JB) t^aa_ikac + sum_KC [(KC|JB) - (KJ|BC)] t_iKaC
          + sum_kc [(ia|kc) - (ik|ac)] t_kJcB + sum_KC (ia|KC) t^bb_JKBC
          - sum_kC (ki|BC) t_kJaC - sum_Kc (KJ|ac) t_iKcB
  c2 = R / D,  E(n) = 1/4 sum c(n-1)_aa K_aa + 1/4 sum c(n-1)_bb K_bb
                     + sum c(n-1)_ab K_ab

Checks:

  S  SYMBOLIC. Every spatial integral is an independent sympy symbol (with
     the 8-fold same-spin and 4-fold + pair-swap mixed-spin symmetries); the
     orbital energies are fixed rationals, so each energy is an exact cubic
     polynomial in the integral symbols. For alpha (3 occ, 2 vir) and beta
     (2 occ, 3 vir), so that alpha != beta in both spaces and every block,
     including the beta same-spin one, is populated:
       S1  E2: the three blocks sum to 1/4 sum <ij||ab> t (spin orbitals).
       S2  E3 by term class, each one an exact polynomial identity:
             pp  spin-blocked ladders   == 1/8 sum t <ab||cd> t
             hh  spin-blocked ladders   == 1/8 sum <kl||ij> t t
             ph  rings + exchange rings == sum t_ijab <kb||cj> t_ikac
           (the spin-orbital terms and coefficients of mp3.rs).
  N  NUMERICAL, independent route: brute-force Rayleigh-Schroedinger E2 and
     E3 = <psi1|V - E1|psi1> in the full determinant space of a random
     UHF-like Hamiltonian (per-spin spatial integrals, one-body part chosen
     so the reference Fock operator is diag(e)), against the spin-blocked
     formulas. alpha 3 occ / 2 vir, beta 2 occ / 3 vir: 10 spin orbitals,
     5 electrons, 252 determinants.

Mutations (set UMP3_MUTATION), each turns the script red:
  drop_ab_exchange   the last alpha-beta exchange ring  - sum_Kc (KJ|ac) t_iKcB
                     is dropped (S2 ph and N E3 fail)
  ladder_half        the same-spin particle ladder carries 1/2 (the spin-orbital
                     1/2 <ab||cd> without the antisymmetry doubling; S2 pp and
                     N E3 fail)

Measured 2026-11-14: the unmutated run exits 0 (S2 compares 186 pp, 186 hh
and 1086 ph monomials exactly; N agrees to the last printed digit, E3 =
-0.0106326 Ha); each mutation exits 1 with exactly its S2 class and N E3 red.

The checks the book cites carry explicit labels in their printed names:
"(claim S1)", "(claim S2)", "(claim N2)", "(claim N3)".

Exit code 0 iff every check passes; "all identities hold" is printed then.
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import sympy as sp

MUTATION = os.environ.get("UMP3_MUTATION", "")
if MUTATION not in ("", "drop_ab_exchange", "ladder_half"):
    sys.exit(f"unknown UMP3_MUTATION {MUTATION!r}")

failures = 0


def check(label, ok, detail=""):
    global failures
    print(f"{'ok  ' if ok else 'FAIL'} {label}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        failures += 1


# ------------------------------------------------------------------ model
class Model:
    """Per-spin orbitals and spatial chemist integrals.

    no[s], nv[s]: occupied / virtual counts of spin s (0 = alpha, 1 = beta);
    spin s's MO index runs over occupied first. g(s1, p, q, s2, r, s) is
    (p q | r s) with p, q of spin s1 and r, s of spin s2. eps[s][p] the
    orbital energies.
    """

    def __init__(self, no, nv, eps, g):
        self.no, self.nv, self.eps, self._g = no, nv, eps, g

    def g(self, s1, p, q, s2, r, s):
        return self._g(s1, p, q, s2, r, s)


def symbolic_model(no, nv, eps):
    cache = {}

    def key(s1, p, q, s2, r, s):
        a, b = tuple(sorted((p, q))), tuple(sorted((r, s)))
        if s1 == s2:
            return (s1, s1) + tuple(sorted((a, b)))
        if s1 > s2:
            s1, s2, a, b = s2, s1, b, a
        return (s1, s2, a, b)

    def g(s1, p, q, s2, r, s):
        k = key(s1, p, q, s2, r, s)
        if k not in cache:
            cache[k] = sp.Symbol("g_" + "_".join(map(str, sum(map(_flat, k), ()))))
        return cache[k]

    return Model(no, nv, eps, g)


def _flat(x):
    return x if isinstance(x, tuple) else (x,)


def random_model(no, nv, rng):
    n = [no[s] + nv[s] for s in (0, 1)]
    eps = []
    for s in (0, 1):
        occ = -1.5 + rng.random(no[s]) * 0.8
        vir = 0.3 + rng.random(nv[s]) * 1.2
        eps.append(np.concatenate([np.sort(occ), np.sort(vir)]))
    blocks = {}
    for s1, s2 in ((0, 0), (1, 1), (0, 1)):
        x = rng.standard_normal((n[s1], n[s1], n[s2], n[s2])) * 0.05
        x = x + x.transpose(1, 0, 2, 3)
        x = x + x.transpose(0, 1, 3, 2)
        if s1 == s2:
            x = x + x.transpose(2, 3, 0, 1)
        blocks[(s1, s2)] = x

    def g(s1, p, q, s2, r, s):
        if s1 <= s2:
            return blocks[(s1, s2)][p, q, r, s]
        return blocks[(s2, s1)][r, s, p, q]

    return Model(no, nv, eps, g)


# ------------------------------------------------ spin-orbital reference route
def spin_orbitals(m):
    occ = [(s, p) for s in (0, 1) for p in range(m.no[s])]
    vir = [(s, m.no[s] + a) for s in (0, 1) for a in range(m.nv[s])]
    return occ, vir


def phys(m, P, Q, R, S):
    """<PQ||RS> on spin orbitals (spin, MO)."""
    val = 0
    if P[0] == R[0] and Q[0] == S[0]:
        val += m.g(P[0], P[1], R[1], Q[0], Q[1], S[1])
    if P[0] == S[0] and Q[0] == R[0]:
        val -= m.g(P[0], P[1], S[1], Q[0], Q[1], R[1])
    return val


def so_energies(m):
    """(E2, pp, hh, ph) of mp3.rs in spin orbitals."""
    occ, vir = spin_orbitals(m)
    e = lambda o: m.eps[o[0]][o[1]]  # noqa: E731
    t = {}
    for i, j in itertools.product(occ, repeat=2):
        for a, b in itertools.product(vir, repeat=2):
            v = phys(m, i, j, a, b)
            if v != 0:
                t[i, j, a, b] = v / (e(i) + e(j) - e(a) - e(b))
    get = lambda *k: t.get(k, 0)  # noqa: E731
    e2 = sum(phys(m, i, j, a, b) * v for (i, j, a, b), v in t.items()) / 4
    pp = hh = ph = 0
    for (i, j, a, b), v in t.items():
        pp += sum(
            v * phys(m, a, b, c, d) * get(i, j, c, d)
            for c, d in itertools.product(vir, repeat=2)
        )
        hh += sum(
            phys(m, k, ll, i, j) * v * get(k, ll, a, b)
            for k, ll in itertools.product(occ, repeat=2)
        )
        ph += sum(v * phys(m, k, b, c, j) * get(i, k, a, c) for k in occ for c in vir)
    return e2, pp / 8, hh / 8, ph


# ---------------------------------------------------- spin-blocked route (code)
def blocked_energies(m):
    """(E2 by block, E3 pp, hh, ph) of the spin-blocked formulas."""
    A, B = 0, 1
    no, nv, eps = m.no, m.nv, m.eps
    OC = lambda s: range(no[s])  # noqa: E731
    V = lambda s: range(no[s], no[s] + nv[s])  # noqa: E731
    g = m.g

    def den(s1, s2, i, j, a, b):
        return eps[s1][i] + eps[s2][j] - eps[s1][a] - eps[s2][b]

    def k_same(s, i, j, a, b):
        return g(s, i, a, s, j, b) - g(s, i, b, s, j, a)

    t, K = {}, {}
    for s in (A, B):
        for i, j, a, b in itertools.product(OC(s), OC(s), V(s), V(s)):
            K[s, s, i, j, a, b] = k_same(s, i, j, a, b)
            t[s, s, i, j, a, b] = K[s, s, i, j, a, b] / den(s, s, i, j, a, b)
    for i, j, a, b in itertools.product(OC(A), OC(B), V(A), V(B)):
        K[A, B, i, j, a, b] = g(A, i, a, B, j, b)
        t[A, B, i, j, a, b] = K[A, B, i, j, a, b] / den(A, B, i, j, a, b)

    def tt(s1, s2, i, j, a, b):
        return t[s1, s2, i, j, a, b]

    e2 = {
        "aa": sum(t[k] * K[k] for k in t if k[:2] == (A, A)) / 4,
        "bb": sum(t[k] * K[k] for k in t if k[:2] == (B, B)) / 4,
        "ab": sum(t[k] * K[k] for k in t if k[:2] == (A, B)),
    }
    ladder = sp.Rational(1, 2) if MUTATION == "ladder_half" else 1

    def same_spin(s):
        o = 1 - s
        pp, hh, ph = {}, {}, {}
        for i, j, a, b in itertools.product(OC(s), OC(s), V(s), V(s)):
            pp[i, j, a, b] = ladder * sum(
                g(s, a, c, s, b, d) * tt(s, s, i, j, c, d)
                for c, d in itertools.product(V(s), V(s))
            )
            hh[i, j, a, b] = sum(
                g(s, k, i, s, ll, j) * tt(s, s, k, ll, a, b)
                for k, ll in itertools.product(OC(s), OC(s))
            )

        def x(i, a, j, b):
            v = sum(
                (g(s, k, c, s, j, b) - g(s, k, j, s, b, c)) * tt(s, s, i, k, a, c)
                for k in OC(s)
                for c in V(s)
            )
            # mixed ring: t^{ab}_{iKaC} for s = alpha, t^{ab}_{kIcA} for beta
            for k in OC(o):
                for c in V(o):
                    amp = tt(A, B, i, k, a, c) if s == A else tt(A, B, k, i, c, a)
                    v += g(o, k, c, s, j, b) * amp
            return v

        for i, j, a, b in itertools.product(OC(s), OC(s), V(s), V(s)):
            ph[i, j, a, b] = (
                x(i, a, j, b) - x(j, a, i, b) - x(i, b, j, a) + x(j, b, i, a)
            )
        return pp, hh, ph

    def opposite_spin():
        pp, hh, ph = {}, {}, {}
        for i, J, a, Bv in itertools.product(OC(A), OC(B), V(A), V(B)):
            pp[i, J, a, Bv] = sum(
                g(A, a, c, B, Bv, D) * tt(A, B, i, J, c, D) for c in V(A) for D in V(B)
            )
            hh[i, J, a, Bv] = sum(
                g(A, k, i, B, L, J) * tt(A, B, k, L, a, Bv)
                for k in OC(A)
                for L in OC(B)
            )
            v = 0
            for k in OC(A):
                for c in V(A):
                    v += g(A, k, c, B, J, Bv) * tt(A, A, i, k, a, c)
                    v += (g(A, i, a, A, k, c) - g(A, i, k, A, a, c)) * tt(
                        A, B, k, J, c, Bv
                    )
            for K_ in OC(B):
                for C in V(B):
                    v += (g(B, K_, C, B, J, Bv) - g(B, K_, J, B, Bv, C)) * tt(
                        A, B, i, K_, a, C
                    )
                    v += g(A, i, a, B, K_, C) * tt(B, B, J, K_, Bv, C)
            for k in OC(A):
                for C in V(B):
                    v -= g(A, k, i, B, Bv, C) * tt(A, B, k, J, a, C)
            if MUTATION != "drop_ab_exchange":
                for K_ in OC(B):
                    for c in V(A):
                        v -= g(A, a, c, B, K_, J) * tt(A, B, i, K_, c, Bv)
            ph[i, J, a, Bv] = v
        return pp, hh, ph

    parts = {(A, A): same_spin(A), (B, B): same_spin(B), (A, B): opposite_spin()}
    e3 = []
    for term in range(3):
        tot = 0
        for (s1, s2), terms in parts.items():
            w = sp.Rational(1, 4) if s1 == s2 else 1
            for (i, j, a, b), r in terms[term].items():
                tot += w * K[s1, s2, i, j, a, b] * r / den(s1, s2, i, j, a, b)
        e3.append(tot)
    return e2, e3


# ------------------------------------------------------------------ S checks
def symbolic_checks():
    no, nv = (3, 2), (2, 3)
    r = sp.Rational
    eps = [
        [r(-17, 10), r(-13, 10), r(-11, 10), r(3, 10), r(9, 10)],
        [r(-8, 5), r(-6, 5), r(1, 2), r(7, 10), r(7, 5)],
    ]
    m = symbolic_model(no, nv, eps)
    e2_so, pp, hh, ph = so_energies(m)
    e2, e3 = blocked_energies(m)
    d2 = sp.expand(e2_so - sum(e2.values()))
    check(
        "S1 E2: aa + bb + ab == 1/4 sum <ij||ab> t (claim S1)",
        d2 == 0,
        f"residual {d2}",
    )
    for label, so, bl in (
        ("pp ladders == 1/8 sum t <ab||cd> t", pp, e3[0]),
        ("hh ladders == 1/8 sum <kl||ij> t t", hh, e3[1]),
        ("ph rings + exchange rings == sum t <kb||cj> t", ph, e3[2]),
    ):
        d = sp.expand(so - bl)
        n_terms = len(sp.Add.make_args(sp.expand(so)))
        check(
            f"S2 E3 {label} (claim S2)",
            d == 0,
            f"{n_terms} monomials; residual terms {len(sp.Add.make_args(d)) if d != 0 else 0}",
        )


# ------------------------------------------------------------------ N checks
def apply_annihilate(det, p):
    if not (det >> p) & 1:
        return None, 0
    sign = -1 if bin(det & ((1 << p) - 1)).count("1") % 2 else 1
    return det & ~(1 << p), sign


def apply_create(det, p):
    if (det >> p) & 1:
        return None, 0
    sign = -1 if bin(det & ((1 << p) - 1)).count("1") % 2 else 1
    return det | (1 << p), sign


def brute_force(m):
    """RS-PT E2, E3 in the full N-electron determinant space."""
    orbs = [(s, p) for s in (0, 1) for p in range(m.no[s] + m.nv[s])]
    n, nel = len(orbs), sum(m.no)
    occ = [k for k, (s, p) in enumerate(orbs) if p < m.no[s]]
    eps = np.array([m.eps[s][p] for s, p in orbs])
    gso = np.zeros((n,) * 4)
    for P, Q, R, S in itertools.product(range(n), repeat=4):
        gso[P, Q, R, S] = phys(m, orbs[P], orbs[Q], orbs[R], orbs[S])
    # h chosen so that the reference Fock operator is diag(eps)
    h = np.diag(eps) - np.einsum("pkqk->pq", gso[:, occ][:, :, :, occ])
    nonzero = [
        (*pqrs, gso[pqrs])
        for pqrs in itertools.product(range(n), repeat=4)
        if gso[pqrs] != 0
    ]
    dets = [sum(1 << k for k in c) for c in itertools.combinations(range(n), nel)]
    index = {d: x for x, d in enumerate(dets)}
    H = np.zeros((len(dets), len(dets)))
    for col, d in enumerate(dets):
        for p, q in itertools.product(range(n), repeat=2):
            if h[p, q] == 0:
                continue
            d1, s1 = apply_annihilate(d, q)
            if d1 is None:
                continue
            d2, s2 = apply_create(d1, p)
            if d2 is not None:
                H[index[d2], col] += s1 * s2 * h[p, q]
        for p, q, r, s, val in nonzero:
            x, sg = d, 1
            for op, k in (
                (apply_annihilate, r),
                (apply_annihilate, s),
                (apply_create, q),
                (apply_create, p),
            ):
                x, si = op(x, k)
                if x is None:
                    break
                sg *= si
            if x is not None:
                H[index[x], col] += 0.25 * sg * val
    h0 = np.array([sum(eps[k] for k in range(n) if (d >> k) & 1) for d in dets])
    ref = index[sum(1 << k for k in occ)]
    Vm = H - np.diag(h0)
    e1 = Vm[ref, ref]
    v0 = Vm[:, ref].copy()
    v0[ref] = 0.0
    denom = h0[ref] - h0
    denom[ref] = np.inf
    psi1 = v0 / denom
    e2 = v0 @ psi1
    e3 = psi1 @ (Vm - e1 * np.eye(len(dets))) @ psi1
    return float(e2), float(e3), len(dets)


def numeric_checks():
    rng = np.random.default_rng(20261114)
    m = random_model((3, 2), (2, 3), rng)
    e2_bf, e3_bf, ndet = brute_force(m)
    e2, e3 = blocked_energies(m)
    e2_bl, e3_bl = float(sum(e2.values())), float(sum(e3))
    # Both routes are double precision on OC(1e-3) energies with ~1e4 terms:
    # agreement to 1e-12 is the rounding floor with two decades of room.
    tol = 1e-12
    check(
        f"N E2 spin-blocked == brute-force RS-PT ({ndet} dets) (claim N2)",
        abs(e2_bl - e2_bf) < tol,
        f"{e2_bl:.15f} vs {e2_bf:.15f}",
    )
    check(
        f"N E3 spin-blocked == brute-force RS-PT ({ndet} dets) (claim N3)",
        abs(e3_bl - e3_bf) < tol,
        f"{e3_bl:.15f} vs {e3_bf:.15f}",
    )
    check("N E3 is not trivially small", abs(e3_bf) > 1e-6, f"|E3| {abs(e3_bf):.2e}")


def main() -> int:
    if MUTATION:
        print(f"MUTATION {MUTATION}")
    symbolic_checks()
    numeric_checks()
    if failures:
        print(f"{failures} check(s) FAILED")
        return 1
    print("all identities hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
