#!/usr/bin/env python3
"""Sympy proof of the OP-MP3 identities that ferric-mp2/src/opmp3.rs relies on.

Paper: K. Bystrom, T. C. Berkelbach, D. Hait, arXiv:2609.25330 (equation
numbers below are theirs). Code under test: crates/ferric-mp2/src/opmp3.rs
(W channels, matrix Pade, models). Derivation:
`book/src/10-mp2/11-opmp3.md#opmp3`. Conventions: spin orbitals, real; <pq|rs> = (pr|qs) when the
spins match; <pq||rs> = <pq|rs> - <pq|sr>; Delta_ijab = e_a + e_b - e_i - e_j.

  W_ij = 1/8 sum_kab ( c_ikab <jk||ab> + c_jkab <ik||ab> )        (Eq. 11a)
  P(A, B) = A (A - B)^-1 A                                         (Eq. 13)

Identities (each an exact sympy identity unless marked numerical):

  (a) Eq. 18 split. In a spin-orbital model (alpha 2 occ / 2 vir, beta
      2 occ / 1 vir), every amplitude and integral an independent symbol
      with only the antisymmetry and spin conservation of the spin-orbital
      tensors, Eq. 11a gives:
        a1  W_{i alpha, J beta} = 0;
        a2  the k-in-sigma part of the sigma block equals the spin-blocked
            same-spin form 1/8 sum c^ss K^ss + (i<->j) (Eq. 17);
        a3  the k-in-sigma-bar part equals Eq. 18 written with the stored
            alpha-first block: alpha  1/4 sum_KaB c_iKaB (ja|KB) + (i<->j),
            beta  1/4 sum_kaB c_kIaB (ka|JB) + (I<->J), plain integrals;
        a4  Tr W^SS = 1/4 sum c^aa K^aa + 1/4 sum c^bb K^bb and
            Tr W^OS = sum c^ab K^ab (the u_mp3 channel energies).
  (b) Pade series. For symbolic symmetric 2x2 A, B and a scale t:
        b1  P(A, tB) = A + tB + t^2 B A^-1 B + O(t^3), elementwise;
        b2  1x1: P(a, b) = a^2/(a - b), the scalar Pade of Eq. 1;
        b3  commuting A, B (one rotation U, rational 3/5, 4/5):
            Tr P = sum_i alpha_i^2 / (alpha_i - beta_i).
  (c) Size consistency. A = A1 (+) A2, B = B1 (+) B2 (2x2 (+) 1x1 symbols):
        c1  Tr P(A, B) = Tr P(A1, B1) + Tr P(A2, B2);
        c2  the scalar Pade of the summed traces is NOT additive: the
            symbolic difference is non-zero, and the counterexample
            A1 = -1, B1 = -1/2, A2 = -1, B2 = 0 gives -8/3 vs -3;
        c3  identical copies (A2 = A1, B2 = B1): the scalar Pade IS additive
            (Eq. 3, size-extensive only for identical copies).
  (d) Traces. For any amplitudes c antisymmetric in ij and in ab:
        d1  Tr W = 1/4 sum c <ij||ab> (Eq. 11b), symbolically
            (spin-orbital model, 3 occ, 2 vir);
        d2  NUMERICAL, independent route: a random antisymmetrized
            Hamiltonian in 6 spin orbitals (3 occupied; one-body part chosen
            so that the Fock operator is diag(e)); brute-force
            Rayleigh-Schroedinger E2 and E3 = <psi1|V - E1|psi1> in the
            20-determinant space against Tr W^(2) and Tr W^(3) from c^(1)
            (Eq. 7) and c^(2) (Eq. 9).
  (N) NUMERICAL, independent route for (b)/(c): Tr P of random symmetric
      W2 (negative definite), W3 equals the paper's eigen route
      (analysis_code/resummation.py): sum_i (S^-1 A S)_ii / (1 - zeta_i),
      zeta = eig(A^-1 B); and is additive over a random block-diagonal pair.

Mutations (set OPMP3_MUTATION), each turns the script red:
  os_eighth         Eq. 18 with prefactor 1/8 instead of 1/4 (the dropped
                    factor 2 of the two virtual-spin orderings): a3, a4 fail.
  pade_swapped      P = (A - B) A^-1 (A - B): b1, b2, b3 and N (eigen route)
                    fail; c1 holds (the swapped form is still a matrix
                    function, so still block-additive).
  scalar_in_block   P = I (Tr A)^2 / (Tr A - Tr B) / n per block (the scalar
                    Pade inside a block): b1, b3, c1 and both N checks fail.
  w_unsymmetrized   W_ij = 1/8 sum c_ikab <jk||ab> without the (i<->j) half:
                    a2, a3, a4, d1 and d2 fail.

Measured 2026-11-14 (4.7 s): the unmutated run exits 0 (d2: RS E2
-0.123479477730 and E3 0.003329488158 equal Tr W^(2), Tr W^(3) to the
printed 12 digits; N agrees to 2.8e-16); each mutation exits 1 with exactly
the checks listed above red.

Exit code 0 iff every check passes; "all identities hold" is printed then.

Book boxes (book/src/10-mp2/11-opmp3.md) cite claim a3 (the Eq. 18 split),
claim b1 (the Pade series, with b2, b3, c1-c3 and N beside it) and claim d2
(the traces against Rayleigh-Schroedinger theory, with d1 beside it).
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import sympy as sp

MUTATION = os.environ.get("OPMP3_MUTATION", "")
MUTATIONS = ("", "os_eighth", "pade_swapped", "scalar_in_block", "w_unsymmetrized")
if MUTATION not in MUTATIONS:
    sys.exit(f"unknown OPMP3_MUTATION {MUTATION!r}")

failures = 0


def check(label, ok, detail=""):
    global failures
    print(f"{'ok  ' if ok else 'FAIL'} {label}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        failures += 1


def is_zero(expr) -> bool:
    return sp.expand(expr) == 0


# ------------------------------------------------------------------ W build
def w_from(c, g, occ):
    """Eq. 11a on spin-orbital dicts: c[(i,k,a,b)], g[(j,k,a,b)] = <jk||ab>."""
    keys = {}
    for (i, k, a, b), v in c.items():
        keys.setdefault(i, []).append(((k, a, b), v))
    w = {}
    for i in occ:
        for j in occ:
            y_ij = sum(v * g.get((j,) + kab, 0) for kab, v in keys.get(i, []))
            y_ji = sum(v * g.get((i,) + kab, 0) for kab, v in keys.get(j, []))
            sym = 0 if MUTATION == "w_unsymmetrized" else y_ji
            w[(i, j)] = sp.Rational(1, 8) * (y_ij + sym)
    return w


class SpinModel:
    """Spin orbitals ('a'|'b', index); amplitudes and integrals from blocks."""

    def __init__(self, no, nv):
        self.no, self.nv = no, nv  # per spin
        self.occ = [(s, i) for s in "ab" for i in range(no[s])]
        self.vir = [(s, a) for s in "ab" for a in range(nv[s])]
        # amplitude blocks: same-spin antisymmetric, alpha-beta independent
        self.c_ss = {s: self._antisym("c" + s, no[s], nv[s]) for s in "ab"}
        self.c_ab = {
            (i, j, a, b): sp.Symbol(f"cab_{i}{j}{a}{b}")
            for i in range(no["a"])
            for j in range(no["b"])
            for a in range(nv["a"])
            for b in range(nv["b"])
        }
        # spatial (ia|jb): same spin symmetric under pair swap, mixed free
        self.g = {}
        for s in "ab":
            for i, a, j, b in itertools.product(
                range(no[s]), range(nv[s]), range(no[s]), range(nv[s])
            ):
                p, q = sorted([(i, a), (j, b)])
                self.g[(s, i, a, s, j, b)] = sp.Symbol(
                    f"g{s}{s}_{p[0]}{p[1]}{q[0]}{q[1]}"
                )
        for i, a, j, b in itertools.product(
            range(no["a"]), range(nv["a"]), range(no["b"]), range(nv["b"])
        ):
            sym = sp.Symbol(f"gab_{i}{a}{j}{b}")
            self.g[("a", i, a, "b", j, b)] = sym
            self.g[("b", j, b, "a", i, a)] = sym

    @staticmethod
    def _antisym(name, no, nv):
        out = {}
        for i, j in itertools.combinations(range(no), 2):
            for a, b in itertools.combinations(range(nv), 2):
                x = sp.Symbol(f"{name}_{i}{j}{a}{b}")
                out[(i, j, a, b)] = x
                out[(j, i, a, b)] = -x
                out[(i, j, b, a)] = -x
                out[(j, i, b, a)] = x
        return out

    def chem(self, p, a, q, b):
        """(pa|qb) of spin orbitals; zero unless spin(p)=spin(a), spin(q)=spin(b)."""
        if p[0] != a[0] or q[0] != b[0]:
            return 0
        return self.g[(p[0], p[1], a[1], q[0], q[1], b[1])]

    def asym(self, p, q, a, b):
        """<pq||ab> = (pa|qb) - (pb|qa)."""
        return self.chem(p, a, q, b) - self.chem(p, b, q, a)

    def t(self, i, j, a, b):
        """Spin-orbital amplitude from the stored blocks."""
        si, sj, sa, sb = i[0], j[0], a[0], b[0]
        if si == sj == sa == sb:
            return self.c_ss[si].get((i[1], j[1], a[1], b[1]), 0)
        if sorted([si, sj]) != ["a", "b"] or sorted([sa, sb]) != ["a", "b"]:
            return 0
        sign = 1
        if si == "b":
            i, j, sign = j, i, -sign
        if sa == "b":
            a, b, sign = b, a, -sign
        return sign * self.c_ab[(i[1], j[1], a[1], b[1])]

    def spin_orbital_tensors(self):
        c, g = {}, {}
        for i, k in itertools.product(self.occ, repeat=2):
            for a, b in itertools.product(self.vir, repeat=2):
                tv = self.t(i, k, a, b)
                if tv != 0:
                    c[(i, k, a, b)] = tv
                gv = self.asym(i, k, a, b)
                if gv != 0:
                    g[(i, k, a, b)] = gv
        return c, g


def prove_a_and_d1():
    print("== (a) Eq. 18 split, (d1) trace, symbolic spin-orbital model")
    m = SpinModel({"a": 2, "b": 2}, {"a": 2, "b": 1})
    c, g = m.spin_orbital_tensors()
    # split Eq. 11a by the spin of k
    by_k = {}
    for ks in "ab":
        ck = {key: v for key, v in c.items() if key[1][0] == ks}
        by_k[ks] = w_from(ck, g, m.occ)
    full = w_from(c, g, m.occ)
    cross = [full[(i, j)] for i in m.occ for j in m.occ if i[0] != j[0]]
    check("a1 W_{i alpha, J beta} = 0", all(is_zero(x) for x in cross))
    os_pref = sp.Rational(1, 8) if MUTATION == "os_eighth" else sp.Rational(1, 4)
    no, nv = m.no, m.nv
    ok_ss = ok_os = True
    for s in "ab":
        sb = "b" if s == "a" else "a"
        for i, j in itertools.product(range(no[s]), repeat=2):
            # a2: same-spin block form, K^ss_jkab = (ja|kb) - (jb|ka)
            yss = 0
            for k, a, b in itertools.product(range(no[s]), range(nv[s]), range(nv[s])):
                kss = lambda p, q: m.g[(s, p, a, s, q, b)] - m.g[(s, p, b, s, q, a)]  # noqa: E731
                cik = m.c_ss[s].get((i, k, a, b), 0)
                cjk = m.c_ss[s].get((j, k, a, b), 0)
                yss += cik * kss(j, k) + cjk * kss(i, k)
            ok_ss &= is_zero(by_k[s][((s, i), (s, j))] - sp.Rational(1, 8) * yss)
            # a3: Eq. 18 with the alpha-first stored block
            yos = 0
            for k, a, b in itertools.product(
                range(no[sb]), range(nv["a"]), range(nv["b"])
            ):
                if s == "a":
                    yos += m.c_ab[(i, k, a, b)] * m.g[("a", j, a, "b", k, b)]
                    yos += m.c_ab[(j, k, a, b)] * m.g[("a", i, a, "b", k, b)]
                else:
                    yos += m.c_ab[(k, i, a, b)] * m.g[("a", k, a, "b", j, b)]
                    yos += m.c_ab[(k, j, a, b)] * m.g[("a", k, a, "b", i, b)]
            ok_os &= is_zero(by_k[sb][((s, i), (s, j))] - os_pref * yos)
    check("a2 k-in-sigma part == Eq. 17 spin-blocked SS", ok_ss)
    check("a3 k-in-sigma-bar part == Eq. 18 (plain integrals, 1/4)", ok_os)
    # a4: traces of the channels vs the u_mp3 channel energies
    tr_ss = sum(by_k[s][((s, i), (s, i))] for s in "ab" for i in range(no[s]))
    tr_os = sum(
        by_k["b" if s == "a" else "a"][((s, i), (s, i))]
        for s in "ab"
        for i in range(no[s])
    )
    e_ss = 0
    for s in "ab":
        for i, j, a, b in itertools.product(
            range(no[s]), range(no[s]), range(nv[s]), range(nv[s])
        ):
            kk = m.g[(s, i, a, s, j, b)] - m.g[(s, i, b, s, j, a)]
            e_ss += sp.Rational(1, 4) * m.c_ss[s].get((i, j, a, b), 0) * kk
    e_os = sum(v * m.g[("a", i, a, "b", j, b)] for (i, j, a, b), v in m.c_ab.items())
    # the traced Eq. 18 matrices (what opmp3.rs traces): each spin block
    # contributes os_pref * 2 * sum c^ab K^ab
    eq18_tr = os_pref * 4 * e_os
    check("a4 Tr W^SS == 1/4 sum c^aa K^aa + 1/4 sum c^bb K^bb", is_zero(tr_ss - e_ss))
    check(
        "a4 Tr W^OS == sum c^ab K^ab (Eq. 18 traced)",
        is_zero(tr_os - e_os) and is_zero(eq18_tr - e_os),
    )
    # d1: Tr W == 1/4 sum c <||> on the same model
    tr = sum(full[(i, i)] for i in m.occ)
    e = sum(sp.Rational(1, 4) * v * g.get(key, 0) for key, v in c.items())
    check("d1 Tr W (Eq. 11a) == 1/4 sum c <ij||ab> (Eq. 11b)", is_zero(tr - e))


# ------------------------------------------------------------------ Pade
def pade(a: sp.Matrix, b: sp.Matrix) -> sp.Matrix:
    if MUTATION == "pade_swapped":
        return (a - b) * a.inv() * (a - b)
    if MUTATION == "scalar_in_block":
        ta, tb = a.trace(), b.trace()
        return sp.eye(a.rows) * (ta * ta / (ta - tb)) / a.rows
    return a * (a - b).inv() * a


def sym2(name):
    x, y, z = sp.symbols(f"{name}11 {name}12 {name}22")
    return sp.Matrix([[x, y], [y, z]])


def prove_b():
    print("== (b) Pade series and the scalar limit")
    a, b, t = sym2("a"), sym2("b"), sp.Symbol("t")
    p = pade(a, t * b)
    target = a + t * b + t**2 * b * a.inv() * b
    ok = True
    for idx in range(4):
        ser = sp.series(p[idx], t, 0, 3).removeO()
        ok &= sp.simplify(ser - target[idx]) == 0
    check("b1 P(A, tB) = A + tB + t^2 B A^-1 B + O(t^3)", ok)
    x, y = sp.symbols("x y")
    p1 = pade(sp.Matrix([[x]]), sp.Matrix([[y]]))[0]
    check("b2 1x1 P(a, b) == a^2 / (a - b)", sp.simplify(p1 - x * x / (x - y)) == 0)
    u = sp.Matrix(
        [
            [sp.Rational(3, 5), -sp.Rational(4, 5)],
            [sp.Rational(4, 5), sp.Rational(3, 5)],
        ]
    )
    al = sp.symbols("al1 al2")
    be = sp.symbols("be1 be2")
    ac = u * sp.diag(*al) * u.T
    bc = u * sp.diag(*be) * u.T
    want = sum(al[i] ** 2 / (al[i] - be[i]) for i in range(2))
    check(
        "b3 commuting A, B: Tr P == sum alpha^2 / (alpha - beta)",
        sp.simplify(pade(ac, bc).trace() - want) == 0,
    )


def block_diag(m1: sp.Matrix, m2: sp.Matrix) -> sp.Matrix:
    return sp.diag(m1, m2)


def prove_c():
    print("== (c) size consistency of the matrix Pade; the scalar Pade fails")
    a1, b1 = sym2("a"), sym2("b")
    a2, b2 = sp.Matrix([[sp.Symbol("A")]]), sp.Matrix([[sp.Symbol("B")]])
    tr = pade(block_diag(a1, a2), block_diag(b1, b2)).trace()
    parts = pade(a1, b1).trace() + pade(a2, b2).trace()
    check("c1 Tr P(A1+A2, B1+B2) == Tr P1 + Tr P2", sp.simplify(tr - parts) == 0)

    def scalar(e2, e3):
        return e2 * e2 / (e2 - e3)

    x1, y1, x2, y2 = sp.symbols("x1 y1 x2 y2")
    diff = sp.simplify(scalar(x1 + x2, y1 + y2) - scalar(x1, y1) - scalar(x2, y2))
    vals = {x1: -1, y1: sp.Rational(-1, 2), x2: -1, y2: 0}
    whole = scalar(x1 + x2, y1 + y2).subs(vals)
    split = (scalar(x1, y1) + scalar(x2, y2)).subs(vals)
    check(
        "c2 scalar Pade of summed traces is not additive",
        diff != 0 and whole != split,
        f"counterexample: whole {whole}, fragments {split}",
    )
    same = sp.simplify(scalar(2 * x1, 2 * y1) - 2 * scalar(x1, y1))
    check("c3 scalar Pade is additive for identical copies (Eq. 3)", same == 0)


# ------------------------------------------------------------------ numerical
def np_pade(a, b):
    if MUTATION == "pade_swapped":
        return (a - b) @ np.linalg.solve(a, a - b)
    if MUTATION == "scalar_in_block":
        ta, tb = np.trace(a), np.trace(b)
        return np.eye(len(a)) * (ta * ta / (ta - tb)) / len(a)
    return a @ np.linalg.solve(a - b, a)


def paper_eigen_route(a, b):
    """analysis_code/resummation.py: sum_i (S^-1 A S)_ii / (1 - zeta_i)."""
    zeta, s = np.linalg.eig(np.linalg.solve(a, b))
    ct = np.diag(np.linalg.solve(s, a @ s))
    return float(np.real(np.sum(ct / (1.0 - zeta))))


def random_w(rng, n):
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    a = q @ np.diag(-rng.uniform(0.05, 0.4, n)) @ q.T
    m = rng.standard_normal((n, n)) * 0.03
    return a, m + m.T


def check_numerical_pade():
    print("== (N) Tr P vs the paper's eigen route; block additivity (numpy)")
    rng = np.random.default_rng(20261114)
    worst = 0.0
    for n in (1, 3, 5):
        a, b = random_w(rng, n)
        worst = max(worst, abs(np.trace(np_pade(a, b)) - paper_eigen_route(a, b)))
    check("N Tr P == eigen route (n = 1, 3, 5)", worst < 1e-12, f"max |d| {worst:.1e}")
    (a1, b1), (a2, b2) = random_w(rng, 3), random_w(rng, 4)
    z1, z2 = np.zeros((3, 4)), np.zeros((4, 3))
    a = np.block([[a1, z1], [z2, a2]])
    b = np.block([[b1, z1], [z2, b2]])
    d = np.trace(np_pade(a, b)) - np.trace(np_pade(a1, b1)) - np.trace(np_pade(a2, b2))
    check("N block-diagonal additivity", abs(d) < 1e-13, f"|d| {abs(d):.1e}")


# ------------------------------------------------------------------ d2: brute-force RS
NSO, NOCC = 6, 3


def random_hamiltonian(rng):
    """Antisymmetrized <pq||rs> from a real 8-fold-symmetric (pq|rs); h so
    that the Fock operator of the reference |0..NOCC-1> is diag(e)."""
    g = rng.standard_normal((NSO,) * 4) * 0.05
    g = g + g.transpose(1, 0, 2, 3)
    g = g + g.transpose(0, 1, 3, 2)
    g = g + g.transpose(2, 3, 0, 1)
    phys = g.transpose(0, 2, 1, 3)  # <pq|rs> = (pr|qs)
    asym = phys - phys.transpose(0, 1, 3, 2)
    e = np.sort(
        np.concatenate(
            [rng.uniform(-2.0, -0.5, NOCC), rng.uniform(0.3, 2.0, NSO - NOCC)]
        )
    )
    occ = range(NOCC)
    fock_2e = sum(asym[:, i, :, i] for i in occ)
    h = np.diag(e) - fock_2e
    return h, asym, e


def apply_ops(det, ops):
    """Apply a product of (dagger, p) right to left to bitstring det."""
    sign = 1
    for dag, p in reversed(ops):
        occ = (det >> p) & 1
        if dag == occ:
            return None, 0
        parity = bin(det & ((1 << p) - 1)).count("1")
        sign *= -1 if parity % 2 else 1
        det ^= 1 << p
    return det, sign


def hamiltonian_matrix(h, asym):
    dets = [sum(1 << p for p in c) for c in itertools.combinations(range(NSO), NOCC)]
    index = {d: n for n, d in enumerate(dets)}
    hm = np.zeros((len(dets), len(dets)))
    for col, d in enumerate(dets):
        for p, q in itertools.product(range(NSO), repeat=2):
            if h[p, q] != 0:
                nd, s = apply_ops(d, [(1, p), (0, q)])
                if nd is not None:
                    hm[index[nd], col] += s * h[p, q]
        for p, q, r, s4 in itertools.product(range(NSO), repeat=4):
            v = asym[p, q, r, s4]
            if v != 0:
                nd, sg = apply_ops(d, [(1, p), (1, q), (0, s4), (0, r)])
                if nd is not None:
                    hm[index[nd], col] += 0.25 * sg * v
    return hm, dets


def rs_e2_e3(h, asym, e):
    hm, dets = hamiltonian_matrix(h, asym)
    occ_e = [sum(e[p] for p in range(NSO) if (d >> p) & 1) for d in dets]
    h0 = np.diag(occ_e)
    v = hm - h0
    ref = dets.index(sum(1 << p for p in range(NOCC)))
    e0 = occ_e[ref]
    e1 = v[ref, ref]
    psi1 = np.zeros(len(dets))
    for n in range(len(dets)):
        if n != ref:
            psi1[n] = v[n, ref] / (e0 - occ_e[n])
    e2 = float(v[ref] @ psi1)
    e3 = float(psi1 @ (v - e1 * np.eye(len(dets))) @ psi1)
    return e2, e3


def amplitudes(asym, e):
    o, v = slice(0, NOCC), slice(NOCC, NSO)
    oovv = asym[o, o, v, v]
    eo, ev = e[o], e[v]
    delta = (
        ev[None, None, :, None]
        + ev[None, None, None, :]
        - eo[:, None, None, None]
        - eo[None, :, None, None]
    )
    c1 = -oovv / delta
    lad = 0.5 * np.einsum("abcd,ijcd->ijab", asym[v, v, v, v], c1)
    hh = 0.5 * np.einsum("klij,klab->ijab", asym[o, o, o, o], c1)
    r = np.einsum("kbcj,ikac->ijab", asym[o, v, v, o], c1)
    ring = (
        r - r.transpose(1, 0, 2, 3) - r.transpose(0, 1, 3, 2) + r.transpose(1, 0, 3, 2)
    )
    c2 = -(lad + hh + ring) / delta
    return oovv, c1, c2


def np_w(c, oovv):
    y = np.einsum("ikab,jkab->ij", c, oovv)
    return 0.125 * (y if MUTATION == "w_unsymmetrized" else y + y.T)


def check_numerical_traces():
    print("== (d2) Tr W^(2), Tr W^(3) vs brute-force Rayleigh-Schroedinger")
    rng = np.random.default_rng(7)
    h, asym, e = random_hamiltonian(rng)
    e2_rs, e3_rs = rs_e2_e3(h, asym, e)
    oovv, c1, c2 = amplitudes(asym, e)
    t2, t3 = np.trace(np_w(c1, oovv)), np.trace(np_w(c2, oovv))
    print(
        f"   RS E2 {e2_rs:.12f}  Tr W2 {t2:.12f}  RS E3 {e3_rs:.12f}  Tr W3 {t3:.12f}"
    )
    check("d2 Tr W^(2) == RS E2", abs(t2 - e2_rs) < 1e-12 * max(1.0, abs(e2_rs)))
    check("d2 Tr W^(3) == RS E3", abs(t3 - e3_rs) < 1e-12 * max(1.0, abs(e3_rs)))


def main() -> int:
    if MUTATION:
        print(f"MUTATION {MUTATION}")
    prove_a_and_d1()
    prove_b()
    prove_c()
    check_numerical_pade()
    check_numerical_traces()
    if failures:
        print(f"{failures} identity check(s) FAILED")
        return 1
    print("all identities hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
