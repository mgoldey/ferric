#!/usr/bin/env python3
"""Exact checks of the fermionic sign conventions in ferric-ci's Slater-Condon code.

Theory and notation: crates/ferric-ci/theory.md. Every claim asserts; the
script exits nonzero on any mismatch. Negative controls (deliberately wrong
variants that must NOT pass) guard against a vacuous pass.

Convention under test (ferric-ci, crates/ferric-ci/src):
  * spin-orbital order is alpha-before-beta: JW mode p = (p, alpha),
    mode n + p = (p, beta), p = 0..n-1;
  * |D> = prod_{p in A, ascending} a+_{p alpha} prod_{q in B, ascending}
    a+_{q beta} |0>, A/B the alpha/beta u64 strings;
  * a single-channel annihilation/creation at orbital p carries
    (-1)^{#occupied below p IN THAT CHANNEL'S STRING}
    (strings.rs:82 phase_below), i.e. the alpha electrons are NOT counted
    for a beta operator.

Claims
  1  The Jordan-Wigner matrices a_P = Z x ... x Z x sigma^- x I x ... x I
     satisfy the CAR {a_P, a+_Q} = delta_PQ, {a_P, a_Q} = 0 (n_so = 8).
  2  The ascending-creation product of claim "convention" applied to |0> is
     the computational basis vector with coefficient +1 (so bit strings ARE
     determinants with no extra phase), and a_P on it has sign
     (-1)^{#occupied modes < P}.
  3  For every (bra, ket) pair in the sectors (n, N_alpha, N_beta) =
     (4,2,2), (5,3,2), (5,2,3), (5,3,3), the full spin-orbital matrix element
       <D'| sum h_pq a+_{p s} a_{q s}
            + 1/2 sum (pq|rs) a+_{p s} a+_{r t} a_{s t} a_{q s} |D>
     built from the explicit JW matrices with SYMBOLIC h_pq = h_qp and
     8-fold-symmetric (pq|rs) equals, as an exact linear form in those
     symbols, a line-for-line port of
       hamiltonian.rs:128 general_element
         -> :63  diagonal_element
         -> :190 single_excitation_same_spin   (phase :206-209)
         -> :231 double_same_spin              (phase :282 double_phase_same_spin)
         -> :252 double_opposite_spin          (phase :266-273)
     The odd-N_alpha sectors are the ones that test the omission of the alpha
     parity from beta operators (it cancels pairwise); with N_alpha even it
     would be invisible.
  4  Negative controls, each must FAIL claim 3 somewhere:
       a  same-spin double, create a before b (swap of :291/:293);
       b  single excitation, phase_a taken in the full ket instead of
          ket-minus-i (:208);
       c  beta single excitation multiplied by (-1)^{N_alpha} only once
          (a non-cancelling alpha parity);
       d  opposite-spin double with the sign dropped from the beta channel;
       e  the same ferric port compared against an INTERLEAVED JW order
          (mode 2p + spin). This is a different but equally valid determinant
          phase convention: it differs from alpha-before-beta by a diagonal
          +-1 similarity, so it must mismatch element-wise yet give the same
          spectrum. That is checked numerically below, which shows claim 3
          is convention-specific (strictly stronger than an energy test).

Run: python3 scripts/casci_slater_condon_phases_sympy.py
"""

import itertools
import sys

import numpy as np
import sympy as sp

FAILED = []


def check(cond, msg):
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        FAILED.append(msg)


# ---------------------------------------------------------------- JW matrices
def jw_annihilators(n_so):
    """Dense integer JW matrices a_P, P = 0..n_so-1, mode 0 = lowest bit."""
    I2 = np.eye(2, dtype=np.int64)
    Z = np.diag([1, -1]).astype(np.int64)
    # basis |n> with bit P of the state index = occupation of mode P;
    # sigma^- = |0><1| on one mode.
    sm = np.array([[0, 1], [0, 0]], dtype=np.int64)
    ops = []
    for P in range(n_so):
        # kron ordering: the LAST factor is bit 0. Build factors for bits
        # n_so-1 .. 0 left to right.
        factors = []
        for bit in reversed(range(n_so)):
            if bit < P:
                factors.append(Z)
            elif bit == P:
                factors.append(sm)
            else:
                factors.append(I2)
        m = factors[0]
        for f in factors[1:]:
            m = np.kron(m, f)
        ops.append(m)
    return ops


def to_sparse(m):
    """Each JW operator column has <= 1 nonzero: col -> (row, sign)."""
    out = {}
    rows, cols = np.nonzero(m)
    for r, c in zip(rows, cols):
        assert c not in out
        out[int(c)] = (int(r), int(m[r, c]))
    return out


class JW:
    def __init__(self, n_so):
        dense = jw_annihilators(n_so)
        self.n_so = n_so
        self.dense = dense
        self.ann = [to_sparse(a) for a in dense]
        self.cre = [to_sparse(a.T) for a in dense]

    def apply(self, ops, state):
        """ops applied right-to-left: ops = [(kind, P), ...] as written."""
        sign = 1
        for kind, P in reversed(ops):
            table = self.ann[P] if kind == "a" else self.cre[P]
            if state not in table:
                return None, 0
            state, s = table[state]
            sign *= s
        return state, sign


# ---------------------------------------------------- symbolic integrals
def hsym(p, q):
    p, q = min(p, q), max(p, q)
    return sp.Symbol(f"h_{p}{q}")


def gkey(p, q, r, s):
    a, b = (p, q) if p <= q else (q, p)
    c, d = (r, s) if r <= s else (s, r)
    return min((a, b, c, d), (c, d, a, b))


def gsym(p, q, r, s):
    k = gkey(p, q, r, s)
    return sp.Symbol("g_" + "".join(map(str, k)))


# -------------------------------------------- ferric port (hamiltonian.rs)
def phase_below(s, p):  # strings.rs:82
    return -1 if bin(s & ((1 << p) - 1)).count("1") % 2 else 1


def occ(s):
    return [p for p in range(64) if s >> p & 1]


def diagonal_element(ao, bo):  # hamiltonian.rs:63
    e = 0
    for p in ao + bo:
        e += hsym(p, p)
    for occl in (ao, bo):
        for i in range(len(occl)):
            for j in range(i + 1, len(occl)):
                p, q = occl[i], occl[j]
                e += gsym(p, p, q, q) - gsym(p, q, q, p)
    for p in ao:
        for q in bo:
            e += gsym(p, p, q, q)
    return e


def single(bra, ket, other, mut):  # hamiltonian.rs:190
    diff = bra ^ ket
    i = (diff & ket).bit_length() - 1
    a = (diff & bra).bit_length() - 1
    phase_i = phase_below(ket, i)
    if mut == "b":
        phase_a = phase_below(ket, a)
    else:
        phase_a = phase_below(ket & ~(1 << i), a)
    val = hsym(a, i)
    spect = ket & ~(1 << i) & ~(1 << a)
    for p in occ(spect):
        val += gsym(a, i, p, p) - gsym(a, p, p, i)
    for p in occ(other):
        val += gsym(a, i, p, p)
    return phase_i * phase_a * val


def double_same(bra, ket, mut):  # hamiltonian.rs:231, :282
    diff = bra ^ ket
    i, j = occ(diff & ket)
    a, b = occ(diff & bra)
    s = ket
    ph = phase_below(s, i)
    s &= ~(1 << i)
    ph *= phase_below(s, j)
    s &= ~(1 << j)
    first, second = (a, b) if mut == "a" else (b, a)
    ph *= phase_below(s, first)
    s |= 1 << first
    ph *= phase_below(s, second)
    return ph * (gsym(a, i, b, j) - gsym(a, j, b, i))


def double_opp(a_bra, a_ket, b_bra, b_ket, mut):  # hamiltonian.rs:252
    da, db = a_bra ^ a_ket, b_bra ^ b_ket
    i = (da & a_ket).bit_length() - 1
    a = (da & a_bra).bit_length() - 1
    j = (db & b_ket).bit_length() - 1
    b = (db & b_bra).bit_length() - 1
    pa = phase_below(a_ket, i) * phase_below(a_ket & ~(1 << i), a)
    pb = phase_below(b_ket, j) * phase_below(b_ket & ~(1 << j), b)
    if mut == "d":
        pb = 1
    return pa * pb * gsym(a, i, b, j)


def general_element(a_bra, b_bra, a_ket, b_ket, mut=None):  # hamiltonian.rs:128
    da = bin(a_bra ^ a_ket).count("1")
    db = bin(b_bra ^ b_ket).count("1")
    nd = da + db
    if nd == 0:
        return diagonal_element(occ(a_ket), occ(b_ket))
    if nd == 2:
        if da == 2:
            return single(a_bra, a_ket, b_ket, mut)
        if db == 2:
            v = single(b_bra, b_ket, a_ket, mut)
            if mut == "c" and bin(a_ket).count("1") % 2:
                v = -v
            return v
        return 0
    if nd == 4:
        if da == 4:
            return double_same(a_bra, a_ket, mut)
        if db == 4:
            return double_same(b_bra, b_ket, mut)
        if da == 2 and db == 2:
            return double_opp(a_bra, a_ket, b_bra, b_ket, mut)
        return 0
    return 0


# ------------------------------------------------ JW Hamiltonian columns
def mode(p, spin, n, order):
    return p + n * spin if order == "ab" else 2 * p + spin


def encode(a, b, n, order):
    st = 0
    for p in occ(a):
        st |= 1 << mode(p, 0, n, order)
    for p in occ(b):
        st |= 1 << mode(p, 1, n, order)
    return st


def h_column(jw, n, ket_state, order):
    """H|ket> as {state: sympy linear form} from the explicit JW operators."""
    col = {}

    def add(st, coef):
        col[st] = col.get(st, 0) + coef

    spins = (0, 1)
    for s_ in spins:
        for p in range(n):
            for q in range(n):
                P, Q = mode(p, s_, n, order), mode(q, s_, n, order)
                st, sg = jw.apply([("c", P), ("a", Q)], ket_state)
                if sg:
                    add(st, sg * hsym(p, q))
    half = sp.Rational(1, 2)
    for s1, s2 in itertools.product(spins, spins):
        for p, q, r, s in itertools.product(range(n), repeat=4):
            P, Q = mode(p, s1, n, order), mode(q, s1, n, order)
            R, S = mode(r, s2, n, order), mode(s, s2, n, order)
            st, sg = jw.apply([("c", P), ("c", R), ("a", S), ("a", Q)], ket_state)
            if sg:
                add(st, half * sg * gsym(p, q, r, s))
    return col


def strings(n, k):
    return [sum(1 << p for p in c) for c in itertools.combinations(range(n), k)]


def compare_sector(n, na, nb, order="ab", mut=None):
    """Return number of mismatching (bra, ket) elements (0 = identical)."""
    jw = JW(2 * n)
    dets = [(a, b) for a in strings(n, na) for b in strings(n, nb)]
    bad = 0
    for ak, bk in dets:
        col = h_column(jw, n, encode(ak, bk, n, order), order)
        for ab, bb in dets:
            jw_val = col.get(encode(ab, bb, n, order), 0)
            ferric = general_element(ab, bb, ak, bk, mut)
            if sp.expand(jw_val - ferric) != 0:
                bad += 1
    return bad, len(dets)


# ------------------------------------------------------------------ claims
def claim1():
    print("\n=== 1  JW matrices satisfy the CAR (n_so = 8)")
    a = [m.astype(float) for m in jw_annihilators(8)]
    ok = True
    eye = np.eye(a[0].shape[0])
    for P, Q in itertools.product(range(8), repeat=2):
        acr = a[P] @ a[Q].T + a[Q].T @ a[P]
        aa = a[P] @ a[Q] + a[Q] @ a[P]
        ok &= np.array_equal(acr, eye if P == Q else 0 * eye)
        ok &= not aa.any()
    check(ok, "{a_P, a+_Q} = delta_PQ and {a_P, a_Q} = 0 for all P, Q")


def claim2():
    print("\n=== 2  ascending-creation determinant = basis vector, sign rule")
    n = 3
    jw = JW(2 * n)
    ok_vec, ok_sign = True, True
    for a in range(1 << n):
        for b in range(1 << n):
            modes = [p for p in occ(a)] + [n + q for q in occ(b)]
            st, sg = jw.apply([("c", P) for P in modes], 0)  # ascending left->right
            ok_vec &= st == encode(a, b, n, "ab") and sg == 1
            for P in modes:
                st2, sg2 = jw.apply([("a", P)], st)
                ok_sign &= sg2 == (-1) ** bin(st & ((1 << P) - 1)).count("1")
    check(ok_vec, "prod_{asc} a+ |0> = +|bits> for every (A, B), alpha-before-beta")
    check(ok_sign, "a_P |bits> sign = (-1)^{#occupied modes < P}")


SECTORS = [(4, 2, 2), (5, 3, 2), (5, 2, 3), (5, 3, 3)]


def claim3():
    print("\n=== 3  general_element == explicit JW Hamiltonian (symbolic, exact)")
    for n, na, nb in SECTORS:
        bad, nd = compare_sector(n, na, nb)
        check(bad == 0, f"n={n} (Na,Nb)=({na},{nb}): {nd}^2 elements, {bad} mismatches")


def claim4():
    print("\n=== 4  negative controls (each must mismatch)")
    for mut, what in [
        ("a", "same-spin double, create a before b"),
        ("b", "single, phase_a in full ket"),
        ("c", "beta single carries uncancelled (-1)^{N_alpha}"),
        ("d", "opposite-spin double, beta sign dropped"),
    ]:
        bad = sum(compare_sector(n, na, nb, mut=mut)[0] for n, na, nb in SECTORS)
        check(bad > 0, f"mutation {mut} ({what}): {bad} mismatches")

    bad, _ = compare_sector(5, 3, 2, order="interleaved")
    check(bad > 0, f"interleaved JW order mismatches element-wise ({bad} elements)")

    # ...but its spectrum is the same: it is a diagonal +-1 similarity.
    n, na, nb = 5, 3, 2
    rng = np.random.default_rng(7)
    hnum = rng.standard_normal((n, n))
    hnum = hnum + hnum.T
    gnum = rng.standard_normal((n,) * 4)
    gnum = sum(
        np.transpose(gnum, perm)
        for perm in [
            (0, 1, 2, 3),
            (1, 0, 2, 3),
            (0, 1, 3, 2),
            (1, 0, 3, 2),
            (2, 3, 0, 1),
            (3, 2, 0, 1),
            (2, 3, 1, 0),
            (3, 2, 1, 0),
        ]
    )
    subs = {}
    for p, q in itertools.product(range(n), repeat=2):
        subs[hsym(p, q)] = float(hnum[p, q])
    for idx in itertools.product(range(n), repeat=4):
        subs[gsym(*idx)] = float(gnum[idx])
    dets = [(a, b) for a in strings(n, na) for b in strings(n, nb)]
    jw = JW(2 * n)

    def matrix(order):
        m = np.zeros((len(dets), len(dets)))
        for k, (ak, bk) in enumerate(dets):
            col = h_column(jw, n, encode(ak, bk, n, order), order)
            for b_, (ab, bb) in enumerate(dets):
                v = col.get(encode(ab, bb, n, order), 0)
                m[b_, k] = float(sp.sympify(v).xreplace(subs))
        return m

    m_ab, m_il = matrix("ab"), matrix("interleaved")
    sim = np.abs(np.abs(m_ab) - np.abs(m_il)).max()
    dspec = np.abs(np.linalg.eigvalsh(m_ab) - np.linalg.eigvalsh(m_il)).max()
    check(sim < 1e-12, f"|H_ab| == |H_interleaved| element-wise (max {sim:.1e})")
    check(dspec < 1e-10, f"same spectrum (max eigenvalue diff {dspec:.1e})")


def main():
    claim1()
    claim2()
    claim3()
    claim4()
    if FAILED:
        print(f"\n{len(FAILED)} FAILED")
        sys.exit(1)
    print("\nall claims hold")


if __name__ == "__main__":
    main()
