#!/usr/bin/env python3
"""Identities behind ferric-cc's spin-blocked UCCSD and UCCSD(T).

Theory and notation: `book/src/16-coupled-cluster/08-uccsd.md#uccsd-equations`. Every claim asserts; the script exits nonzero on any mismatch.
Negative controls (deliberately wrong variants that must NOT pass) guard
against a vacuous pass.

Inputs are random, not converged amplitudes: each claim is an identity that
must hold for any spin-blocked t1/t2 (t2aa, t2bb antisymmetric in each index
pair) and any per-spin RI factors, (pq|rs) = sum_P B^P_pq B^P_rs. The alpha
and beta spaces have different sizes and different orbital energies, so a
swapped spin label cannot cancel.

Claims
  1  (sympy) The (T) class weight is (ordered spin-orbital triples per
     canonical triple) / 36 = 3! C(3, n_beta) / 36, i.e. 1/6, 1/2, 1/2, 1/6
     for aaa, aab, abb, bbb (u_ccsd_t.rs CLASS_WEIGHTS). A brute-force count
     of spin-conserving ordered (ijk, abc) labels at several sizes equals the
     sum over classes of canonical-loop size times 3! C(3, n_beta).
  2  The spin-blocked (T) (a transcription of u_ccsd_t.rs: canonical occupied
     triples per class, X / Y built block by block from t2aa / t2ab / t2bb
     with the t2 sign table, P(i/jk) P(a/bc), class weights) equals the dense
     spin-orbital (T) of ccsd_t.rs, E = 1/36 sum (W + V) W / D, on the
     spin-orbital expansion of the same amplitudes and integrals, in total
     and per occupied spin class.
  3  The UCCSD energy split of u_ccsd.rs `energy` (aa, ab, bb) equals the
     spin-orbital energy 1/4 sum <ij||ab> tau with tau = t2 + t1 t1 - t1 t1,
     split by the spins of (i, j).
  4  Closed-shell limit: identical alpha and beta spaces with
     t2aa = t2ab - t2ab(a<->b) give aaa == bbb and aab == abb, and the
     CCSD aa == bb.
  5  Negative controls for 2/3: aab weight 1/6, a + sign on the
     (alpha beta beta alpha) block of the t2 table, the alpha and beta
     virtual energies swapped in the aab class, beta t1 dropped from V, and
     the exchange term dropped from the aa energy must each FAIL.

Independent route: claims 2 and 3 compare with dense spin-orbital
expressions built from the antisymmetrized integrals <pq||rs> = (pr|qs) -
(ps|qr) over the alpha-then-beta spin-orbital space; they share only the
random inputs with the spin-blocked transcription.

Mutations (each turns the script red):
  - aab class weight 0.5 -> 1/6 in the transcription (marker
    "[FAIL] claim 2");
  - drop the minus sign of the (alpha beta beta alpha) entry of the t2
    table (marker "[FAIL] claim 2");
  - drop the exchange (ibja) term of the same-spin pair energy (marker
    "[FAIL] claim 3").

Run: python3 scripts/cc_uccsd_spin_blocks_sympy.py
"""

import itertools
import sys
from math import comb, factorial

import numpy as np
import sympy as sp

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def control(name, wrong, right, tol):
    """A deliberately wrong variant must differ from `right` by more than tol."""
    dev = abs(wrong - right) / max(abs(right), 1e-300)
    check(f"control fails: {name}", dev > tol, f"(rel dev {dev:.2e})")


A, B = 0, 1

# ---------------------------------------------------------------- claim 1
nb_sym = sp.symbols("n_beta", integer=True, nonnegative=True)
weights = [sp.Rational(factorial(3) * comb(3, k), 36) for k in range(4)]
check(
    "claim 1 (sympy): weights 3! C(3,nb)/36 = 1/6, 1/2, 1/2, 1/6",
    weights
    == [sp.Rational(1, 6), sp.Rational(1, 2), sp.Rational(1, 2), sp.Rational(1, 6)],
)
na_s, nb_s, va_s, vb_s = sp.symbols("n_a n_b v_a v_b", integer=True, positive=True)
canon = [
    sp.binomial(na_s, 3) * va_s**3,
    sp.binomial(na_s, 2) * nb_s * va_s**2 * vb_s,
    na_s * sp.binomial(nb_s, 2) * va_s * vb_s**2,
    sp.binomial(nb_s, 3) * vb_s**3,
]
ordered_from_classes = sp.expand_func(
    sum(c * factorial(3) * comb(3, k) for k, c in enumerate(canon))
)


def brute_ordered(na, nb, va, vb):
    """Ordered (ijk, abc) spin-orbital labels, i,j,k distinct, spins conserved."""
    occ = [(A, p) for p in range(na)] + [(B, p) for p in range(nb)]
    vir = [(A, p) for p in range(va)] + [(B, p) for p in range(vb)]
    n = 0
    for o in itertools.permutations(occ, 3):
        so = sorted(s for s, _ in o)
        for v in itertools.product(vir, repeat=3):
            if sorted(s for s, _ in v) == so:
                n += 1
    return n


ok = True
for sizes in [(3, 2, 2, 3), (4, 3, 2, 2), (2, 2, 3, 1), (5, 1, 1, 2)]:
    want = ordered_from_classes.subs(dict(zip((na_s, nb_s, va_s, vb_s), sizes)))
    ok &= int(want) == brute_ordered(*sizes)
check("claim 1: brute-force ordered count == sum_class canonical * 3! C(3,nb)", ok)

# ---------------------------------------------------------------- inputs
rng = np.random.default_rng(20261116)
NO = [4, 3]  # alpha, beta active occupied
NV = [3, 4]  # alpha, beta virtual
NAUX = 7


def random_inputs(no, nv, same_spaces=False):
    eo = [np.sort(rng.uniform(-2.0, -0.5, no[s])) for s in (A, B)]
    ev = [np.sort(rng.uniform(0.3, 2.0, nv[s])) for s in (A, B)]
    nmo = [no[s] + nv[s] for s in (A, B)]
    b = []
    for s in (A, B):
        x = rng.normal(size=(NAUX, nmo[s], nmo[s])) * 0.3
        b.append(0.5 * (x + x.transpose(0, 2, 1)))
    t1 = [rng.normal(size=(no[s], nv[s])) * 0.05 for s in (A, B)]
    t2ab = rng.normal(size=(no[A], no[B], nv[A], nv[B])) * 0.05

    def anti(n_o, n_v):
        x = rng.normal(size=(n_o, n_o, n_v, n_v)) * 0.05
        x = x - x.transpose(1, 0, 2, 3)
        return x - x.transpose(0, 1, 3, 2)

    t2aa, t2bb = anti(no[A], nv[A]), anti(no[B], nv[B])
    if same_spaces:
        eo[B], ev[B], b[B], t1[B] = eo[A], ev[A], b[A], t1[A]
        t2ab = 0.5 * (t2ab + t2ab.transpose(1, 0, 3, 2))
        t2aa = t2ab - t2ab.transpose(0, 1, 3, 2)
        t2bb = t2aa
    return eo, ev, b, t1, (t2aa, t2ab, t2bb)


# ------------------------------------------- spin-orbital (independent) route
def spin_orbital(no, nv, eo, ev, b, t1, t2):
    """Alpha-then-beta spin-orbital tensors from the blocks (dense)."""
    t2aa, t2ab, t2bb = t2
    noa, nob = no
    nva, nvb = nv
    n_o, n_v = noa + nob, nva + nvb
    # spin-orbital index -> (spin, spatial index in the full per-spin MO list)
    occ = [(A, p) for p in range(noa)] + [(B, p) for p in range(nob)]
    vir = [(A, no[A] + p) for p in range(nva)] + [(B, no[B] + p) for p in range(nvb)]
    orb = occ + vir
    n = len(orb)
    chem = np.zeros((n, n, n, n))
    for p, (sp_, ip) in enumerate(orb):
        for q, (sq, iq) in enumerate(orb):
            if sp_ != sq:
                continue
            for r, (sr, ir) in enumerate(orb):
                for s_, (ss, is_) in enumerate(orb):
                    if sr != ss:
                        continue
                    chem[p, q, r, s_] = np.dot(b[sp_][:, ip, iq], b[sr][:, ir, is_])
    g = chem.transpose(0, 2, 1, 3) - chem.transpose(0, 2, 3, 1)  # <pq||rs>
    e = np.concatenate([eo[A], eo[B], ev[A], ev[B]])
    T1 = np.zeros((n_o, n_v))
    T1[:noa, :nva] = t1[A]
    T1[noa:, nva:] = t1[B]
    T2 = np.zeros((n_o, n_o, n_v, n_v))
    T2[:noa, :noa, :nva, :nva] = t2aa
    T2[noa:, noa:, nva:, nva:] = t2bb
    T2[:noa, noa:, :nva, nva:] = t2ab
    T2[noa:, :noa, nva:, :nva] = t2ab.transpose(1, 0, 3, 2)
    T2[:noa, noa:, nva:, :nva] = -t2ab.transpose(0, 1, 3, 2)
    T2[noa:, :noa, :nva, nva:] = -t2ab.transpose(1, 0, 2, 3)
    spin_o = np.array([A] * noa + [B] * nob)
    return g, e[:n_o], e[n_o:], T1, T2, spin_o


def dense_triples(g, eo, ev, t1, t2, spin_o):
    """ccsd_t.rs module-doc (T), dense, split by the occupied beta count."""
    n_o, n_v = t1.shape
    o, v = slice(0, n_o), slice(n_o, n_o + n_v)
    g_vvvo = g[v, v, v, o]  # <bc||ei>
    g_ovoo = g[o, v, o, o]  # <ma||jk>
    g_vvoo = g[v, v, o, o]  # <bc||jk>
    x = np.einsum("jkae,bcei->ijkabc", t2, g_vvvo) - np.einsum(
        "imbc,majk->ijkabc", t2, g_ovoo
    )
    y = np.einsum("ia,bcjk->ijkabc", t1, g_vvoo)

    def p(z):
        z = z - z.transpose(1, 0, 2, 3, 4, 5) - z.transpose(2, 1, 0, 3, 4, 5)
        return z - z.transpose(0, 1, 2, 4, 3, 5) - z.transpose(0, 1, 2, 5, 4, 3)

    w, vv = p(x), p(y)
    d = (
        eo[:, None, None, None, None, None]
        + eo[None, :, None, None, None, None]
        + eo[None, None, :, None, None, None]
        - ev[None, None, None, :, None, None]
        - ev[None, None, None, None, :, None]
        - ev[None, None, None, None, None, :]
    )
    f = (w + vv) * w / d / 36.0
    nbeta = spin_o[:, None, None] + spin_o[None, :, None] + spin_o[None, None, :]
    per = np.einsum("ijkabc->ijk", f)
    return [float(per[nbeta == k].sum()) for k in range(4)]


def dense_energy(g, t1, t2, spin_o):
    """1/4 sum <ij||ab> tau, tau = t2 + t1 t1 - t1 t1, split by (i, j) spins."""
    n_o, n_v = t1.shape
    gv = g[:n_o, :n_o, n_o:, n_o:]
    tau = t2 + np.einsum("ia,jb->ijab", t1, t1) - np.einsum("ib,ja->ijab", t1, t1)
    per = 0.25 * np.einsum("ijab,ijab->ij", gv, tau)
    pair = spin_o[:, None] + spin_o[None, :]
    return (
        float(per[pair == 0].sum()),
        float(per[pair == 1].sum()),
        float(per[pair == 2].sum()),
    )


# ------------------------------- spin-blocked transcription (u_ccsd*.rs)
W_AAA = 1.0 / 6.0
W_AAB = 0.5
W_ABB = 0.5
W_BBB = 1.0 / 6.0


def chem2(b1, b2):
    return np.einsum("Ppq,Prs->pqrs", b1, b2)


class Blocks:
    """u_ccsd_t.rs, transcribed: per-spin blocks indexed by spin (0 = alpha)."""

    def __init__(self, no, eo, ev, b, t1, t2, ev_swap_aab=False, drop_t1b=False):
        self.no = no
        self.nv = [ev[A].size, ev[B].size]
        self.eo, self.ev = eo, ev
        self.t1 = [t1[A], np.zeros_like(t1[B]) if drop_t1b else t1[B]]
        self.t2aa, self.t2ab, self.t2bb = t2
        self.ev_swap_aab = ev_swap_aab
        bov = [b[s][:, : no[s], no[s] :] for s in (A, B)]
        boo = [b[s][:, : no[s], : no[s]] for s in (A, B)]
        bvv = [b[s][:, no[s] :, no[s] :] for s in (A, B)]
        self.vvvo, self.oovo, self.vovo = {}, {}, {}
        for s1 in (A, B):
            for s2 in (A, B):
                bvo2 = bov[s2].transpose(0, 2, 1)
                self.vvvo[s1, s2] = chem2(bvv[s1], bvo2)  # (te|up)
                self.oovo[s1, s2] = chem2(boo[s1], bvo2)  # (mq|sr)
                self.vovo[s1, s2] = chem2(bov[s1].transpose(0, 2, 1), bvo2)

    def t2_pair(self, q, r, ss, se):
        """t2[q,r,s,e] over (s,e) for fixed spin orbitals q, r (the sign table)."""
        (sq, iq), (sr, ir) = q, r
        key = (sq, sr, ss, se)
        if key == (A, A, A, A):
            return self.t2aa[iq, ir]
        if key == (B, B, B, B):
            return self.t2bb[iq, ir]
        if key == (A, B, A, B):
            return self.t2ab[iq, ir]
        if key == (A, B, B, A):
            return -self.t2ab[iq, ir].T
        if key == (B, A, A, B):
            return -self.t2ab[ir, iq]
        if key == (B, A, B, A):
            return self.t2ab[ir, iq].T
        return None

    def t2_row(self, p, sm, st, su):
        """t2[p,m,t,u] over (m,t,u) for a fixed spin orbital p."""
        sp_, ip = p
        key = (sp_, sm, st, su)
        if key == (A, A, A, A):
            return self.t2aa[ip]
        if key == (B, B, B, B):
            return self.t2bb[ip]
        if key == (A, B, A, B):
            return self.t2ab[ip]
        if key == (A, B, B, A):
            return -self.t2ab[ip].transpose(0, 2, 1)
        if key == (B, A, A, B):
            return -self.t2ab[:, ip]
        if key == (B, A, B, A):
            return self.t2ab[:, ip].transpose(0, 2, 1)
        return None

    def x_block(self, p, q, r, ss, st, su):
        out = np.zeros((self.nv[ss], self.nv[st], self.nv[su]))
        sp_, ip = p
        sq, iq = q
        sr, ir = r
        for se in (A, B):
            t2qr = self.t2_pair(q, r, ss, se)
            if t2qr is None:
                continue
            if st == se and su == sp_:
                out += np.einsum("se,teu->stu", t2qr, self.vvvo[st, su][:, :, :, ip])
            if st == sp_ and su == se:
                out -= np.einsum("se,uet->stu", t2qr, self.vvvo[su, st][:, :, :, ip])
        for sm in (A, B):
            t2p = self.t2_row(p, sm, st, su)
            if t2p is None:
                continue
            if sm == sq and ss == sr:
                out -= np.einsum("ms,mtu->stu", self.oovo[sm, ss][:, iq, :, ir], t2p)
            if sm == sr and ss == sq:
                out += np.einsum("ms,mtu->stu", self.oovo[sm, ss][:, ir, :, iq], t2p)
        return out

    def y_block(self, p, q, r, ss, st, su):
        out = np.zeros((self.nv[ss], self.nv[st], self.nv[su]))
        sp_, ip = p
        sq, iq = q
        sr, ir = r
        if sp_ != ss:
            return out
        g = np.zeros((self.nv[st], self.nv[su]))
        if st == sq and su == sr:
            g += self.vovo[st, su][:, iq, :, ir]
        if st == sr and su == sq:
            g -= self.vovo[st, su][:, ir, :, iq]
        return np.einsum("s,tu->stu", self.t1[ss][ip], g)

    def wv(self, i, j, k, spins_v, fn):
        sa, sb, sc = spins_v
        out = np.zeros((self.nv[sa], self.nv[sb], self.nv[sc]))
        for occ, so in (((i, j, k), 1.0), ((j, i, k), -1.0), ((k, j, i), -1.0)):
            out += so * fn(*occ, sa, sb, sc)
            out -= so * fn(*occ, sb, sa, sc).transpose(1, 0, 2)
            out -= so * fn(*occ, sc, sb, sa).transpose(2, 1, 0)
        return out

    def triple(self, i, j, k, spins_v):
        w = self.wv(i, j, k, spins_v, self.x_block)
        v = self.wv(i, j, k, spins_v, self.y_block)
        ev = list(self.ev)
        if self.ev_swap_aab and spins_v == (A, A, B):
            ev = [self.ev[B][: self.nv[A]], np.resize(self.ev[A], self.nv[B])]
        sa, sb, sc = spins_v
        d = (
            self.eo[i[0]][i[1]]
            + self.eo[j[0]][j[1]]
            + self.eo[k[0]][k[1]]
            - ev[sa][:, None, None]
            - ev[sb][None, :, None]
            - ev[sc][None, None, :]
        )
        return float(np.sum((w + v) * w / d))

    def classes(self, w_aab=None):
        na, nb = self.no
        w_aab = W_AAB if w_aab is None else w_aab
        aaa = W_AAA * sum(
            self.triple((A, i), (A, j), (A, k), (A, A, A))
            for i, j, k in itertools.combinations(range(na), 3)
        )
        aab = w_aab * sum(
            self.triple((A, i), (A, j), (B, k), (A, A, B))
            for i, j in itertools.combinations(range(na), 2)
            for k in range(nb)
        )
        abb = W_ABB * sum(
            self.triple((A, i), (B, j), (B, k), (A, B, B))
            for i in range(na)
            for j, k in itertools.combinations(range(nb), 2)
        )
        bbb = W_BBB * sum(
            self.triple((B, i), (B, j), (B, k), (B, B, B))
            for i, j, k in itertools.combinations(range(nb), 3)
        )
        return [aaa, aab, abb, bbb]


def blocked_energy(no, b, t1, t2, drop_exchange=False):
    """u_ccsd.rs `energy`, transcribed on per-spin chemist (ia|jb) blocks."""
    t2aa, t2ab, t2bb = t2
    bov = [b[s][:, : no[s], no[s] :] for s in (A, B)]
    ovov = [np.einsum("Pia,Pjb->iajb", bov[s], bov[s]) for s in (A, B)]
    ov_ov = np.einsum("Pia,PJB->iaJB", bov[A], bov[B])
    x = 0.0 if drop_exchange else 1.0

    def same(t1s, t2s, g):
        e = 0.25 * np.einsum("ijab,iajb", t2s, g) - x * 0.25 * np.einsum(
            "ijab,ibja", t2s, g
        )
        e += 0.5 * np.einsum("ia,jb,iajb", t1s, t1s, g) - 0.5 * np.einsum(
            "ia,jb,ibja", t1s, t1s, g
        )
        return float(e)

    e_aa = same(t1[A], t2aa, ovov[A])
    e_bb = same(t1[B], t2bb, ovov[B])
    e_ab = float(
        np.einsum("iJaB,iaJB", t2ab, ov_ov)
        + np.einsum("ia,JB,iaJB", t1[A], t1[B], ov_ov)
    )
    return e_aa, e_ab, e_bb


# ---------------------------------------------------------------- claim 2
eo, ev, b, t1, t2 = random_inputs(NO, NV)
g, eo_so, ev_so, T1, T2, spin_o = spin_orbital(NO, NV, eo, ev, b, t1, t2)
dense = dense_triples(g, eo_so, ev_so, T1, T2, spin_o)
blk = Blocks(NO, eo, ev, b, t1, t2)
cls = blk.classes()
for name, x, y in zip(("aaa", "aab", "abb", "bbb"), cls, dense):
    dev = abs(x - y) / max(abs(y), 1e-300)
    check(
        f"claim 2: (T) class {name} spin-blocked == dense",
        dev < 1e-12,
        f"({x:.6e} vs {y:.6e})",
    )
dev = abs(sum(cls) - sum(dense)) / abs(sum(dense))
check("claim 2: (T) total spin-blocked == dense", dev < 1e-12, f"(rel dev {dev:.1e})")
check("claim 2: every class is nonzero here", min(abs(c) for c in dense) > 1e-10)

# ---------------------------------------------------------------- claim 3
e_blk = blocked_energy(NO, b, t1, t2)
e_den = dense_energy(g, T1, T2, spin_o)
for name, x, y in zip(("aa", "ab", "bb"), e_blk, e_den):
    dev = abs(x - y) / max(abs(y), 1e-300)
    check(
        f"claim 3: CCSD energy {name} spin-blocked == dense",
        dev < 1e-12,
        f"({x:.6e} vs {y:.6e})",
    )

# ---------------------------------------------------------------- claim 4
NO_CS, NV_CS = [3, 3], [4, 4]
eo_c, ev_c, b_c, t1_c, t2_c = random_inputs(NO_CS, NV_CS, same_spaces=True)
cls_c = Blocks(NO_CS, eo_c, ev_c, b_c, t1_c, t2_c).classes()
check(
    "claim 4: closed-shell limit aaa == bbb, aab == abb",
    abs(cls_c[0] - cls_c[3]) < 1e-14 and abs(cls_c[1] - cls_c[2]) < 1e-14,
    f"({cls_c})",
)
ec = blocked_energy(NO_CS, b_c, t1_c, t2_c)
check("claim 4: closed-shell limit CCSD aa == bb", abs(ec[0] - ec[2]) < 1e-14)
g_c, eo_cs, ev_cs, T1c, T2c, so_c = spin_orbital(
    NO_CS, NV_CS, eo_c, ev_c, b_c, t1_c, t2_c
)
dc = dense_triples(g_c, eo_cs, ev_cs, T1c, T2c, so_c)
check(
    "claim 4: closed-shell limit spin-blocked (T) == dense",
    abs(sum(cls_c) - sum(dc)) / abs(sum(dc)) < 1e-12,
)

# ---------------------------------------------------------------- claim 5
right = sum(dense)
control("aab weight 1/6", sum(blk.classes(w_aab=1.0 / 6.0)), right, 1e-6)


class FlipSign(Blocks):
    def t2_pair(self, q, r, ss, se):
        (sq, iq), (sr, ir) = q, r
        if (sq, sr, ss, se) == (A, B, B, A):
            return self.t2ab[iq, ir].T
        return super().t2_pair(q, r, ss, se)


control(
    "+ sign on the (a b b a) t2 block",
    sum(FlipSign(NO, eo, ev, b, t1, t2).classes()),
    right,
    1e-6,
)
control(
    "alpha/beta virtual energies swapped in aab",
    sum(Blocks(NO, eo, ev, b, t1, t2, ev_swap_aab=True).classes()),
    right,
    1e-6,
)
control(
    "beta t1 dropped from V",
    sum(Blocks(NO, eo, ev, b, t1, t2, drop_t1b=True).classes()),
    right,
    1e-6,
)
control(
    "aa energy without exchange",
    blocked_energy(NO, b, t1, t2, drop_exchange=True)[0],
    e_den[0],
    1e-6,
)

if FAILS:
    print(f"\n{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("\nall claims verified")
