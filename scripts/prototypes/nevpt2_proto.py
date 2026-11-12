#!/usr/bin/env python
"""Strongly contracted NEVPT2 prototype in numpy on ferric's CASSCF output.

Correctness reference for a Rust SC-NEVPT2 (T048), after Angeli, Cimiraglia
and Malrieu, JCP 117, 9138 (2002). Orbitals and the CI vector come from
``ferric.run_casscf``; integrals from ``ferric.AoIntegrals``. Equations,
conventions and the measured tables: nevpt2_derivation.md. Nothing about speed
may be quoted from this prototype.

Three constructions of the eight class energies:

* ``wick`` (the port target): the perturber of class l is P_l H Psi_0, with H
  normal ordered against the core. Its norm <V+ P V> and the Dyall excitation
  energy <V+ P [H_act, V]> are reduced by Wick's theorem (filled core, empty
  virtuals, vacuum-normal-ordered active strings) to contractions of
  integrals with the spin-free 1- to 4-RDMs. The reduction is symbolic, done
  once per class; evaluation is einsum.
* ``det``: the same definition evaluated in the full determinant space
  (alpha x beta strings over all orbitals) with explicit E_pq matrices. Only
  for small systems (N2/STO-3G, 14400 determinants).
* ``pyscf``: pyscf.mrpt.nevpt2 on ferric's integrals, fed ferric's orbitals
  and CI vector (``shared``), or its own CASSCF on ferric's integrals
  (``own``).

RDMs come from ferric's CI vector by two constructions that must agree: dense
annihilation chains in the active Fock space, and pyscf.fci.rdm.

Anchors, in order (``--only`` picks one): ``full`` (active = full space gives
zero), ``mp2`` (one doubly occupied active orbital on RHF orbitals equals the
MP2 pair sums class by class), ``size`` (H2...H2 at 100 Angstrom is twice H2),
``det`` (wick == det per class), ``ref`` (wick vs pyscf per class, with the
first-order orbital bar), ``mut`` (each mutation must fail a check).

Usage (one heavy op at a time on the shared box):
  cd ~/qc/ferric && PYTHONPATH=<shim> box-lease exec -- env \\
    OPENBLAS_NUM_THREADS=1 .venv/bin/python \\
    ~/qc/ferric-nevpt2/scripts/prototypes/nevpt2_proto.py
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np

import ferric

sys.path.insert(0, str(Path(__file__).resolve().parent))
import casscf_proto as cp  # noqa: E402

EPS = np.finfo(float).eps
NORM_ZERO = 1e-14  # pyscf.mrpt.nevpt2.NUMERICAL_ZERO; labels below are skipped
JK_THRESH = 1e-14

# (name, virtual particles, core holes); PySCF names in PYSCF_NAME.
CLASSES = (
    ("ijrs", 2, 2),
    ("ijr", 1, 2),
    ("rsi", 2, 1),
    ("ij", 0, 2),
    ("rs", 2, 0),
    ("ir", 1, 1),
    ("r", 1, 0),
    ("i", 0, 1),
)
PYSCF_NAME = {
    "ijrs": "Sijrs",
    "ijr": "Sijr",
    "rsi": "Srsi",
    "ij": "Sij",
    "rs": "Srs",
    "ir": "Sir",
    "r": "Sr",
    "i": "Si",
}


def log(*a) -> None:
    print(*a, flush=True)


# ================================================================ Wick engine
#
# An operator string is a Term: coefficient, integral factors (name, slots)
# and ladder operators (dagger, slot, spin). Slots are symbolic orbital
# indices with a space ('c' core, 'a' active, 'v' virtual); spins are
# symbolic and summed. Every spin id sits on exactly one creator and one
# annihilator, which keeps every reduced term a spin-free RDM.


@dataclass(frozen=True)
class Term:
    coef: float
    tens: tuple
    ops: tuple


class Sym:
    def __init__(self):
        self.space: list[str] = []
        self.nspin = 0

    def slot(self, sp):
        self.space.append(sp)
        return len(self.space) - 1

    def spin(self):
        self.nspin += 1
        return self.nspin - 1

    def fresh(self, t, keep):
        """Copy of t with new dummy slots and spins (slots in keep stay)."""
        smap, pmap = {}, {}

        def ms(s):
            if s in keep:
                return s
            if s not in smap:
                smap[s] = self.slot(self.space[s])
            return smap[s]

        def mp(p):
            if p not in pmap:
                pmap[p] = self.spin()
            return pmap[p]

        tens = tuple((n, tuple(ms(s) for s in sl)) for n, sl in t.tens)
        ops = tuple((d, ms(s), mp(p)) for d, s, p in t.ops)
        return Term(t.coef, tens, ops)

    def dagger(self, t, keep):
        u = self.fresh(t, keep)
        return Term(u.coef, u.tens, tuple((1 - d, s, p) for d, s, p in u.ops[::-1]))


def perturber(sym, nv, nc):
    """Terms of H (normal ordered against the core) that take the reference
    to nv virtual particles and nc core holes. On the reference only terms
    without core creators and virtual annihilators survive; their one-body
    coefficient is the inactive Fock matrix F."""
    terms = []
    for sp, sq in product("av", "ac"):
        if ((sp == "v"), (sq == "c")) == (nv, nc):
            p, q, s = sym.slot(sp), sym.slot(sq), sym.spin()
            terms.append(Term(1.0, (("F", (p, q)),), ((1, p, s), (0, q, s))))
    for sp, sr, sq, ss in product("av", "av", "ac", "ac"):
        if ((sp == "v") + (sr == "v"), (sq == "c") + (ss == "c")) == (nv, nc):
            p, q, r, s = (sym.slot(x) for x in (sp, sq, sr, ss))
            a, b = sym.spin(), sym.spin()
            ops = ((1, p, a), (1, r, b), (0, s, b), (0, q, a))
            terms.append(Term(0.5, (("g", (p, q, r, s)),), ops))
    return terms


def projector(sym, vl, cl, veq, ceq):
    """Number-operator projector on the labelled external occupations.
    veq/ceq: one label slot standing for a doubly occupied virtual / doubly
    emptied core orbital (factor 1/2 from the two spin orderings)."""
    coef, ops = 1.0, []
    if veq:
        r, a, b = vl[0], sym.spin(), sym.spin()
        ops += [(1, r, a), (1, r, b), (0, r, b), (0, r, a)]
        coef *= 0.5
    else:
        for r in vl:
            a = sym.spin()
            ops += [(1, r, a), (0, r, a)]
    if ceq:
        i, a, b = cl[0], sym.spin(), sym.spin()
        ops += [(0, i, a), (0, i, b), (1, i, b), (1, i, a)]
        coef *= 0.5
    else:
        for i in cl:
            a = sym.spin()
            ops += [(0, i, a), (1, i, a)]
    return Term(coef, (), tuple(ops))


def comm_hact(sym, t, w2=1.0):
    """[H_act, t] by the derivation rule, H_act = sum h_pq E_pq
    + 1/2 sum (pq|rs) a+_p a+_r a_s a_q over active orbitals (Dyall).
    w2 scales the two-electron part (the 'flip_koopmans' mutation)."""
    out = []
    for j, (d, x, tau) in enumerate(t.ops):
        if sym.space[x] != "a":
            continue
        pre, post = t.ops[:j], t.ops[j + 1 :]
        p, q, r, s = (sym.slot("a") for _ in range(4))
        sg, rh = sym.spin(), sym.spin()

        def mk(c, tens, rep):
            out.append(Term(t.coef * c, t.tens + tens, pre + rep + post))

        if d == 0:  # [H, a_x]
            mk(-1.0, (("h", (x, q)),), ((0, q, tau),))
            mk(
                -0.5 * w2,
                (("ga", (x, q, r, s)),),
                ((1, r, rh), (0, s, rh), (0, q, tau)),
            )
            mk(0.5 * w2, (("ga", (p, q, x, s)),), ((1, p, sg), (0, s, tau), (0, q, sg)))
        else:  # [H, a+_x]
            mk(1.0, (("h", (p, x)),), ((1, p, tau),))
            mk(0.5 * w2, (("ga", (p, x, r, s)),), ((1, p, tau), (1, r, rh), (0, s, rh)))
            mk(
                -0.5 * w2,
                (("ga", (p, q, r, x)),),
                ((1, p, sg), (1, r, tau), (0, q, sg)),
            )
    return out


def parity(seq):
    n = 0
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            n += seq[i] > seq[j]
    return -1 if n & 1 else 1


def full_matchings(pos, ops, left_dag):
    """Full contractions of one external space: pairs (left, right) with the
    left operator of type left_dag (core: a+ a; virtual: a a+)."""
    if not pos:
        yield ()
        return
    f = pos[0]
    if ops[f][0] != left_dag:
        return
    for k in range(1, len(pos)):
        j = pos[k]
        if ops[j][0] == left_dag:
            continue
        for m in full_matchings(pos[1:k] + pos[k + 1 :], ops, left_dag):
            yield ((f, j),) + m


def partial_contractions(pos, ops):
    """Vacuum Wick contractions (a_x left of a+_y) of the active string."""
    ann = [p for p in pos if ops[p][0] == 0]
    cre = [p for p in pos if ops[p][0] == 1]
    if len(ann) != len(cre):
        return

    def rec(k, used, pairs):
        if k == len(ann):
            yield pairs, used
            return
        a = ann[k]
        yield from rec(k + 1, used, pairs)
        for c in cre:
            if c > a and c not in used:
                yield from rec(k + 1, used | {c}, pairs + ((a, c),))

    for pairs, used in rec(0, frozenset(), ()):
        done = {a for a, _ in pairs}
        yield (
            pairs,
            [c for c in cre if c not in used],
            [a for a in ann if a not in done],
        )


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def expect(sym, t, outputs):
    """<ref| t |ref> as a list of (coef, operands); operands are integral
    factors plus one spin-free RDM ('dmk', slots in PySCF pair order)."""
    ops = t.ops
    cpos = [k for k, o in enumerate(ops) if sym.space[o[1]] == "c"]
    vpos = [k for k, o in enumerate(ops) if sym.space[o[1]] == "v"]
    apos = [k for k, o in enumerate(ops) if sym.space[o[1]] == "a"]
    s0 = parity(cpos + vpos + apos)
    cms = [
        (m, parity([x for pr in m for x in pr])) for m in full_matchings(cpos, ops, 1)
    ]
    if not cms:
        return []
    vms = [
        (m, parity([x for pr in m for x in pr])) for m in full_matchings(vpos, ops, 0)
    ]
    if not vms:
        return []
    out = []
    for pa, rc, ra in partial_contractions(apos, ops):
        k = len(rc)
        if k > 4:
            raise RuntimeError("string needs a >4-body RDM")
        sa = parity([x for pr in pa for x in pr] + rc + ra)
        for (mc, sc), (mv, sv) in product(cms, vms):
            pairs = mc + mv + pa
            us, uo = UF(), UF()
            for a, b in pairs:
                us.union(ops[a][2], ops[b][2])
            partner = {us.find(ops[a][2]): a for a in ra}
            target = [partner[us.find(ops[c][2])] for c in reversed(rc)]
            spi = parity([ra.index(x) for x in target])
            roots = {us.find(o[2]) for o in ops}
            free = len(roots) - k
            # slot identification; two different output labels may not merge
            for a, b in pairs:
                uo.union(ops[a][1], ops[b][1])
            groups = {}
            for s in {o[1] for o in ops} | {s for _, sl in t.tens for s in sl}:
                groups.setdefault(uo.find(s), []).append(s)
            rep, ok = {}, True
            for members in groups.values():
                outs = [s for s in members if s in outputs]
                if len(outs) > 1:
                    ok = False
                    break
                r = outs[0] if outs else min(members)
                for s in members:
                    rep[s] = r
            if not ok:
                continue
            coef = t.coef * s0 * sc * sv * sa * spi * 2.0**free
            opnds = [(n, tuple(rep.get(s, s) for s in sl)) for n, sl in t.tens]
            if k:
                # dm order: creator m and its spin partner, m = 1..k
                sl = []
                for c in rc:
                    a = partner[us.find(ops[c][2])]
                    sl += [rep.get(ops[c][1], ops[c][1]), rep.get(ops[a][1], ops[a][1])]
                opnds.append((f"dm{k}", tuple(sl)))
            out.append((coef, opnds))
    return out


def canonical(opnds, outputs, space):
    """Hashable form with dummy slots renamed by first appearance."""

    def skey(s):
        return (0, outputs.index(s)) if s in outputs else (1, space[s])

    order = sorted(opnds, key=lambda o: (o[0], tuple(skey(s) for s in o[1])))
    ren = {}
    out = []
    for n, sl in order:
        new = []
        for s in sl:
            if s in outputs:
                new.append(("o", outputs.index(s)))
            else:
                if s not in ren:
                    ren[s] = ("d", len(ren), space[s])
                new.append(ren[s])
        out.append((n, tuple(new)))
    return tuple(out)


@dataclass
class Compiled:
    """One label variant of a class: summed norm and Koopmans-energy terms."""

    name: str
    vout: int  # number of virtual output axes
    cout: int
    veq: bool
    ceq: bool
    norm: dict = field(default_factory=dict)
    kop: dict = field(default_factory=dict)


def compile_class(name, nv, nc, mutation=None):
    variants = []
    for veq in [False, True] if nv == 2 else [False]:
        for ceq in [False, True] if nc == 2 else [False]:
            sym = Sym()
            vl = [sym.slot("v") for _ in range(1 if veq else nv)]
            cl = [sym.slot("c") for _ in range(1 if ceq else nc)]
            outputs = vl + cl
            keep = set(outputs)
            vterms = perturber(sym, nv, nc)
            w2 = -1.0 if mutation == ("flip_koopmans", name) else 1.0
            cterms = [c for v in vterms for c in comm_hact(sym, v, w2)]
            proj = projector(sym, vl, cl, veq, ceq)
            comp = Compiled(name, len(vl), len(cl), veq, ceq)
            for target, kets in ((comp.norm, vterms), (comp.kop, cterms)):
                for u in vterms:
                    bra = sym.dagger(u, keep)
                    for ket in kets:
                        s = Term(
                            bra.coef * proj.coef * ket.coef,
                            bra.tens + ket.tens,
                            bra.ops + proj.ops + ket.ops,
                        )
                        for coef, opnds in expect(sym, s, outputs):
                            key = canonical(opnds, outputs, sym.space)
                            target[key] = target.get(key, 0.0) + coef
            for d in (comp.norm, comp.kop):
                for key in [k for k, c in d.items() if abs(c) < 1e-14]:
                    del d[key]
            if mutation == ("drop_rdm3", name):
                k3 = sorted(k for k in comp.norm if any(n == "dm3" for n, _ in k))
                if k3:
                    del comp.norm[k3[0]]
            variants.append(comp)
    return variants


def rdm_rank(key):
    return max([int(n[2]) for n, _ in key if n.startswith("dm")], default=0)


def cost_report():
    """Per class: summed terms by RDM rank (norm / Koopmans part) and the
    largest index count per space in one term (naive einsum scaling)."""
    log("\n== compiled terms per class (all label variants) ==")
    log(
        "class  norm terms by RDM rank 0..4 | K terms by rank 0..4 | worst term (c,a,v)"
    )
    for name, nv, nc in CLASSES:
        nr, kr = [0] * 5, [0] * 5
        worst = (0, 0, 0)
        for comp in compile_class(name, nv, nc):
            for d, cnt in ((comp.norm, nr), (comp.kop, kr)):
                for key in d:
                    cnt[rdm_rank(key)] += 1
                    sl = {s for _, x in key for s in x}
                    sp = [
                        s[2] if s[0] == "d" else ("v" if s[1] < comp.vout else "c")
                        for s in sl
                    ]
                    w = (sp.count("c"), sp.count("a"), sp.count("v"))
                    worst = max(worst, w, key=lambda t: (t[1], sum(t)))
        log(f"{name:5s}  {nr} | {kr} | {worst}")


LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


class Evaluator:
    """Numeric evaluation of compiled terms on one set of integrals/RDMs."""

    def __init__(self, nc, na, nv, F, g, h, ga, dms):
        self.rng = {
            "c": slice(0, nc),
            "a": slice(nc, nc + na),
            "v": slice(nc + na, nc + na + nv),
        }
        self.F, self.g, self.h, self.ga, self.dms = F, g, h, ga, dms
        self.cache = {}

    def operand(self, name, spaces):
        key = (name, spaces)
        if key not in self.cache:
            if name == "F":
                arr = self.F[tuple(self.rng[s] for s in spaces)]
            elif name == "g":
                arr = self.g[tuple(self.rng[s] for s in spaces)]
            elif name == "h":
                arr = self.h
            elif name == "ga":
                arr = self.ga
            else:
                arr = self.dms[int(name[2])]
            self.cache[key] = np.ascontiguousarray(arr)
        return self.cache[key]

    def run(self, terms, out_spaces):
        shape = tuple(self.F[self.rng[s], 0].shape[0] for s in out_spaces)
        total = np.zeros(shape)
        letters = {}
        for key, coef in terms.items():
            letters.clear()
            subs, arrs = [], []
            for n, sl in key:
                sp = []
                for s in sl:
                    if s not in letters:
                        letters[s] = LETTERS[len(letters)]
                    sp.append(s[2] if s[0] == "d" else out_spaces[s[1]])
                subs.append("".join(letters[s] for s in sl))
                arrs.append(self.operand(n, tuple(sp)))
            outl = "".join(letters[("o", k)] for k in range(len(out_spaces)))
            expr = ",".join(subs) + "->" + outl
            total += coef * np.einsum(expr, *arrs, optimize="greedy")
        return total


def class_energy(ev, variants, eps, nc, na):
    """Sum over labels of -N/(Delta + K/N); unordered pairs for equal spaces.
    Returns (energy, norm, number of labels kept)."""
    ec, evv = eps[:nc], eps[nc + na :]
    e, nrm, nlab = 0.0, 0.0, 0
    for comp in variants:
        osp = ["v"] * comp.vout + ["c"] * comp.cout
        n = ev.run(comp.norm, osp)
        k = ev.run(comp.kop, osp) if comp.kop else np.zeros_like(n)
        delta = np.zeros(n.shape)
        for ax, s in enumerate(osp):
            vals = evv if s == "v" else -ec
            mult = 2.0 if (s == "v" and comp.veq) or (s == "c" and comp.ceq) else 1.0
            shape = [1] * len(osp)
            shape[ax] = -1
            delta = delta + mult * vals.reshape(shape)
        mask = np.abs(n) > NORM_ZERO
        if comp.vout == 2:
            m = np.triu(np.ones((len(evv),) * 2, dtype=bool), 1)
            mask &= m.reshape(m.shape + (1,) * comp.cout)
        if comp.cout == 2:
            m = np.triu(np.ones((nc, nc), dtype=bool), 1)
            mask &= m.reshape((1,) * comp.vout + m.shape)
        e -= np.sum(n[mask] / (delta[mask] + k[mask] / n[mask]))
        nrm += np.sum(n[mask])
        nlab += int(mask.sum())
    return e, nrm, nlab


# ================================================================ RDMs


def dense_rdms(ci, n, nelec, kmax=4):
    """Spin-free 1..kmax RDMs, dm_k[p1,q1,..,pk,qk] = sum_spin
    <a+p1 .. a+pk a_qk .. a_q1>, from annihilation chains in the active
    Fock space (alpha modes 0..n-1, beta modes n..2n-1, Jordan-Wigner)."""
    from pyscf.fci import cistring

    sa = cistring.make_strings(range(n), nelec[0])
    sb = cistring.make_strings(range(n), nelec[1])
    nel = nelec[0] + nelec[1]
    allst = np.arange(1 << (2 * n))
    pop = np.array([bin(x).count("1") for x in allst])
    sectors = [allst[pop == nel - k] for k in range(min(kmax, nel) + 1)]
    psi = np.zeros(len(sectors[0]))
    idx0 = {int(x): i for i, x in enumerate(sectors[0])}
    for ia, a in enumerate(sa):
        for ib, b in enumerate(sb):
            psi[idx0[int(a) | (int(b) << n)]] = ci[ia, ib]
    maps = []  # maps[k][m] = (src positions in sector k, dst positions in k+1, sign)
    for k in range(min(kmax, nel)):
        src_sec, dst_sec = sectors[k], sectors[k + 1]
        mk = []
        for m in range(2 * n):
            sel = np.nonzero(src_sec & (1 << m))[0]
            st = src_sec[sel]
            dst = np.searchsorted(dst_sec, st ^ (1 << m))
            below = st & ((1 << m) - 1)
            sign = np.where(np.array([bin(x).count("1") for x in below]) % 2, -1.0, 1.0)
            mk.append((sel, dst, sign))
        maps.append(mk)
    phi = psi[None, :]  # rows: mode tuples (m1..mk), m = q + sigma*n
    out = {k: np.zeros((n,) * (2 * k)) for k in range(nel + 1, kmax + 1)}
    for k in range(1, min(kmax, nel) + 1):
        nxt = np.zeros((phi.shape[0], 2 * n, len(sectors[k])))
        for m, (sel, dst, sign) in enumerate(maps[k - 1]):
            nxt[:, m, dst] = phi[:, sel] * sign
        phi = nxt.reshape(-1, len(sectors[k]))
        # rows are mode tuples (m1..mk), m = q + sigma*n -> (sigma1..k, q1..k)
        ph = phi.reshape((2, n) * k + (-1,))
        perm = [2 * j for j in range(k)] + [2 * j + 1 for j in range(k)] + [2 * k]
        ph = ph.transpose(perm).reshape(2**k, n**k, -1)
        d = np.einsum("sqx,sQx->qQ", ph, ph).reshape((n,) * (2 * k))
        out[k] = d.transpose([x for j in range(k) for x in (j, k + j)])
    return out


def pyscf_rdms(ci, n, nelec):
    from pyscf.fci import rdm

    d1, d2, d3, d4 = rdm.make_dm1234("FCI4pdm_kern_sf", ci, ci, n, nelec)
    d1, d2, d3, d4 = rdm.reorder_dm1234(d1, d2, d3, d4)
    return {1: d1, 2: d2, 3: d3, 4: d4}


def rotate_rdms(dms, u):
    """dm in the basis C_a u (real orthogonal u)."""
    out = {}
    for k, d in dms.items():
        for ax in range(2 * k):
            d = np.moveaxis(np.tensordot(d, u, axes=([ax], [0])), -1, ax)
        out[k] = d
    return out


# ================================================================ pipeline


@dataclass
class Ref:
    """Canonical orbitals, integrals and RDMs for one CAS reference."""

    nc: int
    na: int
    nv: int
    c: np.ndarray  # canonical MOs
    eps: np.ndarray
    fi: np.ndarray  # inactive Fock, MO
    g: np.ndarray  # (pq|rs), MO
    dms: dict
    e0: float  # Dyall active energy
    ecore: float
    hmo: np.ndarray
    enuc: float
    ci: np.ndarray
    u_act: np.ndarray


def build_ref(
    ints,
    c,
    nc,
    na,
    ci,
    nelecas,
    active="pseudo",
    rdm_check=True,
    dms=None,
    canon=True,
    rot=None,
):
    nmo = c.shape[1]
    nv = nmo - nc - na
    if dms is None:
        dms = dense_rdms(ci, na, nelecas) if na else {}
        if rdm_check and na:
            pd = pyscf_rdms(ci, na, nelecas)
            ndet = ci.size
            for k in range(1, 5):
                diff = np.abs(dms[k] - pd[k]).max()
                bar = 100 * ndet * EPS * max(1.0, np.abs(pd[k]).max())
                if diff > bar:
                    raise RuntimeError(
                        f"dm{k}: dense vs pyscf.fci.rdm {diff:.1e} > {bar:.1e}"
                    )
    h = ints.hcore()
    cc, ca = c[:, :nc], c[:, nc : nc + na]
    dc = 2.0 * cc @ cc.T
    if nc:
        j, k = ints.jk(dc, thresh=JK_THRESH)
        fi_ao = h + j - 0.5 * k
    else:
        fi_ao = h.copy()
    ecore = ints.nuclear_repulsion() + 0.5 * np.sum(dc * (h + fi_ao))
    if na:
        j, k = ints.jk(ca @ dms[1] @ ca.T, thresh=JK_THRESH)
        fa_ao = j - 0.5 * k
    else:
        fa_ao = 0.0 * h
    f_ao = fi_ao + fa_ao
    blocks = [range(0, nc), range(nc, nc + na), range(nc + na, nmo)]
    u = np.zeros((nmo, nmo))
    eps = np.zeros(nmo)
    for bi, b in enumerate(blocks):
        b = list(b)
        if not b:
            continue
        fb = c[:, b].T @ f_ao @ c[:, b]
        if not canon or (bi == 1 and active == "asis"):
            w, v = np.diag(fb).copy(), np.eye(len(b))
        elif bi == 1 and active == "natural":
            occ, v = np.linalg.eigh(-dms[1])
            w = np.diag(v.T @ fb @ v)
        else:
            w, v = np.linalg.eigh(fb)
        u[np.ix_(b, b)] = v
        eps[b] = w
    if rot is not None:  # extra rotation within degenerate external groups
        u = u @ rot
        eps = np.diag(u.T @ c.T @ f_ao @ c @ u).copy()
    cn = c @ u
    u_act = u[nc : nc + na, nc : nc + na]
    if na:
        dms = rotate_rdms(dms, u_act)
    fi = cn.T @ fi_ao @ cn
    g = ints.ao2mo(cn, cn, cn, cn)
    a = slice(nc, nc + na)
    e0 = 0.0
    if na:
        e0 = np.sum(fi[a, a] * dms[1]) + 0.5 * np.sum(g[a, a, a, a] * dms[2])
    return Ref(
        nc,
        na,
        nv,
        cn,
        eps,
        fi,
        g,
        dms,
        e0,
        ecore,
        cn.T @ h @ cn,
        ints.nuclear_repulsion(),
        ci,
        u_act,
    )


_COMPILED = {}


def compiled(mutation=None):
    key = mutation
    if key not in _COMPILED:
        _COMPILED[key] = {
            n: compile_class(n, nv, nc, mutation) for n, nv, nc in CLASSES
        }
    return _COMPILED[key]


def nevpt2_wick(ref, mutation=None):
    comp = compiled(mutation)
    a = slice(ref.nc, ref.nc + ref.na)
    ev = Evaluator(
        ref.nc, ref.na, ref.nv, ref.fi, ref.g, ref.fi[a, a], ref.g[a, a, a, a], ref.dms
    )
    out = {}
    for name, nv, nc in CLASSES:
        if mutation == ("omit", name):
            out[name] = 0.0
            continue
        if (nv and not ref.nv) or (nc and not ref.nc):
            out[name] = 0.0
            continue
        out[name] = class_energy(ev, comp[name], ref.eps, ref.nc, ref.na)[0]
    return out


# ------------------------------------------------ determinant-space definition


def nevpt2_det(ref, nelecas):
    """SC-NEVPT2 by definition in the full determinant space (small systems):
    Psi_l = P_l H Psi_0, N_l = |Psi_l|^2, E_l - E_0 = Delta_l
    + <Psi_l|H_act - E_0|Psi_l>/N_l with explicit E_pq matrices."""
    from pyscf.fci import cistring

    nc, na, nv = ref.nc, ref.na, ref.nv
    nmo = nc + na + nv
    nal, nbe = nc + nelecas[0], nc + nelecas[1]
    sa = cistring.make_strings(range(nmo), nal)
    sb = cistring.make_strings(range(nmo), nbe)

    def emats(strs):
        idx = {int(x): i for i, x in enumerate(strs)}
        e = np.zeros((nmo, nmo, len(strs), len(strs)))
        for col, s in enumerate(strs):
            s = int(s)
            for q in range(nmo):
                if not s >> q & 1:
                    continue
                s1 = s ^ (1 << q)
                sg1 = (-1) ** bin(s1 & ((1 << q) - 1)).count("1")
                for p in range(nmo):
                    if s1 >> p & 1:
                        continue
                    s2 = s1 | (1 << p)
                    sg2 = (-1) ** bin(s1 & ((1 << p) - 1)).count("1")
                    e[p, q, idx[s2], col] = sg1 * sg2
        return e

    ea, eb = emats(sa), emats(sb)

    def apply_e(p, q, v):
        return ea[p, q] @ v + v @ eb[p, q].T

    # reference: core doubly occupied, active CI, virtuals empty
    asa = cistring.make_strings(range(na), nelecas[0])
    asb = cistring.make_strings(range(na), nelecas[1])
    ia = {int(x): i for i, x in enumerate(sa)}
    ib = {int(x): i for i, x in enumerate(sb)}
    core = (1 << nc) - 1
    psi = np.zeros((len(sa), len(sb)))
    for i, x in enumerate(asa):
        for j, y in enumerate(asb):
            psi[ia[core | (int(x) << nc)], ib[core | (int(y) << nc)]] = ref.ci[i, j]
    # the CI vector is in the original active basis; rotate the canonical
    # integrals back to it: use the pre-rotation active block.
    u = np.eye(nmo)
    u[nc : nc + na, nc : nc + na] = ref.u_act
    hmo = u @ ref.hmo @ u.T
    fi = u @ ref.fi @ u.T
    g = np.einsum("pP,qQ,rR,sS,PQRS->pqrs", u, u, u, u, ref.g, optimize=True)
    phi = np.array([[apply_e(r, s, psi) for s in range(nmo)] for r in range(nmo)])
    x = np.einsum("pqrs,rsij->pqij", g, phi, optimize=True)
    kk = np.einsum("pqqs->ps", g)
    hpsi = np.zeros_like(psi)
    for p in range(nmo):
        for q in range(nmo):
            hpsi += apply_e(p, q, hmo[p, q] * psi + 0.5 * x[p, q])
            hpsi -= 0.5 * kk[p, q] * phi[p, q]
    e_cas = np.sum(psi * hpsi) + ref.enuc
    # Dyall active Hamiltonian on any vector (active indices only)
    act = range(nc, nc + na)

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

    e0 = np.sum(psi * hact(psi))
    # external labels of every determinant
    cocc_a = np.array([[int(s) >> i & 1 for i in range(nc)] for s in sa]).reshape(
        len(sa), nc
    )
    cocc_b = np.array([[int(s) >> i & 1 for i in range(nc)] for s in sb]).reshape(
        len(sb), nc
    )
    vocc_a = np.array(
        [[int(s) >> (nc + na + r) & 1 for r in range(nv)] for s in sa]
    ).reshape(len(sa), nv)
    vocc_b = np.array(
        [[int(s) >> (nc + na + r) & 1 for r in range(nv)] for s in sb]
    ).reshape(len(sb), nv)
    y = hpsi.copy()
    labels = {}
    for i in range(len(sa)):
        for j in range(len(sb)):
            holes = 2 - cocc_a[i] - cocc_b[j]
            parts = vocc_a[i] + vocc_b[j]
            lab = (
                tuple(np.repeat(np.arange(nv), parts)),
                tuple(np.repeat(np.arange(nc), holes)),
            )
            if not lab[0] and not lab[1]:
                y[i, j] = 0.0
                continue
            labels.setdefault(lab, []).append((i, j))
    hy = hact(y)
    ec, evv = ref.eps[:nc], ref.eps[nc + na :]
    out = {n: 0.0 for n, _, _ in CLASSES}
    cls_of = {(nv_, nc_): n for n, nv_, nc_ in CLASSES}
    for (vl, cl), members in labels.items():
        ii, jj = np.array(members).T
        nn = np.sum(y[ii, jj] ** 2)
        if nn <= NORM_ZERO:
            continue
        kk_ = np.sum(y[ii, jj] * hy[ii, jj]) - e0 * nn
        delta = sum(evv[r] for r in vl) - sum(ec[i] for i in cl)
        out[cls_of[(len(vl), len(cl))]] -= nn / (delta + kk_ / nn)
    return out, e_cas, e0


# ------------------------------------------------ PySCF reference


def pyscf_mc(ints, pmol, c, ci, nc, na, nelecas):
    """pyscf CASCI object on ferric integrals carrying (c, ci)."""
    from pyscf import mcscf

    mf = cp.pyscf_rhf_on_ferric_integrals(pmol, ints)
    mc = mcscf.CASCI(mf, na, nelecas)
    mc.verbose = 0
    mc.mo_coeff = c
    mc.ci = ci
    assert mc.ncore == nc
    return mc, mf


def pyscf_nevpt_classes(mc):
    """pyscf.mrpt.nevpt2 per class (the steps of NEVPT.kernel, non-DMRG)."""
    from pyscf import fci, mrpt
    from pyscf.mrpt import nevpt2 as nv2

    nev = mrpt.NEVPT(mc)
    nev.verbose = 0
    nev.mo_coeff, ci, nev.mo_energy = nev.canonicalize(
        nev.mo_coeff, ci=nev.load_ci(), cas_natorb=True, verbose=0
    )
    nev.ci = ci
    ncas, nelecas, ncore = nev.ncas, nev.nelecas, nev.ncore
    dm1, dm2, dm3 = fci.rdm.make_dm123("FCI3pdm_kern_sf", ci, ci, ncas, nelecas)
    dms = {"1": dm1, "2": dm2, "3": dm3, "4": None}
    eris = nv2._ERIS(nev, nev.mo_coeff)
    li = (
        fci.cistring.gen_linkstr_index(range(ncas), nelecas[0]),
        fci.cistring.gen_linkstr_index(range(ncas), nelecas[1]),
    )
    aaaa = eris["ppaa"][ncore : ncore + ncas, ncore : ncore + ncas].copy()
    dms["f3ca"] = nv2._contract4pdm("NEVPTkern_cedf_aedf", aaaa, ci, ncas, nelecas, li)
    dms["f3ac"] = nv2._contract4pdm("NEVPTkern_aedf_ecdf", aaaa, ci, ncas, nelecas, li)
    out = {}
    out["r"] = nv2.Sr(nev, ci, dms, eris)[1]
    out["i"] = nv2.Si(nev, ci, dms, eris)[1]
    out["ijrs"] = nv2.Sijrs(nev, eris)[1]
    out["ijr"] = nv2.Sijr(nev, dms, eris)[1]
    out["rsi"] = nv2.Srsi(nev, dms, eris)[1]
    out["rs"] = nv2.Srs(nev, dms, eris)[1]
    out["ij"] = nv2.Sij(nev, dms, eris)[1]
    out["ir"] = nv2.Sir(nev, dms, eris)[1]
    return out, np.asarray(nev.mo_coeff), np.asarray(ci)


# ================================================================ set-up


def mol_from_atoms(atoms, charge=0):
    lines = [str(len(atoms)), ""] + [
        f"{s} {x:.10f} {y:.10f} {z:.10f}" for s, (x, y, z) in atoms
    ]
    return ferric.Molecule.from_xyz_string("\n".join(lines), charge)


def ferric_casscf(mol, basis, na, nelecas, nc=None, gtol=1e-8):
    bs = ferric.BasisSet.bundled(basis)
    ints = ferric.AoIntegrals(mol, bs)
    kw = dict(casscf_grad_tol=gtol)
    if nc is not None:
        kw["n_core"] = nc
    r = ferric.run_casscf(mol, bs, na, nelecas, **kw)
    if not r.converged:
        raise RuntimeError("ferric CASSCF did not converge")
    return ints, r


def system(label):
    """name -> (molecule, basis, ncas, nelecas, ncore or None)."""
    water = ferric.Molecule.from_xyz(str(cp.MOLS / "water.xyz"))
    if label == "h2o":
        return water, "6-31g", 4, (2, 2), None
    if label == "n2":
        return (
            mol_from_atoms([("N", (0, 0, 0)), ("N", (0, 0, 1.0977))]),
            "6-31g",
            6,
            (3, 3),
            None,
        )
    if label == "n2-stretched":
        return (
            mol_from_atoms([("N", (0, 0, 0)), ("N", (0, 0, 2.2))]),
            "6-31g",
            6,
            (3, 3),
            None,
        )
    if label == "n2-sto3g":
        return (
            mol_from_atoms([("N", (0, 0, 0)), ("N", (0, 0, 1.6))]),
            "sto-3g",
            6,
            (5, 5),
            2,
        )
    raise KeyError(label)


# The prototype and ferric build the RDMs and E_CAS from the same CI vector and
# orbitals, so the differences are rounding; 1e-10 is the shared-input bar used
# by compare_ref.
CONSISTENCY_BAR = 1e-10


def ref_from_casscf(ints, r, nelecas, **kw):
    c = np.asarray(r.mo_coeff)
    ci = np.asarray(r.ci_vector)
    na = int(np.asarray(r.dm1).shape[0])
    ref = build_ref(ints, c, r.n_core, na, ci, nelecas, **kw)
    # dense RDMs vs ferric's own 1-/2-RDM (in the original active basis)
    back = rotate_rdms({1: ref.dms[1], 2: ref.dms[2]}, ref.u_act.T)
    d1 = np.abs(back[1] - np.asarray(r.dm1)).max()
    d2 = np.abs(back[2] - np.asarray(r.dm2)).max()
    ecas = ref.ecore + ref.e0
    chk = {
        "dm1_vs_ferric": d1,
        "dm2_vs_ferric": d2,
        "ecas_err": abs(ecas - r.e_total),
    }
    bad = {k: v for k, v in chk.items() if not v < CONSISTENCY_BAR}
    if bad:
        raise RuntimeError(
            f"prototype reference disagrees with ferric (bar {CONSISTENCY_BAR:.0e}): {bad}"
        )
    return ref, chk


def fmt_classes(d):
    return " ".join(f"{n}={d[n]: .10f}" for n, _, _ in CLASSES)


# ================================================================ anchors


def anchor_full():
    log("\n== anchor 1: active = full space gives E_PT2 = 0 ==")
    mol = mol_from_atoms([("H", (0, 0, 0)), ("H", (0, 0, 0.74))])
    ints, r = ferric_casscf(mol, "6-31g", 4, (1, 1))
    ref, chk = ref_from_casscf(ints, r, (1, 1))
    e = nevpt2_wick(ref)
    tot = sum(e.values())
    log(f"H2/6-31G CAS(2,4): nc={ref.nc} nv={ref.nv} E_PT2={tot:.3e} checks {chk}")
    return tot == 0.0 and ref.nc == 0 and ref.nv == 0


def anchor_mp2(mutation=None):
    """One doubly occupied active orbital (HOMO) on ferric RHF orbitals.

    Expected per class: ijrs = MP2 core-core pairs; rsi = 2 sum_i e_(i,t);
    rs = sum_ab (ta|tb)^2 / (2 F^I_tt + (tt|tt) - e_a - e_b) (Dyall
    denominator: the active vacuum has energy 0, E_0 = 2 F^I_tt + (tt|tt));
    r, ir are second order in the RHF Fock off-diagonal (zero to rounding);
    ijr, ij, i have zero norm (no active holes)."""
    mol = ferric.Molecule.from_xyz(str(cp.MOLS / "water.xyz"))
    bs = ferric.BasisSet.bundled("6-31g")
    ints = ferric.AoIntegrals(mol, bs)
    rh = ferric.run_rhf(mol, bs, energy_conv=1e-12, density_conv=1e-10)
    c = np.asarray(rh.mo_coefficients())
    nocc = ints.nelec // 2
    nc, na = nocc - 1, 1
    ci = np.ones((1, 1))
    ref = build_ref(
        ints,
        c,
        nc,
        na,
        ci,
        (1, 1),
        rdm_check=False,
        dms={
            1: np.full((1, 1), 2.0),
            2: np.full((1,) * 4, 2.0),
            3: np.zeros((1,) * 6),
            4: np.zeros((1,) * 8),
        },
    )
    eps, g = ref.eps, ref.g
    occ, vir = slice(0, nocc), slice(nocc, None)
    ov = g[occ, vir, occ, vir]
    eo, evv = eps[occ], eps[vir]
    den = (
        eo[:, None, None, None]
        + eo[None, None, :, None]
        - evv[None, :, None, None]
        - evv[None, None, None, :]
    )
    pair = np.einsum("iajb,iajb->ij", ov * (2 * ov - ov.transpose(0, 3, 2, 1)), 1 / den)
    t = nc
    gt = g[t, vir, t, vir]
    e_rs = -np.sum(
        gt**2 / (evv[:, None] + evv[None, :] - 2 * ref.fi[t, t] - g[t, t, t, t])
    )
    expect_ = {
        "ijrs": pair[:nc, :nc].sum(),
        "rsi": 2 * pair[:nc, t].sum(),
        "rs": e_rs,
        "ijr": 0.0,
        "ij": 0.0,
        "i": 0.0,
        "r": 0.0,
        "ir": 0.0,
    }
    e = nevpt2_wick(ref, mutation)
    bar = 1e-12
    worst = max(abs(e[n] - expect_[n]) for n, _, _ in CLASSES)
    log(f"H2O/6-31G CAS(2,1) on RHF orbitals; E_MP2 = {pair.sum():.12f}")
    for n, _, _ in CLASSES:
        log(
            f"  {n:5s} wick {e[n]: .14f}  expected {expect_[n]: .14f}  diff {e[n] - expect_[n]: .1e}"
        )
    log(f"  max diff {worst:.1e} (bar {bar:.0e})")
    return worst < bar


SIZE_BAR = [1e-8]


def anchor_size(mutation=None, fd=True):
    log("\n== anchor 3: H2...H2 at 100 Angstrom equals twice H2 ==")
    mono = mol_from_atoms([("H", (0, 0, 0)), ("H", (0, 0, 0.74))])
    dim = mol_from_atoms(
        [
            ("H", (0, 0, 0)),
            ("H", (0, 0, 0.74)),
            ("H", (100.0, 0, 0)),
            ("H", (100.0, 0, 0.74)),
        ]
    )
    res = {}
    for lab, mol, na, ne in (("mono", mono, 2, (1, 1)), ("dimer", dim, 4, (2, 2))):
        ints, r = ferric_casscf(mol, "cc-pvdz", na, ne)
        ref, chk = ref_from_casscf(ints, r, ne)
        e = nevpt2_wick(ref, mutation)
        ob = orbital_bar(ints, ref, ne) if fd else None
        res[lab] = (e, ob)
        log(f"  {lab}: E_CASSCF {r.e_total:.12f} |g| {r.grad_norm:.1e} checks {chk}")
        if ob:
            log(
                f"    |dE_PT2/dk| {ob['gtot']:.2e} |g| {ob['g']:.1e} lambda_min {ob['lam']:.3e}"
            )
    em, ed = res["mono"][0], res["dimer"][0]
    diff = {n: ed[n] - 2 * em[n] for n, _, _ in CLASSES}
    tot = sum(diff.values())
    # E_PT2 is first order in each code path's orbital error ||g||/lambda;
    # the monomer error enters twice. Rounding floor 1e-12.
    if fd:
        bm, bd = res["mono"][1], res["dimer"][1]
        bar = (
            2 * bm["gtot"] * bm["g"] / bm["lam"]
            + bd["gtot"] * bd["g"] / bd["lam"]
            + 1e-12
        )
        SIZE_BAR[0] = bar
    else:
        bar = SIZE_BAR[0]  # from the unmutated run (run_mutations runs it first)
    log("  dimer - 2 mono per class: " + fmt_classes(diff))
    log(f"  total {tot:.2e} (bar {bar:.1e}); E_PT2(mono) {sum(em.values()):.12f}")
    return abs(tot) < bar and all(abs(v) < bar for v in diff.values())


def anchor_det(mutation=None):
    log(
        "\n== check: wick vs determinant-space definition (N2/STO-3G 1.6 A, core 2, CAS(10,6)) =="
    )
    mol, basis, na, ne, nc = system("n2-sto3g")
    ints, r = ferric_casscf(mol, basis, na, ne, nc)
    ref, chk = ref_from_casscf(ints, r, ne)
    occ = np.linalg.eigvalsh(ref.dms[1])[::-1]
    log(
        f"  E_CASSCF {r.e_total:.12f}; natural occupations {np.round(occ, 4)}; checks {chk}"
    )
    t0 = time.time()
    ew = nevpt2_wick(ref, mutation)
    t1 = time.time()
    ed, e_cas, e0 = nevpt2_det(ref, ne)
    t2 = time.time()
    log(
        f"  det: E_CAS {e_cas:.12f} (ferric {r.e_total:.12f}); E0 {e0:.12f} vs {ref.e0:.12f}"
    )
    bar = 1e-11
    worst = 0.0
    for n, _, _ in CLASSES:
        d = ew[n] - ed[n]
        worst = max(worst, abs(d))
        log(f"  {n:5s} wick {ew[n]: .14f}  det {ed[n]: .14f}  diff {d: .1e}")
    log(
        f"  max diff {worst:.1e} (bar {bar:.0e}); wick {t1 - t0:.1f}s det {t2 - t1:.1f}s"
    )
    return worst < bar


# ------------------------------------------------ PySCF comparison and the bar


def fd_pt2_gradient(ints, ref, nelecas, h=1e-4):
    """dE_PT2/dkappa per class, central differences over every nonredundant
    rotation pair; CI re-solved, orbitals re-canonicalized at each point."""
    nc, na = ref.nc, ref.na
    cas = cp.Cas(ints, nc, na, nelecas, cp.PyscfFci())
    cas.setup_pairs(ref.c.shape[1])
    grads = {n: np.zeros(len(cas.cls)) for n, _, _ in CLASSES}

    def pt2(c):
        _, res = cas.energy(c)
        ci = res.ci
        rf = build_ref(
            ints,
            c,
            nc,
            na,
            ci,
            nelecas,
            rdm_check=False,
            dms=pyscf_rdms(ci, na, nelecas),
        )
        return nevpt2_wick(rf)

    for k in range(len(cas.cls)):
        kv = np.zeros(len(cas.cls))
        kv[k] = h
        ep, em = pt2(cas.rotate(ref.c, kv)), pt2(cas.rotate(ref.c, -kv))
        for n in grads:
            grads[n][k] = (ep[n] - em[n]) / (2 * h)
    return grads, cas


def orbital_bar(ints, ref, nelecas, c_other=None):
    """|dE_PT2/dkappa| (per class and total, FD), ||g|| and lambda_min of
    the CI-relaxed orbital Hessian at ref.c (casscf_proto conventions)."""
    grads, cas = fd_pt2_gradient(ints, ref, nelecas)
    out = {
        "grads": {n: np.linalg.norm(v) for n, v in grads.items()},
        "gtot": np.linalg.norm(sum(grads.values())),
        "g": np.linalg.norm(cas.gradient(ref.c)["g"]),
        "lam": cp.fd_hessian_min_eig(cas, ref.c)[0],
    }
    if c_other is not None:
        out["g_other"] = np.linalg.norm(cas.gradient(c_other)["g"])
    return out


def degenerate_spread(ints, r, ne, ref, ntrial=4, tol=1e-6, seed=11):
    """Max change of each class energy under random rotations inside groups
    of (near-)degenerate core or virtual orbitals (|de| < tol). SC-NEVPT2
    pair classes are not invariant to these rotations."""
    eps = ref.eps
    groups = []
    for lo, hi in ((0, ref.nc), (ref.nc + ref.na, len(eps))):
        idx = list(range(lo, hi))
        while idx:
            g = [k for k in idx if abs(eps[k] - eps[idx[0]]) < tol]
            if len(g) > 1:
                groups.append(g)
            idx = [k for k in idx if k not in g]
    base = nevpt2_wick(ref)
    spread = {n: 0.0 for n, _, _ in CLASSES}
    if not groups:
        return spread
    rng = np.random.default_rng(seed)
    c0 = ref.c @ np.eye(len(eps))
    for _ in range(ntrial):
        rot = np.eye(len(eps))
        for g in groups:
            q, _ = np.linalg.qr(rng.standard_normal((len(g), len(g))))
            rot[np.ix_(g, g)] = q
        rf = build_ref(
            ints,
            c0,
            ref.nc,
            ref.na,
            None,
            ne,
            canon=False,
            rdm_check=False,
            dms=rotate_rdms(ref.dms, np.eye(ref.na)),
            rot=rot,
        )
        e = nevpt2_wick(rf)
        for n in spread:
            spread[n] = max(spread[n], abs(e[n] - base[n]))
    return spread


def compare_ref(label, mutation=None, fd=True):
    mol, basis, na, ne, nc = system(label)
    ints, r = ferric_casscf(mol, basis, na, ne, nc)
    ref, chk = ref_from_casscf(ints, r, ne)
    occ = np.linalg.eigvalsh(ref.dms[1])[::-1]
    log(f"\n-- {label} {basis} CAS({sum(ne)},{na}) core {ref.nc} virt {ref.nv}")
    log(
        f"  ferric E_CASSCF {r.e_total:.12f} |g| {r.grad_norm:.1e}; occ {np.round(occ, 4)}; {chk}"
    )
    t0 = time.time()
    ew = nevpt2_wick(ref, mutation)
    t_wick = time.time() - t0
    pmol = cp.pyscf_mol(mol, basis)
    # shared inputs: pyscf gets ferric's (canonical) orbitals and CI vector
    mc, mf = pyscf_mc(
        ints, pmol, np.asarray(r.mo_coeff), np.asarray(r.ci_vector), ref.nc, na, ne
    )
    es, mo_p, ci_p = pyscf_nevpt_classes(mc)
    # same comparison on PySCF's canonical orbitals and CI (no re-diagonalization)
    ref_p = build_ref(ints, mo_p, ref.nc, na, ci_p, ne, canon=False)
    ew_p = nevpt2_wick(ref_p, mutation)
    spread = degenerate_spread(ints, r, ne, ref)
    # own CASSCF: pyscf converges on ferric's integrals from its own RHF
    mco, t_p, g_p = cp.pyscf_casscf(mf, na, ne, 1e-7)
    eo = pyscf_nevpt_classes(mco)[0]
    row = {
        "label": label,
        "basis": basis,
        "wick": ew,
        "shared": es,
        "own": eo,
        "wick_pcanon": ew_p,
        "spread": spread,
        "e_f": r.e_total,
        "e_p": mco.e_tot,
        "g_f": r.grad_norm,
        "occ": occ,
        "t_wick": t_wick,
    }
    # shared-input bar: identical (C, c); rounding of ~1e4 summed terms of
    # magnitude <= |E_class|, plus the CI transform to natural orbitals.
    bar_shared = 1e-10
    row["bar_shared"] = bar_shared
    row["max_shared"] = max(abs(ew_p[n] - es[n]) for n, _, _ in CLASSES)
    if fd:
        t0 = time.time()
        ob = orbital_bar(ints, ref, ne, np.asarray(mco.mo_coeff))
        row.update(
            grads=ob["grads"],
            gtot=ob["gtot"],
            g_f_proto=ob["g"],
            g_p=ob["g_other"],
            lam=ob["lam"],
            t_fd=time.time() - t0,
        )
        dk = (ob["g"] + ob["g_other"]) / ob["lam"]
        row["bar_own"] = {
            n: row["grads"][n] * dk + 2 * spread[n] + 1e-10 for n, _, _ in CLASSES
        }
        row["bar_own_tot"] = row["gtot"] * dk + 2 * sum(spread.values()) + 1e-10
    log(
        f"  pyscf own CASSCF E {mco.e_tot:.12f} (diff {mco.e_tot - r.e_total:.1e}) |g|_pyscf {g_p:.1e}"
    )
    for n, _, _ in CLASSES:
        s = f"  {n:5s} wick {ew[n]: .12f} shared {es[n]: .12f} d {ew[n] - es[n]: .1e}"
        s += f" d(pyscf orbs) {ew_p[n] - es[n]: .1e} degen {spread[n]:.1e}"
        s += f" | own {eo[n]: .12f} d {ew[n] - eo[n]: .1e}"
        if fd:
            s += f" bar {row['bar_own'][n]:.1e} |dE/dk| {row['grads'][n]:.2e}"
        log(s)
    tw, ts, to = sum(ew.values()), sum(es.values()), sum(eo.values())
    log(f"  total wick {tw:.12f} shared {ts:.12f} own {to:.12f}")
    if fd:
        log(
            f"  |g| ferric {row['g_f_proto']:.1e} pyscf {row['g_p']:.1e} lambda_min {row['lam']:.3e} |dE_PT2/dk| {row['gtot']:.3e}"
            f" bar_own {row['bar_own_tot']:.1e} (FD {row['t_fd']:.0f}s)"
        )
    return row


def ref_ok(row):
    ok = row["max_shared"] < row["bar_shared"]
    if "bar_own" in row:
        for n, _, _ in CLASSES:
            ok &= abs(row["wick"][n] - row["own"][n]) < row["bar_own"][n]
    return ok


MUTATIONS = (("drop_rdm3", "r"), ("flip_koopmans", "i"), ("omit", "ir"))


def run_mutations():
    log("\n== mutations: each must fail at least one check ==")
    table = []
    # The unmutated checks must pass first: a mutation counts as detected only
    # against a passing baseline. anchor_size() also sets SIZE_BAR.
    base = {
        "size": anchor_size(),
        "mp2": anchor_mp2(),
        "det": anchor_det(),
    }
    for lab in ("h2o", "n2-stretched"):
        row = compare_ref(lab, None, fd=False)
        base[lab] = row["max_shared"] < row["bar_shared"]
    failed = [k for k, v in base.items() if not v]
    if failed:
        log(f"  unmutated checks fail {failed}; mutations cannot be judged")
        return None
    for m in MUTATIONS:
        res = {}
        res["mp2"] = anchor_mp2(m)
        res["size"] = anchor_size(m, fd=False)
        res["det"] = anchor_det(m)
        for lab in ("h2o", "n2-stretched"):
            row = compare_ref(lab, m, fd=False)
            res[lab] = row["max_shared"] < row["bar_shared"]
        caught = [k for k, v in res.items() if not v]
        log(f"  mutation {m}: failing checks {caught}")
        table.append((m, res))
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only", choices=["full", "mp2", "size", "det", "ref", "mut", "cost"]
    )
    ap.add_argument("--systems", default="h2o,n2,n2-stretched")
    ap.add_argument("--no-fd", action="store_true")
    a = ap.parse_args()
    log(f"ferric build {ferric.__build__}")
    ok = True
    steps = ["full", "mp2", "size", "det", "ref", "mut"] if a.only is None else [a.only]
    for s in steps:
        if s == "cost":
            cost_report()
            r = True
        elif s == "full":
            r = anchor_full()
        elif s == "mp2":
            log("\n== anchor 2: single-determinant active space vs canonical MP2 ==")
            r = anchor_mp2()
        elif s == "size":
            r = anchor_size()
        elif s == "det":
            r = anchor_det()
        elif s == "ref":
            log("\n== per-class energies vs pyscf.mrpt.NEVPT on ferric integrals ==")
            rows = [compare_ref(lab, fd=not a.no_fd) for lab in a.systems.split(",")]
            r = all([ref_ok(x) for x in rows])
        else:
            tab = run_mutations()
            r = tab is not None and all(
                any(not v for v in res.values()) for _, res in tab
            )
        log(f"[{s}] {'PASS' if r else 'FAIL'}")
        ok &= bool(r)
        if not r and s in ("full", "mp2", "size", "det"):
            log("anchor failed; stopping")
            return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
