#!/usr/bin/env python3
"""Identities ferric-cc relies on: closed-shell (T) banding and LinLCCD spin adaptation.

Theory and notation: crates/ferric-cc/theory.md. Every claim asserts; the
script exits nonzero on any mismatch. Negative controls (deliberately wrong
variants that must NOT pass) guard against a vacuous pass.

Inputs are random, not converged CC amplitudes: each claim is an algebraic
identity that must hold for any closed-shell-structured t1/t2 and any
8-fold-symmetric RI integrals (pq|rs) = sum_P B^P_pq B^P_rs.

Claims
  1  (sympy) sum_{i<=j<=k<n} m(i,j,k) = n^3 with m = 1/3/6 for
     i=j=k / exactly two equal / all distinct
     (ccsd_t_closed_shell.rs:171 occ_triple_multiplicity).
  2  The closed-shell per-triple sum f(i,j,k) = sum_abc Wt V / D, ported from
     ccsd_t_closed_shell.rs:297 raw_w_block, :348 w_block, :376 add_v_singles
     and the Wt pattern at :601-607, is the same for all 6 orderings of (i,j,k).
  3  sum_{i<=j<=k} m f / 3  ==  sum_{i,j,k} f / 3   (the band at :611-622, the
     divisor at :634; also dlpno_ccsd_t.rs:374 screened_triple_energy).
  4  sum_{i,j,k} f / 3 equals the dense spin-orbital (T) of ccsd_t.rs (module
     doc formula, E = 1/36 sum (W+V) W / D) on the spin-orbital expansion of
     the same t1/t2/integrals. This pins the divisor 3 against an independent
     construction.
  5  Negative controls for 3/4: the spin-orbital banding transplanted
     (i<j<k, weight 6), the divisor 6, unit weights, and dropping the
     W(bca)+W(cab) terms of Wt must each FAIL.
  6  LinLCCD spin adaptation: the spin-orbital equations of linlccd.rs:345-388
       D t = <ij||ab> + 1/2 sum_kl <kl||ij> t_klab [+ 1/2 sum_cd <ab||cd> t_ijcd],
       E = 1/4 sum <ij||ab> t
     and the spatial equations of linlccd_amplitude.rs:316-357 / :391-407
       D T_iajb = (ia|jb) + sum_kl (ik|jl) T_kalb [+ sum_cd (ac|bd) T_icjd],
       E = sum (ia|jb) (2 T_iajb - T_ibja)
     give the same energy for drivers-only, hh and full, and
     T_iajb = t[i alpha, j beta, a alpha, b beta].
  7  Negative controls for 6: hh coefficient 1 instead of 1/2 (spin-orbital)
     and (ij|kl) instead of (ik|jl) (spatial) must each FAIL.

Run: python3 scripts/cc_triples_weights_and_linlccd_spin_adaptation_sympy.py
"""

import itertools
import sys

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


# ---------------------------------------------------------------- claim 1
def multiplicity(i, j, k):
    if i == k:
        return 1
    if i == j or j == k:
        return 3
    return 6


n = sp.symbols("n", integer=True, positive=True)
# classes: i=j=k (n), exactly two equal (2 * C(n,2) banded triples), distinct (C(n,3))
count = n * 1 + 2 * sp.binomial(n, 2) * 3 + sp.binomial(n, 3) * 6
check("claim 1 (sympy): sum m = n^3", sp.simplify(sp.expand_func(count) - n**3) == 0)
for nn in range(1, 7):
    s = sum(
        multiplicity(i, j, k)
        for i in range(nn)
        for j in range(i, nn)
        for k in range(j, nn)
    )
    assert s == nn**3, (nn, s)

# ---------------------------------------------------------------- inputs
rng = np.random.default_rng(31)
no, nv, naux = 4, 4, 14
nmo = no + nv
B = rng.normal(size=(naux, nmo, nmo))
B = 0.5 * (B + B.transpose(0, 2, 1))
B *= 0.15
eri = np.einsum("Ppq,Prs->pqrs", B, B)  # chemist (pq|rs), 8-fold symmetric
eps_o = np.sort(rng.uniform(-2.0, -0.5, no))
eps_v = np.sort(rng.uniform(0.3, 2.0, nv))
o = slice(0, no)
v = slice(no, nmo)
t1 = 0.1 * rng.normal(size=(no, nv))
x = 0.1 * rng.normal(size=(no, no, nv, nv))
t2 = x + x.transpose(1, 0, 3, 2)  # closed-shell t2[i,j,a,b] = t2[j,i,b,a]

ovvv = eri[o, v, v, v]  # (ia|bd) at [i,a,b,d]
ovoo = eri[o, v, o, o]  # (ia|lj) at [i,a,l,j]
ovov = eri[o, v, o, v]  # (ia|jb) at [i,a,j,b]

PERMS3 = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


def raw_w(i, j, k):
    """ccsd_t_closed_shell.rs:297 raw_w_block."""
    term1 = np.einsum("abd,cd->abc", ovvv[i], t2[k, j])
    term2 = np.einsum("al,lbc->abc", ovoo[i, :, :, j], t2[:, k])
    return term1 - term2


def w_block(i, j, k):
    """ccsd_t_closed_shell.rs:348 w_block (np.transpose == ndarray permuted_axes)."""
    occ = (i, j, k)
    out = np.zeros((nv, nv, nv))
    for p in PERMS3:
        r = raw_w(occ[p[0]], occ[p[1]], occ[p[2]])
        inv = [0, 0, 0]
        for t, pt in enumerate(p):
            inv[pt] = t
        out += np.transpose(r, inv)
    return out


def f_triple(i, j, k, drop_cyclic=False):
    w = w_block(i, j, k)
    vv = w.copy()  # add_v_singles, :376
    vv += np.einsum("bc,a->abc", ovov[j, :, k, :], t1[i])
    vv += np.einsum("ac,b->abc", ovov[i, :, k, :], t1[j])
    vv += np.einsum("ab,c->abc", ovov[i, :, j, :], t1[k])
    wt = 4.0 * w
    if not drop_cyclic:
        wt += np.transpose(w, [1, 2, 0]) + np.transpose(w, [2, 0, 1])
    wt -= 2.0 * (
        np.transpose(w, [2, 1, 0]) + np.transpose(w, [0, 2, 1]) + np.transpose(w, [1, 0, 2])
    )
    d = (
        eps_o[i]
        + eps_o[j]
        + eps_o[k]
        - eps_v[:, None, None]
        - eps_v[None, :, None]
        - eps_v[None, None, :]
    )
    return float(np.sum(wt * vv / d))


# ---------------------------------------------------------------- claim 2
f = {}
for i, j, k in itertools.product(range(no), repeat=3):
    f[(i, j, k)] = f_triple(i, j, k)
dev2 = 0.0
for i in range(no):
    for j in range(i, no):
        for k in range(j, no):
            vals = [f[tuple((i, j, k)[q] for q in p)] for p in PERMS3]
            dev2 = max(dev2, max(vals) - min(vals))
scale = max(abs(z) for z in f.values())
check("claim 2: f invariant over the 6 orderings", dev2 < 1e-13 * scale, f"(max spread {dev2:.2e})")

# ---------------------------------------------------------------- claim 3
unrestricted = sum(f.values())
banded = sum(
    multiplicity(i, j, k) * f[(i, j, k)]
    for i in range(no)
    for j in range(i, no)
    for k in range(j, no)
)
e_cs = unrestricted / 3.0
check(
    "claim 3: banded m-weighted sum == unrestricted sum",
    abs(banded - unrestricted) < 1e-12 * abs(unrestricted),
    f"({banded:.15e} vs {unrestricted:.15e})",
)
repeated = sum(
    multiplicity(i, j, k) * f[(i, j, k)]
    for i in range(no)
    for j in range(i, no)
    for k in range(j, no)
    if len({i, j, k}) < 3
)
print(f"       repeated-index share of the banded sum: {repeated / banded:.3f}")

# ---------------------------------------------------------------- claim 4
no2, nv2 = 2 * no, 2 * nv
nso = no2 + nv2


def spat(p):
    return p // 2 if p < no2 else no + (p - no2) // 2


def spin(p):
    return p % 2 if p < no2 else (p - no2) % 2


sp_idx = np.array([spat(p) for p in range(nso)])
sg = np.array([spin(p) for p in range(nso)])
same = (sg[:, None] == sg[None, :]).astype(float)
g_sp = eri[np.ix_(sp_idx, sp_idx, sp_idx, sp_idx)]  # (p r | q s) by spatial index
# <pq||rs> = (pr|qs) d(p,r) d(q,s) - (ps|qr) d(p,s) d(q,r)
direct = np.einsum("prqs,pr,qs->pqrs", g_sp, same, same)
exch = np.einsum("psqr,ps,qr->pqrs", g_sp, same, same)
asym = direct - exch
O = slice(0, no2)
V = slice(no2, nso)

oi = np.array([spat(p) for p in range(no2)])
va = np.array([spat(p) - no for p in range(no2, nso)])
so_o = sg[:no2]
so_v = sg[no2:]
t2so = np.einsum(
    "ijab,ia,jb->ijab",
    t2[np.ix_(oi, oi, va, va)],
    (so_o[:, None] == so_v[None, :]).astype(float),
    (so_o[:, None] == so_v[None, :]).astype(float),
) - np.einsum(
    "ijba,ib,ja->ijab",
    t2[np.ix_(oi, oi, va, va)],
    (so_o[:, None] == so_v[None, :]).astype(float),
    (so_o[:, None] == so_v[None, :]).astype(float),
)
t1so = t1[np.ix_(oi, va)] * (so_o[:, None] == so_v[None, :])
assert np.allclose(t2so, -t2so.transpose(1, 0, 2, 3))
assert np.allclose(t2so, -t2so.transpose(0, 1, 3, 2))


def p_ijk(x):
    return x - x.transpose(1, 0, 2, 3, 4, 5) - x.transpose(2, 1, 0, 3, 4, 5)


def p_abc(x):
    return x - x.transpose(0, 1, 2, 4, 3, 5) - x.transpose(0, 1, 2, 5, 4, 3)


raw = np.einsum("jkae,bcei->ijkabc", t2so, asym[V, V, V, O]) - np.einsum(
    "imbc,majk->ijkabc", t2so, asym[O, V, O, O]
)
W = p_ijk(p_abc(raw))
Vd = p_ijk(p_abc(np.einsum("ia,bcjk->ijkabc", t1so, asym[V, V, O, O])))
eo_so = eps_o[oi]
ev_so = eps_v[va]
D = (
    eo_so[:, None, None, None, None, None]
    + eo_so[None, :, None, None, None, None]
    + eo_so[None, None, :, None, None, None]
    - ev_so[None, None, None, :, None, None]
    - ev_so[None, None, None, None, :, None]
    - ev_so[None, None, None, None, None, :]
)
e_so = float(np.sum((W + Vd) * W / D) / 36.0)
check(
    "claim 4: closed-shell sum/3 == spin-orbital (T)",
    abs(e_cs - e_so) < 1e-12 * abs(e_so),
    f"({e_cs:.15e} vs {e_so:.15e})",
)

# ---------------------------------------------------------------- claim 5
distinct_x6 = sum(
    6 * f[(i, j, k)] for i in range(no) for j in range(i + 1, no) for k in range(j + 1, no)
)
control("i<j<k band x6 (spin-orbital banding)", distinct_x6 / 3.0, e_so, 1e-3)
control("divisor 6", banded / 6.0, e_so, 1e-3)
control(
    "unit weights",
    sum(f[(i, j, k)] for i in range(no) for j in range(i, no) for k in range(j, no)) / 3.0,
    e_so,
    1e-3,
)
no_cyc = sum(f_triple(i, j, k, drop_cyclic=True) for i, j, k in itertools.product(range(no), repeat=3))
control("Wt without W(bca)+W(cab)", no_cyc / 3.0, e_so, 1e-3)

# ---------------------------------------------------------------- claim 6
no_l, nv_l = 3, 4  # LinLCCD: smaller so the spin-orbital system solves densely
nmo_l = no_l + nv_l
Bl = rng.normal(size=(12, nmo_l, nmo_l))
Bl = 0.08 * (Bl + Bl.transpose(0, 2, 1))
g = np.einsum("Ppq,Prs->pqrs", Bl, Bl)
eo = np.sort(rng.uniform(-2.0, -0.6, no_l))
ev = np.sort(rng.uniform(0.4, 2.0, nv_l))
ol, vl = slice(0, no_l), slice(no_l, nmo_l)


def spinorbital_linlccd(variant, hh_coeff=0.5):
    """linlccd.rs:345-388, solved as one dense linear system."""
    n2o, n2v = 2 * no_l, 2 * nv_l

    def asym_block(p_sp, q_sp, r_sp, s_sp):
        # p_sp etc: (spatial index array, spin array) per axis; <pq||rs>
        (pi, ps), (qi, qs), (ri, rs), (si, ss) = p_sp, q_sp, r_sp, s_sp
        dirr = g[np.ix_(pi, ri, qi, si)].transpose(0, 2, 1, 3) * (
            (ps[:, None, None, None] == rs[None, None, :, None])
            & (qs[None, :, None, None] == ss[None, None, None, :])
        )
        exc = g[np.ix_(pi, si, qi, ri)].transpose(0, 2, 3, 1) * (
            (ps[:, None, None, None] == ss[None, None, None, :])
            & (qs[None, :, None, None] == rs[None, None, :, None])
        )
        return dirr - exc

    occ = (np.repeat(np.arange(no_l), 2), np.tile([0, 1], no_l))
    vir = (no_l + np.repeat(np.arange(nv_l), 2), np.tile([0, 1], nv_l))
    voovv = asym_block(occ, occ, vir, vir)
    oooo = asym_block(occ, occ, occ, occ)
    vvvv = asym_block(vir, vir, vir, vir)
    e_o = eo[occ[0]]
    e_v = ev[vir[0] - no_l]
    d = e_o[:, None, None, None] + e_o[None, :, None, None] - e_v[None, None, :, None] - e_v[None, None, None, :]
    nn = n2o * n2o * n2v * n2v
    A = np.diag(d.reshape(nn))
    if variant in ("hh", "full"):
        A -= hh_coeff * np.einsum("klij,ac,bd->ijabklcd", oooo, np.eye(n2v), np.eye(n2v)).reshape(nn, nn)
    if variant == "full":
        A -= 0.5 * np.einsum("abcd,ik,jl->ijabklcd", vvvv, np.eye(n2o), np.eye(n2o)).reshape(nn, nn)
    t = np.linalg.solve(A, voovv.reshape(nn)).reshape(d.shape)
    return 0.25 * float(np.sum(voovv * t)), t


def spatial_linlccd(variant, hh_exchange_bug=False):
    """linlccd_amplitude.rs:316-357 (operator) and :391-407 (energy)."""
    iajb = g[ol, vl, ol, vl]
    d = eo[:, None, None, None] - ev[None, :, None, None] + eo[None, None, :, None] - ev[None, None, None, :]
    nn = no_l * nv_l * no_l * nv_l
    A = np.diag(d.reshape(nn))
    if variant in ("hh", "full"):
        if hh_exchange_bug:
            coef = g[ol, ol, ol, ol]  # (ij|kl) at [i,j,k,l]: wrong pairing
            A -= np.einsum("ijkl,ac,bd->iajbkcld", coef, np.eye(nv_l), np.eye(nv_l)).reshape(nn, nn)
        else:
            ikjl = g[ol, ol, ol, ol]  # (ik|jl) at [i,k,j,l]
            A -= np.einsum("ikjl,ac,bd->iajbkcld", ikjl, np.eye(nv_l), np.eye(nv_l)).reshape(nn, nn)
    if variant == "full":
        acbd = g[vl, vl, vl, vl]  # (ac|bd) at [a,c,b,d]
        A -= np.einsum("acbd,ik,jl->iajbkcld", acbd, np.eye(no_l), np.eye(no_l)).reshape(nn, nn)
    T = np.linalg.solve(A, iajb.reshape(nn)).reshape(iajb.shape)
    e = float(np.sum(iajb * (2.0 * T - T.transpose(0, 3, 2, 1))))
    return e, T


for variant in ("drivers-only", "hh", "full"):
    e_so_l, t_so = spinorbital_linlccd(variant)
    e_sp_l, T_sp = spatial_linlccd(variant)
    check(
        f"claim 6: LinLCCD {variant} spin-orbital E == spatial E",
        abs(e_so_l - e_sp_l) < 1e-12 * abs(e_so_l),
        f"({e_so_l:.15e} vs {e_sp_l:.15e})",
    )
    # T_iajb = t[i alpha, j beta, a alpha, b beta] (even = alpha, odd = beta)
    t_ab = t_so[0::2, 1::2, 0::2, 1::2]  # [i,j,a,b]
    dev = np.max(np.abs(t_ab.transpose(0, 2, 1, 3) - T_sp))
    check(f"claim 6: LinLCCD {variant} T_iajb == t(i a, j b, a a, b b)", dev < 1e-12, f"(max dev {dev:.1e})")

# ---------------------------------------------------------------- claim 7
e_ref, _ = spatial_linlccd("hh")
control("spin-orbital hh coefficient 1 (not 1/2)", spinorbital_linlccd("hh", hh_coeff=1.0)[0], e_ref, 1e-6)
control("spatial hh with (ij|kl) for (ik|jl)", spatial_linlccd("hh", hh_exchange_bug=True)[0], e_ref, 1e-6)

if FAILS:
    print(f"\n{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("\nall claims verified")
