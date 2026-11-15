#!/usr/bin/env python3
"""Sympy second-quantization proofs for the MP3 ring prefactor and the OO-MP2 orbital gradient.

Checks, in spin-orbital notation (<pq||rs> = g[p,q,r,s], t_ij^ab = t[a,b,i,j]):

  A  MP3 (crates/ferric-mp2/src/mp3.rs, mp3_energy).
     E3 = <T2| V_N |T2> with T2 the first-order doubles. Every vertex is built
     with sympy.physics.secondquant.wicks over DISTINCT dummy indices (bra,
     vertex and ket never share a symbol; sharing symbols is what produced a
     spurious 1/4 for the ring term).
       A1  hh and pp ladders: 1/8 on the full unrestricted sums, as in mp3.rs
           (e_pp, e_hh).
       A2  Particle-hole ring. The OVVO block of V_N = 1/4 g^{pq}_{rs}{p+q+sr}
           has FOUR orderings of (hole, particle) in the creators and in the
           annihilators, equal by antisymmetry, so the vertex carries
           4 * 1/4 = 1. Wick gives  -g^{a2 i2}_{a i} t^{ab}_{i2 j} t^{a2 b}_{ij}
           with coefficient exactly 1 on the full unrestricted sums, and that
           equals mp3.rs's `einsum!("iakc,iakc->", z_t, t_iakc_t)` with
           z[i,a,k,c] = sum_{jb} t[i,j,a,b] ovvo[k,b,c,j], coefficient 1. A
           single ordering gives 1/4 of it (the 1/4 the wiki used to quote).
       A3  Brute force: Rayleigh-Schroedinger E2 and E3 = <psi1|(V - E1)|psi1>
           in the full 7-spin-orbital, 3-electron determinant space (35
           determinants, Jordan-Wigner), for a random antisymmetrized
           Hamiltonian with a diagonal Fock operator. E2 = 1/4 <ij||ab> t and
           E3 = e_pp + e_hh + e_ph of mp3.rs, to rounding.

  B  OO-MP2 orbital gradient (crates/ferric-mp2/src/oo_rimp2.rs).
       B1  Hylleraas functional J(t) = <T2|V_N|0> + <0|V_N|T2> + <T2|F_N|T2>
           (doubles only, full non-canonical Fock blocks f_oo, f_vv) and the
           reference energy E_HF = <0|H|0>, both from wicks. J is stationary
           at the amplitudes t = <ab||ij>/(e_i+e_j-e_a-e_b) of the code
           (frame where f_oo, f_vv are diagonal) and J(t*) = E2.
       B2  dE/dkappa_ai at fixed t (the Hylleraas theorem: the t-response
           vanishes). With E = sum h gamma + sum g Gamma, the densities are
           obtained by differentiating the sympy expressions with respect to
           the h, f and g tensors (f_pq = h_pq + sum_k g_pkqk, so the
           unrelaxed MP2 density W = dJ/df enters Gamma at (p,k,q,k)):
             gamma_pq = [p=q occ] + W_pq
             Gamma    = 1/2 [pqrs = ijij] + W_pq [r=k, s=k] + Gamma_t,
             Gamma_t(ijab) = Gamma_t(abij) = 1/2 t^{ab}_{ij}  (from J)
           Generalized Fock X_tp = sum_q h_tq gamma_pq + h_qt gamma_qp
             + 4 index placements of g.Gamma;  g_ai = dE/dkappa_ai = X_ai - X_ia
           for the rotation C -> C(1 + kappa), kappa_ai = x, kappa_ia = -x (the
           canonical Cayley of orbital_rotation.rs, U = (1 - k/2)^-1 (1 + k/2)).
           Checked against (i) exact 5-point finite differences of E(kappa) at
           fixed t (E is a degree-4 polynomial in kappa, the stencil is exact),
           (ii) finite differences of the fully relaxed E_HF + E2 (t re-solved
           in the rotated frame), (iii) a finite difference through the real
           Cayley transform.
       B3  The closed-shell loop that oo_rimp2.rs codes
             g_ck = 4 F_ck + 2 (G_V + G_O)_ck                  (hf_and_integral_response_gradient)
                  + 4 [G[W]]_ck + 2 (F W - W F)_ck             (fock_response_gradient)
           transcribed from orbital_gradient_scalar_reference and
           fock_response_gradient with W = P + P^T from the
           mp2_density_scalar_reference formulas, on a random closed-shell
           model in the semicanonical frame with F_ov != 0 (so the Brillouin
           term is live), against dE/dkappa_ck of the spin-orbital energy with
           the spatial kappa applied to both spins.

The checks the book cites carry an explicit label "(claim A2)", "(claim A3)",
"(claim B0)" or "(claim B1)" in their printed name; book/ boxes cite these
labels (claim=B0), not the whole quoted string.

Residuals are printed. Exits nonzero on the first failed identity (all are run).

Mutations, each of which turns this script red (set OOMP2_MP3_MUTATION):
  ring_quarter     ring coefficient 1 -> 1/4 (A2 and A3 fail)
  kappa_sign       rotation generator C(1 - kappa) in the FD and chain-rule
                   paths (B2 and B3 fail: the analytic gradient changes sign
                   relative to the energy derivative)
  drop_two_particle  the correlation two-particle density Gamma_t removed from
                   the generalized Fock (B2 and B3 fail)

Measured on this tree: all three exit 1; the unmutated run exits 0.
"""

import itertools
import os
import sys

import numpy as np
from sympy import Add, Dummy, Mul, Rational, symbols
from sympy.physics.secondquant import (
    NO,
    AntiSymmetricTensor,
    F,
    Fd,
    evaluate_deltas,
    substitute_dummies,
    wicks,
)

MUTATION = os.environ.get("OOMP2_MP3_MUTATION", "")
if MUTATION not in ("", "ring_quarter", "kappa_sign", "drop_two_particle"):
    sys.exit(f"unknown OOMP2_MP3_MUTATION {MUTATION!r}")
RING_COEFF = Rational(1, 4) if MUTATION == "ring_quarter" else Rational(1)
KAPPA_SIGN = -1.0 if MUTATION == "kappa_sign" else 1.0
DROP_GAMMA_T = MUTATION == "drop_two_particle"

failures = 0


def check(label, ok, detail=""):
    global failures
    print(f"{'ok  ' if ok else 'FAIL'} {label}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        failures += 1


# ---------------------------------------------------------------- evaluator
def _cls(s):
    a = s.assumptions0
    if a.get("below_fermi"):
        return "o"
    return "v" if a.get("above_fermi") else "p"


def _ranges(no, nv):
    n = no + nv
    return {"o": np.arange(no), "v": np.arange(no, n), "p": np.arange(n)}


def evaluate(expr, tensors, no, nv, free=None):
    """Numerically evaluate a sympy expression that is a sum of products of
    AntiSymmetricTensor factors. `tensors` maps a tensor name to a full
    (n,)*rank array; each index is restricted to the range of its Fermi class
    (below_fermi -> occupied, above_fermi -> virtual, else all). With `free`
    (a list of index symbols) the result is the array over those indices
    instead of a scalar."""
    rng = _ranges(no, nv)
    free = free or []
    shape = tuple(len(rng[_cls(s)]) for s in free)
    tot = np.zeros(shape) if free else 0.0
    for term in Add.make_args(expr):
        coef, facs = 1, []
        for f in Mul.make_args(term):
            if isinstance(f, AntiSymmetricTensor):
                facs.append(f)
            else:
                coef *= f
        letters = {}
        for s in free:
            letters.setdefault(s, chr(97 + len(letters)))
        ops, subs = [], []
        for f in facs:
            idx = list(f.upper) + list(f.lower)
            for s in idx:
                letters.setdefault(s, chr(97 + len(letters)))
            arr = tensors[str(f.symbol)][np.ix_(*[rng[_cls(s)] for s in idx])]
            ops.append(arr)
            subs.append("".join(letters[s] for s in idx))
        out = "".join(letters[s] for s in free)
        tot = tot + float(coef) * np.einsum(
            ",".join(subs) + "->" + out, *ops, optimize=True
        )
    return tot


def derivative_array(expr, name, tensors, no, nv):
    """d expr / d (entries of tensor `name`) as a full (n,)*rank array: for
    each term, the factor `name` is removed and the rest evaluated with that
    factor's indices free, then scattered into the factor's index ranges."""
    n = no + nv
    rng = _ranges(no, nv)
    out = None
    for term in Add.make_args(expr):
        coef, facs, target = 1, [], None
        for f in Mul.make_args(term):
            if isinstance(f, AntiSymmetricTensor):
                if str(f.symbol) == name and target is None:
                    target = f
                else:
                    facs.append(f)
            else:
                coef *= f
        if target is None:
            continue
        idx = list(target.upper) + list(target.lower)
        if out is None:
            out = np.zeros((n,) * len(idx))
        uniq = list(dict.fromkeys(idx))  # h_ii-type factors repeat a symbol
        rest = coef
        for f in facs:
            rest = rest * f
        if facs:
            val = evaluate(rest, tensors, no, nv, free=uniq)
        else:
            val = float(coef) * np.ones(tuple(len(rng[_cls(s)]) for s in uniq))
        grids = np.meshgrid(*[rng[_cls(s)] for s in uniq], indexing="ij")
        out[tuple(grids[uniq.index(s)] for s in idx)] += val
    return out


def antisym_model(n, seed):
    """Random <pq||rs> with g = -g(qp rs) = -g(pq sr) = g(rs pq)."""
    rs = np.random.RandomState(seed)
    g = rs.randn(n, n, n, n)
    g = g - g.transpose(1, 0, 2, 3)
    g = g - g.transpose(0, 1, 3, 2)
    return g + g.transpose(2, 3, 0, 1)


def amplitudes(g, eps, no, nv):
    n = no + nv
    t = np.zeros((n,) * 4)
    for a, b, i, j in itertools.product(
        range(no, n), range(no, n), range(no), range(no)
    ):
        t[a, b, i, j] = g[a, b, i, j] / (eps[i] + eps[j] - eps[a] - eps[b])
    return t


def dummies():
    d = {}
    for nm in ("i", "j", "k", "l", "i2", "j2", "k2", "l2", "kk", "jj", "m", "m2"):
        d[nm] = symbols(nm, cls=Dummy, below_fermi=True)
    for nm in ("a", "b", "c", "d", "a2", "b2", "c2", "d2", "bb", "cc", "e", "e2"):
        d[nm] = symbols(nm, cls=Dummy, above_fermi=True)
    return d


def wick_expect(op):
    return substitute_dummies(
        evaluate_deltas(wicks(op, keep_only_fully_contracted=True))
    )


# ====================================================================== A: MP3
def part_a():
    print("== A. MP3 ladder and ring prefactors")
    D = dummies()
    i, j, a, b = D["i"], D["j"], D["a"], D["b"]
    i2, j2, a2, b2 = D["i2"], D["j2"], D["a2"], D["b2"]
    q = Rational(1, 4)
    t_bra = q * AntiSymmetricTensor("t", (a, b), (i, j)) * Fd(i) * Fd(j) * F(b) * F(a)
    t_ket = (
        q
        * AntiSymmetricTensor("t", (a2, b2), (i2, j2))
        * Fd(a2)
        * Fd(b2)
        * F(j2)
        * F(i2)
    )

    # vertices, all indices fresh
    c1, c2, c3, c4 = (
        symbols(f"p{n}", cls=Dummy, above_fermi=True) for n in range(1, 5)
    )
    o1, o2, o3, o4 = (
        symbols(f"h{n}", cls=Dummy, below_fermi=True) for n in range(1, 5)
    )
    v_pp = (
        q
        * AntiSymmetricTensor("g", (c1, c2), (c3, c4))
        * NO(Fd(c1) * Fd(c2) * F(c4) * F(c3))
    )
    v_hh = (
        q
        * AntiSymmetricTensor("g", (o1, o2), (o3, o4))
        * NO(Fd(o1) * Fd(o2) * F(o4) * F(o3))
    )
    pp = wick_expect(t_bra * v_pp * t_ket)
    hh = wick_expect(t_bra * v_hh * t_ket)

    # ph vertex: every (hole, particle) ordering of creators and annihilators,
    # each with the 1/4 of V_N = 1/4 g^{pq}_{rs} {p+ q+ s r}
    k_, b_ = (
        symbols("kk", cls=Dummy, below_fermi=True),
        symbols("bb", cls=Dummy, above_fermi=True),
    )
    c_, j_ = (
        symbols("cc", cls=Dummy, above_fermi=True),
        symbols("jj", cls=Dummy, below_fermi=True),
    )
    orderings = []
    for cre in ((k_, b_), (b_, k_)):
        for ann in ((c_, j_), (j_, c_)):
            p_, q_ = cre
            r_, s_ = ann
            orderings.append(
                q
                * AntiSymmetricTensor("g", (p_, q_), (r_, s_))
                * NO(Fd(p_) * Fd(q_) * F(s_) * F(r_))
            )
    ph_each = [wick_expect(t_bra * v * t_ket) for v in orderings]
    ph_full = Add(*ph_each)

    no, nv = 3, 4
    n = no + nv
    g = antisym_model(n, 11)
    rs = np.random.RandomState(5)
    eps = np.concatenate([-1.0 - rs.rand(no), 0.5 + rs.rand(nv)])
    t = amplitudes(g, eps, no, nv)
    T = {"g": g, "t": t}

    # mp3.rs formulas, same index order as the einsum! calls
    tt = t[no:, no:, :no, :no].transpose(2, 3, 0, 1)  # t[i,j,a,b]
    oovv = g[:no, :no, no:, no:]
    vvvv, oooo = g[no:, no:, no:, no:], g[:no, :no, :no, :no]
    ovvo = g[:no, no:, no:, :no]  # <kb||cj>
    e_mp2 = 0.25 * np.einsum("ijab,ijab->", tt, oovv)
    e_pp = 0.125 * np.einsum("ijcd,ijcd->", np.einsum("ijab,abcd->ijcd", tt, vvvv), tt)
    e_hh = 0.125 * np.einsum("klab,klab->", np.einsum("klij,ijab->klab", oooo, tt), tt)
    z = np.einsum("ijab,kbcj->iakc", tt, ovvo)
    t_iakc = tt.transpose(0, 2, 1, 3)
    ring_raw = np.einsum("iakc,iakc->", z, t_iakc)
    e_ph = float(RING_COEFF) * ring_raw

    s_pp, s_hh, s_ph = (
        evaluate(pp, T, no, nv),
        evaluate(hh, T, no, nv),
        evaluate(ph_full, T, no, nv),
    )
    check(
        "A1 pp ladder: wicks == 1/8 t.<ab||cd>.t (mp3.rs e_pp)",
        abs(s_pp - e_pp) < 1e-11,
        f"{abs(s_pp - e_pp):.1e}",
    )
    check(
        "A1 hh ladder: wicks == 1/8 <kl||ij> t t (mp3.rs e_hh)",
        abs(s_hh - e_hh) < 1e-11,
        f"{abs(s_hh - e_hh):.1e}",
    )
    coeff_sym = s_ph / ring_raw
    check(
        "A2 ring: wicks (four orderings) == coefficient 1 on the mp3.rs contraction (claim A2)",
        abs(coeff_sym - 1.0) < 1e-11,
        f"coefficient {coeff_sym:.12f}",
    )
    check(
        "A2 ring: the mp3.rs coefficient equals the derived one",
        abs(float(RING_COEFF) - coeff_sym) < 1e-11,
        f"mp3.rs {float(RING_COEFF)} vs derived {coeff_sym:.12f}",
    )
    single = evaluate(ph_each[0], T, no, nv) / ring_raw
    check(
        "A2 negative control: one ordering alone carries 1/4 (the former wiki value)",
        abs(single - 0.25) < 1e-11
        and all(
            abs(evaluate(x, T, no, nv) - evaluate(ph_each[0], T, no, nv)) < 1e-11
            for x in ph_each
        ),
        f"single ordering {single:.12f}",
    )

    # ---- brute-force Rayleigh-Schroedinger in the determinant space
    nso = n
    dim = 2**nso
    ann = []
    for p in range(nso):
        m = np.zeros((dim, dim))
        for s in range(dim):
            if (s >> p) & 1:
                sign = (-1) ** bin(s & ((1 << p) - 1)).count("1")
                m[s ^ (1 << p), s] = sign
        ann.append(m)
    cre = [m.T for m in ann]
    hmat = np.diag(eps) - np.einsum(
        "pkqk->pq", g[:, :no, :, :no]
    )  # f = h + sum_k <pk||qk> = diag(eps)
    pair = np.empty((nso, nso, dim, dim))
    pair_d = np.empty((nso, nso, dim, dim))
    for p in range(nso):
        for r in range(nso):
            pair[p, r] = ann[p] @ ann[r]  # a_s a_r with (s, r) = (p, r)
            pair_d[p, r] = cre[p] @ cre[r]
    h_op = sum(hmat[p, r] * (cre[p] @ ann[r]) for p in range(nso) for r in range(nso))
    v2 = np.zeros((dim, dim))
    for p in range(nso):
        for q_ in range(nso):
            inner = np.einsum("rs,srxy->xy", g[p, q_], pair)
            v2 += 0.25 * pair_d[p, q_] @ inner
    h_full = h_op + v2
    det_e = np.array(
        [sum(eps[p] for p in range(nso) if (s >> p) & 1) for s in range(dim)]
    )
    nel = np.array([bin(s).count("1") for s in range(dim)])
    ref = (1 << no) - 1
    h0 = np.diag(det_e)
    v = h_full - h0
    phi = np.zeros(dim)
    phi[ref] = 1.0
    e0 = det_e[ref]
    vphi = v @ phi
    e1 = phi @ vphi
    psi1 = np.zeros(dim)
    for s in range(dim):
        if s != ref and abs(vphi[s]) > 1e-14:
            assert nel[s] == no
            psi1[s] = vphi[s] / (e0 - det_e[s])
    e2_rs = phi @ v @ psi1
    e3_rs = psi1 @ (v - e1 * np.eye(dim)) @ psi1
    check(
        "A3 RS E2 == 1/4 <ij||ab> t (singles vanish by the diagonal Fock operator) (claim A3)",
        abs(e2_rs - e_mp2) < 1e-11,
        f"{abs(e2_rs - e_mp2):.1e}",
    )
    e3_mp3 = e_pp + e_hh + e_ph
    check(
        "A3 RS E3 == e_pp + e_hh + e_ph of mp3.rs",
        abs(e3_rs - e3_mp3) < 1e-10,
        f"RS {e3_rs:.10f}, formula {e3_mp3:.10f}, diff {abs(e3_rs - e3_mp3):.1e}",
    )
    parts = {"pp": e_pp, "hh": e_hh, "ph": e_ph}
    print("    E3 parts:", ", ".join(f"{k} {v_:+.8f}" for k, v_ in parts.items()))


# ================================================================ B: OO-MP2
def sympy_functionals():
    """J (Hylleraas, doubles) and E_HF as sympy expressions via wicks."""
    D = dummies()
    i, j, a, b = D["i"], D["j"], D["a"], D["b"]
    i2, j2, a2, b2 = D["i2"], D["j2"], D["a2"], D["b2"]
    k, l2_, c, d = D["k"], D["l"], D["c"], D["d"]
    q = Rational(1, 4)
    t_ket = q * AntiSymmetricTensor("t", (a, b), (i, j)) * Fd(a) * Fd(b) * F(j) * F(i)
    t_bra = (
        q
        * AntiSymmetricTensor("t", (a2, b2), (i2, j2))
        * Fd(i2)
        * Fd(j2)
        * F(b2)
        * F(a2)
    )
    # V_N pieces that connect to the reference through doubles
    p1, p2, h1, h2 = D["c2"], D["d2"], D["k2"], D["l2"]
    v_deex = (
        q
        * AntiSymmetricTensor("g", (h1, h2), (p1, p2))
        * NO(Fd(h1) * Fd(h2) * F(p2) * F(p1))
    )
    q1, q2, m1, m2 = D["e"], D["e2"], D["m"], D["m2"]
    v_exc = (
        q
        * AntiSymmetricTensor("g", (q1, q2), (m1, m2))
        * NO(Fd(q1) * Fd(q2) * F(m2) * F(m1))
    )
    vt_deex = wick_expect(v_deex * t_ket)
    vt_exc = wick_expect(t_bra * v_exc)
    f_vv = AntiSymmetricTensor("f", (c,), (d,)) * NO(Fd(c) * F(d))
    f_oo = AntiSymmetricTensor("f", (k,), (l2_,)) * NO(Fd(k) * F(l2_))
    tft_vv = wick_expect(t_bra * f_vv * t_ket)
    tft_oo = wick_expect(t_bra * f_oo * t_ket)
    j_expr = vt_deex + vt_exc + tft_vv + tft_oo
    # E_HF = <0|H|0> = sum_i h_ii + 1/2 sum_ij <ij||ij>
    o1, o2, o3, o4 = (
        symbols(f"r{n}", cls=Dummy, below_fermi=True) for n in range(1, 5)
    )
    e1 = wick_expect(AntiSymmetricTensor("h", (o1,), (o2,)) * Fd(o1) * F(o2))
    e2 = wick_expect(
        q
        * AntiSymmetricTensor("g", (o1, o2), (o3, o4))
        * Fd(o1)
        * Fd(o2)
        * F(o4)
        * F(o3)
    )
    return (
        {"vt_deex": vt_deex, "vt_exc": vt_exc, "tft_vv": tft_vv, "tft_oo": tft_oo},
        j_expr,
        e1 + e2,
    )


def fock_from(h, g, no):
    return h + np.einsum("pkqk->pq", g[:, :no, :, :no])


def rotate(h, g, u):
    h2 = u.T @ h @ u
    g2 = np.einsum("ap,bq,cr,ds,abcd->pqrs", u, u, u, u, g, optimize=True)
    return h2, g2


def d_rotate(h, g, kap):
    """First-order change of h and g under C -> C (1 + kap)."""
    dh = kap.T @ h + h @ kap
    dg = (
        np.einsum("tp,tqrs->pqrs", kap, g)
        + np.einsum("tq,ptrs->pqrs", kap, g)
        + np.einsum("tr,pqts->pqrs", kap, g)
        + np.einsum("ts,pqrt->pqrs", kap, g)
    )
    return dh, dg


def kappa_matrix(x_ai, no, nv):
    n = no + nv
    kap = np.zeros((n, n))
    kap[no:, :no] = KAPPA_SIGN * x_ai
    kap[:no, no:] = -KAPPA_SIGN * x_ai.T
    return kap


def solve_t(h, g, no, nv, n_iter=400):
    """Stationary doubles of J in the frame of (h, g): g_abij + P(ab) f_bc t_ijac
    - P(ij) f_kj t_ikab = 0, by Jacobi iteration on the diagonal of f."""
    f = fock_from(h, g, no)
    e = np.diag(f)
    t = amplitudes(g, e, no, nv)
    gab = g[no:, no:, :no, :no]
    fvv, foo = f[no:, no:], f[:no, :no]
    dvv = np.diag(fvv)
    doo = np.diag(foo)
    den = (doo[None, None, :, None] + doo[None, None, None, :]) - (
        dvv[:, None, None, None] + dvv[None, :, None, None]
    )
    tv = t[no:, no:, :no, :no]
    off_vv = fvv - np.diag(dvv)
    off_oo = foo - np.diag(doo)
    for _ in range(n_iter):
        r = np.einsum("bc,acij->abij", off_vv, tv)
        r = r - r.transpose(1, 0, 2, 3)
        s = np.einsum("kj,abik->abij", off_oo, tv)
        s = s - s.transpose(0, 1, 3, 2)
        # 0 = g + (e_a+e_b - e_i - e_j) t + r - s  ->  t = (g + r - s)/(e_i+e_j-e_a-e_b)
        new = (gab + r - s) / den
        if np.max(np.abs(new - tv)) < 1e-15:
            tv = new
            break
        tv = new
    t[no:, no:, :no, :no] = tv
    return t


def part_b():
    print("== B. OO-MP2 orbital gradient")
    terms, j_expr, ehf_expr = sympy_functionals()
    ns, no_s, nv_s = 4, 2, 2  # spatial
    no, nv = 2 * no_s, 2 * nv_s
    n = no + nv
    rs = np.random.RandomState(7)
    naux = 7
    bq = rs.randn(naux, ns, ns)
    bq = 0.5 * (bq + bq.transpose(0, 2, 1)) * 0.6
    eri = np.einsum("Ppq,Prs->pqrs", bq, bq)  # chemist (pq|rs), 8-fold symmetric
    # semicanonical Fock: diagonal oo and vv blocks, arbitrary ov block
    fsp = np.zeros((ns, ns))
    eo_sp = -1.5 + 0.4 * rs.rand(no_s)
    ev_sp = 0.7 + 0.5 * rs.rand(nv_s)
    fsp[np.arange(no_s), np.arange(no_s)] = eo_sp
    fsp[no_s + np.arange(nv_s), no_s + np.arange(nv_s)] = ev_sp
    fov = 0.3 * rs.randn(nv_s, no_s)
    fsp[no_s:, :no_s] = fov
    fsp[:no_s, no_s:] = fov.T
    jk = 2.0 * np.einsum("pqkk->pq", eri[:, :, :no_s, :no_s]) - np.einsum(
        "pkqk->pq", eri[:, :no_s, :, :no_s]
    )
    hsp = fsp - jk

    # spin-orbital model: occupied (i, s) -> 2i+s, virtual (a, s) -> 2 no_s + 2 (a-no_s) + s
    def so(p, s):
        return 2 * p + s if p < no_s else 2 * no_s + 2 * (p - no_s) + s

    h_so = np.zeros((n, n))
    g_so = np.zeros((n, n, n, n))
    for p, q_, r, s in itertools.product(range(ns), repeat=4):
        for sp_, sq_, sr_, ss_ in itertools.product(range(2), repeat=4):
            val = 0.0
            if sp_ == sr_ and sq_ == ss_:
                val += eri[p, r, q_, s]
            if sp_ == ss_ and sq_ == sr_:
                val -= eri[p, s, q_, r]
            g_so[so(p, sp_), so(q_, sq_), so(r, sr_), so(s, ss_)] = val
    for p, q_ in itertools.product(range(ns), repeat=2):
        for s in range(2):
            h_so[so(p, s), so(q_, s)] = hsp[p, q_]
    f_so = fock_from(h_so, g_so, no)
    fsp_so = np.zeros((n, n))
    for p, q_ in itertools.product(range(ns), repeat=2):
        for s in range(2):
            fsp_so[so(p, s), so(q_, s)] = fsp[p, q_]
    check(
        "B0 model: spin-orbital Fock == F (x) 1, f_oo and f_vv diagonal, f_ov != 0 (claim B0)",
        np.max(np.abs(f_so - fsp_so)) < 1e-13 and np.max(np.abs(f_so[:no, no:])) > 1e-2,
    )

    t0 = amplitudes(g_so, np.diag(f_so), no, nv)

    def energy(h, g, t):
        tens = {"h": h, "g": g, "f": fock_from(h, g, no), "t": t}
        return evaluate(ehf_expr, tens, no, nv), evaluate(j_expr, tens, no, nv)

    # ---- B1: stationarity and E2
    e_hf0, j0 = energy(h_so, g_so, t0)
    e2 = 0.25 * np.einsum(
        "abij,abij->", t0[no:, no:, :no, :no], g_so[no:, no:, :no, :no]
    )
    check(
        "B1 J(t*) == E2 = 1/4 <ij||ab> t*", abs(j0 - e2) < 1e-12, f"{abs(j0 - e2):.1e}"
    )
    worst = 0.0
    for sd in range(4):
        dd = np.zeros_like(t0)
        r_ = np.random.RandomState(100 + sd).randn(nv, nv, no, no)
        r_ = r_ - r_.transpose(1, 0, 2, 3)
        r_ = r_ - r_.transpose(0, 1, 3, 2)
        dd[no:, no:, :no, :no] = r_
        tp, tm = t0 + 1e-2 * dd, t0 - 1e-2 * dd
        slope = (energy(h_so, g_so, tp)[1] - energy(h_so, g_so, tm)[1]) / 2e-2
        worst = max(worst, abs(slope))
    check(
        "B1 dJ/dt = 0 at t* along random antisymmetric directions (J quadratic: central difference exact)",
        worst < 1e-10,
        f"{worst:.1e}",
    )
    e_hf_ref = np.trace(h_so[:no, :no]) + 0.5 * np.einsum(
        "ijij->", g_so[:no, :no, :no, :no]
    )
    check(
        "B1 E_HF (wicks) == sum h_ii + 1/2 sum <ij||ij> (claim B1)",
        abs(e_hf0 - e_hf_ref) < 1e-12,
    )

    # ---- B2: gradient from the sympy densities
    tens0 = {"h": h_so, "g": g_so, "f": f_so, "t": t0}
    w = derivative_array(j_expr, "f", tens0, no, nv)  # unrelaxed MP2 density dJ/df
    gam_t = derivative_array(j_expr, "g", tens0, no, nv)
    dehf_dh = derivative_array(ehf_expr, "h", tens0, no, nv)
    dehf_dg = derivative_array(ehf_expr, "g", tens0, no, nv)
    gamma = dehf_dh + w
    big_gamma = dehf_dg.copy()
    for p, q_ in itertools.product(range(n), repeat=2):
        if w[p, q_] != 0.0:
            for k in range(no):
                big_gamma[p, k, q_, k] += w[p, q_]
    if not DROP_GAMMA_T:
        big_gamma = big_gamma + gam_t

    def gen_fock(gam, bg):
        x = np.einsum("tq,pq->tp", h_so, gam) + np.einsum("qt,qp->tp", h_so, gam)
        x += (
            np.einsum("tqrs,pqrs->tp", g_so, bg)
            + np.einsum("qtrs,qprs->tp", g_so, bg)
            + np.einsum("qrts,qrps->tp", g_so, bg)
            + np.einsum("qrst,qrsp->tp", g_so, bg)
        )
        return x

    x = gen_fock(gamma, big_gamma)
    g_formula = x[no:, :no] - x[:no, no:].T  # g_ai = X_ai - X_ia

    def e_fixed_t(kap_scale, x_ai, rot):
        kap = kappa_matrix(x_ai, no, nv)
        u = rot(kap_scale * kap)
        h2, g2 = rotate(h_so, g_so, u)
        eh, jj = energy(h2, g2, t0)
        return eh + jj

    lin = lambda k: np.eye(n) + k  # noqa: E731
    cayley = lambda k: np.linalg.solve(np.eye(n) - 0.5 * k, np.eye(n) + 0.5 * k)  # noqa: E731

    g_fd = np.zeros((nv, no))
    g_fd_cay = np.zeros((nv, no))
    g_chain = np.zeros((nv, no))
    for a, i in itertools.product(range(nv), range(no)):
        unit = np.zeros((nv, no))
        unit[a, i] = 1.0
        s = [e_fixed_t(m, unit, lin) for m in (-2, -1, 1, 2)]
        g_fd[a, i] = (s[0] - 8 * s[1] + 8 * s[2] - s[3]) / 12.0
        hh_ = 1e-3
        sc = [e_fixed_t(m * hh_, unit, cayley) for m in (-2, -1, 1, 2)]
        g_fd_cay[a, i] = (sc[0] - 8 * sc[1] + 8 * sc[2] - sc[3]) / (12.0 * hh_)
        kap = kappa_matrix(unit, no, nv)
        dh, dg = d_rotate(h_so, g_so, kap)
        g_chain[a, i] = np.sum(gamma * dh) + np.sum(big_gamma * dg)
    d1 = np.max(np.abs(g_formula - g_fd))
    d2 = np.max(np.abs(g_chain - g_fd))
    d3 = np.max(np.abs(g_formula - g_fd_cay))
    scale = np.max(np.abs(g_fd))
    check(
        "B2 generalized Fock g_ai = X_ai - X_ia == 5-point FD of E(kappa) at fixed t",
        d1 < 1e-9,
        f"max|d| {d1:.1e} (|g| up to {scale:.2f})",
    )
    check("B2 chain rule sum(gamma dh) + sum(Gamma dg) == FD", d2 < 1e-9, f"{d2:.1e}")
    check(
        "B2 FD through the real Cayley transform (orbital_rotation.rs) agrees to O(h^4)",
        d3 < 1e-7,
        f"{d3:.1e}",
    )

    # fully relaxed energy: t re-solved in the rotated frame (Hylleraas theorem)
    def e_relaxed(m, x_ai):
        kap = kappa_matrix(x_ai, no, nv)
        u = lin(m * kap)
        h2, g2 = rotate(h_so, g_so, u)
        t2 = solve_t(h2, g2, no, nv)
        eh, jj = energy(h2, g2, t2)
        return eh + jj

    g_rel = np.zeros((nv, no))
    for a, i in itertools.product(range(nv), range(no)):
        unit = np.zeros((nv, no))
        unit[a, i] = 1.0
        hh_ = 2e-3
        s = [e_relaxed(m * hh_, unit) for m in (-2, -1, 1, 2)]
        g_rel[a, i] = (s[0] - 8 * s[1] + 8 * s[2] - s[3]) / (12.0 * hh_)
    d4 = np.max(np.abs(g_formula - g_rel))
    check(
        "B2 gradient == FD of the fully relaxed E_HF + E2 (t re-solved): t-response is zero",
        d4 < 1e-7,
        f"{d4:.1e}",
    )

    # ---- B3: the closed-shell loop of oo_rimp2.rs
    nocc, nvir = no_s, nv_s
    ex = lambda p, q_, r, s: eri[p, q_, r, s]  # noqa: E731
    o = lambda i: i  # noqa: E731
    v = lambda a: no_s + a  # noqa: E731
    eo, ev = np.diag(fsp)[:no_s], np.diag(fsp)[no_s:]
    t2 = np.zeros((nocc, nvir, nocc, nvir))
    for i, a, j, b in itertools.product(
        range(nocc), range(nvir), range(nocc), range(nvir)
    ):
        t2[i, a, j, b] = ex(o(i), v(a), o(j), v(b)) / (eo[i] + eo[j] - ev[a] - ev[b])
    # amplitudes of B1 are the antisymmetrized spin-orbital version of t2
    tsp = np.zeros((nv, nv, no, no))
    for a, b, i, j in itertools.product(range(nv), range(nv), range(no), range(no)):
        pa, sa = divmod(a, 2)
        pb, sb = divmod(b, 2)
        pi_, si = divmod(i, 2)
        pj, sj = divmod(j, 2)
        val = 0.0
        if sa == si and sb == sj:
            val += t2[pi_, pa, pj, pb]
        if sa == sj and sb == si:
            val -= t2[pj, pa, pi_, pb]
        tsp[a, b, i, j] = val
    check(
        "B3 model: spatial t2 antisymmetrizes to the spin-orbital t*",
        np.max(np.abs(tsp - t0[no:, no:, :no, :no])) < 1e-13,
    )

    g_code = np.zeros((nvir, nocc))
    fmo = fsp
    for c, k in itertools.product(range(nvir), range(nocc)):
        acc = 4.0 * fmo[v(c), o(k)]
        resp = 0.0
        for j, a, b in itertools.product(range(nocc), range(nvir), range(nvir)):
            cajb, cbja = ex(v(c), v(a), o(j), v(b)), ex(v(c), v(b), o(j), v(a))
            resp += t2[k, a, j, b] * (2.0 * cajb - cbja)
            resp += t2[j, a, k, b] * (2.0 * cbja - cajb)
        for i, j, xx in itertools.product(range(nocc), range(nocc), range(nvir)):
            ikjx, jkix = ex(o(i), o(k), o(j), v(xx)), ex(o(j), o(k), o(i), v(xx))
            resp -= t2[i, c, j, xx] * (2.0 * ikjx - jkix)
            resp -= t2[i, xx, j, c] * (2.0 * jkix - ikjx)
        g_code[c, k] = acc + 2.0 * resp
    # Fock response: W = P + P^T, P_oo = -sum t_ik^ab (2 t_jk^ab - t_jk^ba), P_vv likewise
    p_oo = np.zeros((nocc, nocc))
    p_vv = np.zeros((nvir, nvir))
    for i, j, k, a, b in itertools.product(
        range(nocc), range(nocc), range(nocc), range(nvir), range(nvir)
    ):
        p_oo[i, j] -= t2[i, a, k, b] * (2.0 * t2[j, a, k, b] - t2[j, b, k, a])
    for a, b, i, j, c in itertools.product(
        range(nvir), range(nvir), range(nocc), range(nocc), range(nvir)
    ):
        p_vv[a, b] += t2[i, a, j, c] * (2.0 * t2[i, b, j, c] - t2[i, c, j, b])
    wsp = np.zeros((ns, ns))
    wsp[:no_s, :no_s] = p_oo + p_oo.T
    wsp[no_s:, no_s:] = p_vv + p_vv.T
    jmat = np.einsum("pqrs,rs->pq", eri, wsp)
    kmat = np.einsum("prqs,rs->pq", eri, wsp)
    gmat = jmat - 0.5 * kmat
    fw, wf = fmo @ wsp, wsp @ fmo
    for c, k in itertools.product(range(nvir), range(nocc)):
        g_code[c, k] += 4.0 * gmat[v(c), o(k)] + 2.0 * (fw[v(c), o(k)] - wf[v(c), o(k)])

    # spin-orbital W of B2 against the closed-shell W: dE/dF_pq convention
    # (dJ/df sums over both spin blocks, so W_so traced over spin == W_sp)
    wso_sp = np.zeros((ns, ns))
    for p, q_ in itertools.product(range(ns), repeat=2):
        wso_sp[p, q_] = sum(w[so(p, s), so(q_, s)] for s in range(2))
    check(
        "B3 W: closed-shell P + P^T == spin-traced dJ/df of the sympy functional",
        np.max(np.abs(wso_sp - wsp)) < 1e-12,
        f"{np.max(np.abs(wso_sp - wsp)):.1e}",
    )

    # spatial kappa on both spins, fixed-t FD of the spin-orbital energy
    g_sp_fd = np.zeros((nvir, nocc))
    for c, k in itertools.product(range(nvir), range(nocc)):

        def e_at(m):
            kap = np.zeros((n, n))
            for s in range(2):
                kap[so(v(c), s), so(k, s)] = KAPPA_SIGN * m
                kap[so(k, s), so(v(c), s)] = -KAPPA_SIGN * m
            h2, g2 = rotate(h_so, g_so, np.eye(n) + kap)
            eh, jj = energy(h2, g2, t0)
            return eh + jj

        s_ = [e_at(m) for m in (-2, -1, 1, 2)]
        g_sp_fd[c, k] = (s_[0] - 8 * s_[1] + 8 * s_[2] - s_[3]) / 12.0
    d5 = np.max(np.abs(g_code - g_sp_fd))
    check(
        "B3 oo_rimp2.rs loop (4F + 2(G_V+G_O) + 4G[W] + 2(FW-WF)) == dE/dkappa_ck of the spin-orbital energy",
        d5 < 1e-9,
        f"max|d| {d5:.1e} (|g| up to {np.max(np.abs(g_sp_fd)):.2f})",
    )
    no_rotation_part = g_code - 2.0 * (fw - wf)[no_s:, :no_s]
    check(
        "B3 negative control: dropping 2(FW-WF) (live because F_ov != 0) breaks the gradient",
        np.max(np.abs(no_rotation_part - g_sp_fd)) > 1e-2,
    )
    hf_only = 4.0 * fmo[no_s:, :no_s]
    check(
        "B3 negative control: the Brillouin term alone is not the gradient",
        np.max(np.abs(hf_only - g_sp_fd)) > 1e-2,
    )
    # per-piece residual with the formula of B2 (spin-orbital, spin-summed)
    g_from_b2 = np.zeros((nvir, nocc))
    for c, k in itertools.product(range(nvir), range(nocc)):
        g_from_b2[c, k] = sum(g_formula[so(v(c), s) - no, so(k, s)] for s in range(2))
    d6 = np.max(np.abs(g_from_b2 - g_code))
    check(
        "B3 spin-summed generalized-Fock gradient of B2 == oo_rimp2.rs loop",
        d6 < 1e-9,
        f"{d6:.1e}",
    )


def main():
    if MUTATION:
        print(f"MUTATION active: {MUTATION}")
    part_a()
    part_b()
    print(f"{failures} failed identities" if failures else "all identities hold")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
