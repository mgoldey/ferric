#!/usr/bin/env python3
"""Sympy proofs: the Hylleraas mask certificate of the amplitude-threshold LMP2.

Conventions (book/src/13-local-mp2/06-mask-certificate.md,
crates/ferric-mp2/src/lmp2_mask_certificate.rs). Closed-shell MP2 in any
orthonormal occupied / virtual basis, amplitude t[i,a,j,b], g = (ia|jb),
Fock superoperator  A t = Fvv t + t Fvv - Foo t - t Foo  (SPD for a positive
gap), swap S: t[i,a,j,b] -> t[i,b,j,a], metric M = 2 - S, Hylleraas
functional E[t] = 2<Mt, g> + <Mt, At>. Kept set P, dropped set Q; t_P
minimizes E over span(P); r = g + A t_P is the full residual.

Claims (each: a symbolic proof and an independent numeric route):
 C1  E_P - E* = r_Q^T S_c^-1 r_Q, S_c = A_QQ - A_QP A_PP^-1 A_PQ.
 C2  (A^-1)_QQ = S_c^-1, so E_P - E* = <r, A^-1 r> with the FULL r.
 C3  [M, A] = 0, the stationary value is <M t*, g>, and on a swap-closed mask
     E_P - E* = <M r, A^-1 r>; this holds for ANY t (an inexact solve too):
     E[t] - E* = <M r(t), A^-1 r(t)>.
 C4  lambda_min(A) = 2(e_LUMO - e_HOMO) in every rotated basis; the local
     Fock diagonal gap 2(min Fvv_aa - max Foo_ii) is not a valid node.
 C5  Gauss-Radau with a node a <= lambda_min is an upper bound for 1/lambda;
     the one-step Lanczos form the Rust code evaluates (alpha, beta^2, omega)
     equals the moment form (mu0, mu1, mu2) in symbols; numeric enclosure.
 C6  the line search from t_P along any d lowers E by <Md, r>^2/<Md, Ad>, a
     rigorous lower bound; x is reported as an estimate, with no bound
     claimed (2x is not a lower bound: the "C6" mutation).
 C7  ||B^T B||_F^2 = ||B B^T||_F^2 (the Gram identity) and the split bound
     <M r, A^-1 r> <= (sqrt e1 + sqrt e2)^2 for r = r1 + r2.
 C8  the cheap route of the Rust code: near set N = {q in Q: |h_q| > tau or
     |h_Sq| > tau} (h = A t_P), v = r on P u N, U = (sqrt Radau1(v) +
     sqrt((sqrt3 |g_far| + |h_far|_M)^2 / lambda_min))^2 with |g_far|^2 from
     the Gram identity, L = x_N^2 / y_N with d = r_N / D: L <= E[t_P] - E*
     <= U on random masked models with INEXACT t_P, and Q empty gives an
     estimate, lower bound and far term of exactly 0.

Independent route: every numeric check solves the full problem by dense
linear algebra (numpy) and compares E[t_P] - E* to the certified quantities.

Every check is an assert, so a failure exits nonzero with an AssertionError.
Mutations (each turns the script red; recorded in
crates/ferric-python/tests/test_derivation_scripts.py):
  - r_Q = g_Q, dropping A_QP t_P (marker "C1");
  - A_QQ in place of the Schur complement (marker "C2");
  - the metric M -> 1 (marker "C3");
  - the local Fock diagonal gap as lambda_min (marker "C4");
  - the Radau node at 1.5 lambda_min (marker "C5");
  - the Lanczos-form Radau with beta^2 dropped (marker "C5: lanczos");
  - 2x in place of x^2/y as the lower bound (marker "C6");
  - the split bound without its cross term, e1 + e2 (marker "C7");
  - the far term without the Gram norm of g (marker "C8: upper");
  - A t_P dropped from the residual on Q, r_Q = g_Q (marker "C8: upper").

Run: python scripts/lmp2_mask_certificate_sympy.py
"""

import numpy as np
import sympy as sp

rng = np.random.default_rng(20261119)


def section(title):
    print(f"\n=== {title}")


# ---------------------------------------------------------------- helpers
def kron_sum(foo, fvv):
    """Dense A in index order (i, a, j, b)."""
    no, nv = foo.shape[0], fvv.shape[0]
    io, iv = np.eye(no), np.eye(nv)

    def k(a, b, c, d):
        return np.kron(np.kron(np.kron(a, b), c), d)

    return (
        k(io, fvv, io, iv)
        + k(io, iv, io, fvv)
        - k(foo, iv, io, iv)
        - k(io, iv, foo, iv)
    )


def swap_mat(no, nv):
    n = no * nv * no * nv
    idx = np.arange(n).reshape(no, nv, no, nv)
    sw = idx.transpose(0, 3, 2, 1).ravel()
    p = np.zeros((n, n))
    p[np.arange(n), sw] = 1.0
    return p


def rand_model(no=2, nv=4, coup=0.15, naux=6):
    """Rotated local Fock blocks and RI integrals g = B^T B (B: naux x no*nv)."""
    eo = np.sort(rng.uniform(-1.3, -0.45, no))
    ev = np.sort(rng.uniform(0.25, 3.0, nv))
    uo = np.linalg.qr(rng.normal(size=(no, no)))[0]
    uv = np.linalg.qr(np.eye(nv) + coup * rng.normal(size=(nv, nv)))[0]
    foo, fvv = uo @ np.diag(eo) @ uo.T, uv @ np.diag(ev) @ uv.T
    b = rng.normal(size=(naux, no * nv)) * np.exp(-rng.uniform(0, 4, size=(1, no * nv)))
    g = (b.T @ b).ravel()  # (ia|jb), flat (i, a, j, b)
    return foo, fvv, eo, ev, g, b


def masked_solve(a, g, p):
    t = np.zeros_like(g)
    t[p] = -np.linalg.solve(a[np.ix_(p, p)], g[p])
    return t


def swap_closed(p, sw):
    return p | (sw @ p.astype(float) > 0.5)


# ---------------------------------------------------------------- C1, C2 symbolic
section("C1/C2: masked error is the Schur norm of the dropped residual")
a = sp.symbols("a11 a12 a13 a22 a23 a33", real=True)
g1, g2, g3 = sp.symbols("g1 g2 g3", real=True)
A3 = sp.Matrix([[a[0], a[1], a[2]], [a[1], a[3], a[4]], [a[2], a[4], a[5]]])
gv = sp.Matrix([g1, g2, g3])
App, Apq, Aqq = A3[:2, :2], A3[:2, 2:], A3[2:, 2:]
tP = -App.inv() * gv[:2, :]
EP = (2 * tP.T * gv[:2, :] + tP.T * App * tP)[0]
Eex = -(gv.T * A3.inv() * gv)[0]
rQ = gv[2:, :] + Apq.T * tP
Sc = Aqq - Apq.T * App.inv() * Apq
assert sp.simplify(EP - Eex - (rQ.T * Sc.inv() * rQ)[0]) == 0, (
    "FAIL [C1 Schur identity]"
)
Sc2 = Aqq - Apq.T * App.inv() * Apq
assert sp.simplify(A3.inv()[2:, 2:] - Sc2.inv()) == sp.zeros(1, 1), (
    "FAIL [C2 inverse block]"
)
print("  C1, C2 symbolic: PASS")

# ---------------------------------------------------------------- C3
section("C3: closed-shell metric, stationarity, <M r, A^-1 r> for any t")


def c3_commutator():
    fo = sp.symbols("o11 o12 o22", real=True)
    fv = sp.symbols("v11 v12 v22", real=True)
    foo = sp.Matrix([[fo[0], fo[1]], [fo[1], fo[2]]])
    fvv = sp.Matrix([[fv[0], fv[1]], [fv[1], fv[2]]])
    i2 = sp.eye(2)
    kp = sp.kronecker_product

    def k4(a_, b_, c_, d_):
        return kp(kp(kp(a_, b_), c_), d_)

    am = (
        k4(i2, fvv, i2, i2)
        + k4(i2, i2, i2, fvv)
        - k4(foo, i2, i2, i2)
        - k4(i2, i2, foo, i2)
    )
    sw = sp.Matrix(swap_mat(2, 2).astype(int))
    m = 2 * sp.eye(16) - sw
    return sp.expand(m * am - am * m) == sp.zeros(16, 16)


assert c3_commutator(), "FAIL [C3 commutator]"


def c3_any_t_symbolic():
    """In a joint eigenbasis of the commuting M, A: E[t] - E* = <M r, A^-1 r>, r = g + A t."""
    t = sp.Matrix(sp.symbols("t0:3", real=True))
    m = sp.symbols("m0:3", positive=True)
    lam = sp.symbols("l0:3", positive=True)
    gg = sp.Matrix(sp.symbols("h0:3", real=True))
    md, ad = sp.diag(*m), sp.diag(*lam)
    e = (2 * (md * t).T * gg + (md * t).T * ad * t)[0]
    tstar = -ad.inv() * gg
    estar = e.subs(dict(zip(t, tstar)))
    r = gg + ad * t
    return sp.simplify(e - estar - ((md * r).T * ad.inv() * r)[0]) == 0


assert c3_any_t_symbolic(), "FAIL [C3 any-t identity]"


def c3_numeric(no=2, nv=4):
    foo, fvv, eo, ev, g, _ = rand_model(no, nv, coup=0.4)
    am = kron_sum(foo, fvv)
    sw = swap_mat(no, nv)
    mtrue = 2 * np.eye(len(g)) - sw
    m = 2 * np.eye(len(g)) - sw
    t_ex = -np.linalg.solve(am, g)
    e_ex = 2 * t_ex @ mtrue @ g + t_ex @ mtrue @ am @ t_ex
    worst = 0.0
    for eps in (3e-2, 1e-2, 3e-3):
        p = swap_closed(np.abs(g) > eps * np.abs(g).max(), sw)
        tp = masked_solve(am, g, p)
        tp[p] += 1e-3 * rng.normal(size=p.sum())  # an inexact solve
        e_h = 2 * tp @ mtrue @ g + tp @ mtrue @ am @ tp
        r = g + am @ tp
        cert = (m @ r) @ np.linalg.solve(am, r)
        worst = max(worst, abs((e_h - e_ex) - cert) / max(abs(cert), 1e-300))
    return worst


w3 = c3_numeric()
assert w3 < 1e-9, f"FAIL [C3 numeric] worst relative {w3:.2e}"
print(f"  C3: PASS (numeric worst relative {w3:.1e})")

# ---------------------------------------------------------------- C4
section("C4: lambda_min(A) = 2(e_LUMO - e_HOMO), rotation invariant")


def c4_symbolic():
    e = sp.symbols("ea eb", real=True)
    o = sp.symbols("oi", real=True)
    fvv = sp.diag(*e)
    foo = sp.Matrix([[o]])
    i1, i2 = sp.eye(1), sp.eye(2)
    kp = sp.kronecker_product
    am = (
        kp(kp(kp(i1, fvv), i1), i2)
        + kp(kp(kp(i1, i2), i1), fvv)
        - kp(kp(kp(foo, i2), i1), i2)
        - kp(kp(kp(i1, i2), foo), i2)
    )
    got = sorted((sp.simplify(x) for x in am.eigenvals()), key=str)
    want = sorted((2 * e[0] - 2 * o, 2 * e[1] - 2 * o, e[0] + e[1] - 2 * o), key=str)
    return got == want


assert c4_symbolic(), "FAIL [C4 spectrum]"


def node(foo, fvv, eo, ev):
    """The Radau node the code uses (canonical gap from the eigenvalues)."""
    return 2 * (ev.min() - eo.max())


worst4 = 0.0
for _ in range(5):
    foo, fvv, eo, ev, g, _ = rand_model(2, 4, coup=0.5)
    lmin = np.linalg.eigvalsh(kron_sum(foo, fvv))[0]
    lam_node = node(foo, fvv, eo, ev)
    assert lam_node <= lmin + 1e-12, (
        f"FAIL [C4 node above lambda_min] {lam_node} > {lmin}"
    )
    worst4 = max(worst4, abs(lam_node - lmin))
assert worst4 < 1e-10, f"FAIL [C4 gap] {worst4:.2e}"
print("  C4: PASS")

# ---------------------------------------------------------------- C5
section("C5: Gauss-Radau upper bound, Lanczos form == moment form")
lam_, a_, th_ = sp.symbols("lambda a theta", positive=True)
c0, c1, c2 = sp.symbols("c0 c1 c2")
poly = c0 + c1 * lam_ + c2 * lam_**2
f = 1 / lam_
sol = sp.solve(
    [
        sp.Eq(poly.subs(lam_, a_), f.subs(lam_, a_)),
        sp.Eq(poly.subs(lam_, th_), f.subs(lam_, th_)),
        sp.Eq(sp.diff(poly, lam_).subs(lam_, th_), sp.diff(f, lam_).subs(lam_, th_)),
    ],
    [c0, c1, c2],
)
rem = sp.simplify(f - poly.subs(sol))
assert sp.simplify(rem + (lam_ - a_) * (lam_ - th_) ** 2 / (lam_ * a_ * th_**2)) == 0, (
    "FAIL [C5 remainder sign]"
)


def radau_moment(mu0, mu1, mu2, a):
    nu1 = mu1 - a * mu0
    nu2 = mu2 - 2 * a * mu1 + a * a * mu0
    th = a + nu2 / nu1
    w = nu1 * nu1 / nu2
    return (mu0 - w) / a + w / th


def radau_lanczos(mu0, alpha_raw, nu, a):
    """Transcription of radau_one_step (lmp2_mask_certificate.rs), no guards."""
    alpha = alpha_raw / mu0
    beta2 = nu / mu0 - alpha * alpha
    omega = a + beta2 / (alpha - a)
    return mu0 * omega / (alpha * omega - beta2)


m0, m1, m2, aa = sp.symbols("mu0 mu1 mu2 a", positive=True)
assert sp.simplify(radau_lanczos(m0, m1, m2, aa) - radau_moment(m0, m1, m2, aa)) == 0, (
    "FAIL [C5: lanczos form differs from the moment form]"
)


def quad(am, m, r, a):
    ar = am @ r
    mu0, mu1, mu2 = (m @ r) @ r, (m @ r) @ ar, (m @ ar) @ ar
    return radau_lanczos(mu0, mu1, mu2, a), mu0 / a


worst5 = np.inf
for _ in range(6):
    foo, fvv, eo, ev, g, _ = rand_model(2, 5, coup=0.5)
    am = kron_sum(foo, fvv)
    sw = swap_mat(2, 5)
    m = 2 * np.eye(len(g)) - sw
    for eps in (1e-1, 1e-2, 1e-3):
        p = swap_closed(np.abs(g) > eps * np.abs(g).max(), sw)
        if p.all():
            continue
        r = g + am @ masked_solve(am, g, p)
        err = (m @ r) @ np.linalg.solve(am, r)
        ru, cu = quad(am, m, r, 1.0 * node(foo, fvv, eo, ev))
        assert err <= ru * (1 + 1e-10) <= cu * (1 + 1e-9), (
            f"FAIL [C5 enclosure] {err} {ru} {cu}"
        )
        worst5 = min(worst5, ru / err)
print(f"  C5: PASS (tightest Radau/err {worst5:.3f})")

# ---------------------------------------------------------------- C6
section("C6: line-search lower bound")
al, x_, y_ = sp.symbols("alpha x y", positive=True)
fl = -2 * al * x_ + al**2 * y_
almin = sp.solve(sp.diff(fl, al), al)[0]
assert sp.simplify(fl.subs(al, almin) + x_**2 / y_) == 0, "FAIL [C6 line search]"
assert sp.simplify(x_**2 / y_ - (2 * x_ - y_) - (x_ - y_) ** 2 / y_) == 0, (
    "FAIL [C6 2x - y]"
)
for _ in range(6):
    foo, fvv, eo, ev, g, _ = rand_model(2, 5, coup=0.6)
    am = kron_sum(foo, fvv)
    sw = swap_mat(2, 5)
    m = 2 * np.eye(len(g)) - sw
    dg = np.diag(am)
    for eps in (1e-1, 1e-2):
        p = swap_closed(np.abs(g) > eps * np.abs(g).max(), sw)
        if p.all():
            continue
        tp = masked_solve(am, g, p)
        r = g + am @ tp
        err = (m @ r) @ np.linalg.solve(am, r)
        d = np.where(p, 0.0, r) / dg
        x = (m @ d) @ r
        y = (m @ d) @ (am @ d)
        lb = x * x / y
        tt = tp - (x / y) * d  # independent route: the Hylleraas value after the step
        e_p = 2 * tp @ m @ g + tp @ m @ am @ tp
        e_t = 2 * tt @ m @ g + tt @ m @ am @ tt
        assert abs((e_p - e_t) - x * x / y) < 1e-12, "FAIL [C6 decrease]"
        assert lb <= err * (1 + 1e-10), f"FAIL [C6 lower bound] {lb} > {err}"
print("  C6: PASS")

# ---------------------------------------------------------------- C7
section("C7: Gram identity and the split bound")
bs = sp.Matrix(3, 4, sp.symbols("b0:12", real=True))
assert sp.expand(sum(v**2 for v in bs.T * bs) - sum(v**2 for v in bs * bs.T)) == 0, (
    "FAIL [C7 Gram]"
)
violated = False
for _ in range(40):
    foo, fvv, eo, ev, g, _ = rand_model(2, 4, coup=0.6)
    am = kron_sum(foo, fvv)
    sw = swap_mat(2, 4)
    m = 2 * np.eye(len(g)) - sw
    r = rng.normal(size=len(g))
    part = swap_closed(rng.random(len(g)) < 0.5, sw)
    r1, r2 = np.where(part, r, 0), np.where(part, 0, r)

    def e(v):
        return (m @ v) @ np.linalg.solve(am, v)

    e1, e2, et = e(r1), e(r2), e(r)
    ub = (np.sqrt(e1) + np.sqrt(e2)) ** 2
    assert et <= ub * (1 + 1e-12), f"FAIL [C7 split bound] {et} > {ub}"
    violated |= et > e1 + e2
assert violated, "FAIL [C7 cross term never needed]"
print("  C7: PASS")

# ---------------------------------------------------------------- C8 the cheap route
section("C8: the cheap near/far route of the Rust code")


def mnorm2(m, v):
    return float((m @ v) @ v)


def cheap_route(foo, fvv, eo, ev, g, b, p, tp, tau):
    """Transcription of mask_certificate (lmp2_mask_certificate.rs) on a dense model."""
    no, nv = foo.shape[0], fvv.shape[0]
    am = kron_sum(foo, fvv)
    sw = swap_mat(no, nv)
    m = 2 * np.eye(len(g)) - sw
    h = am @ tp
    q = ~p
    near = q & ((np.abs(h) > tau) | (np.abs(sw @ h) > tau))
    far = q & ~near
    r = g + h
    v = np.where(p | near, r, 0.0)
    lam = node(foo, fvv, eo, ev)
    av = am @ v
    mu0 = mnorm2(m, v)
    near_up = radau_lanczos(mu0, (m @ v) @ av, mnorm2(m, av), lam) if mu0 > 0 else 0.0
    g_tot = float(np.sum((b @ b.T) ** 2))
    g_far2 = (
        max(g_tot - np.sum(g[p] ** 2) - np.sum(g[near] ** 2), 0.0) if far.any() else 0.0
    )
    h_far = np.sqrt(max(mnorm2(m, np.where(far, h, 0.0)), 0.0))
    far_up = (np.sqrt(3.0 * g_far2) + h_far) ** 2 / lam
    up = (np.sqrt(near_up) + np.sqrt(far_up)) ** 2
    d = np.where(near, r, 0.0) / np.diag(am)  # D = the local Fock diagonal of A
    x = float((m @ d) @ r)
    y = float((m @ d) @ (am @ d)) if near.any() else 1.0
    lo = x * x / y if x > 0 else 0.0
    return up, lo, x, far_up, int(near.sum())


worst_u, worst_l = np.inf, 0.0
for trial in range(12):
    no, nv = 2, 4
    foo, fvv, eo, ev, g, b = rand_model(no, nv, coup=0.5)
    am = kron_sum(foo, fvv)
    sw = swap_mat(no, nv)
    m = 2 * np.eye(len(g)) - sw
    t_ex = -np.linalg.solve(am, g)
    e_star = 2 * t_ex @ m @ g + t_ex @ m @ am @ t_ex
    gmax = np.abs(g).max()
    for eps in (1e-1, 3e-2, 1e-2):
        p = swap_closed(np.abs(g) > eps * gmax, sw)
        if p.all():
            continue
        tp = masked_solve(am, g, p)
        tp[p] += 1e-6 * rng.normal(size=p.sum())  # the PCG stops short of the exact t_P
        e_h = 2 * tp @ m @ g + tp @ m @ am @ tp
        err = e_h - e_star
        up, lo, x, far_up, n_near = cheap_route(
            foo, fvv, eo, ev, g, b, p, tp, eps * gmax
        )
        assert err <= up * (1 + 1e-10), f"FAIL [C8: upper] {err:.3e} > {up:.3e}"
        assert lo <= err * (1 + 1e-10), f"FAIL [C8: lower] {lo:.3e} > {err:.3e}"
        worst_u = min(worst_u, up / err)
        worst_l = max(worst_l, lo / err)

# Q empty: estimate, lower bound and far term exactly zero
foo, fvv, eo, ev, g, b = rand_model(2, 4)
am = kron_sum(foo, fvv)
p = np.ones(len(g), bool)
up0, lo0, x0, far0, nn0 = cheap_route(
    foo, fvv, eo, ev, g, b, p, masked_solve(am, g, p), 0.0
)
assert (lo0, x0, far0, nn0) == (0.0, 0.0, 0.0, 0), (
    "FAIL [C8 anchor] Q empty is not exactly zero"
)
assert up0 < 1e-20, f"FAIL [C8 anchor] upper {up0:.2e} with an exact solve"
print(f"  C8: PASS (tightest U/err {worst_u:.3f}, largest L/err {worst_l:.3f})")
print("\nall claims PASS")
