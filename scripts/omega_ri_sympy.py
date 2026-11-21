#!/usr/bin/env python3
"""omega-RI fit and omega-RI-CDD-MP2: the identities, symbolic and numerical.

Derivation: `book/src/06-density-fitting/05-omega-ri.md#omega-ri-fit` and
`book/src/12-oo-laplace/05-omega-ri-cdd-mp2.md#omega-ri-cdd-mp2`. Code:
crates/ferric-mp2/src/omega_ri.rs (fit_dressing, omega_ri_dressing_from_metrics),
crates/ferric-mp2/src/omega_ri_cdd.rs (pivoted_cholesky_factored, cdd_factors,
cdd_transform, screened_exchange, omega_ri_cdd_mp2), crates/ferric-rpa/src/lib.rs
(run_pdep_rpa_fitted). Sources: Glasbrenner, Graf, Ochsenfeld, JCTC 16, 6856
(2020), eqs 15-27; Luenser, Schurkus, Ochsenfeld, JCTC 13, 1647 (2017), eq 29.

As coded: V = L L^T (Coulomb metric), V~ the erfc(w r)/r metric, A = (P|ia)_w;
  D = L^T V~^-1,  B = D A,  (ia|jb) ~ B_ia . B_jb = A^T V~^-1 V V~^-1 A.
CDD: P_occ(t) = C_o diag(e^{t(e_i-e_F)}) C_o^T = G G^T, G = C_o diag(e^{t(e_i-e_F)/2});
the pivoted Cholesky of P_occ is computed on the Gram M = G^T G and gives
X_occ (n_occ x r) with G^T-space factor; B_cdd = (X_occ (x) X_vir)^T B_ov.
  E_C(t) = -2 || B_cdd B_cdd^T ||_F^2,  E_X(t) = + sum_ij tr(G_ij G_ij),
  E_OS = (1/2) sum_t w_t E_C(t),  E_SS = (1/2) sum_t w_t E_C(t) + sum_t w_t E_X(t).

Checks:
  C1  FT[erfc(wr)/r] = 4 pi/k^2 (1 - exp(-k^2/4w^2)) > 0: V~ is positive definite
      for every w > 0 (symbolic + quadrature).
  C2  w -> 0: (V - eps W)^-1 V (V - eps W)^-1 -> V^-1 (symbolic); on the
      Gaussian-charge model the energy reaches the Coulomb fit, first order in
      w because the fit leaks pair charge (numeric).
  C3  w -> oo: int erfc(wr)/r d^3r = pi/w^2, the w^-2 delta weight cancels in
      the metric product, giving the overlap-metric fit; the model's error
      against the overlap-metric fit falls as w^-2.
  C4  robustness order: the Coulomb fit is Galerkin, error second order in the
      fit residual; the erfc fit is first order; the robust 3-term form is
      second order (symbolic, random Gram model slopes, model identity).
  C5  the coded dressing: D^T D = V~^-1 V V~^-1; D(w=0) = L^-1; the fitted
      4-index matrix is PSD (numeric, random SPD pair).
  C6  CDD per-t identity: with untruncated pivoted Cholesky (factored form,
      as coded) E_C(t), E_X(t) equal the canonical
      sum (ia|jb)^2 e^{-t Delta} and sum (ia|jb)(ib|ja) e^{-t Delta} to
      rounding, and sum_t w_t over an exact quadrature of 1/Delta reproduces
      E_OS, E_SS (independent route: explicit 4-index loop).
  C7  charge leak (recorded, Galerkin argument asserted): the Coulomb fit's
      residual delta satisfies (delta|P) = 0 for every P, so its leaked
      monopole q_delta enters the energy only at second order; the erfc fit's
      residual has (delta|P)_w = 0 but (delta|P) != 0, so the first-order term
      (delta_ia|rho~_jb) is nonzero. Model: fitted pair charges of both fits,
      and |(delta|P)| for each.

The numerical model for C2, C3, C4 and C7 is closed form (numpy + mpmath, no
quantum chemistry package): zero-charge occupied-virtual pair densities built
from normalized s Gaussians, fitted in a smaller s-Gaussian auxiliary set, with
the Coulomb, erfc and overlap integrals of Gaussian charge densities. The
water/cc-pVDZ numbers in the book come from scripts/validation/gen_omega_ri.py
(PySCF) and its committed reference JSON.

Mutations (each recorded in crates/ferric-python/tests/test_derivation_scripts.py):
  * KERNEL erfc -> erf                         -> FAIL [C1]
  * GALERKIN_COULOMB = False                   -> FAIL [C4-coulomb-order]
  * dressing uses V~^-1 only (no L^T)          -> FAIL [C5]
  * CDD exchange without the off-diagonal 2    -> FAIL [C6]
  * Fermi shift sign flipped in the virtuals   -> FAIL [C6]
  * model erfc kernel replaced by erf         -> FAIL [C3-model]
Exit status: 0 iff every check passes.
"""

from __future__ import annotations

import sys

import numpy as np
import sympy as sp

failures: list[str] = []


def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} [{label}] {detail}")
    if not ok:
        failures.append(label)


r, k, w, t = sp.symbols("r k omega t", positive=True)
KERNEL = sp.erfc(w * r) / r
GALERKIN_COULOMB = True


# ------------------------------------------------------------------ C1
def claim1():
    ft = sp.simplify(
        sp.integrate(4 * sp.pi * r**2 * KERNEL * sp.sin(k * r) / (k * r), (r, 0, sp.oo))
    )
    target = 4 * sp.pi / k**2 * (1 - sp.exp(-(k**2) / (4 * w**2)))
    ok_sym = sp.simplify(ft - target) == 0
    from scipy.integrate import quad
    from scipy.special import erfc

    def num(kv, wv):
        def f(x):
            return 4 * np.pi * x**2 * (erfc(wv * x) / x) * np.sin(kv * x) / (kv * x)

        return quad(f, 0, 60 / wv, limit=400)[0]

    ok_num = all(
        abs(num(kv, wv) - float(target.subs({k: kv, w: wv}))) < 1e-6
        for kv in (0.2, 1.0, 3.0)
        for wv in (0.1, 0.5)
    )
    ok_pos = all(
        float(target.subs({k: kv, w: wv})) > 0
        for kv in (0.01, 0.3, 3, 30)
        for wv in (0.05, 1)
    )
    check(
        "C1",
        ok_sym and ok_num and ok_pos,
        f"symbolic {ok_sym} quad {ok_num} positive {ok_pos}",
    )


# ------------------------------------------------------------------ C2 / C3 symbolic
def claim2_symbolic():
    a, b, c, d, e, f, eps = sp.symbols("a b c d e f epsilon")
    V = sp.Matrix([[a, b], [b, c]])
    W = sp.Matrix([[d, e], [e, f]])
    Vt = V - eps * W
    ct = Vt.inv() * V * Vt.inv()
    lim = (ct - V.inv()).applyfunc(lambda x: sp.simplify(sp.limit(x, eps, 0)))
    check("C2-symbolic", lim == sp.zeros(2, 2))


def claim3_symbolic():
    m0 = sp.simplify(sp.integrate(4 * sp.pi * r**2 * KERNEL, (r, 0, sp.oo)))
    m2 = sp.simplify(sp.integrate(4 * sp.pi * r**4 * KERNEL, (r, 0, sp.oo)))
    check(
        "C3-symbolic",
        sp.simplify(m0 - sp.pi / w**2) == 0 and sp.simplify(m2 * w**4).is_number,
        f"m0 = {m0}",
    )


# ------------------------------------------------------------------ C4
def claim4():
    x1, y1, z2, w0 = sp.symbols("x1 y1 z2 w0")
    x, y, z = t * x1, t * y1, t**2 * z2
    exact = w0 + x + y + z

    def order(err):
        return min(m[0] for m in sp.Poly(sp.expand(err), t).monoms())

    e_coul = (exact - w0).subs({x1: 0, y1: 0}) if GALERKIN_COULOMB else exact - w0
    robust = (w0 + y) + (w0 + x) - w0
    check("C4-coulomb-order", order(e_coul) == 2, f"order {order(e_coul)}")
    check("C4-omega-order", order(exact - w0) == 1)
    check("C4-robust-order", order(exact - robust) == 2)
    # random Gram model: log-log slopes of |error| against the residual scale
    rng = np.random.default_rng(3)
    n, naux = 14, 5
    m1 = rng.standard_normal((n, n))
    g = m1 @ m1.T + n * np.eye(n)
    m2 = rng.standard_normal((n, n))
    gw = m2 @ m2.T + n * np.eye(n)
    a_ = rng.standard_normal((n, naux))
    xa, xb = rng.standard_normal(naux), rng.standard_normal(naux)
    ra, rb = rng.standard_normal(n), rng.standard_normal(n)

    def errs(tv):
        va, vb = a_ @ xa + tv * ra, a_ @ xb + tv * rb
        ex = va @ g @ vb
        out = {}
        for tag, gm in (("coul", g), ("omega", gw)):
            mf = a_.T @ gm @ a_
            fa = a_ @ np.linalg.solve(mf, a_.T @ gm @ va)
            fb = a_ @ np.linalg.solve(mf, a_.T @ gm @ vb)
            out[tag] = abs(ex - fa @ g @ fb)
            if tag == "omega":
                out["robust"] = abs(ex - (fa @ g @ vb + va @ g @ fb - fa @ g @ fb))
        return out

    e1, e2 = errs(1e-3), errs(1e-4)
    sl = {kk: np.log10(e1[kk] / e2[kk]) for kk in e1}
    check(
        "C4-gram-slopes",
        abs(sl["coul"] - 2) < 0.1
        and abs(sl["omega"] - 1) < 0.1
        and abs(sl["robust"] - 2) < 0.1,
        str({kk: round(v, 3) for kk, v in sl.items()}),
    )


# ------------------------------------------------------------------ C5 (dressing as coded)
def dressing(v, v_w):
    """ferric omega_ri_dressing_from_metrics: L = chol(V), W = V~^-1 L, D = W^T."""
    import scipy.linalg as sla

    l_ = np.linalg.cholesky(v)
    wmat = sla.cho_solve(sla.cho_factor(v_w, lower=True), l_)
    return wmat.T


def claim5():
    rng = np.random.default_rng(7)
    n = 9
    m1 = rng.standard_normal((n, n))
    v = m1 @ m1.T + n * np.eye(n)
    m2 = rng.standard_normal((n, n))
    v_w = 0.6 * v + 0.4 * (m2 @ m2.T + n * np.eye(n))
    d = dressing(v, v_w)
    vwi = np.linalg.inv(v_w)
    gram = d.T @ d
    target = vwi @ v @ vwi
    e1 = np.abs(gram - target).max() / np.abs(target).max()
    d0 = dressing(v, v)
    e2 = np.abs(d0 - np.linalg.inv(np.linalg.cholesky(v))).max()
    a_ = rng.standard_normal((n, 30))
    lam = np.linalg.eigvalsh(a_.T @ gram @ a_)
    check(
        "C5",
        e1 < 1e-12 and e2 < 1e-12 and lam.min() > -1e-10 * lam.max(),
        f"DtD {e1:.1e} D(0) {e2:.1e}",
    )


# ------------------------------------------------------------------ C6 (CDD per-t identity)
def pivoted_cholesky_factored(g, tol):
    """ferric pivoted_cholesky_factored: pivoted Cholesky of P = G G^T without
    forming P; returns the factor L (nrows x r) with L L^T = P."""
    m = g.shape[0]
    d = np.einsum("ij,ij->i", g, g).copy()
    d0 = d.max()
    cols = []
    for _ in range(m):
        p = int(np.argmax(d))
        if d[p] <= max(tol, m * np.finfo(float).eps * d0):
            break
        col = g @ g[p] - sum(c * c[p] for c in cols) if cols else g @ g[p]
        col = col / np.sqrt(d[p])
        cols.append(col)
        d = d - col**2
    return np.array(cols).T


FERMI_SIGN_VIR = -1.0
EXCHANGE_OFFDIAG = 2.0


def cdd_energies(b, eo, ev, tval, c_o, c_v):
    """omega_ri_cdd.rs at one Laplace point: B (naux, no*nv) in MO; returns E_C, E_X."""
    no, nv = eo.size, ev.size
    ef = 0.5 * (eo.max() + ev.min())
    go = c_o * np.exp(0.5 * tval * (eo - ef))
    gv = c_v * np.exp(FERMI_SIGN_VIR * 0.5 * tval * (ev - ef))
    lo = pivoted_cholesky_factored(go, 0.0)
    lv = pivoted_cholesky_factored(gv, 0.0)
    # X in the MO basis: the AO-space factor L = C X with C orthonormal columns
    xo = np.linalg.lstsq(c_o, lo, rcond=None)[0]
    xv = np.linalg.lstsq(c_v, lv, rcond=None)[0]
    b3 = b.reshape(-1, no, nv)
    bc = np.einsum("Pia,ik,al->Pkl", b3, xo, xv)
    ro = bc.shape[1]
    bm = bc.reshape(bc.shape[0], -1)
    e_c = -2.0 * np.sum((bm.T @ bm) ** 2)
    e_x = 0.0
    for i in range(ro):
        for j in range(i, ro):
            gij = np.einsum("Pa,Pb->ab", bc[:, i, :], bc[:, j, :])
            kij = np.sum(gij * gij.T)
            e_x += kij if i == j else EXCHANGE_OFFDIAG * kij
    return e_c, e_x


def claim6():
    rng = np.random.default_rng(11)
    nao, no, nv, naux = 12, 3, 6, 10
    q, _ = np.linalg.qr(rng.standard_normal((nao, no + nv)))
    c_o, c_v = q[:, :no], q[:, no:]
    eo = np.sort(rng.uniform(-2.0, -0.4, no))
    ev = np.sort(rng.uniform(0.3, 3.0, nv))
    b = rng.standard_normal((naux, no * nv)) * 0.3
    g4 = (b.T @ b).reshape(no, nv, no, nv)
    dlt = (
        -eo[:, None, None, None]
        + ev[None, :, None, None]
        - eo[None, None, :, None]
        + ev[None, None, None, :]
    )
    worst = 0.0
    for tval in (0.05, 0.7, 3.0):
        e_c, e_x = cdd_energies(b, eo, ev, tval, c_o, c_v)
        ref_c = -2.0 * np.sum(g4**2 * np.exp(-tval * dlt))
        ref_x = np.sum(g4 * g4.transpose(0, 3, 2, 1) * np.exp(-tval * dlt))
        worst = max(worst, abs(e_c - ref_c) / abs(ref_c), abs(e_x - ref_x) / abs(ref_x))
    # quadrature of 1/Delta on [0, oo) by Gauss-Laguerre after scaling (exactness route)
    xs, ws = np.polynomial.laguerre.laggauss(60)
    s = dlt.min()
    e_os = e_ss = 0.0
    for x_, w_ in zip(xs, ws):
        tval = x_ / s
        wt = w_ * np.exp(x_) / s
        e_c, e_x = cdd_energies(b, eo, ev, tval, c_o, c_v)
        e_os += 0.5 * wt * e_c
        e_ss += 0.5 * wt * e_c + wt * e_x
    ref_os = -np.sum(g4**2 / dlt)
    ref_ss = -np.sum(g4 * (g4 - g4.transpose(0, 3, 2, 1)) / dlt)
    d_q = max(abs(e_os - ref_os) / abs(ref_os), abs(e_ss - ref_ss) / abs(ref_ss))
    check(
        "C6", worst < 1e-12 and d_q < 1e-6, f"per-t {worst:.1e}, quadrature {d_q:.1e}"
    )


# ------------------------------------------------------------------ Gaussian-charge model (C2, C3, C4, C7)
# Every integral below is closed form: for two normalized s-Gaussian charge
# densities with exponents a, b a distance R apart, c = ab/(a+b),
#   Coulomb  (a|b)   = erf(sqrt(c) R)/R,
#   erf(wr)/r        = erf(sqrt(c_w) R)/R,   1/c_w = 1/c + 1/w^2,
#   erfc(wr)/r       = Coulomb - erf,
#   overlap  (a|b)_S = (c/pi)^(3/2) exp(-c R^2).
# mpmath at 40 digits removes the Coulomb - erf cancellation at large w. The
# occupied-virtual pair densities are zero-charge combinations of a pair
# basis; the auxiliary set is a different, smaller s set, so the fit is
# inexact and leaks charge, as a real auxiliary basis does.
GAUSS_CENTRES = [(0.0, 0.0, 0.0), (1.8, 0.3, 0.0), (3.5, -0.2, 0.4)]
PAIR_EXPONENTS = (0.25, 0.9, 3.2)
AUX_EXPONENTS = (0.4, 1.6, 6.0)


def gauss_kernel(kind, a, b, rr, wv=None):
    import mpmath as mp

    mp.mp.dps = 40
    a, b, rr = mp.mpf(a), mp.mpf(b), mp.mpf(rr)
    c = a * b / (a + b)
    if kind == "ovlp":
        return float((c / mp.pi) ** 1.5 * mp.exp(-c * rr**2))

    def coul(cc):
        return 2 * mp.sqrt(cc / mp.pi) if rr == 0 else mp.erf(mp.sqrt(cc) * rr) / rr

    if kind == "coul":
        return float(coul(c))
    return float(coul(c) - coul(1 / (1 / c + 1 / mp.mpf(wv) ** 2)))


def gauss_matrix(kind, fa, fb, wv=None):
    return np.array(
        [
            [
                gauss_kernel(kind, ea, eb, np.linalg.norm(np.subtract(ra, rb)), wv)
                for eb, rb in fb
            ]
            for ea, ra in fa
        ]
    )


def model_system():
    pairb = [(e, r) for r in GAUSS_CENTRES for e in PAIR_EXPONENTS]
    auxb = [(e, r) for r in GAUSS_CENTRES for e in AUX_EXPONENTS]
    rng = np.random.default_rng(5)
    no, nv = 2, 4
    cpair = rng.standard_normal((len(pairb), no * nv))
    cpair = 0.2 * (cpair - cpair.mean(axis=0))  # each ov pair density carries no charge
    eo = np.array([-1.1, -0.6])
    ev = np.array([0.3, 0.7, 1.2, 2.0])

    def mp2(g):
        t4 = g.reshape(no, nv, no, nv)
        dl = (
            eo[:, None, None, None]
            - ev[None, :, None, None]
            + eo[None, None, :, None]
            - ev[None, None, None, :]
        )
        return float(
            np.einsum("iajb,iajb->", 2 * t4 - t4.transpose(0, 3, 2, 1), t4 / dl)
        )

    v = gauss_matrix("coul", auxb, auxb)
    a0 = gauss_matrix("coul", auxb, pairb) @ cpair
    c0 = np.linalg.solve(v, a0)
    g0 = a0.T @ c0
    e0 = mp2(g0)

    def fit(wv):
        vw = gauss_matrix("erfc", auxb, auxb, wv)
        aw = gauss_matrix("erfc", auxb, pairb, wv) @ cpair
        cw = np.linalg.solve(vw, aw)
        return cw, cw.T @ v @ cw

    # C2: w -> 0 reaches the Coulomb fit. The leaked pair charge makes the
    # approach first order in w (erfc = Coulomb - 2w/sqrt(pi) + O(w^3)).
    d2 = {wv: abs(mp2(fit(wv)[1]) - e0) for wv in (1e-3, 1e-4, 1e-5)}
    r1, r2 = d2[1e-3] / d2[1e-4], d2[1e-4] / d2[1e-5]
    check(
        "C2-model",
        9.0 < r1 < 11.0 and 9.0 < r2 < 11.0 and d2[1e-5] < 1e-7 * abs(e0),
        f"|dE| {d2[1e-5]:.1e} at w=1e-5, ratios per decade {r1:.2f} {r2:.2f}",
    )
    # C3: w -> oo reaches the overlap-metric fit c = S^-1 (P|ia)_S, error ~ w^-2
    cs = np.linalg.solve(
        gauss_matrix("ovlp", auxb, auxb), gauss_matrix("ovlp", auxb, pairb) @ cpair
    )
    e_ovl = mp2(cs.T @ v @ cs)
    dd = {wv: abs(mp2(fit(wv)[1]) - e_ovl) for wv in (32.0, 64.0, 128.0)}
    r1, r2 = dd[32.0] / dd[64.0], dd[64.0] / dd[128.0]
    check("C3-model", 3.5 < r1 < 4.5 and 3.5 < r2 < 4.5, f"ratios {r1:.2f} {r2:.2f}")
    # C4 identity: plain - CoulRI = linear + quadratic in D = c0 - cw
    cw, gw = fit(0.1)
    dmat = c0 - cw
    lin = -(dmat.T @ v @ c0) - (c0.T @ v @ dmat)
    quad = dmat.T @ v @ dmat
    ident = np.abs(gw - g0 - lin - quad).max()
    check(
        "C4-model",
        ident < 1e-12 * np.abs(g0).max() and np.abs(lin).max() > 5 * np.abs(quad).max(),
        f"identity {ident:.1e}",
    )
    # C7 charge leak and the Galerkin condition. Each normalized aux Gaussian
    # carries unit charge, so the fitted pair charge is the column sum of c.
    q0, qw = c0.sum(axis=0), cw.sum(axis=0)
    gal0 = np.abs(a0 - v @ c0).max() / np.abs(a0).max()
    galw = np.abs(a0 - v @ cw).max() / np.abs(a0).max()
    print(
        f"     fitted ov pair charges: Coulomb max {np.abs(q0).max():.2e}, erfc(0.1) max {np.abs(qw).max():.2e}"
    )
    print(f"     |(delta|P)|/|(ia|P)|: Coulomb {gal0:.1e}, erfc(0.1) {galw:.1e}")
    check(
        "C7",
        gal0 < 1e-10 and galw > 1e-4,
        f"Galerkin Coulomb {gal0:.1e}, erfc {galw:.1e}",
    )


def main():
    claim1()
    claim2_symbolic()
    claim3_symbolic()
    claim4()
    claim5()
    claim6()
    model_system()
    if failures:
        print(f"{len(failures)} check(s) failed: {failures}")
        sys.exit(1)
    print("all omega-RI identities verified")


if __name__ == "__main__":
    main()
