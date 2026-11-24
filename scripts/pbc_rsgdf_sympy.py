#!/usr/bin/env python3
"""Checks of the range-separated Gaussian density fitting (RS-GDF) behind
ferric-pbc's periodic J and K. Theory: the RS-GDF section of the book (`book/src/24-periodic/05-rsgdf.md`).

Claims (each prints "(claim X)" and exits nonzero on a mismatch):

  erfc-pair     sympy: the erfc(w r)/r interaction of two unit Gaussian
                charges of exponents a, b at distance d is
                [erf(g d) - erf(g_w d)]/d with 1/g^2 = 1/a + 1/b and
                1/g_w^2 = 1/a + 1/b + 1/w^2, its d -> 0 limit is
                2 (g - g_w)/sqrt(pi), and its integral over all space is
                pi/w^2 (the G = 0 constant c0 Omega).
  split-j2      the Ewald-split metric J2 = SR (real-space erfc lattice sum)
                + LR (G != 0, kernel 4 pi/G^2 e^{-G^2/4w^2}) - c0 q q^T,
                c0 = pi/(w^2 Omega), equals the pure reciprocal G != 0 sum,
                at two w, on s Gaussians in a triclinic cell (independent
                route: no erfc, no real-space sum).
  split-j3      the same for the three-index J3[mu nu, P] with the lattice-
                summed orbital pair (mu_0 nu_L) and the subtraction
                c0 S_mu nu q_P.
  fit-psd       the Dunlap-robust fit B = s^{-1/2} U^T J3^T over the metric
                eigenvalues s > lindep gives B^T B = J3 J2^+ J3^T, and the
                fitting error I - B^T B is positive semidefinite (the fit is
                variational in the kernel the energy uses, charged aux and
                all, because G = 0 is dropped from I, J2 and J3 alike).
  fit-trivial   exactness anchor: when the aux functions span the pair
                densities, B^T B = I (zero fitting error).
  fit-jk        J = B^T (B d) and K = sum_k B_k D B_k^T are the Coulomb and
                exchange contractions of the fitted ERI sum_k B_k B_k^T.
  auto-omega-gcount   the gdf_omega = "auto" cost model's half-sphere count
                N_G(w) = Omega (2 w sqrt(ln 1/p))^3/(12 pi^2): symbolic
                derivation and a direct count on the model lattice (5%).

Rust: crates/ferric-pbc/src/rsgdf/auto_omega.rs (half_sphere_g_count),
crates/ferric-pbc/src/rsgdf.rs subtract_g0 (c0 q q^T, c0 S q),
fit_with_metric (B = (J3 W)^T, W = U s^{-1/2}), RsGdfJ/RsGdfK
(J = B^T (B d), K = sum_k B_k D B_k^T + v_M S D S).

Mutations (each turns the script red):
  - drop the three-index G = 0 subtraction c0 S q_P (the code's
    G0Handling::MetricOnlyMutant) (marker "[FAIL] (claim split-j3)");
  - scale the kept eigenvectors by 1/s instead of 1/sqrt(s)
    (marker "[FAIL] (claim fit-trivial)");
  - the full ball instead of the half sphere in the direct G count
    (marker "[FAIL] (claim auto-omega-gcount)").

Run: python3 scripts/pbc_rsgdf_sympy.py   (a few seconds)
"""

import math
import sys

import numpy as np
import sympy as sp
from scipy.special import erf

FAILURES = []


def check(claim, ok, detail=""):
    tag = "ok" if ok else "FAIL"
    print(f"[{tag}] (claim {claim}){(': ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(claim)


# ---------------------------------------------------------------------------
# erfc-pair: symbolic pieces
# ---------------------------------------------------------------------------


def claim_erfc_pair():
    a, b, w, d, r = sp.symbols("a b w d r", positive=True)
    g = 1 / sp.sqrt(1 / a + 1 / b)
    gw = 1 / sp.sqrt(1 / a + 1 / b + 1 / w**2)
    f = (sp.erf(g * d) - sp.erf(gw * d)) / d
    lim = sp.limit(f, d, 0)
    ok_lim = sp.simplify(lim - 2 * (g - gw) / sp.sqrt(sp.pi)) == 0
    # The erf kernel of two unit Gaussians: Coulomb of Gaussians whose
    # exponents add reciprocally (erf(w r)/r is the potential of a unit
    # Gaussian of exponent w^2); erfc = 1/r - erf. At w -> oo the erfc
    # interaction vanishes, at w -> 0 it is the bare erf(g d)/d.
    ok_w_inf = sp.simplify(sp.limit(f, w, sp.oo)) == 0
    ok_w_0 = sp.simplify(sp.limit(f, w, 0) - sp.erf(g * d) / d) == 0
    # Integral of erfc(w r)/r over all space: pi/w^2.
    vol = sp.integrate(4 * sp.pi * r**2 * sp.erfc(w * r) / r, (r, 0, sp.oo))
    ok_vol = sp.simplify(vol - sp.pi / w**2) == 0
    # Numerical spot check of the pair formula against a direct 1-D radial
    # quadrature of the erf interaction of two Gaussians (independent route).
    from scipy.integrate import quad

    av, bv, wv, dv = 1.3, 0.7, 0.9, 1.7
    gv = 1 / math.sqrt(1 / av + 1 / bv)
    gwv = 1 / math.sqrt(1 / av + 1 / bv + 1 / wv**2)
    closed = (math.erf(gv * dv) - math.erf(gwv * dv)) / dv
    # Potential of a unit Gaussian of exponent c at distance s: erf(sqrt(c) s)/s.
    # The erfc(w r)/r interaction = 1/r - erf(w r)/r; convolve rho_a with
    # rho_b: the pair is a unit Gaussian of exponent ab/(a+b) in the relative
    # coordinate, so integrate its density against erfc(w s)/s, shifted by d.
    c = av * bv / (av + bv)

    def radial(s):
        # angular average of exp(-c |s - d|^2) over the sphere |s| = s
        x = 2 * c * s * dv
        ang = (math.exp(-c * (s - dv) ** 2) - math.exp(-c * (s + dv) ** 2)) / (2 * x)
        return (c / math.pi) ** 1.5 * 4 * math.pi * s * s * ang * math.erfc(wv * s) / s

    num, _ = quad(radial, 1e-12, 40.0, limit=400, epsabs=1e-14, epsrel=1e-13)
    ok_num = abs(num - closed) < 1e-11
    check(
        "erfc-pair",
        ok_lim and ok_w_inf and ok_w_0 and ok_vol and ok_num,
        f"d->0 {ok_lim}, w->oo {ok_w_inf}, w->0 {ok_w_0}, int = pi/w^2 {ok_vol}, "
        f"quadrature {abs(num - closed):.1e}",
    )


# ---------------------------------------------------------------------------
# Model cell: s Gaussians in a triclinic cell
# ---------------------------------------------------------------------------

LAT = np.array([[4.0, 0.0, 0.0], [0.4, 4.2, 0.0], [0.3, -0.2, 4.4]])
VOL = abs(np.linalg.det(LAT))
RECIP = 2 * math.pi * np.linalg.inv(LAT).T  # rows b_i, a_i . b_j = 2 pi delta_ij
# Orbital primitives (centre, exponent); unnormalised s Gaussians.
ORB = [
    (np.array([0.3, 0.2, 0.1]), 1.1),
    (np.array([1.6, 1.1, 0.9]), 0.8),
    (np.array([2.5, 2.8, 3.1]), 0.6),
]
# Aux primitives (centre, exponent); charged, unnormalised s Gaussians.
AUX = [
    (np.array([0.5, 0.5, 0.5]), 1.5),
    (np.array([2.0, 2.0, 2.2]), 0.7),
    (np.array([1.0, 2.5, 1.5]), 2.2),
]
GCUT = 9.5
NBOX = 5  # integer box for lattice sums (radius > 13 Bohr in this cell)


def lattice_vectors(nbox):
    r = np.arange(-nbox, nbox + 1)
    n = np.array(np.meshgrid(r, r, r, indexing="ij")).reshape(3, -1).T
    return n @ LAT


def gvectors(gcut):
    r = np.arange(-8, 9)
    n = np.array(np.meshgrid(r, r, r, indexing="ij")).reshape(3, -1).T
    g = n @ RECIP
    g2 = np.einsum("ij,ij->i", g, g)
    keep = (g2 > 1e-12) & (g2 <= gcut * gcut)
    return g[keep], g2[keep]


TVEC = lattice_vectors(NBOX)
GV, G2 = gvectors(GCUT)


def pair_images(m, n):
    """Images L of the pair (m_0, n_L): (charge S_L, centre, exponent)."""
    (ra, a), (rb, b) = ORB[m], ORB[n]
    p = a + b
    rbl = rb[None, :] + TVEC
    d2 = np.einsum("ij,ij->i", ra[None, :] - rbl, ra[None, :] - rbl)
    s = (math.pi / p) ** 1.5 * np.exp(-a * b / p * d2)
    keep = s > 1e-20
    centre = (a * ra[None, :] + b * rbl[keep]) / p
    return s[keep], centre, p


def aux_charge(P):
    return (math.pi / AUX[P][1]) ** 1.5


def ft_pair(m, n):
    """P_mn(G) = sum_L S_L e^{-G^2/4p} e^{-i G . centre_L}."""
    s, c, p = pair_images(m, n)
    return np.exp(-G2 / (4 * p)) * (np.exp(-1j * GV @ c.T) @ s)


def ft_aux(P):
    r, a = AUX[P]
    return aux_charge(P) * np.exp(-G2 / (4 * a)) * np.exp(-1j * GV @ r)


def recip_sum(x, y, w=None):
    """(1/Omega) sum_{G != 0} v(G) conj(x) y, v = 4 pi/G^2 [e^{-G^2/4w^2}]."""
    v = 4 * math.pi / G2
    if w is not None:
        v = v * np.exp(-G2 / (4 * w * w))
    return float(np.real(np.sum(v * np.conj(x) * y))) / VOL


def erfc_pair(a, b, w, d):
    g = 1 / math.sqrt(1 / a + 1 / b)
    gw = 1 / math.sqrt(1 / a + 1 / b + 1 / (w * w))
    small = d < 1e-10
    ds = np.where(small, 1.0, d)
    val = (erf(g * ds) - erf(gw * ds)) / ds
    return np.where(small, 2 * (g - gw) / math.sqrt(math.pi), val)


def sr_metric(P, Q, w):
    (rp, ap), (rq, aq) = AUX[P], AUX[Q]
    d = np.linalg.norm(rp[None, :] - rq[None, :] - TVEC, axis=1)
    return aux_charge(P) * aux_charge(Q) * float(np.sum(erfc_pair(ap, aq, w, d)))


def sr_three(m, n, P, w):
    s, c, p = pair_images(m, n)
    rp, ap = AUX[P]
    tot = 0.0
    for sl, cl in zip(s, c):
        d = np.linalg.norm(cl[None, :] - rp[None, :] - TVEC, axis=1)
        tot += sl * float(np.sum(erfc_pair(p, ap, w, d)))
    return aux_charge(P) * tot


def overlap(m, n):
    return float(np.sum(pair_images(m, n)[0]))


def claim_split():
    naux = len(AUX)
    xs = [ft_aux(P) for P in range(naux)]
    q = np.array([aux_charge(P) for P in range(naux)])
    pure_j2 = np.array(
        [[recip_sum(xs[P], xs[Q]) for Q in range(naux)] for P in range(naux)]
    )
    pairs = [(m, n) for m in range(len(ORB)) for n in range(len(ORB))]
    fts = {mn: ft_pair(*mn) for mn in pairs}
    pure_j3 = np.array(
        [[recip_sum(fts[mn], xs[P]) for P in range(naux)] for mn in pairs]
    )
    s_flat = np.array([overlap(*mn) for mn in pairs])
    err2, err3, spread3 = 0.0, 0.0, []
    for w in (1.0, 1.6):
        c0 = math.pi / (w * w * VOL)
        sr2 = np.array([[sr_metric(P, Q, w) for Q in range(naux)] for P in range(naux)])
        lr2 = np.array(
            [[recip_sum(xs[P], xs[Q], w) for Q in range(naux)] for P in range(naux)]
        )
        j2_split = sr2 + lr2 - c0 * np.outer(q, q)
        err2 = max(err2, float(np.max(np.abs(j2_split - pure_j2))))
        sr3 = np.array([[sr_three(*mn, P, w) for P in range(naux)] for mn in pairs])
        lr3 = np.array(
            [[recip_sum(fts[mn], xs[P], w) for P in range(naux)] for mn in pairs]
        )
        g0 = c0 * np.outer(s_flat, q)
        j3_split = sr3 + lr3
        j3_split = j3_split - g0
        err3 = max(err3, float(np.max(np.abs(j3_split - pure_j3))))
        spread3.append(j3_split)
    w_dep = float(np.max(np.abs(spread3[0] - spread3[1])))
    scale = float(np.max(np.abs(pure_j3)))
    scale2 = float(np.max(np.abs(pure_j2)))
    check(
        "split-j2",
        err2 < 1e-10 * max(1.0, scale2),
        f"max |J2_split - J2_pure| = {err2:.1e} (scale {scale2:.2f}) at w = 1.0, 1.6",
    )
    check(
        "split-j3",
        err3 < 1e-10 * max(1.0, scale),
        f"max |J3_split - J3_pure| = {err3:.1e} (scale {scale:.2f}), w spread {w_dep:.1e}",
    )
    return pure_j2, pure_j3, fts, xs, pairs


# ---------------------------------------------------------------------------
# The fit: B = s^{-1/2} U^T J3^T
# ---------------------------------------------------------------------------


def robust_b(j2, j3, lindep=1e-10):
    evals, evecs = np.linalg.eigh(j2)
    keep = evals > lindep
    w = evecs[:, keep] / np.sqrt(evals[keep])
    return (j3 @ w).T


def claim_fit(j2, j3, fts, xs, pairs):
    npair = len(pairs)
    ii = np.array([[recip_sum(fts[a], fts[b]) for b in pairs] for a in pairs])
    b = robust_b(j2, j3)
    btb = b.T @ b
    ref = j3 @ np.linalg.pinv(j2) @ j3.T
    d_pinv = float(np.max(np.abs(btb - ref)))
    lam_min = float(np.min(np.linalg.eigvalsh(ii - btb)))
    charged = all(abs(np.real(x[0])) > 0 for x in xs)
    check(
        "fit-psd",
        d_pinv < 1e-10 and lam_min > -1e-10 and charged,
        f"|B^T B - J3 J2^+ J3^T| {d_pinv:.1e}, min eig(I - B^T B) {lam_min:.2e}",
    )
    # Exactness anchor: the aux set IS the pair densities (the distinct
    # ones; (mn) and (nm) are the same function).
    uniq = [mn for mn in pairs if mn[0] <= mn[1]]
    j2t = np.array([[recip_sum(fts[a], fts[c]) for c in uniq] for a in uniq])
    j3t = np.array([[recip_sum(fts[a], fts[c]) for c in uniq] for a in pairs])
    bt = robust_b(j2t, j3t)
    d_triv = float(np.max(np.abs(bt.T @ bt - ii)))
    check(
        "fit-trivial",
        d_triv < 1e-10,
        f"|B^T B - I| = {d_triv:.1e} with aux = pair densities",
    )
    # J and K contractions of the fitted ERI.
    nao = len(ORB)
    rng = np.random.default_rng(7)
    dm = rng.standard_normal((nao, nao))
    dm = dm + dm.T
    eri = (b.T @ b).reshape(nao, nao, nao, nao)  # (mn|ls), pair index m*nao+n
    j_ref = np.einsum("mnls,ls->mn", eri, dm)
    k_ref = np.einsum("mlns,ls->mn", eri, dm)
    j_b = (b.T @ (b @ dm.reshape(-1))).reshape(nao, nao)
    bk = b.reshape(-1, nao, nao)
    k_b = sum(bk[k] @ dm @ bk[k].T for k in range(bk.shape[0]))
    d_jk = max(float(np.max(np.abs(j_b - j_ref))), float(np.max(np.abs(k_b - k_ref))))
    check(
        "fit-jk",
        d_jk < 1e-12 and npair == nao * nao,
        f"max |J, K - ERI contraction| = {d_jk:.1e}",
    )


# ---------------------------------------------------------------------------
# gdf_omega = "auto": the long-range G count
# ---------------------------------------------------------------------------


def claim_auto_omega_gcount():
    """The chooser's closed form for the half-sphere G count,
    N_G(w) = Omega (2 w sqrt(ln 1/p))^3 / (12 pi^2): sympy derives it as half the
    ball of radius gcut = 2 w sqrt(ln 1/p) times the reciprocal-lattice density
    Omega/(2 pi)^3, and a direct count on the triclinic model lattice (every G
    whose first nonzero component is positive, as hcore.rs half_gvectors)
    agrees within 5% at three w, the bar of the Rust test."""
    om, w, lnp = sp.symbols("Omega omega L", positive=True)
    gc = 2 * w * sp.sqrt(lnp)
    half_ball = (
        sp.Rational(1, 2) * sp.Rational(4, 3) * sp.pi * gc**3 * om / (2 * sp.pi) ** 3
    )
    sym_ok = sp.simplify(half_ball - om * gc**3 / (12 * sp.pi**2)) == 0
    r = np.arange(-30, 31)
    n = np.array(np.meshgrid(r, r, r, indexing="ij")).reshape(3, -1).T
    g = n @ RECIP
    g2 = np.einsum("ij,ij->i", g, g)
    first = np.where(g[:, 0] != 0, g[:, 0], np.where(g[:, 1] != 0, g[:, 1], g[:, 2]))
    worst = 0.0
    lnp_v = math.log(1e13)
    for wv in (1.0, 1.4, 2.0):
        gcut = 2 * wv * math.sqrt(lnp_v)
        count = int(np.sum((g2 <= gcut * gcut) & (first > 0)))
        est = VOL * gcut**3 / (12 * math.pi**2)
        worst = max(worst, abs(est / count - 1))
    check(
        "auto-omega-gcount",
        sym_ok and worst < 0.05,
        f"half-ball volume x density = Omega gcut^3/(12 pi^2): {sym_ok}; "
        f"max |N_G formula / count - 1| = {worst:.3f} at w = 1, 1.4, 2",
    )


def main():
    claim_erfc_pair()
    claim_auto_omega_gcount()
    j2, j3, fts, xs, pairs = claim_split()
    claim_fit(j2, j3, fts, xs, pairs)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
