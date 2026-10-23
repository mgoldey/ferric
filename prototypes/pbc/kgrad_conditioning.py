"""#512: conditioning of the lindep-projected DF fit derivative (rsgdf/kpoint/kderiv.rs).

Model: an s-only molecular DF fit with EXACT integrals (closed forms in mpmath, dps 40), so integral rounding and
linear-algebra rounding can be separated.  The functional is kderiv.rs's at one q (real):

    E = tr[f(J2) H],  H = 1/2 rho rho^T - 1/4 A,  rho_P = tr[J3_P D],  A_PQ = tr[J3_P D J3_Q D]
    f = U diag(1/s kept, 0 dropped) U^T                         (hard lindep cut, absolute)
    dE = sum dJ3 . Z + tr[Wm dJ2],   Z = c D - 1/2 sum_Q (D J3_Q D) f_QP,  c = f rho
    Wm = U (Lo o U^T H U) U^T   (Loewner: kept-kept -f_i f_j, kept-dropped (f_i - f_j)/(s_i - s_j))

Gradient forms compared (all the SAME function):
  'explicit'  -- the pre-#512 kderiv.rs: Jinv = U f U^T formed, Z = T Jinv, c = Jinv rho, H formed in the aux
                 basis then rotated (U^T H U).
  'eigen'     -- the fix: J3 and T rotated first (J3 U, T U), U^T H U accumulated in the eigenbasis,
                 Z = ((T U) o f) U^T, c = U (f o U^T rho).  Jinv is never formed.
  'eigen_z'   -- only Z / c fixed;  'eigen_wm' -- only Wm fixed (to show both are needed).

HYPOTHESES (written before measuring):
  physics: the Loewner derivative is exact for the hard-cut energy, so with exact arithmetic it equals FD (anchor);
           with a well-conditioned aux, double arithmetic is at roundoff.
  artifact A (linear algebra): forming 1/s_min-scaled matrices spreads eps/s_min into well-conditioned directions
           -> the double-LA gradient misses the exact one by >> the miss of exact LA on the same rounded integrals.
  artifact B (integrals): rounding of J3/J2/dJ amplified by the conditioning -> exact LA on rounded integrals misses
           as much as double LA does.  A and B are distinguished by the 'rounded integrals + exact LA' row.

MEASURED (H4 chain, uncontracted H 6-31G s, aux s(4.7,1.9,0.75,0.3) per H + a near-duplicate s 0.75(1+delta) on H0;
fixed random density; max |g - g_exact|; 2026-10-23):
  anchor: exact Loewner gradient vs exact FD (h 1e-12, dps 40): 2e-25 (no duplicate), 3e-22 (delta 1e-4).
  delta   s_min    explicit  eigen     rounded ints + exact LA (floor)
  none    1.5e-2   4.6e-13   1.1e-14   8.9e-16
  1e-3    9.2e-9   5.2e-7    1.4e-11   1.7e-11
  1e-4    9.2e-11  1.0e-5    8.5e-10   7.0e-10
  1e-5    9.2e-13  3.3e-3    6.8e-7    1.4e-7
  with an active cut (+ a dropped duplicate at 5e-15, lindep 1e-12, kept s_min 9.2e-11): explicit 1.0e-4,
  eigen 1.05e-8, floor 6.8e-10.  'eigen_z' / 'eigen_wm' alone: 2.6e-5 / 1.6e-5 at delta 1e-4 (both are needed).
  Eigendecomposition is NOT the source (separate A/B, not in this script): exact eig of the rounded J2 + double contractions 2.9e-5 at delta 1e-4;
  LAPACK eig + exact contractions 8.5e-10.  The kept-dropped Loewner term is not the source either (removing it
  changes nothing when nothing is dropped).  Energies are untouched (gradient-only change).

Usage: python kgrad_conditioning.py            (the table; ~3 min, single-threaded mpmath)
       python kgrad_conditioning.py anchor     (the exact-arithmetic FD anchor)
"""

import sys

import mpmath as mp
import numpy as np

mp.mp.dps = 40
PI52 = 2 * mp.pi ** mp.mpf(2.5)
ORB = [18.731137, 2.8253937, 0.6401217, 0.1612778]
AUX = [4.7, 1.9, 0.75, 0.3]
GEOM = [(0.0, 0.0, 0.0), (0.0, 0.1, 1.45), (1.6, -0.2, 2.3), (2.1, 0.3, 3.9)]


def f01(t):
    if t < mp.mpf("1e-12"):
        return 1 - t / 3 + t * t / 10, mp.mpf(1) / 3 - t / 5 + t * t / 14
    st = mp.sqrt(t)
    f0 = mp.sqrt(mp.pi) / (2 * st) * mp.erf(st)
    return f0, (f0 - mp.exp(-t)) / (2 * t)


def norm_s(a):
    return (2 * a / mp.pi) ** mp.mpf(0.75)


def basis(dups):
    """dups: list of (delta, atom, base exponent) near-duplicate s shells."""
    orb = [(mp.mpf(a), i) for i in range(len(GEOM)) for a in ORB]
    aux = [(mp.mpf(a), i) for i in range(len(GEOM)) for a in AUX]
    aux += [(mp.mpf(b) * (1 + mp.mpf(d)), i) for d, i, b in dups]
    return orb, aux


def integrals(geom, dups, deriv=True):
    """J3[m,n,P], J2[P,Q] (normalised s primitives) and dJ3[A,x,m,n,P], dJ2[A,x,P,Q] (nuclear)."""
    R = [mp.matrix([mp.mpf(c) for c in r]) for r in geom]
    orb, aux = basis(dups)
    no, na, nat = len(orb), len(aux), len(geom)
    obj = dict(dtype=object)
    J3, J2 = np.empty((no, no, na), **obj), np.empty((na, na), **obj)
    dJ3 = np.full((nat, 3, no, no, na), mp.mpf(0), **obj)
    dJ2 = np.full((nat, 3, na, na), mp.mpf(0), **obj)
    for P, (g, ic) in enumerate(aux):
        for Q in range(P, na):
            d, jd = aux[Q]
            mu = g * d / (g + d)
            ab = R[ic] - R[jd]
            f0, f1 = f01(mu * (ab.T * ab)[0])
            pref = PI52 / (g * d * mp.sqrt(g + d)) * norm_s(g) * norm_s(d)
            J2[P, Q] = J2[Q, P] = pref * f0
            for x in range(3):
                v = -pref * f1 * 2 * mu * ab[x]
                for pp, qq in {(P, Q), (Q, P)}:
                    dJ2[ic, x, pp, qq] += v
                    dJ2[jd, x, pp, qq] -= v
    for m, (a, ia) in enumerate(orb):
        for n in range(m, no):
            b, ib = orb[n]
            p, mu = a + b, a * b / (a + b)
            ab = R[ia] - R[ib]
            K = mp.exp(-mu * (ab.T * ab)[0])
            pc0 = (a * R[ia] + b * R[ib]) / p
            for P, (g, ic) in enumerate(aux):
                rho = p * g / (p + g)
                pc = pc0 - R[ic]
                f0, f1 = f01(rho * (pc.T * pc)[0])
                pref = (
                    PI52 / (p * g * mp.sqrt(p + g)) * norm_s(a) * norm_s(b) * norm_s(g)
                )
                J3[m, n, P] = J3[n, m, P] = pref * K * f0
                if not deriv:
                    continue
                for x in range(3):
                    dka = -2 * mu * ab[x] * K
                    da = pref * (dka * f0 - K * f1 * 2 * rho * pc[x] * a / p)
                    db = pref * (-dka * f0 - K * f1 * 2 * rho * pc[x] * b / p)
                    dc = pref * K * f1 * 2 * rho * pc[x]
                    for mm, nn in {(m, n), (n, m)}:
                        dJ3[ia, x, mm, nn, P] += da
                        dJ3[ib, x, mm, nn, P] += db
                        dJ3[ic, x, mm, nn, P] += dc
    return J3, J2, dJ3, dJ2


to_f = np.vectorize(float, otypes=[float])
to_mp = np.vectorize(mp.mpf, otypes=[object])


def eigh(J2, exact):
    if not exact:
        return np.linalg.eigh(J2)
    E, Q = mp.eigsy(mp.matrix(J2.tolist()))
    return np.array([E[i] for i in range(len(E))], object), np.array(Q.tolist(), object)


def fvec(s, lindep):
    keep = np.array([float(x) > lindep for x in s])
    return np.array(
        [1 / x if k else 0 * x for x, k in zip(s, keep)], dtype=s.dtype
    ), keep


def energy(J3, J2, D, lindep, exact):
    """The build's form: B = s^-1/2 U^T J3 (kept), E = 1/2 |B.D|^2 - 1/4 sum_i tr[B_i D B_i D]."""
    s, U = eigh(J2, exact)
    f, _ = fvec(s, lindep)
    sq = np.array([mp.sqrt(x) for x in f], object) if exact else np.sqrt(f)
    B = np.einsum("mnP,Pi->mni", J3, U) * sq
    rb = np.einsum("mni,nm->i", B, D)
    BD = np.einsum("mni,nk->mki", B, D)
    return 0.5 * np.sum(rb * rb) - 0.25 * np.einsum("mki,kmi->", BD, BD)


def grad(J3, J2, dJ3, dJ2, D, lindep, exact, form="eigen", eig=None):
    s, U = eigh(J2, exact) if eig is None else eig
    f, keep = fvec(s, lindep)
    n = len(s)
    Lo = np.empty((n, n), dtype=s.dtype)
    for i in range(n):
        for j in range(n):
            if keep[i] and keep[j]:
                Lo[i, j] = -f[i] * f[j]
            elif s[i] == s[j]:
                Lo[i, j] = 0 * s[i]
            else:
                Lo[i, j] = (f[i] - f[j]) / (s[i] - s[j])
    rho = np.einsum("mnP,nm->P", J3, D)
    T = np.einsum("ab,bcP,cd->adP", D, J3, D)  # D J3_P D
    # explicit pieces (pre-#512)
    Jinv = (U * f) @ U.T
    c_x = Jinv @ rho
    Z_x = np.einsum("P,mn->mnP", c_x, D) - 0.5 * np.einsum("adQ,QP->adP", T, Jinv)
    H = 0.5 * np.outer(rho, rho) - 0.25 * np.einsum("mnP,nmQ->PQ", J3, T)
    Wm_x = U @ (Lo * (U.T @ H @ U)) @ U.T
    # eigenbasis pieces (the fix)
    Tu = np.einsum("adQ,Qi->adi", T, U)
    J3u = np.einsum("mnP,Pi->mni", J3, U)
    rt = U.T @ rho
    c_e = U @ (f * rt)
    Z_e = np.einsum("P,mn->mnP", c_e, D) - 0.5 * np.einsum("adi,Pi->adP", Tu * f, U)
    Ht = 0.5 * np.outer(rt, rt) - 0.25 * np.einsum("mni,nmj->ij", J3u, Tu)
    Wm_e = U @ (Lo * Ht) @ U.T
    Z, Wm = {
        "explicit": (Z_x, Wm_x),
        "eigen": (Z_e, Wm_e),
        "eigen_z": (Z_e, Wm_x),
        "eigen_wm": (Z_x, Wm_e),
    }[form]
    return np.einsum("Axmnp,mnp->Ax", dJ3, Z) + np.einsum("AxPQ,QP->Ax", dJ2, Wm)


def density(no, seed=7, nocc=3):
    C = np.random.default_rng(seed).standard_normal((no, nocc)) * 0.3
    return C @ C.T


def row(dups, lindep, D):
    J3m, J2m, dJ3m, dJ2m = integrals(GEOM, dups)
    ints_d = [to_f(x) for x in (J3m, J2m, dJ3m, dJ2m)]
    Dm = to_mp(D)
    g_ex = to_f(grad(J3m, J2m, dJ3m, dJ2m, Dm, lindep, True))
    floor = to_f(grad(*[to_mp(x) for x in ints_d], Dm, lindep, True))
    s = np.linalg.eigvalsh(ints_d[1])
    out = {
        f: abs(grad(*ints_d, D, lindep, False, f) - g_ex).max()
        for f in ("explicit", "eigen", "eigen_z", "eigen_wm")
    }
    print(
        f"dups {dups} lindep {lindep:.0e}: s<1e-6 {[f'{x:.1e}' for x in s[s < 1e-6]]} | "
        + " ".join(f"{k} {v:.2e}" for k, v in out.items())
        + f" | floor {abs(floor - g_ex).max():.2e}",
        flush=True,
    )


def anchor(D):
    for dups in ([], [(1e-4, 0, 0.75)]):
        J3, J2, dJ3, dJ2 = integrals(GEOM, dups)
        Dm = to_mp(D)
        g = grad(J3, J2, dJ3, dJ2, Dm, 1e-14, True)
        h, (A, x) = mp.mpf("1e-12"), (0, 2)
        es = []
        for sg in (1, -1):
            geom = [list(r) for r in GEOM]
            geom[A][x] = mp.mpf(geom[A][x]) + sg * h
            j3, j2, _, _ = integrals([tuple(r) for r in geom], dups, deriv=False)
            es.append(energy(j3, j2, Dm, 1e-14, True))
        print(
            f"anchor dups {dups}: |exact Loewner - exact FD| {float(abs((es[0] - es[1]) / (2 * h) - g[A, x])):.1e}"
        )


if __name__ == "__main__":
    D = density(len(ORB) * len(GEOM))
    if sys.argv[1:] == ["anchor"]:
        anchor(D)
    else:
        for dups, lindep in [
            ([], 1e-10),
            ([(1e-3, 0, 0.75)], 1e-14),
            ([(1e-4, 0, 0.75)], 1e-14),
            ([(1e-5, 0, 0.75)], 1e-14),
            ([(1e-4, 0, 0.75), (1e-6, 0, 1.9)], 1e-12),
        ]:
            row(dups, lindep, D)
