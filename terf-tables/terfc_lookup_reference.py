#!/usr/bin/env python3
"""
REFERENCE implementation of the terfc primitive-integral lookup from the
Dutoi & Head-Gordon interpolation tables (JPCA 112, 2110 (2008), Sec. 3).

This is the machine-precision Python reference that the C++ TerfcEngine in
shim.cc must reproduce. It is SELF-VERIFYING: run it and it compares the
table-lookup terfc (s|s|s) base integral against the independent 1e-60 closed-form
oracle (base_terfc_closed.py) and asserts agreement to ~1e-12 (poly-10 interp).

THE RECIPE (all argument mappings — the crux; earlier attempts used the WRONG ones):
  theta = (1/p + 1/q)^{-1/2}          # Coulomb reduced exponent (theta^2 = pq/(p+q))
  phi   = (1/p + 1/q + 1/omega^2)^{-1/2}   # FOLDS IN 1/omega^2  <-- the key
  T = (theta*R)^2      S = (phi*R)^2      s = (phi*r0)^2         # R = |R_PQ|
  omega = 1/(r0*sqrt2) (curvature constraint)
Fundamental integrals (Dutoi Eq 6/9/11), in the "average interaction" normalization:
  I_pq[coulomb](R) = (2*theta/sqrt(pi)) * F_0(T)
  I_pq[terf](R)    = (2*phi/sqrt(pi))   * G_0(S,s)     # the 1/2 in terf is baked into G_0
  I_pq[terfc](R)   = I_pq[coulomb] - I_pq[terf]
Higher angular momentum uses F_m(T) and G_m^(n)(S,s) (m up to 4*l_max) as the Boys
replacements in the STANDARD Coulomb Obara-Saika recurrences (R-derivatives in T;
the n-index is the s-derivative order, 0 for energies). The tables store G_m^(n)
(our generate_tables.py G[m][n] == Dutoi Eq 19, with the gS[m+1] offset).

Interpolation: Dutoi uses 10x10-term polynomial interpolation on the 1/pts-integer
grid; that reaches machine precision (~1e-15) vs ~1e-4 for bilinear. The finest
covering table is selected (16_4_2 -> 8_10_5 -> 4_20_20 -> 2_20_80); for S>~70 or
s>~150 the asymptotic Eq 21 replaces the table (not needed for bound pairs).
"""
import os, sys, struct
import numpy as np
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ---- table loading -----------------------------------------------------------
# (pts, S_max, s_max, filename), finest-first for query-time selection.
_SPECS = [(16, 4, 2, "16_4_2.bin"), (8, 10, 5, "8_10_5.bin"),
          (4, 20, 20, "4_20_20.bin"), (2, 20, 80, "2_20_80.bin")]

def load_tables(directory=HERE):
    tbls = []
    for pts, Smax, smax, fn in _SPECS:
        path = os.path.join(directory, fn)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            nS, ns, dimm, dimn = struct.unpack("<iiii", f.read(16))
            data = np.fromfile(f, dtype=np.float64).reshape(nS, ns, dimm, dimn)
        tbls.append(dict(pts=pts, Smax=Smax, smax=smax, nS=nS, ns=ns,
                         dimm=dimm, dimn=dimn, G=data))
    return tbls

def _pick(tbls, S, s):
    for t in tbls:
        if S <= t["Smax"] and s <= t["smax"]:
            return t
    return None

# ---- 10x10 polynomial interpolation of G_m^(n)(S,s) --------------------------
def _lagrange_1d(nodes, vals, x):
    """Lagrange interpolation on consecutive-integer nodes."""
    tot = 0.0
    for j, xj in enumerate(nodes):
        term = vals[j]
        for k, xk in enumerate(nodes):
            if k != j:
                term *= (x - xk) / (xj - xk)
        tot += term
    return tot

def interp_G(tbls, S, s, m, n, K=10):
    """Poly-K interpolation of G_m^(n)(S,s) from the finest covering table."""
    t = _pick(tbls, S, s)
    if t is None:
        raise ValueError(f"(S={S}, s={s}) outside all tables (use asymptotic Eq 21)")
    pts, nS, ns, G = t["pts"], t["nS"], t["ns"], t["G"]
    fS, fs = S * pts, s * pts

    def window(f, N):
        i0 = int(np.floor(f)) - K // 2 + 1
        return max(0, min(i0, N - K)) if N >= K else 0

    K_S = min(K, nS); K_s = min(K, ns)
    iS0 = window(fS, nS); is0 = window(fs, ns)
    Si = list(range(iS0, iS0 + K_S)); si = list(range(is0, is0 + K_s))
    col = []
    for iS in Si:
        vals = [G[iS, js, m, n] for js in si]
        col.append(_lagrange_1d(si, vals, fs))
    return _lagrange_1d(Si, col, fS)

# ---- terfc fundamental integral (s|s|s), Dutoi normalization -----------------
def boys0(T):
    T = mp.mpf(T)
    if T == 0:
        return mp.mpf(1)
    return mp.gammainc(mp.mpf('0.5'), 0, T) / (2 * T ** mp.mpf('0.5'))

def terfc_base_avg(tbls, p, q, R, r0):
    """I_pq[terfc](R) in the average-interaction normalization (Dutoi Eq 2)."""
    p, q, R, r0 = map(mp.mpf, (p, q, R, r0))
    omega = 1 / (r0 * mp.sqrt(2))
    theta = 1 / mp.sqrt(1 / p + 1 / q)
    phi = 1 / mp.sqrt(1 / p + 1 / q + 1 / omega ** 2)
    T = float((theta * R) ** 2); S = float((phi * R) ** 2); s = float((phi * r0) ** 2)
    G0 = mp.mpf(interp_G(tbls, S, s, 0, 0))
    coul = (2 * theta / mp.sqrt(mp.pi)) * boys0(T)
    terf = (2 * phi / mp.sqrt(mp.pi)) * G0
    return coul - terf, theta  # theta returned for the pref calibration


if __name__ == "__main__":
    from gt_check import setup, gt_phi_over_pref
    tbls = load_tables()
    if not any(t["pts"] == 16 for t in tbls):
        print("Need at least 16_4_2.bin; run generate_tables.py first.")
        sys.exit(1)
    p, a, b = mp.mpf('1.3'), mp.mpf('0.9'), mp.mpf('0.7')
    Ra = [mp.mpf(0)] * 3; Rb = [mp.mpf(0), mp.mpf(0), mp.mpf('0.8')]
    print("terfc table-lookup (poly-10) vs 1e-60 closed-form oracle")
    print("%-6s %-7s %-8s %-8s %-14s %-14s %-9s" %
          ("r0/A", "R", "S", "s", "table", "oracle", "reldiff"))
    A2B = 1.8897259886
    worst = mp.mpf(0)
    for r0A in [0.75, 1.05, 1.35, 2.0]:
        r0 = mp.mpf(r0A) * A2B
        omega = 1 / (r0 * mp.sqrt(2))
        for Rpz in [mp.mpf('0.6'), mp.mpf('1.2'), mp.mpf('2.2'), mp.mpf('3.5')]:
            Rp = [mp.mpf('0.2'), mp.mpf(0), Rpz]
            par = setup(p, a, b, Ra, Rb, Rp, r0)
            q, R = par['q'], par['D']
            theta = 1 / mp.sqrt(1 / p + 1 / q)
            phi = 1 / mp.sqrt(1 / p + 1 / q + 1 / omega ** 2)
            S = float((phi * R) ** 2); s = float((phi * r0) ** 2)
            t = _pick(tbls, S, s)
            if t is None:
                continue
            terfc_avg, _ = terfc_base_avg(tbls, p, q, R, r0)
            # calibrate avg-norm -> oracle [0|op|0]/pref units via the Coulomb anchor
            coul_avg = (2 * theta / mp.sqrt(mp.pi)) * boys0(float((theta * R) ** 2))
            coul_orc = gt_phi_over_pref(par, lambda r: 1 / r)
            C = coul_orc / coul_avg
            table_val = C * terfc_avg
            def terfcop(r, r0=r0, w=omega):
                return 1 / r - (mp.erf(w * (r - r0)) + mp.erf(w * (r + r0))) / (2 * r)
            oracle_val = gt_phi_over_pref(par, terfcop)
            rel = abs(table_val - oracle_val) / abs(oracle_val)
            worst = max(worst, rel)
            print("%-6.2f %-7.3f %-8.4f %-8.4f %-14.8g %-14.8g %.2e" %
                  (r0A, float(R), S, s, float(table_val), float(oracle_val), float(rel)))
    print()
    ok = worst < mp.mpf('1e-10')
    print("WORST rel diff =", mp.nstr(worst, 4), "->", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
