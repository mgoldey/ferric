"""DERIVATION (no ferric run) of the production-vs-oracle tolerance of the pair-FT kernel
after the two value-changing options V2 (image split of the phase) and V6 (per-site F-row
premultiply), on the EXACT inputs of tests/pbc_parallel_bitwise.rs's pair-FT cases.

Replays PairFtPlan's screen (same cc, gauss, mag >= thresh, g2max window, g2min drop) on the
triclinic 4-H cell with spd_basis_h / gc_basis_h, the scrambled G list (x1, x6) and the
residue K = G + q list, and for every (survivor, G) evaluates in f64, operation for operation:

  oracle : ph = G.P_c, P_c = (a A + b (B + L)) / p           u = (cos ph, -sin ph)
  V2     : ph0 = G.P0, P0 = (a A + b B) / p; beta = b / p; k = m.n (m Miller of G, n of L)
           T(k) = sincos(-2 pi frac(beta k)); c = sincos(-2 pi frac(beta f.n)) (f the common
           fractional Miller offset of a K = G + q list); u = home(ph0) * (T(k) * c)

Per Cartesian element (shell pair, u, v, bucket, G) it then reports
  BOUND  = sum over its terms |term| (|u_V2 - u_oracle| + 8 eps)      (triangle inequality;
           the 8 eps covers V6's re-association, measured 5.2e-16 per term by
           phase_options.py, and the changed summation order)
and, COHERENTLY (the actual element change, not a bound), the MUTANTS
  M1 image factor conjugated (sign of b/p G.L flipped)   M2 b/p dropped (beta = 1)
  M3 Miller index m_0 + 1                                M4 a/p used instead of b/p
  M5 split forced on a NON-lattice list (G x 1.5, m = round(x - f)): what the detection
     prevents
Elements are Cartesian (the pure-d transform only mixes them with |c| <= ~1.1 per term).

Also: the Miller residual |x - f - m| / (eps (2 + S_g + S_*)) of every G of every list (the
detection criterion of plan.rs, MILLER_TOL_EPS = 64), and whether the x1.5 list is rejected.

Usage: python split_tolerance.py   (~4.5 min, one core: run it niced, not beside timings)
"""

import math

import numpy as np

EPS = 2.0**-52
TAU = 2.0 * math.pi
THRESH = 1e-15
MARGIN = 10.0
MILLER_TOL_EPS = 64.0

TRI_A = [[4.6, 0.0, 0.0], [0.9, 4.3, 0.0], [0.5, 0.7, 4.8]]
ATOMS = [[0.1, 0.2, 0.3], [0.1, 0.2, 1.7], [2.4, 2.5, 2.2], [3.6, 2.9, 2.6]]
SE = [3.42525091, 0.62391373, 0.1688554]


def cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def reciprocal(lat):
    a1, a2, a3 = lat
    det = dot(a1, cross(a2, a3))
    cols = [cross(a2, a3), cross(a3, a1), cross(a1, a2)]
    inv = [[cols[c][r] / det for c in range(3)] for r in range(3)]
    return [[TAU * inv[k][i] for k in range(3)] for i in range(3)]


def index_combination(rows, n):
    if n == (0, 0, 0):
        return [0.0, 0.0, 0.0]
    f0, f1, f2 = float(n[0]), float(n[1]), float(n[2])
    return [f0 * rows[0][c] + f1 * rows[1][c] + f2 * rows[2][c] for c in range(3)]


def scrambled(b, shift, scale):
    out = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                out.append(
                    [
                        (i * b[0][c] + j * b[1][c] + k * b[2][c] + shift[c]) * scale
                        for c in range(3)
                    ]
                )
    n = len(out)
    return np.array([out[(37 * i) % n] for i in range(n)])


def renorm(ang, exps, coefs):
    s = sum(
        ca * cb * (2 * math.sqrt(a * b) / (a + b)) ** (ang + 1.5)
        for a, ca in zip(exps, coefs)
        for b, cb in zip(exps, coefs)
    )
    return [c / math.sqrt(s) for c in coefs]


def prim_norm(a, ang):
    return (
        (2 * a / math.pi) ** 0.75
        * math.sqrt((4 * a) ** ang)
        / math.sqrt([1, 1, 3, 15][ang])
    )


def shells_of(kind):
    raw = {
        "spd": [
            (0, SE, [0.15432897, 0.53532814, 0.44463454]),
            (1, [0.8], [1.0]),
            (2, [0.9], [1.0]),
        ],
        "gc": [
            (0, SE, [0.15432897, 0.53532814, 0.44463454]),
            (0, SE, [-0.3, 0.2, 1.0]),
            (0, SE, [0.0, 0.0, 1.0]),
            (1, [1.2, 0.35], [0.6, 0.5]),
            (1, [1.2, 0.35], [0.0, 1.0]),
            (2, [0.9], [1.0]),
        ],
    }[kind]
    out = []
    for at, c in enumerate(ATOMS):
        for ang, e, co in raw:
            cn = renorm(ang, e, co)
            out.append((at, c, ang, e, [x * prim_norm(a, ang) for x, a in zip(cn, e)]))
    return out


def comps(ang):
    return [(ang - i, i - j, j) for i in range(ang + 1) for j in range(i + 1)]


def e_table(la, lb, a, b, q):
    """E[i][j][t] (1-D McMurchie-Davidson, as md3c1e::e_table)."""
    p = a + b
    mu, inv2p = a * b / p, 0.5 / p
    xpa, xpb = -(b / p) * q, (a / p) * q
    E = np.zeros((la + 1, lb + 1, la + lb + 2))
    E[0, 0, 0] = math.exp(-mu * q * q)
    for i in range(la):
        for t in range(i + 2):
            E[i + 1, 0, t] = (
                inv2p * (E[i, 0, t - 1] if t else 0)
                + xpa * E[i, 0, t]
                + (t + 1) * E[i, 0, t + 1]
            )
    for j in range(lb):
        for i in range(la + 1):
            for t in range(i + j + 2):
                E[i, j + 1, t] = (
                    inv2p * (E[i, j, t - 1] if t else 0)
                    + xpb * E[i, j, t]
                    + (t + 1) * E[i, j, t + 1]
                )
    return E


def translations(lat, rcut):
    """(n, L) of every image with a point-point distance <= rcut (L as index_combination)."""
    nb = 8
    out = []
    pos = np.array(ATOMS)
    for n0 in range(-nb, nb + 1):
        for n1 in range(-nb, nb + 1):
            for n2 in range(-nb, nb + 1):
                n = (n0, n1, n2)
                L = index_combination(lat, n)
                d = pos[:, None, :] - (pos[None, :, :] + np.array(L))
                if np.sqrt((d * d).sum(-1)).min() <= rcut:
                    out.append((n, L))
    return out


def miller(g, lat, f):
    """plan.rs miller_of: m = round(x - f), accepted when |x - f - m| <= tol."""
    x = np.array(
        [
            (g[:, 0] * lat[d][0] + g[:, 1] * lat[d][1] + g[:, 2] * lat[d][2]) / TAU
            for d in range(3)
        ]
    ).T
    s = np.array(
        [
            (
                np.abs(g[:, 0] * lat[d][0])
                + np.abs(g[:, 1] * lat[d][1])
                + np.abs(g[:, 2] * lat[d][2])
            )
            / TAU
            for d in range(3)
        ]
    ).T
    return x, s


def detect(g, lat):
    x, s = miller(g, lat, None)
    f = np.zeros(3)
    sstar = np.zeros(3)
    for d in range(3):
        i = int(np.argmin(s[:, d]))
        sstar[d] = s[i, d]
        fd = x[i, d] - np.round(x[i, d])
        f[d] = 0.0 if abs(fd) <= MILLER_TOL_EPS * EPS * (2 + 2 * s[i, d]) else fd
    m = np.round(x - f)
    ratio = np.abs(x - f - m) / (EPS * (2 + s + sstar))
    return f, m.astype(np.int64), float(ratio.max())


def sincos_neg(x):
    return np.cos(x) - 1j * np.sin(x)


def frac_phase(beta, k):
    xk = beta * k
    return sincos_neg(TAU * (xk - np.round(xk)))


def run(kind, gv, lat, moduli, force_split=False):
    shells = shells_of(kind)
    amin = min(min(s[3]) for s in shells)
    rpair = math.sqrt(2 * math.log(1e3 / THRESH) / amin) + 2
    imgs = translations(lat, rpair)
    g2 = (gv * gv).sum(1)
    gmax = math.sqrt(g2.max())
    g2min = g2.min()
    f, m, ratio = detect(gv, lat)
    if ratio > MILLER_TOL_EPS and not force_split:
        return None, ratio
    bound, mut = {}, {k: {} for k in ("M1", "M2", "M3", "M4", "V2")}
    count, sumabs = {}, {}
    pmax = 0.0
    oracle_sum = {}
    for ia, sa in enumerate(shells):
        for ib, sb in enumerate(shells):
            (_, A, la, ea, ca), (_, B, lb, eb, cb) = sa, sb
            lg = (la + lb) * math.log(max(gmax, 1.0))
            ca_l, cb_l = comps(la), comps(lb)
            for n, L in imgs:
                bc = [B[d] + L[d] for d in range(3)]
                ab = [A[d] - bc[d] for d in range(3)]
                r2 = dot(ab, ab)
                bucket = (
                    0
                    if moduli is None
                    else ((n[0] % moduli[0]) * moduli[1] + n[1] % moduli[1]) * moduli[2]
                    + n[2] % moduli[2]
                )
                for ka, a in enumerate(ea):
                    for kb, b in enumerate(eb):
                        p = a + b
                        cc = ca[ka] * cb[kb] * (math.pi / p) ** 1.5
                        mag = abs(cc * math.exp(-a * b / p * r2))
                        if mag < THRESH:
                            continue
                        g2w = 4 * p * (max(math.log(mag / THRESH), 0) + MARGIN + lg)
                        if g2w < g2min:
                            continue
                        sel = g2 <= g2w
                        g = gv[sel]
                        pc = [(a * A[d] + b * bc[d]) / p for d in range(3)]
                        ph = g[:, 0] * pc[0] + g[:, 1] * pc[1] + g[:, 2] * pc[2]
                        u_or = sincos_neg(ph)
                        p0 = [(a * A[d] + b * B[d]) / p for d in range(3)]
                        home = sincos_neg(
                            g[:, 0] * p0[0] + g[:, 1] * p0[1] + g[:, 2] * p0[2]
                        )
                        mm = m[sel]
                        k = mm @ np.array(n)
                        fn = f[0] * n[0] + f[1] * n[1] + f[2] * n[2]
                        beta = b / p

                        def split(beta_, k_):
                            c = frac_phase(beta_, fn)
                            return home * (frac_phase(beta_, k_) * c)

                        u_v2 = split(beta, k)
                        variants = {
                            "V2": u_v2,
                            "M1": home
                            * np.conj(frac_phase(beta, k) * frac_phase(beta, fn)),
                            "M2": split(1.0, k),
                            "M3": split(beta, k + n[0]),
                            "M4": split(a / p, k),
                        }
                        e = np.exp(-g2[sel] / (4 * p))
                        Ex, Ey, Ez = (e_table(la, lb, a, b, ab[d]) for d in range(3))
                        F = []
                        for d, E in enumerate((Ex, Ey, Ez)):
                            pw = np.array(
                                [(-1j * g[:, d]) ** t for t in range(la + lb + 2)]
                            )
                            F.append(np.einsum("ijt,tg->ijg", E, pw))
                        gidx = np.nonzero(sel)[0]
                        for u, (ax, ay, az) in enumerate(ca_l):
                            for v, (bx, by, bz) in enumerate(cb_l):
                                fprod = F[0][ax, bx] * F[1][ay, by] * F[2][az, bz]
                                term0 = cc * e * fprod
                                key = (ia, ib, u, v, bucket)
                                bnd = np.abs(term0) * (np.abs(u_v2 - u_or) + 8 * EPS)
                                acc = bound.setdefault(key, np.zeros(len(gv)))
                                np.add.at(acc, gidx, bnd)
                                cnt = count.setdefault(key, np.zeros(len(gv)))
                                np.add.at(cnt, gidx, 1.0)
                                sab = sumabs.setdefault(key, np.zeros(len(gv)))
                                np.add.at(sab, gidx, np.abs(term0))
                                o = oracle_sum.setdefault(
                                    key, np.zeros(len(gv), complex)
                                )
                                np.add.at(o, gidx, term0 * u_or)
                                for name, uu in variants.items():
                                    dd = mut[name].setdefault(
                                        key, np.zeros(len(gv), complex)
                                    )
                                    np.add.at(dd, gidx, term0 * (uu - u_or))
    pmax = max(np.abs(v).max() for v in oracle_sum.values())
    res = {"bound": max(v.max() for v in bound.values()), "pmax": pmax}
    # Worst-case re-ordering of an element's sum: (n - 1) eps sum|terms|.
    res["reorder"] = max(
        (np.maximum(count[k] - 1, 0) * EPS * sumabs[k]).max() for k in count
    )
    res["nterms"] = max(v.max() for v in count.values())
    for name, d in mut.items():
        res[name] = max(np.abs(v).max() for v in d.values())
    return res, ratio


def main():
    lat = TRI_A
    b = reciprocal(lat)
    q = [0.3 * b[0][c] + 0.5 * b[2][c] for c in range(3)]
    cases = [
        ("spd", "Gamma x1", scrambled(b, [0.0] * 3, 1.0), None),
        ("spd", "resid x1", scrambled(b, q, 1.0), (2, 1, 3)),
        ("spd", "Gamma x6", scrambled(b, [0.0] * 3, 6.0), None),
        ("spd", "resid x6", scrambled(b, q, 6.0), (2, 1, 3)),
        ("gc", "Gamma x6", scrambled(b, [0.0] * 3, 6.0), None),
        ("gc", "resid x6", scrambled(b, q, 6.0), (2, 1, 3)),
    ]
    print(
        "case                max|P|     BOUND     V2 coh    M1        M2        M3        M4   miller-ratio  max-terms  reorder-worst"
    )
    for kind, name, gv, mod in cases:
        r, ratio = run(kind, gv, lat, mod)
        print(
            f"{kind:3s} {name:9s}  {r['pmax']:9.2e} {r['bound']:9.2e} {r['V2']:9.2e} "
            f"{r['M1']:9.2e} {r['M2']:9.2e} {r['M3']:9.2e} {r['M4']:9.2e}   {ratio:.2f}"
            f"   {r['nterms']:6.0f}   {r['reorder']:.2e}"
        )
    for name, sh in (("Gamma x1.5", [0.0] * 3), ("resid x1.5", q)):
        gv = scrambled(b, sh, 1.5)
        r, ratio = run("spd", gv, lat, None)
        assert r is None, name
        rf, _ = run("spd", gv, lat, None, force_split=True)
        print(
            f"spd {name}: miller ratio {ratio:.3g} -> REJECTED (direct phase); "
            f"M5 (split forced) coherent change {rf['V2']:.2e} of max|P| {rf['pmax']:.2e}"
        )


if __name__ == "__main__":
    main()
