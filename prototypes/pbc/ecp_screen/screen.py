"""Replica of the periodic-ECP triple screen of crates/ferric-pbc/src/ecp.rs (EcpPlan::build/kept) on HI/LANL2DZ
(tests/pbc_ecp.rs hi_cell), plus exact per-triple magnitudes from the ferric-owned quadrature engine
(prototypes/pbc/ecp_quadrature/ecpq.py, screen=False)."""

import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ecp_quadrature"))
import ecpq  # noqa: E402
import fixtures as fx  # noqa: E402

LAT = np.array([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 7.0]])
POS = np.array([[0.3, 0.2, 0.4], [0.3, 0.2, 3.44]])  # H, I
MARGIN = 3.0


def mu(x, y):
    return x * y / (x + y)


def translations(rcut, pos=POS, lat=LAT):
    """Cell.translations: min atom-atom distance between cell and image <= rcut, sorted by |L|."""
    n = [int(math.ceil((rcut + 20.0) / np.linalg.norm(lat[i]))) + 1 for i in range(3)]
    out = []
    for i, j, k in itertools.product(*[range(-m, m + 1) for m in n]):
        t = i * lat[0] + j * lat[1] + k * lat[2]
        d = np.min(np.linalg.norm(pos[:, None, :] - (pos[None, :, :] + t), axis=2))
        if d <= rcut:
            out.append(t)
    out.sort(key=lambda v: float(np.linalg.norm(v)))
    return np.array(out)


def home_shells():
    return fx.hi_shells(POS[0], POS[1])


def template():
    e = fx.ecp_i(POS[1])
    return e


class OldScreen:
    """ecp.rs as of 2026-09-29."""

    def __init__(self, prec):
        self.sh = home_shells()
        self.e = template()
        self.amin = np.array([min(s["exponents"]) for s in self.sh])
        self.log_pref = np.array(
            [
                math.log(sum(abs(c) for c in s["coefficients"]))
                + 0.75 * math.log(math.pi / a)
                for s, a in zip(self.sh, self.amin)
            ]
        )
        self.zmin = min(self.e["exponents"])
        dsum = sum(abs(c) for c in self.e["coefficients"])
        self.x = max(math.log(dsum / prec) + MARGIN, 0.0)
        pref_max = 2 * max(0.0, self.log_pref.max())
        self.x_max = self.x + pref_max
        alo = self.amin.min()
        self.r_ecp = math.sqrt(self.x_max / mu(alo, self.zmin))
        self.r_pair = min(math.sqrt(self.x_max / mu(alo, alo)), 2 * self.r_ecp)

    def keep(self, a, b, ra, rb, rc):
        dab2 = np.sum((ra - rb) ** 2)
        e_ab = mu(self.amin[a], self.amin[b]) * dab2
        if e_ab > self.x_max:
            return False
        e = max(
            e_ab,
            mu(self.amin[a], self.zmin) * np.sum((ra - rc) ** 2),
            mu(self.amin[b], self.zmin) * np.sum((rb - rc) ** 2),
        )
        return e <= self.x + max(self.log_pref[a] + self.log_pref[b], 0.0)


def triple_value(sa, sb, rc, channels="all"):
    """max-abs of the spherical <sa|U_C|sb> block (exact quadrature, engine screen OFF)."""
    e = template()
    e = dict(e, center=list(rc))
    v = ecpq.triple_cart(sa, sb, e, screen=False, channels=channels)
    return float(np.max(np.abs(ecpq.to_sph(v, sa["l"], sb["l"]))))


def shifted(s, t):
    return dict(s, center=list(np.asarray(s["center"], float) + t))


# ----------------------------------------------------------------------------- the new (rigorous) bound
EPS = 0.1  # share of a polynomial-carrying Gaussian's exponent spent on bounding its polynomial


def kappa(ang):
    """sup over the unit sphere of a libcint-normalised real solid harmonic / r^l (addition theorem)."""
    return math.sqrt((2 * ang + 1) / (4 * math.pi))


def poly_q(n, a):
    """sup_s s^n e^{-EPS a s^2} = (n / (2 e EPS a))^{n/2}; 1 for n = 0."""
    return 1.0 if n == 0 else (n / (2 * math.e * EPS * a)) ** (n / 2)


def shrink(n, a):
    return a if n == 0 else (1 - EPS) * a


class NewScreen:
    """|<a|U_C|b>| <= sum_k 4pi |d_k| P_a P_b W_k sqrt(pi/p_k) e^{-K_k},
    P = kappa_l sum_i |c_i N_i| Q_l(alpha_i), alpha -> alpha' = (1-EPS) alpha for l > 0 (alpha_min),
    W_k = Q_{n_k}(zeta_k), zeta' likewise for n_k > 0, p = alpha' + beta' + zeta',
    K = (alpha' beta' (R_A - R_B)^2 + alpha' zeta' R_A^2 + beta' zeta' R_B^2) / p (radial, |A-B| never enters)."""

    def __init__(self, prec):
        self.prec = prec
        self.sh = home_shells()
        self.e = template()
        self.amin = np.array([min(s["exponents"]) for s in self.sh])
        self.ap = np.array([shrink(s["l"], a) for s, a in zip(self.sh, self.amin)])
        self.P = np.array(
            [
                kappa(s["l"])
                * sum(
                    abs(c) * poly_q(s["l"], x)
                    for c, x in zip(s["coefficients"], s["exponents"])
                )
                for s in self.sh
            ]
        )
        self.terms = [
            (abs(d) * poly_q(n, z) * 4 * math.pi, shrink(n, z))
            for n, z, d in zip(
                self.e["ns"], self.e["exponents"], self.e["coefficients"]
            )
        ]

    def bound(self, a, b, RA, RB):
        al, be = self.ap[a], self.ap[b]
        tot = 0.0
        for w, z in self.terms:
            p = al + be + z
            K = (al * be * (RA - RB) ** 2 + al * z * RA**2 + be * z * RB**2) / p
            tot += w * math.sqrt(math.pi / p) * math.exp(-K)
        return self.P[a] * self.P[b] * tot

    def keep(self, a, b, ra, rb, rc):
        return (
            self.bound(a, b, np.linalg.norm(ra - rc), np.linalg.norm(rb - rc))
            >= self.prec
        )
