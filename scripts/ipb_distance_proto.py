#!/usr/bin/env python3
"""A DISTANCE-DEPENDENT integral partition bound for 3c-1e / COSX screening.

Companion pre-registration: ``scripts/queue/out/ipb_distance_prereg.md`` (written
and committed BEFORE this file).  Results: ``scripts/queue/out/ipb_distance_results.md``.

THE QUESTION
------------
Whitepaper §6.2b named the open lever: ferric's coarse sphere bound collapses the
shell pair onto its midpoint and subtracts ``|AB|/2`` from the distance, which
eats the distance entirely on 29-37% of pair-batch decisions, a share that GROWS
with system size.  §6.2a found the complementary weakness: the form of Thompson &
Ochsenfeld's integral partition bound (IPB) that sn-LinK actually uses is
deliberately batch-independent and carries NO distance decay at all, though it
treats the radial moment exactly.  Two bounds, loose in complementary places.

This script implements the third construction -- exact radial treatment PLUS a
valid distance factor -- and measures it against both.

THE FORMULA, AND THE TRAP IT AVOIDS
-----------------------------------
A previous attempt was rejected by its own exactness anchor (3678-8758
violations) because **Eq (A10)'s ``R`` is the PARTITIONING-BALL radius**::

    R_ab = max(0, R - |P_ab - C_uv|)                                   (A10)

``S_R`` and ``V_R`` are TAIL quantities: they integrate ``|Omega|`` only over the
COMPLEMENT of a ball of radius ``R`` about the pair centre.  ``V_R`` bounds the
potential *of the charge outside that ball*, maximised over all probe points.  It
is not "the potential at distance R".  Substituting a probe distance for ``R``
drops the charge INSIDE the ball -- which is most of it -- hence the violations.

The IPB paper's own Eq (A15), Newton's shell theorem, is the missing half.  For
a spherically symmetric ``S`` about ``p`` and a probe at distance ``D``::

    int S(r)/|r-r'| dr = (1/D) int_{|r|<=D} S  +  int_{|r|>D} S(r)/|r| dr   (A15)

Both terms are present: the inside charge gets an exact ``1/D`` from the theorem,
and the outside charge is EXACTLY ``V_D`` -- Eq (A16) with its ``R`` used as the
partitioning-ball radius it actually is.  Bounding the inside charge by the total
absolute charge ``S_0`` gives the bound implemented here::

    IPB-D(pair, D) = min( V_0 ,  S_0 / D  +  V_D )                        (*)

``V_0`` is precisely the batch-independent bound ``Xi_{nu,lam}`` of sn-LinK Eq (15),
so (*) reduces to it exactly at ``D = 0`` and can never be worse at any distance.

Note what (*) does NOT do: it never subtracts ``|AB|/2``.  The shell-pair extent
is handled exactly INSIDE ``V_D`` and ``S_0``, through each primitive pair's own
``R_ab``.  That is precisely the "treat the pair extent exactly instead of
halving it" step §6.2b names as the lever.

Run ``python3 scripts/ipb_distance_proto.py`` for the anchor table and the
comparison.  ``--mutate NAME`` runs a deliberately broken variant to prove an
anchor CAN fail (repo rule: a test you have never seen fail is an assumption).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

try:
    from pyscf import gto, scf, dft
    from pyscf import lib as pyscf_lib
except ImportError:  # pragma: no cover
    print("PySCF is required: pip install pyscf", file=sys.stderr)
    raise

from scipy.special import gammaincc, gamma as _gamma_fn

# One thread everywhere: another agent is running CPU-heavy work.
pyscf_lib.num_threads(1)

BATCH_POINTS = 256             # matches ferric's COSX_SUB_BATCH_POINTS
ROUNDING_SLACK = 1.0 + 1e-10   # cosx_screen.rs
COARSE_SLACK = 1.0 + 1e-9      # cosx_screen.rs

MUTATION = None                # set by --mutate; consumed at named points below

DFACT = [1.0, 1.0, 3.0, 15.0, 105.0, 945.0]


def upper_gamma(s: float, x: float) -> float:
    """Upper incomplete Gamma(s, x) = int_x^inf t^{s-1} e^{-t} dt, s > 0."""
    return float(gammaincc(s, x) * _gamma_fn(s))


def prim_norm(a: float, l: int) -> float:
    """N(a,l) = (2a/pi)^{3/4} (4a)^{l/2} / sqrt((2l-1)!!) -- md3c1e.rs:420."""
    return (2.0 * a / math.pi) ** 0.75 * math.sqrt((4.0 * a) ** l) / math.sqrt(DFACT[l])


def pure_factor(l: int, pure: bool, c2s) -> float:
    """Max absolute column sum of the cart->sph matrix (1 for cart or l < 2)."""
    if not pure or l < 2:
        return 1.0
    return float(np.max(np.sum(np.abs(c2s), axis=0)))


# ===========================================================================
# Shared shell data
# ===========================================================================

def collect_shells(mol: gto.Mole) -> list[dict]:
    """Per-shell exponents, contraction magnitudes, centre and pure factor.

    A generally contracted shell (e.g. cc-pVDZ oxygen's 8-primitive s shell,
    nctr=2) is several basis functions sharing one exponent set; a bound must
    cover ALL of them, hence the max over contractions.  Reading only the first
    contraction is the classic silent error here (3c1e_spec.md §5.7).
    """
    c2s = {l: gto.cart2sph(l) for l in range(6)}
    out = []
    for s in range(mol.nbas):
        l = mol.bas_angular(s)
        if l > 4:
            raise ValueError(f"shell {s} has l={l}; the bound tables stop at l=4")
        out.append(dict(
            l=l,
            center=np.asarray(mol.bas_coord(s), dtype=float),
            exps=np.asarray(mol.bas_exp(s), dtype=float),
            coefs=np.max(np.abs(mol.bas_ctr_coeff(s)), axis=1),
            sfac=pure_factor(l, mol.cart is False, c2s.get(l)),
        ))
    return out


# ===========================================================================
# Bound 1: ferric's Hoelder sphere bound (port of cosx_screen.rs)
# ===========================================================================

@dataclass
class FerricPair:
    mid: np.ndarray
    half: float          # |AB|/2 -- the term §6.2b identifies as the problem
    total: float         # bound at R = 0, valid everywhere
    sum_bg: float        # far-field numerator


class BoundsFerric:
    """ferric's current coarse bound, ported verbatim.

    For a primitive pair:  p = a+b, P = (aA+bB)/p, K_AB = exp(-ab/p |AB|^2),
    d_A = |P-A|, d_B = |P-B|.

        |P_mu(r-A) P_nu(r-B)| <= (rho+d_A)^la (rho+d_B)^lb = sum_k q_k rho^k
        int e^{-p rho^2}/|r-C|         = (2pi/p) F_0(p R^2)
        int rho^k e^{-p rho^2}/|r-C|  <= M_k (4pi/p) F_0(p R^2/2), M_k=(k/(p e))^{k/2}
        F_0(T) <= min(1, 0.5 sqrt(pi/T))

    The sphere query takes R_c = max(0, |M - centre| - radius - |AB|/2): every
    product centre lies on the segment AB, so its own R >= R_c.
    """
    name = "ferric"

    def __init__(self, mol: gto.Mole):
        self.mol = mol
        self.nsh = mol.nbas
        sh = collect_shells(mol)
        self.pairs = {}
        for s1 in range(self.nsh):
            for s2 in range(s1 + 1):
                self.pairs[(s1, s2)] = self._build(sh[s1], sh[s2])

    @staticmethod
    def _build(sa, sb) -> FerricPair:
        la, lb = sa["l"], sb["l"]
        q = sa["center"] - sb["center"]
        ab2 = float(q @ q)
        ab = math.sqrt(ab2)
        sfac = sa["sfac"] * sb["sfac"]
        tot = 0.0
        sum_bg = 0.0
        for a, ca in zip(sa["exps"], sa["coefs"]):
            ca_n = abs(ca) * prim_norm(a, la)
            for b, cb in zip(sb["exps"], sb["coefs"]):
                cb_n = abs(cb) * prim_norm(b, lb)
                p = a + b
                w = ca_n * cb_n * math.exp(-(a * b / p) * ab2) * sfac
                if w == 0.0:
                    continue
                da, db = b / p * ab, a / p * ab
                qk = np.zeros(la + lb + 1)
                for m in range(la + 1):
                    cm = math.comb(la, m) * da ** (la - m)
                    for n in range(lb + 1):
                        qk[m + n] += cm * math.comb(lb, n) * db ** (lb - n)
                beta0 = w * (2.0 * math.pi / p) * qk[0]
                s1k = sum(qk[k] * (k / (p * math.e)) ** (0.5 * k)
                          for k in range(1, len(qk)) if qk[k] != 0.0)
                beta1 = w * (4.0 * math.pi / p) * s1k
                g0 = 0.5 * math.sqrt(math.pi / p)
                g1 = 0.5 * math.sqrt(2.0 * math.pi / p)
                tot += beta0 + beta1
                sum_bg += beta0 * g0 + beta1 * g1
        slack = ROUNDING_SLACK * COARSE_SLACK
        return FerricPair(mid=0.5 * (sa["center"] + sb["center"]), half=0.5 * ab,
                          total=tot * slack, sum_bg=sum_bg * slack)

    def sphere(self, s1, s2, centre, radius) -> float:
        if s1 < s2:
            s1, s2 = s2, s1
        c = self.pairs[(s1, s2)]
        d = centre - c.mid
        rc = math.sqrt(float(d @ d)) - radius - c.half
        if rc <= 0.0:
            return c.total
        return min(c.total, c.sum_bg / rc)

    def degenerate(self, s1, s2, centre, radius) -> bool:
        """True when the query falls back to the distance-free R = 0 value."""
        if s1 < s2:
            s1, s2 = s2, s1
        c = self.pairs[(s1, s2)]
        d = centre - c.mid
        rc = math.sqrt(float(d @ d)) - radius - c.half
        return rc <= 0.0 or c.sum_bg / rc >= c.total


# ===========================================================================
# Bound 2 & 3: the integral partition bounds (Thompson & Ochsenfeld 2019)
# ===========================================================================
#
# For one primitive pair (a at A, l = la; b at B, l = lb):
#
#   p = a+b,  P_ab = (aA+bB)/p,  K_ab = exp(-ab/p |AB|^2)                (A7-A9)
#   |chi_a chi_b| <= N_a N_b |c_a c_b| K_ab sum_k F_k rho_P^k e^{-p rho_P^2}
#       with F_k = sum_{t+u=k} C(la,t) C(lb,u) |P-A|^{la-t} |P-B|^{lb-u}  (A12-A13)
#
# Both closed forms below integrate that spherically symmetric radial majorant
# over the complement of a ball of radius R_ab about P_ab:
#
#   S_R = K_ab sum_k F_k * 2 pi * Gamma((3+k)/2, p R_ab^2) / p^{(3+k)/2}  (~A14)
#   V_R = K_ab sum_k F_k * 2 pi * Gamma((2+k)/2, p R_ab^2) / p^{(2+k)/2}  (~A16)
#
# V_R uses Newton's shell theorem in the form sn-LinK Eq (16) states: for a
# spherically symmetric S the maximum of int S/|r-r'| over r' is attained at the
# spherical centre, so the tail's maximal potential is int S(r)/|r-P| over the
# tail region -- one power of rho less than the mass integral.
#
# BOTH closed forms were verified against an independent 4000-node
# Gauss-Legendre radial quadrature of the same majorant, to <= 5e-11 relative,
# for (la,lb) up to (3,4) and R up to 3.0 Bohr, before this file was written.
#
# R_ab = max(0, R - |P_ab - C_uv|)   (A10) -- R is the PARTITIONING-BALL radius.
# C_uv = P_min, the centre of the primitive pair with the smallest combined
# exponent p (A9 text): the outer region is dominated by that primitive.

@dataclass
class IpbPrim:
    p: float
    w: float            # N_a N_b |c_a c_b| K_ab * pure factor
    F: np.ndarray       # radial polynomial coefficients F_k
    off: float          # |P_ab - C_uv|, the (A10) shift
    s_tot: float = 0.0  # w * S^ab_0, the primitive's TOTAL absolute mass
    v_tot: float = 0.0  # w * V^ab_0, its R=0 maximal potential


@dataclass
class IpbPair:
    center: np.ndarray  # C_uv = P_min
    prims: list


class BoundsIPB:
    """The integral partition bound, in both its batch-independent and its
    distance-dependent form.

    ``V(pair, 0.0)``       -- Eq (15) of sn-LinK: batch-independent, no distance.
    ``distance(pair, D)``  -- (*) of the pre-registration: min(V_0, S_0/D + V_D).
    """
    name = "ipb"

    def __init__(self, mol: gto.Mole):
        self.mol = mol
        self.nsh = mol.nbas
        sh = collect_shells(mol)
        self.pairs = {}
        self._v0 = {}
        for s1 in range(self.nsh):
            for s2 in range(s1 + 1):
                pr = self._build(sh[s1], sh[s2])
                self.pairs[(s1, s2)] = pr
                self._v0[(s1, s2)] = self._V(pr, 0.0)

    @staticmethod
    def _build(sa, sb) -> IpbPair:
        la, lb = sa["l"], sb["l"]
        A, B = sa["center"], sb["center"]
        ab2 = float((A - B) @ (A - B))
        sfac = sa["sfac"] * sb["sfac"]
        prims, centers, pmin, cmin = [], [], math.inf, None
        for a, ca in zip(sa["exps"], sa["coefs"]):
            ca_n = abs(ca) * prim_norm(a, la)
            for b, cb in zip(sb["exps"], sb["coefs"]):
                cb_n = abs(cb) * prim_norm(b, lb)
                p = a + b
                P = (a * A + b * B) / p
                # C_uv = P_min is chosen over ALL primitive pairs, including any
                # whose weight underflows to zero -- the paper's justification is
                # that the smallest-p primitive dominates the OUTER region, which
                # is a statement about the exponent, not the coefficient.
                if p < pmin:
                    pmin, cmin = p, P
                w = ca_n * cb_n * math.exp(-(a * b / p) * ab2) * sfac
                if w == 0.0:
                    continue
                dA = float(np.linalg.norm(P - A))
                dB = float(np.linalg.norm(P - B))
                F = np.zeros(la + lb + 1)
                for t in range(la + 1):
                    ct = math.comb(la, t) * dA ** (la - t)
                    for u in range(lb + 1):
                        F[t + u] += ct * math.comb(lb, u) * dB ** (lb - u)
                prims.append(IpbPrim(p=p, w=w, F=F, off=0.0))
                centers.append(P)
        for pr, P in zip(prims, centers):
            pr.off = float(np.linalg.norm(P - cmin))    # Eq (A10)'s |P_ab - C_uv|
            pr.s_tot = pr.w * BoundsIPB._prim_moment(pr, 0.0, 3.0)
            pr.v_tot = pr.w * BoundsIPB._prim_moment(pr, 0.0, 2.0)
        return IpbPair(center=cmin, prims=prims)

    @staticmethod
    def _rab(pr: IpbPrim, R: float) -> float:
        """Eq (A10): R_ab = max(0, R - |P_ab - C_uv|)."""
        if MUTATION == "rab_unclamped":
            # DELIBERATELY BROKEN: a negative radius makes Gamma(s, p R^2) use
            # R^2 > 0 again, silently SHRINKING the tail for near pairs.
            return R - pr.off
        return max(0.0, R - pr.off)

    def _S(self, pair: IpbPair, R: float) -> float:
        """Absolute tail overlap S_R -- Eq (3.1) / (A14)."""
        return ROUNDING_SLACK * sum(
            pr.w * self._prim_moment(pr, pr.p * self._rab(pr, R) ** 2, 3.0)
            for pr in pair.prims)

    def _V(self, pair: IpbPair, R: float) -> float:
        """Maximal tail potential V_R -- Eq (3.2) / (A16)."""
        return ROUNDING_SLACK * sum(
            pr.w * self._prim_moment(pr, pr.p * self._rab(pr, R) ** 2, 2.0)
            for pr in pair.prims)

    # ---- the two public bounds ------------------------------------------
    def flat(self, s1, s2) -> float:
        """Bound 2: batch-independent IPB = V_0.  sn-LinK Eq (15)."""
        if s1 < s2:
            s1, s2 = s2, s1
        return self._v0[(s1, s2)]

    def distance(self, s1, s2, centre, radius) -> float:
        """Bound 3: the distance-dependent IPB, Eq (*) of the pre-registration.

        THE REFERENCE-POINT SUBTLETY, found by anchor A0 and worth stating in
        full because it is the same class of error the previous attempt made.

        Newton's shell theorem (A15) is a statement about a function that is
        spherically symmetric ABOUT A PARTICULAR POINT.  Each primitive pair's
        radial majorant is spherically symmetric about its OWN centre ``P_ab``,
        not about the contracted pair centre ``C_uv = P_min``.  So the split must
        be taken per primitive, about ``P_ab``, at that primitive's own probe
        distance ``D_ab = |probe - P_ab|``::

            pot_ab  <=  S^ab_0 / D_ab  +  V^ab_{D_ab}

        Both terms are non-increasing in ``D_ab``, so replacing ``D_ab`` by any
        rigorous LOWER bound stays valid.  Eq (A10) supplies exactly that::

            D_ab  =  |probe - P_ab|  >=  |probe - C_uv| - |P_ab - C_uv|
                  >=  R_ab  =  max(0, D - off_ab)

        A first version used ``R_ab`` in ``V`` (correct) but the UNSHIFTED ``D``
        in the ``S_0/D`` inside term.  That mixes two reference points: it charges
        the inside term the full distance to ``C_uv`` while the sphere it is
        collapsing is centred ``off_ab`` closer to the probe.  A0 caught it as
        288 violations on water/cc-pVDZ, all two-centre s-s, max ratio 1.014 --
        a 1.4% underestimate, small enough to be mistaken for rounding and large
        enough to drop significant integrals.  The fix is to use ``R_ab`` in BOTH
        terms, which is what the loop below does.
        """
        if s1 < s2:
            s1, s2 = s2, s1
        key = (s1, s2)
        v0 = self._v0[key]
        pair = self.pairs[key]
        d = centre - pair.center
        D = math.sqrt(float(d @ d)) - radius
        if D <= 0.0:
            return v0
        tot = 0.0
        for pr in pair.prims:
            rab = self._rab(pr, D)
            if rab <= 0.0:
                # The probe may lie inside this primitive's own sphere: no
                # distance decay is available for it, fall back to its R=0
                # potential (which is a valid bound at every probe point).
                tot += pr.v_tot
                continue
            # inside: ALL the primitive's mass, collapsed onto P_ab by Newton,
            # at the rigorous minimum distance R_ab.  The bound on the inside
            # mass is the TOTAL mass -- not the tail mass beyond R_ab, which
            # would under-count by exactly the charge Newton is collapsing.
            inside = pr.s_tot / rab
            if MUTATION == "drop_inside_term":
                # DELIBERATELY BROKEN: exactly the previous attempt's error --
                # the tail potential alone bounds only the charge OUTSIDE the
                # ball, silently dropping the inside charge (most of it).
                inside = 0.0
            # + V^ab_{R_ab}: the tail's own maximal potential, Eq (A16).
            tot += inside + pr.w * self._prim_moment(pr, pr.p * rab * rab, 2.0)
        tot *= ROUNDING_SLACK
        if MUTATION == "no_min_with_flat":
            # DELIBERATELY BROKEN: drop the min, so A1a/A1b can fail.
            return tot
        return min(v0, tot)

    @staticmethod
    def _prim_moment(pr: IpbPrim, x: float, power: float) -> float:
        """``sum_k F_k * 2 pi * Gamma((power+k)/2, x) / p^{(power+k)/2}``, the
        radial moment of one primitive pair's majorant outside ``|r-P_ab|^2 =
        x/p``, without the weight ``w``.

        ``power=3`` -> the tail MASS ``S_R`` (A14);  ``power=2`` -> the tail
        POTENTIAL ``V_R`` (A16).  ``x = 0`` gives the corresponding R=0 total,
        since ``Gamma(s, 0) = Gamma(s)``.

        Both closed forms were verified against an independent 4000-node
        Gauss-Legendre radial quadrature of the same majorant to <= 5e-11
        relative, for (la,lb) up to (3,4) and R up to 3.0 Bohr.
        """
        acc = 0.0
        for k, Fk in enumerate(pr.F):
            if Fk == 0.0:
                continue
            s = (power + k) / 2.0
            acc += Fk * 2.0 * math.pi * upper_gamma(s, x) / pr.p ** s
        return acc

    def degenerate(self, s1, s2, centre, radius) -> bool:
        """True when (*) selects the distance-free branch V_0 -- the same
        disease ferric's sphere bound has on 45-80% of decisions (§6.2b)."""
        if s1 < s2:
            s1, s2 = s2, s1
        return self.distance(s1, s2, centre, radius) >= self._v0[(s1, s2)]


# ===========================================================================
# The three screens (one injectable bound each; the WEIGHT is held fixed)
# ===========================================================================
#
# Holding the density/AO weight fixed at ferric's `max(fmax[s1], fmax[s2])` is
# what isolates the BOUND, which is the question asked.  §6.2a already measured
# the screening STRUCTURE with the bound held fixed and found it worth ~0 pp;
# this study is the mirror experiment.

class ScreenBase:
    def __init__(self, t: float):
        self.t = t

    def keep(self, s1, s2, centre, radius, fmax) -> bool:
        return self.est(s1, s2, centre, radius) * max(fmax[s1], fmax[s2]) >= self.t


class ScreenFerric(ScreenBase):
    name = "ferric"

    def __init__(self, t, bnd_fe, bnd_ipb):
        super().__init__(t)
        self.b = bnd_fe

    def est(self, s1, s2, centre, radius):
        return self.b.sphere(s1, s2, centre, radius)


class ScreenIpbFlat(ScreenBase):
    name = "ipb-flat"

    def __init__(self, t, bnd_fe, bnd_ipb):
        super().__init__(t)
        self.b = bnd_ipb

    def est(self, s1, s2, centre, radius):
        return self.b.flat(s1, s2)


class ScreenIpbDist(ScreenBase):
    name = "ipb-dist"

    def __init__(self, t, bnd_fe, bnd_ipb):
        super().__init__(t)
        self.b = bnd_ipb

    def est(self, s1, s2, centre, radius):
        return self.b.distance(s1, s2, centre, radius)


class ScreenBest(ScreenBase):
    """min of ferric's and IPB-D: both are valid, so their min is valid.  This
    is what a production implementation would actually do, and it separates
    "IPB-D is better" from "IPB-D adds something ferric does not have"."""
    name = "min(ferric,ipb-dist)"

    def __init__(self, t, bnd_fe, bnd_ipb):
        super().__init__(t)
        self.fe, self.ipb = bnd_fe, bnd_ipb

    def est(self, s1, s2, centre, radius):
        return min(self.fe.sphere(s1, s2, centre, radius),
                   self.ipb.distance(s1, s2, centre, radius))


class ScreenNone:
    name = "unscreened"

    def keep(self, *a, **k):
        return True

    def est(self, *a, **k):
        return math.inf


# ===========================================================================
# Grid, X, batches  (identical to snlink_proto.py; see its docstring for the
# negative-weight trap that forces prune=None)
# ===========================================================================

def build_grid(mol: gto.Mole, atom_grid=(50, 110)):
    g = dft.gen_grid.Grids(mol)
    g.prune = None
    g.atom_grid = {a: tuple(atom_grid)
                   for a in {mol.atom_symbol(i) for i in range(mol.natm)}}
    g.build()
    if np.any(g.weights < 0.0):
        raise AssertionError(
            f"grid has {(g.weights < 0).sum()} negative weights; X = sqrt(w) chi "
            "is undefined and abs() silently corrupts the quadrature")
    return g.coords, g.weights


def batch_sphere(pts: np.ndarray):
    c = pts.mean(axis=0)
    return c, float(np.max(np.linalg.norm(pts - c, axis=1)))


@dataclass
class ShellInfo:
    ao_slice: list
    ncart: list


def shell_info(mol: gto.Mole) -> ShellInfo:
    loc = mol.ao_loc_nr()
    return ShellInfo(ao_slice=[(int(loc[s]), int(loc[s + 1])) for s in range(mol.nbas)],
                     ncart=[int(loc[s + 1] - loc[s]) for s in range(mol.nbas)])


def shell_max(mat, info) -> np.ndarray:
    return np.array([np.max(np.abs(mat[a:b])) if b > a else 0.0
                     for a, b in info.ao_slice])


# ===========================================================================
# The K build
# ===========================================================================

@dataclass
class BuildResult:
    K: np.ndarray
    kept: int = 0
    total: int = 0
    kept_weighted: float = 0.0
    total_weighted: float = 0.0
    kept_set: set = field(default_factory=set)


def build_k(mol, coords, weights, D, screen, info, record_set=False) -> BuildResult:
    nbf, nsh, npts = mol.nao, mol.nbas, coords.shape[0]
    Ktilde = np.zeros((nbf, nbf))
    res = BuildResult(K=np.zeros((nbf, nbf)))
    for b0 in range(0, npts, BATCH_POINTS):
        b1 = min(b0 + BATCH_POINTS, npts)
        pts, w, nb = coords[b0:b1], weights[b0:b1], b1 - b0
        X = (mol.eval_gto("GTOval", pts) * np.sqrt(w)[:, None]).T
        F = D @ X
        fmax = shell_max(F, info)
        centre, radius = batch_sphere(pts)
        A = mol.intor("int1e_grids", grids=pts)
        G = np.zeros((nbf, nb))
        for s1 in range(nsh):
            a0, a1 = info.ao_slice[s1]
            for s2 in range(s1 + 1):
                c0, c1 = info.ao_slice[s2]
                unit_w = info.ncart[s1] * info.ncart[s2] * nb
                res.total += 1
                res.total_weighted += unit_w
                if not screen.keep(s1, s2, centre, radius, fmax):
                    continue
                res.kept += 1
                res.kept_weighted += unit_w
                if record_set:
                    res.kept_set.add((b0, s1, s2))
                blk = A[:, a0:a1, c0:c1]
                G[a0:a1] += np.einsum("gmn,ng->mg", blk, F[c0:c1], optimize=True)
                if MUTATION == "drop_mirror":
                    continue    # DELIBERATELY BROKEN: A1a must fail
                if s1 != s2:
                    G[c0:c1] += np.einsum("gmn,mg->ng", blk, F[a0:a1], optimize=True)
        Ktilde += X @ G.T
    res.K = 0.5 * (Ktilde + Ktilde.T)
    return res


def analytic_k(mol, D) -> np.ndarray:
    return np.einsum("ls,mlsn->mn", D, mol.intor("int2e"), optimize=True)


# ===========================================================================
# Systems
# ===========================================================================

SYSTEMS = {
    "water": ("O 0.0000 0.0000 0.1173; H 0.0000 0.7572 -0.4692; "
              "H 0.0000 -0.7572 -0.4692"),
    "methane": ("C 0.0000 0.0000 0.0000; H 0.6276 0.6276 0.6276; "
                "H 0.6276 -0.6276 -0.6276; H -0.6276 0.6276 -0.6276; "
                "H -0.6276 -0.6276 0.6276"),
    "ethane": ("C 0.0000 0.0000 0.7680; C 0.0000 0.0000 -0.7680; "
               "H 0.0000 1.0192 1.1573; H -0.8825 -0.5096 1.1573; "
               "H 0.8825 -0.5096 1.1573; H 0.0000 -1.0192 -1.1573; "
               "H 0.8825 0.5096 -1.1573; H -0.8825 0.5096 -1.1573"),
}

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _atom_spec(name: str) -> str:
    if name in SYSTEMS:
        return SYSTEMS[name]
    path = os.path.join(_HERE, "testdata", "molecules", f"{name}.xyz")
    if not os.path.exists(path):
        raise KeyError(f"unknown system {name!r} (not in SYSTEMS and no {path})")
    lines = open(path).read().splitlines()[2:]
    return "; ".join(ln.strip() for ln in lines if ln.strip())


def molecular_diameter(mol) -> float:
    c = mol.atom_coords()
    return float(np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1).max())


def prepare_system(name, basis, atom_grid=(50, 110), do_scf=True):
    mol = gto.M(atom=_atom_spec(name), basis=basis, unit="Angstrom", verbose=0)
    D = None
    if do_scf:
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        mf.kernel()
        D = mf.make_rdm1()
    coords, weights = build_grid(mol, atom_grid=atom_grid)
    return mol, D, coords, weights, BoundsFerric(mol), BoundsIPB(mol), shell_info(mol)


# ===========================================================================
# Anchors
# ===========================================================================

class Anchors:
    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail):
        self.rows.append((name, bool(ok), detail))
        return ok

    def report(self) -> bool:
        print()
        print(f"{'anchor':<58} {'':<5} detail")
        print("-" * 132)
        allok = True
        for name, ok, detail in self.rows:
            allok &= ok
            print(f"{name:<58} {'PASS' if ok else 'FAIL':<5} {detail}")
        print("-" * 132)
        return allok


def probe_set(mol, seed=0, n_cloud=60, n_far=20):
    """>= 200 probes: every nucleus (T=0), 1e-4 and 1e-8 Bohr off each nucleus,
    a cloud through the molecular volume, and shells at 50/100/200 Bohr."""
    rng = np.random.default_rng(seed)
    nuc = mol.atom_coords()
    def offset(eps):
        d = rng.normal(size=nuc.shape)
        d /= np.linalg.norm(d, axis=1)[:, None]
        return nuc + eps * d
    def shell(r, n):
        d = rng.normal(size=(n, 3))
        return d / np.linalg.norm(d, axis=1)[:, None] * r
    return np.vstack([
        nuc,                                  # exactly on-nucleus, T = 0
        offset(1e-4), offset(1e-8),           # the near-singular regime
        rng.normal(0.0, 3.0, (n_cloud, 3)),   # through the molecular volume
        rng.normal(0.0, 20.0, (n_far, 3)),    # near field / mid field
        shell(50.0, 20), shell(100.0, 20), shell(200.0, 20),   # far field
    ])


def anchor_a0(mol, bnd_fe, bnd_ipb, info, anchors, label):
    """A0 -- NO bound may UNDERestimate.  The only anchor that can invalidate
    the whole construction (whitepaper §4, layer 1: an underestimating bound
    cost 1.06 Ha of K error).  Truth is PySCF's int1e_grids.

    Same-centre pairs are reported SEPARATELY: that is where ferric's original
    signed-overlap bound failed catastrophically, so a pooled statistic could
    hide the exact failure mode this anchor exists to catch.
    """
    pts = probe_set(mol)
    A = mol.intor("int1e_grids", grids=pts)
    ctr = np.array([mol.bas_coord(s) for s in range(mol.nbas)])
    stats = {}   # (bound, same_centre) -> [viol, ratios]
    for g, p in enumerate(pts):
        for s1 in range(mol.nbas):
            a0, a1 = info.ao_slice[s1]
            for s2 in range(s1 + 1):
                c0, c1 = info.ao_slice[s2]
                true = float(np.max(np.abs(A[g, a0:a1, c0:c1])))
                same = bool(np.allclose(ctr[s1], ctr[s2]))
                for nm, bd in (("ferric", bnd_fe.sphere(s1, s2, p, 0.0)),
                               ("ipb-flat", bnd_ipb.flat(s1, s2)),
                               ("ipb-dist", bnd_ipb.distance(s1, s2, p, 0.0))):
                    st = stats.setdefault((nm, same), [0, []])
                    if true > bd * (1.0 + 1e-12):
                        st[0] += 1
                    if true > 0.0 and bd > 0.0:
                        st[1].append(true / bd)
    npair = mol.nbas * (mol.nbas + 1) // 2
    for nm in ("ferric", "ipb-flat", "ipb-dist"):
        for same in (False, True):
            if (nm, same) not in stats:
                continue
            viol, rt = stats[(nm, same)]
            r = np.array(rt) if rt else np.array([0.0])
            tag = "same-centre" if same else "two-centre"
            anchors.check(
                f"A0 {nm} never underestimates [{label}, {tag}]",
                viol == 0,
                f"violations={viol}/{len(rt)}; true/bound "
                f"p50={np.median(r):.2e} p90={np.percentile(r, 90):.2e} "
                f"max={r.max():.2e}",
            )
    return pts, A


def anchor_trivial_limit(bnd_ipb, mol, anchors, label):
    """A1 -- (*) must reduce EXACTLY to the batch-independent IPB at D = 0, and
    must never EXCEED it at any distance (the `min` guarantees both)."""
    worst_eq, worst_le = 0.0, -math.inf
    ctr = np.array([mol.bas_coord(s) for s in range(mol.nbas)])
    for s1 in range(mol.nbas):
        for s2 in range(s1 + 1):
            f = bnd_ipb.flat(s1, s2)
            c = bnd_ipb.pairs[(s1, s2)].center
            # D = 0 exactly: probe AT the pair centre with zero radius.
            d0 = bnd_ipb.distance(s1, s2, c, 0.0)
            worst_eq = max(worst_eq, abs(d0 - f) / max(f, 1e-300))
            for R in (0.1, 1.0, 5.0, 25.0, 100.0):
                dd = bnd_ipb.distance(s1, s2, c + np.array([R, 0.0, 0.0]), 0.0)
                worst_le = max(worst_le, dd / max(f, 1e-300) - 1.0)
    anchors.check(
        f"A1a trivial limit: IPB-D(D=0) == IPB-flat [{label}]",
        worst_eq == 0.0,
        f"max relative difference = {worst_eq:.3e} (bar: exactly 0)",
    )
    anchors.check(
        f"A1b IPB-D never exceeds IPB-flat [{label}]",
        worst_le <= 0.0,
        f"max (IPB-D / IPB-flat - 1) over D in {{0.1..100}} = {worst_le:.3e}",
    )


def anchor_non_inert(bnd_ipb, bnd_fe, mol, coords, weights, info, anchors, label):
    """A2 -- the distance factor must BITE (repo rule, whitepaper §8).

    Reports the fraction of real (pair, batch) decisions on which IPB-D
    degenerates to IPB-flat, alongside ferric's own degenerate fraction on the
    SAME decisions.  Pre-registered INERT criterion: if IPB-D lands in ferric's
    45-80% band, the lane closes.
    """
    deg_ipb = deg_fe = tot = 0
    gains = []
    for b0 in range(0, coords.shape[0], BATCH_POINTS):
        pts = coords[b0:b0 + BATCH_POINTS]
        centre, radius = batch_sphere(pts)
        for s1 in range(mol.nbas):
            for s2 in range(s1 + 1):
                tot += 1
                if bnd_ipb.degenerate(s1, s2, centre, radius):
                    deg_ipb += 1
                if bnd_fe.degenerate(s1, s2, centre, radius):
                    deg_fe += 1
                f = bnd_ipb.flat(s1, s2)
                d = bnd_ipb.distance(s1, s2, centre, radius)
                if f > 0.0:
                    gains.append(f / d if d > 0 else math.inf)
    g = np.array(gains)
    frac = deg_ipb / tot
    anchors.check(
        f"A2 non-inert: IPB-D's distance factor bites [{label}]",
        frac < 0.45,
        f"IPB-D degenerate on {100*frac:.2f}% of {tot} (pair,batch) decisions "
        f"(ferric: {100*deg_fe/tot:.2f}%); tightening IPB-flat/IPB-D "
        f"p50={np.median(g):.3g}x p90={np.percentile(g,90):.3g}x max={g.max():.3g}x",
    )
    return frac, deg_fe / tot, g


# ===========================================================================
# Driver
# ===========================================================================

SCREENS = {"ferric": ScreenFerric, "ipb-flat": ScreenIpbFlat,
           "ipb-dist": ScreenIpbDist, "min(ferric,ipb-dist)": ScreenBest}


def run(system, basis, atom_grid, anchors, thresholds, do_sweep=True):
    print(f"\n{'='*132}")
    print(f"SYSTEM {system}/{basis}  grid {atom_grid[0]}x{atom_grid[1]} (unpruned)")
    print("=" * 132)
    mol, D, coords, weights, bfe, bipb, info = prepare_system(system, basis, atom_grid)
    nb = math.ceil(len(weights) / BATCH_POINTS)
    print(f"  nbf={mol.nao} nsh={mol.nbas} npts={len(weights)} batches={nb} "
          f"pairs={mol.nbas*(mol.nbas+1)//2} diameter={molecular_diameter(mol):.2f} Bohr")

    anchor_a0(mol, bfe, bipb, info, anchors, f"{system}/{basis}")
    anchor_trivial_limit(bipb, mol, anchors, f"{system}/{basis}")
    anchor_non_inert(bipb, bfe, mol, coords, weights, info, anchors, f"{system}/{basis}")

    ref = build_k(mol, coords, weights, D, ScreenNone(), info)
    K_an = analytic_k(mol, D)
    e_grid = float(np.max(np.abs(ref.K - K_an)))
    scale = float(np.max(np.abs(K_an)))
    print(f"  grid error max|K_grid - K_analytic| = {e_grid:.3e} (max|K| = {scale:.3e})")
    anchors.check(
        f"A3 unscreened K reproduces analytic K [{system}/{basis}]",
        e_grid <= 1e-3 * scale,
        f"E_grid={e_grid:.2e} vs 1e-3*max|K|={1e-3*scale:.2e} "
        f"(guards the accumulation, not just screen-vs-unscreened)",
    )
    for nm, cls in SCREENS.items():
        r = build_k(mol, coords, weights, D, cls(0.0, bfe, bipb), info)
        err = float(np.max(np.abs(r.K - ref.K)))
        anchors.check(
            f"A4 threshold-0 trivial limit [{system}/{basis}] {nm}",
            err <= 1e-14 and r.kept == r.total,
            f"max|dK|={err:.2e}, kept={r.kept}/{r.total}",
        )

    if not do_sweep:
        return

    # ---- threshold sweep: kept work AND K error, per bound ----------------
    print(f"\n  {'bound':<22} {'thresh':>8} {'kept %':>9} {'kept-wt %':>10} "
          f"{'K error':>11}   (E_grid = {e_grid:.2e})")
    table = {}
    for nm, cls in SCREENS.items():
        table[nm] = []
        for t in thresholds:
            r = build_k(mol, coords, weights, D, cls(t, bfe, bipb), info)
            err = float(np.max(np.abs(r.K - ref.K)))
            kp = 100.0 * r.kept / r.total
            kw = 100.0 * r.kept_weighted / r.total_weighted
            table[nm].append((t, kp, kw, err))
            print(f"  {nm:<22} {t:8.0e} {kp:9.3f} {kw:10.3f} {err:11.3e}")
    return table


def matched_error_report(table, targets):
    """Kept work at MATCHED K ERROR, not matched threshold.

    §6.2a's recorded trap: a threshold cuts on `bound x weight`, so equal
    thresholds are NOT equal operating points -- sn-LinK looked 10.8 pp better
    at equal threshold while buying it with 2.6x the K error.  Here each bound's
    kept work is interpolated (log-log) to the same measured K error.
    """
    print(f"\n  kept-work % at MATCHED K error (log-interpolated over threshold)")
    hdr = "  " + f"{'K error':>10}" + "".join(f"{n:>24}" for n in table)
    print(hdr)
    for tgt in targets:
        row = f"  {tgt:10.0e}"
        for nm, rows in table.items():
            errs = np.array([r[3] for r in rows])
            kws = np.array([r[2] for r in rows])
            ok = errs > 0
            if ok.sum() < 2 or tgt < errs[ok].min() or tgt > errs[ok].max():
                row += f"{'--':>24}"
                continue
            o = np.argsort(errs[ok])
            v = np.interp(math.log(tgt), np.log(errs[ok][o]), kws[ok][o])
            row += f"{v:24.3f}"
        print(row)


def size_axis(names, basis, atom_grid, thresholds):
    """Kept-work COUNTS vs system size -- deterministic, load-immune."""
    print(f"\n{'='*132}")
    print(f"SIZE AXIS  basis={basis}  grid={atom_grid}  (kept-work %, weighted)")
    print("=" * 132)
    print(f"  {'system':<12} {'diam':>7} {'nsh':>5} {'thresh':>8} "
          + "".join(f"{n:>24}" for n in SCREENS)
          + f"{'deg ipb':>9}{'deg fe':>9}")
    for name in names:
        mol, D, coords, weights, bfe, bipb, info = prepare_system(
            name, basis, atom_grid, do_scf=True)
        dia = molecular_diameter(mol)
        dfrac, ffrac, _ = _degeneracy(mol, coords, bfe, bipb)
        for t in thresholds:
            row = f"  {name:<12} {dia:7.2f} {mol.nbas:5d} {t:8.0e}"
            for nm, cls in SCREENS.items():
                r = build_k(mol, coords, weights, D, cls(t, bfe, bipb), info)
                row += f"{100.0*r.kept_weighted/r.total_weighted:24.3f}"
            row += f"{100*dfrac:9.2f}{100*ffrac:9.2f}"
            print(row, flush=True)


def _degeneracy(mol, coords, bfe, bipb):
    di = df = tot = 0
    gains = []
    for b0 in range(0, coords.shape[0], BATCH_POINTS):
        centre, radius = batch_sphere(coords[b0:b0 + BATCH_POINTS])
        for s1 in range(mol.nbas):
            for s2 in range(s1 + 1):
                tot += 1
                di += bipb.degenerate(s1, s2, centre, radius)
                df += bfe.degenerate(s1, s2, centre, radius)
                f = bipb.flat(s1, s2)
                d = bipb.distance(s1, s2, centre, radius)
                if f > 0 and d > 0:
                    gains.append(f / d)
    return di / tot, df / tot, np.array(gains)


def main():
    global MUTATION
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mutate", default=None,
                    help="run a deliberately broken variant: drop_inside_term, "
                         "no_min_with_flat, rab_unclamped, drop_mirror")
    ap.add_argument("--mode", default="anchors",
                    choices=["anchors", "sweep", "size", "all"])
    ap.add_argument("--systems", default=None)
    ap.add_argument("--basis", default=None)
    args = ap.parse_args()
    MUTATION = args.mutate
    if MUTATION:
        print(f"*** MUTATION ACTIVE: {MUTATION} -- an anchor MUST fail ***")

    anchors = Anchors()
    thresholds = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

    if args.mode in ("anchors", "all"):
        cases = [("water", "ccpvdz", (50, 110)),
                 ("butane", "def2-svp", (35, 110)),
                 ("butane", "def2-qzvp", (25, 50))]
        if args.systems:
            cases = [(s, args.basis or "ccpvdz", (35, 110))
                     for s in args.systems.split(",")]
        for sys_, bas, grid in cases:
            run(sys_, bas, grid, anchors, thresholds, do_sweep=False)

    if args.mode in ("sweep", "all"):
        for sys_, bas, grid in [("water", "ccpvdz", (50, 110)),
                                ("butane", "def2-svp", (35, 110))]:
            tb = run(sys_, bas, grid, anchors, thresholds, do_sweep=True)
            if tb:
                matched_error_report(tb, [1e-5, 1e-6, 1e-7])

    if args.mode in ("size", "all"):
        names = args.systems.split(",") if args.systems else \
            ["methane", "ethane", "alkane_4", "alkane_8", "alkane_12", "alkane_16"]
        size_axis(names, args.basis or "sto-3g", (25, 50), [1e-6, 1e-8])

    ok = anchors.report()
    if MUTATION:
        print(f"\nMUTATION {MUTATION}: anchors {'ALL PASSED (BAD -- the mutation was inert)' if ok else 'FAILED as required'}")
        return 0 if not ok else 1
    print("\nALL ANCHORS PASS" if ok else "\nANCHOR FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
