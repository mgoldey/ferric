#!/usr/bin/env python3
"""Reference implementation of sn-LinK-style screening for seminumerical exchange.

Companion pre-registration: ``scripts/queue/out/snlink_python_prereg.md`` (written
and committed BEFORE this file).  Results: ``scripts/queue/out/snlink_python_results.md``.

THE QUESTION
------------
ferric has COSX in production (``crates/ferric-scf/src/cosx_k.rs``) with a
one-threshold, density-driven per-batch pair screen.  sn-LinK (Laqua, Kussmann &
Ochsenfeld) is the SAME seminumerical core with different screening.  Does
sn-LinK's screening structure keep LESS (shell-pair, grid-batch) work at equal K
accuracy?  This script answers that in deterministic COUNTS, not timings.

WHAT IS SOURCED AND WHAT IS RECONSTRUCTED -- read the prereg's §1.1 before
citing anything here.  Every primary source (JCTC 14, 3451 (2018); JCTC 16, 1456
(2020); JCP 150, 044101 (2019); JCP 138, 134114 (2013)) is paywalled and could
not be read.  What IS established from OA metadata/abstracts and the OA review
(Pure Appl. Chem. 2025, DOI 10.1515/pac-2025-0603, PMC12645584):

  * sn-LinK is "a seminumerical counterpart to the LinK method" -- same core,
    different screening (review, verbatim).
  * The 2018 CPU method "combin[es] the preLinK method [Kussmann & Ochsenfeld,
    JCP 2013, 138, 134114] with explicit screening of integrals for batches of
    grid points to minimize the screening overhead" (abstract, verbatim).
  * preLinK is "a simple but accurate preselection method based on Schwarz
    integral estimates to determine the significant elements of the exact
    exchange matrix before its evaluation" (abstract, verbatim).
  * The 2020 GPU paper adds "our recently developed integral bounds [Thompson &
    Ochsenfeld, JCP 2019, 150, 044101]" (abstract, verbatim).

What is RECONSTRUCTED here, and must not be cited as Laqua's:
  * the two-threshold eps_E / eps_K split and its exact keep rule;
  * the pair-resolved density weight's exact form.
What is deliberately SUBSTITUTED:
  * the integral estimate.  Thompson & Ochsenfeld's integral partition bound is
    unobtainable, so BOTH screens here use ferric's own Hoelder primitive-pair
    bound (ported from ``crates/ferric-integrals/src/cosx_screen.rs``).  Holding
    the bound fixed is what isolates the SCREENING STRUCTURE, which is the
    question asked.  It also means this study cannot say how much of Laqua's win
    comes from a tighter bound -- a scope limit, not a result.

METHOD
------
    X[mu,g]   = sqrt(w_g) chi_mu(r_g)
    F         = D X
    G[nu,g]  += A^g[nu,lam] F[lam,g]      (per kept shell pair, both orderings)
    Ktilde    = X G^T
    K         = 0.5 (Ktilde + Ktilde^T)

A^g comes from PySCF's ``int1e_grids`` (the +1/|r-r_g| convention; see
``scripts/3c1e_spec.md`` §1), sliced per shell pair so the kept/dropped decision
is exactly the decision a Rust kernel would make.

Run ``python3 scripts/snlink_proto.py`` for the anchor table and the kept-work
comparison.  ``--mutate NAME`` runs a deliberately broken variant to prove an
anchor can fail (repo rule: a test you have never seen fail is an assumption).
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

# One thread everywhere: this harness is deliberately CPU-light.
pyscf_lib.num_threads(1)

BATCH_POINTS = 256          # matches ferric's COSX_SUB_BATCH_POINTS
ROUNDING_SLACK = 1.0 + 1e-10   # cosx_screen.rs
COARSE_SLACK = 1.0 + 1e-9      # cosx_screen.rs
FERRIC_DEFAULT_T = 1e-7        # COSX_DEFAULT_SCREEN_THRESH

# Active mutation (set by --mutate); consumed at the few named points below.
MUTATION = None


# ===========================================================================
# The integral bound (a faithful port of crates/ferric-integrals/src/cosx_screen.rs)
# ===========================================================================
#
# For a primitive pair (a at A, b at B):  p = a+b, P = (aA+bB)/p,
# K_AB = exp(-ab/p |A-B|^2), d_A = |P-A| = (b/p)|AB|, d_B = (a/p)|AB|.
#
#   |P_mu(r-A) P_nu(r-B)| <= (rho+d_A)^la (rho+d_B)^lb = sum_k q_k rho^k
#   int e^{-p rho^2}/|r-C|            = (2pi/p) F_0(p R^2)
#   int rho^k e^{-p rho^2}/|r-C|     <= M_k (4pi/p) F_0(p R^2/2),  M_k=(k/(p e))^{k/2}
#   F_0(T) <= min(1, 0.5 sqrt(pi/T))
#
# giving, per primitive pair,
#   beta0 * F0plus(pR^2) + beta1 * F0plus(pR^2/2)
# with F0plus(pR^2) = min(1, g0/R), g0 = 0.5 sqrt(pi/p), g1 = 0.5 sqrt(2pi/p).
#
# Every step is a pointwise inequality, so the bound cannot UNDERESTIMATE --
# which is the one property a screen must have (whitepaper §4, layer 1).

DFACT = [1.0, 1.0, 3.0, 15.0, 105.0]


def prim_norm(a: float, l: int) -> float:
    """N(a,l) = (2a/pi)^{3/4} (4a)^{l/2} / sqrt((2l-1)!!)  -- md3c1e.rs:420."""
    return (2.0 * a / math.pi) ** 0.75 * math.sqrt((4.0 * a) ** l) / math.sqrt(DFACT[l])


def pure_factor(l: int, pure: bool, c2s: np.ndarray | None) -> float:
    """Max absolute column sum of the cart->sph matrix (1 for cart or l<2)."""
    if not pure or l < 2:
        return 1.0
    return float(np.max(np.sum(np.abs(c2s), axis=0)))


@dataclass
class PairBound:
    """Per-shell-pair Hoelder bound data (the `Coarse` struct of cosx_screen.rs)."""
    mid: np.ndarray     # midpoint of A and B
    half: float         # |A-B|/2
    total: float        # sum(beta0+beta1): the bound at R=0, valid everywhere
    sum_bg: float       # sum(beta0*g0 + beta1*g1): far-field numerator


class Bounds:
    """All shell-pair bounds for a molecule, plus the sphere query.

    This is the SINGLE injectable integral estimate: swap this class (e.g. for
    Thompson & Ochsenfeld's integral partition bound, if someone obtains it) and
    both screens change together, which is what keeps their comparison fair.
    """

    def __init__(self, mol: gto.Mole):
        self.mol = mol
        self.nsh = mol.nbas
        # PySCF cart->sph matrices, for the pure-shell column-sum factor.
        c2s = {l: gto.cart2sph(l) for l in range(5)}
        sh = []
        for s in range(self.nsh):
            l = mol.bas_angular(s)
            if l > 4:
                raise ValueError(f"shell {s} has l={l}; bound table stops at l=4")
            exps = mol.bas_exp(s)
            # PySCF stores normalized contraction coefficients; strip the
            # primitive norm the way ferric stores them, then re-apply it, so
            # the product |c| * prim_norm matches ferric's `coefs` exactly.
            # (nprim, nctr).  A generally contracted shell (e.g. the 8-primitive
            # s shell on oxygen in cc-pVDZ, nctr=2) is several basis functions
            # sharing one exponent set; the bound must cover ALL of them, so take
            # the max magnitude over contractions -- that is an upper bound on any
            # single one (see 3c1e_spec.md §5.7 for why reading only the first
            # contraction is the classic silent error here).
            #
            # PySCF's `bas_ctr_coeff` already carries the primitive
            # normalization, whereas ferric stores it separately and applies
            # `prim_norm` at use.  Rather than reason about which convention is
            # which, the whole product is validated end-to-end by anchor A0
            # against true `int1e_grids` blocks: 0 violations with a max
            # true/bound ratio of exactly 1.00 (the tight same-centre s-s
            # on-nucleus case the Rust module documents), which pins both the
            # normalization and the bound.
            coefs = np.max(np.abs(mol.bas_ctr_coeff(s)), axis=1)
            sh.append(dict(
                l=l,
                center=np.asarray(mol.bas_coord(s), dtype=float),
                exps=np.asarray(exps, dtype=float),
                coefs=coefs,
                sfac=pure_factor(l, mol.cart is False, c2s.get(l)),
            ))
        self.sh = sh
        self.pairs: dict[tuple[int, int], PairBound] = {}
        for s1 in range(self.nsh):
            for s2 in range(s1 + 1):
                self.pairs[(s1, s2)] = self._build_pair(sh[s1], sh[s2])

    @staticmethod
    def _build_pair(sa, sb) -> PairBound:
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
                kab = math.exp(-(a * b / p) * ab2)
                w = ca_n * cb_n * kab * sfac
                if w == 0.0:
                    continue
                da = b / p * ab
                db = a / p * ab
                # q_k: coefficients of rho^k in (rho+d_A)^la (rho+d_B)^lb
                qk = np.zeros(la + lb + 1)
                for m in range(la + 1):
                    ca_m = math.comb(la, m) * da ** (la - m)
                    for n in range(lb + 1):
                        qk[m + n] += ca_m * math.comb(lb, n) * db ** (lb - n)
                beta0 = w * (2.0 * math.pi / p) * qk[0]
                s1k = 0.0
                for k in range(1, len(qk)):
                    if qk[k] != 0.0:
                        s1k += qk[k] * (k / (p * math.e)) ** (0.5 * k)
                beta1 = w * (4.0 * math.pi / p) * s1k
                g0 = 0.5 * math.sqrt(math.pi / p)
                g1 = 0.5 * math.sqrt(2.0 * math.pi / p)
                tot += beta0 + beta1
                sum_bg += beta0 * g0 + beta1 * g1
        slack = ROUNDING_SLACK * COARSE_SLACK
        return PairBound(
            mid=0.5 * (sa["center"] + sb["center"]),
            half=0.5 * ab,
            total=tot * slack,
            sum_bg=sum_bg * slack,
        )

    def sphere(self, s1: int, s2: int, centre: np.ndarray, radius: float) -> float:
        """Upper bound on |A^g_block| for EVERY g within `radius` of `centre`.

        `coarse_estimate_sphere` of cosx_screen.rs: R_c = max(0, |M-centre| -
        radius - |AB|/2); every product centre lies on the segment AB so its own
        R >= R_c, and every term is non-increasing in R.
        """
        if s1 < s2:
            s1, s2 = s2, s1
        c = self.pairs[(s1, s2)]
        d = centre - c.mid
        rc = math.sqrt(float(d @ d)) - radius - c.half
        if rc <= 0.0:
            return c.total
        return min(c.total, c.sum_bg / rc)


# ===========================================================================
# Grid, X, batches
# ===========================================================================

def build_grid(mol: gto.Mole, atom_grid=(50, 110)):
    """Becke grid matching ferric's COSX default, with pruning DISABLED.

    HARNESS TRAP, found the hard way and worth stating: PySCF's default
    ``nwchem_prune`` produces grids with NEGATIVE weights (120 of 2328 points at
    level 0, 1036 of 33704 at level 3 for water).  Seminumerical exchange needs
    ``X = sqrt(w) chi``, so a negative weight has no square root; ferric takes
    ``w.abs().sqrt()`` (``cosx_k.rs:469``) with the comment "Becke weights are
    non-negative in practice", which is true of ferric's own grid generator and
    false of PySCF's pruned one.  With ``abs()`` the sign flip corrupts the
    quadrature: water/cc-pVDZ integrates to 10.15 and 10.20 electrons at levels
    2 and 3, and the K grid error goes the WRONG WAY with refinement
    (6.3e-5 at level 1 -> 7.8e-2 at level 2).  A finer grid giving a 1000x worse
    K is a construction signal, not chemistry.

    With ``prune=None`` every weight is non-negative, nelec is exact to 1e-6 or
    better, and the error falls monotonically (6.3e-5 / 3.5e-6 / 2.9e-7 at
    levels 1/2/3).  ``(50,110)`` is ferric's own COSX default and reproduces its
    measured water grid error (~6.3e-5 here; whitepaper §4 quotes 2.4e-5 for
    ferric's grid, same order).  The non-negativity is asserted below so this
    trap cannot come back silently.
    """
    g = dft.gen_grid.Grids(mol)
    g.prune = None
    g.atom_grid = {a: tuple(atom_grid) for a in {mol.atom_symbol(i)
                                                 for i in range(mol.natm)}}
    g.build()
    if np.any(g.weights < 0.0):
        raise AssertionError(
            f"grid has {(g.weights < 0).sum()} negative weights; X = sqrt(w) chi "
            "is undefined and abs() silently corrupts the quadrature")
    return g.coords, g.weights


def batch_sphere(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Centroid and enclosing radius (matches cosx_k.rs's bounding_sphere)."""
    c = pts.mean(axis=0)
    r = float(np.max(np.linalg.norm(pts - c, axis=1)))
    return c, r


# ===========================================================================
# The two screens
# ===========================================================================

@dataclass
class ShellInfo:
    ao_slice: list[tuple[int, int]]   # (start, stop) AO range per shell
    ncart: list[int]                  # cost weight per shell


def shell_info(mol: gto.Mole) -> ShellInfo:
    loc = mol.ao_loc_nr()
    return ShellInfo(
        ao_slice=[(int(loc[s]), int(loc[s + 1])) for s in range(mol.nbas)],
        ncart=[int(loc[s + 1] - loc[s]) for s in range(mol.nbas)],
    )


def shell_max(mat: np.ndarray, info: ShellInfo) -> np.ndarray:
    """max |mat[mu, :]| over the AOs of each shell -- `fmax` / `xmax`."""
    return np.array([np.max(np.abs(mat[a:b])) if b > a else 0.0
                     for a, b in info.ao_slice])


def shell_dmax(D: np.ndarray, info: ShellInfo) -> np.ndarray:
    """dmax[l, s] = max_{mu in l, nu in s} |D_{mu,nu}|.  Once per build, O(nbf^2)."""
    n = len(info.ao_slice)
    out = np.zeros((n, n))
    for i, (a0, a1) in enumerate(info.ao_slice):
        for j, (b0, b1) in enumerate(info.ao_slice):
            out[i, j] = np.max(np.abs(D[a0:a1, b0:b1]))
    return out


class ScreenFerric:
    """ferric's current screen -- cosx_k.rs:1096-1105.

        keep(s1,s2,b) iff bound(s1,s2,sphere_b) * max(fmax[s1], fmax[s2]) >= t

    One threshold; the density enters only through F = D X, collapsed to ONE
    SCALAR PER SHELL.  That collapse is exactly what this study puts under test.
    """
    name = "ferric"

    def __init__(self, t: float):
        self.t = t

    def prepare(self, D, dmax, info):
        pass

    def keep(self, s1, s2, bnd, centre, radius, fmax, xmax, dmax, info) -> bool:
        est = bnd.sphere(s1, s2, centre, radius)
        return est * max(fmax[s1], fmax[s2]) >= self.t


class ScreenSnLink:
    """sn-LinK-style screening: two branches ORed, per grid batch.

        E-branch:  bound(s1,s2,sphere_b) * max(xmax[s1], xmax[s2]) >= eps_E
        K-branch:  bound(s1,s2,sphere_b) * dweight(s1,s2,b)        >= eps_K

        dweight(s1,s2,b) = max over l of ( dmax[l,s2] * xmax[l,b] )   [and s1<->s2]

    The E-branch is DENSITY-FREE: it keeps a pair on the strength of the
    integral times the AO magnitude alone, so a pair whose contribution is large
    but whose density coupling happens to be small in this SCF iteration is
    still evaluated (this is what makes the scheme robust across iterations, and
    it is the branch that fires on small molecules).

    The K-branch is the seminumerical analogue of LinK's density-pair list: it
    asks not "is |F| large in this shell" (a shell scalar, as ferric does) but
    "is there any shell l whose density coupling to s2 AND whose AO magnitude on
    this batch together make the (l, s1, s2, b) contribution significant" -- the
    (bra pair, ket pair) combination the whitepaper's §4.1 identifies as where
    real locality lives.

    Note the structural relationship, which is the reason the two can be
    compared at all:

        fmax[s1,b] = max_g |(D X)[mu in s1, g]| <= max_l ( dmax[s1,l] * xmax[l,b] ) * (AO count factor)

    i.e. ferric's fmax is bounded by SN's dweight up to the sum-vs-max over the
    contracted index.  So SN's K-branch is a LOOSER (more conservative) test
    than ferric's on the same threshold, and any work SN saves must come from
    resolving the pair rather than from a tighter number.  That is a prediction
    this harness measures, not an assumption.
    """
    name = "snlink"

    def __init__(self, eps_e: float, eps_k: float):
        self.eps_e = eps_e
        self.eps_k = eps_k

    def prepare(self, D, dmax, info):
        pass

    def keep(self, s1, s2, bnd, centre, radius, fmax, xmax, dmax, info) -> bool:
        est = bnd.sphere(s1, s2, centre, radius)
        # E-branch: density-free integral estimate.
        if est * max(xmax[s1], xmax[s2]) >= self.eps_e:
            return True
        # K-branch: pair-resolved density weight.
        if MUTATION == "snlink_shell_scalar_density":
            # DELIBERATELY BROKEN (mutation): collapse the density to a shell
            # scalar, i.e. reduce SN's K-branch to ferric's.  Used as the A4
            # positive control.
            w = max(fmax[s1], fmax[s2])
        else:
            w = max(float(np.max(dmax[:, s1] * xmax)),
                    float(np.max(dmax[:, s2] * xmax)))
        return est * w >= self.eps_k


class ScreenNone:
    """The trivial limit: keep everything.  A1's reference."""
    name = "unscreened"

    def prepare(self, D, dmax, info):
        pass

    def keep(self, *a, **k) -> bool:
        return True


# ===========================================================================
# The K build
# ===========================================================================

@dataclass
class BuildResult:
    K: np.ndarray
    kept: int = 0            # (shell pair, batch) units reaching the kernel
    total: int = 0           # the same, unscreened
    kept_weighted: float = 0.0    # weighted by ncart(s1)*ncart(s2)*len(batch)
    total_weighted: float = 0.0
    kept_set: set = field(default_factory=set)   # {(batch_idx, s1, s2)} for A4


def build_k(mol, coords, weights, D, bnd, screen, info, record_set=False) -> BuildResult:
    """Seminumerical K with `screen` deciding each (shell pair, batch).

    Structure follows cosx_k.rs: per batch, X and F are formed for the batch,
    then every shell pair is offered to the screen; survivors accumulate into G
    in BOTH orderings (the mirror fold), and Ktilde = X G^T is accumulated
    across batches.  K = 1/2 (Ktilde + Ktilde^T).
    """
    nbf = mol.nao
    nsh = mol.nbas
    npts = coords.shape[0]
    Ktilde = np.zeros((nbf, nbf))
    res = BuildResult(K=np.zeros((nbf, nbf)))
    dmax = shell_dmax(D, info)
    screen.prepare(D, dmax, info)

    for b0 in range(0, npts, BATCH_POINTS):
        b1 = min(b0 + BATCH_POINTS, npts)
        pts = coords[b0:b1]
        w = weights[b0:b1]
        nb = b1 - b0
        ao = mol.eval_gto("GTOval", pts)            # (nb, nbf)
        X = (ao * np.sqrt(w)[:, None]).T             # (nbf, nb)
        # NO abs(): build_grid() asserts w >= 0, because abs() would silently
        # flip the sign of a point's contribution (see build_grid's docstring).
        F = D @ X                                    # (nbf, nb)
        fmax = shell_max(F, info)
        xmax = shell_max(X, info)
        centre, radius = batch_sphere(pts)
        A = mol.intor("int1e_grids", grids=pts)      # (nb, nbf, nbf), +1/|r-r_g|
        G = np.zeros((nbf, nb))

        for s1 in range(nsh):
            a0, a1 = info.ao_slice[s1]
            for s2 in range(s1 + 1):
                c0, c1 = info.ao_slice[s2]
                unit_w = info.ncart[s1] * info.ncart[s2] * nb
                res.total += 1
                res.total_weighted += unit_w
                if not screen.keep(s1, s2, bnd, centre, radius, fmax, xmax, dmax, info):
                    continue
                res.kept += 1
                res.kept_weighted += unit_w
                if record_set:
                    res.kept_set.add((b0, s1, s2))
                blk = A[:, a0:a1, c0:c1]             # (nb, na, nc)
                # G[s1] += A[s1,s2] F[s2];  and the mirror G[s2] += A[s2,s1] F[s1].
                # A is symmetric in (mu,nu) so A[s2,s1] = A[s1,s2]^T.
                G[a0:a1] += np.einsum("gmn,ng->mg", blk, F[c0:c1], optimize=True)
                if MUTATION == "drop_mirror":
                    continue   # DELIBERATELY BROKEN: A1 must fail
                if s1 != s2:
                    G[c0:c1] += np.einsum("gmn,mg->ng", blk, F[a0:a1], optimize=True)

        if MUTATION == "counter_only":
            # DELIBERATELY BROKEN: skip the accumulation for every 4th batch
            # without touching the counter.  K is then wrong while the kept-work
            # counts are unchanged, which is what proves the counter and the
            # matrix are INDEPENDENTLY observable (A5) -- a counter that moved
            # with every K error could not distinguish a screening decision from
            # an accumulation bug.
            if (b0 // BATCH_POINTS) % 4 == 0:
                continue
        Ktilde += X @ G.T

    res.K = 0.5 * (Ktilde + Ktilde.T)
    return res


def analytic_k(mol, D) -> np.ndarray:
    """Exact K from four-centre ERIs: K_{mu,nu} = sum D_{lam,sig} (mu lam|sig nu)."""
    eri = mol.intor("int2e")
    return np.einsum("ls,mlsn->mn", D, eri, optimize=True)


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


def _atom_spec(name: str) -> str:
    """Geometry for `name`: a key of SYSTEMS, or `alkane_N` from testdata."""
    if name in SYSTEMS:
        return SYSTEMS[name]
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "testdata", "molecules", f"{name}.xyz")
    if not os.path.exists(path):
        raise KeyError(f"unknown system {name!r} (not in SYSTEMS and no {path})")
    lines = open(path).read().splitlines()[2:]      # skip the xyz count + comment
    return "; ".join(ln.strip() for ln in lines if ln.strip())


def molecular_diameter(mol: gto.Mole) -> float:
    """Max internuclear distance, Bohr -- the size axis the whitepaper uses."""
    c = mol.atom_coords()
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
    return float(d.max())


def prepare_system(name: str, basis: str, atom_grid=(50, 110)):
    mol = gto.M(atom=_atom_spec(name), basis=basis, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    D = mf.make_rdm1()
    coords, weights = build_grid(mol, atom_grid=atom_grid)
    return mol, D, coords, weights, Bounds(mol), shell_info(mol)


# ===========================================================================
# Anchors
# ===========================================================================

def anchor_bound_never_underestimates(mol, bnd, info, anchors, label):
    """A0 -- the bound must never UNDERestimate (whitepaper §4, layer 1).

    Both screens multiply this bound by a density/AO weight, so an
    underestimating bound silently drops significant pairs and NO downstream
    anchor is meaningful.  This is checked directly against true `int1e_grids`
    blocks on a probe set that includes ON-NUCLEUS points (where the bound is
    tight to rounding for same-centre s-s pairs) and far probes.

    The tightness ratio (true/bound) is REPORTED, never asserted -- per the
    Rust module's own convention.
    """
    rng = np.random.default_rng(0)
    pts = np.vstack([rng.normal(0.0, 3.0, (60, 3)),
                     mol.atom_coords(),                   # on-nucleus, T=0
                     rng.normal(0.0, 20.0, (20, 3))])     # far field
    A = mol.intor("int1e_grids", grids=pts)
    viol = 0
    ratios = []
    for g, p in enumerate(pts):
        for s1 in range(mol.nbas):
            a0, a1 = info.ao_slice[s1]
            for s2 in range(s1 + 1):
                c0, c1 = info.ao_slice[s2]
                true = float(np.max(np.abs(A[g, a0:a1, c0:c1])))
                bd = bnd.sphere(s1, s2, p, 0.0)
                if true > bd * (1.0 + 1e-9):
                    viol += 1
                if true > 0.0:
                    ratios.append(true / bd if bd > 0 else math.inf)
    r = np.array(ratios)
    anchors.check(
        f"A0 bound never underestimates [{label}]",
        viol == 0,
        f"violations={viol}/{len(pts)*mol.nbas*(mol.nbas+1)//2}, "
        f"tightness true/bound: median {np.median(r):.2e} max {r.max():.2e}",
    )


class Anchors:
    def __init__(self):
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str):
        self.rows.append((name, bool(ok), detail))
        return ok

    def report(self) -> bool:
        print()
        print(f"{'anchor':<46} {'':<5} detail")
        print("-" * 110)
        allok = True
        for name, ok, detail in self.rows:
            allok &= ok
            print(f"{name:<46} {'PASS' if ok else 'FAIL':<5} {detail}")
        print("-" * 110)
        return allok


def run(system: str, basis: str, atom_grid, thresholds, anchors: Anchors,
        sweep: bool = True):
    print(f"\n{'='*110}")
    print(f"SYSTEM {system}/{basis}  grid {atom_grid[0]}x{atom_grid[1]} (unpruned)")
    print("=" * 110)
    mol, D, coords, weights, bnd, info = prepare_system(system, basis, atom_grid)
    print(f"  nbf={mol.nao}  nsh={mol.nbas}  npts={len(weights)}  "
          f"batches={math.ceil(len(weights)/BATCH_POINTS)}  "
          f"pairs={mol.nbas*(mol.nbas+1)//2}")

    # ---- A0: the bound must never underestimate ---------------------------
    anchor_bound_never_underestimates(mol, bnd, info, anchors, system)

    # ---- unscreened reference, and the grid error -------------------------
    ref = build_k(mol, coords, weights, D, bnd, ScreenNone(), info)
    K_an = analytic_k(mol, D)
    e_grid = float(np.max(np.abs(ref.K - K_an)))
    print(f"  grid error  max|K_grid - K_analytic| = {e_grid:.3e}   "
          f"(max|K| = {np.max(np.abs(K_an)):.3e})")

    # ---- A1: trivial limit -------------------------------------------------
    #
    # NOTE, learned by mutation: comparing screened-vs-unscreened is NOT enough.
    # A defect in the ACCUMULATION (a dropped mirror term, a skipped batch)
    # breaks BOTH sides equally, so the difference stays 0.00 and A1 passes on a
    # K that is wrong by 1.35.  A1 therefore also requires the unscreened K to
    # reproduce the analytic K to within the grid error scale -- that is the
    # part the mutations `drop_mirror` and `counter_only` actually break.
    scale = float(np.max(np.abs(K_an)))
    anchors.check(
        f"A1a unscreened K reproduces analytic K to grid accuracy [{system}]",
        e_grid <= 1e-3 * scale,
        f"E_grid={e_grid:.2e} vs 1e-3*max|K|={1e-3*scale:.2e} "
        f"(guards the accumulation itself, not just screen-vs-unscreened)",
    )
    for scr, label in ((ScreenFerric(0.0), "ferric t=0"),
                       (ScreenSnLink(0.0, 0.0), "snlink eps=0")):
        r = build_k(mol, coords, weights, D, bnd, scr, info)
        err = float(np.max(np.abs(r.K - ref.K)))
        anchors.check(
            f"A1b trivial limit [{system}] {label}",
            err <= 1e-14 and r.kept == r.total,
            f"max|dK|={err:.2e} (bar 1e-14), kept={r.kept}/{r.total}",
        )

    # ---- production thresholds --------------------------------------------
    t = thresholds["ferric_t"]
    ee, ek = thresholds["eps_e"], thresholds["eps_k"]
    fe = build_k(mol, coords, weights, D, bnd, ScreenFerric(t), info, record_set=True)
    sn = build_k(mol, coords, weights, D, bnd, ScreenSnLink(ee, ek), info, record_set=True)

    e_fe = float(np.max(np.abs(fe.K - ref.K)))
    e_sn = float(np.max(np.abs(sn.K - ref.K)))

    # ---- A2: correctness below the grid error ------------------------------
    #
    # REACHABILITY of A2 itself: when a screen drops nothing, E_scr is exactly
    # 0.00 and A2 passes without measuring anything.  That is reported as
    # VACUOUS rather than PASS, so a green row can never be mistaken for
    # evidence (repo rule: a test you have never seen fail is an assumption; a
    # test that CANNOT fail is worse).
    for lbl, r, e in ((f"ferric t={t:g}", fe, e_fe), (f"snlink eps={ee:g}", sn, e_sn)):
        if r.kept == r.total:
            anchors.check(
                f"A2 screen error <= 0.1*grid error [{system}] {lbl}",
                True,
                f"VACUOUS -- screen dropped nothing, so E_scr=0 by construction; "
                f"E_grid={e_grid:.2e}. Not evidence.",
            )
            continue
        anchors.check(
            f"A2 screen error <= 0.1*grid error [{system}] {lbl}",
            e <= 0.1 * e_grid,
            f"E_scr={e:.2e}  E_grid={e_grid:.2e}  ratio={e/e_grid if e_grid else float('nan'):.2e}",
        )

    # ---- A3: reachability --------------------------------------------------
    for lbl, r in ((f"ferric t={t:g}", fe), (f"snlink eps={ee:g}", sn)):
        anchors.check(
            f"A3 reachability: screen drops something [{system}] {lbl}",
            r.kept < r.total,
            f"kept={r.kept}/{r.total} ({100.0*r.kept/r.total:.2f}%), "
            f"weighted {100.0*r.kept_weighted/r.total_weighted:.2f}%",
        )

    # ---- A4: the screens actually differ, and the difference-counter works --
    only_fe = fe.kept_set - sn.kept_set
    only_sn = sn.kept_set - fe.kept_set
    print(f"  kept-set symmetric difference: |FE only|={len(only_fe)}  "
          f"|SN only|={len(only_sn)}")

    # Positive control on the difference-counter itself.  Disable the E-branch
    # (eps_E = +inf, NOT 0 -- at 0 the branch fires unconditionally and the
    # control silently measures nothing; that error was caught by this anchor
    # failing on ethane) and collapse the K-branch's density weight to ferric's
    # shell scalar.  SN must then reproduce FE's kept set EXACTLY, which is what
    # proves a later nonzero symmetric difference is structural rather than a
    # bookkeeping artifact.
    global MUTATION
    saved = MUTATION
    MUTATION = "snlink_shell_scalar_density"
    sn_as_fe = build_k(mol, coords, weights, D, bnd, ScreenSnLink(math.inf, t), info,
                       record_set=True)
    MUTATION = saved
    anchors.check(
        f"A4 positive control: SN(eps_E=inf, dweight->fmax) == FE [{system}]",
        sn_as_fe.kept_set == fe.kept_set,
        f"symmetric difference = {len(sn_as_fe.kept_set ^ fe.kept_set)} "
        f"(kept {sn_as_fe.kept}/{sn_as_fe.total} vs FE {fe.kept}/{fe.total})",
    )

    # ---- density branch: does it ever fire beyond the geometric one? -------
    # SN with eps_K = +inf is the E-branch alone (density-free).
    e_only = build_k(mol, coords, weights, D, bnd, ScreenSnLink(ee, math.inf), info,
                     record_set=True)
    k_only = build_k(mol, coords, weights, D, bnd, ScreenSnLink(math.inf, ek), info,
                     record_set=True)
    added_by_k = len(sn.kept_set - e_only.kept_set)
    print(f"  SN branch decomposition: E-branch alone keeps {e_only.kept}/{e_only.total}, "
          f"K-branch alone keeps {k_only.kept}/{k_only.total}, "
          f"K-branch adds {added_by_k} pair-batches the E-branch would have dropped")

    # ---- the kept-work table ----------------------------------------------
    print()
    print(f"  {'configuration':<28} {'kept/total':>16} {'kept %':>9} "
          f"{'weighted %':>11} {'max|dK|':>11} {'vs E_grid':>11}")
    for lbl, r, e in (
        ("unscreened", ref, 0.0),
        (f"ferric  t={t:g}", fe, e_fe),
        (f"sn-LinK eps={ee:g}", sn, e_sn),
        (f"  sn-LinK E-branch only", e_only, float(np.max(np.abs(e_only.K - ref.K)))),
        (f"  sn-LinK K-branch only", k_only, float(np.max(np.abs(k_only.K - ref.K)))),
    ):
        print(f"  {lbl:<28} {r.kept:>7}/{r.total:<8} "
              f"{100.0*r.kept/r.total:>8.2f}% "
              f"{100.0*r.kept_weighted/r.total_weighted:>10.2f}% "
              f"{e:>11.2e} {(e/e_grid if e_grid else float('nan')):>11.2e}")

    # ---- threshold sweep (reported, for the rescaling check of §3) ----------
    if sweep:
        print()
        print(f"  threshold sweep (kept %, max|dK|):")
        print(f"  {'thresh':>9}  {'ferric kept%':>13} {'ferric dK':>11}  "
              f"{'snlink kept%':>13} {'snlink dK':>11}")
        for th in (1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-11):
            rf = build_k(mol, coords, weights, D, bnd, ScreenFerric(th), info)
            rs = build_k(mol, coords, weights, D, bnd, ScreenSnLink(th, th), info)
            ef = float(np.max(np.abs(rf.K - ref.K)))
            es = float(np.max(np.abs(rs.K - ref.K)))
            print(f"  {th:>9.0e}  {100.0*rf.kept/rf.total:>12.2f}% {ef:>11.2e}  "
                  f"{100.0*rs.kept/rs.total:>12.2f}% {es:>11.2e}")

    return dict(system=system, nbf=mol.nao, e_grid=e_grid, fe=fe, sn=sn, ref=ref,
                e_fe=e_fe, e_sn=e_sn, added_by_k=added_by_k,
                only_fe=len(only_fe), only_sn=len(only_sn))


def diagnostics(system: str, basis: str, atom_grid):
    """Why the screens do or do not bite: the distribution of the screening
    products, and how often the sphere bound degenerates.

    This is what turns "kept 100%" from an uninformative null into a
    quantitative statement: the MINIMUM screening product over all pair-batches
    is how large the threshold would have to be to drop even one, and the
    fraction of pair-batches with `R_c <= 0` says whether the geometric decay
    is available at this molecule/grid size at all.
    """
    mol, D, coords, weights, bnd, info = prepare_system(system, basis, atom_grid)
    dmax = shell_dmax(D, info)
    fes, sne, snk, bds, radii, nfall, ntot = [], [], [], [], [], 0, 0
    for b0 in range(0, len(weights), BATCH_POINTS):
        b1 = min(b0 + BATCH_POINTS, len(weights))
        pts = coords[b0:b1]
        ao = mol.eval_gto("GTOval", pts)
        X = (ao * np.sqrt(weights[b0:b1])[:, None]).T
        F = D @ X
        fmax = shell_max(F, info)
        xmax = shell_max(X, info)
        c, r = batch_sphere(pts)
        radii.append(r)
        for s1 in range(mol.nbas):
            for s2 in range(s1 + 1):
                est = bnd.sphere(s1, s2, c, r)
                pb = bnd.pairs[(s1, s2)]
                ntot += 1
                if float(np.linalg.norm(c - pb.mid)) - r - pb.half <= 0.0:
                    nfall += 1
                bds.append(est)
                fes.append(est * max(fmax[s1], fmax[s2]))
                sne.append(est * max(xmax[s1], xmax[s2]))
                snk.append(est * max(float(np.max(dmax[:, s1] * xmax)),
                                     float(np.max(dmax[:, s2] * xmax))))
    fes, sne, snk, bds = map(np.array, (fes, sne, snk, bds))
    radii = np.array(radii)
    ratio = snk / fes
    print(f"\n  DIAGNOSTICS [{system}/{basis}]")
    print(f"    batch radius: median {np.median(radii):.2f}  max {radii.max():.2f} Bohr")
    print(f"    sphere bound degenerates to its R=0 value (R_c<=0) for "
          f"{100.0*nfall/ntot:.1f}% of pair-batches; min bound = {bds.min():.3e}")
    print(f"    MIN screening product over all {len(fes)} pair-batches:")
    print(f"       ferric  bound*fmax    = {fes.min():.3e}   "
          f"({fes.min()/FERRIC_DEFAULT_T:.0f}x above the 1e-7 default)")
    print(f"       sn-LinK bound*dweight = {snk.min():.3e}   "
          f"({snk.min()/FERRIC_DEFAULT_T:.0f}x above)")
    print(f"       sn-LinK bound*xmax    = {sne.min():.3e}   (E-branch; the only one that bites)")
    print(f"    SN-K / FE product ratio: min {ratio.min():.3f} median {np.median(ratio):.3f} "
          f"max {ratio.max():.3f}  (spread {ratio.max()/ratio.min():.1f}x)")
    print(f"       SN tighter than FE on {100.0*(ratio<1).mean():.1f}% of pair-batches.")
    print(f"       A spread of ~1.0 would mean SN is merely a THRESHOLD RESCALING of FE;")
    print(f"       a large spread means the two orderings genuinely differ (the §3 artifact check).")


def size_sweep(systems, basis, atom_grid, thresholds=(1e-5, 1e-6, 1e-7, 1e-8)):
    """Kept-work vs MOLECULAR DIAMETER -- the axis the locality question lives on.

    Counts only (no K accumulation), so this reaches sizes the full anchor table
    cannot.  For each system it reports, at each threshold, what ferric's screen
    keeps, what the sn-LinK-style screen keeps, and -- the question that matters
    for the whitepaper's §4.1 -- what the DENSITY-weighted branch keeps ON ITS
    OWN versus what the density-free geometric branch keeps on its own.

    If the density branch tracks the geometric one, the density is contributing
    nothing that geometry has not already contributed, which is the fourth
    independent sighting of the same effect (COSX's row mask, LinK's
    density-pair list, the #52 threshold sweep being the first three).
    """
    print(f"\n{'='*118}")
    print(f"SIZE SWEEP  basis={basis}  grid={atom_grid[0]}x{atom_grid[1]}  "
          f"(counts only -- no K accumulation)")
    print("=" * 118)
    hdr = (f"{'system':>10} {'diam':>7} {'nbf':>5} {'pairxb':>8} {'thresh':>8} | "
           f"{'FE kept%':>9} {'SN kept%':>9} {'SN-FE pp':>9} | "
           f"{'geom-only%':>10} {'dens-only%':>10} {'dens adds':>9}")
    print(hdr)
    print("-" * 118)
    rows = []
    for name in systems:
        mol, D, coords, weights, bnd, info = prepare_system(name, basis, atom_grid)
        diam = molecular_diameter(mol)
        dmax = shell_dmax(D, info)
        fe, sE, sK, geom, wt = [], [], [], [], []
        for b0 in range(0, len(weights), BATCH_POINTS):
            b1 = min(b0 + BATCH_POINTS, len(weights))
            pts = coords[b0:b1]
            nb = b1 - b0
            ao = mol.eval_gto("GTOval", pts)
            X = (ao * np.sqrt(weights[b0:b1])[:, None]).T
            F = D @ X
            fmax = shell_max(F, info)
            xmax = shell_max(X, info)
            c, r = batch_sphere(pts)
            for s1 in range(mol.nbas):
                for s2 in range(s1 + 1):
                    est = bnd.sphere(s1, s2, c, r)
                    fe.append(est * max(fmax[s1], fmax[s2]))
                    sE.append(est * max(xmax[s1], xmax[s2]))
                    sK.append(est * max(float(np.max(dmax[:, s1] * xmax)),
                                        float(np.max(dmax[:, s2] * xmax))))
                    # The purely GEOMETRIC estimate: the integral bound alone,
                    # with no density and no AO magnitude.  This is the control
                    # that says whether the density factors contribute anything.
                    geom.append(est)
                    wt.append(info.ncart[s1] * info.ncart[s2] * nb)
        fe, sE, sK, geom, wt = map(np.array, (fe, sE, sK, geom, wt))
        for th in thresholds:
            kfe = fe >= th
            ksn = (sE >= th) | (sK >= th)
            kg = geom >= th        # geometry alone
            kd = sK >= th          # density-weighted branch alone
            # Does the density branch drop anything the geometric bound keeps?
            dens_adds = int(np.sum(kg & ~kd))
            print(f"{name:>10} {diam:7.2f} {mol.nao:5d} {len(fe):8d} {th:8.0e} | "
                  f"{100*kfe.mean():8.3f}% {100*ksn.mean():8.3f}% "
                  f"{100*(ksn.mean()-kfe.mean()):+8.3f} | "
                  f"{100*kg.mean():9.3f}% {100*kd.mean():9.3f}% {dens_adds:9d}")
            rows.append((name, diam, mol.nao, th, kfe.mean(), ksn.mean(),
                         kg.mean(), kd.mean(), dens_adds))
        print("-" * 118)
    print("\n  'geom-only%'  = kept by the integral bound ALONE (no density, no AO magnitude)")
    print("  'dens-only%'  = kept by the density-weighted branch alone")
    print("  'dens adds'   = pair-batches the GEOMETRIC bound keeps but the DENSITY branch drops")
    print("                  -> this is the density's entire contribution. 0 means vacuous.")
    return rows


def density_decomposition(systems, basis, atom_grid, t=FERRIC_DEFAULT_T):
    """How much of the pruning is GEOMETRY and how much is the DENSITY MATRIX?

    This is the whitepaper §4.1 question asked at pair-batch granularity, and it
    needs care because BOTH screens' "density" factors secretly contain the AO
    values on the grid:

        ferric:  fmax[s]  = max |(D X)[s]|          <- contains X
        sn-LinK: dweight  = max_l dmax[l,s] xmax[l] <- contains X

    So a screen that "uses the density" may in fact be riding on the Gaussian
    decay of X away from the batch, which is pure geometry.  The decomposition
    replaces D by a CONSTANT matrix of the same magnitude, leaving X untouched:

        geom       = bound alone                     (no D, no X)
        +AO(X)     = bound * xmax                    (X only)
        flatD      = bound * dweight with D -> const (X only, dweight-shaped)
        +realD     = bound * dweight with real D     (X and D)

    `flatD -> +realD` is then the density's TRUE contribution, with the AO
    factor held fixed, and `geom -> flatD` is what the AO magnitude contributes
    through the same expression.
    """
    print(f"\n{'='*112}")
    print(f"DENSITY vs GEOMETRY decomposition  basis={basis}  t={t:.0e}  "
          f"grid={atom_grid[0]}x{atom_grid[1]}")
    print("=" * 112)
    print(f"{'system':>10} {'diam':>6} {'nbf':>5} | {'geom':>8} {'+AO(X)':>8} "
          f"{'flatD':>8} {'+realD':>8} | {'X does':>7} {'D does':>7} {'D share':>8}")
    print("-" * 112)
    for name in systems:
        mol, D, coords, weights, bnd, info = prepare_system(name, basis, atom_grid)
        diam = molecular_diameter(mol)
        dmax = shell_dmax(D, info)
        dmax_flat = shell_dmax(np.ones_like(D) * float(np.max(np.abs(D))), info)
        est_l, sK_l, sKf_l, xo_l = [], [], [], []
        for b0 in range(0, len(weights), BATCH_POINTS):
            b1 = min(b0 + BATCH_POINTS, len(weights))
            pts = coords[b0:b1]
            ao = mol.eval_gto("GTOval", pts)
            X = (ao * np.sqrt(weights[b0:b1])[:, None]).T
            xmax = shell_max(X, info)
            c, r = batch_sphere(pts)
            for s1 in range(mol.nbas):
                for s2 in range(s1 + 1):
                    est = bnd.sphere(s1, s2, c, r)
                    est_l.append(est)
                    xo_l.append(est * max(xmax[s1], xmax[s2]))
                    sK_l.append(est * max(float(np.max(dmax[:, s1] * xmax)),
                                          float(np.max(dmax[:, s2] * xmax))))
                    sKf_l.append(est * max(float(np.max(dmax_flat[:, s1] * xmax)),
                                           float(np.max(dmax_flat[:, s2] * xmax))))
        est, sK, sKf, xo = map(np.array, (est_l, sK_l, sKf_l, xo_l))
        g = 100 * (est >= t).mean()
        ax = 100 * (xo >= t).mean()
        f = 100 * (sKf >= t).mean()
        rd = 100 * (sK >= t).mean()
        xdoes, ddoes = f - g, rd - f
        tot = abs(xdoes) + abs(ddoes)
        print(f"{name:>10} {diam:6.1f} {mol.nao:5d} | {g:7.2f}% {ax:7.2f}% "
              f"{f:7.2f}% {rd:7.2f}% | {xdoes:+7.2f} {ddoes:+7.2f} "
              f"{100*abs(ddoes)/tot if tot else 0.0:7.1f}%")
    print("-" * 112)
    print("  'X does'  = pp pruned by the AO magnitude through dweight (geometry)")
    print("  'D does'  = pp pruned by the DENSITY MATRIX with the AO factor held fixed")
    print("  'D share' = |D does| / (|X does| + |D does|): the density's share of the pruning")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="water,methane")
    ap.add_argument("--basis", default="cc-pvdz")
    ap.add_argument("--grid", default="50,110",
                    help="radial,angular atom grid (ferric's COSX default is 50,110)")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--decompose", action="store_true",
                    help="split the pruning into its geometry and density-matrix "
                         "parts (the whitepaper 4.1 question at pair-batch granularity)")
    ap.add_argument("--size-sweep", action="store_true",
                    help="kept-work vs molecular diameter across the given systems "
                         "(counts only; reaches sizes the anchor table cannot)")
    ap.add_argument("--diagnostics", action="store_true",
                    help="report the screening-product distribution instead of "
                         "the anchor table (explains WHY a screen does/does not bite)")
    ap.add_argument("--mutate", default=None,
                    help="run a deliberately broken variant: drop_mirror | "
                         "counter_only | strict_gt | bound_underestimate")
    args = ap.parse_args()

    global MUTATION
    MUTATION = args.mutate
    if MUTATION:
        print(f"*** MUTATION ACTIVE: {MUTATION} -- anchors are EXPECTED to fail ***")

    if MUTATION == "bound_underestimate":
        # DELIBERATELY BROKEN: divide the bound by 1e6 so it UNDERestimates.
        orig = Bounds.sphere
        Bounds.sphere = lambda self, s1, s2, c, r: orig(self, s1, s2, c, r) * 1e-6
    if MUTATION == "strict_gt":
        # DELIBERATELY BROKEN: `>` instead of `>=`, so threshold 0 drops any
        # pair whose bound underflows to exactly 0.
        ScreenFerric.keep = (lambda self, s1, s2, bnd, c, r, fmax, xmax, dmax, info:
                             bnd.sphere(s1, s2, c, r) * max(fmax[s1], fmax[s2]) > self.t)
        ScreenSnLink.keep = (lambda self, s1, s2, bnd, c, r, fmax, xmax, dmax, info:
                             bnd.sphere(s1, s2, c, r) * max(xmax[s1], xmax[s2]) > self.eps_e)

    atom_grid = tuple(int(x) for x in args.grid.split(","))
    if args.decompose:
        density_decomposition([x.strip() for x in args.systems.split(",")],
                              args.basis, atom_grid)
        return 0
    if args.size_sweep:
        size_sweep([x.strip() for x in args.systems.split(",")], args.basis, atom_grid)
        return 0
    if args.diagnostics:
        for s_ in args.systems.split(","):
            diagnostics(s_.strip(), args.basis, atom_grid)
        return 0
    thresholds = dict(ferric_t=FERRIC_DEFAULT_T, eps_e=FERRIC_DEFAULT_T,
                      eps_k=FERRIC_DEFAULT_T)
    anchors = Anchors()
    for s in args.systems.split(","):
        run(s.strip(), args.basis, atom_grid, thresholds, anchors,
            sweep=not args.no_sweep)

    ok = anchors.report()
    if MUTATION:
        print(f"\nMutation `{MUTATION}`: "
              f"{'ANCHORS STILL PASSED -- the guard proves nothing' if ok else 'anchors failed as required'}")
        return 0 if not ok else 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
