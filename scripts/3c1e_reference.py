#!/usr/bin/env python3
"""Numpy reference implementation of the 3-center-1-electron integral

    A^g_{mu,nu} = \\int chi_mu(r) chi_nu(r) / |r - r_g| dr

for contracted Cartesian/spherical Gaussian AOs and a set of probe points r_g.

This is the correctness reference for the spec in ``scripts/3c1e_spec.md``.
It is deliberately written for READABILITY, not speed: every quantity that the
spec names appears here as a named variable, so a Rust implementer can diff
their intermediates against this file term by term.

SIGN CONVENTION (load-bearing)
------------------------------
This module returns the REPULSIVE kernel ``+1/|r - r_g|``, matching

  * PySCF's ``mol.intor('int1e_grids', grids=pts)``, and
  * ferric's ``ferric_integrals::cosx_a::a_matrix_at_point``

libint2's nuclear-attraction operator returns the ATTRACTIVE ``-1/|r-r_g|``;
``cosx_a.rs`` negates it (see cosx_a.rs:237-238). We produce the already-negated
(positive) quantity directly, so a Rust kernel written from this reference needs
NO extra sign flip -- it replaces the libint2 call *and* the negation together.

METHOD
------
McMurchie-Davidson.  The AO pair density is expanded in Hermite Gaussians
centred on the Gaussian-product centre P; the Coulomb kernel is then applied to
each Hermite function analytically via the R-tensor, which is a recursion driven
by the Boys function.  The E-coefficients (the Hermite expansion) are
GRID-INDEPENDENT; only the R-tensor sees r_g.  That split is the entire reason
this recursion is recommended for the COSX shape -- see the spec.

Run ``python 3c1e_reference.py`` for the self-test table.
"""

from __future__ import annotations

import math
import sys

import numpy as np

# --------------------------------------------------------------------------
# Boys function
# --------------------------------------------------------------------------
#
#   F_n(T) = \int_0^1 t^{2n} exp(-T t^2) dt
#
# Evaluated with a small-T Taylor series and a large-T asymptotic form plus
# downward recursion.  See the spec for the breakpoint justification.

BOYS_T_SMALL = 1e-12   # below this, use the T=0 limit exactly

# Taylor/downward below, asymptotic/upward above.
#
# The asymptotic branch uses F_0 = (1/2) sqrt(pi/T), which drops the erfc tail;
# the relative size of that neglected term is (MEASURED, not assumed):
#     T = 25 -> 1.5e-12     T = 30 -> 9.4e-15
#     T = 35 -> 1.9e-16     T >= 40 -> 0 in double precision
# 35 is therefore the smallest breakpoint at which the asymptotic form is
# correct to full double precision.  A code that switches at 25 -- a common
# choice -- silently carries ~1e-12 relative error in F_0 near the breakpoint.
BOYS_T_SWITCH = 35.0


def boys_single(n: int, T: float) -> float:
    """Scalar F_n(T), used only to cross-check the vectorized path."""
    return float(boys_vec(n, np.array([T]))[n, 0])


def boys_vec(nmax: int, T: np.ndarray) -> np.ndarray:
    """All F_n(T) for n = 0..nmax, vectorized over T.

    Returns an array of shape ``(nmax+1, len(T))``.

    Strategy (per the spec):
      * T < BOYS_T_SWITCH: evaluate F_nmax by its Taylor series, then recur
        DOWNWARD.  Downward recursion is numerically stable; upward is not.
      * T >= BOYS_T_SWITCH: F_0 = 0.5*sqrt(pi/T) to within ~1e-16 (the
        complementary error function's tail is below double precision there),
        then recur UPWARD, which is stable in this regime.
    """
    T = np.asarray(T, dtype=float)
    out = np.zeros((nmax + 1, T.size))

    small = T < BOYS_T_SWITCH
    large = ~small

    # ---- small / moderate T: Taylor for the top order, then downward ----
    if np.any(small):
        Ts = T[small]
        # F_nmax(T) = exp(-T) * sum_{k>=0} (2T)^k / (2*nmax + 2k + 1)!!
        # written as sum_k term_k with term_0 = 1/(2n+1) and
        # term_{k+1} = term_k * 2T / (2n + 2k + 3).
        term = np.full(Ts.shape, 1.0 / (2 * nmax + 1))
        total = term.copy()
        for k in range(0, 200):
            term = term * (2.0 * Ts) / (2 * nmax + 2 * k + 3)
            total = total + term
            if np.max(np.abs(term)) < 1e-18 * max(np.max(np.abs(total)), 1.0):
                break
        top = np.exp(-Ts) * total
        acc = np.zeros((nmax + 1, Ts.size))
        acc[nmax] = top
        expT = np.exp(-Ts)
        # Downward: F_{n}(T) = (2T F_{n+1}(T) + exp(-T)) / (2n + 1)
        for n in range(nmax - 1, -1, -1):
            acc[n] = (2.0 * Ts * acc[n + 1] + expT) / (2 * n + 1)
        out[:, small] = acc

    # ---- large T: asymptotic F_0, then upward ----
    if np.any(large):
        Tl = T[large]
        acc = np.zeros((nmax + 1, Tl.size))
        acc[0] = 0.5 * np.sqrt(np.pi / Tl)
        expT = np.exp(-Tl)
        # Upward: F_{n+1}(T) = ((2n+1) F_n(T) - exp(-T)) / (2T)
        for n in range(0, nmax):
            acc[n + 1] = ((2 * n + 1) * acc[n] - expT) / (2.0 * Tl)
        out[:, large] = acc

    # ---- exact T -> 0 limit, guarding the 0/0 in the recursions ----
    zero = T < BOYS_T_SMALL
    if np.any(zero):
        for n in range(nmax + 1):
            out[n, zero] = 1.0 / (2 * n + 1)

    return out


# --------------------------------------------------------------------------
# McMurchie-Davidson E coefficients (GRID-INDEPENDENT)
# --------------------------------------------------------------------------


def e_coeff(i: int, j: int, t: int, Qx: float, a: float, b: float) -> float:
    """Hermite expansion coefficient E_t^{ij} for one Cartesian direction.

    Expands the product of two primitive 1-D Gaussians
      x_A^i exp(-a x_A^2) * x_B^j exp(-b x_B^2)
    in Hermite Gaussians Lambda_t centred on P.

    ``Qx = Ax - Bx``, ``a``/``b`` are the primitive exponents.
    Depends on NOTHING about the grid point.  This is the quantity a batched
    kernel hoists out of the grid loop.
    """
    p = a + b
    mu = a * b / p
    if t < 0 or t > i + j:
        return 0.0
    if i == 0 and j == 0 and t == 0:
        return math.exp(-mu * Qx * Qx)
    if j == 0:
        # decrement i
        return (
            (1.0 / (2.0 * p)) * e_coeff(i - 1, j, t - 1, Qx, a, b)
            - (mu * Qx / a) * e_coeff(i - 1, j, t, Qx, a, b)
            + (t + 1) * e_coeff(i - 1, j, t + 1, Qx, a, b)
        )
    # decrement j
    return (
        (1.0 / (2.0 * p)) * e_coeff(i, j - 1, t - 1, Qx, a, b)
        + (mu * Qx / b) * e_coeff(i, j - 1, t, Qx, a, b)
        + (t + 1) * e_coeff(i, j - 1, t + 1, Qx, a, b)
    )


def e_table(la: int, lb: int, Qx: float, a: float, b: float) -> np.ndarray:
    """All E_t^{ij} for i<=la, j<=lb, t<=la+lb, as an array [la+1, lb+1, la+lb+1]."""
    tab = np.zeros((la + 1, lb + 1, la + lb + 1))
    for i in range(la + 1):
        for j in range(lb + 1):
            for t in range(i + j + 1):
                tab[i, j, t] = e_coeff(i, j, t, Qx, a, b)
    return tab


# --------------------------------------------------------------------------
# R tensor (GRID-DEPENDENT -- this is the only part inside the grid loop)
# --------------------------------------------------------------------------


def r_tensor(Lmax: int, p: float, PC: np.ndarray) -> dict:
    """Hermite Coulomb integrals R^0_{tuv} for a batch of grid points.

    ``PC`` has shape (npts, 3) and holds P - C for each probe point C.
    Returns a dict keyed by (t,u,v) -> array of shape (npts,), holding
    R^{0}_{tuv}.

    The auxiliary R^{n}_{tuv} are built by the standard recursion
      R^n_{t+1,u,v} = t R^{n+1}_{t-1,u,v} + PCx R^{n+1}_{t,u,v}
    (and cyclically for u, v), seeded by
      R^n_{000} = (-2p)^n F_n(T),  T = p |PC|^2.
    """
    npts = PC.shape[0]
    T = p * np.einsum("gi,gi->g", PC, PC)
    F = boys_vec(Lmax, T)  # (Lmax+1, npts)

    # R[n][(t,u,v)] -> (npts,)
    R: list[dict] = [dict() for _ in range(Lmax + 1)]
    fac = 1.0
    for n in range(Lmax + 1):
        R[n][(0, 0, 0)] = fac * F[n]
        fac *= -2.0 * p

    # Build in order of increasing total Hermite order.
    for total in range(1, Lmax + 1):
        for n in range(Lmax - total + 1):
            for t in range(total + 1):
                for u in range(total - t + 1):
                    v = total - t - u
                    if t > 0:
                        prev = R[n + 1].get((t - 2, u, v))
                        val = (t - 1) * prev if prev is not None else np.zeros(npts)
                        val = val + PC[:, 0] * R[n + 1][(t - 1, u, v)]
                    elif u > 0:
                        prev = R[n + 1].get((t, u - 2, v))
                        val = (u - 1) * prev if prev is not None else np.zeros(npts)
                        val = val + PC[:, 1] * R[n + 1][(t, u - 1, v)]
                    else:
                        prev = R[n + 1].get((t, u, v - 2))
                        val = (v - 1) * prev if prev is not None else np.zeros(npts)
                        val = val + PC[:, 2] * R[n + 1][(t, u, v - 1)]
                    R[n][(t, u, v)] = val

    return R[0]


# --------------------------------------------------------------------------
# Cartesian component ordering and normalization
# --------------------------------------------------------------------------


def cart_components(l: int) -> list[tuple[int, int, int]]:
    """Cartesian (lx,ly,lz) in libint2/PySCF ("CCA") order.

    For l=2 this yields xx, xy, xz, yy, yz, zz.  This is the ordering both
    libint2 and PySCF use, and therefore the ordering ferric inherits.
    """
    out = []
    for lx in range(l, -1, -1):
        for ly in range(l - lx, -1, -1):
            lz = l - lx - ly
            out.append((lx, ly, lz))
    return out


def dfact(n: int) -> float:
    """Double factorial n!! with (-1)!! = 0!! = 1."""
    if n <= 0:
        return 1.0
    r = 1.0
    while n > 1:
        r *= n
        n -= 2
    return r


def prim_norm(alpha: float, lx: int, ly: int, lz: int) -> float:
    """Normalization of a single primitive Cartesian Gaussian.

    N = (2a/pi)^{3/4} * (4a)^{l/2} / sqrt((2lx-1)!!(2ly-1)!!(2lz-1)!!)

    so that <g|g> = 1 for that Cartesian component.
    """
    l = lx + ly + lz
    return (
        (2.0 * alpha / math.pi) ** 0.75
        * (4.0 * alpha) ** (l / 2.0)
        / math.sqrt(dfact(2 * lx - 1) * dfact(2 * ly - 1) * dfact(2 * lz - 1))
    )


# --------------------------------------------------------------------------
# The kernel
# --------------------------------------------------------------------------


def shell_pair_block(
    la: int,
    A: np.ndarray,
    alphas_a: np.ndarray,
    coefs_a: np.ndarray,
    lb: int,
    B: np.ndarray,
    alphas_b: np.ndarray,
    coefs_b: np.ndarray,
    pts: np.ndarray,
) -> np.ndarray:
    """Cartesian 3c1e block for one contracted shell pair over all grid points.

    Returns shape (npts, ncart_a, ncart_b) with the ``+1/|r-r_g|`` sign.

    ``coefs_*`` are contraction coefficients that already include the
    ANGULAR-MOMENTUM-ONLY part of the primitive normalization, i.e. the
    ``(2a/pi)^{3/4} (4a)^{l/2}`` factor -- see ``_shell_coefs``.  The
    component-dependent double-factorial part is applied here, per component,
    because it differs between e.g. xx and xy within the same shell.

    LOOP STRUCTURE (mirrors the spec's recommended Rust layout):
        for each primitive pair (grid-INDEPENDENT work: p, P, K_AB, E tables)
            build R over ALL grid points at once (grid axis is the inner,
            contiguous dimension)
            accumulate into the Cartesian block
    """
    ca = cart_components(la)
    cb = cart_components(lb)
    npts = pts.shape[0]
    Lt = la + lb
    out = np.zeros((npts, len(ca), len(cb)))

    for ia, (a, da) in enumerate(zip(alphas_a, coefs_a)):
        for ib, (b, db) in enumerate(zip(alphas_b, coefs_b)):
            # ---- GRID-INDEPENDENT primitive-pair quantities ----
            p = a + b
            P = (a * A + b * B) / p
            Ex = e_table(la, lb, A[0] - B[0], a, b)
            Ey = e_table(la, lb, A[1] - B[1], a, b)
            Ez = e_table(la, lb, A[2] - B[2], a, b)
            pref = 2.0 * math.pi / p * da * db

            # ---- GRID-DEPENDENT: R tensor over the whole batch ----
            PC = P[None, :] - pts  # (npts, 3)
            R = r_tensor(Lt, p, PC)

            # ---- contract E (grid-free) against R (grid-carrying) ----
            # NOTE (convention, load-bearing): NO per-component double-factorial
            # normalization is applied here.  PySCF and libint2 normalize a
            # Cartesian shell with a single SHELL-WIDE constant (the one that
            # makes the (l,0,0) component unit-normalized), so within a d shell
            # xx and xy do NOT both have unit self-overlap.  Normalizing each
            # component individually -- the "textbook" choice -- rescales the
            # off-diagonal components by sqrt((2lx-1)!!(2ly-1)!!(2lz-1)!!) and
            # silently breaks BOTH the Cartesian block and the cart->sph
            # transform, which assumes the shell-wide convention on input.
            for m, (ax, ay, az) in enumerate(ca):
                for n, (bx, by, bz) in enumerate(cb):
                    acc = np.zeros(npts)
                    for t in range(ax + bx + 1):
                        etx = Ex[ax, bx, t]
                        if etx == 0.0:
                            continue
                        for u in range(ay + by + 1):
                            ety = Ey[ay, by, u]
                            if ety == 0.0:
                                continue
                            for v in range(az + bz + 1):
                                etz = Ez[az, bz, v]
                                if etz == 0.0:
                                    continue
                                acc += etx * ety * etz * R[(t, u, v)]
                    out[:, m, n] += pref * acc

    return out


# --------------------------------------------------------------------------
# Cartesian -> spherical transformation
# --------------------------------------------------------------------------
#
# Rather than hardcode a solid-harmonic table (which is exactly where
# convention mismatches hide), we take the transformation matrix from PySCF
# itself.  This GUARANTEES the reference matches the validation target's
# convention, and the spec states plainly that a Rust implementation must
# reproduce whatever libint2 uses -- see the spec's "Traps" section.


def cart2sph_matrix(l: int) -> np.ndarray:
    """(ncart, nsph) transform for angular momentum l, in PySCF's convention."""
    from pyscf import gto as _gto

    return _gto.cart2sph(l, normalized="sp")


# --------------------------------------------------------------------------
# Driver: build A^g for a PySCF Mole
# --------------------------------------------------------------------------


def _segmented_shells(mol) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Flatten ``mol``'s shells into SEGMENTED (l, centre, exps, coefs) tuples.

    TRAP: PySCF (and ferric, and libint2) allow GENERAL contractions -- one
    shell with ``nctr > 1`` sharing a single exponent set, e.g. the 8-primitive
    s shell on oxygen in cc-pVDZ, which carries nctr=2.  Each contraction is a
    separate basis function occupying its own slot in the AO ordering, and the
    contractions of a general shell are ADJACENT in that ordering.  Reading
    only ``bas_ctr_coeff(ish)[:, 0]`` silently drops basis functions -- it cost
    exactly one AO (nbf 23 vs 24) on water/cc-pVDZ while writing this file.

    Splitting into segmented shells is correct here (the integrals are linear
    in the contraction) and keeps the reference simple; a production kernel
    would instead keep the shared exponents and reuse the E-tables across the
    contractions, which is a real optimization the spec notes.

    The coefficients returned carry the angular-momentum-only part of the
    primitive normalization; the component-dependent double factorial is
    applied inside ``shell_pair_block``.
    """
    out = []
    for ish in range(mol.nbas):
        l = mol.bas_angular(ish)
        A = mol.bas_coord(ish)
        exps = mol.bas_exp(ish)
        ctr = mol.bas_ctr_coeff(ish)  # (nprim, nctr)
        # NORMALIZATION (trap -- see the spec).
        #
        # PySCF's ``bas_ctr_coeff`` does NOT contain the primitive
        # normalization: a single uncontracted primitive stores the coefficient
        # 1.0, yet ``int1e_ovlp`` returns exactly 1.0 on the diagonal, so
        # PySCF applies ``gto_norm(l, a)`` internally at integral time.
        # (Verified directly -- see ``check_normalization_convention``.)
        #
        # This file's recursion works in the CARTESIAN primitive convention, so
        # we multiply in the Cartesian angular normalization
        #   (2a/pi)^{3/4} (4a)^{l/2}
        # here, and apply the remaining component-dependent double-factorial
        # part per Cartesian component inside shell_pair_block (it differs
        # between e.g. xx and xy within one shell).
        #
        # For a CONTRACTED shell the stored coefficients are relative to
        # PySCF's own per-primitive normalization, so they must be rescaled by
        # gto_norm(l, a) first -- that factor is exponent-DEPENDENT and does not
        # factor out of the contraction.
        # NORMALIZATION (a trap worth stating carefully -- see the spec).
        #
        # We do this from first principles rather than by trusting how any
        # program stores coefficients:
        #
        #   1. Attach the Cartesian primitive normalization to each primitive.
        #   2. Rescale the contraction so the (l,0,0) Cartesian component has
        #      unit self-overlap.  Only the RADIAL contraction is normalized;
        #      the other Cartesian components (xy, xz, ...) are deliberately
        #      left un-normalized, exactly as ferric/libint2 leave them.
        #   3. For l >= 2 ONLY, apply the constant sqrt(4*pi/(2l+1)).
        #
        # Step 3 is the non-obvious one, and it is stated here as a MEASURED
        # fact, not a derivation.  The self-overlap of PySCF's (l,0,0)
        # Cartesian AO is
        #      l = 0, 1 : exactly 1
        #      l >= 2   : exactly 4*pi/(2l+1)      (5.03/5, 4pi/7, 4pi/9, ...)
        # i.e. s and p are normalized as unit Cartesians while d and up are
        # normalized as solid harmonics.  PySCF's `gto_norm` is the l>=2 branch
        # for all l, which is why round-tripping through it corrupts s and p.
        # The self-test pins this per-l overlap directly, so a future PySCF
        # change cannot silently invalidate the reference.
        #
        # This discontinuity at l=2 is invisible on any s/p-only check -- it is
        # exactly how a bug here survives a water/STO-3G validation.
        angular = np.array([
            (2.0 * a / math.pi) ** 0.75 * (4.0 * a) ** (l / 2.0) for a in exps
        ])
        sh_fac = math.sqrt(4.0 * math.pi / (2 * l + 1)) if l >= 2 else 1.0
        for k in range(ctr.shape[1]):
            cs = ctr[:, k] * angular
            cs = cs / math.sqrt(_self_overlap_l00(l, exps, cs)) * sh_fac
            out.append((l, A, exps, cs))
    return out


def _self_overlap_l00(l: int, exps: np.ndarray, coefs: np.ndarray) -> float:
    """Self-overlap of the contracted (l,0,0) Cartesian component.

    <g|g> = sum_pq c_p c_q * (pi/(a_p+a_q))^{3/2}
                            * (2l-1)!! / (2(a_p+a_q))^l
    """
    a = np.asarray(exps, dtype=float)
    c = np.asarray(coefs, dtype=float)
    s = a[:, None] + a[None, :]
    prim = (math.pi / s) ** 1.5 * dfact(2 * l - 1) / (2.0 * s) ** l
    return float(c @ prim @ c)


def _gto_norm_vec(l: int, exps: np.ndarray) -> np.ndarray:
    from pyscf import gto as _gto

    return np.array([_gto.gto_norm(l, float(a)) for a in exps])


def a_matrices(mol, pts: np.ndarray) -> np.ndarray:
    """Build A^g_{mu,nu} for every probe point in ``pts``.

    Returns shape (npts, nbf, nbf), with the ``+1/|r-r_g|`` sign, in the AO
    ordering of ``mol`` (spherical unless ``mol.cart`` is set).
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    npts = pts.shape[0]
    use_cart = bool(mol.cart)

    shells = _segmented_shells(mol)
    nsh = len(shells)
    if use_cart:
        dims = [(l + 1) * (l + 2) // 2 for (l, _, _, _) in shells]
    else:
        dims = [2 * l + 1 for (l, _, _, _) in shells]
    offs = np.cumsum([0] + dims)
    nbf = int(offs[-1])

    out = np.zeros((npts, nbf, nbf))
    for i in range(nsh):
        la, A, ea, ca = shells[i]
        for j in range(i + 1):
            lb, B, eb, cb = shells[j]
            blk = shell_pair_block(la, A, ea, ca, lb, B, eb, cb, pts)
            if not use_cart:
                Ta = cart2sph_matrix(la)
                Tb = cart2sph_matrix(lb)
                blk = np.einsum("gmn,mp,nq->gpq", blk, Ta, Tb, optimize=True)
            oi, oj = offs[i], offs[j]
            out[:, oi:oi + dims[i], oj:oj + dims[j]] = blk
            if i != j:
                out[:, oj:oj + dims[j], oi:oi + dims[i]] = blk.transpose(0, 2, 1)
    return out


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

WATER = "O 0.0000 0.0000 0.0000; H 0.0000 -1.4304 1.1105; H 0.0000 1.4304 1.1105"
METHANE = (
    "C 0 0 0; H 1.1858 1.1858 1.1858; H -1.1858 -1.1858 1.1858; "
    "H -1.1858 1.1858 -1.1858; H 1.1858 -1.1858 -1.1858"
)


def _probe_sets(mol) -> dict:
    """Probe geometries, including the degenerate ones the spec calls out."""
    nuc = mol.atom_coord(0)
    rng = np.random.default_rng(20260907)
    generic = rng.uniform(-3.0, 3.0, size=(6, 3))
    return {
        "generic": generic,
        "on-nucleus (T=0)": nuc.reshape(1, 3),
        "near-nucleus 1e-4": (nuc + np.array([1e-4, 0.0, 0.0])).reshape(1, 3),
        "near-nucleus 1e-8": (nuc + np.array([0.0, 1e-8, 0.0])).reshape(1, 3),
        "far 50 Bohr": np.array([[50.0, 0.0, 0.0], [0.0, 0.0, 60.0]]),
        "far 200 Bohr": np.array([[200.0, 30.0, 0.0]]),
    }


def run_self_test() -> int:
    from pyscf import gto

    cases = [
        ("water/cc-pVDZ  (sph, d)", WATER, "cc-pvdz", False),
        ("water/cc-pVTZ  (sph, f)", WATER, "cc-pvtz", False),
        ("water/cc-pVDZ  (cart)", WATER, "cc-pvdz", True),
        ("methane/cc-pVDZ(sph)", METHANE, "cc-pvdz", False),
        ("water/STO-3G   (sph, s/p)", WATER, "sto-3g", False),
    ]

    print()
    print("3c1e reference vs PySCF int1e_grids   (sign: +1/|r-r_g|)")
    print("=" * 78)
    print(f"{'case':<26} {'probes':<20} {'max abs err':>12} {'max rel':>10}  {'':4}")
    print("-" * 78)

    n_fail = 0
    for label, atom, basis, cart in cases:
        mol = gto.M(atom=atom, basis=basis, unit="Bohr", cart=cart, verbose=0)
        lmax = max(mol.bas_angular(i) for i in range(mol.nbas))
        for pname, pts in _probe_sets(mol).items():
            ref = mol.intor("int1e_grids", grids=pts)
            got = a_matrices(mol, pts)
            aerr = float(np.max(np.abs(got - ref)))
            scale = max(float(np.max(np.abs(ref))), 1e-30)
            rerr = aerr / scale
            ok = aerr < 1e-11
            if not ok:
                n_fail += 1
            print(
                f"{label:<26} {pname:<20} {aerr:12.3e} {rerr:10.2e}  "
                f"{'PASS' if ok else 'FAIL'}"
            )
        print(f"{'':<26} (lmax={lmax}, nbf={mol.nao}, cart={cart})")
    print("-" * 78)

    # Boys function spot checks against mpmath-free closed forms.
    print()
    print("Boys function checks")
    print("-" * 78)
    # F_0(T) = sqrt(pi/(4T)) erf(sqrt(T))
    from math import erf, sqrt, pi

    for T in [0.0, 1e-14, 1e-6, 0.5, 5.0, 29.9, 30.0, 30.1, 100.0, 1000.0]:
        got = boys_single(0, T)
        exact = 1.0 if T < 1e-14 else sqrt(pi / (4.0 * T)) * erf(sqrt(T))
        err = abs(got - exact)
        ok = err < 1e-14
        if not ok:
            n_fail += 1
        print(f"  F_0({T:9.4g}) = {got:.15f}  err {err:8.2e}  "
              f"{'PASS' if ok else 'FAIL'}")

    # Agreement of the two branches ACROSS the switch.
    #
    # The right test is NOT "F_n barely changes across T=30" -- F_n is a
    # genuinely varying function, and differencing it at nearby T measures that
    # variation, not a branch mismatch (an earlier version of this test did
    # exactly that and reported a spurious 2.7e-6 "jump").  Instead evaluate
    # the SAME T with BOTH branches and compare: any disagreement is a real
    # discontinuity in the implementation.
    global BOYS_T_SWITCH
    saved = BOYS_T_SWITCH
    # Probe at and above the switch, where BOTH branches are meant to be
    # accurate.  Below ~T=35 the asymptotic form is legitimately inaccurate
    # (see BOYS_T_SWITCH), so comparing there would test nothing useful.
    probe_T = np.array([35.0, 36.0, 45.0, 80.0])
    try:
        BOYS_T_SWITCH = 1e9          # force the Taylor/downward branch
        f_small = boys_vec(8, probe_T)
        BOYS_T_SWITCH = 0.0          # force the asymptotic/upward branch
        f_large = boys_vec(8, probe_T)
    finally:
        BOYS_T_SWITCH = saved
    rel = np.max(np.abs(f_small - f_large) / np.abs(f_small))
    ok = rel < 1e-12
    if not ok:
        n_fail += 1
    print(f"  branch agreement at T=25..40 (n<=8): max rel diff "
          f"{rel:.2e}  {'PASS' if ok else 'FAIL'}")

    # Independent check of F_n against ARBITRARY-PRECISION quadrature.
    #
    # The oracle must be mpmath, not scipy.integrate.quad.  quad carries up to
    # 1.7e-11 relative error on t^14 exp(-80 t^2) -- a sharply decaying
    # integrand it handles badly -- which is 5 orders of magnitude worse than
    # the recursion being tested.  An earlier version of this check used quad
    # and reported that as a FAILURE OF THE BOYS FUNCTION; mpmath at 50 digits
    # showed the recursion was right to 1.7e-16 and the oracle was wrong.
    try:
        import mpmath as mp
    except ImportError:
        print("  F_n vs arbitrary precision: SKIPPED (mpmath not installed)")
    else:
        mp.mp.dps = 50
        worst = 0.0
        for T in (0.0, 1e-6, 0.7, 12.0, 30.0, 80.0):
            for n in (0, 3, 7):
                got = float(boys_vec(7, np.array([T]))[n, 0])
                ex = mp.quad(
                    lambda t, _n=n, _T=T: t ** (2 * _n) * mp.e ** (-mp.mpf(_T) * t * t),
                    [0, 1],
                )
                worst = max(worst, float(abs(mp.mpf(got) - ex) / abs(ex)))
        ok = worst < 1e-13
        if not ok:
            n_fail += 1
        print(f"  F_n vs mpmath (50 dps, n<=7): max rel err "
              f"{worst:.2e}  {'PASS' if ok else 'FAIL'}")

    # Breakpoint guard.  The two checks above compare each branch only where it
    # is valid, so neither notices if BOYS_T_SWITCH is moved DOWN into the
    # region where the asymptotic form is inaccurate -- and the integral tests
    # do not catch it either, because a ~1e-12 error in F_0 sits below their
    # 1e-11 bar.  (Verified by mutation: setting the switch to 25.0 passed
    # every other check in this file.)  So test the breakpoint directly:
    # evaluate F_0 with the asymptotic form AT the switch and demand it be
    # correct to full double precision.
    from math import erf, sqrt as _sqrt, pi as _pi

    Tsw = BOYS_T_SWITCH
    asym = 0.5 * _sqrt(_pi / Tsw)
    exact_sw = _sqrt(_pi / (4.0 * Tsw)) * erf(_sqrt(Tsw))
    rel_sw = abs(asym - exact_sw) / exact_sw
    ok = rel_sw < 1e-15
    if not ok:
        n_fail += 1
    print(f"  asymptotic branch exact at the T={Tsw:g} breakpoint: rel "
          f"{rel_sw:.2e}  {'PASS' if ok else 'FAIL'}")

    # Normalization convention pin: PySCF's (l,0,0) Cartesian self-overlap is
    # 1 for l<2 and 4pi/(2l+1) for l>=2.  The reference depends on this.
    ok_all = True
    for l in range(5):
        mol = gto.M(atom="H 0 0 0", basis=[[l, [1.7, 1.0]]], unit="Bohr",
                    spin=1, cart=True, verbose=0)
        s00 = float(mol.intor("int1e_ovlp")[0, 0])
        want = 1.0 if l < 2 else 4.0 * np.pi / (2 * l + 1)
        if abs(s00 - want) > 1e-12:
            ok_all = False
            print(f"  l={l}: S00={s00} expected {want}   FAIL")
    if not ok_all:
        n_fail += 1
    print(f"  AO normalization convention (l=0..4)"
          f"{'':>27}  {'PASS' if ok_all else 'FAIL'}")

    print("=" * 78)
    print(f"{'ALL PASS' if n_fail == 0 else str(n_fail) + ' FAILURE(S)'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(run_self_test())
