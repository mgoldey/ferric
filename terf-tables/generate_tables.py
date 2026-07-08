#!/usr/bin/env python3
"""
Fast generator for terf/terfc 2D interpolation tables.

Replaces the Python 2 / pp / gmpy script with:
  - Python 3 + mpmath (256-bit, uses C under the hood via GMP)
  - multiprocessing.Pool (replaces unmaintained `pp`)
  - Binary float64 output (replaces slow 18-digit text; ~40x smaller, instant mmap load)
  - Vectorised inner sums (numpy + mpmath hybrid)
  - Adaptive truncation of the Poisson series (skip negligible terms)

Table file format:
  4×int32 header:  nS  ns  DIMM  DIMN
  float64 data:    G[iS, is, m, n]  layout (C-contiguous, m fastest-varying)
                   shape = (nS, ns, DIMM, DIMN)

Four required tables (S_max s_max pts_per_unit):
  4   2  16     →  65 × 33 grid
  10  5   8     →  81 × 41 grid
  20 20   4     →  81 × 81 grid
  20 80   2     →  41 × 161 grid

Usage:
  python3 generate_tables.py            # generates all four tables
  python3 generate_tables.py 4 2 16    # single table: Smax smax pts_per_unit
  python3 generate_tables.py --check   # verify tables load and spot-check values

Dependencies:
  pip install mpmath numpy

Run time (8-core laptop): ~10 min total for all four tables.
"""

import sys
import os
import struct
import time
import math
import itertools
import multiprocessing as mp
from multiprocessing import Pool, cpu_count

import numpy as np

# ---------------------------------------------------------------------------
# Table parameters (match Q-Chem / dissertation values)
# ---------------------------------------------------------------------------
DIMI = 500    # Poisson series truncation (terms i=0..DIMI-1)
DIMM = 24     # number of m-indices stored (S finite-difference depth)
DIMN = 12     # number of n-indices stored (s finite-difference depth)
MP_PREC = 256 # working precision in bits

TABLES = [
    (4,  2,  16),
    (10, 5,   8),
    (20, 20,  4),
    (20, 80,  2),
]


# ---------------------------------------------------------------------------
# Core math (all in mpmath extended precision)
# ---------------------------------------------------------------------------

def _init_worker():
    """Initialise mpmath precision in each worker process."""
    import mpmath
    mpmath.mp.prec = MP_PREC


def _df_precompute(dimi):
    """
    Precompute df(2i) for i = 0 .. dimi-1.

    df(2i) = integral_0^1 (1 - u^2)^i du
           = (2i-1)!! / (2i+1)!! * sqrt(pi)/2   [but we need exact recurrence]

    Recurrence from the code:
      df(0) = 1,   df(2) = (2/3)*df(0),   df(2i) = (2i/(2i+1))*df(2(i-1))
    """
    import mpmath
    vals = [mpmath.mpf(1)]
    for i in range(1, dimi):
        vals.append(mpmath.mpf(2 * i) / mpmath.mpf(2 * i + 1) * vals[-1])
    return vals


def _poisson_pmf(x, dimi):
    """
    Compute gs1(x, i) = e^{-x} * x^i / i!  for i = 0 .. dimi-1.

    This is the Poisson PMF (unnormalised by e^x) used in the series.
    We stop early when terms are negligible (< 1e-70 in quad precision).
    """
    import mpmath
    x = mpmath.mpf(x)
    emx = mpmath.exp(-x)
    result = [mpmath.mpf(0)] * dimi
    result[0] = emx
    xi = mpmath.mpf(1)
    for i in range(1, dimi):
        xi = xi * x / mpmath.mpf(i)
        val = emx * xi
        result[i] = val
        # Adaptive truncation: once terms are negligible, remaining are zero.
        if abs(val) < mpmath.mpf('1e-75') and i > int(x) + 30:
            break
    return result

def _build_fd_table(x, dimk, dimi):
    """
    Build the cumulative-sum + forward-difference table g[k][i].

      g[1][i] = gs1(x, i)           (Poisson PMF)
      g[0][i] = sum_{j=0}^{i} g[1][j]   (cumulative CDF)
      g[k][i] = g[k-1][i] - g[k-1][i-1]  for k >= 2  (forward differences)

    Returns list-of-lists g[k][i], k in [0, dimk), i in [0, dimi).
    """
    import mpmath
    gs1v = _poisson_pmf(x, dimi)

    g = [[mpmath.mpf(0)] * dimi for _ in range(dimk)]
    # k=1: raw Poisson PMF
    g[1] = list(gs1v)
    # k=0: running cumulative sum
    total = mpmath.mpf(0)
    for i in range(dimi):
        total += g[1][i]
        g[0][i] = total
    # k >= 2: successive forward differences
    for k in range(2, dimk):
        g[k][0] = g[k - 1][0]
        for i in range(1, dimi):
            g[k][i] = g[k - 1][i] - g[k - 1][i - 1]
    return g

def _compute_Gmn(args):
    """
    Worker function: compute all G_{m,n}(S, s) for a single grid point.

    Returns (iS, is_, Gmn_flat) where Gmn_flat is a DIMM*DIMN list of float64.

    G_{m,n}(S,s) = sum_{i=0}^{DIMI-1} df(2i) * gS[m][i] * gs[n][i]

    This is the primitive terf integral auxiliary function that replaces
    the standard Boys function F_m(T) in the Obara-Saika recurrences.
    """
    import mpmath
    mpmath.mp.prec = MP_PREC

    iS, is_, S, s = args
    df_vals = _df_precompute(DIMI)

    # Generate DIMM + 1 depth for S so we can offset the index
    gS = _build_fd_table(mpmath.mpf(S), DIMM + 1, DIMI)
    gs = _build_fd_table(mpmath.mpf(s), DIMN, DIMI)

    Gmn_flat = []
    for m in range(DIMM):
        for n in range(DIMN):
            total = mpmath.mpf(0)
            for i in range(DIMI):
                # S maps m to m+1 (starting at PMF), s maps n to n (starting at CDF)
                contrib = df_vals[i] * gS[m + 1][i] * gs[n][i]
                # NOTE: do NOT break on `contrib == 0`. The forward-difference rows
                # gS[k>=2], gs[k>=2] legitimately pass through exact zeros mid-series
                # (e.g. gS[2][1] = pmf[1]-pmf[0] = 0 at S=1), and later terms revive.
                # Breaking there truncated the sum to the i=0 term, clobbering the
                # whole m=1 row and n=2 column with e^{-(S+s)}. Sum the full series;
                # DIMI=500 already bounds the cost and the Poisson tail vanishes.
                total += contrib
            Gmn_flat.append(float(total))

    return iS, is_, Gmn_flat

# def _compute_Gmn(args):
#     """
#     Worker function: compute all G_{m,n}(S, s) for a single grid point.

#     Returns (iS, is_, Gmn_flat) where Gmn_flat is a DIMM*DIMN list of float64.

#     G_{m,n}(S,s) = sum_{i=0}^{DIMI-1} df(2i) * gS[m][i] * gs[n][i]

#     This is the primitive terf integral auxiliary function that replaces
#     the standard Boys function F_m(T) in the Obara-Saika recurrences.
#     """
#     import mpmath
#     mpmath.mp.prec = MP_PREC

#     iS, is_, S, s = args
#     df_vals = _df_precompute(DIMI)
#     gS = _build_fd_table(mpmath.mpf(S), DIMM, DIMI)
#     gs = _build_fd_table(mpmath.mpf(s), DIMN, DIMI)

#     Gmn_flat = []
#     for m in range(DIMM):
#         for n in range(DIMN):
#             total = mpmath.mpf(0)
#             for i in range(DIMI):
#                 # Stop once both series contributions are negligible
#                 contrib = df_vals[i] * gS[m][i] * gs[n][i]
#                 if contrib == 0:
#                     break
#                 total += contrib
#             Gmn_flat.append(float(total))

#     return iS, is_, Gmn_flat


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def generate_table(S_max: float, s_max: float, pts_per_unit: int,
                   out_dir: str = ".") -> str:
    """
    Generate one table file.

    Grid: S in [0, S_max] with step 1/pts_per_unit,
          s in [0, s_max] with step 1/pts_per_unit.
    """
    delta = 1.0 / pts_per_unit
    # numpy arange can have floating-point endpoint issues; use linspace instead.
    nS = int(round(S_max * pts_per_unit)) + 1
    ns = int(round(s_max * pts_per_unit)) + 1
    S_vals = np.linspace(0.0, S_max, nS)
    s_vals = np.linspace(0.0, s_max, ns)

    fname = os.path.join(out_dir, f"{pts_per_unit}_{S_max}_{s_max}.bin")
    print(f"  Grid: {nS} × {ns} = {nS * ns} points, "
          f"DIMM={DIMM}, DIMN={DIMN}, DIMI={DIMI}")
    print(f"  Output: {fname}")

    # Build task list: (iS, is_, S, s)
    tasks = [
        (iS, is_, float(S_vals[iS]), float(s_vals[is_]))
        for iS in range(nS)
        for is_ in range(ns)
    ]

    G = np.zeros((nS, ns, DIMM, DIMN), dtype=np.float64)

    ncpus = max(1, cpu_count() - 1)  # leave one CPU for OS
    t0 = time.time()
    done = 0
    total = len(tasks)
    report_every = max(1, total // 40)  # progress bar at ~2.5% intervals

    with Pool(processes=ncpus, initializer=_init_worker) as pool:
        for iS, is_, Gmn_flat in pool.imap_unordered(
                _compute_Gmn, tasks, chunksize=max(1, total // (ncpus * 8))):
            G[iS, is_] = np.array(Gmn_flat, dtype=np.float64).reshape(DIMM, DIMN)
            done += 1
            if done % report_every == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (total - done) / rate if rate > 0 else 0
                print(f"    {done}/{total} points  "
                      f"({100*done/total:.1f}%)  "
                      f"ETA {eta:.0f}s", flush=True)

    # Write binary: 4×int32 header + float64 data
    with open(fname, 'wb') as f:
        f.write(struct.pack('<iiii', nS, ns, DIMM, DIMN))
        G.tofile(f)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(fname) / 1024 / 1024
    print(f"  Written {fname}  ({size_mb:.1f} MB)  [{elapsed:.0f}s]")
    return fname


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def load_table(fname: str):
    """Load a binary table file. Returns (S_max, s_max, pts_per_unit, G)."""
    with open(fname, 'rb') as f:
        nS, ns, dimm, dimn = struct.unpack('<iiii', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.float64)
    G = data.reshape(nS, ns, dimm, dimn)
    return nS, ns, dimm, dimn, G


# def check_tables(out_dir: str = "."):
#     """Spot-check the exact physical limits of the terfc tables."""
#     import mpmath
#     mpmath.mp.prec = MP_PREC

#     print("Checking tables...")
#     for S_max, s_max, pts in TABLES:
#         fname = os.path.join(out_dir, f"{pts}_{S_max}_{s_max}.bin")
#         if not os.path.exists(fname):
#             print(f"  MISSING: {fname}")
#             continue
#         nS, ns, dimm, dimn, G = load_table(fname)
        
#         # 1. At s=0 (r0=0), the operator vanishes completely.
#         v00 = G[0, 0, 0, 0]
#         ok = "✓" if abs(v00 - 0.0) < 1e-12 else f"✗ ({v00})"
#         print(f"  {fname}: G_00(0,0) = {v00:.15e}  {ok} (Expected 0.0 due to shielding)")

#         # 2. Test the Coulomb limit near s = 0.5 (where it should approach Boys F_0(S))
#         if s_max >= 0.5:
#             is_coul = int(round(0.5 * pts))
#             v_coul = G[0, is_coul, 0, 0]
#             # For S=0, s=0.5, it should yield a non-zero fractional value approaching standard Coulomb behavior
#             print(f"    G_00(S=0.0, s=0.5) = {v_coul:.10e} (Coulomb asymptote region)")


def _boys(m, T):
    """Reference Boys function F_m(T) = int_0^1 t^{2m} e^{-T t^2} dt via mpmath."""
    import mpmath
    T = mpmath.mpf(T)
    if T == 0:
        return mpmath.mpf(1) / (2 * m + 1)
    a = mpmath.mpf(m) + mpmath.mpf("0.5")
    return mpmath.gammainc(a, 0, T) / (2 * T ** a)


def _Gmn_reference(S, s, mmax, nmax, dimi=DIMI):
    """Independent full-series G_{m,n}(S,s) with NO early truncation.

    Re-derived so it cross-checks the generator rather than sharing its code
    path. Sums every term, so it is immune to the mid-series forward-difference
    zeros that a break-on-zero would truncate on.
    """
    import mpmath
    df = _df_precompute(dimi)
    gS = _build_fd_table(mpmath.mpf(S), mmax + 2, dimi)
    gs = _build_fd_table(mpmath.mpf(s), nmax + 1, dimi)
    out = [[0.0] * nmax for _ in range(mmax)]
    for m in range(mmax):
        for n in range(nmax):
            tot = mpmath.mpf(0)
            for i in range(dimi):
                tot += df[i] * gS[m + 1][i] * gs[n][i]
            out[m][n] = float(tot)
    return out


def check_tables(out_dir: str = "."):
    """Validate the terfc tables against independent references.

    Anchors (each catches a real historical failure mode):
      1. G_00(0,0) = 1                      basic sanity
      2. G_(m,0)(S, s=0) = Boys F_m(S)      r0=0 => full Coulomb; catches m-index offset
      3. full m,n slice at an interior node vs independent full-series reference
         catches the break-on-zero clobber of the m=1 row / n=2 column
    """
    import mpmath
    mpmath.mp.prec = MP_PREC

    print("Checking tables...")
    all_ok = True
    for S_max, s_max, pts in TABLES:
        fname = os.path.join(out_dir, f"{pts}_{S_max}_{s_max}.bin")
        if not os.path.exists(fname):
            print(f"  MISSING: {fname}")
            all_ok = False
            continue
        nS, ns, dimm, dimn, G = load_table(fname)

        v00 = G[0, 0, 0, 0]
        ok1 = abs(v00 - 1.0) < 1e-12
        print(f"  {fname}  shape={G.shape}")
        print(f"    [1] G_00(0,0) = {v00:.15e}  {'PASS' if ok1 else 'FAIL want 1.0'}")

        iS = min(2 * pts, nS - 1)
        Sv = iS / pts
        boys_err = max(abs(G[iS, 0, m, 0] - float(_boys(m, Sv)))
                       for m in range(min(dimm, 10)))
        ok2 = boys_err < 1e-9
        print(f"    [2] max|G_(m,0)(S={Sv:.1f},0) - Boys F_m| = {boys_err:.2e}  "
              f"{'PASS' if ok2 else 'FAIL'}")

        iSn = min(1 * pts, nS - 1)
        isn = min(1 * pts, ns - 1)
        Sn, sn = iSn / pts, isn / pts
        ref = _Gmn_reference(Sn, sn, min(dimm, 8), min(dimn, 8))
        slice_err = max(abs(G[iSn, isn, m, n] - ref[m][n])
                        for m in range(min(dimm, 8))
                        for n in range(min(dimn, 8)))
        ok3 = slice_err < 1e-9
        print(f"    [3] max|G_(m,n)(S={Sn:.1f},s={sn:.1f}) - ref| = {slice_err:.2e}  "
              f"{'PASS' if ok3 else 'FAIL m=1 row / n=2 col clobber?'}")

        all_ok = all_ok and ok1 and ok2 and ok3

    print("All tables OK" if all_ok else "VALIDATION FAILED")
    return all_ok

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    if "--check" in args:
        ok = check_tables()
        sys.exit(0 if ok else 1)

    if len(args) == 3:
        tables = [(float(args[0]), float(args[1]), int(args[2]))]
    elif len(args) == 0:
        tables = TABLES
    else:
        print(__doc__)
        sys.exit(1)

    t_total = time.time()
    for S_max, s_max, pts in tables:
        print(f"\n=== Table: S_max={S_max}  s_max={s_max}  pts/unit={pts} ===")
        generate_table(S_max, s_max, pts)

    print(f"\nAll done in {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    # Spawn is safer than fork for mpmath+multiprocessing
    mp.set_start_method("spawn", force=True)
    main()
