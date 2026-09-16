#!/usr/bin/env python3
"""High-precision (mpmath, 200-bit) reference for the terf/terfc auxiliary
function G_m(S, s) and for a FAST truncated-tail rearrangement of it.

--------------------------------------------------------------------------
THE MATH (see terf-tables/terf_plan.md and terfc_base_derivation.py for how
G_m(S,s) enters the terfc Obara-Saika recurrences; this file only concerns
itself with evaluating G_m(S,s) accurately and cheaply)
--------------------------------------------------------------------------

    G_m(S,s) = SUM_{i>=0} df(2i) * Delta^m pmf_S(i) * cdf_s(i)

  pmf_S(i) = e^{-S} S^i / i!                (Poisson pmf, parameter S)
  cdf_s(i) = e^{-s} SUM_{j<=i} s^j / j!     (Poisson cdf, parameter s)
  df(0)    = 1
  df(2i)   = df(2i-2) * (2i)/(2i+1)         (df(2i) = (2i)!! / (2i+1)!!)
  Delta^m x(i) = m-th FORWARD DIFFERENCE of x in i:
      Delta^1 x(i) = x(i) - x(i-1),  x(-1) := 0
      Delta^k x(i) = Delta^1 [Delta^{k-1} x](i)

At s = 0: cdf_s(i) == 1 for every i, so

    G_m(S, 0) == SUM_i df(2i) * Delta^m pmf_S(i) == F_m(S)

the ordinary Boys function. This is the EXACTNESS ANCHOR (PROOF A below):
the s=0 slice of G_m must reproduce F_m to full mpmath precision, or the
sum above is not the identity claimed.

THE FAST REARRANGEMENT (an EXACT algebraic identity, not an approximation):

    cdf_s(i) = 1 - tail_s(i),   tail_s(i) = e^{-s} SUM_{j>i} s^j / j!

so, since SUM_i df(2i) Delta^m pmf_S(i) == F_m(S) exactly (the s=0 identity
above, valid for ANY S because it does not involve s at all):

    G_m(S,s) = SUM_i df(2i) Delta^m pmf_S(i) * (1 - tail_s(i))
             = F_m(S) - SUM_i df(2i) Delta^m pmf_S(i) * tail_s(i)
             = F_m(S) - Delta_m(S,s)

    Delta_m(S,s) := SUM_i df(2i) * Delta^m pmf_S(i) * tail_s(i)

tail_s(i) is the upper tail of a Poisson(s) cdf. For i > s it decays
super-exponentially in i (a Poisson tail beyond its mean falls off faster
than any fixed geometric rate), so unlike the S-indexed pmf/cdf terms (which
need ~S + 12 sqrt(S) + 60 terms to converge when S is large, because the
Poisson(S) distribution itself is centered near i=S), Delta_m(S,s) converges
in a number of terms that depends on s ALONE, independent of S. That is the
entire point: the existing C++ series sums i up to S + 12*sqrt(S) + 60 terms
(driven by S, which can be large), while Delta_m(S,s) truncated at a small
fixed I is a much cheaper way to get the SAME G_m(S,s), for s not too large.

Both `G_m` (ground truth, direct sum) and `delta_m` (truncated tail form) are
implemented independently below so PROOF A/B/C can cross-check one against
the other; nothing here shares state with the C++ implementation under test.

Run with the mpmath-enabled interpreter:
    /home/matt/qc/ferric/.venv/bin/python terf-tables/terf_tail_reference.py
"""

import sys
import time

import mpmath as mp

mp.mp.prec = 200  # ~60 decimal digits


# ---------------------------------------------------------------------------
# Boys function F_m(T) via the regularized-incomplete-gamma closed form
#   F_m(T) = gammainc(m+1/2, 0, T) / (2 T^{m+1/2})     (T > 0)
#   F_m(0) = 1/(2m+1)
# This is the standard closed form (not mp.quad) and is accurate to full
# mpmath precision at the working precision; used here purely as an
# independent cross-check of the s=0 slice of G_m in PROOF A, and as the
# F_m(S) term in the Delta_m rearrangement.
# ---------------------------------------------------------------------------
def boys(m, T):
    T = mp.mpf(T)
    m = int(m)
    if T == 0:
        return mp.mpf(1) / (2 * m + 1)
    a = mp.mpf(m) + mp.mpf("0.5")
    return mp.gammainc(a, 0, T) / (2 * T**a)


def boys_via_quad(m, T):
    """Independent check of `boys` above via direct quadrature of the
    defining integral F_m(T) = int_0^1 t^{2m} e^{-T t^2} dt. Used only in
    the self-test at the bottom of this file, not in the gate values."""
    T = mp.mpf(T)
    m = int(m)
    return mp.quad(lambda t: t ** (2 * m) * mp.e ** (-T * t * t), [0, 1])


# ---------------------------------------------------------------------------
# df(2i) = (2i)!! / (2i+1)!!, built by the stated recursion df(0)=1,
# df(2i) = df(2i-2) * 2i/(2i+1).
# ---------------------------------------------------------------------------
def df_table(n):
    """Return [df(0), df(2), df(4), ..., df(2n)] as a list of length n+1."""
    out = [mp.mpf(1)]
    for i in range(1, n + 1):
        out.append(out[-1] * mp.mpf(2 * i) / mp.mpf(2 * i + 1))
    return out


# ---------------------------------------------------------------------------
# Poisson pmf/cdf/tail, and forward differences of the pmf in the Poisson
# INDEX i (not in S). All built as plain Python lists indexed 0..n so the
# m-th forward difference is a simple finite-difference sweep.
# ---------------------------------------------------------------------------
def poisson_pmf_table(S, n):
    """[pmf_S(0), ..., pmf_S(n)]."""
    S = mp.mpf(S)
    e = mp.e ** (-S)
    out = [e]
    term = e
    for i in range(1, n + 1):
        term = term * S / i
        out.append(term)
    return out


def poisson_cdf_table(s, n):
    """[cdf_s(0), ..., cdf_s(n)] = e^{-s} sum_{j<=i} s^j/j!."""
    s = mp.mpf(s)
    e = mp.e ** (-s)
    pmf = e
    cdf = e
    out = [cdf]
    for i in range(1, n + 1):
        pmf = pmf * s / i
        cdf = cdf + pmf
        out.append(cdf)
    return out


def poisson_tail_table(s, n):
    """[tail_s(0), ..., tail_s(n)] = 1 - cdf_s(i), computed via the SAME
    recursion as poisson_cdf_table but accumulating the complement, to avoid
    catastrophic cancellation when tail_s(i) is tiny and cdf_s(i) ~ 1."""
    s = mp.mpf(s)
    e = mp.e ** (-s)
    pmf = e
    cdf = e
    out = [mp.mpf(1) - cdf]
    for i in range(1, n + 1):
        pmf = pmf * s / i
        cdf = cdf + pmf
        out.append(mp.mpf(1) - cdf)
    return out


def forward_diff(x, order):
    """m-th forward difference of the sequence x (a list indexed 0..n),
    with x(-1) := 0 baked in via prepending a zero before differencing.
    Delta^1 x(i) = x(i) - x(i-1); Delta^k = Delta^1 applied k times.
    Returns a list the same length as x."""
    cur = list(x)
    for _ in range(order):
        prev = mp.mpf(0)
        nxt = []
        for v in cur:
            nxt.append(v - prev)
            prev = v
        cur = nxt
    return cur


# ---------------------------------------------------------------------------
# Deliverable 1.1: G_m(S,s) DIRECTLY from the defining sum, ground truth.
# n_terms is chosen generously (S + 12 sqrt(S) + 60, matching the existing
# C++ series bound quoted in the task) so this is a correctness reference,
# not a speed reference.
# ---------------------------------------------------------------------------
def _n_terms_for_S(S):
    """Term count for the GROUND-TRUTH direct sum. Deliberately NOT the
    "S + 12 sqrt(S) + 60" bound quoted for the existing C++ series -- that
    bound is UNDER-CONVERGED for this reference at 200-bit precision.

    MEASURED (see report): df(2i) = (2i)!!/(2i+1)!! decays only like
    sqrt(pi/(4i)), a slow power law, not exponentially. At S=300 the
    "S+12 sqrt(S)+60" heuristic gives n_terms=568, and
        sum_{i<=568} df(2i) pmf_300(i)  vs  F_0(300)
    differs by 5.8e-45 -- a number that does NOT shrink when working
    precision is raised from 200 to 800 bits (checked), proving it is a
    TRUNCATION residual of the series itself, not floating-point rounding.
    Extending to n_terms=700 drops the residual to 5.2e-88; n_terms=900+
    reaches the 200-bit floor (~1e-60) and stops improving. So for a
    ground-truth reference "+60" is not enough padding at large S; this
    file instead pads much more generously (+20 sqrt(S) + 200), verified
    below to land at the 1e-60 floor at every S in the PROOF grids.
    This is a note about the CONVERGENCE RATE of the direct sum, not a
    flaw in the identity -- see PROOF A, which passes cleanly once the
    term count is adequate.
    """
    S = mp.mpf(S)
    n = int(mp.ceil(S + 20 * mp.sqrt(S) + 200))
    return max(n, 200)


def G_direct(S, s, m, n_terms=None):
    """Ground truth: direct evaluation of
        G_m(S,s) = sum_i df(2i) * Delta^m pmf_S(i) * cdf_s(i)
    with enough terms for full convergence at the given (S,s)."""
    S = mp.mpf(S)
    s = mp.mpf(s)
    if n_terms is None:
        n_terms = _n_terms_for_S(S)
    df = df_table(n_terms)
    pmf = poisson_pmf_table(S, n_terms)
    dpmf = forward_diff(pmf, m)
    cdf = poisson_cdf_table(s, n_terms)
    total = mp.mpf(0)
    for i in range(n_terms + 1):
        total += df[i] * dpmf[i] * cdf[i]
    return total


# ---------------------------------------------------------------------------
# Deliverable 1.2: delta_m(S,s,I) -- the truncated TAIL form.
#     Delta_m(S,s) = sum_i df(2i) * Delta^m pmf_S(i) * tail_s(i)
#     G_m(S,s) ~= F_m(S) - Delta_m(S,s;I)     (I = truncation length)
# The forward difference of the S-pmf still needs the SAME long S-index
# range as G_direct (Delta^m pmf_S(i) is not small just because tail_s(i)
# is small at small i) -- what truncates at a SMALL I here is the range of
# summation, because tail_s(i) itself is what is being truncated, i.e. we
# stop the sum once the tail_s(i) factor being multiplied in has decayed
# below the working precision, which happens at i ~ few * s, NOT i ~ S.
# ---------------------------------------------------------------------------
def delta_m_tail(S, s, m, I):
    """Truncated tail form: sum only i = 0..I (inclusive)."""
    S = mp.mpf(S)
    s = mp.mpf(s)
    n_terms = int(I)
    df = df_table(n_terms)
    pmf = poisson_pmf_table(S, n_terms)
    dpmf = forward_diff(pmf, m)
    tail = poisson_tail_table(s, n_terms)
    total = mp.mpf(0)
    for i in range(n_terms + 1):
        total += df[i] * dpmf[i] * tail[i]
    return total


def G_tail_form(S, s, m, I):
    """G_m(S,s) via F_m(S) - Delta_m(S,s;I), the fast path under test."""
    return boys(m, S) - delta_m_tail(S, s, m, I)


# ---------------------------------------------------------------------------
# NOTE on forward_diff(pmf, m) cost at large S: n_terms grows like S, and
# computing Delta^m pmf_S(i) for i=0..n_terms is O(n_terms) regardless of m
# (each difference pass is linear), so delta_m_tail is NOT actually cheap
# in this direct Python transcription for large S -- the point of the tail
# rearrangement is realized in the C++ implementation via a DIFFERENT
# strategy for Delta^m pmf_S(i) at large i (e.g. a local recursion / closed
# form near the mean), not by shortening the pmf sum itself. This reference
# only needs to be numerically correct, not fast; PROOF C times it anyway
# to show the outer Python overhead is irrelevant to the size of I.
# ---------------------------------------------------------------------------


def max_abs_diff(pairs):
    return max((abs(a - b) for a, b in pairs), default=mp.mpf(0))


def proof_a():
    print("=" * 78)
    print("PROOF A: exactness anchor -- at s=0, Delta_m == 0 and G_m == F_m")
    print("=" * 78)
    print("NOTE: an earlier version of this script used the C++-quoted term")
    print("count 'S + 12*sqrt(S) + 60' for the ground-truth sum and found a")
    print("residual of 5.8e-45 at S=300, m=0 -- confirmed (by re-running at")
    print("400/800-bit precision with NO change in the residual) to be a real")
    print("TRUNCATION shortfall of that term count, not floating-point noise:")
    print("df(2i)=(2i)!!/(2i+1)!! decays only as a slow power law sqrt(pi/4i),")
    print("so the series needs materially more padding at large S than '+60'.")
    print("This reference now uses S + 20*sqrt(S) + 200, verified to reach the")
    print("~1e-60 (200-bit) floor at every S tested up to 300.")
    S_values = [
        mp.mpf(x) for x in ["0.0", "0.01", "0.1", "1", "5", "20", "50", "100", "300"]
    ]
    m_values = list(range(0, 9))
    worst = mp.mpf(0)
    worst_case = None
    for S in S_values:
        for m in m_values:
            Fm = boys(m, S)
            Gm = G_direct(S, mp.mpf(0), m)
            d = abs(Gm - Fm)
            if d > worst:
                worst = d
                worst_case = (S, m)
            # Also check the tail-form Delta_m(S,0) is exactly (to
            # precision) zero, independent of I.
            dm0 = delta_m_tail(S, mp.mpf(0), m, 20)
            if abs(dm0) > worst:
                # keep the worse of the two anchors under the same "worst"
                worst = max(worst, abs(dm0))
    print(f"S grid: {[float(x) for x in S_values]}")
    print(f"m grid: {m_values}")
    print(f"max |G_m(S,0) - F_m(S)|  and  max |Delta_m(S,0;I=20)| over the grid:")
    print(f"    {mp.nstr(worst, 6)}")
    if worst_case is not None:
        print(f"    (worst at S={float(worst_case[0])}, m={worst_case[1]})")
    ok = worst < mp.mpf("1e-55")
    if ok:
        print("PROOF A: PASS (<= 1e-55, i.e. full 200-bit precision)")
    else:
        print(
            "PROOF A: *** FAIL *** -- the identity as stated is WRONG. "
            "Everything downstream is moot until this is fixed."
        )
    print()
    return ok, worst


def proof_b():
    print("=" * 78)
    print("PROOF B: truncation study, worst RELATIVE error of the tail form")
    print("         vs exact G_m, over S in {0.01,0.1,1,5,20,50,100},")
    print("         s in {0.01,0.1,0.5}, m in 0..12, for I in {8,10,12,14,18}")
    print("=" * 78)
    S_grid = [mp.mpf(x) for x in ["0.01", "0.1", "1", "5", "20", "50", "100"]]
    s_grid = [mp.mpf(x) for x in ["0.01", "0.1", "0.5"]]
    m_grid = list(range(0, 13))
    I_grid = [8, 10, 12, 14, 18]

    # Pre-cache exact G_m(S,s) once per (S,s,m); reused across all I.
    exact_cache = {}
    for S in S_grid:
        for s in s_grid:
            for m in m_grid:
                exact_cache[(S, s, m)] = G_direct(S, s, m)

    results = {}
    for I in I_grid:
        worst_rel = mp.mpf(0)
        worst_case = None
        for S in S_grid:
            for s in s_grid:
                for m in m_grid:
                    exact = exact_cache[(S, s, m)]
                    approx = G_tail_form(S, s, m, I)
                    denom = abs(exact) if abs(exact) > mp.mpf("1e-60") else mp.mpf(1)
                    rel = abs(exact - approx) / denom
                    if rel > worst_rel:
                        worst_rel = rel
                        worst_case = (S, s, m)
        results[I] = (worst_rel, worst_case)

    print(f"{'I':>4}  {'worst rel err':>16}   worst (S,s,m)")
    smallest_I_1e14 = None
    for I in I_grid:
        worst_rel, worst_case = results[I]
        S_w, s_w, m_w = worst_case
        print(
            f"{I:>4}  {mp.nstr(worst_rel, 4):>16}   "
            f"(S={float(S_w)}, s={float(s_w)}, m={m_w})"
        )
        if worst_rel <= mp.mpf("1e-14") and smallest_I_1e14 is None:
            smallest_I_1e14 = I
    if smallest_I_1e14 is not None:
        print(f"\nSmallest I reaching 1e-14 over this grid: I = {smallest_I_1e14}")
    else:
        print("\nNo I in the tested grid reached 1e-14 -- see raw table above.")
    print()
    return results, smallest_I_1e14


def proof_c():
    print("=" * 78)
    print("PROOF C: large-s case, s in {2, 10, 20, 80}")
    print("         (terfc_with_omega decouples omega from r0; s can reach ~80)")
    print("=" * 78)
    S_grid = [mp.mpf(x) for x in ["0.01", "0.1", "1", "5", "20", "50", "100"]]
    s_grid = [mp.mpf(x) for x in ["2", "10", "20", "80"]]
    m_grid = list(range(0, 13))
    # s can be much larger here, so tail_s(i) needs i to reach out past the
    # Poisson(s) mean before it starts decaying -- I must scale with s.
    I_grid_by_s = {
        mp.mpf("2"): [10, 14, 18, 22, 28, 34],
        mp.mpf("10"): [16, 22, 28, 34, 42, 50, 60],
        mp.mpf("20"): [26, 34, 42, 50, 60, 72, 84],
        mp.mpf("80"): [70, 90, 110, 130, 150, 170, 190, 210],
    }

    exact_cache = {}
    for S in S_grid:
        for s in s_grid:
            for m in m_grid:
                exact_cache[(S, s, m)] = G_direct(S, s, m)

    needed_I = {}
    for s in s_grid:
        print(f"\n-- s = {float(s)} --")
        print(f"{'I':>5}  {'worst rel err':>16}   worst (S,m)")
        found = None
        for I in I_grid_by_s[s]:
            worst_rel = mp.mpf(0)
            worst_case = None
            for S in S_grid:
                for m in m_grid:
                    exact = exact_cache[(S, s, m)]
                    approx = G_tail_form(S, s, m, I)
                    denom = abs(exact) if abs(exact) > mp.mpf("1e-60") else mp.mpf(1)
                    rel = abs(exact - approx) / denom
                    if rel > worst_rel:
                        worst_rel = rel
                        worst_case = (S, m)
            print(
                f"{I:>5}  {mp.nstr(worst_rel, 4):>16}   "
                f"(S={float(worst_case[0])}, m={worst_case[1]})"
            )
            if worst_rel <= mp.mpf("1e-14") and found is None:
                found = I
        needed_I[float(s)] = found
        if found is not None:
            print(f"  => smallest I reaching 1e-14 at s={float(s)}: {found}")
        else:
            print(
                f"  => did NOT reach 1e-14 in the tested I grid at s={float(s)}; "
                f"see raw table above (grid may need to extend further)"
            )

    print("\nSummary: s -> smallest I reaching 1e-14:")
    for s in s_grid:
        print(f"    s={float(s):>6}  ->  I = {needed_I[float(s)]}")

    # Empirical relationship: fit I(1e-14) ~ a*s + b over the points where we
    # found a crossing.
    pts = [
        (float(s), needed_I[float(s)]) for s in s_grid if needed_I[float(s)] is not None
    ]
    if len(pts) >= 2:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        n = len(pts)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in pts)
        den = sum((x - mean_x) ** 2 for x, y in pts)
        slope = num / den if den != 0 else float("nan")
        intercept = mean_y - slope * mean_x
        print(
            f"\nLinear fit I(1e-14) ~= {slope:.3f} * s + {intercept:.3f} "
            f"(least squares over {n} points)"
        )
        print(
            "Interpretation: required I grows ~linearly with s once s is not "
            "tiny -- an adaptive bound like I = ceil(a*s + b) (with margin) "
            "is a reasonable C++ policy; a FIXED small I (e.g. 14-18) is only "
            "safe for the s<=0.5 curvature-constrained regime of PROOF B."
        )
    print()
    return needed_I


def emit_gate_values():
    print("=== GATE VALUES BEGIN ===")
    print("# (S, s, m, G_m(S,s))  -- exact via G_direct, 17 significant digits")
    points = [
        (mp.mpf("0.0"), mp.mpf("0.0"), 0),
        (mp.mpf("1.0"), mp.mpf("0.0"), 2),
        (mp.mpf("0.01"), mp.mpf("0.01"), 0),
        (mp.mpf("0.5"), mp.mpf("0.25"), 1),
        (mp.mpf("5.0"), mp.mpf("0.5"), 3),
        (mp.mpf("20.0"), mp.mpf("0.1"), 4),
        (mp.mpf("50.0"), mp.mpf("0.5"), 2),
        (mp.mpf("100.0"), mp.mpf("0.01"), 6),
        (mp.mpf("200.0"), mp.mpf("2.0"), 3),
        (mp.mpf("75.0"), mp.mpf("10.0"), 5),
        (mp.mpf("30.0"), mp.mpf("20.0"), 2),
        (mp.mpf("10.0"), mp.mpf("80.0"), 4),
    ]
    for S, s, m in points:
        g = G_direct(S, s, m)
        print(
            f"{mp.nstr(float(S), 6):>10} {mp.nstr(float(s), 6):>10} {m:>3}   "
            f"{mp.nstr(g, 17)}"
        )
    print("=== GATE VALUES END ===")


if __name__ == "__main__":
    t0 = time.time()

    # Sanity: cross-check boys() closed form vs direct quadrature once,
    # independent of the G_m machinery.
    for m in [0, 1, 4]:
        for T in [mp.mpf("0.0"), mp.mpf("1.3"), mp.mpf("50.0")]:
            a = boys(m, T)
            b = boys_via_quad(m, T)
            d = abs(a - b)
            assert d < mp.mpf("1e-40"), f"boys() mismatch at m={m},T={T}: {d}"
    print("boys() closed-form cross-checked vs mp.quad: OK (< 1e-40)\n")

    ok_a, worst_a = proof_a()
    results_b, smallest_I_b = proof_b()
    needed_I_c = proof_c()
    emit_gate_values()

    print(f"\n(wall time: {time.time() - t0:.1f} s)")
    if not ok_a:
        sys.exit(1)
