# COSX L-axis pre-registration (2026-09-07, branch feat/cosx-seminumerical-k)

Written BEFORE any timing was taken. Nothing below is edited after measurement.

## What is being measured

For (butane; alkane_8) x (def2-SVP L=2; def2-TZVP L=3; def2-QZVP L=4), on one thread:

1. A-build cost per grid point (ferric_integrals::cosx_a, existing Stage 1/2
   harness), on >= 500 grid points drawn uniformly at random (fixed-seed LCG,
   no replacement) from the full (50,110) Becke grid. Unscreened AND with the
   Stage 2 shell-pair screen at 1e-7. "Per K build" = per-point cost x full
   (50,110) point count; this is a MEASURED per-point cost times a KNOWN
   count, not an extrapolation in nbf.
2. Analytic K via LinK (Schwarz bound, thresh 1e-12 = production
   integral_thresh) on a converged RHF density: warm build wall time
   (engine pool already built, DensityPairs rebuilt inside the timed region,
   which is what an SCF iteration pays).
3. DF-K (df_k.rs, def2-universal-jkfit): warm build wall time for BOTH the
   density path (`build`) and the occupied path (`build_from_occ`, what the
   SCF actually uses once MOs exist), plus the resident dressed 3-index
   tensor size naux*nbf^2*8 B. Where that exceeds ~2.5 GB the SIZE is the
   finding and the timing is skipped.

All timed segments run inside a 1-thread rayon pool with OPENBLAS_NUM_THREADS=1,
so every number is thread-invariant and A/K ratios do not depend on how many
cores each side happened to get. /proc/pressure/memory `full avg10` is read
before and after every timed segment and printed next to it; any segment with
nonzero full avg10 is discarded and rerun.

The SCF density is converged with DF-JK (jkfit aux) on the default pool; the
density's provenance cancels out of every ratio because all three builders
contract the SAME matrix.

## Hypotheses

H_phys (the literature's L-argument is real in ferric): ratio(A-build / LinK-K)
FALLS monotonically SVP -> TZVP -> QZVP, by a large factor (the design study's
own butane cc-pVXZ numbers imply ~5x smaller at QZ than TZ; ORCA quotes ~2x at
TZ vs ~16x at QZ for COSX's advantage, i.e. ~8x movement). Mechanism: analytic
K per shell quartet grows steeply with L (4 shells of high L, Rys/HGP quartet
cost ~L^4..L^6) while a 3c1e A block over a shell pair grows as ~L^2..L^3, so
K cost per nbf^2 rises much faster with L than A cost per point per nbf^2.

H_art (my harness is broken, or the argument does not apply to ferric's
libint2-backed 3c1e path): the ratio is FLAT or RISING with L. A specific
artifact route: libint2's nuclear-attraction engine at high L may be as slow
per shell pair as its ERI engine per quartet is, so the L penalty lands on both
sides equally.

Decision rule, fixed now: if the QZVP ratio vs LinK is single-digit, the earlier
closure was measured in the wrong regime and the engine question REOPENS. If it
stays >= 30x, the closure stands and now covers the L axis. In between (10-30x)
is "moved but not reopened" and gets reported as such with no tuning either way.

Secondary (vs DF-K): expected to be a SMALLER ratio than vs LinK at every L
only if DF-K's O(naux*nbf^2*nocc) GEMMs are slower than LinK here; at butane
size DF-K is probably faster than LinK, so the A/DF-K ratio is expected to be
LARGER than A/LinK. The memory column is expected to be the real DF-K story
(alkane_8/QZVP tensor ~7.3 GB, above the box).

## Expected artifact-vs-physics separation

Both H_phys and H_art are distinguishable by the two per-nbf^2 growth rates
reported alongside: if K-cost/nbf^2 rises steeply with L while A-cost/pt/nbf^2
is roughly flat, H_phys; if both rise together, H_art (or genuinely no
advantage). Exactness anchors for cosx_a (esp_at_points reconstruction,
zero-threshold == unscreened) already exist in cosx_a_anchor.rs and are run
again in release before the sweep.
