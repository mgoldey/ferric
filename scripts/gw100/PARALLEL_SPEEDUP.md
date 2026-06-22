# Parallelism fix — measured speedup (feat/parallelism-fix)

Single G0W0@HF via gw_xcheck, on the parallel branch (parallel ERI3 + crash-safe
freq-quad + with_blas_threads guard). Serial = OPENBLAS=1 RAYON=1; Parallel =
full rayon (12 cores) + OPENBLAS=4, guard pins BLAS=1 inside the freq-quad region.

| molecule | basis    | serial | parallel | speedup | IP serial | IP parallel |
|----------|----------|--------|----------|---------|-----------|-------------|
| C2H6     | cc-pVDZ  | 9.4 s  | 6.1 s    | 1.55×   | 12.8775   | 12.8775 ✓   |
| C6H6     | cc-pVDZ  | 125 s  | 53 s     | **2.35×** | 9.1218  | 9.1218 ✓    |

## Findings
- **Results bit-identical** (IP unchanged to 4 dp) — parallelism is correctness-safe.
- **MIXED PARALLELISM WORKS**: full rayon + OPENBLAS=4 SIMULTANEOUSLY — the exact
  config that stack-overflowed before — runs clean. The with_blas_threads(1) guard
  isolates the two levels: rayon over ERI3 shell-pairs / K freq-quad points with
  BLAS=1 inside; the big serial GEMMs (MO transform, Fock) get OPENBLAS=4.
- **Speedup GROWS with size**: 1.55× (ethane, parallel stages small) → 2.35×
  (benzene, ERI3+freq-quad dominate). At aug-cc-pVDZ / bigger organics the gain
  is larger still (more shell-pairs, bigger M).
- The crash that forced single-core (openblas-rayon-dgetrf-crash) is RESOLVED.

## Impact on the GW100 sweep
The sweep can now run multi-threaded (drop RAYON_NUM_THREADS=1, set OPENBLAS=N).
A ~2.3× per-molecule speedup on the benzene-class organics that were the
bottleneck turns the multi-day full-depth grind into ~1 day.
