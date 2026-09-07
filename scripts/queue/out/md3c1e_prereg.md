# md3c1e kernel measurement — pre-registration (2026-09-07, branch feat/3c1e-md-kernel)

Written and committed BEFORE any timing of the kernel was taken. The kernel
(`crates/ferric-integrals/src/md3c1e.rs`, commit 1731be6c) had at this point
only been run for correctness (anchor vs `cosx_a`, ≤ 8.4e-15 scaled) and for
the mutation tests; no wall-clock number of it exists yet.

## What is being measured

Butane (`testdata/molecules/alkane_4.xyz`, 14 atoms) × {def2-SVP (L=2, nbf
106), def2-TZVP (L=3, 184), def2-QZVP (L=4, 528)}, one process per cell,
harness `crates/ferric-integrals/tests/md3c1e_bench.rs` (ignored test):

1. **cosx_a per point** — `ferric_integrals::cosx_a::a_matrix_at_point_with`
   with ONE reused libint2 nuclear engine, unscreened, over N ≥ 2000 grid
   points drawn uniformly at random without replacement (fixed-seed LCG, seed
   20260907, the same sampler as the L-axis harness) from the full (50,110)
   Becke grid (77000 points on butane; positions rebuilt from the TA-M4
   radial formula + the 110-point Lebedev set, count asserted).
2. **md3c1e batched per point** — `Md3c1e::for_each_pair` over the SAME
   points in batches of B = 256 (also reported at B = 64 and 1024), unscreened,
   with a trivial consumer that touches every block (a checksum) so nothing
   is dead-code-eliminated. Reported as s/point = wall / N.
3. **md3c1e single-point drop-in** (`a_matrix_at_point_with`) per point, for
   completeness — expected to be much slower than the batch path because it
   pays the primitive-pair setup for a single padded tile.
4. **Achieved GFLOP/s** for the batched kernel = `Md3c1e::flops_per_point()
   / (s/point)`, operation count in the spec's convention (R-tensor: 3 flops
   per two-source recursion element, 1 per one-source, plus seeding;
   contraction: 2 flops per nonzero `E E E` coefficient; Boys and cart→sph
   excluded). The same count divided by cosx_a's s/point gives libint2's
   effective throughput for comparison with the spec's 0.56 / 1.04 GFLOP/s
   (water DZ/TZ). The harness also prints the count for water/cc-pVDZ next to
   the spec's 13,360 / 27,801 / 41,161 so the convention can be compared.

Protocol: single thread (the harness runs no rayon; `OPENBLAS_NUM_THREADS=1`
is set), release build, `/proc/pressure/memory` `full avg10` printed before
and after every timed segment; any segment with nonzero PSI is discarded and
rerun. Process CPU-seconds printed next to wall (cpu ≈ wall confirms no
contention). A preflight compares the batched kernel against cosx_a on the
first 3 sampled points to 1e-12 before anything is timed, so a wrong kernel
cannot produce a speedup.

## Hypotheses (fixed now)

The spec's realistic estimate (§7, 4 GFLOP/s scalar-equivalent) is **7.1x
at DZ and 3.8x at TZ** on water, falling with L because contraction FLOPs
grow faster than libint2's per-call overhead. This host is a Broadwell-E
(i7-6800K, AVX2+FMA, 3.4 GHz; peak ~54 GFLOP/s/core, realistic 20-30% of it
on short inner dimensions).

* **Expected**: SVP 6–12x, TZVP 4–8x, QZVP **3–6x** over cosx_a per point,
  monotonically falling with L; achieved throughput 4–12 GFLOP/s at TZ/QZ,
  lower at SVP where s/p pairs dominate and the per-tile fixed cost (Boys,
  `exp`, loop overhead) is a larger fraction.
* **Does NOT beat parity**: QZVP < 1.5x. The measured A-build is 1.51x one
  LinK K at QZVP (l_axis_results.md), so anything ≥ 1.5x at QZVP makes COSX
  the fastest exact-quality K there; below 1.5x the kernel has not changed
  the QZ verdict.
* **Artifact hypotheses**: (a) if the batched speedup does NOT grow from
  B = 64 to B = 256 the per-batch primitive-pair setup is not being amortized
  and the layout claim is wrong; (b) if GFLOP/s comes out above ~40 the
  operation count is undercounting (or the consumer let work be eliminated);
  (c) if the drop-in single-point path is FASTER than the batch path the
  batching is broken. Each of these is reported regardless.

No tuning toward the target after the first number: TILE = 8, B, and the
Boys table spacing are what is committed at 1731be6c. A 1.2x result honestly
reported is a complete deliverable.
