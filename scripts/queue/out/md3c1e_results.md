# md3c1e kernel measurement — results (2026-09-07, branch feat/3c1e-md-kernel)

Pre-registration: `scripts/queue/out/md3c1e_prereg.md` (commit 83acef31,
written and committed before any number below was taken). Kernel: commit
1731be6c (`crates/ferric-integrals/src/md3c1e.rs`). Harness:
`crates/ferric-integrals/tests/md3c1e_bench.rs` (ignored test, one process
per cell). Nothing in the kernel was changed between the pre-registration and
these numbers (TILE = 8, Boys table spacing 1/8, 10 Taylor terms).

## Protocol actually followed

* Butane (`alkane_4.xyz`, 14 atoms) x def2-SVP / TZVP / QZVP (bundled).
  2000 points drawn uniformly at random without replacement (LCG seed
  20260907, the L-axis sampler) from the (50,110) Becke grid, positions
  rebuilt from the TA-M4 radial formula + Lebedev-110; count asserted 77000.
* Preflight in every cell: batched md3c1e vs `cosx_a` on the first 3 sampled
  points, max|diff| 1.1e-15 / 8.9e-15 / 4.9e-15 (SVP/TZVP/QZVP) against a
  1e-12 abort.
* cosx_a: `a_matrix_at_point_with`, ONE reused libint2 nuclear engine,
  unscreened. md3c1e: `Md3c1e::for_each_pair` unscreened at B = 64 / 256 /
  1024 with a checksum consumer touching every block; drop-in single-point
  path on 200 points.
* Single thread (no rayon; OPENBLAS_NUM_THREADS=1), release, AVX2+FMA path
  active (`fma=true`). Every timed segment: PSI `full avg10` = 0.00 before
  AND after; cpu-seconds == wall to <1%. No segment was discarded.

## Table (per grid point, unscreened)

| basis | L | nbf | nsh | prim pairs/pt | cosx_a s/pt | md3c1e B=64 | B=256 | B=1024 | drop-in | speedup B=64/256/1024 | op count/pt | GF/s md (B=256) | GF/s libint2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| def2-SVP  | 2 | 106 | 54  | 4,466  | 4.4265e-4 | 1.3420e-4 | **1.2576e-4** | 1.3002e-4 | 1.791e-3 | 3.30 / **3.52** / 3.40 | 90,972    | 0.72 | 0.21 |
| def2-TZVP | 3 | 184 | 84  | —      | 1.0359e-3 | 3.1338e-4 | **2.9289e-4** | 2.9355e-4 | 4.670e-3 | 3.31 / **3.54** / 3.53 | 363,962   | 1.24 | 0.35 |
| def2-QZVP | 4 | 528 | 168 | 26,655 | 5.1457e-3 | 1.8186e-3 | **1.6003e-3** | 1.6120e-3 | 3.267e-2 | 2.83 / **3.22** / 3.19 | 4,959,385 | 3.10 | 0.96 |

cosx_a per point reproduces the L-axis harness (0.438 / 1.026 / 5.09 ms
there vs 0.443 / 1.036 / 5.146 ms here, 1-2%), so the two measurements are
on the same footing.

Consequence for the L-axis verdict: the A-build was 1.51x one LinK K build
at butane/QZVP (l_axis_results.md). At 3.22x per point that becomes
**0.47x** one LinK K build (0.40x of ferric's default direct J+K) — the
unscreened A-build is now cheaper than the analytic K at QZ.

## Pre-registered hypotheses vs data

* Expected SVP 6-12x / TZVP 4-8x / QZVP 3-6x, falling with L: **QZVP inside
  the band (3.22x); SVP and TZVP BELOW their bands (3.5x); the fall with L
  is weak (3.52 -> 3.54 -> 3.22).** The parity rule (< 1.5x at QZVP = does
  not beat parity) is cleared by 2.1x.
* Artifact checks: (a) speedup grows B=64 -> 256 (+7% SVP/TZVP, +14% QZVP),
  flat 256 -> 1024 — the per-batch setup IS amortized by B=256 and the
  layout claim holds; (b) GF/s 0.7-3.1, nowhere near the >40 undercount
  flag; (c) the drop-in path is 4-20x SLOWER than batched, as predicted.
  None of the three artifact hypotheses fired.
* Expected 4-12 GF/s at TZ/QZ: **not reached (1.2 / 3.1)**. The reason is
  measured below, not guessed.

## Where the time goes (perf is root-only on this box; measured by parts)

`md3c1e_boys_cost_attribution` (same binary): the table Boys evaluator costs
20.5-22 ns (n_max = 0), 27.5 ns (4), 35 ns (8) per (primitive pair, point),
dominated by one libm `exp(-T)` per call. Every surviving primitive pair
costs exactly one such call per point, and the FLOP count excludes it:

| basis | prim pairs/pt | Boys cost/pt (n_max 0 / 4 / 8 evaluator) | measured md3c1e/pt | Boys share |
|---|---|---|---|---|
| def2-SVP  | 4,466  | 98 / 122 / 158 us   | 126 us  | ~80-100% (s/p-dominated: essentially all of it) |
| def2-QZVP | 26,655 | 0.55 / 0.73 / 0.93 ms | 1.60 ms | ~35-58% |

So at SVP the kernel is Boys/`exp`-bound and the FMA work is invisible; at
QZVP roughly half the time is Boys and the other half is R-tensor +
contraction running at an effective ~5-7 GF/s (4.96 Mflop in ~0.7-1.0 ms).
This is why the speedup is flat in L instead of falling as the spec
predicted: the spec's model had libint2's per-call overhead as the thing
being removed and the contraction as the residual; in fact the residual
per-primitive-pair scalar cost (exp + Boys + PC/T) is the same order as
libint2's, and the FMA-bound part only becomes visible at L = 4.

## Operation-count convention (the GF/s columns are NOT comparable to the spec)

For water/cc-pVDZ (12 segmented shells, 78 pairs — the spec's own case) this
kernel's count is R-tensor 2,489 / contraction 5,168 / total **7,657** per
point vs the spec's 13,360 / 27,801 / **41,161**: 5.4x lower for the same
shell-pair list. My convention: R-tensor 3 flops per two-source recursion
element (1 for one-source) + seeding, contraction 2 flops per NONZERO
`E_t E_u E_v` coefficient restricted to `t <= ax+bx` etc. The spec does not
state its per-element convention; it evidently counts a denser structure.
So the libint2 figures here (0.21 / 0.35 / 0.96 GF/s) are not the spec's
0.56 / 1.04, and neither set should be quoted against the other. The
speedup RATIO (3.2-3.5x) is convention-free and is the deliverable.

## Not done (deliberately — no tuning after the first number)

Concrete, exactness-preserving next steps the attribution points at, for a
SEPARATELY pre-registered round:
1. Remove the libm `exp` from the Boys hot path: store `exp(-T_k)` at the
   table nodes and Taylor-expand `exp(-T) = exp(-T_k) exp(delta)` (6 terms,
   |delta| <= 1/16) — ~20 ns -> ~3 ns per call; at SVP this alone is most
   of the remaining time.
2. Vectorize the Boys/PC/T stage over the tile (gathered table rows) so the
   per-primitive-pair fixed cost also runs 4-wide.
3. Primitive-pair x tile screening (skip a pair on a tile when
   `|pref| K_AB F_0(T_min)` is below a threshold), WITH its trivial-limit
   anchor — this changes exactness and needs the threshold=0 == unscreened
   test before any sweep.
4. Share E-tables across general-contraction siblings (same exponents).

## Verdict (dated, provisional)

The MD kernel is exact vs cosx_a (<= 8.4e-15 scaled over all 15 (l_a,l_b)
combinations incl. g-g, T=0 on nuclei, 1e-8 Bohr off-nucleus, 200 Bohr) and
3.2-3.5x faster per grid point than libint2's path on butane at every L,
which clears the pre-registered parity bar at QZVP (3.22x >= 1.5x) and turns
the A-build from 1.51x to ~0.47x of one LinK K build there. It did not reach
the pre-registered upper expectation at DZ/TZ, and the measured reason is a
scalar per-primitive-pair Boys/`exp` cost that the spec's FLOP model does
not contain.
