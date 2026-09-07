# S3 kernel swap — pre-registration (2026-09-07, branch feat/cosx-s3-swap-screen)

Written and committed BEFORE the md3c1e backend was wired into `CosxK` and
before any full-K or SCF number with it was taken. Baseline numbers below are
the ones handed over from the cosx_a-backed builder (HEAD 6ed30beb).

## What changes

`crates/ferric-scf/src/cosx_k.rs` gets a second A-build backend: instead of
"per grid point: `A^g` via `cosx_a::a_matrix_at_point_with` (libint2), then
`G_g = A^g F_g` (GEMV)", the per-block loop becomes "per sub-batch of 256
points: `Md3c1e::for_each_pair` over shell pairs `s1 >= s2`, accumulating
`Y[o1+i][g] += blk[i][j][g] F[o2+j][g]` and the `s1 != s2` mirror
`Y[o2+j][g] += blk[i][j][g] F[o1+i][g]` directly from the kernel block".
Everything else (X, F = D X, Ktilde += X G^T, symmetrize, overlap fit, block
partition, grid) is unchanged. Both backends stay selectable
(`CosxConfig::backend`, default md3c1e) so the anchor below is runnable forever.
Unscreened only; `screen_thresh > 0` stays refused.

## Exactness anchor (must pass before any timing)

`cosx_k_md3c1e_matches_cosx_a_backend` (tests/cosx_k_anchors.rs): water/cc-pVDZ
converged RHF density, identical `CosxConfig` except the backend, (25,50) grid,
fit on: `max|K_md - K_cosxa| <= 1e-12`. Mutation tests to be applied by hand and
recorded: (i) negate the kernel output inside the accumulation (sign trap) ->
must FAIL; (ii) drop the `s1 != s2` mirror -> must FAIL. The four existing
anchors (dense-grid limit, screen zero == unscreened, build_from_occ ==
build, no leaked state) and `tests/cosx_scf.rs` must pass unchanged with the
new default backend.

## Measurement A: full COSX K, butane/def2-QZVP (nbf 528, 77000 points)

Harness `crates/ferric-scf/tests/cosx_full_k.rs`, density loaded from
`/tmp/cosx_butane_qzvp.dens` (the converged DF-JK density from the earlier
process), 1 rayon thread, OPENBLAS_NUM_THREADS=1, release, fit on, unscreened,
under `scripts/ferric-limited --max=4G --high=3600M` with FERRIC_MEM_BUDGET_GB=2.
PSI `full avg10` must read 0.00 before AND after; otherwise the number is
discarded and rerun.

Baseline (cosx_a backend, same harness, same density): total 402.09 s, of
which A-build 391.65 s (97.4%); LinK K on the same density 255.85 s
=> COSX/LinK 1.57x.

Expected with md3c1e (fixed now):
* A-build (kernel time, excluding the accumulation) ~ 391.65 / 3.22 = **~122 s**
  (the batched per-point speedup measured on 2000 random points transfers to
  the full 77000-point sweep; B = 256 sub-batches).
* Accumulation ("contract") ~ 10-25 s: 2 nbf^2 flops per point = 0.56 Mflop/pt
  vs the kernel's 4.96 Mflop/pt, but it is a strided axpy, not a GEMM.
* AO-eval + block GEMMs + fit unchanged, ~10 s.
* **Total ~132 s (band 120-160 s) => COSX/LinK ~0.5x (band 0.47-0.63x).**

Artifact hypotheses:
* If the total lands above ~200 s, the batched speedup did NOT transfer from
  the 2000-point sample to the full sweep (cache footprint of Y + F + block at
  B=256, or the callback dominating). Report the split; do not tune before
  reporting.
* If the total lands below ~100 s, work is being skipped — re-run the anchor
  on THIS build before believing it.
* If `pairs_kept != pairs_total` the run was screened; discard.

## Measurement B: full RHF SCF, butane/def2-TZVP (nbf 184)

`ferric-cli` TOML, `[scf] k_builder = "cosx"` (md3c1e, (50,110), fit on) vs the
default direct JK, otherwise identical config, 1 thread, same ferric-limited
wrapper. Report converged energy, iteration count, total wall for each.

Expected:
* |E_cosx - E_direct| in the 1e-5..1e-3 Ha range (water/cc-pVDZ was ~5e-5 Ha;
  butane has 4x the electrons and def2-TZVP is a larger basis; the reaction-
  energy audit put the (50,110)+fit error at 0.02 kcal/mol = 3e-5 Ha on
  isodesmic differences, absolute errors are larger).
* Iteration counts within +-3 of each other.
* Wall: COSX is expected to be SLOWER than direct at TZVP. The L-axis table has
  the cosx_a A-build at 79 s per K at TZVP => ~22 s with the kernel, plus the
  direct J every iteration; direct JK at TZVP is ~10 s per iteration. So COSX
  SCF ~2-3x the direct SCF wall at TZVP. The crossover the kernel buys is at
  QZVP (Measurement A), not TZVP — this run is a correctness/robustness check
  of the SCF loop with the new backend, not a speed claim.
* butane/def2-QZVP SCF: direct JK is ~400 s PER ITERATION on one thread
  (l_axis_results.md), so a direct SCF is ~1.5-2 h and even the COSX SCF is
  ~15 x (132 + J) s > 40 min. Neither fits the 8-minute foreground window and
  there is no SCF restart-from-density in the CLI, so the QZVP SCF will NOT be
  run; this is stated up front, not discovered.

No tuning toward the target after the first number: the kernel is untouched
(ferric-integrals is not mine to modify), sub-batch B = 256 is fixed before
the first run, and a result outside the band is reported as such.
