# S3 kernel swap — results (2026-09-07, branch feat/cosx-s3-swap-screen)

Pre-registration: `scripts/queue/out/s3_swap_prereg.md` (commit c870983d,
written and committed before the backend was wired and before any number
below). Code: 26b15f2c (CosxK md3c1e backend + anchor), ee2a2480 (CLI knob).
Nothing in the kernel (`crates/ferric-integrals/src/md3c1e.rs`) was touched;
sub-batch size 256 was fixed in the prereg and not tuned.

## Exactness first

`cosx_k_md3c1e_matches_cosx_a_backend` (tests/cosx_k_anchors.rs), water/cc-pVDZ
converged RHF density, (25,50), backend the only difference:

| path | fit | max abs(K_md3c1e - K_cosx_a) | max abs(K) |
|---|---|---|---|
| build (density)   | on  | 1.998e-15 | 9.767 |
| build_from_occ    | on  | 9.992e-16 | |
| build (density)   | off | 1.998e-15 | |
| build_from_occ    | off | 8.882e-16 | |

Bar 1e-12. Both backends carry the same grid error vs the analytic K (asserted
to 1e-12). Mutation proofs, applied by hand to `accumulate_pair` and reverted
(anchor re-run green at 1.998e-15 afterwards):

* negate the kernel contribution (`y -= a*b`, the sign trap): max abs(dK) =
  **1.953e1** = 2 * max abs(K) -> RED.
* drop the `s1 != s2` mirror (`if s1 == usize::MAX`): max abs(dK) =
  **9.775e-1** -> RED.

The four prior anchors pass with the md3c1e default and the SAME numbers as
before the swap (dense-grid limit: plain/fitted 2.009e-3/9.636e-4 at (25,50),
4.7587e-5/5.6001e-5 at (50,110), 3.275e-7/2.849e-7 at (99,302); screen-0 ==
unscreened; build_from_occ == build; no leaked state). `tests/cosx_scf.rs`
(4 tests) passes: water/cc-pVDZ direct E = -76.0267720534 (14 it) vs
COSX(50,110)+fit -76.0267671912 (14 it), dE = +4.862e-6 Ha; refinement
3.037e-4 -> 4.862e-6 -> 3.659e-8 Ha. Unit tests (6, incl. a toy-layout check of
the fold vs a dense `A F`), clippy, rustdoc -D warnings, complexity gate: clean.

## Measurement A: full COSX K, butane/def2-QZVP (nbf 528, nocc 17, 77000 pts)

Harness `crates/ferric-scf/tests/cosx_full_k.rs`, density loaded from
`/tmp/cosx_butane_qzvp.dens` (converged DF-JK density of the earlier process),
fresh 1-thread rayon pool, OPENBLAS_NUM_THREADS=1, release, fit on,
unscreened (pairs kept 1093092000 / 1093092000), `scripts/ferric-limited
--max=4G --high=3600M`, FERRIC_MEM_BUDGET_GB=2 (budget 2.15 GB). PSI full
avg10 = 0.00 before AND after; cpu 136.84 s vs wall 137.04 s (0.1%).

| segment | cosx_a backend (handed-over baseline) | md3c1e backend (this run) |
|---|---|---|
| A-build (kernel) | 391.65 s (97.4%) | **122.89 s** (89.7%) |
| G = A F contraction | (in the 10.44 s remainder) | 8.86 s |
| AO-eval on grid | | 1.38 s |
| block GEMMs (F = D X, Ktilde += X G^T, S_num) | | 3.48 s |
| overlap-fit finalize | | 0.02 s |
| **total wall** | **402.09 s** | **137.04 s** |
| per-point kernel | 5.086e-3 s/pt | 1.596e-3 s/pt |

* Speedup of the full K: 402.09 / 137.04 = **2.93x**.
* vs LinK K on the same density (255.85 s, handed over): 137.04 / 255.85 =
  **0.536x** — COSX K is now cheaper than the analytic K at QZVP.
* The per-point kernel time (1.596e-3 s/pt) reproduces the standalone bench at
  B=256 (1.6003e-3, md3c1e_results.md) to 0.3%: the batched speedup transferred
  from the 2000-point sample to the full 77000-point sweep without loss.

Prereg vs data: expected total ~132 s, band 120-160 s, ratio band 0.47-0.63x;
measured 137.04 s / 0.536x — INSIDE the band. Expected kernel ~122 s: measured
122.89 s. Expected accumulation 10-25 s: measured 8.86 s (slightly below the
band — the strided axpy over 256-wide rows vectorizes well). Neither artifact
hypothesis fired (>200 s: no; <100 s: no; screened: no).

## Measurement B: full RHF SCF, butane/def2-TZVP (nbf 184)

`target/release/ferric` on `scripts/queue/out/s3_butane_tzvp_{cosx,direct}.toml`
(identical except `[scf] k_builder = "cosx"`, `cosx_grid = {50,110}`,
`cosx_overlap_fit = true`, `cosx_backend = "md3c1e"`), energy_conv 1e-8,
density_conv 1e-7, integral_thresh 1e-12, RAYON_NUM_THREADS=1,
OPENBLAS_NUM_THREADS=1, same ferric-limited wrapper and budget. Wall is the
whole process (`date` around the invocation).

| builder | E (Ha) | iterations | converged | wall (s) | PSI before / after |
|---|---|---|---|---|---|
| direct 4-centre J+K | -157.3532854293 | 12 | yes | 98.01 | 0.00 / 0.00 |
| COSX md3c1e (50,110)+fit, direct J | -157.3534010150 | 12 | yes | **358.42** | 0.00 / 0.00 |
| (COSX, first attempt) | -157.3534010150 | 12 | yes | 364.75 | 0.00 / 0.00 but avg60 = 0.45 after -> discarded (contention), rerun above |

* E_cosx - E_direct = **-1.156e-4 Ha** (0.073 kcal/mol absolute) at identical
  iteration counts and an identical convergence trace shape. Inside the prereg
  band (1e-5..1e-3 Ha absolute); the (50,110)+fit operating point was chosen
  on isodesmic DIFFERENCES (0.02 kcal/mol), absolute errors are larger.
* Wall: COSX 3.66x the direct SCF at TZVP. Per iteration: direct J+K ~8.2 s,
  COSX ~29.9 s of which the kernel is ~22-23 s (77000 pts x 2.93e-4 s/pt from
  the TZVP bench) plus the direct J — exactly the prereg's "2-3x slower at
  TZVP, the crossover is at QZVP". Not a speed claim; a robustness check of
  the SCF loop on the new backend, which converged in the same 12 iterations.
* butane/def2-QZVP SCF: NOT run, as pre-registered. Direct JK is ~400 s per
  iteration on one thread, the COSX SCF would be 12 x (137 s + direct J) > 40
  min; neither fits the 8-minute foreground window and there is no SCF
  restart-from-density in the CLI. Its K-build ratio is Measurement A.

## What was wired

* `ferric_scf::cosx_k::CosxBackend { Md3c1e (default), CosxA }` with strict
  `parse_config_str` ("md3c1e" | "cosx-a"; anything else errors) and
  `CosxConfig::backend`.
* `CosxK`: per-worker state is now `Workers::{CosxA(engines), Md3c1e(scratch)}`
  (generic `ThreadSlots<T>` replaced `NuclearPool`); `contract_block` dispatches
  to `contract_block_cosx_a` (unchanged numerics, returns the same memory as
  a transposed view) or `contract_block_md3c1e` (parallel over fixed 256-point
  sub-batches, `Md3c1e::for_each_pair` + `accumulate_pair` fold with the
  `s1 != s2` mirror, kernel/accumulation timing split via in-callback timers).
  Budget check now sizes the per-thread footprint per backend.
* CLI `[scf] cosx_backend = "md3c1e" | "cosx-a"`: dead-knob (k_builder != cosx)
  and unknown-value hard errors, tested in `cosx_knobs_resolve_strictly`;
  `examples/water-rhf-cosx.toml` carries the key and still parses.
* Untouched: ferric-integrals, the `screen_thresh > 0` refusal, the
  `cosx_a_screen_is_unsound_tripwire` test (still fires: 15/21 same-centre
  pairs dropped at 1e-7).

## Verdict (dated, provisional)

With the md3c1e backend the full unscreened COSX K at butane/def2-QZVP costs
137 s = 0.54x one LinK K build (was 1.57x with the libint2 path), exact vs the
libint2 path to 2e-15, bit-identical across thread counts by construction,
and the SCF loop converges on it in the same iteration count as direct. The
remaining 90% of the K is the kernel itself; the attribution in
md3c1e_results.md (Boys/exp ~35-58% at QZVP) says where the next factor is,
and that is a ferric-integrals change, not an SCF one.
