<!-- status: verify; registry: ri-mp2 -->

# Parallel performance, then and now {#parallel-performance}

This section has no equations. It records what the 2014 algorithm measured,
which design changes answer those measurements, and what the code has not
yet measured. The operation counts of sections @sec:four-step-algorithm to @sec:pair-kernel say where the
work is; only a timing says where the wall time goes.

## What the dissertation measured {#parallel-performance-dissertation-measured}

The test was strong scaling of RI-MP2/cc-pVDZ on linear polyglycines with 4,
8, 16 and 32 subunits, on one 12-core node, for the whole algorithm and for
Function 4 alone (chapter 4, section 4.3, Figure 4.1; the figure is not
redrawn here).

| Finding | Dissertation, chapter 4 |
|---|---|
| The share of the serial cost spent in Function 4 grows with size: about 60% at 4 subunits (308 basis functions) and 95% at 32 subunits (2296), from its Table 4.2 | section 4.3 |
| Parallel efficiency grows with size, because the parallelized step becomes the dominant one | section 4.3 and conclusion 1 |
| The lower-scaling steps parallelize poorly: the MO transform ran in auxiliary batches by shell, and the integral storage was seek-bound to follow the atomic order of the auxiliary index | section 4.3 |
| Recommendation: use whole nodes for large molecules, to minimize disk reads | conclusion 1 |
| On the application systems, RI-MP2/aug-cc-pVDZ cost less than 40% of the SCF, and RI-MP2/aug-cc-pVTZ on the tetrapeptides less than 10% | section 4.4 and conclusion 2 |

The first finding is the ratio of section @sec:four-step-algorithm-algorithm-code seen in wall time: step 4
over the dressing grows as $(n_o+1)n_v/(2N_{\mathrm{aux}})$, linearly in
the size. The per-system timings and energies of section 4.4 are not carried
here.

## How the code answers each finding {#parallel-performance-code-answers-each-finding}

| 2014 bottleneck | Change in the code | Section |
|---|---|---|
| Transform batched by auxiliary shell, single-threaded per batch | each auxiliary row of a chunk is a task above a work threshold; the dressing is one thread-count-independent `par_gemm` per chunk | @sec:streaming-transform-pass |
| Seek-bound storage of $(ia\vert P)$ and $B$ | neither is stored; only the raw AO tensor spills, read sequentially in blocks | @sec:four-step-algorithm-changed, @sec:memory-budget-run-holds |
| Function 4 reading $B$ from disk, batch $n_B$ tied to the core count | all of $B$ in memory; one GEMM task per $i$ over the $j \ge i$ tail | @sec:pair-kernel-batching-over-occupied-index |
| Reductions in thread-dependent order | per-$i$ partials summed in order on one thread; chunk width independent of the thread count | @sec:streaming-transform-reproducibility, @sec:pair-kernel-parallel-loop-fixed-summation |
| One node only | auxiliary bands across MPI ranks (library only) | @sec:distributed-rimp2 |

The first and third rows remove the reason the 2014 efficiency fell at small
sizes. Section @sec:pair-kernel-parallel-loop-fixed-summation derives a limit of the current design: with one task per
$i$, the pair stage cannot speed up by more than $(n_o+1)/2$, so a small
molecule on a wide node leaves workers idle in that stage. Whether the small
systems now scale well is a question for a measurement, below.

## What has not been measured {#parallel-performance-has-been-measured}

No strong-scaling or thread-count timing of the closed-shell RI-MP2 energy is
committed: `site/src/reference/validation.md` grades RI-MP2's accuracy, not
its parallel speed, and no test asserts a speedup. Comments in
`spin_components_from_b_ov_kappa_cpu` record a single-system profile of an
earlier version of the pair stage. Those numbers were not taken under the
measurement rules of the repository and are not quoted here.

A measurement that could support a statement in this section needs:

| Requirement | Reason |
|---|---|
| A series of sizes past the point where the pair stage dominates, with the tail fitted, not the whole series | the dissertation's own finding is that efficiency is size dependent |
| The transform and the pair stage timed separately | they have different parallel structure (@sec:streaming-transform, @sec:pair-kernel) |
| Fixed `[memory] budget_gb`, `[parallel] blas_threads = 1` and `OPENBLAS_NUM_THREADS=1` | the budget changes the chunk and the tier; the default serial BLAS count is the pool size, which moves the metric factor in the last bits (section @sec:determinism-and-memory-blas-thread-counts); BLAS threads inside rayon oversubscribe |
| The bits of the energy compared across thread counts | the committed bit-identity test (section @sec:pair-kernel-parallel-loop-fixed-summation) runs water/cc-pVDZ only; a timing series would extend it to the measured sizes |
| Runs on a quiet machine (`box-lease`, `scripts/README.md`) | timings taken beside other jobs are biased |

## Known limits (2026-11-15) {#parallel-performance-known-limits-2026-11}

- No committed timing of RI-MP2 parallel scaling.
- The committed test that the CPU energy is bit-identical across rayon pool
  sizes covers one small system (H2O/cc-pVDZ, pools of 1, 2 and 6), and only
  with the serial BLAS count at 1: it runs inside `pool.install`, where a
  rayon worker's serial count is 1. With the default serial count the energy
  moves in the last bits with the pool size.

## Sources

| Source | Used for |
|---|---|
| Dissertation, chapter 4, sections 4.3 to 4.5 (Figure 4.1, Table 4.2, conclusions 1 and 2) | section @sec:parallel-performance-dissertation-measured |
| `crates/ferric-mp2/src/rimp2.rs` | section @sec:parallel-performance-code-answers-each-finding |
| `crates/ferric-mp2/tests/mo_stream_dressing_threads.rs` | the bit-identity test |
| `site/src/reference/validation.md` | the absence of a graded timing |

Changes from the sources: none to equations; the dissertation's findings are
paraphrased, and only the two end points of its Table 4.2 are quoted.
