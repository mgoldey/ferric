# Issue #507: RS-GDF long-range GEMM growth, dryice to dryice_112

Raw records: `bench_results/lr_chunk_507.jsonl` (one line per run: stage wall/CPU,
counters, energy, load before/after). Inputs: `toml_ri/dryice_rhf.toml`,
`toml_ri/dryice_112_rhf.toml`, `toml_ri/dryice_112_rhf_auto.toml` (cc-pVDZ /
cc-pvdz-ri, Gamma RHF, ferric defaults, gdf_omega 1.0 unless noted).
Protocol: release CLI, `OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=6`, both heavy slots
held per run (`flock -o` on both compile locks); no quiet-wait gate, so the
1-minute load average was 4.4-9.7 when a run started (one foreign desktop process).
"main" = origin/main 8e4ee762 plus a temporary env hook that overrides the LR chunk
cap (unset it is main's 64 MiB cap); "fix" = this branch.

## Measurement

Sizes (counters of the build):

| cell | nao | naux | LR half-G | LR chunks (main) | G per chunk (= GEMM k) |
|------|-----|------|-----------|------------------|------------------------|
| dryice | 168 | 672 | 13372 | 189 | 71 |
| dryice_112 | 336 | 1344 | 26667 | 1482 | 18 |

GEMM shapes per chunk (J3: `(nao², k) x (k, naux)`, two terms re/im, row-blocked by
512 rows into 56 / 221 single-threaded BLAS calls per term; J2: `(naux, k) x (k, naux)`,
two terms). Only one chunk at a time; the aux FT is computed once per chunk (no
per-q / per-k recomputation on Gamma: aux FT is 0.09 s / 0.46 s).

Thread utilisation of the GEMM stage (sum of task walls / wall): dryice 5.3 of 6,
dryice_112 5.7 of 6. Both cells keep all threads busy.

Main, 3 reps each (medians, wall s):

| cell | total | LR pair FT | LR aux FT + GEMM | J3 + J2 GEMMs | GEMM GFLOP/s |
|------|-------|------------|------------------|---------------|--------------|
| dryice | 20.45 | 7.06 | 5.76 | 5.06 | 200 |
| dryice_112 | 317.0 | 45.9 | 221.7 | 217.1 | 75 |

(GFLOP/s = 4 nao² naux nG_half / GEMM wall: 1.01e12 and 1.62e13 flop, ratio 16.0.)

Chunk cap sweep on dryice_112 (1 rep each, main + env hook):

| cap | chunks | k | GEMMs | pair FT | total |
|-----|--------|---|-------|---------|-------|
| 64 MiB (main) | 1482 | 18 | 216.4 | 44.8 | 315.3 |
| 256 MiB | 371 | 72 | 76.7 | 31.0 | 161.3 |
| 480 MiB | 197 | 135 | 75.1 | 30.2 | 159.5 |
| 1024 MiB | 92 | 290 | 69.0 | 27.0 | 150.4 |

Same sweep on dryice: 64 MiB (k 71) GEMMs 4.99 s, 256 MiB (k 285) 5.16 s, 1024 MiB
(k 1114) 6.22 s (load rose to 8 during that run).

## Interpretation

nao, naux and the half-G count each double, as the 16x cost model assumes. What
does not stay fixed is the GEMM inner dimension: the LR pair-FT chunk is capped at
`G_CHUNK_BYTES` = 64 MiB and one G vector costs ~32 nao² bytes (pair FT plus its
re/im packing), so k fell from 71 to 18. Every chunk re-reads and re-writes all of
J3 (nao² x naux: 152 MB on dryice, 1.2 GB on dryice_112), so with 8x the chunks on
an 8x larger J3 the J3 traffic grows 64x while the flops grow 16x. The GEMMs drop
from 200 to 75 GFLOP/s with all threads busy, which fits a memory-bound stage rather
than an idle one. At a fixed k of ~72 (256 MiB cap) the dryice_112 GEMMs are 15.2x
dryice's, in line with the 16x model, which supports this reading.

## Fix

`rsgdf::lr_chunk_budget`: the Gamma RS-GDF LR chunk is sized for at least
`LR_GEMM_MIN_K` = 64 G vectors (`max(G_CHUNK_BYTES, 64 x bytes per G)`, still capped
by the ledger's unreserved budget; a constant, never thread-derived). When 64 G
vectors fit in 64 MiB (nao up to about 170 at cc-pVDZ, dryice included) the budget
is unchanged, and so are the chunks and the bits. dryice_112 now runs 417 chunks
(k = 64).

Memory: the fix trades memory for GEMM width. On dryice_112 the transient LR chunk
buffer grows from 64 MiB to about 236 MB (64 G vectors at its per-G size). Peak RSS
was not measured. The raised budget is still capped by the ledger's unreserved
budget, so when that cap binds the chunk falls back below k = 64 and the stage is
memory-bound again.

Fix vs main, 3 interleaved reps each (medians, wall s):

| cell | binary | total | LR pair FT | LR aux FT + GEMM | GEMMs | energy (Ha/cell) |
|------|--------|-------|------------|------------------|-------|------------------|
| dryice | main | 20.45 | 7.06 | 5.76 | 5.06 | -750.821118438244 |
| dryice | fix | 20.51 | 7.11 | 5.80 | 5.11 | -750.821118438244 (identical) |
| dryice_112 | main | 317.0 | 45.9 | 221.7 | 217.1 | -1501.4642014457143 |
| dryice_112 | fix | 164.1 | 31.5 | 84.5 | 80.2 | -1501.4642014457177 |

dryice_112 |dE| = 3.4e-12 Ha/cell (2.3e-15 relative, from the changed GEMM k-ranges;
the issue's gate is 1e-8). Fix rep 2 ran under a load of 9.7 (187.5 s total); the
other two were 164.0 and 164.1 s. The dryice_112 / dryice GEMM ratio is now 15.9x (was 43x).
The LR part (pair FT + sink) is now 116 s vs 12.8 s, 9.1x for the 2x cell.

The chunking-invariance anchor `rsgdf_lr_sums_are_invariant_to_the_g_chunking`
(`tests/pbc_parallel_bitwise.rs`, 8 chunks vs 1) measures J3 8.4e-16 and J2 1.3e-15
relative.

## gdf_omega = "auto" on dryice_112

auto picks 0.35 Bohr^-1. The LR half-G drops to 1140 and the SR 3-centre grows to 18 s.
1 run each: main total 72.6 s (LR pair FT 2.39, aux FT + GEMM 9.59 at 64 chunks, k 18);
fix total 65.4 s (pair FT 1.66, aux FT + GEMM 3.57 at 18 chunks, k 63). Energy
-1501.4642014456995 (fix), 1.8e-11 from the default-omega energy.

## Not changed (open)

The same 64 MiB `G_CHUNK_BYTES` cap feeds the k-point RS-GDF LR stage (`crates/ferric-pbc/src/rsgdf/kpoint.rs`,
`crates/ferric-pbc/src/rsgdf/kpoint/klr.rs`) and the gradient and stress builds
(`crates/ferric-pbc/src/rsgdf/deriv.rs`, `crates/ferric-pbc/src/rsgdf/strain.rs`,
`crates/ferric-pbc/src/rsgdf/kpoint/kderiv.rs`). Those builds were not measured here.
