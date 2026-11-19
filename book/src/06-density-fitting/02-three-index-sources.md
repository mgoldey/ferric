# Three-index sources: in core, packed, spilled, recomputed {#three-index-sources}

<!-- status: edit; registry: none -->

The raw tensor $(P|\mu\nu)$ is the largest object most density-fitted methods
hold: $N_{\mathrm{aux}}N^2$ doubles. With $N = 500$ basis functions and
$N_{\mathrm{aux}} = 1500$ auxiliary functions that is
$1500 \times 500^2 \times 8$ bytes, 3.0 GB. An SCF reads it on every J pass,
two per Fock build. Whether it fits in memory decides how the run is
organized, and the answer changes with the molecule, the basis and the
machine.

ferric keeps that decision in one place. Every consumer reads the tensor
through one type, which decides at build time where the tensor lives and
serves it in blocks of auxiliary rows. Consumers apply their own metric
(section @sec:factorization-and-metric), so the same source serves RI-J, RI-K, RI-MP2 and RPA. This
section lists the storage tiers, how one is chosen, and what each tier does
and does not change in the numbers.

| Entry point | Location |
|---|---|
| Rust | `ferric-integrals/src/three_index_source.rs::ThreeIndexSource` (`build_band`, `build_band_screened`, `build_dressed`, `for_each_block`) |
| CLI | `[scf] jk_storage`, `[memory] budget_gb` (`site/src/reference/input.md`) |
| Environment | `FERRIC_JK_STORAGE` |

## The backends {#three-index-sources-backends}

`ThreeIndexSource` holds the raw tensor $(P|\mu\nu)$ for one operator. It is
chosen at build time against the memory budget, from these backends:

| Backend | Resident | Per pass |
|---|---|---|
| in core | $N_{\mathrm{aux}}N^2$ doubles | none |
| in core, packed | $N_{\mathrm{aux}}N(N+1)/2$ doubles (the $\mu\nu$ triangle) | unpack a block |
| in core, pair screened (RI-J only) | the Schwarz-kept $\mu\nu$ pairs | none |
| in core, single precision (RI-J only, `ri_j_precision = "f32"`) | the packed triangle or the kept pairs, as f32 | none; f64 accumulation |
| disk spill | one block, the tensor on disk | read the file |
| recompute | one block | recompute the integrals |

The packed tier uses the symmetry $(P|\mu\nu) = (P|\nu\mu)$: it stores the
lower $\mu\nu$ triangle, about half the footprint, and unpacks one block at a
time when a consumer streams it. In the example above that is 1.5 GB instead
of 3.0 GB. The pair-screened tier keeps only the $\mu\nu$ pairs that survive a
Schwarz screen (chapter @sec:screening). The single-precision tier rounds each f64 integral
once to f32 and accumulates in f64. Both of those are RI-J only, because RI-J
reads them directly and never needs the full unpacked $\mu\nu$ block.

Spill and recompute both hold one block of auxiliary rows. Spill writes the
tensor to disk once and reads the whole file on every pass. Recompute writes
nothing and rebuilds each block's integrals on every pass. Which one is cheaper
depends on the disk and the integral cost, so ferric measures both on the
machine (below).

Every consumer reads the tensor through one loop. Here is the dispatch in
`ferric-integrals/src/three_index_source.rs::ThreeIndexSource::for_each_block`,
with the two RI-J-only arms (which return an error here) and the packed
tier's memory charge removed. The in-core arm hands out borrowed slices, the
packed arm unpacks a block into scratch, the recompute arm rebuilds the
block's integrals, and the spill arm reads the block from the file:

```rust
match &mut self.backend {
    Backend::InCore(eri) => stream_in_core(eri, band, band_p0, step, f),
    // ...
    Backend::InCorePacked { packed, scratch } => {
        // ...
        stream_packed(packed, scratch, nao, band, band_p0, step, f)
    }
    Backend::Recompute {
        op,
        obs,
        dfbs,
        scratch,
        screen,
    } => {
        let scr = screen.as_ref().map(|(b, t)| (b.as_ref(), *t));
        ensure_block(scratch, step, nao);
        stream_recompute((*op, obs, dfbs, scr), scratch, band, band_p0, step, f)
    }
    Backend::DiskSpill { file, scratch } => {
        ensure_block(scratch, step, nao);
        stream_spill(file, scratch, nao, band, band_p0, step, f)
    }
}
```

Each call of `f` receives an `AuxBlock` of `step` auxiliary rows with the
global index `p0` of its first row. A consumer that slices a metric or a
coefficient vector with `p0` places its contribution correctly whatever the
backend.

**Bands.** A source may hold only a band of auxiliary rows
(`ferric-integrals/src/three_index_source.rs::ThreeIndexSource::build_band`).
Distributed runs use this: each rank holds its own band.

**A fixed screen.** A source may carry a QQR-3 screen
(`ferric-integrals/src/three_index_source.rs::ThreeIndexSource::build_band_screened`,
section @sec:qqr). The screen must stay fixed for the lifetime of the source. A
recomputed block then skips exactly the triples the first build skipped, so a
recomputed block equals the stored one.

## The options {#three-index-sources-options}

`[scf] jk_storage` selects the RI-J/K policy: `auto`, `memory`, `disk`,
`direct` or `split`. The `FERRIC_JK_STORAGE` environment variable sets the
same policy.

| `jk_storage` | Where the RI-J/K tensor lives |
|---|---|
| `auto` (default) | in memory if it fits, else the packed half, else (one rank, Coulomb) Split-RI-J for J; a tensor still needed goes to `disk` or `direct`, whichever two timings on this machine predict cheaper over 30 passes |
| `memory` | in memory, packed if only that fits; an error if neither fits |
| `disk` | spilled to `$TMPDIR` |
| `direct` | recomputed per pass; nothing written |
| `split` | no RI-J tensor; J by Split-RI-J on one rank |

Under `auto` the first memory tier that fits wins. Between spill and
recompute the choice comes from two timings taken on the machine when the
tensor is built (`crates/ferric-integrals/src/three_index_source/jk_storage.rs`).
The two timings are the integral cost per auxiliary row (four packed
blocks at the centres of the band's quarters) and the cold write and read
rates of the spill directory (a file of at most 64 MiB, synced, then read
with the page cache dropped). Spill costs one build overlapped with the
write plus $N$ reads; recompute costs $N$ builds; ties go to recompute,
which needs no scratch disk. The cost model takes $N = 30$ passes, two J
passes per Fock build times the iterations a converging SCF typically takes. `site/src/reference/input.md` is the
authority for these rules, including the shared-memory pool and the
multi-rank cases.

## Reproducibility {#three-index-sources-reproducibility}

The tiers hold the same numbers. What can differ is the order in which a
consumer sums them.

| Consumer | Across tiers |
|---|---|
| RI-J from the packed, spilled and recomputed tensors | bit-identical: one packed block stream |
| RI-J from the unpacked in-core tensor | last bits differ (summation order) |
| RI-K and other consumers | can differ at summation order between tiers |
| the dressed tensor across thread counts | bit-identical: row split on a 64-row grid, contraction in 128 blocks |

**The dressed tensor.**
`ferric-integrals/src/three_index_source.rs::ThreeIndexSource::build_dressed`
forms $\sum_Q M_{PQ}(Q|\mu\nu)$, the $B$ of (@eq:factorization-and-metric-1) in the AO basis, by
streaming the raw source. Two constants keep that sum independent of the
thread count and of the backend.

- Output rows are split on a grid of 64 rows (`DRESS_ROW_BLOCK`). The height
  is even because an odd split changes the vector lanes of the matrix
  product's accumulation, and with them the last bits.
- The contracted index $Q$ is summed in blocks of 128 (`DRESS_K_BLOCK`). The
  partial sums are added in ascending $Q$. This is blocked summation: each
  partial sum covers 128 terms, so less magnitude is lost to rounding than in
  one running sum over all $Q$. Against a compensated reference on
  benzene / def2-SVP the RMS error is $3.07\times10^{-17}$ blocked and
  $1.18\times10^{-16}$ with one sum over $Q$.

Here is the $Q$ loop of the in-core fast path,
`ferric-integrals/src/three_index_source.rs::dress_resident_into_packed`; the
gathering of the panel of raw rows and the parallel loop over column tiles
are removed:

```rust
let m_rows = m.slice(ndarray::s![band_p0..band_p0 + h, ..]);  // M_PQ for this band's P
for qa in (0..naux).step_by(DRESS_K_BLOCK) {                  // Q in blocks of 128
    let qb = (qa + DRESS_K_BLOCK).min(naux);
    dress_tile(
        &m_rows.slice(ndarray::s![.., qa..qb]),               // M_{P, Q in block}
        &panel.slice(ndarray::s![qa..qb, ..]),                // (Q in block | mu nu)
        &mut t,                                               // t += M (Q|mu nu)
    );
}
```

The block boundaries are compile-time constants. Neither the thread count nor
the budget can move them. The tile GEMMs run inside the rayon loop over
column tiles, where BLAS runs single-threaded, so the serial-section
`[parallel] blas_threads` setting does not reach them either. So no thread
setting can move a bit of $B$: `dressed_tensor_bitidentical_across_thread_counts`
(in `three_index_source.rs`) compares 1 thread with 2, 3, 5 and 8 on
benzene / def2-SVP bit for bit.

**Which tier `auto` picks** depends on the budget and on the free memory in
the shared pool, so the same input can choose differently on a busier
machine. Pinning `jk_storage` removes that dependence.

## Numerical anchors {#three-index-sources-numerical-anchors}

| What | Test | Bar |
|---|---|---|
| spilled DF tensor bitwise equal to in core | `crates/ferric-scf/tests/mwe_spilled_df_is_bitwise_in_core.rs` | bitwise |
| recompute backend writes nothing | `crates/ferric-integrals/tests/mwe_recompute_backend_avoids_disk.rs` | — |
| spill packs the $\mu\nu$ triangle | `crates/ferric-integrals/tests/mwe_spill_packs_the_mu_nu_triangle.rs` | — |
| the packed tier respects the shared pool | `crates/ferric-integrals/tests/mwe_packed_tier_respects_pool.rs` | — |
| blocked dressing that overshoots the budget is reported | `crates/ferric-integrals/tests/mwe_blocked_dressing_overshoot_always_reported.rs` | — |

No proof script covers this section. It changes where the numbers live and
the order they are summed in, not what they are, and the bitwise tests above
are its evidence.
