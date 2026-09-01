//! Bounded, thread-count-independent deterministic accumulation of parallel
//! matrix partials.
//!
//! ## The problem this solves
//!
//! The Fock builders (DF-K, direct-K, direct-JK, LinK) compute the K (and J)
//! matrix as a sum of many `nbf×nbf` partials produced in parallel. Two failure
//! modes must be avoided *simultaneously*:
//!
//! 1. **Non-determinism.** A rayon `reduce`/`try_reduce` combines partials in a
//!    binary tree whose shape depends on the worker count. Floating-point `+` is
//!    non-associative, so the summed matrix — and hence the SCF energy and every
//!    downstream MP2/RPA number — drifts by ~µHa with `RAYON_NUM_THREADS`.
//!    Commit `a8ec76f` fixed DF-K by collecting every per-chunk partial into a
//!    chunk-ordered `Vec` and summing sequentially. That pins the order but…
//! 2. **Unbounded memory.** …holding *all* partials at once is the DF-K scaling
//!    hazard: (naux/chunk)·nbf²·8 ≈ 97 GB at 50-atom/aug-cc-pVTZ. The direct
//!    fold/reduce trees have the same shape (one nbf² partial per work-chunk).
//!
//! ## The fix: two-level deterministic grouped sum
//!
//! Partition the `n_groups` work groups (chunks) into fixed-size **bands** whose
//! width is chosen from a byte budget. Process the groups of one band in parallel
//! — `into_par_iter().map().collect()` preserves index order, so the collected
//! `Vec` is band-local group order regardless of thread count — then fold that
//! band's partials into the running accumulator **in strict group order** before
//! moving to the next band.
//!
//! The total addition order is exactly group `0, 1, 2, …, n_groups-1`, identical
//! to the old collect-*all*-then-serial-sum, so the result is **bit-identical**
//! and **independent of thread count**. Only the live set changes: at most one
//! band of partials (`band_width` matrices) plus the single accumulator, instead
//! of all `n_groups`.
//!
//! This is preferred over a bounded-channel pipeline (single accumulator thread +
//! reorder buffer): it reuses rayon's already-proven index-order-preserving
//! `collect`, adds no accumulator thread, and the fold order is manifestly the
//! ascending group index — trivial to see it cannot depend on thread count.

use ferric_core::FerricError;
use ndarray::Array2;
use rayon::prelude::*;

/// Live-set byte budget for one band of partials. `band_width` is chosen so that
/// `band_width * nbf² * 8 ≲ this`, clamped to at least 1 group. 512 MiB keeps the
/// DF-K path well under the ~97 GB worst case while staying wide enough to
/// saturate the worker pool for typical thread counts. Overridable via
/// `FERRIC_REDUCE_BAND_BYTES` for tuning / testing.
const DEFAULT_BAND_BYTES: usize = 512 * 1024 * 1024;

/// Resolve the live-set byte budget for one band of reduction partials from a
/// FULLY-RESOLVED memory budget (TOML > env > auto — the caller resolves it
/// once, e.g. `rhf::resolve_three_index_budget`, and passes it down; this
/// function never re-reads the budget env vars itself). Precedence:
///
/// 1. An explicit `FERRIC_REDUCE_BAND_BYTES` env override wins verbatim
///    (tuning/testing knob — the reduce tests use it to force narrow bands).
/// 2. Otherwise `min(512 MiB, mem_budget_bytes / 4)` (see
///    [`ferric_core::memory::Share::Quarter`]): the band scratch is
///    ADDITIVE to the allocations the memory budget already governs (the
///    3-index tensor + the accumulator), so it gets a quarter, not the whole.
///
/// Band width affects ONLY the live set and the degree of parallelism, never
/// the fold order (see [`band_width`]), so this cannot perturb any result.
pub fn resolve_band_bytes(mem_budget_bytes: usize) -> usize {
    static BAND_BYTES: ferric_core::config::ConfigVar<usize> = ferric_core::config::ConfigVar {
        env_name: "FERRIC_REDUCE_BAND_BYTES",
        default: DEFAULT_BAND_BYTES,
        parse: |s| s.parse::<usize>().map_err(|e| e.to_string()),
        validate: |b| (*b > 0).then_some(()).ok_or_else(|| "must be > 0".to_string()),
    };
    match BAND_BYTES.get() {
        Ok(r) if r.source == ferric_core::config::ConfigSource::Env => r.value,
        Ok(r) => r.value.min(ferric_core::memory::transient_share(
            mem_budget_bytes,
            ferric_core::memory::Share::Quarter,
        )),
        Err(e) => {
            eprintln!("[config] FERRIC_REDUCE_BAND_BYTES: {e}; using default {DEFAULT_BAND_BYTES}");
            DEFAULT_BAND_BYTES
        }
    }
}

/// Band bytes for callers with NO plumbed memory budget (gradients, RPA
/// helpers, `rhf::build_jk`'s Newton/test path): resolve the unified budget
/// from env / auto-detect (TOML values can't reach here — plumb the resolved
/// budget and use [`resolve_band_bytes`] directly where it matters).
pub fn default_band_bytes() -> usize {
    resolve_band_bytes(ferric_core::memory::resolve_budget_bytes(None))
}

/// Compute a band width (number of `nbf×nbf` partials held live at once) from
/// the byte budget.
///
/// # Why the worker floor was removed
///
/// This used to read `(budget_bytes / per_partial).max(rayon::current_num_threads())`.
/// The `.max(nthreads)` was there for a good reason — a band narrower than the
/// pool leaves workers idle — but it made the byte budget **not a ceiling at
/// all**. On a many-core box with a tight budget the band came out `nthreads`
/// partials wide *regardless of the budget*, so `resolve_band_bytes`'s careful
/// `min(512 MiB, budget/4)` was silently discarded: at nbf = 2000 and 12
/// threads the floor alone pins 12 × 2000² × 8 = 384 MB live, whatever the
/// budget said. A budget that is quietly overridden is worse than no budget,
/// because callers reason as though it held.
///
/// The conflict is now resolved in favour of the budget, by **reducing
/// concurrency to what the budget affords** rather than by exceeding it. This
/// is the safe direction of the two the trade-off allows:
///
/// * It cannot change any result. The fold is always the strict ascending
///   group order `0, 1, …, n_groups-1`, independent of how the groups are
///   partitioned into bands (see [`grouped_deterministic_sum`]), so a narrower
///   band costs parallelism and nothing else. `grouped_sum_matches_flat_serial_sum_and_is_thread_independent`
///   pins that.
/// * It cannot deadlock or stall: the width still floors at **1**, so progress
///   is always possible; the degenerate case is a serial fold, which is slow
///   but correct.
/// * The alternative — keep the floor and warn — leaves the process actually
///   holding memory it was told not to. On this codebase that has meant a
///   46-minute reclaim stall inside a cgroup rather than a clean failure, which
///   is far harder to diagnose than reduced throughput.
///
/// A caller who would rather have the parallelism than the ceiling can raise
/// the ceiling: `FERRIC_REDUCE_BAND_BYTES` overrides it verbatim (see
/// [`resolve_band_bytes`]).
pub fn band_width(nbf: usize, budget_bytes: usize) -> usize {
    let per_partial = nbf.max(1) * nbf.max(1) * std::mem::size_of::<f64>();
    // Floor of 1, NOT of the worker count: a single partial must always be
    // representable (otherwise no progress is possible), but nothing wider is
    // owed to the pool at the budget's expense.
    (budget_bytes / per_partial.max(1)).max(1)
}

/// Band bytes left for the collected partials after the per-worker scratch a
/// `make` closure holds *while producing* one partial is subtracted.
///
/// # Why this exists (defect C)
///
/// [`band_width`] sizes the band from the partials it *collects* — one `nbf²`
/// matrix per group. But a `make` closure is not free: it allocates its own
/// working buffers, and rayon runs one closure per worker concurrently, so that
/// scratch is live `workers` times over at exactly the moment the band is also
/// live. `df_k`'s density path holds `3·c·n²` doubles per worker (the `bswap`,
/// `zt`, and `bt` repack/GEMM buffers) and its occupied path holds `n·c·nocc`;
/// none of it reached the sizing. The 3·c·n² figure was even written down in
/// `build_impl`'s own comment — it simply never made it into a number the
/// budget saw. At n = 2000, c = 4, 12 threads that is ~4.6 GB outside any
/// budget.
///
/// Subtracting it here keeps the ONE knob (`band_bytes`) governing the whole
/// live set: scratch first, because it is not optional, then as many partials
/// as the remainder affords. [`band_width`] still floors at one partial, so a
/// budget entirely consumed by scratch degrades to a serial fold rather than
/// failing — and, as ever, band width cannot change the fold order or the
/// result.
///
/// `per_worker_elems` is the element count (not bytes) one worker's scratch
/// holds; `workers` is the concurrent fan-out (`rayon::current_num_threads()`).
pub fn band_bytes_after_worker_scratch(
    band_bytes: usize,
    per_worker_elems: usize,
    workers: usize,
) -> usize {
    let scratch = per_worker_elems
        .saturating_mul(std::mem::size_of::<f64>())
        .saturating_mul(workers.max(1));
    band_bytes.saturating_sub(scratch)
}

/// Fixed target group count for partitioning a work list ahead of
/// [`grouped_deterministic_sum`]. The partition MUST be a pure function of the
/// work list — never of the thread count — because group boundaries determine
/// the floating-point association of the per-group serial folds. 1024 groups
/// keeps rayon well fed at any realistic worker count while amortizing the
/// per-group nbf² partial allocation over many work items.
pub const TARGET_GROUPS: usize = 1024;

/// Thread-count-independent group size for `n_items` work items: `n_items`
/// split into at most [`TARGET_GROUPS`] equal groups (each ≥ 1 item).
pub fn deterministic_group_size(n_items: usize) -> usize {
    n_items.div_ceil(TARGET_GROUPS).max(1)
}

/// Deterministic, memory-bounded parallel accumulation.
///
/// Produces group partials `make(g)` for `g in 0..n_groups` (in parallel within a
/// band), adding each onto `acc` in strict ascending group order. `acc` keeps any
/// prior contents (equivalent to `*acc += total`, as the direct builders need).
/// Live set ≤ one band of partials + `acc`.
///
/// `band_bytes` is the live-set budget for one band of partials — pass
/// [`resolve_band_bytes`]`(mem_budget)` when a resolved memory budget is in
/// hand (the JK builders), or [`default_band_bytes`]`()` otherwise.
///
/// The fold order (0,1,…,n_groups-1) is independent of thread count AND of
/// `band_bytes`, so the result is bit-identical across `RAYON_NUM_THREADS`
/// and across band budgets.
pub fn grouped_deterministic_sum<F>(
    acc: &mut Array2<f64>,
    n_groups: usize,
    nbf: usize,
    band_bytes: usize,
    make: F,
) -> Result<(), FerricError>
where
    F: Fn(usize) -> Result<Array2<f64>, FerricError> + Sync,
{
    let bw = band_width(nbf, band_bytes);
    let mut g0 = 0usize;
    while g0 < n_groups {
        let g1 = (g0 + bw).min(n_groups);
        // Parallel over this band's groups; `collect` preserves ascending index
        // order regardless of worker count.
        let partials: Vec<Array2<f64>> = (g0..g1)
            .into_par_iter()
            .map(&make)
            .collect::<Result<Vec<_>, FerricError>>()?;
        // Serial fold in group order — the determinism anchor.
        //
        // Do NOT "parallelize" this with a pairwise/tree reduction to shorten
        // the O(band_width) serial chain. A tree-fold is deterministic, but it
        // is not bit-identical: FP addition is non-associative, so re-bracketing
        // the sum returns different bits than this linear scan and would shift
        // every SCF energy in the codebase. The guarantee documented above —
        // identical results across RAYON_NUM_THREADS and across band budgets —
        // is exactly this ascending order. Parallelism here is only available
        // at the cost of the invariant, which is not a trade worth making.
        for p in &partials {
            *acc += p;
        }
        g0 = g1;
    }
    Ok(())
}

/// Like [`grouped_deterministic_sum`] but each group yields a **pair** of
/// matrices `(J, K)` folded into `(acc0, acc1)` respectively. Used by the
/// combined direct-JK builder. Both accumulators must start zeroed.
pub fn grouped_deterministic_sum_pair<F>(
    acc0: &mut Array2<f64>,
    acc1: &mut Array2<f64>,
    n_groups: usize,
    nbf: usize,
    band_bytes: usize,
    make: F,
) -> Result<(), FerricError>
where
    F: Fn(usize) -> Result<(Array2<f64>, Array2<f64>), FerricError> + Sync,
{
    // Two live matrices per group → halve the byte budget per band so the total
    // live set still respects it (the worker-count floor inside band_width is
    // kept so parallelism is never throttled).
    let bw = band_width(nbf, band_bytes / 2);
    let mut g0 = 0usize;
    while g0 < n_groups {
        let g1 = (g0 + bw).min(n_groups);
        let partials: Vec<(Array2<f64>, Array2<f64>)> = (g0..g1)
            .into_par_iter()
            .map(&make)
            .collect::<Result<Vec<_>, FerricError>>()?;
        for (p0, p1) in &partials {
            *acc0 += p0;
            *acc1 += p1;
        }
        g0 = g1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_width_respects_budget_and_floors() {
        // Pin the worker count so any lingering thread-count dependence shows.
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        pool.install(|| {
            // 100×100 f64 = 80_000 bytes/partial; 800_000 byte budget → 10.
            assert_eq!(band_width(100, 800_000), 10);
            // Budget below one partial floors at ONE, not at the worker count:
            // a single partial must be representable so progress is possible,
            // but the pool is not owed extra width at the budget's expense.
            assert_eq!(band_width(100, 1), 1);
            assert_eq!(band_width(0, 1), 1);
        });
    }

    /// REGRESSION (defect B): the byte budget is a HARD ceiling.
    ///
    /// `band_width` used to end in `.max(rayon::current_num_threads())`, so on
    /// a many-core box with a tight budget the live set was `nthreads` partials
    /// wide no matter what the budget said — `resolve_band_bytes` was not a
    /// ceiling at all. The floor is now 1. The live set must therefore never
    /// exceed the budget except in the single unavoidable case where one
    /// partial alone is already over it.
    #[test]
    fn band_width_never_exceeds_the_byte_budget() {
        let per_partial = 100 * 100 * std::mem::size_of::<f64>(); // 80_000
        for threads in [1usize, 2, 8, 32] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                // A budget affording exactly 3 partials must yield 3 at EVERY
                // thread count — 32 workers must not widen it to 32.
                let budget = 3 * per_partial;
                let bw = band_width(100, budget);
                assert_eq!(bw, 3, "thread count {threads} must not widen the band");
                assert!(
                    bw * per_partial <= budget,
                    "live set {} exceeds budget {budget} at {threads} threads",
                    bw * per_partial
                );
                // Sub-partial budget: floors at 1 (progress), and that single
                // partial is the ONLY sanctioned overshoot.
                assert_eq!(band_width(100, 1), 1, "must floor at 1, not at {threads}");
            });
        }
    }

    /// The other direction: narrowing the band must not change the answer.
    ///
    /// Removing the worker floor makes bands narrower on tight budgets. That is
    /// only acceptable because the fold order is the strict ascending group
    /// index regardless of banding. Pin it: a one-partial-wide band and a
    /// band-everything budget must agree bit-for-bit, at several thread counts.
    #[test]
    fn narrow_bands_are_bit_identical_to_wide_bands() {
        let n = 6;
        let n_groups = 41;
        let make = |g: usize| -> Result<Array2<f64>, FerricError> {
            let mut a = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    a[(i, j)] = ((g as f64) + 1.0) * 0.1 * (((i * 3 + j * 5 + g) % 13) as f64)
                        / ((g as f64) + 1.7);
                }
            }
            Ok(a)
        };
        let run = |threads: usize, band_bytes: usize| -> Array2<f64> {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                let mut acc = Array2::<f64>::zeros((n, n));
                grouped_deterministic_sum(&mut acc, n_groups, n, band_bytes, make).unwrap();
                acc
            })
        };
        let per_partial = n * n * std::mem::size_of::<f64>();
        // Widest possible band (all groups live) as the reference.
        let wide = run(1, n_groups * per_partial);
        for threads in [1usize, 2, 4, 8] {
            // Narrowest possible band: 1 byte → width 1.
            assert_eq!(
                run(threads, 1),
                wide,
                "a 1-wide band must be bit-identical to a full-width band ({threads} threads)"
            );
        }
    }

    #[test]
    fn group_size_is_thread_count_independent() {
        // Pure function of the item count — identical under any pool.
        let g = deterministic_group_size(1_000_000);
        let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let pool8 = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
        assert_eq!(pool1.install(|| deterministic_group_size(1_000_000)), g);
        assert_eq!(pool8.install(|| deterministic_group_size(1_000_000)), g);
        assert_eq!(deterministic_group_size(0), 1);
        assert_eq!(deterministic_group_size(5), 1); // < TARGET_GROUPS → 1 item/group
        assert_eq!(deterministic_group_size(2048), 2);
    }

    #[test]
    fn grouped_sum_matches_flat_serial_sum_and_is_thread_independent() {
        let n = 7;
        let n_groups = 37; // spans several bands at a small budget
        let make = |g: usize| -> Result<Array2<f64>, FerricError> {
            // Deterministic per-group partial with irregular magnitudes so the
            // sum is genuinely order-sensitive in floating point.
            let mut a = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    a[(i, j)] = ((g as f64) + 1.0) * 0.1 * (((i * 3 + j * 5 + g) % 13) as f64)
                        / ((g as f64) + 1.7);
                }
            }
            Ok(a)
        };

        // Ground truth: strict serial fold in group order.
        let mut expected = Array2::<f64>::zeros((n, n));
        for g in 0..n_groups {
            expected += &make(g).unwrap();
        }

        // Tiny explicit band budget so bands are narrow (band width =
        // worker-count floor) and the two-level path is actually exercised
        // across several bands. Banding must not perturb the fold order.
        let run = |threads: usize| -> Array2<f64> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut acc = Array2::<f64>::zeros((n, n));
                grouped_deterministic_sum(&mut acc, n_groups, n, 800, make).unwrap();
                acc
            })
        };

        let r1 = run(1);
        let r4 = run(4);
        let r8 = run(8);

        // Bit-identical across thread counts AND equal to the flat serial fold.
        assert_eq!(r1, expected, "grouped sum must equal flat serial fold");
        assert_eq!(r1, r4, "must be bit-identical across thread counts (1 vs 4)");
        assert_eq!(r1, r8, "must be bit-identical across thread counts (1 vs 8)");
    }

    #[test]
    fn grouped_sum_pair_matches_flat_serial_sum() {
        let n = 5;
        let n_groups = 23;
        let make = |g: usize| -> Result<(Array2<f64>, Array2<f64>), FerricError> {
            let mut j = Array2::<f64>::zeros((n, n));
            let mut k = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for jj in 0..n {
                    j[(i, jj)] = 0.03 * (((i + jj + g) % 7) as f64);
                    k[(i, jj)] = 0.07 * (((i * 2 + jj + g) % 5) as f64);
                }
            }
            Ok((j, k))
        };
        let mut ej = Array2::<f64>::zeros((n, n));
        let mut ek = Array2::<f64>::zeros((n, n));
        for g in 0..n_groups {
            let (j, k) = make(g).unwrap();
            ej += &j;
            ek += &k;
        }
        let mut aj = Array2::<f64>::zeros((n, n));
        let mut ak = Array2::<f64>::zeros((n, n));
        grouped_deterministic_sum_pair(&mut aj, &mut ak, n_groups, n, 400, make).unwrap();
        assert_eq!(aj, ej);
        assert_eq!(ak, ek);
    }

    #[test]
    fn resolve_band_bytes_caps_by_quarter_budget_env_wins() {
        // No env override: min(512 MiB default, budget/4).
        std::env::remove_var("FERRIC_REDUCE_BAND_BYTES");
        let gib = 1024 * 1024 * 1024;
        // Large budget → the 512 MiB default is the binding cap.
        assert_eq!(resolve_band_bytes(4 * gib), DEFAULT_BAND_BYTES);
        // Small budget → budget/4 binds.
        assert_eq!(resolve_band_bytes(gib), gib / 4);
        // Degenerate zero budget → floors at 1 (band_width still floors at the
        // worker count, so this never serializes the reduction).
        assert_eq!(resolve_band_bytes(0), 1);
        // Explicit env override wins verbatim over both. (Env is process-global;
        // a transient value here can only alter another test's band width, which
        // never changes results — see the band_width doc.)
        std::env::set_var("FERRIC_REDUCE_BAND_BYTES", "12345");
        assert_eq!(resolve_band_bytes(gib), 12345);
        std::env::remove_var("FERRIC_REDUCE_BAND_BYTES");
    }
}
