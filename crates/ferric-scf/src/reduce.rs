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

fn band_bytes_budget() -> usize {
    static BAND_BYTES: ferric_core::config::ConfigVar<usize> = ferric_core::config::ConfigVar {
        env_name: "FERRIC_REDUCE_BAND_BYTES",
        default: DEFAULT_BAND_BYTES,
        parse: |s| s.parse::<usize>().map_err(|e| e.to_string()),
        validate: |b| (*b > 0).then_some(()).ok_or_else(|| "must be > 0".to_string()),
    };
    BAND_BYTES.get().map(|r| r.value).unwrap_or_else(|e| {
        eprintln!("[config] FERRIC_REDUCE_BAND_BYTES: {e}; using default {DEFAULT_BAND_BYTES}");
        DEFAULT_BAND_BYTES
    })
}

/// Compute a band width (number of `nbf×nbf` partials held live at once) from the
/// byte budget. Floored at the rayon worker count so the byte budget never
/// throttles parallelism below one group per worker (band width affects ONLY the
/// live set and the degree of parallelism — the fold order is always ascending
/// group index regardless of banding, so a thread-count-dependent band width
/// cannot perturb the result). Always ≥ 1 so an oversized partial still
/// makes progress.
pub fn band_width(nbf: usize, budget_bytes: usize) -> usize {
    let per_partial = nbf.max(1) * nbf.max(1) * std::mem::size_of::<f64>();
    (budget_bytes / per_partial.max(1))
        .max(rayon::current_num_threads())
        .max(1)
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
/// The fold order (0,1,…,n_groups-1) is independent of thread count, so the
/// result is bit-identical across `RAYON_NUM_THREADS`.
pub fn grouped_deterministic_sum<F>(
    acc: &mut Array2<f64>,
    n_groups: usize,
    nbf: usize,
    make: F,
) -> Result<(), FerricError>
where
    F: Fn(usize) -> Result<Array2<f64>, FerricError> + Sync,
{
    let bw = band_width(nbf, band_bytes_budget());
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
    make: F,
) -> Result<(), FerricError>
where
    F: Fn(usize) -> Result<(Array2<f64>, Array2<f64>), FerricError> + Sync,
{
    // Two live matrices per group → halve the byte budget per band so the total
    // live set still respects it (the worker-count floor inside band_width is
    // kept so parallelism is never throttled).
    let bw = band_width(nbf, band_bytes_budget() / 2);
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
        // Pin the worker count so the parallelism floor is known.
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        pool.install(|| {
            // 100×100 f64 = 80_000 bytes/partial; 800_000 byte budget → 10.
            assert_eq!(band_width(100, 800_000), 10);
            // Budget below one partial floors at the worker count (band width
            // affects only memory/parallelism, never the fold order).
            assert_eq!(band_width(100, 1), 2);
            assert_eq!(band_width(0, 1), 2);
        });
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

        // Force a tiny budget so bands are narrow (band width = worker-count
        // floor) and the two-level path is actually exercised across several
        // bands. Banding must not perturb the ascending-group fold order.
        std::env::set_var("FERRIC_REDUCE_BAND_BYTES", "800");

        let run = |threads: usize| -> Array2<f64> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut acc = Array2::<f64>::zeros((n, n));
                grouped_deterministic_sum(&mut acc, n_groups, n, make).unwrap();
                acc
            })
        };

        let r1 = run(1);
        let r4 = run(4);
        let r8 = run(8);
        std::env::remove_var("FERRIC_REDUCE_BAND_BYTES");

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
        std::env::set_var("FERRIC_REDUCE_BAND_BYTES", "400");
        let mut aj = Array2::<f64>::zeros((n, n));
        let mut ak = Array2::<f64>::zeros((n, n));
        grouped_deterministic_sum_pair(&mut aj, &mut ak, n_groups, n, make).unwrap();
        std::env::remove_var("FERRIC_REDUCE_BAND_BYTES");
        assert_eq!(aj, ej);
        assert_eq!(ak, ek);
    }
}
