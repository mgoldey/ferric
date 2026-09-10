//! MWE: the estimate must scale the frequency scratch by the WORKER COUNT.
//!
//! # What this file used to be, and why it was replaced
//!
//! Its previous version defined its own `scratch_bytes` / `total_scratch_bytes`
//! helpers and asserted things about them. It imported nothing —
//! `grep -c 'ferric_'` over the file returned **0** — so all four "contracts"
//! tested locally-declared arithmetic and could not fail for any change to
//! ferric. A green run implied coverage that did not exist. It was found by an
//! adversarially-verified memory-budget audit and confirmed 2/3.
//!
//! Its narrative was also stale twice over:
//!
//! * It described `b_ov.clone()` inside `properties.rs`'s `map_init` closures
//!   copying 23.5 GB of data that is immediately overwritten. **That site is
//!   already fixed** — `properties.rs:1529` now builds the per-worker scratch
//!   with `Array2::zeros(inter_a.b_ov.raw_dim())`, and its comment records the
//!   same 23.5 GB figure as the reason.
//! * Four `b_ov.clone()` calls do remain (properties.rs:280, 451, 608, 1198),
//!   but they are NOT the same defect: each clones and then scales in place
//!   (`col.mapv_inplace(|x| x * s)`), so the cloned contents ARE read.
//!   Replacing those with `zeros` would be wrong, and the old file's premise
//!   does not apply to them.
//!
//! # What it tests now
//!
//! The accounting that actually matters and is actually reachable: that
//! `estimate_peak_bytes` multiplies the per-worker frequency scratch by
//! `n_workers`. Charging it once where rayon allocates one per worker is
//! defect pattern 3 in this repo's taxonomy, and it under-predicts by the core
//! count — the mechanism behind the 16-17 GB anon-RSS incidents, where memory
//! scaled with cores rather than with the budget.

use ferric_rpa::budget::{estimate_peak_bytes, PeakEstimateShape};

/// The audit shape from the incident-scale RPA runs.
const NAUX: usize = 2976;
const NOCC: usize = 42;
const NVIR: usize = 1470;

fn shape(n_workers: usize) -> PeakEstimateShape {
    PeakEstimateShape {
        naux: NAUX,
        nocc: NOCC,
        nvir: NVIR,
        n_quad: 12,
        n_workers,
        n_keep: NAUX,
        grid: None,
        need_inv_dielectric: false,
        nao: 0,
    }
}

/// CONTRACT 1: the estimate GROWS with the worker count.
///
/// The property the old file gestured at but never checked against ferric.
/// Each rayon worker in the frequency loop holds its own scratch, so an
/// estimate flat in `n_workers` under-predicts by exactly that factor.
#[test]
fn the_estimate_scales_with_the_worker_count() {
    let one = estimate_peak_bytes(shape(1));
    let eight = estimate_peak_bytes(shape(8));
    assert!(
        eight > one,
        "the per-worker frequency scratch must be charged per worker: 1 worker gives {one} \
         bytes and 8 gives {eight}. An estimate flat in n_workers is defect pattern 3 — it \
         under-predicts by the core count, which is how memory came to scale with cores \
         instead of with the budget."
    );
}

/// CONTRACT 2: the growth is proportional, not a token increment.
///
/// CONTRACT 1 would pass if the estimate rose by one byte per worker. The
/// scratch is `(m, nov) + (m, m)` per worker, so going 1 -> 8 workers must add
/// roughly seven more of them.
#[test]
fn the_growth_is_proportional_to_the_worker_count() {
    let one = estimate_peak_bytes(shape(1));
    let eight = estimate_peak_bytes(shape(8));
    let delta = eight - one;
    let per_worker = NAUX * NOCC * NVIR * 8 + NAUX * NAUX * 8;
    // Seven extra workers, allowing generous slack for the other terms in the
    // estimate that do not scale with workers.
    assert!(
        delta >= 7 * per_worker / 2,
        "8 workers added only {delta} bytes over 1 worker, but seven extra per-worker \
         scratches are ~{} bytes. The charge is present but not proportional.",
        7 * per_worker
    );
}

/// CONTRACT 3: monotone in the worker count.
///
/// Cheap, and it catches a sign error or a saturating-divide that the two
/// endpoints above could miss.
#[test]
fn the_estimate_is_monotone_in_workers() {
    let mut last = 0usize;
    for w in 1..=16 {
        let e = estimate_peak_bytes(shape(w));
        assert!(
            e >= last,
            "estimate fell going from {} to {w} workers: {last} -> {e}",
            w - 1
        );
        last = e;
    }
}

/// CONTRACT 4 (the reachability guard): the worker term is a real share of the
/// total, not a rounding artifact.
///
/// If the per-worker term were tiny next to the resident tensors, CONTRACTS 1-2
/// could pass while the accounting was effectively inert — the "a passing test
/// may be measuring inertness" trap this repo has hit repeatedly. At the audit
/// shape the scratch dominates, so assert that.
#[test]
fn the_worker_term_is_a_material_share_of_the_estimate() {
    let eight = estimate_peak_bytes(shape(8));
    let one = estimate_peak_bytes(shape(1));
    let worker_share = (eight - one) as f64 / eight as f64;
    assert!(
        worker_share > 0.25,
        "the worker-scaled part is only {:.1}% of the 8-worker estimate, so CONTRACTS 1-2 \
         are testing a rounding term rather than the dominant allocation",
        worker_share * 100.0
    );
}
