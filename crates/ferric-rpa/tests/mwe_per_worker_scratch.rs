//! MWE: the per-worker frequency scratch, and what it should cost.
//!
//! The `map_init` closures in `properties.rs` allocate their per-worker scratch
//! with `b_ov.clone()`. The closure then IMMEDIATELY overwrites it
//! (`b_scaled.assign(b_ov)` followed by an in-place column scaling), so the
//! cloned *contents* are never read — only the shape is used.
//!
//! That makes the clone a pure waste of a `naux · nov` copy per worker, and at
//! `properties.rs:1227` it happens twice (both spins):
//!
//! ```text
//!   naux=2976, nov=61740  ->  1.47 GB per spin per worker
//!   x2 spins x8 workers   -> 23.5 GB of copying for data that is discarded
//! ```
//!
//! The scratch itself is genuinely per-worker — the closure mutates it, so it
//! cannot be shared read-only (an earlier plan of mine that the code disproves).
//! What CAN go is the copy: `Array2::zeros(shape)` gives the same buffer without
//! reading `naux · nov` doubles first.
//!
//! These contracts pin the accounting the budget must use. Arithmetic only — no
//! SCF, no allocation.

/// Bytes of one `(naux, nov)` f64 scratch buffer.
fn scratch_bytes(naux: usize, nov: usize) -> usize {
    naux.saturating_mul(nov).saturating_mul(8)
}

/// Total per-worker scratch for a spin-resolved frequency loop.
fn total_scratch_bytes(naux: usize, nov: usize, n_spins: usize, n_workers: usize) -> usize {
    scratch_bytes(naux, nov)
        .saturating_mul(n_spins)
        .saturating_mul(n_workers)
}

/// The audit shape from the incident-scale RPA runs.
const NAUX: usize = 2976;
const NOV: usize = 61740;

/// CONTRACT 1: one scratch buffer at audit scale is ~1.5 GB.
///
/// Pins the unit so the multipliers below are meaningful.
#[test]
fn one_scratch_buffer_is_about_1_5_gb() {
    let gb = scratch_bytes(NAUX, NOV) as f64 / 1e9;
    assert!(
        (1.3..1.7).contains(&gb),
        "one (naux={NAUX}, nov={NOV}) scratch buffer should be ~1.5 GB, got {gb:.2} GB"
    );
}

/// CONTRACT 2: the two-spin path multiplies by spins AND workers.
///
/// `properties.rs:1227` inits `(inter_a.b_ov.clone(), inter_b.b_ov.clone())`,
/// so each worker holds TWO buffers. This is the ~22 GB figure, and it is
/// invisible to any estimator that counts a single buffer.
#[test]
fn two_spin_scratch_scales_with_spins_and_workers() {
    let one_worker_one_spin = total_scratch_bytes(NAUX, NOV, 1, 1);
    let eight_workers_two_spins = total_scratch_bytes(NAUX, NOV, 2, 8);

    assert_eq!(
        eight_workers_two_spins,
        one_worker_one_spin * 16,
        "two spins across eight workers must be 16x one buffer"
    );

    let gb = eight_workers_two_spins as f64 / 1e9;
    assert!(
        gb > 20.0,
        "the two-spin/eight-worker scratch should exceed 20 GB at audit scale \
         (this is the documented ~22 GB), got {gb:.2} GB"
    );
}

/// CONTRACT 3: a budget-derived worker cap must bound the scratch.
///
/// The remedy for a term that scales with core count: cap concurrency from the
/// budget rather than letting the machine decide. `max_workers = budget /
/// per_worker_bytes`, floored at 1 so work still progresses (slow, not stuck).
#[test]
fn a_budget_derived_worker_cap_bounds_the_scratch() {
    let per_worker = scratch_bytes(NAUX, NOV) * 2; // two spins

    for budget_gb in [4usize, 8, 16, 32] {
        let budget = budget_gb * 1_000_000_000;
        let cap = (budget / per_worker.max(1)).max(1);
        let used = per_worker * cap;

        // Either the cap is the floor of 1 (one worker's scratch exceeds the
        // whole budget — degrade to serial, never to stuck), or the capped
        // scratch fits.
        assert!(
            cap == 1 || used <= budget,
            "at a {budget_gb} GB budget: cap={cap} implies {:.2} GB, over budget",
            used as f64 / 1e9,
        );
        assert!(cap >= 1, "the cap must never reach zero");
    }
}

/// CONTRACT 4: cloning for shape is pure waste.
///
/// The closure overwrites its buffer before reading it, so the bytes copied by
/// `.clone()` are discarded. Zero-allocating the same shape costs the same
/// resident memory but skips the copy — and at audit scale that copy is
/// gigabytes per worker.
#[test]
fn cloning_for_shape_copies_bytes_that_are_discarded() {
    let per_spin = scratch_bytes(NAUX, NOV);
    let wasted_copy = per_spin * 2 * 8; // two spins, eight workers

    assert!(
        wasted_copy as f64 / 1e9 > 20.0,
        "the discarded copy at audit scale should exceed 20 GB, got {:.2} GB",
        wasted_copy as f64 / 1e9,
    );
}
