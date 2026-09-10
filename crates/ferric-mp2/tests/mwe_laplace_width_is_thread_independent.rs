//! MWE: the Laplace-MP2 panel widths must not depend on the ambient thread count.
//!
//! # The defect
//!
//! `laplace.rs`'s `per_task_budget_bytes` divides the resolved memory budget by
//! `rayon::current_num_threads()`:
//!
//! ```text
//! let threads = rayon::current_num_threads().max(1);
//! let total = ferric_core::memory::resolve_budget_bytes(explicit);
//! let share = total / threads;
//! ```
//!
//! Its return value then sizes two panel widths, and BOTH k-block a
//! floating-point accumulation:
//!
//! * `block_mu` (`compute_ao`, `laplace_ao_coulomb_energy`) blocks
//!   `j_mat += &m_panel.dot(&n_panel.t())` — a partial-sum accumulation over
//!   the μ axis, so the number of partial sums is `nbas / block_mu`.
//! * `block_p` (`laplace_exchange_energy`) blocks
//!   `(0..naux).step_by(block_p).map(..).sum()` — the number of partial sums
//!   IS `naux / block_p`.
//!
//! k-blocking a reduction is a float reassociation: it is a coarse pairwise
//! summation whose grouping changes the rounding. `three_index_source.rs`'s
//! `DRESS_ROW_BLOCK` measured ~7e-15 per odd-vs-even split on this hardware.
//!
//! So the same molecule, basis and `[memory] budget_gb` returned different last
//! digits under `RAYON_NUM_THREADS=1` and `RAYON_NUM_THREADS=12` — the energy
//! depended on ambient machine state that is not part of the configuration.
//!
//! # The contract this pins
//!
//! The sibling path already states the correct rule, in
//! `rimp2::mo_stream_chunk_for`'s doc and `mwe_mo_stream_chunk_budget.rs`
//! CONTRACT 5: a k-blocking width may derive from the BUDGET (same config →
//! same answer) but never from the ambient thread count, free memory, or wall
//! clock. Pin the budget and the numerics are pinned with it.
//!
//! Note what is deliberately NOT asserted here: that the width is independent
//! of the budget. It is not, and that is by design — the budget is what bounds
//! the transient. Only the *ambient* dependence is the bug.
//!
//! Arithmetic only — no SCF, no integrals, no allocation.

use ferric_mp2::laplace::per_task_budget_bytes_for_test as per_task;

/// A representative production budget: 8 GiB.
const BUDGET: usize = 8 * 1024 * 1024 * 1024;

/// Worker counts spanning a laptop, this box, and a large node.
const THREAD_COUNTS: [usize; 5] = [1, 2, 8, 12, 64];

/// Run `f` inside a rayon pool pinned to exactly `n` workers.
///
/// Pinning matters: an earlier budget MWE in this repo used the ambient pool and
/// therefore could not see a worker-count-dependent defect at all — the floor
/// and the byte cap happened to coincide at the ambient width. A test that
/// cannot vary the worker count cannot observe this class of bug.
fn with_threads<T: Send>(n: usize, f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap().install(f)
}

/// CONTRACT 1: the per-task ceiling is the same at every worker count.
///
/// This is the assertion that fails on the unfixed tree: `total / threads`
/// returns 8 GiB at 1 thread and 0.67 GiB at 12, a 12x swing in a quantity that
/// sizes a reduction's k-blocking.
#[test]
fn the_per_task_ceiling_does_not_depend_on_the_thread_count() {
    let reference = with_threads(1, || per_task(Some(BUDGET)));
    for n in THREAD_COUNTS {
        let got = with_threads(n, || per_task(Some(BUDGET)));
        assert_eq!(
            got, reference,
            "the Laplace per-task ceiling moved from {reference} to {got} bytes when the pool \
             went from 1 to {n} workers. It sizes block_mu and block_p, both of which k-block a \
             float accumulation, so this makes the ENERGY depend on RAYON_NUM_THREADS."
        );
    }
}

/// CONTRACT 2: the same holds across the whole budget range, not just one point.
///
/// A fix that happened to collapse the thread dependence only where some floor
/// binds would pass CONTRACT 1 while leaving the defect live everywhere else —
/// that is the "a passing test may be measuring inertness" failure mode. Sweep
/// budgets from far below any floor to far above, and require thread invariance
/// at every one.
#[test]
fn thread_invariance_holds_at_every_budget() {
    for budget in [
        1024 * 1024usize,              // 1 MiB — below any floor
        64 * 1024 * 1024,              // 64 MiB — at the historical MIN_PER_TASK
        512 * 1024 * 1024,             // 512 MiB
        8 * 1024 * 1024 * 1024,        // 8 GiB
        128 * 1024 * 1024 * 1024,      // 128 GiB — far above anything that binds
    ] {
        let reference = with_threads(1, || per_task(Some(budget)));
        for n in THREAD_COUNTS {
            assert_eq!(
                with_threads(n, || per_task(Some(budget))),
                reference,
                "budget {budget} bytes: the per-task ceiling depends on the worker count ({n})"
            );
        }
    }
}

/// CONTRACT 3: the ceiling still TRACKS the budget.
///
/// The cheap wrong fix is to stop reading the budget at all — return a
/// constant. That would pass CONTRACTS 1 and 2 while silently removing the
/// memory bound this function exists to impose, which is the "an over-
/// estimating guard is also a bug" hazard in reverse: a guard that no longer
/// guards. Require a strictly larger budget to buy a strictly larger ceiling
/// somewhere in the range.
#[test]
fn the_ceiling_still_tracks_the_budget() {
    let small = per_task(Some(64 * 1024 * 1024));
    let large = per_task(Some(64 * 1024 * 1024 * 1024));
    assert!(
        large > small,
        "a 1024x larger budget must buy a larger per-task ceiling, got {small} -> {large}. \
         If these are equal the budget is no longer bounding the panels."
    );
}

/// CONTRACT 4: it is a pure function — same input, same answer, repeatedly.
///
/// Nothing ambient (free memory, wall clock, a cached first call) may leak in.
/// Without this, a run is not reproducible from its config alone.
#[test]
fn the_ceiling_is_a_pure_function_of_the_budget() {
    for budget in [1024 * 1024usize, 512 * 1024 * 1024, 8 * 1024 * 1024 * 1024] {
        let first = per_task(Some(budget));
        for _ in 0..16 {
            assert_eq!(
                per_task(Some(budget)),
                first,
                "the per-task ceiling must be a pure function of the budget"
            );
        }
    }
}
