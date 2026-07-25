//! MWE: does the RPA quadrature panel width honor a small memory budget?
//!
//! Companion to `mwe_budget_respected.rs` (the dipole-banding contracts). Same
//! discipline: tiny shapes, explicit worker counts, no SCF — the claims are
//! structural, so they need neither scale nor a warm box.
//!
//! `quad_panel_width` bounds the per-worker quadrature scratch in the frequency
//! loop, which is `n_workers · (m·k + m²) · 8` bytes for panel width `k`. It is
//! one of the few knobs in ferric that was already budget-aware, so these
//! contracts are mostly a REGRESSION NET rather than a bug report — with one
//! sharp exception (contract 4).
//!
//! # Why every test pins an explicit worker count
//!
//! Both in-tree tests of this function passed `rayon::current_num_threads()`
//! and a 200-byte "absurdly small" budget. Because the full-width fast path
//! triggers when `y_bytes + n_workers·(m·nov + m²)·8 <= budget`, that budget is
//! only "tiny" on a box with enough workers. Measured for the m=2, nov=2 shape
//! those tests use:
//!
//! ```text
//!   workers | full-width scratch + y | k(budget=200) | k(budget=64)
//!         1 |                     96 |             2 |            1
//!         2 |                    160 |             2 |            1   <- narrow box
//!         4 |                    288 |             1 |            1
//!        12 |                    800 |             1 |            1
//!        64 |                   4128 |             1 |            1
//! ```
//!
//! So they failed on any box with <= 2 rayon workers and passed on wider ones,
//! while believing they had forced panelling. That is a test defect rather than
//! a production bug — returning full width when full width genuinely fits is
//! correct — but it hid the panelled code path from CI on narrow machines.

use ferric_rpa::energy::quad_panel_width_for_test as quad_panel_width;

/// Worker counts spanning narrow and fat boxes.
const THREAD_SWEEP: [usize; 5] = [1, 2, 4, 12, 64];

/// Bytes of per-worker scratch implied by a panel width, plus the shared `y`
/// projection: `y (m·nov) + n_workers · (m·k + m²)`, all f64.
fn implied_bytes(m: usize, nov: usize, k: usize, n_workers: usize) -> usize {
    let y = m * nov * 8;
    let per_worker = (m * k + m * m) * 8;
    y + n_workers * per_worker
}

/// CONTRACT 1: a starvation budget must force the narrowest panel, at ANY
/// worker count.
///
/// This is the property both in-tree tests believed they were asserting. A
/// budget that forces k=1 on a 64-worker box but not on a 2-worker box makes
/// the panelled path untested exactly where it is cheapest to test.
#[test]
fn starvation_budget_forces_narrowest_panel_at_every_worker_count() {
    for nthreads in THREAD_SWEEP {
        let k = quad_panel_width(2, 2, nthreads, Some(64));
        assert_eq!(
            k, 1,
            "at {nthreads} workers a 64-byte budget must force k=1, got {k}"
        );
    }
}

/// CONTRACT 2: the panel width must never imply more scratch than the budget.
///
/// The core budget-respect property. Swept across worker counts because the
/// per-worker term is what makes memory scale with core count — the same defect
/// class as the dipole-banding thread floor.
#[test]
fn panel_width_never_exceeds_budget() {
    let (m, nov) = (8usize, 64usize);

    for nthreads in THREAD_SWEEP {
        // Budgets spanning "cannot fit even one column" to "comfortably fits".
        for budget in [64usize, 1_024, 16_384, 1_048_576] {
            let k = quad_panel_width(m, nov, nthreads, Some(budget));
            let used = implied_bytes(m, nov, k, nthreads);

            // k=1 is the floor: if even one column busts the budget, the
            // algorithm must still make progress (contract 3), so only widths
            // above the floor are held to the ceiling.
            if k > 1 {
                assert!(
                    used <= budget,
                    "at {nthreads} workers with a {budget}-byte budget: k={k} \
                     implies {used} bytes ({:.1}x over)",
                    used as f64 / budget as f64,
                );
            }
        }
    }
}

/// CONTRACT 3: the panel must never collapse to zero width.
///
/// Bounding memory must degrade to *slow*, never to *stuck*. Guards against
/// overcorrecting contract 2.
#[test]
fn panel_width_floors_at_one() {
    for nthreads in THREAD_SWEEP {
        for budget in [0usize, 1, 8] {
            let k = quad_panel_width(8, 64, nthreads, Some(budget));
            assert!(
                k >= 1,
                "at {nthreads} workers with a {budget}-byte budget: panel width \
                 must be >=1 (slow, not stuck), got {k}"
            );
        }
    }
}

/// CONTRACT 4: an unset budget must not mean "unlimited".
///
/// `quad_panel_width` returns the FULL width `nov` when handed `None`, so every
/// caller that passes `None` opts out of the bound entirely — and in
/// `properties.rs` nine call sites hardcode exactly that. This test documents
/// the present behavior rather than asserting the desired one, so the contract
/// is visible and versioned; the planner work is what will let `None` resolve
/// to the process-wide budget instead of to infinity.
#[test]
fn unset_budget_currently_means_full_width() {
    let (m, nov) = (8usize, 64usize);
    for nthreads in THREAD_SWEEP {
        assert_eq!(
            quad_panel_width(m, nov, nthreads, None),
            nov,
            "documenting today's behavior: a None budget yields full width \
             (i.e. no bound at all) regardless of worker count"
        );
    }
}
