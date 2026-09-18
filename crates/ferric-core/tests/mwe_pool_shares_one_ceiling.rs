//! The property the pre-pool tree LACKED: a ceiling that is spent once.
//!
//! `resolve_budget_bytes` is called independently at ~50 sites and each call
//! returns the same full number, so five subsystems each believe they may use
//! the entire budget. Every individual "does this fit?" passes while the
//! process total is unbounded. That is how a 27-atom def2-SVP B3LYP RI-JK job
//! printed `memory budget: 4.72 GiB` and then died at `MAXRSS 6.04 GiB` to a
//! global (`CONSTRAINT_NONE`) OOM kill, taking the user's browser, terminal
//! multiplexer and editing session with it.
//!
//! This file pins the inverse: against one pool, two reservations that each
//! fit in isolation but sum past the capacity must NOT both succeed.
//!
//! ## Artifact hypothesis (before measuring)
//!
//! If the pool really debits, the second reservation of a pair fails and the
//! failure message names the second plane and the shortfall. If instead the
//! pool were secretly re-reading the ceiling (the defect it exists to fix),
//! both would succeed and `outstanding_bytes` would sit at either `0` or the
//! larger of the two rather than their sum. Those differ, so the test
//! distinguishes them.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ferric_core::memory::plan::{Lifetime, MemoryPlan};
use ferric_core::memory::pool::MemoryPool;

const GB: usize = 1_000_000_000;

/// The measured incident's shape, in miniature: a 4.72 GiB pool, a 3 GiB DF
/// 3-index tensor, a 3 GiB grid AO cache.
#[test]
fn the_measured_overshoot_shape_is_now_refused() {
    let pool = MemoryPool::with_capacity_bytes(4_720_000_000);

    let df = pool
        .reserve("DF 3-index (P|mn)", 3 * GB)
        .expect("3 GB fits a 4.72 GB pool on its own");

    // Under the OLD semantics this ALSO fit: it compared 3 GB against the
    // whole 4.72 GB ceiling, exactly as the DF tensor had. The sum (6 GB)
    // is the 6.04 GiB MAXRSS that got the process killed.
    let err = pool
        .reserve("KS grid AO cache", 3 * GB)
        .expect_err("the SECOND 3 GB must not also be admitted");

    let msg = err.to_string();
    assert!(
        msg.contains("KS grid AO cache"),
        "the error must name the plane that was refused: {msg}"
    );
    assert!(
        msg.contains("short by"),
        "the error must quantify the shortfall: {msg}"
    );
    assert!(
        msg.contains("DF 3-index"),
        "the breakdown must name the DOMINANT incumbent so the user knows \
         what to shrink: {msg}"
    );
    assert!(
        !msg.to_lowercase().contains("panic"),
        "must be a typed error, never a panic: {msg}"
    );

    assert_eq!(
        pool.outstanding_bytes(),
        3 * GB,
        "a refused reservation must debit nothing"
    );
    drop(df);
    assert_eq!(pool.outstanding_bytes(), 0);
}

#[test]
fn three_planes_each_fitting_alone_do_not_all_fit_together() {
    // Generalisation: the failure is not special to two.
    let pool = MemoryPool::with_capacity_bytes(10 * GB);
    let _a = pool.reserve("plane_a", 4 * GB).unwrap();
    let _b = pool.reserve("plane_b", 4 * GB).unwrap();
    assert!(
        pool.reserve("plane_c", 4 * GB).is_err(),
        "12 GB of planes must not fit a 10 GB pool"
    );
    assert_eq!(pool.outstanding_bytes(), 8 * GB);
}

#[test]
fn plans_built_in_two_subsystems_share_the_one_ceiling() {
    // The same property expressed through MemoryPlan, which is how a migrated
    // call site will actually reach the pool.
    let pool = MemoryPool::with_capacity_bytes(4 * GB);

    let mut scf = MemoryPlan::from_pool(&pool, "DF-JK");
    scf.reserve("B(P|mn)", 3 * GB / 8, Lifetime::Resident);
    let _held = scf.commit().expect("first subsystem fits");

    let mut ks = MemoryPlan::from_pool(&pool, "KS-DFT grid");
    ks.reserve("chi+dchi", 3 * GB / 8, Lifetime::Resident);
    assert!(
        ks.commit().is_err(),
        "the second subsystem must see only what the first left"
    );
}

#[test]
fn a_long_loop_of_transients_does_not_exhaust_the_pool() {
    // Release-on-drop is what makes enforcement usable: an 80-iteration SCF
    // loop reserving a batch buffer each pass must not spuriously fail.
    let pool = MemoryPool::with_capacity_bytes(2 * GB);
    for i in 0..200 {
        let r = pool
            .reserve("grid batch", GB)
            .unwrap_or_else(|e| panic!("iteration {i} spuriously refused: {e}"));
        assert_eq!(pool.outstanding_bytes(), GB);
        drop(r);
    }
    assert_eq!(pool.outstanding_bytes(), 0);
    assert_eq!(pool.peak_bytes(), GB, "never more than one batch at a time");
}

#[test]
fn nested_transients_stack_and_unwind_in_order() {
    let pool = MemoryPool::with_capacity_bytes(6 * GB);
    {
        let _outer = pool.reserve("outer", 2 * GB).unwrap();
        {
            let _inner = pool.reserve("inner", 3 * GB).unwrap();
            assert_eq!(pool.outstanding_bytes(), 5 * GB);
            assert!(
                pool.reserve("too much", 2 * GB).is_err(),
                "only 1 GB is left"
            );
        }
        assert_eq!(pool.outstanding_bytes(), 2 * GB);
        assert!(pool.reserve("now fits", 3 * GB).is_ok());
    }
    assert_eq!(pool.outstanding_bytes(), 0);
}

#[test]
fn rayon_workers_contend_on_one_ledger_without_oversubscribing() {
    // ferric reserves from rayon worker threads; the ledger must be sound
    // under contention, not merely sound serially.
    let pool = MemoryPool::with_capacity_bytes(16 * GB);
    let max_seen = Arc::new(AtomicUsize::new(0));
    let granted = Arc::new(AtomicUsize::new(0));
    let held: Mutex<Vec<_>> = Mutex::new(Vec::new());

    rayon::scope(|s| {
        for i in 0..64 {
            let pool = pool.clone();
            let max_seen = max_seen.clone();
            let granted = granted.clone();
            let held = &held;
            s.spawn(move |_| {
                if let Ok(r) = pool.reserve(&format!("worker_{}", i % 4), GB) {
                    granted.fetch_add(1, Ordering::SeqCst);
                    max_seen.fetch_max(pool.outstanding_bytes(), Ordering::SeqCst);
                    held.lock().unwrap().push(r);
                }
            });
        }
    });

    assert_eq!(
        granted.load(Ordering::SeqCst),
        16,
        "exactly capacity/request reservations may be granted"
    );
    assert!(
        max_seen.load(Ordering::SeqCst) <= 16 * GB,
        "the ledger was observed oversubscribed at {} bytes",
        max_seen.load(Ordering::SeqCst)
    );
    drop(held);
    assert_eq!(pool.outstanding_bytes(), 0);
}
