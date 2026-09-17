//! EXACTNESS ANCHOR for the shared memory pool.
//!
//! Every approximation has a trivial limit where it does nothing. For the
//! pool, that limit is "no pool installed": the budget was never configured,
//! nothing should be debited, no gate should refuse anything, and behaviour
//! must be bit-identical to the pre-pool tree.
//!
//! Per `CLAUDE.md`'s Experimental Protocol this file was written and made to
//! pass BEFORE any call site was migrated. If the pool ever acquires a
//! behaviour that leaks into the unbudgeted path — an implicit default
//! capacity, a lazily auto-resolved global, a gate that treats "no pool" as
//! "no memory" — these assertions fail, which is the point.
//!
//! ## Artifact hypothesis (stated before measuring, per the protocol)
//!
//! If the no-op property is REAL, an unbudgeted gate returns an inert
//! reservation and an unbudgeted `global_available_bytes()` is `None`, so
//! every downstream `if let Some(..)` sizing branch is skipped and the old
//! code path runs unchanged.
//!
//! If the implementation is BROKEN in the most likely way — the global slot
//! auto-resolving a budget on first read, which is exactly the ambient-state
//! defect the pool exists to retire — then `global()` would return `Some`
//! with a ~0.8×RAM capacity, gates would start debiting, and
//! `global_available_bytes()` would be `Some(huge)`. Those two predictions
//! differ, so this test can distinguish them.
//!
//! These tests manipulate the process-global slot, so they must not run
//! concurrently with each other. They serialize on `GLOBAL_LOCK`.

use std::sync::{Mutex, MutexGuard, OnceLock};

use ferric_core::memory::pool::{
    self, clear_global, global, global_available_bytes, install_global, reserve_global,
    try_reserve_global, MemoryPool,
};

const GB: usize = 1_000_000_000;

fn global_lock() -> MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    // A poisoned lock here means a sibling test panicked; the slot state is
    // still recoverable because every test clears it up front.
    match L.get_or_init(|| Mutex::new(())).lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    }
}

/// RAII: clear the global slot on entry AND on exit, so one test's pool can
/// never leak into another's "unbudgeted" assertions.
struct CleanSlot(#[allow(dead_code)] MutexGuard<'static, ()>);

impl CleanSlot {
    fn acquire() -> Self {
        let g = global_lock();
        clear_global();
        Self(g)
    }
}

impl Drop for CleanSlot {
    fn drop(&mut self) {
        clear_global();
    }
}

#[test]
fn with_no_pool_installed_there_is_no_global_pool() {
    let _s = CleanSlot::acquire();
    assert!(
        global().is_none(),
        "the global slot must NOT lazily auto-resolve a budget — that would be \
         the ambient-state defect the pool exists to retire"
    );
    assert!(
        global_available_bytes().is_none(),
        "unbudgeted `available` must be None (= 'do what you did before'), \
         never Some(0) (= 'refuse everything')"
    );
}

#[test]
fn with_no_pool_installed_a_reservation_is_inert_and_infallible() {
    let _s = CleanSlot::acquire();
    // An absurd request that no real pool could ever satisfy.
    let r = reserve_global("DF 3-index (P|mn)", usize::MAX)
        .expect("an unbudgeted reservation must never fail");
    assert!(r.is_inert(), "unbudgeted reservation must hold no pool");
    assert_eq!(r.bytes(), 0, "unbudgeted reservation must debit nothing");
    drop(r);

    // ...and repeatedly, because a long SCF loop reserves every iteration.
    for _ in 0..1000 {
        let r = reserve_global("grid batch", 10 * GB).expect("must never fail unbudgeted");
        assert!(r.is_inert());
    }
}

#[test]
fn with_no_pool_installed_try_reserve_always_admits() {
    let _s = CleanSlot::acquire();
    let r = try_reserve_global("KS grid AO cache", usize::MAX);
    assert!(
        r.is_some(),
        "unbudgeted `try_reserve` must ADMIT (None means 'fall back to \
         batching', which would change the unbudgeted code path)"
    );
    assert!(r.unwrap().is_inert());
}

#[test]
fn installing_then_clearing_restores_the_no_op_limit() {
    let _s = CleanSlot::acquire();
    install_global(MemoryPool::with_capacity_bytes(GB));
    assert!(global().is_some());
    // Budgeted: the absurd ask is refused.
    assert!(reserve_global("huge", 10 * GB).is_err());

    clear_global();
    // Unbudgeted again: the identical ask is admitted, inert.
    let r = reserve_global("huge", 10 * GB).expect("must be a no-op once cleared");
    assert!(r.is_inert());
    assert!(global_available_bytes().is_none());
}

#[test]
fn an_installed_pool_does_not_leak_capacity_across_clears() {
    let _s = CleanSlot::acquire();
    let pool = MemoryPool::with_capacity_bytes(4 * GB);
    install_global(pool.clone());
    let held = reserve_global("resident", 3 * GB).unwrap();
    assert_eq!(pool.outstanding_bytes(), 3 * GB);

    // Clearing the slot does NOT forcibly release outstanding reservations —
    // the guard still owns them, and dropping it must still credit the right
    // pool (not the newly-installed one).
    clear_global();
    let fresh = MemoryPool::with_capacity_bytes(4 * GB);
    install_global(fresh.clone());
    drop(held);
    assert_eq!(pool.outstanding_bytes(), 0, "guard credits its OWN pool");
    assert_eq!(
        fresh.outstanding_bytes(),
        0,
        "a dropped guard must not credit an unrelated pool"
    );
}

#[test]
fn resolve_honours_an_explicit_capacity_without_touching_the_environment() {
    // `MemoryPool::resolve(Some(b))` must return exactly `b` — the same
    // precedence `resolve_budget` already documents. This pins that the pool
    // did not introduce a second, divergent resolution chain.
    let _s = CleanSlot::acquire();
    let p = MemoryPool::resolve(Some(7 * GB));
    assert_eq!(p.capacity_bytes(), 7 * GB);
    assert_eq!(p.outstanding_bytes(), 0);
}

#[test]
fn pool_module_path_is_stable() {
    // Cheap guard that the public surface the migrated gates import actually
    // exists under the documented path.
    let _: fn(&str, usize) -> Option<pool::Reservation> = pool::try_reserve_global;
    let _: fn() -> Option<usize> = pool::global_available_bytes;
}
