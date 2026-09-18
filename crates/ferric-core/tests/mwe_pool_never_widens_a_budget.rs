//! A pool may NARROW a plan's ceiling. It must never WIDEN one.
//!
//! `MemoryPlan::with_pool` used to assign `pool.available_bytes()` over
//! whatever ceiling the plan already had. That silently discarded a caller's
//! deliberately narrow budget: `[memory] budget_gb = 1` on a box whose pool
//! had 20 GB free produced a 20 GB ceiling, so the one knob a user has to
//! bound a job was ignored exactly when it mattered.
//!
//! ## Artifact hypothesis (stated before measuring, per CLAUDE.md)
//!
//! If the `min` is REAL, a 1-unit explicit budget attached to a 20-unit pool
//! yields a 1-unit ceiling, and a plan whose projected peak is 5 units is
//! REFUSED.
//!
//! If the assignment were still live, the same plan would carry a 20-unit
//! ceiling and the 5-unit peak would be ADMITTED. The two predictions differ
//! in both the ceiling and the accept/reject outcome, so this test can tell
//! them apart -- it is not merely re-reading the value it set.

use ferric_core::memory::plan::{Lifetime, MemoryPlan};
use ferric_core::memory::pool::MemoryPool;

const MB: usize = 1_000_000;
/// One megabyte, named so the narrow-budget cases read in the same units as
/// their `20 * MB` counterparts without tripping clippy's `identity_op`.
const ONE_MB: usize = MB;

#[test]
fn a_pool_with_room_to_spare_does_not_raise_an_explicit_budget() {
    let pool = MemoryPool::with_capacity_bytes(20 * MB);
    let plan = MemoryPlan::with_budget_bytes(ONE_MB, "narrow caller").with_pool(&pool);
    assert_eq!(
        plan.budget_bytes(),
        ONE_MB,
        "the caller asked to be held to 1 MB; a 20 MB pool must not hand back \
         20 MB. A budget a caller sets is a ceiling, not a suggestion."
    );
}

#[test]
fn a_pool_tighter_than_the_caller_still_narrows() {
    // The direction the pool exists for must keep working.
    let pool = MemoryPool::with_capacity_bytes(2 * MB);
    let plan = MemoryPlan::with_budget_bytes(20 * MB, "generous caller").with_pool(&pool);
    assert_eq!(
        plan.budget_bytes(),
        2 * MB,
        "a pool with only 2 MB free must narrow a 20 MB caller budget -- that \
         is the composition the pool was built to provide"
    );
}

#[test]
fn an_outstanding_reservation_narrows_what_a_later_plan_sees() {
    let pool = MemoryPool::with_capacity_bytes(20 * MB);
    let _held = pool.reserve("incumbent", 18 * MB).expect("fits");
    let plan = MemoryPlan::with_budget_bytes(20 * MB, "second plane").with_pool(&pool);
    assert_eq!(
        plan.budget_bytes(),
        2 * MB,
        "the second plane must see only what the first LEFT (20 - 18), which \
         is the whole point of a debited ledger"
    );
}

#[test]
fn the_narrow_budget_actually_refuses_a_plan_that_exceeds_it() {
    // The ceiling is not cosmetic: it must change the accept/reject outcome.
    // This is the half of the artifact hypothesis that the assignment bug
    // would have inverted.
    let pool = MemoryPool::with_capacity_bytes(20 * MB);
    let mut plan = MemoryPlan::with_budget_bytes(ONE_MB, "narrow caller").with_pool(&pool);
    // `reserve` counts f64 ELEMENTS, not bytes: 5 MB / 8 B.
    plan.reserve("a big plane", 5 * MB / 8, Lifetime::Resident);
    assert!(
        plan.check().is_err(),
        "a 5 MB plane under a 1 MB explicit budget must be REFUSED even though \
         the 20 MB pool could hold it -- otherwise the min is decoration"
    );
}

#[test]
fn an_unbudgeted_plan_still_takes_the_pool_ceiling() {
    // `with_budget_bytes(usize::MAX)` stands in for "no explicit ceiling":
    // min(MAX, pool) must be the pool, or attaching a pool to an unbudgeted
    // plan would be a no-op and nothing would ever be charged.
    let pool = MemoryPool::with_capacity_bytes(7 * MB);
    let plan = MemoryPlan::with_budget_bytes(usize::MAX, "unbudgeted").with_pool(&pool);
    assert_eq!(plan.budget_bytes(), 7 * MB);
}
