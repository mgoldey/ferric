//! MWE: does an explicit `[memory] budget_gb` actually reach the RI-MP2
//! sub-config the RPA property paths build?
//!
//! `PdepRpaConfig` carries `memory_budget_bytes`, but `properties.rs` constructs
//! its `RiMp2Config` with a hardcoded `memory_budget_bytes: None` at nine sites
//! (194, 375, 454, 816, 1441, 1730, 1956, 2098, 2221). The user's explicit
//! budget is dropped on the floor there.
//!
//! # What `None` actually means (a correction worth stating precisely)
//!
//! `None` does NOT mean "unlimited". `resolve_budget` falls through:
//! explicit -> `FERRIC_MEM_BUDGET_GB` -> legacy env vars -> 0.8 x detected
//! available RAM -> 2 GiB floor. So the nine sites do not escape the budget
//! system; they discard the user's EXPLICIT budget and silently substitute an
//! env/auto-detected one.
//!
//! That distinction matters for severity. On a 23 GB box with nothing else
//! running, a user who sets `budget_gb = 1` gets roughly 0.8 x available
//! instead — over an order of magnitude more than they asked for — but the
//! process is not literally unbounded. The bug is "your budget is ignored",
//! not "there is no budget".
//!
//! These contracts pin the resolution semantics that the fix depends on, at the
//! `ferric-core` layer where they are cheap and deterministic to test. They are
//! deliberately env-sensitive-free: each asserts a property that holds
//! regardless of what `FERRIC_MEM_BUDGET_GB` happens to be set to, because the
//! test process inherits the ambient environment.

use ferric_core::memory::{resolve_budget, BudgetSource};

/// CONTRACT 1: an explicit budget always wins, and is returned verbatim.
///
/// This is the property the nine `None` sites defeat by construction: if the
/// explicit value never reaches `resolve_budget`, this guarantee is moot.
/// Pinned across a wide range so no clamping or scaling creeps in.
#[test]
fn explicit_budget_wins_and_is_verbatim() {
    for gib in [1usize, 2, 4, 16, 64] {
        let bytes = gib * 1024 * 1024 * 1024;
        let got = resolve_budget(Some(bytes));
        assert_eq!(
            got.bytes, bytes,
            "an explicit {gib} GiB budget must be returned verbatim, got {} bytes",
            got.bytes
        );
        assert_eq!(
            got.source,
            BudgetSource::Explicit,
            "an explicit budget must report BudgetSource::Explicit"
        );
    }
}

/// CONTRACT 2: a tiny explicit budget is honored, not floored to the default.
///
/// The 2 GiB `DEFAULT_BUDGET_BYTES` is a FALLBACK for when nothing resolves --
/// it must never raise a budget the user deliberately set low. A user pinning
/// 64 MiB on a shared box is asking to be constrained, and silently handing
/// them 2 GiB is how a "safe" run still competes for memory it was told not to
/// take.
#[test]
fn tiny_explicit_budget_is_not_floored_to_the_default() {
    let tiny = 64 * 1024 * 1024; // 64 MiB, far below the 2 GiB fallback
    let got = resolve_budget(Some(tiny));
    assert_eq!(
        got.bytes, tiny,
        "a 64 MiB explicit budget must be honored, not raised to the 2 GiB default"
    );
}

/// CONTRACT 3: `None` is not "unlimited".
///
/// Whatever the ambient environment, an unset budget must resolve to a FINITE
/// ceiling from a named source. This is what makes the nine `None` sites a
/// "wrong budget" bug rather than an "unbounded" one -- and it is the reason
/// the fix is to thread the explicit value through, not to add a bound where
/// none existed.
#[test]
fn unset_budget_resolves_to_a_finite_named_ceiling() {
    let got = resolve_budget(None);
    assert!(got.bytes > 0, "an unset budget must resolve to a positive ceiling");
    assert!(
        got.bytes < usize::MAX,
        "an unset budget must be finite, got {}",
        got.bytes
    );
    assert_ne!(
        got.source,
        BudgetSource::Explicit,
        "a None budget must not report itself as explicit"
    );
}

/// CONTRACT 4: a zero explicit budget means "unset", not "no memory".
///
/// Guards the boundary between "the user pinned a budget" and "the field is
/// defaulted". Zero must fall through to the resolution chain rather than
/// producing a 0-byte ceiling that would starve every allocation.
#[test]
fn zero_explicit_budget_falls_through_to_resolution() {
    let got = resolve_budget(Some(0));
    assert!(
        got.bytes > 0,
        "a zero explicit budget must fall through to the chain, not yield 0"
    );
    assert_ne!(
        got.source,
        BudgetSource::Explicit,
        "zero must be treated as unset, not as an explicit 0-byte budget"
    );
}
