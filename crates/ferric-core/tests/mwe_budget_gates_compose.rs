//! MWE: a budget gate must account for what is ALREADY resident.
//!
//! # The defect
//!
//! Every `check_alloc`-style gate in ferric compares its allocation against the
//! WHOLE budget:
//!
//! ```text
//! if needed <= budget { /* admit */ }
//! ```
//!
//! So two stages that are simultaneously live each pass independently and the
//! process holds their SUM. `plan.rs`'s own module doc names this first among
//! the three consequences it was written to fix:
//!
//! > **The gates do not compose.** Two stages that each independently pass
//! > `check_alloc` still OOM when run back to back, because neither subtracts
//! > what the other left resident. Exactly one site in the tree accounts for
//! > prior residency (`ferric_rpa::energy::quad_panel_width`); everywhere else
//! > the `Share` enum stands in with an author-time guess.
//!
//! The concrete instance from the audit: a KS-DFT job holds the DF 3-index
//! tensor AND the grid AO cache at the same time, and each is sized against
//! 100% of `budget_gb`.
//!
//! # What PySCF does, and why it is the right model
//!
//! `pyscf/df/df.py:164` subtracts live usage before deciding:
//!
//! ```python
//! max_memory = self.max_memory - lib.current_memory()[0]
//! if (nao_pair*naux*8/1e6 < .9*max_memory and not is_custom_storage):
//! ```
//!
//! Two ideas worth stealing: subtract what is already resident
//! (`current_memory()` reads `/proc/self/statm`), and keep a fraction back
//! (`.9`) so the admitted allocation is not the last byte available.
//!
//! ferric already has the reader — `memory::read_own_rss_bytes()`, parsing
//! `/proc/self/status` — but uses it ONLY observationally, in
//! `warn_if_rss_over`, which fires *after* a stage has already allocated.
//!
//! # Scope
//!
//! This pins the arithmetic of a new `available_budget_bytes()` helper. It is
//! deliberately a pure function of (budget, rss) so it can be tested without
//! allocating anything: the live-RSS reader is the caller's business.

use ferric_core::memory::available_budget_bytes;

const GIB: usize = 1024 * 1024 * 1024;

/// CONTRACT 1: resident bytes are subtracted from the ceiling.
///
/// The defect itself. A gate that ignores prior residency admits a second
/// allocation the process cannot actually hold.
#[test]
fn resident_bytes_are_subtracted() {
    let budget = 8 * GIB;
    let full = available_budget_bytes(budget, Some(0));
    let half = available_budget_bytes(budget, Some(4 * GIB));
    assert!(
        half < full,
        "4 GiB already resident must reduce what is still available (got {half} vs {full}). \
         Otherwise two co-resident stages each pass against the same 8 GiB and the process \
         holds 16."
    );
    assert!(
        half <= 4 * GIB,
        "with 4 of 8 GiB resident, at most 4 GiB can remain, got {half}"
    );
}

/// CONTRACT 2: a headroom fraction is held back.
///
/// PySCF's `.9`. Admitting an allocation that exactly fills the remaining
/// budget leaves nothing for the temporaries the consumer will build on top of
/// it, so the gate should never hand out the last byte.
#[test]
fn a_headroom_fraction_is_reserved() {
    let budget = 8 * GIB;
    let avail = available_budget_bytes(budget, Some(0));
    assert!(
        avail < budget,
        "with nothing resident the available figure ({avail}) must still be strictly below \
         the raw budget ({budget}) — a headroom fraction is reserved so the admitted \
         allocation is not the last byte."
    );
    assert!(
        avail >= budget / 2,
        "the headroom must be a modest fraction, not a halving: {avail} of {budget}"
    );
}

/// CONTRACT 3: over-subscription floors at zero rather than wrapping.
///
/// `budget - rss` underflows when a stage has already exceeded the ceiling —
/// which is exactly the situation `warn_if_rss_over` exists to report, so it is
/// reachable. A wrapped `usize` would become a near-infinite budget and admit
/// everything: the precise inversion of what the gate is for.
#[test]
fn exceeding_the_budget_yields_zero_not_a_wrap() {
    let budget = 4 * GIB;
    let avail = available_budget_bytes(budget, Some(6 * GIB));
    assert_eq!(
        avail, 0,
        "with 6 GiB resident against a 4 GiB budget the available figure must be 0, got \
         {avail}. A wrapping subtraction would yield ~1.8e19 and admit any allocation."
    );
}

/// CONTRACT 4: an unreadable RSS must not tighten the budget.
///
/// `read_own_rss_bytes` returns `None` on any platform or container where
/// `/proc/self/status` cannot be read. Treating that as "0 available" would
/// refuse every job on those systems; treating it as "0 resident" preserves
/// exactly the previous behaviour. Observability must never break the
/// computation it watches — the same rule `warn_if_rss_over` already follows.
#[test]
fn an_unreadable_rss_falls_back_to_the_plain_budget() {
    let budget = 8 * GIB;
    assert_eq!(
        available_budget_bytes(budget, None),
        available_budget_bytes(budget, Some(0)),
        "an unreadable RSS must behave as though nothing is resident, not as though \
         everything is"
    );
}

/// CONTRACT 5: monotone in both arguments.
///
/// A bigger budget never yields less available; more resident never yields
/// more. Cheap, and it catches a sign error or a swapped argument that the
/// point checks above could miss.
#[test]
fn the_result_is_monotone_in_budget_and_residency() {
    for rss in [0usize, GIB, 3 * GIB] {
        let mut last = 0usize;
        for gb in 1..=8 {
            let a = available_budget_bytes(gb * GIB, Some(rss));
            assert!(a >= last, "not monotone in budget at {gb} GiB, rss={rss}: {a} < {last}");
            last = a;
        }
    }
    let budget = 8 * GIB;
    let mut last = usize::MAX;
    for gb in 0..=8 {
        let a = available_budget_bytes(budget, Some(gb * GIB));
        assert!(a <= last, "not monotone in residency at {gb} GiB: {a} > {last}");
        last = a;
    }
}
