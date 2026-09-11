//! MWE: the blocked/spilled DF-dressing over-budget warning must fire
//! UNCONDITIONALLY on `check()` failure, not only under `FERRIC_OOC_TRACE`.
//!
//! # Verifying the finding against current source
//!
//! `build_dressed_band`'s SEQUENTIAL (blocked/spilled) path builds a
//! `MemoryPlan` reserving the in-core band (when `in_core`) plus TWO
//! `block_naux·nao²` blocks (`accum` and the per-iteration `contrib` GEMM
//! result), matching the doc comment's claim that this pair is "≈2× budget by
//! construction" on a spilled source (`block_naux` is sized to the WHOLE
//! budget via `block_naux_for`, unlike the RAW path's `spill_block_naux_for`,
//! which halves it for exactly this two-live-blocks reason).
//!
//! Before this fix, the report was only printed when
//! `plan.check().is_err() && ooc_trace()` — and `ooc_trace()` reads
//! `FERRIC_OOC_TRACE`, an env var that defaults to `false`. So in every
//! production run (no one sets a debug trace var) the honest overshoot
//! report the comment promises ("the honest figure is REPORTED... rather
//! than enforced") was computed and then silently thrown away.
//!
//! # Scope: the decision logic, not the eprintln! side effect
//!
//! Intercepting real stderr output from a `#[test]` cheaply (no new
//! dependency, no subprocess) is not practical in this crate, so this suite
//! pins the DECISION the warning is conditioned on via
//! `blocked_dressing_overshoot_report_for_test` — the pure function
//! extracted from the call site that mirrors its exact `MemoryPlan`
//! reservations. The actual `eprintln!` at the call site is now a one-line,
//! visibly-unconditional wrapper around `Option<String>` from this function
//! (`if let Some(report) = ... { eprintln!(...) }`), so pinning "the
//! function returns `Some` exactly when the shape overshoots" is equivalent
//! to pinning "the warning fires exactly when the shape overshoots" for any
//! reader who checks the call site by eye. This is stated rather than
//! implied: this suite does NOT capture stderr text.

use ferric_integrals::three_index_source::blocked_dressing_overshoot_report_for_test as overshoot;

const NAO: usize = 900; // danuglipron/def2-SVP-scale AO count

/// CONTRACT 1 (reachability / exactness anchor): a shape sized so the two
/// live blocks are each ~half the budget must overshoot — this is the
/// documented "≈2x budget by construction" regime, not a rounding artifact.
#[test]
fn two_live_blocks_at_half_budget_each_overshoots() {
    let nao = NAO;
    let row_bytes = nao * nao * 8;
    let budget = 10 * row_bytes; // budget fits exactly 10 aux rows of ONE block
    let block_naux = 10; // one block sized to (nearly) the whole budget
    let report = overshoot(budget, false, 0, block_naux, nao);
    assert!(
        report.is_some(),
        "two co-resident block_naux={block_naux} blocks at budget={budget} \
         (exactly 1x one block) must be reported as overshooting"
    );
}

/// CONTRACT 2 (over-rejection guard): a block sized to a THIRD of the budget
/// (so two of them plus slack still fit) must NOT be reported as
/// overshooting. A warning that fires unconditionally regardless of shape
/// would be as useless as one that never fires.
#[test]
fn small_block_relative_to_budget_does_not_overshoot() {
    let nao = NAO;
    let row_bytes = nao * nao * 8;
    let budget = 10 * row_bytes;
    let block_naux = 3; // two blocks = 6 rows worth, well under the 10-row budget
    let report = overshoot(budget, false, 0, block_naux, nao);
    assert!(
        report.is_none(),
        "two co-resident block_naux={block_naux} blocks should fit comfortably \
         inside budget={budget} and must not be reported"
    );
}

/// CONTRACT 3: the in-core additional reservation (the resident band) also
/// participates in the decision — this is the same plan construction as the
/// in-core dressing branch, just with an extra term.
#[test]
fn in_core_band_reservation_can_tip_an_otherwise_fitting_shape_over_budget() {
    let nao = NAO;
    let row_bytes = nao * nao * 8;
    let budget = 10 * row_bytes;
    let block_naux = 3; // fits alone (see CONTRACT 2)
    let band = 5; // + 5 more rows resident -> 6 (blocks) + 5 (band) = 11 > 10
    let without_band = overshoot(budget, false, 0, block_naux, nao);
    let with_band = overshoot(budget, true, band, block_naux, nao);
    assert!(without_band.is_none(), "sanity: without the band this shape must fit");
    assert!(
        with_band.is_some(),
        "adding the in-core band reservation must push this shape over budget"
    );
}

/// CONTRACT 4 (report content sanity): when present, the report must mention
/// both co-resident block terms by name, so an operator reading stderr can
/// tell which reservations are responsible (the whole point of using
/// `MemoryPlan::report()` instead of a bare "over budget" message).
#[test]
fn report_names_both_co_resident_blocks() {
    let nao = NAO;
    let row_bytes = nao * nao * 8;
    let budget = 10 * row_bytes;
    let report = overshoot(budget, false, 0, 10, nao).expect("this shape overshoots (CONTRACT 1)");
    assert!(
        report.contains("accum"),
        "report must name the accum block:\n{report}"
    );
    assert!(
        report.contains("contrib"),
        "report must name the contrib GEMM block:\n{report}"
    );
}
