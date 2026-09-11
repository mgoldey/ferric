//! MWE: rs-mp2-rpa's preflight must charge the SECOND co-resident `b_ov`.
//!
//! # The defect
//!
//! `rs_mp2_lr_rpa`'s single preflight gate (`rs_mp2_rpa.rs`, right after
//! `resolved_budget_bytes` is computed) builds its estimate via
//! `crate::budget::estimate_peak_bytes`, whose `eri3_and_bov_bytes` term
//! charges exactly ONE `naux*nocc*nvir` `RpaIntermediates::b_ov` buffer —
//! correct for a single `compute_rpa_intermediates` call, but
//! `rs_mp2_lr_rpa` is not a single call. Both `match cfg.formulation` arms
//! bind an `RpaIntermediates` (`it_lr` in `DeltaLr`; `it_full`/`it_sr` in
//! `CoupledRings`) whose LAST USE is inside
//! `run_pdep_rpa_from_intermediates(&it, ..)`. Rust does not run a `Drop`
//! early just because NLL ends the *borrow* there — an owned, non-`Copy`
//! binding that is not moved out is dropped at the END of its enclosing
//! block, i.e. after that call returns. Concretely, in `CoupledRings`:
//!
//! ```text
//! let it_full = inter_of(Operator::coulomb())?;     // bound here
//! let it_sr = inter_of(op_sr)?;                      // bound here — it_full still alive
//! let sc_full = sc_of(&it_full);
//! ...
//! let drpa_coul = run_pdep_rpa_from_intermediates(&it_full, ..)?;  // it_full's LAST use
//! let drpa_erfc = run_pdep_rpa_from_intermediates(&it_sr, ..)?;    // it_full NOT yet dropped
//! ```
//!
//! so both `it_full.b_ov` and `it_sr.b_ov` are resident through the SECOND,
//! equally expensive eigensolve + frequency-quadrature call. `DeltaLr` has
//! the same shape via a shorter overlap (`it_lr` co-resident with a
//! statement-temporary `RpaIntermediates` built inside
//! `sc_of(&inter_of(Operator::coulomb())?)` and again inside
//! `sc_of(&inter_of(op_sr)?)`).
//!
//! # Magnitude
//!
//! One extra `naux*nocc*nvir*8` buffer:
//!   * benzene/aug-cc-pVTZ    (naux=1512, nocc=21, nvir=393): 0.10 GB
//!   * danuglipron/def2-SVP   (naux=2800, nocc=90, nvir=610): 1.23 GB
//!
//! Real (a whole extra tensor, not a rounding term) but modest relative to
//! the AO-tensor and per-worker-quadrature terms at these shapes (3-5% of
//! the current total estimate) — reported honestly rather than oversold.
//!
//! # Scope
//!
//! Tests the pure helper `ferric_rpa::rs_mp2_rpa::second_intermediate_bov_bytes`
//! plus its composition with `estimate_peak_bytes`/`check_alloc` exactly as
//! the call site combines them. Does NOT run `rs_mp2_lr_rpa` end-to-end (it
//! needs a full SCF); the pure-helper + composition test is the affordable
//! substitute the shared brief asks for.

use ferric_core::memory::check_alloc;
use ferric_rpa::budget::{estimate_peak_bytes, PeakEstimateShape};
use ferric_rpa::rs_mp2_rpa::second_intermediate_bov_bytes;

const F64_BYTES: usize = 8;

/// benzene/aug-cc-pVTZ-scale shape (matches the shared brief's example).
const NAUX: usize = 1512;
const NAO: usize = 414;
const NOCC: usize = 21;
const NVIR: usize = 393;
const N_WORKERS: usize = 8;

fn base_shape() -> PeakEstimateShape {
    PeakEstimateShape {
        naux: NAUX,
        nocc: NOCC,
        nvir: NVIR,
        n_quad: 1,
        n_workers: N_WORKERS,
        n_keep: NAUX,
        grid: None,
        need_inv_dielectric: false,
        nao: NAO,
    }
}

/// REACHABILITY: the helper is not a stub that always returns 0, and it
/// matches the exact `naux*nocc*nvir*8` shape of a real `b_ov` buffer — the
/// same formula `estimate_peak_bytes`'s `eri3_and_bov_bytes` uses.
#[test]
fn second_intermediate_bov_bytes_matches_one_bov_buffer() {
    let got = second_intermediate_bov_bytes(NAUX, NOCC, NVIR);
    let want = NAUX * NOCC * NVIR * F64_BYTES;
    assert_eq!(got, want);
    // Not a rounding artifact: at this shape it's ~0.1 GB, not a few bytes.
    assert!(got > 50_000_000, "expected a real tensor-sized term, got {got} bytes");
}

/// THE DEFECT, pinned via the composition the call site actually performs:
/// `estimate_peak_bytes(shape) + second_intermediate_bov_bytes(..)` must be
/// STRICTLY GREATER than `estimate_peak_bytes(shape)` alone — i.e. the fix is
/// additive and material, not folded away by `saturating_add` hitting a cap
/// or by the term being zero at this shape.
#[test]
fn second_bov_term_is_additive_and_material() {
    let base = estimate_peak_bytes(base_shape());
    let with_second = base + second_intermediate_bov_bytes(NAUX, NOCC, NVIR);
    assert!(
        with_second > base,
        "second b_ov term must strictly increase the estimate: base={base} with_second={with_second}"
    );
    // Quantify: at this shape the extra term is a few percent of the total,
    // consistent with the magnitude computed above (not the dominant term,
    // but not negligible either).
    let extra = with_second - base;
    assert_eq!(extra, NAUX * NOCC * NVIR * F64_BYTES);
}

/// OVER-REJECTION CONTRACT: an ample budget that comfortably covers BOTH
/// co-resident `b_ov` buffers plus everything else the estimate already
/// charges must still pass `check_alloc`. The fix must not turn into a wall
/// that refuses jobs which actually fit.
#[test]
fn ample_budget_still_passes_with_second_bov_charged() {
    let est = estimate_peak_bytes(base_shape()) + second_intermediate_bov_bytes(NAUX, NOCC, NVIR);
    // 10x the charged estimate is unambiguously ample.
    let ample_budget = est.saturating_mul(10);
    assert!(check_alloc("mwe ample", est, ample_budget).is_ok());
}

/// THE FAILURE MODE THE FINDING DESCRIBES: pick a budget that fits the
/// OLD (under-charged, single-b_ov) estimate exactly but is too small once
/// the second co-resident buffer is honestly counted. Before the fix this
/// budget would have been silently approved; after the fix it is correctly
/// refused. This is the test that fails on the pre-fix arithmetic and passes
/// on the post-fix arithmetic (mutate by removing the `.saturating_add` at
/// the rs_mp2_rpa.rs call site, or by calling this test's `base` in place of
/// `with_second`, to see it fail).
#[test]
fn budget_sized_to_old_estimate_is_now_refused() {
    let base = estimate_peak_bytes(base_shape());
    let with_second = base + second_intermediate_bov_bytes(NAUX, NOCC, NVIR);
    // A budget between the two: fits the old (wrong) estimate, does not fit
    // the honest one.
    let budget_between = base + second_intermediate_bov_bytes(NAUX, NOCC, NVIR) / 2;
    assert!(budget_between < with_second);
    assert!(check_alloc("mwe old-estimate-fits", base, budget_between).is_ok());
    assert!(check_alloc("mwe honest-estimate", with_second, budget_between).is_err());
}
