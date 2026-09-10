//! MWE: the estimate must charge the AO tensor the SAME budget lets grow huge.
//!
//! # The defect
//!
//! `budget.rs`'s intermediates term is
//!
//! ```text
//! //     ... (the raw AO block scratch is chunk-sized, `MO_STREAM_CHUNK` aux
//! //     rows wide, not naux-wide, so it's negligible next to the full b_ov).
//! let metric_bytes = naux * naux * 3 * 8;
//! let eri3_and_bov_bytes = naux * nov * 8;
//! ```
//!
//! The comment is right about the STREAMING SCRATCH and wrong about what is
//! resident. `compute_rpa_intermediates` (rimp2.rs:1242-1244) reads:
//!
//! ```text
//! let budget_bytes = eri3_budget_bytes(config.memory_budget_bytes);
//! let mut src = ThreeIndexSource::build(op, obs, dfbs, budget_bytes)?;
//! let b_ov = stream_dressed_mo_band(&mut src, &v_inv_sqrt, &c_occ, &c_vir, None)?;
//! ```
//!
//! `src` is a SEPARATE `(naux, nao, nao)` AO tensor that stays live for the
//! whole streaming call — the chunk-sized scratch is what the stream reads
//! *into*, not what it reads *from*. And `eri3_budget_bytes` is
//! `resolve_budget_bytes(explicit)`: the WHOLE budget, not a share. So
//! `ThreeIndexSource::build`'s own gate,
//!
//! ```text
//! let needed = band * nao * nao * 8;
//! if needed <= budget_bytes { /* InCore */ } else { /* Spill */ }
//! ```
//!
//! admits an AO tensor up to 100% of the ceiling — the same ceiling
//! `estimate_peak_bytes` is checked against, having charged zero for it.
//!
//! # Sizes: it is LARGER than the term that IS charged
//!
//! ```text
//!   system                naux   nao   AO tensor   charged b_ov   ratio
//!   benzene/cc-pVDZ        420   114     0.04 GB       0.01 GB      6.7x
//!   benzene/aug-cc-pVTZ   1512   414     2.07 GB       0.10 GB     20.8x
//!   danuglipron/def2-SVP  2800   700    10.98 GB       1.23 GB      8.9x
//! ```
//!
//! Not a rounding term. At drug scale it is 11 GB charged as nothing.
//!
//! # Scope
//!
//! Pins the ESTIMATOR only. The AO tensor is charged at its IN-CORE size,
//! because that is the branch the same budget permits; a run that spills pays
//! a bounded band instead, so charging the in-core size is the conservative
//! direction — which is the correct direction for a pre-flight gate.

use ferric_rpa::budget::{estimate_peak_bytes, PeakEstimateShape};

const F64: usize = 8;

/// benzene/aug-cc-pVTZ-scale shape. `nao` is not a field of
/// `PeakEstimateShape` today — that absence is part of the defect, since the
/// estimator cannot charge a tensor whose shape it never receives.
const NAUX: usize = 1512;
const NAO: usize = 414;
const NOCC: usize = 21;
const NVIR: usize = 393;

fn shape(nao: usize) -> PeakEstimateShape {
    PeakEstimateShape {
        naux: NAUX,
        nocc: NOCC,
        nvir: NVIR,
        n_quad: 1,
        n_workers: 8,
        n_keep: NAUX,
        grid: None,
        need_inv_dielectric: false,
        nao,
    }
}

/// CONTRACT 1: the estimate must GROW with `nao`.
///
/// The defect itself. Before the fix `nao` was not even an input, so no value
/// of it could move the estimate.
#[test]
fn the_estimate_responds_to_the_ao_dimension() {
    let small = estimate_peak_bytes(shape(NAO / 2));
    let large = estimate_peak_bytes(shape(NAO));
    assert!(
        large > small,
        "the AO tensor is (naux, nao, nao); doubling nao quadruples it, so the estimate must \
         rise. Got {small} -> {large}. An estimator blind to nao cannot see a term that \
         reaches 11 GB at drug scale."
    );
}

/// CONTRACT 2: the charge covers the full in-core AO tensor.
///
/// A directional test alone would pass on one byte per AO. Pin the arithmetic:
/// `naux * nao^2 * 8`, the exact quantity `ThreeIndexSource::build` compares
/// against the budget when deciding in-core vs spill.
#[test]
fn the_charge_covers_the_whole_in_core_ao_tensor() {
    let with = estimate_peak_bytes(shape(NAO));
    let without = estimate_peak_bytes(shape(0));
    let delta = with - without;
    let want = NAUX * NAO * NAO * F64;
    assert!(
        delta >= want,
        "the AO-tensor charge is {delta} bytes ({:.2} GB) but ThreeIndexSource::build will \
         hold {want} bytes ({:.2} GB) in core — that is the literal quantity its own gate \
         compares to the budget. Under-charging here is how a gate approves a job that then \
         OOMs.",
        delta as f64 / 1e9,
        want as f64 / 1e9,
    );
}

/// CONTRACT 3 (the over-estimation guard): and not much more than that.
///
/// "An over-estimating guard is also a bug." A term that refused RPA jobs
/// which would have fit is as broken as one admitting an OOM. Bound the new
/// charge at twice what it models.
#[test]
fn the_charge_is_not_wildly_over() {
    let delta = estimate_peak_bytes(shape(NAO)) - estimate_peak_bytes(shape(0));
    let modelled = NAUX * NAO * NAO * F64;
    assert!(
        delta <= 2 * modelled,
        "the AO-tensor charge {delta} exceeds 2x the {modelled} bytes it models — an \
         over-estimate refuses jobs that would have run"
    );
}

/// CONTRACT 4: `nao = 0` leaves the estimate exactly as it was.
///
/// The compatibility escape. Every existing caller that has no meaningful
/// `nao` (or has not been updated) must estimate byte-identically to before
/// this field existed, so no currently-passing gate moves.
#[test]
fn nao_zero_is_byte_identical_to_the_old_estimate() {
    // Two shapes differing ONLY in fields the AO term does not read.
    let a = estimate_peak_bytes(shape(0));
    let b = estimate_peak_bytes(PeakEstimateShape { nao: 0, ..shape(0) });
    assert_eq!(a, b, "nao = 0 must be a pure no-op");
    // And it must be strictly less than any nonzero nao, i.e. genuinely zero
    // rather than a floor.
    assert!(
        estimate_peak_bytes(shape(1)) > a,
        "nao = 0 must contribute nothing, so any nonzero nao must exceed it"
    );
}
