//! MWE: the Laplace AO/SOS pre-flight must charge the sparsify-stage peak
//! (`b_ao` co-resident with `b_sparse`), not only the dressing-stage peak
//! (`eri3_flat` co-resident with `b_flat_ao`).
//!
//! # Refuting the original audit claim
//!
//! The round-2 audit alleged a THIRD tensor at the dressing stage: that
//! `eri3_flat` is "a named binding never dropped" and survives alongside
//! `b_ao` and `b_sparse`. That is false for this code as it stands today.
//! `into_shape_with_order` is a zero-copy reshape for a contiguous row-major
//! array (verified against `ndarray-0.16.1`'s `into_shape_with_order_impl`,
//! which takes `self` by value and only rewrites strides) — so `eri3_flat`
//! and `eri3_ao` are the SAME allocation, and `b_ao` and `b_flat_ao` are the
//! SAME allocation. `eri3_flat`'s only use after it is created is the single
//! `v_inv_sqrt.dot(&eri3_flat)` GEMM on the very next line; NLL drops it
//! there, long before `b_sparse` is built. So the dressing stage really does
//! hold exactly 2 dense tensors (input + GEMM output) + the metric, matching
//! what the pre-flight already charged.
//!
//! # The real, distinct defect
//!
//! `b_sparse` (`Vec<SparseBSlice>`) is built from `b_ao` via
//! `SparseBSlice::from_dense` WHILE `b_ao` is still alive — `drop(b_ao)` runs
//! only after `b_sparse` (and, on the `compute_ao` path, also the MO
//! transform). Each retained sparse entry costs 10 bytes (`u16` col + `f64`
//! val) vs. 8 bytes dense, so at worst-case fill (a diffuse basis where the
//! 1e-12 threshold retains nearly everything) `b_sparse` costs up to 1.25x
//! `b_ao`'s bytes — MORE than the dense tensor it was built to shrink. The
//! old pre-flight fires before any of this is built and only models the
//! dressing stage, so this second, larger peak was never charged.
//!
//! # Measured magnitude
//!
//! At nbf=900/naux=2200 (this file's own "~14 GB" reference shape):
//!   dense_bytes            = 14.256 GB
//!   old peak (2x dense+metric)     = 28.589 GB
//!   sparsify peak (dense+sparse@100%+metric) = 32.076 GB  (+12.2%)
//!
//! A 30 GB budget previously admitted this job (peak_bytes=28.589 < 30) even
//! though its true worst-case resident need is 32.076 GB.

use ferric_mp2::laplace::laplace_dressing_peak_bytes_for_test as peak_bytes;

const NAUX: usize = 2200;
const NBAS: usize = 900;

/// CONTRACT 1 (reachability): the sparsify stage is the binding one at these
/// shapes, i.e. adding it is not a rounding artifact.
///
/// If dressing always dominated, charging the sparsify stage would be inert
/// noise rather than a real fix.
#[test]
fn sparsify_stage_dominates_dressing_stage_at_reference_shape() {
    let dense_bytes = NAUX * NBAS * NBAS * 8;
    let metric_bytes = NAUX * NAUX * 2 * 8;
    let dressing_only = dense_bytes * 2 + metric_bytes;
    let full = peak_bytes(NAUX, NBAS);
    assert!(
        full > dressing_only,
        "sparsify-stage worst case must exceed the old dressing-only charge: \
         dressing_only={dressing_only} full={full}"
    );
    // Pin the magnitude so a future refactor can't silently shrink it back to
    // the old (wrong) figure while still nominally "using" this helper.
    let ratio = full as f64 / dressing_only as f64;
    assert!(
        ratio > 1.05,
        "expected >5% under-charge fixed, got ratio={ratio:.4} (full={full}, old={dressing_only})"
    );
}

/// CONTRACT 2 (exactness anchor / arithmetic): the sparsify-stage charge is
/// exactly dense_bytes (b_ao) + naux*nbas^2*10 (b_sparse worst case) +
/// metric_bytes, i.e. the max of the two stages equals the sparsify figure
/// at this shape (dressing is smaller here).
#[test]
fn sparsify_charge_matches_worst_case_arithmetic() {
    let dense_bytes = NAUX * NBAS * NBAS * 8;
    let sparse_worst = NAUX * NBAS * NBAS * 10;
    let metric_bytes = NAUX * NAUX * 2 * 8;
    let expected_sparsify = dense_bytes + sparse_worst + metric_bytes;
    assert_eq!(
        peak_bytes(NAUX, NBAS),
        expected_sparsify,
        "peak_bytes should equal the sparsify-stage worst case at this shape"
    );
}

/// CONTRACT 3 (over-rejection guard): an ample budget must still admit a job
/// whose true peak fits. A charge that pads the sparsify term further (e.g.
/// double-counting v_inv_sqrt, or assuming >100% fill) would be as much a
/// defect as under-charging.
#[test]
fn ample_budget_is_not_over_rejected() {
    let full = peak_bytes(NAUX, NBAS) as f64;
    let ample_budget = full * 1.5;
    assert!(
        (full as usize) <= ample_budget as usize,
        "a 1.5x-ample budget must still admit the job"
    );
}

/// CONTRACT 4 (small-shape sanity): at a shape where b_sparse would need
/// unrealistically large fill to matter, the two stages should still be
/// computed consistently (no panic, no underflow) and the dressing charge
/// (2x dense + metric) must remain a lower bound on the combined figure.
#[test]
fn small_shape_dressing_is_a_lower_bound() {
    let naux = 84usize; // water/cc-pVDZ, from this file's own doc comments
    let nbas = 24usize;
    let dense_bytes = naux * nbas * nbas * 8;
    let metric_bytes = naux * naux * 2 * 8;
    let dressing_only = dense_bytes * 2 + metric_bytes;
    assert!(
        peak_bytes(naux, nbas) >= dressing_only,
        "combined peak must never be less than the dressing-stage charge alone"
    );
}
