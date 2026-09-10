//! MWE: when the caller needs `inv_dielectric_freq`, the estimate must charge it.
//!
//! # The defect
//!
//! `estimate_peak_bytes` discards `n_quad` outright:
//!
//! ```text
//! let n_quad_unused_guard = n_quad; // n_quad only affects wall-time, not peak
//! let _ = n_quad_unused_guard;      // resident bytes here ...
//! ```
//!
//! `mwe_estimator_sees_grid.rs` even documents that as correct, under the
//! heading "A note on what is NOT a defect", with the instruction "Do not
//! 'fix' it". **That is right for the ENERGY path and wrong for GW**, which is
//! the distinction this file adds.
//!
//! On the energy path, quadrature points are processed per worker via
//! `map_init` and never retained, so `n_quad` really does drive wall time only.
//! But `run_pdep_rpa` has a second mode:
//!
//! ```text
//! let inv_dielectric_freq = match (config.need_inv_dielectric_freq, ..) {
//!     (true, None) => Some(energy::eval_inv_dielectric_matrices(..)),
//!     _ => None,
//! };
//! ```
//!
//! and `eval_inv_dielectric_matrices` ends in `per_frequency(..)`, whose
//! `.collect::<Result<Vec<T>, _>>()` RETAINS one owned `(m, m)` matrix per
//! frequency. Under that flag `n_quad` is a peak-resident multiplier.
//!
//! It is not an exotic mode. `ferric_gw::with_inv_dielectric` forces the flag
//! on for EVERY GW method, because Σ_c reads the stack:
//!
//! ```text
//! fn with_inv_dielectric(cfg: &PdepRpaConfig) -> PdepRpaConfig {
//!     let mut c = cfg.clone();
//!     c.need_inv_dielectric_freq = true;
//!     c
//! }
//! ```
//!
//! So every G0W0/evGW/COHSEX/BSE run is pre-flighted by an estimate that
//! charges zero for one of its largest allocations.
//!
//! # Sizes (benzene-dimer/aug-cc-pVTZ, the shape `lib.rs` itself quotes)
//!
//! naux = m = 2976, nov = 61,740, n_quad = 20, 8 workers:
//!
//! ```text
//!   term                                    bytes    charged before?
//!   inv_dielectric stack  n_quad * m^2      1.42 GB  NO
//!   per-worker y clones   min(nq,nw)*m*nov 11.76 GB  NO
//!   y itself              m * nov           1.47 GB  yes
//! ```
//!
//! The per-worker clones are the larger of the two omissions: `per_frequency`
//! calls `dielectric_matrix_from_projection`, which does `let mut rhs_scaled =
//! y.clone()` — a full `(m, nov)` per concurrent worker. A scratch-reusing
//! variant `dielectric_matrix_from_projection_into` already exists next to it
//! and is not used here.
//!
//! # Scope
//!
//! This file pins the ESTIMATOR only. It asserts the charge appears when the
//! flag is set, stays exactly zero when it is not (so no existing energy-path
//! gate moves by a byte), and scales with the terms it models.

use ferric_rpa::budget::{estimate_peak_bytes, PeakEstimateShape};

const F64: usize = 8;

/// Benzene-dimer/aug-cc-pVTZ, the shape `ferric_rpa::lib`'s own comment quotes
/// when it says the stack is "~1.85 GB at dimer/aTZ scale".
fn dimer_atz(n_quad: usize, need_inv: bool) -> PeakEstimateShape {
    PeakEstimateShape {
        // AO tensor not modelled by this case.
        nao: 0,
        naux: 2976,
        nocc: 42,
        nvir: 1470,
        n_quad,
        n_workers: 8,
        n_keep: 2976,
        grid: None,
        need_inv_dielectric: need_inv,
    }
}

/// CONTRACT 1: with the flag OFF, `n_quad` still changes nothing.
///
/// The property `mwe_estimator_sees_grid.rs` protects, preserved verbatim. On
/// the energy path quadrature points are genuinely transient, so an
/// energy-only run must estimate byte-identically at any `n_quad` — otherwise
/// this change would perturb `run_pdep_rpa`/`rs_mp2_rpa`/`ao_rpa`'s existing
/// gates.
#[test]
fn without_the_flag_n_quad_is_still_a_no_op() {
    let base = estimate_peak_bytes(dimer_atz(1, false));
    for nq in [3usize, 7, 12, 20, 64] {
        assert_eq!(
            estimate_peak_bytes(dimer_atz(nq, false)),
            base,
            "energy-path estimate moved with n_quad={nq}. Quadrature points are processed \
             per-worker and never retained there, so this must stay a no-op — see \
             mwe_estimator_sees_grid.rs's \"what is NOT a defect\" note."
        );
    }
}

/// CONTRACT 2: with the flag ON, the estimate GROWS with `n_quad`.
///
/// The defect itself. Fails before the fix, where both branches returned the
/// same number.
#[test]
fn with_the_flag_the_estimate_grows_with_n_quad() {
    let small = estimate_peak_bytes(dimer_atz(4, true));
    let large = estimate_peak_bytes(dimer_atz(20, true));
    assert!(
        large > small,
        "with need_inv_dielectric the retained stack is n_quad * m^2, so raising n_quad \
         from 4 to 20 must raise the estimate; got {small} -> {large}"
    );
}

/// CONTRACT 3: the charge is at least the retained stack plus the y clones.
///
/// A directional test (CONTRACT 2) would pass if the new term were one byte per
/// frequency. Pin the actual arithmetic: `n_quad * m^2` retained, plus
/// `min(n_quad, n_workers) * m * nov` of concurrent `y.clone()` scratch.
#[test]
fn the_charge_covers_the_stack_and_the_per_worker_clones() {
    let s = dimer_atz(20, true);
    let off = estimate_peak_bytes(PeakEstimateShape { need_inv_dielectric: false, ..s });
    let on = estimate_peak_bytes(s);
    let delta = on - off;

    let m = s.n_keep;
    let nov = s.nocc * s.nvir;
    let stack = s.n_quad * m * m * F64;
    let clones = s.n_quad.min(s.n_workers) * m * nov * F64;
    let want = stack + clones;

    assert!(
        delta >= want,
        "the need_inv_dielectric charge is {delta} bytes ({:.2} GB) but the allocations are \
         {want} bytes ({:.2} GB): {stack} for the retained n_quad x m^2 stack ({:.2} GB) and \
         {clones} for the concurrent y clones ({:.2} GB). An estimator that under-charges is \
         how a job the gate approved gets OOM-killed.",
        delta as f64 / 1e9,
        want as f64 / 1e9,
        stack as f64 / 1e9,
        clones as f64 / 1e9,
    );
}

/// CONTRACT 4 (the over-estimation guard): the charge must not be wild.
///
/// "An over-estimating guard is also a bug" — a term that refused GW jobs which
/// would have fit is as broken as one that admits an OOM. Bound the new charge
/// at twice the allocations it models, which leaves room for a reviewer to add
/// a genuine term without letting a fat-fingered factor through.
#[test]
fn the_charge_is_not_wildly_over() {
    let s = dimer_atz(20, true);
    let off = estimate_peak_bytes(PeakEstimateShape { need_inv_dielectric: false, ..s });
    let delta = estimate_peak_bytes(s) - off;

    let m = s.n_keep;
    let nov = s.nocc * s.nvir;
    let modelled = s.n_quad * m * m * F64 + s.n_quad.min(s.n_workers) * m * nov * F64;
    assert!(
        delta <= 2 * modelled,
        "the need_inv_dielectric charge {delta} exceeds 2x the {modelled} bytes it models — \
         an over-estimate refuses jobs that would have run"
    );
}

/// CONTRACT 5: the clone term saturates at the worker count, not at `n_quad`.
///
/// Only `min(n_quad, n_workers)` clones are ever concurrent — rayon runs at
/// most `n_workers` closures at once. Charging `n_quad` of them would grow
/// without bound on a large quadrature grid and refuse jobs that fit, so this
/// pins the saturation directly rather than trusting CONTRACT 4's factor-of-2
/// slack to catch it.
#[test]
fn the_clone_charge_saturates_at_the_worker_count() {
    let charge = |nq: usize| -> usize {
        let s = dimer_atz(nq, true);
        let off = estimate_peak_bytes(PeakEstimateShape { need_inv_dielectric: false, ..s });
        estimate_peak_bytes(s) - off
    };
    let m = 2976usize;
    let stack_of = |nq: usize| nq * m * m * F64;

    // Past the 8-worker saturation point, each extra frequency may add the
    // retained (m,m) matrix but must NOT add another (m,nov) clone.
    let d16 = charge(16) - stack_of(16);
    let d32 = charge(32) - stack_of(32);
    assert_eq!(
        d16, d32,
        "the non-stack part of the charge grew from {d16} to {d32} between n_quad=16 and 32 \
         at 8 workers. At most n_workers clones are ever live, so this term must saturate."
    );
}
