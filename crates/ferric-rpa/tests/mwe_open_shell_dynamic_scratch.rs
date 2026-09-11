//! MWE: the open-shell dynamic-Becke preflight must charge BOTH spins' b_ov
//! and BOTH live naux² buffers, not one of each.
//!
//! # The defect
//!
//! `pdep_polarizability_becke_dynamic`'s open-shell branch (properties.rs)
//! builds two complete `RpaIntermediates` (`inter_a`, `inter_b`) and keeps
//! BOTH resident for the whole frequency loop. Its preflight call to
//! `preflight_grid_path` passes `(nocc, nvir)` = the LARGER spin channel, so
//! `estimate_peak_bytes`'s `nov = nocc*nvir` term — which structurally takes
//! only ONE `(naux, nocc, nvir)` shape — counts exactly one spin's resident
//! `b_ov` and one worker-local `(naux, nov) + (naux, naux)` scratch pair.
//!
//! The real per-worker `map_init` closure seeds a `(b_scaled_a, b_scaled_b)`
//! PAIR (one `(naux, nov_a)` and one `(naux, nov_b)` buffer) and forms a
//! single `eps_mat` (naux²) that is built via `eps_mat += &chi_s` inside a
//! loop over both spins — so a second, transient `chi_s` (naux²) is live at
//! the moment of that `+=`. Net: two `(naux, nov_σ)` buffers and two naux²
//! buffers per worker, not one of each. A comment at the call site used to
//! claim "the per-worker frequency scratch is sized from one channel at a
//! time" — that claim is false; see properties.rs's `map_init` seed.
//!
//! `molecular_dynamic_polarizability`'s open-shell branch has the identical
//! shape (same `map_init` pattern, same `eps_mat += ...` accumulation) and,
//! unlike the Becke path, had NO preflight of any kind before this fix.
//!
//! This file tests the pure helper `open_shell_dynamic_extra_bytes`, which
//! both call sites now use to charge what `preflight_grid_path`/
//! `preflight_molecular_path` structurally cannot express on their own. It is
//! a pure function of `(naux, nov_min, n_workers)`, so these contracts run in
//! microseconds — no SCF, no grid, no basis needed.
//!
//! # What is NOT covered end-to-end
//!
//! There is no cheap fixture that reaches the open-shell frequency loop
//! itself (it needs a converged UHF/ROHF reference plus RI intermediates).
//! These contracts pin the arithmetic `properties.rs` calls into
//! `check_alloc`, not the allocation at the call site — that is the same
//! testing boundary `mwe_per_worker_scratch.rs` and
//! `mwe_rs_mp2_rpa_second_intermediate.rs` already accept for this family of
//! defect.

use ferric_rpa::properties::open_shell_dynamic_extra_bytes;

const NAUX: usize = 1000;
const NOV_MIN: usize = 20_000;
const F64_BYTES: usize = 8;

/// CONTRACT 1 (REACHABILITY): the extra term is not a rounding artifact.
///
/// At the audit's shape (naux=1000, nov=20000/spin, 12 workers), the
/// currently-charged total is ~2.18 GB; the true resident peak is ~4.35 GB.
/// The extra term alone must therefore be on the order of GIGABYTES, not
/// bytes -- a helper that returns near-zero here would pass CONTRACT 2/3
/// below without actually fixing anything (a test that cannot distinguish
/// "inert" from "fixed" is worthless per this repo's testing conventions).
#[test]
fn the_extra_term_is_gigabytes_not_a_rounding_artifact() {
    let workers = 12;
    let extra = open_shell_dynamic_extra_bytes(NAUX, NOV_MIN, workers);
    let one_gb = 1_000_000_000usize;
    assert!(
        extra > one_gb,
        "expected the uncharged second-spin scratch to exceed 1 GB at \
         naux={NAUX}, nov_min={NOV_MIN}, workers={workers}; got {:.3} GB. This \
         term is supposed to close a ~2x under-estimate, not a rounding term.",
        extra as f64 / 1e9
    );
}

/// CONTRACT 2: the extra term matches the hand-derived formula exactly.
///
/// extra = naux*nov_min*8 (second spin's resident b_ov)
///       + workers * (naux*nov_min + naux*naux) * 8 (second spin's per-worker
///         b_scaled plus the second live naux² buffer)
///
/// Pinning the exact value (not just "> 0") means a mutation that drops the
/// per-worker term, or the naux² term, or forgets to multiply by workers,
/// changes this number and fails the test -- see the mutations list in the
/// report.
#[test]
fn the_extra_term_matches_the_hand_derived_formula() {
    let workers = 8;
    let expected_bov = NAUX * NOV_MIN * F64_BYTES;
    let expected_per_worker = (NAUX * NOV_MIN + NAUX * NAUX) * F64_BYTES;
    let expected = expected_bov + expected_per_worker * workers;
    let got = open_shell_dynamic_extra_bytes(NAUX, NOV_MIN, workers);
    assert_eq!(
        got, expected,
        "open_shell_dynamic_extra_bytes({NAUX}, {NOV_MIN}, {workers}) = {got}, expected {expected}"
    );
}

/// CONTRACT 3: the term SCALES with worker count.
///
/// The per-worker piece must dominate at realistic worker counts and must
/// grow linearly with them -- a flat number here would silently reintroduce
/// the "memory scales with cores, gate does not" failure mode this repo's
/// budget module exists to prevent.
#[test]
fn the_extra_term_scales_with_worker_count() {
    let one = open_shell_dynamic_extra_bytes(NAUX, NOV_MIN, 1);
    let twelve = open_shell_dynamic_extra_bytes(NAUX, NOV_MIN, 12);
    assert!(
        twelve > one,
        "expected the extra term to grow with worker count: 1 worker gives {one}, 12 workers \
         gives {twelve}"
    );
    // Roughly proportional in the per-worker component (the resident b_ov
    // term is worker-independent, so the ratio is sublinear but must still
    // be well above 1x).
    assert!(
        twelve as f64 / one as f64 > 3.0,
        "expected a strong (>3x) increase from 1 to 12 workers given the per-worker term \
         dominates at this shape; got {:.2}x",
        twelve as f64 / one as f64
    );
}

/// CONTRACT 4 (OVER-REJECTION): a zero-worker / zero-nov_min call must not
/// panic or overflow, and must return exactly the resident-b_ov-only term.
///
/// `n_workers` is clamped to at least 1 inside the helper (mirroring
/// `estimate_peak_bytes`'s own `n_workers.max(1)`), so passing 0 must not
/// silently return 0 -- that would be an under-charge in the opposite
/// direction of the original bug (a starved worker count must still charge
/// at least one worker's scratch, since rayon always runs with >=1 thread).
#[test]
fn zero_workers_is_clamped_to_one_not_zero() {
    let zero = open_shell_dynamic_extra_bytes(NAUX, NOV_MIN, 0);
    let one = open_shell_dynamic_extra_bytes(NAUX, NOV_MIN, 1);
    assert_eq!(
        zero, one,
        "n_workers=0 must be clamped to 1 (rayon never runs zero workers), not treated as \
         'charge nothing'"
    );
}

/// CONTRACT 5 (OVER-REJECTION): a symmetric-spin closed-shell-like call
/// (nov_min == 0, e.g. a spin channel with zero occupied orbitals) charges
/// only the per-worker naux² term, never a spurious b_ov charge.
///
/// This is the "ample budget must still run" direction: an all-alpha open-
/// shell system (nocc_b == 0) must not be charged as if a full second
/// (naux, nov) buffer existed when it structurally cannot (nov_b == 0 makes
/// `b_ov` for spin b a (naux, 0) no-op array).
#[test]
fn zero_nov_min_charges_only_the_second_naux_squared_buffer() {
    let workers = 4;
    let got = open_shell_dynamic_extra_bytes(NAUX, 0, workers);
    let expected = workers * NAUX * NAUX * F64_BYTES;
    assert_eq!(
        got, expected,
        "with nov_min=0 the only remaining extra term is the second naux² buffer per worker; \
         got {got}, expected {expected}"
    );
}
