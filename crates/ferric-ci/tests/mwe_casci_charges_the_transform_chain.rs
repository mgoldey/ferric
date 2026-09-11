//! MWE: the CAS-CI integral guard must charge the transform chain too.
//!
//! # The defect
//!
//! `integrals.rs:111` gates on the AO block alone:
//!
//! ```text
//! check_alloc(
//!     &format!("CAS-CI dense AO ERIs (nbas={nbas}; nbas^4 = {n4} f64)"),
//!     n4 * 8,
//!     resolve_budget_bytes(memory_budget_bytes),
//! )?;
//! let mut ao_eri = vec![0.0f64; n4];              // :116
//! ```
//!
//! and then allocates four more large buffers with no further check:
//!
//! ```text
//! let mut mo_eri = vec![0.0f64; n_needed^4];       // :156
//! let mut t1     = vec![0.0f64; n_needed * nbas^3]; // :162
//! let mut t2     = vec![0.0f64; n_needed^2 * nbas^2]; // :184
//! let mut t3     = vec![0.0f64; n_needed^3 * nbas]; // :206
//! ```
//!
//! `grep -n 'drop(' crates/ferric-ci/src/integrals.rs` returns **nothing**, so
//! all five live to the end of `build_active_space_integrals`. The
//! co-residency is forced by the code, not incidental: the `t1` loop reads
//! `ao_eri`, and the `t2` loop reads `t1`.
//!
//! # How bad, honestly
//!
//! Less than the raw buffer count suggests — `ao_eri` is `nbas^4` and the
//! transforms carry `n_needed` factors, so the ratio depends on how much of
//! the basis the active space plus inactive core covers. `n_needed =
//! active_start + n_active` INCLUDES the inactive block, so it is not small:
//!
//! ```text
//!   case                               nbas  n_needed   ao_eri  true peak  ratio
//!   water/cc-pVDZ CAS(6,6) core 2        24         8  0.00 GB    0.00 GB  1.49x
//!   benzene/cc-pVDZ CAS(6,6) core 18    114        24  1.35 GB    1.71 GB  1.27x
//!   benzene, large core                 114        60  1.35 GB    2.74 GB  2.03x
//!   worst case n_needed = nbas          114       114  1.35 GB    6.76 GB  5.00x
//! ```
//!
//! So 1.3x typically and up to 5x — a real under-count, not the order of
//! magnitude the raw "five buffers" framing implies. Recorded at the measured
//! size rather than the alarming one.

use ferric_ci::integrals::active_space_integral_elems;

const F64: usize = 8;

fn ao_only(nbas: usize) -> usize {
    nbas.pow(4)
}

fn hand_computed_peak(nbas: usize, n_needed: usize) -> usize {
    nbas.pow(4)
        + n_needed.pow(4)
        + n_needed * nbas.pow(3)
        + n_needed.pow(2) * nbas.pow(2)
        + n_needed.pow(3) * nbas
}

/// CONTRACT 1: the charge exceeds the AO block alone.
///
/// The defect itself. Before the fix the gate saw only `nbas^4`.
#[test]
fn the_charge_exceeds_the_ao_block() {
    let (nbas, n_needed) = (114usize, 24usize);
    assert!(
        active_space_integral_elems(nbas, n_needed) > ao_only(nbas),
        "the guard must charge more than ao_eri: mo_eri, t1, t2 and t3 are all allocated \
         after it and never dropped (no `drop(` appears in integrals.rs), and the t1 loop \
         reads ao_eri so co-residency is forced."
    );
}

/// CONTRACT 2: the charge is exactly the five co-resident buffers.
///
/// CONTRACT 1 would pass on any padding factor. Pin the actual sum, so a term
/// added or dropped later shows up here rather than as an OOM.
#[test]
fn the_charge_is_the_sum_of_all_five_buffers() {
    for (nbas, n_needed) in [(24usize, 8usize), (41, 10), (114, 24), (114, 114)] {
        assert_eq!(
            active_space_integral_elems(nbas, n_needed),
            hand_computed_peak(nbas, n_needed),
            "nbas={nbas}, n_needed={n_needed}: the charge must be \
             nbas^4 + n^4 + n·nbas^3 + n²·nbas² + n³·nbas"
        );
    }
}

/// CONTRACT 3: it grows with the active space, not just the basis.
///
/// `n_needed = active_start + n_active`, so a bigger active space (or a bigger
/// inactive core) costs more at fixed basis. A charge flat in `n_needed` would
/// be the old defect wearing a larger constant.
#[test]
fn the_charge_grows_with_the_active_space() {
    let nbas = 114;
    let small = active_space_integral_elems(nbas, 12);
    let large = active_space_integral_elems(nbas, 60);
    assert!(
        large > small,
        "a larger n_needed must cost more at fixed nbas: {small} vs {large}"
    );
}

/// CONTRACT 4 (the honest-magnitude guard): the under-count is real but bounded.
///
/// Both directions matter here. If the transform chain were negligible the fix
/// would be noise; if this test claimed a 10x under-count it would overstate
/// the finding. At the worst case (`n_needed == nbas`) the true peak is exactly
/// 5x the AO block, and at a typical CAS it is ~1.3x — assert that band so a
/// future change to the chain cannot quietly move it.
#[test]
fn the_under_count_is_between_one_and_five_times() {
    let nbas = 114;
    let typical = active_space_integral_elems(nbas, 24) as f64 / ao_only(nbas) as f64;
    let worst = active_space_integral_elems(nbas, nbas) as f64 / ao_only(nbas) as f64;
    assert!(
        (1.2..1.4).contains(&typical),
        "typical CAS ratio drifted to {typical:.2}x (expected ~1.27x)"
    );
    assert!(
        (4.9..5.1).contains(&worst),
        "worst-case ratio drifted to {worst:.2}x (expected exactly 5x when n_needed == nbas)"
    );
    let _ = F64;
}
