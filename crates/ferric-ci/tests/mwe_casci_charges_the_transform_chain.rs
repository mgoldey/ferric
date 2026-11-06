//! MWE: the CAS-CI integral guard must charge the transform chain too.
//!
//! # The defect this guards against
//!
//! The gate once charged the `nbas^4` AO block alone, while the transform
//! allocated further large buffers after it. The first quarter transform reads
//! the AO block while it writes its `n_needed·nbas^3` output, so those two are
//! co-resident whatever the implementation.
//!
//! # What is charged now
//!
//! `transform_eri_quarter_gemm` consumes the AO block and drops each quarter's
//! input once its output exists, so the peak is the first pair:
//!
//! ```text
//!   ao_eri             nbas^4
//!   first quarter out  n_needed·nbas^3
//! ```
//!
//! Later pairs (`n·nbas^3 + n²·nbas²`, ...) are smaller when `n_needed <= nbas`.
//! `n_needed = active_start + n_active` INCLUDES the inactive block:
//!
//! ```text
//!   case                               nbas  n_needed   ao_eri      peak  ratio
//!   water/cc-pVDZ CAS(6,6) core 2        24         8  0.00 GB   0.00 GB  1.33x
//!   benzene/cc-pVDZ CAS(6,6) core 18    114        24  1.35 GB   1.64 GB  1.21x
//!   benzene, large core                 114        60  1.35 GB   2.06 GB  1.53x
//!   worst case n_needed = nbas          114       114  1.35 GB   2.70 GB  2.00x
//! ```

use ferric_ci::integrals::active_space_integral_elems;

fn ao_only(nbas: usize) -> usize {
    nbas.pow(4)
}

fn hand_computed_peak(nbas: usize, n_needed: usize) -> usize {
    nbas.pow(4) + n_needed * nbas.pow(3)
}

/// CONTRACT 1: the charge exceeds the AO block alone.
#[test]
fn the_charge_exceeds_the_ao_block() {
    let (nbas, n_needed) = (114usize, 24usize);
    assert!(
        active_space_integral_elems(nbas, n_needed) > ao_only(nbas),
        "the guard must charge more than ao_eri: the first quarter transform reads \
         ao_eri while it writes its n_needed·nbas^3 output."
    );
}

/// CONTRACT 2: the charge is exactly the co-resident first pair.
///
/// CONTRACT 1 would pass on any padding factor. Pin the actual sum, so a term
/// added or dropped later shows up here rather than as an OOM.
#[test]
fn the_charge_is_the_ao_block_plus_the_first_quarter() {
    for (nbas, n_needed) in [(24usize, 8usize), (41, 10), (114, 24), (114, 114)] {
        assert_eq!(
            active_space_integral_elems(nbas, n_needed),
            hand_computed_peak(nbas, n_needed),
            "nbas={nbas}, n_needed={n_needed}: the charge must be nbas^4 + n·nbas^3"
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

/// CONTRACT 4: the magnitude band. ~1.2x at a typical CAS, exactly 2x when
/// `n_needed == nbas`, so a future change to the chain cannot quietly move it.
#[test]
fn the_charge_is_between_one_and_two_times_the_ao_block() {
    let nbas = 114;
    let typical = active_space_integral_elems(nbas, 24) as f64 / ao_only(nbas) as f64;
    let worst = active_space_integral_elems(nbas, nbas) as f64 / ao_only(nbas) as f64;
    assert!(
        (1.15..1.25).contains(&typical),
        "typical CAS ratio drifted to {typical:.2}x (expected ~1.21x)"
    );
    assert!(
        (1.99..2.01).contains(&worst),
        "worst-case ratio drifted to {worst:.2}x (expected exactly 2x when n_needed == nbas)"
    );
}
