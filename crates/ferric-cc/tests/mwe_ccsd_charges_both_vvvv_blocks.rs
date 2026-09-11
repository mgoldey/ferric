//! MWE: the CCSD guard must charge BOTH co-resident `(2nv)^4` blocks.
//!
//! # The defect
//!
//! `ccsd.rs` gated on a flat `3*(2nv)^4 + eri3_ao`. Two things were wrong with
//! it, in opposite directions, and they partly cancelled.
//!
//! **The `3x` was an over-count.** It was meant for the VVVV build:
//!
//! ```text
//! let d: ArrayD<f64> = einsum!("Pab,Pcd->abcd", &b_vv, &b_vv);
//! let e: ArrayD<f64> = einsum!("Pab,Pcd->abcd", &b_vv, &b_vv);
//! asym_phys(&d, &e, nv, nv, nv, nv)
//! ```
//!
//! but `b_vv` is contracted with `c_vir`, which is SPATIAL — so `d` and `e`
//! are `nv^4` each, 16x smaller than `(2nv)^4`. Only the `asym_phys` output is
//! spin-orbital-sized. The honest build peak is `2*nv^4 + (2nv)^4`, not
//! `3*(2nv)^4`.
//!
//! **A whole resident block was missing.** `vvvv` is built once and lives for
//! the entire amplitude loop; `wabef` is a SECOND `(2nv)^4` buffer, rebuilt
//! every iteration by cloning `vvvv` and still live when the ladder term
//! `einsum!("ijef,abef->ijab", &tau_t, &wabef_t)` reads it. The sustained peak
//! is `2*(2nv)^4`, which the old formula never named.
//!
//! ```text
//!   benzene/aug-cc-pVTZ-scale, nv = 393 spatial -> nv2 = 786
//!     old charge         3*(2nv)^4          ~9160 GB
//!     honest build       2*nv^4 + (2nv)^4   ~3435 GB
//!     sustained peak     2*(2nv)^4          ~6107 GB   <- what actually must fit
//! ```
//!
//! The old number exceeded the true peak only by coincidence of the unrelated
//! build-stage bug. Correcting the build over-count WITHOUT adding the
//! iteration term would have turned a lucky over-estimate into a real
//! under-charge.
//!
//! # Why this tests the plan and not a CCSD run
//!
//! The terms above dominate only where `(2nv)^4` dwarfs `naux*nbas^2`, i.e. at
//! shapes far too large to execute in a test. At any runnable fixture
//! (H2/STO-3G has `(2nv)^4 = 16` elements) the `eri3_ao` term sets the
//! threshold and the VVVV terms are inert — an end-to-end budget test there
//! passes whether or not VVVV is charged at all, which is precisely the trap
//! this repo keeps hitting. So the arithmetic is exercised directly through
//! `ccsd_memory_plan`, at a realistic shape.

use ferric_cc::ccsd::{ccsd_memory_plan, CcsdShape};

/// benzene/aug-cc-pVTZ-scale. `(2nv)^4` dominates every other term here.
const NO: usize = 21;
const NV: usize = 393;
const NBAS: usize = 414;
const NAUX: usize = 1512;
const F64: usize = 8;

fn shape(diis_subspace: usize) -> CcsdShape {
    CcsdShape { no: NO, nv: NV, nbas: NBAS, naux: NAUX, diis_subspace }
}

/// Smallest budget the plan accepts, found by bisection.
///
/// `MemoryPlan` exposes a pass/fail `check()`, so bisect rather than reading a
/// private byte count — this tests the decision the plan actually makes.
fn smallest_accepted_budget(diis_subspace: usize) -> usize {
    let (mut lo, mut hi) = (0usize, usize::MAX / 4);
    assert!(
        ccsd_memory_plan(shape(diis_subspace), Some(hi)).check().is_ok(),
        "fixture is broken: even an enormous budget was refused"
    );
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if ccsd_memory_plan(shape(diis_subspace), Some(mid)).check().is_ok() {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

fn one_vvvv_bytes() -> usize {
    (2 * NV).pow(4) * F64
}

/// CONTRACT 1: the charge covers TWO `(2nv)^4` blocks, not one.
///
/// The defect. `vvvv` and `wabef` are independently allocated and co-resident,
/// so a plan that charges either one alone under-counts by ~3000 GB at this
/// shape. Bisecting for the accept threshold means a dropped reservation moves
/// the number, rather than being absorbed by a re-derivation of the same sum.
#[test]
fn the_charge_covers_both_co_resident_vvvv_blocks() {
    let threshold = smallest_accepted_budget(6);
    let two_blocks = 2 * one_vvvv_bytes();
    assert!(
        threshold >= two_blocks,
        "the plan accepts {threshold} bytes, but `vvvv` and `wabef` are both (2nv)^4 and live \
         at the same time, so at least {two_blocks} bytes must be required. Charging one block \
         (or dropping either reservation) is the defect."
    );
}

/// CONTRACT 2 (the over-rejection guard): it does NOT charge the old `3x`.
///
/// "An over-estimating guard is also a bug." The build blocks `d`/`e` are
/// spatial `nv^4`, so the honest total is close to `2*(2nv)^4` plus small
/// terms — comfortably under the old `3*(2nv)^4`. A plan that still demanded
/// the old figure would refuse jobs that fit in ~2/3 the memory.
#[test]
fn the_charge_does_not_keep_the_old_three_times_over_count() {
    let threshold = smallest_accepted_budget(6);
    let old_charge = 3 * one_vvvv_bytes();
    assert!(
        threshold < old_charge,
        "the plan still demands {threshold} bytes, at or above the old 3x(2nv)^4 figure \
         ({old_charge}). The build-stage blocks d/e are SPATIAL nv^4 (b_vv is contracted with \
         c_vir), so charging them at spin-orbital size over-rejects by ~2.67x."
    );
}

/// CONTRACT 3: the build blocks are charged at spatial size.
///
/// Pins the direction CONTRACT 2 bounds. Two `nv^4` blocks are 1/8 of one
/// `(2nv)^4` block, so the total must sit between two and three blocks — above
/// the two resident ones, below the old over-count. A plan that charged `d`/`e`
/// at spin-orbital size would land at or past 4x.
#[test]
fn the_total_sits_between_two_and_three_blocks() {
    let threshold = smallest_accepted_budget(6) as f64 / one_vvvv_bytes() as f64;
    assert!(
        (2.0..3.0).contains(&threshold),
        "the charge is {threshold:.3}x one (2nv)^4 block; expected just over 2x — the two \
         resident spin-orbital blocks plus small spatial/DIIS terms"
    );
}

/// CONTRACT 4: the DIIS history is charged and scales with the subspace.
///
/// `Diis::step` clones both the amplitude and the error vector into separate
/// ring histories, so the subspace depth is a real memory knob. A plan flat in
/// `diis_subspace` would under-charge every default run.
#[test]
fn the_diis_history_scales_with_the_subspace() {
    let shallow = smallest_accepted_budget(1);
    let deep = smallest_accepted_budget(12);
    assert!(
        deep > shallow,
        "a deeper DIIS subspace holds more amplitude history and must cost more: \
         subspace 1 -> {shallow}, subspace 12 -> {deep}"
    );
}

/// CONTRACT 5 (the reachability guard): at this shape VVVV really is the
/// dominant term.
///
/// Without this, CONTRACTS 1-3 could be pinning a threshold set by `eri3_ao`
/// while the VVVV reservations contributed nothing — the exact inertness that
/// made the first version of this guard pass with BOTH VVVV reservations
/// deleted. Assert the fixture puts VVVV in charge.
#[test]
fn vvvv_dominates_the_other_terms_at_this_shape() {
    let eri3 = NAUX * NBAS * NBAS * F64;
    let build = 2 * NV.pow(4) * F64;
    // Measured at this shape: 3053 GB for one VVVV block vs 384 GB for the
    // other two together, i.e. ~8x. The bar is set below that so the contract
    // fails if the fixture drifts toward a regime where VVVV stops dominating,
    // without pretending to a ratio the shape does not have.
    assert!(
        one_vvvv_bytes() > 5 * (eri3 + build),
        "fixture no longer isolates VVVV: one block is {} bytes vs {} for eri3_ao + the \
         spatial build blocks. At a shape where VVVV does not dominate, these contracts \
         would pass without testing it.",
        one_vvvv_bytes(),
        eri3 + build
    );
}
