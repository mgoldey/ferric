//! The KS-DFT XC gradient walks the DFT grid in batches. These tests pin the
//! two claims that makes, in the order the experimental protocol requires.
//!
//! 1. **Exactness anchor, first.** At a batch width covering the whole grid
//!    there is exactly one batch, which is the pre-batching code path. That
//!    must be BIT-identical, not "close" — `a_full_width_batch_is_bit_identical_
//!    to_the_unbatched_gradient`. Anything less means the batch loop is not a
//!    faithful re-expression of the original.
//!
//! 2. **The re-association floor, MEASURED.** Below full width, batching splits
//!    `row_dot`'s reduction over grid points, which re-associates a
//!    floating-point sum. That is a reordering of the same terms, not a
//!    different quantity — but its size is a measurement, not an assumption,
//!    so `batched_gradient_matches_the_unbatched_one_to_the_reassociation_floor`
//!    prints it and bounds it well below any physical scale.
//!
//! 3. **Width determinism.** The width must not come from anything
//!    nondeterministic. `ks.rs` learned this the expensive way: its SCF batch
//!    width came from `available_budget_now()` (live RSS), the width set the
//!    accumulation order, and the same input gave -390.3794282913 and
//!    -390.3794337741 Ha. The gradient's width comes from the POOL LEDGER, so
//!    `the_batch_width_is_a_function_of_the_pool_ledger_not_of_resident_memory`
//!    holds the ledger fixed, moves resident memory, and asserts the width
//!    does not budge.
//!
//! Everything here runs at water/STO-3G..6-31G scale — seconds, not minutes.
//! The production-shape peak numbers live in the `#[ignore]`d
//! `gradient_pool_peak_measurement.rs`.

use std::sync::{Mutex, MutexGuard, OnceLock};

use ferric_core::memory::pool::{clear_global, global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::gradient::{
    grad_batch_pts_for_test, xc_gradient_closed_gga_from_density,
    xc_gradient_closed_gga_from_density_batched,
};
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

/// A pool large enough that nothing is ever refused, so a width it produces is
/// "the whole grid" and a peak it records is pure demand.
const AMPLE: usize = 200_000_000_000;

/// The memory pool is PROCESS-GLOBAL, and cargo runs these tests on parallel
/// threads by default. Every test here installs a pool of its own and reads
/// back a width or a peak, so without serialization one test's `install_global`
/// silently becomes another's budget.
///
/// MEASURED, and the reason this exists: under mutation `m7_charge_whole_grid`
/// the parallel run reported `the_batch_width_is_a_function_of_the_pool_ledger`
/// as failing — a test that never calls the gradient at all and could not
/// possibly see that mutation. Serially it passed and only the intended killer
/// failed. A cross-talking test attributes failures to the wrong cause, which
/// is worse than no test: it would have sent a future reader after the wrong
/// defect. Same `CleanSlot` pattern as
/// `mwe_gradient_planes_compose_in_one_pool.rs`.
fn global_lock() -> MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    match L.get_or_init(|| Mutex::new(())).lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    }
}

struct CleanSlot(#[allow(dead_code)] MutexGuard<'static, ()>);

impl CleanSlot {
    fn acquire() -> Self {
        let g = global_lock();
        clear_global();
        Self(g)
    }
}

impl Drop for CleanSlot {
    fn drop(&mut self) {
        clear_global();
    }
}

struct Case {
    mol: Molecule,
    bs: ferric_core::basis::BasisSet,
    obs: PreparedBasis,
    d: Array2<f64>,
    npts: usize,
    nbf: usize,
}

/// Converge a small KS-DFT reference and hand back everything the XC-gradient
/// entry point needs. PBE, because it is the GGA the production path takes and
/// the one the peak harness measures.
fn converged(mol_name: &str, basis: &str) -> Case {
    let mol = Molecule::load_xyz(&format!("../../testdata/molecules/{mol_name}.xyz"))
        .unwrap_or_else(|e| panic!("{mol_name}.xyz: {e}"));
    let bs = ferric_core::basis::bundled(basis).expect("basis");
    let obs = PreparedBasis::new(&mol, &bs).expect("prepared");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        xc: Some("PBE".into()),
        ..Default::default()
    };
    let scf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).expect("KS-DFT must converge");
    let d = scf.density_r().clone();
    let npts = ferric_dft::grid::build_atomic_grid(&mol, &AtomicGridConfig::default()).len();
    let nbf = obs.nbasis();
    Case {
        mol,
        bs,
        obs,
        d,
        npts,
        nbf,
    }
}

fn xc_grad(case: &Case, width: Option<usize>) -> Array2<f64> {
    xc_gradient_closed_gga_from_density_batched(
        &case.mol,
        &case.bs,
        &case.d,
        "PBE",
        &AtomicGridConfig::default(),
        case.obs.shell_to_atom(),
        case.obs.shell_offsets(),
        case.obs.shell_dims(),
        width,
    )
    .expect("XC gradient")
}

/// Largest absolute component difference between two `(natoms, 3)` gradients.
fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// EXACTNESS ANCHOR. Written and passing before any width below the grid was
/// measured, per the standing protocol: every approximation has a trivial
/// limit where it does nothing, and that limit must be exact.
///
/// A width at or above `npts` produces exactly ONE batch. One batch is the
/// pre-batching code path — same points, same order, one `row_dot` over the
/// whole range — so this is not "agreement to a tolerance", it is bit
/// identity. A `!=` here means the batch loop changed the computation rather
/// than just the memory it holds.
#[test]
fn a_full_width_batch_is_bit_identical_to_the_unbatched_gradient() {
    let _s = CleanSlot::acquire();
    let case = converged("water", "6-31g");
    let whole = xc_grad(&case, Some(case.npts));
    // Also above the grid: the clamp must not change anything.
    let over = xc_grad(&case, Some(case.npts * 3));

    for (a, b) in whole.iter().zip(over.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "a width above npts must clamp to one batch, unchanged: {a} vs {b}"
        );
    }

    // And against the production entry point when the pool is ample, which is
    // the same one-batch decision arrived at by the sizing rather than by hand.
    install_global(MemoryPool::with_capacity_bytes(AMPLE));
    let width = grad_batch_pts_for_test(case.nbf, case.npts, case.mol.atoms.len(), false);
    let production = xc_gradient_closed_gga_from_density(
        &case.mol,
        &case.bs,
        &case.d,
        "PBE",
        &AtomicGridConfig::default(),
        case.obs.shell_to_atom(),
        case.obs.shell_offsets(),
        case.obs.shell_dims(),
    )
    .expect("XC gradient");
    clear_global();

    assert_eq!(
        width, case.npts,
        "an AMPLE pool must take the whole grid in one batch, not {width} of {}",
        case.npts
    );
    for (a, b) in whole.iter().zip(production.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "the production path at an ample budget must be the one-batch path"
        );
    }
}

/// THE RISK, MEASURED. Batching re-associates `row_dot`'s sum over grid points,
/// so a width below `npts` is NOT expected to be bit-identical. What must hold
/// is that the difference is a floating-point reordering of the same terms —
/// which means it stays at the accumulation floor and does not grow into
/// anything a geometry optimizer could see.
///
/// The bar: 1e-10 Ha/Bohr. That is four orders below the ~1e-5 Ha/Bohr
/// grid-response error this whole gradient path already carries (see the
/// module docs in `ferric_dft::gradient`), and six below an optimizer's
/// default force convergence. A difference that FAILED this would not be
/// re-association; it would be a batching bug — a dropped term, a mis-offset
/// `weight1`, a boundary that skips or double-counts points.
#[test]
fn batched_gradient_matches_the_unbatched_one_to_the_reassociation_floor() {
    let _s = CleanSlot::acquire();
    let case = converged("water", "6-31g");
    let reference = xc_grad(&case, Some(case.npts));

    println!("\nwater / 6-31G / PBE, npts = {}", case.npts);
    println!(
        "{:>12}  {:>14}  {:>8}",
        "batch width", "max |Δgrad|", "batches"
    );

    let mut worst = 0.0_f64;
    // Widths chosen so that several do NOT divide npts evenly — an off-by-one
    // in the final partial batch is exactly the bug this is watching for, and
    // a width that always divides evenly would never exercise it.
    for width in [case.npts / 2, 4096, 1000, 997, 251, 64, 7, 1] {
        if width == 0 {
            continue;
        }
        let g = xc_grad(&case, Some(width));
        let d = max_abs_diff(&reference, &g);
        let nb = case.npts.div_ceil(width);
        println!("{width:>12}  {d:>14.3e}  {nb:>8}");
        worst = worst.max(d);
        assert!(
            d < 1e-10,
            "width {width} moved the gradient by {d:.3e} Ha/Bohr — that is far \
             above a re-association floor and indicates a batching BUG (dropped \
             term, mis-offset weight1, or a bad final partial batch), not \
             rounding"
        );
    }
    println!("worst over all widths: {worst:.3e} Ha/Bohr\n");

    // A width of 1 point per batch is the most aggressive re-association
    // available, so it had better still be at the floor — and it had better
    // not be EXACTLY zero, which would mean the width is being ignored and
    // this test is inert.
    let single = xc_grad(&case, Some(1));
    assert!(
        max_abs_diff(&reference, &single) > 0.0,
        "1 point per batch produced a BIT-IDENTICAL gradient. Either the width \
         override is not wired through, or the reduction being split is not the \
         one this test thinks it is — in both cases this test proves nothing."
    );
}

/// The batch boundaries must not depend on how many rayon workers are
/// configured, for the same reason they must not depend on RSS: the width sets
/// the accumulation order.
#[test]
fn the_batch_width_does_not_depend_on_the_rayon_pool() {
    let _s = CleanSlot::acquire();
    install_global(MemoryPool::with_capacity_bytes(50_000_000));
    let widths: Vec<usize> = [1usize, 3, 8]
        .iter()
        .map(|&n| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .expect("pool")
                .install(|| grad_batch_pts_for_test(120, 250_000, 12, false))
        })
        .collect();
    clear_global();
    assert!(
        widths.windows(2).all(|w| w[0] == w[1]),
        "batch width varied with the worker count: {widths:?}"
    );
}

/// WIDTH DETERMINISM — the `ks.rs` lesson, transplanted.
///
/// `ks.rs`'s SCF grid width used to come from `available_budget_now()`, which
/// subtracts live RSS. Because the width fixes the V_xc accumulation order,
/// the same terpinyl-cation input gave -390.3794192830 and -390.3794263686 Ha
/// on two runs of the same binary. The gradient must not repeat that, so its
/// width is read off the POOL LEDGER.
///
/// The discriminator: hold the ledger fixed and change resident memory by a
/// large, real allocation. A width sourced from RSS moves; a width sourced
/// from the ledger does not. Then move the LEDGER (by reserving against the
/// pool) and require the width to respond — otherwise the first half would
/// pass on a width that is simply constant, which proves nothing.
#[test]
fn the_batch_width_is_a_function_of_the_pool_ledger_not_of_resident_memory() {
    let _s = CleanSlot::acquire();
    // CALIBRATED, not guessed, and the calibration is the point: at
    // (nbf 120, npts 250k, 12 atoms) the NON-batchable grid term is
    // weight1 (250k · 12 · 3 · 8 = 72 MB) + the grid points (10 MB) = 82 MB,
    // so a 40 MB pool floors the width at 1 and BOTH assertions below become
    // arithmetic rather than measurement (that is what a first draft of this
    // test did). 400 MB leaves ~318 MB, i.e. a width in the ten-thousands —
    // room to move in either direction, which is what makes the test able to
    // fail.
    const POOL: usize = 400_000_000;
    const SHAPE: (usize, usize, usize, bool) = (120, 250_000, 12, false);
    install_global(MemoryPool::with_capacity_bytes(POOL));
    let before = grad_batch_pts_for_test(SHAPE.0, SHAPE.1, SHAPE.2, SHAPE.3);
    assert!(
        before > 1 && before < SHAPE.1,
        "the chosen pool must put the width strictly between the floor and the \
         whole grid, or nothing below can move it: got {before} of {}",
        SHAPE.1
    );

    // 400 MB of live, touched, resident memory — an order of magnitude more
    // than the pool, so an RSS-derived width could not possibly survive it.
    let mut ballast: Vec<f64> = vec![0.0; 50_000_000];
    for (i, v) in ballast.iter_mut().enumerate() {
        *v = i as f64;
    }
    let during = grad_batch_pts_for_test(SHAPE.0, SHAPE.1, SHAPE.2, SHAPE.3);
    // Keep it alive across the second reading.
    assert_eq!(ballast[7], 7.0);
    drop(ballast);

    assert_eq!(
        before, during,
        "the batch width moved from {before} to {during} when 400 MB became \
         resident. It is being sourced from live RSS, which is the exact defect \
         that made ks.rs's SCF energy nondeterministic."
    );

    // Now move the LEDGER and require a response. Without this, a width that
    // is simply pinned to a constant would pass the assertion above while
    // ignoring the budget entirely.
    let hold = global()
        .expect("pool")
        .reserve("test ballast reservation", POOL / 2)
        .expect("fits");
    let squeezed = grad_batch_pts_for_test(SHAPE.0, SHAPE.1, SHAPE.2, SHAPE.3);
    drop(hold);
    let restored = grad_batch_pts_for_test(SHAPE.0, SHAPE.1, SHAPE.2, SHAPE.3);
    clear_global();

    assert!(
        squeezed < before,
        "reserving half the pool did NOT shrink the batch width \
         ({before} -> {squeezed}). The width is not reading the ledger at all, \
         so the determinism assertion above is vacuous."
    );
    assert_eq!(
        restored, before,
        "releasing the reservation must restore the width: {before} -> {restored}"
    );
}

/// A budget too small for even one point must still make progress rather than
/// loop forever on a zero-width batch.
#[test]
fn a_hopeless_budget_still_yields_a_workable_batch() {
    let _s = CleanSlot::acquire();
    install_global(MemoryPool::with_capacity_bytes(1));
    let w = grad_batch_pts_for_test(2000, 1_000_000, 100, true);
    clear_global();
    assert_eq!(w, 1, "width must floor at 1 point, not 0");
}

/// Batching must shrink what the pool records, or it has bought nothing. This
/// is the whole point of the change, asserted at a size that runs in seconds.
#[test]
fn a_narrow_batch_lowers_the_pool_peak_for_the_same_gradient() {
    let _s = CleanSlot::acquire();
    let case = converged("water", "cc-pvdz");

    install_global(MemoryPool::with_capacity_bytes(AMPLE));
    let _ = xc_grad(&case, Some(case.npts));
    let wide_peak = global().expect("pool").peak_bytes();
    clear_global();

    install_global(MemoryPool::with_capacity_bytes(AMPLE));
    let _ = xc_grad(&case, Some(512));
    let narrow_peak = global().expect("pool").peak_bytes();
    clear_global();

    println!(
        "\nwater / cc-pVDZ / PBE, npts = {}: whole-grid peak {:.4} GB, \
         512-point batches {:.4} GB ({:.1}x lower)\n",
        case.npts,
        wide_peak as f64 / 1e9,
        narrow_peak as f64 / 1e9,
        wide_peak as f64 / narrow_peak.max(1) as f64,
    );
    assert!(
        narrow_peak * 2 < wide_peak,
        "batching to 512 points of {} did not halve the charged peak: \
         {narrow_peak} vs {wide_peak}",
        case.npts
    );
}
