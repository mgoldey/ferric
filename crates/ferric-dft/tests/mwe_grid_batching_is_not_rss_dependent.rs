//! The grid batch width -- and therefore the SCF energy -- must NOT depend on
//! the process's transient resident memory.
//!
//! # STATUS (2026-09, screened-batch XC): passes BY CONSTRUCTION
//!
//! `KsXc` now integrates the main grid in screened spatial batches
//! (`ferric_dft::xc_batch`) whose boundaries are a pure function of the grid
//! coordinates and a fixed 128-point cap; the budget only chooses whether the
//! per-batch AO blocks stay resident or are recomputed, and those two modes
//! are bit-identical (pinned in `ks.rs`'s `storage_tests`). So the ENERGY can
//! no longer depend on RSS through this path at all. `batch_pts_for_test`
//! now reports the storage mode (`None` resident, `Some(cap)` recompute); this
//! test still checks that the MODE is RSS-independent under a pool (the pool
//! ledger decides), which keeps memory behaviour reproducible, but it can no
//! longer catch an energy defect. The history below describes the retired
//! `resolve_batch_size` design and is kept as the record.
//!
//! # The defect
//!
//! `KsXc::new_with_omega_budgeted` sizes its working budget with
//! `available_budget_now(resolve_budget_bytes(..))`, which subtracts the
//! process's LIVE RSS (`read_own_rss_bytes`). That figure then feeds
//! `resolve_batch_size`, which sets the grid batch boundaries. Batch
//! boundaries fix the floating-point accumulation order of the V_xc/E_xc
//! sums, so they are load-bearing for the ENERGY -- `resolve_batch_size`'s
//! own doc says exactly this about thread counts:
//!
//! ```text
//! /// A pure function of `(nbf, npts, budget, is_uks, needs_tau)` only --
//! /// NEVER of thread count -- so batch boundaries ... are identical no
//! /// matter how many rayon workers are configured, which is what keeps the
//! /// batched-path energy thread-count-invariant.
//! ```
//!
//! The same argument applies with full force to RSS, and RSS is *worse* than
//! a thread count: it is not even a configuration, it drifts within a single
//! process depending on what ran before, whether the allocator returned pages
//! to the OS, and what else the job has already allocated.
//!
//! Measured on the 27-atom terpinyl cation (6-31G/PBE/RI-JK, budget_gb =
//! 0.30): two runs of the SAME binary on the SAME input gave
//!
//! ```text
//!   -390.3794192830 Ha  in 20 iterations
//!   -390.3794263686 Ha  in 35 iterations
//! ```
//!
//! a 7.1e-6 Ha spread with no input change. That is the same order as the
//! 2.9e-6 Ha benzene/aTZ thread-count perturbation this codebase already
//! treats as a bug (see `three_index_source.rs`'s `DRESS_ROW_BLOCK` note).
//!
//! # Artifact hypothesis (stated before measuring, per the protocol)
//!
//! If the defect is REAL, then holding every input fixed and varying ONLY the
//! process's resident memory changes the batch width. If instead the sizing
//! were already RSS-independent, the width would be constant and this test
//! could not distinguish the two -- so the test varies RSS and nothing else.

use ferric_core::basis;
use ferric_core::memory::pool::{clear_global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_dft::grid::AtomicGridConfig;
use ferric_dft::ks::KsXc;

/// CALIBRATED to the RSS-sensitive regime, not guessed.
///
/// Measured widths for water/cc-pVDZ/PBE with a pool installed at this
/// capacity, before vs after allocating ~320 MB of ballast:
///
/// ```text
///    30 MB -> Some(13032) vs Some(1)   DIFFER   <- the cliff
///    40 MB -> None        vs None      same
///    50 MB -> None        vs None      same
///   120 MB -> None        vs None      same
/// ```
///
/// Only 30 MB sits on the cliff. At every other budget probed, the width is
/// already stable and the test would PASS with the defect fully live -- i.e.
/// it would be INERT. Mutation M8 (revert the fix; read live RSS again even
/// when a pool is installed) is what proves this particular value is load
/// bearing: M8 survived at 50 MB and is killed here.
const TIGHT_BUDGET: usize = 30_000_000;

fn water() -> Molecule {
    Molecule::load_xyz("../../testdata/molecules/water.xyz").expect("water.xyz")
}

fn build(budget: usize) -> KsXc {
    let mol = water();
    let bs = basis::bundled("cc-pvdz").expect("cc-pvdz");
    let main = AtomicGridConfig::default();
    let nlc = AtomicGridConfig {
        n_radial: 50,
        n_angular: 50,
        ..Default::default()
    };
    KsXc::new_with_omega_budgeted(&mol, &bs, "PBE", &main, &nlc, None, Some(budget))
        .expect("KsXc must build")
}

/// The fix only applies where a pool is installed -- which is every CLI run,
/// because `ferric-cli` installs one from the resolved budget. With a pool,
/// "what is already spent" is the pool's own ledger: a deterministic function
/// of which planes this job reserved, not of allocator behaviour.
#[test]
fn the_grid_batch_width_does_not_move_when_resident_memory_does() {
    clear_global();
    install_global(MemoryPool::with_capacity_bytes(TIGHT_BUDGET));

    // Baseline: build with nothing extra resident.
    let a = build(TIGHT_BUDGET);
    let width_a = a.batch_pts_for_test();
    drop(a);

    // Now make the process HOLD a few hundred MB and build again with the
    // IDENTICAL budget and identical molecule/basis/functional. Only RSS
    // differs.
    let ballast: Vec<f64> = vec![1.0; 40_000_000]; // ~320 MB, measured above
    std::hint::black_box(&ballast);
    let b = build(TIGHT_BUDGET);
    let width_b = b.batch_pts_for_test();
    drop(b);
    drop(ballast);
    clear_global();

    assert_eq!(
        width_a, width_b,
        "the grid batch width moved from {width_a:?} to {width_b:?} purely because \
         the process was holding more memory. Batch boundaries fix the V_xc \
         summation order, so this makes the SCF ENERGY depend on transient RSS \
         -- measured as a 7.1e-6 Ha spread between two identical runs of the \
         27-atom terpinyl cation."
    );
}
