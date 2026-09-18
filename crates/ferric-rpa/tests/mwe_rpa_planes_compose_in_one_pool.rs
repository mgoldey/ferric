//! The property this migration exists to create: two ferric-rpa planes that
//! each fit the ceiling individually must NOT both be admitted when their SUM
//! does not.
//!
//! # The measured defect (from the shared pool brief)
//!
//! A 27-atom def2-SVP B3LYP RI-JK single point printed `memory budget: 4.72
//! GiB` and died at `MAXRSS 6.04 GiB` to a global OOM kill with NOT ONE GATE
//! HAVING FAILED. Every gate re-read the same ceiling instead of debiting a
//! shared ledger. ferric-rpa had the same shape at five preflight gates plus
//! two `MemoryPlan` sites: `preflight_check_closed_shell`, the RS-MP2-RPA
//! driver gate, `preflight_grid_path`, `preflight_hirshfeld_path`,
//! `preflight_molecular_path`, `preflight_hirshfeld_grid_scan`, and the two
//! `ao_rpa.rs` gates -- all of them `resolve_budget_bytes(..)` against the
//! FULL ceiling, so an RPA energy run and a Becke property scan in the same
//! process each saw 100% of the budget.
//!
//! # Calibration (measured on this tree, not guessed)
//!
//! water/cc-pVDZ + cc-pvdz-ri, `OPENBLAS_NUM_THREADS=1`:
//!
//! | path | pool peak | outstanding after |
//! |---|---|---|
//! | `run_pdep_rpa` (n_quad=8) | 2_353_344 B (0.00235 GB) | 0 |
//! | `pdep_polarizability_becke` | 44_112_624 B (0.04411 GB) | 0 |
//!
//! Of the RPA peak, 522_816 B is the HARD in-core half and 1_443_456 B (61%)
//! is the SOFT per-worker frequency scratch at 12 rayon workers. The
//! remaining 387_072 B is the `(naux, nao, nao)` AO tensor, which
//! `ferric_integrals::ThreeIndexSource` charges itself under
//! `"DF 3-index (P|mn) in-core"` -- this crate deliberately does NOT
//! double-charge it (see `budget::ao_tensor_bytes`), which is why the peak
//! here is 2_353_344 and not 2_740_416.
//!
//! Those are the numbers the capacities below are chosen against. A capacity
//! that admits either alone but not both is the whole experiment.
//!
//! # Artifact hypothesis (stated before measuring)
//!
//! If composition is REAL: holding a reservation that leaves less headroom
//! than the RPA run needs makes `run_pdep_rpa` FAIL with the pool's
//! "memory pool exhausted" message naming the incumbent, while the same run
//! against the same capacity with nothing held SUCCEEDS.
//!
//! If the gate is DECORATION (the guard dropped at the end of the preflight,
//! or the pool advisory): both runs succeed, because the ledger never held
//! anything at the moment the second asker looked. That prediction is
//! distinguishable, which is what makes this test able to kill mutations M1
//! and M2.
//!
//! The pool is process-global, so these tests serialize and clear the slot.

use std::sync::{Mutex, MutexGuard, OnceLock};

use ferric_core::basis;
use ferric_core::memory::pool::{clear_global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{QuadratureConfig, QuadratureScheme};
use ferric_rpa::{run_pdep_rpa, PdepRpaConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;

/// Measured pool peak of one `run_pdep_rpa` on the fixture below, bytes.
/// Recorded, not assumed: see the table in the module doc.
///
/// # Worker-count portability (fixed 2026-09-18)
///
/// The SOFT half of this peak is PER-WORKER frequency scratch, so the total
/// is a function of `rayon::current_num_threads()`. These constants were
/// first frozen at the 12 workers of the box they were measured on, and
/// 3 of this file's 7 tests FAILED at `RAYON_NUM_THREADS=4` -- which is a
/// typical CI runner, so they would have gone red immediately.
///
/// The fix is to derive the worker-dependent part rather than freeze it.
/// `MEASURED_RPA_SOFT_PER_WORKER` is the per-worker figure (the measured
/// 1_443_456 B at 12 workers / 12); `rpa_soft()` and `rpa_peak()` scale it.
/// The HARD half and the integrals-owned AO tensor are worker-INDEPENDENT
/// and stay frozen, which is what keeps the decomposition exact enough to
/// have killed mutation M1.
///
/// NOTE this is a property of the TEST's arithmetic, not of the energy: the
/// three bit-identity anchors in `mwe_rpa_pool_is_inert_without_a_pool.rs`
/// pass unchanged at 2, 4 and 12 workers. Per-worker scratch is a real
/// allocation that genuinely scales; that is different from a budget-derived
/// PANEL WIDTH reading the thread count, which would move accumulation order
/// and therefore the energy, and which this crate does not do.
fn rpa_peak() -> usize {
    MEASURED_RPA_HARD + rpa_soft() + MEASURED_RPA_AO_TENSOR
}

/// The SOFT per-worker frequency scratch at the live worker count.
fn rpa_soft() -> usize {
    MEASURED_RPA_SOFT_PER_WORKER * rayon::current_num_threads().max(1)
}
/// The HARD half of that peak: the in-core planes with no streaming variant,
/// EXCLUDING the AO tensor that ferric-integrals charges for itself.
/// Measured 522_816 B at this shape (total 2_353_344 B, of which 1_443_456 B
/// -- 61% -- is the SOFT per-worker frequency scratch at 12 rayon workers,
/// and 387_072 B is the integrals-owned AO tensor). The split is what makes
/// the two halves testable separately: a pool squeezed below the hard figure
/// must REFUSE, and a pool squeezed below the total but above it must
/// COMPLETE (panelled).
const MEASURED_RPA_HARD: usize = 522_816;
/// The SOFT half, PER WORKER: `(m, nov) + (m, m)` frequency scratch.
/// Measured 1_443_456 B total at 12 rayon workers => 120_288 B per worker.
const MEASURED_RPA_SOFT_PER_WORKER: usize = 1_443_456 / 12;
/// The `(naux, nao, nao)` = 84 x 24 x 24 x 8 B AO tensor that
/// `ferric_integrals::ThreeIndexSource` charges under its OWN guard. Recorded
/// here because the peak decomposition below must account for it: it is
/// co-resident with this crate's two charges, and it is deliberately NOT
/// charged twice (see `budget::ao_tensor_bytes`).
const MEASURED_RPA_AO_TENSOR: usize = 387_072;
/// Measured pool peak of one `pdep_polarizability_becke` on the same fixture.
/// Worker-dependent like the RPA peak: 44_112_624 B at 12 rayon workers,
/// 43_111_920 B at 4. Used where an absolute size is needed (capacities,
/// incumbents); the peak ASSERTION uses the band below instead.
const MEASURED_BECKE_PEAK: usize = 44_112_624;
/// Lower edge of the measured Becke band (4 workers) with 5% margin.
const MEASURED_BECKE_PEAK_FLOOR: usize = 40_000_000;
/// Upper edge (12 workers) with margin. A peak above this is real drift.
const MEASURED_BECKE_PEAK_CEIL: usize = 48_000_000;

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

struct Fixture {
    mol: Molecule,
    obs: PreparedBasis,
    obs_bs: ferric_core::basis::BasisSet,
    dfbs: PreparedBasis,
    op: Operator,
    rhf: ScfResult,
}

fn fixture() -> Fixture {
    let ctx = ParallelContext::default();
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").expect("water.xyz");
    let obs_bs = basis::bundled("cc-pvdz").expect("cc-pvdz");
    let dfbs_bs = basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri");
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(&mol, &obs_bs).expect("obs");
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).expect("dfbs");
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).expect("rhf");
    Fixture {
        mol,
        obs,
        obs_bs,
        dfbs,
        op,
        rhf,
    }
}

fn rpa_cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 8,
            u0: 0.5,
        },
        frozen_core: 0,
        trunc_thresh: 0.0,
        eigensolver_conv_thresh: 1e-10,
        ..Default::default()
    }
}

fn run_rpa(f: &Fixture) -> Result<f64, ferric_core::FerricError> {
    run_pdep_rpa(&f.mol, &f.obs, &f.dfbs, f.op, &f.rhf, &rpa_cfg()).map(|r| r.e_rpa)
}

fn run_becke(f: &Fixture) -> Result<Vec<[[f64; 3]; 3]>, ferric_core::FerricError> {
    ferric_rpa::properties::pdep_polarizability_becke(
        &f.mol,
        &f.obs,
        &f.obs_bs,
        &f.dfbs,
        &f.rhf,
        f.op,
        &PdepRpaConfig::default(),
    )
}

/// REACHABILITY CHECK, run first: the pass condition of every test below
/// depends on the RPA run actually charging the pool a nonzero, roughly-known
/// amount. A gate that charges 0 makes every "does not fit" assertion below
/// vacuous (it would fail for a different reason, or not at all). Per
/// `CLAUDE.md` ("a gate whose GO conditions are mutually exclusive returns
/// arithmetic, not measurement"), pin the measurement itself.
#[test]
fn the_rpa_path_charges_the_pool_a_nonzero_measured_amount() {
    let _s = CleanSlot::acquire();
    let f = fixture();
    let pool = MemoryPool::with_capacity_bytes(64_000_000_000);
    install_global(pool.clone());
    let e = run_rpa(&f).expect("ample pool must admit");
    clear_global();

    assert!(
        e < 0.0,
        "sanity: RPA correlation energy is negative, got {e}"
    );
    let peak = pool.peak_bytes();
    assert!(
        peak > 0,
        "the RPA path must DEBIT the pool -- a peak of 0 means every gate here \
         is decoration and every composition assertion below is vacuous"
    );
    // EXACT, not a band. The measured peak decomposes exactly:
    //
    //   522_816 (hard in-core, this crate)
    // + 1_443_456 (soft frequency scratch, this crate)
    // +   387_072 (AO tensor, ferric-integrals' own `_charge`)
    // = 2_353_344  == rpa_peak()
    //
    // That equality is the load-bearing claim: all three are outstanding AT
    // THE SAME MOMENT, which is what "the planes compose" means. A band of
    // +/-4x cannot see it -- and did not: with the hard reservation dropped
    // immediately after being taken (mutation M1), the run still refuses
    // under a squeezed pool (the transient debit is enough), every
    // composition test still passes, and only the PEAK moves. So the peak
    // decomposition is the assertion that makes the lifetime testable at all.
    assert_eq!(
        peak,
        MEASURED_RPA_HARD + rpa_soft() + MEASURED_RPA_AO_TENSOR,
        "the peak must be the SUM of the three co-resident charges. Got \
         {peak} B, expected {} (hard) + {} (soft) + {} (integrals AO tensor). \
         A peak SHORT of the sum means a guard was released while its buffer \
         was still live -- the charge is decoration. A peak ABOVE it means \
         something is double-charged.",
        MEASURED_RPA_HARD,
        rpa_soft(),
        MEASURED_RPA_AO_TENSOR
    );
    assert_eq!(
        peak,
        rpa_peak(),
        "and the sum must equal the separately recorded whole-run peak"
    );
    assert_eq!(
        pool.outstanding_bytes(),
        0,
        "every guard must release when the run returns -- a nonzero residue \
         means a charge outlived its buffer and an 80-iteration driver would \
         exhaust the pool"
    );
}

/// THE LIFETIME CONTRACT of `preflight_grid_path`, asserted DIRECTLY.
///
/// # Why this is not an end-to-end test
///
/// On `pdep_polarizability_becke` the RI intermediates are built BEFORE this
/// gate runs, so nothing asks the pool again while the guard is held and
/// dropping it early leaves the high-water mark unchanged. Measured: mutation
/// M4b (drop the guard inside the preflight instead of returning it) left ALL
/// SIX end-to-end pool tests in this file green. A contract no test can see is
/// an assumption, so assert it where it is observable: hold the returned
/// guard, ask for what is left, and require the refusal.
///
/// This is the test that kills M4b. Mutation-verified.
#[test]
fn the_grid_preflight_guard_keeps_its_bytes_debited_until_the_caller_drops_it() {
    let _s = CleanSlot::acquire();
    // Shapes big enough that the estimate is a meaningful fraction of the pool
    // but small enough to be arithmetic, not a calculation.
    const CAP: usize = 200_000_000;
    let pool = MemoryPool::with_capacity_bytes(CAP);
    install_global(pool.clone());

    let (_budget, guard) = ferric_rpa::properties::preflight_grid_path_for_test(
        "lifetime probe",
        Some(CAP),
        84,     // naux
        5,      // nocc
        19,     // nvir
        24_750, // npts
        24,     // nbf
        3,      // natoms
    )
    .expect("the gate must admit at this capacity");

    let debited = pool.outstanding_bytes();
    assert!(
        debited > 0,
        "REACHABILITY: the gate must debit something while the guard is held, \
         or the refusal below would prove nothing"
    );

    // While the guard is HELD, the rest of the pool must be gone.
    let leftover = CAP - debited;
    let second = pool.reserve("a second plane asking during the grid work", leftover + 1);
    assert!(
        second.is_err(),
        "with the grid guard held ({debited} B debited of {CAP} B), a request \
         for more than the {leftover} B remaining must be REFUSED. If it \
         succeeds, the guard is not holding its bytes and a second subsystem \
         is being admitted against memory this one is spending -- the exact \
         double-spend that OOM-killed the 27-atom def2-SVP case."
    );
    // ...and something that DOES fit in the leftover must still be admitted,
    // so the refusal above is a ledger effect and not a blanket denial.
    let fits = pool
        .reserve("a small second plane", leftover / 2)
        .expect("what genuinely fits must still be admitted");
    drop(fits);

    // Dropping the guard returns the bytes.
    drop(guard);
    assert_eq!(
        pool.outstanding_bytes(),
        0,
        "the guard must credit its bytes back on drop"
    );
    assert!(
        pool.reserve("after release", CAP).is_ok(),
        "the whole pool must be available again once the caller drops the guard"
    );
    clear_global();
}

/// LIFETIME OF THE GRID-PROPERTY CHARGE, by exact peak decomposition.
///
/// `preflight_grid_path` returns its `Reservation` to the caller precisely so
/// the caller can hold it across the grid accumulation. That contract is only
/// testable through the PEAK: a guard released at the end of the preflight
/// (mutation M4b) still refuses an over-budget job -- the transient debit is
/// enough for that -- so every refusal-shaped assertion passes with the
/// lifetime broken. Measured: with the grid charge dropped early, all five of
/// the other tests in this file stayed green.
///
/// What DOES move is the high-water mark. Hold a known incumbent across the
/// run: if the grid charge is live at the same moment, the peak is exactly
/// `incumbent + grid charge`. If it was released early, the peak is whatever
/// the grid run reaches on its own and the sum does not hold.
#[test]
fn the_grid_property_charge_is_live_while_the_grid_work_runs() {
    let _s = CleanSlot::acquire();
    let f = fixture();

    let pool = MemoryPool::with_capacity_bytes(64_000_000_000);
    install_global(pool.clone());
    // Baseline: the run alone.
    run_becke(&f).expect("ample pool");
    let alone = pool.peak_bytes();
    clear_global();

    // The Becke peak carries a per-worker term too (44_112_624 B at 12
    // workers, 43_111_920 B at 4), so it is bounded rather than pinned to one
    // box's arithmetic. The load-bearing claim of this test is COMPOSITION --
    // that a held incumbent raises the peak by exactly its own size -- which
    // is asserted below and is worker-independent.
    assert!(
        (MEASURED_BECKE_PEAK_FLOOR..=MEASURED_BECKE_PEAK_CEIL).contains(&alone),
        "the Becke path's pool peak ({alone} B) left the measured band \
         [{MEASURED_BECKE_PEAK_FLOOR}, {MEASURED_BECKE_PEAK_CEIL}] B \
         (43_111_920 at 4 workers .. 44_112_624 at 12, plus margin). A value \
         OUTSIDE this band is a real drift in what the grid path allocates, \
         not a worker-count difference."
    );

    // Same run, with a KNOWN incumbent held across it.
    const INCUMBENT: usize = 1_000_000;
    let pool2 = MemoryPool::with_capacity_bytes(64_000_000_000);
    install_global(pool2.clone());
    let held = pool2
        .reserve("incumbent held across the grid run", INCUMBENT)
        .expect("fits");
    run_becke(&f).expect("ample pool");
    let with_incumbent = pool2.peak_bytes();
    drop(held);
    clear_global();

    assert_eq!(
        with_incumbent,
        alone + INCUMBENT,
        "the grid charge and the incumbent must be outstanding AT THE SAME \
         MOMENT: peak with a {INCUMBENT} B incumbent held was {with_incumbent} \
         B, expected {} B. A peak short of the sum means the grid charge was \
         credited back before the grid work ran -- the guard is decoration and \
         a second subsystem could be admitted against bytes this one is \
         spending.",
        alone + INCUMBENT
    );
    assert_eq!(pool2.outstanding_bytes(), 0);
}

/// THE COMPOSITION PROPERTY. A pool sized so that ONE RPA run fits with room
/// to spare, but a pre-held incumbent leaves less than the run needs, must
/// REFUSE the run -- and must name the incumbent in the breakdown.
///
/// Pre-migration this could not fail: the preflight resolved the whole
/// ceiling, so the incumbent was invisible to it.
#[test]
fn a_held_incumbent_makes_the_rpa_path_refuse() {
    let _s = CleanSlot::acquire();
    let f = fixture();

    // Capacity comfortably above one RPA run.
    let capacity = rpa_peak() * 4;
    let pool = MemoryPool::with_capacity_bytes(capacity);
    install_global(pool.clone());

    // Baseline: nothing held, the run must SUCCEED at this capacity. Without
    // this the refusal below could be "the capacity was always too small",
    // which measures nothing.
    let baseline = run_rpa(&f).expect("with nothing held, this capacity must admit the run");
    assert!(baseline < 0.0);
    assert_eq!(pool.outstanding_bytes(), 0);

    // Hold an incumbent leaving strictly LESS than the HARD half -- the part
    // with no fallback. Leaving less than the TOTAL is not enough to force a
    // refusal, and learning that was the point of splitting hard from soft:
    // at 12 workers the soft frequency scratch is 61% of the estimate, so a
    // pool at half the total still completes (panelled). Only the hard floor
    // refuses.
    let held = pool
        .reserve(
            "DF 3-index tensor (incumbent)",
            capacity - MEASURED_RPA_HARD / 2,
        )
        .expect("incumbent fits");
    let err = run_rpa(&f).expect_err(
        "with an incumbent leaving less than HALF the run's hard in-core need, \
         the RPA preflight MUST refuse -- if it succeeds, the gate is re-reading \
         the ceiling, which is exactly the defect that OOM-killed the 27-atom \
         def2-SVP case",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("memory pool exhausted"),
        "the refusal must come from the POOL (a debited ledger), not from a \
         re-read ceiling: {msg}"
    );
    assert!(
        msg.contains("DF 3-index tensor (incumbent)"),
        "the occupancy breakdown must name who is holding the bytes: {msg}"
    );
    // ATTRIBUTION. Without this, the test passes when THIS crate's gate is
    // advisory (mutation M2) and the refusal comes from a DOWNSTREAM gate
    // in ferric-integrals instead. Measured: with M2 applied the message
    // named `"DF 3-index (P|mn) in-core"` and every assertion above still
    // held, so the test was scoring another crate's gate as this one's.
    //
    // The refusing plane must be THIS preflight's label.
    assert!(
        msg.contains("PDEP-RPA preflight") && msg.contains("[in-core]"),
        "the REFUSING plane must be ferric-rpa's own preflight gate, not a \
         downstream one -- otherwise this crate's gate could be purely \
         advisory and this test would not notice: {msg}"
    );

    // Releasing the incumbent must make the same run fit again, bit-identically.
    drop(held);
    let after = run_rpa(&f).expect("releasing the incumbent must restore headroom");
    assert_eq!(
        baseline.to_bits(),
        after.to_bits(),
        "a refused-then-retried run must give the SAME energy to the bit"
    );
    clear_global();
}

/// The SAME property across two DIFFERENT subsystems in one process: the RPA
/// energy path and the Becke grid property path. This is the composition the
/// brief describes (two gates, one ceiling, both pass, process holds the sum).
#[test]
fn the_grid_property_plane_and_the_rpa_plane_share_one_ledger() {
    let _s = CleanSlot::acquire();
    let f = fixture();

    // Sized so the Becke path fits alone, and the RPA path fits alone, but a
    // held Becke-sized incumbent does not leave room for the RPA path.
    let capacity = MEASURED_BECKE_PEAK * 2;
    let pool = MemoryPool::with_capacity_bytes(capacity);
    install_global(pool.clone());

    // Each alone fits.
    run_becke(&f).expect("Becke path alone must fit");
    assert_eq!(pool.outstanding_bytes(), 0);
    run_rpa(&f).expect("RPA path alone must fit");
    assert_eq!(pool.outstanding_bytes(), 0);

    // A Becke-sized incumbent leaves capacity - 44.1 MB = 44.1 MB, which is
    // plenty -- so ALSO hold a second block that squeezes it below the RPA
    // path's hard in-core floor. Two labels, so the breakdown shows both.
    let _a = pool
        .reserve("grid AO cache (incumbent)", MEASURED_BECKE_PEAK)
        .expect("first incumbent fits");
    let _b = pool
        .reserve(
            "b_ov (incumbent)",
            capacity - MEASURED_BECKE_PEAK - MEASURED_RPA_HARD / 2,
        )
        .expect("second incumbent fits");

    let err = run_rpa(&f).expect_err("the sum must be refused even though each part fit");
    let msg = err.to_string();
    assert!(msg.contains("memory pool exhausted"), "{msg}");
    assert!(
        msg.contains("grid AO cache (incumbent)"),
        "the breakdown must name the dominant incumbent: {msg}"
    );
    clear_global();
}

/// LIFETIME. The charge must be held for as long as the planes are, and must
/// be RELEASED when the call returns -- not at some later point and not
/// early.
///
/// The "not early" half is what makes the gate real, and it is what mutation
/// M4 breaks. The "released on return" half is what stops a long driver from
/// exhausting the pool on transients it already freed: run the same path
/// FOUR times against a pool that only fits ONE at a time.
#[test]
fn charges_release_so_a_repeated_driver_does_not_exhaust_the_pool() {
    let _s = CleanSlot::acquire();
    let f = fixture();

    // Only ~1.5 runs' worth of capacity: if a single charge outlived its
    // call, run 2 would refuse.
    let pool = MemoryPool::with_capacity_bytes(rpa_peak() * 3 / 2);
    install_global(pool.clone());
    let mut energies = Vec::new();
    for i in 0..4 {
        let e = run_rpa(&f).unwrap_or_else(|e| {
            panic!(
                "run {i} refused -- a charge from an earlier run is still held. \
                 outstanding = {} B. {e}",
                pool.outstanding_bytes()
            )
        });
        energies.push(e);
        assert_eq!(
            pool.outstanding_bytes(),
            0,
            "run {i} left {} B outstanding after returning",
            pool.outstanding_bytes()
        );
    }
    clear_global();

    for (i, e) in energies.iter().enumerate() {
        assert_eq!(
            e.to_bits(),
            energies[0].to_bits(),
            "run {i} moved the energy ({e:.17e} vs {:.17e}) -- a pool that fills \
             and drains must not perturb any width",
            energies[0]
        );
    }
}

/// AN OPTIONAL PLANE MUST NOT STARVE A MANDATORY ONE.
///
/// This is the regression test for the gate bug the worker-count sweep
/// exposed on 2026-09-18, and it is the property that makes the window in
/// the test below worker-INDEPENDENT.
///
/// `preflight_check_closed_shell` takes the HARD charge, then the SOFT
/// frequency scratch. Immediately afterwards -- OUTSIDE this crate --
/// `compute_rpa_intermediates` builds a `ThreeIndexSource`, which HARD-charges
/// the `(naux, nao, nao)` AO tensor. A bare `try_reserve(soft)` is greedy: it
/// takes the scratch whenever it happens to fit right then, and can leave the
/// mandatory AO tensor with nothing.
///
/// The failure is WORKER-DEPENDENT and therefore easy to miss, which is why
/// it is pinned here rather than left to the capacity arithmetic. The soft
/// term scales with worker count, so at LOW widths it is small enough to
/// squeeze in and do the damage, while at high widths it fails on its own and
/// the fallback engages correctly. Measured minima (binary search) before the
/// fix:
///
/// ```text
///   workers   soft       hard+AO   hard+soft+AO   measured minimum
///   2          240_576   909_888      1_150_464   1_150_834  (starved)
///   4          481_152   909_888      1_391_040   1_391_076  (starved)
///   12       1_443_456   909_888      2_353_344     910_380  (panelled, OK)
/// ```
///
/// After the fix the minimum is `hard + AO` at all three -- 910_066..910_380 B,
/// the excess being binary-search granularity.
///
/// The assertion: at a capacity that fits BOTH hard planes but not the soft
/// scratch, the run must complete. This is genuinely distinct from the soft
/// test below -- that one can pass on a greedy gate whenever the soft term is
/// large (as it did at 12 workers), while this one is written to state the
/// contract directly.
#[test]
fn the_soft_charge_leaves_room_for_the_mandatory_plane_that_asks_next() {
    let _s = CleanSlot::acquire();
    let f = fixture();

    // # Deriving the capacity that EXPOSES greediness at any worker count
    //
    // A capacity pinned near the floor does not work, and finding that out is
    // the whole reason this test is written the way it is. A greedy gate only
    // starves the AO tensor when it actually TAKES the scratch, i.e. when
    // `soft <= capacity - hard`. At a fixed near-floor capacity that is true
    // only for small `soft`, so the mutation is caught at 1-2 workers and
    // SURVIVES from 4 up -- measured: restoring the greedy `try_reserve`
    // (mutation M6) failed this test at 2 workers and passed it at 4 and 12.
    // A test that only fails on narrow machines is the same worker-dependence
    // that hid the bug in the first place.
    //
    // So scale the capacity with the scratch. Greediness is observable when
    // BOTH hold:
    //
    //   soft <= capacity - hard          (the greedy gate takes the scratch)
    //   AO   >  capacity - hard - soft   (and the AO tensor then does not fit)
    //
    // i.e. `hard + soft <= capacity < hard + soft + AO`, non-empty for any
    // `AO > 0`. The midpoint `capacity = hard + soft + AO/2` sits inside it at
    // every worker count.
    //
    // The CORRECT gate declines the scratch here and needs only `hard + AO`,
    // so it completes -- which is the assertion. Verified: M6 now fails at 2,
    // 4 and 12.
    let hard = MEASURED_RPA_HARD;
    let soft = rpa_soft();
    let ao = MEASURED_RPA_AO_TENSOR;
    let capacity = hard + soft + ao / 2;
    let floor = hard + ao;

    // REACHABILITY. At very narrow widths `soft` is small enough that the
    // greedy-exposing window falls BELOW the floor, and no capacity both
    // exposes greediness and obliges the run to succeed. Skip loudly rather
    // than pass vacuously -- a test that cannot fail is an assumption
    // (CLAUDE.md). Measured: this bites at 1 worker (capacity 836_640 B vs
    // floor 909_888 B) and is satisfied from 2 workers up.
    if capacity < floor {
        eprintln!(
            "SKIP the_soft_charge_leaves_room_...: at {} worker(s) the soft \
             term ({soft} B) is too small for the greedy-exposing window \
             [{}, {}) to reach the {floor} B floor. No capacity here can \
             distinguish a greedy gate from a correct one, so this test is \
             inert at this width and says so rather than passing silently.",
            rayon::current_num_threads().max(1),
            hard + soft,
            hard + soft + ao
        );
        return;
    }

    let pool = MemoryPool::with_capacity_bytes(capacity);
    install_global(pool.clone());
    let r = run_rpa(&f);
    clear_global();

    let e = r.unwrap_or_else(|e| {
        panic!(
            "at {capacity} B -- enough for the hard planes ({hard} B) AND the \
             downstream AO tensor ({ao} B), floor {floor} B -- the run must \
             complete. This width is {} workers (soft {soft} B). A refusal \
             naming \"DF 3-index\" means the soft frequency scratch was \
             admitted greedily and starved a plane that has no fallback: an \
             OPTIONAL allocation killed a MANDATORY one. {e}",
            rayon::current_num_threads().max(1)
        )
    });
    assert!(e < 0.0, "sanity: correlation energy is negative, got {e}");
}

/// The SOFT half must actually be soft. The per-worker frequency-quadrature
/// scratch is charged softly because `energy::quad_panel_width` already
/// narrows the panel instead of failing.
///
/// # What the capacity has to satisfy (derived, not tuned)
///
/// Three planes are charged, in this order, across one `run_pdep_rpa`:
///
/// 1. `MEASURED_RPA_HARD` -- the in-core planes. HARD, no fallback.
/// 2. `rpa_soft()` -- per-worker frequency scratch. SOFT; declining it makes
///    the run use the panelled assembly instead.
/// 3. `MEASURED_RPA_AO_TENSOR` -- charged by `ferric_integrals::
///    ThreeIndexSource` *after* this crate's preflight returns, inside
///    `compute_rpa_intermediates`. HARD, no fallback.
///
/// For the FALLBACK to be what is under test, the capacity must satisfy BOTH:
///
/// * `capacity >= MEASURED_RPA_HARD + MEASURED_RPA_AO_TENSOR`
///   -- otherwise the run dies on a mandatory plane and the test is measuring
///   nothing about softness. This bound is worker-INDEPENDENT.
/// * `capacity < MEASURED_RPA_HARD + rpa_soft() + MEASURED_RPA_AO_TENSOR`
///   -- otherwise everything fits, the fallback never engages, and turning
///   the soft gate hard (mutation M3) would not fail. This bound SHRINKS with
///   worker count, because the soft term does.
///
/// The window is therefore `[hard + AO, hard + soft + AO)`, non-empty for any
/// `soft > 0`, i.e. at every worker count.
///
/// # Why the old constant broke, and the gate bug it exposed
///
/// This test used a frozen `MEASURED_RPA_HARD * 2` = 1_045_632 B, which is
/// BELOW `hard + AO` = 909_888 B... no: it is above it. The constant was not
/// the whole problem. Binary-searching the smallest completing capacity with
/// the ORIGINAL greedy `try_reserve` gave:
///
/// ```text
///   workers   soft        hard+AO    hard+soft+AO   measured minimum
///   2          240_576    909_888       1_150_464   1_150_834  (no fallback!)
///   4          481_152    909_888       1_391_040   1_391_076  (no fallback!)
///   12       1_443_456    909_888       2_353_344     910_380  (panelled)
/// ```
///
/// The fallback only engaged at 12 workers. At 2 and 4 the scratch was SMALL
/// enough to fit, so `try_reserve` took it -- and then starved the AO tensor,
/// which has no fallback. An optional plane was killing a mandatory one, and
/// the run died at a capacity where the panelled path would have completed.
/// That is a real over-charge, fixed in `preflight_check_closed_shell` by
/// making the soft charge reserve the downstream hard plane's headroom. After
/// the fix the measured minimum is `hard + AO` at 2, 4 AND 12 workers
/// (910_066..910_380 B, the excess being search granularity), so the window
/// below is valid everywhere.
#[test]
fn the_frequency_scratch_is_soft_so_a_tight_pool_still_completes() {
    let _s = CleanSlot::acquire();
    let f = fixture();

    // The floor: both HARD planes must fit, or the test measures nothing.
    // Worker-independent by construction.
    let floor = MEASURED_RPA_HARD + MEASURED_RPA_AO_TENSOR;
    let peak = rpa_peak();

    // REACHABILITY, checked before the run: the window must be non-empty, or
    // the pass condition is arithmetic rather than measurement (CLAUDE.md).
    assert!(
        floor < peak,
        "REACHABILITY: the window [{floor}, {peak}) is empty at \
         {} workers -- there is no capacity that admits the mandatory planes \
         while excluding the soft scratch, so this test cannot distinguish a \
         soft gate from a hard one and must not be trusted as if it could.",
        rayon::current_num_threads().max(1)
    );

    // Sit just inside the floor. Taking the LOW end (rather than the midpoint)
    // is deliberate: it is the furthest point from `peak`, so it maximises the
    // margin by which mutation M3 (soft -> hard) must fail, at every worker
    // count including the smallest soft term.
    let capacity = floor + (peak - floor) / 4;
    assert!(
        capacity >= floor && capacity < peak,
        "the derived capacity {capacity} must lie in [{floor}, {peak})"
    );

    let pool = MemoryPool::with_capacity_bytes(capacity);
    install_global(pool.clone());
    let tight = run_rpa(&f);
    clear_global();

    let e = tight.unwrap_or_else(|e| {
        panic!(
            "a pool at {capacity} B -- at or above the {floor} B floor (hard \
             {MEASURED_RPA_HARD} + AO tensor {MEASURED_RPA_AO_TENSOR}) but \
             below the {peak} B full peak -- must still complete. The \
             frequency scratch is charged SOFTLY precisely so it falls back to \
             the panelled assembly rather than refusing a job the \
             pre-migration tree runs. A refusal naming \"[freq scratch]\" \
             means the soft gate was turned hard; a refusal naming \
             \"DF 3-index\" means the soft charge is greedy again and is \
             starving a mandatory plane. {e}"
        )
    });
    assert!(
        e < 0.0,
        "the panelled fallback must still give an energy: {e}"
    );
}
