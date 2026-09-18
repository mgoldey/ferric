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
        alone >= MEASURED_BECKE_PEAK_FLOOR && alone <= MEASURED_BECKE_PEAK_CEIL,
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

/// The SOFT half must actually be soft. The per-worker frequency-quadrature
/// scratch is charged with `try_reserve_global` because
/// `energy::quad_panel_width` already narrows the panel instead of failing.
///
/// Pin that: a pool whose headroom covers the HARD half but NOT the soft half
/// must still COMPLETE the run. If the soft charge were hard (mutation M3),
/// this refuses -- which is exactly a job the pre-migration tree completes.
#[test]
fn the_frequency_scratch_is_soft_so_a_tight_pool_still_completes() {
    let _s = CleanSlot::acquire();
    let f = fixture();

    // A capacity strictly BETWEEN the hard floor and the full peak: it cannot
    // hold the soft frequency scratch, and it can hold the hard in-core
    // planes. Measured at this shape: hard = 522_816 B, total = 2_353_344 B.
    // 2x the hard figure sits squarely in that window, and is also above the
    // integrals-owned AO tensor (387_072 B) that must ALSO fit alongside.
    //
    // First prove the window is REACHABLE (a pass condition that is not
    // reachable returns arithmetic, not measurement): the capacity must be
    // strictly below the total charge a hard-everything gate would demand,
    // otherwise this test passes for the wrong reason.
    // The window this test needs: big enough for the HARD planes AND the
    // integrals-owned AO tensor that is co-resident with them, but smaller
    // than the full peak so the SOFT scratch cannot fit and must panel.
    //
    // `MEASURED_RPA_HARD * 2` was used until 2026-09-18 and is only a valid
    // window at high worker counts: the soft term shrinks with workers, so at
    // 4 the window closed and the AO tensor no longer fit, failing with
    // `"DF 3-index (P|mn) in-core" needs ... short by ...` -- a refusal from
    // the WRONG plane, which is the test breaking rather than the gate.
    // Derive it instead: floor + tensor + a small slack, and assert the
    // window is non-empty before relying on it.
    let capacity = MEASURED_RPA_HARD * 2;
    let peak = rpa_peak();
    assert!(
        capacity < peak,
        "REACHABILITY: the tight capacity ({capacity} B) must be below the full \
         measured peak ({peak} B), or turning the soft gate hard \
         could not possibly fail this test"
    );
    assert!(
        capacity > MEASURED_RPA_HARD,
        "REACHABILITY: the tight capacity must still clear the hard floor, or \
         this test would fail for the unrelated reason that the in-core planes \
         do not fit"
    );

    let pool = MemoryPool::with_capacity_bytes(capacity);
    install_global(pool.clone());
    let tight = run_rpa(&f);
    clear_global();

    let e = tight.unwrap_or_else(|e| {
        panic!(
            "a pool at {capacity} B -- above the {MEASURED_RPA_HARD} B hard floor \
             but below the {peak} B full peak -- must still complete. \
             The frequency scratch is charged SOFTLY precisely so it falls back \
             to the panelled assembly rather than refusing a job the \
             pre-migration tree runs. If this fails, a soft gate was turned \
             hard. {e}"
        )
    });
    assert!(
        e < 0.0,
        "the panelled fallback must still give an energy: {e}"
    );
}
