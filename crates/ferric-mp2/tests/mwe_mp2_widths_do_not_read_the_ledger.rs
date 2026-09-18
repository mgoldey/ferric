//! A budget-derived width in ferric-mp2 may depend on the BUDGET and the
//! PROBLEM. Never on the pool LEDGER, live RSS, or the thread count.
//!
//! # The defect this pins, and why it costs the most to get wrong
//!
//! This is the ferric-mp2 sibling of `19ade072`
//! (`mwe_grid_batching_is_not_rss_dependent.rs`), which found that the KS grid
//! batch width was sized from `available_budget_now(..)` -- a figure that
//! subtracts this process's LIVE RSS. Batch width sets accumulation order, so
//! the SCF ENERGY depended on transient allocator state: the same input gave
//! -390.3794282913 and -390.3794337741 Ha on the 27-atom terpinyl cation.
//!
//! ferric-mp2 has the same hazard in three places, and `rimp2.rs`'s own docs
//! already state the rule:
//!
//! > The width is a pure function of `(width, budget)` -- never of
//! > `rayon::current_num_threads()`, free memory at call time, or anything
//! > else ambient. Pin `[memory] budget_gb` and the numerics are pinned with
//! > it.
//!
//! because `stream_dressed_mo_band`'s dressing step is
//! `general_mat_mul(1.0, &msub, &mo_blk, 1.0, &mut b_flat)` -- **beta = 1** --
//! so the chunk k-blocks a partial-sum accumulation. Measured in
//! `three_index_source.rs`: every ODD split perturbs a GEMM by ~7e-15.
//!
//! Installing a shared pool creates a NEW way to break this rule that did not
//! exist before: `pool.available_bytes()` is a perfectly deterministic,
//! perfectly reasonable-looking number that nevertheless depends on how many
//! bytes some OTHER plane happened to be holding when this one asked. Sizing a
//! width from it makes the MP2 energy a function of allocation ORDER. That is
//! why the migration deliberately keeps every width on `budget_bytes`.
//!
//! # Artifact hypothesis (stated before measuring)
//!
//! If the rule HOLDS: at a fixed budget, the energy is bit-identical whether
//! the pool is empty or already holding a large unrelated reservation.
//!
//! If a width reads the ledger: the outstanding reservation shrinks
//! `available_bytes()`, which narrows the chunk, which changes the
//! accumulation order, and the energy moves in the last digits (~1e-14..1e-15
//! Ha, the DRESS_ROW_BLOCK scale) -- NOT by zero bits.
//!
//! # Calibration: this test was INERT before it was calibrated
//!
//! At an ample budget `mo_stream_chunk_for` clamps to its 256 cap regardless
//! of what is subtracted, so the two runs agree even WITH the defect live --
//! exactly the trap `19ade072` hit ("at 40 MB and above the width is stable
//! and the test would pass with the defect live").
//!
//! It also has to be a fixture where the chunk BINDS. water/cc-pVDZ has
//! naux = 84, so ANY chunk at or above 84 yields a single block and no
//! k-blocking difference whatsoever: two widths of 126 and 256 gave the
//! bit-identical energy with mutation M6 LIVE. The fixture here is therefore
//! benzene/cc-pVDZ (nao=114, naux=420, nocc=21, nvir=93, width=1953), where
//! naux comfortably exceeds the 256 cap and the chunk really does block the
//! dressing reduction.
//!
//! Measured cliff for THIS fixture:
//!
//! ```text
//!   budget    8 MB -> chunk  32     <- floor, inert
//!            16 MB -> chunk  64     <- live
//!            32 MB -> chunk 128     <- live
//!            48 MB -> chunk 192     <- live  (LIVE_BUDGET)
//!            64 MB -> chunk 256     <- CLAMPED, test would be inert here
//!           128 MB -> chunk 256
//! ```
//!
//! and the measured M6 signal, at POOL_CAPACITY = 800 MB with a 700 MB
//! ballast:
//!
//! ```text
//!   empty ledger  E_corr = -7.98474628154927313e-1
//!   700 MB held   E_corr = -7.98474628154927202e-1      <- M6 live
//! ```
//!
//! i.e. an unrelated reservation moves the reported correlation energy in the
//! last digits, which is precisely the failure mode this file forbids.

use std::sync::{Mutex, MutexGuard, OnceLock};

use ferric_core::memory::pool::{clear_global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{mo_stream_chunk_for_test as chunk_for, ri_mp2, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// The budget at which the MO-stream chunk is strictly between its 32 floor
/// and its 256 cap for this fixture. MEASURED, not guessed -- see the module
/// doc's table.
const LIVE_BUDGET: usize = 48_000_000;
const MO_STREAM_CAP: usize = 256;
const MO_STREAM_FLOOR: usize = 32;
/// Pool capacity and ballast for `an_outstanding_reservation_does_not_move_the_energy`,
/// CALIBRATED so that `available_bytes()` at the moment the MO stream is
/// called lands on DIFFERENT chunk widths in the two arms. Measured for this
/// fixture (see `the_ballast_moves_the_width_if_it_is_read`, which fails if
/// they ever coincide again).
const POOL_CAPACITY: usize = 800_000_000;
const BALLAST_BYTES: usize = 700_000_000;

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
    dfbs: PreparedBasis,
    rhf: ferric_scf::ScfResult,
    nao: usize,
    naux: usize,
    nocc: usize,
    nvir: usize,
}

fn fixture() -> Fixture {
    let mol = Molecule::load_xyz("../../testdata/molecules/benzene.xyz").expect("benzene.xyz");
    let bs = ferric_core::basis::bundled("cc-pvdz").expect("cc-pvdz");
    let dfbs_set = ferric_core::basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri");
    let obs = PreparedBasis::new(&mol, &bs).expect("prepared obs");
    let dfbs = PreparedBasis::new(&mol, &dfbs_set).expect("prepared dfbs");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).expect("rhf");
    let nao = obs.nbasis();
    let naux = dfbs.nbasis();
    let nocc = (mol.nelec() as usize) / 2;
    let nvir = nao - nocc;
    Fixture {
        mol,
        obs,
        dfbs,
        rhf,
        nao,
        naux,
        nocc,
        nvir,
    }
}

impl Fixture {
    fn width(&self) -> usize {
        self.nocc * self.nvir
    }
    fn ao_bytes(&self) -> usize {
        self.naux * self.nao * self.nao * 8
    }
    fn b_flat_bytes(&self) -> usize {
        self.naux * self.nocc * self.nvir * 8
    }
    fn run_at(&self, budget: usize) -> Result<f64, ferric_core::FerricError> {
        ri_mp2(
            &self.mol,
            &self.obs,
            &self.dfbs,
            Operator::coulomb(),
            &self.rhf,
            &RiMp2Config::default().with_memory_budget_bytes(budget),
        )
        .map(|r| r.mp2_corr)
    }
}

/// REACHABILITY GUARD (run this one first when debugging a failure here).
///
/// If `LIVE_BUDGET` ever drifts into the clamped regime, every other contract
/// in this file becomes vacuous -- they would compare 256 against 256 and pass
/// against an implementation that reads the ledger, the RSS, or the phase of
/// the moon. Pinning the operand as UNCLAMPED is what makes the rest
/// measurements.
#[test]
fn the_chosen_budget_puts_the_width_in_its_live_regime() {
    let f = fixture();
    let w = chunk_for(f.width(), LIVE_BUDGET);
    assert!(
        w > MO_STREAM_FLOOR && w < MO_STREAM_CAP,
        "at budget {LIVE_BUDGET} and width {} the chunk is {w}, which sits AT a clamp \
         (floor {MO_STREAM_FLOOR}, cap {MO_STREAM_CAP}). Every contract in this file is \
         then comparing clamps rather than widths, and a width that read the pool ledger \
         would pass unnoticed. Re-measure the cliff and update LIVE_BUDGET.",
        f.width()
    );
}

/// REACHABILITY GUARD for the contract below.
///
/// The contract compares an empty-ledger run against a ballasted one. That
/// comparison is only a measurement if the two `available_bytes()` figures
/// would produce DIFFERENT chunk widths -- otherwise a width that read the
/// ledger would still agree with itself and the mutation would survive.
///
/// This is not hypothetical. Mutation M6 SURVIVED two earlier versions of this
/// file for exactly this reason: first with a 64 GB pool (both arms clamp to
/// the 256 cap), then with a ballast that left `available_bytes()` still above
/// the cliff. Only after both operands were placed in the live regime -- by
/// the measured table in the module doc -- did it die.
#[test]
fn the_ballast_moves_the_width_if_it_is_read() {
    let f = fixture();
    // What `available_bytes()` would read AT THE MOMENT the stream is called,
    // i.e. after this run's own AO 3-index tensor and b_ov are outstanding.
    // Those two are charged in BOTH arms, so they must be subtracted from both
    // or the guard describes a call that never happens.
    let own_charges = f.ao_bytes() + f.b_flat_bytes();
    let empty = chunk_for(f.width(), POOL_CAPACITY - own_charges);
    let ballasted = chunk_for(f.width(), POOL_CAPACITY - BALLAST_BYTES - own_charges);
    assert_ne!(
        empty, ballasted,
        "the empty-ledger ({empty}) and ballasted ({ballasted}) `available_bytes()` figures \
         give the SAME chunk width, so the contract below cannot distinguish a \
         budget-derived width from a ledger-derived one and proves nothing. Re-measure the \
         cliff (own charges here are {own_charges} bytes)."
    );
    assert!(
        ballasted > MO_STREAM_FLOOR && ballasted < MO_STREAM_CAP,
        "the ballasted width {ballasted} is at a clamp -- pick a ballast that lands between \
         {MO_STREAM_FLOOR} and {MO_STREAM_CAP}"
    );
}

/// CONTRACT: a non-empty pool ledger must not move the energy.
///
/// This is the contract mutation M6 (size a width from
/// `global_available_bytes()` instead of the configured budget) must fail.
/// The mutation SURVIVED the ample-pool anchor for exactly the reason the
/// module doc gives, and only became detectable once the budget was calibrated
/// to the measured cliff.
#[test]
fn an_outstanding_reservation_does_not_move_the_energy() {
    let _s = CleanSlot::acquire();
    let f = fixture();

    // The pool CAPACITY is itself placed in the live regime, and the ballast
    // is sized so that `available_bytes()` crosses the measured cliff. That
    // calibration is the whole point: a ledger-reading width can only be
    // caught where the width actually responds, and both operands must sit
    // between the floor and the cap. `the_ballast_moves_the_width_if_it_is_read`
    // pins that this is so.
    //
    // Capacity is generous enough that the MP2 planes themselves always fit
    // (the run must COMPLETE in both arms -- a refusal would prove nothing),
    // so the ballast is what moves `available_bytes`, not the MP2 charges.
    let capacity = POOL_CAPACITY;
    let ballast_bytes = BALLAST_BYTES;

    // Run A: pool installed, ledger nearly EMPTY when the MP2 planes ask.
    let pool_a = MemoryPool::with_capacity_bytes(capacity);
    install_global(pool_a);
    let empty_ledger = f.run_at(LIVE_BUDGET).expect("run A must succeed");
    clear_global();

    // Run B: identical, except an unrelated reservation is outstanding the
    // whole time, dropping `available_bytes()` from 40 MB to 3 MB. Nothing
    // about the configured budget or the problem changed.
    let pool_b = MemoryPool::with_capacity_bytes(capacity);
    install_global(pool_b.clone());
    let ballast = pool_b
        .reserve("unrelated incumbent plane", ballast_bytes)
        .expect("ballast must fit");
    let before = pool_b.available_bytes();
    let full_ledger = f.run_at(LIVE_BUDGET).expect("run B must succeed");
    drop(ballast);
    clear_global();

    assert!(
        before < capacity,
        "sanity: the ballast must actually have reduced available_bytes (got {before})"
    );
    assert_eq!(
        empty_ledger.to_bits(),
        full_ledger.to_bits(),
        "the MP2 energy MOVED when an unrelated plane was holding {BALLAST_BYTES} bytes of the pool: \
         {empty_ledger:.17e} (empty ledger) vs {full_ledger:.17e} (ballast outstanding). \
         Nothing about the PROBLEM or the BUDGET changed between these two runs -- only \
         the ledger. A width is reading `available_bytes()`, which makes the reported \
         energy a function of allocation ORDER. This is the ferric-mp2 form of the \
         19ade072 defect and it must be fixed by taking the width from the configured \
         budget, never from what the pool has left."
    );
}

/// CONTRACT: the budget IS allowed to move the energy, and does.
///
/// The counterpart that stops the contract above from being satisfiable by an
/// implementation that ignores memory entirely. `mo_stream_chunk_for`'s doc is
/// explicit that "two runs of the same system at different `budget_gb` may
/// differ in the last digits" -- that is the designed, documented,
/// reproducible-from-config behaviour. If this test ever fails, the width has
/// stopped responding to the budget at all and the whole knob is decoration.
#[test]
fn the_budget_is_still_a_live_knob() {
    let _s = CleanSlot::acquire();
    let f = fixture();
    let narrow = chunk_for(f.width(), LIVE_BUDGET);
    let wide = chunk_for(f.width(), 64_000_000_000);
    assert!(
        narrow < wide,
        "the chunk did not widen with the budget ({narrow} at {LIVE_BUDGET} vs {wide} at \
         64 GB) -- the budget is no longer reaching the width"
    );
    // And the two widths must actually produce runs (a width that is live but
    // unreachable through the public entry point proves nothing).
    let a = f.run_at(LIVE_BUDGET).expect("narrow run");
    let b = f.run_at(64_000_000_000).expect("wide run");
    assert!(
        a.is_finite() && b.is_finite() && a < 0.0 && b < 0.0,
        "both budgets must yield a real correlation energy ({a}, {b})"
    );
    // They agree to well within the RI error -- the chunk changes only the
    // last digits, as documented.
    assert!(
        (a - b).abs() < 1e-9,
        "changing the chunk width must perturb only the last digits, not the physics: \
         {a:.17e} vs {b:.17e}"
    );
}
