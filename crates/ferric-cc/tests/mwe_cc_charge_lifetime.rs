//! LIFETIME: a charge must be HELD for its buffer's life, not borrowed across
//! the gate and handed straight back.
//!
//! # Why this file exists separately, and what it caught
//!
//! `mwe_cc_planes_compose_in_one_pool.rs` pins ADMISSION: two planes whose sum
//! exceeds the pool cannot both be admitted. Mutation M1 of this migration --
//! `let _charge = plan.commit()?; drop(_charge);`, i.e. charge and immediately
//! release while every tensor is still resident -- SURVIVED all four of those
//! tests. That survival is instructive:
//!
//! * the admission test passes because `MemoryPlan::commit` is `check()` THEN
//!   reserve, and `with_pool` re-bases the plan's ceiling onto
//!   `pool.available_bytes()`. So the CHECK is already pool-aware; throwing
//!   the RESERVE away does not change who gets refused at the gate.
//! * `peak_bytes()` is a high-water mark, so it records even a reservation
//!   that lived for one instruction.
//! * the release tests pass vacuously: a charge never held is certainly
//!   released.
//!
//! In other words, admission tests cannot see lifetime. Only a SECOND
//! reservation taken while the first is supposed to be outstanding can.
//!
//! # The deterministic hook
//!
//! `ccsd_t` composes two charges inside one function, in a fixed order with no
//! threading and no timing:
//!
//! ```text
//!   let _floor_charge = reserve_global("CCSD(T) precomputed blocks", floor)?;
//!   ... build bcei / majk / bcjk ...
//!   let band = try_reserve_global("CCSD(T) triple band", width * per_triple);
//! ```
//!
//! The band reserve happens WHILE the floor charge is supposed to be held, so
//! a pool sized to `floor + k*per_triple` leaves exactly `k` triples of
//! headroom and the ledger's PEAK must land on exactly `floor + k*per_triple`.
//! Release the floor at the gate and the band instead sees the whole pool
//! free, takes a much wider band, and the peak lands on `width*per_triple`
//! with no floor underneath -- a DIFFERENT number.
//!
//! That peak is the observable, and it is exact: no polling, no threads, no
//! wall clock.
//!
//! It must also be asserted as an EQUALITY. This test first carried two
//! inequalities (`peak > floor` and `peak <= capacity`), and mutation M1b
//! slipped straight between them: at this fixture the held peak is 75968 bytes
//! and the dropped peak is 73728, both inside that window. M1b was killed for
//! a while only because an unrelated `Share::Half` was still narrowing the
//! band, and it came back to life the moment that halving was removed for
//! being what made the soft gate unreachable. Two inequalities a mutation can
//! pass through are not a measurement.
//!
//! # What is NOT pinned here, and honestly why
//!
//! The same mutation applied to `ccsd.rs`'s own `_charge` -- commit then drop
//! immediately -- still SURVIVES every test in this crate, and that survival
//! is reported rather than papered over.
//!
//! The reason is structural, not a gap in the tests: `ccsd` takes exactly ONE
//! reservation and nothing inside it ever reserves again (`eri3_tensor`,
//! `build_b` and the amplitude loop all allocate without touching the pool).
//! So within a single-threaded run there is no second observer, and
//! "held for the driver's lifetime" and "released at the gate" produce
//! identical ledger histories apart from the instant of the reserve, which
//! `peak_bytes()` records either way.
//!
//! The hold is therefore not load-bearing for `ccsd` TODAY. It is still the
//! correct shape -- it costs one `usize` on the stack, it is what makes the
//! charge composable the moment any nested gate appears below it (a pooled
//! `ThreeIndexSource`, a pooled DIIS, a second driver on another thread), and
//! writing it the other way would be a latent defect waiting for that change.
//! Two candidate tests were considered and rejected as dishonest: a polling
//! watcher thread (a race dressed as a measurement) and a two-thread admission
//! count (passes under the mutation, because a serialized schedule admits both
//! either way). A test that cannot fail is an assumption, so neither was
//! shipped.

use std::sync::{Mutex, MutexGuard, OnceLock};

use ferric_cc::ccsd::ccsd;
use ferric_cc::ccsd_t::{ccsd_t, t_floor_bytes, t_per_triple_bytes};
use ferric_cc::CcConfig;
use ferric_core::basis;
use ferric_core::memory::pool::{clear_global, global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// H2O / STO-3G: no2 = 10, nv2 = 4 spin-orbitals.
const NO2: usize = 10;
const NV2: usize = 4;

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
    rhf: ScfResult,
}

fn fixture() -> Fixture {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").expect("water.xyz");
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").expect("sto-3g")).expect("obs");
    let dfbs =
        PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri")).expect("dfbs");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).expect("RHF");
    Fixture {
        mol,
        obs,
        dfbs,
        rhf,
    }
}

fn cfg() -> CcConfig {
    CcConfig {
        frozen_core: 0,
        max_iter: 30,
        memory_budget_bytes: Some(8_000_000_000),
        ..Default::default()
    }
}

#[test]
fn the_t_floor_charge_is_held_while_the_band_is_charged() {
    let _s = CleanSlot::acquire();
    let floor = t_floor_bytes(NO2, NV2);
    let per_triple = t_per_triple_bytes(NV2);

    let f = fixture();
    let cc = ccsd(&f.mol, &f.obs, &f.dfbs, Operator::coulomb(), &f.rhf, &cfg())
        .expect("CCSD must converge");

    // Room for the floor plus exactly K extra triples' worth of band.
    const K: usize = 3;
    let capacity = floor + K * per_triple;
    let pool = MemoryPool::with_capacity_bytes(capacity);
    install_global(pool.clone());

    let _ = ccsd_t(
        &f.mol,
        &f.obs,
        &f.dfbs,
        Operator::coulomb(),
        &f.rhf,
        &cc,
        &cfg(),
    )
    .expect("(T) must still run: the band is charged softly");

    let peak = pool.peak_bytes();

    // THE DISCRIMINATOR, and it has to be an EQUALITY.
    //
    // With the floor HELD across the band reserve:
    //   available = capacity - floor = K * per_triple
    //   width     = K
    //   peak      = floor + K * per_triple = capacity
    //
    // With the floor DROPPED at the gate:
    //   available = capacity  (the floor is not on the ledger)
    //   width     = capacity / per_triple  (much wider -- 24 here, not 3)
    //   peak      = width * per_triple, with NO floor underneath it
    //
    // At this fixture those are 75968 and 73728 bytes. Both are <= capacity
    // and both are > floor, which is why the inequality assertions this test
    // originally carried were VACUOUS: mutation M1b (drop the floor charge)
    // was killed by them at one point only because a `Share::Half` was still
    // narrowing the band, and it came back to life the moment that halving was
    // removed. Two inequalities that a mutation can slip between are not a
    // measurement.
    let expected_peak = floor + K * per_triple;
    assert_eq!(
        peak,
        expected_peak,
        "ledger peak {peak} != floor {floor} + a K={K} band ({} bytes). \
         The band was sized against {} bytes of apparent headroom instead of \
         the {} the floor-held ledger leaves -- i.e. the floor charge was NOT \
         outstanding when the band asked.",
        K * per_triple,
        peak,
        K * per_triple,
    );
    assert!(
        peak <= capacity,
        "ledger oversubscribed: peak {peak} > capacity {capacity}"
    );
    assert_eq!(
        pool.outstanding_bytes(),
        0,
        "everything must be released once (T) returns"
    );
    clear_global();
}

#[test]
fn the_t_floor_charge_blocks_a_concurrent_incumbent_for_the_whole_step() {
    // The complementary construction, and the one that most directly mirrors
    // the original incident: while (T) holds its precomputed blocks, a second
    // subsystem asking for the same bytes must be REFUSED. Pre-pool both would
    // have passed against the same re-read ceiling.
    let _s = CleanSlot::acquire();
    let floor = t_floor_bytes(NO2, NV2);

    let f = fixture();
    let cc = ccsd(&f.mol, &f.obs, &f.dfbs, Operator::coulomb(), &f.rhf, &cfg())
        .expect("CCSD must converge");

    // Capacity for one floor and a sliver.
    let pool = MemoryPool::with_capacity_bytes(floor + floor / 4);
    install_global(pool.clone());

    // An incumbent holding the sliver: (T)'s floor now cannot fit.
    let _incumbent = pool
        .reserve("incumbent subsystem", floor / 2)
        .expect("the sliver fits");

    let err = match ccsd_t(
        &f.mol,
        &f.obs,
        &f.dfbs,
        Operator::coulomb(),
        &f.rhf,
        &cc,
        &cfg(),
    ) {
        Err(e) => e.to_string(),
        Ok(_) => panic!(
            "(T) was admitted although only {} of its {floor}-byte floor was \
             free -- the gate is re-reading a ceiling, not debiting a ledger",
            pool.available_bytes()
        ),
    };
    assert!(
        err.contains("incumbent subsystem"),
        "the refusal's occupancy breakdown must name the incumbent, so the \
         report says WHICH plane is dominant: {err}"
    );
    assert!(
        global().is_some(),
        "the pool must still be installed after a refusal"
    );
    clear_global();
}

/// The band width must be a deterministic function of the POOL LEDGER, never
/// of live RSS.
///
/// # What mutation testing found, and why this test had to be added
///
/// Mutation M7 replaced the pooled branch's `available_bytes()` with
/// `ferric_core::memory::available_budget_now(budget)` -- which subtracts LIVE
/// RSS from the budget. That is the precise shape of the KS-DFT grid
/// batch-width defect. It SURVIVED every test in this crate.
///
/// The survival is worth stating carefully rather than treating as a bug in
/// the tests. On the ksdft path the RSS-derived width was a CORRECTNESS defect
/// because batch width set accumulation order, so the SCF energy moved
/// (-390.3794282913 vs -390.3794337741 Ha on one input). Here the fold order
/// across chunks is ascending regardless of width, which the sibling tests in
/// this file prove to the bit -- so an RSS-derived width could not move `et`,
/// and no energy assertion can ever see it.
///
/// What it WOULD do is make the band width, and therefore the run's memory
/// profile and parallel shape, depend on transient allocator state: the same
/// job on the same machine would band differently depending on what else had
/// run first. That is a determinism regression, and the only way to catch it
/// is to assert on the width directly against a KNOWN ledger state.
#[test]
fn the_pooled_band_width_is_a_pure_function_of_the_ledger() {
    // Shares this file's `CleanSlot`, which both serializes against the other
    // pool-installing tests here AND clears the process-global slot on entry
    // and exit. Both halves are required: cargo runs a binary's tests
    // concurrently, so a leaked pool refuses a sibling's CC driver.
    let _s = CleanSlot::acquire();

    let per_triple = t_per_triple_bytes(NV2);
    let floor = t_floor_bytes(NO2, NV2);

    // A pool with a KNOWN amount free after the floor: exactly K triples.
    for k in [1usize, 4, 9] {
        clear_global();
        let pool = MemoryPool::with_capacity_bytes(floor + k * per_triple);
        install_global(pool.clone());
        let _floor = pool.reserve("floor", floor).expect("floor fits");

        // THE DRIVER'S OWN EXPRESSION, not a reconstruction of it: `ccsd_t`
        // calls `t_band_width_now`, so a defect inside it is visible here.
        // While this was a reconstruction, mutation M7 (swap the ledger for
        // live RSS) survived.
        let expected = ferric_cc::ccsd_t::t_band_width_now(NO2, NV2, 8_000_000_000);
        assert_eq!(
            expected, k,
            "ledger arithmetic drifted: {k} triples free should give width {k}"
        );

        // Allocate a large heap buffer to move LIVE RSS without touching the
        // ledger. A width derived from RSS would change here; a width derived
        // from the ledger cannot.
        let ballast: Vec<u8> = vec![7u8; 512 * 1024 * 1024];
        std::hint::black_box(&ballast);
        let after_rss_moved = ferric_cc::ccsd_t::t_band_width_now(NO2, NV2, 8_000_000_000);
        drop(ballast);

        assert_eq!(
            expected, after_rss_moved,
            "the band width moved when LIVE RSS moved, although the LEDGER did \
             not: {expected} -> {after_rss_moved}. The width is reading the OS, \
             not the pool -- this is the KS-DFT grid batch-width defect."
        );
    }
    clear_global();
}
