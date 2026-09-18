//! MWE: the two large KS-DFT planes must COMPOSE against one shared pool.
//!
//! # The measured defect
//!
//! ```text
//! [ferric] memory budget: 4.72 GiB  [source: auto (0.8 x available RAM)]
//! 27-atom C10H17+ cation, def2-SVP, B3LYP, RI-JK
//! WALL 1344 s   MAXRSS 6.04 GiB   SP_EXIT=137      <- global OOM kill
//! ```
//!
//! Not one gate failed. The DF 3-index tensor asked "do I fit in 4.72 GiB?"
//! (yes); the grid AO cache asked the same question against the same full
//! number (yes); the process held the SUM. Two gates, each correct in
//! isolation, that do not compose.
//!
//! # What this file asserts
//!
//! 1. A charge is HELD for its plane's lifetime, not just across the gate. A
//!    reservation dropped when the check returns is decoration: the bytes are
//!    credited back while the array is still resident, so the next plane is
//!    admitted against them a second time and nothing has changed.
//! 2. The second plane to ask sees only what the first LEFT.
//! 3. A KS-DFT job whose planes together exceed the pool is REFUSED up front,
//!    with a breakdown naming the dominant plane -- instead of being
//!    OOM-killed.
//!
//! # Artifact hypothesis (stated before measuring, per the protocol)
//!
//! If the composition property is REAL, a pool sized between one plane and
//! two admits the first plane and refuses/diverts the second, and the
//! refusal text names a DF or grid plane.
//!
//! If the wiring is BROKEN in the most likely way -- the guard dropped at the
//! end of the gate's own statement, which is what "charge the check, not the
//! tensor" looks like -- then `outstanding_bytes()` returns to 0 after the
//! build and BOTH planes are admitted, exactly as before the pool existed.
//! Those predictions differ (outstanding > 0 vs == 0; refused vs admitted),
//! so this test distinguishes them.

use std::sync::{Mutex, MutexGuard, OnceLock};

use ferric_core::memory::pool::{clear_global, global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

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

fn water() -> Molecule {
    Molecule::load_xyz("../../testdata/molecules/water.xyz").expect("water.xyz")
}

/// A PBE/RI-J single point. `Err` carries the refusal text.
fn try_ksdft(basis: &str) -> Result<f64, String> {
    let mol = water();
    let bs = ferric_core::basis::bundled(basis).expect("basis");
    let obs = PreparedBasis::new(&mol, &bs).expect("prepared basis");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        xc: Some("PBE".into()),
        df_j_aux: Some("def2-universal-jkfit".into()),
        ..Default::default()
    };
    solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg)
        .map(|r| r.energy)
        .map_err(|e| e.to_string())
}

/// THE property item 4 of the task is about: a charge must OUTLIVE the gate.
///
/// If the DF 3-index reservation were taken and dropped inside the build
/// function, `outstanding_bytes()` would be back to 0 the moment the source
/// was constructed, and the grid cache would then be admitted against the
/// full pool again -- the original double-spend.
#[test]
fn a_built_three_index_source_keeps_its_charge_outstanding() {
    let _s = CleanSlot::acquire();
    let pool = MemoryPool::with_capacity_bytes(8_000_000_000);
    install_global(pool.clone());

    let mol = water();
    let bs = ferric_core::basis::bundled("cc-pvdz").expect("cc-pvdz");
    let aux = ferric_core::basis::bundled("def2-universal-jkfit").expect("jkfit");
    let obs = PreparedBasis::new(&mol, &bs).expect("obs");
    let dfbs = PreparedBasis::new(&mol, &aux).expect("dfbs");

    let before = pool.outstanding_bytes();
    let src = ferric_integrals::three_index_source::ThreeIndexSource::build(
        Operator::coulomb(),
        &obs,
        &dfbs,
        8_000_000_000,
    )
    .expect("in-core build must fit an 8 GB pool");

    let held = pool.outstanding_bytes();
    assert!(
        held > before,
        "a LIVE ThreeIndexSource must keep its bytes debited; outstanding went \
         {before} -> {held}. If this is 0, the reservation was dropped when the \
         build returned and the charge is decoration, not accounting."
    );
    // And the breakdown must name it, so a refusal can point at the culprit.
    let rep = pool.occupancy_report();
    assert!(
        rep.contains("DF 3-index"),
        "the occupancy breakdown must name the DF plane:\n{rep}"
    );

    // Dropping the source releases the bytes -- an SCF that builds and frees
    // a source per stage must not leak the pool.
    drop(src);
    assert_eq!(
        pool.outstanding_bytes(),
        before,
        "dropping the source must credit its bytes back"
    );
    clear_global();
}

/// The grid AO cache must hold its charge for the CACHE's lifetime too.
///
/// The DF sibling of this test (`a_built_three_index_source_keeps_its_charge_outstanding`)
/// cannot see this: the two planes are charged in different crates, by
/// different mechanisms (`ThreeIndexSource::_charge` vs
/// `GridCache::Full::_charge`), so a lifetime test for one says nothing about
/// the other. This is the test that caught mutation M4 (grid charge dropped
/// after the Full-vs-Batched decision instead of stored beside chi/dchi),
/// which every other test in this file survived.
///
/// The observable has to be taken WHILE the cache is alive, which means
/// reaching inside the SCF is not an option -- `solve_rhf` builds and drops
/// the KsXc internally. Instead: build the KsXc directly and watch the
/// ledger, which is exactly the plane's own lifetime.
#[test]
fn a_live_grid_cache_keeps_its_charge_outstanding() {
    let _s = CleanSlot::acquire();
    let pool = MemoryPool::with_capacity_bytes(8_000_000_000);
    install_global(pool.clone());

    let mol = water();
    let bs = ferric_core::basis::bundled("cc-pvdz").expect("cc-pvdz");
    let main = ferric_dft::grid::AtomicGridConfig::default();
    let nlc = ferric_dft::grid::AtomicGridConfig {
        n_radial: 50,
        n_angular: 50,
        ..Default::default()
    };

    let before = pool.outstanding_bytes();
    let ks = ferric_dft::ks::KsXc::new_with_omega_budgeted(
        &mol,
        &bs,
        "PBE",
        &main,
        &nlc,
        None,
        Some(8_000_000_000),
    )
    .expect("an 8 GB pool must admit the Full grid cache");

    let held = pool.outstanding_bytes();
    assert!(
        held > before,
        "a LIVE grid AO cache must keep its bytes debited; outstanding went \
         {before} -> {held}. If this is 0, the guard was dropped after the \
         Full-vs-Batched decision and the charge is decoration -- the DF \
         tensor would then be admitted against the same bytes a second time."
    );
    let rep = pool.occupancy_report();
    assert!(
        rep.contains("grid AO cache"),
        "the occupancy breakdown must name the grid plane:\n{rep}"
    );

    drop(ks);
    assert_eq!(
        pool.outstanding_bytes(),
        before,
        "dropping the KsXc must credit the grid cache's bytes back"
    );
    clear_global();
}

/// The composition property itself: plane two sees only what plane one left.
#[test]
fn the_second_plane_sees_only_what_the_first_left() {
    let _s = CleanSlot::acquire();
    let pool = MemoryPool::with_capacity_bytes(4_000_000_000);
    install_global(pool.clone());

    // Plane one takes 3 of the 4 GB.
    let first = pool
        .reserve("DF 3-index (P|mn) in-core", 3_000_000_000)
        .expect("3 GB of 4 GB fits");
    assert_eq!(pool.available_bytes(), 1_000_000_000);

    // Plane two asks for 3 GB. Against the OLD ceiling check it would compare
    // 3 GB against the whole 4 GB budget and pass -- that is the defect. It
    // must now see 1 GB.
    let second = pool.try_reserve("KS-DFT grid AO cache (chi + grad chi)", 3_000_000_000);
    assert!(
        second.is_none(),
        "the second plane must be measured against what is LEFT (1 GB), not \
         against the whole ceiling (4 GB) -- otherwise the process holds the sum"
    );

    drop(first);
    assert!(
        pool.try_reserve("KS-DFT grid AO cache (chi + grad chi)", 3_000_000_000)
            .is_some(),
        "once the first plane releases, the same ask must fit"
    );
    clear_global();
}

/// End to end: a pool too small for the job REFUSES it up front, and says
/// which plane is dominant.
#[test]
fn an_undersized_pool_refuses_the_job_and_names_the_dominant_plane() {
    let _s = CleanSlot::acquire();
    // Too small for water/cc-pVDZ's DF tensor + grid cache, but not absurd:
    // the measured peak of this job against an ample pool is ~0.52 MB, so a
    // 200 kB pool cannot cover the dominant plane while still being the same
    // SHAPE of failure as the 27-atom case (planes that each "fit" a ceiling
    // but not a shared ledger), scaled down to a test-sized system.
    install_global(MemoryPool::with_capacity_bytes(200_000));

    let err = try_ksdft("cc-pvdz")
        .err()
        .expect("an undersized pool must REFUSE, not OOM-kill");
    assert!(
        err.contains("memory pool exhausted") || err.contains("pool cannot cover"),
        "the refusal must come from the pool, not a downstream panic:\n{err}"
    );
    assert!(
        err.contains("DF 3-index") || err.contains("grid AO cache"),
        "the refusal must NAME the plane that could not be covered:\n{err}"
    );
    clear_global();
}

/// The gate whose over-budget answer is "batch it" must keep answering "batch
/// it" -- not start erroring.
///
/// The grid AO cache has a real fallback (batch the grid arbitrarily finely),
/// which is why it uses `try_reserve`. A pool that cannot hold the full grid
/// cache but can hold the DF tensor and a batch must still RUN.
#[test]
fn a_pool_too_small_for_the_full_grid_cache_still_runs_by_batching() {
    let _s = CleanSlot::acquire();
    // CALIBRATED, not guessed. Measured peaks for this exact job:
    //   2 GB pool  -> peak 29.70 MB  (grid cache resident: the Full path)
    //   1 MB pool  -> peak  0.15 MB  (grid gate diverted to batching)
    // So 1 MB is inside the batching regime while still covering the DF
    // tensor. A cap in the Full regime would make this test INERT for its
    // stated purpose -- it would pass without the fallback ever being taken.
    // Mutation M3 (grid gate hard-fails instead of batching) is what proves
    // this cap actually exercises the fallback.
    const BATCHING_REGIME_CAP: usize = 1_000_000;
    install_global(MemoryPool::with_capacity_bytes(BATCHING_REGIME_CAP));
    let e = try_ksdft("6-31g").expect("a batching fallback must still converge");
    assert!(
        e.is_finite() && e < 0.0,
        "batched KS-DFT must produce a sane energy, got {e}"
    );
    // The batched path must reach the SAME energy the Full path does -- the
    // fallback is a memory strategy, not an approximation. Measured: both
    // give -76.29805950374167 on water/6-31G/PBE.
    assert!(
        (e - -76.298_059_503_741_67f64).abs() < 1e-9,
        "the batched grid path must agree with the Full path; got {e}"
    );
    // And the pool must have been genuinely tight -- if the whole grid cache
    // had fit, the fallback was never taken and this test proves nothing.
    let peak = global().map(|p| p.peak_bytes()).unwrap_or(0);
    assert!(
        peak < 29_000_000,
        "peak {peak} suggests the FULL grid cache was resident, so the \
         batching fallback was never exercised and this test is inert"
    );
    // The pool must be fully unwound afterwards -- an 80-iteration SCF that
    // leaked per-iteration charges would exhaust it.
    let outstanding = global().map(|p| p.outstanding_bytes()).unwrap_or(0);
    assert_eq!(
        outstanding, 0,
        "every reservation must have been released by the end of the SCF"
    );
    clear_global();
}
