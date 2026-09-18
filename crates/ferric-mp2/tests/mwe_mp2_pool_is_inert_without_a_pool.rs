//! EXACTNESS ANCHOR for wiring the shared memory pool into the ferric-mp2
//! lanes.
//!
//! Written and made to pass BEFORE any ferric-mp2 gate was migrated, per
//! `CLAUDE.md`'s Experimental Protocol ("EXACTNESS ANCHOR FIRST"), and
//! mirroring `ferric-scf/tests/mwe_ksdft_pool_is_inert_without_a_pool.rs`,
//! which is the ksdft path's equivalent.
//!
//! The trivial limit of the pool is "no pool installed". In that limit an
//! RI-MP2 / U-RI-MP2 / Laplace-SOS-MP2 run must be BIT-IDENTICAL to the
//! pre-pool tree: every migrated gate must take the same branch, hold an inert
//! reservation, and produce the same energy to the last bit.
//!
//! ## Artifact hypothesis (stated before measuring, per the protocol)
//!
//! If the no-op property is REAL: with no pool installed, `reserve_global`
//! hands back an inert reservation, `try_reserve_global` ADMITS, and the
//! in-core-vs-spill decision inside `ThreeIndexSource`, the
//! `mo_stream_chunk_for` width, and the Laplace MO-stream chunk are taken on
//! exactly the numbers they were taken on before — so two runs of the same
//! input differ by exactly 0 bits.
//!
//! If the implementation is BROKEN in the most likely way — a migrated gate
//! that treats "no pool" as "no memory available" (`try_reserve` returning
//! `None`, or `global_available_bytes()` unwrapped to 0) — then the
//! `ThreeIndexSource` would start choosing the spill/recompute backend, or a
//! stream chunk would clamp to `MO_STREAM_CHUNK_MIN`. Both change a
//! partial-sum accumulation order (`stream_dressed_mo_band` dresses with
//! `beta = 1`), so the correlation energy would move at ~1e-14..1e-15 Ha
//! rather than by 0 bits. That prediction differs from the real one, so this
//! test can distinguish them.
//!
//! The second, sharper failure mode a pool can introduce is an AMPLE pool
//! nevertheless perturbing a width — e.g. if a migrated site started sizing a
//! chunk from `pool.available_bytes()` (which shrinks as other planes are
//! charged) rather than from the configured budget. `installing_an_ample_pool_
//! does_not_move_*` is the test for that, and it is the reason a width may
//! depend on the BUDGET but never on the LEDGER.
//!
//! The pool is process-global, so these tests serialize on a lock and clear
//! the slot on entry and exit.

use std::sync::{Mutex, MutexGuard, OnceLock};

use ferric_core::memory::pool::{clear_global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_mp2::u_rimp2::u_ri_mp2;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, UhfConfig};

fn global_lock() -> MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    match L.get_or_init(|| Mutex::new(())).lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    }
}

/// Clear the process-global pool on entry AND exit so one test's pool can
/// never leak into another's "unbudgeted" assertions. This is the guard
/// `e9949eed` had to add on the integrals path after a 100 kB pool one test
/// installed refused eight sibling tests that never asked for it.
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

/// Closed-shell RI-MP2 on water/6-31G, returning (E_corr, E_total).
fn run_rimp2() -> (f64, f64) {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").expect("water.xyz");
    let bs = ferric_core::basis::bundled("6-31g").expect("6-31g");
    let dfbs_set = ferric_core::basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri");
    let obs = PreparedBasis::new(&mol, &bs).expect("prepared obs");
    let dfbs = PreparedBasis::new(&mol, &dfbs_set).expect("prepared dfbs");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).expect("rhf");
    let r = ri_mp2(&mol, &obs, &dfbs, op, &rhf, &RiMp2Config::default()).expect("ri_mp2");
    (r.mp2_corr, r.total_energy)
}

/// Open-shell U-RI-MP2 on a lithium atom (doublet) — small, but it exercises
/// `check_u_mo_side_alloc` and holds BOTH spin intermediates at once.
fn run_u_rimp2() -> (f64, f64) {
    let mol = Molecule::parse_xyz("1\nLi doublet\nLi 0.0 0.0 0.0\n", 0, 2).expect("Li atom");
    let bs = ferric_core::basis::bundled("6-31g").expect("6-31g");
    let dfbs_set = ferric_core::basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri");
    let obs = PreparedBasis::new(&mol, &bs).expect("prepared obs");
    let dfbs = PreparedBasis::new(&mol, &dfbs_set).expect("prepared dfbs");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let ctx = ParallelContext::default();
    let uhf = solve_uhf(&ctx, &mol, &obs, &bounds, &UhfConfig::default()).expect("uhf");
    let r = u_ri_mp2(&mol, &obs, &dfbs, op, &uhf, &RiMp2Config::default()).expect("u_ri_mp2");
    (r.mp2_corr, r.total_energy)
}

// ---------------------------------------------------------------------------
// CONTRACT 1: unbudgeted is deterministic to the bit.
// ---------------------------------------------------------------------------

#[test]
fn unbudgeted_rimp2_energy_is_bit_identical_across_repeated_runs() {
    let _s = CleanSlot::acquire();
    let (ca, ta) = run_rimp2();
    let (cb, tb) = run_rimp2();
    assert_eq!(
        ca.to_bits(),
        cb.to_bits(),
        "unbudgeted RI-MP2 correlation energy must be deterministic to the BIT \
         (got {ca:.17e} vs {cb:.17e})"
    );
    assert_eq!(ta.to_bits(), tb.to_bits(), "and so must the total energy");
}

#[test]
fn unbudgeted_u_rimp2_energy_is_bit_identical_across_repeated_runs() {
    let _s = CleanSlot::acquire();
    let (ca, _) = run_u_rimp2();
    let (cb, _) = run_u_rimp2();
    assert_eq!(
        ca.to_bits(),
        cb.to_bits(),
        "unbudgeted U-RI-MP2 must be deterministic to the BIT ({ca:.17e} vs {cb:.17e})"
    );
}

// ---------------------------------------------------------------------------
// CONTRACT 2: an AMPLE pool must not move a single bit.
//
// This is the property that makes the pool safe to install by default: when
// every plane fits, charging it must not change which branch runs and must not
// change any accumulation width. A migrated site that sized a chunk from
// `pool.available_bytes()` rather than the configured budget would fail here,
// because the ledger is non-empty by the time the second plane asks.
// ---------------------------------------------------------------------------

#[test]
fn installing_an_ample_pool_does_not_move_rimp2_one_bit() {
    let _s = CleanSlot::acquire();
    let (unbudgeted, unbudgeted_total) = run_rimp2();

    install_global(MemoryPool::with_capacity_bytes(64_000_000_000));
    let (pooled, pooled_total) = run_rimp2();
    clear_global();

    assert_eq!(
        unbudgeted.to_bits(),
        pooled.to_bits(),
        "an AMPLE pool must not perturb E_corr by even one bit \
         (unbudgeted {unbudgeted:.17e} vs pooled {pooled:.17e}) -- if this fails, \
         charging a plane changed a blocking width, which moves an energy"
    );
    assert_eq!(
        unbudgeted_total.to_bits(),
        pooled_total.to_bits(),
        "and the total energy likewise"
    );
}

#[test]
fn installing_an_ample_pool_does_not_move_u_rimp2_one_bit() {
    let _s = CleanSlot::acquire();
    let (unbudgeted, _) = run_u_rimp2();

    install_global(MemoryPool::with_capacity_bytes(64_000_000_000));
    let (pooled, _) = run_u_rimp2();
    clear_global();

    assert_eq!(
        unbudgeted.to_bits(),
        pooled.to_bits(),
        "an AMPLE pool must not perturb open-shell E_corr by even one bit \
         ({unbudgeted:.17e} vs {pooled:.17e})"
    );
}

// ---------------------------------------------------------------------------
// CONTRACT 3: the pool must be EMPTY when the job is done.
//
// Every reservation this crate takes is RAII. If a guard is leaked (stored in
// a struct that is itself leaked, or `mem::forget`-ed), the ledger only grows
// and the SECOND MP2 in one process is refused against bytes nothing holds.
// This is the "release" half of the pool contract, and it is the half that was
// empty when `MemoryPool` was first written.
// ---------------------------------------------------------------------------

#[test]
fn every_mp2_reservation_is_released_when_the_run_finishes() {
    let _s = CleanSlot::acquire();
    let pool = MemoryPool::with_capacity_bytes(64_000_000_000);
    install_global(pool.clone());
    let _ = run_rimp2();
    let outstanding = pool.outstanding_bytes();
    clear_global();
    assert_eq!(
        outstanding,
        0,
        "RI-MP2 leaked {outstanding} bytes of reservation after returning -- a leaked \
         guard makes the ledger monotone and refuses the next job against bytes \
         nothing holds:\n{}",
        pool.occupancy_report()
    );
    // And the peak must be non-zero, or the charges are decoration: a test
    // that only checks "outstanding == 0" passes trivially against a tree
    // where nothing was ever charged.
    assert!(
        pool.peak_bytes() > 0,
        "the pool recorded a ZERO peak over a full RI-MP2 -- nothing was charged, so \
         CONTRACT 3 is vacuous"
    );
}
