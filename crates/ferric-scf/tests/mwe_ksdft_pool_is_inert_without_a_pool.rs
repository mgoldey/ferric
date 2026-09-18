//! EXACTNESS ANCHOR for wiring the shared memory pool into the KS-DFT path.
//!
//! Written and made to pass BEFORE any ksdft gate was migrated, per
//! `CLAUDE.md`'s Experimental Protocol ("EXACTNESS ANCHOR FIRST").
//!
//! The trivial limit of the pool is "no pool installed". In that limit a
//! KS-DFT run must be BIT-IDENTICAL to the pre-pool tree: every migrated gate
//! must take the same branch, hold an inert reservation, and produce the same
//! energy to the last bit.
//!
//! ## Artifact hypothesis (stated before measuring, per the protocol)
//!
//! If the no-op property is REAL: with no pool installed, `reserve_global`
//! hands back an inert reservation, `try_reserve_global` ADMITS, and the
//! Full-vs-Batched grid decision and the DF in-core/spill decision are taken
//! on exactly the numbers they were taken on before — so two runs of the same
//! input differ by exactly 0 bits.
//!
//! If the implementation is BROKEN in the most likely way — a migrated gate
//! that treats "no pool" as "no memory available" (`try_reserve` returning
//! `None`, or `global_available_bytes()` unwrapped to 0) — then the grid gate
//! would start CHOOSING BATCHING where it previously chose Full, or the DF
//! source would start choosing the spill/recompute backend. Both of those are
//! different code paths with different summation shapes, so the energy would
//! move at ~1e-6 Ha or worse, and `grid_cache_is_full_when_unbudgeted` would
//! flip. Those predictions differ from the real one (0 bits), so this test can
//! distinguish them.
//!
//! The pool is process-global, so these tests serialize on `GLOBAL_LOCK` and
//! clear the slot on entry and exit.

use std::sync::{Mutex, MutexGuard, OnceLock};

use ferric_core::memory::pool::{clear_global, install_global, MemoryPool};
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

/// Clear the process-global pool on entry AND exit so one test's pool can
/// never leak into another's "unbudgeted" assertions.
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

/// Run a PBE/RI-J single point on water/6-31G and return the total energy.
///
/// RI-J is on (`df_j_aux`), so this exercises BOTH migrated planes: the DF
/// 3-index tensor and the KS grid AO cache.
fn run_ksdft() -> f64 {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").expect("water.xyz");
    let bs = ferric_core::basis::bundled("6-31g").expect("6-31g");
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
        .expect("KS-DFT must converge")
        .energy
}

#[test]
fn unbudgeted_ksdft_energy_is_bit_identical_across_repeated_runs() {
    let _s = CleanSlot::acquire();
    // No pool installed: every migrated gate must be inert.
    let a = run_ksdft();
    let b = run_ksdft();
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "unbudgeted KS-DFT must be deterministic to the BIT (got {a:.17e} vs {b:.17e})"
    );
}

#[test]
fn installing_an_ample_pool_does_not_move_the_energy_one_bit() {
    let _s = CleanSlot::acquire();
    // The reference: no pool at all.
    let unbudgeted = run_ksdft();

    // An ample pool: every reservation fits, so every gate must take the
    // SAME branch it took unbudgeted. This is the property that makes the
    // pool safe to install by default -- charging a plane must not change
    // which code path runs when the plane fits.
    install_global(MemoryPool::with_capacity_bytes(64_000_000_000));
    let budgeted = run_ksdft();
    clear_global();

    assert_eq!(
        unbudgeted.to_bits(),
        budgeted.to_bits(),
        "an AMPLE pool must not perturb the energy by even one bit \
         (unbudgeted {unbudgeted:.17e} vs pooled {budgeted:.17e}) -- if this \
         fails, charging a plane changed a blocking width, which moves an energy"
    );
}
