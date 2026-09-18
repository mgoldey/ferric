//! EXACTNESS ANCHOR for wiring the shared memory pool into the ferric-cc
//! coupled-cluster drivers.
//!
//! Written and made to pass BEFORE any ferric-cc gate was migrated, per
//! `CLAUDE.md`'s Experimental Protocol ("EXACTNESS ANCHOR FIRST"), and
//! mirroring `ferric-scf/tests/mwe_ksdft_pool_is_inert_without_a_pool.rs`.
//!
//! The trivial limit of the pool is "no pool installed". In that limit a CC
//! run must be BIT-IDENTICAL to the pre-pool tree: every migrated gate must
//! take the same branch, hold an inert reservation, and produce the same
//! correlation energy to the last bit.
//!
//! ## Artifact hypothesis (stated before measuring, per the protocol)
//!
//! If the no-op property is REAL: with no pool installed, `MemoryPlan::commit`
//! degenerates to `check()` plus an inert guard, `try_reserve_global` ADMITS,
//! and `global_available_bytes()` returns `None` so the (T) band width falls
//! back to exactly the byte budget it used before. Two runs of the same input
//! then differ by exactly 0 bits.
//!
//! If the implementation is BROKEN in the most likely way -- a migrated gate
//! that treats "no pool" as "no memory available" (`try_reserve` returning
//! `None`, or `global_available_bytes()` unwrapped to 0) -- then:
//!   * the CCSD/CCD/closed-shell drivers would ERROR instead of running, and
//!   * the (T) triple band would collapse to width 1.
//!
//! The first is loud. The second is the dangerous one, and it is exactly why
//! `mwe_t_band_width_is_not_an_energy_knob.rs` exists alongside this file: the
//! band width must be a PERFORMANCE knob, never an energy knob, so a
//! pool-derived width cannot smuggle in the grid-batch-width defect (a
//! budget-derived width that read live RSS and moved a KS-DFT energy in the
//! 8th decimal).
//!
//! The pool is process-global, so these tests serialize on a private lock and
//! clear the slot on entry and exit.

use std::sync::{Mutex, MutexGuard, OnceLock};

use ferric_cc::{ccd::ccd, ccsd::ccsd, ccsd_closed_shell::ccsd_closed_shell, CcConfig};
use ferric_core::basis;
use ferric_core::memory::pool::{clear_global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::result::ScfResult;
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

struct Fixture {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    rhf: ScfResult,
}

/// H2O / STO-3G with a cc-pVDZ-RI auxiliary basis: the smallest fixture whose
/// VVVV block, DIIS ring and (T) triple blocks are all non-degenerate.
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
        ..Default::default()
    }
}

fn run_ccsd(f: &Fixture) -> f64 {
    ccsd(&f.mol, &f.obs, &f.dfbs, Operator::coulomb(), &f.rhf, &cfg())
        .expect("CCSD must converge")
        .correlation_energy
}

fn run_ccd(f: &Fixture) -> f64 {
    ccd(&f.mol, &f.obs, &f.dfbs, Operator::coulomb(), &f.rhf, &cfg())
        .expect("CCD must converge")
        .correlation_energy
}

fn run_ccsd_cs(f: &Fixture) -> f64 {
    ccsd_closed_shell(&f.mol, &f.obs, &f.dfbs, Operator::coulomb(), &f.rhf, &cfg())
        .expect("closed-shell CCSD must converge")
        .correlation_energy
}

#[test]
fn unbudgeted_cc_energies_are_bit_identical_across_repeated_runs() {
    let _s = CleanSlot::acquire();
    let f = fixture();
    for (name, run) in [
        ("CCSD", &run_ccsd as &dyn Fn(&Fixture) -> f64),
        ("CCD", &run_ccd),
        ("closed-shell CCSD", &run_ccsd_cs),
    ] {
        let a = run(&f);
        let b = run(&f);
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "unbudgeted {name} must be deterministic to the BIT \
             (got {a:.17e} vs {b:.17e})"
        );
    }
}

/// The trivial limit must mean "do exactly what you did before", not
/// "assume nothing is available".
///
/// # Why an ENERGY test cannot catch this
///
/// Mutation M5 of this migration replaced the (T) band's two-branch sizing
/// with `global_available_bytes().unwrap_or(0)`, i.e. "no pool installed means
/// zero memory free". That collapses the band to width 1 on every unbudgeted
/// run -- a serial (T) step for no reason.
///
/// It SURVIVED every bit-identity test in this crate, and correctly so: the
/// companion file `mwe_t_band_width_is_not_an_energy_knob.rs` proves the width
/// does not move `et`, so by construction no energy assertion can see a width
/// regression. Catching it requires asserting on the WIDTH.
///
/// That is the shape the brief warns about ("`global_available_bytes()`
/// returning None must skip the sizing branch"), and it is only observable
/// through the driver's own width function.
#[test]
fn the_unbudgeted_band_width_is_the_budget_width_not_a_collapsed_one() {
    let _s = CleanSlot::acquire();
    // No pool installed. The width the driver will use must be the one the
    // resolved byte budget funds -- many triples wide at an ample budget --
    // not the width-1 floor that an `unwrap_or(0)` would produce.
    assert!(
        ferric_core::memory::pool::global_available_bytes().is_none(),
        "this test is about the NO-POOL limit; a pool leaked in from a sibling"
    );

    // H2O/STO-3G shapes: no2 = 10, nv2 = 4 spin-orbital; no = 5, nv = 2.
    let so = ferric_cc::ccsd_t::t_band_width(10, 4, 8_000_000_000);
    let cs = ferric_cc::ccsd_t_closed_shell::t_band_width(5, 2, 8_000_000_000);
    assert!(
        so > 1,
        "unbudgeted spin-orbital (T) collapsed to a width-1 band ({so}) at an \
         8 GB budget -- the None branch is treating 'unbudgeted' as 'nothing \
         free' instead of 'behave exactly as before'"
    );
    assert!(
        cs > 1,
        "unbudgeted closed-shell (T) collapsed to a width-1 band ({cs}) at an \
         8 GB budget"
    );
}

#[test]
fn installing_an_ample_pool_does_not_move_a_cc_energy_one_bit() {
    let _s = CleanSlot::acquire();
    let f = fixture();

    // The reference: no pool at all.
    let unbudgeted = [run_ccsd(&f), run_ccd(&f), run_ccsd_cs(&f)];

    // An ample pool: every reservation fits, so every gate must take the SAME
    // branch it took unbudgeted. This is the property that makes the pool safe
    // to install by default -- charging a plane must not change which code
    // path runs when the plane fits.
    install_global(MemoryPool::with_capacity_bytes(64_000_000_000));
    let budgeted = [run_ccsd(&f), run_ccd(&f), run_ccsd_cs(&f)];
    clear_global();

    for (i, name) in ["CCSD", "CCD", "closed-shell CCSD"].iter().enumerate() {
        assert_eq!(
            unbudgeted[i].to_bits(),
            budgeted[i].to_bits(),
            "an AMPLE pool must not perturb {name} by even one bit \
             (unbudgeted {:.17e} vs pooled {:.17e}) -- if this fails, charging \
             a plane changed a blocking width, which moves an energy",
            unbudgeted[i],
            budgeted[i],
        );
    }
}
