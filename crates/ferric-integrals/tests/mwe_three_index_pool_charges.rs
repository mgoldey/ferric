//! How the DF 3-index planes charge the shared memory pool.
//!
//! # Why these live in their OWN test binary
//!
//! They install a PROCESS-GLOBAL pool. Cargo runs the unit tests in
//! `three_index_source.rs` concurrently in a single process, so a pool
//! installed by one of them is visible to every sibling that happens to be
//! running at that instant -- and a sibling that legitimately passes
//! `budget_bytes = usize::MAX` is then refused by a 100 kB pool it never asked
//! for. That was observed as 8 spurious failures under the default parallel
//! runner, all of which passed under `--test-threads=1`.
//!
//! A serializing mutex does NOT fix it: it orders the pool-installing tests
//! against each other, but the siblings run outside that lock. A separate
//! integration binary is its own process, which removes the shared slot
//! entirely.
//!
//! This is the flip side of the usual warning -- running tests together can
//! make a killed mutation look like a survivor, and it can equally
//! manufacture failures. Either way the fix is to own the global state
//! exclusively, and here that means a separate process.

use ferric_core::basis;
use ferric_core::memory::pool::{clear_global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::ThreeIndexSource;

/// Serializes the tests in THIS binary, which all install the process-global
/// pool, and clears the slot on entry and exit.
///
/// The separate binary stops the leak reaching other test files; this stops
/// these two leaking into each other. Both are needed: with only the separate
/// binary, mutation M7 (hard-charge the spill scratch) SURVIVED when the two
/// ran concurrently and was killed when either ran alone -- the exact
/// "multiple --test files at once can make a killed mutation look like a
/// survivor" trap, reproduced inside a single file.
struct GlobalPoolGuard(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);

impl GlobalPoolGuard {
    fn acquire() -> Self {
        static L: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        let g = match L.get_or_init(|| std::sync::Mutex::new(())).lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        clear_global();
        Self(g)
    }
}

impl Drop for GlobalPoolGuard {
    fn drop(&mut self) {
        clear_global();
    }
}

fn water() -> Molecule {
    Molecule::load_xyz("../../testdata/molecules/water.xyz").expect("water.xyz")
}

fn bases() -> (Molecule, PreparedBasis, PreparedBasis) {
    let mol = water();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").expect("cc-pvdz"))
        .expect("obs");
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").expect("cc-pvdz-ri"))
        .expect("dfbs");
    (mol, obs, dfbs)
}

/// A SPILLED source must charge the pool only for what it holds RESIDENT (one
/// read-back scratch block), never for the full band that lives on disk.
///
/// "An over-estimating guard is also a bug": charging the whole band for a
/// tensor that is on disk would refuse jobs the spill path exists to make
/// possible -- the spill is chosen precisely BECAUSE the band does not fit, so
/// charging the band would make every spill self-refusing.
///
/// This caught mutation M6, which every value-asserting test in this crate
/// survived (over-charging changes only admission, never a number).
#[test]
fn a_spilled_source_charges_only_its_resident_block_not_the_whole_band() {
    let _slot = GlobalPoolGuard::acquire();
    let (_mol, obs, dfbs) = bases();
    let op = Operator::coulomb();
    let naux = dfbs.nbasis();
    let nao = obs.nbasis();
    let full_band = naux * nao * nao * 8;

    // A budget that forces a spill, and a POOL that can hold several scratch
    // blocks but NOT the full band. If the spill path charged the band, this
    // build would be refused.
    let tiny_budget = nao * nao * 8 * 3;
    let pool = MemoryPool::with_capacity_bytes(full_band / 2);
    clear_global();
    install_global(pool.clone());

    let src = ThreeIndexSource::build(op, &obs, &dfbs, tiny_budget)
        .expect("a spilled source must NOT be refused by a pool that cannot hold the band");
    assert!(!src.is_spilled_for_test() || !src.is_incore(), "budget must have spilled");
    assert!(!src.is_incore(), "tiny budget must have spilled");

    let held = pool.outstanding_bytes();
    assert!(
        held < full_band / 2,
        "a spilled source must never charge the {full_band}-byte band that lives \
         on disk (charged {held}) -- over-charging makes every spill self-refusing"
    );
    // The charge is SOFT: a spill has no fallback left and its block size is
    // fixed by numerics, so an over-budget scratch block is recorded if it
    // fits and skipped if it does not -- never a refusal. What must never
    // happen is the build failing, which the `expect` above pins.

    drop(src);
    assert_eq!(pool.outstanding_bytes(), 0, "drop must credit the block back");
    clear_global();
}

/// TWO co-resident spilled sources must not refuse each other.
///
/// REGRESSION for an over-enforcement defect found by the end-to-end
/// acceptance run, not by any unit test. `DfK::from_raw` borrows the raw
/// source (its scratch block live) while `build_dressed*` allocates the
/// dressed source's own scratch block, and `spill_block_naux_for` /
/// `block_naux_for` size each block against the WHOLE budget -- so the honest
/// co-resident total is ~2x budget whenever anything spills.
///
/// Hard-charging both refused the 27-atom terpinyl cation at
/// `budget_gb = 0.10`, a job the pre-migration tree completes:
///
/// ```text
/// error: memory pool exhausted: "DF 3-index dressed B[P,mn] spill scratch"
///        needs 0.107 GB but only 0.054 GB of the 0.107 GB pool is free
/// ```
///
/// Refusing a job that would have finished is as much a bug as admitting one
/// that OOMs, and the only way to make it fit -- shrinking `block_naux` --
/// would move the SCF energy. Hence the spill charges are soft. This test
/// fails if anyone makes them hard again (mutation M7).
#[test]
fn two_co_resident_spilled_sources_do_not_refuse_each_other() {
    let _slot = GlobalPoolGuard::acquire();
    let (_mol, obs, dfbs) = bases();
    let op = Operator::coulomb();
    let naux = dfbs.nbasis();
    let nao = obs.nbasis();

    // A budget that forces a spill, and a pool of the SAME size -- the
    // production shape, since the CLI installs the pool at the budget. One
    // scratch block is sized to ~that budget, so a second co-resident one
    // cannot possibly fit; it must be skipped, not refused.
    let budget = nao * nao * 8 * 3;
    clear_global();
    install_global(MemoryPool::with_capacity_bytes(budget));

    let mut raw =
        ThreeIndexSource::build(op, &obs, &dfbs, budget).expect("raw spill must not be refused");
    assert!(!raw.is_incore(), "budget must have forced a spill");

    // Dress it while `raw` is still alive -- exactly DfK::from_raw's shape.
    // Before the fix this returned `memory pool exhausted`.
    let m = ndarray::Array2::<f64>::eye(naux);
    let dressed = ThreeIndexSource::build_dressed(&mut raw, &m, budget)
        .expect("a co-resident dressed spill must NOT be refused by the pool");
    assert!(!dressed.is_incore(), "dressed must have spilled too");

    drop(dressed);
    drop(raw);
    clear_global();
}
