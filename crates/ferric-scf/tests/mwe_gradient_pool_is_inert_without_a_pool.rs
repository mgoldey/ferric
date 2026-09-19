//! EXACTNESS ANCHOR for wiring the shared memory pool into the GRADIENT paths.
//!
//! Written and made to pass BEFORE any gradient gate was migrated, per
//! `CLAUDE.md`'s Experimental Protocol ("EXACTNESS ANCHOR FIRST").
//!
//! The trivial limit of the pool is "no pool installed". In that limit every
//! gradient must be BIT-IDENTICAL to the pre-migration tree: each migrated
//! gate must take the same branch, hold an inert reservation, and produce the
//! same forces to the last bit. A second, stronger limit is "an AMPLE pool":
//! charging a plane that comfortably fits must not change which code path
//! runs, so an ample pool must also move nothing.
//!
//! ## Artifact hypothesis (stated before measuring, per the protocol)
//!
//! If the no-op property is REAL: with no pool installed `reserve_global`
//! hands back an inert reservation, `try_reserve_global` ADMITS, and every
//! width/branch in the gradient path is chosen on exactly the numbers it was
//! chosen on before — so two runs of the same input differ by exactly 0 bits,
//! and the ample-pool run matches them to 0 bits as well.
//!
//! If the implementation is BROKEN in the most likely way — a migrated gate
//! treating "no pool" as "no memory available" (`try_reserve` returning
//! `None`, or `global_available_bytes()` unwrapped to 0) — then the KS-DFT
//! gradient's grid path would start refusing where it did not, and a refusal
//! surfaces as an `Err` (the test panics on `expect`) rather than a moved
//! number. If a gate instead SIZED something from the ledger, the ample-pool
//! gradient would drift from the unbudgeted one while the two unbudgeted runs
//! still matched. Those three outcomes are distinguishable, so this test can
//! tell them apart.
//!
//! The pool is process-global, so these tests serialize on a private lock and
//! clear the slot on entry and exit.

use std::sync::{Mutex, MutexGuard, OnceLock};

use ferric_core::memory::pool::{clear_global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::{rhf_gradient, uhf_gradient};
use ferric_scf::ks_gradient::ks_gradient_closed;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;

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

fn water() -> Molecule {
    Molecule::load_xyz("../../testdata/molecules/water.xyz").expect("water.xyz")
}

/// Assert two gradients agree to the BIT, naming the first offending element.
fn assert_bit_identical(a: &Array2<f64>, b: &Array2<f64>, what: &str) {
    assert_eq!(a.dim(), b.dim(), "{what}: shape changed");
    for ((i, j), &x) in a.indexed_iter() {
        let y = b[(i, j)];
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{what}: element ({i},{j}) moved: {x:.17e} vs {y:.17e} -- a pool \
             charge changed a branch or a width, which moves a number"
        );
    }
}

/// Analytic RHF nuclear gradient on water/6-31G.
fn run_rhf_gradient() -> Array2<f64> {
    let mol = water();
    let bs = ferric_core::basis::bundled("6-31g").expect("6-31g");
    let obs = PreparedBasis::new(&mol, &bs).expect("prepared basis");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let ctx = ParallelContext::default();
    let cfg = RhfConfig::default();
    let scf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).expect("RHF must converge");
    rhf_gradient(&mol, &obs, op, &bounds, &scf, None).expect("RHF gradient")
}

/// Analytic UHF nuclear gradient on a water cation doublet / STO-3G.
fn run_uhf_gradient() -> Array2<f64> {
    let mut mol = water();
    mol.charge = 1;
    mol.multiplicity = 2;
    let bs = ferric_core::basis::bundled("sto-3g").expect("sto-3g");
    let obs = PreparedBasis::new(&mol, &bs).expect("prepared basis");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let ctx = ParallelContext::default();
    let cfg = RhfConfig::default();
    let scf = solve_uhf(&ctx, &mol, &obs, &bounds, &cfg).expect("UHF must converge");
    uhf_gradient(&mol, &obs, op, &bounds, &scf, None).expect("UHF gradient")
}

/// Analytic KS-DFT (PBE) nuclear gradient on water/6-31G.
///
/// This is THE path the migration is aimed at: it builds the AO Hessian
/// (`ddchi`) on the DFT grid, which is the largest single allocation in the
/// whole gradient stack.
fn run_ks_gradient() -> Array2<f64> {
    let mol = water();
    let bs = ferric_core::basis::bundled("6-31g").expect("6-31g");
    let obs = PreparedBasis::new(&mol, &bs).expect("prepared basis");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).expect("schwarz");
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        xc: Some("PBE".into()),
        ..Default::default()
    };
    let scf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).expect("KS-DFT must converge");
    ks_gradient_closed(&mol, &obs, &bs, op, &bounds, "PBE", &scf, None).expect("KS gradient")
}

#[test]
fn unbudgeted_rhf_gradient_is_bit_identical_across_repeated_runs() {
    let _s = CleanSlot::acquire();
    let a = run_rhf_gradient();
    let b = run_rhf_gradient();
    assert_bit_identical(&a, &b, "unbudgeted RHF gradient");
}

#[test]
fn unbudgeted_uhf_gradient_is_bit_identical_across_repeated_runs() {
    let _s = CleanSlot::acquire();
    let a = run_uhf_gradient();
    let b = run_uhf_gradient();
    assert_bit_identical(&a, &b, "unbudgeted UHF gradient");
}

#[test]
fn unbudgeted_ks_gradient_is_bit_identical_across_repeated_runs() {
    let _s = CleanSlot::acquire();
    let a = run_ks_gradient();
    let b = run_ks_gradient();
    assert_bit_identical(&a, &b, "unbudgeted KS-DFT gradient");
}

#[test]
fn an_ample_pool_does_not_move_the_rhf_gradient_one_bit() {
    let _s = CleanSlot::acquire();
    let unbudgeted = run_rhf_gradient();
    install_global(MemoryPool::with_capacity_bytes(64_000_000_000));
    let budgeted = run_rhf_gradient();
    clear_global();
    assert_bit_identical(&unbudgeted, &budgeted, "ample-pool RHF gradient");
}

#[test]
fn an_ample_pool_does_not_move_the_uhf_gradient_one_bit() {
    let _s = CleanSlot::acquire();
    let unbudgeted = run_uhf_gradient();
    install_global(MemoryPool::with_capacity_bytes(64_000_000_000));
    let budgeted = run_uhf_gradient();
    clear_global();
    assert_bit_identical(&unbudgeted, &budgeted, "ample-pool UHF gradient");
}

#[test]
fn an_ample_pool_does_not_move_the_ks_gradient_one_bit() {
    let _s = CleanSlot::acquire();
    let unbudgeted = run_ks_gradient();
    install_global(MemoryPool::with_capacity_bytes(64_000_000_000));
    let budgeted = run_ks_gradient();
    clear_global();
    assert_bit_identical(&unbudgeted, &budgeted, "ample-pool KS-DFT gradient");
}
