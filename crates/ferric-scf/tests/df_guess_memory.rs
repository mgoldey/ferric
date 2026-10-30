//! The DF-guess pre-stage only chooses a starting density, so running short of
//! memory for it must not end a run that could otherwise proceed.
//!
//! Own test binary: these install the PROCESS-GLOBAL pool (see
//! `ferric-integrals/tests/mwe_three_index_pool_charges.rs` for why that must
//! not share a process with unit tests), and the mutex keeps them apart.
//!
//! EXACTNESS ANCHOR: all routes must reach the same fixed point. Stages differ
//! only in where they start, so the energy is compared to the SCF tolerance
//! rather than bitwise, and the ample run is pinned bitwise against no pool.

use ferric_core::basis;
use ferric_core::memory::pool::{clear_global, install_global, MemoryPool};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ladder::{solve_rhf_with_df_guess, DfGuessResult, DF_GUESS_DEFAULT_AUX};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use std::sync::{Mutex, MutexGuard};

static SLOT: Mutex<()> = Mutex::new(());

fn slot() -> MutexGuard<'static, ()> {
    let g = SLOT.lock().unwrap_or_else(|e| e.into_inner());
    clear_global();
    g
}

fn water() -> (Molecule, PreparedBasis) {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    (mol, prep)
}

/// Bytes of one unpacked DF 3-index tensor for this system.
fn tensor_bytes(mol: &Molecule, prep: &PreparedBasis) -> usize {
    let aux = PreparedBasis::new(mol, &basis::bundled(DF_GUESS_DEFAULT_AUX).unwrap()).unwrap();
    aux.nbasis() * prep.nbasis() * prep.nbasis() * 8
}

fn run(budget: usize, aux: Option<&str>) -> Result<DfGuessResult, FerricError> {
    let (mol, prep) = water();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        three_index_budget_bytes: budget,
        ..Default::default()
    };
    solve_rhf_with_df_guess(
        &ParallelContext::default(),
        &mol,
        &prep,
        op,
        &bounds,
        &cfg,
        aux,
    )
}

fn same_energy(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-8, "{a:.12} vs {b:.12}: {:.3e}", a - b);
}

/// FAILURE 1, at its smallest: the tensor fits the budget once but not twice,
/// which is what C20/6-31G hit at 2 GiB. The guess must still RUN, on a spilled
/// dressed tensor; falling back would also converge, so it is asserted against.
#[test]
fn the_budget_band_runs_the_guess_on_a_spilled_tensor() {
    let _s = slot();
    let (mol, prep) = water();
    let needed = tensor_bytes(&mol, &prep);
    let ample = run(needed * 4, None).expect("ample");
    assert!(ample.fallback.is_none() && ample.df_iterations > 0);

    let budget = needed + needed / 2;
    let pool = MemoryPool::with_capacity_bytes(budget);
    install_global(pool.clone());
    let got = run(budget, None).expect("the band must not refuse the run");
    assert!(
        pool.peak_bytes() >= needed,
        "the raw tensor must have charged"
    );
    assert_eq!(
        pool.outstanding_bytes(),
        0,
        "stage charges must be released"
    );
    clear_global();

    assert!(got.result.converged);
    assert!(
        got.fallback.is_none(),
        "guess fell back: {:?}",
        got.fallback
    );
    assert!(
        got.df_iterations > 0,
        "the DF pre-stage must really have run"
    );
    same_energy(got.result.energy, ample.result.energy);
}

/// A guess that genuinely cannot get its memory gives way, says so in the
/// result, and reaches the same energy. Reachable through the pool: a pool too
/// small for even the raw tensor, under a budget that admits it in core.
#[test]
fn a_memory_failure_of_the_guess_falls_back_to_the_same_energy() {
    let _s = slot();
    let (mol, prep) = water();
    let needed = tensor_bytes(&mol, &prep);
    let ample = run(needed * 4, None).expect("ample");

    let pool = MemoryPool::with_capacity_bytes(needed / 2);
    install_global(pool.clone());
    let got = run(needed * 4, None).expect("a guess's memory failure must not end the run");
    assert_eq!(pool.outstanding_bytes(), 0, "the failed stage must release");
    clear_global();

    let reason = got.fallback.expect("must report the fallback");
    assert!(reason.contains("memory pool exhausted"), "{reason}");
    assert_eq!(got.df_iterations, 0, "must record that no pre-stage ran");
    assert!(!got.df_converged && got.df_energy.is_nan());
    assert!(got.result.converged);
    same_energy(got.result.energy, ample.result.energy);
}

/// Any other failure of the guess is a real error: a bad aux name must not be
/// papered over by quietly running without the stage it asked for.
#[test]
fn a_non_memory_failure_of_the_guess_still_propagates() {
    let _s = slot();
    let err = run(0, Some("no-such-aux-basis")).expect_err("must propagate");
    assert!(
        !matches!(err, FerricError::MemoryExhausted(_)),
        "wrong classification: {err}"
    );
    assert!(err.to_string().contains("no-such-aux-basis"), "{err}");
}

/// A run where the guess works takes the identical path whether or not a pool
/// is watching it.
#[test]
fn a_run_where_the_guess_works_is_bit_identical_under_an_ample_pool() {
    let _s = slot();
    let (mol, prep) = water();
    let needed = tensor_bytes(&mol, &prep);
    let free_run = run(needed * 4, None).expect("no pool");
    install_global(MemoryPool::with_capacity_bytes(needed * 4));
    let pooled = run(needed * 4, None).expect("ample pool");
    clear_global();
    assert_eq!(
        free_run.result.energy.to_bits(),
        pooled.result.energy.to_bits()
    );
    assert_eq!(free_run.result.iterations, pooled.result.iterations);
    assert_eq!(free_run.df_iterations, pooled.df_iterations);
    assert_eq!(free_run.df_energy.to_bits(), pooled.df_energy.to_bits());
}

/// Under MPI a memory failure must stay a hard error: one rank falling back while
/// the others sit in DF collectives would hang the job.
#[cfg(not(feature = "mpi"))]
#[test]
fn under_mpi_a_memory_failure_of_the_guess_stays_an_error() {
    let _s = slot();
    let (mol, prep) = water();
    let needed = tensor_bytes(&mol, &prep);
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        three_index_budget_bytes: needed * 4,
        ..Default::default()
    };
    // Two ranks' worth of band, on a pool too small for even one raw tensor.
    let ctx = ParallelContext { rank: 0, size: 2 };
    install_global(MemoryPool::with_capacity_bytes(needed / 8));
    let got = solve_rhf_with_df_guess(&ctx, &mol, &prep, op, &bounds, &cfg, None);
    clear_global();
    assert!(
        matches!(got, Err(FerricError::MemoryExhausted(_))),
        "must not fall back alone under MPI: {:?}",
        got.map(|r| r.fallback)
    );
}
