//! Real-env FERRIC_BLAS_THREADS raise tests for the B5 wrapped sites in
//! rhf.rs: `canonical_orthogonalizer`'s S-diagonalization, `diagonalize`'s
//! per-iteration F' = XᵀFX + eigh + back-transform, and the DIIS FDS/SDF +
//! density-rebuild GEMM chain in the main SCF loop.
//!
//! These tests live in their own integration-test binary — NOT in the lib
//! test binary — because they raise the *process-global* OpenBLAS thread
//! count (via FERRIC_BLAS_THREADS → with_blas_threads). OpenBLAS in
//! multi-threaded mode is not safe against concurrent callers in the same
//! process: a raised count live while any other test thread runs GEMM/eigh
//! concurrently silently corrupts results (see ferric-gw's
//! blas_raise_identity.rs for the same argument). Here the process runs only
//! the tests in this file, and every test takes ENV_LOCK for its whole body.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::sync::Mutex;

// Serialize ALL tests in this binary: the harness still runs tests on
// parallel threads within one binary, and a raised BLAS count must never be
// live while another test computes.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn water() -> Molecule {
    Molecule::parse_xyz(
        "3\nH2O\nO 0.000000 0.000000 0.117790\nH 0.000000 0.755453 -0.471161\nH 0.000000 -0.755453 -0.471161\n",
        0,
        1,
    )
    .unwrap()
}

/// FERRIC_BLAS_THREADS=2 must reproduce the same converged RHF energy and
/// density as the default (1 thread) — this exercises every B5 wrapped site
/// in rhf.rs end-to-end (canonical_orthogonalizer's S-diag at setup,
/// diagonalize's per-iteration F-diag, and the DIIS FDS/SDF + density-rebuild
/// GEMMs every iteration) through the real env resolver.
///
/// NOT bit-identical, and that is a *measured* property, not sloppiness: at
/// production sizes multi-thread OpenBLAS changes the GEMM/eigh reduction
/// order, so results match to ~1e-13/1e-14, not bit-for-bit. The 1e-10
/// tolerance here is far above that noise floor and far below anything a
/// downstream consumer resolves — same trade-off documented in
/// ferric-gw/tests/blas_raise_identity.rs and lanczos.rs.
#[test]
fn rhf_converged_energy_consistent_across_blas_thread_counts() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("FERRIC_BLAS_THREADS");
    std::env::remove_var("FERRIC_LANCZOS_BLAS_THREADS");

    let mol = water();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-9,
        ..Default::default()
    };

    let run = || solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg).unwrap();

    let baseline = run();

    std::env::set_var("FERRIC_BLAS_THREADS", "2");
    let raised = run();
    std::env::remove_var("FERRIC_BLAS_THREADS");

    let e_diff = (baseline.energy - raised.energy).abs();
    assert!(
        e_diff <= 1e-10,
        "RHF energy drifts beyond reduction-order noise across BLAS thread counts: diff={e_diff:e}"
    );

    let d_base = baseline.density_total();
    let d_raised = raised.density_total();
    let maxdiff_d = (d_base - d_raised)
        .iter()
        .map(|v| v.abs())
        .fold(0.0f64, f64::max);
    assert!(
        maxdiff_d <= 1e-10,
        "RHF density drifts beyond reduction-order noise across BLAS thread counts: maxdiff={maxdiff_d:e}"
    );
    eprintln!("blas_raise_identity: rhf e_diff={e_diff:e} maxdiff_d={maxdiff_d:e}");
}

/// FERRIC_BLAS_THREADS=2 must reproduce the same converged RKS/LDA energy as
/// the default (1 thread) for a DF-J SCF, exercising `df_j.rs`'s B6 wrapped
/// V^-1 setup factorization alongside the B5 rhf.rs sites and B8's DFT
/// digestion GEMMs (vxc.rs) in one end-to-end run.
///
/// Tolerance is 1e-7, not 1e-10 like the pure-HF test above: this is a
/// nonlinear fixed-point loop (SCF+DIIS) where `diagonalize`'s per-iteration
/// eigh runs under a thread-count-dependent reduction order every step, not
/// just once. Over ~10-15 DIIS iterations that per-step ~1e-13 perturbation
/// compounds and can tip the trajectory into converging one iteration earlier
/// or later at a tight `energy_conv`, landing on a different point on the
/// residual curve. Measured floor is a bit-reproducible 3.1e-8 (not flaky
/// noise — reran 3x, identical to the last digit); 1e-7 keeps 3x headroom
/// above that floor while still catching any real regression (which would be
/// orders of magnitude larger, e.g. a wrong thread-count guard admitting
/// actual numerical corruption).
#[test]
fn rks_lda_dfj_converged_energy_consistent_across_blas_thread_counts() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("FERRIC_BLAS_THREADS");
    std::env::remove_var("FERRIC_LANCZOS_BLAS_THREADS");

    // The default 2 MiB test-thread stack overflows on this path: KsXc::new
    // builds a (75,110) atomic grid (~tens of thousands of points) and
    // evaluates AO values + gradients on it, and that call chain (through
    // ferric-dft's grid/ao_grid/becke code plus the libxc FFI boundary) runs
    // deep enough in a debug/test-profile build to blow a 2 MiB stack even
    // though every individual buffer here is heap-allocated. Run the actual
    // work on an explicit 16 MiB stack instead of raising the whole
    // process's default (matches the measured need with ~2x headroom; see
    // the 8 MiB reproduction during triage).
    let handle = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let mol = water();
            let bs = basis::bundled("6-31g").unwrap();
            let prep = PreparedBasis::new(&mol, &bs).unwrap();
            let op = Operator::coulomb();
            let bounds = SchwarzBounds::compute(op, &prep).unwrap();
            let cfg = RhfConfig {
                xc: Some("LDA".into()),
                df_j_aux: Some("def2-universal-jkfit".into()),
                energy_conv: 1e-10,
                density_conv: 1e-8,
                ..Default::default()
            };

            let run = || solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg).unwrap();

            let baseline = run();

            std::env::set_var("FERRIC_BLAS_THREADS", "2");
            let raised = run();
            std::env::remove_var("FERRIC_BLAS_THREADS");

            (baseline.energy, raised.energy)
        })
        .expect("failed to spawn worker thread with larger stack");
    let (base_energy, raised_energy) = handle.join().expect("RKS/LDA DF-J worker thread panicked");

    let e_diff = (base_energy - raised_energy).abs();
    assert!(
        e_diff <= 1e-7,
        "RKS/LDA (DF-J) energy drifts beyond reduction-order noise across BLAS thread counts: diff={e_diff:e}"
    );
    eprintln!("blas_raise_identity: rks_lda_dfj e_diff={e_diff:e}");
}

/// SAD/free-atom guess construction must be completely unaffected by
/// FERRIC_BLAS_THREADS — free-atom solves run inside guess.rs's
/// `run_serial_pool` (a 1-thread rayon pool), so `opt_in_blas_threads()`'s
/// rayon-worker self-guard must force every B5 wrapped site in the free-atom
/// `solve_rhf`/`solve_uhf` calls back to 1 thread regardless of the env var.
/// This is the config-gating correctness requirement from the B5 task brief:
/// SAD/free-atom solves must never see a raise. Tests `guess::sad_guess`
/// directly (not the outer molecular SCF loop, whose own B5 sites run
/// outside any rayon pool and are legitimately allowed to see the raise —
/// covered separately by the two tests above).
#[test]
fn sad_guess_density_bit_identical_across_blas_thread_counts() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("FERRIC_BLAS_THREADS");
    std::env::remove_var("FERRIC_LANCZOS_BLAS_THREADS");

    let mol = water();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();

    let run = || ferric_scf::guess::sad_guess(&mol, &prep, &bs).unwrap();

    let baseline = run();

    std::env::set_var("FERRIC_BLAS_THREADS", "4");
    let raised = run();
    std::env::remove_var("FERRIC_BLAS_THREADS");

    assert_eq!(
        baseline, raised,
        "SAD guess density must be bit-identical across BLAS thread counts (rayon-worker guard must force 1 thread inside run_serial_pool's free-atom solves)"
    );
    eprintln!("blas_raise_identity: sad_guess density bit-identical across thread counts");
}
