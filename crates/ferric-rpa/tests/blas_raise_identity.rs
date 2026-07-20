//! Real-env FERRIC_BLAS_THREADS raise tests for the B4 wrapped site
//! (davidson.rs whole-iteration dense ops) and the Lanczos umbrella fallback.
//!
//! These tests live in their own integration-test binary — NOT in the lib
//! test binary — because they raise the *process-global* OpenBLAS thread
//! count (via FERRIC_BLAS_THREADS → with_blas_threads). OpenBLAS in
//! multi-threaded mode is not safe against concurrent callers in the same
//! process: a raised count live while any other test thread runs GEMM/eigh
//! concurrently silently corrupts results. Here the process runs only the
//! tests in this file, and every test takes ENV_LOCK for its whole body.

use ferric_rpa::davidson::run_davidson_static;
use ferric_rpa::lanczos::run_lanczos_seeded;
use ndarray::{array, Array2};
use std::sync::Mutex;

// Serialize ALL tests in this binary: the harness still runs tests on
// parallel threads within one binary, and a raised BLAS count must never be
// live while another test computes.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn toy_dielectric(v_mat: &Array2<f64>, _omega: f64) -> Array2<f64> {
    let fixed = array![[2.0f64, 0.5], [0.5, 3.0]];
    v_mat.t().dot(&fixed.dot(v_mat))
}

/// FERRIC_BLAS_THREADS=2 (umbrella var; Lanczos-specific var unset) must
/// reproduce the exact same Davidson eigenpairs as the default (1 thread) —
/// this exercises the B4 wrapped site (run_davidson_seeded's
/// with_blas_threads scope) end-to-end through the real env resolver.
/// Bit-identical is expected HERE because the 2×2 problem is far below
/// OpenBLAS's threading threshold (it stays single-threaded regardless of
/// the count). At production sizes multi-threaded BLAS changes the reduction
/// order (~1e-15 drift) — see ferric-gw/tests/blas_raise_identity.rs, which
/// measures that, and the lanczos.rs doc (reproducibility point 2).
#[test]
fn davidson_identical_across_blas_thread_counts() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("FERRIC_BLAS_THREADS");
    std::env::remove_var("FERRIC_LANCZOS_BLAS_THREADS");

    let run = || run_davidson_static(2, toy_dielectric, 1e-10, 20, 2, false).unwrap();

    let baseline = run();

    std::env::set_var("FERRIC_BLAS_THREADS", "2");
    let raised = run();
    std::env::remove_var("FERRIC_BLAS_THREADS");

    assert_eq!(baseline.eigenvalues, raised.eigenvalues);
    assert_eq!(baseline.eigenvectors, raised.eigenvectors);
}

/// Same check for the Lanczos path via the umbrella var (its own var unset,
/// exercising the new FERRIC_BLAS_THREADS fallback in lanczos_blas_threads).
#[test]
fn lanczos_identical_across_blas_thread_counts_via_umbrella() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("FERRIC_BLAS_THREADS");
    std::env::remove_var("FERRIC_LANCZOS_BLAS_THREADS");

    let a = array![[2.0f64, 0.5, 0.0], [0.5, 3.0, 0.1], [0.0, 0.1, 1.5]];
    let matvec = |v: &Array2<f64>| a.dot(v);
    let seed = Array2::<f64>::eye(3);

    let run = || run_lanczos_seeded(seed.clone(), matvec, 3, 50, 1e-10, false).unwrap();

    let baseline = run();

    std::env::set_var("FERRIC_BLAS_THREADS", "2");
    let raised = run();
    std::env::remove_var("FERRIC_BLAS_THREADS");

    assert_eq!(baseline.eigenvalues, raised.eigenvalues);
    assert_eq!(baseline.eigenvectors, raised.eigenvectors);
}
