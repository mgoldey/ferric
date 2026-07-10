//! Real-env FERRIC_BLAS_THREADS raise tests for the B3 wrapped sites
//! (mo_b metric factorization + MO-transform/dressing GEMMs, w_pdep inv).
//!
//! These tests live in their own integration-test binary — NOT in the lib
//! test binary — because they raise the *process-global* OpenBLAS thread
//! count (via FERRIC_BLAS_THREADS → with_blas_threads). OpenBLAS in
//! multi-threaded mode is not safe against concurrent callers in the same
//! process: raising the count while any other test thread runs GEMM/Cholesky
//! work concurrently was observed to silently corrupt results (5%-level
//! garbage, not 1-ulp drift) when this test lived in the parallel lib test
//! binary. Here the process runs only the tests in this file, and every test
//! takes ENV_LOCK for its whole body, so no BLAS call ever runs concurrently
//! with a raised count.

use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::basis;
use ferric_gw::{mo_b, w_pdep};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::array;
use std::sync::Mutex;

// Serialize ALL tests in this binary: the harness still runs tests on
// parallel threads within one binary, and a raised BLAS count must never be
// live while another test computes.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// FERRIC_BLAS_THREADS=2 must reproduce the dressed MoB tensor of the default
/// (1 thread) across the wrapped sites in mo_b.rs (metric Cholesky +
/// triangular solve, per-aux-block MO-transform GEMMs, pair-column panel
/// dressing GEMM) to reduction-order precision.
///
/// NOT bit-identical, and that is a *measured* property, not sloppiness: at
/// water/cc-pVDZ-RI sizes (naux ≈ 116) OpenBLAS at 2 threads splits the
/// GEMM/potrf reduction, changing the floating-point summation order — the
/// first differing elements sit at the ~1e-15 level while the leading
/// elements are exactly equal. This is the same documented trade-off the
/// FERRIC_LANCZOS_BLAS_THREADS precedent carries (lanczos.rs doc, point 2:
/// "Multi-threaded OpenBLAS GEMM/eigh changes the reduction order"): the
/// DEFAULT (1 thread) is bit-deterministic; opting in trades bit-level
/// reproducibility for speed. The tolerance here (1e-12 absolute on O(0.5)
/// entries) is ~100× above reduction noise and ~orders below anything any
/// downstream consumer resolves.
#[test]
fn build_full_b_consistent_across_blas_thread_counts() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("FERRIC_BLAS_THREADS");
    std::env::remove_var("FERRIC_LANCZOS_BLAS_THREADS");

    let mol = Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    // RHF runs BEFORE any raise window (env unset here), so its rayon-parallel
    // Fock builds never see a raised BLAS count.
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

    let build = || mo_b::build_full_b(&mol, &obs, &dfbs, op, &rhf, 1).unwrap();

    let baseline = build();

    std::env::set_var("FERRIC_BLAS_THREADS", "2");
    let raised = build();
    std::env::remove_var("FERRIC_BLAS_THREADS");

    let maxdiff_b = (&baseline.b_full - &raised.b_full)
        .iter()
        .map(|v| v.abs())
        .fold(0.0f64, f64::max);
    assert!(
        maxdiff_b <= 1e-12,
        "b_full drifts beyond reduction-order noise across BLAS thread counts: maxdiff={maxdiff_b:e}"
    );
    let maxdiff_v = (&baseline.v_inv_sqrt - &raised.v_inv_sqrt)
        .iter()
        .map(|v| v.abs())
        .fold(0.0f64, f64::max);
    assert!(
        maxdiff_v <= 1e-12,
        "v_inv_sqrt drifts beyond reduction-order noise across BLAS thread counts: maxdiff={maxdiff_v:e}"
    );
    eprintln!("blas_raise_identity: maxdiff_b={maxdiff_b:e} maxdiff_v={maxdiff_v:e}");
}

/// FERRIC_BLAS_THREADS=2 must reproduce the exact same redressed
/// eigenpotentials as the default (1 thread) — the wrapped inv() + GEMM in
/// w_pdep::redress_eigenpotentials must be bit-identical regardless of BLAS
/// thread count.
#[test]
fn redress_identical_across_blas_thread_counts() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("FERRIC_BLAS_THREADS");
    std::env::remove_var("FERRIC_LANCZOS_BLAS_THREADS");

    // Well-conditioned lower-triangular v_inv_sqrt (as solve_triangular would
    // produce) and an arbitrary phys-basis eigenpotential matrix.
    let v_inv_sqrt = array![[1.0f64, 0.0, 0.0], [0.2, 0.9, 0.0], [0.1, -0.3, 1.1]];
    let eigenpotentials_phys = array![[0.5f64, 1.0], [-0.2, 0.3], [0.7, -0.4]];

    let baseline = w_pdep::redress_eigenpotentials(&v_inv_sqrt, &eigenpotentials_phys).unwrap();

    std::env::set_var("FERRIC_BLAS_THREADS", "2");
    let raised = w_pdep::redress_eigenpotentials(&v_inv_sqrt, &eigenpotentials_phys).unwrap();
    std::env::remove_var("FERRIC_BLAS_THREADS");

    assert_eq!(
        baseline, raised,
        "redressed eigenpotentials differ across BLAS thread counts"
    );
}
