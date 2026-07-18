//! Real-env FERRIC_BLAS_THREADS raise tests for the B6/B7 wrapped sites:
//! `rimp2.rs`'s metric factorizations (`cholesky_inverse_sqrt`,
//! `eigh_inverse_sqrt`) and dressing GEMMs (`ri_mp2_spin_components`'s
//! `b_flat`, `compute_rpa_intermediates`'s `b_ov`), plus `gradient.rs`'s
//! `gamma_2c` 2-center metric-derivative GEMM in the analytical RI-MP2
//! gradient.
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
use ferric_mp2::gradient::rimp2_gradient_analytical;
use ferric_mp2::rimp2::{compute_rpa_intermediates, ri_mp2_spin_components, RiMp2Config};
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

/// FERRIC_BLAS_THREADS=2 must reproduce the RI-MP2 spin-resolved energies and
/// the dressed `b_flat` tensor of the default (1 thread) — exercises the B6
/// metric-factorization site (`cholesky_inverse_sqrt`, called inside
/// `ri_mp2_spin_components`) and the B7 dressing-GEMM site (`b_flat =
/// v2c_inv_sqrt.dot(&eri3_flat)`) together, end-to-end through the real env
/// resolver.
///
/// NOT bit-identical, and that is a *measured* property, not sloppiness:
/// multi-thread OpenBLAS changes the GEMM/Cholesky reduction order at
/// production sizes, so results match to ~1e-13, not bit-for-bit. The
/// tolerance here is far above that noise floor — same trade-off documented
/// in ferric-gw/tests/blas_raise_identity.rs.
#[test]
fn ri_mp2_spin_components_consistent_across_blas_thread_counts() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("FERRIC_BLAS_THREADS");
    std::env::remove_var("FERRIC_LANCZOS_BLAS_THREADS");

    let mol = water();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    // RHF runs BEFORE any raise window (env unset here), so its rayon-parallel
    // Fock builds never see a raised BLAS count.
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    let config = RiMp2Config::default();

    let run = || ri_mp2_spin_components(&mol, &obs, &dfbs, op, &rhf, &config).unwrap();

    let (sc_base, b_base) = run();

    std::env::set_var("FERRIC_BLAS_THREADS", "2");
    let (sc_raised, b_raised) = run();
    std::env::remove_var("FERRIC_BLAS_THREADS");

    let e_os_diff = (sc_base.e_os - sc_raised.e_os).abs();
    let e_ss_diff = (sc_base.e_ss - sc_raised.e_ss).abs();
    assert!(
        e_os_diff <= 1e-11 && e_ss_diff <= 1e-11,
        "RI-MP2 spin components drift beyond reduction-order noise across BLAS thread counts: e_os_diff={e_os_diff:e} e_ss_diff={e_ss_diff:e}"
    );

    assert_eq!(b_base.dim(), b_raised.dim());
    let maxdiff_b = (&b_base - &b_raised)
        .iter()
        .map(|v| v.abs())
        .fold(0.0f64, f64::max);
    assert!(
        maxdiff_b <= 1e-11,
        "dressed b_flat drifts beyond reduction-order noise across BLAS thread counts: maxdiff={maxdiff_b:e}"
    );
    eprintln!("blas_raise_identity: ri_mp2 e_os_diff={e_os_diff:e} e_ss_diff={e_ss_diff:e} maxdiff_b={maxdiff_b:e}");
}

/// FERRIC_BLAS_THREADS=2 must reproduce `compute_rpa_intermediates`'s dressed
/// `b_ov` and `v_inv_sqrt` (the B6 `eigh_inverse_sqrt`/`cholesky_inverse_sqrt`
/// auto-select plus the B7 dressing GEMM) across thread counts.
#[test]
fn compute_rpa_intermediates_consistent_across_blas_thread_counts() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("FERRIC_BLAS_THREADS");
    std::env::remove_var("FERRIC_LANCZOS_BLAS_THREADS");

    let mol = water();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    let config = RiMp2Config::default();

    let run = || compute_rpa_intermediates(&mol, &obs, &dfbs, op, &rhf, &config).unwrap();

    let baseline = run();

    std::env::set_var("FERRIC_BLAS_THREADS", "2");
    let raised = run();
    std::env::remove_var("FERRIC_BLAS_THREADS");

    let maxdiff_b = (&baseline.b_ov - &raised.b_ov)
        .iter()
        .map(|v| v.abs())
        .fold(0.0f64, f64::max);
    assert!(
        maxdiff_b <= 1e-11,
        "RPA intermediates b_ov drifts beyond reduction-order noise across BLAS thread counts: maxdiff={maxdiff_b:e}"
    );
    let maxdiff_v = (&baseline.v_inv_sqrt - &raised.v_inv_sqrt)
        .iter()
        .map(|v| v.abs())
        .fold(0.0f64, f64::max);
    assert!(
        maxdiff_v <= 1e-11,
        "RPA intermediates v_inv_sqrt drifts beyond reduction-order noise across BLAS thread counts: maxdiff={maxdiff_v:e}"
    );
    eprintln!("blas_raise_identity: compute_rpa_intermediates maxdiff_b={maxdiff_b:e} maxdiff_v={maxdiff_v:e}");
}

/// FERRIC_BLAS_THREADS=2 must reproduce the analytical RI-MP2 nuclear
/// gradient of the default (1 thread) — exercises the B7 `gamma_2c` 2-center
/// metric-derivative GEMM in `integral_response_gradient_3c2c`, plus the B6
/// metric-factorization sites feeding the underlying MP2 intermediates.
#[test]
fn rimp2_gradient_analytical_consistent_across_blas_thread_counts() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("FERRIC_BLAS_THREADS");
    std::env::remove_var("FERRIC_LANCZOS_BLAS_THREADS");

    let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(
        &ctx, &mol, &obs, op, &bounds,
        &RhfConfig { energy_conv: 1e-10, ..Default::default() },
    )
    .unwrap();
    let config = RiMp2Config::default();

    let run = || rimp2_gradient_analytical(&mol, &obs, &dfbs, op, &bounds, &rhf, &config).unwrap();

    let baseline = run();

    std::env::set_var("FERRIC_BLAS_THREADS", "2");
    let raised = run();
    std::env::remove_var("FERRIC_BLAS_THREADS");

    let maxdiff = (&baseline - &raised)
        .iter()
        .map(|v| v.abs())
        .fold(0.0f64, f64::max);
    assert!(
        maxdiff <= 1e-10,
        "RI-MP2 analytical gradient drifts beyond reduction-order noise across BLAS thread counts: maxdiff={maxdiff:e}"
    );
    eprintln!("blas_raise_identity: rimp2_gradient_analytical maxdiff={maxdiff:e}");
}
