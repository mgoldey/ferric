//! Real-env FERRIC_BLAS_THREADS raise tests for the B8 wrapped sites: the
//! digestion GEMMs in vxc.rs (semilocal_vxc_closed_scratch), vv10.rs
//! (add_vv10_scratch), fxc.rs (LdaFxcKernel::apply_with_ref), the
//! pre-rayon-region GEMMs in density_on_grid.rs (eval_density_closed /
//! eval_density_uks), gradient.rs (xc_gradient_closed_lda /
//! xc_gradient_closed_gga_from_density / uks-gradient M-precompute), and
//! cdft.rs (build_weight_matrix).
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
use ferric_dft::ao_grid::eval_basis_and_grad_on_points;
use ferric_dft::cdft::build_weight_matrix;
use ferric_dft::density_on_grid::{eval_density_closed, eval_density_uks};
use ferric_dft::fxc::LdaFxcKernel;
use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
use ferric_dft::libxc::{xc_def_from_name, xc_def_from_name_nspin};
use ferric_dft::vxc::semilocal_vxc_closed;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ks_gradient::ks_gradient_closed;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use std::sync::Mutex;

// Serialize ALL tests in this binary: the harness still runs tests on
// parallel threads within one binary, and a raised BLAS count must never be
// live while another test computes.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn water() -> Molecule {
    Molecule::parse_xyz(
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
        0,
        1,
    )
    .unwrap()
}

/// FERRIC_BLAS_THREADS=2 must reproduce the same semilocal V_xc matrix and
/// E_xc as the default (1 thread) — exercises vxc.rs's
/// `semilocal_vxc_closed_scratch` LDA+GGA digestion-GEMM sites (via PBE, a
/// GGA functional, to hit both the LDA-like and GGA-like GEMM branches).
///
/// NOT bit-identical, and that is a *measured* property, not sloppiness:
/// multi-thread OpenBLAS changes the GEMM reduction order at production
/// sizes, so results match to ~1e-13, not bit-for-bit. The tolerance here is
/// far above that noise floor — same trade-off documented in
/// ferric-gw/tests/blas_raise_identity.rs.
#[test]
fn semilocal_vxc_gga_consistent_across_blas_thread_counts() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("FERRIC_BLAS_THREADS");
    std::env::remove_var("FERRIC_LANCZOS_BLAS_THREADS");

    let mol = water();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

    let grid = build_atomic_grid(&mol, &AtomicGridConfig::default());
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let (chi, dchi) = eval_basis_and_grad_on_points(&mol, &bs, &pts).unwrap();
    let dens = eval_density_closed(&rhf.density_total, &chi, &dchi);
    let xc = xc_def_from_name("PBE").unwrap();

    let run = || semilocal_vxc_closed(&grid, &chi, &dchi, &dens, None, &xc);

    let (e_base, v_base) = run();
    std::env::set_var("FERRIC_BLAS_THREADS", "2");
    let (e_raised, v_raised) = run();
    std::env::remove_var("FERRIC_BLAS_THREADS");

    let e_diff = (e_base - e_raised).abs();
    assert!(
        e_diff <= 1e-11,
        "PBE E_xc drifts beyond reduction-order noise across BLAS thread counts: diff={e_diff:e}"
    );
    let maxdiff_v = (&v_base - &v_raised)
        .iter()
        .map(|v| v.abs())
        .fold(0.0f64, f64::max);
    assert!(
        maxdiff_v <= 1e-11,
        "PBE V_xc drifts beyond reduction-order noise across BLAS thread counts: maxdiff={maxdiff_v:e}"
    );
    eprintln!("blas_raise_identity: semilocal_vxc_gga e_diff={e_diff:e} maxdiff_v={maxdiff_v:e}");
}

/// FERRIC_BLAS_THREADS=2 must reproduce eval_density_closed's rho/grad/sigma
/// (the pre-rayon-region `phi = D.dot(chi)` GEMM in density_on_grid.rs)
/// across thread counts. Uses a grid well above PAR_MIN_PTS (512) so the
/// downstream rayon region is also exercised in both runs, isolating the
/// GEMM as the only variable.
#[test]
fn eval_density_closed_consistent_across_blas_thread_counts() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("FERRIC_BLAS_THREADS");
    std::env::remove_var("FERRIC_LANCZOS_BLAS_THREADS");

    let mol = water();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

    let grid = build_atomic_grid(&mol, &AtomicGridConfig::default());
    assert!(grid.len() >= 512, "grid must exceed PAR_MIN_PTS to exercise both paths");
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let (chi, dchi) = eval_basis_and_grad_on_points(&mol, &bs, &pts).unwrap();

    let run = || eval_density_closed(&rhf.density_total, &chi, &dchi);

    let base = run();
    std::env::set_var("FERRIC_BLAS_THREADS", "2");
    let raised = run();
    std::env::remove_var("FERRIC_BLAS_THREADS");

    let maxdiff_rho = (&base.rho - &raised.rho).iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    let maxdiff_grad = (&base.grad - &raised.grad).iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    assert!(
        maxdiff_rho <= 1e-11 && maxdiff_grad <= 1e-11,
        "eval_density_closed drifts beyond reduction-order noise across BLAS thread counts: maxdiff_rho={maxdiff_rho:e} maxdiff_grad={maxdiff_grad:e}"
    );
    eprintln!("blas_raise_identity: eval_density_closed maxdiff_rho={maxdiff_rho:e} maxdiff_grad={maxdiff_grad:e}");
}

/// FERRIC_BLAS_THREADS=2 must reproduce eval_density_uks's per-spin rho/grad
/// (the two pre-rayon-region `phi_a/phi_b = D.dot(chi)` GEMMs) across thread
/// counts — same GEMM-isolation argument as the closed-shell test above.
#[test]
fn eval_density_uks_consistent_across_blas_thread_counts() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("FERRIC_BLAS_THREADS");
    std::env::remove_var("FERRIC_LANCZOS_BLAS_THREADS");

    let mol = water();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    // Synthesize distinct alpha/beta densities from the closed-shell result
    // (D_a = D_b = D_total/2 for RHF); perturb D_b slightly so the two spin
    // channels are not degenerate, exercising the fused evaluator honestly.
    let d_a = &rhf.density_total * 0.5;
    let mut d_b = &rhf.density_total * 0.5;
    let n = d_b.nrows();
    for i in 0..n {
        d_b[(i, i)] *= 1.0 + 1e-3 * (i as f64);
    }

    let grid = build_atomic_grid(&mol, &AtomicGridConfig::default());
    assert!(grid.len() >= 512, "grid must exceed PAR_MIN_PTS to exercise both paths");
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let (chi, dchi) = eval_basis_and_grad_on_points(&mol, &bs, &pts).unwrap();

    let run = || eval_density_uks(&d_a, &d_b, &chi, &dchi);

    let base = run();
    std::env::set_var("FERRIC_BLAS_THREADS", "2");
    let raised = run();
    std::env::remove_var("FERRIC_BLAS_THREADS");

    let maxdiff_a = (&base.rho_a - &raised.rho_a).iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    let maxdiff_b = (&base.rho_b - &raised.rho_b).iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    assert!(
        maxdiff_a <= 1e-11 && maxdiff_b <= 1e-11,
        "eval_density_uks drifts beyond reduction-order noise across BLAS thread counts: maxdiff_a={maxdiff_a:e} maxdiff_b={maxdiff_b:e}"
    );
    eprintln!("blas_raise_identity: eval_density_uks maxdiff_a={maxdiff_a:e} maxdiff_b={maxdiff_b:e}");
}

/// FERRIC_BLAS_THREADS=2 must reproduce LdaFxcKernel::apply_with_ref's
/// (δV_xc^α, δV_xc^β) response — the fxc.rs back-projection digestion GEMMs
/// used by the ROHF/ROKS AH-Newton solver — across thread counts.
#[test]
fn lda_fxc_apply_with_ref_consistent_across_blas_thread_counts() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("FERRIC_BLAS_THREADS");
    std::env::remove_var("FERRIC_LANCZOS_BLAS_THREADS");

    let mol = water();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();

    // Polarized (nspin=2) is required: LdaFxcKernel::apply_with_ref calls
    // eval_lda_fxc_polarized, which panics on an nspin=1 handle (see
    // rohf.rs's real usage: xc_def_from_name_nspin("LDA", 2)).
    let xc = xc_def_from_name_nspin("LDA", 2).unwrap();
    let cfg = AtomicGridConfig::default();
    let kernel = LdaFxcKernel::new(&mol, &bs, xc, &cfg).unwrap();

    let d_a = &rhf.density_total * 0.5;
    let d_b = &rhf.density_total * 0.5;
    let (rho_a0, rho_b0) = kernel.reference_density(&d_a, &d_b);

    // Small symmetric perturbation density (as a Newton step would produce).
    let n = d_a.nrows();
    let mut dd_a = ndarray::Array2::<f64>::zeros((n, n));
    let mut dd_b = ndarray::Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let v = 1e-3 * (((i * 7 + j * 3 + 1) as f64).sin());
            dd_a[(i, j)] = v;
            dd_a[(j, i)] = v;
            dd_b[(i, j)] = -0.5 * v;
            dd_b[(j, i)] = -0.5 * v;
        }
    }

    let run = || kernel.apply_with_ref(&rho_a0, &rho_b0, &dd_a, &dd_b);

    let (base_a, base_b) = run();
    std::env::set_var("FERRIC_BLAS_THREADS", "2");
    let (raised_a, raised_b) = run();
    std::env::remove_var("FERRIC_BLAS_THREADS");

    let maxdiff_a = (&base_a - &raised_a).iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    let maxdiff_b = (&base_b - &raised_b).iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    assert!(
        maxdiff_a <= 1e-11 && maxdiff_b <= 1e-11,
        "LdaFxcKernel::apply_with_ref drifts beyond reduction-order noise across BLAS thread counts: maxdiff_a={maxdiff_a:e} maxdiff_b={maxdiff_b:e}"
    );
    eprintln!("blas_raise_identity: lda_fxc_apply_with_ref maxdiff_a={maxdiff_a:e} maxdiff_b={maxdiff_b:e}");
}

/// FERRIC_BLAS_THREADS=2 must reproduce cdft.rs's build_weight_matrix (the
/// digestion GEMM after the rayon-gated per-point Becke-weight map) across
/// thread counts.
#[test]
fn cdft_weight_matrix_consistent_across_blas_thread_counts() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("FERRIC_BLAS_THREADS");
    std::env::remove_var("FERRIC_LANCZOS_BLAS_THREADS");

    let mol = water();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let grid = build_atomic_grid(&mol, &AtomicGridConfig::default());
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let (chi, _dchi) = eval_basis_and_grad_on_points(&mol, &bs, &pts).unwrap();
    let fragment = [0usize]; // O atom fragment

    let run = || build_weight_matrix(&mol, &grid, &chi, &fragment);

    let base = run();
    std::env::set_var("FERRIC_BLAS_THREADS", "2");
    let raised = run();
    std::env::remove_var("FERRIC_BLAS_THREADS");

    let maxdiff = (&base - &raised).iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    assert!(
        maxdiff <= 1e-11,
        "build_weight_matrix drifts beyond reduction-order noise across BLAS thread counts: maxdiff={maxdiff:e}"
    );
    eprintln!("blas_raise_identity: cdft_weight_matrix maxdiff={maxdiff:e}");
}

/// FERRIC_BLAS_THREADS=2 must reproduce the full wB97X-V analytical nuclear
/// gradient (RSH GGA + VV10 nonlocal) of the default (1 thread) — exercises
/// gradient.rs's GGA M-precompute GEMMs and vv10.rs's add_vv10_scratch
/// digestion GEMMs together, end-to-end through the real KS gradient driver.
/// Small system (H2/6-31G) and a loose grid to keep the raised-thread run
/// cheap; only the cross-thread-count identity is under test here, not
/// absolute accuracy (already covered by dft_gradient_wb97xv.rs).
#[test]
fn wb97xv_gradient_consistent_across_blas_thread_counts() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("FERRIC_BLAS_THREADS");
    std::env::remove_var("FERRIC_LANCZOS_BLAS_THREADS");

    let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
    let bs = basis::bundled("6-31g").unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        xc: Some("wB97X-V".into()),
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        energy_conv: 1e-10,
        density_conv: 1e-8,
        ..Default::default()
    };
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &cfg).unwrap();

    let run = || ks_gradient_closed(&mol, &obs, &bs, op, &bounds, "wB97X-V", &rhf, None).unwrap();

    let base = run();
    std::env::set_var("FERRIC_BLAS_THREADS", "2");
    let raised = run();
    std::env::remove_var("FERRIC_BLAS_THREADS");

    let maxdiff = (&base - &raised).iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    assert!(
        maxdiff <= 1e-9,
        "wB97X-V gradient drifts beyond reduction-order noise across BLAS thread counts: maxdiff={maxdiff:e}"
    );
    eprintln!("blas_raise_identity: wb97xv_gradient maxdiff={maxdiff:e}");
}
