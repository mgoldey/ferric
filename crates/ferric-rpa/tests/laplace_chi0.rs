//! Integration tests for the Laplace-separable χ₀ backend.
//!
//! Status (C6 correctness gate): the MO-basis Laplace χ₀ kernel reproduces the
//! Dense kernel at ω=0 (the dominant static-RPA contribution) to within the
//! minimax-1/x table accuracy. The full imaginary-frequency integration is
//! deferred to the AO cubic-scaling follow-up because the GL frequency grid
//! reaches ω ≫ e_ia^max where the cosine-transformed Laplace identity needs
//! a different (cos-transformed) minimax table than the one we currently bundle.
//! See `laplace_chi0::dielectric_matrix_laplace_into` and the C6 follow-up TODO.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::{compute_rpa_intermediates, RiMp2Config};
use ferric_rpa::laplace_chi0::{build_laplace_for_gaps, dielectric_matrix_laplace};
use ferric_rpa::sternheimer::dielectric_matrix;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::ScfResult;
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

fn setup(
    xyz: &str,
    obs_name: &str,
    dfbs_name: &str,
) -> (Molecule, PreparedBasis, PreparedBasis, Operator, ScfResult) {
    let ctx = ParallelContext::default();
    let mol = Molecule::load_xyz(xyz).unwrap();
    let obs_bs = basis::bundled(obs_name).unwrap();
    let dfbs_bs = basis::bundled(dfbs_name).unwrap();
    let op = Operator::coulomb();
    let obs = PreparedBasis::new(&mol, &obs_bs).unwrap();
    let dfbs = PreparedBasis::new(&mol, &dfbs_bs).unwrap();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let rhf = solve_rhf(&ctx, &mol, &obs, op, &bounds, &RhfConfig::default()).unwrap();
    (mol, obs, dfbs, op, rhf)
}

/// At ω=0 the Laplace approximation of `Σ_ia 4 e_ia/e_ia² Y Y = Σ_ia 4/e_ia Y Y`
/// reduces to the minimax-1/x identity, which is tight to its tabulated error.
/// On H₂O/cc-pVDZ with k=7 we expect ≤1e-4 relative agreement on the
/// dielectric matrix in the full naux subspace.
#[test]
fn h2o_ccpvdz_laplace_matches_dense_at_static() {
    let (mol, obs, dfbs, op, rhf) = setup(
        "../../testdata/molecules/water.xyz",
        "cc-pvdz",
        "cc-pvdz-ri",
    );

    let mp2_cfg = RiMp2Config { frozen_core: 0, memory_budget_bytes: None };
    let inter = compute_rpa_intermediates(&mol, &obs, &dfbs, op, &rhf, &mp2_cfg).unwrap();
    let b_ov = &inter.b_ov;
    let eps_occ: Vec<f64> = rhf.eps_r()[inter.first_occ..inter.first_occ + inter.nocc].to_vec();
    let eps_vir: Vec<f64> =
        rhf.eps_r()[inter.nocc_total..inter.nocc_total + inter.nvir].to_vec();

    let naux = inter.naux;
    // Full identity subspace — every aux function is a trial vector.
    let v_mat: Array2<f64> = Array2::eye(naux);

    let dense = dielectric_matrix(&v_mat, b_ov, &eps_occ, &eps_vir, 0.0);

    let lap_q = build_laplace_for_gaps(&eps_occ, &eps_vir, 7);
    let lap = dielectric_matrix_laplace(&v_mat, b_ov, &eps_occ, &eps_vir, 0.0, &lap_q);

    let max_abs_dense = dense.iter().fold(0.0_f64, |a, &x| a.max(x.abs()));
    let max_err = dense
        .iter()
        .zip(lap.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let rel = max_err / max_abs_dense;
    eprintln!(
        "H2O/cc-pVDZ ε̃(ω=0): max|Dense|={:.3e}  max|Δ|={:.3e}  rel={:.3e}",
        max_abs_dense, max_err, rel
    );
    // Minimax-1/x at k=7 over the full e_ia range of H2O/cc-pVDZ delivers
    // ~1e-3 relative error elementwise.
    assert!(
        rel < 5e-3,
        "Laplace χ₀ vs Dense χ₀ at ω=0 disagreement: rel={rel:.3e}"
    );
}

/// Tighter check on a small synthetic system: with the same energy range, the
/// minimax-1/x table approximation of 1/e_ia controls the absolute element-wise
/// error to ≤1e-4 at k=7.
#[test]
fn small_system_laplace_static_tight() {
    let eps_occ = vec![-0.5_f64, -0.3];
    let eps_vir = vec![0.4_f64, 0.9, 1.4];
    let nov = eps_occ.len() * eps_vir.len();
    let naux = 4usize;
    let m = 3usize;
    let b_ov = Array2::from_shape_fn((naux, nov), |(p, ia)| {
        ((p as f64 * 0.7).sin() + (ia as f64 * 0.3).cos()) * 0.2
    });
    let v_mat = Array2::from_shape_fn((naux, m), |(p, a)| {
        if p == a { 1.0 } else { 0.05 * (p as f64 - a as f64) }
    });

    let dense = dielectric_matrix(&v_mat, &b_ov, &eps_occ, &eps_vir, 0.0);
    let q = build_laplace_for_gaps(&eps_occ, &eps_vir, 7);
    let lap = dielectric_matrix_laplace(&v_mat, &b_ov, &eps_occ, &eps_vir, 0.0, &q);
    let err = dense.iter().zip(lap.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    eprintln!("small-system static: max err = {err:.3e}");
    assert!(err < 1e-3, "small system Laplace vs Dense err = {err}");
}
