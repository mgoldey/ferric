//! Becke grid-response (P2.1) translational-invariance check.
//!
//! With the PySCF-style `weight1 + grid-coord-response` correction applied,
//! Σ_A ∂E_xc/∂R_A drops from ~5e-4 (no response) to ~1e-5 on the (75, 110)
//! grid. The remaining residual is dominated by the Becke quadrature noise
//! floor (improves with finer Lebedev orders).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ks_gradient::ks_gradient_closed;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn h2o() -> Molecule {
    Molecule::parse_xyz(
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
        0,
        1,
    )
    .unwrap()
}

fn run(xc: &str, basis_name: &str) -> ndarray::Array2<f64> {
    let mol = h2o();
    let bs = basis::bundled(basis_name).unwrap();
    let obs = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &obs).unwrap();
    let cfg = RhfConfig {
        xc: Some(xc.into()),
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        ..Default::default()
    };
    let rhf = solve_rhf(&ParallelContext::default(), &mol, &obs, op, &bounds, &cfg).unwrap();
    ks_gradient_closed(&mol, &obs, &bs, op, &bounds, xc, &rhf, None).unwrap()
}

#[test]
fn lda_h2o_ccpvdz_translational_invariance() {
    let g = run("LDA", "cc-pvdz");
    let sum = g.sum_axis(ndarray::Axis(0));
    let max_drift = sum.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
    eprintln!("LDA/cc-pvdz Σ_A ∂E/∂R = {sum:?}, max drift = {max_drift:.3e}");
    // With P2.1 response: machine precision (~1e-13). Without it was ~3e-4.
    assert!(
        max_drift < 1e-10,
        "translational drift {max_drift:.3e} — grid response may be broken"
    );
}

#[test]
fn pbe_h2o_ccpvdz_translational_invariance() {
    let g = run("PBE", "cc-pvdz");
    let sum = g.sum_axis(ndarray::Axis(0));
    let max_drift = sum.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
    eprintln!("PBE/cc-pvdz Σ_A ∂E/∂R = {sum:?}, max drift = {max_drift:.3e}");
    // With GGA grid response: ~1e-13. Without it was ~2e-4.
    assert!(
        max_drift < 1e-10,
        "PBE translational drift {max_drift:.3e} — grid response may be broken"
    );
}

#[test]
fn b3lyp_h2o_ccpvdz_translational_invariance() {
    let g = run("B3LYP", "cc-pvdz");
    let sum = g.sum_axis(ndarray::Axis(0));
    let max_drift = sum.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
    eprintln!("B3LYP/cc-pvdz Σ_A ∂E/∂R = {sum:?}, max drift = {max_drift:.3e}");
    assert!(
        max_drift < 1e-10,
        "B3LYP translational drift {max_drift:.3e} — grid response may be broken"
    );
}

#[test]
fn uks_b3lyp_h2o_ccpvdz_translational_invariance() {
    // UKS on closed-shell H2O reduces to RKS but exercises the UKS code path.
    use ferric_scf::ks_gradient::ks_gradient_uks;
    use ferric_scf::uhf::solve_uhf;

    let xc = "B3LYP";
    let mol = h2o();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        xc: Some(xc.into()),
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        energy_conv: 1e-10,
        density_conv: 1e-8,
        ..Default::default()
    };
    let uhf = solve_uhf(&ParallelContext::default(), &mol, &prep, &bounds, &cfg).unwrap();

    let g = ks_gradient_uks(&mol, &prep, &bs, op, &bounds, xc, &uhf, None).unwrap();
    let sum = g.sum_axis(ndarray::Axis(0));
    let max_drift = sum.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
    eprintln!("UKS B3LYP/cc-pvdz H2O Σ_A ∂E/∂R = {sum:?}, max drift = {max_drift:.3e}");
    // UHF on closed-shell H2O preserves spin symmetry only up to SCF
    // convergence — D_α ≈ D_β at ~1e-8 in matrix norm, propagating to
    // ~1e-9 in the gradient sum. Still ~10⁵× better than the no-response
    // baseline (~3e-4) and proves the UKS response paths add correctly.
    assert!(
        max_drift < 1e-7,
        "UKS B3LYP translational drift {max_drift:.3e}"
    );
}
