//! Verify DFT now converges on def2-TZVP after contraction-renormalization fix
//! at basis-load time (P1.2.b). All four functionals must match the
//! cc-pVDZ-quality energy ballpark for H2O.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn run_h2o(xc: &str, basis_name: &str) -> f64 {
    let mol = Molecule::parse_xyz(
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n", 0, 1).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        xc: Some(xc.into()),
        energy_conv: 1e-9,
        density_conv: 1e-7,
        max_iter: 200,
        ..Default::default()
    };
    let res = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg).unwrap();
    eprintln!("{xc} H2O/{basis_name}: E = {:.8} ({} iter)", res.energy, res.iterations);
    res.energy
}

/// Pre-fix, all DFT XCs diverged on def2-TZVP. They should now land near the
/// expected PySCF values.
#[test]
fn lda_h2o_def2_tzvp() {
    let e = run_h2o("LDA", "def2-tzvp");
    // SVWN H2O ≈ -75.91 Ha at near-equilibrium geometry
    assert!((e - (-75.91)).abs() < 0.1, "LDA energy {e}");
}

#[test]
fn pbe_h2o_def2_tzvp() {
    let e = run_h2o("PBE", "def2-tzvp");
    // PBE H2O ≈ -76.38 Ha at def2-TZVP
    assert!((e - (-76.38)).abs() < 0.1, "PBE energy {e}");
}

#[test]
fn b3lyp_h2o_def2_tzvp() {
    let e = run_h2o("B3LYP", "def2-tzvp");
    // B3LYP H2O ≈ -76.45 Ha at def2-TZVP
    assert!((e - (-76.45)).abs() < 0.1, "B3LYP energy {e}");
}
