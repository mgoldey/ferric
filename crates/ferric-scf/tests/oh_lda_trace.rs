//! Diagnostic-only: trace OH/LDA SCF iterations to see what the plateau
//! actually looks like. Set FERRIC_ROHF_TRACE=1 to enable the per-iter
//! output added to solve_rohf. This test always passes (no assertion); it
//! exists to make it convenient to capture the trace for analysis.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;

#[test]
fn trace_oh_lda_plateau() {
    if std::env::var("FERRIC_ROHF_TRACE").is_err() {
        eprintln!("Set FERRIC_ROHF_TRACE=1 to see the trace; test skipped silently.");
        return;
    }
    let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0, 2).unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    eprintln!("=== DIIS-only OH/LDA ===");
    let _ = solve_rohf(&ctx, &mol, &prep, op, &bounds, &RhfConfig {
        xc: Some("LDA".into()),
        energy_conv: 1e-9,
        density_conv: 1e-5,
        max_iter: 60,
        level_shift: 0.2,
        ..Default::default()
    });

    eprintln!("=== AH-enabled OH/LDA ===");
    let _ = solve_rohf(&ctx, &mol, &prep, op, &bounds, &RhfConfig {
        xc: Some("LDA".into()),
        energy_conv: 1e-9,
        density_conv: 1e-5,
        max_iter: 60,
        level_shift: 0.2,
        ah_trigger: 1e-2,
        ..Default::default()
    });
}
