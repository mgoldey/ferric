//! When xc is hybrid/RSH, `solve_rhf` should auto-default df_j_aux and
//! df_k_aux to def2-universal-jkfit so the caller doesn't need to set them
//! explicitly. This test sets xc = "B3LYP" with no aux entries and checks
//! the SCF still converges to the same energy as the explicit-aux version.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn run_h2(xc: &str, explicit: bool) -> f64 {
    let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = if explicit {
        RhfConfig {
            xc: Some(xc.into()),
            df_j_aux: Some("def2-universal-jkfit".into()),
            df_k_aux: Some("def2-universal-jkfit".into()),
            energy_conv: 1e-10,
            density_conv: 1e-8,
            ..Default::default()
        }
    } else {
        RhfConfig {
            xc: Some(xc.into()),
            energy_conv: 1e-10,
            density_conv: 1e-8,
            ..Default::default()
        }
    };
    let res = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg).unwrap();
    res.energy
}

#[test]
fn b3lyp_auto_aux_matches_explicit() {
    let e_explicit = run_h2("B3LYP", true);
    let e_auto = run_h2("B3LYP", false);
    let diff = (e_explicit - e_auto).abs();
    eprintln!("B3LYP H2/STO-3G: explicit aux={e_explicit:.10}, auto aux={e_auto:.10}, diff={diff:.2e}");
    assert!(diff < 1e-12, "B3LYP auto-default aux gave different energy: diff={diff:.2e}");
}

#[test]
fn wb97xv_auto_aux_matches_explicit() {
    let e_explicit = run_h2("wB97X-V", true);
    let e_auto = run_h2("wB97X-V", false);
    let diff = (e_explicit - e_auto).abs();
    eprintln!("wB97X-V H2/STO-3G: explicit={e_explicit:.10}, auto={e_auto:.10}, diff={diff:.2e}");
    assert!(diff < 1e-12, "wB97X-V auto-default aux gave different energy: diff={diff:.2e}");
}
