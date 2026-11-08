//! `ferric_scf::certificate::scf_certificate` on real solves.
//!
//! Trivial limit: a converged, all-exact RHF has its table grade (Proven),
//! the page as its source, no approximations and no warnings. Every other
//! test moves exactly one input away from that and checks the one field it
//! should change.
//!
//! Artifact hypothesis: a certificate built from the config alone (not the
//! result) would grade the unconverged run Proven and report RI-J for a run
//! that asked for none; one built from the result alone would miss the
//! meta-GGA auto level shift. Each is a separate assertion below.

use ferric_core::basis::bundled;
use ferric_core::certificate::{Grade, GRADE_SOURCE_PAGE};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::certificate::scf_certificate;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;

const WATER: &str = "3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";
const OH: &str = "2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n";

fn solve(xyz: &str, mult: usize, cfg: &RhfConfig) -> (Molecule, ScfResult) {
    let mol = Molecule::parse_xyz(xyz, 0, mult).unwrap();
    let prep = PreparedBasis::new(&mol, &bundled("sto-3g").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let ctx = ParallelContext::default();
    let r = if mult == 1 {
        solve_rhf(&ctx, &mol, &prep, Operator::coulomb(), &bounds, cfg).expect("rhf")
    } else {
        solve_uhf(&ctx, &mol, &prep, &bounds, cfg).expect("uhf")
    };
    (mol, r)
}

fn names(c: &ferric_core::certificate::Certificate) -> Vec<&str> {
    c.approximations.iter().map(|a| a.name.as_str()).collect()
}

#[test]
fn converged_exact_rhf_is_proven_with_nothing_attached() {
    let cfg = RhfConfig::default();
    let (mol, r) = solve(WATER, 1, &cfg);
    assert!(r.converged);
    let c = scf_certificate(&mol, &r, &cfg);
    assert_eq!(c.capability, "rhf");
    assert_eq!(c.grade, Grade::Proven);
    assert_eq!(c.grade_source.as_deref(), Some(GRADE_SOURCE_PAGE));
    assert!(c.approximations.is_empty(), "{:?}", c.approximations);
    assert!(c.warnings.is_empty(), "{:?}", c.warnings);
    assert_eq!(c.build, None, "the library never stamps a build");
    assert_eq!(c.precision.as_ref().unwrap()["precision"], "f64");
}

#[test]
fn unconverged_rhf_is_capped_to_ungraded() {
    let cfg = RhfConfig {
        max_iter: 2,
        ..Default::default()
    };
    let (mol, r) = solve(WATER, 1, &cfg);
    assert!(!r.converged, "max_iter = 2 must not converge water");
    let c = scf_certificate(&mol, &r, &cfg);
    assert_eq!(c.capability, "rhf");
    assert_eq!(c.grade, Grade::Ungraded);
    assert!(
        c.warnings
            .iter()
            .any(|w| w.starts_with("grade capped from Proven to not graded: SCF did not converge")),
        "{:?}",
        c.warnings
    );
}

#[test]
fn ri_j_is_reported_from_the_route_the_solver_ran() {
    let cfg = RhfConfig {
        df_j_aux: Some("def2-universal-jkfit".into()),
        ..Default::default()
    };
    let (mol, r) = solve(WATER, 1, &cfg);
    let c = scf_certificate(&mol, &r, &cfg);
    assert_eq!(names(&c), ["ri-j"]);
    assert_eq!(
        c.approximations[0].detail,
        "aux def2-universal-jkfit, f32 tensor storage"
    );
    assert_eq!(c.grade, Grade::Proven);

    // An explicit f64 is honoured and recorded.
    let f64_cfg = RhfConfig {
        ri_j_precision: Some(ferric_scf::df_j::RiJPrecision::F64),
        ..cfg
    };
    let (mol, r) = solve(WATER, 1, &f64_cfg);
    let c = scf_certificate(&mol, &r, &f64_cfg);
    assert_eq!(
        c.approximations[0].detail,
        "aux def2-universal-jkfit, f64 tensor storage"
    );
}

#[test]
fn rks_reports_the_production_grid_and_meta_gga_its_auto_level_shift() {
    let pbe = RhfConfig {
        xc: Some("PBE".into()),
        ..Default::default()
    };
    let (mol, r) = solve(WATER, 1, &pbe);
    let c = scf_certificate(&mol, &r, &pbe);
    assert_eq!(c.capability, "ksdft");
    assert_eq!(c.grade, Grade::Proven);
    let grid = c
        .approximations
        .iter()
        .find(|a| a.name == "xc-grid")
        .expect("xc-grid");
    assert!(
        grid.detail.contains("(75, 110), nwchemradial"),
        "{}",
        grid.detail
    );
    assert!(!names(&c).contains(&"level-shift"), "{:?}", names(&c));

    let scan = RhfConfig {
        xc: Some("SCAN".into()),
        ..Default::default()
    };
    let (mol, r) = solve(WATER, 1, &scan);
    let c = scf_certificate(&mol, &r, &scan);
    let shift = c
        .approximations
        .iter()
        .find(|a| a.name == "level-shift")
        .expect("meta-GGA auto level shift");
    assert!(
        shift.detail.contains("auto-applied for meta-GGA"),
        "{}",
        shift.detail
    );
}

#[test]
fn uhf_warns_when_stability_was_not_checked() {
    let cfg = RhfConfig::default();
    let (mol, r) = solve(OH, 2, &cfg);
    let c = scf_certificate(&mol, &r, &cfg);
    assert_eq!(c.capability, "uhf");
    assert_eq!(c.grade, Grade::Proven, "a warning is not a cap");
    assert!(
        c.warnings
            .iter()
            .any(|w| w.starts_with("internal stability not checked")),
        "{:?}",
        c.warnings
    );

    let checked = RhfConfig {
        check_stability: true,
        ..Default::default()
    };
    let (mol, r) = solve(OH, 2, &checked);
    let c = scf_certificate(&mol, &r, &checked);
    assert!(r.stability.is_some());
    assert!(
        !c.warnings
            .iter()
            .any(|w| w.starts_with("internal stability not checked")),
        "{:?}",
        c.warnings
    );
}
