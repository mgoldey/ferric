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
//!
//! Every solve goes through [`solve`], which holds [`SOLVE_LOCK`]. The
//! precision record of a result is the difference of process-wide kernel
//! counters (`ferric_core::gpu::provenance`), so an f32 RI-J solve in a
//! concurrent test would be counted into an all-f64 solve's record.
//! Thread-local counters would not fix this: a KS solve with RI-J builds J
//! under `rayon::join` (the J/XC overlap in `rhf.rs`), so its `dfj-f32` call
//! is counted on a rayon worker. Serializing the solves keeps every record
//! exact.

use ferric_core::basis::bundled;
use ferric_core::certificate::{Grade, GRADE_SOURCE_PAGE};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::certificate::{scf_certificate, with_scf_envelope};
use ferric_scf::df_j::RiJPrecision;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;

const WATER: &str = "3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";
const OH: &str = "2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n";

/// Serializes the solves of this file (see the module doc).
static SOLVE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn solve(xyz: &str, mult: usize, cfg: &RhfConfig) -> (Molecule, ScfResult) {
    // A test that panicked while holding the lock leaves the counters valid.
    let _serial = SOLVE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
        "aux def2-universal-jkfit, f32 tensor storage (requested f32: default)"
    );
    assert_eq!(c.grade, Grade::Proven);

    // An explicit f64 is honoured and recorded.
    let f64_cfg = RhfConfig {
        ri_j_precision: Some(ferric_scf::df_j::RiJPrecision::F64),
        ..cfg.clone()
    };
    let (mol, r) = solve(WATER, 1, &f64_cfg);
    let c = scf_certificate(&mol, &r, &f64_cfg);
    assert_eq!(
        c.approximations[0].detail,
        "aux def2-universal-jkfit, f64 tensor storage (requested f64: explicit)"
    );

    // A geometry task with nothing set records the f64 task default.
    let geom_cfg = cfg.for_geometry_task();
    let (mol, r) = solve(WATER, 1, &geom_cfg);
    let c = scf_certificate(&mol, &r, &geom_cfg);
    assert_eq!(
        c.approximations[0].detail,
        "aux def2-universal-jkfit, f64 tensor storage (requested f64: geometry-task default)"
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

/// `measured_error` is the solver's energy anchor, chosen from the result's
/// spin and the presence of a functional; the envelope adds warnings and
/// leaves every other field as it was.
#[test]
fn measured_error_is_the_solver_anchor_and_the_envelope_only_warns() {
    let hf = RhfConfig::default();
    let (water, r) = solve(WATER, 1, &hf);
    let c = scf_certificate(&water, &r, &hf);
    let e = c.measured_error.as_ref().expect("RHF anchor");
    assert_eq!(
        (e.quantity.as_str(), e.value, e.units.as_str()),
        ("energy", 1.8e-12, "Ha")
    );
    // Water / STO-3G is inside the RHF anchor: the envelope adds nothing.
    let sto = bundled("STO-3G").unwrap();
    assert_eq!(with_scf_envelope(c.clone(), &water, &r, &hf, &sto), c);
    // A basis the anchor never ran: one warning, nothing else moves.
    let mut other = sto.clone();
    other.name = "6-31g(2df,p)".to_string();
    other.shells.get_mut(&8).unwrap()[0].coefficients[0] *= 1.01;
    let out = with_scf_envelope(c.clone(), &water, &r, &hf, &other);
    assert_eq!(out.warnings.len(), 1, "{:?}", out.warnings);
    assert!(out.warnings[0].starts_with("basis 6-31g(2df,p) is outside the RHF"));
    let mut back = out.clone();
    back.warnings.clear();
    assert_eq!(back, c, "the envelope changes warnings only");

    let (oh, r) = solve(OH, 2, &hf);
    let c = scf_certificate(&oh, &r, &hf);
    assert_eq!(c.measured_error.as_ref().unwrap().value, 4.0e-12);

    // O is outside the def2-SVP KS anchor's elements.
    let pbe = RhfConfig {
        xc: Some("PBE".into()),
        ri_j_precision: Some(RiJPrecision::F64),
        dft_grid: Some(AtomicGridConfig::default()),
        ..Default::default()
    };
    let (water, r) = solve(WATER, 1, &pbe);
    let c = scf_certificate(&water, &r, &pbe);
    assert_eq!(c.measured_error.as_ref().unwrap().value, 4.2e-12);
    let out = with_scf_envelope(c.clone(), &water, &r, &pbe, &bundled("def2-svp").unwrap());
    assert_eq!(out.grade, c.grade);
    assert_eq!(
        out.warnings.len(),
        c.warnings.len() + 1,
        "{:?}",
        out.warnings
    );
    assert!(out
        .warnings
        .last()
        .unwrap()
        .starts_with("element O (covered: H, Si, S, Cl)"));
}
