//! What rs-mp2-rpa refuses with the terf attenuator in formulation B.
//!
//! # Refusals
//!
//! The long-range dRPA of B with terf is the signed log-determinant of Coulomb
//! RI minus terfc RI (`ferric_rpa::energy::signed_drpa_energy`). It forms no
//! static eigenbasis, so the dRPA options that need one are refused by
//! `refuse_unsupported_signed_drpa` before any integral is built, with the
//! typed `FerricError::Unsupported`. One test per refusal builds the smallest
//! system that reaches it (H2 / STO-3G with the cc-pVDZ-RI fit; the refusal
//! fires before the fit is formed, so no terf table is read) and matches the
//! variant and the option it names. Three controls show the refusal is the
//! combination and not the option: erf with B, and terf with T, run with the
//! same option and are not refused.
//!
//! # Mutations
//!
//! Each test fails by assertion when its refusal is removed; the outcomes are
//! in the commit message that adds the tests.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_rpa::rs_mp2_rpa::{rs_mp2_lr_rpa, Attenuator, RsMp2RpaConfig, RsMp2RpaFormulation};
use ferric_rpa::{Chi0Backend, Chi0Sparsity};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::ScfResult;

const H2: &str = "2\nH2\nH 0 0 0\nH 0 0 0.74\n";
const OH: &str = "2\nOH\nO 0 0 0\nH 0 0 0.97\n";

fn tables() -> bool {
    std::env::var("FERRIC_TERF_TABLE_DIR")
        .map(|d| std::path::Path::new(&d).join("16_4_2.bin").exists())
        .unwrap_or(false)
}

struct Sys {
    mol: Molecule,
    obs: PreparedBasis,
    dfbs: PreparedBasis,
    scf: ScfResult,
}

fn h2() -> Sys {
    let mol = Molecule::parse_xyz(H2, 0, 1).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let scf = solve_rhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        Operator::coulomb(),
        &bounds,
        &RhfConfig {
            energy_conv: 1e-11,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(scf.converged);
    Sys {
        mol,
        obs,
        dfbs,
        scf,
    }
}

fn oh_uhf() -> Sys {
    let mol = Molecule::parse_xyz(OH, 0, 2).unwrap();
    let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &obs).unwrap();
    let scf = ferric_scf::uhf::solve_uhf(
        &ParallelContext::default(),
        &mol,
        &obs,
        &bounds,
        &RhfConfig {
            energy_conv: 1e-11,
            density_conv: 1e-9,
            max_iter: 300,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(scf.converged);
    Sys {
        mol,
        obs,
        dfbs,
        scf,
    }
}

fn terf_b() -> RsMp2RpaConfig {
    RsMp2RpaConfig {
        attenuator: Attenuator::Terf,
        formulation: RsMp2RpaFormulation::DeltaLr,
        ..Default::default()
    }
}

fn run(sys: &Sys, cfg: &RsMp2RpaConfig) -> Result<ferric_rpa::RsMp2RpaResult, FerricError> {
    rs_mp2_lr_rpa(&sys.mol, &sys.obs, &sys.dfbs, &sys.scf, cfg)
}

/// The refusal: `Unsupported` whose `feature` names `option` and whose
/// `Display` carries both the option and the alternative.
fn assert_refused(r: Result<ferric_rpa::RsMp2RpaResult, FerricError>, option: &str) {
    match r {
        Err(FerricError::Unsupported { feature, reason }) => {
            assert!(
                feature.contains("terf") && feature.contains(option),
                "feature {feature:?} does not name {option:?}"
            );
            assert!(
                reason.contains("signed log-det") && reason.contains("chi0_sparsity = dense"),
                "reason {reason:?} lacks the cause or the alternative"
            );
        }
        Err(other) => panic!("expected Unsupported for {option}, got {other:?}"),
        Ok(r) => panic!("{option} was not refused (e_corr {})", r.e_corr),
    }
}

#[test]
fn terf_b_refuses_pdep_truncation() {
    let sys = h2();
    let mut cfg = terf_b();
    cfg.drpa.trunc_thresh = 1e-4;
    assert_refused(run(&sys, &cfg), "trunc_thresh");
}

#[test]
fn terf_b_refuses_the_laplace_chi0() {
    let sys = h2();
    let mut cfg = terf_b();
    cfg.drpa.chi0_backend = Chi0Backend::Laplace { n_quad: 5 };
    assert_refused(run(&sys, &cfg), "chi0_backend");
}

#[test]
fn terf_b_refuses_the_boys_screened_chi0() {
    let sys = h2();
    let mut cfg = terf_b();
    cfg.drpa.chi0_sparsity = Chi0Sparsity::BoysScreened {
        thresh: 1e-4,
        dist_cutoff: f64::INFINITY,
    };
    assert_refused(run(&sys, &cfg), "chi0_sparsity");
}

/// `Auto` resolves to the Boys-screened chi0 once the molecule has at least
/// `atom_cutoff` atoms; H2 has two.
#[test]
fn terf_b_refuses_an_auto_sparsity_that_resolves_to_boys() {
    let sys = h2();
    let mut cfg = terf_b();
    cfg.drpa.chi0_sparsity = Chi0Sparsity::Auto {
        boys_thresh: 1e-4,
        atom_cutoff: 2,
        dist_cutoff: f64::INFINITY,
    };
    assert_refused(run(&sys, &cfg), "chi0_sparsity");
}

#[test]
fn terf_b_refuses_the_inverse_dielectric() {
    let sys = h2();
    let mut cfg = terf_b();
    cfg.drpa.need_inv_dielectric_freq = true;
    assert_refused(run(&sys, &cfg), "need_inv_dielectric_freq");
}

/// The open-shell driver takes the same refusal, before its own checks.
#[test]
fn terf_b_refuses_pdep_truncation_open_shell() {
    let sys = oh_uhf();
    let mut cfg = terf_b();
    cfg.drpa.trunc_thresh = 1e-4;
    assert_refused(run(&sys, &cfg), "trunc_thresh");
}

/// Control 1: erf with B takes every refused option. No terf table is read.
#[test]
fn erf_b_is_not_refused_for_the_same_options() {
    let sys = h2();
    let mut cfg = RsMp2RpaConfig {
        formulation: RsMp2RpaFormulation::DeltaLr,
        ..Default::default()
    };
    cfg.drpa.trunc_thresh = 1e-4;
    cfg.drpa.need_inv_dielectric_freq = true;
    let r = run(&sys, &cfg);
    assert!(
        !matches!(r, Err(FerricError::Unsupported { .. })),
        "erf B was refused: {:?}",
        r.err()
    );
}

/// Control 2: terf with T does not form the signed log-det, so the options
/// are not refused. Needs the terf tables to run to the end.
#[test]
fn terf_t_is_not_refused_for_the_same_options() {
    if !tables() {
        eprintln!("SKIP terf_t_is_not_refused_for_the_same_options: terf tables absent");
        return;
    }
    let sys = h2();
    let mut cfg = RsMp2RpaConfig {
        attenuator: Attenuator::Terf,
        formulation: RsMp2RpaFormulation::CoupledRings,
        r0: 3.18,
        ..Default::default()
    };
    cfg.drpa.trunc_thresh = 1e-4;
    let r = run(&sys, &cfg).expect("terf T with trunc_thresh runs");
    assert!(r.e_corr.is_finite());
}
