//! Which Hessian `harmonic_frequencies` builds (`FrequencyConfig::hessian`).
//!
//! `Auto` must take the analytic RHF Hessian exactly where
//! `analytic_hessian_available` accepts the system, the analytic UHF Hessian
//! for a UHF reference where `analytic_uhf_hessian_available` accepts it, and
//! fall back to finite differences everywhere else (always for ROHF, which has
//! no analytic Hessian); `Analytic` must refuse instead of falling back.
//! The probe is checked on its own for the cases that need no SCF: an ECP
//! molecule, and a basis with g functions (above the 4-centre second-derivative
//! angular momentum of the conda-forge libint2 2.13.1 build, f).
//!
//! If the dispatch were broken the way it most plausibly could be — always
//! analytic, or always finite difference — `hessian_source` and
//! `n_gradient_evaluations` would disagree with the expectations below in
//! every case, not just one.

use ferric_core::basis;
use ferric_core::external_potential::{ExternalPotential, PointCharge};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::libint_max_deriv_order;
use ferric_integrals::operator::Operator;
use ferric_scf::frequencies::{
    harmonic_frequencies, FrequencyConfig, FrequencyReference, HessianMethod, HessianSource,
};
use ferric_scf::hessian::analytic_hessian_available;
use ferric_scf::rhf::RhfConfig;

fn water() -> Molecule {
    Molecule::parse_xyz(
        "3\nwater\nO 0.000000 0.000000 0.117300\n\
         H 0.000000 0.757200 -0.469200\nH 0.000000 -0.757200 -0.469200\n",
        0,
        1,
    )
    .unwrap()
}

fn run(scf: &RhfConfig, fc: &FrequencyConfig) -> ferric_scf::frequencies::FrequencyResult {
    harmonic_frequencies(
        &ParallelContext::default(),
        &water(),
        "sto-3g",
        Operator::coulomb(),
        scf,
        fc,
    )
    .expect("frequencies")
}

fn has_deriv2() -> bool {
    let order = libint_max_deriv_order();
    if order < 2 {
        eprintln!("SKIP: libint2 generated with derivative order {order}; no analytic Hessian");
    }
    order >= 2
}

#[test]
fn auto_is_analytic_for_plain_rhf_and_matches_fd() {
    if !has_deriv2() {
        return;
    }
    let scf = RhfConfig::default();
    let an = run(&scf, &FrequencyConfig::default());
    assert_eq!(an.hessian_source, HessianSource::Analytic);
    assert_eq!(an.n_gradient_evaluations, 0);
    let fd = run(
        &scf,
        &FrequencyConfig {
            hessian: HessianMethod::FiniteDifference,
            ..Default::default()
        },
    );
    assert_eq!(fd.hessian_source, HessianSource::FiniteDifference);
    assert_eq!(fd.n_gradient_evaluations, 18);
    // Same molecule, two constructions: agree to the FD step's truncation.
    let dev = an
        .frequencies
        .iter()
        .zip(&fd.frequencies)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("water/STO-3G analytic vs FD frequencies: max |d| {dev:.3e} cm^-1");
    assert!(
        dev < 1.0,
        "analytic and FD frequencies differ by {dev} cm^-1"
    );
}

#[test]
fn auto_falls_back_for_unsupported_configurations() {
    let ri = RhfConfig {
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        ..Default::default()
    };
    let embedded = RhfConfig {
        external_potential: Some(ExternalPotential {
            point_charges: vec![PointCharge {
                q: 0.5,
                x: 0.0,
                y: 0.0,
                z: 6.0,
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    for (what, scf) in [("RI J/K", ri), ("external potential", embedded)] {
        let r = run(&scf, &FrequencyConfig::default());
        assert_eq!(r.hessian_source, HessianSource::FiniteDifference, "{what}");
        assert_eq!(r.n_gradient_evaluations, 18, "{what}");
        let refused = harmonic_frequencies(
            &ParallelContext::default(),
            &water(),
            "sto-3g",
            Operator::coulomb(),
            &scf,
            &FrequencyConfig {
                hessian: HessianMethod::Analytic,
                ..Default::default()
            },
        );
        assert!(
            refused.is_err(),
            "{what}: Analytic must refuse, not fall back"
        );
    }
}

/// NH2 ²B1 at the CCCBDB experimental geometry.
fn nh2() -> Molecule {
    Molecule::parse_xyz(
        "3\nNH2\nN 0.0 0.0 0.0\n\
         H 0.0 0.803611 -0.634654\nH 0.0 -0.803611 -0.634654\n",
        0,
        2,
    )
    .unwrap()
}

fn open_shell(
    reference: FrequencyReference,
    hessian: HessianMethod,
) -> Result<ferric_scf::frequencies::FrequencyResult, ferric_core::FerricError> {
    harmonic_frequencies(
        &ParallelContext::default(),
        &nh2(),
        "sto-3g",
        Operator::coulomb(),
        &RhfConfig::default(),
        &FrequencyConfig {
            reference,
            hessian,
            ..Default::default()
        },
    )
}

#[test]
fn auto_is_analytic_for_a_uhf_doublet_and_matches_fd() {
    if !has_deriv2() {
        return;
    }
    let an = open_shell(FrequencyReference::Uhf, HessianMethod::Auto).expect("UHF auto");
    assert_eq!(an.hessian_source, HessianSource::Analytic);
    assert_eq!(an.n_gradient_evaluations, 0);
    let fd = open_shell(FrequencyReference::Uhf, HessianMethod::FiniteDifference).expect("UHF FD");
    assert_eq!(fd.hessian_source, HessianSource::FiniteDifference);
    assert_eq!(fd.n_gradient_evaluations, 18);
    let dev = an
        .frequencies
        .iter()
        .zip(&fd.frequencies)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("NH2/STO-3G UHF analytic vs FD frequencies: max |d| {dev:.3e} cm^-1");
    assert!(
        dev < 1.0,
        "UHF analytic and FD frequencies differ by {dev} cm^-1"
    );
}

#[test]
fn rohf_stays_finite_difference_and_analytic_refuses_it() {
    let auto = open_shell(FrequencyReference::Rohf, HessianMethod::Auto).expect("ROHF auto");
    assert_eq!(auto.hessian_source, HessianSource::FiniteDifference);
    assert_eq!(auto.n_gradient_evaluations, 18);
    let err = open_shell(FrequencyReference::Rohf, HessianMethod::Analytic)
        .expect_err("ROHF with Analytic must be refused");
    eprintln!("ROHF analytic refusal: {err}");
    assert!(
        err.to_string().contains("ROHF"),
        "refusal does not name ROHF: {err}"
    );
}

#[test]
fn probe_refuses_g_functions_and_ecps_without_an_scf() {
    if !has_deriv2() {
        return;
    }
    let cfg = RhfConfig::default();
    let mol = water();
    let sto = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    analytic_hessian_available(&mol, &sto, Operator::coulomb(), &cfg)
        .expect("water/STO-3G is supported");

    let qz = basis::bundled("def2-qzvp").unwrap();
    let qz_prep = PreparedBasis::new(&mol, &qz).unwrap();
    assert!(
        qz_prep.max_l() >= 4,
        "def2-QZVP must carry g functions for this check to mean anything"
    );
    let err = analytic_hessian_available(&mol, &qz_prep, Operator::coulomb(), &cfg)
        .expect_err("g functions exceed the second-derivative angular momentum");
    eprintln!("def2-QZVP refusal: {err}");

    let svp = basis::bundled("def2-svp").unwrap();
    let mut hi = Molecule::parse_xyz("2\nHI\nH 0 0 0\nI 0 0 1.61\n", 0, 1).unwrap();
    hi.apply_ecp(&svp);
    let hi_prep = PreparedBasis::new(&hi, &svp).unwrap();
    assert!(
        analytic_hessian_available(&hi, &hi_prep, Operator::coulomb(), &cfg).is_err(),
        "an ECP molecule must be refused"
    );
}
