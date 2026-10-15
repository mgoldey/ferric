//! Which Hessian `harmonic_frequencies` builds (`FrequencyConfig::hessian`).
//!
//! `Auto` must take the analytic RHF Hessian exactly where
//! `analytic_hessian_available` accepts the system, the analytic UHF Hessian
//! for a UHF reference where `analytic_uhf_hessian_available` accepts it, the
//! analytic ROHF Hessian for a ROHF reference where
//! `analytic_rohf_hessian_available` accepts it, and fall back to finite
//! differences everywhere else (always for ROKS and the other KS references);
//! `Analytic` must refuse instead of falling back.
//! The probe is checked on its own for the cases that need no SCF: an ECP
//! molecule whose core charge was never applied (`Molecule::apply_ecp` not run),
//! and a basis with g functions (above the 4-centre second-derivative
//! angular momentum of the conda-forge libint2 2.13.1 build, f). An ECP molecule
//! with the ECP applied is analytic (`ecp_molecule_is_analytic_and_matches_fd`).
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
use ferric_scf::hessian::{
    analytic_hessian_available, analytic_rohf_hessian_available, analytic_uhf_hessian_available,
    rohf_hessian, uhf_hessian,
};
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
    // Fitted J/K next to an external potential is untested and refused.
    let ri = RhfConfig {
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        external_potential: Some(ExternalPotential {
            point_charges: vec![PointCharge {
                q: 0.3,
                x: 5.0,
                y: 0.0,
                z: 0.0,
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    // Point charges, smeared charges and a uniform field are analytic (see
    // `embedded_point_charges_are_analytic`, `smeared_charges_are_analytic_and_match_fd`,
    // `uniform_field_is_analytic_and_matches_fd`); implicit solvation is not.
    let solvated = RhfConfig {
        cosmo: Some(ferric_scf::cosmo::CosmoConfig::default()),
        ..Default::default()
    };
    for (what, scf) in [
        ("RI J/K with an external potential", ri),
        ("implicit solvation", solvated),
    ] {
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

#[test]
fn ri_j_with_exact_k_is_analytic_where_the_engines_exist_and_falls_back_otherwise() {
    // RI-J with exact K: analytic wherever the four-centre second-derivative
    // engines exist (the DF second derivatives are a stencil of first-derivative
    // blocks, available everywhere); a first-derivative libint (the wheel)
    // makes the probe refuse and Auto fall back to finite differences of the
    // DF gradient, with no error. Either branch checks the frequencies agree.
    let scf = RhfConfig {
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some(String::new()),
        ..Default::default()
    };
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&water(), &bs).unwrap();
    let probe = analytic_hessian_available(&water(), &prep, Operator::coulomb(), &scf);
    let auto = run(&scf, &FrequencyConfig::default());
    let fd = run(
        &scf,
        &FrequencyConfig {
            hessian: HessianMethod::FiniteDifference,
            ..Default::default()
        },
    );
    assert_eq!(fd.hessian_source, HessianSource::FiniteDifference);
    match &probe {
        Ok(()) => {
            assert_eq!(auto.hessian_source, HessianSource::Analytic);
            assert_eq!(auto.n_gradient_evaluations, 0);
        }
        Err(e) => {
            eprintln!("RI-J probe refused on this libint: {e}");
            assert_eq!(auto.hessian_source, HessianSource::FiniteDifference);
            assert_eq!(auto.n_gradient_evaluations, 18);
        }
    }
    let dev = auto
        .frequencies
        .iter()
        .zip(&fd.frequencies)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("water/STO-3G RI-J: Auto vs FD frequencies max |d| {dev:.3e} cm^-1");
    assert!(dev < 1.0, "frequencies differ by {dev} cm^-1");
}

#[test]
fn ri_k_is_analytic_alone_and_with_ri_j_and_matches_fd() {
    // Both routes need a libint with second derivatives (one-electron and
    // overlap skeletons; RI-K with exact J also the four-centre Coulomb
    // skeleton). On a first-derivative libint the probe refuses and Auto runs
    // finite differences; either branch checks the frequencies against finite
    // differences of the DF gradient.
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&water(), &bs).unwrap();
    for (what, j_aux) in [
        ("RI-K, exact J", String::new()),
        ("RI-J + RI-K", "def2-universal-jkfit".to_string()),
    ] {
        let scf = RhfConfig {
            df_j_aux: Some(j_aux),
            df_k_aux: Some("def2-universal-jkfit".into()),
            ..Default::default()
        };
        let probe = analytic_hessian_available(&water(), &prep, Operator::coulomb(), &scf);
        let auto = run(&scf, &FrequencyConfig::default());
        let fd = run(
            &scf,
            &FrequencyConfig {
                hessian: HessianMethod::FiniteDifference,
                ..Default::default()
            },
        );
        match &probe {
            Ok(()) => {
                assert_eq!(auto.hessian_source, HessianSource::Analytic, "{what}");
                assert_eq!(auto.n_gradient_evaluations, 0, "{what}");
            }
            Err(e) => {
                eprintln!("{what}: probe refused on this libint: {e}");
                assert_eq!(auto.hessian_source, HessianSource::FiniteDifference);
            }
        }
        let dev = auto
            .frequencies
            .iter()
            .zip(&fd.frequencies)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("water/STO-3G {what}: Auto vs FD frequencies max |d| {dev:.3e} cm^-1");
        assert!(dev < 1.0, "{what}: frequencies differ by {dev} cm^-1");
    }
}

#[test]
fn embedded_point_charges_are_analytic() {
    if !has_deriv2() {
        return;
    }
    let scf = RhfConfig {
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
    let r = run(&scf, &FrequencyConfig::default());
    assert_eq!(r.hessian_source, HessianSource::Analytic);
    assert_eq!(r.n_gradient_evaluations, 0);
}

#[test]
fn smeared_charges_are_analytic_and_match_fd() {
    if !has_deriv2() {
        return;
    }
    let scf = RhfConfig {
        external_potential: Some(ExternalPotential {
            smeared_charges: vec![ferric_core::external_potential::SmearedCharge {
                q: 0.5,
                x: 0.0,
                y: 0.0,
                z: 4.0,
                width: 1.0,
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
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
    let dev = an
        .frequencies
        .iter()
        .zip(&fd.frequencies)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("water/STO-3G smeared: analytic vs FD frequencies max |d| {dev:.3e} cm^-1");
    assert!(dev < 1.0, "frequencies differ by {dev} cm^-1");
}

#[test]
fn uniform_field_is_analytic_and_matches_fd() {
    if !has_deriv2() {
        return;
    }
    let scf = RhfConfig {
        external_potential: Some(ExternalPotential {
            field: Some([0.01, -0.02, 0.03]),
            ..Default::default()
        }),
        ..Default::default()
    };
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
    let dev = an
        .frequencies
        .iter()
        .zip(&fd.frequencies)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("water/STO-3G field: analytic vs FD frequencies max |d| {dev:.3e} cm^-1");
    assert!(dev < 1.0, "frequencies differ by {dev} cm^-1");
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
fn uhf_density_fitting_is_analytic_and_matches_fd() {
    density_fitting_is_analytic_and_matches_fd(FrequencyReference::Uhf);
}

#[test]
fn rohf_density_fitting_is_analytic_and_matches_fd() {
    density_fitting_is_analytic_and_matches_fd(FrequencyReference::Rohf);
}

fn density_fitting_is_analytic_and_matches_fd(reference: FrequencyReference) {
    // UHF with RI-J, RI-K or both: analytic where the second-derivative
    // engines exist (one-electron/overlap skeletons), 0 gradients; on a
    // first-derivative libint the probe refuses and Auto runs finite
    // differences of the DF gradient. Either branch checks the frequencies
    // against finite differences.
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&nh2(), &bs).unwrap();
    for (what, j_aux, k_aux) in [
        ("RI-J, exact K", "def2-universal-jkfit", ""),
        ("RI-K, exact J", "", "def2-universal-jkfit"),
        (
            "RI-J + RI-K",
            "def2-universal-jkfit",
            "def2-universal-jkfit",
        ),
    ] {
        let scf = RhfConfig {
            df_j_aux: Some(j_aux.into()),
            df_k_aux: Some(k_aux.into()),
            ..Default::default()
        };
        let run = |hessian| {
            harmonic_frequencies(
                &ParallelContext::default(),
                &nh2(),
                "sto-3g",
                Operator::coulomb(),
                &scf,
                &FrequencyConfig {
                    reference,
                    hessian,
                    ..Default::default()
                },
            )
            .expect(what)
        };
        let probe = match reference {
            FrequencyReference::Rohf => {
                analytic_rohf_hessian_available(&nh2(), &prep, Operator::coulomb(), &scf)
            }
            _ => analytic_uhf_hessian_available(&nh2(), &prep, Operator::coulomb(), &scf),
        };
        let auto = run(HessianMethod::Auto);
        let fd = run(HessianMethod::FiniteDifference);
        assert_eq!(fd.hessian_source, HessianSource::FiniteDifference, "{what}");
        match &probe {
            Ok(()) => {
                assert_eq!(auto.hessian_source, HessianSource::Analytic, "{what}");
                assert_eq!(auto.n_gradient_evaluations, 0, "{what}");
            }
            Err(e) => {
                eprintln!("{what}: probe refused on this libint: {e}");
                assert_eq!(auto.hessian_source, HessianSource::FiniteDifference);
            }
        }
        let dev = auto
            .frequencies
            .iter()
            .zip(&fd.frequencies)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        eprintln!(
            "NH2/STO-3G {reference:?} {what}: Auto vs FD frequencies max |d| {dev:.3e} cm^-1"
        );
        // Measured 8.6e-2 cm^-1 for every route (the same as the exact-integral
        // UHF test above): the truncation of the central difference at the
        // default step, not an analytic error. Bar 2x the measured level.
        assert!(dev < 0.2, "{what}: frequencies differ by {dev} cm^-1");
    }
}

#[test]
fn uhf_density_fitting_falls_back_before_the_scf_for_ecp() {
    // UHF/ROHF RI combined with an ECP is refused by the probe (needs only
    // the molecule and the basis).
    if !has_deriv2() {
        return;
    }
    let scf = RhfConfig {
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        ..Default::default()
    };
    let svp = basis::bundled("def2-svp").unwrap();
    let mut hi = Molecule::parse_xyz("2\nHI\nH 0 0 0\nI 0 0 1.75\n", 0, 1).unwrap();
    hi.apply_ecp(&svp);
    let hi_prep = PreparedBasis::new(&hi, &svp).unwrap();
    let e = analytic_uhf_hessian_available(&hi, &hi_prep, Operator::coulomb(), &scf)
        .expect_err("UHF RI + ECP must be refused by the probe")
        .to_string();
    assert!(e.contains("effective core potential"), "{e}");

    let e = analytic_rohf_hessian_available(&hi, &hi_prep, Operator::coulomb(), &scf)
        .expect_err("ROHF RI + ECP must be refused by the probe")
        .to_string();
    assert!(e.contains("effective core potential"), "{e}");

    // Auto runs the probe BEFORE the SCF and falls back to finite differences
    // of the DF gradient (no error); Analytic stays an error.
    let run_hi = |hessian| {
        harmonic_frequencies(
            &ParallelContext::default(),
            &hi,
            "def2-svp",
            Operator::coulomb(),
            &scf,
            &FrequencyConfig {
                reference: FrequencyReference::Uhf,
                hessian,
                ..Default::default()
            },
        )
    };
    let auto = run_hi(HessianMethod::Auto).expect("Auto must fall back for UHF RI + ECP");
    assert_eq!(auto.hessian_source, HessianSource::FiniteDifference);
    assert_eq!(auto.n_gradient_evaluations, 12);
    assert!(run_hi(HessianMethod::Analytic).is_err());
}

#[test]
fn auto_is_analytic_for_a_rohf_doublet_and_matches_fd() {
    if !has_deriv2() {
        return;
    }
    let an = open_shell(FrequencyReference::Rohf, HessianMethod::Auto).expect("ROHF auto");
    assert_eq!(an.hessian_source, HessianSource::Analytic);
    assert_eq!(an.n_gradient_evaluations, 0);
    let explicit =
        open_shell(FrequencyReference::Rohf, HessianMethod::Analytic).expect("ROHF analytic");
    assert_eq!(explicit.hessian_source, HessianSource::Analytic);
    let fd =
        open_shell(FrequencyReference::Rohf, HessianMethod::FiniteDifference).expect("ROHF FD");
    assert_eq!(fd.hessian_source, HessianSource::FiniteDifference);
    assert_eq!(fd.n_gradient_evaluations, 18);
    let dev = an
        .frequencies
        .iter()
        .zip(&fd.frequencies)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("NH2/STO-3G ROHF analytic vs FD frequencies: max |d| {dev:.3e} cm^-1");
    assert!(
        dev < 1.0,
        "ROHF analytic and FD frequencies differ by {dev} cm^-1"
    );
    // The ROHF Hessian is not the UHF one: the two references have different
    // orbitals and energies, so a dispatch that sent ROHF to the UHF path
    // would reproduce the UHF number, not this one.
    let uhf = open_shell(FrequencyReference::Uhf, HessianMethod::Analytic).expect("UHF analytic");
    let split = an
        .frequencies
        .iter()
        .zip(&uhf.frequencies)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("NH2/STO-3G ROHF vs UHF analytic frequencies: max |d| {split:.3e} cm^-1");
    assert!(
        split > 5.0 * dev,
        "ROHF {split} cm^-1 from UHF vs FD gap {dev}"
    );
}

/// ROKS and every other KS reference stay finite-difference: `Auto` falls back,
/// `Analytic` refuses (before any SCF).
#[test]
fn roks_stays_finite_difference_and_analytic_refuses_it() {
    let scf = RhfConfig {
        xc: Some("PBE".into()),
        ..Default::default()
    };
    let run_roks = |hessian| {
        harmonic_frequencies(
            &ParallelContext::default(),
            &nh2(),
            "sto-3g",
            Operator::coulomb(),
            &scf,
            &FrequencyConfig {
                reference: FrequencyReference::Rohf,
                hessian,
                ..Default::default()
            },
        )
    };
    let auto = run_roks(HessianMethod::Auto).expect("ROKS auto");
    assert_eq!(auto.hessian_source, HessianSource::FiniteDifference);
    assert_eq!(auto.n_gradient_evaluations, 18);
    let err = run_roks(HessianMethod::Analytic).expect_err("ROKS with Analytic must be refused");
    eprintln!("ROKS analytic refusal: {err}");
    let msg = err.to_string();
    assert!(msg.contains("ROHF") && msg.contains("Kohn-Sham"), "{msg}");
}

/// The probe and the entry point agree on what ROHF refuses, with no SCF for
/// the probe.
#[test]
fn rohf_probe_refuses_what_the_hessian_does_not_cover() {
    if !has_deriv2() {
        return;
    }
    let mol = nh2();
    let prep = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
    let op = Operator::coulomb();
    analytic_rohf_hessian_available(&mol, &prep, op, &RhfConfig::default())
        .expect("NH2/STO-3G ROHF is supported");
    let charges = RhfConfig {
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
    let cosx = RhfConfig {
        k_builder: Some("cosx".into()),
        ..Default::default()
    };
    let ks = RhfConfig {
        xc: Some("PBE".into()),
        ..Default::default()
    };
    for (what, cfg) in [("external charges", charges), ("COSX", cosx), ("KS", ks)] {
        let err = analytic_rohf_hessian_available(&mol, &prep, op, &cfg)
            .expect_err(&format!("{what} must be refused by the ROHF probe"));
        eprintln!("ROHF probe, {what}: {err}");
        assert!(err.to_string().contains("ROHF"), "{what}: {err}");
    }
    // An ECP molecule with its core charge applied: analytic for RHF/UHF, but
    // not validated for ROHF.
    let svp = basis::bundled("def2-svp").unwrap();
    let mut hi = Molecule::parse_xyz("2\nHI\nH 0 0 0\nI 0 0 1.61\n", 0, 3).unwrap();
    hi.apply_ecp(&svp);
    let hi_prep = PreparedBasis::new(&hi, &svp).unwrap();
    let err = analytic_rohf_hessian_available(&hi, &hi_prep, op, &RhfConfig::default())
        .expect_err("an ECP must be refused by the ROHF probe");
    eprintln!("ROHF probe, ECP: {err}");
    assert!(
        err.to_string().contains("effective core potential"),
        "{err}"
    );
}

/// A semicanonicalized ROHF result (`Spin::Unrestricted` with the ROHF spin
/// Focks) is mapped back to ROHF orbitals by `rohf_hessian`; a plain UHF
/// result is refused rather than treated as one. NH2 has one open orbital, the
/// CH2 triplet two (the open block is a rotation, not a single vector).
#[test]
fn semicanonicalized_result_gives_the_rohf_hessian_and_uhf_is_refused() {
    if !has_deriv2() {
        return;
    }
    let ch2 = Molecule::parse_xyz(
        "3\nCH2 triplet\nC 0.0 0.0 0.0\nH 0.0 0.98 -0.55\nH 0.01 -0.97 -0.56\n",
        0,
        3,
    )
    .unwrap();
    for (name, mol) in [("NH2", nh2()), ("CH2 triplet", ch2)] {
        let op = Operator::coulomb();
        let prep = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
        let bounds = ferric_scf::screening::SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let cfg = RhfConfig {
            density_conv: 1e-10,
            energy_conv: 1e-12,
            ..Default::default()
        };
        let rohf = ferric_scf::rohf::solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
        let direct = rohf_hessian(&ctx, &mol, &prep, op, &bounds, &rohf, &cfg).unwrap();
        let semi = ferric_scf::semicanonical::unrestricted_reference(&mol, &rohf).unwrap();
        assert_eq!(semi.spin, ferric_scf::result::Spin::Unrestricted);
        let mapped = rohf_hessian(&ctx, &mol, &prep, op, &bounds, &semi, &cfg).unwrap();
        let dev = (&direct - &mapped)
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()));
        eprintln!("{name}: semicanonicalized vs ROHF Hessian max |d| {dev:.3e}");
        assert!(dev < 1e-8, "{name}: mapped-back Hessian differs by {dev}");
        // Negative controls: uhf_hessian refuses it, and the Hessian is not zero.
        assert!(uhf_hessian(&ctx, &mol, &prep, op, &bounds, &semi, &cfg).is_err());
        assert!(direct.iter().any(|v| v.abs() > 0.1));
        // A semicanonicalized result without beta orbitals cannot be mapped: a
        // typed error with a readable message.
        let mut broken = semi.clone().into_owned();
        broken.mos_beta = None;
        let err = rohf_hessian(&ctx, &mol, &prep, op, &bounds, &broken, &cfg)
            .expect_err("no beta orbitals");
        let msg = err.to_string();
        assert!(
            msg.contains("cannot map a semicanonicalized result back to ROHF orbitals: no β"),
            "{msg}"
        );
        assert!(!msg.contains("  "), "stray whitespace in: {msg}");
        let uhf = ferric_scf::uhf::solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
        let err = rohf_hessian(&ctx, &mol, &prep, op, &bounds, &uhf, &cfg)
            .expect_err("a UHF result is not a ROHF result");
        assert!(err.to_string().contains("ROHF"), "{err}");
    }
}

/// The ROHF frequency driver runs the full availability probe BEFORE the SCF:
/// an ECP molecule is refused without solving anything.
#[test]
fn rohf_frequency_driver_refuses_an_ecp_before_the_scf() {
    let svp = basis::bundled("def2-svp").unwrap();
    let mut hi = Molecule::parse_xyz("2\nHI\nH 0 0 0\nI 0 0 1.61\n", 0, 3).unwrap();
    hi.apply_ecp(&svp);
    let t0 = std::time::Instant::now();
    let err = ferric_scf::frequencies::harmonic_frequencies_analytic_rohf(
        &ParallelContext::default(),
        &hi,
        "def2-svp",
        Operator::coulomb(),
        &RhfConfig::default(),
    )
    .expect_err("ROHF analytic frequencies with an ECP must be refused");
    assert!(
        err.to_string().contains("effective core potential"),
        "{err}"
    );
    assert!(
        t0.elapsed().as_secs_f64() < 5.0,
        "refusal came after an SCF"
    );
}

#[test]
fn probe_refuses_g_functions_and_unapplied_ecps_without_an_scf() {
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
    let hi_xyz = "2\nHI\nH 0 0 0\nI 0 0 1.61\n";
    let mut hi = Molecule::parse_xyz(hi_xyz, 0, 1).unwrap();
    hi.apply_ecp(&svp);
    let hi_prep = PreparedBasis::new(&hi, &svp).unwrap();
    analytic_hessian_available(&hi, &hi_prep, Operator::coulomb(), &cfg)
        .expect("an ECP molecule with its core charge applied is supported");

    // The ECP basis without `apply_ecp` would use bare nuclear charges against
    // an ECP potential: refused, never a silent wrong Hessian.
    let bare = Molecule::parse_xyz(hi_xyz, 0, 1).unwrap();
    let bare_prep = PreparedBasis::new(&bare, &svp).unwrap();
    let err = analytic_hessian_available(&bare, &bare_prep, Operator::coulomb(), &cfg)
        .expect_err("an ECP not applied to the molecule must be refused");
    assert!(err.to_string().contains("apply_ecp"), "{err}");
}

#[test]
fn ri_j_with_an_ecp_falls_back_before_the_scf() {
    // RI-J combined with an ECP is refused by the PRE-SCF probe (it needs only
    // the molecule and the basis), so Auto runs finite differences of the DF
    // gradient instead of failing after the SCF, and Analytic is an error.
    // Hypothesis: the probe refuses -> Auto = FD with 12 gradients; had the
    // refusal been post-SCF, Auto would return the error instead.
    if !has_deriv2() {
        return;
    }
    let svp = basis::bundled("def2-svp").unwrap();
    let mut hi = Molecule::parse_xyz("2\nHI\nH 0 0 0\nI 0 0 1.75\n", 0, 1).unwrap();
    hi.apply_ecp(&svp);
    let scf = RhfConfig {
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some(String::new()),
        ..Default::default()
    };
    let prep = PreparedBasis::new(&hi, &svp).unwrap();
    let probe = analytic_hessian_available(&hi, &prep, Operator::coulomb(), &scf);
    let msg = probe
        .expect_err("RI-J + ECP must be refused by the probe")
        .to_string();
    assert!(msg.contains("effective core potential"), "{msg}");
    let run = |hessian| {
        harmonic_frequencies(
            &ParallelContext::default(),
            &hi,
            "def2-svp",
            Operator::coulomb(),
            &scf,
            &FrequencyConfig {
                hessian,
                ..Default::default()
            },
        )
    };
    let auto = run(HessianMethod::Auto).expect("Auto must fall back, not fail");
    assert_eq!(auto.hessian_source, HessianSource::FiniteDifference);
    assert_eq!(auto.n_gradient_evaluations, 12);
    assert!(run(HessianMethod::Analytic).is_err());
}

#[test]
fn ecp_molecule_is_analytic_and_matches_fd() {
    if !has_deriv2() {
        return;
    }
    let svp = basis::bundled("def2-svp").unwrap();
    let mut hi = Molecule::parse_xyz("2\nHI\nH 0 0 0\nI 0 0 1.75\n", 0, 1).unwrap();
    hi.apply_ecp(&svp);
    let freq = |hessian| {
        harmonic_frequencies(
            &ParallelContext::default(),
            &hi,
            "def2-svp",
            Operator::coulomb(),
            &RhfConfig::default(),
            &FrequencyConfig {
                hessian,
                ..Default::default()
            },
        )
        .expect("HI frequencies")
    };
    let auto = freq(HessianMethod::Auto);
    assert_eq!(auto.hessian_source, HessianSource::Analytic);
    assert_eq!(auto.n_gradient_evaluations, 0);
    let fd = freq(HessianMethod::FiniteDifference);
    assert_eq!(fd.hessian_source, HessianSource::FiniteDifference);
    assert_eq!(fd.n_gradient_evaluations, 12);
    let dev = (auto.frequencies[0] - fd.frequencies[0]).abs();
    eprintln!(
        "HI/def2-SVP analytic {:.3} vs FD {:.3} cm^-1",
        auto.frequencies[0], fd.frequencies[0]
    );
    assert!(
        dev < 0.5,
        "analytic and FD HI frequencies differ by {dev} cm^-1"
    );
}
