//! `HessianMethod::AnalyticMovingGrid` (issue #490): the opt-in route from
//! `harmonic_frequencies` to the moving-grid analytic KS Hessian. The Hessian
//! is validated in `ks_moving_grid_hessian.rs`; the CLI and Python wiring
//! (bit for bit against this library entry point, and against finite
//! differences) in `ferric-cli/tests/ks_moving_grid_frequencies.rs` and
//! `ferric-python/tests/test_ks_moving_grid_frequencies.py`. This file pins
//! the strict parser, the route, and the refusals.
//!
//! The refusals fire before the SCF (the test runs in milliseconds; the CLI
//! test checks it through the run log, which holds no SCF iteration).
//!
//! `the_route_reaches_the_moving_grid_hessian` is the per-commit routing
//! check: `harmonic_frequencies(AnalyticMovingGrid)` against a direct
//! `rks_moving_grid_hessian` call on the same SCF, bit for bit. The CLI ==
//! library test cannot catch a wrong route (both sides carry it), and the
//! FD comparison is validation tier.
//!
//! Mutation ledger (each measured to fail): the route sent to the fixed-grid
//! Hessian (`rks_fixed_grid_hessian_parts` in
//! `analytic_frequencies_ks_moving_grid`) fails
//! `the_route_reaches_the_moving_grid_hessian` (first entry
//! -0.10702 vs -0.10920 Ha/Bohr^2, measured 2026-10-23); the RHF
//! configuration checklist dropped from the preflight fails the MOM case (the
//! SCF runs and the RHF Hessian's post-SCF message lacks the KS prefix); the
//! `check_scf_correction_config` branch deleted fails
//! `the_dispersion_driver_refuses_it`; the preflight call
//! deleted from `harmonic_frequencies_ks_moving_grid` fails the HF and
//! pruned-grid cases here (the post-SCF messages differ) and every preflight
//! case of the CLI test (SCF iterations logged).

use ferric_core::basis;
use ferric_core::external_potential::{ExternalPotential, PointCharge};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::operator::Operator;
use ferric_scf::frequencies::{
    check_scf_correction_config, harmonic_frequencies, FrequencyConfig, FrequencyReference,
    HessianMethod, HessianSource,
};
use ferric_scf::hessian::rks_moving_grid_hessian;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

const WATER: &str = "3\nwater\nO 0 0 0.1178\nH 0 0.7555 -0.4712\nH 0 -0.7555 -0.4712\n";
const PREFIX: &str = "the analytic Kohn-Sham Hessian (fixed or moving grid) does not support";

#[test]
fn the_parser_accepts_the_new_value_and_lists_it() {
    assert_eq!(
        HessianMethod::parse_config_str("analytic-moving-grid").unwrap(),
        HessianMethod::AnalyticMovingGrid
    );
    let err = HessianMethod::parse_config_str("moving-grid").unwrap_err();
    assert!(
        err.to_string().contains("\"analytic-moving-grid\""),
        "{err}"
    );
    assert_eq!(
        HessianSource::AnalyticMovingGrid.label(),
        "analytic-moving-grid"
    );
    assert_eq!(HessianMethod::default(), HessianMethod::Auto);
}

fn run(mol: &Molecule, basis_name: &str, cfg: RhfConfig, reference: FrequencyReference) -> String {
    run_op(mol, basis_name, cfg, reference, Operator::coulomb())
}

fn run_op(
    mol: &Molecule,
    basis_name: &str,
    cfg: RhfConfig,
    reference: FrequencyReference,
    op: Operator,
) -> String {
    harmonic_frequencies(
        &ParallelContext::default(),
        mol,
        basis_name,
        op,
        &cfg,
        &FrequencyConfig {
            reference,
            hessian: HessianMethod::AnalyticMovingGrid,
            ..Default::default()
        },
    )
    .expect_err("must be refused")
    .to_string()
}

fn ks(xc: &str) -> RhfConfig {
    RhfConfig {
        xc: Some(xc.into()),
        ..Default::default()
    }
}

#[test]
fn every_refusal_is_named() {
    let water = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let h2_triplet = Molecule::parse_xyz("2\nh2\nH 0 0 0\nH 0 0 0.74\n", 0, 3).unwrap();
    let svp = basis::bundled("def2-svp").unwrap();
    let mut hi = Molecule::parse_xyz("2\nHI\nH 0 0 0\nI 0 0 1.75\n", 0, 1).unwrap();
    hi.apply_ecp(&svp);
    let rhf = FrequencyReference::Rhf;
    let pruned = RhfConfig {
        dft_grid: Some(AtomicGridConfig {
            prune: Some(ferric_dft::prune::PruneScheme::NwchemLike),
            ..Default::default()
        }),
        ..ks("LDA")
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
        ..ks("LDA")
    };
    // (label, message, fragments it must carry)
    let cases: Vec<(&str, String, Vec<&str>)> = vec![
        (
            "HF",
            run(&water, "sto-3g", RhfConfig::default(), rhf),
            vec!["Hartree-Fock has no XC grid"],
        ),
        (
            "UKS",
            run(&h2_triplet, "sto-3g", ks("LDA"), FrequencyReference::Uhf),
            vec!["open-shell (Uhf", "spin-polarized XC kernel"],
        ),
        (
            "ROKS",
            run(&h2_triplet, "sto-3g", ks("LDA"), FrequencyReference::Rohf),
            vec!["open-shell (Rohf"],
        ),
        (
            "meta-GGA",
            run(&water, "sto-3g", ks("SCAN"), rhf),
            vec![PREFIX, "a meta-GGA ('SCAN')"],
        ),
        (
            "VV10",
            run(&water, "sto-3g", ks("wB97X-V"), rhf),
            vec![PREFIX, "VV10"],
        ),
        (
            "RSH",
            run(&water, "sto-3g", ks("HSE06"), rhf),
            vec![PREFIX, "a range-separated hybrid ('HSE06')"],
        ),
        (
            "ECP",
            run(&hi, "def2-svp", ks("LDA"), rhf),
            vec![PREFIX, "an ECP or an external potential"],
        ),
        (
            "external potential",
            run(&water, "sto-3g", embedded, rhf),
            vec![PREFIX, "an ECP or an external potential"],
        ),
        (
            "pruned grid",
            run(&water, "sto-3g", pruned, rhf),
            vec![PREFIX, "an angular-pruned XC grid"],
        ),
        (
            "MOM",
            run(
                &water,
                "sto-3g",
                RhfConfig {
                    mom_after_iter: 3,
                    ..ks("LDA")
                },
                rhf,
            ),
            vec![PREFIX, "maximum-overlap (MOM) occupations"],
        ),
        (
            "smearing",
            run(
                &water,
                "sto-3g",
                RhfConfig {
                    smearing_sigma: Some(1e-3),
                    ..ks("LDA")
                },
                rhf,
            ),
            vec![PREFIX, "fractional occupations"],
        ),
        (
            "erf operator",
            run_op(&water, "sto-3g", ks("LDA"), rhf, Operator::erf(0.4)),
            vec![PREFIX, "two-electron operator"],
        ),
    ];
    for (label, msg, fragments) in cases {
        for f in fragments {
            assert!(msg.contains(f), "{label}: {msg:?} lacks {f:?}");
        }
        eprintln!("{label}: {msg}");
    }
}

#[test]
fn the_dispersion_driver_refuses_it() {
    let err = check_scf_correction_config(&FrequencyConfig {
        hessian: HessianMethod::AnalyticMovingGrid,
        ..Default::default()
    })
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("hessian = \"analytic-moving-grid\" is not available"),
        "{err}"
    );
}

#[test]
fn the_route_reaches_the_moving_grid_hessian() {
    let order = ferric_integrals::engine::libint_max_deriv_order();
    if order < 2 {
        eprintln!("SKIP: libint2 generated with derivative order {order}; no analytic Hessian");
        return;
    }
    let water = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let cfg = RhfConfig {
        density_conv: 1e-9,
        ..ks("PBE")
    };
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let routed = harmonic_frequencies(
        &ctx,
        &water,
        "sto-3g",
        op,
        &cfg,
        &FrequencyConfig {
            hessian: HessianMethod::AnalyticMovingGrid,
            ..Default::default()
        },
    )
    .expect("moving-grid frequencies");
    assert_eq!(routed.hessian_source, HessianSource::AnalyticMovingGrid);
    assert_eq!(routed.n_gradient_evaluations, 0);

    let bs = basis::bundled("sto-3g").unwrap();
    let prep = ferric_integrals::basis_bridge::PreparedBasis::new(&water, &bs).unwrap();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let res = solve_rhf(&ctx, &water, &prep, op, &bounds, &cfg).unwrap();
    let direct = rks_moving_grid_hessian(&ctx, &water, &prep, op, &bounds, &res, &cfg).unwrap();
    assert_eq!(routed.energy.to_bits(), res.energy.to_bits(), "energy");
    let miss = routed
        .cartesian_hessian
        .iter()
        .zip(direct.iter())
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("routed vs direct moving-grid Hessian: max |dH| {miss:e}");
    for (i, (a, b)) in routed
        .cartesian_hessian
        .iter()
        .zip(direct.iter())
        .enumerate()
    {
        assert_eq!(a.to_bits(), b.to_bits(), "Hessian[{i}]: {a:e} vs {b:e}");
    }
}
