//! `HessianMethod::AnalyticMovingGrid` (issue #490): the route from
//! `harmonic_frequencies` to the moving-grid analytic KS Hessian, which is
//! also what `HessianMethod::Auto` runs for a closed-shell KS reference
//! wherever `rks_moving_grid_hessian_available` accepts the system. The
//! Hessian is validated in `ks_moving_grid_hessian.rs`; the CLI and Python
//! wiring (bit for bit against this library entry point, and against finite
//! differences) in `ferric-cli/tests/ks_moving_grid_frequencies.rs` and
//! `ferric-python/tests/test_ks_moving_grid_frequencies.py`. This file pins
//! the strict parser, the route, the refusals, and the `Auto` dispatch.
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
//! `auto_is_the_moving_grid_hessian_for_closed_shell_ks`: `Auto` ==
//! `AnalyticMovingGrid` bit for bit (water/STO-3G, LDA and PBE).
//! `auto_falls_back_to_finite_differences_where_unsupported`: `Auto` ==
//! `FiniteDifference` bit for bit for meta-GGA (SCAN), an open-shell UKS
//! reference and a pruned grid (H2/STO-3G). The dispersion surface is a
//! separate driver (`harmonic_frequencies_with_scf_correction`, always finite
//! differences); its `Auto` is pinned through the CLI and Python.
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
//! case of the CLI test (SCF iterations logged). `Auto` routed back to finite
//! differences for KS (the `Auto` arm of `analytic_route` deleted) fails
//! `auto_is_the_moving_grid_hessian_for_closed_shell_ks` (`hessian_source`
//! finite-difference); `Auto` sent to the moving-grid route for every KS
//! reference without the support predicate or the fallback fails
//! `auto_falls_back_to_finite_differences_where_unsupported` (SCAN and UKS
//! raise the moving-grid refusal instead of running finite differences).
//! The response-table size check dropped from
//! `rks_moving_grid_hessian_available` fails
//! `auto_sends_a_system_above_the_response_table_cap_to_fd_before_the_scf`
//! (the explicit route runs its SCF and returns `ScfConvergence`; `Auto`
//! reaches the Hessian's own post-SCF refusal and errors).

use ferric_core::basis;
use ferric_core::external_potential::{ExternalPotential, PointCharge};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::operator::Operator;
use ferric_scf::frequencies::{
    check_scf_correction_config, harmonic_frequencies, FrequencyConfig, FrequencyReference,
    FrequencyResult, HessianMethod, HessianSource,
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

fn has_deriv2() -> bool {
    let order = ferric_integrals::engine::libint_max_deriv_order();
    if order < 2 {
        eprintln!("SKIP: libint2 generated with derivative order {order}; no analytic Hessian");
    }
    order >= 2
}

fn freq(
    mol: &Molecule,
    cfg: &RhfConfig,
    reference: FrequencyReference,
    hessian: HessianMethod,
) -> FrequencyResult {
    harmonic_frequencies(
        &ParallelContext::default(),
        mol,
        "sto-3g",
        Operator::coulomb(),
        cfg,
        &FrequencyConfig {
            reference,
            hessian,
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| panic!("{hessian:?} frequencies: {e}"))
}

/// Bit-for-bit identity of two frequency results (both ran in this process
/// on the same rayon pool, so the same code gives the same bits).
fn assert_same(a: &FrequencyResult, b: &FrequencyResult, what: &str) {
    assert_eq!(a.hessian_source, b.hessian_source, "{what}: source");
    assert_eq!(
        a.n_gradient_evaluations, b.n_gradient_evaluations,
        "{what}: gradient count"
    );
    assert_eq!(a.energy.to_bits(), b.energy.to_bits(), "{what}: energy");
    for (i, (x, y)) in a
        .cartesian_hessian
        .iter()
        .zip(b.cartesian_hessian.iter())
        .enumerate()
    {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{what}: Hessian[{i}] {x:e} vs {y:e}"
        );
    }
    assert_eq!(a.frequencies.len(), b.frequencies.len(), "{what}: modes");
    for (x, y) in a.frequencies.iter().zip(&b.frequencies) {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: frequency {x} vs {y}");
    }
}

#[test]
fn auto_is_the_moving_grid_hessian_for_closed_shell_ks() {
    if !has_deriv2() {
        return;
    }
    let water = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let rhf = FrequencyReference::Rhf;
    for xc in ["LDA", "PBE"] {
        let cfg = ks(xc);
        let auto = freq(&water, &cfg, rhf, HessianMethod::Auto);
        let moving = freq(&water, &cfg, rhf, HessianMethod::AnalyticMovingGrid);
        assert_eq!(
            auto.hessian_source,
            HessianSource::AnalyticMovingGrid,
            "{xc}"
        );
        assert_eq!(auto.n_gradient_evaluations, 0, "{xc}");
        assert_same(&auto, &moving, xc);
        eprintln!(
            "{xc}: Auto == AnalyticMovingGrid, {:?} cm^-1",
            auto.frequencies
        );
    }
}

#[test]
fn auto_falls_back_to_finite_differences_where_unsupported() {
    let h2 = Molecule::parse_xyz("2\nh2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
    let pruned = RhfConfig {
        dft_grid: Some(AtomicGridConfig {
            prune: Some(ferric_dft::prune::PruneScheme::NwchemLike),
            ..Default::default()
        }),
        ..ks("LDA")
    };
    let cases = [
        ("meta-GGA (SCAN)", ks("SCAN"), FrequencyReference::Rhf),
        ("open-shell UKS", ks("LDA"), FrequencyReference::Uhf),
        ("pruned grid", pruned, FrequencyReference::Rhf),
    ];
    for (label, cfg, reference) in cases {
        let auto = freq(&h2, &cfg, reference, HessianMethod::Auto);
        let fd = freq(&h2, &cfg, reference, HessianMethod::FiniteDifference);
        assert_eq!(
            auto.hessian_source,
            HessianSource::FiniteDifference,
            "{label}"
        );
        assert_eq!(auto.n_gradient_evaluations, 12, "{label}");
        assert_same(&auto, &fd, label);
        eprintln!(
            "{label}: Auto == FiniteDifference, {:?} cm^-1",
            auto.frequencies
        );
    }
}

/// The response's resident AO table is the one refusal that depends on the
/// system's SIZE (basis functions x grid points), so `Auto` must decide it
/// before the SCF, through the same size check the Hessian runs
/// (`ks_fixed_grid::check_response_table`). A test-only cap on this thread
/// makes H2 / STO-3G "too large": 2 basis functions x 16500 points x 1 plane
/// (LDA) x 8 bytes = 264000 bytes. Below that cap the explicit route is
/// refused BEFORE its SCF (the SCF configured here cannot converge in one
/// iteration, so a refusal after it would be `ScfConvergence`), and `Auto`
/// runs finite differences, bit for bit; just above it `Auto` is the
/// moving-grid Hessian (the pass condition is reachable).
#[test]
fn auto_sends_a_system_above_the_response_table_cap_to_fd_before_the_scf() {
    if !has_deriv2() {
        return;
    }
    use ferric_scf::ks_fixed_grid::ResponseTableCapOverride;
    let h2 = Molecule::parse_xyz("2\nh2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
    let cfg = ks("LDA");
    let rhf = FrequencyReference::Rhf;
    {
        let _cap = ResponseTableCapOverride::new(263_999);
        let one_iteration = RhfConfig {
            max_iter: 1,
            energy_conv: 1e-14,
            density_conv: 1e-14,
            ..ks("LDA")
        };
        let msg = run(&h2, "sto-3g", one_iteration, rhf);
        assert!(
            msg.contains("resident AO table") && msg.contains("2 basis functions x 16500"),
            "{msg}"
        );
        let auto = freq(&h2, &cfg, rhf, HessianMethod::Auto);
        let fd = freq(&h2, &cfg, rhf, HessianMethod::FiniteDifference);
        assert_eq!(auto.hessian_source, HessianSource::FiniteDifference);
        assert_same(&auto, &fd, "above the cap");
    }
    let _cap = ResponseTableCapOverride::new(264_000);
    let auto = freq(&h2, &cfg, rhf, HessianMethod::Auto);
    assert_eq!(auto.hessian_source, HessianSource::AnalyticMovingGrid);
}

/// An unset `df_j_aux` / `df_k_aux` is the SCF's default fitting basis
/// (`def2-universal-jkfit` for J, and for K of a hybrid): the predicate probes
/// it and `Auto` runs the moving-grid Hessian on that route.
#[test]
fn auto_takes_the_default_fitting_route() {
    if !has_deriv2() {
        return;
    }
    let water = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    for xc in ["PBE", "B3LYP"] {
        let cfg = ks(xc);
        assert!(cfg.df_j_aux.is_none() && cfg.df_k_aux.is_none());
        let auto = freq(&water, &cfg, FrequencyReference::Rhf, HessianMethod::Auto);
        assert_eq!(
            auto.hessian_source,
            HessianSource::AnalyticMovingGrid,
            "{xc}"
        );
        eprintln!("{xc} (default RI): {:?} cm^-1", auto.frequencies);
    }
}

/// As for Hartree-Fock, an error after the decision is returned: a
/// non-converged SCF under `Auto` is `ScfConvergence`, not a silent
/// finite-difference run on the same SCF settings.
#[test]
fn auto_returns_a_non_converged_scf() {
    if !has_deriv2() {
        return;
    }
    let water = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let cfg = RhfConfig {
        max_iter: 1,
        energy_conv: 1e-14,
        density_conv: 1e-14,
        ..ks("LDA")
    };
    let err = harmonic_frequencies(
        &ParallelContext::default(),
        &water,
        "sto-3g",
        Operator::coulomb(),
        &cfg,
        &FrequencyConfig::default(),
    )
    .expect_err("one SCF iteration cannot converge");
    assert!(
        matches!(err, ferric_core::FerricError::ScfConvergence { .. }),
        "{err}"
    );
}
