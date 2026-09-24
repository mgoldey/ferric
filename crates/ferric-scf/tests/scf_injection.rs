//! Stage-1 PBC commit 1: `solve_rhf_injected` / `PeriodicInjection` /
//! `validate_injected` (reference/pbc/stage1-design.md §1).
//!
//! # Anchors
//!
//! * **Bitwise**: injecting the molecular `(S, h, V_nn)` plus a `DirectJ` +
//!   `LinkK` pair reproduces `solve_rhf` with `k_builder = "link"` bit for
//!   bit. That molecular path runs the SAME builder sequence (DirectJ.build,
//!   LinkK.update_density, LinkK.build) with the same bounds, so any
//!   difference is a defect in the injection plumbing, not arithmetic noise.
//! * **Structural**: injecting `DirectJ` + `DirectK` against the DEFAULT
//!   molecular path (combined `DirectJK` with the incremental ΔD Fock build)
//!   agrees to 1e-10 Ha, not bitwise: the default path reduces J and K in one
//!   combined quartet sweep and accumulates ΔJ/ΔK between full rebuilds, so
//!   its rounding differs from two separate full builds per iteration.
//!
//! Artifact hypothesis: if the injected (S, h, V_nn) were ignored (molecular
//! rebuild still used) the anchors would ALSO pass — so a negative control
//! injects a perturbed V_nn and requires the energy to move by exactly that
//! shift (and the density not at all).
//!
//! Rejections: every refused `RhfConfig` field is asserted BY NAME, both on
//! the typed `InjectedConfigError::field` and in the `FerricError` message
//! returned by `solve_rhf_injected`.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::direct_j::DirectJ;
use ferric_scf::direct_k::DirectK;
use ferric_scf::link_k::LinkK;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{
    resolve_three_index_budget, solve_rhf, solve_rhf_injected, validate_injected,
    PeriodicInjection, RhfConfig,
};
use ferric_scf::screening::{LinkBound, SchwarzBounds, ScreeningKind};
use ndarray::Array2;

const WATER: &str = "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
";

struct Setup {
    ctx: ParallelContext,
    mol: Molecule,
    prep: PreparedBasis,
    op: Operator,
    bounds: SchwarzBounds,
}

fn setup(basis_name: &str) -> Setup {
    let mol = Molecule::parse_xyz(WATER, 0, 1).expect("water");
    let bs = basis::bundled(basis_name).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    Setup {
        ctx: ParallelContext::default(),
        mol,
        prep,
        op,
        bounds,
    }
}

/// The molecular (S, h, V_nn) exactly as `driver::prepare` builds them with
/// no external potential.
fn molecular_one_electron(su: &Setup) -> (Array2<f64>, Array2<f64>, f64) {
    let s = oneelectron::overlap(&su.prep);
    let h = oneelectron::hcore_ecp_with_external(&su.prep, &su.mol, su.prep.basis_set(), None)
        .expect("hcore");
    // `driver::prepare` adds `map_or(0.0, ..)` = +0.0, which is bitwise a no-op.
    let vnn = su.mol.nuclear_repulsion();
    (s, h, vnn)
}

/// A config the injected path accepts: everything default except the MINAO
/// guess (projects onto molecular atoms, refused by `validate_injected`).
fn injectable_config() -> RhfConfig {
    RhfConfig {
        use_sad_guess: false,
        ..Default::default()
    }
}

fn assert_bitwise(label: &str, a: &ScfResult, b: &ScfResult) {
    assert!(a.converged && b.converged, "{label}: both must converge");
    assert_eq!(
        a.energy.to_bits(),
        b.energy.to_bits(),
        "{label}: energy {} vs {}",
        a.energy,
        b.energy
    );
    assert_eq!(a.iterations, b.iterations, "{label}: iteration count");
    assert_eq!(
        a.density_total, b.density_total,
        "{label}: density not bitwise"
    );
    assert_eq!(
        a.eps_alpha, b.eps_alpha,
        "{label}: orbital energies not bitwise"
    );
    assert_eq!(a.fock_alpha, b.fock_alpha, "{label}: Fock not bitwise");
}

#[test]
fn injected_molecular_link_pair_reproduces_solve_rhf_bitwise() {
    let su = setup("cc-pvdz");
    let cfg = injectable_config();
    let budget = resolve_three_index_budget(cfg.three_index_budget_bytes);

    // Reference: molecular solve_rhf through the pluggable-K arm (DirectJ + LinkK).
    let ref_cfg = RhfConfig {
        k_builder: Some("link".into()),
        ..injectable_config()
    };
    let reference =
        solve_rhf(&su.ctx, &su.mol, &su.prep, su.op, &su.bounds, &ref_cfg).expect("reference");

    // Injected: the same builders, constructed exactly as solve_rhf does for
    // k_builder = "link" with the default ScreeningKind.
    let link_schwarz =
        SchwarzBounds::compute_for_screening(su.op, &su.prep, ScreeningKind::default())
            .expect("link schwarz");
    let link_bound = LinkBound::SchwarzRef(&link_schwarz);
    let (s, h, vnn) = molecular_one_electron(&su);
    let inj = PeriodicInjection {
        s,
        h,
        vnn,
        j: Box::new(DirectJ::new(
            &su.ctx,
            &su.prep,
            &su.bounds,
            cfg.integral_thresh,
            budget,
        )),
        k: Box::new(LinkK::new(
            &su.ctx,
            &su.prep,
            &link_bound,
            su.op,
            cfg.integral_thresh,
            budget,
        )),
    };
    let injected = solve_rhf_injected(&su.ctx, &su.mol, &su.prep, su.op, &su.bounds, &cfg, inj)
        .expect("injected");
    assert_bitwise("water/cc-pVDZ link", &reference, &injected);
}

#[test]
fn injected_direct_pair_matches_default_direct_jk_path() {
    let su = setup("cc-pvdz");
    let cfg = RhfConfig {
        density_conv: 1e-9,
        ..injectable_config()
    };
    let budget = resolve_three_index_budget(cfg.three_index_budget_bytes);
    let reference =
        solve_rhf(&su.ctx, &su.mol, &su.prep, su.op, &su.bounds, &cfg).expect("reference");

    let (s, h, vnn) = molecular_one_electron(&su);
    let inj = PeriodicInjection {
        s,
        h,
        vnn,
        j: Box::new(DirectJ::new(
            &su.ctx,
            &su.prep,
            &su.bounds,
            cfg.integral_thresh,
            budget,
        )),
        k: Box::new(DirectK::new(
            &su.ctx,
            &su.prep,
            &su.bounds,
            cfg.integral_thresh,
            budget,
        )),
    };
    let injected = solve_rhf_injected(&su.ctx, &su.mol, &su.prep, su.op, &su.bounds, &cfg, inj)
        .expect("injected");
    assert!(reference.converged && injected.converged);
    let de = (reference.energy - injected.energy).abs();
    let dd = (&reference.density_total - &injected.density_total)
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()));
    println!("default DirectJK vs injected DirectJ+DirectK: |dE| = {de:.3e}, max|dD| = {dd:.3e}");
    assert!(de < 1e-10, "|dE| = {de:.3e}");
    assert!(dd < 1e-7, "max|dD| = {dd:.3e}");
}

/// Negative control: the injected V_nn and h must actually be the ones used.
/// Shifting V_nn by c moves E by exactly c (density untouched); shifting h by
/// c·S (a constant potential) moves every orbital energy by c and E by c·N.
#[test]
fn injected_matrices_are_the_ones_used() {
    let su = setup("sto-3g");
    let cfg = injectable_config();
    let budget = resolve_three_index_budget(cfg.three_index_budget_bytes);
    let run = |s: Array2<f64>, h: Array2<f64>, vnn: f64| {
        let inj = PeriodicInjection {
            s,
            h,
            vnn,
            j: Box::new(DirectJ::new(
                &su.ctx,
                &su.prep,
                &su.bounds,
                cfg.integral_thresh,
                budget,
            )),
            k: Box::new(DirectK::new(
                &su.ctx,
                &su.prep,
                &su.bounds,
                cfg.integral_thresh,
                budget,
            )),
        };
        solve_rhf_injected(&su.ctx, &su.mol, &su.prep, su.op, &su.bounds, &cfg, inj)
            .expect("injected")
    };
    let (s, h, vnn) = molecular_one_electron(&su);
    let base = run(s.clone(), h.clone(), vnn);

    let shifted_vnn = run(s.clone(), h.clone(), vnn + 0.25);
    assert!(
        (shifted_vnn.energy - base.energy - 0.25).abs() < 1e-12,
        "V_nn shift not honoured: dE = {}",
        shifted_vnn.energy - base.energy
    );
    assert_eq!(shifted_vnn.density_total, base.density_total);

    let c = 0.1;
    let h_shift = &h + &(c * &s);
    let shifted_h = run(s, h_shift, vnn);
    let nelec = su.mol.nelec() as f64;
    let de = shifted_h.energy - base.energy;
    assert!(
        (de - c * nelec).abs() < 1e-8,
        "h + c·S not honoured: dE = {de}, expected {}",
        c * nelec
    );
    let de0 = shifted_h.eps_alpha[0] - base.eps_alpha[0];
    assert!((de0 - c).abs() < 1e-8, "eps_0 shift {de0}, expected {c}");
}

/// One case per refused field: (field name as reported, config mutation).
fn rejected_cases() -> Vec<(&'static str, RhfConfig)> {
    let base = injectable_config;
    vec![
        (
            "k_builder",
            RhfConfig {
                k_builder: Some("direct".into()),
                ..base()
            },
        ),
        (
            "df_j_aux",
            RhfConfig {
                df_j_aux: Some("def2-universal-jkfit".into()),
                ..base()
            },
        ),
        (
            "df_k_aux",
            RhfConfig {
                df_k_aux: Some("def2-universal-jkfit".into()),
                ..base()
            },
        ),
        (
            "xc",
            RhfConfig {
                xc: Some("PBE".into()),
                ..base()
            },
        ),
        (
            "newton_trigger",
            RhfConfig {
                newton_trigger: 1e-3,
                ..base()
            },
        ),
        (
            "aurora.enabled",
            RhfConfig {
                aurora: ferric_scf::aurora::AuroraConfig {
                    enabled: true,
                    ..Default::default()
                },
                ..base()
            },
        ),
        (
            "trah_trigger",
            RhfConfig {
                trah_trigger: Some(1e-2),
                ..base()
            },
        ),
        (
            "use_sad_guess",
            RhfConfig {
                use_sad_guess: true,
                ..base()
            },
        ),
        (
            "external_potential",
            RhfConfig {
                external_potential: Some(Default::default()),
                ..base()
            },
        ),
        (
            "cosmo",
            RhfConfig {
                cosmo: Some(Default::default()),
                ..base()
            },
        ),
        (
            "pcm",
            RhfConfig {
                pcm: Some(Default::default()),
                ..base()
            },
        ),
        (
            "polarizable",
            RhfConfig {
                polarizable: Some(Default::default()),
                ..base()
            },
        ),
        (
            "check_stability",
            RhfConfig {
                check_stability: true,
                ..base()
            },
        ),
    ]
}

#[test]
fn baseline_injectable_config_is_accepted() {
    // Reachability: the rejection tests below each differ from this config in
    // ONE field, so this must pass or they prove nothing.
    validate_injected(&injectable_config()).expect("baseline must validate");
    // use_sad_guess is irrelevant once an explicit density is supplied (same
    // precedence as the molecular path), so it is not refused then.
    let with_density = RhfConfig {
        use_sad_guess: true,
        init_guess_density: Some(Array2::zeros((2, 2))),
        ..Default::default()
    };
    validate_injected(&with_density).expect("explicit density overrides the guess choice");
}

#[test]
fn each_rejected_field_is_named_in_the_typed_error() {
    for (field, cfg) in rejected_cases() {
        let err = validate_injected(&cfg).expect_err(field);
        assert_eq!(err.field, field, "wrong field reported");
    }
}

#[test]
fn each_rejected_field_is_named_by_solve_rhf_injected() {
    let su = setup("sto-3g");
    for (field, cfg) in rejected_cases() {
        let (s, h, vnn) = molecular_one_electron(&su);
        let inj = PeriodicInjection {
            s,
            h,
            vnn,
            j: Box::new(DirectJ::new(&su.ctx, &su.prep, &su.bounds, 1e-12, 1 << 30)),
            k: Box::new(DirectK::new(&su.ctx, &su.prep, &su.bounds, 1e-12, 1 << 30)),
        };
        let err = solve_rhf_injected(&su.ctx, &su.mol, &su.prep, su.op, &su.bounds, &cfg, inj)
            .expect_err(field)
            .to_string();
        let needle = format!("RhfConfig.{field} is not supported");
        assert!(err.contains(&needle), "expected {needle:?} in {err:?}");
    }
}

#[test]
fn wrong_shape_and_nonfinite_vnn_are_rejected() {
    let su = setup("sto-3g");
    let cfg = injectable_config();
    let n = su.prep.nbasis();
    let mk = |s: Array2<f64>, h: Array2<f64>, vnn: f64| PeriodicInjection {
        s,
        h,
        vnn,
        j: Box::new(DirectJ::new(&su.ctx, &su.prep, &su.bounds, 1e-12, 1 << 30)),
        k: Box::new(DirectK::new(&su.ctx, &su.prep, &su.bounds, 1e-12, 1 << 30)),
    };
    let (s, h, vnn) = molecular_one_electron(&su);

    let bad_s = Array2::<f64>::zeros((n + 1, n + 1));
    let e = solve_rhf_injected(
        &su.ctx,
        &su.mol,
        &su.prep,
        su.op,
        &su.bounds,
        &cfg,
        mk(bad_s, h.clone(), vnn),
    )
    .expect_err("bad S shape")
    .to_string();
    assert!(e.contains("PeriodicInjection.s has shape"), "{e}");

    let bad_h = Array2::<f64>::zeros((n, n - 1));
    let e = solve_rhf_injected(
        &su.ctx,
        &su.mol,
        &su.prep,
        su.op,
        &su.bounds,
        &cfg,
        mk(s.clone(), bad_h, vnn),
    )
    .expect_err("bad h shape")
    .to_string();
    assert!(e.contains("PeriodicInjection.h has shape"), "{e}");

    let e = solve_rhf_injected(
        &su.ctx,
        &su.mol,
        &su.prep,
        su.op,
        &su.bounds,
        &cfg,
        mk(s, h, f64::NAN),
    )
    .expect_err("NaN vnn")
    .to_string();
    assert!(e.contains("PeriodicInjection.vnn is not finite"), "{e}");
}
