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
//! Stage 4 (UHF, bottom of this file): the same three anchors for
//! `solve_uhf_injected` on the water cation doublet (bitwise vs `solve_uhf`
//! with `k_builder = "link"`; 1e-10 vs the default combined `DirectJK`;
//! V_nn negative control), plus the explicit-density guess path and every
//! refused field by name, including the UHF-only `scf_stability_descent`.
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
use ferric_scf::uhf::{solve_uhf, solve_uhf_injected, validate_injected_uhf};
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
    setup_spin(basis_name, 0, 1)
}

fn setup_spin(basis_name: &str, charge: i32, mult: usize) -> Setup {
    let mol = Molecule::parse_xyz(WATER, charge, mult).expect("water");
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
        xc: None,
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
        xc: None,
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

/// Stage 2 (periodic KS): an injected [`XcBuilder`] wrapping the MOLECULAR
/// `KsXc` (same grid as `solve_rhf`'s) plus `DirectJ` + `DirectK` reproduces
/// molecular KS with exact J/K to the 1e-10 of the HF structural anchor above
/// — for a pure GGA (injected K never built) and a global hybrid (injected K
/// scaled by the exact-exchange fraction). Negative control: the same
/// builder claiming a fraction of 1 (K unscaled, i.e. the "forgot to scale"
/// defect) moves PBE0 by far more than the tolerance.
#[test]
fn injected_xc_builder_reproduces_molecular_ks() {
    use ferric_dft::grid::AtomicGridConfig;
    use ferric_dft::ks::KsXc;
    use ferric_dft::xc_trait::XcContribution;
    use ferric_scf::rhf::XcBuilder;

    struct MolXc {
        ks: KsXc,
        frac: Option<f64>,
    }
    impl XcBuilder for MolXc {
        fn build(
            &mut self,
            d: &Array2<f64>,
        ) -> Result<(f64, Array2<f64>), ferric_core::FerricError> {
            let mut v = Array2::<f64>::zeros(d.dim());
            let e = self.ks.add_xc(d, &mut v);
            Ok((e, v))
        }
        fn exact_exchange_fraction(&self) -> f64 {
            self.frac.unwrap_or_else(|| self.ks.k_mix().sr)
        }
    }

    let su = setup("sto-3g");
    let bs = basis::bundled("sto-3g").expect("basis");
    let main = AtomicGridConfig::default();
    let nlc = AtomicGridConfig {
        n_radial: 50,
        n_angular: 50,
        ..Default::default()
    };
    let cfg = RhfConfig {
        density_conv: 1e-9,
        ..injectable_config()
    };
    let budget = resolve_three_index_budget(cfg.three_index_budget_bytes);
    let injected = |xc: &str, frac: Option<f64>| {
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
            xc: Some(Box::new(MolXc {
                ks: KsXc::new(&su.mol, &bs, xc, &main, &nlc).expect("KsXc"),
                frac,
            })),
        };
        solve_rhf_injected(&su.ctx, &su.mol, &su.prep, su.op, &su.bounds, &cfg, inj)
            .expect("injected KS")
    };
    for xc in ["PBE", "PBE0"] {
        let ref_cfg = RhfConfig {
            xc: Some(xc.into()),
            // Exact four-centre J/K, as the injected DirectJ/DirectK.
            df_j_aux: Some(String::new()),
            df_k_aux: Some(String::new()),
            ..cfg.clone()
        };
        let reference = solve_rhf(&su.ctx, &su.mol, &su.prep, su.op, &su.bounds, &ref_cfg)
            .expect("molecular KS");
        let inj = injected(xc, None);
        assert!(reference.converged && inj.converged, "{xc}");
        let de = (reference.energy - inj.energy).abs();
        println!(
            "{xc}: molecular {:.12} injected {:.12} |dE| {de:.3e}",
            reference.energy, inj.energy
        );
        assert!(de < 1e-10, "{xc}: |dE| = {de:.3e}");
        if xc == "PBE0" {
            let wrong = injected(xc, Some(1.0));
            let dw = (wrong.energy - reference.energy).abs();
            assert!(dw > 1e-3, "unscaled K moved PBE0 by only {dw:.3e}");
        }
    }
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
            xc: None,
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
            xc: None,
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
        xc: None,
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

// ===========================================================================
// Stage 4: solve_uhf_injected
// ===========================================================================

/// Water cation doublet (N_α = 5, N_β = 4): a genuine open shell, so K_α ≠
/// K_β and a K built from the TOTAL density (the prototype's main mutant)
/// changes the answer.
fn uhf_setup() -> Setup {
    setup_spin("6-31g", 1, 2)
}

/// Injected UHF with the given builders and the molecular (S, h, V_nn).
fn run_uhf_injected<'a>(
    su: &'a Setup,
    cfg: &RhfConfig,
    vnn_shift: f64,
    j: Box<dyn ferric_scf::fock::JBuilder + 'a>,
    k: Box<dyn ferric_scf::fock::KBuilder + 'a>,
) -> ScfResult {
    let (s, h, vnn) = molecular_one_electron(su);
    let inj = PeriodicInjection {
        s,
        h,
        vnn: vnn + vnn_shift,
        j,
        k,
        xc: None,
    };
    solve_uhf_injected(&su.ctx, &su.mol, &su.prep, &su.bounds, cfg, inj, None)
        .expect("injected UHF")
}

fn assert_bitwise_uhf(label: &str, a: &ScfResult, b: &ScfResult) {
    assert_bitwise(label, a, b);
    assert_eq!(a.density_alpha, b.density_alpha, "{label}: D_alpha");
    assert_eq!(a.density_beta, b.density_beta, "{label}: D_beta");
    assert_eq!(a.eps_beta, b.eps_beta, "{label}: eps_beta");
    assert_eq!(a.fock_beta, b.fock_beta, "{label}: F_beta");
}

/// Bitwise: `solve_uhf` with `k_builder = "link"` runs DirectJ.build(D_α+D_β)
/// then, per spin, LinkK.update_density(D_σ) + LinkK.build(D_σ) from the
/// core guess (`use_sad_guess = false`). Injecting the same DirectJ + LinkK
/// (same bounds, thresh, budget) must reproduce it bit for bit.
#[test]
fn injected_molecular_link_pair_reproduces_solve_uhf_bitwise() {
    let su = uhf_setup();
    let cfg = injectable_config();
    let budget = resolve_three_index_budget(cfg.three_index_budget_bytes);
    let ref_cfg = RhfConfig {
        k_builder: Some("link".into()),
        ..injectable_config()
    };
    let reference =
        solve_uhf(&su.ctx, &su.mol, &su.prep, &su.bounds, &ref_cfg).expect("reference UHF");
    // solve_uhf builds LinkK on LinkBound::SchwarzRef(bounds) with bounds.op.
    let link_bound = LinkBound::SchwarzRef(&su.bounds);
    let injected = run_uhf_injected(
        &su,
        &cfg,
        0.0,
        Box::new(DirectJ::new(
            &su.ctx,
            &su.prep,
            &su.bounds,
            cfg.integral_thresh,
            budget,
        )),
        Box::new(LinkK::new(
            &su.ctx,
            &su.prep,
            &link_bound,
            su.op,
            cfg.integral_thresh,
            budget,
        )),
    );
    assert_bitwise_uhf("H2O+/6-31G link UHF", &reference, &injected);
    assert!(
        reference.density_alpha != *reference.density_beta.as_ref().unwrap(),
        "vacuous: the cation must be spin-polarised"
    );
}

/// Structural: the default molecular UHF runs the combined single-pass
/// `DirectJK::build_uhf` (one quartet sweep for J, K_α, K_β), whose reduction
/// order differs from separate DirectJ + DirectK builds, so agreement is
/// 1e-10 Ha, not bitwise.
#[test]
fn injected_direct_pair_matches_default_uhf_path() {
    let su = uhf_setup();
    let cfg = RhfConfig {
        density_conv: 1e-9,
        ..injectable_config()
    };
    let budget = resolve_three_index_budget(cfg.three_index_budget_bytes);
    let reference = solve_uhf(&su.ctx, &su.mol, &su.prep, &su.bounds, &cfg).expect("reference");
    let injected = run_uhf_injected(
        &su,
        &cfg,
        0.0,
        Box::new(DirectJ::new(
            &su.ctx,
            &su.prep,
            &su.bounds,
            cfg.integral_thresh,
            budget,
        )),
        Box::new(DirectK::new(
            &su.ctx,
            &su.prep,
            &su.bounds,
            cfg.integral_thresh,
            budget,
        )),
    );
    let de = (reference.energy - injected.energy).abs();
    let dd = (&reference.density_total - &injected.density_total)
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()));
    println!(
        "UHF default DirectJK vs injected DirectJ+DirectK: |dE| = {de:.3e}, max|dD| = {dd:.3e}"
    );
    assert!(de < 1e-10, "|dE| = {de:.3e}");
    assert!(dd < 1e-7, "max|dD| = {dd:.3e}");
}

/// Negative control: the injected V_nn is the one used (E moves by exactly
/// the shift, densities bitwise unchanged).
#[test]
fn injected_uhf_vnn_is_the_one_used() {
    let su = setup_spin("sto-3g", 1, 2);
    let cfg = injectable_config();
    let mk = || {
        (
            Box::new(DirectJ::new(&su.ctx, &su.prep, &su.bounds, 1e-12, 1 << 30)),
            Box::new(DirectK::new(&su.ctx, &su.prep, &su.bounds, 1e-12, 1 << 30)),
        )
    };
    let (j0, k0) = mk();
    let base = run_uhf_injected(&su, &cfg, 0.0, j0, k0);
    let (j1, k1) = mk();
    let shifted = run_uhf_injected(&su, &cfg, 0.25, j1, k1);
    assert!(
        (shifted.energy - base.energy - 0.25).abs() < 1e-12,
        "V_nn shift not honoured: dE = {}",
        shifted.energy - base.energy
    );
    assert_eq!(shifted.density_alpha, base.density_alpha);
    assert_eq!(shifted.density_beta, base.density_beta);
}

/// The explicit-density guess builds its Fock from the INJECTED builders
/// (never MINAO / molecular build_jk). With the converged total density as
/// the guess and `use_sad_guess = true` (legal once a density is given), the
/// SCF lands on the same state.
#[test]
fn injected_uhf_explicit_density_guess_reaches_the_same_state() {
    let su = setup_spin("sto-3g", 1, 2);
    let cfg = injectable_config();
    let mk = || {
        (
            Box::new(DirectJ::new(&su.ctx, &su.prep, &su.bounds, 1e-12, 1 << 30)),
            Box::new(DirectK::new(&su.ctx, &su.prep, &su.bounds, 1e-12, 1 << 30)),
        )
    };
    let (j0, k0) = mk();
    let base = run_uhf_injected(&su, &cfg, 0.0, j0, k0);
    let guess_cfg = RhfConfig {
        use_sad_guess: true,
        init_guess_density: Some(base.density_total.clone()),
        ..injectable_config()
    };
    let (j1, k1) = mk();
    let from_d = run_uhf_injected(&su, &guess_cfg, 0.0, j1, k1);
    let de = (from_d.energy - base.energy).abs();
    println!(
        "explicit-density guess: |dE| = {de:.3e}, iterations {} (core guess {})",
        from_d.iterations, base.iterations
    );
    assert!(de < 1e-8, "|dE| = {de:.3e}");

    // Wrong-shape density is a named error, not a silent core-guess fallback.
    let bad_cfg = RhfConfig {
        init_guess_density: Some(Array2::zeros((2, 2))),
        ..injectable_config()
    };
    let (j2, k2) = mk();
    let (s, h, vnn) = molecular_one_electron(&su);
    let inj = PeriodicInjection {
        s,
        h,
        vnn,
        j: j2,
        k: k2,
        xc: None,
    };
    let e = solve_uhf_injected(&su.ctx, &su.mol, &su.prep, &su.bounds, &bad_cfg, inj, None)
        .expect_err("bad guess density shape")
        .to_string();
    assert!(e.contains("init_guess_density shape"), "{e}");
}

fn rejected_cases_uhf() -> Vec<(&'static str, RhfConfig)> {
    let mut v = rejected_cases();
    v.push((
        "scf_stability_descent",
        RhfConfig {
            scf_stability_descent: true,
            ..injectable_config()
        },
    ));
    v
}

#[test]
fn uhf_baseline_injectable_config_is_accepted() {
    validate_injected_uhf(&injectable_config()).expect("baseline must validate");
}

#[test]
fn each_rejected_field_is_named_by_validate_injected_uhf() {
    for (field, cfg) in rejected_cases_uhf() {
        let err = validate_injected_uhf(&cfg).expect_err(field);
        assert_eq!(err.field, field, "wrong field reported");
    }
}

#[test]
fn each_rejected_field_is_named_by_solve_uhf_injected() {
    let su = setup_spin("sto-3g", 1, 2);
    for (field, cfg) in rejected_cases_uhf() {
        let (s, h, vnn) = molecular_one_electron(&su);
        let inj = PeriodicInjection {
            s,
            h,
            vnn,
            j: Box::new(DirectJ::new(&su.ctx, &su.prep, &su.bounds, 1e-12, 1 << 30)),
            k: Box::new(DirectK::new(&su.ctx, &su.prep, &su.bounds, 1e-12, 1 << 30)),
            xc: None,
        };
        let err = solve_uhf_injected(&su.ctx, &su.mol, &su.prep, &su.bounds, &cfg, inj, None)
            .expect_err(field)
            .to_string();
        let needle = format!("solve_uhf_injected: RhfConfig.{field} is not supported");
        assert!(err.contains(&needle), "expected {needle:?} in {err:?}");
    }
}

#[test]
fn uhf_wrong_shape_and_nonfinite_vnn_are_rejected() {
    let su = setup_spin("sto-3g", 1, 2);
    let cfg = injectable_config();
    let n = su.prep.nbasis();
    let (s, h, vnn) = molecular_one_electron(&su);
    for (label, s_, h_, v_, needle) in [
        (
            "bad S",
            Array2::<f64>::zeros((n + 1, n + 1)),
            h.clone(),
            vnn,
            "solve_uhf_injected: PeriodicInjection.s has shape",
        ),
        (
            "bad h",
            s.clone(),
            Array2::<f64>::zeros((n, n - 1)),
            vnn,
            "solve_uhf_injected: PeriodicInjection.h has shape",
        ),
        (
            "NaN vnn",
            s.clone(),
            h.clone(),
            f64::NAN,
            "solve_uhf_injected: PeriodicInjection.vnn is not finite",
        ),
    ] {
        let inj = PeriodicInjection {
            s: s_,
            h: h_,
            vnn: v_,
            j: Box::new(DirectJ::new(&su.ctx, &su.prep, &su.bounds, 1e-12, 1 << 30)),
            k: Box::new(DirectK::new(&su.ctx, &su.prep, &su.bounds, 1e-12, 1 << 30)),
            xc: None,
        };
        let e = solve_uhf_injected(&su.ctx, &su.mol, &su.prep, &su.bounds, &cfg, inj, None)
            .expect_err(label)
            .to_string();
        assert!(e.contains(needle), "{label}: {e}");
    }
}
