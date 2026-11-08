//! The analytic UKS Hessian on the production (moving) grid,
//! `ferric_scf::hessian::uks_moving_grid_hessian`.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test uks_moving_grid_hessian -- --include-ignored
//! ```
//!
//! # What is compared to what
//!
//! * Exactness anchor: a closed-shell singlet run through UKS (`D^α = D^β`)
//!   against the closed-shell `rks_moving_grid_hessian` (and fixed ≡ fixed).
//!   ARTIFACT HYPOTHESIS: every singlet perturbation is spin-symmetric, so a
//!   construction right only for spin-symmetric perturbations passes it
//!   (the total-density-kernel mutant is shown to); it never stands alone.
//! * The full Hessian vs a central difference of the PRODUCTION UKS gradient
//!   (`ks_gradient_uks_with_exchange_on_grids`, a UKS SCF at every displaced
//!   geometry converged past the iteration-count flip) on doublet and triplet
//!   radicals (unequal α/β densities and responses).
//! * Frequencies of the analytic route vs the finite-difference route of
//!   `harmonic_frequencies`.
//! * Mutants (αβ kernel block dropped, the total-density kernel, the weight
//!   terms dropped, the exchange fraction forgotten, the kernel dropped from
//!   the response) each miss the gradient difference.

mod uks_hessian_common;

use ferric_core::parallel::ParallelContext;
use ferric_integrals::operator::Operator;
use ferric_scf::frequencies::{
    harmonic_frequencies, FrequencyConfig, FrequencyReference, HessianMethod, HessianSource,
};
use ferric_scf::hessian::{
    rks_fixed_grid_hessian, rks_moving_grid_hessian, uks_hessian_parts_with,
    uks_moving_grid_hessian, KsFixedGridMutants,
};
use ferric_scf::ks_fixed_grid::{FixedGridKs, GridMode};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ndarray::Array2;
use uks_hessian_common::*;

/// Singlet UKS vs RKS Hessian, Ha/Bohr².
const TOL_SINGLET: f64 = 1e-9;
/// Full UKS Hessian vs FD of the production UKS gradient, Ha/Bohr² (the RKS
/// moving-grid bar, `ks_moving_grid_common::TOL_HESS_FD`).
const TOL_HESS_FD: f64 = 2e-9;
const MUTANT_FACTOR: f64 = 10.0;

fn analytic(p: &Pt, cfg: &RhfConfig) -> Array2<f64> {
    uks_moving_grid_hessian(
        &ParallelContext::default(),
        &p.mol,
        &p.prep,
        Operator::coulomb(),
        &p.bounds,
        &p.res,
        cfg,
    )
    .unwrap()
}

fn mutated(p: &Pt, cfg: &RhfConfig, xc: &str, m: &KsFixedGridMutants) -> Array2<f64> {
    let grid = FixedGridKs::new(&p.mol, xc, &p.grid).unwrap();
    uks_hessian_parts_with(
        &ParallelContext::default(),
        &p.mol,
        &p.prep,
        Operator::coulomb(),
        &p.bounds,
        &p.res,
        cfg,
        &grid,
        m,
        GridMode::Moving,
    )
    .unwrap()
    .total()
}

// ---------------------------------------------------------------------------
// Singlet anchor
// ---------------------------------------------------------------------------

/// A closed-shell water singlet through UKS equals the RKS analytic Hessian,
/// moving and fixed grid, LDA / PBE / B3LYP, exact J/K and RI-J/RI-K.
#[ignore = "tier medium (10-60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
#[test]
fn singlet_uks_hessian_equals_the_rks_hessian() {
    if !has_deriv2() {
        return;
    }
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    // UKS fits only a named auxiliary basis (unset = exact), so name it.
    let ri = |xc: &str| RhfConfig {
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        ..config(xc)
    };
    let cases: Vec<(&str, RhfConfig)> = vec![
        ("LDA", config("LDA")),
        ("PBE", config("PBE")),
        ("B3LYP", config("B3LYP")),
        ("PBE", ri("PBE")),
        ("B3LYP", ri("B3LYP")),
    ];
    for (xc, cfg) in cases {
        let cfg = fd_reference_config(&cfg);
        let u = scf(&water(), "6-31g", &cfg);
        let r = solve_rhf(&ctx, &u.mol, &u.prep, op, &u.bounds, &cfg).unwrap();
        let (da, db) = spin_densities(&u);
        let hu = analytic(&u, &cfg);
        let hr = rks_moving_grid_hessian(&ctx, &u.mol, &u.prep, op, &u.bounds, &r, &cfg).unwrap();
        let grid = FixedGridKs::new(&u.mol, xc, &u.grid).unwrap();
        let fu = uks_hessian_parts_with(
            &ctx,
            &u.mol,
            &u.prep,
            op,
            &u.bounds,
            &u.res,
            &cfg,
            &grid,
            &KsFixedGridMutants::default(),
            GridMode::Fixed,
        )
        .unwrap()
        .total();
        let fr = rks_fixed_grid_hessian(&ctx, &u.mol, &u.prep, op, &u.bounds, &r, &cfg).unwrap();
        // The total-density-kernel mutant passes the singlet anchor.
        let mut m = KsFixedGridMutants::default();
        m.xc.spin.total_kernel = true;
        let ht = mutated(&u, &cfg, xc, &m);
        let ri_label = if cfg.df_j_aux.as_deref().is_some_and(|n| !n.is_empty()) {
            "RI-J/K"
        } else {
            "exact"
        };
        let (e_m, e_f, e_t) = (
            max_abs_diff(&hu, &hr),
            max_abs_diff(&fu, &fr),
            max_abs_diff(&ht, &hr),
        );
        eprintln!(
            "{xc} {ri_label}: |Da-Db| {:.1e}; singlet UKS vs RKS moving {e_m:.2e}, fixed {e_f:.2e} \
             (max |H| {:.2e}); total-density-kernel mutant vs RKS {e_t:.2e}",
            max_abs_diff(&da, &db),
            max_abs(&hr)
        );
        assert!(e_m < TOL_SINGLET, "{xc} {ri_label} moving: {e_m:.3e}");
        assert!(e_f < TOL_SINGLET, "{xc} {ri_label} fixed: {e_f:.3e}");
        assert!(
            e_t < TOL_SINGLET,
            "the singlet anchor is blind to it: {e_t:.3e}"
        );
    }
}

// ---------------------------------------------------------------------------
// Full Hessian vs FD of the production UKS gradient
// ---------------------------------------------------------------------------

/// One radical case; returns the miss.
fn fd_case(name: &str, mol: &ferric_core::mol::Molecule, basis_name: &str, cfg: &RhfConfig) -> f64 {
    let xc = cfg.xc.clone().unwrap();
    let cfg = &fd_reference_config(cfg);
    let p = scf(mol, basis_name, cfg);
    let (da, db) = spin_densities(&p);
    let h = analytic(&p, cfg);
    let (fd, err) = fd_of_production_gradient(&p, basis_name, cfg, &xc);
    let miss = max_abs_diff(&h, &fd);
    eprintln!(
        "{name} {basis_name} {xc}: |Da-Db| {:.2e}; analytic vs FD of the production UKS gradient \
         {miss:.3e} (max |H| {:.2e}, sum rule {:.1e}, FD asymmetry {:.1e}, worst SCF err_max \
         {err:.1e})",
        max_abs_diff(&da, &db),
        max_abs(&h),
        sum_rule(&h),
        max_abs_diff(&fd, &fd.t().to_owned())
    );
    miss
}

/// NH2 and CH3 doublets and the O2 triplet, STO-3G, LDA / PBE / B3LYP, exact
/// J/K, the 50x110 test grid (the per-commit subset; the 6-31G matrix is in
/// `uks_hessian_matches_fd_at_6_31g`).
#[ignore = "tier medium (10-60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
#[test]
fn uks_hessian_matches_fd_of_the_production_gradient() {
    if !has_deriv2() {
        return;
    }
    let mut worst = 0.0f64;
    for (name, mol) in [("NH2", nh2()), ("CH3", ch3()), ("O2", o2())] {
        for xc in ["LDA", "PBE", "B3LYP"] {
            let miss = fd_case(name, &mol, "sto-3g", &config(xc));
            worst = worst.max(miss);
            assert!(miss < TOL_HESS_FD, "{name} {xc}: {miss:.3e}");
        }
    }
    // UKS fits only a named auxiliary basis (unset = exact), so name it.
    let ri = |xc: &str| RhfConfig {
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        ..config(xc)
    };
    for (label, cfg) in [("NH2 RI-J", ri("PBE")), ("NH2 RI-J/RI-K", ri("B3LYP"))] {
        let miss = fd_case(label, &nh2(), "sto-3g", &cfg);
        worst = worst.max(miss);
        assert!(miss < TOL_HESS_FD, "{label}: {miss:.3e}");
    }
    eprintln!("worst STO-3G miss {worst:.3e}");
}

/// The same matrix at 6-31G, plus RI-J / RI-K (the production default fits)
/// and the default (pruned) grid on NH2.
#[ignore = "tier slow (>60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
#[test]
fn uks_hessian_matches_fd_at_6_31g() {
    if !has_deriv2() {
        return;
    }
    let mut worst = 0.0f64;
    for (name, mol) in [("NH2", nh2()), ("CH3", ch3()), ("O2", o2())] {
        for xc in ["LDA", "PBE", "B3LYP"] {
            let miss = fd_case(name, &mol, "6-31g", &config(xc));
            worst = worst.max(miss);
            assert!(miss < TOL_HESS_FD, "{name} {xc}: {miss:.3e}");
        }
    }
    // UKS fits only a named auxiliary basis (unset = exact), so name it.
    let ri = |xc: &str| RhfConfig {
        df_j_aux: Some("def2-universal-jkfit".into()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        ..config(xc)
    };
    for (label, cfg) in [
        ("NH2 RI-J", ri("PBE")),
        ("NH2 RI-J/RI-K", ri("B3LYP")),
        (
            "NH2 default grid",
            RhfConfig {
                dft_grid: None,
                ..config("PBE")
            },
        ),
    ] {
        let miss = fd_case(label, &nh2(), "6-31g", &cfg);
        worst = worst.max(miss);
        assert!(miss < TOL_HESS_FD, "{label}: {miss:.3e}");
    }
    eprintln!("worst 6-31G miss {worst:.3e}");
}

// ---------------------------------------------------------------------------
// Frequencies
// ---------------------------------------------------------------------------

/// `harmonic_frequencies` with `AnalyticMovingGrid` and with `Auto` (which
/// must pick it) vs the finite-difference route, NH2 / 6-31G PBE and the O2
/// triplet B3LYP.
#[ignore = "tier medium (10-60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
#[test]
fn uks_frequencies_match_the_finite_difference_route() {
    if !has_deriv2() {
        return;
    }
    for (name, mol, xc) in [("NH2", nh2(), "PBE"), ("O2", o2(), "B3LYP")] {
        let cfg = fd_reference_config(&config(xc));
        let run = |hessian| {
            harmonic_frequencies(
                &ParallelContext::default(),
                &mol,
                "6-31g",
                Operator::coulomb(),
                &cfg,
                &FrequencyConfig {
                    reference: FrequencyReference::Uhf,
                    hessian,
                    // A 1e-3 Bohr step. MEASURED: the default 5e-3 leaves
                    // 5.9e-2 cm^-1 on NH2 / PBE, 1e-3 leaves 2.4e-3 (O(h^2):
                    // 25x for a 5x step), so the residual is FD truncation.
                    delta: 1e-3,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let a = run(HessianMethod::AnalyticMovingGrid);
        let auto = run(HessianMethod::Auto);
        let fd = run(HessianMethod::FiniteDifference);
        assert_eq!(a.hessian_source, HessianSource::AnalyticMovingGrid);
        assert_eq!(auto.hessian_source, HessianSource::AnalyticMovingGrid);
        assert_eq!(fd.hessian_source, HessianSource::FiniteDifference);
        let dnu = a
            .frequencies
            .iter()
            .zip(&fd.frequencies)
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
        let dauto = a
            .frequencies
            .iter()
            .zip(&auto.frequencies)
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
        eprintln!(
            "{name} {xc}: analytic {:?} cm^-1; max |analytic - FD| {dnu:.2e} cm^-1, |analytic - \
             auto| {dauto:.1e}",
            a.frequencies
        );
        assert!(dauto == 0.0, "Auto must run the same analytic route");
        assert!(dnu < 0.01, "{name}: {dnu:.3e} cm^-1");
    }
}

// ---------------------------------------------------------------------------
// Mutation ledger
// ---------------------------------------------------------------------------

/// Each mutant misses the FD of the production gradient by >= 10 bars: NH2 /
/// STO-3G, B3LYP (so the exchange mutants have a target).
#[ignore = "tier medium (10-60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
#[test]
fn uks_hessian_mutants_each_fail_the_gradient_difference() {
    if !has_deriv2() {
        return;
    }
    let xc = "B3LYP";
    let cfg = fd_reference_config(&config(xc));
    let p = scf(&nh2(), "sto-3g", &cfg);
    let (fd, _) = fd_of_production_gradient(&p, "sto-3g", &cfg, xc);
    let base = max_abs_diff(&analytic(&p, &cfg), &fd);
    eprintln!("production miss {base:.3e}");
    assert!(base < TOL_HESS_FD);
    let with = |f: fn(&mut KsFixedGridMutants)| {
        let mut m = KsFixedGridMutants::default();
        f(&mut m);
        m
    };
    let cases: Vec<(&str, KsFixedGridMutants)> = vec![
        (
            "drop the alpha-beta kernel block",
            with(|m| m.xc.spin.cross_kernel = 0.0),
        ),
        (
            "total-density kernel",
            with(|m| m.xc.spin.total_kernel = true),
        ),
        (
            "alpha kernel rows for beta",
            with(|m| m.xc.spin.alpha_rows_for_beta = true),
        ),
        (
            "drop the weight terms (W'')",
            with(|m| m.xc.moving.weight_hess = 0.0),
        ),
        (
            "drop the weight term of F^x",
            with(|m| m.xc.moving.fock_weight = 0.0),
        ),
        (
            "drop the kernel from the response",
            with(|m| m.response_fxc = 0.0),
        ),
        (
            "exchange fraction forgotten in the skeleton",
            with(|m| m.skeleton_exx = 1.0 / 0.2),
        ),
        (
            "exchange dropped from the response",
            with(|m| m.response_exx = 0.0),
        ),
    ];
    let mut weakest = f64::INFINITY;
    for (name, m) in cases {
        let miss = max_abs_diff(&mutated(&p, &cfg, xc, &m), &fd);
        let bars = miss / TOL_HESS_FD;
        eprintln!("mutant [{name}]: miss {miss:.3e} ({bars:.1e} bars)");
        weakest = weakest.min(bars);
        assert!(
            bars >= MUTANT_FACTOR,
            "mutant [{name}] misses by only {bars:.1} bars"
        );
    }
    eprintln!("weakest mutant: {weakest:.1e} bars");
}
