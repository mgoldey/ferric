//! Validation tier of the moving-grid analytic KS Hessian (issue #490): the
//! heavier cases of `ks_moving_grid_hessian.rs` (6-31G and cc-pVDZ, global
//! hybrids, density-fitted J and K, the default grid) and the cost against the
//! fixed-grid Hessian. Same references (central differences of the production
//! gradient with a production SCF at every displaced geometry, and of the
//! moving-grid XC gradient / rebuilt `V_xc` at frozen density); see that file
//! for the artifact hypotheses and the mutation ledger.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release \
//!     --test validation_ks_moving_grid_hessian -- --ignored --nocapture
//! ```

mod ks_moving_grid_common;

use ferric_core::parallel::ParallelContext;
use ferric_integrals::operator::Operator;
use ferric_scf::hessian::{rks_fixed_grid_hessian, rks_moving_grid_hessian};
use ferric_scf::ks_fixed_grid::XcMutants;
use ferric_scf::rhf::RhfConfig;
use ks_moving_grid_common::*;
use std::time::Instant;

/// GGA and global hybrids with exact J and K: PBE at 6-31G, PBE0 at STO-3G
/// and 6-31G, on the 50x110 grid.
#[test]
#[ignore = "validation: moving-grid KS Hessian"]
fn gga_and_hybrid_full_hessians_match_the_production_gradient() {
    if !has_deriv2() {
        return;
    }
    full_hessian_case(&config("PBE"), "PBE", "6-31g", TOL_VALIDATION);
    full_hessian_case(&config("PBE0"), "PBE0", "sto-3g", TOL_VALIDATION);
    full_hessian_case(&config("PBE0"), "PBE0", "6-31g", TOL_VALIDATION);
}

/// The production KS default fits: RI-J (pure functional) and RI-J + RI-K
/// (hybrid) over the default auxiliary basis, route recorded by the SCF.
#[test]
#[ignore = "validation: moving-grid KS Hessian"]
fn density_fitted_full_hessians_match_the_production_gradient() {
    if !has_deriv2() {
        return;
    }
    for xc in ["PBE", "PBE0"] {
        let cfg = RhfConfig {
            df_j_aux: None,
            df_k_aux: None,
            ..config(xc)
        };
        full_hessian_case(&cfg, xc, "sto-3g", TOL_VALIDATION);
    }
}

/// d functions on the default grid: the frozen-density pieces (PBE) and the
/// full Hessian (LDA) at cc-pVDZ, plus the cost of the moving-grid Hessian
/// against the fixed-grid one on the same SCF (one shared box, so the times
/// are indicative; the ratio is the number to carry).
#[test]
#[ignore = "validation: moving-grid KS Hessian"]
fn cc_pvdz_pieces_full_hessian_and_cost() {
    if !has_deriv2() {
        return;
    }
    let default_grid = |xc: &str| RhfConfig {
        dft_grid: None,
        ..config(xc)
    };
    let p = scf(&water(), "cc-pvdz", &default_grid("PBE"));
    let r = frozen_reference(&p, "PBE");
    let (skel, fock) = frozen_misses(&p, "PBE", &r, &XcMutants::default());
    eprintln!("PBE cc-pVDZ: moving skeleton vs FD {skel:.3e}, moving F^x vs FD {fock:.3e}");
    assert!(skel < TOL_SKELETON, "skeleton {skel:.3e}");
    assert!(fock < TOL_FOCK, "F^x {fock:.3e}");

    full_hessian_case(&default_grid("LDA"), "LDA", "cc-pvdz", TOL_VALIDATION);

    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    for xc in ["LDA", "PBE"] {
        let cfg = default_grid(xc);
        let p = scf(&water(), "cc-pvdz", &cfg);
        let t = Instant::now();
        let fixed = rks_fixed_grid_hessian(&ctx, &p.mol, &p.prep, op, &p.bounds, &p.res, &cfg);
        let t_fixed = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let moving = rks_moving_grid_hessian(&ctx, &p.mol, &p.prep, op, &p.bounds, &p.res, &cfg);
        let t_moving = t.elapsed().as_secs_f64();
        let (fixed, moving) = (fixed.unwrap(), moving.unwrap());
        eprintln!(
            "{xc} H2O/cc-pVDZ, 75x110 grid: fixed-grid Hessian {t_fixed:.2} s, moving-grid \
             {t_moving:.2} s (ratio {:.2}); max |H_moving - H_fixed| {:.3e}",
            t_moving / t_fixed,
            max_abs_diff(&moving, &fixed)
        );
    }
}

/// Full Hessian vs FD of the production gradient in this tier, Ha/Bohr².
/// Measured max 1.05e-9 (PBE / 6-31G, h = 5e-4); the other cases 6.0e-11 to
/// 2.6e-10 (PBE0 STO-3G and 6-31G, RI-J PBE, RI-J + RI-K PBE0, LDA cc-pVDZ). The
/// PBE / 6-31G residual is NOT stencil truncation: at h = 1e-3, 5e-4, 2.5e-4 it
/// is 5.5e-10, 1.05e-9, 1.6e-10 (no h^4 trend), i.e. SCF / GGA-grid noise
/// divided by h. Bar 2e-8: 19x that, and 1e5 below the fixed-grid misses
/// (2.3e-3 to 9.7e-3) and the weakest mutant.
const TOL_VALIDATION: f64 = 2e-8;
