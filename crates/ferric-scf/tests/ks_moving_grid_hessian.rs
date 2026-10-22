//! Opt-in analytic Hessian of closed-shell Kohn-Sham on the MOVING (production)
//! grid: `ferric_scf::hessian::rks_moving_grid_hessian`, issue #490. The grid
//! points ride their home atoms and the Becke weights move with the nuclei, so
//! this is the derivative of the surface `ks_gradient_closed` differentiates.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test ks_moving_grid_hessian
//! ```
//!
//! # What is compared to what
//!
//! * Exactness anchor: one atom. Every point rides the only nucleus, so the
//!   XC energy at fixed density is translation invariant and its skeleton and
//!   first-order Fock matrix are EXACTLY zero (the fixed grid's are not). This
//!   anchor is blind to the weight terms (one atom has no Becke derivatives).
//! * XC pieces at a FROZEN density (no SCF): the moving skeleton vs a central
//!   difference of `ferric_dft`'s moving-grid XC gradient (an independent code:
//!   it differentiates the weights through the riding `weight1`, not through
//!   point-fixed second derivatives and a translation fill); the moving `F^x`
//!   vs a central difference of `V_xc` on a grid REBUILT at each displaced
//!   geometry.
//! * The translation sum rule on the skeleton, `F^x` and the full Hessian. It
//!   is blind to the weight-term mutants, so it never stands alone.
//! * The full Hessian vs a central difference of the PRODUCTION gradient
//!   (`ks_gradient_closed_with_exchange_on_grids`, production SCF at every
//!   displaced geometry, exact J, same grid).
//!
//! # Artifact hypothesis next to the physics hypothesis
//!
//! * Right: the frozen-density pieces match their differences at the stencil's
//!   truncation; the full Hessian matches the gradient difference at the
//!   SCF-noise / truncation floor; the sum rule holds to rounding.
//! * A missing or mis-signed weight term, or a wrong translation fill: the
//!   frozen-density pieces miss by the weight-response scale (O(1) terms that
//!   cancel to O(1e-3)), which each mutant below must exceed by >= 10 bars.
//! * The reference secretly differentiates a frozen grid: the moving Hessian
//!   would miss by the weight-response scale; the fixed-grid Hessian's miss of
//!   the same reference is recorded next to the moving one.
//!
//! # Mutation ledger (each must fail an anchor; measured in
//! `moving_grid_mutants_each_fail_a_frozen_density_anchor`)
//!
//! drop `W''`; drop the `W'·f'` cross terms; drop the weight term of `F^x`;
//! skip the translation fill; flip the sign of the fill; the riding `weight1`
//! cross terms unfilled in place of the point-fixed ones filled; batches not
//! split at home-atom boundaries (wrong home past a boundary).

mod ks_moving_grid_common;

use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::operator::Operator;
use ferric_scf::hessian::{rks_moving_grid_hessian_parts_with, KsFixedGridMutants};
use ferric_scf::ks_fixed_grid::{FixedGridKs, MovingMutants, XcGeometry, XcMutants};
use ferric_scf::rhf::RhfConfig;
use ks_moving_grid_common::*;

// ---------------------------------------------------------------------------
// Exactness anchor
// ---------------------------------------------------------------------------

/// One atom: the moving-grid XC skeleton and `F^x` are exactly zero (every
/// point rides the only nucleus), while the fixed grid's are not.
#[test]
fn single_atom_moving_xc_pieces_are_exactly_zero() {
    let mol = Molecule::parse_xyz("1\nNe\nNe 0.1 -0.2 0.3\n", 0, 1).unwrap();
    for xc in ["LDA", "PBE"] {
        let p = scf(&mol, "6-31g", &config(xc));
        let grid = FixedGridKs::new(&mol, xc, &grid_cfg()).unwrap();
        let geo = XcGeometry::new(&mol, &p.prep).unwrap();
        let d = p.res.density_r();
        let m = XcMutants::default();
        let skel = grid.skeleton_hessian_moving(&geo, d, &m).unwrap();
        let fock = grid.first_order_fock_moving(&geo, d, &m).unwrap();
        let fixed = grid.skeleton_hessian(&geo, d, &m).unwrap();
        eprintln!(
            "{xc} Ne: moving skeleton {:.1e}, moving F^x {:.1e}; fixed-grid skeleton {:.3e}",
            max_abs(&skel),
            fock.iter().map(max_abs).fold(0.0, f64::max),
            max_abs(&fixed)
        );
        assert_eq!(max_abs(&skel), 0.0);
        assert!(fock.iter().all(|f| max_abs(f) == 0.0));
        assert!(
            max_abs(&fixed) > 1e-6,
            "the fixed grid must not be trivially zero"
        );
    }
}

// ---------------------------------------------------------------------------
// Frozen-density pieces
// ---------------------------------------------------------------------------

/// Moving skeleton and `F^x` vs the frozen-density differences (LDA and PBE,
/// water / STO-3G and 6-31G).
#[test]
fn moving_xc_pieces_match_frozen_density_differences() {
    for xc in ["LDA", "PBE"] {
        for basis_name in ["sto-3g", "6-31g"] {
            let p = scf(&water(), basis_name, &config(xc));
            let r = frozen_reference(&p, xc);
            let (skel, fock) = frozen_misses(&p, xc, &r, &XcMutants::default());
            let grid = FixedGridKs::new(&p.mol, xc, &grid_cfg()).unwrap();
            let geo = XcGeometry::new(&p.mol, &p.prep).unwrap();
            let fixed = grid
                .skeleton_hessian(&geo, p.res.density_r(), &XcMutants::default())
                .unwrap();
            eprintln!(
                "{xc} {basis_name}: moving skeleton vs FD {skel:.3e}, moving F^x vs FD {fock:.3e}; \
                 fixed-grid skeleton vs the same FD {:.3e}",
                max_abs_diff(&fixed, &r.skel)
            );
            assert!(
                skel < TOL_SKELETON,
                "{xc} {basis_name}: skeleton {skel:.3e}"
            );
            assert!(fock < TOL_FOCK, "{xc} {basis_name}: F^x {fock:.3e}");
            assert!(
                max_abs_diff(&fixed, &r.skel) > MUTANT_FACTOR * TOL_SKELETON,
                "the fixed grid must miss the moving reference"
            );
        }
    }
}

/// The translation sum rule of the moving pieces (skeleton and `F^x`) holds to
/// rounding; the fixed grid's does not.
#[test]
fn moving_xc_pieces_satisfy_the_translation_sum_rule() {
    let p = scf(&water(), "6-31g", &config("PBE"));
    let grid = FixedGridKs::new(&p.mol, "PBE", &grid_cfg()).unwrap();
    let geo = XcGeometry::new(&p.mol, &p.prep).unwrap();
    let d = p.res.density_r();
    let m = XcMutants::default();
    let skel = grid.skeleton_hessian_moving(&geo, d, &m).unwrap();
    let fixed = grid.skeleton_hessian(&geo, d, &m).unwrap();
    let fock = grid.first_order_fock_moving(&geo, d, &m).unwrap();
    let fsum = (0..3)
        .map(|a| {
            let s = &fock[a] + &fock[3 + a] + &fock[6 + a];
            max_abs(&s)
        })
        .fold(0.0, f64::max);
    eprintln!(
        "sum rule: moving skeleton {:.2e}, moving F^x {fsum:.2e}, fixed-grid skeleton {:.2e}",
        sum_rule(&skel),
        sum_rule(&fixed)
    );
    assert!(sum_rule(&skel) < 1e-12 && fsum < 1e-12);
    assert!(sum_rule(&fixed) > 1e-6);
}

/// Each moving-grid mutant misses a frozen-density anchor by >= 10 bars.
#[test]
fn moving_grid_mutants_each_fail_a_frozen_density_anchor() {
    let xc = "PBE";
    let p = scf(&water(), "6-31g", &config(xc));
    let r = frozen_reference(&p, xc);
    let with = |f: fn(&mut MovingMutants)| {
        let mut m = XcMutants::default();
        f(&mut m.moving);
        m
    };
    let cases: Vec<(&str, XcMutants)> = vec![
        ("drop W''", with(|m| m.weight_hess = 0.0)),
        ("drop the W' f' cross terms", with(|m| m.weight_cross = 0.0)),
        ("drop the weight term of F^x", with(|m| m.fock_weight = 0.0)),
        ("skip the translation fill", with(|m| m.fill = 0.0)),
        ("flip the sign of the fill", with(|m| m.fill = 1.0)),
        (
            "riding weight1 cross terms, unfilled",
            with(|m| m.riding_cross = true),
        ),
        (
            "wrong home past a batch boundary",
            with(|m| m.unsplit_batches = true),
        ),
    ];
    let mut weakest = f64::INFINITY;
    for (name, m) in cases {
        let (skel, fock) = frozen_misses(&p, xc, &r, &m);
        let bars = (skel / TOL_SKELETON).max(fock / TOL_FOCK);
        eprintln!(
            "mutant [{name}]: skeleton miss {skel:.3e}, F^x miss {fock:.3e} ({bars:.1e} bars)"
        );
        weakest = weakest.min(bars);
        assert!(
            bars >= MUTANT_FACTOR,
            "mutant [{name}] misses by only {bars:.1} bars"
        );
    }
    eprintln!("weakest mutant: {weakest:.1e} bars");
}

// ---------------------------------------------------------------------------
// Full Hessian vs the production gradient
// ---------------------------------------------------------------------------

#[test]
fn moving_grid_hessian_matches_difference_of_the_production_gradient() {
    if !has_deriv2() {
        return;
    }
    for xc in ["LDA", "PBE"] {
        full_hessian_case(&config(xc), xc, "sto-3g", TOL_HESS_FD);
    }
}

/// The counterpart of `ks_fixed_grid_hessian.rs`'s negative control
/// `production_gradient_difference_differs_by_the_weight_response_scale`:
/// the same system (water / 6-31G, LDA, exact J, DEFAULT grid, which is what
/// the plain `ks_gradient_closed` differentiates) and the same reference (FD
/// of the production gradient), which the fixed-grid Hessian misses and the
/// moving-grid Hessian must match.
#[test]
fn production_gradient_difference_matches_the_moving_grid_hessian() {
    if !has_deriv2() {
        return;
    }
    let cfg = RhfConfig {
        dft_grid: None,
        ..config("LDA")
    };
    full_hessian_case(&cfg, "LDA", "6-31g", TOL_HESS_FD);
}

/// The grid of the moving Hessian must be the one the SCF ran on: a grid
/// from explicit points (no home atoms) is refused.
#[test]
fn a_grid_without_home_atoms_is_refused() {
    let p = scf(&water(), "sto-3g", &config("LDA"));
    let built = FixedGridKs::new(&p.mol, "LDA", &grid_cfg()).unwrap();
    let (pts, w) = built.points_and_weights();
    let bare = FixedGridKs::from_points("LDA", pts.to_vec(), w.to_vec()).unwrap();
    let Err(err) = rks_moving_grid_hessian_parts_with(
        &ParallelContext::default(),
        &p.mol,
        &p.prep,
        Operator::coulomb(),
        &p.bounds,
        &p.res,
        &config("LDA"),
        &bare,
        &KsFixedGridMutants::default(),
    ) else {
        panic!("a grid without home atoms must be refused");
    };
    assert!(format!("{err:?}").contains("home atoms"), "{err:?}");
}
