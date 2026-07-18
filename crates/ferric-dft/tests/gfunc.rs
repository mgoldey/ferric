//! Correctness tests for g-function (l=4) support in the DFT grid AO evaluator.
//!
//! The gold-standard check is grid-integrated overlap vs. the analytic overlap
//! from the integrals engine: it validates BOTH the angular normalization and
//! the libint2 function ordering simultaneously. A wrong g angular function
//! silently corrupts every DFT/KS calculation on a g-basis, so these tests must
//! pass before the l=4 arms may ship.

use ferric_core::basis::{self, BasisSet, Shell};
use ferric_core::mol::Molecule;
use ferric_dft::ao_grid::eval_basis_on_points;
use ferric_dft::ao_grid::{eval_basis_and_grad_on_points, eval_basis_grad_hess_on_points};
use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron::overlap;

/// Numerically integrate S_grid[i,j] = Σ_g w_g χ_i(r_g) χ_j(r_g) on a dense
/// Becke-Lebedev grid and compare to the analytic overlap matrix. Agreement to
/// grid accuracy proves the g angular functions match libint2 in both
/// normalization and ordering.
fn grid_vs_analytic_overlap(mol: &Molecule, bs: &BasisSet, tol: f64) -> f64 {
    let obs = PreparedBasis::new(mol, bs).unwrap();
    let s_analytic = overlap(&obs);

    // Dense grid: g functions are high angular frequency, need many angular pts.
    let cfg = AtomicGridConfig { n_radial: 120, n_angular: 302 };
    let grid = build_atomic_grid(mol, &cfg);
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let chi = eval_basis_on_points(mol, bs, &pts).unwrap();

    let nbf = chi.nrows();
    assert_eq!(nbf, s_analytic.nrows(), "nbasis mismatch grid vs analytic");

    let weights: Vec<f64> = grid.iter().map(|g| g.weight).collect();

    let mut max_err = 0.0_f64;
    for i in 0..nbf {
        for j in 0..nbf {
            let mut s = 0.0;
            for g in 0..pts.len() {
                s += weights[g] * chi[(i, g)] * chi[(j, g)];
            }
            let err = (s - s_analytic[(i, j)]).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    eprintln!("max |S_grid - S_analytic| = {max_err:.3e} (tol {tol:.1e})");
    max_err
}

#[test]
fn grid_overlap_matches_analytic_pure_g_h2o_def2qzvp() {
    // def2-QZVP puts a pure (spherical) g shell on oxygen — the hot path for
    // spherical bases (aug-cc-pVTZ etc.). This is the primary correctness gate.
    let mol = Molecule::parse_xyz(
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
        0,
        1,
    )
    .unwrap();
    let bs = basis::bundled("def2-qzvp").unwrap();
    let max_err = grid_vs_analytic_overlap(&mol, &bs, 1e-4);
    assert!(
        max_err < 1e-4,
        "pure-g grid overlap mismatch: {max_err:.3e}"
    );
}

/// Build a minimal single-atom BasisSet carrying one Cartesian g shell (plus a
/// tight s so the atom is a real element) to exercise the (4, false) arm.
fn cartesian_g_basis() -> BasisSet {
    use std::collections::HashMap;
    // A single, moderately diffuse primitive g shell (one contraction).
    let g = Shell {
        l: 4,
        pure: false,
        exponents: vec![0.8],
        coefficients: vec![1.0],
    };
    let s = Shell {
        l: 0,
        pure: false,
        exponents: vec![1.0],
        coefficients: vec![1.0],
    };
    let mut shells = HashMap::new();
    shells.insert(6, vec![s, g]); // put on carbon (Z=6)
    BasisSet {
        name: "synthetic-cart-g".to_string(),
        shells,
        ecps: HashMap::new(),
    }
}

#[test]
fn grid_overlap_matches_analytic_cartesian_g() {
    // Single carbon atom at origin with one Cartesian g shell.
    let mol = Molecule::parse_xyz("1\nC\nC 0 0 0\n", 0, 1).unwrap();
    let bs = cartesian_g_basis();
    let max_err = grid_vs_analytic_overlap(&mol, &bs, 1e-4);
    assert!(
        max_err < 1e-4,
        "Cartesian-g grid overlap mismatch: {max_err:.3e}"
    );
}

#[test]
fn gradient_matches_central_difference_covers_g_shells() {
    // def2-QZVP has a g shell on O. Probe off the cusp so radial derivatives
    // are well-conditioned.
    let mol = Molecule::parse_xyz(
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
        0,
        1,
    )
    .unwrap();
    let bs = basis::bundled("def2-qzvp").unwrap();
    let p = [[0.15_f64, 0.1, -0.2]];
    let (_chi, dchi) = eval_basis_and_grad_on_points(&mol, &bs, &p).unwrap();
    let eps = 1e-5_f64;
    let mut max_err = 0.0_f64;
    for axis in 0..3 {
        let mut p_plus = p;
        let mut p_minus = p;
        p_plus[0][axis] += eps;
        p_minus[0][axis] -= eps;
        let chi_plus = eval_basis_on_points(&mol, &bs, &p_plus).unwrap();
        let chi_minus = eval_basis_on_points(&mol, &bs, &p_minus).unwrap();
        for mu in 0..chi_plus.nrows() {
            let fd = (chi_plus[(mu, 0)] - chi_minus[(mu, 0)]) / (2.0 * eps);
            let analytical = dchi[(axis, mu, 0)];
            let err = (fd - analytical).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    eprintln!("[def2-qzvp] max |grad_ana - grad_FD| = {max_err:.2e}");
    assert!(max_err < 1e-5, "def2-qzvp AO grad FD mismatch {max_err:.2e}");
}

#[test]
fn gradient_matches_fd_cartesian_g() {
    let mol = Molecule::parse_xyz("1\nC\nC 0 0 0\n", 0, 1).unwrap();
    let bs = cartesian_g_basis();
    let p = [[0.3_f64, -0.2, 0.4]];
    let (_chi, dchi) = eval_basis_and_grad_on_points(&mol, &bs, &p).unwrap();
    let eps = 1e-5_f64;
    let mut max_err = 0.0_f64;
    for axis in 0..3 {
        let mut pp = p;
        let mut pm = p;
        pp[0][axis] += eps;
        pm[0][axis] -= eps;
        let chi_p = eval_basis_on_points(&mol, &bs, &pp).unwrap();
        let chi_m = eval_basis_on_points(&mol, &bs, &pm).unwrap();
        for mu in 0..chi_p.nrows() {
            let fd = (chi_p[(mu, 0)] - chi_m[(mu, 0)]) / (2.0 * eps);
            let err = (fd - dchi[(axis, mu, 0)]).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    eprintln!("[cart-g] max |grad_ana - grad_FD| = {max_err:.2e}");
    assert!(max_err < 1e-5, "cart-g AO grad FD mismatch {max_err:.2e}");
}

#[test]
fn hessian_matches_fd_covers_g_shells() {
    let mol = Molecule::parse_xyz(
        "3\nH2O\nO 0 0 0\nH 0 0.7572 0.5868\nH 0 -0.7572 0.5868\n",
        0,
        1,
    )
    .unwrap();
    let bs = basis::bundled("def2-qzvp").unwrap();
    let pts = vec![[0.3_f64, 0.2, 0.4], [-0.5, 0.6, 0.1], [1.1, -0.3, 0.5]];
    let (_chi, _dchi, ddchi) = eval_basis_grad_hess_on_points(&mol, &bs, &pts).unwrap();
    let h = 1e-5_f64;
    let mut max_diff = 0.0_f64;
    for (g, p) in pts.iter().enumerate() {
        for b in 0..3 {
            let mut pp = *p;
            pp[b] += h;
            let mut pm = *p;
            pm[b] -= h;
            let (_cp, dchi_p) = eval_basis_and_grad_on_points(&mol, &bs, &[pp]).unwrap();
            let (_cm, dchi_m) = eval_basis_and_grad_on_points(&mol, &bs, &[pm]).unwrap();
            let nbf = dchi_p.dim().1;
            for a in 0..3 {
                for mu in 0..nbf {
                    let fd = (dchi_p[(a, mu, 0)] - dchi_m[(a, mu, 0)]) / (2.0 * h);
                    let ana = ddchi[(a, b, mu, g)];
                    let diff = (ana - fd).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                }
            }
        }
    }
    eprintln!("[def2-qzvp] max |Hess_ana - Hess_FD| = {max_diff:.2e}");
    assert!(max_diff < 1e-5, "def2-qzvp Hessian FD mismatch {max_diff:.2e}");
}

#[test]
fn hessian_matches_fd_cartesian_g() {
    let mol = Molecule::parse_xyz("1\nC\nC 0 0 0\n", 0, 1).unwrap();
    let bs = cartesian_g_basis();
    let pts = vec![[0.3_f64, 0.2, 0.4], [-0.5, 0.6, 0.1]];
    let (_chi, _dchi, ddchi) = eval_basis_grad_hess_on_points(&mol, &bs, &pts).unwrap();
    let h = 1e-5_f64;
    let mut max_diff = 0.0_f64;
    for (g, p) in pts.iter().enumerate() {
        for b in 0..3 {
            let mut pp = *p;
            pp[b] += h;
            let mut pm = *p;
            pm[b] -= h;
            let (_cp, dchi_p) = eval_basis_and_grad_on_points(&mol, &bs, &[pp]).unwrap();
            let (_cm, dchi_m) = eval_basis_and_grad_on_points(&mol, &bs, &[pm]).unwrap();
            let nbf = dchi_p.dim().1;
            for a in 0..3 {
                for mu in 0..nbf {
                    let fd = (dchi_p[(a, mu, 0)] - dchi_m[(a, mu, 0)]) / (2.0 * h);
                    let diff = (ddchi[(a, b, mu, g)] - fd).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                }
            }
        }
    }
    eprintln!("[cart-g] max |Hess_ana - Hess_FD| = {max_diff:.2e}");
    assert!(max_diff < 1e-5, "cart-g Hessian FD mismatch {max_diff:.2e}");
}
