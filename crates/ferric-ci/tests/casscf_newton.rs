//! Anchors of the fixed-RDM orbital Hessian and the augmented-Hessian
//! CASSCF step (`src/casscf_newton.rs`).
//!
//! 1. `H_f x` ([`CasScfProblem::hessian_vector_frame`]) against a central
//!    difference of [`CasScfProblem::gradient_at_fixed_rdms`] along `x`, at
//!    RHF orbitals rotated by a random `κ` (scale 0.05, a point with a large
//!    gradient) and a random unit `x`. The central difference at step `h`
//!    carries the truncation error `c h²` (plus `O(h⁴)`), so with
//!    `Δ = FD(h) − FD(h/2) = ¾ c h²` the error of `FD(h/2)` is `Δ/3` to
//!    leading order. Bar per element: `2 |Δ|/3 + r`, the factor 2 for the
//!    `O(h⁴)` remainder, and `r = 2 · 64 ε max|g| / (h/2)` the rounding of two
//!    gradients (each a sum of O(64) rounded terms of size `max|g|`) divided
//!    by the step. The gradient at fixed RDMs has no CI error.
//! 2. `H` ([`CasScfProblem::hessian_vector`]) is symmetric:
//!    `|yᵀHx − xᵀHy| <= 64 ε n (max|Hx| + max|Hy|)` for unit `x`, `y`
//!    (`n` pairs; each dot product sums `n` terms rounded at `64 ε` of their
//!    size). The frame derivative is not symmetric away from a stationary
//!    point, so this checks the `−½ [G, X]` correction.
//! 3. AH and inverse BFGS converge to the same energy: bar
//!    `(‖g_AH‖² + ‖g_BFGS‖²)/(2 λ_min) + 8 ε |E| √N_det`, `λ_min` the lowest
//!    eigenvalue of the finite-difference CI-relaxed orbital Hessian at the
//!    AH orbitals (`tests/casscf.rs` anchor 3).
//!
//! Measured numbers and mutations: `crates/ferric-ci/theory.md`.

mod casscf_common;

use casscf_common::{hessian_min_eig, maxabs, molecule_text, setup_text, EPS};
use ferric_ci::casscf::{run_casscf, CasScfConfig, CasScfOptimizer, CasScfProblem};
use ndarray::Array2;

fn normals(n: usize, seed: u64) -> Vec<f64> {
    let mut st = seed;
    let mut next = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let (u1, u2) = (next(), next());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        })
        .collect()
}

fn unit(mut v: Vec<f64>) -> Vec<f64> {
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.iter_mut().for_each(|x| *x /= n);
    v
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn hessian_case(file: &str, n_core: usize, n_active: usize, nel: (usize, usize)) {
    let (mol, prep, rhf) = setup_text(&molecule_text(file), "6-31g");
    let cfg = CasScfConfig {
        n_active,
        n_elec_active: nel,
        n_core,
        ci_conv_thresh: 1e-11,
        optimizer: CasScfOptimizer::AugmentedHessian,
        ..Default::default()
    };
    let c_rhf = rhf.mos_r().clone();
    let problem = CasScfProblem::new(&mol, &prep, c_rhf.ncols(), &cfg).unwrap();
    let n = problem.pairs().len();
    let kappa: Vec<f64> = normals(n, 7).iter().map(|x| 0.05 * x).collect();
    let c = problem.rotate(&c_rhf, &kappa).unwrap();
    let pt = problem.evaluate(&c).unwrap();
    let rdms = pt.rdms().clone();
    let hess = problem.orbital_hessian(&c, &pt).unwrap();
    let x = unit(normals(n, 11));

    let fd = |h: f64| -> Vec<f64> {
        let xp: Vec<f64> = x.iter().map(|v| h * v).collect();
        let xm: Vec<f64> = x.iter().map(|v| -h * v).collect();
        let gp = problem
            .gradient_at_fixed_rdms(&problem.rotate(&c, &xp).unwrap(), &rdms.dm1, &rdms.dm2)
            .unwrap();
        let gm = problem
            .gradient_at_fixed_rdms(&problem.rotate(&c, &xm).unwrap(), &rdms.dm1, &rdms.dm2)
            .unwrap();
        gp.iter()
            .zip(&gm)
            .map(|(a, b)| (a - b) / (2.0 * h))
            .collect()
    };
    let h = 1e-3;
    let (f1, f2) = (fd(h), fd(h / 2.0));
    let an = problem.hessian_vector_frame(&hess, &x).unwrap();
    let gmax = maxabs(&pt.gradient);
    let round = 2.0 * 64.0 * EPS * gmax / (h / 2.0);
    let mut worst = 0.0f64;
    let mut worst_bar = 0.0f64;
    let mut worst_ratio = 0.0f64;
    for k in 0..n {
        let delta = (f1[k] - f2[k]).abs();
        let bar = 2.0 * delta / 3.0 + round;
        let err = (an[k] - f2[k]).abs();
        if err / bar > worst_ratio {
            (worst, worst_bar, worst_ratio) = (err, bar, err / bar);
        }
    }
    let rich: Vec<f64> = f1
        .iter()
        .zip(&f2)
        .map(|(a, b)| (4.0 * b - a) / 3.0)
        .collect();
    let rich_err = maxabs(&an.iter().zip(&rich).map(|(a, b)| a - b).collect::<Vec<_>>());
    eprintln!(
        "{file}: {n} pairs, max|H_f x| {:.3}, max|g| {gmax:.3}; worst |H_f x − FD(h/2)| \
         {worst:.2e} against bar {worst_bar:.2e} (ratio {worst_ratio:.2e}); \
         max|FD(h) − FD(h/2)| {:.2e}; max|H_f x − Richardson| {rich_err:.2e}",
        maxabs(&an),
        maxabs(&f1.iter().zip(&f2).map(|(a, b)| a - b).collect::<Vec<_>>()),
    );
    assert!(
        maxabs(&an) > 1e-2,
        "{file}: the product vanishes, the check is vacuous"
    );
    assert!(
        worst_ratio <= 1.0,
        "{file}: |H_f x − FD| {worst:e} > bar {worst_bar:e}"
    );

    let y = unit(normals(n, 13));
    let hx = problem.hessian_vector(&hess, &x).unwrap();
    let hy = problem.hessian_vector(&hess, &y).unwrap();
    let asym = (dot(&y, &hx) - dot(&x, &hy)).abs();
    let sym_bar = 64.0 * EPS * n as f64 * (maxabs(&hx) + maxabs(&hy));
    let frame_asym =
        (dot(&y, &an) - dot(&x, &problem.hessian_vector_frame(&hess, &y).unwrap())).abs();
    eprintln!(
        "{file}: |yᵀHx − xᵀHy| {asym:.2e} (bar {sym_bar:.2e}); frame derivative \
         asymmetry {frame_asym:.2e}"
    );
    assert!(
        frame_asym > 1e3 * sym_bar,
        "{file}: the frame derivative is symmetric here, so the symmetry check is vacuous"
    );
    assert!(
        asym <= sym_bar,
        "{file}: Hessian asymmetry {asym:e} > bar {sym_bar:e}"
    );
}

#[test]
fn hessian_vector_matches_fd_of_the_gradient_h2o_cas44_631g() {
    hessian_case("water.xyz", 3, 4, (2, 2));
}

#[test]
fn hessian_vector_matches_fd_of_the_gradient_n2_cas66_631g() {
    hessian_case("n2.xyz", 4, 6, (3, 3));
}

#[test]
fn hessian_vector_matches_fd_of_the_gradient_o2_triplet_cas86_631g() {
    hessian_case("o2.xyz", 4, 6, (5, 3));
}

fn same_minimum(file: &str, n_core: usize, n_active: usize, nel: (usize, usize)) {
    let (mol, prep, rhf) = setup_text(&molecule_text(file), "6-31g");
    let run = |optimizer| {
        let cfg = CasScfConfig {
            n_active,
            n_elec_active: nel,
            n_core,
            optimizer,
            ..Default::default()
        };
        let res = run_casscf(&mol, &prep, &rhf, &cfg).unwrap();
        assert!(res.converged, "{file}: {optimizer:?} did not converge");
        (cfg, res)
    };
    let (cfg, ah) = run(CasScfOptimizer::AugmentedHessian);
    let (_, bfgs) = run(CasScfOptimizer::Bfgs);
    let problem = CasScfProblem::new(&mol, &prep, ah.mo_coeff.ncols(), &cfg).unwrap();
    let c: &Array2<f64> = &ah.mo_coeff;
    let lam = hessian_min_eig(&problem, c, 1e-4);
    assert!(lam > 0.0, "{file}: lowest Hessian eigenvalue {lam:e} <= 0");
    let round = 8.0 * EPS * ah.e_total.abs() * (ah.n_determinants as f64).sqrt();
    let bar = (ah.grad_norm.powi(2) + bfgs.grad_norm.powi(2)) / (2.0 * lam) + round;
    let de = ah.e_total - bfgs.e_total;
    eprintln!(
        "{file}: AH E {:.13} ({} iterations, |g| {:.1e}), BFGS E {:.13} ({} iterations, \
         |g| {:.1e}); dE {de:.2e} (bar {bar:.2e}); lambda_min {lam:.4}",
        ah.e_total, ah.iterations, ah.grad_norm, bfgs.e_total, bfgs.iterations, bfgs.grad_norm
    );
    assert!(de.abs() <= bar, "{file}: |dE| {:e} > bar {bar:e}", de.abs());
}

#[test]
fn ah_and_bfgs_reach_the_same_minimum_h2o_cas44_631g() {
    same_minimum("water.xyz", 3, 4, (2, 2));
}

#[test]
fn ah_and_bfgs_reach_the_same_minimum_o2_triplet_cas86_631g() {
    same_minimum("o2.xyz", 4, 6, (5, 3));
}
