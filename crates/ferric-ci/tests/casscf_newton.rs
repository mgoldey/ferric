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
//! 3. AH (quadratic and constant forcing) and inverse BFGS converge to the
//!    same energy: bar `(‖g_AH‖² + ‖g_BFGS‖²)/(2 λ_min) + 8 ε |E| √N_det`,
//!    `λ_min` the lowest eigenvalue of the finite-difference CI-relaxed
//!    orbital Hessian at the AH orbitals (`tests/casscf.rs` anchor 3).
//! 4. The occupied-kernel product ([`HessianProducts::OccupiedKernel`])
//!    against the AO J/K product ([`HessianProducts::AoJk`]) on the same
//!    dense AO integrals, same point and unit `x`: only the summation order
//!    differs. With `e_max = max Q²` (Schwarz) bounding every AO integral, `c1`, `co1`, `ca1` the
//!    largest column 1-norms of all, the occupied and the active orbitals,
//!    and `xs` that of `X`:
//!    * AO J/K: `δG <= 2 · 1.5 nbas² ε e_max Σ|dD|` (the bar of
//!      `tests/casscf_direct.rs`), `Σ|dD^c| <= 4 n_c c1² xs`,
//!      `Σ|dD^a| <= 4 n_a² c1 ca1 xs`, carried to the MO basis by `c1²`;
//!    * kernel: `(pq|jk)`, `(pj|qk)` are four chained sums of `nbas` terms,
//!      `δQ <= 4 nbas ε e_max c1² co1²` per route; `L` combines three with
//!      weights 2, ½, ½, and `L vec(Y)` sums `nmo n_occ` terms, so
//!      `δG <= (3 δQ + nmo n_occ ε max|L|) Σ|Y|`, `Σ|Y^c| <= 2 n_c xs`,
//!      `Σ|Y^a| <= 2 n_a² xs`;
//!    * the active blocks of `W` and `Σ (pu|vw) Γ` come from transforms with
//!      `n_occ` and `n_a` columns, `δQ_a <= 2 · 4 nbas ε e_max c1² co1²`.
//!
//!    `|δ(Hx)| <= 4 (2 (δG_c + δG_a) + 2 n_a δG_c + 12 n_a² δQ_a ‖x‖₁
//!    + 4 n_a³ δQ_a xs)` as in `tests/casscf_direct.rs`.
//!
//! Measured numbers and mutations:
//! `book/src/17-multireference/05-casscf.md#casscf-proven-optimizers`.

mod casscf_common;

use casscf_common::{hessian_min_eig, maxabs, molecule_text, setup_text, EPS};
use ferric_ci::casscf::{run_casscf, AhForcing, CasScfConfig, CasScfOptimizer, CasScfProblem};
use ferric_ci::casscf_newton::HessianProducts;
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

fn same_minimum(
    file: &str,
    n_core: usize,
    n_active: usize,
    nel: (usize, usize),
    ah_forcing: AhForcing,
) {
    let (mol, prep, rhf) = setup_text(&molecule_text(file), "6-31g");
    let run = |optimizer| {
        let cfg = CasScfConfig {
            n_active,
            n_elec_active: nel,
            n_core,
            optimizer,
            ah_forcing,
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
        "{file} {ah_forcing:?}: AH E {:.13} ({} iterations, |g| {:.1e}), BFGS E {:.13} ({} iterations, \
         |g| {:.1e}); dE {de:.2e} (bar {bar:.2e}); lambda_min {lam:.4}",
        ah.e_total, ah.iterations, ah.grad_norm, bfgs.e_total, bfgs.iterations, bfgs.grad_norm
    );
    assert!(de.abs() <= bar, "{file}: |dE| {:e} > bar {bar:e}", de.abs());
}

#[test]
fn ah_and_bfgs_reach_the_same_minimum_h2o_cas44_631g() {
    same_minimum("water.xyz", 3, 4, (2, 2), AhForcing::Quadratic);
}

#[test]
fn ah_and_bfgs_reach_the_same_minimum_o2_triplet_cas86_631g() {
    same_minimum("o2.xyz", 4, 6, (5, 3), AhForcing::Quadratic);
}

#[test]
fn ah_constant_forcing_and_bfgs_reach_the_same_minimum_h2o_cas44_631g() {
    same_minimum("water.xyz", 3, 4, (2, 2), AhForcing::Constant(0.3));
}

#[test]
fn ah_constant_forcing_and_bfgs_reach_the_same_minimum_o2_triplet_cas86_631g() {
    same_minimum("o2.xyz", 4, 6, (5, 3), AhForcing::Constant(0.3));
}

#[test]
fn an_ah_forcing_constant_outside_zero_one_is_refused() {
    let (mol, prep, rhf) = setup_text(&molecule_text("water.xyz"), "sto-3g");
    for eta in [0.0, 1.0, -0.1, f64::NAN] {
        let cfg = CasScfConfig {
            n_active: 4,
            n_elec_active: (2, 2),
            n_core: 3,
            optimizer: CasScfOptimizer::AugmentedHessian,
            ah_forcing: AhForcing::Constant(eta),
            ..Default::default()
        };
        let err = run_casscf(&mol, &prep, &rhf, &cfg).unwrap_err().to_string();
        assert!(err.contains("ah_forcing"), "eta {eta}: {err}");
    }
}

fn col1(c: ndarray::ArrayView2<'_, f64>) -> f64 {
    c.columns()
        .into_iter()
        .map(|col| col.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

fn kernel_case(file: &str, n_core: usize, n_active: usize, nel: (usize, usize)) {
    let (mol, prep, rhf) = setup_text(&molecule_text(file), "6-31g");
    let cfg = CasScfConfig {
        n_active,
        n_elec_active: nel,
        n_core,
        ci_conv_thresh: 1e-12,
        integrals: ferric_ci::casscf::CasScfIntegrals::Dense,
        optimizer: CasScfOptimizer::AugmentedHessian,
        ..Default::default()
    };
    let nmo = rhf.mos_r().ncols();
    let problem = CasScfProblem::new(&mol, &prep, nmo, &cfg).unwrap();
    let npair = problem.pairs().len();
    let kappa: Vec<f64> = normals(npair, 7).iter().map(|x| 0.05 * x).collect();
    let c = problem.rotate(rhf.mos_r(), &kappa).unwrap();
    let pt = problem.evaluate(&c).unwrap();
    let hk = problem
        .orbital_hessian_with(&c, &pt, HessianProducts::OccupiedKernel)
        .unwrap();
    let ha = problem
        .orbital_hessian_with(&c, &pt, HessianProducts::AoJk)
        .unwrap();
    let x = unit(normals(npair, 11));
    let a = problem.hessian_vector(&hk, &x).unwrap();
    let b = problem.hessian_vector(&ha, &x).unwrap();

    // Schwarz bound max Q² on every AO integral.
    let q = ferric_scf::screening::SchwarzBounds::compute(
        ferric_integrals::operator::Operator::coulomb(),
        &prep,
    )
    .unwrap()
    .q;
    let eri_max = maxabs(q.iter()).powi(2);
    let n = prep.nbasis() as f64;
    let (nc, na) = (n_core as f64, n_active as f64);
    let nocc = n_core + n_active;
    let c1 = col1(c.view());
    let co1 = col1(c.slice(ndarray::s![.., ..nocc]));
    let ca1 = col1(c.slice(ndarray::s![.., n_core..nocc]));
    let mut xm = Array2::<f64>::zeros((nmo, nmo));
    for (&(p, q), &v) in problem.pairs().iter().zip(&x) {
        xm[(p, q)] = v;
        xm[(q, p)] = -v;
    }
    let xs = col1(xm.view());
    let x1: f64 = x.iter().map(|v| v.abs()).sum();
    let dq = 4.0 * n * EPS * eri_max * (c1 * co1).powi(2);
    let l_max = 3.0 * eri_max * (c1 * co1).powi(2);
    let kernel = |sum_y: f64| (3.0 * dq + (nmo * nocc) as f64 * EPS * l_max) * sum_y;
    let ao = |sum_d: f64| c1 * c1 * 2.0 * 1.5 * n * n * EPS * eri_max * sum_d;
    let dg_c = kernel(2.0 * nc * xs) + ao(4.0 * nc * c1 * c1 * xs);
    let dg_a = kernel(2.0 * na * na * xs) + ao(4.0 * na * na * c1 * ca1 * xs);
    let dqa = 2.0 * dq;
    let bar = 4.0
        * (2.0 * (dg_c + dg_a)
            + 2.0 * na * dg_c
            + 12.0 * na * na * dqa * x1
            + 4.0 * na.powi(3) * dqa * xs);
    let d = maxabs(&a.iter().zip(&b).map(|(u, v)| u - v).collect::<Vec<_>>());
    eprintln!(
        "{file}: max|Hx| {:.3}, |kernel − AO J/K| {d:.2e} (bar {bar:.1e})",
        maxabs(&a)
    );
    assert!(maxabs(&a) > 1e-2, "{file}: the product is not exercised");
    assert!(d <= bar, "{file}: |kernel − AO J/K| {d:e} > bar {bar:e}");
}

#[test]
fn occupied_kernel_product_matches_the_ao_jk_product_h2o_cas44_631g() {
    kernel_case("water.xyz", 3, 4, (2, 2));
}

#[test]
fn occupied_kernel_product_matches_the_ao_jk_product_n2_cas66_631g() {
    kernel_case("n2.xyz", 4, 6, (3, 3));
}

#[test]
fn occupied_kernel_product_matches_the_ao_jk_product_o2_triplet_cas86_631g() {
    kernel_case("o2.xyz", 4, 6, (5, 3));
}
