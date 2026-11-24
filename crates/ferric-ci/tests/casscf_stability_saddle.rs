//! The CASSCF second-order check on the two cc-pVDZ systems where the
//! optimizers stop above a lower point
//! (`book/src/17-multireference/05-casscf.md#casscf-second-order-check`): N2 CAS(10,10) above 2
//! core, where the point the
//! RHF start converges to is a saddle, and benzene CAS(6,6) above 18 core
//! on the dense route, which stops 5.9 mHa above PySCF.
//!
//! Slow (323 s for both, release, on a shared box; the N2 case converges
//! twice and both cases run finite-difference checks): deferred to the
//! nightly tier (`scripts/ci-shard-plan.sh`).
//!
//! # Bars
//!
//! The eigenpair `(λ, v)` of the check is compared with the central
//! difference of the CI-relaxed analytic gradient along `v`,
//! `D(s) = (g(C e^{sV}) − g(C e^{−sV})) / 2s` at `s = h, h/2`, `h = 1e-3`:
//! `max_k |D(h/2) − λ v|_k <= (2/3) max_k |D(h) − D(h/2)|_k + 2 δg / h + r + ‖g‖`
//! (truncation, the gradient error `δg = 8 r_CI + 4 ε n_mo² max|F|` over
//! `h/2`, the Davidson residual `r`, the frame term `½ |[G, V]| <= ‖g‖`).
//! A pair that passes is an eigenpair of the finite-difference Hessian to
//! that bar, whatever its sign.
//!
//! The followed N2 point is compared with the minimum in
//! `book/src/17-multireference/05-casscf.md#casscf-proven-stationary`
//! (`−109.1347620089` Ha, BFGS to `‖g‖ < 1e-9`, `‖g_ref‖ = 6.2e-10`):
//! `|ΔE| <= (‖g‖² + ‖g_ref‖²) / (2 λ_min) + 8 ε |E| √N_det`, `λ_min` the
//! check's eigenvalue at the followed point.
//!
//! Physics vs artifact: a saddle shows as a negative `λ` whose mode passes
//! the finite-difference bar and lowers the energy when followed; a broken
//! CI response gives a `λ` that fails the bar (the fixed-RDM Hessian is
//! stiffer), and a broken step gives no energy drop.

mod casscf_common;

use casscf_common::{maxabs, molecule_text, setup_text, EPS};
use ferric_ci::casscf::{optimize, CasScfConfig, CasScfProblem, CasScfResult};
use ferric_ci::CasScfStabilityMode;
use ferric_scf::stability::StabilityVerdict;

/// The N2 CAS(10,10)/cc-pVDZ local minimum below the RHF-start saddle.
const N2_MINIMUM: f64 = -109.1347620089;
/// `‖g‖` of that reference point.
const N2_MINIMUM_GRAD: f64 = 6.2e-10;

fn fd(problem: &CasScfProblem, c: &ndarray::Array2<f64>, x: &[f64], h: f64) -> Vec<f64> {
    let kp: Vec<f64> = x.iter().map(|v| h * v).collect();
    let km: Vec<f64> = x.iter().map(|v| -h * v).collect();
    let gp = problem
        .evaluate(&problem.rotate(c, &kp).unwrap())
        .unwrap()
        .gradient;
    let gm = problem
        .evaluate(&problem.rotate(c, &km).unwrap())
        .unwrap()
        .gradient;
    gp.iter()
        .zip(&gm)
        .map(|(a, b)| (a - b) / (2.0 * h))
        .collect()
}

/// The eigenpair of `r`'s check against the finite difference along it.
fn eigenpair_matches_finite_differences(
    name: &str,
    problem: &CasScfProblem,
    r: &CasScfResult,
    cfg: &CasScfConfig,
) {
    let st = r.stability.as_ref().expect("check ran");
    let c = &r.mo_coeff;
    let pt = problem.evaluate(c).unwrap();
    let v = &st.eigenvector;
    let h = 1e-3;
    let d1 = fd(problem, c, v, h);
    let d2 = fd(problem, c, v, h / 2.0);
    let nmo = c.ncols() as f64;
    let dg = 8.0 * cfg.ci_conv_thresh + 4.0 * EPS * nmo * nmo * maxabs(&pt.f_general);
    let trunc = d1
        .iter()
        .zip(&d2)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    let bar = 2.0 * trunc / 3.0 + 2.0 * dg / h + st.residual + r.grad_norm;
    let err = d2.iter().zip(v).fold(0.0f64, |m, (a, b)| {
        m.max((a - st.lowest_eigenvalue * b).abs())
    });
    let rayleigh: f64 = d2.iter().zip(v).map(|(a, b)| a * b).sum();
    let fixed = problem.stability_with(c, &pt, false).unwrap();
    eprintln!(
        "{name}: E {:.10} |g| {:.1e} iterations {}; {}\n  FD along v: Rayleigh {rayleigh:.8}, \
         max|D v - lambda v| {err:.2e} (bar {bar:.2e}); fixed-RDM lambda_min {:.6} ({} \
         products)",
        r.e_total,
        r.grad_norm,
        r.iterations,
        st.summary(),
        fixed.lowest_eigenvalue,
        fixed.products
    );
    assert!(err <= bar, "{name}: |D v - lambda v| {err:e} > bar {bar:e}");
}

fn run(xyz: &str, cfg: &CasScfConfig) -> (CasScfResult, impl Fn(&str, &CasScfResult)) {
    let (mol, prep, rhf) = setup_text(xyz, "cc-pvdz");
    let c0 = rhf.mos_r().clone();
    let problem = CasScfProblem::new(&mol, &prep, c0.ncols(), cfg).unwrap();
    let r = optimize(&problem, c0, cfg).unwrap();
    assert!(r.converged, "not converged in {} iterations", r.iterations);
    let cfg = *cfg;
    let check = move |name: &str, r: &CasScfResult| {
        let problem = CasScfProblem::new(&mol, &prep, r.mo_coeff.ncols(), &cfg).unwrap();
        eigenpair_matches_finite_differences(name, &problem, r, &cfg);
    };
    (r, check)
}

fn n2_cfg(stability: CasScfStabilityMode) -> CasScfConfig {
    CasScfConfig {
        n_core: 2,
        n_active: 10,
        n_elec_active: (5, 5),
        stability,
        ..Default::default()
    }
}

#[test]
fn n2_cas1010_ccpvdz_start_is_a_saddle_and_follow_reaches_the_minimum() {
    let xyz = molecule_text("n2.xyz");
    let (saddle, check) = run(&xyz, &n2_cfg(CasScfStabilityMode::Check));
    let st = saddle.stability.as_ref().unwrap();
    assert_eq!(st.verdict(), StabilityVerdict::Unstable, "{}", st.summary());
    check("N2 CAS(10,10) saddle", &saddle);

    let (min, check) = run(&xyz, &n2_cfg(CasScfStabilityMode::Follow));
    let st = min.stability.as_ref().unwrap();
    assert_eq!(st.verdict(), StabilityVerdict::Stable, "{}", st.summary());
    assert!(st.follow_steps >= 1);
    assert_eq!(
        st.energy_before_follow.map(f64::to_bits),
        Some(saddle.e_total.to_bits())
    );
    check("N2 CAS(10,10) followed", &min);
    let bar = (min.grad_norm.powi(2) + N2_MINIMUM_GRAD.powi(2)) / (2.0 * st.lowest_eigenvalue)
        + 8.0 * EPS * min.e_total.abs() * (min.n_determinants as f64).sqrt();
    let de = min.e_total - N2_MINIMUM;
    eprintln!(
        "N2 followed: E {:.10} vs {N2_MINIMUM}: dE {de:.1e} (bar {bar:.1e}); saddle - minimum \
         {:.4e} Ha; {} follow steps, {} iterations in all",
        min.e_total,
        saddle.e_total - min.e_total,
        st.follow_steps,
        min.iterations
    );
    assert!(
        de.abs() <= bar,
        "N2 followed point: |dE| {:e} > {bar:e}",
        de.abs()
    );
}

#[test]
fn benzene_cas66_ccpvdz_dense_endpoint_is_a_local_minimum() {
    let xyz = molecule_text("benzene.xyz");
    let cfg = CasScfConfig {
        n_core: 18,
        n_active: 6,
        n_elec_active: (3, 3),
        stability: CasScfStabilityMode::Follow,
        ..Default::default()
    };
    let (r, check) = run(&xyz, &cfg);
    let st = r.stability.as_ref().unwrap();
    assert_eq!(st.verdict(), StabilityVerdict::Stable, "{}", st.summary());
    assert_eq!(st.follow_steps, 0);
    check("benzene CAS(6,6) dense", &r);
}
