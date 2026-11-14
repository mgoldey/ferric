//! The CI-relaxed orbital Hessian of `ferric_ci::casscf_stability` against
//! finite differences of the CI-relaxed analytic gradient.
//!
//! Systems (6-31G, ferric exact-J/K RHF start, the partitions of
//! `tests/casscf.rs`): H2O CAS(4,4) above 3 core, N2 CAS(6,6) above 4,
//! O2 CAS(8,6) `(5, 3)` above 4, and H2O CAS(4,4) on the RI route
//! (def2-universal-jkfit). Each is converged with the default settings
//! (`grad_tol` 1e-7), so the frame derivative of the gradient differs from
//! the symmetric Hessian by `½ [G, X]`, `O(‖g‖)`.
//!
//! # Exactness anchor
//!
//! H2O CAS(2,1) above 4 core: one determinant, so the CI complement is empty
//! and `H_rel x` must equal the fixed-RDM product `H_oo x` of
//! `CasScfProblem::hessian_vector` exactly (the CI response adds exact
//! zeros; compared with `==`, so `-0.0` and `+0.0` match).
//!
//! # Products against finite differences
//!
//! Random unit `x`. `D(s) = (g(C e^{sX}) − g(C e^{−sX})) / 2s`, the CASSCF
//! gradient re-solving the CI at each displaced orbital set, at `s = h` and
//! `h/2`, `h = 1e-3`. The error of `D(h/2)` is its truncation, `|D(h) −
//! D(h/2)| / 3` to leading order (twice that is the bar term), plus the
//! gradient error `δg = 8 r + 4 ε n_mo² max|F|` (`r` the CI threshold, the
//! bar of `tests/casscf.rs` anchor 1) divided by `h/2`, plus the frame term
//! `½ |[G, X]|` bounded by `‖g‖`:
//!
//! ```text
//!   max_k |H_rel x − D(h/2)|_k  <=  (2/3) max_k |D(h) − D(h/2)|_k + 2 δg / h + ‖g‖.
//! ```
//!
//! Symmetry: `|y · H_rel x − x · H_rel y| <= 1e3 ε n_pair (max|H_rel x| +
//! max|H_rel y|) + (r_x + r_y) max(1, max|H_oc z|)`: rounding of the two dot
//! products and of the products themselves, plus the CI-response solve's
//! relative residuals `r_x`, `r_y` times the size of the CI-coupling part.
//!
//! Physics vs artifact: a wrong factor or sign in the CI response shows as
//! a product error of the size of `H_oc z` (1e-2 to 1 here), far above the
//! bar; dropping it entirely makes the product equal the fixed-RDM one,
//! which the same comparison rejects (printed: the size of `H_oc z`).
//!
//! # Measured (2026-11-14)
//!
//! Tables in `crates/ferric-ci/theory.md` ("Second-order check").

mod casscf_common;

use casscf_common::{maxabs, molecule_text, setup_text, EPS};
use ferric_ci::casscf::{
    optimize, CasScfConfig, CasScfIntegrals, CasScfProblem, CasScfResult, CasScfStabilityMode,
};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_scf::stability::StabilityVerdict;
use ferric_scf::ScfResult;

fn unit_random(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut v: Vec<f64> = (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        })
        .collect();
    let nv = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.iter_mut().for_each(|x| *x /= nv);
    v
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

struct Sys {
    mol: Molecule,
    prep: PreparedBasis,
    rhf: ScfResult,
    cfg: CasScfConfig,
}

fn system(file: &str, n_core: usize, n_active: usize, nel: (usize, usize)) -> Sys {
    let (mol, prep, rhf) = setup_text(&molecule_text(file), "6-31g");
    let cfg = CasScfConfig {
        n_core,
        n_active,
        n_elec_active: nel,
        stability: CasScfStabilityMode::Check,
        ..Default::default()
    };
    Sys {
        mol,
        prep,
        rhf,
        cfg,
    }
}

fn converge<'a>(s: &'a Sys, cfg: &CasScfConfig) -> (CasScfProblem<'a>, CasScfResult) {
    let c0 = s.rhf.mos_r().clone();
    let problem = CasScfProblem::new(&s.mol, &s.prep, c0.ncols(), cfg).unwrap();
    let r = optimize(&problem, c0, cfg).unwrap();
    assert!(r.converged, "not converged in {} iterations", r.iterations);
    (problem, r)
}

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

fn products_match_finite_differences(name: &str, s: &Sys, cfg: &CasScfConfig) {
    let (problem, r) = converge(s, cfg);
    let c = &r.mo_coeff;
    let pt = problem.evaluate(c).unwrap();
    let hr = problem.ci_relaxed_hessian(c, &pt).unwrap();
    let n = problem.pairs().len();
    let x = unit_random(n, 0x9E37_79B9_7F4A_7C15);
    let y = unit_random(n, 0xD1B5_4A32_D192_ED03);
    let (hx, rel_x) = problem
        .ci_relaxed_hessian_vector_with(&hr, &x, true)
        .unwrap();
    let (hx_fixed, _) = problem
        .ci_relaxed_hessian_vector_with(&hr, &x, false)
        .unwrap();
    let (hy, rel_y) = problem
        .ci_relaxed_hessian_vector_with(&hr, &y, true)
        .unwrap();
    let h = 1e-3;
    let d1 = fd(&problem, c, &x, h);
    let d2 = fd(&problem, c, &x, h / 2.0);
    let nmo = c.ncols() as f64;
    let dg = 8.0 * cfg.ci_conv_thresh + 4.0 * EPS * nmo * nmo * maxabs(&pt.f_general);
    let trunc = d1
        .iter()
        .zip(&d2)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    let bar = 2.0 * trunc / 3.0 + 2.0 * dg / h + r.grad_norm;
    let err = hx
        .iter()
        .zip(&d2)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    let ci_part = hx
        .iter()
        .zip(&hx_fixed)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    let asym = (dot(&y, &hx) - dot(&x, &hy)).abs();
    let sym_bar =
        1e3 * EPS * n as f64 * (maxabs(&hx) + maxabs(&hy)) + (rel_x + rel_y) * ci_part.max(1.0);
    eprintln!(
        "{name}: max|H_rel x - FD| {err:.2e} (bar {bar:.2e}; FD(h) - FD(h/2) {trunc:.1e}); \
         max|H_oc z| {ci_part:.3e}; asymmetry {asym:.1e} (bar {sym_bar:.1e}); CI response \
         residuals {rel_x:.1e}, {rel_y:.1e}; |g| {:.1e}",
        r.grad_norm
    );
    assert!(err <= bar, "{name}: product error {err:e} > bar {bar:e}");
    assert!(
        ci_part > 10.0 * bar,
        "{name}: the CI response {ci_part:e} is not resolved above the bar {bar:e}"
    );
    assert!(asym <= sym_bar, "{name}: asymmetry {asym:e} > {sym_bar:e}");
}

#[test]
fn h2o_cas44_products_match_finite_differences() {
    let s = system("water.xyz", 3, 4, (2, 2));
    products_match_finite_differences("H2O CAS(4,4)", &s, &s.cfg);
}

#[test]
fn n2_cas66_products_match_finite_differences() {
    let s = system("n2.xyz", 4, 6, (3, 3));
    products_match_finite_differences("N2 CAS(6,6)", &s, &s.cfg);
}

#[test]
fn o2_triplet_cas86_products_match_finite_differences() {
    let s = system("o2.xyz", 4, 6, (5, 3));
    products_match_finite_differences("O2 CAS(8,6) (5,3)", &s, &s.cfg);
}

#[test]
fn h2o_cas44_ri_products_match_finite_differences() {
    let s = system("water.xyz", 3, 4, (2, 2));
    let cfg = CasScfConfig {
        integrals: CasScfIntegrals::Ri {
            aux: "def2-universal-jkfit",
        },
        ..s.cfg
    };
    products_match_finite_differences("H2O CAS(4,4) RI", &s, &cfg);
}

#[test]
fn a_single_determinant_has_no_ci_response() {
    let s = system("water.xyz", 4, 1, (1, 1));
    let (problem, r) = converge(&s, &s.cfg);
    let pt = problem.evaluate(&r.mo_coeff).unwrap();
    let hr = problem.ci_relaxed_hessian(&r.mo_coeff, &pt).unwrap();
    let oh = problem.orbital_hessian(&r.mo_coeff, &pt).unwrap();
    let x = unit_random(problem.pairs().len(), 7);
    let (hx, rel) = problem
        .ci_relaxed_hessian_vector_with(&hr, &x, true)
        .unwrap();
    let hoo = problem.hessian_vector(&oh, &x).unwrap();
    assert_eq!(rel, 0.0);
    for (k, (a, b)) in hx.iter().zip(&hoo).enumerate() {
        // `==`, not bits: the CI part adds `+0.0`, which turns a `-0.0` of
        // `H_oo x` into `+0.0`.
        assert!(a == b, "element {k}: {a} vs {b}");
    }
    // The one active orbital is doubly occupied, so the core-active rotations
    // are redundant: an exact zero mode, which the check must call MARGINAL.
    let st = r.stability.as_ref().expect("check ran");
    assert_eq!(st.verdict(), StabilityVerdict::Marginal, "{}", st.summary());
}

#[test]
fn check_and_follow_leave_a_minimum_unchanged_and_off_skips() {
    let s = system("n2.xyz", 4, 6, (3, 3));
    let (_, check) = converge(&s, &s.cfg);
    let follow_cfg = CasScfConfig {
        stability: CasScfStabilityMode::Follow,
        ..s.cfg
    };
    let (_, follow) = converge(&s, &follow_cfg);
    let off_cfg = CasScfConfig {
        stability: CasScfStabilityMode::Off,
        ..s.cfg
    };
    let (_, off) = converge(&s, &off_cfg);
    assert!(off.stability.is_none());
    let (a, b) = (
        check.stability.as_ref().unwrap(),
        follow.stability.as_ref().unwrap(),
    );
    assert_eq!(a.verdict(), StabilityVerdict::Stable);
    assert_eq!(b.follow_steps, 0);
    assert!(b.energy_before_follow.is_none());
    assert_eq!(a.lowest_eigenvalue.to_bits(), b.lowest_eigenvalue.to_bits());
    for r in [&follow, &off] {
        assert_eq!(r.e_total.to_bits(), check.e_total.to_bits());
        assert_eq!(r.iterations, check.iterations);
    }
}

#[test]
fn stability_mode_parse_is_strict() {
    for m in [
        CasScfStabilityMode::Off,
        CasScfStabilityMode::Check,
        CasScfStabilityMode::Follow,
    ] {
        assert_eq!(CasScfStabilityMode::parse_config_str(m.as_str()), Ok(m));
    }
    for bad in ["Check", "on", "", "true"] {
        let e = CasScfStabilityMode::parse_config_str(bad).unwrap_err();
        assert!(e.starts_with("stability must be"), "{e}");
    }
}
