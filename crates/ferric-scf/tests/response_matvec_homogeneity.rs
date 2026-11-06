//! The orbital-Hessian matvecs are linear in the rotation κ, so
//! `H(s·κ) = s·H(κ)`. Their J/K builds screen quartets on the ABSOLUTE
//! `bound · max|δD| < thresh`, which breaks that for small κ (Newton/PCG
//! search directions near convergence) unless the operand is first scaled to
//! `max|δD| ∈ (½, 1]` (`rhf::build_jk_with_pool_homogeneous`).
//!
//! Each test asserts bit-exact homogeneity at `s = 2⁻³⁰` (power-of-two scaling
//! is exact in floating point, so any miss is the screen) and prints the
//! relative miss `max|H(s·κ)/s − H(κ)| / max|H(κ)|` over `s = 1e-2 .. 1e-9`.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::oneelectron;
use ferric_integrals::operator::Operator;
use ferric_scf::engine_pool::EnginePool;
use ferric_scf::rhf::{build_jk, solve_rhf, RhfConfig};
use ferric_scf::rhf_newton::RhfNewtonInputs;
use ferric_scf::rohf::{solve_rohf, RohfConfig};
use ferric_scf::rohf_newton::RohfNewtonInputs;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf_newton::UhfNewtonInputs;
use ndarray::Array2;

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// A smooth deterministic rotation block (rows × cols), max ~0.1.
fn kappa(rows: usize, cols: usize, phase: f64) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(a, i)| {
        0.1 * (phase + 0.7 * a as f64 + 1.3 * i as f64).sin() / (1.0 + 0.05 * a as f64)
    })
}

/// `max|H(s·κ)/s − H(κ)| / max|H(κ)|` for each `s`.
fn misses<F>(h: F, k: &[Array2<f64>], scales: &[f64]) -> Vec<f64>
where
    F: Fn(&[Array2<f64>]) -> Vec<Array2<f64>>,
{
    let full = h(k);
    let hmax = full.iter().map(max_abs).fold(0.0_f64, f64::max);
    scales
        .iter()
        .map(|&s| {
            let ks: Vec<Array2<f64>> = k.iter().map(|m| s * m).collect();
            let small = h(&ks);
            small
                .iter()
                .zip(&full)
                .map(|(a, b)| max_abs(&(a / s - b)))
                .fold(0.0_f64, f64::max)
                / hmax
        })
        .collect()
}

const DECADES: [f64; 8] = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9];

/// Prints the decade sweep and asserts bit-exactness at 2⁻³⁰ and < 1e-12
/// relative at every decade. Decimal scaling rounds the operand once, which
/// alone leaves ~1e-13 (measured: identical at s = 1e-2 with and without the
/// operand scaling); the absolute screen on the raw operand misses by
/// 4.5e-6 .. 5.4e-5 already at s = 1e-7.
fn check<F>(what: &str, h: F, k: &[Array2<f64>])
where
    F: Fn(&[Array2<f64>]) -> Vec<Array2<f64>>,
{
    let sweep = misses(&h, k, &DECADES);
    for (s, m) in DECADES.iter().zip(&sweep) {
        eprintln!("{what}: s = {s:.0e}  rel miss = {m:.3e}");
    }
    let exact = misses(&h, k, &[2f64.powi(-30)])[0];
    eprintln!("{what}: s = 2^-30  rel miss = {exact:.3e}");
    assert!(
        exact == 0.0,
        "{what}: H(2^-30 κ)/2^-30 − H(κ) rel {exact:.3e}"
    );
    let worst = sweep.iter().fold(0.0_f64, |m, v| m.max(*v));
    assert!(worst < 1e-12, "{what}: worst decade rel miss {worst:.3e}");
}

struct Closed {
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    c: Array2<f64>,
    f_mo: Array2<f64>,
    nocc: usize,
}

/// CH4/aug-cc-pVDZ RHF (diffuse functions: many small-bound quartets).
fn ch4_aug() -> Closed {
    let xyz = "5\nch4\nC 0 0 0\nH 0.6276 0.6276 0.6276\nH -0.6276 -0.6276 0.6276\n\
               H -0.6276 0.6276 -0.6276\nH 0.6276 -0.6276 -0.6276\n";
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("aug-cc-pvdz").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        ..Default::default()
    };
    let res = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg).unwrap();
    let c = res.mos_alpha.clone();
    let f_mo = c.t().dot(&res.fock_alpha).dot(&c);
    Closed {
        prep,
        bounds,
        c,
        f_mo,
        nocc: 5,
    }
}

fn rhf_inputs(s: &Closed) -> RhfNewtonInputs<'_> {
    RhfNewtonInputs {
        prep: &s.prep,
        bounds: &s.bounds,
        c: &s.c,
        f_mo: &s.f_mo,
        nocc: s.nocc,
        k_mix_sr: 1.0,
        rsh: None,
        fxc: None,
        thresh: RhfConfig::default().integral_thresh,
        ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
    }
}

#[test]
fn rhf_singlet_and_triplet_matvecs_are_homogeneous() {
    let s = ch4_aug();
    let ctx = ParallelContext::default();
    let inp = rhf_inputs(&s);
    let pool = EnginePool::new(s.bounds.op, &s.prep, 1e-14).unwrap();
    let nv = s.c.ncols() - s.nocc;
    let k = [kappa(nv, s.nocc, 0.3)];
    check(
        "rhf_newton::hessian_matvec",
        |k| vec![ferric_scf::rhf_newton::hessian_matvec(&ctx, &inp, &k[0], &pool).unwrap()],
        &k,
    );
    check(
        "rhf_newton::triplet_hessian_matvec",
        |k| vec![ferric_scf::rhf_newton::triplet_hessian_matvec(&ctx, &inp, &k[0], &pool).unwrap()],
        &k,
    );
}

#[test]
fn uhf_matvec_is_homogeneous() {
    // The UHF operator at the RHF point (C_α = C_β): the same J/K builds as a
    // genuine open-shell state, on a system already converged above.
    let s = ch4_aug();
    let ctx = ParallelContext::default();
    let inp = UhfNewtonInputs {
        prep: &s.prep,
        bounds: &s.bounds,
        c_a: &s.c,
        c_b: &s.c,
        f_a_mo: &s.f_mo,
        f_b_mo: &s.f_mo,
        nocc_a: s.nocc,
        nocc_b: s.nocc,
        k_mix_sr: 1.0,
        rsh: None,
        fxc: None,
        thresh: RhfConfig::default().integral_thresh,
        ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
    };
    let pool = EnginePool::new(s.bounds.op, &s.prep, 1e-14).unwrap();
    let nv = s.c.ncols() - s.nocc;
    let k = [kappa(nv, s.nocc, 0.3), kappa(nv, s.nocc, 1.9)];
    check(
        "uhf_newton::hessian_matvec",
        |k| {
            let (a, b) =
                ferric_scf::uhf_newton::hessian_matvec(&ctx, &inp, &k[0], &k[1], &pool).unwrap();
            vec![a, b]
        },
        &k,
    );
}

#[test]
fn rohf_matvec_is_homogeneous() {
    let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 1.10\n", 0, 2).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("aug-cc-pvdz").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RohfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        ..Default::default()
    };
    let res = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    let n = prep.nbasis();
    let h = oneelectron::hcore(&prep);
    let d_a = res.density_alpha.clone();
    let d_b = res.density_beta.clone().expect("ROHF stores density_beta");
    let mut j = Array2::<f64>::zeros((n, n));
    let mut scratch = Array2::<f64>::zeros((n, n));
    build_jk(
        &ctx,
        &prep,
        &bounds,
        1e-12,
        &(&d_a + &d_b),
        &mut j,
        &mut scratch,
    )
    .unwrap();
    let mut k_a = Array2::<f64>::zeros((n, n));
    let mut k_b = Array2::<f64>::zeros((n, n));
    scratch.fill(0.0);
    build_jk(&ctx, &prep, &bounds, 1e-12, &d_a, &mut scratch, &mut k_a).unwrap();
    scratch.fill(0.0);
    build_jk(&ctx, &prep, &bounds, 1e-12, &d_b, &mut scratch, &mut k_b).unwrap();
    let c = &res.mos_alpha;
    let f_a_mo = c.t().dot(&(&h + &j - &k_a)).dot(c);
    let f_b_mo = c.t().dot(&(&h + &j - &k_b)).dot(c);
    let (nc, no) = (4, 1);
    let inp = RohfNewtonInputs {
        prep: &prep,
        bounds: &bounds,
        c,
        f_a_mo: &f_a_mo,
        f_b_mo: &f_b_mo,
        nocc_double: nc,
        nocc_open: no,
        k_mix_sr: 1.0,
        rsh: None,
        fxc: None,
        thresh: RhfConfig::default().integral_thresh,
        ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
    };
    let pool = EnginePool::new(bounds.op, &prep, 1e-14).unwrap();
    let nv = c.ncols() - nc - no;
    let k = [kappa(nv, nc, 0.3), kappa(nv, no, 1.1), kappa(no, nc, 2.3)];
    check(
        "rohf_newton::hessian_matvec",
        |k| {
            let (a, b, c) =
                ferric_scf::rohf_newton::hessian_matvec(&ctx, &inp, &k[0], &k[1], &k[2], &pool)
                    .unwrap();
            vec![a, b, c]
        },
        &k,
    );
}
