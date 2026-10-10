//! Analytic RHF/UHF Hessian in a uniform external electric field (issue #286).
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test field_hessian
//! ```
//!
//! The field-density GRADIENT term is itself a finite difference of the dipole
//! integrals, so FD-of-gradient is not an independent check of the integral
//! derivatives. The independent anchor is `analytic_matches_energy_second_differences`:
//! fifth-order stencils on the converged SCF ENERGY (no gradient involved).
//!
//! # MUTATION (each must turn a named test red; see `src/hessian_field.rs`)
//!
//! * skip `field_skeleton` in `rhf_hessian_parts` → both FD tests;
//! * skip `add_field_first_order` (∂(E·r)/∂x missing from F^x) → both FD tests;
//! * negate the field in `field_first_derivative_matrices` → both FD tests;
//! * drop the `c, a` transposed scatter in `field_first_derivative_matrices`
//!   → both FD tests.

use ferric_core::basis;
use ferric_core::external_potential::ExternalPotential;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::{rhf_gradient, uhf_gradient};
use ferric_scf::hessian::{rhf_hessian, rhf_hessian_preflight, uhf_hessian};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, solve_uhf_with_guess};
use ndarray::Array2;

/// Analytic vs FD of the analytic gradient (FD floor ~5e-7 at this step).
const TOL_FD: f64 = 5e-6;
/// Analytic vs fifth-order second differences of the SCF energy.
const TOL_ENERGY: f64 = 1e-6;
const H_GRAD: f64 = 1e-3;
const H_ENERGY: f64 = 0.02;

const WATER: &str = "3\nH2O distorted off C2v\n\
O      0.010000    -0.020000     0.120000\n\
H      0.030000     0.770000    -0.460000\n\
H     -0.020000    -0.740000    -0.490000\n";

fn has_deriv2() -> bool {
    ferric_integrals::engine::libint_max_deriv_order() >= 2
}

fn field() -> ExternalPotential {
    ExternalPotential {
        field: Some([0.02, -0.03, 0.025]),
        ..Default::default()
    }
}

fn config(ext: Option<&ExternalPotential>) -> RhfConfig {
    RhfConfig {
        max_iter: 200,
        energy_conv: 1e-13,
        density_conv: 1e-11,
        df_j_aux: Some(String::new()),
        df_k_aux: Some(String::new()),
        external_potential: ext.cloned(),
        ..Default::default()
    }
}

struct Solved {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    rhf: ferric_scf::result::ScfResult,
}

fn solve(mol: &Molecule, ext: Option<&ExternalPotential>) -> Solved {
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let rhf = solve_rhf(
        &ParallelContext::default(),
        mol,
        &prep,
        op,
        &bounds,
        &config(ext),
    )
    .unwrap();
    assert!(rhf.converged);
    Solved {
        mol: mol.clone(),
        prep,
        bounds,
        rhf,
    }
}

fn analytic(s: &Solved, ext: Option<&ExternalPotential>) -> Array2<f64> {
    rhf_hessian(
        &ParallelContext::default(),
        &s.mol,
        &s.prep,
        Operator::coulomb(),
        &s.bounds,
        &s.rhf,
        &config(ext),
    )
    .unwrap()
}

fn displaced(mol: &Molecule, coord: usize, h: f64) -> Molecule {
    let mut m = mol.clone();
    let a = &mut m.atoms[coord / 3];
    match coord % 3 {
        0 => a.x += h,
        1 => a.y += h,
        _ => a.zpos += h,
    }
    m
}

fn max_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter().zip(b).fold(0.0f64, |m, (x, y)| {
        assert!(x.is_finite() && y.is_finite(), "non-finite Hessian entry");
        m.max((x - y).abs())
    })
}

fn fd_of_gradient(mol: &Molecule, ext: &ExternalPotential) -> Array2<f64> {
    let n3 = 3 * mol.atoms.len();
    let mut out = Array2::<f64>::zeros((n3, n3));
    for b in 0..n3 {
        let g = |h: f64| {
            let s = solve(&displaced(mol, b, h), Some(ext));
            rhf_gradient(
                &s.mol,
                &s.prep,
                Operator::coulomb(),
                &s.bounds,
                &s.rhf,
                Some(ext),
            )
            .unwrap()
        };
        let (gp, gm) = (g(H_GRAD), g(-H_GRAD));
        for a in 0..n3 {
            out[(a, b)] = (gp[(a / 3, a % 3)] - gm[(a / 3, a % 3)]) / (2.0 * H_GRAD);
        }
    }
    out
}

#[test]
fn field_hessian_matches_fd_of_field_gradient() {
    if !has_deriv2() {
        return;
    }
    let ext = field();
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let an = analytic(&solve(&mol, Some(&ext)), Some(&ext));
    let fd = fd_of_gradient(&mol, &ext);
    let d = max_diff(&an, &fd);
    eprintln!("field RHF: max|analytic - FD(gradient)| = {d:.3e}");
    assert!(d < TOL_FD, "max|analytic - FD| = {d:.3e}");

    // Negative control: the zero-field Hessian must miss by far more than the bar.
    let gas = analytic(&solve(&mol, None), None);
    let miss = max_diff(&gas, &fd);
    eprintln!("zero-field vs field FD: {miss:.3e}");
    assert!(miss > 10.0 * TOL_FD, "field too weak: {miss:.3e}");
}

/// Field and point charges together (both terms in one potential).
#[test]
fn field_plus_point_charges_matches_fd_of_gradient() {
    if !has_deriv2() {
        return;
    }
    let mut ext = field();
    ext.point_charges = vec![ferric_core::external_potential::PointCharge {
        q: -0.6,
        x: 3.0,
        y: 0.5,
        z: 1.5,
    }];
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let an = analytic(&solve(&mol, Some(&ext)), Some(&ext));
    let d = max_diff(&an, &fd_of_gradient(&mol, &ext));
    eprintln!("field + charge: max|analytic - FD| = {d:.3e}");
    assert!(d < TOL_FD, "max|analytic - FD| = {d:.3e}");
}

/// Hessian element (i, j) by 5-point stencils on the SCF energy only.
fn energy_second(mol: &Molecule, ext: &ExternalPotential, i: usize, j: usize) -> f64 {
    let e = |di: f64, dj: f64| {
        let m = displaced(&displaced(mol, i, di), j, dj);
        solve(&m, Some(ext)).rhf.energy
    };
    let h = H_ENERGY;
    if i == j {
        let w = [
            (-2.0, -1.0),
            (-1.0, 16.0),
            (0.0, -30.0),
            (1.0, 16.0),
            (2.0, -1.0),
        ];
        return w.iter().map(|&(o, c)| c * e(o * h, 0.0)).sum::<f64>() / (12.0 * h * h);
    }
    let w = [(-2.0, 1.0), (-1.0, -8.0), (1.0, 8.0), (2.0, -1.0)];
    let mut acc = 0.0;
    for &(oi, ci) in &w {
        for &(oj, cj) in &w {
            acc += ci * cj * e(oi * h, oj * h);
        }
    }
    acc / (144.0 * h * h)
}

/// Independent anchor: no gradient (and therefore no gradient-side finite
/// difference of the dipole integrals) is involved.
#[test]
fn analytic_matches_energy_second_differences() {
    if !has_deriv2() {
        return;
    }
    let ext = field();
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let an = analytic(&solve(&mol, Some(&ext)), Some(&ext));
    // Diagonal elements and a spread of mixed (same-atom, cross-atom) ones.
    let pairs = [
        (0, 0),
        (4, 4),
        (8, 8),
        (0, 1),
        (2, 5),
        (3, 7),
        (1, 8),
        (5, 6),
    ];
    let mut worst = 0.0f64;
    for (i, j) in pairs {
        let fd = energy_second(&mol, &ext, i, j);
        let d = (an[(i, j)] - fd).abs();
        eprintln!(
            "H[{i},{j}] analytic {:+.8} energy-FD {fd:+.8} diff {d:.2e}",
            an[(i, j)]
        );
        worst = worst.max(d);
    }
    assert!(
        worst < TOL_ENERGY,
        "max |analytic - energy FD| = {worst:.3e}"
    );
}

#[test]
fn zero_and_absent_field_are_bit_identical() {
    if !has_deriv2() {
        return;
    }
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let zero = ExternalPotential {
        field: Some([0.0; 3]),
        ..Default::default()
    };
    let a = analytic(&solve(&mol, None), None);
    let b = analytic(&solve(&mol, Some(&zero)), Some(&zero));
    assert_eq!(max_diff(&a, &b), 0.0);
}

/// A neutral molecule in a uniform field is translation invariant (the dipole
/// of a neutral system does not change), so every row sums to zero.
#[test]
fn neutral_translational_invariance_in_field() {
    if !has_deriv2() {
        return;
    }
    let ext = field();
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let h = analytic(&solve(&mol, Some(&ext)), Some(&ext));
    let mut worst = 0.0f64;
    for r in 0..9 {
        for axis in 0..3 {
            let sum: f64 = (0..3).map(|a| h[(r, 3 * a + axis)]).sum();
            worst = worst.max(sum.abs());
        }
    }
    eprintln!("field translational invariance: max |row sum| = {worst:.3e}");
    assert!(worst < 1e-7, "row sum {worst:.3e}");
}

#[test]
fn field_is_no_longer_refused() {
    let ext = field();
    rhf_hessian_preflight(Operator::coulomb(), &config(Some(&ext))).unwrap();
}

const OH: &str = "2\nOH tilted\nO 0.0 0.0 0.0\nH 0.35 0.2 0.93\n";

#[test]
fn uhf_field_hessian_matches_fd_of_field_gradient() {
    if !has_deriv2() {
        return;
    }
    let ext = field();
    let mol = Molecule::parse_xyz(OH, 0, 2).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let prep_for = |m: &Molecule| {
        let prep = PreparedBasis::new(m, &bs).unwrap();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        (prep, bounds)
    };
    let cfg = config(Some(&ext));
    let (prep, bounds) = prep_for(&mol);
    let r0 = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
    assert!(r0.converged);
    let an = uhf_hessian(&ctx, &mol, &prep, op, &bounds, &r0, &cfg).unwrap();
    let ca = r0.mos_alpha.clone();
    let cb = r0.mos_beta.clone().unwrap();
    let mut fd = Array2::<f64>::zeros((6, 6));
    for b in 0..6 {
        let g = |step: f64| {
            let m = displaced(&mol, b, step);
            let (prep, bounds) = prep_for(&m);
            let r = solve_uhf_with_guess(&ctx, &m, &prep, &bounds, &cfg, Some((&ca, &cb))).unwrap();
            uhf_gradient(&m, &prep, op, &bounds, &r, Some(&ext)).unwrap()
        };
        let (gp, gm) = (g(H_GRAD), g(-H_GRAD));
        for a in 0..6 {
            fd[(a, b)] = (gp[(a / 3, a % 3)] - gm[(a / 3, a % 3)]) / (2.0 * H_GRAD);
        }
    }
    let d = max_diff(&an, &fd);
    eprintln!("field UHF: max|analytic - FD| = {d:.3e}");
    assert!(d < TOL_FD, "max|analytic - FD| = {d:.3e}");
}
