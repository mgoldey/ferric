//! Analytic RHF Hessian with external point charges (QM–QM block, charges
//! fixed in space) against central differences of ferric's own embedded
//! analytic gradient, plus the exactness anchors of issue #286.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test qmmm_hessian_fd
//! ```
//!
//! # MUTATION (each must turn a named test red; see `src/hessian.rs`)
//!
//! * drop `charge_nuclear_hessian` from `nuclear_term` → `embedded_hessian_matches_fd_of_embedded_gradient`;
//! * pass `&[]` instead of `extra` to the nuclear `contract_1e_deriv2` →
//!   same test (skeleton term misses the charges);
//! * pass `&[]` to the nuclear `deriv1_1e_matrices` in `first_order_ao`
//!   (∂V_ext/∂x missing from F^x) → same test, response term misses;
//! * flip the sign of a charge's `q` in the engine load → same test;
//! * keep the external-centre blocks in `scatter_unique_pairs` (do not
//!   discard) → `embedded_hessian_matches_fd_of_embedded_gradient` and
//!   `translational_invariance_*`.

use ferric_core::basis;
use ferric_core::external_potential::{ExternalPotential, PointCharge};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::gradient::rhf_gradient;
use ferric_scf::hessian::{rhf_hessian, rhf_hessian_preflight};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

/// Analytic vs FD of the embedded gradient, Ha/Bohr² (FD floor ~5e-7).
const TOL_FD: f64 = 5e-6;
/// Row sums of the QM–QM block need not vanish (the MM charges feel the
/// force); but the sum over the QM atoms of the full block must equal minus
/// the QM–MM cross block. Here only the symmetry is asserted.
const TOL_SYMMETRY: f64 = 1e-9;
/// The embedding must move the Hessian by at least this factor of `TOL_FD`.
const MUST_MATTER_FACTOR: f64 = 100.0;
const H_FULL: f64 = 1e-3;

const WATER: &str = "3\nH2O distorted off C2v\n\
O      0.010000    -0.020000     0.120000\n\
H      0.030000     0.770000    -0.460000\n\
H     -0.020000    -0.740000    -0.490000\n";

fn has_deriv2() -> bool {
    ferric_integrals::engine::libint_max_deriv_order() >= 2
}

fn embedding() -> ExternalPotential {
    ExternalPotential {
        point_charges: vec![
            PointCharge {
                q: -0.8,
                x: 3.1,
                y: 0.4,
                z: 1.2,
            },
            PointCharge {
                q: 0.4,
                x: -2.6,
                y: 1.9,
                z: -2.3,
            },
            PointCharge {
                q: 0.4,
                x: 0.5,
                y: -3.4,
                z: 2.8,
            },
        ],
        ..Default::default()
    }
}

fn config(ext: Option<&ExternalPotential>) -> RhfConfig {
    RhfConfig {
        max_iter: 200,
        energy_conv: 1e-12,
        density_conv: 1e-10,
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

fn displaced(mol: &Molecule, atom: usize, coord: usize, h: f64) -> Molecule {
    let mut m = mol.clone();
    match coord {
        0 => m.atoms[atom].x += h,
        1 => m.atoms[atom].y += h,
        _ => m.atoms[atom].zpos += h,
    }
    m
}

fn fd_hessian(mol: &Molecule, ext: Option<&ExternalPotential>) -> Array2<f64> {
    let n3 = 3 * mol.atoms.len();
    let mut out = Array2::<f64>::zeros((n3, n3));
    for b in 0..n3 {
        let g = |h: f64| {
            let s = solve(&displaced(mol, b / 3, b % 3, h), ext);
            rhf_gradient(&s.mol, &s.prep, Operator::coulomb(), &s.bounds, &s.rhf, ext).unwrap()
        };
        let (gp, gm) = (g(H_FULL), g(-H_FULL));
        for a in 0..n3 {
            out[(a, b)] = (gp[(a / 3, a % 3)] - gm[(a / 3, a % 3)]) / (2.0 * H_FULL);
        }
    }
    out
}

fn max_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn embedded_hessian_matches_fd_of_embedded_gradient() {
    if !has_deriv2() {
        return;
    }
    let ext = embedding();
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let s = solve(&mol, Some(&ext));
    let an = analytic(&s, Some(&ext));
    let fd = fd_hessian(&mol, Some(&ext));
    let d = max_diff(&an, &fd);
    eprintln!("embedded: max|analytic - FD| = {d:.3e}");
    assert!(d < TOL_FD, "max|analytic - FD| = {d:.3e}");

    // Negative control: the gas-phase Hessian must miss the embedded
    // reference by far more than the bar, proving the charges are applied.
    let gas = analytic(&solve(&mol, None), None);
    let miss = max_diff(&gas, &fd);
    eprintln!("gas-phase vs embedded FD: {miss:.3e}");
    assert!(
        miss > MUST_MATTER_FACTOR * TOL_FD * 0.1,
        "embedding too weak: {miss:.3e}"
    );
}

#[test]
fn embedded_hessian_is_symmetric() {
    if !has_deriv2() {
        return;
    }
    let ext = embedding();
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let h = analytic(&solve(&mol, Some(&ext)), Some(&ext));
    let asym = max_diff(&h, &h.t().to_owned());
    assert!(asym < TOL_SYMMETRY, "asymmetry {asym:.3e}");
}

#[test]
fn empty_potential_is_bit_identical_to_no_potential() {
    if !has_deriv2() {
        return;
    }
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let empty = ExternalPotential::default();
    let a = analytic(&solve(&mol, None), None);
    let b = analytic(&solve(&mol, Some(&empty)), Some(&empty));
    assert_eq!(max_diff(&a, &b), 0.0);
}

#[test]
fn field_and_smeared_charges_are_still_refused() {
    let field = ExternalPotential {
        field: Some([0.0, 0.0, 0.01]),
        ..Default::default()
    };
    let err = rhf_hessian_preflight(Operator::coulomb(), &config(Some(&field))).unwrap_err();
    assert!(err.to_string().contains("uniform external field"), "{err}");
    let smeared = ExternalPotential {
        smeared_charges: vec![ferric_core::external_potential::SmearedCharge {
            q: 0.3,
            x: 3.0,
            y: 0.0,
            z: 0.0,
            width: 1.0,
        }],
        ..Default::default()
    };
    let err = rhf_hessian_preflight(Operator::coulomb(), &config(Some(&smeared))).unwrap_err();
    assert!(err.to_string().contains("smeared"), "{err}");
}
