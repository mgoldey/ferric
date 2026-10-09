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
//! * pass `&[]` instead of `extra` to `contract_external_deriv2` →
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
use ferric_scf::gradient::{rhf_gradient, uhf_gradient};
use ferric_scf::hessian::{rhf_hessian, rhf_hessian_preflight, uhf_hessian};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, solve_uhf_with_guess};
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
    a.iter().zip(b).fold(0.0f64, |m, (x, y)| {
        // f64::max would silently ignore a NaN entry.
        assert!(x.is_finite() && y.is_finite(), "non-finite Hessian entry");
        m.max((x - y).abs())
    })
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

/// Translational invariance including the charges: shifting every QM atom AND
/// every charge leaves the energy unchanged, so for each QM coordinate row
/// the QM–QM row sum plus the QM–MM cross block (central FD of the embedded
/// QM gradient w.r.t. the charge positions, SCF re-solved) vanishes.
#[test]
fn translational_invariance_including_charge_cross_block() {
    if !has_deriv2() {
        return;
    }
    let ext = embedding();
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let h = analytic(&solve(&mol, Some(&ext)), Some(&ext));
    let mut worst = 0.0f64;
    for axis in 0..3 {
        // Row sums of the QM-QM block over the three atoms' `axis` columns.
        let mut sums = [0.0f64; 9];
        for (r, sum) in sums.iter_mut().enumerate() {
            for a in 0..3 {
                *sum += h[(r, 3 * a + axis)];
            }
        }
        // Cross block: d g_r / d (charge_i, axis), summed over charges.
        for i in 0..ext.point_charges.len() {
            let g = |step: f64| {
                let mut e = ext.clone();
                match axis {
                    0 => e.point_charges[i].x += step,
                    1 => e.point_charges[i].y += step,
                    _ => e.point_charges[i].z += step,
                }
                let s = solve(&mol, Some(&e));
                rhf_gradient(
                    &s.mol,
                    &s.prep,
                    Operator::coulomb(),
                    &s.bounds,
                    &s.rhf,
                    Some(&e),
                )
                .unwrap()
            };
            let (gp, gm) = (g(H_FULL), g(-H_FULL));
            for (r, sum) in sums.iter_mut().enumerate() {
                *sum += (gp[(r / 3, r % 3)] - gm[(r / 3, r % 3)]) / (2.0 * H_FULL);
            }
        }
        worst = sums.iter().fold(worst, |m, v| m.max(v.abs()));
    }
    eprintln!("charge-inclusive translational invariance: max |row sum| = {worst:.3e}");
    assert!(worst < TOL_FD, "row sum {worst:.3e}");
}

const OH: &str = "2\nOH tilted\nO 0.0 0.0 0.0\nH 0.35 0.2 0.93\n";

/// UHF (OH radical) with the same style of embedding vs FD of `uhf_gradient`.
#[test]
fn embedded_uhf_hessian_matches_fd_of_embedded_gradient() {
    if !has_deriv2() {
        return;
    }
    let ext = embedding();
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
    let n3 = 6;
    let mut fd = Array2::<f64>::zeros((n3, n3));
    for b in 0..n3 {
        let g = |step: f64| {
            let m = displaced(&mol, b / 3, b % 3, step);
            let (prep, bounds) = prep_for(&m);
            let r = solve_uhf_with_guess(&ctx, &m, &prep, &bounds, &cfg, Some((&ca, &cb))).unwrap();
            assert!((r.energy - r0.energy).abs() < 0.05, "state changed");
            uhf_gradient(&m, &prep, op, &bounds, &r, Some(&ext)).unwrap()
        };
        let (gp, gm) = (g(H_FULL), g(-H_FULL));
        for a in 0..n3 {
            fd[(a, b)] = (gp[(a / 3, a % 3)] - gm[(a / 3, a % 3)]) / (2.0 * H_FULL);
        }
    }
    let d = max_diff(&an, &fd);
    eprintln!("embedded UHF: max|analytic - FD| = {d:.3e}");
    assert!(d < TOL_FD, "max|analytic - FD| = {d:.3e}");
}

/// More charges than one libint2 chunk (`EXTERNAL_CHUNK` = 4), so the
/// chunked external-charge contraction is exercised across chunk boundaries.
#[test]
fn many_charges_span_several_chunks() {
    if !has_deriv2() {
        return;
    }
    let mut ext = embedding();
    for k in 0..6 {
        let t = k as f64;
        ext.point_charges.push(PointCharge {
            q: if k % 2 == 0 { 0.3 } else { -0.3 },
            x: 4.0 * (1.3 * t).cos(),
            y: 4.0 * (0.7 * t).sin(),
            z: -3.5 + 1.1 * t,
        });
    }
    assert!(ext.point_charges.len() > 2 * 4);
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let s = solve(&mol, Some(&ext));
    let an = analytic(&s, Some(&ext));
    let fd = fd_hessian(&mol, Some(&ext));
    let d = max_diff(&an, &fd);
    eprintln!("9 charges: max|analytic - FD| = {d:.3e}");
    assert!(d < TOL_FD, "max|analytic - FD| = {d:.3e}");
}
