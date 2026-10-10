//! Analytic RHF/UHF Hessian with Gaussian-smeared external charges (QM–QM
//! block, sites fixed in space; issue #358).
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --release --test smeared_hessian
//! ```
//!
//! Anchors (each has a negative control with the bar's other side printed):
//!
//! * `embedded_*_matches_fd_of_smeared_gradient`: analytic vs central FD of
//!   ferric's own analytic smeared gradient; the gas-phase Hessian must miss
//!   the FD by >= `MUST_MATTER_FACTOR` x the bar.
//! * `tight_width_limit_is_point_charge_hessian`: width -> 0 reproduces the
//!   #343 point-charge Hessian; a wrong width (1.0 Bohr) misses it.
//! * `translational_invariance_including_site_cross_block`.
//! * empty `smeared_charges` bit-identical to none (below, and
//!   `qmmm_hessian_fd.rs::empty_potential_is_bit_identical_to_no_potential`).
//!
//! # MUTATION (each must turn a named test red; see `src/hessian_smeared.rs`)
//!
//! * drop the smeared term from `charge_nuclear_hessian`;
//! * drop `smeared_skeleton_hessian` from `extra_skeleton`;
//! * drop `smeared_first_derivative_matrices` from `add_extra_first_order`;
//! * displace a single shell instead of the atom in `atom_shift`;
//! * flip the sign of `q` in `Sites::coeff`;
//! * use the wrong width for a site in `Sites::new`.

use ferric_core::basis;
use ferric_core::external_potential::{ExternalPotential, PointCharge, SmearedCharge};
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

/// Analytic vs FD of the analytic gradient, Ha/Bohr² (FD floor ~5e-7).
const TOL_FD: f64 = 5e-6;
/// The smeared embedding must move the Hessian by this factor of `TOL_FD`.
const MUST_MATTER_FACTOR: f64 = 10.0;
/// Tight-width (1e-3 Bohr) vs point-charge Hessian bar.
const TOL_TIGHT: f64 = 1e-6;
const H_FULL: f64 = 1e-3;

const WATER: &str = "3\nH2O distorted off C2v\n\
O      0.010000    -0.020000     0.120000\n\
H      0.030000     0.770000    -0.460000\n\
H     -0.020000    -0.740000    -0.490000\n";
const OH: &str = "2\nOH tilted\nO 0.0 0.0 0.0\nH 0.35 0.2 0.93\n";

fn has_deriv2() -> bool {
    ferric_integrals::engine::libint_max_deriv_order() >= 2
}

fn site(q: f64, x: f64, y: f64, z: f64, width: f64) -> SmearedCharge {
    SmearedCharge { q, x, y, z, width }
}

/// Three sites with DIFFERENT widths (so a mix-up between sites shows).
fn sites() -> Vec<SmearedCharge> {
    vec![
        site(-0.8, 3.1, 0.4, 1.2, 0.7),
        site(0.4, -2.6, 1.9, -2.3, 1.1),
        site(0.4, 0.5, -3.4, 2.8, 1.6),
    ]
}

fn embedding() -> ExternalPotential {
    ExternalPotential {
        smeared_charges: sites(),
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
    let ctx = ParallelContext::default();
    let rhf = solve_rhf(&ctx, mol, &prep, op, &bounds, &config(ext)).unwrap();
    assert!(rhf.converged);
    Solved {
        mol: mol.clone(),
        prep,
        bounds,
        rhf,
    }
}

fn analytic(s: &Solved, ext: Option<&ExternalPotential>) -> Array2<f64> {
    let (ctx, op) = (ParallelContext::default(), Operator::coulomb());
    rhf_hessian(&ctx, &s.mol, &s.prep, op, &s.bounds, &s.rhf, &config(ext)).unwrap()
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

fn rhf_grad(mol: &Molecule, ext: &ExternalPotential) -> Array2<f64> {
    let s = solve(mol, Some(ext));
    let op = Operator::coulomb();
    rhf_gradient(&s.mol, &s.prep, op, &s.bounds, &s.rhf, Some(ext)).unwrap()
}

fn fd_hessian(mol: &Molecule, ext: &ExternalPotential) -> Array2<f64> {
    let n3 = 3 * mol.atoms.len();
    let mut out = Array2::<f64>::zeros((n3, n3));
    for b in 0..n3 {
        let g = |h: f64| rhf_grad(&displaced(mol, b / 3, b % 3, h), ext);
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
fn embedded_rhf_hessian_matches_fd_of_smeared_gradient() {
    if !has_deriv2() {
        return;
    }
    let ext = embedding();
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let an = analytic(&solve(&mol, Some(&ext)), Some(&ext));
    let fd = fd_hessian(&mol, &ext);
    let d = max_diff(&an, &fd);
    eprintln!("smeared RHF: max|analytic - FD| = {d:.3e} (bar {TOL_FD:.0e})");
    assert!(d < TOL_FD, "max|analytic - FD| = {d:.3e}");

    // Negative control: without the smeared environment the Hessian must miss
    // the embedded FD reference by far more than the bar.
    let gas = analytic(&solve(&mol, None), None);
    let miss = max_diff(&gas, &fd);
    eprintln!("smeared RHF: gas-phase Hessian misses the reference by {miss:.3e}");
    assert!(
        miss > MUST_MATTER_FACTOR * TOL_FD,
        "embedding too weak: {miss:.3e}"
    );
}

#[test]
fn embedded_uhf_hessian_matches_fd_of_smeared_gradient() {
    if !has_deriv2() {
        return;
    }
    let ext = embedding();
    let mol = Molecule::parse_xyz(OH, 0, 2).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let (ctx, op) = (ParallelContext::default(), Operator::coulomb());
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
    let (ca, cb) = (r0.mos_alpha.clone(), r0.mos_beta.clone().unwrap());
    let mut fd = Array2::<f64>::zeros((6, 6));
    for b in 0..6 {
        let g = |step: f64| {
            let m = displaced(&mol, b / 3, b % 3, step);
            let (prep, bounds) = prep_for(&m);
            let guess = Some((&ca, &cb));
            let r = solve_uhf_with_guess(&ctx, &m, &prep, &bounds, &cfg, guess).unwrap();
            assert!((r.energy - r0.energy).abs() < 0.05, "state changed");
            uhf_gradient(&m, &prep, op, &bounds, &r, Some(&ext)).unwrap()
        };
        let (gp, gm) = (g(H_FULL), g(-H_FULL));
        for a in 0..6 {
            fd[(a, b)] = (gp[(a / 3, a % 3)] - gm[(a / 3, a % 3)]) / (2.0 * H_FULL);
        }
    }
    let d = max_diff(&an, &fd);
    eprintln!("smeared UHF: max|analytic - FD| = {d:.3e} (bar {TOL_FD:.0e})");
    assert!(d < TOL_FD, "max|analytic - FD| = {d:.3e}");
}

/// Width -> 0 (width 1e-3 Bohr, zeta = 1e6): the smeared Hessian is the #343
/// point-charge Hessian. The wrong-width control (1.0 Bohr) must miss it.
#[test]
fn tight_width_limit_is_point_charge_hessian() {
    if !has_deriv2() {
        return;
    }
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let with_width = |w: f64| {
        let mut e = embedding();
        e.smeared_charges.iter_mut().for_each(|s| s.width = w);
        e
    };
    let point = ExternalPotential {
        point_charges: sites()
            .iter()
            .map(|s| PointCharge {
                q: s.q,
                x: s.x,
                y: s.y,
                z: s.z,
            })
            .collect(),
        ..Default::default()
    };
    let hp = analytic(&solve(&mol, Some(&point)), Some(&point));
    let dist = |w: f64| {
        let e = with_width(w);
        max_diff(&analytic(&solve(&mol, Some(&e)), Some(&e)), &hp)
    };
    let (tight, wrong) = (dist(1e-3), dist(1.0));
    eprintln!("tight-width vs point charges: {tight:.3e}; wrong width (1.0): {wrong:.3e}");
    assert!(tight < TOL_TIGHT, "tight-width limit: {tight:.3e}");
    assert!(
        wrong > MUST_MATTER_FACTOR * TOL_TIGHT,
        "wrong width invisible: {wrong:.3e}"
    );
}

/// Shifting every QM atom AND every site leaves the energy unchanged: per QM
/// coordinate row, the QM-QM row sum plus the QM-site cross block (central FD
/// of the QM gradient w.r.t. the site positions, SCF re-solved) vanishes.
#[test]
fn translational_invariance_including_site_cross_block() {
    if !has_deriv2() {
        return;
    }
    let ext = embedding();
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let h = analytic(&solve(&mol, Some(&ext)), Some(&ext));
    let mut worst = 0.0f64;
    for axis in 0..3 {
        let mut sums = [0.0f64; 9];
        for (r, sum) in sums.iter_mut().enumerate() {
            *sum = (0..3).map(|a| h[(r, 3 * a + axis)]).sum();
        }
        for i in 0..ext.smeared_charges.len() {
            let g = |step: f64| {
                let mut e = ext.clone();
                let s = &mut e.smeared_charges[i];
                match axis {
                    0 => s.x += step,
                    1 => s.y += step,
                    _ => s.z += step,
                }
                rhf_grad(&mol, &e)
            };
            let (gp, gm) = (g(H_FULL), g(-H_FULL));
            for (r, sum) in sums.iter_mut().enumerate() {
                *sum += (gp[(r / 3, r % 3)] - gm[(r / 3, r % 3)]) / (2.0 * H_FULL);
            }
        }
        worst = sums.iter().fold(worst, |m, v| m.max(v.abs()));
    }
    eprintln!("smeared translational invariance: max |row sum| = {worst:.3e}");
    assert!(worst < TOL_FD, "row sum {worst:.3e}");
}

#[test]
fn empty_smeared_list_is_bit_identical_to_no_potential() {
    if !has_deriv2() {
        return;
    }
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let empty = ExternalPotential {
        smeared_charges: Vec::new(),
        ..Default::default()
    };
    let a = analytic(&solve(&mol, None), None);
    let b = analytic(&solve(&mol, Some(&empty)), Some(&empty));
    assert_eq!(max_diff(&a, &b), 0.0);
}

#[test]
fn smeared_charges_pass_preflight_and_polarizable_is_still_refused() {
    let op = Operator::coulomb();
    rhf_hessian_preflight(op, &config(Some(&embedding()))).unwrap();
    let pol = RhfConfig {
        polarizable: Some(Default::default()),
        ..config(Some(&embedding()))
    };
    let err = rhf_hessian_preflight(op, &pol).unwrap_err();
    assert!(err.to_string().contains("polarizable"), "{err}");
}
