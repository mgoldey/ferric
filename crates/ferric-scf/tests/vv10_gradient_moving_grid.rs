//! The VV10 nuclear gradient is the exact derivative of ferric's moving-grid
//! VV10 energy (book/src/14-derivatives/03-xc-gradient.md, section
//! xc-gradient-vv10).
//!
//! Every test here holds the AO density matrix D fixed and differentiates
//! `E_nl(R) = Σ_i W_i ρ_i [β + ½ f_i]` on the NLC grid rebuilt at R, which is
//! what `vv10_gradient_from_density` claims to return. The SCF only supplies a
//! realistic D; the identities hold for any D with ρ ≥ 0.
//!
//! - `..._lone_atom_...` is the exactness anchor: one atom carries every AO
//!   and every grid point, so E_nl does not depend on its position and the
//!   gradient is zero. A gradient that keeps only the AO term returns the
//!   grid error of a generic (non-spherical) density instead.
//! - `..._forces_sum_to_zero` checks translational invariance, which the
//!   four terms satisfy on any grid and the AO term alone does not.
//! - `..._matches_central_difference_*` compares with a central difference
//!   of E_nl. The bar is derived from the measured FD floor (see `FD_BAR`).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::{build_atomic_grid_pruned, AtomicGridConfig};
use ferric_dft::libxc::{xc_def_from_name, Vv10Params};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ferric_testkit::compare::max_abs;
use ndarray::Array2;

/// Asymmetric water (Å): no component of the gradient vanishes by symmetry.
const WATER: &str = "3\n\nO 0.0 0.0 0.0\nH 0.05 0.7572 0.5868\nH -0.03 -0.70 0.62\n";
/// OH doublet (Å), off-axis so x, y and z all carry force.
const OH: &str = "2\n\nO 0.0 0.0 0.0\nH 0.12 0.21 0.95\n";

/// Central-difference step (Bohr).
const STEP: f64 = 1e-4;
/// Analytic-vs-FD bar (Ha/Bohr). Measured agreement is 6.8e-12 (water) and
/// 4.1e-12 (OH), the O(h²) and rounding floor of the step-1e-4 difference;
/// the AO term alone misses by 5.5e-6 and 3.5e-6.
const FD_BAR: f64 = 1e-10;
/// Net-force bar (Ha/Bohr). Measured 3e-17 (rounding of the four sums); the
/// AO term alone leaves 2.3e-6.
const NET_BAR: f64 = 1e-14;

fn params() -> Vv10Params {
    xc_def_from_name("wB97X-V").unwrap().vv10.unwrap()
}

fn nlc_cfg() -> AtomicGridConfig {
    AtomicGridConfig::default_nlc()
}

/// E_nl at fixed D on the NLC grid built for `mol` (points and weights ride
/// the nuclei, the AOs ride their atoms).
fn nlc_energy(mol: &Molecule, bs: &basis::BasisSet, d: &Array2<f64>) -> f64 {
    let cfg = nlc_cfg();
    let grid = build_atomic_grid_pruned(mol, &cfg, cfg.prune).unwrap();
    let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
    let (chi, dchi) = ferric_dft::ao_grid::eval_basis_and_grad_on_points(mol, bs, &pts).unwrap();
    let dens = ferric_dft::density_on_grid::eval_density_closed(d, &chi, &dchi);
    ferric_dft::vv10::compute_vv10_energy_and_potentials(&grid, &dens, &params()).0
}

fn analytic(mol: &Molecule, bs: &basis::BasisSet, d: &Array2<f64>) -> Array2<f64> {
    let prep = PreparedBasis::new(mol, bs).unwrap();
    ferric_dft::gradient::vv10_gradient_from_density(
        mol,
        bs,
        d,
        &params(),
        &nlc_cfg(),
        prep.shell_to_atom(),
        prep.shell_offsets(),
        prep.shell_dims(),
    )
    .unwrap()
}

fn displaced(mol: &Molecule, atom: usize, axis: usize, h: f64) -> Molecule {
    let mut m = mol.clone();
    match axis {
        0 => m.atoms[atom].x += h,
        1 => m.atoms[atom].y += h,
        _ => m.atoms[atom].zpos += h,
    }
    m
}

fn central_difference(mol: &Molecule, bs: &basis::BasisSet, d: &Array2<f64>) -> Array2<f64> {
    let n = mol.atoms.len();
    let mut g = Array2::<f64>::zeros((n, 3));
    for a in 0..n {
        for x in 0..3 {
            let ep = nlc_energy(&displaced(mol, a, x, STEP), bs, d);
            let em = nlc_energy(&displaced(mol, a, x, -STEP), bs, d);
            g[(a, x)] = (ep - em) / (2.0 * STEP);
        }
    }
    g
}

fn rks_density(xyz: &str, basis_name: &str) -> (Molecule, basis::BasisSet, Array2<f64>) {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        xc: Some("wB97X-V".into()),
        energy_conv: 1e-8,
        density_conv: 1e-6,
        ..Default::default()
    };
    let res = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg).unwrap();
    (mol, bs, res.density_total)
}

fn uks_total_density(xyz: &str, basis_name: &str) -> (Molecule, basis::BasisSet, Array2<f64>) {
    let mol = Molecule::parse_xyz(xyz, 0, 2).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let cfg = RhfConfig {
        xc: Some("wB97X-V".into()),
        energy_conv: 1e-8,
        density_conv: 1e-6,
        max_iter: 400,
        ..Default::default()
    };
    let res = solve_uhf(&ParallelContext::default(), &mol, &prep, &bounds, &cfg).unwrap();
    // E_nl is a functional of the total density; the UKS/ROKS gradients pass
    // alpha + beta (`ks_gradient.rs::add_xc_gradient`).
    (mol, bs, res.density_total)
}

fn net_force(g: &Array2<f64>) -> f64 {
    (0..3)
        .map(|x| g.column(x).sum().abs())
        .fold(0.0_f64, f64::max)
}

fn check_fd(label: &str, mol: &Molecule, bs: &basis::BasisSet, d: &Array2<f64>) {
    let g = analytic(mol, bs, d);
    let fd = central_difference(mol, bs, d);
    let diff = max_abs(&(&g - &fd));
    eprintln!(
        "{label}: max|analytic - FD| = {diff:.2e} (|G| = {:.2e}, net = {:.2e})",
        max_abs(&g),
        net_force(&g)
    );
    assert!(
        diff < FD_BAR,
        "{label}: VV10 gradient misses the central difference of E_nl by {diff:.2e} > {FD_BAR:.0e}"
    );
}

/// Exactness anchor. One O atom with a generic sp-hybrid density (not
/// spherical, so the Lebedev grid's symmetry does not zero the grid error):
/// E_nl is independent of the atom's position, so the gradient is zero.
#[test]
fn vv10_gradient_of_a_lone_atom_is_zero() {
    let mol = Molecule::parse_xyz("1\n\nO 0.1 -0.2 0.3\n", 0, 1).unwrap();
    let bs = basis::bundled("6-31g").unwrap();
    let nbf = PreparedBasis::new(&mol, &bs).unwrap().nbasis();
    assert_eq!(nbf, 9);
    // 6-31G O: 1s, 2s, 2p(3), 3s, 3p(3).
    let c1 = [1.0, 0.3, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0];
    let c2 = [0.0, 0.4, 0.3, 0.5, 0.7, 0.2, 0.1, 0.2, 0.3];
    let mut d = Array2::<f64>::zeros((nbf, nbf));
    for i in 0..nbf {
        for j in 0..nbf {
            d[(i, j)] = 2.0 * (c1[i] * c1[j] + c2[i] * c2[j]);
        }
    }
    let fd = central_difference(&mol, &bs, &d);
    let g = analytic(&mol, &bs, &d);
    eprintln!(
        "lone O: max|G| = {:.2e}, max|FD| = {:.2e}",
        max_abs(&g),
        max_abs(&fd)
    );
    // The moving-grid energy really is translation-invariant (bit-level up to
    // the AO evaluation's rounding).
    assert!(max_abs(&fd) < 1e-10, "lone-atom FD {:.2e}", max_abs(&fd));
    // Measured 5.5e-18; the AO term alone gives 2.9e-6.
    assert!(
        max_abs(&g) < 1e-14,
        "lone-atom VV10 gradient {:.2e} is not zero",
        max_abs(&g)
    );
}

#[test]
fn vv10_gradient_forces_sum_to_zero() {
    let (mol, bs, d) = rks_density(WATER, "6-31g");
    let g = analytic(&mol, &bs, &d);
    let net = net_force(&g);
    eprintln!(
        "water VV10: net force {net:.2e}, max|G| {:.2e}",
        max_abs(&g)
    );
    assert!(
        net < NET_BAR,
        "VV10 forces sum to {net:.2e} > {NET_BAR:.0e}"
    );
}

#[test]
fn vv10_gradient_matches_central_difference_rks_water() {
    let (mol, bs, d) = rks_density(WATER, "6-31g");
    check_fd("RKS water/6-31G wB97X-V", &mol, &bs, &d);
}

#[test]
fn vv10_gradient_matches_central_difference_uks_oh() {
    let (mol, bs, d) = uks_total_density(OH, "6-31g");
    check_fd("UKS OH/6-31G wB97X-V (total density)", &mol, &bs, &d);
}
