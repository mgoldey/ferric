//! The RI-MP2 gradient on an RI-JK reference (`scf_jk = "rijk"`): the
//! derivative of the energy the fitted SCF and RI-MP2 compute.
//!
//! # What is compared
//!
//! [`total_rimp2_gradient_on`] with `scf_jk = "rijk"` (RI-J + RI-K on
//! def2-universal-jkfit; SCF, Z-vector, correlation-density potential and the
//! separable two-electron derivative all fitted,
//! `book/src/15-correlated-gradients/02-rimp2-gradient.md#rimp2-gradient-fitted-ri-jk-reference`)
//! against a
//! central finite difference of the SAME fitted total energy, `E_HF[RI-JK] +
//! E_MP2[RI]`, on distorted water and methane (cc-pVDZ / cc-pvdz-ri) and on
//! water with one frozen core orbital.
//!
//! # Bar: 2e-7 Ha/Bohr, from the FD step and the SCF and Z-vector tolerances
//!
//! * central-difference truncation `h²|E'''|/6` with `h = 5e-4` Bohr and
//!   `|E'''| ≲ 2 Ha/Bohr³` (bond stretches at these geometries): ≤ 8e-8;
//! * energy noise between the displaced SCFs, `ε_E/h`: the SCF converges
//!   `ΔP` to 1e-9, so its energy is converged to `O(ΔP²)` and the
//!   geometry-to-geometry noise is that of the integral screens, ≲ 1e-11 Ha,
//!   giving ≤ 2e-8;
//! * the analytic side's orbital residual (`ΔP ≤ 1e-9` times an O(1)
//!   coupling) and Z-vector residual (`max|r| < 1e-8` times an O(1)
//!   Lagrangian): ≤ 2e-8.
//!
//! Sum ≤ 1.2e-7, rounded up to 2e-7. The bar must stay this tight: four-centre
//! K in place of the SCF's RI-K in the Z-vector right-hand side alone moves
//! water by 9e-7 Ha/Bohr, which a 1e-6 bar would pass. The four-centre route
//! is checked against FD in `gradient.rs`'s `test_analytical_vs_fd_*`.
//!
//! # Exactness anchor
//!
//! `ferric_scf::response_jk`'s own tests: the four-centre route through the
//! route type is bit for bit the direct calls it replaced, and the fitted
//! bilinear derivative at `d1 = d2` is the fitted SCF's two-electron gradient.
//!
//! # Reported, not asserted
//!
//! Fitted minus four-centre gradient next to the fitted minus four-centre
//! energy: the RI-JK fit error, which the gradient should mirror in size.

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_mp2::gradient::total_rimp2_gradient_on;
use ferric_mp2::rimp2::{ri_mp2, RiMp2Config};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::scf_jk::ScfJk;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::warm_start::ScfWarmStart;
use ndarray::Array2;

const JK_AUX: &str = "def2-universal-jkfit";
const H: f64 = 5e-4;
const BAR: f64 = 2e-7;

fn rijk() -> ScfJk {
    ScfJk::rijk(JK_AUX)
}

/// `E_HF + E_MP2` on `scf_jk`, the SCF converged as tightly as the gradient's.
fn total_energy(
    mol: &Molecule,
    obs: &BasisSet,
    aux: &BasisSet,
    cfg: &RiMp2Config,
    jk: &ScfJk,
) -> f64 {
    let op = Operator::coulomb();
    let prep = PreparedBasis::new(mol, obs).unwrap();
    let dfbs = PreparedBasis::new(mol, aux).unwrap();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let mut rhf_cfg = RhfConfig::default()
        .with_density_conv(1e-9)
        .for_geometry_task();
    jk.apply(&mut rhf_cfg);
    let ctx = ferric_core::parallel::ParallelContext::default();
    let rhf = solve_rhf(&ctx, mol, &prep, op, &bounds, &rhf_cfg).unwrap();
    assert!(rhf.converged);
    ri_mp2(mol, &prep, &dfbs, op, &rhf, cfg)
        .unwrap()
        .total_energy
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

fn analytic(
    mol: &Molecule,
    obs: &BasisSet,
    aux: &BasisSet,
    cfg: &RiMp2Config,
    jk: &ScfJk,
) -> (f64, Array2<f64>) {
    total_rimp2_gradient_on(
        mol,
        obs,
        aux,
        Operator::coulomb(),
        cfg,
        None,
        jk,
        &mut ScfWarmStart::new(false),
    )
    .unwrap()
}

/// Analytic fitted gradient vs central FD of the fitted energy, every
/// component; also prints the fit error against the four-centre route.
fn check(label: &str, xyz: &str, cfg: &RiMp2Config) {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs = basis::bundled("cc-pvdz").unwrap();
    let aux = basis::bundled("cc-pvdz-ri").unwrap();
    let (e_fit, g_fit) = analytic(&mol, &obs, &aux, cfg, &rijk());
    let e_direct = total_energy(&mol, &obs, &aux, cfg, &rijk());
    assert!(
        (e_fit - e_direct).abs() < 1e-9,
        "{label}: gradient driver energy {e_fit} vs direct {e_direct}"
    );
    let mut worst = 0.0f64;
    for atom in 0..mol.atoms.len() {
        for coord in 0..3 {
            let ep = total_energy(&displaced(&mol, atom, coord, H), &obs, &aux, cfg, &rijk());
            let em = total_energy(&displaced(&mol, atom, coord, -H), &obs, &aux, cfg, &rijk());
            let fd = (ep - em) / (2.0 * H);
            let d = (g_fit[(atom, coord)] - fd).abs();
            eprintln!(
                "{label} atom {atom} coord {coord}: analytic {:+.10} fd {fd:+.10} |d| {d:.2e}",
                g_fit[(atom, coord)]
            );
            worst = worst.max(d);
        }
    }
    let (e_ex, g_ex) = analytic(&mol, &obs, &aux, cfg, &ScfJk::exact());
    let dg = (&g_fit - &g_ex).iter().fold(0.0f64, |m, v| m.max(v.abs()));
    eprintln!(
        "{label}: max |analytic - FD| = {worst:.3e} (bar {BAR:.0e}); RI-JK fit error: \
         E_fit - E_exact = {:+.3e} Ha, max |g_fit - g_exact| = {dg:.3e} Ha/Bohr",
        e_fit - e_ex
    );
    assert!(worst < BAR, "{label}: max |analytic - FD| = {worst:.3e}");
}

const WATER: &str = "3\nwater, off C2v\nO 0.02 -0.01 0.117790\nH 0.0 0.785453 -0.451161\nH 0.03 -0.735453 -0.491161\n";

#[test]
fn rijk_gradient_matches_fd_of_the_fitted_energy_water() {
    check("H2O", WATER, &RiMp2Config::default());
}

#[test]
fn rijk_gradient_matches_fd_of_the_fitted_energy_methane() {
    let xyz = "5\nmethane, distorted\nC 0.01 0.00 -0.02\nH 0.64 0.62 0.66\nH -0.60 -0.65 0.61\nH -0.66 0.60 -0.62\nH 0.62 -0.63 -0.67\n";
    check("CH4", xyz, &RiMp2Config::default());
}

#[test]
fn rijk_gradient_matches_fd_of_the_fitted_energy_water_frozen_core() {
    let cfg = RiMp2Config {
        frozen_core: 1,
        ..Default::default()
    };
    check("H2O fc=1", WATER, &cfg);
}
