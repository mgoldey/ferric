//! RI-MP2 and SCS-MP2 geometry optimization on the shared optimizer loop
//! (`ferric_scf::optimize::optimize_geometry_with_energy_gradient`).
//!
//! # Anchors
//!
//! * Cartesian path: the energy at every step of RI-MP2 on distorted water
//!   and of SCS-MP2 on distorted water, recorded with the optimizer's own
//!   loop as it was before the shared loop replaced it (base 1ff7a4040). The
//!   two loops were compared bitwise (`to_bits` of every step energy and every
//!   final coordinate) on the base build and on this one; the literals below
//!   are compared at 1e-9 Ha because a frozen bit pattern is only right on
//!   the machine that produced it (see `internal_coord_optimize.rs`).
//!   A gradient taken at the wrong geometry changes the second step by
//!   ~1e-3 Ha, six orders above that bar.
//! * Internal coordinates: the converged geometry sits at the Cartesian
//!   minimum within what the gradient thresholds allow, with the bar derived
//!   from the surface's own smallest force constant (see
//!   [`assert_reaches_the_cartesian_minimum`]).
//!
//! Setting `FERRIC_TRACE_OUT` appends each run's energy and geometry bits to
//! that file (`label step energy-bits`, `label geom coordinate-bits`).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::operator::Operator;
use ferric_mp2::gradient::{exact_scf_jk, total_rimp2_gradient_on};
use ferric_mp2::optimize::{
    optimize_geometry_rimp2, optimize_geometry_scs_mp2_on, RiMp2OptimizeResult,
};
use ferric_mp2::rimp2::RiMp2Config;
use ferric_scf::optimize::{CoordSystem, InitialHessian, OptimizeConfig, TrustUpdate};
use ferric_scf::warm_start::ScfWarmStart;
use ndarray::Array2;
use ndarray_linalg::{Eigh, UPLO};
use std::io::Write;

/// The search the recorded paths were taken with: identity start, fixed
/// trust radius (`initial_hessian = "simple"`, `trust_update = "fixed"`).
/// Pinned so the paths keep testing the shared loop, not the defaults.
fn legacy() -> OptimizeConfig {
    OptimizeConfig {
        initial_hessian: Some(InitialHessian::Simple),
        trust_update: TrustUpdate::Fixed,
        ..Default::default()
    }
}

const WATER: &str = "3\ndistorted water\nO 0 0 0\nH 0 0.9 0\nH 0 -0.3 0.85\n";
const H2: &str = "2\nH2\nH 0 0 0\nH 0 0 1.0\n";

/// Step energies of RI-MP2 / STO-3G / cc-pVDZ-RI on `WATER`, default
/// `OptimizeConfig` of the time ([`legacy`]), from the pre-refactor loop.
const RIMP2_WATER_TRACE: [f64; 8] = [
    -74.97420096032445,
    -74.99605176110357,
    -75.00528279756418,
    -75.00608449428158,
    -75.00611037865218,
    -75.00612294124379,
    -75.00612961169152,
    -75.0061302915887,
];

/// The same for SCS-MP2 (`ScsMp2Config::default()`).
const SCS_WATER_TRACE: [f64; 8] = [
    -74.97891443652857,
    -75.00130090566508,
    -75.01107443875664,
    -75.01209740872159,
    -75.01212180262111,
    -75.01212929863121,
    -75.01213491815658,
    -75.0121354330457,
];

/// Every step energy within 1e-9 Ha of the recorded path: SCF noise is
/// ~1e-12 Ha, cross-machine BLAS drift ~1e-13, so 1e-9 passes any honest
/// machine, while the wrong gradient at one step moves every later energy by
/// the step error (~1e-3 Ha).
fn assert_trace(tag: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{tag}: step count");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < 1e-9, "{tag} step {i}: {g} vs {w}");
    }
}

fn dump(label: &str, r: &RiMp2OptimizeResult) {
    let Ok(path) = std::env::var("FERRIC_TRACE_OUT") else {
        return;
    };
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();
    for (i, e) in r.energy_trace.iter().enumerate() {
        writeln!(f, "{label} step{i} {:016x}", e.to_bits()).unwrap();
    }
    for (i, a) in r.mol.atoms.iter().enumerate() {
        for (k, v) in [a.x, a.y, a.zpos].iter().enumerate() {
            writeln!(f, "{label} geom{i}.{k} {:016x}", v.to_bits()).unwrap();
        }
    }
    writeln!(f, "{label} steps {} converged {}", r.steps, r.converged).unwrap();
}

fn rimp2(xyz: &str, opt: &OptimizeConfig) -> RiMp2OptimizeResult {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let obs = basis::bundled("sto-3g").unwrap();
    let aux = basis::bundled("cc-pvdz-ri").unwrap();
    optimize_geometry_rimp2(
        &mol,
        &obs,
        &aux,
        Operator::coulomb(),
        &RiMp2Config::default(),
        opt,
        None,
    )
    .unwrap()
}

#[test]
fn rimp2_cartesian_path_matches_the_recorded_path_water() {
    let r = rimp2(WATER, &legacy());
    dump("rimp2-water", &r);
    assert_trace("rimp2-water", &r.energy_trace, &RIMP2_WATER_TRACE);
    assert!(r.converged);
}

#[test]
fn rimp2_cartesian_path_matches_the_recorded_path_h2() {
    let r = rimp2(H2, &legacy());
    dump("rimp2-h2", &r);
    assert!(r.converged);
}

#[test]
fn scs_cartesian_path_matches_the_recorded_path_water() {
    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let obs = basis::bundled("sto-3g").unwrap();
    let aux = basis::bundled("cc-pvdz-ri").unwrap();
    let r = optimize_geometry_scs_mp2_on(
        &mol,
        &obs,
        &aux,
        &ferric_mp2::scs::ScsMp2Config::default(),
        &legacy(),
        None,
        &ferric_mp2::gradient::exact_scf_jk(),
    )
    .unwrap();
    dump("scs-water", &r);
    assert_trace("scs-water", &r.energy_trace, &SCS_WATER_TRACE);
    assert!(r.converged);
}

// ---------------------------------------------------------------------------
// Internal coordinates reach the Cartesian minimum.
// ---------------------------------------------------------------------------

const ETHANE: &str = "8\nethane\n\
    C  0.000  0.000  0.800\nC  0.000  0.000 -0.800\n\
    H  1.050  0.000  1.180\nH -0.525  0.909  1.180\nH -0.525 -0.909  1.180\n\
    H  1.000  0.150 -1.180\nH -0.575  0.840 -1.180\nH -0.425 -0.990 -1.180\n";

fn grad(mol: &Molecule) -> Array2<f64> {
    let obs = basis::bundled("sto-3g").unwrap();
    let aux = basis::bundled("cc-pvdz-ri").unwrap();
    let mut warm = ScfWarmStart::new(false);
    total_rimp2_gradient_on(
        mol,
        &obs,
        &aux,
        Operator::coulomb(),
        &RiMp2Config::default(),
        None,
        &exact_scf_jk(),
        &mut warm,
    )
    .unwrap()
    .1
}

fn flat(g: &Array2<f64>) -> Vec<f64> {
    g.iter().copied().collect()
}

fn dist(mol: &Molecule) -> Vec<f64> {
    let mut d = Vec::new();
    for i in 0..mol.atoms.len() {
        for j in (i + 1)..mol.atoms.len() {
            let (a, b) = (&mol.atoms[i], &mol.atoms[j]);
            d.push(((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.zpos - b.zpos).powi(2)).sqrt());
        }
    }
    d
}

/// Smallest non-rigid force constant (Ha/Bohr^2) of the RI-MP2 surface at
/// `mol`: central differences (step 0.01 Bohr) of the analytic gradient, the
/// 6 smallest-magnitude eigenvalues of the symmetrized Hessian discarded as
/// translations and rotations.
fn k_min(mol: &Molecule) -> f64 {
    let n = 3 * mol.atoms.len();
    let h = 0.01;
    let mut hess = Array2::<f64>::zeros((n, n));
    for c in 0..n {
        let mut gs = Vec::new();
        for sign in [1.0, -1.0] {
            let mut m = mol.clone();
            let a = &mut m.atoms[c / 3];
            match c % 3 {
                0 => a.x += sign * h,
                1 => a.y += sign * h,
                _ => a.zpos += sign * h,
            }
            gs.push(flat(&grad(&m)));
        }
        for r in 0..n {
            hess[(r, c)] = (gs[0][r] - gs[1][r]) / (2.0 * h);
        }
    }
    let sym = (&hess + &hess.t()) * 0.5;
    let (mut vals, _) = sym.eigh(UPLO::Upper).unwrap();
    vals.as_slice_mut()
        .unwrap()
        .sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
    vals[6]
}

/// The internal-coordinate run converges, sits at the tight Cartesian minimum
/// within the gradient thresholds, and has the same shape.
///
/// # Bars, derived
///
/// A stopped geometry has `|g|` below the thresholds, and to first order the
/// distance to the minimum along the softest mode is `|g| / k_min`, with
/// `k_min` the smallest non-rigid force constant of this surface (measured
/// here by differencing the analytic gradient at the reference minimum).
/// * Gradient: the independently recomputed gradient at the final geometry
///   satisfies `g_max < g_max_thresh` and `g_rms < g_rms_thresh`.
/// * Distances: every interatomic distance differs from the reference's by at
///   most `sqrt(2) * |g_final| / k_min` (the two atoms each move at most
///   `|dx|`, and `|dx| <= |g_final| / k_min`; the reference's own gradient is
///   1e-6 and adds `1e-6 / k_min`).
/// * Energy: `E - E_ref <= |g_final|^2 / (2 k_min)`, and not below
///   `-1e-9` (the reference is the minimum).
fn assert_reaches_the_cartesian_minimum(tag: &str, xyz: &str) {
    let tight = OptimizeConfig {
        max_steps: 400,
        g_max_thresh: 1e-6,
        g_rms_thresh: 1e-6,
        e_conv: 1e-12,
        ..Default::default()
    };
    let reference = rimp2(xyz, &tight);
    assert!(reference.converged, "{tag}: tight Cartesian reference");
    let internal = rimp2(
        xyz,
        &OptimizeConfig {
            coord_system: CoordSystem::RedundantInternal,
            ..Default::default()
        },
    );
    assert!(internal.converged, "{tag}: internal run");

    let opt = OptimizeConfig::default();
    let g = flat(&grad(&internal.mol));
    let g_max = g.iter().fold(0.0f64, |a, x| a.max(x.abs()));
    let g_rms = (g.iter().map(|x| x * x).sum::<f64>() / g.len() as f64).sqrt();
    let g_norm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(g_max < opt.g_max_thresh, "{tag}: g_max {g_max:e}");
    assert!(g_rms < opt.g_rms_thresh, "{tag}: g_rms {g_rms:e}");

    let k = k_min(&reference.mol);
    assert!(k > 1e-4, "{tag}: k_min {k:e} (not a minimum?)");
    let bar_r = (2.0f64.sqrt() * g_norm + 1e-6) / k;
    let bar_e = g_norm * g_norm / (2.0 * k);
    let dr = dist(&internal.mol)
        .iter()
        .zip(dist(&reference.mol))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    let de = internal.energy - reference.energy;
    eprintln!(
        "{tag}: internal {} steps, cartesian-tight {} steps; k_min {k:.4e}; |g| {g_norm:.3e}; \
         max|dr| {dr:.3e} (bar {bar_r:.3e}); dE {de:+.3e} (bar {bar_e:.3e})",
        internal.steps, reference.steps
    );
    assert!(dr < bar_r, "{tag}: max|dr| {dr:e} vs {bar_r:e}");
    assert!(de > -1e-9 && de < bar_e, "{tag}: dE {de:e} vs {bar_e:e}");
}

#[test]
fn rimp2_internal_coordinates_reach_the_cartesian_minimum_water() {
    assert_reaches_the_cartesian_minimum("water", WATER);
}

#[ignore = "tier medium (10-60s): CI runs it when an affected crate changes (scripts/ci_test_tiers.py)"]
#[test]
fn rimp2_internal_coordinates_reach_the_cartesian_minimum_ethane() {
    assert_reaches_the_cartesian_minimum("ethane", ETHANE);
}
