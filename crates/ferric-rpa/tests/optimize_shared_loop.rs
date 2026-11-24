//! RPA geometry optimization on the shared optimizer loop.
//!
//! Anchor: the per-step energies and the final geometry of the Cartesian path
//! are the path the loop had before it moved into `ferric_scf::optimize`
//! (recorded at the base commit with the old loop; the energies are compared
//! at 1e-9 Ha, the bitwise comparison was made against the base build and
//! lives in the commit message, because golden bits are not portable across
//! machines).
//!
//! Internal coordinates: the converged H2 bond equals the tight Cartesian
//! one within `2 * g_max_thresh / k`, `k` the measured force constant (the
//! stopped bond is `|g| / k` from the minimum, each run's own residual).
//!
//! Setting `FERRIC_TRACE_OUT` appends each run's energy and geometry bits to
//! that file (`label step energy-bits`, `label geom coordinate-bits`).

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_integrals::operator::Operator;
use ferric_rpa::config::{QuadratureConfig, QuadratureScheme};
use ferric_rpa::gradient::RpaGradientMethod;
use ferric_rpa::optimize::{optimize_geometry_rpa, RpaOptimizeResult};
use ferric_rpa::PdepRpaConfig;
use ferric_scf::optimize::{CoordSystem, InitialHessian, OptimizeConfig, TrustUpdate};
use ferric_scf::warm_start::ScfWarmStart;
use std::io::Write;

fn cfg() -> PdepRpaConfig {
    PdepRpaConfig {
        frozen_core: 0,
        trunc_thresh: 0.0,
        eigensolver_conv_thresh: 1e-10,
        quadrature: QuadratureConfig {
            scheme: QuadratureScheme::GaussLegendre,
            n_points: 8,
            u0: 0.5,
        },
        ..Default::default()
    }
}

fn run(gradient: RpaGradientMethod, opt: &OptimizeConfig) -> RpaOptimizeResult {
    let mol = Molecule::parse_xyz("2\nH2\nH 0 0 0\nH 0 0 0.95\n", 0, 1).unwrap();
    let obs = basis::bundled("sto-3g").unwrap();
    let aux = basis::bundled("cc-pvdz-ri").unwrap();
    optimize_geometry_rpa(&mol, &obs, &aux, Operator::coulomb(), &cfg(), opt, gradient).unwrap()
}

fn dump(label: &str, r: &RpaOptimizeResult) {
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

/// Path recorded with the pre-refactor loop (base 1ff7a4040), H2/STO-3G
/// from r = 0.95 Angstrom, analytic gradient, default thresholds.
#[test]
fn rpa_cartesian_path_matches_the_recorded_path_analytic() {
    let r = run(RpaGradientMethod::Analytic, &legacy());
    dump("rpa-analytic", &r);
    assert!(r.converged);
    assert_eq!(r.energy_trace.len(), r.steps + 1);
    for (i, (g, w)) in r.energy_trace.iter().zip(&RPA_H2_TRACE).enumerate() {
        assert!((g - w).abs() < 1e-9, "step {i}: {g} vs {w}");
    }
    assert_eq!(r.energy_trace.len(), RPA_H2_TRACE.len());
}

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

/// Step energies of H2/STO-3G RPA (analytic gradient, 8-point Gauss-Legendre,
/// full rank) from r = 0.95 Angstrom, from the pre-refactor loop (base
/// 1ff7a4040; bitwise equal on the base and the shared-loop build).
const RPA_H2_TRACE: [f64; 6] = [
    -1.1073747376267167,
    -1.1224571231002207,
    -1.1333306622909114,
    -1.1375252189584948,
    -1.137529108403189,
    -1.1375293442324141,
];

#[test]
fn rpa_cartesian_path_matches_the_recorded_path_fd() {
    let r = run(RpaGradientMethod::FiniteDifference { h: 5e-4 }, &legacy());
    dump("rpa-fd", &r);
    assert!(r.converged);
}

fn bond(r: &RpaOptimizeResult) -> f64 {
    (r.mol.atoms[0].zpos - r.mol.atoms[1].zpos).abs()
}

#[test]
fn rpa_internal_coordinates_reach_the_cartesian_minimum_h2() {
    let tight = OptimizeConfig {
        max_steps: 100,
        g_max_thresh: 1e-6,
        g_rms_thresh: 1e-6,
        e_conv: 1e-12,
        ..Default::default()
    };
    let reference = run(RpaGradientMethod::Analytic, &tight);
    let internal = run(
        RpaGradientMethod::Analytic,
        &OptimizeConfig {
            coord_system: CoordSystem::RedundantInternal,
            ..Default::default()
        },
    );
    assert!(reference.converged && internal.converged);
    // Force constant along the bond at the reference minimum, Ha/Bohr^2.
    let obs = basis::bundled("sto-3g").unwrap();
    let aux = basis::bundled("cc-pvdz-ri").unwrap();
    let g_at = |dz: f64| {
        let mut m = reference.mol.clone();
        m.atoms[1].zpos += dz;
        let mut warm = ScfWarmStart::new(false);
        ferric_rpa::gradient::total_rpa_gradient_by(
            &m,
            &obs,
            &aux,
            Operator::coulomb(),
            &cfg(),
            RpaGradientMethod::Analytic,
            &mut warm,
        )
        .unwrap()
        .1[(1, 2)]
    };
    let k = (g_at(0.01) - g_at(-0.01)) / 0.02;
    let defaults = OptimizeConfig::default();
    let bar = 2.0 * defaults.g_max_thresh / k;
    let dr = (bond(&internal) - bond(&reference)).abs();
    eprintln!("H2 RPA: k {k:.4} Ha/Bohr^2, |dr| {dr:.3e} Bohr, bar {bar:.3e}");
    assert!(k > 0.1, "k {k}");
    assert!(dr < bar, "|dr| {dr:e} vs {bar:e}");
}
