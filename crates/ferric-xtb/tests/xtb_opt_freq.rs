//! GFN-xTB geometry optimization and harmonic frequencies through the shared
//! `ferric-scf` optimizer / finite-difference Hessian machinery.
//!
//! Run serially-safe (a process-wide lock guards libxtb, which is not
//! thread-safe):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 cargo test -p ferric-xtb --features xtb --test xtb_opt_freq
//! ```

#![cfg(feature = "xtb")]

use ferric_core::mol::Molecule;
use ferric_scf::optimize::OptimizeConfig;
use ferric_xtb::{
    harmonic_frequencies_xtb, optimize_geometry_xtb, xtb_energy_gradient, XtbConfig,
    DEFAULT_HESSIAN_DELTA,
};
use std::sync::{Mutex, MutexGuard, OnceLock};

fn xtb_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Stretched, bent water (Angstrom).
fn water_distorted() -> Molecule {
    Molecule::parse_xyz(
        "3\nd\nO 0.0 0.0 0.15\nH 0.0 0.90 -0.45\nH 0.0 -0.80 -0.50\n",
        0,
        1,
    )
    .unwrap()
}

fn co2_distorted() -> Molecule {
    Molecule::parse_xyz(
        "3\nd\nC 0.0 0.0 0.0\nO 0.0 0.0 1.20\nO 0.0 0.0 -1.10\n",
        0,
        1,
    )
    .unwrap()
}

fn tight() -> OptimizeConfig {
    OptimizeConfig {
        max_steps: 200,
        g_max_thresh: 1.0e-4,
        g_rms_thresh: 5.0e-5,
        e_conv: 1.0e-8,
        ..Default::default()
    }
}

fn gmax(g: &ndarray::Array2<f64>) -> f64 {
    g.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

/// The optimizer must converge, lower the energy, and leave a final gradient
/// below the threshold as measured by an INDEPENDENT xTB evaluation at the
/// returned geometry (not the optimizer's own bookkeeping).
#[test]
fn optimize_water_converges_to_small_gradient() {
    let _g = xtb_lock();
    let cfg = XtbConfig::default();
    let m0 = water_distorted();
    let (e0, g0) = xtb_energy_gradient(&m0, cfg).unwrap();
    assert!(gmax(&g0) > 1e-2, "start must be far from stationary");

    let opt = tight();
    let r = optimize_geometry_xtb(&m0, cfg, &opt).unwrap();
    assert!(
        r.converged,
        "optimization did not converge in {} steps",
        r.steps
    );
    assert!(r.energy < e0 - 1e-4);
    // The first step follows -gradient from a far-from-stationary start, so it
    // must go DOWNHILL; a sign error still ends at a stationary point but its
    // first probe climbs.
    assert!(
        r.energy_trace[1] < r.energy_trace[0],
        "first step did not descend: {:?}",
        &r.energy_trace[..2]
    );

    let (e1, g1) = xtb_energy_gradient(&r.mol, cfg).unwrap();
    assert!(
        (e1 - r.energy).abs() < 1e-9,
        "reported energy != xTB at geometry"
    );
    assert!(
        gmax(&g1) < opt.g_max_thresh,
        "final max gradient {:.3e} above threshold",
        gmax(&g1)
    );
}

/// Exactness anchor (trivial limit): started at the optimized geometry, the
/// optimizer must not move it.
#[test]
fn optimize_from_stationary_point_does_not_move() {
    let _g = xtb_lock();
    let cfg = XtbConfig::default();
    let r = optimize_geometry_xtb(&water_distorted(), cfg, &tight()).unwrap();
    let r2 = optimize_geometry_xtb(&r.mol, cfg, &tight()).unwrap();
    assert!(r2.converged);
    assert!((r2.energy - r.energy).abs() < 1e-8);
    for (a, b) in r.mol.atoms.iter().zip(&r2.mol.atoms) {
        let d = ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.zpos - b.zpos).powi(2)).sqrt();
        assert!(d < 1e-3, "atom moved {d:.2e} Bohr from a stationary point");
    }
}

/// An SCC that cannot converge inside the optimizer must raise, not silently
/// drive the search with a non-self-consistent surface.
#[test]
fn unconverged_scc_inside_optimize_raises() {
    let _g = xtb_lock();
    let cfg = XtbConfig {
        scc_max_iter: 1,
        ..Default::default()
    };
    let err = optimize_geometry_xtb(&water_distorted(), cfg, &tight())
        .expect_err("a 1-iteration SCC must not yield an optimized geometry");
    let msg = err.to_string();
    assert!(msg.contains("xtb"), "unexpected error text: {msg}");
}

/// Same guard on the Hessian path.
#[test]
fn unconverged_scc_inside_frequencies_raises() {
    let _g = xtb_lock();
    let cfg = XtbConfig {
        scc_max_iter: 1,
        ..Default::default()
    };
    assert!(harmonic_frequencies_xtb(&water_distorted(), cfg, DEFAULT_HESSIAN_DELTA).is_err());
}

/// Nonlinear water: 3 vibrations (all real, in the physical range), 6
/// translation/rotation modes near zero.
#[test]
fn water_frequencies_have_six_zero_modes() {
    let _g = xtb_lock();
    let cfg = XtbConfig::default();
    let r = optimize_geometry_xtb(&water_distorted(), cfg, &tight()).unwrap();
    let f = harmonic_frequencies_xtb(&r.mol, cfg, DEFAULT_HESSIAN_DELTA).unwrap();
    assert!(!f.is_linear);
    assert_eq!(f.trans_rot_frequencies.len(), 6);
    assert_eq!(f.frequencies.len(), 3);
    assert_eq!(f.n_imaginary(), 0);
    for t in &f.trans_rot_frequencies {
        assert!(t.abs() < 30.0, "trans/rot mode at {t} cm^-1");
    }
    assert!(
        f.frequencies[0] > 1000.0 && f.frequencies[0] < 2200.0,
        "{:?}",
        f.frequencies
    );
    assert!(
        f.frequencies[2] > 3000.0 && f.frequencies[2] < 4500.0,
        "{:?}",
        f.frequencies
    );
    assert_eq!(f.n_gradient_evaluations, 18);
}

/// Linear CO2: 5 translation/rotation modes near zero, 4 vibrations.
#[test]
fn co2_frequencies_have_five_zero_modes() {
    let _g = xtb_lock();
    let cfg = XtbConfig::default();
    let r = optimize_geometry_xtb(&co2_distorted(), cfg, &tight()).unwrap();
    let f = harmonic_frequencies_xtb(&r.mol, cfg, DEFAULT_HESSIAN_DELTA).unwrap();
    assert!(f.is_linear);
    assert_eq!(f.trans_rot_frequencies.len(), 5);
    assert_eq!(f.frequencies.len(), 4);
    for t in &f.trans_rot_frequencies {
        assert!(t.abs() < 30.0, "trans/rot mode at {t} cm^-1");
    }
    assert_eq!(f.n_imaginary(), 0);
}

/// The Hessian is the derivative of the surface the optimizer walks: a
/// diagonal element from second differences of the ENERGY must match the
/// Hessian built from differences of the GRADIENT.
#[test]
fn hessian_diagonal_matches_energy_second_difference() {
    let _g = xtb_lock();
    let cfg = XtbConfig::default();
    let r = optimize_geometry_xtb(&water_distorted(), cfg, &tight()).unwrap();
    let f = harmonic_frequencies_xtb(&r.mol, cfg, DEFAULT_HESSIAN_DELTA).unwrap();
    let h = 2.0e-2;
    let e = |d: f64| {
        let mut m = r.mol.clone();
        m.atoms[1].zpos += d; // H z
        xtb_energy_gradient(&m, cfg).unwrap().0
    };
    let fd = (e(h) - 2.0 * e(0.0) + e(-h)) / (h * h);
    let hz = f.cartesian_hessian[(5, 5)];
    assert!(
        (fd - hz).abs() < 5e-3 * hz.abs().max(1.0),
        "energy 2nd difference {fd:.6} vs gradient-difference Hessian {hz:.6}"
    );
    // Acoustic sum rule: rigid translation along z costs nothing.
    for a in 0..9 {
        let s: f64 = (0..3).map(|k| f.cartesian_hessian[(a, 3 * k + 2)]).sum();
        assert!(
            s.abs() < 1e-4,
            "translation sum rule violated: row {a}: {s:.2e}"
        );
    }
}
