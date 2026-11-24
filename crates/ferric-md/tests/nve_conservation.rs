//! NVE conservation laws on anharmonic model potentials: total energy,
//! linear momentum (COM removed at start) and angular momentum (central
//! pair forces), plus the exact initial-temperature rescale.

mod common;

use common::{angular_momentum, lj_trimer, momentum, morse_dimer};
use ferric_md::{run_md, MdConfig, MdResult, MdSystem};

fn nve(
    sys: &MdSystem,
    provider: &mut dyn ferric_md::ForceProvider,
    t_k: f64,
    dt_fs: f64,
    n: usize,
) -> MdResult {
    let cfg = MdConfig {
        n_steps: n,
        dt_fs,
        init_temperature_k: Some(t_k),
        seed: 11,
        ..MdConfig::default()
    };
    run_md(sys, provider, &cfg, &mut []).unwrap()
}

fn energy_spread(res: &MdResult) -> f64 {
    let e0 = res.frames[0].total;
    res.frames
        .iter()
        .map(|f| (f.total - e0).abs())
        .fold(0.0, f64::max)
}

#[test]
fn initial_temperature_is_rescaled_exactly() {
    let (sys, mut p) = lj_trimer();
    let res = nve(&sys, &mut p, 300.0, 1.0, 0);
    assert_eq!(res.n_dof, 6);
    let t0 = res.frames[0].temperature_k;
    assert!((t0 - 300.0).abs() < 1e-10, "T0 = {t0}");
    let p0 = momentum(&sys.masses_amu, &res.velocities[0]);
    assert!(p0.iter().all(|x| x.abs() < 1e-12), "COM momentum {p0:?}");
}

#[test]
fn morse_dimer_conserves_energy_and_momentum() {
    let (sys, mut p) = morse_dimer();
    let res = nve(&sys, &mut p, 300.0, 0.1, 2000);
    let drift = res.max_conserved_drift();
    let ke = res.frames[0].kinetic;
    let pmax = res
        .velocities
        .iter()
        .flat_map(|v| momentum(&sys.masses_amu, v))
        .fold(0.0, |m: f64, x| m.max(x.abs()));
    eprintln!("morse drift {drift:e} (KE0 {ke:e}), max |P| {pmax:e}");
    // Measured drift 3.3e-6 Eh at dt = 0.1 fs over 200 fs (KE0 1.4e-3).
    assert!(drift < 1e-5, "drift {drift:e}");
    assert!(drift > 0.0);
    assert_eq!(drift, energy_spread(&res), "NVE conserved must equal total");
    // Measured ~1e-16 (momentum scale ~1).
    assert!(pmax < 1e-13, "momentum {pmax:e}");
}

#[test]
fn morse_energy_error_halves_quadratically() {
    let (sys, mut p) = morse_dimer();
    let coarse = nve(&sys, &mut p, 300.0, 0.2, 1000).max_conserved_drift();
    let fine = nve(&sys, &mut p, 300.0, 0.1, 2000).max_conserved_drift();
    let ratio = coarse / fine;
    eprintln!("morse dt ratio {ratio}");
    // Measured 4.0; second-order integrator.
    assert!((3.5..4.5).contains(&ratio), "ratio {ratio}");
}

#[test]
fn lj_trimer_conserves_energy_and_angular_momentum() {
    let (sys, mut p) = lj_trimer();
    let res = nve(&sys, &mut p, 30.0, 2.0, 2000);
    let drift = res.max_conserved_drift();
    let l0 = angular_momentum(&sys.masses_amu, &res.positions[0], &res.velocities[0]);
    let l_scale = l0.iter().fold(0.0, |m: f64, x| m.max(x.abs()));
    let dl = res
        .positions
        .iter()
        .zip(&res.velocities)
        .map(|(x, v)| {
            let l = angular_momentum(&sys.masses_amu, x, v);
            (0..3).fold(0.0, |m: f64, k| m.max((l[k] - l0[k]).abs()))
        })
        .fold(0.0, f64::max);
    let pmax = res
        .velocities
        .iter()
        .flat_map(|v| momentum(&sys.masses_amu, v))
        .fold(0.0, |m: f64, x| m.max(x.abs()));
    eprintln!(
        "lj drift {drift:e} (KE0 {:e}), |L0| {l_scale:e} dL {dl:e}, |P| {pmax:e}",
        res.frames[0].kinetic
    );
    // Measured drift 6.1e-9 Eh at dt = 2 fs over 4 ps (KE0 2.9e-4).
    assert!(drift < 3e-8, "drift {drift:e}");
    let max_pair = res
        .positions
        .iter()
        .flat_map(|x| {
            [(0, 1), (0, 2), (1, 2)]
                .map(|(i, j)| (&x.row(i) - &x.row(j)).dot(&(&x.row(i) - &x.row(j))).sqrt())
        })
        .fold(0.0, f64::max);
    assert!(
        max_pair < 12.0,
        "trimer dissociated: max pair distance {max_pair}"
    );
    assert!(l_scale > 1.0, "L0 should be non-trivial: {l_scale:e}");
    assert!(dl < 1e-12 * l_scale, "dL {dl:e}");
    assert!(pmax < 1e-12, "momentum {pmax:e}");
}
