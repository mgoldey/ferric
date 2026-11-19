//! Velocity Verlet on a 3D harmonic well against its closed-form discrete
//! solution.
//!
//! For `x'' = -w^2 x` velocity Verlet obeys `x_{n+1} = (2 - w^2 dt^2) x_n -
//! x_{n-1}`, so `x_n = x_0 cos(n th) + dt v_0 sin(n th) / sin(th)` with
//! `cos(th) = 1 - w^2 dt^2 / 2`, and it conserves the shadow energy
//! `p^2/2m + (k x^2/2)(1 - w^2 dt^2/4)` exactly.

mod common;

use common::{dt_au, harmonic_wells, unit_mass_me};
use ferric_md::{run_md, MdConfig, MdResult, MdSystem};
use ndarray::{arr2, Array2};

const K: f64 = 0.5;

fn single_atom() -> MdSystem {
    MdSystem::new(vec!["X".into()], vec![1.0], arr2(&[[0.3, -0.2, 0.1]])).unwrap()
}

fn v0() -> Array2<f64> {
    arr2(&[[1.0e-3, 0.0, -2.0e-3]])
}

fn run(k: f64, sys: &MdSystem, v: Array2<f64>, dt_fs: f64, n_steps: usize) -> MdResult {
    let cfg = MdConfig {
        n_steps,
        dt_fs,
        initial_velocities: Some(v),
        ..MdConfig::default()
    };
    run_md(sys, &mut harmonic_wells(vec![k]), &cfg, &mut []).unwrap()
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0, |m, x| m.max(x.abs()))
}

/// Trivial limit: with no force the integrator is an exact straight line.
#[test]
fn free_particle_moves_in_a_straight_line() {
    let sys = single_atom();
    let res = run(0.0, &sys, v0(), 0.5, 100);
    let dt = dt_au(0.5);
    for (n, x) in res.positions.iter().enumerate() {
        let exact = &sys.coords + &(&v0() * (n as f64 * dt));
        assert!(max_abs_diff(x, &exact) < 1e-14, "step {n}");
        assert_eq!(res.velocities[n], v0(), "velocity changed at step {n}");
    }
    assert_eq!(res.frames[0].potential, 0.0);
}

#[test]
fn velocity_verlet_matches_closed_form_discrete_solution() {
    let sys = single_atom();
    let n_steps = 400;
    let res = run(K, &sys, v0(), 0.5, n_steps);
    let dt = dt_au(0.5);
    let w2 = K / unit_mass_me();
    let theta = (1.0 - 0.5 * w2 * dt * dt).acos();
    let exact = |n: usize| {
        let nt = n as f64 * theta;
        &sys.coords * nt.cos() + &v0() * (dt * nt.sin() / theta.sin())
    };
    let mut worst: f64 = 0.0;
    for (n, x) in res.positions.iter().enumerate() {
        worst = worst.max(max_abs_diff(x, &exact(n)));
    }
    // Velocities: v_n = (x_{n+1} - x_{n-1}) / (2 dt) for velocity Verlet.
    for n in 1..n_steps {
        let fd = (&exact(n + 1) - &exact(n - 1)) / (2.0 * dt);
        assert!(
            max_abs_diff(&res.velocities[n], &fd) < 1e-14,
            "v at step {n}"
        );
    }
    // Measured 1.0e-14 Bohr over 400 steps; |x| ~ 0.3 Bohr.
    eprintln!("max position error {worst:e}");
    assert!(worst < 1e-12, "max position error {worst:e}");
}

#[test]
fn shadow_energy_is_conserved_while_true_energy_oscillates() {
    let sys = single_atom();
    let res = run(K, &sys, v0(), 0.5, 400);
    let dt = dt_au(0.5);
    let m = unit_mass_me();
    let w2dt2 = K / m * dt * dt;
    let shadow: Vec<f64> = res
        .positions
        .iter()
        .zip(&res.velocities)
        .map(|(x, v)| {
            0.5 * m * v.iter().map(|a| a * a).sum::<f64>()
                + 0.5 * K * x.iter().map(|a| a * a).sum::<f64>() * (1.0 - 0.25 * w2dt2)
        })
        .collect();
    let spread = |e: &[f64]| {
        let (lo, hi) = e
            .iter()
            .fold((f64::MAX, f64::MIN), |(l, h), &x| (l.min(x), h.max(x)));
        hi - lo
    };
    let totals: Vec<f64> = res.frames.iter().map(|f| f.total).collect();
    let (s, t) = (spread(&shadow), spread(&totals));
    eprintln!("shadow spread {s:e}, true spread {t:e}, E {:e}", shadow[0]);
    // Measured: shadow spread 1.6e-16 Eh (5e-15 relative, round-off).
    assert!(s < 1e-13 * shadow[0].abs(), "shadow spread {s:e}");
    assert!(
        t > 1e6 * s,
        "true energy spread {t:e} should dwarf shadow {s:e}"
    );
}

#[test]
fn energy_error_scales_as_dt_squared() {
    let sys = single_atom();
    let err = |dt_fs: f64| {
        let n = (100.0 / dt_fs).round() as usize;
        let res = run(K, &sys, v0(), dt_fs, n);
        let e0 = res.frames[0].total;
        res.frames
            .iter()
            .map(|f| (f.total - e0).abs())
            .fold(0.0, f64::max)
    };
    let ratio = err(0.4) / err(0.2);
    eprintln!("dt^2 ratio {ratio}");
    // Leading order: exactly 4; measured 3.99.
    assert!((ratio - 4.0).abs() < 0.1, "ratio {ratio}");
}

#[test]
fn reversing_velocities_retraces_the_trajectory() {
    let sys = single_atom();
    let fwd = run(K, &sys, v0(), 0.5, 300);
    let back_sys = MdSystem::new(
        sys.symbols.clone(),
        sys.masses_amu.clone(),
        fwd.final_state.coords.clone(),
    )
    .unwrap();
    let back = run(K, &back_sys, -&fwd.final_state.velocities, 0.5, 300);
    let dx = max_abs_diff(&back.final_state.coords, &sys.coords);
    let dv = max_abs_diff(&back.final_state.velocities, &(-v0()));
    eprintln!("reversal dx {dx:e} dv {dv:e}");
    // Measured dx 7e-16 Bohr, dv 5e-18.
    assert!(dx < 1e-12, "position error {dx:e}");
    assert!(dv < 1e-14, "velocity error {dv:e}");
}
