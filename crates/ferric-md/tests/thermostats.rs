//! Berendsen and Langevin (BAOAB) thermostats: trivial limits that must
//! reproduce NVE, relaxation to the target, conservation of `E - Q`, and
//! canonical averages on independent harmonic wells.

mod common;

use common::{harmonic_system, harmonic_wells, lj_trimer};
use ferric_md::{run_md, MdConfig, MdResult, Thermostat, BOLTZMANN_HARTREE_PER_K};

fn trimer_run(thermostat: Thermostat, remove_com_motion: bool) -> MdResult {
    let (sys, mut p) = lj_trimer();
    let cfg = MdConfig {
        n_steps: 500,
        dt_fs: 2.0,
        thermostat,
        temperature_k: Some(30.0),
        seed: 3,
        remove_com_motion,
        ..MdConfig::default()
    };
    run_md(&sys, &mut p, &cfg, &mut []).unwrap()
}

#[test]
fn berendsen_with_infinite_tau_is_nve() {
    let nve = trimer_run(Thermostat::Nve, true);
    let ber = trimer_run(
        Thermostat::Berendsen {
            tau_fs: f64::INFINITY,
        },
        true,
    );
    assert_eq!(nve.positions, ber.positions);
    assert_eq!(nve.velocities, ber.velocities);
    for (a, b) in nve.frames.iter().zip(&ber.frames) {
        assert_eq!(a.total.to_bits(), b.total.to_bits());
        assert_eq!(a.conserved.to_bits(), b.conserved.to_bits());
    }
}

#[test]
fn langevin_zero_friction_matches_velocity_verlet() {
    // remove_com_motion = false so both runs use 3N dof and the same draw.
    let nve = trimer_run(Thermostat::Nve, false);
    let lan = trimer_run(
        Thermostat::Langevin {
            friction_per_ps: 0.0,
        },
        false,
    );
    assert_eq!(nve.n_dof, lan.n_dof);
    let mut dx: f64 = 0.0;
    let mut de: f64 = 0.0;
    for ((xa, xb), (fa, fb)) in nve
        .positions
        .iter()
        .zip(&lan.positions)
        .zip(nve.frames.iter().zip(&lan.frames))
    {
        dx = dx.max((xa - xb).iter().fold(0.0, |m, x| m.max(x.abs())));
        de = de.max((fa.conserved - fb.conserved).abs());
    }
    eprintln!("langevin(0) vs VV: dx {dx:e}, d conserved {de:e}");
    // Measured ~1e-13 Bohr (A(dt/2)A(dt/2) vs A(dt) round-off), 0 Eh.
    assert!(dx < 1e-12, "dx {dx:e}");
    assert!(de < 1e-12, "de {de:e}");
}

/// Harmonic run with `n` independent wells of force constant `k`.
fn wells_run(n: usize, k: f64, cfg: &MdConfig) -> MdResult {
    let sys = harmonic_system(n);
    run_md(&sys, &mut harmonic_wells(vec![k; n]), cfg, &mut []).unwrap()
}

#[test]
fn berendsen_relaxes_to_target_and_conserves_e_minus_q() {
    let cfg = MdConfig {
        n_steps: 4000,
        dt_fs: 0.5,
        thermostat: Thermostat::Berendsen { tau_fs: 20.0 },
        temperature_k: Some(300.0),
        init_temperature_k: Some(50.0),
        remove_com_motion: false,
        seed: 5,
        keep_trajectory: false,
        ..MdConfig::default()
    };
    let res = wells_run(16, 0.05, &cfg);
    let half = res.frames.len() / 2;
    let late = res.mean_temperature(half);
    for skip in [0, 1, half] {
        let kept = &res.frames[skip..];
        let manual = kept.iter().map(|f| f.temperature_k).sum::<f64>() / kept.len() as f64;
        let got = res.mean_temperature(skip);
        assert!(
            (got - manual).abs() < 1e-12 * manual,
            "skip {skip}: {got} vs {manual}"
        );
    }
    assert!(res.mean_temperature(res.frames.len()).is_nan());
    let heat = res.frames.last().unwrap().total - res.frames[0].total;
    let drift = res.max_conserved_drift();
    eprintln!("berendsen late T {late}, heat {heat:e}, E-Q drift {drift:e}");
    assert!(res.frames[0].temperature_k < 51.0);
    // Measured 299.x K over the second half.
    assert!((late - 300.0).abs() < 10.0, "late mean T {late}");
    // E - Q moves only by the velocity-Verlet energy error, whose amplitude
    // for a harmonic well is ~(w dt)^2 / 8 of the energy (measured 3.5e-5
    // Eh against this bound of ~1.3e-4), while E itself grows by 3e-2.
    let w2dt2 = 0.05 / common::unit_mass_me() * common::dt_au(0.5).powi(2);
    let bound = 0.25 * w2dt2 * res.frames.last().unwrap().total;
    assert!(heat > 1e-2, "heat {heat:e}");
    assert!(
        drift < bound,
        "drift {drift:e} vs bound {bound:e} (heat {heat:e})"
    );
}

#[test]
fn langevin_samples_the_canonical_harmonic_ensemble() {
    let (n, k, t0) = (32, 0.05, 300.0);
    let cfg = MdConfig {
        n_steps: 40_000,
        dt_fs: 0.5,
        thermostat: Thermostat::Langevin {
            friction_per_ps: 20.0,
        },
        temperature_k: Some(t0),
        init_temperature_k: Some(t0),
        seed: 17,
        stride: 4,
        ..MdConfig::default()
    };
    let res = wells_run(n, k, &cfg);
    assert_eq!(res.n_dof, 3 * n);
    let skip = 500;
    let mean_t = res.mean_temperature(skip);
    let kept = &res.positions[skip..];
    let x2: f64 = kept
        .iter()
        .map(|x| x.iter().map(|a| a * a).sum::<f64>())
        .sum::<f64>()
        / (kept.len() * 3 * n) as f64;
    let x2_exact = BOLTZMANN_HARTREE_PER_K * t0 / k;
    let drift = res.max_conserved_drift();
    let q = res.frames.last().unwrap().total - res.frames.last().unwrap().conserved;
    eprintln!(
        "langevin <T> {mean_t}, <x^2>/exact {}, E-Q drift {drift:e}, |Q| {q:e}",
        x2 / x2_exact
    );
    // Measured within ~1% of target; statistical error ~0.5%.
    assert!((mean_t / t0 - 1.0).abs() < 0.03, "mean T {mean_t}");
    assert!(
        (x2 / x2_exact - 1.0).abs() < 0.03,
        "<x^2> ratio {}",
        x2 / x2_exact
    );
    // Measured E - Q drift 9.4e-5 Eh; energy scale n_dof kT = 9.1e-2 Eh.
    let scale = res.n_dof as f64 * BOLTZMANN_HARTREE_PER_K * t0;
    assert!(
        drift < 5e-3 * scale,
        "E - Q drift {drift:e} vs scale {scale:e}"
    );
}

/// Trivial limit of the O step: at T = 0 with no force, BAOAB is pure
/// friction, so `v_n = v_0 exp(-gamma n dt)` with `gamma dt` computed here
/// directly in user units (`friction_per_ps * dt_fs / 1000`).
#[test]
fn langevin_at_zero_temperature_damps_a_free_particle_exactly() {
    let sys = harmonic_system(1);
    let v0 = ndarray::arr2(&[[1e-3, -2e-3, 5e-4]]);
    let (friction_per_ps, dt_fs) = (40.0, 0.5);
    let cfg = MdConfig {
        n_steps: 200,
        dt_fs,
        thermostat: Thermostat::Langevin { friction_per_ps },
        temperature_k: Some(0.0),
        initial_velocities: Some(v0.clone()),
        ..MdConfig::default()
    };
    let res = run_md(&sys, &mut harmonic_wells(vec![0.0]), &cfg, &mut []).unwrap();
    for (n, v) in res.velocities.iter().enumerate() {
        let decay = (-friction_per_ps * dt_fs / 1000.0 * n as f64).exp();
        let err = (v - &(&v0 * decay))
            .iter()
            .fold(0.0, |m: f64, x| m.max(x.abs()));
        assert!(err < 1e-15, "step {n}: error {err:e}");
    }
    // All kinetic energy is drained into Q, so E - Q is exactly conserved.
    assert!(res.final_state.frame.kinetic < 1e-3 * res.frames[0].kinetic);
    assert!(
        res.max_conserved_drift() < 1e-15,
        "{:e}",
        res.max_conserved_drift()
    );
}
