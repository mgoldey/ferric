//! Seeds, strides, trajectory retention, observers, degrees of freedom and
//! initial-velocity rules.

mod common;

use common::{harmonic_system, harmonic_wells, lj_trimer};
use ferric_core::error::FerricError;
use ferric_md::{run_md, MdConfig, MdObserver, MdResult, MdState, MdSystem, Thermostat};
use ndarray::Array2;

fn langevin_cfg(seed: u64) -> MdConfig {
    MdConfig {
        n_steps: 60,
        dt_fs: 2.0,
        thermostat: Thermostat::Langevin {
            friction_per_ps: 5.0,
        },
        temperature_k: Some(30.0),
        seed,
        ..MdConfig::default()
    }
}

fn trimer(cfg: &MdConfig) -> MdResult {
    let (sys, mut p) = lj_trimer();
    run_md(&sys, &mut p, cfg, &mut []).unwrap()
}

#[test]
fn same_seed_is_bitwise_reproducible() {
    let a = trimer(&langevin_cfg(42));
    let b = trimer(&langevin_cfg(42));
    assert_eq!(a.positions, b.positions);
    assert_eq!(a.velocities, b.velocities);
    let bits = |r: &MdResult| {
        r.frames
            .iter()
            .map(|f| f.conserved.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&a), bits(&b));
}

#[test]
fn different_seeds_give_different_trajectories() {
    let a = trimer(&langevin_cfg(1));
    let b = trimer(&langevin_cfg(2));
    assert_ne!(a.velocities[0], b.velocities[0]);
    assert_ne!(a.positions.last(), b.positions.last());
}

#[test]
fn stride_records_a_bitwise_subset_of_the_full_run() {
    let full = trimer(&langevin_cfg(7));
    let strided = trimer(&MdConfig {
        stride: 7,
        ..langevin_cfg(7)
    });
    let steps: Vec<usize> = strided.frames.iter().map(|f| f.step).collect();
    assert_eq!(steps, (0..=60).step_by(7).collect::<Vec<_>>());
    for (k, f) in strided.frames.iter().enumerate() {
        assert_eq!(
            f.conserved.to_bits(),
            full.frames[f.step].conserved.to_bits()
        );
        assert_eq!(f.time_fs, 2.0 * f.step as f64);
        assert_eq!(strided.positions[k], full.positions[f.step]);
    }
    // Step 60 is not a multiple of 7, but the final state is still step 60.
    assert_eq!(strided.final_state.frame.step, 60);
    assert_eq!(strided.final_state.coords, full.positions[60]);
    assert_eq!(strided.n_force_evaluations, 61);
}

#[test]
fn keep_trajectory_false_drops_arrays_but_not_frames() {
    let full = trimer(&langevin_cfg(9));
    let lean = trimer(&MdConfig {
        keep_trajectory: false,
        ..langevin_cfg(9)
    });
    assert!(lean.positions.is_empty() && lean.velocities.is_empty());
    assert_eq!(lean.frames.len(), full.frames.len());
    assert_eq!(lean.final_state.coords, full.final_state.coords);
}

struct Counter {
    steps: Vec<usize>,
}

impl MdObserver for Counter {
    fn on_frame(&mut self, _s: &MdSystem, state: &MdState) -> Result<(), FerricError> {
        self.steps.push(state.frame.step);
        Ok(())
    }
}

#[test]
fn observers_see_every_recorded_frame_and_provider_calls_are_counted() {
    let (sys, mut p) = lj_trimer();
    let mut calls = 0usize;
    let mut counting = ferric_md::FnProvider {
        n_atoms: 3,
        label: "counted".into(),
        f: |x: &Array2<f64>| {
            calls += 1;
            (p.f)(x)
        },
    };
    let mut a = Counter { steps: vec![] };
    let mut b = Counter { steps: vec![] };
    let cfg = MdConfig {
        n_steps: 10,
        stride: 3,
        ..MdConfig::default()
    };
    let res = run_md(&sys, &mut counting, &cfg, &mut [&mut a, &mut b]).unwrap();
    assert_eq!(a.steps, vec![0, 3, 6, 9]);
    assert_eq!(b.steps, a.steps);
    assert_eq!(res.label, "counted");
    assert_eq!(res.n_force_evaluations, 11);
    assert_eq!(calls, 11);
}

#[test]
fn degrees_of_freedom_follow_the_com_rules() {
    let dof = |thermostat, remove_com_motion, n| {
        let cfg = MdConfig {
            thermostat,
            temperature_k: Some(10.0),
            remove_com_motion,
            ..MdConfig::default()
        };
        run_md(
            &harmonic_system(n),
            &mut harmonic_wells(vec![0.1; n]),
            &cfg,
            &mut [],
        )
        .unwrap()
        .n_dof
    };
    let ber = Thermostat::Berendsen { tau_fs: 10.0 };
    let lan = Thermostat::Langevin {
        friction_per_ps: 1.0,
    };
    assert_eq!(dof(Thermostat::Nve, true, 4), 9);
    assert_eq!(dof(ber, true, 4), 9);
    assert_eq!(dof(Thermostat::Nve, false, 4), 12);
    assert_eq!(dof(lan, true, 4), 12);
    assert_eq!(dof(Thermostat::Nve, true, 1), 3);
}

#[test]
fn langevin_removes_com_motion_at_initialisation_only() {
    let res = trimer(&langevin_cfg(4));
    let p0 = common::momentum(&[39.948; 3], &res.velocities[0]);
    let p_end = common::momentum(&[39.948; 3], res.velocities.last().unwrap());
    assert!(p0.iter().all(|x| x.abs() < 1e-12), "initial P {p0:?}");
    assert!(
        p_end.iter().any(|x| x.abs() > 1e-6),
        "noise should move the COM: {p_end:?}"
    );
    // 3N dof and exact rescale at step 0.
    assert!((res.frames[0].temperature_k - 30.0).abs() < 1e-10);
}

#[test]
fn explicit_velocities_are_used_verbatim_and_default_is_at_rest() {
    let (sys, mut p) = lj_trimer();
    let v = Array2::from_shape_fn((3, 3), |(i, k)| 1e-5 * (i as f64 + 0.5 * k as f64));
    let cfg = MdConfig {
        initial_velocities: Some(v.clone()),
        init_temperature_k: Some(500.0),
        ..MdConfig::default()
    };
    let res = run_md(&sys, &mut p, &cfg, &mut []).unwrap();
    assert_eq!(res.velocities[0], v);
    let rest = run_md(&sys, &mut p, &MdConfig::default(), &mut []).unwrap();
    assert!(rest.velocities[0].iter().all(|&x| x == 0.0));
    assert_eq!(rest.frames[0].temperature_k, 0.0);
    assert_eq!(rest.frames.len(), 1);
}

#[test]
fn init_temperature_overrides_target_temperature() {
    let cfg = MdConfig {
        thermostat: Thermostat::Berendsen { tau_fs: 100.0 },
        temperature_k: Some(300.0),
        init_temperature_k: Some(20.0),
        ..MdConfig::default()
    };
    let res = trimer(&cfg);
    assert!((res.frames[0].temperature_k - 20.0).abs() < 1e-10);
}
