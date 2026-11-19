//! Input validation and error propagation.

mod common;

use common::{lj_trimer, EnergyGradient};
use ferric_core::error::FerricError;
use ferric_core::mol::Molecule;
use ferric_md::{run_md, FnProvider, MdConfig, MdObserver, MdState, MdSystem, Thermostat};
use ndarray::Array2;

fn message(e: FerricError) -> String {
    e.to_string()
}

/// Wraps the LJ trimer so that step `bad_step` returns `fault`.
fn faulty(
    bad_step: usize,
    fault: fn(&mut (f64, Array2<f64>)) -> Option<FerricError>,
) -> FnProvider<EnergyGradient> {
    let (_, mut inner) = lj_trimer();
    let mut call = 0usize;
    FnProvider {
        n_atoms: 3,
        label: "faulty".into(),
        f: Box::new(move |x: &Array2<f64>| {
            let mut out = (inner.f)(x)?;
            let step = call;
            call += 1;
            if step == bad_step {
                if let Some(e) = fault(&mut out) {
                    return Err(e);
                }
            }
            Ok(out)
        }),
    }
}

fn run_faulty(p: &mut FnProvider<EnergyGradient>) -> String {
    let (sys, _) = lj_trimer();
    let cfg = MdConfig {
        n_steps: 5,
        ..MdConfig::default()
    };
    message(run_md(&sys, p, &cfg, &mut []).unwrap_err())
}

#[test]
fn provider_errors_are_wrapped_with_the_step_index() {
    let msg = run_faulty(&mut faulty(3, |_| {
        Some(FerricError::General("boom".into()))
    }));
    assert!(msg.contains("MD step 3") && msg.contains("boom"), "{msg}");
}

#[test]
fn non_finite_energy_or_gradient_is_an_error() {
    let msg = run_faulty(&mut faulty(2, |o| {
        o.0 = f64::NAN;
        None
    }));
    assert!(
        msg.contains("MD step 2") && msg.contains("non-finite"),
        "{msg}"
    );
    let msg = run_faulty(&mut faulty(0, |o| {
        o.1[[1, 2]] = f64::INFINITY;
        None
    }));
    assert!(
        msg.contains("MD step 0") && msg.contains("non-finite"),
        "{msg}"
    );
}

#[test]
fn misshapen_gradient_is_an_error() {
    let msg = run_faulty(&mut faulty(1, |o| {
        o.1 = Array2::zeros((2, 3));
        None
    }));
    assert!(msg.contains("MD step 1") && msg.contains("shape"), "{msg}");
}

#[test]
fn provider_atom_count_must_match_the_system() {
    let (sys, mut p) = lj_trimer();
    p.n_atoms = 4;
    let msg = message(run_md(&sys, &mut p, &MdConfig::default(), &mut []).unwrap_err());
    assert!(msg.contains("expects 4"), "{msg}");
}

struct FailAt(usize);

impl MdObserver for FailAt {
    fn on_frame(&mut self, _s: &MdSystem, state: &MdState) -> Result<(), FerricError> {
        if state.frame.step == self.0 {
            return Err(FerricError::General("observer full".into()));
        }
        Ok(())
    }
}

#[test]
fn observer_errors_abort_the_run() {
    let (sys, mut p) = lj_trimer();
    let cfg = MdConfig {
        n_steps: 5,
        ..MdConfig::default()
    };
    let msg = message(run_md(&sys, &mut p, &cfg, &mut [&mut FailAt(2)]).unwrap_err());
    assert!(msg.contains("observer full"), "{msg}");
}

#[test]
fn config_validation_rejects_bad_parameters() {
    let ok = MdConfig::default();
    assert!(ok.validate(3).is_ok());
    let cases: Vec<(MdConfig, &str)> = vec![
        (
            MdConfig {
                dt_fs: 0.0,
                ..ok.clone()
            },
            "dt_fs",
        ),
        (
            MdConfig {
                dt_fs: f64::NAN,
                ..ok.clone()
            },
            "dt_fs",
        ),
        (
            MdConfig {
                stride: 0,
                ..ok.clone()
            },
            "stride",
        ),
        (
            MdConfig {
                thermostat: Thermostat::Langevin {
                    friction_per_ps: 1.0,
                },
                ..ok.clone()
            },
            "requires temperature_k",
        ),
        (
            MdConfig {
                thermostat: Thermostat::Berendsen { tau_fs: 10.0 },
                ..ok.clone()
            },
            "requires temperature_k",
        ),
        (
            MdConfig {
                thermostat: Thermostat::Berendsen { tau_fs: 0.0 },
                temperature_k: Some(1.0),
                ..ok.clone()
            },
            "tau_fs",
        ),
        (
            MdConfig {
                thermostat: Thermostat::Langevin {
                    friction_per_ps: -1.0,
                },
                temperature_k: Some(1.0),
                ..ok.clone()
            },
            "friction",
        ),
        (
            MdConfig {
                temperature_k: Some(-5.0),
                ..ok.clone()
            },
            "temperature_k",
        ),
        (
            MdConfig {
                init_temperature_k: Some(f64::INFINITY),
                ..ok.clone()
            },
            "init_temperature_k",
        ),
        (
            MdConfig {
                initial_velocities: Some(Array2::zeros((2, 3))),
                ..ok.clone()
            },
            "shape",
        ),
        (
            MdConfig {
                initial_velocities: Some(Array2::from_elem((3, 3), f64::NAN)),
                ..ok.clone()
            },
            "non-finite",
        ),
    ];
    for (cfg, needle) in cases {
        let msg = message(cfg.validate(3).unwrap_err());
        assert!(msg.contains(needle), "expected {needle:?} in {msg}");
    }
    assert!(message(ok.validate(0).unwrap_err()).contains("no atoms"));
    // run_md validates before evaluating anything.
    let (sys, mut p) = lj_trimer();
    assert!(run_md(&sys, &mut p, &MdConfig { stride: 0, ..ok }, &mut []).is_err());
}

#[test]
fn thermostat_strings_parse_strictly() {
    assert_eq!(
        Thermostat::parse_config_str("nve", None, None),
        Ok(Thermostat::Nve)
    );
    assert_eq!(
        Thermostat::parse_config_str("Berendsen", Some(50.0), None),
        Ok(Thermostat::Berendsen { tau_fs: 50.0 })
    );
    assert_eq!(
        Thermostat::parse_config_str("langevin", None, Some(2.0)),
        Ok(Thermostat::Langevin {
            friction_per_ps: 2.0
        })
    );
    for t in [
        Thermostat::Nve,
        Thermostat::Berendsen { tau_fs: 1.0 },
        Thermostat::Langevin {
            friction_per_ps: 1.0,
        },
    ] {
        let (tau, fr) = match t {
            Thermostat::Berendsen { tau_fs } => (Some(tau_fs), None),
            Thermostat::Langevin { friction_per_ps } => (None, Some(friction_per_ps)),
            Thermostat::Nve => (None, None),
        };
        assert_eq!(Thermostat::parse_config_str(t.as_str(), tau, fr), Ok(t));
    }
    let err = |k, tau, fr| Thermostat::parse_config_str(k, tau, fr).unwrap_err();
    assert!(err("nose-hoover", None, None).contains("nve, berendsen, langevin"));
    assert!(err("berendsen", None, None).contains("tau_fs"));
    assert!(err("langevin", None, None).contains("friction_per_ps"));
    assert!(err("nve", Some(10.0), None).contains("tau_fs"));
    assert!(err("berendsen", Some(10.0), Some(1.0)).contains("friction_per_ps"));
    assert!(err("berendsen", Some(-1.0), None).contains("positive"));
    assert!(err("langevin", None, Some(f64::NAN)).contains("non-negative"));
}

#[test]
fn system_construction_is_validated() {
    let c = Array2::zeros((2, 3));
    let h2 = || vec!["H".to_string(), "H".to_string()];
    assert!(MdSystem::new(h2(), vec![1.0, 1.0], c.clone()).is_ok());
    assert!(MdSystem::new(h2(), vec![1.0], c.clone()).is_err());
    assert!(MdSystem::new(h2(), vec![1.0, 0.0], c.clone()).is_err());
    assert!(MdSystem::new(h2(), vec![1.0, f64::NAN], c.clone()).is_err());
    assert!(MdSystem::new(h2(), vec![1.0, 1.0], Array2::zeros((2, 2))).is_err());
    assert!(MdSystem::new(h2(), vec![1.0, 1.0], Array2::from_elem((2, 3), f64::NAN)).is_err());
    assert!(MdSystem::new(vec![], vec![], Array2::zeros((0, 3))).is_err());
}

#[test]
fn from_molecule_uses_standard_masses_and_bohr_coordinates() {
    let mol = Molecule::parse_xyz("2\n\nO 0 0 0\nH 0 0 1\n", 0, 2).unwrap();
    let sys = MdSystem::from_molecule(&mol, None).unwrap();
    assert_eq!(sys.symbols, vec!["O", "H"]);
    assert_eq!(sys.masses_amu, vec![15.999, 1.008]);
    assert_eq!(sys.coords[[1, 2]], mol.atoms[1].zpos);
    let heavy = MdSystem::from_molecule(&mol, Some(&[16.0, 2.014])).unwrap();
    assert_eq!(heavy.masses_amu, vec![16.0, 2.014]);
    assert!(MdSystem::from_molecule(&mol, Some(&[16.0])).is_err());
}

#[test]
fn from_molecule_rejects_ghosts_and_untabulated_elements() {
    let ghost = Molecule::parse_xyz("2\n\nH 0 0 0\n@H 0 0 1\n", 0, 2).unwrap();
    assert!(matches!(
        MdSystem::from_molecule(&ghost, None),
        Err(FerricError::Unsupported { .. })
    ));
    // The XYZ parser stops at Xe too, so promote a parsed atom to gold.
    let mut gold = Molecule::parse_xyz("1\n\nXe 0 0 0\n", 0, 1).unwrap();
    gold.atoms[0].symbol = "Au".into();
    gold.atoms[0].z = 79;
    let msg = message(MdSystem::from_molecule(&gold, None).unwrap_err());
    assert!(
        msg.contains("Z = 79") && msg.contains("masses_amu"),
        "{msg}"
    );
    assert!(MdSystem::from_molecule(&gold, Some(&[196.97])).is_ok());
}
