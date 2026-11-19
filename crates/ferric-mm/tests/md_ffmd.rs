//! Force-field MD (`run_ffmd`, `MmForceProvider`): the provider is exactly
//! the MM gradient, and NVE / Langevin runs on flexible TIP3P water behave.
//!
//! Anchor first: the provider must hand the integrator `ferric_mm::gradient`
//! bit for bit, and `run_ffmd` must be bitwise `run_md` over a closure that
//! calls that gradient, so every property measured afterwards is a property
//! of the integrator on the MM surface and not of the adapter.

use ferric_core::units::ANGSTROM_TO_BOHR;
use ferric_md::{run_md, FnProvider, ForceProvider, MdConfig, MdResult, MdSystem, Thermostat};
use ferric_mm::{
    gradient, gradient_with, run_ffmd, CutoffParams, MmForceProvider, MmTopology, NonbondedMethod,
};
use ndarray::{array, Array2};

const O_MASS: f64 = 15.999;
const H_MASS: f64 = 1.008;

/// Flexible TIP3P (charges, O LJ, H-O bonds and the H-O-H angle) for
/// `n_waters` molecules laid out O, H, H, O, H, H, ...
fn tip3p_flexible(n_waters: usize) -> MmTopology {
    let mut charges = Vec::new();
    let mut lj = Vec::new();
    let mut bonds = Vec::new();
    let mut angles = Vec::new();
    for w in 0..n_waters {
        let o = 3 * w;
        charges.extend([-0.834, 0.417, 0.417]);
        lj.extend([(3.150_61, 0.1521), (0.0, 0.0), (0.0, 0.0)]);
        bonds.push((o, o + 1, 553.0, 0.9572));
        bonds.push((o, o + 2, 553.0, 0.9572));
        angles.push((o + 1, o, o + 2, 100.0, 104.52));
    }
    MmTopology::from_amber_units(charges, lj, bonds, angles, vec![]).expect("TIP3P topology")
}

/// One water, slightly off its equilibrium geometry (Angstrom input).
fn water_coords() -> Array2<f64> {
    array![[0.0, 0.0, 0.0], [0.98, 0.0, 0.0], [-0.22, 0.93, 0.05],] * ANGSTROM_TO_BOHR
}

/// A hydrogen-bonded water dimer (donor H1 -> acceptor O3), Angstrom input.
fn dimer_coords() -> Array2<f64> {
    array![
        [0.0, 0.0, 0.0],
        [0.9572, 0.0, 0.0],
        [-0.24, 0.927, 0.0],
        [2.90, 0.0, 0.0],
        [3.20, 0.80, 0.45],
        [3.20, -0.80, 0.45],
    ] * ANGSTROM_TO_BOHR
}

fn system(coords: Array2<f64>) -> MdSystem {
    let n_waters = coords.nrows() / 3;
    let mut symbols = Vec::new();
    let mut masses = Vec::new();
    for _ in 0..n_waters {
        symbols.extend(["O", "H", "H"].map(String::from));
        masses.extend([O_MASS, H_MASS, H_MASS]);
    }
    MdSystem::new(symbols, masses, coords).expect("system")
}

fn nve(n_steps: usize, dt_fs: f64, init_t: f64, seed: u64) -> MdConfig {
    MdConfig {
        n_steps,
        dt_fs,
        init_temperature_k: Some(init_t),
        seed,
        ..MdConfig::default()
    }
}

fn assert_frames_bitwise_equal(a: &MdResult, b: &MdResult) {
    assert_eq!(a.frames.len(), b.frames.len());
    for (fa, fb) in a.frames.iter().zip(&b.frames) {
        assert_eq!(
            fa.potential.to_bits(),
            fb.potential.to_bits(),
            "step {}",
            fa.step
        );
        assert_eq!(
            fa.kinetic.to_bits(),
            fb.kinetic.to_bits(),
            "step {}",
            fa.step
        );
        assert_eq!(
            fa.conserved.to_bits(),
            fb.conserved.to_bits(),
            "step {}",
            fa.step
        );
    }
    for (pa, pb) in a.positions.iter().zip(&b.positions) {
        for (x, y) in pa.iter().zip(pb) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }
}

#[test]
fn provider_gradient_is_bitwise_the_mm_gradient() {
    let top = tip3p_flexible(2);
    let x = dimer_coords();
    let mut p = MmForceProvider::new(&top, NonbondedMethod::NoCutoff);
    assert_eq!(p.n_atoms(), 6);
    let (e, g) = p.energy_gradient(&x).unwrap();
    let (e_ref, g_ref) = gradient(&top, &x).unwrap();
    assert_eq!(e.to_bits(), e_ref.total.to_bits());
    assert_eq!(
        p.last, e_ref,
        "`last` must hold the components just evaluated"
    );
    for (a, b) in g.iter().zip(&g_ref) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    let cutoff = NonbondedMethod::CutoffNonPeriodic(CutoffParams::new(2.0 * ANGSTROM_TO_BOHR));
    let mut pc = MmForceProvider::new(&top, cutoff);
    let (ec, gc) = pc.energy_gradient(&x).unwrap();
    let (ec_ref, gc_ref) = gradient_with(&top, &x, &cutoff).unwrap();
    assert_eq!(ec.to_bits(), ec_ref.total.to_bits());
    assert_ne!(
        ec.to_bits(),
        e.to_bits(),
        "a 2 A cutoff must change the dimer energy"
    );
    for (a, b) in gc.iter().zip(&gc_ref) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert_eq!(p.label(), "mm");
    assert_eq!(pc.label(), "mm:cutoff");
}

#[test]
fn run_ffmd_is_bitwise_run_md_over_the_mm_gradient() {
    let top = tip3p_flexible(2);
    let sys = system(dimer_coords());
    let cfg = nve(50, 0.5, 300.0, 7);
    let ff = run_ffmd(&top, &sys, &NonbondedMethod::NoCutoff, &cfg, &mut []).unwrap();
    let mut closure = FnProvider {
        n_atoms: 6,
        label: "mm".to_string(),
        f: |x: &Array2<f64>| gradient(&top, x).map(|(e, g)| (e.total, g)),
    };
    let reference = run_md(&sys, &mut closure, &cfg, &mut []).unwrap();
    assert_frames_bitwise_equal(&ff, &reference);
    assert_eq!(ff.label, "mm");
    assert_eq!(ff.n_force_evaluations, 51);
    // Step 0 is the starting geometry's MM energy.
    let e0 = gradient(&top, &sys.coords).unwrap().0.total;
    assert_eq!(ff.frames[0].potential.to_bits(), e0.to_bits());
}

#[test]
fn a_topology_system_size_mismatch_is_refused() {
    let top = tip3p_flexible(2);
    let err = run_ffmd(
        &top,
        &system(water_coords()),
        &NonbondedMethod::NoCutoff,
        &nve(1, 0.5, 300.0, 0),
        &mut [],
    )
    .expect_err("3 atoms against a 6-atom topology");
    assert!(err.to_string().contains("topology"), "{err}");
}

/// Drift of `E_pot + E_kin` relative to the mean kinetic energy.
fn relative_drift(r: &MdResult) -> f64 {
    let mean_ke = r.frames.iter().map(|f| f.kinetic).sum::<f64>() / r.frames.len() as f64;
    r.max_conserved_drift() / mean_ke
}

#[test]
fn flexible_water_nve_conserves_energy() {
    // O-H stretch period ~9 fs; 0.1 fs resolves it ~90 times per period.
    let top = tip3p_flexible(1);
    let r = run_ffmd(
        &top,
        &system(water_coords()),
        &NonbondedMethod::NoCutoff,
        &nve(2000, 0.1, 300.0, 3),
        &mut [],
    )
    .unwrap();
    let rel = relative_drift(&r);
    eprintln!(
        "water NVE: max |dE| = {:.3e} Eh, rel to <KE> = {rel:.3e}",
        r.max_conserved_drift()
    );
    assert!(rel < 2e-3, "relative drift {rel:.3e}");
    // Velocity Verlet's error is O(dt^2): 4x the step must raise the drift.
    let coarse = run_ffmd(
        &top,
        &system(water_coords()),
        &NonbondedMethod::NoCutoff,
        &nve(500, 0.4, 300.0, 3),
        &mut [],
    )
    .unwrap();
    let rel_coarse = relative_drift(&coarse);
    eprintln!("water NVE dt=0.4: rel = {rel_coarse:.3e}");
    assert!(rel_coarse > 8.0 * rel, "{rel_coarse:.3e} vs {rel:.3e}");
}

#[test]
fn water_dimer_nve_conserves_energy_and_momentum() {
    let top = tip3p_flexible(2);
    let sys = system(dimer_coords());
    let r = run_ffmd(
        &top,
        &sys,
        &NonbondedMethod::NoCutoff,
        &nve(4000, 0.1, 200.0, 11),
        &mut [],
    )
    .unwrap();
    let rel = relative_drift(&r);
    eprintln!(
        "dimer NVE: max |dE| = {:.3e} Eh, rel = {rel:.3e}",
        r.max_conserved_drift()
    );
    assert!(rel < 1e-3, "relative drift {rel:.3e}");
    assert_eq!(r.n_dof, 15, "COM removed: 3N - 3");
    // Internal forces only: the total momentum stays at its initial zero.
    let v = &r.final_state.velocities;
    let masses = [O_MASS, H_MASS, H_MASS, O_MASS, H_MASS, H_MASS];
    for c in 0..3 {
        let p: f64 = (0..6).map(|i| masses[i] * v[(i, c)]).sum();
        let scale: f64 = (0..6).map(|i| masses[i] * v[(i, c)].abs()).sum();
        assert!(
            p.abs() < 1e-12 * scale,
            "momentum component {c}: {p:e} (scale {scale:e})"
        );
    }
}

#[test]
fn langevin_dimer_samples_the_target_temperature() {
    let top = tip3p_flexible(2);
    let cfg = MdConfig {
        n_steps: 40_000,
        dt_fs: 0.25,
        thermostat: Thermostat::Langevin {
            friction_per_ps: 20.0,
        },
        temperature_k: Some(300.0),
        seed: 5,
        keep_trajectory: false,
        ..MdConfig::default()
    };
    let r = run_ffmd(
        &top,
        &system(dimer_coords()),
        &NonbondedMethod::NoCutoff,
        &cfg,
        &mut [],
    )
    .unwrap();
    let mean_t = r.mean_temperature(4000);
    eprintln!("Langevin dimer: <T> = {mean_t:.2} K (target 300 K)");
    assert!((mean_t - 300.0).abs() < 15.0, "<T> = {mean_t}");
}

#[test]
fn same_seed_is_bitwise_reproducible_and_a_new_seed_is_not() {
    let top = tip3p_flexible(2);
    let sys = system(dimer_coords());
    let cfg = |seed| MdConfig {
        n_steps: 200,
        dt_fs: 0.25,
        thermostat: Thermostat::Langevin {
            friction_per_ps: 5.0,
        },
        temperature_k: Some(300.0),
        seed,
        ..MdConfig::default()
    };
    let run = |seed| run_ffmd(&top, &sys, &NonbondedMethod::NoCutoff, &cfg(seed), &mut []).unwrap();
    let (a, b, c) = (run(42), run(42), run(43));
    assert_frames_bitwise_equal(&a, &b);
    let last = a.frames.len() - 1;
    assert_ne!(
        a.frames[last].potential.to_bits(),
        c.frames[last].potential.to_bits(),
        "a different seed must give a different trajectory"
    );
}
