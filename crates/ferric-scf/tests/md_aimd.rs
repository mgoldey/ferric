//! Ab initio MD (`run_aimd`, `ScfForceProvider`) on small STO-3G systems.
//!
//! Anchor first: with zero steps the run is one single-point SCF energy and
//! analytic gradient, so step 0 must reproduce `solve_rhf` + `rhf_gradient`
//! at the same geometry, and a zero correction must be bitwise no
//! correction. The NVE tolerances below are measured drifts of the conserved
//! energy with headroom; they hold only because the SCF is converged tightly
//! (`density_conv = 1e-10`), since an unconverged density makes the force
//! inconsistent with the energy.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_md::{MdConfig, Thermostat, AMU_TO_ELECTRON_MASS};
use ferric_scf::gradient::rhf_gradient;
use ferric_scf::md::{run_aimd, AimdConfig, AimdResult, ScfCorrection};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::Spin;
use ndarray::Array2;

const BASIS: &str = "sto-3g";

/// H2 stretched to 0.80 A (equilibrium STO-3G ~0.71 A), so it vibrates.
fn h2() -> Molecule {
    Molecule::parse_xyz("2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.80\n", 0, 1).expect("H2")
}

fn water() -> Molecule {
    Molecule::parse_xyz(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.0 0.76 0.59\nH 0.0 -0.76 0.59\n",
        0,
        1,
    )
    .expect("water")
}

/// OH radical (doublet).
fn oh() -> Molecule {
    Molecule::parse_xyz("2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 1.00\n", 0, 2).expect("OH")
}

fn tight_scf() -> RhfConfig {
    RhfConfig::default()
        .with_density_conv(1e-10)
        .with_energy_conv(1e-12)
}

fn nve(n_steps: usize, dt_fs: f64, init_t: Option<f64>, seed: u64) -> AimdConfig {
    AimdConfig {
        md: MdConfig {
            n_steps,
            dt_fs,
            init_temperature_k: init_t,
            seed,
            ..MdConfig::default()
        },
        ..AimdConfig::default()
    }
}

fn run(mol: &Molecule, scf: &RhfConfig, spin: Spin, cfg: &AimdConfig) -> AimdResult {
    let ctx = ParallelContext::default();
    run_aimd(
        &ctx,
        mol,
        BASIS,
        Operator::coulomb(),
        scf,
        spin,
        cfg,
        None,
        &mut [],
    )
    .expect("AIMD run")
}

fn report(tag: &str, r: &AimdResult) -> f64 {
    let drift = r.md.max_conserved_drift();
    let e: Vec<f64> = r.md.frames.iter().map(|f| f.potential).collect();
    let span =
        e.iter().cloned().fold(f64::MIN, f64::max) - e.iter().cloned().fold(f64::MAX, f64::min);
    eprintln!(
        "{tag}: max |dE_cons| = {drift:.3e} Eh, E_pot span = {span:.3e} Eh, SCF iters {:?}",
        r.scf_iterations
    );
    eprintln!("{tag}: {:?}", r.warm_start_summary);
    drift
}

#[test]
fn md_mass_unit_is_the_frequency_code_mass_unit() {
    // `frequencies.rs` imports this very constant; pin its CODATA 2018 value.
    assert_eq!(AMU_TO_ELECTRON_MASS, 1.0 / 5.485_799_090_65e-4);
}

#[test]
fn zero_steps_is_the_single_point_energy_and_gradient() {
    let mol = h2();
    let scf = tight_scf();
    let r = run(&mol, &scf, Spin::Restricted, &nve(0, 0.2, None, 0));

    let ctx = ParallelContext::default();
    let op = Operator::coulomb();
    let bs = basis::bundled(BASIS).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let sp = solve_rhf(&ctx, &mol, &prep, op, &bounds, &scf).unwrap();
    let g = rhf_gradient(&mol, &prep, op, &bounds, &sp, None).unwrap();

    assert_eq!(r.md.frames.len(), 1);
    assert_eq!(r.md.n_force_evaluations, 1);
    assert_eq!(
        r.scf_iterations,
        vec![r.last_scf.iterations],
        "one cold SCF"
    );
    assert_eq!(r.md.label, "rhf/sto-3g");
    assert_eq!(
        r.md.frames[0].potential.to_bits(),
        r.last_scf.energy.to_bits()
    );
    let de = (r.md.frames[0].potential - sp.energy).abs();
    let dg = (&r.md.final_state.gradient - &g)
        .iter()
        .fold(0.0f64, |a, &x| a.max(x.abs()));
    eprintln!("step 0 vs single point: |dE| = {de:.3e}, max|dG| = {dg:.3e}");
    assert!(de < 1e-10, "|dE| = {de:e}");
    assert!(dg < 1e-8, "max|dG| = {dg:e}");
    assert!(
        g.iter().any(|x| x.abs() > 1e-3),
        "stretched H2 has a real force"
    );
    for (a, b) in r.mol.atoms.iter().zip(&mol.atoms) {
        assert_eq!((a.x, a.y, a.zpos), (b.x, b.y, b.zpos));
    }
}

#[test]
fn a_zero_correction_is_bitwise_no_correction_and_a_constant_shifts_only_energy() {
    let mol = h2();
    let scf = tight_scf();
    let cfg = nve(3, 0.2, Some(300.0), 1);
    let plain = run(&mol, &scf, Spin::Restricted, &cfg);
    let ctx = ParallelContext::default();
    let with = |c: &mut ScfCorrection<'_>| {
        run_aimd(
            &ctx,
            &mol,
            BASIS,
            Operator::coulomb(),
            &scf,
            Spin::Restricted,
            &cfg,
            Some(c),
            &mut [],
        )
    };
    let zero = with(&mut |_, _| Ok((0.0, None))).unwrap();
    let shift = 0.125;
    let shifted =
        with(&mut |m: &Molecule, _| Ok((shift, Some(Array2::zeros((m.atoms.len(), 3)))))).unwrap();
    for ((p, z), s) in plain
        .md
        .frames
        .iter()
        .zip(&zero.md.frames)
        .zip(&shifted.md.frames)
    {
        assert_eq!(p.potential.to_bits(), z.potential.to_bits());
        assert_eq!(p.kinetic.to_bits(), z.kinetic.to_bits());
        assert_eq!(
            p.kinetic.to_bits(),
            s.kinetic.to_bits(),
            "zero gradient: same path"
        );
        assert_eq!((p.potential + shift).to_bits(), s.potential.to_bits());
    }
    let err = with(&mut |_, _| Ok((-1e-3, None))).expect_err("energy without gradient");
    assert!(err.to_string().contains("gradient"), "{err}");
}

#[test]
fn h2_rhf_nve_conserves_energy_to_velocity_verlet_order() {
    // The conserved-energy excursion of velocity Verlet is O(dt^2); an SCF
    // that is not converged tightly enough adds an error that does not
    // shrink with dt. Halving dt over the same 4 fs must cut it ~4x.
    let r = run(
        &h2(),
        &tight_scf(),
        Spin::Restricted,
        &nve(20, 0.2, Some(300.0), 2),
    );
    let drift = report("H2 RHF NVE dt=0.2", &r);
    assert_eq!(r.md.frames.len(), 21);
    assert_eq!(r.scf_iterations.len(), 21);
    assert!(drift < 1.5e-4, "drift {drift:e}");
    // The returned molecule is the template moved to the final geometry.
    let x = &r.md.final_state.coords;
    assert_eq!(r.mol.atoms.len(), x.nrows());
    for (i, a) in r.mol.atoms.iter().enumerate() {
        assert_eq!([a.x, a.y, a.zpos], [x[(i, 0)], x[(i, 1)], x[(i, 2)]]);
    }
    let fine = run(
        &h2(),
        &tight_scf(),
        Spin::Restricted,
        &nve(40, 0.1, Some(300.0), 2),
    );
    let drift_fine = report("H2 RHF NVE dt=0.1", &fine);
    assert!(
        drift_fine < drift / 3.0,
        "dt/2 should cut the drift ~4x: {drift_fine:e} vs {drift:e}"
    );
}

#[test]
fn water_rhf_nve_conserves_energy() {
    let r = run(
        &water(),
        &tight_scf(),
        Spin::Restricted,
        &nve(10, 0.2, Some(300.0), 3),
    );
    let drift = report("water RHF NVE", &r);
    assert!(drift < 2.5e-5, "drift {drift:e}");
}

#[test]
fn warm_start_agrees_with_cold_and_saves_scf_iterations() {
    let mol = h2();
    let mut cfg = nve(10, 0.2, Some(300.0), 4);
    let warm = run(&mol, &tight_scf(), Spin::Restricted, &cfg);
    cfg.scf_warm_start = false;
    let cold = run(&mol, &tight_scf(), Spin::Restricted, &cfg);
    let max_de = warm
        .md
        .frames
        .iter()
        .zip(&cold.md.frames)
        .map(|(a, b)| (a.potential - b.potential).abs())
        .fold(0.0f64, f64::max);
    let (wi, ci): (usize, usize) = (
        warm.scf_iterations[1..].iter().sum(),
        cold.scf_iterations[1..].iter().sum(),
    );
    eprintln!(
        "warm vs cold: max |dE_pot| = {max_de:.3e}; SCF iterations after step 0: {wi} vs {ci}"
    );
    assert!(max_de < 1e-9, "{max_de:e}");
    assert!(wi < ci, "warm {wi} vs cold {ci}");
    assert!(warm.warm_start_summary.is_some());
    assert!(cold.warm_start_summary.is_none());
}

#[test]
fn oh_radical_uhf_nve_conserves_energy() {
    let r = run(
        &oh(),
        &tight_scf(),
        Spin::Unrestricted,
        &nve(10, 0.2, Some(300.0), 5),
    );
    let drift = report("OH UHF NVE", &r);
    assert_eq!(r.md.label, "uhf/sto-3g");
    assert!(r.last_scf.spin == Spin::Unrestricted);
    assert!(drift < 2.5e-5, "drift {drift:e}");
}

#[test]
fn h2_rks_lda_nve_conserves_energy() {
    let scf = tight_scf().with_xc("LDA".to_string());
    let r = run(&h2(), &scf, Spin::Restricted, &nve(10, 0.2, Some(300.0), 6));
    let drift = report("H2 RKS/LDA NVE", &r);
    assert_eq!(r.md.label, "rks:LDA/sto-3g");
    assert!(drift < 6e-5, "drift {drift:e}");
}

#[test]
fn same_seed_is_bitwise_reproducible_and_a_new_seed_is_not() {
    let cfg = |seed| AimdConfig {
        md: MdConfig {
            n_steps: 4,
            dt_fs: 0.2,
            thermostat: Thermostat::Langevin {
                friction_per_ps: 10.0,
            },
            temperature_k: Some(300.0),
            seed,
            ..MdConfig::default()
        },
        ..AimdConfig::default()
    };
    let go = |seed| run(&h2(), &tight_scf(), Spin::Restricted, &cfg(seed));
    let (a, b, c) = (go(9), go(9), go(10));
    for (fa, fb) in a.md.frames.iter().zip(&b.md.frames) {
        assert_eq!(fa.potential.to_bits(), fb.potential.to_bits());
        assert_eq!(fa.kinetic.to_bits(), fb.kinetic.to_bits());
    }
    assert_ne!(
        a.md.frames[4].kinetic.to_bits(),
        c.md.frames[4].kinetic.to_bits(),
        "a different seed must give a different trajectory"
    );
}
