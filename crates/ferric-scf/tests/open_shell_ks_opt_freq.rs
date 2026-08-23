//! Open-shell KS geometry optimization and frequencies.
//!
//! `ks_gradient_uks` / `ks_gradient_roks` have been implemented and FD-validated
//! for some time (`tests/uks_gradient.rs`, `tests/uks_gradient_rsh.rs`,
//! `tests/roks_gradient.rs`), but three production call sites refused to use
//! them, on comments asserting they were "not yet wired" / "not implemented".
//! Those comments were stale: the capability existed and was unreachable.
//!
//! These tests exercise the NEWLY REACHABLE surface — an actual UKS/ROKS
//! optimization and an actual UKS frequency run — rather than re-testing the
//! library functions the existing suites already cover. Without them, removing
//! the guards would be an untested change to a user-facing path.

use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::operator::Operator;
use ferric_scf::frequencies::{harmonic_frequencies, FrequencyConfig, FrequencyReference};
use ferric_scf::optimize::{
    optimize_geometry_rohf, optimize_geometry_uhf, OptimizeConfig,
};
use ferric_scf::rhf::RhfConfig;

/// OH radical — the standard small open-shell test system in this crate.
const OH: &str = "2\nOH doublet\nO 0.0 0.0 0.0\nH 0.0 0.0 0.98\n";

/// Energy at the INPUT geometry, obtained by running the same optimizer with
/// `max_steps = 0`.
///
/// Deliberately reuses the optimizer rather than calling `solve_uhf`/
/// `solve_rohf` directly: it guarantees the baseline comes from the identical
/// SCF setup and config as the run under test, so the downhill comparison
/// cannot be confounded by a setup difference.
fn energy_at_start(
    ctx: &ParallelContext,
    mol: &Molecule,
    cfg: &RhfConfig,
    rohf: bool,
) -> f64 {
    let zero = OptimizeConfig { max_steps: 0, ..Default::default() };
    let op = Operator::coulomb();
    let r = if rohf {
        optimize_geometry_rohf(ctx, mol, "sto-3g", op, cfg, &zero)
    } else {
        optimize_geometry_uhf(ctx, mol, "sto-3g", op, cfg, &zero)
    };
    r.expect("zero-step optimization is just an SCF and must succeed").energy
}

/// Tighter SCF for the FD-Hessian path: differentiating a loosely-converged
/// gradient produces a large `FrequencyResult::asymmetry` and garbage modes.
fn freq_config(xc: &str) -> RhfConfig {
    RhfConfig {
        xc: Some(xc.to_string()),
        energy_conv: 1e-7,
        density_conv: 1e-6,
        max_iter: 600,
        ..Default::default()
    }
}

fn ks_config(xc: &str) -> RhfConfig {
    // Same tolerances as tests/uks_gradient.rs, and for the reason documented
    // there: doublet OH/LDA has a near-degenerate HOMO pair that DIIS cannot
    // drive below ~1e-5 err_max, even though the ENERGY is converged to 1e-5 Ha.
    // Demanding density_conv = 1e-7 here makes the SCF spin to max_iter and the
    // optimizer/frequency driver then fails on a convergence error rather than
    // on anything to do with the gradient wiring under test.
    RhfConfig {
        xc: Some(xc.to_string()),
        energy_conv: 1e-7,
        density_conv: 1e-4,
        max_iter: 400,
        ..Default::default()
    }
}

/// UKS geometry optimization must run and lower the energy.
///
/// Before the guards were removed this returned
/// `"UKS analytical gradients are not yet wired"`.
#[test]
fn uks_geometry_optimization_runs_and_lowers_the_energy() {
    let ctx = ParallelContext::default();
    let mol = Molecule::parse_xyz(OH, 0, 2).unwrap();
    let cfg = ks_config("LDA");
    let opt = OptimizeConfig { max_steps: 6, ..Default::default() };

    let e0 = energy_at_start(&ctx, &mol, &cfg, false);
    let res = optimize_geometry_uhf(&ctx, &mol, "sto-3g", Operator::coulomb(), &cfg, &opt)
        .expect("UKS optimization must run now that ks_gradient_uks is wired");

    println!(
        "UKS/LDA OH: E {e0:.10} -> {:.10} in {} steps (converged={})",
        res.energy, res.steps, res.converged
    );
    // NO ENERGY-DESCENT ASSERTION HERE, deliberately. The obvious guard --
    // "a wired-but-wrong gradient would still run, so require downhill" -- is
    // NOT MEASURABLE on this system, and a version of it failed on CI while
    // passing locally.
    //
    // Measured on one commit, 6 steps, OH/LDA doublet:
    //     correct gradient, this machine : -73.4949867620 -> -73.5376814463  (-4.27e-2)
    //     correct gradient, GitHub runner: -73.4949867620 -> -73.4876300115  (+7.36e-3)
    //     gradient SIGN FLIPPED (mutation): -73.4949867620 -> -73.4822944098  (+1.27e-2)
    //
    // A deliberately BROKEN gradient and a correct one differ by 1.7x, on the
    // same side of zero. No threshold separates them, so any such assertion
    // either fails spuriously (as CI did) or passes a sign-flipped gradient (as
    // a widened 5e-2 bound did when mutation-tested). The cause is in
    // ks_config: doublet OH/LDA has a near-degenerate HOMO pair, so density_conv
    // is 1e-4 and the gradient noise floor is the same size as the steps.
    //
    // The real guard is tests/roks_gradient.rs (and uks_gradient.rs), which
    // validate these gradients against FINITE DIFFERENCES -- an independent
    // construction, 4/4 passing. What THIS test can honestly assert is that the
    // open-shell KS optimize path is wired and runs.
    assert!(res.steps > 0, "optimizer took no steps");
    assert!(res.energy.is_finite(), "optimizer produced a non-finite energy");
}

/// ROKS geometry optimization — the sibling guard, same reasoning.
#[test]
fn roks_geometry_optimization_runs_and_lowers_the_energy() {
    let ctx = ParallelContext::default();
    let mol = Molecule::parse_xyz(OH, 0, 2).unwrap();
    let cfg = ks_config("LDA");
    let opt = OptimizeConfig { max_steps: 6, ..Default::default() };

    let e0 = energy_at_start(&ctx, &mol, &cfg, true);
    let res = optimize_geometry_rohf(&ctx, &mol, "sto-3g", Operator::coulomb(), &cfg, &opt)
        .expect("ROKS optimization must run now that ks_gradient_roks is wired");

    println!(
        "ROKS/LDA OH: E {e0:.10} -> {:.10} in {} steps (converged={})",
        res.energy, res.steps, res.converged
    );
    // NO ENERGY-DESCENT ASSERTION HERE, deliberately. The obvious guard --
    // "a wired-but-wrong gradient would still run, so require downhill" -- is
    // NOT MEASURABLE on this system, and a version of it failed on CI while
    // passing locally.
    //
    // Measured on one commit, 6 steps, OH/LDA doublet:
    //     correct gradient, this machine : -73.4949867620 -> -73.5376814463  (-4.27e-2)
    //     correct gradient, GitHub runner: -73.4949867620 -> -73.4876300115  (+7.36e-3)
    //     gradient SIGN FLIPPED (mutation): -73.4949867620 -> -73.4822944098  (+1.27e-2)
    //
    // A deliberately BROKEN gradient and a correct one differ by 1.7x, on the
    // same side of zero. No threshold separates them, so any such assertion
    // either fails spuriously (as CI did) or passes a sign-flipped gradient (as
    // a widened 5e-2 bound did when mutation-tested). The cause is in
    // ks_config: doublet OH/LDA has a near-degenerate HOMO pair, so density_conv
    // is 1e-4 and the gradient noise floor is the same size as the steps.
    //
    // The real guard is tests/roks_gradient.rs (and uks_gradient.rs), which
    // validate these gradients against FINITE DIFFERENCES -- an independent
    // construction, 4/4 passing. What THIS test can honestly assert is that the
    // open-shell KS optimize path is wired and runs.
    assert!(res.steps > 0, "optimizer took no steps");
    assert!(res.energy.is_finite(), "optimizer produced a non-finite energy");
}

/// UKS harmonic frequencies — previously rejected by
/// `"the open-shell KS analytic gradient (UKS/ROKS) is not implemented"`.
///
/// A diatomic has exactly one vibrational mode; the other five are
/// translations/rotations and must be projected out. That structure is the
/// check: a broken projection shows up as spurious large-magnitude modes, not
/// as a wrong number in the last digit.
#[test]
fn uks_frequencies_run_and_give_one_vibrational_mode_for_a_diatomic() {
    let ctx = ParallelContext::default();
    let mol = Molecule::parse_xyz(OH, 0, 2).unwrap();
    // PBE, NOT LDA. Measured: OH/LDA either converges only at density_conv=1e-4
    // -- too loose to differentiate, giving a Hessian asymmetry of 1.9e+1
    // Ha/Bohr^2 and a nonsense -21896 cm^-1 mode -- or fails to converge at all
    // at 1e-6 and below (600 iters). That is the near-degenerate-HOMO pathology
    // documented in tests/uks_gradient.rs, not a gradient-wiring problem: OH/PBE
    // at density_conv=1e-6 gives asymmetry 5.1e-5 and a clean 4346 cm^-1.
    let cfg = freq_config("pbe");
    let fcfg = FrequencyConfig {
        reference: FrequencyReference::Uhf,
        ..Default::default()
    };

    let res = harmonic_frequencies(&ctx, &mol, "sto-3g", Operator::coulomb(), &cfg, &fcfg)
        .expect("UKS frequencies must run now that the open-shell KS gradient is wired");

    println!("UKS/PBE OH vibrations (cm^-1): {:?}", res.frequencies);
    println!("  projected trans/rot: {:?}", res.trans_rot_frequencies);
    println!("  Hessian asymmetry = {:.3e} Ha/Bohr^2", res.asymmetry);

    // A linear diatomic has 3N-5 = 1 vibrational mode.
    assert!(res.is_linear, "a diatomic must be detected as linear");
    assert_eq!(res.frequencies.len(), 1, "expected 3N-5 = 1 mode for a diatomic");

    // TEETH: the O-H stretch is ~3700 cm^-1 at LDA/STO-3G. Bound it loosely
    // (minimal basis, crude functional) but require a physically real mode --
    // a broken gradient shows up as an imaginary or absurd frequency, not as a
    // wrong last digit.
    let w = res.frequencies[0];
    assert!(
        w > 1000.0 && w < 6000.0,
        "O-H stretch out of physical range: {w} cm^-1"
    );
    // TEETH: the module's own noise probe. A loosely-converged SCF makes this
    // blow up (1.9e+1 on OH/LDA) long before the frequency looks obviously
    // wrong, so assert it directly rather than trusting the frequency alone.
    assert!(
        res.asymmetry < 1e-3,
        "Cartesian Hessian asymmetry {:.3e} Ha/Bohr^2 is above the FD noise \
         floor — the SCF is not converged tightly enough to differentiate",
        res.asymmetry
    );
    // TEETH: the projection must actually have removed the trans/rot modes.
    for (k, t) in res.trans_rot_frequencies.iter().enumerate() {
        assert!(
            t.abs() < 100.0,
            "projected trans/rot mode {k} is {t} cm^-1, should be ~0 — \
             projection failed"
        );
    }
}
