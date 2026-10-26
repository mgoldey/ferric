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
use ferric_scf::optimize::{optimize_geometry_rohf, optimize_geometry_uhf, OptimizeConfig};
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
fn energy_at_start(ctx: &ParallelContext, mol: &Molecule, cfg: &RhfConfig, rohf: bool) -> f64 {
    let zero = OptimizeConfig {
        max_steps: 0,
        ..Default::default()
    };
    let op = Operator::coulomb();
    let r = if rohf {
        optimize_geometry_rohf(ctx, mol, "sto-3g", op, cfg, &zero)
    } else {
        optimize_geometry_uhf(ctx, mol, "sto-3g", op, cfg, &zero)
    };
    r.expect("zero-step optimization is just an SCF and must succeed")
        .energy
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
    // Same tolerances as tests/uks_gradient.rs. Doublet OH/LDA has a degenerate
    // π hole: at the stationary ground state the β hole flips π_x <-> π_y every
    // iteration, so ΔP_max = 1 and the ΔP gate alone never passes. UHF accepts
    // that swap once the orbital gradient is below density_conv
    // (tests/uhf_degenerate_swap.rs); ROHF holds the occupation by continuity
    // (rohf_occupation, F6).
    RhfConfig {
        xc: Some(xc.to_string()),
        energy_conv: 1e-7,
        density_conv: 1e-4,
        max_iter: 400,
        ..Default::default()
    }
}

/// Measured descent of a 2-step optimization from the input geometry (see the
/// tests); a correct gradient lowers E by ~5.2e-3 Ha. Half of that separates it
/// from a stalled or uphill run with margin on both sides.
const MIN_DESCENT: f64 = 2.5e-3;

/// UKS geometry optimization must run and lower the energy.
///
/// Before the guards were removed this returned
/// `"UKS analytical gradients are not yet wired"`.
#[test]
fn uks_geometry_optimization_runs_and_lowers_the_energy() {
    let ctx = ParallelContext::default();
    let mol = Molecule::parse_xyz(OH, 0, 2).unwrap();
    let cfg = ks_config("LDA");
    let opt = OptimizeConfig {
        max_steps: 6,
        ..Default::default()
    };

    let e0 = energy_at_start(&ctx, &mol, &cfg, false);
    let res = optimize_geometry_uhf(&ctx, &mol, "sto-3g", Operator::coulomb(), &cfg, &opt)
        .expect("UKS optimization must run now that ks_gradient_uks is wired");

    println!(
        "UKS/LDA OH: E {e0:.10} -> {:.10} in {} steps (converged={})",
        res.energy, res.steps, res.converged
    );
    // MEASURED (fix/uhf-convergence-gradient-guard): -74.0755553242 ->
    // -74.0807743024 in 2 steps, converged. The numbers this comment used to
    // carry (-73.4949867620 -> -73.5376814463 / -73.4876300115, and the
    // "unmeasurable" verdict) came from SCFs that the ΔP gate had accepted on a
    // stagnated non-stationary state ~0.58 Ha above the ground state; the
    // gradient at such a state is not the derivative of anything, so descent
    // was noise. With the UHF orbital-gradient guard the SCF lands on the
    // stationary state and both checks below are real.
    //
    // Which check catches what (mutation-tested): with the gradient SIGN
    // FLIPPED the optimizer still descends (-74.0807743087, its energy-based
    // step control finds the minimum) but too slowly to converge in 6 steps
    // (the correct gradient takes 2), so the sign flip is caught by
    // convergence speed through `converged`; the descent bound catches an
    // SCF that lands on the wrong state (a 0.58 Ha stagnated state is far
    // outside it). FD validation of the gradient itself stays in
    // tests/uks_gradient.rs.
    assert!(res.steps > 0, "optimizer took no steps");
    assert!(
        res.converged,
        "optimizer did not converge in {} steps",
        res.steps
    );
    assert!(
        res.energy < e0 - MIN_DESCENT,
        "optimizer did not descend: {e0:.10} -> {:.10}",
        res.energy
    );
}

/// ROKS geometry optimization — the sibling guard, same reasoning.
#[test]
fn roks_geometry_optimization_runs_and_lowers_the_energy() {
    let ctx = ParallelContext::default();
    let mol = Molecule::parse_xyz(OH, 0, 2).unwrap();
    let cfg = ks_config("LDA");
    let opt = OptimizeConfig {
        max_steps: 6,
        ..Default::default()
    };

    let e0 = energy_at_start(&ctx, &mol, &cfg, true);
    let res = optimize_geometry_rohf(&ctx, &mol, "sto-3g", Operator::coulomb(), &cfg, &opt)
        .expect("ROKS optimization must run now that ks_gradient_roks is wired");

    println!(
        "ROKS/LDA OH: E {e0:.10} -> {:.10} in {} steps (converged={})",
        res.energy, res.steps, res.converged
    );
    // MEASURED (fix/uhf-convergence-gradient-guard): -74.0752223408 ->
    // -74.0803874578 in 2 steps, converged. ROHF already carried the F6
    // gradient guard; the stale numbers this comment used to carry were the
    // UKS ones (see the sibling test). Gradient sign-flip mutant: descends to
    // -74.0803874586 but too slowly to converge in 6 steps (caught by
    // convergence speed, through `converged`).
    assert!(res.steps > 0, "optimizer took no steps");
    assert!(
        res.converged,
        "optimizer did not converge in {} steps",
        res.steps
    );
    assert!(
        res.energy < e0 - MIN_DESCENT,
        "optimizer did not descend: {e0:.10} -> {:.10}",
        res.energy
    );
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
    // PBE. The old reason for avoiding LDA here ("OH/LDA converges only at
    // density_conv=1e-4 and gives a nonsense -21896 cm^-1 mode, or never
    // converges at 1e-6") was the UHF gate accepting a stagnated
    // non-stationary state, fixed by the orbital-gradient guard
    // (tests/uhf_degenerate_swap.rs). MEASURED after the fix: OH/LDA gives
    // 4357.29 cm^-1, asymmetry 5.3e-6, at density_conv=1e-6 (4.3e-6 at 1e-8).
    // OH/PBE at 1e-6: asymmetry 5.1e-5 and a clean 4346 cm^-1.
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
    assert_eq!(
        res.frequencies.len(),
        1,
        "expected 3N-5 = 1 mode for a diatomic"
    );

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
