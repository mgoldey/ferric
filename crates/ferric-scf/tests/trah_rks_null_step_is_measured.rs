//! RHF and RKS: a declined TRAH null step must not END the run by itself.
//!
//! When the closed-shell loop's `predicted_min` guard declines a TRAH step,
//! the density does not move. If the loop then records a zero density change
//! and skips to the next iteration, that iteration rebuilds the Fock matrix at
//! the SAME density, sees ΔE = 0 and ΔP = 0, and passes the convergence test
//! whatever the orbital gradient is: convergence is declared, not measured.
//! At the default `predicted_min` (1e-12 Ha) the declined step is genuinely
//! negligible, which hid the defect; with the bound raised to 1e-6 the run
//! reported `converged = true` 3.0e-7 Ha (RKS) above the DIIS energy (#315).
//!
//! This test raises `predicted_min` to 1e-6 so that the ONE step TRAH would
//! take (|predicted| 3.0e-7 RKS, 4.4e-7 RHF) is declined, and asserts that
//! the run still reaches the DIIS energy. It lives in its own binary, as ONE
//! test, so the process-wide `TRAH_NULL_STEPS_DECLINED` delta is this run's
//! alone: the assertion that the guard's branch was TAKEN is what separates
//! "convergence measured after a decline" from "the decline never happened".

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::trah::{TrahConfig, TRAH_NULL_STEPS_DECLINED};
use std::sync::atomic::Ordering;

/// Energy bar against DIIS, placed between the two measured sides: with the
/// defect the RHF run reports convergence +4.437e-7 Ha above DIIS (7
/// iterations); measured, RHF and RKS/PBE both land on the DIIS energy to the
/// 12 printed decimals (|ΔE| < 1e-12).
const TOL_E: f64 = 1e-8;

fn water() -> Molecule {
    Molecule::parse_xyz(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.0 0.757 0.587\nH 0.0 -0.757 0.587\n",
        0,
        1,
    )
    .unwrap()
}

fn assert_decline_does_not_declare_convergence(what: &str, cfg_diis: RhfConfig) {
    let mol = water();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let r_diis = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_diis).unwrap();

    let cfg_trah = RhfConfig {
        trah_trigger: Some(1e-3),
        trah: TrahConfig {
            predicted_min: 1e-6,
            ..TrahConfig::default()
        },
        ..cfg_diis.clone()
    };
    let d0 = TRAH_NULL_STEPS_DECLINED.load(Ordering::Relaxed);
    let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_trah).unwrap();
    let declined = TRAH_NULL_STEPS_DECLINED.load(Ordering::Relaxed) - d0;
    let de = r.energy - r_diis.energy;
    eprintln!(
        "{what}  DIIS: E={:.12} iters={}   TRAH(predicted_min=1e-6): E={:.12} \
         iters={} converged={} declined={declined}  E-E_DIIS={de:+.3e}",
        r_diis.energy, r_diis.iterations, r.energy, r.iterations, r.converged
    );

    assert!(r_diis.converged, "{what}: the DIIS reference must converge");
    assert!(
        declined >= 1,
        "{what}: the null-step guard never declined a step, so this run does \
         not exercise the decline path"
    );
    assert!(r.converged, "{what}: TRAH must converge");
    assert!(
        de.abs() < TOL_E,
        "{what}: reported converged {de:+.3e} Ha from the DIIS energy: a \
         declined null step declared convergence instead of measuring it"
    );
}

/// ONE test, two sequential cases, so the counter delta belongs to each case
/// alone (see the module doc).
#[test]
fn declined_null_step_does_not_declare_convergence() {
    assert_decline_does_not_declare_convergence(
        "RHF/H2O/cc-pVDZ",
        RhfConfig {
            energy_conv: 1e-10,
            density_conv: 1e-8,
            max_iter: 200,
            level_shift: 0.2,
            ..Default::default()
        },
    );
    assert_decline_does_not_declare_convergence(
        "RKS/PBE/H2O/cc-pVDZ",
        RhfConfig {
            xc: Some("PBE".into()),
            energy_conv: 1e-10,
            density_conv: 1e-8,
            max_iter: 200,
            level_shift: 0.0,
            ..Default::default()
        },
    );
}
