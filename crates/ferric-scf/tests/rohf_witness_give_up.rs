//! **The F6 swap witness's give-up exit returns ONE state.**
//!
//! When the witness has forced more restarts than its budget, `solve_rohf`
//! stops. At that moment the loop's own variables are a mixture: `c`, `D_α`,
//! `D_β` hold the swapped PROBE determinant, `f_eff_last` the Fock of the last
//! full iteration, and the monitor's previous energy belongs to the iteration
//! BEFORE the last convergence. The first version of the guard fell through to
//! the generic non-converged exit, which assembled those into one
//! `ScfResult` — densities of one determinant, MOs of another, an energy of
//! neither, and `exit = MaxIter` though no limit was reached (CodeRabbit on
//! PR #172). The exit now returns the held converged state, marked
//! `converged = false`, `exit = NotCertified`.
//!
//! This file is its own test binary ON PURPOSE: it lowers the process-global
//! `ROHF_MAX_AUFBAU_RESTARTS_OVERRIDE` to 0, which would change the behaviour
//! of any other ROHF test running concurrently in the same binary.
//!
//! The system: OH/6-31G from the hcore guess. It first converges to the σ-hole
//! state at −75.2037249529 (4.3 eV high); the witness finds a one-electron move
//! 0.154 Ha lower; with a budget of 0 that first restart already exceeds it.

use std::sync::atomic::Ordering;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::result::ScfExit;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::rohf::{solve_rohf, solve_rohf_best_effort, ROHF_MAX_AUFBAU_RESTARTS_OVERRIDE};
use ferric_scf::screening::SchwarzBounds;
use ndarray::{s, Array2};

/// The σ-hole OH/6-31G ROHF stationary point the hcore guess converges to
/// (pinned in `rohf_state_selection.rs`).
const OH_631G_EXCITED: f64 = -75.203_724_952_9;

fn cfg(guard: bool) -> RhfConfig {
    RhfConfig {
        max_iter: 400,
        density_conv: 1e-10,
        energy_conv: 1e-11,
        use_sad_guess: false,
        rohf_occupation_guard: guard,
        ..Default::default()
    }
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

#[test]
fn give_up_exit_returns_one_self_consistent_state() {
    let mol = Molecule::parse_xyz("2\nOH\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n", 0, 2).unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("6-31g").unwrap()).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let (nocc_double, nocc_open) = (4usize, 1usize);

    ROHF_MAX_AUFBAU_RESTARTS_OVERRIDE.store(0, Ordering::Relaxed);
    let r = solve_rohf_best_effort(&ctx, &mol, &prep, op, &bounds, &cfg(true))
        .expect("best-effort ROHF must return a state");
    let strict = solve_rohf(&ctx, &mol, &prep, op, &bounds, &cfg(true));
    ROHF_MAX_AUFBAU_RESTARTS_OVERRIDE.store(usize::MAX, Ordering::Relaxed);

    println!(
        "give-up exit: converged={} exit={:?} iters={} E={:.10}",
        r.converged, r.exit, r.iterations, r.energy
    );

    // 1. The give-up path was actually reached (reachability: with the
    //    default budget this system converges to the ground state instead).
    assert!(!r.converged, "the give-up exit must report NOT converged");
    assert_eq!(r.exit, ScfExit::NotCertified, "wrong exit reason");

    // 2. The energy is the held state's, not a stale monitor value.
    assert!(
        (r.energy - OH_631G_EXCITED).abs() < 1e-8,
        "returned energy {:.10} is not the held state's {OH_631G_EXCITED:.10}",
        r.energy
    );
    // ...and the strict entry point reports that same energy.
    match strict {
        Err(ferric_core::FerricError::ScfConvergence { last_energy, .. }) => assert!(
            (last_energy - r.energy).abs() < 1e-12,
            "solve_rohf's last_energy {last_energy:.10} != best-effort energy {:.10}",
            r.energy
        ),
        other => panic!("solve_rohf should return ScfConvergence here, got {other:?}"),
    }

    // 3. The MOs reproduce the densities (same determinant).
    let c = &r.mos_alpha;
    let cd = c.slice(s![.., ..nocc_double]);
    let co = c.slice(s![.., nocc_double..nocc_double + nocc_open]);
    let d_b = cd.dot(&cd.t());
    let d_a = &d_b + &co.dot(&co.t());
    let d_b_ret = r.density_beta.as_ref().expect("ROHF has a beta density");
    assert!(
        max_abs(&(&d_a - &r.density_alpha)) < 1e-10,
        "MOs do not reproduce density_alpha"
    );
    assert!(
        max_abs(&(&d_b - d_b_ret)) < 1e-10,
        "MOs do not reproduce density_beta"
    );
    assert!(
        max_abs(&(&(&d_a + &d_b) - &r.density_total)) < 1e-10,
        "density_total != alpha + beta"
    );

    // 4. The eigenvalues belong to these MOs and the returned Fock.
    let f_mo = c.t().dot(&r.fock_alpha).dot(c);
    for (k, &e) in r.eps_alpha.iter().enumerate() {
        assert!(
            (f_mo[(k, k)] - e).abs() < 1e-8,
            "eps[{k}] = {e} but C^T F C [{k},{k}] = {}",
            f_mo[(k, k)]
        );
    }

    // 5. Independent recompute: start the pre-F6 solver (guard off) FROM the
    //    returned density. It is a converged ROHF stationary point, so the
    //    solver must stay there and reproduce the returned energy and density.
    let back = solve_rohf(
        &ctx,
        &mol,
        &prep,
        op,
        &bounds,
        &RhfConfig {
            init_guess_density: Some(r.density_total.clone()),
            ..cfg(false)
        },
    )
    .expect("guard-off solve from the returned density");
    println!(
        "recomputed from the returned density: E={:.10} ({} iters)",
        back.energy, back.iterations
    );
    assert!(
        (back.energy - r.energy).abs() < 1e-8,
        "the returned densities describe a state at {:.10}, not the returned {:.10}",
        back.energy,
        r.energy
    );
    assert!(
        max_abs(&(&back.density_total - &r.density_total)) < 1e-5,
        "the returned density is not the converged density of its own state"
    );
}
