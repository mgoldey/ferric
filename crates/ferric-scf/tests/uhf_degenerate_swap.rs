//! UHF/UKS convergence on a degenerate open shell: the orbital-gradient guard.
//!
//! OH doublet, UKS-LDA/STO-3G, the `ks_config` of `open_shell_ks_opt_freq.rs`.
//! The β hole sits in a degenerate π pair. MEASURED before the guard (origin/main
//! b971808fd, per-iteration probe):
//! - the true ground state, −74.0755553242, is reached and is stationary
//!   (`err_max` ~2e-9), but the β hole flips π_x ↔ π_y every iteration, so
//!   ΔP_max = 1 and the ΔP/ΔE gate never passes;
//! - DIIS then stagnates (max |c| 63.7) and jumps to a non-aufbau state at
//!   −73.4899520144 with `err_max` 0.136; the next extrapolation gives it ~0
//!   weight, D freezes, and the ΔP gate ACCEPTED it, 0.586 Ha high.
//!
//! The guard (`uhf::uhf_exit_decision`) accepts the swap at the stationary
//! point, vetoes the stagnated state, and returns MOs whose aufbau density is
//! the returned density.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::{s, Array2};

const OH: &str = "2\nOH doublet\nO 0.0 0.0 0.0\nH 0.0 0.0 0.98\n";

/// The stationary UKS-LDA/STO-3G ground state of OH at this geometry.
const E_GROUND: f64 = -74.0755553242;
/// The stagnated non-aufbau state the unguarded gate accepted.
const E_STAGNATED: f64 = -73.4899520144;

fn aufbau_density(c: &Array2<f64>, nocc: usize) -> Array2<f64> {
    let occ = c.slice(s![.., ..nocc]);
    occ.dot(&occ.t())
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

#[test]
fn oh_uks_lda_converges_to_the_stationary_ground_state_not_a_stagnated_one() {
    let ctx = ParallelContext::default();
    let mol = Molecule::parse_xyz(OH, 0, 2).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let cfg = RhfConfig {
        xc: Some("LDA".to_string()),
        energy_conv: 1e-7,
        density_conv: 1e-4,
        max_iter: 400,
        ..Default::default()
    };
    let r = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).expect("OH UKS-LDA must solve");
    let err_max = r.convergence.as_ref().map(|c| c.err_max).unwrap();
    println!(
        "OH UKS-LDA: E {:.10} in {} iters (converged={}, err_max={err_max:.3e})",
        r.energy, r.iterations, r.converged
    );

    assert!(
        r.converged,
        "did not converge in {} iterations",
        r.iterations
    );
    assert!(
        (r.energy - E_STAGNATED).abs() > 1e-3,
        "accepted the stagnated non-stationary state {:.10}",
        r.energy
    );
    assert!(
        (r.energy - E_GROUND).abs() <= 1e-8,
        "E {:.10} vs stationary ground state {E_GROUND:.10}",
        r.energy
    );
    assert!(
        err_max < cfg.density_conv,
        "orbital gradient {err_max:.3e} at the accepted state"
    );

    // The returned MOs must build the returned density (aufbau of C).
    let (nocc_a, nocc_b) = (5, 4);
    let da = max_abs_diff(&aufbau_density(&r.mos_alpha, nocc_a), &r.density_alpha);
    let db = max_abs_diff(
        &aufbau_density(r.mos_beta.as_ref().unwrap(), nocc_b),
        r.density_beta.as_ref().unwrap(),
    );
    assert!(
        da < 1e-5 && db < 1e-5,
        "MOs inconsistent with the density: |D(C_occ) - D| alpha {da:.2e}, beta {db:.2e}"
    );
    let eps_b = r.eps_beta.as_ref().unwrap();
    assert!(
        eps_b[..nocc_b].windows(2).all(|w| w[0] <= w[1])
            && eps_b[nocc_b..].windows(2).all(|w| w[0] <= w[1]),
        "occupied and virtual blocks must each stay in ascending order: {eps_b:?}"
    );

    // Pinned: the guarded trajectory is deterministic.
    assert_eq!(r.iterations, ITERATIONS, "iteration count moved");
}

/// MEASURED on this branch (see the PR body for the per-iteration table).
const ITERATIONS: usize = 6;
