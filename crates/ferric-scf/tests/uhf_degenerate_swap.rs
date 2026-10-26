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
//! The guard (`uhf::UhfExitGate`) accepts the swap at the stationary point,
//! vetoes the stagnated state, and — only after a swap — returns MOs whose
//! aufbau density is the returned density. A MOM state keeps main's C and eps.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::hessian::uhf_hessian;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::ScfResult;
use ndarray::{s, Array2};

const OH: &str = "2\nOH doublet\nO 0.0 0.0 0.0\nH 0.0 0.0 0.98\n";

/// The stationary UKS-LDA/STO-3G ground state of OH at this geometry.
const E_GROUND: f64 = -74.0755553242;
/// The stagnated non-aufbau state the unguarded gate accepted.
const E_STAGNATED: f64 = -73.4899520144;
/// α/β occupied counts of the OH doublet.
const NOCC: (usize, usize) = (5, 4);

struct Oh {
    mol: Molecule,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
}

fn oh() -> Oh {
    let mol = Molecule::parse_xyz(OH, 0, 2).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    Oh { mol, prep, bounds }
}

fn lda(density_conv: f64, mom_after_iter: usize) -> RhfConfig {
    RhfConfig {
        xc: Some("LDA".to_string()),
        energy_conv: 1e-7,
        density_conv,
        max_iter: 400,
        mom_after_iter,
        ..Default::default()
    }
}

fn solve(sys: &Oh, cfg: &RhfConfig) -> ScfResult {
    solve_uhf(
        &ParallelContext::default(),
        &sys.mol,
        &sys.prep,
        &sys.bounds,
        cfg,
    )
    .expect("OH UKS-LDA must solve")
}

/// max |C_occ C_occᵀ − D| per spin: how far the returned MOs are from building
/// the returned density.
fn aufbau_deviation(r: &ScfResult) -> (f64, f64) {
    let dev = |c: &Array2<f64>, d: &Array2<f64>, n: usize| {
        let occ = c.slice(s![.., ..n]);
        (occ.dot(&occ.t()) - d)
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()))
    };
    (
        dev(&r.mos_alpha, &r.density_alpha, NOCC.0),
        dev(
            r.mos_beta.as_ref().unwrap(),
            r.density_beta.as_ref().unwrap(),
            NOCC.1,
        ),
    )
}

fn ascending(e: &[f64]) -> bool {
    e.windows(2).all(|w| w[0] <= w[1])
}

/// Why the analytic UHF Hessian refuses this result's STATE.
///
/// The Hessian refuses Kohn-Sham and a MOM config outright (config checks that
/// run first), so the result is handed over with a plain HF config: what is
/// exercised is `uhf_state_refusal`, the backstop that inspects the RESULT —
/// the only guard when a MOM or reordered state arrives with a config that
/// does not say so (the ferric-cli MOM references, free-atom proatoms).
fn hessian_state_refusal(sys: &Oh, r: &ScfResult, cfg: &RhfConfig) -> String {
    let hf = RhfConfig {
        xc: None,
        mom_after_iter: 0,
        ..cfg.clone()
    };
    uhf_hessian(
        &ParallelContext::default(),
        &sys.mol,
        &sys.prep,
        Operator::coulomb(),
        &sys.bounds,
        r,
        &hf,
    )
    .expect_err("a non-aufbau UHF state must be refused")
    .to_string()
}

#[test]
fn oh_uks_lda_converges_to_the_stationary_ground_state_not_a_stagnated_one() {
    let sys = oh();
    let cfg = lda(1e-4, 0);
    let r = solve(&sys, &cfg);
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

    // Accepted as a degenerate swap, so the returned MOs are reordered to
    // build the returned density (aufbau of C reproduces D) ...
    let (da, db) = aufbau_deviation(&r);
    assert!(
        da < 1e-5 && db < 1e-5,
        "MOs inconsistent with the density: |D(C_occ) - D| alpha {da:.2e}, beta {db:.2e}"
    );
    let eps_b = r.eps_beta.as_ref().unwrap();
    assert!(
        ascending(&eps_b[..NOCC.1]) && ascending(&eps_b[NOCC.1..]),
        "occupied and virtual blocks must each stay in ascending order: {eps_b:?}"
    );
    // ... which leaves an occupied β eigenvalue above a virtual one: the
    // Hessian's state check must refuse it (its response assumes aufbau),
    // although C now builds D.
    let why = hessian_state_refusal(&sys, &r, &cfg);
    assert!(
        why.contains("aufbau"),
        "refused for the wrong reason: {why}"
    );

    // Pinned: the guarded trajectory is deterministic.
    assert_eq!(r.iterations, ITERATIONS, "iteration count moved");
}

/// MEASURED on this branch (see the PR body for the per-iteration table).
const ITERATIONS: usize = 6;

/// A tight `density_conv` must still converge on a degenerate open shell.
///
/// CI failure this pins (PR #556, first push): at `density_conv` 1e-9 the swap
/// acceptance demanded a gradient below 1e-9, but the stationary state's
/// gradient floors on the arithmetic at 9.1e-10 (OpenBLAS Nehalem), 1.1e-9
/// (Prescott, the CI kernels) or 2.2e-9 (native FMA). On Prescott the run
/// never accepted, cycled ground ↔ stagnated state (the veto refusing the
/// latter 7 times) and returned `ScfConvergence` after 400 iterations at the
/// right energy. The swap bound is now `max(density_conv, 1e-6)`.
///
/// Arithmetic-robust by construction: no exact iteration count (the
/// trajectory differs by kernel), only a bound well below the failure.
/// The veto + DIIS-reset wiring is pinned arithmetic-free in
/// `uhf::tests::exit_gate_vetoes_a_settled_nonstationary_state_and_clears_diis`.
#[test]
fn oh_uks_lda_tight_density_conv_converges_on_the_ground_state() {
    let sys = oh();
    let r = solve(&sys, &lda(1e-9, 0));
    let v = r.convergence.and_then(|c| c.gradient_vetoes);
    println!(
        "OH UKS-LDA dconv=1e-9: E {:.10} in {} iters, vetoes {v:?}",
        r.energy, r.iterations
    );
    assert!(r.converged);
    assert!(
        (r.energy - E_GROUND).abs() <= 1e-8,
        "E {:.10} vs stationary ground state {E_GROUND:.10}",
        r.energy
    );
    assert!(
        r.iterations <= 30,
        "{} iterations: the swap was not accepted at the gradient floor",
        r.iterations
    );
}

/// A MOM state is non-aufbau by design: its returned C and eps must be main's
/// (aufbau eigenpairs of F, eps ascending), NOT reordered to build D. The
/// exit reorder is reserved for a degenerate-swap acceptance. MOM consumers
/// (ferric-cli pdep-rpa/gw/mp2-v open-shell references, free-atom proatoms)
/// and the Hessian's MOM refusal rely on it.
#[test]
fn mom_state_keeps_aufbau_eigenpairs_and_the_hessian_refuses_it() {
    let sys = oh();
    let cfg = lda(1e-6, 3);
    let r = solve(&sys, &cfg);
    let (da, db) = aufbau_deviation(&r);
    println!(
        "OH UKS-LDA MOM: E {:.10} in {} iters, |D(C_occ)-D| alpha {da:.2e} beta {db:.2e}",
        r.energy, r.iterations
    );
    assert!(r.converged);
    // MOM holds the hole the aufbau eigenvectors do not, so the returned
    // (aufbau) β MOs must NOT build D. A small deviation means either the exit
    // reordered the MOM state (the defect) or MOM stayed aufbau (vacuous).
    assert!(
        db > 0.5,
        "β MOs build D ({db:.2e}): MOM exit reordered, or MOM state aufbau (vacuous)"
    );
    assert!(
        ascending(&r.eps_alpha) && ascending(r.eps_beta.as_ref().unwrap()),
        "MOM exit eigenvalues were reordered"
    );
    let why = hessian_state_refusal(&sys, &r, &cfg);
    assert!(
        why.contains("aufbau"),
        "refused for the wrong reason: {why}"
    );
}
