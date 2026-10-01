//! TRAH in the trust-region **hard case**: a symmetry-breaking negative mode
//! the gradient cannot see.
//!
//! N2 / STO-3G at R = 1.6 Angstrom, RHF. The orbital Hessian has a negative
//! eigenvalue (lambda_min = -0.21 at the DIIS endpoint, the same order at
//! the MINAO guess) whose eigenvector lies in a different irrep from the
//! gradient, which stays totally symmetric because the density does. So
//! g . v_neg = 0 EXACTLY (measured 5e-17 relative at the guess, by symmetry,
//! not because g is small: |g| = 3e-2 there).
//!
//! In that case the lowest eigenvector of the augmented Hessian
//! [[0, a g^T], [a g, H]] is (0, v_neg): its eigenvalue is lambda_min(H)
//! itself and its leading entry is zero, so the AH ansatz
//! kappa = (lower block)/(leading entry) has nothing to divide by. This is
//! the textbook trust-region hard case (More & Sorensen 1983).
//!
//! References (PySCF 2.13, same geometry/basis, via
//! scripts/scf_optimizer_spike): DIIS converges to a SADDLE at
//! E = -107.1848464608 (PySCF `stability()` says internally unstable); the
//! stable RHF state reached by following v_neg is E = -107.2256692254,
//! 40.8 mHa lower.
//!
//! That state breaks the molecule's cylindrical symmetry, so rotating it
//! about the bond axis costs nothing: its orbital Hessian has exactly ONE
//! zero mode (PySCF Hessian there: -2.0e-6, then +0.206, +0.272, +0.272).
//! ferric's stability check therefore reports MARGINAL (lambda_min ~ 1e-7,
//! under its 1e-6 noise floor), which is the physically correct verdict --
//! the test asserts "not a saddle", not "Stable".

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::stability::StabilityVerdict;

/// The TRAH counters are process-wide; every test in this file holds this so
/// a counter delta taken around one solve sees only that solve.
static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

const E_SADDLE: f64 = -107.184_846_460_8;
const E_STABLE: f64 = -107.225_669_225_4;

fn n2_stretched() -> Molecule {
    Molecule::parse_xyz("2\nn2\nN 0 0 0\nN 0 0 1.6\n", 0, 1).unwrap()
}

fn run(trah: Option<f64>) -> Result<ferric_scf::result::ScfResult, ferric_core::FerricError> {
    let mol = n2_stretched();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        check_stability: true,
        trah_trigger: trah,
        ..Default::default()
    };
    solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg)
}

/// Negative control: DIIS lands on the saddle and the stability check says so.
/// If this ever changes, the system no longer exercises the hard case and the
/// TRAH test below proves nothing.
#[test]
fn diis_lands_on_the_symmetric_saddle() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let r = run(None).expect("DIIS converges");
    assert!(r.converged);
    assert!(
        (r.energy - E_SADDLE).abs() < 1e-7,
        "DIIS E = {:.10}, expected the saddle {E_SADDLE:.10}",
        r.energy
    );
    let st = r.stability.expect("check_stability on");
    assert_eq!(
        st.verdict(),
        StabilityVerdict::Unstable,
        "lambda_min = {:+.3e}",
        st.lowest_eigenvalue
    );
}

/// TRAH armed from the first iteration must handle the hard case: no error,
/// and -- because the trust-region step along v_neg is a descent direction
/// the quadratic model sees (0.5 * lambda_min * tau^2 < 0) -- it should leave
/// the symmetric saddle and finish at the stable state.
#[test]
fn trah_handles_the_hard_case_and_reaches_the_stable_state() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let r = run(Some(1e6)).expect("TRAH must not error in the hard case");
    assert!(r.converged, "TRAH did not converge (E = {:.10})", r.energy);
    let st = r.stability.expect("check_stability on");
    assert_ne!(
        st.verdict(),
        StabilityVerdict::Unstable,
        "TRAH endpoint E = {:.10} is still a saddle: lambda_min = {:+.3e}",
        r.energy,
        st.lowest_eigenvalue
    );
    // The only near-zero mode is the axial-rotation Goldstone mode.
    assert!(
        st.lowest_eigenvalue.abs() < 1e-5,
        "expected the zero mode of the symmetry-broken state, got lambda_min = {:+.3e}",
        st.lowest_eigenvalue
    );
    assert!(
        E_SADDLE - r.energy > 0.04,
        "TRAH E = {:.10} did not leave the saddle {E_SADDLE:.10}",
        r.energy
    );
    assert!(
        (r.energy - E_STABLE).abs() < 1e-7,
        "TRAH E = {:.10}, expected the stable state {E_STABLE:.10} (saddle is {E_SADDLE:.10})",
        r.energy
    );
}

/// TRAH as a TAIL accelerator (the configuration the existing trah tests use:
/// armed once err_max < 1e-2). The hard-case escape step raises err_max back
/// above the trigger; the gate then handed the next iteration to DIIS, which
/// DISCARDED the step's pending prediction unscored and, with a history built
/// at the saddle, walked straight back into it. Measured on N2/cc-pVDZ this
/// cycled to max_iter (-108.6102 -> -108.5965, repeat).
///
/// Here at STO-3G the cycle happens too (iterations 4->6 and 6->8 escape and
/// return) but breaks by luck at iteration 9, so "converged" alone cannot see
/// the defect. The assertion is on the mechanism: no TRAH prediction may be
/// discarded unscored by a hand-off to DIIS.
#[test]
fn trah_tail_mode_does_not_hand_a_pending_step_to_diis() {
    use std::sync::atomic::Ordering;
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let d0 = ferric_scf::trah::TRAH_PREDICTIONS_DISCARDED.load(Ordering::Relaxed);
    let s0 = ferric_scf::trah::TRAH_STEPS_TAKEN.load(Ordering::Relaxed);
    let r = run(Some(1e-2)).expect("tail-mode TRAH must not error");
    let discarded = ferric_scf::trah::TRAH_PREDICTIONS_DISCARDED.load(Ordering::Relaxed) - d0;
    let steps = ferric_scf::trah::TRAH_STEPS_TAKEN.load(Ordering::Relaxed) - s0;
    assert!(steps > 0, "TRAH never engaged; the test exercises nothing");
    assert_eq!(
        discarded, 0,
        "{discarded} of {steps} TRAH step predictions were discarded unscored at a hand-off to DIIS"
    );
    assert!(
        r.converged,
        "tail-mode TRAH did not converge (E = {:.10})",
        r.energy
    );
    assert!(
        (r.energy - E_STABLE).abs() < 1e-7,
        "tail-mode TRAH E = {:.10}, expected the stable state {E_STABLE:.10} (saddle {E_SADDLE:.10})",
        r.energy
    );
}

// ── UHF: the arming gate in the spin channel ────────────────────────────────
//
// Stretched H2 singlet (STO-3G, R = 2.5 Angstrom) run UHF. The instability is
// the spin-symmetry-breaking mode kappa_alpha = -kappa_beta. This is NOT an
// exact hard case: `uhf_guess_mos` deliberately mixes HOMO/LUMO for a
// forced-UHF closed shell, so the gradient has a small component along the
// mode and TRAH's REGULAR AH step follows it (measured: a mutant that errors
// on every hard-case root leaves this test passing). What it does pin is the
// UHF arming gate: without the "keep TRAH armed while a step is pending" rule,
// 34 of 37 UHF TRAH predictions were discarded at hand-offs to DIIS (mutant
// measured). The hard-case step itself is pinned by the RHF tests above and
// the `trah::tests::hard_*` unit tests, through the solver both loops share.
//
// DIIS, meanwhile, restores the spin symmetry and stops on the saddle. PySCF
// 2.13: default UHF also stops on that SADDLE at -0.7029435997 (`stability()`
// internally unstable); following the instability reaches the spin-broken
// minimum at -0.9338672031 (<S^2> = 0.99), i.e. two H atoms.

const H2_E_SADDLE: f64 = -0.702_943_600_2;
const H2_E_BROKEN: f64 = -0.933_867_204_8;

fn run_uhf_h2(
    trah: Option<f64>,
) -> Result<ferric_scf::result::ScfResult, ferric_core::FerricError> {
    let mol = Molecule::parse_xyz("2\nh2\nH 0 0 0\nH 0 0 2.5\n", 0, 1).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let cfg = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        check_stability: true,
        trah_trigger: trah,
        ..Default::default()
    };
    ferric_scf::uhf::solve_uhf(&ParallelContext::default(), &mol, &prep, &bounds, &cfg)
}

#[test]
fn uhf_diis_lands_on_the_spin_symmetric_saddle() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let r = run_uhf_h2(None).expect("DIIS converges");
    assert!(
        (r.energy - H2_E_SADDLE).abs() < 1e-7,
        "DIIS E = {:.10}",
        r.energy
    );
    let st = r.stability.expect("check_stability on");
    assert_eq!(
        st.verdict(),
        StabilityVerdict::Unstable,
        "lambda_min = {:+.3e}",
        st.lowest_eigenvalue
    );
}

#[test]
fn uhf_trah_tail_breaks_spin_symmetry_without_handing_off_to_diis() {
    use std::sync::atomic::Ordering;
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let d0 = ferric_scf::trah::TRAH_PREDICTIONS_DISCARDED.load(Ordering::Relaxed);
    let s0 = ferric_scf::trah::TRAH_STEPS_TAKEN.load(Ordering::Relaxed);
    let r = run_uhf_h2(Some(1e-2)).expect("UHF TRAH must not error");
    let discarded = ferric_scf::trah::TRAH_PREDICTIONS_DISCARDED.load(Ordering::Relaxed) - d0;
    let steps = ferric_scf::trah::TRAH_STEPS_TAKEN.load(Ordering::Relaxed) - s0;
    assert!(steps > 0, "TRAH never engaged");
    assert_eq!(
        discarded, 0,
        "{discarded} of {steps} UHF TRAH predictions discarded at a hand-off to DIIS"
    );
    assert!(r.converged);
    assert!(
        (r.energy - H2_E_BROKEN).abs() < 1e-7,
        "UHF TRAH E = {:.10}, expected the spin-broken minimum {H2_E_BROKEN:.10} (saddle {H2_E_SADDLE:.10})",
        r.energy
    );
    let st = r.stability.expect("check_stability on");
    assert_eq!(
        st.verdict(),
        StabilityVerdict::Stable,
        "lambda_min = {:+.3e}",
        st.lowest_eigenvalue
    );
}

/// A REJECTED TRAH step must be undone and retried from the restored point --
/// not handed to DIIS. The RHF loop's rejection branch restored (C, D) but
/// then fell through to DIIS (only the accept path set `trah_took_step`; the
/// UHF loop sets it on both), and DIIS extrapolated from its own history over
/// the restored density. From the core guess, one rejection threw TRAH back to
/// its first step's saddle and it retraced the same path deterministically.
///
/// No per-iteration energies leave the solver, so this pins the iteration
/// count, measured on both sides (single-thread-independent: the J/K
/// reduction is deterministic): rejection falls through -> 37 iterations;
/// rejection retries -> 29. Bound halfway.
#[test]
fn a_rejected_trah_step_is_retried_not_handed_to_diis() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mol = n2_stretched();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let cfg = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        use_sad_guess: false,
        trah_trigger: Some(1e-2),
        ..Default::default()
    };
    let r0 = ferric_scf::trah::TRAH_STEPS_REJECTED.load(std::sync::atomic::Ordering::Relaxed);
    let r = solve_rhf(&ParallelContext::default(), &mol, &prep, op, &bounds, &cfg).unwrap();
    let rej = ferric_scf::trah::TRAH_STEPS_REJECTED.load(std::sync::atomic::Ordering::Relaxed) - r0;
    assert!(rej > 0, "no rejection happened; the test exercises nothing");
    assert!(r.converged);
    assert!((r.energy - E_STABLE).abs() < 1e-7, "E = {:.10}", r.energy);
    assert!(
        r.iterations <= 33,
        "{} iterations with {rej} rejections (retry path: 29; fall-through-to-DIIS: 37)",
        r.iterations
    );
}
