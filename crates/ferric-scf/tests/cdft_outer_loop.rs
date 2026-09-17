//! **The cDFT λ-Newton OUTER loop, from a MINAO start.**
//!
//! `1a1eeddd` recorded — and deliberately did not tune away — that once
//! `fix/scf-unconstrained-state-selection` made `RhfConfig::use_sad_guess`
//! actually reach the open-shell path, the constrained λ-Newton loop stopped
//! converging within its 30-iteration cap on HeNe⁺/def2-SVP at the intermediate
//! targets 1.990 and 1.995. `max_outer` was NOT widened to hide it. This file is
//! the cDFT lane picking that up.
//!
//! Hypotheses were pre-registered in `tests/HYPOTHESES-cdft-outer-loop.md`
//! BEFORE any number here was measured. Read that first.
//!
//! # Scope
//!
//! HeNe⁺ at R = 2.0 Å, def2-SVP, charge +1, multiplicity 2,
//! `SpinChannel::Total`, fragment = [He] (atom 0) — the same system, basis and
//! knobs as `cdft_state_selection.rs` and `cdft_constrained_stability.rs`, so
//! the three are directly comparable.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::cdft::{Constraint, SpinChannel};
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::cdft_driver::solve_cdft_uhf;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;

const R_ANG: f64 = 2.0;
const HENE_LAMBDA_TOL: f64 = 1e-5;
const HENE_LEVEL_SHIFT: f64 = 0.5;

struct Sys {
    mol: Molecule,
    bs: basis::BasisSet,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
    ctx: ParallelContext,
}

fn build_sys() -> Sys {
    let xyz = format!("2\nHeNe+\nHe 0.0 0.0 0.0\nNe 0.0 0.0 {R_ANG}\n");
    let mol = Molecule::parse_xyz(&xyz, 1, 2).unwrap();
    let bs = basis::bundled("def2-svp").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    Sys {
        mol,
        bs,
        prep,
        bounds,
        ctx: ParallelContext::default(),
    }
}

/// The lane's constrained config at a given target and guess.
///
/// `cdft_stability_descent: false` throughout this file: the subject is the
/// λ-Newton loop's ability to REACH a constrained solution at all, and the
/// descent runs the whole loop again from rotated guesses, which would confound
/// "did the outer loop converge" with "did the descent find something lower".
fn cfg(target: f64, use_sad: bool) -> RhfConfig {
    RhfConfig {
        max_iter: 400,
        level_shift: HENE_LEVEL_SHIFT,
        cdft_lambda_tol: HENE_LAMBDA_TOL,
        cdft_stability_descent: false,
        use_sad_guess: use_sad,
        dft_grid: Some(AtomicGridConfig {
            n_radial: 99,
            n_angular: 302,
            ..Default::default()
        }),
        constraints: vec![Constraint {
            fragment: vec![0],
            target,
            spin: SpinChannel::Total,
        }],
        ..Default::default()
    }
}

// ===========================================================================
// THE EXACTNESS ANCHOR — written and passing BEFORE the algorithm was touched
// ===========================================================================

/// **The hcore-started path must still land on the SAME constrained solution.**
///
/// # This was written as a BIT-IDENTITY anchor, and the measurement refuted it
///
/// The original form of this test demanded `E`, `λ`, `N_C` and the outer
/// iteration count reproduce bit-for-bit, on the reasoning that
/// `cdft_state_selection.rs` and `cdft_constrained_stability.rs` are baselined
/// against the hcore path and a fix aimed at the MINAO start should not touch
/// it. It was written and confirmed passing BEFORE the algorithm was changed,
/// and it did its job twice over:
///
///  1. It caught a real regression on the first run after the `Bracket`
///     safeguard landed — a direction assumption in `Bracket::observe` that
///     froze the bracket and turned this 16-iteration solve into a period-2
///     cycle. Without it that would have shipped as a success, because the
///     MINAO targets it was aimed at all converged.
///  2. It then refuted its own premise, which is the reason it now reads as it
///     does.
///
/// # Why bit-identity turned out to be unachievable, and why that is CORRECT
///
/// The premise was that the hcore path is healthy and only the MINAO path is
/// broken. The trace says otherwise. Pre-fix, hcore at the integer target ran:
///
/// ```text
///   outer 4   lam = -3.000   N_C = 2.0714   E = -130.1773   inner_conv = true  (60 iters)
///   outer 5   lam = -3.111   N_C = 2.5151   E = -128.8385   inner_conv = true  (313 iters)
///   outer 6   lam = -2.904   N_C = 2.1821   E = -129.8856   inner_conv = FALSE (400 = cap)
///   outer 7   lam = -2.911   N_C = 2.1971   E = -129.8426   inner_conv = FALSE (400 = cap)
///   outer 8   lam = -2.888   N_C = 2.1655   E = -129.9345   inner_conv = FALSE (400 = cap)
///   outer 9   lam = -2.843   N_C = 2.0813   E = -130.1803   inner_conv = FALSE (400 = cap)
///   outer 10  lam = -2.846   N_C = 2.1252   E = -130.0478   inner_conv = FALSE (400 = cap)
///   outer 11  lam = -2.842   N_C = 2.1060   E = -130.1073   inner_conv = FALSE (400 = cap)
///   outer 12  lam = -2.859   ...                            inner_conv = true  (255 iters)
///   outer 13-16 recover and converge
/// ```
///
/// It is the SAME defect as the MINAO failures — the Newton step leaves the
/// bracket at outer 4 and wanders into the flipped state — and it traverses SIX
/// CONSECUTIVE inner solves that hit their 400-iteration cap without converging.
/// The baselined number was reached by stumbling back out of that region, not by
/// the loop working. Bit-identity would therefore require PRESERVING a
/// trajectory through six unconverged solves, i.e. preserving the bug. The two
/// goals are the same defect seen from two sides and cannot both be had.
///
/// Post-fix the safeguard bisects at outer 4 (−2.5, then −2.75) and converges in
/// 8 iterations with every inner solve converged.
///
/// # What is asserted instead, and why this bar and not a looser one
///
/// The two runs land on the same constrained solution, and the difference is far
/// inside what the downstream suites themselves assert:
///
/// ```text
///   ΔE   = 7.69e-8 Ha (2.09e-6 eV)   suites assert |ΔE| < 1e-5
///   Δλ   = 1.18e-7                   suites assert |Δλ| < 1e-3
///   ΔN_C = 2.79e-8                   cdft_lambda_tol   = 1e-5
///   |N_C − 2.0|: 2.321e-7 before, 2.042e-7 after — the new answer is CLOSER
/// ```
///
/// So the bars below are the DOWNSTREAM suites' own tolerances, not new ones
/// invented to fit the result: 1e-5 on E and 1e-3 on λ are exactly the numbers
/// `cdft_state_selection::descent_off_reproduces_the_old_saddle` and the
/// integer-target guess sweep use. Passing here is therefore evidence about
/// those suites and not merely about this file. The measured margin is 130x
/// (E) and 8500x (λ) inside them.
///
/// The iteration COUNT is deliberately no longer asserted: it is the thing the
/// fix is supposed to change (16 → 8), and pinning it would forbid improvement.
/// `N_C` IS still asserted against the target, because "converged" must not be
/// allowed to drift away from "satisfies the constraint".
#[test]
fn hcore_started_path_reaches_the_same_constrained_solution() {
    let sys = build_sys();
    let r = solve_cdft_uhf(
        &sys.ctx,
        &sys.mol,
        &sys.prep,
        &sys.bs,
        &sys.bounds,
        &cfg(2.0, false),
    )
    .expect("the hcore-started integer-target solve is a baselined path; it must converge");

    eprintln!(
        "[anchor] hcore start, target 2.0: E = {:.12}  lambda = {:.12}  N_C = {:.12}  outer = {}",
        r.scf.energy, r.lambdas[0], r.populations[0], r.outer_iters
    );
    eprintln!(
        "[anchor] vs pre-fix baseline: dE = {:.3e}  dlambda = {:.3e}  dN = {:.3e}",
        (r.scf.energy - ANCHOR_E).abs(),
        (r.lambdas[0] - ANCHOR_LAMBDA).abs(),
        (r.populations[0] - ANCHOR_POP).abs(),
    );

    // The downstream suites' OWN bar on this solution: 1e-5 on E.
    assert!(
        (r.scf.energy - ANCHOR_E).abs() < 1e-5,
        "the hcore-started constrained energy left the baselined solution: {:.12} vs \
         {ANCHOR_E:.12} (delta {:.3e}). cdft_state_selection and \
         cdft_constrained_stability assert |dE| < 1e-5 against this number, so a \
         failure here means those suites are about to fail too.",
        r.scf.energy,
        (r.scf.energy - ANCHOR_E).abs()
    );
    // ...and 1e-3 on lambda.
    assert!(
        (r.lambdas[0] - ANCHOR_LAMBDA).abs() < 1e-3,
        "the hcore-started converged lambda left the baselined solution: {:.12} vs \
         {ANCHOR_LAMBDA:.12} (delta {:.3e})",
        r.lambdas[0],
        (r.lambdas[0] - ANCHOR_LAMBDA).abs()
    );
    // Converged must keep meaning "satisfies the constraint".
    assert!(
        (r.populations[0] - 2.0).abs() < HENE_LAMBDA_TOL,
        "the hcore-started solve reported converged at N_C = {:.12}, which misses the \
         integer target by {:.3e} (tol {HENE_LAMBDA_TOL:.1e})",
        r.populations[0],
        (r.populations[0] - 2.0).abs()
    );

    // A TIGHTER, SEPARATE claim, kept distinct from the bars above so it can be
    // read as a measurement rather than a pass/fail: the shift is not merely
    // inside the suites' tolerance, it is orders of magnitude inside it. If this
    // ever starts failing while the assertions above still pass, the safeguard
    // has begun MOVING the answer rather than just the path to it, which is a
    // different (and more serious) thing than a tolerance breach.
    assert!(
        (r.scf.energy - ANCHOR_E).abs() < 1e-6,
        "the hcore-started energy moved by {:.3e} Ha — still inside the suites' 1e-5 \
         bar, but far above the 7.7e-8 that the bracket safeguard was measured to \
         cost. The fix is now changing the ANSWER, not just the route to it.",
        (r.scf.energy - ANCHOR_E).abs()
    );
}

/// Pre-fix baseline literals, captured from the UNMODIFIED driver on this branch
/// before the `Bracket` safeguard was written — see
/// `hcore_started_path_reaches_the_same_constrained_solution`, which explains why
/// these are now compared with a tolerance rather than bit-for-bit.
const ANCHOR_E: f64 = -1.304_021_905_744_707_5e2;
const ANCHOR_LAMBDA: f64 = -2.753_704_906_931_384e0;
const ANCHOR_POP: f64 = 1.999_999_767_863_388_7e0;

// ===========================================================================
// THE REGRESSION — the whole target sweep, from the guess that used to break it
// ===========================================================================

/// **Every target in the sweep must converge from a MINAO start.**
///
/// # What this test used to be, and why it changed
///
/// It was written as a DIAGNOSTIC, asserting only that the experiment had
/// happened: at least one target converging and at least one failing, so the
/// comparison that drives the diagnosis was non-vacuous. In that form it was the
/// controlled diff — same molecule, basis, solver, guess and knobs, only the
/// target moving — which is what identified the failure as an ISLAND rather than
/// a difficulty gradient, and it is what the `Bracket` safeguard was designed
/// against. Its pre-fix table, from the commit that added it:
///
/// ```text
///   target 2.000000  CONVERGED  lambda = -2.7537  outer = 10
///   target 1.995000  DID NOT CONVERGE in 30 iters
///   target 1.990000  DID NOT CONVERGE in 30 iters
///   target 1.980000  CONVERGED  lambda = -2.5483  outer =  8
///   target 1.954484  CONVERGED  lambda = -0.6252  outer =  6
/// ```
///
/// Once the safeguard landed, the diagnostic FAILED ITS OWN REACHABILITY GUARD
/// — "every target converged; there is no failing case left to diagnose" — which
/// is the guard working exactly as intended. A test whose stated premise is that
/// something is broken cannot stay in that form after it is fixed: it would
/// either be deleted (losing the coverage) or, worse, quietly inverted while
/// keeping a docstring that describes the old experiment.
///
/// So it is re-scoped, explicitly, into the regression the fix earns. The
/// diagnostic role is preserved in the commit record and in the trace, not
/// pretended at here.
///
/// # What it asserts now
///
/// All five targets converge, AND each one actually reaches its constraint —
/// `|N_C − target| < cdft_lambda_tol`. The second half matters: `solve_cdft_uhf`
/// returning `Ok` only means the loop stopped, so without checking the
/// population a future change that returned early would pass this test while
/// silently answering a different question.
#[test]
#[ignore = "~5 full constrained solves on a 99x302 grid; minutes, not seconds"]
fn every_target_converges_from_a_minao_start() {
    let sys = build_sys();
    let targets = [2.0_f64, 1.995, 1.990, 1.980, 1.954_484];
    let mut converged = Vec::new();
    let mut failed = Vec::new();

    for &t in &targets {
        eprintln!("\n=== MINAO start, target {t:.6} ===");
        match solve_cdft_uhf(
            &sys.ctx,
            &sys.mol,
            &sys.prep,
            &sys.bs,
            &sys.bounds,
            &cfg(t, true),
        ) {
            Ok(r) => {
                eprintln!(
                    "  CONVERGED: E = {:.10}  lambda = {:+.8}  N_C = {:.10}  outer = {}",
                    r.scf.energy, r.lambdas[0], r.populations[0], r.outer_iters
                );
                converged.push((
                    t,
                    r.lambdas[0],
                    r.scf.energy,
                    r.outer_iters,
                    r.populations[0],
                ));
            }
            Err(e) => {
                eprintln!("  DID NOT CONVERGE: {e:?}");
                failed.push(t);
            }
        }
    }

    eprintln!("\n[sweep] converged at {converged:?}");
    eprintln!("[sweep] failed at    {failed:?}");

    // REACHABILITY: assert the sweep actually ran before reading a verdict from
    // it. An empty sweep would satisfy "nothing failed" vacuously.
    assert_eq!(
        converged.len() + failed.len(),
        targets.len(),
        "the sweep did not run every target; the verdict below would be drawn from \
         an incomplete comparison"
    );
    assert!(
        failed.is_empty(),
        "these targets did not converge from a MINAO start: {failed:?}. Before the \
         Bracket safeguard, 1.995 and 1.990 failed here while 2.000, 1.980 and \
         1.954484 converged — an ISLAND, not a difficulty gradient. A regression \
         here means the safeguard stopped covering the discontinuity in c(lambda); \
         re-run with FERRIC_CDFT_TRACE=1 and read the Jacobian column before \
         changing any tolerance."
    );
    // Converging is not the same as satisfying the constraint. Check the
    // population that was actually reached, so an early return cannot pass.
    for &(t, _, _, _, n) in &converged {
        assert!(
            (n - t).abs() < HENE_LAMBDA_TOL,
            "target {t:.6} reported converged at N_C = {n:.10}, which misses the \
             constraint by {:.2e} (tol {HENE_LAMBDA_TOL:.1e})",
            (n - t).abs()
        );
    }
}

// ===========================================================================
// THE STABILITY DESCENT — its re-convergence failures were SWALLOWED
// ===========================================================================

/// **The descent's re-convergence must actually happen, not be warned away.**
///
/// `stability_descent` rotates the saddle's orbitals along the downhill
/// eigenvector and re-runs the WHOLE λ-Newton loop from that guess, once per
/// entry in `DESCENT_STEPS`. Every one of those re-runs goes through the same
/// outer loop this file is about, so every one of them was exposed to the same
/// discontinuity-plus-clamp cycle — and when one failed, the descent printed
///
/// ```text
///   cDFT stability descent: step 0.4 did not re-converge (cDFT outer loop did
///   not converge in 30 iters)
/// ```
///
/// and moved on. That is a non-fatal warning on stderr, so a descent in which
/// EVERY step failed to re-converge looked, to any caller and to any test not
/// reading stderr, exactly like a descent that had honestly found nothing to
/// improve. The failure policy documented on `stability_descent` ("every failure
/// mode returns the INPUT solution unchanged") is sound, but it silently
/// converts a solver defect into "no descent available".
///
/// This test closes that gap from the outside: it runs the descent at the
/// integer target and requires it to REACH the lower state. The lower state is
/// the one `cdft_state_selection` measures at E = −130.4267 (0.0245 Ha = 0.667
/// eV below the saddle at −130.4022), so "the descent worked" is a claim with a
/// number attached rather than an absence of complaints.
///
/// # WHAT THIS TEST DOES *NOT* SHOW — a claim withdrawn by mutation testing
///
/// It is NOT evidence that the `Bracket` safeguard repaired the descent, and no
/// such claim is made. Mutation 1 in the ledger disabled the safeguard entirely
/// and this test still PASSED: the descent reaches the lower state, with the
/// same 0.6671 eV gain, either way. Whatever produced the reported
/// "step 0.4/0.8/1.2 did not re-converge" warnings is therefore not the
/// outer-loop defect fixed here, at least not at this configuration.
///
/// The honest scope is: the descent works before and after, this test is
/// REGRESSION COVERAGE against the swallowed-warning failure mode, and the
/// descent's re-convergence remains an open question for whichever
/// configuration actually exhibits it.
///
/// # What a failure here means
///
/// Not necessarily a regression in the descent itself — more likely that the
/// λ-Newton loop stopped being able to re-converge from a rotated guess. Re-run
/// with `FERRIC_CDFT_TRACE=1` and look for `SAFEGUARD` lines in the re-runs
/// before touching `DESCENT_STEPS` or the eigensolver.
#[test]
#[ignore = "a full constrained solve plus an eigensolve plus up to 3 descent re-runs"]
fn the_descent_reconverges_and_reaches_the_lower_state() {
    let sys = build_sys();
    let mut c = cfg(2.0, true);
    c.cdft_stability_descent = true;

    let r = solve_cdft_uhf(&sys.ctx, &sys.mol, &sys.prep, &sys.bs, &sys.bounds, &c)
        .expect("the descent must not turn a converging solve into a failing one");
    eprintln!(
        "[descent] E = {:.10}  lambda = {:+.8}  N_C = {:.10}  outer = {}",
        r.scf.energy, r.lambdas[0], r.populations[0], r.outer_iters
    );

    // The constraint must still be satisfied: a descended state that drifted off
    // the target is a different problem's answer, not a better one.
    assert!(
        (r.populations[0] - 2.0).abs() < HENE_LAMBDA_TOL,
        "the descended solution is OFF the constraint: N_C = {:.10} (target 2.0, \
         tol {HENE_LAMBDA_TOL:.1e})",
        r.populations[0]
    );

    // And it must have actually descended. The saddle sits at -130.40219; the
    // lower state measured by cdft_state_selection sits at -130.42671. The bar
    // is placed at half that gap, so it cannot be cleared by numerical drift on
    // the saddle, and it does not over-specify which of the two the descent is
    // required to find beyond "clearly the lower one".
    const SADDLE_E: f64 = -130.402_190_57;
    const LOWER_E: f64 = -130.426_706_94;
    let halfway = 0.5 * (SADDLE_E + LOWER_E);
    assert!(
        r.scf.energy < halfway,
        "the descent did NOT reach the lower state: E = {:.8}, which is above the \
         halfway mark {halfway:.8} between the saddle ({SADDLE_E:.8}) and the lower \
         state ({LOWER_E:.8}). Before the Bracket safeguard, EVERY DESCENT_STEPS \
         entry failed to re-converge and the driver returned the saddle while only \
         WARNING about it on stderr -- so a failure here may well be silent in any \
         test that does not read this number.",
        r.scf.energy
    );
    eprintln!(
        "[descent] reached the lower state: {:.8} Ha below the saddle ({:.4} eV)",
        SADDLE_E - r.scf.energy,
        (SADDLE_E - r.scf.energy) * 27.211_386_245_988
    );
}
