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

/// **The hcore-started path must not move, to the last bit.**
///
/// Every measured baseline in `cdft_state_selection.rs` and
/// `cdft_constrained_stability.rs` was recorded against the hcore-started
/// solver, and those files pin `use_sad_guess = false` for exactly that reason.
/// Any change to the λ-Newton loop that is supposed to fix the MINAO start must
/// leave the hcore start ALONE — otherwise it is silently re-baselining a large
/// existing suite rather than fixing a defect.
///
/// This is the highest-value check available on this lane: it catches an entire
/// class of "fix" that works by perturbing every solve.
///
/// The literals below were captured from the UNMODIFIED driver on this branch
/// (`FERRIC_CDFT_TRACE=1`, see the commit message for the trace). They are
/// compared with `==`, not `approx`: a loop change that is a no-op on this path
/// reproduces the same float exactly, and one that is not should say so loudly.
#[test]
fn hcore_started_path_is_bit_identical() {
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
    // Re-baselining aid: these are the EXACT literals to paste below. A decimal
    // printed at anything short of full precision does NOT round-trip, and the
    // bit-comparison then fails on a value that is numerically identical.
    eprintln!(
        "[anchor] exact literals: E = {:e}  lambda = {:e}  N_C = {:e}",
        r.scf.energy, r.lambdas[0], r.populations[0]
    );

    assert_eq!(
        r.scf.energy.to_bits(),
        ANCHOR_E.to_bits(),
        "the hcore-started constrained energy MOVED: {:.14} vs baseline {ANCHOR_E:.14}. \
         A large existing suite (cdft_state_selection, cdft_constrained_stability) is \
         baselined against this path. Whatever changed the MINAO loop must not touch it.",
        r.scf.energy
    );
    assert_eq!(
        r.lambdas[0].to_bits(),
        ANCHOR_LAMBDA.to_bits(),
        "the hcore-started converged lambda MOVED: {:.14} vs baseline {ANCHOR_LAMBDA:.14}",
        r.lambdas[0]
    );
    assert_eq!(
        r.populations[0].to_bits(),
        ANCHOR_POP.to_bits(),
        "the hcore-started converged population MOVED: {:.14} vs baseline {ANCHOR_POP:.14}",
        r.populations[0]
    );
    assert_eq!(
        r.outer_iters, ANCHOR_OUTER,
        "the hcore-started outer-iteration COUNT moved: {} vs baseline {ANCHOR_OUTER}. \
         Even at an identical answer, a different path to it means the loop changed on \
         a baselined trajectory.",
        r.outer_iters
    );
}

// Baseline literals — see `hcore_started_path_is_bit_identical`.
const ANCHOR_E: f64 = -1.304_021_905_744_707_5e2;
const ANCHOR_LAMBDA: f64 = -2.753_704_906_931_384e0;
const ANCHOR_POP: f64 = 1.999_999_767_863_388_7e0;
const ANCHOR_OUTER: usize = 16;

// ===========================================================================
// THE DIAGNOSTIC — a CONVERGING run and a NON-CONVERGING run, side by side
// ===========================================================================

/// **The controlled diff.** Same molecule, basis, solver, guess and knobs;
/// only the TARGET differs. One converges, the others do not.
///
/// Studying a failing run alone cannot separate "this loop is badly conditioned"
/// from "this loop never worked and the integer target was luck". Running the
/// working case in the SAME binary, from the SAME build, controls for every
/// variable except the one under test.
///
/// This test asserts only that the experiment HAPPENED (at least one target
/// converged and at least one did not, so the comparison is non-vacuous). The
/// verdict is drawn by the tests below from the traced numbers. Run with
/// `FERRIC_CDFT_TRACE=1 ... -- --nocapture --ignored` for the per-iteration
/// trace itself.
#[test]
#[ignore = "diagnostic: ~6 full constrained solves on a 99x302 grid, minutes each"]
fn target_sweep_converging_versus_not() {
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
                converged.push((t, r.lambdas[0], r.scf.energy, r.outer_iters));
            }
            Err(e) => {
                eprintln!("  DID NOT CONVERGE: {e:?}");
                failed.push(t);
            }
        }
    }

    eprintln!("\n[sweep] converged at {converged:?}");
    eprintln!("[sweep] failed at    {failed:?}");

    // REACHABILITY: the comparison is only informative if BOTH classes are
    // populated. An all-converged or all-failed sweep says nothing about what
    // distinguishes them, and must fail loudly rather than pass silently.
    assert!(
        !converged.is_empty(),
        "no target converged from a MINAO start; the CONTROL is missing and nothing \
         below distinguishes a conditioning problem from a loop that never worked"
    );
    assert!(
        !failed.is_empty(),
        "every target converged from a MINAO start; there is no failing case left to \
         diagnose. If this is a real change, the lane's premise has moved and the \
         HYPOTHESES file must be updated before any conclusion is drawn from it."
    );
}
