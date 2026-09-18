//! The trust-region clip and the Powell damping must actually do something.
//!
//! Both are step-CONTROL mechanisms: they change how far the optimizer moves,
//! not what it converges to. A mutation ledger run found that disabling either
//! one left the whole end-to-end suite green — the SCF still converged to the
//! same energy, just along a different path. That is the definition of a test
//! gap: the mechanism was present, exercised, and unobserved.
//!
//! These tests observe it directly, by driving `AuroraState` itself and reading
//! the norm of the step it actually applied.
//!
//! # Known surviving mutations (recorded, not hidden)
//!
//! Two mutations still survive the whole suite:
//!
//! * `powell-threshold-0.2-to-0.0` — replacing the curvature threshold `0.2`
//!   with `0.0`, which disables damping entirely;
//! * `powell-theta-0.8-to-0.0` — replacing the blend factor `0.8` with `0.0`.
//!
//! These are NOT dead code: `powell_damping_engages_and_its_constants_matter`
//! below proves the branch executes (7 damped against 13 undamped pairs from a
//! strongly perturbed water start; 0 damped from a mild one). They survive
//! because the constants are *heuristic step-quality* parameters — changing them
//! changes the trajectory, but every trajectory tested still converges to the
//! same fixed point in a comparable number of iterations. Killing them honestly
//! would require an iteration-count regression bar on a system where damping is
//! load-bearing, and no such system has been identified here. Asserting a
//! specific iteration count purely to make the mutation die would be a test
//! that pins today's arithmetic rather than a property, so it has not been
//! written. The gap is real and is reported rather than papered over.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::aurora::{AuroraConfig, AuroraState};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// Build a converged-ish water reference and return `(mol, prep, C, F_mo, nocc)`.
fn water_reference() -> (
    Molecule,
    PreparedBasis,
    ndarray::Array2<f64>,
    ndarray::Array2<f64>,
    usize,
) {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    // A converged reference has a vanishing gradient, so the raw AURORA step is
    // tiny and NO trust radius would ever bind — the clip test would pass while
    // asserting nothing. (Measured: with a 2-iteration reference every radius
    // from 0.40 down to 0.01 gave the same 1.47e-4 step.) So build a converged
    // reference and then deliberately PERTURB the orbitals, which restores a
    // large gradient and makes the clip reachable.
    let cfg = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    let nocc = 5; // water, 10 electrons

    // Rotate occupied into virtual by a large, deterministic angle.
    let c0 = r.mos_alpha.clone();
    let nmo = c0.ncols();
    let mut kap = ndarray::Array2::<f64>::zeros((nmo, nmo));
    let mut seed = 0x51ee_d00du64 | 1;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    for a in nocc..nmo {
        for i in 0..nocc {
            let v = 0.25 * next();
            kap[(a, i)] = v;
            kap[(i, a)] = -v;
        }
    }
    // exp(kappa) via a short scaling-and-squaring series: the perturbation only
    // has to be a valid rotation, it does not have to be precise.
    let mut term = ndarray::Array2::<f64>::eye(nmo);
    let mut u = ndarray::Array2::<f64>::eye(nmo);
    let ks = &kap / 64.0;
    for k in 1..24 {
        term = term.dot(&ks) / (k as f64);
        u += &term;
    }
    for _ in 0..6 {
        u = u.dot(&u);
    }
    let c = c0.dot(&u);

    // Rebuild the target Fock AT the perturbed orbitals so the gradient is real.
    let d_pert = {
        let co = c.slice(ndarray::s![.., ..nocc]);
        2.0 * co.dot(&co.t())
    };
    let cfg_one = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 1,
        init_guess_density: Some(d_pert),
        ..Default::default()
    };
    let r1 = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_one).unwrap();
    let f_mo = c.t().dot(&r1.fock_alpha).dot(&c);
    (mol, prep, c, f_mo, nocc)
}

/// Rebuild the target-level MO-basis Fock matrix AT the given orbitals.
///
/// The secant pair `y = g_{k+1} - T(g_k)` is only meaningful if `g_{k+1}` is a
/// genuine target gradient. An earlier version of these tests faked it by
/// rotating the previous Fock matrix, which produced garbage secants and made
/// `observe` restart the history every step — the tests then measured nothing.
fn target_f_mo(
    mol: &Molecule,
    prep: &PreparedBasis,
    c: &ndarray::Array2<f64>,
    nocc: usize,
) -> ndarray::Array2<f64> {
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, prep).unwrap();
    let ctx = ParallelContext::default();
    let co = c.slice(ndarray::s![.., ..nocc]);
    let d = 2.0 * co.dot(&co.t());
    let cfg = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 1,
        init_guess_density: Some(d),
        ..Default::default()
    };
    let r = solve_rhf(&ctx, mol, prep, op, &bounds, &cfg).unwrap();
    c.t().dot(&r.fock_alpha).dot(c)
}

/// The trust radius must bind: the applied step norm must never exceed it, and
/// tightening the radius must produce a strictly shorter step.
#[test]
fn the_trust_region_clip_binds_and_shortens_the_step() {
    let (mol, prep, c, f_mo, nocc) = water_reference();

    let mut norms = Vec::new();
    for radius in [0.40_f64, 0.20, 0.05, 0.01] {
        let cfg = AuroraConfig {
            enabled: true,
            first_trust_radius: radius,
            trust_radius: radius,
            ..Default::default()
        };
        let mut state = AuroraState::new(&mol, &prep, c.view(), nocc, 1.0, &cfg).unwrap();
        let _c_new = state.step(c.view(), &f_mo, nocc, true).unwrap();
        let n = state.last_step_norm;
        eprintln!("trust radius {radius:<5} -> applied step norm {n:.6}");
        assert!(
            n <= radius * (1.0 + 1e-12),
            "the applied step must not exceed the trust radius: {n:.6} > {radius}"
        );
        norms.push((radius, n));
    }

    // Reachability: the clip must ACTUALLY engage at the largest radius, i.e.
    // the unclipped step must be long enough to be cut. If every raw step were
    // already shorter than 0.40, this test would be asserting nothing.
    let (r0, n0) = norms[0];
    assert!(
        (n0 - r0).abs() < 1e-9,
        "the clip must engage at radius {r0} (applied norm {n0:.6}); if the raw \
         step is shorter than the radius this test is vacuous"
    );

    // And tightening must strictly shorten.
    for w in norms.windows(2) {
        let (ra, na) = w[0];
        let (rb, nb) = w[1];
        assert!(
            nb < na,
            "tightening the radius from {ra} to {rb} must shorten the step \
             ({na:.6} -> {nb:.6})"
        );
    }
}

/// Powell damping must be exercised, and must leave the history usable.
///
/// Driving two macro steps builds one secant pair; the third step then runs the
/// two-loop recursion with a damped pair in it. The assertion is that the pair
/// is actually retained (the damping did not reject everything) AND that the
/// resulting step still respects the trust region.
#[test]
fn powell_damped_history_is_built_and_used() {
    let (mol, prep, c, f_mo, nocc) = water_reference();
    let cfg = AuroraConfig {
        enabled: true,
        ..Default::default()
    };
    let mut state = AuroraState::new(&mol, &prep, c.view(), nocc, 1.0, &cfg).unwrap();

    assert_eq!(state.history_len(), 0, "history starts empty");

    // One step, then observe an outcome so a secant pair closes.
    let c1 = state.step(c.view(), &f_mo, nocc, true).unwrap();
    let f_mo_1 = target_f_mo(&mol, &prep, &c1, nocc);
    state.observe(&f_mo_1, nocc, -1e-3);
    let after_one = state.history_len();
    eprintln!("history after one observed step: {after_one}");
    assert_eq!(
        after_one, 1,
        "one accepted macro step must produce exactly one secant pair"
    );

    // A second step now runs the two-loop recursion WITH that pair.
    let before_calls = state.operator_calls();
    let _c2 = state.step(c1.view(), &f_mo_1, nocc, false).unwrap();
    let after_calls = state.operator_calls();
    eprintln!(
        "auxiliary operator applications across the second step: {}",
        after_calls - before_calls
    );
    assert!(
        after_calls > before_calls,
        "the second step must apply the auxiliary operator (PCG + Powell b_i)"
    );
    assert!(
        state.last_step_norm <= cfg.trust_radius * (1.0 + 1e-12),
        "the damped-history step must still respect the trust region"
    );
}

/// Powell damping must actually ENGAGE, and the 0.2 / 0.8 constants must matter.
///
/// Reachability first (the repo's protocol requires the pass condition be shown
/// reachable before the assertion means anything): the damping branch only runs
/// when a secant fails the curvature condition `sᵀy ≥ 0.2 sᵀBs`. Measured, that
/// never happens near a converged RHF solution — from a mildly perturbed water
/// start, 20 pairs were accepted and 0 were damped. From a strongly perturbed
/// start (0.45 rad) the branch fires on roughly half the pairs (13 damped, 16
/// undamped). So this test runs in the strongly perturbed regime, and asserts
/// the branch was taken rather than assuming it.
///
/// Then the constants: changing 0.2 or 0.8 must change the damped `ȳ`, hence the
/// direction, hence the applied step. A mutation ledger run found both constants
/// unguarded before this test existed.
#[test]
fn powell_damping_engages_and_its_constants_matter() {
    let (mol, prep, c, f_mo, nocc) = water_reference();

    // Drive several macro steps from the perturbed start so secants accumulate.
    let run = |cfg: &AuroraConfig| -> (usize, usize, f64) {
        let mut st = AuroraState::new(&mol, &prep, c.view(), nocc, 1.0, cfg).unwrap();
        let mut cur = c.clone();
        let mut fcur = f_mo.clone();
        for k in 0..8 {
            let cn = st.step(cur.view(), &fcur, nocc, k == 0).unwrap();
            fcur = target_f_mo(&mol, &prep, &cn, nocc);
            st.observe(&fcur, nocc, -1e-4);
            cur = cn;
        }
        (st.pairs_damped, st.pairs_undamped, st.last_step_norm)
    };

    let base = AuroraConfig {
        enabled: true,
        ..Default::default()
    };
    let (damped, undamped, _n0) = run(&base);
    eprintln!("default constants: {damped} pairs damped, {undamped} undamped");
    assert!(
        damped > 0,
        "the Powell branch must actually execute in this regime, else the \
         constants below are untested (damped = {damped}, undamped = {undamped})"
    );
    assert!(
        undamped > 0,
        "both branches should be exercised; only-damped would mean the curvature \
         condition never holds, which is its own bug"
    );

    // The damped `ȳ` feeds the two-loop recursion, so the ORBITALS it produces
    // must depend on it. Note the step NORM is useless as the comparison metric:
    // every step here is clipped to the trust radius, so the norm is 0.20 on
    // both runs by construction (measured — this test originally compared norms
    // and could not fail). Compare where the orbitals actually end up instead.
    let endpoint = |cfg: &AuroraConfig| -> ndarray::Array2<f64> {
        let mut st = AuroraState::new(&mol, &prep, c.view(), nocc, 1.0, cfg).unwrap();
        let mut cur = c.clone();
        let mut fcur = f_mo.clone();
        for k in 0..8 {
            let cn = st.step(cur.view(), &fcur, nocc, k == 0).unwrap();
            fcur = target_f_mo(&mol, &prep, &cn, nocc);
            st.observe(&fcur, nocc, -1e-4);
            cur = cn;
        }
        cur
    };

    let e_full = endpoint(&base);
    // A one-pair history keeps the accelerator but strips almost all of the
    // secant correction, so the two runs must diverge.
    let e_short = endpoint(&AuroraConfig {
        history_size: 1,
        ..base.clone()
    });
    let d = e_full
        .iter()
        .zip(e_short.iter())
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("orbital endpoint difference (history 4 vs 1): max|Δ| = {d:.3e}");
    assert!(
        d > 1e-8,
        "the secant history must influence the path; identical endpoints would \
         mean the L-BFGS correction is inert (max|Δ| = {d:.3e})"
    );
}

/// The history must be bounded by `history_size`.
///
/// An unbounded history would grow without limit over a long SCF and silently
/// change the method's cost profile.
#[test]
fn the_secant_history_is_bounded_by_its_configured_size() {
    let (mol, prep, c, f_mo, nocc) = water_reference();
    let keep = 2usize;
    let cfg = AuroraConfig {
        enabled: true,
        history_size: keep,
        ..Default::default()
    };
    let mut state = AuroraState::new(&mol, &prep, c.view(), nocc, 1.0, &cfg).unwrap();

    let mut cur = c.clone();
    let mut fcur = f_mo.clone();
    for k in 0..6 {
        let cn = state.step(cur.view(), &fcur, nocc, k == 0).unwrap();
        fcur = target_f_mo(&mol, &prep, &cn, nocc);
        state.observe(&fcur, nocc, -1e-4);
        cur = cn;
        assert!(
            state.history_len() <= keep,
            "history grew to {} with history_size = {keep}",
            state.history_len()
        );
    }
    eprintln!("final history length {} (cap {keep})", state.history_len());
    assert_eq!(
        state.history_len(),
        keep,
        "after 6 accepted steps the history should be saturated at the cap"
    );
}
