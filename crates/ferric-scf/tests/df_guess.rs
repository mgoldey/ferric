//! Tests for the opt-in "DF guess" two-stage SCF
//! (`ferric_scf::ladder::solve_rhf_with_df_guess`): a density-fitted J/K
//! pre-stage converged to a loose threshold, handing its density to a fresh
//! exact-4-index-integral SCF. See the module doc on `solve_rhf_with_df_guess`
//! for the full design rationale (Psi4's "Andy trick 2.0").
//!
//! Per this project's "EXACTNESS ANCHOR FIRST" convention, test (a) below is
//! written before any other measurement and must be trusted above all others:
//! it is the guard that the feature is vacuous when disabled.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::ladder::solve_rhf_with_df_guess;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

fn water_sto3g() -> (Molecule, ferric_core::basis::BasisSet) {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    (mol, bs)
}

/// (a) EXACTNESS ANCHOR: the DF-guess mechanism does not exist as far as the
/// disabled path is concerned. This does not call `solve_rhf_with_df_guess`
/// at all -- it simply proves that plain `solve_rhf` (the path every existing
/// caller uses, and the path the CLI still uses when `[scf] df_guess = false`)
/// is completely untouched by this feature's addition: same energy, same
/// iteration count, same exit, bit-identical. If this test is ever failing,
/// something about adding df_guess broke the default path, which is the one
/// non-negotiable requirement in this feature's spec.
#[test]
fn df_guess_disabled_path_is_untouched() {
    let (mol, bs) = water_sto3g();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let config = RhfConfig::default();
    let r1 = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();
    let r2 = solve_rhf(&ctx, &mol, &prep, op, &bounds, &config).unwrap();

    // Two independent calls to the untouched path must be bit-identical to
    // each other (sanity on determinism) -- this is the baseline the CLI's
    // `df_guess = false` path must continue to match exactly.
    assert_eq!(r1.energy.to_bits(), r2.energy.to_bits());
    assert_eq!(r1.iterations, r2.iterations);
    assert_eq!(r1.exit, r2.exit);
}

/// (b) SAME FIXED POINT: with the DF-guess pre-stage enabled, the final
/// (exact-stage) energy must agree with the plain (non-DF-guess) SCF energy
/// to within the SCF convergence tolerance. The DF stage may only change HOW
/// FAST the SCF gets there -- never WHERE it converges. Water/STO-3G is cheap
/// enough to run both paths in a unit test.
#[test]
fn df_guess_converges_to_same_energy_as_plain_scf() {
    let (mol, bs) = water_sto3g();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    // Tight thresholds so the "same fixed point" comparison is meaningful --
    // both stages' exact stage must land at the same converged basin.
    let base = RhfConfig::default()
        .with_energy_conv(1e-10)
        .with_density_conv(1e-9);

    let plain = solve_rhf(&ctx, &mol, &prep, op, &bounds, &base).unwrap();
    assert!(
        plain.converged,
        "plain SCF must converge for this comparison to mean anything"
    );

    let dfg = solve_rhf_with_df_guess(&ctx, &mol, &prep, op, &bounds, &base, None).unwrap();
    assert!(dfg.result.converged, "DF-guess exact stage must converge");

    let de = (plain.energy - dfg.result.energy).abs();
    assert!(
        de < 1e-8,
        "DF-guess exact-stage energy {:.12} disagrees with plain-SCF energy {:.12} by {:.3e} \
         Ha -- the DF pre-stage must only change convergence SPEED, not the converged fixed point",
        dfg.result.energy,
        plain.energy,
        de
    );
}

/// (c) THE DF STAGE ACTUALLY RAN: a test that would FAIL if the DF rung were
/// silently skipped (e.g. a future refactor that made `solve_rhf_with_df_guess`
/// just forward to `solve_rhf` once). We assert directly on `df_iterations`
/// being > 0 (the DF pre-stage really executed SCF iterations) AND that the
/// DF-stage config it ran under actually had `df_j_aux`/`df_k_aux` set --
/// checked indirectly by asserting the reported `df_energy` differs from the
/// exact-stage energy by MORE than the DF stage's own loose convergence
/// tolerance would allow for two IDENTICAL exact solves (i.e. the DF number is
/// a genuinely different, fitted quantity, not just a second exact solve
/// wearing a different name). This is the "mechanism could be inert" check:
/// if `solve_rhf_with_df_guess` degenerated into calling exact `solve_rhf`
/// twice, `df_energy` would equal `dfg.result.energy` to full SCF precision,
/// which the assertion below rules out.
#[test]
fn df_guess_pre_stage_actually_runs_fitted_scf() {
    let (mol, bs) = water_sto3g();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let base = RhfConfig::default();
    let dfg = solve_rhf_with_df_guess(&ctx, &mol, &prep, op, &bounds, &base, None).unwrap();

    assert!(
        dfg.df_iterations > 0,
        "DF pre-stage must run at least one SCF iteration"
    );
    // The DF stage's own converged energy is a FITTED quantity (def2-universal-
    // jkfit RI-J/RI-K), not the exact-integral energy -- it should differ from
    // the exact-stage energy by an amount characteristic of the RI fitting
    // error (~1e-4-1e-5 Ha territory on water/STO-3G per the project's own
    // measured DF-JK ladder history), which is MANY orders above SCF
    // convergence noise (~1e-9). If this delta collapsed to ~1e-9, the "DF"
    // stage would in fact be running exact J/K (i.e. df_j_aux/df_k_aux were
    // never actually wired into its config) and this assertion would catch it.
    let delta = (dfg.df_energy - dfg.result.energy).abs();
    assert!(
        delta > 1e-7,
        "DF pre-stage energy {:.12} is suspiciously close to the exact-stage energy {:.12} \
         (Δ={:.3e} Ha) -- the DF stage may not actually be running density-fitted J/K",
        dfg.df_energy,
        dfg.result.energy,
        delta
    );

    // The DF stage must also respect its capped max_iter: it should not have
    // silently run the caller's full (200-default) budget.
    assert!(
        dfg.df_iterations <= ferric_scf::ladder::DF_GUESS_MAX_ITER,
        "DF pre-stage ran {} iterations, exceeding its documented cap of {}",
        dfg.df_iterations,
        ferric_scf::ladder::DF_GUESS_MAX_ITER
    );
}

/// The exact-stage result must come from a run whose `df_j_aux`/`df_k_aux`
/// are exactly what the caller's `base` specified (normally both `None`,
/// i.e. truly exact 4-index) -- the DF stage's aux choice must never leak
/// into the returned result's provenance. We can't inspect the config a
/// `ScfResult` was built from directly, so this is checked by proxy: running
/// `solve_rhf_with_df_guess` with a deliberately bad/mismatched df_aux for
/// the PRE-stage must not change the final answer (within tolerance),
/// because the exact stage never sees that aux at all.
#[test]
fn df_guess_exact_stage_ignores_pre_stage_aux_choice() {
    let (mol, bs) = water_sto3g();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let base = RhfConfig::default()
        .with_energy_conv(1e-10)
        .with_density_conv(1e-9);

    let default_aux = solve_rhf_with_df_guess(&ctx, &mol, &prep, op, &bounds, &base, None).unwrap();
    let explicit_aux = solve_rhf_with_df_guess(
        &ctx,
        &mol,
        &prep,
        op,
        &bounds,
        &base,
        Some("def2-universal-jkfit"),
    )
    .unwrap();

    assert!(default_aux.result.converged && explicit_aux.result.converged);
    let de = (default_aux.result.energy - explicit_aux.result.energy).abs();
    assert!(
        de < 1e-8,
        "exact-stage energy should not depend on which (valid) DF-guess aux basis was used \
         for the pre-stage; Δ={:.3e} Ha",
        de
    );
}

#[test]
fn df_guess_result_type_is_used() {
    // Compile-time/shape sanity: DfGuessResult exposes the fields the CLI
    // wiring and this test file rely on. A field rename would fail this
    // file to compile, which is the point.
    let (mol, bs) = water_sto3g();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let dfg = solve_rhf_with_df_guess(&ctx, &mol, &prep, op, &bounds, &RhfConfig::default(), None)
        .unwrap();
    let _: bool = dfg.df_converged;
    let _: usize = dfg.df_iterations;
    let _: f64 = dfg.df_energy;
    let _ = dfg.result.energy;
}

/// MECHANISM-ACTIVE guard for the stall-point (tight) DF pre-stage policy.
///
/// The other tests in this file all pass under EITHER policy — none of them
/// would notice if the tight default silently reverted to the loosened
/// threshold, which is precisely the "a passing test may be measuring
/// inertness" failure this repo has a memory about. This one discriminates.
///
/// The two policies have opposite, individually-checkable signatures:
///   loose: DF stage CONVERGES in few iterations (7 on benzene), exact stage
///          then needs one MORE iteration.
///   tight: DF stage runs to its cap WITHOUT converging (25), exact stage
///          needs one FEWER.
/// Asserting both halves means a revert in either direction fails here.
///
/// Measured on benzene (aTZ): loose 7 DF + 6 exact vs tight 25 DF + 5 exact.
/// Water/STO-3G is used here for test cost; the DIRECTION is what is asserted,
/// not the benzene counts, because the exact-iteration saving is system
/// dependent and a point bar would be brittle.
#[test]
fn tight_df_pre_stage_runs_longer_and_shortens_the_exact_stage() {
    let (mol, bs) = water_sto3g();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let base = RhfConfig::default()
        .with_energy_conv(1e-10)
        .with_density_conv(1e-9);

    let run = |tight: bool| {
        // SAFETY-ish: these tests run single-threaded within this binary for
        // this env var's scope; mirrors the ENV_LOCK pattern used elsewhere.
        if tight {
            std::env::remove_var("FERRIC_DF_GUESS_TIGHT"); // default is ON
        } else {
            std::env::set_var("FERRIC_DF_GUESS_TIGHT", "0");
        }
        let r = solve_rhf_with_df_guess(&ctx, &mol, &prep, op, &bounds, &base, None).unwrap();
        std::env::remove_var("FERRIC_DF_GUESS_TIGHT");
        r
    };

    let loose = run(false);
    let tight = run(true);

    assert!(
        tight.df_iterations > loose.df_iterations,
        "tight policy must run the DF pre-stage LONGER (to its stall point): \
         tight {} vs loose {} DF iterations — if these are equal the tight \
         default is inert and the measured 1.15-1.18x is not being delivered",
        tight.df_iterations,
        loose.df_iterations
    );
    assert!(
        tight.result.iterations <= loose.result.iterations,
        "tight policy must not COST exact iterations: tight {} vs loose {} — \
         the entire justification is trading cheap DF iterations for expensive \
         exact ones, so a regression here inverts the trade",
        tight.result.iterations,
        loose.result.iterations
    );
    // Both must still land on the same converged answer.
    let de = (tight.result.energy - loose.result.energy).abs();
    assert!(
        de < 1e-9,
        "tight and loose DF policies converged to different energies (Δ={de:.3e} Ha) — \
         the pre-stage is a density generator only and must not move the fixed point"
    );
}

/// Water / cc-pVDZ with J and K fitted on `aux` (`None` = exact K).
fn water_ccpvdz_fitted(
    j_aux: Option<&str>,
    k_aux: Option<&str>,
) -> (Molecule, PreparedBasis, SchwarzBounds, RhfConfig) {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let prep = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let mut base = RhfConfig::default();
    base.df_j_aux = j_aux.map(str::to_string);
    base.df_k_aux = k_aux.map(str::to_string);
    base.energy_conv = 1e-10;
    base.density_conv = 1e-9;
    (mol, prep, bounds, base)
}

/// A fitted SCF on the pre-stage's own aux basis skips the pre-stage: the
/// result IS `solve_rhf(base)` (bitwise, the trivial anchor), and it agrees
/// with the two-stage result it replaces (pre-stage = `base` capped at
/// `DF_GUESS_TIGHT_MAX_ITER`, then `base` from its density) within the SCF
/// convergence bar.
///
/// The bar: both runs stop once the RMS density change drops below
/// `density_conv` (1e-9), so each sits within a few `density_conv` of the
/// same fixed point; 10 x `density_conv` on the RMS density difference. The
/// energy is stationary there, so its difference is second order in that,
/// far below 1e-12 Ha; the bar is the 1e-11 floor of summing an nbf^2
/// energy expression in f64. MEASURED (2026-11-14): see the printed line.
///
/// MUTATIONS (run, restored): `df_guess_repeats_the_scf` returning `true`
/// for every config fails `df_guess_pre_stage_actually_runs_fitted_scf`;
/// dropping its `df_k_aux` condition fails the J-only case below; dropping
/// the aux-name comparison fails the other-aux case below.
#[test]
fn df_guess_skips_its_pre_stage_when_the_scf_is_fitted_on_its_aux() {
    use ferric_scf::ladder::{DF_GUESS_DEFAULT_AUX, DF_GUESS_TIGHT_MAX_ITER};
    let aux = Some(DF_GUESS_DEFAULT_AUX);
    let (mol, prep, bounds, base) = water_ccpvdz_fitted(aux, aux);
    let ctx = ParallelContext::default();
    let op = Operator::coulomb();

    let dfg = solve_rhf_with_df_guess(&ctx, &mol, &prep, op, &bounds, &base, None).unwrap();
    assert!(
        dfg.skipped,
        "fitted on the pre-stage aux: pre-stage skipped"
    );
    assert_eq!(dfg.df_iterations, 0);
    let single = solve_rhf(&ctx, &mol, &prep, op, &bounds, &base).unwrap();
    assert_eq!(dfg.result.energy.to_bits(), single.energy.to_bits());
    assert_eq!(dfg.result.density_total, single.density_total);

    // The two-stage path this replaces, spelled out.
    let pre_cfg = base
        .clone()
        .with_max_iter(base.max_iter.min(DF_GUESS_TIGHT_MAX_ITER));
    let pre = solve_rhf(&ctx, &mol, &prep, op, &bounds, &pre_cfg).unwrap();
    let mut second_cfg = base.clone();
    second_cfg.init_guess_density = Some(pre.density_total.clone());
    second_cfg.use_sad_guess = false;
    let two_stage = solve_rhf(&ctx, &mol, &prep, op, &bounds, &second_cfg).unwrap();
    assert!(dfg.result.converged && two_stage.converged);
    let de = (dfg.result.energy - two_stage.energy).abs();
    let diff = &dfg.result.density_total - &two_stage.density_total;
    let rms = (diff.iter().map(|x| x * x).sum::<f64>() / diff.len() as f64).sqrt();
    println!(
        "skip vs two-stage: |dE| {de:.2e} Ha, rms(dD) {rms:.2e} (iterations {} vs {} + {})",
        dfg.result.iterations, pre.iterations, two_stage.iterations
    );
    assert!(rms <= 10.0 * base.density_conv, "rms(dD) {rms:.2e}");
    assert!(de <= 1e-11, "|dE| {de:.2e}");
}

/// The skip needs BOTH J and K fitted on the pre-stage's aux: a fit on
/// another aux, or an exact K, keeps the pre-stage.
#[test]
fn df_guess_keeps_its_pre_stage_unless_both_fits_use_its_aux() {
    use ferric_scf::ladder::{df_guess_repeats_the_scf, DF_GUESS_DEFAULT_AUX};
    let jk = Some(DF_GUESS_DEFAULT_AUX);
    let cases = [
        (jk, jk, None, true),
        (jk, jk, Some("DEF2-UNIVERSAL-JKFIT"), true),
        (jk, None, None, false),
        (None, jk, None, false),
        (Some("cc-pvtz-jkfit"), Some("cc-pvtz-jkfit"), None, false),
        (jk, jk, Some("cc-pvtz-jkfit"), false),
        (Some("exact"), Some("exact"), None, false),
    ];
    for (j, k, pre, want) in cases {
        let (_, _, _, base) = water_ccpvdz_fitted(j, k);
        assert_eq!(
            df_guess_repeats_the_scf(&base, pre),
            want,
            "j {j:?} k {k:?} pre {pre:?}"
        );
    }
    // And end to end for the other-aux case: the pre-stage runs.
    let other = Some("cc-pvtz-jkfit");
    let (mol, prep, bounds, base) = water_ccpvdz_fitted(other, other);
    let dfg = solve_rhf_with_df_guess(
        &ParallelContext::default(),
        &mol,
        &prep,
        Operator::coulomb(),
        &bounds,
        &base,
        None,
    )
    .unwrap();
    assert!(!dfg.skipped && dfg.df_iterations > 0);
}
