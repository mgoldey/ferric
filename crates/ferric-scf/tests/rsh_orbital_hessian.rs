//! The range-separated orbital Hessian (#314): exactness anchor, analytic-vs-FD
//! at ω ≠ 0, and the N2⁺ λ_min sign flip.
//!
//! # What changed and why these are the tests
//!
//! The Hessian matvecs used to build their exchange response from ONE
//! `build_jk_with_pool` call at the ambient (Coulomb) operator, scaled by the
//! single scalar `k_mix_sr`. For a range-separated hybrid the converged Fock
//! instead assembles `c_SR·K[erfc(ω)] + c_LR·K[erf(ω)]` from two `DfK` fitters,
//! so the Hessian was a different operator and
//! `stability::ks_reference_is_analysable` refused every ω ≠ 0.
//!
//! `UhfNewtonInputs::rsh` / `RhfNewtonInputs::rsh` / `RohfNewtonInputs::rsh` now
//! carry a [`ferric_scf::rsh_response::RshResponse`] borrowing the SAME two
//! fitters, and `None` (ω = 0) leaves the old plain-Coulomb expression
//! untouched.
//!
//! # Hypotheses (committed in `reference/hypotheses/314-rsh-orbital-hessian.md`)
//!
//! * **H_physics** — the SR/LR response is the true second derivative of the
//!   density-fitted RSH energy. Then ω = 0 is bit-identical, analytic-vs-FD at
//!   ω ≠ 0 holds the ω = 0 bar, and N2⁺/def2-SVP λ_min straddles zero between
//!   ω = 0.53 and 0.56 as PySCF's probe says.
//! * **A1 erf/erfc swapped**, **A2 c_LR dropped**, **A3 `k_mix.sr` for both** —
//!   each, made inside the Hessian's exchange response, leaves the converged
//!   energy untouched. They do NOT make the in-crate analytic-vs-FD tests fail:
//!   those build their reference Fock through the same `RshResponse`, so a
//!   mutation there moves Fock and Hessian together (measured in the #292
//!   mutation ledger: all three survive FD). The test that sees them is
//!   `rsh_response_matches_an_independent_four_centre_construction` below,
//!   against direct four-centre integrals; the weekly
//!   `validation_rsh_stability.rs` PySCF comparison sees them too.
//! * **A4 DF-vs-direct inconsistency** — had the response been built from direct
//!   four-centre erf/erfc integrals instead of the Fock's own fitters, the FD
//!   residual would sit at the DF fitting error and NOT shrink with the step.
//!   The in-crate `rsh_response::tests::fd_holds_*` tests measure the residual
//!   at two steps and assert it shrinks, which is what distinguishes a wrong
//!   operator from a coarse difference.
//! * **A6 the λ_min test cannot fail** — asserting only "the signs differ" is
//!   arithmetic if the two values do not actually straddle zero.
//!   `validation_rsh_stability.rs::n2_cation_is_stable_below_the_onset_and_unstable_above`
//!   asserts the PREMISE
//!   (λ_min(0.40) > 0 with margin AND λ_min(0.60) < 0 with margin, either side
//!   of the 0.53–0.56 onset) before the conclusion.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ndarray::Array2;

/// Deterministic xorshift64 PRNG (mirrors `uhf_newton_smoke.rs`; no rand dep).
struct Xorshift64(u64);
impl Xorshift64 {
    fn next_f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }
}

const AUX: &str = "def2-universal-jkfit";
const FUNCTIONAL: &str = "wB97X-V";

fn oh_doublet() -> Molecule {
    Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0, 2).unwrap()
}

// ---------------------------------------------------------------------------
// 1. ω = 0: determinism, equivalence to the pre-#292 code, and wiring.
// ---------------------------------------------------------------------------
//
// Three separate questions, three separate tests. Only the second answers "did
// #292 leave the ω = 0 path mathematically unchanged?":
//
// * `omega_zero_matvec_is_deterministic_through_the_none_path` — DETERMINISM
//   only (two runs of the same code agree to the bit).
// * `omega_zero_matvec_matches_the_pre_292_reference` — EQUIVALENCE to the
//   pre-#292 `hessian_matvec`, on fixed inputs, against a reference generated
//   from that code.
// * `omega_zero_production_stability_takes_the_coulomb_arm` — WIRING: the
//   production solver hands the matvec `rsh: None` at ω = 0.

/// **Determinism through `rsh: None`, and nothing more.**
///
/// Drives the ω = 0 matvec twice with identical inputs and asserts every element
/// is bit-identical. That pins the property the crate's bit-identity convention
/// rests on — the ω = 0 arm is deterministic and reduction-order-stable — and
/// that the `rsh` dispatch adds no run-to-run nondeterminism.
///
/// It does NOT check equivalence to the pre-#292 code: both calls execute
/// whatever the ω = 0 arm currently is, so any change to that arm, wrong or not,
/// leaves them equal to each other. (An earlier version of this doc claimed a
/// reassociation mutant made it fail. That claim was never run, and it cannot
/// hold, for exactly this reason.) Equivalence is
/// `omega_zero_matvec_matches_the_pre_292_reference`.
#[test]
fn omega_zero_matvec_is_deterministic_through_the_none_path() {
    let mol = oh_doublet();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    let res = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
    assert!(res.converged);

    let n = prep.nbasis();
    let c_a = res.mos_alpha.clone();
    let c_b = res.mos_beta.clone().unwrap();
    let nelec = mol.nelec() as usize;
    let two_s = mol.multiplicity - 1;
    let nocc_a = (nelec + two_s) / 2;
    let nocc_b = (nelec - two_s) / 2;
    let f_a_mo = c_a.t().dot(&res.fock_alpha).dot(&c_a);
    let f_b_mo = c_b.t().dot(res.fock_beta.as_ref().unwrap()).dot(&c_b);

    let inputs = ferric_scf::uhf_newton::UhfNewtonInputs {
        prep: &prep,
        bounds: &bounds,
        c_a: &c_a,
        c_b: &c_b,
        f_a_mo: &f_a_mo,
        f_b_mo: &f_b_mo,
        nocc_a,
        nocc_b,
        k_mix_sr: 1.0, // pure HF
        rsh: None,     // the trivial limit: omega == 0
        fxc: None,
        thresh: cfg.integral_thresh,
        ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
    };

    let mut rng = Xorshift64(0x9E3779B97F4A7C15);
    let k_a = Array2::<f64>::from_shape_fn((n - nocc_a, nocc_a), |_| 0.01 * rng.next_f64());
    let k_b = Array2::<f64>::from_shape_fn((n - nocc_b, nocc_b), |_| 0.01 * rng.next_f64());

    let pool = ferric_scf::engine_pool::EnginePool::new(
        bounds.op,
        &prep,
        ferric_integrals::engine_pool::eri_precision(),
    )
    .unwrap();
    let (h1_a, h1_b) =
        ferric_scf::uhf_newton::hessian_matvec(&ctx, &inputs, &k_a, &k_b, &pool).unwrap();
    let (h2_a, h2_b) =
        ferric_scf::uhf_newton::hessian_matvec(&ctx, &inputs, &k_a, &k_b, &pool).unwrap();

    // BIT-identical, asserted on the raw bit patterns so a NaN or a -0.0/+0.0
    // difference cannot pass as equal either.
    for (x, y) in h1_a.iter().zip(h2_a.iter()) {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "alpha block: omega = 0 matvec is not bit-identical ({x:.17e} vs {y:.17e})"
        );
    }
    for (x, y) in h1_b.iter().zip(h2_b.iter()) {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "beta block: omega = 0 matvec is not bit-identical ({x:.17e} vs {y:.17e})"
        );
    }
    eprintln!(
        "omega = 0 anchor: {} + {} elements bit-identical through rsh: None",
        h1_a.len(),
        h1_b.len()
    );
    // The anchor's pass condition must be REACHABLE: a matvec that returned an
    // all-zero block would trivially satisfy bit-equality. Assert the operator
    // actually did something.
    let m = h1_a
        .iter()
        .chain(h1_b.iter())
        .fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(
        m > 1e-6,
        "the omega = 0 matvec returned a (near-)zero block ({m:.3e}); \
         bit-equality would then be vacuous"
    );
}

/// The ω = 0 SCF energy of a **global hybrid** (B3LYP, `k_mix.sr == k_mix.lr`,
/// ω = 0) with Newton acceleration engaged must be EXACTLY what it was before
/// #314, to the last printed digit and in the same iteration count.
///
/// Complements the matvec-level anchor above by pinning the end-to-end numbers.
/// The pinned values were MEASURED on `origin/main` @ af833f6c, before this
/// branch's change, by running the identical configuration in a detached
/// worktree:
///
/// ```text
///   DIIS   E = -75.7319378517   23 iterations
///   Newton E = -75.7319382227   19 iterations
/// ```
///
/// Pinning both energies and both iteration counts is deliberately stricter
/// than comparing them to each other. The pre-existing DIIS-vs-Newton gap on
/// this system is 3.71e-7 Ha — larger than one might guess, because the two
/// paths take different trajectories to the same `energy_conv = 1e-9`
/// stationary point — so a "DIIS ≈ Newton" assertion would have to be loosened
/// to ~1e-6 and would then no longer be able to see a small regression. Pinning
/// the two absolute values at 1e-9 can: a dispatch that took the RSH arm for a
/// global hybrid, or any perturbation of the ω = 0 Coulomb expression, moves
/// at least one of these four numbers.
///
/// **These pins are machine-dependent, so this test runs only on the box that
/// measured them.** OH is a ²Π radical with a near-degenerate SOMO, so its SCF
/// endpoint moves with the floating-point environment. On CI the DIIS path,
/// which #314 does not touch, lands at −75.7319390095, 1.16e-6 Ha from the value
/// measured here. CI's machine-independent guards on the ω = 0 path are
/// `omega_zero_matvec_matches_the_pre_292_reference` (the matvec equals the
/// pre-#292 code on fixed inputs) and
/// `omega_zero_production_stability_takes_the_coulomb_arm` (the solver feeds it
/// `rsh: None`).
#[test]
#[ignore = "pins absolute OH/B3LYP energies and iteration counts measured on the dev box; machine-dependent (CI's DIIS lands 1.2e-6 Ha away), so run it locally with --ignored. The CI guards are omega_zero_matvec_matches_the_pre_292_reference and omega_zero_production_stability_takes_the_coulomb_arm."]
fn omega_zero_hybrid_newton_still_matches_diis() {
    let mol = oh_doublet();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let cfg_diis = RhfConfig {
        xc: Some("B3LYP".into()),
        energy_conv: 1e-9,
        density_conv: 1e-7,
        max_iter: 300,
        ..Default::default()
    };
    let cfg_newton = RhfConfig {
        newton_trigger: 1e-2,
        ..cfg_diis.clone()
    };
    // Measured on origin/main @ af833f6c (see the doc comment).
    const E_DIIS_MAIN: f64 = -75.7319378517;
    const E_NEWTON_MAIN: f64 = -75.7319382227;
    const IT_DIIS_MAIN: usize = 23;
    const IT_NEWTON_MAIN: usize = 19;

    let r_diis = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg_diis).unwrap();
    let r_newton = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg_newton).unwrap();
    assert!(r_diis.converged && r_newton.converged);
    eprintln!(
        "UKS/B3LYP OH  DIIS {:.10} ({} it)  Newton {:.10} ({} it)  |dE| {:.2e}",
        r_diis.energy,
        r_diis.iterations,
        r_newton.energy,
        r_newton.iterations,
        (r_diis.energy - r_newton.energy).abs()
    );
    assert!(
        (r_diis.energy - E_DIIS_MAIN).abs() < 1e-9,
        "DIIS energy moved from the pre-#314 value: {:.10} vs {E_DIIS_MAIN:.10}",
        r_diis.energy
    );
    assert!(
        (r_newton.energy - E_NEWTON_MAIN).abs() < 1e-9,
        "Newton energy moved from the pre-#314 value: {:.10} vs {E_NEWTON_MAIN:.10}. \
         The Newton path is the one #314 changed, so this is the assertion that sees a \
         regression in the omega = 0 dispatch.",
        r_newton.energy
    );
    assert_eq!(
        r_diis.iterations, IT_DIIS_MAIN,
        "DIIS iteration count moved from the pre-#314 value"
    );
    assert_eq!(
        r_newton.iterations, IT_NEWTON_MAIN,
        "Newton iteration count moved from the pre-#314 value"
    );
}

// ---------------------------------------------------------------------------
// 2. Analytic-vs-finite-difference at ω != 0.
// ---------------------------------------------------------------------------

// Analytic-vs-finite-difference at omega != 0 lives IN-CRATE, in
// `src/rsh_response.rs`'s test module: the FD reference has to rebuild the
// SR/LR Fock `F_sigma = h + J - (c_SR K_SR + c_LR K_LR)_sigma` and then
// re-diagonalize it, and `rhf::canonical_orthogonalizer` is `pub(crate)`. An
// integration test would have to reimplement the orthogonalizer, which would
// make the FD reference depend on a SECOND, test-local construction of
// something the solver already owns -- the opposite of what an independent
// reference should be. See `rsh_response::tests`.

/// `erfc(ω) + erf(ω) ≡ Coulomb`: a SECOND exactness anchor, independent of the
/// `rsh: None` one.
///
/// With `c_SR = c_LR = 1` the SR/LR response must reproduce the plain-Coulomb
/// `δK` the ω = 0 path builds. It cannot be bit-identical — one side is
/// density-fitted and the other is a direct four-centre contraction — so the bar
/// is the DF fitting error, and the test REPORTS that error rather than hiding
/// it behind a loose bar. This is the number artifact hypothesis A4 is about:
/// it is what the FD residual would be stuck at if the response and the Fock
/// disagreed about DF-vs-direct.
#[test]
fn erfc_plus_erf_response_reproduces_the_coulomb_response() {
    let mol = oh_doublet();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();
    let n = prep.nbasis();
    let budget = ferric_core::memory::resolve_budget_bytes(None);

    // A symmetric probe density perturbation (the shape a matvec feeds in).
    let mut rng = Xorshift64(0xB5026F5AA96619E9);
    let mut dd = Array2::<f64>::from_shape_fn((n, n), |_| 0.01 * rng.next_f64());
    dd = 0.5 * (&dd + &dd.t());

    // SR/LR density-fitted response at c_SR = c_LR = 1.
    let dfbs = basis::bundled(AUX).unwrap();
    let dfbs_prep = PreparedBasis::new(&mol, &dfbs).unwrap();
    let omega = 0.40;
    let sr = std::cell::RefCell::new(
        ferric_scf::df_k::DfK::new(Operator::erfc(omega), &prep, &dfbs_prep, budget).unwrap(),
    );
    let lr = std::cell::RefCell::new(
        ferric_scf::df_k::DfK::new(Operator::erf(omega), &prep, &dfbs_prep, budget).unwrap(),
    );
    let rsh = ferric_scf::rsh_response::RshResponse::new(&sr, &lr, 1.0, 1.0, omega);
    let k_rsh = rsh.exchange_response(&dd).unwrap();

    // Plain-Coulomb direct response, the ω = 0 path's kernel.
    let mut j_dum = Array2::<f64>::zeros((n, n));
    let mut k_coulomb = Array2::<f64>::zeros((n, n));
    ferric_scf::rhf::build_jk(&ctx, &prep, &bounds, 1e-14, &dd, &mut j_dum, &mut k_coulomb)
        .unwrap();

    let max_dev = k_rsh
        .iter()
        .zip(k_coulomb.iter())
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    let scale = k_coulomb.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let rel = max_dev / scale;
    eprintln!(
        "erfc(w) + erf(w) == Coulomb at w = {omega}: max|dK_SRLR - dK_Coulomb| = {max_dev:.3e} \
         (rel {rel:.3e}); scale {scale:.3e}. This IS the DF fitting error, and it is the \
         floor a DF-vs-direct inconsistency (artifact A4) would park the FD residual at."
    );
    // Reachability: the probe must actually produce a non-trivial K.
    assert!(
        scale > 1e-6,
        "probe response is ~zero ({scale:.3e}); the test is vacuous"
    );
    assert!(
        rel < 5e-3,
        "the SR/LR response does not reproduce the Coulomb response at c_SR = c_LR = 1 \
         (rel {rel:.3e}); erfc + erf == Coulomb is an identity, so this is a construction \
         error, not a fitting error"
    );
    // And it must NOT be bit-identical: if it were, the two sides are not the
    // two independent constructions this anchor relies on.
    assert!(
        max_dev > 0.0,
        "SR/LR and direct Coulomb agreed to the BIT, which cannot happen between a \
         density-fitted and a four-centre contraction — one of the two is not what it claims"
    );
}

// ---------------------------------------------------------------------------
// 3. The production gate after dropping the omega != 0 refusal.
// ---------------------------------------------------------------------------
//
// The DECISIVE lambda_min measurement (N2+ / def2-SVP / wB97X-V at omega =
// 0.53 and 0.56 vs PySCF's symmetric_cation_probe) lives IN-CRATE, in
// `src/rsh_response.rs`'s test module. It needs `rohf::FxcKernelStore`, which
// is `pub(crate)`: the f_xc response kernel is deliberately not public, and
// exporting it just to let a test build one would widen the crate's surface for
// a measurement rather than for a user. See `rsh_response::tests`.

/// `check_stability` must STILL refuse ωB97X-V, for the VV10 reason rather than
/// the range-separation reason.
///
/// Dropping the ω ≠ 0 refusal without this would turn a correct skip into a
/// silently incomplete λ_min: the exchange response is now right, but the VV10
/// nonlocal correlation response does not exist anywhere in the workspace
/// (`ferric_dft::fxc` has no VV10 term, and `zvector_ks` /
/// `lr_kernel::resolve_singlet_response_xc` both refuse VV10 functionals for
/// exactly this reason).
#[test]
fn wb97xv_is_still_refused_but_for_the_vv10_reason() {
    use ferric_scf::stability::{ks_reference_is_analysable, StabilitySkip};

    // Range separation alone is NO LONGER a refusal.
    assert!(
        ks_reference_is_analysable(Some("CAM-B3LYP"), 0.33).is_ok(),
        "a range-separated functional with no VV10 must now be analysable: the matvec \
         builds the SR/LR exchange response from the Fock's own fitters"
    );
    // VV10 is, and names itself.
    assert_eq!(
        ks_reference_is_analysable(Some(FUNCTIONAL), 0.3),
        Err(StabilitySkip::Vv10Kernel),
        "wB97X-V carries VV10 and must be refused for THAT reason"
    );
    let reason = StabilitySkip::Vv10Kernel.reason();
    assert!(
        reason.contains("VV10"),
        "the skip reason must name VV10: {reason}"
    );
    assert!(
        !reason.contains("plain Coulomb"),
        "the skip reason still blames the plain-Coulomb kernel, which is no longer \
         what happens: {reason}"
    );
    // And the meta-GGA arm is untouched.
    assert_eq!(
        ks_reference_is_analysable(Some("SCAN"), 0.0),
        Err(StabilitySkip::MetaGga)
    );
}

/// **#292's exactness anchor (a).** With `c_SR = c_LR = c` the RSH exchange
/// response must equal the GLOBAL-HYBRID response with coefficient `c`, to
/// ≤1e-12, because `erf(ωr)/r + erfc(ωr)/r = 1/r` pointwise for every ω.
///
/// Unlike the ω = 0 anchor this one exercises the erf/erfc wiring AT nonzero ω,
/// which is exactly what ω = 0 cannot do.
///
/// # The mapping onto ferric's actual `KMix` fields
///
/// `ferric_dft::xc_trait::KMix` carries `{ sr, lr, omega }` and
/// `fock_assembly::subtract_rsh_exchange` consumes them as
///
/// ```text
///   K_total(D) = sr · K[erfc(omega)](D) + lr · K[erf(omega)](D)
/// ```
///
/// so the global-hybrid limit is `sr = lr = c`. `KMix::default()` is
/// `{ sr: 1.0, lr: 1.0, omega: 0.0 }` — the `omega = 0` corner of the same
/// mapping — which is why the plain-HF `k_mix_sr = 1.0` callers and the `c = 1`
/// case here describe one operator.
///
/// # Where the 1e-12 bar is reachable, and where it is NOT
///
/// This test compares the three exchange matrices at the INTEGRAL level
/// (direct four-centre `build_jk` at `Operator::erfc(ω)`, `Operator::erf(ω)`
/// and `Operator::coulomb()`), because that is where the identity is exact:
/// measured **6.4e-15 to 8.2e-15 relative** across ω = 0.11 / 0.30 / 0.56,
/// three decades inside the bar.
///
/// It is NOT reachable through ferric's production DF path, and that is a fact
/// about density fitting rather than a defect. Each of the three `DfK` fitters
/// forms its own `V^{-1/2}` from its own two-centre metric — `(P|erfc|Q)`,
/// `(P|erf|Q)`, `(P|1/r|Q)` — so the identity cancels most but not all of the
/// fitting error. MEASURED on this system at ω = 0.30, `c = 1`:
///
/// ```text
///   aux basis              naux   SR+LR vs DF-Coulomb   DF-Coulomb vs direct
///   def2-universal-jkfit     95          6.87e-5               2.34e-3
///   cc-pvdz-ri               70          5.71e-5               7.73e-3
///   aug-cc-pvtz              69          1.45e-4               4.78e-2
/// ```
///
/// The SR+LR-vs-DF-Coulomb column is 30–300× SMALLER than the DF-vs-direct
/// column and tracks the aux basis, which is the signature of a residual
/// fitting error rather than a wiring error: a wrong coefficient or a swapped
/// kernel would not shrink toward the DF reference at all, and
/// `erfc_plus_erf_response_reproduces_the_coulomb_response` below reports that
/// production-path number (1.68e-3 relative) on its own terms.
///
/// Three ω and three `c` are swept, because a single pair can be satisfied by a
/// construction that is wrong in a way that happens to cancel there.
#[test]
fn equal_sr_lr_coefficients_reproduce_the_global_hybrid_response() {
    let mol = oh_doublet();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let ctx = ParallelContext::default();
    let n = prep.nbasis();

    // A symmetric probe perturbation, the shape a matvec feeds in.
    let mut rng = Xorshift64(0xCBBB9D5DC1059ED8);
    let mut dd = Array2::<f64>::from_shape_fn((n, n), |_| 0.01 * rng.next_f64());
    dd = 0.5 * (&dd + &dd.t());

    let kbuild = |op: Operator| -> Array2<f64> {
        let b = SchwarzBounds::compute(op, &prep).unwrap();
        let mut j = Array2::<f64>::zeros((n, n));
        let mut k = Array2::<f64>::zeros((n, n));
        ferric_scf::rhf::build_jk(&ctx, &prep, &b, 1e-14, &dd, &mut j, &mut k).unwrap();
        k
    };
    let k_coul = kbuild(Operator::coulomb());
    let scale = k_coul.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(
        scale > 1e-6,
        "the Coulomb reference response is ~zero ({scale:.3e}); the test would be vacuous"
    );

    let mut worst = 0.0f64;
    for omega in [0.11_f64, 0.30, 0.56] {
        let k_sr = kbuild(Operator::erfc(omega));
        let k_lr = kbuild(Operator::erf(omega));

        // REACHABILITY, per omega: the identity must not hold trivially. If the
        // SR half alone were already ~equal to the full Coulomb exchange, then
        // agreement at c_SR = c_LR would say nothing about the SPLIT.
        let sr_alone = k_sr
            .iter()
            .zip(k_coul.iter())
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()))
            / scale;
        assert!(
            sr_alone > 1e-2,
            "omega {omega}: the SR half alone is within {sr_alone:.3e} of the full Coulomb \
             exchange, so the equal-coefficient identity is nearly vacuous here"
        );

        for c in [1.0_f64, 0.25, 0.167] {
            // Exactly the combination `subtract_rsh_exchange` forms, at sr = lr = c.
            let combined = c * &k_sr + c * &k_lr;
            let dev = combined
                .iter()
                .zip(k_coul.iter())
                .fold(0.0f64, |m, (a, b)| m.max((a - c * b).abs()));
            let rel = dev / (c * scale);
            eprintln!(
                "c_SR = c_LR = {c:<6} omega {omega:<5}: max|c(K_erfc + K_erf) - c*K_Coulomb| \
                 = {dev:.3e}  (rel {rel:.3e});  SR alone vs Coulomb: rel {sr_alone:.3e}"
            );
            worst = worst.max(rel);
            assert!(
                rel <= 1e-12,
                "c_SR = c_LR = {c} at omega = {omega}: the SR/LR combination does not \
                 reproduce the global-hybrid exchange (rel {rel:.3e} > 1e-12). \
                 erf + erfc = Coulomb is an IDENTITY in the integrals, so this is an \
                 erf/erfc wiring or coefficient error, not a fitting error."
            );
        }
    }
    eprintln!("worst relative deviation across 3 omega x 3 c: {worst:.3e} (bar 1e-12)");
}

/// Bar for [`rsh_response_matches_an_independent_four_centre_construction`]:
/// relative max-deviation of the DF `RshResponse` from the direct four-centre
/// SR/LR combination. DERIVED from both sides (OH / cc-pVDZ, jkfit aux,
/// c_SR = 0.167, c_LR = 1.0, ω = 0.30):
///
/// ```text
///   correct construction (DF fitting error)   7.7e-4
///   omega doubled                              4.20e-1   <- nearest artifact
///   erf/erfc swapped                           5.66e-1
///   c_SR for both                              6.95e-1
///   c_LR dropped                               8.34e-1
/// ```
///
/// 8e-3 is ~10x the measured value and 52x below the nearest artifact.
const TOL_DF_VS_DIRECT_SPLIT: f64 = 8e-3;

/// **The fast-tier test that pins WHICH kernel `RshResponse` builds.**
///
/// The in-crate finite-difference tests (`rsh_response::tests`) cannot do this,
/// and that is a property of their design, not a weakness of their bar: their
/// reference Fock is assembled through the SAME `RshResponse::exchange_response`
/// the matvec uses, so a mutation inside it (erf↔erfc swapped, `c_LR` dropped,
/// `c_SR` used for both, the wrong ω) changes the Fock and the Hessian
/// TOGETHER and the FD check sees a self-consistent derivative of a wrong
/// functional. They validate the matvec's STRUCTURE (δJ, the per-spin factor,
/// the MO projection) given the exchange operator. The equal-coefficient
/// identity test above works at the integral level and never calls
/// `RshResponse`.
///
/// This test supplies the INDEPENDENT construction: `RshResponse` (density
/// fitted, the production code) at `c_SR ≠ c_LR` against
/// `c_SR·K[erfc(ω)] + c_LR·K[erf(ω)]` from direct four-centre integrals, which
/// share nothing with `RshResponse` but the operator definitions. The two can
/// only differ by the DF fitting error.
///
/// Its discrimination is ASSERTED, not assumed: the same comparison is made for
/// the four artifact constructions — kernels swapped, `c_SR` for both
/// coefficients, `c_LR` dropped, ω doubled — each built from the same direct
/// integrals, and each must sit at least 10× the bar away from the DF response.
#[test]
fn rsh_response_matches_an_independent_four_centre_construction() {
    let mol = oh_doublet();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let ctx = ParallelContext::default();
    let n = prep.nbasis();
    let budget = ferric_core::memory::resolve_budget_bytes(None);
    let dfbs = basis::bundled(AUX).unwrap();
    let dfbs_prep = PreparedBasis::new(&mol, &dfbs).unwrap();

    let mut rng = Xorshift64(0x5851F42D4C957F2D);
    let mut dd = Array2::<f64>::from_shape_fn((n, n), |_| 0.01 * rng.next_f64());
    dd = 0.5 * (&dd + &dd.t());

    let kdirect = |op: Operator| -> Array2<f64> {
        let b = SchwarzBounds::compute(op, &prep).unwrap();
        let mut j = Array2::<f64>::zeros((n, n));
        let mut k = Array2::<f64>::zeros((n, n));
        ferric_scf::rhf::build_jk(&ctx, &prep, &b, 1e-14, &dd, &mut j, &mut k).unwrap();
        k
    };

    // wB97X-V's own split at its published omega (KMix sr = 0.167, lr = 1.0,
    // omega = 0.3, read from libxc in this session) — the production case.
    let (c_sr, c_lr, omega) = (0.167_f64, 1.0_f64, 0.30_f64);
    let sr = std::cell::RefCell::new(
        ferric_scf::df_k::DfK::new(Operator::erfc(omega), &prep, &dfbs_prep, budget).unwrap(),
    );
    let lr = std::cell::RefCell::new(
        ferric_scf::df_k::DfK::new(Operator::erf(omega), &prep, &dfbs_prep, budget).unwrap(),
    );
    let k_df = ferric_scf::rsh_response::RshResponse::new(&sr, &lr, c_sr, c_lr, omega)
        .exchange_response(&dd)
        .unwrap();

    let (k_erfc, k_erf) = (
        kdirect(Operator::erfc(omega)),
        kdirect(Operator::erf(omega)),
    );
    let (k_erfc2, k_erf2) = (
        kdirect(Operator::erfc(2.0 * omega)),
        kdirect(Operator::erf(2.0 * omega)),
    );
    let reference = c_sr * &k_erfc + c_lr * &k_erf;
    let scale = reference.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(
        scale > 1e-6,
        "reference response is ~zero ({scale:.3e}); vacuous"
    );
    let rel = |a: &Array2<f64>, b: &Array2<f64>| {
        a.iter()
            .zip(b.iter())
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
            / scale
    };

    let d_correct = rel(&k_df, &reference);
    let artifacts = [
        ("erf/erfc swapped", c_sr * &k_erf + c_lr * &k_erfc),
        ("c_SR for both", c_sr * &k_erfc + c_sr * &k_erf),
        ("c_LR dropped", c_sr * &k_erfc),
        ("omega doubled", c_sr * &k_erfc2 + c_lr * &k_erf2),
    ];
    eprintln!(
        "DF RshResponse vs direct c_SR*K[erfc] + c_LR*K[erf] (c_SR {c_sr}, c_LR {c_lr}, \
         omega {omega}): rel {d_correct:.3e}   (bar {TOL_DF_VS_DIRECT_SPLIT:.0e})"
    );
    for (name, wrong) in &artifacts {
        let d = rel(&k_df, wrong);
        eprintln!(
            "  artifact '{name}': DF response is rel {d:.3e} from it  ({:.1}x the bar)",
            d / TOL_DF_VS_DIRECT_SPLIT
        );
        assert!(
            d >= 10.0 * TOL_DF_VS_DIRECT_SPLIT,
            "the DF response is only rel {d:.3e} from the '{name}' construction — under 10x \
             the bar, so this test could not tell that artifact from the right answer"
        );
    }
    assert!(
        d_correct < TOL_DF_VS_DIRECT_SPLIT,
        "RshResponse differs from the independent four-centre SR/LR combination by rel \
         {d_correct:.3e} (bar {TOL_DF_VS_DIRECT_SPLIT:.0e})"
    );
}

// ---- FIXED, MACHINE-INDEPENDENT INPUTS (shared verbatim with the generator) ----
// Plain-Rust loops only (no LAPACK, no rayon), so every input is bit-identical on
// any IEEE-754 machine: seeded xorshift64 -> modified Gram-Schmidt for C,
// seeded sorted diagonal "orbital energies" for F_MO, seeded rotations k.
struct Rng292(u64);
impl Rng292 {
    fn next(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }
}
/// Modified Gram-Schmidt on the columns of a seeded random n x n matrix.
fn mgs_orthonormal(n: usize, seed: u64) -> ndarray::Array2<f64> {
    let mut r = Rng292(seed);
    let mut m = ndarray::Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            m[(i, j)] = r.next();
        }
    }
    for j in 0..n {
        for k in 0..j {
            let mut d = 0.0;
            for i in 0..n {
                d += m[(i, k)] * m[(i, j)];
            }
            for i in 0..n {
                m[(i, j)] -= d * m[(i, k)];
            }
        }
        let mut nn = 0.0;
        for i in 0..n {
            nn += m[(i, j)] * m[(i, j)];
        }
        let nn = nn.sqrt();
        for i in 0..n {
            m[(i, j)] /= nn;
        }
    }
    m
}
/// Diagonal MO "Fock" with sorted seeded energies: occupied in [-20, -0.3],
/// virtual in [0.05, 3.0]. hessian_matvec reads only the diagonal.
fn seeded_fock_mo(n: usize, nocc: usize, seed: u64) -> ndarray::Array2<f64> {
    let mut r = Rng292(seed);
    let mut occ: Vec<f64> = (0..nocc)
        .map(|_| -0.3 - 19.7 * (0.5 + 0.5 * r.next()))
        .collect();
    let mut vir: Vec<f64> = (nocc..n)
        .map(|_| 0.05 + 2.95 * (0.5 + 0.5 * r.next()))
        .collect();
    occ.sort_by(|a, b| a.partial_cmp(b).unwrap());
    vir.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut f = ndarray::Array2::<f64>::zeros((n, n));
    for (i, e) in occ.iter().chain(vir.iter()).enumerate() {
        f[(i, i)] = *e;
    }
    f
}
fn seeded_kappa(n: usize, nocc: usize, seed: u64) -> ndarray::Array2<f64> {
    let mut r = Rng292(seed);
    ndarray::Array2::<f64>::from_shape_fn((n - nocc, nocc), |_| 0.01 * r.next())
}
/// (sum, sum of squares) fingerprint, so drift in the generator is detected.
fn fingerprint(a: &ndarray::Array2<f64>) -> (f64, f64) {
    let mut s = 0.0;
    let mut q = 0.0;
    for v in a.iter() {
        s += v;
        q += v * v;
    }
    (s, q)
}
const G292_NOCC_A: usize = 5;
const G292_NOCC_B: usize = 4;
const G292_K_MIX_SR: f64 = 0.2; // hybrid-like, NOT pure HF
const G292_THRESH: f64 = 1e-12;
const G292_OOC_BUDGET: usize = 1 << 30; // fixed: band width only, never results
                                        // ---- end fixed inputs ----

/// Relative bar for `omega_zero_matvec_matches_the_pre_292_reference`.
///
/// MEASURED (this branch vs the pre-#292 reference, same box, release build):
/// rel **1.41e-16** — 14 of 130 elements differ in the last bit only. Mutants,
/// same test: ω = 0 coefficient `c_k × 0.999` → rel 1.83e-5; δK dropped →
/// rel 1.83e-2.
///
/// Bar 1e-12, deliberately NOT 10× the measured 1.4e-16. That figure is
/// same-machine; across machines the matvec's ERIs (libint built with or
/// without FMA) and its BLAS GEMMs legitimately differ in the last digits,
/// plausibly to ~1e-13 relative, so a 1.4e-15 bar would make CI fail at random
/// rather than guard anything. 1e-12 sits 1.8e7× below the weakest real kill.
const TOL_PRE292: f64 = 1e-12;

/// **Equivalence of the ω = 0 matvec to the pre-#292 code** — the
/// machine-independent CI guard that #292 left the ω = 0 path mathematically
/// unchanged.
///
/// The inputs are FIXED, not a converged SCF, because the SCF endpoint is what
/// moves across machines (OH's DIIS endpoint lands 1.16e-6 Ha apart on CI). They
/// are built in plain-Rust loops — a seeded xorshift64, modified Gram–Schmidt for
/// the α/β MO coefficients, sorted seeded diagonal energies, seeded κ — so they
/// are bit-identical on any IEEE-754 machine, and the reference stores a
/// (sum, sum-of-squares) fingerprint of each so a change to the generator fails
/// HERE rather than as an unexplained output difference. `k_mix_sr = 0.2`, a
/// hybrid-like coefficient, so the exchange term is scaled rather than the
/// pure-HF `1.0` that a dropped coefficient would also reproduce.
///
/// The reference (`testdata/reference/rsh_hessian/omega0_matvec_pre292.json`) was
/// generated by running THIS function's inputs through the pre-#292
/// `uhf_newton::hessian_matvec` at the commit recorded in its
/// `generating_commit` field (8637a5d5, the branch's merge-base with main: the
/// pre-#292 code including #313). The generator is this file's fixed-input block
/// verbatim, minus the `rsh` field the old struct did not have.
///
/// A tolerance, not bit equality, even though the two agree to the bit on the
/// machine that measured them: CI is a different machine (BLAS build, rayon
/// thread count, libint compile), and the matvec contains BLAS GEMMs and a
/// thread-count-dependent reduction width, so last-digit differences across
/// machines are expected and are not a change in the mathematics.
///
/// What this test CANNOT see, stated rather than implied: a pure reassociation
/// of the ω = 0 expression (e.g. `&dj - &dk_a * c_k`) changes no physics and
/// moves results at most by an ulp-scale amount, inside any tolerance a
/// cross-machine test can hold. Catching that is out of scope; it is not a
/// defect.
#[test]
fn omega_zero_matvec_matches_the_pre_292_reference() {
    let path = workspace_root().join("testdata/reference/rsh_hessian/omega0_matvec_pre292.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing reference {} ({e})", path.display()));
    let r: serde_json::Value = serde_json::from_str(&text).unwrap();
    let sha = r["generating_commit"].as_str().expect("generating_commit");
    assert!(
        sha.starts_with("8637a5d5"),
        "reference generated at an unexpected commit {sha}"
    );

    let mol = oh_doublet();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let ctx = ParallelContext::default();
    let n = prep.nbasis();
    assert_eq!(
        n as u64,
        r["nbasis"].as_u64().unwrap(),
        "basis size changed"
    );
    let c_a = mgs_orthonormal(n, 0x1F2E3D4C5B6A7988);
    let c_b = mgs_orthonormal(n, 0x2A3B4C5D6E7F8091);
    let f_a = seeded_fock_mo(n, G292_NOCC_A, 0x3C4D5E6F708192A3);
    let f_b = seeded_fock_mo(n, G292_NOCC_B, 0x4E5F60718293A4B5);
    let k_a = seeded_kappa(n, G292_NOCC_A, 0x5061728394A5B6C7);
    let k_b = seeded_kappa(n, G292_NOCC_B, 0x62738495A6B7C8D9);

    // The inputs must be the ones the reference was generated from.
    for (name, a) in [
        ("c_a", &c_a),
        ("c_b", &c_b),
        ("f_a_mo", &f_a),
        ("f_b_mo", &f_b),
        ("k_a", &k_a),
        ("k_b", &k_b),
    ] {
        let (s0, q0) = fingerprint(a);
        let fp = &r["input_fingerprints"][name];
        let (s1, q1) = (fp[0].as_f64().unwrap(), fp[1].as_f64().unwrap());
        assert!(
            (s0 - s1).abs() <= 1e-14 * s1.abs().max(1.0)
                && (q0 - q1).abs() <= 1e-14 * q1.abs().max(1.0),
            "input '{name}' differs from the one the reference was generated from \
             ({s0:.17e},{q0:.17e}) vs ({s1:.17e},{q1:.17e}): the generator drifted"
        );
    }

    let inputs = ferric_scf::uhf_newton::UhfNewtonInputs {
        prep: &prep,
        bounds: &bounds,
        c_a: &c_a,
        c_b: &c_b,
        f_a_mo: &f_a,
        f_b_mo: &f_b,
        nocc_a: G292_NOCC_A,
        nocc_b: G292_NOCC_B,
        k_mix_sr: G292_K_MIX_SR,
        rsh: None,
        fxc: None,
        thresh: G292_THRESH,
        ooc_budget: G292_OOC_BUDGET,
    };
    let pool = ferric_scf::engine_pool::EnginePool::new(
        bounds.op,
        &prep,
        ferric_integrals::engine_pool::eri_precision(),
    )
    .unwrap();
    let (h_a, h_b) =
        ferric_scf::uhf_newton::hessian_matvec(&ctx, &inputs, &k_a, &k_b, &pool).unwrap();

    let want = |key: &str| -> Vec<f64> {
        r[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect()
    };
    let (ra, rb) = (want("h_a"), want("h_b"));
    assert_eq!(ra.len(), h_a.len(), "h_a size changed");
    assert_eq!(rb.len(), h_b.len(), "h_b size changed");
    let got: Vec<f64> = h_a.iter().chain(h_b.iter()).cloned().collect();
    let refv: Vec<f64> = ra.iter().chain(rb.iter()).cloned().collect();
    let scale = refv.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(
        scale > 1e-6,
        "reference block is ~zero ({scale:.3e}); the comparison is vacuous"
    );
    let dmax = got
        .iter()
        .zip(refv.iter())
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    let rel = dmax / scale;
    let nbit = got
        .iter()
        .zip(refv.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    eprintln!(
        "omega = 0 matvec vs pre-#292 reference ({sha:.8}): max|d| = {dmax:.3e}, rel {rel:.3e} \
         (scale {scale:.3e}); {nbit} of {} elements differ in any bit; bar {TOL_PRE292:.0e}",
        got.len()
    );
    assert!(
        rel <= TOL_PRE292,
        "the omega = 0 Hessian matvec differs from the pre-#292 code by rel {rel:.3e} \
         (bar {TOL_PRE292:.0e}): #292 changed the omega = 0 mathematics"
    );
}

/// **Wiring at ω = 0: the production solver hands the matvec `rsh: None`.**
///
/// `omega_zero_matvec_matches_the_pre_292_reference` fixes `rsh: None` itself, so
/// it cannot see a solver that wrongly passes `Some(rsh)` at ω = 0 (e.g.
/// `driver::prepare` building the SR/LR fitters for `omega >= 0.0`), which would
/// silently swap the direct four-centre exchange response for a density-fitted
/// `erfc(0)`/`erf(0)` one. This test closes that.
///
/// UHF/NH₂ (²B₁) / 6-31G with `check_stability` set: the production path
/// computes λ_min internally. NOT OH: OH is a ²Π radical whose λ_min is an exact
/// zero mode (rotating the π hole about the axis is a symmetry of the energy for
/// ANY exchange operator), so a density-fitted miswiring would leave it at zero
/// and this comparison would be blind — the first draft of this test used OH and
/// measured |d| = 0 against λ_min = −1.5e-10 for exactly that reason. NH₂ has a
/// non-degenerate ground state and no such mode. The test then recomputes λ_min at the SAME converged MOs and
/// Fock with an explicit `rsh: None`, `k_mix_sr = 1.0` matvec. Both read the
/// one SCF endpoint this run produced, so the comparison is machine-independent
/// even though that endpoint is not. At baseline both run identical code on
/// identical inputs; under the miswiring the production side carries the DF
/// fitting error. The bar is derived from both measurements at `TOL_WIRING`.
#[test]
fn omega_zero_production_stability_takes_the_coulomb_arm() {
    use ferric_scf::stability::{uhf_internal_stability, StabilityConfig};
    let xyz = workspace_root().join("testdata/molecules/validation/nh2.xyz");
    let mol = Molecule::load_xyz_with_charge(xyz.to_str().unwrap(), 0, 2).unwrap();
    let bs = basis::bundled("6-31g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        check_stability: true,
        ..Default::default()
    };
    let res = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
    assert!(res.converged);
    let prod = res
        .stability
        .as_ref()
        .expect("check_stability produced no verdict");

    let c_a = res.mos_alpha.clone();
    let c_b = res.mos_beta.clone().unwrap();
    let f_a_mo = c_a.t().dot(&res.fock_alpha).dot(&c_a);
    let f_b_mo = c_b.t().dot(res.fock_beta.as_ref().unwrap()).dot(&c_b);
    let nelec = mol.nelec() as usize;
    let two_s = mol.multiplicity - 1;
    let inputs = ferric_scf::uhf_newton::UhfNewtonInputs {
        prep: &prep,
        bounds: &bounds,
        c_a: &c_a,
        c_b: &c_b,
        f_a_mo: &f_a_mo,
        f_b_mo: &f_b_mo,
        nocc_a: (nelec + two_s) / 2,
        nocc_b: (nelec - two_s) / 2,
        k_mix_sr: 1.0,
        rsh: None,
        fxc: None,
        thresh: cfg.integral_thresh,
        ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
    };
    let mine = uhf_internal_stability(&ctx, &inputs, &StabilityConfig::default()).unwrap();
    // Not a symmetry zero mode: a vanishing lambda_min here would make the
    // comparison blind to the miswiring it exists to catch.
    assert!(
        mine.lowest_eigenvalue.abs() > 1e-3,
        "NH2 lambda_min {:.3e} is ~0; the wiring comparison would be blind",
        mine.lowest_eigenvalue
    );
    let d = (prod.lowest_eigenvalue - mine.lowest_eigenvalue).abs();
    eprintln!(
        "omega = 0 wiring: production lambda_min {:+.12e} vs explicit rsh: None {:+.12e}, \
         |d| {d:.3e} (bar {TOL_WIRING:.0e})",
        prod.lowest_eigenvalue, mine.lowest_eigenvalue
    );
    assert!(
        d <= TOL_WIRING,
        "at omega = 0 the production stability path disagrees with an explicit rsh: None \
         matvec at the same state by {d:.3e}: the solver is not handing the Hessian the \
         plain-Coulomb exchange arm"
    );
}

/// Absolute bar (Ha) for `omega_zero_production_stability_takes_the_coulomb_arm`.
///
/// MEASURED on NH₂ / 6-31G: baseline |Δλ_min| = **0** exactly (both sides run
/// the same code on the same converged state in one process, so this holds on
/// any machine up to run-to-run nondeterminism, of which the solver has none);
/// with `driver::prepare` miswired to build the SR/LR fitters at ω = 0, the
/// production λ_min moves +7.500756250e-2 → +7.500817900e-2, |Δ| = 6.165e-7 —
/// the density-fitting error of an `erfc(0)`/`erf(0)` response. Bar 1e-8, 62×
/// below that signal.
const TOL_WIRING: f64 = 1e-8;

fn workspace_root() -> std::path::PathBuf {
    use std::path::{Path, PathBuf};
    let looks_like_root = |p: &Path| {
        p.join("Cargo.toml").is_file() && p.join("testdata").is_dir() && p.join("crates").is_dir()
    };
    if let Ok(cwd) = std::env::current_dir() {
        let mut here: Option<&Path> = Some(cwd.as_path());
        while let Some(p) = here {
            if looks_like_root(p) {
                return p.to_path_buf();
            }
            here = p.parent();
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("manifest dir should be <root>/crates/ferric-scf")
        .to_path_buf()
}
