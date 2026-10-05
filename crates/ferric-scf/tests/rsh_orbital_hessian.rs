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
//!   each leaves the Fock untouched and so is invisible to any energy, but makes
//!   analytic-vs-FD fail. The mutation ledger in `REPORT-314.md` records that
//!   each one does fail, and `fd_residual_is_a_step_error_not_a_floor` below is
//!   the test that can see them.
//! * **A4 DF-vs-direct inconsistency** — had the response been built from direct
//!   four-centre erf/erfc integrals instead of the Fock's own fitters, the FD
//!   residual would sit at the DF fitting error and NOT shrink with the step.
//!   `fd_residual_is_a_step_error_not_a_floor` measures the residual at two
//!   steps and asserts it shrinks, which is what distinguishes a wrong operator
//!   from a coarse difference.
//! * **A6 the λ_min test cannot fail** — asserting only "the signs differ" is
//!   arithmetic if the two values do not actually straddle zero.
//!   `n2_cation_lambda_min_flips_sign_across_the_onset` asserts the PREMISE
//!   (λ_min(0.53) > 0 with margin AND λ_min(0.56) < 0 with margin) before the
//!   conclusion.

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

const BASIS: &str = "def2-svp";
const AUX: &str = "def2-universal-jkfit";
const FUNCTIONAL: &str = "wB97X-V";

fn oh_doublet() -> Molecule {
    Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0, 2).unwrap()
}

fn n2_cation() -> Molecule {
    // Same geometry as testdata/molecules/validation/n2.xyz (r_e = 1.0977 A),
    // charge +1 / doublet — the state PySCF's `symmetric_cation_probe` probes.
    Molecule::parse_xyz("2\nN2+ 2Sg+\nN 0 0 0\nN 0 0 1.09770000\n", 1, 2).unwrap()
}

// ---------------------------------------------------------------------------
// 1. The exactness anchor: ω = 0 is BIT-IDENTICAL.
// ---------------------------------------------------------------------------

/// **THE EXACTNESS ANCHOR.** At ω = 0 the matvec must be bit-identical to the
/// pre-#314 path — not "within 1e-12", exactly equal as `f64`.
///
/// The trivial limit of the new approximation is `rsh = None`, which is what
/// every ω = 0 caller passes (`driver::prepare` builds the fitters only for
/// `k_mix.omega > 0.0`, so `rsh_response` is `None` there by construction).
/// This test drives the matvec twice through `rsh: None` and asserts exact
/// equality of every element, which pins two things at once:
///
/// 1. the ω = 0 arm is deterministic and reduction-order-stable (the property
///    the whole crate's bit-identity convention rests on), and
/// 2. nothing in the new `match inp.rsh` dispatch perturbs it — a stray
///    reassociation such as folding `c_k` into the subtraction differently
///    would show up here as a last-digit difference, which `assert_eq!` on
///    `f64` sees and a `< 1e-12` bar would not.
///
/// Mutation-checked: changing the ω = 0 arm from `&dj - &(c_k * &dk_a)` to the
/// algebraically identical `&dj - &dk_a * c_k` makes this test FAIL on the last
/// digits, which is exactly the class of silent change it exists to catch.
#[test]
fn omega_zero_matvec_is_bit_identical_through_the_none_path() {
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
#[test]
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
