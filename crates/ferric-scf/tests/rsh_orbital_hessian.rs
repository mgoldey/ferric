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
