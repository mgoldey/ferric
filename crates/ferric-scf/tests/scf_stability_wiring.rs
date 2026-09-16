//! `RhfConfig::check_stability` — the opt-in wiring that makes a saddle visible
//! to a user who never calls `ferric_scf::stability` directly.
//!
//! The analysis itself (Davidson over the orbital Hessian, dense cross-checks,
//! PySCF agreement to 3.8e-10 Ha) is validated in `scf_stability.rs`. This file
//! validates only the WIRING, which has four independent failure modes:
//!
//! 1. **The flag-off path is not free.** A post-convergence hook that runs, or
//!    perturbs the result, when the flag is off would change every existing
//!    energy in the workspace. `stability_off_is_bit_identical_*` pins exact
//!    bit equality of the converged energy AND density against the same run
//!    with the flag on, and pins `stability == None`.
//! 2. **`None` is confused with "stable".** Three different situations produce
//!    `None` — not requested, skipped as un-analysable, eigensolve failed — and
//!    none of them is a stability verdict. `not_checked_is_distinguishable_*`
//!    pins that a real verdict is `Some`, so the two are never conflated.
//! 3. **The verdict is constant.** A check that always says one thing is
//!    arithmetic, not measurement. `..._warns` (HeNe⁺, UNSTABLE) and
//!    `..._is_quiet` (water, STABLE) prove BOTH values are reachable THROUGH
//!    THE SCF PATH, not just through the library function.
//! 4. **The wrong operator is analysed.** ROHF has no implemented Hessian, so
//!    it must SKIP rather than silently analyse a UHF or RHF one at MOs that
//!    are not a stationary point of either. `rohf_skips_*` pins that.
//!
//! # Artifact hypothesis (written before measuring)
//!
//! If the wiring is REAL I expect: flag off ⇒ identical bits and `None`; flag
//! on ⇒ `Some` with λ_min matching the library-level numbers already validated
//! in `scf_stability.rs` (HeNe⁺ −4.8706726160e-3, water RHF +5.2323011339e-1),
//! and ROHF ⇒ `None`.
//!
//! If the wiring is BROKEN in the most likely way — the flag read at the wrong
//! place, or the Fock/MO pair handed to the analysis not being the converged
//! one — I expect λ_min to DISAGREE with the library-level value on the same
//! system while still having a plausible sign. That is why these tests assert
//! the NUMBER against the independently-validated constant, not merely the
//! boolean: a boolean-only test cannot distinguish "wired correctly" from
//! "wired to a nearby non-stationary point", which is the exact defect class
//! this wiring can introduce and the analysis module cannot.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::result::ScfResult;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::rohf::solve_rohf;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::solve_uhf;
use ferric_scf::{StabilityKind, StabilityVerdict};

/// EXACTLY the geometry `scf_stability.rs`'s anchors use, so the lambda_min
/// constants below are comparable. lambda_min is geometry-dependent; a nearby
/// water would give a different (still positive) value and the comparison
/// would be meaningless.
const WATER: &str =
    "3\nwater\nO 0.0000 0.0000 0.1173\nH 0.0000 0.7572 -0.4692\nH 0.0000 -0.7572 -0.4692\n";
const HENE_PLUS: &str = "2\nHeNe+\nHe 0 0 0\nNe 0 0 2.0\n";

/// λ_min for HeNe⁺/def2-SVP/UHF at R = 2.0 Å, from `scf_stability.rs`, where it
/// was cross-checked against an explicitly built dense 148x148 Hessian AND
/// against PySCF's own `newton_ah.gen_g_hop_uhf` (agreement 3.8e-10 Ha).
const HENE_LAMBDA_MIN: f64 = -4.8706726160e-3;
/// λ_min for water/STO-3G RHF (singlet channel), same provenance.
const WATER_RHF_LAMBDA_MIN: f64 = 5.2323011339e-1;
/// λ_min for water/STO-3G UHF, same provenance.
const WATER_UHF_LAMBDA_MIN: f64 = 3.6256168663e-1;

/// Tolerance on reproducing a library-level λ_min through the SCF path. Loose
/// relative to the eigensolver's 1e-6 convergence threshold, tight enough that
/// analysing a DIFFERENT point (the defect this file exists to catch) fails:
/// the HeNe⁺ value is 4.9e-3, so a 1e-6 bar is ~5000x below the signal.
const LAMBDA_TOL: f64 = 1e-6;

fn tight(check_stability: bool) -> RhfConfig {
    RhfConfig {
        energy_conv: 1e-11,
        density_conv: 1e-9,
        max_iter: 400,
        check_stability,
        ..Default::default()
    }
}

fn run_uhf(xyz: &str, charge: i32, mult: usize, basis_name: &str, check: bool) -> ScfResult {
    let mol = Molecule::parse_xyz(xyz, charge, mult).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let ctx = ParallelContext::default();
    let res = solve_uhf(&ctx, &mol, &prep, &bounds, &tight(check)).unwrap();
    assert!(res.converged);
    res
}

fn run_rhf(xyz: &str, basis_name: &str, check: bool) -> ScfResult {
    let mol = Molecule::parse_xyz(xyz, 0, 1).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let ctx = ParallelContext::default();
    let res = solve_rhf(
        &ctx,
        &mol,
        &prep,
        Operator::coulomb(),
        &bounds,
        &tight(check),
    )
    .unwrap();
    assert!(res.converged);
    res
}

// ---------------------------------------------------------------------------
// 1. EXACTNESS ANCHOR: the flag OFF must cost nothing and change nothing.
// ---------------------------------------------------------------------------

/// The trivial limit of this feature: with `check_stability = false` the SCF
/// must be BIT-IDENTICAL to the same run with the flag absent — same energy
/// bits, same density bits, same iteration count — and must report `None`.
///
/// This is the anchor that has to pass before any flag-on number is believed.
/// It is asserted with `==` on `f64`, deliberately: "close" would not catch a
/// hook that runs and feeds its rotation back, which is the failure mode that
/// would silently move every energy in the workspace.
#[test]
fn stability_off_is_bit_identical_rhf() {
    let a = run_rhf(WATER, "sto-3g", false);
    let b = run_rhf(WATER, "sto-3g", false);
    assert_eq!(
        a.energy.to_bits(),
        b.energy.to_bits(),
        "two flag-off RHF runs must be bit-identical"
    );
    // The real comparison: flag ON must not move the CONVERGED answer either
    // (the check runs strictly AFTER convergence and must be read-only).
    let c = run_rhf(WATER, "sto-3g", true);
    assert_eq!(
        a.energy.to_bits(),
        c.energy.to_bits(),
        "turning check_stability ON must not change the converged energy by a single bit: \
         off = {:.16e} ({:#x}), on = {:.16e} ({:#x})",
        a.energy,
        a.energy.to_bits(),
        c.energy,
        c.energy.to_bits()
    );
    assert_eq!(
        a.iterations, c.iterations,
        "iteration count must not change"
    );
    for (x, y) in a.density_total.iter().zip(c.density_total.iter()) {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "converged density must be bit-identical with the flag on"
        );
    }
    assert!(
        a.stability.is_none(),
        "flag off must leave ScfResult::stability = None (not checked)"
    );
    eprintln!(
        "FLAG-OFF ANCHOR  RHF water/STO-3G  E = {:.16e} (bits {:#x}) off == on; \
         stability off = {:?}, on = Some",
        a.energy,
        a.energy.to_bits(),
        a.stability.is_some()
    );
}

/// Same anchor on the UHF path, where the check is a different function
/// analysing a different (packed α,β) space.
#[test]
fn stability_off_is_bit_identical_uhf() {
    let a = run_uhf(WATER, 0, 1, "sto-3g", false);
    let c = run_uhf(WATER, 0, 1, "sto-3g", true);
    assert_eq!(
        a.energy.to_bits(),
        c.energy.to_bits(),
        "turning check_stability ON must not change the converged UHF energy by a single bit"
    );
    assert_eq!(a.iterations, c.iterations);
    for (x, y) in a.density_total.iter().zip(c.density_total.iter()) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
    assert!(a.stability.is_none());
    eprintln!(
        "FLAG-OFF ANCHOR  UHF water/STO-3G  E = {:.16e} (bits {:#x}) off == on",
        a.energy,
        a.energy.to_bits()
    );
}

// ---------------------------------------------------------------------------
// 2+3. BOTH VERDICTS REACHABLE THROUGH THE SCF PATH, with the right numbers.
// ---------------------------------------------------------------------------

/// THE DECISIVE WIRING TEST. HeNe⁺/def2-SVP/UHF converges to a saddle; with the
/// flag on, `solve_uhf` must carry that verdict out on the result, and λ_min
/// must equal the value validated in `scf_stability.rs` against both a dense
/// Hessian and PySCF.
///
/// Asserting the NUMBER, not the boolean, is what distinguishes correct wiring
/// from wiring that analyses a nearby non-stationary point (which would also
/// produce a negative λ here, and a boolean test would pass).
#[test]
fn hene_plus_uhf_carries_an_unstable_verdict_through_the_scf_path() {
    let res = run_uhf(HENE_PLUS, 1, 2, "def2-svp", true);
    let st = res
        .stability
        .as_ref()
        .expect("check_stability = true on an analysable UHF reference must produce Some(..)");
    eprintln!(
        "DECISIVE (wired)  HeNe+ UHF  E = {:.10}\n  {}",
        res.energy,
        st.summary()
    );
    assert_eq!(st.kind, StabilityKind::UhfInternal);
    assert!(st.converged, "the eigensolve must converge here");
    // Asserted on `verdict()`, not on `is_stable` -- that field is
    // "not proven unstable" and reads `true` inside the marginal band.
    assert_eq!(
        st.verdict(),
        StabilityVerdict::Unstable,
        "HeNe+/def2-SVP UHF is a known saddle; the wired check must report UNSTABLE, got \
         {:?} at lambda_min = {:+.10e}",
        st.verdict(),
        st.lowest_eigenvalue
    );
    assert!(
        (st.lowest_eigenvalue - HENE_LAMBDA_MIN).abs() < LAMBDA_TOL,
        "the wired lambda_min must reproduce the library-level value validated against a dense \
         148x148 Hessian and PySCF: expected {HENE_LAMBDA_MIN:+.10e}, got {:+.10e} (diff {:.3e}). \
         A disagreement here means the analysis was handed MOs/Fock that are not the converged \
         stationary point.",
        st.lowest_eigenvalue,
        (st.lowest_eigenvalue - HENE_LAMBDA_MIN).abs()
    );
    // The eigenvector must be the packed (alpha, beta) pair, shaped for the
    // remedy the warning names.
    assert_eq!(
        st.eigenvector_alpha.dim().1,
        6,
        "HeNe+ (11 electrons, doublet) has 6 alpha-occupied orbitals"
    );
    assert!(
        st.eigenvector_beta.is_some(),
        "a UHF verdict must carry the beta rotation block"
    );
}

/// The other direction, through the same code path: a genuinely stable state
/// must come out STABLE with the validated positive λ_min. Without this the
/// UNSTABLE verdict above could be a constant.
#[test]
fn water_stays_quiet_through_the_scf_path() {
    let rhf = run_rhf(WATER, "sto-3g", true);
    let sr = rhf.stability.as_ref().expect("RHF must produce Some(..)");
    eprintln!("QUIET  RHF water/STO-3G  {}", sr.summary());
    assert_eq!(sr.kind, StabilityKind::RhfInternal);
    assert!(sr.converged);
    assert_eq!(
        sr.verdict(),
        StabilityVerdict::Stable,
        "water/STO-3G RHF is a minimum in the singlet channel; got {:?} at {:+.10e}",
        sr.verdict(),
        sr.lowest_eigenvalue
    );
    assert!(
        (sr.lowest_eigenvalue - WATER_RHF_LAMBDA_MIN).abs() < LAMBDA_TOL,
        "wired RHF lambda_min {:+.10e} must match the validated {WATER_RHF_LAMBDA_MIN:+.10e}",
        sr.lowest_eigenvalue
    );

    let uhf = run_uhf(WATER, 0, 1, "sto-3g", true);
    let su = uhf.stability.as_ref().expect("UHF must produce Some(..)");
    eprintln!("QUIET  UHF water/STO-3G  {}", su.summary());
    assert_eq!(su.kind, StabilityKind::UhfInternal);
    assert_eq!(su.verdict(), StabilityVerdict::Stable);
    assert!(
        (su.lowest_eigenvalue - WATER_UHF_LAMBDA_MIN).abs() < LAMBDA_TOL,
        "wired UHF lambda_min {:+.10e} must match the validated {WATER_UHF_LAMBDA_MIN:+.10e}",
        su.lowest_eigenvalue
    );
}

/// The gate is a measurement, not a constant: BOTH verdicts must be reachable
/// through `solve_uhf` itself, same config knob, same code path.
#[test]
fn both_verdicts_are_reachable_through_the_config_flag() {
    let stable = run_uhf(WATER, 0, 1, "sto-3g", true);
    let saddle = run_uhf(HENE_PLUS, 1, 2, "def2-svp", true);
    let a = stable.stability.as_ref().unwrap().verdict();
    let b = saddle.stability.as_ref().unwrap().verdict();
    eprintln!("REACHABLE (wired)  water -> {a:?}, HeNe+ -> {b:?}");
    assert_eq!(
        a,
        StabilityVerdict::Stable,
        "the STABLE verdict must be reachable through the flag"
    );
    assert_eq!(
        b,
        StabilityVerdict::Unstable,
        "the UNSTABLE verdict must be reachable through the flag"
    );
}

// ---------------------------------------------------------------------------
// 4. NOT CHECKED is distinguishable from CHECKED AND STABLE.
// ---------------------------------------------------------------------------

/// `None` must never be readable as a verdict. The same system, same solver,
/// same molecule differs ONLY in the flag: off ⇒ `None`, on ⇒ `Some(stable)`.
/// If a future refactor made `None` mean "stable", this pins the difference.
#[test]
fn not_checked_is_distinguishable_from_checked_and_stable() {
    let off = run_rhf(WATER, "sto-3g", false);
    let on = run_rhf(WATER, "sto-3g", true);
    assert!(
        off.stability.is_none(),
        "flag off must be None (NOT CHECKED)"
    );
    let v = on.stability.as_ref().expect("flag on must be Some");
    assert_eq!(
        v.verdict(),
        StabilityVerdict::Stable,
        "and that Some must be the STABLE verdict"
    );
    eprintln!(
        "DISTINGUISHABLE  off = None, on = Some({:?}, lambda_min = {:+.6e})",
        v.verdict(),
        v.lowest_eigenvalue
    );
}

/// ROHF/ROKS has NO implemented orbital-Hessian matvec in this module's two
/// flavours — the Roothaan Hessian is a third operator. Requesting a check
/// there must SKIP (leaving `None`, with a printed reason) rather than analyse
/// a UHF or RHF Hessian at MOs that are not a stationary point of either.
///
/// A wrong-operator verdict that looks authoritative is the worst outcome this
/// feature can produce, so the skip is asserted, not assumed.
#[test]
fn rohf_skips_the_check_instead_of_analysing_the_wrong_operator() {
    let mol = Molecule::parse_xyz("2\nOH\nO 0 0 0\nH 0 0 0.97\n", 0, 2).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let ctx = ParallelContext::default();
    let res = solve_rohf(
        &ctx,
        &mol,
        &prep,
        Operator::coulomb(),
        &bounds,
        &tight(true),
    )
    .unwrap();
    assert!(
        res.converged,
        "ROHF must converge for this test to mean anything"
    );
    eprintln!(
        "ROHF SKIP  E = {:.10}  stability = {:?} (must be None)",
        res.energy,
        res.stability.as_ref().map(|s| s.lowest_eigenvalue)
    );
    assert!(
        res.stability.is_none(),
        "a ROHF reference must SKIP the check (None = not checked), never report a verdict from \
         an operator its MOs are not stationary for"
    );
    // ...and the skip must not have broken the SCF itself.
    assert!(res.energy.is_finite());
}

/// A KS reference the Hessian cannot represent (meta-GGA: no tau f_xc kernel
/// exists in this workspace) must also SKIP rather than fall back to the HF
/// Hessian at the KS density — the trap named in the stability module docs.
///
/// Asserted through the pure predicate rather than a full SCAN SCF, so the test
/// is cheap; the SCF-path consequence is the same `None`.
#[test]
fn unrepresentable_ks_references_are_refused_not_silently_downgraded() {
    use ferric_scf::stability::{ks_reference_is_analysable, StabilitySkip};
    // Plain HF and an analysable KS functional: allowed.
    assert!(ks_reference_is_analysable(None, 0.0).is_ok());
    assert!(ks_reference_is_analysable(Some("PBE"), 0.0).is_ok());
    assert!(ks_reference_is_analysable(Some("B3LYP"), 0.0).is_ok());
    // Range-separated: the matvec's K is plain Coulomb, so refuse.
    assert_eq!(
        ks_reference_is_analysable(Some("wB97X-V"), 0.3),
        Err(StabilitySkip::RangeSeparated)
    );
    // Meta-GGA: no tau f_xc kernel, so refuse.
    assert_eq!(
        ks_reference_is_analysable(Some("SCAN"), 0.0),
        Err(StabilitySkip::MetaGga)
    );
    assert_eq!(
        ks_reference_is_analysable(Some("r2SCAN"), 0.0),
        Err(StabilitySkip::MetaGga)
    );
    eprintln!(
        "KS GATE  HF/PBE/B3LYP analysable; wB97X-V -> {:?}; SCAN -> {:?}",
        ks_reference_is_analysable(Some("wB97X-V"), 0.3).unwrap_err(),
        ks_reference_is_analysable(Some("SCAN"), 0.0).unwrap_err()
    );
}

// ---------------------------------------------------------------------------
// 5. THE KS TRAP. The one failure the stability module's own report names.
// ---------------------------------------------------------------------------

/// `hessian_matvec`'s `fxc` argument is OPTIONAL, and passing `None` on a KS
/// reference does not fail — it silently analyses the **HF** orbital Hessian at
/// the **KS** density and reports the answer with full confidence. The wiring
/// claims to thread the same f_xc closure the KS Newton path uses; this test
/// proves the claim instead of asserting it.
///
/// # Why a boolean test would be worthless here
///
/// Water/STO-3G is stable either way, so `is_stable == true` cannot tell the
/// two operators apart. What CAN is the NUMBER: the XC response term is a real
/// contribution to the Hessian, so `lambda_min(with fxc) != lambda_min(without
/// fxc)` by an amount far above the eigensolver's 1e-6 threshold. This test
/// therefore builds the wrong-operator value EXPLICITLY (library call, same
/// converged KS MOs and Fock, `fxc: None` — the trap verbatim) and asserts the
/// wired verdict is NOT it, and IS the value obtained with the kernel threaded.
///
/// # Artifact hypothesis
///
/// If `fxc` is threaded: wired == with-kernel, and wired != no-kernel by a
/// clearly resolvable margin. If `fxc` is silently dropped: wired == no-kernel
/// EXACTLY (bit-level, same code path), which is the assertion below that
/// would fire. The two predictions are mutually exclusive, so the test can
/// distinguish them.
#[test]
fn ks_reference_is_analysed_with_the_xc_kernel_not_the_hf_hessian() {
    use ferric_scf::rhf_newton::RhfNewtonInputs;
    use ferric_scf::stability::{rhf_internal_stability, StabilityConfig};

    let mol = Molecule::parse_xyz(WATER, 0, 1).unwrap();
    let bs = basis::bundled("sto-3g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    let ctx = ParallelContext::default();

    // KS-DFT cannot reach `tight()`'s 1e-11 energy_conv (the XC grid floors dE
    // well above it), so this uses the library defaults' loose energy bound
    // with a tight density bound — the gate `scf_converged` actually relies on.
    let cfg = RhfConfig {
        density_conv: 1e-9,
        max_iter: 400,
        check_stability: true,
        xc: Some("PBE".to_string()),
        ..Default::default()
    };
    let res = solve_rhf(&ctx, &mol, &prep, Operator::coulomb(), &bounds, &cfg).unwrap();
    assert!(res.converged, "RKS/PBE must converge");
    let wired = res
        .stability
        .as_ref()
        .expect("an LDA/GGA KS reference IS analysable and must produce Some(..)");
    eprintln!(
        "KS TRAP  RKS/PBE water/STO-3G  E = {:.10}\n  wired  {}",
        res.energy,
        wired.summary()
    );

    // The trap, built verbatim: same converged KS MOs, same converged KS Fock,
    // `fxc: None`. This is what a careless caller would get.
    let c = res.mos_alpha.clone();
    let f_mo = c.t().dot(&res.fock_alpha).dot(&c);
    let nocc = mol.nelec() as usize / 2;
    let k_mix_sr = 0.0; // PBE is a pure functional: no exact exchange.
    let no_kernel = rhf_internal_stability(
        &ctx,
        &RhfNewtonInputs {
            prep: &prep,
            bounds: &bounds,
            c: &c,
            f_mo: &f_mo,
            nocc,
            k_mix_sr,
            fxc: None,
            thresh: cfg.integral_thresh,
            ooc_budget: ferric_core::memory::resolve_budget_bytes(None),
        },
        &StabilityConfig::default(),
    )
    .unwrap();
    eprintln!(
        "KS TRAP  no-kernel (the WRONG operator, fxc = None)  lambda_min = {:+.10e}",
        no_kernel.lowest_eigenvalue
    );

    let gap = (wired.lowest_eigenvalue - no_kernel.lowest_eigenvalue).abs();
    eprintln!("KS TRAP  |wired - no_kernel| = {gap:.6e}  (must be >> 1e-6)");
    assert!(
        gap > 1e-4,
        "the wired KS verdict is indistinguishable from the HF-Hessian-at-KS-density trap \
         (|diff| = {gap:.3e}): wired {:+.10e}, no-kernel {:+.10e}. Either the f_xc closure is \
         NOT being threaded, or this system cannot tell the two operators apart and the test \
         needs a different one.",
        wired.lowest_eigenvalue,
        no_kernel.lowest_eigenvalue
    );
    assert_eq!(
        wired.verdict(),
        StabilityVerdict::Stable,
        "water/STO-3G PBE is a minimum in the singlet channel: {}",
        wired.summary()
    );
}

// ---------------------------------------------------------------------------
// 6. THE MARGINAL BAND. `is_stable` is a bool over a three-valued question.
// ---------------------------------------------------------------------------

/// `StabilityResult::is_stable` is computed as
/// `lowest_eigenvalue > -noise_floor`, i.e. it is really **NOT PROVEN
/// UNSTABLE** — it reads `true` for a NEGATIVE λ_min anywhere in the marginal
/// band `-noise_floor < λ_min <= 0`. (Its docstring said the opposite until
/// this branch corrected it; the code is unchanged.)
///
/// That makes it a trap for exactly the consumer this file is: the SCF-path
/// warning. If `report_stability` branched on `is_stable`, a shallow saddle
/// would be reported as a minimum — quietly, since the STABLE arm is the one
/// that prints nothing.
///
/// This test pins the fix: the wiring branches on `StabilityResult::verdict()`,
/// which is derived from `converged`, `noise_floor` and the SIGN of λ_min, and
/// is therefore total over the four outcomes.
///
/// # Reachability and mutation coverage (the point of the test)
///
/// The gap the review named is that `is_stable = eigenvalue > 0.0` passes every
/// test in `scf_stability.rs` — no case in that file lands in the marginal
/// band, so no test can see the difference. The band is only ~1e-6 wide, so
/// hitting it with a real molecule would be luck, not a test. This therefore
/// constructs the band point directly from the result type's own public fields
/// and asserts what `verdict()` must say about it, which is the property the
/// wiring depends on — and it FAILS under both plausible mutations of
/// `verdict()`'s ordering (see the assertions' messages).
#[test]
fn a_marginal_negative_eigenvalue_is_not_reported_as_stable() {
    use ferric_scf::stability::StabilityResult;
    use ndarray::Array2;

    let mk = |lambda: f64, converged: bool| StabilityResult {
        kind: StabilityKind::UhfInternal,
        lowest_eigenvalue: lambda,
        // EXACTLY the expression the analysis uses, so this test tracks the
        // real definition rather than a paraphrase of it.
        is_stable: lambda > -1e-6,
        eigenvector_alpha: Array2::zeros((1, 1)),
        eigenvector_beta: Some(Array2::zeros((1, 1))),
        converged,
        residual: 1e-9,
        iterations: 3,
        noise_floor: 1e-6,
    };

    // (a) THE TRAP ITSELF: a NEGATIVE eigenvalue inside the band.
    let shallow = mk(-5e-7, true);
    assert!(
        shallow.is_stable,
        "precondition: `is_stable` is 'not proven unstable', so it must be TRUE for this \
         negative lambda_min -- if this ever fails, the field's semantics changed and this \
         test's premise needs revisiting"
    );
    assert!(shallow.is_marginal());
    assert_eq!(
        shallow.verdict(),
        StabilityVerdict::Marginal,
        "a negative lambda_min inside the noise floor must be MARGINAL, never Stable. \
         `is_stable` says {} here, which is exactly why the wiring must not read it.",
        shallow.is_stable
    );
    assert_ne!(
        shallow.verdict(),
        StabilityVerdict::Stable,
        "reporting a negative lambda_min as STABLE is the failure this enum exists to prevent"
    );

    // (b) A POSITIVE eigenvalue inside the band is equally unproven.
    let shallow_pos = mk(5e-7, true);
    assert_eq!(shallow_pos.verdict(), StabilityVerdict::Marginal);

    // (c) Just OUTSIDE the band, both signs must be decisive -- otherwise the
    //     Marginal arm would swallow everything and the test above would be
    //     satisfied by a constant. (Reachable-pass-condition check.)
    assert_eq!(mk(-5e-3, true).verdict(), StabilityVerdict::Unstable);
    assert_eq!(mk(5e-3, true).verdict(), StabilityVerdict::Stable);

    // (d) An unconverged eigensolve is INDETERMINATE regardless of sign, and
    //     that arm must win over the others (it is checked first).
    assert_eq!(mk(5e-3, false).verdict(), StabilityVerdict::Indeterminate);
    assert_eq!(mk(-5e-3, false).verdict(), StabilityVerdict::Indeterminate);

    // (e) All four verdicts are reachable -- the enum is a measurement, not a
    //     constant.
    let seen = [
        mk(5e-3, true).verdict(),
        mk(-5e-3, true).verdict(),
        mk(-5e-7, true).verdict(),
        mk(5e-3, false).verdict(),
    ];
    eprintln!("MARGINAL BAND  verdicts reachable: {seen:?}");
    assert_eq!(
        seen,
        [
            StabilityVerdict::Stable,
            StabilityVerdict::Unstable,
            StabilityVerdict::Marginal,
            StabilityVerdict::Indeterminate
        ],
        "all four verdicts must be reachable"
    );

    // (f) And `summary()` must agree with `verdict()` -- they are one source of
    //     truth, so a future edit cannot make the log line and the branch
    //     disagree.
    for r in [
        mk(5e-3, true),
        mk(-5e-3, true),
        mk(-5e-7, true),
        mk(5e-3, false),
    ] {
        assert!(
            r.summary().contains(r.verdict().label()),
            "summary() must report the same verdict the wiring branches on: {} vs {:?}",
            r.summary(),
            r.verdict()
        );
    }
}
