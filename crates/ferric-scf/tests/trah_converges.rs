//! TRAH ON: correctness, engagement, and the trust-region mechanics.
//!
//! The companion to `trah_off_is_bit_identical.rs`. That file proves TRAH
//! disabled changes nothing; this one proves TRAH enabled reaches the SAME
//! answer DIIS reaches, that the branch actually ran (via the engagement
//! counter, not inferred from the energy), and that the trust region's moving
//! parts — the level shift, the α search, the ρ test — are genuinely exercised
//! rather than being dead code that happens to sit next to a working solver.
//!
//! # Mutation ledger for this file
//!
//! - `assert_trah_engaged` with the arming gate reverted to `false`: KILLED
//!   (every test fails on the counter).
//! - The α-search assertion with the bisection removed (α pinned at 1): KILLED
//!   on `trah_alpha_search_engages_when_the_radius_binds`, which forces a tiny
//!   radius so the constraint MUST bind.
//! - `RHF_ENERGY_SCALE` 4 → 1 and `UHF_ENERGY_SCALE` 2 → 1: KILLED on
//!   `trah_rho_is_order_unity_on_a_real_scf` (measured ρ = 3.83 and 2.00
//!   respectively). That test exists BECAUSE this mutation was a live bug, not
//!   a hypothetical — see `crate::trah`'s `RHF_ENERGY_SCALE` docs.
//! - The ρ-rejection assertion: see the note on
//!   `trah_rejection_is_wired_into_the_loop` — NOT claimed as a kill at the
//!   default threshold, and the reason is recorded there rather than hidden.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::trah::TrahConfig;
use ferric_scf::uhf::solve_uhf;
use std::sync::atomic::Ordering;

fn water() -> Molecule {
    Molecule::parse_xyz(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.0 0.757 0.587\nH 0.0 -0.757 0.587\n",
        0,
        1,
    )
    .unwrap()
}

fn oh_doublet() -> Molecule {
    Molecule::parse_xyz("2\noh\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n", 0, 2).unwrap()
}

fn steps() -> usize {
    ferric_scf::trah::TRAH_STEPS_TAKEN.load(Ordering::Relaxed)
}
fn rejections() -> usize {
    ferric_scf::trah::TRAH_STEPS_REJECTED.load(Ordering::Relaxed)
}

/// The branch must have RUN. Asserted on the counter, never inferred from the
/// energy — on an easy system both paths converge to identical bits, which is
/// what let an earlier version of the sibling test be fooled.
fn assert_trah_engaged(before: usize, what: &str) -> usize {
    let ran = steps().saturating_sub(before);
    assert!(
        ran > 0,
        "{what}: TRAH was enabled but no TRAH step executed. Matching DIIS's \
         energy does NOT prove the path ran."
    );
    ran
}

/// RHF: TRAH must reach the same energy DIIS reaches, and must actually engage.
#[test]
fn rhf_trah_matches_diis_and_engages() {
    let mol = water();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let cfg_diis = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };
    let cfg_trah = RhfConfig {
        trah_trigger: Some(1e-2),
        ..cfg_diis.clone()
    };

    let r_diis = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_diis).unwrap();
    let before = steps();
    let r_trah = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_trah).unwrap();
    let ran = assert_trah_engaged(before, "RHF/H2O/cc-pVDZ");

    eprintln!(
        "RHF/H2O/cc-pVDZ  DIIS: E={:.10} iters={}   TRAH: E={:.10} iters={} trah_steps={ran}",
        r_diis.energy, r_diis.iterations, r_trah.energy, r_trah.iterations
    );

    assert!(r_diis.converged && r_trah.converged);
    assert!(
        (r_diis.energy - r_trah.energy).abs() < 1e-8,
        "TRAH must reach the DIIS energy: ΔE = {:.3e}",
        (r_diis.energy - r_trah.energy).abs()
    );
}

/// RKS/PBE: the closed-shell KS path, where TRAH additionally has to build and
/// use the GGA f_xc kernel in its Hessian matvec.
#[test]
fn rks_pbe_trah_matches_diis_and_engages_the_gga_kernel() {
    let mol = water();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let cfg_diis = RhfConfig {
        xc: Some("PBE".into()),
        energy_conv: 1e-9,
        density_conv: 1e-7,
        max_iter: 200,
        level_shift: 0.2,
        ..Default::default()
    };
    let cfg_trah = RhfConfig {
        trah_trigger: Some(1e-2),
        ..cfg_diis.clone()
    };

    let r_diis = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_diis).unwrap();

    let kern_before = ferric_scf::rohf::GGA_FXC_KERNEL_BUILDS.load(Ordering::Relaxed);
    let before = steps();
    let r_trah = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_trah).unwrap();
    let ran = assert_trah_engaged(before, "RKS/PBE/H2O");
    let kernels = ferric_scf::rohf::GGA_FXC_KERNEL_BUILDS
        .load(Ordering::Relaxed)
        .saturating_sub(kern_before);

    eprintln!(
        "RKS/PBE/H2O  DIIS: E={:.10} iters={}   TRAH: E={:.10} iters={} trah_steps={ran} gga_kernels={kernels}",
        r_diis.energy, r_diis.iterations, r_trah.energy, r_trah.iterations
    );

    assert!(r_trah.converged, "TRAH RKS/PBE must converge");
    assert!(
        kernels >= 1,
        "the RKS TRAH path must build a GGA f_xc kernel for PBE, not silently \
         analyse the HF Hessian at a KS density (got {kernels} builds)"
    );
    assert!(
        (r_diis.energy - r_trah.energy).abs() < 1e-7,
        "TRAH RKS must reach the DIIS energy: ΔE = {:.3e}",
        (r_diis.energy - r_trah.energy).abs()
    );
}

/// UHF: the open-shell path, where α and β rotations are solved as one coupled
/// trust-region problem.
#[test]
fn uhf_trah_matches_diis_and_engages() {
    let mol = oh_doublet();
    let bs = basis::bundled("6-31g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let cfg_diis = RhfConfig {
        energy_conv: 1e-9,
        density_conv: 1e-7,
        max_iter: 300,
        level_shift: 0.2,
        ..Default::default()
    };
    let cfg_trah = RhfConfig {
        trah_trigger: Some(1e-2),
        ..cfg_diis.clone()
    };

    let r_diis = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg_diis).unwrap();
    let before = steps();
    let r_trah = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg_trah).unwrap();
    let ran = assert_trah_engaged(before, "UHF/OH/6-31G");

    eprintln!(
        "UHF/OH/6-31G  DIIS: E={:.10} iters={}   TRAH: E={:.10} iters={} trah_steps={ran}",
        r_diis.energy, r_diis.iterations, r_trah.energy, r_trah.iterations
    );

    assert!(r_trah.converged, "TRAH UHF must converge");
    assert!(
        (r_diis.energy - r_trah.energy).abs() < 1e-7,
        "TRAH UHF must reach the DIIS energy: ΔE = {:.3e}",
        (r_diis.energy - r_trah.energy).abs()
    );
}

/// UKS/PBE: the open-shell KS path.
#[test]
fn uks_pbe_trah_matches_diis_and_engages() {
    let mol = oh_doublet();
    let bs = basis::bundled("6-31g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let cfg_diis = RhfConfig {
        xc: Some("PBE".into()),
        energy_conv: 1e-8,
        density_conv: 1e-6,
        max_iter: 300,
        level_shift: 0.3,
        ..Default::default()
    };
    let cfg_trah = RhfConfig {
        trah_trigger: Some(1e-2),
        ..cfg_diis.clone()
    };

    let r_diis = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg_diis).unwrap();
    let before = steps();
    let r_trah = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg_trah).unwrap();
    let ran = assert_trah_engaged(before, "UKS/PBE/OH");

    eprintln!(
        "UKS/PBE/OH  DIIS: E={:.10} iters={}   TRAH: E={:.10} iters={} trah_steps={ran}",
        r_diis.energy, r_diis.iterations, r_trah.energy, r_trah.iterations
    );

    assert!(r_trah.converged, "TRAH UKS must converge");

    // NOT "must equal DIIS". MEASURED on this system: TRAH reaches
    // −75.621241952447 with dE = 0 and dp_rms = 4.9e-7, while DIIS stops at
    // −75.6212406478 — TRAH lands 1.3e-6 Ha LOWER and better converged, because
    // DIIS exits as soon as the (deliberately loose) gate is satisfied.
    //
    // Asserting equality to 1e-6 here would be asserting that the second-order
    // method must reproduce the first-order method's early exit, i.e. it would
    // fail BECAUSE TRAH did better. The physically meaningful assertions are
    // (a) same solution to chemical accuracy and (b) TRAH is not HIGHER.
    let de = r_trah.energy - r_diis.energy;
    assert!(
        de.abs() < 1e-4,
        "TRAH UKS must find the same solution as DIIS to well within chemical \
         accuracy: ΔE = {de:.3e}"
    );
    assert!(
        de < 1e-9,
        "TRAH UKS must not land ABOVE the DIIS solution (a variational \
         regression); TRAH − DIIS = {de:.3e}"
    );
}

/// The α search must ENGAGE when the radius binds.
///
/// With a radius far below the natural Newton step length, the constraint is
/// active and the solver must spend more than one augmented-Hessian eigensolve
/// (α > α_min) to land the step on the boundary. This is the test that
/// distinguishes a real level-shifted trust region from ferric's pre-existing
/// "solve then clip" behaviour: a clip would take exactly one solve at α = 1
/// and then truncate.
///
/// The check runs on the SOLVER directly rather than through an SCF loop, so
/// the pass condition is reachable by construction and does not depend on a
/// molecule happening to present a long step.
#[test]
fn trah_alpha_search_engages_when_the_radius_binds() {
    use ferric_core::FerricError;
    // A deliberately ill-conditioned Hessian: small positive curvature means a
    // long Newton step, so any modest radius binds.
    let hdiag = [0.02f64, 0.05, 0.09, 0.15];
    let g = [0.06f64, -0.04, 0.05, -0.03];
    let mv = |v: &[f64]| -> Result<Vec<f64>, FerricError> {
        Ok(v.iter().zip(hdiag.iter()).map(|(a, b)| a * b).collect())
    };
    let cfg = TrahConfig::default();

    let radius = 0.1;

    // Reachability guard: the test is only meaningful if the UNCONSTRAINED step
    // is much longer than the radius, so the constraint genuinely binds. Stated
    // relative to `radius` rather than as a bare magic number — an absolute
    // bound here is a second, independent thing to keep in sync, and it
    // previously failed at ‖κ‖ = 0.805 against a hardcoded 1.0 while the
    // constraint it was guarding (0.805 ≫ 0.1) was amply satisfied.
    let big = ferric_scf::trah::solve_trust_region(&g, &mv, &hdiag, 1e6, &cfg).unwrap();
    assert!(
        big.norm > 5.0 * radius,
        "the reference system must have a Newton step much longer than Δ={radius} \
         for this test to be meaningful, got ‖κ‖ = {}",
        big.norm
    );
    let s = ferric_scf::trah::solve_trust_region(&g, &mv, &hdiag, radius, &cfg).unwrap();
    eprintln!(
        "α-search: unconstrained ‖κ‖={:.3e}; at Δ={radius} got ‖κ‖={:.3e} α={:.1} μ={:.3e} solves={}",
        big.norm, s.norm, s.alpha, s.level_shift, s.shift_iterations
    );

    assert!(
        s.norm <= radius * (1.0 + 1e-6),
        "the trust region must be respected: ‖κ‖ = {} > Δ = {radius}",
        s.norm
    );
    assert!(
        s.alpha > 1.0,
        "a binding radius must drive the α search above α_min = 1, got α = {}",
        s.alpha
    );
    assert!(
        s.shift_iterations > 1,
        "a binding radius must cost more than the single α = 1 eigensolve — \
         one solve means the step was CLIPPED, not level-shifted (got {} solves)",
        s.shift_iterations
    );
    assert!(
        s.level_shift < big.level_shift,
        "the constrained step must carry a HARDER level shift than the \
         unconstrained one: {} vs {}",
        s.level_shift,
        big.level_shift
    );
    assert!(
        s.predicted_residual < 1e-5,
        "the two predicted-reduction formulas must still agree under the α \
         search: rel resid {:e}",
        s.predicted_residual
    );
}

/// ρ must be ORDER UNITY on a real SCF — the test that caught a 4× scale bug.
///
/// # What this is really checking
///
/// ρ = ΔE_act/ΔE_pred is meaningful only if ΔE_pred is a genuine energy. It is
/// not automatic that it is: `rhf_newton`/`uhf_newton` intentionally drop the
/// constant prefactor relating `F_ai` to the true orbital gradient, because a
/// Newton solve is invariant under it. A trust region is not. With the
/// prefactor missing, the first working TRAH runs reported ρ = 3.999 and 3.835
/// (RHF) and ρ = 2.0047, 2.0051 (UHF) — suspiciously clean constants, which is
/// the fingerprint of a scale error rather than a modelling error, and which
/// would have silently mis-classified every step against Fletcher's 0.25/0.75
/// thresholds.
///
/// Asserting ρ ∈ [0.5, 2] near convergence therefore pins the prefactors
/// (`RHF_ENERGY_SCALE` = 4, `UHF_ENERGY_SCALE` = 2) at both spin cases. Removing
/// either constant moves ρ to ≈4 or ≈2 respectively and FAILS this test — the
/// mutation that motivated writing it.
///
/// The window is deliberately wide: ρ is only expected to be ≈1 where the
/// quadratic model is accurate, so this is a scale check, not a model-quality
/// claim.
#[test]
fn trah_rho_is_order_unity_on_a_real_scf() {
    let bs6 = basis::bundled("6-31g").unwrap();
    let op = Operator::coulomb();

    // ---- closed shell (scale 4) ----
    {
        let mol = water();
        let prep = PreparedBasis::new(&mol, &bs6).unwrap();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let cfg = RhfConfig {
            trah_trigger: Some(1e-2),
            energy_conv: 1e-10,
            density_conv: 1e-8,
            max_iter: 200,
            ..Default::default()
        };
        let before = steps();
        let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
        assert_trah_engaged(before, "RHF ρ-scale probe");
        let rho = ferric_scf::trah::last_rho_rhf();
        eprintln!("RHF/H2O/6-31G  final ρ = {rho:?}  (E = {:.10})", r.energy);
        let rho = rho.expect("a TRAH step ran, so a ρ must have been formed");
        assert!(
            (0.5..=2.0).contains(&rho),
            "RHF ρ must be order unity; got {rho}. A ρ pinned near 4 means \
             RHF_ENERGY_SCALE is missing (the packed gradient is F_ai, the true \
             one is 4·F_ai)."
        );
    }

    // ---- open shell (scale 2) ----
    {
        let mol = oh_doublet();
        let prep = PreparedBasis::new(&mol, &bs6).unwrap();
        let bounds = SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let cfg = RhfConfig {
            trah_trigger: Some(1e-2),
            energy_conv: 1e-9,
            density_conv: 1e-7,
            max_iter: 300,
            level_shift: 0.2,
            ..Default::default()
        };
        let before = steps();
        let r = solve_uhf(&ctx, &mol, &prep, &bounds, &cfg).unwrap();
        assert_trah_engaged(before, "UHF ρ-scale probe");
        let rho = ferric_scf::trah::last_rho_uhf();
        eprintln!("UHF/OH/6-31G  final ρ = {rho:?}  (E = {:.10})", r.energy);
        let rho = rho.expect("a TRAH step ran, so a ρ must have been formed");
        assert!(
            (0.5..=2.0).contains(&rho),
            "UHF ρ must be order unity; got {rho}. A ρ pinned near 2 means \
             UHF_ENERGY_SCALE is missing."
        );
    }
}

/// Rejection must be REACHABLE, and a rejection storm must degrade gracefully.
///
/// # Honest scope
///
/// With `rho_reject` set just below 1, most steps are rejected by construction.
/// That proves the rejection path is WIRED INTO THE LOOP (it is not dead code)
/// and that the collapsed-radius fallback keeps the run finite — it does NOT
/// prove the criterion fires on its own merits at the default threshold.
///
/// It cannot, on a system this size: measured ρ values on water/6-31G are ~1.00
/// with the scale fixed, so a default-threshold rejection essentially never
/// occurs — the quadratic model is simply accurate there. A test that forced
/// one would be measuring its own contrivance. The per-branch ρ arithmetic is
/// instead exercised exhaustively, and mutation-checked branch by branch, in
/// `trah.rs`'s own unit tests.
#[test]
fn trah_rejection_is_wired_into_the_loop() {
    let mol = water();
    let bs = basis::bundled("6-31g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let cfg = RhfConfig {
        trah_trigger: Some(1e-2),
        trah: TrahConfig {
            // Reject anything that does not beat the model by 20× — reachable
            // by construction, which is the point.
            rho_reject: 20.0,
            ..Default::default()
        },
        energy_conv: 1e-9,
        density_conv: 1e-7,
        max_iter: 200,
        ..Default::default()
    };

    let rej_before = rejections();
    let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    let rejected = rejections().saturating_sub(rej_before);

    eprintln!(
        "TRAH rejection probe: E={:.10} converged={} iters={} rejections={rejected}",
        r.energy, r.converged, r.iterations
    );

    assert!(
        rejected > 0,
        "with rho_reject = 20 every step must be rejected — if none is, the \
         rejection path is not reachable from the SCF loop at all"
    );
    assert!(
        r.energy.is_finite(),
        "a rejection-heavy run must still produce a finite energy, not diverge"
    );
    // Rejection must not cost correctness: the run still has to land on the
    // right solution, because a rejected step is UNDONE, not applied badly.
    let cfg_plain = RhfConfig {
        energy_conv: 1e-9,
        density_conv: 1e-7,
        max_iter: 200,
        ..Default::default()
    };
    let r_plain = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg_plain).unwrap();
    assert!(
        (r.energy - r_plain.energy).abs() < 1e-7,
        "a rejection-heavy TRAH run must still reach the DIIS energy \
         ({:.10} vs {:.10}) — rejection undoes a step, it does not corrupt one",
        r.energy,
        r_plain.energy
    );
}

/// Thread-count independence: TRAH's answer must not depend on rayon width.
///
/// Constants frozen at one worker count have broken this repo's tests before,
/// and the TRAH path adds a Davidson eigensolve whose matvec runs the parallel
/// J/K build. The energies are compared at SCF-convergence tolerance (not
/// bitwise): the underlying `build_jk` reduction is deterministic, but the
/// α-bisection path can differ in the last bits and thus take a different
/// number of iterations to the same solution.
#[test]
fn trah_energy_is_thread_count_independent() {
    let mol = water();
    let bs = basis::bundled("6-31g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();

    let cfg = RhfConfig {
        trah_trigger: Some(1e-2),
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };

    let mut out = Vec::new();
    for width in [1usize, 2, 4, 12] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(width)
            .build()
            .unwrap();
        let (e, it) = pool.install(|| {
            let ctx = ParallelContext::default();
            let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
            (r.energy, r.iterations)
        });
        eprintln!("TRAH rayon width {width}: E = {e:.12}, iters = {it}");
        out.push(e);
    }
    let e0 = out[0];
    for (i, &e) in out.iter().enumerate().skip(1) {
        assert!(
            (e - e0).abs() < 1e-9,
            "TRAH energy must not depend on rayon width: run {i} gave {e:.12} vs {e0:.12}"
        );
    }
}

/// When BOTH `trah_trigger` and `newton_trigger` are set, TRAH must win.
///
/// The two branches sit adjacent in the SCF loop and TRAH `continue`s, so the
/// precedence is a property of statement ORDER — the kind of thing that breaks
/// silently when someone reorders a loop body. Pinning it here means a future
/// edit that lets the PCG Newton branch steal TRAH's iterations is a test
/// failure rather than a quiet performance change.
///
/// Asserted via the engagement counter, because both paths converge to the
/// same energy and the energy therefore cannot distinguish them.
#[test]
fn trah_takes_precedence_over_newton_when_both_are_armed() {
    let mol = water();
    let bs = basis::bundled("6-31g").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let cfg = RhfConfig {
        newton_trigger: 1e-2,
        trah_trigger: Some(1e-2),
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };

    let before = steps();
    let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
    let ran = assert_trah_engaged(before, "TRAH-vs-Newton precedence");

    eprintln!(
        "both armed: E={:.10} iters={} trah_steps={ran}",
        r.energy, r.iterations
    );
    assert!(r.converged, "a doubly-armed run must still converge");
}
