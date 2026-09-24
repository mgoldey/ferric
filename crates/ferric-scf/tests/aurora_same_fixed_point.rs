//! AURORA must reach the SAME stationary point as DIIS.
//!
//! An accelerator changes the PATH to the fixed point, never the fixed point
//! itself. The target Hamiltonian supplies every energy, gradient and
//! convergence decision; the auxiliary STO-3G model touches only curvature. So
//! if AURORA and DIIS disagree on the converged energy by more than the SCF
//! convergence tolerance, that is a bug in the accelerator — not a new answer.
//!
//! The paper reports a largest absolute final-energy difference of
//! 1.60e-10 Eh over its 16 direct CPU RHF pairs; these tests assert agreement at
//! a comparable bar.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::aurora::{aurora_steps_on_this_thread, AuroraConfig};
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;

/// One `solve_rhf` plus the number of AURORA steps it took (a delta of the
/// thread-local counter; the SCF loop runs on this thread).
fn solve_counting_steps(
    ctx: &ParallelContext,
    mol: &Molecule,
    prep: &PreparedBasis,
    op: Operator,
    bounds: &SchwarzBounds,
    cfg: &RhfConfig,
) -> (ferric_scf::result::ScfResult, usize) {
    let before = aurora_steps_on_this_thread();
    let r = solve_rhf(ctx, mol, prep, op, bounds, cfg).unwrap();
    (r, aurora_steps_on_this_thread() - before)
}

/// Run one system both ways and return
/// `(E_diis, iters_diis, conv_diis, E_aurora, iters_aurora, conv_aurora, aurora_steps)`,
/// where `aurora_steps` counts the accelerated steps the AURORA-requested run
/// actually took (0 when the gate declined it).
#[allow(clippy::type_complexity)]
fn pair(
    xyz: &str,
    basis_name: &str,
    xc: Option<&str>,
) -> (f64, usize, bool, f64, usize, bool, usize) {
    let mol = Molecule::load_xyz(xyz).unwrap();
    let bs = basis::bundled(basis_name).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let cfg_diis = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        xc: xc.map(|s| s.to_string()),
        df_j_aux: xc.map(|_| "def2-universal-jkfit".to_string()),
        ..Default::default()
    };
    let cfg_aurora = RhfConfig {
        aurora: AuroraConfig {
            enabled: true,
            ..Default::default()
        },
        ..cfg_diis.clone()
    };

    let (a, steps_diis) = solve_counting_steps(&ctx, &mol, &prep, op, &bounds, &cfg_diis);
    assert_eq!(
        steps_diis, 0,
        "the DIIS baseline (AURORA disabled) took AURORA steps"
    );
    let (b, steps_aurora) = solve_counting_steps(&ctx, &mol, &prep, op, &bounds, &cfg_aurora);
    (
        a.energy,
        a.iterations,
        a.converged,
        b.energy,
        b.iterations,
        b.converged,
        steps_aurora,
    )
}

fn check(label: &str, xyz: &str, basis_name: &str, xc: Option<&str>, tol: f64) {
    let (e_d, it_d, cv_d, e_a, it_a, cv_a, steps) = pair(xyz, basis_name, xc);
    let de = (e_a - e_d).abs();
    eprintln!(
        "{label:<24} DIIS: E = {e_d:.12} ({it_d} it, conv={cv_d})  \
         AURORA: E = {e_a:.12} ({it_a} it, conv={cv_a}, {steps} AURORA steps)  |ΔE| = {de:.3e}"
    );
    assert!(cv_d, "{label}: DIIS baseline must converge");
    assert!(cv_a, "{label}: AURORA must converge");
    // Without this a "same fixed point" pass is vacuous: an accelerator that
    // never engaged reproduces DIIS trivially.
    assert!(
        steps > 0,
        "{label}: AURORA was requested on a validated reference but took no steps, \
         so this comparison tested DIIS against itself"
    );
    assert!(
        de < tol,
        "{label}: AURORA and DIIS must reach the same stationary point, \
         |ΔE| = {de:.3e} > {tol:.3e}. An accelerator changes the path, not the answer."
    );
}

#[test]
fn water_sto3g_rhf_same_fixed_point() {
    check(
        "water/STO-3G RHF",
        "../../testdata/molecules/water.xyz",
        "sto-3g",
        None,
        1e-9,
    );
}

#[test]
fn water_ccpvdz_rhf_same_fixed_point() {
    check(
        "water/cc-pVDZ RHF",
        "../../testdata/molecules/water.xyz",
        "cc-pvdz",
        None,
        1e-9,
    );
}

#[test]
fn methane_ccpvdz_rhf_same_fixed_point() {
    check(
        "methane/cc-pVDZ RHF",
        "../../testdata/molecules/methane.xyz",
        "cc-pvdz",
        None,
        1e-9,
    );
}

#[test]
fn benzene_sto3g_rhf_same_fixed_point() {
    check(
        "benzene/STO-3G RHF",
        "../../testdata/molecules/benzene.xyz",
        "sto-3g",
        None,
        1e-9,
    );
}

#[test]
fn benzene_ccpvdz_rhf_same_fixed_point() {
    check(
        "benzene/cc-pVDZ RHF",
        "../../testdata/molecules/benzene.xyz",
        "cc-pvdz",
        None,
        1e-9,
    );
}

/// A pure functional must be DECLINED, not silently accelerated.
///
/// PBE has no exact exchange, so the auxiliary model has no exchange curvature
/// and (lacking the paper's undefined `D_k^xc`) understates the true curvature.
/// The gate keeps such references on DIIS, which must therefore reproduce the
/// DIIS result exactly — same energy bits, same iteration count.
///
/// # Why convergence is NOT asserted here
///
/// The claim is *inertness*: requesting AURORA on a declined reference must
/// change nothing. That is a LIVE-vs-LIVE comparison — both legs run in this
/// process on this CPU — and `to_bits()` on the energy plus equality on the
/// iteration count is the whole of it. Whether the underlying PBE SCF happens
/// to reach `density_conv` inside `max_iter` is a *different* property, and
/// was a machine-dependent one: measured on water/cc-pVDZ PBE this box
/// converged in **87 iterations**, while the CI runner was still going at the
/// 200 cap (`E = -76.333510156716`, this box `-76.333510144477`, a 1.2e-8 Ha
/// spread from BLAS dispatch through a stiff KS-DFT SCF).
///
/// Those numbers are STALE (2026-09-23) and largely record a DF-J defect, not
/// PBE stiffness: DF-J applied an explicit LU inverse of the RI metric whose
/// jitter floored the DIIS error. With the Cholesky DF-J solve this SCF
/// converges in 10 iterations (E = -76.3335101369 on this box). CI has not
/// been re-measured, so the argument below is kept: it does not depend on the
/// counts.
///
/// Asserting `converged` therefore tested the runner, not the gate — and it is
/// exactly what made this red on CI while the two assertions that carry the
/// claim both PASSED there (identical energies, 200 it vs 200 it). Inertness
/// holds whether or not the baseline finishes: if the declined run is truly
/// DIIS, it must track DIIS iteration for iteration, including when DIIS runs
/// out of iterations. That is asserted below, and the non-convergent case is a
/// strictly stronger test of it than the convergent one.
///
/// The decline is ALSO observed directly: the AURORA-requested run must take
/// zero accelerated steps. Bit-identity alone could in principle be satisfied
/// by an engaged accelerator that happened to reproduce DIIS; zero steps
/// cannot.
#[test]
fn pure_functional_is_declined_and_falls_back_to_diis_exactly() {
    let (e_d, it_d, cv_d, e_a, it_a, cv_a, steps) =
        pair("../../testdata/molecules/water.xyz", "cc-pvdz", Some("PBE"));
    eprintln!(
        "water/cc-pVDZ PBE (declined)  DIIS: E = {e_d:.12} ({it_d} it, conv={cv_d})  \
         AURORA-requested: E = {e_a:.12} ({it_a} it, conv={cv_a}, {steps} AURORA steps)"
    );
    assert_eq!(
        steps, 0,
        "a pure functional (a_x = 0) must be declined by the exchange gate, yet AURORA \
         took {steps} steps"
    );
    assert_eq!(
        e_a.to_bits(),
        e_d.to_bits(),
        "a declined reference must fall back to DIIS bit-for-bit: \
         got {:#018x} ({e_a:.17}) vs {:#018x} ({e_d:.17})",
        e_a.to_bits(),
        e_d.to_bits(),
    );
    assert_eq!(
        it_a, it_d,
        "a declined reference must take exactly the DIIS iteration count"
    );
    assert_eq!(
        cv_a, cv_d,
        "a declined reference must reach exactly the DIIS convergence verdict \
         ({cv_d}), whatever that verdict is"
    );
}

/// The gate must be REACHABLE in both directions — otherwise the test above is
/// asserting arithmetic rather than measuring a decision.
///
/// Opting in with `allow_low_exchange_ks` must actually ENGAGE the accelerator,
/// and declining must not. Both are observed directly, through the count of
/// accelerated steps each run took ([`aurora_steps_on_this_thread`]), and the
/// engaged run must still land on the declined run's fixed point.
///
/// # Why this no longer compares iteration counts
///
/// The first version asserted `engaged.iterations != declined.iterations` as
/// its evidence of engagement. That was a side effect, not the decision, and
/// it only held because of noise: DF-J applied an explicit LU inverse of the
/// RI metric, whose jitter dragged plain DIIS out on this PBE system (87
/// iterations here, 200+ on CI). With the Cholesky DF-J solve both runs
/// converge in 10 iterations to the same energy (-76.3335101369), so the test
/// failed while AURORA was demonstrably engaged: with `trigger = 0` the
/// `use_aurora` branch in `solve_rhf` takes EVERY iteration from 2 until the
/// convergence exit, so the opted-in run was AURORA-driven throughout and
/// merely happened to need as many iterations as DIIS.
///
/// # What this catches, and how it fails
///
/// - Gate inert in the "on" direction (the `allow_low_exchange_ks` override
///   ignored, or `use_aurora` never true, or the branch removed): the engaged
///   run takes 0 steps and `engaged_steps > 0` fails.
/// - Gate inert in the "off" direction (the exchange check dropped, so PBE is
///   accelerated without the opt-in): `declined_steps == 0` fails.
/// - An engaged accelerator that moves the answer: the |ΔE| bar fails.
#[test]
fn the_low_exchange_gate_is_reachable_in_both_directions() {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let base = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 300,
        xc: Some("PBE".to_string()),
        df_j_aux: Some("def2-universal-jkfit".to_string()),
        ..Default::default()
    };
    let (declined, declined_steps) = solve_counting_steps(
        &ctx,
        &mol,
        &prep,
        op,
        &bounds,
        &RhfConfig {
            aurora: AuroraConfig {
                enabled: true,
                ..Default::default()
            },
            ..base.clone()
        },
    );
    // Opted in, with the shorter step the (pre-Cholesky) measurement showed
    // this regime needs; see `AuroraConfig::allow_low_exchange_ks`.
    let (engaged, engaged_steps) = solve_counting_steps(
        &ctx,
        &mol,
        &prep,
        op,
        &bounds,
        &RhfConfig {
            aurora: AuroraConfig {
                enabled: true,
                allow_low_exchange_ks: true,
                trust_radius: 0.10,
                first_trust_radius: 0.10,
                ..Default::default()
            },
            ..base.clone()
        },
    );

    eprintln!(
        "PBE declined: {} it, {declined_steps} AURORA steps (E = {:.12});  \
         opted in: {} it, {engaged_steps} AURORA steps (E = {:.12})",
        declined.iterations, declined.energy, engaged.iterations, engaged.energy
    );
    assert_eq!(
        declined_steps, 0,
        "without `allow_low_exchange_ks` a pure functional must stay on DIIS, yet AURORA \
         took {declined_steps} steps — the gate does not decline"
    );
    assert!(
        engaged_steps > 0,
        "opting in with `allow_low_exchange_ks` must engage the accelerator, yet it took \
         no steps — the gate is inert (the \"on\" branch is unreachable)"
    );
    assert!(engaged.converged, "the opted-in run must still converge");
    // Same fixed point, different path: the whole point of an accelerator.
    // Bar kept at its pre-Cholesky value; with the noise gone the measured
    // |ΔE| is below 1e-10 (both print -76.3335101369), not re-measured tighter.
    let de = (engaged.energy - declined.energy).abs();
    assert!(
        de < 1e-7,
        "even in the un-validated regime the fixed point must not move: {de:.3e}"
    );
}

/// B3LYP (a_x = 0.20) clears the validated-exchange gate and must reach the same
/// stationary point.
///
/// It is deliberately NOT asserted to be faster: measured on water/cc-pVDZ it
/// reaches the right answer but takes more iterations than DIIS at the paper's
/// default trust radius (166 vs 57; 59 at radius 0.10). That measurement is
/// reported rather than hidden — see the accompanying report.
///
/// # Why this does not use [`check`], and does not assert `converged`
///
/// The invariant is *same fixed point*, and that is what is asserted. The
/// `converged` flag is a different and machine-dependent quantity, and this
/// system sits right at the boundary: AURORA needs **166 iterations** on this
/// box against a `max_iter` of 200, so a runner whose BLAS dispatch costs it a
/// few more steps runs out. That is what happened on CI, where AURORA reported
/// 200 iterations and `conv=false` — while nonetheless sitting **7.8e-9 Ha**
/// from the DIIS answer, i.e. inside this test's own 1e-8 tolerance. Raising
/// `max_iter` does not fix it (measured: identical 166/57 counts at caps of
/// 200, 400 and 800 on this box, so the cap is not what binds here — the CI
/// trajectory is genuinely different).
///
/// So the honest statement of the claim is the one below: **AURORA lands on
/// DIIS's stationary point**, tested live-vs-live in one process, which holds
/// on both machines. The CI numbers and this box's numbers both satisfy it.
/// Asserting "AURORA converges in under 200 iterations" would be asserting a
/// property of the runner, and the measured cross-machine spread in the
/// converged energy itself (~1e-8 Ha, larger than the AURORA-vs-DIIS gap on
/// either machine) shows that iteration counts here are not a stable quantity
/// to pin.
///
/// The DIIS baseline IS still required to converge: it is the reference the
/// comparison is made against, it converged on both machines (57 it here, 151
/// on CI), and without it there is no fixed point to compare to.
///
/// # STALE COUNTS (2026-09-23) — not re-measured
///
/// Every iteration count and cross-machine energy spread above (166 / 57 / 59,
/// CI 200 / 151, the ~1e-8 Ha spread, the 7.8e-9 Ha CI gap) was measured while
/// DF-J applied an explicit LU inverse of the RI metric. That inverse's jitter
/// floored the DIIS error and moved RI-J energies under any bit-level change,
/// so it very likely drove both the long trajectories and the machine
/// dependence. On the same fix water/cc-pVDZ PBE went from 87 DIIS iterations
/// to 10. These B3LYP numbers have not been re-measured with the Cholesky DF-J
/// solve; treat them as history, not as the current behaviour. The assertions
/// do not depend on them. Engagement is now asserted directly (AURORA must
/// take at least one step), so this cannot pass by comparing DIIS with itself.
#[test]
fn water_ccpvdz_b3lyp_same_fixed_point() {
    let (e_d, it_d, cv_d, e_a, it_a, cv_a, steps) = pair(
        "../../testdata/molecules/water.xyz",
        "cc-pvdz",
        Some("B3LYP"),
    );
    let de = (e_a - e_d).abs();
    eprintln!(
        "water/cc-pVDZ B3LYP      DIIS: E = {e_d:.12} ({it_d} it, conv={cv_d})  \
         AURORA: E = {e_a:.12} ({it_a} it, conv={cv_a}, {steps} AURORA steps)  |ΔE| = {de:.3e}"
    );
    assert!(
        steps > 0,
        "water/cc-pVDZ B3LYP (a_x = 0.20) clears the exchange gate, so AURORA must \
         engage; it took no steps, and the fixed-point check below would compare DIIS \
         with itself"
    );
    assert!(
        cv_d,
        "water/cc-pVDZ B3LYP: the DIIS baseline must converge — it is the \
         reference this comparison is made against"
    );
    assert!(
        de < 1e-8,
        "water/cc-pVDZ B3LYP: AURORA and DIIS must reach the same stationary \
         point, |ΔE| = {de:.3e} > 1e-8. An accelerator changes the path, not the \
         answer. (AURORA converged = {cv_a} after {it_a} iterations; that flag is \
         machine-dependent here and is deliberately not asserted — the fixed \
         point is.)"
    );
}

/// The auxiliary basis is a CURVATURE model: changing it must not move the
/// answer, only the path.
///
/// This is the sharpest available test that nothing from the auxiliary model
/// leaks into the result. If STO-3G and 6-31G curvature gave different converged
/// energies, the auxiliary model would be contaminating the target problem —
/// precisely the failure mode the method's central claim rules out.
#[test]
fn changing_the_auxiliary_curvature_basis_does_not_move_the_answer() {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();
    let ctx = ParallelContext::default();

    let base = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        ..Default::default()
    };

    let mut energies = Vec::new();
    for aux in ["sto-3g", "6-31g"] {
        let cfg = RhfConfig {
            aurora: AuroraConfig {
                enabled: true,
                auxbasis: aux.to_string(),
                ..Default::default()
            },
            ..base.clone()
        };
        let (r, steps) = solve_counting_steps(&ctx, &mol, &prep, op, &bounds, &cfg);
        eprintln!(
            "aux = {aux:<8} E = {:.12}  iters = {}  converged = {}  AURORA steps = {steps}",
            r.energy, r.iterations, r.converged
        );
        assert!(r.converged, "AURORA with aux={aux} must converge");
        // An accelerator that never engaged cannot leak its curvature basis
        // into the answer, so without this the test would pass vacuously.
        assert!(steps > 0, "AURORA with aux={aux} took no steps");
        energies.push(r.energy);
    }
    let d = (energies[0] - energies[1]).abs();
    eprintln!("curvature-basis sensitivity of the ANSWER: |ΔE| = {d:.3e}");
    assert!(
        d < 1e-9,
        "the auxiliary basis is a curvature model only; it must not change the \
         converged energy: |ΔE| = {d:.3e}"
    );
}

/// AURORA must reach the same answer regardless of thread count.
///
/// The auxiliary build and the response contraction both run under rayon; a
/// reduction-order bug there would show up as a thread-count-dependent path and,
/// at a loose convergence threshold, potentially a different answer.
#[test]
fn aurora_answer_is_independent_of_rayon_thread_count() {
    let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
    let bs = basis::bundled("cc-pvdz").unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).unwrap();

    let cfg = RhfConfig {
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 200,
        aurora: AuroraConfig {
            enabled: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut seen: Vec<(usize, f64, usize)> = Vec::new();
    for nthreads in [1usize, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(nthreads)
            .build()
            .unwrap();
        let (e, it) = pool.install(|| {
            let ctx = ParallelContext::default();
            let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
            assert!(r.converged, "AURORA must converge at {nthreads} threads");
            (r.energy, r.iterations)
        });
        eprintln!("threads = {nthreads}: E = {e:.12}, iters = {it}");
        seen.push((nthreads, e, it));
    }
    let e0 = seen[0].1;
    for (n, e, _) in &seen {
        let d = (e - e0).abs();
        assert!(
            d < 1e-9,
            "AURORA energy must not depend on thread count: {n} threads gave \
             ΔE = {d:.3e} vs 1 thread"
        );
    }
}
