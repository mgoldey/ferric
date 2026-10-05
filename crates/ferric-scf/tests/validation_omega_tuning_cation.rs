//! Validation tier for ω tuning's cation-state control: the N2⁺ instability
//! onset, the ω* before/after comparison, and the H2O no-flag case.
//!
//! These tests are SLOW (full ω sweeps with per-ω UHF cation solves) and are
//! `#[ignore = "validation: RSH omega tuning"]`. They live in a
//! `validation_*.rs` binary because the weekly job selects the tier with
//! `--run-ignored only -E 'binary(/^validation_/)'` — a `validation:` ignore in
//! any other file is skipped per-commit AND never selected weekly, i.e. dead
//! (enforced by `ferric-cli::ci_validation_tier_guard::
//! validation_ignores_live_only_in_validation_files`).
//!
//! The fast, always-run tests for the same feature are in
//! `omega_tuning_cation_branch.rs`.
//!
//! # What these establish, and what they cannot
//!
//! Neither continuation nor the branch check is a stability verdict.
//! `stability::ks_reference_is_analysable` refuses ω ≠ 0
//! (`StabilitySkip::RangeSeparated`), so a converged cation that is an internal
//! SADDLE cannot be identified here. `n2_onset_is_not_visible_without_an_
//! orbital_hessian` is the measured statement of that limit, and issue #314
//! tracks the fix.
//!
//! # Measured: N2/def2-SVP ωB97X-V, ω = 0.50…0.60
//!
//! `n2_cation_branch_sweep_measurement` prints the per-ω table; the numbers the
//! ASSERTING tests depend on are restated at each assertion.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::omega_tuning::{
    eval_j, eval_j_seeded, spin_population_asymmetry, tune_omega, OmegaSeedState, OmegaTuneConfig,
};
use ferric_scf::rhf::RhfConfig;
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::uhf::{solve_uhf, solve_uhf_with_guess};
const FUNCTIONAL: &str = "wB97X-V";

fn load(name: &str, basis: &str) -> (Molecule, PreparedBasis, SchwarzBounds) {
    let path = format!(
        "{}/../../testdata/molecules/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    let mol = Molecule::load_xyz(&path).unwrap_or_else(|e| panic!("{path}: {e:?}"));
    let bs = basis::bundled(basis).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
    (mol, prep, bounds)
}
fn cfg(basis_cfg: RhfConfig, lo: f64, hi: f64) -> OmegaTuneConfig {
    OmegaTuneConfig {
        functional: FUNCTIONAL.into(),
        omega_lo: lo,
        omega_hi: hi,
        omega_tol: 0.02,
        max_evals: 16,
        scf: basis_cfg,
        ..Default::default()
    }
}
/// NEGATIVE CONTROL on the flag (artifact hypothesis A4 — a tolerance below
/// the metric's smooth ω-drift flags everything, which looks like a working
/// check). H2O⁺'s hole sits on O, not on the two equivalent H, and stays
/// there across the whole bracket, so a correctly-scaled check must raise
/// ZERO flags over a full tuning run.
#[test]
#[ignore = "validation: RSH omega tuning"]
fn h2o_tuning_raises_no_branch_flag() {
    let (mol, prep, bounds) = load("water.xyz", "6-31g");
    let ctx = ParallelContext::default();
    let c = cfg(
        RhfConfig {
            df_j_aux: Some(String::new()),
            energy_conv: 1e-10,
            density_conv: 1e-7,
            max_iter: 500,
            ..Default::default()
        },
        0.2,
        0.8,
    );
    let t = tune_omega(&ctx, &mol, &prep, &bounds, &c).unwrap();
    for e in &t.evals {
        eprintln!(
            "  w={:.4} J={:+.3e} S2={:.6} asym={:.3e} seed={} flag={}",
            e.omega, e.j, e.cation_s_squared, e.cation_spin_asymmetry, e.seed, e.branch_changed
        );
    }
    assert!(
        t.branch_warning.is_none(),
        "H2O raised a branch flag over its whole tuning run: {:?}",
        t.branch_warning
    );
}

/// MEASUREMENT (not a pass/fail bar): N2/def2-SVP ωB97X-V across the cation
/// instability onset, with and without continuation, printing per ω the cation
/// energy, ⟨S²⟩, spin asymmetry, J and the seed used. The table it prints is
/// the evidence in REPORT-288.md; run it with `--nocapture`.
///
/// Cost: 2 SCFs per ω per arm; ~22 SCF pairs.
#[test]
#[ignore = "validation: RSH omega tuning"]
fn n2_cation_branch_sweep_measurement() {
    let (mol, prep, bounds) = load("validation/n2.xyz", "def2-svp");
    let ctx = ParallelContext::default();
    let base = RhfConfig {
        df_j_aux: Some(String::new()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        energy_conv: 1e-10,
        density_conv: 1e-7,
        max_iter: 500,
        ..Default::default()
    };
    let c = cfg(base, 0.2, 0.8);
    let omegas: Vec<f64> = (50..=60).map(|k| k as f64 / 100.0).collect();

    eprintln!("\n=== ARM A: independent (today's behaviour, continuation off) ===");
    eprintln!("   omega       E_cation        E_neutral            J        <S^2>     asym");
    let mut arm_a = Vec::new();
    for &w in &omegas {
        let e = eval_j(&ctx, &mol, &prep, &bounds, &c, w).unwrap();
        eprintln!(
            "  {:.4}  {:+.9}  {:+.9}  {:+.3e}  {:.6}  {:.3e}",
            w, e.e_cation, e.e_neutral, e.j, e.cation_s_squared, e.cation_spin_asymmetry
        );
        arm_a.push(e);
    }

    eprintln!("\n=== ARM B: continuation, monotone ω order (seed = nearest done) ===");
    eprintln!("   omega       E_cation        E_neutral            J        <S^2>     asym   seed");
    // Replays continuation by hand in a fixed ω order so the table is
    // comparable row-by-row with ARM A; tune_omega's own order is golden
    // section's, which visits different ω.
    let mut arm_b = Vec::new();
    let mut prev: Option<OmegaSeedState> = None;
    for &w in &omegas {
        let (e, carry) = eval_j_seeded(&ctx, &mol, &prep, &bounds, &c, w, prev.as_ref()).unwrap();
        eprintln!(
            "  {:.4}  {:+.9}  {:+.9}  {:+.3e}  {:.6}  {:.3e}  {}",
            w, e.e_cation, e.e_neutral, e.j, e.cation_s_squared, e.cation_spin_asymmetry, e.seed
        );
        prev = Some(carry);
        arm_b.push(e);
    }

    eprintln!("\n=== ARM A vs ARM B: |ΔE_cation| per ω ===");
    for (a, b) in arm_a.iter().zip(arm_b.iter()) {
        eprintln!(
            "  {:.4}  |dE_cat| {:.3e}  |dJ| {:.3e}  asym A {:.3e} B {:.3e}",
            a.omega,
            (a.e_cation - b.e_cation).abs(),
            (a.j - b.j).abs(),
            a.cation_spin_asymmetry,
            b.cation_spin_asymmetry
        );
    }
}

/// MEASUREMENT: can ferric's UKS reach the symmetry-BROKEN N2⁺ branch at all?
///
/// The sweep above shows the symmetric branch is what both the default guess
/// and continuation converge to across ω = 0.50…0.60, so the branch check
/// never sees a switch there. That leaves one question the check's worth
/// depends on: if a broken state existed and were reached, WOULD the metric
/// separate it? This probe seeds the cation from MOs whose β hole has been
/// rotated onto one nitrogen (a localizing rotation of the symmetric HOMO
/// with its symmetry partner) and reports what the SCF converges to.
///
/// Printed, not asserted: a negative outcome here is a statement about
/// ferric's UKS and ωB97X-V on this system, not a defect, and the report
/// records it as such.
#[test]
#[ignore = "validation: RSH omega tuning"]
fn n2_broken_branch_probe_measurement() {
    let (mol, prep, bounds) = load("validation/n2.xyz", "def2-svp");
    let ctx = ParallelContext::default();
    let scf_base = RhfConfig {
        df_j_aux: Some(String::new()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        energy_conv: 1e-10,
        density_conv: 1e-7,
        max_iter: 500,
        ..Default::default()
    };
    let mut cation = mol.clone();
    cation.charge += 1;
    cation.multiplicity = 2;
    let nelec = cation.nelec() as usize;
    let nocc_b = nelec / 2;
    let n = prep.nbasis();

    for &w in &[0.50_f64, 0.56, 0.60, 0.70] {
        let c = cfg(scf_base.clone(), 0.2, 0.8);
        let scf = ferric_scf::omega_tuning::state_scf_config(&c, w).unwrap();
        let sym = solve_uhf(&ctx, &cation, &prep, &bounds, &scf).unwrap();
        let s2_of = |r: &ferric_scf::ScfResult| -> f64 {
            let ov = ferric_integrals::oneelectron::overlap(&prep);
            let ca = r.mos_alpha.slice(ndarray::s![.., ..nocc_b + 1]);
            let cb2 = r
                .mos_beta
                .as_ref()
                .unwrap()
                .slice(ndarray::s![.., ..nocc_b]);
            let m = ca.t().dot(&ov).dot(&cb2);
            0.75 + nocc_b as f64 - m.iter().map(|v| v * v).sum::<f64>()
        };
        let sym_asym = spin_population_asymmetry(
            &cation,
            &prep,
            &sym.density_alpha,
            sym.density_beta.as_ref().unwrap(),
        )
        .unwrap();

        // Localizing rotation: mix the β HOMO with the β LUMO at 45°.
        //
        // It must be an occupied-VIRTUAL rotation. A rotation WITHIN the
        // occupied block leaves D_β = C_occ C_occᵀ invariant, so the SCF
        // restarts from the identical density and converges in 2 iterations
        // to the state it came from — a probe that cannot break symmetry and
        // whose "no broken branch found" would be arithmetic, not a result.
        let mut cb = sym.mos_beta.clone().unwrap();
        let (h, p) = (nocc_b - 1, nocc_b);
        assert!(p < n, "no beta virtual to rotate into");
        let r = std::f64::consts::FRAC_1_SQRT_2;
        for mu in 0..n {
            let (a, b) = (cb[(mu, h)], cb[(mu, p)]);
            cb[(mu, h)] = r * (a + b);
            cb[(mu, p)] = r * (a - b);
        }
        // The seed really is a different density (guards the probe itself).
        let d_seed = cb
            .slice(ndarray::s![.., ..nocc_b])
            .dot(&cb.slice(ndarray::s![.., ..nocc_b]).t());
        let d_sym = sym.density_beta.as_ref().unwrap();
        let seed_moved = (&d_seed - d_sym)
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(
            seed_moved > 1e-3,
            "the rotated seed has the same beta density as the symmetric state \
             (max|dD| = {seed_moved:.2e}) -- it cannot break symmetry, so any \
             negative outcome below would be an artifact of the probe"
        );
        let broken = solve_uhf_with_guess(
            &ctx,
            &cation,
            &prep,
            &bounds,
            &scf,
            Some((&sym.mos_alpha, &cb)),
        );
        match broken {
            Ok(br) => {
                let br_asym = spin_population_asymmetry(
                    &cation,
                    &prep,
                    &br.density_alpha,
                    br.density_beta.as_ref().unwrap(),
                )
                .unwrap();
                eprintln!(
                    "w={w:.2}: seed max|dD_b|={seed_moved:.2e}\n  symmetric      E={:+.9} \
                     asym={:.3e} S2={:.8}\n  broken-seeded  E={:+.9} asym={:.3e} S2={:.8}\
                     \n  dE={:+.3e} Ha  dS2={:+.3e}  dasym={:+.3e}  ({} iters)",
                    sym.energy,
                    sym_asym,
                    s2_of(&sym),
                    br.energy,
                    br_asym,
                    s2_of(&br),
                    br.energy - sym.energy,
                    s2_of(&br) - s2_of(&sym),
                    br_asym - sym_asym,
                    br.iterations
                );
            }
            Err(e) => eprintln!("w={w:.2}: broken-seeded UKS failed: {e:?}"),
        }
    }
}

/// BEFORE/AFTER: ω* and J from every system any existing ω-tuning test tunes,
/// with continuation OFF (the behaviour before this file existed) and ON.
///
/// Defaulting continuation ON is only justified if it leaves those results
/// UNCHANGED — not merely "still inside its bar". This prints the pair so the
/// claim is a number, not an argument. H2/6-31G is `omega_tuning.rs`'s driver
/// test; H2O and NH3 / def2-SVP are `validation_rsh_omega.rs`'s ω* rows.
#[test]
#[ignore = "validation: RSH omega tuning"]
fn omega_star_before_and_after_continuation() {
    let ctx = ParallelContext::default();
    let cases: [(&str, &str, f64, f64, f64, usize); 3] = [
        ("h2.xyz", "6-31g", 0.2, 1.2, 0.02, 16),
        ("validation/h2o.xyz", "def2-svp", 0.2, 0.8, 1e-5, 40),
        ("validation/nh3.xyz", "def2-svp", 0.2, 0.8, 1e-5, 40),
    ];
    eprintln!("\n=== omega* and J: continuation OFF vs ON ===");
    for (xyz, basis, lo, hi, tol, maxev) in cases {
        let (mol, prep, bounds) = load(xyz, basis);
        let scf = if basis == "6-31g" {
            RhfConfig {
                energy_conv: 1e-9,
                ..Default::default()
            }
        } else {
            RhfConfig {
                df_j_aux: Some(String::new()),
                df_k_aux: Some("def2-universal-jkfit".into()),
                max_iter: 500,
                energy_conv: 1e-10,
                density_conv: 1e-7,
                ..Default::default()
            }
        };
        let base = OmegaTuneConfig {
            omega_tol: tol,
            max_evals: maxev,
            ..cfg(scf, lo, hi)
        };
        let off = tune_omega(
            &ctx,
            &mol,
            &prep,
            &bounds,
            &OmegaTuneConfig {
                continuation: false,
                ..base.clone()
            },
        )
        .unwrap();
        let on = tune_omega(&ctx, &mol, &prep, &bounds, &base).unwrap();
        eprintln!(
            "{xyz}/{basis}:\n  OFF omega*={:.9} J={:+.6e} evals={} conv={}\n  \
             ON  omega*={:.9} J={:+.6e} evals={} conv={} warn={:?}\n  \
             |d omega*|={:.3e}  |dJ|={:.3e}",
            off.omega,
            off.j,
            off.evals.len(),
            off.converged,
            on.omega,
            on.j,
            on.evals.len(),
            on.converged,
            on.branch_warning,
            (off.omega - on.omega).abs(),
            (off.j - on.j).abs(),
        );
        // Both arms must visit the same ω: golden section's sequence depends
        // only on the bracket, so a mismatch would mean the comparison is not
        // like-for-like and every |d| above is meaningless.
        assert_eq!(
            off.evals.len(),
            on.evals.len(),
            "{xyz}: the two arms ran a different number of evaluations"
        );
        for (a, b) in off.evals.iter().zip(on.evals.iter()) {
            assert_eq!(a.omega, b.omega, "{xyz}: the two arms visited different ω");
            eprintln!(
                "    w={:.9}  J off {:+.9e}  on {:+.9e}  |dJ| {:.3e}  |dE_cat| {:.3e}",
                a.omega,
                a.j,
                b.j,
                (a.j - b.j).abs(),
                (a.e_cation - b.e_cation).abs()
            );
        }
    }
}

/// The issue's N2 case, as the MEASUREMENT found it: ω = 0.50, 0.53, 0.56,
/// 0.60 on N2/def2-SVP ωB97X-V.
///
/// What this asserts is what was measured, which is NOT what the issue
/// anticipated. Across that window:
///
/// * Every evaluation — default guess or continued — converges to the SAME
///   symmetric cation: |ΔE_cation| ≤ 2.8e-13 Ha between the two arms, spin
///   asymmetry at the 1e-10…1e-8 numerical floor, ⟨S²⟩ a smooth
///   0.755622 → 0.756686.
/// * J is therefore smooth and monotone across the onset PySCF's orbital
///   Hessian puts between ω = 0.53 (λ_min +1.1e-4) and 0.56 (λ_min −2.7e-3).
///   ferric's J on that branch matches PySCF's probe to 7.4e-8 / 4.7e-7 Ha.
/// * A lower cation state DOES exist past the onset and ferric reaches it when
///   SEEDED toward it (β HOMO/LUMO 45° rotation): −8.9e-5 Ha at ω = 0.56,
///   −4.6e-4 at 0.60, −2.1e-3 at 0.70.
///
/// So neither continuation nor any observable this module can compute flags
/// this onset: the curve is self-consistent, single-branch and smooth, and
/// what has changed is a CURVATURE, which only an orbital Hessian sees.
/// ferric's refuses ω ≠ 0 (`StabilitySkip::RangeSeparated`). The assertions
/// below pin that state of affairs so it cannot regress silently in either
/// direction — if a future change makes the plain tuner reach the lower
/// branch, the monotonicity assert fails and this doc is wrong.
#[test]
#[ignore = "validation: RSH omega tuning"]
fn n2_onset_is_not_visible_without_an_orbital_hessian() {
    let (mol, prep, bounds) = load("validation/n2.xyz", "def2-svp");
    let ctx = ParallelContext::default();
    let scf = RhfConfig {
        df_j_aux: Some(String::new()),
        df_k_aux: Some("def2-universal-jkfit".into()),
        energy_conv: 1e-10,
        density_conv: 1e-7,
        max_iter: 500,
        ..Default::default()
    };
    let c = cfg(scf.clone(), 0.2, 0.8);
    let omegas = [0.50_f64, 0.53, 0.56, 0.60];

    // Independent solves (the default) and continued solves must agree: both
    // are on the symmetric branch, so this is the "continuation changed
    // nothing here" measurement, asserted.
    let mut indep = Vec::new();
    let mut cont = Vec::new();
    let mut prev: Option<OmegaSeedState> = None;
    for &w in &omegas {
        let a = eval_j(&ctx, &mol, &prep, &bounds, &c, w).unwrap();
        let (b, carry) = eval_j_seeded(&ctx, &mol, &prep, &bounds, &c, w, prev.as_ref()).unwrap();
        eprintln!(
            "w={w:.2}: indep E_cat={:+.9} J={:+.6e} asym={:.2e} S2={:.6} | \
             cont E_cat={:+.9} J={:+.6e} asym={:.2e} | dE={:.2e}",
            a.e_cation,
            a.j,
            a.cation_spin_asymmetry,
            a.cation_s_squared,
            b.e_cation,
            b.j,
            b.cation_spin_asymmetry,
            (a.e_cation - b.e_cation).abs()
        );
        // Measured 1.1e-13 … 2.8e-13 Ha; the bar is two decades above that
        // and far below the 8.9e-5 Ha a branch switch would cost.
        assert!(
            (a.e_cation - b.e_cation).abs() < 1e-11,
            "w={w}: continuation moved the cation energy by {:.2e} Ha — it reached a \
             different state than the independent solve, which this window was \
             measured NOT to do",
            (a.e_cation - b.e_cation).abs()
        );
        // Both arms sit at the asymmetry floor: the hole is SHARED over the two
        // equivalent N at every ω, including past the onset.
        for (tag, e) in [("indep", &a), ("cont", &b)] {
            assert!(
                e.cation_spin_asymmetry < 1e-6,
                "w={w} ({tag}): spin asymmetry {:.2e} — the plain tuner reached a \
                 hole-localized cation, which it was measured not to do; this test's \
                 doc and DEFAULT_BRANCH_TOL's derivation both need re-measuring",
                e.cation_spin_asymmetry
            );
        }
        indep.push(a);
        cont.push(b);
        prev = Some(carry);
    }

    // J is strictly decreasing across the onset on this branch, with no kink:
    // there is no discontinuity for a branch check to find.
    for pair in indep.windows(2) {
        assert!(
            pair[1].j < pair[0].j,
            "J is not monotone across {:.2}→{:.2} ({:+.3e} → {:+.3e})",
            pair[0].omega,
            pair[1].omega,
            pair[0].j,
            pair[1].j
        );
    }
    // ⟨S²⟩ drifts smoothly and in one direction: no ⟨S²⟩ bar can separate
    // "ω moved" from "state changed" here, which is why branch_tol gates on
    // the asymmetry alone.
    for pair in indep.windows(2) {
        let d = pair[1].cation_s_squared - pair[0].cation_s_squared;
        assert!(
            d > 0.0 && d < 1e-3,
            "⟨S²⟩ step {:.3e} between ω {:.2} and {:.2} is not the measured smooth drift",
            d,
            pair[0].omega,
            pair[1].omega
        );
    }

    // The lower state EXISTS and the tuner does not find it. Seeded toward it
    // (β occ→virt rotation), ω = 0.60's cation drops measurably below the
    // symmetric one — so the smooth J curve above is not the ground-state
    // curve, and nothing in this module says so.
    let scf60 = ferric_scf::omega_tuning::state_scf_config(&c, 0.60).unwrap();
    let mut cation = mol.clone();
    cation.charge += 1;
    cation.multiplicity = 2;
    let nocc_b = (cation.nelec() as usize) / 2;
    let n = prep.nbasis();
    let sym = solve_uhf(&ctx, &cation, &prep, &bounds, &scf60).unwrap();
    let mut cb = sym.mos_beta.clone().unwrap();
    let r = std::f64::consts::FRAC_1_SQRT_2;
    for mu in 0..n {
        let (a, b) = (cb[(mu, nocc_b - 1)], cb[(mu, nocc_b)]);
        cb[(mu, nocc_b - 1)] = r * (a + b);
        cb[(mu, nocc_b)] = r * (a - b);
    }
    let lower = solve_uhf_with_guess(
        &ctx,
        &cation,
        &prep,
        &bounds,
        &scf60,
        Some((&sym.mos_alpha, &cb)),
    )
    .unwrap();
    let drop = sym.energy - lower.energy;
    eprintln!(
        "w=0.60: symmetric {:+.9}, seeded-lower {:+.9}, drop {:.3e} Ha",
        sym.energy, lower.energy, drop
    );
    // Measured 4.58e-4 Ha. A bar well below it, and clearly above the 1e-13
    // agreement of two solves of the SAME state.
    assert!(
        drop > 1e-5,
        "the seeded cation at ω = 0.60 is not below the symmetric one (drop {drop:.3e} Ha) — \
         then the premise that a lower branch exists there is wrong and the limitation \
         documented in this module needs re-measuring, not restating"
    );
    // ...and the plain tuner's ω = 0.60 point is the HIGHER state, i.e. today's
    // J curve past the onset is not on the lowest cation state available.
    let indep_60 = indep.last().unwrap();
    assert!(
        indep_60.e_cation > lower.energy + 1e-5,
        "the independent ω = 0.60 solve already found the lower state ({:+.9} vs {:+.9})",
        indep_60.e_cation,
        lower.energy
    );
}
