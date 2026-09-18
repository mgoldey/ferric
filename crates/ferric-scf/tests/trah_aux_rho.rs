//! **THE CENTRAL MEASUREMENT: the rho distribution under auxiliary curvature.**
//!
//! `trah_aux_curvature.rs` establishes that the substitution is correctly
//! scaled, correctly gated, and reaches the same fixed point. None of that
//! answers the actual research question, which is whether AURORA's cheap
//! STO-3G curvature can carry TRAH's trust region.
//!
//! It can only do that if the quadratic model it defines predicts the real
//! energy change. The trust region's whole control law -- accept, reject,
//! expand, contract -- is a function of
//!
//! ```text
//!   rho = dE_actual / dE_predicted
//! ```
//!
//! so rho near 1 is not a nice-to-have, it IS the criterion. A model that gives
//! the right search direction but the wrong predicted magnitude will still
//! converge on easy systems while quietly mis-classifying every step against
//! Fletcher's 0.25/0.75 thresholds -- and will drive spurious rejections
//! exactly on the hard systems where TRAH is supposed to earn its keep.
//!
//! # Reading the output
//!
//! Iteration counts and matvec counts are integers and are load-immune. Wall
//! time is NOT reported here on purpose: this file is meant to be runnable on a
//! busy box, and a seconds column would invite exactly the comparison that
//! this workspace has already been burned by (138% run-to-run spread).
//!
//! # What a repeated constant means
//!
//! If rho comes back as the SAME value to many decimals across iterations, that
//! is arithmetic, not measurement -- this module's own history records the
//! pattern twice (`trah.rs`, the `predicted_min` and `RHF_ENERGY_SCALE` notes).
//! The assertions below therefore check the SPREAD as well as the level.

use std::sync::Mutex;

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ferric_scf::trah::{aux_matvecs, exact_matvecs, reset_rho_log, rho_log_rhf, TrahConfig};

static LOCK: Mutex<()> = Mutex::new(());

struct Run {
    iters: usize,
    matvecs: usize,
    energy: f64,
    converged: bool,
    rhos: Vec<f64>,
}

fn run(xyz: &str, bas: &str, xc: Option<&str>, aux: bool) -> Run {
    let mol = Molecule::load_xyz(xyz).expect("xyz");
    let bs = basis::bundled(bas).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let ctx = ParallelContext::default();

    let cfg = RhfConfig {
        xc: xc.map(|s| s.to_string()),
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 300,
        trah_trigger: Some(1e-2),
        trah: TrahConfig {
            aux_curvature: aux,
            ..Default::default()
        },
        ..Default::default()
    };

    reset_rho_log();
    let m0 = if aux { aux_matvecs() } else { exact_matvecs() };
    let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).expect("scf");
    let m1 = if aux { aux_matvecs() } else { exact_matvecs() };

    Run {
        iters: r.iterations,
        matvecs: m1 - m0,
        energy: r.energy,
        converged: r.converged,
        rhos: rho_log_rhf(),
    }
}

fn describe(tag: &str, r: &Run) {
    let n = r.rhos.len();
    println!(
        "  {tag:<26} iters={:<4} matvecs={:<6} conv={} E={:.10}",
        r.iters, r.matvecs, r.converged, r.energy
    );
    if n == 0 {
        println!("  {:<26} (no rho formed -- TRAH never assessed a step)", "");
        return;
    }
    let mean = r.rhos.iter().sum::<f64>() / n as f64;
    let mut sorted = r.rhos.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = sorted[n / 2];
    let dev = (r.rhos.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    let worst = r.rhos.iter().fold(0.0f64, |m, x| {
        if (x - 1.0).abs() > m {
            (x - 1.0).abs()
        } else {
            m
        }
    });
    println!(
        "  {:<26} rho: n={n} mean={mean:+.4} median={med:+.4} sd={dev:.4} \
         min={:+.4} max={:+.4} worst|rho-1|={worst:.4}",
        "",
        sorted[0],
        sorted[n - 1]
    );
    let shown: Vec<String> = r.rhos.iter().map(|x| format!("{x:+.4}")).collect();
    println!("  {:<26} [{}]", "", shown.join(" "));
}

/// The measurement. `#[ignore]`d: it is a harness, not a gate, and it runs
/// several SCFs.
#[test]
#[ignore = "measurement harness: run deliberately"]
fn rho_distribution_aux_vs_exact() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let systems: &[(&str, &str, &str, Option<&str>)] = &[
        (
            "water/cc-pVDZ RHF",
            "../../testdata/molecules/water.xyz",
            "cc-pvdz",
            None,
        ),
        (
            "water/cc-pVDZ B3LYP",
            "../../testdata/molecules/water.xyz",
            "cc-pvdz",
            Some("B3LYP"),
        ),
        (
            "water/cc-pVDZ PBE",
            "../../testdata/molecules/water.xyz",
            "cc-pvdz",
            Some("PBE"),
        ),
        (
            "benzene/STO-3G RHF",
            "../../testdata/molecules/benzene.xyz",
            "sto-3g",
            None,
        ),
    ];

    println!("\n=== rho distribution: EXACT vs AUXILIARY curvature ===\n");
    for (name, xyz, bas, xc) in systems {
        println!("{name}");
        let ex = run(xyz, bas, *xc, false);
        describe("exact curvature", &ex);
        let ax = run(xyz, bas, *xc, true);
        describe("aux curvature", &ax);
        println!(
            "  {:<26} dE(aux-exact) = {:.3e} Ha",
            "",
            (ax.energy - ex.energy).abs()
        );
        println!();
    }
    println!(
        "Iterations and matvecs are load-immune integers. A rho that repeats to \
         many decimals is arithmetic, not measurement.\n"
    );
}

/// **The cost comparison, on load-immune integers only.**
///
/// The premise being tested is a WALL-TIME one: TRAH's iteration win is real
/// but each step pays a Davidson of full-basis J/K builds, so a cheaper matvec
/// should convert the iteration win into a time win. That argument needs the
/// matvec COUNT to hold roughly steady while the per-matvec cost drops.
///
/// It does not. Measured here (and this is why no seconds column appears: the
/// counts already settle it). DIIS is included as the reference both TRAH arms
/// have to beat.
#[test]
#[ignore = "measurement harness: run deliberately"]
fn matvec_and_iteration_counts() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let systems: &[(&str, &str, &str, Option<&str>)] = &[
        (
            "water/cc-pVDZ RHF",
            "../../testdata/molecules/water.xyz",
            "cc-pvdz",
            None,
        ),
        (
            "water/cc-pVDZ B3LYP",
            "../../testdata/molecules/water.xyz",
            "cc-pvdz",
            Some("B3LYP"),
        ),
        (
            "benzene/STO-3G RHF",
            "../../testdata/molecules/benzene.xyz",
            "sto-3g",
            None,
        ),
    ];

    println!(
        "\n{:<24}{:>10}{:>12}{:>10}{:>12}{:>10}",
        "system", "it_exact", "mv_exact", "it_aux", "mv_aux", "it_DIIS"
    );
    println!("{}", "-".repeat(78));
    for (name, xyz, bas, xc) in systems {
        let ex = run(xyz, bas, *xc, false);
        let ax = run(xyz, bas, *xc, true);
        let diis = run_diis(xyz, bas, *xc);
        println!(
            "{name:<24}{:>10}{:>12}{:>10}{:>12}{:>10}",
            ex.iters, ex.matvecs, ax.iters, ax.matvecs, diis
        );
    }
    println!(
        "\nA cheaper matvec only pays if the COUNT holds. It does not: the \
         auxiliary operator needs more Davidson iterations per step AND more \
         macro steps.\n"
    );
}

/// DIIS iteration count for the same system (the arm both TRAH variants must
/// beat on wall time to be worth anything).
fn run_diis(xyz: &str, bas: &str, xc: Option<&str>) -> usize {
    let mol = Molecule::load_xyz(xyz).expect("xyz");
    let bs = basis::bundled(bas).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        xc: xc.map(|s| s.to_string()),
        energy_conv: 1e-10,
        density_conv: 1e-8,
        max_iter: 300,
        ..Default::default()
    };
    solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg)
        .expect("scf")
        .iterations
}

/// **THE FALSIFICATION RESULT, pinned.** Auxiliary curvature biases rho
/// systematically BELOW 1; exact curvature does not.
///
/// # Why this assertion and not "|rho - 1| < tol"
///
/// That was the test written first, with a generous |rho - 1| <= 0.5 band
/// chosen a priori from Fletcher's 0.25/0.75 classification thresholds. It
/// PASSED (worst aux deviation 0.286) -- and passing was uninformative, because
/// the band was wider than the effect. Recording it as a pass would have meant
/// reporting "aux curvature keeps rho near 1", which the numbers underneath it
/// flatly contradict:
///
/// ```text
///   exact  [+0.9933 +0.9928 +0.9895]                          sd 0.0017
///   aux    [+0.8870 +0.8210 +0.7823 +0.7496 +0.7309 +0.7229 +0.7143]
/// ```
///
/// Every aux value is below every exact value, and they DECREASE monotonically
/// as the step shortens. That is a systematic bias, not scatter, so the right
/// instrument is a comparison of the two distributions -- not a tolerance on
/// one of them. A max-deviation test is blind to bias by construction: it
/// cannot tell 0.71 from 1.29.
///
/// The bias has the sign the physics predicts. The STO-3G auxiliary model
/// UNDERSTATES the Coulomb response (7 fitting functions for a 24x24
/// orbital-product space in cc-pVDZ), so the model thinks the surface is
/// flatter than it is, promises a larger energy drop than materializes, and
/// lands rho below 1. Prediction written before the measurement; direction and
/// sign both confirmed.
#[test]
#[ignore = "measurement harness: run deliberately"]
fn aux_curvature_biases_rho_below_one() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let ex = run("../../testdata/molecules/water.xyz", "cc-pvdz", None, false);
    let ax = run("../../testdata/molecules/water.xyz", "cc-pvdz", None, true);

    describe("exact", &ex);
    describe("aux", &ax);

    // The comparison is only meaningful if the exact path itself behaves.
    assert!(
        !ex.rhos.is_empty(),
        "the exact baseline formed no rho -- nothing to compare against"
    );
    let ex_worst = ex.rhos.iter().fold(0.0f64, |m, x| m.max((x - 1.0).abs()));
    assert!(
        ex_worst < 0.5,
        "the EXACT-curvature baseline is itself off 1 (worst |rho-1| = \
         {ex_worst:.3}); the aux comparison would be measuring the wrong thing"
    );

    assert!(
        !ax.rhos.is_empty(),
        "the aux path formed no rho -- the branch did not run"
    );

    // The exact path's rho sits on 1 with tight spread. Pin that, so this test
    // is comparing against a baseline that is itself behaving.
    let ex_mean = ex.rhos.iter().sum::<f64>() / ex.rhos.len() as f64;
    let ax_mean = ax.rhos.iter().sum::<f64>() / ax.rhos.len() as f64;
    assert!(
        ex_mean > 0.97,
        "the exact baseline's mean rho is {ex_mean:.4}, not ~1 -- the \
         comparison below would be measuring the wrong thing"
    );

    // THE RESULT: every aux rho is below every exact rho. This is the
    // separation that a tolerance band could not see.
    let ex_min = ex.rhos.iter().cloned().fold(f64::INFINITY, f64::min);
    let ax_max = ax.rhos.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        ax_max < ex_min,
        "expected a clean separation (every aux rho below every exact rho): \
         aux max {ax_max:.4} vs exact min {ex_min:.4}.\n  exact {:?}\n  aux   {:?}",
        ex.rhos,
        ax.rhos
    );
    assert!(
        ax_mean < 0.85,
        "aux mean rho is {ax_mean:.4}; the measured bias was 0.77. If this has \
         risen towards 1 the auxiliary model got better and the negative \
         verdict in this file should be re-examined, not patched."
    );

    // And the bias is not noise: it is larger than the exact path's spread by
    // orders of magnitude. Without this the separation above could be luck.
    let ex_sd =
        (ex.rhos.iter().map(|x| (x - ex_mean).powi(2)).sum::<f64>() / ex.rhos.len() as f64).sqrt();
    let bias = ex_mean - ax_mean;
    assert!(
        bias > 10.0 * ex_sd.max(1e-6),
        "the aux-vs-exact bias ({bias:.4}) must dominate the exact path's own \
         scatter ({ex_sd:.6}) for the separation to mean anything"
    );

    // The aux path still reaches the same answer -- it is a worse MODEL, not a
    // wrong one. An accelerator changes the path, never the fixed point.
    let de = (ax.energy - ex.energy).abs();
    assert!(
        de < 1e-8,
        "aux curvature moved the fixed point by {de:.3e} Ha"
    );

    // And it costs MORE, not less, in matvecs -- the opposite of the premise.
    assert!(
        ax.matvecs > ex.matvecs,
        "measured: aux needed {} matvecs vs exact {}. If aux is now CHEAPER in \
         matvec count too, the cost argument in this file's verdict changes.",
        ax.matvecs,
        ex.matvecs
    );
}
