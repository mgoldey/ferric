//! Stage 2 KILL GATE for seminumerical exchange (COSX / sn-LinK).
//!
//! # The question
//!
//! A prior design study measured that a COSX A-build on butane is ~140x
//! SLOWER than ferric's own analytic K at cc-pVTZ, and that the A-matrix is
//! ~93-97% dense at 1e-10 on that system. Butane is 10.5 Bohr across, far
//! below the ~30 Bohr locality onset, so that density measurement cannot
//! settle the question -- the repo's DO-NOT-DECLARE-A-NEGATIVE-BELOW-THE-ONSET
//! rule forbids concluding from it.
//!
//! This sweep therefore measures the surviving-shell-pair fraction on an
//! alkane series that REACHES PAST the onset:
//!
//! ```text
//!   alkane_4  14 atoms  10.5 Bohr   (below onset -- the old butane datum)
//!   alkane_8  26 atoms  19.9 Bohr   (below)
//!   alkane_12 38 atoms  29.3 Bohr   (at)
//!   alkane_16 50 atoms  38.7 Bohr   (past)
//!   alkane_20 62 atoms  48.1 Bohr   (well past)
//! ```
//!
//! # GO requires ALL of
//!
//! (a) surviving-pair FRACTION falls with system size. If it is flat there is
//!     no asymptotic win and the method can never close a 140x deficit.
//! (b) a tail-fit exponent (LAST THREE POINTS ONLY, per repo rule) measurably
//!     below the dense path's exponent of 2 (pairs grow as nsh^2).
//! (c) a credible extrapolated crossover vs ferric's OWN analytic K at a size
//!     someone would actually run.
//!
//! # Anchors (both written BEFORE this sweep, in cosx_a_anchor.rs)
//!
//! (i)  `cosx_a_zero_threshold_matches_unscreened` -- the trivial limit.
//! (ii) `cosx_screen_actually_drops_pairs` -- reachability, so a "fraction
//!      falls" result cannot be an artifact of a screen that drops nothing.
//!
//! # Artifact hypothesis (pre-registered)
//!
//! If locality is REAL: the surviving-pair COUNT PER GRID POINT saturates to a
//! constant once the molecule exceeds the decay length, so the FRACTION
//! (count / total pairs) falls like 1/nsh^2 and the tail exponent of the count
//! approaches ~0-1 rather than 2.
//! If my screen is BROKEN or merely geometric-without-decay: the fraction is
//! FLAT in system size (a fixed proportion of pairs survives at every size),
//! because a scale-free 1/d criterion has no intrinsic length and just rescales
//! with the molecule. FLAT is the artifact signature AND the no-win signature
//! simultaneously -- they are indistinguishable here, which is FINE because
//! both imply NO-GO. The distinguishing case is only needed for a GO, where a
//! falling fraction must additionally be shown non-vacuous by anchor (ii).
//!
//! Run with `--ignored --nocapture`; these are measurements, not assertions
//! about the physics, and must not gate CI.

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_dft::grid::{build_atomic_grid, AtomicGridConfig};
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::cosx_a::{a_matrix_at_point_with, CosxScreen, PairBounds};
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use std::time::Instant;

/// Production screening threshold for the sweep. Chosen ONCE, before any
/// measurement, and NOT tuned afterwards (the brief forbids tuning until it
/// looks good). 1e-7 is a standard COSX-class accuracy target.
const PROD_THRESH: f64 = 1e-7;

struct Row {
    label: String,
    natoms: usize,
    nsh: usize,
    nbf: usize,
    npts: usize,
    total_pairs: usize,
    kept_mean: f64,
    frac: f64,
    secs_per_point: f64,
}

/// Measure the surviving-pair statistics over a SAMPLE of grid points.
///
/// Sampling (rather than the whole grid) is deliberate: the fraction is an
/// average over grid points, and a uniform stride across the full point list
/// samples every region of the molecule (each atom's radial shells appear in
/// order), so the mean is unbiased. This keeps the sweep affordable at
/// alkane_20/cc-pVTZ where the full grid is millions of points.
/// Resolve a repo-relative testdata path. Cargo runs integration tests with
/// the CRATE directory as cwd, not the workspace root, so a bare relative path
/// does not resolve.
fn testdata(rel: &str) -> String {
    format!("{}/../../{}", env!("CARGO_MANIFEST_DIR"), rel)
}

fn measure(label: &str, xyz: &str, basis: &str, sample: usize) -> Row {
    let mol = Molecule::load_xyz(&testdata(xyz)).expect("xyz");
    let bs = bundled(basis).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let bounds = PairBounds::build(&prep).expect("bounds");

    let cfg = AtomicGridConfig { n_radial: 50, n_angular: 110, ..Default::default() };
    let grid = build_atomic_grid(&mol, &cfg);
    let npts = grid.len();

    // Uniform stride over the full grid.
    let stride = (npts / sample).max(1);
    let pts: Vec<[f64; 3]> = grid.iter().step_by(stride).map(|g| g.xyz).collect();

    let mut eng = Engine::new_1e(ffi::OP_NUCLEAR, &prep, 1e-14).expect("engine");

    let mut kept_sum = 0usize;
    let mut total_pairs = 0usize;
    let t0 = Instant::now();
    for r in &pts {
        let p = a_matrix_at_point_with(&mut eng, &prep, r, Some(&bounds), CosxScreen::at(PROD_THRESH))
            .expect("A build");
        kept_sum += p.pairs_kept;
        total_pairs = p.pairs_total;
    }
    let elapsed = t0.elapsed().as_secs_f64();

    let kept_mean = kept_sum as f64 / pts.len() as f64;
    Row {
        label: label.to_string(),
        natoms: mol.atoms.len(),
        nsh: prep.nshells(),
        nbf: prep.nbasis(),
        npts,
        total_pairs,
        kept_mean,
        frac: kept_mean / total_pairs as f64,
        secs_per_point: elapsed / pts.len() as f64,
    }
}

/// Tail-fit exponent over the LAST THREE points only (repo rule: a global fit
/// that averages in pre-onset points hides the effect).
fn tail_exponent(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    assert!(n >= 3, "need >= 3 points for a tail fit");
    let (x, y) = (&xs[n - 3..], &ys[n - 3..]);
    let lx: Vec<f64> = x.iter().map(|v| v.ln()).collect();
    let ly: Vec<f64> = y.iter().map(|v| v.ln()).collect();
    let mx = lx.iter().sum::<f64>() / 3.0;
    let my = ly.iter().sum::<f64>() / 3.0;
    let num: f64 = lx.iter().zip(&ly).map(|(a, b)| (a - mx) * (b - my)).sum();
    let den: f64 = lx.iter().map(|a| (a - mx).powi(2)).sum();
    num / den
}

fn run_series(basis: &str, sample: usize) {
    let systems = [
        ("alkane_4", "testdata/molecules/alkane_4.xyz"),
        ("alkane_8", "testdata/molecules/alkane_8.xyz"),
        ("alkane_12", "testdata/molecules/alkane_12.xyz"),
        ("alkane_16", "testdata/molecules/alkane_16.xyz"),
        ("alkane_20", "testdata/molecules/alkane_20.xyz"),
    ];

    println!("\n=== COSX Stage 2 sweep: basis={basis}, threshold={PROD_THRESH:.0e}, \
              grid=(50,110), sample={sample} pts ===");
    println!(
        "{:>10} {:>7} {:>6} {:>6} {:>10} {:>11} {:>11} {:>9} {:>12}",
        "system", "natoms", "nsh", "nbf", "npts", "pairs_tot", "kept_mean", "frac", "s/point"
    );

    let mut rows = Vec::new();
    for (label, path) in systems {
        let r = measure(label, path, basis, sample);
        println!(
            "{:>10} {:>7} {:>6} {:>6} {:>10} {:>11} {:>11.1} {:>9.4} {:>12.3e}",
            r.label, r.natoms, r.nsh, r.nbf, r.npts, r.total_pairs, r.kept_mean, r.frac,
            r.secs_per_point
        );
        rows.push(r);
    }

    // --- GO bar (a): does the fraction FALL with system size? ---
    let fracs: Vec<f64> = rows.iter().map(|r| r.frac).collect();
    let falling = fracs.windows(2).all(|w| w[1] < w[0]);
    let drop = fracs[0] / fracs[fracs.len() - 1];
    println!("\n(a) surviving-pair fraction monotonically falling: {falling}  \
              (first {:.4} -> last {:.4}, ratio {:.2}x)", fracs[0], fracs[fracs.len()-1], drop);

    // --- GO bar (b): tail exponent of the KEPT-PAIR COUNT vs nsh ---
    // Dense reference: total pairs grow as nsh^2, so the dense exponent is 2.
    let nsh: Vec<f64> = rows.iter().map(|r| r.nsh as f64).collect();
    let kept: Vec<f64> = rows.iter().map(|r| r.kept_mean).collect();
    let tot: Vec<f64> = rows.iter().map(|r| r.total_pairs as f64).collect();
    let e_kept = tail_exponent(&nsh, &kept);
    let e_tot = tail_exponent(&nsh, &tot);
    println!("(b) tail exponent (last 3) of kept-pairs vs nsh : {e_kept:.3}");
    println!("    tail exponent (last 3) of TOTAL pairs vs nsh: {e_tot:.3}  (dense reference)");
    println!("    -> screened path is {} the dense path",
        if e_kept < e_tot - 0.1 { "ASYMPTOTICALLY BETTER than" } else { "NOT better than" });

    // --- Per-point cost scaling, the thing that actually decides feasibility ---
    let spp: Vec<f64> = rows.iter().map(|r| r.secs_per_point).collect();
    let e_cost = tail_exponent(&nsh, &spp);
    println!("(c) tail exponent (last 3) of A-build s/point vs nsh: {e_cost:.3}");
    for r in &rows {
        // Cost of ONE full K build = s/point * npts (one A sweep per grid point).
        println!("    {:>10}: full-grid A-build per SCF iteration = {:.1} s",
            r.label, r.secs_per_point * r.npts as f64);
    }
}

#[test]
#[ignore = "Stage 2 measurement sweep; run explicitly with --ignored"]
fn cosx_stage2_dz() {
    run_series("cc-pvdz", 60);
}

#[test]
#[ignore = "Stage 2 measurement sweep; run explicitly with --ignored"]
fn cosx_stage2_tz() {
    run_series("cc-pvtz", 30);
}
