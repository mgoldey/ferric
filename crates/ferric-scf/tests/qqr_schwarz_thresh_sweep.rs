//! THRESHOLD SWEEP for the QQR-vs-Schwarz screening difference on
//! alkane_8/cc-pVDZ.
//!
//! Adapted from the assertion-free audit harness on branch
//! `audit/qqr-vs-schwarz` (`scripts/queue/out/qqr_vs_schwarz_verdict.md`), with
//! the assertions ADDED so this is a guard rather than a printout.
//!
//! # The question this answers that a single-point bar cannot
//!
//! `screening_exactness::link_k_qqr_matches_schwarz_at_production_thresh`
//! measures one number at one threshold. That number alone cannot tell
//! TRUNCATION (the two arms discard different tails, and the difference shrinks
//! as the threshold shrinks) from an ENVELOPE DEFECT (QQR discards quartets
//! carrying real weight, leaving an error FLOOR that does not follow the
//! threshold down). This sweep asserts the falling behaviour directly, which
//! is what actually distinguishes the two.
//!
//! # Measured, 2026-09-08, on `fix/link-pairwise-screen`
//!
//! ```text
//!   thresh   max|K_QQR-K_Sch|   max|K_Sch-K_dense|   max|K_QQR-K_dense|
//!   1e-6     1.2213e-5          6.4645e-6            1.3929e-5
//!   1e-8     1.4889e-7          5.9965e-8            1.4917e-7
//!   1e-10    6.1421e-10         6.6451e-10           6.6050e-10
//!   1e-12    6.7678e-12         6.3324e-12           7.0008e-12
//! ```
//!
//! The difference falls by ~six orders of magnitude across six orders of
//! magnitude of threshold (~6-15x thresh throughout). There is no floor.
//!
//! Runtime is dominated by the alkane_8 RHF, which is run ONCE and shared by
//! every threshold; `#[ignore]`d because the remaining quartet contractions
//! still cost minutes.
//!
//! Run:
//! ```text
//! OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
//!   cargo test -p ferric-scf --release --test qqr_schwarz_thresh_sweep \
//!   -- --ignored --nocapture
//! ```

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::fock::KBuilder;
use ferric_scf::link_k::LinkK;
use ferric_scf::qqr::QqrBounds;
use ferric_scf::rhf::{build_jk, solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

/// FALLING-WITH-THRESHOLD GUARD: the QQR-vs-Schwarz difference must track the
/// screening threshold, which is what makes it truncation rather than a defect.
///
/// Calibrated from the table in this file's module doc. Between thresh=1e-8 and
/// thresh=1e-10 — a 100x drop in threshold — the measured difference falls
/// 1.4889e-7 -> 6.1421e-10, a factor of 242. The assertion demands only 10x, so
/// it has a 24x margin against jitter while still being unreachable for any
/// error FLOOR: a floor by definition does not fall at all.
///
/// # SCOPE LIMIT — what these assertions do NOT catch, measured
///
/// This is recorded rather than hidden, because it is the difference between a
/// guard and an assumption. MUTATION TESTED on this branch, 2026-09-08: the
/// "envelope 100x too aggressive" mutant (`decay *= 0.01` in
/// [`ferric_scf::qqr`]) raises the thresh=1e-8 difference from 1.4889e-7 to
/// **1.3581e-5**, a 91x separation — and yet it PASSES BOTH assertions below:
///
/// ```text
///   quantity                       correct     100x-aggressive mutant
///   fall 1e-8 -> 1e-10             242x        95.6x    (clears the 10x gate)
///   |QQR-Sch| / |QQR-dense| @1e-8  0.998       0.9991   (indistinguishable)
///   max|K_QQR - K_Schwarz| @1e-8   1.4889e-7   1.3581e-5  <-- ONLY this separates
/// ```
///
/// The reason is structural, not a calibration accident: a UNIFORMLY scaled-down
/// envelope is still a monotone function of the threshold, so its error still
/// falls; and it errs against the dense reference by the same amount it errs
/// against the Schwarz arm, so the ratio stays ~1. The ratio invariant tests
/// that QQR's disagreement with Schwarz IS its disagreement with truth, which
/// catches an envelope wrong in a DIRECTION Schwarz is not — a different defect
/// class from being wrong in MAGNITUDE.
///
/// So the absolute-magnitude bar in
/// `screening_exactness::link_k_qqr_matches_schwarz_at_production_thresh` is
/// what catches the over-aggressive envelope, and these two assertions cover
/// the defect classes IT cannot see (an error floor, and a direction error).
/// The three are complementary; none is redundant.
#[test]
#[ignore = "sweep: minutes (one alkane_8 RHF + 4 thresholds x 3 K builds)"]
fn qqr_vs_schwarz_difference_falls_with_threshold_alkane_8() {
    let mol = Molecule::load_xyz("../../testdata/molecules/alkane_8.xyz").expect("alkane_8");
    let bs = basis::bundled("cc-pvdz").expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("PreparedBasis");
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let n = prep.nbasis();

    // ONE converged density, shared by every threshold, so the sweep varies
    // exactly one thing (and so the SCF is paid once, not four times).
    let schwarz_for_scf = SchwarzBounds::compute(op, &prep).expect("Schwarz");
    let res = solve_rhf(
        &ctx,
        &mol,
        &prep,
        op,
        &schwarz_for_scf,
        &RhfConfig::default(),
    )
    .expect("RHF");
    assert!(res.converged, "RHF must converge before measuring screening");
    let d = res.density_total;

    let thresholds = [1e-6f64, 1e-8, 1e-10, 1e-12];
    let mut qqr_vs_schwarz = Vec::new();
    let mut qqr_vs_dense = Vec::new();

    println!("# alkane_8/cc-pVDZ, nbf={n}, converged RHF density");
    println!("# thresh  max|K_QQR-K_Schwarz|  max|K_Sch-K_dense|  max|K_QQR-K_dense|");

    for &thresh in &thresholds {
        let schwarz = SchwarzBounds::compute(op, &prep).expect("Schwarz");
        let qqr = QqrBounds::new(
            SchwarzBounds::compute(op, &prep).expect("Schwarz"),
            &mol,
            &bs,
            &prep,
            op,
        );

        let mut k_sch = Array2::zeros((n, n));
        {
            let mut link = LinkK::new(&ctx, &prep, &schwarz, op, thresh, usize::MAX);
            link.update_density(&d);
            link.build(&d, &mut k_sch).expect("LinK Schwarz");
        }

        let mut k_qqr = Array2::zeros((n, n));
        {
            let mut link = LinkK::new(&ctx, &prep, &qqr, op, thresh, usize::MAX);
            link.update_density(&d);
            link.build(&d, &mut k_qqr).expect("LinK QQR");
        }

        // Dense reference at the SAME threshold, so each arm's own truncation
        // is visible next to their mutual difference.
        let mut j_ref = Array2::zeros((n, n));
        let mut k_dense = Array2::zeros((n, n));
        build_jk(&ctx, &prep, &schwarz, thresh, &d, &mut j_ref, &mut k_dense)
            .expect("dense build_jk");

        let d_qs = max_abs_diff(&k_qqr, &k_sch);
        let d_sd = max_abs_diff(&k_sch, &k_dense);
        let d_qd = max_abs_diff(&k_qqr, &k_dense);
        println!("{thresh:.0e}  {d_qs:.4e}  {d_sd:.4e}  {d_qd:.4e}");

        qqr_vs_schwarz.push(d_qs);
        qqr_vs_dense.push(d_qd);
    }

    // (1) The difference must FALL with the threshold. Asserted between 1e-8
    // and 1e-10 (indices 1 and 2), where the measured ratio is 242x; the bar
    // is 10x. This catches an error FLOOR, which by definition does not fall.
    // It does NOT catch a uniformly over-aggressive envelope (the 100x mutant
    // falls 95.6x here) — see the scope limit in this test's doc comment.
    let e_1e8 = qqr_vs_schwarz[1];
    let e_1e10 = qqr_vs_schwarz[2];
    assert!(
        e_1e10 * 10.0 < e_1e8,
        "alkane_8: max|K_QQR - K_Schwarz| is {e_1e8:.4e} at thresh=1e-8 but {e_1e10:.4e} at \
         thresh=1e-10 — a factor of only {:.1}x for a 100x tighter threshold. Measured 242x \
         (1.4889e-7 -> 6.1421e-10) when this was calibrated. A difference that does NOT fall \
         with the threshold is an error FLOOR, i.e. the QQR envelope \
         (crates/ferric-scf/src/qqr.rs) is discarding quartets that carry real weight rather \
         than merely truncating a smaller tail than Schwarz does.",
        e_1e8 / e_1e10.max(f64::MIN_POSITIVE)
    );

    // (2) Monotone across the whole sweep, so a floor appearing at any point in
    // the range is caught, not just between the two calibrated points.
    for w in qqr_vs_schwarz.windows(2) {
        assert!(
            w[1] < w[0],
            "alkane_8: max|K_QQR - K_Schwarz| is not monotone in the threshold: {:.4e} -> {:.4e} \
             across the sweep {thresholds:?}. Every tightening of the threshold must reduce the \
             difference; a step that does not is a floor.",
            w[0],
            w[1]
        );
    }

    // (3) The ratio invariant, at the production threshold 1e-8. QQR's
    // disagreement with the Schwarz arm must be its disagreement with truth.
    // Measured 1.4889e-7 / 1.4917e-7 = 0.998. This catches an envelope wrong in
    // a DIRECTION Schwarz is not; it does NOT catch one merely wrong in
    // MAGNITUDE (the 100x mutant scores 0.9991) — see the doc comment.
    let ratio = qqr_vs_schwarz[1] / qqr_vs_dense[1];
    assert!(
        (0.5..=2.0).contains(&ratio),
        "alkane_8 at thresh=1e-8: max|K_QQR - K_Schwarz| / max|K_QQR - K_dense| = {ratio:.4} \
         ({:.4e} / {:.4e}), outside [0.5, 2.0]. Measured 0.998 when calibrated. These two are \
         the same quantity for a VALID bound — QQR's difference from the Schwarz arm is entirely \
         its own legitimate truncation against the exact K. A ratio far from 1 means QQR errs \
         where Schwarz does not: the two arms are diverging for a reason other than QQR simply \
         truncating a slightly longer tail.",
        qqr_vs_schwarz[1],
        qqr_vs_dense[1]
    );
}
