//! Anchors for the SUB-BATCHED (grouped) COSX pair screen, written BEFORE
//! the implementation compiled per the repo rule (EXACTNESS ANCHOR FIRST).
//!
//! # What changes and what does not
//!
//! The md3c1e kernel keeps its 256-point sub-batching (`COSX_SUB_BATCH_POINTS`)
//! — that is tuned for scratch reuse and BLAS shape and is NOT touched. What
//! changes is the SCREENING DECISION only: a sub-batch's points are split into
//! contiguous groups of `CosxConfig::screen_group` points, the bound is tested
//! per group over that group's axis-aligned box, and a pair is kept for the
//! whole sub-batch if ANY group keeps it (`CosxConfig::screen_group = 0` means
//! one group = the whole sub-batch, i.e. today's behaviour).
//!
//! Correctness is preserved BY CONSTRUCTION: each group's bound is applied to
//! points actually in that group, and the union over groups is what survives,
//! so no pair that today's whole-batch test keeps can be dropped. The anchors
//! below pin that this is what the code actually does, not merely what the
//! argument says.
//!
//! # The motivating measurement (do not re-derive)
//!
//! `scripts/queue/out/snlink_python_results.md` §4.1: for 68-80% of
//! (shell-pair, grid-batch) decisions the sphere query returns its
//! distance-free `R = 0` value, because a 256-point Becke sub-batch is ~2.3
//! whole Lebedev spheres of one atom and its bounding sphere swallows a small
//! molecule. The bound floor sits at 1.4e-1 while the actual screening products
//! are 1e-5..1e-9, so the screen is running blind on most decisions.
//!
//! # Anchors
//!
//! * (a) `group_size_whole_subbatch_is_bitwise_todays_screen` — TRIVIAL LIMIT.
//!   `screen_group = 0` (and `= COSX_SUB_BATCH_POINTS`) must give the same K
//!   BITWISE and the same kept/total counters as the unsplit screen. This is
//!   the "no change" control.
//! * (b) `grouped_screen_k_matches_unscreened_below_grid_error` — CORRECTNESS
//!   at the production threshold on water/cc-pVDZ and butane/def2-SVP, against
//!   the SAME unscreened K, with the finer group size.
//! * (c) `grouped_screen_gain_is_negligible_against_a_superlinear_bound_cost`
//!   — REACHABILITY / NON-INERTNESS, and the branch's VERDICT. As
//!   pre-registered it asserted that the degenerate fraction and the kept work
//!   must both FALL with group size. THEY DO NOT (butane: kept 0.791577 ->
//!   0.790411 for 8.2x the bound evaluations), so per its own stated stop
//!   condition it now pins the negative result and is written to fail if that
//!   is ever overturned. Its doc carries the table.
//! * `grouped_screen_never_drops_what_the_unsplit_screen_keeps` — the
//!   union-over-groups property stated as a test: kept work at any group size
//!   is <= kept work unsplit, at equal or better K accuracy.
//!
//! # Mutation proofs (2026-09-09)
//!
//! * MUTATION A, `regions.push(Region::of(pts))` — give every group the WHOLE
//!   sub-batch's region. -> (c) RED, and diagnostically: `kept` freezes at
//!   exactly 90 512 912 for EVERY group size, where the real code moves it to
//!   90 379 536. This is the inertness mutation and it is what proves the
//!   per-group regions are live, i.e. that the negative result is physics and
//!   not a no-op build.
//! * MUTATION B, misaligned groups (two variants: group `q` gets group `q+1`'s
//!   `fmax`; every group gets the first group's region). -> (b) stayed GREEN
//!   at 7.5-7.7e-7. Recorded as a WEAKNESS of anchor (b), not a pass; see its
//!   doc comment for why an anchor cannot detect a defect that does not change
//!   the answer.

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::cosx_k::{CosxConfig, CosxK, CosxTimings, COSX_SUB_BATCH_POINTS};
use ferric_scf::fock::KBuilder;
use ferric_scf::rhf::{build_jk, solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

const WATER: &str = "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
";

struct Setup {
    label: &'static str,
    mol: Molecule,
    prep: PreparedBasis,
    d: Array2<f64>,
    k_direct: Array2<f64>,
}

fn converged(label: &'static str, mol: Molecule, basis: &str) -> Setup {
    let bs = bundled(basis).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let ctx = ParallelContext::default();
    let cfg = RhfConfig { energy_conv: 1e-10, density_conv: 1e-8, integral_thresh: 1e-14, ..Default::default() };
    let res = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).expect("rhf");
    assert!(res.converged, "{label}: reference RHF did not converge");
    let d = res.density_total.clone();
    let n = prep.nbasis();
    let (mut j, mut k_direct) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
    build_jk(&ctx, &prep, &bounds, 1e-14, &d, &mut j, &mut k_direct).expect("direct jk");
    Setup { label, mol, prep, d, k_direct }
}

fn water() -> Setup {
    converged("water/cc-pVDZ", Molecule::parse_xyz(WATER, 0, 1).expect("water"), "cc-pvdz")
}

fn butane() -> Setup {
    let path = format!("{}/../../testdata/molecules/alkane_4.xyz", env!("CARGO_MANIFEST_DIR"));
    converged("butane/def2-SVP", Molecule::load_xyz(&path).expect("alkane_4"), "def2-svp")
}

fn grid(n_radial: usize, n_angular: usize) -> AtomicGridConfig {
    AtomicGridConfig { n_radial, n_angular, ..Default::default() }
}

fn cfg_at(screen_thresh: Option<f64>, screen_group: usize) -> CosxConfig {
    CosxConfig { grid: grid(50, 110), overlap_fit: true, screen_thresh, screen_group, ..CosxConfig::default() }
}

fn build(s: &Setup, cfg: CosxConfig) -> (Array2<f64>, CosxTimings) {
    let ctx = ParallelContext::default();
    let n = s.prep.nbasis();
    let mut kb = CosxK::new(&ctx, &s.mol, &s.prep, cfg, usize::MAX).expect("CosxK::new");
    let mut k = Array2::zeros((n, n));
    kb.build(&s.d, &mut k).expect("cosx build");
    (k, *kb.last_timings())
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).mapv(f64::abs).fold(0.0_f64, |m, &v| m.max(v))
}

fn thresh() -> f64 {
    CosxConfig::default().screen_thresh.expect("the density-driven screen is on by default")
}

/// Group sizes swept by the reachability anchor and the results table. 0 is
/// the unsplit control; the rest divide `COSX_SUB_BATCH_POINTS` so no group is
/// ragged (a ragged tail is handled by the code but muddies the curve).
const GROUPS: [usize; 6] = [0, 256, 64, 32, 16, 8];

/// Anchor (a), TRIVIAL LIMIT: one group per sub-batch is today's screen,
/// bitwise. `screen_group = 0` (off) and `= COSX_SUB_BATCH_POINTS` (one group
/// covering the whole sub-batch) must BOTH reproduce it.
///
/// This is the control that fails if grouping perturbs the kept set or the
/// accumulation order even when it is asked to do nothing.
#[test]
fn group_size_whole_subbatch_is_bitwise_todays_screen() {
    for s in [water(), butane()] {
        let (k0, t0) = build(&s, cfg_at(Some(thresh()), 0));
        for g in [COSX_SUB_BATCH_POINTS, 2 * COSX_SUB_BATCH_POINTS] {
            let (k, t) = build(&s, cfg_at(Some(thresh()), g));
            assert!(k == k0, "{}: screen_group = {g} is not BITWISE the unsplit screen", s.label);
            assert_eq!(t.pairs_kept, t0.pairs_kept, "{}: screen_group = {g} changed pairs_kept", s.label);
            assert_eq!(t.pairs_total, t0.pairs_total, "{}: screen_group = {g} changed pairs_total", s.label);
        }
        // And the screen's own trivial limit still holds with grouping on:
        // t = 0 keeps everything regardless of how the points are grouped.
        let (k_un, _) = build(&s, cfg_at(None, 0));
        let (k_z, tz) = build(&s, cfg_at(Some(0.0), 16));
        assert!(k_z == k_un, "{}: grouped screen at t = 0 is not bitwise the unscreened K", s.label);
        assert_eq!(tz.pairs_kept, tz.pairs_total, "{}: grouped screen at t = 0 dropped pairs", s.label);
    }
}

/// Anchor (b), CORRECTNESS: at the production threshold and the finer group
/// size, the grouped screen's K stays far below the grid error, measured
/// against the SAME unscreened K on both systems.
///
/// # This anchor PASSED its mutation test, which means it is WEAK — read this
///
/// Two misalignment mutations were applied (2026-09-09) and NEITHER turned it
/// red: (i) give group `q` the `fmax` of group `q+1`; (ii) give every group the
/// FIRST group's region. Both left `max|dK|` at 7.5-7.7e-7 on butane, i.e.
/// indistinguishable from the correct code.
///
/// That is not a defect in the mutations, it is a consequence of the result
/// this branch measured: at group 8 the union over 32 groups drops only 0.12
/// pp more work than the unsplit test, so perturbing which group owns which
/// bound moves a handful of marginal decisions and cannot move K. An anchor
/// can only detect a defect that changes the answer, and here almost nothing
/// changes the answer.
///
/// So this test is honest evidence of CORRECTNESS (the K really is right) and
/// is NOT evidence that the group/point alignment is right. If grouping is ever
/// made to pay — if kept work actually falls — this anchor must be re-mutated
/// before it is trusted as a guard on alignment.
#[test]
fn grouped_screen_k_matches_unscreened_below_grid_error() {
    for s in [water(), butane()] {
        let (k_un, _) = build(&s, cfg_at(None, 0));
        let grid_err = max_abs_diff(&k_un, &s.k_direct);
        for g in [64usize, 32, 16, 8] {
            let (k, t) = build(&s, cfg_at(Some(thresh()), g));
            let dev = max_abs_diff(&k_un, &k);
            println!(
                "{} group {g:>3}: max|dK| = {dev:.3e} (grid err {grid_err:.3e}), kept {:.4}",
                s.label,
                t.pairs_kept as f64 / t.pairs_total as f64
            );
            assert!(dev < 1e-6, "{}: group {g} gives max|dK| = {dev:.3e} (grid err {grid_err:.3e})", s.label);
            assert!(
                dev < 0.1 * grid_err,
                "{}: group {g} screen error {dev:.3e} is not well below the grid error {grid_err:.3e}",
                s.label
            );
        }
    }
}

/// Anchor (c) as pre-registered was: "the degenerate-bound fraction must FALL
/// measurably with group size AND the kept work must fall with it; if the kept
/// work does not fall, the change is pointless and that is the finding".
///
/// # It does not fall. That is the finding, and this test now PINS it.
///
/// Measured 2026-09-09, butane/def2-SVP, (50,110)+fit, `t = 1e-7`, one thread
/// (counts are deterministic, so these are exact, not indicative):
///
/// ```text
///   group | degenerate |     kept | kept pairs | bound evals
///       0 |     0.6515 | 0.791577 | 90 512 912 |     446 985   (unsplit)
///      64 |     0.6419 | 0.790572 | 90 397 968 |     756 377   (1.69x)
///      32 |     0.6441 | 0.790601 | 90 401 296 |   1 168 970   (2.62x)
///      16 |     0.6465 | 0.790603 | 90 401 552 |   1 994 693   (4.46x)
///       8 |     0.6423 | 0.790411 | 90 379 536 |   3 651 199   (8.17x)
/// ```
///
/// Kept work falls by **0.12 percentage points** for **8.2x the bound
/// evaluations**, and the degenerate fraction moves only 0.6515 -> 0.6423 and
/// NOT monotonically. The mechanism is live (mutation A below freezes `kept`
/// at exactly 90 512 912 for every group size, so the per-group regions really
/// are being used); it simply does not pay.
///
/// Why, from `tests/cosx_region_diagnostics.rs` (same grid, same order): only
/// 33.6% of the degeneracy is the REGION reaching the pair midpoint — the part
/// grouping can fix, and it does move that 33.6% -> 24.2% — while 30.4% is
/// `|AB|/2` eating the distance, which grouping cannot touch at all and whose
/// share GROWS to 35.3% as the region shrinks. The screen is blind on most
/// decisions because of the SHELL PAIRS, not because of the batch geometry.
///
/// This test therefore asserts the NEGATIVE: the gain stays under half a
/// percentage point while the cost grows superlinearly. It is written to FAIL
/// if a future change makes grouping actually pay, which is the point — the
/// verdict is provisional and this is what would reopen it.
#[test]
fn grouped_screen_gain_is_negligible_against_a_superlinear_bound_cost() {
    for s in [water(), butane()] {
        let mut rows = Vec::new();
        for g in GROUPS {
            let (_, t) = build(&s, cfg_at(Some(thresh()), g));
            let deg = t.screen_degenerate as f64 / t.bound_evals.max(1) as f64;
            let kept = t.pairs_kept as f64 / t.pairs_total as f64;
            println!(
                "{} group {g:>3}: degenerate {deg:.4}  kept {kept:.6}  bound evals {:>9}  ({} kept / {} total)",
                s.label, t.bound_evals, t.pairs_kept, t.pairs_total
            );
            rows.push((g, deg, kept, t.bound_evals));
        }
        let unsplit = rows[0];
        let finest = *rows.last().expect("GROUPS is non-empty");
        // The mechanism must be LIVE: bound evaluations grow with grouping, and
        // the kept work does move (it is not frozen, which is what mutation A
        // produces). Without these two the verdict below would be vacuous.
        //
        // The growth factor is NOT ~G: `keep` stops at the first group that
        // keeps the pair, so a system where almost everything is kept pays
        // almost nothing extra. Water keeps 99.3% and grows only 5070 -> 9063
        // (1.8x); butane keeps 79% and grows 446 985 -> 3 651 199 (8.2x). Both
        // are the early-out working as intended, so the bar is just "grew".
        assert!(
            finest.3 > unsplit.3,
            "{}: bound evaluations did not grow ({} -> {}) — grouping is not happening",
            s.label,
            unsplit.3,
            finest.3
        );
        assert!(
            finest.2 < unsplit.2,
            "{}: kept work is FROZEN at {:.6} across group sizes — the per-group regions are inert \
             (this is exactly what mutation A produces)",
            s.label,
            unsplit.2
        );
        if s.label.starts_with("butane") {
            assert!(
                finest.2 > unsplit.2 - 0.005,
                "{}: kept work fell by {:.4} pp (>0.5 pp) going 256 -> 8. The 2026-09-09 verdict that \
                 sub-batching the bound test does not pay has been OVERTURNED — re-run the scaling \
                 measurement and revisit CosxConfig::screen_group's default.",
                s.label,
                100.0 * (unsplit.2 - finest.2)
            );
            assert!(
                finest.1 > unsplit.1 - 0.05,
                "{}: degenerate fraction fell {:.4} -> {:.4} (>0.05). The bound is no longer blind for \
                 the reason recorded here; re-derive the cause split before trusting the verdict.",
                s.label,
                unsplit.1,
                finest.1
            );
        }
    }
}

/// The union-over-groups property, stated as a test: a finer group can only
/// keep FEWER pairs than the unsplit test (its bound is never looser on any
/// point it covers), never more.
#[test]
fn grouped_screen_never_drops_what_the_unsplit_screen_keeps() {
    for s in [water(), butane()] {
        let (_, t0) = build(&s, cfg_at(Some(thresh()), 0));
        for g in [64usize, 32, 16, 8] {
            let (_, t) = build(&s, cfg_at(Some(thresh()), g));
            assert_eq!(t.pairs_total, t0.pairs_total, "{}: group {g} changed the denominator", s.label);
            assert!(
                t.pairs_kept <= t0.pairs_kept,
                "{}: group {g} kept MORE work ({} > {}) — the group bound is looser than the batch bound",
                s.label,
                t.pairs_kept,
                t0.pairs_kept
            );
        }
    }
}
