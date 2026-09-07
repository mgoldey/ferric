//! Exactness anchors for the COSX seminumerical exchange builder
//! (`ferric_scf::cosx_k::CosxK`). Written BEFORE the builder compiled, per
//! the repo rule (EXACTNESS ANCHOR FIRST).
//!
//! A quadrature scheme has no vacuous parameter that makes it algebraically
//! exact, so the trivial limit is the GRID limit: `K_COSX -> K_analytic` as
//! the Becke-Lebedev grid is refined. Anchor (a) therefore asserts
//! CONVERGENCE (monotone fall of `max|dK|`, below 1e-6 at (99,302)), not a
//! single equality. The other three anchors are algebraic identities of the
//! implementation and hold to round-off.
//!
//! Artifact hypotheses (what a broken builder looks like), from
//! `scripts/cosx_proto.py`:
//!   * transpose/index error in the G contraction: `max|dK|` PLATEAUS at
//!     O(1e-2..1) under refinement;
//!   * wrong sign on A: `max|dK| ~ 2*||K||_max`, does not fall;
//!   * missing `0.5*(Kt + Kt^T)`: `max|dK|` stalls at the level of the
//!     Ktilde asymmetry;
//!   * missing `sqrt(w)`: nonsense magnitude, does not fall.
//! Each of these was applied by hand to the builder while writing it and the
//! observed failure of anchor (a) is recorded in the commit message / report.

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::cosx_k::{CosxBackend, CosxConfig, CosxK};
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
    mol: Molecule,
    prep: PreparedBasis,
    d: Array2<f64>,
    c_occ: Array2<f64>,
    k_direct: Array2<f64>,
}

/// Converged RHF on water/cc-pVDZ + the analytic (direct, dense) K of that density.
fn setup() -> Setup {
    let mol = Molecule::parse_xyz(WATER, 0, 1).expect("water");
    let bs = bundled("cc-pvdz").expect("cc-pvdz");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let ctx = ParallelContext::default();
    let cfg = RhfConfig {
        energy_conv: 1e-12,
        density_conv: 1e-10,
        integral_thresh: 1e-14,
        ..Default::default()
    };
    let res = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).expect("rhf");
    assert!(res.converged, "reference RHF did not converge");
    let nocc = (mol.nelec() / 2) as usize;
    let c_occ = res.mos_r().slice(ndarray::s![.., ..nocc]).to_owned();
    let d = res.density_total.clone();
    let n = prep.nbasis();
    let mut j = Array2::zeros((n, n));
    let mut k_direct = Array2::zeros((n, n));
    build_jk(&ctx, &prep, &bounds, 1e-14, &d, &mut j, &mut k_direct).expect("direct jk");
    Setup { mol, prep, d, c_occ, k_direct }
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).mapv(f64::abs).fold(0.0_f64, |m, &v| m.max(v))
}

fn grid(n_radial: usize, n_angular: usize) -> AtomicGridConfig {
    AtomicGridConfig { n_radial, n_angular, ..Default::default() }
}

fn cosx_config(g: AtomicGridConfig, overlap_fit: bool, screen_thresh: Option<f64>) -> CosxConfig {
    CosxConfig { grid: g, overlap_fit, screen_thresh, ..CosxConfig::default() }
}

fn build_k(s: &Setup, ctx: &ParallelContext, cfg: CosxConfig, d: &Array2<f64>) -> Array2<f64> {
    let n = s.prep.nbasis();
    let mut kb = CosxK::new(ctx, &s.mol, &s.prep, cfg, usize::MAX).expect("CosxK::new");
    let mut k = Array2::zeros((n, n));
    kb.build(d, &mut k).expect("cosx build");
    k
}

/// Anchor (a): the grid limit. Both the plain and the overlap-fitted K must
/// converge MONOTONICALLY to the analytic K and be below 1e-6 at (99,302).
/// The fit's own trivial limit is `Q -> I` on a dense grid, so both series
/// must land on the SAME matrix.
#[test]
fn cosx_matches_direct_k_in_the_dense_grid_limit() {
    let s = setup();
    let ctx = ParallelContext::default();
    // Lebedev orders available in ferric_quadrature: 6,14,26,50,110,302.
    let grids = [(25usize, 50usize), (50, 110), (99, 302)];

    let k_norm = s.k_direct.mapv(f64::abs).fold(0.0_f64, |m, &v| m.max(v));
    assert!(k_norm > 0.1, "reference K is ~0 ({k_norm:.3e}); the anchor would pass vacuously");

    let mut plain = Vec::new();
    let mut fitted = Vec::new();
    let mut last_plain = None;
    let mut last_fit = None;
    for &(nr, na) in &grids {
        let kp = build_k(&s, &ctx, cosx_config(grid(nr, na), false, None), &s.d);
        let kf = build_k(&s, &ctx, cosx_config(grid(nr, na), true, None), &s.d);
        let dp = max_abs_diff(&kp, &s.k_direct);
        let df = max_abs_diff(&kf, &s.k_direct);
        println!("grid ({nr:3},{na:3}): max|dK| plain {dp:.6e}  fitted {df:.6e}");
        // K of a symmetric D is symmetric; the 0.5(Kt + Kt^T) step makes this
        // exact, so a dropped symmetrization is caught here even on a fine
        // grid where the raw Ktilde asymmetry might already be < 1e-6.
        let asym_p = max_abs_diff(&kp, &kp.t().to_owned());
        let asym_f = max_abs_diff(&kf, &kf.t().to_owned());
        assert!(asym_p <= 1e-15, "plain COSX K not symmetric at ({nr},{na}): {asym_p:.3e}");
        assert!(asym_f <= 1e-15, "fitted COSX K not symmetric at ({nr},{na}): {asym_f:.3e}");
        plain.push(dp);
        fitted.push(df);
        last_plain = Some(kp);
        last_fit = Some(kf);
    }
    for w in plain.windows(2) {
        assert!(w[1] < w[0], "plain COSX K error did not fall monotonically: {plain:?}");
    }
    for w in fitted.windows(2) {
        assert!(w[1] < w[0], "fitted COSX K error did not fall monotonically: {fitted:?}");
    }
    let fp = *plain.last().unwrap();
    let ff = *fitted.last().unwrap();
    assert!(fp < 1e-6, "plain COSX K at (99,302): max|dK| = {fp:.3e} >= 1e-6");
    assert!(ff < 1e-6, "fitted COSX K at (99,302): max|dK| = {ff:.3e} >= 1e-6");
    // Fit trivial limit: Q -> I on the dense grid, both converge to the same K.
    let dpf = max_abs_diff(last_plain.as_ref().unwrap(), last_fit.as_ref().unwrap());
    println!("(99,302): max|K_fit - K_plain| = {dpf:.3e}");
    assert!(dpf < 1e-6, "fitted and plain COSX K disagree at (99,302): {dpf:.3e}");
    // Reachability: the fit must actually DO something on the coarse grid (it is
    // not an identity in disguise). Measured in the prototype: fitted beats plain.
    assert!(
        (plain[0] - fitted[0]).abs() > 1e-9,
        "fit changed nothing at (25,50) — Q is the identity, the fit is not wired"
    );
}

/// Anchor (e): the two A-build backends are the same K. Same config, same
/// density, only `backend` differs: the batched md3c1e fold (default) vs the
/// per-point libint2 `A^g` + GEMV (cosx_a). Both fit and no-fit, and both the
/// density and the `C_occ` entry points, so the fold is checked on every path
/// the SCF uses. Mutation proofs (applied by hand to `accumulate_pair`, recorded
/// in the S3 swap report): negating the kernel block (the sign trap) and
/// dropping the `s1 != s2` mirror each turn `max|dK|` from ~1e-14 to O(1).
#[test]
fn cosx_k_md3c1e_matches_cosx_a_backend() {
    let s = setup();
    let ctx = ParallelContext::default();
    let n = s.prep.nbasis();
    assert_eq!(CosxConfig::default().backend, CosxBackend::Md3c1e, "md3c1e must be the default backend");
    for fit in [true, false] {
        let mut cfg_md = cosx_config(grid(25, 50), fit, None);
        cfg_md.backend = CosxBackend::Md3c1e;
        let mut cfg_a = cfg_md.clone();
        cfg_a.backend = CosxBackend::CosxA;

        let k_md = build_k(&s, &ctx, cfg_md.clone(), &s.d);
        let k_a = build_k(&s, &ctx, cfg_a.clone(), &s.d);
        let scale = k_a.mapv(f64::abs).fold(0.0_f64, |m, &v| m.max(v));
        assert!(scale > 0.05, "K ~ 0 ({scale:.3e}); vacuous");
        let dev = max_abs_diff(&k_md, &k_a);
        println!("backend anchor (fit={fit}, build): max|K_md3c1e - K_cosx_a| = {dev:.3e} (||K||max {scale:.3e})");
        assert!(dev <= 1e-12, "md3c1e and cosx_a backends disagree (fit={fit}): {dev:.3e}");

        // Same through the C_occ half transform.
        let mut kb_md = CosxK::new(&ctx, &s.mol, &s.prep, cfg_md, usize::MAX).expect("md");
        let mut kb_a = CosxK::new(&ctx, &s.mol, &s.prep, cfg_a, usize::MAX).expect("cosx_a");
        let mut kc_md = Array2::zeros((n, n));
        let mut kc_a = Array2::zeros((n, n));
        kb_md.build_from_occ(&s.c_occ, &mut kc_md).expect("md occ");
        kb_a.build_from_occ(&s.c_occ, &mut kc_a).expect("cosx_a occ");
        let dev_c = max_abs_diff(&kc_md, &kc_a);
        println!("backend anchor (fit={fit}, build_from_occ): max|dK| = {dev_c:.3e}");
        assert!(dev_c <= 1e-12, "backends disagree on build_from_occ (fit={fit}): {dev_c:.3e}");
        // Both carry the same (grid) error against the analytic K — the backend
        // swap changed the kernel, not the quadrature.
        let e_md = max_abs_diff(&k_md, &s.k_direct);
        let e_a = max_abs_diff(&k_a, &s.k_direct);
        assert!((e_md - e_a).abs() <= 1e-12, "grid errors differ: md {e_md:.3e} vs cosx_a {e_a:.3e}");
    }
}

/// Anchor (b): the screen's trivial limit. A builder carrying pair bounds at
/// threshold 0 must reproduce the unscreened builder (`screen_thresh: None`,
/// no bounds built at all) bit-for-bit — the same blocks in the same order.
#[test]
fn cosx_screen_zero_threshold_matches_unscreened() {
    let s = setup();
    let ctx = ParallelContext::default();
    let g = grid(25, 50);
    let k_un = build_k(&s, &ctx, cosx_config(g.clone(), true, None), &s.d);
    let n = s.prep.nbasis();
    let mut kb = CosxK::new(&ctx, &s.mol, &s.prep, cosx_config(g, true, Some(0.0)), usize::MAX).expect("t=0");
    let mut k_z = Array2::zeros((n, n));
    kb.build(&s.d, &mut k_z).expect("t=0 build");
    let dev = max_abs_diff(&k_un, &k_z);
    assert!(dev <= 1e-14, "screen at threshold 0 differs from unscreened by {dev:.3e}");
    // The counters must say the same thing (a silent floor on `t` that drops
    // nothing on water — e.g. 1e-9 — is invisible to `dev`; one at 1e-7 is not,
    // and the counters catch any dropped pair at all).
    let t = *kb.last_timings();
    assert_eq!(t.pairs_kept, t.pairs_total, "t = 0 dropped {} pairs", t.pairs_total - t.pairs_kept);
    assert_eq!(t.pairs_kept_geom, t.pairs_total, "t = 0: geometry-only counter is not the total");
    // Sharp detector for a silent floor on `t`: with D scaled by 1e-8, F is
    // ~1e-8 and ANY positive threshold drops most pairs (water at 1e-7 or even
    // 1e-9 drops nothing at full scale on this grid — measured, mutation M1).
    // At true t = 0 the keep rule is `est * fmax >= 0`, always true.
    let d_small = &s.d * 1e-8;
    let k_un_small = build_k(&s, &ctx, cosx_config(grid(25, 50), true, None), &d_small);
    let mut k_z_small = Array2::zeros((n, n));
    kb.build(&d_small, &mut k_z_small).expect("t=0 build, scaled D");
    let t = *kb.last_timings();
    assert_eq!(t.pairs_kept, t.pairs_total, "t = 0 on 1e-8 D dropped {} pairs (a floor on t)", t.pairs_total - t.pairs_kept);
    assert!(k_un_small == k_z_small, "t = 0 on 1e-8 D is not bitwise the unscreened K");
}

/// Positive screen anchor, replacing the `cosx_a_screen_is_unsound_tripwire`
/// that pinned the OLD signed-overlap bound's defect (15/21 same-centre pairs
/// dropped with `|A^g|` up to 0.233; `max|K_scr - K_unscr| = 0.71` on
/// water/cc-pVDZ at t = 1e-7 against a grid error of ~5e-5). The tripwire was
/// designed to FAIL once the bound was replaced; it did (2026-09-07: "0 of 21
/// same-centre shell pairs are DROPPED"), and this is what took its place.
///
/// The screened K at the production threshold must sit far below the grid
/// error of the unscreened K: `max|K_scr - K_unscr| < 1e-6` on the (50,110)
/// operating grid with the fit on (the SCF path). Reachability: the screen
/// must have dropped SOMETHING (kept < total), or the anchor is vacuous.
#[test]
fn cosx_screened_k_matches_unscreened_below_grid_error() {
    let s = setup();
    let r = screened_vs_unscreened(&s, "water/cc-pVDZ");
    assert!(r.kept < r.total, "screen dropped nothing — the anchor would pass vacuously");
    assert!(r.dev < 1e-6, "screened K differs from unscreened by {:.3e} (grid error {:.3e})", r.dev, r.grid_err);
    assert!(r.dev < 0.1 * r.grid_err, "screen error {:.3e} is not well below the grid error {:.3e}", r.dev, r.grid_err);
}

struct ScreenResult {
    dev: f64,
    grid_err: f64,
    kept: usize,
    kept_geom: usize,
    total: usize,
}

/// Unscreened vs screened-at-default K on the (50,110)+fit operating grid.
fn screened_vs_unscreened(s: &Setup, label: &str) -> ScreenResult {
    let ctx = ParallelContext::default();
    let n = s.prep.nbasis();
    let g = grid(50, 110);
    let thresh = CosxConfig::default().screen_thresh.expect("density-driven screen is on by default");
    let k_un = build_k(s, &ctx, cosx_config(g.clone(), true, None), &s.d);
    let mut kb = CosxK::new(&ctx, &s.mol, &s.prep, cosx_config(g, true, Some(thresh)), usize::MAX).expect("screened");
    let mut k_sc = Array2::zeros((n, n));
    kb.build(&s.d, &mut k_sc).expect("screened build");
    let t = *kb.last_timings();
    let dev = max_abs_diff(&k_un, &k_sc);
    let grid_err = max_abs_diff(&k_un, &s.k_direct);
    println!(
        "screen anchor {label} (50,110)+fit t={thresh:e}: max|K_scr - K_unscr| = {dev:.3e} (grid error {grid_err:.3e}); \
         pairs kept {}/{} ({:.4}), geometry-only would keep {:.4}",
        t.pairs_kept,
        t.pairs_total,
        t.pairs_kept as f64 / t.pairs_total as f64,
        t.pairs_kept_geom as f64 / t.pairs_total as f64
    );
    ScreenResult { dev, grid_err, kept: t.pairs_kept, kept_geom: t.pairs_kept_geom, total: t.pairs_total }
}

/// Converged RHF on butane (testdata alkane_4) / def2-SVP + its direct K.
fn setup_butane() -> Setup {
    let path = format!("{}/../../testdata/molecules/alkane_4.xyz", env!("CARGO_MANIFEST_DIR"));
    let mol = Molecule::load_xyz(&path).expect("alkane_4.xyz");
    let bs = bundled("def2-svp").expect("def2-svp");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let ctx = ParallelContext::default();
    let cfg = RhfConfig { energy_conv: 1e-10, density_conv: 1e-8, integral_thresh: 1e-14, ..Default::default() };
    let res = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).expect("rhf");
    assert!(res.converged, "reference RHF did not converge");
    let nocc = (mol.nelec() / 2) as usize;
    let c_occ = res.mos_r().slice(ndarray::s![.., ..nocc]).to_owned();
    let d = res.density_total.clone();
    let n = prep.nbasis();
    let mut j = Array2::zeros((n, n));
    let mut k_direct = Array2::zeros((n, n));
    build_jk(&ctx, &prep, &bounds, 1e-14, &d, &mut j, &mut k_direct).expect("direct jk");
    Setup { mol, prep, d, c_occ, k_direct }
}

/// Anchor (b) on a molecule where the screen bites: butane/def2-SVP at the
/// default threshold. Measured 2026-09-07 (threshold sweep): max|dK| 7.6e-7
/// against a grid error of 2.9e-4; density-driven keeps 0.79 of the
/// (pair, batch) work where a geometry-only bound keeps 0.95 — the `fmax`
/// factor is what does the dropping. Reachability bars are set with margin
/// on those numbers (kept < 0.85; geometry-only > 0.90).
///
/// Mutation proof (applied by hand to `BatchScreen::keep`, recorded in the
/// commit message): using `fmax[s2]` alone instead of `max(fmax[s1], fmax[s2])`
/// drops blocks whose MIRROR contribution `G_{s2} += A F_{s1}` is significant.
#[test]
fn cosx_density_driven_screen_butane_def2svp() {
    let s = setup_butane();
    let r = screened_vs_unscreened(&s, "butane/def2-SVP");
    let kept = r.kept as f64 / r.total as f64;
    let geom = r.kept_geom as f64 / r.total as f64;
    assert!(r.dev < 1e-6, "screened K differs from unscreened by {:.3e} (grid error {:.3e})", r.dev, r.grid_err);
    assert!(r.dev < 0.1 * r.grid_err, "screen error {:.3e} is not well below the grid error {:.3e}", r.dev, r.grid_err);
    assert!(kept < 0.85, "density-driven screen kept {kept:.4} of the work — expected < 0.85 (measured 0.79)");
    assert!(geom > 0.90, "geometry-only bound kept only {geom:.4} — expected > 0.90 (measured 0.95); the bound changed");
    assert!(r.kept < r.kept_geom, "density-driven screen must drop strictly more than the geometry-only bound");
}

/// Anchor (c): `build_from_occ(C)` and `build(C C^T)` are the same K.
#[test]
fn cosx_build_from_occ_matches_build_from_density() {
    let s = setup();
    let ctx = ParallelContext::default();
    let n = s.prep.nbasis();
    let d1 = s.c_occ.dot(&s.c_occ.t());
    let mut kb = CosxK::new(&ctx, &s.mol, &s.prep, cosx_config(grid(25, 50), true, None), usize::MAX)
        .expect("CosxK::new");
    let mut k_d = Array2::zeros((n, n));
    let mut k_c = Array2::zeros((n, n));
    kb.build(&d1, &mut k_d).expect("build");
    kb.build_from_occ(&s.c_occ, &mut k_c).expect("build_from_occ");
    let dev = max_abs_diff(&k_d, &k_c);
    let scale = k_d.mapv(f64::abs).fold(0.0_f64, |m, &v| m.max(v));
    assert!(scale > 0.05, "K(C C^T) ~ 0 ({scale:.3e}); vacuous");
    assert!(dev < 1e-12, "build_from_occ vs build differ by {dev:.3e}");
}

/// Anchor (d): no leaked state between densities. K(D1) then K(D2) from ONE
/// instance must equal K from a fresh instance bit-for-bit (this is the
/// UHF alpha/beta trap even though only RHF is wired).
#[test]
fn cosx_two_densities_no_leaked_state() {
    let s = setup();
    let ctx = ParallelContext::default();
    let n = s.prep.nbasis();
    let cfg = cosx_config(grid(25, 50), true, None);
    // D2: a genuinely different (still symmetric) matrix — the C C^T of the
    // occupied block, i.e. D1/2 mixed with a rank-1 perturbation.
    let d1 = s.d.clone();
    let mut d2 = s.c_occ.dot(&s.c_occ.t());
    for i in 0..n {
        for j in 0..n {
            d2[(i, j)] += 0.01 * (((i + 1) * (j + 1)) as f64).sqrt() / n as f64;
        }
    }
    assert!(max_abs_diff(&d1, &d2) > 0.1, "D1 and D2 are not different enough to test leakage");

    let mut shared = CosxK::new(&ctx, &s.mol, &s.prep, cfg.clone(), usize::MAX).expect("shared");
    let mut k1_shared = Array2::zeros((n, n));
    let mut k2_shared = Array2::zeros((n, n));
    shared.build(&d1, &mut k1_shared).expect("shared D1");
    shared.update_density(&d2);
    shared.build(&d2, &mut k2_shared).expect("shared D2");
    // And back to D1 after D2 (order matters for a state leak).
    let mut k1_again = Array2::zeros((n, n));
    shared.build(&d1, &mut k1_again).expect("shared D1 again");

    let k1_fresh = build_k(&s, &ctx, cfg.clone(), &d1);
    let k2_fresh = build_k(&s, &ctx, cfg, &d2);

    assert!(k1_shared == k1_fresh, "K(D1) from shared instance != fresh (bitwise)");
    assert!(k2_shared == k2_fresh, "K(D2) after D1 from shared instance != fresh (bitwise)");
    assert!(k1_again == k1_fresh, "K(D1) after D2 from shared instance != fresh (bitwise)");
    assert!(max_abs_diff(&k1_fresh, &k2_fresh) > 1e-3, "K(D1) == K(D2): the two densities are not distinguishing");
}
