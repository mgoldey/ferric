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
use ferric_scf::cosx_k::{CosxConfig, CosxK};
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
    CosxConfig { grid: g, overlap_fit, screen_thresh }
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

/// Anchor (b): the screen's trivial limit. A builder carrying pair bounds at
/// threshold 0 must reproduce the unscreened builder (`screen_thresh: None`,
/// no bounds built at all) bit-for-bit — the same blocks in the same order.
#[test]
fn cosx_screen_zero_threshold_matches_unscreened() {
    let s = setup();
    let ctx = ParallelContext::default();
    let g = grid(25, 50);
    let k_un = build_k(&s, &ctx, cosx_config(g.clone(), true, None), &s.d);
    let k_z = build_k(&s, &ctx, cosx_config(g.clone(), true, Some(0.0)), &s.d);
    let dev = max_abs_diff(&k_un, &k_z);
    assert!(dev <= 1e-14, "screen at threshold 0 differs from unscreened by {dev:.3e}");
    // A positive threshold is refused (see `CosxConfig::screen_thresh` and the
    // tripwire below) — never a silently wrong K.
    let err = CosxK::new(&ctx, &s.mol, &s.prep, cosx_config(g, false, Some(1e-7)), usize::MAX);
    assert!(err.is_err(), "screen_thresh > 0 must be refused while the cosx_a screen is unsound");
}

/// Tripwire pinning the DEFECT that motivates the `screen_thresh > 0` refusal:
/// the cosx_a screen's magnitude bound is `sqrt(max|S_block|)`, which is
/// exactly zero for same-centre shell pairs whose overlap vanishes by angular
/// symmetry (s–p, px–py, …), so those pairs are dropped at ANY positive
/// threshold although their `A^g` at an off-centre grid point is O(1).
/// Measured before the guard went in: water/cc-pVDZ, (50,110), t = 1e-7 gave
/// `max|K_scr - K_unscr| = 0.71` (grid error 4.8e-5).
///
/// This test FAILS when the screen is fixed (no zero-estimate same-centre
/// pairs remain) — that is the signal to lift the refusal in `CosxK::new`.
#[test]
fn cosx_a_screen_is_unsound_tripwire() {
    use ferric_integrals::cosx_a::{a_matrix_at_point, CosxScreen, PairBounds};
    let s = setup();
    let bounds = PairBounds::build(&s.prep).expect("pair bounds");
    let nsh = s.prep.nshells();
    let dims = s.prep.shell_dims();
    let offs = s.prep.shell_offsets();
    let centres = s.prep.shell_centers();
    // An off-centre probe (Bohr), ~0.7 Bohr from O: same-centre s-p pairs have
    // an O(0.1..1) potential integral here while their overlap is ~0.
    let probe = [0.37, -0.21, 0.55];
    let a_full = a_matrix_at_point(&s.prep, &probe, None, CosxScreen::none()).expect("A").a;
    let thresh = 1e-7; // the Stage 2 "production" screen
    let mut unsound = 0usize;
    let mut same_centre = 0usize;
    let mut worst = 0.0_f64;
    for s1 in 0..nsh {
        for s2 in 0..s1 {
            if centres[s1] != centres[s2] {
                continue;
            }
            same_centre += 1;
            let est = bounds.estimate(s1, s2, &probe);
            let blk = a_full.slice(ndarray::s![offs[s1]..offs[s1] + dims[s1], offs[s2]..offs[s2] + dims[s2]]);
            let a_max = blk.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            if est < thresh && a_max > 1e-3 {
                unsound += 1;
                worst = worst.max(a_max);
            }
        }
    }
    println!(
        "cosx_a screen tripwire: {unsound} of {same_centre} same-centre shell pairs are DROPPED at t={thresh:e} \
         while their |A^g| block is > 1e-3 (largest dropped |A^g| = {worst:.3e})"
    );
    assert!(
        unsound > 0,
        "the cosx_a screen no longer drops O(1) same-centre pairs — lift the screen_thresh > 0 refusal in \
         CosxK::new and re-measure max|K_scr - K_unscr| before allowing a default"
    );
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
