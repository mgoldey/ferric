//! A pure functional's DF gradient reads the SCF's PAIR-SCREENED RI-J tensor
//! (`ferric_scf::raw_handoff`), with pairs the screen dropped contributing
//! zero, as in the SCF energy. Derivation:
//! `book/src/14-derivatives/02-df-jk-gradient.md#df-jk-gradient-pair-screened-ri-j`.
//!
//! Anchors and bars
//!
//! 1. EXACTNESS. With `integral_thresh = 0` nothing is dropped, so the reused
//!    tensor is the whole one: reuse and recompute agree to 0 bits.
//! 2. IDENTITY. Let `D~` be the density with every dropped pair zeroed. Reuse
//!    returns `g_J(D~)` plus the unchanged rest, recompute returns `g_J(D)`
//!    plus the same rest, so `g_reuse - g_rec = g_J(D~) - g_J(D)` where both
//!    terms come from the public recomputing `df_jk_gradient`. Built from an
//!    independent route (a fresh pair screen from `ri_j_pair_screen_for_test`
//!    and the full tensor), so a wrong pair map or a dropped pair that keeps
//!    its pre-screen value breaks it. Bar: 1e-11 Ha/Bohr (summation order of
//!    ~1e4 terms of size <= 1e1, times 1e-16, with ample margin).
//! 3. BOUNDS.
//!    (a) `d_P`: `|d_P - d~_P| <= thresh * sum_{dropped} |w_mu,nu|`, with `w` the
//!    pair weights (Schwarz: a dropped pair has `Q_mn q_aux_max <= thresh`,
//!    `|(P|mn)| <= Q_mn Q_P`). Rigorous.
//!    (b) gradient: `g` is a quadratic form `G(D, D)` in the density, so
//!    `g_J(D~) - g_J(D)` is `O(r)` with `r = |D - D~|_F / |D|_F`; the
//!    scaling bar is `2 r max|g_J(D)|` times a safety factor of 4 (the
//!    bilinear form can be larger on the dropped part than on `D`).
//!    A scaling bar, not a theorem: the first-order Schwarz bound on `d`
//!    is (a).
//! 4. CONSISTENCY WITH THE ENERGY. Central finite difference of the SCF energy
//!    run with the same screening. Report both routes' error; the screened
//!    route's must be no larger than the recompute route's plus the FD floor.
//!
//! Artifact hypothesis. If the implementation were broken by reading a stale
//! or unscreened tensor, reuse would equal recompute (anchor 2 fails: the
//! difference is nonzero by construction when pairs are dropped, so a test that
//! asserts it is nonzero and equals the identity cannot pass vacuously); by
//! zeroing the wrong pairs it would miss the identity by O(r).

use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::df_gradient::{df_jk_gradient, dfg_3c_blocks_computed_for_test};
use ferric_scf::df_j::ri_j_pair_screen_for_test;
use ferric_scf::gradient::restricted_scf_gradient;
use ferric_scf::raw_handoff::with_fresh_handoff;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

const AUX: &str = "def2-universal-jkfit";
const BASIS: &str = "def2-svp";

fn config(thresh: f64) -> RhfConfig {
    {
        let mut c = RhfConfig::default().with_xc("PBE").with_df_j_aux(AUX);
        c.df_k_aux = None;
        c.integral_thresh = thresh;
        c.energy_conv = 1e-11;
        c.density_conv = 1e-9;
        c
    }
    .for_geometry_task()
}

struct Setup {
    mol: Molecule,
    bs: ferric_core::basis::BasisSet,
    prep: PreparedBasis,
    bounds: SchwarzBounds,
}

fn setup(mol: Molecule) -> Setup {
    let bs = ferric_core::basis::bundled(BASIS).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).expect("schwarz");
    Setup {
        mol,
        bs,
        prep,
        bounds,
    }
}

fn scf(s: &Setup, cfg: &RhfConfig) -> ferric_scf::result::ScfResult {
    let res = solve_rhf(
        &ParallelContext::default(),
        &s.mol,
        &s.prep,
        Operator::coulomb(),
        &s.bounds,
        cfg,
    )
    .expect("SCF");
    assert!(res.converged, "SCF did not converge");
    res
}

fn gradient(
    s: &Setup,
    cfg: &RhfConfig,
    res: &ferric_scf::result::ScfResult,
) -> (Array2<f64>, usize) {
    let before = dfg_3c_blocks_computed_for_test();
    let g = restricted_scf_gradient(
        &s.mol,
        &s.prep,
        &s.bs,
        Operator::coulomb(),
        &s.bounds,
        cfg,
        res,
    )
    .expect("gradient");
    (g, dfg_3c_blocks_computed_for_test() - before)
}

fn bits(g: &Array2<f64>) -> Vec<u64> {
    g.iter().map(|x| x.to_bits()).collect()
}

fn max_abs(g: &Array2<f64>) -> f64 {
    g.iter().fold(0.0, |m, x| m.max(x.abs()))
}

/// The smallest threshold of a fixed ladder at which the screen drops a pair
/// on `s` (water's pairs all survive the usual thresholds): a test needs the
/// screen to act, and fails loudly if no rung does.
fn first_dropping_thresh(s: &Setup) -> f64 {
    let aux = ferric_core::basis::bundled(AUX).expect("aux");
    let dfbs = PreparedBasis::new(&s.mol, &aux).expect("aux prep");
    [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
        .into_iter()
        .find(|&t| {
            ri_j_pair_screen_for_test(Operator::coulomb(), &s.prep, &dfbs, t)
                .expect("screen")
                .is_some()
        })
        .expect("no threshold on the ladder drops a pair")
}

fn butane() -> Molecule {
    Molecule::load_xyz("../../testdata/molecules/alkane_4.xyz").expect("butane")
}

fn water() -> Molecule {
    Molecule::load_xyz("../../testdata/molecules/water.xyz").expect("water.xyz")
}

/// SCF with a hand-off, then the gradient reusing it and the one recomputing.
/// Asserts the reuse happened (slot filled, no raw blocks computed).
fn reuse_and_recompute(s: &Setup, cfg0: &RhfConfig) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let cfg = with_fresh_handoff(cfg0);
    let res = scf(s, &cfg);
    let slot = cfg.raw_handoff.as_ref().expect("hand-off installed");
    assert!(
        slot.is_filled(),
        "the RI-J tensor was not kept (reachability)"
    );
    let (g_reuse, computed) = gradient(s, &cfg, &res);
    assert_eq!(computed, 0, "reuse computed raw blocks");
    let (g_rec, computed_rec) = gradient(s, &cfg, &res);
    assert!(computed_rec > 0, "recompute computed no raw blocks");
    (g_reuse, g_rec, res.density_total)
}

/// `d` with the pairs of `screen` kept only.
fn on_kept(
    d: &Array2<f64>,
    screen: &ferric_integrals::three_index_source::PairScreen,
) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros(d.dim());
    for &(m, n) in screen.cols() {
        let (m, n) = (m as usize, n as usize);
        out[(m, n)] = d[(m, n)];
        out[(n, m)] = d[(n, m)];
    }
    out
}

struct Anchors {
    g_reuse: Array2<f64>,
    g_rec: Array2<f64>,
    ident: f64,
    delta: f64,
    r: f64,
    g_j_max: f64,
    dropped: usize,
    total: usize,
}

fn anchors(s: &Setup, thresh: f64) -> Anchors {
    let cfg0 = config(thresh);
    let (g_reuse, g_rec, d) = reuse_and_recompute(s, &cfg0);
    let aux = ferric_core::basis::bundled(AUX).expect("aux");
    let dfbs = PreparedBasis::new(&s.mol, &aux).expect("aux prep");
    let op = Operator::coulomb();
    let screen = ri_j_pair_screen_for_test(op, &s.prep, &dfbs, thresh)
        .expect("screen")
        .expect("the screen dropped no pair: raise the threshold (reachability)");
    let d_kept = on_kept(&d, &screen);
    let budget = 1usize << 30;
    let n = s.mol.atoms.len();
    let gj = df_jk_gradient(n, &s.prep, &dfbs, op, Some(&d), &[], budget).expect("gJ(D)");
    let gjk = df_jk_gradient(n, &s.prep, &dfbs, op, Some(&d_kept), &[], budget).expect("gJ(D~)");
    let identity = &(&g_reuse - &g_rec) - &(&gjk - &gj);
    let drop = &d - &d_kept;
    let fro = |a: &Array2<f64>| a.iter().map(|x| x * x).sum::<f64>().sqrt();
    Anchors {
        ident: max_abs(&identity),
        delta: max_abs(&(&g_reuse - &g_rec)),
        r: fro(&drop) / fro(&d),
        g_j_max: max_abs(&gj),
        dropped: screen.total_pairs() - screen.ncols(),
        total: screen.total_pairs(),
        g_reuse,
        g_rec,
    }
}

#[test]
fn without_screening_reuse_is_bitwise_the_recompute() {
    let s = setup(butane());
    let (reuse, rec, _) = reuse_and_recompute(&s, &config(0.0));
    assert_eq!(bits(&reuse), bits(&rec));
    assert!(max_abs(&rec) > 1e-3, "vacuous: ~zero gradient");
}

fn check_identity_and_bound(s: &Setup, thresh: f64, label: &str) {
    let a = anchors(s, thresh);
    let bar_g = 4.0 * 2.0 * a.r * a.g_j_max;
    eprintln!(
        "[{label}] thresh {thresh:e}: dropped {}/{} pairs, r = {:.3e}, |g_reuse - g_rec| = {:.3e}, \
         identity residual = {:.3e} (bar 1e-11), scaling bar {:.3e} (ratio {:.3e}), max|gJ| = {:.3e}",
        a.dropped, a.total, a.r, a.delta, a.ident, bar_g, a.delta / bar_g, a.g_j_max
    );
    assert!(a.delta > 1e-10, "vacuous: screening changed nothing");
    assert!(a.ident < 1e-11, "identity residual {:.3e}", a.ident);
    assert!(
        a.delta < bar_g,
        "difference {:.3e} above bar {bar_g:.3e}",
        a.delta
    );
    assert!(max_abs(&a.g_reuse) > 1e-3 && max_abs(&a.g_rec) > 1e-3);
}

#[test]
fn screened_reuse_matches_the_density_masked_recompute_on_water() {
    let s = setup(water());
    let t = first_dropping_thresh(&s);
    check_identity_and_bound(&s, t, "water");
}

#[test]
fn screened_reuse_matches_the_density_masked_recompute_on_butane() {
    check_identity_and_bound(&setup(butane()), 1e-5, "butane");
}

/// Rigorous Schwarz bound on `d_P` (anchor 3a), from the full tensor.
#[test]
fn dropped_pairs_move_d_p_by_less_than_thresh_times_their_weight() {
    let s = setup(butane());
    let thresh = 1e-5;
    let aux = ferric_core::basis::bundled(AUX).expect("aux");
    let dfbs = PreparedBasis::new(&s.mol, &aux).expect("aux prep");
    let op = Operator::coulomb();
    let screen = ri_j_pair_screen_for_test(op, &s.prep, &dfbs, thresh)
        .expect("screen")
        .expect("nothing dropped");
    let res = scf(&s, &config(thresh));
    let d = &res.density_total;
    let n = s.prep.nbasis();
    let kept = on_kept(d, &screen);
    let eri = ferric_integrals::threeindex::eri3_block(op, &s.prep, &dfbs, 0, dfbs.nbasis())
        .expect("eri3");
    let drop = d - &kept;
    let mut worst: f64 = 0.0;
    for p in 0..dfbs.nbasis() {
        let dp: f64 = (0..n)
            .flat_map(|m| (0..n).map(move |nu| (m, nu)))
            .map(|(m, nu)| eri[(p, m, nu)] * drop[(m, nu)])
            .sum();
        worst = worst.max(dp.abs());
    }
    // sum over dropped function pairs of |w| (both triangles).
    let wsum: f64 = drop.iter().map(|x| x.abs()).sum();
    let bound = thresh * wsum;
    eprintln!(
        "[d_P] max|d_P - d~_P| = {worst:.3e}, Schwarz bound {bound:.3e}, ratio {:.3e}",
        worst / bound
    );
    assert!(worst > 0.0, "vacuous");
    assert!(
        worst <= bound,
        "{worst:.3e} above the Schwarz bound {bound:.3e}"
    );
}

/// Anchor 4: the screened route against a central difference of the energy
/// computed with the same screening.
#[test]
fn screened_gradient_is_the_derivative_of_the_screened_energy() {
    let s = setup(water());
    let thresh = first_dropping_thresh(&s);
    eprintln!("[FD] threshold {thresh:e}");
    let (g_reuse, g_rec, _) = reuse_and_recompute(&s, &config(thresh));
    let h = 2e-3; // Bohr
    let cfg = config(thresh);
    let mut worst_reuse: f64 = 0.0;
    let mut worst_rec: f64 = 0.0;
    let mut floor: f64 = 0.0;
    for (atom, axis) in [(0usize, 2usize), (1, 0), (2, 2)] {
        let e = |step: f64| {
            let mut mol = s.mol.clone();
            match axis {
                0 => mol.atoms[atom].x += step,
                1 => mol.atoms[atom].y += step,
                _ => mol.atoms[atom].zpos += step,
            }
            let s2 = setup(mol);
            scf(&s2, &cfg).energy
        };
        let fd = (e(h) - e(-h)) / (2.0 * h);
        let fd2 = (e(h / 2.0) - e(-h / 2.0)) / h;
        floor = floor.max((fd - fd2).abs());
        let (er, ec) = (
            (g_reuse[(atom, axis)] - fd).abs(),
            (g_rec[(atom, axis)] - fd).abs(),
        );
        eprintln!(
            "[FD] atom {atom} axis {axis}: fd {fd:.9}, reuse err {er:.3e}, recompute err {ec:.3e}, fd(h)-fd(h/2) {:.2e}",
            (fd - fd2).abs()
        );
        worst_reuse = worst_reuse.max(er);
        worst_rec = worst_rec.max(ec);
    }
    assert!(
        (&g_reuse - &g_rec).iter().any(|v| v.abs() > 1e-9),
        "vacuous: the routes agree, the screen changed nothing"
    );
    eprintln!(
        "[FD] worst: reuse {worst_reuse:.3e}, recompute {worst_rec:.3e}, FD step floor {floor:.2e}"
    );
    assert!(
        worst_reuse <= worst_rec + 2.0 * floor + 1e-7,
        "the screened route is farther from the screened energy's derivative"
    );
}
