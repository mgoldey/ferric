//! Anchors for the shell-SPARSE COSX half transforms
//! (`CosxHalfTransform::Sparse`: `F = D[Λ,A] X[A,:]`, `Ktilde[A,B] += X[A,:] G[B,:]^T`,
//! `S_num[A,A] += X[A,:] X[A,:]^T`) against the dense path
//! (`CosxHalfTransform::Dense`, the pre-change builder, kept as the cross-check).
//! Written BEFORE the sparse path compiled (repo rule: EXACTNESS ANCHOR FIRST);
//! pre-registration in `scripts/queue/out/cosx_sparse_prereg.md`.
//!
//! (a) TRIVIAL LIMIT — `eps_ao = eps_d = 0` keeps every shell (the rule is
//!     `>=`, so exact zeros are kept too) and must reproduce the dense K
//!     BITWISE, fit on and off, on water/cc-pVDZ and butane/def2-SVP, AND on a
//!     1e-8-scaled density (a silent floor on either eps is invisible at full
//!     scale — the density-driven-screen anchor learned this). With the fit on
//!     a bitwise K also pins `S_num` (it enters K through the Cholesky solve).
//!     The counters must report every AO active (`active_ao_frac == 1`,
//!     `lambda_frac == 1`).
//! (b) PRODUCTION eps — `max|K_sparse - K_dense| < 1e-6` on both systems at
//!     the default thresholds (grid errors are 5.6e-5 / 2.9e-4), and the
//!     converged COSX-RHF energy of water moves by `< 1e-7` Ha. Reachability
//!     inside (b): the sparse path must have DROPPED something
//!     (`active_ao_frac < 1`), or the accuracy claim is vacuous.
//! (c) REACHABILITY COUNTS — alkane_8/def2-SVP mean `|A|/nbf < 0.5` and mean
//!     `|Λ|/nbf < 0.7` as stated in the brief (the prereg EXPECTS these to be
//!     missed at 20 Bohr with a sound eps: AO reach ~12 Bohr, D decay ~30
//!     Bohr — the numbers are reported, eps is not tuned). A bar that IS
//!     reachable by construction, and that the "skip the Λ restriction"
//!     mutant must turn RED, is the water dimer 28 Bohr apart: each block's
//!     A is one monomer and D is block-diagonal to round-off, so both
//!     fractions sit near 0.5-0.65 (outer radial shells reach the other
//!     monomer through the sqrt(w)-damped points) — bar `< 0.75`.
//!
//! Mutation proofs (applied BY HAND to `cosx_k.rs`, run, reverted; results in
//! the commit message / report):
//!   M1 drop one genuinely active shell from A (e.g. skip shell 0 in the mask)
//!      -> (a) not bitwise AND (b) > 1e-6 (RED, RED);
//!   M2 skip the Λ restriction (Λ = every shell) -> K unchanged but
//!      `lambda_frac == 1` -> the dimer counters anchor RED;
//!   M3 wrong scatter offset in `Ktilde[A,B]` (shift the column map by one)
//!      -> (a) and (b) RED at O(1).

use ferric_core::basis::bundled;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_dft::grid::AtomicGridConfig;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::cosx_k::{CosxConfig, CosxHalfTransform, CosxK, CosxTimings, COSX_DEFAULT_EPS_AO, COSX_DEFAULT_EPS_D};
use ferric_scf::fock::KBuilder;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

const WATER: &str = "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
";

/// Two waters 15 Å (28.3 Bohr) apart — the same geometry `cosx_a_anchor.rs`
/// uses for the screen's reachability.
const WATER_DIMER_FAR: &str = "6
water dimer, 15 A apart
O 0 0 0
H 0 0.7572 0.5868
H 0 -0.7572 0.5868
O 15 0 0
H 15 0.7572 0.5868
H 15 -0.7572 0.5868
";

struct Setup {
    mol: Molecule,
    prep: PreparedBasis,
    d: Array2<f64>,
    c_occ: Array2<f64>,
}

fn testdata(rel: &str) -> String {
    format!("{}/../../{rel}", env!("CARGO_MANIFEST_DIR"))
}

/// Converged RHF density (+ C_occ) for `mol` / `basis`. `df` selects the
/// DF-JK SCF (for the larger systems, where the direct SCF is slow at one
/// thread); the provenance of D cancels out of every sparse-vs-dense
/// comparison (both paths contract the same matrix).
fn setup_with(mol: Molecule, basis: &str, df: bool, density_conv: f64) -> Setup {
    let bs = bundled(basis).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let ctx = ParallelContext::default();
    let aux = "def2-universal-jkfit";
    // The DF-JK reference keeps the default energy_conv (a sanity bound, not a
    // target — 1e-10 with a loose density_conv does not converge).
    let cfg = RhfConfig {
        energy_conv: if df { RhfConfig::default().energy_conv } else { 1e-10 },
        density_conv,
        integral_thresh: 1e-14,
        df_j_aux: df.then(|| aux.to_string()),
        df_k_aux: df.then(|| aux.to_string()),
        ..Default::default()
    };
    let res = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).expect("rhf");
    assert!(res.converged, "reference RHF did not converge");
    let nocc = (mol.nelec() / 2) as usize;
    let c_occ = res.mos_r().slice(ndarray::s![.., ..nocc]).to_owned();
    Setup { mol, prep, d: res.density_total.clone(), c_occ }
}

fn setup_water() -> Setup {
    setup_with(Molecule::parse_xyz(WATER, 0, 1).expect("water"), "cc-pvdz", false, 1e-8)
}

fn setup_butane() -> Setup {
    let mol = Molecule::load_xyz(&testdata("testdata/molecules/alkane_4.xyz")).expect("alkane_4.xyz");
    setup_with(mol, "def2-svp", false, 1e-8)
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a - b).mapv(f64::abs).fold(0.0_f64, |m, &v| m.max(v))
}

fn max_abs(a: &Array2<f64>) -> f64 {
    a.mapv(f64::abs).fold(0.0_f64, |m, &v| m.max(v))
}

fn grid(n_radial: usize, n_angular: usize) -> AtomicGridConfig {
    AtomicGridConfig { n_radial, n_angular, ..Default::default() }
}

fn cfg(half: CosxHalfTransform, fit: bool, screen: Option<f64>) -> CosxConfig {
    CosxConfig { grid: grid(50, 110), overlap_fit: fit, screen_thresh: screen, half_transform: half, ..CosxConfig::default() }
}

/// The production screen (the SCF path): `CosxConfig::default().screen_thresh`.
fn prod_screen() -> Option<f64> {
    let t = CosxConfig::default().screen_thresh;
    assert!(matches!(t, Some(v) if v > 0.0), "the density-driven screen is expected ON by default");
    t
}

const DENSE: CosxHalfTransform = CosxHalfTransform::Dense;
const SPARSE_ZERO: CosxHalfTransform = CosxHalfTransform::Sparse { eps_ao: 0.0, eps_d: 0.0 };
const SPARSE_PROD: CosxHalfTransform = CosxHalfTransform::Sparse { eps_ao: COSX_DEFAULT_EPS_AO, eps_d: COSX_DEFAULT_EPS_D };

/// One `build(D)` from a fresh builder: `(K, timings)`.
fn build_k(s: &Setup, c: CosxConfig, d: &Array2<f64>) -> (Array2<f64>, CosxTimings) {
    let ctx = ParallelContext::default();
    let n = s.prep.nbasis();
    let mut kb = CosxK::new(&ctx, &s.mol, &s.prep, c, usize::MAX).expect("CosxK::new");
    let mut k = Array2::zeros((n, n));
    kb.build(d, &mut k).expect("cosx build");
    (k, *kb.last_timings())
}

fn assert_all_active(t: &CosxTimings, label: &str) {
    assert!(t.n_blocks > 0, "{label}: no blocks counted");
    assert_eq!(t.active_ao_frac, 1.0, "{label}: eps = 0 dropped AOs from A (a floor on eps_ao)");
    assert_eq!(t.lambda_frac, 1.0, "{label}: eps = 0 dropped rows from Λ (a floor on eps_d)");
}

/// Anchor (a) on one system: eps = 0 is bitwise the dense path, fit on and
/// off, at full scale and on a 1e-8-scaled density.
fn trivial_limit_on(s: &Setup, label: &str) {
    for fit in [true, false] {
        for (scale, tag) in [(1.0, "full"), (1e-8, "1e-8 x D")] {
            let d = &s.d * scale;
            let (k_dense, _) = build_k(s, cfg(DENSE, fit, Some(0.0)), &d);
            let (k_zero, t) = build_k(s, cfg(SPARSE_ZERO, fit, Some(0.0)), &d);
            let scale_k = max_abs(&k_dense);
            assert!(scale_k > 0.05 * scale, "{label} {tag}: K ~ 0 ({scale_k:.3e}); vacuous");
            let dev = max_abs_diff(&k_dense, &k_zero);
            println!("trivial limit {label} fit={fit} {tag}: max|K_sparse0 - K_dense| = {dev:.3e}; A {:.4} Λ {:.4} B {:.4} over {} blocks",
                t.active_ao_frac, t.lambda_frac, t.out_ao_frac, t.n_blocks);
            assert!(k_dense == k_zero, "{label} fit={fit} {tag}: eps = 0 is not bitwise the dense K (max dev {dev:.3e})");
            assert_all_active(&t, &format!("{label} fit={fit} {tag}"));
        }
    }
}

/// Anchor (a): water/cc-pVDZ.
#[test]
fn sparse_zero_eps_matches_dense_bitwise_water() {
    let s = setup_water();
    trivial_limit_on(&s, "water/cc-pVDZ");
}

/// Anchor (a): butane/def2-SVP (the screen is ON at its default here as well,
/// so the Λ-zero rows and the `fmax = 0` interplay are exercised on a system
/// where the screen actually drops pairs).
#[test]
fn sparse_zero_eps_matches_dense_bitwise_butane() {
    let s = setup_butane();
    trivial_limit_on(&s, "butane/def2-SVP");
    for screen in [None, prod_screen()] {
        let (k_dense, td) = build_k(&s, cfg(DENSE, true, screen), &s.d);
        let (k_zero, t) = build_k(&s, cfg(SPARSE_ZERO, true, screen), &s.d);
        println!("trivial limit butane screen={screen:?}: max dev {:.3e}; B {:.4}; pairs kept {}/{} (dense {}/{})",
            max_abs_diff(&k_dense, &k_zero), t.out_ao_frac, t.pairs_kept, t.pairs_total, td.pairs_kept, td.pairs_total);
        assert!(k_dense == k_zero, "butane, screen {screen:?}: eps = 0 not bitwise dense");
        assert_all_active(&t, &format!("butane screen {screen:?}"));
        assert_eq!((t.pairs_kept, t.pairs_total), (td.pairs_kept, td.pairs_total), "eps = 0 changed the pair screen's decisions");
    }
}

/// Anchor (b) on one system: production eps vs dense on the (50,110)+fit
/// operating grid with the default screen.
fn production_eps_on(s: &Setup, label: &str) -> CosxTimings {
    let (k_dense, td) = build_k(s, cfg(DENSE, true, prod_screen()), &s.d);
    let (k_sp, t) = build_k(s, cfg(SPARSE_PROD, true, prod_screen()), &s.d);
    let dev = max_abs_diff(&k_dense, &k_sp);
    println!(
        "production eps {label}: max|K_sparse - K_dense| = {dev:.3e} (||K||max {:.3e}); A {:.4} Λ {:.4} B {:.4} over {} blocks; \
         GEMM s dense half/ktilde/snum {:.3}/{:.3}/{:.3} sparse {:.3}/{:.3}/{:.3} (+gather {:.3})",
        max_abs(&k_dense),
        t.active_ao_frac, t.lambda_frac, t.out_ao_frac, t.n_blocks,
        td.half_s, td.ktilde_s, td.snum_s, t.half_s, t.ktilde_s, t.snum_s, t.gather_s
    );
    assert!(dev < 1e-6, "{label}: sparse K differs from dense by {dev:.3e} >= 1e-6");
    assert!(t.active_ao_frac < 1.0, "{label}: production eps dropped nothing — the accuracy claim is vacuous");
    // Too-clean check (prereg): the active fraction must VARY across blocks
    // (inner vs outer radial shells), so min < mean < max strictly.
    assert!(
        t.active_ao_frac_min < t.active_ao_frac && t.active_ao_frac < t.active_ao_frac_max,
        "{label}: |A|/nbf is the same in every block (min {:.4} mean {:.4} max {:.4}) — mask independent of the points",
        t.active_ao_frac_min, t.active_ao_frac, t.active_ao_frac_max
    );
    t
}

#[test]
fn sparse_production_eps_k_error_below_1e6_water() {
    let s = setup_water();
    production_eps_on(&s, "water/cc-pVDZ");
}

#[test]
fn sparse_production_eps_k_error_below_1e6_butane() {
    let s = setup_butane();
    production_eps_on(&s, "butane/def2-SVP");
}

/// Anchor (b), SCF level: converged COSX-RHF energy of water, sparse vs dense.
#[test]
fn sparse_cosx_rhf_water_energy_matches_dense() {
    let mol = Molecule::parse_xyz(WATER, 0, 1).expect("water");
    let bs = bundled("cc-pvdz").expect("cc-pvdz");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let ctx = ParallelContext::default();
    let run = |half: CosxHalfTransform| {
        let c = RhfConfig {
            k_builder: Some("cosx".into()),
            cosx: CosxConfig { half_transform: half, ..CosxConfig::default() },
            energy_conv: 1e-10,
            density_conv: 1e-8,
            ..Default::default()
        };
        let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &c).expect("rhf");
        assert!(r.converged);
        (r.energy, r.iterations)
    };
    let (e_dense, it_dense) = run(DENSE);
    let (e_sp, it_sp) = run(SPARSE_PROD);
    let de = e_sp - e_dense;
    println!("water/cc-pVDZ COSX-RHF: dense E={e_dense:.10} ({it_dense} it), sparse E={e_sp:.10} ({it_sp} it), dE={de:+.3e}");
    assert!(de.abs() < 1e-7, "sparse COSX SCF energy differs from dense by {de:e} Ha");
    assert_eq!(CosxConfig::default().half_transform, SPARSE_PROD, "sparse at the pre-registered eps must be the default");
}

/// The sparse `build_from_occ(C)` forms `D = C C^T` and takes the D path, so
/// it must equal `build(C C^T)` BITWISE; and at production eps it must stay
/// within the (b) bar of the dense `C (C^T X)` transform.
#[test]
fn sparse_build_from_occ_is_the_density_path() {
    let s = setup_water();
    let ctx = ParallelContext::default();
    let n = s.prep.nbasis();
    let d_occ = s.c_occ.dot(&s.c_occ.t());
    let mut kb = CosxK::new(&ctx, &s.mol, &s.prep, cfg(SPARSE_PROD, true, None), usize::MAX).expect("sparse");
    let mut k_c = Array2::zeros((n, n));
    let mut k_d = Array2::zeros((n, n));
    kb.build_from_occ(&s.c_occ, &mut k_c).expect("occ");
    kb.build(&d_occ, &mut k_d).expect("density");
    assert!(k_c == k_d, "sparse build_from_occ(C) is not bitwise build(C C^T): {:.3e}", max_abs_diff(&k_c, &k_d));
    let mut kb_dense = CosxK::new(&ctx, &s.mol, &s.prep, cfg(DENSE, true, None), usize::MAX).expect("dense");
    let mut k_dense = Array2::zeros((n, n));
    kb_dense.build_from_occ(&s.c_occ, &mut k_dense).expect("dense occ");
    let dev = max_abs_diff(&k_c, &k_dense);
    println!("build_from_occ sparse vs dense: max|dK| = {dev:.3e}");
    assert!(dev < 1e-6, "sparse build_from_occ vs dense differ by {dev:.3e}");
}

/// Anchor (c): reachability COUNTS on alkane_8/def2-SVP at production eps.
/// The bars are the brief's (`< 0.5`, `< 0.7`); the prereg expects them to be
/// missed at 20 Bohr with a sound eps, in which case the printed numbers are
/// the deliverable and eps is NOT tuned.
#[test]
fn sparse_reachability_counts_alkane_8() {
    let mol = Molecule::load_xyz(&testdata("testdata/molecules/alkane_8.xyz")).expect("alkane_8.xyz");
    let s = setup_with(mol, "def2-svp", true, 1e-6);
    let (_k, t) = build_k(&s, cfg(SPARSE_PROD, true, prod_screen()), &s.d);
    println!(
        "alkane_8/def2-SVP nbf={} eps_ao={COSX_DEFAULT_EPS_AO:e} eps_d={COSX_DEFAULT_EPS_D:e}: mean |A|/nbf {:.4} (min {:.4} max {:.4}), mean |Λ|/nbf {:.4}, mean |B|/nbf {:.4}, {} blocks",
        s.prep.nbasis(), t.active_ao_frac, t.active_ao_frac_min, t.active_ao_frac_max, t.lambda_frac, t.out_ao_frac, t.n_blocks
    );
    assert!(t.active_ao_frac < 0.5, "alkane_8: mean |A|/nbf = {:.4} >= 0.5 (reported, not tuned — see prereg)", t.active_ao_frac);
    assert!(t.lambda_frac < 0.7, "alkane_8: mean |Λ|/nbf = {:.4} >= 0.7 (reported, not tuned — see prereg)", t.lambda_frac);
}

/// Anchor (c'), reachable by construction: water dimer 28 Bohr apart. Both
/// fractions must sit below 0.75 (derivation in the prereg: 0.5 per monomer
/// block plus the outer radial shells that reach the other monomer). This is
/// the counter the M2 mutant (Λ = all shells) must turn RED, and the K must
/// still agree with dense to the (b) bar.
#[test]
fn sparse_reachability_water_dimer_far() {
    let mol = Molecule::parse_xyz(WATER_DIMER_FAR, 0, 1).expect("dimer");
    let s = setup_with(mol, "cc-pvdz", false, 1e-8);
    let (k_dense, _) = build_k(&s, cfg(DENSE, true, prod_screen()), &s.d);
    let (k_sp, t) = build_k(&s, cfg(SPARSE_PROD, true, prod_screen()), &s.d);
    let dev = max_abs_diff(&k_dense, &k_sp);
    println!(
        "water dimer 28 Bohr / cc-pVDZ: max|K_sparse - K_dense| = {dev:.3e}; mean |A|/nbf {:.4} (min {:.4} max {:.4}), |Λ|/nbf {:.4}, |B|/nbf {:.4}, {} blocks",
        t.active_ao_frac, t.active_ao_frac_min, t.active_ao_frac_max, t.lambda_frac, t.out_ao_frac, t.n_blocks
    );
    assert!(dev < 1e-6, "dimer: sparse K differs from dense by {dev:.3e}");
    assert!(t.active_ao_frac < 0.75, "dimer: mean |A|/nbf = {:.4} >= 0.75", t.active_ao_frac);
    assert!(t.lambda_frac < 0.75, "dimer: mean |Λ|/nbf = {:.4} >= 0.75 — the D-sparsity restriction is not biting", t.lambda_frac);
    // With the production screen the kernel touches only shells with a
    // significant F (one monomer for the inner blocks), so B must be sparse too.
    assert!(t.out_ao_frac < 0.75, "dimer: mean |B|/nbf = {:.4} >= 0.75 — the touched-shell set is not biting", t.out_ao_frac);
    assert!(t.active_ao_frac_min <= 0.5 + 1e-12, "dimer: no block is confined to one monomer (min |A|/nbf {:.4})", t.active_ao_frac_min);
}

/// The dense path reports every AO as active (honest counters, not zeros or
/// NaN), and the config parser for the half transform is strict.
#[test]
fn dense_counters_and_strict_parser() {
    let s = setup_water();
    let (_k, t) = build_k(&s, cfg(DENSE, true, None), &s.d);
    assert_eq!((t.active_ao_frac, t.lambda_frac, t.out_ao_frac), (1.0, 1.0, 1.0));
    assert!(t.n_blocks > 0);
    assert_eq!(CosxHalfTransform::parse_config_str("dense").unwrap(), DENSE);
    assert_eq!(CosxHalfTransform::parse_config_str("sparse").unwrap(), SPARSE_PROD);
    assert!(CosxHalfTransform::parse_config_str("Dense").is_err(), "case-sensitive: no silent coercion");
    assert!(CosxHalfTransform::parse_config_str("").is_err());
    assert_eq!(DENSE.as_str(), "dense");
    assert_eq!(SPARSE_PROD.as_str(), "sparse");
}
