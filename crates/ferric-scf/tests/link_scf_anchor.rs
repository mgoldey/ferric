//! SCF-level exactness anchors for `k_builder = "link"`.
//!
//! # Why this file exists
//!
//! Every pre-existing LinK correctness test compares a single K build against
//! the direct builder on a CONVERGED density of a molecule small enough
//! (water, CH4) that every shell pair is significant. `tests/screening_exactness.rs`
//! added an alkane_8 threshold sweep, but it too contracts one converged
//! density. None of them drive a full SCF through LinK on a system where the
//! density-pair list is genuinely sparse — which is exactly where the
//! ket-loop coverage of `LinkK::build` was wrong (see the module doc of
//! `link_k.rs`):
//!
//! ```text
//! butane / def2-SVP, RhfConfig::default(), origin/main (2026-09-07)
//!   k_builder = "direct":  E = -157.1861392016 Ha, 12 iterations, converged
//!   k_builder = "link":    E = -162.7632969403 Ha, 31 iterations, NOT converged
//! ```
//!
//! Per-iteration instrumentation (`FERRIC_LINK_DEBUG=1`) showed the density
//! path was taken every iteration and the density-pair list WAS refreshed —
//! iteration 1 (SAD guess, atom-block-diagonal D, 120/2916 density pairs kept)
//! already had `max|K_link - K_direct| = 0.55` against `max|K| = 7.2`, and the
//! resulting Fock drove the density into the near-linear-dependent modes
//! (max|D| = 478 at iteration 2 vs ~2 for a physical AO density). Water hides
//! all of this because its pair lists are complete at any threshold.
//!
//! The tests here run the FULL SCF through both builders and require the
//! converged energies, convergence flags and iteration counts to agree.

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::fock::KBuilder;
use ferric_scf::link_k::LinkK;
use ferric_scf::rhf::{build_jk, solve_rhf, RhfConfig};
use ferric_scf::result::ScfResult;
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

const WATER: &str = "3
water
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
";

fn load(stem: &str) -> Molecule {
    Molecule::load_xyz(&format!("../../testdata/molecules/{stem}.xyz"))
        .unwrap_or_else(|e| panic!("cannot load {stem}.xyz: {e}"))
}

fn run(mol: &Molecule, bs: &BasisSet, cfg: &RhfConfig) -> ScfResult {
    let prep = PreparedBasis::new(mol, bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    solve_rhf(&ParallelContext::default(), mol, &prep, op, &bounds, cfg).expect("rhf")
}

/// Both builders at the repo-default thresholds. Returns (direct, link).
fn direct_and_link(mol: &Molecule, basis_name: &str) -> (ScfResult, ScfResult) {
    let bs = basis::bundled(basis_name).expect("basis");
    let direct = run(mol, &bs, &RhfConfig::default());
    let link = run(mol, &bs, &RhfConfig { k_builder: Some("link".into()), ..Default::default() });
    (direct, link)
}

fn check_scf_agreement(label: &str, direct: &ScfResult, link: &ScfResult, e_bar: f64) {
    let de = link.energy - direct.energy;
    let di = link.iterations as i64 - direct.iterations as i64;
    println!(
        "{label}: direct E={:.10} ({} iters, converged={}); link E={:.10} ({} iters, converged={}); \
         dE={de:+.3e} dIter={di:+}",
        direct.energy, direct.iterations, direct.converged, link.energy, link.iterations, link.converged
    );
    assert!(direct.converged, "{label}: direct SCF did not converge");
    assert!(link.converged, "{label}: LinK SCF did not converge (E={:.10}, {} iters)", link.energy, link.iterations);
    assert!(
        de.abs() <= e_bar,
        "{label}: LinK SCF energy differs from direct by {de:+.3e} Ha (bar {e_bar:.0e})"
    );
    assert!(
        di.abs() <= 3,
        "{label}: LinK took {} iterations vs direct {} (more than 3 apart)",
        link.iterations, direct.iterations
    );
}

/// The anchor. Butane/def2-SVP is the smallest system on which the
/// density-pair list is sparse at the SAD guess AND the significant-pair
/// list is incomplete at the default 1e-12 threshold (2698/2916 kept), so
/// both defects of the old ket loop fire. Failed on the unfixed code with
/// E = -162.76 Ha / 31 iterations / not converged.
#[test]
fn link_scf_matches_direct_scf_butane() {
    let mol = load("alkane_4");
    let (direct, link) = direct_and_link(&mol, "def2-svp");
    check_scf_agreement("butane/def2-SVP", &direct, &link, 1e-6);
}

/// (d) The same SCF-level agreement one size up, on alkane_8/def2-SVP.
///
/// The butane case above is the one that CAUGHT the #50 defects; this one
/// guards the screening change against a different failure. A per-quartet
/// density screen is applied afresh at every SCF iteration, on densities that
/// are far from converged early on (the SAD guess is atom-block-diagonal, so
/// its far-field blocks are exactly zero and a density screen prunes them
/// aggressively). A screen that is correct on a converged density can still
/// destabilise the iteration by pruning differently as D evolves — which shows
/// up as an iteration-count change or a non-variational trajectory, not as a
/// K-matrix error on the final density.
///
/// Asserting the ENERGY and the ITERATION COUNT together is what makes this
/// distinct from the single-build comparisons: the energy alone would pass for
/// a screen that converges to the right answer along a worse path.
#[test]
fn link_scf_matches_direct_scf_alkane_8() {
    let mol = load("alkane_8");
    let (direct, link) = direct_and_link(&mol, "def2-svp");
    check_scf_agreement("alkane_8/def2-SVP", &direct, &link, 1e-9);
}

/// Water/cc-pVDZ: every shell pair is significant, so LinK and the direct
/// builder walk the same quartets. Kept so the fix cannot regress the case
/// that always worked (agreement here was 1e-10 before the fix).
#[test]
fn link_scf_matches_direct_scf_water() {
    let mol = Molecule::parse_xyz(WATER, 0, 1).expect("water");
    let (direct, link) = direct_and_link(&mol, "cc-pvdz");
    check_scf_agreement("water/cc-pVDZ", &direct, &link, 1e-9);
}

/// Standalone single-build comparison on the CONVERGED butane density at the
/// production threshold: LinK vs the dense screened build at the SAME
/// threshold. Before the fix this was 2.784e-3 (the value a separate harness
/// reported) — three orders of magnitude above what a 1e-12 screen can
/// explain. See `LINK_VS_DIRECT_K_BAR` for the post-fix measurement.
#[test]
fn link_k_matches_direct_k_butane_at_production_thresh() {
    let mol = load("alkane_4");
    let bs = basis::bundled("def2-svp").expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let ctx = ParallelContext::default();
    let d = run(&mol, &bs, &RhfConfig::default()).density_total;
    let thresh = 1e-12;
    let n = prep.nbasis();

    let mut j = Array2::zeros((n, n));
    let mut k_direct = Array2::zeros((n, n));
    build_jk(&ctx, &prep, &bounds, thresh, &d, &mut j, &mut k_direct).expect("direct jk");

    let mut link = LinkK::new(&ctx, &prep, &bounds, op, thresh, usize::MAX);
    link.update_density(&d);
    let mut k_link = Array2::zeros((n, n));
    link.build(&d, &mut k_link).expect("link");

    let max_dk = (&k_link - &k_direct).iter().fold(0.0f64, |m, v| m.max(v.abs()));
    println!("butane/def2-SVP converged D, thresh=1e-12: max|K_link - K_direct| = {max_dk:.3e}");
    assert!(
        max_dk < LINK_VS_DIRECT_K_BAR,
        "LinK K differs from the direct K at the same 1e-12 threshold by {max_dk:.3e}"
    );
}

/// (b) Production-threshold correctness at alkane_16 — the size where the
/// density screen MUST bite.
///
/// # Why C16 and not another butane variant
///
/// Every pre-existing K-value comparison in this repo runs on a system at or
/// below butane. Those systems sit BELOW the ~30 Bohr alkane density-matrix
/// decay length (repo memory), so their density-pair lists are essentially
/// complete and a density screen has nothing to remove. That makes them unable
/// to catch the failure mode that matters for a screening change: a screen
/// that is too TIGHT removes contributions only in the far field, which by
/// construction does not exist on a small molecule.
///
/// alkane_16 (198 shells, 394 def2-SVP functions, ~50 Bohr end to end) is past
/// the onset and is the system whose counts are tabulated in
/// `scripts/queue/out/link_fixed_counts.md`. A pairwise density screen that
/// drops genuinely contributing far-field blocks shows up HERE first and
/// nowhere else in the suite.
///
/// Bar is the ticket's 1e-12. Post-#50 measured `max|K_link - K_directK|` at
/// this threshold is 2.4e-14 on both C8 and C16, so 1e-12 clears the correct
/// value by ~40x while sitting far below anything a broken bound produces (the
/// three pre-#50 defects gave 1.7e-3 .. 2.8e-3).
#[test]
fn link_k_matches_direct_k_alkane_16_at_production_thresh() {
    let mol = load("alkane_16");
    let bs = basis::bundled("def2-svp").expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("prep");
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, &prep).expect("schwarz");
    let ctx = ParallelContext::default();
    let d = run(&mol, &bs, &RhfConfig::default()).density_total;
    let thresh = 1e-12;
    let n = prep.nbasis();

    let mut j = Array2::zeros((n, n));
    let mut k_direct = Array2::zeros((n, n));
    build_jk(&ctx, &prep, &bounds, thresh, &d, &mut j, &mut k_direct).expect("direct jk");

    let mut link = LinkK::new(&ctx, &prep, &bounds, op, thresh, usize::MAX);
    link.update_density(&d);
    let mut k_link = Array2::zeros((n, n));
    link.build(&d, &mut k_link).expect("link");

    let max_dk = (&k_link - &k_direct).iter().fold(0.0f64, |m, v| m.max(v.abs()));
    println!("alkane_16/def2-SVP converged D, thresh=1e-12: max|K_link - K_direct| = {max_dk:.3e}");
    assert!(
        max_dk < LINK_VS_DIRECT_K_BAR,
        "LinK K differs from the direct K at the same 1e-12 threshold by {max_dk:.3e} on \
         alkane_16 — past the ~30 Bohr locality onset, this is where an over-tight density \
         screen drops real far-field contributions."
    );
}

/// Bar for the standalone butane K comparison at thresh = 1e-12.
///
/// Measured, butane/def2-SVP, converged density, LinK vs the direct build at
/// the same 1e-12 threshold, as the three pair-list defects were removed:
///
/// ```text
/// unfixed                                            2.784e-3
/// + ket loop over dp(ish) ∪ dp(jsh)                  1.716e-3
/// + density-pair criterion |D|·qmax(j)·qmax(σ)       3.716e-7   (= 0.37·sqrt(thresh): the Q² pair cut)
/// + significant-pair criterion Q·Qmax > thresh       4.563e-12  (== direct(1e-12) vs direct(0))
/// ```
///
/// 1e-9 is ~200x above the measured value and ~400x below the least-wrong
/// partial fix, so the absence of ANY one of the three fixes fails here.
const LINK_VS_DIRECT_K_BAR: f64 = 1e-9;
