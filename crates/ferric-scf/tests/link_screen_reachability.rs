//! REACHABILITY anchors for LinK's density screening — the asserts whose
//! absence let a non-screening screen ship.
//!
//! # Why this file exists
//!
//! Post-#50, LinK was CORRECT and SLOWER than the builder it replaces. The
//! measured state (thresh 1e-12, def2-SVP, converged direct-SCF density,
//! `scripts/queue/out/link_fixed_counts.md`):
//!
//! ```text
//! system      DirectK quartets   fixed LinK    build_jk (pairwise D)   LinK/build_jk
//! alkane_8      8,775,381         8,775,277      8,107,064               1.082
//! alkane_16    48,644,165        48,642,325     42,916,434               1.133
//!
//! kept pair fractions:  sp `Q·Qmax > t`   0.855 (C8) / 0.560 (C16)
//!                       dp `|D|·qmax(j)·qmax(σ) > t`   1.000 / 1.000
//! ```
//!
//! Every correctness test in the repo passed on that state, because every one
//! of them asks only "does LinK reproduce K?". LinK did — to 2.4e-14, as a
//! strict SUBSET of DirectK's quartets. What no test asked was whether the
//! screen SCREENS. The density-pair list retained 100% of pairs at both sizes
//! and the per-quartet screen used a single global `max|D|` scalar, so LinK
//! evaluated 8%/13% MORE quartets than the default `build_jk` — a screening
//! kernel doing more work than the dense builder, with every light green.
//!
//! A screen that prunes nothing is not a screen, and a correctness suite alone
//! cannot see that. These are COUNT assertions (deterministic, load-immune, no
//! wall-clock) and they fail if the screening quality regresses even while K
//! stays perfect.
//!
//! # The two things asserted, and why both are needed
//!
//! 1. **LinK's quartet count is BELOW `build_jk`'s** on the same density. This
//!    is the end-to-end statement: LinK must beat the builder it is offered as
//!    an alternative to, on the metric that is not load-dependent.
//! 2. **The density-pair list prunes a non-trivial fraction** past the ~30 Bohr
//!    locality onset. This is the mechanism. Without it, (1) could be satisfied
//!    by the per-quartet screen alone while the dp list stays vacuous — which
//!    is a reportable state, not a passing one.
//!
//! # These bars can be WRONG in the passing direction — read this before moving one
//!
//! A count bar has a failure mode a correctness bar does not: an INVALID
//! (too-tight) screen makes counts go DOWN, which looks like success here. The
//! count assertions in this file are therefore NOT self-sufficient, and must
//! never be "fixed" by loosening them when they fail. They are paired with the
//! value assertions in `link_scf_anchor.rs` and `screening_exactness.rs`
//! (LinK K == DirectK K at the production threshold, and == dense K at
//! thresh -> 0). Counts falling while K holds is a fix; counts falling while K
//! moves is a broken bound. Neither test alone can tell those apart.

use ferric_core::basis::{self, BasisSet};
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_scf::direct_k::DirectK;
use ferric_scf::fock::KBuilder;
use ferric_scf::link_k::LinkK;
use ferric_scf::pairs::{DensityPairs, SignificantPairs};
use ferric_scf::rhf::{build_jk, solve_rhf, RhfConfig};
use ferric_scf::screening::SchwarzBounds;
use ndarray::Array2;

/// The threshold a production SCF actually runs at (`RhfConfig::default()`),
/// and the one every count in `link_fixed_counts.md` was taken at.
const PRODUCTION_THRESH: f64 = 1e-12;

fn load_mol(stem: &str) -> Molecule {
    Molecule::load_xyz(&format!("../../testdata/molecules/{stem}.xyz"))
        .unwrap_or_else(|e| panic!("cannot load {stem}.xyz: {e}"))
}

/// Converged direct-SCF density — the same provenance as the audit harness's
/// counts, so the numbers here are comparable to `link_fixed_counts.md`.
///
/// Stock `RhfConfig::default()` on purpose: what these anchors need from the
/// density is that it be FIXED and physical. Tightening convergence here buys
/// nothing and costs enormously at C16 (see the note in
/// `screening_exactness.rs::converged_density`).
fn converged_density(mol: &Molecule, prep: &PreparedBasis) -> Array2<f64> {
    let op = Operator::coulomb();
    let bounds = SchwarzBounds::compute(op, prep).expect("Schwarz bounds");
    let res = solve_rhf(&ParallelContext::default(), mol, prep, op, &bounds, &RhfConfig::default())
        .expect("RHF solve");
    assert!(
        res.converged,
        "RHF did not converge — refusing to measure screening counts on an unconverged density"
    );
    res.density_total
}

/// Quartet counts for all three builders on ONE density at ONE threshold.
///
/// All three return the number of unique canonical shell quartets that survived
/// their own screen and for which libint2 returned a block — the same quantity
/// `link_fixed_counts.md` tabulates. Counts are deterministic and
/// thread-invariant (the builders' group partitions are pure functions of the
/// pair list, never of the thread count), so these assertions are load-immune.
struct Counts {
    direct_k: usize,
    build_jk: usize,
    link: usize,
    /// Kept fraction of the `nsh²` ordered density pairs.
    dp_frac: f64,
    /// Kept fraction of the `nsh²` ordered significant pairs.
    sp_frac: f64,
    /// `max|K_link - K_directk|` — the guard that a count drop is a real fix
    /// and not a broken bound.
    k_dev: f64,
}

fn measure(stem: &str, basis_name: &str) -> Counts {
    let mol = load_mol(stem);
    let bs: BasisSet = basis::bundled(basis_name).expect("basis");
    let prep = PreparedBasis::new(&mol, &bs).expect("PreparedBasis");
    let nsh = prep.nshells();
    let nbf = prep.nbasis();
    let op = Operator::coulomb();
    let ctx = ParallelContext::default();
    let bounds = SchwarzBounds::compute(op, &prep).expect("Schwarz bounds");
    let d = converged_density(&mol, &prep);

    let mut k_direct = Array2::<f64>::zeros((nbf, nbf));
    let mut dk = DirectK::new(&ctx, &prep, &bounds, PRODUCTION_THRESH, usize::MAX);
    let direct_k = dk.build(&d, &mut k_direct).expect("DirectK build");

    let mut j_b = Array2::<f64>::zeros((nbf, nbf));
    let mut k_b = Array2::<f64>::zeros((nbf, nbf));
    let build_jk_n = build_jk(&ctx, &prep, &bounds, PRODUCTION_THRESH, &d, &mut j_b, &mut k_b)
        .expect("build_jk");

    let mut k_link = Array2::<f64>::zeros((nbf, nbf));
    let mut link = LinkK::new(&ctx, &prep, &bounds, op, PRODUCTION_THRESH, usize::MAX);
    link.update_density(&d);
    let link_n = link.build(&d, &mut k_link).expect("LinK build");

    let sq = (nsh * nsh) as f64;
    let dp = DensityPairs::build(&d, &bounds, &prep, PRODUCTION_THRESH).total_pairs();
    let sp = SignificantPairs::build(&bounds, nsh, PRODUCTION_THRESH).total_pairs();
    let k_dev = (&k_link - &k_direct).iter().fold(0.0f64, |m, v| m.max(v.abs()));

    eprintln!(
        "\n=== {stem}/{basis_name} nsh={nsh} nbf={nbf} thresh={PRODUCTION_THRESH:.0e} ===\n  \
         DirectK {direct_k}  build_jk {build_jk_n}  LinK {link_n}  \
         (LinK/build_jk = {:.4}, LinK/DirectK = {:.5})\n  \
         sp kept {sp}/{nsh}² = {:.4}   dp kept {dp}/{nsh}² = {:.4}   max|K_link-K_directK| = {k_dev:.3e}",
        link_n as f64 / build_jk_n as f64,
        link_n as f64 / direct_k as f64,
        sp as f64 / sq,
        dp as f64 / sq,
    );

    Counts {
        direct_k,
        build_jk: build_jk_n,
        link: link_n,
        dp_frac: dp as f64 / sq,
        sp_frac: sp as f64 / sq,
        k_dev,
    }
}

/// Bar for `max|K_link - K_directK|` at the production threshold.
///
/// Measured post-#50 at 2.4e-14 on both C8 and C16. 1e-12 is the ticket's
/// stated correctness bar and ~40x above the measured value, so a screen that
/// has become an invalid (non-)bound fails here rather than being rewarded by
/// the count assertions below.
const K_DEV_BAR: f64 = 1e-12;

/// Maximum acceptable ratio of LinK quartets to `build_jk` quartets.
///
/// **Where this number comes from.** Post-#50 MEASURED ratios were 1.082 (C8)
/// and 1.133 (C16) — LinK doing 8%/13% MORE work than the default builder. The
/// requirement is not "a bit better than that"; it is that LinK must be BELOW
/// `build_jk`, which is what its existence has to justify. So the bar is 1.0,
/// with the ratio asserted strictly less.
///
/// This is a bar the pre-fix code CANNOT pass by construction: with a global
/// `max|D|` per-quartet screen, LinK's screen is strictly LOOSER than
/// `build_jk`'s six-pairwise one, so its count is necessarily >= `build_jk`'s.
/// The assertion is therefore reachable in exactly one way — by making LinK's
/// per-quartet screen at least as tight as the default builder's.
const LINK_VS_BUILD_JK_MAX_RATIO: f64 = 1.0;

/// Minimum fraction of density pairs the dp list must PRUNE past the locality
/// onset.
///
/// **Where this number comes from, and why it is deliberately modest.**
/// Post-#50 the dp list pruned exactly NOTHING: 1.000 kept at both C8 and C16,
/// because `|D|·qmax(j)·qmax(σ) > t` degenerates to `|D| > t/Qmax²` with
/// `qmax` a molecule-wide constant (see
/// `scripts/queue/out/link_pairwise_screen_design.md`). Any nonzero pruning is
/// therefore a strict improvement over the shipped state.
///
/// The design note pre-registers the expectation that the dp list is NOT where
/// the win comes from — the per-quartet screen is — so this bar is set to catch
/// "the dp criterion is still vacuous", not to claim a large effect. 5% is
/// above the 0% the old criterion achieved and above any plausible boundary
/// noise, while staying honest about the size of the effect being claimed. If
/// the measured pruning turns out to be large, this bar is RAISED to just under
/// the measured value in the same commit that measures it; if it turns out to
/// be under 5%, that is a reportable negative about the dp list and gets
/// written down rather than accommodated by lowering the bar.
const DP_MIN_PRUNED_FRACTION: f64 = 0.05;

/// Shared body: the reachability assertions on one system.
///
/// Ordering is deliberate — the CORRECTNESS guard runs FIRST. A count
/// improvement bought by an invalid bound must be reported as a broken bound,
/// not as a pass.
fn check_reachability(stem: &str, basis_name: &str, require_dp_pruning: bool) {
    let c = measure(stem, basis_name);

    assert!(
        c.k_dev < K_DEV_BAR,
        "{stem}/{basis_name}: max|K_link - K_directK| = {:.3e} exceeds {K_DEV_BAR:.0e}. The count \
         assertions below are meaningless while this fails — a tighter screen that CHANGES K is \
         an invalid bound, not a speedup. Fix the bound; do not loosen this bar.",
        c.k_dev
    );

    let ratio = c.link as f64 / c.build_jk as f64;
    assert!(
        ratio < LINK_VS_BUILD_JK_MAX_RATIO,
        "{stem}/{basis_name}: LinK evaluated {} quartets vs build_jk's {} (ratio {ratio:.4}, bar \
         {LINK_VS_BUILD_JK_MAX_RATIO}). LinK is doing MORE work than the default builder it is \
         offered as an alternative to. Post-#50 this ratio was 1.082 (C8) / 1.133 (C16) because \
         LinK's per-quartet screen used a single global max|D| scalar while build_jk uses the \
         six-pairwise Haser-Ahlrichs table. K is correct (checked above), so this is a SCREENING \
         QUALITY regression, not a correctness one.",
        c.link, c.build_jk
    );

    // LinK must also remain a subset of DirectK — the #50 property. Same
    // per-quartet Schwarz product, but LinK screens on a tighter density key,
    // so it can only ever evaluate fewer.
    assert!(
        c.link <= c.direct_k,
        "{stem}/{basis_name}: LinK evaluated {} quartets vs DirectK's {} — LinK is no longer a \
         subset of the canonical loop, which means the pair lists are admitting quartets the \
         dense screen rejects (a dedup/ownership defect).",
        c.link, c.direct_k
    );

    if require_dp_pruning {
        let pruned = 1.0 - c.dp_frac;
        assert!(
            pruned >= DP_MIN_PRUNED_FRACTION,
            "{stem}/{basis_name}: the density-pair list kept {:.4} of all ordered shell pairs, \
             pruning only {:.2}% (bar {:.0}%). A bound that prunes nothing at the production \
             threshold is not a screen. Post-#50 this was exactly 1.0000 kept because \
             |D|*qmax(j)*qmax(sigma) > t degenerates to |D| > t/Qmax^2 with qmax a molecule-wide \
             constant. Do NOT raise the threshold to force pruning — fix what the criterion bounds.",
            c.dp_frac,
            100.0 * pruned,
            100.0 * DP_MIN_PRUNED_FRACTION
        );
    }

    // Sanity: the significant-pair list must still be doing its (already
    // working) job, so a regression there cannot hide behind the dp assertions.
    // Measured 0.855 (C8) / 0.560 (C16); 0.99 is far above both and only fires
    // if sp screening has effectively stopped.
    assert!(
        c.sp_frac < 0.99,
        "{stem}/{basis_name}: SignificantPairs kept {:.4} of all ordered pairs — the geometric \
         pair screen has stopped screening (measured 0.855 at C8 / 0.560 at C16).",
        c.sp_frac
    );
}

/// THE anchor: past the ~30 Bohr locality onset, LinK must evaluate fewer
/// quartets than the default builder AND its density-pair list must prune.
///
/// alkane_16 (C16H34, 198 shells / 394 def2-SVP functions, ~50 Bohr end to end)
/// is the system `link_fixed_counts.md` measured the 1.133 over-evaluation on,
/// and repo memory puts the alkane density-matrix decay length at ~30 Bohr — so
/// this is past the onset, where a locality screen is REQUIRED to bite. A
/// negative result measured below the onset would be meaningless (repo rule:
/// "do not declare a negative below the onset"); this system is chosen so that
/// the measurement is licensed.
#[test]
fn link_screens_below_build_jk_alkane_16() {
    check_reachability("alkane_16", "def2-svp", true);
}

/// The same anchor one size further out. C20 is deeper past the onset, so the
/// dp list has strictly more far-field pairs available to prune: if the
/// criterion is right, pruning here must be at least as good as at C16, and
/// this is the case that would expose a screen whose pruning does not grow with
/// system size (i.e. one that is cutting a fixed fraction for a reason
/// unrelated to locality).
#[test]
#[ignore = "expensive: converged SCF + three full K builds on 20 heavy atoms; run explicitly"]
fn link_screens_below_build_jk_alkane_20() {
    check_reachability("alkane_20", "def2-svp", true);
}

/// alkane_8 — BELOW the locality onset (~26 Bohr end to end vs a ~30 Bohr decay
/// length), so the dp list is NOT required to prune here and that assertion is
/// switched off.
///
/// The `build_jk` count comparison still applies and is the point of this case:
/// the per-quartet pairwise screen is a LOCAL property of each quartet, not a
/// locality effect, so it must pay off at C8 too (measured 1.082 over-evaluation
/// pre-fix). Keeping the dp requirement off here is what stops this test from
/// making a locality claim the system size cannot support.
#[test]
fn link_screens_below_build_jk_alkane_8() {
    check_reachability("alkane_8", "def2-svp", false);
}
