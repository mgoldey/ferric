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
//! # What is asserted
//!
//! 1. **LinK's quartet count is BELOW `build_jk`'s** on the same density. This
//!    is the load-bearing assertion and the end-to-end statement: LinK must beat
//!    the builder it is offered as an alternative to, on a metric that is not
//!    load-dependent.
//! 2. **LinK remains a subset of `DirectK`** — the #50 property, which a
//!    dedup/ownership defect would break.
//! 3. The **density-pair pruning fraction is MEASURED and printed**, but its bar
//!    is currently disabled. The ticket asked for the dp criterion to be made to
//!    prune; working out what it bounds says it cannot, at pair-list
//!    granularity, and that negative is recorded in full on
//!    `DP_MIN_PRUNED_FRACTION` and `DensityPairs::build` rather than papered
//!    over. `qmax_spread_is_small_enough_to_explain_the_vacuous_dp_list` is the
//!    measurement that would OVERTURN that reasoning if it is wrong, and it
//!    fails loudly in that case.
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
/// # This bar was 1e-12 and it was WRONG — the reasoning, because the
/// # correction looks like a loosening and must not be mistaken for one
///
/// 1e-12 came from the post-#50 measurement of 2.4e-14 (C8 and C16). But that
/// number was small for a reason that stopped being true: pre-fix, LinK and
/// `DirectK` used the IDENTICAL per-quartet screen (both global `max|D|`), so
/// they walked essentially the same quartets and their difference was near-zero
/// by construction rather than by accuracy. It measured screen SAMENESS, not
/// correctness.
///
/// Now that LinK screens on the pairwise exchange key and `DirectK` still
/// screens on the global scalar, the two deliberately walk different quartet
/// sets, and the difference moves to the threshold-scale residual that any two
/// differently-screened builders show. The right calibration is therefore the
/// deviation between two builders that were ALREADY accepted as correct and
/// already screen differently from each other — `build_jk` (six-pairwise) vs
/// `DirectK` (global scalar), which has nothing to do with LinK:
///
/// ```text
///                                    C8          C16
/// build_jk   vs DirectK  (baseline)  9.176e-12   9.399e-12
/// LinK       vs DirectK  (this fix)  9.923e-12   1.288e-11
/// ```
///
/// LinK has landed in the same company as the pre-existing pair, ~1.1-1.4x the
/// baseline — not a new error mode. The independent confirmations that this is
/// a screening residual and not a broken bound:
///
/// * `screening_exactness::link_k_matches_dense_in_the_trivial_limit` — at
///   thresh -> 0 the screen does nothing and LinK reproduces the dense build to
///   machine precision. An invalid bound cannot pass that at ANY threshold.
/// * `link_scf_anchor` — SCF energies agree to 1.1e-11 (C8) with IDENTICAL
///   iteration counts, and `max|K_link - K_direct|` = 5.764e-12 at C16, both
///   far inside their own bars.
///
/// 1e-10 is ~8x above the largest measured value and an order of magnitude
/// below the smallest deviation any of the pre-#50 defects produced (1.7e-3),
/// so a genuine bound regression still fails here. Do NOT raise it further to
/// accommodate a future measurement: if this bar starts failing, the screen has
/// changed what it computes, and that is the thing this file exists to catch.
const K_DEV_BAR: f64 = 1e-10;

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
///
/// # Why 1.0 was too weak, and where 0.995 came from (MUTATION-DERIVED)
///
/// A bare `< 1.0` was measured to be nearly vacuous, by the M2 mutation
/// (`FourPairK` -> `SixPair`, i.e. LinK screening on the SAME six-pairing key
/// as `build_jk`). MEASURED at alkane_8:
///
/// ```text
///                     LinK count   / build_jk (8,107,064)   margin
/// M2  SixPair          8,106,960     0.999987               104 quartets
/// fix FourPairK        8,023,337     0.989672            83,727 quartets
/// ```
///
/// M2 PASSED a `< 1.0` bar — by 104 quartets out of 8.1 million (0.0013%),
/// which is just pair-list bookkeeping, not screening. That is the "LinK merely
/// TIES the builder it is supposed to beat" state the design note predicted and
/// the bar existed to reject, and it slipped through on luck.
///
/// 0.995 sits between the two measured regimes: it rejects the tie (0.999987)
/// and is cleared by the real fix at both sizes (0.9897 at C8, 0.9539 at C16).
/// It demands a >=0.5% quartet reduction — an order of magnitude above the
/// bookkeeping noise M2 exhibits, and comfortably below what the exchange-only
/// key actually delivers.
///
/// Note this is the one bar in this file that was TIGHTENED after measurement
/// rather than relaxed. Tightening onto a measured separation is legitimate;
/// the direction that would not be is loosening a bar to admit a result that
/// failed it.
const LINK_VS_BUILD_JK_MAX_RATIO: f64 = 0.995;

/// Minimum fraction of density pairs the dp list must PRUNE past the locality
/// onset.
///
/// **This bar is currently 0.0 — i.e. DISABLED — and that is a recorded
/// NEGATIVE RESULT, not an oversight.** Read this before setting it nonzero.
///
/// The ticket asked for the dp criterion to be made to prune. Working out what
/// the list actually bounds (see the long note on `DensityPairs::build`) says it
/// cannot be, at this granularity:
///
/// * `D[j,σ]` reaches K only through `(i j|σ l)`, bounded by
///   `Q(i,j)·Q(σ,l)·|D[j,σ]|`.
/// * LinK really does range `l` over all of `sp(σ)`, so `qmax(σ)` is the honest
///   bound on that side; tightening it makes the condition UNNECESSARY and
///   drops real contributions.
/// * The same is true of `qmax(j)` on the bra side. Replacing it with the
///   pair's own `Q(j,σ)` restores locality but over-tightens — `Q` decays like
///   a Gaussian overlap while `D` decays exponentially. That was the PRE-#50
///   criterion, and it cost `max|K_LinK - K_direct|` = 1.7e-3 on butane.
///
/// So the shipped `|D|·qmax(j)·qmax(σ) > t` is already the tightest bound
/// available from `(j,σ)` alone. It prunes nothing because both maxima are
/// realized by core s-shell pairs and their product is a molecule-wide
/// constant `≈ Qmax²` — measured by
/// `qmax_spread_is_small_enough_to_explain_the_vacuous_dp_list`, which FAILS if
/// that reasoning is wrong.
///
/// The locality the list is missing is a property of the (bra pair, ket pair)
/// COMBINATION, which does not exist until both are known — per quartet. That
/// is where it now lives (`DensityScreen::FourPairK`), and it is why the
/// `build_jk` count assertion, not this one, is what this file rests on.
///
/// Setting this above 0.0 requires either a new bound with a stated derivation,
/// or evidence that the spread measurement above overturns the analysis. Do NOT
/// raise it by raising the threshold: that trades correctness for counts and is
/// explicitly out of bounds.
const DP_MIN_PRUNED_FRACTION: f64 = 0.0;

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

    // The dp-pruning bar is 0.0 by default — see DP_MIN_PRUNED_FRACTION for the
    // derivation of why the pair list cannot prune at this granularity. A
    // `pruned >= 0.0` assertion would be unreachable-by-construction (a test
    // that cannot fail is an assumption, per the repo rules), so the measured
    // value is REPORTED unconditionally and only asserted when someone has
    // actually raised the bar on the strength of a new bound.
    let pruned = 1.0 - c.dp_frac;
    eprintln!(
        "  dp list pruned {:.2}% of ordered pairs (bar {:.2}%{})",
        100.0 * pruned,
        100.0 * DP_MIN_PRUNED_FRACTION,
        if DP_MIN_PRUNED_FRACTION <= 0.0 { ", disabled — see DP_MIN_PRUNED_FRACTION" } else { "" }
    );
    if require_dp_pruning && DP_MIN_PRUNED_FRACTION > 0.0 {
        assert!(
            pruned >= DP_MIN_PRUNED_FRACTION,
            "{stem}/{basis_name}: the density-pair list kept {:.4} of all ordered shell pairs, \
             pruning only {:.2}% (bar {:.2}%). Do NOT raise the threshold to force pruning — fix \
             what the criterion bounds.",
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

/// The load-bearing claim behind the density-pair analysis, measured rather
/// than asserted.
///
/// `DensityPairs::build`'s doc argues that `qmax(j)·qmax(σ)` is effectively a
/// molecule-wide constant `≈ Qmax²`, which is WHY the criterion degenerates to
/// `|D| > t/Qmax²` and prunes nothing. That argument is the entire basis for
/// concluding that the pair list cannot be tightened further and that the fix
/// belongs in the per-quartet screen instead — so it must not stay an
/// assumption.
///
/// This measures the spread of `qmax` across shells directly. If the ratio
/// `max_x qmax(x) / min_x qmax(x)` is small, the product really is
/// near-constant and the analysis holds. If it is large, the analysis is WRONG
/// and there is per-pair information in `qmax` that a better criterion could
/// exploit — in which case this test failing is the signal to go reopen (A)
/// rather than accept the negative.
///
/// Deliberately has no pass/fail bar on the "large" side beyond a generous
/// sanity ceiling: the point is to PRINT the number and to fail loudly only if
/// it contradicts the recorded reasoning. The value is reported so the negative
/// in `DensityPairs::build` can be cited with a measurement behind it.
#[test]
fn qmax_spread_is_small_enough_to_explain_the_vacuous_dp_list() {
    for (stem, basis_name) in [("alkane_8", "def2-svp"), ("alkane_16", "def2-svp")] {
        let mol = load_mol(stem);
        let bs: BasisSet = basis::bundled(basis_name).expect("basis");
        let prep = PreparedBasis::new(&mol, &bs).expect("PreparedBasis");
        let nsh = prep.nshells();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).expect("Schwarz bounds");

        // qmax(x) = max_y Q(x,y) — exactly what DensityPairs::build computes.
        let qmax: Vec<f64> = (0..nsh)
            .map(|x| (0..nsh).map(|y| bounds.q[(x, y)]).fold(0.0f64, f64::max))
            .collect();
        let hi = qmax.iter().cloned().fold(0.0f64, f64::max);
        let lo = qmax.iter().cloned().fold(f64::INFINITY, f64::min);
        let spread = hi / lo;

        eprintln!(
            "{stem}/{basis_name}: qmax spread = {hi:.4e} / {lo:.4e} = {spread:.2}x across {nsh} shells"
        );

        assert!(
            lo > 0.0,
            "{stem}: some shell has qmax = 0, which would make its density pairs unreachable at \
             every threshold (the invalid-bound failure mode from schwarz.rs)"
        );
        // A spread of a few orders of magnitude still leaves the PRODUCT of two
        // such factors within a couple of decades of Qmax^2, which against a
        // density tail that decays slowly is not enough to prune. If this ever
        // exceeds 1e6 the "near-constant" reading is untenable and the analysis
        // in DensityPairs::build must be revisited.
        assert!(
            spread < 1e6,
            "{stem}: qmax varies by {spread:.2e}x across shells. DensityPairs::build's doc claims \
             qmax(j)*qmax(sigma) is effectively a molecule-wide constant, which is the entire \
             basis for concluding the pair list cannot prune further. That claim does not survive \
             this spread — reopen the density-pair criterion."
        );
    }
}

/// THE anchor: past the ~30 Bohr locality onset, LinK must evaluate fewer
/// quartets than the default builder. The dp pruning fraction is measured and
/// printed here too (its bar is disabled — see `DP_MIN_PRUNED_FRACTION`).
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
/// per-quartet screen has strictly more far-field quartets available to reject:
/// the `LinK/build_jk` ratio should be no worse than at C16. This is the case
/// that would expose a screen whose advantage does not grow with system size —
/// i.e. one cutting a fixed fraction for a reason unrelated to locality, which
/// is the "too clean" fingerprint of arithmetic rather than physics.
///
/// It also prints the C20 dp fraction, which is the datum that would refute the
/// recorded negative if the dp list DOES start pruning once the molecule is
/// long enough — the one way the analysis on `DP_MIN_PRUNED_FRACTION` could be
/// wrong in the direction of being too pessimistic.
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
