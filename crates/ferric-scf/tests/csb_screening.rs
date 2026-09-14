//! Anchors for CSB — the **combined Schwarz bound**, Eq. (8) of Thompson &
//! Ochsenfeld, J. Chem. Phys. 147, 144101 (2017):
//!
//! ```text
//!   |(µν|λσ)|  ≤  min{ Q_µν Q_λσ ,  M_µλ M_νσ ,  M_µσ M_νλ } ,
//!   M_µλ = sqrt(|(µµ|λλ)|)
//! ```
//!
//! # Reading order (the exactness anchor comes FIRST, per CLAUDE.md)
//!
//! [`csb_is_a_valid_upper_bound`] is THE test. CSB's entire claim over the
//! non-rigorous CSAM family is that it never underestimates, so validity is
//! not one property among several — it is the whole thing. Everything else in
//! this file exists to stop that test from passing for the wrong reason.
//!
//! # What each test catches, and how each could FAIL
//!
//! Per "a test you have never seen failing is an assumption", each test below
//! records the concrete defect that would trip it. A test whose failure mode I
//! could not name was not written.
//!
//! | test | catches |
//! |---|---|
//! | `csb_is_a_valid_upper_bound` | any `M`-table defect that UNDERestimates: wrong sub-block index, a spurious floored/prescreened zero, an operator mismatch between `M` and `Q`, or the `min` selecting a term that is not actually a bound |
//! | `csb_is_never_looser_than_schwarz` | a wiring error that drops the Schwarz seed from the `min` chain (structurally impossible today — asserted so it STAYS impossible) |
//! | `csb_is_strictly_tighter_somewhere` | an INERT bound: a `min` that never selects an `M` term buys nothing, and would make the other tests pass vacuously |
//! | `csb_trivial_limit_matches_schwarz_pair_list` | the exactness anchor at threshold 0: a bound that spuriously zeroes an entry makes a pair unreachable at EVERY threshold, including 0 |
//! | `csb_hot_loop_matches_the_bound_trait` | drift between the two implementations of Eq. (8) (the `Bound` impl and `scatter_bra_pair`'s hoisted row slices) |
//! | `csb_view_agrees_with_csb_bounds_and_degrades_to_schwarz` | the third implementation, `CsbView` (open-shell LinK), drifting — AND the degenerate case where a `CsbView` over table-less bounds is not exactly plain Schwarz |
//! | `csb_rhf_energy_matches_schwarz` | any end-to-end wiring defect that changes a converged energy |
//! | `csb_rhf_is_thread_count_bit_identical` | a non-deterministic reduction introduced by the extra screen |
//!
//! # The 218,493× case is deliberately included
//!
//! `benzene/cc-pVDZ` under `Operator::erfc(1.0)` is exactly where the sibling
//! CSAM branch measured a 218,493× bound violation. It is included in the
//! validity sweep NOT as a stress test but as the discriminating arm: that
//! failure came from a normalized ratio collapsing to ~0 under aggressive
//! attenuation, which is precisely the regime where a floored or prescreened
//! `M` entry would do the same thing to CSB's `min`. If `csb_m_table`'s
//! precision/floor discipline were ever dropped, THIS is the arm that fires.
//!
//! If it fires, the correct response is to report a real bug in the `M` table,
//! NOT to loosen the assertion. CSB is a plain `min` of three Cauchy-Schwarz
//! upper bounds; there is no legitimate reason for it to be violated.

use ferric_core::basis;
use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::operator::Operator;
use ferric_scf::rhf::{solve_rhf, RhfConfig};
use ferric_scf::screening::{Bound, CsbBounds, CsbView, ScreeningKind, SchwarzBounds};

fn prep_for(path: &str, basis: &str) -> (Molecule, PreparedBasis) {
    let mol = Molecule::load_xyz(path).unwrap();
    let bs = basis::bundled(basis).unwrap();
    let prep = PreparedBasis::new(&mol, &bs).unwrap();
    (mol, prep)
}

/// Worst `|true integral| / CSB estimate` over every canonical shell quartet
/// of a system, plus where it occurred and how many quartets were sampled.
///
/// The true value is `max |(µν|λσ)|` over the functions of the quartet, which
/// is exactly the quantity the shell-granular bound must dominate (see
/// `csb_m_table`'s doc for the max-of-independent-maxima lift).
///
/// The reference engine is built at `1e-30` — MUCH tighter than the tables
/// under test — so libint2 cannot prescreen a small-but-real quartet to
/// nothing and hand back a spuriously satisfied bound. A reference computed at
/// the same precision as the table would share the table's defect and this
/// whole file would prove nothing.
fn worst_violation(prep: &PreparedBasis, csb: &CsbBounds, op: Operator) -> (f64, (usize, usize, usize, usize), usize) {
    let nsh = prep.nshells();
    let mut eng = Engine::new_2e(op, prep, 1e-30).unwrap();
    let mut worst = 0.0f64;
    let mut worst_at = (0, 0, 0, 0);
    let mut sampled = 0usize;
    for i in 0..nsh {
        for j in 0..=i {
            for k in 0..=i {
                let lmax = if k == i { j } else { k };
                for l in 0..=lmax {
                    let bound = csb.estimate(i, j, k, l);
                    let tru = match eng.compute_quartet(prep, i, j, k, l) {
                        Some(block) => block.iter().fold(0.0f64, |m, v| m.max(v.abs())),
                        None => 0.0,
                    };
                    sampled += 1;
                    if bound > 0.0 {
                        let ratio = tru / bound;
                        if ratio > worst {
                            worst = ratio;
                            worst_at = (i, j, k, l);
                        }
                    } else if tru > 0.0 {
                        // A zero bound on a nonzero integral is an infinite
                        // violation; report it as such rather than dividing.
                        worst = f64::INFINITY;
                        worst_at = (i, j, k, l);
                    }
                }
            }
        }
    }
    (worst, worst_at, sampled)
}

/// (a) **THE validity test.** `csb_estimate >= |true (pq|rs)|`, strictly, on
/// every canonical shell quartet of three real systems under three operators.
///
/// This is the whole claim. Unlike QQR (whose analogous test has to allow the
/// Schwarz table's own ULP-scale slack, because QQR multiplies a MODEL envelope
/// onto it), CSB is a `min` of three exact Cauchy-Schwarz upper bounds, so it
/// has no modelling error of its own to excuse. The only tolerance granted is
/// a relative `1e-12` for floating-point evaluation of the products — nowhere
/// near enough to hide a real defect, which the CSAM experience says arrives at
/// 2x-2e5x scale, not at 1 ULP.
///
/// # Sampling
///
/// Every canonical `i>=j, k<=i, l<=lmax` quartet of each system — not a random
/// subsample — so the "worst case" reported is the true worst case over the
/// screened population, not an estimate of it. At benzene/cc-pVDZ that is
/// ~1.2M quartets; at water/cc-pVDZ ~2.2k.
///
/// # Systems and why each is here
///
/// * **water/cc-pVDZ** — fast, dense, everything overlaps; catches gross
///   indexing errors immediately.
/// * **benzene/cc-pVDZ** — extended enough (~9.4 Bohr) that shell pairs
///   genuinely separate, so the `M` cross-terms are actually engaged rather
///   than always losing the `min` to Schwarz.
/// * **benzene/cc-pVDZ + `erfc(1.0)`** — THE discriminating arm; see this
///   file's header. This is the exact (system, basis, operator) triple on
///   which CSAM was measured violating by 218,493×.
///
/// # If this fails
///
/// Report it. Do not loosen the assertion, do not add a safety factor, and do
/// not restrict the operator set to make it pass. A violation means the `M`
/// table is wrong (most likely a floored/prescreened zero, or the wrong
/// sub-block of `(PP|QQ)`), and a safety factor applied to a `min` of exact
/// bounds would be covering up a bug rather than accounting for a model error.
#[test]
fn csb_is_a_valid_upper_bound() {
    let cases: &[(&str, &str, Operator)] = &[
        ("../../testdata/molecules/water.xyz", "cc-pvdz", Operator::coulomb()),
        ("../../testdata/molecules/water.xyz", "cc-pvdz", Operator::erfc(1.0)),
        ("../../testdata/molecules/water.xyz", "cc-pvdz", Operator::erf(1.0)),
        ("../../testdata/molecules/benzene.xyz", "cc-pvdz", Operator::coulomb()),
        // (e) THE 218,493x arm.
        ("../../testdata/molecules/benzene.xyz", "cc-pvdz", Operator::erfc(1.0)),
        ("../../testdata/molecules/benzene.xyz", "cc-pvdz", Operator::erf(1.0)),
    ];
    for &(path, bas, op) in cases {
        let (_mol, prep) = prep_for(path, bas);
        let csb = CsbBounds::compute(op, &prep).unwrap();
        let (worst, at, sampled) = worst_violation(&prep, &csb, op);
        eprintln!(
            "{path}/{bas} {:?}: worst |true|/CSB = {worst:.6e} at {at:?} over {sampled} quartets",
            op.kind
        );
        assert!(
            worst <= 1.0 + 1e-12,
            "CSB is NOT an upper bound on {path}/{bas} under {:?}: worst |true|/CSB = {worst:.6e} \
             at quartet {at:?} (over {sampled} sampled). CSB is a min of three exact \
             Cauchy-Schwarz bounds, so a violation is a real defect in the M table — most likely \
             a floored/prescreened zero winning the min, or the wrong sub-block of (PP|QQ). Fix \
             the table; do NOT loosen this assertion.",
            op.kind
        );
        // REACHABILITY guard: if the bound were merely enormous everywhere,
        // the assertion above would pass while proving nothing. Cauchy-Schwarz
        // is an EQUALITY on diagonal quartets `(ij|ij)`, which the sweep
        // always includes, so the worst ratio must come close to 1.
        assert!(
            worst > 0.5,
            "worst ratio {worst:.6e} is suspiciously loose on {path}/{bas} under {:?} — the \
             bound may be vacuously large rather than genuinely tight (Cauchy-Schwarz is an \
             equality on the (ij|ij) quartets this sweep includes, so this should approach 1)",
            op.kind
        );
    }
}

/// (b) **Never looser than Schwarz**, at every quartet and every operator.
///
/// Guaranteed by the `.min()` seeded with `schwarz_est` in
/// `CsbBounds::estimate` — which is exactly why it is asserted. The test does
/// not catch an `M`-table bug (it cannot: an `M` bug makes CSB tighter, and
/// tighter passes here). It catches a WIRING error: someone reordering the
/// `min` chain so the Schwarz term is no longer the seed, or replacing it with
/// an `M` product. That would silently remove the one-sided safety guarantee
/// the whole design rests on.
#[test]
fn csb_is_never_looser_than_schwarz() {
    for (path, bas) in [
        ("../../testdata/molecules/water.xyz", "cc-pvdz"),
        ("../../testdata/molecules/benzene.xyz", "sto-3g"),
    ] {
        for op in [Operator::coulomb(), Operator::erfc(1.0), Operator::erf(1.0)] {
            let (_mol, prep) = prep_for(path, bas);
            let csb = CsbBounds::compute(op, &prep).unwrap();
            let nsh = prep.nshells();
            for i in 0..nsh {
                for j in 0..=i {
                    for k in 0..=i {
                        let lmax = if k == i { j } else { k };
                        for l in 0..=lmax {
                            let s = csb.schwarz().estimate(i, j, k, l);
                            let c = csb.estimate(i, j, k, l);
                            assert!(
                                c <= s,
                                "{path}/{bas} {:?}: CSB({i},{j},{k},{l}) = {c:.6e} > Schwarz = \
                                 {s:.6e}. The .min() in CsbBounds::estimate must be SEEDED with \
                                 the Schwarz product; if it is not, the 'can only ever tighten' \
                                 safety property is gone.",
                                op.kind
                            );
                        }
                    }
                }
            }
        }
    }
}

/// (c) **Strictly tighter somewhere** — and quantified, not merely existential.
///
/// A `min` that never selects an `M` term is an inert bound: valid, never
/// looser, and worth exactly nothing. Requiring only "at least one quartet is
/// tighter" would be satisfied by a single accidental tie, so this measures
/// the FRACTION of quartets on which CSB strictly improves and requires it to
/// be substantial.
///
/// # Where CSB is STRUCTURALLY inert, and why that is not a defect
///
/// On a DIAGONAL quartet `(ij|ij)` the `min` must select Schwarz, always, for
/// every operator. Cauchy-Schwarz is an EQUALITY there — `Q(i,j)² =
/// |(ij|ij)|` exactly, which the paper states directly ("the CSB estimates
/// are exact for both (µν|µν)-type and (µµ|νν)-type integrals ... the QQ
/// estimates are exact for the former") — so no other valid upper bound can
/// be smaller. That is the same structural fact `tests/qqr_diagonal_noop.rs`
/// records for QQR, and it has a consequence worth stating explicitly:
/// **CSB cannot shrink `SignificantPairs`/`DensityPairs`**, both of which
/// screen on `estimate(i,j,i,j)`. CSB acts only on the innermost per-quartet
/// test. So the population this test measures is the right one, and a "why
/// didn't the pair lists shrink" question later has its answer here.
///
/// # The bars, stated BEFORE running (repo rule)
///
/// * under **`erfc(1.0)`** the `M` terms carry the kernel's exponential
///   distance decay while the `Q` factors carry overlap; the paper reports CSB
///   "provides a much tighter bound" for exactly this kernel. Bar: improvement
///   on at least 5% of quartets. This bar is MY choice, not a figure from the
///   paper — its job is to be reachable if the mechanism works and unreachable
///   if the `M` terms never win the `min`. If it fails, the first thing to
///   check is whether the failure is "0.0%" (inert bound, a wiring bug) or
///   "3%" (bar set too aggressively on this particular system); only the
///   former is a defect.
/// * under **Coulomb** the paper says outright that "the CSB estimate is no
///   more useful than the QQ estimate for currently tractable systems" (its
///   Table I reports F_min = 1.000 for CSB under `1/r12`). So the Coulomb arm
///   asserts only that improvement EXISTS, with the fraction printed. **A
///   small Coulomb number here is the literature's prediction being
///   confirmed, not a defect** — and ferric's motivating benzene/aug-cc-pVDZ
///   profile is a Coulomb workload, so this is the arm that predicts how much
///   that particular benchmark can gain.
///
/// # Artifact hypothesis vs physics hypothesis
///
/// If CSB is real, I expect erfc ≫ Coulomb in improved fraction (the paper's
/// central claim). If the `M` table is broken by collapsing to near-zero, I
/// expect BOTH to show near-100% improvement AND
/// `csb_is_a_valid_upper_bound` to fail — the two are distinguishable, which
/// is what makes this experiment able to tell them apart.
#[test]
fn csb_is_strictly_tighter_somewhere() {
    let (_mol, prep) = prep_for("../../testdata/molecules/benzene.xyz", "cc-pvdz");
    let nsh = prep.nshells();
    for (op, min_fraction) in [
        (Operator::erfc(1.0), 0.05),
        (Operator::coulomb(), 0.0),
    ] {
        let csb = CsbBounds::compute(op, &prep).unwrap();
        let mut improved = 0usize;
        let mut total = 0usize;
        let mut best_ratio = 1.0f64;
        for i in 0..nsh {
            for j in 0..=i {
                for k in 0..=i {
                    let lmax = if k == i { j } else { k };
                    for l in 0..=lmax {
                        let s = csb.schwarz().estimate(i, j, k, l);
                        let c = csb.estimate(i, j, k, l);
                        total += 1;
                        if s > 0.0 && c / s < 0.99 {
                            improved += 1;
                            best_ratio = best_ratio.min(c / s);
                        }
                    }
                }
            }
        }
        let frac = improved as f64 / total as f64;
        eprintln!(
            "benzene/cc-pVDZ {:?}: CSB strictly tighter on {improved}/{total} = {:.2}% of \
             quartets; best CSB/Schwarz = {best_ratio:.3e}",
            op.kind,
            100.0 * frac
        );
        assert!(
            improved > 0,
            "CSB is INERT under {:?}: it never selected an M term on any of {total} quartets, so \
             it is plain Schwarz wearing a second table. Check that csb_m_table is being built \
             and that CsbBounds::estimate actually consults it.",
            op.kind
        );
        assert!(
            frac >= min_fraction,
            "CSB improved only {:.4}% of quartets under {:?}, below the {:.1}% bar. For the \
             attenuated kernel this bar encodes the paper's claim that CSB captures distance \
             decay the Q factors miss; falling short means the M terms are rarely winning the \
             min, which points at the table rather than at the physics.",
            100.0 * frac,
            op.kind,
            100.0 * min_fraction
        );
    }
}

/// (d) **Trivial limit / exactness anchor.** At threshold 0 the surviving
/// significant-pair list under CSB is EXACTLY Schwarz's.
///
/// This is the repo's mandated "does nothing in the trivial limit" anchor, and
/// it is discriminating for a specific, previously-observed failure:
/// `SignificantPairs::build` compares with a strict `estimate(..) > threshold`,
/// so any bound that stores an exact `0.0` makes that pair unreachable at EVERY
/// threshold INCLUDING 0. That is precisely the defect
/// `SCHWARZ_TABLE_PRECISION` was introduced to fix on the `Q` table (1049
/// spurious zeros of 5253 pairs at alkane_8/cc-pVDZ), and the `M` table faces
/// the identical hazard — made WORSE by the `min`, where a zero wins.
///
/// Run under `erfc(1.0)` as well as Coulomb: attenuation pushes `M` entries
/// toward the engine's precision cliff, so a lost floor shows up there first.
///
/// Note the pair list is built on DIAGONAL quartets `estimate(i,j,i,j)`, where
/// CSB and Schwarz are expected to agree closely anyway — so this test is
/// specifically a zero-detector, not a tightness comparison. Its value is
/// exactly that: it is the one place a spurious zero cannot hide behind a
/// merely-tighter bound.
#[test]
fn csb_trivial_limit_matches_schwarz_pair_list() {
    for (path, bas) in [
        ("../../testdata/molecules/water.xyz", "cc-pvdz"),
        // alkane_8 is where the Q-table zeros were originally characterised;
        // water alone would pass while proving nothing (its shells are all
        // close enough that nothing approaches the precision cliff).
        ("../../testdata/molecules/alkane_8.xyz", "cc-pvdz"),
    ] {
        for op in [Operator::coulomb(), Operator::erfc(1.0)] {
            let (_mol, prep) = prep_for(path, bas);
            let nsh = prep.nshells();
            let schwarz = SchwarzBounds::compute(op, &prep).unwrap();
            let csb = CsbBounds::compute(op, &prep).unwrap();

            let mut n_pairs = 0usize;
            for i in 0..nsh {
                for j in 0..nsh {
                    let s_keeps = schwarz.estimate(i, j, i, j) > 0.0;
                    let c_keeps = csb.estimate(i, j, i, j) > 0.0;
                    if s_keeps {
                        n_pairs += 1;
                    }
                    assert_eq!(
                        s_keeps, c_keeps,
                        "{path}/{bas} {:?}: at threshold 0 the CSB and Schwarz pair lists differ \
                         at ({i},{j}) — Schwarz keeps={s_keeps}, CSB keeps={c_keeps}. A CSB \
                         estimate of exactly 0 makes this pair unreachable at EVERY threshold, \
                         destroying the trivial limit. Check csb_m_table's SCHWARZ_Q_FLOOR.",
                        op.kind
                    );
                }
            }
            // Reachable-pass check: the comparison is only meaningful if
            // Schwarz kept pairs at all.
            assert_eq!(
                n_pairs,
                nsh * nsh,
                "{path}/{bas} {:?}: Schwarz itself dropped pairs at threshold 0 — the baseline \
                 for this comparison is broken, so the equality above proves nothing",
                op.kind
            );
        }
    }
}

/// (e-bis) **Per-operator table independence.** A CSB built with one operator
/// must NOT be reusable under another: the `M` table has to carry the same
/// kernel as the `Q` table.
///
/// ferric has a scar here — `qqr.rs` once multiplied in an `exp(-ω²R²)` factor
/// that double-counted attenuation already living in the erfc Schwarz factors,
/// invalidating the bound at EVERY sampled quartet (worst |true|/bound 6.0e11).
/// CSB carries no separate geometric factor, so the check here is the positive
/// one: the erfc `M` table must be strictly smaller than the Coulomb one
/// somewhere (attenuation IS entering through `(µµ|λλ)`), and never larger
/// anywhere (`erfc(ωr)/r ≤ 1/r` pointwise, so every `(µµ|λλ)_erfc` is bounded
/// by its Coulomb counterpart).
///
/// A build that silently reused a Coulomb table under erfc would fail the
/// "strictly smaller somewhere" half.
#[test]
fn csb_m_table_is_operator_specific() {
    let (_mol, prep) = prep_for("../../testdata/molecules/benzene.xyz", "sto-3g");
    let nsh = prep.nshells();
    let c = CsbBounds::compute(Operator::coulomb(), &prep).unwrap();
    let e = CsbBounds::compute(Operator::erfc(1.0), &prep).unwrap();
    let mut found_smaller = false;
    for i in 0..nsh {
        for j in 0..nsh {
            let mc = c.m()[(i, j)];
            let me = e.m()[(i, j)];
            assert!(
                me <= mc * (1.0 + 1e-12),
                "M_erfc[{i},{j}] = {me:.6e} exceeds M_coulomb = {mc:.6e}, but \
                 erfc(wr)/r <= 1/r pointwise so every (uu|ll)_erfc is bounded by its Coulomb \
                 counterpart — this indicates the wrong operator reached one of the tables"
            );
            if mc > 1e-10 && me / mc < 0.99 {
                found_smaller = true;
            }
        }
    }
    assert!(
        found_smaller,
        "the erfc M table is nowhere strictly smaller than the Coulomb one — attenuation is not \
         entering through the (uu|ll) integrals, which means csb_m_table is (re)using a Coulomb \
         table under erfc"
    );
}

/// **The two implementations of Eq. (8) must agree.**
///
/// CSB exists twice on purpose: as `CsbBounds::estimate` (the `Bound` trait
/// face, used by LinK and by every test in this file) and inline in
/// `quartet_scatter::scatter_bra_pair` (the hot loop, which hoists `M[s1,:]`
/// and `M[s2,:]` into row slices once per bra pair to avoid a vtable call and
/// strided lookups in a ~46M-candidate inner loop).
///
/// Two copies of a formula drift. This reproduces the hot loop's arithmetic
/// EXACTLY — same row-slice indexing, same `min` order — and requires bitwise
/// equality with the trait method, so a change to either without the other
/// fails here rather than silently producing a differently-screened Fock
/// matrix.
#[test]
fn csb_hot_loop_matches_the_bound_trait() {
    let (_mol, prep) = prep_for("../../testdata/molecules/benzene.xyz", "sto-3g");
    let nsh = prep.nshells();
    for op in [Operator::coulomb(), Operator::erfc(1.0)] {
        let csb = CsbBounds::compute(op, &prep).unwrap();
        let m = csb.m();
        let q = &csb.schwarz().q;
        for s1 in 0..nsh {
            for s2 in 0..=s1 {
                // Verbatim transcription of scatter_bra_pair's hoist,
                // including the `to_slice()` step — so this checks the SLICE
                // path the hot loop actually takes, not an `ArrayView` that
                // merely happens to hold the same numbers. A row that stopped
                // being contiguous would fail here the same way it fails
                // there.
                let b12 = q[(s1, s2)];
                let r1 = m.row(s1);
                let r2 = m.row(s2);
                let mr1 = r1.to_slice().expect("M row(s1) must be contiguous");
                let mr2 = r2.to_slice().expect("M row(s2) must be contiguous");
                for s3 in 0..=s1 {
                    let s4max = if s3 == s1 { s2 } else { s3 };
                    for s4 in 0..=s4max {
                        let b34 = q[(s3, s4)];
                        let mut hot = b12 * b34;
                        let eq6 = mr1[s3] * mr2[s4];
                        let eq7 = mr1[s4] * mr2[s3];
                        hot = hot.min(eq6).min(eq7);
                        let trait_est = csb.estimate(s1, s2, s3, s4);
                        assert_eq!(
                            hot.to_bits(),
                            trait_est.to_bits(),
                            "{:?}: hot-loop CSB at ({s1},{s2},{s3},{s4}) = {hot:.17e} differs \
                             bitwise from CsbBounds::estimate = {trait_est:.17e}. The two \
                             implementations of Eq. (8) have drifted.",
                            op.kind
                        );
                    }
                }
            }
        }
    }
}

/// **The THIRD implementation of Eq. (8) must also agree** — and must degrade
/// to exactly plain Schwarz when there is no `M` table.
///
/// [`CsbView`] is how `[scf] screening = "csb"` reaches open-shell LinK
/// (`solve_uhf`/`solve_rohf` hand LinK the caller's `&SchwarzBounds`, and have
/// no config-reading construction site of their own). It therefore evaluates
/// Eq. (8) for a third time, from a `&SchwarzBounds` rather than from a
/// `CsbBounds`, and must agree bitwise with both other implementations.
///
/// The second half is the one that would actually catch a regression: a
/// `CsbView` over table-less bounds must be BITWISE identical to plain
/// Schwarz, because every default (non-CSB) open-shell LinK run now goes
/// through it. If that ever stopped holding, `screening = "schwarz"` would
/// silently stop being byte-identical to a pre-CSB build for UHF/ROHF — the
/// exact invariant the whole opt-in design exists to preserve.
#[test]
fn csb_view_agrees_with_csb_bounds_and_degrades_to_schwarz() {
    let (_mol, prep) = prep_for("../../testdata/molecules/benzene.xyz", "sto-3g");
    let nsh = prep.nshells();
    for op in [Operator::coulomb(), Operator::erfc(1.0)] {
        let plain = SchwarzBounds::compute_for_screening(op, &prep, ScreeningKind::Schwarz)
            .unwrap();
        let withm =
            SchwarzBounds::compute_for_screening(op, &prep, ScreeningKind::Csb).unwrap();
        let csb = CsbBounds::compute(op, &prep).unwrap();

        let v_plain = CsbView::new(&plain);
        let v_csb = CsbView::new(&withm);
        assert!(!v_plain.is_csb(), "a view over table-less bounds must NOT be applying CSB");
        assert!(v_csb.is_csb(), "a view over Csb-built bounds must be applying CSB");

        for i in 0..nsh {
            for j in 0..=i {
                for k in 0..=i {
                    let lmax = if k == i { j } else { k };
                    for l in 0..=lmax {
                        // Half 1: the CSB view matches the CsbBounds impl.
                        let a = v_csb.estimate(i, j, k, l);
                        let b = csb.estimate(i, j, k, l);
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "{:?}: CsbView at ({i},{j},{k},{l}) = {a:.17e} differs bitwise from \
                             CsbBounds = {b:.17e} — the third implementation of Eq. (8) has \
                             drifted",
                            op.kind
                        );
                        // Half 2: the table-less view IS plain Schwarz, bitwise.
                        let p = v_plain.estimate(i, j, k, l);
                        let s = plain.estimate(i, j, k, l);
                        assert_eq!(
                            p.to_bits(),
                            s.to_bits(),
                            "{:?}: a CsbView over table-less bounds is not bitwise plain Schwarz \
                             at ({i},{j},{k},{l}): {p:.17e} vs {s:.17e}. Every DEFAULT \
                             open-shell LinK run goes through this path, so this breaks the \
                             byte-identity guarantee for `screening = \"schwarz\"`.",
                            op.kind
                        );
                    }
                }
            }
        }
    }
}

/// (f) **SCF energy invariance.** RHF with `screening = "csb"` must reproduce
/// `"schwarz"` far more tightly than convergence.
///
/// The bar here is deliberately MUCH tighter than the CSAM sibling's would be,
/// and the reason is the point of this whole branch: CSAM is an ESTIMATE whose
/// error is traded against the threshold, so the honest bar there is "error
/// shrinks as the threshold tightens". CSB is a rigorous bound, so it discards
/// only quartets that plain Schwarz would ALSO have discarded had its bound
/// been as tight — nothing real is lost, and the energies must agree to the
/// SCF's own convergence floor, not merely to a threshold-controlled error.
///
/// A CSB energy that differed at, say, 1e-6 Ha would mean the bound is
/// discarding significant integrals, i.e. it is not a bound — which
/// `csb_is_a_valid_upper_bound` should already have caught. This test is the
/// end-to-end confirmation that the wiring carries the same property the unit
/// tests establish for the bound itself.
///
/// Both arms run at a tight `integral_thresh` and tight convergence so the
/// residual difference is the screen's, not DIIS noise's.
#[test]
fn csb_rhf_energy_matches_schwarz() {
    let ctx = ParallelContext::default();
    let (mol, prep) = prep_for("../../testdata/molecules/water.xyz", "cc-pvdz");
    let op = Operator::coulomb();
    let cfg = RhfConfig {
        energy_conv: 1e-12,
        density_conv: 1e-9,
        integral_thresh: 1e-12,
        ..Default::default()
    };

    let b_schwarz =
        SchwarzBounds::compute_for_screening(op, &prep, ScreeningKind::Schwarz).unwrap();
    let b_csb = SchwarzBounds::compute_for_screening(op, &prep, ScreeningKind::Csb).unwrap();
    assert!(
        b_schwarz.csb_m.is_none(),
        "ScreeningKind::Schwarz must leave csb_m as None — otherwise the default path is not \
         byte-identical to a pre-CSB build"
    );
    assert!(
        b_csb.csb_m.is_some(),
        "ScreeningKind::Csb must attach the M table, or this test compares Schwarz with itself \
         and proves nothing"
    );

    let e_schwarz = solve_rhf(&ctx, &mol, &prep, op, &b_schwarz, &cfg).unwrap();
    let e_csb = solve_rhf(&ctx, &mol, &prep, op, &b_csb, &cfg).unwrap();
    let diff = (e_schwarz.energy - e_csb.energy).abs();
    eprintln!(
        "water/cc-pVDZ RHF: schwarz {:.12} vs csb {:.12} (diff {diff:.3e}); quartets {} vs {}",
        e_schwarz.energy, e_csb.energy, e_schwarz.computed_quartets, e_csb.computed_quartets
    );
    assert!(
        diff < 1e-9,
        "CSB changed the converged RHF energy by {diff:.3e} Ha (schwarz {:.12}, csb {:.12}). A \
         RIGOROUS bound discards nothing real, so any difference above the SCF convergence floor \
         means CSB is underestimating somewhere — see csb_is_a_valid_upper_bound.",
        e_schwarz.energy,
        e_csb.energy
    );
    // Reachable-pass check: if CSB screened exactly as much as Schwarz, the
    // agreement above would be trivially guaranteed and would test no wiring
    // at all. Under COULOMB the paper predicts little to no difference, so
    // this is reported rather than asserted — see
    // `csb_is_strictly_tighter_somewhere` for where the discriminating
    // assertion lives.
    if e_csb.computed_quartets == e_schwarz.computed_quartets {
        eprintln!(
            "[note] CSB computed the same quartet count as Schwarz on this Coulomb job. That is \
             the literature's prediction for 1/r12 (Thompson & Ochsenfeld report F_min = 1.000 \
             for CSB under the Coulomb operator), not evidence the wiring is inert — \
             csb_is_strictly_tighter_somewhere asserts the bound is genuinely tighter."
        );
    }
}

/// (g) **Thread-count bit-identity is preserved with CSB active.**
///
/// The screen decides WHICH quartets are computed, and the reduction decides in
/// what order their contributions are summed. Both must be independent of the
/// worker count, or an energy becomes irreproducible. The existing direct/LinK
/// reductions are deterministic by construction (`reduce.rs`'s grouped sum);
/// this confirms that adding a second table to the per-quartet predicate did
/// not introduce any thread-dependence — e.g. by making a bound value depend on
/// which worker hoisted which row slice.
///
/// `RAYON_NUM_THREADS` is set per-process, so this runs both arms via a
/// rayon thread pool of an explicitly pinned size rather than by mutating the
/// environment (which would race other tests in the same binary — see the
/// "memory tests must pin worker count" note in MEMORY.md for why an ambient
/// pool has hidden defects here three times).
#[test]
fn csb_rhf_is_thread_count_bit_identical() {
    let (mol, prep) = prep_for("../../testdata/molecules/water.xyz", "cc-pvdz");
    let op = Operator::coulomb();
    let cfg = RhfConfig {
        energy_conv: 1e-12,
        density_conv: 1e-9,
        integral_thresh: 1e-12,
        ..Default::default()
    };
    let bounds = SchwarzBounds::compute_for_screening(op, &prep, ScreeningKind::Csb).unwrap();
    assert!(bounds.csb_m.is_some(), "CSB table missing — this test would be vacuous");

    let run = |threads: usize| -> (u64, usize) {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let ctx = ParallelContext::default();
            let r = solve_rhf(&ctx, &mol, &prep, op, &bounds, &cfg).unwrap();
            (r.energy.to_bits(), r.computed_quartets)
        })
    };

    let (e1, q1) = run(1);
    let (e4, q4) = run(4);
    assert_eq!(
        e1, e4,
        "CSB RHF energy is not bit-identical across thread counts: 1 thread \
         {:.17e} vs 4 threads {:.17e}",
        f64::from_bits(e1),
        f64::from_bits(e4)
    );
    assert_eq!(
        q1, q4,
        "CSB screened a different number of quartets at 1 thread ({q1}) than at 4 ({q4}) — the \
         per-quartet predicate has become thread-dependent"
    );
}
