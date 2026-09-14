//! Schwarz integral screening for efficient Fock matrix construction.

use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::csb;
use ferric_integrals::operator::Operator;
use ferric_integrals::schwarz;
use ndarray::Array2;

/// Generic trait for shell-quartet integral upper bounds.
///
/// Both Schwarz and QQR implement this, so LinK can be agnostic to the bound type.
///
/// `Sync + Send` supertraits: both implementors (`SchwarzBounds`, `QqrBounds`)
/// are plain read-only data (no interior mutability), and `DensityPairs::build`
/// (pairs.rs) shares `&dyn Bound` across rayon workers — the trait object needs
/// `Sync` for that `&` to cross threads.
pub trait Bound: Sync + Send {
    /// Upper bound estimate for |(sh1 sh2 | sh3 sh4)|.
    fn estimate(&self, sh1: usize, sh2: usize, sh3: usize, sh4: usize) -> f64;
}

/// Precomputed Schwarz screening bounds for shell-pair integrals.
///
/// Used to skip negligible shell quartets during Fock matrix construction:
/// if Q(s1,s2) * Q(s3,s4) * max|D| < threshold, the quartet is skipped.
///
/// Optionally carries the CSB `M` table (see [`SchwarzBounds::csb_m`]), which
/// upgrades the screen in [`crate::quartet_scatter::scatter_bra_pair`] from
/// plain Schwarz to Eq. (8) without changing any builder's signature.
#[derive(Debug, Clone)]
#[must_use = "Schwarz bounds are expensive to compute; dropping them wastes work"]
pub struct SchwarzBounds {
    pub q: Array2<f64>,
    pub q_shell: Vec<f64>,
    pub op: Operator,
    pub nshells: usize,
    /// CSB `M` table, `M[P][Q] = sqrt(max_{µ∈P,λ∈Q} |(µµ|λλ)|)`, built for the
    /// SAME `op` as `q`. `None` (the default, and what [`Self::compute`]
    /// always produces) leaves every consumer on plain Schwarz, byte-identical
    /// to every pre-CSB build. `Some` is produced only by
    /// [`Self::compute_for_screening`] with [`ScreeningKind::Csb`].
    ///
    /// This lives on `SchwarzBounds` rather than being a separate type at the
    /// builder boundary for one reason: `DirectJ`/`DirectK`/`DirectJK` and
    /// `build_jk_with_pool` all take a concrete `&SchwarzBounds`, and
    /// `solve_rhf`/`solve_uhf`/`solve_rohf` all receive one from the caller.
    /// Attaching the table to that single VALUE makes one `[scf] screening`
    /// choice govern every direct builder without touching ~40 signatures.
    /// [`CsbBounds`] remains the `Bound`-trait face of the same math, for the
    /// generic LinK path and for the tests that check the bound itself.
    ///
    /// Both representations compute the identical formula; see
    /// `tests/csb_screening.rs::csb_hot_loop_matches_the_bound_trait` for the
    /// assertion that keeps them from drifting.
    pub csb_m: Option<Array2<f64>>,
}

impl SchwarzBounds {
    /// Compute Schwarz screening bounds for all shell pairs.
    ///
    /// Always leaves `csb_m` as `None`, i.e. plain Schwarz. Use
    /// [`Self::compute_for_screening`] to opt into CSB.
    pub fn compute(op: Operator, prep: &PreparedBasis) -> Result<Self, FerricError> {
        let q = schwarz::schwarz(op, prep)?;
        let nsh = prep.nshells();
        let mut q_shell = vec![0.0; nsh];
        for i in 0..nsh {
            let mut max_val = 0.0f64;
            for j in 0..nsh {
                max_val = max_val.max(q[(i, j)]);
            }
            q_shell[i] = max_val;
        }
        Ok(SchwarzBounds {
            q,
            q_shell,
            op,
            nshells: nsh,
            csb_m: None,
        })
    }

    /// Compute Schwarz bounds, additionally building the CSB `M` table when
    /// `kind == ScreeningKind::Csb`.
    ///
    /// `ScreeningKind::Schwarz` delegates verbatim to [`Self::compute`] and is
    /// therefore byte-identical to it — there is no separate code path that
    /// could drift.
    ///
    /// The `M` table is built from the SAME `op` passed here, so the
    /// per-operator contract cannot be violated by a caller: there is no way
    /// to supply two operators.
    pub fn compute_for_screening(
        op: Operator,
        prep: &PreparedBasis,
        kind: ScreeningKind,
    ) -> Result<Self, FerricError> {
        let mut bounds = Self::compute(op, prep)?;
        if kind == ScreeningKind::Csb {
            bounds.csb_m = Some(csb::csb_m_table(op, prep)?);
        }
        Ok(bounds)
    }

    /// Upper bound estimate for the shell quartet (sh1 sh2 | sh3 sh4).
    pub fn estimate(&self, sh1: usize, sh2: usize, sh3: usize, sh4: usize) -> f64 {
        self.q[(sh1, sh2)] * self.q[(sh3, sh4)]
    }
}

impl Bound for SchwarzBounds {
    fn estimate(&self, sh1: usize, sh2: usize, sh3: usize, sh4: usize) -> f64 {
        self.q[(sh1, sh2)] * self.q[(sh3, sh4)]
    }
}

/// Which shell-quartet screening bound to use.
///
/// Both variants are RIGOROUS upper bounds, so this knob is a
/// speed-vs-setup-cost choice, NOT an accuracy tradeoff: neither can discard a
/// quartet carrying weight above the threshold. (Contrast the non-rigorous
/// CSAM family of Eqs. (9)/(11)/(12) of the same paper, which ferric
/// deliberately does not offer through this enum.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScreeningKind {
    /// Plain Cauchy-Schwarz, `Q_µν · Q_λσ`. The default, and byte-identical to
    /// every build predating CSB.
    #[default]
    Schwarz,
    /// The combined Schwarz bound (CSB),
    /// `min{Q_µν Q_λσ, M_µλ M_νσ, M_µσ M_νλ}` — Thompson & Ochsenfeld,
    /// JCP 147, 144101 (2017), Eq. (8). Rigorous, and never looser than
    /// `Schwarz` by construction (see [`CsbBounds`]).
    Csb,
}

impl ScreeningKind {
    /// Strict parse of the `[scf] screening` TOML value. Unknown values are a
    /// hard error naming the accepted set — never a silent default. Matches
    /// the convention of every other string knob in ferric's config
    /// (`QuadratureScheme`/`C6Source`/`DispersionPartition::parse_config_str`);
    /// see the "config honesty" section of CLAUDE.md for why silent-defaulting
    /// a typo'd knob is treated as a defect here.
    pub fn parse_config_str(s: &str) -> Result<Self, FerricError> {
        match s {
            "schwarz" => Ok(ScreeningKind::Schwarz),
            "csb" => Ok(ScreeningKind::Csb),
            other => Err(FerricError::General(format!(
                "unknown screening kind {other:?}; expected \"schwarz\" or \"csb\""
            ))),
        }
    }
}

/// CSB — the **combined Schwarz bound** of Thompson & Ochsenfeld, JCP 147,
/// 144101 (2017), Eq. (8):
///
/// ```text
///   |(µν|λσ)|  ≤  min{ Q_µν Q_λσ ,  M_µλ M_νσ ,  M_µσ M_νλ }
/// ```
///
/// with `M_µλ = sqrt(|(µµ|λλ)|)` from [`ferric_integrals::csb::csb_m_table`].
/// That module's header carries the verbatim quotations of Eqs. (6)/(7)/(8),
/// Appendix B's positive-definiteness condition and Table VI's list of
/// operators for which it is proven. Read it before changing anything here.
///
/// # Safety is STRUCTURAL, not documentary
///
/// [`CsbBounds::estimate`] computes the Schwarz product FIRST and then applies
/// `.min()` against the two `M`-derived terms. The consequence is worth
/// stating precisely, because it is the design's whole point:
///
/// * **CSB can never exceed Schwarz.** Guaranteed by the `min` itself,
///   independent of whether the `M` table is right. A bug in `csb_m_table`
///   therefore cannot produce a LOOSER bound — the failure mode that no
///   validity test would flag, because a looser bound is still valid.
/// * **CSB can still fall BELOW the truth if `M` is wrong**, because the `min`
///   will happily select a spuriously small `M` term. This is NOT hypothetical:
///   it is exactly the direction in which the sibling CSAM branch measured a
///   218,493× violation on benzene/cc-pVDZ under `erfc(1.0)`. The `min` is a
///   one-sided guarantee and must not be mistaken for a two-sided one.
///
/// So the `min` does not remove the need for a validity test — it CONCENTRATES
/// the entire failure surface into one sign, which one test
/// (`csb_is_a_valid_upper_bound` in `tests/csb_screening.rs`) then covers.
/// The defence on the other side is `csb_m_table`'s precision/floor discipline;
/// see its header.
///
/// # Both tables share one operator, by construction
///
/// `q` and `m` are built from the SAME [`Operator`], which this type stores and
/// exposes. There is no way to construct a `CsbBounds` whose two tables
/// disagree, because [`CsbBounds::compute`] builds both from its single `op`
/// argument. ferric has a scar here: `qqr.rs` once multiplied in an extra
/// `exp(-ω²R²)` factor that double-counted attenuation already present in the
/// erfc Schwarz factors, invalidating the bound at every sampled quartet (worst
/// |true|/bound 6.0e11). CSB has no analogous separate geometric factor —
/// attenuation enters `M` through the `(µµ|λλ)` integrals themselves, exactly
/// as it enters `Q` through `(µν|µν)`.
///
/// # CSB is provably INERT on diagonal quartets — so it cannot shrink a pair list
///
/// On `(ij|ij)` plain Schwarz is EXACT (`Q(i,j)² = max|(ij|ij)|`; the paper
/// states it directly — "the CSB estimates are exact for both (µν|µν)-type and
/// (µµ|νν)-type integrals ... the QQ estimates are exact for the former"), and
/// no valid upper bound can be smaller than an exact one. So the `min` always
/// selects the Schwarz term there.
///
/// The consequence is architectural and worth knowing before expecting a win
/// from the wrong place: `SignificantPairs::build` and `DensityPairs::build`
/// both screen on `estimate(i, j, i, j)`, so **swapping Schwarz for CSB cannot
/// shrink either pair list.** CSB acts only on LinK's innermost quartet test
/// and on `scatter_bra_pair`'s per-candidate screen. This is the same
/// structural fact `qqr.rs` records for QQR (proven over all pairs by
/// `tests/qqr_diagonal_noop.rs`), and it is why the bra-pair prefilters in
/// `direct_j.rs`/`direct_jk.rs` are deliberately left on plain Schwarz.
///
/// # Where it is expected to pay, and where it is not
///
/// The paper is explicit that under the long-range Coulomb operator "the CSB
/// estimate is no more useful than the QQ estimate for currently tractable
/// systems" (its Table I reports F_min = 1.000 for CSB under `1/r12`), and
/// that the win appears for strongly distance-decaying kernels — `e^(-r12)`,
/// `erfc(ω r12)/r12`. ferric's motivating profile (benzene/aug-cc-pVDZ RHF,
/// `quartet_scatter::scatter_bra_pair` at 28.96%) is a COULOMB workload, so a
/// small or null win there is the LITERATURE'S PREDICTION, not a symptom of a
/// broken port. Recorded before measuring, deliberately.
///
/// Nothing in this type has been benchmarked — no build or run was performed
/// while writing it.
#[derive(Debug, Clone)]
#[must_use = "CSB bounds are expensive to compute; dropping them wastes work"]
pub struct CsbBounds {
    /// The plain Schwarz bound this one takes a `min` against. Owned rather
    /// than borrowed so `CsbBounds` is a drop-in `Bound` with no lifetime.
    schwarz: SchwarzBounds,
    /// `m[(P, Q)] = sqrt(max_{µ∈P, λ∈Q} |(µµ|λλ)|)`, same shape as `q`.
    m: Array2<f64>,
}

impl CsbBounds {
    /// Build both tables for one operator.
    ///
    /// Roughly doubles screening-table setup: the `M` pass is one `(PP|QQ)`
    /// quartet per unordered shell pair, the same `O(nshells²)` quartet count
    /// as the Schwarz `(ij|ij)` pass. Geometry-only — built once per bound
    /// construction, never per SCF iteration. UNMEASURED (no benchmark was
    /// run); `csb_m_table`'s header records what to do if it ever shows up in
    /// a profile.
    pub fn compute(op: Operator, prep: &PreparedBasis) -> Result<Self, FerricError> {
        let schwarz = SchwarzBounds::compute(op, prep)?;
        let m = csb::csb_m_table(op, prep)?;
        Ok(CsbBounds { schwarz, m })
    }

    /// Build from an ALREADY-COMPUTED Schwarz table, adding only the `M` pass.
    ///
    /// The two tables must share an operator, so this checks rather than
    /// trusts: `op` is taken from `schwarz.op`, never from a second parameter
    /// that could disagree with it.
    pub fn from_schwarz(
        schwarz: SchwarzBounds,
        prep: &PreparedBasis,
    ) -> Result<Self, FerricError> {
        let m = csb::csb_m_table(schwarz.op, prep)?;
        Ok(CsbBounds { schwarz, m })
    }

    /// The underlying plain-Schwarz bound (the `min`'s first argument).
    pub fn schwarz(&self) -> &SchwarzBounds {
        &self.schwarz
    }

    /// The `M` table, `M[P][Q] = sqrt(max_{µ∈P,λ∈Q} |(µµ|λλ)|)`.
    pub fn m(&self) -> &Array2<f64> {
        &self.m
    }

    /// The operator both tables were built with.
    pub fn op(&self) -> Operator {
        self.schwarz.op
    }

    /// Number of shells.
    pub fn nshells(&self) -> usize {
        self.schwarz.nshells
    }
}

impl Bound for CsbBounds {
    /// Eq. (8), evaluated at shell granularity.
    ///
    /// The `.min()` chain is the safety property described in the type docs:
    /// `schwarz_est` is computed first and is the seed of the chain, so the
    /// returned value is `≤ schwarz_est` unconditionally — no matter what the
    /// `M` table contains.
    fn estimate(&self, sh1: usize, sh2: usize, sh3: usize, sh4: usize) -> f64 {
        // Argument 1 of the min: the plain Schwarz product, Eq. (1). Seeding
        // the chain with this is what makes "never looser than Schwarz"
        // structural rather than a comment.
        let schwarz_est = self.schwarz.estimate(sh1, sh2, sh3, sh4);
        // Arguments 2 and 3: Eqs. (6) and (7), the λ↔σ swap.
        //   Eq. (6):  M_µλ M_νσ  ->  M[s1,s3] * M[s2,s4]
        //   Eq. (7):  M_µσ M_νλ  ->  M[s1,s4] * M[s2,s3]
        let eq6 = self.m[(sh1, sh3)] * self.m[(sh2, sh4)];
        let eq7 = self.m[(sh1, sh4)] * self.m[(sh2, sh3)];
        schwarz_est.min(eq6).min(eq7)
    }
}

/// A zero-copy `Bound` view over a `&SchwarzBounds`, applying CSB when (and
/// only when) that value carries a `csb_m` table.
///
/// # Why this exists
///
/// `SchwarzBounds::estimate` deliberately stays plain Schwarz: `QqrBounds`
/// composes on top of it (`self.schwarz.estimate(..)`) and `pairs.rs` calls it
/// on diagonal quartets, and silently changing what those see would be a
/// non-local behaviour change for a knob that is meant to be opt-in. So the
/// CSB refinement is applied at the two places that actually screen quartets:
/// `quartet_scatter::scatter_bra_pair` (the dense direct path) and, via this
/// adapter, the LinK pair/quartet screen.
///
/// `solve_rhf` builds its own LinK bound from `config.screening` and does not
/// need this. `solve_uhf`/`solve_rohf` take the CALLER's `&SchwarzBounds` and
/// have no `RhfConfig::screening`-equivalent construction site of their own —
/// they wrap it here instead, so `[scf] screening = "csb"` reaches open-shell
/// LinK through the same single `bounds` value that carries it to the direct
/// builders. Without this wrapper, open-shell LinK would silently stay on
/// plain Schwarz while closed-shell LinK and every direct builder used CSB —
/// exactly the kind of half-wired knob CLAUDE.md's "config honesty" section
/// treats as a defect.
///
/// Borrowing, never cloning: the `nshells²` tables are read through `&`.
#[derive(Debug, Clone, Copy)]
pub struct CsbView<'a> {
    bounds: &'a SchwarzBounds,
}

impl<'a> CsbView<'a> {
    /// Wrap a `&SchwarzBounds`. If it carries no `csb_m` table (the default),
    /// this is exactly plain Schwarz — byte-identical, not merely equivalent.
    pub fn new(bounds: &'a SchwarzBounds) -> Self {
        CsbView { bounds }
    }

    /// Whether this view is actually applying CSB (i.e. the wrapped bounds
    /// carry an `M` table). Used by tests to prove non-vacuity.
    pub fn is_csb(&self) -> bool {
        self.bounds.csb_m.is_some()
    }
}

impl Bound for CsbView<'_> {
    #[inline]
    fn estimate(&self, sh1: usize, sh2: usize, sh3: usize, sh4: usize) -> f64 {
        // Same `.min()`-seeded-with-Schwarz structure as `CsbBounds::estimate`
        // — see that impl's doc for why the seed order is the safety property.
        let schwarz_est = self.bounds.q[(sh1, sh2)] * self.bounds.q[(sh3, sh4)];
        match &self.bounds.csb_m {
            None => schwarz_est,
            Some(m) => {
                let eq6 = m[(sh1, sh3)] * m[(sh2, sh4)];
                let eq7 = m[(sh1, sh4)] * m[(sh2, sh3)];
                schwarz_est.min(eq6).min(eq7)
            }
        }
    }
}

#[cfg(test)]
mod csb_unit_tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;

    fn water_csb(op: Operator) -> (CsbBounds, usize) {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let csb = CsbBounds::compute(op, &prep).unwrap();
        let nsh = prep.nshells();
        (csb, nsh)
    }

    /// The `min` is structural: CSB ≤ Schwarz at EVERY quartet, for every
    /// operator. This cannot fail while the `.min()` chain is seeded with
    /// `schwarz_est` — which is the point. It is asserted anyway because it
    /// would catch a WIRING error (e.g. someone replacing the seed with an
    /// `M` product, or reordering the chain so a bug could escape).
    #[test]
    fn csb_is_never_looser_than_schwarz() {
        for op in [Operator::coulomb(), Operator::erfc(1.0), Operator::erf(1.0)] {
            let (csb, nsh) = water_csb(op);
            for i in 0..nsh {
                for j in 0..nsh {
                    for k in 0..nsh {
                        for l in 0..nsh {
                            let s = csb.schwarz().estimate(i, j, k, l);
                            let c = csb.estimate(i, j, k, l);
                            assert!(
                                c <= s,
                                "op={op:?}: CSB({i},{j},{k},{l}) = {c} > Schwarz = {s} — the \
                                 .min() seed in CsbBounds::estimate has been broken"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Eq. (8) is symmetric under the permutations the true integral is
    /// symmetric under. `(µν|λσ) = (νµ|λσ) = (µν|σλ) = (λσ|µν)`, and the
    /// bound must follow — otherwise the screen would keep or drop a quartet
    /// depending on which of its eight equivalent index orders the caller
    /// happened to present. `scatter_bra_pair` relies on exactly this when it
    /// screens one canonical ordering and scatters all eight.
    ///
    /// The `min{Eq6, Eq7}` structure is what supplies it: swapping λ↔σ
    /// exchanges the two M terms, and the `min` of a set is invariant under
    /// reordering. A version that used only Eq. (6) would FAIL this test.
    #[test]
    fn csb_respects_the_eightfold_permutational_symmetry() {
        for op in [Operator::coulomb(), Operator::erfc(1.0)] {
            let (csb, nsh) = water_csb(op);
            for i in 0..nsh {
                for j in 0..nsh {
                    for k in 0..nsh {
                        for l in 0..nsh {
                            let base = csb.estimate(i, j, k, l);
                            for (a, b, c, d) in [
                                (j, i, k, l),
                                (i, j, l, k),
                                (j, i, l, k),
                                (k, l, i, j),
                                (l, k, i, j),
                                (k, l, j, i),
                                (l, k, j, i),
                            ] {
                                let permuted = csb.estimate(a, b, c, d);
                                assert!(
                                    (permuted - base).abs() <= 1e-12 * base.max(1.0),
                                    "op={op:?}: CSB not permutation-symmetric: \
                                     ({i},{j},{k},{l}) = {base:.6e} vs \
                                     ({a},{b},{c},{d}) = {permuted:.6e}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Strict parsing: unknown values error, they never silent-default.
    #[test]
    fn screening_kind_parses_strictly() {
        assert_eq!(
            ScreeningKind::parse_config_str("schwarz").unwrap(),
            ScreeningKind::Schwarz
        );
        assert_eq!(
            ScreeningKind::parse_config_str("csb").unwrap(),
            ScreeningKind::Csb
        );
        assert_eq!(ScreeningKind::default(), ScreeningKind::Schwarz);
        for bad in ["CSB", "Schwarz", "csam", "qqr", ""] {
            let err = ScreeningKind::parse_config_str(bad)
                .expect_err("unknown screening kind must be a hard error, not a silent default");
            let msg = format!("{err}");
            assert!(
                msg.contains("csb") && msg.contains("schwarz"),
                "the error must name the accepted set; got: {msg}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_integrals::basis_bridge::PreparedBasis;

    #[test]
    fn test_schwarz_bounds_estimate() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        let est = bounds.estimate(0, 1, 2, 3);
        let want = bounds.q[(0, 1)] * bounds.q[(2, 3)];
        assert!((est - want).abs() < 1e-15);
    }

    #[test]
    fn test_bound_trait_dispatch() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let bounds = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();

        // Call via trait object to prove dispatch works
        let bound: &dyn Bound = &bounds;
        let est = bound.estimate(0, 1, 2, 3);
        let direct = bounds.q[(0, 1)] * bounds.q[(2, 3)];
        assert!((est - direct).abs() < 1e-15, "trait dispatch mismatch: {est} vs {direct}");
    }
}
