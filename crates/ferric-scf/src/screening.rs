//! Schwarz integral screening for efficient Fock matrix construction.

use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::csam;
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
/// Optionally carries ONE refinement table — the CSB `M` table (see
/// [`SchwarzBounds::csb_m`]) or the CSAM `X` table (see
/// [`SchwarzBounds::csam_x`]) — which upgrades the screen in
/// `quartet_scatter::scatter_bra_pair` (a code span, not an intra-doc link:
/// the fn is `pub(crate)`, so a public item cannot link to it) from plain
/// Schwarz to Eq. (8)
/// or Eq. (9)/(11)/(12) respectively, without changing any builder's
/// signature. At most one is ever attached: they come from a single
/// three-way `match` in [`SchwarzBounds::compute_for_screening`].
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
    /// CSAM `X` table, `X[P][Q] = max_{µ∈P,λ∈Q} |(µµ|λλ)| / sqrt(|(µµ|µµ)||(λλ|λλ)|)`,
    /// built for the SAME `op` as `q`. `None` unless
    /// [`Self::compute_for_screening`] was called with [`ScreeningKind::Csam`].
    ///
    /// Distinct from `csb_m` because the two are different quantities feeding
    /// different formulas: `csb_m` is a plain `sqrt` of a diagonal integral and
    /// composes via `min` (rigorous); `csam_x` is a normalised RATIO and
    /// composes multiplicatively (non-rigorous). They are deliberately kept
    /// side by side rather than unified — a single "refinement table" field
    /// would invite exactly the confusion that makes CSAM look like a bound.
    pub csam_x: Option<Array2<f64>>,
}

impl SchwarzBounds {
    /// Compute Schwarz screening bounds for all shell pairs.
    ///
    /// Always leaves BOTH `csb_m` and `csam_x` as `None`, i.e. plain Schwarz.
    /// Use [`Self::compute_for_screening`] to opt into CSB or CSAM.
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
            csam_x: None,
        })
    }

    /// Compute Schwarz bounds, additionally building the CSB `M` table when
    /// `kind == ScreeningKind::Csb`, or the CSAM `X` table when
    /// `kind == ScreeningKind::Csam`.
    ///
    /// `ScreeningKind::Schwarz` delegates verbatim to [`Self::compute`] and is
    /// therefore byte-identical to it — there is no separate code path that
    /// could drift.
    ///
    /// This three-way `match` is what makes `csb_m` and `csam_x` MUTUALLY
    /// EXCLUSIVE: exactly one arm runs, so at most one table is ever attached.
    /// Every downstream consumer relies on that (`scatter_bra_pair` and
    /// `LinkBound`'s `schwarz_ref_estimate` both check CSB first and CSAM only
    /// in the `else`, so the rigorous one would win if it were ever violated).
    ///
    /// The refinement table is built from the SAME `op` passed here, so the
    /// per-operator contract cannot be violated by a caller: there is no way
    /// to supply two operators. Under `Csam` a short-range (`erfc`) `op` is a
    /// hard error naming `screening = "csb"`, surfaced by the `?` below.
    pub fn compute_for_screening(
        op: Operator,
        prep: &PreparedBasis,
        kind: ScreeningKind,
    ) -> Result<Self, FerricError> {
        let mut bounds = Self::compute(op, prep)?;
        match kind {
            ScreeningKind::Schwarz => {}
            ScreeningKind::Csb => {
                bounds.csb_m = Some(csb::csb_m_table(op, prep)?);
            }
            ScreeningKind::Csam => {
                // `csam_x_table` itself refuses short-range kernels and names
                // `screening = "csb"` in the error, so an erfc run under
                // `csam` fails loudly here rather than silently under-screening.
                bounds.csam_x = Some(csam::csam_x_table(op, prep)?);
            }
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
/// `Schwarz` and `Csb` are RIGOROUS upper bounds, so choosing between THOSE
/// TWO is a speed-vs-setup-cost choice, NOT an accuracy tradeoff: neither can
/// discard a quartet carrying weight above the threshold.
///
/// `Csam` is NOT a bound. It is the non-rigorous estimate of Eqs. (9)/(11)/(12)
/// of the same paper, it can underestimate a true integral, and selecting it IS
/// an accuracy-vs-threshold tradeoff. The distinction is the single most
/// important thing about this enum — see each variant's own doc, and
/// [`CsamBounds`] for the full accounting.
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
    /// The combined Schwarz approximation (CSAM), same paper, Eq. (9)/(11)/(12).
    ///
    /// **NON-RIGOROUS for every kernel, including Coulomb** — the paper says so
    /// in its own words ("we use M̃_µλ to formulate our non-rigorous combined
    /// Schwarz approximations", `≈` not `≤`). It can and does underestimate
    /// true integrals, so it is an *estimate*, not a `Bound`, and is exposed
    /// only through the audited `LinkBound::Csam` seam.
    ///
    /// Measured energy errors are nonetheless small and threshold-controlled:
    /// −0.20 … +1.80 **nanohartree** at ϑ = 1e-12 (the paper's Table V) and
    /// 0.05–9.35 µH at ϑ = 1e-10 (Table III), growing LINEARLY with system
    /// size (Fig. 2). This is why Psi4 ships it as its own default
    /// (`SCREENING=CSAM`) for Coulomb and erf.
    ///
    /// Short-range kernels route to [`ScreeningKind::Csb`] instead — the
    /// paper's own recommendation (p. 144101-8/9).
    Csam,
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
            "csam" => Ok(ScreeningKind::Csam),
            other => Err(FerricError::General(format!(
                "unknown screening kind {other:?}; expected \"schwarz\", \"csb\" or \"csam\""
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

/// CSAM screening ESTIMATE: a tight but **non-rigorous** (underestimating)
/// multiplicative refinement of [`SchwarzBounds`], ported from Psi4's
/// `shell_significant_csam()` (`psi4/src/psi4/libmints/twobody.cc`,
/// lines 206-224 / 363-398).
///
/// # NOT a `Bound` — deliberately
///
/// This type does **not** implement [`Bound`], and must not be made to. The
/// `Bound` contract is "never underestimates `|(sh1 sh2|sh3 sh4)|`"; CSAM
/// violates it. See `ferric_integrals::csam`'s module header for the
/// primary-source citation (Thompson & Ochsenfeld, J. Chem. Phys. 147,
/// 144101 (2017) — the Coulomb operator appears only in that abstract's
/// *non-rigorous estimate* sentence) and for the independent numerical
/// reproduction showing `true/estimate` up to 2.53 (Coulomb) and 9.37
/// (`erfc`) using *Psi4's own* formula on a harness where plain Schwarz
/// saturates at exactly 1.000000.
///
/// Consequently this is an accuracy-vs-threshold tradeoff knob, NOT a
/// free speedup: any quartet it discards may carry real weight, and the
/// justification for using it is that the discarded weight shrinks with the
/// screening threshold — a claim that must be MEASURED (see
/// `csam_screening.rs`'s error-vs-threshold characterization test), never
/// assumed.
///
/// Contrast [`CsbBounds`] directly above, which is the RIGOROUS member of the
/// same paper's family (Eq. (8), a `min` of three genuine Cauchy-Schwarz
/// bounds) and therefore carries none of the caveats in this doc comment.
///
/// Structured like [`crate::qqr::QqrBounds`] — a Schwarz bound composed with
/// a `<= 1` multiplicative factor — except QQR's factor is a rigorous
/// GEOMETRIC (distance) one, while this one is a non-rigorous ALGEBRAIC
/// (exchange-type) estimate. They are not combined here (Psi4 does not
/// combine them either; CSAM is presented there as an alternative to, not a
/// composition with, its own MBFS/QQR-style distance bound).
///
/// # Square-root convention (read before touching this file)
///
/// ferric's `SchwarzBounds::estimate` is the UNSQUARED bound
/// `Q(i,j) * Q(k,l)` on `|(ij|kl)|` (verified: `Q(i,j) =
/// sqrt(max|(ij|ij)|)` in `ferric_integrals::schwarz::schwarz_pair`, and
/// `estimate` multiplies two such `Q`s with no further rooting/squaring).
/// Psi4's CSAM works throughout in SQUARED quantities and its `csam_2` factor
/// is meant to multiply a squared Schwarz product. Composing it onto ferric's
/// UNSQUARED estimate therefore requires `sqrt(csam_2)`, not `csam_2` itself —
/// see the full derivation in `ferric_integrals::csam`'s module doc. Using the
/// bare (squared) factor here would silently square the effective per-pair
/// ratio and over-tighten the bound into an invalid (underestimating) one.
///
/// # Rigor, honestly stated
///
/// CSAM is NON-RIGOROUS for **every** kernel, Coulomb included — the paper
/// labels Eqs. (9)/(11)/(12) "our non-rigorous combined Schwarz
/// approximations" in its own voice. What that costs is a threshold-controlled
/// error, not a guarantee: Thompson & Ochsenfeld's Table V (aug-cc-pVDZ,
/// `theta = 1e-12`) measures CSAM HF energy errors of **-0.20 to +1.80
/// nanohartree** across six systems, against the rigorous QQ bound's own
/// -1.50..+0.70 nH on the same rows; Table III (cc-pVDZ, `theta = 1e-10`)
/// gives 0.05-9.35 microhartree, roughly 2x QQ's. The error does grow linearly
/// with system size at fixed threshold (Fig. 2) — that, not any measured
/// blow-up, is the reason to prefer a rigorous bound where one is available.
///
/// # Per-operator
///
/// [`ferric_integrals::csam::csam_x_table`] accepts `Coulomb` and
/// `ErfCoulomb`; it routes `ErfcCoulomb` to the rigorous CSB bound (`screening
/// = "csb"`) with a typed error, following the paper's own conclusion that CSB
/// should be the default estimate for short-range operators. Plain
/// [`SchwarzBounds`] also remains rigorous and supports all three.
///
/// The `X` table passed to [`CsamBounds::new`] MUST still have been built by
/// [`ferric_integrals::csam::csam_x_table`] with the SAME [`Operator`] as the
/// wrapped [`SchwarzBounds`]. Unlike `qqr.rs`'s geometric envelope (which has an entirely
/// separate, previously-buggy operator-attenuation history — see that
/// module's doc), CSAM's attenuation lives inside the integrals that build
/// `X` itself, so there is no separate operator-dependent factor to apply or
/// omit here; the hazard is purely "don't mix tables built under different
/// operators", which [`CsamBounds::new`] cannot itself detect (the caller
/// must build both from the same `op`).
#[derive(Debug, Clone)]
pub struct CsamBounds {
    schwarz: SchwarzBounds,
    /// `X[(p,q)]`, symmetric, `X[(p,p)] == 1.0`, built by
    /// [`ferric_integrals::csam::csam_x_table`] for the SAME operator as
    /// `schwarz`.
    x: Array2<f64>,
}

impl CsamBounds {
    /// Build CSAM bounds from an already-computed [`SchwarzBounds`] and its
    /// matching CSAM `X` table (see [`ferric_integrals::csam::csam_x_table`]).
    /// Does not itself validate that `x` was built with `schwarz.op` — that
    /// is the caller's responsibility (mirrors [`crate::qqr::QqrBounds::new`],
    /// which similarly trusts its caller to pass a `Molecule`/`BasisSet`
    /// consistent with `schwarz`).
    pub fn new(schwarz: SchwarzBounds, x: Array2<f64>) -> Self {
        let nshells = schwarz.nshells;
        assert_eq!(
            x.dim(),
            (nshells, nshells),
            "CSAM X table shape {:?} does not match Schwarz nshells {nshells}",
            x.dim()
        );
        CsamBounds { schwarz, x }
    }

    /// Convenience constructor: computes both the Schwarz table and the CSAM
    /// `X` table for `op` from scratch, guaranteeing they share the same
    /// operator (closes the one hazard [`CsamBounds::new`] cannot check).
    pub fn compute(op: Operator, prep: &PreparedBasis) -> Result<Self, FerricError> {
        let schwarz = SchwarzBounds::compute(op, prep)?;
        let x = csam::csam_x_table(op, prep)?;
        Ok(Self::new(schwarz, x))
    }

    /// Access the underlying Schwarz bounds.
    pub fn schwarz(&self) -> &SchwarzBounds {
        &self.schwarz
    }

    /// The CSAM `X` table.
    pub fn x(&self) -> &Array2<f64> {
        &self.x
    }

    /// The CSAM exchange-type refinement factor for shell quartet
    /// `(sh1,sh2|sh3,sh4)`: `sqrt(max(X[sh1,sh3]*X[sh2,sh4],
    /// X[sh1,sh4]*X[sh2,sh3]))` — see the type-level doc for why the square
    /// root is required in ferric's (unsquared) convention. This mirrors
    /// Psi4's `csam_2 = max(mm_rr*nn_ss, mm_ss*nn_rr)` with Psi4's
    /// `(M,N|R,S)` renamed to ferric's `(sh1,sh2|sh3,sh4)`: `mm_rr` pairs
    /// bra-shell-1 with ket-shell-1 (`X[sh1,sh3]`), `nn_ss` pairs bra-shell-2
    /// with ket-shell-2 (`X[sh2,sh4]`) — one cross term — and the other term
    /// swaps which ket shell pairs with which bra shell (`X[sh1,sh4]`,
    /// `X[sh2,sh3]`), matching Psi4's other listed cross term.
    fn refinement_factor(&self, sh1: usize, sh2: usize, sh3: usize, sh4: usize) -> f64 {
        let term_a = self.x[(sh1, sh3)] * self.x[(sh2, sh4)];
        let term_b = self.x[(sh1, sh4)] * self.x[(sh2, sh3)];
        term_a.max(term_b).max(0.0).sqrt()
    }
}

impl CsamBounds {
    /// The CSAM **estimate** (NOT a bound) of `|(sh1 sh2|sh3 sh4)|`.
    ///
    /// Named `estimate_nonrigorous` rather than `estimate`, and deliberately
    /// NOT an impl of [`Bound`], so that no call site can obtain this value
    /// through a `&dyn Bound` and treat it as an upper bound. It can fall
    /// BELOW the true integral magnitude — see the type-level doc.
    pub fn estimate_nonrigorous(&self, sh1: usize, sh2: usize, sh3: usize, sh4: usize) -> f64 {
        let schwarz_est = self.schwarz.estimate(sh1, sh2, sh3, sh4);
        let factor = self.refinement_factor(sh1, sh2, sh3, sh4);
        // `factor` is `sqrt(product of two X ratios each in [0,1])`, so it is
        // itself in [0,1] up to roundoff; clamp defensively so a tiny
        // overshoot from floating-point error cannot make the estimate exceed
        // plain Schwarz. NOTE this clamp bounds the estimate from ABOVE only:
        // nothing here (and nothing in Psi4) clamps it from BELOW against the
        // true integral, which is precisely why it is not a bound.
        schwarz_est * factor.min(1.0)
    }
}

/// A `Bound` implementation chosen at runtime from [`ScreeningKind`], for
/// call sites (currently: `solve_rhf`'s LinK-builder construction, and
/// `solve_uhf`/`solve_rohf`'s) that need to pass either an owned freshly-built
/// table OR a borrowed already-existing one to the SAME generic
/// `build_pluggable_k::<B: Bound>` call without duplicating that call per
/// variant.
///
/// # `SchwarzRef` — the borrowing variant, and why it is not merely a placeholder
///
/// `SchwarzRef` borrows a caller-supplied `&SchwarzBounds` and applies
/// WHICHEVER refinement table that value happens to carry:
///
/// * no table (the default) → plain Schwarz, BITWISE, not merely equivalent;
/// * `csb_m` present → CSB Eq. (8), the same `min` as [`CsbBounds::estimate`];
/// * `csam_x` present → the CSAM multiplicative estimate, the same formula as
///   [`CsamBounds::estimate_nonrigorous`].
///
/// That is what makes this ONE enum sufficient for every LinK seam.
/// `solve_rhf` builds its own LinK bound from `config.screening` and uses the
/// owned variants. `solve_uhf`/`solve_rohf` take the CALLER's
/// `&SchwarzBounds` and have no `RhfConfig::screening`-equivalent construction
/// site of their own — they pass `SchwarzRef(bounds)` instead, so
/// `[scf] screening = "csb"`/`"csam"` reaches open-shell LinK through the same
/// single `bounds` value that carries it to the direct builders. Without that,
/// open-shell LinK would silently stay on plain Schwarz while closed-shell
/// LinK and every direct builder used the selected refinement — exactly the
/// kind of half-wired knob CLAUDE.md's "config honesty" section treats as a
/// defect.
///
/// `SchwarzRef` additionally serves as the zero-cost placeholder for the
/// non-"link" branch (the common "direct"/"cosx" case, where
/// `build_pluggable_k` never reads this argument at all — see that function's
/// `match kind`), so no table is cloned or recomputed there.
///
/// Borrowing, never cloning: the `nshells²` tables are read through `&`.
///
/// # Historical note
///
/// This enum replaces an earlier `CsbView<'a>` borrowing wrapper that did
/// exactly what `SchwarzRef` does, but for CSB only. The two were merged
/// because the enum is the more general shape and one adapter at this seam is
/// easier to audit than two.
pub enum LinkBound<'a> {
    Schwarz(SchwarzBounds),
    SchwarzRef(&'a SchwarzBounds),
    Csb(CsbBounds),
    Csam(CsamBounds),
}

impl LinkBound<'_> {
    /// Whether this bound is actually applying CSB — i.e. it is an owned
    /// [`CsbBounds`], or a `SchwarzRef` over bounds carrying a `csb_m` table.
    /// Used by tests to prove non-vacuity (a refinement that never engages
    /// makes every other assertion pass for the wrong reason).
    pub fn is_csb(&self) -> bool {
        match self {
            LinkBound::Csb(_) => true,
            LinkBound::SchwarzRef(b) => b.csb_m.is_some(),
            LinkBound::Schwarz(b) => b.csb_m.is_some(),
            LinkBound::Csam(_) => false,
        }
    }

    /// Whether this bound is applying the non-rigorous CSAM estimate.
    pub fn is_csam(&self) -> bool {
        match self {
            LinkBound::Csam(_) => true,
            LinkBound::SchwarzRef(b) => b.csam_x.is_some(),
            LinkBound::Schwarz(b) => b.csam_x.is_some(),
            LinkBound::Csb(_) => false,
        }
    }
}

/// NOTE: implementing [`Bound`] here means a `LinkBound::Csam` reaches LinK
/// through a trait whose contract is "never underestimates" — which CSAM
/// violates (see [`CsamBounds`]). That is tolerated ONLY because selecting
/// the `Csam` variant is an explicit, non-default opt-in
/// (`[scf] screening = "csam"`) into a documented non-rigorous estimate; the
/// `estimate_nonrigorous` call below is deliberately spelled out rather than
/// routed through a `Bound` impl on `CsamBounds` itself, so that this is the
/// single auditable place where the non-rigorous value enters bound-typed
/// code. Do not add further `Bound` impls that forward to CSAM.
///
/// The `Csb` arm carries NO such caveat: CSB (Eq. (8)) is a RIGOROUS upper
/// bound — a `min` seeded with the plain Schwarz product — so forwarding it to
/// the real [`CsbBounds::estimate`] through this `Bound` impl honours the
/// trait's contract exactly as `Schwarz` does. Likewise `SchwarzRef` is
/// rigorous unless the borrowed value carries a `csam_x` table, which only
/// `compute_for_screening(.., Csam)` can attach.
impl Bound for LinkBound<'_> {
    #[inline]
    fn estimate(&self, sh1: usize, sh2: usize, sh3: usize, sh4: usize) -> f64 {
        match self {
            LinkBound::Schwarz(b) => schwarz_ref_estimate(b, sh1, sh2, sh3, sh4),
            LinkBound::SchwarzRef(b) => schwarz_ref_estimate(b, sh1, sh2, sh3, sh4),
            LinkBound::Csb(b) => b.estimate(sh1, sh2, sh3, sh4),
            LinkBound::Csam(b) => b.estimate_nonrigorous(sh1, sh2, sh3, sh4),
        }
    }
}

/// The estimate a borrowed [`SchwarzBounds`] implies, honouring whichever
/// refinement table it carries.
///
/// Kept as a free function so both the `Schwarz` (owned) and `SchwarzRef`
/// (borrowed) arms of [`LinkBound`]'s `Bound` impl share ONE body — a second
/// copy is exactly how the old `CsbView`/`CsbBounds` pair could drift.
///
/// The `csb_m` arm reproduces [`CsbBounds::estimate`]'s `.min()`-seeded-with-
/// Schwarz structure verbatim (see that impl's doc for why the seed order is
/// the safety property), and the `csam_x` arm reproduces
/// [`CsamBounds::estimate_nonrigorous`]'s clamped multiply. A value carrying
/// NEITHER table — every default build — returns the bare Schwarz product,
/// bitwise.
///
/// `csb_m` and `csam_x` are mutually exclusive by construction:
/// [`SchwarzBounds::compute_for_screening`] is a three-way `match` on one
/// [`ScreeningKind`] and attaches at most one of them. The `csb_m` arm is
/// checked FIRST, so were both ever attached (only reachable by hand-mutating
/// the struct), the RIGOROUS one would win — the safe direction.
#[inline]
fn schwarz_ref_estimate(
    b: &SchwarzBounds,
    sh1: usize,
    sh2: usize,
    sh3: usize,
    sh4: usize,
) -> f64 {
    let schwarz_est = b.q[(sh1, sh2)] * b.q[(sh3, sh4)];
    if let Some(m) = &b.csb_m {
        let eq6 = m[(sh1, sh3)] * m[(sh2, sh4)];
        let eq7 = m[(sh1, sh4)] * m[(sh2, sh3)];
        return schwarz_est.min(eq6).min(eq7);
    }
    if let Some(x) = &b.csam_x {
        let term_a = x[(sh1, sh3)] * x[(sh2, sh4)];
        let term_b = x[(sh1, sh4)] * x[(sh2, sh3)];
        let factor = term_a.max(term_b).max(0.0).sqrt().min(1.0);
        return schwarz_est * factor;
    }
    schwarz_est
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
        assert_eq!(
            ScreeningKind::parse_config_str("csam").unwrap(),
            ScreeningKind::Csam
        );
        assert_eq!(ScreeningKind::default(), ScreeningKind::Schwarz);
        // "csam" moved OUT of this list when the CSAM branch merged in — it is
        // now an accepted (non-rigorous, opt-in) value, asserted above.
        for bad in ["CSB", "Schwarz", "CSAM", "qqr", ""] {
            let err = ScreeningKind::parse_config_str(bad)
                .expect_err("unknown screening kind must be a hard error, not a silent default");
            let msg = format!("{err}");
            assert!(
                msg.contains("csb") && msg.contains("schwarz") && msg.contains("csam"),
                "the error must name the accepted set; got: {msg}"
            );
        }
    }

    /// On a DIAGONAL quartet (sh1==sh3, sh2==sh4) the CSAM refinement factor is
    /// `sqrt(X[sh1,sh1]*X[sh2,sh2]) = sqrt(1*1) = 1` (X's diagonal is exactly
    /// 1.0 by Cauchy-Schwarz equality), so CSAM must equal Schwarz exactly
    /// there — the same "self-quartet" case QQR's overlapping-pair test checks
    /// for its own bound, and the same structural inertness `CsbBounds`'s type
    /// doc records for CSB.
    #[test]
    fn csam_bounds_compute_matches_schwarz_on_diagonal_quartets() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let csam = CsamBounds::compute(Operator::coulomb(), &prep).unwrap();
        let nsh = prep.nshells();
        for i in 0..nsh {
            for j in 0..nsh {
                let s = csam.schwarz().estimate(i, j, i, j);
                let c = csam.estimate_nonrigorous(i, j, i, j);
                assert!(
                    (s - c).abs() < 1e-12 * s.max(1.0),
                    "diagonal quartet ({i},{j},{i},{j}): CSAM = {c} != Schwarz = {s}"
                );
            }
        }
    }

    #[test]
    fn csam_bounds_new_rejects_mismatched_table_shape() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let schwarz = SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        let nsh = prep.nshells();
        let wrong_shape = Array2::<f64>::zeros((nsh + 1, nsh + 1));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            CsamBounds::new(schwarz, wrong_shape)
        }));
        assert!(result.is_err(), "CsamBounds::new must reject a mismatched X table shape");
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
