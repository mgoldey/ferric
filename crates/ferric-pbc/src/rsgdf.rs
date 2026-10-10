//! Gamma-point range-separated Gaussian density fitting (RS-GDF) and the
//! J/K builders on it (Stage 1 step 9, `reference/pbc/stage1-design.md` §4;
//! Rust port of the validated prototype `reference/pbc/pbc_gdf.py`,
//! `reference/pbc/FINDINGS.md` "Iteration 2 (Python, RS-GDF)").
//!
//! # Target
//!
//! `B[k, μν]` with `I[μν, λσ] ≈ Σ_k B[k,μν] B[k,λσ]`, where `I` is the
//! Gamma ERI of [`crate::dense_aft`] in its G = 0 convention:
//!
//! ```text
//! I[μν,λσ] = (1/Ω) Σ_{G≠0} 4π/G² conj(P_μν(G)) P_λσ(G)
//! ```
//!
//! Every fitted quantity lives in the SAME G = 0-dropped kernel
//! `v'(G) = 4π/G²` (G ≠ 0):
//!
//! ```text
//! J2[P,Q]  = (P|Q)'  = (1/Ω) Σ_{G≠0} v(G) conj(X_P(G)) X_Q(G)       (aux metric)
//! J3[μν,P] = (μν|P)' = (1/Ω) Σ_{G≠0} v(G) conj(P_μν(G)) X_P(G)
//! J2 = U s Uᵀ,  B = s^{-1/2} Uᵀ J3ᵀ   over the eigenvalues s > lindep       (Dunlap-robust)
//! ```
//!
//! Because the fitting metric IS the target kernel, the fit is variational in
//! the kernel the energy uses; charged aux functions need no compensating
//! charge because the only divergent term (G = 0) is removed identically from
//! J2 and J3. This is the PySCF RSGDF convention (`rsdf_builder.py`
//! `get_2c2e` `g0_fac`, `gen_j3c_loader` `vbar`).
//!
//! # Evaluation (Ewald split `1/r = erfc(ωr)/r + erf(ωr)/r`)
//!
//! * **SR** — real-space lattice sums of ordinary libint2 `erfc(ω)` integrals
//!   on translated shells (no image bases):
//!   `J2 += Σ_T (P_0|Q_T)` via `Engine::compute_eri2_shifted`,
//!   `J3 += Σ_{L,T} (μ_0 ν_L|P_T)` via `Engine::compute_eri3_shifted`.
//!   These sums implicitly contain the erfc G = 0 term.
//! * **LR** — `(1/Ω) Σ_{G≠0} 4π/G² e^{−G²/4ω²} conj(A(G)) X_P(G)` on the half
//!   G sphere (weight 2), `A` = aux FT ([`aux_ft`]) or pair FT
//!   ([`crate::pair_ft::pair_ft_chunked`], G-chunked, never all G at once).
//! * **G = 0** — ONE function ([`subtract_g0`]) removes `c0 q qᵀ` from J2 and
//!   `c0 S_μν q_P` from J3, `c0 = π/(ω²Ω)`, `q_P = ∫χ_P = X_P(0)`. The result
//!   is ω-independent (tested).
//!
//! # Metric solve
//!
//! Symmetric eigendecomposition with an ABSOLUTE eigenvalue cut `lindep`
//! (default 1e-10 = PySCF `LINEAR_DEP_THR`) and a REPORTED dropped count
//! ([`RsGdfStats::n_dropped`]); never plain Cholesky or LU. Diffuse aux in a
//! small cell make the periodic metric near-singular (prototype LiH: 43/240
//! eigenvalues < 1e-10), and the eig cut was stable over 8 decades of
//! threshold where PySCF's default path was not.
//!
//! # Screening (all derived from `precision`)
//!
//! For a (μ-shell, ν-shell at L) pair with s-type charge bound
//! `q_ab = max |c_a c_b| (π/p)^{3/2} e^{−ab R²/p}` and an aux shell with charge
//! bound `Q_P = max_k |c_k| (π/α_k)^{3/2} (1 + α_k^{−1/2})^{l}` (the prototype's
//! "SR cutoff scaled by |q_P|": normalised diffuse aux carry |q| ≫ 1), the
//! erfc interaction of two Gaussian distributions at distance `d` is at most
//! `q_ab Q_P (1 + 2μ/√π) e^{−ν² (d − 2)²}` with `μ² = pα/(p+α)` (largest
//! exponents) and `1/ν² = 1/p_min + 1/α_min + 1/ω²` (smallest). `d` is the
//! distance from the aux image centre to the segment `[A, B+L]` (every product
//! centre lies on it); the 2-Bohr margin absorbs the l > 0 polynomial factors
//! (the same construction as `hcore`'s SR nucleus screen). A triplet is
//! computed iff that bound is `>= precision`. `sr_screen = false` replaces
//! the per-triplet radius by the global maximum (the prototype's unscreened
//! image-shell form), for measuring what the screen drops.
//!
//! # Memory
//!
//! Every buffer that grows with the lattice, the G sphere or `naux·nao²` is
//! reserved on a [`crate::budget`] ledger before allocation against
//! [`RsGdfConfig::budget_bytes`] (default: ferric's unified budget). The
//! resident B is `8 naux nao²` bytes (dense, in core — no spill path yet).
//!
//! # Consumption (J/K)
//!
//! [`RsGdfJ`]/[`RsGdfK`] implement `ferric_scf::fock::{JBuilder, KBuilder}`
//! directly from the in-core B (J = Σ_k B_k (B_k·D), K = Σ_k B_k D B_kᵀ +
//! `v_M S D S` for `exxdiv = ewald`). `DfJ`/`DfK` were NOT reused: both
//! recompute a molecular Coulomb metric from the aux `PreparedBasis`, and
//! `DfJ` inverts it with LU (`.inv()`), which the prototype showed is the
//! wrong solve for a near-singular periodic metric; reusing them would need
//! new `from_parts` constructors plus a `ThreeIndexSource::from_blocks` in
//! ferric-scf. Using ONE B for both J and K also keeps the fitted ERI a single
//! positive-semidefinite `BᵀB`.
//!
//! # Range split (opt-in, Gamma energy path)
//!
//! [`RsGdfConfig::range_split`] = `Some(`[`RangeSplit`]`)` splits every
//! shell by primitive exponent and moves the blocks whose FT converges in
//! the LR sphere (`(any pair | smooth aux)`, `(ss pair | compact aux)`,
//! diffuse metric terms) from the SR real-space walks to G space; the
//! G = 0 subtract gets the kept inputs (PySCF grouping). FINDINGS
//! "Iteration 23"; details in the `split` module. `None` is today's
//! construction bit for bit. The Gamma forces and stress differentiate the
//! split energy (FINDINGS "Iteration 26"); the k-point build refuses a
//! split config ([`RsGdf::range_split`]).
//!
//! # Orbital-pair symmetry of the Gamma SR 3-centre sum (s2)
//!
//! By lattice translation, `(ν_0 μ_L | P_T) = (μ_0 ν_{−L} | P_{T−L})`, and the
//! pair-image set, the aux walk around the segment and every screen radius
//! are closed under that map, so at Gamma `J3_SR[νμ] = J3_SR[μν]` exactly in
//! exact arithmetic. The Gamma build therefore evaluates each UNORDERED shell
//! pair `i1 ≤ i2` once over EVERY pair image `L` (the same per-pair body and
//! per-element addend sequence as the ordered walk) and one task COPIES the
//! finished block into both row `μν` and row `νμ`. Diagonal pairs
//! (`i1 == i2`) also run over every `L` (not `L = 0` plus a half-space) and
//! write the function pair `i ≤ j` of the block into both `(i, j)` and
//! `(j, i)`: a half-space would save only the diagonal pairs' share of the
//! work (`~1/(2 nsh)` of the total) and needs an `L ↔ −L` map of the image
//! list, including the strained frozen-index frame. Consequences:
//!
//! * `J3_SR` is EXACTLY symmetric, and every element is bitwise one of the
//!   two ordered evaluations the old walk averaged (unsplit: the one with
//!   `μ ≤ ν`). `|new − old| ≤ ½|x − y| + ½ ulp` per element, where `x, y` are
//!   the two ordered sums (round-off only; `RsGdfStats::asym_j3` of the old
//!   build). The remaining `symmetrize_pairs` averages only the LR and G = 0
//!   parts' asymmetry.
//! * Every output element is still owned by exactly one task (a copy, not an
//!   addition), so the build stays bitwise identical across thread counts.
//! * Range split: the kept part `χ_iχ_j − χ_i^sχ_j^s` is symmetric per
//!   unordered pair, so either orientation's calls give both rows; the task
//!   evaluates the orientation with FEWER kept calls (compact × split shell:
//!   1 call instead of 2), ties in index order, so a split that moves nothing
//!   is still bitwise the unsplit build.
//! * Counters: `RsGdfStats::n_sr3_triplets` counts the triplets COMPUTED;
//!   `n_sr3_triplets_ordered` weights them 2 (off-diagonal) / 1 (diagonal),
//!   which for the unsplit walk is the pre-s2 ordered count (compare
//!   benchmarks on it). For the range split the pre-s2 counter also paid the
//!   costlier orientation; [`sr_walk_counts`] reports it as
//!   `n_sr3_triplets_s1`.
//! * Frozen oracle: [`RsGdf::build_pair_s1_oracle`] is the pre-s2 build
//!   (ordered walk + averaging), bit for bit.
//! * k-points keep the ordered walk: `(ν μ)` lands in residue bin
//!   `(−r_L, r_T − r_L)`, so s2 there needs that residue map and a
//!   non-TRIM-mesh test first. The SR metric `J2` (aux pairs) also keeps the
//!   ordered walk: it is ~0.1% of the SR3 calls (diamond cc-pVDZ: 159 k
//!   pairs vs 197 M triplets). The force/stress derivative walks
//!   (`deriv`, `strain`, `split`'s `deriv`) have their own ordered loops and
//!   are unchanged.
//!
//! # Column rotation of the Gamma SR 3-centre walk (on by default)
//!
//! [`RsGdfConfig::sr_column_rotation`] = [`SrColumnRotation::Auto`] (the
//! default) or [`SrColumnRotation::On`] evaluates the Gamma SR 3-centre
//! sum (unsplit, or the kept part of a range split) on the COLUMN-ROTATED orbital basis of [`crate::sr_rotation`]
//! (generally contracted shell groups: the single-primitive columns
//! subtracted from the others, zero primitives dropped) and back-transforms
//! the finished `J3_SR` rows EXACTLY into the parent AO basis
//! (`J3 = (T ⊗ T) J3'`, a serial pass over (group, group) pairs) before the
//! LR and G = 0 terms are added. Everything else — the pair images (from
//! the parent shells; the rotated primitive sets are subsets, so the parent
//! radius covers them), the metric, the LR and G = 0 parts, `S_ss` and the
//! moved `(ss | X_c)` block of a split — stays in the parent basis; the
//! split's compact/smooth partition is linear in the coefficient vector, so
//! the kept part is covariant too. The kept triplet set is the rotated
//! shells' own screen, so the result matches the unrotated build to the
//! screening precision, not bitwise; `n_sr3_triplets` counts the rotated
//! walk's calls. [`SrColumnRotation::Off`] is the unrotated build bit for
//! bit, and so is a basis with nothing to rotate (detection returns the
//! identity; counter `rsgdf SR3 rotated columns` = 0). Gamma energy builds
//! only: [`RsGdf::build_for_gradient`] and the frozen s1 oracle run
//! unrotated under `Auto` and refuse an explicit `On` (the forces must
//! differentiate exactly the walk the energy ran, and the gradient build IS
//! the energy build of a force run), and so does the k-point build
//! ([`kpoint`]).
//!
//! # Forces
//!
//! [`RsGdf::build_for_gradient`] keeps the metric eigen-data the analytic
//! forces need (the dropped eigenvectors and `J3` projected on them, for the
//! kept–dropped Loewner term); the derivative contractions live in
//! `deriv` and share this module's SR walks
//! (`Stage::sr_three_index_walk` / `sr_metric_walk`), pair images and G
//! sphere, so the force differentiates exactly the truncated energy built
//! here. Formulas: `deriv`'s module doc and FINDINGS "Iteration 18".
//!
//! Units: Bohr and Hartree; ω in Bohr⁻¹.

use crate::budget::{bytes_of, Ledger};
use crate::dense_aft::ExxDiv;
use crate::ewald::madelung_constant;
use crate::hcore::{gvector_list_bytes, half_gvectors, G_CHUNK_BYTES};
use crate::kpts::lattice_coords;
use crate::lattice::Cell;
use crate::ordered::{ordered_units, Stored};
use crate::pair_ft::residues::residue_index;
use crate::pair_ft::{pair_ft_chunked_serial_oracle, pair_ft_chunked_timed};
use crate::sr_rotation::{ColumnRotationMutant, RotatedBasis, SrColumnRotation};
use crate::timing::{CallClock, PbcTimings, StageClock};
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::engine::Engine;
use ferric_integrals::engine_pool::EnginePool;
use ferric_integrals::md3c1e::{cart_components, e_table, ferric_cart2sph, prim_norm, MAX_L};
use ferric_integrals::operator::Operator;
use ferric_scf::fock::{JBuilder, KBuilder};
use ndarray::linalg::general_mat_mul;
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use ndarray_linalg::{Eigh, UPLO};
use num_complex::Complex64;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

pub mod auto_omega;
pub(crate) mod deriv;
pub mod kpoint;
mod split;
pub(crate) mod strain;

pub use split::{
    sr_triplet_estimate, sr_walk_counts, RangeSplit, RangeSplitMutant, SrWalkCounts,
    DEFAULT_RANGE_SPLIT_LAMBDA, RANGE_SPLIT_NEG_EIG_GUARD,
};

pub use deriv::RsGdfFitDiagnostics;
pub(crate) use split::{split_g0, SplitG0};

/// Default Ewald split for the RS-GDF build (Bohr⁻¹; the prototype's `w=1`).
pub const DEFAULT_RSGDF_OMEGA: f64 = 1.0;
/// Default per-term truncation target (the prototype's `prec`).
pub const DEFAULT_RSGDF_PRECISION: f64 = 1e-13;
/// Default absolute metric-eigenvalue cut (PySCF `LINEAR_DEP_THR`).
pub const DEFAULT_RSGDF_LINDEP: f64 = 1e-10;
/// Default hard cap for [`RsGdf::fitted_eri`] (test/diagnostic `nao⁴`).
pub const DEFAULT_FITTED_ERI_MAX_BYTES: usize = 512 << 20;

/// libint2 engine precision: effectively off — our own distance screen
/// decides what is computed.
const ENGINE_PRECISION: f64 = 1e-20;
/// Extra Bohr on every derived real-space radius (l > 0 polynomial factors).
const SR_MARGIN_BOHR: f64 = 2.0;
/// Hard cap on the integer box one lattice-sphere visit may enumerate.
const MAX_BOX_POINTS: u64 = 50_000_000;

/// How the G = 0 term is removed. Production is ALWAYS
/// [`G0Handling::Consistent`]; the others are deliberately BROKEN variants
/// that exist only as negative controls for the exactness anchor (an anchor
/// that cannot tell them apart from `Consistent` certifies nothing).
/// Prototype measurements on the anchor (FINDINGS "Iteration 2"): metric-only
/// max|ΔI| 5.3, neither 6.1e-2, consistent 1.6e-12.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum G0Handling {
    /// Remove `c0 q qᵀ` from J2 AND `c0 S qᵀ` from J3 (PySCF RSGDF).
    Consistent,
    /// MUTATION: remove it from the metric only (one-sided).
    MetricOnlyMutant,
    /// MUTATION: remove it from neither (fits a different, ω-dependent kernel).
    NeitherMutant,
}

/// Settings for [`RsGdf::build`].
#[derive(Debug, Clone, Copy)]
pub struct RsGdfConfig {
    /// Ewald split (Bohr⁻¹). Any ω > 0 gives the same B Bᵀ up to truncation.
    pub omega: f64,
    /// Per-term truncation target, in `(0, 1)`.
    pub precision: f64,
    /// Absolute metric-eigenvalue cut (eigenvalues `<= lindep` are dropped).
    pub lindep: f64,
    /// Exchange divergence treatment applied by [`RsGdfK`].
    pub exxdiv: ExxDiv,
    /// Per-triplet SR screen (`true`, production) or the global image radius
    /// for every triplet (`false`, the prototype's unscreened form).
    pub sr_screen: bool,
    /// Memory budget in bytes (`None` = ferric's unified budget).
    pub budget_bytes: Option<usize>,
    /// G = 0 handling — [`G0Handling::Consistent`] except in mutation tests.
    pub g0: G0Handling,
    /// Opt-in primitive-level range split of the SR sums (`split` module
    /// doc; Gamma energy, forces and stress). `None` = today's construction,
    /// bit for bit. The k-point build refuses it.
    pub range_split: Option<RangeSplit>,
    /// Column rotation of generally contracted orbital shells inside the
    /// Gamma SR 3-centre walk (module doc "Column rotation"). Default
    /// [`SrColumnRotation::Auto`]: on in the Gamma energy builds, off in
    /// [`RsGdf::build_for_gradient`], the s1 oracle and the k-point build
    /// (each refuses an explicit `On`). `Off` = the unrotated construction,
    /// bit for bit.
    pub sr_column_rotation: SrColumnRotation,
}

impl Default for RsGdfConfig {
    /// Defaults: ω = [`DEFAULT_RSGDF_OMEGA`], [`DEFAULT_RSGDF_PRECISION`],
    /// [`DEFAULT_RSGDF_LINDEP`], Ewald exchange divergence, per-triplet SR screening, ferric's
    /// unified budget, consistent G = 0, no range split, automatic column rotation.
    fn default() -> Self {
        Self {
            omega: DEFAULT_RSGDF_OMEGA,
            precision: DEFAULT_RSGDF_PRECISION,
            lindep: DEFAULT_RSGDF_LINDEP,
            exxdiv: ExxDiv::Ewald,
            sr_screen: true,
            budget_bytes: None,
            g0: G0Handling::Consistent,
            range_split: None,
            sr_column_rotation: SrColumnRotation::Auto,
        }
    }
}

impl RsGdfConfig {
    /// Rejects a non-finite or non-positive `omega`, a `precision` outside `(0, 1)` and a negative
    /// or non-finite `lindep` with [`FerricError::General`].
    fn validate(&self) -> Result<(), FerricError> {
        if !(self.omega > 0.0) || !self.omega.is_finite() {
            return Err(FerricError::General(format!(
                "RsGdf: omega must be finite and > 0, got {}",
                self.omega
            )));
        }
        if !(f64::MIN_POSITIVE..1.0).contains(&self.precision) {
            return Err(FerricError::General(format!(
                "RsGdf: precision must lie in (0, 1), got {}",
                self.precision
            )));
        }
        if !(self.lindep >= 0.0) || !self.lindep.is_finite() {
            return Err(FerricError::General(format!(
                "RsGdf: lindep must be finite and >= 0, got {}",
                self.lindep
            )));
        }
        Ok(())
    }
}

/// What the build did (counts and conditioning; the only cost statements).
#[derive(Debug, Clone)]
pub struct RsGdfStats {
    /// Aux functions.
    pub naux: usize,
    /// Metric eigenvectors kept (`naux − n_dropped`) = rows of B.
    pub naux_kept: usize,
    /// Metric eigenvalues `<= lindep` dropped.
    pub n_dropped: usize,
    /// Smallest / largest metric eigenvalue.
    pub metric_eig_min: f64,
    pub metric_eig_max: f64,
    /// The ω and precision used.
    pub omega: f64,
    pub precision: f64,
    /// Pair images `L` visited.
    pub n_pair_images: usize,
    /// Shifted 3-centre shell triplets computed (SR; each unordered shell
    /// pair once at Gamma, module doc "Orbital-pair symmetry").
    pub n_sr3_triplets: usize,
    /// `n_sr3_triplets` in ordered-pair units: off-diagonal pairs weighted
    /// 2, diagonal 1. Unsplit, this is the pre-s2 ordered-walk count (the
    /// number to compare benchmarks on); with a range split see
    /// [`SrWalkCounts::n_sr3_triplets_s1`].
    pub n_sr3_triplets_ordered: usize,
    /// Shifted 2-centre shell pairs computed (SR metric).
    pub n_sr2_pairs: usize,
    /// Half-sphere G vectors (LR) and the `pair_ft` chunks they used.
    pub n_g_half: usize,
    pub n_g_chunks: usize,
    /// max |J2 − J2ᵀ| and max |J3\[μν\] − J3\[νμ\]| before symmetrisation
    /// (truncation / image-set defects show up here). The Gamma SR 3-centre
    /// part is exactly symmetric by construction (s2), so `asym_j3` measures
    /// the LR and G = 0 parts only.
    pub asym_j2: f64,
    pub asym_j3: f64,
    /// Resolved budget and the bytes reserved before the LR chunks.
    pub budget_bytes: usize,
    pub resident_bytes: usize,
}

/// The fitted three-index tensor and what the K builder needs.
#[derive(Debug, Clone)]
pub struct RsGdf {
    nao: usize,
    /// `(naux_kept, nao²)`, row k, column `μ·nao+ν`; each row symmetric in (μ,ν).
    b: Array2<f64>,
    s: Array2<f64>,
    madelung: f64,
    stats: RsGdfStats,
    /// `W = U_kept s_kept^{-1/2}`, `(naux, naux_kept)`: the aux-space map of
    /// B's rows (B = Wᵀ J3ᵀ). Retained for the dRPA driver's
    /// `RpaIntermediates::v_inv_sqrt` (eigenpotential back-transform only;
    /// never part of an energy). Counted by the build ledger's metric line.
    metric_inv_sqrt: Array2<f64>,
    /// Retained only by [`RsGdf::build_for_gradient`] (the forces' metric
    /// eigen-data); `None` on every energy-only build.
    grad: Option<MetricGradParts>,
    /// Build stage timings and counters ([`crate::timing`]).
    timings: PbcTimings,
    /// Accumulated [`RsGdfJ`] / [`RsGdfK`] build calls (every builder
    /// borrowing this B, i.e. every SCF stage run on it).
    j_clock: CallClock,
    k_clock: CallClock,
    /// The build's [`RsGdfConfig::range_split`] (the derivative walks
    /// rebuild the same partition from it).
    range_split: Option<RangeSplit>,
}

/// The pre-solve pieces of an RS-GDF build ([`RsGdf::build_with_fit_parts`]),
/// for per-pair domain-local fits in the PERIODIC metric: with the full aux
/// set as the domain and the same eig/`lindep` pseudo-inverse,
/// `A_i J2⁺ A_jᵀ` reproduces `B_iᵀ B_j` (the trivial-radius anchor,
/// `tests/pbc_lmp2.rs`).
#[derive(Debug, Clone)]
pub struct PeriodicFitParts {
    /// Symmetrised `(P|Q)'`, `(naux, naux)` — exactly the matrix the build's
    /// metric solve decomposed.
    pub j2: Array2<f64>,
    /// Symmetrised `(μν|P)'`, `(nao², naux)`, row `μ·nao+ν`.
    pub j3: Array2<f64>,
    /// Centre of every aux function (its shell's centre), Bohr, as placed in
    /// the cell (NOT wrapped: distances to it must be minimum-image).
    pub aux_centers: Vec<[f64; 3]>,
    /// The absolute eigenvalue cut the build used.
    pub lindep: f64,
}

/// Centre of every basis function of `prep` (its shell's centre).
fn aux_function_centers(prep: &PreparedBasis) -> Vec<[f64; 3]> {
    let mut out = vec![[0.0; 3]; prep.nbasis()];
    let offs = prep.shell_offsets();
    let dims = prep.shell_dims();
    for (sh, ls) in prep.located_shells().iter().enumerate() {
        for f in 0..dims[sh] {
            out[offs[sh] + f] = ls.center;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Shell data (ferric normalisation: prim_norm folded in, CCA order, ferric
// cart→sph for pure l >= 2 — the conventions of `pair_ft`).
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct GShell {
    l: usize,
    pure: bool,
    center: [f64; 3],
    exps: Vec<f64>,
    /// Contraction coefficients WITH `prim_norm(a, l)` folded in.
    coefs: Vec<f64>,
    nfun: usize,
    off: usize,
    amin: f64,
    amax: f64,
    /// `max_k |c_k| (π/α_k)^{3/2} (1 + α_k^{−1/2})^l` (aux screening).
    qbound: f64,
}

/// Per-shell Rust data of `prep` (ferric normalisation: `prim_norm` folded into the coefficients).
/// Errors on `l > MAX_L`, a pure p shell, malformed or non-positive exponents, or a function count
/// that disagrees with libint2; `who` prefixes the message.
fn gshells(prep: &PreparedBasis, who: &str) -> Result<Vec<GShell>, FerricError> {
    let dims = prep.shell_dims();
    let offs = prep.shell_offsets();
    let mut out = Vec::with_capacity(prep.nshells());
    for (s, sh) in prep.located_shells().iter().enumerate() {
        if sh.l < 0 || sh.l as usize > MAX_L {
            return Err(FerricError::Basis(format!(
                "{who}: shell {s} has l={} (supported 0..={MAX_L})",
                sh.l
            )));
        }
        let l = sh.l as usize;
        if sh.pure && l == 1 {
            return Err(FerricError::Basis(format!(
                "{who}: shell {s} is a pure p shell (libint2 orders it y,z,x); unsupported"
            )));
        }
        if sh.exponents.is_empty() || sh.exponents.len() != sh.coefficients.len() {
            return Err(FerricError::Basis(format!(
                "{who}: shell {s} has {} exponents and {} coefficients",
                sh.exponents.len(),
                sh.coefficients.len()
            )));
        }
        if sh.exponents.iter().any(|&a| !(a > 0.0) || !a.is_finite()) {
            return Err(FerricError::Basis(format!(
                "{who}: shell {s} has a non-positive or non-finite exponent"
            )));
        }
        let pure = sh.pure && l >= 2;
        let ncart = (l + 1) * (l + 2) / 2;
        let nfun = if pure { 2 * l + 1 } else { ncart };
        if nfun != dims[s] {
            return Err(FerricError::Basis(format!(
                "{who}: shell {s} (l={l}, pure={pure}) has {nfun} functions but libint2 reports {}",
                dims[s]
            )));
        }
        let coefs: Vec<f64> = sh
            .exponents
            .iter()
            .zip(sh.coefficients)
            .map(|(&a, &c)| c * prim_norm(a, l))
            .collect();
        let qbound = sh
            .exponents
            .iter()
            .zip(&coefs)
            .map(|(&a, &c)| c.abs() * (PI / a).powf(1.5) * (1.0 + a.powf(-0.5)).powi(l as i32))
            .fold(0.0_f64, f64::max);
        out.push(GShell {
            l,
            pure,
            center: sh.center,
            exps: sh.exponents.to_vec(),
            coefs,
            nfun,
            off: offs[s],
            amin: sh.exponents.iter().copied().fold(f64::INFINITY, f64::min),
            amax: sh.exponents.iter().copied().fold(0.0_f64, f64::max),
            qbound,
        });
    }
    Ok(out)
}

/// Euclidean dot product of two Cartesian 3-vectors.
fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Euclidean norm of a Cartesian 3-vector.
fn norm3(a: &[f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

/// Distance from `x` to the segment `[a, b]`.
fn segment_distance(x: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ax = [x[0] - a[0], x[1] - a[1], x[2] - a[2]];
    let l2 = dot3(&ab, &ab);
    let t = if l2 > 0.0 {
        (dot3(&ax, &ab) / l2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    norm3(&[ax[0] - t * ab[0], ax[1] - t * ab[1], ax[2] - t * ab[2]])
}

// ---------------------------------------------------------------------------
// One-centre aux FT
// ---------------------------------------------------------------------------

/// Fourier transform of every aux function, `X[P, g] = ∫ χ_P(r) e^{−iG_g·r} dr`,
/// shape `(nbasis, gvecs.len())`, in the AO order and normalisation libint2
/// uses for `aux` (the [`mod@crate::pair_ft`] conventions: `prim_norm` folded in,
/// CCA Cartesian order, `ferric_cart2sph` for pure l >= 2). Single-Gaussian
/// case of the pair FT: `FT[x^i e^{−a x²}](G) = (π/a)^{1/2} e^{−G²/4a}
/// Σ_t E_t^{i0}(a, 0) (−iG)^t` per direction, times the phase `e^{−iG·A}`.
/// `aux` may sit on any centres (atoms, ghosts, `SiteBasis` sites).
/// `X[:, g]` at `G = 0` is the charge `q_P = ∫ χ_P`.
pub fn aux_ft(aux: &PreparedBasis, gvecs: &[[f64; 3]]) -> Result<Array2<Complex64>, FerricError> {
    if gvecs.iter().flatten().any(|v| !v.is_finite()) {
        return Err(FerricError::General("aux_ft: non-finite G vector".into()));
    }
    let shells = gshells(aux, "aux_ft")?;
    Ok(aux_ft_shells(&shells, aux.nbasis(), gvecs))
}

/// [`aux_ft`] on prepared shells, parallel over aux SHELLS: each shell's
/// rows are computed from zero by [`aux_ft_shell_rows`] (no element depends on
/// another shell) and copied in shell order, so the result is bit for bit the
/// serial shell loop's at any thread count.
fn aux_ft_shells(shells: &[GShell], naux: usize, gvecs: &[[f64; 3]]) -> Array2<Complex64> {
    let ng = gvecs.len();
    let rows: Vec<Vec<Complex64>> = shells
        .par_iter()
        .map(|sh| aux_ft_shell_rows(sh, gvecs))
        .collect();
    let mut out = Array2::<Complex64>::zeros((naux, ng));
    for (sh, rows) in shells.iter().zip(&rows) {
        for j in 0..sh.nfun {
            for g in 0..ng {
                out[[sh.off + j, g]] = rows[j * ng + g];
            }
        }
    }
    out
}

/// Rows `(nfun, ng)` (row-major) of one aux shell's FT (the body of the
/// pre-parallel serial shell loop, expression for expression).
fn aux_ft_shell_rows(sh: &GShell, gvecs: &[[f64; 3]]) -> Vec<Complex64> {
    let ng = gvecs.len();
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let mut ebuf = vec![0.0_f64; (MAX_L + 1) * (MAX_L + 1)];
    let l = sh.l;
    let comps = cart_components(l);
    let ncart = comps.len();
    let mut cart = vec![zero; ncart * ng];
    for (&a, &c) in sh.exps.iter().zip(&sh.coefs) {
        // E^{i0}_t at index i*(l+1) + t (lb = 0).
        e_table(l, 0, a, 0.0, 0.0, &mut ebuf);
        let pref = c * (PI / a).powf(1.5);
        for (g, gv) in gvecs.iter().enumerate() {
            let mag = pref * (-dot3(gv, gv) / (4.0 * a)).exp();
            let ph = dot3(gv, &sh.center);
            // e^{−iG·A}
            let common = Complex64::new(mag * ph.cos(), -mag * ph.sin());
            let mut f = [[zero; MAX_L + 1]; 3];
            for (d, fd) in f.iter_mut().enumerate() {
                let step = Complex64::new(0.0, -gv[d]);
                for (i, fdi) in fd.iter_mut().enumerate().take(l + 1) {
                    let mut acc = zero;
                    let mut pw = one;
                    for t in 0..=i {
                        acc += pw * ebuf[i * (l + 1) + t];
                        pw *= step;
                    }
                    *fdi = acc;
                }
            }
            for (u, lc) in comps.iter().enumerate() {
                cart[u * ng + g] +=
                    common * f[0][lc[0] as usize] * f[1][lc[1] as usize] * f[2][lc[2] as usize];
            }
        }
    }
    if !sh.pure {
        return cart;
    }
    let c2s = ferric_cart2sph(l);
    let nf = sh.nfun;
    let mut rows = vec![zero; nf * ng];
    for j in 0..nf {
        for g in 0..ng {
            let mut acc = zero;
            for m in 0..ncart {
                let cm = c2s[m * nf + j];
                if cm != 0.0 {
                    acc += cart[m * ng + g] * cm;
                }
            }
            rows[j * ng + g] = acc;
        }
    }
    rows
}

// ---------------------------------------------------------------------------
// G = 0 bookkeeping — the ONE place the convention lives.
// ---------------------------------------------------------------------------

/// Remove the erfc real-space sums' implicit G = 0 term
/// `c0 = π/(ω²Ω)`: `J2 −= c0 q qᵀ` and `J3[μν,P] −= c0 S_μν q_P`
/// (`s_flat` = S raveled `μ·nao+ν`). `mode` selects the production form or a
/// negative-control mutant ([`G0Handling`]).
pub fn subtract_g0(
    j2: &mut Array2<f64>,
    j3: &mut Array2<f64>,
    s_flat: &[f64],
    q: &[f64],
    c0: f64,
    mode: G0Handling,
) {
    let (do_metric, do_three) = match mode {
        G0Handling::Consistent => (true, true),
        G0Handling::MetricOnlyMutant => (true, false),
        G0Handling::NeitherMutant => (false, false),
    };
    if do_metric {
        for (p, qp) in q.iter().enumerate() {
            for (r, qr) in q.iter().enumerate() {
                j2[(p, r)] -= c0 * qp * qr;
            }
        }
    }
    if do_three {
        for (mn, smn) in s_flat.iter().enumerate() {
            for (p, qp) in q.iter().enumerate() {
                j3[(mn, p)] -= c0 * smn * qp;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Lattice-sphere walker
// ---------------------------------------------------------------------------

/// Enumerates lattice vectors `T` with `|T − x0| <= r` by an integer box:
/// `T·b_j = 2π n_j`, so `|n_j − x0·b_j/2π| <= r |b_j|/2π`.
///
/// For a strained cell ([`Cell::strained`]) the selection is made in the
/// REFERENCE frame (`x0` mapped by fractional coordinates, distances in the
/// reference lattice) and the selected `T` is built from the strained
/// lattice — the frozen-index convention of `crate::lattice`, so the SR walks
/// do not re-select images under strain.
struct LatticeWalker {
    /// Selection lattice (the reference's for a strained cell).
    a: [[f64; 3]; 3],
    b: [[f64; 3]; 3],
    /// The strained cell itself when its index sets are frozen (`None`: the
    /// selection lattice IS the output lattice, bit for bit as before).
    frozen: Option<Cell>,
}

impl LatticeWalker {
    /// Walker over `cell`'s lattice; a strained cell selects images in its reference (index) frame,
    /// see the type doc.
    fn new(cell: &Cell) -> Self {
        match cell.index_reference() {
            None => Self {
                a: *cell.lattice(),
                b: cell.reciprocal(),
                frozen: None,
            },
            Some(r) => Self {
                a: *r.lattice(),
                b: r.reciprocal(),
                frozen: Some(cell.clone()),
            },
        }
    }

    /// Calls `f(T)` for every lattice vector `T` (Bohr, Cartesian, output lattice) with `|T − x0|
    /// <= r`, in ascending `(n0, n1, n2)` order. `x0` and `r` are in Bohr (`x0` in the output
    /// Cartesian frame). Errors on a non-finite centre or radius, or when the integer box exceeds
    /// the size caps; errors from `f` abort the walk.
    fn visit<F>(&self, x0: [f64; 3], r: f64, mut f: F) -> Result<(), FerricError>
    where
        F: FnMut([f64; 3]) -> Result<(), FerricError>,
    {
        if !(r >= 0.0) || !r.is_finite() || x0.iter().any(|v| !v.is_finite()) {
            return Err(FerricError::General(format!(
                "RsGdf lattice walk: bad centre {x0:?} / radius {r}"
            )));
        }
        let x0 = match &self.frozen {
            Some(c) => c.to_index_frame(x0),
            None => x0,
        };
        let tp = 2.0 * PI;
        let mut lo = [0i64; 3];
        let mut hi = [0i64; 3];
        let mut count: u64 = 1;
        for j in 0..3 {
            let fj = dot3(&x0, &self.b[j]) / tp;
            let wj = r * norm3(&self.b[j]) / tp;
            if !(wj < 1e6) || fj.abs() > 1e9 {
                return Err(FerricError::General(format!(
                    "RsGdf lattice walk: radius {r} needs {wj:.3e} cells along b_{j}"
                )));
            }
            lo[j] = (fj - wj).floor() as i64;
            hi[j] = (fj + wj).ceil() as i64;
            count = count.saturating_mul((hi[j] - lo[j] + 1) as u64);
        }
        if count > MAX_BOX_POINTS {
            return Err(FerricError::General(format!(
                "RsGdf lattice walk: radius {r} would enumerate {count} lattice triples \
                 (cap {MAX_BOX_POINTS}); raise precision or omega"
            )));
        }
        let a = &self.a;
        let r2 = r * r;
        for n0 in lo[0]..=hi[0] {
            for n1 in lo[1]..=hi[1] {
                for n2 in lo[2]..=hi[2] {
                    let (f0, f1, f2) = (n0 as f64, n1 as f64, n2 as f64);
                    let t = [
                        f0 * a[0][0] + f1 * a[1][0] + f2 * a[2][0],
                        f0 * a[0][1] + f1 * a[1][1] + f2 * a[2][1],
                        f0 * a[0][2] + f1 * a[1][2] + f2 * a[2][2],
                    ];
                    let d = [t[0] - x0[0], t[1] - x0[1], t[2] - x0[2]];
                    if dot3(&d, &d) <= r2 {
                        match &self.frozen {
                            None => f(t)?,
                            Some(c) => f(c.translation_from_index([n0, n1, n2]))?,
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SR bounds
// ---------------------------------------------------------------------------

/// s-type charge bound of a (shell, shell) pair at separation² `r2`, with the
/// smallest and largest pair exponent.
fn pair_bound(a: &GShell, b: &GShell, r2: f64) -> (f64, f64, f64) {
    let (mut q, mut pmin, mut pmax) = (0.0_f64, f64::INFINITY, 0.0_f64);
    for (&ea, &ca) in a.exps.iter().zip(&a.coefs) {
        for (&eb, &cb) in b.exps.iter().zip(&b.coefs) {
            let p = ea + eb;
            q = q.max((ca * cb).abs() * (PI / p).powf(1.5) * (-ea * eb / p * r2).exp());
            pmin = pmin.min(p);
            pmax = pmax.max(p);
        }
    }
    (q, pmin, pmax)
}

/// Radius (Bohr, incl. margin) beyond which the erfc interaction of two
/// Gaussian distributions with charge bounds `qa`, `qb` and exponent ranges
/// `[amin, amax]`, `[bmin, bmax]` is below `thresh`; `None` if it is below
/// everywhere (module doc, "Screening").
#[allow(clippy::too_many_arguments)]
fn sr_radius(
    qa: f64,
    qb: f64,
    amin: f64,
    amax: f64,
    bmin: f64,
    bmax: f64,
    omega: f64,
    thresh: f64,
) -> Option<f64> {
    let mu = (amax * bmax / (amax + bmax)).sqrt();
    let nu = 1.0 / (1.0 / amin + 1.0 / bmin + 1.0 / (omega * omega)).sqrt();
    let pref = qa * qb * (1.0 + 2.0 * mu / PI.sqrt());
    if !(pref > thresh) {
        return None;
    }
    Some((pref / thresh).ln().sqrt() / nu + SR_MARGIN_BOHR)
}

/// Largest `|m[i,j] − m[j,i]|` over the strict lower triangle (the asymmetry of a nominally
/// symmetric matrix). Assumes `m` is square.
fn max_abs_asym(m: &Array2<f64>) -> f64 {
    let mut w = 0.0_f64;
    for i in 0..m.nrows() {
        for j in 0..i {
            w = w.max((m[(i, j)] - m[(j, i)]).abs());
        }
    }
    w
}

// ---------------------------------------------------------------------------
// The build
// ---------------------------------------------------------------------------

/// Everything the SR/LR stages share.
struct Stage<'a> {
    cell: &'a Cell,
    obs: &'a PreparedBasis,
    aux: &'a PreparedBasis,
    obs_sh: Vec<GShell>,
    aux_sh: Vec<GShell>,
    omega: f64,
    thresh: f64,
    sr_screen: bool,
    walker: LatticeWalker,
}

/// Sum per-pair counts in pair order; the first error (in pair order, so
/// deterministic) is returned instead.
fn sum_counts_in_pair_order(counts: Vec<Result<usize, FerricError>>) -> Result<usize, FerricError> {
    let mut count = 0usize;
    for c in counts {
        count += c?;
    }
    Ok(count)
}

/// Which Gamma SR 3-centre walk a build runs (module doc "Orbital-pair
/// symmetry"). Production is ALWAYS [`PairSym::S2`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PairSym {
    /// Each unordered shell pair once, both rows written (production).
    S2,
    /// FROZEN pre-s2 construction: every ordered pair, averaged afterwards
    /// by `symmetrize_pairs` ([`RsGdf::build_pair_s1_oracle`] only).
    S1Oracle,
}

/// Unordered shell pairs `(i1, i2)`, `i1 <= i2`, row-major: the Gamma s2
/// task list (module doc "Orbital-pair symmetry").
pub(crate) fn unordered_pairs(nsh: usize) -> Vec<(usize, usize)> {
    (0..nsh)
        .flat_map(|i| (i..nsh).map(move |j| (i, j)))
        .collect()
}

/// Sum s2 per-pair counts in pair order: `(computed, ordered-equivalent)`,
/// the latter weighting an off-diagonal pair 2 (it stands for both
/// orders) and a diagonal pair 1. The first error in pair order is
/// returned instead.
fn sum_s2_counts(
    pairs: &[(usize, usize)],
    counts: Vec<Result<usize, FerricError>>,
) -> Result<(usize, usize), FerricError> {
    let (mut n, mut n_ordered) = (0usize, 0usize);
    for (&(i1, i2), c) in pairs.iter().zip(counts) {
        let c = c?;
        n += c;
        n_ordered += if i1 == i2 { c } else { 2 * c };
    }
    Ok((n, n_ordered))
}

/// COPY one unordered pair's finished Gamma block into BOTH row `μν` and row
/// `νμ` of `j3` (`(nao², naux)`, `n` = nao; s2, module doc). `acc[(i nb + j)
/// naux + P]` belongs to `μ = a.off + i`, `ν = b.off + j`. A diagonal pair
/// (`same`) writes only its function pairs `i <= j` (each into both rows), so
/// every output row receives exactly one value from exactly one task.
fn copy_pair_rows_mirrored(
    j3: &mut Array2<f64>,
    n: usize,
    acc: &[f64],
    a: &GShell,
    b: &GShell,
    same: bool,
) {
    let naux = j3.ncols();
    let (na, nb) = (a.nfun, b.nfun);
    for i in 0..na {
        for j in 0..nb {
            if same && i > j {
                continue;
            }
            let src = &acc[(i * nb + j) * naux..(i * nb + j + 1) * naux];
            for row in [(a.off + i) * n + b.off + j, (b.off + j) * n + a.off + i] {
                j3.row_mut(row)
                    .iter_mut()
                    .zip(src)
                    .for_each(|(d, &s)| *d = s);
            }
        }
    }
}

/// Residue binning of the SR sums: pair images `L` by `n(L) mod mod_l`, aux
/// images `T` by `n(T) mod mod_t` (`n` = integer lattice coordinates; the
/// k-point RS-GDF, [`kpoint`]). Bin `(r_L, r_T)` is `r_L · R_T + r_T`. The
/// Gamma build is the single bin [`SrBinning::GAMMA`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SrBinning {
    pub(crate) mod_l: [usize; 3],
    pub(crate) mod_t: [usize; 3],
}

impl SrBinning {
    /// One bin: the Gamma sums.
    pub(crate) const GAMMA: Self = Self {
        mod_l: [1; 3],
        mod_t: [1; 3],
    };

    /// `R_L`.
    fn n_l(&self) -> usize {
        self.mod_l.iter().product()
    }

    /// `R_T`.
    fn n_t(&self) -> usize {
        self.mod_t.iter().product()
    }

    /// Residue of lattice vector `x` modulo `moduli` (0 for a single bin,
    /// without evaluating coordinates). The same function the serial binned
    /// walks apply (`residue_index(lattice_coords(b, x), moduli)`).
    fn residue(b: &[[f64; 3]; 3], x: &[f64; 3], moduli: [usize; 3]) -> usize {
        if moduli == [1; 3] {
            0
        } else {
            residue_index(lattice_coords(b, x), moduli)
        }
    }
}

/// The loop invariants of one [`Stage::sr_three_index_binned`] call.
struct Sr3Ctx<'a> {
    pool: &'a EnginePool,
    images: &'a [[f64; 3]],
    /// `r_L` of each image (same order as `images`).
    l_bin: &'a [usize],
    /// Reciprocal lattice (residue of `T`).
    recip: [[f64; 3]; 3],
    bins: SrBinning,
    global: f64,
    /// `R_L · R_T` zeroed `(nao², naux)` bins every task copies into.
    out: &'a Mutex<Vec<Array2<f64>>>,
}

/// The loop invariants of one [`Stage::sr_metric_binned`] call.
struct Sr2Ctx<'a> {
    pool: &'a EnginePool,
    recip: [[f64; 3]; 3],
    mod_t: [usize; 3],
    global: f64,
    /// `R_T` zeroed `(naux, naux)` bins every task copies into.
    out: &'a Mutex<Vec<Array2<f64>>>,
}

impl<'a> Stage<'a> {
    /// The same stage (cell, ω, precision, screen mode) on other orbital and
    /// aux bases with the parent's AO layout (the column-rotated SR walk,
    /// module doc "Column rotation").
    fn with_bases<'b>(
        &self,
        obs: &'b PreparedBasis,
        aux: &'b PreparedBasis,
    ) -> Result<Stage<'b>, FerricError>
    where
        'a: 'b,
    {
        Ok(Stage {
            cell: self.cell,
            obs,
            aux,
            obs_sh: gshells(obs, "RsGdf rotated orbital basis")?,
            aux_sh: gshells(aux, "RsGdf aux basis")?,
            omega: self.omega,
            thresh: self.thresh,
            sr_screen: self.sr_screen,
            walker: LatticeWalker::new(self.cell),
        })
    }
}

impl Stage<'_> {
    /// [`sr_radius`] for charge bounds `qa`, `qb` and `(min, max)` exponent ranges `a`, `b`, at
    /// this stage's `omega` and `thresh`.
    fn radius(&self, qa: f64, qb: f64, a: (f64, f64), b: (f64, f64)) -> Option<f64> {
        sr_radius(qa, qb, a.0, a.1, b.0, b.1, self.omega, self.thresh)
    }

    /// SR metric `Σ_T (P_0 | Q_T)_erfc` (unsymmetrised) and the pair count:
    /// the single bin of [`Stage::sr_metric_binned`].
    fn sr_metric(&self) -> Result<(Array2<f64>, usize), FerricError> {
        let (mut bins, count) = self.sr_metric_binned(SrBinning::GAMMA.mod_t)?;
        Ok((bins.swap_remove(0), count))
    }

    /// SR metric binned by the residue of `T` modulo `mod_t`: `bins[r]` =
    /// `Σ_{T ≡ r} (P_0|Q_T)_erfc` (reals), and the pair count.
    ///
    /// PARALLEL over ordered aux shell pairs `(P, Q)`, BIT-IDENTICAL to the
    /// serial `P → Q → T` walk ([`Stage::sr_metric_each`]) and across thread
    /// counts: element `(r, p, q)` receives addends ONLY from the shell pair
    /// owning `(p, q)`, in walker `T` order (the bin only filters). The
    /// serial walk is already pair-outer, so each task runs exactly the
    /// serial per-pair body ([`Stage::sr2_pair`]) into a zeroed per-pair
    /// scratch `(R_T, nP, nQ)` and COPIES it into the zeroed bins. One erfc
    /// 2-centre engine per rayon worker; counts are integer sums; errors are
    /// returned in pair order.
    fn sr_metric_binned(
        &self,
        mod_t: [usize; 3],
    ) -> Result<(Vec<Array2<f64>>, usize), FerricError> {
        let naux = self.aux.nbasis();
        let nsh = self.aux_sh.len();
        let rt: usize = mod_t.iter().product();
        let pool = EnginePool::from_fn(|| {
            Engine::new_2center(Operator::erfc(self.omega), self.aux, ENGINE_PRECISION)
        })?;
        let out = Mutex::new(
            (0..rt)
                .map(|_| Array2::<f64>::zeros((naux, naux)))
                .collect::<Vec<_>>(),
        );
        let ctx = Sr2Ctx {
            pool: &pool,
            recip: self.cell.reciprocal(),
            mod_t,
            global: self.sr2_global_radius(),
            out: &out,
        };
        let counts: Vec<Result<usize, FerricError>> = (0..nsh * nsh)
            .into_par_iter()
            .map(|pair| self.sr2_pair_task(&ctx, (pair / nsh, pair % nsh)))
            .collect();
        let count = sum_counts_in_pair_order(counts)?;
        Ok((out.into_inner().unwrap_or_else(|e| e.into_inner()), count))
    }

    /// One [`Stage::sr_metric_binned`] task: aux pair `(ip, iq)` over every
    /// kept `T` into a zeroed `(R_T, nP, nQ)` scratch, then COPIED.
    fn sr2_pair_task(
        &self,
        ctx: &Sr2Ctx<'_>,
        (ip, iq): (usize, usize),
    ) -> Result<usize, FerricError> {
        let (p, q) = (&self.aux_sh[ip], &self.aux_sh[iq]);
        let bl = p.nfun * q.nfun;
        let rt: usize = ctx.mod_t.iter().product();
        let mut acc = vec![0.0_f64; rt * bl];
        let mut count = 0usize;
        ctx.pool.with(|eng| -> Result<(), FerricError> {
            let mut visit = |ip: usize, iq: usize, t: [f64; 3]| -> Result<(), FerricError> {
                let blk = eng.compute_eri2_shifted(self.aux, ip, iq, t)?;
                let r = SrBinning::residue(&ctx.recip, &t, ctx.mod_t);
                // == j2[r][(p.off + i, q.off + j)] += blk[i nQ + j]
                for (d, &s) in acc[r * bl..(r + 1) * bl].iter_mut().zip(&blk[..bl]) {
                    *d += s;
                }
                Ok(())
            };
            self.sr2_pair(ip, iq, ctx.global, &mut count, &mut visit)
        })?;
        if count > 0 {
            let mut bins = ctx.out.lock().unwrap_or_else(|e| e.into_inner());
            for (r, j2) in bins.iter_mut().enumerate() {
                for i in 0..p.nfun {
                    for j in 0..q.nfun {
                        j2[(p.off + i, q.off + j)] = acc[r * bl + i * q.nfun + j];
                    }
                }
            }
        }
        Ok(count)
    }

    /// Bytes of ONE parallel SR task's scratch under `bins`: the larger of
    /// the 3-centre `R_T nfun_max² naux` and the metric `R_T nfun_aux,max²`
    /// blocks (the two walks never run at the same time).
    fn sr_scratch_bytes_per_task(&self, bins: SrBinning) -> usize {
        let rt = bins.n_t() as u64;
        let nf = self.obs_sh.iter().map(|s| s.nfun).max().unwrap_or(0) as u64;
        let nq = self.aux_sh.iter().map(|s| s.nfun).max().unwrap_or(0) as u64;
        let j3 = rt
            .saturating_mul(nf.saturating_mul(nf))
            .saturating_mul(self.aux.nbasis() as u64);
        let j2 = rt.saturating_mul(nq.saturating_mul(nq));
        bytes_of(j3.max(j2), 8)
    }

    /// CHECK (not reserve) the transient per-thread scratch of the parallel
    /// SR walks: one task's scratch per possible concurrent task (threads +
    /// the spare slot). Call it after every numerical width (LR chunk
    /// budget) is fixed, so no width ever depends on the thread count.
    fn check_sr_scratch(
        &self,
        ledger: &Ledger,
        who: &str,
        bins: SrBinning,
    ) -> Result<(), FerricError> {
        let threads = rayon::current_num_threads();
        ledger.check(
            &format!(
                "{who} SR per-thread scratch ({threads} threads, R_T = {})",
                bins.n_t()
            ),
            self.sr_scratch_bytes_per_task(bins)
                .saturating_mul(threads.saturating_add(1)),
        )
    }

    /// Every SR metric block `(P_0 | Q_T)_erfc` the screen keeps, in a fixed
    /// order: `sink(P shell, Q shell, T, block (nP, nQ))`. Returns the count.
    /// SERIAL: only the frozen serial oracles of [`kpoint`] use it (the
    /// production sums are the parallel [`Stage::sr_metric_binned`]).
    fn sr_metric_each<F>(&self, mut sink: F) -> Result<usize, FerricError>
    where
        F: FnMut(&GShell, &GShell, [f64; 3], &[f64]),
    {
        let mut eng = Engine::new_2center(Operator::erfc(self.omega), self.aux, ENGINE_PRECISION)?;
        self.sr_metric_walk(|ip, iq, t| {
            let blk = eng.compute_eri2_shifted(self.aux, ip, iq, t)?;
            sink(&self.aux_sh[ip], &self.aux_sh[iq], t, blk);
            Ok(())
        })
    }

    /// The SR metric walk alone: `visit(P shell, Q shell, T)` for every
    /// `(P_0 | Q_T)` the screen keeps, in the fixed order
    /// [`Stage::sr_metric_each`] sums them (shared with the gradient's
    /// derivative walk, [`deriv`]). Returns the count.
    fn sr_metric_walk<F>(&self, mut visit: F) -> Result<usize, FerricError>
    where
        F: FnMut(usize, usize, [f64; 3]) -> Result<(), FerricError>,
    {
        let mut count = 0usize;
        let global = self.sr2_global_radius();
        let nsh = self.aux_sh.len();
        for ip in 0..nsh {
            for iq in 0..nsh {
                self.sr2_pair(ip, iq, global, &mut count, &mut visit)?;
            }
        }
        Ok(count)
    }

    /// The unscreened (`sr_screen = false`) metric radius; 0 when screening.
    fn sr2_global_radius(&self) -> f64 {
        if self.sr_screen {
            return 0.0;
        }
        let mut r = 0.0_f64;
        for p in &self.aux_sh {
            for q in &self.aux_sh {
                if let Some(x) = self.radius(p.qbound, q.qbound, (p.amin, p.amax), (q.amin, q.amax))
                {
                    r = r.max(x);
                }
            }
        }
        r
    }

    /// One aux pair `(ip, iq)` of the SR metric walk: every kept `T` in
    /// walker order. The single body of the serial [`Stage::sr_metric_walk`]
    /// and the parallel [`Stage::sr_metric_binned`].
    fn sr2_pair<F>(
        &self,
        ip: usize,
        iq: usize,
        global: f64,
        count: &mut usize,
        visit: &mut F,
    ) -> Result<(), FerricError>
    where
        F: FnMut(usize, usize, [f64; 3]) -> Result<(), FerricError>,
    {
        let (p, q) = (&self.aux_sh[ip], &self.aux_sh[iq]);
        let rad = if self.sr_screen {
            match self.radius(p.qbound, q.qbound, (p.amin, p.amax), (q.amin, q.amax)) {
                Some(r) => r,
                None => return Ok(()),
            }
        } else {
            global
        };
        // |R_P − (R_Q + T)| <= rad.
        let x0 = [
            p.center[0] - q.center[0],
            p.center[1] - q.center[1],
            p.center[2] - q.center[2],
        ];
        self.walker.visit(x0, rad, |t| {
            *count += 1;
            visit(ip, iq, t)
        })
    }

    /// SR 3-index `Σ_{L,T} (μ_0 ν_L | P_T)_erfc` over the pair `images`,
    /// `(nao², naux)` (unsymmetrised), and the triplet count: the single bin
    /// of [`Stage::sr_three_index_binned`], i.e. the ORDERED (pre-s2) walk.
    /// Production Gamma builds use [`Stage::sr_three_index_s2`]; this one
    /// serves the frozen oracles.
    fn sr_three_index(&self, images: &[[f64; 3]]) -> Result<(Array2<f64>, usize), FerricError> {
        let (mut bins, count) = self.sr_three_index_binned(images, SrBinning::GAMMA)?;
        Ok((bins.swap_remove(0), count))
    }

    /// Gamma SR 3-index sum over UNORDERED shell pairs (s2, module doc
    /// "Orbital-pair symmetry"): `(J3_SR (nao², naux), computed triplets,
    /// ordered-equivalent triplets)`. `J3_SR` is EXACTLY symmetric in
    /// `μ ↔ ν`; row `μν` with `μ ≤ ν` is BITWISE row `μν` of the ordered walk
    /// [`Stage::sr_three_index`] (the same task body,
    /// [`Stage::sr3_pair_acc`], with the same single-bin context), and row
    /// `νμ` is a copy of it.
    ///
    /// PARALLEL over pairs `i1 ≤ i2`; each task COPIES its finished block into
    /// both rows under the mutex ([`copy_pair_rows_mirrored`]). The row sets
    /// of distinct unordered pairs are disjoint, so every element is written
    /// by exactly one task and the result is bitwise identical across thread
    /// counts. Per-task scratch is that of the ordered walk (`nfun_max² naux`).
    fn sr_three_index_s2(
        &self,
        images: &[[f64; 3]],
    ) -> Result<(Array2<f64>, usize, usize), FerricError> {
        let n = self.obs.nbasis();
        let pool = self.sr3_engine_pool()?;
        let out = Mutex::new(vec![Array2::<f64>::zeros((n * n, self.aux.nbasis()))]);
        // Gamma: every image is residue 0, exactly as the single-bin walk
        // computes it (`SrBinning::residue` returns 0 for unit moduli).
        let l_bin = vec![0usize; images.len()];
        let ctx = Sr3Ctx {
            pool: &pool,
            images,
            l_bin: &l_bin,
            recip: self.cell.reciprocal(),
            bins: SrBinning::GAMMA,
            global: self.sr3_global_radius(),
            out: &out,
        };
        let pairs = unordered_pairs(self.obs_sh.len());
        let counts: Vec<Result<usize, FerricError>> = pairs
            .par_iter()
            .map(|&(i1, i2)| {
                let (acc, count) = self.sr3_pair_acc(&ctx, 0, (i1, i2))?;
                if count > 0 {
                    let mut bins = ctx.out.lock().unwrap_or_else(|e| e.into_inner());
                    copy_pair_rows_mirrored(
                        &mut bins[0],
                        n,
                        &acc,
                        &self.obs_sh[i1],
                        &self.obs_sh[i2],
                        i1 == i2,
                    );
                }
                Ok(count)
            })
            .collect();
        let (count, ordered) = sum_s2_counts(&pairs, counts)?;
        let mut bins = out.into_inner().unwrap_or_else(|e| e.into_inner());
        Ok((bins.swap_remove(0), count, ordered))
    }

    /// SR 3-index binned by `(L mod mod_l, T mod mod_t)`: `bins[r_L R_T + r_T]`
    /// = `Σ (μ_0 ν_L | P_T)_erfc`, `(nao², naux)` reals, and the triplet count.
    ///
    /// PARALLEL over `(ordered shell pair (i1, i2), pair-image residue r_L)`
    /// tasks (FINDINGS "Performance plan (research)", §3 Class A),
    /// BIT-IDENTICAL to the serial `L → i1 → i2 → P → T` walk
    /// ([`Stage::sr_three_index_each`], the frozen oracle of
    /// [`kpoint::sr_bins_parallel_and_serial`]) and across thread counts:
    ///
    /// * Element `(bin (r_L, r_T), μν, P)` lives in row `μ n + ν` with `μ` in
    ///   shell `i1` and `ν` in shell `i2` (shells partition the AO range), so
    ///   it receives addends ONLY from pair `(i1, i2)`, aux shell `P`, images
    ///   `L ≡ r_L` and aux images `T ≡ r_T`. In the serial walk those arrive
    ///   in the order "L ascending (the `images` order), then T in walker
    ///   order"; the task nest `(i1, i2, r_L) → L ≡ r_L ascending → P → T`
    ///   delivers exactly that sequence (the bin only FILTERS the serial
    ///   sequence; the visit decision and block for `(i1, i2, P, L, T)` are
    ///   pure functions of those indices, [`Stage::sr3_pair_image`] is the ONE
    ///   body both walks run).
    /// * Each task accumulates its `R_T` bins' rows of the pair in a
    ///   zero-initialised scratch `(R_T, nμ nν, naux)` with that `+=`
    ///   sequence (0.0 + x == x, so the first addend lands exactly as in the
    ///   zeroed bins), then COPIES it into the bins under a mutex: no two
    ///   tasks own the same element, and a copy is not an addition, so task
    ///   finishing order cannot change a bit.
    /// * One libint engine per rayon worker ([`EnginePool::from_fn`]); every
    ///   `compute_eri3_shifted` call is stateless (the shim copies and moves
    ///   the three shells per call). Counts are integer sums; errors are
    ///   returned in task order.
    ///
    /// Per-task scratch is `R_T nfun_max² naux` reals
    /// ([`Stage::check_sr_scratch`], checked by the caller), independent of
    /// the thread count. Load balance: `nsh² R_L` tasks of uneven cost; a
    /// finer `(pair, P)` split would not change any element's order either.
    fn sr_three_index_binned(
        &self,
        images: &[[f64; 3]],
        bins: SrBinning,
    ) -> Result<(Vec<Array2<f64>>, usize), FerricError> {
        let n = self.obs.nbasis();
        let naux = self.aux.nbasis();
        let nsh = self.obs_sh.len();
        let rl = bins.n_l();
        let recip = self.cell.reciprocal();
        let l_bin: Vec<usize> = images
            .iter()
            .map(|l| SrBinning::residue(&recip, l, bins.mod_l))
            .collect();
        let pool = self.sr3_engine_pool()?;
        let out = Mutex::new(
            (0..rl * bins.n_t())
                .map(|_| Array2::<f64>::zeros((n * n, naux)))
                .collect::<Vec<_>>(),
        );
        let ctx = Sr3Ctx {
            pool: &pool,
            images,
            l_bin: &l_bin,
            recip,
            bins,
            global: self.sr3_global_radius(),
            out: &out,
        };
        let counts: Vec<Result<usize, FerricError>> = (0..nsh * nsh * rl)
            .into_par_iter()
            .map(|task| {
                let (pair, r_l) = (task / rl, task % rl);
                self.sr3_pair_task(&ctx, r_l, (pair / nsh, pair % nsh))
            })
            .collect();
        let count = sum_counts_in_pair_order(counts)?;
        Ok((out.into_inner().unwrap_or_else(|e| e.into_inner()), count))
    }

    /// One erfc 3-centre engine per rayon worker for
    /// [`Stage::sr_three_index_binned`].
    fn sr3_engine_pool(&self) -> Result<EnginePool, FerricError> {
        EnginePool::from_fn(|| {
            Engine::new_3center(
                Operator::erfc(self.omega),
                self.obs,
                self.aux,
                ENGINE_PRECISION,
            )
        })
    }

    /// One [`Stage::sr_three_index_binned`] task: pair `(i1, i2)` over every
    /// image `L ≡ r_L` in `images` order (via [`Stage::sr3_pair_image`], the
    /// serial walk's body) into a zeroed `(R_T, nμ nν, naux)` scratch, then
    /// COPIED into bins `(r_L, ·)` under the mutex. Returns the triplet count.
    fn sr3_pair_task(
        &self,
        ctx: &Sr3Ctx<'_>,
        r_l: usize,
        (i1, i2): (usize, usize),
    ) -> Result<usize, FerricError> {
        let (acc, count) = self.sr3_pair_acc(ctx, r_l, (i1, i2))?;
        if count > 0 {
            self.sr3_copy_pair_rows(ctx, r_l, &acc, &self.obs_sh[i1], &self.obs_sh[i2]);
        }
        Ok(count)
    }

    /// The accumulation of one [`Stage::sr3_pair_task`] (shared with the
    /// Gamma s2 walk [`Stage::sr_three_index_s2`]): pair `(i1, i2)` over every
    /// image `L ≡ r_L` in `images` order into a zeroed `(R_T, nμ nν, naux)`
    /// scratch. Returns `(scratch, triplet count)`.
    fn sr3_pair_acc(
        &self,
        ctx: &Sr3Ctx<'_>,
        r_l: usize,
        (i1, i2): (usize, usize),
    ) -> Result<(Vec<f64>, usize), FerricError> {
        let (a, b) = (&self.obs_sh[i1], &self.obs_sh[i2]);
        let (na, nb) = (a.nfun, b.nfun);
        let bl = na * nb * self.aux.nbasis();
        // scratch[r_T bl + (i nb + j) naux + P]
        //   == bins[r_L R_T + r_T][(a.off+i) n + b.off + j, P]
        let mut acc = vec![0.0_f64; ctx.bins.n_t() * bl];
        let mut count = 0usize;
        ctx.pool.with(|eng| -> Result<(), FerricError> {
            let mut visit = |i1: usize,
                             i2: usize,
                             ip: usize,
                             l: [f64; 3],
                             t: [f64; 3]|
             -> Result<(), FerricError> {
                if let Some(blk) =
                    eng.compute_eri3_shifted(self.obs, self.aux, ip, i1, i2, [t, [0.0; 3], l])?
                {
                    let r_t = SrBinning::residue(&ctx.recip, &t, ctx.bins.mod_t);
                    let dst = &mut acc[r_t * bl..(r_t + 1) * bl];
                    self.sr3_accumulate_block(dst, blk, ip, na, nb);
                }
                Ok(())
            };
            for (l, _) in ctx.images.iter().zip(ctx.l_bin).filter(|(_, r)| **r == r_l) {
                self.sr3_pair_image(i1, i2, l, ctx.global, &mut count, &mut visit)?;
            }
            Ok(())
        })?;
        Ok((acc, count))
    }

    /// `acc[(i nb + j) naux + P] += blk[(pp na + i) nb + j]` for aux shell
    /// `ip`'s functions; `blk` has layout `(nP, n1, n2)`, as
    /// [`Stage::sr_three_index_each`]'s sinks.
    fn sr3_accumulate_block(&self, acc: &mut [f64], blk: &[f64], ip: usize, na: usize, nb: usize) {
        let naux = self.aux.nbasis();
        let p = &self.aux_sh[ip];
        for pp in 0..p.nfun {
            for i in 0..na {
                let src = (pp * na + i) * nb;
                for j in 0..nb {
                    acc[(i * nb + j) * naux + p.off + pp] += blk[src + j];
                }
            }
        }
    }

    /// COPY (not add) pair `(a, b)`'s finished scratch rows of every `r_T`
    /// into bin `(r_l, r_T)` under the mutex, so task finishing order cannot
    /// change a bit.
    fn sr3_copy_pair_rows(
        &self,
        ctx: &Sr3Ctx<'_>,
        r_l: usize,
        acc: &[f64],
        a: &GShell,
        b: &GShell,
    ) {
        let n = self.obs.nbasis();
        let naux = self.aux.nbasis();
        let (na, nb) = (a.nfun, b.nfun);
        let rt = ctx.bins.n_t();
        let bl = na * nb * naux;
        let mut bins = ctx.out.lock().unwrap_or_else(|e| e.into_inner());
        for r_t in 0..rt {
            let j3 = &mut bins[r_l * rt + r_t];
            for i in 0..na {
                for j in 0..nb {
                    let row = (a.off + i) * n + b.off + j;
                    let src = r_t * bl + (i * nb + j) * naux;
                    j3.row_mut(row)
                        .iter_mut()
                        .zip(&acc[src..src + naux])
                        .for_each(|(d, &s)| *d = s);
                }
            }
        }
    }

    /// Every SR 3-centre block `(μ_0 ν_L | P_T)_erfc` the screen keeps, in a
    /// fixed order: `sink(μ shell, ν shell, P shell, L, T, block (nP, nμ, nν))`.
    /// Returns the triplet count. SERIAL: only the frozen serial oracles of
    /// [`kpoint`] use it; the production Gamma and k-point sums are the
    /// parallel [`Stage::sr_three_index_binned`], which reproduces this
    /// walk's per-element order.
    fn sr_three_index_each<F>(&self, images: &[[f64; 3]], mut sink: F) -> Result<usize, FerricError>
    where
        F: FnMut(&GShell, &GShell, &GShell, [f64; 3], [f64; 3], &[f64]),
    {
        let mut eng = Engine::new_3center(
            Operator::erfc(self.omega),
            self.obs,
            self.aux,
            ENGINE_PRECISION,
        )?;
        self.sr_three_index_walk(images, |i1, i2, ip, l, t| {
            if let Some(blk) =
                eng.compute_eri3_shifted(self.obs, self.aux, ip, i1, i2, [t, [0.0; 3], l])?
            {
                sink(
                    &self.obs_sh[i1],
                    &self.obs_sh[i2],
                    &self.aux_sh[ip],
                    l,
                    t,
                    blk,
                );
            }
            Ok(())
        })
    }

    /// The SR 3-centre walk alone: `visit(μ shell, ν shell, P shell, L, T)`
    /// for every `(μ_0 ν_L | P_T)` the screen keeps, in the fixed order
    /// [`Stage::sr_three_index_each`] sums them (shared with the gradient's
    /// derivative walk, [`deriv`], so both see the same triplet set).
    /// Returns the triplet count. SERIAL, `L → i1 → i2 → P → T`; the body
    /// per `(L, i1, i2)` is [`Stage::sr3_pair_image`], shared with the
    /// parallel [`Stage::sr_three_index_binned`].
    fn sr_three_index_walk<F>(
        &self,
        images: &[[f64; 3]],
        mut visit: F,
    ) -> Result<usize, FerricError>
    where
        F: FnMut(usize, usize, usize, [f64; 3], [f64; 3]) -> Result<(), FerricError>,
    {
        let mut count = 0usize;
        let global = self.sr3_global_radius();
        let nsh = self.obs_sh.len();
        for l in images {
            for i1 in 0..nsh {
                for i2 in 0..nsh {
                    self.sr3_pair_image(i1, i2, l, global, &mut count, &mut visit)?;
                }
            }
        }
        Ok(count)
    }

    /// The unscreened (`sr_screen = false`) global radius; 0 when screening.
    fn sr3_global_radius(&self) -> f64 {
        if self.sr_screen {
            return 0.0;
        }
        let mut r = 0.0_f64;
        for a in &self.obs_sh {
            for b in &self.obs_sh {
                let (qab, pmin, pmax) = pair_bound(a, b, 0.0);
                for p in &self.aux_sh {
                    if let Some(x) = self.radius(qab, p.qbound, (pmin, pmax), (p.amin, p.amax)) {
                        r = r.max(x);
                    }
                }
            }
        }
        r
    }

    /// One `(pair (i1, i2), image L)` of the SR 3-centre walk: every aux
    /// shell `P` in order, then every kept `T` in walker order. The single
    /// body of both the serial [`Stage::sr_three_index_walk`] and the
    /// parallel [`Stage::sr_three_index_binned`] (so both see the same
    /// triplets in the same per-pair order).
    fn sr3_pair_image<F>(
        &self,
        i1: usize,
        i2: usize,
        l: &[f64; 3],
        global: f64,
        count: &mut usize,
        visit: &mut F,
    ) -> Result<(), FerricError>
    where
        F: FnMut(usize, usize, usize, [f64; 3], [f64; 3]) -> Result<(), FerricError>,
    {
        let a = &self.obs_sh[i1];
        let b = &self.obs_sh[i2];
        let bc = [b.center[0] + l[0], b.center[1] + l[1], b.center[2] + l[2]];
        let ab = [
            a.center[0] - bc[0],
            a.center[1] - bc[1],
            a.center[2] - bc[2],
        ];
        let r2 = dot3(&ab, &ab);
        let (qab, pmin, pmax) = pair_bound(a, b, r2);
        let mid = [
            0.5 * (a.center[0] + bc[0]),
            0.5 * (a.center[1] + bc[1]),
            0.5 * (a.center[2] + bc[2]),
        ];
        let half = 0.5 * r2.sqrt();
        for (ip, p) in self.aux_sh.iter().enumerate() {
            let rad = if self.sr_screen {
                match self.radius(qab, p.qbound, (pmin, pmax), (p.amin, p.amax)) {
                    Some(r) => r,
                    None => continue,
                }
            } else {
                global
            };
            // Every aux image within `rad` of the segment lies
            // within `rad + half` of its midpoint.
            let x0 = [
                mid[0] - p.center[0],
                mid[1] - p.center[1],
                mid[2] - p.center[2],
            ];
            self.walker.visit(x0, rad + half, |t| {
                let x = [p.center[0] + t[0], p.center[1] + t[1], p.center[2] + t[2]];
                if segment_distance(x, a.center, bc) > rad {
                    return Ok(());
                }
                *count += 1;
                visit(i1, i2, ip, *l, t)
            })?;
        }
        Ok(())
    }

    /// LR `(2/Ω) Σ_{G∈half} 4π/G² e^{−G²/4ω²} Re[conj(A) X]` added into `j2`
    /// and `j3`; `pair_ft` G-chunked within `chunk_budget`. Returns the chunk
    /// count and the `(wall, CPU)` seconds spent in the per-chunk sink (aux
    /// FT, packing and the four GEMMs); the rest of the call is the pair FT.
    /// `kernel` selects production or the FROZEN serial path ([`LrKernel`]).
    /// Sub-stages and counters of both parts go to `sub` (observation only).
    fn lr_accumulate(
        &self,
        gv: &[[f64; 3]],
        j2: &mut Array2<f64>,
        j3: &mut Array2<f64>,
        chunk_budget: usize,
        kernel: LrKernel,
        sub: &mut PbcTimings,
    ) -> Result<(usize, f64, Option<f64>), FerricError> {
        let n = self.obs.nbasis();
        let n2 = n * n;
        let naux = self.aux.nbasis();
        let vol = self.cell.volume();
        let omega = self.omega;
        // pr, pi (n² reals each) + X (naux complex) + xr, xi, xrw, xiw.
        let extra_per_g = n2
            .saturating_mul(16)
            .saturating_add(naux.saturating_mul(48))
            .saturating_add(64);
        let pair_ft_thresh = (0.01 * self.thresh).min(crate::pair_ft::DEFAULT_PAIR_FT_THRESH);
        let aux_sh = &self.aux_sh;
        let (mut sink_wall, mut sink_cpu) = (0.0_f64, None::<f64>);
        let mut sink_t = PbcTimings::default();
        let mut bufs = PackBufs::default();
        let sink =
            |_g0: usize, gs: &[[f64; 3]], pft: &Array3<Complex64>| -> Result<(), FerricError> {
                let clock = StageClock::start();
                let ng = gs.len();
                let c = StageClock::start();
                let x = aux_ft_shells(aux_sh, naux, gs);
                sink_t.stop_sub(SUB_AUX_FT, &c);
                let c = StageClock::start();
                let w: Vec<f64> = gs
                    .iter()
                    .map(|gvec| {
                        let g2 = dot3(gvec, gvec);
                        2.0 / vol * 4.0 * PI / g2 * (-g2 / (4.0 * omega * omega)).exp()
                    })
                    .collect();
                let (pr, pim) = pack_pair_ft(pft, Some(w.as_slice()), kernel, &mut bufs);
                sink_t.stop_sub(SUB_P_PACK, &c);
                let c = StageClock::start();
                let mut xr = Array2::<f64>::zeros((naux, ng));
                let mut xi = Array2::<f64>::zeros((naux, ng));
                let mut xrw = Array2::<f64>::zeros((naux, ng));
                let mut xiw = Array2::<f64>::zeros((naux, ng));
                for (g, &w) in w.iter().enumerate() {
                    for p in 0..naux {
                        let z = x[(p, g)];
                        xr[(p, g)] = z.re;
                        xi[(p, g)] = z.im;
                        xrw[(p, g)] = w * z.re;
                        xiw[(p, g)] = w * z.im;
                    }
                }
                sink_t.stop_sub(SUB_XY_PACK, &c);
                // Re[conj(A) X] = A.re X.re + A.im X.im
                let j2_terms: [(&Array2<f64>, &Array2<f64>); 2] = [(&xrw, &xr), (&xiw, &xi)];
                lr_gemms(
                    j3,
                    &[(pr, &xr), (pim, &xi)],
                    Some((&mut *j2, &j2_terms[..])),
                    kernel,
                    &mut sink_t,
                );
                let (w, c) = clock.elapsed();
                sink_wall += w;
                if let Some(c) = c {
                    sink_cpu = Some(sink_cpu.unwrap_or(0.0) + c);
                }
                Ok(())
            };
        let n_chunks = lr_pair_ft_chunked(
            kernel,
            self.cell,
            self.obs,
            gv,
            pair_ft_thresh,
            chunk_budget,
            extra_per_g,
            sub,
            sink,
        )?;
        sub.accumulate(&sink_t);
        Ok((n_chunks, sink_wall, sink_cpu))
    }
}

// ---------------------------------------------------------------------------
// LR (G ≠ 0) kernels shared by the Gamma build and its range split
// ---------------------------------------------------------------------------

/// Which kernels the LR stage runs. Production: the survivor-cached,
/// shell-pair-parallel pair FT ([`crate::pair_ft::pair_ft_chunked`]), the
/// per-row parallel packing into reused buffers, the row-blocked parallel J3
/// GEMMs ([`lr_gemm_acc`]) with the J2 GEMMs run beside them ([`lr_gemms`]).
/// `SerialOracle`: the FROZEN pre-parallel path (serial pair FT that re-walks
/// the screen per chunk, serial packing, one GEMM per chunk), used only by
/// [`lr_sums_parallel_and_serial`]. Both use the SAME G chunks, so every J2/J3
/// element receives the same GEMM k-ranges in the same order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LrKernel {
    Production,
    SerialOracle,
}

/// Rows of J3 per task of the parallel LR accumulation GEMMs. A CONSTANT: it
/// must never depend on the thread count or the budget (it decides which
/// GEMM call a J3 row belongs to, never the k-summation of an element).
const LR_GEMM_ROW_BLOCK: usize = 512;

/// Sub-stage: the aux FTs of one LR chunk (inside "rsgdf LR aux FT + GEMM").
pub(crate) const SUB_AUX_FT: &str = "LR sink: aux FT";
/// Sub-stage: the pair FT's (re, im) packing, weights included.
pub(crate) const SUB_P_PACK: &str = "LR sink: P pack (re/im)";
/// Sub-stage: the aux-side real rows (X, Y, weighted copies).
pub(crate) const SUB_XY_PACK: &str = "LR sink: X/Y pack + weights";
/// Sub-stage: the J3 (row-blocked, parallel) and J2 GEMMs, run concurrently.
pub(crate) const SUB_GEMM: &str = "LR sink: J3 + J2 GEMMs";

/// `c += Σ_t a_t · b_tᵀ` in term order (`a_t`: `(m, k)`, `b_t`: `(q, k)`,
/// `c`: `(m, q)`); returns the summed wall of its GEMM tasks (ns).
///
/// Production: `c`'s and every `a_t`'s rows in fixed blocks of
/// [`LR_GEMM_ROW_BLOCK`]; ONE rayon task per block runs the single-threaded
/// BLAS GEMMs of all terms on that block, in term order (so the block stays
/// in cache between terms). Every element of `c` still gets, per term,
/// exactly one `+=` of one GEMM's length-`k` dot product over the same k
/// range (the G chunk) and with the same call shape as the unsplit-by-term
/// row-blocked loop, in the same term order; only which task runs it
/// changes. The oracle is one unsplit call per term.
fn lr_gemm_acc(
    terms: &[(ArrayView2<'_, f64>, &Array2<f64>)],
    c: &mut Array2<f64>,
    kernel: LrKernel,
) -> u64 {
    let elapsed_ns = |t0: Instant| u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX);
    if kernel == LrKernel::SerialOracle || c.nrows() <= LR_GEMM_ROW_BLOCK {
        let t0 = Instant::now();
        for (a, b) in terms {
            general_mat_mul(1.0, a, &b.t(), 1.0, c);
        }
        return elapsed_ns(t0);
    }
    let busy = AtomicU64::new(0);
    let cs: Vec<_> = c.axis_chunks_iter_mut(Axis(0), LR_GEMM_ROW_BLOCK).collect();
    cs.into_par_iter().enumerate().for_each(|(blk, mut cb)| {
        let t0 = Instant::now();
        let r0 = blk * LR_GEMM_ROW_BLOCK;
        let r1 = r0 + cb.nrows();
        for (a, b) in terms {
            general_mat_mul(1.0, &a.slice(s![r0..r1, ..]), &b.t(), 1.0, &mut cb);
        }
        busy.fetch_add(elapsed_ns(t0), Ordering::Relaxed);
    });
    busy.into_inner()
}

/// The GEMMs of one LR chunk: `c += Σ a_t b_tᵀ` ([`lr_gemm_acc`]) and, if
/// given, `j2 += Σ x_t y_tᵀ` (one unsplit GEMM per term, in order).
/// Production runs the J2 terms CONCURRENTLY with the J3 row blocks
/// (`rayon::join`: disjoint outputs, each keeps its own call sequence, so
/// no element changes); the oracle runs them one after the other. Timed
/// into `t` ([`SUB_GEMM`] wall; busy thread-µs as counters).
#[allow(clippy::type_complexity)]
fn lr_gemms(
    c: &mut Array2<f64>,
    terms: &[(ArrayView2<'_, f64>, &Array2<f64>)],
    j2: Option<(&mut Array2<f64>, &[(&Array2<f64>, &Array2<f64>)])>,
    kernel: LrKernel,
    t: &mut PbcTimings,
) {
    let clock = StageClock::start();
    let j2_part = move || -> u64 {
        let t0 = Instant::now();
        if let Some((j2, jt)) = j2 {
            for (x, y) in jt {
                general_mat_mul(1.0, *x, &y.t(), 1.0, &mut *j2);
            }
        }
        u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX)
    };
    let (j3_ns, j2_ns) = if kernel == LrKernel::SerialOracle {
        let a = lr_gemm_acc(terms, c, kernel);
        (a, j2_part())
    } else {
        rayon::join(|| lr_gemm_acc(terms, c, kernel), j2_part)
    };
    t.stop_sub(SUB_GEMM, &clock);
    t.add_counter("LR J3 GEMM busy us (sum of task walls)", j3_ns / 1000);
    t.add_counter("LR J2 GEMM busy us", j2_ns / 1000);
}

/// Reusable `(re, im)` buffers of [`pack_pair_ft`] (one per LR pass).
#[derive(Default)]
struct PackBufs {
    re: Vec<f64>,
    im: Vec<f64>,
}

/// `buf` at exactly `len` elements. Production reuse: existing elements are
/// left STALE (every one is overwritten by [`pack_pair_ft`]); growth is
/// zero-filled in parallel. `fresh`: all `+0.0` (the oracle's fresh array).
fn size_buffer(buf: &mut Vec<f64>, len: usize, fresh: bool) {
    if fresh {
        buf.clear();
        buf.resize(len, 0.0);
    } else if buf.len() >= len {
        buf.truncate(len);
    } else {
        let grow = len - buf.len();
        buf.reserve_exact(grow);
        buf.par_extend(rayon::iter::repeat_n(0.0_f64, grow));
    }
}

/// `(re, im)` of a pair FT chunk `P[m, k, g]` as `(n², ng)` real matrices,
/// row `m·n + k`, each scaled by `w[g]` (`w[g] * z.re`) or copied (`None`),
/// in `bufs`. Production fills the rows in parallel into reused buffers
/// (every element written, so stale values never survive); the oracle
/// starts from zeros and runs the frozen serial loop. Every element is the
/// same single expression either way.
fn pack_pair_ft<'b>(
    pft: &Array3<Complex64>,
    w: Option<&[f64]>,
    kernel: LrKernel,
    bufs: &'b mut PackBufs,
) -> (ArrayView2<'b, f64>, ArrayView2<'b, f64>) {
    let (n, n1, ng) = pft.dim();
    let n2 = n * n1;
    let oracle = kernel == LrKernel::SerialOracle;
    let PackBufs { re, im } = bufs;
    size_buffer(re, n2 * ng, oracle);
    size_buffer(im, n2 * ng, oracle);
    if ng > 0 && n2 > 0 {
        if oracle {
            for g in 0..ng {
                for m in 0..n {
                    for k in 0..n1 {
                        let z = pft[[m, k, g]];
                        let (zr, zi) = match w {
                            Some(w) => (w[g] * z.re, w[g] * z.im),
                            None => (z.re, z.im),
                        };
                        re[(m * n1 + k) * ng + g] = zr;
                        im[(m * n1 + k) * ng + g] = zi;
                    }
                }
            }
        } else {
            let src = pft.as_standard_layout();
            let src = src.as_slice().expect("standard layout");
            re.par_chunks_mut(ng)
                .zip(im.par_chunks_mut(ng))
                .zip(src.par_chunks(ng))
                .for_each(|((r, i), z)| {
                    for g in 0..ng {
                        let (zr, zi) = match w {
                            Some(w) => (w[g] * z[g].re, w[g] * z[g].im),
                            None => (z[g].re, z[g].im),
                        };
                        r[g] = zr;
                        i[g] = zi;
                    }
                });
        }
    }
    let re: &'b [f64] = re;
    let im: &'b [f64] = im;
    (
        ArrayView2::from_shape((n2, ng), re).expect("packed length"),
        ArrayView2::from_shape((n2, ng), im).expect("packed length"),
    )
}

/// [`pair_ft_chunked_timed`] (production, sub-stages into `sub`) or its
/// frozen serial oracle.
#[allow(clippy::too_many_arguments)]
fn lr_pair_ft_chunked<F>(
    kernel: LrKernel,
    cell: &Cell,
    prep: &PreparedBasis,
    gvecs: &[[f64; 3]],
    thresh: f64,
    chunk_budget_bytes: usize,
    extra_bytes_per_g: usize,
    sub: &mut PbcTimings,
    sink: F,
) -> Result<usize, FerricError>
where
    F: FnMut(usize, &[[f64; 3]], &Array3<Complex64>) -> Result<(), FerricError>,
{
    match kernel {
        LrKernel::Production => pair_ft_chunked_timed(
            cell,
            prep,
            gvecs,
            thresh,
            chunk_budget_bytes,
            extra_bytes_per_g,
            sub,
            sink,
        ),
        LrKernel::SerialOracle => pair_ft_chunked_serial_oracle(
            cell,
            prep,
            gvecs,
            thresh,
            chunk_budget_bytes,
            extra_bytes_per_g,
            sink,
        ),
    }
}

/// `(J2_LR, J3_LR, G chunks)` of one LR kernel.
pub type LrSums = (Array2<f64>, Array2<f64>, usize);

/// TEST ORACLE for the LR stage of [`RsGdf::build`]: the LR (G ≠ 0)
/// contributions for `cfg` (range split included, exactly the dispatch the
/// build runs), accumulated from zero at an EXPLICIT `chunk_budget` (so a
/// test can force several G chunks), as `[production, frozen serial]`.
/// J2 (aux only) must agree BIT FOR BIT and J3 within the derived round-off
/// tolerance of the production pair FT (its image-split phase and
/// premultiplied F rows are not the serial kernel's bits;
/// `tests/pbc_parallel_bitwise.rs`, `pair_ft::plan` module doc).
#[doc(hidden)]
pub fn lr_sums_parallel_and_serial(
    cell: &Cell,
    obs: &PreparedBasis,
    aux: &PreparedBasis,
    cfg: &RsGdfConfig,
    chunk_budget: usize,
) -> Result<[LrSums; 2], FerricError> {
    require_pure_aux(aux, "RsGdf")?;
    let (st, images) = kpoint::diagnostic_stage(cell, obs, aux, cfg)?;
    let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
    let plan = split::SplitPlan::maybe(&st, cfg, &images, &mut ledger)?;
    let gcut = 2.0 * cfg.omega * (1.0 / cfg.precision).ln().sqrt();
    let gv = half_gvectors(cell, gcut)?;
    let (n2, naux) = (obs.nbasis() * obs.nbasis(), aux.nbasis());
    let run = |kernel: LrKernel| -> Result<LrSums, FerricError> {
        let mut j2 = Array2::<f64>::zeros((naux, naux));
        let mut j3 = Array2::<f64>::zeros((n2, naux));
        let (chunks, _, _) = split::lr_accumulate(
            &st,
            plan.as_ref(),
            &gv,
            &mut j2,
            &mut j3,
            chunk_budget,
            kernel,
            &mut PbcTimings::default(),
        )?;
        Ok((j2, j3, chunks))
    };
    Ok([run(LrKernel::Production)?, run(LrKernel::SerialOracle)?])
}

/// One Gamma SR 3-centre result of [`sr3_gamma_s2_and_s1`]: `(J3_SR
/// unsymmetrised (nao², naux), triplets computed, ordered-equivalent
/// triplets)`.
pub type Sr3Parts = (Array2<f64>, usize, usize);

/// TEST ORACLE for the Gamma s2 SR 3-centre walk (module doc "Orbital-pair
/// symmetry"): `[s2, s1]` = the SR `J3` [`RsGdf::build`] computes at `cfg`
/// (range split honoured) and the FROZEN ordered walk
/// [`RsGdf::build_pair_s1_oracle`] computes, both before the LR and G = 0
/// terms and before any symmetrisation. Expected: `s2` exactly symmetric;
/// each `s2[μν]` bitwise equal to `s1[μν]` or `s1[νμ]` (unsplit: `s1[μν]`
/// for `μ ≤ ν`); the unsplit ordered-equivalent count equal to `s1`'s count
/// (`tests/pbc_pair_symmetry.rs`).
#[doc(hidden)]
pub fn sr3_gamma_s2_and_s1(
    cell: &Cell,
    obs: &PreparedBasis,
    aux: &PreparedBasis,
    cfg: &RsGdfConfig,
) -> Result<[Sr3Parts; 2], FerricError> {
    require_pure_aux(aux, "RsGdf")?;
    let (st, images) = kpoint::diagnostic_stage(cell, obs, aux, cfg)?;
    let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
    let plan = split::SplitPlan::maybe(&st, cfg, &images, &mut ledger)?;
    Ok([
        split::sr_three_index(&st, plan.as_ref(), &images)?,
        split::sr_three_index_s1(&st, plan.as_ref(), &images)?,
    ])
}

/// One unit of an ordered SR derivative walk ([`Stage::sr_three_index_ordered`],
/// [`Stage::sr_metric_ordered`]): the unit's screened count and, per kept
/// entry with a contribution, its `(aux shell, T)` and value, in walk order.
struct SrUnit<V> {
    count: usize,
    items: Vec<(usize, [f64; 3], V)>,
}

impl<V: Stored> Stored for SrUnit<V> {
    /// Heap bytes held: the struct plus 32 bytes of bookkeeping and the value's own bytes per item.
    fn stored_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self
                .items
                .iter()
                .map(|(_, _, v)| 32 + v.stored_bytes())
                .sum::<usize>()
    }
}

/// The ordered-parallel SR derivative walks of the forces and the stress
/// ([`deriv`], [`strain`], [`kpoint`]'s `kderiv`). See [`crate::ordered`]:
/// the pure part (screen + derivative integrals + any per-triplet subtotal
/// the serial code forms from zero) runs in parallel over a window of
/// units; `apply` runs SERIALLY in exactly the serial walk's order, so every
/// shared output (atom rows, aux rows, the stress) receives the serial
/// scalar sequence: BIT-IDENTICAL to [`Stage::sr_three_index_walk`] /
/// [`Stage::sr_metric_walk`] driving the same body, at any thread count.
impl Stage<'_> {
    /// One erfc 3-centre DERIVATIVE engine per rayon worker.
    fn sr3_deriv_pool(&self) -> Result<EnginePool, FerricError> {
        EnginePool::from_fn(|| {
            Engine::new_3center_deriv(
                Operator::erfc(self.omega),
                self.obs,
                self.aux,
                ENGINE_PRECISION,
            )
        })
    }

    /// One erfc 2-centre DERIVATIVE engine per rayon worker.
    fn sr2_deriv_pool(&self) -> Result<EnginePool, FerricError> {
        EnginePool::from_fn(|| {
            Engine::new_2center_deriv(Operator::erfc(self.omega), self.aux, ENGINE_PRECISION)
        })
    }

    /// One `(L, i1, i2)` unit of [`Stage::sr_three_index_ordered`]: the
    /// serial walk's own body [`Stage::sr3_pair_image`], collecting
    /// `eval`'s values in walk order.
    fn sr3_unit<V, E>(
        &self,
        eng: &mut Engine,
        (l, i1, i2): ([f64; 3], usize, usize),
        global: f64,
        eval: &E,
    ) -> Result<SrUnit<V>, FerricError>
    where
        E: Fn(
            &mut Engine,
            usize,
            usize,
            usize,
            [f64; 3],
            [f64; 3],
        ) -> Result<Option<V>, FerricError>,
    {
        let mut count = 0usize;
        let mut items = Vec::new();
        let mut visit =
            |_: usize, _: usize, ip: usize, _: [f64; 3], t: [f64; 3]| -> Result<(), FerricError> {
                if let Some(v) = eval(eng, i1, i2, ip, l, t)? {
                    items.push((ip, t, v));
                }
                Ok(())
            };
        self.sr3_pair_image(i1, i2, &l, global, &mut count, &mut visit)?;
        Ok(SrUnit { count, items })
    }

    /// One `(P, Q)` unit of [`Stage::sr_metric_ordered`]: the serial walk's
    /// own body [`Stage::sr2_pair`], collecting `eval`'s values in walk
    /// order.
    fn sr2_unit<V, E>(
        &self,
        eng: &mut Engine,
        (ip, iq): (usize, usize),
        global: f64,
        eval: &E,
    ) -> Result<SrUnit<V>, FerricError>
    where
        E: Fn(&mut Engine, usize, usize, [f64; 3]) -> Result<Option<V>, FerricError>,
    {
        let mut count = 0usize;
        let mut items = Vec::new();
        let mut visit = |_: usize, _: usize, t: [f64; 3]| -> Result<(), FerricError> {
            if let Some(v) = eval(eng, ip, iq, t)? {
                items.push((iq, t, v));
            }
            Ok(())
        };
        self.sr2_pair(ip, iq, global, &mut count, &mut visit)?;
        Ok(SrUnit { count, items })
    }

    /// [`Stage::sr_three_index_walk`], ordered-parallel: units `(L, i1, i2)`
    /// in `L → i1 → i2` order, each running [`Stage::sr3_pair_image`] (the
    /// serial walk's own body) with `eval(engine, i1, i2, P, L, T)` (PURE;
    /// `None` = no contribution) in parallel; then `apply(i1, i2, P, L, T,
    /// value)` serially in walk order. Returns the triplet count.
    fn sr_three_index_ordered<V, E, A>(
        &self,
        images: &[[f64; 3]],
        pool: &EnginePool,
        budget: usize,
        eval: E,
        mut apply: A,
    ) -> Result<usize, FerricError>
    where
        V: Send + Stored,
        E: Fn(
                &mut Engine,
                usize,
                usize,
                usize,
                [f64; 3],
                [f64; 3],
            ) -> Result<Option<V>, FerricError>
            + Sync,
        A: FnMut(usize, usize, usize, [f64; 3], [f64; 3], V) -> Result<(), FerricError>,
    {
        let nsh = self.obs_sh.len();
        let global = self.sr3_global_radius();
        let unit_of = |u: usize| (images[u / (nsh * nsh)], (u / nsh) % nsh, u % nsh);
        let mut count = 0usize;
        ordered_units(
            images.len() * nsh * nsh,
            budget,
            0,
            |u| pool.with(|eng| self.sr3_unit(eng, unit_of(u), global, &eval)),
            |u, unit: SrUnit<V>| {
                let (l, i1, i2) = unit_of(u);
                count += unit.count;
                for (ip, t, v) in unit.items {
                    apply(i1, i2, ip, l, t, v)?;
                }
                Ok(())
            },
        )?;
        Ok(count)
    }

    /// [`Stage::sr_metric_walk`], ordered-parallel: units `(P, Q)` in
    /// `P → Q` order, each running [`Stage::sr2_pair`] with `eval(engine, P,
    /// Q, T)` (PURE) in parallel; then `apply(P, Q, T, value)` serially in
    /// walk order. Returns the pair count.
    fn sr_metric_ordered<V, E, A>(
        &self,
        pool: &EnginePool,
        budget: usize,
        eval: E,
        mut apply: A,
    ) -> Result<usize, FerricError>
    where
        V: Send + Stored,
        E: Fn(&mut Engine, usize, usize, [f64; 3]) -> Result<Option<V>, FerricError> + Sync,
        A: FnMut(usize, usize, [f64; 3], V) -> Result<(), FerricError>,
    {
        let nsh = self.aux_sh.len();
        let global = self.sr2_global_radius();
        let mut count = 0usize;
        ordered_units(
            nsh * nsh,
            budget,
            0,
            |u| pool.with(|eng| self.sr2_unit(eng, (u / nsh, u % nsh), global, &eval)),
            |u, unit: SrUnit<V>| {
                let (ip, iq) = (u / nsh, u % nsh);
                count += unit.count;
                for (_, t, v) in unit.items {
                    apply(ip, iq, t, v)?;
                }
                Ok(())
            },
        )?;
        Ok(count)
    }
}

/// Pair-image radius of the SR 3-centre sum: every pair whose charge bound
/// times the largest aux charge and potential factor reaches `precision`
/// (pair_ft's radius rule at the equivalent pair threshold). Shared by the
/// Gamma and k-point builds.
fn pair_image_radius(st: &Stage<'_>, precision: f64) -> f64 {
    let amin_orb = st
        .obs_sh
        .iter()
        .map(|s| s.amin)
        .fold(f64::INFINITY, f64::min);
    let pmax_orb = 2.0 * st.obs_sh.iter().map(|s| s.amax).fold(0.0_f64, f64::max);
    let qaux_max = st.aux_sh.iter().map(|p| p.qbound).fold(0.0_f64, f64::max);
    let vfac_max = st
        .aux_sh
        .iter()
        .map(|p| 1.0 + 2.0 * (pmax_orb * p.amax / (pmax_orb + p.amax)).sqrt() / PI.sqrt())
        .fold(1.0_f64, f64::max);
    let pair_thresh = 0.1 * precision / (qaux_max * vfac_max).max(1.0);
    (2.0 * (1e3 / pair_thresh).ln() / amin_orb).sqrt() + SR_MARGIN_BOHR
}

/// The pair images are generated from the cell's atoms, so the orbital basis
/// must sit on exactly those atoms (`pair_ft` re-checks this).
fn check_obs_on_cell(cell: &Cell, obs: &PreparedBasis) -> Result<(), FerricError> {
    let pos = cell.positions();
    if obs.atoms().len() != pos.len() {
        return Err(FerricError::General(format!(
            "RsGdf: orbital PreparedBasis has {} atoms but the cell has {}; build it from cell.mol()",
            obs.atoms().len(),
            pos.len()
        )));
    }
    for (k, (a, p)) in obs.atoms().iter().zip(&pos).enumerate() {
        let d = ((a.x - p[0]).powi(2) + (a.y - p[1]).powi(2) + (a.z - p[2]).powi(2)).sqrt();
        if d > 1e-10 {
            return Err(FerricError::General(format!(
                "RsGdf: orbital PreparedBasis atom {k} is {d:.3e} Bohr from the cell's atom {k}"
            )));
        }
    }
    Ok(())
}

/// The Gamma build's SR 3-index sum under `pair_sym`, or on the
/// column-rotated shells when `rotation` is set (s2 only; module doc
/// "Column rotation"): `(J3_SR, computed, ordered-equivalent)` (module doc
/// "Orbital-pair symmetry").
fn gamma_sr_three_index(
    pair_sym: PairSym,
    st: &Stage<'_>,
    plan: Option<&split::SplitPlan>,
    rotation: Option<&Sr3Rotation>,
    range_split: Option<RangeSplit>,
    images: &[[f64; 3]],
) -> Result<(Array2<f64>, usize, usize), FerricError> {
    match (pair_sym, rotation) {
        (_, Some(rot)) => gamma_sr_three_index_rotated(st, rot, range_split, images),
        (PairSym::S2, None) => split::sr_three_index(st, plan, images),
        (PairSym::S1Oracle, None) => split::sr_three_index_s1(st, plan, images),
    }
}

/// The rotated bases of the Gamma SR 3-centre walk (module doc "Column
/// rotation"): the rotated orbital basis and, for
/// [`ColumnRotationMutant::RotateAux`] only, a rotated aux basis whose
/// columns are (deliberately) never transformed back.
pub(super) struct Sr3Rotation {
    pub(super) obs: RotatedBasis,
    aux: Option<RotatedBasis>,
}

impl Sr3Rotation {
    /// The aux basis the rotated walk calls into.
    pub(super) fn aux_prep<'b>(&'b self, parent: &'b PreparedBasis) -> &'b PreparedBasis {
        self.aux.as_ref().map_or(parent, |a| &a.prep)
    }
}

/// The column rotation `cfg` asks for on `(obs, aux)` in a Gamma energy
/// build (`Auto` = on): `Ok(None)` when it is off or nothing in `obs`
/// rotates (the identity: the caller runs the unrotated walk, bit for bit).
pub(super) fn sr3_rotation(
    cell: &Cell,
    obs: &PreparedBasis,
    aux: &PreparedBasis,
    cfg: &RsGdfConfig,
) -> Result<Option<Sr3Rotation>, FerricError> {
    let Some(rot) = cfg.sr_column_rotation.resolve_supported() else {
        return Ok(None);
    };
    let Some(obs_rot) = RotatedBasis::detect(cell, obs, rot, "RsGdf")? else {
        return Ok(None);
    };
    let aux_rot = if rot.mutant == ColumnRotationMutant::RotateAux {
        RotatedBasis::detect(cell, aux, rot, "RsGdf aux (mutant)")?
    } else {
        None
    };
    Ok(Some(Sr3Rotation {
        obs: obs_rot,
        aux: aux_rot,
    }))
}

/// The build's SR plans: the range split ([`split::SplitPlan::maybe`]) and
/// the column rotation ([`sr3_rotation`]; `Auto` resolves off on the
/// gradient build and the frozen s1 oracle, which must walk the parent
/// shells, and an explicit `On` is refused there), with the `rsgdf SR3
/// rotated columns` counter set on `timings` whenever the rotation ran
/// (0 = nothing rotates, the identity).
#[allow(clippy::too_many_arguments)]
fn sr_plans(
    st: &Stage<'_>,
    cfg: &RsGdfConfig,
    images: &[[f64; 3]],
    ledger: &mut Ledger,
    retain_grad: bool,
    pair_sym: PairSym,
    timings: &mut PbcTimings,
) -> Result<(Option<split::SplitPlan>, Option<Sr3Rotation>), FerricError> {
    let plan = split::SplitPlan::maybe(st, cfg, images, ledger)?;
    if retain_grad || pair_sym != PairSym::S2 {
        cfg.sr_column_rotation.refuse_explicit(
            "RsGdf",
            "the column rotation applies to the Gamma energy build only; the gradient build \
             and the frozen s1 oracle walk the unrotated shells",
        )?;
        return Ok((plan, None));
    }
    if cfg.sr_column_rotation.resolve_supported().is_none() {
        return Ok((plan, None));
    }
    let rotation = sr3_rotation(st.cell, st.obs, st.aux, cfg)?;
    timings.set_counter(
        "rsgdf SR3 rotated columns",
        rotation.as_ref().map_or(0, |r| r.obs.n_rotated_columns) as u64,
    );
    Ok((plan, rotation))
}

/// The Gamma s2 SR 3-index sum on the column-rotated stage, back-transformed
/// into the parent AO basis: `(J3_SR, computed, ordered-equivalent)` of the
/// ROTATED walk (module doc "Column rotation"). The split plan of the
/// rotated walk carries only its orbital and aux pieces
/// ([`split::SplitPlan::sr3_only`]); the build's own plan keeps the LR,
/// `S_ss` and G = 0 parts in the parent basis.
fn gamma_sr_three_index_rotated(
    st: &Stage<'_>,
    rot: &Sr3Rotation,
    range_split: Option<RangeSplit>,
    images: &[[f64; 3]],
) -> Result<(Array2<f64>, usize, usize), FerricError> {
    let st_rot = st.with_bases(&rot.obs.prep, rot.aux_prep(st.aux))?;
    let plan = match range_split {
        None => None,
        Some(rs) => Some(split::SplitPlan::sr3_only(&st_rot, rs)?),
    };
    let (mut j3, count, ordered) = split::sr_three_index(&st_rot, plan.as_ref(), images)?;
    let w = j3.ncols();
    let data = j3.as_slice_mut().ok_or_else(|| {
        FerricError::General("RsGdf column rotation: J3_SR is not contiguous".into())
    })?;
    rot.obs.back_transform_pair_rows(data, w)?;
    Ok((j3, count, ordered))
}

/// Symmetrise `j3` over μ↔ν in place; returns the largest asymmetry seen.
fn symmetrize_pairs(j3: &mut Array2<f64>, n: usize) -> f64 {
    let naux = j3.ncols();
    let mut asym = 0.0_f64;
    for m in 0..n {
        for k in 0..m {
            let (r1, r2) = (m * n + k, k * n + m);
            for p in 0..naux {
                let (x, y) = (j3[(r1, p)], j3[(r2, p)]);
                asym = asym.max((x - y).abs());
                let avg = 0.5 * (x + y);
                j3[(r1, p)] = avg;
                j3[(r2, p)] = avg;
            }
        }
    }
    asym
}

/// What the RS-GDF FORCES need from the metric solve beyond `B` and `W`
/// ([`RsGdf::build_for_gradient`]; FINDINGS "Iteration 18"): the kept and
/// dropped eigenvalues, the dropped eigenvectors `U_d` and `J3` projected on
/// them, `J_d = U_dᵀ J3ᵀ` — the pieces of the kept–dropped (Loewner /
/// Daleckii–Krein) block of the metric derivative. Energies never read it.
#[derive(Debug, Clone)]
pub(crate) struct MetricGradParts {
    /// Kept eigenvalues, in the column order of `W` (`> lindep`).
    pub(crate) s_kept: Vec<f64>,
    /// Dropped eigenvalues (`<= lindep`), in the column order of `u_drop`.
    pub(crate) s_drop: Vec<f64>,
    /// `(naux, n_dropped)` dropped eigenvectors.
    pub(crate) u_drop: Array2<f64>,
    /// `(n_dropped, nao²)` = `U_dᵀ J3ᵀ` (symmetrised J3, G = 0 removed).
    pub(crate) jd: Array2<f64>,
    /// The build's SR screen mode (the derivative walk must visit the same
    /// triplets).
    pub(crate) sr_screen: bool,
    pub(crate) lindep: f64,
}

/// `(s_kept, s_drop, U_d, J_d)` of [`MetricGradParts`].
type DroppedParts = (Vec<f64>, Vec<f64>, Array2<f64>, Array2<f64>);

/// `(B, eigenvalues ascending, W, gradient parts)` of [`fit_with_metric`].
type MetricFit = (Array2<f64>, Vec<f64>, Array2<f64>, Option<DroppedParts>);

/// The dropped-subspace pieces of [`MetricGradParts`] from `J2 = U s Uᵀ`
/// (`evals`, `evecs`) and the kept indices: `(s_kept, s_drop, U_d, J_d)`,
/// reserved on `ledger` once the dropped count is known. `None` (energy-only
/// build, no ledger) forms nothing.
fn dropped_subspace_parts(
    evals: &Array1<f64>,
    evecs: &Array2<f64>,
    keep: &[usize],
    j3: &Array2<f64>,
    lindep: f64,
    ledger: Option<&mut Ledger>,
) -> Result<Option<DroppedParts>, FerricError> {
    let Some(ledger) = ledger else {
        return Ok(None);
    };
    let naux = evecs.nrows();
    let drop_idx: Vec<usize> = (0..naux).filter(|&k| !(evals[k] > lindep)).collect();
    let nd = drop_idx.len();
    ledger.reserve(
        &format!(
            "RsGdf gradient parts: dropped eigenvectors + projected J3 \
             (n_dropped = {nd}, naux = {naux}, nao² = {})",
            j3.nrows()
        ),
        bytes_of(
            (nd as u64).saturating_mul((naux as u64).saturating_add(j3.nrows() as u64)),
            8,
        ),
    )?;
    let mut u_drop = Array2::<f64>::zeros((naux, nd));
    for (c, &k) in drop_idx.iter().enumerate() {
        for r in 0..naux {
            u_drop[(r, c)] = evecs[(r, k)];
        }
    }
    // J_d = U_dᵀ J3ᵀ = (J3 U_d)ᵀ, (nd, n²)
    let jd = j3.dot(&u_drop).t().as_standard_layout().into_owned();
    let s_kept: Vec<f64> = keep.iter().map(|&k| evals[k]).collect();
    let s_drop: Vec<f64> = drop_idx.iter().map(|&k| evals[k]).collect();
    Ok(Some((s_kept, s_drop, u_drop, jd)))
}

/// `B = (J3 W)ᵀ`, `W = U_keep s_keep^{-1/2}` from `J2 = U s Uᵀ` with the
/// eigenvalues `<= lindep` dropped. Returns `(B, eigenvalues ascending, W,
/// gradient parts)` (`W` is `(naux, naux_kept)`; B is computed from it
/// exactly as before). With `grad_ledger = Some(..)` the dropped-subspace
/// pieces of [`MetricGradParts`] are also formed (reserved on that ledger
/// once the dropped count is known); B is bitwise the same either way.
fn fit_with_metric(
    j2: &Array2<f64>,
    j3: Array2<f64>,
    lindep: f64,
    grad_ledger: Option<&mut Ledger>,
    timings: &mut PbcTimings,
) -> Result<MetricFit, FerricError> {
    let naux = j2.nrows();
    let clock = StageClock::start();
    let (evals, evecs) = j2
        .eigh(UPLO::Upper)
        .map_err(|e| FerricError::Lapack(format!("RsGdf metric eigh: {e}")))?;
    timings.stop("rsgdf metric eigh", &clock);
    let clock = StageClock::start();
    let keep: Vec<usize> = (0..naux).filter(|&k| evals[k] > lindep).collect();
    if keep.is_empty() {
        return Err(FerricError::General(format!(
            "RsGdf: every metric eigenvalue is <= lindep {lindep:e} (max {:e}); the aux basis \
             has no fitting power in the G = 0-dropped kernel",
            evals[naux - 1]
        )));
    }
    let mut w = Array2::<f64>::zeros((naux, keep.len()));
    for (c, &k) in keep.iter().enumerate() {
        let sc = 1.0 / evals[k].sqrt();
        for r in 0..naux {
            w[(r, c)] = evecs[(r, k)] * sc;
        }
    }
    let bt = j3.dot(&w); // (n², nkeep)
    let grad = dropped_subspace_parts(&evals, &evecs, &keep, &j3, lindep, grad_ledger)?;
    drop(j3);
    let b = bt.t().as_standard_layout().into_owned(); // (nkeep, n²)
    timings.stop("rsgdf B = (J3 W)^T", &clock);
    Ok((b, evals.to_vec(), w, grad))
}

/// libint2 (as built for ferric) assumes solid-harmonic shells for l > 1 in
/// 2- and 3-centre two-body integrals and ABORTS the process on a Cartesian
/// one (`engine.impl.h`, `ERI2_PURE_SH`). Refuse such an aux basis up front
/// with a typed error instead.
pub(crate) fn require_pure_aux(aux: &PreparedBasis, who: &str) -> Result<(), FerricError> {
    for (i, sh) in aux.located_shells().iter().enumerate() {
        if sh.l > 1 && !sh.pure {
            return Err(FerricError::Basis(format!(
                "{who}: aux shell {i} has l = {} and is Cartesian; libint2's 2/3-centre \
                 integrals require solid-harmonic (pure) shells above l = 1",
                sh.l
            )));
        }
    }
    Ok(())
}

impl RsGdf {
    /// Build B for `cell` with orbital basis `obs` (built from `cell.mol()`)
    /// and aux basis `aux` (any centres: `PreparedBasis::new(cell.mol(),
    /// aux_set)` for atom-centred aux, or a `SiteBasis`). `s` is the lattice
    /// overlap (e.g. `PeriodicHcore::s`), used by the G = 0 term and the
    /// Madelung shift. See the module doc for the method.
    pub fn build(
        cell: &Cell,
        obs: &PreparedBasis,
        aux: &PreparedBasis,
        s: &Array2<f64>,
        cfg: &RsGdfConfig,
    ) -> Result<Self, FerricError> {
        Self::build_impl(cell, obs, aux, s, cfg, false, false, PairSym::S2).map(|(gdf, _)| gdf)
    }

    /// [`RsGdf::build`] that also retains what the analytic RS-GDF forces
    /// need from the metric solve ([`crate::grad`]'s `*_rsgdf` entry points;
    /// FINDINGS "Iteration 18"): the kept/dropped eigenvalues, the dropped
    /// eigenvectors `U_d` and `U_dᵀ J3ᵀ` (`n_dropped × nao²`, the kept–dropped
    /// Loewner term). B, W, the stats and therefore every energy are bitwise
    /// those of [`RsGdf::build`] (same code path; the extra pieces are copies
    /// taken inside the metric solve, reserved on the build ledger once the
    /// dropped count is known).
    ///
    /// A [`RsGdfConfig::range_split`] is accepted (Gamma forces and stress
    /// walk the same partition, FINDINGS "Iteration 26"); the derivative of
    /// a build made with a [`RangeSplitMutant`] other than `Production` is
    /// refused by the derivative consumers.
    pub fn build_for_gradient(
        cell: &Cell,
        obs: &PreparedBasis,
        aux: &PreparedBasis,
        s: &Array2<f64>,
        cfg: &RsGdfConfig,
    ) -> Result<Self, FerricError> {
        Self::build_impl(cell, obs, aux, s, cfg, false, true, PairSym::S2).map(|(gdf, _)| gdf)
    }

    /// Whether this B carries the gradient parts
    /// ([`RsGdf::build_for_gradient`]).
    pub fn has_gradient_parts(&self) -> bool {
        self.grad.is_some()
    }

    /// The range split B was built with (`None`: today's construction).
    /// The Gamma forces and stress follow it (`split`'s `deriv`); the
    /// k-point build refuses a split config.
    pub fn range_split(&self) -> Option<RangeSplit> {
        self.range_split
    }

    /// The metric-derivative parts kept by [`RsGdf::build_for_gradient`]; `None` for any other
    /// build.
    pub(crate) fn gradient_parts(&self) -> Option<&MetricGradParts> {
        self.grad.as_ref()
    }

    /// [`RsGdf::build`] that ALSO returns the symmetrised periodic metric
    /// `J2`, the 3-index `J3` (both in the G = 0-dropped kernel, before the
    /// metric solve) and the aux function centres — what a per-pair
    /// domain-local fit in the PERIODIC metric needs
    /// ([`crate::lmp2`]). The returned `RsGdf` is bitwise the one
    /// [`RsGdf::build`] gives (same code path; the parts are copies taken
    /// just before the metric solve). The extra `8 naux (naux + nao²)`
    /// bytes are reserved on the build ledger.
    pub fn build_with_fit_parts(
        cell: &Cell,
        obs: &PreparedBasis,
        aux: &PreparedBasis,
        s: &Array2<f64>,
        cfg: &RsGdfConfig,
    ) -> Result<(Self, PeriodicFitParts), FerricError> {
        let (gdf, parts) = Self::build_impl(cell, obs, aux, s, cfg, true, false, PairSym::S2)?;
        let parts = parts.ok_or_else(|| {
            FerricError::General("RsGdf::build_with_fit_parts: parts not retained".into())
        })?;
        Ok((gdf, parts))
    }

    /// TEST ORACLE (FROZEN; do not "improve"): [`RsGdf::build`] with the
    /// pre-s2 Gamma SR 3-centre walk — every ORDERED shell pair, then the μ↔ν
    /// average of `symmetrize_pairs` (module doc "Orbital-pair symmetry").
    /// Bit for bit the build before s2, including its counters (computed =
    /// ordered-equivalent = the ordered count). The production build differs
    /// from it only by `≤ ½|x − y|` per J3 element, `x, y` the two ordered
    /// evaluations (round-off).
    #[doc(hidden)]
    pub fn build_pair_s1_oracle(
        cell: &Cell,
        obs: &PreparedBasis,
        aux: &PreparedBasis,
        s: &Array2<f64>,
        cfg: &RsGdfConfig,
    ) -> Result<Self, FerricError> {
        Self::build_impl(cell, obs, aux, s, cfg, false, false, PairSym::S1Oracle)
            .map(|(gdf, _)| gdf)
    }

    /// Shared body of every Gamma build. `retain_parts` also returns the [`PeriodicFitParts`]
    /// copies, `retain_grad` keeps the [`MetricGradParts`], and `pair_sym` picks the production
    /// (S2) or frozen ordered (S1) SR 3-centre walk. `s` is the `(nao, nao)` overlap.
    #[allow(clippy::too_many_arguments)]
    fn build_impl(
        cell: &Cell,
        obs: &PreparedBasis,
        aux: &PreparedBasis,
        s: &Array2<f64>,
        cfg: &RsGdfConfig,
        retain_parts: bool,
        retain_grad: bool,
        pair_sym: PairSym,
    ) -> Result<(Self, Option<PeriodicFitParts>), FerricError> {
        cfg.validate()?;
        require_pure_aux(aux, "RsGdf")?;
        let n = obs.nbasis();
        let n2 = n * n;
        let naux = aux.nbasis();
        if s.dim() != (n, n) {
            return Err(FerricError::General(format!(
                "RsGdf: overlap has shape {:?}, expected ({n}, {n})",
                s.dim()
            )));
        }
        if naux == 0 || n == 0 {
            return Err(FerricError::General(format!(
                "RsGdf: empty basis (nao = {n}, naux = {naux})"
            )));
        }
        check_obs_on_cell(cell, obs)?;
        let total = StageClock::start();
        let mut timings = PbcTimings::default();
        let clock = StageClock::start();
        let st = Stage {
            cell,
            obs,
            aux,
            obs_sh: gshells(obs, "RsGdf orbital basis")?,
            aux_sh: gshells(aux, "RsGdf aux basis")?,
            omega: cfg.omega,
            thresh: cfg.precision,
            sr_screen: cfg.sr_screen,
            walker: LatticeWalker::new(cell),
        };

        let mut ledger = Ledger::new(crate::budget::resolve(cfg.budget_bytes));
        ledger.reserve(
            &format!("RsGdf n×n matrices (n = {n}: S copy + Madelung S D S temporaries)"),
            bytes_of(n2 as u64, 8 * 4),
        )?;
        ledger.reserve(
            &format!("RsGdf 3-index J3 + B + transpose temporary (naux = {naux}, nao = {n})"),
            bytes_of((n2 as u64).saturating_mul(naux as u64), 8 * 3),
        )?;
        ledger.reserve(
            &format!("RsGdf aux metric + eigenvectors + scaled copy (naux = {naux})"),
            bytes_of((naux as u64).saturating_mul(naux as u64), 8 * 4),
        )?;
        if retain_parts {
            ledger.reserve(
                &format!("RsGdf retained fit parts: J2 + J3 copies (naux = {naux}, nao = {n})"),
                bytes_of(
                    (naux as u64).saturating_mul((naux as u64).saturating_add(n2 as u64)),
                    8,
                ),
            )?;
        }

        // Pair images: every pair whose charge bound times the largest aux
        // charge and potential factor reaches `precision` (pair_ft's radius
        // rule at the equivalent pair threshold).
        let rpair = pair_image_radius(&st, cfg.precision);
        ledger.reserve(
            &format!("RsGdf pair-image list (r_pair = {rpair:.2} Bohr)"),
            bytes_of(cell.translation_count_bound(rpair)?, 24),
        )?;
        let images = cell.translations(rpair)?;

        let gcut = 2.0 * cfg.omega * (1.0 / cfg.precision).ln().sqrt();
        ledger.reserve(
            &format!("RsGdf LR G list (|G| <= {gcut:.3})"),
            gvector_list_bytes(cell, gcut)?,
        )?;
        let gv = half_gvectors(cell, gcut)?;
        // Opt-in range split (`split`) and column rotation of the SR
        // 3-centre walk (module doc "Column rotation"): `None` leaves every
        // stage below as it was, bit for bit.
        let (plan, rotation) = sr_plans(
            &st,
            cfg,
            &images,
            &mut ledger,
            retain_grad,
            pair_sym,
            &mut timings,
        )?;
        let resident_bytes = ledger.resident();
        let chunk_budget = ledger.remaining().min(G_CHUNK_BYTES);
        timings.stop("rsgdf setup (shells, pair images, G list)", &clock);

        // --- SR (real space), LR (G ≠ 0), then the G = 0 term.
        // Transient per-task scratch of the parallel SR walks: one block per
        // possible concurrent task (threads + the spare slot). CHECKED, not
        // reserved, and only after `chunk_budget` is fixed, so no numerical
        // width ever depends on the thread count.
        st.check_sr_scratch(&ledger, "RsGdf", SrBinning::GAMMA)?;
        let clock = StageClock::start();
        let (mut j2, n_sr2) = split::sr_metric(&st, plan.as_ref())?;
        timings.stop("rsgdf SR metric (2-centre)", &clock);
        let clock = StageClock::start();
        let (mut j3, n_sr3, n_sr3_ordered) = gamma_sr_three_index(
            pair_sym,
            &st,
            plan.as_ref(),
            rotation.as_ref(),
            cfg.range_split,
            &images,
        )?;
        timings.stop("rsgdf SR 3-centre", &clock);
        let clock = StageClock::start();
        let mut lr_sub = PbcTimings::default();
        let (n_g_chunks, sink_wall, sink_cpu) = split::lr_accumulate(
            &st,
            plan.as_ref(),
            &gv,
            &mut j2,
            &mut j3,
            chunk_budget,
            LrKernel::Production,
            &mut lr_sub,
        )?;
        let (lr_wall, lr_cpu) = clock.elapsed();
        timings.add(
            "rsgdf LR pair FT",
            (lr_wall - sink_wall).max(0.0),
            match (lr_cpu, sink_cpu) {
                (Some(a), Some(b)) => Some((a - b).max(0.0)),
                (a, None) => a,
                (None, Some(_)) => None,
            },
            1,
        );
        timings.add(
            "rsgdf LR aux FT + GEMM",
            sink_wall,
            sink_cpu.or(lr_cpu.map(|_| 0.0)),
            n_g_chunks as u64,
        );
        // Breakdown of the two LR stages (sub-stages, not leaves) + counters.
        timings.absorb(&lr_sub);
        let clock = StageClock::start();
        let q: Vec<f64> = aux_ft_shells(&st.aux_sh, naux, &[[0.0; 3]])
            .column(0)
            .iter()
            .map(|z| z.re)
            .collect();
        let s_std = s.as_standard_layout();
        let s_flat = s_std.as_slice().expect("standard layout");
        let c0 = PI / (cfg.omega * cfg.omega * cell.volume());
        split::subtract_g0_build(plan.as_ref(), &mut j2, &mut j3, s_flat, &q, c0, cfg.g0);

        let asym_j2 = max_abs_asym(&j2);
        let j2 = 0.5 * (&j2 + &j2.t());
        let asym_j3 = symmetrize_pairs(&mut j3, n);
        timings.stop("rsgdf G=0 + symmetrise", &clock);
        let parts = if retain_parts {
            Some(PeriodicFitParts {
                j2: j2.clone(),
                j3: j3.clone(),
                aux_centers: aux_function_centers(aux),
                lindep: cfg.lindep,
            })
        } else {
            None
        };
        let (b, evals, metric_inv_sqrt, grad_eig) = fit_with_metric(
            &j2,
            j3,
            cfg.lindep,
            retain_grad.then_some(&mut ledger),
            &mut timings,
        )?;
        split::finish(plan.as_ref(), evals[0], &mut timings)?;
        let nkeep = b.nrows();
        let grad = grad_eig.map(|(s_kept, s_drop, u_drop, jd)| MetricGradParts {
            s_kept,
            s_drop,
            u_drop,
            jd,
            sr_screen: cfg.sr_screen,
            lindep: cfg.lindep,
        });

        let madelung = match cfg.exxdiv {
            ExxDiv::None => 0.0,
            ExxDiv::Ewald => madelung_constant(cell)?,
        };
        ferric_core::memory::warn_if_rss_over("ferric-pbc RsGdf", ledger.budget(), 1.1);
        let stats = RsGdfStats {
            naux,
            naux_kept: nkeep,
            n_dropped: naux - nkeep,
            metric_eig_min: evals[0],
            metric_eig_max: evals[naux - 1],
            omega: cfg.omega,
            precision: cfg.precision,
            n_pair_images: images.len(),
            n_sr3_triplets: n_sr3,
            n_sr3_triplets_ordered: n_sr3_ordered,
            n_sr2_pairs: n_sr2,
            n_g_half: gv.len(),
            n_g_chunks,
            asym_j2,
            asym_j3,
            budget_bytes: ledger.budget(),
            resident_bytes,
        };
        for (name, v) in [
            ("rsgdf pair images", stats.n_pair_images),
            ("rsgdf SR2 pairs", stats.n_sr2_pairs),
            ("rsgdf SR3 triplets", stats.n_sr3_triplets),
            (
                "rsgdf SR3 triplets (ordered-equivalent)",
                stats.n_sr3_triplets_ordered,
            ),
            ("rsgdf LR half-G", stats.n_g_half),
            ("rsgdf LR chunks", stats.n_g_chunks),
            ("rsgdf naux", stats.naux),
            ("rsgdf naux kept", stats.naux_kept),
            ("rsgdf aux dropped", stats.n_dropped),
        ] {
            timings.set_counter(name, v as u64);
        }
        timings.finish(&total);
        Ok((
            Self {
                nao: n,
                b,
                s: s.clone(),
                madelung,
                stats,
                metric_inv_sqrt,
                grad,
                timings,
                j_clock: CallClock::default(),
                k_clock: CallClock::default(),
                range_split: cfg.range_split,
            },
            parts,
        ))
    }

    /// The lattice overlap `S` B was built with (the SCF's `S`).
    pub fn overlap(&self) -> &Array2<f64> {
        &self.s
    }

    /// The same B with another exchange-divergence treatment (B does not
    /// depend on it; only the K builder's `v_M` changes). `cell` must be the
    /// cell B was built for.
    pub fn with_exxdiv(mut self, cell: &Cell, exxdiv: ExxDiv) -> Result<Self, FerricError> {
        self.madelung = match exxdiv {
            ExxDiv::None => 0.0,
            ExxDiv::Ewald => madelung_constant(cell)?,
        };
        Ok(self)
    }

    /// `(naux_kept, nao²)` fitted tensor, row k, column `μ·nao+ν`.
    pub fn b(&self) -> &Array2<f64> {
        &self.b
    }

    /// `U_kept s_kept^{-1/2}`, `(naux, naux_kept)`, from the metric solve
    /// (`B = (J3 W)ᵀ` with this `W`).
    pub fn metric_inv_sqrt(&self) -> &Array2<f64> {
        &self.metric_inv_sqrt
    }

    /// Build statistics (counts, conditioning, dropped eigenvalues).
    pub fn stats(&self) -> &RsGdfStats {
        &self.stats
    }

    /// Build stage timings and counters, plus the accumulated J and K build
    /// calls of every builder borrowed from this B so far (`"scf J (rsgdf)"`,
    /// `"scf K (rsgdf)"`). `wall_s` is the BUILD total; the J/K stages lie
    /// outside it (they run later, in the SCF).
    pub fn timings(&self) -> PbcTimings {
        let mut t = self.timings.clone();
        t.add_stage(&self.j_clock.timing("scf J (rsgdf)"));
        t.add_stage(&self.k_clock.timing("scf K (rsgdf)"));
        t
    }

    /// `v_M` applied in K (0 for [`ExxDiv::None`]).
    pub fn madelung(&self) -> f64 {
        self.madelung
    }

    /// TEST/DIAGNOSTIC: the fitted ERI `BᵀB`, `(nao², nao²)` with the
    /// [`crate::dense_aft::DenseAftEri::eri`] layout. Errors above `max_bytes`.
    pub fn fitted_eri(&self, max_bytes: usize) -> Result<Array2<f64>, FerricError> {
        let n2 = self.nao * self.nao;
        let bytes = 8u128 * (n2 as u128) * (n2 as u128);
        if bytes > max_bytes as u128 {
            return Err(FerricError::General(format!(
                "RsGdf::fitted_eri: nao^4 tensor needs {bytes} bytes (nao = {}) > cap {max_bytes} bytes",
                self.nao
            )));
        }
        Ok(self.b.t().dot(&self.b))
    }

    /// J builder borrowing this B.
    pub fn j_builder(&self) -> RsGdfJ<'_> {
        RsGdfJ { gdf: self }
    }

    /// K builder (with the Madelung term) borrowing this B.
    pub fn k_builder(&self) -> RsGdfK<'_> {
        RsGdfK {
            gdf: self,
            madelung: self.madelung,
        }
    }

    /// K builder with an explicit Madelung shift `v_M` (0 = `exxdiv=None`),
    /// ignoring B's own [`RsGdf::madelung`]. Lets one B serve both stages of
    /// the staged Gamma UHF (`crate::uhf::gamma_uhf`) without cloning it.
    pub fn k_builder_with_madelung(&self, madelung: f64) -> RsGdfK<'_> {
        RsGdfK {
            gdf: self,
            madelung,
        }
    }

    /// Gate the aux-group scratch of [`exchange_aux_grouped`] on the build's
    /// resolved budget ([`RsGdfStats::budget_bytes`]), after the tensor
    /// resident during the SCF (B itself; the build's J3 / metric
    /// temporaries are freed by then). Over budget is the typed
    /// `check_alloc` refusal. There is deliberately NO fallback to fewer
    /// groups or to the serial loop: either would make K's bits depend on
    /// the budget. With the build's own budget this gate cannot refuse: B
    /// plus `2 min(EXCHANGE_AUX_GROUPS, naux_kept)` scratch matrices is at
    /// most `3 naux_kept nao²` reals, within the build's J3 + B + transpose
    /// reservation.
    fn check_exchange_scratch(&self) -> Result<(), FerricError> {
        let n = self.nao;
        let naux = self.b.nrows();
        let mut ledger = Ledger::new(self.stats.budget_bytes);
        ledger.reserve(
            &format!("RsGdfK resident B (naux_kept = {naux}, nao = {n})"),
            bytes_of(self.b.len() as u64, 8),
        )?;
        ledger.check(
            &format!(
                "RsGdfK exchange aux-group scratch ({} groups x 2 nao² matrices, nao = {n})",
                exchange_aux_groups(naux).len()
            ),
            exchange_group_scratch_bytes(naux, n, 2, 8),
        )
    }

    /// Checks that the density `d` and output `out` are both `(nao, nao)`; `who` names the builder
    /// in the error.
    fn check(&self, d: &Array2<f64>, out: &Array2<f64>, who: &str) -> Result<(), FerricError> {
        let n = self.nao;
        if d.dim() != (n, n) || out.dim() != (n, n) {
            return Err(FerricError::General(format!(
                "{who}: density {:?} / output {:?}, expected ({n}, {n})",
                d.dim(),
                out.dim()
            )));
        }
        Ok(())
    }
}

/// [`JBuilder`] over an [`RsGdf`]: `J = Σ_k B_k (B_k · D)`. Overwrites `j`.
pub struct RsGdfJ<'a> {
    gdf: &'a RsGdf,
}

impl JBuilder for RsGdfJ<'_> {
    /// `j = Σ_k B_k (B_k · D)` for the `(nao, nao)` density `d`; returns the multiply-add count
    /// (`B` elements). Time accrues to the build's J clock.
    fn build(&mut self, d: &Array2<f64>, j: &mut Array2<f64>) -> Result<usize, FerricError> {
        let gdf = self.gdf;
        gdf.j_clock.time(|| self.build_untimed(d, j))
    }

    /// Stateless: nothing to reset.
    fn reset(&mut self) {}
}

impl RsGdfJ<'_> {
    /// [`JBuilder::build`] without the clock. `B` is `(naux_kept, nao²)` with the AO pair index
    /// `m·nao + n` (row-major), so `D` is flattened row-major.
    fn build_untimed(
        &mut self,
        d: &Array2<f64>,
        j: &mut Array2<f64>,
    ) -> Result<usize, FerricError> {
        self.gdf.check(d, j, "RsGdfJ")?;
        let n = self.gdf.nao;
        let d_std = d.as_standard_layout();
        let dflat = ArrayView1::from(d_std.as_slice().expect("standard layout"));
        let c = self.gdf.b.dot(&dflat); // (nkeep)
        let jflat = self.gdf.b.t().dot(&c); // (n²)
        for m in 0..n {
            for k in 0..n {
                j[(m, k)] = jflat[m * n + k];
            }
        }
        Ok(self.gdf.b.len())
    }
}

/// [`KBuilder`] over an [`RsGdf`]: `K = Σ_k B_k D B_kᵀ + v_M S D S`.
/// Overwrites `k`.
///
/// Does NOT override [`KBuilder::build_from_occ`] (audited for Stage 4): the
/// trait default reconstructs `D = C Cᵀ` and calls [`KBuilder::build`], so
/// the Madelung term is kept. A future DfK-style half-transform override
/// (`Σ_k (B_k C)(B_k C)ᵀ`) MUST also add `v_M S C Cᵀ S`, or it silently drops
/// the Madelung term on the occupied path only (guarded by
/// `tests/pbc_uhf.rs::k_builders_keep_madelung_on_the_occupied_path`).
pub struct RsGdfK<'a> {
    gdf: &'a RsGdf,
    madelung: f64,
}

impl KBuilder for RsGdfK<'_> {
    /// `K = Σ_k B_k D B_kᵀ + v_M S D S` for the `(nao, nao)` density `d`; returns the multiply-add
    /// count. Time accrues to the build's K clock.
    fn build(&mut self, d: &Array2<f64>, k: &mut Array2<f64>) -> Result<usize, FerricError> {
        let gdf = self.gdf;
        gdf.k_clock.time(|| self.build_untimed(d, k))
    }

    /// No density-dependent state: nothing to update.
    fn update_density(&mut self, _d: &Array2<f64>) {}

    /// Stateless: nothing to reset.
    fn reset(&mut self) {}
}

impl RsGdfK<'_> {
    /// [`KBuilder::build`] without the clock: checks shapes and the exchange scratch budget,
    /// accumulates over the fixed aux groups, then adds the Madelung term `madelung · S D S` when
    /// nonzero.
    fn build_untimed(
        &mut self,
        d: &Array2<f64>,
        k: &mut Array2<f64>,
    ) -> Result<usize, FerricError> {
        self.gdf.check(d, k, "RsGdfK")?;
        let n = self.gdf.nao;
        self.gdf.check_exchange_scratch()?;
        exchange_aux_grouped(&self.gdf.b, d, n, k)?;
        if self.madelung != 0.0 {
            let sds = self.gdf.s.dot(d).dot(&self.gdf.s);
            k.scaled_add(self.madelung, &sds);
        }
        Ok(self.gdf.b.len() * n)
    }
}

/// Number of fixed aux groups of the RS-GDF exchange builds
/// (`exchange_aux_grouped`, `kpoint::add_exchange`). A CONSTANT: it never
/// depends on the thread count or the budget, so the grouping (and hence
/// every bit of K) is a pure function of `naux`.
pub const EXCHANGE_AUX_GROUPS: usize = 24;

/// The aux groups of an `naux`-row B: contiguous ranges of
/// `ceil(naux / EXCHANGE_AUX_GROUPS)` rows, ascending, the last one shorter
/// when the size does not divide `naux`. There are
/// `ceil(naux / size) <= EXCHANGE_AUX_GROUPS` of them (never an empty one;
/// `naux < EXCHANGE_AUX_GROUPS` gives `naux` groups of one row; `naux = 0`
/// none).
#[doc(hidden)]
pub fn exchange_aux_groups(naux: usize) -> Vec<std::ops::Range<usize>> {
    if naux == 0 {
        return Vec::new();
    }
    let size = naux.div_ceil(EXCHANGE_AUX_GROUPS);
    (0..naux)
        .step_by(size)
        .map(|a0| a0..(a0 + size).min(naux))
        .collect()
}

/// Bytes of the aux-group scratch of one exchange call: one `n × n`
/// partial plus `per_task_mats − 1` per-aux-row temporaries per group, all
/// counted as live at once (the thread count is not read, so the gate
/// decision is a pure function of the shapes and the budget).
pub(crate) fn exchange_group_scratch_bytes(
    naux: usize,
    n: usize,
    per_task_mats: usize,
    elem_bytes: usize,
) -> usize {
    let groups = exchange_aux_groups(naux).len() as u64;
    bytes_of(
        groups
            .saturating_mul(per_task_mats as u64)
            .saturating_mul((n as u64).saturating_mul(n as u64)),
        elem_bytes,
    )
}

/// `K = Σ_a B_a D B_aᵀ` (`B_a` the `(n, n)` row a of `b`), OVERWRITING `k`.
///
/// Production: the aux rows are split into the fixed contiguous groups of
/// [`exchange_aux_groups`]; one rayon task per group accumulates its own
/// zeroed `(n, n)` partial `K_g` with exactly the serial loop's full-size
/// calls (`tmp = B_a · D`, `K_g += tmp · B_aᵀ`, aux ascending within the
/// group); then `K` is set to zero and the partials are added onto it in
/// ascending group order. Each aux row's GEMM product is the serial
/// loop's; only the association of the aux sum changes, from one running
/// sum to a sum of per-group running sums. The result is therefore NOT
/// bitwise equal to the serial loop ([`exchange_serial_oracle`]) but
/// differs from it at rounding level (measured on dry ice, n = 168, naux =
/// 672: ~25.7k of 28.2k elements differ, max relative 2.9e-15). It is
/// deterministic and independent of the thread count by construction: the
/// grouping depends on `naux` alone, each partial is one task's in-order
/// sum, and the partials are combined in a fixed order.
/// `tests/pbc_parallel_bitwise.rs` checks bits across thread counts and
/// the per-element rounding bound against the serial loop.
///
/// Measured speed-up over the serial loop (dry ice, n = 168, naux = 672,
/// OPENBLAS_NUM_THREADS=1, 6 physical cores): 1.00x at 1 thread, 5.45x at
/// 6 threads. Scratch: [`exchange_group_scratch_bytes`] (two `n × n`
/// matrices per group), gated by the caller.
fn exchange_aux_grouped(
    b: &Array2<f64>,
    d: &Array2<f64>,
    n: usize,
    k: &mut Array2<f64>,
) -> Result<(), FerricError> {
    let partials = exchange_aux_groups(b.nrows())
        .into_par_iter()
        .map(|g| {
            let mut kg = Array2::<f64>::zeros((n, n));
            for a in g {
                let bk = b
                    .row(a)
                    .into_shape_with_order((n, n))
                    .map_err(|e| FerricError::General(format!("RsGdfK: B row reshape: {e}")))?;
                let tmp = bk.dot(d);
                general_mat_mul(1.0, &tmp, &bk.t(), 1.0, &mut kg);
            }
            Ok(kg)
        })
        .collect::<Result<Vec<_>, FerricError>>()?;
    k.fill(0.0);
    for kg in &partials {
        *k += kg;
    }
    Ok(())
}

/// The serial RS-GDF exchange loop: one running sum over the aux rows
/// (FROZEN; oracle only — do not "improve"). Overwrites `k`.
fn exchange_serial_oracle(
    b: &Array2<f64>,
    d: &Array2<f64>,
    n: usize,
    k: &mut Array2<f64>,
) -> Result<(), FerricError> {
    k.fill(0.0);
    for row in b.rows() {
        let bk = row
            .into_shape_with_order((n, n))
            .map_err(|e| FerricError::General(format!("RsGdfK: B row reshape: {e}")))?;
        let tmp = bk.dot(d);
        general_mat_mul(1.0, &tmp, &bk.t(), 1.0, &mut *k);
    }
    Ok(())
}

/// TEST ORACLE for the aux-grouped exchange: `[grouped, serial]`
/// `K = Σ_a B_a D B_aᵀ` for `b` `(naux, n²)` and `d` `(n, n)` (no Madelung
/// term, no budget gate). They agree to rounding, not bit for bit (see
/// [`exchange_aux_grouped`]).
#[doc(hidden)]
pub fn exchange_grouped_and_serial(
    b: &Array2<f64>,
    d: &Array2<f64>,
) -> Result<[Array2<f64>; 2], FerricError> {
    let n = d.nrows();
    if d.dim() != (n, n) || b.ncols() != n * n {
        return Err(FerricError::General(format!(
            "exchange_grouped_and_serial: B {:?} / D {:?}",
            b.dim(),
            d.dim()
        )));
    }
    let mut kp = Array2::<f64>::zeros((n, n));
    let mut ks = Array2::<f64>::zeros((n, n));
    exchange_aux_grouped(b, d, n, &mut kp)?;
    exchange_serial_oracle(b, d, n, &mut ks)?;
    Ok([kp, ks])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::basis::{BasisSet, Shell};
    use ferric_core::mol::{Atom, Molecule};
    use std::collections::HashMap;

    /// A one-hydrogen doublet orbital basis with a single shell of angular
    /// momentum `l` (`pure` or Cartesian), exponent `a` and coefficient 1, on
    /// the atom at `at` (Bohr).
    fn one_shell_prep(l: i32, pure: bool, a: f64, at: [f64; 3]) -> PreparedBasis {
        let mut shells = HashMap::new();
        shells.insert(
            1,
            vec![Shell {
                l,
                pure,
                exponents: vec![a],
                coefficients: vec![1.0],
            }],
        );
        let bs = BasisSet {
            name: "one".into(),
            shells,
            ecps: HashMap::new(),
        };
        let mol = Molecule {
            atoms: vec![Atom {
                symbol: "H".into(),
                z: 1,
                x: at[0],
                y: at[1],
                zpos: at[2],
                ghost: false,
                n_core_ecp: 0,
            }],
            charge: 0,
            multiplicity: 2,
        };
        PreparedBasis::new(&mol, &bs).unwrap()
    }

    #[test]
    fn aux_ft_charge_of_s_is_the_closed_form_and_pure_d_is_neutral() {
        let a = 0.37;
        let s = one_shell_prep(0, false, a, [0.3, -0.2, 1.1]);
        let x = aux_ft(&s, &[[0.0; 3]]).unwrap();
        // unit-self-overlap s: ∫g = 2^{3/4} π^{3/4} a^{-3/4}
        let want = 2f64.powf(0.75) * PI.powf(0.75) * a.powf(-0.75);
        assert!(
            (x[(0, 0)].re - want).abs() < 1e-13 * want,
            "{} vs {want}",
            x[(0, 0)].re
        );
        assert!(x[(0, 0)].im.abs() < 1e-15);
        let d = one_shell_prep(2, true, a, [0.3, -0.2, 1.1]);
        let xd = aux_ft(&d, &[[0.0; 3]]).unwrap();
        assert_eq!(xd.nrows(), 5);
        for p in 0..5 {
            assert!(
                xd[(p, 0)].norm() < 1e-13,
                "pure d charge {p}: {}",
                xd[(p, 0)]
            );
        }
        // Cartesian d: xx/yy/zz carry equal nonzero charge, off-diagonal none.
        let dc = one_shell_prep(2, false, a, [0.0; 3]);
        let xc = aux_ft(&dc, &[[0.0; 3]]).unwrap();
        let q: Vec<f64> = (0..6).map(|p| xc[(p, 0)].re).collect();
        // CCA order: xx, xy, xz, yy, yz, zz
        assert!(q[0] > 0.1 && (q[0] - q[3]).abs() < 1e-13 && (q[0] - q[5]).abs() < 1e-13);
        assert!(q[1].abs() < 1e-15 && q[2].abs() < 1e-15 && q[4].abs() < 1e-15);
    }

    #[test]
    fn aux_ft_phase_is_minus_i_g_dot_a() {
        // Translating the function by A multiplies X(G) by e^{−iG·A}.
        let a = 0.8;
        let g = [[0.7, -0.3, 1.2]];
        let at = [0.4, 1.1, -0.6];
        let x0 = aux_ft(&one_shell_prep(1, false, a, [0.0; 3]), &g).unwrap();
        let x1 = aux_ft(&one_shell_prep(1, false, a, at), &g).unwrap();
        let ph = dot3(&g[0], &at);
        let f = Complex64::new(ph.cos(), -ph.sin());
        for p in 0..3 {
            assert!((x1[(p, 0)] - x0[(p, 0)] * f).norm() < 1e-14);
            assert!(x0[(p, 0)].norm() > 1e-3, "vacuous");
        }
    }

    #[test]
    fn lattice_walker_is_complete() {
        let mol = Molecule {
            atoms: vec![Atom {
                symbol: "H".into(),
                z: 1,
                x: 0.0,
                y: 0.0,
                zpos: 0.0,
                ghost: false,
                n_core_ecp: 0,
            }],
            charge: 0,
            multiplicity: 2,
        };
        let cell = Cell::new(mol, [[4.6, 0.0, 0.0], [0.9, 4.3, 0.0], [0.5, 0.7, 4.8]]).unwrap();
        let w = LatticeWalker::new(&cell);
        let x0 = [1.7, -2.3, 0.4];
        let r = 9.5;
        let mut got = 0usize;
        w.visit(x0, r, |_| {
            got += 1;
            Ok(())
        })
        .unwrap();
        let a = cell.lattice();
        let mut brute = 0usize;
        for n0 in -10i32..=10 {
            for n1 in -10i32..=10 {
                for n2 in -10i32..=10 {
                    let t: Vec<f64> = (0..3)
                        .map(|k| n0 as f64 * a[0][k] + n1 as f64 * a[1][k] + n2 as f64 * a[2][k])
                        .collect();
                    let d =
                        ((t[0] - x0[0]).powi(2) + (t[1] - x0[1]).powi(2) + (t[2] - x0[2]).powi(2))
                            .sqrt();
                    if d <= r {
                        brute += 1;
                    }
                }
            }
        }
        assert!(brute > 20);
        assert_eq!(got, brute);
    }

    #[test]
    fn g0_mutants_differ_only_where_they_should() {
        let mut j2 = Array2::<f64>::zeros((2, 2));
        let mut j3 = Array2::<f64>::zeros((1, 2));
        subtract_g0(
            &mut j2,
            &mut j3,
            &[2.0],
            &[1.0, 3.0],
            0.5,
            G0Handling::Consistent,
        );
        assert_eq!(j2[(0, 1)], -1.5);
        assert_eq!(j3[(0, 1)], -3.0);
        let mut j2m = Array2::<f64>::zeros((2, 2));
        let mut j3m = Array2::<f64>::zeros((1, 2));
        subtract_g0(
            &mut j2m,
            &mut j3m,
            &[2.0],
            &[1.0, 3.0],
            0.5,
            G0Handling::MetricOnlyMutant,
        );
        assert_eq!(j2m, j2);
        assert_eq!(j3m[(0, 1)], 0.0);
    }
}
