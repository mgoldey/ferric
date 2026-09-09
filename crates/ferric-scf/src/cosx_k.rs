//! COSX (chain-of-spheres) seminumerical exchange builder.
//!
//! Neese, Wennmohs, Hansen, Becker, Chem. Phys. 356, 98 (2009); overlap fit
//! from Izsák & Neese, JCP 135, 144105 (2011). A faithful port of the
//! validated Stage 0 prototype `scripts/cosx_proto.py`, whose working
//! equations are the spec for the numerics:
//!
//! ```text
//!     X_{mu,g}   = sqrt(w_g) chi_mu(r_g)                    (nbf, npts)
//!     F_{lam,g}  = (D X)_{lam,g}                            half transform
//!     A^g        = [ \int chi_mu chi_lam / |r - r_g| ]      per grid point
//!     G_{nu,g}   = sum_lam A^g_{nu,lam} F_{lam,g}
//!     Ktilde     = X G^T
//!     K_plain    = 0.5 (Ktilde + Ktilde^T)                  REQUIRED
//!     K_fit      = 0.5 (Q Ktilde + (Q Ktilde)^T),  Q = S S_num^{-1},
//!                  S_num = X X^T  (density-independent; Cholesky, never inv)
//! ```
//!
//! The `A^g` blocks come from one of two backends (`CosxConfig::backend`):
//!
//! * `CosxBackend::Md3c1e` (default): `ferric_integrals::md3c1e`, a batched
//!   McMurchie–Davidson 3c1e kernel. Per sub-batch of `COSX_SUB_BATCH_POINTS`
//!   points it sweeps shell pairs `s1 >= s2` ONCE and the contraction
//!   `G = A^g F` is accumulated straight from each `[nf1][nf2][g]` block —
//!   `G[o1+i][g] += blk[i][j][g] F[o2+j][g]` plus the `s1 != s2` mirror
//!   `G[o2+j][g] += blk[i][j][g] F[o1+i][g]` — so no `A^g` matrix is ever
//!   materialized. Exact vs the libint2 path to round-off (anchored).
//! * `CosxBackend::CosxA`: `ferric_integrals::cosx_a` (libint2
//!   nuclear-attraction engine with a unit probe charge, sign already flipped
//!   to the repulsive `+1/|r-r_g|` COSX needs), one dense `A^g` per point then
//!   a GEMV. Measured 3.2–3.5x slower per point; kept as the cross-check.
//!
//! # Grid, blocking, determinism
//!
//! The builder owns its OWN Becke–Lebedev grid (`CosxConfig::grid`, default
//! (50,110) — the measured operating point: the composed-budget audit on this
//! branch found grids coarser than (50,110) fail the 0.1 kcal/mol isodesmic
//! reaction-energy bar, and at (50,110) the overlap fit took that error from
//! 0.2068 to 0.0190 kcal/mol, so the fit defaults ON). Points are processed
//! in fixed blocks of `COSX_BLOCK_POINTS` points; per block the three `(nbf, B)`
//! planes `X`, `F`, `G` are resident. The A-build inside a block is parallel
//! over points (cosx_a: one libint2 engine per rayon worker) or over fixed
//! sub-batches of `COSX_SUB_BATCH_POINTS` points (md3c1e: one kernel scratch
//! per worker). Every parallel write is to its own columns of `G`, the
//! accumulation order within a column is the fixed shell-pair order, and the
//! two GEMMs per block run in block order, so the result is bit-identical
//! across thread counts (block and sub-batch partitions are pure functions of
//! the point count).
//!
//! # Sparse half transforms
//!
//! Measured on main (`scripts/queue/out/cosx_scaling_results.md`, def2-SVP
//! alkanes, one thread): with the density-driven screen the A-build is
//! N^1.54 but the full K is N^2.16, because the two dense block GEMMs
//! `F = D X` and `Ktilde += X G^T` (nbf x nbf x B each) are 32% of the build
//! at C20. `CosxHalfTransform::Sparse` (default) makes them shell-sparse per
//! block, the standard COSX/sn-LinK structure:
//!
//! ```text
//!     A = { shells s : max_{mu in s, g in block} |X_{mu g}| >= eps_ao }   (X is local to the block)
//!     Λ = { shells l : max_{s in A} max|D_{l s}| >= eps_d }               (D decays with distance)
//!     F[Λ, blk] = D[Λ, A] X[A, blk],   F[not Λ] = 0 exactly
//!     B = shells touched by the kernel's surviving pairs (G[not B] = 0 exactly)
//!     Ktilde[A, B] += X[A, blk] G[B, blk]^T
//!     S_num[A, A]  += X[A, blk] X[A, blk]^T
//! ```
//!
//! Both GEMMs become `|A| x |B| x B` with `|A|, |B| -> O(1)` as the molecule
//! outgrows the AO reach (~12 Bohr at 1e-10 for def2-SVP) and the density
//! decay length (~30 Bohr for alkanes). Both thresholds compare with `>=`,
//! so `eps = 0` is the vacuous mask and reproduces `Dense` bit-for-bit; the
//! defaults come from an error model, not a sweep (`COSX_DEFAULT_EPS_AO`).
//! `Dense` is the pre-sparse builder and the cross-check
//! (`tests/cosx_sparse_anchors.rs`). ferric-dft's KS path has no AO-on-grid
//! screening to share (its density loop is a dense `D chi`), so the mask
//! machinery lives here.
//!
//! # Scope
//!
//! Reached via `k_builder = "cosx"` from `solve_rhf`, `solve_uhf` and
//! `solve_rohf` alike (until 2026-09-08 the open-shell solvers never read
//! `k_builder`). The open-shell path drives ONE instance for both spins: this
//! builder holds no density-dependent state (`update_density` is a no-op) and
//! its `S_num` overlap-fit factor is geometry-only, so K(D_α) is unaffected by
//! an interleaved K(D_β) build — anchored in `tests/k_builder_open_shell.rs`.
//! Single MPI rank only — a multi-rank
//! `ParallelContext` is refused at construction rather than silently
//! computing the full K on every rank. Weights enter as `sqrt(|w|)`, as in
//! the prototype (Becke weights are non-negative in practice).

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use ferric_core::mol::Molecule;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_dft::grid::{build_atomic_grid_pruned, AtomicGridConfig};
use ferric_integrals::ao_grid::eval_basis_on_points;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::cosx_a::{a_matrix_at_point_with, CosxScreen, PairBounds};
use ferric_integrals::engine::Engine;
use ferric_integrals::ffi;
use ferric_integrals::md3c1e::{Md3c1e, Md3c1eScratch};
use ndarray::Array2;
use ndarray_linalg::{Cholesky, Diag, SolveTriangular, UPLO};
use rayon::prelude::*;

use crate::fock::KBuilder;

/// Grid points per block. Fixed (not budget-derived) so the floating-point
/// association of the per-block GEMM accumulation — and hence the K matrix —
/// does not depend on the memory budget. 1024 points × 3 planes × nbf × 8 B
/// is ~300 MB at nbf = 12 800 (320 atoms / TZVP).
pub const COSX_BLOCK_POINTS: usize = 1024;

/// Grid points per md3c1e kernel call inside a block. Measured (butane,
/// def2-SVP/TZVP/QZVP, `scripts/queue/out/md3c1e_results.md`): the per-batch
/// primitive-pair setup is amortized by 256 points (B=64 -> 256 gained 7-14%,
/// 256 -> 1024 was flat), so 256 is where the kernel saturates. Fixed, not
/// budget-derived, for the same reason as `COSX_BLOCK_POINTS`.
pub const COSX_SUB_BATCH_POINTS: usize = 256;

/// AO-evaluation chunk inside a block (parallel over chunks).
const AO_EVAL_CHUNK: usize = 64;

/// Which 3c1e kernel supplies the `A^g` blocks (see the module doc).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CosxBackend {
    /// Batched McMurchie–Davidson kernel (`ferric_integrals::md3c1e`). Default.
    #[default]
    Md3c1e,
    /// libint2 nuclear-attraction engine, one dense `A^g` per point
    /// (`ferric_integrals::cosx_a`). Slower; the cross-check backend.
    CosxA,
}

impl CosxBackend {
    /// Strict config-string parser: `"md3c1e"` or `"cosx-a"`; anything else is
    /// an error (never a silent default), per the config-honesty convention.
    pub fn parse_config_str(s: &str) -> Result<Self, FerricError> {
        match s {
            "md3c1e" => Ok(Self::Md3c1e),
            "cosx-a" => Ok(Self::CosxA),
            other => Err(FerricError::General(format!(
                "unknown cosx_backend '{other}': valid options are 'md3c1e' (default) and 'cosx-a'"
            ))),
        }
    }

    /// The config-string spelling of this backend (inverse of `parse_config_str`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Md3c1e => "md3c1e",
            Self::CosxA => "cosx-a",
        }
    }
}

/// How the per-block half transforms `F = D X`, `Ktilde += X G^T` and the
/// fit's `S_num += X X^T` are evaluated (see the module doc, "Sparse half
/// transforms").
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CosxHalfTransform {
    /// Dense `nbf x nbf x B` GEMMs on the full block planes — the pre-sparse
    /// builder, byte-identical to it, kept as the cross-check path.
    Dense,
    /// Shell-sparse: per block, `A` = shells with `max |X| >= eps_ao`,
    /// `Λ` = shells with `max_{s in A} max|D_{Λ s}| >= eps_d`, `B` = shells
    /// touched by the kernel's surviving pairs; then `F[Λ] = D[Λ,A] X[A]`,
    /// `Ktilde[A,B] += X[A] G[B]^T`, `S_num[A,A] += X[A] X[A]^T`. Both `eps`
    /// compare with `>=`, so `0.0` keeps every shell (exact zeros included)
    /// and reproduces `Dense` bit-for-bit (anchored).
    Sparse {
        /// Active-AO threshold on `|X| = sqrt(w) |chi|` over the block.
        eps_ao: f64,
        /// Output-row threshold on the shell-block maxima of `|D|`.
        eps_d: f64,
    },
}

/// Default active-AO threshold (`CosxHalfTransform::Sparse::eps_ao`), from
/// the error model in `scripts/queue/out/cosx_sparse_prereg.md` (NOT tuned to
/// a measurement): the worst-case `Ktilde` perturbation from dropping
/// `|X| < eps_ao` rows is `~ sqrt(npts) ||A^g||_inf ||D||_inf eps_ao ~ 1e4 eps_ao`,
/// so 1e-10 puts the 1e-6 accuracy bar 1e2 away at the bound; the AO reach
/// only grows as `sqrt(ln(1/eps))`, so 1e-10 vs 1e-8 costs ~12% reach.
pub const COSX_DEFAULT_EPS_AO: f64 = 1e-10;

/// Default output-row threshold on `|D|` (`CosxHalfTransform::Sparse::eps_d`);
/// same error model and the same value as `COSX_DEFAULT_EPS_AO`.
pub const COSX_DEFAULT_EPS_D: f64 = 1e-10;

impl CosxHalfTransform {
    /// The sparse path at the pre-registered default thresholds.
    pub const SPARSE_DEFAULT: Self = Self::Sparse { eps_ao: COSX_DEFAULT_EPS_AO, eps_d: COSX_DEFAULT_EPS_D };

    /// Strict config-string parser: `"sparse"` (default thresholds) or
    /// `"dense"`; anything else is an error, never a silent default.
    pub fn parse_config_str(s: &str) -> Result<Self, FerricError> {
        match s {
            "sparse" => Ok(Self::SPARSE_DEFAULT),
            "dense" => Ok(Self::Dense),
            other => Err(FerricError::General(format!(
                "unknown cosx_half_transform '{other}': valid options are 'sparse' (default) and 'dense'"
            ))),
        }
    }

    /// The config-string spelling (`"sparse"` for any thresholds).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Sparse { .. } => "sparse",
        }
    }
}

/// Lebedev orders ferric's quadrature tables provide. `ferric_quadrature::lebedev`
/// PANICS on any other order, so a grid config is validated against this list
/// up front and rejected with a typed error instead.
pub const SUPPORTED_ANGULAR_ORDERS: [usize; 6] = [6, 14, 26, 50, 110, 302];

/// Validate a COSX grid config: positive radial count and a tabulated Lebedev order.
pub fn validate_grid(grid: &AtomicGridConfig) -> Result<(), FerricError> {
    if grid.n_radial == 0 {
        return Err(FerricError::General("cosx grid: radial point count must be > 0".into()));
    }
    if !SUPPORTED_ANGULAR_ORDERS.contains(&grid.n_angular) {
        return Err(FerricError::General(format!(
            "cosx grid: angular order {} is not tabulated (supported: {:?})",
            grid.n_angular, SUPPORTED_ANGULAR_ORDERS
        )));
    }
    Ok(())
}

/// User-facing COSX knobs. Carried in `RhfConfig::cosx`.
#[derive(Debug, Clone)]
pub struct CosxConfig {
    /// The exchange grid. Default (50,110), unpruned — see the module doc for
    /// why not coarser.
    pub grid: AtomicGridConfig,
    /// Overlap fitting `K = 0.5(S S_num^{-1} Ktilde + h.c.)`. Default `true`.
    /// Measured to be net-negative on grids coarser than (50,110) and a ~10x
    /// reaction-energy improvement at (50,110) (water-favourable in that
    /// audit); disable to get the plain symmetrized Ktilde.
    pub overlap_fit: bool,
    /// DENSITY-DRIVEN shell-pair screen threshold for the A-build (md3c1e
    /// backend). Per sub-batch of `COSX_SUB_BATCH_POINTS` points, with
    /// `F = D X` already in hand, pair `(s1, s2)` is evaluated iff
    ///
    /// ```text
    ///     bound_A(s1, s2, batch) * max(fmax[s1], fmax[s2]) >= t,
    ///     fmax[s] = max_{mu in s, g in batch} |F_{mu,g}|,
    /// ```
    ///
    /// where `bound_A` is the Hölder pair bound of `ferric_integrals::cosx_screen`
    /// (`PairBounds::coarse_estimate_sphere` over the batch's bounding sphere;
    /// never underestimates `max|A^g_{mu,nu}|`, anchored there over all
    /// `(la, lb)` to `(4, 4)`). The `max` over BOTH shells is required because
    /// the block feeds `G_{s1} += A F_{s2}` AND the mirror `G_{s2} += A F_{s1}`;
    /// a dropped block therefore perturbs every `G` element by `< t`. This is
    /// the Neese (2009) S-junction / sn-LinK `eps^K` idea: an integral-only
    /// screen keeps every significant pair at every point (kept fraction ~1 on
    /// small molecules, ~N^1 per point on alkanes — the A-build stays O(N^2)),
    /// while `F` is local to the point and is what makes the kept work O(N).
    ///
    /// `None` = unscreened (no pair bounds built at all). `Some(0.0)` is the
    /// screen's trivial limit and reproduces `None` bit-for-bit (anchored: the
    /// same blocks in the same order). The cosx_a backend has no batched
    /// screen; `Some(t > 0)` with `CosxBackend::CosxA` is refused.
    ///
    /// History: the first screen (2026-09-06) bounded a pair by
    /// `sqrt(max|S_block|)/d` from the SIGNED overlap, which is exactly zero
    /// for same-centre pairs whose overlap vanishes by symmetry (s–p, px–py)
    /// while their `A^g` is O(1) — water/cc-pVDZ at `t = 1e-7` gave
    /// `max|K_scr - K_unscr| = 0.71`. `t > 0` was refused until the bound was
    /// replaced (2026-09-07); `cosx_screened_k_matches_unscreened_below_grid_error`
    /// in `tests/cosx_k_anchors.rs` now pins the positive result.
    pub screen_thresh: Option<f64>,
    /// Which 3c1e kernel builds the `A^g` blocks. Default `Md3c1e`; `CosxA`
    /// is the slower libint2 path, kept so the two stay cross-checkable
    /// (`cosx_k_md3c1e_matches_cosx_a_backend` in `tests/cosx_k_anchors.rs`).
    pub backend: CosxBackend,
    /// Dense or shell-sparse block half transforms. Default
    /// `CosxHalfTransform::SPARSE_DEFAULT`; `Dense` is the byte-identical
    /// pre-sparse builder (`tests/cosx_sparse_anchors.rs`).
    pub half_transform: CosxHalfTransform,
    /// Points per group in the SCREENING decision (md3c1e backend). The
    /// kernel's `COSX_SUB_BATCH_POINTS` blocking is NOT affected — this splits
    /// only the bound test, and a pair kept by ANY group is evaluated for the
    /// whole sub-batch, so correctness is preserved by construction.
    ///
    /// `0` (the default) means one group per sub-batch, i.e. the unsplit
    /// screen, reproduced BITWISE (`tests/cosx_group_screen_anchors.rs`).
    ///
    /// # Why this knob exists, and why its default is `0`
    ///
    /// A 256-point Becke sub-batch is ~2.3 whole Lebedev spheres of one atom
    /// (the grid is emitted atom-major / radial-major / angular-minor), so its
    /// enclosing region is spatially large and `R_c` clamps to 0: the bound
    /// degenerates to its distance-free value on 68-80% of (pair, batch)
    /// decisions (`scripts/queue/out/snlink_python_results.md` §4.1). Splitting
    /// the bound test into contiguous groups shrinks the region and was
    /// expected to recover that.
    ///
    /// MEASURED (2026-09-09, butane/def2-SVP, (50,110)+fit, `t = 1e-7`, counts
    /// so exact rather than indicative): it does not pay.
    ///
    /// ```text
    ///   group | degenerate |     kept | bound evals
    ///       0 |     0.6515 | 0.791573 |     446 985   (unsplit)
    ///      64 |     0.6419 | 0.790568 |     756 377   (1.69x)
    ///      32 |     0.6441 | 0.790597 |   1 168 970   (2.62x)
    ///      16 |     0.6465 | 0.790599 |   1 994 693   (4.46x)
    ///       8 |     0.6423 | 0.790406 |   3 651 199   (8.17x)
    /// ```
    ///
    /// Kept work falls 0.12 PERCENTAGE POINTS for 8.2x the bound evaluations,
    /// and the degenerate fraction barely moves and not monotonically. The
    /// mechanism is live, not inert — forcing every group back to the whole
    /// sub-batch's region freezes `kept` at exactly the unsplit 90 512 912,
    /// where the real code reaches 90 379 536.
    ///
    /// `tests/cosx_region_diagnostics.rs` says why, and the reason is
    /// structural: only 33.6% of the degeneracy is the REGION reaching the pair
    /// midpoint (the part grouping can fix — it moves that 33.6% -> 24.2%),
    /// while 30.4% is `|AB|/2` eating the distance, which grouping cannot touch
    /// at all and whose share GROWS to 35.3% as the region shrinks. The screen
    /// is blind on most decisions because of the SHELL PAIRS, not because of
    /// the batch geometry — so a tighter enclosing volume is the wrong lever.
    ///
    /// The knob is kept so the measurement stays reproducible, and its default
    /// is `0`. A non-zero value is NOT recommended. Pinned by
    /// `tests/cosx_group_screen_anchors.rs`, which is written to FAIL if this
    /// verdict is ever overturned.
    pub screen_group: usize,
}

/// Default density-driven screen threshold (`CosxConfig::screen_thresh`).
/// Measured 2026-09-07 (`tests/cosx_screen_sweep.rs`, (50,110)+fit, one
/// thread, `max|K_scr - K_unscr|` vs the grid error against the direct K):
///
/// ```text
///   water/cc-pVDZ   grid err 5.6e-5 | t=1e-6: 3.1e-8 | t=1e-7: 1.6e-10 | t=1e-8: 2.9e-12
///   butane/def2-SVP grid err 2.9e-4 | t=1e-6: 1.3e-5 | t=1e-7: 7.6e-7  | t=1e-8: 7.4e-8
/// ```
///
/// 1e-7 is the loosest value that keeps the screen error below 1e-6 on both
/// (2.6e-3 of the butane grid error); 1e-6 breaks the bar on butane.
pub const COSX_DEFAULT_SCREEN_THRESH: f64 = 1e-7;

impl Default for CosxConfig {
    fn default() -> Self {
        Self {
            grid: AtomicGridConfig { n_radial: 50, n_angular: 110, ..Default::default() },
            overlap_fit: true,
            screen_thresh: Some(COSX_DEFAULT_SCREEN_THRESH),
            backend: CosxBackend::Md3c1e,
            half_transform: CosxHalfTransform::SPARSE_DEFAULT,
            screen_group: 0,
        }
    }
}

/// Wall-time split of the most recent `build`, for measurement. Seconds are
/// summed over all worker threads for the A-build segments (`a_build_s`,
/// `contract_s`), i.e. they equal wall time only on one thread.
#[derive(Debug, Clone, Copy, Default)]
pub struct CosxTimings {
    /// AO values on the grid (`X = sqrt(w) chi`), all blocks.
    pub ao_eval_s: f64,
    /// `A^g` construction, summed over threads: the libint2 per-point sweep
    /// (cosx_a) or the md3c1e kernel time with the callback time subtracted.
    pub a_build_s: f64,
    /// The `G = A^g F` contraction, summed over threads: per-point GEMVs
    /// (cosx_a) or the in-callback block accumulation (md3c1e).
    pub contract_s: f64,
    /// Block GEMMs: `F = D X` (or `C (C^T X)`), `Ktilde += X G^T`, `S_num += X X^T`
    /// (`= half_s + ktilde_s + snum_s`; sparse: including their gathers/scatters).
    pub blas_s: f64,
    /// The `F` half transform alone (sparse: `D[Λ,A]` gather + GEMM + row scatter).
    pub half_s: f64,
    /// `Ktilde += X G^T` alone (sparse: `G[B]` gather + GEMM + scatter-add).
    pub ktilde_s: f64,
    /// `S_num += X X^T` alone (first build with the fit only).
    pub snum_s: f64,
    /// Sparse only: mask construction (`A` from `X`, `Λ` from the shell-block
    /// maxima of `D`) plus the `X[A]` gather. 0 on the dense path.
    pub gather_s: f64,
    /// Mean over blocks of `|A| / nbf` (active AOs; 1.0 on the dense path).
    pub active_ao_frac: f64,
    /// Smallest per-block `|A| / nbf`.
    pub active_ao_frac_min: f64,
    /// Largest per-block `|A| / nbf`.
    pub active_ao_frac_max: f64,
    /// Mean over blocks of `|Λ| / nbf` (rows of `F` computed; 1.0 dense).
    pub lambda_frac: f64,
    /// Mean over blocks of `|B| / nbf` (columns of `Ktilde` accumulated; 1.0
    /// dense, and 1.0 for the cosx_a backend which reports no touched shells).
    pub out_ao_frac: f64,
    /// Blocks processed (the denominator of the three means).
    pub n_blocks: usize,
    /// Overlap-fit finalization (Cholesky on first build + two triangular solves + `S Z`).
    pub fit_s: f64,
    /// Whole `build` call, wall.
    pub total_s: f64,
    /// Shell pairs evaluated after screening, summed over points.
    pub pairs_kept: usize,
    /// Shell pairs that pass the GEOMETRY-ONLY part of the screen
    /// (`bound_A >= t`, i.e. what an integral-magnitude screen would keep),
    /// summed over points. Equals `pairs_total` when unscreened;
    /// `pairs_kept <= pairs_kept_geom` always. Diagnostic only — it is what
    /// lets the density-driven screen be compared against an integral-only
    /// one without a second mode.
    pub pairs_kept_geom: usize,
    /// Shell pairs considered, summed over points.
    pub pairs_total: usize,
    /// Region-bound evaluations performed by the screen, summed over points.
    /// The COST side of `CosxConfig::screen_group`: grouping by `G` multiplies
    /// this by up to `G` (less, because the scan stops at the first group that
    /// keeps the pair). 0 when unscreened.
    pub bound_evals: usize,
    /// Of `bound_evals`, how many returned the bound's distance-free `R = 0`
    /// value — the screen running blind. This is the fraction the sub-batched
    /// screen exists to move; see `CosxConfig::screen_group` for what it was
    /// measured to do.
    pub screen_degenerate: usize,
}

/// One `T` per rayon worker (plus a spare for non-pool threads) — same
/// rationale as `ferric_integrals::engine_pool`, which only builds 2e
/// engines. Holds libint2 nuclear engines (cosx_a) or kernel scratch (md3c1e).
struct ThreadSlots<T> {
    slots: Vec<Mutex<T>>,
}

impl<T> ThreadSlots<T> {
    fn new(mut make: impl FnMut() -> Result<T, FerricError>) -> Result<Self, FerricError> {
        let n = rayon::current_num_threads().max(1) + 1;
        let mut slots = Vec::with_capacity(n);
        for _ in 0..n {
            slots.push(Mutex::new(make()?));
        }
        Ok(Self { slots })
    }

    #[inline]
    fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        let idx = rayon::current_thread_index().unwrap_or(self.slots.len() - 1);
        let slot = idx.min(self.slots.len() - 1);
        let mut v = self.slots[slot].lock().unwrap();
        f(&mut v)
    }
}

/// Backend-specific per-worker state, created on the first `build` (the slot
/// count follows the rayon pool the build runs in).
enum Workers {
    CosxA(ThreadSlots<Engine>),
    Md3c1e(ThreadSlots<Md3c1eScratch>),
}

/// Cached Cholesky factor of `S_num = X X^T` (lower `L` and its transpose,
/// both in standard layout for the LAPACK triangular solves).
struct SnumFactor {
    l: Array2<f64>,
    lt: Array2<f64>,
}

/// COSX exchange builder. See the module doc.
pub struct CosxK<'a> {
    ctx: &'a ParallelContext,
    mol: &'a Molecule,
    prep: &'a PreparedBasis,
    cfg: CosxConfig,
    points: Vec<[f64; 3]>,
    sqrt_w: Vec<f64>,
    bounds: Option<PairBounds>,
    /// Analytic AO overlap `S`, only when `overlap_fit`.
    s_ao: Option<Array2<f64>>,
    /// Geometry-only; built during the first `build` and reused.
    snum: Option<SnumFactor>,
    /// md3c1e kernel state (`Some` iff `cfg.backend == Md3c1e`); geometry-only.
    kernel: Option<Md3c1e>,
    workers: Option<Workers>,
    /// AO offset of every shell (libint2 order; the sparse masks are shell-granular).
    shell_off: Vec<usize>,
    /// Functions per shell.
    shell_dim: Vec<usize>,
    last: CosxTimings,
}

impl<'a> std::fmt::Debug for CosxK<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CosxK")
            .field("cfg", &self.cfg)
            .field("npts", &self.points.len())
            .finish_non_exhaustive()
    }
}

impl<'a> CosxK<'a> {
    /// Build the grid (and pair bounds / overlap if configured) for `mol`.
    ///
    /// `mem_budget` is the resolved byte budget (0 = resolve via
    /// `ferric_core::memory::resolve_budget_bytes`); construction fails fast
    /// if one block's scratch plus the per-thread `A^g` matrices exceed it.
    pub fn new(
        ctx: &'a ParallelContext,
        mol: &'a Molecule,
        prep: &'a PreparedBasis,
        cfg: CosxConfig,
        mem_budget: usize,
    ) -> Result<Self, FerricError> {
        if ctx.size > 1 {
            return Err(FerricError::General(format!(
                "CosxK: MPI striping is not implemented (world size {}); run COSX on a single rank",
                ctx.size
            )));
        }
        validate_grid(&cfg.grid)?;
        let grid = build_atomic_grid_pruned(mol, &cfg.grid, cfg.grid.prune)?;
        if grid.is_empty() {
            return Err(FerricError::General("CosxK: empty exchange grid".into()));
        }
        let points: Vec<[f64; 3]> = grid.iter().map(|p| p.xyz).collect();
        let sqrt_w: Vec<f64> = grid.iter().map(|p| p.weight.abs().sqrt()).collect();
        check_budget(prep.nbasis(), cfg.backend, cfg.half_transform, mem_budget)?;
        if let CosxHalfTransform::Sparse { eps_ao, eps_d } = cfg.half_transform {
            if !(eps_ao >= 0.0 && eps_ao.is_finite() && eps_d >= 0.0 && eps_d.is_finite()) {
                return Err(FerricError::General(format!(
                    "CosxK: sparse half-transform thresholds must be finite and >= 0 (eps_ao = {eps_ao}, eps_d = {eps_d}); 0 is the dense limit"
                )));
            }
        }
        let kernel = match cfg.backend {
            CosxBackend::Md3c1e => Some(Md3c1e::new(prep).map_err(|e| {
                FerricError::General(format!(
                    "CosxK: the md3c1e backend cannot handle this basis ({e}); select cosx_backend = \"cosx-a\""
                ))
            })?),
            CosxBackend::CosxA => None,
        };

        if cfg.backend == CosxBackend::CosxA && matches!(cfg.screen_thresh, Some(t) if t > 0.0) {
            return Err(FerricError::General(
                "CosxK: screen_thresh > 0 is implemented for the md3c1e backend only (density-driven, per \
                 sub-batch); the cosx-a cross-check backend runs unscreened. Use backend Md3c1e, or set \
                 screen_thresh to None / Some(0.0) with CosxA (the CLI does this for cosx_backend = \"cosx-a\")"
                    .into(),
            ));
        }
        let bounds = match cfg.screen_thresh {
            Some(_) => Some(PairBounds::build(prep)?),
            None => None,
        };
        let s_ao = cfg.overlap_fit.then(|| ferric_integrals::oneelectron::overlap(prep));

        Ok(Self {
            ctx,
            mol,
            prep,
            cfg,
            points,
            sqrt_w,
            bounds,
            s_ao,
            snum: None,
            kernel,
            workers: None,
            shell_off: prep.shell_offsets()[..prep.nshells()].to_vec(),
            shell_dim: prep.shell_dims().to_vec(),
            last: CosxTimings::default(),
        })
    }

    /// Number of grid points in the exchange grid.
    pub fn npts(&self) -> usize {
        self.points.len()
    }

    /// Timing split of the most recent `build` / `build_from_occ`.
    pub fn last_timings(&self) -> &CosxTimings {
        &self.last
    }

    /// The configuration this builder was constructed with.
    pub fn config(&self) -> &CosxConfig {
        &self.cfg
    }

    /// `X_blk = sqrt(w) chi` for one block of points, `(nbf, B)`.
    fn eval_x_block(&self, pts: &[[f64; 3]], sw: &[f64]) -> Result<Array2<f64>, FerricError> {
        let nbf = self.prep.nbasis();
        let bs = self.prep.basis_set();
        let chunks: Vec<Array2<f64>> = pts
            .par_chunks(AO_EVAL_CHUNK)
            .map(|c| {
                eval_basis_on_points(self.mol, bs, c)
                    .map_err(|e| FerricError::General(format!("CosxK AO eval: {e:?}")))
            })
            .collect::<Result<_, _>>()?;
        let mut x = Array2::<f64>::zeros((nbf, pts.len()));
        let mut col = 0usize;
        for c in &chunks {
            let w = c.ncols();
            x.slice_mut(ndarray::s![.., col..col + w]).assign(c);
            col += w;
        }
        for (g, &s) in sw.iter().enumerate() {
            x.column_mut(g).mapv_inplace(|v| v * s);
        }
        Ok(x)
    }

    /// Screen arguments for the A-build (`None` bounds + vacuous screen unless
    /// `screen_thresh` is set).
    fn screen_args(&self) -> (Option<&PairBounds>, CosxScreen) {
        match self.cfg.screen_thresh {
            None => (None, CosxScreen::none()),
            Some(t) => (self.bounds.as_ref(), CosxScreen::at(t)),
        }
    }

    /// `G_{nu,g} = sum_lam A^g_{nu,lam} F_{lam,g}` for one block, `(nbf, B)`,
    /// dispatched on the backend's worker state. The second value is the set
    /// of shells whose rows of `G` were written (md3c1e: both shells of every
    /// surviving pair, union over the block's sub-batches — rows outside it
    /// are exactly zero); `None` from the cosx_a backend, which builds every
    /// row (the sparse path then takes `B` = all shells).
    fn contract_block(
        &self,
        workers: &Workers,
        pts: &[[f64; 3]],
        f: &Array2<f64>,
        acc: &BlockCounters,
    ) -> Result<(Array2<f64>, Option<Vec<bool>>), FerricError> {
        match workers {
            Workers::CosxA(engines) => Ok((self.contract_block_cosx_a(engines, pts, f, acc)?, None)),
            Workers::Md3c1e(scratch) => {
                let kern = self.kernel.as_ref().expect("md3c1e kernel built in new() for this backend");
                self.contract_block_md3c1e(kern, scratch, pts, f, acc)
            }
        }
    }

    /// cosx_a backend: per-point dense `A^g` then a GEMV, parallel over
    /// points; each point owns one contiguous row of `G^T` `(B, nbf)`, returned
    /// transposed as a `(nbf, B)` view of the same memory (so the block GEMM
    /// sees exactly the layout it always did).
    fn contract_block_cosx_a(
        &self,
        engines: &ThreadSlots<Engine>,
        pts: &[[f64; 3]],
        f: &Array2<f64>,
        acc: &BlockCounters,
    ) -> Result<Array2<f64>, FerricError> {
        let nbf = self.prep.nbasis();
        let (bounds, screen) = self.screen_args();
        let prep = self.prep;
        let mut gt = Array2::<f64>::zeros((pts.len(), nbf));
        let rows = gt.as_slice_mut().expect("freshly allocated standard layout");
        rows.par_chunks_mut(nbf)
            .enumerate()
            .try_for_each(|(g, row)| -> Result<(), FerricError> {
                let t0 = Instant::now();
                let pt = engines.with(|eng| a_matrix_at_point_with(eng, prep, &pts[g], bounds, screen))?;
                let t1 = Instant::now();
                let col = pt.a.dot(&f.column(g));
                row.copy_from_slice(col.as_slice().expect("dot result contiguous"));
                acc.a_ns.fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
                acc.c_ns.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);
                acc.kept.fetch_add(pt.pairs_kept, Ordering::Relaxed);
                acc.total.fetch_add(pt.pairs_total, Ordering::Relaxed);
                Ok(())
            })?;
        Ok(gt.reversed_axes())
    }

    /// md3c1e backend: parallel over fixed sub-batches of
    /// `COSX_SUB_BATCH_POINTS` points; each sub-batch builds its density-driven
    /// pair screen (`BatchScreen`, from its own columns of `F`), sweeps the
    /// surviving shell pairs once and accumulates its own columns of `G` (see
    /// `accumulate_pair`). `pairs_*` are scaled by the sub-batch size so they
    /// stay "summed over points" like the cosx_a path. The screen setup time
    /// is counted as A-build time.
    fn contract_block_md3c1e(
        &self,
        kern: &Md3c1e,
        scratch: &ThreadSlots<Md3c1eScratch>,
        pts: &[[f64; 3]],
        f: &Array2<f64>,
        acc: &BlockCounters,
    ) -> Result<(Array2<f64>, Option<Vec<bool>>), FerricError> {
        let nbf = self.prep.nbasis();
        let nsh = kern.nshells();
        let f_std = f.as_standard_layout();
        let ys: Vec<(Vec<f64>, Vec<bool>)> = pts
            .par_chunks(COSX_SUB_BATCH_POINTS)
            .enumerate()
            .map(|(c, sub)| -> Result<(Vec<f64>, Vec<bool>), FerricError> {
                let c0 = c * COSX_SUB_BATCH_POINTS;
                let n = sub.len();
                let fsub = copy_columns(&f_std.view(), c0, n);
                let mut y = vec![0.0_f64; nbf * n];
                let mut touched = vec![false; nsh];
                let mut c_ns = 0u64;
                let t0 = Instant::now();
                let mut screen = self.batch_screen(kern, sub, &fsub, n);
                let (kept, total) = scratch.with(|scr| {
                    kern.for_each_pair_where(
                        sub,
                        |s1, s2| screen.as_mut().is_none_or(|sc| sc.keep(s1, s2)),
                        scr,
                        |s1, s2, blk| {
                            let t = Instant::now();
                            accumulate_pair(kern, s1, s2, n, blk, &fsub, &mut y);
                            touched[s1] = true;
                            touched[s2] = true;
                            c_ns += t.elapsed().as_nanos() as u64;
                        },
                    )
                })?;
                let geom = screen.as_ref().map_or(total, |sc| sc.geom_kept);
                let (bev, deg) = screen.as_ref().map_or((0, 0), |sc| (sc.bound_evals, sc.degenerate));
                let all_ns = t0.elapsed().as_nanos() as u64;
                acc.a_ns.fetch_add(all_ns.saturating_sub(c_ns), Ordering::Relaxed);
                acc.c_ns.fetch_add(c_ns, Ordering::Relaxed);
                acc.kept.fetch_add(kept * n, Ordering::Relaxed);
                acc.kept_geom.fetch_add(geom * n, Ordering::Relaxed);
                acc.total.fetch_add(total * n, Ordering::Relaxed);
                acc.bound_evals.fetch_add(bev, Ordering::Relaxed);
                acc.degenerate.fetch_add(deg, Ordering::Relaxed);
                Ok((y, touched))
            })
            .collect::<Result<_, _>>()?;
        let mut g = Array2::<f64>::zeros((nbf, pts.len()));
        let mut touched = vec![false; nsh];
        for (c, (y, tch)) in ys.iter().enumerate() {
            let c0 = c * COSX_SUB_BATCH_POINTS;
            let n = y.len() / nbf;
            for (mu, row) in y.chunks_exact(n).enumerate() {
                g.slice_mut(ndarray::s![mu, c0..c0 + n])
                    .as_slice_mut()
                    .expect("row segment of a standard-layout matrix is contiguous")
                    .copy_from_slice(row);
            }
            for (t, &v) in touched.iter_mut().zip(tch) {
                *t |= v;
            }
        }
        Ok((g, Some(touched)))
    }

    /// The density-driven screen for one sub-batch (`None` when
    /// `screen_thresh` is `None`): one [`Region`] and one `fmax` vector per
    /// contiguous group of `CosxConfig::screen_group` points (`0` = one group
    /// covering the whole sub-batch). See `CosxConfig::screen_thresh` and
    /// `CosxConfig::screen_group`.
    fn batch_screen<'b>(&'b self, kern: &Md3c1e, pts: &[[f64; 3]], f: &[f64], n: usize) -> Option<BatchScreen<'b>> {
        let thresh = self.cfg.screen_thresh?;
        let bounds = self.bounds.as_ref()?;
        let nsh = kern.nshells();
        // `0` and anything >= the sub-batch size are the same single group.
        let g = if self.cfg.screen_group == 0 { n } else { self.cfg.screen_group.min(n) }.max(1);
        let mut regions = Vec::with_capacity(n.div_ceil(g));
        let mut fmax = Vec::with_capacity(n.div_ceil(g) * nsh);
        for (q, grp) in pts.chunks(g).enumerate() {
            regions.push(Region::of(grp));
            let g0 = q * g;
            fmax.extend(shell_fmax_range(kern, f, n, g0, g0 + grp.len()));
        }
        Some(BatchScreen {
            bounds,
            thresh,
            regions,
            fmax,
            nsh,
            geom_kept: 0,
            bound_evals: 0,
            degenerate: 0,
        })
    }

    /// Shared driver for `build` / `build_from_occ`. The dense path maps
    /// `X_blk` to `F_blk` as `D X` or `C (C^T X)`; the sparse path always
    /// works from `D` (forming `C C^T` once for `Occ` — canonical MOs are
    /// delocalized, so only `D` carries the row sparsity `Λ` needs).
    fn build_with(&mut self, src: HalfSource<'_>, k: &mut Array2<f64>) -> Result<usize, FerricError> {
        let t_start = Instant::now();
        self.ctx.check_interrupted()?;
        if self.workers.is_none() {
            self.workers = Some(self.make_workers()?);
        }
        let nbf = self.prep.nbasis();
        let need_snum = self.cfg.overlap_fit && self.snum.is_none();
        let mut ktilde = Array2::<f64>::zeros((nbf, nbf));
        let mut snum = need_snum.then(|| Array2::<f64>::zeros((nbf, nbf)));
        let acc = BlockCounters::default();
        let mut t = CosxTimings::default();

        let workers = self.workers.as_ref().expect("workers initialized above");
        match self.cfg.half_transform {
            CosxHalfTransform::Dense => {
                let ct;
                let half: Box<dyn Fn(&Array2<f64>) -> Array2<f64>> = match src {
                    HalfSource::Density(d) => Box::new(move |x| d.dot(x)),
                    HalfSource::Occ(c) => {
                        ct = c.t().to_owned();
                        Box::new(move |x| c.dot(&ct.dot(x)))
                    }
                };
                self.blocks_dense(&half, workers, &acc, &mut t, &mut ktilde, snum.as_mut())?;
            }
            CosxHalfTransform::Sparse { eps_ao, eps_d } => {
                let d_occ;
                let d = match src {
                    HalfSource::Density(d) => d,
                    HalfSource::Occ(c) => {
                        d_occ = c.dot(&c.t());
                        &d_occ
                    }
                };
                self.blocks_sparse(d, eps_ao, eps_d, workers, &acc, &mut t, &mut ktilde, snum.as_mut())?;
            }
        }
        t.blas_s = t.half_s + t.ktilde_s + t.snum_s;

        let t0 = Instant::now();
        if self.cfg.overlap_fit {
            if let Some(s) = snum {
                self.snum = Some(factorize_snum(&s)?);
            }
            let fac = self.snum.as_ref().expect("S_num factor present when fitting");
            let s_ao = self.s_ao.as_ref().expect("overlap present when fitting");
            finalize_fitted(fac, s_ao, &ktilde, k)?;
        } else {
            finalize_plain(&ktilde, k);
        }
        t.fit_s = t0.elapsed().as_secs_f64();

        t.a_build_s = acc.a_ns.load(Ordering::Relaxed) as f64 * 1e-9;
        t.contract_s = acc.c_ns.load(Ordering::Relaxed) as f64 * 1e-9;
        t.pairs_kept = acc.kept.load(Ordering::Relaxed);
        t.pairs_kept_geom = acc.kept_geom.load(Ordering::Relaxed);
        t.pairs_total = acc.total.load(Ordering::Relaxed);
        t.bound_evals = acc.bound_evals.load(Ordering::Relaxed);
        t.screen_degenerate = acc.degenerate.load(Ordering::Relaxed);
        t.total_s = t_start.elapsed().as_secs_f64();
        self.last = t;
        // No shell quartets are computed by this builder; the work counter the
        // SCF reports as `computed_quartets` gets 0 (see `last_timings` for pairs).
        Ok(0)
    }

    /// The DENSE block loop — the pre-sparse builder, operation for operation
    /// (only the timers were split), so it stays byte-identical to it.
    #[allow(clippy::too_many_arguments)]
    fn blocks_dense(
        &self,
        half: &dyn Fn(&Array2<f64>) -> Array2<f64>,
        workers: &Workers,
        acc: &BlockCounters,
        t: &mut CosxTimings,
        ktilde: &mut Array2<f64>,
        mut snum: Option<&mut Array2<f64>>,
    ) -> Result<(), FerricError> {
        for (pts, sw) in self.points.chunks(COSX_BLOCK_POINTS).zip(self.sqrt_w.chunks(COSX_BLOCK_POINTS)) {
            self.ctx.check_interrupted()?;
            let t0 = Instant::now();
            let x = self.eval_x_block(pts, sw)?;
            t.ao_eval_s += t0.elapsed().as_secs_f64();

            let t0 = Instant::now();
            if let Some(s) = snum.as_deref_mut() {
                *s += &x.dot(&x.t());
            }
            t.snum_s += t0.elapsed().as_secs_f64();
            let t0 = Instant::now();
            let f = half(&x);
            t.half_s += t0.elapsed().as_secs_f64();

            let (g, _touched) = self.contract_block(workers, pts, &f, acc)?;

            let t0 = Instant::now();
            *ktilde += &x.dot(&g.t());
            t.ktilde_s += t0.elapsed().as_secs_f64();
            t.n_blocks += 1;
        }
        // Honest counters: the dense path uses every row and column.
        t.active_ao_frac = 1.0;
        t.active_ao_frac_min = 1.0;
        t.active_ao_frac_max = 1.0;
        t.lambda_frac = 1.0;
        t.out_ao_frac = 1.0;
        Ok(())
    }

    /// The SPARSE block loop (`CosxHalfTransform::Sparse`; module doc
    /// "Sparse half transforms"). Per block: `A` from `X`, `Λ` from the
    /// shell-block maxima of `D` (built once here), `F[Λ] = D[Λ,A] X[A]` with
    /// the other rows exactly zero, the kernel's fold as before, `B` = the
    /// shells it touched, `Ktilde[A,B] += X[A] G[B]^T`, `S_num[A,A] += X[A] X[A]^T`.
    /// Gathers preserve each operand's memory orientation and the scatters
    /// are one addition per element, so with every shell active this is the
    /// dense loop's dgemm calls on equal inputs (bitwise, anchored).
    #[allow(clippy::too_many_arguments)]
    fn blocks_sparse(
        &self,
        d: &Array2<f64>,
        eps_ao: f64,
        eps_d: f64,
        workers: &Workers,
        acc: &BlockCounters,
        t: &mut CosxTimings,
        ktilde: &mut Array2<f64>,
        mut snum: Option<&mut Array2<f64>>,
    ) -> Result<(), FerricError> {
        let nbf = self.prep.nbasis();
        let t0 = Instant::now();
        let dmax = shell_block_max(d, &self.shell_off, &self.shell_dim);
        t.gather_s += t0.elapsed().as_secs_f64();
        t.active_ao_frac_min = f64::INFINITY;
        t.active_ao_frac_max = 0.0;
        for (pts, sw) in self.points.chunks(COSX_BLOCK_POINTS).zip(self.sqrt_w.chunks(COSX_BLOCK_POINTS)) {
            self.ctx.check_interrupted()?;
            let t0 = Instant::now();
            let x = self.eval_x_block(pts, sw)?;
            t.ao_eval_s += t0.elapsed().as_secs_f64();

            let t0 = Instant::now();
            let a = active_shells(&x, eps_ao, &self.shell_off, &self.shell_dim);
            let lam = lambda_shells(&dmax, &a.shells, eps_d, &self.shell_off, &self.shell_dim);
            let x_a = gather_rows(&x, &a.aos);
            t.gather_s += t0.elapsed().as_secs_f64();
            let fa = a.aos.len() as f64 / nbf as f64;
            t.n_blocks += 1;
            t.active_ao_frac += fa;
            t.active_ao_frac_min = t.active_ao_frac_min.min(fa);
            t.active_ao_frac_max = t.active_ao_frac_max.max(fa);
            t.lambda_frac += lam.aos.len() as f64 / nbf as f64;
            if a.aos.is_empty() {
                // Every |X| on this block is below eps_ao: it contributes
                // nothing to Ktilde or S_num (never reached at eps_ao = 0).
                continue;
            }

            if let Some(s) = snum.as_deref_mut() {
                let t0 = Instant::now();
                let s_aa = x_a.dot(&x_a.t());
                scatter_add(s, &a.aos, &a.aos, &s_aa);
                t.snum_s += t0.elapsed().as_secs_f64();
            }
            if lam.aos.is_empty() {
                // F is exactly zero on this block: G and its Ktilde term vanish
                // (never reached at eps_d = 0).
                continue;
            }

            let t0 = Instant::now();
            let d_la = gather_block(d, &lam.aos, &a.aos);
            let f_la = d_la.dot(&x_a);
            let mut f = Array2::<f64>::zeros((nbf, pts.len()));
            for (i, &lambda) in lam.aos.iter().enumerate() {
                f.row_mut(lambda).assign(&f_la.row(i));
            }
            t.half_s += t0.elapsed().as_secs_f64();

            let (g, touched) = self.contract_block(workers, pts, &f, acc)?;

            let t0 = Instant::now();
            let b_aos: Vec<usize> = match touched {
                None => (0..nbf).collect(),
                Some(tch) => aos_of_shells(tch.iter().enumerate().filter(|(_, &v)| v).map(|(s, _)| s), &self.shell_off, &self.shell_dim),
            };
            t.out_ao_frac += b_aos.len() as f64 / nbf as f64;
            if !b_aos.is_empty() {
                let g_b = gather_rows_same_layout(&g, &b_aos);
                let prod = x_a.dot(&g_b.t());
                scatter_add(ktilde, &a.aos, &b_aos, &prod);
            }
            t.ktilde_s += t0.elapsed().as_secs_f64();
        }
        let nb = t.n_blocks.max(1) as f64;
        t.active_ao_frac /= nb;
        t.lambda_frac /= nb;
        t.out_ao_frac /= nb;
        if t.n_blocks == 0 {
            t.active_ao_frac_min = 0.0;
        }
        Ok(())
    }

    /// Per-worker state for the configured backend.
    fn make_workers(&self) -> Result<Workers, FerricError> {
        Ok(match self.cfg.backend {
            CosxBackend::CosxA => {
                Workers::CosxA(ThreadSlots::new(|| Engine::new_1e(ffi::OP_NUCLEAR, self.prep, 1e-14))?)
            }
            CosxBackend::Md3c1e => {
                let kern = self.kernel.as_ref().expect("md3c1e kernel built in new() for this backend");
                Workers::Md3c1e(ThreadSlots::new(|| Ok(kern.scratch()))?)
            }
        })
    }
}

/// What a `build_with` call transforms: the density itself, or the occupied
/// coefficients it factorizes (`D = C C^T`).
enum HalfSource<'h> {
    Density(&'h Array2<f64>),
    Occ(&'h Array2<f64>),
}

/// A shell-granular AO subset: the shells (ascending) and the AO indices
/// they cover (ascending, concatenated shell ranges).
#[derive(Debug, Clone, PartialEq, Eq)]
struct ShellMask {
    shells: Vec<usize>,
    aos: Vec<usize>,
}

/// AO indices covered by `shells` (each shell contributes its full range).
fn aos_of_shells(shells: impl Iterator<Item = usize>, off: &[usize], dim: &[usize]) -> Vec<usize> {
    shells.flat_map(|s| off[s]..off[s] + dim[s]).collect()
}

/// `A` for one block: shells with `max_{mu in s, g} |x[mu, g]| >= eps`
/// (`>=`, so `eps = 0` keeps every shell — the vacuous mask).
fn active_shells(x: &Array2<f64>, eps: f64, off: &[usize], dim: &[usize]) -> ShellMask {
    let shells: Vec<usize> = (0..off.len())
        .filter(|&s| {
            let m = x.slice(ndarray::s![off[s]..off[s] + dim[s], ..]).iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
            m >= eps
        })
        .collect();
    let aos = aos_of_shells(shells.iter().copied(), off, dim);
    ShellMask { shells, aos }
}

/// `dmax[l * nsh + s] = max |d[mu, nu]|` over `mu in shell l, nu in shell s`.
fn shell_block_max(d: &Array2<f64>, off: &[usize], dim: &[usize]) -> Vec<f64> {
    let nsh = off.len();
    let mut out = vec![0.0_f64; nsh * nsh];
    for l in 0..nsh {
        for s in 0..nsh {
            out[l * nsh + s] = d
                .slice(ndarray::s![off[l]..off[l] + dim[l], off[s]..off[s] + dim[s]])
                .iter()
                .fold(0.0_f64, |m, &v| m.max(v.abs()));
        }
    }
    out
}

/// `Λ` for one block: shells `l` with `max_{s in A} dmax[l][s] >= eps`
/// (`>=`: `eps = 0` keeps every shell whenever `A` is non-empty).
fn lambda_shells(dmax: &[f64], a_shells: &[usize], eps: f64, off: &[usize], dim: &[usize]) -> ShellMask {
    let nsh = off.len();
    let shells: Vec<usize> =
        (0..nsh).filter(|&l| a_shells.iter().any(|&s| dmax[l * nsh + s] >= eps)).collect();
    let aos = aos_of_shells(shells.iter().copied(), off, dim);
    ShellMask { shells, aos }
}

/// `m[rows, ..]` as a fresh standard-layout `(rows.len(), ncols)` matrix.
fn gather_rows(m: &Array2<f64>, rows: &[usize]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((rows.len(), m.ncols()));
    for (i, &r) in rows.iter().enumerate() {
        out.row_mut(i).assign(&m.row(r));
    }
    out
}

/// `m[rows, ..]` in the SAME memory orientation as `m`: standard layout when
/// `m` is standard; otherwise (a transposed view of standard memory, which is
/// how the cosx_a backend hands `G` over) the transpose of a standard
/// `(ncols, rows.len())` buffer. This keeps the stride pattern the block GEMM
/// sees identical to the dense path's, so the vacuous mask is bitwise.
fn gather_rows_same_layout(m: &Array2<f64>, rows: &[usize]) -> Array2<f64> {
    if m.is_standard_layout() || !m.t().is_standard_layout() {
        gather_rows(m, rows)
    } else {
        let mut out = Array2::<f64>::zeros((m.ncols(), rows.len()));
        for (i, &r) in rows.iter().enumerate() {
            out.column_mut(i).assign(&m.row(r));
        }
        out.reversed_axes()
    }
}

/// `m[rows, cols]` as a fresh standard-layout matrix.
fn gather_block(m: &Array2<f64>, rows: &[usize], cols: &[usize]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((rows.len(), cols.len()));
    for (i, &r) in rows.iter().enumerate() {
        let src = m.row(r);
        for (j, &c) in cols.iter().enumerate() {
            out[(i, j)] = src[c];
        }
    }
    out
}

/// `k[rows[i], cols[j]] += v[i, j]` — one addition per element, exactly the
/// elementwise `k += v` of the dense path when the maps are the identity.
fn scatter_add(k: &mut Array2<f64>, rows: &[usize], cols: &[usize], v: &Array2<f64>) {
    debug_assert_eq!(v.dim(), (rows.len(), cols.len()));
    for (i, &r) in rows.iter().enumerate() {
        for (j, &c) in cols.iter().enumerate() {
            k[(r, c)] += v[(i, j)];
        }
    }
}

/// `F[.., c0..c0+n]` as a contiguous `(nbf, n)` row-major buffer.
fn copy_columns(f: &ndarray::ArrayView2<f64>, c0: usize, n: usize) -> Vec<f64> {
    let nbf = f.nrows();
    let mut out = vec![0.0_f64; nbf * n];
    for (mu, dst) in out.chunks_exact_mut(n).enumerate() {
        for (d, &v) in dst.iter_mut().zip(f.slice(ndarray::s![mu, c0..c0 + n]).iter()) {
            *d = v;
        }
    }
    out
}

/// `y += a * b` elementwise over one sub-batch row (length `n`).
#[inline(always)]
fn axpy_rows(a: &[f64], b: &[f64], y: &mut [f64]) {
    for ((y, &a), &b) in y.iter_mut().zip(a).zip(b) {
        *y += a * b;
    }
}

/// Fold one kernel block into `Y = G` for the sub-batch (`n` points, all
/// buffers `(nbf, n)` row-major): `Y[o1+i] += blk[i][j] F[o2+j]` and, for
/// `s1 != s2`, the mirror `Y[o2+j] += blk[i][j] F[o1+i]` (the kernel emits
/// only `s1 >= s2`; the diagonal block is already the full `nf x nf` square).
/// The kernel's sign is already `+1/|r-r_g|` — nothing is negated here.
#[inline]
fn accumulate_pair(kern: &Md3c1e, s1: usize, s2: usize, n: usize, blk: &[f64], f: &[f64], y: &mut [f64]) {
    let (o1, o2) = (kern.shell_offset(s1), kern.shell_offset(s2));
    let (nf1, nf2) = (kern.shell_dim(s1), kern.shell_dim(s2));
    for i in 0..nf1 {
        let r1 = o1 + i;
        for j in 0..nf2 {
            let r2 = o2 + j;
            let b = &blk[(i * nf2 + j) * n..(i * nf2 + j + 1) * n];
            axpy_rows(b, &f[r2 * n..(r2 + 1) * n], &mut y[r1 * n..(r1 + 1) * n]);
            if s1 != s2 {
                axpy_rows(b, &f[r1 * n..(r1 + 1) * n], &mut y[r2 * n..(r2 + 1) * n]);
            }
        }
    }
}

#[derive(Default)]
struct BlockCounters {
    a_ns: AtomicU64,
    c_ns: AtomicU64,
    kept: AtomicUsize,
    kept_geom: AtomicUsize,
    total: AtomicUsize,
    bound_evals: AtomicUsize,
    degenerate: AtomicUsize,
}

/// One screening region: a contiguous group of the sub-batch's points,
/// described by BOTH its centroid ball and its axis-aligned box.
///
/// Both enclose the group's points, so both bounds are valid and so is their
/// `min` — which is what [`Region::bound`] returns. Neither dominates the
/// other: for a Lebedev arc the box's corners stick out of the centroid ball
/// and the ball's caps stick out of the box (measured, `cosx_screen_box_anchors.rs`:
/// on butane/def2-SVP the box is tighter on 8.8% of queries and the sphere on
/// 14.3%), so taking the `min` is strictly better than either alone at the
/// cost of one extra clamped-difference evaluation.
#[derive(Clone, Copy)]
struct Region {
    centre: [f64; 3],
    radius: f64,
    lo: [f64; 3],
    hi: [f64; 3],
}

impl Region {
    /// Centroid ball + AABB of `pts`. Empty input gives a degenerate region at
    /// the origin, which is never reached (groups are non-empty).
    fn of(pts: &[[f64; 3]]) -> Self {
        let n = pts.len().max(1) as f64;
        let mut c = [0.0_f64; 3];
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in pts {
            for d in 0..3 {
                c[d] += p[d];
                lo[d] = lo[d].min(p[d]);
                hi[d] = hi[d].max(p[d]);
            }
        }
        for v in &mut c {
            *v /= n;
        }
        if pts.is_empty() {
            (lo, hi) = ([0.0; 3], [0.0; 3]);
        }
        let r2 = pts
            .iter()
            .map(|p| (0..3).map(|d| (p[d] - c[d]) * (p[d] - c[d])).sum::<f64>())
            .fold(0.0_f64, f64::max);
        Self { centre: c, radius: r2.sqrt(), lo, hi }
    }

    /// The tightest valid geometric bound for this region: the smaller of the
    /// ball and the box bounds, both of which hold over every point in it.
    #[inline]
    fn bound(&self, bounds: &PairBounds, s1: usize, s2: usize) -> f64 {
        let sph = bounds.coarse_estimate_sphere(s1, s2, &self.centre, self.radius);
        let bx = bounds.coarse_estimate_box(s1, s2, &self.lo, &self.hi);
        sph.min(bx)
    }
}

/// Density-driven pair screen for ONE sub-batch (see `CosxConfig::screen_thresh`).
///
/// The sub-batch's points are partitioned into contiguous GROUPS
/// (`CosxConfig::screen_group`) and each group carries its own [`Region`] and
/// its own `fmax`. A pair is evaluated for the whole sub-batch iff SOME group
/// keeps it, so the kernel's 256-point blocking is untouched and correctness
/// is preserved by construction: every group's bound is applied only to points
/// in that group, and the union is what survives.
struct BatchScreen<'b> {
    bounds: &'b PairBounds,
    thresh: f64,
    /// One region per group, in point order.
    regions: Vec<Region>,
    /// `fmax[grp][s] = max_{mu in s, g in group} |F_{mu,g}|`, flattened
    /// `grp * nsh + s`.
    fmax: Vec<f64>,
    nsh: usize,
    /// Pairs some SCANNED group's geometry-only bound reached (diagnostic).
    /// With more than one group the scan stops at the first group that keeps
    /// the pair, so this is a LOWER bound on "some group's geometry-only bound
    /// reached `thresh`" — exact at one group, which is the configuration the
    /// existing anchors compare against.
    geom_kept: usize,
    /// Region-bound evaluations performed (cost counter: grouping by `G`
    /// multiplies this by up to `G`, early-outs aside).
    bound_evals: usize,
    /// Of those, how many returned the distance-free `R = 0` value — the
    /// measurement the sub-batched screen exists to move.
    degenerate: usize,
}

impl BatchScreen<'_> {
    /// Keep `(s1, s2)` iff SOME group `q` has
    /// `min(ball, box)_q(s1, s2) * max(fmax_q[s1], fmax_q[s2]) >= thresh`.
    ///
    /// With one group covering the whole sub-batch this is exactly the old
    /// single-region rule (bar the `min` with the box, which can only tighten
    /// it), which is the trivial limit `screen_group = 0` reproduces bitwise.
    /// For `thresh <= 0` the first group already keeps everything (every
    /// factor is `>= 0`). The `max` over both shells covers both orderings of
    /// the mirror fold.
    #[inline]
    fn keep(&mut self, s1: usize, s2: usize) -> bool {
        let mut kept = false;
        let mut geom = false;
        for (q, reg) in self.regions.iter().enumerate() {
            let f = self.fmax[q * self.nsh + s1].max(self.fmax[q * self.nsh + s2]);
            let est = reg.bound(self.bounds, s1, s2);
            self.bound_evals += 1;
            if est >= self.bounds.max_estimate(s1, s2) {
                self.degenerate += 1;
            }
            if est >= self.thresh {
                geom = true;
            }
            if est * f >= self.thresh {
                kept = true;
                break;
            }
        }
        if geom {
            self.geom_kept += 1;
        }
        kept
    }
}

/// `fmax[s] = max_{mu in shell s, g in [g0, g1)} |f[mu * n + g]|` for `f` in
/// the `(nbf, n)` row-major sub-batch layout. `g0 = 0, g1 = n` is the whole
/// sub-batch (the unsplit screen); a narrower range is one screening group.
fn shell_fmax_range(kern: &Md3c1e, f: &[f64], n: usize, g0: usize, g1: usize) -> Vec<f64> {
    (0..kern.nshells())
        .map(|s| {
            let (o, nf) = (kern.shell_offset(s), kern.shell_dim(s));
            (o..o + nf)
                .flat_map(|mu| f[mu * n + g0..mu * n + g1].iter())
                .fold(0.0_f64, |m, &v| m.max(v.abs()))
        })
        .collect()
}

/// Fail fast if one block's scratch does not fit the budget. The per-thread
/// footprint is one dense `A^g` (nbf^2) for cosx_a, or `Y` + `F` sub-batch
/// planes (2 nbf x `COSX_SUB_BATCH_POINTS`) for md3c1e. The sparse half
/// transform adds, at its dense worst case, two more planes (`X[A]`, `G[B]`)
/// and two squares (`D[Λ,A]`, `D_occ`); its gathered products are bounded by
/// the planes already counted.
fn check_budget(nbf: usize, backend: CosxBackend, half: CosxHalfTransform, mem_budget: usize) -> Result<(), FerricError> {
    let budget = if mem_budget == 0 { ferric_core::memory::resolve_budget_bytes(None) } else { mem_budget };
    let threads = rayon::current_num_threads().max(1) + 1;
    let (n_planes, n_squares) = match half {
        CosxHalfTransform::Dense => (3usize, 5usize),
        CosxHalfTransform::Sparse { .. } => (5, 7),
    };
    let planes = n_planes.saturating_mul(COSX_BLOCK_POINTS).saturating_mul(nbf).saturating_mul(8);
    let per_thread = match backend {
        CosxBackend::CosxA => nbf.saturating_mul(nbf).saturating_mul(8),
        CosxBackend::Md3c1e => 2usize.saturating_mul(nbf).saturating_mul(COSX_SUB_BATCH_POINTS).saturating_mul(8),
    };
    // Ktilde + S_num + S + L + L^T (+ D[Λ,A] + D_occ when sparse)
    let squares = n_squares.saturating_mul(nbf).saturating_mul(nbf).saturating_mul(8);
    let needed = planes.saturating_add(squares).saturating_add(threads.saturating_mul(per_thread));
    if needed > budget {
        return Err(FerricError::General(format!(
            "CosxK: one grid block needs {:.2} GB (nbf={nbf}, {COSX_BLOCK_POINTS} pts x {n_planes} planes + {threads} per-thread {} buffers) \
             but the memory budget is {:.2} GB — raise [memory] budget_gb / FERRIC_MEM_BUDGET_GB or use fewer threads",
            needed as f64 / 1e9,
            backend.as_str(),
            budget as f64 / 1e9
        )));
    }
    Ok(())
}

/// `K = 0.5 (Ktilde + Ktilde^T)`.
fn finalize_plain(ktilde: &Array2<f64>, k: &mut Array2<f64>) {
    k.assign(ktilde);
    *k += &ktilde.t();
    *k *= 0.5;
}

/// Cholesky-factorize `S_num`; a non-positive-definite Gram matrix is a hard
/// error (never a pseudo-inverse fallback, which would hide a rank-deficient grid).
fn factorize_snum(snum: &Array2<f64>) -> Result<SnumFactor, FerricError> {
    let l = snum.cholesky(UPLO::Lower).map_err(|e| {
        FerricError::General(format!(
            "CosxK overlap fit: S_num = X X^T is not positive definite on this grid ({e}); \
             use a finer cosx grid or disable the overlap fit"
        ))
    })?;
    let lt = l.t().to_owned();
    Ok(SnumFactor { l, lt })
}

/// `K = 0.5 (Q Ktilde + (Q Ktilde)^T)` with `Q Ktilde = S (S_num^{-1} Ktilde)`
/// via two triangular solves against the cached factor.
fn finalize_fitted(
    fac: &SnumFactor,
    s_ao: &Array2<f64>,
    ktilde: &Array2<f64>,
    k: &mut Array2<f64>,
) -> Result<(), FerricError> {
    let y = fac
        .l
        .solve_triangular(UPLO::Lower, Diag::NonUnit, ktilde)
        .map_err(|e| FerricError::General(format!("CosxK overlap fit: forward solve failed: {e}")))?;
    let z = fac
        .lt
        .solve_triangular(UPLO::Upper, Diag::NonUnit, &y)
        .map_err(|e| FerricError::General(format!("CosxK overlap fit: back solve failed: {e}")))?;
    let qk = s_ao.dot(&z);
    k.assign(&qk);
    *k += &qk.t();
    *k *= 0.5;
    Ok(())
}

impl<'a> KBuilder for CosxK<'a> {
    /// K from a raw density — the mandatory path (iteration 1, Fermi smearing
    /// and fractional occupations supply no `C_occ`).
    fn build(&mut self, d: &Array2<f64>, k: &mut Array2<f64>) -> Result<usize, FerricError> {
        self.build_with(HalfSource::Density(d), k)
    }

    /// K for `D = C_occ C_occ^T`. Dense path: the cheaper half transform
    /// `F = C (C^T X)`, identical to `build(C C^T)` to round-off (anchored).
    /// Sparse path: forms `C C^T` once and takes the density path (bitwise
    /// `build(C C^T)`, anchored) — the row mask needs the decaying `D`, not
    /// the delocalized `C`.
    fn build_from_occ(&mut self, c_occ: &Array2<f64>, k: &mut Array2<f64>) -> Result<usize, FerricError> {
        self.build_with(HalfSource::Occ(c_occ), k)
    }

    /// No density-dependent state (no pair lists): nothing to update.
    fn update_density(&mut self, _d: &Array2<f64>) {}

    /// No density-dependent state to drop. The grid, pair bounds and the
    /// `S_num` factor are geometry-only and are kept.
    fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_grid_is_the_measured_operating_point() {
        let c = CosxConfig::default();
        assert_eq!((c.grid.n_radial, c.grid.n_angular), (50, 110));
        assert!(c.overlap_fit);
        assert_eq!(c.screen_thresh, Some(COSX_DEFAULT_SCREEN_THRESH));
        assert_eq!(COSX_DEFAULT_SCREEN_THRESH, 1e-7);
        assert_eq!(c.backend, CosxBackend::Md3c1e);
        assert_eq!(c.half_transform, CosxHalfTransform::Sparse { eps_ao: 1e-10, eps_d: 1e-10 });
    }

    #[test]
    fn half_transform_parser_is_strict_and_round_trips() {
        assert_eq!(CosxHalfTransform::parse_config_str("dense").unwrap(), CosxHalfTransform::Dense);
        assert_eq!(CosxHalfTransform::parse_config_str("sparse").unwrap(), CosxHalfTransform::SPARSE_DEFAULT);
        assert!(CosxHalfTransform::parse_config_str("Sparse").is_err());
        assert!(CosxHalfTransform::parse_config_str("").is_err());
        assert_eq!(CosxHalfTransform::Dense.as_str(), "dense");
        assert_eq!(CosxHalfTransform::Sparse { eps_ao: 1.0, eps_d: 1.0 }.as_str(), "sparse");
    }

    /// Masks on a toy layout: three shells of dims 1, 3, 1 (AOs 0 | 1..4 | 4).
    #[test]
    fn masks_keep_everything_at_zero_and_are_shell_granular() {
        let off = [0usize, 1, 4];
        let dim = [1usize, 3, 1];
        // x rows: shell 0 = 0.5; shell 1 = (0, 1e-12, 0); shell 2 = exactly 0.
        let mut x = Array2::<f64>::zeros((5, 2));
        x[(0, 1)] = 0.5;
        x[(2, 0)] = 1e-12;
        let all = active_shells(&x, 0.0, &off, &dim);
        assert_eq!(all, ShellMask { shells: vec![0, 1, 2], aos: vec![0, 1, 2, 3, 4] }, "eps = 0 must keep exact zeros");
        let a = active_shells(&x, 1e-10, &off, &dim);
        assert_eq!(a, ShellMask { shells: vec![0], aos: vec![0] });
        let a = active_shells(&x, 1e-12, &off, &dim);
        assert_eq!(a.shells, vec![0, 1], ">= at the threshold keeps the shell (whole shell, not one AO)");
        assert_eq!(a.aos, vec![0, 1, 2, 3]);

        // D: shell block (2,0) large, (1,0) tiny, everything else 0.
        let mut d = Array2::<f64>::zeros((5, 5));
        d[(4, 0)] = 0.3;
        d[(2, 0)] = 1e-11;
        let dmax = shell_block_max(&d, &off, &dim);
        assert_eq!(dmax[2 * 3], 0.3);
        assert_eq!(dmax[3], 1e-11);
        let lam = lambda_shells(&dmax, &[0], 1e-10, &off, &dim);
        assert_eq!(lam.shells, vec![2]);
        let lam = lambda_shells(&dmax, &[0], 0.0, &off, &dim);
        assert_eq!(lam.shells, vec![0, 1, 2], "eps_d = 0 keeps every row");
        let lam = lambda_shells(&dmax, &[], 0.0, &off, &dim);
        assert!(lam.shells.is_empty(), "empty A gives empty Λ");
    }

    #[test]
    fn gathers_preserve_orientation_and_scatter_is_one_add() {
        let m = ndarray::arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let g = gather_rows(&m, &[2, 0]);
        assert_eq!(g, ndarray::arr2(&[[7.0, 8.0, 9.0], [1.0, 2.0, 3.0]]));
        assert!(g.is_standard_layout());
        let gs = gather_rows_same_layout(&m, &[2, 0]);
        assert_eq!(gs, g);
        assert!(gs.is_standard_layout());
        // A transposed view (the cosx_a G layout): the gather must come back transposed too.
        let mt = m.clone().reversed_axes();
        assert!(!mt.is_standard_layout());
        let gt = gather_rows_same_layout(&mt, &[1, 2]);
        assert_eq!(gt, ndarray::arr2(&[[2.0, 5.0, 8.0], [3.0, 6.0, 9.0]]));
        assert!(!gt.is_standard_layout() && gt.t().is_standard_layout());
        assert_eq!(gather_block(&m, &[0, 2], &[1]), ndarray::arr2(&[[2.0], [8.0]]));
        let mut k = Array2::<f64>::ones((3, 3));
        scatter_add(&mut k, &[2, 0], &[1], &ndarray::arr2(&[[10.0], [20.0]]));
        assert_eq!(k, ndarray::arr2(&[[1.0, 21.0, 1.0], [1.0, 1.0, 1.0], [1.0, 11.0, 1.0]]));
    }

    #[test]
    fn batch_screen_keep_covers_both_orderings_and_is_vacuous_at_zero() {
        // Two-shell toy: bounds from a real (tiny) basis; fmax asymmetric.
        let mol = ferric_core::mol::Molecule::parse_xyz("2\nh2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let bs = ferric_core::basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let bounds = PairBounds::build(&prep).unwrap();
        // A single-group screen over a small ball, built from the points whose
        // Region reproduces the old (centre, radius) pair exactly.
        let pts = [[0.0, 0.0, 0.2], [0.0, 0.0, 1.2]];
        let region = Region::of(&pts);
        assert_eq!(region.centre, [0.0, 0.0, 0.7]);
        assert!((region.radius - 0.5).abs() < 1e-15);
        let est = region.bound(&bounds, 1, 0);
        assert!(est > 0.0);
        let mk = |fmax: Vec<f64>, thresh: f64| BatchScreen {
            bounds: &bounds,
            thresh,
            regions: vec![region],
            fmax,
            nsh: 2,
            geom_kept: 0,
            bound_evals: 0,
            degenerate: 0,
        };
        // Only shell 1 has a large F: pair (1,0) must still be kept (mirror
        // G_0 += A F_1), so the keep rule must use max(fmax[1], fmax[0]).
        let t = 0.5 * est;
        assert!(mk(vec![0.0, 1.0], t).keep(1, 0));
        assert!(mk(vec![1.0, 0.0], t).keep(1, 0));
        assert!(!mk(vec![0.0, 0.0], t).keep(1, 0), "zero F on both shells must drop the pair");
        assert!(!mk(vec![1e-3, 1e-3], est).keep(1, 0), "est * 1e-3 < est must drop");
        // Trivial limit: threshold 0 keeps everything, even with F == 0.
        assert!(mk(vec![0.0, 0.0], 0.0).keep(1, 0));
        // The geometry-only counter ignores F.
        let mut sc = mk(vec![0.0, 0.0], t);
        sc.keep(1, 0);
        assert_eq!(sc.geom_kept, 1);
    }

    #[test]
    fn shell_fmax_range_and_region_toy() {
        let mol = ferric_core::mol::Molecule::parse_xyz("2\nh2\nH 0 0 0\nH 0 0 0.74\n", 0, 1).unwrap();
        let bs = ferric_core::basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let kern = Md3c1e::new(&prep).unwrap();
        // nbf = 2, n = 3: row 0 = [1, -4, 2], row 1 = [0.5, 0, -0.25]
        let f = vec![1.0, -4.0, 2.0, 0.5, 0.0, -0.25];
        // Whole sub-batch (the unsplit screen).
        assert_eq!(shell_fmax_range(&kern, &f, 3, 0, 3), vec![4.0, 0.5]);
        // Groups: [0,2) sees the -4 and the 0.5; [2,3) sees the 2 and the -0.25.
        assert_eq!(shell_fmax_range(&kern, &f, 3, 0, 2), vec![4.0, 0.5]);
        assert_eq!(shell_fmax_range(&kern, &f, 3, 2, 3), vec![2.0, 0.25]);
        // Region: centroid ball as before, plus the AABB of the same points.
        let r = Region::of(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0]]);
        assert_eq!(r.centre, [1.0, 0.0, 0.0]);
        assert!((r.radius - 1.0).abs() < 1e-15);
        assert_eq!(r.lo, [0.0, -1.0, 0.0]);
        assert_eq!(r.hi, [2.0, 1.0, 0.0]);
    }

    #[test]
    fn backend_parser_is_strict_and_round_trips() {
        for b in [CosxBackend::Md3c1e, CosxBackend::CosxA] {
            assert_eq!(CosxBackend::parse_config_str(b.as_str()).unwrap(), b);
        }
        assert!(CosxBackend::parse_config_str("libint").is_err());
        assert!(CosxBackend::parse_config_str("MD3C1E").is_err(), "case-sensitive: no silent coercion");
        assert!(CosxBackend::parse_config_str("").is_err());
    }

    /// The fold must reproduce the dense `A F` product on a toy layout: two
    /// shells (dims 1 and 2), three points, a hand-built symmetric A.
    #[test]
    fn accumulate_pair_matches_dense_product_on_toy_layout() {
        // Use a real kernel only for offsets/dims: water/STO-3G has 5 shells;
        // exercise shells 0 (s, dim 1) and 2 (p, dim 3).
        let mol = ferric_core::mol::Molecule::parse_xyz(
            "3\nw\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n", 0, 1,
        )
        .unwrap();
        let bs = ferric_core::basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let kern = Md3c1e::new(&prep).unwrap();
        let nbf = kern.nbasis();
        let n = 3usize;
        let (s1, s2) = (2usize, 0usize);
        let (nf1, nf2) = (kern.shell_dim(s1), kern.shell_dim(s2));
        let (o1, o2) = (kern.shell_offset(s1), kern.shell_offset(s2));
        // blk[i][j][g] = 1 + i + 10 j + 100 g ; F[mu][g] = mu + 0.5 g
        let blk: Vec<f64> = (0..nf1 * nf2 * n)
            .map(|k| {
                let (ij, g) = (k / n, k % n);
                1.0 + (ij / nf2) as f64 + 10.0 * (ij % nf2) as f64 + 100.0 * g as f64
            })
            .collect();
        let f: Vec<f64> = (0..nbf * n).map(|k| (k / n) as f64 + 0.5 * (k % n) as f64).collect();
        let mut y = vec![0.0; nbf * n];
        accumulate_pair(&kern, s1, s2, n, &blk, &f, &mut y);
        // Dense reference: A has the block at (o1.., o2..) and its transpose.
        let mut a = Array2::<f64>::zeros((nbf, nbf));
        for g in 0..n {
            for i in 0..nf1 {
                for j in 0..nf2 {
                    let v = blk[(i * nf2 + j) * n + g];
                    a[(o1 + i, o2 + j)] = v;
                    a[(o2 + j, o1 + i)] = v;
                }
            }
            for mu in 0..nbf {
                let want: f64 = (0..nbf).map(|lam| a[(mu, lam)] * f[lam * n + g]).sum();
                assert!((y[mu * n + g] - want).abs() < 1e-12, "mu={mu} g={g}: {} vs {want}", y[mu * n + g]);
            }
        }
    }

    #[test]
    fn finalize_plain_symmetrizes() {
        let kt = ndarray::arr2(&[[1.0, 2.0], [4.0, 3.0]]);
        let mut k = Array2::zeros((2, 2));
        finalize_plain(&kt, &mut k);
        assert_eq!(k, ndarray::arr2(&[[1.0, 3.0], [3.0, 3.0]]));
    }

    #[test]
    fn grid_validation_rejects_untabulated_orders_and_accepts_tabulated() {
        let ok = AtomicGridConfig { n_radial: 50, n_angular: 110, ..Default::default() };
        assert!(validate_grid(&ok).is_ok());
        let bad_ang = AtomicGridConfig { n_radial: 50, n_angular: 194, ..Default::default() };
        assert!(validate_grid(&bad_ang).is_err(), "194 is not tabulated and must be refused, not panic");
        let bad_rad = AtomicGridConfig { n_radial: 0, n_angular: 110, ..Default::default() };
        assert!(validate_grid(&bad_rad).is_err());
    }

    #[test]
    fn budget_check_errors_when_too_small_and_passes_when_ample() {
        for b in [CosxBackend::Md3c1e, CosxBackend::CosxA] {
            for h in [CosxHalfTransform::Dense, CosxHalfTransform::SPARSE_DEFAULT] {
                assert!(check_budget(100, b, h, 1).is_err());
                assert!(check_budget(100, b, h, usize::MAX).is_ok());
            }
        }
    }
}
