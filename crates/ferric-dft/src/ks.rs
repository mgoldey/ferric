//! `KsXc`: caches the molecular grid + AO + ∇AO + (optional) NLC grid + AO,
//! and implements `XcContribution` for use in the closed-shell KS SCF.
//!
//! When the full χ/∇χ cache for the main grid would exceed the resolved
//! memory budget, `KsXc`/`KsXcUks` fall back to a **batched** mode instead of
//! failing: the main grid is walked in point-batches, each batch's AO values
//! evaluated on demand (never resident all at once), and the V_xc/E_xc
//! contribution accumulated batch-by-batch. See `GridCache` and
//! `resolve_batch_size` below.

use std::sync::Mutex;

use ndarray::{Array1, Array2, Array3};
use thiserror::Error;

use ferric_core::basis::BasisSet;
use ferric_core::mol::Molecule;

use crate::ao_grid::{
    collect_shells, eval_basis_and_grad_on_points, eval_basis_and_grad_on_points_unchecked,
    nbasis, AoGridKind, GtoEvalError,
};
use crate::density_on_grid::{
    eval_density_closed, eval_density_uks, eval_tau_closed, eval_tau_uks, DensityGrid,
};
use crate::libxc::FunctionalFamily;
use crate::grid::{build_atomic_grid, AtomicGridConfig, GridPoint};
use crate::libxc::{xc_def_from_name, xc_def_from_name_nspin, LibxcError, XcDef};
use crate::vv10::add_vv10_scratch;
use crate::vxc::{semilocal_vxc_closed_scratch, semilocal_vxc_polarized_scratch, VxcScratch};
use crate::xc_trait::{KMix, UksXcContribution, XcContribution};

#[derive(Error, Debug)]
pub enum KsXcError {
    #[error("AO evaluation failed: {0:?}")]
    Eval(GtoEvalError),
    #[error("libxc resolver failed: {0}")]
    Libxc(LibxcError),
    #[error(
        "DFT grid AO cache needs {needed_gb:.2} GB (nbf={nbf}, npts={npts}{vv10}) \
         but the budget is {budget_gb:.2} GB — raise [memory] budget_gb / \
         FERRIC_MEM_BUDGET_GB, use a smaller grid, or a smaller basis"
    )]
    OverBudget {
        needed_gb: f64,
        budget_gb: f64,
        nbf: usize,
        npts: usize,
        vv10: &'static str,
    },
}

impl From<GtoEvalError> for KsXcError { fn from(e: GtoEvalError) -> Self { Self::Eval(e) } }
impl From<LibxcError>  for KsXcError { fn from(e: LibxcError)  -> Self { Self::Libxc(e) } }

/// Number of resident `(nbf, batch_pts)` `f64` planes a batched semilocal
/// V_xc pass keeps alive at once: `chi` + `dchi` (4, same as
/// `AoGridKind::ValueAndGrad`) + 1 more for the `VxcScratch` GEMM operand
/// buffer (`semilocal_vxc_closed_scratch` reuses one `(nbf, npts)` buffer
/// across its LDA/GGA/τ terms, but it is sized to the *batch*, not the full
/// grid, once batching is active).
const BATCH_PLANES: usize = AoGridKind::ValueAndGrad.planes() + 1;

/// Byte cost of the full-cache χ + ∇χ tensors (the same formula
/// `check_grid_budget` has always used), used both to decide whether to fall
/// back to batching and to size the batch itself.
fn full_cache_bytes(nbf: usize, npts: usize) -> usize {
    AoGridKind::ValueAndGrad
        .planes()
        .saturating_mul(nbf)
        .saturating_mul(npts)
        .saturating_mul(8)
}

/// Fail fast if the resident χ + ∇χ cache would exceed the memory budget.
///
/// The cache is `chi (nbf·npts·8)` + `dchi (3·nbf·npts·8)` =
/// `AoGridKind::ValueAndGrad.planes() (4) · nbf·npts·8` — the same per-plane
/// formula `ao_grid::check_ao_grid_budget` uses internally for its own callers
/// (this function predates that shared helper and is kept as its own error
/// variant for the richer `OverBudget` message fields, but shares the byte
/// count so the two never silently diverge). With VV10 a second (smaller) NLC
/// grid's cache is resident too, so we bound by `2×` when `has_vv10`. This
/// catches the 50-atom/aTZ case (~30 GB, doubling to ~60 GB with VV10) before
/// the allocation aborts the process.
///
/// `budget` is resolved ONCE by the caller (`KsXc::new`/`KsXcUks::new`, via
/// [`ferric_core::memory::resolve_budget_bytes`]) and reused for both the
/// Full-vs-Batched decision here AND `resolve_batch_size`'s sizing — NOT
/// re-resolved per call. The auto-detect budget is a *live* 0.8× MemAvailable
/// reading that shrinks as the SCF allocates, so resolving it more than once
/// per `KsXc`/`KsXcUks` construction risks the decision and the sizing being
/// made against two different numbers.
///
/// Returns `Ok(true)` when the full cache fits (use it as-is), `Ok(false)`
/// when it does not fit but the NLC (VV10) grid alone still fits (fall back
/// to batching the main grid only), and `Err` when even the always-resident
/// VV10 NLC cache alone cannot fit (nothing left to fall back to — batching
/// only ever shrinks the *main*-grid cache, so an over-budget NLC grid by
/// itself is still a hard failure).
fn check_grid_budget(nbf: usize, npts: usize, has_vv10: bool, budget: usize) -> Result<bool, KsXcError> {
    let base = full_cache_bytes(nbf, npts);
    let needed = if has_vv10 { base.saturating_mul(2) } else { base };
    if needed <= budget {
        return Ok(true);
    }
    // Full cache (main + NLC, if any) doesn't fit. If there's no VV10, the
    // main grid alone is the whole ask, and batching can always shrink it
    // arbitrarily far (down to 1 point at a time) — so this is never a hard
    // failure once we can batch; return `Ok(false)` (batch it).
    //
    // If VV10 *is* present, the NLC grid's own cache (`base`, since `has_vv10`
    // doubles it) is NOT batchable (VV10 is an O(npts²) pair sum needing the
    // whole NLC grid resident at once — see `vv10.rs`), so it alone must fit
    // in the budget or there is truly nothing we can do.
    if has_vv10 && base > budget {
        return Err(KsXcError::OverBudget {
            needed_gb: needed as f64 / 1e9,
            budget_gb: budget as f64 / 1e9,
            nbf,
            npts,
            vv10: ", +VV10 NLC grid (not batchable, and does not fit alone)",
        });
    }
    Ok(false)
}

/// Resolve the batch size (points per batch) for the batched semilocal V_xc
/// fallback. A pure function of `(npts, nbf, budget)` only — NEVER of thread
/// count — so batch boundaries (and therefore which grid points land in which
/// batch) are identical no matter how many rayon workers are configured,
/// which is what keeps the batched-path energy thread-count-invariant.
///
/// Sizes the batch so `BATCH_PLANES` resident `(nbf, batch_pts)` planes fit
/// the budget, clamped to `[1, npts]` (always makes forward progress, and
/// never batches more finely than the whole grid when it already fits).
fn resolve_batch_size(nbf: usize, npts: usize, budget: usize) -> usize {
    let per_point = BATCH_PLANES.saturating_mul(nbf).saturating_mul(8).max(1);
    let batch_pts = (budget / per_point).max(1);
    batch_pts.min(npts.max(1))
}

/// Closed-shell batched V_xc/E_xc: walk `grid` in contiguous `batch_pts`-sized
/// ranges, evaluating χ/∇χ for just that range (never the whole grid at
/// once), and accumulate the V_xc contribution into `f` and the E_xc total.
///
/// Batches are processed in ascending order, one at a time (serial across
/// batches — each batch's `f +=` and running `e_xc` sum happens before the
/// next batch starts); all the intra-batch parallel structure inside
/// `eval_basis_and_grad_on_points`/`semilocal_vxc_closed_scratch` (rayon over
/// that batch's points) is unchanged and still applies per batch. Because
/// batch boundaries are a pure function of `(npts, nbf, budget)` — see
/// `resolve_batch_size` — and batches are always summed in the same ascending
/// order, the result is thread-count independent, matching the invariant the
/// `Full`/non-batched path already provides.
///
/// The result is NOT expected to be bit-identical to the `Full`-cache path:
/// `semilocal_vxc_closed_scratch`'s internal `deterministic_point_sum` groups
/// the full point range into `EXC_SUM_GROUPS` chunks for its Σ w·ρ·ε_xc
/// reduction; calling it once per (smaller) batch changes which points land
/// in which reduction group, re-associating the same sum — a ~1e-13-relative
/// floating-point reordering, not a physics difference (see the
/// `batched_matches_full_cache_small_system` regression test, which asserts
/// agreement to ≤1e-10 Ha).
#[allow(clippy::too_many_arguments)]
fn batched_add_xc_closed(
    mol: &Molecule,
    bs: &BasisSet,
    grid: &[GridPoint],
    batch_pts: usize,
    d: &Array2<f64>,
    xc: &XcDef,
    is_mgga: bool,
    scratch: &mut VxcScratch,
    f: &mut Array2<f64>,
) -> Result<f64, KsXcError> {
    // Shells depend only on (mol, bs), not on the batch — collect ONCE outside
    // the loop instead of re-parsing them every batch.
    let shells = collect_shells(mol, bs)?;
    let nbf: usize = shells.iter().map(|s| ferric_core::basis::num_functions(s.l, s.pure)).sum();

    let npts = grid.len();
    let batch_pts = batch_pts.max(1);
    let mut e_xc_total = 0.0_f64;
    let mut g0 = 0usize;
    while g0 < npts.max(1) {
        let g1 = (g0 + batch_pts).min(npts);
        if g1 <= g0 {
            break;
        }
        let batch_grid = &grid[g0..g1];
        let pts: Vec<[f64; 3]> = batch_grid.iter().map(|g| g.xyz).collect();
        // Each batch's χ/∇χ live only for this loop iteration — the whole
        // point of batching is that the full-grid tensors are never
        // materialized at once.
        //
        // `_unchecked`: the caller (`KsXc::new`) already resolved the memory
        // budget ONCE and sized `batch_pts` against it — re-resolving and
        // re-checking here (as `eval_basis_and_grad_on_points` would) reads a
        // budget that may have drifted (shrunk) since `new()` ran, which
        // protects nothing (the sizing already accounted for the real budget
        // with better information) and was the mid-run panic site this
        // function used to hit. The only error this can still return is a
        // genuine per-shell `UnsupportedL`, propagated via `?` — never a
        // budget failure.
        let (chi, dchi) = eval_basis_and_grad_on_points_unchecked(&shells, nbf, &pts)?;
        let dens = eval_density_closed(d, &chi, &dchi);
        let tau: Option<Array1<f64>> = if is_mgga { Some(eval_tau_closed(d, &dchi)) } else { None };
        let (e_xc_batch, vxc_batch) = semilocal_vxc_closed_scratch(
            batch_grid, &chi, &dchi, &dens, tau.as_ref(), xc, scratch,
        );
        *f += &vxc_batch;
        e_xc_total += e_xc_batch;
        g0 = g1;
    }
    Ok(e_xc_total)
}

/// Spin-polarized (UKS/ROKS) counterpart of [`batched_add_xc_closed`] — same
/// batching/accumulation-order contract, calling the polarized density/V_xc
/// evaluators per batch instead.
#[allow(clippy::too_many_arguments)]
fn batched_add_xc_uks(
    mol: &Molecule,
    bs: &BasisSet,
    grid: &[GridPoint],
    batch_pts: usize,
    d_a: &Array2<f64>,
    d_b: &Array2<f64>,
    xc: &XcDef,
    is_mgga: bool,
    scratch: &mut VxcScratch,
    f_a: &mut Array2<f64>,
    f_b: &mut Array2<f64>,
) -> Result<f64, KsXcError> {
    let shells = collect_shells(mol, bs)?;
    let nbf: usize = shells.iter().map(|s| ferric_core::basis::num_functions(s.l, s.pure)).sum();

    let npts = grid.len();
    let batch_pts = batch_pts.max(1);
    let mut e_xc_total = 0.0_f64;
    let mut g0 = 0usize;
    while g0 < npts.max(1) {
        let g1 = (g0 + batch_pts).min(npts);
        if g1 <= g0 {
            break;
        }
        let batch_grid = &grid[g0..g1];
        let pts: Vec<[f64; 3]> = batch_grid.iter().map(|g| g.xyz).collect();
        // See `batched_add_xc_closed`'s comment: `_unchecked` skips the
        // per-batch budget re-check that used to be a mid-run panic site
        // under a drifted (auto-detect) budget reading.
        let (chi, dchi) = eval_basis_and_grad_on_points_unchecked(&shells, nbf, &pts)?;
        let dens = eval_density_uks(d_a, d_b, &chi, &dchi);
        let tau = if is_mgga { Some(eval_tau_uks(d_a, d_b, &dchi)) } else { None };
        let tau_ref = tau.as_ref().map(|(a, b)| (a, b));
        let (e_xc_batch, vxc_a_batch, vxc_b_batch) = semilocal_vxc_polarized_scratch(
            batch_grid, &chi, &dchi, &dens, tau_ref, xc, scratch,
        );
        *f_a += &vxc_a_batch;
        *f_b += &vxc_b_batch;
        e_xc_total += e_xc_batch;
        g0 = g1;
    }
    Ok(e_xc_total)
}

/// How the main grid's χ/∇χ are made available to the per-iteration V_xc
/// build.
///
/// `Full` is today's behavior (and remains the default whenever the tensors
/// fit the memory budget): χ/∇χ for the *entire* main grid are evaluated once
/// in `new()` and reused unchanged every SCF iteration — byte-identical to
/// the pre-batching code path.
///
/// `Batched` activates only when the full cache would exceed the resolved
/// budget: the main grid is walked in `batch_pts`-sized point-batches, each
/// batch's χ/∇χ evaluated on demand (never all resident at once) and its
/// V_xc/E_xc contribution accumulated into the running total. Batch
/// boundaries are a pure function of `(npts, nbf, budget)` (`resolve_batch_size`)
/// — never of thread count — so which points land in which batch is fixed
/// for a given system/budget, independent of how rayon happens to schedule
/// the intra-batch parallel work.
enum GridCache {
    Full { chi: Array2<f64>, dchi: Array3<f64> },
    Batched { batch_pts: usize },
}

/// Caches everything needed to compute V_xc and V_nl per SCF iteration:
/// a Becke-Lebedev grid plus precomputed χ, ∇χ on its points (or, over
/// budget, a batch size to reconstruct them on demand — see [`GridCache`]).
/// If the XC definition includes VV10, also caches a smaller NLC grid + AO
/// (VV10's O(npts²) pair sum needs its own grid fully resident regardless —
/// see `check_grid_budget` — so the NLC cache is never batched).
pub struct KsXc {
    pub xc: XcDef,
    pub grid: Vec<GridPoint>,
    cache: GridCache,
    /// Owned clones of the molecule/basis, kept ONLY so `GridCache::Batched`
    /// can re-evaluate χ/∇χ for one batch of grid points at a time from
    /// `add_xc` (which takes `&self`, long after `new()`'s borrowed
    /// `mol`/`bs` references have expired). Unused (and cheap to have
    /// cloned once) on the `Full` path.
    mol: Molecule,
    bs: BasisSet,
    pub nlc_grid: Option<Vec<GridPoint>>,
    pub nlc_chi: Option<Array2<f64>>,
    pub nlc_dchi: Option<Array3<f64>>,
    /// True when any component functional is a meta-GGA (needs τ per iteration).
    /// Cached so the per-Fock-build path skips the τ GEMMs for LDA/GGA/hybrid.
    is_mgga: bool,
    /// Pre-scaled χ scratch reused across SCF iterations (`add_xc` takes
    /// `&self`, hence the Mutex; it is uncontended — one lock per Fock build).
    /// Under `GridCache::Batched` this is reused across batches too — since
    /// `batch_pts` is fixed for the life of `self`, every batch (but possibly
    /// the last, shorter one) shares one shape, so the "reallocate only on
    /// shape change" behavior of `VxcScratch::ensure` still amortizes.
    scratch: Mutex<VxcScratch>,
    /// VV10's own scratch: it is sized for the NLC grid, and sharing the
    /// semilocal scratch would flip the buffer shape twice per iteration
    /// (main ↔ NLC), turning the amortization into two big reallocs per
    /// Fock build. `(nbf, npts_nlc)` — much smaller than the main scratch.
    nlc_scratch: Mutex<VxcScratch>,
}

impl KsXc {
    pub fn new(
        mol: &Molecule,
        bs: &BasisSet,
        xc_name: &str,
        main: &AtomicGridConfig,
        nlc: &AtomicGridConfig,
    ) -> Result<Self, KsXcError> {
        Self::new_with_omega(mol, bs, xc_name, main, nlc, None)
    }

    /// [`Self::new`] with an optional range-separation ω override (Bohr⁻¹) —
    /// hard-errors for functionals without an `_omega` parameter (see
    /// `xc_def_from_name_nspin_omega`). `None` is byte-identical to `new`.
    pub fn new_with_omega(
        mol: &Molecule,
        bs: &BasisSet,
        xc_name: &str,
        main: &AtomicGridConfig,
        nlc: &AtomicGridConfig,
        omega: Option<f64>,
    ) -> Result<Self, KsXcError> {
        let xc = match omega {
            None => xc_def_from_name(xc_name)?,
            Some(w) => crate::libxc::xc_def_from_name_nspin_omega(xc_name, 1, w)?,
        };

        let grid = build_atomic_grid(mol, main);
        let nbf = nbasis(mol, bs)?;
        // Resolve the memory budget ONCE here and reuse the same value for
        // both the Full-vs-Batched decision and (if batching) sizing the
        // batch — see `check_grid_budget`'s doc comment for why re-resolving
        // (a live, SCF-allocation-shrinking reading in the auto-detect case)
        // between these two steps would be unsound.
        let budget = ferric_core::memory::resolve_budget_bytes(None);
        let fits = check_grid_budget(nbf, grid.len(), xc.vv10.is_some(), budget)?;
        let cache = if fits {
            let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
            let (chi, dchi) = eval_basis_and_grad_on_points(mol, bs, &pts)?;
            GridCache::Full { chi, dchi }
        } else {
            let batch_pts = resolve_batch_size(nbf, grid.len(), budget);
            GridCache::Batched { batch_pts }
        };

        let (nlc_grid, nlc_chi, nlc_dchi) = if xc.vv10.is_some() {
            let g = build_atomic_grid(mol, nlc);
            let p: Vec<[f64; 3]> = g.iter().map(|gp| gp.xyz).collect();
            let (c, dc) = eval_basis_and_grad_on_points(mol, bs, &p)?;
            (Some(g), Some(c), Some(dc))
        } else {
            (None, None, None)
        };

        let is_mgga = xc.funcs.iter().any(|f| matches!(f.family(), FunctionalFamily::MetaGga));
        Ok(Self {
            xc, grid, cache, mol: mol.clone(), bs: bs.clone(),
            nlc_grid, nlc_chi, nlc_dchi, is_mgga,
            scratch: Mutex::new(VxcScratch::new()),
            nlc_scratch: Mutex::new(VxcScratch::new()),
        })
    }
}

impl XcContribution for KsXc {
    fn add_xc(&self, d: &Array2<f64>, f: &mut Array2<f64>) -> f64 {
        let mut scratch = self.scratch.lock().expect("vxc scratch mutex poisoned");
        let e_xc = match &self.cache {
            GridCache::Full { chi, dchi } => {
                let dens = eval_density_closed(d, chi, dchi);
                // τ (kinetic-energy density) is only needed by meta-GGA
                // functionals. Compute it from D and ∇χ (no explicit occupied
                // MOs required — see eval_tau_closed) only when a meta-GGA
                // component is present.
                let tau = if self.is_mgga { Some(eval_tau_closed(d, dchi)) } else { None };
                let (e_xc, vxc) = semilocal_vxc_closed_scratch(
                    &self.grid, chi, dchi, &dens, tau.as_ref(), &self.xc, &mut scratch,
                );
                *f += &vxc;
                e_xc
            }
            GridCache::Batched { batch_pts } => batched_add_xc_closed(
                &self.mol, &self.bs, &self.grid, *batch_pts, d, &self.xc, self.is_mgga,
                &mut scratch, f,
            )
            .expect(
                "batched AO evaluation failed for a basis already accepted by KsXc::new \
                 (nbasis/full-grid eval succeeded there) — this can only be a genuine \
                 per-shell error (e.g. UnsupportedL), never the memory-budget re-check \
                 that used to panic here",
            ),
        };

        // VV10 nonlocal correlation (add_vv10_scratch, vv10.rs): full energy
        // + potential on the (usually coarser) NLC grid, only when the
        // functional carries VV10 params (e.g. wB97X-V) and the NLC grid was
        // built; 0.0 for any functional without an NLC term.
        let e_nl = if let (Some(g), Some(c), Some(dc), Some(params)) = (
            self.nlc_grid.as_ref(),
            self.nlc_chi.as_ref(),
            self.nlc_dchi.as_ref(),
            self.xc.vv10.as_ref(),
        ) {
            let nlc_dens = eval_density_closed(d, c, dc);
            let mut nlc_scratch =
                self.nlc_scratch.lock().expect("vv10 scratch mutex poisoned");
            add_vv10_scratch(g, c, dc, &nlc_dens, params, f, &mut nlc_scratch)
        } else {
            0.0
        };

        e_xc + e_nl
    }

    fn k_mix(&self) -> KMix {
        if let Some(cam) = self.xc.cam {
            return KMix { sr: cam.c_sr, lr: cam.c_lr, omega: cam.omega };
        }
        if let Some(mix) = self.xc.b3lyp_mix {
            return KMix { sr: mix, lr: mix, omega: 0.0 };
        }
        // Pure functional (LDA, PBE): no exact exchange
        KMix { sr: 0.0, lr: 0.0, omega: 0.0 }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Spin-polarized counterpart (UKS / ROKS)
// ────────────────────────────────────────────────────────────────────────────

/// UKS analog of `KsXc`. Caches the same grids + AO data, but the
/// `XcDef` is built with `nspin=2` so libxc returns spin-resolved
/// v_ρ / v_σ. VV10 is closed-shell-friendly (function of total ρ and |∇ρ|²);
/// the V_nl piece is added equally to V_α and V_β.
pub struct KsXcUks {
    pub xc: XcDef,
    pub grid: Vec<GridPoint>,
    cache: GridCache,
    /// See `KsXc::mol`/`bs` — only used by the `GridCache::Batched` path.
    mol: Molecule,
    bs: BasisSet,
    pub nlc_grid: Option<Vec<GridPoint>>,
    pub nlc_chi: Option<Array2<f64>>,
    pub nlc_dchi: Option<Array3<f64>>,
    /// True when any component functional is a meta-GGA (needs per-spin τ).
    is_mgga: bool,
    /// See `KsXc::scratch` — shared by both spin builds.
    scratch: Mutex<VxcScratch>,
    /// See `KsXc::nlc_scratch` — VV10-only, sized for the NLC grid.
    nlc_scratch: Mutex<VxcScratch>,
}

impl KsXcUks {
    pub fn new(
        mol: &Molecule,
        bs: &BasisSet,
        xc_name: &str,
        main: &AtomicGridConfig,
        nlc: &AtomicGridConfig,
    ) -> Result<Self, KsXcError> {
        Self::new_with_omega(mol, bs, xc_name, main, nlc, None)
    }

    /// UKS twin of `KsXc::new_with_omega`.
    pub fn new_with_omega(
        mol: &Molecule,
        bs: &BasisSet,
        xc_name: &str,
        main: &AtomicGridConfig,
        nlc: &AtomicGridConfig,
        omega: Option<f64>,
    ) -> Result<Self, KsXcError> {
        let xc = match omega {
            None => xc_def_from_name_nspin(xc_name, 2)?,
            Some(w) => crate::libxc::xc_def_from_name_nspin_omega(xc_name, 2, w)?,
        };

        let grid = build_atomic_grid(mol, main);
        let nbf = nbasis(mol, bs)?;
        // See `KsXc::new` — resolve the budget ONCE and reuse it for both the
        // decision and the batch sizing.
        let budget = ferric_core::memory::resolve_budget_bytes(None);
        let fits = check_grid_budget(nbf, grid.len(), xc.vv10.is_some(), budget)?;
        let cache = if fits {
            let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
            let (chi, dchi) = eval_basis_and_grad_on_points(mol, bs, &pts)?;
            GridCache::Full { chi, dchi }
        } else {
            let batch_pts = resolve_batch_size(nbf, grid.len(), budget);
            GridCache::Batched { batch_pts }
        };

        let (nlc_grid, nlc_chi, nlc_dchi) = if xc.vv10.is_some() {
            let g = build_atomic_grid(mol, nlc);
            let p: Vec<[f64; 3]> = g.iter().map(|gp| gp.xyz).collect();
            let (c, dc) = eval_basis_and_grad_on_points(mol, bs, &p)?;
            (Some(g), Some(c), Some(dc))
        } else {
            (None, None, None)
        };

        let is_mgga = xc.funcs.iter().any(|f| matches!(f.family(), FunctionalFamily::MetaGga));
        Ok(Self {
            xc, grid, cache, mol: mol.clone(), bs: bs.clone(),
            nlc_grid, nlc_chi, nlc_dchi, is_mgga,
            scratch: Mutex::new(VxcScratch::new()),
            nlc_scratch: Mutex::new(VxcScratch::new()),
        })
    }
}

impl UksXcContribution for KsXcUks {
    fn add_xc_uks(
        &self,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
        f_a: &mut Array2<f64>,
        f_b: &mut Array2<f64>,
    ) -> f64 {
        // Semilocal: build (ρ_α, ρ_β, σ_αα, σ_αβ, σ_ββ) on the main grid
        // then call the polarized libxc path.
        let mut scratch = self.scratch.lock().expect("vxc scratch mutex poisoned");
        let e_xc = match &self.cache {
            GridCache::Full { chi, dchi } => {
                let dens = eval_density_uks(d_a, d_b, chi, dchi);
                // Per-spin τ only for meta-GGA (from the two spin density matrices).
                let tau = if self.is_mgga { Some(eval_tau_uks(d_a, d_b, dchi)) } else { None };
                let tau_ref = tau.as_ref().map(|(a, b)| (a, b));
                let (e_xc, vxc_a, vxc_b) = semilocal_vxc_polarized_scratch(
                    &self.grid, chi, dchi, &dens, tau_ref, &self.xc, &mut scratch,
                );
                *f_a += &vxc_a;
                *f_b += &vxc_b;
                e_xc
            }
            GridCache::Batched { batch_pts } => batched_add_xc_uks(
                &self.mol, &self.bs, &self.grid, *batch_pts, d_a, d_b, &self.xc, self.is_mgga,
                &mut scratch, f_a, f_b,
            )
            .expect(
                "batched AO evaluation failed for a basis already accepted by KsXcUks::new \
                 (nbasis/full-grid eval succeeded there) — this can only be a genuine \
                 per-shell error (e.g. UnsupportedL), never the memory-budget re-check \
                 that used to panic here",
            ),
        };

        // VV10 nonlocal correlation: function of (ρ_tot, |∇ρ_tot|²); the
        // V_nl matrix is the same for both spins. Cache the d_total → DensityGrid
        // and reuse the closed-shell add_vv10.
        let e_nl = if let (Some(g), Some(c), Some(dc), Some(params)) = (
            self.nlc_grid.as_ref(),
            self.nlc_chi.as_ref(),
            self.nlc_dchi.as_ref(),
            self.xc.vv10.as_ref(),
        ) {
            let d_total = d_a + d_b;
            let dens_total: DensityGrid = eval_density_closed(&d_total, c, dc);
            // Apply VV10 to a single Fock buffer, then add to both spins.
            let mut nlc_scratch =
                self.nlc_scratch.lock().expect("vv10 scratch mutex poisoned");
            let mut v_nl = Array2::<f64>::zeros(f_a.dim());
            let e = add_vv10_scratch(g, c, dc, &dens_total, params, &mut v_nl, &mut nlc_scratch);
            *f_a += &v_nl;
            *f_b += &v_nl;
            e
        } else {
            0.0
        };

        e_xc + e_nl
    }

    fn k_mix(&self) -> KMix {
        if let Some(cam) = self.xc.cam {
            return KMix { sr: cam.c_sr, lr: cam.c_lr, omega: cam.omega };
        }
        if let Some(mix) = self.xc.b3lyp_mix {
            return KMix { sr: mix, lr: mix, omega: 0.0 };
        }
        KMix { sr: 0.0, lr: 0.0, omega: 0.0 }
    }
}

#[cfg(test)]
mod batching_tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;

    // FERRIC_MEM_BUDGET_GB is process-global; serialize any test that sets it
    // (same convention as ao_grid.rs's budget_guard_tests).
    // Shared crate-wide lock (see lib.rs) — a module-local lock cannot stop
    // cross-module races on the process-global budget env var.
    use crate::TEST_BUDGET_ENV_LOCK as ENV_LOCK;
    const VAR: &str = ferric_core::memory::ENV_UNIFIED;

    fn clear_budget_env() {
        std::env::remove_var(VAR);
    }

    #[test]
    fn resolve_batch_size_is_pure_function_of_npts_nbf_budget() {
        // Same inputs -> same batch size, regardless of anything
        // thread-count-related (the function doesn't even see a thread count).
        let a = resolve_batch_size(25, 35_000, 10_000_000);
        let b = resolve_batch_size(25, 35_000, 10_000_000);
        assert_eq!(a, b);
        // Sanity: never zero, never bigger than npts.
        assert!(a >= 1);
        assert!(a <= 35_000);
        // A tiny budget still makes forward progress (>=1 point per batch).
        assert_eq!(resolve_batch_size(1500, 400_000, 1), 1);
        // A huge budget batches to (at most) the whole grid in one go.
        assert_eq!(resolve_batch_size(7, 100, usize::MAX / 2), 100);
    }

    /// Batched V_xc/E_xc must agree with the cached (`Full`) path to ≤1e-10 Ha
    /// on a small system, with batching forced via a tiny `FERRIC_MEM_BUDGET_GB`
    /// override (task requirement: regression test comparing batched vs cached).
    #[test]
    fn batched_matches_full_cache_small_system_closed_shell() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_budget_env();

        let mol = Molecule::parse_xyz("3\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.96\nH 0.93 0.0 -0.24\n", 0, 1)
            .unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = ferric_scf::screening::SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();

        // A converged RHF density is a fine, purely-mechanical test input here
        // — this test checks that batching reproduces the cached-path numbers
        // for a FIXED D, not that PBE/SCF converges to anything in particular.
        let rhf = ferric_scf::rhf::solve_rhf(
            &ctx, &mol, &prep, op, &bounds, &ferric_scf::rhf::RhfConfig::default(),
        )
        .unwrap();
        let d = rhf.density_total;

        // Small, coarse grid so the `Full` cache is tiny (a handful of KB) —
        // this keeps the "over-budget" trigger firmly in the test's control
        // via a tiny FERRIC_MEM_BUDGET_GB rather than actually needing GBs.
        let main = AtomicGridConfig { n_radial: 20, n_angular: 26, ..Default::default() };
        let nlc = AtomicGridConfig { n_radial: 10, n_angular: 26, ..Default::default() };

        // 1) Full-cache path: plenty of budget.
        clear_budget_env();
        std::env::set_var(VAR, "1000"); // 1000 GiB — never over budget
        let ks_full = KsXc::new(&mol, &bs, "PBE", &main, &nlc).unwrap();
        assert!(
            matches!(ks_full.cache, GridCache::Full { .. }),
            "large budget must take the Full cache path"
        );
        let mut f_full = Array2::<f64>::zeros(d.dim());
        let e_full = ks_full.add_xc(&d, &mut f_full);

        // 2) Batched path: force it via a tiny budget override. The grid here
        // is nbf~7 (STO-3G water) * npts ~ 20*26*3 ≈ 1560 points, so the Full
        // cache is 4*7*1560*8 ≈ 350 KB — a ~10 KB budget guarantees batching
        // while resolve_batch_size still yields >=1 point/batch (forward
        // progress guaranteed by construction, see the test above).
        clear_budget_env();
        std::env::set_var(VAR, "0.00001"); // ≈ 10.7 KB
        let ks_batched = KsXc::new(&mol, &bs, "PBE", &main, &nlc).unwrap();
        assert!(
            matches!(ks_batched.cache, GridCache::Batched { .. }),
            "tiny budget must take the Batched fallback path (must NOT fail)"
        );
        let mut f_batched = Array2::<f64>::zeros(d.dim());
        let e_batched = ks_batched.add_xc(&d, &mut f_batched);

        clear_budget_env();

        let e_diff = (e_full - e_batched).abs();
        assert!(
            e_diff <= 1e-10,
            "batched E_xc+E_nl must match the cached path to <=1e-10 Ha, got \
             full={e_full:.15}, batched={e_batched:.15}, diff={e_diff:.3e}"
        );

        let f_diff = (&f_full - &f_batched)
            .iter()
            .fold(0.0_f64, |m, &x| m.max(x.abs()));
        assert!(
            f_diff <= 1e-10,
            "batched V_xc Fock contribution must match the cached path to \
             <=1e-10 Ha elementwise, max abs diff={f_diff:.3e}"
        );
    }

    /// Same cross-check for the spin-polarized (UKS) path.
    #[test]
    fn batched_matches_full_cache_small_system_uks() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_budget_env();

        // OH radical (doublet) — exercises genuinely different α/β densities.
        let mol = Molecule::parse_xyz("2\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n", 0, 2).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = ferric_scf::screening::SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let uhf = ferric_scf::uhf::solve_uhf(
            &ctx, &mol, &prep, &bounds, &ferric_scf::uhf::UhfConfig::default(),
        )
        .unwrap();
        let d_a: Array2<f64> = uhf.density_alpha;
        let d_b: Array2<f64> = uhf.density_beta.unwrap();

        let main = AtomicGridConfig { n_radial: 20, n_angular: 26, ..Default::default() };
        let nlc = AtomicGridConfig { n_radial: 10, n_angular: 26, ..Default::default() };

        clear_budget_env();
        std::env::set_var(VAR, "1000");
        let ks_full = KsXcUks::new(&mol, &bs, "PBE", &main, &nlc).unwrap();
        assert!(matches!(ks_full.cache, GridCache::Full { .. }));
        let mut fa_full = Array2::<f64>::zeros(d_a.dim());
        let mut fb_full = Array2::<f64>::zeros(d_b.dim());
        let e_full = ks_full.add_xc_uks(&d_a, &d_b, &mut fa_full, &mut fb_full);

        clear_budget_env();
        std::env::set_var(VAR, "0.00001");
        let ks_batched = KsXcUks::new(&mol, &bs, "PBE", &main, &nlc).unwrap();
        assert!(matches!(ks_batched.cache, GridCache::Batched { .. }));
        let mut fa_batched = Array2::<f64>::zeros(d_a.dim());
        let mut fb_batched = Array2::<f64>::zeros(d_b.dim());
        let e_batched = ks_batched.add_xc_uks(&d_a, &d_b, &mut fa_batched, &mut fb_batched);

        clear_budget_env();

        let e_diff = (e_full - e_batched).abs();
        assert!(e_diff <= 1e-10, "UKS batched E_xc must match cached to <=1e-10 Ha, diff={e_diff:.3e}");

        let fa_diff = (&fa_full - &fa_batched).iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
        let fb_diff = (&fb_full - &fb_batched).iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
        assert!(fa_diff <= 1e-10, "UKS batched V_alpha must match cached, diff={fa_diff:.3e}");
        assert!(fb_diff <= 1e-10, "UKS batched V_beta must match cached, diff={fb_diff:.3e}");
    }

    /// Regression for the mid-run panic this task fixes: the OLD batched path
    /// called `eval_basis_and_grad_on_points` per batch, which re-resolved
    /// `FERRIC_MEM_BUDGET_GB` (a LIVE reading in the auto-detect case) inside
    /// `check_ao_grid_budget` on every single batch. If that budget had
    /// shrunk since `KsXc::new` originally sized `batch_pts` against it, the
    /// per-batch re-check could reject a batch the caller had already sized
    /// correctly, and the `.expect(...)` at the `add_xc` call site would
    /// panic a running SCF job.
    ///
    /// This test constructs `KsXc` under one tiny budget (forcing `Batched`
    /// and sizing `batch_pts` against it), then — BEFORE calling `add_xc` —
    /// shrinks the budget env var further still, simulating memory draining
    /// mid-SCF. `add_xc` must still succeed: `KsXc::new` resolves the budget
    /// ONCE and the batched path (`eval_basis_and_grad_on_points_unchecked`)
    /// never re-resolves or re-checks it, so a shrinking live budget between
    /// construction and use cannot make it panic.
    #[test]
    fn batched_add_xc_survives_budget_shrinking_after_construction() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_budget_env();

        let mol = Molecule::parse_xyz("3\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.96\nH 0.93 0.0 -0.24\n", 0, 1)
            .unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = ferric_scf::screening::SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let rhf = ferric_scf::rhf::solve_rhf(
            &ctx, &mol, &prep, op, &bounds, &ferric_scf::rhf::RhfConfig::default(),
        )
        .unwrap();
        let d = rhf.density_total;

        let main = AtomicGridConfig { n_radial: 20, n_angular: 26, ..Default::default() };
        let nlc = AtomicGridConfig { n_radial: 10, n_angular: 26, ..Default::default() };

        // Construct under a tiny (but not absurdly tiny) budget: forces
        // Batched, and sizes batch_pts against ~10.7 KB.
        std::env::set_var(VAR, "0.00001");
        let ks = KsXc::new(&mol, &bs, "PBE", &main, &nlc).unwrap();
        assert!(matches!(ks.cache, GridCache::Batched { .. }), "expected Batched under a tiny budget");

        // Simulate the budget draining further mid-SCF (e.g. another
        // allocation elsewhere in the process shrinking 0.8×MemAvailable):
        // set the env var to something even smaller than what `batch_pts`
        // was sized against. A re-resolving per-batch check (the old bug)
        // would now see a smaller ceiling than `new()` did and could reject
        // the very batch size `new()` already committed to.
        std::env::set_var(VAR, "0.000001"); // ~1 KB — smaller than the ~10.7 KB construction budget

        let mut f = Array2::<f64>::zeros(d.dim());
        // Must NOT panic: the batched path is immune to the budget having
        // changed after construction.
        let e_xc = ks.add_xc(&d, &mut f);

        clear_budget_env();

        assert!(e_xc.is_finite(), "batched add_xc must produce a finite E_xc despite budget shrinking mid-run");
        assert!(f.iter().all(|x| x.is_finite()), "batched V_xc Fock contribution must be finite");
    }

    /// Directly exercises `eval_basis_and_grad_on_points_unchecked`: sized
    /// for exactly one Batched-path batch, it must succeed even when a fresh
    /// `check_ao_grid_budget` call for the SAME (nbf, npts) would fail under
    /// an absurdly tiny budget — proving the unchecked evaluator really does
    /// skip the re-check rather than just happening not to trip it.
    #[test]
    fn eval_basis_and_grad_on_points_unchecked_ignores_a_failing_live_budget() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_budget_env();

        let mol = Molecule::parse_xyz("3\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.96\nH 0.93 0.0 -0.24\n", 0, 1)
            .unwrap();
        let bs = basis::bundled("sto-3g").unwrap();

        let shells = crate::ao_grid::collect_shells(&mol, &bs).unwrap();
        let nbf = crate::ao_grid::nbasis(&mol, &bs).unwrap();
        let pts: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]; 8];

        // Confirm a tiny live budget really would make the CHECKED path fail
        // for this (nbf, npts) — otherwise this test would prove nothing.
        // nbf=7 (STO-3G water) * npts=8 * 4 planes * 8 bytes = 1792 bytes
        // needed; ~1 KB is smaller than that but not gratuitously extreme
        // (matches the magnitude the rest of this module's tests already use).
        std::env::set_var(VAR, "0.000001"); // ~1 KB
        let checked = eval_basis_and_grad_on_points(&mol, &bs, &pts);
        assert!(checked.is_err(), "sanity: tiny budget must fail the checked path");

        // The unchecked variant, called with the SAME tiny budget still set,
        // must still succeed — it never resolves or checks the budget at all.
        let unchecked = eval_basis_and_grad_on_points_unchecked(&shells, nbf, &pts);
        clear_budget_env();
        assert!(unchecked.is_ok(), "unchecked evaluator must succeed regardless of the live budget");
    }
}
