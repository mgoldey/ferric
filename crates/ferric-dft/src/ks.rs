//! `KsXc` / `KsXcUks`: the Kohn-Sham exchange-correlation contribution to the
//! SCF Fock matrix, implementing `XcContribution` / `UksXcContribution`.
//!
//! The main grid is integrated by the batched, basis-function-screened path in
//! [`crate::xc_batch`]: the grid is split into spatially compact point batches,
//! each batch keeps only the shells that are numerically significant on it,
//! and every Fock build runs parallel over batches with a fixed-order,
//! thread-count-invariant reduction. The per-batch compact AO blocks are held
//! resident when they fit the memory budget, and re-evaluated per Fock build
//! otherwise; the two modes are bit-identical, so the budget can move memory
//! and time but never an energy. There is no dense `(nbf, npts)` χ/∇χ cache
//! for the main grid any more.
//!
//! The optional VV10 NLC grid keeps its dense χ/∇χ cache (VV10's O(npts²)
//! pair sum needs the whole NLC grid resident; see `vv10.rs`).

use std::sync::Mutex;

use ndarray::{Array2, Array3};
use thiserror::Error;

use ferric_core::basis::BasisSet;
use ferric_core::memory::pool::Reservation;
use ferric_core::mol::Molecule;
use ferric_core::precision_ladder::{LadderSite, PrecisionLadder};

use crate::ao_grid::{collect_shells, eval_basis_and_grad_on_points, nbasis, GtoEvalError};
use crate::density_on_grid::{eval_density_closed, DensityGrid};
use crate::grid::{build_atomic_grid_pruned, AtomicGridConfig, GridPoint};
use crate::libxc::{xc_def_from_name, xc_def_from_name_nspin, FunctionalFamily, LibxcError, XcDef};
use crate::vv10::add_vv10_scratch;
use crate::vxc::VxcScratch;
use crate::xc_batch::{
    mixed_vxc_requested, owned_shells, AoStorage, DensityInput, Exec, ScreenedGrid, XcBatchConfig,
    XcScreeningStats,
};
use crate::xc_trait::{KMix, UksXcContribution, XcContribution};

/// Errors from the Kohn-Sham exchange-correlation integration path.
#[derive(Error, Debug)]
#[non_exhaustive]
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
    /// Grid construction failed. In practice this is angular-order pruning
    /// rejecting the requested `n_angular` (see [`crate::prune::region_orders`]
    /// — notably `n_angular = 50`, which has no useful pruned table on ferric's
    /// Lebedev set). Surfaced rather than silently falling back to a flat grid,
    /// per the config-honesty convention.
    #[error("DFT grid construction failed: {0}")]
    Grid(ferric_core::error::FerricError),
    /// Invalid internal batching/screening configuration
    /// ([`crate::xc_batch::XcBatchConfig`]) — e.g. a NaN threshold, which
    /// would otherwise silently drop every shell.
    #[error("invalid XC batch config: {0}")]
    BatchConfig(String),
}

impl From<GtoEvalError> for KsXcError {
    fn from(e: GtoEvalError) -> Self {
        Self::Eval(e)
    }
}
impl From<LibxcError> for KsXcError {
    fn from(e: LibxcError) -> Self {
        Self::Libxc(e)
    }
}

impl From<KsXcError> for ferric_core::error::FerricError {
    fn from(e: KsXcError) -> Self {
        Self::General(e.to_string())
    }
}

/// Bytes the dense VV10 NLC-grid cache and its per-iteration chain occupy:
/// χ + ∇χ (4 planes) + the NLC `VxcScratch` buffer (1) + `Φ = D·χ` inside
/// `eval_density_closed` (1), plus ~20 O(npts) companion doubles — the same
/// closed-shell accounting the old full-cache gate used, now applied to the
/// NLC grid it actually describes (the old gate doubled the MAIN-grid figure
/// as a proxy for it).
fn nlc_cache_bytes(nbf: usize, npts_nlc: usize) -> usize {
    nbf.saturating_mul(6)
        .saturating_add(20)
        .saturating_mul(npts_nlc)
        .saturating_mul(8)
}

/// Label under which a resident screened AO cache is charged to the pool.
/// Contains "grid AO cache" so occupancy reports keep naming the plane.
const RESIDENT_LABEL: &str = "KS-DFT grid AO cache (screened batches)";

/// Label for the recompute-mode charge (per-Fock-build working set + NLC).
const RECOMPUTE_LABEL: &str = "KS-DFT grid working set (screened batches, recompute)";

/// Stable phrase of the warning printed when the budget forces recompute.
/// `ferric-cli`'s budget MWE asserts on it, so keep it verbatim.
pub const RECOMPUTE_WARNING: &str = "KS-DFT grid AO blocks recomputed every Fock build";

/// Outcome of [`decide_storage`].
#[derive(Debug)]
struct StorageDecision {
    /// Hold the compact AO blocks resident (else recompute per Fock build).
    resident: bool,
    /// Pool charge held for the evaluator's lifetime (inert without a pool,
    /// or when a recompute-mode charge could not be covered — see below).
    charge: Reservation,
    /// Bytes that could NOT be charged (recompute mode only): the working set
    /// runs anyway, because nothing smaller exists. 0 when fully charged.
    uncharged: usize,
}

/// Decide whether the screened main-grid AO blocks are held resident, and
/// charge the pool for what the evaluator will hold.
///
/// * `resident` — bytes of the compact AO blocks.
/// * `transient` — the per-Fock-build working set (O(npts) vectors, group
///   accumulators, per-worker scratch; `ScreenedGrid::transient_bytes`).
/// * `nlc` — the dense VV10 NLC cache bytes when VV10 is present.
///
/// # What is charged, and for how long
///
/// Resident: `resident + transient + nlc`. Recompute: `transient + nlc`.
/// Either charge is held for the evaluator's whole LIFETIME, including
/// `transient`, which is only live during `add_xc`. That is deliberate: the
/// SCF calls `add_xc` every iteration while the DF 3-index tensor and the
/// other planes are live, so the process peak is "everything else + this
/// working set". Charging it only inside `add_xc` would let another plane be
/// admitted against those bytes between iterations and collide with them on
/// the next Fock build. The NLC cache genuinely lives for the evaluator's
/// lifetime.
///
/// # Why recompute never errors
///
/// Recomputation needs no resident AO blocks, and its working set cannot be
/// made smaller (the old per-point batching fallback could shrink to 1 point;
/// this one is bounded below by the O(npts) density/kernel vectors). So an
/// over-budget main grid takes the recompute branch; if even its working set
/// cannot be covered, the job runs uncharged for that amount and reports it
/// via `uncharged` (the caller warns) rather than refusing a job the old
/// fallback would have run. Only the non-batchable VV10 NLC cache is a hard
/// error.
///
/// # Determinism
///
/// The two modes are bit-identical by construction, so this decision —
/// whatever it reads — cannot move an energy.
fn decide_storage(
    storage: AoStorage,
    resident: usize,
    transient: usize,
    nlc: Option<usize>,
    budget: usize,
    nbf: usize,
    npts: usize,
) -> Result<StorageDecision, KsXcError> {
    let nlc_bytes = nlc.unwrap_or(0);
    let has_vv10 = nlc.is_some();
    let lean = transient.saturating_add(nlc_bytes);
    let needed = resident.saturating_add(lean);
    let pool = ferric_core::memory::pool::global();

    // The VV10 NLC cache alone must fit, whatever happens to the main grid.
    let nlc_fits = || -> Result<(), KsXcError> {
        if !has_vv10 {
            return Ok(());
        }
        let (fits, avail) = match &pool {
            Some(p) => (
                p.try_reserve("KS-DFT VV10 NLC probe", nlc_bytes).is_some(),
                p.available_bytes(),
            ),
            None => (nlc_bytes <= budget, budget),
        };
        if fits {
            Ok(())
        } else {
            Err(KsXcError::OverBudget {
                needed_gb: needed as f64 / 1e9,
                budget_gb: avail as f64 / 1e9,
                nbf,
                npts,
                vv10: ", +VV10 NLC grid (not batchable, and does not fit alone)",
            })
        }
    };
    // Recompute-mode charge: `transient + nlc`, or inert + `uncharged` when
    // it cannot be covered.
    let recompute = || -> StorageDecision {
        let covered = match &pool {
            Some(p) => p.try_reserve(RECOMPUTE_LABEL, lean),
            None => (lean <= budget).then(|| Reservation::inert(RECOMPUTE_LABEL)),
        };
        match covered {
            Some(charge) => StorageDecision {
                resident: false,
                charge,
                uncharged: 0,
            },
            None => StorageDecision {
                resident: false,
                charge: Reservation::inert(RECOMPUTE_LABEL),
                uncharged: lean,
            },
        }
    };

    match storage {
        AoStorage::Recompute => {
            nlc_fits()?;
            Ok(recompute())
        }
        // Forced resident (tests): charge the pool when one is installed and
        // it has room, otherwise hold an inert guard — the caller asked for
        // residency explicitly, so a full pool does not veto it.
        AoStorage::Resident => {
            nlc_fits()?;
            let charge = pool
                .as_ref()
                .and_then(|p| p.try_reserve(RESIDENT_LABEL, needed))
                .unwrap_or_else(|| Reservation::inert(RESIDENT_LABEL));
            Ok(StorageDecision {
                resident: true,
                charge,
                uncharged: 0,
            })
        }
        AoStorage::Auto => {
            let fits = match &pool {
                Some(p) => p.try_reserve(RESIDENT_LABEL, needed),
                // No pool: the historical ceiling comparison (trivial limit).
                None => (needed <= budget).then(|| Reservation::inert(RESIDENT_LABEL)),
            };
            if let Some(charge) = fits {
                return Ok(StorageDecision {
                    resident: true,
                    charge,
                    uncharged: 0,
                });
            }
            nlc_fits()?;
            Ok(recompute())
        }
    }
}

/// Everything `KsXc::new*` / `KsXcUks::new*` build besides the `XcDef`.
struct GridBuild {
    grid: Vec<GridPoint>,
    screened: ScreenedGrid,
    charge: Reservation,
    nlc_grid: Option<Vec<GridPoint>>,
    nlc_chi: Option<Array2<f64>>,
    nlc_dchi: Option<Array3<f64>>,
    pending: Option<PendingStorage>,
}

/// The storage decision a deferred evaluator has not taken yet.
///
/// # Why the decision can wait
///
/// The SCF needs the functional (`k_mix`) before it can set up RI-J/K, but not
/// the grid's memory. Deciding storage first let XC take the pool from the raw
/// 3-index tensor, the one plane with no bit-identical alternative: XC can
/// always recompute, raw can only stop the job. So the SCF drivers build the
/// evaluator with `new_deferred`, set up RI-J/K, then call `commit_storage`, and
/// XC gets what is left. That also keeps the raw decision off `transient_bytes`,
/// which reads the thread count.
#[derive(Debug, Clone, Copy)]
struct PendingStorage {
    storage: AoStorage,
    /// Resolved once, at construction: see `grid_working_budget`. Without a pool
    /// it subtracts live RSS, and reading it later would move the no-pool
    /// resident-or-recompute choice.
    budget: usize,
    nbf: usize,
    is_uks: bool,
}

impl PendingStorage {
    /// Charge the pool, keep or skip the AO blocks, evaluate the NLC cache.
    fn settle(
        self,
        screened: &mut ScreenedGrid,
        grid: &[GridPoint],
        nlc_grid: Option<&[GridPoint]>,
        (mol, bs): (&Molecule, &BasisSet),
    ) -> Result<(Reservation, Option<Array2<f64>>, Option<Array3<f64>>), KsXcError> {
        let nlc_bytes = nlc_grid.map(|g| nlc_cache_bytes(self.nbf, g.len()));
        let resident_bytes = screened.resident_ao_bytes();
        let decision = decide_storage(
            self.storage,
            resident_bytes,
            screened.transient_bytes(self.is_uks),
            nlc_bytes,
            self.budget,
            self.nbf,
            grid.len(),
        )?;
        if decision.resident {
            screened.materialize(grid)?;
        } else if self.storage == AoStorage::Auto {
            // The budget, not the caller, forced recompute: say so. Results are
            // bit-identical to resident mode; only speed changes.
            let extra = if decision.uncharged > 0 {
                format!(
                    "; the {:.3} GB per-build working set does not fit what is left either \
                     and runs uncharged, so the job can exceed its budget by that much",
                    decision.uncharged as f64 / 1e9
                )
            } else {
                String::new()
            };
            eprintln!(
                "warning: {RECOMPUTE_WARNING} (bit-identical, slower): the screened AO blocks \
                 need {:.3} GB, which the memory budget cannot hold beside what is already \
                 resident{extra}",
                resident_bytes as f64 / 1e9
            );
        }
        let (nlc_chi, nlc_dchi) = if let Some(g) = nlc_grid {
            let p: Vec<[f64; 3]> = g.iter().map(|gp| gp.xyz).collect();
            let (c, dc) = eval_basis_and_grad_on_points(mol, bs, &p)?;
            (Some(c), Some(dc))
        } else {
            (None, None)
        };
        Ok((decision.charge, nlc_chi, nlc_dchi))
    }
}

/// Stop before any expensive setup when the VV10 NLC cache, which has no
/// fallback, cannot fit even an empty ledger. Takes nothing.
fn nlc_precheck(nlc_bytes: usize, budget: usize, nbf: usize, npts: usize) -> Result<(), KsXcError> {
    let room = ferric_core::memory::pool::global_available_bytes().unwrap_or(budget);
    if nlc_bytes <= room {
        return Ok(());
    }
    Err(KsXcError::OverBudget {
        needed_gb: nlc_bytes as f64 / 1e9,
        budget_gb: room as f64 / 1e9,
        nbf,
        npts,
        vv10: ", +VV10 NLC grid (not batchable, and does not fit alone)",
    })
}

/// Shared constructor body for the closed- and open-shell evaluators.
#[allow(clippy::too_many_arguments)]
fn build_grids(
    mol: &Molecule,
    bs: &BasisSet,
    xc: &XcDef,
    main: &AtomicGridConfig,
    nlc: &AtomicGridConfig,
    memory_budget_bytes: Option<usize>,
    is_uks: bool,
    batch: &XcBatchConfig,
    defer: bool,
) -> Result<GridBuild, KsXcError> {
    // Config honesty: a NaN threshold would compare false against every value
    // and silently drop every shell (E_xc = 0); a zero batch cap is meaningless.
    if !batch.screen_thresh.is_finite() {
        return Err(KsXcError::BatchConfig(format!(
            "screen_thresh must be finite, got {}",
            batch.screen_thresh
        )));
    }
    if batch.max_batch_pts == 0 {
        return Err(KsXcError::BatchConfig(
            "max_batch_pts must be >= 1".to_string(),
        ));
    }

    // Honour `main.prune`. `prune = None` (the default) delegates to
    // `build_atomic_grid` and is bit-identical to the historical grid;
    // `Some(scheme)` drops the angular order on core/tail radial shells.
    // Errors rather than silently un-pruning if the order has no table.
    let grid = build_atomic_grid_pruned(mol, main, main.prune).map_err(KsXcError::Grid)?;
    let nbf = nbasis(mol, bs)?;
    // Resolve the budget ONCE per construction (see `grid_working_budget`).
    let budget = grid_working_budget(memory_budget_bytes);

    // Partition + screen. This evaluates every shell on every point once (the
    // same work the old dense cache construction did), keeping only per-batch
    // shell lists — no AO values are retained yet. It also surfaces a genuine
    // per-shell error (UnsupportedL) here, at construction, so the per-Fock
    // `.expect` below can never see one.
    let shells = collect_shells(mol, bs)?;
    let mut screened = ScreenedGrid::build(&grid, owned_shells(&shells), nbf, batch)?;

    // The NLC grid carries its OWN prune setting, independent of the main
    // grid's. Every construction site builds it at 50x50 with `prune: None`,
    // and pruning has no table at n_angular = 50, so enabling pruning on the
    // main grid must never reach this one. Built before the storage decision
    // so its real size enters the budget.
    let nlc_grid = if xc.vv10.is_some() {
        Some(build_atomic_grid_pruned(mol, nlc, nlc.prune).map_err(KsXcError::Grid)?)
    } else {
        None
    };
    let nlc_bytes = nlc_grid.as_ref().map(|g| nlc_cache_bytes(nbf, g.len()));

    let pending = PendingStorage {
        storage: batch.storage,
        budget,
        nbf,
        is_uks,
    };
    if defer {
        if let Some(n) = nlc_bytes {
            nlc_precheck(n, budget, nbf, grid.len())?;
        }
        return Ok(GridBuild {
            grid,
            screened,
            charge: Reservation::inert(RECOMPUTE_LABEL),
            nlc_grid,
            nlc_chi: None,
            nlc_dchi: None,
            pending: Some(pending),
        });
    }
    let (charge, nlc_chi, nlc_dchi) =
        pending.settle(&mut screened, &grid, nlc_grid.as_deref(), (mol, bs))?;
    Ok(GridBuild {
        grid,
        screened,
        charge,
        nlc_grid,
        nlc_chi,
        nlc_dchi,
        pending: None,
    })
}

/// RMS change of a spin density matrix between two consecutive Fock builds
/// (`sqrt(Σ ΔD²/nbf²)`, AO basis) above which the later build may form V_xc
/// with the device's mixed-precision `xc-vxc` kernel (any one spin moving by
/// more than this is enough).
///
/// Why a ladder. V_xc in mixed precision perturbs only the Fock matrix; the
/// energy is variational in the density, so the converged energy moves at
/// second order. But the f32 panel sums are rounding NOISE that changes with
/// the density, and an SCF iterated to a tight `density_conv` stalls on it:
/// with every build mixed (host twin of the kernel, benzene/cc-pVDZ PBE) the
/// SCF took 77 instead of 12 iterations at `density_conv = 1e-9` and did not
/// converge in 200 at 1e-10. (Rounding only the AO values to f32, a fixed
/// perturbation, costs nothing; rounding only `A` costs 2-6 iterations; the
/// f32 panel accumulation is the rest.) So mixed precision runs only while
/// the density still moves by more than this. The ladder alone does not
/// decide what a converged SCF returns: see `MixedLadder`.
///
/// Chosen by measurement (host twin, benzene and water⁺ PBE, water r2SCAN,
/// cc-pVDZ, `density_conv` 1e-6 / 1e-9 / 1e-10): τ = 1e-3, 1e-4, 1e-5 ran 3-4,
/// 4-5 and 6 of the ~10 builds mixed, cost at most 3, 3 and 2 extra
/// iterations, and moved the converged energy by at most 6e-13 Eh on those
/// systems (9.1e-13 Eh over the validation-page table, which adds n-octane).
/// 1e-5 keeps the most builds mixed for the fewest extra iterations.
pub const MIXED_VXC_LADDER_THRESH: f64 = 1e-5;

pub use crate::xc_batch::MIXED_VXC_BUILDS;

/// Per-evaluator memory of the last Fock build, deciding whether the next
/// build may use the mixed `xc-vxc` kernel. A build is mixed only when mixed
/// precision was requested (`precision = "mixed"` with `xc-vxc` allowed, and a
/// ready device), the functional is not a meta-GGA (see `mixed_vxc_allowed`),
/// a previous build of the same SCF exists, and some spin density moved by
/// more than [`MIXED_VXC_LADDER_THRESH`] since it. The SCF solvers reset the
/// ladder when they start (`start_scf`), so the first build of every SCF and
/// any one-shot build are f64.
///
/// The ladder does not by itself make a converged exit f64: the exit gate
/// (ΔP_rms of the total density below `density_conv`, or the UKS
/// degenerate-swap acceptance, which accepts a large ΔP) is a different test.
/// So a mixed build keeps its densities and V_xc here, and the solvers call
/// `refine_last_to_f64` at a converged exit, which swaps that V_xc for an f64
/// rebuild before the Fock matrix is diagonalised.
///
/// The step/reset/keep/take logic is the shared
/// [`ferric_core::precision_ladder::PrecisionLadder`] (extracted from here
/// without a change of arithmetic; the RI-J/DF-K ladder of `ferric_scf::jk_ladder`
/// uses the same type).
#[derive(Debug)]
struct MixedLadder(PrecisionLadder<MixedBuild>);

impl Default for MixedLadder {
    fn default() -> Self {
        Self(PrecisionLadder::new(LadderSite::XcVxc))
    }
}

/// A build the ladder ran mixed: its dense spin densities and the V_xc it added.
#[derive(Debug)]
struct MixedBuild {
    ds: Vec<Array2<f64>>,
    vs: Vec<Array2<f64>>,
}

impl MixedLadder {
    /// Record `ds` and say whether this build may use the mixed kernel (the
    /// shared [`PrecisionLadder::step_with`] with this kernel's request and
    /// threshold).
    fn step(&self, ds: &[&Array2<f64>]) -> bool {
        let requested = mixed_vxc_requested();
        if requested {
            ferric_core::gpu::stats::note_ladder_build(ferric_core::gpu::MixedKernel::XcVxc);
        }
        self.0.step_with(requested, MIXED_VXC_LADDER_THRESH, ds)
    }

    /// Forget every earlier build: the next one has no previous build.
    fn reset(&self) {
        self.0.reset();
    }

    /// Keep the mixed build just made (dense densities `ds`, potentials `vs`).
    fn keep_mixed(&self, ds: &[&Array2<f64>], vs: &[&Array2<f64>]) {
        self.0.keep_mixed(MixedBuild {
            ds: ds.iter().map(|d| (*d).clone()).collect(),
            vs: vs.iter().map(|v| (*v).clone()).collect(),
        });
    }

    /// The last build, if it was mixed (and forget it).
    fn take_mixed(&self) -> Option<MixedBuild> {
        self.0.take_mixed()
    }
}

/// Replace each mixed `vs[i]` in `fs[i]` by the f64 `v64s[i]`.
fn swap_in_f64(fs: &mut [&mut Array2<f64>], vs: &[Array2<f64>], v64s: &[&Array2<f64>]) {
    for ((f, v), v64) in fs.iter_mut().zip(vs).zip(v64s) {
        **f -= v;
        **f += *v64;
    }
}

/// The `expect` message of every XC integration inside a Fock build.
const EVAL_INFALLIBLE: &str = "per-shell AO evaluation failed inside a Fock build, but the \
     evaluator's constructor already evaluated every shell once (ScreenedGrid::build) and the \
     only per-shell error (UnsupportedL) depends on l alone, so the constructor would have \
     returned it";

/// Whether this Fock build may form V_xc with the mixed `xc-vxc` kernel: never
/// for a meta-GGA (its potential carries the τ blocks, and meta-GGA τ is on the
/// never-f32 list of `ferric_core::gpu::precision`), otherwise the ladder's
/// answer. A meta-GGA does not step the ladder at all.
fn mixed_vxc_allowed(xc: &XcDef, ladder: &MixedLadder, ds: &[&Array2<f64>]) -> bool {
    let mgga = xc
        .funcs
        .iter()
        .any(|f| matches!(f.family(), FunctionalFamily::MetaGga));
    !mgga && ladder.step(ds)
}

/// Closed-shell Kohn-Sham XC evaluator: Becke-Lebedev grid + screened batches
/// (see [`crate::xc_batch`]) + optional VV10 NLC grid and its dense AO cache.
#[derive(Debug)]
pub struct KsXc {
    pub xc: XcDef,
    pub grid: Vec<GridPoint>,
    /// Batched, screened main grid (and its resident AO blocks, if any).
    screened: ScreenedGrid,
    /// Pool charge for what this evaluator holds (see `decide_storage`: the
    /// resident AO blocks if any, the per-build working set, the NLC cache),
    /// held for exactly its lifetime. Declared AFTER `screened` so the arrays
    /// drop first. Inert when no pool is installed (which keeps the
    /// unbudgeted path identical).
    _charge: Reservation,
    pub nlc_grid: Option<Vec<GridPoint>>,
    pub nlc_chi: Option<Array2<f64>>,
    pub nlc_dchi: Option<Array3<f64>>,
    /// VV10's scratch, sized for the NLC grid. (`add_xc` takes `&self`, hence
    /// the Mutex; uncontended — one lock per Fock build.)
    nlc_scratch: Mutex<VxcScratch>,
    /// When a Fock build may form V_xc in mixed precision (see [`MixedLadder`]).
    ladder: MixedLadder,
    /// Set until `commit_storage` on an evaluator built with `new_deferred`.
    pending: Option<PendingStorage>,
}

/// The working budget the grid cache sizes itself against.
///
/// # Scope under the screened-batch design
///
/// Batch boundaries are now a pure function of the grid and `max_batch_pts`,
/// and this budget only chooses resident vs recompute AO storage, which are
/// bit-identical. So the energy can no longer depend on this figure at all;
/// the history below is kept because the resolve-once / pool-ledger choice
/// still makes the MEMORY behaviour reproducible, which is worth keeping.
///
/// # Why this is not `available_budget_now`
///
/// It used to be, and that made the SCF ENERGY depend on transient resident
/// memory. `available_budget_now` subtracts this process's LIVE RSS, and the
/// figure feeds `resolve_batch_size`, which sets the grid batch boundaries.
/// Batch boundaries fix the floating-point accumulation order of the
/// V_xc/E_xc sums, so a drifting budget silently moves the energy.
/// `resolve_batch_size`'s own doc already makes this argument about thread
/// counts ("NEVER of thread count -- so batch boundaries ... are identical no
/// matter how many rayon workers are configured"); RSS is strictly worse,
/// because it is not even a configuration.
///
/// MEASURED, water/cc-pVDZ/PBE at a 50 MB budget:
///
/// ```text
///   RSS   8.5 MB -> Full cache   (available_budget_now = 36 MB)
///   RSS  351   MB -> batched, 1 point per batch (available_budget_now = 0)
/// ```
///
/// Nothing but resident memory changed, and the code path flipped. On the
/// 27-atom terpinyl cation (6-31G/PBE/RI-JK, budget_gb = 0.30) two runs of
/// the same binary on the same input gave -390.3794192830 Ha (20 iters) and
/// -390.3794263686 Ha (35 iters) -- a 7.1e-6 Ha spread, the same order as the
/// 2.9e-6 Ha benzene/aTZ thread-count perturbation this codebase already
/// treats as a bug.
///
/// # What it is instead
///
/// When a pool is installed, what is ALREADY SPENT is exactly what the pool's
/// ledger says -- a deterministic number that depends only on which planes
/// this job has reserved, not on allocator behaviour or what ran earlier in
/// the process. That is the composition figure the RSS subtraction was
/// reaching for, without the nondeterminism: the DF tensor's reservation is
/// visible here precisely because it is still held.
///
/// With no pool installed there is nothing better available, so the historical
/// `available_budget_now` reading is kept -- that path is unchanged, which is
/// what keeps the unbudgeted limit bit-identical.
fn grid_working_budget(memory_budget_bytes: Option<usize>) -> usize {
    let ceiling = ferric_core::memory::resolve_budget_bytes(memory_budget_bytes);
    match ferric_core::memory::pool::global() {
        // Deterministic: capacity minus what THIS job has reserved.
        Some(pool) => pool.available_bytes().min(ceiling),
        // Unbudgeted: unchanged historical behaviour.
        None => ferric_core::memory::available_budget_now(ceiling),
    }
}

impl KsXc {
    /// `Some(max points per batch)` when the screened AO blocks are
    /// recomputed per Fock build, `None` when they are resident.
    ///
    /// Test hook. Batch boundaries no longer depend on the budget at all (they
    /// are a pure function of the grid and `max_batch_pts`), so this now
    /// reports the storage MODE the budget selected; the two modes are
    /// bit-identical.
    #[doc(hidden)]
    pub fn batch_pts_for_test(&self) -> Option<usize> {
        if self.screened.is_resident() {
            None
        } else {
            Some(self.screened.max_batch_pts())
        }
    }

    /// Batching/screening summary (batches, active fraction, residency).
    #[doc(hidden)]
    pub fn xc_screening_stats(&self) -> XcScreeningStats {
        self.screened.stats()
    }

    /// `(nact, nb)` of every screened batch of the main grid, in batch order
    /// (the raw screening data; tests derive expected launch counts from it).
    #[doc(hidden)]
    pub fn xc_batch_shapes(&self) -> Vec<(usize, usize)> {
        self.screened.batch_shapes()
    }

    /// True when the main-grid XC GEMMs run on the device (feature `gpu`: the
    /// AO blocks were uploaded by an earlier Fock build and no failure has
    /// declined the grid since). Always false without the feature.
    #[doc(hidden)]
    pub fn xc_on_device(&self) -> bool {
        self.screened.on_device()
    }

    /// Page-locked host staging bytes the device XC waves hold (feature `gpu`;
    /// 0 off the device or when the driver refused page-locked memory).
    #[doc(hidden)]
    pub fn xc_pinned_staging_bytes(&self) -> usize {
        self.screened.pinned_staging_bytes()
    }

    /// Build a closed-shell XC evaluator: resolve the functional, construct the Becke-Lebedev grid, and screen it.
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
        Self::new_with_omega_budgeted(mol, bs, xc_name, main, nlc, omega, None)
    }

    /// [`Self::new_with_omega`] plus an explicit memory budget in bytes.
    ///
    /// # Why this exists
    ///
    /// The grid AO cache is the largest single allocation in a DFT job, and
    /// the constructors resolved it with `resolve_budget_bytes(None)` — the
    /// env / cgroup / RAM-auto-detected ceiling — which silently DISCARDS a
    /// user's `[memory] budget_gb`. Setting `budget_gb = 4` on a 64 GB box
    /// still sized the grid cache against ~51 GB. The guard message even
    /// tells the user to raise `FERRIC_MEM_BUDGET_GB`, which works, while the
    /// documented primary knob did not.
    ///
    /// `None` preserves the previous behaviour exactly (resolve from env /
    /// auto-detect); `Some(bytes)` is an explicit ceiling that takes
    /// precedence, per `ferric_core::memory::resolve_budget`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_omega_budgeted(
        mol: &Molecule,
        bs: &BasisSet,
        xc_name: &str,
        main: &AtomicGridConfig,
        nlc: &AtomicGridConfig,
        omega: Option<f64>,
        memory_budget_bytes: Option<usize>,
    ) -> Result<Self, KsXcError> {
        Self::new_with_batch_config(
            mol,
            bs,
            xc_name,
            main,
            nlc,
            omega,
            memory_budget_bytes,
            XcBatchConfig::default(),
        )
    }

    /// [`Self::new_with_omega_budgeted`] with explicit batching/screening
    /// knobs. Internal: production always uses `XcBatchConfig::default()`;
    /// tests use `screen_thresh = 0.0` (the exactness anchor) and forced
    /// storage modes.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_batch_config(
        mol: &Molecule,
        bs: &BasisSet,
        xc_name: &str,
        main: &AtomicGridConfig,
        nlc: &AtomicGridConfig,
        omega: Option<f64>,
        memory_budget_bytes: Option<usize>,
        batch: XcBatchConfig,
    ) -> Result<Self, KsXcError> {
        Self::construct(
            (mol, bs, xc_name),
            (main, nlc),
            omega,
            memory_budget_bytes,
            batch,
            false,
        )
    }

    /// [`Self::new_with_omega_budgeted`] that takes no memory yet. Call
    /// [`XcContribution::commit_storage`] before the first `add_xc`; see
    /// `PendingStorage` for why the SCF drivers want this.
    pub fn new_deferred(
        mol: &Molecule,
        bs: &BasisSet,
        xc_name: &str,
        main: &AtomicGridConfig,
        nlc: &AtomicGridConfig,
        omega: Option<f64>,
        memory_budget_bytes: Option<usize>,
    ) -> Result<Self, KsXcError> {
        Self::construct(
            (mol, bs, xc_name),
            (main, nlc),
            omega,
            memory_budget_bytes,
            XcBatchConfig::default(),
            true,
        )
    }

    fn construct(
        (mol, bs, xc_name): (&Molecule, &BasisSet, &str),
        (main, nlc): (&AtomicGridConfig, &AtomicGridConfig),
        omega: Option<f64>,
        memory_budget_bytes: Option<usize>,
        batch: XcBatchConfig,
        defer: bool,
    ) -> Result<Self, KsXcError> {
        let xc = match omega {
            None => xc_def_from_name(xc_name)?,
            Some(w) => crate::libxc::xc_def_from_name_nspin_omega(xc_name, 1, w)?,
        };
        let b = build_grids(
            mol,
            bs,
            &xc,
            main,
            nlc,
            memory_budget_bytes,
            false,
            &batch,
            defer,
        )?;
        Ok(Self {
            xc,
            grid: b.grid,
            screened: b.screened,
            _charge: b.charge,
            nlc_grid: b.nlc_grid,
            nlc_chi: b.nlc_chi,
            nlc_dchi: b.nlc_dchi,
            nlc_scratch: Mutex::new(VxcScratch::new()),
            ladder: MixedLadder::default(),
            pending: b.pending,
        })
    }
}

/// Largest `|D − scale·C·Cᵀ|` entry allowed before a caller-supplied occupied
/// block is judged stale and the dense-`D` path is used instead. The SCF forms
/// `D` from the same `C` with one GEMM, so a consistent pair differs by a few
/// ulp of `max|D|`; `1e-12` relative is orders above that and orders below any
/// real density change.
const OCC_CONSISTENCY_TOL: f64 = 1e-12;

/// `c_occ` as a factored input for `d = scale·c_occ·c_occᵀ`, or the dense
/// input when `c_occ` is absent or does not reproduce `d` (stale cache,
/// fractional occupations): the dense path is always correct, the factored one
/// only for an exact `C·Cᵀ` density.
fn density_input<'a>(
    d: &'a Array2<f64>,
    c_occ: Option<&'a Array2<f64>>,
    scale: f64,
) -> DensityInput<'a> {
    let Some(c) = c_occ else {
        return DensityInput::Dense(d);
    };
    if c.nrows() != d.nrows() || d.nrows() != d.ncols() {
        return DensityInput::Dense(d);
    }
    let mut dc = c.dot(&c.t());
    dc *= scale;
    let dmax = d.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let err = (&dc - d).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if err <= OCC_CONSISTENCY_TOL * dmax.max(1.0) {
        DensityInput::Factored { c_occ: c, scale }
    } else {
        DensityInput::Dense(d)
    }
}

impl KsXc {
    /// Shared body of [`XcContribution::add_xc`] and `add_xc_occ`: semilocal
    /// XC from `input`, VV10 (if any) from the dense `d`.
    fn add_xc_input(&self, d: &Array2<f64>, input: DensityInput, f: &mut Array2<f64>) -> f64 {
        assert!(
            self.pending.is_none(),
            "commit_storage must run before the first add_xc on a deferred evaluator"
        );
        let mixed = mixed_vxc_allowed(&self.xc, &self.ladder, &[d]);
        let (e_xc, vxc) = self
            .screened
            .integrate_closed_exec(
                &self.grid,
                input,
                &self.xc,
                Exec::auto().with_mixed_vxc(mixed),
            )
            .expect(EVAL_INFALLIBLE);
        *f += &vxc;
        if mixed {
            self.ladder.keep_mixed(&[d], &[&vxc]);
        }

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
            // Recover from poisoning rather than panicking on it: the scratch
            // is reusable buffer space, fully overwritten before use.
            let mut nlc_scratch = self.nlc_scratch.lock().unwrap_or_else(|e| e.into_inner());
            add_vv10_scratch(g, c, dc, &nlc_dens, params, f, &mut nlc_scratch)
        } else {
            0.0
        };

        e_xc + e_nl
    }
}

impl XcContribution for KsXc {
    fn add_xc(&self, d: &Array2<f64>, f: &mut Array2<f64>) -> f64 {
        self.add_xc_input(d, DensityInput::Dense(d), f)
    }

    fn add_xc_occ(&self, d: &Array2<f64>, c_occ: &Array2<f64>, f: &mut Array2<f64>) -> f64 {
        self.add_xc_input(d, density_input(d, Some(c_occ), 2.0), f)
    }

    fn start_scf(&self) {
        self.ladder.reset();
    }

    fn commit_storage(&mut self, mol: &Molecule, bs: &BasisSet) -> Result<(), String> {
        let Some(p) = self.pending.take() else {
            return Ok(());
        };
        let (charge, chi, dchi) = p
            .settle(
                &mut self.screened,
                &self.grid,
                self.nlc_grid.as_deref(),
                (mol, bs),
            )
            .map_err(|e| e.to_string())?;
        self._charge = charge;
        self.nlc_chi = chi;
        self.nlc_dchi = dchi;
        Ok(())
    }

    fn refine_last_to_f64(&self, f: &mut Array2<f64>) -> bool {
        let Some(b) = self.ladder.take_mixed() else {
            return false;
        };
        let (_, v64) = self
            .screened
            .integrate_closed_exec(
                &self.grid,
                DensityInput::Dense(&b.ds[0]),
                &self.xc,
                Exec::auto(),
            )
            .expect(EVAL_INFALLIBLE);
        swap_in_f64(&mut [f], &b.vs, &[&v64]);
        ferric_core::gpu::stats::note_exit_rebuild(ferric_core::gpu::MixedKernel::XcVxc);
        true
    }

    fn k_mix(&self) -> KMix {
        if let Some(cam) = self.xc.cam {
            return KMix {
                sr: cam.c_sr,
                lr: cam.c_lr,
                omega: cam.omega,
            };
        }
        if let Some(mix) = self.xc.b3lyp_mix {
            return KMix {
                sr: mix,
                lr: mix,
                omega: 0.0,
            };
        }
        // Pure functional (LDA, PBE): no exact exchange
        KMix {
            sr: 0.0,
            lr: 0.0,
            omega: 0.0,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Spin-polarized counterpart (UKS / ROKS)
// ────────────────────────────────────────────────────────────────────────────

/// UKS analog of `KsXc`. Same grids and screened batches, but the `XcDef` is
/// built with `nspin=2` so libxc returns spin-resolved v_ρ / v_σ. VV10 is
/// closed-shell-friendly (function of total ρ and |∇ρ|²); the V_nl piece is
/// added equally to V_α and V_β.
#[derive(Debug)]
pub struct KsXcUks {
    pub xc: XcDef,
    pub grid: Vec<GridPoint>,
    /// See `KsXc::screened`.
    screened: ScreenedGrid,
    /// See `KsXc::_charge`.
    _charge: Reservation,
    pub nlc_grid: Option<Vec<GridPoint>>,
    pub nlc_chi: Option<Array2<f64>>,
    pub nlc_dchi: Option<Array3<f64>>,
    /// See `KsXc::nlc_scratch`.
    nlc_scratch: Mutex<VxcScratch>,
    /// When a Fock build may form V_xc in mixed precision (see [`MixedLadder`]).
    ladder: MixedLadder,
    /// Set until `commit_storage` on an evaluator built with `new_deferred`.
    pending: Option<PendingStorage>,
}

impl KsXcUks {
    /// See [`KsXc::batch_pts_for_test`].
    #[doc(hidden)]
    pub fn batch_pts_for_test(&self) -> Option<usize> {
        if self.screened.is_resident() {
            None
        } else {
            Some(self.screened.max_batch_pts())
        }
    }

    /// See [`KsXc::xc_screening_stats`].
    #[doc(hidden)]
    pub fn xc_screening_stats(&self) -> XcScreeningStats {
        self.screened.stats()
    }

    /// True when the main-grid XC GEMMs run on the device (feature `gpu`: the
    /// AO blocks were uploaded by an earlier Fock build and no failure has
    /// declined the grid since). Always false without the feature.
    #[doc(hidden)]
    pub fn xc_on_device(&self) -> bool {
        self.screened.on_device()
    }

    /// Page-locked host staging bytes the device XC waves hold (feature `gpu`;
    /// 0 off the device or when the driver refused page-locked memory).
    #[doc(hidden)]
    pub fn xc_pinned_staging_bytes(&self) -> usize {
        self.screened.pinned_staging_bytes()
    }

    /// Build an open-shell (UKS) XC evaluator with spin-resolved libxc response.
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
        Self::new_with_omega_budgeted(mol, bs, xc_name, main, nlc, omega, None)
    }

    /// UKS twin of [`KsXc::new_with_omega_budgeted`] (see its doc for why the
    /// explicit budget exists).
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_omega_budgeted(
        mol: &Molecule,
        bs: &BasisSet,
        xc_name: &str,
        main: &AtomicGridConfig,
        nlc: &AtomicGridConfig,
        omega: Option<f64>,
        memory_budget_bytes: Option<usize>,
    ) -> Result<Self, KsXcError> {
        Self::new_with_batch_config(
            mol,
            bs,
            xc_name,
            main,
            nlc,
            omega,
            memory_budget_bytes,
            XcBatchConfig::default(),
        )
    }

    /// UKS twin of [`KsXc::new_with_batch_config`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_batch_config(
        mol: &Molecule,
        bs: &BasisSet,
        xc_name: &str,
        main: &AtomicGridConfig,
        nlc: &AtomicGridConfig,
        omega: Option<f64>,
        memory_budget_bytes: Option<usize>,
        batch: XcBatchConfig,
    ) -> Result<Self, KsXcError> {
        Self::construct(
            (mol, bs, xc_name),
            (main, nlc),
            omega,
            memory_budget_bytes,
            batch,
            false,
        )
    }

    /// UKS twin of `KsXc::new_deferred`: [`Self::new_with_omega_budgeted`] that takes no memory yet. Call
    /// [`UksXcContribution::commit_storage`] before the first `add_xc`; see
    /// `PendingStorage` for why the SCF drivers want this.
    pub fn new_deferred(
        mol: &Molecule,
        bs: &BasisSet,
        xc_name: &str,
        main: &AtomicGridConfig,
        nlc: &AtomicGridConfig,
        omega: Option<f64>,
        memory_budget_bytes: Option<usize>,
    ) -> Result<Self, KsXcError> {
        Self::construct(
            (mol, bs, xc_name),
            (main, nlc),
            omega,
            memory_budget_bytes,
            XcBatchConfig::default(),
            true,
        )
    }

    fn construct(
        (mol, bs, xc_name): (&Molecule, &BasisSet, &str),
        (main, nlc): (&AtomicGridConfig, &AtomicGridConfig),
        omega: Option<f64>,
        memory_budget_bytes: Option<usize>,
        batch: XcBatchConfig,
        defer: bool,
    ) -> Result<Self, KsXcError> {
        let xc = match omega {
            None => xc_def_from_name_nspin(xc_name, 2)?,
            Some(w) => crate::libxc::xc_def_from_name_nspin_omega(xc_name, 2, w)?,
        };
        let b = build_grids(
            mol,
            bs,
            &xc,
            main,
            nlc,
            memory_budget_bytes,
            true,
            &batch,
            defer,
        )?;
        Ok(Self {
            xc,
            grid: b.grid,
            screened: b.screened,
            _charge: b.charge,
            nlc_grid: b.nlc_grid,
            nlc_chi: b.nlc_chi,
            nlc_dchi: b.nlc_dchi,
            nlc_scratch: Mutex::new(VxcScratch::new()),
            ladder: MixedLadder::default(),
            pending: b.pending,
        })
    }
}

impl KsXcUks {
    /// Shared body of `add_xc_uks` and `add_xc_uks_occ`: semilocal XC from the
    /// per-spin `inputs`, VV10 (if any) from the dense `d_a + d_b`.
    fn add_xc_uks_input(
        &self,
        (d_a, d_b): (&Array2<f64>, &Array2<f64>),
        (in_a, in_b): (DensityInput, DensityInput),
        f_a: &mut Array2<f64>,
        f_b: &mut Array2<f64>,
    ) -> f64 {
        assert!(
            self.pending.is_none(),
            "commit_storage must run before the first add_xc on a deferred evaluator"
        );
        // Semilocal: (ρ_α, ρ_β, σ_αα, σ_αβ, σ_ββ[, τ_α, τ_β]) on the screened
        // batches, then the polarized libxc kernel.
        let mixed = mixed_vxc_allowed(&self.xc, &self.ladder, &[d_a, d_b]);
        let (e_xc, vxc_a, vxc_b) = self
            .screened
            .integrate_polarized_exec(
                &self.grid,
                in_a,
                in_b,
                &self.xc,
                Exec::auto().with_mixed_vxc(mixed),
            )
            .expect(EVAL_INFALLIBLE);
        *f_a += &vxc_a;
        *f_b += &vxc_b;
        if mixed {
            self.ladder.keep_mixed(&[d_a, d_b], &[&vxc_a, &vxc_b]);
        }

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
            let mut nlc_scratch = self.nlc_scratch.lock().unwrap_or_else(|e| e.into_inner());
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
}

impl UksXcContribution for KsXcUks {
    fn add_xc_uks(
        &self,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
        f_a: &mut Array2<f64>,
        f_b: &mut Array2<f64>,
    ) -> f64 {
        let inputs = (DensityInput::Dense(d_a), DensityInput::Dense(d_b));
        self.add_xc_uks_input((d_a, d_b), inputs, f_a, f_b)
    }

    fn add_xc_uks_occ(
        &self,
        (d_a, d_b): (&Array2<f64>, &Array2<f64>),
        (c_a, c_b): (Option<&Array2<f64>>, Option<&Array2<f64>>),
        f_a: &mut Array2<f64>,
        f_b: &mut Array2<f64>,
    ) -> f64 {
        let inputs = (density_input(d_a, c_a, 1.0), density_input(d_b, c_b, 1.0));
        self.add_xc_uks_input((d_a, d_b), inputs, f_a, f_b)
    }

    fn start_scf(&self) {
        self.ladder.reset();
    }

    fn commit_storage(&mut self, mol: &Molecule, bs: &BasisSet) -> Result<(), String> {
        let Some(p) = self.pending.take() else {
            return Ok(());
        };
        let (charge, chi, dchi) = p
            .settle(
                &mut self.screened,
                &self.grid,
                self.nlc_grid.as_deref(),
                (mol, bs),
            )
            .map_err(|e| e.to_string())?;
        self._charge = charge;
        self.nlc_chi = chi;
        self.nlc_dchi = dchi;
        Ok(())
    }

    fn refine_last_to_f64_uks(&self, f_a: &mut Array2<f64>, f_b: &mut Array2<f64>) -> bool {
        let Some(b) = self.ladder.take_mixed() else {
            return false;
        };
        let (_, v_a, v_b) = self
            .screened
            .integrate_polarized_exec(
                &self.grid,
                DensityInput::Dense(&b.ds[0]),
                DensityInput::Dense(&b.ds[1]),
                &self.xc,
                Exec::auto(),
            )
            .expect(EVAL_INFALLIBLE);
        swap_in_f64(&mut [f_a, f_b], &b.vs, &[&v_a, &v_b]);
        ferric_core::gpu::stats::note_exit_rebuild(ferric_core::gpu::MixedKernel::XcVxc);
        true
    }

    fn k_mix(&self) -> KMix {
        if let Some(cam) = self.xc.cam {
            return KMix {
                sr: cam.c_sr,
                lr: cam.c_lr,
                omega: cam.omega,
            };
        }
        if let Some(mix) = self.xc.b3lyp_mix {
            return KMix {
                sr: mix,
                lr: mix,
                omega: 0.0,
            };
        }
        KMix {
            sr: 0.0,
            lr: 0.0,
            omega: 0.0,
        }
    }
}

#[cfg(test)]
mod storage_tests {
    use super::*;
    use ferric_core::basis;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;

    // FERRIC_MEM_BUDGET_GB is process-global; serialize any test that sets it.
    // Shared crate-wide lock (see lib.rs) — a module-local lock cannot stop
    // cross-module races on the process-global budget env var.
    use crate::TEST_BUDGET_ENV_LOCK as ENV_LOCK;
    const VAR: &str = ferric_core::memory::ENV_UNIFIED;

    fn clear_budget_env() {
        std::env::remove_var(VAR);
    }

    fn water() -> Molecule {
        Molecule::parse_xyz(
            "3\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.96\nH 0.93 0.0 -0.24\n",
            0,
            1,
        )
        .unwrap()
    }

    fn small_grids() -> (AtomicGridConfig, AtomicGridConfig) {
        (
            AtomicGridConfig {
                n_radial: 20,
                n_angular: 26,
                ..Default::default()
            },
            AtomicGridConfig {
                n_radial: 10,
                n_angular: 26,
                ..Default::default()
            },
        )
    }

    fn rhf_density(mol: &Molecule, bs: &BasisSet) -> Array2<f64> {
        let prep = PreparedBasis::new(mol, bs).unwrap();
        let op = Operator::coulomb();
        let bounds = ferric_scf::screening::SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        ferric_scf::rhf::solve_rhf(
            &ctx,
            mol,
            &prep,
            op,
            &bounds,
            &ferric_scf::rhf::RhfConfig::default(),
        )
        .unwrap()
        .density_total
    }

    // ── decide_storage: the budget decision in isolation (no pool installed in
    // these; the pool branches are covered by ferric-scf's
    // mwe_ksdft_planes_compose_in_one_pool.rs, which runs in its own process).

    /// Catches: a gate that refuses (Err) instead of falling back to
    /// recompute when the main grid is over budget — recompute is always
    /// available, so only VV10 may hard-fail — and a working set that is
    /// silently neither charged nor reported.
    #[test]
    fn over_budget_without_vv10_recomputes_instead_of_failing() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if ferric_core::memory::pool::global().is_some() {
            return; // a pool from another test in this binary; the pool path is tested elsewhere
        }
        let r = decide_storage(AoStorage::Auto, 1_000_000, 1_000, None, 10, 24, 1000).unwrap();
        assert!(!r.resident, "over budget must select recompute");
        // The 1000 B working set does not fit 10 B either: reported, not refused.
        assert_eq!(r.uncharged, 1_000);
        let r = decide_storage(AoStorage::Auto, 1_000_000, 5, None, 10, 24, 1000).unwrap();
        assert!(!r.resident);
        assert_eq!(r.uncharged, 0, "a working set that fits is charged");
    }

    /// Catches: an over-estimating gate that refuses residency for a job that
    /// fits (as much a bug as an under-estimating one), and an off-by-one at
    /// the boundary.
    #[test]
    fn an_ample_budget_is_resident() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if ferric_core::memory::pool::global().is_some() {
            return;
        }
        let big = usize::MAX / 2;
        let r = decide_storage(AoStorage::Auto, 1_000_000, 1_000, None, big, 24, 1000).unwrap();
        assert!(r.resident, "ample budget must be resident");
        // Exactly at the boundary still fits (<=).
        let r = decide_storage(AoStorage::Auto, 900, 100, None, 1000, 24, 1000).unwrap();
        assert!(r.resident);
        let r = decide_storage(AoStorage::Auto, 901, 100, None, 1000, 24, 1000).unwrap();
        assert!(!r.resident);
    }

    /// Catches: dropping the VV10 hard error (the NLC cache is not batchable),
    /// and — the converse — a VV10 error when only the MAIN grid is too big.
    #[test]
    fn vv10_fails_only_when_the_nlc_cache_itself_cannot_fit() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if ferric_core::memory::pool::global().is_some() {
            return;
        }
        let err = decide_storage(AoStorage::Auto, 10, 10, Some(5_000), 1_000, 24, 1000)
            .expect_err("NLC 5000 B over a 1000 B budget must fail");
        assert!(format!("{err}").contains("VV10"), "{err}");
        // Recompute mode enforces the same NLC bar.
        let r = decide_storage(AoStorage::Recompute, 10, 10, Some(5_000), 1_000, 24, 1000);
        assert!(r.is_err());
        // Main grid too big but NLC fits: recompute, no error.
        let r = decide_storage(AoStorage::Auto, 1_000_000, 10, Some(500), 1_000, 24, 1000);
        assert!(!r.unwrap().resident);
    }

    /// Catches: forced modes being overridden by the budget (the anchor and
    /// bit-identity tests rely on them).
    #[test]
    fn forced_modes_are_honoured() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if ferric_core::memory::pool::global().is_some() {
            return;
        }
        let r = decide_storage(AoStorage::Resident, usize::MAX / 4, 0, None, 1, 24, 1000);
        assert!(r.unwrap().resident);
        let r = decide_storage(AoStorage::Recompute, 1, 0, None, usize::MAX / 2, 24, 1000);
        assert!(!r.unwrap().resident);
    }

    /// Catches: a NaN threshold silently dropping every shell (E_xc = 0), and
    /// a zero batch cap.
    #[test]
    fn invalid_batch_config_is_rejected() {
        let mol = water();
        let bs = basis::bundled("sto-3g").unwrap();
        let (main, nlc) = small_grids();
        for cfg in [
            XcBatchConfig {
                screen_thresh: f64::NAN,
                ..Default::default()
            },
            XcBatchConfig {
                max_batch_pts: 0,
                ..Default::default()
            },
        ] {
            let e = KsXc::new_with_batch_config(&mol, &bs, "PBE", &main, &nlc, None, None, cfg)
                .expect_err("invalid config must error");
            assert!(matches!(e, KsXcError::BatchConfig(_)), "{e:?}");
        }
    }

    /// Resident and recompute modes are bit-identical, with the mode chosen by
    /// the BUDGET (not forced) — the successor of the old
    /// `batched_matches_full_cache_small_system_closed_shell` test, now with a
    /// stronger (bitwise) bar because both modes run the same arithmetic.
    ///
    /// Catches: any divergence between the resident and recompute code paths
    /// (e.g. a stale resident block, a different evaluation order), and a
    /// budget that fails to reach the storage decision.
    /// A deferred evaluator ends up as the eager one: same storage, same bits.
    #[test]
    fn deferred_storage_matches_eager_after_commit() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_budget_env();
        let mol = water();
        let bs = basis::bundled("sto-3g").unwrap();
        let d = rhf_density(&mol, &bs);
        let (main, nlc) = small_grids();
        let budget = Some(8_000_000_000);
        let eager =
            KsXc::new_with_omega_budgeted(&mol, &bs, "PBE", &main, &nlc, None, budget).unwrap();
        let mut late = KsXc::new_deferred(&mol, &bs, "PBE", &main, &nlc, None, budget).unwrap();
        late.commit_storage(&mol, &bs).unwrap();
        assert_eq!(late.batch_pts_for_test(), eager.batch_pts_for_test());
        let (mut fa, mut fb) = (Array2::zeros(d.raw_dim()), Array2::zeros(d.raw_dim()));
        let (ea, eb) = (eager.add_xc(&d, &mut fa), late.add_xc(&d, &mut fb));
        assert_eq!(ea.to_bits(), eb.to_bits());
        assert!(fa.iter().zip(&fb).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    /// VV10's cache is built at commit, so a missed commit must not silently
    /// lose V_nl.
    #[test]
    #[should_panic(expected = "commit_storage must run")]
    fn add_xc_before_commit_is_refused() {
        let mol = water();
        let bs = basis::bundled("sto-3g").unwrap();
        let (main, nlc) = small_grids();
        let late = KsXc::new_deferred(&mol, &bs, "PBE", &main, &nlc, None, None).unwrap();
        let n = nbasis(&mol, &bs).unwrap();
        let d = Array2::<f64>::zeros((n, n));
        late.add_xc(&d, &mut d.clone());
    }

    #[test]
    fn budget_selected_storage_modes_are_bit_identical_closed_shell() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_budget_env();
        let mol = water();
        let bs = basis::bundled("sto-3g").unwrap();
        let d = rhf_density(&mol, &bs);
        let (main, nlc) = small_grids();

        std::env::set_var(VAR, "1000"); // 1000 GiB — resident
        let ks_res = KsXc::new(&mol, &bs, "PBE", &main, &nlc).unwrap();
        std::env::set_var(VAR, "0.00001"); // ≈ 10.7 KB — recompute
        let ks_rec = KsXc::new(&mol, &bs, "PBE", &main, &nlc).unwrap();
        clear_budget_env();

        if ferric_core::memory::pool::global().is_none() {
            assert_eq!(
                ks_res.batch_pts_for_test(),
                None,
                "large budget must be resident"
            );
            assert!(
                ks_rec.batch_pts_for_test().is_some(),
                "tiny budget must recompute (and must NOT fail)"
            );
        }
        let mut f_res = Array2::<f64>::zeros(d.dim());
        let mut f_rec = Array2::<f64>::zeros(d.dim());
        let e_res = ks_res.add_xc(&d, &mut f_res);
        let e_rec = ks_rec.add_xc(&d, &mut f_rec);
        assert_eq!(e_res.to_bits(), e_rec.to_bits(), "{e_res:e} vs {e_rec:e}");
        for (a, b) in f_res.iter().zip(f_rec.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "{a:e} vs {b:e}");
        }
    }

    /// UKS twin of the above.
    #[test]
    fn budget_selected_storage_modes_are_bit_identical_uks() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_budget_env();
        // OH radical (doublet) — genuinely different α/β densities.
        let mol = Molecule::parse_xyz("2\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n", 0, 2).unwrap();
        let bs = basis::bundled("sto-3g").unwrap();
        let prep = PreparedBasis::new(&mol, &bs).unwrap();
        let op = Operator::coulomb();
        let bounds = ferric_scf::screening::SchwarzBounds::compute(op, &prep).unwrap();
        let ctx = ParallelContext::default();
        let uhf = ferric_scf::uhf::solve_uhf(
            &ctx,
            &mol,
            &prep,
            &bounds,
            &ferric_scf::uhf::UhfConfig::default(),
        )
        .unwrap();
        let d_a: Array2<f64> = uhf.density_alpha;
        let d_b: Array2<f64> = uhf.density_beta.unwrap();
        let (main, nlc) = small_grids();

        std::env::set_var(VAR, "1000");
        let ks_res = KsXcUks::new(&mol, &bs, "PBE", &main, &nlc).unwrap();
        std::env::set_var(VAR, "0.00001");
        let ks_rec = KsXcUks::new(&mol, &bs, "PBE", &main, &nlc).unwrap();
        clear_budget_env();
        if ferric_core::memory::pool::global().is_none() {
            assert_eq!(ks_res.batch_pts_for_test(), None);
            assert!(ks_rec.batch_pts_for_test().is_some());
        }

        let run = |ks: &KsXcUks| {
            let mut fa = Array2::<f64>::zeros(d_a.dim());
            let mut fb = Array2::<f64>::zeros(d_b.dim());
            let e = ks.add_xc_uks(&d_a, &d_b, &mut fa, &mut fb);
            (e, fa, fb)
        };
        let (e1, fa1, fb1) = run(&ks_res);
        let (e2, fa2, fb2) = run(&ks_rec);
        assert_eq!(e1.to_bits(), e2.to_bits());
        for (a, b) in fa1.iter().zip(fa2.iter()).chain(fb1.iter().zip(fb2.iter())) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    /// Regression for the old mid-run panic: construct under a tiny budget
    /// (recompute), then shrink the budget env var BEFORE `add_xc`. The Fock
    /// build must not consult the budget again, so it must not panic.
    #[test]
    fn recompute_add_xc_survives_budget_shrinking_after_construction() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_budget_env();
        let mol = water();
        let bs = basis::bundled("sto-3g").unwrap();
        let d = rhf_density(&mol, &bs);
        let (main, nlc) = small_grids();

        std::env::set_var(VAR, "0.00001");
        let ks = KsXc::new(&mol, &bs, "PBE", &main, &nlc).unwrap();
        std::env::set_var(VAR, "0.000001"); // smaller than at construction
        let mut f = Array2::<f64>::zeros(d.dim());
        let e_xc = ks.add_xc(&d, &mut f);
        clear_budget_env();

        assert!(e_xc.is_finite());
        assert!(f.iter().all(|x| x.is_finite()));
    }

    /// Directly exercises `eval_basis_and_grad_on_points_unchecked` (still a
    /// public ao_grid entry point): it must succeed even when the checked
    /// variant for the same (nbf, npts) fails under a tiny budget.
    #[test]
    fn eval_basis_and_grad_on_points_unchecked_ignores_a_failing_live_budget() {
        use crate::ao_grid::eval_basis_and_grad_on_points_unchecked;
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_budget_env();

        let mol = water();
        let bs = basis::bundled("sto-3g").unwrap();
        let shells = crate::ao_grid::collect_shells(&mol, &bs).unwrap();
        let nbf = crate::ao_grid::nbasis(&mol, &bs).unwrap();
        let pts: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]; 8];

        // nbf=7 · npts=8 · 4 planes · 8 B = 1792 B > ~1 KB.
        std::env::set_var(VAR, "0.000001");
        let checked = eval_basis_and_grad_on_points(&mol, &bs, &pts);
        assert!(
            checked.is_err(),
            "sanity: tiny budget must fail the checked path"
        );
        let unchecked = eval_basis_and_grad_on_points_unchecked(&shells, nbf, &pts);
        clear_budget_env();
        assert!(unchecked.is_ok());
    }
}

/// Exactness anchor, screening tolerance and thread-count bit-identity for the
/// batched/screened SCF XC path against the dense reference
/// (`eval_density_*` + `eval_tau_*` + `semilocal_vxc_*`, which remain public
/// and unchanged — they ARE the reference implementation, and the analytic
/// gradient / f_xc / VV10 paths still use them).
#[cfg(test)]
mod anchor_tests {
    use super::*;
    use crate::ao_grid::eval_basis_and_grad_on_points_unchecked;
    use crate::density_on_grid::{eval_density_uks, eval_tau_closed, eval_tau_uks};
    use crate::libxc::FunctionalFamily;
    use crate::vxc::{semilocal_vxc_closed, semilocal_vxc_polarized};
    use ferric_core::basis;
    use ferric_core::parallel::ParallelContext;
    use ferric_integrals::basis_bridge::PreparedBasis;
    use ferric_integrals::operator::Operator;

    // These tests read the budget (resident-vs-recompute) and must not race
    // tests that set it.
    use crate::TEST_BUDGET_ENV_LOCK as ENV_LOCK;

    const ANCHOR: XcBatchConfig = XcBatchConfig {
        screen_thresh: 0.0,
        max_batch_pts: crate::xc_batch::DEFAULT_MAX_BATCH_PTS,
        storage: AoStorage::Resident,
    };

    fn main_grid() -> AtomicGridConfig {
        AtomicGridConfig {
            n_radial: 40,
            n_angular: 110,
            ..Default::default()
        }
    }
    fn nlc_grid() -> AtomicGridConfig {
        AtomicGridConfig {
            n_radial: 10,
            n_angular: 26,
            ..Default::default()
        }
    }

    fn water() -> Molecule {
        Molecule::parse_xyz(
            "3\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.96\nH 0.93 0.0 -0.24\n",
            0,
            1,
        )
        .unwrap()
    }

    /// Two waters 9 Å apart: each molecule's batches cannot see the other
    /// molecule's functions at 1e-10, so screening genuinely removes work
    /// (a single water at cc-pVDZ keeps 100% at 1e-10 — measured — which
    /// would make the default-threshold test vacuous).
    fn far_water_dimer() -> Molecule {
        Molecule::parse_xyz(
            "6\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.96\nH 0.93 0.0 -0.24\n\
             O 9.0 0.0 0.0\nH 9.0 0.0 0.96\nH 9.93 0.0 -0.24\n",
            0,
            1,
        )
        .unwrap()
    }

    /// Water + OH radical 9 Å apart (doublet): the open-shell twin of
    /// `far_water_dimer`.
    fn water_oh_far() -> Molecule {
        Molecule::parse_xyz(
            "5\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.96\nH 0.93 0.0 -0.24\n\
             O 9.0 0.0 0.0\nH 9.0 0.0 0.97\n",
            0,
            2,
        )
        .unwrap()
    }

    fn rhf_density(mol: &Molecule, bs: &BasisSet) -> Array2<f64> {
        let prep = PreparedBasis::new(mol, bs).unwrap();
        let op = Operator::coulomb();
        let bounds = ferric_scf::screening::SchwarzBounds::compute(op, &prep).unwrap();
        ferric_scf::rhf::solve_rhf(
            &ParallelContext::default(),
            mol,
            &prep,
            op,
            &bounds,
            &ferric_scf::rhf::RhfConfig::default(),
        )
        .unwrap()
        .density_total
    }

    fn uhf_densities(mol: &Molecule, bs: &BasisSet) -> (Array2<f64>, Array2<f64>) {
        let prep = PreparedBasis::new(mol, bs).unwrap();
        let bounds =
            ferric_scf::screening::SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
        let uhf = ferric_scf::uhf::solve_uhf(
            &ParallelContext::default(),
            mol,
            &prep,
            &bounds,
            &ferric_scf::uhf::UhfConfig::default(),
        )
        .unwrap();
        (uhf.density_alpha, uhf.density_beta.unwrap())
    }

    fn is_mgga(xc: &XcDef) -> bool {
        xc.funcs
            .iter()
            .any(|f| matches!(f.family(), FunctionalFamily::MetaGga))
    }

    fn dense_chi(mol: &Molecule, bs: &BasisSet, grid: &[GridPoint]) -> (Array2<f64>, Array3<f64>) {
        let shells = collect_shells(mol, bs).unwrap();
        let nbf = nbasis(mol, bs).unwrap();
        let pts: Vec<[f64; 3]> = grid.iter().map(|g| g.xyz).collect();
        eval_basis_and_grad_on_points_unchecked(&shells, nbf, &pts).unwrap()
    }

    /// Dense closed-shell reference on the SAME grid and SAME XcDef.
    fn dense_closed(
        ks: &KsXc,
        mol: &Molecule,
        bs: &BasisSet,
        d: &Array2<f64>,
    ) -> (f64, Array2<f64>) {
        let (chi, dchi) = dense_chi(mol, bs, &ks.grid);
        let dens = eval_density_closed(d, &chi, &dchi);
        let tau = is_mgga(&ks.xc).then(|| eval_tau_closed(d, &dchi));
        semilocal_vxc_closed(&ks.grid, &chi, &dchi, &dens, tau.as_ref(), &ks.xc)
    }

    /// Dense polarized reference on the SAME grid and SAME XcDef.
    fn dense_uks(
        ks: &KsXcUks,
        mol: &Molecule,
        bs: &BasisSet,
        d_a: &Array2<f64>,
        d_b: &Array2<f64>,
    ) -> (f64, Array2<f64>, Array2<f64>) {
        let (chi, dchi) = dense_chi(mol, bs, &ks.grid);
        let dens = eval_density_uks(d_a, d_b, &chi, &dchi);
        let tau = is_mgga(&ks.xc).then(|| eval_tau_uks(d_a, d_b, &dchi));
        let tau_ref = tau.as_ref().map(|(a, b)| (a, b));
        semilocal_vxc_polarized(&ks.grid, &chi, &dchi, &dens, tau_ref, &ks.xc)
    }

    fn max_abs(a: &Array2<f64>) -> f64 {
        a.iter().fold(0.0_f64, |m, &x| m.max(x.abs()))
    }
    fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .fold(0.0_f64, |m, (&x, &y)| m.max((x - y).abs()))
    }

    /// Relative agreement bar for the trivial limit. Summation ORDER differs
    /// (batched GEMMs, grouped V reduction, sorted batches), so bit-identity
    /// is not expected; the prototype of this exact algorithm agreed with
    /// PySCF's dense integrator to 2e-15-5e-15 absolute. 1e-12 relative is
    /// ~100x above that noise and ~1e3x below any real defect (a dropped
    /// point, a wrong factor of 2 or a missing GGA axis are O(1e-3..1)).
    const ANCHOR_REL: f64 = 1e-12;

    /// EXACTNESS ANCHOR (closed shell). With screening off (`thresh = 0`) the
    /// batched path must reproduce the dense path for LDA, GGA, hybrid GGA and
    /// meta-GGA.
    ///
    /// Catches (construction errors, independent of any screening): a
    /// partition that drops/duplicates points or detaches weights; a wrong
    /// factor in the merged A·Bᵀ (½s, f_a, ½t — the symmetrization doubling);
    /// a missing or transposed GGA axis; τ not halved or τ blocks misaligned;
    /// the DENSITY_FLOOR gate applied to the wrong ρ; a scatter into the wrong
    /// (μ,ν) or a lost Vᵀ half; a wrong column block in the stored AO layout.
    #[test]
    fn batched_matches_dense_in_the_trivial_limit_closed_shell() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = water();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let d = rhf_density(&mol, &bs);
        for name in ["LDA", "PBE", "B3LYP", "SCAN"] {
            let ks = KsXc::new_with_batch_config(
                &mol,
                &bs,
                name,
                &main_grid(),
                &nlc_grid(),
                None,
                None,
                ANCHOR,
            )
            .unwrap();
            let st = ks.xc_screening_stats();
            assert_eq!(
                st.active_fraction, 1.0,
                "{name}: anchor must keep every function"
            );
            assert!(st.nbatches > 1, "{name}: must actually batch");
            let (e_ref, v_ref) = dense_closed(&ks, &mol, &bs, &d);
            let mut f = Array2::<f64>::zeros(d.dim());
            let e = ks.add_xc(&d, &mut f);
            let de = (e - e_ref).abs();
            let dv = max_abs_diff(&f, &v_ref);
            eprintln!("anchor closed {name}: E={e:.12} dE={de:.2e} max|dV|={dv:.2e}");
            assert!(
                de <= ANCHOR_REL * e_ref.abs(),
                "{name}: E {e:.15} vs {e_ref:.15}"
            );
            assert!(
                dv <= ANCHOR_REL * max_abs(&v_ref).max(1.0),
                "{name}: max|dV| {dv:e}"
            );
        }
    }

    /// EXACTNESS ANCHOR (open shell): the polarized path, including the σ_αβ
    /// cross-coupling and per-spin τ, against the dense polarized reference.
    ///
    /// Catches, beyond the closed-shell list: α/β swapped in the factors, the
    /// σ_αβ cross term with coefficient 2 instead of 1 or gated on the wrong
    /// spin, and per-spin τ fed the wrong density matrix.
    #[test]
    fn batched_matches_dense_in_the_trivial_limit_uks() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = Molecule::parse_xyz("2\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n", 0, 2).unwrap();
        let bs = basis::bundled("cc-pvdz").unwrap();
        let (d_a, d_b) = uhf_densities(&mol, &bs);
        for name in ["LDA", "PBE", "B3LYP", "SCAN"] {
            let ks = KsXcUks::new_with_batch_config(
                &mol,
                &bs,
                name,
                &main_grid(),
                &nlc_grid(),
                None,
                None,
                ANCHOR,
            )
            .unwrap();
            assert_eq!(ks.xc_screening_stats().active_fraction, 1.0);
            let (e_ref, va_ref, vb_ref) = dense_uks(&ks, &mol, &bs, &d_a, &d_b);
            let mut fa = Array2::<f64>::zeros(d_a.dim());
            let mut fb = Array2::<f64>::zeros(d_b.dim());
            let e = ks.add_xc_uks(&d_a, &d_b, &mut fa, &mut fb);
            let de = (e - e_ref).abs();
            let dva = max_abs_diff(&fa, &va_ref);
            let dvb = max_abs_diff(&fb, &vb_ref);
            eprintln!("anchor uks {name}: E={e:.12} dE={de:.2e} dVa={dva:.2e} dVb={dvb:.2e}");
            assert!(
                de <= ANCHOR_REL * e_ref.abs(),
                "{name}: E {e:.15} vs {e_ref:.15}"
            );
            assert!(
                dva <= ANCHOR_REL * max_abs(&va_ref).max(1.0),
                "{name}: dVa {dva:e}"
            );
            assert!(
                dvb <= ANCHOR_REL * max_abs(&vb_ref).max(1.0),
                "{name}: dVb {dvb:e}"
            );
            // α ≠ β here, so a swapped-spin defect cannot pass by symmetry.
            assert!(
                max_abs_diff(&va_ref, &vb_ref) > 1e-3,
                "{name}: fixture must be spin-polarized"
            );
        }
    }

    /// Derived tolerance bars for the DEFAULT screening threshold (1e-10), per
    /// basis, for PBE. MEASURED with a numpy/PySCF reimplementation of exactly
    /// this algorithm on exactly these fixtures (40x110 Becke grid, RHF/UHF
    /// densities, deltas vs the unscreened batched sum), |dE_xc| / max|dV|:
    ///
    /// ```text
    ///                     at 1e-10 (default)    at 1e-8 (mutant)      chi-only 1e-10
    ///   dimer   6-31G     1.1e-14 / 3.4e-14     1.4e-12 / 3.0e-12     1.5e-13 / 3.0e-13
    ///   W+OH    6-31G     7.1e-15 / 5.3e-14     1.7e-12 / 4.7e-12     1.3e-13 / 5.0e-13
    ///   dimer   cc-pVDZ   0       / 1.4e-13     0       / 1.2e-11     0       / 6.3e-13
    ///   W+OH    cc-pVDZ   0       / 6.3e-13     5.3e-14 / 2.9e-11     0       / 6.5e-13
    /// ```
    ///
    /// Bars sit between the default's measured error and the 1e-8 mutant's:
    /// 6-31G E 1e-13 (9x above measured, 14x below the mutant), V 5e-13 (9x
    /// above, 6x below); cc-pVDZ V 3e-12 (4.8x above, 4x below; E cannot
    /// discriminate there, so it only keeps the 1e-13 sanity bar). The
    /// dense-vs-batched summation-order noise these comparisons also carry
    /// is ~1e-15 (anchor tests). A χ-only screening criterion is NOT
    /// separable by any energy/V bar (see the chi-only column); it is caught
    /// by `xc_batch::tests::every_dropped_shell_is_below_threshold_and_every_kept_one_above`.
    /// The calibration used PySCF's radial grid, not ferric's TA-M4; if a bar
    /// fails with the DEFAULT threshold, compare the printed values against
    /// this table before loosening it.
    fn pbe_screen_bars(basis: &str) -> (f64, f64) {
        match basis {
            "6-31g" => (1e-13, 5e-13),
            "cc-pvdz" => (1e-13, 3e-12),
            other => panic!("no calibrated bars for {other}"),
        }
    }

    /// SCAN bars: coverage of the τ path under screening, NOT calibrated (the
    /// prototype was PBE only). 1e-10 Ha / 1e-9 is the acceptance bar.
    const SCAN_SCREEN_TOL_E: f64 = 1e-10;
    const SCAN_SCREEN_TOL_V: f64 = 1e-9;

    fn screen_bars(name: &str, basis: &str) -> (f64, f64) {
        if name == "PBE" {
            pbe_screen_bars(basis)
        } else {
            (SCAN_SCREEN_TOL_E, SCAN_SCREEN_TOL_V)
        }
    }

    /// Default screening vs dense, closed shell, 6-31G (s/p) and cc-pVDZ
    /// (pure d), on a fixture where screening genuinely removes work
    /// (asserted, so the test cannot pass vacuously).
    ///
    /// Catches: a default threshold loosened past its measurement (1e-8 fails
    /// the PBE V bars by 4-6x), and gross screening errors (dropping clearly
    /// significant shells, the wrong batch's points).
    #[test]
    fn default_screening_matches_dense_within_derived_tolerance_closed_shell() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = far_water_dimer();
        for basis_name in ["6-31g", "cc-pvdz"] {
            let bs = basis::bundled(basis_name).unwrap();
            let d = rhf_density(&mol, &bs);
            for name in ["PBE", "SCAN"] {
                let ks = KsXc::new_with_batch_config(
                    &mol,
                    &bs,
                    name,
                    &main_grid(),
                    &nlc_grid(),
                    None,
                    None,
                    XcBatchConfig::default(),
                )
                .unwrap();
                let st = ks.xc_screening_stats();
                eprintln!("screen closed {basis_name} {name}: {st:?}");
                assert!(
                    st.active_fraction < 0.9,
                    "{basis_name} {name}: fixture must exercise screening, active {}",
                    st.active_fraction
                );
                let (e_ref, v_ref) = dense_closed(&ks, &mol, &bs, &d);
                let mut f = Array2::<f64>::zeros(d.dim());
                let e = ks.add_xc(&d, &mut f);
                let de = (e - e_ref).abs();
                let dv = max_abs_diff(&f, &v_ref);
                let (tol_e, tol_v) = screen_bars(name, basis_name);
                eprintln!("screen closed {basis_name} {name}: dE={de:.2e} max|dV|={dv:.2e}");
                assert!(de <= tol_e, "{basis_name} {name}: dE {de:e} > {tol_e:e}");
                assert!(
                    dv <= tol_v,
                    "{basis_name} {name}: max|dV| {dv:e} > {tol_v:e}"
                );
            }
        }
    }

    /// Default screening vs dense, open shell (a far water / OH pair), 6-31G
    /// and cc-pVDZ. Same bars and catches as the closed-shell test.
    #[test]
    fn default_screening_matches_dense_within_derived_tolerance_uks() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = water_oh_far();
        for basis_name in ["6-31g", "cc-pvdz"] {
            let bs = basis::bundled(basis_name).unwrap();
            let (d_a, d_b) = uhf_densities(&mol, &bs);
            for name in ["PBE", "SCAN"] {
                let ks = KsXcUks::new_with_batch_config(
                    &mol,
                    &bs,
                    name,
                    &main_grid(),
                    &nlc_grid(),
                    None,
                    None,
                    XcBatchConfig::default(),
                )
                .unwrap();
                let st = ks.xc_screening_stats();
                assert!(
                    st.active_fraction < 0.9,
                    "{basis_name} {name}: active {}",
                    st.active_fraction
                );
                let (e_ref, va_ref, vb_ref) = dense_uks(&ks, &mol, &bs, &d_a, &d_b);
                let mut fa = Array2::<f64>::zeros(d_a.dim());
                let mut fb = Array2::<f64>::zeros(d_b.dim());
                let e = ks.add_xc_uks(&d_a, &d_b, &mut fa, &mut fb);
                let de = (e - e_ref).abs();
                let dv = max_abs_diff(&fa, &va_ref).max(max_abs_diff(&fb, &vb_ref));
                let (tol_e, tol_v) = screen_bars(name, basis_name);
                eprintln!("screen uks {basis_name} {name}: dE={de:.2e} max|dV|={dv:.2e}");
                assert!(de <= tol_e, "{basis_name} {name}: dE {de:e} > {tol_e:e}");
                assert!(
                    dv <= tol_v,
                    "{basis_name} {name}: max|dV| {dv:e} > {tol_v:e}"
                );
            }
        }
    }

    fn in_pool<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(f)
    }

    fn assert_bits(a: &Array2<f64>, b: &Array2<f64>, what: &str) {
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{what}[{i}]: {x:e} vs {y:e}");
        }
    }

    /// DETERMINISM (closed shell): bit-identical E_xc and V_xc across 1 vs 4
    /// rayon threads, in both storage modes, with screening on and a
    /// meta-GGA (every pass-1/pass-2 code path live), for 6-31G and cc-pVDZ
    /// (pure d). Also resident vs recompute bitwise.
    ///
    /// Catches: any per-thread accumulator, a group count or batch boundary
    /// that reads the thread pool, rayon `reduce`/`sum` over floats, BLAS
    /// raised inside a worker, and a divergence between storage modes.
    #[test]
    fn add_xc_is_bit_identical_across_thread_counts_and_storage_modes_closed_shell() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = far_water_dimer();
        for basis_name in ["6-31g", "cc-pvdz"] {
            let bs = basis::bundled(basis_name).unwrap();
            let d = rhf_density(&mol, &bs);
            let mut results = Vec::new();
            for storage in [AoStorage::Resident, AoStorage::Recompute] {
                let cfg = XcBatchConfig {
                    storage,
                    ..Default::default()
                };
                let ks = KsXc::new_with_batch_config(
                    &mol,
                    &bs,
                    "SCAN",
                    &main_grid(),
                    &nlc_grid(),
                    None,
                    None,
                    cfg,
                )
                .unwrap();
                assert!(ks.xc_screening_stats().active_fraction < 1.0);
                for threads in [1usize, 4] {
                    let (e, f) = in_pool(threads, || {
                        let mut f = Array2::<f64>::zeros(d.dim());
                        let e = ks.add_xc(&d, &mut f);
                        (e, f)
                    });
                    results.push((format!("{storage:?}/{threads}t"), e, f));
                }
            }
            let (n0, e0, f0) = &results[0];
            for (n, e, f) in &results[1..] {
                assert_eq!(
                    e0.to_bits(),
                    e.to_bits(),
                    "{basis_name} {n0} vs {n}: {e0:e} vs {e:e}"
                );
                assert_bits(f0, f, &format!("{basis_name} V {n0} vs {n}"));
            }
        }
    }

    /// A synthetic occupied block `(nbf, nocc)`: only the arithmetic path of
    /// the factored density pass is under test, not the physics.
    fn synth_c(nbf: usize, nocc: usize, salt: usize) -> Array2<f64> {
        Array2::from_shape_fn((nbf, nocc), |(i, j)| {
            ((i * 7 + j * 3 + salt) % 11) as f64 * 0.05 - 0.2
        })
    }

    /// The occupied-factored density pass agrees with the dense-D pass to a
    /// rounding-derived bound, and is bit-identical across 1 vs 4 rayon
    /// threads.
    ///
    /// Bound: the two forms evaluate `Φ = D·χ` as one length-`nact` dot per
    /// entry versus two (`nocc` then `nact`) with an exact power-of-two scale.
    /// Each carries a relative error of at most `γ_nact ≈ nact·u` against
    /// `Σ|D||χ|` (u = 1.1e-16, nact ≲ 100 here: ≲ 1.1e-14), so ρ, ∇ρ and τ
    /// differ by a few `γ` of their absolute sums, and `E_xc`, a weighted sum
    /// of smooth functions of those, by the same relative amount: with
    /// |E_xc| = O(10) Eh that is ≲ 1e-12 Eh. The asserted 1e-10 is 100× that
    /// bound. Meta-GGA exercises every ncols branch.
    ///
    /// Catches: a wrong scale (2 vs 1), a transposed C, a gather that drops
    /// the row selection, and a thread-count dependence in the new path.
    #[test]
    fn add_xc_occ_matches_dense_density_and_is_thread_invariant_closed_shell() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = far_water_dimer();
        for (basis_name, name) in [("6-31g", "PBE"), ("cc-pvdz", "SCAN")] {
            let bs = basis::bundled(basis_name).unwrap();
            let nbf = nbasis(&mol, &bs).unwrap();
            let c = synth_c(nbf, 5, 0);
            let d = 2.0 * c.dot(&c.t());
            let ks = KsXc::new(&mol, &bs, name, &main_grid(), &nlc_grid()).unwrap();
            let mut f_dense = Array2::<f64>::zeros(d.dim());
            let e_dense = ks.add_xc(&d, &mut f_dense);
            let mut results = Vec::new();
            for threads in [1usize, 4] {
                let (e, f) = in_pool(threads, || {
                    let mut f = Array2::<f64>::zeros(d.dim());
                    let e = ks.add_xc_occ(&d, &c, &mut f);
                    (e, f)
                });
                results.push((e, f));
            }
            let (e1, f1) = &results[0];
            let (e4, f4) = &results[1];
            assert_eq!(e1.to_bits(), e4.to_bits(), "{name} E 1t vs 4t");
            assert_bits(f1, f4, &format!("{basis_name} {name} V 1t vs 4t"));
            assert!(
                (e1 - e_dense).abs() <= 1e-10,
                "{basis_name} {name}: dE {:e}",
                (e1 - e_dense).abs()
            );
            let dv = max_abs_diff(f1, &f_dense);
            assert!(dv <= 1e-10, "{basis_name} {name}: max|dV| {dv:e}");
            // The factored path really ran: it is NOT the dense bits.
            assert!(
                f1.iter()
                    .zip(f_dense.iter())
                    .any(|(a, b)| a.to_bits() != b.to_bits()),
                "{basis_name} {name}: factored result identical to dense bits"
            );
        }
    }

    /// Real converged densities (water, butane; cc-pVDZ RHF orbitals): the
    /// factored pass reproduces the dense pass's E_xc and V_xc to 1e-10 for
    /// GGA, the same bound as the synthetic-C test (rounding of a length-nact
    /// dot, ~1e-14 relative, times |E_xc| ≲ 1e2 Eh). The SCF feeds V_xc into a
    /// variational energy, so the SCF energy moves at the same order.
    ///
    /// Catches: a factored pass that is only right for synthetic C (wrong
    /// MO ordering, wrong occupied count, scale mismatch on a real density).
    #[test]
    fn add_xc_occ_matches_dense_on_converged_water_and_butane() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/molecules");
        let bs = basis::bundled("cc-pvdz").unwrap();
        for file in ["water.xyz", "alkane_4.xyz"] {
            let mol = Molecule::load_xyz(root.join(file).to_str().unwrap()).unwrap();
            let prep = PreparedBasis::new(&mol, &bs).unwrap();
            let bounds =
                ferric_scf::screening::SchwarzBounds::compute(Operator::coulomb(), &prep).unwrap();
            let r = ferric_scf::rhf::solve_rhf(
                &ParallelContext::default(),
                &mol,
                &prep,
                Operator::coulomb(),
                &bounds,
                &ferric_scf::rhf::RhfConfig::default(),
            )
            .unwrap();
            let nocc = (mol.nelec() / 2) as usize;
            let c = r.mos_alpha.slice(ndarray::s![.., ..nocc]).to_owned();
            // The result's density can lag its final orbitals by the SCF
            // convergence error; rebuild D from C so the pair is consistent.
            let d = 2.0 * c.dot(&c.t());
            let d = &d;
            let ks = KsXc::new(&mol, &bs, "PBE", &main_grid(), &nlc_grid()).unwrap();
            let mut f_dense = Array2::<f64>::zeros(d.dim());
            let e_dense = ks.add_xc(d, &mut f_dense);
            let mut f_occ = Array2::<f64>::zeros(d.dim());
            let e_occ = ks.add_xc_occ(d, &c, &mut f_occ);
            let (de, dv) = ((e_occ - e_dense).abs(), max_abs_diff(&f_occ, &f_dense));
            eprintln!("{file}: E_xc {e_dense:.10} dE {de:.2e} max|dV| {dv:.2e}");
            assert!(de <= 1e-10, "{file}: dE {de:e}");
            assert!(dv <= 1e-10, "{file}: max|dV| {dv:e}");
            // The factored path really ran (not the dense fallback).
            assert!(
                dv > 0.0,
                "{file}: factored result is bit-identical to dense"
            );
        }
    }

    /// A stale or inconsistent occupied block must fall back to the dense `D`
    /// (bit-identical to `add_xc`), never be trusted.
    ///
    /// Catches: removing the consistency guard (a stale cache would silently
    /// integrate the wrong density).
    #[test]
    fn add_xc_occ_falls_back_to_dense_when_c_does_not_reproduce_d() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = far_water_dimer();
        let bs = basis::bundled("6-31g").unwrap();
        let nbf = nbasis(&mol, &bs).unwrap();
        let c_true = synth_c(nbf, 5, 0);
        let c_stale = synth_c(nbf, 5, 4);
        let d = 2.0 * c_true.dot(&c_true.t());
        let ks = KsXc::new(&mol, &bs, "PBE", &main_grid(), &nlc_grid()).unwrap();
        let mut f_dense = Array2::<f64>::zeros(d.dim());
        let e_dense = ks.add_xc(&d, &mut f_dense);
        let mut f_stale = Array2::<f64>::zeros(d.dim());
        let e_stale = ks.add_xc_occ(&d, &c_stale, &mut f_stale);
        assert_eq!(e_dense.to_bits(), e_stale.to_bits());
        assert_bits(&f_dense, &f_stale, "stale-C fallback V");
        // A wrong scale (D = C·Cᵀ passed as 2·C·Cᵀ) is also rejected.
        let d1 = c_true.dot(&c_true.t());
        let mut f1 = Array2::<f64>::zeros(d.dim());
        let mut f1_dense = Array2::<f64>::zeros(d.dim());
        let e1 = ks.add_xc_occ(&d1, &c_true, &mut f1);
        let e1_dense = ks.add_xc(&d1, &mut f1_dense);
        assert_eq!(e1.to_bits(), e1_dense.to_bits());
        assert_bits(&f1, &f1_dense, "wrong-scale fallback V");
    }

    /// Open-shell twin of the closed-shell factored test: per-spin
    /// `D_σ = C_σ·C_σᵀ` (scale 1), 1 vs 4 threads bit-identical, agreement
    /// with the dense polarized pass within the same bound, and a `None`
    /// block for one spin mixes the dense and factored forms.
    #[test]
    fn add_xc_uks_occ_matches_dense_density_and_is_thread_invariant() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = water_oh_far();
        let bs = basis::bundled("6-31g").unwrap();
        let nbf = nbasis(&mol, &bs).unwrap();
        let (c_a, c_b) = (synth_c(nbf, 5, 0), synth_c(nbf, 4, 2));
        let (d_a, d_b) = (c_a.dot(&c_a.t()), c_b.dot(&c_b.t()));
        for name in ["PBE", "SCAN"] {
            let ks = KsXcUks::new(&mol, &bs, name, &main_grid(), &nlc_grid()).unwrap();
            let (mut fa0, mut fb0) = (
                Array2::<f64>::zeros(d_a.dim()),
                Array2::<f64>::zeros(d_a.dim()),
            );
            let e0 = ks.add_xc_uks(&d_a, &d_b, &mut fa0, &mut fb0);
            let run = |threads: usize, cb: Option<&Array2<f64>>| {
                in_pool(threads, || {
                    let mut fa = Array2::<f64>::zeros(d_a.dim());
                    let mut fb = Array2::<f64>::zeros(d_a.dim());
                    let e = ks.add_xc_uks_occ((&d_a, &d_b), (Some(&c_a), cb), &mut fa, &mut fb);
                    (e, fa, fb)
                })
            };
            let (e1, fa1, fb1) = run(1, Some(&c_b));
            let (e4, fa4, fb4) = run(4, Some(&c_b));
            assert_eq!(e1.to_bits(), e4.to_bits(), "{name} uks E 1t vs 4t");
            assert_bits(&fa1, &fa4, &format!("{name} Va 1t vs 4t"));
            assert_bits(&fb1, &fb4, &format!("{name} Vb 1t vs 4t"));
            let (em, fam, fbm) = run(1, None);
            for (what, e, fa, fb) in [("both", e1, &fa1, &fb1), ("alpha-only", em, &fam, &fbm)] {
                assert!(
                    (e - e0).abs() <= 1e-10,
                    "{name} {what}: dE {:e}",
                    (e - e0).abs()
                );
                assert!(max_abs_diff(fa, &fa0) <= 1e-10, "{name} {what}: dVa");
                assert!(max_abs_diff(fb, &fb0) <= 1e-10, "{name} {what}: dVb");
            }
        }
    }

    /// DETERMINISM (open shell): same as above for the polarized path.
    #[test]
    fn add_xc_is_bit_identical_across_thread_counts_and_storage_modes_uks() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = water_oh_far();
        for basis_name in ["6-31g", "cc-pvdz"] {
            let bs = basis::bundled(basis_name).unwrap();
            let (d_a, d_b) = uhf_densities(&mol, &bs);
            let mut results = Vec::new();
            for storage in [AoStorage::Resident, AoStorage::Recompute] {
                let cfg = XcBatchConfig {
                    storage,
                    ..Default::default()
                };
                let ks = KsXcUks::new_with_batch_config(
                    &mol,
                    &bs,
                    "SCAN",
                    &main_grid(),
                    &nlc_grid(),
                    None,
                    None,
                    cfg,
                )
                .unwrap();
                for threads in [1usize, 4] {
                    let r = in_pool(threads, || {
                        let mut fa = Array2::<f64>::zeros(d_a.dim());
                        let mut fb = Array2::<f64>::zeros(d_b.dim());
                        let e = ks.add_xc_uks(&d_a, &d_b, &mut fa, &mut fb);
                        (e, fa, fb)
                    });
                    results.push((format!("{storage:?}/{threads}t"), r));
                }
            }
            let (n0, (e0, fa0, fb0)) = &results[0];
            for (n, (e, fa, fb)) in &results[1..] {
                assert_eq!(e0.to_bits(), e.to_bits(), "{basis_name} {n0} vs {n}");
                assert_bits(fa0, fa, &format!("{basis_name} Va {n0} vs {n}"));
                assert_bits(fb0, fb, &format!("{basis_name} Vb {n0} vs {n}"));
            }
        }
    }

    /// NEGATIVE CONTROL for the tolerance tests: an absurd threshold (1e-2)
    /// MUST break agreement with dense. If this ever passes within the
    /// default tolerances, the comparison is not seeing the screened sum
    /// (e.g. both sides silently use the same path) and the tolerance tests
    /// above prove nothing.
    #[test]
    fn an_absurd_threshold_visibly_breaks_agreement() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mol = far_water_dimer();
        let bs = basis::bundled("6-31g").unwrap();
        let d = rhf_density(&mol, &bs);
        let cfg = XcBatchConfig {
            screen_thresh: 1e-2,
            ..Default::default()
        };
        let ks = KsXc::new_with_batch_config(
            &mol,
            &bs,
            "PBE",
            &main_grid(),
            &nlc_grid(),
            None,
            None,
            cfg,
        )
        .unwrap();
        let (e_ref, _) = dense_closed(&ks, &mol, &bs, &d);
        let mut f = Array2::<f64>::zeros(d.dim());
        let e = ks.add_xc(&d, &mut f);
        assert!(
            (e - e_ref).abs() > 1e-7, // 1e3x the loosest (SCAN) bar
            "thresh 1e-2 should visibly perturb E_xc: {e} vs {e_ref}"
        );
    }
}
