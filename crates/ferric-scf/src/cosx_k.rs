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
//! The per-point `A^g` comes from `ferric_integrals::cosx_a` (libint2
//! nuclear-attraction engine with a unit probe charge, sign already flipped to
//! the repulsive `+1/|r-r_g|` COSX needs). That A-build is the dominant cost;
//! this module is the harness a faster 3c1e kernel drops into.
//!
//! # Grid, blocking, determinism
//!
//! The builder owns its OWN Becke–Lebedev grid (`CosxConfig::grid`, default
//! (50,110) — the measured operating point: the composed-budget audit on this
//! branch found grids coarser than (50,110) fail the 0.1 kcal/mol isodesmic
//! reaction-energy bar, and at (50,110) the overlap fit took that error from
//! 0.2068 to 0.0190 kcal/mol, so the fit defaults ON). Points are processed
//! in fixed blocks of `COSX_BLOCK_POINTS` points; per block the three `(nbf, B)`
//! planes `X`, `F`, `G` are resident, and the per-point `A^g` loop is
//! parallel over points with one libint2 engine per rayon worker. Every
//! parallel write is to its own column of `G`, and the two GEMMs per block
//! run in block order, so the result is bit-identical across thread counts
//! (the block partition is a pure function of the point count).
//!
//! # Scope
//!
//! Closed-shell RHF only via `k_builder = "cosx"` (like LinK: `solve_uhf` /
//! `solve_rohf` never read `k_builder`). Single MPI rank only — a multi-rank
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
use ndarray::Array2;
use ndarray_linalg::{Cholesky, Diag, SolveTriangular, UPLO};
use rayon::prelude::*;

use crate::fock::KBuilder;

/// Grid points per block. Fixed (not budget-derived) so the floating-point
/// association of the per-block GEMM accumulation — and hence the K matrix —
/// does not depend on the memory budget. 1024 points × 3 planes × nbf × 8 B
/// is ~300 MB at nbf = 12 800 (320 atoms / TZVP).
pub const COSX_BLOCK_POINTS: usize = 1024;

/// AO-evaluation chunk inside a block (parallel over chunks).
const AO_EVAL_CHUNK: usize = 64;

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
    /// Shell-pair screen threshold for the per-point A-build
    /// (`ferric_integrals::cosx_a::CosxScreen`). `None` = unscreened (no pair
    /// bounds built at all). `Some(t)` builds `PairBounds` and screens at `t`;
    /// `Some(0.0)` is the screen's trivial limit and reproduces `None`
    /// bit-for-bit (anchored). Default `None`.
    ///
    /// **`t > 0` is REFUSED at construction (2026-09-07).** The cosx_a screen
    /// bounds a pair by `sqrt(max|S_block|)/d`, which is exactly zero for
    /// same-centre pairs whose overlap vanishes by angular symmetry (s–p,
    /// px–py, …) while their `A^g` at an off-centre grid point is O(1); the
    /// screen therefore drops real contributions at ANY positive threshold.
    /// Measured: water/cc-pVDZ, (50,110), `t = 1e-7`: `max|K_scr - K_unscr|
    /// = 0.71` against a grid error of 4.8e-5. The guard is pinned by
    /// `cosx_a_screen_is_unsound_tripwire` in `tests/cosx_k_anchors.rs`; lift
    /// it when that tripwire fails (i.e. when the screen is fixed).
    pub screen_thresh: Option<f64>,
}

impl Default for CosxConfig {
    fn default() -> Self {
        Self {
            grid: AtomicGridConfig { n_radial: 50, n_angular: 110, ..Default::default() },
            overlap_fit: true,
            screen_thresh: None,
        }
    }
}

/// Wall-time split of the most recent `build`, for measurement. Seconds are
/// summed over all worker threads for the per-point segments (`a_build_s`,
/// `contract_s`), i.e. they equal wall time only on one thread.
#[derive(Debug, Clone, Copy, Default)]
pub struct CosxTimings {
    /// AO values on the grid (`X = sqrt(w) chi`), all blocks.
    pub ao_eval_s: f64,
    /// Per-point `A^g` construction (the libint2 3c1e sweep), summed over threads.
    pub a_build_s: f64,
    /// Per-point `G_g = A^g F_g` GEMVs, summed over threads.
    pub contract_s: f64,
    /// Block GEMMs: `F = D X` (or `C (C^T X)`), `Ktilde += X G^T`, `S_num += X X^T`.
    pub blas_s: f64,
    /// Overlap-fit finalization (Cholesky on first build + two triangular solves + `S Z`).
    pub fit_s: f64,
    /// Whole `build` call, wall.
    pub total_s: f64,
    /// Shell pairs evaluated after screening, summed over points.
    pub pairs_kept: usize,
    /// Shell pairs considered, summed over points.
    pub pairs_total: usize,
}

/// One libint2 nuclear-attraction engine per rayon worker (plus a spare for
/// non-pool threads) — same rationale as `ferric_integrals::engine_pool`,
/// which only builds 2e engines.
struct NuclearPool {
    engines: Vec<Mutex<Engine>>,
}

impl NuclearPool {
    fn new(prep: &PreparedBasis) -> Result<Self, FerricError> {
        let n = rayon::current_num_threads().max(1) + 1;
        let mut engines = Vec::with_capacity(n);
        for _ in 0..n {
            engines.push(Mutex::new(Engine::new_1e(ffi::OP_NUCLEAR, prep, 1e-14)?));
        }
        Ok(Self { engines })
    }

    #[inline]
    fn with<R>(&self, f: impl FnOnce(&mut Engine) -> R) -> R {
        let idx = rayon::current_thread_index().unwrap_or(self.engines.len() - 1);
        let slot = idx.min(self.engines.len() - 1);
        let mut eng = self.engines[slot].lock().unwrap();
        f(&mut eng)
    }
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
    pool: Option<NuclearPool>,
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
        check_budget(prep.nbasis(), mem_budget)?;

        let bounds = match cfg.screen_thresh {
            Some(t) if t > 0.0 => {
                return Err(FerricError::General(format!(
                    "CosxK: screen_thresh = {t:e} refused — the cosx_a shell-pair screen is unsound at any \
                     positive threshold (its overlap-magnitude bound is exactly zero for same-centre pairs \
                     whose overlap vanishes by symmetry, e.g. s-p, while A^g for them is O(1); measured \
                     max|dK| = 0.71 on water/cc-pVDZ at 1e-7). Use None (unscreened) or Some(0.0)."
                )));
            }
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
            pool: None,
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

    /// Per-point loop: `G^T_{g,nu} = sum_lam A^g_{nu,lam} F_{lam,g}`, returned
    /// as `(B, nbf)` so each point owns one contiguous row.
    fn contract_block(
        &self,
        pool: &NuclearPool,
        pts: &[[f64; 3]],
        f: &Array2<f64>,
        acc: &BlockCounters,
    ) -> Result<Array2<f64>, FerricError> {
        let nbf = self.prep.nbasis();
        let (bounds, screen) = match self.cfg.screen_thresh {
            None => (None, CosxScreen::none()),
            Some(t) => (self.bounds.as_ref(), CosxScreen::at(t)),
        };
        let prep = self.prep;
        let mut gt = Array2::<f64>::zeros((pts.len(), nbf));
        let rows = gt.as_slice_mut().expect("freshly allocated standard layout");
        rows.par_chunks_mut(nbf)
            .enumerate()
            .try_for_each(|(g, row)| -> Result<(), FerricError> {
                let t0 = Instant::now();
                let pt = pool.with(|eng| a_matrix_at_point_with(eng, prep, &pts[g], bounds, screen))?;
                let t1 = Instant::now();
                let col = pt.a.dot(&f.column(g));
                row.copy_from_slice(col.as_slice().expect("dot result contiguous"));
                acc.a_ns.fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
                acc.c_ns.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);
                acc.kept.fetch_add(pt.pairs_kept, Ordering::Relaxed);
                acc.total.fetch_add(pt.pairs_total, Ordering::Relaxed);
                Ok(())
            })?;
        Ok(gt)
    }

    /// Shared driver for `build` / `build_from_occ`: `half` maps `X_blk` to
    /// `F_blk` (`D X` or `C (C^T X)`).
    fn build_with(
        &mut self,
        half: &dyn Fn(&Array2<f64>) -> Array2<f64>,
        k: &mut Array2<f64>,
    ) -> Result<usize, FerricError> {
        let t_start = Instant::now();
        self.ctx.check_interrupted()?;
        if self.pool.is_none() {
            self.pool = Some(NuclearPool::new(self.prep)?);
        }
        let nbf = self.prep.nbasis();
        let need_snum = self.cfg.overlap_fit && self.snum.is_none();
        let mut ktilde = Array2::<f64>::zeros((nbf, nbf));
        let mut snum = need_snum.then(|| Array2::<f64>::zeros((nbf, nbf)));
        let acc = BlockCounters::default();
        let mut t = CosxTimings::default();

        let pool = self.pool.as_ref().expect("pool initialized above");
        for (pts, sw) in self.points.chunks(COSX_BLOCK_POINTS).zip(self.sqrt_w.chunks(COSX_BLOCK_POINTS)) {
            self.ctx.check_interrupted()?;
            let t0 = Instant::now();
            let x = self.eval_x_block(pts, sw)?;
            t.ao_eval_s += t0.elapsed().as_secs_f64();

            let t0 = Instant::now();
            if let Some(s) = snum.as_mut() {
                *s += &x.dot(&x.t());
            }
            let f = half(&x);
            t.blas_s += t0.elapsed().as_secs_f64();

            let gt = self.contract_block(pool, pts, &f, &acc)?;

            let t0 = Instant::now();
            ktilde += &x.dot(&gt);
            t.blas_s += t0.elapsed().as_secs_f64();
        }

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
        t.pairs_total = acc.total.load(Ordering::Relaxed);
        t.total_s = t_start.elapsed().as_secs_f64();
        self.last = t;
        // No shell quartets are computed by this builder; the work counter the
        // SCF reports as `computed_quartets` gets 0 (see `last_timings` for pairs).
        Ok(0)
    }
}

#[derive(Default)]
struct BlockCounters {
    a_ns: AtomicU64,
    c_ns: AtomicU64,
    kept: AtomicUsize,
    total: AtomicUsize,
}

/// Fail fast if one block's scratch does not fit the budget.
fn check_budget(nbf: usize, mem_budget: usize) -> Result<(), FerricError> {
    let budget = if mem_budget == 0 { ferric_core::memory::resolve_budget_bytes(None) } else { mem_budget };
    let threads = rayon::current_num_threads().max(1) + 1;
    let planes = 3usize.saturating_mul(COSX_BLOCK_POINTS).saturating_mul(nbf).saturating_mul(8);
    // per-thread A^g + Ktilde + S_num + S + L + L^T
    let squares = (threads + 5).saturating_mul(nbf).saturating_mul(nbf).saturating_mul(8);
    let needed = planes.saturating_add(squares);
    if needed > budget {
        return Err(FerricError::General(format!(
            "CosxK: one grid block needs {:.2} GB (nbf={nbf}, {COSX_BLOCK_POINTS} pts x 3 planes + {threads} per-thread A matrices) \
             but the memory budget is {:.2} GB — raise [memory] budget_gb / FERRIC_MEM_BUDGET_GB or use fewer threads",
            needed as f64 / 1e9,
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
        let half = |x: &Array2<f64>| d.dot(x);
        self.build_with(&half, k)
    }

    /// K for `D = C_occ C_occ^T` with the cheaper half transform
    /// `F = C (C^T X)`; identical to `build(C C^T)` to round-off (anchored).
    fn build_from_occ(&mut self, c_occ: &Array2<f64>, k: &mut Array2<f64>) -> Result<usize, FerricError> {
        let ct = c_occ.t().to_owned();
        let half = |x: &Array2<f64>| c_occ.dot(&ct.dot(x));
        self.build_with(&half, k)
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
        assert!(c.screen_thresh.is_none());
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
        assert!(check_budget(100, 1).is_err());
        assert!(check_budget(100, usize::MAX).is_ok());
    }
}
