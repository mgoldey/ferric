//! Density-fitted Coulomb matrix builder (RI-J).
//!
//! Replaces the O(N^4) direct ERI Coulomb build with O(N^2 · naux) GEMMs using
//! a 3-center auxiliary expansion:
//!
//!   J_{μν} = Σ_{P,Q} (μν|P) (P|Q)^{-1} (Q|λσ) D_{λσ}
//!          = Σ_P (μν|P) c_P,    c_P = Σ_Q (P|Q)^{-1} d_Q,    d_Q = Σ_{λσ} (Q|λσ) D_{λσ}
//!
//! The 3-center tensor B_{P,μν} = (P|μν) and the inverse Coulomb metric V^{-1}
//! are precomputed once at construction. Unlike DF-K's rank-2-per-P contraction,
//! DF-J's two passes are rank-1-per-P (a GEMV against the flattened density /
//! flattened accumulator): each block's GEMV already runs the full block width
//! `b` as its contraction/output dimension in one BLAS-adjacent call, so there
//! is no per-P GEMM stack to collapse into a wider GEMM. The idiom adopted from
//! DF-K (`df_k.rs::build`) here is about *parallelism and determinism*, not
//! GEMM shape:
//!
//!   Pass 1: d_P = Σ_μν B\[P,μν\] D\[μν\]. Each block is split into rayon-parallel
//!   aux-chunks; every chunk writes its GEMV result into a *disjoint* slice of
//!   `d_p` (indexed by aux row), so this pass needs no reduction at all — the
//!   result is bit-identical and thread-count independent by construction
//!   (disjoint writes, not summed partials).
//!
//!   Pass 2: J\[μν\] = Σ_P B\[P,μν\] c_P is a true reduction (every P contributes
//!   to the same μν output), so it uses
//!   `crate::reduce::grouped_deterministic_fold_columns`, which has the bits of the
//!   serial accumulator DF-K uses, over the same rayon aux-chunks.
//!
//! Block streaming (and its disk IO on the spill path) stays sequential in
//! `for_each_block`, exactly as DF-K; only the in-memory contraction of the
//! current block fans out over chunks.

use crate::fock::JBuilder;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::{PairScreen, ThreeIndexSource};
use ferric_integrals::threeindex::coulomb_metric_2c;
use ndarray::linalg::general_mat_vec_mul;
use ndarray::{Array1, Array2, OwnedRepr};
use ndarray_linalg::cholesky::{CholeskyFactorized, FactorizeC, SolveC};
use ndarray_linalg::UPLO;
use rayon::prelude::*;

/// DF-J Coulomb builder. Uses a budget-bounded ThreeIndexSource for raw (P|μν).
///
/// ## MPI aux-band striping
/// Under MPI (`new_banded` with a multi-rank [`ParallelContext`]), each rank
/// builds and holds ONLY its own contiguous aux band `[band_p0, band_p1)` of the
/// raw (P|μν) tensor — so the resident B footprint scales with rank count, not
/// just the FLOPs. DF-J's two passes bracket a global coupling (`c_P = V^{-1}
/// d_P` mixes all P), so `build` does TWO all_reduces:
///   1. after Pass 1, sum the per-rank partial `d_P` vectors → the full `d_P`
///      (a naux-vector, cheap) so every rank can apply the full V^{-1};
///   2. after Pass 2, sum the per-rank partial J matrices → the full J.
/// With one rank (or the `mpi` feature off) the band is `0..naux` and both
/// reductions are skipped, so behavior is byte-identical to the serial path.
pub struct DfJ<'a> {
    /// Budget-bounded raw 3-index source (in-core or disk-spill) — holds only
    /// this rank's aux band `[band_p0, band_p1)`.
    source: ThreeIndexSource,
    /// Cholesky factor of the (naux, naux) Coulomb metric V. FULL (all ranks
    /// hold it — it is small, O(naux²), and the V^{-1} d_P solve couples all P).
    ///
    /// NOT an explicit inverse. JK-fit metrics are ill-conditioned (MEASURED
    /// cond(V) = 7.1e8 for benzene/def2-universal-jkfit), and the old
    /// `v.inv()` (LU + dgetri) followed by a GEMV has a forward error of
    /// ~eps·cond that is NOT a smooth function of the density: with D fixed
    /// to 1e-9 it made J jitter by ~8e-7 and the SCF energy by ~1e-5 Ha on
    /// benzene/STO-3G (HF and KS alike), floored the DIIS commutator at ~1e-7,
    /// and let ANY bit-level perturbation (a 1e-9 Å rigid translation, a
    /// reordered V_xc sum) move a "converged" energy by 1e-6..1e-4 Ha. A
    /// backward-stable Cholesky solve makes E_J = ½ dᵀV⁻¹d smooth to ~1e-13
    /// (see `df_j_energy_is_smooth_in_the_density`).
    v_chol: CholeskyFactorized<OwnedRepr<f64>>,
    /// Parallel context for the cross-rank reductions. `None` → serial / non-MPI
    /// (no reduction, full band).
    #[allow(dead_code)]
    ctx: Option<&'a ParallelContext>,
    /// The memory budget this source was built under — also caps the pass-2
    /// reduction band scratch via `resolve_band_bytes`.
    budget_bytes: usize,
    /// Device-resident copy state (feature `gpu`); `Untried` until the first
    /// build asks for it. See `crate::df_j_gpu`.
    #[cfg(feature = "gpu")]
    device: crate::df_j_gpu::DeviceSlot,
    /// Device bytes the same SCF's DF-K will want (0 = none): the J upload leaves
    /// that much of the pool free, so the larger K tensor is not locked out by
    /// the J build that runs first each iteration.
    #[cfg(feature = "gpu")]
    k_reserve_bytes: usize,
}

impl<'a> std::fmt::Debug for DfJ<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DfJ")
            .field("budget_bytes", &self.budget_bytes)
            .finish_non_exhaustive()
    }
}

impl<'a> DfJ<'a> {
    /// Build the DF-J cache from orbital and auxiliary bases (FULL aux range,
    /// serial / single-rank — byte-identical to the pre-MPI path).
    ///
    /// `budget_bytes` is the hard ceiling for the resident raw 3-index footprint.
    /// Pass `usize::MAX` for the old in-core behaviour.
    pub fn new(
        op: Operator,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        budget_bytes: usize,
    ) -> Result<Self, FerricError> {
        Self::new_banded(op, obs, dfbs, budget_bytes, None)
    }

    /// Build the DF-J cache, striping the aux band across the MPI ranks of `ctx`.
    /// When `ctx` is `None` or `ctx.size == 1`, this is exactly [`DfJ::new`]
    /// (full band, no reduction). Otherwise this rank builds/holds only its band
    /// `ctx.aux_band(naux)`, and `build` all_reduces the intermediates + result.
    pub fn new_banded(
        op: Operator,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        budget_bytes: usize,
        ctx: Option<&'a ParallelContext>,
    ) -> Result<Self, FerricError> {
        let naux = dfbs.nbasis();
        let (p0, p1) = match ctx {
            Some(c) => c.aux_band(naux),
            None => (0, naux),
        };
        let source = ThreeIndexSource::build_band(op, obs, dfbs, budget_bytes, p0, p1)?;
        Self::from_source(source, op, dfbs, budget_bytes, ctx)
    }

    /// Same as [`DfJ::new_banded`] but places the raw tensor by the effective
    /// `[scf] jk_storage` policy (memory, disk spill or recompute), which
    /// needs the molecule to re-prepare the bases for the recompute backend.
    ///
    /// `pair_thresh > 0` stores only the AO pairs whose rows can exceed it
    /// (Schwarz, see [`PairScreen`]) whenever that compressed tensor fits in
    /// core; `<= 0` keeps every pair and is byte-identical to the unscreened
    /// build, as is a screen that would drop nothing (small molecules). The
    /// screen is skipped when the device path may take J (its
    /// upload expects the full packed triangle) and for an operator without a
    /// Schwarz table.
    pub fn new_for_jk(
        op: Operator,
        mol: &ferric_core::mol::Molecule,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        budget_bytes: usize,
        ctx: Option<&'a ParallelContext>,
        pair_thresh: f64,
    ) -> Result<Self, FerricError> {
        let naux = dfbs.nbasis();
        let band = match ctx {
            Some(c) => c.aux_band(naux),
            None => (0, naux),
        };
        let (source, v) = jk_source(op, mol, (obs, dfbs), budget_bytes, band, pair_thresh)?;
        Self::from_source_and_metric(source, v, budget_bytes, ctx)
    }

    /// Same as [`DfJ::new_banded`] but adopts an ALREADY-BUILT raw `(P|μν)`
    /// source instead of generating one.
    ///
    /// Exists so a DF-JK run generates the raw 3-index tensor ONCE. `DfJ` and
    /// `DfK` were otherwise independent and each called into
    /// `ThreeIndexSource`, computing the identical ~95.6M integrals twice
    /// (~600 ms per build at benzene/aug-cc-pVTZ). `DfK` only needs the raw
    /// tensor transiently (it keeps the V^{-1/2}-dressed copy), so the shared
    /// flow is: build raw once, dress DfK from `&mut raw`, then move raw here.
    ///
    /// `source` MUST cover exactly this rank's aux band, i.e. what
    /// `build_band(.., p0, p1)` would have produced — `build` reports GLOBAL aux
    /// indices and indexes the full-metric `c_P` with them, so a mismatched band would silently
    /// contract the wrong rows. Single-rank (`ctx` None or size 1) is the only
    /// case where DfJ's band and DfK's full range coincide; `build_df_jk` gates
    /// the sharing on exactly that.
    pub fn from_source(
        source: ThreeIndexSource,
        op: Operator,
        dfbs: &PreparedBasis,
        budget_bytes: usize,
        ctx: Option<&'a ParallelContext>,
    ) -> Result<Self, FerricError> {
        let v = coulomb_metric_2c(op, dfbs)?;
        Self::from_source_and_metric(source, v, budget_bytes, ctx)
    }

    /// [`DfJ::from_source`] with the `(naux, naux)` metric `v` already built.
    fn from_source_and_metric(
        source: ThreeIndexSource,
        v: Array2<f64>,
        budget_bytes: usize,
        ctx: Option<&'a ParallelContext>,
    ) -> Result<Self, FerricError> {
        // Cholesky (dpotrf), NOT the LU inverse this used to form: see the
        // `v_chol` field doc for why an explicit inverse is numerically wrong
        // here. Also sidesteps the multi-threaded dgetrf_parallel stack
        // overflow the LU path was exposed to (openblas-rayon-dgetrf-crash;
        // `.cholesky()` was never observed to crash at these sizes). A metric
        // that is not numerically positive definite is a hard error, never a
        // silently garbage inverse.
        let v_chol = v.factorizec(UPLO::Lower).map_err(|e| {
            FerricError::Lapack(format!(
                "Cholesky of the DF-J Coulomb metric failed (V is not numerically \
                 positive definite: the auxiliary basis is linearly dependent): {e}"
            ))
        })?;
        // Store ctx only when it actually implies a reduction (>1 rank); a size-1
        // ctx behaves exactly like None (full band, no all_reduce).
        let ctx = ctx.filter(|c| c.size > 1);
        Ok(DfJ {
            source,
            v_chol,
            ctx,
            budget_bytes,
            #[cfg(feature = "gpu")]
            device: crate::df_j_gpu::DeviceSlot::default(),
            #[cfg(feature = "gpu")]
            k_reserve_bytes: 0,
        })
    }

    /// Leave `bytes` of the device pool free when the RI-J tensor is uploaded
    /// (the DF-K footprint of the same SCF, see
    /// [`DfK::device_footprint_bytes`](crate::df_k::DfK::device_footprint_bytes)).
    /// If J and K together do not fit, J stays on the CPU and K gets the card.
    #[cfg(feature = "gpu")]
    pub fn reserve_device_for_k(&mut self, bytes: usize) {
        self.k_reserve_bytes = bytes;
    }

    /// The raw packed band as `(naux, n(n+1)/2)`; `None` unless the packed
    /// in-core tier. Test hook.
    #[cfg(feature = "gpu")]
    #[doc(hidden)]
    pub fn packed_incore_for_test(&self) -> Option<ndarray::ArrayView2<'_, f64>> {
        self.source.packed_flat()
    }

    /// `V⁻¹ d` through the stored Cholesky factor (the solve `build` uses). Test hook.
    #[cfg(feature = "gpu")]
    #[doc(hidden)]
    pub fn solve_metric_for_test(&self, d_p: &Array1<f64>) -> Result<Array1<f64>, FerricError> {
        self.solve_metric(d_p)
    }

    /// Does this builder currently hold its raw tensor on the device? Test hook.
    #[cfg(feature = "gpu")]
    #[doc(hidden)]
    pub fn device_resident_for_test(&self) -> bool {
        matches!(self.device, crate::df_j_gpu::DeviceSlot::Ready(_))
    }
}

impl DfJ<'_> {
    /// `c = V^{-1} d` by a backward-stable Cholesky solve against the stored
    /// factor (see the `v_chol` field doc).
    fn solve_metric(&self, d_p: &Array1<f64>) -> Result<Array1<f64>, FerricError> {
        self.v_chol
            .solvec(d_p)
            .map_err(|e| FerricError::Lapack(format!("DF-J metric solve failed: {e}")))
    }
}

/// The raw tensor `new_for_jk` holds, and the aux metric it built on the way:
/// the pair-screened in-core tier when [`pair_screen_for`] screens and the
/// compressed band fits, otherwise the full tensor under `[scf] jk_storage`.
fn jk_source(
    op: Operator,
    mol: &ferric_core::mol::Molecule,
    (obs, dfbs): (&PreparedBasis, &PreparedBasis),
    budget_bytes: usize,
    band: (usize, usize),
    pair_thresh: f64,
) -> Result<(ThreeIndexSource, Array2<f64>), FerricError> {
    let v = coulomb_metric_2c(op, dfbs)?;
    if let Some(screen) = pair_screen_for(op, obs, &v, pair_thresh) {
        let built = ThreeIndexSource::build_pair_screened_for_jk(
            op,
            obs,
            dfbs,
            budget_bytes,
            band,
            screen,
        )?;
        if let Some(source) = built {
            return Ok((source, v));
        }
    }
    let source = ThreeIndexSource::build_for_jk(op, mol, obs, dfbs, budget_bytes, band, false)?;
    Ok((source, v))
}

/// The RI-J pair screen for `pair_thresh`, or `None` to keep every pair:
/// `pair_thresh <= 0`, a device path that may take J, or an operator with no
/// Schwarz table. `v` is the operator's own aux metric; its diagonal gives
/// `max_P sqrt((P|P))`.
fn pair_screen_for(
    op: Operator,
    obs: &PreparedBasis,
    v: &Array2<f64>,
    pair_thresh: f64,
) -> Option<std::sync::Arc<PairScreen>> {
    if pair_thresh <= 0.0 || device_may_take_j() {
        return None;
    }
    let q_aux_max = v.diag().iter().fold(0.0_f64, |m, &x| m.max(x.abs().sqrt()));
    PairScreen::schwarz(op, obs, q_aux_max, pair_thresh)
        .ok()
        // Nothing dropped: keep the packed tier, whose bits every other tier
        // reproduces, rather than the same values in a different column order.
        .filter(|s| s.ncols() < s.total_pairs())
        .map(std::sync::Arc::new)
}

/// Could the device path build this J? Its upload reads the full packed
/// triangle, so a pair-screened source would only make it decline.
fn device_may_take_j() -> bool {
    #[cfg(feature = "gpu")]
    {
        ferric_core::gpu::settings().mode != ferric_core::gpu::GpuMode::Off
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// Aux-chunk width for one block's rayon fan-out. Mirrors DF-K's sizing
/// (`4096 / n`, clamped to [4, 64]): wide enough that each chunk's GEMV runs a
/// non-trivial contraction even for small n, capped so per-chunk scratch stays
/// modest for large n.
fn chunk_width(n: usize) -> usize {
    ferric_integrals::three_index_source::df_chunk_width(n)
}

/// Partials pass 2 keeps at once: what the band budget affords, at most
/// [`crate::reduce::MAX_FOLD_PARTIALS`]. Sized with `n²` per partial on the packed
/// path as well, which is only cautious. A result cannot depend on it.
fn pass2_band(n: usize, budget_bytes: usize) -> usize {
    pass2_band_for_bytes(n, crate::reduce::resolve_band_bytes(budget_bytes))
}

fn pass2_band_for_bytes(n: usize, band_bytes: usize) -> usize {
    crate::reduce::band_width(n, band_bytes).min(crate::reduce::MAX_FOLD_PARTIALS)
}

/// `acc += flat^T c` for one block, one fixed-width aux chunk at a time. Chunk
/// edges depend only on the block height and `chunk`.
fn pass2_block(
    acc: &mut [f64],
    flat: ndarray::ArrayView2<'_, f64>,
    c: ndarray::ArrayView1<'_, f64>,
    chunk: usize,
    band: usize,
    scratch: &mut Vec<Vec<f64>>,
) {
    let b = flat.nrows();
    crate::reduce::grouped_deterministic_fold_columns(
        acc,
        b.div_ceil(chunk),
        band,
        scratch,
        |ci, y| {
            let q0 = ci * chunk;
            let q1 = (q0 + chunk).min(b);
            let sub = flat.slice(ndarray::s![q0..q1, ..]);
            let c_sub = c.slice(ndarray::s![q0..q1]);
            // contrib[μν] = Σ_{P in chunk} B[P,μν] c_P, as `sub.t().dot(&c_sub)`
            // computes it. beta = 0, so `y`'s old contents are not read.
            let mut y = ndarray::ArrayViewMut1::from(y);
            general_mat_vec_mul(1.0, &sub.t(), &c_sub, 0.0, &mut y);
        },
    );
}

/// Weights `w` such that `Σ_{μν} B[μν] D[μν] = Σ_k Bp[k] w[k]` for any
/// μν-symmetric `B` stored as its packed lower triangle `Bp` in `(μ, ν≤μ)`
/// row-major order: `w[μμ] = D[μμ]` and `w[μν] = D[μν] + D[νμ]` off the
/// diagonal. This is the symmetrised density `(D + Dᵀ)/2` with the off-diagonal
/// doubled, written without the intermediate halving, so a non-symmetric `D`
/// gives exactly the contraction the unpacked path computes.
pub(crate) fn packed_density_weights(d: &Array2<f64>, n: usize) -> Array1<f64> {
    let mut w = Array1::<f64>::zeros(n * (n + 1) / 2);
    let mut k = 0usize;
    for mu in 0..n {
        for nu in 0..mu {
            w[k] = d[(mu, nu)] + d[(nu, mu)];
            k += 1;
        }
        w[k] = d[(mu, mu)];
        k += 1;
    }
    w
}

/// [`packed_density_weights`] at the stored columns of a pair-screened tensor:
/// `w[k] = D[μμ]` on a diagonal pair, `D[μν] + D[νμ]` off it, for
/// `cols[k] = (μ, ν)`.
fn pair_density_weights(d: &Array2<f64>, cols: &[(u32, u32)]) -> Array1<f64> {
    cols.iter()
        .map(|&(mu, nu)| {
            let (mu, nu) = (mu as usize, nu as usize);
            if mu == nu {
                d[(mu, mu)]
            } else {
                d[(mu, nu)] + d[(nu, mu)]
            }
        })
        .collect()
}

/// Write the per-column accumulator of a pair-screened J into the full
/// symmetric `j`; every pair the screen dropped is zero.
fn scatter_pair_cols(acc: &[f64], cols: &[(u32, u32)], j: &mut Array2<f64>) {
    j.fill(0.0);
    for (&v, &(mu, nu)) in acc.iter().zip(cols) {
        let (mu, nu) = (mu as usize, nu as usize);
        j[(mu, nu)] = v;
        j[(nu, mu)] = v;
    }
}

/// Expand a packed lower triangle into the full symmetric matrix `j`.
pub(crate) fn unpack_symmetric(packed: &[f64], n: usize, j: &mut Array2<f64>) {
    let mut k = 0usize;
    for mu in 0..n {
        for nu in 0..=mu {
            j[(mu, nu)] = packed[k];
            j[(nu, mu)] = packed[k];
            k += 1;
        }
    }
}

impl DfJ<'_> {
    /// Two-pass J over the packed in-core tensor.
    ///
    /// Same structure and determinism as the unpacked path (pass 1: disjoint
    /// per-chunk GEMV slices of `d_P`; pass 2: `grouped_deterministic_sum` over
    /// the same fixed chunks) but every GEMV runs over `n(n+1)/2` columns, so
    /// the resident tensor and the per-iteration bytes are half. Chunk edges
    /// depend only on the band height and `n`, never on the thread count.
    ///
    /// The result equals the unpacked path to summation-order rounding: the
    /// packed dot product adds each off-diagonal pair `B[μν] (D[μν] + D[νμ])`
    /// where the unpacked one adds `B[μν] D[μν] + B[νμ] D[νμ]`.
    fn build_packed(&mut self, d: &Array2<f64>, j: &mut Array2<f64>) -> Result<(), FerricError> {
        let naux = self.source.naux();
        let n = self.source.nao();
        let chunk = chunk_width(n);
        let w = packed_density_weights(d, n);
        let pair = n * (n + 1) / 2;

        let mut d_p = Array1::<f64>::zeros(naux);
        self.source.for_each_packed_block(chunk, |blk| {
            let b = blk.data.nrows();
            let parts: Vec<Array1<f64>> = (0..b.div_ceil(chunk))
                .into_par_iter()
                .map(|ci| {
                    let q0 = ci * chunk;
                    let q1 = (q0 + chunk).min(b);
                    blk.data.slice(ndarray::s![q0..q1, ..]).dot(&w)
                })
                .collect();
            for (ci, part) in parts.iter().enumerate() {
                let q0 = blk.p0 + ci * chunk;
                d_p.slice_mut(ndarray::s![q0..q0 + part.len()]).assign(part);
            }
            Ok(())
        })?;
        self.reduce_d_p(&mut d_p);
        let c_p = self.solve_metric(&d_p)?;

        let mut acc = vec![0.0_f64; pair];
        let band = pass2_band(n, self.budget_bytes);
        let mut scratch = Vec::new();
        self.source.for_each_packed_block(chunk, |blk| {
            let c_blk = c_p.slice(ndarray::s![blk.p0..blk.p0 + blk.data.nrows()]);
            pass2_block(&mut acc, blk.data, c_blk, chunk, band, &mut scratch);
            Ok(())
        })?;
        unpack_symmetric(&acc, n, j);
        Ok(())
    }

    /// Two-pass J over the pair-screened in-core tensor (see [`PairScreen`]).
    ///
    /// [`Self::build_packed`] restricted to the stored columns: pass 1 dots
    /// each fixed aux chunk with the packed density weights gathered at the
    /// kept pairs (disjoint `d_P` slices), pass 2 folds the same chunks into
    /// one accumulator per kept pair. Chunk edges depend only on the band
    /// height and `n`, so the bits are thread-count independent. A dropped
    /// pair's J is zero; its true magnitude is at most
    /// `thresh · Σ_P |c_P| · Q_P / max_P Q_P` (the `pair_screen` module doc).
    fn build_pair_screened(&self, d: &Array2<f64>, j: &mut Array2<f64>) -> Result<(), FerricError> {
        let (data, screen) = self
            .source
            .pair_screened()
            .ok_or_else(|| FerricError::General("DF-J: source is not pair-screened".into()))?;
        let (naux, n, p0) = (self.source.naux(), self.source.nao(), self.source.band().0);
        let chunk = chunk_width(n);
        let w = pair_density_weights(d, screen.cols());
        let b = data.nrows();

        let parts: Vec<Array1<f64>> = (0..b.div_ceil(chunk))
            .into_par_iter()
            .map(|ci| {
                let q1 = ((ci + 1) * chunk).min(b);
                data.slice(ndarray::s![ci * chunk..q1, ..]).dot(&w)
            })
            .collect();
        let mut d_p = Array1::<f64>::zeros(naux);
        for (ci, part) in parts.iter().enumerate() {
            let q0 = p0 + ci * chunk;
            d_p.slice_mut(ndarray::s![q0..q0 + part.len()]).assign(part);
        }
        self.reduce_d_p(&mut d_p);
        let c_p = self.solve_metric(&d_p)?;

        let mut acc = vec![0.0_f64; screen.ncols()];
        let band = pass2_band(n, self.budget_bytes);
        let c_blk = c_p.slice(ndarray::s![p0..p0 + b]);
        pass2_block(&mut acc, data, c_blk, chunk, band, &mut Vec::new());
        scatter_pair_cols(&acc, screen.cols(), j);
        Ok(())
    }

    /// MPI: sum the per-rank partial `d_P` (disjoint bands, zero elsewhere) so
    /// every rank holds the FULL `d_P` before the global V^{-1} coupling.
    fn reduce_d_p(&self, d_p: &mut Array1<f64>) {
        #[cfg(feature = "mpi")]
        if let Some(ctx) = self.ctx {
            use mpi::traits::CommunicatorCollectives;
            if let Some(world) = ctx.world() {
                let mut d_p_global = Array1::<f64>::zeros(d_p.len());
                world.all_reduce_into(
                    d_p.as_slice().unwrap(),
                    d_p_global.as_slice_mut().unwrap(),
                    mpi::collective::SystemOperation::sum(),
                );
                *d_p = d_p_global;
            }
        }
        #[cfg(not(feature = "mpi"))]
        let _ = d_p;
    }

    /// MPI: sum the per-rank partial J matrices into the full J on every rank.
    fn reduce_j(&self, j: &mut Array2<f64>) {
        #[cfg(feature = "mpi")]
        if let Some(ctx) = self.ctx {
            use mpi::traits::CommunicatorCollectives;
            if let Some(world) = ctx.world() {
                let mut j_global = Array2::<f64>::zeros(j.dim());
                world.all_reduce_into(
                    j.as_slice().unwrap(),
                    j_global.as_slice_mut().unwrap(),
                    mpi::collective::SystemOperation::sum(),
                );
                *j = j_global;
            }
        }
        #[cfg(not(feature = "mpi"))]
        let _ = j;
    }
}

impl JBuilder for DfJ<'_> {
    fn build(&mut self, d: &Array2<f64>, j: &mut Array2<f64>) -> Result<usize, FerricError> {
        // Device-resident path (opt-in `gpu` feature, `[gpu] mode` != off). It
        // declines to the host path below for every reason it cannot run.
        #[cfg(feature = "gpu")]
        {
            let v_chol = &self.v_chol;
            let solve = |d_p: &Array1<f64>| {
                v_chol
                    .solvec(d_p)
                    .map_err(|e| FerricError::Lapack(format!("DF-J metric solve failed: {e}")))
            };
            if crate::df_j_gpu::try_build(
                &mut self.device,
                &mut self.source,
                (self.ctx, self.k_reserve_bytes),
                d,
                j,
                solve,
            )? {
                return Ok(0);
            }
        }
        if self.source.pair_screened().is_some() {
            self.build_pair_screened(d, j)?;
            self.reduce_j(j);
            return Ok(0);
        }
        if self.source.supports_packed_stream() {
            self.build_packed(d, j)?;
            self.reduce_j(j);
            return Ok(0);
        }
        let naux = self.source.naux();
        let n = self.source.nao();
        let chunk = chunk_width(n);

        let d_flat = d
            .view()
            .into_shape_with_order(n * n)
            .map_err(|e| FerricError::General(format!("D reshape: {e}")))?;

        // Pass 1: d_P = Σ_{μν} B[P,μν] D[μν]. Each block is split into
        // rayon-parallel aux-chunks; every chunk's GEMV writes a disjoint
        // slice of d_p (indexed by aux row P), so no reduction/ordering is
        // needed — the result is bit-identical regardless of thread count by
        // construction.
        let mut d_p = ndarray::Array1::<f64>::zeros(naux);
        self.source.for_each_block(|blk| {
            let b = blk.data.shape()[0];
            let flat = blk
                .data
                .into_shape_with_order((b, n * n))
                .map_err(|e| FerricError::General(format!("blk reshape: {e}")))?;
            let n_chunks = b.div_ceil(chunk);
            let parts: Vec<(usize, usize, ndarray::Array1<f64>)> = (0..n_chunks)
                .into_par_iter()
                .map(|ci| -> Result<_, FerricError> {
                    let q0 = ci * chunk;
                    let q1 = (q0 + chunk).min(b);
                    let sub = flat.slice(ndarray::s![q0..q1, ..]);
                    let part = sub.dot(&d_flat); // (q1-q0,)
                    Ok((q0, q1, part))
                })
                .collect::<Result<Vec<_>, FerricError>>()?;
            for (q0, q1, part) in parts {
                d_p.slice_mut(ndarray::s![blk.p0 + q0..blk.p0 + q1])
                    .assign(&part);
            }
            Ok(())
        })?;

        self.reduce_d_p(&mut d_p);

        // c_P = V^{-1} d_P by Cholesky solve (full d_P → identical c_P on
        // every rank).
        let c_p = self.solve_metric(&d_p)?;

        // Pass 2: J[μν] = Σ_P B[P,μν] c_P. This IS a true reduction (every P
        // contributes to every μν), so accumulate through the ordered column
        // fold, over rayon-parallel aux-chunks within each block.
        let band = pass2_band(n, self.budget_bytes);
        let mut scratch = Vec::new();
        let mut acc = vec![0.0_f64; n * n];
        self.source.for_each_block(|blk| {
            let b = blk.data.shape()[0];
            let flat = blk
                .data
                .into_shape_with_order((b, n * n))
                .map_err(|e| FerricError::General(format!("blk reshape: {e}")))?;
            let c_blk = c_p.slice(ndarray::s![blk.p0..blk.p0 + b]);
            pass2_block(&mut acc, flat, c_blk, chunk, band, &mut scratch);
            Ok(())
        })?;
        let acc = ndarray::ArrayView2::from_shape((n, n), &acc)
            .map_err(|e| FerricError::General(format!("J reshape: {e}")))?;
        j.assign(&acc);

        self.reduce_j(j);

        Ok(0)
    }

    fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct_j::DirectJ;
    use crate::screening::SchwarzBounds;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_core::parallel::ParallelContext;
    use ndarray::Array2;

    /// Pass 2 has the bits of the accumulator it replaced (`grouped_deterministic_sum`
    /// over `dot`), at any thread count and band, from a dirty scratch, over several
    /// aux chunks. The tensor is synthetic, with no μν symmetry and a wide range of
    /// sizes, so that a transposed result or a reordered sum would show.
    #[test]
    fn pass2_has_the_bits_of_the_grouped_sum_over_dot() {
        let (n, naux) = (23usize, 611usize);
        let chunk = chunk_width(n);
        assert!(naux > 4 * chunk, "too few aux functions for several chunks");
        let flat = Array2::from_shape_fn((naux, n * n), |(p, k)| {
            let x = ((p * 13 + k * 7) % 29) as f64 - 14.0;
            x * 10.0_f64.powi(((p + 2 * k) % 13) as i32 - 6) / (1.3 + k as f64)
        });
        let c = Array1::from_shape_fn(naux, |p| ((p * 5 % 17) as f64 - 8.0) * 0.37);
        let mut reference = Array2::<f64>::zeros((n, n));
        crate::reduce::grouped_deterministic_sum(
            &mut reference,
            naux.div_ceil(chunk),
            n,
            1 << 30,
            |ci| {
                let q0 = ci * chunk;
                let q1 = (q0 + chunk).min(naux);
                let sub = flat.slice(ndarray::s![q0..q1, ..]);
                let part = sub.t().dot(&c.slice(ndarray::s![q0..q1]));
                Ok(part.into_shape_with_order((n, n)).unwrap())
            },
        )
        .unwrap();
        let expected: Vec<u64> = reference.iter().map(|x| x.to_bits()).collect();
        for threads in [1usize, 2, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for band in [1usize, 5, crate::reduce::MAX_FOLD_PARTIALS] {
                let mut acc = vec![0.0_f64; n * n];
                let mut scratch = vec![vec![f64::NAN; n * n]; band];
                pool.install(|| {
                    pass2_block(&mut acc, flat.view(), c.view(), chunk, band, &mut scratch);
                });
                let got: Vec<u64> = acc.iter().map(|x| x.to_bits()).collect();
                assert!(got == expected, "threads {threads} band {band}");
            }
        }
    }

    /// The unpacked path places a block by its global aux index. An in-core band
    /// starting past zero is the case where getting that wrong shows, in pass 1 and in
    /// pass 2: without a reduction, this J is exactly the band's own contribution.
    #[test]
    fn unpacked_j_uses_the_global_aux_index_of_a_band() {
        let mol = alkane(2);
        let obs = PreparedBasis::new(&mol, &basis::bundled("def2-svp").unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let (n, naux) = (obs.nbasis(), dfbs.nbasis());
        let p0 = naux / 3;
        assert!(p0 > 0);
        let d = dense_density(n, false);

        let mut full = DfJ::new(op, &obs, &dfbs, usize::MAX).unwrap();
        assert!(
            !full.source.supports_packed_stream(),
            "not the unpacked path"
        );
        let mut b = Array2::<f64>::zeros((naux, n * n));
        full.source
            .for_each_block(|blk| {
                for (i, slab) in blk.data.outer_iter().enumerate() {
                    let row = slab.as_standard_layout().to_owned();
                    let row = row.into_shape_with_order(n * n).unwrap();
                    b.row_mut(blk.p0 + i).assign(&row);
                }
                Ok(())
            })
            .unwrap();

        let d_flat = d.view().into_shape_with_order(n * n).unwrap();
        let mut d_p = Array1::<f64>::zeros(naux);
        for p in p0..naux {
            d_p[p] = b.row(p).dot(&d_flat);
        }
        let c = full.solve_metric(&d_p).unwrap();
        let mut expected = Array1::<f64>::zeros(n * n);
        for p in p0..naux {
            expected.scaled_add(c[p], &b.row(p));
        }

        let src = ThreeIndexSource::build_band(op, &obs, &dfbs, usize::MAX, p0, naux).unwrap();
        assert!(!src.supports_packed_stream(), "not the unpacked path");
        let mut band = DfJ::from_source(src, op, &dfbs, usize::MAX, None).unwrap();
        let mut j = Array2::<f64>::zeros((n, n));
        band.build(&d, &mut j).unwrap();
        let worst = j
            .iter()
            .zip(expected.iter())
            .map(|(a, e)| (a - e).abs())
            .fold(0.0, f64::max);
        // Relative: the band's d_P makes c large, and its solve carries that rounding.
        let scale = expected.iter().map(|x| x.abs()).fold(0.0, f64::max);
        assert!(scale > 1e-3, "a vacuous J");
        assert!(
            worst < 1e-8 * scale,
            "band J differs by {worst:e} of {scale:e}"
        );
    }

    /// The band can never pass the fixed cap or the budget's share, and it does not
    /// read the thread count.
    #[test]
    fn pass2_band_is_capped_and_ignores_the_pool() {
        // Takes the bytes directly: `resolve_band_bytes` reads an env var other tests
        // set, and this test is about what happens after it.
        let gib = 1usize << 30;
        for threads in [1usize, 2, 12] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                assert_eq!(
                    pass2_band_for_bytes(490, gib),
                    crate::reduce::MAX_FOLD_PARTIALS
                );
                assert_eq!(pass2_band_for_bytes(490, 490 * 490 * 8 * 3), 3);
                assert_eq!(pass2_band_for_bytes(490, 1), 1);
            });
        }
    }

    #[test]
    fn df_j_matches_direct_j_within_fit_error() {
        let mol =
            Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
        let obs_set = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_set).unwrap();
        let dfbs_set = basis::bundled("cc-pvdz-ri").unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();

        let op = Operator::coulomb();
        let n = obs.nbasis();

        // Make a representative non-trivial density (approximate; just for J-build comparison).
        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n.min(5) {
            d[(i, i)] = 2.0;
        }

        let mut j_direct = Array2::zeros((n, n));
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let ctx = ParallelContext::default();
        let mut dj = DirectJ::new(&ctx, &obs, &bounds, 1e-12, usize::MAX);
        <DirectJ as JBuilder>::build(&mut dj, &d, &mut j_direct).unwrap();

        let mut j_df = Array2::zeros((n, n));
        let mut dfj = DfJ::new(op, &obs, &dfbs, usize::MAX).unwrap();
        dfj.build(&d, &mut j_df).unwrap();

        let max_diff: f64 = (&j_df - &j_direct)
            .iter()
            .map(|v| v.abs())
            .fold(0.0, f64::max);
        // RI-fit basis is tuned for correlation, not J — accept ~1e-3 Ha-scale error.
        assert!(
            max_diff < 5e-3,
            "DF-J vs direct-J max diff = {} too large",
            max_diff
        );
    }

    #[test]
    fn df_j_source_backed_matches_incore() {
        let mol =
            Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let n = obs.nbasis();
        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n.min(5) {
            d[(i, i)] = 2.0;
        }

        // huge budget = in-core path
        let mut j_big = Array2::zeros((n, n));
        let mut dfj_big = DfJ::new(op, &obs, &dfbs, usize::MAX).unwrap();
        dfj_big.build(&d, &mut j_big).unwrap();

        // tiny budget = spill path; must be bit-identical
        let tiny = n * n * 8 * 3;
        let mut j_small = Array2::zeros((n, n));
        let mut dfj_small = DfJ::new(op, &obs, &dfbs, tiny).unwrap();
        dfj_small.build(&d, &mut j_small).unwrap();

        let maxdiff = (&j_big - &j_small)
            .iter()
            .map(|v| v.abs())
            .fold(0.0, f64::max);
        assert!(maxdiff < 1e-10, "spill J != in-core J, maxdiff={maxdiff}");
    }

    #[test]
    fn df_j_chunked_matches_naive_contraction() {
        // Implementation-equivalence test for the rayon-chunked restructure:
        // the chunked GEMV contraction in `build` must reproduce the naive
        // per-block (unchunked) two-pass contraction over the SAME raw B
        // tensor and metric solve to machine precision. A dense symmetric density
        // exercises every (P,μν) coupling, unlike the diagonal-D accuracy
        // test above (which measures RI fitting error vs direct J and cannot
        // separate algebra bugs from fitting error). Mirrors
        // `df_k_wide_gemm_matches_naive_contraction`.
        let mol =
            Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let n = obs.nbasis();

        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                d[(i, j)] = 0.02 * (((i * j + 3) % 11) as f64);
            }
            d[(i, i)] += 1.0;
        }

        let mut dfj = DfJ::new(op, &obs, &dfbs, usize::MAX).unwrap();
        let naux = dfj.source.naux();

        // Reference: naive per-block, unchunked two-pass contraction (the
        // pre-restructure algorithm) over the same raw source and metric.
        let d_flat = d.view().into_shape_with_order(n * n).unwrap();
        let mut d_p_ref = ndarray::Array1::<f64>::zeros(naux);
        dfj.source
            .for_each_block(|blk| {
                let b = blk.data.shape()[0];
                let flat = blk.data.into_shape_with_order((b, n * n)).unwrap();
                let part = flat.dot(&d_flat);
                d_p_ref
                    .slice_mut(ndarray::s![blk.p0..blk.p0 + b])
                    .assign(&part);
                Ok(())
            })
            .unwrap();
        let c_p_ref = dfj.solve_metric(&d_p_ref).unwrap();
        let mut j_ref = Array2::<f64>::zeros((n, n));
        {
            let mut j_flat = j_ref.view_mut().into_shape_with_order(n * n).unwrap();
            dfj.source
                .for_each_block(|blk| {
                    let b = blk.data.shape()[0];
                    let flat = blk.data.into_shape_with_order((b, n * n)).unwrap();
                    let c_blk = c_p_ref.slice(ndarray::s![blk.p0..blk.p0 + b]);
                    let contrib = flat.t().dot(&c_blk);
                    j_flat += &contrib;
                    Ok(())
                })
                .unwrap();
        }

        let mut j_df = Array2::zeros((n, n));
        dfj.build(&d, &mut j_df).unwrap();

        let max_diff: f64 = (&j_df - &j_ref).iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(
            max_diff < 1e-10,
            "chunked DF-J vs naive per-block contraction max diff = {} too large",
            max_diff
        );
    }

    /// A zig-zag alkane CnH(2n+2) with approximate geometry; the numbers only
    /// need to be non-degenerate, not relaxed.
    fn alkane(nc: usize) -> Molecule {
        let mut atoms: Vec<(&str, f64, f64, f64)> = Vec::new();
        for i in 0..nc {
            let x = 1.27 * i as f64;
            let y = if i % 2 == 0 { 0.0 } else { 0.5 };
            atoms.push(("C", x, y, 0.0));
            let s = if i % 2 == 0 { -1.0 } else { 1.0 };
            atoms.push(("H", x, y + s * 0.65, 0.9));
            atoms.push(("H", x, y + s * 0.65, -0.9));
        }
        atoms.push(("H", -1.0, 0.0, 0.0));
        atoms.push((
            "H",
            1.27 * (nc - 1) as f64 + 1.0,
            0.5 * ((nc - 1) % 2) as f64,
            0.0,
        ));
        let mut xyz = format!("{}\nalkane\n", atoms.len());
        for (e, x, y, z) in atoms {
            xyz.push_str(&format!("{e} {x} {y} {z}\n"));
        }
        Molecule::parse_xyz(&xyz, 0, 1).unwrap()
    }

    /// A budget that forces the packed in-core tier: above the packed footprint
    /// plus a scratch row, below the unpacked one.
    fn packed_tier_budget(obs: &PreparedBasis, dfbs: &PreparedBasis) -> usize {
        let (n, naux) = (obs.nbasis(), dfbs.nbasis());
        let unpacked = naux * n * n * 8;
        let packed = naux * (n * (n + 1) / 2) * 8;
        (unpacked + packed) / 2
    }

    fn dense_density(n: usize, symmetric: bool) -> Array2<f64> {
        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                d[(i, j)] = 0.01 * ((i * 7 + j * 3) % 11) as f64;
            }
        }
        if symmetric {
            d = 0.5 * (&d + &d.t());
        }
        d
    }

    /// Packed J vs the unpacked in-core J on the same raw tensor.
    ///
    /// Tolerance: the two paths differ only in summation order. A dot product
    /// of m terms of size |B||D| carries a rounding error of at most
    /// ~m*eps*sum|B||D|; with m = pair ~ 1e3..1e4 and the aux sum on top, that
    /// bounds the entrywise difference well under 1e-11 for the unit-scale
    /// densities used here, which `assert_packed_matches` checks relative to
    /// max|J|.
    fn assert_packed_matches(mol: &Molecule, orb: &str, aux: &str, symmetric: bool) -> f64 {
        let obs = PreparedBasis::new(mol, &basis::bundled(orb).unwrap()).unwrap();
        let dfbs = PreparedBasis::new(mol, &basis::bundled(aux).unwrap()).unwrap();
        let op = Operator::coulomb();
        let n = obs.nbasis();
        let d = dense_density(n, symmetric);
        let mut j_ref = Array2::zeros((n, n));
        let mut ref_j = DfJ::new(op, &obs, &dfbs, usize::MAX).unwrap();
        assert!(ref_j.source.is_incore());
        ref_j.build(&d, &mut j_ref).unwrap();
        let mut j_packed = Array2::zeros((n, n));
        let budget = packed_tier_budget(&obs, &dfbs);
        let mut pj = DfJ::new(op, &obs, &dfbs, budget).unwrap();
        assert!(pj.source.is_packed_incore_for_test(), "must be packed tier");
        pj.build(&d, &mut j_packed).unwrap();
        let scale = j_ref.iter().fold(1.0_f64, |a, v| a.max(v.abs()));
        let diff = (&j_packed - &j_ref)
            .iter()
            .fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(
            diff < 1e-11 * scale,
            "packed vs unpacked max|dJ|={diff} scale={scale}"
        );
        assert_eq!(j_packed, j_packed.t(), "packed J must be exactly symmetric");
        diff
    }

    #[test]
    fn packed_j_matches_unpacked_propane_symmetric_and_nonsymmetric_density() {
        let mol = alkane(3);
        assert_packed_matches(&mol, "def2-svp", "def2-universal-jkfit", true);
        // The unpacked path contracts the raw D; the symmetric B makes the
        // antisymmetric part of D drop out of J, which the packed weights
        // D[μν]+D[νμ] must reproduce.
        assert_packed_matches(&mol, "def2-svp", "def2-universal-jkfit", false);
    }

    #[test]
    fn packed_j_matches_unpacked_c4_and_twenty_atom_alkane() {
        let d4 = assert_packed_matches(&alkane(4), "def2-svp", "def2-universal-jkfit", true);
        // C6H14 has 20 atoms.
        let d6 = assert_packed_matches(&alkane(6), "def2-svp", "def2-universal-jkfit", false);
        println!("packed vs unpacked max|dJ|: C4 {d4:e}, C6H14 {d6:e}");
    }

    #[test]
    fn packed_j_is_bit_identical_across_packed_spilled_and_recompute_sources() {
        // The packed streaming API hands out blocks whose row counts are
        // multiples of the chunk width, so the fixed chunks and their ascending
        // fold are the same for every backend: exact equality, not a tolerance.
        let mol = alkane(3);
        let obs = PreparedBasis::new(&mol, &basis::bundled("def2-svp").unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let n = obs.nbasis();
        let d = dense_density(n, false);

        let mut j_packed = Array2::zeros((n, n));
        let mut pj = DfJ::new(op, &obs, &dfbs, packed_tier_budget(&obs, &dfbs)).unwrap();
        assert!(pj.source.is_packed_incore_for_test());
        pj.build(&d, &mut j_packed).unwrap();

        // A budget below the packed footprint forces the spill backend.
        let tiny = n * n * 8 * 40;
        let mut j_spill = Array2::zeros((n, n));
        let mut sj = DfJ::new(op, &obs, &dfbs, tiny).unwrap();
        assert!(sj.source.is_spilled_for_test());
        sj.build(&d, &mut j_spill).unwrap();
        assert_eq!(j_packed, j_spill, "spilled packed J != packed in-core J");

        let src = ferric_integrals::three_index_source::ThreeIndexSource::build_recomputing(
            op,
            std::sync::Arc::new(
                PreparedBasis::new(&mol, &basis::bundled("def2-svp").unwrap()).unwrap(),
            ),
            std::sync::Arc::new(
                PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap(),
            ),
            tiny,
        )
        .unwrap();
        assert!(src.is_recompute_for_test());
        let mut rj = DfJ::from_source(src, op, &dfbs, tiny, None).unwrap();
        let mut j_re = Array2::zeros((n, n));
        rj.build(&d, &mut j_re).unwrap();
        assert_eq!(j_packed, j_re, "recompute packed J != packed in-core J");
    }

    #[test]
    fn packed_j_bit_identical_across_worker_counts() {
        let mol = alkane(3);
        let obs = PreparedBasis::new(&mol, &basis::bundled("def2-svp").unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let n = obs.nbasis();
        let d = dense_density(n, false);
        let budget = packed_tier_budget(&obs, &dfbs);
        let build_j = |threads: usize| -> Array2<f64> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut j = Array2::zeros((n, n));
                let mut dfj = DfJ::new(op, &obs, &dfbs, budget).unwrap();
                assert!(dfj.source.is_packed_incore_for_test());
                dfj.build(&d, &mut j).unwrap();
                j
            })
        };
        assert_eq!(
            build_j(1),
            build_j(4),
            "packed DfJ must be bit-identical across worker counts"
        );
    }

    #[test]
    fn df_j_bit_identical_across_thread_counts() {
        // Regression guard mirroring df_k_bit_identical_across_thread_counts:
        // Pass 1's chunked GEMV writes disjoint slices of d_p (no reduction),
        // and Pass 2's chunked GEMV reduction goes through
        // grouped_deterministic_sum, so the J matrix (and hence the SCF
        // energy) must be bit-identical regardless of RAYON_NUM_THREADS.
        // Uses several heavy atoms so the aux dimension spans multiple
        // chunks (making order actually matter).
        let mol =
            Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let n = obs.nbasis();

        // Dense symmetric density that couples every (P,μν) pair.
        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                d[(i, j)] = 0.01 * ((i * 7 + j * 3) % 11) as f64;
            }
        }
        let d = 0.5 * (&d + &d.t());

        let build_j = |threads: usize| -> Array2<f64> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut j = Array2::zeros((n, n));
                let mut dfj = DfJ::new(op, &obs, &dfbs, usize::MAX).unwrap();
                dfj.build(&d, &mut j).unwrap();
                j
            })
        };

        let j1 = build_j(1);
        let j4 = build_j(4);
        // Bit-identical: exact equality, not a tolerance.
        assert_eq!(
            j1, j4,
            "DfJ build must be bit-identical across thread counts (rayon reduction order leak)"
        );
    }

    #[test]
    fn df_j_bit_identical_across_threads_with_narrow_bands() {
        // Same guarantee as above, but with a deliberately tiny reduce-band
        // budget so the two-level grouped_deterministic_sum path actually
        // splits the aux chunks across several bands in Pass 2. The banding
        // must not perturb the ascending-chunk fold order, so J stays
        // bit-identical at 1/2/8 threads.
        let mol =
            Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let n = obs.nbasis();

        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                d[(i, j)] = 0.01 * ((i * 7 + j * 3) % 11) as f64;
            }
        }
        let d = 0.5 * (&d + &d.t());

        std::env::set_var("FERRIC_REDUCE_BAND_BYTES", "4096");
        let build_j = |threads: usize| -> Array2<f64> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut j = Array2::zeros((n, n));
                let mut dfj = DfJ::new(op, &obs, &dfbs, usize::MAX).unwrap();
                dfj.build(&d, &mut j).unwrap();
                j
            })
        };
        let j1 = build_j(1);
        let j2 = build_j(2);
        let j8 = build_j(8);
        std::env::remove_var("FERRIC_REDUCE_BAND_BYTES");
        assert_eq!(
            j1, j2,
            "narrow-band DfJ must be bit-identical at 1 vs 2 threads"
        );
        assert_eq!(
            j1, j8,
            "narrow-band DfJ must be bit-identical at 1 vs 8 threads"
        );
    }

    /// E_J(D) = ½ Σ D∘J(D) is an exact quadratic in D, and J(D) exactly
    /// linear, so along a line D(t) = D + t·X a quadratic (resp. linear) fit
    /// must leave only rounding-level residuals. This is the property the old
    /// explicit-inverse metric (`v.inv()` + GEMV) violated: with cond(V) ~
    /// 7e8 its forward error is not a smooth function of D.
    ///
    /// MEASURED on exactly this fixture (numpy replica, same B/V/D/X, 21
    /// points): explicit inverse → E residual 1.2e-5 Ha, J residual 7.8e-7;
    /// Cholesky solve → E 2.8e-13, J 5.7e-14 (eigh: J 6.4e-14). The bars
    /// below sit ~4 orders from each side. In the SCF this jitter made the
    /// benzene/STO-3G energy wander ±1e-5 Ha with D fixed to 1e-9 (HF and
    /// PBE alike), floored the commutator at ~1e-7, and made a 71-atom
    /// "converged" PBE energy move 3.6e-4 Ha under a bit-level V_xc change.
    ///
    /// Artifact check: the fit is exact in exact arithmetic (orthogonal
    /// polynomial projection on a symmetric grid), so a large residual cannot
    /// come from the fit itself; the J-linearity half is independent of the
    /// energy contraction.
    #[test]
    fn df_j_energy_is_smooth_in_the_density() {
        let mol = Molecule::load_xyz("../../testdata/molecules/benzene.xyz").unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("sto-3g").unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let n = obs.nbasis();

        let mut d = Array2::<f64>::zeros((n, n));
        let mut x = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                d[(i, j)] = 0.02 * (((i * j + 3) % 11) as f64);
                x[(i, j)] = 1e-6 * ((((i * 7 + j * 3) % 13) as f64) - 6.0);
            }
            d[(i, i)] += 1.0;
        }
        let d = 0.5 * (&d + &d.t());
        let x = 0.5 * (&x + &x.t());

        let mut dfj = DfJ::new(op, &obs, &dfbs, usize::MAX).unwrap();
        let npt = 21usize;
        let ts: Vec<f64> = (0..npt)
            .map(|k| -1.0 + 2.0 * k as f64 / (npt - 1) as f64)
            .collect();
        let mut es = Vec::with_capacity(npt);
        let mut js = Vec::with_capacity(npt);
        for &t in &ts {
            let dt = &d + &(t * &x);
            let mut j = Array2::zeros((n, n));
            dfj.build(&dt, &mut j).unwrap();
            es.push(0.5 * (&dt * &j).sum());
            js.push(j);
        }

        // Orthogonal polynomials on the symmetric grid: 1, t, t² − <t²>.
        let m2 = ts.iter().map(|t| t * t).sum::<f64>() / npt as f64;
        let basis_fns: [Box<dyn Fn(f64) -> f64>; 3] = [
            Box::new(|_| 1.0),
            Box::new(|t| t),
            Box::new(move |t| t * t - m2),
        ];
        let fit_resid = |ys: &[f64], deg: usize| -> f64 {
            let mut r = ys.to_vec();
            for p in basis_fns.iter().take(deg + 1) {
                let num: f64 = ts.iter().zip(ys).map(|(&t, &y)| p(t) * y).sum();
                let den: f64 = ts.iter().map(|&t| p(t) * p(t)).sum();
                let c = num / den;
                for (k, &t) in ts.iter().enumerate() {
                    r[k] -= c * p(t);
                }
            }
            r.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
        };

        let e_resid = fit_resid(&es, 2);
        let mut j_resid = 0.0_f64;
        for mu in 0..n {
            for nu in 0..n {
                let col: Vec<f64> = js.iter().map(|j| j[(mu, nu)]).collect();
                j_resid = j_resid.max(fit_resid(&col, 1));
            }
        }
        assert!(
            e_resid < 1e-9,
            "E_J(D + tX) departs from a quadratic by {e_resid:.3e} Ha: the metric solve \
             is not backward stable (explicit-inverse regression?)"
        );
        assert!(
            j_resid < 1e-10,
            "J(D + tX) departs from linear by {j_resid:.3e}: the metric solve is not \
             backward stable (explicit-inverse regression?)"
        );
    }

    /// Timing demo, run explicitly:
    ///   OPENBLAS_NUM_THREADS=1 cargo test -p ferric-scf --lib \
    ///     df_j_timing_chunked_vs_naive_demo -- --ignored --nocapture
    /// Min-of-3 wall time for the rayon-chunked `build` vs a naive per-block
    /// unchunked two-pass contraction (the pre-restructure algorithm, same
    /// code path exercised for correctness by
    /// `df_j_chunked_matches_naive_contraction`) on benzene/cc-pVDZ with
    /// cc-pVDZ-RI aux — large enough (na=132, naux~450) that the aux
    /// dimension spans several rayon chunks, unlike the 24-bf water fixture
    /// used by the other unit tests.
    #[test]
    #[ignore]
    fn df_j_timing_chunked_vs_naive_demo() {
        use std::time::Instant;
        let mol = Molecule::load_xyz("../../testdata/molecules/benzene.xyz").unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let op = Operator::coulomb();
        let n = obs.nbasis();

        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                d[(i, j)] = 0.01 * (((i * 7 + j * 3) % 11) as f64);
            }
        }
        let d = 0.5 * (&d + &d.t());

        let mut dfj = DfJ::new(op, &obs, &dfbs, usize::MAX).unwrap();
        let naux = dfj.source.naux();
        println!("benzene/cc-pVDZ: nbf={n}, naux={naux}");

        // "After": current rayon-chunked build.
        let mut best_after = f64::MAX;
        for _ in 0..3 {
            let mut j = Array2::zeros((n, n));
            let t0 = Instant::now();
            dfj.build(&d, &mut j).unwrap();
            best_after = best_after.min(t0.elapsed().as_secs_f64());
        }

        // "Before": naive per-block, unchunked two-pass contraction (the
        // pre-restructure serial GEMV chain this task replaced).
        let d_flat = d.view().into_shape_with_order(n * n).unwrap();
        let mut best_before = f64::MAX;
        for _ in 0..3 {
            let t0 = Instant::now();
            let mut d_p_ref = ndarray::Array1::<f64>::zeros(naux);
            dfj.source
                .for_each_block(|blk| {
                    let b = blk.data.shape()[0];
                    let flat = blk.data.into_shape_with_order((b, n * n)).unwrap();
                    let part = flat.dot(&d_flat);
                    d_p_ref
                        .slice_mut(ndarray::s![blk.p0..blk.p0 + b])
                        .assign(&part);
                    Ok(())
                })
                .unwrap();
            let c_p_ref = dfj.solve_metric(&d_p_ref).unwrap();
            let mut j_ref = Array2::<f64>::zeros((n, n));
            {
                let mut j_flat = j_ref.view_mut().into_shape_with_order(n * n).unwrap();
                dfj.source
                    .for_each_block(|blk| {
                        let b = blk.data.shape()[0];
                        let flat = blk.data.into_shape_with_order((b, n * n)).unwrap();
                        let c_blk = c_p_ref.slice(ndarray::s![blk.p0..blk.p0 + b]);
                        let contrib = flat.t().dot(&c_blk);
                        j_flat += &contrib;
                        Ok(())
                    })
                    .unwrap();
            }
            best_before = best_before.min(t0.elapsed().as_secs_f64());
        }

        println!(
            "DF-J build, benzene/cc-pVDZ (min-of-3): naive={:.4}s  chunked={:.4}s  speedup={:.2}x",
            best_before,
            best_after,
            best_before / best_after
        );
    }

    /// Pair-screened DF-J fixture: a six-carbon chain in def2-SVP with the JK-fit
    /// aux basis, screened at `thresh` (loose enough that pairs ARE dropped).
    struct PairFixture {
        obs: PreparedBasis,
        dfbs: PreparedBasis,
        mol: Molecule,
        op: Operator,
    }

    fn pair_fixture() -> PairFixture {
        let mol = alkane(6);
        let obs = PreparedBasis::new(&mol, &basis::bundled("def2-svp").unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        PairFixture {
            obs,
            dfbs,
            mol,
            op: Operator::coulomb(),
        }
    }

    impl PairFixture {
        fn dfj(&self, budget: usize, thresh: f64) -> DfJ<'static> {
            DfJ::new_for_jk(
                self.op, &self.mol, &self.obs, &self.dfbs, budget, None, thresh,
            )
            .unwrap()
        }
    }

    /// `kept[μ][ν]` for the screen's stored pairs, symmetric.
    fn kept_mask(screen: &PairScreen, n: usize) -> Array2<bool> {
        let mut m = Array2::from_elem((n, n), false);
        for &(mu, nu) in screen.cols() {
            m[(mu as usize, nu as usize)] = true;
            m[(nu as usize, mu as usize)] = true;
        }
        m
    }

    /// Dense RI-J from the one-shot `eri3_tensor`, over the pairs `mask` admits:
    /// `d_P = Σ B[P,μν] D[μν]`, `c = V⁻¹ d`, `J = Σ_P B[P] c_P`. Returns `(J, c)`.
    fn dense_rij(
        dfj: &DfJ<'_>,
        b: &ndarray::Array3<f64>,
        d: &Array2<f64>,
        mask: &Array2<bool>,
    ) -> (Array2<f64>, Array1<f64>) {
        let (naux, n, _) = b.dim();
        let dm = Array2::from_shape_fn((n, n), |(i, k)| if mask[(i, k)] { d[(i, k)] } else { 0.0 });
        let d_p = Array1::from_shape_fn(naux, |p| (&b.index_axis(ndarray::Axis(0), p) * &dm).sum());
        let c = dfj.solve_metric(&d_p).unwrap();
        let mut j = Array2::<f64>::zeros((n, n));
        for p in 0..naux {
            j.scaled_add(c[p], &b.index_axis(ndarray::Axis(0), p));
        }
        j.zip_mut_with(mask, |x, &k| {
            if !k {
                *x = 0.0;
            }
        });
        (j, c)
    }

    /// The stored rows are the unscreened integrals, bit for bit, at exactly
    /// `ncols · naux · 8` bytes, and J is the dense contraction restricted to
    /// the kept pairs (to summation-order rounding).
    #[test]
    fn pair_screened_j_is_the_dense_contraction_over_the_kept_pairs() {
        let f = pair_fixture();
        let mut dfj = f.dfj(usize::MAX, 1e-6);
        let n = f.obs.nbasis();
        let b = ferric_integrals::threeindex::eri3_tensor(f.op, &f.obs, &f.dfbs).unwrap();
        let (data, screen) = dfj.source.pair_screened().expect("pair-screened tier");
        assert!(
            screen.ncols() < screen.total_pairs(),
            "nothing dropped: a vacuous fixture"
        );
        assert_eq!(
            data.len() * 8,
            ThreeIndexSource::pair_screened_bytes(f.dfbs.nbasis(), screen)
        );
        for (k, &(mu, nu)) in screen.cols().iter().enumerate() {
            for p in 0..f.dfbs.nbasis() {
                assert_eq!(
                    data[(p, k)].to_bits(),
                    b[(p, mu as usize, nu as usize)].to_bits(),
                    "stored (P={p}|{mu},{nu}) differs from the unscreened integral"
                );
            }
        }
        let mask = kept_mask(screen, n);
        let d = dense_density(n, false);
        let (j_ref, _) = dense_rij(&dfj, &b, &d, &mask);
        let mut j = Array2::<f64>::zeros((n, n));
        dfj.build(&d, &mut j).unwrap();
        let scale = j_ref.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        let err = (&j - &j_ref).iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        assert!(
            err <= 1e-12 * scale,
            "max |J - J_ref| = {err:e} (scale {scale:e})"
        );
        assert!(mask.iter().zip(j.iter()).all(|(&k, &x)| k || x == 0.0));
    }

    /// Every dropped pair's unscreened J obeys the documented bound
    /// `|J_μν| <= thresh · Σ_P |c_P| · Q_P / max_P Q_P`.
    #[test]
    fn dropped_pairs_obey_the_schwarz_bound() {
        let f = pair_fixture();
        let thresh = 1e-6;
        let dfj = f.dfj(usize::MAX, thresh);
        let n = f.obs.nbasis();
        let b = ferric_integrals::threeindex::eri3_tensor(f.op, &f.obs, &f.dfbs).unwrap();
        let (_, screen) = dfj.source.pair_screened().expect("pair-screened tier");
        let kept = kept_mask(screen, n);
        let all = Array2::from_elem((n, n), true);
        let d = dense_density(n, true);
        let (j_full, c) = dense_rij(&dfj, &b, &d, &all);
        let q = coulomb_metric_2c(f.op, &f.dfbs)
            .unwrap()
            .diag()
            .mapv(|x| x.abs().sqrt());
        let q_max = q.iter().fold(0.0_f64, |m, &x| m.max(x));
        let bound = thresh
            * c.iter()
                .zip(&q)
                .map(|(c, q)| c.abs() * q / q_max)
                .sum::<f64>();
        let mut worst = 0.0_f64;
        for ((i, k), &x) in j_full.indexed_iter() {
            if !kept[(i, k)] {
                worst = worst.max(x.abs());
            }
        }
        assert!(
            worst > 0.0,
            "no dropped pair carries any J: a vacuous check"
        );
        assert!(
            worst <= bound,
            "dropped |J| {worst:e} exceeds the bound {bound:e}"
        );
    }

    /// `pair_thresh <= 0` keeps every pair and gives the unscreened build's bits.
    #[test]
    fn zero_pair_threshold_is_byte_identical_to_the_unscreened_build() {
        let f = pair_fixture();
        let n = f.obs.nbasis();
        let d = dense_density(n, false);
        let mut off = f.dfj(usize::MAX, 0.0);
        assert!(off.source.pair_screened().is_none());
        let mut plain = DfJ::new(f.op, &f.obs, &f.dfbs, usize::MAX).unwrap();
        let (mut a, mut z) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
        off.build(&d, &mut a).unwrap();
        plain.build(&d, &mut z).unwrap();
        assert!(a
            .iter()
            .zip(z.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    /// The in-core decision is made on the COMPRESSED size: a budget between
    /// the compressed and the packed footprint keeps the screened tensor in
    /// core, while the unscreened one cannot stay there.
    #[test]
    fn in_core_decision_uses_the_compressed_size() {
        let f = pair_fixture();
        let (naux, n) = (f.dfbs.nbasis(), f.obs.nbasis());
        let screen = f
            .dfj(usize::MAX, 1e-6)
            .source
            .pair_screened()
            .unwrap()
            .1
            .clone();
        let compressed = ThreeIndexSource::pair_screened_bytes(naux, &screen);
        let packed = naux * n * (n + 1) / 2 * 8;
        assert!(compressed < packed);
        let budget = (compressed + packed) / 2;
        let screened = f.dfj(budget, 1e-6);
        assert!(
            screened.source.pair_screened().is_some(),
            "compressed tensor left core"
        );
        let unscreened = f.dfj(budget, 0.0);
        assert!(unscreened.source.pair_screened().is_none());
        assert!(
            !unscreened.source.is_incore() && !unscreened.source.is_packed_incore_for_test(),
            "the full tensor should not fit this budget"
        );
    }
}
