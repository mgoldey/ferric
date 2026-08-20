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
//!   Pass 1: d_P = Σ_μν B[P,μν] D[μν]. Each block is split into rayon-parallel
//!   aux-chunks; every chunk writes its GEMV result into a *disjoint* slice of
//!   `d_p` (indexed by aux row), so this pass needs no reduction at all — the
//!   result is bit-identical and thread-count independent by construction
//!   (disjoint writes, not summed partials).
//!
//!   Pass 2: J[μν] = Σ_P B[P,μν] c_P is a true reduction (every P contributes
//!   to the same μν output), so it uses `crate::reduce::grouped_deterministic_sum`
//!   — the same bounded, thread-count-independent accumulator DF-K uses — over
//!   the same rayon aux-chunks.
//!
//! Block streaming (and its disk IO on the spill path) stays sequential in
//! `for_each_block`, exactly as DF-K; only the in-memory contraction of the
//! current block fans out over chunks.

use crate::fock::JBuilder;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::operator::Operator;
use ferric_integrals::threeindex::coulomb_metric_2c;
use ferric_integrals::three_index_source::ThreeIndexSource;
use ndarray::Array2;
use ndarray_linalg::Inverse;
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
    /// (naux, naux) inverse Coulomb metric V^{-1}. FULL (all ranks hold it — it
    /// is small, O(naux²), and the V^{-1} d_P multiply couples all P).
    v_inv: Array2<f64>,
    /// Parallel context for the cross-rank reductions. `None` → serial / non-MPI
    /// (no reduction, full band).
    #[allow(dead_code)]
    ctx: Option<&'a ParallelContext>,
    /// The memory budget this source was built under — also caps the pass-2
    /// reduction band scratch via `resolve_band_bytes`.
    budget_bytes: usize,
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
    /// indices and slices `v_inv` with them, so a mismatched band would silently
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
        // NOT wrapped in with_blas_threads, deliberately: `.inv()` is
        // LU-based (dgetrf/dgetri). Verified 2026-07-10 that OpenBLAS's
        // multi-threaded dgetrf_parallel overflows the stack even OUTSIDE any
        // rayon region — reproduced in isolation on a plain OS thread (8 MB
        // ulimit -s) at a realistic (113, 113) JK-fit metric size with
        // FERRIC_BLAS_THREADS=2 (SIGABRT "stack overflow, aborting"); the
        // unwrapped call at the same size succeeds. This is the same
        // dgetrf_parallel hazard the openblas-rayon-dgetrf-crash memory
        // documents for rayon workers, just proven to also hit plain threads
        // — `.cholesky()`/`.eigh()` at comparable sizes were NOT observed to
        // crash, so this is specific to the LU/getri path, not a general BLAS
        // thread-count issue. Stays at the OPENBLAS_NUM_THREADS=1 process
        // default (correct, safe, and identical to pre-B6 behavior).
        let v_inv = v.inv().map_err(|e| FerricError::Lapack(format!("V^-1 in DfJ: {e}")))?;
        // Store ctx only when it actually implies a reduction (>1 rank); a size-1
        // ctx behaves exactly like None (full band, no all_reduce).
        let ctx = ctx.filter(|c| c.size > 1);
        Ok(DfJ { source, v_inv, ctx, budget_bytes })
    }
}

/// Aux-chunk width for one block's rayon fan-out. Mirrors DF-K's sizing
/// (`4096 / n`, clamped to [4, 64]): wide enough that each chunk's GEMV runs a
/// non-trivial contraction even for small n, capped so per-chunk scratch stays
/// modest for large n.
fn chunk_width(n: usize) -> usize {
    (4096 / n.max(1)).clamp(4, 64)
}

impl JBuilder for DfJ<'_> {
    fn build(&mut self, d: &Array2<f64>, j: &mut Array2<f64>) -> Result<usize, FerricError> {
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
                d_p.slice_mut(ndarray::s![blk.p0 + q0..blk.p0 + q1]).assign(&part);
            }
            Ok(())
        })?;

        // MPI: this rank filled only its aux band of `d_P` (disjoint bands across
        // ranks, zero elsewhere). Sum the partials so every rank holds the FULL
        // `d_P` before the global V^{-1} coupling. Cheap: a naux-vector reduce.
        #[cfg(feature = "mpi")]
        if let Some(ctx) = self.ctx {
            use mpi::traits::CommunicatorCollectives;
            if let Some(world) = ctx.world() {
                let mut d_p_global = ndarray::Array1::<f64>::zeros(naux);
                world.all_reduce_into(
                    d_p.as_slice().unwrap(),
                    d_p_global.as_slice_mut().unwrap(),
                    mpi::collective::SystemOperation::sum(),
                );
                d_p = d_p_global;
            }
        }

        // c_P = V^{-1} d_P  (full d_P → identical c_P on every rank)
        let c_p = self.v_inv.dot(&d_p);

        // Pass 2: J[μν] = Σ_P B[P,μν] c_P. This IS a true reduction (every P
        // contributes to every μν), so accumulate via grouped_deterministic_sum
        // — the same bounded, thread-count-independent accumulator DF-K uses —
        // over rayon-parallel aux-chunks within each block.
        j.fill(0.0);
        self.source.for_each_block(|blk| {
            let b = blk.data.shape()[0];
            let flat = blk
                .data
                .into_shape_with_order((b, n * n))
                .map_err(|e| FerricError::General(format!("blk reshape: {e}")))?;
            let c_blk_full = c_p.slice(ndarray::s![blk.p0..blk.p0 + b]);
            let n_chunks = b.div_ceil(chunk);
            let band_bytes = crate::reduce::resolve_band_bytes(self.budget_bytes);
            crate::reduce::grouped_deterministic_sum(
                j,
                n_chunks,
                n,
                band_bytes,
                |ci| -> Result<Array2<f64>, FerricError> {
                    let q0 = ci * chunk;
                    let q1 = (q0 + chunk).min(b);
                    let sub = flat.slice(ndarray::s![q0..q1, ..]);
                    let c_sub = c_blk_full.slice(ndarray::s![q0..q1]);
                    // contrib[μν] = Σ_{P in chunk} B[P,μν] c_P — one
                    // (n*n, c)×(c,) GEMV.
                    let contrib_flat = sub.t().dot(&c_sub); // (n*n,)
                    let contrib = contrib_flat
                        .into_shape_with_order((n, n))
                        .map_err(|e| FerricError::General(format!("contrib reshape: {e}")))?;
                    Ok(contrib)
                },
            )?;
            Ok(())
        })?;

        // MPI: Pass 2 accumulated only this rank's band contribution to J. Sum
        // the per-rank partial J matrices → the full J on every rank.
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

    #[test]
    fn df_j_matches_direct_j_within_fit_error() {
        let mol = Molecule::parse_xyz(
            "3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n",
            0,
            1,
        )
        .unwrap();
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

        let max_diff: f64 = (&j_df - &j_direct).iter().map(|v| v.abs()).fold(0.0, f64::max);
        // RI-fit basis is tuned for correlation, not J — accept ~1e-3 Ha-scale error.
        assert!(max_diff < 5e-3, "DF-J vs direct-J max diff = {} too large", max_diff);
    }

    #[test]
    fn df_j_source_backed_matches_incore() {
        let mol = Molecule::parse_xyz(
            "3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n",
            0,
            1,
        )
        .unwrap();
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

        let maxdiff = (&j_big - &j_small).iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(maxdiff < 1e-10, "spill J != in-core J, maxdiff={maxdiff}");
    }

    #[test]
    fn df_j_chunked_matches_naive_contraction() {
        // Implementation-equivalence test for the rayon-chunked restructure:
        // the chunked GEMV contraction in `build` must reproduce the naive
        // per-block (unchunked) two-pass contraction over the SAME raw B
        // tensor and V^{-1} to machine precision. A dense symmetric density
        // exercises every (P,μν) coupling, unlike the diagonal-D accuracy
        // test above (which measures RI fitting error vs direct J and cannot
        // separate algebra bugs from fitting error). Mirrors
        // `df_k_wide_gemm_matches_naive_contraction`.
        let mol = Molecule::parse_xyz(
            "3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n",
            0,
            1,
        )
        .unwrap();
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
                d_p_ref.slice_mut(ndarray::s![blk.p0..blk.p0 + b]).assign(&part);
                Ok(())
            })
            .unwrap();
        let c_p_ref = dfj.v_inv.dot(&d_p_ref);
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

    #[test]
    fn df_j_bit_identical_across_thread_counts() {
        // Regression guard mirroring df_k_bit_identical_across_thread_counts:
        // Pass 1's chunked GEMV writes disjoint slices of d_p (no reduction),
        // and Pass 2's chunked GEMV reduction goes through
        // grouped_deterministic_sum, so the J matrix (and hence the SCF
        // energy) must be bit-identical regardless of RAYON_NUM_THREADS.
        // Uses several heavy atoms so the aux dimension spans multiple
        // chunks (making order actually matter).
        let mol = Molecule::parse_xyz(
            "3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n",
            0,
            1,
        )
        .unwrap();
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
        let mol = Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1)
            .unwrap();
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
        assert_eq!(j1, j2, "narrow-band DfJ must be bit-identical at 1 vs 2 threads");
        assert_eq!(j1, j8, "narrow-band DfJ must be bit-identical at 1 vs 8 threads");
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
                    d_p_ref.slice_mut(ndarray::s![blk.p0..blk.p0 + b]).assign(&part);
                    Ok(())
                })
                .unwrap();
            let c_p_ref = dfj.v_inv.dot(&d_p_ref);
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
}
