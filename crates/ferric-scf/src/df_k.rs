//! Density-fitted exchange matrix builder (DF-K / RI-K).
//!
//! Replaces the O(N^4) direct ERI exchange build with O(N^3 · naux) GEMMs using
//! the same 3-center auxiliary expansion as DF-J. For closed-shell RHF:
//!
//!   K_{μν} = Σ_{λσ} (μλ|νσ) D_{λσ}
//!          ≈ Σ_P Σ_{λσ} B^P_{μλ} B^P_{νσ} D_{λσ}        (RI with V^{-1/2}-dressed B)
//!
//! Computed as two passes over (P,μ,ν):
//!   Z[P,μ,σ] = Σ_λ B[P,μ,λ] · D[λ,σ]
//!   K[μ,ν]   = Σ_{P,σ} Z[P,μ,σ] · B[P,ν,σ]
//!
//! Each aux-block is contracted in rayon-parallel aux-chunks; BLAS stays
//! single-threaded (OPENBLAS_NUM_THREADS=1 under rayon), so chunks are the
//! parallel unit. The two builders differ in how a chunk is contracted:
//!
//! * [`KBuilder::build`] (density path) collapses the per-P (n,n)×(n,n) GEMM
//!   stack into two wide GEMMs on repacked operands — O(naux·n³).
//! * [`KBuilder::build_from_occ`] (occupied path, used once MOs exist) never
//!   repacks: B[P] is already a contiguous symmetric (n,n) slice, so each P is
//!   one DSYMM into a stacked (n, c·nocc) panel, reduced by ONE DSYRK per chunk
//!   — O(naux·n²·nocc), with both symmetries taken. At benzene/aug-cc-pVTZ
//!   (n=414, nocc=21) that is ~12.8x faster than the density path per build.
//!   The repack the density path amortizes over an n-wide contraction was 35%
//!   of the occ path's runtime, since its contraction is only nocc-wide.
//!
//! The dressed 3-center tensor B[P,μ,ν] = Σ_Q V^{-1/2}_{PQ} (Q|μν) is built once
//! at construction (via ThreeIndexSource::build_dressed) and reused every SCF
//! iteration.  The dressed source is budget-bounded: in-core when
//! naux·nao²·8 ≤ budget_bytes, else aux-blocked disk-spill.
//!
//! Accuracy of DF-K depends critically on the auxiliary basis: use a JK-fit
//! basis (e.g. `def2-universal-jkfit`), not an RI/MP2-fit basis.

use crate::fock::KBuilder;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::ThreeIndexSource;
use ferric_integrals::threeindex::coulomb_metric_2c;
use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, Array3};
use ndarray_linalg::{Eigh, UPLO};

// Direct BLAS bindings for the occupied-path exchange build. OpenBLAS is already
// linked via openblas-src; these avoid a cblas/blas-sys dep for two calls.
// Fortran signatures:
//   dsymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
//   dsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
extern "C" {
    fn dsymm_(
        side: *const u8,
        uplo: *const u8,
        m: *const i32,
        n: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        b: *const f64,
        ldb: *const i32,
        beta: *const f64,
        c: *mut f64,
        ldc: *const i32,
    );
    fn dsyrk_(
        uplo: *const u8,
        trans: *const u8,
        n: *const i32,
        k: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        beta: *const f64,
        c: *mut f64,
        ldc: *const i32,
    );
}

/// DF-K exchange builder. Caches the V^{-1/2}-dressed 3-center source; the
/// per-chunk repack/GEMM scratch in `build` is allocated per rayon task
/// (O(chunk·n²), negligible against the O(chunk·n³) GEMM it feeds).
///
/// ## MPI aux-band striping
/// Under MPI (`new_banded` with a multi-rank [`ParallelContext`]), each rank
/// dresses and holds ONLY its own contiguous aux band `[band_p0, band_p1)` of
/// B[P,μ,ν] — the resident B footprint scales with rank count. Because
/// K = Σ_P B_P D B_Pᵀ is a plain sum over P, each rank's band-restricted `build`
/// yields a partial K and one final all_reduce sums them. (Dressing a band still
/// reads all Q of the raw tensor — that raw is budget-bounded / spillable, so it
/// does not defeat the resident-memory scaling.) With one rank (or the `mpi`
/// feature off) the band is `0..naux` and the reduction is skipped, so behavior
/// is byte-identical to the serial path.
pub struct DfK<'a> {
    /// Budget-bounded dressed 3-center source B[P,μ,ν] = Σ_Q V^{-1/2}_{PQ} (Q|μν)
    /// — holds only this rank's aux band `[band_p0, band_p1)`.
    dressed: ThreeIndexSource,
    /// Parallel context for the final cross-rank K reduction. `None` → serial /
    /// non-MPI (no reduction, full band).
    #[allow(dead_code)]
    ctx: Option<&'a ParallelContext>,
    /// The memory budget this source was built under — also caps the K
    /// reduction band scratch via `resolve_band_bytes`.
    budget_bytes: usize,
}

impl<'a> std::fmt::Debug for DfK<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DfK")
            .field("budget_bytes", &self.budget_bytes)
            .finish_non_exhaustive()
    }
}

/// V^{-1/2} via symmetric eigendecomposition with canonical orthogonalization.
/// The 2-center metric `(P|w(r12)|Q)` is positive-definite analytically, but
/// for range-separated operators (erf, erfc) with JK-fit aux on heavy atoms,
/// some eigenvalues can be near-zero and turn slightly negative under
/// floating-point roundoff. Drop those modes — equivalent to PySCF's
/// `lindep` threshold in `df.aux_e2`.
fn v_inv_sqrt_lindep(v: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    let naux = v.nrows();
    // One-time (naux, naux) setup factorization, called once per SCF
    // construction outside any rayon region. Opt-in BLAS raise via
    // FERRIC_BLAS_THREADS (default 1, unchanged behavior);
    // opt_in_blas_threads()'s rayon-worker self-guard also covers any caller
    // reached from a single-thread rayon pool (e.g. free-atom SAD), resolving
    // to 1 there regardless of the env var.
    let (evals, evecs) = with_blas_threads(opt_in_blas_threads(), || v.eigh(UPLO::Upper))
        .map_err(|e| FerricError::Lapack(format!("V eigh in DfK: {e}")))?;
    const LINDEP_THRESH: f64 = 1e-10;
    let mut u_scaled = evecs.clone();
    let mut n_dropped: usize = 0;
    for k in 0..naux {
        if evals[k] < LINDEP_THRESH {
            // Zero out this column so its (column-vector outer product) contributes
            // nothing to V^{-1/2}.
            for r in 0..naux {
                u_scaled[(r, k)] = 0.0;
            }
            n_dropped += 1;
        } else {
            let s = 1.0 / evals[k].sqrt();
            for r in 0..naux {
                u_scaled[(r, k)] *= s;
            }
        }
    }
    // Silent on n_dropped: this is expected for range-separated operators
    // (erf, erfc) with JK-fit aux on heavy atoms and is benign.
    let _ = n_dropped;
    // Same one-time-setup opt-in raise as the eigh above.
    Ok(with_blas_threads(opt_in_blas_threads(), || {
        u_scaled.dot(&evecs.t())
    })) // (naux, naux)
}

impl<'a> DfK<'a> {
    /// Build the DF-K cache from orbital and auxiliary bases (FULL aux range,
    /// serial / single-rank — byte-identical to the pre-MPI path).
    ///
    /// Computes V^{-1/2} = U · diag(λ^{-1/2}) · U^T from the symmetric eigendecomp
    /// of the (P|Q) Coulomb metric, then forms B[P,μ,ν] = Σ_Q V^{-1/2}_{PQ} (Q|μν)
    /// via ThreeIndexSource::build_dressed, honouring `budget_bytes`.
    pub fn new(
        op: Operator,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        budget_bytes: usize,
    ) -> Result<Self, FerricError> {
        Self::new_banded(op, obs, dfbs, budget_bytes, None)
    }

    /// Build the DF-K cache, striping the aux band across the MPI ranks of `ctx`.
    /// When `ctx` is `None` or `ctx.size == 1`, this is exactly [`DfK::new`]
    /// (full band, no reduction). Otherwise this rank dresses/holds only its band
    /// `ctx.aux_band(naux)`, and `build` all_reduces the partial K.
    pub fn new_banded(
        op: Operator,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        budget_bytes: usize,
        ctx: Option<&'a ParallelContext>,
    ) -> Result<Self, FerricError> {
        let v = coulomb_metric_2c(op, dfbs)?;
        let v_inv_sqrt = v_inv_sqrt_lindep(&v)?;

        let naux = dfbs.nbasis();
        let (p0, p1) = match ctx {
            Some(c) => c.aux_band(naux),
            None => (0, naux),
        };

        // Build the FULL raw source (dressing a band sums over all Q). The raw is
        // budget-bounded / spillable, so its full footprint need not be resident;
        // only this rank's dressed BAND is retained.
        let mut raw = ThreeIndexSource::build(op, obs, dfbs, budget_bytes)?;
        Self::from_raw(&mut raw, dfbs, v_inv_sqrt, budget_bytes, ctx, p0, p1)
    }

    /// Same as [`DfK::new_banded`] but dresses an ALREADY-BUILT full raw
    /// `(P|μν)` source instead of generating one.
    ///
    /// Exists so a DF-JK run computes the raw 3-index tensor ONCE: `DfJ` and
    /// `DfK` are otherwise independent and each called `ThreeIndexSource::build`
    /// on the identical `(op, obs, dfbs)`, generating ~95.6M integrals twice
    /// (~600 ms each at benzene/aug-cc-pVTZ). `raw` is consumed — DfK only needs
    /// it to produce the dressed tensor and drops it before the SCF loop —
    /// so the caller must hand over a source it no longer needs, or clone.
    ///
    /// `raw` MUST span the full aux range `[0, naux)`: dressing any band sums
    /// over all Q. `build_dressed_band` asserts this.
    pub fn from_full_raw(
        raw: &mut ThreeIndexSource,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        op: Operator,
        budget_bytes: usize,
        ctx: Option<&'a ParallelContext>,
    ) -> Result<Self, FerricError> {
        let _ = obs;
        let v = coulomb_metric_2c(op, dfbs)?;
        let v_inv_sqrt = v_inv_sqrt_lindep(&v)?;
        let naux = dfbs.nbasis();
        let (p0, p1) = match ctx {
            Some(c) => c.aux_band(naux),
            None => (0, naux),
        };
        Self::from_raw(raw, dfbs, v_inv_sqrt, budget_bytes, ctx, p0, p1)
    }

    /// Shared tail of both constructors: dress `raw` into this rank's band.
    fn from_raw(
        raw: &mut ThreeIndexSource,
        dfbs: &PreparedBasis,
        v_inv_sqrt: Array2<f64>,
        budget_bytes: usize,
        ctx: Option<&'a ParallelContext>,
        p0: usize,
        p1: usize,
    ) -> Result<Self, FerricError> {
        let naux = dfbs.nbasis();
        let dressed = if p0 == 0 && p1 == naux {
            // Full band: identical call to the serial path (byte-identical B).
            ThreeIndexSource::build_dressed(raw, &v_inv_sqrt, budget_bytes)?
        } else {
            ThreeIndexSource::build_dressed_band(raw, &v_inv_sqrt, budget_bytes, p0, p1)?
        };
        // The caller owns `raw`; DfK retains only the dressed tensor.

        let ctx = ctx.filter(|c| c.size > 1);
        Ok(DfK { dressed, ctx, budget_bytes })
    }
}

impl DfK<'_> {
    /// Density-based exchange contraction (O(naux·n³)); see [`KBuilder::build`].
    /// Kept as an inherent method so both the trait `build` and the C_occ-based
    /// `build_from_occ` share one struct-level definition.
    fn build_impl(&mut self, d: &Array2<f64>, k: &mut Array2<f64>) -> Result<usize, FerricError> {
        let n = self.dressed.nao();
        k.fill(0.0);
        // Aux-chunk width: wide enough that the (n, c·n)×(c·n, n) contraction
        // below runs at BLAS3 efficiency even for small n, capped so per-task
        // scratch (3·c·n² doubles) stays modest for large n.
        let chunk = (4096 / n.max(1)).clamp(4, 64);
        self.dressed.for_each_block(|blk| {
            let b = blk.data.shape()[0];
            // Contract this block in rayon-parallel aux-chunks, each collapsing
            // its per-P GEMM stack into two wide GEMMs. Block streaming (and its
            // disk IO on the spill path) stays sequential in for_each_block; only
            // the in-memory contraction of the current block fans out.
            // Compute each aux-chunk's K contribution in rayon-parallel, then
            // fold into `k` in strict chunk order. A rayon `try_reduce` combines
            // partials in a tree whose shape depends on the worker count, so
            // floating-point non-associativity made the K matrix (and thus the
            // SCF energy) vary with RAYON_NUM_THREADS by ~µHa. Collecting *all*
            // per-chunk partials then serial-summing pins the order but holds
            // (b/chunk)·n²·8 bytes live at once — the DF-K scaling hazard
            // (~97 GB at 50-atom/aug-cc-pVTZ). `grouped_deterministic_sum`
            // processes the chunks in byte-budgeted bands, summing each band in
            // chunk order before the next: same ascending-chunk fold order (so
            // still bit-identical across thread counts), but the live set is one
            // band (≤512 MiB), not every chunk.
            let n_chunks = b.div_ceil(chunk);
            // Defect C: `band_width` budgets ONLY the `kc` partials it collects
            // (n² each). The closure below additionally holds, PER RAYON
            // WORKER, three c·n² buffers — `bswap`/`bswap_flat`, `zt`/`zt_wide`,
            // and `bt`/`bt_flat` (each pair is one allocation: the `_flat`
            // reshape consumes its source rather than copying). That is the
            // `3·c·n²` this function's own comment above names, and until now
            // it never reached the sizing: at n = 2000, c = 4, 12 threads it is
            // ~4.6 GB entirely outside the budget. Charge it before sizing the
            // band, so the one knob covers the whole live set.
            let workers = rayon::current_num_threads().max(1).min(n_chunks.max(1));
            let band_bytes = crate::reduce::band_bytes_after_worker_scratch(
                crate::reduce::resolve_band_bytes(self.budget_bytes),
                3 * chunk * n * n,
                workers,
            );
            crate::reduce::grouped_deterministic_sum(
                k,
                n_chunks,
                n,
                band_bytes,
                |ci| -> Result<Array2<f64>, FerricError> {
                    let q0 = ci * chunk;
                    let q1 = (q0 + chunk).min(b);
                    let c = q1 - q0;
                    let bchunk = blk.data.slice(ndarray::s![q0..q1, .., ..]);

                    // Zt[μ,P,σ] = Σ_λ B[P,μ,λ] · D[λ,σ] as one (c·n, n)×(n, n)
                    // GEMM on the (μ,P,λ)-repacked chunk, so (P,σ) comes out as
                    // the contiguous trailing axis pair needed below.
                    let mut bswap = Array3::<f64>::zeros((n, c, n));
                    bswap.assign(&bchunk.permuted_axes([1, 0, 2]));
                    let bswap_flat = bswap
                        .into_shape_with_order((n * c, n))
                        .map_err(|e| FerricError::General(format!("B repack reshape: {e}")))?;
                    let mut zt = Array2::<f64>::zeros((n * c, n));
                    general_mat_mul(1.0, &bswap_flat, d, 0.0, &mut zt);
                    let zt_wide = zt
                        .into_shape_with_order((n, c * n))
                        .map_err(|e| FerricError::General(format!("Z reshape: {e}")))?;

                    // K_chunk[μ,ν] = Σ_{P,σ} Zt[μ,(P,σ)] · Bt[(P,σ),ν] with
                    // Bt[P,σ,ν] = B[P,ν,σ]: one wide (n, c·n)×(c·n, n) GEMM
                    // instead of c separate (n,n)×(n,n) ones.
                    let mut bt = Array3::<f64>::zeros((c, n, n));
                    bt.assign(&bchunk.permuted_axes([0, 2, 1]));
                    let bt_flat = bt
                        .into_shape_with_order((c * n, n))
                        .map_err(|e| FerricError::General(format!("Bt reshape: {e}")))?;
                    let mut kc = Array2::<f64>::zeros((n, n));
                    general_mat_mul(1.0, &zt_wide, &bt_flat, 0.0, &mut kc);
                    Ok(kc)
                },
            )?;
            Ok(())
        })?;

        // MPI: `build` accumulated only this rank's aux-band contribution to K
        // (K = Σ_P B_P D B_Pᵀ is a plain sum over P, and bands are disjoint).
        // Sum the per-rank partial K matrices → the full K on every rank.
        self.reduce_k_across_ranks(k);

        Ok(0)
    }

    /// Build K from occupied MO coefficients via an O(naux·n²·nocc)
    /// half-transform, replacing the O(naux·n³) density contraction in [`build`].
    ///
    /// With `D = C_occ · C_occᵀ` the two-pass exchange
    ///   K[μ,ν] = Σ_{P,λ,σ} B[P,μ,λ] D[λ,σ] B[P,ν,σ]
    /// factorises through the half-transformed intermediate
    ///   M[P,μ,i] = Σ_λ B[P,μ,λ] C[λ,i]   (the RI-MP2 AO→MO half-transform shape)
    ///   K[μ,ν]  = Σ_{P,i} M[P,μ,i] M[P,ν,i].
    /// Because `nocc ≪ n` at production basis sizes, both passes cost
    /// O(naux·n²·nocc) instead of O(naux·n³) — the whole point of threading
    /// `C_occ` through the API rather than the assembled density.
    ///
    /// Determinism, banding, and the MPI aux-band reduction are handled exactly
    /// as in [`build`] (the grouped ascending-chunk fold pins the fold order
    /// across thread counts; the outer-product `M Mᵀ` is symmetric by
    /// construction).
#[allow(clippy::manual_c_str_literals)] // b"R\0" byte literals match dsymm_'s *const u8 args; c".." would be *const i8
    fn build_from_occ_impl(
        &mut self,
        c_occ: &Array2<f64>,
        k: &mut Array2<f64>,
    ) -> Result<usize, FerricError> {
        let n = self.dressed.nao();
        if c_occ.nrows() != n {
            return Err(FerricError::General(format!(
                "DfK::build_from_occ: c_occ rows ({}) != nao ({})",
                c_occ.nrows(),
                n
            )));
        }
        // Test-only escape hatch: force the O(naux·n³) density path even though
        // C_occ was supplied, so a regression test can run the *identical* SCF
        // both ways and assert the energies agree to the DF-K reassociation
        // floor (see df_k_occ_path_matches_density_path_scf_energy). Never set in
        // production; the fast path is the default whenever C_occ is available.
        if std::env::var_os("FERRIC_DFK_FORCE_DENSITY").is_some() {
            let d = c_occ.dot(&c_occ.t());
            return self.build_impl(&d, k);
        }
        let nocc = c_occ.ncols();
        k.fill(0.0);
        // nocc == 0 (e.g. an empty β channel): K is identically zero. The GEMMs
        // below are well-defined at nocc = 0 but do no work; short-circuit to
        // avoid a zero-width reshape edge case.
        if nocc == 0 {
            self.reduce_k_across_ranks(k);
            return Ok(0);
        }

        // Aux-chunk width: same BLAS3-efficiency vs per-task-scratch tradeoff as
        // `build`, but the half-transform scratch scales with nocc, not n, so the
        // trailing GEMM operand is (c·nocc) wide.
        let chunk = (4096 / n.max(1)).clamp(4, 64);
        self.dressed.for_each_block(|blk| {
            let b = blk.data.shape()[0];
            let n_chunks = b.div_ceil(chunk);
            // Defect C, occupied path: the per-worker scratch here is the
            // half-transform panel `m_chunk` (n × c·nocc), which scales with
            // nocc rather than n — smaller than the density path's 3·c·n², but
            // equally absent from the sizing before this. `kc` (n²) is the
            // collected partial and is already covered by `band_width`.
            let workers = rayon::current_num_threads().max(1).min(n_chunks.max(1));
            let band_bytes = crate::reduce::band_bytes_after_worker_scratch(
                crate::reduce::resolve_band_bytes(self.budget_bytes),
                n * chunk * nocc,
                workers,
            );
            crate::reduce::grouped_deterministic_sum(
                k,
                n_chunks,
                n,
                band_bytes,
                |ci| -> Result<Array2<f64>, FerricError> {
                    let q0 = ci * chunk;
                    let q1 = (q0 + chunk).min(b);
                    let c = q1 - q0;
                    let bchunk = blk.data.slice(ndarray::s![q0..q1, .., ..]);

                    // Per-aux-row, repack-free (the shape PySCF uses —
                    // nr_ao2mo.c:1016 / df_jk.py:371): B[P] is already a
                    // contiguous, SYMMETRIC (n,n) slice, so
                    //   M_P[μ,i] = Σ_λ B[P,μ,λ]·C[λ,i]   via DSYMM
                    //   K      += Σ_P M_P · M_Pᵀ         via DSYRK
                    // needs no permuted copy at all. The previous formulation
                    // repacked B to [μ,P,λ] so both contractions could be single
                    // wide GEMMs; measured at benzene/aTZ (n=414, nocc=21) that
                    // repack was 35% of this function's entire runtime — the
                    // density path amortizes it over an n-wide contraction, but
                    // the occ path's is only nocc-wide, so it cannot. Dropping
                    // the repack and taking both symmetries measured 147→66.5 ms
                    // per build at benzene/aTZ (2.2x), lifting this path from 55
                    // to 121 GFLOP/s (benchmarks/harness/examples/dfk_mwe.rs).
                    //
                    // BLAS layout: ndarray is row-major, BLAS is column-major, so
                    // a row-major (r,c) buffer is the Fortran (c,r) matrix.
                    //  - DSYMM with SIDE='R' computes the Fortran (nocc,n)
                    //    product Cᵀ·B, which IS the row-major (n,nocc) M we want.
                    //    B symmetric ⇒ UPLO is immaterial to the result.
                    //  - DSYRK with TRANS='T' on the Fortran (nocc,n) view of M
                    //    gives Mᵀ_F·M_F = the row-major M·Mᵀ. UPLO='L' fills the
                    //    Fortran lower triangle = the row-major UPPER triangle,
                    //    so the mirror below copies upper→lower. (Getting that
                    //    direction backwards yields a silently wrong K — it is
                    //    covered by dfk_occ_symmetry_matches_reference.)
                    let ni = n as i32;
                    let no = nocc as i32;
                    let one = 1.0f64;
                    let zero = 0.0f64;
                    // Half-transform every P of this chunk into one wide
                    // (n, c·nocc) buffer, then reduce with a SINGLE
                    // DSYRK of k = c·nocc. Accumulating one rank-nocc DSYRK per P
                    // instead (beta=1) is the obvious variant but is measurably
                    // LESS accurate: on water/cc-pVDZ it moved the converged SCF
                    // energy 1.6e-8 off the density path, versus 4e-10 for a wide
                    // reduction, because a sequential f64 sum over c updates lacks
                    // the blocking BLAS applies along a long k axis.
                    // W[μ, (P,i)] row-major (n, c·nocc): M_P occupies COLUMNS
                    // [p·nocc, (p+1)·nocc), so each DSYMM writes a strided
                    // sub-block (ldc = c·nocc, column offset p·nocc) — not a
                    // contiguous row range.
                    let ldw = (c * nocc) as i32;
                    let mut m_chunk = Array2::<f64>::zeros((n, c * nocc));
                    for p in 0..c {
                        let bp = bchunk.slice(ndarray::s![p, .., ..]);
                        // `blk.data` is standard-layout, so this row slice is
                        // contiguous; `to_slice` therefore never falls back.
                        let bp_slice = bp.to_slice().ok_or_else(|| {
                            FerricError::General("DfK occ: non-contiguous B row".into())
                        })?;
                        // M_P as the (nocc, n)-Fortran block at column offset
                        // p·nocc of W.
                        let off = p * nocc;
                        // SAFETY: dimensions match the buffers above; the
                        // strided (nocc, n) write at column offset p·nocc with
                        // ldc = c·nocc stays inside the (n, c·nocc) buffer.
                        unsafe {
                            dsymm_(
                                b"R\0".as_ptr(),
                                b"U\0".as_ptr(),
                                &no,
                                &ni,
                                &one,
                                bp_slice.as_ptr(),
                                &ni,
                                c_occ.as_ptr(),
                                &no,
                                &zero,
                                m_chunk.as_mut_ptr().add(off),
                                &ldw,
                            );
                        }
                    }
                    let mut kc = Array2::<f64>::zeros((n, n));
                    let k_dim = (c * nocc) as i32;
                    // SAFETY: m_chunk is (n, c·nocc) row-major = Fortran
                    // (c·nocc, n) with lda = c·nocc; kc is (n, n). TRANS='T'
                    // gives W_Fᵀ·W_F = the row-major W·Wᵀ = Σ_P M_P·M_Pᵀ.
                    unsafe {
                        dsyrk_(
                            b"L\0".as_ptr(),
                            b"T\0".as_ptr(),
                            &ni,
                            &k_dim,
                            &one,
                            m_chunk.as_ptr(),
                            &ldw,
                            &zero,
                            kc.as_mut_ptr(),
                            &ni,
                        );
                    }
                    // DSYRK wrote only one triangle (see the layout note above).
                    for i in 0..n {
                        for j in (i + 1)..n {
                            kc[[j, i]] = kc[[i, j]];
                        }
                    }
                    Ok(kc)
                },
            )?;
            Ok(())
        })?;

        self.reduce_k_across_ranks(k);
        Ok(0)
    }

    /// MPI aux-band reduction shared by [`build`] and [`build_from_occ_impl`]:
    /// each rank holds only its disjoint aux band, so the partial K matrices sum
    /// to the full K on every rank. No-op without the `mpi` feature or a
    /// single-rank context.
    fn reduce_k_across_ranks(&self, k: &mut Array2<f64>) {
        #[cfg(feature = "mpi")]
        if let Some(ctx) = self.ctx {
            use mpi::traits::CommunicatorCollectives;
            if let Some(world) = ctx.world() {
                let mut k_global = Array2::<f64>::zeros(k.dim());
                world.all_reduce_into(
                    k.as_slice().unwrap(),
                    k_global.as_slice_mut().unwrap(),
                    mpi::collective::SystemOperation::sum(),
                );
                *k = k_global;
            }
        }
        #[cfg(not(feature = "mpi"))]
        let _ = k;
    }
}

impl KBuilder for DfK<'_> {
    fn build(&mut self, d: &Array2<f64>, k: &mut Array2<f64>) -> Result<usize, FerricError> {
        self.build_impl(d, k)
    }

    fn build_from_occ(
        &mut self,
        c_occ: &Array2<f64>,
        k: &mut Array2<f64>,
    ) -> Result<usize, FerricError> {
        self.build_from_occ_impl(c_occ, k)
    }

    fn update_density(&mut self, _d: &Array2<f64>) {}

    fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct_k::DirectK;
    use crate::screening::SchwarzBounds;
    use ferric_core::basis;
    use ferric_core::mol::Molecule;
    use ferric_core::parallel::ParallelContext;

    #[test]
    fn df_k_matches_direct_k_with_jkfit() {
        let mol = Molecule::parse_xyz(
            "3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n",
            0,
            1,
        )
        .unwrap();
        let obs_set = basis::bundled("cc-pvdz").unwrap();
        let obs = PreparedBasis::new(&mol, &obs_set).unwrap();
        let dfbs_set = basis::bundled("def2-universal-jkfit").unwrap();
        let dfbs = PreparedBasis::new(&mol, &dfbs_set).unwrap();

        let op = Operator::coulomb();
        let n = obs.nbasis();

        // Simple diagonal mock density
        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n.min(5) {
            d[(i, i)] = 2.0;
        }

        let mut k_direct = Array2::zeros((n, n));
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let ctx = ParallelContext::default();
        let mut dk = DirectK::new(&ctx, &obs, &bounds, 1e-12, usize::MAX);
        <DirectK as KBuilder>::build(&mut dk, &d, &mut k_direct).unwrap();

        let mut k_df = Array2::zeros((n, n));
        let mut dfk = DfK::new(op, &obs, &dfbs, usize::MAX).unwrap();
        dfk.build(&d, &mut k_df).unwrap();

        let max_diff: f64 = (&k_df - &k_direct).iter().map(|v| v.abs()).fold(0.0, f64::max);
        // JK-fit basis should give K accurate to ~1e-3 for this small system.
        assert!(max_diff < 5e-3, "DF-K vs direct-K max diff = {} too large", max_diff);
    }

    #[test]
    fn df_k_bit_identical_across_thread_counts() {
        // Regression guard: the aux-chunk accumulation in `build` must be
        // bit-identical regardless of RAYON_NUM_THREADS. A rayon `try_reduce`
        // combined partials in a worker-count-dependent tree, so the K matrix
        // (and hence the SCF energy) drifted ~µHa between thread counts. The
        // collect-then-serial-sum fix pins the order. Uses several heavy atoms so
        // the aux dimension spans multiple chunks (making order actually matter).
        let mol = Molecule::parse_xyz(
            "3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n",
            0,
            1,
        )
        .unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let n = obs.nbasis();

        // Dense symmetric density that couples every (P,σ) pair.
        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                d[(i, j)] = 0.01 * ((i * 7 + j * 3) % 11) as f64;
            }
        }
        let d = 0.5 * (&d + &d.t());

        let build_k = |threads: usize| -> Array2<f64> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut k = Array2::zeros((n, n));
                let mut dfk = DfK::new(op, &obs, &dfbs, usize::MAX).unwrap();
                dfk.build(&d, &mut k).unwrap();
                k
            })
        };

        let k1 = build_k(1);
        let k4 = build_k(4);
        // Bit-identical: exact equality, not a tolerance.
        assert_eq!(
            k1, k4,
            "DfK build must be bit-identical across thread counts (rayon reduction order leak)"
        );
    }

    #[test]
    fn df_k_bit_identical_across_threads_with_narrow_bands() {
        // Same guarantee as above, but with a deliberately tiny reduce-band
        // budget so the two-level grouped_deterministic_sum path actually splits
        // the aux chunks across several bands. The banding must not perturb the
        // ascending-chunk fold order, so K stays bit-identical at 1/2/8 threads.
        let mol = Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1)
            .unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let n = obs.nbasis();

        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                d[(i, j)] = 0.01 * ((i * 7 + j * 3) % 11) as f64;
            }
        }
        let d = 0.5 * (&d + &d.t());

        // One nbf² partial for cc-pVDZ water is ~24² · 8 ≈ 4.6 kB; a 4 kB band
        // budget forces band width 1 (each aux chunk its own band).
        std::env::set_var("FERRIC_REDUCE_BAND_BYTES", "4096");
        let build_k = |threads: usize| -> Array2<f64> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut k = Array2::zeros((n, n));
                let mut dfk = DfK::new(op, &obs, &dfbs, usize::MAX).unwrap();
                dfk.build(&d, &mut k).unwrap();
                k
            })
        };
        let k1 = build_k(1);
        let k2 = build_k(2);
        let k8 = build_k(8);
        std::env::remove_var("FERRIC_REDUCE_BAND_BYTES");
        assert_eq!(k1, k2, "narrow-band DfK must be bit-identical at 1 vs 2 threads");
        assert_eq!(k1, k8, "narrow-band DfK must be bit-identical at 1 vs 8 threads");
    }

    #[test]
    fn df_k_wide_gemm_matches_naive_contraction() {
        // Implementation-equivalence test for the wide-GEMM restructure: the
        // chunked two-wide-GEMM contraction in `build` must reproduce the naive
        // per-P GEMM stack over the SAME dressed B tensor to machine precision.
        // A dense symmetric density exercises every (P,σ) coupling, unlike the
        // diagonal-D accuracy test above (which measures RI fitting error vs
        // direct K and cannot separate algebra bugs from fitting error).
        let mol = Molecule::parse_xyz(
            "3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n",
            0,
            1,
        )
        .unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let n = obs.nbasis();

        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                // Symmetric, deterministic, modest magnitude.
                d[(i, j)] = 0.02 * (((i * j + 3) % 11) as f64);
            }
            d[(i, i)] += 1.0;
        }

        let mut dfk = DfK::new(op, &obs, &dfbs, usize::MAX).unwrap();

        // Reference: naive per-P contraction (the pre-restructure algorithm)
        // over the same dressed tensor.
        let mut k_ref = Array2::<f64>::zeros((n, n));
        dfk.dressed
            .for_each_block(|blk| {
                let b = blk.data.shape()[0];
                for p in 0..b {
                    let bp = blk.data.slice(ndarray::s![p, .., ..]);
                    let zp = bp.dot(&d);
                    k_ref += &zp.dot(&bp.t());
                }
                Ok(())
            })
            .unwrap();

        let mut k_df = Array2::zeros((n, n));
        dfk.build(&d, &mut k_df).unwrap();

        let max_diff: f64 = (&k_df - &k_ref).iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(
            max_diff < 1e-10,
            "wide-GEMM DF-K vs naive per-P contraction max diff = {} too large",
            max_diff
        );
    }

    /// Determinism demo, run explicitly at several thread counts:
    ///   RAYON_NUM_THREADS=N cargo test -p ferric-scf --lib \
    ///     df_k_scf_energy_determinism_demo -- --ignored --nocapture
    /// Full DF-JK RHF (water/cc-pVDZ, def2-universal-jkfit) on the AMBIENT rayon
    /// pool; prints the total energy to 17 significant digits plus its raw f64
    /// bit pattern. The printed value must be identical at N = 1, 2, 8: DF-J is
    /// serial GEMM and DF-K is the only rayon reduction in this configuration,
    /// so this pins the grouped deterministic accumulation end-to-end through a
    /// real SCF energy.
    #[test]
    #[ignore]
    fn df_k_scf_energy_determinism_demo() {
        use crate::rhf::{solve_rhf, RhfConfig};
        let mol = Molecule::parse_xyz(
            "3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n",
            0,
            1,
        )
        .unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let op = Operator::coulomb();
        let bounds = SchwarzBounds::compute(op, &obs).unwrap();
        let ctx = ParallelContext::default();
        let config = RhfConfig {
            energy_conv: 1e-12,
            density_conv: 1e-10,
            df_j_aux: Some("def2-universal-jkfit".into()),
            df_k_aux: Some("def2-universal-jkfit".into()),
            ..Default::default()
        };
        let result = solve_rhf(&ctx, &mol, &obs, op, &bounds, &config).unwrap();
        assert!(result.converged);
        println!(
            "DF-JK RHF energy @ {} rayon threads: {:.17e}  bits=0x{:016x}",
            rayon::current_num_threads(),
            result.energy,
            result.energy.to_bits()
        );
    }

    #[test]
    fn df_k_build_from_occ_matches_density_build() {
        // Correctness gate for the C_occ half-transform: build_from_occ(C) must
        // reproduce build(D) with D = C·Cᵀ to machine precision — they contract
        // the SAME dressed B tensor, just in a different (mathematically
        // identical) order, so the residual is pure floating-point reassociation.
        let mol = Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1)
            .unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let n = obs.nbasis();
        let nocc = 5; // water: 5 doubly-occupied MOs

        // A deterministic (n, nocc) "C_occ" — need not be orthonormal for the
        // algebraic equivalence K[C] == K[C·Cᵀ] to hold.
        let mut c_occ = Array2::<f64>::zeros((n, nocc));
        for mu in 0..n {
            for i in 0..nocc {
                c_occ[(mu, i)] = 0.1 * (((mu * 3 + i * 7) % 13) as f64 - 6.0);
            }
        }
        let d = c_occ.dot(&c_occ.t());

        let mut dfk = DfK::new(op, &obs, &dfbs, usize::MAX).unwrap();

        let mut k_density = Array2::zeros((n, n));
        dfk.build(&d, &mut k_density).unwrap();

        let mut k_occ = Array2::zeros((n, n));
        dfk.build_from_occ(&c_occ, &mut k_occ).unwrap();

        let max_diff: f64 = (&k_occ - &k_density).iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(
            max_diff < 1e-10,
            "build_from_occ vs build(D=C·Cᵀ) max diff = {max_diff} too large"
        );
    }

    #[test]
    fn df_k_build_from_occ_bit_identical_across_thread_counts() {
        // The C_occ path must inherit the same deterministic-fold guarantee as
        // `build`: bit-identical K regardless of RAYON_NUM_THREADS.
        let mol = Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1)
            .unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let n = obs.nbasis();
        let nocc = 5;

        let mut c_occ = Array2::<f64>::zeros((n, nocc));
        for mu in 0..n {
            for i in 0..nocc {
                c_occ[(mu, i)] = 0.05 * (((mu * 5 + i * 3) % 17) as f64 - 8.0);
            }
        }

        let build_k = |threads: usize| -> Array2<f64> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut k = Array2::zeros((n, n));
                let mut dfk = DfK::new(op, &obs, &dfbs, usize::MAX).unwrap();
                dfk.build_from_occ(&c_occ, &mut k).unwrap();
                k
            })
        };
        let k1 = build_k(1);
        let k4 = build_k(4);
        assert_eq!(
            k1, k4,
            "build_from_occ must be bit-identical across thread counts"
        );
    }

    #[test]
    fn df_k_build_from_occ_zero_nocc_is_zero() {
        // Empty channel (e.g. UHF β with zero β electrons): K must be exactly zero.
        let mol = Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1)
            .unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let n = obs.nbasis();
        let c_occ = Array2::<f64>::zeros((n, 0));
        let mut dfk = DfK::new(op, &obs, &dfbs, usize::MAX).unwrap();
        let mut k = Array2::from_elem((n, n), 1.0);
        dfk.build_from_occ(&c_occ, &mut k).unwrap();
        assert!(k.iter().all(|&v| v == 0.0), "K must be exactly zero for nocc=0");
    }

    #[test]
    fn df_k_source_backed_matches_incore() {
        let mol = Molecule::parse_xyz(
            "3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n",
            0,
            1,
        )
        .unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs = PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let n = obs.nbasis();

        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n.min(5) {
            d[(i, i)] = 2.0;
        }

        // Huge budget → in-core path
        let mut k_big = Array2::zeros((n, n));
        let mut dfk_big = DfK::new(op, &obs, &dfbs, usize::MAX).unwrap();
        dfk_big.build(&d, &mut k_big).unwrap();

        // Tiny budget → spill path; must match to ≤1e-10
        let tiny = n * n * 8 * 3;
        let mut k_small = Array2::zeros((n, n));
        let mut dfk_small = DfK::new(op, &obs, &dfbs, tiny).unwrap();
        dfk_small.build(&d, &mut k_small).unwrap();

        let maxdiff = (&k_big - &k_small).iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(maxdiff < 1e-10, "spill K != in-core K, maxdiff={maxdiff}");
    }
}
