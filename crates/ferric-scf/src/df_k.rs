//! Density-fitted exchange matrix builder (DF-K / RI-K).
//!
//! Replaces the O(N^4) direct ERI exchange build with O(N^3 · naux) GEMMs using
//! the same 3-center auxiliary expansion as DF-J. For closed-shell RHF:
//!
//!   K_{μν} = Σ_{λσ} (μλ|νσ) D_{λσ}
//!          ≈ Σ_P Σ_{λσ} B^P_{μλ} B^P_{νσ} D_{λσ}        (RI with V^{-1/2}-dressed B)
//!
//! Computed as two passes over (P,μ,ν):
//!   Z\[P,μ,σ\] = Σ_λ B\[P,μ,λ\] · D\[λ,σ\]
//!   K\[μ,ν\]   = Σ_{P,σ} Z\[P,μ,σ\] · B\[P,ν,σ\]
//!
//! Each aux-block is contracted in rayon-parallel aux-chunks; BLAS stays
//! single-threaded (OPENBLAS_NUM_THREADS=1 under rayon), so chunks are the
//! parallel unit. The two builders differ in how a chunk is contracted:
//!
//! * [`crate::fock::KBuilder::build`] (density path) collapses the per-P (n,n)×(n,n) GEMM
//!   stack into two wide GEMMs on repacked operands — O(naux·n³).
//! * [`crate::fock::KBuilder::build_from_occ`] (occupied path, used once MOs exist) never
//!   repacks: B\[P\] is already a contiguous symmetric (n,n) slice, so each P is
//!   one DSYMM into a stacked (n, c·nocc) panel, reduced by ONE DSYRK per chunk
//!   — O(naux·n²·nocc), with both symmetries taken. At benzene/aug-cc-pVTZ
//!   (n=414, nocc=21) that is ~12.8x faster than the density path per build.
//!   The repack the density path amortizes over an n-wide contraction was 35%
//!   of the occ path's runtime, since its contraction is only nocc-wide.
//!
//! The dressed 3-center tensor B\[P,μ,ν\] = Σ_Q V^{-1/2}_{PQ} (Q|μν) is built once
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
/// `B[P,μ,ν]` — the resident B footprint scales with rank count. Because
/// K = Σ_P B_P D B_Pᵀ is a plain sum over P, each rank's band-restricted `build`
/// yields a partial K and one final all_reduce sums them. (Dressing a band still
/// reads all Q of the raw tensor — that raw is budget-bounded / spillable, so it
/// does not defeat the resident-memory scaling.) With one rank (or the `mpi`
/// feature off) the band is `0..naux` and the reduction is skipped, so behavior
/// is byte-identical to the serial path.
pub struct DfK<'a> {
    /// Budget-bounded dressed 3-center source B\[P,μ,ν\] = Σ_Q V^{-1/2}_{PQ} (Q|μν)
    /// — holds only this rank's aux band `[band_p0, band_p1)`.
    dressed: ThreeIndexSource,
    /// Parallel context for the final cross-rank K reduction. `None` → serial /
    /// non-MPI (no reduction, full band).
    #[allow(dead_code)]
    ctx: Option<&'a ParallelContext>,
    /// The memory budget this source was built under — also caps the K
    /// reduction band scratch via `resolve_band_bytes`.
    budget_bytes: usize,
    /// Device-resident copy state (feature `gpu`); `Untried` until the first
    /// occupied-path build asks for it. See `crate::df_k_gpu`.
    #[cfg(feature = "gpu")]
    device: crate::df_k_gpu::DeviceSlot,
}

impl<'a> std::fmt::Debug for DfK<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DfK")
            .field("budget_bytes", &self.budget_bytes)
            .finish_non_exhaustive()
    }
}

/// Nominal metric eigenvalue below which [`DfK`] drops a mode from V^{-1/2}.
///
/// The cut actually applied is [`metric_lindep_cut`], which keeps this
/// nominal value whenever no eigenvalue lies within rounding of it and
/// otherwise moves the cut into a spectral gap. Every consumer of the
/// truncated metric ([`crate::df_gradient`], the DF Hessians) classifies modes
/// through that one function, never by comparing against this constant.
pub(crate) const DFK_LINDEP_THRESH: f64 = 1e-10;

/// Required distance of every eigenvalue from the applied cut, in units of
/// the eigenvalue rounding scale `δ = ε · max|λ|` (see [`metric_lindep_cut`]).
///
/// MEASURED between `dsyev` and `dsyevd` on the def2-universal-jkfit metrics
/// of water, sulfamethoxazole and danuglipron (Coulomb, erfc(0.33),
/// erf(0.33)): `|Δλ|` at most 3.2 δ over the whole spectrum, p95 at most
/// 1.5 δ for the eigenvalues in `[1e-12, 1e-8]`. The erf spectra there are
/// dense (median spacing 8.8 δ for sulfamethoxazole, 1.8 δ for danuglipron),
/// so a cut needs a margin well beyond one spacing for its CHOICE to be stable.
/// Under independent `N(0, σ²)` noise on every eigenvalue (1000 draws), with
/// 100 the cut classifies identically in 1000/1000 draws at σ = 1 δ on both
/// erf spectra (the old 1e-10 cut: 525/1000 on danuglipron). Smaller values
/// are not monotonically safer: at 5, 10, 20, 30 it flips in 26, 68, 0, 96
/// (sulfamethoxazole) and 73, 102, 8, 0 (danuglipron) of 200 draws, because a
/// small margin leaves gaps right at the safe/unsafe boundary (wiki notes
/// df-gap-cut-2026-11-07).
pub(crate) const LINDEP_CUT_MARGIN_ROUNDINGS: f64 = 100.0;

/// The cut may move at most this factor either side of [`DFK_LINDEP_THRESH`].
/// The DF-K gradient already treats every mode below
/// `crate::df_gradient::DFK_GRADIENT_NOISE_FLOOR` (1e-7) as dropped, so a cut
/// moved up to 1e-8 stays below that floor.
pub(crate) const LINDEP_CUT_WINDOW: f64 = 100.0;

/// How [`metric_lindep_cut`] placed its cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LindepCutKind {
    /// Every eigenvalue is at least the margin from [`DFK_LINDEP_THRESH`]:
    /// the cut is that value, and the kept set is the one the plain absolute
    /// cut gives.
    Nominal,
    /// An eigenvalue lay within the margin of [`DFK_LINDEP_THRESH`]; the cut
    /// is the nearest point (in log distance) that is at least the margin
    /// from every eigenvalue.
    Gap,
    /// An eigenvalue lay within the margin of [`DFK_LINDEP_THRESH`] and no
    /// point of the window is the margin from every eigenvalue: the cut stays
    /// at the nominal value and the classification of the modes next to it
    /// is rounding-dependent ([`warn_if_no_safe_gap`] says so).
    NoSafeGap,
}

/// The eigenvalue cut of the truncated auxiliary metric: mode `k` is kept
/// iff `λ_k >= cut` ([`LindepCut::keeps`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LindepCut {
    pub cut: f64,
    pub n_dropped: usize,
    pub kind: LindepCutKind,
    /// `δ = ε · max|λ|`, the eigenvalue rounding scale.
    pub rounding: f64,
    /// Distance from the cut of the nearest eigenvalue (`+∞` if none).
    pub margin: f64,
}

impl LindepCut {
    #[inline]
    pub fn keeps(&self, lambda: f64) -> bool {
        lambda >= self.cut
    }

    /// The cut for a message: `1e-10` when nominal, else the moved value.
    pub fn describe(&self) -> String {
        match self.kind {
            LindepCutKind::Gap => format!(
                "{:.4e} (moved from {DFK_LINDEP_THRESH:.0e} into a spectral gap)",
                self.cut
            ),
            LindepCutKind::Nominal | LindepCutKind::NoSafeGap => format!("{:.0e}", self.cut),
        }
    }
}

/// Where to cut the spectrum of an auxiliary metric (`evals` in any order).
///
/// A fixed absolute cut is arbitrary for a mode whose eigenvalue lies within
/// rounding of it: two correct eigensolvers then keep or drop it differently.
/// On the erf(0.33) metric of danuglipron the nearest eigenvalue sat 0.04 δ
/// from 1e-10 and `dsyev` dropped 1020 modes where `dsyevd` dropped 1019.
///
/// With `δ = ε · max|λ|`, `m` = [`LINDEP_CUT_MARGIN_ROUNDINGS`] · δ and
/// `τ` = [`DFK_LINDEP_THRESH`], a point `c` is SAFE when every eigenvalue is
/// at least `m` from it; the safe set is the union over the gaps between
/// consecutive eigenvalues (and the two unbounded ends) of
/// `[λ_i + m, λ_{i+1} − m]`. The cut is the safe point in the window
/// `[τ / W, W τ]` (`W` = [`LINDEP_CUT_WINDOW`]) nearest to `τ` in log
/// distance, the lower one on an exact tie:
///
/// * `τ` itself when it is safe ([`LindepCutKind::Nominal`]): such a
///   spectrum keeps exactly what the plain absolute cut kept, so every metric
///   whose spectrum is not crowded at `τ` is unchanged (the Coulomb and erfc
///   metrics above all have no eigenvalue below 7e-7);
/// * else an end of a safe interval ([`LindepCutKind::Gap`]);
/// * `τ` if the window holds no safe point ([`LindepCutKind::NoSafeGap`]).
///
/// Nearest-safe-point rather than largest-gap-ratio: the cut is continuous
/// in the spectrum where `τ` stops being safe (it slides from `τ` to the end
/// of the same interval, so the classification does not change at that
/// boundary), and it moves the cut least. On the dense erf spectra the
/// largest ratios are near-tied (1.067 / 1.060 / 1.059 for sulfamethoxazole),
/// which makes a ratio argmax a rounding-sensitive choice at small margins.
///
/// No deterministic rule removes the arbitrariness entirely: with the rest
/// of the spectrum fixed, sweeping one eigenvalue through the window must
/// flip its classification somewhere. Here the flips sit where two safe
/// points are equidistant from `τ`, or where a gap's half-width crosses `m`;
/// the measured flip rates are in [`LINDEP_CUT_MARGIN_ROUNDINGS`].
pub(crate) fn metric_lindep_cut(evals: &[f64]) -> LindepCut {
    let tau = DFK_LINDEP_THRESH;
    let lmax = evals.iter().fold(0.0f64, |a, &l| a.max(l.abs()));
    let rounding = f64::EPSILON * lmax;
    let m = LINDEP_CUT_MARGIN_ROUNDINGS * rounding;
    let mut sorted = evals.to_vec();
    sorted.sort_by(f64::total_cmp);
    let (lo_w, hi_w) = (tau / LINDEP_CUT_WINDOW, tau * LINDEP_CUT_WINDOW);
    let n = sorted.len();
    // (log distance to τ, cut); gaps visited in ascending order, so a strict
    // `<` keeps the lower of two equidistant safe points.
    let mut best: Option<(f64, f64)> = None;
    for i in 0..=n {
        let lo = if i == 0 {
            f64::NEG_INFINITY
        } else {
            sorted[i - 1] + m
        };
        let hi = if i == n { f64::INFINITY } else { sorted[i] - m };
        let (a, b) = (lo.max(lo_w), hi.min(hi_w));
        if !matches!(
            a.partial_cmp(&b),
            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
        ) {
            continue;
        }
        let c = tau.clamp(a, b);
        let d = (c / tau).ln().abs();
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, c));
        }
    }
    let (cut, kind) = match best {
        Some((0.0, _)) => (tau, LindepCutKind::Nominal),
        Some((_, c)) => (c, LindepCutKind::Gap),
        None => (tau, LindepCutKind::NoSafeGap),
    };
    LindepCut {
        cut,
        n_dropped: evals.len() - evals.iter().filter(|&&l| l >= cut).count(),
        kind,
        rounding,
        margin: evals
            .iter()
            .fold(f64::INFINITY, |a, &l| a.min((l - cut).abs())),
    }
}

/// V^{-1/2} via symmetric eigendecomposition with canonical orthogonalization.
/// The 2-center metric `(P|w(r12)|Q)` is positive-definite analytically, but
/// for range-separated operators (erf, erfc) with JK-fit aux on heavy atoms,
/// some eigenvalues can be near-zero and turn slightly negative under
/// floating-point roundoff. Drop those modes — equivalent to PySCF's
/// `lindep` threshold in `df.aux_e2`.
/// Relative eigenvalue cut of [`DfK::build_from_density_factor`]: a guess or
/// projector density's spectrum is its rank's worth of O(1) eigenvalues and a
/// floor at the `eigh` rounding level (about `nao · u64 · max|λ|`, 1e-13 at
/// nao = 528), so 1e-11 keeps every real eigenvalue and drops the floor; a
/// dropped `λ` changes `K` by at most `|λ| · max_P ‖B_P v‖²`.
const DENSITY_FACTOR_REL_CUT: f64 = 1e-11;

/// `(X₊, X₋)` with `D = X₊X₊ᵀ − X₋X₋ᵀ`, or `None` when the factor would not be
/// cheaper than the density path or `D` is not symmetric (see
/// [`DfK::build_from_density_factor`]).
fn density_factor(d: &Array2<f64>) -> Option<(Array2<f64>, Array2<f64>)> {
    let n = d.nrows();
    if n == 0 || d.ncols() != n {
        return None;
    }
    let scale = d.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let asym = (0..n)
        .flat_map(|i| (0..i).map(move |j| (i, j)))
        .fold(0.0f64, |m, (i, j)| m.max((d[(i, j)] - d[(j, i)]).abs()));
    if !scale.is_finite() || asym > 1e-12 * scale.max(f64::MIN_POSITIVE) {
        return None;
    }
    // Divide and conquer (`dsyevd`), not the QR `dsyev`: `D` enters K only through
    // `X₊X₊ᵀ` and `X₋X₋ᵀ`, sums of `λ v vᵀ` over the kept eigenvalues of one
    // sign, which do not depend on the basis chosen inside a degenerate block
    // (a guess density's spectrum is full of them), so the solvers differ only in
    // rounding. The cut sits two orders above the rounding floor (see
    // `DENSITY_FACTOR_REL_CUT`), so no mode changes side between them.
    let (evals, evecs) = with_blas_threads(opt_in_blas_threads(), || {
        ferric_core::linalg::eigh_dc(d, ferric_core::linalg::Uplo::Lower)
    })
    .ok()?;
    let lmax = evals.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if lmax == 0.0 {
        return Some((Array2::zeros((n, 0)), Array2::zeros((n, 0))));
    }
    let cut = DENSITY_FACTOR_REL_CUT * lmax;
    let pos: Vec<usize> = (0..n).filter(|&i| evals[i] > cut).collect();
    let neg: Vec<usize> = (0..n).filter(|&i| evals[i] < -cut).collect();
    if 4 * (pos.len() + neg.len()) > 3 * n {
        return None;
    }
    let columns = |idx: &[usize]| {
        let mut x = Array2::<f64>::zeros((n, idx.len()));
        for (c, &i) in idx.iter().enumerate() {
            let w = evals[i].abs().sqrt();
            x.column_mut(c).assign(&evecs.column(i).mapv(|v| v * w));
        }
        x
    };
    Some((columns(&pos), columns(&neg)))
}

/// The report [`metric_lindep_cut`]'s fallback promises: with no safe gap the
/// modes next to the nominal cut are classified by rounding.
pub(crate) fn warn_if_no_safe_gap(cut: &LindepCut, who: &str) {
    if cut.kind == LindepCutKind::NoSafeGap {
        eprintln!(
            "[ferric] warning: {who}: an auxiliary-metric eigenvalue lies {:.1} roundings \
             (eps*max|lambda| = {:.2e}) from the {DFK_LINDEP_THRESH:.0e} cut and no spectral gap \
             of half-width >= {LINDEP_CUT_MARGIN_ROUNDINGS} roundings exists in [{:.0e}, {:.0e}]; \
             cutting at {DFK_LINDEP_THRESH:.0e} ({} mode(s) dropped), so whether that mode is kept \
             depends on eigensolver rounding",
            cut.margin / cut.rounding.max(f64::MIN_POSITIVE),
            cut.rounding,
            DFK_LINDEP_THRESH / LINDEP_CUT_WINDOW,
            DFK_LINDEP_THRESH * LINDEP_CUT_WINDOW,
            cut.n_dropped,
        );
    }
}

fn v_inv_sqrt_lindep(v: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    let naux = v.nrows();
    // One-time (naux, naux) setup factorization, called once per SCF
    // construction outside any rayon region. Divide and conquer (`dsyevd`):
    // the QR `dsyev` behind `Eigh::eigh` took 3.5 s of a 6.0 s DF-K setup on
    // one core at naux = 1538 (sulfamethoxazole, def2-universal-jkfit),
    // against 0.65 s for this call. V^{-1/2} is a symmetric matrix function,
    // invariant to the eigenbasis chosen inside a degenerate block, so the
    // solvers differ only in rounding — except for a mode whose eigenvalue
    // lies within rounding of the cut, which one could keep and the other
    // drop; `metric_lindep_cut` places the cut so that none does.
    // `crate::df_gradient` decomposes the same metric with the same solver and
    // the same cut rule, so energy and gradient classify every mode identically.
    // Opt-in BLAS raise via FERRIC_BLAS_THREADS (default: the rayon pool size; 1 on a rayon
    // worker); opt_in_blas_threads()'s rayon-worker self-guard also covers
    // any caller reached from a single-thread rayon pool (e.g. free-atom SAD),
    // resolving to 1 there regardless of the env var.
    let t_st = ferric_core::timing::Stage::start("dfk:metric-eigh");
    let (evals, evecs) = with_blas_threads(opt_in_blas_threads(), || {
        ferric_core::linalg::eigh_dc(v, ferric_core::linalg::Uplo::Upper)
    })
    .map_err(|e| FerricError::Lapack(format!("V eigh in DfK: {e}")))?;
    t_st.end();
    let cut = metric_lindep_cut(&evals);
    warn_if_no_safe_gap(&cut, "DF-K");
    let mut u_scaled = evecs.clone();
    for k in 0..naux {
        if !cut.keeps(evals[k]) {
            // Zero out this column so its (column-vector outer product) contributes
            // nothing to V^{-1/2}.
            for r in 0..naux {
                u_scaled[(r, k)] = 0.0;
            }
        } else {
            let s = 1.0 / evals[k].sqrt();
            for r in 0..naux {
                u_scaled[(r, k)] *= s;
            }
        }
    }
    // Silent on cut.n_dropped: dropped modes are expected for range-separated
    // operators (erf, erfc) with JK-fit aux on heavy atoms and are benign.
    // O(naux³) (7.3 GFLOP at naux = 1538): fixed column blocks over the rayon
    // pool, one-thread BLAS each, bit-identical across thread counts. NOT
    // under the opt-in BLAS raise, which would multiply every block's threads.
    let t_st = ferric_core::timing::Stage::start("dfk:metric-inv-sqrt-gemm");
    let mut m = Array2::<f64>::zeros((naux, naux));
    ferric_core::linalg::par_gemm(1.0, &u_scaled.view(), &evecs.t(), 0.0, &mut m);
    t_st.end();
    Ok(m) // (naux, naux)
}

impl<'a> DfK<'a> {
    /// Build the DF-K cache from orbital and auxiliary bases (FULL aux range,
    /// serial / single-rank — byte-identical to the pre-MPI path).
    ///
    /// Computes V^{-1/2} = U · diag(λ^{-1/2}) · U^T from the symmetric eigendecomp
    /// of the (P|Q) Coulomb metric, then forms `B[P,μ,ν] = Σ_Q V^{-1/2}_{PQ} (Q|μν)`
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
        let naux = dfbs.nbasis();
        let (p0, p1) = match ctx {
            Some(c) => c.aux_band(naux),
            None => (0, naux),
        };

        // Build the FULL raw source (dressing a band sums over all Q). The raw is
        // budget-bounded / spillable, so its full footprint need not be resident;
        // only this rank's dressed BAND is retained. Before the metric, because a
        // spilling build already uses the whole budget.
        let mut raw = ThreeIndexSource::build(op, obs, dfbs, budget_bytes)?;
        let v = coulomb_metric_2c(op, dfbs)?;
        let v_inv_sqrt = v_inv_sqrt_lindep(&v)?;
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
        let t_st = ferric_core::timing::Stage::start("dfk:dress");
        let dressed = if p0 == 0 && p1 == naux {
            // Full band: identical call to the serial path (byte-identical B).
            ThreeIndexSource::build_dressed(raw, &v_inv_sqrt, budget_bytes)?
        } else {
            ThreeIndexSource::build_dressed_band(raw, &v_inv_sqrt, budget_bytes, p0, p1)?
        };
        // The caller owns `raw`; DfK retains only the dressed tensor. Dressing
        // is the one streaming pass over `raw`; a packed `raw` that moves on
        // into DfJ must not keep its unpacked scratch block for the whole SCF.
        raw.release_scratch();
        t_st.end();

        let ctx = ctx.filter(|c| c.size > 1);
        Ok(DfK {
            dressed,
            ctx,
            budget_bytes,
            #[cfg(feature = "gpu")]
            device: crate::df_k_gpu::DeviceSlot::default(),
        })
    }
}

impl DfK<'_> {
    /// `K(D)` for a symmetric `D` with no orbital factor at hand (the first Fock
    /// build from a guess density, ROHF's spin densities, a fractionally occupied
    /// density), through the occupied path. `D = V Λ Vᵀ` gives `D = X₊X₊ᵀ − X₋X₋ᵀ`
    /// with `X± = V·√|λ|` over the eigenvalues of each sign, and K is linear in D,
    /// so `K(D) = K(X₊X₊ᵀ) − K(X₋X₋ᵀ)`: two [`build_from_occ`](KBuilder::build_from_occ)
    /// calls whose cost scales with the columns kept (a guess density's rank is
    /// the minimal-basis size; a spin projector's is its occupation) instead of
    /// the density path's `nao`.
    ///
    /// Eigenvalues with `|λ| ≤ DENSITY_FACTOR_REL_CUT · max|λ|` are dropped (the
    /// rank-deficient part of the spectrum, rounding noise of `eigh`). Falls back
    /// to [`build`](KBuilder::build) (the density path) when `D` is not
    /// symmetric to rounding, when the kept columns exceed `3·nao/4`, or when
    /// `eigh` fails. The occupied path costs about `m/nao` of the density path
    /// for `m` columns (two `naux·nao²·m` stages against two `naux·nao³`), so a
    /// factor pays up to `m ≈ nao`; the margin covers its narrower GEMMs. A
    /// guess density keeps more columns than the minimal basis (158 of 298 at
    /// n-dodecane / def2-SVP, 180 of 414 at benzene / aug-cc-pVTZ). Not bit-identical to the density path: it
    /// agrees with it to the DF-K reassociation floor, as the occupied path does.
    pub fn build_from_density_factor(
        &mut self,
        d: &Array2<f64>,
        k: &mut Array2<f64>,
    ) -> Result<usize, FerricError> {
        match density_factor(d) {
            Some((x_pos, x_neg)) => {
                self.build_from_occ_impl(&x_pos, k)?;
                if x_neg.ncols() > 0 {
                    let mut k_neg = Array2::<f64>::zeros(k.raw_dim());
                    self.build_from_occ_impl(&x_neg, &mut k_neg)?;
                    *k -= &k_neg;
                }
                Ok(0)
            }
            None => self.build_impl(d, k),
        }
    }

    /// Device bytes this builder's resident tensor needs under `gpu = on`: the
    /// dressed band plus the K accumulator when the dressed tensor is in core, 0
    /// when it is spilled or recomputed (the device path declines it).
    #[cfg(feature = "gpu")]
    pub fn device_footprint_bytes(&self) -> usize {
        if !self.dressed.is_incore() {
            return 0;
        }
        crate::df_k_gpu::resident_bytes(self.dressed.band_naux(), self.dressed.nao()).unwrap_or(0)
    }

    /// The in-core dressed band as `(band_naux × n²)`; `None` when spilled. Test hook.
    #[cfg(feature = "gpu")]
    #[doc(hidden)]
    pub fn dressed_incore_flat_for_test(&self) -> Option<ndarray::ArrayView2<'_, f64>> {
        self.dressed.incore_flat()
    }

    /// Density-based exchange contraction (O(naux·n³)); see [`KBuilder::build`].
    /// Kept as an inherent method so both the trait `build` and the C_occ-based
    /// `build_from_occ` share one struct-level definition.
    fn build_impl(&mut self, d: &Array2<f64>, k: &mut Array2<f64>) -> Result<usize, FerricError> {
        let n = self.dressed.nao();
        k.fill(0.0);
        // Aux-chunk width: wide enough that the (n, c·n)×(c·n, n) contraction
        // below runs at BLAS3 efficiency even for small n, capped so per-task
        // scratch (3·c·n² doubles) stays modest for large n.
        let chunk = ferric_integrals::three_index_source::df_chunk_width(n);
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
        // J runs between K passes and an RSH K pair shares the budget, so a spilled
        // block must not stay resident between them.
        self.dressed.release_scratch();

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

        // Device-resident path (opt-in `gpu` feature, `[gpu] mode` != off). Placed AFTER
        // the FERRIC_DFK_FORCE_DENSITY and nocc == 0 shortcuts so the test switch and the
        // empty channel never reach the card. `true` = K written; the cross-rank
        // reduction below is shared with the CPU path.
        #[cfg(feature = "gpu")]
        if crate::df_k_gpu::try_build_from_occ(&mut self.device, &self.dressed, self.ctx, c_occ, k)
        {
            self.reduce_k_across_ranks(k);
            return Ok(0);
        }

        // Aux-chunk width: same BLAS3-efficiency vs per-task-scratch tradeoff as
        // `build`, but the half-transform scratch scales with nocc, not n, so the
        // trailing GEMM operand is (c·nocc) wide.
        let chunk = ferric_integrals::three_index_source::df_chunk_width(n);
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
        self.dressed.release_scratch();

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
    use ndarray_linalg::{Eigh, UPLO};

    /// Catches: a factor of the wrong rank or sign (a dropped negative part, a
    /// missing √λ, a wrong cut), a factored K that differs from the density
    /// path beyond reassociation, and a fallback that is not the density path
    /// itself (a full-rank or a non-symmetric D must take `build` bit for bit).
    #[test]
    fn a_factored_density_gives_the_density_path_k_and_falls_back_when_it_should() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let aux =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let n = obs.nbasis();
        let mut dfk = DfK::new(Operator::coulomb(), &obs, &aux, usize::MAX).unwrap();
        let cols = |m: usize, salt: usize| {
            Array2::from_shape_fn((n, m), |(i, j)| {
                0.1 * ((((i + 1) * (j + 3) * 7 + salt * 11) % 23) as f64 - 11.0)
            })
        };
        let (a, b) = (cols(3, 1), cols(2, 2));
        let psd = cols(5, 3).dot(&cols(5, 3).t()) * 2.0;
        let indefinite = a.dot(&a.t()) - b.dot(&b.t());
        let k_of = |dfk: &mut DfK, d: &Array2<f64>, factored: bool| {
            let mut k = Array2::<f64>::zeros((n, n));
            if factored {
                dfk.build_from_density_factor(d, &mut k).unwrap();
            } else {
                dfk.build(d, &mut k).unwrap();
            }
            k
        };
        for (d, want) in [(&psd, (5, 0)), (&indefinite, (3, 2))] {
            let (xp, xn) = density_factor(d).expect("low rank: factored");
            assert_eq!((xp.ncols(), xn.ncols()), want);
            let back = xp.dot(&xp.t()) - xn.dot(&xn.t());
            let scale = d.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let err = (&back - d).iter().fold(0.0f64, |m, v| m.max(v.abs()));
            assert!(err <= 1e-12 * scale, "X₊X₊ᵀ − X₋X₋ᵀ misses D by {err:e}");
            let (kf, kd) = (k_of(&mut dfk, d, true), k_of(&mut dfk, d, false));
            let big = kd.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let diff = (&kf - &kd).iter().fold(0.0f64, |m, v| m.max(v.abs()));
            assert!(big > 1e-3, "K must not be ~zero");
            assert!(
                diff <= 1e-12 * big,
                "factored K off by {diff:e} (max |K| {big:e})"
            );
        }
        let full = Array2::<f64>::eye(n);
        let mut skew = psd.clone();
        skew[(0, 1)] += 1e-6;
        for d in [&full, &skew] {
            assert!(density_factor(d).is_none());
            let (kf, kd) = (k_of(&mut dfk, d, true), k_of(&mut dfk, d, false));
            assert!(
                kf.iter()
                    .zip(kd.iter())
                    .all(|(x, y)| x.to_bits() == y.to_bits()),
                "the fallback must be the density path itself"
            );
        }
    }

    /// J runs between K passes, and a range-separated pair holds two of these under
    /// one budget, so a spilled block must not outlive either kind of pass.
    #[test]
    fn a_spilled_k_frees_its_block_after_each_pass() {
        let mol = Molecule::load_xyz("../../testdata/molecules/water.xyz").unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let aux = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz-ri").unwrap()).unwrap();
        let (naux, nao) = (aux.nbasis(), obs.nbasis());
        let budget = naux * nao * nao * 8 * 45 / 100;
        let mut dfk = DfK::new(Operator::coulomb(), &obs, &aux, budget).unwrap();
        assert!(!dfk.dressed.is_incore(), "fixture must spill");
        assert_eq!(dfk.dressed.scratch_bytes(), 0, "allocated before first use");

        let d = Array2::<f64>::eye(nao);
        let mut k = Array2::<f64>::zeros((nao, nao));
        dfk.build(&d, &mut k).unwrap();
        assert_eq!(dfk.dressed.scratch_bytes(), 0, "kept after a density pass");

        let c_occ = d.slice(ndarray::s![.., 0..5]).to_owned();
        dfk.build_from_occ(&c_occ, &mut k).unwrap();
        assert_eq!(
            dfk.dressed.scratch_bytes(),
            0,
            "kept after an occupied pass"
        );
    }

    /// Dressing is the one streaming pass over a shared packed `raw`; the
    /// unpacked scratch block it allocated must be gone once `from_full_raw`
    /// returns, since `raw` then lives on inside DfJ for the whole SCF.
    #[test]
    fn dressing_a_packed_raw_releases_its_unpack_scratch() {
        let mol = Molecule::load_xyz("../../testdata/molecules/alkane_3.xyz").unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("def2-svp").unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let (n, naux) = (obs.nbasis(), dfbs.nbasis());
        let budget = (naux * n * n * 8 + naux * (n * (n + 1) / 2) * 8) / 2;
        let mut raw = ThreeIndexSource::build(op, &obs, &dfbs, budget).unwrap();
        assert!(raw.is_packed_incore_for_test(), "must be the packed tier");
        assert_eq!(raw.scratch_bytes(), 0);
        let mut probe = ThreeIndexSource::build(op, &obs, &dfbs, budget).unwrap();
        probe.for_each_block(|_| Ok(())).unwrap();
        assert!(
            probe.scratch_bytes() > 0,
            "a streaming pass allocates scratch"
        );
        probe.release_scratch();
        assert_eq!(probe.scratch_bytes(), 0);
        DfK::from_full_raw(&mut raw, &obs, &dfbs, op, budget, None).unwrap();
        assert_eq!(
            raw.scratch_bytes(),
            0,
            "dressing left the scratch allocated"
        );
    }

    #[test]
    fn df_k_matches_direct_k_with_jkfit() {
        let mol =
            Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
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

        let max_diff: f64 = (&k_df - &k_direct)
            .iter()
            .map(|v| v.abs())
            .fold(0.0, f64::max);
        // JK-fit basis should give K accurate to ~1e-3 for this small system.
        assert!(
            max_diff < 5e-3,
            "DF-K vs direct-K max diff = {} too large",
            max_diff
        );
    }

    #[test]
    fn df_k_bit_identical_across_thread_counts() {
        // Regression guard: the aux-chunk accumulation in `build` must be
        // bit-identical regardless of RAYON_NUM_THREADS. A rayon `try_reduce`
        // combined partials in a worker-count-dependent tree, so the K matrix
        // (and hence the SCF energy) drifted ~µHa between thread counts. The
        // collect-then-serial-sum fix pins the order. Uses several heavy atoms so
        // the aux dimension spans multiple chunks (making order actually matter).
        let mol =
            Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
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
        let mol =
            Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
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
        assert_eq!(
            k1, k2,
            "narrow-band DfK must be bit-identical at 1 vs 2 threads"
        );
        assert_eq!(
            k1, k8,
            "narrow-band DfK must be bit-identical at 1 vs 8 threads"
        );
    }

    #[test]
    fn df_k_wide_gemm_matches_naive_contraction() {
        // Implementation-equivalence test for the wide-GEMM restructure: the
        // chunked two-wide-GEMM contraction in `build` must reproduce the naive
        // per-P GEMM stack over the SAME dressed B tensor to machine precision.
        // A dense symmetric density exercises every (P,σ) coupling, unlike the
        // diagonal-D accuracy test above (which measures RI fitting error vs
        // direct K and cannot separate algebra bugs from fitting error).
        let mol =
            Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
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
        let mol =
            Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
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
        let mol =
            Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
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

        let max_diff: f64 = (&k_occ - &k_density)
            .iter()
            .map(|v| v.abs())
            .fold(0.0, f64::max);
        assert!(
            max_diff < 1e-10,
            "build_from_occ vs build(D=C·Cᵀ) max diff = {max_diff} too large"
        );
    }

    #[test]
    fn df_k_build_from_occ_bit_identical_across_thread_counts() {
        // The C_occ path must inherit the same deterministic-fold guarantee as
        // `build`: bit-identical K regardless of RAYON_NUM_THREADS.
        let mol =
            Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
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
        let mol =
            Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let n = obs.nbasis();
        let c_occ = Array2::<f64>::zeros((n, 0));
        let mut dfk = DfK::new(op, &obs, &dfbs, usize::MAX).unwrap();
        let mut k = Array2::from_elem((n, n), 1.0);
        dfk.build_from_occ(&c_occ, &mut k).unwrap();
        assert!(
            k.iter().all(|&v| v == 0.0),
            "K must be exactly zero for nocc=0"
        );
    }

    #[test]
    fn df_k_source_backed_matches_incore() {
        let mol =
            Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("cc-pvdz").unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
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

        let maxdiff = (&k_big - &k_small)
            .iter()
            .map(|v| v.abs())
            .fold(0.0, f64::max);
        assert!(maxdiff < 1e-10, "spill K != in-core K, maxdiff={maxdiff}");
    }

    // ---- metric_lindep_cut ----

    const TAU: f64 = DFK_LINDEP_THRESH;

    /// The plain absolute cut, for the comparisons below.
    fn old_drop(evals: &[f64]) -> usize {
        evals.iter().filter(|&&l| l < TAU).count()
    }

    /// `m` for a spectrum whose largest eigenvalue is `lmax`.
    fn margin_for(lmax: f64) -> f64 {
        LINDEP_CUT_MARGIN_ROUNDINGS * f64::EPSILON * lmax
    }

    /// Catches: a rule that moves a nominal cut that already has its margin
    /// (a gap of LARGER ratio, 25 against 6, sits in the window below τ, so
    /// "largest ratio wins" would move the cut and drop one mode fewer), and
    /// a lost `Nominal` classification.
    #[test]
    fn lindep_cut_keeps_a_nominal_cut_that_has_its_margin() {
        let evals = [2.0, -1e-14, 2e-12, 5e-11, 3e-10, 1.0];
        let c = metric_lindep_cut(&evals);
        assert_eq!(c.kind, LindepCutKind::Nominal);
        assert_eq!(c.cut, TAU);
        assert_eq!(c.n_dropped, 3);
        assert_eq!(c.n_dropped, old_drop(&evals));
        assert!((c.margin - 5e-11).abs() < 1e-24, "margin {}", c.margin);
        // A Coulomb-like spectrum (nothing in the window) and an empty one.
        let c = metric_lindep_cut(&[7.3e-7, 1e-3, 3.0e3]);
        assert_eq!(
            (c.kind, c.cut, c.n_dropped),
            (LindepCutKind::Nominal, TAU, 0)
        );
        let c = metric_lindep_cut(&[]);
        assert_eq!(
            (c.kind, c.cut, c.n_dropped),
            (LindepCutKind::Nominal, TAU, 0)
        );
    }

    /// `base` plus one eigenvalue at `τ ± 0.15 δ` (danuglipron's sat at
    /// 0.04 δ): the old cut keeps it for `+` and drops it for `−`, the rule
    /// must give both spectra the same kept count, `want_drop`, with the cut
    /// at `want_cut` (to rounding) and every eigenvalue at least `m` from it.
    fn assert_same_either_side(base: &[f64], want_drop: usize, want_cut: f64) {
        let lmax = base.iter().fold(0.0f64, |a, &l| a.max(l.abs()));
        let (delta, m) = (f64::EPSILON * lmax, margin_for(lmax));
        let mut seen = Vec::new();
        for s in [0.15, -0.15] {
            let mut evals = base.to_vec();
            evals.push(TAU + s * delta);
            let c = metric_lindep_cut(&evals);
            assert_eq!(c.kind, LindepCutKind::Gap, "s={s}");
            assert!(
                c.margin >= m * (1.0 - 1e-9),
                "s={s}: margin {}",
                c.margin / delta
            );
            assert!(
                (c.cut - want_cut).abs() <= delta,
                "s={s}: cut {} want {want_cut}",
                c.cut
            );
            seen.push((c.n_dropped, old_drop(&evals)));
        }
        assert_eq!(seen[0].0, want_drop);
        assert_eq!(seen[0].0, seen[1].0, "the rule follows the rounding");
        assert_ne!(
            seen[0].1, seen[1].1,
            "the old cut did not flip: vacuous construction"
        );
    }

    /// Catches: no margin (`m = 0` is the old cut), a cut chosen by any
    /// criterion but the nearest safe point (the second layout has a far
    /// wider-ratio safe gap above the crowding), a cut at a gap midpoint, and
    /// a wrong `keeps` side.
    #[test]
    fn lindep_cut_classifies_an_eigenvalue_at_the_nominal_cut_the_same_either_side() {
        // m = 6.7e-11: nothing safe below τ in the window, the nearest safe
        // point is τ + m and the crowding eigenvalue drops.
        assert_same_either_side(
            &[1e-14, 2e-11, 4e-10, 1e-9, 3.0e3],
            3,
            TAU + margin_for(3.0e3),
        );
        // m = 6.7e-12: eigenvalues crowd (τ, 3e-10] at spacing 5e-12 < 2m, the
        // safe interval (3e-11 + m, τ − m) is nearest, the cut is τ − m and
        // the crowding eigenvalue is kept.
        let mut base = vec![1e-14, 3e-11, 1e-8, 300.0];
        base.extend((1..=40).map(|k| TAU + 5e-12 * f64::from(k)));
        assert_same_either_side(&base, 2, TAU - margin_for(300.0));
    }

    /// Crowding at τ with no safe point inside `[τ/W, Wτ]` (λ_max = 1e6 makes
    /// m = 2.2e-8, more than the window allows): the cut stays at τ and the
    /// kind says the classification is rounding-dependent.
    ///
    /// Catches: a window that is not enforced (the safe point at 2.2e-8 would
    /// be taken) and a fallback other than the nominal cut.
    #[test]
    fn lindep_cut_falls_back_to_the_nominal_cut_without_a_safe_gap() {
        let c = metric_lindep_cut(&[1e-14, TAU * (1.0 + 1e-9), 1e6]);
        assert_eq!(c.kind, LindepCutKind::NoSafeGap);
        assert_eq!(c.cut, TAU);
        assert_eq!(c.n_dropped, 1);
        // With a safe point inside the window the same crowding moves the cut.
        let c = metric_lindep_cut(&[1e-14, TAU * (1.0 + 1e-9), 1e4]);
        assert_eq!(c.kind, LindepCutKind::Gap);
    }

    /// No seam where τ stops being safe: an eigenvalue just outside and just
    /// inside `τ + m` is kept both times (the cut slides from τ to `λ − m`,
    /// the end of the same safe interval).
    ///
    /// Catches: a two-branch rule ("nominal if safe, else the best gap")
    /// whose second branch picks another gap at that boundary (the safe gaps
    /// above, (λ + 1e-11, 2e-9) and (2e-9, 1.5e-8), have the larger ratios).
    #[test]
    fn lindep_cut_has_no_seam_where_the_nominal_cut_stops_being_safe() {
        let lmax = 3.0e3;
        let m = margin_for(lmax);
        let mut kinds = Vec::new();
        for f in [1.0 + 1e-6, 1.0 - 1e-6] {
            let lj = TAU + f * m;
            let c = metric_lindep_cut(&[1e-14, 1e-12, lj, lj + 1e-11, 2e-9, 1.5e-8, lmax]);
            assert_eq!(c.n_dropped, 2, "f={f}: cut {}", c.cut);
            assert!(c.keeps(lj));
            kinds.push(c.kind);
        }
        assert_eq!(kinds, [LindepCutKind::Nominal, LindepCutKind::Gap]);
    }

    /// The real erf(0.33) metric of sulfamethoxazole / def2-universal-jkfit,
    /// whose nearest eigenvalue sits 1.9 δ (`dsyev`) / 2.7 δ (`dsyevd`) from
    /// 1e-10: both solvers must give the rule the same kept count with every
    /// eigenvalue at least `m` from the cut. The old cut's margin is asserted
    /// to be below 5 δ, so the test cannot pass on a metric that was never
    /// tight. (Danuglipron, where the two solvers' old cuts DO disagree, 1020
    /// against 1019, is the ignored companion below.)
    fn assert_solver_independent_on(rel_xyz: &str) {
        let p = format!(
            "{}/../../testdata/molecules/{rel_xyz}",
            env!("CARGO_MANIFEST_DIR")
        );
        let mol = Molecule::load_xyz_with_charge(&p, 0, 1).unwrap();
        let set = basis::bundled("def2-universal-jkfit").unwrap();
        let dfbs = PreparedBasis::new(&mol, &set).unwrap();
        let v = coulomb_metric_2c(Operator::erf(0.33), &dfbs).unwrap();
        let (l_qr, _) = v.eigh(UPLO::Upper).unwrap();
        let (l_dc, _) = ferric_core::linalg::eigh_dc(&v, ferric_core::linalg::Uplo::Upper).unwrap();
        let l_qr = l_qr.to_vec();
        let (c_qr, c_dc) = (metric_lindep_cut(&l_qr), metric_lindep_cut(&l_dc));
        let delta = c_dc.rounding;
        let old_margin = l_dc
            .iter()
            .fold(f64::INFINITY, |a, &l| a.min((l - TAU).abs()));
        assert!(
            old_margin < 5.0 * delta,
            "not tight: {} delta",
            old_margin / delta
        );
        eprintln!(
            "{rel_xyz}: old drops {}/{} (margin {:.2} delta); rule drops {}/{} at {:.4e} \
             (margin {:.1}/{:.1} delta)",
            old_drop(&l_qr),
            old_drop(&l_dc),
            old_margin / delta,
            c_qr.n_dropped,
            c_dc.n_dropped,
            c_dc.cut,
            c_qr.margin / delta,
            c_dc.margin / delta
        );
        assert_eq!(c_qr.n_dropped, c_dc.n_dropped);
        assert_eq!(
            (c_qr.kind, c_dc.kind),
            (LindepCutKind::Gap, LindepCutKind::Gap)
        );
        for c in [&c_qr, &c_dc] {
            let m = LINDEP_CUT_MARGIN_ROUNDINGS * c.rounding;
            assert!(
                c.margin >= m * (1.0 - 1e-9),
                "margin {} delta",
                c.margin / c.rounding
            );
        }
    }

    #[test]
    fn lindep_cut_is_solver_independent_on_the_sulfamethoxazole_erf_metric() {
        assert_solver_independent_on("sulfamethoxazole.xyz");
    }

    #[test]
    #[ignore = "danuglipron def2-universal-jkfit metric (naux 3635): dsyev alone ~50 s in release"]
    fn lindep_cut_is_solver_independent_on_the_danuglipron_erf_metric() {
        assert_solver_independent_on("c9_systems/danuglipron/conf_01_pubchem.xyz");
    }
}
