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
//!
//! # Storage of the dressed tensor (`[scf] ri_k_precision`)
//!
//! `B[P,μν]` is exactly symmetric in μν (the dressing computes the lower
//! triangle and mirrors it), and both contractions read only that triangle or
//! can be handed it unpacked one aux chunk at a time, so on the CPU the band
//! is held as the packed lower triangle
//! (`ferric_integrals::three_index_source::dressed_packed`): half the square
//! tensor at f64, a quarter at f32 ([`RiKPrecision`](crate::df_k::RiKPrecision)). The square f64 tensor is
//! kept when a `[gpu]` mode may build K on the device (its upload reads it);
//! a band that does not fit in core is spilled as before.
//!
//! **f64** (the default) is bit-identical to the square tensor: the occupied
//! path unpacks each `B_P` into the lower triangle of an `n × n` buffer, exactly
//! the triangle DSYMM (`UPLO = 'U'` on the row-major data) reads, and the
//! density path unpacks each chunk to the full square; chunks, GEMMs and the
//! fold are those of the square tier (`tests/dfk_packed_storage.rs`).
//!
//! **f32 storage** (opt-in, `ri_k_precision = "f32"`): each stored element is
//! the nearest f32 of the f64 value, `|B̂ − B| ≤ u32|B| + η` (`u32 = 2⁻²⁴`,
//! `η = 2⁻¹⁵⁰`), widened to f64 before the same f64 GEMMs, so with
//! `A_P = |B_P||C|`, `ε = u32 + γ_n(1+u32)`, elementwise
//! `|K̂ − K| ≤ ((2ε + ε²) + 2γ_{n+naux·nocc}(1+ε)²)·Σ_P A_P A_Pᵀ` (tested, with
//! a twin check against the test's own sums over the f32 values). No f64 copy
//! is kept, so there is no ladder and no f64 exit: the SCF converges to the
//! stationary point of the rounded-`B̂` functional `E'`, and with `D*` / `D'`
//! the f64 / f32 minima, `E'_K(D') − E_K(D') ≤ E_f32 − E_f64 ≤ E'_K(D*) −
//! E_K(D*)`, both sides first order in `u32` (tested on water). MEASURED (CLI,
//! def2-SVP, def2-universal-jkfit, density_conv 1e-8): `|E_f32 − E_f64|` 2.6e-9
//! to 9.6e-9 Ha for water (B3LYP, PBE0, CAM-B3LYP, UKS B3LYP cation) and 1.0e-8
//! to 1.08e-7 Ha for sulfamethoxazole (same functionals, UKS triplet), same
//! iteration counts; 2.1e-7 Ha for HF water / cc-pVDZ (wiki notes
//! dfk-memory-2026-11-08).
//!
//! Why f32 is not the default: the rounding pattern is re-drawn at every
//! geometry (an element moves by far more than its rounding when an atom moves
//! 1e-3 Bohr), so the f32 ENERGY is not a smooth function of the geometry at the
//! `|E_f32 − E_f64|` level. MEASURED, HF water / cc-pVDZ, h = 1e-3 Bohr: central
//! FD of the f32 energy misses the analytic gradient by 2.0e-4 Ha/Bohr (f64:
//! 1.3e-7), against a jitter bound `|E_f32 − E_f64|/h` = 2.1e-4. The analytic
//! gradient (`crate::df_gradient`) recomputes the f64 integrals: it is the
//! gradient of the f64 functional at the f32 SCF's density, 7.5e-9 Ha/Bohr from
//! the f64 SCF's gradient there (`tests/df_jk_gradient.rs`,
//! `hf_ri_jk_f32_gradient_*`). So f32 suits single points and analytic-gradient
//! work; anything that differences energies across geometries (FD gradients or
//! Hessians, energy line searches, tight scans) sees noise of `|E_f32 −
//! E_f64|`. Removing it needs an f64 energy exit (the f64 functional evaluated
//! once at the f32 density, which is second order in `u32`); that is not
//! implemented.

use crate::fock::KBuilder;
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::basis_bridge::PreparedBasis;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ferric_integrals::operator::Operator;
use ferric_integrals::three_index_source::{
    DressedLayout, DressedRows, F32Rounding, PackedDressOutcome, ThreeIndexSource,
};
use ferric_integrals::threeindex::coulomb_metric_2c;
use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, Array3, ArrayView3};

mod response;
pub use response::{factored_pass_width, OccHalfTransform};

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

/// How the SCF's DF-K (RI-K) V^{-1/2}-dressed 3-index tensor is stored:
/// `[scf] ri_k_precision` (CLI), `ri_k_precision=` (Python),
/// `RhfConfig::ri_k_precision` (library).
///
/// Both values hold the tensor as the packed lower μν triangle on the CPU
/// (`ferric_integrals::three_index_source::dressed_packed`), and K is
/// contracted in f64 throughout (each stored row is widened to f64 and fed to
/// the same DSYMM/DSYRK or GEMMs). Only the STORAGE precision differs:
///
/// * `F64` (the default): the packed f64 tensor, half the square one this
///   builder held before 2026-11-08. K is bit-for-bit the square tensor's, so
///   an f64 SCF reproduces the former default bit for bit.
/// * `F32` (opt-in): one round-to-nearest f32 per stored element, a quarter of
///   the square tensor; a first-order energy change and an energy that is not
///   smooth in the geometry at that level (module docs, "f32 storage").
///
/// With a `[gpu]` mode that may build K on the device, the square f64 tensor
/// the device upload reads is kept. A tensor that does not fit in core is
/// spilled (packed f64 on disk) as before, whatever this says. What was used
/// is recorded in [`crate::result::DfJkRoute::ri_k_precision`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RiKPrecision {
    /// One round-to-nearest f32 per stored element of the dressed tensor.
    F32,
    /// The f64 tensor (packed; K bit-identical to the square f64 tensor).
    F64,
}

impl RiKPrecision {
    /// The default for an unset precision on a CPU path.
    pub const DEFAULT: Self = Self::F64;

    /// Strict parse of the TOML / Python spelling: `"f32"` or `"f64"`.
    pub fn parse_config_str(s: &str) -> Result<Self, FerricError> {
        match s {
            "f32" => Ok(Self::F32),
            "f64" => Ok(Self::F64),
            other => Err(FerricError::General(format!(
                "unknown ri_k_precision {other:?} (expected \"f32\" or \"f64\")"
            ))),
        }
    }

    /// The TOML / Python spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    /// The precision an SCF of `nbf` basis functions requests for
    /// `requested`: an explicit value wins; unset is [`Self::DEFAULT`], except
    /// that a `[gpu]` mode that sends K to the device at this size keeps the
    /// f64 tensor the device path needs.
    pub fn resolve(requested: Option<Self>, nbf: usize) -> Self {
        requested.unwrap_or_else(|| {
            if device_may_take_k(nbf) {
                Self::F64
            } else {
                Self::DEFAULT
            }
        })
    }
}

impl std::fmt::Display for RiKPrecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Could the device path build this K at `nbf` basis functions? Its upload
/// reads the square in-core f64 tensor, so a packed tensor would only make it
/// decline. `auto` answers by the size gate
/// ([`ferric_core::gpu::routes_to_device`]).
fn device_may_take_k(nbf: usize) -> bool {
    #[cfg(feature = "gpu")]
    {
        ferric_core::gpu::routes_to_device(
            ferric_core::gpu::settings().mode,
            ferric_core::gpu::AutoPath::DfK,
            ferric_core::gpu::DeviceSite::plain(nbf),
        )
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = nbf;
        false
    }
}

const RIK_UNSET: u8 = u8::MAX;
static DFK_F32_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(RIK_UNSET);

/// Force the f32/f64 storage of every DfK built from now on (`Some(true)`
/// f32, `Some(false)` f64), whatever precision its caller passes, or return
/// to the caller's precision (`None`). Test hook: process-wide, so tests that
/// use it serialise.
#[doc(hidden)]
pub fn set_dfk_f32_for_test(on: Option<bool>) {
    let v = match on {
        None => RIK_UNSET,
        Some(false) => 0,
        Some(true) => 1,
    };
    DFK_F32_OVERRIDE.store(v, std::sync::atomic::Ordering::SeqCst);
}

/// What a builder asked for `precision` stores at, after the test hook.
fn effective_precision(precision: RiKPrecision) -> RiKPrecision {
    match DFK_F32_OVERRIDE.load(std::sync::atomic::Ordering::SeqCst) {
        0 => RiKPrecision::F64,
        1 => RiKPrecision::F32,
        _ => precision,
    }
}

const LAYOUT_UNSET: u8 = u8::MAX;
static DFK_SQUARE_OVERRIDE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(LAYOUT_UNSET);

/// Force the square f64 tensor (the layout before 2026-11-08) for every f64
/// DfK built from now on (`true`), or return to the packed layout (`false`).
/// Test hook for the bit-identity tests: process-wide, so tests that use it
/// serialise.
#[doc(hidden)]
pub fn set_dfk_square_for_test(on: bool) {
    DFK_SQUARE_OVERRIDE.store(u8::from(on), std::sync::atomic::Ordering::SeqCst);
}

fn square_forced() -> bool {
    DFK_SQUARE_OVERRIDE.load(std::sync::atomic::Ordering::SeqCst) == 1
}

seam_flag!(
    /// Test seam (`test-seams` feature, mutation): the f32 tier rounds the
    /// dressed tensor toward zero instead of to nearest.
    TRUNCATE_DFK_F32,
    truncate_dfk_f32
);
seam_flag!(
    /// Test seam (`test-seams` feature, mutation): the packed tiers' occupied
    /// path unpacks each row's lower triangle shifted by one column within the
    /// row (a wrong packed index map).
    SKEW_DFK_PACKED_UNPACK,
    skew_dfk_packed_unpack
);

/// DF-K exchange builder. Caches the metric-dressed 3-center source
/// `B = M (Q|μν)`, `MᵀM = V⁻¹` (`M = L⁻¹` or the truncated `V^{-1/2}`, see
/// [`DfkMetricRoute`]); the
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
    /// Budget-bounded dressed 3-center source B\[P,μ,ν\] = Σ_Q M_{PQ} (Q|μν)
    /// — holds only this rank's aux band `[band_p0, band_p1)`; or the
    /// pair-major tensor shared with RI-J ([`crate::df_pair`]).
    dressed: Dressed,
    /// Parallel context for the final cross-rank K reduction. `None` → serial /
    /// non-MPI (no reduction, full band).
    #[allow(dead_code)]
    ctx: Option<&'a ParallelContext>,
    /// The memory budget this source was built under — also caps the K
    /// reduction band scratch via `resolve_band_bytes`.
    budget_bytes: usize,
    /// The storage precision the dressed tensor actually has (after any
    /// fallback): `F32` only for the packed f32 tier.
    precision: RiKPrecision,
    /// The layout the dressed band was built in (square f64 when the packed
    /// dressing was not taken, whatever the reason), resident or spilled.
    layout: DressedLayout,
    /// The layout the request and the inputs alone determine
    /// ([`DfK::nominal_layout`]), whatever the pool did.
    nominal_layout: DressedLayout,
    /// How the metric was factored for the dressing.
    metric_route: DfkMetricRoute,
    /// Device-resident copy state (feature `gpu`); `Untried` until the first
    /// occupied-path build asks for it. See `crate::df_k_gpu`.
    #[cfg(feature = "gpu")]
    device: crate::df_k_gpu::DeviceSlot,
}

/// Where a [`DfK`] holds its dressed tensor. The pair-major tensor answers
/// the aux-major streaming calls by unpacking one block at a time, so every
/// consumer that is not the occupied path (the density path, the CPHF half
/// transform) reads it unchanged.
// One per DfK, never in a collection: boxing the common (source) variant
// would only add an indirection.
#[allow(clippy::large_enum_variant)]
enum Dressed {
    /// The aux-major source (packed, square or spilled).
    Source(ThreeIndexSource),
    /// The pair-major tensor shared with RI-J.
    Pair(std::sync::Arc<crate::df_pair::PairMajor>),
}

/// Bytes one unpacked block of the pair-major tensor may take.
const PAIR_UNPACK_BLOCK_BYTES: usize = 256 * 1024 * 1024;

impl Dressed {
    fn nao(&self) -> usize {
        match self {
            Dressed::Source(s) => s.nao(),
            Dressed::Pair(t) => t.nao(),
        }
    }

    fn band_naux(&self) -> usize {
        match self {
            Dressed::Source(s) => s.band_naux(),
            Dressed::Pair(t) => t.naux(),
        }
    }

    fn dressed_packed(&self) -> Option<DressedRows<'_>> {
        match self {
            Dressed::Source(s) => s.dressed_packed(),
            Dressed::Pair(_) => None,
        }
    }

    fn resident_tensor_bytes(&self) -> usize {
        match self {
            Dressed::Source(s) => s.resident_tensor_bytes(),
            Dressed::Pair(t) => t.bytes(),
        }
    }

    fn release_scratch(&mut self) {
        if let Dressed::Source(s) = self {
            s.release_scratch();
        }
    }

    /// Bytes of streaming scratch held between passes (the pair-major tensor
    /// keeps none: its unpack block lives for one pass).
    #[cfg(test)]
    fn scratch_bytes(&self) -> usize {
        match self {
            Dressed::Source(s) => s.scratch_bytes(),
            Dressed::Pair(_) => 0,
        }
    }

    /// Is the tensor the aux-major in-core square band? Never the pair-major one.
    #[cfg(test)]
    fn is_incore(&self) -> bool {
        match self {
            Dressed::Source(s) => s.is_incore(),
            Dressed::Pair(_) => false,
        }
    }

    /// [`ThreeIndexSource::for_each_block`]; the pair-major tensor is unpacked
    /// into dense symmetric blocks whose height is a multiple of twice the
    /// DF-K chunk, so chunks fall as they do for a resident band.
    fn for_each_block(
        &mut self,
        mut f: impl FnMut(ferric_integrals::three_index_source::AuxBlock<'_>) -> Result<(), FerricError>,
    ) -> Result<(), FerricError> {
        let t = match self {
            Dressed::Source(s) => return s.for_each_block(f),
            Dressed::Pair(t) => t,
        };
        let (n, naux) = (t.nao(), t.naux());
        let grain = 2 * ferric_integrals::three_index_source::df_chunk_width(n);
        let rows = (PAIR_UNPACK_BLOCK_BYTES / (n * n * 8).max(1) / grain * grain)
            .max(grain)
            .min(naux.max(1));
        let _charge = ferric_core::memory::pool::try_reserve_global(
            "DF pair-major unpack scratch",
            rows * n * n * 8,
        );
        let mut scratch = Array3::<f64>::zeros((rows, n, n));
        let mut p0 = 0usize;
        while p0 < naux {
            let p1 = (p0 + rows).min(naux);
            if p1 - p0 != scratch.shape()[0] {
                scratch = Array3::<f64>::zeros((p1 - p0, n, n));
            }
            t.unpack_square(p0, p1, &mut scratch);
            f(ferric_integrals::three_index_source::AuxBlock {
                p0,
                data: scratch.view(),
            })?;
            p0 = p1;
        }
        Ok(())
    }
}

impl DfK<'static> {
    /// DF-K on the pair-major tensor `tensor` ([`crate::df_pair`]), shared
    /// with the run's RI-J. Single rank; f64; the Cholesky metric route.
    pub(crate) fn from_pair_major(
        tensor: std::sync::Arc<crate::df_pair::PairMajor>,
        budget_bytes: usize,
    ) -> Self {
        DfK {
            dressed: Dressed::Pair(tensor),
            ctx: None,
            budget_bytes,
            precision: RiKPrecision::F64,
            layout: DressedLayout::PackedF64,
            nominal_layout: DressedLayout::PackedF64,
            metric_route: DfkMetricRoute::Cholesky,
            #[cfg(feature = "gpu")]
            device: crate::df_k_gpu::DeviceSlot::default(),
        }
    }

    /// The pair-major tensor for an RI-JK run that asks for it
    /// (`FERRIC_DF_LAYOUT=pair`) and qualifies, else `None` (the caller takes
    /// the aux-major route): the Coulomb operator, f64 K storage, no device K,
    /// no geometry task capturing the metric, a metric the Cholesky route
    /// certifies, and a tensor that fits `budget_bytes`. `thresh` is the
    /// Schwarz threshold of the pair screen (`[scf] integral_thresh`).
    pub(crate) fn try_pair_major(
        op: Operator,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        budget_bytes: usize,
        thresh: f64,
        precision: RiKPrecision,
    ) -> Result<Option<std::sync::Arc<crate::df_pair::PairMajor>>, FerricError> {
        use crate::df_pair::{PairLayout, PairMajor};
        use ferric_core::timing::Stage;
        if !crate::df_pair::pair_layout_requested()? {
            return Ok(None);
        }
        let nao = obs.nbasis();
        if op.kind != ferric_integrals::operator::OperatorKind::Coulomb
            || device_may_take_k(nao)
            || square_forced()
            || effective_precision(precision) != RiKPrecision::F64
            || crate::raw_handoff::metric_capture_active()
            || effective_metric_setting()? != DfkMetricSetting::Auto
        {
            return Ok(None);
        }
        let v = coulomb_metric_2c(op, dfbs)?;
        let t = Stage::start("dfk:metric-cholesky");
        let linv = certified_cholesky_factor(&v);
        t.end();
        let Some(linv) = linv else {
            return Ok(None);
        };
        let q_aux_max = v.diag().iter().fold(0.0f64, |m, x| m.max(x.abs().sqrt()));
        let t = Stage::start("dfpair:schwarz");
        let layout = PairLayout::schwarz(op, obs, q_aux_max, thresh)?;
        t.end();
        if layout.tensor_bytes(dfbs.nbasis()) > budget_bytes {
            return Ok(None);
        }
        let t = Stage::start("dfpair:fill");
        let mut tensor = PairMajor::fill(op, obs, dfbs, layout)?;
        t.end();
        let t = Stage::start("dfpair:dress");
        tensor.dress(&linv)?;
        t.end();
        crate::df_pair::note_pair_major_taken();
        Ok(Some(std::sync::Arc::new(tensor)))
    }

    /// Whether this DF-K reads the pair-major tensor.
    pub fn is_pair_major(&self) -> bool {
        matches!(self.dressed, Dressed::Pair(_))
    }
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

/// How [`DfK`] factored its auxiliary metric `V` for the dressing
/// `B = M (Q|μν)`. Any `M` with `MᵀM = V⁻¹` gives the same `K = Σ_P B_P D B_P`;
/// the two routes differ in cost and in rounding, never in which modes are
/// fitted. MEASURED: at a fixed density `K` agrees to about 1e-13 relative
/// (water/def2-SVP); converged RHF energies of C8–C16 alkanes agree within
/// 3e-11 Ha at `energy_conv = 1e-10`, part of which is convergence noise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DfkMetricRoute {
    /// `M = L⁻¹` with `V = L Lᵀ` (LAPACK `dpotrf` + `dtrtri`): taken when the
    /// factorization succeeds and its own bound on the smallest eigenvalue of
    /// `V` certifies (to rounding) that `metric_lindep_cut` would drop no mode
    /// (`certified_cholesky_factor`), on every MPI rank. The dressing then
    /// skips the zero upper triangle: half the flops of a dense `M`, a
    /// measured 20–35% of the dressing's wall time (it is partly memory bound).
    Cholesky,
    /// `M = U_s λ_s^{-1/2} U_sᵀ` over the eigenpairs kept by
    /// `metric_lindep_cut` (`dsyevd`): every metric the Cholesky route does
    /// not certify (near-singular ones), the erf (long-range) metric of a
    /// range-separated hybrid, which always has near-null modes and is not
    /// tried, every DF-K built while a geometry task's gradient captures the
    /// metric `eigh` (it reuses this decomposition), any DF-K for which one
    /// MPI rank does not take Cholesky, and `FERRIC_DFK_METRIC=eigh`.
    Eigh,
}

/// The parsed `FERRIC_DFK_METRIC`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DfkMetricSetting {
    /// Cholesky when certified, else `eigh` (the default).
    Auto,
    /// Always the eigendecomposition (the route before 2026-11-23).
    Eigh,
}

impl DfkMetricSetting {
    /// The canonical spelling (`auto` / `eigh`).
    pub fn as_str(self) -> &'static str {
        match self {
            DfkMetricSetting::Auto => "auto",
            DfkMetricSetting::Eigh => "eigh",
        }
    }
}

impl std::fmt::Display for DfkMetricSetting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// `FERRIC_DFK_METRIC`: a debug escape hatch, not a typed option. `auto`
/// (default; Cholesky when certified, else `eigh`) or `eigh` (always the
/// eigendecomposition), case-insensitive, surrounding blanks ignored.
static DFK_METRIC: ferric_core::config::ConfigVar<DfkMetricSetting> =
    ferric_core::config::ConfigVar {
        env_name: "FERRIC_DFK_METRIC",
        default: DfkMetricSetting::Auto,
        parse: |s| match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(DfkMetricSetting::Auto),
            "eigh" => Ok(DfkMetricSetting::Eigh),
            _ => Err(format!(
                "invalid value {s:?}: expected \"auto\" or \"eigh\" (unset means \"auto\")"
            )),
        },
        validate: ferric_core::config::accept_any,
    };

/// The `FERRIC_DFK_METRIC` setting of this process, ignoring the test hook.
/// A malformed value is an input error naming the variable.
pub fn dfk_metric_setting() -> Result<DfkMetricSetting, FerricError> {
    DFK_METRIC
        .get()
        .map(|r| r.value)
        .map_err(|e| FerricError::General(format!("FERRIC_DFK_METRIC: {e}")))
}

thread_local! {
    /// Test override of `FERRIC_DFK_METRIC` on this thread.
    static DFK_METRIC_OVERRIDE: std::cell::Cell<Option<DfkMetricSetting>> =
        const { std::cell::Cell::new(None) };
}

/// Restores the previous [`override_dfk_metric_for_test`] setting on drop.
#[doc(hidden)]
#[must_use = "the override ends when the guard is dropped"]
pub struct DfkMetricOverrideGuard(Option<DfkMetricSetting>);

impl Drop for DfkMetricOverrideGuard {
    fn drop(&mut self) {
        let prev = self.0.take();
        DFK_METRIC_OVERRIDE.with(|c| c.set(prev));
    }
}

/// Use `setting` in place of `FERRIC_DFK_METRIC` for every DfK built on THIS
/// thread until the guard drops. Test hook. THREAD-LOCAL: it does not reach a
/// DfK built on another thread (a rayon worker, another test), so parallel
/// tests do not see each other's setting.
#[doc(hidden)]
pub fn override_dfk_metric_for_test(setting: DfkMetricSetting) -> DfkMetricOverrideGuard {
    DfkMetricOverrideGuard(DFK_METRIC_OVERRIDE.with(|c| c.replace(Some(setting))))
}

/// Run `f` with every DfK built on THIS thread on `setting`, whatever
/// `FERRIC_DFK_METRIC` says (the guess's free-atom solves, see
/// `crate::guess`).
pub(crate) fn with_dfk_metric<R>(setting: DfkMetricSetting, f: impl FnOnce() -> R) -> R {
    let _guard = DfkMetricOverrideGuard(DFK_METRIC_OVERRIDE.with(|c| c.replace(Some(setting))));
    f()
}

fn effective_metric_setting() -> Result<DfkMetricSetting, FerricError> {
    match DFK_METRIC_OVERRIDE.with(|c| c.get()) {
        Some(s) => Ok(s),
        None => dfk_metric_setting(),
    }
}

/// Safety factor on the eigenvalue bound of [`certified_cholesky_factor`].
const CHOLESKY_CERT_FACTOR: f64 = 2.0;

/// `L⁻¹` for `V = L Lᵀ` when the Cholesky route is certified (to rounding) to
/// fit exactly what the `eigh` route fits, else `None` (the caller takes
/// `eigh`).
///
/// The `eigh` route keeps every mode iff [`metric_lindep_cut`] drops none,
/// which holds when every eigenvalue is at least
/// `τ + m` (`τ` = [`DFK_LINDEP_THRESH`], `m` = [`LINDEP_CUT_MARGIN_ROUNDINGS`]
/// `· ε · max|λ|`): the nominal cut is then safe and below the spectrum. From
/// the factor itself, `1/λ_min = ‖V⁻¹‖₂ = ‖L⁻¹‖₂² ≤ min(‖L⁻¹‖_F², ‖L⁻¹‖₁‖L⁻¹‖_∞)`
/// and `max|λ| ≤ ‖V‖_∞`. The computed `L⁻¹` is the exact factor of `V + E`
/// with `‖E‖ ≲ c · naux · ε · ‖V‖`, which moves `λ_min` by as much, so that
/// term joins the margin: the route is certified when the bound is at least
/// [`CHOLESKY_CERT_FACTOR`] `· (τ + (100 + naux) · ε · ‖V‖_∞)`. This
/// certifies no mode is dropped up to the rounding of the factorization
/// itself, which is not a proof in exact arithmetic but sits far inside the
/// eigensolver's own rounding of the cut. The bound is loose by at most a
/// factor `naux`, so a well-conditioned metric whose `λ_min` lies within
/// that factor of the cut also goes to `eigh` (correct, only slower).
/// `dpotrf` failing (an indefinite metric) and any non-finite norm also
/// return `None`.
pub(crate) fn certified_cholesky_factor(v: &Array2<f64>) -> Option<Array2<f64>> {
    let linv = with_blas_threads(opt_in_blas_threads(), || {
        ferric_core::linalg::cholesky_inverse_lower(v)
    })
    .ok()?;
    let abs_row_max = |a: &Array2<f64>| {
        a.rows()
            .into_iter()
            .map(|r| r.iter().map(|x| x.abs()).sum::<f64>())
            .fold(0.0f64, f64::max)
    };
    let abs_col_max = |a: &Array2<f64>| {
        a.columns()
            .into_iter()
            .map(|c| c.iter().map(|x| x.abs()).sum::<f64>())
            .fold(0.0f64, f64::max)
    };
    let frob2: f64 = linv.iter().map(|x| x * x).sum();
    let norm_product = abs_row_max(&linv) * abs_col_max(&linv);
    let v_inf = abs_row_max(v);
    if !(frob2.is_finite() && norm_product.is_finite() && v_inf.is_finite()) {
        return None;
    }
    let lambda_min_lower = 1.0 / frob2.min(norm_product);
    let roundings = LINDEP_CUT_MARGIN_ROUNDINGS + v.nrows() as f64;
    let need = CHOLESKY_CERT_FACTOR * (DFK_LINDEP_THRESH + roundings * f64::EPSILON * v_inf);
    (lambda_min_lower.is_finite() && lambda_min_lower >= need).then_some(linv)
}

/// Whether every rank of `ctx` votes for Cholesky (a logical AND across
/// ranks); `local` alone without MPI or on one rank. All ranks must dress
/// with the same KIND of factor: `K` sums rows of `B` over the ranks' bands,
/// and rows of `L⁻¹(Q|μν)` mixed with rows of `V^{-1/2}(Q|μν)` are no fit at
/// all. The vote can differ between ranks (the certificate's rounding
/// follows each rank's BLAS thread count, the env var or the capture may
/// differ), so it is reduced here.
fn all_ranks_vote_cholesky(ctx: Option<&ParallelContext>, local: bool) -> bool {
    #[cfg(feature = "mpi")]
    if let Some(world) = ctx.filter(|c| c.size > 1).and_then(|c| c.world()) {
        use mpi::traits::CommunicatorCollectives;
        let mine = [i32::from(local)];
        let mut all = [0i32];
        world.all_reduce_into(
            &mine[..],
            &mut all[..],
            mpi::collective::SystemOperation::min(),
        );
        return all[0] == 1;
    }
    #[cfg(not(feature = "mpi"))]
    let _ = ctx;
    local
}

/// The dressing factor `M` of DF-K's metric and the route that produced it,
/// agreed across the ranks of `ctx`.
fn dfk_metric_factor(
    op: Operator,
    dfbs: &PreparedBasis,
    v: &Array2<f64>,
    ctx: Option<&ParallelContext>,
) -> Result<(Array2<f64>, DfkMetricRoute), FerricError> {
    dfk_metric_factor_agreed(op, dfbs, v, |local| all_ranks_vote_cholesky(ctx, local))
}

/// [`dfk_metric_factor`] with the cross-rank vote injected (`agree` maps this
/// rank's vote to the collective one), so a disagreement can be tested
/// without MPI.
fn dfk_metric_factor_agreed(
    op: Operator,
    dfbs: &PreparedBasis,
    v: &Array2<f64>,
    agree: impl FnOnce(bool) -> bool,
) -> Result<(Array2<f64>, DfkMetricRoute), FerricError> {
    // A malformed setting still votes (no) before it errors, so no rank
    // waits on one that left the collective.
    let setting = effective_metric_setting();
    // A geometry task's gradient reuses this DF-K's `eigh` (raw_handoff), so
    // there the eigendecomposition is not extra work and keeps the energy
    // and gradient on one decomposition. The erf metric of a range-separated
    // hybrid always has near-null modes in a JK-fit basis, so the factorization
    // would be refused: it is not tried.
    let attempt = matches!(setting, Ok(DfkMetricSetting::Auto))
        && !crate::raw_handoff::metric_capture_active()
        && op.kind != ferric_integrals::operator::OperatorKind::ErfCoulomb;
    let linv = if attempt {
        let t_st = ferric_core::timing::Stage::start("dfk:metric-cholesky");
        let linv = certified_cholesky_factor(v);
        t_st.end();
        linv
    } else {
        None
    };
    let agreed = agree(linv.is_some());
    setting?;
    match linv {
        Some(linv) if agreed => Ok((linv, DfkMetricRoute::Cholesky)),
        _ => Ok((v_inv_sqrt_lindep(op, dfbs, v)?, DfkMetricRoute::Eigh)),
    }
}

fn v_inv_sqrt_lindep(
    op: Operator,
    dfbs: &PreparedBasis,
    v: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
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
    drop(u_scaled);
    // The gradient that follows a geometry-task SCF diagonalises this same
    // metric; hand it this `eigh` (kept only inside a capturing solver,
    // `crate::raw_handoff::with_metric_capture`).
    crate::raw_handoff::offer_metric(op, dfbs, v, evals, evecs);
    Ok(m) // (naux, naux)
}

impl<'a> DfK<'a> {
    /// Build the DF-K cache from orbital and auxiliary bases (FULL aux range,
    /// serial / single-rank — byte-identical to the pre-MPI path).
    ///
    /// Factors the (P|Q) metric as `M` with `MᵀM = V⁻¹` ([`DfkMetricRoute`]:
    /// the Cholesky `L⁻¹` when certified, else the truncated symmetric
    /// `V^{-1/2} = U · diag(λ^{-1/2}) · U^T`), then forms
    /// `B[P,μ,ν] = Σ_Q M_{PQ} (Q|μν)` via ThreeIndexSource::build_dressed,
    /// honouring `budget_bytes`.
    ///
    /// Stores the tensor at [`RiKPrecision::F64`]; [`DfK::new_with_precision`]
    /// chooses.
    pub fn new(
        op: Operator,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        budget_bytes: usize,
    ) -> Result<Self, FerricError> {
        Self::new_with_precision(op, obs, dfbs, budget_bytes, RiKPrecision::F64)
    }

    /// [`DfK::new`] storing the dressed tensor at `precision`.
    pub fn new_with_precision(
        op: Operator,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        budget_bytes: usize,
        precision: RiKPrecision,
    ) -> Result<Self, FerricError> {
        Self::new_banded_with_precision(op, obs, dfbs, budget_bytes, None, precision)
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
        Self::new_banded_with_precision(op, obs, dfbs, budget_bytes, ctx, RiKPrecision::F64)
    }

    /// [`DfK::new_banded`] storing the dressed tensor at `precision`.
    pub fn new_banded_with_precision(
        op: Operator,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        budget_bytes: usize,
        ctx: Option<&'a ParallelContext>,
        precision: RiKPrecision,
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
        let (factor, route) = dfk_metric_factor(op, dfbs, &v, ctx)?;
        Self::from_raw(
            &mut raw,
            dfbs,
            (factor, route),
            budget_bytes,
            ctx,
            (p0, p1),
            precision,
        )
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
        Self::from_full_raw_with_precision(raw, obs, dfbs, op, budget_bytes, ctx, RiKPrecision::F64)
    }

    /// [`DfK::from_full_raw`] storing the dressed tensor at `precision`.
    pub fn from_full_raw_with_precision(
        raw: &mut ThreeIndexSource,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        op: Operator,
        budget_bytes: usize,
        ctx: Option<&'a ParallelContext>,
        precision: RiKPrecision,
    ) -> Result<Self, FerricError> {
        let _ = obs;
        let v = coulomb_metric_2c(op, dfbs)?;
        let (factor, route) = dfk_metric_factor(op, dfbs, &v, ctx)?;
        let naux = dfbs.nbasis();
        let (p0, p1) = match ctx {
            Some(c) => c.aux_band(naux),
            None => (0, naux),
        };
        Self::from_raw(
            raw,
            dfbs,
            (factor, route),
            budget_bytes,
            ctx,
            (p0, p1),
            precision,
        )
    }

    /// Shared tail of both constructors: dress `raw` into this rank's band,
    /// held packed at `precision` when the packed tier takes it (CPU, the band
    /// fits, and raw resident or, for f64, streamed once), else in the square /
    /// spilled layouts.
    fn from_raw(
        raw: &mut ThreeIndexSource,
        dfbs: &PreparedBasis,
        (v_inv_sqrt, metric_route): (Array2<f64>, DfkMetricRoute),
        budget_bytes: usize,
        ctx: Option<&'a ParallelContext>,
        (p0, p1): (usize, usize),
        precision: RiKPrecision,
    ) -> Result<Self, FerricError> {
        let naux = dfbs.nbasis();
        let nao = raw.nao();
        let requested = effective_precision(precision);
        let t_st = ferric_core::timing::Stage::start("dfk:dress");
        let mut packed = None;
        let forced_square =
            device_may_take_k(nao) || (requested == RiKPrecision::F64 && square_forced());
        // The layout the request and the inputs determine: square only when
        // forced, else packed at the request, or packed f64 after an
        // input-driven (out-of-f32-range) fallback.
        let mut nominal_layout = if forced_square {
            DressedLayout::Square
        } else if requested == RiKPrecision::F32 {
            DressedLayout::PackedF32
        } else {
            DressedLayout::PackedF64
        };
        if !forced_square {
            let layout = match requested {
                RiKPrecision::F32 => DressedLayout::PackedF32,
                RiKPrecision::F64 => DressedLayout::PackedF64,
            };
            let rounding = if truncate_dfk_f32() {
                F32Rounding::TowardZero
            } else {
                F32Rounding::Nearest
            };
            match ThreeIndexSource::build_dressed_band_packed(
                raw,
                &v_inv_sqrt,
                budget_bytes,
                (p0, p1),
                layout,
                rounding,
            )? {
                PackedDressOutcome::Built(src) => packed = Some((*src, requested)),
                PackedDressOutcome::F32OutOfRange { row, col, value } => {
                    nominal_layout = DressedLayout::PackedF64;
                    ferric_core::gpu::stats::note_mixed_fallback(
                        ferric_core::gpu::MixedKernel::DfkF32,
                        ferric_core::gpu::stats::MixedFallback::F32Range,
                    );
                    eprintln!(
                        "[ferric] RI-K: ri_k_precision = \"f32\" but the dressed tensor stays \
                         f64 (B[{row}, {col}] = {value:e} is beyond the f32 range)"
                    );
                    if let PackedDressOutcome::Built(src) =
                        ThreeIndexSource::build_dressed_band_packed(
                            raw,
                            &v_inv_sqrt,
                            budget_bytes,
                            (p0, p1),
                            DressedLayout::PackedF64,
                            rounding,
                        )?
                    {
                        packed = Some((*src, RiKPrecision::F64));
                    }
                }
                PackedDressOutcome::NotTaken => {}
            }
        } else if requested == RiKPrecision::F32 {
            eprintln!(
                "[ferric] RI-K: ri_k_precision = \"f32\" but the dressed tensor stays f64 \
                 (a [gpu] mode that may build K on the device needs the f64 square tensor)"
            );
        }
        let (dressed, stored, layout) = match packed {
            Some((src, RiKPrecision::F32)) => (src, RiKPrecision::F32, DressedLayout::PackedF32),
            Some((src, RiKPrecision::F64)) => (src, RiKPrecision::F64, DressedLayout::PackedF64),
            None => {
                if requested == RiKPrecision::F32 && !device_may_take_k(nao) {
                    ferric_core::gpu::stats::note_mixed_fallback(
                        ferric_core::gpu::MixedKernel::DfkF32,
                        ferric_core::gpu::stats::MixedFallback::CpuF64,
                    );
                    eprintln!(
                        "[ferric] RI-K: ri_k_precision = \"f32\" but the dressed tensor stays \
                         f64 (it is not held in memory as a packed band: the raw tensor is not \
                         resident, or the band does not fit the budget and the pool)"
                    );
                }
                let d = if p0 == 0 && p1 == naux {
                    // Full band: identical call to the serial path (byte-identical B).
                    ThreeIndexSource::build_dressed(raw, &v_inv_sqrt, budget_bytes)?
                } else {
                    ThreeIndexSource::build_dressed_band(raw, &v_inv_sqrt, budget_bytes, p0, p1)?
                };
                (d, RiKPrecision::F64, DressedLayout::Square)
            }
        };
        // The caller owns `raw`; DfK retains only the dressed tensor. Dressing
        // is the one streaming pass over `raw`; a packed `raw` that moves on
        // into DfJ must not keep its unpacked scratch block for the whole SCF.
        raw.release_scratch();
        t_st.end();

        let ctx = ctx.filter(|c| c.size > 1);
        Ok(DfK {
            dressed: Dressed::Source(dressed),
            ctx,
            budget_bytes,
            precision: stored,
            layout,
            nominal_layout,
            metric_route,
            #[cfg(feature = "gpu")]
            device: crate::df_k_gpu::DeviceSlot::default(),
        })
    }

    /// How the auxiliary metric was factored for the dressing
    /// ([`DfkMetricRoute`]).
    pub fn metric_route(&self) -> DfkMetricRoute {
        self.metric_route
    }

    /// Bytes of the packed dressed band of `band` aux rows at `precision`
    /// that [`DfK`] would hold in core on the CPU, or 0 when it would keep the
    /// square tensor (a `[gpu]` mode that may build K on the device).
    pub fn packed_band_bytes(band: usize, nao: usize, precision: RiKPrecision) -> usize {
        let precision = effective_precision(precision);
        if device_may_take_k(nao) || (precision == RiKPrecision::F64 && square_forced()) {
            return 0;
        }
        let layout = match precision {
            RiKPrecision::F32 => DressedLayout::PackedF32,
            RiKPrecision::F64 => DressedLayout::PackedF64,
        };
        ThreeIndexSource::dressed_band_bytes(band, nao, layout)
    }

    /// Would a [`DfK`] at `precision` dress a STREAMED (spilled or recomputed)
    /// raw tensor into its packed band? Only the f64 packed tier does
    /// (`ThreeIndexSource::build_dressed_band_packed`); the f32 tier and the
    /// square tensor of a device mode need raw resident.
    pub(crate) fn dresses_streamed_raw_packed(precision: RiKPrecision, nbf: usize) -> bool {
        !device_may_take_k(nbf)
            && effective_precision(precision) == RiKPrecision::F64
            && !square_forced()
    }

    /// The layout this DfK's dressed band was built in: packed f32, packed
    /// f64 (after an f32 fallback too) or the square f64 tensor (a device
    /// mode, the square hook, or a packed dressing that was not taken). A
    /// property of the object, whether the band is resident or spilled.
    pub fn stored_layout(&self) -> DressedLayout {
        self.layout
    }

    /// Nominal bytes of this DfK's dressed band in its stored layout
    /// ([`DfK::stored_layout`]), resident or not. Unlike
    /// [`DfK::packed_band_bytes`] (0 for the square layout, the room a raw
    /// placement reserves), this is the band's own size in every layout.
    pub fn stored_band_bytes(&self) -> usize {
        if let Dressed::Pair(t) = &self.dressed {
            return t.bytes();
        }
        ThreeIndexSource::dressed_band_bytes(
            self.dressed.band_naux(),
            self.dressed.nao(),
            self.layout,
        )
    }

    /// The layout the request and the inputs alone give this DfK's band: the
    /// square f64 tensor only when it is forced (a device K mode or the square
    /// hook), else packed at the requested precision, or packed f64 when f32
    /// fell back because a value is beyond the f32 range. Unlike
    /// [`DfK::stored_layout`] it ignores the square fallback taken when the
    /// pool cannot hold the packed band, so it depends on the shapes, the
    /// request and the budget only.
    pub fn nominal_layout(&self) -> DressedLayout {
        self.nominal_layout
    }

    /// Nominal bytes of this DfK's band in its [`DfK::nominal_layout`]: what a
    /// deterministic memory decision (the f32 RI-J conversion beside it)
    /// counts. When the pool forced the square layout the real band is larger
    /// by at most the square minus the packed size.
    pub fn nominal_band_bytes(&self) -> usize {
        if let Dressed::Pair(t) = &self.dressed {
            return t.bytes();
        }
        ThreeIndexSource::dressed_band_bytes(
            self.dressed.band_naux(),
            self.dressed.nao(),
            self.nominal_layout,
        )
    }

    /// The storage precision of the dressed tensor (after any fallback).
    pub fn precision(&self) -> RiKPrecision {
        self.precision
    }

    /// Bytes of the dressed tensor resident in core (0 when spilled).
    pub fn resident_tensor_bytes(&self) -> usize {
        self.dressed.resident_tensor_bytes()
    }

    /// Is the dressed tensor held as the packed lower triangle (f64 or f32)?
    pub fn is_packed(&self) -> bool {
        self.dressed.dressed_packed().is_some()
    }

    /// The packed dressed band (rows band-local); `None` for the square and
    /// spilled tiers. Test hook.
    #[doc(hidden)]
    pub fn dressed_rows_for_test(&self) -> Option<DressedRows<'_>> {
        self.dressed.dressed_packed()
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

    /// Device bytes this builder's resident tensor will take: the dressed band
    /// plus the K accumulator when the dressed tensor is the in-core square
    /// band and the `[gpu]` mode routes DF-K to the device at this size (the
    /// gate the device build itself applies); 0 otherwise (packed, spilled or
    /// recomputed, or a K that stays on the CPU), so the RI-J upload leaves
    /// the pool only for a K that can take it.
    #[cfg(feature = "gpu")]
    pub fn device_footprint_bytes(&self) -> usize {
        let Dressed::Source(src) = &self.dressed else {
            return 0;
        };
        if !src.is_incore() || !device_may_take_k(src.nao()) {
            return 0;
        }
        crate::df_k_gpu::resident_bytes(self.dressed.band_naux(), self.dressed.nao()).unwrap_or(0)
    }

    /// The in-core dressed band as `(band_naux × n²)`; `None` when spilled. Test hook.
    #[cfg(feature = "gpu")]
    #[doc(hidden)]
    pub fn dressed_incore_flat_for_test(&self) -> Option<ndarray::ArrayView2<'_, f64>> {
        match &self.dressed {
            Dressed::Source(src) => src.incore_flat(),
            Dressed::Pair(_) => None,
        }
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
        let budget_bytes = self.budget_bytes;
        // Packed tiers: the band is one resident block whose chunks are
        // unpacked to the square one at a time (one more c·n² per worker).
        // The chunks are the square tier's chunks (same width, from row 0 of
        // the band), so K is the square tier's bit for bit at f64.
        if let Some(rows) = self.dressed.dressed_packed() {
            if matches!(rows, DressedRows::F32(_)) {
                ferric_core::gpu::stats::note_kernel_mixed(ferric_core::gpu::MixedKernel::DfkF32);
            }
            let b = rows.nrows();
            let n_chunks = b.div_ceil(chunk);
            let workers = rayon::current_num_threads().max(1).min(n_chunks.max(1));
            let band_bytes = crate::reduce::band_bytes_after_worker_scratch(
                crate::reduce::resolve_band_bytes(budget_bytes),
                4 * chunk * n * n,
                workers,
            );
            crate::reduce::grouped_deterministic_sum(k, n_chunks, n, band_bytes, |ci| {
                let q0 = ci * chunk;
                let q1 = (q0 + chunk).min(b);
                let mut sq = Array3::<f64>::zeros((q1 - q0, n, n));
                for (p, mut dst) in sq.outer_iter_mut().enumerate() {
                    let dst = dst
                        .as_slice_mut()
                        .ok_or_else(|| FerricError::General("DfK: non-contiguous unpack".into()))?;
                    rows.unpack_square_into(q0 + p, dst, n);
                }
                density_chunk_k(n, d, sq.view())
            })?;
            self.reduce_k_across_ranks(k);
            return Ok(0);
        }
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
                crate::reduce::resolve_band_bytes(budget_bytes),
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
                    density_chunk_k(n, d, blk.data.slice(ndarray::s![q0..q1, .., ..]))
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
        // The pair-major tensor: one GEMM per AO function, then the
        // triangular reduction (`crate::df_pair`).
        if let Dressed::Pair(t) = &self.dressed {
            t.build_k_occ(c_occ, k)?;
            self.reduce_k_across_ranks(k);
            return Ok(0);
        }

        #[cfg(feature = "gpu")]
        if let Dressed::Source(src) = &self.dressed {
            if crate::df_k_gpu::try_build_from_occ(&mut self.device, src, self.ctx, c_occ, k) {
                self.reduce_k_across_ranks(k);
                return Ok(0);
            }
        }

        // Aux-chunk width: same BLAS3-efficiency vs per-task-scratch tradeoff as
        // `build`, but the half-transform scratch scales with nocc, not n, so the
        // trailing GEMM operand is (c·nocc) wide.
        let chunk = ferric_integrals::three_index_source::df_chunk_width(n);
        let budget_bytes = self.budget_bytes;
        // Packed tiers: the band is one resident block; each P is unpacked into
        // the lower triangle of one n×n buffer per task (the triangle DSYMM
        // reads), so the chunks, their GEMMs and the fold are the square
        // tier's, bit for bit at f64.
        if let Some(rows) = self.dressed.dressed_packed() {
            if matches!(rows, DressedRows::F32(_)) {
                ferric_core::gpu::stats::note_kernel_mixed(ferric_core::gpu::MixedKernel::DfkF32);
            }
            let b = rows.nrows();
            let n_chunks = b.div_ceil(chunk);
            let workers = rayon::current_num_threads().max(1).min(n_chunks.max(1));
            let band_bytes = crate::reduce::band_bytes_after_worker_scratch(
                crate::reduce::resolve_band_bytes(budget_bytes),
                n * chunk * nocc + n * n,
                workers,
            );
            crate::reduce::grouped_deterministic_sum(k, n_chunks, n, band_bytes, |ci| {
                let q0 = ci * chunk;
                let q1 = (q0 + chunk).min(b);
                occ_chunk_k(n, c_occ, OccChunk::Packed(rows, q0, q1))
            })?;
            self.reduce_k_across_ranks(k);
            return Ok(0);
        }
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
                crate::reduce::resolve_band_bytes(budget_bytes),
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
                    occ_chunk_k(
                        n,
                        c_occ,
                        OccChunk::Square(blk.data.slice(ndarray::s![q0..q1, .., ..])),
                    )
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
        if let Some(ctx) = self.ctx {
            ctx.all_reduce_sum_array(k);
        }
    }
}

/// One aux chunk's density-path K partial: `Σ_{P∈chunk} B_P D B_Pᵀ` as two
/// wide GEMMs on repacked operands ([`DfK::build`](KBuilder::build)).
fn density_chunk_k(
    n: usize,
    d: &Array2<f64>,
    bchunk: ArrayView3<'_, f64>,
) -> Result<Array2<f64>, FerricError> {
    let c = bchunk.shape()[0];
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
}

/// Where an occupied-path chunk's `B_P` come from.
enum OccChunk<'a> {
    /// `(c, n, n)` square rows (in-core square tier or a streamed block).
    Square(ArrayView3<'a, f64>),
    /// Rows `[q0, q1)` of a packed band (f64 or f32), unpacked per P.
    Packed(DressedRows<'a>, usize, usize),
}

/// One aux chunk's occupied-path K partial `Σ_{P∈chunk} M_P M_Pᵀ`,
/// `M_P = B_P C_occ` ([`DfK::build_from_occ`](KBuilder::build_from_occ)).
#[allow(clippy::manual_c_str_literals)] // b"L\0" byte literals match dsyrk_'s *const u8 args; c".." would be *const i8
fn occ_chunk_k(
    n: usize,
    c_occ: &Array2<f64>,
    src: OccChunk<'_>,
) -> Result<Array2<f64>, FerricError> {
    let nocc = c_occ.ncols();
    let c = match &src {
        OccChunk::Square(v) => v.shape()[0],
        OccChunk::Packed(_, q0, q1) => q1 - q0,
    };
    let m_chunk = occ_half_panel(n, c_occ, &src)?;
    let ni = n as i32;
    let ldw = (c * nocc) as i32;
    let one = 1.0f64;
    let zero = 0.0f64;
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
    // DSYRK wrote only one triangle (see the layout note in `occ_half_panel`).
    for i in 0..n {
        for j in (i + 1)..n {
            kc[[j, i]] = kc[[i, j]];
        }
    }
    Ok(kc)
}

/// The half-transformed panel `[M_{P0} | M_{P0+1} | …]` (`n × c·nocc`,
/// `M_P = B_P C_occ`) of one aux chunk: the first stage of [`occ_chunk_k`]
/// and the `L_P` of [`DfK::occupied_half_transform`].
#[allow(clippy::manual_c_str_literals)] // b"R\0" byte literals match dsymm_'s *const u8 args; c".." would be *const i8
fn occ_half_panel(
    n: usize,
    c_occ: &Array2<f64>,
    src: &OccChunk<'_>,
) -> Result<Array2<f64>, FerricError> {
    let nocc = c_occ.ncols();
    let c = match src {
        OccChunk::Square(v) => v.shape()[0],
        OccChunk::Packed(_, q0, q1) => q1 - q0,
    };
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
    //    B symmetric ⇒ UPLO is immaterial to the result. UPLO='U'
    //    reads the Fortran upper triangle = the row-major LOWER
    //    triangle, which is exactly what a packed row holds: the
    //    packed tiers unpack only that triangle.
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
    // The packed tiers' unpack buffer: only its lower triangle is written
    // (and read, by DSYMM); the upper stays zero.
    let mut unpacked = match src {
        OccChunk::Square(_) => Vec::new(),
        OccChunk::Packed(..) => vec![0.0f64; n * n],
    };
    for p in 0..c {
        let bp_slice: &[f64] = match src {
            OccChunk::Square(v) => {
                // `blk.data` is standard-layout, so this row slice is
                // contiguous; `to_slice` therefore never falls back.
                v.slice_move(ndarray::s![p, .., ..])
                    .to_slice()
                    .ok_or_else(|| FerricError::General("DfK occ: non-contiguous B row".into()))?
            }
            OccChunk::Packed(rows, q0, _) => {
                rows.unpack_lower_into(q0 + p, &mut unpacked, n);
                if skew_dfk_packed_unpack() && n > 1 {
                    // Mutation: every lower-triangle row read one column late.
                    for mu in 1..n {
                        unpacked.copy_within(mu * n..mu * n + mu, mu * n + 1);
                    }
                }
                &unpacked
            }
        };
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
    Ok(m_chunk)
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

    /// `DfK` on `op` with the metric route forced (`eigh`) or left to `auto`.
    fn dfk_on_route(
        op: Operator,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        eigh: bool,
    ) -> DfK<'static> {
        let _route = override_dfk_metric_for_test(if eigh {
            DfkMetricSetting::Eigh
        } else {
            DfkMetricSetting::Auto
        });
        DfK::new(op, obs, dfbs, usize::MAX).unwrap()
    }

    /// `DfK` dressed with `factor` (the constructors' shared tail).
    fn dfk_from_factor(
        op: Operator,
        obs: &PreparedBasis,
        dfbs: &PreparedBasis,
        factor: (Array2<f64>, DfkMetricRoute),
    ) -> DfK<'static> {
        let mut raw = ThreeIndexSource::build(op, obs, dfbs, usize::MAX).unwrap();
        let naux = dfbs.nbasis();
        DfK::from_raw(
            &mut raw,
            dfbs,
            factor,
            usize::MAX,
            None,
            (0, naux),
            RiKPrecision::F64,
        )
        .unwrap()
    }

    /// `v` with its smallest eigenvalue replaced by `lambda_min`.
    fn with_lambda_min(v: &Array2<f64>, lambda_min: f64) -> Array2<f64> {
        let (mut ev, u) =
            ferric_core::linalg::eigh_dc(v, ferric_core::linalg::Uplo::Upper).unwrap();
        ev[0] = lambda_min;
        let mut us = u.clone();
        for (k, &l) in ev.iter().enumerate() {
            us.column_mut(k).mapv_inplace(|x| x * l);
        }
        let w = us.dot(&u.t());
        (&w + &w.t()) * 0.5
    }

    fn benzene_svp_jkfit() -> (PreparedBasis, PreparedBasis) {
        let p = format!(
            "{}/../../testdata/molecules/benzene.xyz",
            env!("CARGO_MANIFEST_DIR")
        );
        let mol = Molecule::load_xyz_with_charge(&p, 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("def2-svp").unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        (obs, dfbs)
    }

    /// A positive definite metric (`dpotrf` succeeds) whose smallest
    /// eigenvalue sits just above the cut: the `eigh` route keeps every mode,
    /// but the certificate cannot vouch for it, so `auto` takes `eigh` and
    /// its K is the forced-eigh K bit for bit. The real benzene/def2-SVP
    /// Coulomb metric (def2-universal-jkfit) with only its smallest eigenvalue
    /// moved to 1.5e-10.
    #[test]
    fn a_metric_the_certificate_refuses_but_eigh_keeps_whole_takes_eigh() {
        let (obs, dfbs) = benzene_svp_jkfit();
        let op = Operator::coulomb();
        let v = with_lambda_min(&coulomb_metric_2c(op, &dfbs).unwrap(), 1.5e-10);
        let (ev, _) = ferric_core::linalg::eigh_dc(&v, ferric_core::linalg::Uplo::Upper).unwrap();
        eprintln!("naux {}, lambda_min {:e}", ev.len(), ev[0]);
        assert!(ev.len() > 300, "a realistic naux");
        assert!(ev[0] > 0.0);
        assert_eq!(metric_lindep_cut(&ev).n_dropped, 0, "eigh keeps every mode");
        assert!(
            ferric_core::linalg::cholesky_inverse_lower(&v).is_ok(),
            "dpotrf must succeed, so the certificate is what refuses"
        );
        assert!(certified_cholesky_factor(&v).is_none());

        let auto = {
            let _g = override_dfk_metric_for_test(DfkMetricSetting::Auto);
            dfk_metric_factor(op, &dfbs, &v, None).unwrap()
        };
        assert_eq!(auto.1, DfkMetricRoute::Eigh);
        let eigh = {
            let _g = override_dfk_metric_for_test(DfkMetricSetting::Eigh);
            dfk_metric_factor(op, &dfbs, &v, None).unwrap()
        };
        assert_eq!(eigh.1, DfkMetricRoute::Eigh);
        let mut auto = dfk_from_factor(op, &obs, &dfbs, auto);
        let mut eigh = dfk_from_factor(op, &obs, &dfbs, eigh);
        assert_eq!(auto.metric_route(), DfkMetricRoute::Eigh);
        let n = obs.nbasis();
        let d = Array2::from_shape_fn((n, n), |(i, j)| 1.0 / (1.0 + (i + j) as f64));
        let (mut k_a, mut k_e) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
        auto.build(&d, &mut k_a).unwrap();
        eigh.build(&d, &mut k_e).unwrap();
        assert!(max_abs(&k_e) > 1e-2);
        assert_eq!(k_a, k_e);
    }

    /// The route is collective: a rank whose own vote is Cholesky (a
    /// certified metric) takes `eigh` when another rank votes no, so no K
    /// mixes rows of `L⁻¹` with rows of `V^{-1/2}`. Two ranks simulated: A
    /// certifies, B is forced to `eigh`; the vote is their AND.
    #[test]
    fn a_rank_disagreement_resolves_to_eigh_on_every_rank() {
        let mol = Molecule::parse_xyz(
            "3
H2O
O 0 0 0
H 0 0 0.96
H 0.93 0 -0.26
",
            0,
            1,
        )
        .unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let v = coulomb_metric_2c(op, &dfbs).unwrap();
        let vote = |setting| {
            let _g = override_dfk_metric_for_test(setting);
            let mut mine = None;
            dfk_metric_factor_agreed(op, &dfbs, &v, |local| {
                mine = Some(local);
                local
            })
            .unwrap();
            mine.unwrap()
        };
        let (vote_a, vote_b) = (vote(DfkMetricSetting::Auto), vote(DfkMetricSetting::Eigh));
        assert!(vote_a && !vote_b, "fixture: A votes Cholesky, B does not");
        let all = vote_a && vote_b;
        for setting in [DfkMetricSetting::Auto, DfkMetricSetting::Eigh] {
            let _g = override_dfk_metric_for_test(setting);
            let (_, route) = dfk_metric_factor_agreed(op, &dfbs, &v, |_| all).unwrap();
            assert_eq!(route, DfkMetricRoute::Eigh, "{setting:?}");
        }
        // And unanimity keeps Cholesky.
        let _g = override_dfk_metric_for_test(DfkMetricSetting::Auto);
        let (_, route) = dfk_metric_factor_agreed(op, &dfbs, &v, |local| local).unwrap();
        assert_eq!(route, DfkMetricRoute::Cholesky);
    }

    /// The erf metric is not factored at all (it always has near-null modes).
    #[test]
    fn the_erf_metric_does_not_try_cholesky() {
        let mol = Molecule::parse_xyz(
            "3
H2O
O 0 0 0
H 0 0 0.96
H 0.93 0 -0.26
",
            0,
            1,
        )
        .unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let op = Operator::erf(0.33);
        let v = coulomb_metric_2c(op, &dfbs).unwrap();
        let _g = override_dfk_metric_for_test(DfkMetricSetting::Auto);
        let mut vote = None;
        let (_, route) = dfk_metric_factor_agreed(op, &dfbs, &v, |l| {
            vote = Some(l);
            l
        })
        .unwrap();
        assert_eq!((vote, route), (Some(false), DfkMetricRoute::Eigh));
    }

    #[test]
    fn the_metric_setting_parses_case_and_blanks_and_names_a_bad_value() {
        let parse = DFK_METRIC.parse;
        assert_eq!(parse(" AUTO ").unwrap(), DfkMetricSetting::Auto);
        assert_eq!(parse("eigh\n").unwrap(), DfkMetricSetting::Eigh);
        let err = parse("cholesky").unwrap_err();
        assert!(err.contains("\"auto\" or \"eigh\""), "{err}");
    }

    /// `J_μν = Σ_P B_P,μν Σ_λσ B_P,λσ D_λσ` from the packed dressed rows.
    fn j_from_packed(dfk: &DfK<'_>, d: &Array2<f64>) -> Array2<f64> {
        let Some(DressedRows::F64(b)) = dfk.dressed_rows_for_test() else {
            panic!("expected the packed f64 tier");
        };
        let n = d.nrows();
        let mut w = Vec::with_capacity(b.ncols());
        for mu in 0..n {
            for nu in 0..=mu {
                w.push(if mu == nu {
                    d[(mu, nu)]
                } else {
                    d[(mu, nu)] + d[(nu, mu)]
                });
            }
        }
        let gamma = b.dot(&ndarray::Array1::from(w));
        let jp = b.t().dot(&gamma);
        let mut j = Array2::zeros((n, n));
        let mut k = 0;
        for mu in 0..n {
            for nu in 0..=mu {
                j[(mu, nu)] = jp[k];
                j[(nu, mu)] = jp[k];
                k += 1;
            }
        }
        j
    }

    fn max_abs(a: &Array2<f64>) -> f64 {
        a.iter().fold(0.0f64, |m, v| m.max(v.abs()))
    }

    /// On a well-conditioned Coulomb metric `auto` takes the Cholesky route,
    /// and its `L⁻¹`-dressed tensor gives the same J and K (density and
    /// occupied paths) as the symmetric `V^{-1/2}` one to reassociation noise.
    #[test]
    fn cholesky_dressing_matches_the_eigh_dressing_on_j_and_k() {
        let mol =
            Molecule::parse_xyz("3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\n", 0, 1).unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("def2-svp").unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        let op = Operator::coulomb();
        let n = obs.nbasis();
        let mut chol = dfk_on_route(op, &obs, &dfbs, false);
        let mut eigh = dfk_on_route(op, &obs, &dfbs, true);
        assert_eq!(chol.metric_route(), DfkMetricRoute::Cholesky);
        assert_eq!(eigh.metric_route(), DfkMetricRoute::Eigh);

        // A symmetric, indefinite density (a guess-like D plus a perturbation)
        // and a 5-column occupied factor.
        let c = Array2::from_shape_fn((n, 5), |(i, j)| ((i * 7 + j * 3) as f64 * 0.37).sin());
        let mut d = c.dot(&c.t());
        for i in 0..n {
            for j in 0..n {
                d[(i, j)] += 0.01 * (((i + 2 * j) as f64).cos() + ((j + 2 * i) as f64).cos());
            }
        }

        let (mut k_c, mut k_e) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
        chol.build(&d, &mut k_c).unwrap();
        eigh.build(&d, &mut k_e).unwrap();
        let (dk, sk) = (max_abs(&(&k_c - &k_e)), max_abs(&k_e));
        eprintln!("K density path: max|dK| = {dk:e}, max|K| = {sk:e}");
        assert!(
            sk > 1e-2 && dk <= 1e-12 * sk,
            "K differs by {dk:e} (|K| {sk:e})"
        );

        let (mut k_c, mut k_e) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
        chol.build_from_occ(&c, &mut k_c).unwrap();
        eigh.build_from_occ(&c, &mut k_e).unwrap();
        let (dk, sk) = (max_abs(&(&k_c - &k_e)), max_abs(&k_e));
        eprintln!("K occupied path: max|dK| = {dk:e}, max|K| = {sk:e}");
        assert!(
            sk > 1e-2 && dk <= 1e-12 * sk,
            "K_occ differs by {dk:e} (|K| {sk:e})"
        );

        let (j_c, j_e) = (j_from_packed(&chol, &d), j_from_packed(&eigh, &d));
        let (dj, sj) = (max_abs(&(&j_c - &j_e)), max_abs(&j_e));
        eprintln!("J: max|dJ| = {dj:e}, max|J| = {sj:e}");
        assert!(
            sj > 1e-1 && dj <= 1e-12 * sj,
            "J differs by {dj:e} (|J| {sj:e})"
        );
    }

    /// The certificate refuses a positive definite metric whose smallest
    /// eigenvalue the `eigh` route would cut (`dpotrf` succeeds there), and an
    /// indefinite one, and accepts one safely above the cut.
    #[test]
    fn cholesky_certificate_refuses_a_metric_the_eigh_route_would_cut() {
        let n = 6;
        // A fixed orthogonal Q from the eigenvectors of a symmetric matrix.
        let a = Array2::from_shape_fn((n, n), |(i, j)| ((i * j + i + j) as f64 * 0.7).sin());
        let (_, q) = (&a + &a.t()).eigh(UPLO::Upper).unwrap();
        let with_spectrum = |lam: &[f64]| {
            let mut ql = q.clone();
            for (k, &l) in lam.iter().enumerate() {
                ql.column_mut(k).mapv_inplace(|x| x * l);
            }
            let v = ql.dot(&q.t());
            (&v + &v.t()) * 0.5
        };
        let base = [3.0, 2.0, 1.5, 1.0, 0.5];
        for (lmin, certified) in [
            (1e-11, false), // below the cut: eigh drops it
            (1e-10, false), // at the cut
            (-1e-3, false), // indefinite: dpotrf fails
            (1e-6, true),
            (0.2, true),
        ] {
            let mut lam = base.to_vec();
            lam.push(lmin);
            let v = with_spectrum(&lam);
            let got = certified_cholesky_factor(&v);
            assert_eq!(got.is_some(), certified, "lambda_min {lmin:e}");
            if let Some(linv) = got {
                // L⁻¹ V L⁻ᵀ = I.
                let id = linv.dot(&v).dot(&linv.t());
                let err = max_abs(&(&id - &Array2::<f64>::eye(n)));
                assert!(err < 1e-8, "L^-1 V L^-T - I = {err:e}");
                assert!((0..n).all(|i| (i + 1..n).all(|j| linv[(i, j)] == 0.0)));
            }
            // At the cut itself `metric_lindep_cut` keeps or drops the mode by
            // the sign of rounding noise, so compare only away from it.
            if lmin > 0.0 && (lmin / DFK_LINDEP_THRESH - 1.0).abs() > 1e-3 {
                let (ev, _) =
                    ferric_core::linalg::eigh_dc(&v, ferric_core::linalg::Uplo::Upper).unwrap();
                // Certified exactly when nothing would be dropped (on these spectra).
                assert_eq!(metric_lindep_cut(&ev).n_dropped == 0, certified);
            }
        }
    }

    /// A near-singular metric (erf, whose smooth kernel leaves near-null modes
    /// in a Coulomb-fitted aux basis) is not certified: `auto` falls back to
    /// `eigh` and keeps exactly the modes the eigh route keeps, so its K is
    /// that route's bit for bit.
    #[test]
    fn near_singular_metric_falls_back_to_eigh_with_the_same_kept_rank() {
        let mol = Molecule::parse_xyz(
            "4\nH2O+H\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.26\nH 0.93 0 -0.2600001\n",
            0,
            2,
        )
        .unwrap();
        let obs = PreparedBasis::new(&mol, &basis::bundled("def2-svp").unwrap()).unwrap();
        let dfbs =
            PreparedBasis::new(&mol, &basis::bundled("def2-universal-jkfit").unwrap()).unwrap();
        for op in [Operator::coulomb(), Operator::erf(0.33)] {
            let v = coulomb_metric_2c(op, &dfbs).unwrap();
            let (ev, _) =
                ferric_core::linalg::eigh_dc(&v, ferric_core::linalg::Uplo::Upper).unwrap();
            let cut = metric_lindep_cut(&ev);
            eprintln!(
                "{:?}: naux {}, lambda_min {:e}, eigh drops {}",
                op.kind,
                ev.len(),
                ev[0],
                cut.n_dropped
            );
            assert!(cut.n_dropped > 0, "{:?}: not near-singular", op.kind);
            assert!(certified_cholesky_factor(&v).is_none());
            let n = obs.nbasis();
            let d = Array2::from_shape_fn((n, n), |(i, j)| 1.0 / (1.0 + (i + j) as f64));
            let mut auto = dfk_on_route(op, &obs, &dfbs, false);
            let mut eigh = dfk_on_route(op, &obs, &dfbs, true);
            assert_eq!(auto.metric_route(), DfkMetricRoute::Eigh);
            let (mut k_a, mut k_e) = (Array2::zeros((n, n)), Array2::zeros((n, n)));
            auto.build(&d, &mut k_a).unwrap();
            eigh.build(&d, &mut k_e).unwrap();
            assert_eq!(k_a, k_e);
        }
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
