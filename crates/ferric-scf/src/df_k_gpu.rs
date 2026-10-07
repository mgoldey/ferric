//! Device-resident DF-K, occupied path. The dressed tensor `B[P,μ,ν]` (this
//! rank's aux band, in-core) is uploaded ONCE per `DfK` under the pool label
//! "DF-K dressed B"; each `build_from_occ` then moves only `C_occ` up and `K`
//! down (`8·n·nocc` and `8·n²` bytes).
//!
//! Per chunk of `c` aux rows (one strided-batched GEMM, one SYRK):
//!   Y_P = C_occᵀ · B_P         (nocc × n each, stacked: Y is (c·nocc) × n row-major;
//!                               B_P is symmetric, so this is (B_P·C_occ)ᵀ)
//!   K_F += Y_F · Y_Fᵀ          (Y_F = the n × (c·nocc) column-major view, ld = n)
//! K is symmetric; SYRK fills the Fortran-lower triangle (= row-major upper, the
//! CPU path's convention) and the host mirrors upper → lower after one n² download.
//! Chunks run in ascending aux order on one stream, accumulating on the device
//! (`beta = 0` for the first chunk, `1` after); the result does not depend on
//! the rayon worker count and is deterministic run to run on one device. It is
//! NOT bit-identical to the CPU path (different summation order): the contract
//! is the two-stage Higham bound of [`k_error_factor`].
//!
//! Error model (Higham, ASNA §3.5; u = 2⁻⁵³, γ_k = ku/(1−ku)): stage 1 has depth
//! n, stage 2 sums `k_chunk + nchunks` terms per element (a SYRK of depth
//! `c·nocc` per chunk, then `nchunks` sequential accumulations). With
//! `S_μν = Σ_{P,i} (|B_P||C|)_μi (|B_P||C|)_νi`:
//! `|K̂ − K| ≤ ε·S`, `ε = 2γ_n + γ_n² + γ_{k_chunk+nchunks}(1+γ_n)²`.
//!
//! Precision: f64 for every `[gpu] precision` setting unless `[gpu] precision =
//! "mixed"` AND the kernel `dfk-occ` is in `mixed_kernels` (`settings().mixed_allows`,
//! decided once per `DfK` when `B` is uploaded). The mixed upload keeps the dressed
//! `B` as f32 (`4·band·n²` bytes, label "DF-K dressed B (f32)") and rounds `C_occ`
//! to f32 on every build; each `Y_P = C_occᵀ·B_P` is the k-panelled SGEMM with f64
//! accumulation (`gemm_f32_f64acc_dev`, panel width `effective_k_panel()`), copied
//! into the f64 chunk scratch `Y`; the SYRK and everything after it are f64. The
//! mixed contract is [`k_error_factor_mixed`]. A finite `B` element beyond
//! `f32::MAX` (`GpuError::F32Range`) or a flush kernel that will not load
//! (`GpuError::Kernel`) runs the f64 device upload for that `DfK` instead, counted
//! once (`mixed_fallback_f64`), never the CPU. A `C_occ` element beyond `f32::MAX`
//! cannot fall back (the resident `B` is f32): the build errors and the dispatcher
//! declines to the CPU as for any build error (counted under `gemm_cpu_f32_range`).
//! The mixed bound assumes IEEE binary32 SGEMM (cuBLAS default math mode, no TF32)
//! and operands in the f32 normal range (the underflow term is omitted).
//! `resident_bytes` is the f64 figure; the mixed charge is `4·band·n²` + `8·n²`.
//! Open-shell builds reuse the alpha-sized scratch for beta: every mixed copy and
//! slice uses `nocc`, never the scratch capacity.
//! Where C2's budget row will need more than this file measures: the shipping bar
//! is |dE_total| <= 1.0e-3 Eh with error growing no faster than linearly in N, so
//! the map needs the error per atom and the slope of ln(error) vs ln(N) with its
//! standard error and degrees of freedom, over the dfk-occ ladder (H2O, butane,
//! octane, dodecane, benzene aDZ/aTZ, UKS, RSH); the tests here pin the kernel,
//! not that map.
use std::sync::Arc;

use ferric_core::gpu::batched::{
    dev_batched_left, dev_batched_right, gemm_f64_strided_batched_dev, syrk_f64_dev, BatchedDims,
};
use ferric_core::gpu::device::{device, Device, GpuError};
use ferric_core::gpu::gemm::{dev_left_padded, dev_right_padded};
use ferric_core::gpu::mixed::{effective_k_panel, gemm_f32_f64acc_dev, round_to_f32};
use ferric_core::gpu::mixed_host::{gamma, mixed_error_factor, round_trip_f32, U64};
use ferric_core::gpu::pool::DevicePool;
use ferric_core::gpu::precision::MixedKernel;
use ferric_core::gpu::resident::DeviceMatrix;
use ferric_core::gpu::stats::{self, CpuReason};
use ferric_core::gpu::{GpuMode, GpuStatus};
use ferric_core::parallel::ParallelContext;
use ferric_integrals::three_index_source::ThreeIndexSource;
use ndarray::{Array2, ArrayView2};

/// Upper bound on the half-transform scratch `Y` ((chunk·nocc) × n f64).
/// A default, not a tuned value (override with [`DeviceDfK::with_scratch_bytes`]); at
/// benzene/aug-cc-pVTZ the scratch for ALL 558 aux rows is 38.8 MB, so one chunk covers the band.
pub const SCRATCH_BYTES_DEFAULT: usize = 256 << 20;

seam_flag!(
    /// Test seam (`test-seams` feature): the dispatcher reports "not handled" so
    /// the CPU path runs.
    FORCE_HOST,
    force_host
);
seam_flag!(
    /// Test seam (`test-seams` feature): `build_from_occ` fails after the upload
    /// AND all chunk work, before anything reaches the host `k` (a mid-build CUDA
    /// error stand-in).
    FORCE_BUILD_FAILURE,
    force_build_failure
);
seam_flag!(
    /// Test seam (`test-seams` feature, mutation): upload `B` rounded through
    /// f32. Measures the "defect present" side of the SCF energy gate.
    ROUND_B_TO_F32,
    round_b_to_f32
);
seam_flag!(
    /// Test seam (`test-seams` feature, mutation): a mixed upload rounds `B`
    /// toward zero instead of to nearest. Measures the "defect present" side of
    /// the mixed K gate.
    TRUNCATE_B_TO_F32,
    truncate_b_to_f32
);
/// Test seam (`test-seams` feature): settings used for the `dfk-occ` allowlist
/// decision instead of the installed ones (the kernel is not in `SHIPPED`, so no
/// installed setting can allow it). Only that one decision reads it; mode and
/// device still come from the installed settings.
#[cfg(feature = "test-seams")]
#[doc(hidden)]
pub static MIXED_SETTINGS_OVERRIDE: std::sync::Mutex<Option<ferric_core::gpu::GpuSettings>> =
    std::sync::Mutex::new(None);

/// Precision of the resident `B` and of the half transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DfkPrecision {
    F64,
    Mixed,
}

/// The precision this process allows for DF-K: mixed iff `precision = mixed`
/// and `dfk-occ` is in the allowlist.
fn dfk_precision() -> DfkPrecision {
    if dfk_occ_allowed() {
        DfkPrecision::Mixed
    } else {
        DfkPrecision::F64
    }
}

#[cfg(feature = "test-seams")]
fn dfk_occ_allowed() -> bool {
    let allows = |s: &ferric_core::gpu::GpuSettings| s.mixed_allows(MixedKernel::DfkOcc);
    let over = MIXED_SETTINGS_OVERRIDE
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    match over.as_ref() {
        Some(s) => allows(s),
        None => allows(ferric_core::gpu::settings()),
    }
}

#[cfg(not(feature = "test-seams"))]
fn dfk_occ_allowed() -> bool {
    ferric_core::gpu::settings().mixed_allows(MixedKernel::DfkOcc)
}

/// Aux rows per chunk and the number of chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkPlan {
    pub chunk: usize,
    pub nchunks: usize,
}

fn checked_elems(dims: &[usize], what: &str) -> Result<usize, GpuError> {
    dims.iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))
        .ok_or_else(|| GpuError::Layout(format!("{what}: {dims:?} elements overflow usize")))
}

fn f64_bytes(elems: usize, what: &str) -> Result<usize, GpuError> {
    elems
        .checked_mul(8)
        .ok_or_else(|| GpuError::Layout(format!("{what}: {elems} f64 overflow usize")))
}

/// Bytes the resident state holds for a band: `8·(band·n² + n²)` (B + K accumulator).
pub fn resident_bytes(band_naux: usize, n: usize) -> Result<usize, GpuError> {
    let b = checked_elems(&[band_naux, n, n], "DF-K dressed B")?;
    let k = checked_elems(&[n, n], "DF-K K accumulator")?;
    let total = b
        .checked_add(k)
        .ok_or_else(|| GpuError::Layout("DF-K resident size overflows usize".into()))?;
    f64_bytes(total, "DF-K resident")
}

/// `chunk = min(scratch_bytes / (8·nocc·n), band_naux, i32::MAX / nocc)`; zero
/// sizes plan no work. A budget that cannot hold one aux row is a typed
/// `PoolFull` naming the scratch. Pure: no thread count, no device query.
pub fn chunk_plan(
    band_naux: usize,
    n: usize,
    nocc: usize,
    scratch_bytes: usize,
) -> Result<ChunkPlan, GpuError> {
    if band_naux == 0 || n == 0 || nocc == 0 {
        return Ok(ChunkPlan {
            chunk: 0,
            nchunks: 0,
        });
    }
    let per_aux = f64_bytes(
        checked_elems(&[nocc, n], "DF-K half-transform scratch")?,
        "DF-K half-transform scratch",
    )?;
    let chunk = (scratch_bytes / per_aux)
        .min(band_naux)
        .min(i32::MAX as usize / nocc);
    if chunk == 0 {
        return Err(GpuError::PoolFull {
            label: "DF-K half-transform scratch".into(),
            detail: format!(
                "one aux row needs {per_aux} B but the scratch budget is {scratch_bytes} B"
            ),
        });
    }
    Ok(ChunkPlan {
        chunk,
        nchunks: band_naux.div_ceil(chunk),
    })
}

/// ε of the module docs: `|K̂ − K| ≤ ε·S` for the device path.
pub fn k_error_factor(n: usize, k_chunk: usize, nchunks: usize) -> f64 {
    let gn = gamma(n, U64);
    let gl = gamma(k_chunk.saturating_add(nchunks), U64);
    2.0 * gn + gn * gn + gl * (1.0 + gn) * (1.0 + gn)
}

/// ε of the mixed path: stage 1 is the mixed GEMM's own factor
/// `ε₁ = mixed_error_factor(n, k_panel)` (f32 rounding of both operands, f32
/// panel sums of depth `k_panel`, f64 sum over `⌈n/k_panel⌉` panels; u32 = 2⁻²⁴),
/// stage 2 is the f64 row: `2ε₁ + ε₁² + γ_{k_chunk+nchunks}(u64)·(1+ε₁)²`.
pub fn k_error_factor_mixed(n: usize, k_panel: usize, k_chunk: usize, nchunks: usize) -> f64 {
    let e1 = mixed_error_factor(n, k_panel);
    let gl = gamma(k_chunk.saturating_add(nchunks), U64);
    2.0 * e1 + e1 * e1 + gl * (1.0 + e1) * (1.0 + e1)
}

/// Resident `B`: f64, or f32 (half the bytes) under the mixed kernel.
enum ResidentB {
    F64(DeviceMatrix<f64>),
    F32(DeviceMatrix<f32>),
}

/// Mixed-path scratch: `C_occ` as f32 (n × nocc), the per-P f32 panel product
/// (nocc × n) and the per-P f64 result (nocc × n) that is copied into `Y`.
struct MixedScratch {
    c32: DeviceMatrix<f32>,
    y32: DeviceMatrix<f32>,
    yp: DeviceMatrix<f64>,
}

struct Scratch {
    /// Largest `nocc` this scratch was sized for (UHF α/β reuse it).
    nocc_cap: usize,
    chunk: usize,
    c_dev: DeviceMatrix<f64>,
    y: DeviceMatrix<f64>,
    mixed: Option<MixedScratch>,
}

/// The resident state for one `DfK`.
pub struct DeviceDfK {
    dev: Arc<Device>,
    pool: DevicePool,
    n: usize,
    band: usize,
    b: ResidentB,
    k_dev: DeviceMatrix<f64>,
    scratch: Option<Scratch>,
    /// Zeros, n² long: stride-only host twins for the operand descriptors
    /// (their contents are never read).
    twin: Vec<f64>,
    scratch_bytes: usize,
}

impl DeviceDfK {
    /// Reserve and upload `B` as f64. `b_flat` is `(band_naux × n²)` row-major (the
    /// in-core band from `ThreeIndexSource::incore_flat`). The small K
    /// accumulator is reserved FIRST so a refusal of the large `B` moves no
    /// bytes and leaks nothing; `B` is charged under "DF-K dressed B".
    pub fn upload(
        dev: &Arc<Device>,
        pool: &DevicePool,
        b_flat: &ArrayView2<f64>,
        n: usize,
    ) -> Result<Self, GpuError> {
        Self::upload_with_precision(dev, pool, b_flat, n, DfkPrecision::F64)
    }

    /// [`upload`](Self::upload) with the precision of the resident `B`. `Mixed`
    /// charges `4·band·n²` under "DF-K dressed B (f32)"; an `F32Range` refusal or
    /// an unloadable flush kernel (`GpuError::Kernel`, probed first) runs the f64
    /// upload instead and counts one `mixed_fallback_f64`.
    pub fn upload_with_precision(
        dev: &Arc<Device>,
        pool: &DevicePool,
        b_flat: &ArrayView2<f64>,
        n: usize,
        precision: DfkPrecision,
    ) -> Result<Self, GpuError> {
        let (band, n2) = b_flat.dim();
        if n == 0 || band == 0 || n.checked_mul(n) != Some(n2) {
            return Err(GpuError::Layout(format!(
                "flat dressed B is {band}x{n2}; expected (band x n*n) with n = {n}, both non-zero"
            )));
        }
        resident_bytes(band, n)?;
        let k_dev = DeviceMatrix::<f64>::zeros(dev, pool, "DF-K K accumulator", n, n)?;
        let b = match precision {
            DfkPrecision::F64 => ResidentB::F64(upload_f64_b(dev, pool, b_flat)?),
            DfkPrecision::Mixed => match upload_f32_b(dev, pool, b_flat) {
                Ok(b32) => ResidentB::F32(b32),
                Err(e @ (GpuError::F32Range(_) | GpuError::Kernel(_))) => {
                    if ferric_core::gpu::config::gpu_trace() {
                        eprintln!(
                            "[gpu] DF-K mixed path unavailable ({e}); running f64 on the device"
                        );
                    }
                    stats::note_mixed_fallback();
                    ResidentB::F64(upload_f64_b(dev, pool, b_flat)?)
                }
                Err(e) => return Err(e),
            },
        };
        Ok(Self {
            dev: Arc::clone(dev),
            pool: pool.clone(),
            n,
            band,
            b,
            k_dev,
            scratch: None,
            twin: vec![0.0; n2],
            scratch_bytes: SCRATCH_BYTES_DEFAULT,
        })
    }

    /// `true` when the resident `B` is f32.
    pub fn is_mixed(&self) -> bool {
        matches!(self.b, ResidentB::F32(_))
    }

    /// Override the scratch ceiling (tests force many chunks).
    pub fn with_scratch_bytes(mut self, bytes: usize) -> Self {
        self.scratch = None;
        self.scratch_bytes = bytes;
        self
    }

    fn ensure_scratch(&mut self, nocc: usize) -> Result<(), GpuError> {
        if matches!(&self.scratch, Some(s) if s.nocc_cap >= nocc) {
            return Ok(());
        }
        // Release the old charge BEFORE sizing the new one.
        self.scratch = None;
        let c_bytes = f64_bytes(checked_elems(&[self.n, nocc], "DF-K C_occ")?, "DF-K C_occ")?;
        let mixed = self.is_mixed();
        // Mixed adds C_occ f32 (4·n·nocc) and the per-P panels (4 + 8 bytes · nocc·n).
        let extra = if mixed {
            c_bytes / 2 + c_bytes * 3 / 2
        } else {
            0
        };
        let budget = self
            .scratch_bytes
            .min(self.pool.available_bytes().saturating_sub(c_bytes + extra));
        let plan = chunk_plan(self.band, self.n, nocc, budget)?;
        let c_dev = DeviceMatrix::<f64>::zeros(&self.dev, &self.pool, "DF-K C_occ", self.n, nocc)?;
        let y = DeviceMatrix::<f64>::zeros(
            &self.dev,
            &self.pool,
            "DF-K half-transform scratch",
            plan.chunk * nocc, // ≤ i32::MAX by chunk_plan
            self.n,
        )?;
        let mixed = if mixed {
            let (dev, pool, n) = (&self.dev, &self.pool, self.n);
            Some(MixedScratch {
                c32: DeviceMatrix::<f32>::zeros(dev, pool, "DF-K C_occ (f32)", n, nocc)?,
                y32: DeviceMatrix::<f32>::zeros(dev, pool, "DF-K panel (f32)", nocc, n)?,
                yp: DeviceMatrix::<f64>::zeros(dev, pool, "DF-K panel (f64)", nocc, n)?,
            })
        } else {
            None
        };
        self.scratch = Some(Scratch {
            nocc_cap: nocc,
            chunk: plan.chunk,
            c_dev,
            y,
            mixed,
        });
        Ok(())
    }

    /// `K = Σ_P (B_P C)(B_P C)ᵀ` for this band, written into `k` (n × n, any
    /// layout). `nocc = 0` writes zeros without touching the device. On error `k`
    /// is untouched.
    pub fn build_from_occ(
        &mut self,
        c_occ: &ArrayView2<f64>,
        k: &mut Array2<f64>,
    ) -> Result<ChunkPlan, GpuError> {
        let n = self.n;
        let lay = |s: String| GpuError::Layout(s);
        if c_occ.nrows() != n || k.dim() != (n, n) {
            return Err(lay(format!(
                "C_occ is {}x{} and K is {:?}; both need {n} rows/columns",
                c_occ.nrows(),
                c_occ.ncols(),
                k.dim()
            )));
        }
        let nocc = c_occ.ncols();
        if nocc > n {
            return Err(lay(format!("C_occ has {nocc} columns for {n} functions")));
        }
        if nocc == 0 {
            k.fill(0.0);
            return Ok(ChunkPlan {
                chunk: 0,
                nchunks: 0,
            });
        }
        self.ensure_scratch(nocc)?;
        let n2 = n * n; // checked at upload
        let Self {
            dev,
            b,
            k_dev,
            scratch,
            twin,
            band,
            ..
        } = self;
        let dev: &Device = dev;
        let sc = scratch
            .as_mut()
            .ok_or_else(|| lay("scratch missing".into()))?;
        let c_std = c_occ.as_standard_layout();
        let c_host = c_std
            .as_slice()
            .ok_or_else(|| lay("C_occ not contiguous".into()))?;
        let cuda = |what: &str, e: &dyn std::fmt::Debug| GpuError::Cuda(format!("{what}: {e:?}"));
        let up_bytes = upload_c_occ(dev, sc, c_host)?;
        let twin_b = ArrayView2::from_shape((n, n), &twin[..n2]).map_err(|e| lay(e.to_string()))?;
        let twin_c =
            ArrayView2::from_shape((n, nocc), &twin[..n * nocc]).map_err(|e| lay(e.to_string()))?;
        let c_t = twin_c.t();
        let chunk = sc.chunk;
        let k_panel = effective_k_panel();
        let mut p0 = 0usize;
        while p0 < *band {
            let c = chunk.min(*band - p0);
            let at = ChunkAt { p0, c, n, nocc };
            half_transform_chunk(dev, b, sc, at, (&twin_b, &c_t), k_panel)?;
            let y_view = sc.y.buf().slice(..c * nocc * n);
            syrk_f64_dev(dev, n, c * nocc, &y_view, k_dev.buf_mut(), p0 != 0)?;
            p0 += c;
        }
        // Failure stand-in at the worst point: every chunk has run and the device
        // accumulator holds a (partial-looking) K, but nothing has reached `k`.
        if force_build_failure() {
            return Err(GpuError::Cuda("injected DF-K build failure".into()));
        }
        let mut flat = vec![0.0f64; n2];
        dev.stream
            .memcpy_dtoh(k_dev.buf(), &mut flat)
            .map_err(|e| cuda("D2H K", &e))?;
        dev.stream.synchronize().map_err(|e| cuda("sync", &e))?;
        // SYRK filled one triangle (row-major upper); mirror it.
        for i in 0..n {
            for j in (i + 1)..n {
                flat[j * n + i] = flat[i * n + j];
            }
        }
        let full = ArrayView2::from_shape((n, n), &flat).map_err(|e| lay(e.to_string()))?;
        k.assign(&full);
        stats::note_dfk_build(up_bytes, 8 * flat.len());
        Ok(ChunkPlan {
            chunk,
            nchunks: band.div_ceil(chunk),
        })
    }
}

/// One chunk of the half transform: aux rows `p0..p0+c`, geometry `n`, `nocc`.
#[derive(Clone, Copy)]
struct ChunkAt {
    p0: usize,
    c: usize,
    n: usize,
    nocc: usize,
}

/// Move this build's `C_occ` to the device (f64, or rounded to f32 for the mixed
/// path; a finite element beyond `f32::MAX` is an `F32Range` error because the
/// resident `B` is already f32). Returns the bytes sent.
fn upload_c_occ(dev: &Device, sc: &mut Scratch, c_host: &[f64]) -> Result<usize, GpuError> {
    let cuda = |what: &str, e: &dyn std::fmt::Debug| GpuError::Cuda(format!("{what}: {e:?}"));
    let up = c_host.len();
    match sc.mixed.as_mut() {
        None => {
            dev.stream
                .memcpy_htod(c_host, &mut sc.c_dev.buf_mut().slice_mut(..up))
                .map_err(|e| cuda("H2D C_occ", &e))?;
            Ok(8 * up)
        }
        Some(m) => {
            let c32 = round_to_f32("DF-K C_occ", c_host)?;
            dev.stream
                .memcpy_htod(&c32, &mut m.c32.buf_mut().slice_mut(..up))
                .map_err(|e| cuda("H2D C_occ (f32)", &e))?;
            Ok(4 * up)
        }
    }
}

/// `Y_P = C_occᵀ·B_P` for every aux row of the chunk, stacked into `sc.y`.
fn half_transform_chunk(
    dev: &Device,
    b: &ResidentB,
    sc: &mut Scratch,
    at: ChunkAt,
    twins: (&ArrayView2<f64>, &ArrayView2<f64>),
    k_panel: usize,
) -> Result<(), GpuError> {
    let ChunkAt { p0, c, n, nocc } = at;
    match b {
        ResidentB::F64(b) => {
            let n2 = n * n;
            let right = dev_batched_right(b.buf().slice(p0 * n2..(p0 + c) * n2), twins.0, c, n2)?;
            let left = dev_batched_left(sc.c_dev.buf().slice(..n * nocc), twins.1, c, 0)?;
            let dims = BatchedDims {
                m: nocc,
                k: n,
                n,
                batch: c,
            };
            gemm_f64_strided_batched_dev(dev, dims, &left, &right, sc.y.buf_mut(), nocc * n)
        }
        ResidentB::F32(b) => half_transform_chunk_mixed(dev, b, sc, at, twins, k_panel),
    }
}

/// The mixed per-P loop: each `Y_P` is one `gemm_f32_f64acc_dev` (f32 panels of
/// depth `k_panel`, flushed into f64) into the per-P f64 buffer, then copied into
/// its slot of the chunk scratch `Y` (the GEMM's output must be a whole buffer).
fn half_transform_chunk_mixed(
    dev: &Device,
    b: &DeviceMatrix<f32>,
    sc: &mut Scratch,
    at: ChunkAt,
    twins: (&ArrayView2<f64>, &ArrayView2<f64>),
    k_panel: usize,
) -> Result<(), GpuError> {
    let ChunkAt { p0, c, n, nocc } = at;
    let n2 = n * n;
    let Scratch { y, mixed, .. } = sc;
    let m = mixed
        .as_mut()
        .ok_or_else(|| GpuError::Layout("mixed stage without its f32 scratch".into()))?;
    let cuda = |what: &str, e: &dyn std::fmt::Debug| GpuError::Cuda(format!("{what}: {e:?}"));
    for p in 0..c {
        let left = dev_left_padded(m.c32.buf().slice(..n * nocc), twins.1)?;
        let right = dev_right_padded(b.buf().slice((p0 + p) * n2..(p0 + p + 1) * n2), twins.0)?;
        let panels = gemm_f32_f64acc_dev(
            dev,
            nocc,
            n,
            n,
            &left,
            &right,
            m.y32.buf_mut(),
            m.yp.buf_mut(),
            k_panel,
        )?;
        stats::note_mixed(panels, 0, 0);
        let mut slot = y.buf_mut().slice_mut(p * nocc * n..(p + 1) * nocc * n);
        dev.stream
            .memcpy_dtod(&m.yp.buf().slice(..nocc * n), &mut slot)
            .map_err(|e| cuda("D2D Y_P", &e))?;
    }
    Ok(())
}

fn upload_f64_b(
    dev: &Device,
    pool: &DevicePool,
    b_flat: &ArrayView2<f64>,
) -> Result<DeviceMatrix<f64>, GpuError> {
    DeviceMatrix::<f64>::upload(dev, pool, "DF-K dressed B", b_flat)
}

/// `x` rounded to f32 toward zero (the TRUNCATE_B_TO_F32 mutant), as f64.
fn truncate_to_f32(x: f64) -> f64 {
    let r = x as f32;
    if f64::from(r).abs() > x.abs() {
        f64::from(f32::from_bits(r.to_bits() - 1))
    } else {
        f64::from(r)
    }
}

/// The f32-resident `B`. The flush kernel is probed first so a missing kernel
/// (`GpuError::Kernel`) moves no bytes; `upload_rounded` refuses `F32Range`
/// before touching the pool.
fn upload_f32_b(
    dev: &Device,
    pool: &DevicePool,
    b_flat: &ArrayView2<f64>,
) -> Result<DeviceMatrix<f32>, GpuError> {
    dev.axpy_f32_to_f64()?;
    if truncate_b_to_f32() {
        let t = b_flat.mapv(truncate_to_f32);
        return DeviceMatrix::<f32>::upload_rounded(dev, pool, "DF-K dressed B (f32)", &t.view());
    }
    DeviceMatrix::<f32>::upload_rounded(dev, pool, "DF-K dressed B (f32)", b_flat)
}

/// Per-`DfK` device state. `Declined` is sticky (and has released the device memory).
#[derive(Default)]
pub enum DeviceSlot {
    #[default]
    Untried,
    Ready(Box<DeviceDfK>),
    Declined,
}

pub(crate) fn real_mpi_world(ctx: Option<&ParallelContext>) -> bool {
    #[cfg(feature = "mpi")]
    {
        ctx.is_some_and(|c| c.world().is_some())
    }
    #[cfg(not(feature = "mpi"))]
    {
        let _ = ctx;
        false
    }
}

/// Why this source can never use the device path, or `None`.
fn ineligible(src: &ThreeIndexSource, ctx: Option<&ParallelContext>) -> Option<&'static str> {
    if !src.is_incore() {
        return Some(
            "the dressed tensor is spilled or recomputed (the device path needs the in-core band)",
        );
    }
    if real_mpi_world(ctx) {
        return Some(
            "MPI ranks would share one device ordinal (a device per rank is not available yet)",
        );
    }
    None
}

/// The first line of an error's Display: the pool's refusal embeds a multi-line
/// occupancy report, which stays in the typed error and out of the notices.
fn one_line(e: &GpuError) -> String {
    let full = e.to_string();
    full.lines().next().unwrap_or_default().to_string()
}

fn decline(slot: &mut DeviceSlot, reason: &str) {
    eprintln!("[ferric] gpu: DF-K stays on the CPU ({reason})");
    stats::note_dfk_declined(reason);
    *slot = DeviceSlot::Declined;
}

fn count_cpu_fallback(e: &GpuError) {
    let reason = match e {
        GpuError::PoolFull { .. } => CpuReason::PoolFull,
        GpuError::Layout(_) => CpuReason::Layout,
        GpuError::F32Range(_) => CpuReason::F32Range,
        _ => CpuReason::CudaError,
    };
    stats::note_cpu_detail(reason, &format!("(DF-K occupied path: {})", one_line(e)));
}

/// Upload the in-core band (rounded through f32 only under the mutation seam).
fn upload_band(
    dev: &Arc<Device>,
    pool: &DevicePool,
    src: &ThreeIndexSource,
) -> Result<DeviceDfK, GpuError> {
    let flat = src
        .incore_flat()
        .ok_or_else(|| GpuError::Layout("dressed tensor is not in core".into()))?;
    let owned;
    let view = if round_b_to_f32() {
        owned = round_trip_f32(&flat);
        owned.view()
    } else {
        flat
    };
    DeviceDfK::upload_with_precision(dev, pool, &view, src.nao(), dfk_precision())
}

/// Try the device. `true` means `k` holds this rank's band contribution (the
/// caller then runs the cross-rank reduction exactly as for the CPU path);
/// `false` means run the CPU path (`k` untouched by this call). Never panics;
/// never touches CUDA when the mode is off.
pub fn try_build_from_occ(
    slot: &mut DeviceSlot,
    src: &ThreeIndexSource,
    ctx: Option<&ParallelContext>,
    c_occ: &Array2<f64>,
    k: &mut Array2<f64>,
) -> bool {
    if force_host() || matches!(slot, DeviceSlot::Declined) {
        return false;
    }
    let settings = ferric_core::gpu::settings();
    if settings.mode == GpuMode::Off {
        return false;
    }
    let GpuStatus::Ready(info) = ferric_core::gpu::status() else {
        return false;
    };
    if rayon::current_thread_index().is_some() {
        stats::note_cpu(CpuReason::InsideRayonWorker); // transient: the slot is untouched
        return false;
    }
    if src.band_naux() == 0 {
        return false; // an empty band contributes nothing; the CPU loop is a no-op too
    }
    if let Some(why) = ineligible(src, ctx) {
        decline(slot, why);
        return false;
    }
    let Some(pool) = ferric_core::gpu::pool() else {
        return false;
    };
    if matches!(slot, DeviceSlot::Untried) {
        let uploaded = device(info.ordinal).and_then(|dev| upload_band(&dev, &pool, src));
        match uploaded {
            Ok(d) => {
                let (word, bytes) = if d.is_mixed() {
                    ("mixed: f32 B, f64 panels; ", 4.0)
                } else {
                    ("", 8.0)
                };
                eprintln!(
                    "[ferric] gpu: DF-K resident on device {} ({word}{:.3} GB dressed B, {} aux rows, nbf {})",
                    info.ordinal,
                    bytes * (src.band_naux() * src.nao() * src.nao()) as f64 / 1e9,
                    src.band_naux(),
                    src.nao()
                );
                *slot = DeviceSlot::Ready(Box::new(d));
            }
            Err(e) => {
                count_cpu_fallback(&e);
                decline(slot, &one_line(&e));
                return false;
            }
        }
    }
    let outcome = match slot {
        DeviceSlot::Ready(d) => d.build_from_occ(&c_occ.view(), k),
        _ => return false,
    };
    match outcome {
        Ok(_) => true,
        Err(e) => {
            count_cpu_fallback(&e);
            decline(slot, &one_line(&e));
            false
        }
    }
}
