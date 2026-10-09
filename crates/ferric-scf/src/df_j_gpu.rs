//! Device-resident RI-J. The raw 3-index tensor `Bp[P, μ≥ν]` (packed lower
//! triangle, `pair = n(n+1)/2` columns) is uploaded ONCE per `DfJ` under the pool
//! label "DF-J raw B (packed)"; each `build` then moves only the packed density
//! weights `w` up, `d_P` down, `c = V⁻¹d_P` up and the packed `J` down
//! (`8·(pair + naux)` bytes each way).
//!
//! Per build (one row-major GEMV per pass, row-chunked so no call touches more
//! than [`GEMV_ELEMS_MAX`] elements):
//!   pass 1: `d_P = Bp·w`      (disjoint output slices, `beta = 0`)
//!   host  : `c = V⁻¹d_P`      (the unchanged Cholesky solve of `DfJ`)
//!   pass 2: `J_packed = Bpᵀ·c` (row chunks in ascending order, `beta = 0` then `1`)
//!
//! The upload accepts every host tier: the packed in-core tensor is copied as it
//! is, the unpacked in-core tensor is packed chunkwise on the host (values equal
//! the packed tier's bit for bit), and a spilled or recomputed tensor is streamed
//! once through `for_each_packed_block`. The pool is charged BEFORE any byte moves
//! or any block is recomputed, so a tensor that does not fit declines with a typed
//! `PoolFull` and costs nothing.
//!
//! The result is deterministic run to run on one device and independent of the
//! rayon worker count; it is NOT bit-identical to the CPU path (different summation
//! order). The contract is the classical GEMV bound `|ŷ − y| ≤ γ_k (|A||x|)`
//! (Higham, ASNA §3.5; u = 2⁻⁵³):
//!   pass 1: `k = pair`                                  ([`d_error_factor`])
//!   pass 2: `k = rows_per_call + ncalls` (a GEMV of depth `rows_per_call`, then
//!           `ncalls` sequential accumulations)         ([`j_error_factor`])
//!
//! Precision: f64 unless `[gpu] precision = "mixed"` AND the kernel `dfj-pack` is in
//! `mixed_kernels` (`df_k_gpu::mixed_kernel_allowed`, decided once per `DfJ` when the
//! tensor is uploaded; `dfj-pack` is named but NOT shipped, so only the
//! `MIXED_SETTINGS_OVERRIDE` seam of `df_k_gpu` can allow it today). The mixed upload
//! keeps `Bp` as f32 (`4·naux·pair` bytes, label "DF-J raw B (packed, f32)"); the
//! vectors `w`, `d_P`, `c` and `J` stay f64 and BOTH passes run
//! [`gemv_f32mat_f64_dev`]: f32 matrix, f64 vector, f64 FMA accumulation, no panel
//! flush. `c` is NOT rounded to f32: the kernel multiplies the f32 matrix by the f64
//! vector, so the only roundings beyond the f64 accumulation are the one rounding
//! of each `Bp` entry (the alternative, SGEMV on a rounded `c` with k-panel flushes,
//! adds `2u32 + γ_b(u32)` per pass, ~`b` times larger, for no capacity gain).
//!
//! Per-pass contract (Higham, ASNA §3.5; u32 = 2⁻²⁴, u64 = 2⁻⁵³; `Bp` in the f32
//! normal range, `|B̂ − B| ≤ u32|B|` elementwise):
//!   pass 1: `|d̂_P − d_P| ≤ ε·(|Bp||w|)_P`,   `ε = u32 + γ_pair(u64)(1+u32)`  ([`d_error_factor_mixed`])
//!   pass 2: `|Ĵ_k − J_k| ≤ ε·(|Bp|ᵀ|c|)_k`,  `ε = u32 + γ_naux(u64)(1+u32)`  ([`j_error_factor_mixed`])
//! where `c` is whatever vector the pass is given. The kernel's strided partial
//! sums plus fixed-order reduction are covered because the γ bound holds for every
//! order. End to end, `J = Bpᵀ V⁻¹ Bp w`; to first order in the roundings
//! `dJ = dBᵀ c + Bpᵀ V⁻¹ (dB w)` with `|dB| ≤ u32|Bp|`; with `V = L Lᵀ`,
//! `X = L⁻¹Bp` the second term is `Xᵀ(L⁻¹ dd)`, so
//! `‖dJ‖₂ ≤ ε (‖|Bp|ᵀ|c|‖₂ + ‖X‖₂ ‖V⁻¹‖₂^½ ‖|Bp||w|‖₂)`, `‖X‖₂² = λmax(Bpᵀ V⁻¹ Bp)`
//! (the pass-1 and pass-2 accumulation terms are inside ε). The metric factors
//! `‖V⁻¹‖₂` and `‖X‖₂` are properties of the aux basis, not derived here:
//! `gpu_dfj_mixed.rs` measures them (power iteration through the real Cholesky
//! solve) and the end-to-end J and energy differences are MEASURED figures. A finite `Bp` entry beyond `f32::MAX` (`GpuError::F32Range`) or a
//! kernel that will not load (`GpuError::Kernel`) re-uploads as f64 for that `DfJ`,
//! counted once (`mixed_fallback_f64`), never the CPU.
//!
//! Test seams (`test-seams` feature): `ROUND_B_TO_F32` makes the f64 upload store
//! `Bp` rounded through f32 (the "twin" of the mixed tensor: same values, f64
//! arithmetic), `TRUNCATE_B_TO_F32` makes the mixed upload round toward zero (the
//! defect the twin comparison must catch).
use std::sync::Arc;

use ferric_core::gpu::device::{device, Device, GpuError};
use ferric_core::gpu::gemv::{
    gemv_f32mat_f64_dev, gemv_f64_dev, mixed_gemv_error_factor, GemvSpec,
};
use ferric_core::gpu::mixed::round_to_f32;
use ferric_core::gpu::mixed_host::{gamma, U64};
use ferric_core::gpu::pool::DevicePool;
use ferric_core::gpu::precision::MixedKernel;
use ferric_core::gpu::resident::DeviceMatrix;
use ferric_core::gpu::stats::{self, CpuReason};
use ferric_core::gpu::{GpuMode, GpuStatus};
use ferric_core::parallel::ParallelContext;
use ferric_core::FerricError;
use ferric_integrals::three_index_source::ThreeIndexSource;
use ndarray::{Array1, Array2, ArrayView2};

/// Largest `rows·pair` a single GEMV call is given (elements). A default, not a
/// tuned value; the row chunking exists so a call never indexes past 2³⁰ elements.
pub const GEMV_ELEMS_MAX: usize = 1 << 30;

/// Host staging for packing the unpacked in-core tensor (bytes).
const STAGE_BYTES: usize = 64 << 20;

seam_flag!(
    /// Test seam (`test-seams` feature): the dispatcher reports "not handled" so
    /// the CPU path runs.
    FORCE_HOST,
    force_host
);
seam_flag!(
    /// Test seam (`test-seams` feature): a device build fails after pass 1 (a
    /// mid-build CUDA error stand-in).
    FORCE_BUILD_FAILURE,
    force_build_failure
);

seam_flag!(
    /// Test seam (`test-seams` feature, twin): the f64 upload stores `Bp` rounded
    /// through f32 (values identical to the mixed tensor, arithmetic f64).
    ROUND_B_TO_F32,
    round_b_to_f32
);
seam_flag!(
    /// Test seam (`test-seams` feature, mutation): a mixed upload rounds `Bp`
    /// toward zero instead of to nearest.
    TRUNCATE_B_TO_F32,
    truncate_b_to_f32
);

seam_flag!(
    /// Test seam (`test-seams` feature): the f32 upload reports an `F32Range`
    /// refusal, as a tensor with a value beyond `f32::MAX` would (a real RI-J tensor
    /// cannot be made to hold one).
    INJECT_F32_RANGE,
    inject_f32_range
);

/// Elements converted and sent per host staging step of the f32 / rounded upload
/// (16 MiB of f32).
const CONVERT_ELEMS: usize = 1 << 22;

/// Precision of the resident `Bp` and of both passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DfjPrecision {
    F64,
    Mixed,
}

impl DfjPrecision {
    /// Bytes per resident `Bp` element.
    pub const fn elem_bytes(self) -> usize {
        match self {
            DfjPrecision::F64 => 8,
            DfjPrecision::Mixed => 4,
        }
    }
}

/// The precision this process allows for RI-J: mixed iff `precision = mixed` and
/// `dfj-pack` is in the allowlist.
fn dfj_precision() -> DfjPrecision {
    if crate::df_k_gpu::mixed_kernel_allowed(MixedKernel::DfjPack) {
        DfjPrecision::Mixed
    } else {
        DfjPrecision::F64
    }
}

/// Packed pair count `n(n+1)/2`, or a typed overflow refusal.
pub fn pair_len(n: usize) -> Result<usize, GpuError> {
    n.checked_mul(n.saturating_add(1))
        .map(|v| v / 2)
        .ok_or_else(|| GpuError::Layout(format!("n = {n}: n(n+1)/2 overflows usize")))
}

/// Bytes the f64 resident state holds: `8·(naux·pair + pair + 2·naux + pair)`
/// (`Bp`, `w`, `d_P`, `c`, packed `J`).
pub fn resident_bytes(naux: usize, pair: usize) -> Result<usize, GpuError> {
    resident_bytes_with(naux, pair, DfjPrecision::F64)
}

/// [`resident_bytes`] for a precision: `Bp` at 4 or 8 bytes per element, the four
/// vectors always f64.
pub fn resident_bytes_with(
    naux: usize,
    pair: usize,
    precision: DfjPrecision,
) -> Result<usize, GpuError> {
    let overflow = || GpuError::Layout(format!("DF-J resident {naux}x{pair} overflows usize"));
    let b = naux
        .checked_mul(pair)
        .and_then(|e| e.checked_mul(precision.elem_bytes()))
        .ok_or_else(overflow)?;
    let vecs = pair
        .checked_mul(2)
        .and_then(|p| p.checked_add(naux.checked_mul(2)?))
        .and_then(|e| e.checked_mul(8))
        .ok_or_else(overflow)?;
    b.checked_add(vecs).ok_or_else(overflow)
}

/// Rows per GEMV call and the number of calls.
pub fn row_plan(naux: usize, pair: usize) -> (usize, usize) {
    if naux == 0 || pair == 0 {
        return (0, 0);
    }
    let rows = (GEMV_ELEMS_MAX / pair).clamp(1, naux);
    (rows, naux.div_ceil(rows))
}

/// ε of pass 1: `|d̂_P − d_P| ≤ ε·(|Bp||w|)_P`.
pub fn d_error_factor(pair: usize) -> f64 {
    gamma(pair, U64)
}

/// ε of pass 2: `|Ĵ_k − J_k| ≤ ε·(|Bp|ᵀ|c|)_k`, for `rows_per_call`, `ncalls` from
/// [`row_plan`].
pub fn j_error_factor(rows_per_call: usize, ncalls: usize) -> f64 {
    gamma(rows_per_call.saturating_add(ncalls), U64)
}

/// ε of pass 1 on the f32-resident tensor: `|d̂_P − d_P| ≤ ε·(|Bp||w|)_P`.
pub fn d_error_factor_mixed(pair: usize) -> f64 {
    mixed_gemv_error_factor(pair)
}

/// ε of pass 2 on the f32-resident tensor: `|Ĵ_k − J_k| ≤ ε·(|Bp|ᵀ|c|)_k`.
pub fn j_error_factor_mixed(naux: usize) -> f64 {
    mixed_gemv_error_factor(naux)
}

/// The resident raw tensor.
enum ResidentB {
    F64(DeviceMatrix<f64>),
    F32(DeviceMatrix<f32>),
}

struct Bufs {
    b: ResidentB,
    w: DeviceMatrix<f64>,
    d: DeviceMatrix<f64>,
    c: DeviceMatrix<f64>,
    j: DeviceMatrix<f64>,
}

/// The resident state for one `DfJ`.
pub struct DeviceDfJ {
    dev: Arc<Device>,
    naux: usize,
    pair: usize,
    rows_per_call: usize,
    /// `None` for an empty tensor (nothing to reserve or launch).
    bufs: Option<Bufs>,
    /// Aux rows written so far by [`write_rows`](Self::write_rows).
    filled: usize,
}

/// `x` rounded to f32 toward zero (the `TRUNCATE_B_TO_F32` mutant). `r` is the
/// nearest-rounded `x as f32` (finite).
fn toward_zero(x: f64, r: f32) -> f32 {
    if f64::from(r).abs() > x.abs() {
        f32::from_bits(r.to_bits() - 1)
    } else {
        r
    }
}

/// Convert `rows` to f32 (nearest, or toward zero under the mutation seam) and
/// copy them into `dst[base..]`, in bounded host steps. A finite value beyond
/// `f32::MAX` is a typed `F32Range` refusal naming the element.
fn copy_as_f32(
    dev: &Device,
    dst: &mut DeviceMatrix<f32>,
    base: usize,
    rows: &[f64],
) -> Result<(), GpuError> {
    if inject_f32_range() {
        return Err(GpuError::F32Range("injected (test seam)".into()));
    }
    let truncate = truncate_b_to_f32();
    let mut off = 0usize;
    while off < rows.len() {
        let end = (off + CONVERT_ELEMS).min(rows.len());
        let chunk = &rows[off..end];
        let mut out = round_to_f32(&format!("DF-J raw B[{}..]", base + off), chunk)?;
        if truncate {
            for (o, &x) in out.iter_mut().zip(chunk) {
                *o = toward_zero(x, *o);
            }
        }
        let mut view = dst.buf_mut().slice_mut(base + off..base + end);
        dev.stream.memcpy_htod(&out, &mut view).map_err(|e| {
            GpuError::Cuda(format!("H2D DF-J raw B (f32) at {}: {e:?}", base + off))
        })?;
        off = end;
    }
    Ok(())
}

/// Copy `rows` into the f64 tensor at `dst[base..]`, rounding each value through
/// f32 first under the `ROUND_B_TO_F32` twin seam (otherwise the host slice is sent
/// as it is).
fn copy_as_f64(
    dev: &Device,
    dst: &mut DeviceMatrix<f64>,
    base: usize,
    rows: &[f64],
) -> Result<(), GpuError> {
    let mut copy = |at: usize, src: &[f64]| {
        let mut view = dst.buf_mut().slice_mut(at..at + src.len());
        dev.stream
            .memcpy_htod(src, &mut view)
            .map_err(|e| GpuError::Cuda(format!("H2D DF-J raw B at {at}: {e:?}")))
    };
    if !round_b_to_f32() {
        return copy(base, rows);
    }
    let mut off = 0usize;
    while off < rows.len() {
        let end = (off + CONVERT_ELEMS).min(rows.len());
        let rounded: Vec<f64> = rows[off..end].iter().map(|&x| x as f32 as f64).collect();
        copy(base + off, &rounded)?;
        off = end;
    }
    Ok(())
}

impl DeviceDfJ {
    /// Reserve the f64 device state for a `(naux × pair)` packed tensor. The small
    /// vectors are reserved FIRST so a refusal of the large `Bp` leaks nothing;
    /// `Bp` is charged under "DF-J raw B (packed)" before any byte moves.
    pub fn reserve(
        dev: &Arc<Device>,
        pool: &DevicePool,
        naux: usize,
        pair: usize,
    ) -> Result<Self, GpuError> {
        Self::reserve_with_precision(dev, pool, naux, pair, DfjPrecision::F64)
    }

    /// [`reserve`](Self::reserve) with the precision of the resident `Bp`. `Mixed`
    /// probes the GEMV kernel first (`GpuError::Kernel` moves nothing) and charges
    /// `4·naux·pair` under "DF-J raw B (packed, f32)".
    pub fn reserve_with_precision(
        dev: &Arc<Device>,
        pool: &DevicePool,
        naux: usize,
        pair: usize,
        precision: DfjPrecision,
    ) -> Result<Self, GpuError> {
        resident_bytes_with(naux, pair, precision)?;
        if precision == DfjPrecision::Mixed {
            dev.gemv_f32_f64()?;
        }
        let (rows_per_call, _) = row_plan(naux, pair);
        let bufs = if naux == 0 || pair == 0 {
            None
        } else {
            let z = |label: &str, len: usize| DeviceMatrix::<f64>::zeros(dev, pool, label, len, 1);
            let w = z("DF-J weights w", pair)?;
            let d = z("DF-J d_P", naux)?;
            let c = z("DF-J c", naux)?;
            let j = z("DF-J packed J", pair)?;
            let b = match precision {
                DfjPrecision::F64 => ResidentB::F64(DeviceMatrix::<f64>::zeros(
                    dev,
                    pool,
                    "DF-J raw B (packed)",
                    naux,
                    pair,
                )?),
                DfjPrecision::Mixed => ResidentB::F32(DeviceMatrix::<f32>::zeros(
                    dev,
                    pool,
                    "DF-J raw B (packed, f32)",
                    naux,
                    pair,
                )?),
            };
            Some(Bufs { b, w, d, c, j })
        };
        Ok(Self {
            dev: Arc::clone(dev),
            naux,
            pair,
            rows_per_call,
            bufs,
            filled: 0,
        })
    }

    /// `true` when the resident `Bp` is f32.
    pub fn is_mixed(&self) -> bool {
        matches!(
            self.bufs,
            Some(Bufs {
                b: ResidentB::F32(_),
                ..
            })
        )
    }

    /// Bytes per resident `Bp` element (8 or 4).
    pub fn b_elem_bytes(&self) -> usize {
        if self.is_mixed() {
            4
        } else {
            8
        }
    }

    /// Copy `rows` (`k·pair` doubles, `k` whole aux rows) into aux rows
    /// `p0..p0+k`. Rows must arrive in ascending order without gaps. A mixed
    /// tensor rounds to f32 on the way (`GpuError::F32Range` for a value beyond
    /// `f32::MAX`; the caller then drops this state and uploads f64).
    pub fn write_rows(&mut self, p0: usize, rows: &[f64]) -> Result<(), GpuError> {
        let lay = GpuError::Layout;
        if self.bufs.is_none() {
            // Nothing to hold (naux = 0 or pair = 0): only an empty write is valid.
            return if rows.is_empty() {
                self.filled = self.naux;
                Ok(())
            } else {
                Err(lay("DF-J upload into an empty tensor".into()))
            };
        }
        if p0 != self.filled {
            return Err(lay(format!(
                "DF-J upload rows must be contiguous: expected row {}, got {p0}",
                self.filled
            )));
        }
        let Some(bufs) = self.bufs.as_mut() else {
            return Ok(());
        };
        if !rows.len().is_multiple_of(self.pair) {
            return Err(lay(format!(
                "DF-J upload of {} doubles is not whole rows of {}",
                rows.len(),
                self.pair
            )));
        }
        let k = rows.len() / self.pair;
        if p0 + k > self.naux {
            return Err(lay(format!(
                "DF-J upload rows {p0}..{} exceed naux {}",
                p0 + k,
                self.naux
            )));
        }
        let base = p0 * self.pair;
        match &mut bufs.b {
            ResidentB::F64(b) => copy_as_f64(&self.dev, b, base, rows)?,
            ResidentB::F32(b) => copy_as_f32(&self.dev, b, base, rows)?,
        }
        self.filled += k;
        Ok(())
    }

    /// Seal the upload: every row must have arrived. Counts one resident upload.
    pub fn finish_upload(&mut self) -> Result<(), GpuError> {
        if self.filled != self.naux {
            return Err(GpuError::Layout(format!(
                "DF-J upload sealed with {} of {} aux rows",
                self.filled, self.naux
            )));
        }
        self.sync("sync after DF-J upload")?;
        if self.bufs.is_some() {
            stats::note_resident_upload(self.b_elem_bytes() * self.naux * self.pair);
        }
        Ok(())
    }

    /// `reserve` + one `write_rows` + `finish_upload` for a host packed tensor.
    pub fn upload_packed(
        dev: &Arc<Device>,
        pool: &DevicePool,
        packed: &ArrayView2<f64>,
    ) -> Result<Self, GpuError> {
        Self::upload_packed_with_precision(dev, pool, packed, DfjPrecision::F64)
    }

    /// [`upload_packed`](Self::upload_packed) with the precision of the resident
    /// `Bp` (no fallback here: `F32Range` and `Kernel` are returned to the caller).
    pub fn upload_packed_with_precision(
        dev: &Arc<Device>,
        pool: &DevicePool,
        packed: &ArrayView2<f64>,
        precision: DfjPrecision,
    ) -> Result<Self, GpuError> {
        let (naux, pair) = packed.dim();
        let mut me = Self::reserve_with_precision(dev, pool, naux, pair, precision)?;
        let std = packed.as_standard_layout();
        let flat = std
            .as_slice()
            .ok_or_else(|| GpuError::Layout("packed tensor not contiguous".into()))?;
        me.write_rows(0, flat)?;
        me.finish_upload()?;
        Ok(me)
    }

    pub fn naux(&self) -> usize {
        self.naux
    }

    pub fn pair(&self) -> usize {
        self.pair
    }

    fn sync(&self, what: &str) -> Result<(), GpuError> {
        self.dev
            .stream
            .synchronize()
            .map_err(|e| GpuError::Cuda(format!("{what}: {e:?}")))
    }

    /// Pass 1: `d_P = Bp·w` for the packed weights `w` (`pair` doubles).
    pub fn d_p(&mut self, w: &[f64]) -> Result<Vec<f64>, GpuError> {
        if w.len() != self.pair {
            return Err(GpuError::Layout(format!(
                "DF-J weights hold {} doubles, expected {}",
                w.len(),
                self.pair
            )));
        }
        let Some(bufs) = self.bufs.as_mut() else {
            return Ok(vec![0.0; self.naux]);
        };
        let cuda = |what: &str, e: &dyn std::fmt::Debug| GpuError::Cuda(format!("{what}: {e:?}"));
        self.dev
            .stream
            .memcpy_htod(w, bufs.w.buf_mut())
            .map_err(|e| cuda("H2D DF-J w", &e))?;
        let (pair, rows) = (self.pair, self.rows_per_call);
        match &bufs.b {
            ResidentB::F32(b) => {
                let spec = GemvSpec {
                    rows: self.naux,
                    cols: pair,
                    transpose: false,
                    accumulate: false,
                };
                let x = bufs.w.buf().slice(..);
                let mut y = bufs.d.buf_mut().slice_mut(..);
                gemv_f32mat_f64_dev(&self.dev, spec, &b.buf().slice(..), &x, &mut y)?;
            }
            ResidentB::F64(b) => {
                let mut r0 = 0usize;
                while r0 < self.naux {
                    let r1 = (r0 + rows).min(self.naux);
                    let a = b.buf().slice(r0 * pair..r1 * pair);
                    let x = bufs.w.buf().slice(..);
                    let mut y = bufs.d.buf_mut().slice_mut(r0..r1);
                    let spec = GemvSpec {
                        rows: r1 - r0,
                        cols: pair,
                        transpose: false,
                        accumulate: false,
                    };
                    gemv_f64_dev(&self.dev, spec, &a, &x, &mut y)?;
                    r0 = r1;
                }
            }
        }
        let mut out = vec![0.0f64; self.naux];
        self.dev
            .stream
            .memcpy_dtoh(bufs.d.buf(), &mut out)
            .map_err(|e| cuda("D2H DF-J d_P", &e))?;
        self.sync("sync after DF-J pass 1")?;
        Ok(out)
    }

    /// Pass 2: `J_packed = Bpᵀ·c` for `c` (`naux` doubles).
    pub fn j_packed(&mut self, c: &[f64]) -> Result<Vec<f64>, GpuError> {
        if c.len() != self.naux {
            return Err(GpuError::Layout(format!(
                "DF-J c holds {} doubles, expected {}",
                c.len(),
                self.naux
            )));
        }
        let Some(bufs) = self.bufs.as_mut() else {
            return Ok(vec![0.0; self.pair]);
        };
        let cuda = |what: &str, e: &dyn std::fmt::Debug| GpuError::Cuda(format!("{what}: {e:?}"));
        self.dev
            .stream
            .memcpy_htod(c, bufs.c.buf_mut())
            .map_err(|e| cuda("H2D DF-J c", &e))?;
        let (pair, rows) = (self.pair, self.rows_per_call);
        match &bufs.b {
            ResidentB::F32(b) => {
                let spec = GemvSpec {
                    rows: self.naux,
                    cols: pair,
                    transpose: true,
                    accumulate: false,
                };
                let x = bufs.c.buf().slice(..);
                let mut y = bufs.j.buf_mut().slice_mut(..);
                gemv_f32mat_f64_dev(&self.dev, spec, &b.buf().slice(..), &x, &mut y)?;
            }
            ResidentB::F64(b) => {
                let mut r0 = 0usize;
                while r0 < self.naux {
                    let r1 = (r0 + rows).min(self.naux);
                    let a = b.buf().slice(r0 * pair..r1 * pair);
                    let x = bufs.c.buf().slice(r0..r1);
                    let mut y = bufs.j.buf_mut().slice_mut(..);
                    let spec = GemvSpec {
                        rows: r1 - r0,
                        cols: pair,
                        transpose: true,
                        accumulate: r0 != 0,
                    };
                    gemv_f64_dev(&self.dev, spec, &a, &x, &mut y)?;
                    r0 = r1;
                }
            }
        }
        let mut out = vec![0.0f64; pair];
        self.dev
            .stream
            .memcpy_dtoh(bufs.j.buf(), &mut out)
            .map_err(|e| cuda("D2H DF-J J", &e))?;
        self.sync("sync after DF-J pass 2")?;
        Ok(out)
    }
}

/// Per-`DfJ` device state. `Declined` is sticky (and has released the device memory).
#[derive(Default)]
pub enum DeviceSlot {
    #[default]
    Untried,
    Ready(Box<DeviceDfJ>),
    Declined,
}

/// Why this source can never use the device path, or `None`.
fn ineligible(src: &ThreeIndexSource, ctx: Option<&ParallelContext>) -> Option<&'static str> {
    if crate::df_k_gpu::real_mpi_world(ctx) || src.band_naux() != src.naux() {
        return Some(
            "MPI ranks would share one device ordinal (a device per rank is not available yet)",
        );
    }
    None
}

/// J is built before K every iteration, so J's upload would take the card first.
/// When the same SCF's DF-K (`k_reserve` device bytes) would no longer fit next to
/// J, J stays on the CPU: the dressed K tensor is the larger device win.
fn yields_to_k(
    pool: &DevicePool,
    src: &ThreeIndexSource,
    k_reserve: usize,
    precision: DfjPrecision,
) -> Option<String> {
    if k_reserve == 0 {
        return None;
    }
    let pair = pair_len(src.nao()).ok()?;
    let need = resident_bytes_with(src.band_naux(), pair, precision).ok()?;
    (pool.available_bytes() < need.saturating_add(k_reserve)).then(|| {
        format!(
            "the device pool cannot hold both this tensor ({need} B) and the DF-K tensor ({k_reserve} B); DF-K gets the device"
        )
    })
}

/// The first line of an error's Display: the pool's refusal embeds a multi-line
/// occupancy report, which stays in the typed error and out of the notices.
fn one_line(e: &GpuError) -> String {
    e.to_string().lines().next().unwrap_or_default().to_string()
}

fn decline(slot: &mut DeviceSlot, reason: &str) {
    eprintln!("[ferric] gpu: RI-J stays on the CPU ({reason})");
    stats::note_dfj_declined(reason);
    *slot = DeviceSlot::Declined;
}

fn count_cpu_fallback(e: &GpuError) {
    let reason = match e {
        GpuError::PoolFull { .. } => CpuReason::PoolFull,
        GpuError::Layout(_) => CpuReason::Layout,
        GpuError::F32Range(_) => CpuReason::F32Range,
        _ => CpuReason::CudaError,
    };
    stats::note_cpu_detail(reason, &format!("(RI-J: {})", one_line(e)));
}

/// Pack rows of an unpacked `(rows, n·n)` chunk into `(rows, pair)`: the lower
/// triangle `(μ, ν ≤ μ)` row-major, the packed tier's own layout.
fn pack_rows(full: &ArrayView2<f64>, n: usize, out: &mut Vec<f64>) {
    out.clear();
    for row in full.rows() {
        for mu in 0..n {
            for nu in 0..=mu {
                out.push(row[mu * n + nu]);
            }
        }
    }
}

/// Move the whole band onto the device (the pool is charged by `reserve` first).
fn fill_from_source(d: &mut DeviceDfJ, src: &mut ThreeIndexSource) -> Result<(), GpuError> {
    let n = src.nao();
    if let Some(p) = src.packed_flat() {
        let std = p.as_standard_layout();
        let flat = std
            .as_slice()
            .ok_or_else(|| GpuError::Layout("packed tensor not contiguous".into()))?;
        d.write_rows(0, flat)?;
    } else if let Some(full) = src.incore_flat() {
        let pair = d.pair().max(1);
        let rows = (STAGE_BYTES / (8 * pair)).max(1);
        let mut stage = Vec::new();
        let mut p0 = 0usize;
        while p0 < full.nrows() {
            let p1 = (p0 + rows).min(full.nrows());
            pack_rows(&full.slice(ndarray::s![p0..p1, ..]), n, &mut stage);
            d.write_rows(p0, &stage)?;
            p0 = p1;
        }
    } else {
        let mut failure: Option<GpuError> = None;
        let streamed = src.for_each_packed_block(1, |blk| {
            let std = blk.data.as_standard_layout();
            let flat = std
                .as_slice()
                .ok_or_else(|| FerricError::General("packed block not contiguous".into()))?;
            d.write_rows(blk.p0, flat).map_err(|e| {
                let msg = e.to_string();
                failure = Some(e);
                FerricError::General(msg)
            })
        });
        match (failure, streamed) {
            (Some(e), _) => return Err(e),
            (None, Err(e)) => return Err(GpuError::Layout(format!("streaming the tensor: {e}"))),
            (None, Ok(())) => {}
        }
    }
    d.finish_upload()
}

fn upload_band(
    dev: &Arc<Device>,
    pool: &DevicePool,
    src: &mut ThreeIndexSource,
    precision: DfjPrecision,
) -> Result<DeviceDfJ, GpuError> {
    let pair = pair_len(src.nao())?;
    let mut d = DeviceDfJ::reserve_with_precision(dev, pool, src.naux(), pair, precision)?;
    fill_from_source(&mut d, src)?;
    Ok(d)
}

/// The mixed upload, or, when `Bp` has a value beyond `f32::MAX` (`F32Range`) or the
/// GEMV kernel will not load (`Kernel`), the f64 upload instead (the failed mixed
/// state is dropped first, which releases its pool charge), counted once.
fn upload_band_resolved(
    dev: &Arc<Device>,
    pool: &DevicePool,
    src: &mut ThreeIndexSource,
    precision: DfjPrecision,
) -> Result<DeviceDfJ, GpuError> {
    match upload_band(dev, pool, src, precision) {
        Err(e @ (GpuError::F32Range(_) | GpuError::Kernel(_)))
            if precision == DfjPrecision::Mixed =>
        {
            if ferric_core::gpu::config::gpu_trace() {
                eprintln!("[gpu] RI-J mixed path unavailable ({e}); running f64 on the device");
            }
            stats::note_mixed_fallback();
            upload_band(dev, pool, src, DfjPrecision::F64)
        }
        other => other,
    }
}

/// One device J build: `Ok(true)` means `j` holds the full J (no cross-rank
/// reduction is needed: the device path never runs under MPI); `Ok(false)` means
/// run the CPU path (`j` untouched by this call). `solve` is `DfJ`'s own metric
/// solve; its error is the caller's error, as on the CPU path. Never touches CUDA
/// when the mode is off or the seam forces the host.
pub fn try_build(
    slot: &mut DeviceSlot,
    src: &mut ThreeIndexSource,
    (ctx, k_reserve): (Option<&ParallelContext>, usize),
    d: &Array2<f64>,
    j: &mut Array2<f64>,
    solve: impl FnOnce(&Array1<f64>) -> Result<Array1<f64>, FerricError>,
) -> Result<bool, FerricError> {
    if force_host() || matches!(slot, DeviceSlot::Declined) {
        return Ok(false);
    }
    let settings = ferric_core::gpu::settings();
    if settings.mode == GpuMode::Off {
        return Ok(false);
    }
    let GpuStatus::Ready(info) = ferric_core::gpu::status() else {
        return Ok(false);
    };
    if rayon::current_thread_index().is_some() {
        stats::note_cpu(CpuReason::InsideRayonWorker); // transient: the slot is untouched
        return Ok(false);
    }
    let n = src.nao();
    if src.naux() == 0 || n == 0 || d.dim() != (n, n) || j.dim() != (n, n) {
        return Ok(false); // empty or mismatched: the CPU path owns the semantics
    }
    if let Some(why) = ineligible(src, ctx) {
        decline(slot, why);
        return Ok(false);
    }
    let Some(pool) = ferric_core::gpu::pool() else {
        return Ok(false);
    };
    if matches!(slot, DeviceSlot::Untried) {
        let precision = dfj_precision();
        if let Some(why) = yields_to_k(&pool, src, k_reserve, precision) {
            decline(slot, &why);
            return Ok(false);
        }
        match device(info.ordinal).and_then(|dev| upload_band_resolved(&dev, &pool, src, precision))
        {
            Ok(dd) => {
                eprintln!(
                    "[ferric] gpu: RI-J resident on device {} ({:.3} GB packed B{}, {} aux rows, nbf {})",
                    info.ordinal,
                    (dd.b_elem_bytes() * dd.naux() * dd.pair()) as f64 / 1e9,
                    if dd.is_mixed() { ", f32" } else { "" },
                    dd.naux(),
                    n
                );
                *slot = DeviceSlot::Ready(Box::new(dd));
            }
            Err(e) => {
                count_cpu_fallback(&e);
                decline(slot, &one_line(&e));
                return Ok(false);
            }
        }
    }
    let DeviceSlot::Ready(dd) = slot else {
        return Ok(false);
    };
    let w = crate::df_j::packed_density_weights(d, n);
    let w_slice = w.as_slice().unwrap_or(&[]);
    let d_p = match dd.d_p(w_slice) {
        Ok(v) => Array1::from(v),
        Err(e) => {
            count_cpu_fallback(&e);
            decline(slot, &one_line(&e));
            return Ok(false);
        }
    };
    let c_p = solve(&d_p)?;
    let outcome = if force_build_failure() {
        Err(GpuError::Cuda("injected RI-J build failure".into()))
    } else {
        match c_p.as_slice() {
            Some(c) => dd.j_packed(c),
            None => Err(GpuError::Layout("c not contiguous".into())),
        }
    };
    match outcome {
        Ok(packed) => {
            crate::df_j::unpack_symmetric(&packed, n, j);
            let (naux, pair) = (dd.naux(), dd.pair());
            stats::note_dfj_build(8 * (pair + naux), 8 * (naux + pair));
            Ok(true)
        }
        Err(e) => {
            count_cpu_fallback(&e);
            decline(slot, &one_line(&e));
            Ok(false)
        }
    }
}
