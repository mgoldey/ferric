//! Device offload of the screened XC grid's per-batch GEMMs (feature `gpu`).
//!
//! Opt-in: the grid uses the device only under `[gpu] mode = "on"`; `auto`
//! keeps it on the CPU (see [`xc_grid_routes_to_device`] for the measurement).
//!
//! # What runs where
//!
//! Everything that is O(nact·nb) per batch stays on the host, exactly as on
//! the CPU path: the sub-density gather, the ρ/∇ρ/τ point reductions, the
//! global libxc call, the pass-2 factor matrix `A`, and the fixed-group
//! `V_b + V_bᵀ` scatter. The two O(nact²·nb) products per batch — the ones
//! that dominate an XC build — run on cuBLAS:
//!
//! * pass 1: `Φ = D_sub · X` (`X = [χ | ∂χ]`, the first `ncols` columns of the
//!   batch's AO block); for a factored density `Ψ = C_subᵀ·X`, `Φ = C_sub·Ψ`,
//!   scaled on the host after the download (the scale is 1 or 2: exact);
//! * pass 2: `V_b = A · Xᵀ`.
//!
//! Every batch's compact AO block is uploaded ONCE per [`ScreenedGrid`] (the
//! first integration that asks for the device), concatenated in batch order,
//! under the pool label [`AO_LABEL`]. Per pass only the per-batch operands move:
//! `D_sub`/`C_sub` or `A` up, `Φ` or `V_b` down. Batches are processed in
//! WAVES — maximal contiguous batch ranges whose operands fit three fixed
//! staging buffers — so one wave costs one upload, a run of stream-ordered
//! GEMM launches, one download and one synchronisation. The host copies of
//! one wave's operands and results (at most two staging buffers' worth, 64 MiB
//! each by default) are transient and not charged to the host pool.
//!
//! # Numerics and determinism
//!
//! The device products are plain f64 `cublasDgemm`, one call per product with
//! no k blocking — except the pass-2 product of a build the precision ladder
//! marks mixed (see below). They are NOT bit-identical to the host GEMMs (cuBLAS orders its
//! sums its own way); each product obeys the classical bound
//! `|ĉ − c| ≤ γ_k (|A||B|)` (Higham, ASNA §3.5) with `k = nact` (pass 1 dense,
//! pass 1 `Ψ`), `k = nocc` (pass 1 `Φ` from `Ψ`) or `k = ncols` (pass 2), the
//! same order of error as the host dgemm. Everything downstream is the host
//! code: the pass-2 contributions are scattered in the same fixed groups, each
//! group in ascending batch order, groups summed in ascending order. The result
//! is therefore deterministic run to run on one device and independent of the
//! rayon thread count.
//!
//! # Mixed precision (`xc-vxc`)
//!
//! With `[gpu] precision = "mixed"` and `xc-vxc` in `mixed_kernels`, the upload
//! also keeps the AO blocks rounded to f32 (round to nearest, ties to even;
//! half the bytes again), and a pass 2 that `KsXc`'s precision ladder marks
//! mixed (LDA/GGA only, density still moving; see `ks::MIXED_VXC_LADDER_THRESH`)
//! forms each `V_b = A·Xᵀ` from f32 `A` and `X` with the k-panelled SGEMM and
//! f64 flush of `ferric_core::gpu::mixed` (panels of 64). Per element
//! `|ΔV_ij| ≤ (2u32 + γ_64(u32) + γ_⌈k/64⌉(u64))·(|A||X|ᵀ)_ij`, `k = ncols`
//! (the host twin's bound, pinned by `xc_batch`'s unit test). Pass 1 (ρ, ∇ρ),
//! the functional and E_xc stay f64. A wave whose panel sums could leave the
//! f32 range (`k_panel·max|A|·max|X| ≥ f32::MAX/2`, or a NaN) runs f64, and a
//! missing flush kernel, an AO value beyond the f32 range or a pool too small
//! for the f32 copies leave the whole grid f64 on the device; each is counted
//! (`mixed_fallback_f64`) and the last three announced.
//!
//! # Fallback (never silent)
//!
//! `[gpu] mode` other than `on`, no ready device, no pool, or a call from
//! inside a rayon worker: the integration runs on the CPU and touches no CUDA. A pool refusal
//! or a CUDA error while uploading, or any device error during a pass, declines
//! the grid for its lifetime (one `[ferric] gpu:` notice, counted in
//! `stats::xc_declined`) and the pass that failed is recomputed on the CPU.
use std::borrow::Cow;
use std::sync::Arc;

use ndarray::{s, Array2};
use rayon::prelude::*;

use ferric_core::gpu::device::{device, Device, GpuError};
use ferric_core::gpu::gemm::{dev_left, dev_right, dev_right_padded, gemm_f64_dev_into};
use ferric_core::gpu::mixed::{effective_k_panel, gemm_f32_f64acc_dev_into, round_to_f32};
use ferric_core::gpu::resident::DeviceMatrix;
use ferric_core::gpu::stats::{self, CpuReason};
use ferric_core::gpu::{GpuMode, GpuStatus, MixedKernel};

use super::{
    batch_density_from_phi, fill_amat, group_ranges, map_indexed, n_groups, scatter_sym,
    BatchDensity, BatchFactors, DensityInput, DevSession, GridPoint, GtoEvalError, ScreenedGrid,
    XcBatch,
};

/// Pool label of the device copy of the compact AO blocks.
pub const AO_LABEL: &str = "KS-DFT grid AO blocks (device)";

/// Default elements per staging buffer (64 MiB of f64). A buffer grows past
/// this only when a single batch needs more; it shrinks to what the whole grid
/// needs when that is less.
const STAGE_ELEMS: usize = 1 << 23;

/// Host bytes evaluated (recompute mode) and sent per step of the AO upload.
const UPLOAD_STAGE_BYTES: usize = 64 << 20;

/// Test seam (`test-seams` feature): every integration runs on the CPU, as
/// with `[gpu] mode = off`, without touching the device slot.
#[cfg(feature = "test-seams")]
#[doc(hidden)]
pub static FORCE_HOST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Test seam (`test-seams` feature): the CPU pass 2 forms each `V_b` with the
/// host twin of the mixed kernel, so its error map is measurable without a device.
#[cfg(feature = "test-seams")]
#[doc(hidden)]
pub static MIXED_VXC_HOST_TWIN: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(feature = "test-seams")]
pub(super) fn mixed_vxc_host_twin() -> bool {
    MIXED_VXC_HOST_TWIN.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether the KS XC grid may use the device under `mode`: only an explicit
/// `mode = "on"` (or a preset that sets it). Under `auto` it stays on the CPU
/// because the device path was measured SLOWER than the CPU on a GTX 1080
/// (Pascal, sm_61; PBE/cc-pVDZ SCF wall time, benzene 2.0 s CPU vs 3.6 s f64
/// device, n-octane 8.5-8.9 s vs 18-23 s; 2.2-3x the host time per XC build at
/// RAYON_NUM_THREADS = 3), so `auto` would silently slow every DFT run on a
/// GPU box. Nothing faster has been measured yet.
pub fn xc_grid_routes_to_device(mode: GpuMode) -> bool {
    mode == GpuMode::On
}

/// See `xc_batch::mixed_vxc_requested`.
pub(super) fn mixed_vxc_requested() -> bool {
    #[cfg(feature = "test-seams")]
    if mixed_vxc_host_twin() {
        return true;
    }
    !force_host()
        && xc_grid_routes_to_device(ferric_core::gpu::settings().mode)
        && ferric_core::gpu::settings().mixed_allows(MixedKernel::XcVxc)
        && matches!(ferric_core::gpu::status(), GpuStatus::Ready(_))
}

fn force_host() -> bool {
    #[cfg(feature = "test-seams")]
    {
        FORCE_HOST.load(std::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(feature = "test-seams"))]
    {
        false
    }
}

/// Per-grid device state. `Declined` is sticky (and has released the device memory).
#[derive(Default)]
pub(crate) enum DeviceSlot {
    #[default]
    Untried,
    Ready(Box<DeviceXc>),
    Declined,
}

impl std::fmt::Debug for DeviceSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            DeviceSlot::Untried => "Untried",
            DeviceSlot::Ready(_) => "Ready",
            DeviceSlot::Declined => "Declined",
        })
    }
}

/// The resident AO blocks plus the three staging buffers.
pub(crate) struct DeviceXc {
    dev: Arc<Device>,
    /// Every batch's `(nact, 4nb)` row-major block, concatenated in batch order.
    ao: DeviceMatrix<f64>,
    /// Element offset of batch `bi`'s block in `ao`.
    ao_off: Vec<usize>,
    /// Per-batch left operands (`D_sub`, `C_sub` or `A`).
    stage_in: DeviceMatrix<f64>,
    /// Per-batch `Ψ = C_subᵀ·X` (factored densities only).
    stage_mid: DeviceMatrix<f64>,
    /// Per-batch results (`Φ` or `V_b`).
    stage_out: DeviceMatrix<f64>,
    /// The f32 operands of the mixed `xc-vxc` kernel; `None` when mixed
    /// precision was not requested or could not be set up (f64 only).
    mixed: Option<MixedState>,
}

/// Device state of the mixed `xc-vxc` kernel.
struct MixedState {
    /// `ao` rounded to f32 (round to nearest, ties to even), same layout.
    ao32: DeviceMatrix<f32>,
    /// Per-batch `A` rounded to f32 (the pass-2 left operands).
    stage_in32: DeviceMatrix<f32>,
    /// f32 panel scratch, `max nact²` elements: one batch at a time (the
    /// launches are stream-ordered, so a batch's panels and flushes finish
    /// before the next batch's first panel overwrites it).
    c32: DeviceMatrix<f32>,
    /// `max |χ|, |∇χ|` over every AO block (the panel-overflow guard).
    max_abs_x: f64,
}

/// Why the mixed state could not be built: a reason to run f64 on the device
/// (`Soft`), or a device failure that declines the grid (`Hard`).
enum MixedSetup {
    Soft(GpuError),
    Hard(GpuError),
}

/// True when every panel sum of a mixed `A·Xᵀ` stays inside the f32 range: a
/// panel of `k_panel` terms is bounded by `k_panel·max|A|·max|X|`, kept below
/// half of `f32::MAX`. A NaN maximum is refused too (the f64 path then
/// propagates it exactly as the CPU does).
pub(crate) fn mixed_panel_fits(k_panel: usize, max_a: f64, max_x: f64) -> bool {
    let worst = k_panel.max(1) as f64 * max_a * max_x;
    worst < 0.5 * f64::from(f32::MAX)
}

/// Why a device pass stopped: a host AO evaluation error (the caller's error,
/// as on the CPU path) or a device error (decline and recompute on the CPU).
enum Fail {
    Eval(GtoEvalError),
    Gpu(GpuError),
}

impl From<GpuError> for Fail {
    fn from(e: GpuError) -> Self {
        Fail::Gpu(e)
    }
}

impl From<GtoEvalError> for Fail {
    fn from(e: GtoEvalError) -> Self {
        Fail::Eval(e)
    }
}

/// The first line of an error's Display: the pool's refusal embeds a
/// multi-line occupancy report, which stays out of the notices.
fn one_line(e: &GpuError) -> String {
    e.to_string().lines().next().unwrap_or_default().to_string()
}

fn decline(slot: &mut DeviceSlot, reason: &str) {
    eprintln!("[ferric] gpu: KS-DFT XC grid stays on the CPU ({reason})");
    stats::note_xc_declined(reason);
    *slot = DeviceSlot::Declined;
}

fn count_cpu_fallback(e: &GpuError) {
    let reason = match e {
        GpuError::PoolFull { .. } => CpuReason::PoolFull,
        GpuError::Layout(_) => CpuReason::Layout,
        GpuError::F32Range(_) => CpuReason::F32Range,
        _ => CpuReason::CudaError,
    };
    stats::note_cpu_detail(reason, &format!("(KS XC: {})", one_line(e)));
}

fn cuda(what: &str, e: impl std::fmt::Debug) -> GpuError {
    GpuError::Cuda(format!("{what}: {e:?}"))
}

/// A zero-byte host twin of a row-major `(rows, cols)` matrix: the operand
/// constructors read only its shape and strides.
fn twin(rows: usize, cols: usize) -> Array2<()> {
    Array2::from_elem((rows, cols), ())
}

/// Open the device session for one integration (see `ScreenedGrid::device_session`).
pub(super) fn session<'a>(sg: &'a ScreenedGrid, grid: &[GridPoint]) -> DevSession<'a> {
    if force_host() || !xc_grid_routes_to_device(ferric_core::gpu::settings().mode) {
        return None;
    }
    let GpuStatus::Ready(info) = ferric_core::gpu::status() else {
        return None;
    };
    if rayon::current_thread_index().is_some() {
        note_rayon_worker(sg);
        return None;
    }
    let mut slot = sg.device.lock().unwrap_or_else(|e| e.into_inner());
    if matches!(*slot, DeviceSlot::Declined) {
        return None;
    }
    let pool = ferric_core::gpu::pool()?;
    if matches!(*slot, DeviceSlot::Untried) {
        match device(info.ordinal).and_then(|dev| DeviceXc::upload(&dev, &pool, sg, grid)) {
            Ok(dx) => {
                eprintln!(
                    "[ferric] gpu: KS-DFT XC grid on device {} ({:.3} GB AO blocks, {} batches, nbf {})",
                    info.ordinal,
                    dx.ao.bytes() as f64 / 1e9,
                    sg.batches.len(),
                    sg.nbf
                );
                *slot = DeviceSlot::Ready(Box::new(dx));
            }
            Err(e) => {
                count_cpu_fallback(&e);
                decline(&mut slot, &one_line(&e));
                return None;
            }
        }
    }
    Some(slot)
}

/// A call from inside a rayon worker runs on the CPU (transient: the slot is
/// untouched). Counted as `InsideRayonWorker` unless the grid is already
/// declined, which is known without blocking: `try_lock` never waits, and a
/// slot that is busy or poisoned counts as not declined.
fn note_rayon_worker(sg: &ScreenedGrid) {
    let declined = matches!(sg.device.try_lock().as_deref(), Ok(DeviceSlot::Declined));
    if !declined {
        stats::note_cpu(CpuReason::InsideRayonWorker);
    }
}

/// True when `sg`'s AO blocks are on the device and no failure declined it.
pub(super) fn is_ready(sg: &ScreenedGrid) -> bool {
    let slot = sg.device.lock().unwrap_or_else(|e| e.into_inner());
    matches!(*slot, DeviceSlot::Ready(_))
}

/// Run `pass` on the session's device state. `Ok(None)` = run the CPU path:
/// no session, or a device failure (the grid is then declined for good and the
/// session closed, so later passes of this integration go straight to the CPU).
fn with_device<T>(
    dev: &mut DevSession<'_>,
    pass: impl FnOnce(&mut DeviceXc) -> Result<T, Fail>,
) -> Result<Option<T>, GtoEvalError> {
    let Some(slot) = dev.as_mut() else {
        return Ok(None);
    };
    let outcome = match &mut **slot {
        DeviceSlot::Ready(dx) => pass(dx),
        _ => return Ok(None),
    };
    match outcome {
        Ok(v) => Ok(Some(v)),
        Err(Fail::Eval(e)) => Err(e),
        Err(Fail::Gpu(e)) => {
            count_cpu_fallback(&e);
            decline(slot, &one_line(&e));
            *dev = None;
            Ok(None)
        }
    }
}

/// Device pass 1 (see `ScreenedGrid::density_pass`); `Ok(None)` = run the CPU path.
pub(super) fn density_pass(
    sg: &ScreenedGrid,
    grid: &[GridPoint],
    ds: &[DensityInput],
    need_tau: bool,
    serial: bool,
    dev: &mut DevSession<'_>,
) -> Result<Option<Vec<Vec<BatchDensity>>>, GtoEvalError> {
    with_device(dev, |dx| dx.density(sg, grid, ds, need_tau, serial))
}

/// Device pass 2 (see `ScreenedGrid::potential_pass`); `Ok(None)` = run the CPU path.
pub(super) fn potential_pass<F>(
    sg: &ScreenedGrid,
    grid: &[GridPoint],
    nmats: usize,
    factors: &F,
    flags: (bool, bool),
    exec: super::Exec,
    dev: &mut DevSession<'_>,
) -> Result<Option<Vec<Array2<f64>>>, GtoEvalError>
where
    F: Fn(&XcBatch, usize) -> BatchFactors + Sync,
{
    with_device(dev, |dx| {
        dx.potential(
            sg,
            grid,
            nmats,
            factors,
            flags,
            (exec.serial, exec.mixed_vxc),
        )
    })
}

/// Elements one staging buffer is given: the largest per-batch need of any
/// pass (≤ 2 densities or matrices, `nocc ≤ nbf`, `ncols ≤ 4nb`), at least, and
/// otherwise [`STAGE_ELEMS`] capped by what the whole grid would need at once.
pub(crate) fn stage_elems(nbf: usize, batches: &[(usize, usize)]) -> Result<usize, GpuError> {
    let overflow = || GpuError::Layout("XC staging size overflows usize".into());
    let mut single = 0usize;
    let mut sum = 0usize;
    for &(na, nb) in batches {
        if na == 0 {
            continue;
        }
        let wide = nb.checked_mul(4).ok_or_else(overflow)?;
        let m = |a: usize, b: usize| a.checked_mul(b).and_then(|x| x.checked_mul(2));
        let need = [m(na, nbf.max(wide)), m(nbf, wide), m(na, wide.max(na))]
            .into_iter()
            .try_fold(0usize, |acc, x| x.map(|x| acc.max(x)))
            .ok_or_else(overflow)?;
        single = single.max(need);
        sum = sum.checked_add(need).ok_or_else(overflow)?;
    }
    Ok(single.max(sum.min(STAGE_ELEMS)))
}

/// One wave: batches `[b0, end)`, each batch's `[in, mid, out]` offsets and
/// lengths in the staging buffers, and the per-buffer totals.
#[derive(Debug, PartialEq)]
pub(crate) struct Wave {
    pub end: usize,
    pub off: Vec<[usize; 3]>,
    pub len: Vec<[usize; 3]>,
    pub total: [usize; 3],
}

/// The maximal wave starting at `b0` whose per-buffer totals fit `stage`. A
/// batch that alone exceeds `stage` is a typed `Layout` refusal (the stage was
/// sized for every batch, so this means an input wider than `nbf`).
pub(crate) fn plan_wave(
    b0: usize,
    nbatches: usize,
    need: impl Fn(usize) -> [usize; 3],
    stage: usize,
) -> Result<Wave, GpuError> {
    let mut w = Wave {
        end: b0,
        off: Vec::new(),
        len: Vec::new(),
        total: [0; 3],
    };
    while w.end < nbatches {
        let n = need(w.end);
        if n.iter().any(|&x| x > stage) {
            return Err(GpuError::Layout(format!(
                "XC batch {} needs {n:?} staging elements but {stage} are reserved",
                w.end
            )));
        }
        if (0..3).any(|i| w.total[i] + n[i] > stage) {
            break;
        }
        w.off.push(w.total);
        w.len.push(n);
        for i in 0..3 {
            w.total[i] += n[i];
        }
        w.end += 1;
    }
    Ok(w)
}

/// Run `f(j, chunk_j)` over consecutive chunks of `buf` (lengths `lens`),
/// serially or on rayon; each chunk is written by exactly one call.
fn fill_chunks<F>(serial: bool, buf: &mut [f64], lens: &[usize], f: F) -> Result<(), GtoEvalError>
where
    F: Fn(usize, &mut [f64]) -> Result<(), GtoEvalError> + Sync + Send,
{
    let mut chunks = Vec::with_capacity(lens.len());
    let mut rest = buf;
    for &len in lens {
        let (head, tail) = rest.split_at_mut(len);
        chunks.push(head);
        rest = tail;
    }
    if serial {
        chunks
            .into_iter()
            .enumerate()
            .try_for_each(|(j, c)| f(j, c))
    } else {
        chunks
            .into_par_iter()
            .enumerate()
            .try_for_each(|(j, c)| f(j, c))
    }
}

/// `D[funcs, funcs]` (dense) or `C[funcs, :]` (factored), row-major, into `out`
/// — the values `DensityInput::gather` produces.
fn gather_into(d: &DensityInput, funcs: &[u32], out: &mut [f64]) {
    match d {
        DensityInput::Dense(m) => {
            let n = funcs.len();
            for (i, &fi) in funcs.iter().enumerate() {
                for (j, &fj) in funcs.iter().enumerate() {
                    out[i * n + j] = m[(fi as usize, fj as usize)];
                }
            }
        }
        DensityInput::Factored { c_occ, .. } => {
            let k = c_occ.ncols();
            for (i, &fi) in funcs.iter().enumerate() {
                let row = c_occ.row(fi as usize);
                for (o, &v) in out[i * k..(i + 1) * k].iter_mut().zip(row.iter()) {
                    *o = v;
                }
            }
        }
    }
}

/// Mixed precision was requested but this grid runs its V_xc in f64 on the
/// device: counted (`mixed_fallback_f64`) and announced, never silent.
fn mixed_unavailable(e: &GpuError) {
    eprintln!(
        "[ferric] gpu: KS-DFT XC potential runs in f64 on the device (mixed xc-vxc unavailable: {})",
        one_line(e)
    );
    stats::note_mixed_fallback();
}

impl MixedState {
    /// Probe the flush kernel (a typed `Kernel` error moves nothing), then
    /// reserve the f32 copies; a pool refusal is `Soft` (f64 stays usable).
    fn reserve(
        dev: &Arc<Device>,
        pool: &ferric_core::gpu::pool::DevicePool,
        ao_len: usize,
        stage: usize,
        max_na2: usize,
    ) -> Result<Self, GpuError> {
        dev.axpy_f32_to_f64()?;
        let z = |label: &str, len: usize| DeviceMatrix::<f32>::zeros(dev, pool, label, len, 1);
        let c32 = z("KS-DFT XC mixed panel scratch (f32)", max_na2.max(1))?;
        let stage_in32 = z("KS-DFT XC staging (operands, f32)", stage)?;
        let ao32 = z("KS-DFT grid AO blocks (device, f32)", ao_len)?;
        Ok(Self {
            ao32,
            stage_in32,
            c32,
            max_abs_x: 0.0,
        })
    }

    /// Round one AO block to f32 and send it to `ao32[at..]`.
    fn write_block(&mut self, dev: &Device, at: usize, src: &[f64]) -> Result<(), MixedSetup> {
        let r = round_to_f32("XC AO block", src).map_err(MixedSetup::Soft)?;
        self.max_abs_x = src.iter().fold(self.max_abs_x, |m, v| m.max(v.abs()));
        dev.stream
            .memcpy_htod(&r, &mut self.ao32.buf_mut().slice_mut(at..at + r.len()))
            .map_err(|e| MixedSetup::Hard(cuda("H2D XC AO block (f32)", e)))
    }
}

impl DeviceXc {
    /// Reserve the staging buffers and the AO copy (charged before any byte
    /// moves), then send every batch's block, evaluating it on the host first in
    /// recompute mode.
    fn upload(
        dev: &Arc<Device>,
        pool: &ferric_core::gpu::pool::DevicePool,
        sg: &ScreenedGrid,
        grid: &[GridPoint],
    ) -> Result<Self, GpuError> {
        let overflow = || GpuError::Layout("XC AO block sizes overflow usize".into());
        let mut ao_off = Vec::with_capacity(sg.batches.len());
        let mut total = 0usize;
        for b in &sg.batches {
            ao_off.push(total);
            let len = b
                .funcs
                .len()
                .checked_mul(4 * b.idx.len())
                .ok_or_else(overflow)?;
            total = total.checked_add(len).ok_or_else(overflow)?;
        }
        if total == 0 {
            return Err(GpuError::Layout(
                "no basis function is significant on any batch; nothing to offload".into(),
            ));
        }
        let shapes: Vec<(usize, usize)> = sg
            .batches
            .iter()
            .map(|b| (b.funcs.len(), b.idx.len()))
            .collect();
        let stage = stage_elems(sg.nbf, &shapes)?;
        let z = |label: &str, len: usize| DeviceMatrix::<f64>::zeros(dev, pool, label, len, 1);
        let stage_in = z("KS-DFT XC staging (operands)", stage)?;
        let stage_mid = z("KS-DFT XC staging (psi)", stage)?;
        let stage_out = z("KS-DFT XC staging (results)", stage)?;
        let mut ao = z(AO_LABEL, total)?;
        let mut mixed = if mixed_vxc_requested() {
            let max_na2 = shapes.iter().map(|&(na, _)| na * na).max().unwrap_or(0);
            match MixedState::reserve(dev, pool, total, stage, max_na2) {
                Ok(m) => Some(m),
                Err(e) => {
                    mixed_unavailable(&e);
                    None
                }
            }
        } else {
            None
        };
        let n = sg.batches.len();
        let mut b0 = 0usize;
        while b0 < n {
            let mut b1 = b0;
            let mut bytes = 0usize;
            while b1 < n {
                let len = 8 * shapes[b1].0 * 4 * shapes[b1].1;
                if b1 > b0 && bytes + len > UPLOAD_STAGE_BYTES {
                    break;
                }
                bytes += len;
                b1 += 1;
            }
            let blocks: Vec<Cow<'_, Array2<f64>>> = (b0..b1)
                .into_par_iter()
                .map(|bi| sg.batch_ao(bi, grid))
                .collect::<Result<_, _>>()
                .map_err(|e| GpuError::Layout(format!("AO evaluation for the upload: {e:?}")))?;
            for (j, blk) in blocks.iter().enumerate() {
                let bi = b0 + j;
                let src = blk
                    .as_slice()
                    .ok_or_else(|| GpuError::Layout("AO block not contiguous".into()))?;
                if src.is_empty() {
                    continue;
                }
                let at = ao_off[bi];
                dev.stream
                    .memcpy_htod(src, &mut ao.buf_mut().slice_mut(at..at + src.len()))
                    .map_err(|e| cuda(&format!("H2D XC AO block {bi}"), e))?;
                if let Some(m) = mixed.as_mut() {
                    match m.write_block(dev, at, src) {
                        Ok(()) => {}
                        Err(MixedSetup::Soft(e)) => {
                            mixed_unavailable(&e);
                            mixed = None;
                        }
                        Err(MixedSetup::Hard(e)) => return Err(e),
                    }
                }
            }
            b0 = b1;
        }
        dev.stream
            .synchronize()
            .map_err(|e| cuda("sync after the XC AO upload", e))?;
        stats::note_resident_upload(ao.bytes());
        if let Some(m) = &mixed {
            stats::note_resident_upload(m.ao32.bytes());
        }
        Ok(Self {
            dev: Arc::clone(dev),
            ao,
            ao_off,
            stage_in,
            stage_mid,
            stage_out,
            mixed,
        })
    }

    fn stage(&self) -> usize {
        self.stage_in
            .buf()
            .len()
            .min(self.stage_mid.buf().len())
            .min(self.stage_out.buf().len())
    }

    /// `host[..]` into the operand staging buffer, launches by `launch`, then
    /// `out_len` results back. One synchronisation.
    fn round_trip(
        &mut self,
        host_in: &[f64],
        out_len: usize,
        launch: impl FnOnce(&mut Self) -> Result<(), GpuError>,
    ) -> Result<Vec<f64>, GpuError> {
        if !host_in.is_empty() {
            self.dev
                .stream
                .memcpy_htod(
                    host_in,
                    &mut self.stage_in.buf_mut().slice_mut(..host_in.len()),
                )
                .map_err(|e| cuda("H2D XC operands", e))?;
        }
        launch(self)?;
        let mut out = vec![0.0f64; out_len];
        if out_len > 0 {
            self.dev
                .stream
                .memcpy_dtoh(&self.stage_out.buf().slice(..out_len), &mut out)
                .map_err(|e| cuda("D2H XC results", e))?;
        }
        self.dev
            .stream
            .synchronize()
            .map_err(|e| cuda("sync after an XC wave", e))?;
        Ok(out)
    }

    /// Pass 1 for every density in `ds`: `out[i][bi]`.
    fn density(
        &mut self,
        sg: &ScreenedGrid,
        grid: &[GridPoint],
        ds: &[DensityInput],
        need_tau: bool,
        serial: bool,
    ) -> Result<Vec<Vec<BatchDensity>>, Fail> {
        let nbat = sg.batches.len();
        let ncols = |nb: usize| if need_tau { 4 * nb } else { nb };
        let need = |bi: usize| -> [usize; 3] {
            let b = &sg.batches[bi];
            let (na, nc) = (b.funcs.len(), ncols(b.idx.len()));
            let mut n = [0usize; 3];
            if na == 0 {
                return n;
            }
            for d in ds {
                match d {
                    DensityInput::Dense(_) => n[0] = n[0].saturating_add(na.saturating_mul(na)),
                    DensityInput::Factored { c_occ, .. } => {
                        let k = c_occ.ncols();
                        n[0] = n[0].saturating_add(na.saturating_mul(k));
                        n[1] = n[1].saturating_add(k.saturating_mul(nc));
                    }
                }
                n[2] = n[2].saturating_add(na.saturating_mul(nc));
            }
            n
        };
        let mut out: Vec<Vec<BatchDensity>> = ds.iter().map(|_| Vec::with_capacity(nbat)).collect();
        let (mut h2d, mut d2h) = (0usize, 0usize);
        let mut b0 = 0usize;
        while b0 < nbat {
            let w = plan_wave(b0, nbat, &need, self.stage())?;
            let in_lens: Vec<usize> = w.len.iter().map(|l| l[0]).collect();
            let mut host_in = vec![0.0f64; w.total[0]];
            fill_chunks(serial, &mut host_in, &in_lens, |j, chunk| {
                let funcs = &sg.batches[b0 + j].funcs;
                let mut at = 0usize;
                for d in ds {
                    let len = match d {
                        DensityInput::Dense(_) => funcs.len() * funcs.len(),
                        DensityInput::Factored { c_occ, .. } => funcs.len() * c_occ.ncols(),
                    };
                    gather_into(d, funcs, &mut chunk[at..at + len]);
                    at += len;
                }
                Ok(())
            })?;
            let host_out = self.round_trip(&host_in, w.total[2], |me| {
                for (j, off) in w.off.iter().enumerate() {
                    me.launch_density(sg, b0 + j, *off, ds, &ncols)?;
                }
                Ok(())
            })?;
            h2d += 8 * host_in.len();
            d2h += 8 * host_out.len();
            let parts = map_indexed(serial, w.end - b0, |j| {
                let bi = b0 + j;
                let b = &sg.batches[bi];
                let (na, nb) = (b.funcs.len(), b.idx.len());
                let len = na * ncols(nb);
                let ao = sg.batch_ao(bi, grid)?;
                let mut at = w.off[j][2];
                Ok(ds
                    .iter()
                    .map(|d| {
                        let p = &host_out[at..at + len];
                        at += len;
                        match d {
                            DensityInput::Factored { scale, .. } => {
                                // `scale` is 1 or 2 (exact in binary): no rounding.
                                let scaled: Vec<f64> = p.iter().map(|&x| x * *scale).collect();
                                batch_density_from_phi(&ao, &scaled, nb, need_tau)
                            }
                            DensityInput::Dense(_) => batch_density_from_phi(&ao, p, nb, need_tau),
                        }
                    })
                    .collect::<Vec<_>>())
            })?;
            for per_batch in parts {
                for (o, p) in out.iter_mut().zip(per_batch) {
                    o.push(p);
                }
            }
            b0 = w.end;
        }
        stats::note_xc_pass(h2d, d2h);
        Ok(out)
    }

    /// Launch batch `bi`'s pass-1 products (stream-ordered, no sync).
    fn launch_density(
        &mut self,
        sg: &ScreenedGrid,
        bi: usize,
        [mut ii, mut mm, mut oo]: [usize; 3],
        ds: &[DensityInput],
        ncols: impl Fn(usize) -> usize,
    ) -> Result<(), GpuError> {
        let b = &sg.batches[bi];
        let (na, nb) = (b.funcs.len(), b.idx.len());
        if na == 0 {
            return Ok(());
        }
        let nc = ncols(nb);
        let Self {
            dev,
            ao,
            ao_off,
            stage_in,
            stage_mid,
            stage_out,
            ..
        } = self;
        let at = ao_off[bi];
        let ao_tw = twin(na, 4 * nb);
        let x_tw = ao_tw.slice(s![.., ..nc]);
        for d in ds {
            let x = dev_right_padded(ao.buf().slice(at..at + na * 4 * nb), &x_tw)?;
            let mut phi = stage_out.buf_mut().slice_mut(oo..oo + na * nc);
            match d {
                DensityInput::Dense(_) => {
                    let dtw = twin(na, na);
                    let dsub = dev_left(stage_in.buf().slice(ii..ii + na * na), &dtw.view())?;
                    gemm_f64_dev_into(dev, na, na, nc, &dsub, &x, &mut phi, na.max(1))?;
                    ii += na * na;
                }
                DensityInput::Factored { c_occ, .. } => {
                    let k = c_occ.ncols();
                    let ctw = twin(na, k);
                    let csub = || stage_in.buf().slice(ii..ii + na * k);
                    {
                        let csub_t = dev_left(csub(), &ctw.t())?;
                        let mut psi = stage_mid.buf_mut().slice_mut(mm..mm + k * nc);
                        gemm_f64_dev_into(dev, k, na, nc, &csub_t, &x, &mut psi, na.max(1))?;
                    }
                    let ptw = twin(k, nc);
                    let psi = dev_right(stage_mid.buf().slice(mm..mm + k * nc), &ptw.view())?;
                    let c = dev_left(csub(), &ctw.view())?;
                    gemm_f64_dev_into(dev, na, k, nc, &c, &psi, &mut phi, k.max(1))?;
                    ii += na * k;
                    mm += k * nc;
                }
            }
            oo += na * nc;
        }
        Ok(())
    }

    /// Pass 2: the `nmats` V matrices, reduced in the CPU path's fixed groups.
    fn potential<F>(
        &mut self,
        sg: &ScreenedGrid,
        grid: &[GridPoint],
        nmats: usize,
        factors: &F,
        flags: (bool, bool),
        (serial, mixed_req): (bool, bool),
    ) -> Result<Vec<Array2<f64>>, Fail>
    where
        F: Fn(&XcBatch, usize) -> BatchFactors + Sync,
    {
        // Meta-GGA τ (flags.1) is on the never-f32 list: never mixed here.
        let mixed_req = mixed_req && !flags.1;
        let (nbat, nbf) = (sg.batches.len(), sg.nbf);
        let k_panel = effective_k_panel();
        let (mut mixed_waves, mut panels) = (0usize, 0usize);
        let ranges = group_ranges(nbat, n_groups(nbat, nbf, nmats));
        let mut accs: Vec<Vec<Array2<f64>>> = ranges
            .iter()
            .map(|_| (0..nmats).map(|_| Array2::zeros((nbf, nbf))).collect())
            .collect();
        let ncols = |nb: usize| if flags.1 { 4 * nb } else { nb };
        let need = |bi: usize| -> [usize; 3] {
            let b = &sg.batches[bi];
            let na = b.funcs.len();
            let nc = ncols(b.idx.len());
            [
                nmats.saturating_mul(na).saturating_mul(nc),
                0,
                nmats.saturating_mul(na).saturating_mul(na),
            ]
        };
        let (mut h2d, mut d2h) = (0usize, 0usize);
        let mut b0 = 0usize;
        while b0 < nbat {
            let w = plan_wave(b0, nbat, &need, self.stage())?;
            let in_lens: Vec<usize> = w.len.iter().map(|l| l[0]).collect();
            let mut host_in = vec![0.0f64; w.total[0]];
            fill_chunks(serial, &mut host_in, &in_lens, |j, chunk| {
                let bi = b0 + j;
                let b = &sg.batches[bi];
                if b.funcs.is_empty() {
                    return Ok(());
                }
                let nb = b.idx.len();
                let len = b.funcs.len() * ncols(nb);
                let ao = sg.batch_ao(bi, grid)?;
                for mat in 0..nmats {
                    let fac = factors(b, mat);
                    fill_amat(&ao, nb, &fac, flags, &mut chunk[mat * len..(mat + 1) * len]);
                }
                Ok(())
            })?;
            // Mixed for this wave only when the ladder asked for it, the f32
            // state exists, and no panel sum can leave the f32 range.
            let mixed_wave = match (&self.mixed, mixed_req) {
                (Some(m), true) => {
                    let fits = mixed_panel_fits(k_panel, max_abs_nan(&host_in), m.max_abs_x);
                    if !fits {
                        stats::note_mixed_fallback();
                    }
                    fits
                }
                _ => false,
            };
            let a32 = if mixed_wave {
                wave_a_f32(&host_in)
            } else {
                None
            };
            let host_out = if let Some(a32) = a32 {
                let out = self.round_trip(&[], w.total[2], |me| {
                    let m = me.mixed.as_mut().expect("mixed_wave implies the f32 state");
                    if !a32.is_empty() {
                        me.dev
                            .stream
                            .memcpy_htod(&a32, &mut m.stage_in32.buf_mut().slice_mut(..a32.len()))
                            .map_err(|e| cuda("H2D XC operands (f32)", e))?;
                    }
                    for (j, off) in w.off.iter().enumerate() {
                        panels += me.launch_potential(sg, b0 + j, *off, (nmats, true), &ncols)?;
                    }
                    Ok(())
                })?;
                h2d += 4 * a32.len();
                mixed_waves += 1;
                out
            } else {
                let out = self.round_trip(&host_in, w.total[2], |me| {
                    for (j, off) in w.off.iter().enumerate() {
                        me.launch_potential(sg, b0 + j, *off, (nmats, false), &ncols)?;
                    }
                    Ok(())
                })?;
                h2d += 8 * host_in.len();
                out
            };
            d2h += 8 * host_out.len();
            // Scatter: every group touched by this wave, its batches in
            // ascending order (waves ascend too), groups independent.
            let scatter = |(gi, acc): (usize, &mut Vec<Array2<f64>>)| {
                let (r0, r1) = ranges[gi];
                for bi in r0.max(b0)..r1.min(w.end) {
                    let funcs = &sg.batches[bi].funcs;
                    let na = funcs.len();
                    if na == 0 {
                        continue;
                    }
                    let base = w.off[bi - b0][2];
                    for (mat, a) in acc.iter_mut().enumerate() {
                        let at = base + mat * na * na;
                        scatter_sym(&host_out[at..at + na * na], funcs, a);
                    }
                }
            };
            let touched = accs
                .iter_mut()
                .enumerate()
                .filter(|(gi, _)| ranges[*gi].0 < w.end && ranges[*gi].1 > b0);
            if serial {
                touched.for_each(scatter);
            } else {
                touched
                    .collect::<Vec<_>>()
                    .into_par_iter()
                    .for_each(scatter);
            }
            b0 = w.end;
        }
        // Ascending group order — the one and only cross-group summation.
        let mut out: Vec<Array2<f64>> = (0..nmats).map(|_| Array2::zeros((nbf, nbf))).collect();
        for group in &accs {
            for (o, a) in out.iter_mut().zip(group) {
                *o += a;
            }
        }
        stats::note_xc_pass(h2d, d2h);
        if mixed_waves > 0 {
            // Bytes are already in the pass's tally above.
            stats::note_mixed(panels, 0, 0);
            super::note_mixed_vxc_pass();
        }
        Ok(out)
    }

    /// Launch batch `bi`'s pass-2 products `V_b = A·Xᵀ` (stream-ordered, no
    /// sync): f64 dgemm, or with `mixed` the f32-operand / f64-accumulate
    /// kernel on `ao32` and `stage_in32`. Returns the mixed panels flushed.
    fn launch_potential(
        &mut self,
        sg: &ScreenedGrid,
        bi: usize,
        [ii, _, oo]: [usize; 3],
        (nmats, mixed): (usize, bool),
        ncols: impl Fn(usize) -> usize,
    ) -> Result<usize, GpuError> {
        let b = &sg.batches[bi];
        let (na, nb) = (b.funcs.len(), b.idx.len());
        if na == 0 {
            return Ok(0);
        }
        let nc = ncols(nb);
        let Self {
            dev,
            ao,
            ao_off,
            stage_in,
            stage_out,
            mixed: mixed_state,
            ..
        } = self;
        let at = ao_off[bi];
        let ao_tw = twin(na, 4 * nb);
        let xt_tw = ao_tw.slice(s![.., ..nc]);
        let xt_tw = xt_tw.t();
        let atw = twin(na, nc);
        let mut panels = 0usize;
        for mat in 0..nmats {
            let a0 = ii + mat * na * nc;
            let o0 = oo + mat * na * na;
            let mut vb = stage_out.buf_mut().slice_mut(o0..o0 + na * na);
            match (mixed, mixed_state.as_mut()) {
                (true, Some(m)) => {
                    let xt = dev_right_padded(m.ao32.buf().slice(at..at + na * 4 * nb), &xt_tw)?;
                    let a = dev_left(m.stage_in32.buf().slice(a0..a0 + na * nc), &atw.view())?;
                    let mut c32 = m.c32.buf_mut().slice_mut(..na * na);
                    panels += gemm_f32_f64acc_dev_into(
                        dev,
                        (na, nc, na),
                        &a,
                        &xt,
                        &mut c32,
                        &mut vb,
                        effective_k_panel(),
                    )?;
                }
                (true, None) => {
                    return Err(GpuError::Layout(
                        "mixed XC launch without the f32 state".into(),
                    ))
                }
                (false, _) => {
                    let xt = dev_right_padded(ao.buf().slice(at..at + na * 4 * nb), &xt_tw)?;
                    let a = dev_left(stage_in.buf().slice(a0..a0 + na * nc), &atw.view())?;
                    gemm_f64_dev_into(dev, na, nc, na, &a, &xt, &mut vb, nc.max(1))?;
                }
            }
        }
        Ok(panels)
    }
}

/// A mixed wave's `A` rounded to f32, or `None` (counted in
/// `mixed_fallback_f64`) when some element does not fit: the wave then runs f64.
fn wave_a_f32(host_in: &[f64]) -> Option<Vec<f32>> {
    match round_to_f32("XC A", host_in) {
        Ok(a32) => Some(a32),
        Err(_) => {
            stats::note_mixed_fallback();
            None
        }
    }
}

/// `max |x|`, NaN if any element is NaN.
fn max_abs_nan(x: &[f64]) -> f64 {
    x.iter().fold(0.0_f64, |m, &v| {
        if v.is_nan() || m.is_nan() {
            f64::NAN
        } else {
            m.max(v.abs())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waves_cover_every_batch_once_within_the_stage() {
        let needs: Vec<[usize; 3]> = (0..37)
            .map(|i| [(i * 7) % 23, (i * 3) % 5, (i * 11) % 19 + 1])
            .collect();
        let stage = 40;
        let mut b0 = 0;
        let mut seen = 0;
        while b0 < needs.len() {
            let w = plan_wave(b0, needs.len(), |bi| needs[bi], stage).unwrap();
            assert!(w.end > b0, "a wave must make progress");
            let mut run = [0usize; 3];
            for (j, (off, len)) in w.off.iter().zip(&w.len).enumerate() {
                assert_eq!(*len, needs[b0 + j]);
                assert_eq!(*off, run, "offsets are the running totals");
                for i in 0..3 {
                    run[i] += len[i];
                }
            }
            assert_eq!(run, w.total);
            assert!(w.total.iter().all(|&t| t <= stage));
            if w.end < needs.len() {
                let n = needs[w.end];
                assert!(
                    (0..3).any(|i| w.total[i] + n[i] > stage),
                    "a wave stops only when the next batch does not fit"
                );
            }
            seen += w.end - b0;
            b0 = w.end;
        }
        assert_eq!(seen, needs.len());
    }

    #[test]
    fn mixed_panels_that_could_overflow_f32_run_f64() {
        assert!(mixed_panel_fits(64, 1e3, 1e3));
        let edge = 0.5 * f64::from(f32::MAX) / 64.0;
        assert!(!mixed_panel_fits(64, edge, 1.0), "the bound is strict");
        assert!(mixed_panel_fits(64, edge * 0.99, 1.0));
        assert!(!mixed_panel_fits(64, f64::NAN, 1.0));
        assert!(!mixed_panel_fits(64, f64::INFINITY, 1.0));
        assert!(max_abs_nan(&[1.0, -3.0, 2.0]) == 3.0);
        assert!(max_abs_nan(&[1.0, f64::NAN, 5.0]).is_nan());
        assert_eq!(max_abs_nan(&[]), 0.0);
    }

    #[test]
    fn a_batch_wider_than_the_stage_is_a_typed_refusal() {
        // Reaching a batch that alone exceeds the stage refuses the wave.
        let need = |bi: usize| if bi == 1 { [0, 9, 0] } else { [1, 1, 1] };
        let e = plan_wave(0, 2, need, 8).unwrap_err();
        assert!(matches!(e, GpuError::Layout(_)), "{e:?}");
        // Ending the range before it plans normally.
        let w = plan_wave(0, 1, need, 8).unwrap();
        assert_eq!((w.end, w.total), (1, [1, 1, 1]));
    }

    #[test]
    fn stage_holds_the_largest_batch_and_caps_small_grids() {
        // nbf 10, batches (nact, nb): the largest single need is
        // max(2·na·max(nbf, 4nb), 2·nbf·4nb, 2·na·max(4nb, na)).
        let shapes = [(3, 2), (0, 128), (10, 4)];
        let single = |na: usize, nb: usize| {
            let w = 4 * nb;
            (2 * na * 10.max(w)).max(2 * 10 * w).max(2 * na * w.max(na))
        };
        let want_sum = single(3, 2) + single(10, 4);
        assert_eq!(stage_elems(10, &shapes).unwrap(), want_sum);
        // A batch bigger than the default stage sets the size on its own.
        let big = [(4000, 128)];
        assert_eq!(stage_elems(4000, &big).unwrap(), single_big());
        fn single_big() -> usize {
            let (na, nbf, w) = (4000usize, 4000usize, 512usize);
            (2 * na * nbf.max(w))
                .max(2 * nbf * w)
                .max(2 * na * w.max(na))
        }
        assert!(single_big() > STAGE_ELEMS);
        assert!(stage_elems(usize::MAX, &[(usize::MAX, 2)]).is_err());
    }
}
