//! Device offload of the screened XC grid's per-batch GEMMs (feature `gpu`).
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
//! The device products are plain f64 `cublasDgemm` (no mixed precision: the
//! XC kernel is not on the mixed allowlist), one call per product with no k
//! blocking. They are NOT bit-identical to the host GEMMs (cuBLAS orders its
//! sums its own way); each product obeys the classical bound
//! `|ĉ − c| ≤ γ_k (|A||B|)` (Higham, ASNA §3.5) with `k = nact` (pass 1 dense,
//! pass 1 `Ψ`), `k = nocc` (pass 1 `Φ` from `Ψ`) or `k = ncols` (pass 2), the
//! same order of error as the host dgemm. Everything downstream is the host
//! code: the pass-2 contributions are scattered in the same fixed groups, each
//! group in ascending batch order, groups summed in ascending order. The result
//! is therefore deterministic run to run on one device and independent of the
//! rayon thread count.
//!
//! # Fallback (never silent)
//!
//! `[gpu] mode = off`, no ready device, no pool, or a call from inside a rayon
//! worker: the integration runs on the CPU and touches no CUDA. A pool refusal
//! or a CUDA error while uploading, or any device error during a pass, declines
//! the grid for its lifetime (one `[ferric] gpu:` notice, counted in
//! `stats::xc_declined`) and the pass that failed is recomputed on the CPU.
use std::borrow::Cow;
use std::sync::Arc;

use ndarray::{s, Array2};
use rayon::prelude::*;

use ferric_core::gpu::device::{device, Device, GpuError};
use ferric_core::gpu::gemm::{dev_left, dev_right, dev_right_padded, gemm_f64_dev_into};
use ferric_core::gpu::resident::DeviceMatrix;
use ferric_core::gpu::stats::{self, CpuReason};
use ferric_core::gpu::{GpuMode, GpuStatus};

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
    if force_host() || ferric_core::gpu::settings().mode == GpuMode::Off {
        return None;
    }
    let GpuStatus::Ready(info) = ferric_core::gpu::status() else {
        return None;
    };
    let mut slot = sg.device.lock().unwrap_or_else(|e| e.into_inner());
    if matches!(*slot, DeviceSlot::Declined) {
        return None;
    }
    if rayon::current_thread_index().is_some() {
        stats::note_cpu(CpuReason::InsideRayonWorker); // transient: the slot is untouched
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
    serial: bool,
    dev: &mut DevSession<'_>,
) -> Result<Option<Vec<Array2<f64>>>, GtoEvalError>
where
    F: Fn(&XcBatch, usize) -> BatchFactors + Sync,
{
    with_device(dev, |dx| {
        dx.potential(sg, grid, nmats, factors, flags, serial)
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
            }
            b0 = b1;
        }
        dev.stream
            .synchronize()
            .map_err(|e| cuda("sync after the XC AO upload", e))?;
        stats::note_resident_upload(ao.bytes());
        Ok(Self {
            dev: Arc::clone(dev),
            ao,
            ao_off,
            stage_in,
            stage_mid,
            stage_out,
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
        serial: bool,
    ) -> Result<Vec<Array2<f64>>, Fail>
    where
        F: Fn(&XcBatch, usize) -> BatchFactors + Sync,
    {
        let (nbat, nbf) = (sg.batches.len(), sg.nbf);
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
            let host_out = self.round_trip(&host_in, w.total[2], |me| {
                for (j, off) in w.off.iter().enumerate() {
                    me.launch_potential(sg, b0 + j, *off, nmats, &ncols)?;
                }
                Ok(())
            })?;
            h2d += 8 * host_in.len();
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
        Ok(out)
    }

    /// Launch batch `bi`'s pass-2 products `V_b = A·Xᵀ` (stream-ordered, no sync).
    fn launch_potential(
        &mut self,
        sg: &ScreenedGrid,
        bi: usize,
        [ii, _, oo]: [usize; 3],
        nmats: usize,
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
            stage_out,
            ..
        } = self;
        let at = ao_off[bi];
        let ao_tw = twin(na, 4 * nb);
        let xt_tw = ao_tw.slice(s![.., ..nc]);
        let xt_tw = xt_tw.t();
        let atw = twin(na, nc);
        for mat in 0..nmats {
            let xt = dev_right_padded(ao.buf().slice(at..at + na * 4 * nb), &xt_tw)?;
            let a0 = ii + mat * na * nc;
            let a = dev_left(stage_in.buf().slice(a0..a0 + na * nc), &atw.view())?;
            let o0 = oo + mat * na * na;
            let mut vb = stage_out.buf_mut().slice_mut(o0..o0 + na * na);
            gemm_f64_dev_into(dev, na, nc, na, &a, &xt, &mut vb, nc.max(1))?;
        }
        Ok(())
    }
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
