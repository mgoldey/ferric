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
//! Precision is f64 for every `[gpu] precision` setting. A mixed variant (f32
//! resident `Bp`, f64 accumulation) is not implemented; see the plan in the commit
//! that introduces this file.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use ferric_core::gpu::device::{device, Device, GpuError};
use ferric_core::gpu::gemv::{gemv_f64_dev, GemvSpec};
use ferric_core::gpu::mixed_host::{gamma, U64};
use ferric_core::gpu::pool::DevicePool;
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

/// Test seam: the dispatcher reports "not handled" so the CPU path runs.
#[doc(hidden)]
pub static FORCE_HOST: AtomicBool = AtomicBool::new(false);
/// Test seam: a device build fails after pass 1 (a mid-build CUDA error stand-in).
#[doc(hidden)]
pub static FORCE_BUILD_FAILURE: AtomicBool = AtomicBool::new(false);

/// Packed pair count `n(n+1)/2`, or a typed overflow refusal.
pub fn pair_len(n: usize) -> Result<usize, GpuError> {
    n.checked_mul(n.saturating_add(1))
        .map(|v| v / 2)
        .ok_or_else(|| GpuError::Layout(format!("n = {n}: n(n+1)/2 overflows usize")))
}

/// Bytes the resident state holds: `8·(naux·pair + pair + 2·naux + pair)`
/// (`Bp`, `w`, `d_P`, `c`, packed `J`).
pub fn resident_bytes(naux: usize, pair: usize) -> Result<usize, GpuError> {
    let overflow = || GpuError::Layout(format!("DF-J resident {naux}x{pair} overflows usize"));
    let b = naux.checked_mul(pair).ok_or_else(overflow)?;
    let vecs = pair
        .checked_mul(2)
        .and_then(|p| p.checked_add(naux.checked_mul(2)?))
        .ok_or_else(overflow)?;
    b.checked_add(vecs)
        .and_then(|e| e.checked_mul(8))
        .ok_or_else(overflow)
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

struct Bufs {
    b: DeviceMatrix<f64>,
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

impl DeviceDfJ {
    /// Reserve the device state for a `(naux × pair)` packed tensor. The small
    /// vectors are reserved FIRST so a refusal of the large `Bp` leaks nothing;
    /// `Bp` is charged under "DF-J raw B (packed)" before any byte moves.
    pub fn reserve(
        dev: &Arc<Device>,
        pool: &DevicePool,
        naux: usize,
        pair: usize,
    ) -> Result<Self, GpuError> {
        resident_bytes(naux, pair)?;
        let (rows_per_call, _) = row_plan(naux, pair);
        let bufs = if naux == 0 || pair == 0 {
            None
        } else {
            let z = |label: &str, len: usize| DeviceMatrix::<f64>::zeros(dev, pool, label, len, 1);
            let w = z("DF-J weights w", pair)?;
            let d = z("DF-J d_P", naux)?;
            let c = z("DF-J c", naux)?;
            let j = z("DF-J packed J", pair)?;
            let b = DeviceMatrix::<f64>::zeros(dev, pool, "DF-J raw B (packed)", naux, pair)?;
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

    /// Copy `rows` (`k·pair` doubles, `k` whole aux rows) into aux rows
    /// `p0..p0+k`. Rows must arrive in ascending order without gaps.
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
        let mut dst = bufs
            .b
            .buf_mut()
            .slice_mut(p0 * self.pair..(p0 + k) * self.pair);
        self.dev
            .stream
            .memcpy_htod(rows, &mut dst)
            .map_err(|e| GpuError::Cuda(format!("H2D DF-J raw B rows {p0}..{}: {e:?}", p0 + k)))?;
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
            stats::note_resident_upload(8 * self.naux * self.pair);
        }
        Ok(())
    }

    /// `reserve` + one `write_rows` + `finish_upload` for a host packed tensor.
    pub fn upload_packed(
        dev: &Arc<Device>,
        pool: &DevicePool,
        packed: &ArrayView2<f64>,
    ) -> Result<Self, GpuError> {
        let (naux, pair) = packed.dim();
        let mut me = Self::reserve(dev, pool, naux, pair)?;
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
        let mut r0 = 0usize;
        while r0 < self.naux {
            let r1 = (r0 + rows).min(self.naux);
            let a = bufs.b.buf().slice(r0 * pair..r1 * pair);
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
        let mut r0 = 0usize;
        while r0 < self.naux {
            let r1 = (r0 + rows).min(self.naux);
            let a = bufs.b.buf().slice(r0 * pair..r1 * pair);
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
) -> Result<DeviceDfJ, GpuError> {
    let pair = pair_len(src.nao())?;
    let mut d = DeviceDfJ::reserve(dev, pool, src.naux(), pair)?;
    fill_from_source(&mut d, src)?;
    Ok(d)
}

/// One device J build: `Ok(true)` means `j` holds the full J (no cross-rank
/// reduction is needed: the device path never runs under MPI); `Ok(false)` means
/// run the CPU path (`j` untouched by this call). `solve` is `DfJ`'s own metric
/// solve; its error is the caller's error, as on the CPU path. Never touches CUDA
/// when the mode is off or the seam forces the host.
pub fn try_build(
    slot: &mut DeviceSlot,
    src: &mut ThreeIndexSource,
    ctx: Option<&ParallelContext>,
    d: &Array2<f64>,
    j: &mut Array2<f64>,
    solve: impl FnOnce(&Array1<f64>) -> Result<Array1<f64>, FerricError>,
) -> Result<bool, FerricError> {
    if FORCE_HOST.load(Ordering::Relaxed) || matches!(slot, DeviceSlot::Declined) {
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
        match device(info.ordinal).and_then(|dev| upload_band(&dev, &pool, src)) {
            Ok(dd) => {
                eprintln!(
                    "[ferric] gpu: RI-J resident on device {} ({:.3} GB packed B, {} aux rows, nbf {})",
                    info.ordinal,
                    8.0 * (dd.naux() * dd.pair()) as f64 / 1e9,
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
    let outcome = if FORCE_BUILD_FAILURE.load(Ordering::Relaxed) {
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
