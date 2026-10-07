//! Process-wide counters for the GPU path. Tests assert deltas on these rather
//! than grepping log text, so a path that is "on" but never taken is visible.
use std::sync::atomic::{AtomicU64, Ordering};

static GEMM_OFFLOADED: AtomicU64 = AtomicU64::new(0);
static GEMM_CPU_INSIDE_WORKER: AtomicU64 = AtomicU64::new(0);
static GEMM_CPU_BELOW_THRESHOLD: AtomicU64 = AtomicU64::new(0);
static GEMM_CPU_POOL_FULL: AtomicU64 = AtomicU64::new(0);
static GEMM_CPU_LAYOUT: AtomicU64 = AtomicU64::new(0);
static GEMM_CPU_CUDA_ERROR: AtomicU64 = AtomicU64::new(0);
static GEMM_CPU_F32_RANGE: AtomicU64 = AtomicU64::new(0);
static BYTES_H2D: AtomicU64 = AtomicU64::new(0);
static BYTES_D2H: AtomicU64 = AtomicU64::new(0);
static GEMM_MIXED: AtomicU64 = AtomicU64::new(0);
static MIXED_PANELS: AtomicU64 = AtomicU64::new(0);
static MIXED_FALLBACK_F64: AtomicU64 = AtomicU64::new(0);
static RESIDENT_UPLOADS: AtomicU64 = AtomicU64::new(0);
static DFK_DEVICE_BUILDS: AtomicU64 = AtomicU64::new(0);
static DFK_DECLINED: AtomicU64 = AtomicU64::new(0);
static DFJ_DEVICE_BUILDS: AtomicU64 = AtomicU64::new(0);
static DFJ_DECLINED: AtomicU64 = AtomicU64::new(0);

/// Why a GEMM ran on the CPU although the GPU mode was not `off`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuReason {
    InsideRayonWorker,
    BelowThreshold,
    PoolFull,
    Layout,
    CudaError,
    /// A mixed-precision operand held a finite value beyond `f32::MAX`.
    F32Range,
}

/// Called by the device GEMM (this crate) — public and hidden because
/// `ferric-tensors`' dispatch also records its refusals here.
#[doc(hidden)]
pub fn note_offloaded(h2d: usize, d2h: usize) {
    GEMM_OFFLOADED.fetch_add(1, Ordering::Relaxed);
    BYTES_H2D.fetch_add(h2d as u64, Ordering::Relaxed);
    BYTES_D2H.fetch_add(d2h as u64, Ordering::Relaxed);
}

/// A mixed-precision GEMM ran on the device (`panels` flushes).
#[doc(hidden)]
pub fn note_mixed(panels: usize, h2d: usize, d2h: usize) {
    GEMM_MIXED.fetch_add(1, Ordering::Relaxed);
    MIXED_PANELS.fetch_add(panels as u64, Ordering::Relaxed);
    BYTES_H2D.fetch_add(h2d as u64, Ordering::Relaxed);
    BYTES_D2H.fetch_add(d2h as u64, Ordering::Relaxed);
}

/// A matrix was uploaded to live on the device beyond one GEMM
/// (`resident::DeviceMatrix`); `bytes` count toward `bytes_h2d` too.
#[doc(hidden)]
pub fn note_resident_upload(bytes: usize) {
    RESIDENT_UPLOADS.fetch_add(1, Ordering::Relaxed);
    BYTES_H2D.fetch_add(bytes as u64, Ordering::Relaxed);
}

/// One resident DF-K occupied-path build ran on the device (`h2d` = C_occ
/// bytes, `d2h` = K bytes; they count toward `bytes_h2d` / `bytes_d2h`).
#[doc(hidden)]
pub fn note_dfk_build(h2d: usize, d2h: usize) {
    DFK_DEVICE_BUILDS.fetch_add(1, Ordering::Relaxed);
    BYTES_H2D.fetch_add(h2d as u64, Ordering::Relaxed);
    BYTES_D2H.fetch_add(d2h as u64, Ordering::Relaxed);
}

/// A DF-K that will stay on the CPU for its lifetime (ineligible source, MPI
/// world, pool refusal, CUDA error). Counting only; the dispatcher prints the notice.
#[doc(hidden)]
pub fn note_dfk_declined(reason: &str) {
    let _ = reason; // the dispatcher prints the one-line notice
    DFK_DECLINED.fetch_add(1, Ordering::Relaxed);
}

/// One resident RI-J build ran on the device (`h2d` = w and c uploaded, `d2h` =
/// d_P and the packed J downloaded; they count toward `bytes_h2d` / `bytes_d2h`).
#[doc(hidden)]
pub fn note_dfj_build(h2d: usize, d2h: usize) {
    DFJ_DEVICE_BUILDS.fetch_add(1, Ordering::Relaxed);
    BYTES_H2D.fetch_add(h2d as u64, Ordering::Relaxed);
    BYTES_D2H.fetch_add(d2h as u64, Ordering::Relaxed);
}

/// An RI-J that will stay on the CPU for its lifetime. Counting only.
#[doc(hidden)]
pub fn note_dfj_declined(reason: &str) {
    let _ = reason; // the dispatcher prints the one-line notice
    DFJ_DECLINED.fetch_add(1, Ordering::Relaxed);
}

/// Mixed was requested and allowed, but the call ran in f64 on the device
/// (kernel unavailable or a mixed-path CUDA error). Never silent.
#[doc(hidden)]
pub fn note_mixed_fallback() {
    MIXED_FALLBACK_F64.fetch_add(1, Ordering::Relaxed);
}

/// `FERRIC_GPU_TRACE`, read once: the per-fallback print costs one cached load
/// per refusal and nothing on the offload path.
fn trace_on() -> bool {
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TRACE.get_or_init(super::config::gpu_trace)
}

/// Count a CPU fallback; with `FERRIC_GPU_TRACE` set, also print one line
/// naming the reason (`InsideRayonWorker`, `BelowThreshold`, `PoolFull`,
/// `Layout`, `CudaError`, `F32Range`).
#[doc(hidden)]
pub fn note_cpu(reason: CpuReason) {
    note_cpu_detail(reason, "");
}

/// [`note_cpu`] with a free-text detail (shape, error) appended to the trace line.
#[doc(hidden)]
pub fn note_cpu_detail(reason: CpuReason, detail: &str) {
    if trace_on() {
        eprintln!("[gpu] GEMM fell back to the CPU: {reason:?} {detail}");
    }
    let c = match reason {
        CpuReason::InsideRayonWorker => &GEMM_CPU_INSIDE_WORKER,
        CpuReason::BelowThreshold => &GEMM_CPU_BELOW_THRESHOLD,
        CpuReason::PoolFull => &GEMM_CPU_POOL_FULL,
        CpuReason::Layout => &GEMM_CPU_LAYOUT,
        CpuReason::CudaError => &GEMM_CPU_CUDA_ERROR,
        CpuReason::F32Range => &GEMM_CPU_F32_RANGE,
    };
    c.fetch_add(1, Ordering::Relaxed);
}

/// A copy of the counters at one instant.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuStatsSnapshot {
    pub gemm_offloaded: u64,
    pub gemm_cpu_inside_worker: u64,
    pub gemm_cpu_below_threshold: u64,
    pub gemm_cpu_pool_full: u64,
    pub gemm_cpu_layout: u64,
    pub gemm_cpu_cuda_error: u64,
    pub gemm_cpu_f32_range: u64,
    pub bytes_h2d: u64,
    pub bytes_d2h: u64,
    pub gemm_mixed: u64,
    pub mixed_panels: u64,
    pub mixed_fallback_f64: u64,
    pub resident_uploads: u64,
    pub dfk_device_builds: u64,
    pub dfk_declined: u64,
    pub dfj_device_builds: u64,
    pub dfj_declined: u64,
}

pub fn stats() -> GpuStatsSnapshot {
    GpuStatsSnapshot {
        gemm_offloaded: GEMM_OFFLOADED.load(Ordering::Relaxed),
        gemm_cpu_inside_worker: GEMM_CPU_INSIDE_WORKER.load(Ordering::Relaxed),
        gemm_cpu_below_threshold: GEMM_CPU_BELOW_THRESHOLD.load(Ordering::Relaxed),
        gemm_cpu_pool_full: GEMM_CPU_POOL_FULL.load(Ordering::Relaxed),
        gemm_cpu_layout: GEMM_CPU_LAYOUT.load(Ordering::Relaxed),
        gemm_cpu_cuda_error: GEMM_CPU_CUDA_ERROR.load(Ordering::Relaxed),
        gemm_cpu_f32_range: GEMM_CPU_F32_RANGE.load(Ordering::Relaxed),
        bytes_h2d: BYTES_H2D.load(Ordering::Relaxed),
        bytes_d2h: BYTES_D2H.load(Ordering::Relaxed),
        gemm_mixed: GEMM_MIXED.load(Ordering::Relaxed),
        mixed_panels: MIXED_PANELS.load(Ordering::Relaxed),
        mixed_fallback_f64: MIXED_FALLBACK_F64.load(Ordering::Relaxed),
        resident_uploads: RESIDENT_UPLOADS.load(Ordering::Relaxed),
        dfk_device_builds: DFK_DEVICE_BUILDS.load(Ordering::Relaxed),
        dfk_declined: DFK_DECLINED.load(Ordering::Relaxed),
        dfj_device_builds: DFJ_DEVICE_BUILDS.load(Ordering::Relaxed),
        dfj_declined: DFJ_DECLINED.load(Ordering::Relaxed),
    }
}
