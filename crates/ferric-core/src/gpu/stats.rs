//! Process-wide counters for the GPU path. Tests assert deltas on these rather
//! than grepping log text, so a path that is "on" but never taken is visible.
use std::sync::atomic::{AtomicU64, Ordering};

static GEMM_OFFLOADED: AtomicU64 = AtomicU64::new(0);
static GEMM_CPU_INSIDE_WORKER: AtomicU64 = AtomicU64::new(0);
static GEMM_CPU_BELOW_THRESHOLD: AtomicU64 = AtomicU64::new(0);
static GEMM_CPU_POOL_FULL: AtomicU64 = AtomicU64::new(0);
static GEMM_CPU_LAYOUT: AtomicU64 = AtomicU64::new(0);
static GEMM_CPU_CUDA_ERROR: AtomicU64 = AtomicU64::new(0);
static BYTES_H2D: AtomicU64 = AtomicU64::new(0);
static BYTES_D2H: AtomicU64 = AtomicU64::new(0);

/// Why a GEMM ran on the CPU although the GPU mode was not `off`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuReason {
    InsideRayonWorker,
    BelowThreshold,
    PoolFull,
    Layout,
    CudaError,
}

/// Called by the device GEMM (this crate) — public and hidden because
/// `ferric-tensors`' dispatch also records its refusals here.
#[doc(hidden)]
pub fn note_offloaded(h2d: usize, d2h: usize) {
    GEMM_OFFLOADED.fetch_add(1, Ordering::Relaxed);
    BYTES_H2D.fetch_add(h2d as u64, Ordering::Relaxed);
    BYTES_D2H.fetch_add(d2h as u64, Ordering::Relaxed);
}

#[doc(hidden)]
pub fn note_cpu(reason: CpuReason) {
    let c = match reason {
        CpuReason::InsideRayonWorker => &GEMM_CPU_INSIDE_WORKER,
        CpuReason::BelowThreshold => &GEMM_CPU_BELOW_THRESHOLD,
        CpuReason::PoolFull => &GEMM_CPU_POOL_FULL,
        CpuReason::Layout => &GEMM_CPU_LAYOUT,
        CpuReason::CudaError => &GEMM_CPU_CUDA_ERROR,
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
    pub bytes_h2d: u64,
    pub bytes_d2h: u64,
}

pub fn stats() -> GpuStatsSnapshot {
    GpuStatsSnapshot {
        gemm_offloaded: GEMM_OFFLOADED.load(Ordering::Relaxed),
        gemm_cpu_inside_worker: GEMM_CPU_INSIDE_WORKER.load(Ordering::Relaxed),
        gemm_cpu_below_threshold: GEMM_CPU_BELOW_THRESHOLD.load(Ordering::Relaxed),
        gemm_cpu_pool_full: GEMM_CPU_POOL_FULL.load(Ordering::Relaxed),
        gemm_cpu_layout: GEMM_CPU_LAYOUT.load(Ordering::Relaxed),
        gemm_cpu_cuda_error: GEMM_CPU_CUDA_ERROR.load(Ordering::Relaxed),
        bytes_h2d: BYTES_H2D.load(Ordering::Relaxed),
        bytes_d2h: BYTES_D2H.load(Ordering::Relaxed),
    }
}
