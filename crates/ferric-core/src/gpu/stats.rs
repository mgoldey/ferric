//! Process-wide counters for the GPU path. Tests assert deltas on these rather
//! than grepping log text, so a path that is "on" but never taken is visible.
use std::sync::atomic::{AtomicU64, Ordering};

use super::precision::MixedKernel;

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
static XC_DEVICE_PASSES: AtomicU64 = AtomicU64::new(0);
static XC_DECLINED: AtomicU64 = AtomicU64::new(0);
static XC_PINNED_FALLBACK: AtomicU64 = AtomicU64::new(0);
static XC_WAVES: AtomicU64 = AtomicU64::new(0);

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

/// One KS-DFT XC grid pass (the density pass of one spin density, or the
/// potential pass of one V matrix) ran its per-batch GEMMs on the device (`h2d`
/// = sub-densities / potential factors uploaded, `d2h` = Φ / V blocks
/// downloaded; they count toward `bytes_h2d` / `bytes_d2h`).
#[doc(hidden)]
pub fn note_xc_pass(h2d: usize, d2h: usize) {
    XC_DEVICE_PASSES.fetch_add(1, Ordering::Relaxed);
    BYTES_H2D.fetch_add(h2d as u64, Ordering::Relaxed);
    BYTES_D2H.fetch_add(d2h as u64, Ordering::Relaxed);
}

/// A screened XC grid that will stay on the CPU for its lifetime. Counting only.
#[doc(hidden)]
pub fn note_xc_declined(reason: &str) {
    let _ = reason; // the dispatcher prints the one-line notice
    XC_DECLINED.fetch_add(1, Ordering::Relaxed);
}

/// A page-locked host staging buffer of the KS XC grid was refused by the
/// driver; that buffer is ordinary heap memory instead. Counting only.
#[doc(hidden)]
pub fn note_xc_pinned_fallback() {
    XC_PINNED_FALLBACK.fetch_add(1, Ordering::Relaxed);
}

/// One wave (one upload, its launches, one download and one sync) of a KS XC
/// device pass. Counting only.
#[doc(hidden)]
pub fn note_xc_wave() {
    XC_WAVES.fetch_add(1, Ordering::Relaxed);
}

/// Mixed was requested and allowed for `kernel`, but the call (or a part of
/// it) ran in f64 for the reason `why`. Counted process-wide
/// (`mixed_fallback_f64`) and per kernel (the precision provenance record,
/// [`crate::gpu::provenance`]). Never silent.
#[doc(hidden)]
pub fn note_mixed_fallback(kernel: MixedKernel, why: MixedFallback) {
    note_mixed_fallbacks(kernel, why, 1);
}

/// [`note_mixed_fallback`] for `n` units at once (a kernel that counts its
/// fallbacks per work unit, e.g. COSX's per-(pair, sub-batch) f32 units).
#[doc(hidden)]
pub fn note_mixed_fallbacks(kernel: MixedKernel, why: MixedFallback, n: u64) {
    MIXED_FALLBACK_F64.fetch_add(n, Ordering::Relaxed);
    KERNEL_FALLBACK[kernel.index()][why as usize].fetch_add(n, Ordering::Relaxed);
}

/// Why a call that was allowed to run a mixed kernel ran (part of) it in f64.
/// The discriminant indexes the per-kernel fallback counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixedFallback {
    /// f64 on the device: the mixed path was unavailable (its kernel would
    /// not load, or a CUDA error on the mixed path).
    DeviceF64 = 0,
    /// An operand held a finite value beyond `f32::MAX`.
    F32Range = 1,
    /// f64 on the CPU: no usable device, pool refusal, a rayon worker, a
    /// layout refusal or a CUDA error moved the whole call to the CPU.
    CpuF64 = 2,
}

impl MixedFallback {
    pub const ALL: [MixedFallback; 3] = [
        MixedFallback::DeviceF64,
        MixedFallback::F32Range,
        MixedFallback::CpuF64,
    ];

    /// The provenance-record spelling.
    pub const fn name(self) -> &'static str {
        match self {
            MixedFallback::DeviceF64 => "device_f64",
            MixedFallback::F32Range => "f32_range",
            MixedFallback::CpuF64 => "cpu_f64",
        }
    }
}

const NKERNEL: usize = MixedKernel::ALL.len();
const NFALLBACK: usize = MixedFallback::ALL.len();

static KERNEL_MIXED: [AtomicU64; NKERNEL] = [const { AtomicU64::new(0) }; NKERNEL];
static KERNEL_FALLBACK: [[AtomicU64; NFALLBACK]; NKERNEL] =
    [const { [const { AtomicU64::new(0) }; NFALLBACK] }; NKERNEL];
static KERNEL_LADDER_BUILDS: [AtomicU64; NKERNEL] = [const { AtomicU64::new(0) }; NKERNEL];
static KERNEL_EXIT_REBUILDS: [AtomicU64; NKERNEL] = [const { AtomicU64::new(0) }; NKERNEL];

/// One call of `kernel` ran (at least partly) in mixed precision: one RI-MP2
/// energy, one XC potential pass, one RI-J or DF-K build, one COSX K build
/// with f32 units. The unit is the kernel's own call, not a GEMM.
#[doc(hidden)]
pub fn note_kernel_mixed(kernel: MixedKernel) {
    KERNEL_MIXED[kernel.index()].fetch_add(1, Ordering::Relaxed);
}

/// One Fock build of an SCF whose precision ladder was consulted for
/// `kernel` while mixed precision was requested (mixed or kept f64 by the
/// ladder; the mixed ones are also counted by [`note_kernel_mixed`]).
#[doc(hidden)]
pub fn note_ladder_build(kernel: MixedKernel) {
    KERNEL_LADDER_BUILDS[kernel.index()].fetch_add(1, Ordering::Relaxed);
}

/// A converged SCF replaced `kernel`'s last mixed contribution by an f64
/// rebuild before returning (the exit rebuild).
#[doc(hidden)]
pub fn note_exit_rebuild(kernel: MixedKernel) {
    KERNEL_EXIT_REBUILDS[kernel.index()].fetch_add(1, Ordering::Relaxed);
}

/// Per-kernel precision counters at one instant (process-wide, monotone).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KernelCounters {
    /// Calls that ran mixed ([`note_kernel_mixed`]).
    pub mixed: u64,
    /// Fallbacks to f64, indexed by `MixedFallback as usize`.
    pub fallback: [u64; NFALLBACK],
    /// Ladder-gated builds while mixed was requested ([`note_ladder_build`]).
    pub ladder_builds: u64,
    /// Exit rebuilds in f64 ([`note_exit_rebuild`]).
    pub exit_rebuilds: u64,
}

impl KernelCounters {
    /// `self - before`, field by field (saturating: the counters only grow).
    pub fn since(&self, before: &KernelCounters) -> KernelCounters {
        let mut fallback = [0u64; NFALLBACK];
        for (i, f) in fallback.iter_mut().enumerate() {
            *f = self.fallback[i].saturating_sub(before.fallback[i]);
        }
        KernelCounters {
            mixed: self.mixed.saturating_sub(before.mixed),
            fallback,
            ladder_builds: self.ladder_builds.saturating_sub(before.ladder_builds),
            exit_rebuilds: self.exit_rebuilds.saturating_sub(before.exit_rebuilds),
        }
    }

    /// Whether any counter is non-zero.
    pub fn any(&self) -> bool {
        self.mixed > 0
            || self.fallback.iter().any(|&f| f > 0)
            || self.ladder_builds > 0
            || self.exit_rebuilds > 0
    }
}

/// The per-kernel counters of every [`MixedKernel`], indexed by
/// [`MixedKernel::index`].
pub fn kernel_counters() -> [KernelCounters; NKERNEL] {
    std::array::from_fn(|i| {
        let mut fallback = [0u64; NFALLBACK];
        for (j, f) in fallback.iter_mut().enumerate() {
            *f = KERNEL_FALLBACK[i][j].load(Ordering::Relaxed);
        }
        KernelCounters {
            mixed: KERNEL_MIXED[i].load(Ordering::Relaxed),
            fallback,
            ladder_builds: KERNEL_LADDER_BUILDS[i].load(Ordering::Relaxed),
            exit_rebuilds: KERNEL_EXIT_REBUILDS[i].load(Ordering::Relaxed),
        }
    })
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
    pub xc_device_passes: u64,
    pub xc_declined: u64,
    pub xc_pinned_fallback: u64,
    pub xc_waves: u64,
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
        xc_device_passes: XC_DEVICE_PASSES.load(Ordering::Relaxed),
        xc_declined: XC_DECLINED.load(Ordering::Relaxed),
        xc_pinned_fallback: XC_PINNED_FALLBACK.load(Ordering::Relaxed),
        xc_waves: XC_WAVES.load(Ordering::Relaxed),
    }
}
