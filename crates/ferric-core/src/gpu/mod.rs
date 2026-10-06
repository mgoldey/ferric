//! CUDA backend surface. Everything that does not touch a device is compiled
//! unconditionally so callers (CLI, Python, einsum) can ask "is a GPU in play?"
//! without `cfg` noise. The device-touching parts live behind `feature = "gpu"`.
//!
//! Mode semantics (`FERRIC_GPU` / `[gpu] mode`): `off` never loads CUDA;
//! `auto` uses a device when one is usable and otherwise prints one notice and
//! runs on the CPU; `on` makes an unusable device a hard error. In every mode a
//! single GEMM may still fall back to the CPU (pool full, layout, CUDA error) —
//! that is recorded in [`stats`](mod@crate::gpu::stats), never silent.

pub mod config;
#[cfg(feature = "gpu")]
pub mod device;
#[cfg(feature = "gpu")]
pub mod gemm;
#[cfg(feature = "gpu")]
pub mod pool;
pub mod stats;

pub use config::{GpuMode, GpuSettings};
pub use stats::{stats, GpuStatsSnapshot};

/// Whether this binary was compiled with the `gpu` feature.
pub const fn gpu_compiled() -> bool {
    cfg!(feature = "gpu")
}

/// Static facts about a usable device, captured at probe time.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuInfo {
    pub ordinal: usize,
    pub name: String,
    pub cc_major: i32,
    pub cc_minor: i32,
    pub free_bytes: usize,
    pub total_bytes: usize,
}

/// Outcome of asking for a device.
#[derive(Debug, Clone, PartialEq)]
pub enum GpuStatus {
    /// Built without `--features gpu`.
    NotCompiled,
    /// Built with the feature, but no usable device (reason is user-facing).
    Unavailable { reason: String },
    /// A device is initialised and its cuBLAS handle exists.
    Ready(GpuInfo),
}

/// Probe device `ordinal`. Never panics: a missing driver library, a failed
/// `cuInit`, or a bad ordinal all become [`GpuStatus::Unavailable`].
pub fn probe(ordinal: usize) -> GpuStatus {
    #[cfg(feature = "gpu")]
    {
        match device::device(ordinal) {
            Ok(d) => GpuStatus::Ready(d.info.clone()),
            Err(e) => GpuStatus::Unavailable {
                // Name the ordinal in every variant: library-absent errors
                // (no driver) carry no ordinal of their own.
                reason: format!("GPU {ordinal}: {e}"),
            },
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = ordinal;
        GpuStatus::NotCompiled
    }
}
