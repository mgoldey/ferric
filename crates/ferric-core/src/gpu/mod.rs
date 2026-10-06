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

use std::sync::OnceLock;

pub use config::{GpuMode, GpuSettings, GpuSettingsExplicit};
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

static INSTALLED: OnceLock<(GpuSettings, GpuStatus)> = OnceLock::new();

/// Resolve `explicit` against the process env, probe if the mode asks for a
/// device, and install the result process-wide. The CLI calls this once before
/// any method runs. Errors: malformed knob; `mode = on` with no usable device;
/// a device mode on a binary built without the `gpu` feature.
pub fn install(explicit: GpuSettingsExplicit) -> Result<&'static GpuStatus, String> {
    let (settings, audit) = GpuSettings::resolve(explicit, crate::config::env_lookup)?;
    for line in &audit {
        eprintln!("[ferric] {line}");
    }
    let status = match settings.mode {
        GpuMode::Off => GpuStatus::Unavailable {
            reason: "mode off".into(),
        },
        GpuMode::Auto | GpuMode::On => {
            if !gpu_compiled() {
                return Err(format!(
                    "[gpu] mode = {} but this binary was built without the gpu feature; \
                     rebuild with `--features ferric-cli/gpu`",
                    settings.mode
                ));
            }
            probe(settings.device)
        }
    };
    if let (GpuMode::On, GpuStatus::Unavailable { reason }) = (settings.mode, &status) {
        return Err(format!("[gpu] mode = on but no usable device: {reason}"));
    }
    if let (GpuMode::Auto, GpuStatus::Unavailable { reason }) = (settings.mode, &status) {
        eprintln!("[ferric] gpu: unavailable ({reason}); running on the CPU");
    }
    if let GpuStatus::Ready(info) = &status {
        eprintln!(
            "[ferric] gpu: device {} {} (cc {}.{}, {:.2} GB free of {:.2} GB)",
            info.ordinal,
            info.name,
            info.cc_major,
            info.cc_minor,
            info.free_bytes as f64 / 1e9,
            info.total_bytes as f64 / 1e9
        );
    }
    let _ = INSTALLED.set((settings, status));
    Ok(&installed().1)
}

/// Library/Python callers that never called [`install`]: resolve from the env
/// on first use (mode defaults to `off`, so this touches no device by default).
/// A malformed env knob degrades to `off` here (the CLI path errors instead).
pub fn status() -> &'static GpuStatus {
    &installed().1
}

/// The installed (or lazily env-resolved) settings.
pub fn settings() -> &'static GpuSettings {
    &installed().0
}

fn installed() -> &'static (GpuSettings, GpuStatus) {
    INSTALLED.get_or_init(|| {
        let settings =
            match GpuSettings::resolve(GpuSettingsExplicit::default(), crate::config::env_lookup) {
                Ok((s, _)) => s,
                Err(e) => {
                    eprintln!("[ferric] gpu: {e}; GPU disabled");
                    GpuSettings {
                        mode: GpuMode::Off,
                        device: 0,
                        memory_gb: None,
                        min_flops: config::FERRIC_GPU_MIN_FLOPS_DEFAULT,
                    }
                }
            };
        let st = match settings.mode {
            GpuMode::Off => GpuStatus::Unavailable {
                reason: "mode off".into(),
            },
            _ => probe(settings.device),
        };
        (settings, st)
    })
}
