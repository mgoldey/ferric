//! CUDA backend surface. Everything that does not touch a device is compiled
//! unconditionally so callers (CLI, Python, einsum) can ask "is a GPU in play?"
//! without `cfg` noise. The device-touching parts live behind `feature = "gpu"`.
//!
//! Mode semantics (`FERRIC_GPU` / `[gpu] mode`): `off` never loads CUDA;
//! `auto` uses a device when one is usable and otherwise prints one notice and
//! runs on the CPU; `on` makes an unusable device a hard error. In every mode a
//! single GEMM may still fall back to the CPU (pool full, layout, CUDA error) —
//! that is recorded in [`stats`](mod@crate::gpu::stats), never silent.

#[cfg(feature = "gpu")]
pub mod batched;
pub mod config;
#[cfg(feature = "gpu")]
pub mod device;
#[cfg(feature = "gpu")]
pub mod gemm;
#[cfg(feature = "gpu")]
pub mod mixed;
pub mod mixed_host;
pub mod mixed_rules;
#[cfg(feature = "gpu")]
pub mod pool;
pub mod precision;
pub mod preset;
#[cfg(feature = "gpu")]
pub mod resident;
pub mod stats;

use std::sync::OnceLock;

pub use config::{GpuMode, GpuSettings, GpuSettingsExplicit};
pub use precision::{MixedKernel, MixedKernelSet, MixedScope, Precision};
pub use preset::GpuPreset;
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

/// The on/auto/off decision, pure (no globals, no printing): `probe_fn` is
/// injected. A build without the `gpu` feature is always `NotCompiled` (`on`
/// errors). `off` never calls it; `auto` degrades to `Unavailable` /
/// `NotCompiled` (the caller prints the notice); `on` with no usable device,
/// or on a build without the `gpu` feature, is an `Err` naming the reason.
pub fn decide(
    settings: &GpuSettings,
    probe_fn: impl Fn(usize) -> GpuStatus,
) -> Result<GpuStatus, String> {
    if !gpu_compiled() {
        return match settings.mode {
            GpuMode::On => Err(
                "[gpu] mode = on but this binary was built without the gpu feature; \
                 rebuild with `--features ferric-cli/gpu`"
                    .to_string(),
            ),
            GpuMode::Off | GpuMode::Auto => Ok(GpuStatus::NotCompiled),
        };
    }
    match settings.mode {
        GpuMode::Off => Ok(GpuStatus::Unavailable {
            reason: "mode off".into(),
        }),
        GpuMode::Auto | GpuMode::On => {
            let status = probe_fn(settings.device);
            match (&status, settings.mode) {
                (GpuStatus::Unavailable { reason }, GpuMode::On) => {
                    Err(format!("[gpu] mode = on but no usable device: {reason}"))
                }
                _ => Ok(status),
            }
        }
    }
}

/// Resolve `explicit` against the process env, probe if the mode asks for a
/// device, and install the result process-wide. The CLI calls this once before
/// any method runs. Errors: malformed knob; `mode = on` with no usable device
/// or without the `gpu` feature; a second install with different settings, or
/// an install after `status()`/`settings()` already initialised the lazy state.
pub fn install(explicit: GpuSettingsExplicit) -> Result<&'static GpuStatus, String> {
    let (settings, audit) = GpuSettings::resolve(explicit, crate::config::env_lookup)?;
    // A repeated identical install is a no-op: announce only the first one.
    let first = INSTALLED.get().is_none();
    if first {
        for line in &audit {
            eprintln!("[ferric] {line}");
        }
    }
    let status = decide(&settings, probe)?;
    match &status {
        _ if !first => {}
        GpuStatus::NotCompiled if settings.mode == GpuMode::Auto => {
            eprintln!("[ferric] gpu: built without the gpu feature; running on the CPU in f64");
        }
        GpuStatus::Unavailable { reason } if settings.mode == GpuMode::Auto => {
            eprintln!("[ferric] gpu: unavailable ({reason}); running on the CPU in f64");
        }
        _ => {}
    }
    if let (true, GpuStatus::Ready(info)) = (first, &status) {
        eprintln!(
            "[ferric] gpu: device {} {} (cc {}.{}, {:.2} GB free of {:.2} GB), precision {}{}",
            info.ordinal,
            info.name,
            info.cc_major,
            info.cc_minor,
            info.free_bytes as f64 / 1e9,
            info.total_bytes as f64 / 1e9,
            settings.precision,
            if settings.precision == Precision::Mixed {
                format!(", mixed kernels {}", settings.mixed_kernels)
            } else {
                String::new()
            }
        );
    }
    match INSTALLED.set((settings, status)) {
        Ok(()) => {
            let (s, st) = installed();
            if let GpuStatus::Ready(info) = st {
                install_default_pool(s, info, true);
            }
            Ok(st)
        }
        Err(_) => {
            let (stored, st) = installed();
            if *stored == settings {
                Ok(st)
            } else {
                Err(format!(
                    "{SETTINGS_ALREADY_INSTALLED} with different values ({}); install() must \
                     run before status()/settings() and only once",
                    stored.differences(&settings).join("; ")
                ))
            }
        }
    }
}

/// Prefix of the [`install`] error for a second install (or an install after
/// the lazy state was read) with different settings; callers match on it.
pub const SETTINGS_ALREADY_INSTALLED: &str = "[gpu] settings already installed";

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
        let mut malformed = None;
        let settings =
            match GpuSettings::resolve(GpuSettingsExplicit::default(), crate::config::env_lookup) {
                Ok((s, _)) => s,
                Err(e) => {
                    eprintln!("[ferric] gpu: {e}; GPU disabled");
                    malformed = Some(e);
                    GpuSettings::degraded_off()
                }
            };
        let st = match malformed {
            Some(reason) if gpu_compiled() => GpuStatus::Unavailable { reason },
            _ => decide(&settings, probe).unwrap_or_else(|reason| {
                let f64_note = if settings.precision == Precision::Mixed {
                    " in f64"
                } else {
                    ""
                };
                eprintln!("[ferric] gpu: {reason}; running on the CPU{f64_note}");
                GpuStatus::Unavailable { reason }
            }),
        };
        if let GpuStatus::Ready(info) = &st {
            install_default_pool(&settings, info, false);
        }
        (settings, st)
    })
}

#[cfg(feature = "gpu")]
static POOL: OnceLock<pool::DevicePool> = OnceLock::new();

/// Pool ceiling: `memory_gb` if set, otherwise 80% of the free bytes seen at probe.
#[cfg(feature = "gpu")]
fn default_pool_capacity(settings: &GpuSettings, info: &GpuInfo) -> usize {
    settings
        .memory_gb
        .map(|g| (g * 1e9) as usize)
        .unwrap_or((info.free_bytes as f64 * 0.8) as usize)
}

/// Size and install the process-wide pool (first install wins); optionally announce it.
fn install_default_pool(settings: &GpuSettings, info: &GpuInfo, announce: bool) {
    #[cfg(feature = "gpu")]
    {
        let p = pool::DevicePool::with_capacity_bytes(default_pool_capacity(settings, info));
        let gb = p.capacity_bytes() as f64 / 1e9;
        if POOL.set(p).is_ok() && announce {
            eprintln!("[ferric] gpu: device pool {gb:.2} GB");
        }
    }
    #[cfg(not(feature = "gpu"))]
    let _ = (settings, info, announce);
}

/// The process-wide device pool; `None` when no device is in play.
#[cfg(feature = "gpu")]
pub fn pool() -> Option<pool::DevicePool> {
    POOL.get().cloned()
}
