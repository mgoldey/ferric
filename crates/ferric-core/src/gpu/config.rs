//! GPU knobs, all through `ConfigVar` (precedence TOML/kwarg > env > default).
//! Names follow `FERRIC_<AREA>_<NAME>`. Every knob here changes which GEMMs run
//! where, and therefore the last digits of a result, so each has a TOML field
//! (`[gpu]` in the CLI) -- only `FERRIC_GPU_TRACE` is env-only (debug print).
use std::fmt;
use std::str::FromStr;

use crate::config::{accept_any, parse_toggle, ConfigVar, Resolved};

/// Whether the GPU backend is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuMode {
    /// Never load CUDA.
    #[default]
    Off,
    /// Use a device when usable, otherwise note it and run on the CPU.
    Auto,
    /// An unusable device is a hard error.
    On,
}

impl FromStr for GpuMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(GpuMode::Auto),
            other => match parse_toggle(other) {
                Ok(true) => Ok(GpuMode::On),
                Ok(false) => Ok(GpuMode::Off),
                Err(_) => Err(format!("invalid GPU mode {s:?} (expected off, auto or on)")),
            },
        }
    }
}

impl fmt::Display for GpuMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            GpuMode::Off => "off",
            GpuMode::Auto => "auto",
            GpuMode::On => "on",
        })
    }
}

/// NOT YET MEASURED. 1 GFLOP is a conservative placeholder that keeps every
/// small GEMM on the CPU in `auto`; it is not derived from this box's
/// CPU-vs-device crossover. The derivation is deferred until the box is quiet:
///
/// ```text
/// RAYON_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 ///   cargo run --release -p ferric-benchmarks --features gpu --example gpu_gemm_crossover
/// ```
///
/// Protocol: `/proc/pressure/cpu` `some avg10 <= 0.05` before and after, no
/// competing heavy processes, same binary, CPU (6 BLAS threads) and GPU
/// (including H2D/D2H) arms interleaved, 7 repeats per shape, min and median
/// reported. Rule for the value: the smallest FLOP count (2*m*n*k) such that
/// GPU median <= CPU median at every larger measured shape, rounded up to a
/// power of two; if the GPU never wins, `usize::MAX / 2` (never offload).
pub const FERRIC_GPU_MIN_FLOPS_DEFAULT: usize = 1 << 30;

pub static GPU_MODE: ConfigVar<GpuMode> = ConfigVar {
    env_name: "FERRIC_GPU",
    default: GpuMode::Off,
    parse: |s| s.parse::<GpuMode>(),
    validate: accept_any,
};

pub static GPU_DEVICE: ConfigVar<usize> = ConfigVar {
    env_name: "FERRIC_GPU_DEVICE",
    default: 0,
    parse: |s| {
        s.trim()
            .parse::<usize>()
            .map_err(|e| format!("invalid device ordinal {s:?}: {e}"))
    },
    validate: accept_any,
};

/// Device-pool capacity in GB (decimal). The default `0.0` means unset (0.8 x
/// free at probe); the default is never validated, so any value a user
/// supplies, including `0`, must be finite and > 0.
pub static GPU_MEM_GB: ConfigVar<f64> = ConfigVar {
    env_name: "FERRIC_GPU_MEM_GB",
    default: 0.0,
    parse: |s| {
        s.trim()
            .parse::<f64>()
            .map_err(|e| format!("invalid GB value {s:?}: {e}"))
    },
    validate: |v| {
        if v.is_finite() && *v > 0.0 {
            Ok(())
        } else {
            Err(format!("must be finite and > 0, got {v}"))
        }
    },
};

pub static GPU_MIN_FLOPS: ConfigVar<usize> = ConfigVar {
    env_name: "FERRIC_GPU_MIN_FLOPS",
    default: FERRIC_GPU_MIN_FLOPS_DEFAULT,
    parse: |s| {
        s.trim()
            .parse::<usize>()
            .map_err(|e| format!("invalid FLOP count {s:?}: {e}"))
    },
    validate: accept_any,
};

static GPU_TRACE: ConfigVar<bool> = ConfigVar {
    env_name: "FERRIC_GPU_TRACE",
    default: false,
    parse: parse_toggle,
    validate: accept_any,
};

/// `FERRIC_GPU_TRACE`: env-only debug print.
pub fn gpu_trace() -> bool {
    GPU_TRACE.toggle()
}

/// The TOML/kwarg side (already typed; `None` = not given).
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuSettingsExplicit {
    pub mode: Option<GpuMode>,
    pub device: Option<usize>,
    pub memory_gb: Option<f64>,
    pub min_flops: Option<usize>,
}

/// Resolved GPU settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuSettings {
    pub mode: GpuMode,
    pub device: usize,
    pub memory_gb: Option<f64>,
    pub min_flops: usize,
}

fn tag<T>(r: Result<Resolved<T>, String>, name: &str) -> Result<Resolved<T>, String> {
    r.map_err(|e| format!("{name}: {e}"))
}

impl GpuSettings {
    /// Resolve every knob (TOML/kwarg > env via `get` > default). Returns the
    /// settings and one audit line per knob.
    pub fn resolve(
        explicit: GpuSettingsExplicit,
        get: impl Fn(&str) -> Option<String>,
    ) -> Result<(GpuSettings, Vec<String>), String> {
        let mode = tag(GPU_MODE.resolve(explicit.mode, &get), "FERRIC_GPU")?;
        let device = tag(
            GPU_DEVICE.resolve(explicit.device, &get),
            "FERRIC_GPU_DEVICE",
        )?;
        let mem = tag(
            GPU_MEM_GB.resolve(explicit.memory_gb, &get),
            "FERRIC_GPU_MEM_GB",
        )?;
        let min_flops = tag(
            GPU_MIN_FLOPS.resolve(explicit.min_flops, &get),
            "FERRIC_GPU_MIN_FLOPS",
        )?;
        let audit = vec![
            mode.audit_line(),
            device.audit_line(),
            mem.audit_line(),
            min_flops.audit_line(),
        ];
        Ok((
            GpuSettings {
                mode: mode.value,
                device: device.value,
                memory_gb: (mem.value > 0.0).then_some(mem.value),
                min_flops: min_flops.value,
            },
            audit,
        ))
    }
}
