//! GPU settings types. Task 0.2 adds the `ConfigVar`s and resolution.
use std::fmt;
use std::str::FromStr;

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
            "off" => Ok(GpuMode::Off),
            "auto" => Ok(GpuMode::Auto),
            "on" => Ok(GpuMode::On),
            other => Err(format!("invalid GPU mode '{other}' (expected off|auto|on)")),
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

/// Resolved GPU settings.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GpuSettings {
    pub mode: GpuMode,
    pub device: usize,
    pub memory_gb: Option<f64>,
    pub min_flops: Option<usize>,
}
