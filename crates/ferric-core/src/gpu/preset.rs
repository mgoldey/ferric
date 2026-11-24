//! `[gpu] preset` / `FERRIC_GPU_PRESET`: one word that sets `mode` and
//! `precision` together. A mixed preset implies `mixed_kernels = SHIPPED`, so
//! a preset can never name a kernel without a shipped budget row. The
//! fine-grained knobs stay as overrides: one that AGREES with the preset is
//! kept, one that DISAGREES is a refusal naming both keys (see
//! `GpuSettings::resolve_with_default`).
use std::fmt;
use std::str::FromStr;

use super::config::GpuMode;
use super::precision::{MixedKernelSet, Precision};
use crate::config::{accept_any, ConfigVar};

/// A `[gpu] preset` / `FERRIC_GPU_PRESET` value: one word that sets `mode` and
/// `precision` together (see [`GpuPreset::implied`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuPreset {
    /// No device, f64 (the default; identical to no `[gpu]` key at all).
    #[default]
    Off,
    /// Device in f64 when one is usable, else the CPU with a notice.
    Auto,
    /// Device in f64; an unusable device is an error.
    On,
    /// Device, mixed precision for every kernel this build ships a row for.
    Mixed,
    /// `Mixed`, degrading to the CPU in f64 with a notice.
    AutoMixed,
}

/// What a preset implies for the knobs it expands into. `mixed_kernels` is
/// `None` for the f64 presets (the knob keeps its own default).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresetImplied {
    /// The `[gpu] mode` the preset sets.
    pub mode: GpuMode,
    /// The precision the preset sets.
    pub precision: Precision,
    /// The mixed-kernel set the preset sets: the shipped set for the mixed
    /// presets, `None` for the f64 presets.
    pub mixed_kernels: Option<MixedKernelSet>,
}

impl GpuPreset {
    /// Every preset, in declaration order.
    pub const ALL: [GpuPreset; 5] = [
        GpuPreset::Off,
        GpuPreset::Auto,
        GpuPreset::On,
        GpuPreset::Mixed,
        GpuPreset::AutoMixed,
    ];

    /// The TOML/env spelling (`off`, `auto`, `on`, `mixed`, `auto-mixed`).
    pub const fn name(self) -> &'static str {
        match self {
            GpuPreset::Off => "off",
            GpuPreset::Auto => "auto",
            GpuPreset::On => "on",
            GpuPreset::Mixed => "mixed",
            GpuPreset::AutoMixed => "auto-mixed",
        }
    }

    /// The knob values this preset expands into. `shipped` is injected (not read from the constant) so the invariant
    /// "a preset names only shipped kernels" holds by construction in tests
    /// that drive an empty shipped set.
    pub const fn implied(self, shipped: MixedKernelSet) -> PresetImplied {
        match self {
            GpuPreset::Off => PresetImplied {
                mode: GpuMode::Off,
                precision: Precision::F64,
                mixed_kernels: None,
            },
            GpuPreset::Auto => PresetImplied {
                mode: GpuMode::Auto,
                precision: Precision::F64,
                mixed_kernels: None,
            },
            GpuPreset::On => PresetImplied {
                mode: GpuMode::On,
                precision: Precision::F64,
                mixed_kernels: None,
            },
            GpuPreset::Mixed => PresetImplied {
                mode: GpuMode::On,
                precision: Precision::Mixed,
                mixed_kernels: Some(shipped),
            },
            GpuPreset::AutoMixed => PresetImplied {
                mode: GpuMode::Auto,
                precision: Precision::Mixed,
                mixed_kernels: Some(shipped),
            },
        }
    }
}

impl FromStr for GpuPreset {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let t = s.trim().to_ascii_lowercase();
        GpuPreset::ALL
            .into_iter()
            .find(|p| p.name() == t)
            .ok_or_else(|| {
                format!("invalid GPU preset {s:?} (expected off, auto, on, mixed or auto-mixed)")
            })
    }
}

impl fmt::Display for GpuPreset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// `[gpu] preset` / `FERRIC_GPU_PRESET`.
pub static GPU_PRESET: ConfigVar<GpuPreset> = ConfigVar {
    env_name: "FERRIC_GPU_PRESET",
    default: GpuPreset::Off,
    parse: |s| s.parse(),
    validate: accept_any,
};
