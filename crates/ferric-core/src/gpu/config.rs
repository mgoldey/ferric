//! GPU knobs, all through `ConfigVar` (precedence TOML/kwarg > env > default).
//! Names follow `FERRIC_<AREA>_<NAME>`. Every knob here changes which GEMMs run
//! where, and therefore the last digits of a result, so each has a TOML field
//! (`[gpu]` in the CLI) -- only `FERRIC_GPU_TRACE` is env-only (debug print).
use std::fmt;
use std::str::FromStr;

use crate::config::{accept_any, parse_toggle, ConfigSource, ConfigVar, Resolved};

use super::precision::{
    MixedKernel, MixedKernelSet, Precision, GPU_MIXED_KERNELS, GPU_PRECISION, PRECISION_DEFAULT,
};
use super::preset::{GpuPreset, GPU_PRESET};

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

/// Smallest `2*m*n*k` the f64 `einsum!` GEMM offload (`try_device_gemm` in
/// `ferric-tensors`) sends to the device in `auto`/`on`. Derived from the
/// quiet-box CPU-vs-device crossover:
///
/// ```text
/// RAYON_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 \
///   cargo run --release -p ferric-benchmarks --features gpu --example gpu_gemm_crossover
/// ```
///
/// Protocol: `/proc/pressure/cpu` `some avg10 <= 0.05` before and after (the
/// harness prints "NOT QUOTABLE: box contested" otherwise), no competing heavy
/// processes, same binary, CPU (6 BLAS threads) and GPU (including H2D/D2H)
/// arms interleaved, 7 repeats per shape, min and median reported (ms; ratios
/// are gpu/cpu, so above 1 the CPU is faster). Rule for the value: the
/// smallest FLOP count (2*m*n*k) such that GPU median <= CPU median at every
/// larger measured shape, rounded up to a power of two; if the GPU never wins,
/// `usize::MAX / 2` (never offload).
///
/// Measured (GTX 1080, CPU pressure `some avg10` 0.00 before and 0.00 after):
///
/// ```text
///      m      k      n          flops   cpu_min   cpu_med   gpu_min   gpu_med  min/min  med/med
///    128    128    128        4194304     0.055     0.056     0.135     0.142     2.47     2.52
///    256    256    256       33554432     0.280     0.287     0.524     0.543     1.87     1.90
///    384    384    384      113246208     0.803     0.818     1.403     1.429     1.75     1.75
///    512    512    512      268435456     2.185     2.599     2.651     2.676     1.21     1.03
///    768    768    768      905969664     3.858     3.955     6.823     6.831     1.77     1.73
///   1024   1024   1024     2147483648    10.104    10.226    14.282    15.734     1.41     1.54
///   1536   1536   1536     7247757312    28.799    28.959    39.136    39.150     1.36     1.35
///   2048   2048   2048    17179869184    69.625    70.008    80.851    81.258     1.16     1.16
///   3072   3072   3072    57982058496   218.866   235.760   252.440   252.578     1.15     1.07
///   4096   4096   4096   137438953472   552.146   566.473   577.170   577.285     1.05     1.02
///   6144   6144   6144   463856467968  1897.142  2029.042  1859.190  1878.231     0.98     0.93
///    414    558    414      191277936     1.505     1.883     1.905     1.919     1.27     1.02
///   3726    414    414     1277242992     6.466     6.746    10.529    10.600     1.63     1.57
///    414   3726    414     1277242992     7.045     7.091    10.935    10.963     1.55     1.55
///    393    912   5895     4225724640    20.068    20.257    30.836    30.897     1.54     1.53
///   1600    128   1600      655360000     3.550     3.638     6.440     6.454     1.81     1.77
/// ```
///
/// The rule gives 549755813888 (2^39), the rounded-up value, above the largest
/// measured shape (6144^3 = 4.6e11 FLOP). The device does not clearly beat 6 CPU
/// cores on a single f64 GEMM at any measured shape up to 6144^3: the best case
/// is a 2-8% median win at 6144^3 in two of three runs, a loss in the contested
/// run, and at 4096^3 and below the CPU won. GEMMs of 2^39 FLOP or more go to
/// the device by this default, and that region is unmeasured.
///
/// Scope: only the f64 `einsum!` GEMM offload reads this threshold. The
/// device RI-MP2 energy (resident f64 and mixed precision, `rimp2_gpu.rs`)
/// does not consult `min_flops`; it is gated by the mode, the device status,
/// not running inside a rayon worker, the pool and, for the mixed lane,
/// `precision`/`mixed_kernels`.
pub const FERRIC_GPU_MIN_FLOPS_DEFAULT: usize = 1usize << 39;

/// The crossover rule documented on [`FERRIC_GPU_MIN_FLOPS_DEFAULT`], as a pure
/// function of measured `(flops, cpu_median_s, gpu_median_s)` rows (any order).
/// Rows are sorted by flops; the threshold is the smallest flops `f` such that
/// the GPU wins (`gpu <= cpu`) at `f` and at EVERY larger measured shape,
/// rounded up to a power of two. A win followed by a later loss therefore does
/// not lower the threshold. Never wins at the largest shape (or empty table):
/// `usize::MAX / 2`. Wins everywhere: the smallest measured flops (rounded up).
pub fn derive_min_flops(table: &[(usize, f64, f64)]) -> usize {
    let mut rows = table.to_vec();
    rows.sort_by_key(|r| r.0);
    let mut threshold = None;
    for &(flops, cpu, gpu) in rows.iter().rev() {
        if gpu <= cpu {
            threshold = Some(flops);
        } else {
            break;
        }
    }
    match threshold {
        Some(f) => f.next_power_of_two(),
        None => usize::MAX / 2,
    }
}

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
    /// A `--gpu <preset>` command-line flag: outranks `preset` (TOML) and the
    /// env var, and is labelled `command line` in the audit lines.
    pub cli_preset: Option<GpuPreset>,
    pub preset: Option<GpuPreset>,
    pub mode: Option<GpuMode>,
    pub device: Option<usize>,
    pub memory_gb: Option<f64>,
    pub min_flops: Option<usize>,
    pub precision: Option<Precision>,
    pub mixed_kernels: Option<MixedKernelSet>,
}

/// Resolved GPU settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuSettings {
    pub mode: GpuMode,
    pub device: usize,
    pub memory_gb: Option<f64>,
    pub min_flops: usize,
    pub precision: Precision,
    pub mixed_kernels: MixedKernelSet,
}

fn tag<T>(r: Result<Resolved<T>, String>, name: &str) -> Result<Resolved<T>, String> {
    r.map_err(|e| format!("{name}: {e}"))
}

/// Source label of a preset given as `ferric --gpu <preset>`.
const COMMAND_LINE_LABEL: &str = "command line";

/// A knob a preset expands into: a default takes the preset's value (source
/// `preset`), a given value that agrees is kept, one that disagrees is a
/// refusal naming both keys and both sources. `shown` is the knob's env /
/// TOML spelling, `key` its TOML key.
fn merge_preset<T: PartialEq + fmt::Display + Copy>(
    r: &mut Resolved<T>,
    implied: T,
    preset: (GpuPreset, &str),
    shown: &str,
    key: &str,
) -> Result<(), String> {
    if r.source == ConfigSource::Default {
        r.value = implied;
        r.source = ConfigSource::Preset;
    } else if r.value != implied {
        return Err(format!(
            "[gpu] preset = {} ({}) implies {key} = {implied} but {shown} = {} ({}); set one of them",
            preset.0,
            preset.1,
            r.value,
            r.source.label(),
        ));
    }
    Ok(())
}

impl GpuSettings {
    /// One `key: installed X, requested Y` entry per field that differs.
    pub fn differences(&self, requested: &GpuSettings) -> Vec<String> {
        let mut out = Vec::new();
        let mut note = |key: &str, a: String, b: String| {
            if a != b {
                out.push(format!("{key}: installed {a}, requested {b}"));
            }
        };
        note("mode", self.mode.to_string(), requested.mode.to_string());
        note(
            "device",
            self.device.to_string(),
            requested.device.to_string(),
        );
        note(
            "memory_gb",
            format!("{:?}", self.memory_gb),
            format!("{:?}", requested.memory_gb),
        );
        note(
            "min_flops",
            self.min_flops.to_string(),
            requested.min_flops.to_string(),
        );
        note(
            "precision",
            self.precision.to_string(),
            requested.precision.to_string(),
        );
        note(
            "mixed_kernels",
            self.mixed_kernels.to_string(),
            requested.mixed_kernels.to_string(),
        );
        out
    }

    /// `true` only when `precision = mixed` AND `k` is in the allowlist.
    pub fn mixed_allows(&self, k: MixedKernel) -> bool {
        self.precision == Precision::Mixed && self.mixed_kernels.contains(k)
    }

    /// The library fallback when the env knobs cannot be resolved: GPU off and
    /// f64 by definition (so a future change of `PRECISION_DEFAULT` cannot
    /// leave an off-mode fallback labelled mixed).
    pub fn degraded_off() -> GpuSettings {
        GpuSettings {
            mode: GpuMode::Off,
            device: 0,
            memory_gb: None,
            min_flops: FERRIC_GPU_MIN_FLOPS_DEFAULT,
            precision: Precision::F64,
            mixed_kernels: MixedKernelSet::EMPTY,
        }
    }

    /// Resolve every knob (TOML/kwarg > env via `get` > default). Returns the
    /// settings and one audit line per knob.
    pub fn resolve(
        explicit: GpuSettingsExplicit,
        get: impl Fn(&str) -> Option<String>,
    ) -> Result<(GpuSettings, Vec<String>), String> {
        Self::resolve_with_default(explicit, get, PRECISION_DEFAULT, MixedKernelSet::SHIPPED)
    }

    /// [`resolve`](Self::resolve) with the default precision and the shipped
    /// kernel set injected, so tests can drive a `Mixed` default or a non-empty
    /// shipped set without editing the constants. Not a user-facing knob.
    #[doc(hidden)]
    pub fn resolve_with_default(
        explicit: GpuSettingsExplicit,
        get: impl Fn(&str) -> Option<String>,
        default_precision: Precision,
        shipped: MixedKernelSet,
    ) -> Result<(GpuSettings, Vec<String>), String> {
        let preset = tag(
            GPU_PRESET.resolve(explicit.cli_preset.or(explicit.preset), &get),
            "FERRIC_GPU_PRESET",
        )?;
        // The flag resolves as an explicit value; only its label differs.
        let preset_label = if explicit.cli_preset.is_some() {
            COMMAND_LINE_LABEL
        } else {
            preset.source.label()
        };
        let preset_given = preset.source != ConfigSource::Default;
        let implied = preset.value.implied(shipped);
        let mut mode = tag(GPU_MODE.resolve(explicit.mode, &get), "FERRIC_GPU")?;
        if preset_given {
            merge_preset(
                &mut mode,
                implied.mode,
                (preset.value, preset_label),
                "FERRIC_GPU / [gpu] mode",
                "mode",
            )?;
        }
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
        let prec_var = ConfigVar {
            env_name: GPU_PRECISION.env_name,
            default: default_precision,
            parse: GPU_PRECISION.parse,
            validate: GPU_PRECISION.validate,
        };
        let kern_var = ConfigVar {
            env_name: GPU_MIXED_KERNELS.env_name,
            default: shipped,
            parse: GPU_MIXED_KERNELS.parse,
            validate: GPU_MIXED_KERNELS.validate,
        };
        let mut precision = tag(
            prec_var.resolve(explicit.precision, &get),
            "FERRIC_GPU_PRECISION",
        )?;
        if preset_given {
            merge_preset(
                &mut precision,
                implied.precision,
                (preset.value, preset_label),
                "FERRIC_GPU_PRECISION / [gpu] precision",
                "precision",
            )?;
        }
        let mut kernels = tag(
            kern_var.resolve(explicit.mixed_kernels, &get),
            "FERRIC_GPU_MIXED_KERNELS",
        )?;
        // A given list is kept (narrowing is allowed; the unshipped-name check
        // below still runs on it); only a default takes the preset's set.
        if let (true, Some(k)) = (preset_given, implied.mixed_kernels) {
            if kernels.source == ConfigSource::Default {
                kernels.value = k;
                kernels.source = ConfigSource::Preset;
            }
        }
        let kernels_given = matches!(kernels.source, ConfigSource::Explicit | ConfigSource::Env);
        let mut precision_line = None;
        if precision.value == Precision::Mixed
            && precision.source == ConfigSource::Default
            && !kernels_given
        {
            // A mixed DEFAULT never errors and never claims mixed it cannot run.
            let why = if mode.value == GpuMode::Off {
                Some("mode off")
            } else if kernels.value.is_empty() {
                Some("no kernel shipped")
            } else {
                None
            };
            if let Some(why) = why {
                precision.value = Precision::F64;
                precision_line = Some(format!(
                    "{}: f64 (default mixed; {why})  [source: {}]",
                    precision.env_name,
                    precision.source.label()
                ));
            }
        }
        match precision.value {
            Precision::F64 if kernels_given => {
                return Err(format!(
                    "FERRIC_GPU_MIXED_KERNELS / [gpu] mixed_kernels = {} given but the precision is f64; \
                     set [gpu] precision = \"mixed\" (or use preset = \"mixed\") or drop the list",
                    kernels.value
                ));
            }
            Precision::Mixed if mode.value == GpuMode::Off => {
                return Err(
                    "[gpu] precision = mixed requires mode = auto or on (FERRIC_GPU / [gpu] mode is off)"
                        .to_string(),
                );
            }
            Precision::Mixed if kernels_given => {
                let missing: Vec<&str> = kernels
                    .value
                    .iter()
                    .filter(|&k| !shipped.contains(k))
                    .map(|k| k.name())
                    .collect();
                if !missing.is_empty() {
                    let have = if shipped.is_empty() {
                        "none are shipped in this build".to_string()
                    } else {
                        format!("shipped in this build: {shipped}")
                    };
                    return Err(format!(
                        "[gpu] mixed_kernels: {} not shipped yet ({have})",
                        missing.join(", ")
                    ));
                }
            }
            Precision::Mixed if kernels.value.is_empty() => {
                return Err(
                    "[gpu] precision = mixed but no mixed-precision kernel is available in this build"
                        .to_string(),
                );
            }
            _ => {}
        }
        // The unset default is 0.0, which would read as "zero GB".
        let mem_line = if mem.value > 0.0 {
            mem.audit_line()
        } else {
            format!(
                "{}: auto (80% of free)  [source: {}]",
                mem.env_name,
                mem.source.label()
            )
        };
        let audit = vec![
            mode.audit_line(),
            device.audit_line(),
            mem_line,
            min_flops.audit_line(),
            precision_line.unwrap_or_else(|| precision.audit_line()),
            kernels.audit_line(),
            format!(
                "{}: {}  [source: {preset_label}]",
                preset.env_name, preset.value
            ),
        ];
        Ok((
            GpuSettings {
                mode: mode.value,
                device: device.value,
                memory_gb: (mem.value > 0.0).then_some(mem.value),
                min_flops: min_flops.value,
                precision: precision.value,
                mixed_kernels: kernels.value,
            },
            audit,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::derive_min_flops;

    #[test]
    fn never_wins_is_never_offload() {
        let t = [(100, 1.0, 2.0), (1000, 1.0, 1.5), (10000, 1.0, 1.1)];
        assert_eq!(derive_min_flops(&t), usize::MAX / 2);
        assert_eq!(derive_min_flops(&[]), usize::MAX / 2);
    }

    #[test]
    fn always_wins_is_smallest_shape_rounded_up() {
        let t = [(1000, 2.0, 1.0), (100, 2.0, 1.0), (10000, 2.0, 1.0)];
        assert_eq!(derive_min_flops(&t), 128);
    }

    #[test]
    fn crossover_in_the_middle() {
        let t = [
            (100, 1.0, 2.0),
            (1000, 1.0, 2.0),
            (5000, 2.0, 1.0),
            (9000, 3.0, 1.0),
        ];
        assert_eq!(derive_min_flops(&t), 8192);
    }

    #[test]
    fn tie_counts_as_a_win() {
        assert_eq!(derive_min_flops(&[(64, 1.0, 1.0)]), 64);
    }

    #[test]
    fn non_monotone_win_then_loss_does_not_lower_threshold() {
        // GPU wins at 200, loses at 3000, wins at 40000: only 40000 qualifies.
        let t = [
            (100, 1.0, 2.0),
            (200, 2.0, 1.0),
            (3000, 1.0, 2.0),
            (40000, 2.0, 1.0),
        ];
        assert_eq!(derive_min_flops(&t), 65536);
        // A loss at the largest shape means never.
        let t = [(200, 2.0, 1.0), (3000, 1.0, 2.0)];
        assert_eq!(derive_min_flops(&t), usize::MAX / 2);
    }
}
