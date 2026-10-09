//! The `--gpu <preset>` / `--gpu=<preset>` command-line flag. Pulled out of
//! `run` before its own argument loop sees the rest, so the flag can sit
//! anywhere on the line and `run` stays at its complexity baseline.
use ferric_core::gpu::GpuPreset;

const MISSING: &str = "--gpu requires a preset (off, auto, on, mixed or auto-mixed)";

/// Remove every `--gpu` flag from `args` and return the remaining argv with
/// the parsed preset. Refuses a missing value, a second `--gpu` and an unknown
/// preset name (listing the valid ones).
pub fn extract(args: &[String]) -> Result<(Vec<String>, Option<GpuPreset>), String> {
    let mut rest = Vec::with_capacity(args.len());
    let mut preset: Option<GpuPreset> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let value = if a == "--gpu" {
            // A preset name never starts with `-`.
            match it.next() {
                Some(v) if !v.starts_with('-') => v.clone(),
                _ => return Err(MISSING.into()),
            }
        } else if let Some(v) = a.strip_prefix("--gpu=") {
            v.to_string()
        } else {
            rest.push(a.clone());
            continue;
        };
        if preset.is_some() {
            return Err("--gpu given more than once; give one preset".into());
        }
        preset = Some(value.parse().map_err(|e| format!("--gpu: {e}"))?);
    }
    Ok((rest, preset))
}

/// The `--gpu` entry of `--help`.
pub fn print_help() {
    eprintln!("  --gpu <preset>  off, auto, on, mixed or auto-mixed (also --gpu=<preset>).");
    eprintln!("                  Overrides `[gpu] preset`, `gpu = \"...\"` and FERRIC_GPU_PRESET;");
    eprintln!(
        "                  a [gpu] mode/precision/mixed_kernels key that disagrees is refused."
    );
}

/// [`extract`], or print `error: ...` and exit 2 (a usage error).
pub fn extract_or_exit(args: Vec<String>) -> (Vec<String>, Option<GpuPreset>) {
    extract(&args).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn both_spellings_parse_and_are_removed() {
        let (r, p) = extract(&v(&["ferric", "--gpu", "mixed", "in.toml"])).unwrap();
        assert_eq!((r, p), (v(&["ferric", "in.toml"]), Some(GpuPreset::Mixed)));
        let (r, p) = extract(&v(&["ferric", "in.toml", "--gpu=auto-mixed", "-v"])).unwrap();
        assert_eq!(
            (r, p),
            (v(&["ferric", "in.toml", "-v"]), Some(GpuPreset::AutoMixed))
        );
    }

    #[test]
    fn absent_flag_leaves_argv_untouched() {
        let a = v(&["ferric", "-v", "in.toml"]);
        assert_eq!(extract(&a).unwrap(), (a, None));
    }

    #[test]
    fn refusals() {
        assert_eq!(extract(&v(&["f", "--gpu"])).unwrap_err(), MISSING);
        assert_eq!(
            extract(&v(&["f", "--gpu", "--verbose"])).unwrap_err(),
            MISSING
        );
        assert_eq!(
            extract(&v(&["f", "--gpu", "on", "--gpu=off"])).unwrap_err(),
            "--gpu given more than once; give one preset"
        );
        assert_eq!(
            extract(&v(&["f", "--gpu", "fast"])).unwrap_err(),
            "--gpu: invalid GPU preset \"fast\" (expected off, auto, on, mixed or auto-mixed)"
        );
    }
}
