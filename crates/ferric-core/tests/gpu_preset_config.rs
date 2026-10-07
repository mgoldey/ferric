//! `[gpu] preset` / `FERRIC_GPU_PRESET`: one word that sets mode and precision.
//! Device-free: every test injects its own env lookup.
use ferric_core::config::ConfigSource;
use ferric_core::gpu::{
    GpuMode, GpuPreset, GpuSettings, GpuSettingsExplicit, MixedKernel, MixedKernelSet, Precision,
};

fn lookup<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
    move |k| {
        pairs
            .iter()
            .find(|(n, _)| *n == k)
            .map(|(_, v)| v.to_string())
    }
}
fn none(_: &str) -> Option<String> {
    None
}
fn with_preset(p: GpuPreset) -> GpuSettingsExplicit {
    GpuSettingsExplicit {
        preset: Some(p),
        ..Default::default()
    }
}

#[test]
fn preset_off_resolves_exactly_like_no_preset() {
    let (plain, _) = GpuSettings::resolve(GpuSettingsExplicit::default(), none).unwrap();
    let (off, lines) = GpuSettings::resolve(with_preset(GpuPreset::Off), none).unwrap();
    assert_eq!(off, plain);
    assert_eq!(off.mode, GpuMode::Off);
    assert_eq!(off.precision, Precision::F64);
    assert_eq!(
        lines.last().map(String::as_str),
        Some("FERRIC_GPU_PRESET: off  [source: explicit (config/TOML/kwarg)]")
    );
}

#[test]
fn preset_mixed_turns_on_the_device_in_mixed_with_only_shipped_kernels() {
    let (s, lines) = GpuSettings::resolve(with_preset(GpuPreset::Mixed), none).unwrap();
    assert_eq!(s.mode, GpuMode::On);
    assert_eq!(s.precision, Precision::Mixed);
    assert_eq!(s.mixed_kernels, MixedKernelSet::SHIPPED);
    assert!(
        lines.contains(&"FERRIC_GPU: on  [source: preset]".to_string()),
        "{lines:?}"
    );
    assert!(
        lines.contains(&"FERRIC_GPU_PRECISION: mixed  [source: preset]".to_string()),
        "{lines:?}"
    );
    assert!(
        lines.contains(&format!(
            "FERRIC_GPU_MIXED_KERNELS: {}  [source: preset]",
            MixedKernelSet::SHIPPED
        )),
        "{lines:?}"
    );
    assert_eq!(
        lines.last().map(String::as_str),
        Some("FERRIC_GPU_PRESET: mixed  [source: explicit (config/TOML/kwarg)]")
    );
}

#[test]
fn preset_auto_mixed_degrades_like_auto() {
    let (s, _) = GpuSettings::resolve(with_preset(GpuPreset::AutoMixed), none).unwrap();
    assert_eq!(s.mode, GpuMode::Auto);
    assert_eq!(s.precision, Precision::Mixed);
    assert_eq!(s.mixed_kernels, MixedKernelSet::SHIPPED);
}

#[test]
fn every_preset_names_only_shipped_kernels() {
    for p in GpuPreset::ALL {
        let (s, _) = GpuSettings::resolve(with_preset(p), none).unwrap();
        for k in s.mixed_kernels.iter() {
            assert!(
                MixedKernelSet::SHIPPED.contains(k),
                "{p}: {} is not shipped",
                k.name()
            );
        }
        // A build that ships nothing: mixed presets are refused, the others run.
        let r = GpuSettings::resolve_with_default(
            with_preset(p),
            none,
            Precision::F64,
            MixedKernelSet::EMPTY,
        );
        match p {
            GpuPreset::Mixed | GpuPreset::AutoMixed => {
                let e = r.unwrap_err();
                assert!(
                    e.contains("no mixed-precision kernel is available in this build"),
                    "{p}: {e}"
                );
            }
            _ => {
                let (s, _) = r.unwrap();
                assert_eq!(s.precision, Precision::F64);
            }
        }
    }
}

#[test]
fn preset_and_an_explicit_key_that_disagree_is_a_refusal_naming_both() {
    let e = GpuSettings::resolve(
        GpuSettingsExplicit {
            preset: Some(GpuPreset::Mixed),
            precision: Some(Precision::F64),
            ..Default::default()
        },
        none,
    )
    .unwrap_err();
    assert_eq!(
        e,
        "[gpu] preset = mixed (explicit (config/TOML/kwarg)) implies precision = mixed but \
         FERRIC_GPU_PRECISION / [gpu] precision = f64 (explicit (config/TOML/kwarg)); set one of them"
    );

    let e = GpuSettings::resolve(with_preset(GpuPreset::Off), lookup(&[("FERRIC_GPU", "on")]))
        .unwrap_err();
    assert_eq!(
        e,
        "[gpu] preset = off (explicit (config/TOML/kwarg)) implies mode = off but \
         FERRIC_GPU / [gpu] mode = on (env); set one of them"
    );
}

#[test]
fn an_agreeing_explicit_key_is_not_a_conflict() {
    let (s, lines) = GpuSettings::resolve(
        GpuSettingsExplicit {
            preset: Some(GpuPreset::Mixed),
            mode: Some(GpuMode::On),
            ..Default::default()
        },
        none,
    )
    .unwrap();
    assert_eq!(s.mode, GpuMode::On);
    assert!(
        lines.contains(&"FERRIC_GPU: on  [source: explicit (config/TOML/kwarg)]".to_string()),
        "{lines:?}"
    );
}

#[test]
fn a_narrower_kernel_list_under_a_mixed_preset_is_kept() {
    let only = MixedKernelSet::EMPTY.with(MixedKernel::RiMp2Energy);
    let (s, _) = GpuSettings::resolve(
        GpuSettingsExplicit {
            preset: Some(GpuPreset::Mixed),
            mixed_kernels: Some(only),
            ..Default::default()
        },
        none,
    )
    .unwrap();
    assert_eq!(s.mixed_kernels, only);
}

#[test]
fn toml_preset_beats_env_preset_and_env_preset_beats_default() {
    let (s, _) = GpuSettings::resolve(
        with_preset(GpuPreset::Off),
        lookup(&[("FERRIC_GPU_PRESET", "mixed")]),
    )
    .unwrap();
    assert_eq!(s.mode, GpuMode::Off);
    let (s, lines) = GpuSettings::resolve(
        GpuSettingsExplicit::default(),
        lookup(&[("FERRIC_GPU_PRESET", "on")]),
    )
    .unwrap();
    assert_eq!(s.mode, GpuMode::On);
    assert_eq!(s.precision, Precision::F64);
    assert_eq!(
        lines.last().map(String::as_str),
        Some("FERRIC_GPU_PRESET: on  [source: env]")
    );
}

#[test]
fn unknown_preset_is_refused_with_the_vocabulary() {
    let e = GpuSettings::resolve(
        GpuSettingsExplicit::default(),
        lookup(&[("FERRIC_GPU_PRESET", "fast")]),
    )
    .unwrap_err();
    assert_eq!(
        e,
        "FERRIC_GPU_PRESET: invalid GPU preset \"fast\" (expected off, auto, on, mixed or auto-mixed)"
    );
    assert_eq!(
        "auto-mixed".parse::<GpuPreset>().unwrap(),
        GpuPreset::AutoMixed
    );
    assert_eq!(GpuPreset::AutoMixed.to_string(), "auto-mixed");
    assert_eq!(ConfigSource::Preset.label(), "preset");
}

#[test]
fn the_audit_line_order_is_pinned() {
    for explicit in [
        GpuSettingsExplicit::default(),
        with_preset(GpuPreset::Mixed),
    ] {
        let (_, lines) = GpuSettings::resolve(explicit, none).unwrap();
        assert_eq!(lines.len(), 7, "{lines:?}");
        assert!(lines[0].starts_with("FERRIC_GPU:"), "{lines:?}");
        assert!(
            lines[5].starts_with("FERRIC_GPU_MIXED_KERNELS"),
            "{lines:?}"
        );
    }
    let (_, lines) = GpuSettings::resolve(GpuSettingsExplicit::default(), none).unwrap();
    assert_eq!(lines[6], "FERRIC_GPU_PRESET: off  [source: default]");
}

#[test]
fn an_env_precision_that_disagrees_with_the_preset_is_refused_with_its_source() {
    let e = GpuSettings::resolve(
        with_preset(GpuPreset::Mixed),
        lookup(&[("FERRIC_GPU_PRECISION", "f64")]),
    )
    .unwrap_err();
    assert_eq!(
        e,
        "[gpu] preset = mixed (explicit (config/TOML/kwarg)) implies precision = mixed but \
         FERRIC_GPU_PRECISION / [gpu] precision = f64 (env); set one of them"
    );
}

#[test]
fn an_agreeing_explicit_precision_keeps_the_explicit_source() {
    let (s, lines) = GpuSettings::resolve(
        GpuSettingsExplicit {
            preset: Some(GpuPreset::Mixed),
            precision: Some(Precision::Mixed),
            ..Default::default()
        },
        none,
    )
    .unwrap();
    assert_eq!(s.precision, Precision::Mixed);
    assert_eq!(
        lines[4],
        "FERRIC_GPU_PRECISION: mixed  [source: explicit (config/TOML/kwarg)]"
    );
}

#[test]
fn explicit_mode_off_under_a_mixed_preset_is_refused_naming_both() {
    let e = GpuSettings::resolve(
        GpuSettingsExplicit {
            preset: Some(GpuPreset::Mixed),
            mode: Some(GpuMode::Off),
            ..Default::default()
        },
        none,
    )
    .unwrap_err();
    assert_eq!(
        e,
        "[gpu] preset = mixed (explicit (config/TOML/kwarg)) implies mode = on but \
         FERRIC_GPU / [gpu] mode = off (explicit (config/TOML/kwarg)); set one of them"
    );
}

#[test]
fn an_unshipped_name_in_a_narrower_list_under_mixed_is_refused() {
    let list = MixedKernelSet::EMPTY
        .with(MixedKernel::RiMp2Energy)
        .with(MixedKernel::DfkOcc);
    assert!(!MixedKernelSet::SHIPPED.contains(MixedKernel::DfkOcc));
    let e = GpuSettings::resolve(
        GpuSettingsExplicit {
            preset: Some(GpuPreset::Mixed),
            mixed_kernels: Some(list),
            ..Default::default()
        },
        none,
    )
    .unwrap_err();
    assert_eq!(
        e,
        "[gpu] mixed_kernels: dfk-occ not shipped yet (shipped in this build: rimp2-energy)"
    );
}

#[test]
fn an_env_kernel_list_under_a_preset_is_kept_with_its_source() {
    let (s, lines) = GpuSettings::resolve(
        with_preset(GpuPreset::Mixed),
        lookup(&[("FERRIC_GPU_MIXED_KERNELS", "rimp2-energy")]),
    )
    .unwrap();
    assert_eq!(
        s.mixed_kernels,
        MixedKernelSet::EMPTY.with(MixedKernel::RiMp2Energy)
    );
    assert_eq!(
        lines[5],
        "FERRIC_GPU_MIXED_KERNELS: rimp2-energy  [source: env]"
    );
}

#[test]
fn a_toml_preset_beats_an_env_preset_in_the_audit_line() {
    let (s, lines) = GpuSettings::resolve(
        with_preset(GpuPreset::On),
        lookup(&[("FERRIC_GPU_PRESET", "off")]),
    )
    .unwrap();
    assert_eq!(s.mode, GpuMode::On);
    assert_eq!(
        lines[6],
        "FERRIC_GPU_PRESET: on  [source: explicit (config/TOML/kwarg)]"
    );
}

#[test]
fn kernels_with_an_f64_preset_name_the_preset_as_the_way_out() {
    let e = GpuSettings::resolve(
        GpuSettingsExplicit {
            preset: Some(GpuPreset::On),
            mixed_kernels: Some(MixedKernelSet::SHIPPED),
            ..Default::default()
        },
        none,
    )
    .unwrap_err();
    assert_eq!(
        e,
        "FERRIC_GPU_MIXED_KERNELS / [gpu] mixed_kernels = rimp2-energy given but the precision is f64; \
         set [gpu] precision = \"mixed\" (or use preset = \"mixed\") or drop the list"
    );
}

#[test]
fn a_preset_on_a_build_without_the_gpu_feature_decides_like_the_mode_it_implies() {
    use ferric_core::gpu::{decide, gpu_compiled, GpuStatus};
    if gpu_compiled() {
        return;
    }
    let probe = |_: usize| -> GpuStatus { panic!("no probe on a build without the gpu feature") };
    let (s, _) = GpuSettings::resolve(with_preset(GpuPreset::Mixed), none).unwrap();
    assert_eq!(
        decide(&s, probe).unwrap_err(),
        "[gpu] mode = on but this binary was built without the gpu feature; \
         rebuild with `--features ferric-cli/gpu`"
    );
    let (s, _) = GpuSettings::resolve(with_preset(GpuPreset::AutoMixed), none).unwrap();
    assert_eq!(decide(&s, probe).unwrap(), GpuStatus::NotCompiled);
    let (s, _) = GpuSettings::resolve(with_preset(GpuPreset::Off), none).unwrap();
    assert_eq!(decide(&s, probe).unwrap(), GpuStatus::NotCompiled);
}
