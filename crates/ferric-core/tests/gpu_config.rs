use ferric_core::gpu::config::{GpuSettings, GpuSettingsExplicit, FERRIC_GPU_MIN_FLOPS_DEFAULT};
use ferric_core::gpu::GpuMode;
use std::collections::HashMap;

fn lookup(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
    let m: HashMap<String, String> = pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    move |k| m.get(k).cloned()
}

#[test]
fn default_mode_is_off_and_audit_names_the_source() {
    let (s, audit) = GpuSettings::resolve(GpuSettingsExplicit::default(), lookup(&[])).unwrap();
    assert_eq!(s.mode, GpuMode::Off);
    assert_eq!(s.device, 0);
    assert_eq!(s.min_flops, FERRIC_GPU_MIN_FLOPS_DEFAULT);
    assert!(
        audit
            .iter()
            .any(|l| l.starts_with("FERRIC_GPU: off") && l.contains("[source: default]")),
        "{audit:?}"
    );
}

#[test]
fn toml_beats_env_beats_default() {
    let env = lookup(&[("FERRIC_GPU", "auto"), ("FERRIC_GPU_DEVICE", "1")]);
    let (s, _) = GpuSettings::resolve(
        GpuSettingsExplicit {
            mode: Some(GpuMode::On),
            ..Default::default()
        },
        &env,
    )
    .unwrap();
    assert_eq!(s.mode, GpuMode::On, "explicit TOML wins over env");
    assert_eq!(s.device, 1, "env wins over default when TOML is silent");
}

#[test]
fn mode_parses_every_documented_spelling_and_rejects_the_rest() {
    for (txt, want) in [
        ("off", GpuMode::Off),
        ("0", GpuMode::Off),
        ("false", GpuMode::Off),
        ("auto", GpuMode::Auto),
        ("on", GpuMode::On),
        ("1", GpuMode::On),
        ("true", GpuMode::On),
    ] {
        assert_eq!(txt.parse::<GpuMode>().unwrap(), want, "{txt}");
    }
    assert!("maybe".parse::<GpuMode>().is_err());
    let err = GpuSettings::resolve(
        GpuSettingsExplicit::default(),
        lookup(&[("FERRIC_GPU", "maybe")]),
    )
    .unwrap_err();
    assert!(err.contains("FERRIC_GPU") && err.contains("maybe"), "{err}");
}

#[test]
fn memory_gb_must_be_finite_and_positive() {
    for bad in ["0", "-1", "nan", "inf"] {
        let r = GpuSettings::resolve(
            GpuSettingsExplicit::default(),
            lookup(&[("FERRIC_GPU_MEM_GB", bad)]),
        );
        assert!(r.is_err(), "FERRIC_GPU_MEM_GB={bad} must be refused");
    }
    let (s, _) = GpuSettings::resolve(
        GpuSettingsExplicit::default(),
        lookup(&[("FERRIC_GPU_MEM_GB", "2.5")]),
    )
    .unwrap();
    assert_eq!(s.memory_gb, Some(2.5));
}
