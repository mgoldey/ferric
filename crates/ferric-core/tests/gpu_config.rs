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

// ---- decide(): the on/auto/off decision, probe injected ----

use ferric_core::gpu::{decide, gpu_compiled, probe, GpuStatus};
use std::cell::Cell;

fn with_mode(mode: GpuMode, device: usize) -> GpuSettings {
    GpuSettings {
        mode,
        device,
        memory_gb: None,
        min_flops: FERRIC_GPU_MIN_FLOPS_DEFAULT,
    }
}

fn unavailable(_: usize) -> GpuStatus {
    GpuStatus::Unavailable {
        reason: "no driver".into(),
    }
}

#[test]
fn off_never_calls_the_probe() {
    let calls = Cell::new(0);
    let st = decide(&with_mode(GpuMode::Off, 0), |_| {
        calls.set(calls.get() + 1);
        unavailable(0)
    })
    .unwrap();
    assert_eq!(calls.get(), 0);
    assert_eq!(
        st,
        GpuStatus::Unavailable {
            reason: "mode off".into()
        }
    );
}

#[test]
fn auto_degrades_to_cpu_without_error() {
    let st = decide(&with_mode(GpuMode::Auto, 0), unavailable).unwrap();
    if gpu_compiled() {
        assert_eq!(
            st,
            GpuStatus::Unavailable {
                reason: "no driver".into()
            }
        );
    } else {
        assert_eq!(st, GpuStatus::NotCompiled);
    }
}

#[test]
fn on_without_a_device_names_the_reason() {
    let err = decide(&with_mode(GpuMode::On, 0), unavailable).unwrap_err();
    if gpu_compiled() {
        assert!(
            err.contains("mode = on but no usable device") && err.contains("no driver"),
            "{err}"
        );
    } else {
        assert!(err.contains("built without the gpu feature"), "{err}");
    }
}

#[test]
fn on_with_a_ready_device_passes_the_status_through() {
    if !gpu_compiled() {
        return;
    }
    let info = ferric_core::gpu::GpuInfo {
        ordinal: 3,
        name: "fake".into(),
        cc_major: 6,
        cc_minor: 1,
        free_bytes: 1,
        total_bytes: 2,
    };
    let seen = Cell::new(usize::MAX);
    let st = decide(&with_mode(GpuMode::On, 3), |o| {
        seen.set(o);
        GpuStatus::Ready(info.clone())
    })
    .unwrap();
    assert_eq!(seen.get(), 3, "configured ordinal reaches the probe");
    assert_eq!(st, GpuStatus::Ready(info));
}

#[test]
fn a_bad_ordinal_never_panics() {
    match probe(9999) {
        GpuStatus::NotCompiled => assert!(!gpu_compiled()),
        GpuStatus::Unavailable { reason } => assert!(reason.contains("9999"), "{reason}"),
        GpuStatus::Ready(i) => panic!("ordinal 9999 cannot be ready: {i:?}"),
    }
    let r = decide(&with_mode(GpuMode::Auto, 9999), probe);
    assert!(r.is_ok(), "auto never errors: {r:?}");
}
