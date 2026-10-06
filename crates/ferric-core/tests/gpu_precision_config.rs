//! The precision knob: default f64 everywhere, mixed only with a device mode and
//! a non-empty shipped allowlist, unknown names refused with the valid list,
//! audit lines name both knobs, and the thread-local scope restores on drop
//! (including unwinding). No device is touched by any test here.
use ferric_core::gpu::config::{GpuSettings, GpuSettingsExplicit};
use ferric_core::gpu::precision::{
    MixedKernel, MixedKernelSet, MixedScope, Precision, GPU_PRECISION, PRECISION_DEFAULT,
};
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
fn default_precision_is_f64_with_an_empty_allowlist_and_audit_names_both() {
    let (s, audit) = GpuSettings::resolve(GpuSettingsExplicit::default(), lookup(&[])).unwrap();
    assert_eq!(s.precision, Precision::F64);
    assert_eq!(s.mixed_kernels, MixedKernelSet::SHIPPED);
    assert!(!s.mixed_allows(MixedKernel::RiMp2Energy));
    assert!(
        audit
            .iter()
            .any(|l| l.starts_with("FERRIC_GPU_PRECISION: f64") && l.contains("[source: default]")),
        "{audit:?}"
    );
    assert!(audit.iter().any(|l| l.starts_with("FERRIC_GPU_MIXED_KERNELS: ") && l.contains("[source: default]")), "{audit:?}");
}

#[test]
fn mixed_requires_a_device_mode() {
    // mode defaults to off: an explicit mixed request cannot be honoured.
    let err = GpuSettings::resolve(
        GpuSettingsExplicit::default(),
        lookup(&[("FERRIC_GPU_PRECISION", "mixed")]),
    )
    .unwrap_err();
    assert!(err.contains("precision") && err.contains("mode"), "{err}");
}

#[test]
fn mixed_with_no_shipped_kernel_is_refused_and_a_shipped_one_resolves() {
    let env = lookup(&[("FERRIC_GPU", "auto"), ("FERRIC_GPU_PRECISION", "mixed")]);
    let r = GpuSettings::resolve(GpuSettingsExplicit::default(), &env);
    if MixedKernelSet::SHIPPED.is_empty() {
        let err = r.unwrap_err();
        assert!(err.contains("no mixed-precision kernel"), "{err}");
    } else {
        let (s, _) = r.unwrap();
        assert_eq!(s.precision, Precision::Mixed);
        assert_eq!(s.mixed_kernels, MixedKernelSet::SHIPPED);
        for k in MixedKernelSet::SHIPPED.iter() {
            assert!(s.mixed_allows(k));
        }
    }
}

#[test]
fn an_explicit_kernel_list_with_f64_precision_is_refused() {
    let err = GpuSettings::resolve(
        GpuSettingsExplicit::default(),
        lookup(&[
            ("FERRIC_GPU", "auto"),
            ("FERRIC_GPU_MIXED_KERNELS", "rimp2-energy"),
        ]),
    )
    .unwrap_err();
    assert!(
        err.contains("FERRIC_GPU_MIXED_KERNELS") && err.contains("f64"),
        "{err}"
    );
}

#[test]
fn unknown_kernel_names_are_refused_with_the_valid_list() {
    for bad in ["diis", "scf-diag", "rimp2_energy", "rimp2-energy,", ""] {
        let err = bad.parse::<MixedKernelSet>().unwrap_err();
        assert!(
            err.contains("rimp2-energy")
                && err.contains("ccsd-amplitudes")
                && err.contains("dfk-occ"),
            "{bad:?}: {err}"
        );
    }
    let ok: MixedKernelSet = " ccsd-amplitudes , rimp2-energy ".parse().unwrap();
    assert!(
        ok.contains(MixedKernel::CcsdAmplitudes)
            && ok.contains(MixedKernel::RiMp2Energy)
            && !ok.contains(MixedKernel::DfkOcc)
    );
    assert_eq!(
        ok.to_string(),
        "rimp2-energy,ccsd-amplitudes",
        "Display is canonical order"
    );
    assert_eq!(MixedKernelSet::EMPTY.to_string(), "none");
}

#[test]
fn toml_beats_env_for_precision_and_kernels() {
    let env = lookup(&[("FERRIC_GPU", "auto"), ("FERRIC_GPU_PRECISION", "f64")]);
    let explicit = GpuSettingsExplicit {
        precision: Some(Precision::Mixed),
        mixed_kernels: Some(MixedKernelSet::EMPTY.with(MixedKernel::CcsdAmplitudes)),
        ..Default::default()
    };
    let (s, _) = GpuSettings::resolve(explicit, &env).unwrap();
    assert_eq!(s.precision, Precision::Mixed);
    assert!(s.mixed_allows(MixedKernel::CcsdAmplitudes));
    assert!(!s.mixed_allows(MixedKernel::RiMp2Energy));
}

#[test]
fn precision_parses_documented_spellings_only() {
    assert_eq!("f64".parse::<Precision>().unwrap(), Precision::F64);
    assert_eq!("MIXED".parse::<Precision>().unwrap(), Precision::Mixed);
    for bad in ["f32", "single", "double", "auto", ""] {
        assert!(bad.parse::<Precision>().is_err(), "{bad:?}");
    }
}

#[test]
fn scope_is_thread_local_nests_and_restores_on_drop_and_on_unwind() {
    assert_eq!(MixedScope::current(), None);
    {
        let _outer = MixedScope::enter(MixedKernel::CcsdAmplitudes);
        assert_eq!(MixedScope::current(), Some(MixedKernel::CcsdAmplitudes));
        {
            let _inner = MixedScope::enter(MixedKernel::RiMp2Energy);
            assert_eq!(MixedScope::current(), Some(MixedKernel::RiMp2Energy));
        }
        assert_eq!(
            MixedScope::current(),
            Some(MixedKernel::CcsdAmplitudes),
            "inner drop restores outer"
        );
        // another thread sees no scope
        std::thread::spawn(|| assert_eq!(MixedScope::current(), None))
            .join()
            .unwrap();
    }
    assert_eq!(MixedScope::current(), None);
    let r = std::panic::catch_unwind(|| {
        let _g = MixedScope::enter(MixedKernel::DfkOcc);
        panic!("unwind through the guard");
    });
    assert!(r.is_err());
    assert_eq!(
        MixedScope::current(),
        None,
        "unwinding must restore the scope"
    );
}

#[test]
fn every_default_resolution_path_agrees_with_precision_default() {
    // The ConfigVar default.
    assert_eq!(GPU_PRECISION.default, PRECISION_DEFAULT);
    // Unset env, nothing explicit: value AND audit source.
    let (s, audit) = GpuSettings::resolve(GpuSettingsExplicit::default(), lookup(&[])).unwrap();
    assert_eq!(s.precision, PRECISION_DEFAULT);
    let line = audit
        .iter()
        .find(|l| l.starts_with("FERRIC_GPU_PRECISION: "))
        .unwrap();
    assert!(
        line.starts_with(&format!("FERRIC_GPU_PRECISION: {PRECISION_DEFAULT}"))
            && line.contains("[source: default]"),
        "{line}"
    );
    // The lazy library fallback (unresolvable env -> GPU off).
    let d = GpuSettings::degraded_off();
    assert_eq!(d.precision, PRECISION_DEFAULT);
    assert_eq!(d.mixed_kernels, MixedKernelSet::SHIPPED);
    assert_eq!(d.mode, GpuMode::Off);
    // An explicit value still beats the default (the constant is a default only).
    let other = if PRECISION_DEFAULT == Precision::F64 {
        Precision::Mixed
    } else {
        Precision::F64
    };
    let ex = GpuSettingsExplicit {
        precision: Some(other),
        ..Default::default()
    };
    let env = lookup(&[("FERRIC_GPU", "auto")]);
    if let Ok((s, _)) = GpuSettings::resolve(ex, &env) {
        assert_eq!(s.precision, other);
    }
}
