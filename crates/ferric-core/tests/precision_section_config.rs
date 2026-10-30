//! The `[precision]` section (`GpuSettingsExplicit::precision_section`): the
//! device-independent home of the precision knobs. `[gpu] precision` /
//! `mixed_kernels` are deprecated aliases: either spelling works, both equal
//! is accepted, both different is an error; with `[gpu] mode = "off"` mixed
//! precision is accepted only for kernels with a CPU mixed path. No device is
//! touched by any test here.
use ferric_core::gpu::config::{GpuSettings, GpuSettingsExplicit, PrecisionExplicit};
use ferric_core::gpu::precision::{MixedKernel, MixedKernelSet, Precision, PRECISION_DEFAULT};
use ferric_core::gpu::{GpuMode, GpuPreset};

fn no_env(_: &str) -> Option<String> {
    None
}

fn section(mode: Option<Precision>, kernels: Option<MixedKernelSet>) -> PrecisionExplicit {
    PrecisionExplicit {
        mode,
        mixed_kernels: kernels,
    }
}

const RIMP2: MixedKernelSet = MixedKernelSet::EMPTY.with(MixedKernel::RiMp2Energy);
const COSX: MixedKernelSet = MixedKernelSet::EMPTY.with(MixedKernel::CosxKern);

#[test]
fn the_section_and_its_alias_resolve_to_the_same_settings() {
    let via_section = GpuSettingsExplicit {
        mode: Some(GpuMode::Auto),
        precision_section: section(Some(Precision::Mixed), Some(RIMP2)),
        ..Default::default()
    };
    let via_alias = GpuSettingsExplicit {
        mode: Some(GpuMode::Auto),
        precision: Some(Precision::Mixed),
        mixed_kernels: Some(RIMP2),
        ..Default::default()
    };
    let both_equal = GpuSettingsExplicit {
        precision: Some(Precision::Mixed),
        mixed_kernels: Some(RIMP2),
        ..via_section
    };
    let (a, audit_a) = GpuSettings::resolve(via_section, no_env).unwrap();
    let (b, audit_b) = GpuSettings::resolve(via_alias, no_env).unwrap();
    let (c, audit_c) = GpuSettings::resolve(both_equal, no_env).unwrap();
    assert_eq!(a, b);
    assert_eq!(a, c);
    assert!(a.mixed_allows(MixedKernel::RiMp2Energy));
    // Only the alias spelling prints the deprecation note.
    let note = |audit: &[String]| audit.iter().any(|l| l.contains("deprecated"));
    assert!(!note(&audit_a), "{audit_a:?}");
    assert!(note(&audit_b), "{audit_b:?}");
    assert!(note(&audit_c), "{audit_c:?}");
}

#[test]
fn an_f64_section_is_the_default_settings() {
    let (default, _) = GpuSettings::resolve(GpuSettingsExplicit::default(), no_env).unwrap();
    let (f64_section, _) = GpuSettings::resolve(
        GpuSettingsExplicit {
            precision_section: section(Some(Precision::F64), None),
            ..Default::default()
        },
        no_env,
    )
    .unwrap();
    assert_eq!(default, f64_section);
    assert_eq!(default.precision, PRECISION_DEFAULT);
    for k in MixedKernel::ALL {
        assert!(!default.mixed_allows(k), "{}", k.name());
    }
}

#[test]
fn a_conflict_between_the_section_and_its_alias_is_an_error_naming_both() {
    let err = GpuSettings::resolve(
        GpuSettingsExplicit {
            mode: Some(GpuMode::Auto),
            precision: Some(Precision::F64),
            precision_section: section(Some(Precision::Mixed), None),
            ..Default::default()
        },
        no_env,
    )
    .unwrap_err();
    assert_eq!(
        err,
        "[precision] mode = mixed and [gpu] precision = f64 disagree ([gpu] precision is a \
         deprecated alias of [precision] mode); set only [precision] mode"
    );
    let err = GpuSettings::resolve(
        GpuSettingsExplicit {
            mode: Some(GpuMode::Auto),
            mixed_kernels: Some(RIMP2.with(MixedKernel::XcVxc)),
            precision_section: section(Some(Precision::Mixed), Some(RIMP2)),
            ..Default::default()
        },
        no_env,
    )
    .unwrap_err();
    assert_eq!(
        err,
        "[precision] mixed_kernels = rimp2-energy and [gpu] mixed_kernels = \
         rimp2-energy,xc-vxc disagree ([gpu] mixed_kernels is a deprecated alias of \
         [precision] mixed_kernels); set only [precision] mixed_kernels"
    );
    // The load-time check is the same rule.
    let ex = GpuSettingsExplicit {
        precision: Some(Precision::Mixed),
        precision_section: section(Some(Precision::F64), None),
        ..Default::default()
    };
    assert!(ex
        .check_precision_aliases()
        .unwrap_err()
        .contains("disagree"));
    let agree = GpuSettingsExplicit {
        precision: Some(Precision::F64),
        precision_section: section(Some(Precision::F64), None),
        ..Default::default()
    };
    agree.check_precision_aliases().unwrap();
}

#[test]
fn the_section_against_a_preset_names_the_section_key() {
    let err = GpuSettings::resolve(
        GpuSettingsExplicit {
            preset: Some(GpuPreset::Mixed),
            precision_section: section(Some(Precision::F64), None),
            ..Default::default()
        },
        no_env,
    )
    .unwrap_err();
    assert!(
        err.contains("implies precision = mixed but FERRIC_GPU_PRECISION / [precision] mode = f64"),
        "{err}"
    );
}

#[test]
fn section_messages_name_the_section_keys() {
    let err = GpuSettings::resolve(
        GpuSettingsExplicit {
            mode: Some(GpuMode::Auto),
            precision_section: section(None, Some(RIMP2)),
            ..Default::default()
        },
        no_env,
    )
    .unwrap_err();
    assert!(
        err.contains("[precision] mixed_kernels = rimp2-energy given but the precision is f64")
            && err.contains("set [precision] mode = \"mixed\""),
        "{err}"
    );
    let err = GpuSettings::resolve(
        GpuSettingsExplicit {
            mode: Some(GpuMode::Auto),
            precision_section: section(Some(Precision::Mixed), Some(COSX)),
            ..Default::default()
        },
        no_env,
    )
    .unwrap_err();
    assert!(
        err.starts_with("[precision] mixed_kernels: cosx-kern not shipped yet"),
        "{err}"
    );
}

/// CPU-only mixed precision: `[gpu] mode = "off"` with `mode = "mixed"`.
/// This build admits no CPU mixed kernel (cosx-kern, the only one with a CPU
/// mixed path, is named but not shipped), so the user-reachable request is
/// refused with the reason, and acceptance is shown with an injected shipped
/// set that contains cosx-kern (the config path only; the kernel itself is
/// the `f32_route` test seam until Task D6 wires it).
#[test]
fn cpu_only_mixed_needs_every_kernel_to_have_a_cpu_path() {
    // Default kernel set (rimp2-energy, a device kernel) with the GPU off.
    let err = GpuSettings::resolve(
        GpuSettingsExplicit {
            precision_section: section(Some(Precision::Mixed), None),
            ..Default::default()
        },
        no_env,
    )
    .unwrap_err();
    assert_eq!(
        err,
        "[precision] mode = mixed requires mode = auto or on (FERRIC_GPU / [gpu] mode is \
         off): rimp2-energy runs only on a device; CPU mixed-precision kernels in this \
         build: none"
    );
    // The historical alias message keeps its prefix.
    let err = GpuSettings::resolve(
        GpuSettingsExplicit {
            precision: Some(Precision::Mixed),
            ..Default::default()
        },
        no_env,
    )
    .unwrap_err();
    assert!(
        err.starts_with("[gpu] precision = mixed requires mode = auto or on"),
        "{err}"
    );
    // cosx-kern named with the GPU off: refused in this build (not shipped).
    let cpu_request = GpuSettingsExplicit {
        mode: Some(GpuMode::Off),
        precision_section: section(Some(Precision::Mixed), Some(COSX)),
        ..Default::default()
    };
    let err = GpuSettings::resolve(cpu_request, no_env).unwrap_err();
    assert!(err.contains("cosx-kern not shipped yet"), "{err}");
    // ...accepted by a build that ships it: CPU-only mixed is expressible.
    let shipped = RIMP2.with(MixedKernel::CosxKern);
    let (s, _) =
        GpuSettings::resolve_with_default(cpu_request, no_env, PRECISION_DEFAULT, shipped).unwrap();
    assert_eq!(s.mode, GpuMode::Off);
    assert_eq!(s.precision, Precision::Mixed);
    assert!(s.mixed_allows(MixedKernel::CosxKern));
    assert!(!s.mixed_allows(MixedKernel::RiMp2Energy));
    // ...but a device kernel next to it is still refused with the GPU off.
    let err = GpuSettings::resolve_with_default(
        GpuSettingsExplicit {
            precision_section: section(Some(Precision::Mixed), Some(shipped)),
            ..cpu_request
        },
        no_env,
        PRECISION_DEFAULT,
        shipped,
    )
    .unwrap_err();
    assert!(
        err.ends_with("rimp2-energy runs only on a device; CPU mixed-precision kernels in this build: cosx-kern"),
        "{err}"
    );
    // ...and the default (shipped) set with the GPU off is refused too.
    let err = GpuSettings::resolve_with_default(
        GpuSettingsExplicit {
            precision_section: section(Some(Precision::Mixed), None),
            ..Default::default()
        },
        no_env,
        PRECISION_DEFAULT,
        shipped,
    )
    .unwrap_err();
    assert!(err.contains("rimp2-energy runs only on a device"), "{err}");
}

#[test]
fn precision_requests_merge_strictly() {
    let a = section(Some(Precision::Mixed), None);
    let b = section(None, Some(RIMP2));
    assert_eq!(
        a.merge(b, "x").unwrap(),
        section(Some(Precision::Mixed), Some(RIMP2))
    );
    assert_eq!(a.merge(a, "x").unwrap(), a);
    let err = a
        .merge(
            section(Some(Precision::F64), None),
            "the request staged by configure_precision",
        )
        .unwrap_err();
    assert_eq!(
        err,
        "[precision] mode = mixed conflicts with the request staged by configure_precision = \
         f64; set it once"
    );
    assert!(PrecisionExplicit::default().is_empty());
    assert!(!a.is_empty());
}
