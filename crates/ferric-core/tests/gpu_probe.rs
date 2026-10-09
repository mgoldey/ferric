//! Device discovery must never panic, must report a reason when unavailable,
//! and must skip cleanly (with a printed line) where there is no CUDA device.
use ferric_core::gpu::{gpu_compiled, probe, GpuStatus};

/// `FERRIC_GPU_TESTS_REQUIRED=1` turns "no device" from a skip into a failure,
/// so a run on the GPU box cannot pass by skipping.
fn require_device() -> bool {
    std::env::var("FERRIC_GPU_TESTS_REQUIRED")
        .map(|v| v == "1")
        .unwrap_or(false)
}

#[test]
fn probe_reports_a_typed_status_and_never_panics() {
    let st = probe(0);
    match (&st, gpu_compiled()) {
        (GpuStatus::NotCompiled, false) => {}
        (GpuStatus::NotCompiled, true) => panic!("compiled with gpu but reported NotCompiled"),
        (GpuStatus::Unavailable { reason }, true) => {
            eprintln!("skipping: no CUDA device ({reason})");
            assert!(
                !reason.is_empty(),
                "an unavailable status must carry a reason"
            );
            assert!(
                !require_device(),
                "FERRIC_GPU_TESTS_REQUIRED=1 but no device: {reason}"
            );
        }
        (GpuStatus::Ready(info), true) => {
            eprintln!(
                "device {}: {} cc {}.{} free {} / total {} B",
                info.ordinal,
                info.name,
                info.cc_major,
                info.cc_minor,
                info.free_bytes,
                info.total_bytes
            );
            assert!(info.total_bytes > 0 && info.free_bytes <= info.total_bytes);
            assert!(
                info.cc_major >= 5,
                "cuBLAS 12 needs sm_50+, got {}.{}",
                info.cc_major,
                info.cc_minor
            );
        }
        (other, false) => {
            panic!("without the gpu feature the only status is NotCompiled, got {other:?}")
        }
    }
}

#[test]
fn unavailable_device_ordinal_is_reported_not_panicked() {
    // Ordinal 99 exists on no machine. Compiled-out builds report NotCompiled;
    // compiled-in builds must turn cudarc's error into Unavailable{reason}.
    let st = probe(99);
    match st {
        GpuStatus::NotCompiled => assert!(!gpu_compiled()),
        GpuStatus::Unavailable { reason } => {
            assert!(
                reason.contains("99"),
                "reason should name the ordinal: {reason}"
            );
        }
        GpuStatus::Ready(info) => panic!("ordinal 99 cannot be ready: {info:?}"),
    }
}
