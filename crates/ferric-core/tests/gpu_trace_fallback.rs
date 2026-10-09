//! `FERRIC_GPU_TRACE=1` prints one line per CPU fallback, whatever the reason.
//! The trace flag is read once per process, so the traced half runs in a child
//! copy of this test binary with the variable set, and its stderr is inspected.
use ferric_core::gpu::stats::{note_cpu, stats, CpuReason};
use std::process::Command;

const ALL: [(CpuReason, &str); 6] = [
    (CpuReason::InsideRayonWorker, "InsideRayonWorker"),
    (CpuReason::BelowThreshold, "BelowThreshold"),
    (CpuReason::PoolFull, "PoolFull"),
    (CpuReason::Layout, "Layout"),
    (CpuReason::CudaError, "CudaError"),
    (CpuReason::F32Range, "F32Range"),
];

/// Child body: only does anything when spawned by the parent test below.
#[test]
fn child_emits_every_fallback_reason() {
    if std::env::var("FERRIC_GPU_TRACE_CHILD").ok().as_deref() != Some("1") {
        return;
    }
    for (r, _) in ALL {
        note_cpu(r);
    }
}

#[test]
fn every_fallback_reason_prints_one_line_when_traced_and_none_when_not() {
    let exe = std::env::current_exe().unwrap();
    let run = |trace: Option<&str>| {
        let mut c = Command::new(&exe);
        c.args([
            "child_emits_every_fallback_reason",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("FERRIC_GPU_TRACE_CHILD", "1")
        .env_remove("FERRIC_GPU_TRACE");
        if let Some(v) = trace {
            c.env("FERRIC_GPU_TRACE", v);
        }
        let out = c.output().unwrap();
        assert!(out.status.success());
        String::from_utf8_lossy(&out.stderr).into_owned()
    };
    let traced = run(Some("1"));
    for (_, name) in ALL {
        let n = traced
            .lines()
            .filter(|l| l.contains("[gpu] GEMM fell back to the CPU:") && l.contains(name))
            .count();
        assert_eq!(n, 1, "{name}: expected exactly one trace line in\n{traced}");
    }
    let quiet = run(None);
    assert!(!quiet.contains("fell back to the CPU"), "{quiet}");
}

#[test]
fn counters_still_move_with_trace_off() {
    let before = stats().gemm_cpu_layout;
    note_cpu(CpuReason::Layout);
    assert!(stats().gemm_cpu_layout > before);
}
