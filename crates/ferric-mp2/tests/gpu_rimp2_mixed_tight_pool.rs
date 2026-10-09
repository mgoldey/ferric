#![cfg(feature = "gpu")]
//! The mixed arm under a tight pool: a process pool too small for the mixed
//! scratch (and a fortiori for the f32 B_ov) makes the dispatcher run the CPU
//! path, count `PoolFull` exactly once, upload and compute nothing on the
//! device, move neither `gemm_mixed` nor `mixed_fallback_f64` (the refusal is a
//! pool refusal, not a mixed-kernel one), and return the mode-off f64 energy
//! bit for bit. The ample direction is gpu_rimp2_mixed.rs. A separate binary
//! because `install` is once per process.
use ferric_core::gpu::{
    install, probe, stats, GpuMode, GpuSettingsExplicit, GpuStatus, MixedKernel, MixedKernelSet,
    Precision,
};
use ferric_mp2::rimp2::{spin_components_from_b_ov_kappa, spin_components_from_b_ov_kappa_cpu};

#[path = "common/rimp2_gpu_fixture.rs"]
mod fixture;

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn ready() -> bool {
    if !matches!(probe(0), GpuStatus::Ready(_)) {
        eprintln!("skipping: no CUDA device");
        assert!(
            std::env::var("FERRIC_GPU_TESTS_REQUIRED").ok().as_deref() != Some("1"),
            "FERRIC_GPU_TESTS_REQUIRED=1 but no device"
        );
        return false;
    }
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| {
        install(GpuSettingsExplicit {
            mode: Some(GpuMode::Auto),
            memory_gb: Some(0.001),
            precision: Some(Precision::Mixed),
            mixed_kernels: Some(MixedKernelSet::EMPTY.with(MixedKernel::RiMp2Energy)),
            ..Default::default()
        })
        .expect("install");
    });
    true
}

#[test]
fn a_tight_pool_runs_the_cpu_path_bit_identically_under_mixed_and_counts_it() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (p, reference) = fixture::prepare_scf("benzene", "cc-pvdz", "cc-pvdz-ri", 0);
    // the pool is 1 MB; the f64 scratch alone is 8·nvir·nov (asserted, not assumed)
    let scratch_bytes = 8 * p.nvir * p.b_ov.ncols();
    assert!(
        scratch_bytes > 1_000_000,
        "fixture no longer exercises a tight pool: scratch {scratch_bytes} B"
    );
    let s0 = stats();
    let via = spin_components_from_b_ov_kappa(
        &p.b_ov,
        &p.eps,
        p.nocc,
        p.nvir,
        p.first_occ,
        p.nocc_total,
        None,
    );
    let s1 = stats();
    let cpu = spin_components_from_b_ov_kappa_cpu(
        &p.b_ov,
        &p.eps,
        p.nocc,
        p.nvir,
        p.first_occ,
        p.nocc_total,
        None,
    );
    assert_eq!(
        s1.gemm_cpu_pool_full - s0.gemm_cpu_pool_full,
        1,
        "PoolFull counted exactly once"
    );
    assert_eq!(
        (
            s1.gemm_offloaded,
            s1.gemm_mixed,
            s1.mixed_fallback_f64,
            s1.resident_uploads,
            s1.bytes_h2d
        ),
        (
            s0.gemm_offloaded,
            s0.gemm_mixed,
            s0.mixed_fallback_f64,
            s0.resident_uploads,
            s0.bytes_h2d
        ),
        "nothing reached the device and no mixed fallback was recorded"
    );
    for (name, a, b) in [
        ("vs mode-off CPU", &via, &cpu),
        ("vs the reference run", &via, &reference),
    ] {
        assert_eq!(
            (a.e_os.to_bits(), a.e_ss.to_bits(), a.e_total.to_bits()),
            (b.e_os.to_bits(), b.e_ss.to_bits(), b.e_total.to_bits()),
            "{name}: a tight pool must give the f64 CPU energy bit for bit"
        );
    }
}
