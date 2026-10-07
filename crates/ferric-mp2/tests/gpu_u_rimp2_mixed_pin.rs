#![cfg(feature = "gpu")]
//! With `precision = mixed` and the `rimp2-energy` mixed kernel allowed, the
//! UNRESTRICTED RI-MP2 energy still runs in f64 on the device: no mixed GEMM is
//! counted, one f64 GEMM per occupied block is. Guards a future edit that
//! consults `mixed_allows(RiMp2Energy)` on the UHF path before its own error
//! rows exist (the closed-shell map does not cover K = g_ab - g_ba).
use ferric_core::gpu::{
    install, probe, stats, GpuMode, GpuSettingsExplicit, GpuStatus, MixedKernel, MixedKernelSet,
    Precision,
};
use ferric_mp2::u_rimp2::{opposite_spin_pair_kernel, same_spin_pair_kernel};

#[path = "common/u_rimp2_gpu_bound.rs"]
mod bound;
#[path = "common/u_rimp2_gpu_fixture.rs"]
mod fixture;
#[path = "common/u_rimp2_gpu_synth.rs"]
mod synth;
use fixture::{cpu_opp, cpu_same};

#[test]
fn mixed_precision_with_rimp2_energy_allowed_leaves_the_unrestricted_energy_f64() {
    if !matches!(probe(0), GpuStatus::Ready(_)) {
        eprintln!("skipping: no CUDA device");
        assert!(
            std::env::var("FERRIC_GPU_TESTS_REQUIRED").ok().as_deref() != Some("1"),
            "FERRIC_GPU_TESTS_REQUIRED=1 but no device"
        );
        return;
    }
    install(GpuSettingsExplicit {
        mode: Some(GpuMode::Auto),
        memory_gb: Some(2.0),
        precision: Some(Precision::Mixed),
        mixed_kernels: Some(MixedKernelSet::EMPTY.with(MixedKernel::RiMp2Energy)),
        ..Default::default()
    })
    .expect("install");
    let a = synth::synthetic(120, 5, 18, 0, 51);
    let b = synth::synthetic(120, 4, 19, 0, 52);
    let s0 = stats();
    let (ss, _) = same_spin_pair_kernel(a.ch(), false);
    let (os, _) = opposite_spin_pair_kernel(a.ch(), b.ch(), false);
    let s1 = stats();
    assert_eq!(
        (
            s1.gemm_mixed - s0.gemm_mixed,
            s1.mixed_panels - s0.mixed_panels,
            s1.mixed_fallback_f64 - s0.mixed_fallback_f64
        ),
        (0, 0, 0),
        "no mixed GEMM on the unrestricted path"
    );
    assert_eq!(
        s1.gemm_offloaded - s0.gemm_offloaded,
        (2 * a.nocc) as u64,
        "one f64 device GEMM per occupied block"
    );
    // f64 means inside the f64 summation-order bound (a mixed result would not be)
    assert!((ss - cpu_same(&a)).abs() <= bound::same_spin_bound(&a));
    assert!((os - cpu_opp(&a, &b)).abs() <= bound::opposite_spin_bound(&a, &b));
}
