#![cfg(feature = "gpu")]
//! Which unrestricted RI-MP2 callers may use the mixed kernel. With
//! `precision = mixed` and `rimp2-energy` allowed:
//!  * `u_ri_mp2` with the Coulomb operator (and no kappa) runs the MIXED kernel
//!    for every occupied block of E_αα, E_ββ and E_αβ;
//!  * `u_ri_mp2` with any other operator (the attenuated U-MP2 of
//!    `u_att_mp2_vv10`) and the public pair kernels (`same_spin_pair_kernel`,
//!    `opposite_spin_pair_kernel`: the FD helper and every other direct caller)
//!    run the f64 device kernel exactly as at `precision = "f64"`: no mixed GEMM,
//!    one f64 GEMM per occupied block, inside the f64 summation-order bound.
//!
//! The mixed error map (`gpu_u_rimp2_mixed.rs`) was measured on Coulomb
//! U-RI-MP2 only.
use ferric_core::gpu::{
    install, probe, stats, GpuMode, GpuSettingsExplicit, GpuStatus, MixedKernel, MixedKernelSet,
    Precision,
};
use ferric_integrals::operator::Operator;
use ferric_mp2::rimp2::RiMp2Config;
use ferric_mp2::u_rimp2::{opposite_spin_pair_kernel, same_spin_pair_kernel, u_ri_mp2};

#[path = "common/u_rimp2_gpu_bound.rs"]
mod bound;
#[path = "common/u_rimp2_gpu_fixture.rs"]
mod fixture;
#[path = "common/u_rimp2_gpu_real.rs"]
mod real;
#[path = "common/u_rimp2_gpu_synth.rs"]
mod synth;
use fixture::{cpu_opp, cpu_same};

#[test]
fn mixed_allowed_coulomb_u_rimp2_runs_mixed_and_every_other_caller_stays_f64() {
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

    // public pair kernels: f64 device kernel
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
        "no mixed GEMM through the public pair kernels"
    );
    assert_eq!(
        s1.gemm_offloaded - s0.gemm_offloaded,
        (2 * a.nocc) as u64,
        "one f64 device GEMM per occupied block"
    );
    assert!((ss - cpu_same(&a)).abs() <= bound::same_spin_bound(&a));
    assert!((os - cpu_opp(&a, &b)).abs() <= bound::opposite_spin_bound(&a, &b));

    let case = real::real_case("o2.xyz", 0, 3);
    let (ca, cb) = real::chans(&case.amps);
    let blocks = (2 * ca.nocc + cb.nocc) as u64;
    let run = |op: Operator| {
        let s0 = stats();
        let r = u_ri_mp2(
            &case.mol,
            &case.obs,
            &case.dfbs,
            op,
            &case.scf,
            &RiMp2Config::default(),
        )
        .unwrap();
        let s1 = stats();
        (
            r,
            s1.gemm_mixed - s0.gemm_mixed,
            s1.gemm_offloaded - s0.gemm_offloaded,
        )
    };
    // Coulomb: the mixed kernel, every block
    let (_, mixed, f64n) = run(Operator::coulomb());
    assert_eq!((mixed, f64n), (blocks, 0), "Coulomb U-RI-MP2 under mixed");
    // erfc (the attenuated U-MP2 operator class): f64 device kernel only
    let (_, mixed, f64n) = run(Operator::erfc(0.5));
    assert_eq!((mixed, f64n), (0, blocks), "non-Coulomb U-RI-MP2 stays f64");
}
