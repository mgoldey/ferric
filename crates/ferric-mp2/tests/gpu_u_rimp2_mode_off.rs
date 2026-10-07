#![cfg(feature = "gpu")]
//! With the device mode off a gpu build computes the U-RI-MP2 energy exactly as
//! the CPU path: every component bit-identical to the same call inside a
//! 1-thread rayon pool (where the dispatcher declines whatever the mode), and
//! no GPU counter moves. Needs no device.
use ferric_core::gpu::{install, stats, GpuMode, GpuSettingsExplicit};
use ferric_mp2::rimp2::RiMp2Config;
use ferric_mp2::u_rimp2::u_ri_mp2;

#[path = "common/u_rimp2_gpu_fixture.rs"]
mod fixture;
#[path = "common/u_rimp2_gpu_real.rs"]
mod real;
use fixture::{cpu_opp, cpu_same};

#[test]
fn mode_off_is_bit_identical() {
    install(GpuSettingsExplicit {
        mode: Some(GpuMode::Off),
        ..Default::default()
    })
    .expect("install");
    for (name, xyz, mult) in [
        ("O2 triplet", "o2.xyz", 3usize),
        ("CH3 doublet", "validation/ch3.xyz", 2),
    ] {
        let case = real::real_case(xyz, 0, mult);
        let (a, b) = real::chans(&case.amps);
        let run = || {
            u_ri_mp2(
                &case.mol,
                &case.obs,
                &case.dfbs,
                ferric_integrals::operator::Operator::coulomb(),
                &case.scf,
                &RiMp2Config::default(),
            )
            .unwrap()
            .components
        };
        let s0 = stats();
        let main = run();
        let s1 = stats();
        let pooled = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(run);
        let bits = |c: &ferric_mp2::u_rimp2::URiMp2Components| {
            (c.e_aa.to_bits(), c.e_bb.to_bits(), c.e_ab.to_bits())
        };
        assert_eq!(bits(&main), bits(&pooled), "{name}");
        // and equal to the kernels called directly
        assert_eq!(
            bits(&main),
            (
                cpu_same(&a).to_bits(),
                cpu_same(&b).to_bits(),
                cpu_opp(&a, &b).to_bits()
            ),
            "{name}: kernels"
        );
        assert_eq!(
            (
                s1.gemm_offloaded,
                s1.resident_uploads,
                s1.bytes_h2d,
                s1.bytes_d2h,
                s1.gemm_cpu_pool_full,
                s1.gemm_cpu_layout,
                s1.gemm_cpu_cuda_error
            ),
            (
                s0.gemm_offloaded,
                s0.resident_uploads,
                s0.bytes_h2d,
                s0.bytes_d2h,
                s0.gemm_cpu_pool_full,
                s0.gemm_cpu_layout,
                s0.gemm_cpu_cuda_error
            ),
            "{name}: the mode-off run must not touch a GPU counter"
        );
    }
}
