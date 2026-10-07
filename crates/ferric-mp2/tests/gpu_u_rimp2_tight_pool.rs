#![cfg(feature = "gpu")]
//! Pool pressure on the U-RI-MP2 device path, in its own binary (`install` is
//! once per process): the process pool fits `B_alpha` and its scratch but not
//! `B_beta`. Then
//!  * same-spin alpha runs on the device (energy inside the derived bound);
//!  * opposite-spin needs both tensors resident: it is refused BEFORE any
//!    transfer (nothing uploaded, nothing left reserved), counted once as
//!    `gemm_cpu_pool_full`, and returns the CPU energy bit for bit;
//!  * same-spin beta (its tensor alone exceeds the pool) is the same refusal.
use ferric_core::gpu::{install, pool, probe, stats, GpuMode, GpuSettingsExplicit, GpuStatus};
use ferric_mp2::u_rimp2::{opposite_spin_pair_kernel, same_spin_pair_kernel};

#[path = "common/u_rimp2_gpu_bound.rs"]
mod bound;
#[path = "common/u_rimp2_gpu_fixture.rs"]
mod fixture;
#[path = "common/u_rimp2_gpu_synth.rs"]
mod synth;
use fixture::{cpu_opp, cpu_same};

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
            memory_gb: Some(0.0001),
            ..Default::default()
        })
        .expect("install");
    });
    true
}

#[test]
fn a_pool_that_fits_alpha_but_not_beta_runs_same_spin_alpha_on_the_device_only() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let cap = pool().expect("process pool").capacity_bytes();
    // naux 100, nvir 20: 16 000 B of B per occupied orbital
    let a = synth::synthetic(100, 2, 20, 0, 31);
    let b = synth::synthetic(100, 8, 20, 0, 32);
    let (ba, bb) = (8 * a.b.len(), 8 * b.b.len());
    let scratch_a = 8 * a.nvir * a.b.ncols();
    let scratch_ab = 8 * a.nvir * b.b.ncols();
    assert!(
        ba + scratch_a <= cap && bb > cap && ba + bb + scratch_ab > cap,
        "fixture no longer exercises the pool: cap {cap}, B_alpha {ba}, B_beta {bb}"
    );
    let (cpu_aa, cpu_bb, cpu_ab) = (cpu_same(&a), cpu_same(&b), cpu_opp(&a, &b));

    let s0 = stats();
    let (aa, _) = same_spin_pair_kernel(a.ch(), false);
    let s1 = stats();
    assert_eq!(
        (
            s1.gemm_offloaded - s0.gemm_offloaded,
            s1.resident_uploads - s0.resident_uploads,
            s1.gemm_cpu_pool_full - s0.gemm_cpu_pool_full
        ),
        (a.nocc as u64, 1, 0),
        "same-spin alpha must run on the device"
    );
    let bd = bound::same_spin_bound(&a);
    assert!(
        (aa - cpu_aa).abs() <= bd,
        "E_aa {:e} > {bd:e}",
        (aa - cpu_aa).abs()
    );

    let (ab, _) = opposite_spin_pair_kernel(a.ch(), b.ch(), false);
    let s2 = stats();
    assert_eq!(
        (
            s2.gemm_cpu_pool_full - s1.gemm_cpu_pool_full,
            s2.gemm_offloaded - s1.gemm_offloaded,
            s2.resident_uploads - s1.resident_uploads,
            s2.bytes_h2d - s1.bytes_h2d,
            s2.gemm_cpu_layout - s1.gemm_cpu_layout,
            s2.gemm_cpu_cuda_error - s1.gemm_cpu_cuda_error
        ),
        (1, 0, 0, 0, 0, 0),
        "opposite-spin: one PoolFull, no transfer, no other reason"
    );
    assert_eq!(ab.to_bits(), cpu_ab.to_bits(), "CPU energy bit for bit");

    let (bbe, _) = same_spin_pair_kernel(b.ch(), false);
    let s3 = stats();
    assert_eq!(s3.gemm_cpu_pool_full - s2.gemm_cpu_pool_full, 1);
    assert_eq!(s3.gemm_offloaded, s2.gemm_offloaded);
    assert_eq!(bbe.to_bits(), cpu_bb.to_bits());
    // control: the refusal was the pool. The same opposite-spin energy on an
    // ample explicit pool is inside the derived bound.
    let big = ferric_core::gpu::pool::DevicePool::with_capacity_bytes(1 << 26);
    let dev = ferric_core::gpu::device::device(0).unwrap();
    let on_big = ferric_mp2::rimp2_gpu::u_opposite_spin_on_device(&dev, &big, a.ch(), b.ch())
        .expect("fits an ample pool");
    assert!((on_big - cpu_ab).abs() <= bound::opposite_spin_bound(&a, &b));
    let p = pool().unwrap();
    assert_eq!(
        p.available_bytes(),
        p.capacity_bytes(),
        "nothing left reserved"
    );
}
