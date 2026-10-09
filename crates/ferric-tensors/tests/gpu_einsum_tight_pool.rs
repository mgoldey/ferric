#![cfg(feature = "gpu")]
//! A device pool too small for the operands makes einsum fall back to the CPU
//! per call and count it. Own binary: the process-wide pool is sized once.
use ferric_core::gpu::{install, probe, stats, GpuMode, GpuSettingsExplicit, GpuStatus};
use ferric_tensors::{einsum, Axis, Tensor};
use ndarray::{Array, IxDyn};

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn skip() -> bool {
    match probe(0) {
        GpuStatus::Ready(_) => {
            install(GpuSettingsExplicit {
                mode: Some(GpuMode::Auto),
                min_flops: Some(0),
                memory_gb: Some(4096e-9), // 4096 bytes < 8*(128*256*2 + 128*128)
                ..Default::default()
            })
            .expect("install explicit settings");
            false
        }
        o => {
            eprintln!("skipping: no CUDA device ({o:?})");
            assert!(std::env::var("FERRIC_GPU_TESTS_REQUIRED").ok().as_deref() != Some("1"));
            true
        }
    }
}

fn t2(shape: &[usize], seed: u64, l: [Axis; 2]) -> Tensor<2> {
    let mut s = seed;
    Tensor::new(
        Array::from_shape_fn(IxDyn(shape), |_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        }),
        l,
    )
}

#[test]
fn tight_device_pool_falls_back_per_call_and_counts_it() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let a = t2(&[128, 256], 7, [Axis::O, Axis::V]);
    let b = t2(&[256, 128], 8, [Axis::V, Axis::O]);
    let before = stats();
    let out = einsum!("ij,jk->ik", &a, &b);
    let after = stats();
    assert_eq!(after.gemm_cpu_pool_full, before.gemm_cpu_pool_full + 1);
    assert_eq!(after.gemm_offloaded, before.gemm_offloaded);
    assert_eq!(out.shape(), &[128, 128]);
}
