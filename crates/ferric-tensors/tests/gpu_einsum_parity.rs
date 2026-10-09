#![cfg(feature = "gpu")]
//! `einsum!` results on the device match the CPU within the Higham bound, the
//! device is used only where the rules say (counters, not log text), and
//! Review Focus 4: eight rayon workers calling einsum with the GPU on must all
//! get correct answers and must all be declined to the CPU.
//!
//! Higham (Accuracy and Stability, 2nd ed., sec. 3.5): for ANY summation order
//! |fl(sum a_l b_l) - sum a_l b_l| <= gamma_k sum |a_l||b_l|, gamma_k = k u/(1-k u),
//! u = 2^-53, so two independently rounded products differ elementwise by at
//! most 2 gamma_k (|A||B|)_ij. Derived with the stated k, not tuned.
use ferric_core::gpu::device::device;
use ferric_core::gpu::{install, stats, GpuMode, GpuSettingsExplicit};
use ferric_core::gpu::{probe, GpuStatus};
use ferric_tensors::{einsum, Axis, Tensor};
use ndarray::{Array, ArrayD, IxDyn};

/// One device test at a time: stats deltas are process-global and the hog test
/// takes the whole device. Poison-recovering so one failure does not cascade.
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Offload threshold used by every test here: well above the 4x4 case (128
/// FLOP), well below the 9.6 MFLOP and 2*64*300*64 = 2.5 MFLOP cases.
/// Installed explicitly so the tests do not depend on the caller's env.
const MIN_FLOPS: usize = 1_000_000;

fn skip() -> bool {
    match probe(0) {
        GpuStatus::Ready(_) => {
            install(GpuSettingsExplicit {
                mode: Some(GpuMode::Auto),
                min_flops: Some(MIN_FLOPS),
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

/// Deterministic LCG fill, mean zero (same generator as gpu_gemm_parity.rs).
fn lcg(shape: &[usize], seed: u64) -> ArrayD<f64> {
    let mut s = seed;
    Array::from_shape_fn(IxDyn(shape), |_| {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    })
}

fn t2(shape: &[usize], seed: u64, l: [Axis; 2]) -> Tensor<2> {
    Tensor::new(lcg(shape, seed), l)
}

fn gamma(k: usize) -> f64 {
    let ku = k as f64 * f64::EPSILON / 2.0;
    ku / (1.0 - ku)
}

/// Device result vs the CPU path (run inside a 1-thread rayon worker, which
/// the dispatch declines), within 2 gamma_k (|A||B|) * 1.01 (the 1% covers the
/// bound's own rounding), k = 60.
#[test]
fn large_einsum_is_offloaded_and_matches_cpu_within_bound() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let a_arr = lcg(&[40, 40, 60], 1);
    let b_arr = lcg(&[60, 50], 2);
    let a = Tensor::new(a_arr.clone(), [Axis::O, Axis::O, Axis::V]);
    let b = Tensor::new(b_arr.clone(), [Axis::V, Axis::Aux]);
    let before = stats();
    let gpu_out = einsum!("ijc,cP->ijP", &a, &b);
    let after = stats();
    assert_eq!(
        after.gemm_offloaded,
        before.gemm_offloaded + 1,
        "exactly one device GEMM expected"
    );
    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let cpu_out = pool1.install(|| einsum!("ijc,cP->ijP", &a, &b));
    let abs_a = Tensor::new(a_arr.mapv(f64::abs), [Axis::O, Axis::O, Axis::V]);
    let abs_b = Tensor::new(b_arr.mapv(f64::abs), [Axis::V, Axis::Aux]);
    let scale = 2.0 * gamma(60) * 1.01;
    let mag = pool1.install(|| einsum!("ijc,cP->ijP", &abs_a, &abs_b));
    let mut worst = 0.0f64;
    for (idx, ((g, c), m)) in gpu_out
        .iter()
        .zip(cpu_out.iter())
        .zip(mag.iter())
        .enumerate()
    {
        let d = (g - c).abs();
        assert!(
            d <= m * scale,
            "element {idx}: |gpu-cpu|={d:e} > {:e}",
            m * scale
        );
        worst = worst.max(d / (m * scale));
    }
    eprintln!("einsum ijc,cP->ijP (k=60): max |gpu-cpu|/bound = {worst:.3e}");
}

#[test]
fn below_threshold_stays_on_cpu_and_says_so() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let a = t2(&[4, 4], 3, [Axis::O, Axis::V]);
    let b = t2(&[4, 4], 4, [Axis::V, Axis::O]);
    let before = stats();
    let _ = einsum!("ij,jk->ik", &a, &b); // 2*4*4*4 = 128 FLOP < MIN_FLOPS
    let after = stats();
    assert_eq!(after.gemm_offloaded, before.gemm_offloaded);
    assert_eq!(
        after.gemm_cpu_below_threshold,
        before.gemm_cpu_below_threshold + 1
    );
}

#[test]
fn einsum_from_rayon_workers_declines_the_device_and_matches_cpu() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    use rayon::prelude::*;
    let a = t2(&[64, 300], 5, [Axis::O, Axis::V]);
    let b = t2(&[300, 64], 6, [Axis::V, Axis::O]);
    let reference = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| einsum!("ij,jk->ik", &a, &b));
    let before = stats();
    let pool8 = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();
    let outs: Vec<_> = pool8.install(|| {
        (0..8)
            .into_par_iter()
            .map(|_| einsum!("ij,jk->ik", &a, &b))
            .collect()
    });
    let after = stats();
    assert_eq!(
        after.gemm_offloaded, before.gemm_offloaded,
        "a rayon worker must not offload (Phase 1 rule)"
    );
    assert_eq!(
        after.gemm_cpu_inside_worker,
        before.gemm_cpu_inside_worker + 8
    );
    for o in outs {
        assert_eq!(
            o, reference,
            "worker results must be bit-identical to the serial CPU path"
        );
    }
}

/// Review Focus 2: the pool ADMITS (default pool, ~80% of free memory) but the
/// device cannot allocate because another allocation holds it. The dispatch
/// must return Err -> CPU result unchanged, `gemm_cpu_cuda_error` moves, no abort.
#[test]
fn pool_admits_but_device_alloc_fails_falls_back_to_cpu() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let a = t2(&[600, 500], 9, [Axis::O, Axis::V]);
    let b = t2(&[500, 600], 10, [Axis::V, Axis::O]);
    let reference = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| einsum!("ij,jk->ik", &a, &b));
    let dev = device(0).unwrap();
    // Hog the device: 64 MiB chunks until the driver refuses one, then 128 KiB
    // chunks, so less than 128 KiB stays free. The operands alone are 2.4 MB each.
    let mut hog = Vec::new();
    for chunk_f64 in [8usize << 20, 16 << 10] {
        while let Ok(chunk) = dev.stream.alloc_zeros::<f64>(chunk_f64) {
            hog.push(chunk);
        }
    }
    assert!(!hog.is_empty());
    let before = stats();
    let out = einsum!("ij,jk->ik", &a, &b);
    let after = stats();
    drop(hog);
    assert_eq!(after.gemm_offloaded, before.gemm_offloaded);
    assert_eq!(after.gemm_cpu_pool_full, before.gemm_cpu_pool_full);
    assert_eq!(after.gemm_cpu_cuda_error, before.gemm_cpu_cuda_error + 1);
    assert_eq!(out, reference, "fallback must produce the plain CPU result");
}
