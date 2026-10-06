#![cfg(feature = "gpu")]
//! GPU dgemm vs CPU dgemm, bounded by Higham (Accuracy and Stability, 2nd ed.,
//! §3.5): for ANY summation order, |fl(Σ a_l b_l) − Σ a_l b_l| ≤ γ_k Σ|a_l||b_l|,
//! γ_k = k u/(1 − k u), u = 2^-53. Two independently-rounded products therefore
//! differ by at most 2 γ_k (|A||B|)_ij elementwise. The bound is derived, not
//! tuned, and machine-independent. Defect side (measured before pinning, see
//! the docstring of `every_accepted_layout_meets_the_higham_bound`): swapping
//! transa, or ignoring the stride of a transposed view, violates it by >1e8x.
use ferric_core::gpu::device::device;
use ferric_core::gpu::gemm::{gemm_f64, offload_bytes};
use ferric_core::gpu::pool::DevicePool;
use ferric_core::gpu::{probe, GpuStatus};
use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, ArrayView2};

fn skip() -> bool {
    match probe(0) {
        GpuStatus::Ready(_) => false,
        o => {
            eprintln!("skipping: no CUDA device ({o:?})");
            assert!(std::env::var("FERRIC_GPU_TESTS_REQUIRED").ok().as_deref() != Some("1"));
            true
        }
    }
}

/// Deterministic pseudo-random operands (LCG), mean zero, so κ_sum is not 1.
fn operand(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut s = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    })
}

fn gamma(k: usize) -> f64 {
    let ku = k as f64 * f64::EPSILON / 2.0;
    ku / (1.0 - ku)
}

/// Elementwise 2γ_k(|A||B|) + a 1% allowance for the bound's own rounding.
fn bound(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> Array2<f64> {
    let mut ab = Array2::zeros((a.nrows(), b.ncols()));
    general_mat_mul(1.0, &a.mapv(f64::abs), &b.mapv(f64::abs), 0.0, &mut ab);
    ab * (2.0 * gamma(a.ncols()) * 1.01)
}

fn check(a: ArrayView2<f64>, b: ArrayView2<f64>, label: &str) {
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(offload_bytes(a.nrows(), a.ncols(), b.ncols()) * 2);
    let mut cpu = Array2::zeros((a.nrows(), b.ncols()));
    general_mat_mul(1.0, &a, &b, 0.0, &mut cpu);
    let mut gpu = Array2::zeros((a.nrows(), b.ncols()));
    gemm_f64(&dev, &pool, &a, &b, &mut gpu.view_mut(), 128).unwrap();
    let bnd = bound(&a, &b);
    let mut worst = 0.0f64;
    for ((i, j), g) in gpu.indexed_iter() {
        let d = (g - cpu[(i, j)]).abs();
        assert!(
            d <= bnd[(i, j)],
            "{label}: ({i},{j}) |gpu-cpu|={d:e} > bound {:e}",
            bnd[(i, j)]
        );
        if bnd[(i, j)] > 0.0 {
            worst = worst.max(d / bnd[(i, j)]);
        }
    }
    eprintln!("{label}: max |gpu-cpu|/bound = {worst:.3e}");
}

/// Measured 2026-10-06 on GTX 1080 (sm_61), cuBLAS via cudarc 0.19.10, k_block
/// 128. Passing: max |gpu-cpu|/bound is 7e-4 at k=1000 (301x1000x97), 2e-4 at
/// k=2000, 9e-5 at k=3000, 0 at k=128, 0.16 at k=5 (one slab, so the two codes
/// agree to a few ulp and the bound is tiny); the bound is never approached.
/// Defect side (first failing shape, 301x1000x97 elementwise |gpu-cpu|/bound):
///   - op-N k_step forced to 1:      std,  1.77 / 1.4e-11  = 1.3e11
///   - op-T k_step forced to 1:      A^T,  4.86 / 1.3e-11  = 3.6e11
///   - op-T ld = ncols (ignores the stride): A^T, 0.78 / 1.3e-11 = 5.8e10
///   - transa/transb swapped: cuBLAS rejects the call (INVALID_VALUE, the
///     leading dimensions no longer fit), so the test fails on the unwrap.
#[test]
fn every_accepted_layout_meets_the_higham_bound() {
    if skip() {
        return;
    }
    for &(m, k, n) in &[
        (301usize, 1000usize, 97usize),
        (128, 2000, 64),
        (65, 3000, 33),
        (512, 128, 512),
        (7, 5, 3),
    ] {
        let a = operand(m, k, 1);
        let b = operand(k, n, 2);
        check(a.view(), b.view(), &format!("std {m}x{k}x{n}"));
        // transposed-standard views: the GEMM must read them with op=T and the
        // right leading dimension (Review Focus #3)
        let at = operand(k, m, 3);
        let bt = operand(n, k, 4);
        check(at.t(), b.view(), &format!("A^T {m}x{k}x{n}"));
        check(a.view(), bt.t(), &format!("B^T {m}x{k}x{n}"));
        check(at.t(), bt.t(), &format!("A^T B^T {m}x{k}x{n}"));
    }
}

#[test]
fn an_interleaved_view_is_refused_not_mangled() {
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(1 << 30);
    let big = operand(64, 64, 5);
    let a = big.slice(ndarray::s![..32, ..32]); // row stride 64 ≠ ncols: neither layout
    let b = operand(32, 16, 6);
    let mut out = Array2::zeros((32, 16));
    let err = gemm_f64(&dev, &pool, &a, &b.view(), &mut out.view_mut(), 128).unwrap_err();
    assert!(
        matches!(err, ferric_core::gpu::device::GpuError::Layout(_)),
        "{err}"
    );
}

#[test]
fn pool_too_small_is_a_pool_error_before_any_transfer() {
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(1024);
    let a = operand(256, 256, 7);
    let b = operand(256, 256, 8);
    let mut out = Array2::zeros((256, 256));
    let before = ferric_core::gpu::stats();
    let err = gemm_f64(&dev, &pool, &a.view(), &b.view(), &mut out.view_mut(), 128).unwrap_err();
    assert!(
        matches!(err, ferric_core::gpu::device::GpuError::PoolFull { .. }),
        "{err}"
    );
    assert_eq!(
        ferric_core::gpu::stats().bytes_h2d,
        before.bytes_h2d,
        "nothing may be uploaded when refused"
    );
}

#[test]
fn device_results_are_run_to_run_identical() {
    // cuBLAS documents bitwise reproducibility for a fixed config on one
    // architecture. MEASURE it here; if this ever fails, record it and drop
    // the claim from the docs rather than loosening the test.
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(1 << 30);
    let a = operand(400, 2000, 9);
    let b = operand(2000, 400, 10);
    let mut r1 = Array2::zeros((400, 400));
    let mut r2 = Array2::zeros((400, 400));
    gemm_f64(&dev, &pool, &a.view(), &b.view(), &mut r1.view_mut(), 128).unwrap();
    gemm_f64(&dev, &pool, &a.view(), &b.view(), &mut r2.view_mut(), 128).unwrap();
    let diffs = r1
        .iter()
        .zip(r2.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    assert_eq!(
        diffs, 0,
        "{diffs} elements differ between two identical device GEMMs"
    );
}

#[test]
fn device_alloc_failure_after_a_granted_reserve_is_an_error_not_an_abort() {
    // The ledger only knows what ferric reserved: another process can take
    // device memory after the reserve is granted. Simulate that by holding
    // nearly all free device memory ourselves, with a pool that happily admits
    // the request. The GEMM must return Err (caller falls back to the CPU),
    // not panic/abort, and must not leave the pool debited.
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    // Grab most of the free memory in one block, then drain the rest in 256 KiB
    // chunks until the driver refuses: what remains is < 256 KiB, less than one
    // 512x512 f64 operand (2 MiB).
    let (free, _) = dev.mem_info().unwrap();
    let mut hogs = vec![dev
        .stream
        .alloc_zeros::<f64>(free.saturating_sub(256 << 20) / 8)
        .expect("big hog")];
    while let Ok(h) = dev.stream.alloc_zeros::<f64>((256 << 10) / 8) {
        hogs.push(h);
    }
    let pool = DevicePool::with_capacity_bytes(1 << 40);
    let a = operand(512, 512, 11);
    let b = operand(512, 512, 12);
    let mut out = Array2::zeros((512, 512));
    let err = gemm_f64(&dev, &pool, &a.view(), &b.view(), &mut out.view_mut(), 128).unwrap_err();
    drop(hogs);
    assert!(
        matches!(err, ferric_core::gpu::device::GpuError::Cuda(_)),
        "{err}"
    );
    assert_eq!(
        pool.available_bytes(),
        pool.capacity_bytes(),
        "lease must be released on the error path"
    );
    // the device is still usable afterwards
    gemm_f64(&dev, &pool, &a.view(), &b.view(), &mut out.view_mut(), 128).unwrap();
}
