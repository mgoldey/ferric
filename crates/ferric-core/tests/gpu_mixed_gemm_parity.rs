#![cfg(feature = "gpu")]
//! Device mixed GEMM vs the CPU f64 product, bounded by
//! (2u32 + γ_b(u32) + γ_{⌈k/b⌉}(u64) + γ_k(u64))·(|A||B|)_ij (plan §3.2 (b);
//! the last term is the f64 reference's own error). u32 = 2^-24, u64 = 2^-53.
//! Measured BOTH sides (GTX 1080, sm_61, one run; bound = factor * 1.01):
//!   agreement, max e/bound over 6 shapes x 4 layouts, b = 128: 0.24 (7x5x3, where
//!     rounding the operands to f32 dominates; <= 0.034 for every k >= 128)
//!   rms e, (256,8192,256) positive operands, b = 128:  5.7e-9 device (r_b), 2.0e-8 host twin
//!   defect "f32 accumulator" = one panel (k_panel = k), same shape: rms 1.57e-7 (r_a)
//!     -> r_a / r_b = 27.3; SEPARATION = floor_1sf(sqrt(27.3)) = 5. (Positive
//!     operands: with zero-mean operands the host twin separates by only 1.6.)
//!   defect "dropped last panel" (`while k0 + kb < k`): max e/bound 6.9e3 (301x1000x97 std);
//!     the separation test's panelled run reads 2.7e3
//!   defect "wrong beta" (`beta: 1.0` in the sgemm cfg): max e/bound 9.4e4 (301x1000x97 std);
//!     the separation test's panelled run reads 4.1e6
//!   defect "flush skipped on panel 0" (`if panels > 0 { launch }`): max e/bound 8.7e3
//!     (301x1000x97 std); the separation test's panelled run reads 2.6e3
//! The three mutations were applied one at a time to `mixed.rs`, confirmed
//! present with `grep` before AND after each run, then reverted (file restored
//! byte-identical); the one-panel defect is permanent in the test because it is
//! reachable through the public API.
use ferric_core::gpu::device::{device, GpuError, FORCE_KERNEL_FAILURE};
use ferric_core::gpu::mixed::{gemm_f32_f64acc, mixed_offload_bytes};
use ferric_core::gpu::mixed_host::{gamma, gemm_f32_f64acc_host, mixed_error_factor, U64};
use ferric_core::gpu::pool::DevicePool;
use ferric_core::gpu::{probe, stats, GpuStatus};
use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, ArrayView2};
use std::sync::atomic::Ordering;

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// floor_1sf(sqrt(r_a / r_b)) = floor_1sf(sqrt(27.3)) = 5; derivation in the module docstring.
const SEPARATION: f64 = 5.0;

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

fn operand(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut s = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    })
}

/// Operands in [0, 1): |A||B| = AB, so an f32 accumulator cannot hide behind
/// cancellation (see `gpu_mixed_host_model.rs`).
fn positive_operand(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    operand(rows, cols, seed).mapv(|x| x + 0.5)
}

fn reference(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> (Array2<f64>, Array2<f64>) {
    let mut c = Array2::zeros((a.nrows(), b.ncols()));
    general_mat_mul(1.0, a, b, 0.0, &mut c);
    let mut ab = Array2::zeros((a.nrows(), b.ncols()));
    general_mat_mul(1.0, &a.mapv(f64::abs), &b.mapv(f64::abs), 0.0, &mut ab);
    (c, ab)
}

/// (max over elements of e/bound_factor, rms e), e = |got − ref| / (|A||B|)_ij.
fn errors(got: &Array2<f64>, r: &Array2<f64>, ab: &Array2<f64>, bound_factor: f64) -> (f64, f64) {
    let (mut worst, mut sum2, mut n) = (0.0f64, 0.0f64, 0usize);
    for ((g, rr), s) in got.iter().zip(r.iter()).zip(ab.iter()) {
        if *s > 0.0 {
            let e = (g - rr).abs() / s;
            worst = worst.max(e / bound_factor);
            sum2 += e * e;
            n += 1;
        }
    }
    (worst, (sum2 / n.max(1) as f64).sqrt())
}

fn run(a: ArrayView2<f64>, b: ArrayView2<f64>, k_panel: usize) -> Array2<f64> {
    let dev = device(0).unwrap();
    let pool =
        DevicePool::with_capacity_bytes(mixed_offload_bytes(a.nrows(), a.ncols(), b.ncols()) * 2);
    let mut out = Array2::zeros((a.nrows(), b.ncols()));
    gemm_f32_f64acc(&dev, &pool, &a, &b, &mut out.view_mut(), k_panel).unwrap();
    out
}

#[test]
fn every_accepted_layout_meets_the_mixed_bound() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    for &(m, k, n) in &[
        (301usize, 1000usize, 97usize),
        (128, 2000, 64),
        (256, 8192, 256),
        (512, 128, 512),
        (7, 5, 3),
        (393, 912, 5895),
    ] {
        let bf = (mixed_error_factor(k, 128) + gamma(k, U64)) * 1.01;
        let a = operand(m, k, 1);
        let b = operand(k, n, 2);
        let at = operand(k, m, 3);
        let bt = operand(n, k, 4);
        for (label, l, r) in [
            ("std", a.view(), b.view()),
            ("A^T", at.t(), b.view()),
            ("B^T", a.view(), bt.t()),
            ("A^T B^T", at.t(), bt.t()),
        ] {
            let (refc, ab) = reference(&l, &r);
            let got = run(l, r, 128);
            let (worst, rms) = errors(&got, &refc, &ab, bf);
            eprintln!("{label} {m}x{k}x{n}: max e/bound = {worst:.3e}, rms e = {rms:.3e}");
            assert!(worst <= 1.0, "{label} {m}x{k}x{n}: max e/bound = {worst:e}");
        }
    }
}

#[test]
fn device_matches_the_host_twins_error_class_and_separates_from_plain_sgemm() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let (m, k, n) = (256usize, 8192usize, 256usize);
    let a = positive_operand(m, k, 5);
    let b = positive_operand(k, n, 6);
    let (refc, ab) = reference(&a.view(), &b.view());
    let bf_b = (mixed_error_factor(k, 128) + gamma(k, U64)) * 1.01;
    let bf_a = (mixed_error_factor(k, k) + gamma(k, U64)) * 1.01;
    let (worst_b, r_dev_b) = errors(&run(a.view(), b.view(), 128), &refc, &ab, bf_b);
    let (worst_a, r_dev_a) = errors(&run(a.view(), b.view(), k), &refc, &ab, bf_a);
    let mut host_b = Array2::zeros((m, n));
    gemm_f32_f64acc_host(&a.view(), &b.view(), &mut host_b.view_mut(), 128);
    let (_, r_host_b) = errors(&host_b, &refc, &ab, bf_b);
    eprintln!(
        "rms: device b=128 {r_dev_b:.3e} (max e/bound {worst_b:.3e}), host twin b=128 {r_host_b:.3e}, device one-panel {r_dev_a:.3e}; ratio {:.2}",
        r_dev_a / r_dev_b
    );
    assert!(worst_b <= 1.0, "panelled run violates its own bound");
    assert!(worst_a <= 1.0, "one-panel run violates its own (a) bound");
    assert!(
        r_dev_a / r_dev_b >= SEPARATION,
        "one-panel (plain sgemm) rms {r_dev_a:e} is not separated from panelled {r_dev_b:e} by {SEPARATION}"
    );
}

#[test]
fn device_mixed_results_are_run_to_run_identical() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let a = operand(400, 2000, 9);
    let b = operand(2000, 400, 10);
    let r1 = run(a.view(), b.view(), 128);
    let r2 = run(a.view(), b.view(), 128);
    let diffs = r1
        .iter()
        .zip(r2.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    assert_eq!(
        diffs, 0,
        "{diffs} elements differ between two identical mixed GEMMs"
    );
}

#[test]
fn flush_kernel_covers_more_elements_than_the_grid_has_threads() {
    // 4096 blocks × 256 threads = 1,048,576 threads; m·n = 3,000,000 forces the
    // grid-stride loop. One panel so the result is exactly (double)(sgemm).
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let (m, k, n) = (1500usize, 16usize, 2000usize);
    let a = operand(m, k, 11);
    let b = operand(k, n, 12);
    let (refc, ab) = reference(&a.view(), &b.view());
    let bf = (mixed_error_factor(k, k) + gamma(k, U64)) * 1.01;
    let got = run(a.view(), b.view(), k);
    let (worst, _) = errors(&got, &refc, &ab, bf);
    assert!(
        worst <= 1.0,
        "grid-stride tail wrong: max e/bound {worst:e}"
    );
    assert!(got.iter().all(|v| v.is_finite()));
}

#[test]
fn pool_too_small_is_refused_before_any_transfer() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(1024);
    let a = operand(256, 256, 13);
    let b = operand(256, 256, 14);
    let mut out = Array2::zeros((256, 256));
    let before = stats();
    let err =
        gemm_f32_f64acc(&dev, &pool, &a.view(), &b.view(), &mut out.view_mut(), 128).unwrap_err();
    assert!(matches!(err, GpuError::PoolFull { .. }), "{err}");
    assert_eq!(stats().bytes_h2d, before.bytes_h2d);
    assert_eq!(stats().gemm_mixed, before.gemm_mixed);
}

#[test]
fn a_kernel_load_failure_is_a_typed_error_that_releases_the_lease() {
    // The caller (einsum in Task 4.3, rimp2 in Task 4.2b) turns this error
    // into an f64 device run + `mixed_fallback_f64`.
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(1 << 30);
    let a = operand(64, 64, 15);
    let b = operand(64, 64, 16);
    let mut out = Array2::zeros((64, 64));
    FORCE_KERNEL_FAILURE.store(true, Ordering::Relaxed);
    let err = gemm_f32_f64acc(&dev, &pool, &a.view(), &b.view(), &mut out.view_mut(), 128);
    FORCE_KERNEL_FAILURE.store(false, Ordering::Relaxed);
    assert!(matches!(err, Err(GpuError::Kernel(_))), "{err:?}");
    assert_eq!(
        pool.available_bytes(),
        pool.capacity_bytes(),
        "lease must be released"
    );
    // and the device still works afterwards
    gemm_f32_f64acc(&dev, &pool, &a.view(), &b.view(), &mut out.view_mut(), 128).unwrap();
}

#[test]
fn an_interleaved_view_is_refused_not_mangled() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(1 << 30);
    let big = operand(64, 64, 17);
    let a = big.slice(ndarray::s![..32, ..32]);
    let b = operand(32, 16, 18);
    let mut out = Array2::zeros((32, 16));
    let err = gemm_f32_f64acc(&dev, &pool, &a, &b.view(), &mut out.view_mut(), 128).unwrap_err();
    assert!(matches!(err, GpuError::Layout(_)), "{err}");
}
