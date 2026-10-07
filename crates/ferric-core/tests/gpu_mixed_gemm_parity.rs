#![cfg(feature = "gpu")]
//! Device mixed GEMM vs the CPU f64 product, bounded by
//! (2u32 + γ_b(u32) + γ_{⌈k/b⌉}(u64) + γ_k(u64))·(|A||B|)_ij (plan §3.2 (b);
//! the last term is the f64 reference's own error). u32 = 2^-24, u64 = 2^-53.
//! Measured BOTH sides (GTX 1080, sm_61, one run; bound = factor * 1.01):
//!   agreement, max e/bound over 6 shapes x 4 layouts, b = 128: 0.24 (7x5x3, where
//!     rounding the operands to f32 dominates; <= 0.034 for every k >= 128)
//!   rms e, (256,8192,256) positive operands, b = 128:  5.7e-9 device (r_b), 2.0e-8 host twin
//!     (cuBLAS panel sums are ~3.5x more accurate than OpenBLAS's; both meet the same bound)
//!   defect "f32 accumulator" = one panel (k_panel = k), same shape: rms 1.57e-7 (r_a)
//!     -> r_a / r_b = 27.3; SEPARATION = floor_1sf(sqrt(27.3)) = 5. (Positive
//!     operands: with zero-mean operands the host twin separates by only 1.6.)
//!   same defect at (128,32768,128): r_a / r_b = 104.5 (8192: 27.8, 65536: 208.5),
//!     so the deep gate is floor_1sf(sqrt(104.5)) = 10; both gates are asserted.
//!     This ratio includes cuBLAS's own internal k-reduction, so it is device- and
//!     library-dependent: calibrated on GTX 1080 (sm_61), driver 580.178.04,
//!     cuBLAS 12.0.2.224 (the banner printed on every failure gives the running
//!     device). Re-derive per device with the command in the docs of
//!     `SEPARATION_CASES`.
//!   The panel WIDTH is not pinned by SEPARATION (b = 1024 reads only 4.3x the
//!   b = 128 rms, inside the ~30x bound headroom): `panel_count_is_exactly_ceil_k_over_b_...`
//!   asserts the exact count ceil(k/b) from the core's return value and from
//!   the stats counter instead.
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

/// Where the SEPARATION constants were calibrated, printed in every failure
/// message: device, compute capability, driver and cuBLAS version integers.
fn banner() -> String {
    let dev = device(0).unwrap();
    let mut driver = 0i32;
    let mut blas_v = 0i32;
    // SAFETY: both calls only write one c_int through a valid pointer; the
    // cuBLAS handle is live for as long as `dev` is.
    unsafe {
        let _ = cudarc::driver::sys::cuDriverGetVersion(&mut driver);
        let blas = dev.blas.lock().unwrap_or_else(|e| e.into_inner());
        let _ = cudarc::cublas::sys::cublasGetVersion_v2(*blas.handle(), &mut blas_v);
    }
    format!(
        "device '{}' cc {}.{}, driver API {driver}, cuBLAS {blas_v}",
        dev.info.name, dev.info.cc_major, dev.info.cc_minor
    )
}

fn expected_panels(k: usize, b: usize) -> usize {
    k.div_ceil(b.max(1))
}

/// Upload f32-rounded standard-layout operands and run the device core;
/// returns (panels returned by the core).
fn core_panels(m: usize, k: usize, n: usize, k_panel: usize) -> usize {
    use ferric_core::gpu::gemm::{dev_left, dev_right};
    use ferric_core::gpu::mixed::gemm_f32_f64acc_dev;
    let dev = device(0).unwrap();
    let a32 = positive_operand(m, k, 21).mapv(|x| x as f32);
    let b32 = positive_operand(k, n, 22).mapv(|x| x as f32);
    let s = &dev.stream;
    let d_a = s.clone_htod(a32.as_slice().unwrap()).unwrap();
    let d_b = s.clone_htod(b32.as_slice().unwrap()).unwrap();
    let mut c32 = s.alloc_zeros::<f32>(m * n).unwrap();
    let mut c64 = s.alloc_zeros::<f64>(m * n).unwrap();
    let lo = dev_left(d_a.slice(..), &a32.view()).unwrap();
    let ro = dev_right(d_b.slice(..), &b32.view()).unwrap();
    gemm_f32_f64acc_dev(&dev, m, k, n, &lo, &ro, &mut c32, &mut c64, k_panel).unwrap()
}

#[test]
fn panel_count_is_exactly_ceil_k_over_b_whatever_cublas_does_inside() {
    // Independent of cuBLAS internals and of the error model: a kernel that
    // silently used a wider panel (error up to ~4.3x larger at b = 1024, which
    // the 30x-headroom bound and SEPARATION cannot see) changes this count.
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    for &(k, b) in &[
        (1000usize, 128usize),
        (1000, 300),
        (100, 128),
        (129, 128),
        (128, 128),
        (256, 128),
        (8192, 1024),
        (300, 300),
        (7, 1),
        (5, 0),
    ] {
        let want = expected_panels(k, b);
        assert_eq!(
            core_panels(8, k, 8, b),
            want,
            "core panel count k={k} b={b} ({})",
            banner()
        );
        let a = positive_operand(8, k, 23);
        let bm = positive_operand(k, 8, 24);
        let before = stats().mixed_panels;
        let _ = run(a.view(), bm.view(), b);
        let got = (stats().mixed_panels - before) as usize;
        assert_eq!(got, want, "stats().mixed_panels delta k={k} b={b}");
    }
}

/// Defect-side calibration of the one-panel (f32 accumulator) defect, two
/// shapes, rms of e = |got - ref| / (|A||B|)_ij, positive operands. The ratio
/// r_a / r_b measures cuBLAS's own internal k-reduction on top of the f32
/// accumulation, so it is device- and library-dependent; the constants below
/// were calibrated on the device in the module docstring and must be
/// re-derived on another device with
/// `cargo test -p ferric-core --features gpu --test gpu_mixed_gemm_parity
///  separates -- --nocapture` (read the printed ratios, take
/// floor_1sf(sqrt(ratio)) of the SMALLER one at each shape).
/// (256,8192,256): ratio 27.3, sqrt 5.2 -> 5.
/// (128,32768,128): ratio 104.5, sqrt 10.2 -> 10 (the more robust gate: a
/// 10x shrink of the defect by split-K would be needed to fail a correct kernel).
const SEPARATION_CASES: [(usize, usize, usize, f64); 2] =
    [(256, 8192, 256, SEPARATION), (128, 32768, 128, 10.0)];

#[test]
fn device_and_host_twin_both_meet_the_bound_and_one_panel_separates_at_two_depths() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    for &(m, k, n, sep) in &SEPARATION_CASES {
        let a = positive_operand(m, k, 5);
        let b = positive_operand(k, n, 6);
        let (refc, ab) = reference(&a.view(), &b.view());
        let bf_b = (mixed_error_factor(k, 128) + gamma(k, U64)) * 1.01;
        let bf_a = (mixed_error_factor(k, k) + gamma(k, U64)) * 1.01;
        let (worst_b, r_dev_b) = errors(&run(a.view(), b.view(), 128), &refc, &ab, bf_b);
        let (worst_a, r_dev_a) = errors(&run(a.view(), b.view(), k), &refc, &ab, bf_a);
        let mut host_b = Array2::zeros((m, n));
        gemm_f32_f64acc_host(&a.view(), &b.view(), &mut host_b.view_mut(), 128);
        let (worst_host, r_host_b) = errors(&host_b, &refc, &ab, bf_b);
        eprintln!(
            "{m}x{k}x{n} [{}]: rms device b=128 {r_dev_b:.3e} (max e/bound {worst_b:.3e}), host twin b=128 {r_host_b:.3e} (max e/bound {worst_host:.3e}), device one-panel {r_dev_a:.3e}; ratio {:.2}",
            banner(),
            r_dev_a / r_dev_b
        );
        assert!(worst_b <= 1.0, "panelled run violates its own bound");
        assert!(worst_a <= 1.0, "one-panel run violates its own (a) bound");
        assert!(
            worst_host <= 1.0,
            "host twin violates the same bound ({m}x{k}x{n})"
        );
        assert!(
            r_dev_a / r_dev_b >= sep,
            "{m}x{k}x{n}: one-panel (plain sgemm) rms {r_dev_a:e} is not separated from panelled {r_dev_b:e} by {sep}; calibrated on a GTX 1080 (cc 6.1, driver 580.178.04, cuBLAS 12.0.2.224), running on {}: re-derive per the docstring of SEPARATION_CASES",
            banner()
        );
    }
}

#[test]
fn short_or_mismatched_device_buffers_return_a_typed_error_and_write_nothing() {
    use ferric_core::gpu::gemm::{dev_left, dev_right, gemm_f64_dev};
    use ferric_core::gpu::mixed::gemm_f32_f64acc_dev;
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    let s = &dev.stream;
    let (m, k, n) = (16usize, 32usize, 8usize);
    let a32 = positive_operand(m, k, 31).mapv(|x| x as f32);
    let b32 = positive_operand(k, n, 32).mapv(|x| x as f32);
    let d_a = s.clone_htod(a32.as_slice().unwrap()).unwrap();
    let d_b = s.clone_htod(b32.as_slice().unwrap()).unwrap();
    let lo = dev_left(d_a.slice(..), &a32.view()).unwrap();
    let ro = dev_right(d_b.slice(..), &b32.view()).unwrap();
    let sentinel = vec![7.0f64; m * n];
    let is_layout =
        |e: &GpuError, needle: &str| matches!(e, GpuError::Layout(msg) if msg.contains(needle));
    let before = stats();

    // short c32
    let mut c32_short = s.alloc_zeros::<f32>(m * n - 1).unwrap();
    let mut c64 = s.clone_htod(&sentinel).unwrap();
    let e = gemm_f32_f64acc_dev(&dev, m, k, n, &lo, &ro, &mut c32_short, &mut c64, 8).unwrap_err();
    assert!(is_layout(&e, "c32"), "{e:?}");
    assert_eq!(s.clone_dtoh(&c64).unwrap(), sentinel, "c64 was written");

    // short c64
    let mut c32 = s.alloc_zeros::<f32>(m * n).unwrap();
    let mut c64_short = s.clone_htod(&sentinel[..m * n - 1]).unwrap();
    let e = gemm_f32_f64acc_dev(&dev, m, k, n, &lo, &ro, &mut c32, &mut c64_short, 8).unwrap_err();
    assert!(is_layout(&e, "c64"), "{e:?}");
    assert_eq!(s.clone_dtoh(&c64_short).unwrap(), sentinel[..m * n - 1]);

    // operand shape disagrees with (m, k, n): the descriptor is for a different matrix
    let mut c64 = s.clone_htod(&sentinel).unwrap();
    let e = gemm_f32_f64acc_dev(&dev, m, k + 1, n, &lo, &ro, &mut c32, &mut c64, 8).unwrap_err();
    assert!(is_layout(&e, "left"), "{e:?}");
    let e = gemm_f32_f64acc_dev(&dev, m, k, n + 1, &lo, &ro, &mut c32, &mut c64, 8).unwrap_err();
    assert!(is_layout(&e, "right"), "{e:?}");
    assert_eq!(s.clone_dtoh(&c64).unwrap(), sentinel, "c64 was written");

    // wrong role: the left operand in the right slot and vice versa (the role
    // is fixed by the constructor, so this is the only way to mix them up)
    let mut c64 = s.clone_htod(&sentinel).unwrap();
    let e = gemm_f32_f64acc_dev(&dev, m, k, n, &ro, &lo, &mut c32, &mut c64, 8).unwrap_err();
    assert!(is_layout(&e, "slot holds"), "{e:?}");
    let e = gemm_f32_f64acc_dev(&dev, m, k, n, &lo, &lo, &mut c32, &mut c64, 8).unwrap_err();
    assert!(is_layout(&e, "right slot"), "{e:?}");
    assert_eq!(s.clone_dtoh(&c64).unwrap(), sentinel, "c64 was written");

    // dev_left/dev_right: view longer / shorter than the host shape
    let e = dev_left(d_a.slice(1..), &a32.view()).err().unwrap();
    assert!(is_layout(&e, "elements"), "{e:?}");
    let strided = a32.slice(ndarray::s![.., ..4]);
    let e = dev_left(d_a.slice(..), &strided).err().unwrap();
    assert!(is_layout(&e, "layout") || is_layout(&e, "neither"), "{e:?}");

    // f64 core: short c
    let a64 = positive_operand(m, k, 33);
    let b64 = positive_operand(k, n, 34);
    let d_a64 = s.clone_htod(a64.as_slice().unwrap()).unwrap();
    let d_b64 = s.clone_htod(b64.as_slice().unwrap()).unwrap();
    let lo64 = dev_left(d_a64.slice(..), &a64.view()).unwrap();
    let ro64 = dev_right(d_b64.slice(..), &b64.view()).unwrap();
    let mut c_short = s.clone_htod(&sentinel[..m * n - 1]).unwrap();
    let e = gemm_f64_dev(&dev, m, k, n, &lo64, &ro64, &mut c_short, 8).unwrap_err();
    assert!(is_layout(&e, "output buffer c"), "{e:?}");
    assert_eq!(s.clone_dtoh(&c_short).unwrap(), sentinel[..m * n - 1]);

    let after = stats();
    assert_eq!(after.mixed_panels, before.mixed_panels);
    assert_eq!(after.gemm_mixed, before.gemm_mixed);
    assert_eq!(after.bytes_h2d, before.bytes_h2d);
}

#[test]
fn degenerate_shapes_and_every_layout_and_panel_width_meet_the_bound() {
    // k = 1, k < panel, m = 1, n = 1, k0 > 0 panels (k > b), all four layouts:
    // the descriptor/extent checks must accept every honest call.
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    for &(m, k, n) in &[
        (1usize, 1usize, 1usize),
        (1, 9, 4),
        (4, 9, 1),
        (3, 1, 5),
        (4, 3, 6),
        (5, 130, 3),
    ] {
        for b in [1usize, 3, 128] {
            let bf = (mixed_error_factor(k, b) + gamma(k, U64)) * 1.01;
            let a = operand(m, k, 51);
            let bm = operand(k, n, 52);
            let at = operand(k, m, 53);
            let bt = operand(n, k, 54);
            for (label, l, r) in [
                ("std", a.view(), bm.view()),
                ("A^T", at.t(), bm.view()),
                ("B^T", a.view(), bt.t()),
                ("A^T B^T", at.t(), bt.t()),
            ] {
                let (refc, ab) = reference(&l, &r);
                let (worst, _) = errors(&run(l, r, b), &refc, &ab, bf);
                assert!(worst <= 1.0, "{label} {m}x{k}x{n} b={b}: {worst:e}");
            }
        }
    }
}

#[test]
fn zero_sized_products_return_ok_and_leave_the_output_as_the_cpu_path_does() {
    // m = 0, n = 0: nothing computed, nothing written (c keeps its contents).
    // k = 0 with m, n > 0: the zero matrix. Both device cores and both host
    // wrappers agree, and none panics or sends ld = 0 to cuBLAS.
    use ferric_core::gpu::gemm::{dev_left, dev_right, gemm_f64, gemm_f64_dev};
    use ferric_core::gpu::mixed::gemm_f32_f64acc_dev;
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    let s = &dev.stream;
    let pool = DevicePool::with_capacity_bytes(1 << 28);
    for &(m, k, n) in &[
        (0usize, 4usize, 3usize),
        (4, 4, 0),
        (4, 0, 3),
        (0, 0, 0),
        (0, 0, 3),
        (3, 0, 0),
    ] {
        let a64 = Array2::<f64>::zeros((m, k));
        let b64 = Array2::<f64>::zeros((k, n));
        let a32 = a64.mapv(|x| x as f32);
        let b32 = b64.mapv(|x| x as f32);
        let sentinel = vec![7.0f64; 12];
        let want: Vec<f64> = if k == 0 && m > 0 && n > 0 {
            let mut w = sentinel.clone();
            w[..m * n].fill(0.0);
            w
        } else {
            sentinel.clone()
        };
        let d_a = s.clone_htod(a64.as_slice().unwrap()).unwrap();
        let d_b = s.clone_htod(b64.as_slice().unwrap()).unwrap();
        let lo = dev_left(d_a.slice(..), &a64.view()).unwrap();
        let ro = dev_right(d_b.slice(..), &b64.view()).unwrap();
        let mut c = s.clone_htod(&sentinel).unwrap();
        gemm_f64_dev(&dev, m, k, n, &lo, &ro, &mut c, 128)
            .unwrap_or_else(|e| panic!("gemm_f64_dev {m}x{k}x{n}: {e:?}"));
        assert_eq!(s.clone_dtoh(&c).unwrap(), want, "gemm_f64_dev {m}x{k}x{n}");

        let d_a32 = s.clone_htod(a32.as_slice().unwrap()).unwrap();
        let d_b32 = s.clone_htod(b32.as_slice().unwrap()).unwrap();
        let lo = dev_left(d_a32.slice(..), &a32.view()).unwrap();
        let ro = dev_right(d_b32.slice(..), &b32.view()).unwrap();
        let mut c32 = s.alloc_zeros::<f32>(12).unwrap();
        let mut c64 = s.clone_htod(&sentinel).unwrap();
        let panels = gemm_f32_f64acc_dev(&dev, m, k, n, &lo, &ro, &mut c32, &mut c64, 128)
            .unwrap_or_else(|e| panic!("gemm_f32_f64acc_dev {m}x{k}x{n}: {e:?}"));
        assert_eq!(panels, 0);
        assert_eq!(
            s.clone_dtoh(&c64).unwrap(),
            want,
            "gemm_f32_f64acc_dev {m}x{k}x{n}"
        );

        // host wrappers: out is an (m, n) array, k = 0 gives zeros, empty stays empty
        let mut out = Array2::<f64>::from_elem((m, n), 7.0);
        gemm_f64(
            &dev,
            &pool,
            &a64.view(),
            &b64.view(),
            &mut out.view_mut(),
            128,
        )
        .unwrap();
        assert!(out.iter().all(|&v| v == 0.0), "gemm_f64 {m}x{k}x{n}");
        let mut out = Array2::<f64>::from_elem((m, n), 7.0);
        gemm_f32_f64acc(
            &dev,
            &pool,
            &a64.view(),
            &b64.view(),
            &mut out.view_mut(),
            128,
        )
        .unwrap();
        let cpu_expect = if m * n == 0 { 7.0 } else { 0.0 };
        assert!(
            out.iter().all(|&v| v == 0.0 || v == cpu_expect),
            "gemm_f32_f64acc {m}x{k}x{n}"
        );
    }
}

#[test]
fn an_operand_beyond_f32_range_is_a_typed_error_before_any_lease_or_transfer() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(1 << 30);
    let mut a = operand(16, 16, 41);
    let b = operand(16, 16, 42);
    a[[3, 5]] = 1e300;
    let mut out = Array2::zeros((16, 16));
    let before = stats();
    let e =
        gemm_f32_f64acc(&dev, &pool, &a.view(), &b.view(), &mut out.view_mut(), 128).unwrap_err();
    assert!(
        matches!(e, GpuError::F32Range(ref s) if s.contains("left[53]")),
        "{e:?}"
    );
    assert_eq!(pool.available_bytes(), pool.capacity_bytes());
    assert_eq!(stats().bytes_h2d, before.bytes_h2d, "nothing uploaded");
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
    let h2d_before = stats().bytes_h2d;
    FORCE_KERNEL_FAILURE.store(true, Ordering::Relaxed);
    let err = gemm_f32_f64acc(&dev, &pool, &a.view(), &b.view(), &mut out.view_mut(), 128);
    FORCE_KERNEL_FAILURE.store(false, Ordering::Relaxed);
    assert!(matches!(err, Err(GpuError::Kernel(_))), "{err:?}");
    assert_eq!(
        stats().bytes_h2d,
        h2d_before,
        "kernel check must precede every transfer"
    );
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
