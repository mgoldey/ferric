#![cfg(feature = "gpu")]
//! A resident matrix is uploaded once, charged to the pool for its lifetime,
//! round-trips bit-for-bit (f64) or as `x as f32` (rounded), and refuses a
//! non-standard layout, a too-small pool and an f32-overflowing value before
//! touching the device. Zero-size matrices and absurd shapes never panic.
use ferric_core::gpu::device::{device, GpuError};
use ferric_core::gpu::pool::DevicePool;
use ferric_core::gpu::resident::DeviceMatrix;
use ferric_core::gpu::{probe, stats, GpuStatus};
use ndarray::Array2;

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn skip() -> bool {
    match probe(0) {
        GpuStatus::Ready(_) => false,
        o => {
            eprintln!("skipping: no CUDA device ({o:?})");
            assert!(
                std::env::var("FERRIC_GPU_TESTS_REQUIRED").ok().as_deref() != Some("1"),
                "FERRIC_GPU_TESTS_REQUIRED=1 but no device"
            );
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

#[test]
fn f64_upload_round_trips_and_holds_its_charge_until_drop() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(1 << 26);
    let a = operand(37, 91, 1);
    let before = stats();
    let m = DeviceMatrix::<f64>::upload(&dev, &pool, "test f64", &a.view()).unwrap();
    assert_eq!((m.rows(), m.cols(), m.bytes()), (37, 91, 37 * 91 * 8));
    assert_eq!(pool.available_bytes(), pool.capacity_bytes() - 37 * 91 * 8);
    let after = stats();
    assert_eq!(after.bytes_h2d - before.bytes_h2d, (37 * 91 * 8) as u64);
    assert_eq!(after.resident_uploads - before.resident_uploads, 1);
    let mut back = vec![0.0f64; 37 * 91];
    dev.stream.memcpy_dtoh(m.buf(), &mut back).unwrap();
    dev.stream.synchronize().unwrap();
    assert!(back
        .iter()
        .zip(a.iter())
        .all(|(x, y)| x.to_bits() == y.to_bits()));
    drop(m);
    assert_eq!(
        pool.available_bytes(),
        pool.capacity_bytes(),
        "charge released on drop"
    );
}

#[test]
fn rounded_upload_is_exactly_as_f32() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(1 << 26);
    let a = operand(13, 29, 2);
    let m = DeviceMatrix::<f32>::upload_rounded(&dev, &pool, "test f32", &a.view()).unwrap();
    assert_eq!(m.bytes(), 13 * 29 * 4);
    assert_eq!(pool.available_bytes(), pool.capacity_bytes() - 13 * 29 * 4);
    let mut back = vec![0.0f32; 13 * 29];
    dev.stream.memcpy_dtoh(m.buf(), &mut back).unwrap();
    dev.stream.synchronize().unwrap();
    assert!(back
        .iter()
        .zip(a.iter())
        .all(|(x, y)| x.to_bits() == (*y as f32).to_bits()));
}

#[test]
fn non_standard_layout_tight_pool_and_f32_overflow_are_refused_before_transfer() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    let a = operand(16, 16, 3);
    let before = stats();
    let pool = DevicePool::with_capacity_bytes(1 << 26);
    let e = DeviceMatrix::<f64>::upload(&dev, &pool, "t", &a.t()).unwrap_err();
    assert!(matches!(e, GpuError::Layout(_)), "{e}");
    let tiny = DevicePool::with_capacity_bytes(64);
    let e = DeviceMatrix::<f64>::upload(&dev, &tiny, "t", &a.view()).unwrap_err();
    assert!(matches!(e, GpuError::PoolFull { .. }), "{e}");
    let mut big = a.clone();
    big[(3, 4)] = 1e300;
    let e = DeviceMatrix::<f32>::upload_rounded(&dev, &pool, "t", &big.view()).unwrap_err();
    assert!(matches!(e, GpuError::F32Range(_)), "{e}");
    assert_eq!(
        pool.available_bytes(),
        pool.capacity_bytes(),
        "a refusal leaves no charge behind"
    );
    let after = stats();
    assert_eq!(
        after.bytes_h2d, before.bytes_h2d,
        "nothing uploaded when refused"
    );
    assert_eq!(after.resident_uploads, before.resident_uploads);
}

#[test]
fn zero_size_matrices_upload_without_panicking_and_hold_no_charge() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(1 << 20);
    for (r, c) in [(0usize, 5usize), (5, 0), (0, 0)] {
        let a = Array2::<f64>::zeros((r, c));
        let m = DeviceMatrix::<f64>::upload(&dev, &pool, "empty f64", &a.view()).unwrap();
        assert_eq!((m.rows(), m.cols(), m.bytes()), (r, c, 0));
        let m = DeviceMatrix::<f32>::upload_rounded(&dev, &pool, "empty f32", &a.view()).unwrap();
        assert_eq!((m.rows(), m.cols(), m.bytes()), (r, c, 0));
    }
    assert_eq!(pool.available_bytes(), pool.capacity_bytes());
}

/// Values where f64 → f32 rounding is easy to get wrong: exact ties (to even),
/// one f64 ulp either side of a tie, the f32 subnormal range (kept, not flushed),
/// f64 values below the f32 subnormals (to ±0), signed zero, the largest finite
/// f32 and the last f64 below its overflow tie, infinities and NaN.
fn hard_values() -> Vec<f64> {
    let tie = 1.0 + 2f64.powi(-24); // halfway between 1 and the next f32
    let ulp = 2f64.powi(-52);
    let max = f32::MAX as f64;
    let overflow_tie = max + 2f64.powi(103); // rounds to +inf (to even)
    let mut v = vec![
        tie,
        tie + ulp,
        tie - ulp,
        1.0 + 3.0 * 2f64.powi(-24), // tie between odd and even: up
        -tie,
        1e-40,
        -1e-40,
        f32::MIN_POSITIVE as f64 / 3.0,
        1e-300,
        -1e-300,
        0.0,
        -0.0,
        max,
        -max,
        overflow_tie - 2f64.powi(75), // just below the tie: rounds to f32::MAX
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    v.extend(operand(17, 23, 7).iter().map(|x| x * 1e3));
    v
}

#[test]
fn device_rounding_is_bit_identical_to_the_host_rounded_upload() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(1 << 26);
    let v = hard_values();
    let a = Array2::from_shape_vec((1, v.len()), v).unwrap();
    let host = DeviceMatrix::<f32>::upload_rounded(&dev, &pool, "host f32", &a.view()).unwrap();
    let src = DeviceMatrix::<f64>::upload(&dev, &pool, "src f64", &a.view()).unwrap();
    let before = stats();
    let m = DeviceMatrix::<f32>::round_from_device(&dev, &pool, "dev f32", &src).unwrap();
    let after = stats();
    assert_eq!(after.bytes_h2d, before.bytes_h2d, "no bytes over PCIe");
    assert_eq!((m.rows(), m.cols(), m.bytes()), (1, a.len(), a.len() * 4));
    let (mut x, mut y) = (vec![0f32; a.len()], vec![0f32; a.len()]);
    dev.stream.memcpy_dtoh(m.buf(), &mut x).unwrap();
    dev.stream.memcpy_dtoh(host.buf(), &mut y).unwrap();
    dev.stream.synchronize().unwrap();
    for (i, ((d, h), v)) in x.iter().zip(&y).zip(a.iter()).enumerate() {
        if v.is_nan() {
            assert!(d.is_nan(), "[{i}] NaN must stay NaN");
        } else {
            assert_eq!(
                d.to_bits(),
                h.to_bits(),
                "[{i}] {v:e}: device {d:e}, host {h:e}"
            );
            assert_eq!(d.to_bits(), (*v as f32).to_bits(), "[{i}] {v:e}");
        }
    }
    assert!(
        x[5] != 0.0 && x[5].is_subnormal(),
        "f32 subnormals are kept"
    );
    let held = host.bytes() + src.bytes() + m.bytes();
    assert_eq!(pool.available_bytes(), pool.capacity_bytes() - held);
}

#[test]
fn device_rounding_refuses_f32_overflow_and_releases_its_charge() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(1 << 26);
    let mut a = operand(9, 11, 3);
    a[[4, 7]] = f32::MAX as f64 + 2f64.powi(103); // the overflow tie: +inf
    let src = DeviceMatrix::<f64>::upload(&dev, &pool, "src", &a.view()).unwrap();
    let e = DeviceMatrix::<f32>::round_from_device(&dev, &pool, "dev f32", &src).unwrap_err();
    assert!(matches!(e, GpuError::F32Range(_)), "{e:?}");
    assert_eq!(pool.available_bytes(), pool.capacity_bytes() - src.bytes());
    // An infinity that is already infinite is not an overflow (the host rule).
    a[[4, 7]] = f64::INFINITY;
    let src = DeviceMatrix::<f64>::upload(&dev, &pool, "src inf", &a.view()).unwrap();
    assert!(DeviceMatrix::<f32>::round_from_device(&dev, &pool, "dev f32", &src).is_ok());
    // A pool without room refuses before any launch.
    let tight = DevicePool::with_capacity_bytes(src.bytes() + 4);
    let src2 = DeviceMatrix::<f64>::upload(&dev, &tight, "src", &a.view()).unwrap();
    let e = DeviceMatrix::<f32>::round_from_device(&dev, &tight, "dev f32", &src2).unwrap_err();
    assert!(matches!(e, GpuError::PoolFull { .. }), "{e:?}");
    // Zero elements: no panic, no charge.
    let z = Array2::<f64>::zeros((0, 5));
    let zsrc = DeviceMatrix::<f64>::upload(&dev, &pool, "zero", &z.view()).unwrap();
    let zm = DeviceMatrix::<f32>::round_from_device(&dev, &pool, "zero f32", &zsrc).unwrap();
    assert_eq!(zm.bytes(), 0);
}
