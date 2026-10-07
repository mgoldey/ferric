#![cfg(feature = "gpu")]
//! Strided-batched DGEMM and SYRK on the device vs the host, bounded by Higham
//! (ASNA §3.5): for ANY summation order |fl(Σ a_l b_l) − Σ a_l b_l| ≤ γ_k Σ|a_l||b_l|,
//! γ_k = k·u/(1−k·u), u = 2⁻⁵³. Two independently rounded products differ by at
//! most 2γ_k(|A||B|) elementwise (the 1% allowance covers the bound's own rounding).
//!
//! Defect side (MEASURED by temporary mutation of `batched.rs`, each reverted
//! and the file compared with the pre-mutation copy): swapping `stride_a` and
//! `stride_b` gives err/bound 1.1e15 (n=7, nocc=3, batch=5); `ldc = n + 1`
//! gives 1.7e15 (same case); FILL_MODE_LOWER -> UPPER fails the SYRK test at
//! n=5 k=1, (0,1): err 3.1e-3 vs bound 6.9e-19; dropping the last-matrix tail
//! from the constructor's extent fails the short-view refusal test; a wrong
//! in-bounds stride (n² + 1) gives 4.7e14. Correct side: worst err/bound 4.0e-2
//! over all GEMM cases and 1.6e-2 over the SYRK cases.
use ferric_core::gpu::batched::{
    dev_batched_left, dev_batched_right, gemm_f64_strided_batched_dev, syrk_f64_dev, BatchedDims,
};
use ferric_core::gpu::device::{device, GpuError};
use ferric_core::gpu::mixed_host::{gamma, U64};
use ferric_core::gpu::pool::DevicePool;
use ferric_core::gpu::resident::DeviceMatrix;
use ferric_core::gpu::{probe, GpuStatus};
use ndarray::{s, Array2};

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

/// Deterministic mean-zero operands (LCG) so κ_sum is not 1.
fn operand(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut st = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    })
}

/// `batch` stacked n×n matrices `b` ((batch·n) × n), the broadcast `c` (n × nocc);
/// device `Y_p = Cᵀ·B_p` at stride `nocc·n`; compare with the host product per matrix.
/// `right_stride` is the claimed stride between B matrices (n² when correct).
fn run_case(
    n: usize,
    nocc: usize,
    batch: usize,
    seed: u64,
    right_stride: usize,
    pad: usize,
) -> f64 {
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(1 << 28);
    let c = operand(n, nocc, seed);
    // `pad` spare elements after the stack so a deliberately wrong stride stays in bounds
    let b = operand(batch * n + pad.div_ceil(n.max(1)), n, seed + 1);
    let c_res = DeviceMatrix::<f64>::upload(&dev, &pool, "C", &c.view()).unwrap();
    let b_res = DeviceMatrix::<f64>::upload(&dev, &pool, "B", &b.view()).unwrap();
    let ct = c.t();
    let left = dev_batched_left(c_res.buf().slice(..), &ct, batch, 0).unwrap();
    let b0 = b.slice(s![0..n, ..]);
    let right = dev_batched_right(b_res.buf().slice(..), &b0, batch, right_stride).unwrap();
    let mut y = DeviceMatrix::<f64>::zeros(&dev, &pool, "Y", batch * nocc, n).unwrap();
    let dims = BatchedDims {
        m: nocc,
        k: n,
        n,
        batch,
    };
    gemm_f64_strided_batched_dev(&dev, dims, &left, &right, y.buf_mut(), nocc * n).unwrap();
    let mut got = vec![0.0f64; batch * nocc * n];
    dev.stream.memcpy_dtoh(y.buf(), &mut got).unwrap();
    dev.stream.synchronize().unwrap();
    let g = gamma(n, U64);
    let mut worst = 0.0f64;
    for p in 0..batch {
        let bp = b.slice(s![p * n..(p + 1) * n, ..]);
        let want = ct.dot(&bp);
        let scale = ct.mapv(f64::abs).dot(&bp.mapv(f64::abs));
        for i in 0..nocc {
            for mu in 0..n {
                let err = (got[p * nocc * n + i * n + mu] - want[(i, mu)]).abs();
                let bound = 2.0 * g * scale[(i, mu)] * 1.01 + f64::MIN_POSITIVE;
                worst = worst.max(err / bound);
            }
        }
    }
    worst
}

#[test]
fn broadcast_left_batched_gemm_matches_the_host_per_matrix_products() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    // odd sizes, nocc = 1, a single batch, a single 1x1 matrix, and a batch past one warp
    for (n, nocc, batch) in [
        (7usize, 3usize, 5usize),
        (33, 5, 9),
        (24, 1, 4),
        (1, 1, 1),
        (50, 8, 1),
        (17, 4, 64),
    ] {
        let worst = run_case(n, nocc, batch, 11, n * n, 0);
        eprintln!("n={n} nocc={nocc} batch={batch}: worst err/bound {worst:.3e}");
        assert!(
            worst <= 1.0,
            "n={n} nocc={nocc} batch={batch}: err/bound {worst:e}"
        );
    }
}

#[test]
fn a_wrong_batch_stride_violates_the_bound_so_the_gate_has_teeth() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    // A stride that is in bounds but wrong (n² + 1) reads shifted matrices:
    // the check cannot know, the numbers must. Measured side of the gate.
    let (n, nocc, batch) = (13usize, 3usize, 6usize);
    let wrong = run_case(n, nocc, batch, 5, n * n + 1, n * n);
    eprintln!("wrong stride: err/bound {wrong:.3e}");
    assert!(
        wrong > 1.0,
        "a wrong stride must exceed the bound by orders of magnitude, got {wrong:e}"
    );
}

#[test]
fn a_view_that_stops_short_of_the_last_batch_or_a_batch_mismatch_is_refused() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(1 << 24);
    let (n, nocc, batch) = (9usize, 2usize, 4usize);
    let c = operand(n, nocc, 1);
    let b = operand(batch * n, n, 2);
    let c_res = DeviceMatrix::<f64>::upload(&dev, &pool, "C", &c.view()).unwrap();
    let b_res = DeviceMatrix::<f64>::upload(&dev, &pool, "B", &b.view()).unwrap();
    let b0 = b.slice(s![0..n, ..]);
    // one element short of the last matrix: refused at construction
    let short = dev_batched_right(b_res.buf().slice(..batch * n * n - 1), &b0, batch, n * n);
    assert!(
        matches!(short, Err(GpuError::Layout(_))),
        "{:?}",
        short.err()
    );
    let ct = c.t();
    let left = dev_batched_left(c_res.buf().slice(..), &ct, batch, 0).unwrap();
    let right = dev_batched_right(b_res.buf().slice(..), &b0, batch, n * n).unwrap();
    let mut y = DeviceMatrix::<f64>::zeros(&dev, &pool, "Y", batch * nocc, n).unwrap();
    // operands built for `batch` but the product claims batch + 1
    let e = gemm_f64_strided_batched_dev(
        &dev,
        BatchedDims {
            m: nocc,
            k: n,
            n,
            batch: batch + 1,
        },
        &left,
        &right,
        y.buf_mut(),
        nocc * n,
    );
    assert!(matches!(e, Err(GpuError::Layout(_))), "{e:?}");
    // an output buffer one element short of the last panel
    let mut small = dev.stream.alloc_zeros::<f64>(batch * nocc * n - 1).unwrap();
    let e = gemm_f64_strided_batched_dev(
        &dev,
        BatchedDims {
            m: nocc,
            k: n,
            n,
            batch,
        },
        &left,
        &right,
        &mut small,
        nocc * n,
    );
    assert!(matches!(e, Err(GpuError::Layout(_))), "{e:?}");
}

/// Upper triangle (row-major r ≤ c) of `got` (n×n flat) vs `want` within `2γ_depth·scale`.
fn upper_within(
    got: &[f64],
    want: &Array2<f64>,
    scale: &Array2<f64>,
    depth: usize,
    label: &str,
) -> f64 {
    let n = want.nrows();
    let g = gamma(depth, U64);
    let mut worst = 0.0f64;
    for r in 0..n {
        for c in r..n {
            let err = (got[r * n + c] - want[(r, c)]).abs();
            let bound = 2.0 * g * scale[(r, c)] * 1.01 + f64::MIN_POSITIVE;
            assert!(
                err <= bound,
                "{label} ({r},{c}): err {err:e} > bound {bound:e}"
            );
            worst = worst.max(err / bound);
        }
    }
    worst
}

#[test]
fn syrk_matches_the_host_upper_triangle_for_beta_zero_and_one_and_leaves_the_other_alone() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(1 << 28);
    // (n, k): single element, k = 1, a typical panel, a deep k, and k < n
    for (n, k) in [(1usize, 1usize), (5, 1), (33, 40), (33, 700), (64, 3)] {
        let y1 = operand(k, n, 7);
        let y2 = operand(k, n, 8);
        let a1 = DeviceMatrix::<f64>::upload(&dev, &pool, "Y1", &y1.view()).unwrap();
        let a2 = DeviceMatrix::<f64>::upload(&dev, &pool, "Y2", &y2.view()).unwrap();
        let mut c = DeviceMatrix::<f64>::zeros(&dev, &pool, "K", n, n).unwrap();
        syrk_f64_dev(&dev, n, k, &a1.buf().slice(..), c.buf_mut(), false).unwrap();
        let mut got = vec![0.0f64; n * n];
        dev.stream.memcpy_dtoh(c.buf(), &mut got).unwrap();
        dev.stream.synchronize().unwrap();
        let want1 = y1.t().dot(&y1);
        let s1 = y1.mapv(f64::abs).t().dot(&y1.mapv(f64::abs));
        let w = upper_within(&got, &want1, &s1, k, &format!("beta0 n={n} k={k}"));
        // the strict lower triangle (row-major r > c) was never written: exactly the initial zeros
        for r in 0..n {
            for cc in 0..r {
                assert_eq!(
                    got[r * n + cc],
                    0.0,
                    "n={n} k={k}: lower ({r},{cc}) was written"
                );
            }
        }
        // accumulate: C += Y2ᵀY2 (one extra rounding per element: depth k + 1 against both products)
        syrk_f64_dev(&dev, n, k, &a2.buf().slice(..), c.buf_mut(), true).unwrap();
        dev.stream.memcpy_dtoh(c.buf(), &mut got).unwrap();
        dev.stream.synchronize().unwrap();
        let want2 = &want1 + &y2.t().dot(&y2);
        let s2 = &s1 + &y2.mapv(f64::abs).t().dot(&y2.mapv(f64::abs));
        let w2 = upper_within(&got, &want2, &s2, k + 1, &format!("beta1 n={n} k={k}"));
        eprintln!("n={n} k={k}: worst err/bound beta0 {w:.3e} beta1 {w2:.3e}");
    }
}

#[test]
fn syrk_degenerate_sizes_are_total() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(1 << 20);
    let mut c =
        DeviceMatrix::<f64>::upload(&dev, &pool, "K", &Array2::from_elem((4, 4), 3.0).view())
            .unwrap();
    let y = DeviceMatrix::<f64>::zeros(&dev, &pool, "Y", 1, 4).unwrap();
    // n = 0 and accumulate with k = 0 leave C alone; k = 0 without accumulate zeroes it
    syrk_f64_dev(&dev, 0, 5, &y.buf().slice(..), c.buf_mut(), false).unwrap();
    syrk_f64_dev(&dev, 4, 0, &y.buf().slice(..0), c.buf_mut(), true).unwrap();
    let mut got = vec![0.0f64; 16];
    dev.stream.memcpy_dtoh(c.buf(), &mut got).unwrap();
    dev.stream.synchronize().unwrap();
    assert!(got.iter().all(|&v| v == 3.0));
    syrk_f64_dev(&dev, 4, 0, &y.buf().slice(..0), c.buf_mut(), false).unwrap();
    dev.stream.memcpy_dtoh(c.buf(), &mut got).unwrap();
    dev.stream.synchronize().unwrap();
    assert!(got.iter().all(|&v| v == 0.0));
    // a too-short input is a typed refusal, not a launch
    let e = syrk_f64_dev(&dev, 4, 5, &y.buf().slice(..), c.buf_mut(), false);
    assert!(matches!(e, Err(GpuError::Layout(_))), "{e:?}");
}
