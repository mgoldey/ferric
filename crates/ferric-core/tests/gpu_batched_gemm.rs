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
//! over the broadcast GEMM cases (0.48 over the general gapped ones) and 1.6e-2 over the SYRK cases. Outputs are pre-filled
//! with a NaN sentinel (beta = 0 must overwrite, never read C) and the unused
//! gaps / triangles must keep the sentinel's exact bits.
use ferric_core::gpu::batched::{
    dev_batched_left, dev_batched_right, gemm_f64_strided_batched_dev, syrk_f64_dev, BatchedDims,
};
use ferric_core::gpu::device::{device, GpuError};
use ferric_core::gpu::mixed_host::{gamma, U64};
use ferric_core::gpu::pool::DevicePool;
use ferric_core::gpu::resident::DeviceMatrix;
use ferric_core::gpu::{probe, GpuStatus};
use ndarray::{s, Array2};

/// NaN with a payload: a read of it poisons any sum, and an untouched element is
/// recognisable bit for bit.
const SENTINEL: u64 = 0x7ff8_dead_beef_0001;

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
    // NaN-filled output: beta = 0 must overwrite and must not read C.
    let nan = Array2::from_elem((batch * nocc, n), f64::NAN);
    let mut y = DeviceMatrix::<f64>::upload(&dev, &pool, "Y", &nan.view()).unwrap();
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
                worst = worst_of(worst, err / bound);
            }
        }
    }
    worst
}

/// `max` that does not swallow a NaN (`f64::max` returns the other operand).
fn worst_of(worst: f64, ratio: f64) -> f64 {
    if ratio.is_nan() {
        f64::INFINITY
    } else {
        worst.max(ratio)
    }
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
        "a wrong stride must violate the Higham bound (correct side <= 0.5), got {wrong:e}"
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
/// The strict lower triangle (row-major r > c) still holds the sentinel's exact bits.
fn assert_lower_untouched(got: &[f64], n: usize, label: &str) {
    for r in 0..n {
        for c in 0..r {
            assert_eq!(
                got[r * n + c].to_bits(),
                SENTINEL,
                "{label}: lower ({r},{c}) was written"
            );
        }
    }
}

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
        let mut c = DeviceMatrix::<f64>::upload(
            &dev,
            &pool,
            "K",
            &Array2::from_elem((n, n), f64::from_bits(SENTINEL)).view(),
        )
        .unwrap();
        syrk_f64_dev(&dev, n, k, &a1.buf().slice(..), c.buf_mut(), false).unwrap();
        let mut got = vec![0.0f64; n * n];
        dev.stream.memcpy_dtoh(c.buf(), &mut got).unwrap();
        dev.stream.synchronize().unwrap();
        let want1 = y1.t().dot(&y1);
        let s1 = y1.mapv(f64::abs).t().dot(&y1.mapv(f64::abs));
        let w = upper_within(&got, &want1, &s1, k, &format!("beta0 n={n} k={k}"));
        assert_lower_untouched(&got, n, &format!("beta0 n={n} k={k}"));
        // accumulate: C += Y2ᵀY2 (one extra rounding per element: depth k + 1 against both products)
        syrk_f64_dev(&dev, n, k, &a2.buf().slice(..), c.buf_mut(), true).unwrap();
        dev.stream.memcpy_dtoh(c.buf(), &mut got).unwrap();
        dev.stream.synchronize().unwrap();
        let want2 = &want1 + &y2.t().dot(&y2);
        let s2 = &s1 + &y2.mapv(f64::abs).t().dot(&y2.mapv(f64::abs));
        let w2 = upper_within(&got, &want2, &s2, k + 1, &format!("beta1 n={n} k={k}"));
        assert_lower_untouched(&got, n, &format!("beta1 n={n} k={k}"));
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
    // only the owned triangle (row-major r <= c) is zeroed; the other keeps its 3.0
    for r in 0..4 {
        for c in 0..4 {
            let want = if c >= r { 0.0 } else { 3.0 };
            assert_eq!(got[r * 4 + c], want, "k = 0 write set at ({r},{c})");
        }
    }
    // a too-short input is a typed refusal, not a launch
    let e = syrk_f64_dev(&dev, 4, 5, &y.buf().slice(..), c.buf_mut(), false);
    assert!(matches!(e, Err(GpuError::Layout(_))), "{e:?}");
}

/// General batched product `out_b = L_b · R_b` with a NON-broadcast left operand
/// (standard, stride m·k) and a transposed-standard right operand (stored n×k
/// per matrix, stride n·k), into a sentinel-filled `c` with a GAP between panels
/// (`c_stride = m·n + gap`). Returns the worst err/bound (Higham, depth k) and
/// asserts the gaps keep the sentinel's exact bits.
fn run_general(m: usize, k: usize, n: usize, batch: usize, gap: usize, seed: u64) -> f64 {
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(1 << 26);
    let l = operand(batch * m, k, seed);
    let r = operand(batch * n, k, seed + 1);
    let l_res = DeviceMatrix::<f64>::upload(&dev, &pool, "L", &l.view()).unwrap();
    let r_res = DeviceMatrix::<f64>::upload(&dev, &pool, "R", &r.view()).unwrap();
    let l0 = l.slice(s![0..m, ..]);
    let r0t = r.slice(s![0..n, ..]).reversed_axes();
    let left = dev_batched_left(l_res.buf().slice(..), &l0, batch, m * k).unwrap();
    let right = dev_batched_right(r_res.buf().slice(..), &r0t, batch, n * k).unwrap();
    let c_stride = m * n + gap;
    let len = (batch - 1) * c_stride + m * n;
    let fill = Array2::from_elem((1, len), f64::from_bits(SENTINEL));
    let mut c = DeviceMatrix::<f64>::upload(&dev, &pool, "C", &fill.view()).unwrap();
    let dims = BatchedDims { m, k, n, batch };
    gemm_f64_strided_batched_dev(&dev, dims, &left, &right, c.buf_mut(), c_stride).unwrap();
    let mut got = vec![0.0f64; len];
    dev.stream.memcpy_dtoh(c.buf(), &mut got).unwrap();
    dev.stream.synchronize().unwrap();
    let g = gamma(k, U64);
    let mut worst = 0.0f64;
    let mut written = vec![false; len];
    for b in 0..batch {
        let lb = l.slice(s![b * m..(b + 1) * m, ..]);
        let rbt = r.slice(s![b * n..(b + 1) * n, ..]).reversed_axes();
        let want = lb.dot(&rbt);
        let scale = lb.mapv(f64::abs).dot(&rbt.mapv(f64::abs));
        for i in 0..m {
            for j in 0..n {
                let at = b * c_stride + i * n + j;
                written[at] = true;
                let err = (got[at] - want[(i, j)]).abs();
                let bound = 2.0 * g * scale[(i, j)] * 1.01 + f64::MIN_POSITIVE;
                worst = worst_of(worst, err / bound);
            }
        }
    }
    for (at, w) in written.iter().enumerate() {
        if !w {
            assert_eq!(got[at].to_bits(), SENTINEL, "gap element {at} was written");
        }
    }
    worst
}

#[test]
fn nonbroadcast_left_transposed_right_gapped_output_matches_and_leaves_gaps_alone() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    for (m, k, n, batch, gap) in [
        (3usize, 7usize, 5usize, 4usize, 3usize),
        (1, 1, 1, 3, 2),
        (4, 9, 6, 1, 0),
        (5, 2, 7, 6, 1),
        (6, 33, 4, 5, 0),
    ] {
        let worst = run_general(m, k, n, batch, gap, 21);
        eprintln!("general m={m} k={k} n={n} batch={batch} gap={gap}: worst err/bound {worst:.3e}");
        assert!(
            worst <= 1.0,
            "m={m} k={k} n={n} batch={batch} gap={gap}: {worst:e}"
        );
    }
}

/// Zero-filled operands with the given product shape, device-resident.
struct ZeroOps {
    l: cudarc::driver::CudaSlice<f64>,
    r: cudarc::driver::CudaSlice<f64>,
    lh: Array2<f64>,
    rh: Array2<f64>,
}

fn zero_ops(dev: &ferric_core::gpu::device::Device, m: usize, k: usize, n: usize) -> ZeroOps {
    ZeroOps {
        l: dev.stream.alloc_zeros::<f64>(m * k).unwrap(),
        r: dev.stream.alloc_zeros::<f64>(k * n).unwrap(),
        lh: Array2::zeros((m, k)),
        rh: Array2::zeros((k, n)),
    }
}

fn sentinel_c(
    dev: &ferric_core::gpu::device::Device,
    len: usize,
) -> cudarc::driver::CudaSlice<f64> {
    let host = vec![f64::from_bits(SENTINEL); len];
    dev.stream.clone_htod(&host).unwrap()
}

fn bits(dev: &ferric_core::gpu::device::Device, c: &cudarc::driver::CudaSlice<f64>) -> Vec<u64> {
    let mut v = vec![0.0f64; c.len()];
    dev.stream.memcpy_dtoh(c, &mut v).unwrap();
    dev.stream.synchronize().unwrap();
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn batched_gemm_degenerate_sizes_are_total_and_write_only_their_panels() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    let run = |m: usize, k: usize, n: usize, batch: usize, c_stride: usize, clen: usize| {
        let o = zero_ops(&dev, m, k, n);
        let left = dev_batched_left(o.l.slice(..), &o.lh.view(), batch, 0).unwrap();
        let right = dev_batched_right(o.r.slice(..), &o.rh.view(), batch, 0).unwrap();
        let mut c = sentinel_c(&dev, clen);
        let r = gemm_f64_strided_batched_dev(
            &dev,
            BatchedDims { m, k, n, batch },
            &left,
            &right,
            &mut c,
            c_stride,
        );
        (r, bits(&dev, &c))
    };
    // k = 0: exactly the batch panels are zeroed, gaps keep the sentinel.
    let (m, n, batch, gap) = (3usize, 4usize, 3usize, 2usize);
    let cs = m * n + gap;
    let len = (batch - 1) * cs + m * n;
    let (r, got) = run(m, 0, n, batch, cs, len);
    r.unwrap();
    for (at, &b) in got.iter().enumerate() {
        let in_panel = (0..batch).any(|p| (p * cs..p * cs + m * n).contains(&at));
        let want = if in_panel { 0.0f64.to_bits() } else { SENTINEL };
        assert_eq!(b, want, "k = 0, element {at}");
    }
    // m = 0, n = 0, batch = 0 and all-zero combinations: c is bit-unchanged.
    for (m, k, n, batch) in [
        (0usize, 4usize, 5usize, 3usize),
        (3, 4, 0, 3),
        (3, 4, 5, 0),
        (0, 0, 0, 0),
        (0, 4, 0, 3),
        (3, 0, 5, 0),
        (0, 0, 5, 3),
    ] {
        let (r, got) = run(m, k, n, batch, 40, 17);
        r.unwrap();
        assert!(
            got.iter().all(|&b| b == SENTINEL),
            "m={m} k={k} n={n} batch={batch}"
        );
    }
    // A too-short c is a typed refusal and nothing is written (also for k = 0).
    for k in [0usize, 4] {
        let (m, n, batch) = (3usize, 4usize, 3usize);
        let (r, got) = run(m, k, n, batch, m * n, batch * m * n - 1);
        assert!(matches!(r, Err(GpuError::Layout(_))), "k={k}: {r:?}");
        assert!(got.iter().all(|&b| b == SENTINEL), "k={k}: c was written");
    }
}

#[test]
fn a_left_operand_in_the_right_slot_is_refused_on_the_device_path() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    let (m, k, n, batch) = (4usize, 4usize, 4usize, 2usize);
    let a = zero_ops(&dev, m, k, n);
    let l1 = dev_batched_left(a.l.slice(..), &a.lh.view(), batch, 0).unwrap();
    let l2 = dev_batched_left(a.l.slice(..), &a.lh.view(), batch, 0).unwrap();
    let mut c = sentinel_c(&dev, batch * m * n);
    let e = gemm_f64_strided_batched_dev(
        &dev,
        BatchedDims { m, k, n, batch },
        &l1,
        &l2,
        &mut c,
        m * n,
    );
    assert!(matches!(e, Err(GpuError::Layout(_))), "{e:?}");
    assert!(bits(&dev, &c).iter().all(|&b| b == SENTINEL));
}

/// `syrk_f64_dev` is a raw cuBLAS call: it must work on a thread that never
/// touched the context (a rayon worker, a fresh `std::thread`).
#[test]
fn syrk_runs_on_a_thread_that_never_bound_the_context() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(1 << 24);
    let (n, k) = (9usize, 12usize);
    let y = operand(k, n, 11);
    let a = DeviceMatrix::<f64>::upload(&dev, &pool, "Y", &y.view()).unwrap();
    let mut c =
        DeviceMatrix::<f64>::upload(&dev, &pool, "K", &Array2::zeros((n, n)).view()).unwrap();
    std::thread::scope(|s| {
        s.spawn(|| syrk_f64_dev(&dev, n, k, &a.buf().slice(..), c.buf_mut(), false))
            .join()
            .unwrap()
    })
    .unwrap();
    let mut got = vec![0.0f64; n * n];
    dev.stream.memcpy_dtoh(c.buf(), &mut got).unwrap();
    dev.stream.synchronize().unwrap();
    let want = y.t().dot(&y);
    let s = y.mapv(f64::abs).t().dot(&y.mapv(f64::abs));
    upper_within(&got, &want, &s, k, "fresh thread");
}
