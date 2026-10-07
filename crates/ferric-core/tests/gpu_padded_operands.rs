#![cfg(feature = "gpu")]
//! `G = B_blockᵀ · B_tail` computed from sub-blocks of ONE resident row-major
//! matrix (leading dimension wider than either block) through
//! `dev_left_padded` / `dev_right_padded` + `gemm_f64_dev`, against the host
//! `dot`. The product has depth k = `naux` (> the 128 panel width, so several
//! k-panels with `k_step = ld` are exercised) and every column offset in the
//! sweep is a different base pointer.
//!
//! Bound (derived, no machine constants): both sides are evaluations of the
//! same inner product in some summation order, so each satisfies
//! |ĉ − c| ≤ γ_k Σ_p |a_p b_p| with γ_k = k·u/(1 − k·u), u = 2⁻⁵³ (Higham,
//! ASNA Eq. 3.5); their difference is within 2·γ_k·Σ|a_p b_p|. A wrong
//! descriptor (wrong ld/k_step/op) changes the product by O(1)·Σ|ab|-scale
//! terms, many orders above this.
use ferric_core::gpu::device::{device, GpuError};
use ferric_core::gpu::gemm::{dev_left_padded, dev_right_padded, gemm_f64_dev};
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
fn padded_column_blocks_of_one_resident_matrix_match_the_host_product() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(1 << 28);
    let (naux, nocc, nvir) = (300usize, 4usize, 6usize);
    let nov = nocc * nvir;
    let b = operand(naux, nov, 7);
    let resident = DeviceMatrix::<f64>::upload(&dev, &pool, "padded test B", &b.view()).unwrap();
    let u = 2.0f64.powi(-53);
    let gamma = naux as f64 * u / (1.0 - naux as f64 * u);
    for i in 0..nocc {
        let off = i * nvir;
        let ntail = nov - off;
        let b_i = b.slice(s![.., off..off + nvir]);
        let b_tail = b.slice(s![.., off..]);
        let left = dev_left_padded(resident.buf().slice(off..), &b_i.t()).unwrap();
        let right = dev_right_padded(resident.buf().slice(off..), &b_tail).unwrap();
        let mut c = dev.stream.alloc_zeros::<f64>(nvir * ntail).unwrap();
        gemm_f64_dev(&dev, nvir, naux, ntail, &left, &right, &mut c, 128).unwrap();
        let mut got = vec![0.0f64; nvir * ntail];
        dev.stream.memcpy_dtoh(&c, &mut got).unwrap();
        dev.stream.synchronize().unwrap();
        let want = b_i.t().dot(&b_tail);
        let abs_b_i = b_i.mapv(f64::abs);
        let abs_tail = b_tail.mapv(f64::abs);
        let scale = abs_b_i.t().dot(&abs_tail);
        let mut worst_ratio = 0.0f64;
        for a in 0..nvir {
            for t in 0..ntail {
                let err = (got[a * ntail + t] - want[(a, t)]).abs();
                let bound = 2.0 * gamma * scale[(a, t)];
                assert!(
                    err <= bound,
                    "i={i} ({a},{t}): err {err:e} > bound {bound:e}"
                );
                worst_ratio = worst_ratio.max(err / bound);
            }
        }
        eprintln!("i={i}: worst err/bound {worst_ratio:.3e}");
    }
}

#[test]
fn a_view_shorter_than_the_block_extent_is_refused_at_construction() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(1 << 24);
    let (naux, nov, nvir) = (40usize, 12usize, 3usize);
    let b = operand(naux, nov, 9);
    let resident = DeviceMatrix::<f64>::upload(&dev, &pool, "short view B", &b.view()).unwrap();
    let b_i = b.slice(s![.., 0..nvir]);
    let b_tail = b.slice(s![.., 0..]);
    // The left block's last element is at (naux-1)·nov + nvir - 1; a view one
    // element short of it never becomes an operand.
    let short = (naux - 1) * nov + nvir - 1;
    let left = dev_left_padded(resident.buf().slice(..short), &b_i.t());
    assert!(matches!(left, Err(GpuError::Layout(_))), "{:?}", left.err());
    // The tail block reaches (naux-1)·nov + nov - 1; one short is refused too.
    let right_view = resident.buf().slice(..(naux - 1) * nov + nov - 1);
    let right = dev_right_padded(right_view, &b_tail);
    assert!(
        matches!(right, Err(GpuError::Layout(_))),
        "{:?}",
        right.err()
    );
    // Exactly long enough is accepted.
    assert!(dev_left_padded(resident.buf().slice(..short + 1), &b_i.t()).is_ok());
}

/// `left (m x k) · right (k x n)` where `left` is a standard column block
/// (Left-N, ld = k + pad) and `right` is the transpose of a column block
/// (Right-T, ld = k + pad): the two layouts the RI-MP2 shape does not use.
/// Same bound as the first test (2·γ_k·Σ|a_p b_p|); `kb` is the panel width,
/// so a small `kb` exercises panels past the first with `k_step = 1`.
fn left_n_right_t(m: usize, k: usize, n: usize, pad: usize, kb: usize, seed: u64) {
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(1 << 26);
    let a_full = operand(m, k + pad, seed);
    let b_full = operand(n, k + pad, seed + 1);
    let a_res = DeviceMatrix::<f64>::upload(&dev, &pool, "Left-N block", &a_full.view()).unwrap();
    let b_res = DeviceMatrix::<f64>::upload(&dev, &pool, "Right-T block", &b_full.view()).unwrap();
    let a = a_full.slice(s![.., ..k]);
    let bt = b_full.slice(s![.., ..k]);
    let left = dev_left_padded(a_res.buf().slice(..), &a).unwrap();
    let right = dev_right_padded(b_res.buf().slice(..), &bt.t()).unwrap();
    let mut c = dev.stream.alloc_zeros::<f64>(m * n).unwrap();
    gemm_f64_dev(&dev, m, k, n, &left, &right, &mut c, kb).unwrap();
    let mut got = vec![0.0f64; m * n];
    dev.stream.memcpy_dtoh(&c, &mut got).unwrap();
    dev.stream.synchronize().unwrap();
    let want = a.dot(&bt.t());
    let scale = a.mapv(f64::abs).dot(&bt.t().mapv(f64::abs));
    let u = 2.0f64.powi(-53);
    let gamma = k as f64 * u / (1.0 - k as f64 * u);
    for r in 0..m {
        for t in 0..n {
            let err = (got[r * n + t] - want[(r, t)]).abs();
            assert!(
                err <= 2.0 * gamma * scale[(r, t)] + f64::MIN_POSITIVE,
                "{m}x{k}x{n} pad {pad} kb {kb} ({r},{t}): {err:e}"
            );
        }
    }
}

#[test]
fn left_n_and_right_t_padded_blocks_match_the_host_product() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    // k = 1, a single row/column (ndarray stores stride 0 for a length-1
    // axis) and a depth past several panels at kb = 1 and 3
    for (m, k, n, pad, kb) in [
        (5usize, 40usize, 7usize, 3usize, 128usize),
        (5, 40, 7, 3, 1),
        (5, 40, 7, 3, 3),
        (6, 1, 4, 2, 128),
        (1, 9, 4, 2, 3),
        (4, 9, 1, 2, 3),
        (1, 1, 1, 1, 128),
    ] {
        left_n_right_t(m, k, n, pad, kb, 31);
    }
}

#[test]
fn left_n_and_right_t_views_one_element_short_or_in_the_wrong_slot_are_refused() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if skip() {
        return;
    }
    let dev = device(0).unwrap();
    let pool = DevicePool::with_capacity_bytes(1 << 24);
    let (m, k, n, pad) = (4usize, 9usize, 5usize, 3usize);
    let a_full = operand(m, k + pad, 3);
    let b_full = operand(n, k + pad, 4);
    let a_res = DeviceMatrix::<f64>::upload(&dev, &pool, "A", &a_full.view()).unwrap();
    let b_res = DeviceMatrix::<f64>::upload(&dev, &pool, "B", &b_full.view()).unwrap();
    let a = a_full.slice(s![.., ..k]);
    let bt = b_full.slice(s![.., ..k]);
    let a_ext = (m - 1) * (k + pad) + k;
    let b_ext = (n - 1) * (k + pad) + k;
    let e = dev_left_padded(a_res.buf().slice(..a_ext - 1), &a);
    assert!(matches!(e, Err(GpuError::Layout(_))), "{:?}", e.err());
    let e = dev_right_padded(b_res.buf().slice(..b_ext - 1), &bt.t());
    assert!(matches!(e, Err(GpuError::Layout(_))), "{:?}", e.err());
    let left = dev_left_padded(a_res.buf().slice(..a_ext), &a).unwrap();
    let right = dev_right_padded(b_res.buf().slice(..b_ext), &bt.t()).unwrap();
    let mut c = dev.stream.alloc_zeros::<f64>(m * n).unwrap();
    // a padded operand of the wrong shape for the product, and in the wrong slot
    let e = gemm_f64_dev(&dev, m + 1, k, n, &left, &right, &mut c, 128);
    assert!(matches!(e, Err(GpuError::Layout(_))), "{e:?}");
    let e = gemm_f64_dev(&dev, m, k, n, &right, &left, &mut c, 128);
    assert!(matches!(e, Err(GpuError::Layout(_))), "{e:?}");
    gemm_f64_dev(&dev, m, k, n, &left, &right, &mut c, 128).unwrap();
}
