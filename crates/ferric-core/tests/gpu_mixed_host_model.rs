//! The §3.2 error model on the HOST twin of the mixed GEMM (OpenBLAS sgemm
//! panels + f64 accumulation). No device; runs in every CI configuration.
//!
//! Bound for one element of variant (b), compared against an f64 k-blocked
//! reference: (2u32 + γ_b(u32) + γ_{⌈k/b⌉}(u64) + γ_k(u64)) · (|A||B|)_ij,
//! u32 = 2^-24, u64 = 2^-53 (Higham, Accuracy and Stability, 2nd ed., §3.1/3.5,
//! applied term by term to the f32 products and the two accumulations).
//!
//! Separation (derived, two sides; positive operands in [0,1), 256x8192x256,
//! rms of e_ij = |got - ref| / (|A||B|)_ij, OPENBLAS_NUM_THREADS = 1 and 6):
//!   defect ABSENT  (b = 128 vs itself):                      ratio 1.0 by definition
//!   defect PRESENT (one panel == plain sgemm + f64 convert):  r_a / r_b = 4.5 (1 thread), 4.7 (6 threads)
//!   committed SEPARATION = floor_1sf(sqrt(1.0 * 4.5)) = 2
//! (Zero-mean operands give only 1.6: OpenBLAS sgemm blocks k internally and
//! cancellation hides the f32 accumulator, so they cannot carry this gate.)
use ferric_core::gpu::mixed_host::{gamma, gemm_f32_f64acc_host, mixed_error_factor, U64};
use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, ArrayView2};

/// Derived in the module docstring (sqrt of the measured defect-side ratio 4.5).
const SEPARATION: f64 = 2.0;

fn operand(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut s = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    })
}

/// Operands in [0, 1): all products are positive, so |A||B| = AB and an f32
/// accumulator cannot hide behind cancellation (zero-mean data separates plain
/// sgemm from the panelled twin by only ~1.6x at k = 8192, because OpenBLAS
/// sgemm already blocks its k loop internally; positive data gives ~4.5x).
fn positive_operand(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    operand(rows, cols, seed).mapv(|x| x + 0.5)
}

fn reference(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> Array2<f64> {
    let mut c = Array2::zeros((a.nrows(), b.ncols()));
    general_mat_mul(1.0, a, b, 0.0, &mut c);
    c
}

fn abs_product(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> Array2<f64> {
    let mut c = Array2::zeros((a.nrows(), b.ncols()));
    general_mat_mul(1.0, &a.mapv(f64::abs), &b.mapv(f64::abs), 0.0, &mut c);
    c
}

/// (max_ij e_ij, rms_ij e_ij) with e_ij = |got − ref| / (|A||B|)_ij.
fn normalized_errors(got: &Array2<f64>, r: &Array2<f64>, ab: &Array2<f64>) -> (f64, f64) {
    let mut max = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut n = 0usize;
    for ((g, rr), s) in got.iter().zip(r.iter()).zip(ab.iter()) {
        if *s > 0.0 {
            let e = (g - rr).abs() / s;
            max = max.max(e);
            sum2 += e * e;
            n += 1;
        }
    }
    (max, (sum2 / n.max(1) as f64).sqrt())
}

fn twin(a: &ArrayView2<f64>, b: &ArrayView2<f64>, k_panel: usize) -> Array2<f64> {
    let mut out = Array2::zeros((a.nrows(), b.ncols()));
    gemm_f32_f64acc_host(a, b, &mut out.view_mut(), k_panel);
    out
}

#[test]
fn panelled_host_twin_meets_the_elementwise_bound() {
    for &(m, k, n) in &[
        (64usize, 8192usize, 64usize),
        (128, 2000, 64),
        (301, 1000, 97),
        (7, 5, 3),
    ] {
        let a = operand(m, k, 1);
        let b = operand(k, n, 2);
        let r = reference(&a.view(), &b.view());
        let ab = abs_product(&a.view(), &b.view());
        let got = twin(&a.view(), &b.view(), 128);
        let bound = (mixed_error_factor(k, 128) + gamma(k, U64)) * 1.01;
        let (max, rms) = normalized_errors(&got, &r, &ab);
        eprintln!(
            "{m}x{k}x{n} b=128: max e = {max:.3e} (bound {bound:.3e}, ratio {:.3e}), rms e = {rms:.3e}",
            max / bound
        );
        assert!(max <= bound, "{m}x{k}x{n}: {max:e} > {bound:e}");
    }
}

#[test]
fn panelled_accumulation_separates_from_plain_sgemm_on_the_host() {
    let (m, k, n) = (256usize, 8192usize, 256usize);
    let a = positive_operand(m, k, 3);
    let b = positive_operand(k, n, 4);
    let r = reference(&a.view(), &b.view());
    let ab = abs_product(&a.view(), &b.view());
    let (_, r_b) = normalized_errors(&twin(&a.view(), &b.view(), 128), &r, &ab);
    // One panel of width k IS plain sgemm followed by one f64 conversion: the
    // "f32 accumulator" defect, reconstructed inline through the public API.
    let (max_a, r_a) = normalized_errors(&twin(&a.view(), &b.view(), k), &r, &ab);
    let bound_a = (mixed_error_factor(k, k) + gamma(k, U64)) * 1.01;
    eprintln!(
        "rms b=128: {r_b:.3e}; rms plain sgemm: {r_a:.3e}; ratio {:.2}",
        r_a / r_b
    );
    assert!(
        max_a <= bound_a,
        "plain sgemm violates even its own bound: {max_a:e} > {bound_a:e}"
    );
    assert!(
        r_a / r_b >= SEPARATION,
        "plain sgemm (ratio {:.2}) is not separated from the panelled twin at SEPARATION {SEPARATION}",
        r_a / r_b
    );
}

#[test]
fn a_panel_at_or_above_k_is_one_panel_bit_for_bit() {
    let a = operand(40, 300, 5);
    let b = operand(300, 24, 6);
    let x = twin(&a.view(), &b.view(), 300);
    let y = twin(&a.view(), &b.view(), 10_000);
    assert!(x
        .iter()
        .zip(y.iter())
        .all(|(p, q)| p.to_bits() == q.to_bits()));
}

#[test]
fn zero_k_is_all_zeros() {
    let a = Array2::<f64>::zeros((3, 0));
    let b = Array2::<f64>::zeros((0, 4));
    let out = twin(&a.view(), &b.view(), 128);
    assert!(out.iter().all(|&v| v == 0.0));
}
