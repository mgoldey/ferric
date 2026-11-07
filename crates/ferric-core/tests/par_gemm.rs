//! [`ferric_core::linalg::par_gemm`] is one GEMM computed in fixed column
//! blocks on the rayon pool.

use ferric_core::linalg::{par_gemm, PAR_GEMM_COL_BLOCK};
use ndarray::Array2;

/// It must agree with a single `general_mat_mul` (to rounding) for a width
/// that is not a multiple of the block, with `beta = 0` and `beta = 1`, plain
/// and transposed operands, and be bit-identical on 1 and 4 threads.
/// FAILS if a block is paired with the wrong columns of `b`, if `beta` or
/// `alpha` is dropped, or if the last partial block is skipped.
#[test]
fn par_gemm_matches_one_gemm_and_ignores_thread_count() {
    let (m, k, n) = (37, 53, 2 * PAR_GEMM_COL_BLOCK + 19);
    let a = Array2::from_shape_fn((m, k), |(i, j)| ((i * 7 + j * 3) as f64).sin());
    let b = Array2::from_shape_fn((k, n), |(i, j)| ((i * 5 + j * 11) as f64).cos());
    let at = a.t().to_owned();
    let bt = b.t().to_owned();
    let c0 = Array2::from_shape_fn((m, n), |(i, j)| (i + 2 * j) as f64 * 1e-2);
    let max_diff =
        |x: &Array2<f64>, y: &Array2<f64>| (x - y).iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    for &(alpha, beta) in &[(1.0, 0.0), (-0.75, 1.0)] {
        let mut want = c0.clone();
        ndarray::linalg::general_mat_mul(alpha, &a, &b, beta, &mut want);
        let mut got = c0.clone();
        par_gemm(alpha, &a.view(), &b.view(), beta, &mut got);
        assert!(max_diff(&got, &want) < 1e-12, "plain, beta {beta}");
        let mut got_t = c0.clone();
        par_gemm(alpha, &at.t(), &bt.t(), beta, &mut got_t);
        assert!(max_diff(&got_t, &want) < 1e-12, "transposed, beta {beta}");
    }
    let run_on = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let mut c = c0.clone();
        pool.install(|| par_gemm(0.5, &a.view(), &bt.t(), 1.0, &mut c));
        c
    };
    assert_eq!(
        run_on(1),
        run_on(4),
        "par_gemm must not depend on the thread count"
    );
}
