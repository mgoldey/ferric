//! The complex BLAS/LAPACK calls the k-point code depends on, checked against
//! definitions rather than against another library.
//!
//! The Gamma-point path is real and the k-point path is complex
//! (`Complex64` Fock matrices diagonalized with `Eigh`, products through
//! ndarray's complex GEMM). A complex-only ABI mismatch with the linked
//! OpenBLAS (for example, how a complex return value is passed) would leave
//! every real-valued test green and break only k-point SCF. These
//! assertions isolate that layer:
//! - `eigh` on a column-major Hermitian matrix (the layout `kscf::eigh_herm`
//!   uses): residual `A V - V Λ` and `V^H V = I`;
//! - complex matrix-matrix and matrix-vector `dot` against naive loops;
//! - the complex vector inner product against a naive sum.

use ndarray::{Array1, Array2, ShapeBuilder};
use ndarray_linalg::{Eigh, UPLO};
use num_complex::Complex64;

fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn complex_matrix(n: usize, m: usize, seed: u64) -> Array2<Complex64> {
    let mut next = lcg(seed);
    Array2::from_shape_fn((n, m), |_| Complex64::new(next(), next()))
}

fn hermitian(n: usize, seed: u64) -> Array2<Complex64> {
    let m = complex_matrix(n, n, seed);
    let mut a = &m + &m.t().mapv(|z| z.conj());
    for i in 0..n {
        a[[i, i]] += Complex64::new(i as f64, 0.0);
    }
    a
}

fn naive_matmul(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    let (n, k) = a.dim();
    let m = b.ncols();
    Array2::from_shape_fn((n, m), |(i, j)| {
        (0..k).fold(Complex64::new(0.0, 0.0), |s, l| s + a[[i, l]] * b[[l, j]])
    })
}

/// Largest |z|. Panics on a non-finite entry: `f64::max` drops a NaN operand,
/// so an all-NaN result would otherwise reduce to 0 and pass every check.
fn max_abs(a: &Array2<Complex64>) -> f64 {
    finite_max(a.iter().map(|z| z.norm()))
}

fn finite_max(errs: impl Iterator<Item = f64>) -> f64 {
    errs.fold(0.0f64, |m, e| {
        assert!(
            e.is_finite(),
            "non-finite value {e} in a BLAS/LAPACK result"
        );
        m.max(e)
    })
}

#[test]
fn complex_eigh_satisfies_its_definition() {
    for &n in &[2usize, 7, 40] {
        let a = hermitian(n, 0x5eed + n as u64);
        // Column-major, as the production path (kscf::eigh_herm) passes it:
        // ndarray-linalg's eigh on a ROW-MAJOR complex matrix does not return
        // eigenvectors of A on any platform.
        let mut f = Array2::<Complex64>::zeros(a.raw_dim().f());
        f.assign(&a);
        let (w, v) = f.eigh(UPLO::Upper).expect("zheev");
        assert!(
            w.iter().all(|x| x.is_finite()),
            "n={n}: non-finite eigenvalues {w}"
        );
        let lam = Array2::from_diag(&w.mapv(|x| Complex64::new(x, 0.0)));
        let resid = max_abs(&(naive_matmul(&a, &v) - naive_matmul(&v, &lam)));
        let vh = v.t().mapv(|z| z.conj());
        let ortho = max_abs(&(naive_matmul(&vh, &v) - Array2::<Complex64>::eye(n)));
        let scale = max_abs(&a);
        assert!(
            resid < 1e-12 * scale * n as f64,
            "n={n}: |A V - V L| = {resid:e}"
        );
        assert!(ortho < 1e-12 * n as f64, "n={n}: |V^H V - I| = {ortho:e}");
        assert!(
            w.windows(2).into_iter().all(|p| p[0] <= p[1]),
            "n={n}: eigenvalues not ascending"
        );
    }
}

#[test]
fn complex_dot_matches_naive_products() {
    for &(n, k, m) in &[(1usize, 1usize, 1usize), (3, 5, 4), (33, 17, 29)] {
        let a = complex_matrix(n, k, 11 + n as u64);
        let b = complex_matrix(k, m, 23 + m as u64);
        let got = a.dot(&b);
        let err = max_abs(&(&got - &naive_matmul(&a, &b)));
        assert!(err < 1e-12 * k as f64, "gemm {n}x{k}x{m}: {err:e}");

        let x: Array1<Complex64> = b.column(0).to_owned();
        let y = a.dot(&x);
        let want = naive_matmul(&a, &x.clone().insert_axis(ndarray::Axis(1)));
        let err = finite_max(y.iter().zip(want.column(0)).map(|(p, q)| (p - q).norm()));
        assert!(err < 1e-12 * k as f64, "gemv {n}x{k}: {err:e}");

        let u: Array1<Complex64> = a.row(0).to_owned();
        let v: Array1<Complex64> = a.row(n - 1).to_owned();
        let naive = u
            .iter()
            .zip(v.iter())
            .fold(Complex64::new(0.0, 0.0), |s, (p, q)| s + p * q);
        let err = (u.dot(&v) - naive).norm();
        assert!(err < 1e-12 * k as f64, "dotu len {k}: {err:e}");
    }
}
