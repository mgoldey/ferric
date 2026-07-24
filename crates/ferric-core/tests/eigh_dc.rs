//! Correctness gate for the divide-and-conquer symmetric eigensolver
//! [`ferric_core::linalg::eigh_dc`].
//!
//! The single most important property: `eigh_dc` (dsyevd_) must agree with the
//! QR-algorithm path `ndarray_linalg::Eigh::eigh` (dsyev_) that it replaces, to
//! within tight tolerance, on real symmetric matrices at the sizes ferric
//! actually diagonalizes. A wrong LAPACK calling convention (workspace-size
//! bug, row/column-major mismatch, sign/ordering divergence) would silently
//! corrupt every downstream physics result, so this test is non-negotiable.

use ndarray::Array2;
use ndarray_linalg::{Eigh, UPLO};

use ferric_core::linalg::{eigh_dc, eigvalsh_dc, Uplo};

/// Deterministic symmetric test matrix of size `n`: A = M + Mᵀ with M filled by
/// a simple LCG, plus a diagonal spread so eigenvalues are well separated.
fn sym_matrix(n: usize) -> Array2<f64> {
    let mut state: u64 = 0x9e3779b97f4a7c15;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // map to [-1, 1)
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut m = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            m[[i, j]] = next();
        }
    }
    let mut a = &m + &m.t();
    for i in 0..n {
        a[[i, i]] += 2.0 * i as f64; // spread the spectrum
    }
    a
}

/// Eigenvector columns are only defined up to sign; align each column of `b` to
/// the sign of `a`'s corresponding column before comparing.
fn max_vec_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let n = a.ncols();
    let mut worst = 0.0f64;
    for j in 0..n {
        // pick the sign that minimizes the difference (handle degenerate sign flip)
        let mut dot = 0.0;
        for i in 0..a.nrows() {
            dot += a[[i, j]] * b[[i, j]];
        }
        let s = if dot < 0.0 { -1.0 } else { 1.0 };
        for i in 0..a.nrows() {
            worst = worst.max((a[[i, j]] - s * b[[i, j]]).abs());
        }
    }
    worst
}

#[test]
fn eigh_dc_matches_ndarray_linalg_eigh_upper() {
    // Sizes: tiny, plus a couple basis/aux-scale ones relevant to ferric.
    for &n in &[3usize, 10, 50, 97] {
        let a = sym_matrix(n);

        let (ref_vals, ref_vecs) = a.eigh(UPLO::Upper).expect("reference eigh failed");
        let (dc_vals, dc_vecs) = eigh_dc(&a, Uplo::Upper).expect("eigh_dc failed");

        assert_eq!(dc_vals.len(), n, "eigenvalue count mismatch at n={n}");
        assert_eq!(dc_vecs.dim(), (n, n), "eigenvector shape mismatch at n={n}");

        // Eigenvalues: ascending, bit-close to the QR path.
        for k in 0..n {
            let d = (ref_vals[k] - dc_vals[k]).abs();
            assert!(
                d < 1e-10,
                "eigenvalue {k} differs at n={n}: {} vs {} (|Δ|={d:.3e})",
                ref_vals[k],
                dc_vals[k]
            );
        }

        // Eigenvectors: same span, sign-aligned, bit-close.
        let ref_vecs_owned = ref_vecs.to_owned();
        let diff = max_vec_abs_diff(&ref_vecs_owned, &dc_vecs);
        assert!(diff < 1e-9, "eigenvector max abs diff {diff:.3e} too large at n={n}");
    }
}

#[test]
fn eigh_dc_reconstructs_matrix() {
    // Independent physics-style check: A ≈ V diag(λ) Vᵀ, and V is orthonormal.
    let n = 64;
    let a = sym_matrix(n);
    let (vals, vecs) = eigh_dc(&a, Uplo::Upper).expect("eigh_dc failed");

    // Orthonormality: VᵀV ≈ I
    let vtv = vecs.t().dot(&vecs);
    for i in 0..n {
        for j in 0..n {
            let expect = if i == j { 1.0 } else { 0.0 };
            assert!(
                (vtv[[i, j]] - expect).abs() < 1e-10,
                "VᵀV not identity at ({i},{j}): {}",
                vtv[[i, j]]
            );
        }
    }

    // Reconstruction: V diag(λ) Vᵀ ≈ A
    let mut vd = vecs.clone();
    for j in 0..n {
        for i in 0..n {
            vd[[i, j]] *= vals[j];
        }
    }
    let recon = vd.dot(&vecs.t());
    let mut worst = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            worst = worst.max((recon[[i, j]] - a[[i, j]]).abs());
        }
    }
    assert!(worst < 1e-9, "reconstruction error {worst:.3e} too large");
}

#[test]
fn eigvalsh_dc_matches_eigh_eigenvalues() {
    // Eigenvalue-only path (jobz='N') must return exactly the same eigenvalues
    // as the full solver and as ndarray_linalg's eigh, at RPA dielectric sizes.
    for &n in &[3usize, 25, 80] {
        let a = sym_matrix(n);
        let (ref_vals, _) = a.eigh(UPLO::Upper).expect("reference eigh failed");
        let vals = eigvalsh_dc(&a, Uplo::Upper).expect("eigvalsh_dc failed");
        assert_eq!(vals.len(), n);
        for k in 0..n {
            let d = (ref_vals[k] - vals[k]).abs();
            assert!(d < 1e-10, "eigenvalue {k} differs at n={n} (|Δ|={d:.3e})");
        }
        // Consistency with the full solver's eigenvalues too.
        let (full_vals, _) = eigh_dc(&a, Uplo::Upper).unwrap();
        for k in 0..n {
            assert!((full_vals[k] - vals[k]).abs() < 1e-12);
        }
    }
}

#[test]
fn eigh_dc_lower_matches_upper_for_symmetric() {
    // For a genuinely symmetric matrix, reading the Upper vs Lower triangle
    // must give the same spectrum (guards against a UPLO plumbing bug).
    let n = 40;
    let a = sym_matrix(n);
    let (vu, _) = eigh_dc(&a, Uplo::Upper).unwrap();
    let (vl, _) = eigh_dc(&a, Uplo::Lower).unwrap();
    for k in 0..n {
        assert!((vu[k] - vl[k]).abs() < 1e-10, "Upper/Lower eigenvalue {k} differ");
    }
}
