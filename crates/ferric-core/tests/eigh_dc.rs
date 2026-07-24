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

// ─────────────────────── logdet_lu (dgetrf_) ───────────────────────
//
// `logdet_lu` is the LU-based replacement for `Σ_α ln λ_α` in the RPA
// correlation-energy trace-log. The gate below is the identity that makes
// the substitution legal: ln det(A) ≡ Σ_α ln λ_α for symmetric positive-
// definite A. Anything wrong in the dgetrf_ calling convention (lda, pivot
// sign accounting, column-major diagonal indexing) breaks it.

/// Deterministic symmetric **positive-definite** matrix: AᵀA + n·I. The `+nI`
/// shift keeps the spectrum comfortably away from zero so `ln λ` is well
/// conditioned, mirroring a physical dielectric ε(iω) ≻ 0.
fn spd_matrix(n: usize, seed: u64) -> Array2<f64> {
    let mut state: u64 = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut m = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            m[[i, j]] = next();
        }
    }
    let mut a = m.t().dot(&m);
    for i in 0..n {
        a[[i, i]] += n as f64;
    }
    a
}

#[test]
fn logdet_lu_matches_sum_log_eigenvalues_spd() {
    use ferric_core::linalg::logdet_lu;
    for (&n, seed) in [3usize, 8, 37, 120].iter().zip([1u64, 2, 3, 4]) {
        let a = spd_matrix(n, 0x9e3779b97f4a7c15 ^ seed);

        let evals = eigvalsh_dc(&a, Uplo::Upper).expect("eigvalsh_dc failed");
        // Guard the premise of the test itself: this really is SPD.
        assert!(evals[0] > 0.0, "n={n}: test matrix is not positive definite (λ_min={})", evals[0]);
        let sum_log: f64 = evals.iter().map(|&l| l.ln()).sum();

        let logdet = logdet_lu(&a).expect("logdet_lu failed");

        let rel = (logdet - sum_log).abs() / sum_log.abs().max(1.0);
        assert!(
            rel < 1e-10,
            "n={n}: logdet_lu={logdet:.16} vs Σ ln λ={sum_log:.16} (rel {rel:.3e})"
        );
    }
}

#[test]
fn logdet_lu_matches_the_full_rpa_trace_log_summand() {
    use ferric_core::linalg::logdet_lu;
    // The exact quantity the RPA quadrature needs at one frequency:
    //   Σ_α [ln λ_α + (1 − λ_α)]  ≡  ln det(ε) + (n − tr ε).
    for &n in &[5usize, 44] {
        let a = spd_matrix(n, 0xdeadbeef ^ n as u64);

        let evals = eigvalsh_dc(&a, Uplo::Upper).unwrap();
        let eig_way: f64 = evals.iter().map(|&l| l.ln() + (1.0 - l)).sum();

        let trace: f64 = (0..n).map(|i| a[[i, i]]).sum();
        let lu_way = logdet_lu(&a).unwrap() + (n as f64 - trace);

        let rel = (lu_way - eig_way).abs() / eig_way.abs().max(1.0);
        assert!(rel < 1e-10, "n={n}: LU={lu_way:.16} vs eig={eig_way:.16} (rel {rel:.3e})");
    }
}

#[test]
fn logdet_lu_errs_on_negative_determinant() {
    use ferric_core::linalg::logdet_lu;
    // diag(1, -1): det = -1 < 0 → ln det undefined over the reals. Must be a
    // clean Err, never NaN (repo convention: a pathological dielectric errors).
    let a = ndarray::array![[1.0f64, 0.0], [0.0, -1.0]];
    let r = logdet_lu(&a);
    assert!(r.is_err(), "negative determinant must error, got {r:?}");
}

#[test]
fn logdet_lu_errs_on_singular_matrix() {
    use ferric_core::linalg::logdet_lu;
    // Rank-deficient: row 1 = 2 × row 0.
    let a = ndarray::array![[1.0f64, 2.0], [2.0, 4.0]];
    let r = logdet_lu(&a);
    assert!(r.is_err(), "exactly singular matrix must error, got {r:?}");
}

#[test]
fn logdet_lu_errs_on_non_square() {
    use ferric_core::linalg::logdet_lu;
    let a = Array2::<f64>::zeros((3, 4));
    assert!(logdet_lu(&a).is_err());
}

#[test]
fn logdet_lu_sign_accounting_under_heavy_pivoting() {
    use ferric_core::linalg::logdet_lu;
    // A permutation-like matrix forces dgetrf_ to swap on nearly every column,
    // which is what actually exercises the (−1)^(#swaps) bookkeeping. An
    // anti-diagonal of ones is the extreme case: det = sign of the reversal
    // permutation = (−1)^(n(n−1)/2), i.e. +1 for n ≡ 0,1 (mod 4).
    //
    // n=4: reversal permutation has 6 inversions → det = +1 → ln det = 0.
    // n=5: 10 inversions → det = +1 → ln det = 0.
    // (n=2,3 give det = −1, which must ERROR — covered separately below.)
    for &n in &[4usize, 5] {
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            a[[i, n - 1 - i]] = 1.0;
        }
        let ld = logdet_lu(&a).unwrap_or_else(|e| panic!("n={n} anti-diagonal should have det=+1: {e}"));
        assert!(ld.abs() < 1e-13, "n={n}: expected ln det = 0 (det=+1), got {ld}");
    }
    // n=2 and n=3 reversals have odd parity → det = −1 → must be a clean Err,
    // proving the sign really is tracked (not silently dropped via abs()).
    for &n in &[2usize, 3] {
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            a[[i, n - 1 - i]] = 1.0;
        }
        assert!(
            logdet_lu(&a).is_err(),
            "n={n}: det = −1 must error, not silently return ln|det| = 0"
        );
    }
}

#[test]
fn logdet_lu_matches_scaled_identity_closed_form() {
    use ferric_core::linalg::logdet_lu;
    // det(cI_n) = c^n → ln det = n ln c. Independent of LAPACK entirely.
    for &(n, c) in &[(3usize, 2.0f64), (7, 0.5), (16, 1.25)] {
        let a = Array2::<f64>::eye(n) * c;
        let got = logdet_lu(&a).unwrap();
        let want = n as f64 * c.ln();
        assert!((got - want).abs() < 1e-12, "n={n} c={c}: got {got}, want {want}");
    }
}

#[test]
fn logdet_lu_transpose_invariant() {
    use ferric_core::linalg::logdet_lu;
    // det(Aᵀ) = det(A) — this is exactly why `logdet_lu` may hand LAPACK the
    // row-major buffer without transposing. Checked on a NON-symmetric matrix
    // (for a symmetric one the property is vacuous).
    let a = ndarray::array![
        [4.0f64, 1.0, 0.5],
        [0.2, 3.0, -1.0],
        [0.1, 0.7, 5.0]
    ];
    let at = a.t().to_owned();
    let la = logdet_lu(&a).unwrap();
    let lat = logdet_lu(&at).unwrap();
    assert!((la - lat).abs() < 1e-13, "logdet(A)={la} vs logdet(Aᵀ)={lat}");
}
