//! The Cayley orbital rotation `U = (I − κ/2)⁻¹ (I + κ/2)` shared by the
//! second-order SCF steps (RHF/UHF/ROHF Newton, ROHF AH, TRAH) and the
//! best-effort descent in `uhf::rotate_mos` / `cdft_driver::rotate_mos`.
//!
//! For antisymmetric κ, `A = I − κ/2` and `B = I + κ/2` satisfy `A + B = 2I`,
//! so `U = A⁻¹ B = A⁻¹ (2I − A) = 2 A⁻¹ − I`. That needs ONE LU of `A` and its
//! inverse (`getrf` + `getri`, both BLAS3-blocked) and no second product.
//! The former copies ran `a.solve(&b.column(j))` for every column `j`, which
//! re-factorized the same `n × n` matrix `n` times: O(n⁴), ~0.45 s per call at
//! nmo 356. The result is the same unitary to rounding (anchored against the
//! per-column construction in the tests below).

use ferric_core::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ferric_core::FerricError;
use ndarray::Array2;
use ndarray_linalg::Inverse;

/// `U = (I − κ/2)⁻¹ (I + κ/2)` for an antisymmetric `κ` (computed as
/// `2 (I − κ/2)⁻¹ − I`, see the module doc). Exactly orthogonal up to
/// rounding; `= exp(κ)` through second order. The inverse runs at the opt-in
/// BLAS thread count, so a calling thread needs the stack that
/// `ferric_core::blas_threads::init_threading` sets for threads spawned after
/// it (the contract of every `with_blas_threads` site).
pub(crate) fn cayley_unitary(kappa: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    let n = kappa.nrows();
    let mut a = Array2::<f64>::eye(n);
    a.scaled_add(-0.5, kappa);
    let mut u = with_blas_threads(opt_in_blas_threads(), || a.inv())
        .map_err(|e| FerricError::Lapack(format!("Cayley inverse: {e}")))?;
    u *= 2.0;
    for i in 0..n {
        u[(i, i)] -= 1.0;
    }
    Ok(u)
}

/// `C · U` for the Cayley unitary of `κ` (see [`cayley_unitary`]).
pub(crate) fn cayley_rotate(
    c: &Array2<f64>,
    kappa: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    let u = cayley_unitary(kappa)?;
    Ok(with_blas_threads(opt_in_blas_threads(), || c.dot(&u)))
}

/// `C ← C · U`, `U` the Cayley unitary of the antisymmetric `n × n` κ whose
/// only nonzero block is the occ→virt one: `κ[a, i] = k_ov[a − nocc, i]`,
/// `κ[i, a] = −k_ov[a − nocc, i]`.
pub(crate) fn apply_cayley_ov(
    c: &Array2<f64>,
    k_ov: &Array2<f64>,
    nocc: usize,
    n: usize,
) -> Result<Array2<f64>, FerricError> {
    let mut kappa = Array2::<f64>::zeros((n, n));
    for (ir, a) in (nocc..n).enumerate() {
        for i in 0..nocc {
            let v = k_ov[(ir, i)];
            kappa[(a, i)] = v;
            kappa[(i, a)] = -v;
        }
    }
    cayley_rotate(c, &kappa)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_linalg::Solve;

    /// The construction every call site used before: one `solve` (fresh LU)
    /// per column of `I + κ/2`, then `C · U`.
    fn per_column_reference(c: &Array2<f64>, kappa: &Array2<f64>) -> Array2<f64> {
        let n = kappa.nrows();
        let half = 0.5 * kappa;
        let eye = Array2::<f64>::eye(n);
        let a_mat = &eye - &half;
        let b_mat = &eye + &half;
        let mut u = Array2::<f64>::zeros((n, n));
        for col in 0..n {
            let sol = a_mat.solve(&b_mat.column(col).to_owned()).unwrap();
            u.column_mut(col).assign(&sol);
        }
        c.dot(&u)
    }

    /// Deterministic pseudo-random numbers in [-1, 1) (LCG; no RNG crate).
    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn random(nr: usize, nc: usize, scale: f64, seed: &mut u64) -> Array2<f64> {
        Array2::from_shape_fn((nr, nc), |_| scale * lcg(seed))
    }

    /// A random orthonormal-in-S pair: S SPD, C = S^{-1/2}-like via Cholesky
    /// of S so that CᵀSC = I exactly up to rounding.
    fn s_and_orthonormal_c(nbf: usize, seed: &mut u64) -> (Array2<f64>, Array2<f64>) {
        use ndarray_linalg::{Cholesky, UPLO};
        let m = random(nbf, nbf, 1.0, seed);
        let s = m.t().dot(&m) * (0.1 / nbf as f64) + Array2::<f64>::eye(nbf);
        // S = L Lᵀ ⇒ C = L⁻ᵀ gives CᵀSC = L⁻¹ L Lᵀ L⁻ᵀ = I.
        let l = s.cholesky(UPLO::Lower).unwrap();
        let c = l.t().to_owned().inv().unwrap();
        (s, c)
    }

    fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        (a - b).iter().fold(0.0_f64, |m, v| m.max(v.abs()))
    }

    fn orthonormality_error(c: &Array2<f64>, s: &Array2<f64>) -> f64 {
        let n = c.ncols();
        max_abs_diff(&c.t().dot(&s.dot(c)), &Array2::<f64>::eye(n))
    }

    /// Exactness anchor: the factor-once rotation equals the old per-column
    /// one, for a dense antisymmetric κ and for the occ→virt block structure
    /// the Newton steps use (incl. n = 1, nocc = 0 and nocc = n), and keeps
    /// the orbitals S-orthonormal.
    #[test]
    fn factor_once_matches_per_column_cayley() {
        let mut seed = 0x5eed_u64;
        for &(n, scale) in &[(1usize, 0.7), (2, 0.5), (7, 1.3), (40, 0.4), (97, 2.0)] {
            let (s, c) = s_and_orthonormal_c(n, &mut seed);
            // Dense antisymmetric κ (ROHF-style multi-block is a special case).
            let m = random(n, n, scale, &mut seed);
            let kappa = &m - &m.t();
            let new = cayley_rotate(&c, &kappa).unwrap();
            let old = per_column_reference(&c, &kappa);
            let d = max_abs_diff(&new, &old);
            assert!(d < 1e-13, "n={n}: dense κ new vs per-column {d:.2e}");
            let o = orthonormality_error(&new, &s);
            assert!(o < 1e-12, "n={n}: CᵀSC − I = {o:.2e}");

            // occ→virt block (rhf_newton / uhf_newton / trah structure).
            for nocc in [0, n / 3, n] {
                let k_ov = random(n - nocc, nocc, scale, &mut seed);
                let new = apply_cayley_ov(&c, &k_ov, nocc, n).unwrap();
                let mut kappa = Array2::<f64>::zeros((n, n));
                for (ir, a) in (nocc..n).enumerate() {
                    for i in 0..nocc {
                        kappa[(a, i)] = k_ov[(ir, i)];
                        kappa[(i, a)] = -k_ov[(ir, i)];
                    }
                }
                let old = per_column_reference(&c, &kappa);
                let d = max_abs_diff(&new, &old);
                assert!(
                    d < 1e-13,
                    "n={n} nocc={nocc}: ov κ new vs per-column {d:.2e}"
                );
                let o = orthonormality_error(&new, &s);
                assert!(o < 1e-12, "n={n} nocc={nocc}: CᵀSC − I = {o:.2e}");
            }
        }
    }

    /// Non-square C (nbf > nmo, e.g. a lindep-filtered basis): only the MO
    /// dimension enters the Cayley factor.
    #[test]
    fn factor_once_matches_per_column_for_rectangular_c() {
        let mut seed = 0xc0ffee_u64;
        let (nbf, n, nocc) = (23usize, 17usize, 5usize);
        let c = random(nbf, n, 1.0, &mut seed);
        let k_ov = random(n - nocc, nocc, 0.8, &mut seed);
        let new = apply_cayley_ov(&c, &k_ov, nocc, n).unwrap();
        let mut kappa = Array2::<f64>::zeros((n, n));
        for (ir, a) in (nocc..n).enumerate() {
            for i in 0..nocc {
                kappa[(a, i)] = k_ov[(ir, i)];
                kappa[(i, a)] = -k_ov[(ir, i)];
            }
        }
        let d = max_abs_diff(&new, &per_column_reference(&c, &kappa));
        assert!(d < 1e-13, "rectangular C: new vs per-column {d:.2e}");
        assert_eq!(new.dim(), (nbf, n));
    }
}
