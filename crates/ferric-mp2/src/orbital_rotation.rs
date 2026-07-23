//! Shared orbital-rotation machinery for orbital-optimized MP2 (`oo_rimp2`,
//! `u_oo_rimp2`).
//!
//! # Cayley convention (unified 2026-07-22)
//!
//! Both the closed-shell (`oo_rimp2`) and unrestricted (`u_oo_rimp2`) drivers
//! previously carried their *own* `cayley_rotation`, with OPPOSITE conventions:
//! the closed-shell file used `U = (I + κ/2)⁻¹(I − κ/2)` while the unrestricted
//! file used `U = (I − κ/2)⁻¹(I + κ/2)`. These are inverse (transpose)
//! rotations of one another — `U_cs(κ) = U_u(−κ) = U_u(κ)ᵀ` — and each file's
//! gradient sign convention was independently matched to its own Cayley, which
//! is why both passed their finite-difference gradient tests despite the
//! mismatch.
//!
//! This module fixes the single canonical convention as the unrestricted one:
//!
//! ```text
//!   U(κ) = (I − κ/2)⁻¹ (I + κ/2) ≈ exp(κ)   (for antisymmetric κ)
//! ```
//!
//! with the paired sign conventions `g = +∂E/∂κ` and descent step
//! `κ = −g/(gap + μ)`. Under this convention the first-order rotation applied to
//! the coefficients is `C_new = C·U ≈ C·(I + κ)`, i.e. the generator is `+κ`, so
//! moving κ opposite to the gradient of E decreases E.
//!
//! `oo_rimp2` was migrated onto this convention by flipping the sign of its
//! analytic orbital gradient (both the `−4·F_ai` HF/Brillouin term and the
//! `−2·grad_ck + denom_term` MP2 response assembly) so that its
//! gradient/step/rotation triple remains self-consistent — a change that is a
//! pure identity for the physical orbital trajectory, the DIIS extrapolation,
//! and the backtracking (a global sign flip of every stored DIIS error vector
//! leaves the constrained-least-squares coefficients unchanged; the flipped κ
//! composed with the flipped Cayley sense reproduces the identical rotation).

use ferric_core::FerricError;
use ndarray::Array2;
use ndarray_linalg::{Factorize, Solve};

/// Cayley transform `U = (I − κ/2)⁻¹(I + κ/2)` for antisymmetric `κ`.
///
/// `U` is exactly orthogonal (`UᵀU = I`) for any antisymmetric `κ`, and equals
/// `exp(κ)` through second order in `κ`. This is the single canonical
/// orbital-rotation used by both `oo_rimp2` and `u_oo_rimp2`; see the module
/// docs for the sign conventions it pairs with.
///
/// `(I − κ/2)` is factorized ONCE (`getrf`, O(n³)) and reused for every
/// column's triangular solve (`getrs`, O(n²) each) — the previous per-column
/// `lhs.solve(..)` re-factorized the SAME matrix for every one of the `n`
/// columns (O(n⁴) total). Same LAPACK routines under the hood
/// (`ndarray_linalg::Solve::solve` itself calls `getrf`+`getrs` per call), so
/// this is bit-identical, not merely equivalent — see
/// `factorize_once_matches_percolumn_solve_bitwise` for a direct check.
pub fn cayley_rotation(kappa: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    let n = kappa.nrows();
    let eye = Array2::<f64>::eye(n);
    let half_k = 0.5 * kappa;
    let lhs = &eye - &half_k; // I − κ/2
    let rhs = &eye + &half_k; // I + κ/2

    // Factorize (I − κ/2) once (LU, getrf), then reuse the factorization for
    // every column's solve (getrs) instead of re-factorizing per column.
    let lhs_factorized = lhs
        .factorize()
        .map_err(|e| FerricError::Lapack(format!("Cayley LU factorize: {e}")))?;

    let mut u = Array2::zeros((n, n));
    for col in 0..n {
        let rhs_col = rhs.column(col).to_owned();
        let u_col = lhs_factorized
            .solve(&rhs_col)
            .map_err(|e| FerricError::Lapack(format!("Cayley solve col {col}: {e}")))?;
        u.column_mut(col).assign(&u_col);
    }
    Ok(u)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic 4×4 antisymmetric κ (fixed seed values, no RNG) filled
    /// into the strict upper triangle and mirrored with a sign flip.
    fn seeded_antisym_4x4(scale: f64) -> Array2<f64> {
        // Six independent strict-upper-triangle entries for a 4×4 antisym matrix.
        let vals = [0.31, -0.42, 0.17, 0.23, -0.11, 0.28];
        let mut kappa = Array2::<f64>::zeros((4, 4));
        let mut idx = 0;
        for i in 0..4 {
            for j in (i + 1)..4 {
                let v = scale * vals[idx];
                kappa[(i, j)] = v;
                kappa[(j, i)] = -v;
                idx += 1;
            }
        }
        kappa
    }

    /// The shared Cayley rotation is exactly orthogonal (`UᵀU = I` to 1e-12) for
    /// a non-trivial antisymmetric κ.
    #[test]
    fn shared_cayley_is_orthogonal() {
        let kappa = seeded_antisym_4x4(1.0);
        let u = cayley_rotation(&kappa).unwrap();
        let utu = u.t().dot(&u);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (utu[(i, j)] - expected).abs() < 1e-12,
                    "UᵀU[{i},{j}] = {} (expected {expected})",
                    utu[(i, j)]
                );
            }
        }
    }

    /// At small κ the shared Cayley matches a truncated `exp(κ)` reference
    /// (built by a deterministic Taylor series, no RNG). The Cayley agrees with
    /// exp(κ) through O(κ²), so the residual is the O(κ³) truncation — at
    /// ‖κ‖ ~ 0.06 this is comfortably below 1e-4, while UᵀU = I holds exactly.
    #[test]
    fn shared_cayley_matches_exp_reference_small_kappa() {
        let kappa = seeded_antisym_4x4(0.05); // small: |entries| ≲ 0.021

        // Deterministic exp(κ) via truncated Taylor series I + κ + κ²/2! + …
        // (16 terms is far past convergence at this magnitude).
        let n = kappa.nrows();
        let mut exp_ref = Array2::<f64>::eye(n);
        let mut term = Array2::<f64>::eye(n);
        for k in 1..16 {
            term = term.dot(&kappa) / (k as f64);
            exp_ref = &exp_ref + &term;
        }

        let u = cayley_rotation(&kappa).unwrap();
        let maxdiff = (&u - &exp_ref)
            .iter()
            .map(|v| v.abs())
            .fold(0.0f64, f64::max);
        assert!(
            maxdiff < 1e-4,
            "shared Cayley vs exp(κ) reference at small κ: maxdiff={maxdiff:.3e}"
        );

        // Orthogonality is exact regardless of the small-κ exp comparison.
        let utu = u.t().dot(&u);
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((utu[(i, j)] - expected).abs() < 1e-12);
            }
        }
    }

    /// Deterministic 6×6 antisymmetric κ (fixed values, no RNG), same
    /// generation scheme as `seeded_antisym_4x4` extended to 6×6 (15
    /// independent strict-upper-triangle entries).
    fn seeded_antisym_6x6(scale: f64) -> Array2<f64> {
        let vals = [
            0.31, -0.42, 0.17, 0.23, -0.11, 0.28, 0.05, -0.19, 0.37, -0.08,
            0.14, -0.26, 0.09, 0.21, -0.33,
        ];
        let mut kappa = Array2::<f64>::zeros((6, 6));
        let mut idx = 0;
        for i in 0..6 {
            for j in (i + 1)..6 {
                let v = scale * vals[idx];
                kappa[(i, j)] = v;
                kappa[(j, i)] = -v;
                idx += 1;
            }
        }
        kappa
    }

    /// `cayley_rotation`'s factorize-once path must match a plain per-column
    /// `Solve::solve` (the pre-M4 approach: re-factorize `(I − κ/2)` on every
    /// column) to `to_bits()` EQUALITY on a deterministic 6×6 antisymmetric κ
    /// — both use the same underlying LAPACK `getrf`+`getrs` routines, so
    /// this is a bitwise identity check, not merely a numerical-tolerance one.
    /// If LAPACK's factorize-once and per-column-refactorize paths ever
    /// diverge (e.g. a different pivoting sequence), this test reports the
    /// max ULP diff per element instead of silently asserting a loose bound.
    #[test]
    fn factorize_once_matches_percolumn_solve_bitwise() {
        use ndarray_linalg::Factorize;

        let kappa = seeded_antisym_6x6(0.37);
        let n = kappa.nrows();
        let eye = Array2::<f64>::eye(n);
        let half_k = 0.5 * &kappa;
        let lhs = &eye - &half_k;
        let rhs = &eye + &half_k;

        // Reference: plain per-column solve, re-factorizing `lhs` every column
        // (the pre-M4 implementation `cayley_rotation` used to run).
        let mut u_percolumn = Array2::<f64>::zeros((n, n));
        for col in 0..n {
            let rhs_col = rhs.column(col).to_owned();
            let u_col = lhs.solve(&rhs_col).unwrap();
            u_percolumn.column_mut(col).assign(&u_col);
        }

        // Factorize-once reference (mirrors the new cayley_rotation body).
        let lhs_factorized = lhs.factorize().unwrap();
        let mut u_factorized = Array2::<f64>::zeros((n, n));
        for col in 0..n {
            let rhs_col = rhs.column(col).to_owned();
            let u_col = lhs_factorized.solve(&rhs_col).unwrap();
            u_factorized.column_mut(col).assign(&u_col);
        }

        // The public function itself must match the factorize-once reference
        // exactly (sanity: cayley_rotation really does use factorize-once now).
        let u_public = cayley_rotation(&kappa).unwrap();

        let mut max_ulp_diff: u64 = 0;
        let mut all_bitwise_equal = true;
        for i in 0..n {
            for j in 0..n {
                let a = u_percolumn[(i, j)];
                let b = u_factorized[(i, j)];
                let c = u_public[(i, j)];
                if a.to_bits() != b.to_bits() {
                    all_bitwise_equal = false;
                    let ulp = a.to_bits().abs_diff(b.to_bits());
                    max_ulp_diff = max_ulp_diff.max(ulp);
                }
                assert_eq!(
                    b.to_bits(),
                    c.to_bits(),
                    "cayley_rotation[{i},{j}] does not match the factorize-once reference \
                     it should be identical to: {b} vs {c}"
                );
            }
        }
        assert!(
            all_bitwise_equal,
            "factorize-once vs per-column-refactorize solve differ: max ULP diff = {max_ulp_diff} \
             (expected bitwise identical — same getrf+getrs LAPACK routines either way)"
        );
    }
}
