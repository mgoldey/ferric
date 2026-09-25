//! Low-rank screened-Coulomb representation from PDEP eigenpairs.
//!
//! On the V^{-1/2}-dressed auxiliary basis,
//!   ε̃(iω) = I + Π(iω)        (dressed)
//!   W̃(iω) = ε̃⁻¹(iω) − I       (dynamic part; v_eff = I here)
//! Because v in the dressed basis is the identity, the *full* screened
//! interaction is W̃_total = ε̃⁻¹, whose dynamic part above carries the
//! frequency dependence. The static (= bare) component contributes Σ_x
//! and is handled separately.
//!
//! With PDEP eigenpairs {λ_α(iω), V_α^dressed}:
//!   W̃_d(iω)_{PQ} = Σ_α [1/λ_α(iω) − 1]  V_α^P  V_α^Q
//!
//! For COHSEX we just need W̃_d(0). For G0W0 we need W̃_d at every iω_k
//! and an analytic continuation to real ω.

use ferric_core::FerricError;
use ferric_integrals::blas_threads::{opt_in_blas_threads, with_blas_threads};
use ndarray::Array2;

/// Re-dress the PDEP eigenpotentials from physical aux-basis coefficients
/// (`PdepRpaResult::eigenpotentials`, the cube/NPZ convention) into the
/// dressed basis of the caller's tensor B̃ = F·(Q|mn), F = `v_inv_sqrt`.
///
/// A physical coefficient vector c is the aux function Σ_P c_P χ_P. Its
/// projection onto a pair density is (Q|mn)ᵀc = B̃ᵀ F⁻ᵀ c, so the dressed
/// vector is `d = F⁻ᵀ c`, i.e. the solution of `Fᵀ d = c`. PDEP builds
/// c = F_pdepᵀ u (u = its dressed eigenvectors), hence for F = F_pdep this
/// returns u exactly (to roundoff), and for any other factorization of the
/// same metric it returns the vector whose B̃-projection equals PDEP's.
///
/// Supported F:
/// * lower triangular (the Cholesky L⁻¹ every GW/BSE `mo_b` builds, and the
///   Coulomb/erfc PDEP factor): Fᵀ is upper triangular → one `dtrtrs`;
/// * symmetric (the erf/terf eigh V^{-1/2}, possibly with dropped null
///   modes): Fᵀ = F → eigh pseudo-inverse, dropping F's null modes.
///
/// Anything else is an error rather than a silent wrong answer.
pub fn redress_eigenpotentials(
    v_inv_sqrt: &Array2<f64>,
    eigenpotentials_phys: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    // Call-path proof: called from top-level GW entry points (lib.rs run_gw /
    // run_u_gw) and from the serial evGW outer loops (sigma.rs, u_sigma.rs —
    // plain `for` loops, not inside any par_iter; see sigma.rs::run_evgw).
    // Also reached via bse.rs's top-level `redress_with_check` calls.
    // `opt_in_blas_threads()` defaults to 1 and self-guards to 1 if ever
    // invoked from a rayon worker.
    check_redress_shapes(v_inv_sqrt, eigenpotentials_phys)?;
    let (scale, max_upper, max_asym) = factor_shape(v_inv_sqrt);
    with_blas_threads(opt_in_blas_threads(), || {
        if max_upper <= 1e-12 * scale {
            transposed_triangular_solve(v_inv_sqrt, eigenpotentials_phys)
        } else if max_asym <= 1e-10 * scale {
            symmetric_pseudo_solve(v_inv_sqrt, eigenpotentials_phys)
        } else {
            Err(FerricError::General(format!(
                "redress_eigenpotentials: v_inv_sqrt is neither lower triangular \
                 (max|upper| = {max_upper:.3e}) nor symmetric (max|F-Fᵀ| = {max_asym:.3e}), \
                 scale {scale:.3e}"
            )))
        }
    })
}

fn check_redress_shapes(f: &Array2<f64>, c: &Array2<f64>) -> Result<(), FerricError> {
    let n = f.nrows();
    if f.ncols() == n && c.nrows() == n {
        return Ok(());
    }
    Err(FerricError::General(format!(
        "redress_eigenpotentials: v_inv_sqrt is {}x{}, eigenpotentials have {} rows",
        f.nrows(),
        f.ncols(),
        c.nrows()
    )))
}

/// Lower-triangular F: solve the upper-triangular system Fᵀ d = c.
fn transposed_triangular_solve(
    f: &Array2<f64>,
    c: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    use ndarray_linalg::{Diag, SolveTriangular, UPLO};
    let ft = f.t().as_standard_layout().into_owned();
    ft.solve_triangular(UPLO::Upper, Diag::NonUnit, c)
        .map_err(|e| FerricError::General(format!("triangular solve(v_inv_sqrtᵀ) failed: {e}")))
}

/// `(max|F|, max|strict upper triangle|, max|F − Fᵀ|)` — how
/// [`redress_eigenpotentials`] tells a Cholesky factor from an eigh factor.
fn factor_shape(f: &Array2<f64>) -> (f64, f64, f64) {
    let n = f.nrows();
    let scale = f.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    let mut max_upper = 0.0_f64;
    let mut max_asym = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let up = f[(i, j)];
            max_upper = max_upper.max(up.abs());
            max_asym = max_asym.max((up - f[(j, i)]).abs());
        }
    }
    (scale, max_upper, max_asym)
}

/// Symmetric F (eigh V^{-1/2}): d = F⁺ c. The eigh factor's dropped null modes
/// are exact zeros up to roundoff (~eps·|F|); kept modes are ≥ λ_max(V)^{-1/2},
/// many decades above the 1e-9·|F| cut.
fn symmetric_pseudo_solve(f: &Array2<f64>, c: &Array2<f64>) -> Result<Array2<f64>, FerricError> {
    use ndarray_linalg::{Eigh, UPLO};
    let (s, u) = f
        .eigh(UPLO::Lower)
        .map_err(|e| FerricError::General(format!("eigh(v_inv_sqrt) failed: {e}")))?;
    let s_max = s.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    let mut ut_c = u.t().dot(c);
    for (k, mut row) in ut_c.rows_mut().into_iter().enumerate() {
        let inv = if s[k].abs() > 1e-9 * s_max {
            1.0 / s[k]
        } else {
            0.0
        };
        row.mapv_inplace(|x| x * inv);
    }
    Ok(u.dot(&ut_c))
}

/// Build the dressed eigenpotentials *and* a consistency check: for the
/// returned `V_dressed`, the column norms should be 1 (Davidson returns
/// orthonormal vectors in the dressed basis). This is checked separately
/// in tests.
pub fn redress_with_check(
    v_inv_sqrt: &Array2<f64>,
    eigenpotentials_phys: &Array2<f64>,
) -> Result<(Array2<f64>, f64), FerricError> {
    let v_dressed = redress_eigenpotentials(v_inv_sqrt, eigenpotentials_phys)?;
    let m = v_dressed.ncols();
    let mut max_dev = 0.0_f64;
    for a in 0..m {
        let col = v_dressed.column(a);
        let n2: f64 = col.iter().map(|&x| x * x).sum();
        max_dev = max_dev.max((n2 - 1.0).abs());
    }
    Ok((v_dressed, max_dev))
}

/// Inverse-dielectric weights w_α(iω) = 1/λ_α(iω) − 1, shape (N_quad, M).
pub fn inverse_dielectric_weights(eigenvalues_freq: &Array2<f64>) -> Array2<f64> {
    let mut w = eigenvalues_freq.clone();
    for v in w.iter_mut() {
        *v = 1.0 / *v - 1.0;
    }
    w
}

/// Static screened weights w_α(0) = 1/λ_α(0) − 1, length M.
pub fn static_weights(eigenvalues_static: &[f64]) -> Vec<f64> {
    eigenvalues_static.iter().map(|&l| 1.0 / l - 1.0).collect()
}

// NOTE: the real-env FERRIC_BLAS_THREADS=2 identity test for the wrapped
// triangular solve in redress_eigenpotentials lives in tests/blas_raise_identity.rs,
// a dedicated integration-test binary: raising the process-global OpenBLAS
// thread count inside the parallel lib test binary races other BLAS-doing
// tests in the same process (observed corruption, not hypothetical).

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_linalg::{Cholesky, Eigh, SolveTriangular, UPLO};

    /// Deterministic SPD "metric" V (n×n), its Cholesky L⁻¹, its symmetric
    /// V^{-1/2}, and 3 orthonormal "dressed eigenvectors" u.
    fn fixture() -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
        let n = 6;
        let a = Array2::from_shape_fn((n, n), |(i, j)| {
            ((i * 7 + j * 3) % 11) as f64 / 11.0 - 0.4 + if i == j { 1.5 } else { 0.0 }
        });
        let v = a.t().dot(&a);
        let l = v.cholesky(UPLO::Lower).unwrap();
        let l_inv = l
            .solve_triangular(
                UPLO::Lower,
                ndarray_linalg::Diag::NonUnit,
                &Array2::<f64>::eye(n),
            )
            .unwrap();
        let (w, q) = v.eigh(UPLO::Lower).unwrap();
        let mut qs = q.clone();
        for (k, mut col) in qs.columns_mut().into_iter().enumerate() {
            col.mapv_inplace(|x| x / w[k].sqrt());
        }
        let s = qs.dot(&q.t());
        let h = Array2::from_shape_fn((n, n), |(i, j)| {
            ((i + 2 * j) % 5) as f64 + ((j + 2 * i) % 5) as f64
        });
        let (_, hv) = h.eigh(UPLO::Lower).unwrap();
        let u = hv.slice(ndarray::s![.., ..3]).to_owned();
        (v, l_inv, s, u)
    }

    fn max_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
    }

    /// Same factor both sides: redress(F, Fᵀu) = u, triangular and symmetric.
    #[test]
    fn redress_inverts_the_transposed_factor() {
        let (_v, l_inv, s, u) = fixture();
        let c_l = l_inv.t().dot(&u);
        let d_l = redress_eigenpotentials(&l_inv, &c_l).unwrap();
        assert!(
            max_diff(&d_l, &u) < 1e-12,
            "Cholesky: {:e}",
            max_diff(&d_l, &u)
        );
        let c_s = s.t().dot(&u);
        let d_s = redress_eigenpotentials(&s, &c_s).unwrap();
        assert!(
            max_diff(&d_s, &u) < 1e-12,
            "symmetric: {:e}",
            max_diff(&d_s, &u)
        );
    }

    /// Cross factorization (the erf-PDEP → Cholesky-mo_b situation): PDEP
    /// built c = Sᵀu with the symmetric factor S; GW dresses with L⁻¹. The
    /// requirement is that the projection onto any pair density is preserved:
    /// B_Lᵀ d = (Q|mn)ᵀ L⁻ᵀ d must equal B_Sᵀ u = (Q|mn)ᵀ Sᵀ u = (Q|mn)ᵀ c for
    /// every (Q|mn), i.e. L⁻ᵀ d = c. The pre-fix redress (d = L·c, solving
    /// with the untransposed factor) satisfies L⁻¹d = c instead and fails
    /// this by O(1). (The correct d = Lᵀ·S·u keeps unit column norms because
    /// Lᵀ·S is orthogonal; the pre-fix L·S·u does not, so only
    /// `redress_with_check`'s norm deviation — logged, never asserted, by
    /// run_gw — would have hinted at it.)
    #[test]
    fn redress_preserves_the_physical_projection_across_factorizations() {
        let (_v, l_inv, s, u) = fixture();
        let c = s.t().dot(&u);
        let d = redress_eigenpotentials(&l_inv, &c).unwrap();
        let back = l_inv.t().dot(&d);
        assert!(
            max_diff(&back, &c) < 1e-12,
            "L⁻ᵀd != c: {:e}",
            max_diff(&back, &c)
        );
        // The pre-fix map, for contrast: its B-projection is wrong.
        let d_old = l_inv
            .solve_triangular(UPLO::Lower, ndarray_linalg::Diag::NonUnit, &c)
            .unwrap();
        let back_old = l_inv.t().dot(&d_old);
        assert!(
            max_diff(&back_old, &c) > 1e-3,
            "fixture too degenerate to distinguish the transpose"
        );
    }

    #[test]
    fn redress_rejects_a_general_nonsymmetric_nontriangular_factor() {
        let f = ndarray::array![[1.0, 0.5], [0.2, 1.0]];
        let c = Array2::<f64>::eye(2);
        assert!(redress_eigenpotentials(&f, &c).is_err());
    }
}
