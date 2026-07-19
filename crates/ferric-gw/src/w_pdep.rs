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

/// Re-dress the PDEP eigenpotentials from physical aux basis (cube-export
/// convention) back to V^{-1/2}-dressed basis.
///
/// `eigenpotentials_phys[(P, α)] = Σ_Q V^{-1/2}_{PQ} V_α^Q_dressed`
/// so `V_α^Q_dressed = Σ_P V^{+1/2}_{QP} eigenpotentials_phys[(P, α)]`.
/// V^{+1/2} = L (Cholesky factor of V), since V = L L^T and
/// V^{-1/2} = solve_lower(L, I).
pub fn redress_eigenpotentials(
    v_inv_sqrt: &Array2<f64>,
    eigenpotentials_phys: &Array2<f64>,
) -> Result<Array2<f64>, FerricError> {
    // PDEP stores `eigenpotentials = v_inv_sqrt · V_dressed` (lib.rs line
    // 391). To recover V_dressed we just invert v_inv_sqrt (it is lower
    // triangular as produced by `solve_triangular(L, I)`), so instead of a
    // general LU inverse we solve the triangular system
    // `v_inv_sqrt · v_sqrt_factor = I` directly (`dtrtrs`, not `dgetrf`) —
    // mirrors the analogous `cholesky_inverse_sqrt` idiom in mo_b.rs.
    //
    // Call-path proof: called from top-level GW entry points (lib.rs:213,
    // :285) and from the serial evGW outer loops (sigma.rs:503,
    // u_sigma.rs:260/:312 — plain `for` loops, not inside any par_iter; see
    // sigma.rs::run_evgw). Also reached via bse.rs's top-level
    // `redress_with_check` calls. `opt_in_blas_threads()` defaults to 1 and
    // self-guards to 1 if ever invoked from a rayon worker.
    use ndarray_linalg::{Diag, SolveTriangular, UPLO};
    with_blas_threads(opt_in_blas_threads(), || {
        let eye = Array2::<f64>::eye(v_inv_sqrt.nrows());
        let v_sqrt_factor = v_inv_sqrt
            .solve_triangular(UPLO::Lower, Diag::NonUnit, &eye)
            .map_err(|e| FerricError::General(format!("triangular solve(v_inv_sqrt) failed: {e}")))?;
        Ok(v_sqrt_factor.dot(eigenpotentials_phys))
    })
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
// inv()+GEMM in redress_eigenpotentials lives in tests/blas_raise_identity.rs,
// a dedicated integration-test binary: raising the process-global OpenBLAS
// thread count inside the parallel lib test binary races other BLAS-doing
// tests in the same process (observed corruption, not hypothetical).
