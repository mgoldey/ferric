//! RPA correlation energy integration via trace-log formula.
//!
//! E_c^RPA = (1/2π) Σ_k w_k Σ_α [ln(λ_α(iω_k)) + (1 − λ_α(iω_k))].

use ndarray::Array2;

/// E_c^RPA = (1/2π) Σ_k w_k Σ_α [ln(λ_α(iω_k)) + (1 − λ_α(iω_k))].
///
/// Computes RPA correlation energy from eigenvalues of the dielectric matrix
/// evaluated at imaginary frequencies (quadrature points).
///
/// # Arguments
/// * `quad_weights` - Gauss-Legendre or other quadrature weights w_k, length N_quad
/// * `eigenvalues_freq` - Eigenvalues λ_α(iω_k) of shape (N_quad, M)
///
/// # Returns
/// RPA correlation energy in Hartree.
pub fn rpa_correlation_energy(
    quad_weights: &[f64],
    eigenvalues_freq: &Array2<f64>,
) -> f64 {
    let n_quad = quad_weights.len();
    assert_eq!(eigenvalues_freq.nrows(), n_quad);

    let mut e_c = 0.0f64;
    for (k, &wk) in quad_weights.iter().enumerate() {
        let row = eigenvalues_freq.row(k);
        let contrib: f64 = row
            .iter()
            .map(|&lam| lam.ln() + (1.0 - lam))
            .sum();
        e_c += wk * contrib;
    }
    e_c / (2.0 * std::f64::consts::PI)
}

/// Evaluate λ_α(iω_k) for each quadrature point given converged eigenpotentials.
///
/// Uses the diagonal of the projected dielectric matrix as λ_α(iω_k).
///
/// # Arguments
/// * `eigenvectors` - Converged eigenvectors V_α from PDEP, shape (naux, M)
/// * `b_ov` - RI-MO tensor B^P_ia, shape (naux, nocc*nvir)
/// * `eps_occ` - Occupied orbital energies
/// * `eps_vir` - Virtual orbital energies
/// * `quad_freqs` - Imaginary quadrature frequencies ω_k
///
/// # Returns
/// Array2 of shape (N_quad, M) containing eigenvalues λ_α(iω_k).
pub fn eval_eigenvalues_at_frequencies(
    eigenvectors: &Array2<f64>,
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    quad_freqs: &[f64],
) -> Array2<f64> {
    use crate::sternheimer::dielectric_matrix;
    use ndarray_linalg::{Eigh, UPLO};

    let n_quad = quad_freqs.len();
    let m = eigenvectors.ncols();
    let mut eigenvalues_freq = Array2::zeros((n_quad, m));

    for (k, &omega) in quad_freqs.iter().enumerate() {
        let eps_proj = dielectric_matrix(eigenvectors, b_ov, eps_occ, eps_vir, omega);
        // Diagonalize at each frequency — eigenvectors at ω=0 don't diagonalize ε̃(iω)
        let (evals, _) = eps_proj.eigh(UPLO::Upper).expect("dielectric eigh failed");
        for alpha in 0..m {
            eigenvalues_freq[(k, alpha)] = evals[alpha];
        }
    }
    eigenvalues_freq
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_log_two_eigenvalues() {
        // E_c = (1/2π) Σ_k w_k Σ_α [ln(λ_α(iω_k)) + (1 − λ_α(iω_k))]
        // 1 quad point w=1, λ1=2.0, λ2=0.5:
        // contrib = (ln(2)+1-2) + (ln(0.5)+1-0.5) = (0.693-1) + (-0.693+0.5) = -0.5
        // E_c = -0.5 / (2π)
        let quad_weights = vec![1.0f64];
        let eigenvalues_freq = ndarray::array![[2.0f64, 0.5]]; // (1 quad, 2 eigs)
        let e_c = rpa_correlation_energy(&quad_weights, &eigenvalues_freq);
        let expected = -0.5 / (2.0 * std::f64::consts::PI);
        assert!(
            (e_c - expected).abs() < 1e-10,
            "E_c = {} expected {}",
            e_c,
            expected
        );
    }
}
