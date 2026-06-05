//! RPA correlation energy integration via trace-log formula.
//!
//! E_c^RPA = (1/2π) Σ_k w_k Σ_α [ln(λ_α(iω_k)) + (1 − λ_α(iω_k))].

use crate::channel::RpaChannel;
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
/// Laplace-separable variant of [`eval_eigenvalues_at_frequencies`].
///
/// Uses the χ₀ kernel from [`crate::laplace_chi0`] in place of the dense
/// `dielectric_matrix`. The output shape and convention are identical.
pub fn eval_eigenvalues_at_frequencies_laplace(
    eigenvectors: &Array2<f64>,
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    quad_freqs: &[f64],
    laplace: &ferric_quadrature::LaplaceQuadrature,
) -> Array2<f64> {
    use crate::laplace_chi0::dielectric_matrix_laplace_into;
    use ndarray::Array2;
    use ndarray_linalg::{Eigh, UPLO};
    use rayon::prelude::*;

    let n_quad = quad_freqs.len();
    let m = eigenvectors.ncols();
    let nov = eps_occ.len() * eps_vir.len();

    // Each rayon worker keeps its own (rhs_scaled, out) scratch and reuses it
    // across the frequencies it processes, instead of allocating an (m×nov) and
    // (m×m) buffer per quadrature point.
    let rows: Vec<Vec<f64>> = quad_freqs
        .par_iter()
        .map_init(
            || (Array2::<f64>::zeros((m, nov)), Array2::<f64>::zeros((m, m))),
            |(rhs_scaled, out), &omega| {
                dielectric_matrix_laplace_into(
                    eigenvectors, b_ov, eps_occ, eps_vir, omega, laplace, rhs_scaled, out,
                );
                let (evals, _) = out.eigh(UPLO::Upper).expect("dielectric eigh failed");
                evals.to_vec()
            },
        )
        .collect();

    let mut eigenvalues_freq = Array2::zeros((n_quad, m));
    for (k, row) in rows.into_iter().enumerate() {
        for (alpha, val) in row.into_iter().enumerate() {
            eigenvalues_freq[(k, alpha)] = val;
        }
    }
    eigenvalues_freq
}

/// Unrestricted variant: per-frequency eigenvalues of ε̃_U = I + Π_α + Π_β.
///
/// Diagonalizes the projected unrestricted dielectric at each ω_k.
/// Output shape (N_quad, M) is identical to the closed-shell path so
/// `rpa_correlation_energy` consumes it unchanged.
pub fn eval_eigenvalues_at_frequencies_unrestricted(
    eigenvectors: &Array2<f64>,
    chan_a: &RpaChannel,
    chan_b: &RpaChannel,
    quad_freqs: &[f64],
) -> Array2<f64> {
    use crate::sternheimer::dielectric_matrix_unrestricted;
    use ndarray_linalg::{Eigh, UPLO};
    use rayon::prelude::*;

    let n_quad = quad_freqs.len();
    let m = eigenvectors.ncols();

    let rows: Vec<Vec<f64>> = quad_freqs
        .par_iter()
        .map(|&omega| {
            let eps_proj = dielectric_matrix_unrestricted(
                eigenvectors, chan_a, chan_b, omega,
            );
            let (evals, _) = eps_proj.eigh(UPLO::Upper).expect("unrestricted dielectric eigh failed");
            evals.to_vec()
        })
        .collect();

    let mut out = Array2::zeros((n_quad, m));
    for (k, row) in rows.into_iter().enumerate() {
        for (alpha, val) in row.into_iter().enumerate() {
            out[(k, alpha)] = val;
        }
    }
    out
}

/// Laplace + unrestricted variant of [`eval_eigenvalues_at_frequencies`].
///
/// Builds `ε̃ = I + Π_α + Π_β` at each ω_k using the Laplace-separable
/// kernel per spin, diagonalizes, returns eigenvalue tensor.
pub fn eval_eigenvalues_at_frequencies_laplace_unrestricted(
    eigenvectors: &Array2<f64>,
    chan_a: &RpaChannel,
    laplace_a: &ferric_quadrature::LaplaceQuadrature,
    chan_b: &RpaChannel,
    laplace_b: &ferric_quadrature::LaplaceQuadrature,
    quad_freqs: &[f64],
) -> Array2<f64> {
    use crate::laplace_chi0::dielectric_matrix_laplace_unrestricted;
    use ndarray_linalg::{Eigh, UPLO};
    use rayon::prelude::*;

    let n_quad = quad_freqs.len();
    let m = eigenvectors.ncols();

    let rows: Vec<Vec<f64>> = quad_freqs
        .par_iter()
        .map(|&omega| {
            let eps_proj = dielectric_matrix_laplace_unrestricted(
                eigenvectors,
                chan_a, laplace_a,
                chan_b, laplace_b,
                omega,
            );
            let (evals, _) = eps_proj.eigh(UPLO::Upper).expect("U-Laplace dielectric eigh failed");
            evals.to_vec()
        })
        .collect();

    let mut out = Array2::zeros((n_quad, m));
    for (k, row) in rows.into_iter().enumerate() {
        for (alpha, val) in row.into_iter().enumerate() {
            out[(k, alpha)] = val;
        }
    }
    out
}

pub fn eval_eigenvalues_at_frequencies(
    eigenvectors: &Array2<f64>,
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    quad_freqs: &[f64],
) -> Array2<f64> {
    use crate::sternheimer::{build_scale_factors, dielectric_matrix_from_projection};
    use ndarray_linalg::{Eigh, UPLO};
    use rayon::prelude::*;

    let n_quad = quad_freqs.len();
    let m = eigenvectors.ncols();

    // The projection y = Vᵀ·B_ov is frequency-independent — compute it ONCE
    // instead of recomputing the (m × nov) GEMM at every quadrature point. Each
    // ω then only does the cheap column scaling + DSYRK.
    let y = eigenvectors.t().dot(b_ov);

    // Each quadrature point is fully independent — parallelize over frequencies.
    let rows: Vec<Vec<f64>> = quad_freqs
        .par_iter()
        .map(|&omega| {
            let scale = build_scale_factors(eps_occ, eps_vir, omega);
            let eps_proj = dielectric_matrix_from_projection(&y, &scale);
            // Diagonalize at each frequency — eigenvectors at ω=0 don't diagonalize ε̃(iω)
            let (evals, _) = eps_proj.eigh(UPLO::Upper).expect("dielectric eigh failed");
            evals.to_vec()
        })
        .collect();

    let mut eigenvalues_freq = Array2::zeros((n_quad, m));
    for (k, row) in rows.into_iter().enumerate() {
        for (alpha, val) in row.into_iter().enumerate() {
            eigenvalues_freq[(k, alpha)] = val;
        }
    }
    eigenvalues_freq
}

/// Per-frequency *dynamic inverse-dielectric* matrices in the PDEP basis.
///
/// Returns `W̃_d(iω_k) = ε̃_proj(iω_k)⁻¹ − I` (shape M×M) for each quadrature
/// frequency, where `ε̃_proj = Uᵀ ε̃(iω) U` is the projected dielectric in the
/// static PDEP eigenvector basis `U` (same basis used to project B̃ in GW).
///
/// This is required by the GW self-energy: the scalar `eigenvalues_freq` tensor
/// (eigenvalues of ε̃(iω) in its *own* ω-dependent eigenbasis) is only valid for
/// basis-invariant traces like the RPA correlation energy. The GW Σ_c contracts
/// the *matrix* W̃_d(iω) with specific B̃_{mn} vectors, so it needs eigenvalues
/// and eigenvectors paired consistently — i.e. the full matrix in the fixed PDEP
/// basis, not diagonal weights computed in a per-ω rotated basis.
pub fn eval_inv_dielectric_matrices(
    eigenvectors: &Array2<f64>,
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    quad_freqs: &[f64],
) -> Vec<Array2<f64>> {
    use crate::sternheimer::{build_scale_factors, dielectric_matrix_from_projection};
    use ndarray_linalg::Inverse;
    use rayon::prelude::*;

    let m = eigenvectors.ncols();
    let y = eigenvectors.t().dot(b_ov);

    quad_freqs
        .par_iter()
        .map(|&omega| {
            let scale = build_scale_factors(eps_occ, eps_vir, omega);
            let eps_proj = dielectric_matrix_from_projection(&y, &scale);
            let mut winv = eps_proj
                .inv()
                .expect("PDEP-basis dielectric inversion failed");
            // Subtract identity → dynamic part W̃_d = ε̃⁻¹ − I.
            for d in 0..m {
                winv[(d, d)] -= 1.0;
            }
            winv
        })
        .collect()
}

/// Unrestricted variant of [`eval_inv_dielectric_matrices`]: full per-frequency
/// `ε̃_U(iω)⁻¹ − I` (M×M) in the fixed PDEP basis, where ε̃_U = I + Π_α + Π_β.
pub fn eval_inv_dielectric_matrices_unrestricted(
    eigenvectors: &Array2<f64>,
    chan_a: &RpaChannel,
    chan_b: &RpaChannel,
    quad_freqs: &[f64],
) -> Vec<Array2<f64>> {
    use crate::sternheimer::dielectric_matrix_unrestricted;
    use ndarray_linalg::Inverse;
    use rayon::prelude::*;

    let m = eigenvectors.ncols();
    quad_freqs
        .par_iter()
        .map(|&omega| {
            let eps_proj = dielectric_matrix_unrestricted(eigenvectors, chan_a, chan_b, omega);
            let mut winv = eps_proj
                .inv()
                .expect("U PDEP-basis dielectric inversion failed");
            for d in 0..m {
                winv[(d, d)] -= 1.0;
            }
            winv
        })
        .collect()
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
