//! RI-dRPA sanity checks: full-basis correlation energy and eigenvalue diagnostics.
//!
//! The RI-dRPA approximation evaluates RPA on the full RI-auxiliary basis without
//! eigenpotential truncation. Used as a sanity check against PDEP-RPA.

use ndarray::Array2;
use ndarray_linalg::{Eigh, UPLO};
use rayon::prelude::*;
use ferric_core::FerricError;

/// Compute RI-dRPA eigenvalues of ε̃(iω) = I − χ₀(iω) in the full RI basis.
///
/// The dielectric matrix is computed directly without truncation to eigenpotentials.
/// Returns eigenvalues in descending order.
///
/// # Arguments
/// * `b_ov` - RI-MO tensor B^P_ia, shape (naux, nocc*nvir)
/// * `eps_occ` - Occupied orbital energies
/// * `eps_vir` - Virtual orbital energies
/// * `omega` - Imaginary frequency ω (typically positive for iω)
///
/// # Returns
/// Vector of eigenvalues sorted in descending order.
pub fn ri_drpa_eigenvalues(
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    omega: f64,
) -> Result<Vec<f64>, FerricError> {
    let naux = b_ov.nrows();
    let nocc = eps_occ.len();
    let nvir = eps_vir.len();
    let nov = nocc * nvir;
    assert_eq!(b_ov.shape()[1], nov);

    // Build Π = 4 Σ_{ia} e_ia/(ω²+e_ia²) B^P B^Q (= −χ₀, RHF: factor 4 = 2 spin × 2 orb)
    let mut chi0: Array2<f64> = Array2::zeros((naux, naux));
    for (i, &eps_i) in eps_occ.iter().enumerate() {
        for (a, &eps_a) in eps_vir.iter().enumerate() {
            let ia = i * nvir + a;
            let e_ia = eps_a - eps_i;
            let scale = 4.0 * e_ia / (omega * omega + e_ia * e_ia);
            let col = b_ov.column(ia);
            for p in 0..naux {
                for q in 0..naux {
                    chi0[(p, q)] += scale * col[p] * col[q];
                }
            }
        }
    }

    // ε̃ = I − χ₀ = I + Π (Π = −χ₀, positive); eigenvalues 1 + μ ≥ 1
    for p in 0..naux {
        chi0[(p, p)] += 1.0;
    }

    let (evals, _) = chi0
        .eigh(UPLO::Upper)
        .map_err(|e| FerricError::General(format!("RI-dRPA diagonalization failed: {e}")))?;

    let mut result: Vec<f64> = evals.to_vec();
    result.sort_by(|a, b| b.total_cmp(a));
    Ok(result)
}

/// Compute RI-dRPA correlation energy via trace-log on full-basis eigenvalues.
///
/// E_c^dRPA = (1/2π) Σ_k w_k [ln det(I − Π(iω_k)) + tr(Π(iω_k))]
///         where Π = 4 Σ_{ia} e_ia/(ω²+e_ia²) B B^T (= −χ₀, RHF).
///
/// Uses ln|det| (real part of log determinant) so the formula stays well-defined
/// even when (I − Π) has negative eigenvalues — matching PySCF's behavior.
pub fn ri_drpa_energy(
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    quad_freqs: &[f64],
    quad_weights: &[f64],
) -> Result<f64, FerricError> {
    // Each quadrature point is fully independent — parallelize over frequencies.
    let contribs: Result<Vec<f64>, FerricError> = quad_freqs
        .par_iter()
        .zip(quad_weights.par_iter())
        .map(|(&omega, &wk)| {
            let evals = ri_drpa_eigenvalues(b_ov, eps_occ, eps_vir, omega)?;
            // ln det(I + Π) − tr(Π) where Π = −χ₀ ≥ 0
            // = Σ_α [ln(λ_α) + (1 − λ_α)] with λ_α = 1 + μ_α ≥ 1
            let contrib: f64 = evals
                .iter()
                .map(|&lam| lam.ln() + (1.0 - lam))
                .sum();
            Ok(wk * contrib)
        })
        .collect();
    let e_c: f64 = contribs?.iter().sum();
    Ok(e_c / (2.0 * std::f64::consts::PI))
}
