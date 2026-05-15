//! RI-dRPA sanity checks: full-basis correlation energy and eigenvalue diagnostics.
//!
//! The RI-dRPA approximation evaluates RPA on the full RI-auxiliary basis without
//! eigenpotential truncation. Used as a sanity check against PDEP-RPA.

use ndarray::Array2;
use ndarray_linalg::{Eigh, UPLO};
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

    // Store +2 Σ_{ia} B^P_ia B^Q_ia / gap so that I + chi0 = I − χ₀ ≥ I
    let mut chi0: Array2<f64> = Array2::zeros((naux, naux));
    for (i, &eps_i) in eps_occ.iter().enumerate() {
        for (a, &eps_a) in eps_vir.iter().enumerate() {
            let ia = i * nvir + a;
            let gap = eps_a - eps_i + omega;
            let scale = 2.0 / gap;
            let col = b_ov.column(ia);
            for p in 0..naux {
                for q in 0..naux {
                    chi0[(p, q)] += scale * col[p] * col[q];
                }
            }
        }
    }

    // ε̃ = I − χ₀
    for p in 0..naux {
        chi0[(p, p)] += 1.0;
    }

    let (evals, _) = chi0
        .eigh(UPLO::Upper)
        .map_err(|e| FerricError::General(format!("RI-dRPA diagonalization failed: {e}")))?;

    let mut result: Vec<f64> = evals.to_vec();
    result.sort_by(|a, b| b.partial_cmp(a).unwrap());
    Ok(result)
}

/// Compute RI-dRPA correlation energy via trace-log on full-basis eigenvalues.
///
/// E_c^dRPA = (1/2π) Σ_k w_k Σ_P [ln(λ_P(iω_k)) + (1 − λ_P(iω_k))].
///
/// This is the full-RI version without eigenpotential truncation, used as a reference check.
///
/// # Arguments
/// * `b_ov` - RI-MO tensor B^P_ia, shape (naux, nocc*nvir)
/// * `eps_occ` - Occupied orbital energies
/// * `eps_vir` - Virtual orbital energies
/// * `quad_freqs` - Imaginary quadrature frequencies ω_k
/// * `quad_weights` - Corresponding quadrature weights w_k
///
/// # Returns
/// RI-dRPA correlation energy in Hartree.
pub fn ri_drpa_energy(
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    quad_freqs: &[f64],
    quad_weights: &[f64],
) -> Result<f64, FerricError> {
    let mut e_c = 0.0f64;
    for (&omega, &wk) in quad_freqs.iter().zip(quad_weights.iter()) {
        let evals = ri_drpa_eigenvalues(b_ov, eps_occ, eps_vir, omega)?;
        let contrib: f64 = evals
            .iter()
            .map(|&lam| lam.ln() + (1.0 - lam))
            .sum();
        e_c += wk * contrib;
    }
    Ok(e_c / (2.0 * std::f64::consts::PI))
}
