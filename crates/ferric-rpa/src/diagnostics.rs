//! RI-dRPA sanity checks: full-basis correlation energy and eigenvalue diagnostics.
//!
//! The RI-dRPA approximation evaluates RPA on the full RI-auxiliary basis without
//! eigenpotential truncation. Used as a sanity check against PDEP-RPA.

use ndarray::{Array2, Axis, Zip};
use ndarray_linalg::{Eigh, UPLO};
use rayon::prelude::*;
use ferric_core::FerricError;

use crate::sternheimer::build_scale_factors;

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
    //
    // SYRK path: b_scaled[p,ia] = b_ov[p,ia] * sqrt(4·e_ia/(ω²+e_ia²))
    //   chi0 = b_scaled @ b_scaled^T via DSYRK (symmetric rank-k update, ~2× DGEMM).
    let scale = build_scale_factors(eps_occ, eps_vir, omega);
    let mut b_scaled: Array2<f64> = b_ov.to_owned();
    let scale_row = scale.view().insert_axis(Axis(0)); // (1, nov)
    Zip::from(&mut b_scaled)
        .and_broadcast(scale_row)
        .for_each(|x, &s| *x *= s);
    let mut chi0 = crate::sternheimer::syrk_aat(&b_scaled);
    let _ = nvir; // nvir factored into scale via build_scale_factors

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
/// Unrestricted RI-dRPA correlation energy via full-basis dielectric.
///
/// Builds ε̃ = I + Π_α + Π_β at each ω_k, evaluates trace-log without
/// any Davidson/PDEP machinery. Used to localize bugs in the U-RPA stack
/// — if the eigensolver path disagrees with this diagnostic, the bug is
/// in eigensolving, not the spin-summed dielectric formula.
pub fn u_ri_drpa_energy(
    b_ov_a: &Array2<f64>, eps_occ_a: &[f64], eps_vir_a: &[f64],
    b_ov_b: &Array2<f64>, eps_occ_b: &[f64], eps_vir_b: &[f64],
    quad_freqs: &[f64],
    quad_weights: &[f64],
) -> Result<f64, FerricError> {
    use crate::sternheimer::build_scale_factors_with_prefactor;
    use crate::sternheimer::syrk_aat;

    let naux = b_ov_a.nrows();

    let contribs: Result<Vec<f64>, FerricError> = quad_freqs
        .par_iter()
        .zip(quad_weights.par_iter())
        .map(|(&omega, &wk)| {
            let mut eps_mat = Array2::<f64>::zeros((naux, naux));
            for p in 0..naux { eps_mat[(p, p)] = 1.0; }
            for (b, eo, ev) in [
                (b_ov_a, eps_occ_a, eps_vir_a),
                (b_ov_b, eps_occ_b, eps_vir_b),
            ] {
                if eo.is_empty() {
                    continue; // empty spin channel adds nothing
                }
                let scale = build_scale_factors_with_prefactor(eo, ev, omega, 2.0);
                let mut bs = b.to_owned();
                let scale_row = scale.view().insert_axis(Axis(0));
                Zip::from(&mut bs)
                    .and_broadcast(scale_row)
                    .for_each(|x, &s| *x *= s);
                let chi_sigma = syrk_aat(&bs);
                eps_mat = eps_mat + &chi_sigma;
            }
            let (evals, _) = eps_mat.eigh(UPLO::Upper)
                .map_err(|e| FerricError::General(format!("U-RI-dRPA eigh: {e}")))?;
            let contrib: f64 = evals.iter().map(|&lam| lam.ln() + (1.0 - lam)).sum();
            Ok(wk * contrib)
        })
        .collect();

    let e_c: f64 = contribs?.iter().sum();
    Ok(e_c / (2.0 * std::f64::consts::PI))
}

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
