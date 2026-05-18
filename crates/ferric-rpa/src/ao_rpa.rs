//! AO-basis (imaginary-time) formulation of dRPA correlation.
//!
//! Reference: Kaltak, Klimeš, Kresse, J. Chem. Theory Comput. 10, 2498 (2014).
//!
//! # Overview
//!
//! Standard MO-basis dRPA has cost O(N⁴) per ω-point because the
//! cosine-modulated Laplace expansion couples ω to per-orbital energies
//! (see commit 108ee8f and task #32 for the resulting accuracy issues).
//!
//! Kaltak-Kresse moves the cosine out of the inner loop by going through
//! the imaginary-time / Laplace conjugate of imaginary frequency:
//!
//!   χ₀(iω, μν, λσ) = -∫_0^∞ dτ cos(ωτ) [G^o_{λμ}(τ) G^v_{νσ}(τ)
//!                                       + G^o_{νσ}(τ) G^v_{λμ}(τ)]
//!
//! where the imaginary-time occupied/virtual Green's functions
//!
//!   G^o_{μν}(τ) = -Σ_i C_{μi} C_{νi} exp(-(ε_i - μ_F)·τ)   τ > 0
//!   G^v_{μν}(τ) =  Σ_a C_{μa} C_{νa} exp(-(ε_a - μ_F)·τ)
//!
//! factor into MO sums independent of ω. With AO sparsity (atom-pair
//! cutoffs on C_μi, C_νa via local orbitals), the (μν,λσ) contraction
//! drops to O(N²) per (τ,ω) pair, and the τ-quadrature size is O(log N)
//! → total cost O(N³ log²N) in the best case.
//!
//! # First-land scope
//!
//! This module implements the **MO-basis** version of the imaginary-time
//! formulation — same scaling as Dense RPA, but with the τ↔ω separation
//! that's the prerequisite for the AO-sparse extension. Validates the
//! τ-grid + cosine-Fourier integration against the existing Dense path.
//!
//! AO sparsity (the real scaling win) is the C9 follow-up after this
//! module proves the conceptual machinery works.

use ferric_core::FerricError;
use ferric_quadrature::LaplaceQuadrature;
use ndarray::Array2;

/// Build the imaginary-time τ-quadrature for the energy-gap range.
///
/// Uses the minimax-Laplace nodes/weights for `1/x` on `[ymin, ymax]`,
/// which are exactly the right nodes for the Laplace transform of the
/// occupied/virtual exponentials. ymin = smallest e_ia (HOMO-LUMO gap),
/// ymax = largest e_ia.
pub fn build_tau_quadrature(
    eps_occ: &[f64],
    eps_vir: &[f64],
    n_quad: usize,
) -> LaplaceQuadrature {
    let eps_homo = eps_occ.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let eps_lumo = eps_vir.iter().cloned().fold(f64::INFINITY, f64::min);
    let eps_min = eps_occ.iter().cloned().fold(f64::INFINITY, f64::min);
    let eps_max = eps_vir.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ymin = eps_lumo - eps_homo;
    let ymax = eps_max - eps_min;
    LaplaceQuadrature::new(n_quad, ymin, ymax)
}

/// MO-basis χ₀(iω) matrix in the RI-aux basis via imaginary-time formulation.
///
/// Returns Π(iω) = -χ₀(iω) of shape (naux, naux), with prefactor 4 baked
/// in for closed-shell. The dielectric is then ε̃ = I + Π.
///
/// Math: in MO basis the (i,a) decomposition gives
///
///   Π^P^Q = 4 Σ_{ia} B^P_ia · e_ia/(ω²+e_ia²) · B^Q_ia
///
/// Going through τ:
///
///   e_ia/(ω²+e_ia²) = ∫_0^∞ dτ cos(ωτ) exp(-e_ia·τ)
///                  ≈ Σ_l w_l cos(ω t_l) exp(-e_ia t_l)
///
/// The Laplace nodes (t_l, w_l) are tuned for 1/x on [ymin, ymax].
/// At a given ω, the per-spin contribution becomes
///
///   Π(ω) = 4 Σ_l w_l cos(ωt_l) X^l (X^l)^T,
///   where X^l_{P,ia} = B^P_ia · exp(-e_ia t_l / 2)
///
/// **This is bit-for-bit the same formula as crate::laplace_chi0**, but
/// implemented in this module as the foundation for the AO-basis
/// extension. We expect identical results to dielectric_matrix_laplace
/// at the same (n_quad, ω, B_ov).
///
/// At ω·t_max ≫ 1 the cosine modulation is faster than the quadrature
/// can resolve; the [bounded-ω fallback fix from commit 108ee8f
/// applies if used inside run_pdep_rpa](crate::laplace_chi0).
pub fn pi_via_imag_time(
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    omega: f64,
    laplace: &LaplaceQuadrature,
) -> Array2<f64> {
    use ndarray::{Axis, Zip};

    let naux = b_ov.shape()[0];
    let nov = b_ov.shape()[1];
    let nocc = eps_occ.len();
    let nvir = eps_vir.len();
    assert_eq!(nov, nocc * nvir);

    let mut e_ia = Vec::with_capacity(nov);
    for i in 0..nocc {
        for a in 0..nvir {
            e_ia.push(eps_vir[a] - eps_occ[i]);
        }
    }

    let mut pi = Array2::<f64>::zeros((naux, naux));

    // Scratch for X^l: shape (naux, nov).
    let mut x_l = Array2::<f64>::zeros((naux, nov));
    for (&t_l, &w_l) in laplace.points.iter().zip(laplace.weights.iter()) {
        // X^l_{P,ia} = B^P_ia · exp(-e_ia · t_l / 2)
        x_l.assign(b_ov);
        let factor: Vec<f64> = e_ia.iter().map(|&e| (-0.5 * t_l * e).exp()).collect();
        let factor_arr = ndarray::Array1::from(factor);
        let factor_row = factor_arr.view().insert_axis(Axis(0));
        Zip::from(&mut x_l).and_broadcast(factor_row).for_each(|x, &f| *x *= f);

        // Π(ω) += 4 · w_l cos(ω t_l) · X^l (X^l)^T
        let coeff = 4.0 * w_l * (omega * t_l).cos();
        // pi += coeff · x_l · x_l.t()
        let xt = x_l.t();
        ndarray::linalg::general_mat_mul(coeff, &x_l.view(), &xt, 1.0, &mut pi);
    }

    pi
}

/// Full AO-basis dielectric ε̃(iω) = I + Π via imaginary-time MO route.
///
/// Same input/output contract as
/// [`crate::laplace_chi0::dielectric_matrix_laplace`], implemented through
/// the τ-grid. Used here for validation against the dense path.
pub fn dielectric_matrix_imag_time(
    v_mat: &Array2<f64>,
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    omega: f64,
    laplace: &LaplaceQuadrature,
) -> Result<Array2<f64>, FerricError> {
    let pi_naux = pi_via_imag_time(b_ov, eps_occ, eps_vir, omega, laplace);
    // Project into the trial-vector subspace: ε̃_proj = V^T (I + Π) V
    let m = v_mat.ncols();
    let i_naux = Array2::<f64>::eye(pi_naux.nrows());
    let mut eps_naux = i_naux;
    eps_naux = eps_naux + &pi_naux;
    let eps_proj = v_mat.t().dot(&eps_naux).dot(v_mat);
    let _ = m;
    Ok(eps_proj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sternheimer::dielectric_matrix;

    #[test]
    fn imag_time_matches_dense_synthetic() {
        // Tiny synthetic: B random, e_ia spaced.
        let naux = 5;
        let nocc = 2;
        let nvir = 3;
        let nov = nocc * nvir;
        let b_ov = Array2::from_shape_fn((naux, nov), |(p, ia)| {
            0.1 + 0.03 * p as f64 - 0.02 * ia as f64
        });
        let eps_occ = vec![-0.5_f64, -0.3];
        let eps_vir = vec![0.2_f64, 0.6, 1.4];
        // Identity V — Π in aux basis directly.
        let v_mat = Array2::<f64>::eye(naux);
        let omega = 0.5;

        let laplace = build_tau_quadrature(&eps_occ, &eps_vir, 8);
        let eps_imag = dielectric_matrix_imag_time(&v_mat, &b_ov, &eps_occ, &eps_vir, omega, &laplace).unwrap();
        let eps_dense = dielectric_matrix(&v_mat, &b_ov, &eps_occ, &eps_vir, omega);

        let max_err = eps_imag.iter().zip(eps_dense.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        eprintln!("imag-time vs dense max elementwise error: {max_err:.3e}");
        assert!(max_err < 5e-3,
            "imag-time τ-route should match dense at low ω: max_err={max_err:.3e}");
    }
}
