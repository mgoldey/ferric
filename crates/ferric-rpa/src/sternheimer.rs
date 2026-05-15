//! PDEP Sternheimer response: independent-particle polarizability kernel.
//!
//! For a trial potential V (vector in the RI auxiliary basis), computes the
//! scalar χ(V, V; iω) = ⟨V|χ₀(iω)|V⟩ using the RI-MO Sternheimer equation.
//!
//! In the PDEP formalism all response is expressed via MO-basis B^P_ia
//! tensors — no AO-space Fock rebuild is needed at this level.

use ndarray::{Array1, Array2};

/// Compute ⟨V|χ₀(iω)|V⟩ for a single trial potential V.
///
/// Returns −2 Σ_{ia} (Σ_P V_P B^P_ia)² / (ε_a − ε_i + ω). This is ≤ 0.
pub fn chi_from_trial_potential(
    v: &Array1<f64>,
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    omega: f64,
) -> f64 {
    let nocc = eps_occ.len();
    let nvir = eps_vir.len();
    let nov = nocc * nvir;
    assert_eq!(b_ov.shape()[1], nov);

    // rhs_ia = Σ_P V_P B^P_ia  (shape: nov)
    let rhs = v.dot(b_ov); // shape (nov,)

    let mut chi = 0.0f64;
    for i in 0..nocc {
        for a in 0..nvir {
            let ia = i * nvir + a;
            let gap = eps_vir[a] - eps_occ[i] + omega;
            chi -= 2.0 * rhs[ia] * rhs[ia] / gap;
        }
    }
    chi
}

/// Compute the dielectric matrix ε̃_αβ(iω) = δ_αβ − χ₀_αβ(iω) in a subspace.
///
/// Given trial potentials V of shape (naux, m) (columns = trial vecs),
/// returns the m×m symmetric matrix with ε̃_αβ = δ_αβ + 2 Σ_{ia} rhs_α_ia rhs_β_ia / gap_ia.
/// Eigenvalues are ≥ 1 for a physical system.
pub fn dielectric_matrix(
    v_mat: &Array2<f64>,
    b_ov: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    omega: f64,
) -> Array2<f64> {
    let m = v_mat.ncols();
    let nocc = eps_occ.len();
    let nvir = eps_vir.len();
    let nov = nocc * nvir;
    assert_eq!(b_ov.shape()[1], nov);

    // rhs_mat: (m, nov) — rhs for each trial vector
    let rhs_mat = v_mat.t().dot(b_ov); // (m, nov)

    // Build m×m chi matrix: χ_αβ = -2 Σ_{ia} rhs_α_ia rhs_β_ia / gap_ia
    let mut chi = Array2::zeros((m, m));
    for i in 0..nocc {
        for a in 0..nvir {
            let ia = i * nvir + a;
            let gap = eps_vir[a] - eps_occ[i] + omega;
            let scale = -2.0 / gap;
            for alpha in 0..m {
                for beta in 0..m {
                    chi[(alpha, beta)] += scale * rhs_mat[(alpha, ia)] * rhs_mat[(beta, ia)];
                }
            }
        }
    }

    // ε̃_αβ = δ_αβ − χ_αβ
    let mut eps_mat = chi;
    for alpha in 0..m {
        eps_mat[(alpha, alpha)] += 1.0;
    }
    eps_mat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chi_static_two_level_system() {
        // 1 occupied, 1 virtual, 1 aux function.
        // B^P_ia = 1.0, eps_occ = -0.5, eps_vir = 0.5
        // χ₀(0) = -2 * B^2 / (eps_vir - eps_occ) = -2 * 1.0 / 1.0 = -2.0
        let b_ov = ndarray::array![[1.0f64]]; // shape (1, 1): naux=1, nocc*nvir=1
        let eps_occ = vec![-0.5f64];
        let eps_vir = vec![0.5f64];
        let v = ndarray::array![1.0f64]; // trial potential, naux=1
        let chi = chi_from_trial_potential(&v, &b_ov, &eps_occ, &eps_vir, 0.0);
        assert!(
            (chi + 2.0).abs() < 1e-12,
            "expected χ = -2.0, got {}",
            chi
        );
    }

    #[test]
    fn chi_freq_shift() {
        // Same system at omega=1.0
        // χ₀(i·1) = -2 * 1.0 / (1.0 + 1.0) = -1.0
        let b_ov = ndarray::array![[1.0f64]];
        let eps_occ = vec![-0.5f64];
        let eps_vir = vec![0.5f64];
        let v = ndarray::array![1.0f64];
        let chi = chi_from_trial_potential(&v, &b_ov, &eps_occ, &eps_vir, 1.0);
        assert!(
            (chi + 1.0).abs() < 1e-12,
            "expected χ = -1.0, got {}",
            chi
        );
    }

    #[test]
    fn dielectric_matrix_identity_at_zero_coupling() {
        // With B=0, dielectric should be identity.
        use ndarray::Array2;
        let b_ov = Array2::zeros((2, 2));
        let v_mat = ndarray::array![[1.0, 0.0], [0.0, 1.0f64]];
        let eps_occ = vec![-0.5f64];
        let eps_vir = vec![0.5f64, 1.5f64];
        let eps = dielectric_matrix(&v_mat, &b_ov, &eps_occ, &eps_vir, 0.0);
        assert!((eps[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((eps[(1, 1)] - 1.0).abs() < 1e-12);
        assert!(eps[(0, 1)].abs() < 1e-12);
    }
}
