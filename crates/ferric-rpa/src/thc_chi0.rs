//! THC-factorized χ₀ / dielectric contraction kernel for PDEP-RPA.
//!
//! **EXPERIMENTAL, feature `thc`, off by default. This is a contraction KERNEL
//! only: there is no builder from a molecule (no ISDF of the projected pair
//! densities), it is not wired into any RPA driver, and no timing claim is
//! made.** No molecular THC-dRPA implementation is known in the literature
//! (published THC-RPA, arXiv:2306.04880, is periodic/k-point).
//!
//! # What it computes
//!
//! The PDEP dielectric contracts over occupied-virtual pairs,
//!
//! ```text
//!   ε̃_αβ(iω) − δ_αβ = Σ_ia y_α,ia s_ia y_β,ia,   s_ia = 4 e_ia/(ω² + e_ia²).
//! ```
//!
//! With an interpolative (ISDF) factorization of the projected pair densities,
//!
//! ```text
//!   y_α,ia = Σ_K Ω_αK φ_i(r_K) φ_a(r_K),
//! ```
//!
//! and the minimax-Laplace split `e/(ω²+e²) ≈ Σ_l w_l cos(ω t_l) e^{−t_l e}`,
//! whose exponential separates as `e^{t_l ε_i} e^{−t_l ε_a}`:
//!
//! ```text
//!   ε̃ − I = Ω [ Σ_l c_l (G^l ∘ H^l) ] Ωᵀ,   c_l = 4 w_l cos(ω t_l)
//!   G^l_KL = Σ_i φ_iK e^{t_l ε_i} φ_iL,      H^l_KL = Σ_a φ_aK e^{−t_l ε_a} φ_aL
//! ```
//!
//! This is EXACTLY the Laplace-separated dense dielectric of that `y` — no
//! further approximation — which is the trivial-limit anchor
//! (`thc_chi0_matches_exact_in_the_trivial_limit`).
//!
//! # Correction relative to the pre-port branch (`feat/thc`, 2026-07)
//!
//! The pre-port kernel also Hadamard-multiplied a THC ERI core `M` into the
//! bracket: `Ω (M ∘ Σ_l c_l G^l∘H^l) Ωᵀ`. That is not χ₀ of any single `y`
//! unless `M ≡ 1` (a rank-r `M = Σ_r u_r u_rᵀ` turns it into a SUM of r
//! different dielectrics). Its only exactness test used `M = ones`, where the
//! factor is the identity of `∘`, so deleting `M` survived every test (run
//! 2026-10-24, recorded in `reference/thc/thc-lane-2026-10.md`). The core now
//! lives where it belongs — in `Ω` (`Ω = ξᵀζ` projects the ISDF functions onto
//! the PDEP eigenbasis) — and is not a separate argument.
//!
//! # Cost
//!
//! Per (frequency, Laplace node) `O((nocc + nvir) · N_IP²)`, plus one
//! `O(m · N_IP²)` projection per frequency, against the dense
//! `O(m² · nov)`. Unmeasured.

use ferric_quadrature::LaplaceQuadrature;
use ndarray::{Array1, Array2, Axis};

/// THC factors of the projected pair densities. Built once per
/// (molecule, basis, grid); reused across all frequencies and Laplace nodes.
pub struct ThcChi0 {
    /// Occupied MO collocation `φ_i(r_K)`, `(nocc, n_ip)`.
    phi_occ: Array2<f64>,
    /// Virtual MO collocation `φ_a(r_K)`, `(nvir, n_ip)`.
    phi_vir: Array2<f64>,
    /// `Ω_αK`, `(m, n_ip)`: ISDF functions projected on the PDEP eigenbasis.
    omega_proj: Array2<f64>,
}

impl ThcChi0 {
    /// Assemble from precomputed factors; column counts must all be `n_ip`.
    pub fn from_parts(
        phi_occ: Array2<f64>,
        phi_vir: Array2<f64>,
        omega_proj: Array2<f64>,
    ) -> Result<Self, String> {
        let n_ip = omega_proj.ncols();
        for (name, a) in [("phi_occ", &phi_occ), ("phi_vir", &phi_vir)] {
            if a.ncols() != n_ip {
                return Err(format!(
                    "ThcChi0: {name} has {} columns, expected n_ip={n_ip}",
                    a.ncols()
                ));
            }
        }
        Ok(Self {
            phi_occ,
            phi_vir,
            omega_proj,
        })
    }

    /// Interpolation points.
    pub fn n_interpolation_points(&self) -> usize {
        self.omega_proj.ncols()
    }

    /// `ε̃(iω) = I + Π`, same convention as
    /// [`crate::sternheimer::dielectric_matrix_from_projection`] with
    /// `prefactor = 4` (closed shell; 2 per spin channel).
    pub fn dielectric_at_omega(
        &self,
        eps_occ: &[f64],
        eps_vir: &[f64],
        omega: f64,
        quad: &LaplaceQuadrature,
        prefactor: f64,
    ) -> Result<Array2<f64>, String> {
        check_len("occupied", eps_occ.len(), self.phi_occ.nrows())?;
        check_len("virtual", eps_vir.len(), self.phi_vir.nrows())?;
        let n_ip = self.n_interpolation_points();
        let mut acc = Array2::<f64>::zeros((n_ip, n_ip));
        for (&t, &w) in quad.points.iter().zip(quad.weights.iter()) {
            let g = weighted_gram(&self.phi_occ, eps_occ, t);
            let h = weighted_gram(&self.phi_vir, eps_vir, -t);
            // coeff may be NEGATIVE when cos(ω t) < 0: no SYRK downstream.
            let coeff = prefactor * w * (omega * t).cos();
            acc.scaled_add(coeff, &(&g * &h));
        }
        let e = self.omega_proj.dot(&acc).dot(&self.omega_proj.t());
        let mut e = (&e + &e.t()) * 0.5;
        e.diag_mut().map_inplace(|d| *d += 1.0);
        Ok(e)
    }

    /// Per-(ω, Laplace-node) FLOP estimate of the contraction above.
    pub fn flops_per_node(&self, nocc: usize, nvir: usize, m_dim: usize) -> f64 {
        let n = self.n_interpolation_points() as f64;
        2.0 * (nocc + nvir + m_dim) as f64 * n * n
    }
}

fn check_len(what: &str, got: usize, want: usize) -> Result<(), String> {
    if got != want {
        return Err(format!(
            "ThcChi0: {got} {what} energies vs {want} collocation rows"
        ));
    }
    Ok(())
}

/// `Σ_p φ_pK e^{t ε_p} φ_pL`.
fn weighted_gram(phi: &Array2<f64>, eps: &[f64], t: f64) -> Array2<f64> {
    let f = Array1::from_iter(eps.iter().map(|&e| (t * e).exp()));
    let scaled = phi * &f.insert_axis(Axis(1));
    phi.t().dot(&scaled)
}

/// Dense-path FLOPs for the same quantity, for comparison.
pub fn dense_chi0_flops(m_dim: usize, nov: usize) -> f64 {
    2.0 * (m_dim as f64).powi(2) * nov as f64
}

/// Dense (non-THC) Laplace-separated dielectric over the explicit `ia` axis
/// (column index `i·nvir + a`). The independent reference for the anchor.
pub fn laplace_reference_dielectric(
    y: &Array2<f64>,
    eps_occ: &[f64],
    eps_vir: &[f64],
    omega: f64,
    quad: &LaplaceQuadrature,
    prefactor: f64,
) -> Array2<f64> {
    let nvir = eps_vir.len();
    let mut e = Array2::<f64>::eye(y.nrows());
    for (&t, &w) in quad.points.iter().zip(quad.weights.iter()) {
        let coeff = prefactor * w * (omega * t).cos();
        let mut scaled = y.clone();
        for (col, mut c) in scaled.axis_iter_mut(Axis(1)).enumerate() {
            let e_ia = eps_vir[col % nvir] - eps_occ[col / nvir];
            c *= (-0.5 * t * e_ia).exp();
        }
        e.scaled_add(coeff, &scaled.dot(&scaled.t()));
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sternheimer::{build_scale_factors, dielectric_matrix_from_projection};

    fn toy(nocc: usize, nvir: usize) -> (Vec<f64>, Vec<f64>) {
        let eps_occ: Vec<f64> = (0..nocc).map(|i| -0.9 + 0.05 * i as f64).collect();
        let eps_vir: Vec<f64> = (0..nvir).map(|a| 0.3 + 0.07 * a as f64).collect();
        (eps_occ, eps_vir)
    }

    fn quad_for(eps_occ: &[f64], eps_vir: &[f64], n: usize) -> LaplaceQuadrature {
        let lo = eps_vir[0] - eps_occ[eps_occ.len() - 1];
        let hi = eps_vir[eps_vir.len() - 1] - eps_occ[0];
        LaplaceQuadrature::new(n, lo, hi).expect("laplace quadrature")
    }

    /// Deterministic generic (non-indicator, sign-mixed) matrix.
    fn mat(r: usize, c: usize, a: usize, b: usize, m: usize) -> Array2<f64> {
        Array2::from_shape_fn((r, c), |(i, j)| {
            ((i * a + j * b + 3) % m) as f64 / m as f64 - 0.4
        })
    }

    /// `y_α,ia = Σ_K Ω_αK φ_iK φ_aK`, built explicitly on the `ia` axis.
    fn explicit_y(thc: &ThcChi0) -> Array2<f64> {
        let (nocc, nvir) = (thc.phi_occ.nrows(), thc.phi_vir.nrows());
        let mut kr = Array2::<f64>::zeros((thc.n_interpolation_points(), nocc * nvir));
        for i in 0..nocc {
            for a in 0..nvir {
                let col = &thc.phi_occ.row(i) * &thc.phi_vir.row(a);
                kr.column_mut(i * nvir + a).assign(&col);
            }
        }
        thc.omega_proj.dot(&kr)
    }

    fn toy_thc(nocc: usize, nvir: usize, n_ip: usize, m: usize) -> ThcChi0 {
        ThcChi0::from_parts(
            mat(nocc, n_ip, 5, 3, 7),
            mat(nvir, n_ip, 11, 2, 13),
            mat(m, n_ip, 3, 5, 17),
        )
        .unwrap()
    }

    fn rel_max(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let d = (a - b).iter().fold(0.0f64, |m, v| m.max(v.abs()));
        d / b.iter().fold(0.0f64, |m, v| m.max(v.abs()))
    }

    /// The Laplace split alone (no THC) reproduces the exact kernel.
    #[test]
    fn laplace_reference_matches_exact_dielectric() {
        let (eps_occ, eps_vir) = toy(4, 10);
        let y = mat(6, 40, 31, 17, 23);
        let quad = quad_for(&eps_occ, &eps_vir, 7);
        for &omega in &[0.0, 0.3, 1.0] {
            let exact = dielectric_matrix_from_projection(
                &y,
                &build_scale_factors(&eps_occ, &eps_vir, omega),
            );
            let lap = laplace_reference_dielectric(&y, &eps_occ, &eps_vir, omega, &quad, 4.0);
            assert!(
                rel_max(&lap, &exact) < 1e-3,
                "omega={omega}: {:.2e}",
                rel_max(&lap, &exact)
            );
        }
    }

    /// EXACTNESS ANCHOR. For generic factors the THC contraction must equal
    /// the dense Laplace-separated dielectric of the explicitly built `y` to
    /// round-off, at zero and nonzero frequency (the latter exercises the
    /// sign-changing `cos(ω t)` weights).
    #[test]
    fn thc_chi0_matches_exact_in_the_trivial_limit() {
        let (eps_occ, eps_vir) = toy(3, 5);
        let thc = toy_thc(3, 5, 9, 4);
        let y = explicit_y(&thc);
        let quad = quad_for(&eps_occ, &eps_vir, 7);
        for &omega in &[0.0, 0.5, 2.0] {
            let got = thc
                .dielectric_at_omega(&eps_occ, &eps_vir, omega, &quad, 4.0)
                .unwrap();
            let want = laplace_reference_dielectric(&y, &eps_occ, &eps_vir, omega, &quad, 4.0);
            assert!(
                rel_max(&got, &want) < 1e-12,
                "omega={omega}: {:.2e}",
                rel_max(&got, &want)
            );
        }
    }

    /// Against the EXACT dense χ₀ of the same `y` the only residual is the
    /// Laplace quadrature, so it must shrink monotonically with n_quad.
    #[test]
    fn thc_chi0_error_vs_dense_chi0_is_quadrature_only() {
        let (eps_occ, eps_vir) = toy(3, 5);
        let thc = toy_thc(3, 5, 9, 4);
        let y = explicit_y(&thc);
        let exact =
            dielectric_matrix_from_projection(&y, &build_scale_factors(&eps_occ, &eps_vir, 0.3));
        let errs: Vec<f64> = [3usize, 5, 7]
            .iter()
            .map(|&n| {
                let q = quad_for(&eps_occ, &eps_vir, n);
                rel_max(
                    &thc.dielectric_at_omega(&eps_occ, &eps_vir, 0.3, &q, 4.0)
                        .unwrap(),
                    &exact,
                )
            })
            .collect();
        assert!(
            errs[0] > errs[1] && errs[1] > errs[2] && errs[2] < 1e-3,
            "errors {errs:?}"
        );
    }

    #[test]
    fn thc_dielectric_is_symmetric() {
        let (eps_occ, eps_vir) = toy(3, 6);
        let thc = toy_thc(3, 6, 12, 5);
        let e = thc
            .dielectric_at_omega(
                &eps_occ,
                &eps_vir,
                0.4,
                &quad_for(&eps_occ, &eps_vir, 7),
                4.0,
            )
            .unwrap();
        let worst = (&e - &e.t()).iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(worst < 1e-12, "dielectric not symmetric: {worst:.3e}");
    }

    #[test]
    fn thc_chi0_rejects_shape_mismatch() {
        assert!(ThcChi0::from_parts(
            Array2::zeros((3, 10)),
            Array2::zeros((5, 10)),
            Array2::zeros((4, 9))
        )
        .is_err());
        let thc = toy_thc(3, 5, 9, 4);
        let q = quad_for(&[-0.9, -0.8], &[0.3, 0.4], 3);
        assert!(thc
            .dielectric_at_omega(&[-0.9, -0.8], &[0.3; 5], 0.0, &q, 4.0)
            .is_err());
    }
}
