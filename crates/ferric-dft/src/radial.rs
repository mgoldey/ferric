//! Radial quadrature for atom-centered grids.
//!
//! Provides the **Treutler-Ahlrichs M4** mapping (J. Chem. Phys. 102, 346,
//! 1995) of Chebyshev-of-the-2nd-kind nodes on `[-1, 1]` to the radial
//! interval `[0, ∞)`:
//!
//! ```text
//!     r(x) = ξ · (1 + x)^α / (1 - x)         (M4 form, α = 0.6)
//!     dr/dx = ξ · [(α (1 + x)^{α-1}) (1 - x) + (1 + x)^α] / (1 - x)²
//! ```
//!
//! `ξ` is an element-dependent scale factor proportional to the atomic
//! Bragg-Slater radius. The Chebyshev-2 weight on `[-1, 1]` is
//! `(π / (n+1)) sin²(θ_k)` at nodes `x_k = cos(π k / (n+1))`. With the
//! mapping and the radial-integration measure `4π r² dr`, the final
//! quadrature weight for `f(r)` integration over space is
//!
//! ```text
//!     w_k = (π / (n+1)) · sin²(θ_k) · (dr/dx)|_{x_k} · 4π · r_k²
//! ```
//!
//! Convention: weights returned by this module already include the
//! `4π r²` Jacobian, so combining with Lebedev weights gives the spherical
//! integral directly: `∫ f dV ≈ Σ_radial Σ_lebedev w_r · w_Ω · f(r, Ω)`.

/// Bragg-Slater atomic radii in atomic units (Bohr), Z=1..18.
/// Treutler-Ahlrichs M4 ξ uses 1/(2 R_BS) — these scale the integration
/// to put nodes near the valence shell of the atom.
fn bragg_slater_bohr(z: i32) -> f64 {
    // Angstroms from Slater 1964, ×1.889726 to Bohr.
    let r_a: f64 = match z {
        1 => 0.35,  2 => 0.30,
        3 => 1.45,  4 => 1.05, 5 => 0.85,  6 => 0.70,  7 => 0.65,  8 => 0.60,
        9 => 0.50, 10 => 0.45,
        11 => 1.80, 12 => 1.50, 13 => 1.25, 14 => 1.10, 15 => 1.00, 16 => 1.00,
        17 => 1.00, 18 => 0.71,
        _ => 1.00,
    };
    r_a * 1.8897259886
}

/// Treutler-Ahlrichs M4 radial nodes for atomic number `z`, `n` points.
///
/// Returns `(radii, weights)` where `weights` already include the
/// `4π r² dr` factor — combine with normalized Lebedev weights (Σ = 1)
/// for the full spherical integral.
pub fn treutler_ahlrichs_m4(z: i32, n: usize) -> (Vec<f64>, Vec<f64>) {
    let alpha = 0.6_f64;
    let xi = bragg_slater_bohr(z) / 2.0_f64.ln();
    // Treutler-Ahlrichs uses ξ_atom · log(2) for the prefactor, but the
    // M4 mapping above already absorbs the log-factor through the
    // (1+x)^α / (1-x) form. We use ξ = R_BS / ln(2) so that r → R_BS
    // at x ~ 0 (mid-shell).
    let pi = std::f64::consts::PI;
    let mut rs = Vec::with_capacity(n);
    let mut ws = Vec::with_capacity(n);
    let np1 = (n + 1) as f64;
    for k in 1..=n {
        let theta = pi * (k as f64) / np1;
        let x = theta.cos();
        let one_plus = 1.0 + x;
        let one_minus = 1.0 - x;
        // r(x) = ξ · (1+x)^α / (1-x)
        let r = xi * one_plus.powf(alpha) / one_minus;
        // dr/dx = ξ · [(α (1+x)^{α-1}) (1-x) + (1+x)^α] / (1-x)²
        let dr_dx = xi * (alpha * one_plus.powf(alpha - 1.0) * one_minus
                          + one_plus.powf(alpha)) / (one_minus * one_minus);
        // Chebyshev-2 weight on [-1, 1]: (π / (n+1)) sin²(θ_k).
        // 1/sqrt(1-x²) = 1/sin(θ_k) absorbed; we use the no-prefactor form:
        //   ∫_{-1}^{1} f dx ≈ Σ (π/(n+1)) sin²(θ_k) · f(x_k) / sin(θ_k)
        //                  = Σ (π/(n+1)) sin(θ_k) · f(x_k)
        let w_cheb = (pi / np1) * theta.sin();
        // Final: w_k = w_cheb · (dr/dx) · 4π r²
        let w_r = w_cheb * dr_dx * 4.0 * pi * r * r;
        rs.push(r);
        ws.push(w_r);
    }
    (rs, ws)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ta_m4_integrates_gaussian_to_high_precision() {
        // ∫_0^∞ 4π r² exp(-α r²) dr = (π/α)^{3/2}
        let alpha = 1.0_f64;
        let exact = (std::f64::consts::PI / alpha).powf(1.5);
        let (rs, ws) = treutler_ahlrichs_m4(8, 50); // Oxygen, 50 radial pts
        let approx: f64 = rs.iter().zip(ws.iter())
            .map(|(r, w)| w * (-alpha * r * r).exp())
            .sum();
        let err = (approx - exact).abs() / exact.abs();
        eprintln!("M4 50pt: exact={exact:.6}, approx={approx:.6}, relerr={err:.2e}");
        assert!(err < 1e-6, "M4 50pt Gaussian relerr {err:.2e}");
    }

    #[test]
    fn ta_m4_integrates_slater_proatom() {
        // ∫_0^∞ 4π r² (Z ξ³/π) exp(-2ξr) dr = Z (integrates to electron count)
        let z = 6_i32; // C: ξ ≈ 1/(0.70 Å · 1.89) = 0.756
        let xi: f64 = 1.0 / (0.70 * 1.8897259886);
        let zf = z as f64;
        let exact = zf;
        let (rs, ws) = treutler_ahlrichs_m4(z, 80);
        let approx: f64 = rs.iter().zip(ws.iter())
            .map(|(r, w)| w * zf * xi.powi(3) / std::f64::consts::PI * (-2.0 * xi * r).exp())
            .sum();
        let err = (approx - exact).abs() / exact.abs();
        eprintln!("Slater(Z=6) 80pt: exact={exact}, approx={approx:.6}, relerr={err:.2e}");
        assert!(err < 1e-3, "Slater Z=6 80pt relerr {err:.2e}");
    }
}
