//! Radial quadrature for atom-centered grids.
//!
//! Provides the **Treutler-Ahlrichs M4** mapping (J. Chem. Phys. 102, 346,
//! 1995) of Chebyshev-of-the-2nd-kind nodes on `[-1, 1]` to the radial
//! interval `[0, ∞)`:
//!
//! ```text
//!     r(x)    = -(ξ/ln2) · (1 + x)^α · ln((1 - x) / 2)    (true M4 form, α = 0.6)
//!     dr/dx   = (ξ/ln2) · (1 + x)^α · [-α/(1+x) · ln((1-x)/2) + 1/(1-x)]
//! ```
//!
//! `ξ` is the atom-specific Treutler-Ahlrichs scaling parameter (dimensionless,
//! from Table I of TA 1995, matching PySCF's `_treutler_ahlrichs_xi` table).
//! The Chebyshev-2 weight on `[-1, 1]` is `(π/(n+1)) · sin(θ_k)` (not
//! `sin²`) because the Chebyshev-T₂ weight is `sin(θ)` in the θ variable,
//! giving `|dx/dθ| = sin(θ)`. With the mapping and the radial-integration
//! measure `4π r² dr`, the final quadrature weight is
//!
//! ```text
//!     w_k = (π / (n+1)) · sin(θ_k) · |dr/dx|_{x_k}| · 4π · r_k²
//! ```
//!
//! Convention: weights returned by this module already include the
//! `4π r²` Jacobian, so combining with Lebedev weights (Σ = 1) gives
//! the spherical integral directly: `∫ f dV ≈ Σ_radial Σ_lebedev w_r · w_Ω · f(r, Ω)`.

/// Treutler-Ahlrichs ξ parameters (dimensionless) for Z = 1..103.
///
/// Source: PySCF `dft.radi._treutler_ahlrichs_xi`, which reproduces
/// Table I of Treutler & Ahlrichs, JCP 102, 346 (1995).
/// Indexed by Z (Z=1 is index 1; index 0 is unused placeholder 1.0).
const TA_XI: &[f64] = &[
    // Z=0 (unused)
    1.0,
    // Z=1..10
    0.8, 0.9, 1.8, 1.4, 1.3, 1.1, 0.9, 0.9, 0.9, 0.9,
    // Z=11..18
    1.4, 1.3, 1.3, 1.2, 1.1, 1.0, 1.0, 1.0,
    // Z=19..36
    1.5, 1.4, 1.3, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.1, 1.1, 1.1, 1.1, 1.0, 0.9, 0.9,
    // Z=37..54
    0.9, 0.9, 2.0, 1.7, 1.5, 1.5, 1.35, 1.35, 1.25, 1.2, 1.25, 1.3, 1.5, 1.5, 1.3, 1.2,
    // Z=55..86
    1.2, 1.15, 1.15, 1.15, 2.5, 2.2, 2.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
    // Z=87..103
    1.5, 2.5, 2.1, 3.685, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
    1.5,
];

/// Return the TA ξ parameter for atomic number `z`.
fn ta_xi(z: i32) -> f64 {
    let z = z as usize;
    if z < TA_XI.len() { TA_XI[z] } else { 1.5 }
}

/// Treutler-Ahlrichs M4 radial nodes for atomic number `z`, `n` points.
///
/// Implements the true TA-M4 formula from JCP 102, 346 (1995):
///   `r(x) = -(ξ/ln2) · (1+x)^0.6 · ln((1-x)/2)`
///
/// Returns `(radii, weights)` where `weights` already include the
/// `4π r² dr` factor — combine with normalized Lebedev weights (Σ = 1)
/// for the full spherical integral.
///
/// Grid points are ordered from smallest to largest radius, matching the
/// PySCF convention after the `r[::-1]` reversal.
pub fn treutler_ahlrichs_m4(z: i32, n: usize) -> (Vec<f64>, Vec<f64>) {
    let alpha = 0.6_f64;
    let xi = ta_xi(z);
    let ln2 = xi / 2.0_f64.ln();   // = ξ / ln(2)
    let pi = std::f64::consts::PI;
    let np1 = (n + 1) as f64;

    let mut rs = Vec::with_capacity(n);
    let mut ws = Vec::with_capacity(n);

    // Build in k = 1..=n order (large x = small r first), then reverse so
    // radii increase from small to large (matching PySCF's `r[::-1]`).
    for k in 1..=n {
        let theta = pi * (k as f64) / np1;
        let x = theta.cos();   // x decreases as k increases; x[1] ≈ +1, x[n] ≈ -1
        let one_plus = 1.0 + x;
        let one_minus = 1.0 - x;
        // r = -ln2 * (1+x)^α * ln((1-x)/2)
        // Note: (1-x)/2 ∈ (0,1) so ln((1-x)/2) < 0, making r > 0.
        let log_term = (one_minus / 2.0).ln();  // negative for x ∈ (-1, 1)
        let r = -ln2 * one_plus.powf(alpha) * log_term;
        // dr/dx = ln2 * (1+x)^α * [-α/(1+x) * ln((1-x)/2) + 1/(1-x)]
        let dr_dx = ln2 * one_plus.powf(alpha) * (-alpha / one_plus * log_term + 1.0 / one_minus);
        // Chebyshev-2 integration weight: |dx/dθ| · dθ = sin(θ) · (π/(n+1))
        let w_cheb = (pi / np1) * theta.sin();
        let w_r = w_cheb * dr_dx * 4.0 * pi * r * r;
        rs.push(r);
        ws.push(w_r);
    }

    // Reverse to small→large r order (PySCF convention).
    rs.reverse();
    ws.reverse();

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
        eprintln!("TA-M4 50pt: exact={exact:.6}, approx={approx:.6}, relerr={err:.2e}");
        assert!(err < 1e-3, "TA-M4 50pt Gaussian relerr {err:.2e}");
    }

    #[test]
    fn ta_m4_integrates_slater_proatom() {
        // ∫_0^∞ 4π r² (Z ξ³/π) exp(-2ξr) dr = Z (integrates to electron count)
        // Using exponential decay rate matching Carbon's TA xi.
        let z = 6_i32; // Carbon
        let xi_slater: f64 = 1.0 / (0.70 * 1.8897259886);
        let zf = z as f64;
        let exact = zf;
        let (rs, ws) = treutler_ahlrichs_m4(z, 80);
        let approx: f64 = rs.iter().zip(ws.iter())
            .map(|(r, w)| w * zf * xi_slater.powi(3) / std::f64::consts::PI * (-2.0 * xi_slater * r).exp())
            .sum();
        let err = (approx - exact).abs() / exact.abs();
        eprintln!("TA-M4 Slater(Z=6) 80pt: exact={exact}, approx={approx:.6}, relerr={err:.2e}");
        assert!(err < 5e-2, "TA-M4 Slater Z=6 80pt relerr {err:.2e}");
    }
}
