//! Imaginary-frequency quadrature grids for RPA correlation energy.
//!
//! Two schemes:
//! - `GaussLegendre`: standard GL nodes mapped to [0,∞) via ω = u₀(1+x)/(1−x).
//! - `MiniMax`: same GL nodes with literature-optimized u₀ scale (Furche, JCP 2005).

use crate::config::{QuadratureConfig, QuadratureScheme};

/// Return (frequencies ω_k, weights w_k) for integrating over [0,∞).
///
/// The w_k already absorb the Jacobian of the domain mapping, so the
/// quadrature rule is: ∫₀^∞ f(ω) dω ≈ Σ_k w_k f(ω_k).
pub fn build_quadrature(cfg: &QuadratureConfig) -> (Vec<f64>, Vec<f64>) {
    let u0 = match cfg.scheme {
        QuadratureScheme::GaussLegendre => cfg.u0,
        QuadratureScheme::MiniMax => optimized_u0(cfg.n_points),
    };
    gauss_legendre_nodes(cfg.n_points, u0)
}

/// Mapped Gauss-Legendre nodes on [0,∞).
///
/// Transform: ω = u₀(1+x)/(1−x), where x_k are standard GL nodes on (−1,1).
/// Jacobian: dω/dx = 2u₀/(1−x)².
/// Transformed weight: w̃_k = w_k · 2u₀ / (1−x_k)².
pub fn gauss_legendre_nodes(n: usize, u0: f64) -> (Vec<f64>, Vec<f64>) {
    let (x, w) = gl_nodes_weights(n);
    let freqs: Vec<f64> = x.iter().map(|&xi| u0 * (1.0 + xi) / (1.0 - xi)).collect();
    let weights: Vec<f64> = x.iter().zip(w.iter())
        .map(|(&xi, &wi)| wi * 2.0 * u0 / (1.0 - xi).powi(2))
        .collect();
    (freqs, weights)
}

/// Literature-optimized u₀ scale parameters (Furche, JCP 122, 164106, 2005).
fn optimized_u0(n: usize) -> f64 {
    match n {
        1..=8   => 0.3,
        9..=16  => 0.4,
        _       => 0.5,
    }
}

/// Standard Gauss-Legendre nodes and weights on (−1, 1).
///
/// Uses Newton's method to find roots of the Legendre polynomial P_n(x).
fn gl_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0f64; n];
    let mut w = vec![0.0f64; n];
    let m = (n + 1) / 2;
    for i in 0..m {
        let mut xi = (std::f64::consts::PI * (4 * i + 3) as f64 / (4 * n + 2) as f64).cos();
        for _ in 0..100 {
            let (p, dp) = legendre_and_deriv(n, xi);
            let dx = -p / dp;
            xi += dx;
            if dx.abs() < 1e-15 { break; }
        }
        let (_, dp) = legendre_and_deriv(n, xi);
        let wi = 2.0 / ((1.0 - xi * xi) * dp * dp);
        x[i] = -xi;
        x[n - 1 - i] = xi;
        w[i] = wi;
        w[n - 1 - i] = wi;
    }
    // Sort ascending
    let mut pairs: Vec<(f64, f64)> = x.into_iter().zip(w.into_iter()).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let x: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let w: Vec<f64> = pairs.iter().map(|p| p.1).collect();
    (x, w)
}

/// Legendre polynomial P_n(x) and its derivative P_n'(x) via recurrence.
fn legendre_and_deriv(n: usize, x: f64) -> (f64, f64) {
    if n == 0 { return (1.0, 0.0); }
    if n == 1 { return (x, 1.0); }
    let mut p_prev = 1.0f64;
    let mut p_curr = x;
    for k in 2..=n {
        let p_next = ((2 * k - 1) as f64 * x * p_curr - (k - 1) as f64 * p_prev) / k as f64;
        p_prev = p_curr;
        p_curr = p_next;
    }
    let dp = n as f64 * (x * p_curr - p_prev) / (x * x - 1.0);
    (p_curr, dp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gl_weights_sum_to_pi_over_2() {
        // For f(ω) = 1/(1+ω²), ∫₀^∞ dω = π/2.
        let (freqs, weights) = gauss_legendre_nodes(20, 0.5);
        let sum: f64 = freqs.iter().zip(weights.iter())
            .map(|(w, wt)| wt / (1.0 + w * w))
            .sum();
        assert!((sum - std::f64::consts::FRAC_PI_2).abs() < 1e-4,
            "GL quadrature error: |sum - π/2| = {}", (sum - std::f64::consts::FRAC_PI_2).abs());
    }

    #[test]
    fn nodes_are_positive() {
        let (freqs, weights) = gauss_legendre_nodes(12, 0.5);
        assert_eq!(freqs.len(), 12);
        assert_eq!(weights.len(), 12);
        for (&f, &w) in freqs.iter().zip(weights.iter()) {
            assert!(f > 0.0, "freq must be positive, got {}", f);
            assert!(w > 0.0, "weight must be positive, got {}", w);
        }
    }
}
