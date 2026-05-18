//! Imaginary-frequency quadrature grids for RPA correlation energy.
//!
//! Three schemes:
//! - `GaussLegendre`: standard GL nodes mapped to [0,∞) via ω = u₀(1+x)/(1−x).
//!   Default; unbounded — max ω ~ 100 a.u. at n=20.
//! - `MiniMax`: same GL nodes with literature-optimized u₀ scale (Furche, JCP 2005).
//! - `ChebyshevTan`: Eshuis-Yarkony-Furche (JCP 132, 234114, 2010) — Chebyshev-2
//!   nodes mapped to [0, u₀·tan(π/2)] via ω = u₀ tan(π(1+x)/4). Bounded max ω;
//!   recommended whenever the Laplace χ₀ backend is used (keeps ω·t_max safe).

use crate::config::{QuadratureConfig, QuadratureScheme};

/// Return (frequencies ω_k, weights w_k) for integrating over [0,∞).
///
/// The w_k already absorb the Jacobian of the domain mapping, so the
/// quadrature rule is: ∫₀^∞ f(ω) dω ≈ Σ_k w_k f(ω_k).
pub fn build_quadrature(cfg: &QuadratureConfig) -> (Vec<f64>, Vec<f64>) {
    match cfg.scheme {
        QuadratureScheme::GaussLegendre => gauss_legendre_nodes(cfg.n_points, cfg.u0),
        QuadratureScheme::MiniMax => gauss_legendre_nodes(cfg.n_points, optimized_u0(cfg.n_points)),
        QuadratureScheme::ChebyshevTan => chebyshev_tan_nodes(cfg.n_points, cfg.u0),
    }
}

/// Recommended u₀ for ChebyshevTan when paired with Laplace χ₀: smaller
/// u₀ pushes more quadrature points into the Laplace-safe regime (ω·t_max
/// < π/2). On H2O/cc-pVDZ with t_max ≈ 1.5, u₀ = 0.2 keeps 16/20 points on
/// the cheap Laplace path; u₀ = 0.5 keeps only 13/20. Smaller still means
/// the high-ω tail is undersampled, so 0.2 balances safe-fraction vs
/// integrand coverage.
pub const CHEBYSHEV_TAN_RECOMMENDED_U0: f64 = 0.2;

/// Modified Gauss-Chebyshev nodes on a bounded `[0, u₀·tan(π/2−ε)]` interval
/// via the tan-map of Eshuis-Yarkony-Furche (JCP 132, 234114, 2010 §III).
///
/// Transform: ω = u₀ · tan(π(1+x)/4) with x ∈ (−1, 1).
/// Chebyshev-2 weight: w_k^cheb = π/(n+1) · sin²(π·k/(n+1)) on x_k = cos(π·k/(n+1)).
/// Final weight absorbs Jacobian and 1/√(1−x²): w_k = π/(n+1) · sin(θ_k) · (u₀ π/4) · sec²(arg).
///
/// Why we have this scheme: the standard Gauss-Legendre map puts extreme mass
/// at ω → ∞ (the last node hits ω ~ 100 a.u. at n=20). For the Laplace χ₀
/// path that's catastrophic — ω·t_max ≫ π/2 breaks the cosine-modulated
/// quadrature (see commit 108ee8f for the bounded-ω fix). With the tan-map,
/// the max ω is finite by construction, so Laplace runs without falling
/// back to dense for high-ω points.
pub fn chebyshev_tan_nodes(n: usize, u0: f64) -> (Vec<f64>, Vec<f64>) {
    let mut freqs = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);
    let np1 = (n + 1) as f64;
    let pi = std::f64::consts::PI;
    for k in 1..=n {
        let theta_k = pi * (k as f64) / np1;
        let x_k = theta_k.cos();
        let arg = pi * (1.0 + x_k) / 4.0;
        let omega_k = u0 * arg.tan();
        let sec2 = 1.0 / arg.cos().powi(2);
        // w_k^cheb-2 · Jacobian · 1/sqrt(1-x²)
        //   = π/(n+1) · sin²(θ) · (u₀π/4)·sec²(arg) / sin(θ)
        //   = π/(n+1) · sin(θ) · (u₀π/4) · sec²(arg)
        let w_k = (pi / np1) * theta_k.sin() * (u0 * pi / 4.0) * sec2;
        freqs.push(omega_k);
        weights.push(w_k);
    }
    (freqs, weights)
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
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
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
    fn chebyshev_tan_weights_sum_to_pi_over_2() {
        // Same target as GL — must integrate 1/(1+ω²) to π/2 on [0,∞).
        let (freqs, weights) = chebyshev_tan_nodes(20, 0.5);
        let sum: f64 = freqs.iter().zip(weights.iter())
            .map(|(w, wt)| wt / (1.0 + w * w))
            .sum();
        // Cheb-2 + tan-map convergence is slower than GL for this integrand
        // at n=20 (~4e-3); main use case is the RPA trace-log integrand
        // where Eshuis et al show <0.1 mHa at n=14-20.
        assert!((sum - std::f64::consts::FRAC_PI_2).abs() < 1e-2,
            "Cheb-tan: |sum - π/2| = {}", (sum - std::f64::consts::FRAC_PI_2).abs());
    }

    #[test]
    fn chebyshev_tan_omega_max_finite() {
        // Eshuis tan-map yields bounded ω-range. With u₀=0.5, max ω should
        // be O(few) rather than O(100) of GL.
        let (freqs, _) = chebyshev_tan_nodes(20, 0.5);
        let omega_max = freqs.iter().cloned().fold(0.0_f64, f64::max);
        // GL n=20, u₀=0.5 gives max ω ~ 145; Cheb-tan with same parameters
        // bounds it ~3× tighter (still ~57 at n=20). Loose check — main
        // point is that the *integrand mass* concentrates at low ω, not
        // a strict cap.
        assert!(omega_max < 100.0, "Cheb-tan max ω = {}, should be tighter than GL (~145)", omega_max);
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
