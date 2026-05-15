//! Configuration types for PDEP-RPA.

/// Top-level PDEP-RPA configuration.
#[derive(Debug, Clone)]
pub struct PdepRpaConfig {
    /// Number of frozen core orbitals.
    pub frozen_core: usize,
    /// Eigenvalue truncation threshold: discard λ_α(0) < trunc_thresh.
    pub trunc_thresh: f64,
    /// Maximum Davidson subspace size before restart.
    pub davidson_max_vecs: usize,
    /// Davidson eigenvalue convergence threshold.
    pub davidson_conv_thresh: f64,
    pub quadrature: QuadratureConfig,
    pub sternheimer: SternheimerConfig,
}

impl Default for PdepRpaConfig {
    fn default() -> Self {
        Self {
            frozen_core: 0,
            trunc_thresh: 1e-4,
            davidson_max_vecs: 0, // 0 = 3*N_aux (set at runtime)
            davidson_conv_thresh: 1e-6,
            quadrature: QuadratureConfig::default(),
            sternheimer: SternheimerConfig::default(),
        }
    }
}

/// Imaginary-frequency quadrature configuration.
#[derive(Debug, Clone)]
pub struct QuadratureConfig {
    pub scheme: QuadratureScheme,
    /// Number of quadrature points (default 20).
    pub n_points: usize,
    /// Gauss-Legendre domain scale parameter u₀ in Eₕ (default 0.5).
    pub u0: f64,
}

impl Default for QuadratureConfig {
    fn default() -> Self {
        Self {
            scheme: QuadratureScheme::MiniMax,
            n_points: 20,
            u0: 0.5,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum QuadratureScheme {
    /// GL nodes with literature-optimized u₀ scale (Furche, JCP 2005).
    MiniMax,
    /// Gauss-Legendre nodes mapped to [0,∞) via ω = u₀(1+x)/(1−x).
    GaussLegendre,
}

/// Sternheimer linear solver configuration.
#[derive(Debug, Clone)]
pub struct SternheimerConfig {
    pub max_iter: usize,
    pub conv_thresh: f64,
}

impl Default for SternheimerConfig {
    fn default() -> Self {
        Self { max_iter: 50, conv_thresh: 1e-8 }
    }
}
