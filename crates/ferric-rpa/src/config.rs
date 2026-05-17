//! Configuration types for PDEP-RPA.

/// Backend for the χ₀ kernel that powers the dielectric matrix.
///
/// `Dense` is the original O(naux² × nocc × nvir) MO-basis path used in
/// `dielectric_matrix_into`. `Laplace { n_quad }` factorizes the energy-gap
/// denominator into a sum of exponentials via minimax-Laplace quadrature; in
/// the MO-basis form it is correctness-equivalent to `Dense` (and the same
/// arithmetic complexity), but it admits an AO-basis cubic-scaling
/// reformulation as a follow-up.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Chi0Backend {
    Dense,
    Laplace { n_quad: usize },
}

impl Default for Chi0Backend {
    fn default() -> Self {
        Chi0Backend::Dense
    }
}

/// Sparsity strategy for the χ₀ build / dielectric matvec.
///
/// `Dense` (default) uses the canonical `(naux × nocc·nvir)` `b_ov` tensor.
///
/// `BoysScreened { thresh }` runs Foster-Boys on the active occupied block,
/// builds per-orbital `B^P_{i_loc, a}` tiles, and drops aux rows P whose
/// per-row L∞ norm `max_a |B^P_{i_loc, a}|` is below `thresh`. The
/// dielectric matvec then iterates over orbitals, gathering and scattering
/// through the per-orbital aux index lists.
///
/// Closed-shell only for now. Open-shell support is C8.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Chi0Sparsity {
    Dense,
    BoysScreened { thresh: f64 },
}

impl Default for Chi0Sparsity {
    fn default() -> Self {
        Chi0Sparsity::Dense
    }
}

/// Choice of subspace eigensolver for the PDEP dielectric matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Eigensolver {
    Davidson,
    Lanczos,
}

impl Default for Eigensolver {
    fn default() -> Self {
        Eigensolver::Davidson
    }
}

/// Top-level PDEP-RPA configuration.
#[derive(Debug, Clone)]
pub struct PdepRpaConfig {
    /// Number of frozen core orbitals.
    pub frozen_core: usize,
    /// Truncate eigenpotentials whose |λ_α(0) − 1| ≤ trunc_thresh.
    pub trunc_thresh: f64,
    /// Maximum Davidson subspace size before restart.
    pub davidson_max_vecs: usize,
    /// Davidson eigenvalue convergence threshold.
    pub davidson_conv_thresh: f64,
    pub quadrature: QuadratureConfig,
    pub sternheimer: SternheimerConfig,
    /// If true, also compute the full-basis RI-dRPA diagnostic energy (expensive).
    pub run_diagnostics: bool,
    /// Eigensolver backend for the static dielectric eigenproblem.
    pub eigensolver: Eigensolver,
    /// χ₀ kernel backend. Default `Dense` preserves legacy behavior.
    pub chi0_backend: Chi0Backend,
    /// χ₀ sparsity strategy. Default `Dense` preserves legacy behavior.
    pub chi0_sparsity: Chi0Sparsity,
}

impl Default for PdepRpaConfig {
    fn default() -> Self {
        Self {
            frozen_core: 0,
            trunc_thresh: 1e-4,
            davidson_max_vecs: 0,
            davidson_conv_thresh: 1e-6,
            quadrature: QuadratureConfig::default(),
            sternheimer: SternheimerConfig::default(),
            run_diagnostics: false,
            eigensolver: Eigensolver::Davidson,
            chi0_backend: Chi0Backend::Dense,
            chi0_sparsity: Chi0Sparsity::Dense,
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
