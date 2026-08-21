//! Numerical quadrature utilities shared across ferric crates.
//!
//! Currently provides:
//! - [`minimax`] — minimax-Laplace quadrature for `1/x` on `[1, R]`, used by both
//!   Laplace-MP2 (in `ferric-mp2`) and Laplace-separable χ₀ for PDEP-RPA
//!   (in `ferric-rpa`).
//!
//! The quadrature points/weights are extracted from the literature tables of
//! Takatsuka, Ten-no, Hackbusch (JCP 129, 044112, 2008) via the Helmich-Paris
//! `laplace-minimax` library.

// Reference quadrature tables are transcribed at full source precision on
// purpose; trimming to f64's last digit is lossy churn. Index loops over
// node/weight arrays read clearer with explicit indices.
#![allow(clippy::excessive_precision)]
#![allow(clippy::needless_range_loop)]

/// Minimax-Laplace quadrature for `1/x` on `[1, R]` (Takatsuka/Ten-no/Hackbusch tables).
pub mod minimax;

/// Lebedev angular quadrature on the unit sphere.
pub mod lebedev;

pub use minimax::{select_minimax_points, LaplaceQuadrature};
