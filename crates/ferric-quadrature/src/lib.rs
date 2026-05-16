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

pub mod minimax;

pub use minimax::{select_minimax_points, LaplaceQuadrature};
