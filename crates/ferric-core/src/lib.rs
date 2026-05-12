//! Core types for the ferric quantum chemistry engine.
//!
//! This crate provides foundational data structures shared across all ferric crates:
//!
//! - [`mol::Molecule`] and [`mol::Atom`] -- molecular geometry (XYZ parser, nuclear repulsion)
//! - [`basis::BasisSet`] and [`basis::Shell`] -- Gaussian basis sets (BSE-JSON, G94, bundled)
//! - [`elements`] -- element symbol/atomic-number lookup tables
//! - [`FerricError`] -- unified error type for the workspace

pub mod error;
pub mod elements;
pub mod mol;
pub mod basis;

pub use error::FerricError;
