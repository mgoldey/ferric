//! Core types for the ferric quantum chemistry engine.
//!
//! This crate provides foundational data structures shared across all ferric crates:
//!
//! - [`mol::Molecule`] and [`mol::Atom`] -- molecular geometry (XYZ parser, nuclear repulsion)
//! - [`basis::BasisSet`] and [`basis::Shell`] -- Gaussian basis sets (BSE-JSON, G94, bundled)
//! - [`elements`] -- element symbol/atomic-number lookup tables
//! - [`orbitals::OrbitalSpace`] -- active occupied/virtual partition for post-HF methods
//! - [`conformers::ConformerEnsemble`] -- multi-geometry ensembles and Boltzmann-weighted
//!   property averaging (conformer *generation* is out of scope; RDKit does that)
//! - [`FerricError`] -- unified error type for the workspace

pub mod error;
pub mod elements;
pub mod mol;
pub mod conformers;
pub mod basis;
pub mod ecp;
pub mod external_potential;
pub mod orbitals;
pub mod parallel;
pub mod memory;
pub mod config;
pub mod blas_threads;
pub mod linalg;

pub use error::FerricError;
pub use orbitals::OrbitalSpace;

use std::sync::atomic::AtomicBool;
pub static INTERRUPT: AtomicBool = AtomicBool::new(false);
