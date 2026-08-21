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

/// Unified error type ([`FerricError`]) for the workspace.
pub mod error;
/// Element symbol/atomic-number/mass lookup tables (Z = 1–118).
pub mod elements;
/// Molecular geometry: [`Atom`](mol::Atom), [`Molecule`](mol::Molecule), XYZ parser, nuclear repulsion.
pub mod mol;
/// Multi-geometry conformer ensembles and Boltzmann-weighted property averaging.
pub mod conformers;
/// Gaussian basis sets: [`BasisSet`](basis::BasisSet), [`Shell`](basis::Shell), BSE-JSON/G94 parsers, 21 bundled sets.
pub mod basis;
/// Effective core potentials (ECPs / pseudopotentials).
pub mod ecp;
/// Classical external potentials: fixed point charges and uniform electric fields.
pub mod external_potential;
/// Occupied/virtual orbital partitions for post-HF methods.
pub mod orbitals;
/// Parallelism context: thread pools, optional MPI world handle.
pub mod parallel;
/// Memory budget discovery and enforcement.
pub mod memory;
/// Typed configuration descriptors (`ConfigVar`).
pub mod config;
/// BLAS thread-count guard for safe OpenBLAS usage under rayon.
pub mod blas_threads;
mod basis_util;
/// Dense linear algebra helpers (Cholesky, eigenvalue, matrix utilities).
pub mod linalg;

pub use error::FerricError;
pub use orbitals::OrbitalSpace;

use std::sync::atomic::AtomicBool;
/// Global interrupt flag — set by SIGINT handler, checked by SCF/optimization loops.
pub static INTERRUPT: AtomicBool = AtomicBool::new(false);
