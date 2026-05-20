//! Electron integral evaluation via libint2 FFI.
//!
//! This crate wraps libint2 through a C++ shim (`shim/shim.cc`) to compute:
//!
//! - **One-electron integrals**: overlap, kinetic, nuclear attraction
//! - **Two-electron repulsion integrals** (ERIs): 4-center shell quartets
//! - **Three-center integrals** (P|mu nu) for density fitting / RI methods
//! - **Two-center integrals** (P|Q) for Coulomb metric
//! - **Derivative integrals**: first derivatives of 1e and 2e integrals
//! - **Schwarz screening**: upper bounds on shell-pair integrals
//!
//! The [`basis_bridge::PreparedBasis`] struct owns the libint2 basis handle and
//! manages the Rust-to-C++ lifetime boundary.

pub mod ffi;
pub mod operator;
pub mod basis_bridge;
pub mod engine;
pub mod oneelectron;
pub mod schwarz;
pub mod qqr3;
pub mod threeindex;
