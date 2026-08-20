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

/// Raw C-ABI bindings to the libint2 shim (`shim/shim.cc`).
pub mod ffi;
/// Raw C-ABI bindings to the ECP shim (`shim/ecp_shim.cc`).
pub mod ecp_ffi;
/// Effective core potential integral evaluation.
pub mod ecp;
/// Integral operator kinds: Coulomb, erf/erfc-attenuated, Yukawa, Slater geminal.
pub mod operator;
/// [`PreparedBasis`](basis_bridge::PreparedBasis): Rust↔libint2 basis handle and shell metadata.
pub mod basis_bridge;
/// Low-level integral engine: 2e/3c/2c shell-block compute calls.
pub mod engine;
/// Thread-safe pool of integral engines for rayon-parallel Fock builds.
pub mod engine_pool;
/// One-electron integrals: overlap, kinetic, nuclear attraction, multipole moments.
pub mod oneelectron;
/// Complementary auxiliary basis sets (CABS) for F12 methods.
pub mod cabs;
/// Schwarz upper bounds for integral screening.
pub mod schwarz;
/// QQR distance-dependent integral screening (Ochsenfeld-style).
pub mod qqr3;
/// BLAS thread-count guard (re-exported from ferric-core for convenience).
pub mod blas_threads;
/// Three-center density-fitting integrals (P|mu nu) with optional batching.
pub mod threeindex;
/// [`ThreeIndexSource`](three_index_source::ThreeIndexSource) trait: abstract 3-index tensor provider.
pub mod three_index_source;
