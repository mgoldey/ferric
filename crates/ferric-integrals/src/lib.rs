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

/// GTO basis-function evaluation on real-space grids (Cartesian, Becke-Lebedev, etc.).
pub mod ao_grid;
/// [`PreparedBasis`](basis_bridge::PreparedBasis): Rust↔libint2 basis handle and shell metadata.
pub mod basis_bridge;
/// BLAS thread-count guard (re-exported from ferric-core for convenience).
pub mod blas_threads;
/// Complementary auxiliary basis sets (CABS) for F12 methods.
pub mod cabs;
/// Per-grid-point three-center-one-electron AO blocks `A^g_{mu,nu}` for
/// seminumerical exchange (COSX / sn-LinK).
pub mod cosx_a;
/// Shell-pair screening bound shared by `cosx_a` and `md3c1e` (Hölder bound on
/// the primitive expansion; never underestimates, decays as `K_AB / R`).
pub mod cosx_screen;
/// CSAM (combined Schwarz approximation) integral screening: the
/// NON-RIGOROUS multiplicative estimate of Eqs. (9)/(11)/(12) of the same
/// paper, ported from Psi4's `shell_significant_csam()`. Tighter than plain
/// Schwarz, but it can UNDERESTIMATE the true integral — an opt-in
/// accuracy-vs-threshold tradeoff, not a free speedup. Contrast [`csb`] below,
/// which is the rigorous member of the family.
pub mod csam;
/// CSB (combined Schwarz bound) integral screening: the RIGOROUS
/// `min{Q_uv Q_ls, M_ul M_vs, M_us M_vl}` bound of Thompson & Ochsenfeld,
/// JCP 147, 144101 (2017), Eq. (8). Never looser than plain Schwarz.
pub mod csb;
/// Effective core potential integral evaluation.
pub mod ecp;
/// Raw C-ABI bindings to the ECP shim (`shim/ecp_shim.cc`).
pub mod ecp_ffi;
/// Low-level integral engine: 2e/3c/2c shell-block compute calls.
pub mod engine;
/// Thread-safe pool of integral engines for rayon-parallel Fock builds.
pub mod engine_pool;
/// Raw C-ABI bindings to the libint2 shim (`shim/shim.cc`).
pub mod ffi;
/// From-scratch McMurchie-Davidson 3-center-1-electron kernel producing the
/// same `A^g` blocks as `cosx_a` for a BATCH of grid points, grid axis innermost.
pub mod md3c1e;
/// One-electron integrals: overlap, kinetic, nuclear attraction, multipole moments.
pub mod oneelectron;
/// Integral operator kinds: Coulomb, erf/erfc-attenuated, Yukawa, Slater geminal.
pub mod operator;
/// QQR distance-dependent integral screening (Ochsenfeld-style).
pub mod qqr3;
/// Schwarz upper bounds for integral screening.
pub mod schwarz;
/// [`SiteBasis`](site_basis::SiteBasis): one Gaussian per MM site as a fake auxiliary
/// `PreparedBasis`, used for Gaussian-smeared QM/MM charge potentials and their gradients.
pub mod site_basis;
/// [`ThreeIndexSource`](three_index_source::ThreeIndexSource) trait: abstract 3-index tensor provider.
pub mod three_index_source;
/// Three-center density-fitting integrals (P|mu nu) with optional batching.
pub mod threeindex;
