//! Second-order Moller-Plesset perturbation theory (MP2) implementations.
//!
//! - [`rimp2`] -- Resolution-of-identity MP2 (density-fitted) using 3-center integrals
//! - [`oo_rimp2`] -- Orbital-optimized RI-MP2 with analytic gradients and Cayley rotations
//! - [`canonical`] -- Canonical MP2 via full 4-center AO-to-MO transformation (for validation)
//! - [`gradient`] -- RI-MP2 nuclear gradient via finite differences
//! - [`mo_transform`] -- AO-to-MO integral transformation utilities

// op_ref: ndarray expressions like `4.0 * &jz - &kz - &kz.t()` reference `kz`
// because it is reused (`.t()`) on the same line; dropping `&` would move it.
// needless_range_loop: primitive/MO-index loops read clearer with explicit indices.
#![allow(clippy::op_ref)]
#![allow(clippy::needless_range_loop)]

pub mod boys;
pub mod mo_transform;
pub mod rimp2;
pub mod mp3;
pub mod spinorbital;
pub mod u_rimp2;
pub mod u_oo_rimp2;
pub mod canonical;
pub mod gradient;
pub mod oo_rimp2;
pub mod attenuated;
mod attenuated_timing;
pub mod scs;
pub mod zvector;
pub mod laplace;
pub mod oo_rimp2_gradient;
pub mod mpi_rimp2;
