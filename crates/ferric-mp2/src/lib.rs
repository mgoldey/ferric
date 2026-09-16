//! Second-order Moller-Plesset perturbation theory (MP2) implementations.
//!
//! - [`crate::rimp2`] -- Resolution-of-identity MP2 (density-fitted) using 3-center integrals
//! - [`crate::oo_rimp2`] -- Orbital-optimized RI-MP2 with analytic gradients and Cayley rotations
//! - [`crate::canonical`] -- Canonical MP2 via full 4-center AO-to-MO transformation (for validation)
//! - [`crate::gradient`] -- RI-MP2 nuclear gradient via finite differences
//! - [`crate::mo_transform`] -- AO-to-MO integral transformation utilities

// op_ref: ndarray expressions like `4.0 * &jz - &kz - &kz.t()` reference `kz`
// because it is reused (`.t()`) on the same line; dropping `&` would move it.
// needless_range_loop: primitive/MO-index loops read clearer with explicit indices.
#![allow(clippy::op_ref)]
#![allow(clippy::needless_range_loop)]

/// Boys localization of occupied orbitals.
pub mod boys;
/// DLPNO-MP2 (domain-based local pair natural orbital MP2).
pub mod dlpno_mp2;
/// Direct RPA amplitude doubles (ring-CCD) from RI intermediates.
pub mod drpa_amplitude;
/// LCCD and CEPA(0) from RI intermediates.
pub mod lccd_cepa0;
/// Amplitude-threshold local MP2 (WSHG23 single-threshold scheme).
pub mod lmp2_amplitude;
/// Ring-CCD family: rCCD, drCCD amplitude solvers.
pub mod rccd_family;

/// Attenuated MP2 + VV10 nonlocal correlation combination.
pub mod att_vv10;
/// Attenuated MP2: erfc(ωr)/r screened correlation energy.
pub mod attenuated;
mod attenuated_timing;
/// Canonical MP2 via full 4-center AO-to-MO transformation (validation only).
pub mod canonical;
/// Coupled-perturbed KS (CPKS) polarizability via response equations.
pub mod cpks_polar;
/// Generic MP2-based double-hybrid DFT (B2PLYP, DSD-PBEP86).
pub mod double_hybrid;
/// F12 explicitly-correlated MP2 with CABS singles.
pub mod f12;
/// Finite-field static polarizability from MP2 energy derivatives.
pub mod ff_polar;
/// RI-MP2 nuclear gradient via Z-vector response.
pub mod gradient;
/// Laplace-transform SOS-MP2 (MO, AO, and AO-sparse formulations).
pub mod laplace;
pub mod lmp2_direct;
/// Pair natural orbital (PNO) construction and truncation.
pub mod local_pno;
/// AO-to-MO integral transformation utilities.
pub mod mo_transform;
/// Third-order Moller-Plesset perturbation theory (MP3).
pub mod mp3;
/// MPI-distributed RI-MP2 (aux-band striping across ranks).
pub mod mpi_rimp2;
/// Orbital-optimized RI-MP2 with analytic gradients and Cayley rotations.
pub mod oo_rimp2;
/// OO-RI-MP2 analytical nuclear gradient.
pub mod oo_rimp2_gradient;
/// Geometry optimization using MP2 gradients.
pub mod optimize;
/// Cayley/exponential orbital rotation helpers for OO-MP2.
pub mod orbital_rotation;
/// Projected atomic orbital (PAO) domain construction for local correlation.
pub mod pair_domains;
/// Energy-based pair screening for local correlation methods.
pub mod pair_energy_screen;
/// Ragged per-pair domain-block storage for local correlation.
pub mod ragged;
/// Resolution-of-identity MP2 (density-fitted) using 3-center integrals.
pub mod rimp2;
/// Spin-component-scaled MP2 (SCS-MP2 and 2-terfc variants).
pub mod scs;
/// Spin-orbital integral/amplitude helpers for closed-shell references.
pub mod spinorbital;
/// Spin-orbital integral/amplitude helpers for unrestricted references.
pub mod spinorbital_u;
/// Unrestricted orbital-optimized RI-MP2.
pub mod u_oo_rimp2;
/// Unrestricted RI-MP2 (α/β spin channels).
pub mod u_rimp2;
/// Z-vector response equations for MP2 relaxed density and gradients.
pub mod zvector;
